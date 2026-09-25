#!/usr/bin/env python3
"""Calcula as metricas do Exp 2 (localizacao por caixa) a partir das rodadas salvas.

Segue a secao "Evaluation Protocol for Experiment 2" do artigo: melhor IoU por caso contra a caixa de
referencia, sem classe (principal) e exigindo que o rotulo previsto projete na classe da referencia,
mais o numero de caixas por caso. Mesma selecao de rodadas e mesmo bootstrap do Exp 1.

Uso:
    tools/py analysis/exp2_metrics.py
Saidas (analysis/exp2/): cases.csv, summary.csv
"""

import csv
import json
import pathlib
import sys

import numpy as np

EVAL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "analysis"))
from bbox_eval import compute_iou, extract_predicted_bboxes  # noqa: E402
from clinical_ontology import project_hypothesis  # noqa: E402
from exp1_metrics import MODELS, VOCABS, bootstrap_indices, ci, collect_agent, collect_zeroshot, resampled_mean  # noqa: E402

BENCH = "exp2_bbox_localization"
MANIFEST = EVAL / "inputs" / "benchmarks" / BENCH / "manifest.json"
OUT = EVAL / "analysis" / "exp2"
THRESHOLDS = [0.1, 0.3, 0.5]
# o degrau modelo + classificador e so do Exp 1 (os classificadores nao localizam)
MODES = [("zeroshot", None), ("single", "single-agent"), ("council", "multi-agent-single-model")]


def valid_localization(data) -> bool:
    return isinstance(data, dict) and isinstance(data.get("localized_lesions"), list)


def best_iou(pred: list[dict], ref_box: list[float], ref_classes: set[str] | None) -> float:
    best = 0.0
    for p in pred:
        if ref_classes is not None and not ref_classes.intersection(project_hypothesis(p["label"])["classes"]):
            continue
        best = max(best, compute_iou(p["bbox"], ref_box))
    return best


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    images = [it["image"] for it in manifest]
    ref = {it["image"]: it["bboxes"][0] for it in manifest}
    n_cases = len(images)
    idx = bootstrap_indices(n_cases)
    OUT.mkdir(parents=True, exist_ok=True)

    case_rows, summary_rows = [], []
    for (vocab, root), model, (mode, mode_dir) in ((v, m, md) for v in VOCABS for m in MODELS for md in MODES):
        suffix = "" if vocab == "open" else ":closed"
        cases = (collect_zeroshot(model, images, BENCH, valid_localization, root=root) if mode == "zeroshot"
                 else collect_agent(mode_dir, model, images, BENCH, valid_localization, root=root))
        vec = {k: [] for k in ["iou", "iou_cls", "n_boxes"] + [f"hit{t}" for t in THRESHOLDS]
               + [f"hit{t}_cls" for t in THRESHOLDS]}
        for img in images:
            c = cases[img]
            pred = extract_predicted_bboxes(c["data"]) if c["valid"] else []
            ref_classes = set(project_hypothesis(ref[img]["label"])["classes"])
            u = best_iou(pred, ref[img]["bbox"], None)
            u_cls = best_iou(pred, ref[img]["bbox"], ref_classes)
            vec["iou"].append(u)
            vec["iou_cls"].append(u_cls)
            vec["n_boxes"].append(len(pred))
            for t in THRESHOLDS:
                vec[f"hit{t}"].append(float(u >= t))
                vec[f"hit{t}_cls"].append(float(u_cls >= t))
            case_rows.append({"condition": f"{model}:{mode}{suffix}", "model": model, "mode": mode, "vocab": vocab, "image": img,
                              "finished": int(c["finished"]), "valid": int(c["valid"]), "source": c["source"],
                              "ref_label": ref[img]["label"], "n_boxes": len(pred),
                              "iou": round(u, 4), "iou_cls": round(u_cls, 4)})
        finished = sum(cases[i]["finished"] for i in images)
        valid = sum(cases[i]["valid"] for i in images)
        row = {"condition": f"{model}:{mode}{suffix}", "model": model, "mode": mode, "vocab": vocab,
               "status": "complete" if finished == n_cases else "in_progress",
               "finished": finished, "valid": valid, "n_cases": n_cases}
        for k, v in vec.items():
            arr = np.array(v, dtype=float)
            lo, hi = ci(resampled_mean(arr, idx))
            row.update({k: float(arr.mean()), f"{k}_lo": lo, f"{k}_hi": hi})
        summary_rows.append(row)

    for name, rows in [("cases.csv", case_rows), ("summary.csv", summary_rows)]:
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    for r in summary_rows:
        print(f"{r['condition']:<26} {r['status']:<11} valid {r['valid']:>2}/{r['n_cases']}  "
              f"IoU>=0.1 {r['hit0.1']:.3f} ({r['hit0.1_cls']:.3f} cls)  IoU>=0.5 {r['hit0.5']:.3f}  "
              f"mIoU {r['iou']:.3f}  caixas {r['n_boxes']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
