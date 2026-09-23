#!/usr/bin/env python3
"""Calcula todas as metricas do Exp 1 (diagnostico diferencial aberto) a partir das rodadas salvas.

Segue o protocolo da secao "Evaluation Protocol for Experiment 1" do artigo. As saidas ficam em
analysis/exp1/ e sao lidas por latex-paper/figures/scripts/exp1_tables_figures.py.

Uso:
    tools/py analysis/exp1_metrics.py

Saidas (analysis/exp1/):
    cases.csv                   uma linha por condicao e caso
    hypotheses.csv              uma linha por hipotese pontuada
    summary.csv                 agregados por condicao com IC bootstrap 95%
    per_class_recall.csv        recall por classe e condicao
    paired_differences.csv      diferencas pareadas entre condicoes do mesmo modelo
    oov_audit.csv               textos OOV unicos, embaralhados e sem condicao, para adjudicacao cega
    oov_audit_by_condition.csv  contagem das categorias da auditoria por condicao
    meta.json                   versao do lexico e parametros
"""

import csv
import hashlib
import json
import os
import pathlib
import sys

import numpy as np

EVAL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "analysis"))
from align_diagnosis import CATEGORIES, K, score_case  # noqa: E402
from clinical_ontology import LEXICON_VERSION, ONTOLOGY, normalize_text  # noqa: E402

MANIFEST = EVAL / "inputs" / "benchmarks" / "exp1_image_level" / "manifest.json"
# versao dos prompts cujas rodadas sao pontuadas (results/<versao>/); a v2 revelava regiao e classes
VERSION = os.environ.get("XRAY_RESULTS_VERSION", "v3")
RESULTS = EVAL / "results" / VERSION
# o baseline supervisionado nao depende de prompt: as predicoes ficam em um caminho fixo
SUPERVISED = EVAL / "results" / "v2" / "supervised"
OUT = EVAL / "analysis" / "exp1"
MODELS = ["gemma4-26b", "qwen3.6-27b", "qwen3.8-27b"]
MODES = [("zeroshot", None), ("single", "single-agent"), ("council", "multi-agent-single-model")]
BASELINE = "densenet121-res224-all"
B = 10_000
SEED = 42
AUDIT_CATEGORIES = ["lexicon_gap", "out_of_label_space", "non_diagnostic"]
METRICS = ["hit1", "hit3", "mrr3", "recall3", "precision3", "f1", "hit1_f"]
PAIRS = [("single", "zeroshot"), ("council", "single"), ("council", "zeroshot")]


def load_json(path: pathlib.Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def valid_differential(data) -> bool:
    return isinstance(data, dict) and isinstance(data.get("differential_diagnosis"), list)


def collect_agent(mode_dir: str, model: str, images: list[str], bench: str = "exp1_image_level",
                  valid=valid_differential, rep: str = "r1") -> dict:
    """Por imagem: a rodada valida mais recente da repeticao pedida; senao, marca tentativa terminada sem saida valida."""
    base = RESULTS / bench / mode_dir / model
    runs: dict[str, list[pathlib.Path]] = {}
    if base.exists():
        for rd in sorted(p for p in base.iterdir() if p.is_dir()):
            stem, _, run_rep = rd.name[16:].rpartition("-")
            if run_rep == rep:
                runs.setdefault(stem, []).append(rd)
    out = {}
    for img in images:
        stem = pathlib.Path(img).stem
        chosen, finished = None, False
        for rd in reversed(runs.get(stem, [])):
            data = load_json(rd / f"{stem}_analysis" / f"{img}.json")
            if valid(data):
                chosen = (rd, data)
                break
            finished = finished or (rd / "harness_result.json").exists()
        if chosen:
            out[img] = {"finished": True, "valid": True, "source": chosen[0].name, "data": chosen[1],
                        "hypotheses": chosen[1].get("differential_diagnosis", [])}
        else:
            out[img] = {"finished": finished, "valid": False, "source": "", "data": None, "hypotheses": []}
    return out


def collect_zeroshot(model: str, images: list[str], bench: str = "exp1_image_level", valid=valid_differential) -> dict:
    base = RESULTS / "zeroshot" / bench / model
    # o zero-shot so grava arquivo quando a resposta e analisada; o marcador indica que o manifesto foi percorrido
    passed = (base / "_complete.json").exists()
    out = {}
    for img in images:
        data = load_json(base / f"{pathlib.Path(img).stem}.json")
        ok = valid(data)
        out[img] = {"finished": ok or passed, "valid": ok, "source": "zeroshot" if ok else "", "data": data if ok else None,
                    "hypotheses": data.get("differential_diagnosis", []) if ok else []}
    return out


def collect_baseline(images: list[str]) -> dict:
    data = load_json(SUPERVISED / "exp1_image_level" / BASELINE / "predictions.json") or {}
    preds = data.get("predictions", {})
    out = {}
    for img in images:
        p = preds.get(img)
        # cada posicao e uma unica classe, escrita pelo seu sinonimo canonico para passar pela mesma projecao
        hyps = [ONTOLOGY[c]["synonyms"][0] for c in p["ranked_classes"][:K]] if p else []
        out[img] = {"finished": bool(p), "valid": bool(p), "source": BASELINE if p else "", "hypotheses": hyps}
    return out


def conditions(images: list[str]) -> list[dict]:
    conds = [{"id": "baseline", "model": BASELINE, "mode": "supervised", "cases": collect_baseline(images)}]
    for model in MODELS:
        for mode, mode_dir in MODES:
            cases = collect_zeroshot(model, images) if mode == "zeroshot" else collect_agent(mode_dir, model, images)
            conds.append({"id": f"{model}:{mode}", "model": model, "mode": mode, "cases": cases})
    return conds


def bootstrap_indices(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))


def ci(samples: np.ndarray) -> tuple[float, float]:
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def resampled_mean(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    res = values[idx]
    with np.errstate(invalid="ignore"):
        counts = (~np.isnan(res)).sum(axis=1)
        sums = np.nansum(res, axis=1)
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def main() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    images = [it["image"] for it in manifest]
    gt = {it["image"]: it["finding_labels"] for it in manifest}
    n_cases = len(images)
    idx = bootstrap_indices(n_cases)
    OUT.mkdir(parents=True, exist_ok=True)

    case_rows, hyp_rows, summary_rows, class_rows = [], [], [], []
    vectors: dict[str, dict[str, np.ndarray]] = {}
    oov_by_condition: dict[str, dict[str, int]] = {}

    for cond in conditions(images):
        scored = []
        for img in images:
            c = cond["cases"][img]
            s = score_case(c["hypotheses"], gt[img])
            scored.append((img, c, s))
            case_rows.append({
                "condition": cond["id"], "model": cond["model"], "mode": cond["mode"], "image": img,
                "finished": int(c["finished"]), "valid": int(c["valid"]), "source": c["source"],
                "gt": "|".join(gt[img]), "asserted": "|".join(s["asserted"]),
                "n": s["n"], "n_oov": s["n_oov"], "n_correct": s["n_correct"], "n_incorrect": s["n_incorrect"],
                "rank": s["rank"] or "", "hit1": int(s["hit1"]), "hit3": int(s["hit3"]), "mrr3": round(s["mrr3"], 4),
                "recall3": round(s["recall3"], 4),
                "precision3": "" if s["precision3"] is None else round(s["precision3"], 4),
                "f1": round(s["f1"], 4), "oov_rate": "" if s["oov_rate"] is None else round(s["oov_rate"], 4),
                "category": s["category"], "hit1_f": int(s["hit1_f"]),
            })
            for pos, h in enumerate(s["window"], start=1):
                hyp_rows.append({"condition": cond["id"], "image": img, "position": pos, "text": h["text"],
                                 "classes": "|".join(h["classes"]), "negated": "|".join(h["negated"]),
                                 "outcome": h["outcome"]})
                if h["outcome"] == "oov":
                    key = normalize_text(h["text"])
                    oov_by_condition.setdefault(cond["id"], {})
                    oov_by_condition[cond["id"]][key] = oov_by_condition[cond["id"]].get(key, 0) + 1

        vec = {m: np.array([np.nan if s[m] is None else float(s[m]) for _, _, s in scored]) for m in METRICS}
        vec["n"] = np.array([s["n"] for _, _, s in scored], dtype=float)
        vec["n_oov"] = np.array([s["n_oov"] for _, _, s in scored], dtype=float)
        vec["oov_rate"] = np.array([np.nan if s["oov_rate"] is None else s["oov_rate"] for _, _, s in scored])
        for cat in CATEGORIES:
            vec[cat] = np.array([float(s["category"] == cat) for _, _, s in scored])
        vec["any_oov"] = vec["partial_oov_hit"] + vec["partial_oov_miss"] + vec["total_oov"]
        vectors[cond["id"]] = vec

        finished = sum(c["finished"] for _, c, _ in scored)
        valid = sum(c["valid"] for _, c, _ in scored)
        row = {"condition": cond["id"], "model": cond["model"], "mode": cond["mode"],
               "status": "complete" if finished == n_cases else "in_progress",
               "finished": finished, "valid": valid, "n_cases": n_cases, "valid_share": valid / n_cases}
        for m in METRICS + ["oov_rate", "any_oov", "total_oov", "empty"]:
            point = float(np.nanmean(vec[m])) if not np.all(np.isnan(vec[m])) else float("nan")
            lo, hi = ci(resampled_mean(vec[m], idx))
            row.update({m: point, f"{m}_lo": lo, f"{m}_hi": hi})
        row["precision3_n"] = int((~np.isnan(vec["precision3"])).sum())
        tot_n, tot_oov = vec["n"].sum(), vec["n_oov"].sum()
        n_correct = sum(s["n_correct"] for _, _, s in scored)
        n_incorrect = sum(s["n_incorrect"] for _, _, s in scored)
        row.update({"hypotheses": int(tot_n), "share_correct": n_correct / tot_n if tot_n else float("nan"),
                    "share_incorrect": n_incorrect / tot_n if tot_n else float("nan"),
                    "share_oov": tot_oov / tot_n if tot_n else float("nan")})
        with np.errstate(invalid="ignore", divide="ignore"):
            pooled = vec["n_oov"][idx].sum(axis=1) / vec["n"][idx].sum(axis=1)
        row["share_oov_lo"], row["share_oov_hi"] = ci(pooled)
        for cat in CATEGORIES:
            row[f"cat_{cat}"] = float(vec[cat].mean())
        summary_rows.append(row)

        for cls in ONTOLOGY:
            support = [s for img, _, s in scored if cls in gt[img]]
            if support:
                hits = sum(cls in s["asserted"] for s in support)
                class_rows.append({"condition": cond["id"], "class": cls, "support": len(support),
                                   "hits": hits, "recall": hits / len(support)})

    paired_rows = []
    for model in MODELS:
        status = {r["condition"]: r["status"] for r in summary_rows}
        for a, b in PAIRS:
            ca, cb = f"{model}:{a}", f"{model}:{b}"
            both = status.get(ca) == "complete" and status.get(cb) == "complete"
            for m in ["hit1", "hit3", "mrr3", "recall3", "f1"]:
                diff = vectors[ca][m] - vectors[cb][m]
                lo, hi = ci(resampled_mean(diff, idx))
                paired_rows.append({"model": model, "comparison": f"{a}-{b}", "metric": m,
                                    "both_complete": int(both), "diff": float(np.nanmean(diff)), "lo": lo, "hi": hi})

    audit_path = OUT / "oov_audit.csv"
    previous = {}
    if audit_path.exists():
        with open(audit_path, newline="", encoding="utf-8") as f:
            previous = {r["text"]: r for r in csv.DictReader(f)}
    totals: dict[str, int] = {}
    for counts in oov_by_condition.values():
        for text, cnt in counts.items():
            totals[text] = totals.get(text, 0) + cnt
    # ordem pseudoaleatoria estavel e sem a condicao de origem: a adjudicacao fica cega a modelo e modo
    audit_rows = sorted(({"text": t, "occurrences": c, "category": previous.get(t, {}).get("category", ""),
                          "notes": previous.get(t, {}).get("notes", "")} for t, c in totals.items()),
                        key=lambda r: hashlib.sha256(f"{SEED}:{r['text']}".encode()).hexdigest())
    bad = {r["category"] for r in audit_rows} - set(AUDIT_CATEGORIES) - {""}
    if bad:
        print(f"Categorias invalidas em {audit_path}: {bad}", file=sys.stderr)
        return 1
    adjudication = {r["text"]: r["category"] for r in audit_rows}
    audit_cond_rows = []
    for cond_id, counts in oov_by_condition.items():
        row = {"condition": cond_id, **{c: 0 for c in AUDIT_CATEGORIES}, "pending": 0}
        for text, cnt in counts.items():
            row[adjudication.get(text) or "pending"] += cnt
        audit_cond_rows.append(row)

    def write(name, rows, fields=None):
        with open(OUT / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields or list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    write("cases.csv", case_rows)
    write("hypotheses.csv", hyp_rows)
    write("summary.csv", summary_rows)
    write("per_class_recall.csv", class_rows)
    write("paired_differences.csv", paired_rows)
    write("oov_audit.csv", audit_rows, ["text", "occurrences", "category", "notes"])
    write("oov_audit_by_condition.csv", audit_cond_rows, ["condition", *AUDIT_CATEGORIES, "pending"])
    (OUT / "meta.json").write_text(json.dumps({
        "results_version": VERSION, "lexicon_version": LEXICON_VERSION, "K": K, "bootstrap_B": B, "seed": SEED, "n_cases": n_cases,
        "baseline_weights": BASELINE,
    }, indent=2) + "\n")

    for r in summary_rows:
        print(f"{r['condition']:<28} {r['status']:<11} valid {r['valid']:>2}/{r['n_cases']}  "
              f"Hit@1 {r['hit1']:.3f}  Hit@3 {r['hit3']:.3f}  MRR@3 {r['mrr3']:.3f}  Rec@3 {r['recall3']:.3f}  "
              f"OOV {r['share_oov']:.3f}")
    print(f"lexico {LEXICON_VERSION} | {len(audit_rows)} textos OOV unicos para auditoria em {audit_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
