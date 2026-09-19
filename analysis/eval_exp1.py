#!/usr/bin/env python3
"""Avalia resultados do Experimento 1 (Diagnostico Diferencial Aberto) no NIH ChestX-ray14."""

import glob
import json
import pathlib
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = EVAL / "inputs" / "benchmarks" / "exp1_image_level" / "manifest.json"

sys.path.insert(0, str(EVAL / "analysis"))
from align_diagnosis import evaluate_differential_diagnosis


def main():
    if not MANIFEST.exists():
        print(f"Erro: {MANIFEST} nao encontrado.")
        return 1

    with open(MANIFEST, encoding="utf-8") as f:
        manifest = {item["image"]: item["finding_labels"] for item in json.load(f)}

    models = ["qwen3.6-27b", "qwen3.8-27b", "gemma4-26b"]
    for model in models:
        print(f"\n{'='*25} MODELO: {model} {'='*25}")
        runs = sorted(glob.glob(str(EVAL / f"results/v2/exp1_image_level/single-agent/{model}/*")))
        mrr_list = []
        hit1_list = []
        hit3_list = []

        for r in runs:
            rdir = pathlib.Path(r)
            harness = rdir / "harness_result.json"
            if not harness.exists():
                continue

            jsons = list(rdir.glob("*_analysis/*.png.json"))
            if not jsons:
                continue

            try:
                data = json.loads(jsons[0].read_text(encoding="utf-8"))
            except Exception:
                continue

            img_name = data.get("image")
            gt_labels = manifest.get(img_name, [])
            dx = data.get("differential_diagnosis", [])
            eval_res = evaluate_differential_diagnosis(dx, gt_labels)

            h1 = eval_res["hit_at_1"]
            h3 = eval_res["hit_at_3"]
            mrr_val = eval_res["mrr"]
            mrr_list.append(mrr_val)
            hit1_list.append(h1)
            hit3_list.append(h3)

            ranked_mapped = eval_res.get("ranked_classes", [])[:3]
            print(f"  Imagem: {img_name}")
            print(f"    Ground Truth: {gt_labels}")
            print(f"    Agente (Top-3): {dx[:3]}")
            print(f"    Classes Mapeadas: {ranked_mapped}")
            print(f"    Hit@1: {h1} | Hit@3: {h3} | MRR: {mrr_val}")

        if mrr_list:
            n = len(mrr_list)
            p_hit1 = (sum(hit1_list) / n) * 100
            p_hit3 = (sum(hit3_list) / n) * 100
            avg_mrr = sum(mrr_list) / n
            print(f"\n  --- RESUMO {model} ({n} casos completados) ---")
            print(f"  Hit@1: {p_hit1:.1f}% ({sum(hit1_list)}/{n})")
            print(f"  Hit@3: {p_hit3:.1f}% ({sum(hit3_list)}/{n})")
            print(f"  MRR Medio: {avg_mrr:.3f}")
        else:
            print(f"  Nenhuma rodada com JSON completo ainda.")


if __name__ == "__main__":
    main()
