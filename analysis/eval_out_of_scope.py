#!/usr/bin/env python3
"""Investiga ocorrencia de diagnosticos Fora do Escopo (Out-of-Scope / OOV) nas hipoteses emitidas."""

import collections
import glob
import json
import pathlib
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL / "analysis"))
from clinical_ontology import match_term_to_class, ONTOLOGY


def main():
    runs = sorted(glob.glob(str(EVAL / "results/v2/exp1_image_level/*/*/*")))
    out_of_scope = collections.Counter()
    model_stats = collections.defaultdict(lambda: {"in_scope": 0, "out_of_scope": 0, "terms": collections.Counter()})
    total = 0
    in_scope_total = 0

    for r in runs:
        jsons = list(pathlib.Path(r).glob("*_analysis/*.png.json"))
        if not jsons:
            continue
        model = pathlib.Path(r).parent.name
        try:
            data = json.loads(jsons[0].read_text(encoding="utf-8"))
            dx = data.get("differential_diagnosis", [])
            for hyp in dx:
                total += 1
                matched = match_term_to_class(hyp)
                valid = [c for c in matched if c in ONTOLOGY]
                if valid:
                    in_scope_total += 1
                    model_stats[model]["in_scope"] += 1
                else:
                    out_of_scope[hyp.strip()] += 1
                    model_stats[model]["out_of_scope"] += 1
                    model_stats[model]["terms"][hyp.strip()] += 1
        except Exception:
            pass

    print(f"Total de hipoteses clinicas avaliadas: {total}")
    print(f"Hipoteses dentro das 14 classes (+ No Finding): {in_scope_total} ({in_scope_total / max(1, total) * 100:.1f}%)")
    oos_total = total - in_scope_total
    print(f"Hipoteses Fora do Escopo (Out-of-Scope): {oos_total} ({oos_total / max(1, total) * 100:.1f}%)")

    print("\n--- Estatisticas de Diagnosticos Fora do Escopo por Modelo ---")
    for m in sorted(model_stats.keys()):
        st = model_stats[m]
        tot = st["in_scope"] + st["out_of_scope"]
        pct = (st["out_of_scope"] / max(1, tot)) * 100
        n_in = st["in_scope"]
        n_out = st["out_of_scope"]
        print(f"  {m}: Total={tot} | In-Scope={n_in} | Out-of-Scope={n_out} ({pct:.1f}%)")
        print("    Termos fora do escopo mais frequentes:")
        for t, cnt in st["terms"].most_common(3):
            print(f"      [{cnt}x] {t}")

    print("\n--- Top 15 Diagnosticos Fora do Escopo Mais Frequentes Globalmente ---")
    for term, count in out_of_scope.most_common(15):
        print(f"  [{count:2d}x] {term}")


if __name__ == "__main__":
    main()
