#!/usr/bin/env python3
"""Subconjunto fixo de casos para a comparacao de conselhos entre os tres modelos (emenda post hoc de 30/09/2026).

O conselho do qwen3.8-27b custava cerca de 4 horas por caso numa unica GPU, o que tornava inviavel rodar os 186 casos
de conselho dele. A comparacao de conselhos passa a usar 10 casos por experimento, so no vocabulario aberto, os mesmos
para os tres modelos. A escolha usa apenas os rotulos de referencia e a ordem fixa dos manifestos, nunca uma saida de
modelo:

* Exp 1 (varios rotulos por imagem): cobertura gulosa dos rotulos. A cada passo entra o caso que acrescenta mais rotulos
  ainda nao cobertos; empates vao para o caso que aparece antes no manifesto.
* Exp 2 (8 classes, 6 casos cada, manifesto ordenado por classe): rodizio pelas classes na ordem do manifesto, o
  primeiro caso de cada classe e depois o segundo caso das primeiras classes, ate 10.

Uso:
    tools/py tools/sample_council_subset.py [--n 10]
Saida: inputs/benchmarks/<experimento>/manifest_council_subset.json
"""

import argparse
import json
import pathlib

EVAL = pathlib.Path(__file__).resolve().parent.parent
BENCH = EVAL / "inputs" / "benchmarks"


def greedy_label_cover(items: list[dict], n: int) -> list[dict]:
    chosen, covered = [], set()
    remaining = list(items)
    while len(chosen) < n and remaining:
        best = max(remaining, key=lambda it: (len(set(it["finding_labels"]) - covered), -items.index(it)))
        chosen.append(best)
        covered |= set(best["finding_labels"])
        remaining.remove(best)
    return sorted(chosen, key=items.index)


def round_robin_by_class(items: list[dict], n: int) -> list[dict]:
    by_class: dict[str, list[dict]] = {}
    for it in items:
        by_class.setdefault(it["bboxes"][0]["label"], []).append(it)
    chosen, rank = [], 0
    while len(chosen) < n:
        for cases in by_class.values():
            if rank < len(cases) and len(chosen) < n:
                chosen.append(cases[rank])
        rank += 1
    return sorted(chosen, key=items.index)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    for exp, pick in [("exp1_image_level", greedy_label_cover), ("exp2_bbox_localization", round_robin_by_class)]:
        items = json.loads((BENCH / exp / "manifest.json").read_text())
        subset = pick(items, args.n)
        (BENCH / exp / "manifest_council_subset.json").write_text(json.dumps(subset, indent=2) + "\n")
        labels = sorted({l for it in subset for l in (it.get("finding_labels") or [b["label"] for b in it["bboxes"]])})
        print(f"{exp}: {len(subset)} casos, {len(labels)} rotulos: {', '.join(labels)}")
        print("   " + " ".join(it["image"] for it in subset))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
