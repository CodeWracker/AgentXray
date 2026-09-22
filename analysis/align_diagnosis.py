#!/usr/bin/env python3
"""Pontuacao de um diagnostico diferencial livre contra os rotulos de referencia (Exp 1).

Implementa o protocolo da secao "Evaluation Protocol for Experiment 1" do artigo:
rank por hipotese, janela de K hipoteses, cobertura multirrotulo por exemplo e resultado
de cada hipotese (correta, incorreta ou fora do vocabulario).

Uso:
    from align_diagnosis import score_case
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from clinical_ontology import project_hypothesis

K = 3
CATEGORIES = ["no_oov_hit", "no_oov_miss", "partial_oov_hit", "partial_oov_miss", "total_oov", "empty"]


def hypothesis_text(item) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("hypothesis", "diagnosis", "name", "label", "finding"):
            if isinstance(item.get(key), str):
                return item[key]
    return json.dumps(item, ensure_ascii=False)


def score_case(hypotheses, ground_truth_labels, k: int = K) -> dict:
    """Pontua um caso. `hypotheses` e a lista ordenada emitida; `ground_truth_labels` o conjunto de referencia."""
    gt = set(ground_truth_labels)
    if not isinstance(hypotheses, list):
        hypotheses = []

    window = []
    for item in hypotheses:
        text = hypothesis_text(item)
        proj = project_hypothesis(text)
        if not proj["informative"]:
            continue
        classes = proj["classes"]
        if not classes:
            outcome = "oov"
        elif gt.intersection(classes):
            outcome = "correct"
        else:
            outcome = "incorrect"
        window.append({"text": text, "classes": classes, "negated": proj["negated"], "outcome": outcome})
        if len(window) == k:
            break

    rank = next((j for j, h in enumerate(window, start=1) if h["outcome"] == "correct"), None)
    in_vocab = [h for h in window if h["outcome"] != "oov"]
    first_in_vocab = in_vocab[0] if in_vocab else None

    asserted = []
    for h in window:
        for c in h["classes"]:
            if c not in asserted:
                asserted.append(c)
    tp = len(gt.intersection(asserted))

    n = len(window)
    n_oov = sum(h["outcome"] == "oov" for h in window)
    if n == 0:
        category = "empty"
    elif n_oov == n:
        category = "total_oov"
    else:
        prefix = "partial_oov" if n_oov else "no_oov"
        category = f"{prefix}_{'hit' if rank else 'miss'}"

    return {
        "window": window,
        "n": n,
        "n_oov": n_oov,
        "n_correct": sum(h["outcome"] == "correct" for h in window),
        "n_incorrect": sum(h["outcome"] == "incorrect" for h in window),
        "rank": rank,
        "hit1": rank == 1,
        "hit3": rank is not None and rank <= 3,
        "mrr3": (1.0 / rank) if rank and rank <= 3 else 0.0,
        "asserted": asserted,
        "recall3": tp / len(gt) if gt else 0.0,
        "precision3": tp / len(asserted) if asserted else None,
        "f1": 2 * tp / (len(asserted) + len(gt)) if (asserted or gt) else 0.0,
        "oov_rate": n_oov / n if n else None,
        "category": category,
        "hit1_f": bool(first_in_vocab and first_in_vocab["outcome"] == "correct"),
    }


def evaluate_differential_diagnosis(hypotheses: list, ground_truth_labels: list[str]) -> dict:
    """Interface antiga (Hit@1, Hit@3, MRR), agora com rank por hipotese e janela K=3."""
    res = score_case(hypotheses, ground_truth_labels)
    gt = set(ground_truth_labels)
    res.update({
        "hit_at_1": res["hit1"],
        "hit_at_3": res["hit3"],
        "mrr": round(res["mrr3"], 3),
        "matched_rank": res["rank"],
        "ranked_classes": res["asserted"],
        "spurious_classes": [c for c in res["asserted"] if c not in gt and c != "No Finding"],
    })
    return res
