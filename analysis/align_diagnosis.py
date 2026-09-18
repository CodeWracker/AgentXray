#!/usr/bin/env python3
"""Modulo de alinhamento pos-hoc de diagnosticos diferenciais com o Ground Truth.

Recebe as hipoteses livres ordenadas do agente e compara contra as classes
do dataset sem forcar escolhas fechadas no prompt, calculando Hit@1, Hit@3 e MRR.

Uso:
    from align_diagnosis import evaluate_differential_diagnosis
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from clinical_ontology import match_term_to_class


def evaluate_differential_diagnosis(hypotheses: list[str], ground_truth_labels: list[str]) -> dict:
    """Compara lista ordenada de hipoteses clinicas contra as classes do ground truth.
    
    Args:
        hypotheses: lista de strings em ordem de verossimilhanca emitida pelo agente
                    ex: ["Pneumonia em lobo inferior direito", "Derrame pleural laminar", "Normal"]
        ground_truth_labels: classes verdadeiras do dataset ex: ["Pneumonia", "Effusion"]
    """
    if not hypotheses:
        return {
            "hit_at_1": False,
            "hit_at_3": False,
            "mrr": 0.0,
            "matched_rank": None,
            "ranked_classes": [],
            "spurious_classes": [],
        }

    # Remove duplicates preservando a ordem
    gt_set = set(ground_truth_labels)
    ranked_classes = []
    seen = set()

    for hyp in hypotheses:
        matched = match_term_to_class(hyp)
        for cls_name in matched:
            if cls_name not in seen:
                ranked_classes.append(cls_name)
                seen.add(cls_name)

    # Identifica o primeiro acerto
    first_hit_rank = None
    for idx, cls_name in enumerate(ranked_classes, start=1):
        if cls_name in gt_set:
            first_hit_rank = idx
            break

    hit_at_1 = first_hit_rank == 1
    hit_at_3 = (first_hit_rank is not None) and (first_hit_rank <= 3)
    mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0

    # Classes diagnosticadas que nao estao no ground truth
    spurious = [c for c in ranked_classes[:3] if c not in gt_set and c != "No Finding"]

    return {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "mrr": round(mrr, 3),
        "matched_rank": first_hit_rank,
        "ranked_classes": ranked_classes,
        "spurious_classes": spurious,
    }
