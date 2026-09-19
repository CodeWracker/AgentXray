#!/usr/bin/env python3
"""Avaliacao de localizacao de lesoes e bounding boxes (Experimento 2).

Compara caixas delimitadoras preditas pelo agente com as anotacoes de especialistas
do arquivo BBox_List_2017.csv, calculando IoU, precisao de localizacao em limiares
IoU >= 0.1, 0.3 e 0.5, e compatibilidade de classe.

Uso:
    from bbox_eval import evaluate_bboxes, compute_iou
"""

import csv
import json
import pathlib
import sys

EVAL = pathlib.Path(__file__).resolve().parent.parent
BBOX_CSV = EVAL / "inputs" / "nih_chestxray" / "BBox_List_2017.csv"


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Calcula IoU entre duas caixas no formato [x, y, w, h]."""
    xa, ya, wa, ha = box_a
    xb, yb, wb, hb = box_b

    xa1, ya1, xa2, ya2 = xa, ya, xa + wa, ya + ha
    xb1, yb1, xb2, yb2 = xb, yb, xb + wb, yb + hb

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = wa * ha
    area_b = wb * hb
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_ground_truth_bboxes() -> dict[str, list[dict]]:
    """Carrega anotações do BBox_List_2017.csv agrupadas por imagem."""
    if not BBOX_CSV.exists():
        return {}

    records = {}
    with open(BBOX_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("Image Index", "").strip()
            label = row.get("Finding Label", "").strip()
            try:
                # Trata chaves do CSV do NIH que podem ter espacos
                x = float(row.get("Bbox [x", row.get("x", 0)))
                y = float(row.get("y", 0))
                w = float(row.get("w", 0))
                h = float(row.get("h]", row.get("h", 0)))
            except (ValueError, TypeError):
                continue

            if img not in records:
                records[img] = []
            records[img].append({"label": label, "bbox": [x, y, w, h]})
    return records


def evaluate_bboxes(predicted_boxes: list[dict], ground_truth_boxes: list[dict]) -> dict:
    """Avalia conjunto de caixas preditas contra o ground truth de uma imagem.
    
    Args:
        predicted_boxes: lista de dicts [{'label': str, 'bbox': [x, y, w, h]}]
        ground_truth_boxes: lista de dicts [{'label': str, 'bbox': [x, y, w, h]}]
    """
    if not ground_truth_boxes:
        return {
            "has_gt": False,
            "max_iou": 0.0,
            "hit_iou_01": False,
            "hit_iou_03": False,
            "hit_iou_05": False,
            "best_match_label": None,
        }

    if not predicted_boxes:
        return {
            "has_gt": True,
            "max_iou": 0.0,
            "hit_iou_01": False,
            "hit_iou_03": False,
            "hit_iou_05": False,
            "best_match_label": None,
        }

    best_iou = 0.0
    best_label = None

    for pred in predicted_boxes:
        p_box = pred.get("bbox", [])
        if len(p_box) != 4:
            continue
        p_label = pred.get("label", "").lower()

        for gt in ground_truth_boxes:
            g_box = gt.get("bbox", [])
            g_label = gt.get("label", "").lower()

            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_label = gt.get("label")

    return {
        "has_gt": True,
        "max_iou": round(best_iou, 3),
        "hit_iou_01": best_iou >= 0.1,
        "hit_iou_03": best_iou >= 0.3,
        "hit_iou_05": best_iou >= 0.5,
        "best_match_label": best_label,
    }


def extract_predicted_bboxes(json_data: dict) -> list[dict]:
    """Extrai bounding boxes do JSON final gerado pelo agente ou zero-shot.
    Suporta campos: 'localized_lesions', 'bboxes', 'bounding_boxes'.
    """
    candidates = json_data.get("localized_lesions") or json_data.get("bboxes") or json_data.get("bounding_boxes") or []
    boxes = []
    if isinstance(candidates, list):
        for item in candidates:
            if isinstance(item, dict):
                label = item.get("label", item.get("pathology", item.get("finding", "abnormality")))
                box = item.get("bbox", item.get("box", item.get("coordinates", [])))
                if isinstance(box, list) and len(box) == 4:
                    try:
                        boxes.append({"label": str(label), "bbox": [float(c) for c in box]})
                    except (ValueError, TypeError):
                        pass
    return boxes
