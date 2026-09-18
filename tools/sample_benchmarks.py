#!/usr/bin/env python3
"""Gera e fixa os subconjuntos estratificados dos dois experimentos.

Experimento 1: Classificacao image-level e diagnostico diferencial aberto.
Experimento 2: Localizacao espacial e deteccao de lesoes com bounding box (BBox_List_2017.csv).

Uso:
    tools/py tools/sample_benchmarks.py [--n-exp1 50] [--n-exp2 50]
"""

import argparse
import csv
import json
import pathlib
import random
import sys
from collections import defaultdict

EVAL = pathlib.Path(__file__).resolve().parent.parent
NIH_DIR = EVAL / "inputs" / "nih_chestxray"
DATA_ENTRY = NIH_DIR / "Data_Entry_2017_v2020.csv"
BBOX_CSV = NIH_DIR / "BBox_List_2017.csv"
TEST_LIST = NIH_DIR / "test_list.txt"
BENCHMARKS_DIR = EVAL / "inputs" / "benchmarks"


def build_exp1_sample(n_target: int = 50, seed: int = 42) -> list[dict]:
    """Amostra estratificada do test set cobrindo as 14 patologias e normais."""
    random.seed(seed)
    test_images = set()
    if TEST_LIST.exists():
        test_images = set(TEST_LIST.read_text().splitlines())

    by_class = defaultdict(list)
    with open(DATA_ENTRY, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("Image Index")
            if test_images and img not in test_images:
                continue
            labels = [lbl.strip() for lbl in row.get("Finding Labels", "").split("|")]
            for lbl in labels:
                by_class[lbl].append(row)

    sampled_images = {}
    classes = sorted(by_class.keys())
    per_class = max(2, n_target // len(classes))

    for cls_name in classes:
        pool = by_class[cls_name]
        selected = random.sample(pool, min(per_class, len(pool)))
        for item in selected:
            img_id = item["Image Index"]
            if img_id not in sampled_images:
                sampled_images[img_id] = {
                    "image": img_id,
                    "finding_labels": [lbl.strip() for lbl in item.get("Finding Labels", "").split("|")],
                    "view_position": item.get("View Position"),
                    "patient_id": item.get("Patient ID"),
                    "patient_age": item.get("Patient Age"),
                    "patient_gender": item.get("Patient Gender"),
                }
            if len(sampled_images) >= n_target:
                break
        if len(sampled_images) >= n_target:
            break

    return list(sampled_images.values())[:n_target]


def build_exp2_sample(n_target: int = 50, seed: int = 42) -> list[dict]:
    """Amostra balanceada das 8 classes anotadas com Bounding Box por radiologistas."""
    random.seed(seed)
    by_class = defaultdict(list)
    with open(BBOX_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("Finding Label", "").strip()
            by_class[lbl].append(row)

    sampled = {}
    classes = sorted(by_class.keys())
    per_class = max(3, n_target // len(classes))

    for cls_name in classes:
        pool = by_class[cls_name]
        selected = random.sample(pool, min(per_class, len(pool)))
        for item in selected:
            img_id = item.get("Image Index", "").strip()
            try:
                x = float(item.get("Bbox [x", item.get("x", 0)))
                y = float(item.get("y", 0))
                w = float(item.get("w", 0))
                h = float(item.get("h]", item.get("h", 0)))
            except (ValueError, TypeError):
                continue

            if img_id not in sampled:
                sampled[img_id] = {
                    "image": img_id,
                    "bboxes": [],
                }
            sampled[img_id]["bboxes"].append({
                "label": cls_name,
                "bbox": [x, y, w, h],
            })
            if len(sampled) >= n_target:
                break
        if len(sampled) >= n_target:
            break

    return list(sampled.values())[:n_target]


def main():
    parser = argparse.ArgumentParser(description="Gera subconjuntos fixos dos dois experimentos.")
    parser.add_argument("--n-exp1", type=int, default=50, help="numero de imagens para o experimento 1")
    parser.add_argument("--n-exp2", type=int, default=50, help="numero de imagens para o experimento 2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    exp1_dir = BENCHMARKS_DIR / "exp1_image_level"
    exp2_dir = BENCHMARKS_DIR / "exp2_bbox_localization"
    exp1_dir.mkdir(parents=True, exist_ok=True)
    exp2_dir.mkdir(parents=True, exist_ok=True)

    exp1_sample = build_exp1_sample(args.n_exp1, args.seed)
    (exp1_dir / "manifest.json").write_text(json.dumps(exp1_sample, indent=2, ensure_ascii=False) + "\n")
    print(f"Experimento 1 fixado: {len(exp1_sample)} imagens em {exp1_dir / 'manifest.json'}")

    exp2_sample = build_exp2_sample(args.n_exp2, args.seed)
    (exp2_dir / "manifest.json").write_text(json.dumps(exp2_sample, indent=2, ensure_ascii=False) + "\n")
    print(f"Experimento 2 fixado: {len(exp2_sample)} imagens em {exp2_dir / 'manifest.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
