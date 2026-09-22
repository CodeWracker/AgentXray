#!/usr/bin/env python3
"""Roda a baseline supervisionada TorchXRayVision DenseNet-121 nas imagens do Exp 1 e grava as probabilidades.

A pontuacao (Hit@k, cobertura, etc.) e feita por analysis/exp1_metrics.py com as mesmas regras dos modelos.

Uso:
    tools/py analysis/eval_supervised_densenet.py
Saida:
    results/v2/supervised/exp1_image_level/densenet121-res224-all/predictions.json
"""

import json
import pathlib

import skimage.io
import torch
import torch.nn.functional as F
import torchxrayvision as xrv

EVAL = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = EVAL / "inputs" / "benchmarks" / "exp1_image_level" / "manifest.json"
IMAGES_DIR = EVAL / "inputs" / "nih_chestxray" / "images"
WEIGHTS = "densenet121-res224-all"
OUT = EVAL / "results" / "v2" / "supervised" / "exp1_image_level" / WEIGHTS / "predictions.json"

NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]


def main():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model = xrv.models.DenseNet(weights=WEIGHTS)
    model.eval()
    index = {p: i for i, p in enumerate(model.pathologies) if p in NIH_CLASSES}
    assert set(index) == set(NIH_CLASSES), f"classes ausentes no modelo: {set(NIH_CLASSES) - set(index)}"

    predictions = {}
    with torch.no_grad():
        for it in manifest:
            img_path = IMAGES_DIR / it["image"]
            if not img_path.exists():
                img_path = next(EVAL.glob(f"inputs/**/{it['image']}"))
            img = xrv.datasets.normalize(skimage.io.imread(str(img_path)), 255)
            if img.ndim > 2:
                img = img[:, :, 0]
            t_img = F.interpolate(torch.from_numpy(img[None, None, ...]).float(), size=(224, 224),
                                  mode="bilinear", align_corners=False)
            out = model(t_img).squeeze().cpu().numpy()
            scores = {c: float(out[i]) for c, i in index.items()}
            predictions[it["image"]] = {
                "scores": scores,
                "ranked_classes": sorted(scores, key=scores.get, reverse=True),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"weights": WEIGHTS, "predictions": predictions}, indent=1) + "\n")
    print(f"{len(predictions)} imagens -> {OUT}")


if __name__ == "__main__":
    main()
