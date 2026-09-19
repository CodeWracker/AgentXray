#!/usr/bin/env python3
"""Avalia baseline supervisionada TorchXRayVision DenseNet-121 no benchmark Exp 1.

Calcula Hit@1, Hit@3 e MRR classificando as 45 imagens do manifesto
e ordenando as probabilidades das patologias.
"""

import json
import pathlib
import numpy as np
import skimage.io
import torch
import torchxrayvision as xrv

EVAL = pathlib.Path(__file__).resolve().parent.parent
MANIFEST_PATH = EVAL / "inputs" / "benchmarks" / "exp1_image_level" / "manifest.json"
IMAGES_DIR = EVAL / "inputs" / "nih_chestxray" / "images"

NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Carregando DenseNet-121 (densenet121-res224-all)...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    # Mapeia indices do modelo para classes do NIH
    pathology_indices = {}
    for idx, p in enumerate(model.pathologies):
        if p in NIH_CLASSES:
            pathology_indices[p] = idx

    results = []
    print(f"Avaliando {len(manifest)} imagens...")

    with torch.no_grad():
        for it in manifest:
            img_name = it["image"]
            gt = it["finding_labels"]
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                cands = list(EVAL.glob(f"**/{img_name}"))
                if cands:
                    img_path = cands[0]
                else:
                    continue

            # Carrega e preprocessa imagem para XRV (escala [-1024, 1024])
            img = skimage.io.imread(str(img_path))
            img = xrv.datasets.normalize(img, 255)
            # Converte para canal unico se for RGB
            if len(img.shape) > 2:
                img = img[:, :, 0]
            # Redimensiona para 224x224
            img = img[None, None, ...]
            transform = torchvision.transforms.Resize((224, 224), antialias=True) if "torchvision" in globals() else None
            t_img = torch.from_numpy(img).float()
            if t_img.shape[-1] != 224 or t_img.shape[-2] != 224:
                import torch.nn.functional as F
                t_img = F.interpolate(t_img, size=(224, 224), mode="bilinear", align_corners=False)

            preds = model(t_img).squeeze().cpu().numpy()

            # Pega scores apenas das 14 classes NIH
            scores = [(cls_name, float(preds[idx])) for cls_name, idx in pathology_indices.items()]
            scores.sort(key=lambda x: x[1], reverse=True)

            ranked_classes = [s[0] for s in scores]
            gt_set = set(gt)

            first_hit_rank = None
            for r_idx, cls_name in enumerate(ranked_classes, start=1):
                if cls_name in gt_set:
                    first_hit_rank = r_idx
                    break

            h1 = first_hit_rank == 1
            h3 = (first_hit_rank is not None) and (first_hit_rank <= 3)
            mrr = (1.0 / first_hit_rank) if first_hit_rank else 0.0

            results.append({
                "image": img_name,
                "gt": gt,
                "top3": ranked_classes[:3],
                "hit_at_1": h1,
                "hit_at_3": h3,
                "mrr": round(mrr, 3)
            })

    n = len(results)
    h1_count = sum(1 for r in results if r["hit_at_1"])
    h3_count = sum(1 for r in results if r["hit_at_3"])
    avg_mrr = sum(r["mrr"] for r in results) / n

    print(f"\n========================================")
    print(f"RESUMO SUPERVISIONADO: DenseNet-121 ({n} casos)")
    print(f"  Hit@1: {h1_count / n * 100:.1f}% ({h1_count}/{n})")
    print(f"  Hit@3: {h3_count / n * 100:.1f}% ({h3_count}/{n})")
    print(f"  MRR Medio: {avg_mrr:.3f}")
    print(f"========================================\n")


if __name__ == "__main__":
    main()
