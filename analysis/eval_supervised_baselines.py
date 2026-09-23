#!/usr/bin/env python3
"""Baselines de conjunto fechado do Exp 1: classificadores do TorchXRayVision e BiomedCLIP zero-shot.

Cada baseline grava, por imagem, um escore por classe do ChestX-ray14 e o ranking; a pontuacao (Hit@k, cobertura
etc.) e feita por analysis/exp1_metrics.py com as mesmas regras dos modelos. Classes que um classificador nao tem
ficam fora do ranking dele (ele nunca pode acerta-las), e isso entra na comparacao.

Grupos de pesos:
- viram o ChestX-ray14 no treino: densenet121-res224-all, densenet121-res224-nih, resnet50-res512-all
  (se a particao oficial de teste ficou fora do treino e testado por analysis/txv_leakage_check.py);
- nunca viram o ChestX-ray14: densenet121-res224-pc, -chex, -mimic_nb, -mimic_ch.

Uso:
    tools/py analysis/eval_supervised_baselines.py
Saida:
    results/supervised/exp1_image_level/<baseline>/predictions.json
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
OUT = EVAL / "results" / "supervised" / "exp1_image_level"
BIOMEDCLIP = EVAL / "models" / "hf" / "microsoft" / "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

NIH_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia",
    "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
]
TXV = {
    "densenet121-res224-all": (xrv.models.DenseNet, 224, True),
    "densenet121-res224-nih": (xrv.models.DenseNet, 224, True),
    "resnet50-res512-all": (xrv.models.ResNet, 512, True),
    "densenet121-res224-pc": (xrv.models.DenseNet, 224, False),
    "densenet121-res224-chex": (xrv.models.DenseNet, 224, False),
    "densenet121-res224-mimic_nb": (xrv.models.DenseNet, 224, False),
    "densenet121-res224-mimic_ch": (xrv.models.DenseNet, 224, False),
}


def image_path(name: str) -> pathlib.Path:
    p = IMAGES_DIR / name
    return p if p.exists() else next(EVAL.glob(f"inputs/**/{name}"))


def load_xrv(name: str, size: int) -> torch.Tensor:
    img = xrv.datasets.normalize(skimage.io.imread(str(image_path(name))), 255)
    if img.ndim > 2:
        img = img[:, :, 0]
    return F.interpolate(torch.from_numpy(img[None, None, ...]).float(), size=(size, size),
                         mode="bilinear", align_corners=False)


def write(baseline: str, predictions: dict, meta: dict):
    out = OUT / baseline / "predictions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"weights": baseline, **meta, "predictions": predictions}, indent=1) + "\n")
    print(f"{len(predictions)} imagens -> {out}")


def run_txv(manifest: list):
    for weights, (cls, size, seen) in TXV.items():
        model = cls(weights=weights).eval()
        index = {p: i for i, p in enumerate(model.pathologies) if p in NIH_CLASSES}
        predictions = {}
        with torch.no_grad():
            for it in manifest:
                out = model(load_xrv(it["image"], size)).squeeze().cpu().numpy()
                scores = {c: float(out[i]) for c, i in index.items()}
                predictions[it["image"]] = {"scores": scores, "ranked_classes": sorted(scores, key=scores.get, reverse=True)}
        write(weights, predictions, {"seen_chestxray14": seen, "classes": sorted(index)})


def run_biomedclip(manifest: list):
    import open_clip
    from PIL import Image
    ref = f"local-dir:{BIOMEDCLIP}"
    model, preprocess = open_clip.create_model_from_pretrained(ref)
    tokenizer = open_clip.get_tokenizer(ref)
    model.eval()
    labels = NIH_CLASSES + ["No Finding"]
    # um prompt por classe; "No Finding" vira radiografia normal
    texts = [f"a chest x-ray showing {c.replace('_', ' ').lower()}" for c in NIH_CLASSES] + ["a normal chest x-ray"]
    predictions = {}
    with torch.no_grad():
        t = model.encode_text(tokenizer(texts, context_length=256))
        t = t / t.norm(dim=-1, keepdim=True)
        for it in manifest:
            im = preprocess(Image.open(image_path(it["image"])).convert("RGB")).unsqueeze(0)
            v = model.encode_image(im)
            v = v / v.norm(dim=-1, keepdim=True)
            probs = (model.logit_scale.exp() * v @ t.T).softmax(dim=-1).squeeze().tolist()
            scores = dict(zip(labels, probs))
            predictions[it["image"]] = {"scores": scores, "ranked_classes": sorted(scores, key=scores.get, reverse=True)}
    write("biomedclip-zeroshot", predictions, {"seen_chestxray14": None, "classes": labels, "prompts": texts})


def main():
    # poucas threads: as filas dos agentes rodam na mesma maquina e o tempo de parede delas nao pode ser afetado
    torch.set_num_threads(8)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    run_txv(manifest)
    run_biomedclip(manifest)


if __name__ == "__main__":
    main()
