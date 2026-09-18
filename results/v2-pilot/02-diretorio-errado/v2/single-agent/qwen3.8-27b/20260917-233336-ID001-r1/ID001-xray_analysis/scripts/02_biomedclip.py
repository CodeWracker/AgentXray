import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import open_clip

base = Path(__file__).resolve().parent.parent
meas = base / "measurements"
meas.mkdir(exist_ok=True)

MODEL_DIR = "local-dir:/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

torch.manual_seed(0)
np.random.seed(0)

model, preprocess = open_clip.create_model_from_pretrained(MODEL_DIR)
model.eval()
tokenizer = open_clip.get_tokenizer(MODEL_DIR)

img = Image.open(base / "images" / "00_original.png").convert("RGB")
x = preprocess(img).unsqueeze(0)

body_prompts = [
    "a radiograph of the foot",
    "a lateral radiograph of the foot and ankle",
    "a radiograph of the lower leg and foot",
    "a radiograph of the hand",
    "a radiograph of the wrist",
    "a chest radiograph",
    "a radiograph of the knee",
    "a lateral projection radiograph",
    "an anteroposterior projection radiograph",
]
fracture_prompts = [
    "a radiograph showing a fracture of the bone",
    "a radiograph of a broken bone",
    "a normal radiograph without fracture",
    "a radiograph of healthy bone",
    "a radiograph of an intact foot",
]

def encode_text(prompts):
    t = tokenizer(prompts)
    with torch.no_grad():
        feats = model.encode_text(t)
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats

with torch.no_grad():
    img_feat = model.encode_image(x)
img_feat /= img_feat.norm(dim=-1, keepdim=True)

results = {}
for name, prompts in [("body_region", body_prompts), ("fracture_probe", fracture_prompts)]:
    tf = encode_text(prompts)
    sims = (img_feat @ tf.T).squeeze(0).numpy()
    results[name] = {p: float(s) for p, s in zip(prompts, sims)}

# sanity: softmax over body region prompts
body_sims = np.array([results["body_region"][p] for p in body_prompts])
exp = np.exp((body_sims - body_sims.max()) * 20)  # temperature 1/20
results["body_region_softmax_t50"] = {p: float(e / exp.sum()) for p, e in zip(body_prompts, exp)}

(meas / "02_biomedclip_zeroshot.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
