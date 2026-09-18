import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

base = Path(__file__).resolve().parent.parent
meas = base / "measurements"
meas.mkdir(exist_ok=True)
P = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/prithivMLmods/Bone-Fracture-Detection"

torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

model = AutoModelForImageClassification.from_pretrained(P)
model.eval()
processor = AutoImageProcessor.from_pretrained(P)
print("id2label:", model.config.id2label)

orig = Image.open(base / "images" / "00_original.png").convert("RGB")

def run(img, name):
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
    out = {}
    for i, (k, v) in enumerate(model.config.id2label.items()):
        out[v] = float(probs[i])
    out["__argmax__"] = model.config.id2label[int(torch.argmax(probs))]
    return {name: out}

results = run(orig, "full_image")

# crops for stability check: leg-only (top half) and foot-only (bottom-right)
w, h = orig.size
leg_crop = orig.crop((0, 0, w, int(h * 0.55)))
foot_crop = orig.crop((int(w * 0.25), int(h * 0.45), w, h))
results.update(run(leg_crop, "crop_leg_only"))
results.update(run(foot_crop, "crop_foot_only"))

# uniform-gray sanity input (no anatomy) to see where OOD/noise lands
noise = Image.new("RGB", (w, h), (60, 60, 60))
results.update(run(noise, "sanity_uniform_gray"))

(meas / "03_bonefracture_classifier.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
