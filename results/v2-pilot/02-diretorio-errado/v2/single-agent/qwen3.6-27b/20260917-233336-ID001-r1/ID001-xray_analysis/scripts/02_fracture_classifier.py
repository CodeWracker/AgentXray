"""Analysis 2: Bone-Fracture-Detection (siglip2-based classifier)."""
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

import os
from PIL import Image
import json
from transformers import AutoModelForImageClassification, AutoImageProcessor

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base, 'images', '00_original.png')
meas_out = os.path.join(base, 'measurements')

model_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/prithivMLmods/Bone-Fracture-Detection"

processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.eval()

img = Image.open(img_path).convert('RGB')
inputs = processor(images=img, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

labels = model.config.id2label
probs_list = probs.squeeze().tolist()
pred_label = labels[preds.item()]

results = {
    "model": "prithivMLmods/Bone-Fracture-Detection",
    "base_model": "google/siglip2-base-patch16-224",
    "labels": labels,
    "predicted_class": pred_label,
    "predicted_index": int(preds.item()),
    "probabilities": {labels[i]: float(p) for i, p in enumerate(probs_list)},
    "note": "Model not peer-reviewed. Training dataset unknown for body region. Must validate against visual findings."
}

with open(os.path.join(meas_out, '02_fracture_classifier.json'), 'w') as f:
    json.dump(results, f, indent=2)
