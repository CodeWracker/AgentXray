"""
Reproduce script for ID001-xray_analysis.
Regenerates every file in images/ (except 00_original.*) and in measurements/
from the original image. Deterministic.
"""
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

import os
import json
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base, 'images', '00_original.png')
img_out = os.path.join(base, 'images')
meas_out = os.path.join(base, 'measurements')

os.makedirs(img_out, exist_ok=True)
os.makedirs(meas_out, exist_ok=True)

# ── Load original image ──
img = Image.open(img_path).convert('L')
arr = np.array(img, dtype=np.float32)

# ═══════════════════════════════════════════════════════════
# Analysis 1: Image enhancement — CLAHE and composite
# ═══════════════════════════════════════════════════════════
arr_uint8 = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(arr_uint8)

# Histogram equalization (for composite only)
hist_eq = cv2.equalizeHist(arr_uint8)

# Gamma correction (for composite only)
gamma_arr = 255 * np.power(arr_uint8 / 255.0, 1.0 / 0.5)
gamma_arr = gamma_arr.astype(np.uint8)

# Save composite
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.imshow(arr_uint8, cmap='gray')
plt.title('Original (normalized)')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.imshow(clahe_result, cmap='gray')
plt.title('CLAHE (clip=3, 8x8)')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.imshow(gamma_arr, cmap='gray')
plt.title('Gamma 0.5 (brighten)')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.imshow(hist_eq, cmap='gray')
plt.title('Global histogram equalization')
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(img_out, '01_enhanced_composite.png'), dpi=150)
plt.close()

# Save CLAHE
Image.fromarray(clahe_result).save(os.path.join(img_out, '02_clahe.png'))

# Save image statistics
stats = {
    "original_shape": list(arr.shape),
    "original_min": float(arr.min()),
    "original_max": float(arr.max()),
    "original_mean": float(arr.mean()),
    "original_std": float(arr.std()),
    "clahe_mean": float(clahe_result.mean()),
    "clahe_std": float(clahe_result.std()),
    "gamma_mean": float(gamma_arr.mean()),
    "gamma_std": float(gamma_arr.std()),
    "histeq_mean": float(hist_eq.mean()),
    "histeq_std": float(hist_eq.std()),
}
with open(os.path.join(meas_out, '01_image_statistics.json'), 'w') as f:
    json.dump(stats, f, indent=2)

# ═══════════════════════════════════════════════════════════
# Analysis 2: Bone-Fracture-Detection classifier
# ═══════════════════════════════════════════════════════════
from transformers import AutoModelForImageClassification, AutoImageProcessor

model_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/prithivMLmods/Bone-Fracture-Detection"
processor = AutoImageProcessor.from_pretrained(model_path)
model = AutoModelForImageClassification.from_pretrained(model_path)
model.eval()

img_rgb = Image.open(img_path).convert('RGB')
inputs = processor(images=img_rgb, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)

labels = model.config.id2label
probs_list = probs.squeeze().tolist()
pred_label = labels[preds.item()]

fracture_results = {
    "model": "prithivMLmods/Bone-Fracture-Detection",
    "base_model": "google/siglip2-base-patch16-224",
    "labels": labels,
    "predicted_class": pred_label,
    "predicted_index": int(preds.item()),
    "probabilities": {labels[i]: float(p) for i, p in enumerate(probs_list)},
    "note": "Model not peer-reviewed. Training dataset unknown for body region. Must validate against visual findings."
}

with open(os.path.join(meas_out, '02_fracture_classifier.json'), 'w') as f:
    json.dump(fracture_results, f, indent=2)

# ═══════════════════════════════════════════════════════════
# Analysis 3: BiomedCLIP zero-shot
# ═══════════════════════════════════════════════════════════
import open_clip

biomedclip_path = "local-dir:/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
bc_model, bc_preprocess = open_clip.create_model_from_pretrained(biomedclip_path)
bc_tokenizer = open_clip.get_tokenizer(biomedclip_path)

positive_prompts = [
    "lateral X-ray of a foot with fracture",
    "lateral X-ray of an ankle with fracture",
    "radiograph of a foot showing abnormality",
    "foot X-ray with bone lesion",
    "ankle radiograph with joint effusion",
    "foot radiograph with soft tissue swelling",
    "fractured calcaneus X-ray",
    "fractured talus radiograph",
    "fractured distal fibula",
    "avulsion fracture foot X-ray",
    "stress fracture foot radiograph",
    "fractured metatarsal X-ray",
]
negative_prompts = [
    "normal lateral foot X-ray",
    "normal lateral ankle radiograph",
    "normal foot radiograph without abnormality",
    "healthy foot X-ray lateral view",
    "unremarkable ankle X-ray",
]
all_prompts = positive_prompts + negative_prompts

img_input = bc_preprocess(img_rgb).unsqueeze(0)

with torch.no_grad():
    image_features = bc_model.encode_image(img_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_inputs = bc_tokenizer(all_prompts)
    text_features = bc_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    logits = (image_features @ text_features.T).squeeze(0)
    probs = torch.softmax(logits, dim=0)

probs_list = probs.tolist()

bc_results = {
    "model": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "all_prompts_scores": {
        all_prompts[i]: float(probs_list[i]) for i in range(len(all_prompts))
    },
    "positive_findings_scores": {
        all_prompts[i]: float(probs_list[i]) for i in range(len(positive_prompts))
    },
    "negative_normal_scores": {
        all_prompts[i + len(positive_prompts)]: float(probs_list[i]) for i in range(len(negative_prompts))
    },
    "top_match": max(all_prompts, key=lambda p: float(probs_list[all_prompts.index(p)])),
    "note": "Zero-shot classification, not diagnostic. Prompt-dependent results."
}

with open(os.path.join(meas_out, '03_biomedclip_zeroshot.json'), 'w') as f:
    json.dump(bc_results, f, indent=2)

# ═══════════════════════════════════════════════════════════
# Analysis 4: MedSAM segmentation
# ═══════════════════════════════════════════════════════════
from transformers import SamModel, SamProcessor

sam_model_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/wanglab/medsam-vit-base"
sam_model = SamModel.from_pretrained(sam_model_path)
sam_processor = SamProcessor.from_pretrained(sam_model_path)

arr_rgb = np.array(img_rgb)
h, w = arr_rgb.shape[:2]

boxes = {
    "calcaneus_region": [int(w*0.18), int(h*0.62), int(w*0.42), int(h*0.95)],
    "talus_region": [int(w*0.22), int(h*0.45), int(w*0.42), int(h*0.68)],
    "distal_tibia_fibula": [int(w*0.22), int(h*0.02), int(w*0.40), int(h*0.55)],
    "metatarsal_region": [int(w*0.42), int(h*0.55), int(w*0.85), int(h*0.85)],
}

sam_results = {}
composite_axes_data = []

for name, box in boxes.items():
    box = [max(0, b) for b in box]
    box = [min(b, [w, h, w, h][i]) for i, b in enumerate(box)]

    inputs = sam_processor(
        images=img_rgb,
        input_boxes=[[box]],
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
        inputs["reshaped_input_sizes"]
    )

    mask = masks[0][0][0].cpu().numpy()

    sam_results[name] = {
        "box": box,
        "mask_pixels": int((mask > 0).sum()),
        "mask_fraction": float(mask.mean()),
    }

    composite_axes_data.append((name, box, mask))

# Save composite visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, (name, box, mask) in enumerate(composite_axes_data):
    axes[idx].imshow(arr_rgb)
    axes[idx].contour(mask, colors='red', linewidths=2)
    axes[idx].set_title(f'MedSAM: {name}\nbox={box}')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(img_out, '06_medsam_composite.png'), dpi=150)
plt.close()

with open(os.path.join(meas_out, '05_medsam.json'), 'w') as f:
    json.dump(sam_results, f, indent=2)

print("Reproduction complete. All files regenerated.")