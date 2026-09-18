"""Analysis 4: MedSAM segmentation of key bone regions."""
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base, 'images', '00_original.png')
img_out = os.path.join(base, 'images')
meas_out = os.path.join(base, 'measurements')

img = Image.open(img_path).convert('RGB')
arr = np.array(img)
h, w = arr.shape[:2]

# Bounding boxes for key regions on the lateral foot X-ray
# Format: [x0, y0, x1, y1]
# I'll define approximate boxes based on visual inspection of the image
boxes = {
    "calcaneus_region": [int(w*0.18), int(h*0.62), int(w*0.42), int(h*0.95)],
    "talus_region": [int(w*0.22), int(h*0.45), int(w*0.42), int(h*0.68)],
    "distal_tibia_fibula": [int(w*0.22), int(h*0.02), int(w*0.40), int(h*0.55)],
    "metatarsal_region": [int(w*0.42), int(h*0.55), int(w*0.85), int(h*0.85)],
}

# Try loading MedSAM
try:
    from transformers import SamModel, SamProcessor
    model_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/wanglab/medsam-vit-base"
    sam_model = SamModel.from_pretrained(model_path)
    sam_processor = SamProcessor.from_pretrained(model_path)
    sam_available = True
except Exception as e:
    print(f"MedSAM unavailable: {e}")
    sam_available = False

results = {}

if sam_available:
    img_pil = Image.open(img_path).convert('RGB')

    for name, box in boxes.items():
        # Ensure box is within image bounds
        box = [max(0, b) for b in box]
        box = [min(b, [w, h, w, h][i]) for i, b in enumerate(box)]

        inputs = sam_processor(
            images=img_pil,
            input_boxes=[[box]],
            return_tensors="pt",
            multimask_output=False
        )

        with torch.no_grad():
            outputs = sam_model(**inputs)

        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )

        mask = masks[0][0][0].cpu().numpy().astype(np.uint8)

        # Save mask
        mask_path = os.path.join(img_out, f'05_medsam_{name}_mask.png')
        Image.fromarray(mask * 255).save(mask_path)

        # Compute stats
        results[name] = {
            "box": box,
            "mask_pixels": int(mask.sum()),
            "mask_fraction": float(mask.mean()),
        }

    # Composite visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    axes = axes.ravel()

    img_rgb = np.array(Image.open(img_path).convert('RGB'))

    for idx, (name, box) in enumerate(boxes.items()):
        inputs = sam_processor(
            images=img_pil,
            input_boxes=[[box]],
            return_tensors="pt",
            multimask_output=False
        )
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"]
        )
        mask = masks[0][0][0].cpu().numpy()

        axes[idx].imshow(img_rgb)
        axes[idx].contour(mask, colors='red', linewidths=2)
        axes[idx].set_title(f'MedSAM: {name}\nbox={box}')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(img_out, '06_medsam_composite.png'), dpi=150)
    plt.close()

else:
    results["error"] = "MedSAM not available"
    # Save a placeholder
    with open(os.path.join(meas_out, '05_medsam.json'), 'w') as f:
        json.dump(results, f, indent=2)
    exit(0)

with open(os.path.join(meas_out, '05_medsam.json'), 'w') as f:
    json.dump(results, f, indent=2)
