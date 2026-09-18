"""Analysis 3: BiomedCLIP zero-shot classification."""
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
import open_clip

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base, 'images', '00_original.png')
meas_out = os.path.join(base, 'measurements')

model_path = "local-dir:/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

model, preprocess = open_clip.create_model_from_pretrained(model_path)
tokenizer = open_clip.get_tokenizer(model_path)

# Prompts for foot/ankle findings
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

img = Image.open(img_path).convert('RGB')
img_input = preprocess(img).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(img_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

    text_inputs = tokenizer(all_prompts)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    logits = (image_features @ text_features.T).squeeze(0)
    probs = torch.softmax(logits, dim=0)

probs_list = probs.tolist()

results = {
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
    json.dump(results, f, indent=2)
