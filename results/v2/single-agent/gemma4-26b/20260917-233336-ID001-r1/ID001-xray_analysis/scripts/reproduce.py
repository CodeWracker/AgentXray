import os
import json
import torch
import numpy as np
import random
from PIL import Image

# Paths relative to the script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH = os.path.join(ANALYSIS_DIR, "images", "00_original.png")
OUTPUT_JSON = os.path.join(ANALYSIS_DIR, "measurements", "fracture_detection.json")
MODEL_PATH = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/prithivMLmods/Bone-Fracture-Detection"

def run_analysis():
    # Determinism
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    from transformers import AutoModelForImageClassification, AutoImageProcessor

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

    print(f"Loading image {IMAGE_PATH}...")
    image = Image.open(IMAGE_PATH).convert("RGB")

    print("Performing inference...")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_idx = logits.argmax(-1).item()
    
    labels = model.config.id2label
    prediction = labels[predicted_class_idx]
    confidence = probabilities[0][predicted_class_idx].item()

    results = {
        "prediction": prediction,
        "confidence": confidence,
        "all_probabilities": {labels[i]: probabilities[0][i].item() for i in range(len(labels))}
    }

    print(f"Result: {prediction} ({confidence:.4f})")
    
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_analysis()
