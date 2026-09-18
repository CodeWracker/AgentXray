import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import json
import os

# Configuration
MODEL_PATH = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/prithivMLmods/Bone-Fracture-Detection"
IMAGE_PATH = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/results/v2/single-agent/gemma4-26b/20260917-233336-ID001-r1/ID001-xray_analysis/images/00_original.png"
OUTPUT_JSON = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/results/v2/single-agent/gemma4-26b/20260917-233336-ID001-r1/ID001-xray_analysis/measurements/fracture_detection.json"
OUTPUT_DIR = os.path.dirname(OUTPUT_JSON)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_analysis():
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
    
    # Map index to label (assuming standard classification)
    # Based on the prompt description, labels are 'Fractured' and 'Not Fractured'
    labels = model.config.id2label
    prediction = labels[predicted_class_idx]
    confidence = probabilities[0][predicted_class_idx].item()

    results = {
        "prediction": prediction,
        "confidence": confidence,
        "all_probabilities": {labels[i]: probabilities[0][i].item() for i in range(len(labels))}
    }

    print(f"Result: {prediction} ({confidence:.4f})")
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_analysis()
