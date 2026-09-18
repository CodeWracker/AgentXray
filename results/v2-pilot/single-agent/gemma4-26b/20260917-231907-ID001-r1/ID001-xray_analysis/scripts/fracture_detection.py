import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import json
import os
import numpy as np

def run_analysis():
    image_path = "images/00_original.png"
    output_dir = "measurements"
    os.makedirs(output_dir, exist_ok=True)

    model_path = "/home/ralph/.cache/xray-bench/hf/prithivMLmods/Bone-Fracture-Detection"
    
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class_idx = logits.argmax(-1).item()
    
    labels = model.config.id2label
    result = {
        "predicted_label": labels[predicted_class_idx],
        "confidence": float(probabilities[0][predicted_class_idx].item()),
        "all_probabilities": {labels[i]: float(probabilities[0][i].item()) for i in range(len(labels))}
    }

    with open(os.path.join(output_dir, "fracture_detection.json"), "w") as f:
        json.dump(result, f, indent=4)

    print(f"Analysis complete. Results: {result}")

if __name__ == "__main__":
    run_analysis()
