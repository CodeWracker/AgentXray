import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import json
import os

# Configuration
IMG_PATH = "/home/ralph/projects/ufsc/pablo-xray-tests/bench/results/v2/single-agent/gemma4-26b/20260917-232029-ID002-r1/ID002-xray_analysis/images/00_original.png"
MODEL_PATH = "/home/ralph/.cache/xray-bench/hf/prithivMLmods/Bone-Fracture-Detection"
OUTPUT_JSON = "/home/ralph/projects/ufsc/pablo-xray-tests/bench/results/v2/single-agent/gemma4-26b/20260917-232029-ID002-r1/ID002-xray_analysis/measurements/bone_fracture_detection.json"
OUTPUT_IMG = "/home/ralph/projects/ufsc/pablo-xray-tests/bench/results/v2/single-agent/gemma4-26b/20260917-232029-ID002-r1/ID002-xray_analysis/images/fracture_detection_result.png"

def run_analysis():
    print(f"Loading model from {MODEL_PATH}")
    processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageClassification.from_pretrained(MODEL_PATH)
    
    print(f"Loading image {IMG_PATH}")
    image = Image.open(IMG_PATH).convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class_idx = logits.argmax(-1).item()
    prediction = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=-1)[0][predicted_class_idx].item()
    
    print(f"Prediction: {prediction} (Confidence: {confidence:.4f})")
    
    # Save result
    result = {
        "prediction": prediction,
        "confidence": confidence
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    # Save image with prediction text (simple way)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    text = f"Prediction: {prediction} ({confidence:.2f})"
    draw.text((10, 10), text, fill="white")
    image.save(OUTPUT_IMG)
    print(f"Results saved to {OUTPUT_JSON} and {OUTPUT_IMG}")

if __name__ == "__main__":
    run_analysis()
