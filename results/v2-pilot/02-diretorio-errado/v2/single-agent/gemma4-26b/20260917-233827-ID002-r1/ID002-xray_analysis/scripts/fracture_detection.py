import os
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# Set seeds for reproducibility
import random
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything()

def run_analysis():
    base_dir = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/results/v2/single-agent/gemma4-26b/20260917-233827-ID002-r1/ID002-xray_analysis"
    img_path = os.path.join(base_dir, "images/00_original.png")
    model_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/prithivMLmods/Bone-Fracture-Detection"
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = AutoModelForImageClassification.from_pretrained(model_path)
    model.eval()

    # Load image
    image = Image.open(img_path).convert("RGB")

    # Inference
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
    
    # Map predictions
    # Based on the prompt, labels are likely ['Fractured', 'Not Fractured'] or similar. 
    # The model class mapping should be checked from config if possible, but let's assume common order or check via model.config.id2label
    label = model.config.id2label[predicted_class_idx]
    confidence = torch.softmax(logits, dim=-1)[0, predicted_class_idx].item()

    # Save results
    # Save measurement
    import json
    meas_path = os.path.join(base_dir, "measurements/fracture_detection.json")
    with open(meas_path, 'w') as f:
        json.dump({"label": label, "confidence": confidence}, f)

    # Reproduce script logic (re-creating images if any were generated, but here we only have measurement)
    # For this simple task, we'll just ensure the script exists.
    print(f"Analysis completed: {label} ({confidence:.4f})")

if __name__ == "__main__":
    run_analysis()
