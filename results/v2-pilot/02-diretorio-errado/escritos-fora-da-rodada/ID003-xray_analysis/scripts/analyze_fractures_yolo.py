import cv2
import json
import os
import torch
from ultralytics import YOLO

def run_analysis():
    model_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/files/grazpedwri-yolov8-best.pt"
    image_path = "images/00_original.png"
    output_dir = "images"
    measurements_dir = "measurements"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(measurements_dir, exist_ok=True)

    model = YOLO(model_path)
    results = model.predict(image_path, imgsz=640, device="cpu")

    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            coords = box.xyxy[0].tolist()
            detections.append({
                "label": label,
                "confidence": conf,
                "box": coords
            })
            
            # Draw on image
            img = r.orig_img
            cv2.rectangle(img, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (int(coords[0]), int(coords[1]-10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save visual output
    res_img = results[0].plot()
    cv2.imwrite(os.path.join(output_dir, "yolov8_detections.png"), res_img)

    # Save numeric output
    with open(os.path.join(measurements_dir, "yolov8_detections.json"), "w") as f:
        json.dump(detections, f, indent=4)

if __name__ == "__main__":
    run_analysis()
