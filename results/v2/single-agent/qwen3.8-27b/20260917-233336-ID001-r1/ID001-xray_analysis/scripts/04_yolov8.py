import json
from pathlib import Path
import torch
from ultralytics import YOLO

base = Path(__file__).resolve().parent.parent
meas = base / "measurements"
meas.mkdir(exist_ok=True)
meas.mkdir(exist_ok=True)
torch.manual_seed(0)

MODEL = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/files/grazpedwri-yolov8-best.pt"
p = base / "images" / "00_original.png"

model = YOLO(MODEL)
print("names:", model.names)
res = model.predict(str(p), imgsz=640, device="cpu", verbose=False)[0]

dets = []
if res.boxes is not None:
    for b in res.boxes:
        dets.append({
            "class": model.names[int(b.cls.item())],
            "conf": float(b.conf.item()),
            "xyxy": [float(v) for v in b.xyxy.tolist()],
        })

out = {
    "note": "trained on pediatric wrist trauma radiographs (GRAZPEDWRI-DX); foot is OOD",
    "detections": dets,
    "n_detections": len(dets),
}
(meas / "04_yolov8_grazpedwri.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
