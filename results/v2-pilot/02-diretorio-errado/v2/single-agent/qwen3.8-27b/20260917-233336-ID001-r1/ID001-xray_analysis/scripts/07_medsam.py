import json
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import SamModel, SamProcessor

base = Path(__file__).resolve().parent.parent
meas = base / "measurements"
meas.mkdir(exist_ok=True)
P = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/models/hf/wanglab/medsam-vit-base"

torch.manual_seed(0)
np.random.seed(0)
torch.use_deterministic_algorithms(True)

model = SamModel.from_pretrained(P)
model.eval()
processor = SamProcessor.from_pretrained(P)

image = Image.open(base / "images" / "00_original.png").convert("RGB")
arr = np.array(image)
H, W = arr.shape[:2]

# boxes chosen from visual inspection of this specific image: [x0,y0,x1,y1]
boxes = {
    "talus":       [140, 255, 235, 330],
    "calcaneus":   [95, 300, 205, 405],
    "metatarsal1": [205, 305, 335, 385],
}
masks = []
for b in boxes.values():
    inputs = processor(images=image, input_boxes=[[[b[0], b[1]], [b[2], b[3]]]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    m = processor.post_process_masks(outputs.pred_masks, inputs["original_sizes"], inputs["resized_images"], mask_threshold=0.5)
    m = m[0]
    if isinstance(m, (list, np.ndarray)) and np.asarray(m).ndim == 3:
        m = np.asarray(m)[0]
    masks.append(np.asarray(m))

# overlay: red = mask
overlay = arr.copy()
stats = {}
for (name, b), m in zip(boxes.items(), masks):
    m = np.asarray(m).astype(bool)
    if m.shape != (H, W):
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((W, H))) > 127
    box_area = (b[2]-b[0])*(b[3]-b[1])
    frac = float(m.sum())/box_area
    overlay[m] = [255, 0, 0]
    stats[name] = {"box": b, "mask_px": int(m.sum()), "frac_of_box": round(frac,3)}

Image.fromarray(overlay).save(base / "images" / "07_medsam_overlay.png")
(meas / "07_medsam.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
