import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

base = Path(__file__).resolve().parent.parent
img_path = base / "images" / "00_original.png"
img = np.array(Image.open(img_path).convert("L"))

h, w = img.shape
out = base / "images"
out.mkdir(exist_ok=True)
meas = base / "measurements"
meas.mkdir(exist_ok=True)

# histogram stats
hist, _ = np.histogram(img, bins=256, range=(0, 256))
stats = {
    "shape": [h, w],
    "min": int(img.min()),
    "max": int(img.max()),
    "mean": float(img.mean()),
    "std": float(img.std()),
    "percentiles": {f"p{p}": float(np.percentile(img, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
    "histogram": {str(i): int(c) for i, c in enumerate(hist) if c > 0},
}
(meas / "01_histogram.json").write_text(json.dumps(stats, indent=2))

# (b) CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(img)
Image.fromarray(clahe_img).save(out / "01_enhanced_clahe.png")

# (c) contrast stretch 2-98 percentile
lo, hi = np.percentile(img, [2, 98])
stretched = np.clip((img.astype(np.int16) - lo) * 255 // max(1, (hi - lo)), 0, 255).astype(np.uint8)
Image.fromarray(stretched).save(out / "02_enhanced_stretch.png")

# (d) unsharp / high-pass
blur = cv2.GaussianBlur(img, (0, 0), 3)
unsharp = np.clip(img.astype(np.int16) + (img.astype(np.int16) - blur.astype(np.int16)) * 2, 0, 255).astype(np.uint8)
Image.fromarray(unsharp).save(out / "03_enhanced_unsharp.png")

# also a combined: stretch + CLAHE on the stretched image
st_clahe = clahe.apply(stretched)
Image.fromarray(st_clahe).save(out / "04_enhanced_stretch_clahe.png")

print("saved:", sorted(p.name for p in out.iterdir()))
print("lo,hi:", lo, hi)
