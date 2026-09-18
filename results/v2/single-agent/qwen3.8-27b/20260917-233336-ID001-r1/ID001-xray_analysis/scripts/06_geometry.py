import json
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

base = Path(__file__).resolve().parent.parent
meas = base / "measurements"
meas.mkdir(exist_ok=True)
img = np.array(Image.open(base / "images" / "00_original.png").convert("L"))
H, W = img.shape

# Bone mask: bright structures. Use Otsu on the image.
_, bone = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# keep only the largest connected component (the limb) to drop background specks
n, labels, stats, cents = cv2.connectedComponentsWithStats(bone, 8)
largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
limb = (labels == largest).astype(np.uint8) * 255

cv2.imwrite(str(base / "images" / "06_bone_mask.png"), limb)

def region_mask(x0, y0, x1, y1):
    m = np.zeros_like(limb)
    m[y0:y1, x0:x1] = limb[y0:y1, x0:x1]
    return m

def pca_angle(m):
    ys, xs = np.nonzero(m > 0)
    if len(xs) < 20:
        return None, None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, np.argmax(evals)]  # (dx, dy)
    ang = float(np.degrees(np.arctan2(major[1], major[0])))
    return ang, mean

# Approximate component boxes (from visual inspection of this specific image).
regions = {
    "tibia":      (150, 20, 260, 250),   # distal tibia shaft
    "fibula":     (110, 40, 175, 250),   # distal fibula
    "talus":      (140, 255, 235, 330),  # talus body/head
    "calcaneus":  (95, 300, 200, 405),   # calcaneus
    "metatarsals":(200, 300, 330, 380),  # metatarsal shafts/head
}
out = {"otsu_threshold": int(cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]),
       "limb_area_px": int((limb>0).sum()),
       "components": {}}
for name, (x0,y0,x1,y1) in regions.items():
    m = region_mask(x0,y0,x1,y1)
    ang, mean = pca_angle(m)
    area = int((m>0).sum())
    out["components"][name] = {
        "box": [x0,y0,x1,y1],
        "area_px": area,
        "principal_axis_angle_deg": (round(ang,2) if ang is not None else None),
        "centroid": ([round(mean[0],1), round(mean[1],1)] if mean is not None else None),
    }

# Coarse congruence screen:
#  - tibia vs talus centroid horizontal offset (talus should sit under the tibia)
tib = out["components"]["tibia"]["centroid"]; tal = out["components"]["talus"]["centroid"]
if tib and tal:
    out["congruence"] = {
        "tibia_centroid": tib, "talus_centroid": tal,
        "dx_talus_minus_tibia_px": round(tal[0]-tib[0],1),
        "note": "small |dx| => talus under tibial plafond; large |dx| => possible talar displacement. Coarse only (oblique plantar-flexed pose).",
    }
(meas / "06_geometry.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
