"""Output validation: every expected image exists, opens, has valid dimensions/format; original copy unmodified."""
import os
import sys
import hashlib
from PIL import Image
import numpy as np

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID001-xray_analysis"
IMG_DIR = os.path.join(BASE, "images")
EXPECTED = [
    "ID001-xray.png",
    "01_flatfield_corrected_full.png",
    "02_clahe_mild_full.png",
    "03_clahe_moderate_full.png",
    "04_roi_ankle_subtalar_clahe.png",
    "05_roi_talus_clahe.png",
    "06_roi_talus_denoised_mild.png",
    "07_roi_talus_denoised_strong.png",
    "08_roi_posterior_calcaneus_clahe.png",
    "09_edge_hindfoot_unsharp.png",
    "10_edge_hindfoot_gradient.png",
    "11_edge_overlay_full.png",
    "12_angles_overlay.png",
    "13_softtissue_contour.png",
    "14_right_edge_crop_flatfield.png",
    "15_noise_floor_chart.png",
    "16_roi_midfoot_forefoot_clahe.png",
]

def md5(p):
    with open(p, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

lines = []
ok = True
for name in EXPECTED:
    p = os.path.join(IMG_DIR, name)
    if not os.path.exists(p):
        lines.append(f"MISSING  {name}")
        ok = False
        continue
    try:
        im = Image.open(p)
        im.verify()
        im = Image.open(p)
        a = np.array(im)
        lines.append(f"OK       {name:40s} {im.format} {im.size[0]}x{im.size[1]} {im.mode} min={a.min()} max={a.max()}")
    except Exception as e:
        lines.append(f"CORRUPT  {name}: {e}")
        ok = False

orig = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID001-xray.png"
same = md5(orig) == md5(os.path.join(IMG_DIR, "ID001-xray.png"))
lines.append(f"original copy unmodified: {same}")
ok = ok and same
lines.append(f"VALIDATION: {'PASS' if ok else 'FAIL'}")
print("\n".join(lines))
with open(os.path.join(BASE, "planning", "validation_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
