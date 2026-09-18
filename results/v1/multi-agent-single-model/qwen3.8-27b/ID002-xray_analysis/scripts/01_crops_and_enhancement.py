import sys
import numpy as np
sys.path.insert(0, "/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray_analysis/scripts")
from common import load_gray, save_gray, save_rgb, crop, upscale_nearest, clahe, window_stretch

arr = load_gray()

ELBOW = (820, 100, 1250, 520)
FOCUS = (840, 180, 1060, 440)
WRIST = (650, 950, 1150, 1280)
SHAFT = (700, 500, 1150, 1000)

# A1: elbow ROI native, 2x
e = crop(arr, ELBOW)
save_gray(upscale_nearest(e, 2), "01_elbow_crop_native.png", "elbow native 2x")

# A2: elbow ROI CLAHE clip 2.5, 2x
save_gray(upscale_nearest(clahe(e, 2.5, 8), 2), "02_elbow_crop_clahe.png", "elbow CLAHE 2.5 2x")

# A3: olecranon focus native 4x
f = crop(arr, FOCUS)
save_gray(upscale_nearest(f, 4), "03_olecranon_focus_native.png", "focus native 4x")

# A4: olecranon focus CLAHE clip 3.0, 4x
save_gray(upscale_nearest(clahe(f, 3.0, 8), 4), "04_olecranon_focus_clahe.png", "focus CLAHE 3.0 4x")

# A5: elbow soft-tissue window (stretch [10,140]: bone saturates white, soft-tissue gradation preserved), 2x
save_gray(upscale_nearest(window_stretch(e, 10, 140), 2), "05_elbow_softtissue_window.png", "elbow soft-tissue window [10,140] 2x")

# A6: edge-enhanced bone window (guide only)
import cv2
eb = crop(arr, ELBOW)
bone = window_stretch(eb, 40, 200)
g = np.clip(np.round(bone), 0, 255).astype(np.uint8)
blurred = cv2.GaussianBlur(g, (5, 5), 0)
edges = cv2.Canny(blurred, 40, 120)
unsharp = cv2.addWeighted(g, 1.6, cv2.GaussianBlur(g, (5, 5), 1.5), -0.6, 0)
overlay = np.stack([unsharp, unsharp, unsharp], axis=-1).astype(np.float64)
emask = edges > 0
overlay[emask] = overlay[emask] * 0.65 + np.array([255, 170, 0]) * 0.35
# upscale before color blend so edge lines are not pixelated
save_rgb(upscale_nearest(overlay, 2), "06_elbow_edge_enhanced.png", "elbow bone window + unsharp + Canny 2x (guide only)")

# A9: wrist native 2x
w = crop(arr, WRIST)
save_gray(upscale_nearest(w, 2), "09_wrist_crop_native.png", "wrist native 2x")

# A10: wrist CLAHE 2.5, 2x
save_gray(upscale_nearest(clahe(w, 2.5, 8), 2), "10_wrist_crop_clahe.png", "wrist CLAHE 2.5 2x")

# A11: mid-forearm shaft bone window [40,200], 2x
s = crop(arr, SHAFT)
save_gray(upscale_nearest(window_stretch(s, 40, 200), 2), "11_forearm_shaft_window.png", "shaft bone window 2x")

print("done")
