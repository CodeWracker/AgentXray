#!/usr/bin/env python3
"""Image processing pipeline for ID001-xray.png second-opinion analysis."""
import cv2
import numpy as np
import os

SRC = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID001-xray.png"
OUT = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID001-xray_analysis"

os.makedirs(OUT, exist_ok=True)

# 1. Load original
original = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)
if original is None:
    raise FileNotFoundError(f"Could not load {SRC}")
cv2.imwrite(os.path.join(OUT, "01_original.png"), original)

# 2. Grayscale
if len(original.shape) == 3:
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
else:
    gray = original.copy()
cv2.imwrite(os.path.join(OUT, "02_grayscale.png"), gray)

# 3. Contrast enhancement (CLAHE - adaptive histogram equalization)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(gray)
cv2.imwrite(os.path.join(OUT, "03_contrast_enhanced.png"), contrast_enhanced)

# 4. Sharpened (unsharp mask on contrast-enhanced image)
blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=3)
sharpened = cv2.addWeighted(contrast_enhanced, 1.5, blurred, -0.5, 0)
cv2.imwrite(os.path.join(OUT, "04_sharpened.png"), sharpened)

# 5. Edge enhanced (Canny edges overlaid on contrast-enhanced grayscale for bone contour visualization)
edges = cv2.Canny(contrast_enhanced, 40, 120)
edge_overlay = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)
edge_overlay[edges > 0] = [0, 255, 0]
cv2.imwrite(os.path.join(OUT, "05_edge_enhanced.png"), edge_overlay)

# 6. Zoom region 1: ankle joint / distal tibia-fibula-talus mortise (upper-middle area)
h, w = gray.shape
zoom1 = contrast_enhanced[int(h*0.35):int(h*0.68), int(w*0.15):int(w*0.85)]
zoom1_big = cv2.resize(zoom1, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT, "06_zoom_region_1_ankle_joint.png"), zoom1_big)

# 7. Zoom region 2: midfoot/forefoot (tarsals and metatarsals, lower-right area)
zoom2 = contrast_enhanced[int(h*0.55):int(h*0.98), int(w*0.35):int(w*1.0)]
zoom2_big = cv2.resize(zoom2, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT, "07_zoom_region_2_midfoot_forefoot.png"), zoom2_big)

# 8. Additional enhancement: bone-density-focused local normalization (helps show cortical continuity)
gray_f = gray.astype(np.float32)
local_mean = cv2.GaussianBlur(gray_f, (0, 0), sigmaX=25)
local_norm = (gray_f - local_mean)
local_norm = cv2.normalize(local_norm, None, 0, 255, cv2.NORM_MINMAX)
local_norm = local_norm.astype(np.uint8)
local_norm_clahe = clahe.apply(local_norm)
cv2.imwrite(os.path.join(OUT, "08_additional_enhancement_local_contrast.png"), local_norm_clahe)

print("All intermediate images generated successfully in:", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print(" -", f)
