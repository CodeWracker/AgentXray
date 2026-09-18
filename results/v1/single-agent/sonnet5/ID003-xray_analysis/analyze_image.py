"""
X-ray image processing pipeline for ID003-xray.png.
Generates intermediate visualizations for visual inspection.
Enhancements are for VISUALIZATION ONLY; the original pixel data is preserved
and all findings must be verified against the original image.
"""
import cv2
import numpy as np
import os

INPUT_PATH = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID003-xray.png"
OUT_DIR = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID003-xray_analysis"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Original
original = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
if original is None:
    raise FileNotFoundError(f"Could not load image at {INPUT_PATH}")
cv2.imwrite(os.path.join(OUT_DIR, "01_original.png"), original)

# 2. Grayscale
gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(OUT_DIR, "02_grayscale.png"), gray)

# 3. Contrast enhanced (CLAHE - Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast_enhanced = clahe.apply(gray)
cv2.imwrite(os.path.join(OUT_DIR, "03_contrast_enhanced.png"), contrast_enhanced)

# 4. Sharpened (unsharp mask on the contrast-enhanced image)
gaussian_blur = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=3)
sharpened = cv2.addWeighted(contrast_enhanced, 1.8, gaussian_blur, -0.8, 0)
cv2.imwrite(os.path.join(OUT_DIR, "04_sharpened.png"), sharpened)

# 5. Edge enhanced (Canny edges overlaid on grayscale for cortical outline visualization)
blurred_for_edges = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(blurred_for_edges, 40, 120)
edges_dilated = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
edge_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
edge_overlay[edges_dilated > 0] = [0, 0, 255]
cv2.imwrite(os.path.join(OUT_DIR, "05_edge_enhanced.png"), edge_overlay)

# 6. Zoom region 1: mid-forearm shaft (site of suspected lucent line on visual review)
h, w = gray.shape
# Forearm shaft region based on visual inspection of original (roughly x:55-63%, y:38-62%)
x1, x2 = int(w * 0.48), int(w * 0.62)
y1, y2 = int(h * 0.35), int(h * 0.65)
zoom1_src = contrast_enhanced[y1:y2, x1:x2]
zoom1 = cv2.resize(zoom1_src, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT_DIR, "06_zoom_region_1.png"), zoom1)

# 7. Zoom region 2: elbow joint
x1b, x2b = int(w * 0.45), int(w * 0.75)
y1b, y2b = int(h * 0.12), int(h * 0.35)
zoom2_src = contrast_enhanced[y1b:y2b, x1b:x2b]
zoom2 = cv2.resize(zoom2_src, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT_DIR, "07_zoom_region_2.png"), zoom2)

# 8. Additional enhancement: zoom on hand region (assess motion blur / digit overlap)
x1c, x2c = int(w * 0.25), int(w * 0.68)
y1c, y2c = int(h * 0.68), int(h * 1.0)
zoom3_src = contrast_enhanced[y1c:y2c, x1c:x2c]
zoom3 = cv2.resize(zoom3_src, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(OUT_DIR, "08_additional_enhancement.png"), zoom3)

print("All intermediate images generated successfully.")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith(".png"):
        path = os.path.join(OUT_DIR, f)
        print(f"  {f}: {os.path.getsize(path)} bytes")
