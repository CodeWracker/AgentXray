import os

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(BASE), "ID001-xray.png")

img = cv2.imread(SRC, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"source: {SRC}  size={w}x{h}  dtype={gray.dtype}  min={gray.min()} max={gray.max()}")

# 01 original (saved as-is, BGR->RGB for consistent viewing)
orig_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(BASE, "01_original.png"), cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR))

# 02 grayscale
cv2.imwrite(os.path.join(BASE, "02_grayscale.png"), gray)

# 03 contrast enhanced: CLAHE on L channel of LAB, mild gamma
lab = cv2.cvtColor(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
l_eq = clahe.apply(l)
eq = cv2.merge((l_eq, a, b))
contrast = cv2.cvtColor(eq, cv2.COLOR_LAB2GRAY)
gamma = 0.85
lut = ((np.arange(256, dtype=np.float32) / 255.0) ** gamma * 255).astype(np.uint8)
contrast = cv2.LUT(contrast, lut)
cv2.imwrite(os.path.join(BASE, "03_contrast_enhanced.png"), contrast)

# 04 sharpened: unsharp mask (Gaussian blur, preserve detail, no ringing)
blur = cv2.GaussianBlur(gray, (0, 0), 3)
sharpened = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
cv2.imwrite(os.path.join(BASE, "04_sharpened.png"), sharpened)

# 05 edge enhanced: Canny edges overlaid on lightly CLAHE'd base
edges = cv2.Canny(gray, 60, 140)
edges_d = cv2.dilate(edges, np.ones((2, 2), np.uint8))
base_mild = cv2.merge([gray] * 3)
base_mild = cv2.cvtColor(base_mild, cv2.COLOR_BGR2LAB)
lm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.split(base_mild)[0])
base_mild = cv2.merge((lm, cv2.split(base_mild)[1], cv2.split(base_mild)[2]))
base_mild = cv2.cvtColor(base_mild, cv2.COLOR_LAB2BGR)
overlay = base_mild.copy()
overlay[edges_d > 0] = (0.35 * overlay[edges_d > 0] + 0.65 * np.array([0, 0, 255], dtype=np.float32)).astype(np.uint8)
edge_enhanced = cv2.addWeighted(gray, 0, overlay, 1, 0)
edge_gray = cv2.cvtColor(edge_enhanced, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(BASE, "05_edge_enhanced.png"), edge_gray)

# 06 zoom region 1: ankle / talocalcaneal region
x0, y0, x1, y1 = int(w * 0.30), int(h * 0.52), int(w * 0.75), int(h * 0.95)
z1 = gray[y0:y1, x0:x1]
z1 = cv2.resize(z1, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
z1 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(z1)
cv2.imwrite(os.path.join(BASE, "06_zoom_region_1.png"), z1)

# 07 zoom region 2: calcaneus / heel with edge overlay for fracture line check
x0, y0, x1, y1 = int(w * 0.30), int(h * 0.60), int(w * 0.70), int(h * 0.95)
z2 = gray[y0:y1, x0:x1]
z2_big = cv2.resize(z2, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
z2_edges = cv2.Canny(z2_big, 50, 130)
z2_m = cv2.merge([z2_big, z2_big, z2_big])
z2_m[z2_edges > 0] = (0.3 * z2_m[z2_edges > 0] + 0.7 * np.array([255, 128, 0], dtype=np.float32)).astype(np.uint8)
cv2.imwrite(os.path.join(BASE, "07_zoom_region_2.png"), z2_m)

# 08 additional: bilateral histogram equalization via CLAHE + gentle unsharp for cortex detail
add = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(gray)
add_blur = cv2.GaussianBlur(add, (0, 0), 2.0)
add = cv2.addWeighted(add, 1.35, add_blur, -0.35, 0)
cv2.imwrite(os.path.join(BASE, "08_additional_enhancement.png"), add)

# verify all outputs open
out = os.path.join(BASE, "05_edge_enhanced.png")
for name in ["01_original.png", "02_grayscale.png", "03_contrast_enhanced.png",
             "04_sharpened.png", "05_edge_enhanced.png", "06_zoom_region_1.png",
             "07_zoom_region_2.png", "08_additional_enhancement.png"]:
    p = os.path.join(BASE, name)
    check = Image.open(p)
    check.verify()
    im2 = Image.open(p)
    print(f"OK {name} {im2.size} mode={im2.mode}")

print("done")
