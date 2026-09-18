import os
import cv2
import numpy as np

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests"
OUT = os.path.join(BASE, "ID003-xray_analysis")
SRC = os.path.join(BASE, "ID003-xray.png")
os.makedirs(OUT, exist_ok=True)

img = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)
if img is None:
    raise SystemExit("failed to load image")
print("loaded:", img.shape, img.dtype)

if img.ndim == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img

def save(name, arr):
    path = os.path.join(OUT, name)
    ok = cv2.imwrite(path, arr)
    print("saved" if ok else "FAILED", name, arr.shape, arr.dtype)
    return path

# 01 original (as captured, unchanged)
save("01_original.png", img)

# 02 grayscale
save("02_grayscale.png", gray)

# 03 contrast enhanced: CLAHE to stretch local contrast in a very dark image
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
contrast = clahe.apply(gray)
save("03_contrast_enhanced.png", contrast)

# 04 sharpened: unsharp mask on the contrast-enhanced image
blur = cv2.GaussianBlur(contrast, (0, 0), sigmaX=2.0)
sharpened = cv2.addWeighted(contrast, 1.6, blur, -0.6, 0)
save("04_sharpened.png", sharpened)

# 05 edge enhanced: Canny edges on smoothed, contrast-enhanced image
smooth = cv2.bilateralFilter(contrast, 9, 40, 40)
edges = cv2.Canny(smooth, 40, 120)
edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
save("05_edge_enhanced.png", edges)

# crop helper: (x, y, w, h) in source pixels, upscale factor
def zoom(name, x, y, w, h, scale=2):
    roi = gray[y : y + h, x : x + w]
    roi = cv2.resize(roi, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    roi = clahe.apply(roi)
    save(name, roi)

# 06 elbow region (humerus / elbow / proximal forearm)
zoom("06_zoom_region_1.png", 460, 160, 380, 340, scale=2)

# 07 wrist + clenched hand region
zoom("07_zoom_region_2.png", 330, 960, 520, 540, scale=2)

# 09 mid-forearm region (radius + ulna diaphyses) for subtle cortical assessment
zoom("09_zoom_midforearm.png", 550, 460, 180, 340, scale=3)

# 08 additional enhancement: gamma correction to lift soft tissues + CLAHE
gamma = 0.6
lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
bright = cv2.LUT(gray, lut)
bright = clahe.apply(bright)
save("08_additional_enhancement.png", bright)

print("done")
