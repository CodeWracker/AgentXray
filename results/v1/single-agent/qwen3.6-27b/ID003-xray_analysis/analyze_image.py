import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "ID003-xray.png")

img = cv2.imread(INPUT_PATH)
h, w = img.shape[:2]
print(f"Image loaded: {w}x{h}")

# 01 - Original (save as-is)
cv2.imwrite(os.path.join(SCRIPT_DIR, "01_original.png"), img)

# 02 - Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(SCRIPT_DIR, "02_grayscale.png"), gray)

# 03 - Contrast enhanced using CLAHE
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
enhanced = cv2.merge([cl, a, b])
contrast = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
cv2.imwrite(os.path.join(SCRIPT_DIR, "03_contrast_enhanced.png"), contrast)

# 04 - Sharpened using unsharp mask
kernel = np.array([[-1, -1, -1],
                   [-1,  9, -1],
                   [-1, -1, -1]])
sharpened = cv2.filter2D(gray, -1, kernel)
cv2.imwrite(os.path.join(SCRIPT_DIR, "04_sharpened.png"), sharpened)

# 05 - Edge enhanced using Canny + original overlay
edges = cv2.Canny(gray, 30, 90)
edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
edge_enhanced = cv2.addWeighted(img, 0.85, edge_color, 0.15, 0)
cv2.imwrite(os.path.join(SCRIPT_DIR, "05_edge_enhanced.png"), edge_enhanced)

# 06 - Zoom on forearm bones (mid-shaft of radius/ulna)
# The forearm is roughly in the upper-middle of the image
y1_f, y2_f = int(h * 0.15), int(h * 0.55)
x1_f, x2_f = int(w * 0.35), int(w * 0.7)
forearm_crop = gray[y1_f:y2_f, x1_f:x2_f]
forearm_zoom = cv2.resize(forearm_crop, (forearm_crop.shape[1]*3, forearm_crop.shape[0]*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(SCRIPT_DIR, "06_zoom_region_1.png"), forearm_zoom)

# 07 - Zoom on wrist/hand region
y1_w, y2_w = int(h * 0.5), int(h * 0.85)
x1_w, x2_w = int(w * 0.15), int(w * 0.65)
wrist_crop = gray[y1_w:y2_w, x1_w:x2_w]
wrist_zoom = cv2.resize(wrist_crop, (wrist_crop.shape[1]*3, wrist_crop.shape[0]*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(SCRIPT_DIR, "07_zoom_region_2.png"), wrist_zoom)

# 08 - Additional: adaptive histogram equalization for bone detail
ae = cv2.equalizeHist(gray)
ae_sharpen = cv2.filter2D(ae, -1, kernel)
cv2.imwrite(os.path.join(SCRIPT_DIR, "08_additional_enhancement.png"), ae_sharpen)

print("All images generated successfully.")
for f in sorted(os.listdir(SCRIPT_DIR)):
    if f.endswith('.png'):
        sz = os.path.getsize(os.path.join(SCRIPT_DIR, f))
        print(f"  {f} — {sz} bytes")
