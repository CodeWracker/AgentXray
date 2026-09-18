import cv2
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import exposure, filters, restoration, feature

WORK_DIR = Path(__file__).parent.parent
OUT_DIR = Path(__file__).parent
IMG_PATH = WORK_DIR / "ID001-xray.png"

# Load image
img = cv2.imread(str(IMG_PATH), cv2.IMREAD_UNCHANGED)
print(f"Loaded image: {img.shape}, dtype={img.dtype}")

# 01 - Original (save as-is)
cv2.imwrite(str(OUT_DIR / "01_original.png"), img)
print("Saved 01_original.png")

# 02 - Grayscale
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()
cv2.imwrite(str(OUT_DIR / "02_grayscale.png"), gray)
print("Saved 02_grayscale.png")

# 03 - Contrast enhanced (CLAHE on grayscale)
# For X-ray, convert to uint8 if needed
if gray.dtype == np.uint16:
    gray8 = ((gray - gray.min()) / (gray.max() - gray.min()) * 255).astype(np.uint8)
else:
    gray8 = gray.copy()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
contrast = clahe.apply(gray8)
cv2.imwrite(str(OUT_DIR / "03_contrast_enhanced.png"), contrast)
print("Saved 03_contrast_enhanced.png")

# 04 - Sharpened (unsharp masking)
blur = cv2.GaussianBlur(contrast, (0, 0), 2)
sharpened = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT_DIR / "04_sharpened.png"), sharpened)
print("Saved 04_sharpened.png")

# 05 - Edge enhanced (Sobel + original)
sobelx = cv2.Sobel(contrast, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(contrast, cv2.CV_64F, 0, 1, ksize=3)
edges = np.hypot(sobelx, sobely).astype(np.float32)
edges = (edges / edges.max() * 255).astype(np.uint8)
# Combine edges with original for edge-enhanced view
edge_enhanced = cv2.addWeighted(contrast, 0.7, edges, 0.3, 0)
edge_enhanced = np.clip(edge_enhanced, 0, 255).astype(np.uint8)
cv2.imwrite(str(OUT_DIR / "05_edge_enhanced.png"), edge_enhanced)
print("Saved 05_edge_enhanced.png")

# 06 - Zoom on ankle joint (distal tibia/fibula + talus)
h, w = gray8.shape
# Ankle joint region - approximate location from visual inspection
# In the lateral view, the ankle is in the upper portion of the foot
y1, y2 = int(h * 0.25), int(h * 0.55)
x1, x2 = int(w * 0.35), int(w * 0.65)
zoom1 = contrast[y1:y2, x1:x2]
zoom1_large = cv2.resize(zoom1, (zoom1.shape[1]*3, zoom1.shape[0]*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(OUT_DIR / "06_zoom_region_1.png"), zoom1_large)
print(f"Saved 06_zoom_region_1.png (ankle joint region: [{y1}:{y2}, {x1}:{x2}])")

# 07 - Zoom on midfoot/hindfoot (calcaneus, talus, midfoot)
y1b, y2b = int(h * 0.45), int(h * 0.78)
x1b, x2b = int(w * 0.25), int(w * 0.60)
zoom2 = contrast[y1b:y2b, x1b:x2b]
zoom2_large = cv2.resize(zoom2, (zoom2.shape[1]*3, zoom2.shape[0]*3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(str(OUT_DIR / "07_zoom_region_2.png"), zoom2_large)
print(f"Saved 07_zoom_region_2.png (hindfoot/midfoot region: [{y1b}:{y2b}, {x1b}:{x2b}])")

# 08 - Adaptive histogram equalization with different parameters for bone detail
# Also try a Wiener deconvolution for slight noise reduction before enhancement
gaussian = cv2.GaussianBlur(contrast, (5, 5), 1.4)
deconvolved = restoration.wiener(gaussian.astype(np.float64) / 255.0, disk_size=3)
deconvolved = (deconvolved / deconvolved.max() * 255).astype(np.uint8)
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
additional = clahe2.apply(deconvolved)
cv2.imwrite(str(OUT_DIR / "08_additional_enhancement.png"), additional)
print("Saved 08_additional_enhancement.png")

print("\nAll images generated successfully.")
