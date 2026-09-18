import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

INPUT = "/home/ralph/projects/ufsc/pablo-xray-tests/ID001-xray.png"
OUT = Path("/home/ralph/projects/ufsc/pablo-xray-tests/ID001-xray_analysis")

img = cv2.imread(INPUT, cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# 01 - Save original
cv2.imwrite(str(OUT / "01_original.png"), img)

# 02 - Grayscale
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()
cv2.imwrite(str(OUT / "02_grayscale.png"), gray)

# 03 - Contrast enhanced (CLAHE on inverted radiograph for better visibility)
gray_float = gray.astype(np.float32)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
clahe_result = clahe.apply(gray)
# Also do histogram equalization for comparison
eq_result = cv2.equalizeHist(gray)
# Blend CLAHE with original for best clinical result
contrast_enhanced = cv2.addWeighted(clahe_result, 0.7, gray, 0.3, 0)
cv2.imwrite(str(OUT / "03_contrast_enhanced.png"), contrast_enhanced)

# 04 - Sharpened
kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])
sharpened = cv2.filter2D(gray, -1, kernel)
# Blend sharpened with original to avoid over-sharpening artifacts
sharpened_blend = cv2.addWeighted(sharpened, 0.4, gray, 0.6, 0)
cv2.imwrite(str(OUT / "04_sharpened.png"), sharpened_blend)

# 05 - Edge enhanced (Canny + original blend)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 30, 120)
# Use edge map as overlay on contrast-enhanced image
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
contrast_color = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)
edge_enhanced = cv2.addWeighted(edges_colored, 0.15, contrast_color, 0.85, 0)
edge_enhanced_gray = cv2.cvtColor(edge_enhanced, cv2.COLOR_BGR2GRAY)
cv2.imwrite(str(OUT / "05_edge_enhanced.png"), edge_enhanced_gray)

# 06 - Zoom on ankle joint region
# The ankle joint (tibiotalar joint) is roughly in the middle of the image
y1, y2 = int(h * 0.35), int(h * 0.65)
x1, x2 = int(w * 0.35), int(w * 0.7)
zoom_ankle = gray[y1:y2, x1:x2]
zoom_ankle_resized = cv2.resize(zoom_ankle, (w // 2, int((y2-y1) * w // (2 * (x2-x1)))), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(str(OUT / "06_zoom_region_1.png"), zoom_ankle_resized)

# 07 - Zoom on calcaneus/heel region
y3, y4 = int(h * 0.6), int(h * 0.85)
x3, x4 = int(w * 0.15), int(w * 0.5)
zoom_calc = gray[y3:y4, x3:x4]
zoom_calc_resized = cv2.resize(zoom_calc, (w // 2, int((y4-y3) * w // (2 * (x4-x3)))), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite(str(OUT / "07_zoom_region_2.png"), zoom_calc_resized)

# 08 - Bone window enhancement for better trabecular detail
# Unsharp masking for bone detail
blur_large = cv2.GaussianBlur(gray, (0, 0), 15)
unsharp = cv2.addWeighted(gray, 1.5, blur_large, -0.5, 0)
# Apply CLAHE to the unsharp masked result
unsharp_clahe = clahe.apply(unsharp)
cv2.imwrite(str(OUT / "08_additional_enhancement.png"), unsharp_clahe)

print("All images generated successfully.")
for f in sorted(OUT.glob("*.png")):
    st = f.stat()
    print(f"  {f.name}: {st.st_size} bytes")
