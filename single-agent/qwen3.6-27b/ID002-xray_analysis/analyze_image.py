import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage

INPUT = "/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray.png"
OUT   = Path("/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray_analysis")

# ── Load image ──────────────────────────────────────────────
img = cv2.imread(INPUT, cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]
print(f"Loaded image: {w}x{h}, dtype={img.dtype}, channels={img.shape[2] if len(img.shape)==3 else 1}")
print(f"Min: {img.min()}, Max: {img.max()}, Mean: {img.mean():.1f}")

# ── 01: Save original ──────────────────────────────────────
# Convert to 8-bit for saving
if img.dtype != np.uint8:
    orig_8u = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
else:
    orig_8u = img.copy()
cv2.imwrite(str(OUT / "01_original.png"), orig_8u)

# ── 02: Grayscale ──────────────────────────────────────────
if len(img.shape) == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img if img.dtype == np.uint8 else orig_8u.copy()
cv2.imwrite(str(OUT / "02_grayscale.png"), gray)

# ── 03: Contrast enhancement (CLAHE) ──────────────────────
# X-rays benefit from CLAHE to bring out subtle contrast differences
# X-ray convention: bone is white (bright), soft tissue dark.
# Invert so that bones are bright in uint8 (they already should be)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_float = gray.astype(np.float32)
# Normalize to 0-1, apply CLAHE, then back to uint8
gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
clahe_img = clahe.apply(gray_norm)
cv2.imwrite(str(OUT / "03_contrast_enhanced.png"), clahe_img)

# ── 04: Sharpened ──────────────────────────────────────────
kernel = np.array([[-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]])
sharpened = cv2.filter2D(gray, cv2.CV_8U, kernel)
cv2.imwrite(str(OUT / "04_sharpened.png"), sharpened)

# ── 05: Edge enhanced (Sobel + original blend) ────────────
sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)
edges = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
# Blend edges with original
edge_blend = cv2.addWeighted(gray, 0.8, edges.astype(np.uint8), 0.2, 0)
cv2.imwrite(str(OUT / "05_edge_enhanced.png"), edge_blend)

# ── 06: Zoom region 1 - Elbow joint area ──────────────────
# The elbow joint is near the center-top of the visible anatomy
# Estimate region based on typical lateral elbow X-ray layout
# Joint is roughly top 30%, centered horizontally in the anatomy area
# From visual inspection: bones run from top-right to bottom-center
# Elbow joint is in the upper portion
# Let's find the bounding box of non-black pixels first
nonzero = np.where(gray > 30)
if len(nonzero[0]) > 0:
    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()
    print(f"Non-black bounding box: ({x_min}, {y_min}) to ({x_max}, {y_max})")

    # Elbow joint is in the upper region of the anatomy
    # Zoom on the joint area: upper third of the bounding box
    joint_y1 = int(y_min)
    joint_y2 = int(y_min + 0.35 * (y_max - y_min))
    joint_x1 = int(x_min + 0.1 * (x_max - x_min))
    joint_x2 = int(x_max - 0.1 * (x_max - x_min))

    # Clamp
    joint_y1 = max(0, joint_y1)
    joint_y2 = min(h, joint_y2)
    joint_x1 = max(0, joint_x1)
    joint_x2 = min(w, joint_x2)

    joint_crop = clahe_img[joint_y1:joint_y2, joint_x1:joint_x2]
    # Upscale for better visibility
    zoom1 = cv2.resize(joint_crop, (joint_crop.shape[1] * 3, joint_crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / "06_zoom_region_1.png"), zoom1)
    print(f"Region 1 (elbow joint): [{joint_y1}:{joint_y2}, {joint_x1}:{joint_x2}]")

# ── 07: Zoom region 2 - Distal forearm / wrist area ────────
if len(nonzero[0]) > 0:
    # Distal forearm is the lower part of the anatomy
    distal_y1 = int(y_max - 0.3 * (y_max - y_min))
    distal_y2 = int(y_max)
    distal_x1 = int(x_min + 0.15 * (x_max - x_min))
    distal_x2 = int(x_max - 0.05 * (x_max - x_min))

    distal_y1 = max(0, distal_y1)
    distal_y2 = min(h, distal_y2)
    distal_x1 = max(0, distal_x1)
    distal_x2 = min(w, distal_x2)

    distal_crop = clahe_img[distal_y1:distal_y2, distal_x1:distal_x2]
    zoom2 = cv2.resize(distal_crop, (distal_crop.shape[1] * 3, distal_crop.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(OUT / "07_zoom_region_2.png"), zoom2)
    print(f"Region 2 (distal forearm): [{distal_y1}:{distal_y2}, {distal_x1}:{distal_x2}]")

# ── 08: Additional - Histogram equalization + adaptive sharpening ──
# Apply mild unsharp mask for fine detail
gray_f = gray.astype(np.float32) / 255.0
blurred = cv2.GaussianBlur(gray_f, (0, 0), 2)
unsharp = gray_f + 0.5 * (gray_f - blurred)
unsharp = np.clip(unsharp, 0, 1)
# Then apply gamma correction to brighten dark areas slightly
gamma = 1.2
additional = (unsharp ** (1 / gamma) * 255).astype(np.uint8)
cv2.imwrite(str(OUT / "08_additional_enhancement.png"), additional)

# ── Summary ───────────────────────────────────────────────
print("\nGenerated files:")
for f in sorted(OUT.glob("*.png")):
    sz = f.stat().st_size
    print(f"  {f.name} ({sz} bytes)")
print("Done.")
