import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import find_peaks
import os

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID002-xray_analysis"
IMG_DIR = os.path.join(BASE, "images")
os.makedirs(IMG_DIR, exist_ok=True)

orig_path = os.path.join(IMG_DIR, "ID002-xray.png")
img = np.array(Image.open(orig_path))
print(f"Original shape: {img.shape}, dtype: {img.dtype}")

if len(img.shape) == 3:
    gray = np.mean(img, axis=2).astype(np.uint8)
else:
    gray = img.copy()

print(f"Gray range: [{gray.min()}, {gray.max()}]")

# ============================================================
# STEP 1: CROP TO ANATOMICAL CONTENT (FIXED)
# ============================================================
print("\n=== STEP 1: CROP TO ANATOMICAL CONTENT ===")

col_means = gray.mean(axis=0)
row_means = gray.mean(axis=1)

active_cols = np.where(col_means > 20)[0]
active_rows = np.where(row_means > 20)[0]

cmin = int(active_cols[0])
cmax = int(active_cols[-1])
rmin = int(active_rows[0])
rmax = int(active_rows[-1])

margin = 30
rmin = max(0, rmin - margin)
rmax = min(gray.shape[0], rmax + margin)
cmin = max(0, cmin - margin)
cmax = min(gray.shape[1], cmax + margin)

cropped = gray[rmin:rmax, cmin:cmax].copy()
print(f"Crop bounds: rows [{rmin}, {rmax}], cols [{cmin}, {cmax}]")
print(f"Cropped shape: {cropped.shape}")

cv2.imwrite(os.path.join(IMG_DIR, "01_cropped_elbow.png"), cropped)
print("Saved: 01_cropped_elbow.png")

# ============================================================
# STEP 8: HISTOGRAM ANALYSIS
# ============================================================
print("\n=== STEP 8: HISTOGRAM ANALYSIS ===")
content_vals = cropped[cropped > 5].flatten()
full_hist, bin_edges = np.histogram(cropped.flatten(), bins=256, range=(0, 256))
content_hist, _ = np.histogram(content_vals, bins=256, range=(0, 256))

p2 = np.percentile(content_vals, 2)
p5 = np.percentile(content_vals, 5)
p95 = np.percentile(content_vals, 95)
p98 = np.percentile(content_vals, 98)

print(f"Content percentiles: p2={p2:.1f}, p5={p5:.1f}, p95={p95:.1f}, p98={p98:.1f}")
print(f"Content mean: {content_vals.mean():.1f}, std: {content_vals.std():.1f}")

bone_lo = 150
bone_hi = 254
soft_lo = int(p5)
soft_hi = 130

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
ax.bar(bin_edges[:-1], full_hist, width=1, color='steelblue', alpha=0.7, label='All pixels')
ax.bar(bin_edges[:-1], content_hist, width=1, color='coral', alpha=0.7, label='Content only (>5)')
ax.axvline(bone_lo, color='red', linestyle='--', alpha=0.7, label=f'Bone window [{bone_lo}, {bone_hi}]')
ax.axvline(bone_hi, color='red', linestyle='--', alpha=0.7)
ax.axvline(soft_lo, color='green', linestyle='--', alpha=0.7, label=f'Soft tissue [{soft_lo}, {soft_hi}]')
ax.axvline(soft_hi, color='green', linestyle='--', alpha=0.7)
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Count')
ax.set_title('Histogram: Cropped Image (Blue) vs Content Region (Red)')
ax.legend()
ax.set_xlim(0, 256)

ax = axes[1]
cum = np.cumsum(full_hist)
total = cum[-1]
ax.plot(bin_edges[:-1], cum / total * 100, 'steelblue')
ax.axhline(2, color='green', linestyle='--', alpha=0.5, label=f'p2 = {p2:.0f}')
ax.axhline(98, color='green', linestyle='--', alpha=0.5, label=f'p98 = {p98:.0f}')
ax.axvline(bone_lo, color='red', linestyle=':', alpha=0.5)
ax.axvline(bone_hi, color='red', linestyle=':', alpha=0.5)
ax.set_xlabel('Pixel Intensity')
ax.set_ylabel('Cumulative %')
ax.set_title('Cumulative Distribution')
ax.legend()
ax.set_xlim(0, 256)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "08_histogram_analysis.png"), dpi=150)
print("Saved: 08_histogram_analysis.png")
plt.close()

# ============================================================
# STEP 2: CLAHE CONTRAST ENHANCEMENT
# ============================================================
print("\n=== STEP 2: CLAHE CONTRAST ENHANCEMENT ===")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(cropped)
cv2.imwrite(os.path.join(IMG_DIR, "02_clahe_enhanced.png"), clahe_result)

linear_stretch = np.clip((cropped - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMG_DIR, "02_linear_stretch.png"), linear_stretch)
print("Saved: 02_clahe_enhanced.png, 02_linear_stretch.png")

# ============================================================
# STEP 3: DUAL WINDOW/LEVEL RENDERING
# ============================================================
print("\n=== STEP 3: DUAL WINDOW/LEVEL ===")
bone_win = np.clip((cropped - bone_lo) / (bone_hi - bone_lo) * 255, 0, 255).astype(np.uint8)
soft_win = np.clip((cropped - soft_lo) / (soft_hi - soft_lo) * 255, 0, 255).astype(np.uint8)
cv2.imwrite(os.path.join(IMG_DIR, "03_bone_window.png"), bone_win)
cv2.imwrite(os.path.join(IMG_DIR, "03_soft_tissue_window.png"), soft_win)
print(f"Bone window: [{bone_lo}, {bone_hi}]")
print(f"Soft tissue window: [{soft_lo}, {soft_hi}]")
print("Saved: 03_bone_window.png, 03_soft_tissue_window.png")

# ============================================================
# STEP 4: EDGE DETECTION
# ============================================================
print("\n=== STEP 4: EDGE DETECTION ===")
blurred = cv2.GaussianBlur(bone_win, (3, 3), 1.5)
edges = cv2.Canny(blurred, 20, 80)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

edge_overlay_rgb = np.stack([cropped, cropped, cropped], axis=2)
edge_mask_rgb = np.zeros_like(edge_overlay_rgb)
edge_mask_rgb[:, :, 0] = 255
combined = (edge_overlay_rgb.astype(float) * 0.6 + edge_mask_rgb.astype(float) * 0.4).astype(np.uint8)

plt.figure(figsize=(16, 5))

plt.subplot(1, 4, 1)
plt.imshow(cropped, cmap='gray')
plt.title('Cropped Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(bone_win, cmap='gray')
plt.title('Bone Window')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edges')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(combined)
plt.title('Edge Overlay (Red)')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "04_edge_detection.png"), dpi=150)
plt.close()
print("Saved: 04_edge_detection.png")

# ============================================================
# STEP 5: FAT PAD ASSESSMENT
# ============================================================
print("\n=== STEP 5: FAT PAD ASSESSMENT ===")
h, w = soft_win.shape

elbow_row_approx = int(h * 0.25)
elbow_col_approx = int(w * 0.5)

anterior_roi_y1 = max(0, elbow_row_approx - 30)
anterior_roi_y2 = elbow_row_approx + 80
anterior_roi_x1 = max(0, elbow_col_approx - 100)
anterior_roi_x2 = min(w, elbow_col_approx + 30)

posterior_roi_y1 = max(0, elbow_row_approx - 20)
posterior_roi_y2 = elbow_row_approx + 80
posterior_roi_x1 = max(0, elbow_col_approx - 20)
posterior_roi_x2 = min(w, elbow_col_approx + 100)

anterior_soft = soft_win[anterior_roi_y1:anterior_roi_y2, anterior_roi_x1:anterior_roi_x2]
posterior_soft = soft_win[posterior_roi_y1:posterior_roi_y2, posterior_roi_x1:posterior_roi_x2]

clahe_small = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
anterior_enhanced = clahe_small.apply(anterior_soft)
posterior_enhanced = clahe_small.apply(posterior_soft)

fat_pad_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
fat_pad_enhanced_full = fat_pad_clahe.apply(soft_win)

plt.figure(figsize=(16, 4))

plt.subplot(1, 5, 1)
plt.imshow(soft_win, cmap='gray')
plt.gca().add_patch(plt.Rectangle((anterior_roi_x1, anterior_roi_y1),
    anterior_roi_x2 - anterior_roi_x1, anterior_roi_y2 - anterior_roi_y1,
    fill=False, edgecolor='lime', linewidth=2))
plt.gca().add_patch(plt.Rectangle((posterior_roi_x1, posterior_roi_y1),
    posterior_roi_x2 - posterior_roi_x1, posterior_roi_y2 - posterior_roi_y1,
    fill=False, edgecolor='cyan', linewidth=2))
plt.title('Fat Pad ROIs on Soft-Tissue Window')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(soft_win, cmap='gray')
plt.title('Soft Tissue Window (Full)')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(fat_pad_enhanced_full, cmap='gray')
plt.title('Fat Pad Region (CLAHE Full)')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(anterior_enhanced, cmap='gray')
plt.title(f'Anterior Fat Pad ROI\nmean={anterior_soft.mean():.1f}')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(posterior_enhanced, cmap='gray')
plt.title(f'Posterior Fat Pad ROI\nmean={posterior_soft.mean():.1f}')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "05_fat_pad_enhanced.png"), dpi=150)
plt.close()
print("Saved: 05_fat_pad_enhanced.png")

# ============================================================
# STEP 6: RADIAL HEAD DETAILED ANALYSIS
# ============================================================
print("\n=== STEP 6: RADIAL HEAD ANALYSIS ===")
radial_row_y1 = max(0, elbow_row_approx - 40)
radial_row_y2 = min(h, elbow_row_approx + 100)
radial_col_x1 = max(0, elbow_col_approx - 80)
radial_col_x2 = min(w, elbow_col_approx + 100)

radial_roi = cropped[radial_row_y1:radial_row_y2, radial_col_x1:radial_col_x2].copy()
radial_roi_bone = bone_win[radial_row_y1:radial_row_y2, radial_col_x1:radial_col_x2].copy()

sharpened = cv2.GaussianBlur(radial_roi, (0, 0), 1)
sharpened = cv2.addWeighted(radial_roi, 1.5, sharpened, -0.5, 0)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

radial_edges = cv2.Canny(cv2.GaussianBlur(radial_roi_bone, (3, 3), 1), 30, 100)

plt.figure(figsize=(16, 4))

plt.subplot(1, 5, 1)
plt.imshow(cropped, cmap='gray')
plt.gca().add_patch(plt.Rectangle((radial_col_x1, radial_row_y1),
    radial_col_x2 - radial_col_x1, radial_row_y2 - radial_row_y1,
    fill=False, edgecolor='yellow', linewidth=2))
plt.title('Radial Head ROI Location')
plt.axis('off')

plt.subplot(1, 5, 2)
plt.imshow(radial_roi, cmap='gray')
plt.title('Radial Head ROI (Original)')
plt.axis('off')

plt.subplot(1, 5, 3)
plt.imshow(sharpened, cmap='gray')
plt.title('Radial Head (Sharpened)')
plt.axis('off')

plt.subplot(1, 5, 4)
plt.imshow(radial_roi_bone, cmap='gray')
plt.title('Radial Head (Bone Window)')
plt.axis('off')

plt.subplot(1, 5, 5)
plt.imshow(radial_edges, cmap='gray')
plt.title('Radial Head Edges')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "06_radial_head_roi.png"), dpi=150)
plt.close()
print("Saved: 06_radial_head_roi.png")

# ============================================================
# STEP 7: LINE PROFILE ANALYSIS
# ============================================================
print("\n=== STEP 7: LINE PROFILE ANALYSIS ===")

profile_rows = [
    (int(h * 0.15), 0, w, "Distal humerus / elbow joint region"),
    (int(h * 0.30), 0, w, "Proximal forearm / radial head region"),
    (int(h * 0.50), 0, w, "Mid-forearm shaft region"),
    (int(h * 0.75), 0, w, "Distal forearm region"),
]

profiles = []
for row_idx, (row, x1, x2, label) in enumerate(profile_rows):
    if row < h:
        profile = cropped[row, x1:x2]
        profiles.append((profile, label, row))

fig, axes = plt.subplots(len(profiles) + 1, 1, figsize=(14, 10),
                          gridspec_kw={'height_ratios': [3] + [1] * len(profiles)})

axes[0].imshow(cropped, cmap='gray')
for i, (profile, label, row) in enumerate(profiles):
    color = ['red', 'lime', 'cyan', 'magenta'][i % 4]
    axes[0].axhline(row, color=color, linestyle='--', alpha=0.7, linewidth=1.5)
axes[0].set_title('Line Profile Transect Lines on Cropped Image')
axes[0].axis('off')

for i, (profile, label, row) in enumerate(profiles):
    ax = axes[i + 1]
    x = np.arange(len(profile))
    ax.plot(x, profile, 'steelblue', linewidth=0.8)

    smoothed = ndimage.gaussian_filter1d(profile, 3)
    deriv1 = np.gradient(smoothed)
    deriv2 = np.gradient(deriv1)

    mean_d2 = np.mean(np.abs(deriv2))
    std_d2 = np.std(np.abs(deriv2))
    anomalies = np.where(np.abs(deriv2) > mean_d2 + 2 * std_d2)[0]
    if len(anomalies) > 0:
        ax.plot(anomalies, profile[anomalies], 'ro', markersize=3, alpha=0.7, label=f'{len(anomalies)} derivative anomalies')

    ax.set_title(f'Row {row}: {label}')
    ax.set_xlabel('Column')
    ax.set_ylabel('Intensity')
    ax.legend()
    ax.set_xlim(0, w)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "07_line_profiles.png"), dpi=150)
plt.close()
print("Saved: 07_line_profiles.png")

# ============================================================
# SUPPLEMENTARY S1: GRADIENT MAGNITUDE MAP
# ============================================================
print("\n=== SUPPLEMENTARY S1: GRADIENT MAGNITUDE ===")
sobel_x = cv2.Sobel(cropped.astype(np.float32), cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(cropped.astype(np.float32), cv2.CV_64F, 0, 1, ksize=3)
grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
grad_mag_norm = np.clip(grad_mag / grad_mag.max() * 255, 0, 255).astype(np.uint8)

pct90 = np.percentile(grad_mag_norm, 90)
suppressed = grad_mag_norm.copy()
suppressed[grad_mag_norm > pct90] = 0
if np.any(suppressed > 0):
    suppressed = np.clip(suppressed / np.percentile(suppressed[suppressed > 0], 95) * 255, 0, 255).astype(np.uint8)

plt.figure(figsize=(14, 4))

plt.subplot(1, 4, 1)
plt.imshow(cropped, cmap='gray')
plt.title('Cropped Original')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(grad_mag_norm, cmap='inferno')
plt.title('Full Gradient Magnitude')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(suppressed, cmap='inferno')
plt.title('Gradient (Cortex Suppressed)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(bone_win, cmap='gray')
plt.title('Bone Window Reference')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "s1_gradient_magnitude.png"), dpi=150)
plt.close()
print("Saved: s1_gradient_magnitude.png")

# ============================================================
# VALIDATION
# ============================================================
print("\n=== VALIDATION ===")
output_files = [
    "01_cropped_elbow.png",
    "02_clahe_enhanced.png",
    "02_linear_stretch.png",
    "03_bone_window.png",
    "03_soft_tissue_window.png",
    "04_edge_detection.png",
    "05_fat_pad_enhanced.png",
    "06_radial_head_roi.png",
    "07_line_profiles.png",
    "08_histogram_analysis.png",
    "s1_gradient_magnitude.png",
]

all_ok = True
for f in output_files:
    path = os.path.join(IMG_DIR, f)
    if os.path.exists(path):
        sz = os.path.getsize(path)
        img_check = Image.open(path)
        print(f"OK: {f} ({img_check.size[0]}x{img_check.size[1]}, {sz} bytes)")
    else:
        print(f"MISSING: {f}")
        all_ok = False

print(f"\nAll outputs generated: {all_ok}")
print("\n=== ALL PROCESSING COMPLETE ===")
