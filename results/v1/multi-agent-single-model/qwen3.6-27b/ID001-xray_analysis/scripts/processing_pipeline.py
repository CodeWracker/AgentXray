import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage, stats
from scipy.signal import find_peaks
import os

INPUT = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images/ID001-xray_original.png"
OUTDIR = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images"
SCRIPTS = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/scripts"

img = np.array(Image.open(INPUT).convert("L"))
print(f"Image shape: {img.shape}, dtype: {img.dtype}, range: [{img.min()}, {img.max()}]")

# ============================================================
# ANALYSIS 1: Display Inversion
# ============================================================
inverted = 255 - img
Image.fromarray(inverted).save(os.path.join(OUTDIR, "01_inverted.png"))
print("01_inverted.png saved")

# ============================================================
# ANALYSIS 2: Global Contrast Stretch
# ============================================================
p1, p99 = np.percentile(img, [1, 99])
contrast_stretched = np.clip((img - p1) / (p99 - p1) * 255, 0, 255).astype(np.uint8)
Image.fromarray(contrast_stretched).save(os.path.join(OUTDIR, "02_contrast_stretched.png"))
print("02_contrast_stretched.png saved")

# ============================================================
# ANALYSIS 3: CLAHE - Full Image
# ============================================================
clahe_full = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_result = clahe_full.apply(img)
Image.fromarray(clahe_result).save(os.path.join(OUTDIR, "03_clahe_full.png"))
print("03_clahe_full.png saved")

# ============================================================
# ANALYSIS 4: CLAHE - Lateral Midfoot ROI (Cuboid Focus)
# ============================================================
h, w = img.shape
midfoot_x1 = int(w * 0.45)
midfoot_y1 = int(h * 0.45)
midfoot_x2 = int(w * 0.85)
midfoot_y2 = int(h * 0.85)
midfoot_roi = img[midfoot_y1:midfoot_y2, midfoot_x1:midfoot_x2]
clahe_midfoot = clahe_full.apply(midfoot_roi)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original Image")
axes[0].plot([midfoot_x1, midfoot_x2, midfoot_x2, midfoot_x1, midfoot_x1],
             [midfoot_y1, midfoot_y1, midfoot_y2, midfoot_y2, midfoot_y1],
             'r-', linewidth=2)
axes[1].imshow(midfoot_roi, cmap="gray")
axes[1].set_title("Midfoot ROI (Original)")
axes[2].imshow(clahe_midfoot, cmap="gray")
axes[2].set_title("Midfoot ROI (CLAHE Enhanced)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "04_clahe_midfoot_roi.png"), dpi=150, bbox_inches="tight")
plt.close()
print("04_clahe_midfoot_roi.png saved")

# ============================================================
# ANALYSIS 5: Unsharp Masking
# ============================================================
blur = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
unsharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
Image.fromarray(unsharp).save(os.path.join(OUTDIR, "05_unsharp_mask.png"))
print("05_unsharp_mask.png saved")

# ============================================================
# ANALYSIS 6: Böhler's Angle and Gissane's Angle
# ============================================================
# Use thresholding + connected components to find bone regions
bone_thresh = img > 80
bone_mask = (bone_thresh * 255).astype(np.uint8)
num_labels, labels, stats_rect, centroids = cv2.connectedComponentsWithStats(bone_mask, connectivity=8)

# Find bone components by area, looking for the calcaneus region (large, bottom-left quadrant)
bone_components = []
for i in range(1, num_labels):
    x, y, bw, bh, area = stats_rect[i]
    if area > 500:
        bone_components.append((i, x, y, bw, bh, area))

bone_components.sort(key=lambda k: k[5], reverse=True)
print(f"Found {len(bone_components)} bone components (area > 500px)")

# The calcaneus should be the large component in the lower-left portion of the image
# Based on the lateral foot anatomy: calcaneus is the heel bone at bottom-left
calcaneus_comp = None
for comp in bone_components:
    idx, x, y, bw, bh, area = comp
    cx_comp = x + bw/2
    cy_comp = y + bh/2
    if area > 2000 and cx_comp < w * 0.4 and cy_comp > h * 0.3:
        calcaneus_comp = comp
        break

# If the above didn't work, try finding the largest component in the lower-left quadrant
if calcaneus_comp is None:
    for comp in bone_components:
        idx, x, y, bw, bh, area = comp
        if area > 1000 and x < w * 0.5 and y > h * 0.3:
            calcaneus_comp = comp
            break

if calcaneus_comp:
    cidx, cx, cy, ccw, cch, carea = calcaneus_comp
    calcaneus_crop = img[cy:cy+cch, cx:cx+ccw]
    print(f"Calcaneus found at ({cx}, {cy}), size {ccw}x{cch}, area {carea}")

    coords = np.column_stack(np.where(calcaneus_crop > np.mean(calcaneus_crop[calcaneus_crop > 0]) * 0.5))
    if len(coords) > 0:
        superior_y = coords[:, 0].min()
        posterior_x = coords[:, 1].min()
        anterior_x = coords[:, 1].max()

        apex_col = int((posterior_x + anterior_x) / 2)
        if 0 <= apex_col < calcaneus_crop.shape[1]:
            col_vals = calcaneus_crop[:, apex_col]
            threshold = np.mean(col_vals[col_vals > 0]) * 0.5
            bone_rows = np.where(col_vals > threshold)[0]
            if len(bone_rows) > 0:
                apex_y = bone_rows[0]
            else:
                apex_y = superior_y + int(cch * 0.3)
        else:
            apex_y = superior_y + int(cch * 0.3)

        posterior_superior_y = superior_y
        posterior_superior_x = posterior_x
        anterior_superior_y = superior_y
        for x_offset in range(posterior_x + 10, anterior_x, 2):
            col = calcaneus_crop[:, x_offset]
            thresh = np.mean(col[col > 0]) * 0.5
            rows = np.where(col > thresh)[0]
            if len(rows) > 0:
                anterior_superior_y = rows[0]
                break

        p1_behler = np.array([posterior_superior_x, cy + posterior_superior_y])
        p2_behler = np.array([apex_col, cy + apex_y])
        p3_behler = np.array([anterior_x, cy + anterior_superior_y])

        v1 = p1_behler - p2_behler
        v2 = p3_behler - p2_behler
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        boehler_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

        anterior_facet_col = int(posterior_x + (anterior_x - posterior_x) * 0.4)
        if 0 <= anterior_facet_col < calcaneus_crop.shape[1]:
            col_vals2 = calcaneus_crop[:, anterior_facet_col]
            thresh2 = np.mean(col_vals2[col_vals2 > 0]) * 0.5
            bone_rows2 = np.where(col_vals2 > thresh2)[0]
            if len(bone_rows2) > 0:
                anterior_facet_y = bone_rows2[0]
            else:
                anterior_facet_y = superior_y + int(cch * 0.4)
        else:
            anterior_facet_y = superior_y + int(cch * 0.4)

        p1_gissane = np.array([anterior_facet_col, cy + anterior_facet_y])
        p2_gissane = np.array([anterior_x, cy + anterior_superior_y])

        mid_point = int(posterior_x + (anterior_x - posterior_x) * 0.3)
        col_vals3 = calcaneus_crop[:, mid_point]
        thresh3 = np.mean(col_vals3[col_vals3 > 0]) * 0.5
        bone_rows3 = np.where(col_vals3 > thresh3)[0]
        if len(bone_rows3) > 0:
            p3_gissane = np.array([mid_point, cy + bone_rows3[0]])
        else:
            p3_gissane = np.array([mid_point, cy + superior_y + int(cch * 0.3)])

        v3 = p3_gissane - p2_gissane
        v4 = p1_gissane - p2_gissane
        cos_gissane = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4) + 1e-10)
        gissane_angle = np.degrees(np.arccos(np.clip(cos_gissane, -1, 1)))

        print(f"Böhler's angle: {boehler_angle:.1f}° (normal: 20-40°)")
        print(f"Gissane's angle: {gissane_angle:.1f}° (normal: 120-145°)")
    else:
        print("Could not identify calcaneal landmarks")
        boehler_angle = None
        gissane_angle = None
else:
    print("Could not identify calcaneus component")
    boehler_angle = None
    gissane_angle = None
    calcaneus_crop = np.zeros((100, 100), dtype=np.uint8)

fig, axes = plt.subplots(1, 2, figsize=(14, 14))
axes[0].imshow(img, cmap="gray")
if calcaneus_comp:
    cidx, cx, cy, ccw, cch, carea = calcaneus_comp
    axes[0].set_title(f"Original - Calcaneus Region\ncenter=({cx+ccw//2}, {cy+cch//2}), size={ccw}x{cch}")
    axes[0].add_patch(plt.Rectangle((cx, cy), ccw, cch, fill=False, edgecolor='red', linewidth=2))
else:
    axes[0].set_title("Original - Calcaneus not auto-identified")
axes[1].imshow(calcaneus_crop, cmap="gray")
axes[1].set_title(f"Calcaneus Crop ({calcaneus_crop.shape[1]}x{calcaneus_crop.shape[0]})")
if boehler_angle is not None:
    axes[1].plot(posterior_superior_x, posterior_superior_y, 'go', markersize=10, label='Post. Sup. Angle')
    axes[1].plot(apex_col, apex_y, 'ro', markersize=10, label='Posterior Facet Apex')
    axes[1].plot(anterior_x, anterior_superior_y, 'bo', markersize=10, label='Ant. Sup. Angle')
    axes[1].plot([posterior_superior_x, apex_col], [posterior_superior_y, apex_y], 'g-', linewidth=2)
    axes[1].plot([apex_col, anterior_x], [apex_y, anterior_superior_y], 'b-', linewidth=2)
    axes[1].legend(loc='lower right')
    title_text = f"Böhler: {boehler_angle:.1f}°, Gissane: {gissane_angle:.1f}°" if gissane_angle else f"Böhler: {boehler_angle:.1f}°"
    axes[1].set_title(title_text)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "06_calcaneal_angles.png"), dpi=150, bbox_inches="tight")
plt.close()
print("06_calcaneal_angles.png saved")

# ============================================================
# ANALYSIS 7: Talar Dome Crop + Inverted
# ============================================================
talar_x1 = int(w * 0.35)
talar_y1 = int(h * 0.25)
talar_x2 = int(w * 0.6)
talar_y2 = int(h * 0.55)
talar_crop = img[talar_y1:talar_y2, talar_x1:talar_x2]
talar_upscaled = cv2.resize(talar_crop, (talar_crop.shape[1]*3, talar_crop.shape[0]*3), interpolation=cv2.INTER_NEAREST)
talar_inverted = 255 - talar_crop

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Talar Dome Location")
axes[0].add_patch(plt.Rectangle((talar_x1, talar_y1), talar_x2-talar_x1, talar_y2-talar_y1, fill=False, edgecolor='red', linewidth=2))
axes[1].imshow(talar_upscaled, cmap="gray")
axes[1].set_title(f"Talar Dome Crop (3x nearest-neighbor, {talar_crop.shape[1]}x{talar_crop.shape[0]} -> {talar_upscaled.shape[1]}x{talar_upscaled.shape[0]})")
axes[2].imshow(talar_inverted, cmap="gray")
axes[2].set_title("Talar Dome Crop (Inverted)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "07_talar_dome_crop.png"), dpi=150, bbox_inches="tight")
plt.close()
Image.fromarray(talar_inverted).save(os.path.join(OUTDIR, "07_talar_dome_inverted.png"))
print("07_talar_dome_crop.png and 07_talar_dome_inverted.png saved")

# ============================================================
# ANALYSIS 8: Anterior Calcaneal Process Crop
# ============================================================
ant_calc_x1 = int(w * 0.4)
ant_calc_y1 = int(h * 0.45)
ant_calc_x2 = int(w * 0.7)
ant_calc_y2 = int(h * 0.75)
ant_calc_crop = img[ant_calc_y1:ant_calc_y2, ant_calc_x1:ant_calc_x2]
ant_calc_upscaled = cv2.resize(ant_calc_crop, (ant_calc_crop.shape[1]*3, ant_calc_crop.shape[0]*3), interpolation=cv2.INTER_NEAREST)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Anterior Calcaneus Location")
axes[0].add_patch(plt.Rectangle((ant_calc_x1, ant_calc_y1), ant_calc_x2-ant_calc_x1, ant_calc_y2-ant_calc_y1, fill=False, edgecolor='red', linewidth=2))
axes[1].imshow(ant_calc_crop, cmap="gray")
axes[1].set_title(f"Anterior Calcaneus Crop ({ant_calc_crop.shape[1]}x{ant_calc_crop.shape[0]})")
axes[2].imshow(ant_calc_upscaled, cmap="gray")
axes[2].set_title("Anterior Calcaneus (3x Nearest-Neighbor Upscale)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "08_anterior_calcaneus_crop.png"), dpi=150, bbox_inches="tight")
plt.close()
print("08_anterior_calcaneus_crop.png saved")

# ============================================================
# ANALYSIS 9: Soft Tissue Thickness Profiling
# ============================================================
fg_mask = img > 15
dorsal_contour = []
plantar_contour = []
for x in range(w):
    col = fg_mask[:, x]
    fg_indices = np.where(col)[0]
    if len(fg_indices) > 10:
        dorsal_contour.append((x, fg_indices[0]))
        plantar_contour.append((x, fg_indices[-1]))

dorsal_contour = np.array(dorsal_contour)
plantar_contour = np.array(plantar_contour)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].imshow(img, cmap="gray")
if len(dorsal_contour) > 0:
    axes[0].plot(dorsal_contour[:, 0], dorsal_contour[:, 1], 'r-', linewidth=1.5, label='Dorsal contour')
    axes[0].plot(plantar_contour[:, 0], plantar_contour[:, 1], 'g-', linewidth=1.5, label='Plantar contour')

    if len(dorsal_contour) > 0:
        thickness = []
        for i in range(len(dorsal_contour)):
            t = plantar_contour[i, 1] - dorsal_contour[i, 1]
            thickness.append((dorsal_contour[i, 0], t))
        thickness = np.array(thickness)
        axes[0].axvline(x=w*0.2, color='yellow', linestyle='--', alpha=0.5, label='Ankle level')
        axes[0].axvline(x=w*0.5, color='cyan', linestyle='--', alpha=0.5, label='Midfoot level')
        axes[0].axvline(x=w*0.75, color='magenta', linestyle='--', alpha=0.5, label='Forefoot level')
        axes[0].legend(loc='upper right')

axes[0].set_title("Soft Tissue Contour Extraction (Original Image)")
axes[0].axis("off")

if len(thickness) > 0:
    axes[1].plot(thickness[:, 0], thickness[:, 1], 'b-', linewidth=1.5)
    axes[1].set_xlabel("Horizontal Position (pixels)")
    axes[1].set_ylabel("Soft Tissue Thickness (pixels)")
    axes[1].set_title("Soft Tissue Thickness Profile Along the Foot")
    axes[1].axvline(x=w*0.2, color='yellow', linestyle='--', alpha=0.5, label='Ankle level')
    axes[1].axvline(x=w*0.5, color='cyan', linestyle='--', alpha=0.5, label='Midfoot level')
    axes[1].axvline(x=w*0.75, color='magenta', linestyle='--', alpha=0.5, label='Forefoot level')

    ankle_idx = np.argmin(np.abs(thickness[:, 0] - w*0.2))
    midfoot_idx = np.argmin(np.abs(thickness[:, 0] - w*0.5))
    forefoot_idx = np.argmin(np.abs(thickness[:, 0] - w*0.75))
    axes[1].annotate(f"Ankle: {thickness[ankle_idx, 1]:.0f}px",
                    (thickness[ankle_idx, 0], thickness[ankle_idx, 1]),
                    textcoords="offset points", xytext=(10, 10))
    axes[1].annotate(f"Midfoot: {thickness[midfoot_idx, 1]:.0f}px",
                    (thickness[midfoot_idx, 0], thickness[midfoot_idx, 1]),
                    textcoords="offset points", xytext=(10, -20))
    axes[1].annotate(f"Forefoot: {thickness[forefoot_idx, 1]:.0f}px",
                    (thickness[forefoot_idx, 0], thickness[forefoot_idx, 1]),
                    textcoords="offset points", xytext=(10, -20))
    axes[1].legend(loc='upper right')
    print(f"Soft tissue thickness - Ankle: {thickness[ankle_idx, 1]:.0f}px, Midfoot: {thickness[midfoot_idx, 1]:.0f}px, Forefoot: {thickness[forefoot_idx, 1]:.0f}px")
else:
    axes[1].text(0.5, 0.5, 'Could not extract thickness profile', ha='center', va='center', transform=axes[1].transAxes)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "09_soft_tissue_profile.png"), dpi=150, bbox_inches="tight")
plt.close()
print("09_soft_tissue_profile.png saved")

# ============================================================
# ANALYSIS 10: Kager's Fat Pad Triangle Assessment
# ============================================================
kager_x1 = int(w * 0.2)
kager_y1 = int(h * 0.35)
kager_x2 = int(w * 0.4)
kager_y2 = int(h * 0.6)
kager_roi = img[kager_y1:kager_y2, kager_x1:kager_x2]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Kager's Triangle Region")
axes[0].add_patch(plt.Rectangle((kager_x1, kager_y1), kager_x2-kager_x1, kager_y2-kager_y1, fill=False, edgecolor='yellow', linewidth=2))

axes[1].imshow(kager_roi, cmap="gray")
axes[1].set_title(f"Kager's Triangle ROI ({kager_roi.shape[1]}x{kager_roi.shape[0]})")
axes[1].add_patch(plt.Polygon([(0.1, 0.8), (0.1, 0.2), (0.9, 0.5)], closed=True, fill=False, edgecolor='red', linewidth=2, transform=axes[1].transAxes))

kager_enhanced = clahe_full.apply(kager_roi)
axes[2].imshow(kager_enhanced, cmap="gray")
axes[2].set_title("Kager's Triangle (CLAHE Enhanced)")
axes[2].add_patch(plt.Polygon([(0.1, 0.8), (0.1, 0.2), (0.9, 0.5)], closed=True, fill=False, edgecolor='red', linewidth=2, transform=axes[2].transAxes))
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "10_kagers_triangle.png"), dpi=150, bbox_inches="tight")
plt.close()
print("10_kagers_triangle.png saved")

# ============================================================
# ANALYSIS 11: Cuboid Texture Analysis (GLCM Comparison)
# ============================================================
cuboid_x1 = int(w * 0.5)
cuboid_y1 = int(h * 0.48)
cuboid_x2 = int(w * 0.65)
cuboid_y2 = int(h * 0.65)
cuboid_roi = img[cuboid_y1:cuboid_y2, cuboid_x1:cuboid_x2]

calc_ctrl_x1 = int(w * 0.08)
calc_ctrl_y1 = int(h * 0.5)
calc_ctrl_x2 = int(w * 0.25)
calc_ctrl_y2 = int(h * 0.7)
calc_roi = img[calc_ctrl_y1:calc_ctrl_y2, calc_ctrl_x1:calc_ctrl_x2]

def compute_glcm_features(image_region):
    region = image_region.astype(float)
    region = np.clip(region, 5, 250)
    bins = np.linspace(0, 255, 32)
    digitized = np.digitize(region.flatten(), bins)
    digitized = digitized.reshape(region.shape)
    h, w = digitized.shape
    glcm = np.zeros((32, 32))
    for i in range(h-1):
        for j in range(w-1):
            r1, r2 = digitized[i, j], digitized[i+1, j]
            glcm[r1, r2] += 1
            glcm[r2, r1] += 1
    if glcm.sum() > 0:
        glcm_norm = glcm / glcm.sum()
    else:
        glcm_norm = glcm
    contrast = 0
    homogeneity = 0
    energy = 0
    for i in range(32):
        for j in range(32):
            contrast += (i-j)**2 * glcm_norm[i, j]
            homogeneity += glcm_norm[i, j] / (1 + abs(i-j))
            energy += glcm_norm[i, j]**2
    local_var = np.var(region)
    entropy = 0
    for i in range(32):
        for j in range(32):
            if glcm_norm[i, j] > 0:
                entropy -= glcm_norm[i, j] * np.log2(glcm_norm[i, j] + 1e-10)
    return {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'entropy': entropy,
        'local_var': local_var
    }

cuboid_feats = compute_glcm_features(cuboid_roi)
calc_feats = compute_glcm_features(calc_roi)

print("\nCuboid Texture Features:")
for k, v in cuboid_feats.items():
    print(f"  {k}: {v:.4f}")
print("\nCalcaneus (Control) Texture Features:")
for k, v in calc_feats.items():
    print(f"  {k}: {v:.4f}")
print("\nRatio (Cuboid/Calcaneus):")
for k in cuboid_feats:
    if k == 'local_var':
        ratio = cuboid_feats[k] / (calc_feats[k] + 1e-10)
    else:
        ratio = cuboid_feats[k] / (calc_feats[k] + 1e-10)
    print(f"  {k}: {ratio:.4f}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original - ROI Locations")
axes[0, 0].add_patch(plt.Rectangle((cuboid_x1, cuboid_y1), cuboid_x2-cuboid_x1, cuboid_y2-cuboid_y1, fill=False, edgecolor='red', linewidth=2, label='Cuboid ROI'))
axes[0, 0].add_patch(plt.Rectangle((calc_ctrl_x1, calc_ctrl_y1), calc_ctrl_x2-calc_ctrl_x1, calc_ctrl_y2-calc_ctrl_y1, fill=False, edgecolor='blue', linewidth=2, label='Calcaneus Control ROI'))
axes[0, 0].legend(loc='upper right')

axes[0, 1].imshow(cuboid_roi, cmap="gray")
axes[0, 1].set_title(f"Cuboid ROI ({cuboid_roi.shape[1]}x{cuboid_roi.shape[0]})")
axes[0, 2].imshow(calc_roi, cmap="gray")
axes[0, 2].set_title(f"Calcaneus Control ROI ({calc_roi.shape[1]}x{calc_roi.shape[0]})")

features = ['contrast', 'homogeneity', 'energy', 'entropy', 'local_var']
colors = ['red', 'blue']
labels = ['Cuboid', 'Calcaneus (Control)']
for idx, feat in enumerate(features):
    ax_idx = idx % 3
    row_idx = 1 if idx < 3 else 1
    if idx >= 3:
        ax_idx = idx - 3
    if ax_idx < 3:
        vals = [cuboid_feats[feat], calc_feats[feat]]
        bars = axes[1, ax_idx].bar(labels, vals, color=colors)
        axes[1, ax_idx].set_title(f"{feat}")
        axes[1, ax_idx].set_ylabel("Feature Value")
        for bar, val in zip(bars, vals):
            axes[1, ax_idx].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02 * max(vals),
                                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "11_cuboid_texture.png"), dpi=150, bbox_inches="tight")
plt.close()
print("11_cuboid_texture.png saved")

# ============================================================
# ANALYSIS 12: Distal Tibia Crop + Enhancement
# ============================================================
tibia_x1 = int(w * 0.3)
tibia_y1 = int(h * 0.02)
tibia_x2 = int(w * 0.55)
tibia_y2 = int(h * 0.4)
tibia_crop = img[tibia_y1:tibia_y2, tibia_x1:tibia_x2]
p1t, p99t = np.percentile(tibia_crop[tibia_crop > 10], [1, 99])
tibia_enhanced = np.clip((tibia_crop - p1t) / (p99t - p1t + 1e-10) * 255, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Distal Tibia Location")
axes[0].add_patch(plt.Rectangle((tibia_x1, tibia_y1), tibia_x2-tibia_x1, tibia_y2-tibia_y1, fill=False, edgecolor='red', linewidth=2))
axes[1].imshow(tibia_crop, cmap="gray")
axes[1].set_title(f"Distal Tibia Crop ({tibia_crop.shape[1]}x{tibia_crop.shape[0]})")
axes[2].imshow(tibia_enhanced, cmap="gray")
axes[2].set_title("Distal Tibia (Contrast Enhanced)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "12_distal_tibia_crop.png"), dpi=150, bbox_inches="tight")
plt.close()
Image.fromarray(tibia_enhanced).save(os.path.join(OUTDIR, "12_distal_tibia_enhanced.png"))
print("12_distal_tibia_crop.png and 12_distal_tibia_enhanced.png saved")

# ============================================================
# ANALYSIS 13: Plantar Calcaneal Spur Region Crop
# ============================================================
spur_x1 = int(w * 0.05)
spur_y1 = int(h * 0.55)
spur_x2 = int(w * 0.25)
spur_y2 = int(h * 0.85)
spur_crop = img[spur_y1:spur_y2, spur_x1:spur_x2]
p1s, p99s = np.percentile(spur_crop[spur_crop > 10], [1, 99])
spur_enhanced = np.clip((spur_crop - p1s) / (p99s - p1s + 1e-10) * 255, 0, 255).astype(np.uint8)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Plantar Calcaneus Location")
axes[0].add_patch(plt.Rectangle((spur_x1, spur_y1), spur_x2-spur_x1, spur_y2-spur_y1, fill=False, edgecolor='red', linewidth=2))
axes[1].imshow(spur_crop, cmap="gray")
axes[1].set_title(f"Plantar Calcaneus Crop ({spur_crop.shape[1]}x{spur_crop.shape[0]})")
axes[2].imshow(spur_enhanced, cmap="gray")
axes[2].set_title("Plantar Calcaneus (Contrast Enhanced)")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "13_plantar_calcaneus_crop.png"), dpi=150, bbox_inches="tight")
plt.close()
print("13_plantar_calcaneus_crop.png saved")

# ============================================================
# ANALYSIS 14: Comparison Montage
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
axes[0, 0].imshow(img, cmap="gray")
axes[0, 0].set_title("Original Image")
axes[0, 1].imshow(inverted, cmap="gray")
axes[0, 1].set_title("Inverted (Negative Format)")
axes[0, 2].imshow(contrast_stretched, cmap="gray")
axes[0, 2].set_title("Contrast Stretched")
axes[1, 0].imshow(clahe_result, cmap="gray")
axes[1, 0].set_title("CLAHE (clip=3.0, tile=8x8)")
axes[1, 1].imshow(unsharp, cmap="gray")
axes[1, 1].set_title("Unsharp Mask (sigma=2.0, amount=1.5)")
axes[1, 2].imshow(clahe_midfoot, cmap="gray")
axes[1, 2].set_title("CLAHE - Midfoot ROI (Cuboid Region)")
for ax in axes.flat:
    ax.axis("off")
plt.suptitle("ID001-xray: Processing Comparison Montage", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "14_comparison_montage.png"), dpi=150, bbox_inches="tight")
plt.close()
print("14_comparison_montage.png saved")

print("\n=== PROCESSING COMPLETE ===")
print(f"All outputs saved to: {OUTDIR}")
