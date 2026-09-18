import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

INPUT = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images/ID001-xray_original.png"
OUTDIR = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images"

img = np.array(Image.open(INPUT).convert("L"))
h, w = img.shape
print(f"Image shape: {img.shape}, range: [{img.min()}, {img.max()}]")

# The image has bones as bright on dark background
# Find the bounding box of the entire foot (all bone + soft tissue)
bone_mask = (img > 30).astype(np.uint8) * 255
num_labels, labels, stats_rect, centroids = cv2.connectedComponentsWithStats(bone_mask, connectivity=8)

# Find largest component (the foot)
largest = max(range(1, num_labels), key=lambda i: stats_rect[i, cv2.CC_STAT_AREA])
lx, ly, lw, lh, larea = stats_rect[largest]
print(f"Largest component: ({lx}, {ly}), size {lw}x{lh}, area {larea}")

# ============================================================
# ANALYSIS 6 (FIXED): Böhler's Angle and Gissane's Angle
# ============================================================
# The calcaneus is the largest tarsal bone in the posterior foot (bottom-left of foot)
# Since bones are connected, we need to identify the calcaneal region by anatomy
# Calcaneus occupies approximately the lower-left quarter of the foot bounding box

# Create a calcaneal search region: bottom-left portion of the foot
calc_search_x1 = max(0, lx - 20)
calc_search_y1 = max(0, ly + int(lh * 0.15))
calc_search_x2 = lx + int(lw * 0.45)
calc_search_y2 = min(h, ly + int(lh * 0.85))

calc_search = img[calc_search_y1:calc_search_y2, calc_search_x1:calc_search_x2]

# Find the superior surface of the calcaneus in this region
# The calcaneus superior surface forms an arch-like curve
# Böhler's angle needs 3 points:
# P1: posterior-superior angle (back top of calcaneus)
# P2: highest point of posterior facet (apex)  
# P3: anterior-superior angle (front top of calcaneus)

# Trace the superior border of the bright bone region in the calcaneal search area
bone_in_search = calc_search > 60
superior_border = []
for x in range(calc_search.shape[1]):
    col_indices = np.where(bone_in_search[:, x])[0]
    if len(col_indices) > 5:
        superior_border.append((x, col_indices[0]))

if len(superior_border) > 10:
    superior_border = np.array(superior_border)
    sup_x = superior_border[:, 0]
    sup_y = superior_border[:, 1]

    # P1: posterior-superior angle = leftmost point of superior border
    p1_local = (sup_x[0], sup_y[0])

    # P2: highest point = minimum y value
    p2_idx = np.argmin(sup_y)
    p2_local = (sup_x[p2_idx], sup_y[p2_idx])

    # P3: anterior-superior angle = rightmost point of superior border
    p3_local = (sup_x[-1], sup_y[-1])

    # Convert to global coordinates
    p1_global = (calc_search_x1 + p1_local[0], calc_search_y1 + p1_local[1])
    p2_global = (calc_search_x1 + p2_local[0], calc_search_y1 + p2_local[1])
    p3_global = (calc_search_x1 + p3_local[0], calc_search_y1 + p3_local[1])

    # Calculate Böhler's angle
    v1 = np.array([p1_global[0] - p2_global[0], p1_global[1] - p2_global[1]])
    v2 = np.array([p3_global[0] - p2_global[0], p3_global[1] - p2_global[1]])
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    boehler_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

    # Gissane's angle: anterior calcaneal angle
    # Formed by the line from anterior superior angle along anterior facet 
    # and the line from anterior superior angle along posterior superior surface
    # Need a 4th point: anterior facet direction
    anterior_facet_x = p3_local[0] - 10
    facet_col = bone_in_search[:, anterior_facet_x]
    facet_indices = np.where(facet_col)[0]
    if len(facet_indices) > 0:
        p4_local = (anterior_facet_x, facet_indices[0])
        p4_global = (calc_search_x1 + p4_local[0], calc_search_y1 + p4_local[1])
        
        v3 = np.array([p3_global[0] - p4_global[0], p3_global[1] - p4_global[1]])
        v4 = np.array([p3_global[0] - p1_global[0], p3_global[1] - p1_global[1]])
        cos_gissane = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4) + 1e-10)
        gissane_angle = np.degrees(np.arccos(np.clip(cos_gissane, -1, 1)))
    else:
        gissane_angle = None

    print(f"Böhler's angle: {boehler_angle:.1f}° (normal: 20-40°)")
    print(f"Gissane's angle: {gissane_angle:.1f}° (normal: 120-145°)" if gissane_angle else "Gissane's angle: could not compute")
else:
    boehler_angle = None
    gissane_angle = None
    print("Could not trace calcaneal superior border")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
axes[0].imshow(img, cmap="gray")
axes[0].set_title(f"Original - Foot Bounding Box\nbbox=({lx}, {ly}), {lw}x{lh}")
axes[0].add_patch(plt.Rectangle((lx, ly), lw, lh, fill=False, edgecolor='green', linewidth=2))
axes[0].add_patch(plt.Rectangle((calc_search_x1, calc_search_y1), calc_search_x2-calc_search_x1, calc_search_y2-calc_search_y1, fill=False, edgecolor='yellow', linewidth=2, label='Calcaneal search region'))

if boehler_angle is not None:
    axes[0].plot(p1_global[0], p1_global[1], 'go', markersize=12, label='P1: Post-Sup Angle')
    axes[0].plot(p2_global[0], p2_global[1], 'ro', markersize=12, label='P2: Facet Apex')
    axes[0].plot(p3_global[0], p3_global[1], 'bo', markersize=12, label='P3: Ant-Sup Angle')
    axes[0].plot([p1_global[0], p2_global[0]], [p1_global[1], p2_global[1]], 'g-', linewidth=2)
    axes[0].plot([p2_global[0], p3_global[0]], [p2_global[1], p3_global[1]], 'b-', linewidth=2)
    if gissane_angle:
        axes[0].plot(p4_global[0], p4_global[1], 'mo', markersize=12, label='P4: Ant Facet')
        axes[0].plot([p4_global[0], p3_global[0]], [p4_global[1], p3_global[1]], 'm-', linewidth=2)
    axes[0].legend(loc='upper right')

    title = f"Böhler: {boehler_angle:.1f}°"
    if gissane_angle:
        title += f", Gissane: {gissane_angle:.1f}°"
    axes[0].set_title(title + "\n(normal Böhler: 20-40°, Gissane: 120-145°)")

axes[1].imshow(calc_search, cmap="gray")
axes[1].set_title(f"Calcaneal Search Region ({calc_search.shape[1]}x{calc_search.shape[0]})")
if len(superior_border) > 10:
    axes[1].plot(sup_x, sup_y, 'r-', linewidth=1.5)
    axes[1].plot(p1_local[0], p1_local[1], 'go', markersize=10)
    axes[1].plot(p2_local[0], p2_local[1], 'ro', markersize=10)
    axes[1].plot(p3_local[0], p3_local[1], 'bo', markersize=10)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "06_calcaneal_angles.png"), dpi=150, bbox_inches="tight")
plt.close()
print("06_calcaneal_angles.png saved")

# ============================================================
# ANALYSIS 9 (FIXED): Soft Tissue Thickness Profiling
# ============================================================
# The soft tissue boundary is where intensity transitions from bone/soft-tissue to background
# Background is near-black (< 10). Soft tissue is > 10 but < bone threshold.
# We need to find the outer contour of the visible anatomy (including soft tissue).

soft_mask = img > 15
dorsal_y = []
plantar_y = []
positions = []
for x in range(w):
    col_indices = np.where(soft_mask[:, x])[0]
    if len(col_indices) > 5:
        dorsal_y.append(col_indices[0])
        plantar_y.append(col_indices[-1])
        positions.append(x)

positions = np.array(positions)
dorsal_y = np.array(dorsal_y)
plantar_y = np.array(plantar_y)
thickness = plantar_y - dorsal_y

print(f"\nSoft tissue profile: {len(positions)} points")
print(f"  Dorsal range: {dorsal_y.min()}-{dorsal_y.max()}")
print(f"  Plantar range: {plantar_y.min()}-{plantar_y.max()}")
print(f"  Thickness range: {thickness.min()}-{thickness.max()} px")
print(f"  Mean thickness: {thickness.mean():.0f} px")

# Key measurement points
mid_idx = len(positions) // 2
third_idx = len(positions) // 3
two_third_idx = 2 * len(positions) // 3

if len(thickness) > 10:
    print(f"  Thickness at 1/3: {thickness[third_idx]} px")
    print(f"  Thickness at 1/2: {thickness[mid_idx]} px")
    print(f"  Thickness at 2/3: {thickness[two_third_idx]} px")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
axes[0].imshow(img, cmap="gray")
if len(positions) > 0:
    axes[0].plot(positions, dorsal_y, 'r-', linewidth=1.5, label='Dorsal contour')
    axes[0].plot(positions, plantar_y, 'g-', linewidth=1.5, label='Plantar contour')
    axes[0].axvline(x=positions[third_idx], color='yellow', linestyle='--', alpha=0.5)
    axes[0].axvline(x=positions[mid_idx], color='cyan', linestyle='--', alpha=0.5)
    axes[0].axvline(x=positions[two_third_idx], color='magenta', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
axes[0].set_title("Soft Tissue Contour Extraction")
axes[0].axis("off")

if len(thickness) > 0:
    axes[1].plot(positions, thickness, 'b-', linewidth=1.5)
    axes[1].set_xlabel("Horizontal Position (pixels)")
    axes[1].set_ylabel("Soft Tissue Thickness (pixels)")
    axes[1].set_title("Soft Tissue Thickness Profile")
    axes[1].axvline(x=positions[third_idx], color='yellow', linestyle='--', alpha=0.5, label='Hindfoot region')
    axes[1].axvline(x=positions[mid_idx], color='cyan', linestyle='--', alpha=0.5, label='Midfoot region')
    axes[1].axvline(x=positions[two_third_idx], color='magenta', linestyle='--', alpha=0.5, label='Forefoot region')
    axes[1].annotate(f"{thickness[third_idx]}px", (positions[third_idx], thickness[third_idx]), textcoords="offset points", xytext=(5, 10))
    axes[1].annotate(f"{thickness[mid_idx]}px", (positions[mid_idx], thickness[mid_idx]), textcoords="offset points", xytext=(5, 10))
    axes[1].annotate(f"{thickness[two_third_idx]}px", (positions[two_third_idx], thickness[two_third_idx]), textcoords="offset points", xytext=(5, 10))
    axes[1].legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "09_soft_tissue_profile.png"), dpi=150, bbox_inches="tight")
plt.close()
print("09_soft_tissue_profile.png saved")

# ============================================================
# ANALYSIS 10 (FIXED): Kager's Fat Pad Triangle
# ============================================================
kager_x1 = lx + int(lw * 0.3)
kager_y1 = ly + int(lh * 0.2)
kager_x2 = lx + int(lw * 0.55)
kager_y2 = ly + int(lh * 0.5)
kager_roi = img[kager_y1:kager_y2, kager_x1:kager_x2]

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
kager_enhanced = clahe.apply(kager_roi)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Kager's Triangle Region")
axes[0].add_patch(plt.Rectangle((kager_x1, kager_y1), kager_x2-kager_x1, kager_y2-kager_y1, fill=False, edgecolor='yellow', linewidth=2))

axes[1].imshow(kager_roi, cmap="gray")
axes[1].set_title(f"Kager's Triangle ROI ({kager_roi.shape[1]}x{kager_roi.shape[0]})")
axes[1].add_patch(plt.Polygon([(0.05, 0.85), (0.05, 0.15), (0.85, 0.5)], closed=True, fill=False, edgecolor='red', linewidth=2, transform=axes[1].transAxes))

axes[2].imshow(kager_enhanced, cmap="gray")
axes[2].set_title("Kager's Triangle (CLAHE Enhanced)")
axes[2].add_patch(plt.Polygon([(0.05, 0.85), (0.05, 0.15), (0.85, 0.5)], closed=True, fill=False, edgecolor='red', linewidth=2, transform=axes[2].transAxes))
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "10_kagers_triangle.png"), dpi=150, bbox_inches="tight")
plt.close()
print("10_kagers_triangle.png saved")

print("\n=== FIX PIPELINE COMPLETE ===")
