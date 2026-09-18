import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

INPUT = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images/ID001-xray_original.png"
OUTDIR = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images"

img = np.array(Image.open(INPUT).convert("L"))
h, w = img.shape
print(f"Image shape: {img.shape}")

# ============================================================
# CORRECTED ANALYSIS 6: Böhler's Angle and Gissane's Angle
# ============================================================
# Background is ~15-20. Bones go up to 237.
# Use threshold > 50 to isolate bone regions.

bone_mask = (img > 50).astype(np.uint8) * 255
num_labels, labels, stats_rect, centroids = cv2.connectedComponentsWithStats(bone_mask, connectivity=8)

# Find bone components
bone_comps = []
for i in range(1, num_labels):
    x, y, bw, bh, area = stats_rect[i]
    if area > 200:
        bone_comps.append((i, x, y, bw, bh, area))

bone_comps.sort(key=lambda k: k[5], reverse=True)
print(f"Found {len(bone_comps)} bone components (threshold > 50, area > 200px)")
for comp in bone_comps[:10]:
    idx, x, y, bw, bh, area = comp
    print(f"  Component {idx}: ({x},{y}), {bw}x{bh}, area={area}, center=({x+bw//2},{y+bh//2})")

# The calcaneus is typically the largest bone component in the hindfoot
# In a lateral foot X-ray, it should be in the lower-left to lower-center portion
# The tibia will be a large vertical component at the top

# Try to identify calcaneus: look for a large component in the lower portion of the image
# that's not the tibia (tibia is at the top of the image)
calcaneus_comp = None
for comp in bone_comps:
    idx, x, y, bw, bh, area = comp
    center_y = y + bh // 2
    center_x = x + bw // 2
    # Calcaneus should be in the lower portion (y > 40% of image height)
    # and not too far right (x < 60% of image width)
    if area > 500 and center_y > h * 0.35 and center_x < w * 0.55:
        calcaneus_comp = comp
        print(f"Selected calcaneus: component {idx} at ({x},{y}), {bw}x{bh}, area={area}")
        break

# If no separate calcaneus found, it's connected to other bones
# Use the main component and identify calcaneal region by position
if calcaneus_comp is None and len(bone_comps) > 0:
    # Find the main bone component
    main_comp = bone_comps[0]
    midx, mx, my, mbw, mbh, marea = main_comp
    print(f"\nNo separate calcaneus found. Main component: ({mx},{my}), {mbw}x{mbh}")

    # Extract the main component mask
    main_mask = (labels == midx).astype(np.uint8)

    # The calcaneus is in the posterior-inferior portion of the foot
    # Let's find the superior surface of the bone in the calcaneal region
    # Search in the lower-left quadrant of the main component

    # Define calcaneal search region: lower-left portion of the main component
    search_x1 = max(0, mx)
    search_y1 = max(0, my + int(mbh * 0.3))
    search_x2 = min(w, mx + int(mbw * 0.5))
    search_y2 = min(h, my + int(mbh * 0.85))

    print(f"Calcaneal search region: ({search_x1},{search_y1}) to ({search_x2},{search_y2})")

    # Trace the superior bone surface in the search region
    search_mask = bone_mask[search_y1:search_y2, search_x1:search_x2]
    superior_points = []
    for x in range(search_mask.shape[1]):
        col = np.where(search_mask[:, x] > 0)[0]
        if len(col) > 3:
            superior_points.append((x, col[0]))

    if len(superior_points) > 10:
        superior_points = np.array(superior_points)
        sp_x = superior_points[:, 0]
        sp_y = superior_points[:, 1]

        # Böhler's angle landmarks:
        # P1: posterior-superior angle (leftmost point on superior surface)
        # P2: highest point of posterior facet (apex - minimum y)
        # P3: anterior-superior angle (rightmost point on superior surface)

        p1_local = (sp_x[0], sp_y[0])
        apex_idx = np.argmin(sp_y)
        p2_local = (sp_x[apex_idx], sp_y[apex_idx])
        p3_local = (sp_x[-1], sp_y[-1])

        # Convert to global
        p1 = (search_x1 + p1_local[0], search_y1 + p1_local[1])
        p2 = (search_x1 + p2_local[0], search_y1 + p2_local[1])
        p3 = (search_x1 + p3_local[0], search_y1 + p3_local[1])

        # Böhler's angle
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_b = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        boehler = np.degrees(np.arccos(np.clip(cos_b, -1, 1)))

        # Gissane's angle: angle between anterior facet and posterior superior surface
        # at the anterior superior angle
        # Find a point on the anterior facet surface (going downward from p3)
        ant_facet_x = max(0, p3_local[0] - 15)
        facet_col = np.where(search_mask[:, ant_facet_x] > 0)[0]
        if len(facet_col) > 0:
            p4_local = (ant_facet_x, facet_col[0])
            p4 = (search_x1 + p4_local[0], search_y1 + p4_local[1])

            v3 = np.array([p4[0] - p3[0], p4[1] - p3[1]])
            v4 = np.array([p1[0] - p3[0], p1[1] - p3[1]])
            cos_g = np.dot(v3, v4) / (np.linalg.norm(v3) * np.linalg.norm(v4) + 1e-10)
            gissane = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))
        else:
            gissane = None

        print(f"\nBöhler's angle: {boehler:.1f}° (normal: 20-40°)")
        if gissane:
            print(f"Gissane's angle: {gissane:.1f}° (normal: 120-145°)")
    else:
        print("Could not trace superior surface in search region")
        boehler = None
        gissane = None

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"Original - Calcaneal Landmarks\nMain bone: ({mx},{my}), {mbw}x{mbh}")
    axes[0].add_patch(plt.Rectangle((mx, my), mbw, mbh, fill=False, edgecolor='green', linewidth=2))
    axes[0].add_patch(plt.Rectangle((search_x1, search_y1), search_x2-search_x1, search_y2-search_y1, fill=False, edgecolor='yellow', linewidth=2, label='Search region'))

    if boehler is not None:
        axes[0].plot(p1[0], p1[1], 'go', markersize=12, label='P1: Post-Sup Angle')
        axes[0].plot(p2[0], p2[1], 'ro', markersize=12, label='P2: Facet Apex')
        axes[0].plot(p3[0], p3[1], 'bo', markersize=12, label='P3: Ant-Sup Angle')
        axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2.5)
        axes[0].plot([p2[0], p3[0]], [p2[1], p3[1]], 'b-', linewidth=2.5)
        if gissane:
            axes[0].plot(p4[0], p4[1], 'mo', markersize=12, label='P4: Ant Facet')
            axes[0].plot([p4[0], p3[0]], [p4[1], p3[1]], 'm-', linewidth=2.5)
        axes[0].legend(loc='upper right')

        title = f"Böhler: {boehler:.1f}°"
        if gissane:
            title += f", Gissane: {gissane:.1f}°"
        title += "\n(normal Böhler: 20-40°, Gissane: 120-145°)"
        axes[0].set_title(title)

    axes[1].imshow(search_mask, cmap="gray")
    axes[1].set_title(f"Bone Mask - Calcaneal Search Region")
    if boehler is not None:
        axes[1].plot(sp_x, sp_y, 'r-', linewidth=1.5)
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
# CORRECTED ANALYSIS 9: Soft Tissue Thickness Profiling
# ============================================================
# Background is ~15-20. Soft tissue is between 20-50.
# Use a conservative threshold to capture soft tissue boundary.

# Soft tissue mask: pixels that are clearly above background
# but not necessarily bone
soft_threshold = 35
soft_mask = img > soft_threshold

dorsal_y = []
plantar_y = []
positions = []
for x in range(w):
    col_indices = np.where(soft_mask[:, x])[0]
    if len(col_indices) > 5:
        # The dorsal contour is the highest (smallest y) soft tissue pixel
        # The plantar contour is the lowest (largest y) soft tissue pixel
        dorsal_y.append(col_indices[0])
        plantar_y.append(col_indices[-1])
        positions.append(x)

if len(positions) > 10:
    positions = np.array(positions)
    dorsal_y = np.array(dorsal_y)
    plantar_y = np.array(plantar_y)
    thickness = plantar_y - dorsal_y

    print(f"\nSoft tissue profile (threshold > {soft_threshold}):")
    print(f"  Points: {len(positions)}")
    print(f"  Dorsal range: {dorsal_y.min()}-{dorsal_y.max()}")
    print(f"  Plantar range: {plantar_y.min()}-{plantar_y.max()}")
    print(f"  Thickness range: {thickness.min()}-{thickness.max()} px")
    print(f"  Mean thickness: {thickness.mean():.0f} px")

    third = len(positions) // 3
    mid = len(positions) // 2
    two_third = 2 * len(positions) // 3
    print(f"  At 1/3: {thickness[third]} px")
    print(f"  At 1/2: {thickness[mid]} px")
    print(f"  At 2/3: {thickness[two_third]} px")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].imshow(img, cmap="gray")
    axes[0].plot(positions, dorsal_y, 'r-', linewidth=1.5, label='Dorsal contour')
    axes[0].plot(positions, plantar_y, 'g-', linewidth=1.5, label='Plantar contour')
    axes[0].axvline(x=positions[third], color='yellow', linestyle='--', alpha=0.5)
    axes[0].axvline(x=positions[mid], color='cyan', linestyle='--', alpha=0.5)
    axes[0].axvline(x=positions[two_third], color='magenta', linestyle='--', alpha=0.5)
    axes[0].legend(loc='upper right')
    axes[0].set_title("Soft Tissue Contour Extraction")
    axes[0].axis("off")

    axes[1].plot(positions, thickness, 'b-', linewidth=1.5)
    axes[1].set_xlabel("Horizontal Position (pixels)")
    axes[1].set_ylabel("Soft Tissue Thickness (pixels)")
    axes[1].set_title("Soft Tissue Thickness Profile")
    axes[1].axvline(x=positions[third], color='yellow', linestyle='--', alpha=0.5, label='Hindfoot')
    axes[1].axvline(x=positions[mid], color='cyan', linestyle='--', alpha=0.5, label='Midfoot')
    axes[1].axvline(x=positions[two_third], color='magenta', linestyle='--', alpha=0.5, label='Forefoot')
    axes[1].annotate(f"{thickness[third]}px", (positions[third], thickness[third]), textcoords="offset points", xytext=(5, 10))
    axes[1].annotate(f"{thickness[mid]}px", (positions[mid], thickness[mid]), textcoords="offset points", xytext=(5, -15))
    axes[1].annotate(f"{thickness[two_third]}px", (positions[two_third], thickness[two_third]), textcoords="offset points", xytext=(5, -15))
    axes[1].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "09_soft_tissue_profile.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("09_soft_tissue_profile.png saved")
else:
    print(f"No soft tissue contour found with threshold > {soft_threshold}")

# ============================================================
# CORRECTED ANALYSIS 10: Kager's Fat Pad Triangle
# ============================================================
# Kager's triangle is anterior to the posterior calcaneus
# In this image: look just anterior to the heel bone

# Find the posterior edge of the main bone component
if len(bone_comps) > 0:
    main_comp = bone_comps[0]
    midx, mx, my, mbw, mbh, marea = main_comp

    kager_x1 = mx + int(mbw * 0.4)
    kager_y1 = my + int(mbh * 0.15)
    kager_x2 = mx + int(mbw * 0.65)
    kager_y2 = my + int(mbh * 0.45)
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

print("\n=== CORRECTED ANALYSIS COMPLETE ===")
