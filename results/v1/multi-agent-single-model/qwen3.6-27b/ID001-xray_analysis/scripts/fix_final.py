import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

INPUT = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images/ID001-xray_original.png"
OUTDIR = "/home/ralph/projects/ufsc/pablo-xray-tests-qwen3.6/ID001-xray_analysis/images"

img = np.array(Image.open(INPUT).convert("L"))
h, w = img.shape
print(f"Image shape: {h}x{w}")

# ============================================================
# CORRECTED Böhler's Angle and Gissane's Angle
# ============================================================
# From analysis: tibia is at cols 120-160, rows 0-144
# Calcaneus is at rows ~250-330, cols ~120-250

# Find bone pixels in the calcaneal region
# Use threshold > 80 for bright bone
calc_y1, calc_y2 = 230, 340
calc_x1, calc_x2 = 100, 280
calc_region = img[calc_y1:calc_y2, calc_x1:calc_x2]
calc_bone = calc_region > 80

# Trace superior surface of calcaneal bone
sup_points = []
for x in range(calc_bone.shape[1]):
    bone_rows = np.where(calc_bone[:, x])[0]
    if len(bone_rows) > 5:
        sup_points.append((x, bone_rows[0]))

if len(sup_points) > 10:
    sup_points = np.array(sup_points)
    sx = sup_points[:, 0]
    sy = sup_points[:, 1]

    # Smooth the superior surface with a running median to find the arch
    from scipy.ndimage import median_filter
    sy_smooth = median_filter(sy, size=5).astype(int)

    # Böhler's angle points (in global coordinates):
    # P1: posterior-superior angle (leftmost point of superior surface)
    p1_local = (sx[0], sy_smooth[0])
    # P2: apex of posterior facet (lowest y = highest point on bone)
    apex_idx = np.argmin(sy_smooth)
    p2_local = (sx[apex_idx], sy_smooth[apex_idx])
    # P3: anterior-superior angle (rightmost point)
    p3_local = (sx[-1], sy_smooth[-1])

    p1 = (calc_x1 + p1_local[0], calc_y1 + p1_local[1])
    p2 = (calc_x1 + p2_local[0], calc_y1 + p2_local[1])
    p3 = (calc_x1 + p3_local[0], calc_y1 + p3_local[1])

    # Böhler's angle: angle at P2 between P1-P2 and P3-P2
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cos_b = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
    boehler = np.degrees(np.arccos(np.clip(cos_b, -1, 1)))

    # Gissane's angle: anterior calcaneal angle
    # At P3, the angle between the anterior facet surface and the posterior superior surface
    # Find point on anterior facet (going down from P3)
    p3_row = p3_local[1]
    p3_col = p3_local[0]

    # Trace the anterior facet: from P3 going downward
    facet_point = None
    for y_offset in range(5, min(40, calc_bone.shape[0] - p3_row)):
        test_y = p3_row + y_offset
        if test_y < calc_bone.shape[0]:
            row_test = calc_bone[test_y, max(0, p3_col-5):min(calc_bone.shape[1], p3_col+5)]
            if np.any(row_test):
                facet_point = (p3_col, test_y)
                break

    if facet_point:
        p4_local = facet_point
        p4 = (calc_x1 + p4_local[0], calc_y1 + p4_local[1])

        v_ant = np.array([p4[0] - p3[0], p4[1] - p3[1]])
        v_post = np.array([p1[0] - p3[0], p1[1] - p3[1]])
        cos_g = np.dot(v_ant, v_post) / (np.linalg.norm(v_ant) * np.linalg.norm(v_post) + 1e-10)
        gissane = np.degrees(np.arccos(np.clip(cos_g, -1, 1)))
    else:
        gissane = None

    print(f"Landmarks (local): P1={p1_local}, P2={p2_local}, P3={p3_local}")
    print(f"Landmarks (global): P1={p1}, P2={p2}, P3={p3}")
    print(f"Böhler's angle: {boehler:.1f}° (normal: 20-40°)")
    if gissane:
        print(f"Gissane's angle: {gissane:.1f}° (normal: 120-145°)")
else:
    print("Could not trace calcaneal superior surface")
    boehler = None
    gissane = None

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 8))
axes[0].imshow(img, cmap="gray")
axes[0].set_title("Original - Calcaneal Region & Landmarks")
axes[0].add_patch(plt.Rectangle((calc_x1, calc_y1), calc_x2-calc_x1, calc_y2-calc_y1, fill=False, edgecolor='yellow', linewidth=2, label='Calcaneal search region'))

if boehler is not None:
    axes[0].plot(p1[0], p1[1], 'go', markersize=15, label='P1: Post-Sup Angle')
    axes[0].plot(p2[0], p2[1], 'ro', markersize=15, label='P2: Facet Apex')
    axes[0].plot(p3[0], p3[1], 'bo', markersize=15, label='P3: Ant-Sup Angle')
    axes[0].plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=3)
    axes[0].plot([p2[0], p3[0]], [p2[1], p3[1]], 'b-', linewidth=3)
    if gissane and 'p4' in dir():
        axes[0].plot(p4[0], p4[1], 'mo', markersize=12, label='P4: Ant Facet')
        axes[0].plot([p4[0], p3[0]], [p4[1], p3[1]], 'm-', linewidth=2.5)
    axes[0].legend(loc='upper right', fontsize=9)
    title = f"Böhler: {boehler:.1f}°"
    if gissane:
        title += f", Gissane: {gissane:.1f}°"
    title += "\n(normal Böhler: 20-40°, Gissane: 120-145°)"
    axes[0].set_title(title)

axes[1].imshow(calc_region, cmap="gray")
axes[1].set_title(f"Calcaneal Region ({calc_x1}:{calc_x2}, {calc_y1}:{calc_y2})")
if boehler is not None:
    axes[1].plot(sx, sy_smooth, 'r-', linewidth=2)
    axes[1].plot(p1_local[0], p1_local[1], 'go', markersize=12)
    axes[1].plot(p2_local[0], p2_local[1], 'ro', markersize=12)
    axes[1].plot(p3_local[0], p3_local[1], 'bo', markersize=12)
    # Also plot the raw superior surface
    axes[1].plot(sx, sy, 'k-', linewidth=0.5, alpha=0.3)
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "06_calcaneal_angles.png"), dpi=150, bbox_inches="tight")
plt.close()
print("06_calcaneal_angles.png saved")

# ============================================================
# CORRECTED Soft Tissue Profile
# ============================================================
# Background is ~15-20. Use threshold > 30 for soft tissue.
# But need to be careful about the gray band around the foot.

# Use the bone mask to find the foot silhouette
# First, find the foot bounding box from bone > 80
bone_mask = img > 80
bone_rows, bone_cols = np.where(bone_mask)
if len(bone_rows) > 0:
    foot_bbox = (bone_cols.min(), bone_rows.min(), bone_cols.max() - bone_cols.min(), bone_rows.max() - bone_rows.min())
    print(f"\nFoot bounding box: {foot_bbox}")

    # For soft tissue, use threshold > 30 in the foot region
    soft_mask = img > 30
    fg_indices = np.where(soft_mask.sum(axis=0) > 10)[0]  # columns with significant content

    dorsal_y = []
    plantar_y = []
    positions = []
    for x in fg_indices:
        col_fg = np.where(soft_mask[:, x])[0]
        if len(col_fg) > 10:
            # Soft tissue extends slightly beyond bone
            # Find where the intensity drops to near background
            col_vals = img[:, x]
            # Dorsal: from top, find where it transitions to background
            dorsal_idx = col_fg[0]
            # Plantar: from bottom, find where it transitions to background
            plantar_idx = col_fg[-1]
            # Extend slightly to include soft tissue beyond bright bone
            # Go upward from first bone pixel to find soft tissue edge
            bone_indices = np.where(col_vals > 80)[0]
            if len(bone_indices) > 0:
                # Dorsal: go up from bone to find where intensity drops below 25
                for dy in range(bone_indices[0], -1, -1):
                    if col_vals[dy] < 25:
                        dorsal_idx = dy + 1
                        break
                # Plantar: go down from bone
                for dy in range(bone_indices[-1], h):
                    if col_vals[dy] < 25:
                        plantar_idx = dy - 1
                        break
            dorsal_y.append(dorsal_idx)
            plantar_y.append(plantar_idx)
            positions.append(x)

    if len(positions) > 10:
        positions = np.array(positions)
        dorsal_y = np.array(dorsal_y)
        plantar_y = np.array(plantar_y)
        thickness = plantar_y - dorsal_y

        print(f"\nSoft tissue profile:")
        print(f"  Points: {len(positions)}")
        print(f"  Thickness range: {thickness.min()}-{thickness.max()} px")
        print(f"  Mean thickness: {thickness.mean():.0f} px")
        print(f"  Std thickness: {thickness.std():.0f} px")

        third = len(positions) // 3
        mid = len(positions) // 2
        two_third = 2 * len(positions) // 3
        print(f"  At 1/3: {thickness[third]} px")
        print(f"  At 1/2: {thickness[mid]} px")
        print(f"  At 2/3: {thickness[two_third]} px")

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        axes[0].imshow(img, cmap="gray")
        axes[0].plot(positions, dorsal_y, 'r-', linewidth=1.5, label='Dorsal soft tissue')
        axes[0].plot(positions, plantar_y, 'g-', linewidth=1.5, label='Plantar soft tissue')
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

print("\n=== FINAL CORRECTIONS COMPLETE ===")
