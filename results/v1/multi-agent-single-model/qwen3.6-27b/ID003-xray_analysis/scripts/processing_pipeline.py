import cv2
import numpy as np
from pathlib import Path
import sys

BASE = Path("/home/ralph/projects/ufsc/pablo-xray-tests-gemma4/ID003-xray_analysis")
IMAGES = BASE / "images"
ORIGINAL = IMAGES / "ID003-xray_original.png"

def load_image():
    img = cv2.imread(str(ORIGINAL), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: Cannot load {ORIGINAL}")
        sys.exit(1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    return gray

def save_image(result, filename, info=""):
    out = IMAGES / filename
    cv2.imwrite(str(out), result)
    h, w = result.shape[:2]
    print(f"  Saved {filename}: {w}x{h} {info}")

def get_fracture_roi(gray):
    h, w = gray.shape
    frac_y1, frac_y2 = int(h * 0.25), int(h * 0.55)
    frac_x1, frac_x2 = int(w * 0.30), int(w * 0.72)
    return gray[frac_y1:frac_y2, frac_x1:frac_x2], (frac_x1, frac_y1, frac_x2, frac_y2)

def get_wrist_roi(gray):
    h, w = gray.shape
    wrist_y1, wrist_y2 = int(h * 0.55), int(h * 0.95)
    wrist_x1, wrist_x2 = int(w * 0.20), int(w * 0.70)
    return gray[wrist_y1:wrist_y2, wrist_x1:wrist_x2], (wrist_x1, wrist_y1, wrist_x2, wrist_y2)

def get_growth_plate_roi(gray):
    h, w = gray.shape
    gp_y1, gp_y2 = int(h * 0.55), int(h * 0.72)
    gp_x1, gp_x2 = int(w * 0.35), int(w * 0.68)
    return gray[gp_y1:gp_y2, gp_x1:gp_x2], (gp_x1, gp_y1, gp_x2, gp_y2)

# Analysis 1: Fracture Zone CLAHE
def analysis_1_clahe(gray):
    print("Analysis 1: Fracture Zone CLAHE Enhancement")
    roi, coords = get_fracture_roi(gray)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    roi_clahe = clahe.apply(roi)
    h_roi, w_roi = roi.shape
    result = gray.copy()
    x1, y1, x2, y2 = coords
    result[y1:y2, x1:x2] = roi_clahe
    overlay = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(overlay, "CLAHE ROI (clip=3.0, 8x8)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    save_image(overlay, "01_fracture_zone_clahe.png", "CLAHE enhanced fracture zone")

    roi_combined = np.hstack([roi, roi_clahe])
    save_image(roi_combined, "01_fracture_zone_clahe_sidebyside.png", "side-by-side comparison")
    return roi, roi_clahe

# Analysis 2: Cortical Edge Detection
def analysis_2_edges(gray, frac_roi):
    print("Analysis 2: Cortical Edge Detection Overlay")
    _, coords = get_fracture_roi(gray)
    x1, y1, x2, y2 = coords
    roi = gray[y1:y2, x1:x2].copy()
    blurred = cv2.GaussianBlur(roi, (5, 5), 1.5)
    low_thresh = np.percentile(blurred, 85)
    high_thresh = np.percentile(blurred, 95)
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    mask = edges > 0
    result_color[y1:y2, x1:x2][mask] = edge_color[mask]
    cv2.putText(result_color, "Canny Edges (p85-p95)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    save_image(result_color, "02_cortical_edge_overlay.png", "Canny edge overlay")
    save_image(edges, "02_cortical_edges_raw.png", "raw edge detection")

# Analysis 3: Displacement Measurement
def analysis_3_displacement(gray):
    print("Analysis 3: Displacement Measurement with Axis Overlay")
    h, w = gray.shape
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    roi_gray = gray[int(h*0.2):int(h*0.65), int(w*0.33):int(w*0.68)].copy()
    _, binary = cv2.threshold(roi_gray, np.percentile(roi_gray, 80), 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        result_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(result_color, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    save_image(result_color, "03_bone_bounding_box.png", "bone ROI detection")

    full_result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frac_roi, coords = get_fracture_roi(gray)
    x1, y1, x2, y2 = coords
    binary_full = gray[y1:y2, x1:x2] > np.percentile(gray[y1:y2, x1:x2], 80)
    for row in range(0, binary_full.shape[0], 5):
        cols = np.where(binary_full[row, :])[0]
        if len(cols) > 10:
            mid_x = np.mean(cols) + x1
            mid_y = row + y1
            cv2.circle(full_result, (int(mid_x), int(mid_y)), 1, (0, 255, 255), -1)
    cv2.putText(full_result, "Bone centerline estimation", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    save_image(full_result, "03_displacement_measurement.png", "centerline estimation")

    y_coords_bone = []
    for row in range(0, binary_full.shape[0]):
        cols = np.where(binary_full[row, :])[0]
        if len(cols) > 10:
            left_edge = cols[0]
            right_edge = cols[-1]
            y_coords_bone.append((row, left_edge, right_edge))
    if len(y_coords_bone) > 100:
        mid_point = len(y_coords_bone) // 2
        proximal = y_coords_bone[:mid_point]
        distal = y_coords_bone[mid_point:]
        proximal_centers = [np.mean([p[1], p[2]]) for p in proximal]
        distal_centers = [np.mean([d[1], d[2]]) for d in distal]
        proximal_angles = []
        for i in range(1, len(proximal_centers)):
            dx = proximal_centers[i] - proximal_centers[i-1]
            angle = np.degrees(np.arctan2(dx, 1))
            proximal_angles.append(angle)
        distal_angles = []
        for i in range(1, len(distal_centers)):
            dx = distal_centers[i] - distal_centers[i-1]
            angle = np.degrees(np.arctan2(dx, 1))
            distal_angles.append(angle)
        if proximal_angles and distal_angles:
            avg_prox = np.mean(proximal_angles)
            avg_dist = np.mean(distal_angles)
            angulation = abs(avg_prox - avg_dist)
            cv2.putText(full_result, f"Angulation: ~{angulation:.1f} deg", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            print(f"  Estimated angulation: {angulation:.1f} degrees")

# Analysis 4: Zoomed Fracture Zone
def analysis_4_zoom(gray):
    print("Analysis 4: Zoomed Fracture Zone Detail")
    roi, _ = get_fracture_roi(gray)
    h_roi, w_roi = roi.shape
    scale = 3
    zoomed = cv2.resize(roi, (w_roi * scale, h_roi * scale), interpolation=cv2.INTER_LANCZOS4)
    amount = 0.5
    gaussian = cv2.GaussianBlur(zoomed, (0, 0), 2)
    sharpened = cv2.addWeighted(zoomed, 1 + amount, gaussian, -amount, 0)
    save_image(sharpened, "04_fracture_zone_zoomed.png", f"3x zoomed + unsharp mask")
    save_image(zoomed, "04_fracture_zone_zoom_only.png", "3x zoomed (no sharpening)")

# Analysis 5: Growth Plate Enhancement
def analysis_5_growth_plate(gray):
    print("Analysis 5: Growth Plate / Physis Enhancement")
    roi, coords = get_growth_plate_roi(gray)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6))
    roi_enhanced = clahe.apply(roi)
    x1, y1, x2, y2 = coords
    result = gray.copy()
    result[y1:y2, x1:x2] = roi_enhanced
    result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_color, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(result_color, "Growth Plate CLAHE (clip=2.5)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    save_image(result_color, "05_growth_plate_enhanced.png", "growth plate CLAHE")
    roi_combined = np.hstack([roi, roi_enhanced])
    save_image(roi_combined, "05_growth_plate_sidebyside.png", "side-by-side growth plate")

# Analysis 6: Soft Tissue Profile
def analysis_6_soft_tissue(gray):
    print("Analysis 6: Soft Tissue Swelling Profile")
    h, w = gray.shape
    high_thresh = np.percentile(gray, 92)
    bone_mask = gray > high_thresh
    filled_mask = bone_mask.astype(np.uint8)
    kernel = np.ones((11, 11), np.uint8)
    dilated = cv2.dilate(filled_mask, kernel, iterations=3)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    soft_tissue = gray.copy().astype(np.float32)
    soft_tissue[eroded > 0] = 0
    soft_tissue = np.clip(soft_tissue, 0, 255).astype(np.uint8)
    st_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    soft_tissue_enhanced = st_clahe.apply(soft_tissue)
    result_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    soft_bgr = cv2.cvtColor(soft_tissue_enhanced, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([result_color, soft_bgr])
    cv2.putText(combined, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, "Soft Tissue Only", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    save_image(combined, "06_soft_tissue_profile.png", "soft tissue isolation")

    profile = np.zeros((h, 3))
    for row in range(h):
        bone_cols = np.where(bone_mask[row, :])[0]
        if len(bone_cols) > 0:
            left = bone_cols[0]
            right = bone_cols[-1]
            soft_left = 0
            for c in range(left - 1, -1, -1):
                if soft_tissue_enhanced[row, c] < 10:
                    break
                soft_left = c
            soft_right = w - 1
            for c in range(right + 1, w):
                if soft_tissue_enhanced[row, c] < 10:
                    soft_right = c - 1
                    break
            total_width = soft_right - soft_left + 1
            profile[row, 0] = total_width
            profile[row, 1] = soft_left
            profile[row, 2] = soft_right
    save_image(soft_tissue_enhanced, "06_soft_tissue_only.png", "isolated soft tissue")

# Analysis 7: Wrist/Hand Enhancement
def analysis_7_wrist(gray):
    print("Analysis 7: Wrist/Hand Regional Enhancement")
    roi, coords = get_wrist_roi(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_enhanced = clahe.apply(roi)
    x1, y1, x2, y2 = coords
    result = gray.copy()
    result[y1:y2, x1:x2] = roi_enhanced
    result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_color, (x1, y1), (x2, y2), (255, 255, 0), 2)
    cv2.putText(result_color, "Wrist CLAHE (clip=2.0)", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    save_image(result_color, "07_wrist_hand_enhanced.png", "wrist/hand CLAHE")
    roi_combined = np.hstack([roi, roi_enhanced])
    save_image(roi_combined, "07_wrist_sidebyside.png", "side-by-side wrist")

# Analysis 8: Periosteal/Chronicity Assessment
def analysis_8_periosteal(gray):
    print("Analysis 8: Periosteal Reaction / Chronicity Assessment")
    roi, coords = get_fracture_roi(gray)
    h_roi, w_roi = roi.shape
    kernel_size = 5
    sobel_v = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=kernel_size)
    sobel_h = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=kernel_size)
    abs_sobel_v = np.absolute(sobel_v)
    abs_sobel_h = np.absolute(sobel_h)
    combined = abs_sobel_v + abs_sobel_h
    combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    x1, y1, x2, y2 = coords
    result = gray.copy()
    result[y1:y2, x1:x2] = combined_norm
    result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result_color, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv2.putText(result_color, "Periosteal Gradient Map", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    save_image(result_color, "08_periosteal_chronicity.png", "periosteal gradient analysis")
    roi_combined = np.hstack([roi, combined_norm])
    save_image(roi_combined, "08_periosteal_sidebyside.png", "side-by-side periosteal")

    for dx in [-1, 1]:
        for dy in [-1, 1]:
            gabor = cv2.getGaborKernel((21, 21), 4.0, np.arctan2(dy, dx), 5.0, 0.5)
            gabor_response = cv2.filter2D(roi.astype(np.float32), cv2.CV_32F, gabor)
            gabor_norm = cv2.normalize(np.abs(gabor_response), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            angle = np.degrees(np.arctan2(dy, dx))
            filename = f"08_periosteal_gabor_{int(angle)}.png"
            save_image(gabor_norm, filename, f"Gabor filter at {angle:.0f} deg")

# Analysis 9: Full-Image Bone Window
def analysis_9_full_window(gray):
    print("Analysis 9: Full-Image Bone Window Optimization")
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gamma = 0.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(normalized, table)
    save_image(gamma_corrected, "09_full_bone_window.png", "gamma=0.8 optimized bone window")

# Main execution
def main():
    print("=" * 60)
    print("X-RAY IMAGE PROCESSING PIPELINE")
    print(f"Model: qwen3.6-27b")
    print(f"Image: ID003-xray.png")
    print("=" * 60)
    gray = load_image()
    h, w = gray.shape
    print(f"Loaded image: {w}x{h}")
    a1_roi, a1_clahe = analysis_1_clahe(gray)
    analysis_2_edges(gray, a1_roi)
    analysis_3_displacement(gray)
    analysis_4_zoom(gray)
    analysis_5_growth_plate(gray)
    analysis_6_soft_tissue(gray)
    analysis_7_wrist(gray)
    analysis_8_periosteal(gray)
    analysis_9_full_window(gray)
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    expected = [
        "01_fracture_zone_clahe.png",
        "01_fracture_zone_clahe_sidebyside.png",
        "02_cortical_edge_overlay.png",
        "02_cortical_edges_raw.png",
        "03_displacement_measurement.png",
        "03_bone_bounding_box.png",
        "04_fracture_zone_zoomed.png",
        "04_fracture_zone_zoom_only.png",
        "05_growth_plate_enhanced.png",
        "05_growth_plate_sidebyside.png",
        "06_soft_tissue_profile.png",
        "06_soft_tissue_only.png",
        "07_wrist_hand_enhanced.png",
        "07_wrist_sidebyside.png",
        "08_periosteal_chronicity.png",
        "08_periosteal_sidebyside.png",
        "09_full_bone_window.png",
    ]
    missing = []
    for f in expected:
        fp = IMAGES / f
        if not fp.exists():
            missing.append(f)
            print(f"  MISSING: {f}")
        else:
            try:
                check = cv2.imread(str(fp))
                if check is None:
                    missing.append(f)
                    print(f"  UNOPENABLE: {f}")
            except Exception as e:
                missing.append(f)
                print(f"  ERROR: {f}: {e}")
    if missing:
        print(f"\nWARNING: {len(missing)} files missing or invalid!")
        return 1
    print("\nAll expected files verified successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
