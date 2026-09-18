"""
Image-processing pipeline for second-opinion visualization support.
Generates enhanced intermediate views of a radiograph to aid visual inspection.
Does not alter or destroy original pixel data - all outputs are derived copies.
"""
import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "ID002-xray.png")

def save(name, img):
    cv2.imwrite(os.path.join(BASE_DIR, name), img)
    print(f"saved {name}  shape={img.shape}")

def main():
    original_bgr = cv2.imread(INPUT_PATH, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"Could not load image at {INPUT_PATH}")

    # 01 - original (copy, untouched)
    save("01_original.png", original_bgr)

    # 02 - grayscale
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    save("02_grayscale.png", gray)

    # 03 - contrast enhanced (CLAHE - adaptive histogram equalization)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    save("03_contrast_enhanced.png", contrast_enhanced)

    # 04 - sharpened (unsharp mask on the contrast-enhanced image)
    blurred = cv2.GaussianBlur(contrast_enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(contrast_enhanced, 1.6, blurred, -0.6, 0)
    save("04_sharpened.png", sharpened)

    # 05 - edge enhanced (Canny edges overlaid on grayscale for cortical margins)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    edge_overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge_overlay[edges > 0] = (0, 0, 255)
    save("05_edge_enhanced.png", edge_overlay)

    h, w = gray.shape

    # 06 - zoom region 1: knee joint (proximal tibia/fibula + distal femur articulation)
    # Bright joint cluster is in the upper-left/central portion of the image.
    y0, y1 = int(h * 0.05), int(h * 0.35)
    x0, x1 = int(w * 0.45), int(w * 0.80)
    knee_crop = contrast_enhanced[y0:y1, x0:x1]
    knee_crop_resized = cv2.resize(knee_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    save("06_zoom_region_1.png", knee_crop_resized)

    # 07 - zoom region 2: mid-to-distal tibia/fibula shaft
    y0, y1 = int(h * 0.35), int(h * 0.75)
    x0, x1 = int(w * 0.40), int(w * 0.70)
    shaft_crop = contrast_enhanced[y0:y1, x0:x1]
    shaft_crop_resized = cv2.resize(shaft_crop, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)
    save("07_zoom_region_2.png", shaft_crop_resized)

    # 08 - additional enhancement: bone-window pseudo-color map to highlight density
    # gradations (soft tissue vs cortical vs medullary bone) using a perceptual colormap
    norm = cv2.normalize(contrast_enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colormapped = cv2.applyColorMap(norm, cv2.COLORMAP_BONE)
    save("08_additional_enhancement.png", colormapped)

    print("Done. All outputs written to:", BASE_DIR)

if __name__ == "__main__":
    main()
