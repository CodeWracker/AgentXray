import cv2
import numpy as np
from PIL import Image

def analyze_image(input_path, output_dir):
    # Load the original image
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image {input_path}")
        return

    # 01_original.png
    cv2.imwrite(f"{output_dir}/01_original.png", img)

    # 02_grayscale.png
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/02_grayscale.png", gray)

    # 03_contrast_enhanced.png (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    cv2.imwrite(f"{output_dir}/03_contrast_enhanced.png", contrast_enhanced)

    # 04_sharpened.png
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    cv2.imwrite(f"{output_dir}/04_sharpened.png", sharpened)

    # 05_edge_enhanced.png (Canny edges overlaid or just edges)
    edges = cv2.Canny(contrast_enhanced, 50, 150)
    # Create an edge image where edges are white on black
    edge_enhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(f"{output_dir}/05_edge_enhanced.png", edge_enhanced)

    # 06_zoom_region_1.png - Focus on the mid-shaft fracture
    # We'll estimate the area for the mid-shaft based on visual inspection of the original
    # The image is quite large, let's assume standard coordinates or relative
    h, w = gray.shape
    # Mid shaft is roughly in the middle vertically and left-ish horizontally
    # Based on original: fracture is roughly in the middle of the forearm
    # Let's try to find the fracture area more dynamically or just pick a region
    # Manual guess: y from 30% to 50%, x from 40% to 60%
    y_start, y_end = int(h * 0.3), int(h * 0.5)
    x_start, x_end = int(w * 0.4), int(w * 0.6)
    zoom1 = img[y_start:y_end, x_start:x_end]
    if zoom1.size > 0:
        cv2.imwrite(f"{output_dir}/06_zoom_region_1.png", zoom1)

    # 07_zoom_region_2.png - Focus on the wrist/distal area
    y_start2, y_end2 = int(h * 0.6), int(h * 0.9)
    x_start2, x_end2 = int(w * 0.2), int(w * 0.7)
    zoom2 = img[y_start2:y_end2, x_start2:x_end2]
    if zoom2.size > 0:
        cv2.imwrite(f"{output_dir}/07_zoom_region_2.png", zoom2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python analyze_image.py <input_path> <output_dir>")
    else:
        analyze_image(sys.argv[1], sys.argv[2])
