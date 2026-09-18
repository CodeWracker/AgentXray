import cv2
import numpy as np
import os
from PIL import Image

def main():
    input_path = "/home/ralph/projects/ufsc/pablo-xray-tests/llm-xray-evaluation/inputs/ID001-xray.png"
    output_dir = "ID001-xray_analysis"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load original
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image at {input_path}")
        return

    # 01_original.png
    cv2.imwrite(os.path.join(output_dir, "01_original.png"), img)

    # 02_grayscale.png
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_dir, "02_grayscale.png"), gray)

    # 03_contrast_enhanced.png (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, "03_contrast_enhanced.png"), contrast_enhanced)

    # 04_sharpened.png
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    cv2.imwrite(os.path.join(output_dir, "04_sharpened.png"), sharpened)

    # 05_edge_enhanced.png (Canny edges overlayed)
    edges = cv2.Canny(contrast_enhanced, 50, 150)
    edge_enhanced = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    edge_enhanced[edges > 0] = [0, 255, 0] # Green edges
    # Blend edges with original for visibility
    blended_edges = cv2.addWeighted(img, 0.7, edge_enhanced, 0.3, 0)
    cv2.imwrite(os.path.join(output_dir, "05_edge_enhanced.png"), blended_edges)

    # 06_zoom_region_1.png (Ankle joint area)
    h, w = gray.shape
    # Rough estimate of ankle area
    y1, y2, x1, x2 = int(h*0.4), int(h*0.7), int(w*0.3), int(w*0.7)
    zoom1 = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(output_dir, "06_zoom_region_1.png"), zoom1)

    # 07_zoom_region_2.png (Calcaneus/Heel area)
    y1, y2, x1, x2 = int(h*0.6), int(h*0.85), int(w*0.1), int(w*0.4)
    zoom2 = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(output_dir, "07_zoom_region_2.png"), zoom2)

    print("Image processing complete.")

if __name__ == "__main__":
    main()
