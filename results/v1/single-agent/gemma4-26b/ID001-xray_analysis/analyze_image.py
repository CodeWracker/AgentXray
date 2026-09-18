import cv2
import numpy as np
import os

def analyze_image(image_path, output_folder):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 01_original.png
    cv2.imwrite(os.path.join(output_folder, '01_original.png'), img)

    # 02_grayscale.png
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, '02_grayscale.png'), gray)

    # 03_contrast_enhanced.png (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_folder, '03_contrast_enhanced.png'), contrast_enhanced)

    # 04_sharpened.png
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    cv2.imwrite(os.path.join(output_folder, '04_sharpened.png'), sharpened)

    # 05_edge_enhanced.png (Sobel)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.magnitude(sobelx, sobely)
    edges_normalized = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, '05_edge_enhanced.png'), edges_normalized)

    # 06_zoom_region_1.png (Ankle Joint)
    # Based on visual inspection, ankle joint is around middle
    h, w = gray.shape
    center_y, center_x = int(h * 0.55), int(w * 0.5)
    zoom_size = 200
    y1, y2 = max(0, center_y - zoom_size), min(h, center_y + zoom_size)
    x1, x2 = max(0, center_x - zoom_size), min(w, center_x + zoom_size)
    zoom1 = img[y1:y2, x1:x2]
    cv2.imwrite(os.path.join(output_folder, '06_zoom_region_1.png'), zoom1)

    # 07_zoom_region_2.png (Calcaneus/Heel)
    center_y_heel, center_x_heel = int(h * 0.7), int(w * 0.4)
    y1_h, y2_h = max(0, center_y_heel - zoom_size), min(h, center_y_heel + zoom_size)
    x1_h, x2_h = max(0, center_x_heel - zoom_size), min(w, center_x_heel + zoom_size)
    zoom2 = img[y1_h:y2_h, x1_h:x2_h]
    cv2.imwrite(os.path.join(output_folder, '07_zoom_region_2.png'), zoom2)

if __name__ == "__main__":
    analyze_image('ID001-xray.png', 'ID001-xray_analysis')
