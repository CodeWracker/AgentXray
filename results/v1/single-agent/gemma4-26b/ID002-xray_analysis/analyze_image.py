import cv2
import numpy as np
from PIL import Image

def analyze():
    input_path = 'ID002-xray.png'
    output_dir = 'ID002-xray_analysis/'
    
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Could not load image at {input_path}")
        return

    # 01_original.png
    cv2.imwrite(output_dir + '01_original.png', img)

    # 02_grayscale.png
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_dir + '02_grayscale.png', gray)

    # 03_contrast_enhanced.png (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced = clahe.apply(gray)
    cv2.imwrite(output_dir + '03_contrast_enhanced.png', contrast_enhanced)

    # 04_sharpened.png
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel)
    cv2.imwrite(output_dir + '04_sharpened.png', sharpened)

    # 05_edge_enhanced.png (Canny)
    edges = cv2.Canny(contrast_enhanced, 30, 150)
    # Create a color version of edges for consistent format if needed, 
    # but saving as grayscale is fine.
    cv2.imwrite(output_dir + '05_edge_enhanced.png', edges)

    # 06_zoom_region_1.png (Elbow joint area)
    # Based on visual inspection, elbow is roughly at the top
    h, w = gray.shape
    zoom_elbow = img[0:int(h*0.4), int(w*0.4):w]
    if zoom_elbow.size > 0:
        cv2.imwrite(output_dir + '06_zoom_region_1.png', zoom_elbow)

    # 07_zoom_region_2.png (Distal forearm/wrist area)
    zoom_wrist = img[int(h*0.7):h, int(w*0.4):w]
    if zoom_wrist.size > 0:
        cv2.imwrite(output_dir + '07_zoom_region_2.png', zoom_wrist)

    print("Analysis complete. Images generated in " + output_dir)

if __name__ == "__main__":
    analyze()
