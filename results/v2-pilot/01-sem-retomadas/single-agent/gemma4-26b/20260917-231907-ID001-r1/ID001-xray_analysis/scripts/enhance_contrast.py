import cv2
import numpy as np
import os

def run_analysis():
    image_path = "images/00_original.png"
    output_dir = "images"
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image.")
        return

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_img = clahe.apply(img)

    # Save enhanced image
    cv2.imwrite(os.path.join(output_dir, "enhanced_clahe.png"), enhanced_img)
    print("Enhanced image saved to images/enhanced_clahe.png")

if __name__ == "__main__":
    run_analysis()
