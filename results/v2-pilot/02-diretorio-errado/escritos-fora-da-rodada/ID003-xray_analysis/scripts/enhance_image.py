import cv2
import numpy as np
import os
import json

def run_analysis():
    image_path = "images/00_original.png"
    output_dir = "images"
    measurements_dir = "measurements"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(measurements_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    
    # Save enhanced image
    cv2.imwrite(os.path.join(output_dir, "enhanced_contrast.png"), enhanced_img)

    # Sobel edge detection to highlight fractures
    sobelx = cv2.Sobel(enhanced_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(enhanced_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    
    cv2.imwrite(os.path.join(output_dir, "sobel_edges.png"), sobel_mag)

    # Save some basic stats
    stats = {
        "mean": float(np.mean(img)),
        "std": float(np.std(img)),
        "max": float(np.max(img)),
        "min": float(np.min(img))
    }
    with open(os.path.join(measurements_dir, "image_stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

if __name__ == "__main__":
    run_analysis()
