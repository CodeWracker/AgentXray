"""Analysis 1: Image enhancement — CLAHE and gamma correction."""
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from skimage import exposure

random.seed(42)
np.random.seed(42)

import os
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(base, 'images', '00_original.png')
img_out = os.path.join(base, 'images')
meas_out = os.path.join(base, 'measurements')

img = Image.open(img_path).convert('L')
arr = np.array(img, dtype=np.float32)

# CLAHE
arr_uint8 = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
clahe_result = clahe.apply(arr_uint8)

# Gamma correction (0.5 — brighten)
gamma_arr = 255 * np.power(arr_uint8 / 255.0, 1.0 / 0.5)
gamma_arr = gamma_arr.astype(np.uint8)

# Histogram equalization
hist_eq = cv2.equalizeHist(arr_uint8)

# Save enhanced images
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.imshow(arr_uint8, cmap='gray')
plt.title('Original (normalized)')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(clahe_result, cmap='gray')
plt.title('CLAHE (clip=3, 8x8)')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(gamma_arr, cmap='gray')
plt.title('Gamma 0.5 (brighten)')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(hist_eq, cmap='gray')
plt.title('Global histogram equalization')
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(img_out, '01_enhanced_composite.png'), dpi=150)
plt.close()

# Save individual enhanced images
Image.fromarray(clahe_result).save(os.path.join(img_out, '02_clahe.png'))
Image.fromarray(gamma_arr).save(os.path.join(img_out, '03_gamma05.png'))
Image.fromarray(hist_eq).save(os.path.join(img_out, '04_histeq.png'))

# Save basic measurements
stats = {
    "original_shape": list(arr.shape),
    "original_min": float(arr.min()),
    "original_max": float(arr.max()),
    "original_mean": float(arr.mean()),
    "original_std": float(arr.std()),
    "clahe_mean": float(clahe_result.mean()),
    "clahe_std": float(clahe_result.std()),
    "gamma_mean": float(gamma_arr.mean()),
    "gamma_std": float(gamma_arr.std()),
    "histeq_mean": float(hist_eq.mean()),
    "histeq_std": float(hist_eq.std()),
}

import json
with open(os.path.join(meas_out, '01_image_statistics.json'), 'w') as f:
    json.dump(stats, f, indent=2)
