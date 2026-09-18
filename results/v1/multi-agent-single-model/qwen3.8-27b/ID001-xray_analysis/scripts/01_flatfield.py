"""[1] Flat-field / de-vignette correction of the full image + [14] right-edge crop from it."""
import os
import sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_gray, save

g = load_gray().astype(np.float32)

# background (exposure-gradient) estimate: large Gaussian blur; bone pixels are
# replaced iteratively so the estimate reflects the beam field, not the bone
bg = cv2.GaussianBlur(g, (0, 0), 40)
for _ in range(3):
    bone_mask = g > 0.8 * bg
    g_nobone = np.where(bone_mask, bg, g).astype(np.float32)
    bg = cv2.GaussianBlur(g_nobone, (0, 0), 40)
bg = np.maximum(bg, 1.0)
# scale constant: level of the CENTRAL beam field (75th percentile of the beam-field
# estimate), so central bone density is preserved and only the exposure
# non-uniformity is divided out, brightening the underexposed periphery.
# (v1 review round: agents 03/06 flagged that normalizing to the global image
#  median crushed central bone density.)
scale = float(np.percentile(bg, 75))
corrected = np.clip(g / bg * scale, 0, 255).astype(np.uint8)

save("01_flatfield_corrected_full.png", corrected)

# characterize the background for the record
print("background stats: median", round(float(np.median(g)), 1))
print("corrected range", corrected.min(), corrected.max())
print("bg range", bg.min().round(1), bg.max().round(1))

# [14] right-edge zone from the corrected image
x0, y0, x1, y1 = 250, 255, 373, 345
crop = corrected[y0:y1, x0:x1]
crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_NEAREST)
save("14_right_edge_crop_flatfield.png", crop)
