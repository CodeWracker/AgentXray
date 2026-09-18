"""Analysis 2: ROI-limited CLAHE on the elbow joint region for fat-pad assessment."""
import cv2
import numpy as np
from PIL import Image
from common import ORIGINAL, IMAGES_DIR, JOINT_BOX
import os

im_gray = np.array(Image.open(ORIGINAL).convert("L"))
crop = im_gray[JOINT_BOX[1]:JOINT_BOX[3], JOINT_BOX[0]:JOINT_BOX[2]]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(crop)

h, w = enhanced.shape
enhanced_up = cv2.resize(enhanced, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
orig_up = cv2.resize(crop, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

# side-by-side original vs enhanced for direct comparison
combo = np.concatenate([orig_up, enhanced_up], axis=1)
out = os.path.join(IMAGES_DIR, "02_elbow_fatpad_clahe.png")
Image.fromarray(combo).save(out)
print("wrote", out, combo.shape)
