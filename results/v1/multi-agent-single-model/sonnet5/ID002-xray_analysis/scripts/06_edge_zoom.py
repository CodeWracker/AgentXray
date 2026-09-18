"""Analysis 6: targeted edge enhancement zoomed on radial neck / coronoid / olecranon."""
import cv2
import numpy as np
from PIL import Image
from common import ORIGINAL, IMAGES_DIR, FRACTURE_SITE_BOX
import os

im_full = np.array(Image.open(ORIGINAL).convert("L"))
crop = im_full[FRACTURE_SITE_BOX[1]:FRACTURE_SITE_BOX[3], FRACTURE_SITE_BOX[0]:FRACTURE_SITE_BOX[2]].astype(np.float32)

blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=2)
amount = 1.8
sharpened = crop + amount * (crop - blurred)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

h, w = sharpened.shape
sharpened_up = cv2.resize(sharpened, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
orig_up = cv2.resize(crop.astype(np.uint8), (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
combo = np.concatenate([orig_up, sharpened_up], axis=1)

out = os.path.join(IMAGES_DIR, "06_edge_enhancement_elbow_zoom.png")
Image.fromarray(combo).save(out)
print("wrote", out, combo.shape)
