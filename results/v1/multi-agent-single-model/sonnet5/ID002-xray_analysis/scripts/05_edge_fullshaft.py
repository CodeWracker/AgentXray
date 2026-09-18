"""Analysis 5: unsharp-mask edge enhancement along the full bone shafts."""
import cv2
import numpy as np
from PIL import Image
from common import ORIGINAL, IMAGES_DIR
import os

im = np.array(Image.open(ORIGINAL).convert("L")).astype(np.float32)

blurred = cv2.GaussianBlur(im, (0, 0), sigmaX=3)
amount = 1.5
sharpened = im + amount * (im - blurred)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

out = os.path.join(IMAGES_DIR, "05_edge_enhancement_fullshaft.png")
Image.fromarray(sharpened).save(out)
print("wrote", out, sharpened.shape)
