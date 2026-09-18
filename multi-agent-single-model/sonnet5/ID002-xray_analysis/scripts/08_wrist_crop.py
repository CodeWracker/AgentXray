"""Analysis 8: local contrast stretch of the cropped bottom-edge carpal fragment."""
import numpy as np
from PIL import Image
from common import ORIGINAL, IMAGES_DIR, WRIST_BOX
import os

im = np.array(Image.open(ORIGINAL).convert("L")).astype(np.float64)
crop = im[WRIST_BOX[1]:WRIST_BOX[3], WRIST_BOX[0]:WRIST_BOX[2]]

lo, hi = np.percentile(crop, [1, 99])
stretched = (crop - lo) * (255.0 / max(hi - lo, 1e-6))
stretched = np.clip(stretched, 0, 255).astype(np.uint8)

h, w = stretched.shape
stretched_img = Image.fromarray(stretched).resize((w * 3, h * 3), Image.BICUBIC)
orig_img = Image.fromarray(crop.astype(np.uint8)).resize((w * 3, h * 3), Image.BICUBIC)

combo = Image.new("L", (w * 6, h * 3))
combo.paste(orig_img, (0, 0))
combo.paste(stretched_img, (w * 3, 0))

out = os.path.join(IMAGES_DIR, "08_wrist_fragment_crop_contrast.png")
combo.save(out)
print("wrote", out, combo.size)
