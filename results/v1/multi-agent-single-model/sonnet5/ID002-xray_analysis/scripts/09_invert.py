"""Analysis 9: inverted-polarity (negative) rendering of the full original image."""
import numpy as np
from PIL import Image, ImageOps
from common import ORIGINAL, IMAGES_DIR
import os

im = Image.open(ORIGINAL).convert("L")
inverted = ImageOps.invert(im)

out = os.path.join(IMAGES_DIR, "09_inverted_polarity.png")
inverted.save(out)
print("wrote", out, inverted.size)
