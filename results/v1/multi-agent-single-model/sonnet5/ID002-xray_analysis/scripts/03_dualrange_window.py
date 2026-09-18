"""Analysis 3: whole-image piecewise/dual-range intensity remap.

Splits the histogram into a soft-tissue (low/mid) band and a bone (high)
band at the empirical valley between them, then stretches the soft-tissue
band aggressively while compressing the already-saturated bone band.
"""
import numpy as np
from PIL import Image
from common import ORIGINAL, IMAGES_DIR
import os

im = np.array(Image.open(ORIGINAL).convert("L")).astype(np.float64)

# Empirical split point: histogram valley between soft-tissue and bone bands
# (see 04_intensity_histogram.py for the underlying distribution).
SPLIT = 150

soft = im < SPLIT
bone = ~soft

out = np.zeros_like(im)
# Soft tissue band: full 0-255 stretch of [0, SPLIT)
out[soft] = im[soft] * (255.0 / SPLIT)
# Bone band: compress [SPLIT, 255] into a narrower bright band, still above
# the soft-tissue range so the two remain visually distinguishable.
out[bone] = 200 + (im[bone] - SPLIT) * (55.0 / (255 - SPLIT))

out = np.clip(out, 0, 255).astype(np.uint8)
outfile = os.path.join(IMAGES_DIR, "03_dualrange_window.png")
Image.fromarray(out).save(outfile)
print("wrote", outfile, out.shape)
