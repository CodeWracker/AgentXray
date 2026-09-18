"""Analysis 4: histogram/clipping diagnostic of the bright proximal (joint) ROI vs whole image."""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import ORIGINAL, IMAGES_DIR, JOINT_BOX
import os

im = np.array(Image.open(ORIGINAL).convert("L"))
roi = im[JOINT_BOX[1]:JOINT_BOX[3], JOINT_BOX[0]:JOINT_BOX[2]]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].hist(im.ravel(), bins=256, range=(0, 255), color="steelblue", log=True)
axes[0].set_title("Whole image intensity histogram")
axes[0].set_xlabel("pixel value")
axes[0].set_ylabel("count (log scale)")

axes[1].hist(roi.ravel(), bins=256, range=(0, 255), color="firebrick", log=True)
axes[1].set_title("Proximal joint ROI intensity histogram")
axes[1].set_xlabel("pixel value")

max_bin_count = np.sum(roi == 255)
frac_at_max = max_bin_count / roi.size
fig.suptitle(
    f"Clipping check: {frac_at_max*100:.2f}% of ROI pixels at max value (255)"
)
plt.tight_layout()

out = os.path.join(IMAGES_DIR, "04_intensity_histogram.png")
fig.savefig(out, dpi=120)
print("wrote", out, "frac_at_max=", frac_at_max)
