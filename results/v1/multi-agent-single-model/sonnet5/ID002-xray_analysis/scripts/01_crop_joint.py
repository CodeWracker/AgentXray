"""Analysis 1: unfiltered crop + upscale of the elbow joint region."""
from PIL import Image
from common import ORIGINAL, IMAGES_DIR, JOINT_BOX
import os

im = Image.open(ORIGINAL).convert("RGB")
crop = im.crop(JOINT_BOX)
w, h = crop.size
upscaled = crop.resize((w * 3, h * 3), Image.BICUBIC)
out = os.path.join(IMAGES_DIR, "01_elbow_joint_crop.png")
upscaled.save(out)
print("wrote", out, upscaled.size)
