import os

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "ID001-xray.png")


def save(img, name):
    path = os.path.join(HERE, name)
    cv2.imwrite(path, img)
    print("saved", name, img.shape)
    return path


orig = cv2.imread(SRC)
assert orig is not None, "failed to load original"

# 01 original (unmodified)
save(orig, "01_original.png")

# 02 grayscale
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
save(gray, "02_grayscale.png")

# 03 contrast enhanced (CLAHE on grayscale)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)
save(enhanced, "03_contrast_enhanced.png")

# 04 sharpened (unsharp mask on grayscale)
blur = cv2.GaussianBlur(enhanced, (5, 5), 1.5)
sharpened = cv2.addWeighted(enhanced, 2.0, blur, -1.0, 0)
save(sharpened, "04_sharpened.png")

# 05 edge enhanced (Canny on CLAHE image, slight dilation for visibility)
edges = cv2.Canny(enhanced, 50, 150)
edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
save(edges, "05_edge_enhanced.png")


def zoom(src_img, x0f, y0f, x1f, y1f, name, scale=3):
    h, w = src_img.shape[:2]
    x0, x1 = int(w * x0f), int(w * x1f)
    y0, y1 = int(h * y0f), int(h * y1f)
    crop = src_img[y0:y1, x0:x1]
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    save(crop, name)
    return (x0, y0, x1, y1)


# 06 zoom region 1: calcaneus / heel / plantar surface (on contrast-enhanced)
r1 = zoom(enhanced, 0.08, 0.52, 0.62, 0.97, "06_zoom_region_1.png")
print("region1 (calcaneus/heel) px:", r1)

# 07 zoom region 2: ankle joint / tibiotalar / subtalar (on contrast-enhanced)
r2 = zoom(enhanced, 0.38, 0.28, 0.90, 0.72, "07_zoom_region_2.png")
print("region2 (ankle joint) px:", r2)

# 08 additional enhancement: gamma correction to bring out soft tissue detail
gamma = 0.5
lut = np.array(
    [((i / 255.0) ** gamma) * 255 for i in range(256)]
).astype(np.uint8)
gamma_img = cv2.LUT(gray, lut)
save(gamma_img, "08_additional_enhancement.png")

print("done")
