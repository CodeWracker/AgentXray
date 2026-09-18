import cv2
import numpy as np
import os

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(BASE), "ID002-xray.png")

color = cv2.imread(SRC, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
print(f"loaded {SRC} shape={color.shape} min={gray.min()} max={gray.max()}")

def save(name, img):
    path = os.path.join(BASE, name)
    ok = cv2.imwrite(path, img)
    print(name, "saved" if ok else "FAILED", img.shape)
    return path

# 01 original (as-is, RGB)
save("01_original.png", cv2.cvtColor(color, cv2.COLOR_BGR2RGB))

# 02 grayscale
save("02_grayscale.png", gray)

# 03 contrast enhanced (CLAHE, preserves detail)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
contrast = clahe.apply(gray)
save("03_contrast_enhanced.png", contrast)

# 04 sharpened (unsharp mask)
blur = cv2.GaussianBlur(gray, (0, 0), 3)
sharpened = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
save("04_sharpened.png", sharpened)

# 05 edge enhanced (Sobel gradient magnitude)
sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
mag = cv2.magnitude(sx, sy)
mag = np.clip(mag / mag.max() * 255, 0, 255).astype(np.uint8)
save("05_edge_enhanced.png", mag)

# 06 zoom region 1: elbow joint (proximal)
z1 = contrast[40:700, 620:1280]
z1 = cv2.resize(z1, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
z1b = cv2.cvtColor(z1, cv2.COLOR_GRAY2BGR)
cv2.putText(z1b, "elbow joint", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
save("06_zoom_region_1.png", z1b)

# 07 zoom region 2: distal forearm / wrist
z2 = contrast[1000:1280, 560:1120]
z2 = cv2.resize(z2, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
z2b = cv2.cvtColor(z2, cv2.COLOR_GRAY2BGR)
cv2.putText(z2b, "distal forearm / wrist", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
save("07_zoom_region_2.png", z2b)

# 08 additional: bone window (strong CLAHE on proximal half) + full soft-tissue gamma
bone = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16)).apply(gray)
soft = (255 * np.power(gray / 255.0, 2.2)).astype(np.uint8)
combo = np.hstack([bone[:, : w // 2], soft[:, : w // 2]])
cv2.putText(combo, "bone window (L) / soft tissue window (R)", (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
save("08_additional_enhancement.png", combo)

print("done")
