import numpy as np
from PIL import Image

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests"
ORIG = f"{BASE}/ID002-xray.png"
IMAGES = f"{BASE}/ID002-xray_analysis/images"


def load_gray():
    return np.array(Image.open(ORIG).convert("L"), dtype=np.float64)


def save_gray(arr, name, comment=""):
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    p = f"{IMAGES}/{name}"
    Image.fromarray(arr, mode="L").save(p, format="PNG")
    print(f"saved {p} shape={arr.shape} {comment}")
    return p


def save_rgb(arr, name, comment=""):
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    p = f"{IMAGES}/{name}"
    Image.fromarray(arr, mode="RGB").save(p, format="PNG")
    print(f"saved {p} shape={arr.shape} {comment}")
    return p


def crop(arr, box):
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1].copy()


def upscale_nearest(arr, scale):
    return np.kron(arr, np.ones((scale, scale), dtype=np.uint8)) if arr.dtype == np.uint8 \
        else np.repeat(np.repeat(arr, scale, axis=0), scale, axis=1)


def clahe(gray, clip_limit=2.5, tile=8):
    import cv2
    g = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return eq.apply(g).astype(np.float64)


def window_stretch(gray, lo, hi):
    out = (gray - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255)
