import os
import numpy as np
import cv2
from PIL import Image

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID001-xray_analysis"
IMG_DIR = os.path.join(BASE, "images")
ORIG = os.path.join(IMG_DIR, "ID001-xray.png")


def load_gray():
    img = Image.open(ORIG).convert("L")
    return np.array(img, dtype=np.uint8)


def load_rgb():
    return np.array(Image.open(ORIG), dtype=np.uint8)


def clahe(gray, clip=3.0, tile=(8, 8)):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(gray)


def crop_zoom(gray, box, z=2):
    x0, y0, x1, y1 = box
    c = gray[y0:y1, x0:x1]
    if z != 1:
        c = cv2.resize(c, (c.shape[1] * z, c.shape[0] * z), interpolation=cv2.INTER_NEAREST)
    return c


def save(name, arr):
    path = os.path.join(IMG_DIR, name)
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").convert("RGB").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
    print("saved", path, arr.shape)
    return path


def unsharp(gray, sigma=2.0, amount=2.0):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    return np.clip(gray.astype(np.int16) + amount * (gray.astype(np.int16) - blurred.astype(np.int16)), 0, 255).astype(np.uint8)
