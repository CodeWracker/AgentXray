"""Shared helpers for the ID003-xray processing pipeline.

All transforms operate in the grayscale (luminance) domain on the original
image. No transform modifies the source file; outputs go to images/.
"""
import os

import cv2
import numpy as np
from PIL import Image

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests/ID003-xray_analysis"
SRC = "/home/ralph/projects/ufsc/pablo-xray-tests/ID003-xray.png"
IMG_DIR = os.path.join(BASE, "images")
PLAN_DIR = os.path.join(BASE, "planning")

W, H = 1280, 1600

# Normalized crop rectangles (x0, x1, y0, y1) from planning/processing_plan.txt
CROPS = {
    "midforearm": (0.40, 0.64, 0.28, 0.50),
    "wrist": (0.40, 0.68, 0.52, 0.80),
    "elbow": (0.42, 0.70, 0.08, 0.34),
    "hand": (0.24, 0.70, 0.58, 1.00),
    "lowerright": (0.38, 0.95, 0.72, 1.00),
}


def load_gray():
    im = Image.open(SRC).convert("L")
    a = np.asarray(im, dtype=np.float32)
    assert a.shape == (H, W), f"unexpected source shape {a.shape}"
    return a


def crop_px(g, x0n, x1n, y0n, y1n):
    xa, xb = int(round(x0n * W)), int(round(x1n * W))
    ya, yb = int(round(y0n * H)), int(round(y1n * H))
    return g[ya:yb, xa:xb], (xa, xb, ya, yb)


def crop_name(g, name):
    return crop_px(g, *CROPS[name])


def save(g, name, mode="L"):
    arr = np.clip(g, 0, 255).astype(np.uint8)
    if mode == "RGB":
        img = Image.fromarray(arr, "RGB")
    else:
        img = Image.fromarray(arr, "L")
    path = os.path.join(IMG_DIR, name)
    img.save(path)
    print(f"[ok] {name}  {arr.shape[1]}x{arr.shape[0]} {mode}")
    return path


def pct_stretch(g, lo=1.0, hi=99.5):
    a, b = np.percentile(g, [lo, hi])
    return np.clip((g - a) / (b - a), 0.0, 1.0) * 255.0


def window(g, plo, phi):
    a, b = np.percentile(g, [plo, phi])
    return np.clip((g - a) / (b - a), 0.0, 1.0) * 255.0


def clahe(g, clip=2.5, tiles=8):
    u8 = np.clip(g, 0, 255).astype(np.uint8)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    return cl.apply(u8).astype(np.float32)


def nn_upscale(arr, factor):
    h, w = arr.shape
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    im = im.resize((int(round(w * factor)), int(round(h * factor))), Image.NEAREST)
    return np.asarray(im, dtype=np.float32)


def unsharp(g, sigma=2.0, amount=0.7):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return g + amount * (g - blur)


def highpass(g, sigma=15.0):
    blur = cv2.GaussianBlur(g, (0, 0), sigma)
    return g - blur


def gray_centered_hp(hp, scale=3.0):
    out = 128.0 + hp * scale
    return np.clip(out, 0, 255)


def append_measurements(text):
    path = os.path.join(PLAN_DIR, "measurements_report.txt")
    with open(path, "a") as f:
        f.write(text)
    print(f"[ok] appended measurements -> {path}")
