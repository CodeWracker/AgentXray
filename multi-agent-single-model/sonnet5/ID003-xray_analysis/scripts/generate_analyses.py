#!/usr/bin/env python3
"""Implements planning/processing_plan.txt for ID003-xray.png.

Reads images/00_original.png and writes each planned derived image into images/.
Every transform is applied only to pixel data already present in the source image
(crop / rescale / contrast / edge-detection); nothing is synthesized.
"""
import sys
import numpy as np
import cv2
from PIL import Image

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests-claude/ID003-xray_analysis"
SRC = f"{BASE}/images/00_original.png"
OUT = f"{BASE}/images"

failures = []


def load_gray():
    img = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {SRC}")
    return img


def save(name, arr):
    try:
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(f"{OUT}/{name}", arr)
        if not ok:
            failures.append((name, "cv2.imwrite returned False"))
        else:
            # validate round-trip open
            Image.open(f"{OUT}/{name}").verify()
    except Exception as e:
        failures.append((name, str(e)))


def contrast_stretch(arr, lo_pct=1, hi_pct=99):
    lo, hi = np.percentile(arr, [lo_pct, hi_pct])
    if hi <= lo:
        return arr
    out = (arr.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255)


def upscale(arr, factor):
    h, w = arr.shape[:2]
    return cv2.resize(arr, (w * factor, h * factor), interpolation=cv2.INTER_LANCZOS4)


def unsharp(arr, radius=3, amount=1.5):
    blurred = cv2.GaussianBlur(arr.astype(np.float32), (0, 0), radius)
    sharp = arr.astype(np.float32) + amount * (arr.astype(np.float32) - blurred)
    return np.clip(sharp, 0, 255)


def gamma(arr, g):
    norm = arr.astype(np.float32) / 255.0
    return np.clip((norm ** g) * 255.0, 0, 255)


def crop(arr, box):
    x0, y0, x1, y1 = box
    return arr[y0:y1, x0:x1]


def main():
    img = load_gray()
    h, w = img.shape
    print(f"Loaded original: {w}x{h}")

    # 1. fracture zone zoom + contrast
    box1 = (560, 560, 820, 720)
    c1 = crop(img, box1)
    c1 = upscale(c1, 4)
    c1 = contrast_stretch(c1)
    save("01_fracture_zone_zoom_contrast.png", c1)

    # 2. fracture zone edge enhancement (unsharp)
    c2 = crop(img, box1)
    c2 = upscale(c2, 4)
    c2 = unsharp(c2, radius=3, amount=1.5)
    save("02_fracture_zone_edge_enhancement.png", c2)

    # 3. hand region deblur/contrast
    box3 = (350, 1000, 850, 1600)
    c3 = crop(img, box3)
    c3 = upscale(c3, 3)
    c3 = unsharp(c3, radius=2, amount=1.2)
    c3 = contrast_stretch(c3)
    save("03_hand_region_deblur_contrast.png", c3)

    # 4. hand region edge detection
    c4 = crop(img, box3)
    c4 = upscale(c4, 3).astype(np.uint8)
    edges = cv2.Canny(c4, 40, 120)
    save("04_hand_region_edge_detection.png", edges)

    # 5. elbow fat pad zoom + soft tissue gamma
    box5 = (600, 150, 1050, 430)
    c5 = crop(img, box5)
    c5 = upscale(c5, 3)
    c5 = gamma(c5, 0.7)
    save("05_elbow_fatpad_zoom_contrast.png", c5)

    # 6. whole image CLAHE with background masked
    bg_thresh = np.percentile(img, 5)
    mask = img > bg_thresh
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    c6 = clahe.apply(img)
    c6_out = img.copy()
    c6_out[mask] = c6[mask]
    save("06_whole_image_clahe.png", c6_out)

    # 7. full forearm/limb edge trace overlay
    sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    mag = (mag / (mag.max() + 1e-6)) * 255.0
    overlay = np.clip(img.astype(np.float32) * 0.6 + mag * 0.4, 0, 255)
    save("07_full_forearm_edge_trace.png", overlay)

    # 8. torso region contrast
    box8 = (1120, 750, min(1280, w), min(1600, h))
    c8 = crop(img, box8)
    c8 = gamma(c8, 0.6)
    c8 = contrast_stretch(c8)
    save("08_torso_region_contrast.png", c8)

    # 9. inverted grayscale full image
    inv = 255 - img
    save("09_inverted_grayscale_fullimage.png", inv)

    # 10. proximal bone gamma recovery
    box10 = (620, 150, 1280, 420)
    c10 = crop(img, box10)
    c10 = gamma(c10, 1.8)
    save("10_proximal_bone_gamma_recovery.png", c10)

    # 11. corner mark crop contrast
    box11 = (1180, 1520, 1280, 1600)
    c11 = crop(img, box11)
    c11 = upscale(c11, 6)
    c11 = contrast_stretch(c11)
    save("11_cornermark_crop_contrast.png", c11)

    if failures:
        print("FAILURES:")
        for name, err in failures:
            print(f"  {name}: {err}")
        sys.exit(1)
    else:
        print("All 11 derived images generated and validated successfully.")


if __name__ == "__main__":
    main()
