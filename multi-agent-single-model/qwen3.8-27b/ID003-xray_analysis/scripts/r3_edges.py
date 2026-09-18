"""Round 3 — edge analyses (plan C3, C10).

  ID003_22_midforearm_canny.png   panel1: Canny edges (white on black)
                                  panel2: edges overlaid red on plain crop
  ID003_51_lowerright_edges.png   panel1: Sobel gradient magnitude
                                  panel2: Canny overlaid on plain crop
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def canny_pair(crop, fname):
    """Two-panel edge image for a native-resolution crop."""
    g8 = np.clip(crop, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g8, (0, 0), 1.5)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_u8 = np.clip(mag / max(float(mag.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    t = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    scale = max(float(mag.max()), 1e-6) / 255.0
    t_hi = max(15.0, float(t) * scale)
    t_lo = t_hi / 2.5
    edges = cv2.Canny(blur, t_lo, t_hi)

    h, w = crop.shape
    panel1 = np.zeros((h, w), dtype=np.float32)
    panel1[edges > 0] = 255.0

    base = np.stack([np.clip(crop, 0, 255)] * 3, axis=-1)
    red = base.copy()
    m = edges > 0
    red[m] = (255, 0, 0)
    panel2 = red.astype(np.float32)

    # stack vertically with a separator
    sep = np.full((4, w, 3), 128, dtype=np.float32)
    p1 = np.stack([panel1] * 3, axis=-1)
    out = np.vstack([p1, sep, panel2])
    C.save(out, fname, mode="RGB")
    return t_lo, t_hi


def sobel_pair(crop, fname):
    g8 = np.clip(crop, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g8, (0, 0), 1.0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    p95 = float(np.percentile(mag, 95))
    disp = np.clip(128.0 + (mag - p95 / 2.0) * 3.0, 0, 255)
    panel1 = np.stack([disp] * 3, axis=-1)

    mag_u8 = np.clip(mag / max(float(mag.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    t = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    scale = max(float(mag.max()), 1e-6) / 255.0
    edges = cv2.Canny(blur, max(10.0, float(t) * scale / 2.5), max(20.0, float(t) * scale))
    base = np.stack([g8] * 3, axis=-1)
    red = base.copy()
    red[edges > 0] = (255, 0, 0)
    panel2 = red.astype(np.float32)

    h, w = crop.shape
    sep = np.full((4, w, 3), 128, dtype=np.float32)
    out = np.vstack([panel1, sep, panel2])
    C.save(out, fname, mode="RGB")


def main():
    g = C.load_gray()

    crop, _ = C.crop_name(g, "midforearm")
    t_lo, t_hi = canny_pair(crop, "ID003_22_midforearm_canny.png")
    print(f"      canny thresholds midforearm: lo={t_lo:.1f} hi={t_hi:.1f}")

    crop, _ = C.crop_name(g, "lowerright")
    sobel_pair(crop, "ID003_51_lowerright_edges.png")

    C.append_measurements(
        f"\n[22] Canny (mid-forearm crop, sigma=1.5): thresholds lo={t_lo:.1f} hi={t_hi:.1f}\n"
    )
    print("[done] r3_edges")


if __name__ == "__main__":
    main()
