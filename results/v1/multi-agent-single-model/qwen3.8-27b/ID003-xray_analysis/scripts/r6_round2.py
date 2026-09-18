"""Round 2 — D2 resolution crops (plan Round 2, R2-1).

Targeted crops of the two oblique long bones in the lower right, at higher
magnification than the Round-1 lowerright crop, to resolve the council
disagreement: second (contralateral) limb (Agent_06) vs torso/soft-tissue
(Agent_04, Agent_03).

Outputs:
  ID003_90_diagonalbones_plain.png  (2x nearest-neighbor, unfiltered)
  ID003_91_diagonalbones_clahe.png  (CLAHE clip 3.0, 8x8)
  ID003_92_diagonalbones_edges.png  (top: Canny white-on-black; bottom: red overlay)
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

BOX = (0.44, 0.74, 0.74, 1.00)  # x0, x1, y0, y1 normalized


def canny_pair(crop, fname):
    g8 = np.clip(crop, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(g8, (0, 0), 1.5)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_u8 = np.clip(mag / max(float(mag.max()), 1e-6) * 255.0, 0, 255).astype(np.uint8)
    t = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    scale = max(float(mag.max()), 1e-6) / 255.0
    edges = cv2.Canny(blur, max(15.0, float(t) * scale / 2.5), max(20.0, float(t) * scale))

    h, w = crop.shape
    panel1 = np.zeros((h, w, 3), dtype=np.float32)
    panel1[edges > 0] = (255, 255, 255)
    base = np.stack([g8] * 3, axis=-1)
    red = base.copy()
    red[edges > 0] = (255, 0, 0)
    sep = np.full((4, w, 3), 128, dtype=np.float32)
    C.save(np.vstack([panel1, sep, red.astype(np.float32)]), fname, mode="RGB")


def main():
    g = C.load_gray()
    crop, box = C.crop_px(g, *BOX)
    C.save(C.nn_upscale(crop, 2), "ID003_90_diagonalbones_plain.png")
    C.save(C.nn_upscale(C.clahe(crop, clip=3.0, tiles=8), 2), "ID003_91_diagonalbones_clahe.png")
    canny_pair(crop, "ID003_92_diagonalbones_edges.png")
    print("[done] r6_round2")


if __name__ == "__main__":
    main()
