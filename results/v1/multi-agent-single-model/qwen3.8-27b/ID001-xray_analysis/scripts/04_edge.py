"""[9] hindfoot unsharp edge, [10] hindfoot Sobel gradient cross-check, [11] full-image edge overlay on original."""
import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_gray, load_rgb, clahe, crop_zoom, unsharp, save

g = load_gray()
rgb = load_rgb()
base = clahe(g, clip=2.5)


def zoom2(region):
    return cv2.resize(region, (region.shape[1] * 2, region.shape[0] * 2), interpolation=cv2.INTER_NEAREST)


x0, y0, x1, y1 = 130, 225, 250, 340

# [9] unsharp mask
un = unsharp(base, sigma=2.0, amount=2.0)
save("09_edge_hindfoot_unsharp.png", zoom2(un[y0:y1, x0:x1]))

# [10] Sobel gradient magnitude
gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
mag = cv2.magnitude(gx, gy)
mag8 = (255.0 * mag / mag.max()).astype("uint8")
save("10_edge_hindfoot_gradient.png", zoom2(mag8[y0:y1, x0:x1]))

# [11] full-image edge overlay on the original (native 373x454)
# ITER-2: round-C reviewers (Agent_04, Agent_06) found the v1 overlay saturated
# (45th-percentile threshold, opaque red) so no candidate line could be isolated.
# Now: 90th-percentile threshold, semi-transparent red blend so the underlying
# grayscale anatomy stays visible at every edge location.
mag = cv2.magnitude(gx, gy)
thr = float(np.percentile(mag, 90))
edge = (mag > thr)
overlay = rgb.copy().astype(np.float32)
red = np.zeros_like(overlay)
red[:, :, 0] = 255
overlay[edge] = 0.45 * overlay[edge] + 0.55 * red[edge]
save("11_edge_overlay_full.png", overlay.astype("uint8"))
print("edge overlay threshold:", round(thr, 1), "edge pixels:", int(edge.sum()))
