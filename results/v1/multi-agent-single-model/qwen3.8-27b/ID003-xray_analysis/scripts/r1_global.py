"""Round 1 — global baseline/enhancement outputs (plan A1-A4, D5).

Outputs:
  ID003_01_pctstretch_full.png
  ID003_02_clahe_full.png
  ID003_03_highpass_forearmhand.png
  ID003_04_saturation_map.png
  ID003_64_histogram_saturation.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    g = C.load_gray()

    # A1 — percentile stretch baseline
    C.save(C.pct_stretch(g, 1.0, 99.5), "ID003_01_pctstretch_full.png")

    # A2 — full-image CLAHE
    C.save(C.clahe(g, clip=2.5, tiles=8), "ID003_02_clahe_full.png")

    # A3 — high-pass, ROI-limited (forearm + hand band), mid-gray centered
    hp = C.highpass(g, sigma=15.0)
    canvas = np.zeros_like(hp)
    ya, yb = int(round(0.05 * C.H)), C.H
    xa, xb = int(round(0.38 * C.W)), int(round(0.72 * C.W))
    canvas[ya:yb, xa:xb] = hp[ya:yb, xa:xb]
    C.save(C.gray_centered_hp(canvas, scale=3.0), "ID003_03_highpass_forearmhand.png")

    # A4 — saturation map (pixels >= 250 = clipped, yellow on black)
    sat = (g >= 250)
    rgb = np.zeros((C.H, C.W, 3), dtype=np.uint8)
    rgb[sat] = (255, 255, 0)
    C.save(rgb, "ID003_04_saturation_map.png", mode="RGB")

    # D5 — histograms + saturation fractions
    rois = {
        "full image": g,
        "hand (x.40-.66 y.68-.95)": g[int(0.68 * C.H):int(0.95 * C.H), int(0.40 * C.W):int(0.66 * C.W)],
        "mid-forearm (x.44-.60 y.30-.50)": g[int(0.30 * C.H):int(0.50 * C.H), int(0.44 * C.W):int(0.60 * C.W)],
        "lower-right (x.45-.90 y.80-1.00)": g[int(0.80 * C.H):C.H, int(0.45 * C.W):int(0.90 * C.W)],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    lines = ["\n[64] Histograms / saturation fractions\n"]
    for ax, (label, roi) in zip(axes.ravel(), rois.items()):
        ax.hist(roi.ravel(), bins=256, range=(0, 256), color="gray")
        f250 = float((roi >= 250).mean())
        f255 = float((roi == 255).mean())
        ax.set_title(f"{label}\n>=250: {f250:.1%}   ==255: {f255:.1%}")
        ax.set_xlim(0, 255)
        lines.append(f"  {label}: pixels >=250 = {f250:.3%}; pixels ==255 = {f255:.3%}")
    fig.suptitle("ID003-xray: intensity histograms (grayscale luminance)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMG_DIR, "ID003_64_histogram_saturation.png"), dpi=110)
    plt.close(fig)
    print("[ok] ID003_64_histogram_saturation.png")
    C.append_measurements("".join(lines) + "\n")


if __name__ == "__main__":
    main()
