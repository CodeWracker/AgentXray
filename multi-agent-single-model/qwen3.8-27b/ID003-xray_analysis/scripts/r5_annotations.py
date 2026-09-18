"""Round 5 — annotation outputs (plan D6-D9).

Marker coordinates were placed by the orchestrator's direct inspection of the
enhanced crops (ID003_30_wrist_bonewindow.png, ID003_31_elbow_clahe.png,
ID003_50_lowerright_clahe.png). They are HYPOTHESES to be confirmed or
corrected by the reviewing agents, not ground truth.

Outputs:
  ID003_70_carpal_census.png
  ID003_71_elbow_census.png
  ID003_72_physeal_width.png
  ID003_80_attribution_overlay.png
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# normalized (x, y) marker hypotheses
CARPAL = [
    (0.525, 0.6175, "C1 rounded center, smooth margin"),
    (0.523, 0.6375, "C2 rounded center, smooth margin"),
    (0.525, 0.6625, "C3 larger center, smooth margin"),
    (0.539, 0.6200, "C4 small/uncertain"),
    (0.4875, 0.6150, "distal ulnar epiphysis (partial)"),
]
ELBOW = [
    (0.555, 0.1750, "capitellum"),
    (0.573, 0.1875, "radial head epiphysis"),
    (0.542, 0.1800, "distal humeral epiphysis (medial)"),
    (0.514, 0.1875, "olecranon"),
    (0.547, 0.1960, "medial epicondyle (uncertain)"),
]
PHYSE_X = [0.506, 0.525, 0.541]   # vertical measurement lines
PHYSE_Y0, PHYSE_Y1 = 0.5825, 0.6200
SHAFT_Y = 0.6000                  # level for shaft width


def census(base_name, box, markers, fname, title):
    crop, (xa, xb, ya, yb) = C.crop_px(C.load_gray(), *box)
    w = xb - xa
    h = yb - ya
    img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).convert("RGB")
    f = 2.5
    img = img.resize((int(round(img.width * f)), int(round(img.height * f))), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    for i, (nx, ny, label) in enumerate(markers, 1):
        cx = (nx * C.W - xa) * f
        cy = (ny * C.H - ya) * f
        r = 14
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 255, 0), width=2)
        dr.text((cx + r + 2, cy - 8), str(i), fill=(255, 255, 0))
    dr.text((6, 4), title, fill=(255, 255, 255))
    img.save(os.path.join(C.IMG_DIR, fname))
    print(f"[ok] {fname} {img.size[0]}x{img.size[1]}")
    return [(i, nx, ny, label) for i, (nx, ny, label) in enumerate(markers, 1)]


def main():
    g = C.load_gray()
    lines = ["\n[70] Carpal ossification center census (marker hypotheses, normalized coords)\n"]
    for i, nx, ny, label in census(
        "ID003_70_carpal_census.png", C.CROPS["wrist"], CARPAL,
        "ID003_70_carpal_census.png",
        "wrist: numbered ossification centers (hypotheses - verify vs plain crop)",
    ):
        lines.append(f"  {i}: ({nx:.4f}, {ny:.4f}) {label}")

    lines.append("\n[71] Elbow ossification center census (marker hypotheses)\n")
    for i, nx, ny, label in census(
        "ID003_71_elbow_census.png", C.CROPS["elbow"], ELBOW,
        "ID003_71_elbow_census.png",
        "elbow: numbered ossification centers (hypotheses - verify vs plain crop)",
    ):
        lines.append(f"  {i}: ({nx:.4f}, {ny:.4f}) {label}")

    # ---------------- D8: physeal width ----------------
    crop, (xa, xb, ya, yb) = C.crop_px(g, *C.CROPS["wrist"])
    img = Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).convert("RGB")
    f = 2.5
    img = img.resize((int(round(img.width * f)), int(round(img.height * f))), Image.NEAREST)
    dr = ImageDraw.Draw(img)
    y0p = int(round(PHYSE_Y0 * C.H))
    y1p = int(round(PHYSE_Y1 * C.H))
    ys_shaft = int(round(SHAFT_Y * C.H))
    lines.append("\n[72] Distal radial physis width (3 perpendicular lines, native px)\n")
    for i, nx in enumerate(PHYSE_X, 1):
        xc = int(round(nx * C.W))
        dr.line([(xc - xa) * f, (y0p - ya) * f, (xc - xa) * f, (y1p - ya) * f],
                fill=(255, 0, 0), width=2)
        col = g[y0p:y1p, xc - 1:xc + 2].mean(axis=1)
        cmin = float(col.min())
        cmax = float(col.max())
        half = cmin + (cmax - cmin) * 0.5
        # local FWHM around the darkest point (physis), not whole window
        i0 = int(np.argmin(col))
        a = i0
        while a > 0 and col[a - 1] < half:
            a -= 1
        b = i0
        while b < len(col) - 1 and col[b + 1] < half:
            b += 1
        pw = int(b - a + 1)
        # shaft width at SHAFT_Y: bone extent with threshold 90, x 0.46-0.58
        row = g[ys_shaft, int(0.46 * C.W):int(0.58 * C.W)]
        b = np.nonzero(row > 90)[0]
        shaft_w = int(b[-1] - b[0] + 1) if len(b) else 0
        ratio = pw / shaft_w if shaft_w else float("nan")
        lines.append(f"  line {i} at x={nx:.3f}: physis band FWHM = {pw}px, "
                     f"local gray min={cmin:.0f} max={cmax:.0f}; shaft width at y={SHAFT_Y:.3f} = {shaft_w}px; "
                     f"ratio physis/shaft = {ratio:.3f}")
    dr.text((6, 4), "distal radial physis: 3 measurement lines (hypotheses)", fill=(255, 255, 255))
    img.save(os.path.join(C.IMG_DIR, "ID003_72_physeal_width.png"))
    print(f"[ok] ID003_72_physeal_width.png {img.size[0]}x{img.size[1]}")

    # ---------------- D9: attribution overlay (full image) ----------------
    img = Image.fromarray(np.clip(g, 0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(img)
    X, Y = lambda v: int(v * C.W), lambda v: int(v * C.H)

    dr.line([(X(0.52), Y(0.19)), (X(0.52), Y(0.59))], fill=(255, 255, 255), width=4)
    dr.text((X(0.30), Y(0.16)), "EXAMINED LIMB axis (humerus->forearm->wrist)", fill=(255, 255, 255))

    cx, cy = X(0.64), Y(0.89)
    dr.ellipse([cx - X(0.16), cy - Y(0.09), cx + X(0.16), cy + Y(0.09)],
               outline=(255, 255, 0), width=3)
    dr.text((X(0.66), Y(0.985)), "2nd long bones - 2nd limb? (Agent_05)", fill=(255, 255, 0))

    dr.rectangle([X(0.90), Y(0.30), X(0.999), Y(1.0)], outline=(0, 255, 255), width=3)
    dr.text((X(0.80), Y(0.27)), "torso band", fill=(0, 255, 255))

    dr.rectangle([X(0.855), Y(0.875), X(0.965), Y(0.985)], outline=(255, 0, 255), width=2)
    dr.text((X(0.70), Y(0.865)), "linear texture (external object?)", fill=(255, 0, 255))

    fx, fy = X(0.63), Y(0.82)
    dr.ellipse([fx - 8, fy - 8, fx + 8, fy + 8], outline=(255, 60, 60), width=3)
    dr.text((fx + 12, fy - 12), "dense focus", fill=(255, 60, 60))

    dr.text((X(0.02), Y(0.01)), "ID003 attribution map - labels are HYPOTHESES", fill=(255, 255, 255))
    img.save(os.path.join(C.IMG_DIR, "ID003_80_attribution_overlay.png"))
    print(f"[ok] ID003_80_attribution_overlay.png {img.size[0]}x{img.size[1]}")

    C.append_measurements("".join(lines) + "\n")
    print("[done] r5_annotations")


if __name__ == "__main__":
    main()
