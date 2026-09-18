"""Round 4 — quantitative measurements (plan D1-D4 + sharpness numbers).

Outputs:
  ID003_60_profiles_midforearm.png
  ID003_61_profile_diff.png
  ID003_62_axis_angulation.png
  ID003_63_width_curve.png
Numbers appended to planning/measurements_report.txt
"""
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

# geometry (full-image pixels)
TX0, TX1 = 563, 794          # transverse profile x-range (0.44-0.62)
Y_FOCUS = [616, 632, 648, 664]   # y 0.385/0.395/0.405/0.415
Y_CTRL_DIST = [720, 752]     # y 0.45/0.47
Y_CTRL_PROX = [528, 560]     # y 0.33/0.35
DIP_W0, DIP_W1 = 620, 740    # window searched for the focal dip
TRK_X0, TRK_X1 = 538, 845    # cortex tracking window (0.42-0.66)
TRK_Y0, TRK_Y1 = 448, 832    # y 0.28-0.52
SEG_PROX = (464, 584)        # y 0.29-0.365
SEG_DIST = (704, 816)        # y 0.44-0.51
Y_STEP = 640                 # y 0.40


def track_cortices(g):
    """Per-row peak detection of the four cortical margins.
    Returns dict row_y -> [x0..x3] for rows with exactly 4 peaks."""
    rows = {}
    for y in range(TRK_Y0, TRK_Y1):
        prof = g[y, TRK_X0:TRK_X1]
        pk, _ = find_peaks(prof, prominence=12, distance=12)
        if len(pk) == 4:
            rows[y] = [TRK_X0 + int(p) for p in pk]
    return rows


def fit(xs, ys):
    A = np.polyfit(ys, xs, 1)  # x = a*y + b
    resid = np.polyval(A, ys) - xs
    return A, float(np.sqrt(np.mean(resid ** 2)))


def main():
    g = C.load_gray()
    lines = ["\n[60/61/62/63] Mid-forearm quantitative battery\n"]

    # ---------------- cortex tracking ----------------
    rows = track_cortices(g)
    ys_sorted = sorted(rows)
    n_rows = len(ys_sorted)
    lines.append(f"  cortex tracking: {n_rows}/{TRK_Y1 - TRK_Y0} rows with exactly 4 peaks "
                 f"(window x {TRK_X0}-{TRK_X1})")
    if n_rows < 40:
        print("WARNING: cortex tracking failed on too many rows")

    # ---------------- D1: transverse + longitudinal profiles ----------------
    def transverse(y):
        return g[y, TX0:TX1]

    focal = np.array([transverse(y) for y in Y_FOCUS])
    ctrl_d = np.array([transverse(y) for y in Y_CTRL_DIST])
    ctrl_p = np.array([transverse(y) for y in Y_CTRL_PROX])

    xs = np.arange(TX0, TX1)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    ax = axes[0]
    for y, row in zip(Y_FOCUS, focal):
        ax.plot(xs, row, "-", alpha=0.35, lw=0.8)
    ax.plot(xs, focal.mean(axis=0), "r-", lw=2, label="focal 4-row mean")
    for y, row in zip(Y_CTRL_DIST, ctrl_d):
        ax.plot(xs, row, "--", alpha=0.35, lw=0.8, color="blue")
    ax.plot(xs, ctrl_d.mean(axis=0), "b--", lw=1.8, label="distal control 2-row mean")
    ax.plot(xs, ctrl_p.mean(axis=0), "g--", lw=1.2, alpha=0.7, label="proximal control 2-row mean")
    ax.axvspan(DIP_W0, DIP_W1, color="orange", alpha=0.12, label="dip-search window")
    ax.set_title("Transverse intensity profiles, mid-forearm (rows in px)")
    ax.legend(fontsize=8)
    ax.set_ylabel("gray level")

    # longitudinal profiles along tracked cortices
    ys_all = np.arange(TRK_Y0, TRK_Y1)
    ax = axes[1]
    for k in range(4):
        vals = []
        for y in ys_all:
            if y in rows:
                xk = rows[y][k]
                vals.append(g[y, max(0, xk - 2):xk + 3].mean())
            else:
                vals.append(np.nan)
        ax.plot(ys_all, vals, lw=0.9, label=f"cortex k={k}")
    ax.axhline(0, color="none")
    for ym in Y_FOCUS:
        ax.axhline(ym, color="red", ls=":", lw=0.8)
    ax.set_title("Longitudinal intensity along tracked cortical margins (y px)")
    ax.set_xlabel("y (px)"); ax.set_ylabel("gray level")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMG_DIR, "ID003_60_profiles_midforearm.png"), dpi=110)
    plt.close(fig)
    print("[ok] ID003_60_profiles_midforearm.png")

    # dip metrics
    def dip_metrics(row_arr):
        out = []
        for row in row_arr:
            sub = g[row, DIP_W0:DIP_W1]
            m = int(sub.min())
            y_i = row
            left_max = int(g[y_i, TX0:660].max())
            right_max = int(g[y_i, 680:TX1].max())
            base = (left_max + right_max) / 2.0
            depth = base - m
            thr = m + depth / 2.0
            width = int((sub < thr).sum()) if depth > 2 else 0
            x_min = DIP_W0 + int(np.argmin(sub))
            out.append((y_i, m, base, depth, width, x_min))
        return out

    fm = dip_metrics(Y_FOCUS)
    cm = dip_metrics(Y_CTRL_DIST)
    lines.append("  transverse dip at focal rows (y_px, min_gray, local_base, depth, half-width_px, x_min):")
    for r in fm:
        lines.append(f"    y={r[0]}: min={r[1]}, base={r[2]:.0f}, depth={r[3]:.0f}, width={r[4]}px, x_min={r[5]}")
    lines.append("  same metrics at distal control rows (noise floor):")
    for r in cm:
        lines.append(f"    y={r[0]}: min={r[1]}, base={r[2]:.0f}, depth={r[3]:.0f}, width={r[4]}px, x_min={r[5]}")

    # ---------------- D2: focal-minus-control difference ----------------
    diff = focal.mean(axis=0) - ctrl_d.mean(axis=0)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(xs, diff * 4, "k-", lw=1.2, label="(focal mean - distal control mean) x4")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvspan(DIP_W0, DIP_W1, color="orange", alpha=0.12)
    imin = int(np.argmin(diff))
    ax.axvline(xs[imin], color="red", ls=":", lw=1)
    ax.set_title("Focal-minus-control transverse profile (amplified x4)")
    ax.set_xlabel("x (px)"); ax.set_ylabel("delta gray level (x4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMG_DIR, "ID003_61_profile_diff.png"), dpi=110)
    plt.close(fig)
    print("[ok] ID003_61_profile_diff.png")
    lines.append(f"  focal-minus-control: max dip depth = {abs(diff[imin]):.1f} gray units at x={xs[imin]} "
                 f"(amplified x4 in image)")

    # ---------------- D3: axis / angulation / step-off ----------------
    def segment_fits(y_lo, y_hi, shift=0):
        fits = {}
        for k in range(4):
            yy = np.array([y for y in ys_sorted if y_lo + shift <= y <= y_hi + shift and y in rows])
            if len(yy) >= 15:
                fits[k] = fit(np.array([rows[y][k] for y in yy]), yy)
        # bone midlines (0,1) and (2,3)
        mids = {}
        for bone, (a, b) in enumerate([(0, 1), (2, 3)]):
            yy = np.array([y for y in ys_sorted if y_lo + shift <= y <= y_hi + shift
                           and y in rows and a in fits and b in fits])
            if len(yy) >= 15:
                xm = np.array([(rows[y][a] + rows[y][b]) / 2.0 for y in yy])
                mids[bone] = fit(xm, yy)
        return fits, mids

    fp, mp = segment_fits(*SEG_PROX)
    fd, md = segment_fits(*SEG_DIST)
    fp2, mp2 = segment_fits(*SEG_PROX, shift=6)
    fd2, md2 = segment_fits(*SEG_DIST, shift=-6)

    def angle_deg(m):
        return float(np.degrees(np.arctan(m[0][0])))

    ang = {}
    sens = []
    for bone in (0, 1):
        if bone in mp and bone in md:
            a_prox, a_dist = angle_deg(mp[bone]), angle_deg(md[bone])
            ang[bone] = (a_prox, a_dist, abs(a_prox - a_dist))
            lines.append(f"  bone {bone + 1} (image {['left','right'][bone]}): proximal axis {a_prox:+.2f} deg, "
                         f"distal axis {a_dist:+.2f} deg (from vertical), |deviation| = {abs(a_prox - a_dist):.2f} deg, "
                         f"fit RMS prox {mp[bone][1]:.2f}px / dist {md[bone][1]:.2f}px")
        if bone in mp2 and bone in md2:
            sens.append(abs(angle_deg(mp2[bone]) - angle_deg(md2[bone])))
    if sens:
        lines.append(f"  sensitivity (segments shifted +/-6 px): max |deviation| change = {max(sens):.2f} deg")

    # step-off at y=640
    steps = []
    for k in range(4):
        if k in fp and k in fd:
            x_prox = float(np.polyval(fp[k][0], Y_STEP))
            x_dist = float(np.polyval(fd[k][0], Y_STEP))
            steps.append((k, x_dist - x_prox))
            lines.append(f"  step-off cortex k={k} at y={Y_STEP}: {x_dist - x_prox:+.2f} px")
    lines.append("  NOTE: angles are image-plane apparent values (oblique projection); 1-2 deg within fit error.")

    # annotate on the plain mid-forearm crop geometry
    from PIL import Image, ImageDraw
    xa, xb, ya, yb = 512, 819, 448, 800
    crop_u8 = np.clip(g[ya:yb, xa:xb], 0, 255).astype(np.uint8)
    img = Image.fromarray(crop_u8).convert("RGB")
    dr = ImageDraw.Draw(img)
    colors = [(0, 255, 0), (0, 160, 255), (255, 0, 255), (255, 160, 0)]
    for k in range(4):
        c = colors[k]
        if k in fp:
            a, b = fp[k][0]
            dr.line([(a * SEG_PROX[0] + b - xa, SEG_PROX[0] - ya),
                     (a * SEG_PROX[1] + b - xa, SEG_PROX[1] - ya)], fill=c, width=1)
        if k in fd:
            a, b = fd[k][0]
            dr.line([(a * SEG_DIST[0] + b - xa, SEG_DIST[0] - ya),
                     (a * SEG_DIST[1] + b - xa, SEG_DIST[1] - ya)], fill=c, width=1)
    for bone, (a, b) in enumerate([(0, 1), (2, 3)]):
        c = colors[b]
        if bone in mp:
            aa, bb = mp[bone][0]
            dr.line([(aa * SEG_PROX[0] + bb - xa, SEG_PROX[0] - ya),
                     (aa * SEG_PROX[1] + bb - xa, SEG_PROX[1] - ya)], fill=(255, 255, 255), width=2)
        if bone in md:
            aa, bb = md[bone][0]
            dr.line([(aa * SEG_DIST[0] + bb - xa, SEG_DIST[0] - ya),
                     (aa * SEG_DIST[1] + bb - xa, SEG_DIST[1] - ya)], fill=(255, 255, 255), width=2)
    dr.line([(0, Y_STEP - ya), (xb - xa, Y_STEP - ya)], fill=(255, 80, 80), width=1)
    dr.text((4, 2), "mid-forearm: white=bone midline fits (proximal+distal), colored=cortex fits, red=line at y=0.40",
            fill=(255, 255, 255))
    txt = "  ".join(f"bone{b + 1}: {ang[b][2]:.2f}deg" for b in sorted(ang))
    dr.text((4, 16), txt, fill=(255, 255, 0))
    big = img.resize((img.width * 3, img.height * 3), Image.NEAREST)
    big.save(os.path.join(C.IMG_DIR, "ID003_62_axis_angulation.png"))
    print("[ok] ID003_62_axis_angulation.png")

    # ---------------- D4: width curve ----------------
    y_lo, y_hi = 400, 880
    xw0, xw1 = 380, 1010
    ys_w = list(range(y_lo, y_hi, 2))
    soft_w, bone_w = [], []
    for y in ys_w:
        seg = g[y, xw0:xw1]
        s = np.nonzero(seg > 20)[0]
        b = np.nonzero(seg > 110)[0]
        soft_w.append((b[-1] - b[0]) if len(b) else np.nan)
        soft_w[-1] = (s[-1] - s[0]) if len(s) else np.nan
        bone_w.append((b[-1] - b[0]) if len(b) else np.nan)
    soft_w = np.array(soft_w, dtype=float)
    bone_w = np.array(bone_w, dtype=float)
    def smooth(a, k=5):
        out = np.convolve(a, np.ones(k) / k, mode="same")
        return out
    fig, ax = plt.subplots(figsize=(8, 5))
    yn = np.array(ys_w) / C.H
    ax.plot(yn, smooth(soft_w), "b-", lw=1.4, label="soft-tissue silhouette width")
    ax.plot(yn, smooth(bone_w), "r-", lw=1.4, label="bone (threshold 110) width")
    ax.axhspan(0.385, 0.415, color="orange", alpha=0.15, label="focal level")
    ax.set_xlabel("y (normalized)"); ax.set_ylabel("width (px)")
    ax.set_title("Forearm width vs level (5-row moving average)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.IMG_DIR, "ID003_63_width_curve.png"), dpi=110)
    plt.close(fig)
    print("[ok] ID003_63_width_curve.png")
    i_f = ys_w.index(632)
    i_p = ys_w.index(528)
    i_d = ys_w.index(720)
    lines.append(f"  width at focal y~0.40: soft={soft_w[i_f]:.0f}px bone={bone_w[i_f]:.0f}px | "
                 f"proximal y~0.33: soft={soft_w[i_p]:.0f}px bone={bone_w[i_p]:.0f}px | "
                 f"distal y~0.45: soft={soft_w[i_d]:.0f}px bone={bone_w[i_d]:.0f}px")

    # ---------------- sharpness (numbers only, council decision E9) ----------------
    rois_s = {
        "lesion (x.45-.60 y.33-.47)": g[int(0.33 * C.H):int(0.47 * C.H), int(0.45 * C.W):int(0.60 * C.W)],
        "elbow (x.47-.63 y.13-.28)": g[int(0.13 * C.H):int(0.28 * C.H), int(0.47 * C.W):int(0.63 * C.W)],
        "distal radius (x.45-.58 y.55-.65)": g[int(0.55 * C.H):int(0.65 * C.H), int(0.45 * C.W):int(0.58 * C.W)],
    }
    lines.append("  sharpness (variance of Laplacian, relative; reported as numbers only):")
    for label, roi in rois_s.items():
        v = float(np.var(cv2.Laplacian(roi, cv2.CV_32F)))
        lines.append(f"    {label}: {v:.1f}")

    C.append_measurements("".join(lines) + "\n")
    print("[done] r4_quant")


if __name__ == "__main__":
    main()
