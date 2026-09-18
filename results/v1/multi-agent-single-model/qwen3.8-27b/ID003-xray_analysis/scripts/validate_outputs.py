"""Validation pass: every expected output must exist, open, and have the
expected dimensions. Failures are recorded (never silent) and the script
exits non-zero if any check fails.
"""
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

H, W = C.H, C.W

# name -> expected (width, height, mode)
def spec(x0, x1, y0, y1, f):
    # must mirror common.crop_px + nn_upscale exactly
    xa, xb = int(round(x0 * W)), int(round(x1 * W))
    ya, yb = int(round(y0 * H)), int(round(y1 * H))
    return (int(round((xb - xa) * f)), int(round((yb - ya) * f)))

EXPECTED = {
    # original (unmodified copy)
    "ID003-xray.png": (W, H, "L"),
    # A
    "ID003_01_pctstretch_full.png": (W, H, "L"),
    "ID003_02_clahe_full.png": (W, H, "L"),
    "ID003_03_highpass_forearmhand.png": (W, H, "L"),
    "ID003_04_saturation_map.png": (W, H, "RGB"),
    "ID003_64_histogram_saturation.png": None,
    # B plain crops
    "ID003_10_crop_midforearm_plain.png": spec(*C.CROPS["midforearm"], 3),
    "ID003_11_crop_wrist_plain.png": spec(*C.CROPS["wrist"], 2.5),
    "ID003_12_crop_elbow_plain.png": spec(*C.CROPS["elbow"], 2.5),
    "ID003_13_crop_hand_plain.png": spec(*C.CROPS["hand"], 2),
    "ID003_14_crop_lowerright_plain.png": spec(*C.CROPS["lowerright"], 2),
    # C enhanced
    "ID003_20_midforearm_clahe.png": spec(*C.CROPS["midforearm"], 3),
    "ID003_21_midforearm_unsharp.png": spec(*C.CROPS["midforearm"], 3),
    # two native-resolution panels stacked + 4px separator
    "ID003_22_midforearm_canny.png": (spec(*C.CROPS["midforearm"], 1)[0],
                                      spec(*C.CROPS["midforearm"], 1)[1] * 2 + 4),
    "ID003_23_midforearm_diff.png": spec(*C.CROPS["midforearm"], 3),
    "ID003_30_wrist_bonewindow.png": spec(*C.CROPS["wrist"], 2.5),
    "ID003_31_elbow_clahe.png": spec(*C.CROPS["elbow"], 2.5),
    "ID003_40_hand_bonewindow.png": spec(*C.CROPS["hand"], 2),
    "ID003_41_hand_softtissue.png": spec(0.24, 0.70, 0.50, 1.00, 2),
    "ID003_50_lowerright_clahe.png": spec(*C.CROPS["lowerright"], 2),
    "ID003_51_lowerright_edges.png": (spec(*C.CROPS["lowerright"], 1)[0],
                                      spec(*C.CROPS["lowerright"], 1)[1] * 2 + 4),
    # D quantitative
    "ID003_60_profiles_midforearm.png": None,
    "ID003_61_profile_diff.png": None,
    "ID003_62_axis_angulation.png": spec(*C.CROPS["midforearm"], 3),
    "ID003_63_width_curve.png": None,
    # D annotations
    "ID003_70_carpal_census.png": spec(*C.CROPS["wrist"], 2.5),
    "ID003_71_elbow_census.png": spec(*C.CROPS["elbow"], 2.5),
    "ID003_72_physeal_width.png": spec(*C.CROPS["wrist"], 2.5),
    "ID003_80_attribution_overlay.png": (W, H),
    # Round 2 — D2 resolution crops (x 0.44-0.74, y 0.74-1.00)
    "ID003_90_diagonalbones_plain.png": (int(round((0.74 - 0.44) * W * 2)),
                                         int(round((1.00 - 0.74) * H * 2))),
    "ID003_91_diagonalbones_clahe.png": (int(round((0.74 - 0.44) * W * 2)),
                                         int(round((1.00 - 0.74) * H * 2))),
    "ID003_92_diagonalbones_edges.png": (int(round((0.74 - 0.44) * W)),
                                         int(round((1.00 - 0.74) * H)) * 2 + 4),
}

def mode_ok(actual, want):
    if want in (None, "L"):
        return actual in ("L", "RGB", "P")
    return actual in (want, "RGB") if want == "RGB" else True

def main():
    report = ["VALIDATION REPORT — ID003-xray pipeline round 1\n"]
    failures = 0
    for name, exp in EXPECTED.items():
        path = os.path.join(C.IMG_DIR, name)
        if not os.path.exists(path):
            report.append(f"  [FAIL] {name}: MISSING")
            failures += 1
            continue
        try:
            im = Image.open(path)
            im.verify()
            im = Image.open(path)
            w, h = im.size
            if exp is not None:
                ew, eh = exp[0], exp[1]
                if (w, h) != (ew, eh):
                    report.append(f"  [FAIL] {name}: size {w}x{h} != expected {ew}x{eh}")
                    failures += 1
                    continue
            report.append(f"  [ok] {name}: {im.size[0]}x{im.size[1]} {im.mode}")
        except Exception as e:
            report.append(f"  [FAIL] {name}: cannot open ({e})")
            failures += 1
    report.append(f"\n{len(EXPECTED) - failures}/{len(EXPECTED)} outputs valid, {failures} failures")
    with open(os.path.join(C.PLAN_DIR, "validation_report.txt"), "w") as f:
        f.write("".join(report))
    print("".join(report))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
