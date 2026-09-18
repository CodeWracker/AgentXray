"""Round 2 — plain reference crops + enhanced crop derivatives (plan B1-B5,
C1-C2, C4-C5, C7-C9).

Plain crops are nearest-neighbor upscaled, unfiltered (ground truth per
council rule R1). Enhanced variants are applied on the native-resolution
crop before any upscaling.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def main():
    g = C.load_gray()

    # ---------- B: plain reference crops (NN upscale, no filtering) ----------
    plain = {}
    specs = {
        "midforearm": ("ID003_10_crop_midforearm_plain.png", 3),
        "wrist": ("ID003_11_crop_wrist_plain.png", 2.5),
        "elbow": ("ID003_12_crop_elbow_plain.png", 2.5),
        "hand": ("ID003_13_crop_hand_plain.png", 2),
        "lowerright": ("ID003_14_crop_lowerright_plain.png", 2),
    }
    for name, (fname, f) in specs.items():
        crop, box = C.crop_name(g, name)
        plain[name] = (crop, box, f)
        C.save(C.nn_upscale(crop, f), fname)

    # ---------- C1: mid-forearm CLAHE (TOP PRIORITY) ----------
    crop, box, f = plain["midforearm"]
    C.save(C.nn_upscale(C.clahe(crop, clip=2.5, tiles=8), f),
           "ID003_20_midforearm_clahe.png")

    # ---------- C2: mid-forearm unsharp mask ----------
    C.save(C.nn_upscale(C.unsharp(crop, sigma=2.0, amount=0.7), f),
           "ID003_21_midforearm_unsharp.png")

    # ---------- C4: enhanced-minus-plain difference (artifact discriminator)
    d = C.clahe(crop, clip=2.5, tiles=8) - crop
    d = np.clip(d * 4.0, -127, 127) + 128.0
    C.save(C.nn_upscale(d, f), "ID003_23_midforearm_diff.png")

    # ---------- C5: wrist bone window ----------
    crop, box, f = plain["wrist"]
    w = C.window(crop, 5, 98)
    w = C.unsharp(w, sigma=1.5, amount=0.4)
    C.save(C.nn_upscale(w, f), "ID003_30_wrist_bonewindow.png")

    # ---------- C6: elbow CLAHE ----------
    crop, box, f = plain["elbow"]
    C.save(C.nn_upscale(C.clahe(crop, clip=2.5, tiles=8), f),
           "ID003_31_elbow_clahe.png")

    # ---------- C7: hand bone window ----------
    crop, box, f = plain["hand"]
    hw = C.window(crop, 10, 99)
    hw = C.unsharp(hw, sigma=1.5, amount=0.4)
    C.save(C.nn_upscale(hw, f), "ID003_40_hand_bonewindow.png")

    # ---------- C8: hand + distal forearm soft-tissue window ----------
    st_crop, _ = C.crop_px(g, 0.24, 0.70, 0.50, 1.00)
    st = C.window(st_crop, 2, 45)
    C.save(C.nn_upscale(st, 2), "ID003_41_hand_softtissue.png")

    # ---------- C9: lower-right CLAHE (D2 test) ----------
    crop, box, f = plain["lowerright"]
    C.save(C.nn_upscale(C.clahe(crop, clip=3.0, tiles=8), f),
           "ID003_50_lowerright_clahe.png")

    print("[done] r2_crops")


if __name__ == "__main__":
    main()
