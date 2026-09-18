#!/usr/bin/env python3
"""
Implements planning/processing_plan.txt for ID001-xray.png.
Generates images 01-08 into images/. Research/educational exercise only.
"""
import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES = os.path.join(BASE, "images")
ORIGINAL = os.path.join(IMAGES, "ID001-xray_original.png")

FAILURES = []

def load_gray():
    img = cv2.imread(ORIGINAL, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {ORIGINAL}")
    return img

def save(path, arr, note=""):
    ok = cv2.imwrite(path, arr)
    if not ok:
        FAILURES.append(f"cv2.imwrite failed: {path} {note}")
    else:
        print(f"wrote {os.path.relpath(path, BASE)}  shape={arr.shape}")

def analysis_01_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out = clahe.apply(gray)
    save(os.path.join(IMAGES, "01_clahe_full.png"), out)

def analysis_02_unsharp(gray):
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
    amount = 1.5
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    save(os.path.join(IMAGES, "02_unsharp_mask_full.png"), sharpened)

def analysis_03_edge_map(gray):
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inverted = 255 - mag_norm  # dark edges on light background per plan
    save(os.path.join(IMAGES, "03_edge_map_sobel.png"), inverted)

def analysis_04_roi_ankle_hindfoot(gray):
    h, w = gray.shape
    y0, y1 = int(h * 0.44), h
    x0, x1 = 0, w
    crop = gray[y0:y1, x0:x1]
    upscale = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    out = clahe.apply(upscale)
    save(os.path.join(IMAGES, "04_roi_ankle_hindfoot_zoom.png"), out,
         note=f"crop=({x0},{y0},{x1},{y1})")

def analysis_05_roi_tibial_metaphysis(gray):
    h, w = gray.shape
    y0, y1 = int(h * 0.27), int(h * 0.52)
    x0, x1 = int(w * 0.25), int(w * 0.82)
    crop = gray[y0:y1, x0:x1]
    upscale = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    save(os.path.join(IMAGES, "05_roi_tibial_metaphysis_zoom.png"), upscale,
         note=f"crop=({x0},{y0},{x1},{y1})")

def analysis_06_histogram_contrast_stretch(gray):
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(stretched, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Linear contrast stretch (0-255)")
    axes[1].axis("off")
    axes[2].hist(gray.ravel(), bins=256, range=(0, 255), color="steelblue", alpha=0.8,
                 label="original")
    axes[2].hist(stretched.ravel(), bins=256, range=(0, 255), color="darkorange",
                 alpha=0.5, label="stretched")
    axes[2].set_title("Intensity histogram")
    axes[2].set_xlabel("pixel value")
    axes[2].set_ylabel("count")
    axes[2].legend()
    fig.tight_layout()
    path = os.path.join(IMAGES, "06_histogram_and_contrast_stretch.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    if not os.path.exists(path):
        FAILURES.append(f"matplotlib savefig failed: {path}")
    else:
        print(f"wrote {os.path.relpath(path, BASE)}")

def analysis_07_background_artifact_map(gray):
    h, w = gray.shape
    # Emphasize low-intensity structure by clipping to a low percentile range
    p_lo, p_hi = np.percentile(gray, [0, 45])
    clipped = np.clip(gray, p_lo, p_hi)
    clipped_norm = ((clipped - p_lo) / max(p_hi - p_lo, 1) * 255).astype(np.uint8)
    colored = cm.inferno(clipped_norm / 255.0)[:, :, :3]

    row_y = int(h * 0.92)   # near-bottom horizontal profile
    col_x = int(w * 0.08)   # near-left vertical profile

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(colored)
    axes[0].axhline(row_y, color="cyan", linewidth=0.8)
    axes[0].axvline(col_x, color="lime", linewidth=0.8)
    axes[0].set_title("Background false-color map (low-intensity range)")
    axes[0].axis("off")

    axes[1].plot(gray[row_y, :], color="cyan")
    axes[1].set_title(f"Horizontal intensity profile @ y={row_y}")
    axes[1].set_xlabel("x (pixels)")
    axes[1].set_ylabel("pixel value")

    axes[2].plot(gray[:, col_x], color="green")
    axes[2].set_title(f"Vertical intensity profile @ x={col_x}")
    axes[2].set_xlabel("y (pixels)")
    axes[2].set_ylabel("pixel value")

    fig.tight_layout()
    path = os.path.join(IMAGES, "07_background_artifact_map.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    if not os.path.exists(path):
        FAILURES.append(f"matplotlib savefig failed: {path}")
    else:
        print(f"wrote {os.path.relpath(path, BASE)}")

def analysis_08_denoise_comparison(gray):
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    diff = cv2.absdiff(gray, denoised)
    diff_amp = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(denoised, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Non-local-means denoised")
    axes[1].axis("off")
    axes[2].imshow(diff_amp, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Removed content (amplified diff)")
    axes[2].axis("off")
    fig.tight_layout()
    path = os.path.join(IMAGES, "08_denoise_comparison.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    if not os.path.exists(path):
        FAILURES.append(f"matplotlib savefig failed: {path}")
    else:
        print(f"wrote {os.path.relpath(path, BASE)}")

def analysis_09_roi_tibia_ankle_junction(gray):
    """Iteration follow-up: Agent_02 revised its own read of image 05 (longitudinal
    lucency, not transverse) and Agent_04 disagreed (saw a smooth transverse band).
    Agent_02 requested a lower ROI closer to the ankle joint to resolve this."""
    h, w = gray.shape
    y0, y1 = int(h * 0.32), int(h * 0.60)
    x0, x1 = int(w * 0.15), int(w * 0.88)
    crop = gray[y0:y1, x0:x1]
    upscale = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    save(os.path.join(IMAGES, "09_roi_tibia_ankle_junction_zoom.png"), upscale,
         note=f"crop=({x0},{y0},{x1},{y1})")

def main():
    gray = load_gray()
    print(f"Loaded original: shape={gray.shape}, dtype={gray.dtype}")
    analysis_01_clahe(gray)
    analysis_02_unsharp(gray)
    analysis_03_edge_map(gray)
    analysis_04_roi_ankle_hindfoot(gray)
    analysis_05_roi_tibial_metaphysis(gray)
    analysis_06_histogram_contrast_stretch(gray)
    analysis_07_background_artifact_map(gray)
    analysis_08_denoise_comparison(gray)
    analysis_09_roi_tibia_ankle_junction(gray)

    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("\nAll analyses completed successfully.")

if __name__ == "__main__":
    main()
