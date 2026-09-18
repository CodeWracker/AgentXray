"""Analysis A: numeric characterization of the original image.

Reads images/00_original.png, writes measurements/01_characterization.json.
Deterministic: no randomness involved.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ORIG = ROOT / "images" / "00_original.png"
OUT = ROOT / "measurements" / "01_characterization.json"


def main():
    im = Image.open(ORIG).convert("L")
    a = np.asarray(im, dtype=np.float64)
    h, w = a.shape

    # sharpness proxy: variance of Laplacian
    from scipy.ndimage import laplace

    lap_var = float(np.var(laplace(a)))

    # local contrast: mean absolute difference between neighbors
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()

    # fraction of pixels in the mid-gray band (tissue vs pure background)
    frac_bg = float((a < 20).mean())
    frac_bright = float((a > 180).mean())
    frac_mid = float(((a >= 20) & (a <= 180)).mean())

    # histogram (32 bins)
    hist, _ = np.histogram(a, bins=32, range=(0, 256))
    hist = [int(v) for v in hist]

    out = {
        "image": ORIG.name,
        "size_w": w,
        "size_h": h,
        "channels": 3,
        "bit_depth": 8,
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "percentiles": {
            "p05": float(np.percentile(a, 5)),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "p995": float(np.percentile(a, 99.5)),
        },
        "laplacian_variance": lap_var,
        "mean_abs_grad_x": float(gx),
        "mean_abs_grad_y": float(gy),
        "frac_dark_bg_lt20": frac_bg,
        "frac_midgray_20_180": frac_mid,
        "frac_bright_gt180": frac_bright,
        "histogram_32bins": hist,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
