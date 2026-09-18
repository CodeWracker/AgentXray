"""[15] Noise floor / practical detection limit, measured on the ORIGINAL only. Outputs a chart figure."""
import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_gray, save, BASE

g = load_gray()

rois = {
    "upper-left": (60, 180, 30, 100),
    "lower-left": (220, 340, 30, 100),
    "upper-right": (20, 200, 330, 365),
}
sigmas = {}
for name, (y0, y1, x0, x1) in rois.items():
    patch = g[y0:y1, x0:x1].astype(np.float64)
    sigmas[name] = float(patch.std())
    print(f"background ROI {name}: mean={patch.mean():.1f} sigma={patch.std():.2f}")


def edge_fwhm(profile, x0, y0, x1, direction="rising"):
    p = profile.astype(np.float64)
    grad = np.gradient(p)
    i = int(np.argmax(np.abs(grad)))
    seg = p[max(0, i - 10):i + 10]
    lo, hi = seg.min(), seg.max()
    if hi - lo < 5:
        return None, i
    rising = seg >= (lo + 0.5 * (hi - lo))
    idx = np.where(rising)[0]
    return (idx.max() - idx.min() + 1) if len(idx) else None, i + max(0, i - 10)


# vertical edge: posterior calcaneal cortical rim at y=295 (profile across x)
fw1, _ = edge_fwhm(g[295, 135:170], 135, 295, 170)
# vertical edge: anterior tibial cortex mid-shaft at y=120 (profile across x)
fw2, _ = edge_fwhm(g[120, 195:235], 195, 120, 235)
# horizontal edge: distal tibial plafond at x=190 (profile across y)
fw3, _ = edge_fwhm(g[225:250, 190], 225, 190, 250)
print("edge FWHM px: calcaneal rim:", fw1, " tibial cortex y120:", fw2, " tibial plafond x190:", fw3)

sigma_bg = float(np.mean(list(sigmas.values())))
fwhm = float(np.mean([v for v in (fw1, fw2, fw3) if v]))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(sigmas.keys(), sigmas.values(), color="#4878a8")
axes[0].set_title("Background noise sigma (grey levels), anatomy-free ROIs (original)")
axes[0].set_ylabel("sigma")
for i, v in enumerate(sigmas.values()):
    axes[0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
vals = [v for v in (fw1, fw2, fw3) if v]
axes[1].bar(["calcaneal\nrim y=295", "tibial cortex\ny=120", "tibial\nplafond x=190"], vals, color="#a87848")
axes[1].set_title("Edge spread (FWHM, px) at sharp cortices (original)")
axes[1].set_ylabel("FWHM px")
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.05, f"{v}px", ha="center", fontsize=9)
fig.suptitle(f"Detection limit estimate: sigma_bg~{sigma_bg:.1f} grey, edge FWHM~{fwhm:.1f}px\n"
             f"A line shorter than ~{2*fwhm:.0f}px or with contrast <~{3*sigma_bg:.0f} grey levels is not reliably detectable.",
             fontsize=10)
fig.tight_layout()
path = os.path.join(BASE, "images", "15_noise_floor_chart.png")
fig.savefig(path, dpi=150)
print("saved", path)

with open(os.path.join(BASE, "planning", "noise_floor.txt"), "w") as f:
    f.write(f"noise sigma per ROI: {sigmas}\nedge FWHM px: calcaneal={fw1}, tibial_cortex_y120={fw2}, tibial_plafond_x190={fw3}\n"
            f"mean sigma_bg={sigma_bg:.2f}; mean FWHM={fwhm:.2f}px\n"
            f"practical detection limit: line length <~{2*fwhm:.0f}px or contrast <~{3*sigma_bg:.0f} grey levels not reliably detectable\n")
