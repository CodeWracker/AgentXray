import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray_analysis/scripts")
from common import load_gray, IMAGES

arr = load_gray()


def profile(p1, p2, step=0.5):
    x1, y1 = p1
    x2, y2 = p2
    length = np.hypot(x2 - x1, y2 - y1)
    n = int(length / step)
    xs = x1 + (x2 - x1) * np.linspace(0, 1, n)
    ys = y1 + (y2 - y1) * np.linspace(0, 1, n)
    x0i = np.clip(np.floor(xs).astype(int), 0, arr.shape[1] - 1)
    y0i = np.clip(np.floor(ys).astype(int), 0, arr.shape[0] - 1)
    x1i = np.clip(x0i + 1, 0, arr.shape[1] - 1)
    y1i = np.clip(y0i + 1, 0, arr.shape[0] - 1)
    dx = xs - x0i
    dy = ys - y0i
    vals = (arr[y0i, x0i] * (1 - dx) * (1 - dy) + arr[y0i, x1i] * dx * (1 - dy) +
            arr[y1i, x0i] * (1 - dx) * dy + arr[y1i, x1i] * dx * dy)
    return np.linspace(0, length, n), vals


# ---- olecranon cleft profiles (perpendicular to the transverse cleft at ~(968,261)) ----
olec = []
for x in [955, 968, 982]:
    d, v = profile((x, 233), (x, 297))
    olec.append((d, v, x))
olec_ctrl_d, olec_ctrl_v = profile((930, 233), (930, 297))

# ---- wrist cleft profiles (distal ulna cleft ~(850,1152), oblique dir (0.55,0.83); perpendicular (0.83,-0.55)) ----
pd = np.array([0.83, -0.55])
cd = np.array([0.55, 0.83])
cleft_c = np.array([850.0, 1152.0])
wrist = []
for off in [-10, 0, 10]:
    c = cleft_c + off * cd
    p1 = tuple(c - 32 * pd)
    p2 = tuple(c + 32 * pd)
    d, v = profile(p1, p2)
    wrist.append((d, v, off))

# ---- control: normal distal RADIAL physis (left bone, upper metaphysis band ~(780-820, 1095-1150)) ----
radph = []
for x in [785, 800, 815]:
    d, v = profile((x, 1095), (x, 1150))
    radph.append((d, v, x))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
for d, v, x in olec:
    ax.plot(d, v, label=f"olecranon cleft profile x={x}")
ax.plot(olec_ctrl_d, olec_ctrl_v, "k--", label="control: olecranon body (x=930)")
ax.set_title("Olecranon tip: 3 parallel profiles across the cleft + control\n(intensity vs distance; cleft spans y=233->297, expect dip ~y=255-270)")
ax.set_xlabel("distance along profile (px)"); ax.set_ylabel("intensity (0-255)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for d, v, off in wrist:
    ax.plot(d, v, label=f"distal-ulna cleft profile (offset {off:+d})")
ax.set_title("Wrist cleft (distal ulna): 3 parallel perpendicular profiles\n(cleft center ~(850,1152); profile length 64px)")
ax.set_xlabel("distance along profile (px)"); ax.set_ylabel("intensity (0-255)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
for d, v, x in radph:
    ax.plot(d, v, label=f"normal distal radial physis x={x}")
ax.set_title("Control: normal distal RADIAL physis (3 parallel profiles)\n(reference morphology of an unfused physis in this child)")
ax.set_xlabel("distance along profile (px)"); ax.set_ylabel("intensity (0-255)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.suptitle("P15: pixel-intensity profiles on the ORIGINAL image (no processing) - physis vs fracture discrimination", y=1.02)
fig.tight_layout()
fig.savefig(f"{IMAGES}/15_intensity_profiles.png", dpi=120, bbox_inches="tight")
print("saved 15_intensity_profiles.png")

for d, v, x in olec:
    imin = int(np.argmin(v))
    print(f"olecranon x={x}: min intensity {v[imin]:.0f} at d={d[imin]:.1f}px (y~{233+d[imin]:.0f}); endpoints {v[0]:.0f}/{v[-1]:.0f}")
for d, v, off in wrist:
    imin = int(np.argmin(v))
    print(f"wrist cleft off={off:+d}: min {v[imin]:.0f} at d={d[imin]:.1f}/64; endpoints {v[0]:.0f}/{v[-1]:.0f}")
for d, v, x in radph:
    imin = int(np.argmin(v))
    print(f"radial physis x={x}: min {v[imin]:.0f} at d={d[imin]:.1f}/55; endpoints {v[0]:.0f}/{v[-1]:.0f}")
