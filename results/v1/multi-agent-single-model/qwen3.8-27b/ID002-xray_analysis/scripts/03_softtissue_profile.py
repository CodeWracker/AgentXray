import sys
sys.path.insert(0, "/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray_analysis/scripts")
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load_gray, IMAGES

arr = load_gray()
H, W = arr.shape
mask = arr > 6

def largest_runs_line(line, min_len=60):
    runs = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] > 6:
            j = i
            while j < n and line[j] > 6:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs

# ---- distal (forearm): row widths, y 480..1240 ----
d_pos, d_w, d_left, d_right = [], [], [], []
for y in range(480, 1240, 4):
    runs = largest_runs_line(arr[y], min_len=80)
    if not runs:
        continue
    i, j = max(runs, key=lambda r: r[1] - r[0])
    d_pos.append(y)
    d_w.append(j - i)
    d_left.append(i)
    d_right.append(j)
d_pos = np.array(d_pos); d_w = np.array(d_w)
d_left = np.array(d_left); d_right = np.array(d_right)

# ---- proximal (humerus/upper arm): column widths, x 1080..1560 ----
p_pos, p_w, p_top, p_bot = [], [], [], []
for x in range(1080, 1560, 4):
    runs = largest_runs_line(arr[:, x], min_len=60)
    if not runs:
        continue
    i, j = max(runs, key=lambda r: r[1] - r[0])
    if j - i < 80:
        continue
    p_pos.append(x)
    p_w.append(j - i)
    p_top.append(i)
    p_bot.append(j)
p_pos = np.array(p_pos); p_w = np.array(p_w)
p_top = np.array(p_top); p_bot = np.array(p_bot)

# joint origin for the signed distance axis
y0 = 480
x0 = 1080
dist_d = d_pos - y0          # distal: positive
dist_p = -(p_pos - x0)       # proximal: negative
w_all = np.concatenate([p_w, d_w])
dist_all = np.concatenate([dist_p, dist_d])
peak_i = int(np.argmax(w_all))
peak_w = w_all[peak_i]
peak_dist = dist_all[peak_i]
w_midforearm = float(np.interp(400, dist_d, d_w))   # 400px distal to origin
ratio = peak_w / w_midforearm

# ---- figure ----
fig, ax2 = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.4, 1]})
a = np.clip(arr, 0, 255).astype(np.uint8)
rgb = np.stack([a]*3, axis=-1)
ax2[0].imshow(rgb)
ax2[0].plot(d_left, d_pos, "r-", lw=1.2)
ax2[0].plot(d_right, d_pos, "r-", lw=1.2)
ax2[0].plot(p_pos, p_top, "c-", lw=1.2)
ax2[0].plot(p_pos, p_bot, "c-", lw=1.2)
ax2[0].set_title("traced soft-tissue contour (red=forearm rows, cyan=humerus columns)")
ax2[0].set_xlim(0, W); ax2[0].set_ylim(H, 0)

ax2[1].plot(dist_d, d_w, "b-", label="distal (forearm)")
ax2[1].plot(dist_p, p_w, "g-", label="proximal (humerus)")
ax2[1].axhline(w_midforearm, color="k", ls=":", lw=1)
ax2[1].set_xlabel("signed distance from joint origin (px); + distal, - proximal")
ax2[1].set_ylabel("soft-tissue width (px)")
ax2[1].set_title(f"peak width {peak_w:.0f}px at {peak_dist:+.0f}px; mid-forearm(400px) {w_midforearm:.0f}px; ratio {ratio:.2f}")
ax2[1].legend()
ax2[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{IMAGES}/08_softtissue_profile.png", dpi=110)
print("saved 08_softtissue_profile.png")
print(f"peak width {peak_w:.0f}px at signed distance {peak_dist:+.0f}px (origin y={y0},x={x0})")
print(f"mid-forearm (400px distal): {w_midforearm:.0f}px ; peak/mid-forearm ratio: {ratio:.2f}")
print("distal sample widths:", [(int(d), int(w)) for d, w in list(zip(dist_d, d_w))[::8]])
print("proximal sample widths:", [(int(d), int(w)) for d, w in list(zip(dist_p, p_w))[::8]])
