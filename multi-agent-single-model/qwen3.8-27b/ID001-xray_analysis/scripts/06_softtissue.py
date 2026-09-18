"""[13] Soft-tissue window (bone-saturating) + skin-line contour overlay + standardized thickness stations."""
import os
import sys
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_gray, save, BASE

g = load_gray()
h, w = g.shape

# bone-saturating display window: bone (~>150) -> white, soft tissue mid-tone
disp = np.clip(127 + (g.astype(np.float32) - 95.0) * 2.3, 0, 255).astype("uint8")
out = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)

# local background from margins per row (left margin) and per column (top/bottom margins)
def row_bg(y):
    return float(np.median(g[y, 5:25]))

def col_bg_top(x):
    return float(np.median(g[5:25, x]))

def col_bg_bot(x):
    return float(np.median(g[h-25:h-5, x]))

post_skin = {}
for y in range(250, 346):
    bg = row_bg(y)
    row = g[y, 100:300]
    xs = np.where(row > bg + 8)[0]
    if len(xs):
        post_skin[y] = 100 + int(xs[0])

dorsal_skin = {}
for x in range(150, 372):
    bg = col_bg_top(x)
    col = g[200:345, x]
    ys = np.where(col > bg + 8)[0]
    if len(ys):
        dorsal_skin[x] = 200 + int(ys[0])

plantar_skin = {}
for x in range(140, 341):
    bg = col_bg_bot(x)
    col = g[345:h-20, x]
    ys = np.where(col > bg + 8)[0]
    if len(ys):
        plantar_skin[x] = 345 + int(ys.max())

YELLOW = (0, 255, 255)
RED = (0, 0, 255)

# draw contours
for y, x in sorted(post_skin.items()):
    cv2.circle(out, (x, y), 1, YELLOW, -1)
for x, y in sorted(dorsal_skin.items()):
    cv2.circle(out, (x, y), 1, YELLOW, -1)
for x, y in sorted(plantar_skin.items()):
    cv2.circle(out, (x, y), 1, YELLOW, -1)

def bone_x_right(y, x_start):
    # posterior calcaneal cortex here measures ~55-75 grey (below the 80 used for
    # midfoot bone); soft tissue in this station is <= ~47, so threshold 55
    row = g[y, x_start:320]
    xs = np.where(row >= 55)[0]
    return x_start + int(xs[0]) if len(xs) else None

def bone_y_down(x, y_start):
    col = g[y_start:345, x]
    ys = np.where(col >= 80)[0]
    return y_start + int(ys[0]) if len(ys) else None

notes = []
# heel-pad stations (posterior skin line -> posterior calcaneal cortex), thickness in px
for y in (295, 305, 315):
    if y in post_skin:
        xs = post_skin[y]
        xc = bone_x_right(y, xs + 2)
        if xc:
            th = xc - xs
            cv2.line(out, (xs, y), (xc, y), RED, 1)
            cv2.putText(out, f"hp y{y}:{th}px", (xs - 55, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1, cv2.LINE_AA)
            notes.append(f"heel pad at y={y}: skin x={xs}, cortex x={xc}, thickness={th}px")
# dorsal band stations (dorsal skin line -> first bone below), thickness in px
# x must be beyond the distal tibia (x>215) to measure foot, not leg, soft tissue
for x in (230, 260, 300):
    if x in dorsal_skin:
        ys = dorsal_skin[x]
        yb = bone_y_down(x, ys + 2)
        if yb:
            th = yb - ys
            cv2.line(out, (x, ys), (x, yb), RED, 1)
            cv2.putText(out, f"ds x{x}:{th}px", (x + 3, ys + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.3, RED, 1, cv2.LINE_AA)
            notes.append(f"dorsal band at x={x}: skin y={ys}, bone y={yb}, thickness={th}px")

save("13_softtissue_contour.png", out)
print("\n".join(notes))
with open(os.path.join(BASE, "planning", "softtissue_measurements.txt"), "w") as f:
    f.write("SOFT-TISSUE MEASUREMENTS (pixels; no physical scale in file)\n")
    f.write("\n".join(notes) + "\n")
