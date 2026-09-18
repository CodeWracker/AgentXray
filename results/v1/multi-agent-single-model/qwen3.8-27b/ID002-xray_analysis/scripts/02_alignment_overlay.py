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

# ---- data-driven shaft fits on the original ----
def band_centers_axis_y(y0, y1, x0, x1, thresh=150, gap=20):
    """return list of (y, [x-centers of bright bands])"""
    out = []
    for y in range(y0, y1, 4):
        row = arr[y, x0:x1] > thresh
        xs = np.where(row)[0]
        if len(xs) == 0:
            continue
        bands = []
        start = xs[0]
        prev = xs[0]
        for x in xs[1:]:
            if x - prev > gap:
                bands.append((start, prev))
                start = x
            prev = x
        bands.append((start, prev))
        out.append((y, [ (a + b) / 2 + x0 for a, b in bands if (b - a) > 6 ]))
    return out

def fit_axis(pts):
    y = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])
    A = np.polyfit(y, x, 1)
    return A  # x = A[0]*y + A[1]

# left forearm bone = ulna (continues from olecranon), right = radius (continues from radial head)
# proximal fit range (closest to the joint, least affected by overall forearm tilt in frame)
rad_pts = []
uln_pts = []
for y, centers in band_centers_axis_y(420, 750, 740, 1060):
    if len(centers) == 2:
        uln_pts.append((y, centers[0]))
        rad_pts.append((y, centers[1]))

rad_axis = fit_axis(rad_pts)
uln_axis = fit_axis(uln_pts)

# separation of the two axes at reference level y=700 (perpendicular approx: dx)
yref = 700
sep = abs((rad_axis[0]*yref + rad_axis[1]) - (uln_axis[0]*yref + uln_axis[1]))

# humeral shaft: rows 150-300, cols 1250-1560
hum_centers = []
for x in range(1250, 1560, 4):
    col = arr[150:300, x] > 150
    ys = np.where(col)[0]
    if len(ys) > 20:
        hum_centers.append((x, ys.mean() + 150, ys.min() + 150, ys.max() + 150))
hx = np.array([c[0] for c in hum_centers])
hmid = np.array([c[1] for c in hum_centers])
hupper = np.array([c[2] for c in hum_centers])
hlower = np.array([c[3] for c in hum_centers])
hum_axis = np.polyfit(hx, hmid, 1)
hum_upper = np.polyfit(hx, hupper, 1)
hlower = np.polyfit(hx, hlower, 1)

# ---- angles (direction vectors) ----
# humerus fitted as y = m*x + c -> direction (1, m); radius/ulna as x = m*y + c -> direction (m, 1)
def ang_between(v1, v2):
    d1 = np.array(v1, float); d1 /= np.linalg.norm(d1)
    d2 = np.array(v2, float); d2 /= np.linalg.norm(d2)
    return np.degrees(np.arccos(np.clip(abs(np.dot(d1, d2)), 0, 1)))

hum_dir = (1.0, hum_axis[0])
rad_dir = (rad_axis[0], 1.0)
uln_dir = (uln_axis[0], 1.0)
ang_rad = ang_between(hum_dir, rad_dir)
ang_uln = ang_between(hum_dir, uln_dir)

# ---- manual landmarks (verified visually against gridded 4x joint zoom) ----
# posterior = upper side of image (olecranon/fossa above, coronoid below)
RAD_HEAD_C = (973, 237)    # radial head center (x,y)
CAP_CENTER = (985, 175)    # capitellum condyle center (incl. ossific center)
CAP_RADIUS = 40            # approx capitellum radius in px
CAP_OC = (971, 172)        # round ossification center (disc) within capitellum
CAP_OC_R = 25
OLEC_TIP = (945, 200)      # olecranon tip (for reference marking)

# ---- render overlay on original ----
rgb = np.stack([arr]*3, axis=-1).astype(np.uint8)
import cv2
def line(p, q, color, w=3):
    cv2.line(rgb, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, w)

# radiocapitellar line: through RAD_HEAD_C along radial axis
y1, y2 = 120, 950
xr1 = rad_axis[0]*y1 + rad_axis[1]
xr2 = rad_axis[0]*y2 + rad_axis[1]
# shift axis line so it passes through radial head center
dxh = RAD_HEAD_C[0] - (rad_axis[0]*RAD_HEAD_C[1] + rad_axis[1])
line((xr1+dxh, y1), (xr2+dxh, y2), (0, 255, 0), 3)

# posterior humeral line: humeral cortex on the OLECRANON (fossa) side, extended to condylar level.
# On this rotated view the olecranon lies below/left of the condyles, so the olecranon-side
# cortical edge is the lower fitted edge (hlower). Both edges are drawn for transparency:
x1, x2 = 1550, 950
line((x1, hlower[0]*x1 + hlower[1]), (x2, hlower[0]*x2 + hlower[1]), (255, 80, 80), 3)
line((x1, hum_upper[0]*x1 + hum_upper[1]), (x2, hum_upper[0]*x2 + hum_upper[1]), (255, 160, 160), 2)

# humeral axis
line((1550, hum_axis[0]*1550 + hum_axis[1]), (1000, hum_axis[0]*1000 + hum_axis[1]), (255, 255, 0), 2)
# radial axis and ulnar axis (proximal-mid)
line((rad_axis[0]*420 + rad_axis[1], 420), (rad_axis[0]*950 + rad_axis[1], 950), (0, 255, 255), 2)
line((uln_axis[0]*420 + uln_axis[1], 420), (uln_axis[0]*950 + uln_axis[1], 950), (160, 0, 255), 2)

# landmark markers
for p, c in [(RAD_HEAD_C, (0,255,0)), (CAP_CENTER, (0,255,255)), (OLEC_TIP, (255,255,255))]:
    cv2.circle(rgb, (int(p[0]), int(p[1])), 7, c, 2)
cv2.circle(rgb, (int(CAP_CENTER[0]), int(CAP_CENTER[1])), CAP_RADIUS, (0,255,255), 1)
cv2.circle(rgb, (int(CAP_OC[0]), int(CAP_OC[1])), CAP_OC_R, (0,128,255), 1)

# distance from radiocapitellar line (x = m*y + c) to capitellum center (perpendicular)
def dist_xform(px, py, m, c):
    return abs(px - m*py - c) / np.sqrt(1 + m*m)
d_rad = dist_xform(CAP_CENTER[0], CAP_CENTER[1], rad_axis[0], rad_axis[1] + dxh)
# posterior humeral line (olecranon-side = hlower) fitted as y = m*x + c  ->  m*x - y + c = 0
def dist_yform(px, py, m, c):
    return abs(m*px - py + c) / np.sqrt(m*m + 1)
d_post = dist_yform(CAP_CENTER[0], CAP_CENTER[1], hlower[0], hlower[1])
# also vs the ossific center disc, and vs the other (upper) cortical edge for comparison
d_post_oc = dist_yform(CAP_OC[0], CAP_OC[1], hlower[0], hlower[1])
d_rad_oc = dist_xform(CAP_OC[0], CAP_OC[1], rad_axis[0], rad_axis[1] + dxh)
d_upper = dist_yform(CAP_CENTER[0], CAP_CENTER[1], hum_upper[0], hum_upper[1])

# ---- annotations ----
img = Image.fromarray(rgb)
d = ImageDraw = None
from PIL import ImageDraw
dr = ImageDraw.Draw(img)
note1 = f"radiocapitellar line: offset from capitellum center = {d_rad:.1f}px (cap radius ~{CAP_RADIUS}px); vs ossific center = {d_rad_oc:.1f}px"
note2 = f"post.humeral line (olecranon side): offset from capitellum center = {d_post:.1f}px; upper cortical edge (ref) = {d_upper:.1f}px"
note3 = f"humeral-radial angle = {ang_rad:.1f} deg | humeral-ulnar angle = {ang_uln:.1f} deg"
note4 = f"radial-ulnar separation at y={yref}: {sep:.0f}px (obliquity indicator; NOT a true lateral)"
note5 = "CAVEAT: rotated/oblique view - line criteria are approximate; verify on unannotated original"
for i, t in enumerate([note1, note2, note3, note4, note5]):
    dr.text((20, 20 + 20*i), t, fill=(255, 255, 0))
dr.text((20, 20 + 20*5), "green=radiocapitellar  red=post.humeral(olecranon side)  pink=upper edge  yellow=hum.axis  cyan=rad.axis  purple=uln.axis", fill=(255, 255, 0))
img.save(f"{IMAGES}/07_alignment_overlay.png")
print("saved 07_alignment_overlay.png")
print(f"radial axis x={rad_axis[0]:.4f}*y+{rad_axis[1]:.1f}")
print(f"ulnar  axis x={uln_axis[0]:.4f}*y+{uln_axis[1]:.1f}")
print(f"hum axis (x->y)  y={hum_axis[0]:.4f}*x+{hum_axis[1]:.1f}")
print(f"hum upper cortex y={hum_upper[0]:.4f}*x+{hum_upper[1]:.1f}")
print(f"angles: hum-rad={ang_rad:.1f} hum-uln={ang_uln:.1f}")
print(f"separation at y={yref}: {sep:.0f}px")
print(f"dist rad-cap: {d_rad:.1f}px ; dist post-cap (olecranon-side): {d_post:.1f}px ; upper edge: {d_upper:.1f}px")
print(f"vs ossific center: rad {d_rad_oc:.1f}px ; post {d_post_oc:.1f}px")
