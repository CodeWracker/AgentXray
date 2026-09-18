"""Analysis 7: radiocapitellar-line / anterior-humeral-line geometric overlay.

Landmark points are derived with simple image heuristics (bright-pixel
centroid for the capitellum, PCA principal axis for shaft directions) rather
than verified anatomic segmentation. This is a first-pass geometric aid, not
a validated measurement -- flagged explicitly in the caption per the
positioning caveat raised independently by Agent_03/Agent_06 and both
council reviewers (elbow flexion beyond the canonical ~90 degrees; possible
obliquity not excluded).
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from common import ORIGINAL, IMAGES_DIR
import os

im_gray = np.array(Image.open(ORIGINAL).convert("L"))
im_rgb = Image.open(ORIGINAL).convert("RGB")
draw = ImageDraw.Draw(im_rgb)

# --- capitellum centroid: brightest blob near the joint articulation ---
capitellum_box = (860, 130, 1020, 280)  # x0,y0,x1,y1
sub = im_gray[capitellum_box[1]:capitellum_box[3], capitellum_box[0]:capitellum_box[2]]
mask = sub > 210
ys, xs = np.where(mask)
if len(xs) > 0:
    cap_x = capitellum_box[0] + xs.mean()
    cap_y = capitellum_box[1] + ys.mean()
else:
    cap_x = (capitellum_box[0] + capitellum_box[2]) / 2
    cap_y = (capitellum_box[1] + capitellum_box[3]) / 2

# --- proximal humeral shaft axis (anterior humeral line proxy) ---
humerus_box = (1250, 60, 1599, 260)
sub_h = im_gray[humerus_box[1]:humerus_box[3], humerus_box[0]:humerus_box[2]]
mask_h = sub_h > 150
ys_h, xs_h = np.where(mask_h)
pts_h = np.column_stack([xs_h + humerus_box[0], ys_h + humerus_box[1]]).astype(np.float64)
mean_h = pts_h.mean(axis=0)
cov_h = np.cov((pts_h - mean_h).T)
eigvals_h, eigvecs_h = np.linalg.eigh(cov_h)
dir_h = eigvecs_h[:, np.argmax(eigvals_h)]

# --- forearm shaft axis (radial axis proxy: combined radius+ulna PCA) ---
forearm_box = (780, 350, 1000, 950)
sub_f = im_gray[forearm_box[1]:forearm_box[3], forearm_box[0]:forearm_box[2]]
mask_f = sub_f > 140
ys_f, xs_f = np.where(mask_f)
pts_f = np.column_stack([xs_f + forearm_box[0], ys_f + forearm_box[1]]).astype(np.float64)
mean_f = pts_f.mean(axis=0)
cov_f = np.cov((pts_f - mean_f).T)
eigvals_f, eigvecs_f = np.linalg.eigh(cov_f)
dir_f = eigvecs_f[:, np.argmax(eigvals_f)]


def extend_line(mean, direction, length=900):
    p1 = mean - direction * length / 2
    p2 = mean + direction * length / 2
    return tuple(p1), tuple(p2)


h_p1, h_p2 = extend_line(mean_h, dir_h)
f_p1, f_p2 = extend_line(mean_f, dir_f)

draw.line([h_p1, h_p2], fill=(255, 60, 60), width=4)  # anterior humeral line proxy
draw.line([f_p1, f_p2], fill=(60, 160, 255), width=4)  # radiocapitellar / forearm axis proxy
r = 10
draw.ellipse([cap_x - r, cap_y - r, cap_x + r, cap_y + r], outline=(50, 255, 50), width=4)

caption = (
    "Red: proximal humeral shaft axis (anterior humeral line proxy).  "
    "Blue: forearm shaft axis (radiocapitellar line proxy, radius+ulna combined -- "
    "not separately isolated).  Green circle: capitellum centroid (heuristic, "
    "bright-pixel based).  Elbow flexion appears > canonical 90 deg; obliquity not "
    "excluded -- treat any apparent malalignment as provisional, not diagnostic."
)

# add caption band below image
w, h = im_rgb.size
band_h = 90
canvas = Image.new("RGB", (w, h + band_h), (0, 0, 0))
canvas.paste(im_rgb, (0, 0))
cdraw = ImageDraw.Draw(canvas)
cdraw.text((10, h + 8), caption, fill=(255, 255, 255))

out = os.path.join(IMAGES_DIR, "07_alignment_lines_overlay.png")
canvas.save(out)
print("wrote", out, canvas.size, "capitellum=", (cap_x, cap_y))
