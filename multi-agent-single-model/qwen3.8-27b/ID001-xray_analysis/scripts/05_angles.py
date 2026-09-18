"""[12] Quantitative angular measurement with landmark lines drawn on the native original + projection checks."""
import os
import sys
import math
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_rgb, save, BASE

rgb = load_rgb().copy()

# Landmarks (x, y) placed on the native original by numeric profile probing.
A = (154, 280)   # posterosuperior calcaneal tuberosity (Achilles insertion apex)
C = (162, 319)   # most inferior plantar point of calcaneus
B = (205, 301)   # anterior calcaneal tuberosity (rise of plantar line)
T1, T2 = (176, 45), (186, 158)      # tibial axis (shaft midpoints)
M1, M2 = (248, 298), (318, 330)     # metatarsal axis (proximal -> distal shaft)
TA1, TA2 = (192, 247), (208, 282)   # talar axis (dome -> head)
FIB, TIB = (216, 210), (198, 210)   # fibula / tibia centers at y=210 (rotation proxy)
THEAD = (200, 275)                   # talar head (rotation proxy marker)


def angle_between(p, q, r):
    """angle in degrees at vertex q between q->p and q->r"""
    v1 = (p[0] - q[0], p[1] - q[1])
    v2 = (r[0] - q[0], r[1] - q[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


bohler = angle_between(B, A, C)
pitch = abs(math.degrees(math.atan2(B[1] - C[1], B[0] - C[0])))
# tibiometatarsal: angle between tibial axis direction and metatarsal axis direction
d_tib = (T2[0] - T1[0], T2[1] - T1[1])
d_mt = (M2[0] - M1[0], M2[1] - M1[1])
cos_tib_mt = (d_tib[0] * d_mt[0] + d_tib[1] * d_mt[1]) / (math.hypot(*d_tib) * math.hypot(*d_mt))
tib_met = abs(math.degrees(math.acos(max(-1.0, min(1.0, cos_tib_mt)))))
# tibiocalcaneal: tibial axis vs posterior calcaneal line A->C
d_cal = (C[0] - A[0], C[1] - A[1])
cos_tib_cal = (d_tib[0] * d_cal[0] + d_tib[1] * d_cal[1]) / (math.hypot(*d_tib) * math.hypot(*d_cal))
tib_cal = abs(math.degrees(math.acos(max(-1.0, min(1.0, cos_tib_cal)))))
# talocalcaneal: talar axis vs A->C
d_tar = (TA2[0] - TA1[0], TA2[1] - TA1[1])
cos_tar_cal = (d_tar[0] * d_cal[0] + d_tar[1] * d_cal[1]) / (math.hypot(*d_tar) * math.hypot(*d_cal))
tar_cal = abs(math.degrees(math.acos(max(-1.0, min(1.0, cos_tar_cal)))))
fib_sep = FIB[0] - TIB[0]

YELLOW = (0, 255, 255)
GREEN = (0, 220, 0)
CYAN = (255, 200, 0)
MAGENTA = (255, 0, 220)
WHITE = (255, 255, 255)


def seg(p, q, color, w=1):
    cv2.line(rgb, p, q, color, w, lineType=cv2.LINE_AA)


def dot(p, color, r=3):
    cv2.circle(rgb, p, r, color, 2, lineType=cv2.LINE_AA)


# Böhler lines
seg(A, C, YELLOW); seg(A, B, YELLOW)
dot(A, YELLOW); dot(B, YELLOW); dot(C, YELLOW)
cv2.putText(rgb, "Bohler", (118, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.35, YELLOW, 1, cv2.LINE_AA)
# calcaneal pitch: plantar line C->B and horizontal reference at C
seg(C, B, GREEN)
seg((C[0] - 30, C[1]), (C[0] + 45, C[1]), GREEN)
cv2.putText(rgb, "pitch", (170, 332), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 1, cv2.LINE_AA)
# tibial axis
seg(T1, T2, CYAN)
cv2.putText(rgb, "tibia", (150, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CYAN, 1, cv2.LINE_AA)
# metatarsal axis
seg(M1, M2, CYAN)
# talar axis
seg(TA1, TA2, MAGENTA)
# rotation proxies
seg((TIB[0], 195), (TIB[0], 225), WHITE)
seg((FIB[0], 195), (FIB[0], 225), WHITE)
dot(THEAD, (0, 255, 255), 4)
cv2.putText(rgb, "talar head", (206, 272), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 255, 255), 1, cv2.LINE_AA)
cv2.putText(rgb, f"tib-fib gap={fib_sep}px", (222, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.32, WHITE, 1, cv2.LINE_AA)

save("12_angles_overlay.png", rgb)

report = f"""ANGLE MEASUREMENTS — ID001-xray.png (native-resolution landmarks, see 12_angles_overlay.png)
Landmarks (x,y): A posterosup calcaneal tuberosity={A}; C inferior plantar point={C};
B anterior calcaneal tuberosity={B}; tibial axis {T1}->{T2}; metatarsal axis {M1}->{M2};
talar axis {TA1}->{TA2}; fibula center x={FIB[0]} / tibia center x={TIB[0]} at y=210.

Boehler angle (A-B vs A-C):            {bohler:6.1f} deg   (normal ~20-40 deg)
Calcaneal pitch (plantar line vs horiz):{pitch:6.1f} deg   (normal ~15-30 deg)
Tibiometatarsal angle:                 {tib_met:6.1f} deg   (normal ~10-25 deg; estimate — MT axis truncated by crop)
Tibiocalcaneal angle:                  {tib_cal:6.1f} deg   (normal ~25-40 deg)
Talocalcaneal angle:                   {tar_cal:6.1f} deg   (normal ~25-50 deg)
Fibula-tibia center separation:        {fib_sep:6d} px     (true lateral: ~0; indicates off-true-lateral rotation)

CAVEATS:
- Landmark placement error ~+/-3-4 px -> angle uncertainty ~+/-5-8 deg (Böhler is the most sensitive).
- The measured fibula-tibia separation (~{fib_sep}px) shows the view is NOT a true lateral;
  ALL angles carry this projection bias (Agent_06 rule: call abnormal only if outside
  normal range by more than estimated landmark error).
- First-metatarsal axis truncated at the right-edge crop -> tibiometatarsal is an estimate.
- No physical scale; angles are scale-invariant so no mm conversion applies.
"""
with open(os.path.join(BASE, "planning", "angle_measurements.txt"), "w") as f:
    f.write(report)
print(report)
