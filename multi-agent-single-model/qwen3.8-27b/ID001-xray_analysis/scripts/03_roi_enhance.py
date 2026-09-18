"""[4] ankle/subtalar ROI CLAHE, [5] talus ROI CLAHE, [6]/[7] talus denoised mild/strong, [8] posterior calcaneus."""
import os
import sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_gray, clahe, crop_zoom, unsharp, save

g = load_gray()


def zoom2(region):
    return cv2.resize(region, (region.shape[1] * 2, region.shape[0] * 2), interpolation=cv2.INTER_NEAREST)


# [4] tibiotalar joint / talar dome / posterior subtalar / sinus tarsi
x0, y0, x1, y1 = 140, 225, 240, 315
save("04_roi_ankle_subtalar_clahe.png", zoom2(clahe(g[y0:y1, x0:x1], clip=3.0)))

# [5] talar body/neck
x0, y0, x1, y1 = 140, 235, 230, 300
talus = g[y0:y1, x0:x1]
save("05_roi_talus_clahe.png", zoom2(clahe(talus, clip=3.0)))

# [6] talus denoised mild (bilateral d=5, sigmaColor=30, sigmaSpace=30)
dn_mild = cv2.bilateralFilter(talus, d=5, sigmaColor=30, sigmaSpace=30)
save("06_roi_talus_denoised_mild.png", zoom2(clahe(dn_mild, clip=3.0)))

# [7] talus denoised strong (bilateral d=9, sigmaColor=50, sigmaSpace=50)
dn_strong = cv2.bilateralFilter(talus, d=9, sigmaColor=50, sigmaSpace=50)
save("07_roi_talus_denoised_strong.png", zoom2(clahe(dn_strong, clip=3.0)))

# [8] posterior calcaneus (Achilles insertion, posteroinferior corner, plantar fascia origin)
x0, y0, x1, y1 = 115, 255, 200, 335
heel = g[y0:y1, x0:x1]
heel_e = unsharp(clahe(heel, clip=3.0), sigma=1.5, amount=0.8)
save("08_roi_posterior_calcaneus_clahe.png", zoom2(heel_e))
