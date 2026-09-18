"""Shared constants for the ID002-xray processing pipeline."""
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE, "images")
ORIGINAL = os.path.join(IMAGES_DIR, "00_original.png")

# Elbow joint region of interest (distal humerus / olecranon / radial head /
# proximal radius-ulna + adjacent fat-pad soft tissue), derived empirically
# from the bright-pixel bounding box of the joint articulation cluster.
JOINT_BOX = (740, 30, 1220, 540)  # left, top, right, bottom

# Bottom-edge partial carpal/wrist fragment.
WRIST_BOX = (690, 970, 970, 1280)

# Radial-neck / coronoid / olecranon sub-region (tighter crop within the
# joint box, centered on the classic occult-fracture sites just below the
# main joint line).
FRACTURE_SITE_BOX = (800, 180, 1120, 480)
