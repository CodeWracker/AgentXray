"""[2] mild and [3] moderate whole-image CLAHE bracket."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_gray, clahe, save

g = load_gray()
save("02_clahe_mild_full.png", clahe(g, clip=2.0))
save("03_clahe_moderate_full.png", clahe(g, clip=3.5))
