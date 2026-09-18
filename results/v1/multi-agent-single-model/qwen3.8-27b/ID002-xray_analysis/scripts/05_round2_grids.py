import sys
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray_analysis/scripts")
from common import load_gray, save_gray, save_rgb, crop, upscale_nearest, clahe, IMAGES, ORIG

arr = load_gray()
im = Image.open(ORIG)

def add_grid(c, x0, y0, scale, grid=50):
    d = ImageDraw.Draw(c)
    for fx in range(int(np.ceil(x0 / grid)) * grid, x0 + c.width // scale + grid, grid):
        zx = int(round((fx - x0) * scale))
        if 0 <= zx < c.width:
            d.line([(zx, 0), (zx, c.height)], fill=(0, 255, 0), width=1)
            d.text((zx + 2, 4), str(int(fx)), fill=(0, 255, 0))
    for fy in range(int(np.ceil(y0 / grid)) * grid, y0 + c.height // scale + grid, grid):
        zy = int(round((fy - y0) * scale))
        if 0 <= zy < c.height:
            d.line([(0, zy), (c.width, zy)], fill=(0, 255, 0), width=1)
            d.text((2, zy + 2), str(int(fy)), fill=(0, 255, 0))
    return c

def grid_zoom(box, scale, name):
    x0, y0, x1, y1 = box
    c = im.crop(box).resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.LANCZOS)
    c = add_grid(c, x0, y0, scale)
    c.save(f"{IMAGES}/{name}")
    print(f"saved {name} {c.size}")

# P12: wrist distal zoom 3x with 50px grid (proximal shaft included for identity tracing)
grid_zoom((700, 900, 1150, 1280), 3, "12_wrist_distal_zoom_grid.png")

# P13: olecranon focus zoom 3x with 50px grid (wider ROI incl. adjacent cortex for control)
grid_zoom((840, 150, 1100, 420), 3, "13_olecranon_focus_zoom_grid.png")

# P16: olecranon tip 1:1 native vs 4x nearest-neighbor side-by-side panel
tip_box = (880, 160, 1060, 340)
left = im.crop(tip_box)
x0, y0 = tip_box[0], tip_box[1]
right = upscale_nearest(crop(arr, tip_box), 4).astype(np.uint8)
panel = np.zeros((right.shape[0], left.width + right.shape[1] + 20, 3), dtype=np.uint8)
la = np.stack([np.array(left.convert("L"))] * 3, axis=-1)
panel[:left.height, :left.width] = la
panel[:, left.width + 20:] = np.stack([right] * 3, axis=-1)
pimg = Image.fromarray(panel, mode="RGB")
d = ImageDraw.Draw(pimg)
d.text((10, 10), "1:1 native (original pixels)", fill=(255, 255, 0))
d.text((left.width + 30, 10), "4x nearest-neighbor (no contrast change)", fill=(255, 255, 0))
pimg.save(f"{IMAGES}/16_olecranon_tip_native_1to1.png")
print("saved 16_olecranon_tip_native_1to1.png", pimg.size)

# P17: full forearm continuity strip 2x with 50px grid (elbow->wrist, both shafts)
grid_zoom((720, 400, 1120, 1280), 2, "17_forearm_continuity_strip_grid.png")
print("done")
