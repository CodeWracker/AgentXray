import sys, json, os
sys.path.insert(0, "/home/ralph/projects/ufsc/pablo-xray-tests/ID002-xray_analysis/scripts")
from PIL import Image
from common import IMAGES, IMAGES as IMG

EXPECTED = [
    "00_original.png",
    "01_elbow_crop_native.png",
    "02_elbow_crop_clahe.png",
    "03_olecranon_focus_native.png",
    "04_olecranon_focus_clahe.png",
    "05_elbow_softtissue_window.png",
    "06_elbow_edge_enhanced.png",
    "07_alignment_overlay.png",
    "08_softtissue_profile.png",
    "09_wrist_crop_native.png",
    "10_wrist_crop_clahe.png",
    "11_forearm_shaft_window.png",
]

results = []
fail = 0
for name in EXPECTED:
    p = os.path.join(IMAGES, name)
    rec = {"file": name, "exists": os.path.exists(p), "opens": False, "mode": None, "size": None, "format": None}
    if rec["exists"]:
        try:
            im = Image.open(p)
            im.load()
            rec["opens"] = True
            rec["mode"] = im.mode
            rec["size"] = list(im.size)
            rec["format"] = im.format
        except Exception as e:
            rec["error"] = str(e)
            fail += 1
    else:
        fail += 1
    results.append(rec)

report = f"{IMAGES}/../planning/output_validation.txt"
with open(report, "w") as f:
    f.write("OUTPUT VALIDATION\n=================\n")
    for r in results:
        status = "OK" if (r["exists"] and r["opens"]) else "FAIL"
        f.write(f"{status}  {r['file']}  exists={r['exists']} opens={r['opens']} "
                f"mode={r['mode']} size={r['size']} format={r.get('format')}\n")
        if "error" in r:
            f.write(f"      ERROR: {r['error']}\n")
    f.write(f"\nTOTAL: {len(EXPECTED)} expected, {len(EXPECTED)-fail} valid, {fail} failed\n")

print(open(report).read())
sys.exit(1 if fail else 0)
