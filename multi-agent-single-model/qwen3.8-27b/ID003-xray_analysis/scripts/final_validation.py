"""FINAL VALIDATION (step 15) — verifies the complete pipeline state.

Checks:
 1. final JSON reopens from disk and parses
 2. original filename correct
 3. required JSON fields exist (and no extras)
 4. working directories exist
 5. processing plan exists
 6. python scripts exist; pipeline execution evidenced by outputs + logs
 7. generated images exist
 8. agent reports exist
 9. every generated image was inspected by >=1 agent (a report references it)
Exit 0 only if all checks pass.
"""
import json
import os
import re
import sys

BASE = "/home/ralph/projects/ufsc/pablo-xray-tests/ID003-xray_analysis"
IMG_DIR = os.path.join(BASE, "images")
REP_DIR = os.path.join(BASE, "agent_reports")
PLAN_DIR = os.path.join(BASE, "planning")
SCR_DIR = os.path.join(BASE, "scripts")

failures = []
checks = []

def check(name, ok, detail=""):
    checks.append(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(name)

# 1-3. final JSON
json_path = os.path.join(BASE, "ID003-xray.json")
check("final JSON exists on disk", os.path.isfile(json_path))
data = None
try:
    with open(json_path) as f:
        data = json.load(f)
    check("final JSON parses", True)
except Exception as e:
    check("final JSON parses", False, str(e))
if data is not None:
    check("original filename correct", data.get("image") == "ID003-xray.png",
          f"got {data.get('image')!r}")
    for field in ("image", "findings", "impression", "limitations"):
        v = data.get(field)
        check(f"field '{field}' present and non-empty", isinstance(v, str) and len(v.strip()) > 0)
    extra = set(data.keys()) - {"image", "findings", "impression", "limitations"}
    check("no extra JSON fields", not extra, f"extra: {sorted(extra)}")

# 4. directories
for d in ("planning", "images", "agent_reports", "scripts"):
    check(f"directory {d}/ exists", os.path.isdir(os.path.join(BASE, d)))

# 5. plan
plan = os.path.join(PLAN_DIR, "processing_plan.txt")
check("processing plan exists", os.path.isfile(plan))
check("processing plan has Round 2 section",
      os.path.isfile(plan) and "ROUND 2" in open(plan).read())

# 6. scripts + execution evidence
scripts = sorted(f for f in os.listdir(SCR_DIR) if f.endswith(".py"))
check("python scripts exist", len(scripts) >= 5, ", ".join(scripts))
meas = os.path.join(PLAN_DIR, "measurements_report.txt")
valr = os.path.join(PLAN_DIR, "validation_report.txt")
check("measurements report (script execution evidence) exists", os.path.isfile(meas))
check("validation report (script execution evidence) exists", os.path.isfile(valr))

# 7. images exist + open
from PIL import Image
imgs = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
check("generated images exist (>=30)", len(imgs) >= 30, f"{len(imgs)} images")
bad = []
for f in imgs:
    try:
        im = Image.open(os.path.join(IMG_DIR, f))
        im.verify()
    except Exception as e:
        bad.append(f"{f}: {e}")
check("all images open/valid", not bad, "; ".join(bad[:3]))
orig_copy = os.path.join(IMG_DIR, "ID003-xray.png")
check("unmodified original copy present in images/", os.path.isfile(orig_copy))

# 8. reports exist
reports = sorted(f for f in os.listdir(REP_DIR)
                 if f.endswith(".txt") and not f.startswith("_quarantine"))
check("agent reports exist (>=50)", len(reports) >= 50, f"{len(reports)} reports")
for a in ("Agent_01", "Agent_02", "Agent_03", "Agent_04", "Agent_05", "Agent_06"):
    phase_ok = all(os.path.isfile(os.path.join(REP_DIR, f"{a}-{p}"))
                   for p in ("01-firstpass.txt", "02-proposals.txt", "03-deliberation.txt"))
    n_img = sum(1 for f in reports if f.startswith(a + "-") and "ID003" in f)
    check(f"{a} council files (firstpass+proposals+deliberation) + image reports",
          phase_ok and n_img >= 2, f"{n_img} image reports, phase files ok={phase_ok}")

# report format spot-check: every report has the required headers
missing_hdr = []
for f in reports:
    s = open(os.path.join(REP_DIR, f)).read()
    for h in ("AGENT:", "MODEL:", "IMAGE:"):
        if h not in s:
            missing_hdr.append(f"{f} (missing {h})")
    if "qwen3.8-27b" not in s:
        missing_hdr.append(f"{f} (wrong model)")
check("all reports contain required headers + MODEL qwen3.8-27b",
      not missing_hdr, "; ".join(missing_hdr[:5]))

# 9. every generated image inspected by >=1 agent
allrep = ""
for f in reports:
    allrep += open(os.path.join(REP_DIR, f)).read()
uninspected = [f for f in imgs if f not in allrep]
check("every image referenced by >=1 agent report", not uninspected,
      f"uninspected: {uninspected}")

# important images got >=2 independent reviewers
# (the D2 disputed region overall is covered by 90/91/92 with multiple reviews)
important = ["ID003_10_crop_midforearm_plain.png", "ID003_20_midforearm_clahe.png",
             "ID003_62_axis_angulation.png", "ID003_30_wrist_bonewindow.png",
             "ID003_90_diagonalbones_plain.png"]
thin = []
for f in important:
    n = sum(1 for r in reports
            if f"IMAGE: {f}" in open(os.path.join(REP_DIR, r)).read())
    if n < 2:
        thin.append(f"{f} ({n})")
check("important images reviewed by >=2 independent agents", not thin, "; ".join(thin))

print("\n".join(checks))
print(f"\n{len(checks) - len(failures)}/{len(checks)} checks passed")
if failures:
    print("FAILURES:", failures)
sys.exit(1 if failures else 0)
