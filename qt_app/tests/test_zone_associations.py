"""
Regression test for "Zone Associations" (Setup page, Standard Tracking
only) -- group 2+ existing zones into one named combined zone for
reporting (time/percent/entries), without redrawing anything. Mirrors
SMART's zone grouping (see the Upgrade Plan doc, Tier 1 #3).

Two independent things are covered:

1. parse_zone_associations() (qt_app/main_window.py) -- the pure text ->
   {group_name: [members]} parser behind the Setup page's "Group = Zone A +
   Zone B; Group2 = ..." box. No Qt/video needed for this part.

2. tracking.location.process_single_video's own zone_groups handling -- a
   synthetic video visits 3 distinct zones (A/B/C), with a group "AB" =
   [A, B], and the resulting summary's AB_time_s/AB_percent/AB_entries are
   checked against the known trajectory.

Run directly (no virtual display needed for part 1 alone, but part 2 uses
OpenCV video I/O like test_stop_conditions.py, so run it the same way):
    python3.12 qt_app/tests/test_zone_associations.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import os
import sys
import tempfile

import cv2
import numpy as np

from qt_app.main_window import parse_zone_associations
from tracking.location import process_single_video, identity_transform

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------
# 1) parse_zone_associations -- pure parsing, valid and invalid input.
# ------------------------------------------------------------------
groups, err = parse_zone_associations("")
check("blank text parses to no groups, no error", groups == {} and err is None)

groups, err = parse_zone_associations("   ")
check("whitespace-only text parses to no groups, no error", groups == {} and err is None)

groups, err = parse_zone_associations("Left Side = Left Arm + Left Corner")
check("a single well-formed group parses correctly",
      err is None and groups == {"Left Side": ["Left Arm", "Left Corner"]})

groups, err = parse_zone_associations(
    "Left Side = Left Arm + Left Corner; Right Side = Right Arm + Right Corner"
)
check("two ';'-separated groups both parse correctly",
      err is None and groups == {
          "Left Side": ["Left Arm", "Left Corner"],
          "Right Side": ["Right Arm", "Right Corner"],
      })

groups, err = parse_zone_associations("  Padded  =  A  +  B  ")
check("surrounding whitespace on the group name and each member is stripped",
      err is None and groups == {"Padded": ["A", "B"]})

groups, err = parse_zone_associations("Left Side + Left Arm")
check("a group missing '=' is rejected with an error, not silently dropped",
      err is not None and groups == {})

groups, err = parse_zone_associations("= Left Arm + Left Corner")
check("a group missing its name is rejected", err is not None and groups == {})

groups, err = parse_zone_associations("Left Side = Left Arm")
check("a group with only 1 member (no '+') is rejected -- needs 2+ zones",
      err is not None and groups == {})

groups, err = parse_zone_associations("Left Side = Left Arm + Left Corner;   ")
check("a trailing empty ';' segment is tolerated, not treated as a malformed group",
      err is None and groups == {"Left Side": ["Left Arm", "Left Corner"]})

# ------------------------------------------------------------------
# 2) process_single_video's zone_groups handling -- synthetic video with 3
# distinct zones (A/B/C) and a group AB = [A, B].
# ------------------------------------------------------------------
W, H, FPS = 240, 160, 30
BG_VAL, BLOB_VAL, BLOB_R = 200, 40, 8

# Every hold uses a DIFFERENT (x, y) spot (see test_stop_conditions.py's
# module docstring for why: a revisited exact pixel can get baked into
# make_background's own median background model). Zone C: x 160-239 ("out
# of the AB group"). Zone A: x 0-79. Zone B: x 80-159.
SEGMENTS = [
    (200, 40, 15),   # hold1: C (outside AB)
    (40, 60, 15),    # hold2: A -> entry #1 into AB
    (120, 80, 15),   # hold3: B (still inside AB, no new entry)
    (200, 100, 15),  # hold4: C (leaves AB)
    (40, 120, 15),   # hold5: A -> entry #2 into AB
    (120, 140, 15),  # hold6: B (still inside AB)
]
POSITIONS = []
for x, y, hold in SEGMENTS:
    POSITIONS.extend([(x, y)] * hold)
TOTAL_FRAMES = len(POSITIONS)
DURATION_S = TOTAL_FRAMES / FPS
EXPECTED_AB_SECONDS = sum(hold for x, y, hold in SEGMENTS if x < 160) / FPS  # holds 2,3,5,6

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "zone_assoc_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for (x, y) in POSITIONS:
    frame = np.full((H, W), BG_VAL, dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), BLOB_R, BLOB_VAL, -1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

matrix, warp_w, warp_h = identity_transform(W, H)
ROI_POINTS = {
    "A": [(0, 0), (80, 0), (80, H), (0, H)],
    "B": [(80, 0), (160, 0), (160, H), (80, H)],
    "C": [(160, 0), (W, 0), (W, H), (160, H)],
}

out_dir = tempfile.mkdtemp(prefix="zoneassoc_")
setup = {
    "start_time": 0.0, "end_time": DURATION_S,
    "matrix": matrix, "warp_w": warp_w, "warp_h": warp_h,
    "arena": (0, 0, W, H),
    "roi_points": ROI_POINTS, "roi_names": ["A", "B", "C"],
    "object_points": {}, "object_names": [],
    "interaction_margin_px": 0, "behavior_names": [],
    "background_samples": 40, "threshold": 30,
    "min_area": 20, "max_area": 2000, "max_jump": 400,
    "use_zone_threshold": False, "reject_shadows": False,
    "use_window": False, "window_size": 120, "window_weight": 0.5,
    "scale_factor": None, "scale_unit": None,
    "preview_samples": 3, "color_mode": "gray",
    "output_dir_override": out_dir,
    "compute_arm_entries": False, "compute_alternation": False,
    "mask_points": [],
    "stop_condition": None, "stop_value": None, "stop_zone_name": "",
    "zone_groups": {"AB": ["A", "B"]},
}
summary = process_single_video(
    VIDEO_PATH, setup, show_display=False, confirm_callback=lambda *a, **k: False
)

check("tracking quality is clean (no lost frames spoiling the timing math)",
      summary["Tracking_quality_percent"] >= 95)
check("AB_time_s exists in the summary", "AB_time_s" in summary)
check("AB_time_s matches the known A+B dwell time (within 1 frame's worth of slack)",
      abs(summary.get("AB_time_s", -999) - EXPECTED_AB_SECONDS) <= (2.0 / FPS))
check("AB_percent is between A_percent+B_percent-ish territory and 100 (sanity bound)",
      0 < summary.get("AB_percent", -1) <= 100)
check("AB_entries counts exactly the 2 C->A/B crossings (not every A<->B hop within the group)",
      summary.get("AB_entries") == 2)
check("the group's OWN entries are independent of its real members' own A_entries -- "
      "A/B/C still report their individual stats too (compute_arm_entries was off, "
      "so just confirm their _time_s are still present and unaffected)",
      "A_time_s" in summary and "B_time_s" in summary and "C_time_s" in summary)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
