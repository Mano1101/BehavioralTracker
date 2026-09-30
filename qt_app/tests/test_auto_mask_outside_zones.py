"""
Regression test for "Auto-mask everything outside my zones" -- the first
half of MM's "Ella workum randu perum best haa panni thanga" request:
once zones are drawn/finalized ("locked"), everything OUTSIDE them should
be excluded from detection automatically, same as Crop Arena already does
inherently for anything outside the cropped area (warpPerspective only
ever keeps the cropped interior -- nothing new needed for that half; see
qt_app/main_window.py's _prepare_source_video docstring).

This is pure tracking/location.py logic plus a synthetic-video run through
process_single_video (same technique as test_zone_associations.py), so
unlike the Qt-driven suite it needs no virtual display -- but it lives
here anyway, run the same way as everything else, so run_all.py picks it
up automatically. The Qt wiring (Setup page checkbox, MainWindow.
_build_setup, MainWindow._prepare_source_video's own zone-complement
painting for Multi-Mouse/Behavior Classification, and the master-detail
Setup page redesign) is covered separately in test_master_detail_setup.py,
since mixing a direct process_single_video() call with a later PySide6
QApplication in the same process trips up this build's Qt5-backed cv2
highgui backend (see location.py's own note on cv2.imshow near show_
display).

Run directly:
    python3.12 qt_app/tests/test_auto_mask_outside_zones.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import os
import sys
import tempfile

import cv2
import numpy as np

from tracking.location import build_exclusion_mask, process_single_video, identity_transform

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------
# 1) build_exclusion_mask -- pure numpy, no Qt/video needed.
# ------------------------------------------------------------------
H, W = 20, 30
zone = [(0, 0), (14, 0), (14, 20), (0, 20)]  # left half, x in [0, 14)

m = build_exclusion_mask((H, W), [], zone_polygons=None, mask_outside_zones=False)
check("no mask_polygons, no zones, flag off -> nothing excluded", not m.any())

m = build_exclusion_mask((H, W), [], zone_polygons=[], mask_outside_zones=True)
check("flag on but NO zones drawn yet -> still a no-op (never blanks the whole frame)",
      not m.any())

m = build_exclusion_mask((H, W), [], zone_polygons=[zone], mask_outside_zones=False)
check("zones given but flag off -> still a no-op (opt-in only)", not m.any())

m = build_exclusion_mask((H, W), [], zone_polygons=[zone], mask_outside_zones=True)
check("inside the zone is NOT excluded", not m[10, 5])
check("outside the zone (right half) IS excluded", bool(m[10, 20]))
# cv2.fillPoly rasterizes its own right/bottom edge column inclusive (x=14
# counts as "inside" too), so the zone occupies columns 0..14 (15 of 30) --
# not a bug, just how the boundary pixel lands; assert against that same
# rasterization rather than a naive x<14 geometric count.
check("excluded area matches the actual rasterized zone boundary "
      "(fillPoly's own edge column counts as inside)",
      m.sum() == H * (W - 15))

hand_drawn_hole = [(2, 2), (6, 2), (6, 6), (2, 6)]  # a small Mask Zone shape, INSIDE the zone
m = build_exclusion_mask((H, W), [hand_drawn_hole], zone_polygons=[zone], mask_outside_zones=True)
check("a hand-drawn Mask Zone shape inside the zone is excluded too (combined, not overridden)",
      bool(m[4, 4]))
check("the rest of the zone (outside that hand-drawn hole) is still NOT excluded",
      not m[10, 5])
check("outside the zone is still excluded alongside the hand-drawn hole",
      bool(m[10, 20]))

# A SECOND zone -- the union of both is "inside", only outside BOTH is masked.
zone_b = [(20, 0), (30, 0), (30, 20), (20, 20)]  # right strip, x in [20, 30)
m = build_exclusion_mask((H, W), [], zone_polygons=[zone, zone_b], mask_outside_zones=True)
check("with 2 zones, inside the FIRST is not excluded", not m[10, 5])
check("with 2 zones, inside the SECOND is not excluded either", not m[10, 25])
check("with 2 zones, the gap BETWEEN them (x=14..20) is still excluded",
      bool(m[10, 17]))

print()

# ------------------------------------------------------------------
# 2) process_single_video end to end: a blob that visits BOTH inside and
# outside a single drawn zone, comparing tracking quality with the new
# setting on vs off. Same synthetic-video technique as
# test_zone_associations.py (a different spot per hold, see its own
# comment, so a revisited pixel never gets baked into the background).
# ------------------------------------------------------------------
VW, VH, FPS = 240, 160, 30
BG_VAL, BLOB_VAL, BLOB_R = 200, 40, 8

# Zone "A" covers only the LEFT half (x < 120). Segments alternate
# inside/outside it.
SEGMENTS = [
    (30, 40, 15),    # inside A
    (190, 60, 15),   # OUTSIDE A
    (50, 90, 15),    # inside A
    (200, 110, 15),  # OUTSIDE A
    (70, 130, 15),   # inside A
]
POSITIONS = []
for x, y, hold in SEGMENTS:
    POSITIONS.extend([(x, y)] * hold)
TOTAL_FRAMES = len(POSITIONS)
DURATION_S = TOTAL_FRAMES / FPS

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "auto_mask_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW, VH))
for (x, y) in POSITIONS:
    frame = np.full((VH, VW), BG_VAL, dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), BLOB_R, BLOB_VAL, -1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

matrix, warp_w, warp_h = identity_transform(VW, VH)
ROI_POINTS = {"A": [(0, 0), (120, 0), (120, VH), (0, VH)]}

base_setup = {
    "start_time": 0.0, "end_time": DURATION_S,
    "matrix": matrix, "warp_w": warp_w, "warp_h": warp_h,
    "arena": (0, 0, VW, VH),
    "roi_points": ROI_POINTS, "roi_names": ["A"],
    "object_points": {}, "object_names": [],
    "interaction_margin_px": 0, "behavior_names": [],
    "background_samples": 40, "threshold": 30,
    "min_area": 20, "max_area": 2000, "max_jump": 300,
    "use_zone_threshold": False, "reject_shadows": False,
    "use_window": False, "window_size": 120, "window_weight": 0.5,
    "scale_factor": None, "scale_unit": None,
    "preview_samples": 3, "color_mode": "gray",
    "compute_arm_entries": False, "compute_alternation": False,
    "mask_points": [],
    "stop_condition": None, "stop_value": None, "stop_zone_name": "",
    "zone_groups": {},
}

out_dir_off = tempfile.mkdtemp(prefix="automask_off_")
setup_off = dict(base_setup, output_dir_override=out_dir_off, auto_mask_outside_zones=False)
summary_off = process_single_video(
    VIDEO_PATH, setup_off, show_display=False, confirm_callback=lambda *a, **k: False
)

out_dir_on = tempfile.mkdtemp(prefix="automask_on_")
setup_on = dict(base_setup, output_dir_override=out_dir_on, auto_mask_outside_zones=True)
summary_on = process_single_video(
    VIDEO_PATH, setup_on, show_display=False, confirm_callback=lambda *a, **k: False
)

q_off = summary_off["Tracking_quality_percent"]
q_on = summary_on["Tracking_quality_percent"]
check(f"auto_mask_outside_zones=False tracks the whole run normally (quality={q_off:.1f}%)",
      q_off >= 90)
check(f"auto_mask_outside_zones=True loses the 'outside the zone' segments (quality={q_on:.1f}%)",
      q_on <= 80)
check("the setting measurably hurts quality only because it's excluding real frames, "
      "by roughly the 'outside A' fraction of the run (2 of 5 segments, each 15 frames "
      "of 75 total = 40%)",
      abs((q_off - q_on) - 40) <= 15)

# A setup dict that never even mentions the key (old callers, e.g. an
# older saved project) must behave exactly like False -- never opt a
# pre-existing project into a new behavior it never asked for.
out_dir_missing = tempfile.mkdtemp(prefix="automask_missing_")
setup_missing = dict(base_setup, output_dir_override=out_dir_missing)
assert "auto_mask_outside_zones" not in setup_missing
summary_missing = process_single_video(
    VIDEO_PATH, setup_missing, show_display=False, confirm_callback=lambda *a, **k: False
)
check("a setup dict with the key entirely absent behaves like False (backward compatible)",
      summary_missing["Tracking_quality_percent"] >= 90)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
