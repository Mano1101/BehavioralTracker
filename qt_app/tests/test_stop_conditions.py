"""
Regression test for the "Stop condition (optional)" feature (Setup page,
Standard Tracking only) -- lets a run end early, before the End (s) time,
once N seconds of immobility / N entries into a named zone / N pixels of
total distance is reached. Inspired by EthoVision's Trial Control rules and
SMART's Status Rules (see the Upgrade Plan doc, Tier 1 #1), but implemented
independently as three simple, hardware-agnostic checks in
tracking.location.process_single_video's own per-frame loop.

This is pure tracking/location.py logic (no Qt widgets involved), so like
test_tracking_robustness.py it needs no virtual display -- run directly:

    python3.12 qt_app/tests/test_stop_conditions.py

or as part of the full suite (xvfb-run -a python3.12 qt_app/tests/run_all.py).

Uses a small synthetic video (not the shared dummy_behavior_test.mp4) so the
mouse's trajectory -- and therefore each stop condition's exact trigger
point -- is fully known ahead of time:

  * A dark blob "teleports" between an x=40 ("Left" zone) and an x=200
    ("Right" zone) position, held for 15 frames each time, 3 round trips
    (6 holds) -- giving exactly 3 entries into "Right" and a sizeable,
    known lower bound on total distance travelled.
  * It then holds still at a 7th, not-previously-visited spot (still in
    "Right") for 50 more frames -- a clean immobility stretch for the
    immobility stop condition to catch.

Every hold uses a DIFFERENT (x, y) pixel spot (only the "Left"/"Right" side
repeats, never the exact same pixel twice) so no single spot is occupied
for anywhere near half of process_single_video's own background samples --
otherwise make_background's per-pixel median would bake the animal itself
into the background at that spot (the exact "revisited spot" ghosting
tracking_robustness's own detect_mouse fix guards against downstream of
background-building; this test just avoids triggering it upstream, in the
background model itself, by construction).

Total: 140 frames @ 30fps ~= 4.67s, matching start_time=0/end_time -- so a
stop_condition of None (the default) is expected to run the full window.
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import os
import sys
import tempfile

import cv2
import numpy as np

from tracking.location import process_single_video, identity_transform

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------
# Build the synthetic video
# ------------------------------------------------------------------
W, H, FPS = 240, 160, 30
BG_VAL, BLOB_VAL, BLOB_R = 200, 40, 8

# 3 round trips Left(x=40) <-> Right(x=200), 15 frames per hold (each at
# its own distinct y so no exact pixel repeats -- see module docstring),
# then a long still hold at a 7th, fresh spot (still in "Right").
SEGMENTS = [
    (40, 60, 15), (200, 60, 15),    # hold1 (Left), hold2 (Right) -> entry 1
    (40, 100, 15), (200, 100, 15),  # hold3 (Left), hold4 (Right) -> entry 2
    (40, 80, 15), (200, 80, 15),    # hold5 (Left), hold6 (Right) -> entry 3
    (200, 40, 50),                  # hold7 (Right, fresh spot) -> immobility stretch
]
POSITIONS = []
for x, y, hold in SEGMENTS:
    POSITIONS.extend([(x, y)] * hold)
TOTAL_FRAMES = len(POSITIONS)
DURATION_S = TOTAL_FRAMES / FPS

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "stop_condition_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for (x, y) in POSITIONS:
    frame = np.full((H, W), BG_VAL, dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), BLOB_R, BLOB_VAL, -1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

matrix, warp_w, warp_h = identity_transform(W, H)
ROI_POINTS = {
    "Left": [(0, 0), (120, 0), (120, H), (0, H)],
    "Right": [(120, 0), (W, 0), (W, H), (120, H)],
}


def make_setup(**overrides):
    setup = {
        "start_time": 0.0, "end_time": DURATION_S,
        "matrix": matrix, "warp_w": warp_w, "warp_h": warp_h,
        "arena": (0, 0, W, H),
        "roi_points": ROI_POINTS, "roi_names": ["Left", "Right"],
        "object_points": {}, "object_names": [],
        "interaction_margin_px": 0, "behavior_names": [],
        "background_samples": 40, "threshold": 30,
        "min_area": 20, "max_area": 2000, "max_jump": 400,
        "use_zone_threshold": False, "reject_shadows": False,
        "use_window": False, "window_size": 120, "window_weight": 0.5,
        "scale_factor": None, "scale_unit": None,
        "preview_samples": 3, "color_mode": "gray",
        "output_dir_override": None,
        "compute_arm_entries": False, "compute_alternation": False,
        "mask_points": [],
        "stop_condition": None, "stop_value": None, "stop_zone_name": "",
    }
    setup.update(overrides)
    return setup


def run(label, **overrides):
    out_dir = tempfile.mkdtemp(prefix=f"stopcond_{label}_")
    setup = make_setup(output_dir_override=out_dir, **overrides)
    summary = process_single_video(
        VIDEO_PATH, setup, show_display=False, confirm_callback=lambda *a, **k: False
    )
    return summary


# ------------------------------------------------------------------
# 1) No stop condition (default) -- runs the full window, same as before
#    this feature existed.
# ------------------------------------------------------------------
summary_none = run("none")
check("no stop condition -> processes (close to) the full frame window",
      summary_none["Frames_processed"] >= TOTAL_FRAMES - 2)
check("no stop condition -> Stop_Reason reports reaching End (s)",
      summary_none["Stop_Reason"] == "Reached End (s)")

# ------------------------------------------------------------------
# 2) Zone-entries stop condition -- 3 entries into "Right" happen at the
#    start of holds #2, #4, #6 (frames 15, 45, 75); the run must stop
#    at/around the 3rd one, well before the 60-frame still tail.
# ------------------------------------------------------------------
summary_zone = run("zone", stop_condition="zone_entries", stop_value=3, stop_zone_name="Right")
check("zone-entries stop condition ends the run well before the full window",
      summary_zone["Frames_processed"] < TOTAL_FRAMES - 30)
check("zone-entries stop condition ends the run after the 3rd entry could occur",
      summary_zone["Frames_processed"] >= 75)
check("zone-entries stop condition reports why in Stop_Reason",
      "3" in summary_zone["Stop_Reason"] and "Right" in summary_zone["Stop_Reason"])

# ------------------------------------------------------------------
# 3) Distance stop condition -- each Left<->Right hop covers ~160px, so a
#    modest pixel budget is used up within the first couple of hops, long
#    before the still tail.
# ------------------------------------------------------------------
summary_dist = run("distance", stop_condition="distance", stop_value=300)
check("distance stop condition ends the run well before the full window",
      summary_dist["Frames_processed"] < TOTAL_FRAMES - 30)
check("distance stop condition's own Total_distance_pixels reaches the target",
      summary_dist["Total_distance_pixels"] >= 300)
check("distance stop condition reports why in Stop_Reason",
      "distance" in summary_dist["Stop_Reason"])

# ------------------------------------------------------------------
# 4) Immobility stop condition -- the animal settles at x=200 well before
#    the deliberate 60-frame still tail (holds #6 and #7 are both at
#    x=200, back to back), so 1s of immobility is reached mid-video, not
#    only in the designed-for-it tail.
# ------------------------------------------------------------------
summary_immobile = run("immobile", stop_condition="immobility", stop_value=1.0)
check("immobility stop condition ends the run before the full window",
      summary_immobile["Frames_processed"] < TOTAL_FRAMES)
check("immobility stop condition ends only once the animal has actually settled "
      "(after the last Left<->Right hop at frame 75)",
      summary_immobile["Frames_processed"] >= 75)
check("immobility stop condition reports why in Stop_Reason",
      "Immobile" in summary_immobile["Stop_Reason"])

# ------------------------------------------------------------------
# 5) Setup page <-> main_window wiring: the combo-box label -> internal
#    code mapping used by both _build_setup and on_start's validation.
# ------------------------------------------------------------------
from qt_app.main_window import STOP_CONDITION_CODES

check("label map covers the 4 combo box options",
      set(STOP_CONDITION_CODES.keys()) == {
          "None", "N seconds of immobility", "N entries into a zone", "N pixels of total distance"
      })
check("'None' maps to no stop condition", STOP_CONDITION_CODES["None"] is None)
check("labels map to the exact codes process_single_video checks for",
      STOP_CONDITION_CODES["N seconds of immobility"] == "immobility"
      and STOP_CONDITION_CODES["N entries into a zone"] == "zone_entries"
      and STOP_CONDITION_CODES["N pixels of total distance"] == "distance")

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
