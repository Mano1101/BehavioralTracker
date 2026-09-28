"""
Regression test for two fixes made after MM sent a ~3-minute screen
recording of a real EPM run (Standard Tracking, live tracking view) that
showed two problems:

1. The live-tracking overlay's "ROI: ... | Entry: ..." status line ran
   straight off the right edge of a narrower video (an EPM/Y-maze/T-maze
   clip is often cropped down close to the apparatus itself) and got
   silently clipped, unreadable, as soon as more than one zone showed up
   in the Entry: list -- routine during a transition between arms. Fixed
   by wrap_status_line() in tracking/location.py, which now wraps the
   status text onto as many lines as it needs to stay fully on-screen.

2. For a long stretch of that same recording (40+ seconds), the tracked
   point silently jumped off the actual mouse and locked onto a small,
   permanently-static high-contrast fixture (a paper label taped to the
   maze wall) instead -- "ROI: None" for the whole stretch, since that
   spot isn't inside either zone. detect_mouse()'s own motion-based
   "reject static debris" filter (see its comment) used to only apply
   when there were 2+ raw candidate blobs that frame, specifically so a
   genuinely still/freezing animal (a single motionless blob) never got
   mistaken for debris and dropped. But that safety net had a blind spot:
   the ONE frame where the real mouse's own blob happened to fail
   detection (occlusion, low contrast against a dark corner, whatever),
   leaving the static fixture as the SOLE candidate, that fixture sailed
   through with no motion check applied at all, got accepted as a fresh
   detection, and then self-reinforced every frame after (sitting still,
   its own distance-to-previous is 0 forever, which keeps beating the
   real, moving, non-zero-distance mouse on score even once the mouse
   wanders back within max_jump of it). Fixed by extending the motion
   check to singleton candidates too, but only rejecting one that shows
   both NO recent motion AND appeared somewhere NEW (far from wherever
   tracking already was) -- a real animal disturbs the scene as it moves
   into a spot, so "motionless AND far away" is a strong static-fixture
   signal, while "motionless AND close by" is still very likely the same
   animal genuinely holding still, which must keep tracking uninterrupted.

This is pure tracking/location.py logic (no Qt widgets involved), so
unlike the rest of this suite it needs no virtual display -- but it lives
here anyway, run the same way as everything else, so run_all.py picks it
up automatically.

Run directly:
    python3.12 qt_app/tests/test_tracking_robustness.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import sys

import cv2
import numpy as np

from tracking.location import detect_mouse, wrap_status_line
from tracking.two_mouse import MotionHistory

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------
# 1) wrap_status_line: the overlay text overflow fix.
# ------------------------------------------------------------------
font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2


def line_width(s):
    return cv2.getTextSize(s, font, scale, thick)[0][0]


short = wrap_status_line(["ROI: Light"], font, scale, thick, max_width=472)
check("a short single segment is left on one line, unchanged", short == ["ROI: Light"])

# The exact shape of status text MM's video showed: a long zone name plus
# a multi-zone Entry: list, on a narrow (512px-wide) EPM clip.
overflowing = ["ROI: Closed Arm_Open Arm", "Entry: Closed Arm:semi, Open Arm:half"]
wrapped = wrap_status_line(overflowing, font, scale, thick, max_width=472)
check("the previously-overflowing status wraps onto more than one line",
      len(wrapped) > 1)
check("every wrapped line fits within the frame's width",
      all(line_width(l) <= 472 for l in wrapped))
joined = " ".join(wrapped)
check("no words were dropped while wrapping",
      all(w in joined for w in
          ["ROI:", "Closed", "Arm_Open", "Arm", "Entry:", "semi,", "Open", "Arm:half"]))

# A single segment that's STILL too wide even alone on its own line (many
# zones in one Entry: list on a very narrow frame) must keep wrapping
# rather than exceeding max_width.
many_zones = ["ROI: None",
              "Entry: Closed Arm:semi, Open Arm:half, Center:full, Extra Zone:semi"]
wrapped2 = wrap_status_line(many_zones, font, scale, thick, max_width=300)
check("a single over-wide segment is itself split across multiple lines",
      len(wrapped2) >= 3)
check("every line of the heavily-wrapped case fits within max_width",
      all(line_width(l) <= 300 for l in wrapped2))

# ------------------------------------------------------------------
# 2) detect_mouse: the static-fixture lock-on fix. A synthetic arena with
# a flat background, a small permanently-present high-contrast "fixture"
# blob (standing in for MM's taped-on paper label), and a separate,
# larger, moving "mouse" blob.
# ------------------------------------------------------------------
H, W = 200, 200
arena = (0, 0, W, H)
bg = np.full((H, W), 120, dtype=np.uint8)
fixture_xy = (90, 100)  # within max_jump of the mouse's parked spot below, but not ON it
fixture_r = 6
fps = 30
max_jump = 100


def make_frame(mouse_xy, fixture_at=fixture_xy):
    frame = bg.copy()
    cv2.circle(frame, fixture_at, fixture_r, 40, -1)
    if mouse_xy is not None:
        cv2.circle(frame, (int(mouse_xy[0]), int(mouse_xy[1])), 10, 20, -1)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def run_detect(frame, motion_history, previous_point, previous_area):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    motion_history.update(gray)
    return detect_mouse(
        frame, bg, arena, previous_point, previous_area,
        threshold=25, min_area=20, max_area=2000, max_jump=max_jump,
        motion_mask=motion_history.mask, debris_motion_fraction=0.02,
    )


# 2a) Warm up: the mouse sits parked away from the fixture for long enough
# (well past MotionHistory's window) that the fixture builds up a solid
# "never moves" record -- the same as a real fixed sticker would across a
# multi-second real clip.
motion_history = MotionHistory(max(1, int(fps * 1.2)), pixel_threshold=12)
previous_point, previous_area = None, None
for _ in range(50):
    candidate, _ = run_detect(make_frame((30, 100)), motion_history, previous_point, previous_area)
    if candidate is not None:
        previous_point, previous_area = candidate["center"], candidate["area"]

check("warm-up tracked the mouse at its parked position",
      previous_point is not None and abs(previous_point[0] - 30) < 5)

# 2b) One frame where the real mouse's own blob vanishes entirely
# (simulating a momentary detection failure) -- the static fixture is the
# ONLY candidate blob left. It must be rejected, not silently adopted.
frame_no_mouse = make_frame(mouse_xy=None)
candidate_new, _ = run_detect(frame_no_mouse, motion_history, previous_point, previous_area)
check("a lone static fixture that appears far from the last tracked spot is rejected",
      candidate_new is None)

# Contrast case, proving this scenario really is what used to slip
# through: with motion gating unavailable (motion_mask=None), the SAME
# lone fixture IS accepted -- confirming the fixture would otherwise have
# won detection outright, exactly the old single-candidate blind spot.
candidate_ungated, _ = detect_mouse(
    frame_no_mouse, bg, arena, previous_point, previous_area,
    threshold=25, min_area=20, max_area=2000, max_jump=max_jump, motion_mask=None,
)
check("without motion gating, that same lone fixture WOULD have been accepted "
      "(confirms the fix actually targets this scenario)",
      candidate_ungated is not None)
if candidate_ungated is not None:
    d = ((candidate_ungated["center"][0] - fixture_xy[0]) ** 2
         + (candidate_ungated["center"][1] - fixture_xy[1]) ** 2) ** 0.5
    check("...and that accepted candidate really is the fixture, not something else", d < 5)

# 2c) Protection preserved: a genuinely still/freezing animal -- a single
# motionless blob CLOSE to where tracking already was -- must keep being
# tracked exactly as before. This is the exact case the original
# len(candidates) > 1 guard existed to protect.
motion_history2 = MotionHistory(max(1, int(fps * 1.2)), pixel_threshold=12)
frozen_xy = (32, 101)  # basically right where the mouse already was
pp, pa = None, None
for _ in range(50):
    candidate, _ = run_detect(make_frame(mouse_xy=None, fixture_at=frozen_xy), motion_history2, pp, pa)
    if candidate is not None:
        pp, pa = candidate["center"], candidate["area"]

check("a genuinely still (frozen/grooming) animal keeps being tracked, unaffected by the fix",
      pp is not None and abs(pp[0] - frozen_xy[0]) < 5)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
