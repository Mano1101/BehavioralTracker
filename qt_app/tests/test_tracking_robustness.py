"""
Regression test for the live-tracking overlay's status-line wrapping fix,
made after MM sent a ~3-minute screen recording of a real EPM run
(Standard Tracking, live tracking view): the "ROI: ..." status line ran
straight off the right edge of a narrower video (an EPM/Y-maze/T-maze clip
is often cropped down close to the apparatus itself) and got silently
clipped, unreadable, once the ROI label got long enough. Fixed by
wrap_status_line() in tracking/location.py, which now wraps the status
text onto as many lines as it needs to stay fully on-screen.

(This file used to also cover detect_mouse()'s motion-history "reject
static debris" filter -- removed along with prior-position weighting and
shadow rejection so Standard Tracking's detection pipeline matches the
simpler, proven light-dark-box tracker MM benchmarked it against; see
tracking/location.py's detect_mouse docstring.)

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

from tracking.location import wrap_status_line

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

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
