"""
Regression test for the Setup-page redesign MM asked for (see the "Ella
workum randu perum best haa panni thanga" message):

1. The master-detail layout ("opration tile athan click panna 2nd column
   la athanoda options vara mathriri") -- every setting that used to sit
   in one long stacked column now lives on its own operation tile,
   reachable by clicking it. Checked for all three analysis types, since
   MM asked for this to apply "Ella behaviour application num" (every
   analysis type): the tile list + detail QStackedWidget exist, list the
   expected operations, and clicking a tile actually swaps the detail
   pane and restyles the tiles. The video queue (displaced from its old
   spot as the sole LEFT column) still works, just relocated to the
   RIGHT side.

2. The Qt wiring for "auto-mask everything outside my zones": the new
   Setup-page checkbox (default ON) -> MainWindow._build_setup's
   setup["auto_mask_outside_zones"], and MainWindow._prepare_source_
   video's own zone-complement painting -- the path Multi-Mouse Tracking/
   Behavior Classification use instead of an exclusion_mask array, since
   two_mouse.track_video()/behavior.extract_features() only ever see a
   plain video file. (build_exclusion_mask itself and the Standard-
   Tracking exclusion_mask pipeline are covered separately in test_auto_
   mask_outside_zones.py, which is pure Python/no Qt -- mixing a direct
   process_single_video() call with a PySide6 QApplication in the same
   process trips up this build's Qt5-backed cv2 highgui backend.)

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_master_detail_setup.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring/sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import tempfile

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton, QStackedWidget, QLabel

from tracking.location import identity_transform
from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def find_button(widget, text):
    for btn in widget.findChildren(QPushButton):
        if btn.text() == text:
            return btn
    return None


# ------------------------------------------------------------------
# Setup: one small synthetic video, queued into a real MainWindow.
# ------------------------------------------------------------------
VW, VH, FPS = 240, 160, 30
DURATION_S = 2.0
VIDEO_PATH = os.path.join(tempfile.gettempdir(), "master_detail_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW, VH))
for _ in range(int(DURATION_S * FPS)):
    frame = np.full((VH, VW, 3), 200, dtype=np.uint8)
    cv2.circle(frame, (40, 40), 8, (40, 40, 40), -1)
    writer.write(frame)
writer.release()

app = QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()

win.videos.append({"path": VIDEO_PATH, "fps": FPS, "duration": DURATION_S, "width": VW, "height": VH})
win.active_index = 0
win.setup_page.refresh_video_list()
app.processEvents()

# ------------------------------------------------------------------
# 1) Auto-mask checkbox: default state + wiring into _build_setup +
# _prepare_source_video (Multi-Mouse/Behavior Classification's own path).
# ------------------------------------------------------------------
check("auto_mask_outside_zones_var exists on the Setup page (standard mode)",
      hasattr(win.setup_page, "auto_mask_outside_zones_var"))
check("...and defaults to CHECKED (MM asked for this to be on by default)",
      win.setup_page.auto_mask_outside_zones_var.isChecked())

win.pending_roi_points = {"A": [(0, 0), (50, 0), (50, 50), (0, 50)]}
setup = win._build_setup(VIDEO_PATH)
check("_build_setup carries auto_mask_outside_zones through from the checkbox (checked)",
      setup["auto_mask_outside_zones"] is True)

win.setup_page.auto_mask_outside_zones_var.setChecked(False)
setup2 = win._build_setup(VIDEO_PATH)
check("...and reflects it being unchecked too", setup2["auto_mask_outside_zones"] is False)

win.setup_page.auto_mask_outside_zones_var.setChecked(True)
win.pending_use_crop = False
win.pending_matrix, win.pending_warp_w, win.pending_warp_h = identity_transform(VW, VH)
win.pending_roi_points = {"A": [(0, 0), (120, 0), (120, VH), (0, VH)]}  # left half
win.pending_mask_points = []
win.setup_page.start_entry.setText("")
win.setup_page.end_entry.setText("")

src_path, tmp_path, fps_out = win._prepare_source_video(VIDEO_PATH)
check("zones drawn + auto-mask on -> _prepare_source_video does NOT take the no-op fast path",
      tmp_path is not None)

cap = cv2.VideoCapture(src_path)
ok, out_frame = cap.read()
cap.release()
check("could read a frame back from the pre-warped/painted temp video", ok)
if ok:
    inside_px = tuple(int(c) for c in out_frame[VH // 2, 30])     # left half -- inside zone A
    outside_px = tuple(int(c) for c in out_frame[VH // 2, 200])   # right half -- outside zone A
    # The temp video is written with a lossy MJPG codec (same as the
    # existing Mask Zone painting this reuses), so the round-tripped pixel
    # isn't exactly (128,128,128) -- allow a small tolerance instead of an
    # exact match.
    check("a pixel OUTSIDE the zone was painted close to the flat gray mask color (~128,128,128)",
          all(abs(c - 128) <= 12 for c in outside_px))
    check("a pixel INSIDE the zone was left alone (still close to its original ~200 gray, not 128)",
          all(c > 160 for c in inside_px))
if tmp_path:
    try:
        os.remove(tmp_path)
    except OSError:
        pass

# No-op guard: no zones drawn at all -> even with the checkbox on, the fast
# path still applies (never blanks a frame when there's nothing to mask).
win.pending_roi_points = {}
win.pending_mask_points = []
win.pending_use_crop = False
src_path2, tmp_path2, _fps2 = win._prepare_source_video(VIDEO_PATH)
check("no zones drawn yet -> auto-mask is a no-op, still takes the fast (no pre-warp) path",
      tmp_path2 is None and src_path2 == VIDEO_PATH)

print()

# ------------------------------------------------------------------
# 2) Master-detail layout: tiles + detail pane, all 3 analysis types.
# ------------------------------------------------------------------
EXPECTED_STANDARD_TILES = [
    "Time Window &\nZone Names", "Analysis", "Stop Condition", "Zone\nAssociations",
    "Zone Formula", "Trajectory\nSmoothing", "Custom\nVariables", "Animals in\nFrame",
    "Detection\nSettings", "Time Bins",
]
EXPECTED_MULTI_MOUSE_TILES = [
    "Time Window &\nZone Names", "Analysis", "Animals in\nFrame",
    "Detection\nSettings", "Time Bins",
]
EXPECTED_BEHAVIOR_TILES = [
    "Time Window", "Behaviors to\nDetect", "Animals in\nFrame", "Detection\nSettings",
    "Time Bins", "Behavior\nThresholds", "ML Classifier",
]


def check_master_detail(mode_label, expected_titles):
    stacks = list(win.setup_page.findChildren(QStackedWidget))
    check(f"[{mode_label}] exactly one master-detail QStackedWidget present", len(stacks) == 1)
    if not stacks:
        return
    stack = stacks[0]
    check(f"[{mode_label}] detail pane has one page per expected operation tile "
          f"({len(expected_titles)} expected, got {stack.count()})",
          stack.count() == len(expected_titles))

    buttons = [find_button(win.setup_page, t) for t in expected_titles]
    check(f"[{mode_label}] every expected tile button was found by its label",
          all(b is not None for b in buttons))
    if not all(b is not None for b in buttons):
        return

    check(f"[{mode_label}] tile 0 starts active", buttons[0].objectName() == "analysisCardActive")
    check(f"[{mode_label}] detail pane starts on page 0", stack.currentIndex() == 0)

    target = len(expected_titles) - 1  # click the LAST tile
    buttons[target].click()
    app.processEvents()
    check(f"[{mode_label}] clicking tile {target} ('{expected_titles[target].strip()}') "
          f"switches the detail pane to its page",
          stack.currentIndex() == target)
    check(f"[{mode_label}] the clicked tile is now styled active",
          buttons[target].objectName() == "analysisCardActive")
    check(f"[{mode_label}] tile 0 is no longer styled active",
          buttons[0].objectName() == "toolBtn")

    # Detection Settings' own auto-mask checkbox lives on its tile's page;
    # findChildren still finds it wherever the current stack index is, but
    # navigate there anyway so isHidden() reflects OUR explicit .hide()
    # call rather than "this tile just isn't the visible one right now".
    det_idx = expected_titles.index("Detection\nSettings")
    buttons[det_idx].click()
    app.processEvents()
    amz = getattr(win.setup_page, "auto_mask_outside_zones_var", None)
    if mode_label == "Behavior Classification":
        check(f"[{mode_label}] auto-mask checkbox is hidden (no zones exist in this mode)",
              amz is not None and amz.isHidden())
    else:
        check(f"[{mode_label}] auto-mask checkbox is shown on its Detection Settings tile",
              amz is not None and not amz.isHidden())


win.analysis_type = "standard"
win.setup_page.rebuild(resave=False)
app.processEvents()
check_master_detail("Standard Tracking", EXPECTED_STANDARD_TILES)

win.analysis_type = "multi_mouse"
win.setup_page.rebuild(resave=False)
app.processEvents()
check_master_detail("Multi-Mouse Tracking", EXPECTED_MULTI_MOUSE_TILES)

win.analysis_type = "behavior"
win.setup_page.rebuild(resave=False)
app.processEvents()
check_master_detail("Behavior Classification", EXPECTED_BEHAVIOR_TILES)

# The video queue moved to the RIGHT side (out of the master-detail panel
# on the left) -- still present and working, just relocated.
check("the queued video's name still appears somewhere on the Setup page",
      any(os.path.basename(VIDEO_PATH) in lbl.text() for lbl in win.setup_page.findChildren(QLabel)))

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
