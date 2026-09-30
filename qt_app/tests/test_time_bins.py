"""
Regression test for "Configurable time-bin reporting for every analysis
type" (Upgrade Plan Tier 2 #9) -- Standard Tracking's "1min Individual"/
"1min Cumulative" report sheets get a configurable bin size (instead of a
fixed 60 seconds), and the same idea is extended to Multi-Mouse Tracking
(distance/frames-tracked per mouse per bin, tracking.two_mouse.
calculate_time_bins) and Behavior Classification (time-in-each-behavior
per mouse per bin, tracking.behavior.calculate_behavior_time_bins) --
both of which had no time-bin reporting at all before this feature.

Four things are covered:

1. tracking.behaviour's shared bin-edge/label helpers (format_bin_label,
   compute_bin_edges, bin_row_label) -- pure functions, no DataFrame
   needed.

2. calculate_time_bins (Multi-Mouse) and calculate_behavior_time_bins
   (Behavior Classification) against small, fully hand-built DataFrames
   with known values -- no video/tracking pipeline needed for this part.

3. tracking.location.process_single_video's own wiring -- a short
   synthetic video, run with the default (unset) bin size vs. an explicit
   30-second one: summary["Bin_size_s"], the resulting Excel workbook's
   sheet names, and the bin row count/labels all reflect the requested
   size.

4. The Setup page field itself (setup_page.py's shared bin_size_entry,
   living in _detection_settings so it reaches all three analysis types)
   driven through a real MainWindow: Multi-Mouse and Behavior Classification
   individual-mode runs both now write a time_bins.csv alongside their
   existing output files.

Run directly (needs a virtual display for part 4 -- see _pathsetup.py's
docstring):
    xvfb-run -a python3.12 qt_app/tests/test_time_bins.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import openpyxl

from tracking.behaviour import format_bin_label, compute_bin_edges, bin_row_label
from tracking.two_mouse import calculate_time_bins
from tracking.behavior import calculate_behavior_time_bins
from tracking.location import process_single_video, identity_transform

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------
# 1) Shared bin-edge/label helpers -- pure functions.
# ------------------------------------------------------------------
check("format_bin_label renders whole minutes as 'Nmin'", format_bin_label(60.0) == "1min")
check("format_bin_label renders 5 minutes as '5min'", format_bin_label(300.0) == "5min")
check("format_bin_label renders a non-minute size in seconds", format_bin_label(30.0) == "30s")
check("format_bin_label renders a fractional-second size sensibly", format_bin_label(2.5) == "2.5s")

edges, unit_divisor, unit_name = compute_bin_edges(0.0, 125.0, 60.0)
check("compute_bin_edges with 60s bins over 125s gives 3 edges (0,60,120,125 -> trimmed)",
      edges == [0.0, 60.0, 120.0, 125.0])
check("compute_bin_edges picks minutes as the unit for whole-minute bins", unit_name == "min")

edges2, unit_divisor2, unit_name2 = compute_bin_edges(0.0, 50.0, 20.0)
check("compute_bin_edges with 20s bins over 50s gives the right trimmed edges",
      edges2 == [0.0, 20.0, 40.0, 50.0])
check("compute_bin_edges picks seconds as the unit for a non-minute bin size", unit_name2 == "s")

check("bin_row_label (non-cumulative) reads like '0-1 min' for the first 60s bin",
      bin_row_label(0.0, 60.0, 0.0, 60.0, "min") == "0-1 min")
check("bin_row_label (cumulative) always starts from 0",
      bin_row_label(60.0, 120.0, 0.0, 60.0, "min", cumulative=True) == "0-2 min")

check("compute_bin_edges falls back to 60.0 for a falsy bin_size_s (e.g. None/0)",
      compute_bin_edges(0.0, 60.0, None)[1:] == compute_bin_edges(0.0, 60.0, 60.0)[1:])

# ------------------------------------------------------------------
# 2) calculate_time_bins (Multi-Mouse) / calculate_behavior_time_bins
# (Behavior Classification) -- small, hand-built DataFrames.
# ------------------------------------------------------------------
tracks_df = pd.DataFrame({
    "frame": [0, 1, 2, 3, 0, 1, 2, 3],
    "time_s": [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0],
    "mouse_id": ["mouse_A"] * 4 + ["mouse_B"] * 4,
    # mouse_A moves 3px each step (0,0)->(3,0)->(6,0)->(9,0): 3px/frame.
    # mouse_B stays put the whole time: 0px moved.
    "x": [0.0, 3.0, 6.0, 9.0, 5.0, 5.0, 5.0, 5.0],
    "y": [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
    "status": ["ok"] * 8,
})
mm_bins = calculate_time_bins(tracks_df, fps=1.0, start_time=0.0, end_time=4.0, bin_size_s=2.0)
check("calculate_time_bins produces one row per (mouse, bin) -- 2 mice x 2 bins",
      len(mm_bins) == 4)
mouse_a_bin0 = mm_bins[(mm_bins["mouse_id"] == "mouse_A") & (mm_bins["Bin"] == "0-2 s")].iloc[0]
check("mouse_A's first bin (frames 0-1) traveled the expected 3px",
      abs(mouse_a_bin0["Distance_pixels"] - 3.0) < 1e-6)
mouse_a_bin1 = mm_bins[(mm_bins["mouse_id"] == "mouse_A") & (mm_bins["Bin"] == "2-4 s")].iloc[0]
# Per-frame step distance is computed once over the WHOLE trajectory (not
# reset at each bin boundary), then summed by whichever bin each frame's
# own timestamp falls into -- so this bin gets both the frame1->frame2 step
# (the boundary crossing, credited to frame2's bin) and the frame2->frame3
# step: 3px + 3px = 6px. This is what keeps bins' distances summing back to
# the true total path length (9px here) instead of silently dropping every
# boundary-crossing step.
check("mouse_A's second bin (frames 2-3, plus the boundary step into it) traveled 6px",
      abs(mouse_a_bin1["Distance_pixels"] - 6.0) < 1e-6)
mouse_b_bin0 = mm_bins[(mm_bins["mouse_id"] == "mouse_B") & (mm_bins["Bin"] == "0-2 s")].iloc[0]
check("mouse_B (stationary) traveled ~0px", abs(mouse_b_bin0["Distance_pixels"]) < 1e-6)
check("calculate_time_bins on an empty/None tracks_df returns an empty DataFrame",
      len(calculate_time_bins(pd.DataFrame(), fps=30.0)) == 0
      and len(calculate_time_bins(None, fps=30.0)) == 0)

labeled_df = pd.DataFrame({
    "time_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    "mouse_id": ["mouse_A"] * 6,
    "label": ["rearing", "rearing", "locomotion", "locomotion", "locomotion", "immobile"],
})
beh_bins = calculate_behavior_time_bins(labeled_df, fps=1.0, start_time=0.0, end_time=6.0, bin_size_s=3.0)
check("calculate_behavior_time_bins produces one row per (mouse, bin) -- 1 mouse x 2 bins",
      len(beh_bins) == 2)
bin0 = beh_bins[beh_bins["Bin"] == "0-3 s"].iloc[0]
check("first 3s bin: 2 frames rearing, 1 frame locomotion (at 1fps -> seconds)",
      abs(bin0["rearing_time_s"] - 2.0) < 1e-6 and abs(bin0["locomotion_time_s"] - 1.0) < 1e-6)
bin1 = beh_bins[beh_bins["Bin"] == "3-6 s"].iloc[0]
check("second 3s bin: 2 frames locomotion, 1 frame immobile",
      abs(bin1["locomotion_time_s"] - 2.0) < 1e-6 and abs(bin1["immobile_time_s"] - 1.0) < 1e-6)
check("a label with 0 frames in a bin still gets a 0 (not a missing) column value",
      bin1.get("rearing_time_s", None) == 0)
check("calculate_behavior_time_bins on empty/None labeled_df returns an empty DataFrame",
      len(calculate_behavior_time_bins(pd.DataFrame(), fps=30.0)) == 0
      and len(calculate_behavior_time_bins(None, fps=30.0)) == 0)

# ------------------------------------------------------------------
# 3) process_single_video wiring (Standard Tracking) -- default (60s,
# unset) vs. an explicit 30s bin size.
# ------------------------------------------------------------------
W, H, FPS = 200, 160, 30
rng = np.random.default_rng(3)
xs = np.linspace(30, 170, 90) + rng.normal(0, 2.0, 90)
ys = 80 + 15 * np.sin(np.linspace(0, 9, 90))
POSITIONS = list(zip(xs, ys))
DURATION_S = len(POSITIONS) / FPS  # 3 seconds

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "time_bins_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for (x, y) in POSITIONS:
    frame = np.full((H, W), 200, dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), 8, 40, -1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

matrix, warp_w, warp_h = identity_transform(W, H)


def make_setup(**overrides):
    setup = {
        "start_time": 0.0, "end_time": DURATION_S,
        "matrix": matrix, "warp_w": warp_w, "warp_h": warp_h,
        "arena": (0, 0, W, H),
        "roi_points": {}, "roi_names": [],
        "object_points": {}, "object_names": [],
        "interaction_margin_px": 0, "behavior_names": [],
        "background_samples": 30, "threshold": 30,
        "min_area": 20, "max_area": 2000, "max_jump": 400,
        "use_zone_threshold": False, "reject_shadows": False,
        "use_window": False, "window_size": 120, "window_weight": 0.5,
        "scale_factor": None, "scale_unit": None,
        "preview_samples": 3, "color_mode": "gray",
        "output_dir_override": None,
        "compute_arm_entries": False, "compute_alternation": False,
        "mask_points": [],
        "stop_condition": None, "stop_value": None, "stop_zone_name": "",
        "zone_groups": {}, "smooth_window": None, "custom_variables": [],
        "bin_size_s": None,
    }
    setup.update(overrides)
    return setup


def run(label, **overrides):
    out_dir = tempfile.mkdtemp(prefix=f"timebins_{label}_")
    setup = make_setup(output_dir_override=out_dir, **overrides)
    summary = process_single_video(
        VIDEO_PATH, setup, show_display=False, confirm_callback=lambda *a, **k: False
    )
    return summary, summary["Output_folder"]


summary_default, out_default = run("default")
check("unset bin size -> summary reports the 60s default", summary_default["Bin_size_s"] == 60.0)
excel_default = os.path.join(out_default, "time_bins_test_video_Analysis.xlsx")
wb_default = openpyxl.load_workbook(excel_default, read_only=True)
check("default bin size -> Excel sheet is named '1min Individual'",
      "1min Individual" in wb_default.sheetnames)
check("default bin size -> Excel sheet is named '1min Cumulative'",
      "1min Cumulative" in wb_default.sheetnames)
wb_default.close()

summary_30s, out_30s = run("30s", bin_size_s=30.0)
check("explicit 30s bin size -> summary reports it", summary_30s["Bin_size_s"] == 30.0)
excel_30s = os.path.join(out_30s, "time_bins_test_video_Analysis.xlsx")
wb_30s = openpyxl.load_workbook(excel_30s, read_only=True)
check("30s bin size -> Excel sheet is named '30s Individual'", "30s Individual" in wb_30s.sheetnames)
check("30s bin size -> Excel sheet is named '30s Cumulative'", "30s Cumulative" in wb_30s.sheetnames)
individual_30s = pd.read_excel(excel_30s, sheet_name="30s Individual")
check("30s bins over a 3s video gives more, smaller-timespan rows than the 1min default",
      len(individual_30s) >= 1 and (individual_30s["End_seconds"] - individual_30s["Start_seconds"]).max() <= 30.0 + 1e-6)
wb_30s.close()

# ------------------------------------------------------------------
# 4) The Setup page field itself, through a real MainWindow -- Multi-Mouse
# and Behavior Classification individual-mode runs both write a new
# time_bins.csv.
# ------------------------------------------------------------------
from PySide6.QtWidgets import QApplication, QMessageBox
from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from tracking.location import compute_output_dir
from _pathsetup import VIDEO

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

_out_dir = compute_output_dir(VIDEO)
import shutil
if os.path.isdir(_out_dir):
    shutil.rmtree(_out_dir)
    os.makedirs(_out_dir, exist_ok=True)

# Part 3 above already ran process_single_video, whose save_plots step
# (output/graphs.py) touches matplotlib's Qt-based backend and can leave a
# QApplication singleton already sitting in this process (see run_all.py's
# own note that only one QApplication can live in a process) -- reuse it
# instead of trying to construct a second one.
app = QApplication.instance() or QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()


def add_video():
    entry = win._probe_video(VIDEO)
    win.videos.append(entry)
    win.active_index = len(win.videos) - 1
    win.setup_page.refresh_video_list()
    app.processEvents()


check("bin_size_entry exists on the Setup page (Standard Tracking, the default analysis type)",
      getattr(win.setup_page, "bin_size_entry", None) is not None)

win._set_analysis_type("multi_mouse")
app.processEvents()
check("bin_size_entry also exists for Multi-Mouse Tracking (shared _detection_settings)",
      getattr(win.setup_page, "bin_size_entry", None) is not None)
win.setup_page.bin_size_entry.setText("2")
add_video()
win.on_tool_no_crop()
win.setup_page.on_num_animals(2)
app.processEvents()

mb_calls.clear()
win.on_start()
app.processEvents()
check("Multi-Mouse individual run with a custom bin size completes with no error dialogs",
      not any(k == "critical" for k, _a in mb_calls))
mm_out_dir = win.results_page.output_dir
mm_time_bins_csv = os.path.join(mm_out_dir, "time_bins.csv")
check("Multi-Mouse individual run wrote a time_bins.csv", os.path.exists(mm_time_bins_csv))
if os.path.exists(mm_time_bins_csv):
    mm_bins_df = pd.read_csv(mm_time_bins_csv)
    check("Multi-Mouse's time_bins.csv has the expected columns",
          {"mouse_id", "Bin", "Distance_pixels", "Tracked_frames"}.issubset(mm_bins_df.columns))

win.on_reset_all()
app.processEvents()
win._set_analysis_type("behavior")
app.processEvents()
check("bin_size_entry also exists for Behavior Classification (shared _detection_settings)",
      getattr(win.setup_page, "bin_size_entry", None) is not None)
win.setup_page.bin_size_entry.setText("2")
add_video()
win.on_tool_no_crop()
win.setup_page.on_num_animals(1)
app.processEvents()

mb_calls.clear()
win.on_start()
app.processEvents()
check("Behavior Classification individual run with a custom bin size completes with no error dialogs",
      not any(k == "critical" for k, _a in mb_calls))
beh_out_dir = win.results_page.output_dir
beh_time_bins_csv = os.path.join(beh_out_dir, "time_bins.csv")
check("Behavior Classification individual run wrote a time_bins.csv", os.path.exists(beh_time_bins_csv))
if os.path.exists(beh_time_bins_csv):
    beh_bins_df = pd.read_csv(beh_time_bins_csv)
    check("Behavior Classification's time_bins.csv has the expected columns",
          {"mouse_id", "Bin", "Tracked_frames"}.issubset(beh_bins_df.columns))
    check("Behavior Classification's time_bins.csv has at least one '<behavior>_time_s' column",
          any(c.endswith("_time_s") for c in beh_bins_df.columns))

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
