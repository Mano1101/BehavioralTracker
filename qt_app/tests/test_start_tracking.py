"""
Headless regression test for Start Tracking -> Results, all 3 analysis
types, run against the REAL (unchanged) tracking/ pipeline -- not mocked.
Covers Task #26 of the PySide6 rewrite (see README.md's "Remaining
roadmap"). Also exercises the show_display=False design decision (see the
comments next to it in qt_app/main_window.py's _run_standard_flow /
_run_multi_mouse_flow): this build's OpenCV is Qt5-based and a live
cv2.imshow window here can hard-crash the process alongside this PySide6
(Qt6) app, so the Qt GUI never asks for one -- this test's cv2.waitKey
monkeypatch is a defensive no-op belt-and-suspenders check, not something
the app path should actually hit anymore.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_start_tracking.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import sys, os

import cv2
from PySide6.QtWidgets import QApplication, QMessageBox, QLabel, QPushButton, QGroupBox
from PySide6.QtCore import Qt

# Headless run: patch QMessageBox statics (shared class object) AND cv2's
# blocking interactive-preview keypress wait, defensively (see module
# docstring -- the Qt app itself should never reach a real cv2.imshow/
# waitKey anymore, but this keeps the test robust either way).
mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)
cv2.waitKey = lambda *a, **k: 13  # Enter -- accepts every blocking prompt immediately

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from _pathsetup import VIDEO

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def widget_texts(widget):
    out = []
    for child in widget.findChildren(QLabel):
        out.append(child.text())
    for child in widget.findChildren(QPushButton):
        out.append(child.text())
    for child in widget.findChildren(QGroupBox):
        out.append(child.title())
    return out


app = QApplication(sys.argv)
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


# ------------------------------------------------------------------
# 1) Standard Tracking, Individual mode: full real run, then check the
# Results page.
# ------------------------------------------------------------------
add_video()
win.on_tool_no_crop()
win.setup_page.roi_names_entry.setText("Left, Right")
win.start_op("zones")
w, h = win.videos[0]["width"], win.videos[0]["height"]
for fx, fy in [(10, 10), (w // 2 - 5, 10), (w // 2 - 5, h - 10), (10, h - 10)]:
    win.on_canvas_press(fx, fy)
win.op_set_active_region("Right")
for fx, fy in [(w // 2 + 5, 10), (w - 10, 10), (w - 10, h - 10), (w // 2 + 5, h - 10)]:
    win.on_canvas_press(fx, fy)
win.finish_op()
check("zones ready before Standard Tracking run", win.pending_roi_points and set(win.pending_roi_points) == {"Left", "Right"})

mb_calls.clear()
win.on_start()
app.processEvents()
check("no error dialogs during Standard Individual run",
      not any(kind in ("critical",) for kind, _a in mb_calls))
check("stack switched to Results page after run", win.stack.currentWidget() is win.results_page)
check("step tracker highlights Results", win.step_labels["results"].objectName() == "stepLabelActive")
out_dir = win.results_page.output_dir
check("output_dir set and exists", bool(out_dir) and os.path.isdir(out_dir))
check("trajectory.png generated", os.path.exists(os.path.join(out_dir, "trajectory.png")))
check("heatmap.png generated", os.path.exists(os.path.join(out_dir, "heatmap.png")))
check("zone_occupancy.png generated (zones were set)", os.path.exists(os.path.join(out_dir, "zone_occupancy.png")))
all_text = widget_texts(win.results_page)
check("Results header shows the video name", any("dummy_behavior_test.mp4" in t for t in all_text))
check("Results view shows a Trajectory panel", "Trajectory" in all_text)
check("Results view shows a Heatmap panel alongside it (no click needed)", "Heatmap" in all_text)

# ------------------------------------------------------------------
# 2) Batch mode, Standard Tracking, 2 videos, 'same camera' = Yes
# (mocked): BatchSummary.csv should be written.
# ------------------------------------------------------------------
win.show_setup_page()
app.processEvents()
win.batch_radio.setChecked(True)
app.processEvents()
add_video()  # 2nd queue entry (same file, just exercises the batch loop)
check("2 videos queued for batch", len(win.videos) == 2)

mb_calls.clear()
win.on_start()
app.processEvents()
check("batch run asked the 'same camera position?' question",
      any(kind == "question" for kind, _a in mb_calls))
batch_csv = os.path.join(os.path.dirname(win.videos[0]["path"]), "BatchSummary.csv")
check("BatchSummary.csv written", os.path.exists(batch_csv))
check("batch complete info dialog shown", any(kind == "information" for kind, _a in mb_calls))

# ------------------------------------------------------------------
# 3) Multi-Mouse Tracking: real run, check tracks/preview + Results page.
# ------------------------------------------------------------------
win.on_reset_all()
app.processEvents()
win._set_analysis_type("multi_mouse")
app.processEvents()
add_video()
win.on_tool_no_crop()
win.setup_page.on_num_animals(2)
app.processEvents()

mb_calls.clear()
win.on_start()
app.processEvents()
check("no error dialogs during Multi-Mouse run",
      not any(kind == "critical" for kind, _a in mb_calls))
check("stack switched to Results page after Multi-Mouse run", win.stack.currentWidget() is win.results_page)
mm_out_dir = win.results_page.output_dir
check("multi-mouse output_dir exists", bool(mm_out_dir) and os.path.isdir(mm_out_dir))
check("tracks.csv generated", os.path.exists(os.path.join(mm_out_dir, "tracks.csv")))
check("preview.mp4 generated", os.path.exists(os.path.join(mm_out_dir, "preview.mp4")))
all_text = widget_texts(win.results_page)
check("Results shows 'Per-Mouse Tracking'", any("Per-Mouse Tracking" in t for t in all_text))
check("Results shows 'Trajectory (per mouse)'", any("Trajectory (per mouse)" in t for t in all_text))

# ------------------------------------------------------------------
# 4) Behavior Classification (rule-based): real run, check features/
# bouts/labeled CSVs + the ethogram/bouts-table Results page.
# ------------------------------------------------------------------
win.on_reset_all()
app.processEvents()
win._set_analysis_type("behavior")
app.processEvents()
add_video()
win.on_tool_no_crop()
win.setup_page.on_num_animals(1)
app.processEvents()

mb_calls.clear()
win.on_start()
app.processEvents()
check("no error dialogs during Behavior Classification run",
      not any(kind == "critical" for kind, _a in mb_calls))
check("stack switched to Results page after Behavior run", win.stack.currentWidget() is win.results_page)
beh_out_dir = win.results_page.output_dir
check("behavior output_dir exists", bool(beh_out_dir) and os.path.isdir(beh_out_dir))
check("features.csv generated", os.path.exists(os.path.join(beh_out_dir, "features.csv")))
check("bouts.csv generated", os.path.exists(os.path.join(beh_out_dir, "bouts.csv")))
check("labeled_frames.csv generated", os.path.exists(os.path.join(beh_out_dir, "labeled_frames.csv")))
all_text = widget_texts(win.results_page)
check("Results shows the ethogram heading", any("Behavior Timeline" in t for t in all_text))
check("Results shows the bouts table heading", any("Behavior Bouts" in t for t in all_text))

# ML-mode guard: checking 'Use trained model' with no checkpoint chosen
# should refuse with a clear error and NOT attempt a run (Task #27 wired
# ML mode up to a real classify_video_ml() inference path -- see
# test_ml_pipeline.py for the full trained-model run this now enables).
win.show_setup_page()
app.processEvents()
win.setup_page.ml_mode_var.setChecked(True)
mb_calls.clear()
win.on_start()
app.processEvents()
check("ML mode with no checkpoint set shows a 'No trained model' error instead of running",
      any(kind == "critical" and "No trained model" in str(a) for kind, a in mb_calls))

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
