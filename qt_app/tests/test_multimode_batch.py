"""
Regression test for "Batch mode for Multi-Mouse Tracking & Behavior
Classification" (Upgrade Plan Tier 1 #5) -- before this feature,
_run_multi_mouse_flow/_run_behavior_flow always processed only the active
video no matter what the Individual/Batch radio said (only Standard
Tracking's _run_standard_flow ever looked at it). Both now share the same
per-video-loop shape Standard Tracking's batch already used (see
_process_one_multi_mouse/_process_one_behavior/_finish_batch_run in
main_window.py), including Subject Database tagging (Upgrade Plan Tier 1
#2) on each summary row.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_multimode_batch.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring/sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import shutil
import tempfile

import pandas as pd
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from tracking.location import compute_output_dir
from _pathsetup import VIDEO

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# Both analysis types share dummy_behavior_test.mp4's own output folder
# (compute_output_dir always resolves the same path for it), so clear it
# and any stale batch summaries first -- same reasoning as the other tests
# sharing this fixture.
_out_dir = compute_output_dir(VIDEO)
if os.path.isdir(_out_dir):
    shutil.rmtree(_out_dir)
    os.makedirs(_out_dir, exist_ok=True)
_mm_batch_csv = os.path.join(os.path.dirname(VIDEO), "BatchSummary_MultiMouse.csv")
_mm_batch_xlsx = os.path.join(os.path.dirname(VIDEO), "BatchSummary_MultiMouse.xlsx")
_beh_batch_csv = os.path.join(os.path.dirname(VIDEO), "BatchSummary_Behavior.csv")
_beh_batch_xlsx = os.path.join(os.path.dirname(VIDEO), "BatchSummary_Behavior.xlsx")
for _p in (_mm_batch_csv, _mm_batch_xlsx, _beh_batch_csv, _beh_batch_xlsx):
    if os.path.exists(_p):
        os.remove(_p)

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


# Subject database, reused across both sections below.
subjects_xlsx = os.path.join(tempfile.gettempdir(), "mm_multimode_test_subjects.xlsx")
pd.DataFrame({"ID": ["S1", "S2"], "Strain": ["C57BL/6", "BALB/c"]}).to_excel(subjects_xlsx, index=False)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (subjects_xlsx, ""))

# ------------------------------------------------------------------
# 1) Multi-Mouse Tracking, Batch mode, 2 queued videos.
# ------------------------------------------------------------------
win.batch_radio.setChecked(True)
app.processEvents()
win._set_analysis_type("multi_mouse")
app.processEvents()
add_video()
add_video()
win.on_import_subjects()
win.on_assign_subject(0, "S1")
win.on_assign_subject(1, "S2")
win.on_tool_no_crop()
win.setup_page.on_num_animals(2)
app.processEvents()
check("2 videos queued for Multi-Mouse batch", len(win.videos) == 2)

mb_calls.clear()
win.on_start()
app.processEvents()
check("no error dialogs during the Multi-Mouse batch run",
      not any(k == "critical" for k, _a in mb_calls))
check("batch complete info dialog shown", any(k == "information" for k, _a in mb_calls))
check("Multi-Mouse batch stays on Setup (matches Standard batch -- no single video 'the' result to show)",
      win.stack.currentWidget() is win.setup_page)

check("BatchSummary_MultiMouse.csv written", os.path.exists(_mm_batch_csv))
check("BatchSummary_MultiMouse.xlsx also written", os.path.exists(_mm_batch_xlsx))
mm_batch_df = pd.read_csv(_mm_batch_csv)
check("BatchSummary_MultiMouse.csv has at least one row per queued video (2 videos x up to 2 mice each)",
      len(mm_batch_df) >= 2)
check("BatchSummary_MultiMouse.csv has a mouse_id column (Multi-Mouse's own row shape)",
      "mouse_id" in mm_batch_df.columns)
check("BatchSummary_MultiMouse.csv tags rows with subject_id", "subject_id" in mm_batch_df.columns)
check("BatchSummary_MultiMouse.csv carries the subject database's Strain column too, prefixed",
      "Subject_Strain" in mm_batch_df.columns)
check("both S1 and S2 appear as subject_id values",
      {"S1", "S2"}.issubset(set(mm_batch_df["subject_id"])))

# Each video's own per-video output (tracks.csv/preview.mp4) still gets
# written too, same paths as Individual mode always used.
mm_out_dir = compute_output_dir(win.videos[0]["path"])
check("video 0's own tracks.csv was written", os.path.exists(os.path.join(mm_out_dir, "tracks.csv")))
check("video 0's own preview.mp4 was written", os.path.exists(os.path.join(mm_out_dir, "preview.mp4")))

# ------------------------------------------------------------------
# 2) Behavior Classification (rule-based), Batch mode, 2 queued videos.
# ------------------------------------------------------------------
win.on_reset_all()
app.processEvents()
win.batch_radio.setChecked(True)
app.processEvents()
win._set_analysis_type("behavior")
app.processEvents()
add_video()
add_video()
win.on_import_subjects()
win.on_assign_subject(0, "S1")
win.on_assign_subject(1, "S2")
win.on_tool_no_crop()
win.setup_page.on_num_animals(1)
app.processEvents()
check("2 videos queued for Behavior batch", len(win.videos) == 2)
check("ml_mode is off by default (rule-based batch path is the one under test)",
      not (getattr(win.setup_page, "ml_mode_var", None) and win.setup_page.ml_mode_var.isChecked()))

mb_calls.clear()
win.on_start()
app.processEvents()
check("no error dialogs during the Behavior batch run",
      not any(k == "critical" for k, _a in mb_calls))
check("batch complete info dialog shown for Behavior batch", any(k == "information" for k, _a in mb_calls))
check("Behavior batch stays on Setup too",
      win.stack.currentWidget() is win.setup_page)

check("BatchSummary_Behavior.csv written", os.path.exists(_beh_batch_csv))
check("BatchSummary_Behavior.xlsx also written", os.path.exists(_beh_batch_xlsx))
beh_batch_df = pd.read_csv(_beh_batch_csv)
check("BatchSummary_Behavior.csv has exactly 2 rows (one per video)", len(beh_batch_df) == 2)
check("BatchSummary_Behavior.csv tags rows with subject_id", "subject_id" in beh_batch_df.columns)
check("BatchSummary_Behavior.csv carries the subject database's Strain column too, prefixed",
      "Subject_Strain" in beh_batch_df.columns)
check("BatchSummary_Behavior.csv has Duration_s and Total_bouts columns",
      "Duration_s" in beh_batch_df.columns and "Total_bouts" in beh_batch_df.columns)
check("at least one per-behavior aggregate column made it in (e.g. '<behavior>_count')",
      any(c.endswith("_count") and c != "Total_bouts" for c in beh_batch_df.columns))

beh_out_dir = compute_output_dir(win.videos[0]["path"])
check("video 0's own features.csv was written", os.path.exists(os.path.join(beh_out_dir, "features.csv")))
check("video 0's own bouts.csv was written", os.path.exists(os.path.join(beh_out_dir, "bouts.csv")))

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
