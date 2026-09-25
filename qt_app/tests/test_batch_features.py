"""
Headless regression test for the 4 features MM asked for after sending
reference videos of SMART 3.0 / ANY-maze (rodent tracking competitors) and
picking a priority list via AskUserQuestion: batch mode's per-video zone
alignment (a separate override per queued video, NOT the montage/live-
camera approach he declined), the Subject database (Excel import +
per-video assignment), and direct Excel export (single run + batch
summary). The Rectangle/Ellipse/Line shape tools + snap-to-grid (the 4th
item) are covered in test_canvas_smoke.py alongside the rest of the canvas
op machinery.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_batch_features.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import shutil
import tempfile

import pandas as pd
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QPushButton

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


# Clean up any stale results from an earlier run sharing this same video
# (see test_ml_pipeline.py for why this matters -- all tests share
# dummy_behavior_test.mp4, so compute_output_dir() always resolves to the
# same folder).
_out_dir = compute_output_dir(VIDEO)
if os.path.isdir(_out_dir):
    shutil.rmtree(_out_dir)
    os.makedirs(_out_dir, exist_ok=True)
_batch_csv = os.path.join(os.path.dirname(VIDEO), "BatchSummary.csv")
_batch_xlsx = os.path.join(os.path.dirname(VIDEO), "BatchSummary.xlsx")
for _p in (_batch_csv, _batch_xlsx):
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


# ------------------------------------------------------------------
# 1) Subject database: import from Excel, assign to queued videos.
# ------------------------------------------------------------------
subjects_xlsx = os.path.join(tempfile.gettempdir(), "mm_test_subjects.xlsx")
pd.DataFrame({"ID": ["M1", "M2", "M3"], "Sex": ["M", "F", "M"], "Group": ["Control", "Treated", "Treated"]}) \
    .to_excel(subjects_xlsx, index=False)
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (subjects_xlsx, ""))

check("no subjects imported yet", win.subjects == [])
win.on_import_subjects()
check("on_import_subjects loaded 3 rows", len(win.subjects) == 3)
check("subject_ids() reads the 'ID' column", win.subject_ids() == ["M1", "M2", "M3"])

win.batch_radio.setChecked(True)
app.processEvents()
add_video()
add_video()
check("2 videos queued for batch", len(win.videos) == 2)

win.on_assign_subject(0, "M1")
check("video 0 assigned to M1", win.videos[0]["subject_id"] == "M1")
win.on_assign_subject(0, "")
check("clearing the assignment removes subject_id", "subject_id" not in win.videos[0])
win.on_assign_subject(0, "M1")
win.on_assign_subject(1, "M2")
check("video 1 assigned to M2", win.videos[1]["subject_id"] == "M2")

win.on_clear_subjects()
check("on_clear_subjects empties the subject list", win.subjects == [])
check("clearing subjects also strips subject_id from queued videos",
      all("subject_id" not in v for v in win.videos))

# Re-import for the rest of the test (batch run below tags summaries with
# subject_id, so we want it populated again).
win.on_import_subjects()
win.on_assign_subject(0, "M1")
win.on_assign_subject(1, "M2")

# ------------------------------------------------------------------
# 2) Per-video batch zone alignment ('Align' button): needs a template
# (pending_roi_points) first, then on_align_video starts a drag-only
# template_zones op tagged for that specific video, and Finish saves into
# THAT video's roi_points_override without touching the shared template.
# ------------------------------------------------------------------
mb_calls.clear()
win.on_align_video(1)
check("on_align_video with no template defined shows a critical dialog",
      any(k == "critical" and "Set up zones first" in str(a) for k, a in mb_calls))
check("no op was started", win._op is None)

win.on_tool_no_crop()
app.processEvents()
win.setup_page.roi_names_entry.setText("Left, Right")
w, h = win.videos[0]["width"], win.videos[0]["height"]
win.start_op("zones")
for fx, fy in [(10, 10), (w // 2 - 5, 10), (w // 2 - 5, h - 10), (10, h - 10)]:
    win.on_canvas_press(fx, fy)
win.op_set_active_region("Right")
for fx, fy in [(w // 2 + 5, 10), (w - 10, 10), (w - 10, h - 10), (w // 2 + 5, h - 10)]:
    win.on_canvas_press(fx, fy)
win.finish_op()
template_before = {k: list(v) for k, v in win.pending_roi_points.items()}
check("shared zone template ('Left'/'Right') set on the active video",
      set(win.pending_roi_points.keys()) == {"Left", "Right"})

win.on_align_video(1)
check("on_align_video started a template_zones op targeted at video 1",
      win._op is not None and win._op["kind"] == "template_zones" and win._op.get("_align_target_index") == 1)
check("align starts from the shared template's own zone shapes",
      set(win._op["regions"].keys()) == {"Left", "Right"})
check("active_index switched to the video being aligned", win.active_index == 1)

no_rename_widgets = [win.setup_page.op_region_row.itemAt(i).widget()
                      for i in range(win.setup_page.op_region_row.count())]
check("no per-zone rename buttons shown while aligning a specific video (names must stay fixed)",
      not any(isinstance(w, QPushButton) for w in no_rename_widgets))

left_pts = win._op["regions"]["Left"]
cx, cy = left_pts[0]
win.on_canvas_press(cx, cy)
win.on_canvas_drag(cx + 12, cy - 8)
win.on_canvas_release()
win.finish_op()
check("Finish saved an override on video 1, not on the shared template",
      win.videos[1].get("roi_points_override") is not None)
check("the override reflects the dragged corner",
      win.videos[1]["roi_points_override"]["Left"][0] == (cx + 12, cy - 8))
check("the shared pending_roi_points template is untouched by aligning a single video",
      win.pending_roi_points == template_before)
check("video 0 has no override (only video 1 was aligned)",
      "roi_points_override" not in win.videos[0])

# ------------------------------------------------------------------
# 3) Batch run: overrides present -> the new 'per-video alignment'
# confirm (not the old same-camera-position question), override used for
# video 1, shared template used for video 0, subject_id tagged onto each
# summary row, and BOTH BatchSummary.csv and .xlsx get written.
# ------------------------------------------------------------------
mb_calls.clear()
win.on_start()
app.processEvents()
check("batch asked the NEW per-video-alignment question (not the old same-camera one)",
      any(k == "question" and "Per-video alignment" in str(a) for k, a in mb_calls))
check("no error dialogs during the batch run",
      not any(k == "critical" for k, _a in mb_calls))
check("BatchSummary.csv written", os.path.exists(_batch_csv))
check("BatchSummary.xlsx also written alongside it", os.path.exists(_batch_xlsx))

batch_df = pd.read_csv(_batch_csv)
check("BatchSummary.csv has 2 rows (one per video)", len(batch_df) == 2)
check("BatchSummary.csv tags each row with its assigned subject_id",
      "subject_id" in batch_df.columns and set(batch_df["subject_id"]) == {"M1", "M2"})
batch_xlsx_df = pd.read_excel(_batch_xlsx)
check("BatchSummary.xlsx round-trips the same row count", len(batch_xlsx_df) == 2)

# ------------------------------------------------------------------
# 4) Export to Excel (Results page): builds one workbook, one sheet per
# result CSV in the run's output folder.
# ------------------------------------------------------------------
win.on_reset_all()
app.processEvents()
# individual_radio/batch_radio live on MainWindow itself (built once in
# _build_mode_row), not inside the rebuilt SetupPage, so on_reset_all()
# does NOT reset mode back to Individual by itself -- do it explicitly.
win.individual_radio.setChecked(True)
app.processEvents()
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

mb_calls.clear()
win.on_start()
app.processEvents()
check("no error dialogs during the single-video run", not any(k == "critical" for k, _a in mb_calls))
out_dir = win.results_page.output_dir
check("output_dir set for the Excel-export test", bool(out_dir) and os.path.isdir(out_dir))

mb_calls.clear()
win.results_page.on_export_excel()
check("Export to Excel showed an 'Exported' confirmation, no error dialog",
      any(k == "information" and "Exported" in str(a) for k, a in mb_calls) and
      not any(k == "critical" for k, a in mb_calls))
xlsx_path = os.path.join(out_dir, "Results_Export.xlsx")
check("Results_Export.xlsx was written", os.path.exists(xlsx_path))
exported = pd.read_excel(xlsx_path, sheet_name=None)
check("Results_Export.xlsx has at least one sheet with rows",
      len(exported) > 0 and any(len(df) > 0 for df in exported.values()))

# Calling export_results_excel directly on a folder with no CSVs should
# raise, not silently produce an empty/garbage workbook.
empty_dir = tempfile.mkdtemp(prefix="mm_test_empty_")
raised = False
try:
    win.export_results_excel(empty_dir)
except ValueError:
    raised = True
check("export_results_excel raises when the folder has no result CSVs", raised)
shutil.rmtree(empty_dir, ignore_errors=True)

# ------------------------------------------------------------------
# 5) Project save/load round-trips subjects + per-video overrides
# (tuples survive the JSON list round-trip as tuples again).
# ------------------------------------------------------------------
proj_path = os.path.join(tempfile.gettempdir(), "mm_test_project.btproj")
win.on_reset_all()
app.processEvents()
win.batch_radio.setChecked(True)
app.processEvents()
add_video()
add_video()
win.on_import_subjects()
win.on_assign_subject(0, "M3")
win.on_tool_no_crop()
win.setup_page.roi_names_entry.setText("Left, Right")
win.start_op("zones")
for fx, fy in [(10, 10), (100, 10), (100, 100), (10, 100)]:
    win.on_canvas_press(fx, fy)
win.op_set_active_region("Right")
for fx, fy in [(150, 10), (250, 10), (250, 100), (150, 100)]:
    win.on_canvas_press(fx, fy)
win.finish_op()
win.on_align_video(1)
for fx, fy in [(10, 10), (100, 10), (100, 100), (10, 100)]:
    win.on_canvas_press(fx, fy)  # corners already exist -> drags, doesn't add points
win.finish_op()
saved_override = win.videos[1]["roi_points_override"]

QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (proj_path, ""))
QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (proj_path, ""))
win.on_save_project()
check("project file written", os.path.exists(proj_path))

win.on_reset_all()
app.processEvents()
win.on_open_project()
check("re-opened project restored the subject list", win.subject_ids() == ["M1", "M2", "M3"])
check("re-opened project restored video 0's subject assignment", win.videos[0].get("subject_id") == "M3")
check("re-opened project restored video 1's zone override",
      win.videos[1].get("roi_points_override", {}).get("Left") == saved_override["Left"])

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
