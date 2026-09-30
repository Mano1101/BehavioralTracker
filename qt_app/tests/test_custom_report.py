"""
Regression test for "Custom report builder" (Upgrade Plan Tier 2 #10,
SMART's Report Definitions Manager) -- the final Tier 2 item: "a simple
'pick which computed stats go in this sheet' UI on the Results screen,
instead of the fixed Summary sheet shape." Adds a "Custom Report" button
to ResultsPage (all three analysis types) that opens a checkbox picker
over whichever "full" stats table that analysis type already builds
(Standard Tracking's summary dict, Multi-Mouse's per_mouse_summary,
Behavior Classification's new tracking.behavior.behavior_summary_table),
and saves the narrowed selection to Custom_Report.csv/.xlsx.

Four things are covered:

1. analysis.custom_report's two pure, analysis-type-agnostic functions --
   reportable_columns()/build_custom_report() -- column narrowing only,
   no stat computation, against small hand-built DataFrames.

2. tracking.behavior.behavior_summary_table() -- pivots a hand-built
   bouts_df (the same shape produced by the rule-based classifier, the ML
   path, and manual scoring alike) into one row per subject with
   <behavior>_count/_total_s/_mean_s columns, against manually-verified
   expected values.

3. ResultsPage wiring, driven directly (no MainWindow needed -- a
   throwaway stand-in `app` object is enough, since ResultsPage only ever
   calls back into it from button clicks this test doesn't trigger): for
   each of the three show_standard/show_multi_mouse/show_behavior calls,
   the right "full" table and id column(s) end up in
   self._report_full_df/_report_id_columns, the "Custom Report" button
   only appears when there's at least one optional stat to pick (an empty
   per_mouse_summary hides it), and driving the picker dialog (the same
   QDialog.exec monkeypatch technique test_settings_screen.py/
   test_ml_pipeline.py use) actually narrows the saved
   Custom_Report.csv/.xlsx to just the checked columns, always keeping
   the id column(s), and refuses (with a warning, no file written) when
   nothing is checked.

4. A real end-to-end check through an actual MainWindow: running Standard
   Tracking for real, then clicking the real "Custom Report" button
   widget (not calling the handler directly) and saving, to confirm the
   button is actually wired up against real pipeline output, not just
   the hand-built results dicts in part 3.

Run directly (needs a virtual display for part 4 -- see _pathsetup.py's
docstring):
    xvfb-run -a python3.12 qt_app/tests/test_custom_report.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import tempfile

import pandas as pd
from PySide6.QtWidgets import QApplication, QDialog, QCheckBox, QPushButton, QMessageBox

# Headless run: patch QMessageBox statics BEFORE any dialog can appear --
# the "Select None" -> "Nothing to save" warning in part 3 would otherwise
# pop a real modal dialog and hang under xvfb.
mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from analysis.custom_report import reportable_columns, build_custom_report
from tracking.behavior import behavior_summary_table

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


def click_named(dialog, object_name):
    for btn in dialog.findChildren(QPushButton):
        if btn.objectName() == object_name:
            btn.click()
            return True
    return False


# ------------------------------------------------------------------
# 1) analysis.custom_report -- pure column-narrowing, no stats computed.
# ------------------------------------------------------------------
full = pd.DataFrame({"mouse_id": ["A", "B"], "Total_distance_pixels": [10.0, 20.0], "Zone_time_s": [1.0, 2.0]})

check("reportable_columns excludes the id column(s)",
      reportable_columns(full, id_columns=["mouse_id"]) == ["Total_distance_pixels", "Zone_time_s"])
check("reportable_columns returns [] for an empty/columnless DataFrame",
      reportable_columns(pd.DataFrame(), id_columns=["mouse_id"]) == []
      and reportable_columns(None, id_columns=["mouse_id"]) == [])

narrowed = build_custom_report(full, ["Total_distance_pixels"], id_columns=["mouse_id"])
check("build_custom_report always keeps the id column even though it wasn't 'selected'",
      list(narrowed.columns) == ["mouse_id", "Total_distance_pixels"])
check("build_custom_report keeps the ORIGINAL column order, not selection order",
      list(build_custom_report(full, ["Zone_time_s", "Total_distance_pixels"], id_columns=["mouse_id"]).columns)
      == ["mouse_id", "Total_distance_pixels", "Zone_time_s"])
check("build_custom_report silently drops an unknown/stale selected column name instead of raising",
      list(build_custom_report(full, ["Total_distance_pixels", "Not_A_Real_Column"], id_columns=["mouse_id"]).columns)
      == ["mouse_id", "Total_distance_pixels"])
check("build_custom_report with nothing selected and no id columns returns an empty DataFrame",
      len(build_custom_report(full, [], id_columns=[]).columns) == 0)
check("build_custom_report on an empty/None full_df returns an empty DataFrame",
      len(build_custom_report(pd.DataFrame(), ["x"], id_columns=["mouse_id"]).columns) == 0
      and len(build_custom_report(None, ["x"], id_columns=["mouse_id"]).columns) == 0)

# ------------------------------------------------------------------
# 2) tracking.behavior.behavior_summary_table -- hand-built bouts_df.
# ------------------------------------------------------------------
bouts_df = pd.DataFrame({
    "subject": ["mouse_A", "mouse_A", "mouse_A", "mouse_A", "mouse_B"],
    "behavior": ["rearing", "rearing", "locomotion", "other", "rearing"],
    "duration_s": [1.0, 2.0, 3.0, 0.5, 4.0],
})
selected_behaviors = {"rearing": True, "locomotion": True, "grooming": True, "immobile": True}
beh_table = behavior_summary_table(bouts_df, selected_behaviors)
check("behavior_summary_table produces one row per subject -- 2 subjects",
      len(beh_table) == 2)
row_a = beh_table[beh_table["subject"] == "mouse_A"].iloc[0]
check("mouse_A: 2 rearing bouts, 3.0s total, 1.5s mean",
      row_a["rearing_count"] == 2 and abs(row_a["rearing_total_s"] - 3.0) < 1e-6
      and abs(row_a["rearing_mean_s"] - 1.5) < 1e-6)
check("mouse_A: 1 locomotion bout, 3.0s total/mean",
      row_a["locomotion_count"] == 1 and abs(row_a["locomotion_total_s"] - 3.0) < 1e-6)
check("mouse_A: the un-checked-but-always-kept 'other' bout is included too (0.5s)",
      row_a["other_count"] == 1 and abs(row_a["other_total_s"] - 0.5) < 1e-6)
row_b = beh_table[beh_table["subject"] == "mouse_B"].iloc[0]
check("mouse_B: 1 rearing bout (4.0s), 0 locomotion (not just a missing column)",
      row_b["rearing_count"] == 1 and row_b["locomotion_count"] == 0
      and abs(row_b["locomotion_total_s"]) < 1e-6)
check("a deselected behavior with zero bouts anywhere contributes no columns at all "
      "('grooming'/'immobile' never occur in bouts_df)",
      not any(c.startswith("grooming_") or c.startswith("immobile_") for c in beh_table.columns))
check("behavior_summary_table on an empty/None bouts_df returns an empty DataFrame",
      len(behavior_summary_table(pd.DataFrame(), selected_behaviors)) == 0
      and len(behavior_summary_table(None, selected_behaviors)) == 0)
check("behavior_summary_table drops behaviors that are off AND have no bouts kept "
      "(here nothing is off, so re-check with locomotion turned off)",
      "locomotion_count" not in behavior_summary_table(
          bouts_df, {"rearing": True, "locomotion": False, "grooming": True, "immobile": True}).columns)

# ------------------------------------------------------------------
# 3) ResultsPage wiring, driven directly (no MainWindow needed).
# ------------------------------------------------------------------
from qt_app.pages.results_page import ResultsPage


class FakeApp:
    show_setup_page = staticmethod(lambda: None)

    def export_results_excel(self, output_dir):
        raise NotImplementedError


app = QApplication.instance() or QApplication(sys.argv)
rp = ResultsPage(FakeApp())

# --- 3a) Standard Tracking ---
std_out_dir = tempfile.mkdtemp(prefix="rp_std_")
summary = {
    "Video": "vid.mp4", "Output_folder": std_out_dir,
    "Total_distance_pixels": 123.4, "Tracking_quality_percent": 99.0,
}
rp.show_standard(summary)
check("Standard Tracking: _report_full_df is a one-row DataFrame with exactly summary's keys",
      list(rp._report_full_df.columns) == list(summary.keys()) and len(rp._report_full_df) == 1)
check("Standard Tracking: 'Custom Report' button appears in the header",
      find_button(rp, "Custom Report") is not None)

_orig_exec = QDialog.exec


def _drive_std(self):
    cbs = self.findChildren(QCheckBox)
    check("dialog lists every summary key as a pickable column (Standard has no id column)",
          {cb.text() for cb in cbs} == set(summary.keys()))
    for cb in cbs:
        if cb.text() in ("Tracking_quality_percent",):
            cb.setChecked(False)
    check("Save Report button found", click_named(self, "accentBtn"))
    return QDialog.Accepted


QDialog.exec = _drive_std
rp.on_custom_report()
QDialog.exec = _orig_exec

std_csv = os.path.join(std_out_dir, "Custom_Report.csv")
std_xlsx = os.path.join(std_out_dir, "Custom_Report.xlsx")
check("Standard Tracking: Custom_Report.csv was written", os.path.exists(std_csv))
check("Standard Tracking: Custom_Report.xlsx was written too", os.path.exists(std_xlsx))
saved_std = pd.read_csv(std_csv)
check("...the unchecked column was left out", "Tracking_quality_percent" not in saved_std.columns)
check("...the still-checked column was kept, with its value intact",
      abs(saved_std.loc[0, "Total_distance_pixels"] - 123.4) < 1e-6)

# --- 3b) Multi-Mouse Tracking: empty per_mouse_summary -> no button at all ---
mm_empty_out_dir = tempfile.mkdtemp(prefix="rp_mm_empty_")
rp.show_multi_mouse({
    "output_dir": mm_empty_out_dir, "video_path": "vid.mp4",
    "tracks_df": pd.DataFrame({"mouse_id": [], "time_s": [], "x": [], "y": [], "status": []}),
    "per_mouse_summary": {}, "annotate_path": os.path.join(mm_empty_out_dir, "preview.mp4"),
})
check("Multi-Mouse with an empty per_mouse_summary: no 'Custom Report' button "
      "(nothing to build a report from)", find_button(rp, "Custom Report") is None)

# --- 3b continued) Multi-Mouse with real per-mouse stats ---
mm_out_dir = tempfile.mkdtemp(prefix="rp_mm_")
results_mm = {
    "output_dir": mm_out_dir, "video_path": "vid.mp4",
    "tracks_df": pd.DataFrame({
        "mouse_id": ["mouse_A", "mouse_B"], "time_s": [0.0, 0.0],
        "x": [1.0, 2.0], "y": [1.0, 2.0], "status": ["ok", "ok"],
    }),
    "per_mouse_summary": {
        "mouse_A": {"Total_distance_pixels": 100.0, "ZoneA_time_s": 5.0},
        "mouse_B": {"Total_distance_pixels": 50.0, "ZoneA_time_s": 2.0},
    },
    "annotate_path": os.path.join(mm_out_dir, "preview.mp4"),
}
rp.show_multi_mouse(results_mm)
check("Multi-Mouse: _report_full_df has one row per mouse_id",
      list(rp._report_full_df["mouse_id"]) == ["mouse_A", "mouse_B"])
check("Multi-Mouse: _report_full_df's columns are mouse_id + the stat keys",
      set(rp._report_full_df.columns) == {"mouse_id", "Total_distance_pixels", "ZoneA_time_s"})
check("Multi-Mouse: 'Custom Report' button appears now that there's something to pick",
      find_button(rp, "Custom Report") is not None)


def _drive_mm(self):
    cbs = self.findChildren(QCheckBox)
    check("Multi-Mouse dialog does NOT offer mouse_id as a pickable column (it's the id column)",
          "mouse_id" not in {cb.text() for cb in cbs})
    for cb in cbs:
        if cb.text() == "ZoneA_time_s":
            cb.setChecked(False)
    check("Save Report button found", click_named(self, "accentBtn"))
    return QDialog.Accepted


QDialog.exec = _drive_mm
rp.on_custom_report()
QDialog.exec = _orig_exec

mm_csv = os.path.join(mm_out_dir, "Custom_Report.csv")
check("Multi-Mouse: Custom_Report.csv was written", os.path.exists(mm_csv))
saved_mm = pd.read_csv(mm_csv)
check("...mouse_id (the id column) is always kept even though it can't be unchecked",
      list(saved_mm["mouse_id"]) == ["mouse_A", "mouse_B"])
check("...the unchecked stat column was excluded", "ZoneA_time_s" not in saved_mm.columns)
check("...the still-checked stat column kept its per-mouse values",
      list(saved_mm["Total_distance_pixels"]) == [100.0, 50.0])

# --- 3c) Behavior Classification ---
beh_out_dir = tempfile.mkdtemp(prefix="rp_beh_")
beh_bouts_df = pd.DataFrame({
    "subject": ["mouse_A"] * 4,
    "behavior": ["rearing", "rearing", "locomotion", "other"],
    "duration_s": [1.0, 2.0, 3.0, 0.5],
})
results_beh = {
    "output_dir": beh_out_dir, "video_path": "vid.mp4",
    "bouts_df": beh_bouts_df,
    "selected_behaviors": {"rearing": True, "locomotion": True, "grooming": True, "immobile": True},
    "labeled_df": pd.DataFrame({"time_s": [0.0, 3.5]}),
}
rp.show_behavior(results_beh)
check("Behavior: _report_full_df has one row (single subject)", len(rp._report_full_df) == 1)
check("Behavior: rearing stats computed correctly on the Results page's own full table",
      rp._report_full_df.loc[0, "rearing_count"] == 2
      and abs(rp._report_full_df.loc[0, "rearing_total_s"] - 3.0) < 1e-6)
check("Behavior: 'Custom Report' button appears", find_button(rp, "Custom Report") is not None)


def _drive_beh(self):
    cbs = self.findChildren(QCheckBox)
    check("Behavior dialog does NOT offer 'subject' as a pickable column (it's the id column)",
          "subject" not in {cb.text() for cb in cbs})
    for cb in cbs:
        if cb.text().startswith("other_"):
            cb.setChecked(False)
    check("Save Report button found", click_named(self, "accentBtn"))
    return QDialog.Accepted


QDialog.exec = _drive_beh
rp.on_custom_report()
QDialog.exec = _orig_exec

beh_csv = os.path.join(beh_out_dir, "Custom_Report.csv")
check("Behavior: Custom_Report.csv was written", os.path.exists(beh_csv))
saved_beh = pd.read_csv(beh_csv)
check("...subject (the id column) was kept", list(saved_beh["subject"]) == ["mouse_A"])
check("...the unchecked other_* columns were excluded",
      not any(c.startswith("other_") for c in saved_beh.columns))
check("...the still-checked rearing/locomotion columns were kept",
      "rearing_count" in saved_beh.columns and "locomotion_count" in saved_beh.columns)

# --- 3d) "Select None" then Save -> warns, writes nothing ---
none_out_dir = tempfile.mkdtemp(prefix="rp_std_none_")
rp.show_standard({"Video": "v2.mp4", "Output_folder": none_out_dir, "Total_distance_pixels": 1.0})


def _drive_none(self):
    for btn in self.findChildren(QPushButton):
        if btn.text() == "Select None":
            btn.click()
            break
    check("Save Report button found", click_named(self, "accentBtn"))
    return QDialog.Accepted


mb_calls.clear()
QDialog.exec = _drive_none
rp.on_custom_report()
QDialog.exec = _orig_exec

check("'Select None' then Save -> a warning was shown", any(k == "warning" for k, _a in mb_calls))
check("...and no Custom_Report.csv was written", not os.path.exists(os.path.join(none_out_dir, "Custom_Report.csv")))

# ------------------------------------------------------------------
# 4) Real end-to-end: an actual MainWindow, a real Standard Tracking run,
# and clicking the REAL 'Custom Report' button widget (not calling the
# handler directly).
# ------------------------------------------------------------------
from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from _pathsetup import VIDEO

win = MainWindow()
win.resize(1600, 980)
win.setStyleSheet(build_stylesheet())
win.show()
app.processEvents()

entry = win._probe_video(VIDEO)
win.videos.append(entry)
win.active_index = 0
win.setup_page.refresh_video_list()
app.processEvents()
win.on_tool_no_crop()

# Standard Tracking (unlike Multi-Mouse/Behavior) refuses to start with no
# zone drawn at all -- draw one small rectangle so on_start() can proceed.
win.setup_page.roi_names_entry.setText("Left")
win.start_op("zones")
w, h = win.videos[0]["width"], win.videos[0]["height"]
for fx, fy in [(10, 10), (w // 2, 10), (w // 2, h - 10), (10, h - 10)]:
    win.on_canvas_press(fx, fy)
win.finish_op()
app.processEvents()

mb_calls.clear()
win.on_start()
app.processEvents()
check("real Standard Tracking run completes with no error dialogs",
      not any(k == "critical" for k, _a in mb_calls))
check("real run's Results page has a 'Custom Report' button",
      find_button(win.results_page, "Custom Report") is not None)

QDialog.exec = _drive_std_all = lambda self: (click_named(self, "accentBtn"), QDialog.Accepted)[1]
btn = find_button(win.results_page, "Custom Report")
btn.click()
QDialog.exec = _orig_exec
app.processEvents()

real_out_dir = win.results_page.output_dir
check("real end-to-end run wrote Custom_Report.csv into the actual results folder",
      os.path.exists(os.path.join(real_out_dir, "Custom_Report.csv")))
check("real end-to-end run wrote Custom_Report.xlsx too",
      os.path.exists(os.path.join(real_out_dir, "Custom_Report.xlsx")))

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
