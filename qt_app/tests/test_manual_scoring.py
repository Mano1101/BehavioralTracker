"""
Headless regression test for the Manual Scoring dialog (Task #27). Covers
the guard checks, unit-level bout open/close/undo/subject-switching/finish()
behavior on the dialog itself, and a full keyboard-driven session through
MainWindow using real QTest key events (a plain method call wouldn't catch
the Tab-focus-chain bug this dialog works around -- see
focusNextPrevChild() in manual_scoring_dialog.py).

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_manual_scoring.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import sys, os

import pandas as pd
from PySide6.QtWidgets import QApplication, QMessageBox, QDialog
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from qt_app.dialogs.manual_scoring_dialog import ManualScoringDialog
from _pathsetup import VIDEO

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


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
# 1) Guard checks: no video, no arena.
# ------------------------------------------------------------------
mb_calls.clear()
win.on_manual_behavior_scoring()
check("no video -> critical dialog", any(k == "critical" for k, _ in mb_calls))

add_video()
mb_calls.clear()
win.on_manual_behavior_scoring()
check("video but no arena -> 'Arena not set' critical dialog",
      any(k == "critical" and "Arena not set" in str(a) for k, a in mb_calls))

# ------------------------------------------------------------------
# 2) Unit-level checks on the dialog itself: bout open/close/undo,
# subject switching, and finish() producing the right DataFrame shape.
# ------------------------------------------------------------------
dlg = ManualScoringDialog(win, VIDEO, ["mouse_A", "mouse_B"])
check("dialog opened and read frame 0", dlg.cur_idx == 0 and dlg.frame is not None)
check("starts paused", dlg.playing is False)
check("active subject defaults to the first", dlg.active_subject == "mouse_A")

dlg.toggle_behavior("locomotion")
check("toggle opens a bout for the active subject",
      dlg.open_bouts["mouse_A"] is not None and dlg.open_bouts["mouse_A"]["behavior"] == "locomotion")

dlg.switch_subject()
check("Tab switches the active subject", dlg.active_subject == "mouse_B")
check("switching subjects leaves mouse_A's bout open (independent per subject)",
      dlg.open_bouts["mouse_A"] is not None)

dlg.toggle_behavior("grooming")
dlg.step(1)
dlg.step(1)
dlg.toggle_behavior("grooming")  # same key again -> closes it
check("pressing the SAME behavior key again closes that subject's bout",
      dlg.open_bouts["mouse_B"] is None)
check("closing produced one completed bout so far", len(dlg.completed) == 1)
check("completed bout is tagged source='Manual'", dlg.completed[0]["source"] == "Manual")

dlg.undo()
check("undo removes the last completed bout", len(dlg.completed) == 0)

# Re-create the same grooming bout so finish() has something to close/save.
dlg.toggle_behavior("grooming")
dlg.step(1)

dlg.change_speed(-10)
check("change_speed clamps at the slowest preset (index 0)", dlg.speed_idx == 0)
dlg.change_speed(10)
check("change_speed clamps at the fastest preset (index 4, 4.0x)", dlg.speed_idx == 4)

dlg.finish()
check("finish() populates bouts_df", dlg.bouts_df is not None and len(dlg.bouts_df) == 2)
check("finish() closed mouse_A's still-open locomotion bout too",
      set(dlg.bouts_df["subject"]) == {"mouse_A", "mouse_B"})
check("finish() released the capture", not dlg.cap.isOpened())
check("finish() is idempotent (calling twice doesn't error / doesn't duplicate)",
      dlg.finish() is None and len(dlg.bouts_df) == 2)

# ------------------------------------------------------------------
# 3) Full flow through MainWindow: keyboard-driven session via a
# patched exec() (QTest delivers REAL key events; exec() itself can't
# be entered headlessly since nothing would ever click a button to
# leave it), then check the CSV + Results page.
# ------------------------------------------------------------------
win._set_analysis_type("behavior")  # on_num_animals(2) is blocked while
                                     # analysis_type == "standard" (its own
                                     # 1-animal-only guard), same as the
                                     # Start Tracking flows.
app.processEvents()
win.on_tool_no_crop()
win.setup_page.on_num_animals(2)
app.processEvents()
check("num_animals actually reached 2 before driving the dialog", win.setup_page.num_animals == 2)


def _driven_exec(self):
    self.show()
    app.processEvents()
    QTest.keyClick(self, Qt.Key_3)        # mouse_A: open locomotion
    app.processEvents()
    QTest.keyClick(self, Qt.Key_Tab)      # switch to mouse_B
    app.processEvents()
    QTest.keyClick(self, Qt.Key_1)        # mouse_B: open grooming
    app.processEvents()
    QTest.keyClick(self, Qt.Key_Period)   # step forward a couple frames
    QTest.keyClick(self, Qt.Key_Period)
    app.processEvents()
    QTest.keyClick(self, Qt.Key_Q)        # finish & save
    app.processEvents()
    return self.result()


ManualScoringDialog.exec = _driven_exec
mb_calls.clear()
win.on_manual_behavior_scoring()
app.processEvents()

check("no error dialogs during the manual-scoring flow",
      not any(k == "critical" for k, _ in mb_calls))
check("stack switched to Results page", win.stack.currentWidget() is win.results_page)
out_dir = win.results_page.output_dir
manual_csv = os.path.join(out_dir, "manual_bouts.csv")
check("manual_bouts.csv written", os.path.exists(manual_csv))

saved = pd.read_csv(manual_csv)
check("saved CSV has both subjects' bouts", set(saved["subject"]) == {"mouse_A", "mouse_B"})
check("saved CSV bouts are all source='Manual'", set(saved["source"]) == {"Manual"})

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
