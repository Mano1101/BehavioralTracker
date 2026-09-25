"""
Headless regression test for the Maze Template dialog (Task #27). Covers
the guard checks, the happy path (defaults -> a template_zones op with
the template's own zone labels), invalid input, and real-world-unit
labeling of 'length' params once a distance calibration is active.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_maze_template_dialog.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import sys, os

from PySide6.QtWidgets import (
    QApplication, QMessageBox, QDialog, QPushButton, QLineEdit, QComboBox,
)

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from tracking.maze_templates import TEMPLATES as MAZE_TEMPLATES
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

entry = win._probe_video(VIDEO)
win.videos.append(entry)
win.active_index = 0
win.setup_page.refresh_video_list()
app.processEvents()

# ------------------------------------------------------------------
# 1) "Arena needed" guard -- no crop/no-crop set yet, dialog must refuse
# to open at all (no QDialog.exec should even be reached).
# ------------------------------------------------------------------
_orig_exec = QDialog.exec
exec_calls = []


def _guard_exec(self):
    exec_calls.append(self.windowTitle())
    return _orig_exec(self)


QDialog.exec = _guard_exec
mb_calls.clear()
win.on_open_maze_template_dialog()
check("no arena -> critical dialog shown", any(k == "critical" for k, _ in mb_calls))
check("no arena -> dialog never opened", exec_calls == [])

# ------------------------------------------------------------------
# 2) Happy path: arena set, accept the first template's DEFAULT params
# (click 'Generate Zones' immediately) -> a template_zones op starts
# with exactly that template's zone labels.
# ------------------------------------------------------------------
win.on_tool_no_crop()
app.processEvents()

first_key = list(MAZE_TEMPLATES.keys())[0]
expected_labels = set()
default_params = {pkey: default for pkey, _l, default, _k in MAZE_TEMPLATES[first_key]["params"]}
for label, _pts in MAZE_TEMPLATES[first_key]["generate"](win.pending_warp_w, win.pending_warp_h, default_params):
    expected_labels.add(label)


def _accept_defaults_exec(self):
    exec_calls.append(self.windowTitle())
    for btn in self.findChildren(QPushButton):
        if btn.objectName() == "accentBtn":
            btn.click()
            break
    return QDialog.Accepted


QDialog.exec = _accept_defaults_exec
exec_calls.clear()
mb_calls.clear()
win.on_open_maze_template_dialog()
app.processEvents()
check("dialog opened with arena set", exec_calls == ["Maze Template"])
check("Generate Zones started a template_zones op",
      win._op is not None and win._op["kind"] == "template_zones")
check("generated regions match the template's own zone labels",
      set(win._op["regions"].keys()) == expected_labels)
win.cancel_op()
app.processEvents()

# ------------------------------------------------------------------
# 3) Invalid numeric input -> critical dialog, op NOT started, dialog
# stays open (mirrors Tkinter: bad value shows an error and returns,
# it doesn't close the dialog).
# ------------------------------------------------------------------
def _bad_value_exec(self):
    exec_calls.append(self.windowTitle())
    edits = self.findChildren(QLineEdit)
    if edits:
        edits[0].setText("not-a-number")
    for btn in self.findChildren(QPushButton):
        if btn.objectName() == "accentBtn":
            btn.click()
            break
    return QDialog.Rejected  # test harness closes it after the click, unlike the app


QDialog.exec = _bad_value_exec
exec_calls.clear()
mb_calls.clear()
win._op = None
win.on_open_maze_template_dialog()
app.processEvents()
check("non-numeric field -> 'Invalid value' critical dialog",
      any(k == "critical" and "Invalid value" in str(a) for k, a in mb_calls))
check("invalid input did not start an op", win._op is None)

# ------------------------------------------------------------------
# 4) Scale factor -> 'length' params are labeled in real-world units,
# not px, and their entered value is converted back to px before
# being handed to generate().
# ------------------------------------------------------------------
win.pending_scale_factor = 0.5   # 1 px = 0.5 cm  (i.e. 2 px/cm)
win.pending_scale_unit = "cm"
captured = {}


def _capture_labels_exec(self):
    exec_calls.append(self.windowTitle())
    from PySide6.QtWidgets import QLabel
    captured["label_texts"] = [w.text() for w in self.findChildren(QLabel)]
    for btn in self.findChildren(QPushButton):
        if btn.objectName() == "accentBtn":
            btn.click()
            break
    return QDialog.Accepted


QDialog.exec = _capture_labels_exec
win._op = None
mb_calls.clear()
win.on_open_maze_template_dialog()
app.processEvents()
check("length params show the calibrated unit ('cm'), not px",
      any("(cm)" in t for t in captured.get("label_texts", [])))
check("op started successfully with scale factor active",
      win._op is not None and win._op["kind"] == "template_zones")

QDialog.exec = _orig_exec
win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
