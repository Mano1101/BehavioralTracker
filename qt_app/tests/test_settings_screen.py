"""
Regression test for "A real Settings screen" (Upgrade Plan Tier 2 #8) --
replaces MainWindow.on_open_settings's old "a dedicated Settings screen
is on the way" placeholder with one that actually persists Detection
Settings defaults and a couple of app-wide preferences (default units,
default color mode) to a small JSON file (qt_app/app_settings.py),
instead of qt_app/main_window.py's self._memory always starting
completely empty every launch.

Three things are covered:

1. app_settings.load_settings()/save_settings() -- pure file I/O against
   a temp path (SETTINGS_PATH is monkeypatched, never touches the real
   user's home directory): a normal round-trip, unknown keys filtered out
   on both save and load, and a missing/corrupt file quietly returning {}
   rather than raising.

2. The Settings dialog, driven through a real MainWindow: change a few
   fields and hit Save -> app._memory is updated immediately (so the
   CURRENT Setup page -- rebuilt right there -- reflects the change with
   no restart needed) and the same values are written to disk.

3. That persistence actually persists: a brand-new MainWindow (simulating
   the next launch) picks the saved values straight up as its starting
   _memory, and Reset All restores THOSE configured defaults rather than
   wiping back to a truly blank slate (see on_reset_all's own comment).

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_settings_screen.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring/sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import json
import shutil
import tempfile

from PySide6.QtWidgets import (
    QApplication, QDialog, QMessageBox, QLineEdit, QCheckBox, QComboBox, QPushButton,
)

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

import qt_app.app_settings as app_settings

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def click_named(dialog, object_name):
    for btn in dialog.findChildren(QPushButton):
        if btn.objectName() == object_name:
            btn.click()
            return True
    return False


# ------------------------------------------------------------------
# 1) load_settings()/save_settings() -- pure file I/O, no Qt/MainWindow.
# All against a throwaway temp path -- never the real user's home dir.
# ------------------------------------------------------------------
tmp_dir = tempfile.mkdtemp(prefix="bt_settings_test_")
app_settings.SETTINGS_PATH = os.path.join(tmp_dir, "nested", "settings.json")

check("no settings file yet -> load_settings() returns {}", app_settings.load_settings() == {})

app_settings.save_settings({"threshold_entry": "77", "units_entry": "mm", "not_a_real_key": "x"})
check("save_settings() creates the file (and its parent folder)",
      os.path.exists(app_settings.SETTINGS_PATH))
loaded = app_settings.load_settings()
check("save_settings() drops keys it doesn't recognize before writing",
      "not_a_real_key" not in json.load(open(app_settings.SETTINGS_PATH)))
check("load_settings() round-trips the recognized values back exactly",
      loaded.get("threshold_entry") == "77" and loaded.get("units_entry") == "mm")

with open(app_settings.SETTINGS_PATH, "w", encoding="utf-8") as f:
    json.dump({"threshold_entry": "50", "some_unknown_key": "y"}, f)
check("load_settings() filters out an unrecognized key found IN the file too",
      "some_unknown_key" not in app_settings.load_settings()
      and app_settings.load_settings().get("threshold_entry") == "50")

with open(app_settings.SETTINGS_PATH, "w", encoding="utf-8") as f:
    f.write("{not valid json")
check("a corrupt settings file -> load_settings() returns {} instead of raising",
      app_settings.load_settings() == {})

with open(app_settings.SETTINGS_PATH, "w", encoding="utf-8") as f:
    json.dump([1, 2, 3], f)
check("a settings file that isn't even a JSON object -> load_settings() returns {}",
      app_settings.load_settings() == {})

os.remove(app_settings.SETTINGS_PATH)

# ------------------------------------------------------------------
# 2) The Settings dialog, driven through a real MainWindow.
# ------------------------------------------------------------------
from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow

app = QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()

check("a fresh MainWindow with no settings file starts from the hardcoded fallback "
      "(threshold_entry not yet in _memory)", "threshold_entry" not in win._memory)
check("Setup page shows the hardcoded default threshold (25) before any Settings are saved",
      win.setup_page.threshold_entry.text() == "25")

_orig_exec = QDialog.exec


def _change_and_save_exec(self):
    edits = self.findChildren(QLineEdit)
    # Grid order matches settings_dialog.NUMERIC_FIELDS: [0]=bg_samples,
    # [1]=threshold, [2]=min_area, [3]=max_area, [4]=max_jump,
    # [5]=window_size, [6]=window_weight, [7]=real_distance, [8]=units,
    # [9]=preview_samples.
    check("dialog pre-fills the threshold field from the hardcoded fallback (25)",
          edits[1].text() == "25")
    edits[1].setText("77")
    edits[8].setText("mm")
    checks = self.findChildren(QCheckBox)
    checks[2].setChecked(True)  # "Reject shadows"
    combos = self.findChildren(QComboBox)
    combos[0].setCurrentIndex(1)  # "gray"
    check("Save Defaults button is present", click_named(self, "accentBtn"))
    return QDialog.Accepted


QDialog.exec = _change_and_save_exec
mb_calls.clear()
win.on_open_settings()
app.processEvents()
QDialog.exec = _orig_exec

check("no error dialogs while saving Settings", not any(k == "critical" for k, _a in mb_calls))
check("_memory reflects the new threshold immediately", win._memory.get("threshold_entry") == "77")
check("_memory reflects the new default units immediately", win._memory.get("units_entry") == "mm")
check("_memory reflects the new 'reject shadows' default immediately",
      win._memory.get("reject_shadows_var") is True)
check("_memory reflects the new default color mode immediately",
      win._memory.get("color_mode_var") == "gray")
check("the CURRENT Setup page was refreshed to match, with no restart needed",
      win.setup_page.threshold_entry.text() == "77")

check("Save Defaults wrote the change to disk", os.path.exists(app_settings.SETTINGS_PATH))
on_disk = json.load(open(app_settings.SETTINGS_PATH))
check("the persisted file has the new threshold", on_disk.get("threshold_entry") == "77")
check("the persisted file has the new default units", on_disk.get("units_entry") == "mm")
check("the persisted file has the new color mode", on_disk.get("color_mode_var") == "gray")

# ------------------------------------------------------------------
# 3) Persistence actually persists: a brand-new MainWindow picks the
# saved values up automatically, and Reset All keeps them (rather than
# reverting to a truly blank slate).
# ------------------------------------------------------------------
win2 = MainWindow()
win2.resize(1600, 980)
win2.show()
app.processEvents()
check("a brand-new MainWindow ('next launch') starts with the persisted threshold",
      win2._memory.get("threshold_entry") == "77")
check("...and its freshly-built Setup page shows it too, with no manual re-entry",
      win2.setup_page.threshold_entry.text() == "77")
check("...and the persisted default color mode too",
      win2.setup_page.get_color_mode() == "gray")
win2.close()

QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)
mb_calls.clear()
win.on_reset_all()
app.processEvents()
check("Reset All re-seeds from the configured Settings, not a truly blank slate",
      win._memory.get("threshold_entry") == "77")
check("...visible on the freshly-rebuilt Setup page too", win.setup_page.threshold_entry.text() == "77")

win.close()
shutil.rmtree(tmp_dir, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
