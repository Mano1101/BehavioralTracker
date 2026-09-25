"""
Headless regression test for the Deep Learning Classifier panel (Task #27):
Prepare Training Data -> Train Model -> Start Tracking with 'Use trained
model', run genuinely end-to-end (real clip-dataset build, real 1-epoch
PyTorch training, real trained-model inference) rather than mocked, plus
the multi-animal confirmation guard. Requires TORCH_AVAILABLE in this
environment (confirmed True).

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_ml_pipeline.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import shutil
import sys, os

from PySide6.QtWidgets import (
    QApplication, QMessageBox, QDialog, QPushButton, QLineEdit,
)

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from _pathsetup import VIDEO
from tracking.location import compute_output_dir

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


_orig_exec = QDialog.exec

# All tests in this suite share the same dummy_behavior_test.mp4, and
# compute_output_dir() puts results next to the video (results/<name>/) --
# so an earlier test file's run (e.g. test_start_tracking.py's Behavior
# Classification step, or test_manual_scoring.py) can leave a real
# bouts.csv/manual_bouts.csv there. That would make section 1's "no
# labeled bouts found" guard check below wrongly find candidates, fall
# through the guard, and hit a REAL (not-yet-mocked) QDialog.exec() that
# blocks forever headless. Start from a clean slate so this test's result
# doesn't depend on what ran before it.
_out_dir = compute_output_dir(VIDEO)
if os.path.isdir(_out_dir):
    shutil.rmtree(_out_dir)
    os.makedirs(_out_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1) Guard checks (fresh window, nothing set up yet).
# ------------------------------------------------------------------
mb_calls.clear()
win.on_prepare_ml_dataset()
check("Prepare Training Data: no video -> critical", any(k == "critical" for k, _ in mb_calls))

add_video()
mb_calls.clear()
win.on_prepare_ml_dataset()
check("Prepare Training Data: no arena -> 'Arena not set' critical",
      any(k == "critical" and "Arena not set" in str(a) for k, a in mb_calls))

win._set_analysis_type("behavior")  # must precede on_tool_no_crop(): switching
                                     # analysis type resets calibration state
app.processEvents()
win.on_tool_no_crop()
win.setup_page.on_num_animals(1)
app.processEvents()

mb_calls.clear()
win.on_prepare_ml_dataset()
check("Prepare Training Data: arena set but no bouts file yet -> 'No labeled bouts found'",
      any(k == "critical" and "No labeled bouts found" in str(a) for k, a in mb_calls))

# ------------------------------------------------------------------
# 2) Run the automatic classifier once to produce a real bouts.csv
# (Prepare Training Data's own labeled-data source).
# ------------------------------------------------------------------
mb_calls.clear()
win.on_start()
app.processEvents()
check("automatic classifier run completed with no error dialogs",
      not any(k == "critical" for k, _ in mb_calls))
out_dir = win.results_page.output_dir
bouts_csv = os.path.join(out_dir, "bouts.csv")
check("bouts.csv exists for Prepare Training Data to use", os.path.exists(bouts_csv))

# ------------------------------------------------------------------
# 3) Prepare Training Data: build a real clip dataset from bouts.csv
# using the dialog's own default settings.
# ------------------------------------------------------------------
win.show_setup_page()
app.processEvents()


def _accept_defaults_exec(self):
    click_named(self, "accentBtn")  # Build Dataset
    return QDialog.Accepted


QDialog.exec = _accept_defaults_exec
mb_calls.clear()
win.on_prepare_ml_dataset()
app.processEvents()
QDialog.exec = _orig_exec

check("no error dialogs while building the dataset", not any(k == "critical" for k, _ in mb_calls))
dataset_dir = win._last_ml_dataset_dir
check("_last_ml_dataset_dir got set", bool(dataset_dir) and os.path.isdir(dataset_dir))

from tracking.ml_dataset import dataset_summary
summary = dataset_summary(dataset_dir)
check("dataset has a 'grooming' or 'rearing' class with clips",
      any(summary.get(k, 0) > 0 for k in ("grooming", "rearing")))
check("dataset has an 'other' (negative) class with clips", summary.get("other", 0) > 0)
check("ml_status_label reflects the built dataset",
      "Dataset:" in win.setup_page.ml_status_label.text())

# ------------------------------------------------------------------
# 4) Train Model: a tiny (1-epoch) real training run on that dataset.
# ------------------------------------------------------------------
def _train_one_epoch_exec(self):
    edits = self.findChildren(QLineEdit)
    # grid order: [0]=dataset dir, [1]=epochs, [2]=output path
    check("Train Model dialog pre-filled the dataset dir from Prepare Training Data",
          edits[0].text().strip() == dataset_dir)
    edits[1].setText("1")
    click_named(self, "accentBtn")  # Start Training
    return QDialog.Accepted


QDialog.exec = _train_one_epoch_exec
mb_calls.clear()
win.on_train_ml_model()
app.processEvents()
QDialog.exec = _orig_exec

check("no error dialogs while training", not any(k == "critical" for k, _ in mb_calls))
checkpoint_path = win.setup_page.ml_checkpoint_entry.text().strip()
check("ml_checkpoint_entry now points at a real .pt file",
      bool(checkpoint_path) and os.path.exists(checkpoint_path))
check("ml_status_label reports a trained model", "Trained model ready" in win.setup_page.ml_status_label.text())

# ------------------------------------------------------------------
# 5) Start Tracking with 'Use trained model' checked -> real inference
# through the trained checkpoint (_run_behavior_flow_ml).
# ------------------------------------------------------------------
win.on_reset_all()
app.processEvents()
win._set_analysis_type("behavior")
app.processEvents()
add_video()
win.on_tool_no_crop()
win.setup_page.on_num_animals(1)
app.processEvents()

win.setup_page.ml_mode_var.setChecked(True)
mb_calls.clear()
win.on_start()
app.processEvents()
check("'Use trained model' with no checkpoint set -> 'No trained model' critical",
      any(k == "critical" and "No trained model" in str(a) for k, a in mb_calls))

win.setup_page.ml_checkpoint_entry.setText(checkpoint_path)
mb_calls.clear()
win.on_start()
app.processEvents()
check("no critical error dialogs during ML inference run",
      not any(k == "critical" for k, _ in mb_calls))
check("stack switched to Results page after ML inference", win.stack.currentWidget() is win.results_page)
ml_out_dir = win.results_page.output_dir
ml_bouts_csv = os.path.join(ml_out_dir, "ml_bouts.csv")
check("ml_bouts.csv written", os.path.exists(ml_bouts_csv))

# ------------------------------------------------------------------
# 6) Multi-animal warning: >1 animal + ML mode asks for confirmation
# first (model sees the whole scene, not one mouse).
# ------------------------------------------------------------------
win.on_reset_all()
app.processEvents()
win._set_analysis_type("behavior")
app.processEvents()
add_video()
win.on_tool_no_crop()
win.setup_page.on_num_animals(2)
win.setup_page.ml_mode_var.setChecked(True)
win.setup_page.ml_checkpoint_entry.setText(checkpoint_path)
app.processEvents()

QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.No)
mb_calls.clear()
win.on_start()
app.processEvents()
check("2 animals + ML mode -> asks the single-animal-model question",
      any(k == "question" for k, _ in mb_calls))
check("answering No to that question cancels the run (stays on Setup page)",
      win.stack.currentWidget() is win.setup_page)

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
