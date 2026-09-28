"""
Regression test for "Live camera acquisition" (Upgrade Plan Tier 2 #6) --
adds a Camera video source (qt_app/dialogs/camera_dialog.py) alongside
file upload: a live preview with a real-time tracking-preview overlay,
and a Record button. Once recording stops, the saved file is probed and
appended to MainWindow's video queue exactly like Add Video(s) does
(see MainWindow.on_camera_recorded), so every existing feature (zones,
Subject Database, batch mode, Start Tracking, ...) works on it unchanged.

No real camera exists in a headless CI sandbox, so cv2.VideoCapture is
wrapped: an int argument (a camera index, exactly what the dialog passes)
goes to a small FakeCapture that mimics OpenCV's capture interface
(isOpened/get/read/release) and produces a moving synthetic blob; any
other argument (a file path, exactly what _probe_video/real playback use)
falls through to the REAL cv2.VideoCapture unchanged. So the actual
recording file written by this test is a genuine, unmocked .mp4 (real
cv2.VideoWriter), and probing/reading it back afterwards is fully real.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_camera_capture.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring/sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import tempfile

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from qt_app.dialogs import camera_dialog

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


class FakeCapture:
    """Mimics the tiny slice of cv2.VideoCapture's interface the dialog
    uses -- no real camera hardware needed. Produces a moving dark blob
    on a plain background, frame after frame, on demand (only when read()
    is called -- there's no background thread/real time involved)."""

    def __init__(self, index):
        self.index = index
        self._opened = True
        self._frame_num = 0
        self.width, self.height = 160, 120

    def isOpened(self):
        return self._opened

    def get(self, prop_id):
        if prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        return 0.0

    def read(self):
        if not self._opened:
            return False, None
        frame = np.full((self.height, self.width, 3), 200, dtype=np.uint8)
        x = 20 + (self._frame_num * 5) % (self.width - 40)
        cv2.circle(frame, (x, self.height // 2), 10, (40, 40, 40), -1)
        self._frame_num += 1
        return True, frame

    def release(self):
        self._opened = False


class DeadCapture(FakeCapture):
    """A camera index that fails to open at all (wrong index / already in
    use by something else)."""
    def isOpened(self):
        return False


_real_video_capture = cv2.VideoCapture


def _capture_router(source):
    # The dialog only ever calls cv2.VideoCapture(<int index>); everything
    # else (a file path, from _probe_video or this test's own verification
    # read) is real footage and must go through the real backend.
    if isinstance(source, int):
        return DeadCapture(source) if source == 99 else FakeCapture(source)
    return _real_video_capture(source)


camera_dialog.cv2.VideoCapture = _capture_router  # module-level cv2 is the same shared object

app = QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()

# ------------------------------------------------------------------
# 1) Opening a bad camera index shows a clear error, no crash, Record
# stays disabled.
# ------------------------------------------------------------------
dialog = camera_dialog.CameraDialog(win, on_recorded=win.on_camera_recorded, default_index=99)
mb_calls.clear()
dialog.open_camera()
check("opening a nonexistent camera index shows a critical dialog",
      any(k == "critical" for k, _a in mb_calls))
check("Record stays disabled when no camera is open", not dialog.record_btn.isEnabled())
check("no capture object retained after a failed open", dialog.cap is None)
dialog._teardown()

# ------------------------------------------------------------------
# 2) Opening a working camera starts the live preview + tracking overlay.
# The real QTimer is stopped right after open_camera() so every frame
# from here on is driven manually and deterministically.
# ------------------------------------------------------------------
dialog = camera_dialog.CameraDialog(win, on_recorded=win.on_camera_recorded, default_index=0)
dialog.open_camera()
check("Record becomes enabled once a camera opens", dialog.record_btn.isEnabled())
check("timer is running for the live preview", dialog.timer.isActive())
dialog.timer.stop()  # drive frames manually below -- deterministic frame counts

for _ in range(10):
    dialog.on_timer()
check("preview_label got a live frame (non-null pixmap)", not dialog.preview_label.pixmap().isNull())
check("last captured frame is stored", dialog._last_frame_bgr is not None)
check("not recording yet -- Start Recording is still the label", dialog.record_btn.text() == "Start Recording")

# ------------------------------------------------------------------
# 3) Start Recording -> N frames -> Stop Recording: a real .mp4 is
# written (via the REAL, unmocked cv2.VideoWriter) and handed to
# MainWindow.on_camera_recorded, which queues it exactly like Add
# Video(s) does.
# ------------------------------------------------------------------
rec_path = os.path.join(tempfile.gettempdir(), "camera_capture_test_recording.mp4")
if os.path.exists(rec_path):
    os.remove(rec_path)
QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (rec_path, ""))

win.videos = []
win.active_index = None
win.individual_radio.setChecked(True)
app.processEvents()

mb_calls.clear()
dialog.toggle_recording()
check("recording started (button now reads Stop Recording)", dialog.record_btn.text() == "Stop Recording")
check("camera controls lock while recording",
      not dialog.open_btn.isEnabled() and not dialog.index_spin.isEnabled())

N_FRAMES = 15
for _ in range(N_FRAMES):
    dialog.on_timer()
check(f"exactly {N_FRAMES} frames were written while recording", dialog.frames_written == N_FRAMES)

dialog.toggle_recording()
check("recording stopped (button back to Start Recording)", dialog.record_btn.text() == "Start Recording")
check("camera controls unlock again after stopping",
      dialog.open_btn.isEnabled() and dialog.index_spin.isEnabled())
check("recorded file exists on disk", os.path.exists(rec_path))
check("'Recording added' info dialog shown", any(k == "information" for k, _a in mb_calls))
check("no error dialogs while adding the recording", not any(k == "critical" for k, _a in mb_calls))

check("the recording was appended to MainWindow's video queue", len(win.videos) == 1)
if win.videos:
    check("the queued entry points at the recorded file", win.videos[0]["path"] == rec_path)
    check("the queued entry has a real, positive frame count (genuinely probed, not a stub)",
          win.videos[0]["frame_count"] > 0)
    check("the queued entry has a real, positive duration", win.videos[0]["duration"] > 0)

# Re-reading the file directly (fully independent of the app, real
# cv2.VideoCapture since the path isn't an int) confirms it's a genuine,
# playable video and not an empty/corrupt stub.
verify_cap = _real_video_capture(rec_path)
check("the saved recording is independently readable by OpenCV", verify_cap.isOpened())
ok, frame = verify_cap.read()
check("the saved recording's first frame reads back fine", ok and frame is not None)
verify_cap.release()

dialog._teardown()
check("teardown releases the capture", dialog.cap is None)
check("teardown stops the timer", not dialog.timer.isActive())

# ------------------------------------------------------------------
# 4) Individual mode with a video already queued -- adding another
# recording should warn instead of silently discarding the existing one.
# ------------------------------------------------------------------
mb_calls.clear()
win.on_camera_recorded(rec_path)
check("Individual mode with an existing video warns instead of adding a 2nd",
      any(k == "warning" for k, _a in mb_calls))
check("video queue still has exactly 1 entry", len(win.videos) == 1)

win.close()
if os.path.exists(rec_path):
    os.remove(rec_path)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
