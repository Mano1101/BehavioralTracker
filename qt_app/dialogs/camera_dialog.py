"""
Camera acquisition dialog -- Upgrade Plan Tier 2 #6, "Live camera
acquisition": adds a live camera (cv2.VideoCapture(index)) as a video
source alongside file upload, matching EthoVision XT/SMART's own live
acquisition window.

Shows a live preview with a lightweight real-time tracking overlay
(background subtraction + largest-blob centroid -- a quick "is it seeing
my animal" check) while a Record button writes the raw, un-annotated
frames to an .mp4 file. Once recording stops, that file is probed and
appended to the SAME video queue Add Video(s) uses (MainWindow.on_camera_
recorded), so every existing feature -- zones, Subject Database, batch
mode, stop conditions, Multi-Mouse/Behavior analysis, the full report --
works on a live recording exactly like it does on a pre-recorded file.
The live overlay itself is never saved or scored; it's acquisition-time
visual feedback only, not a substitute for the real tracking pipeline
(which always runs afterwards, through the normal Start Tracking flow,
on the clean recorded file).

Kept dependency-light and synchronous, same as every other dialog in this
app (see ml_train_dialog.py): a QTimer parented to the dialog polls the
capture device roughly every 33ms (~30fps) and repaints the preview. A
QDialog's exec() runs its own local Qt event loop, so the QTimer still
fires normally while the dialog is open.
"""

import os
import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QMessageBox, QFileDialog,
)


def _frame_to_pixmap(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = np.ascontiguousarray(frame_rgb)
    h, w, _ = frame_rgb.shape
    qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class CameraDialog(QDialog):
    def __init__(self, parent, on_recorded=None, default_index=0):
        super().__init__(parent)
        self.setWindowTitle("Camera")
        self.on_recorded = on_recorded

        self.cap = None
        self.writer = None
        self.writer_path = None
        self.bg_subtractor = None
        self.frames_written = 0
        self._record_fps = 30.0
        self._last_frame_bgr = None

        layout = QVBoxLayout(self)

        open_row = QHBoxLayout()
        open_row.addWidget(QLabel("Camera index:"))
        self.index_spin = QSpinBox()
        self.index_spin.setRange(0, 99)
        self.index_spin.setValue(default_index)
        open_row.addWidget(self.index_spin)
        self.open_btn = QPushButton("Open camera")
        self.open_btn.setObjectName("accentBtn")
        self.open_btn.clicked.connect(self.open_camera)
        open_row.addWidget(self.open_btn)
        open_row.addStretch()
        layout.addLayout(open_row)

        self.preview_label = QLabel("Open a camera to see its live preview.")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(480, 360)
        self.preview_label.setStyleSheet("background: #111111; color: #999999; border-radius: 6px;")
        layout.addWidget(self.preview_label)

        self.status_label = QLabel("No camera open yet.")
        self.status_label.setStyleSheet("color: #666666; font-size: 9.5px;")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        self.record_btn = QPushButton("Start Recording")
        self.record_btn.setObjectName("accentBtn")
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self.toggle_recording)
        btn_row.addWidget(self.record_btn)
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.on_timer)

    # ------------------------------------------------------------------

    def open_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        index = self.index_spin.value()
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            QMessageBox.critical(
                self, "Camera not found",
                f"Could not open camera index {index}. Check it's connected, powered on, "
                "and not already in use by another application."
            )
            return
        self.cap = cap
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=120, varThreshold=25, detectShadows=False)
        self.record_btn.setEnabled(True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.status_label.setText(
            f"Camera {index} open -- {w}x{h}" + (f" @ {fps:.0f}fps" if fps else "")
            + ". Live preview running; the green marker is a quick tracking preview only."
        )
        self.timer.start()

    def on_timer(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status_label.setText("Lost the camera feed -- check the connection.")
            return
        self._last_frame_bgr = frame

        display = frame.copy()
        if self.bg_subtractor is not None:
            mask = self.bg_subtractor.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) >= 30:
                    (cx, cy), radius = cv2.minEnclosingCircle(largest)
                    cv2.circle(display, (int(cx), int(cy)), max(int(radius), 4), (0, 255, 0), 2)
                    cv2.drawMarker(display, (int(cx), int(cy)), (0, 255, 0),
                                   markerType=cv2.MARKER_CROSS, markerSize=10)

        if self.writer is not None:
            self.writer.write(frame)
            self.frames_written += 1
            cv2.circle(display, (18, 18), 8, (0, 0, 255), -1)  # REC dot -- preview only

        self.preview_label.setPixmap(
            _frame_to_pixmap(display).scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if self.writer is not None:
            elapsed = self.frames_written / max(self._record_fps, 1.0)
            self.status_label.setText(
                f"Recording... {self.frames_written} frame(s), {elapsed:.1f}s "
                f"-> {os.path.basename(self.writer_path)}")

    def toggle_recording(self):
        if self.writer is None:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        default_name = f"camera_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        path, _ = QFileDialog.getSaveFileName(self, "Save recording as", default_name,
                                               "MP4 video (*.mp4)")
        if not path:
            return
        if not path.lower().endswith(".mp4"):
            path += ".mp4"

        fps = self.cap.get(cv2.CAP_PROP_FPS) or 0
        self._record_fps = fps if fps and fps > 1 else 30.0
        fallback_h, fallback_w = (
            self._last_frame_bgr.shape[:2] if self._last_frame_bgr is not None else (480, 640))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or fallback_w)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or fallback_h)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), self._record_fps, (w, h))
        if not writer.isOpened():
            QMessageBox.critical(self, "Could not start recording",
                                  f"Failed to open a video file for writing at:\n{path}")
            return
        self.writer = writer
        self.writer_path = path
        self.frames_written = 0
        self.record_btn.setText("Stop Recording")
        self.index_spin.setEnabled(False)
        self.open_btn.setEnabled(False)

    def _stop_recording(self):
        if self.writer is not None:
            self.writer.release()
        path = self.writer_path
        frames = self.frames_written
        self.writer = None
        self.writer_path = None
        self.frames_written = 0
        self.record_btn.setText("Start Recording")
        self.index_spin.setEnabled(True)
        self.open_btn.setEnabled(True)

        if path and frames > 0:
            self.status_label.setText(f"Saved {frames} frame(s) -> {path}")
            if self.on_recorded:
                self.on_recorded(path)
        elif path:
            self.status_label.setText("No frames were captured -- nothing saved.")
            try:
                os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------

    def closeEvent(self, event):
        self._teardown()
        super().closeEvent(event)

    def reject(self):
        self._teardown()
        super().reject()

    def _teardown(self):
        if self.writer is not None:
            self._stop_recording()
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def open_camera_dialog(app, parent):
    dialog = CameraDialog(parent, on_recorded=app.on_camera_recorded)
    dialog.exec()
