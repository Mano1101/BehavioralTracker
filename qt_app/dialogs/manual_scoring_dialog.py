"""
Manual Behavior Scoring window -- the Qt-native replacement for
tracking.behavior.manual_score_video()'s interactive cv2.imshow/waitKey
keyboard-driven loop (gui/main_window.py's on_manual_behavior_scoring /
_run_behavior_manual_flow). That OpenCV window can't be reused here: it's a
blocking native highgui loop, and this environment's OpenCV is built
against Qt5 while this app runs on Qt6 -- running both in one process
risks the same hard crash documented next to the show_display=False fixes
in MainWindow's Start Tracking flows. So this reimplements the whole
interaction natively in Qt instead: a QTimer drives playback (in place of
a blocking cv2.waitKey loop -- and unlike Tkinter, Qt's event loop makes
this the natural way to do it, not a workaround), and keyPressEvent
mirrors manual_score_video()'s own key-to-action mapping one for one, so
the muscle memory described in its docstring still applies here:

    SPACE       play / pause (starts paused)
    , / .       step one frame back / forward (while paused)
    [ / ]       slower / faster playback
    1-4         toggle GROOMING / REARING / LOCOMOTION / IMMOBILE for the
                active subject
    Tab         switch the active subject (only matters with >1 animal)
    U           undo the last completed bout
    Q / Esc / closing the window   finish and save -- there is no separate
                "cancel without saving" here either, matching the
                original: any way of closing this window closes any open
                bouts and keeps what was scored so far.

On finish, self.bouts_df / self.duration_s / self.fps are populated with
the same bout-row shape classify_behaviors() produces (source="Manual"),
for the caller (MainWindow._run_behavior_manual_flow) to save to CSV and
hand to the Results page exactly like the automatic classifier's output --
this dialog itself does no file I/O.
"""

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy,
)

from qt_app.widgets.results_charts import BEHAVIOR_COLORS

SPEEDS = [0.25, 0.5, 1.0, 2.0, 4.0]
BEHAVIOR_KEYS = [
    (Qt.Key_1, "grooming"),
    (Qt.Key_2, "rearing"),
    (Qt.Key_3, "locomotion"),
    (Qt.Key_4, "immobile"),
]
_BOUT_COLUMNS = ["subject", "behavior", "start_frame", "stop_frame",
                 "start_s", "stop_s", "duration_s", "confidence", "source"]


class ManualScoringDialog(QDialog):

    def __init__(self, parent, video_path, subject_names):
        super().__init__(parent)
        self.setWindowTitle("Manual Behavior Scoring")
        self.setModal(True)
        self.resize(900, 680)

        self.video_path = video_path
        self.subject_names = list(subject_names) if subject_names else ["mouse_A"]

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n_frames = n if n > 0 else None

        self.cur_idx = -1
        self.frame = None
        self.playing = False
        self.speed_idx = 2  # 1.0x

        self.open_bouts = {s: None for s in self.subject_names}
        self.completed = []
        self.active_subject = self.subject_names[0]

        self.bouts_df = None
        self.duration_s = 0.0
        self._finished = False

        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._on_tick)

        if not self._read_next():
            self.cap.release()
            raise RuntimeError("Could not read any frames from the video.")
        self._refresh()

    # ------------------------------------------------------------------
    # Video I/O (mirrors manual_score_video()'s read_next/seek_to)
    # ------------------------------------------------------------------

    def _time_s(self, idx):
        return round(idx / self.fps, 3) if idx is not None and idx >= 0 else 0.0

    def _read_next(self):
        ok, f = self.cap.read()
        if ok:
            self.cur_idx += 1
            self.frame = f
        return ok

    def _seek_to(self, idx):
        idx = max(0, idx if self.n_frames is None else min(idx, self.n_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = self.cap.read()
        if ok:
            self.cur_idx = idx
            self.frame = f
        return ok

    # ------------------------------------------------------------------
    # Bout bookkeeping (mirrors close_bout/open_bout/toggle)
    # ------------------------------------------------------------------

    def _close_bout(self, subject, stop_idx):
        b = self.open_bouts.get(subject)
        if b is None:
            return
        stop_time = self._time_s(stop_idx)
        self.completed.append(dict(
            subject=subject, behavior=b["behavior"],
            start_frame=b["start_frame"], stop_frame=stop_idx,
            start_s=b["start_s"], stop_s=stop_time,
            duration_s=round(stop_time - b["start_s"], 3),
            confidence=100.0, source="Manual",
        ))
        self.open_bouts[subject] = None

    def _open_bout(self, subject, behavior, start_idx):
        self.open_bouts[subject] = dict(
            behavior=behavior, start_frame=start_idx, start_s=self._time_s(start_idx))

    def toggle_behavior(self, behavior):
        subject = self.active_subject
        b = self.open_bouts.get(subject)
        if b is not None and b["behavior"] == behavior:
            self._close_bout(subject, self.cur_idx)
        else:
            if b is not None:
                self._close_bout(subject, self.cur_idx)
            self._open_bout(subject, behavior, self.cur_idx)
        self._refresh()

    def undo(self):
        if self.completed:
            self.completed.pop()
        self._refresh()

    def switch_subject(self):
        if len(self.subject_names) <= 1:
            return
        i = self.subject_names.index(self.active_subject)
        self.active_subject = self.subject_names[(i + 1) % len(self.subject_names)]
        self._refresh()

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def toggle_play(self):
        self.playing = not self.playing
        self._sync_timer()
        self._refresh()

    def _sync_timer(self):
        if self.playing:
            base_delay = max(1, int(1000 / self.fps))
            delay = max(1, int(base_delay / SPEEDS[self.speed_idx]))
            self.timer.start(delay)
        else:
            self.timer.stop()

    def _on_tick(self):
        if not self._read_next():
            # reached the end -- stay open so trailing bouts can still be
            # closed by hand, same as manual_score_video()
            self.playing = False
            self._sync_timer()
        self._refresh()

    def step(self, delta):
        self.playing = False
        self._sync_timer()
        if delta < 0:
            self._seek_to(self.cur_idx - 1)
        else:
            self._read_next()
        self._refresh()

    def change_speed(self, delta):
        self.speed_idx = max(0, min(len(SPEEDS) - 1, self.speed_idx + delta))
        if self.playing:
            self._sync_timer()
        self._refresh()

    # ------------------------------------------------------------------
    # Finish (mirrors the tail of manual_score_video(): close any bouts
    # still open, build the DataFrame, release the capture)
    # ------------------------------------------------------------------

    def finish(self):
        if self._finished:
            return
        self.playing = False
        self.timer.stop()
        for subj in self.subject_names:
            if self.open_bouts.get(subj) is not None:
                self._close_bout(subj, self.cur_idx)
        if not self.completed:
            self.bouts_df = pd.DataFrame(columns=_BOUT_COLUMNS)
        else:
            self.bouts_df = pd.DataFrame(self.completed)[_BOUT_COLUMNS].sort_values(
                ["start_s", "subject"]).reset_index(drop=True)
        self.duration_s = (self.n_frames / self.fps) if self.n_frames else self._time_s(self.cur_idx)
        self.cap.release()
        self._finished = True
        self.accept()

    # ------------------------------------------------------------------
    # Qt event plumbing
    # ------------------------------------------------------------------

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Comma:
            self.step(-1)
        elif key == Qt.Key_Period:
            self.step(1)
        elif key == Qt.Key_BracketLeft:
            self.change_speed(-1)
        elif key == Qt.Key_BracketRight:
            self.change_speed(1)
        elif key == Qt.Key_Tab:
            self.switch_subject()
        elif key == Qt.Key_U:
            self.undo()
        elif key in (Qt.Key_Q, Qt.Key_Escape):
            self.finish()
        else:
            for qkey, behavior in BEHAVIOR_KEYS:
                if key == qkey:
                    self.toggle_behavior(behavior)
                    return
            super().keyPressEvent(event)

    def closeEvent(self, event):
        # No separate "cancel" concept, matching manual_score_video(): any
        # way of closing this window (the titlebar X included) finishes
        # and keeps whatever was scored so far, rather than discarding it.
        self.finish()
        event.accept()

    def focusNextPrevChild(self, next):
        # Qt's default QWidget.event() intercepts a plain Tab/Shift+Tab
        # keypress for focus-chain navigation BEFORE it ever reaches
        # keyPressEvent, whenever the dialog has more than one focusable
        # child (it does: seven transport/behavior buttons). Left alone,
        # that silently swallows the documented "Tab = switch active
        # subject" shortcut -- caught by a real QTest.keyClick(Key_Tab)
        # end-to-end test, not by calling switch_subject() directly.
        # Returning False here tells Qt not to move focus for Tab/
        # Shift+Tab at all, so the key event falls through to
        # keyPressEvent instead, same as every other shortcut key.
        return False

    def reject(self):
        # QDialog's own Esc-closes-as-Rejected wiring is overridden by
        # keyPressEvent above, but reject() can still be reached
        # programmatically -- route it through the same finish() so it
        # can never bypass saving/releasing the capture.
        self.finish()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(420)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet("background: #141a22; border-radius: 6px; color: #888;")
        root.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel()
        self.status_label.setStyleSheet("font-size: 10.5px; font-weight: 700; color: #1a1a1a;")
        root.addWidget(self.status_label)

        # Per-subject bout status
        subj_box = QFrame()
        subj_box.setStyleSheet("background: white; border: 1px solid #c9c9c9; border-radius: 6px;")
        subj_layout = QVBoxLayout(subj_box)
        subj_layout.setContentsMargins(10, 8, 10, 8)
        self.subject_labels = {}
        for subj in self.subject_names:
            lbl = QLabel()
            lbl.setStyleSheet("font-size: 10.5px;")
            subj_layout.addWidget(lbl)
            self.subject_labels[subj] = lbl
        root.addWidget(subj_box)

        # Behavior toggle buttons
        beh_row = QHBoxLayout()
        self.behavior_buttons = {}
        for i, (qkey, behavior) in enumerate(BEHAVIOR_KEYS, start=1):
            btn = QPushButton(f"{i}  {behavior.title()}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFocusPolicy(Qt.NoFocus)  # this is a keyboard-first tool; see focusNextPrevChild
            btn.clicked.connect(lambda _checked, b=behavior: self.toggle_behavior(b))
            beh_row.addWidget(btn)
            self.behavior_buttons[behavior] = btn
        root.addLayout(beh_row)

        # Transport controls
        transport = QHBoxLayout()
        self.step_back_btn = QPushButton(",  Step back")
        self.play_btn = QPushButton("Space  Play")
        self.step_fwd_btn = QPushButton(".  Step fwd")
        self.speed_down_btn = QPushButton("[")
        self.speed_up_btn = QPushButton("]")
        self.subject_btn = QPushButton("Tab  Switch subject")
        self.undo_btn = QPushButton("U  Undo")
        for b in (self.step_back_btn, self.play_btn, self.step_fwd_btn,
                  self.speed_down_btn, self.speed_up_btn, self.subject_btn, self.undo_btn):
            b.setObjectName("toolBtn")
            b.setCursor(Qt.PointingHandCursor)
            b.setFocusPolicy(Qt.NoFocus)  # this is a keyboard-first tool; see focusNextPrevChild
            transport.addWidget(b)
        transport.addStretch()
        root.addLayout(transport)

        self.step_back_btn.clicked.connect(lambda: self.step(-1))
        self.step_fwd_btn.clicked.connect(lambda: self.step(1))
        self.play_btn.clicked.connect(self.toggle_play)
        self.speed_down_btn.clicked.connect(lambda: self.change_speed(-1))
        self.speed_up_btn.clicked.connect(lambda: self.change_speed(1))
        self.subject_btn.clicked.connect(self.switch_subject)
        self.undo_btn.clicked.connect(self.undo)
        self.subject_btn.setVisible(len(self.subject_names) > 1)

        # Bottom bar: bout counter + Finish & Save
        bottom = QHBoxLayout()
        self.bouts_count_label = QLabel()
        self.bouts_count_label.setStyleSheet("font-size: 10.5px; color: #4b5563;")
        bottom.addWidget(self.bouts_count_label)
        bottom.addStretch()
        finish_btn = QPushButton("Q / Esc  Finish && Save")
        finish_btn.setObjectName("accentBtn")
        finish_btn.setCursor(Qt.PointingHandCursor)
        finish_btn.setFocusPolicy(Qt.NoFocus)  # this is a keyboard-first tool; see focusNextPrevChild
        finish_btn.clicked.connect(self.finish)
        bottom.addWidget(finish_btn)
        root.addLayout(bottom)

    def _refresh(self):
        self._refresh_video()
        self._refresh_status()
        self._refresh_subjects()
        self._refresh_behavior_buttons()
        self.play_btn.setText("Space  Pause" if self.playing else "Space  Play")
        self.bouts_count_label.setText(f"Bouts saved: {len(self.completed)}")

    def _refresh_video(self):
        if self.frame is None:
            return
        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        h, w, _ = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        target = self.video_label.size()
        if target.width() > 10 and target.height() > 10:
            pix = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pix)

    def _refresh_status(self):
        status = "PLAYING" if self.playing else "PAUSED"
        total = f"/{self.n_frames}" if self.n_frames else ""
        self.status_label.setText(
            f"{status}   frame {self.cur_idx}{total}   t={self._time_s(self.cur_idx):.2f}s   "
            f"speed={SPEEDS[self.speed_idx]:.2f}x"
        )

    def _refresh_subjects(self):
        for subj in self.subject_names:
            lbl = self.subject_labels[subj]
            b = self.open_bouts.get(subj)
            marker = "▶ " if subj == self.active_subject else "   "
            if b is not None:
                text = f"{marker}{subj}:  {b['behavior'].upper()} (open)"
                color = BEHAVIOR_COLORS.get(b["behavior"], "#1a1a1a")
            else:
                text = f"{marker}{subj}:  --"
                color = "#888888"
            weight = "700" if subj == self.active_subject else "500"
            lbl.setStyleSheet(f"font-size: 10.5px; font-weight: {weight}; color: {color};")
            lbl.setText(text)

    def _refresh_behavior_buttons(self):
        b = self.open_bouts.get(self.active_subject)
        active_behavior = b["behavior"] if b is not None else None
        for behavior, btn in self.behavior_buttons.items():
            color = BEHAVIOR_COLORS.get(behavior, "#888888")
            if behavior == active_behavior:
                btn.setStyleSheet(
                    f"background: {color}; color: white; font-weight: 700; "
                    f"border: 2px solid #1a1a1a; border-radius: 6px; padding: 7px 12px;"
                )
            else:
                btn.setStyleSheet(
                    f"background: white; color: {color}; font-weight: 700; "
                    f"border: 1px solid {color}; border-radius: 6px; padding: 7px 12px;"
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_video()
