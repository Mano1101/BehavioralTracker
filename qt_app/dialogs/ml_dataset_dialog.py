"""
Prepare Training Data dialog -- Qt port of TrackerApp._on_prepare_ml_dataset
in gui/main_window.py. Slices an already-labeled video (Behavior
Classification's automatic bouts.csv, or Manual Scoring's manual_bouts.csv)
into short training clips for the optional deep-learning classifier
(tracking/ml_dataset.py's build_clip_dataset), one folder per behavior plus
an automatically-sampled "other" (negative) class. Needs no PyTorch --
tracking.ml_dataset has no torch dependency at all; only actually training
(ml_train_dialog.py) or using a trained model (MainWindow._run_behavior_flow_ml)
does.
"""

import os

import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox, QCheckBox,
    QLineEdit, QPushButton, QMessageBox, QFileDialog, QWidget, QApplication,
)

from tracking.location import compute_output_dir
from tracking.ml_dataset import build_clip_dataset, dataset_summary


def open_prepare_ml_dataset_dialog(app, parent):
    if not app.videos:
        QMessageBox.critical(parent, "No video", "Add a video first.")
        return
    if app.active_index is None:
        QMessageBox.critical(parent, "No video selected", "Click a video in the list to select it.")
        return
    if app.pending_matrix is None:
        QMessageBox.critical(parent, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
        return

    active_path = app.videos[app.active_index]["path"]
    output_dir = compute_output_dir(active_path)
    candidates = []
    for fname, src_label in [("bouts.csv", "Automatic classification"), ("manual_bouts.csv", "Manual scoring")]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            candidates.append((f"{src_label} ({fname})", fpath))
    if not candidates:
        QMessageBox.critical(
            parent, "No labeled bouts found",
            "Run 'Start Tracking' (Behavior Classification) or 'Manual Scoring' on this video "
            "first -- training data comes from a bouts file those produce."
        )
        return

    cap = cv2.VideoCapture(active_path)
    fps_guess = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    dialog = QDialog(parent)
    dialog.setWindowTitle("Prepare Training Data")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)

    grid = QGridLayout()
    layout.addLayout(grid)

    grid.addWidget(QLabel("Bouts source:"), 0, 0)
    bouts_labels = [c[0] for c in candidates]
    bouts_combo = QComboBox()
    bouts_combo.addItems(bouts_labels)
    grid.addWidget(bouts_combo, 0, 1)

    grid.addWidget(QLabel("Include behaviors:"), 1, 0)
    beh_row = QHBoxLayout()
    groom_check = QCheckBox("Grooming")
    groom_check.setChecked(True)
    rear_check = QCheckBox("Rearing")
    rear_check.setChecked(True)
    beh_row.addWidget(groom_check)
    beh_row.addWidget(rear_check)
    beh_widget = QWidget()
    beh_widget.setLayout(beh_row)
    grid.addWidget(beh_widget, 1, 1)

    grid.addWidget(QLabel("Clip length (s):"), 2, 0)
    clip_len_entry = QLineEdit("1.5")
    clip_len_entry.setFixedWidth(80)
    grid.addWidget(clip_len_entry, 2, 1)

    grid.addWidget(QLabel("Dataset folder:"), 3, 0)
    ds_row = QHBoxLayout()
    default_ds_dir = os.path.join(os.path.dirname(active_path), "ml_dataset")
    ds_entry = QLineEdit(default_ds_dir)
    ds_row.addWidget(ds_entry)
    browse_ds_btn = QPushButton("...")
    browse_ds_btn.setFixedWidth(30)
    ds_row.addWidget(browse_ds_btn)
    ds_widget = QWidget()
    ds_widget.setLayout(ds_row)
    grid.addWidget(ds_widget, 3, 1)

    def browse_ds():
        path = QFileDialog.getExistingDirectory(dialog, "Choose (or create) a dataset folder")
        if path:
            ds_entry.setText(path)

    browse_ds_btn.clicked.connect(browse_ds)

    hint = QLabel("Re-running this on more videos adds to the same folder -- point every video "
                  "at the SAME dataset folder as you label more footage. Keep the Crop/Time-window "
                  "settings the same as when you made the bouts file.")
    hint.setWordWrap(True)
    hint.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(hint)

    status_label = QLabel("")
    status_label.setWordWrap(True)
    layout.addWidget(status_label)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    close_btn = QPushButton("Close")
    build_btn = QPushButton("Build Dataset")
    build_btn.setObjectName("accentBtn")
    btn_row.addWidget(close_btn)
    btn_row.addWidget(build_btn)
    layout.addLayout(btn_row)
    close_btn.clicked.connect(dialog.reject)

    candidates_map = dict(candidates)

    def do_build():
        behaviors = [b for b, checked in [("grooming", groom_check.isChecked()),
                                           ("rearing", rear_check.isChecked())] if checked]
        if not behaviors:
            QMessageBox.critical(dialog, "Nothing selected", "Choose at least one behavior.")
            return
        try:
            clip_len_s = float(clip_len_entry.text())
            window_frames = max(1, int(round(clip_len_s * fps_guess)))
        except ValueError:
            QMessageBox.critical(dialog, "Invalid value", "Clip length must be a number.")
            return
        dataset_dir = ds_entry.text().strip()
        if not dataset_dir:
            QMessageBox.critical(dialog, "No dataset folder", "Choose a dataset folder.")
            return
        bouts_csv = candidates_map[bouts_combo.currentText()]

        build_btn.setEnabled(False)
        status_label.setText("Preparing video and slicing clips...")
        QApplication.processEvents()

        tmp_path = None
        try:
            source_path, tmp_path, _fps = app._prepare_source_video(active_path)
            counts = build_clip_dataset(
                source_path, bouts_csv, dataset_dir, behaviors=behaviors,
                window_frames=window_frames, stride_frames=max(1, window_frames // 2),
            )
        except Exception as exc:
            status_label.setText("")
            QMessageBox.critical(dialog, "Could not build dataset", str(exc))
            build_btn.setEnabled(True)
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        app._last_ml_dataset_dir = dataset_dir
        summary = dataset_summary(dataset_dir)
        lines = [f"  {cls}: {n} clips" for cls, n in summary.items()]
        status_label.setText(
            "Added " + ", ".join(f"{k}: {v}" for k, v in counts.items()) +
            ".\n\nDataset now has:\n" + "\n".join(lines)
        )
        app.setup_page.ml_status_label.setText(f"Dataset: {dataset_dir}\n" + "\n".join(lines))
        build_btn.setEnabled(True)

    build_btn.clicked.connect(do_build)
    dialog.exec()
