"""
Train Model dialog -- Qt port of TrackerApp._on_train_ml_model in
gui/main_window.py. Trains a small deep-learning clip classifier
(tracking/ml_train.py's train_model) on a dataset folder built by Prepare
Training Data, and saves a checkpoint (.pt) that can then be picked with
'Browse...' + 'Use trained model' back on the Setup page's Deep Learning
Classifier panel.

Kept synchronous (no QThread), same as every other long-running operation
in this app (Start Tracking, etc.): train_model()'s progress_callback
calls QApplication.processEvents() after every epoch to keep the dialog
responsive during the (possibly multi-minute, CPU-bound) training loop.
"""

import os

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QFileDialog, QProgressBar, QWidget, QApplication,
)

from tracking.ml_train import TORCH_AVAILABLE, train_model
from tracking.ml_dataset import dataset_summary


def open_train_ml_model_dialog(app, parent):
    if not TORCH_AVAILABLE:
        QMessageBox.critical(
            parent, "PyTorch not installed",
            "Training needs PyTorch. Install it (see pytorch.org for the right command for "
            "your machine/GPU), then try again."
        )
        return

    dataset_dir_default = getattr(app, "_last_ml_dataset_dir", "") or app._default_ml_dataset_dir()

    dialog = QDialog(parent)
    dialog.setWindowTitle("Train Model")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)

    grid = QGridLayout()
    layout.addLayout(grid)

    grid.addWidget(QLabel("Dataset folder:"), 0, 0)
    ds_row = QHBoxLayout()
    ds_entry = QLineEdit(dataset_dir_default)
    ds_row.addWidget(ds_entry)
    browse_ds_btn = QPushButton("...")
    browse_ds_btn.setFixedWidth(30)
    ds_row.addWidget(browse_ds_btn)
    ds_widget = QWidget()
    ds_widget.setLayout(ds_row)
    grid.addWidget(ds_widget, 0, 1)

    summary_label = QLabel("")
    summary_label.setWordWrap(True)
    summary_label.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(summary_label)

    def refresh_summary():
        d = ds_entry.text().strip()
        summary = dataset_summary(d) if d else {}
        if summary:
            lines = ", ".join(f"{k}: {v}" for k, v in summary.items())
            summary_label.setText(f"Found -- {lines}")
        else:
            summary_label.setText("No clips found in that folder yet -- use 'Prepare Training "
                                   "Data' first.")
        return summary

    def browse_ds():
        path = QFileDialog.getExistingDirectory(dialog, "Choose a dataset folder", ds_entry.text() or ".")
        if path:
            ds_entry.setText(path)
            refresh_summary()

    browse_ds_btn.clicked.connect(browse_ds)
    refresh_summary()

    grid.addWidget(QLabel("Training passes (epochs):"), 1, 0)
    epochs_entry = QLineEdit("15")
    epochs_entry.setFixedWidth(80)
    grid.addWidget(epochs_entry, 1, 1)

    grid.addWidget(QLabel("Save trained model as:"), 2, 0)
    out_row = QHBoxLayout()
    default_out = os.path.join(dataset_dir_default, "model.pt") if dataset_dir_default else ""
    out_entry = QLineEdit(default_out)
    out_row.addWidget(out_entry)
    browse_out_btn = QPushButton("...")
    browse_out_btn.setFixedWidth(30)
    out_row.addWidget(browse_out_btn)
    out_widget = QWidget()
    out_widget.setLayout(out_row)
    grid.addWidget(out_widget, 2, 1)

    def browse_out():
        path, _ = QFileDialog.getSaveFileName(dialog, "Save trained model as", "",
                                               "PyTorch checkpoint (*.pt)")
        if path:
            if not path.endswith(".pt"):
                path += ".pt"
            out_entry.setText(path)

    browse_out_btn.clicked.connect(browse_out)

    hint = QLabel("More epochs and more labeled clips both help, but each epoch takes longer "
                  "without a GPU -- this runs on the CPU here, which is much slower than a GPU. "
                  "A small first run is a good way to check everything works.")
    hint.setWordWrap(True)
    hint.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(hint)

    progress = QProgressBar()
    progress.setRange(0, 100)
    layout.addWidget(progress)
    status_label = QLabel("")
    status_label.setWordWrap(True)
    layout.addWidget(status_label)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    close_btn = QPushButton("Close")
    train_btn = QPushButton("Start Training")
    train_btn.setObjectName("accentBtn")
    btn_row.addWidget(close_btn)
    btn_row.addWidget(train_btn)
    layout.addLayout(btn_row)
    close_btn.clicked.connect(dialog.reject)

    def do_train():
        dataset_dir = ds_entry.text().strip()
        summary = refresh_summary()
        if not summary:
            QMessageBox.critical(dialog, "No training data", "That folder has no clips yet.")
            return
        class_names = sorted(summary.keys(), key=lambda k: (k == "other", k))
        try:
            epochs = max(1, int(float(epochs_entry.text())))
        except ValueError:
            QMessageBox.critical(dialog, "Invalid value", "Training passes must be a whole number.")
            return
        output_path = out_entry.text().strip()
        if not output_path:
            QMessageBox.critical(dialog, "No save location", "Choose where to save the trained model.")
            return

        train_btn.setEnabled(False)

        def on_epoch(epoch, total_epochs, train_loss, val_loss, val_acc):
            progress.setValue(int(100 * epoch / total_epochs))
            status_label.setText(f"Epoch {epoch}/{total_epochs} -- train loss {train_loss:.3f}, "
                                  f"val loss {val_loss:.3f}, val accuracy {val_acc * 100:.0f}%")
            QApplication.processEvents()

        try:
            _out_path, best_val_acc = train_model(
                dataset_dir, class_names, output_path,
                epochs=epochs, progress_callback=on_epoch,
            )
        except Exception as exc:
            status_label.setText("")
            QMessageBox.critical(dialog, "Training failed", str(exc))
            train_btn.setEnabled(True)
            return

        status_label.setText(f"Done. Best validation accuracy: {best_val_acc * 100:.0f}%. "
                              f"Saved to {output_path}")
        sp = app.setup_page
        sp.ml_checkpoint_entry.setText(output_path)
        sp.ml_status_label.setText(
            f"Trained model ready: {os.path.basename(output_path)} "
            f"(val accuracy {best_val_acc * 100:.0f}%). Check 'Use trained model' above to use it."
        )
        train_btn.setEnabled(True)

    train_btn.clicked.connect(do_train)
    dialog.exec()
