"""
Maze Template dialog -- the Qt equivalent of TrackerApp._open_maze_template_dialog
/ _maze_dialog_generate in gui/main_window.py. Lets the researcher pick one of the
built-in maze/arena templates (Elevated Plus Maze, Y-Maze, T-Maze, Open Field,
3-Chamber Social Test, ...) and tune its real-world dimensions, then generates
the zone polygons and hands off to the SAME interactive template_zones op that
Quick Setup uses (MainWindow.start_template_zone_op) -- so the researcher still
gets to click each generated zone on the canvas to confirm/rename it (e.g. swap
which arm is "Open") before Finish commits it.

Unlike the zone-label dialog, this one doesn't return a value to its caller --
it calls start_template_zone_op(...) itself once "Generate Zones" succeeds, then
closes. That mirrors the Tkinter version, which calls
self._start_template_zone_op(...) directly from _maze_dialog_generate.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QComboBox,
    QLineEdit, QPushButton, QMessageBox, QWidget,
)

from tracking.maze_templates import TEMPLATES as MAZE_TEMPLATES


def open_maze_template_dialog(app, parent):
    """app: the MainWindow (for pending_matrix/pending_warp_w/h/pending_scale_*
    and start_template_zone_op). parent: the widget to parent the QDialog to."""
    if app.pending_matrix is None:
        QMessageBox.critical(parent, "Arena needed", "Use 'Crop Arena' or 'No Crop' first.")
        return

    dialog = QDialog(parent)
    dialog.setWindowTitle("Maze Template")
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)

    top_row = QHBoxLayout()
    top_row.addWidget(QLabel("Template:"))
    key_list = list(MAZE_TEMPLATES.keys())
    label_list = [MAZE_TEMPLATES[k]["label"] for k in key_list]
    combo = QComboBox()
    combo.addItems(label_list)
    top_row.addWidget(combo)
    top_row.addStretch()
    layout.addLayout(top_row)

    param_container = QWidget()
    param_grid = QGridLayout(param_container)
    param_grid.setContentsMargins(0, 6, 0, 6)
    layout.addWidget(param_container)

    entries = {}  # pkey -> (QLineEdit, kind)

    def current_key():
        return key_list[combo.currentIndex()]

    def rebuild_params():
        while param_grid.count():
            item = param_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        entries.clear()
        key = current_key()
        unit_suffix = f" ({app.pending_scale_unit})" if getattr(app, "pending_scale_factor", None) else " (px)"
        for r, (pkey, label_text, default, kind) in enumerate(MAZE_TEMPLATES[key]["params"]):
            # Every current "angle"-kind param already spells out its own
            # unit in label_text (e.g. "Rotation (deg)"), so only add a
            # suffix here for "length" (px/cm/etc, which the label text
            # never includes on its own) -- avoids a "(deg) (deg)"-style
            # doubled-up label.
            suffix = unit_suffix if kind == "length" else ""
            lbl = QLabel(label_text + suffix)
            param_grid.addWidget(lbl, r, 0)
            edit = QLineEdit(str(default))
            edit.setFixedWidth(90)
            param_grid.addWidget(edit, r, 1)
            entries[pkey] = (edit, kind)

    combo.currentIndexChanged.connect(lambda _i: rebuild_params())
    rebuild_params()

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    cancel_btn = QPushButton("Cancel")
    gen_btn = QPushButton("Generate Zones")
    gen_btn.setObjectName("accentBtn")
    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(gen_btn)
    layout.addLayout(btn_row)
    cancel_btn.clicked.connect(dialog.reject)

    def generate():
        key = current_key()
        scale_factor = getattr(app, "pending_scale_factor", None)
        params = {}
        try:
            for pkey, (edit, kind) in entries.items():
                value = float(edit.text())
                if kind == "length" and scale_factor:
                    value = value / scale_factor  # real-world units -> px
                params[pkey] = value
        except ValueError:
            QMessageBox.critical(dialog, "Invalid value", "Every field needs a plain number.")
            return

        try:
            zones = MAZE_TEMPLATES[key]["generate"](app.pending_warp_w, app.pending_warp_h, params)
        except Exception as exc:
            QMessageBox.critical(dialog, "Could not generate zones", str(exc))
            return

        dialog.accept()
        app.start_template_zone_op(key, zones)

    gen_btn.clicked.connect(generate)

    dialog.exec()
