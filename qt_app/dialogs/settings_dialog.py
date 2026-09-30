"""
Settings dialog -- Upgrade Plan Tier 2 #8, replacing MainWindow.
on_open_settings's old "a dedicated Settings screen is on the way"
placeholder with a real one.

Edits the SAME app._memory keys the Setup page's Detection Settings panel
(setup_page.py's _detection_settings/_setting_field/_color_mode_group)
already reads on every rebuild -- so Save takes effect immediately on the
current Setup page too (via app.setup_page.rebuild), not just on the next
launch -- and persists them (qt_app/app_settings.py) so a brand-new
session starts from these instead of the hardcoded fallbacks every time.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QCheckBox, QComboBox, QPushButton, QMessageBox,
)

from qt_app.app_settings import save_settings, SETTINGS_PATH

# (label, memory key, hardcoded fallback) -- the SAME fallback values
# setup_page.py's own _setting_field calls use for these keys, so a field
# this dialog has never touched shows exactly what a fresh Setup page
# would already show.
NUMERIC_FIELDS = [
    ("Background samples", "bg_samples_entry", "100"),
    ("Difference threshold", "threshold_entry", "25"),
    ("Min mouse area, px", "min_area_entry", "15"),
    ("Max object area, px", "max_area_entry", "5000"),
    ("Max movement / frame, px", "max_jump_entry", "100"),
    ("Default real-world distance", "real_distance_entry", "30"),
    ("Default units (e.g. cm, mm, in)", "units_entry", "cm"),
    ("Preview frames to save/check", "preview_samples_entry", "6"),
    ("Default time bin size, seconds (blank = 60)", "bin_size_entry", ""),
]
CHECK_FIELDS = [
    ("Per-zone adaptive threshold (on by default)", "use_zone_threshold_var"),
]
COLOR_MODES = [("auto", "Auto (recommended)"), ("gray", "Grayscale (faster)"), ("rgb", "RGB / Color")]

ALL_KEYS = [key for _label, key, _fallback in NUMERIC_FIELDS] + \
    [key for _label, key in CHECK_FIELDS] + ["color_mode_var"]


def open_settings_dialog(app, parent):
    dialog = QDialog(parent)
    dialog.setWindowTitle("Settings")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)

    intro = QLabel(
        "Starting defaults for every new session's Detection Settings.\n\n"
        f"Saved to: {SETTINGS_PATH}"
    )
    intro.setWordWrap(True)
    intro.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(intro)

    grid = QGridLayout()
    layout.addLayout(grid)
    entries = {}
    for row, (label, key, fallback) in enumerate(NUMERIC_FIELDS):
        grid.addWidget(QLabel(label + ":"), row, 0)
        entry = QLineEdit(str(app._memory.get(key, fallback)))
        grid.addWidget(entry, row, 1)
        entries[key] = entry

    checks = {}
    for label, key in CHECK_FIELDS:
        cb = QCheckBox(label)
        cb.setChecked(bool(app._memory.get(key, False)))
        layout.addWidget(cb)
        checks[key] = cb

    color_row = QHBoxLayout()
    color_row.addWidget(QLabel("Default color mode:"))
    color_combo = QComboBox()
    for value, text in COLOR_MODES:
        color_combo.addItem(text, value)
    current_color = app._memory.get("color_mode_var", "auto")
    color_combo.setCurrentIndex(next(
        (i for i, (value, _text) in enumerate(COLOR_MODES) if value == current_color), 0))
    color_row.addWidget(color_combo)
    color_row.addStretch()
    layout.addLayout(color_row)

    status_label = QLabel("")
    status_label.setWordWrap(True)
    status_label.setStyleSheet("color: #666666; font-size: 9px;")
    layout.addWidget(status_label)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    close_btn = QPushButton("Close")
    close_btn.clicked.connect(dialog.reject)
    save_btn = QPushButton("Save Defaults")
    save_btn.setObjectName("accentBtn")
    btn_row.addWidget(close_btn)
    btn_row.addWidget(save_btn)
    layout.addLayout(btn_row)

    def do_save():
        for key, entry in entries.items():
            app._memory[key] = entry.text().strip()
        for key, cb in checks.items():
            app._memory[key] = cb.isChecked()
        app._memory["color_mode_var"] = color_combo.currentData()

        try:
            save_settings({key: app._memory[key] for key in ALL_KEYS if key in app._memory})
        except OSError as exc:
            QMessageBox.critical(dialog, "Could not save Settings", str(exc))
            return

        app.setup_page.rebuild(resave=False)
        status_label.setText(
            "Saved -- new sessions (and Reset All) will start from these, and the current "
            "Setup page has been refreshed to match."
        )

    save_btn.clicked.connect(do_save)
    dialog.exec()
