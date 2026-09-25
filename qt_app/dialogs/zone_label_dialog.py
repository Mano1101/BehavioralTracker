"""
Zone rename dialog -- the Qt equivalent of TrackerApp._prompt_zone_label in
gui/main_window.py. Shown when a researcher clicks a template-generated
zone on the preview canvas (the "select instead of freehand-draw" flow the
Maze Template / Quick Setup pipeline uses): confirm the geometry's guess,
pick one of the template's other suggested names (e.g. swap which arm is
actually "Open"), or type a custom name.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox,
)


def prompt_zone_label(parent, current_name, suggestions, existing_names):
    """Blocking modal dialog. Returns the chosen name (str) on OK, or None
    if cancelled. `existing_names` is the set of OTHER zones' names already
    in use (i.e. not including current_name itself) -- picking one of
    those is rejected as a duplicate, same as the Tkinter version."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Name this zone")
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)
    label = QLabel(f"What is this zone? (currently \"{current_name}\")")
    label.setWordWrap(True)
    layout.addWidget(label)

    combo = QComboBox()
    combo.setEditable(True)
    if suggestions:
        combo.addItems(suggestions)
    combo.setCurrentText(current_name)
    layout.addWidget(combo)

    btn_row = QHBoxLayout()
    btn_row.addStretch()
    cancel_btn = QPushButton("Cancel")
    ok_btn = QPushButton("OK")
    ok_btn.setObjectName("accentBtn")
    btn_row.addWidget(cancel_btn)
    btn_row.addWidget(ok_btn)
    layout.addLayout(btn_row)

    result = {"name": None}

    def confirm():
        new_name = combo.currentText().strip()
        if not new_name:
            QMessageBox.critical(dialog, "Name needed", "Enter or pick a zone name.")
            return
        if new_name != current_name and new_name in existing_names:
            QMessageBox.critical(dialog, "Name in use", f"'{new_name}' is already used by another zone.")
            return
        result["name"] = new_name
        dialog.accept()

    ok_btn.clicked.connect(confirm)
    cancel_btn.clicked.connect(dialog.reject)

    dialog.exec()
    return result["name"]
