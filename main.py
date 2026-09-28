"""Entry point (PySide6/Qt GUI). Run this file: `python main.py`

This is the current, actively maintained app. The old Tkinter GUI
(`gui/main_window.py`) still exists as an unmaintained fallback at
`main_legacy_tkinter.py` -- run `python main_legacy_tkinter.py` only if
you specifically need it; everything new (batch mode, subject database,
shape tools, Excel export, behavior classification, etc.) is built here.

Renamed from `main_qt.py` on 2026-09-24, once this GUI became the primary
app rather than a parallel/experimental one -- the previous name led to a
real mix-up: this filename now matches what someone would naturally run
first, and `BehavioralTracker.spec` (the PyInstaller build used for the
downloadable .exe/.app) was updated to match at the same time.
"""

import os
import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from qt_app.theme import build_stylesheet, PALETTE
from qt_app.main_window import MainWindow

ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources", "icon.png")


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("BehavioralTracker")
    app.setOrganizationName("BehavioralTracker")
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    # MainWindow.__init__ also applies this same stylesheet; doing it here
    # too just avoids an unstyled flash before the window is constructed.
    app.setStyleSheet(build_stylesheet(PALETTE))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
