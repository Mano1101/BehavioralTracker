"""OLD, unmaintained Tkinter GUI entry point. Run this file:
`python main_legacy_tkinter.py`

This used to be `main.py`. The current, actively developed app is
`main.py` (PySide6/Qt GUI, in `qt_app/`) -- run that instead unless you
specifically need this old Tkinter version. This file was renamed (not
deleted) on 2026-09-24 so the old GUI keeps working for anyone who still
needs it, but no longer gets run by accident just because it used to be
named `main.py`.
"""

import sys
import os

# A windowed (no-console) Windows build has no real stdout/stderr -- some
# PyInstaller/Python combinations leave sys.stdout/sys.stderr as None in
# that case, and this codebase calls print() throughout (calibration
# steps, batch progress, etc.). print()-ing to None raises an exception,
# which would crash the app the moment any of those lines run. Redirect
# to a harmless sink instead so those prints just go nowhere.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

from gui.main_window import main

if __name__ == "__main__":
    main()
