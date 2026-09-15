"""Entry point. Run this file: `python main.py`"""

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
