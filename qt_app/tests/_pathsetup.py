"""
Shared path/environment setup for the qt_app headless test suite. Import
this FIRST, before any PySide6/qt_app/tracking import, in every test
module -- it puts the repo root on sys.path, chdir()s into it (several
tests use relative paths like "dummy_behavior_test.mp4"), and sets
QT_QPA_PLATFORM=offscreen before PySide6 ever gets imported anywhere,
since PySide6 reads that env var at first-import/init time.

Import it as a BARE module -- `import _pathsetup`, not
`import qt_app.tests._pathsetup` -- since these test files are run
directly (`python3.12 qt_app/tests/test_X.py`), and when Python runs a
script that way it only puts that script's OWN directory (qt_app/tests) on
sys.path, not the repo root. A package-qualified import of this same file
would need the repo root on sys.path already -- exactly the chicken-and-
egg problem this module exists to solve -- while a bare import resolves
directly against qt_app/tests, which Python has already made importable.

These tests build and drive a real MainWindow -- video decoding, real Qt
mouse/keyboard events via QTest, and (in test_ml_pipeline.py) real
PyTorch training/inference -- rather than mocking the app apart, so they
need an actual (virtual) display server even under the offscreen platform
plugin in this environment; run them under `xvfb-run -a`, e.g.:

    xvfb-run -a python3.12 qt_app/tests/test_canvas_smoke.py

or run the whole suite with run_all.py (see qt_app/tests/README.md).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

VIDEO = os.path.join(REPO_ROOT, "dummy_behavior_test.mp4")
