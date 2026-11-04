"""BehavioralTracker 1.0.0 -- start the application.

Run from source:   python main_pro.py      (Windows, macOS, Linux)
Installed (Windows): BehavioralTracker.exe runs this file with the bundled Python.

Licensed edition: the _license folder is present -> a valid licence for this
computer is required before the window opens.  Unlicensed edition: no _license
folder -> the app starts freely.
"""
import datetime
import os
import sys
import traceback


def _log_dir():
    base = os.environ.get("LOCALAPPDATA")
    d = os.path.join(base, "BehavioralTracker") if base else os.path.join(os.path.expanduser("~"), ".behavioraltracker")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = os.path.expanduser("~")
    return d


def _console_less_streams():
    """pythonw.exe has no console (sys.stdout is None): keep messages in a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        f = open(os.path.join(_log_dir(), "BehavioralTracker.log"), "a", encoding="utf-8", buffering=1)
        f.write(f"\n===== started {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f
    except Exception:
        pass


def _show_fatal(text):
    path = os.path.join(_log_dir(), "crash.log")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} =====\n{text}\n")
    except Exception:
        pass
    msg = ("BehavioralTracker could not start.\n\n" + text.strip().splitlines()[-1] +
           f"\n\nDetails were saved in:\n{path}")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "BehavioralTracker", msg)
        app.processEvents()
        return
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "BehavioralTracker", 0x10)
            return
        except Exception:
            pass
    print(msg, file=sys.stderr)


def main():
    _console_less_streams()
    try:
        try:
            from _license import ensure_licensed      # present only in the licensed edition
        except ImportError:
            ensure_licensed = None
        if ensure_licensed is not None and not ensure_licensed():
            return 0                                  # the user chose Quit on the activation window
        from pro_ui.ux_controller import run
        app, _controller = run()
    except Exception:
        _show_fatal(traceback.format_exc())
        return 1
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
