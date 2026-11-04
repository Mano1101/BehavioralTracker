"""Licence gate for the LICENSED edition (Windows, macOS, Linux).

main_pro.py calls ensure_licensed() before the window opens.  The unlicensed
edition simply does not contain this _license folder, so the app starts freely.

The licence file is node-locked (it works only on the computer whose Machine ID
it was issued for) and Ed25519-signed: this folder holds only the PUBLIC key, so
it can check a licence but can never create one.
"""
import os
import sys
from pathlib import Path

from .check import machine_id, verify_license_bytes

HERE = Path(__file__).resolve().parent
SUPPORT_EMAIL = "manojkumarmunusamy1101@outlook.com"
SUPPORT_PHONE = "9080338669"


def _app_dir():
    if os.environ.get("BEHAVIORALTRACKER_HOME"):
        d = Path(os.environ["BEHAVIORALTRACKER_HOME"])
    elif sys.platform == "win32" and os.environ.get("APPDATA"):
        d = Path(os.environ["APPDATA"]) / "BehavioralTracker"
    else:                                         # macOS / Linux
        d = Path.home() / ".behavioraltracker"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path.home()
    return d


LICENSE_PATH = _app_dir() / "license.key"


def _public_key():
    return (HERE / "public.pem").read_bytes()


def ensure_licensed():
    """True when a valid licence for this computer is installed (asks for one if not).
    False = the user chose Quit."""
    pub = _public_key()
    mid = machine_id()
    blob = LICENSE_PATH.read_bytes() if LICENSE_PATH.exists() else b""
    ok, info = verify_license_bytes(blob, pub) if blob else (False, "No licence installed yet.")
    while not ok:
        path = _license_dialog(mid, info if isinstance(info, str) else "")
        if path is None:
            return False
        try:
            data = Path(path).read_bytes()
        except Exception:
            data = b""
        ok, info = verify_license_bytes(data, pub)
        if ok:
            try:
                LICENSE_PATH.write_bytes(data)
            except Exception:
                pass
        else:
            _error(info if isinstance(info, str) else "This licence is not valid.")
    try:                                              # licence holder + number for the app (audit trail, sidebar)
        from tracking.project_file import set_licence
        set_licence(info)
    except Exception:
        pass
    return True


def _license_dialog(mid, reason):
    """Returns the chosen licence-file path, or None to quit."""
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog, QFrame, QHBoxLayout, QLabel,
                                   QLineEdit, QPushButton, QVBoxLayout)
    app = QApplication.instance() or QApplication(sys.argv)
    try:                                    # light / dark like the computer, readable in both
        app.setStyle("Fusion")
        from pro_ui import theme
        theme.install(app)
    except Exception:
        pass
    dlg = QDialog()
    dlg.setWindowTitle("BehavioralTracker — Activation")
    dlg.setMinimumWidth(480)
    dlg.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
    v = QVBoxLayout(dlg)
    title = QLabel("Activate BehavioralTracker")
    title.setStyleSheet("font-size:15pt; font-weight:700;")
    v.addWidget(title)
    msg = QLabel("This copy needs a licence for this computer." + (f"\n\n{reason}" if reason else ""))
    msg.setWordWrap(True)
    v.addWidget(msg)
    box = QFrame()
    box.setObjectName("actbox")
    box.setStyleSheet("QFrame#actbox{background:#f3f6fb;border:1px solid #d5dce8;border-radius:6px;}")
    bl = QVBoxLayout(box)
    bl.addWidget(QLabel("1.  Send this Machine ID to get your licence:"))
    row = QHBoxLayout()
    field = QLineEdit(mid)
    field.setReadOnly(True)
    field.setStyleSheet("font-weight:700; font-size:12pt;")
    copy = QPushButton("Copy")
    copy.clicked.connect(lambda: QApplication.clipboard().setText(mid))
    row.addWidget(field, 1)
    row.addWidget(copy)
    bl.addLayout(row)
    bl.addWidget(QLabel(f"2.  Email it to:  {SUPPORT_EMAIL}   (questions: phone {SUPPORT_PHONE})"))
    bl.addWidget(QLabel("3.  When you receive the licence file, click below."))
    v.addWidget(box)
    btns = QHBoxLayout()
    quit_b = QPushButton("Quit")
    load_b = QPushButton("I have a licence — load it…")
    load_b.setStyleSheet("padding:8px 14px; font-weight:600; background:#1f6feb; color:white; border:none;"
                         "border-radius:6px;")
    btns.addWidget(quit_b)
    btns.addStretch(1)
    btns.addWidget(load_b)
    v.addLayout(btns)
    state = {"path": None}

    def do_load():
        p, _f = QFileDialog.getOpenFileName(dlg, "Select your licence file", str(Path.home()),
                                            "Licence (*.key *.lic);;All files (*)")
        if p:
            state["path"] = p
            dlg.accept()
    load_b.clicked.connect(do_load)
    quit_b.clicked.connect(dlg.reject)
    return state["path"] if dlg.exec() == QDialog.Accepted else None


def _error(text):
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
        from pro_ui import theme
        theme.install(app)
    except Exception:
        pass
    QMessageBox.warning(None, "BehavioralTracker — Activation", text)
