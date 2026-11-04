"""Turn the plain application into a protected bundle.

    python protect/build_secure.py --src <app_src> --out <secured_app> [--licensed]

What it produces in <secured_app>:
  main_pro.py            tiny plain bootstrap (installs the loader, then runs)
  _sec/loader.pyc        compiled loader; holds the split/masked decryption key
  _sec/manifest.json     sha256 of every blob + an HMAC over the manifest
  _sec/pubkey.pem        Ed25519 public key (licensed build only)
  _sec/build.json        {"licensed": bool}
  _sec/mods/*.btx        every app module, AES-encrypted (Fernet); no source ships
  _sec/mods/__ui__.btx   the Qt .ui, encrypted (loaded from memory at runtime)
  pro_ui/icons/…, resources/…   kept (images only; referenced by the stylesheet)

Protection layers:
  1  no plain source   — modules ship only as AES-encrypted blobs
  2  encrypted UI       — the .ui is decrypted in memory, never written to disk
  3  compiled key-split loader — the key is bytecode-only, broken up and masked
  4  integrity lock     — any edited/replaced blob fails the hash/HMAC -> refuse
  5  node-locked licence — Ed25519-signed, bound to one machine (licensed build)
"""
import argparse
import hashlib
import hmac
import json
import marshal

import py_compile
import secrets
import shutil
import sys
from pathlib import Path

from cryptography.fernet import Fernet

HERE = Path(__file__).resolve().parent

# app modules to protect: (dotted name, is_package).  Submodules are discovered.
PACKAGES = ["pro_ui", "tracking", "analysis", "output"]
EXTRA_MODULES = {"btlicense": HERE / "licensing.py"}      # licence logic, shipped encrypted
KEEP_DIRS = {"icons", "resources"}                         # images referenced by the stylesheet / window


def collect_modules(src):
    """dotted name -> (source_path, is_package)."""
    mods = {}
    for pkg in PACKAGES:
        pdir = src / pkg
        if not pdir.is_dir():
            continue
        for py in sorted(pdir.rglob("*.py")):
            rel = py.relative_to(src).with_suffix("")
            dotted = ".".join(rel.parts)
            is_pkg = py.name == "__init__.py"
            if is_pkg:
                dotted = ".".join(rel.parts[:-1])
            mods[dotted] = (py, is_pkg)
    for name, path in EXTRA_MODULES.items():
        if path.exists():
            mods[name] = (path, False)
    return mods


LOADER_TEMPLATE = r'''# Auto-generated secure loader. Bytecode only -- do not ship the source.
import hashlib, hmac, importlib.util, json, marshal, sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_APP = _HERE.parent
_PARTS = __PARTS__
_MASKS = __MASKS__


def _key():
    out = bytearray()
    for chunk, m in zip(_PARTS, _MASKS):
        out += bytes(b ^ m for b in chunk)
    return bytes(out)


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(_key())


_REPAIR = ("This usually means an antivirus program quarantined or changed it.\n"
           "To repair:\n"
           "  1. Open your antivirus program, restore the BehavioralTracker file from its quarantine and add the "
           "BehavioralTracker folder to its exceptions (allowed list).\n"
           "  2. Run the BehavioralTracker Setup again (Repair / reinstall).\n\n"
           "Your settings, licence and results are kept.\n\n"
           "Still not working? Contact Manojkumar Munusamy: manojkumarmunusamy1101@outlook.com, "
           "phone 9080338669")


def _fail(msg):
    # fail closed: show a message if we can, then stop. Nothing is deleted.
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "BehavioralTracker", msg)
        app.processEvents()
    except Exception:
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, "BehavioralTracker", 0x10)
        except Exception:
            sys.stderr.write(msg + "\n")
    sys.exit(3)


_MANIFEST = None
_INTEGRITY_OK = False


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_manifest():
    global _MANIFEST, _INTEGRITY_OK
    data = json.loads((_HERE / "manifest.json").read_text("utf-8"))
    sig = bytes.fromhex(data.pop("hmac"))
    want = hmac.new(_key(), _canonical(data), hashlib.sha256).digest()
    if not hmac.compare_digest(sig, want):
        _fail("BehavioralTracker cannot start: its program files have been changed.\n\n" + _REPAIR)
    _MANIFEST = data
    # verify every blob's hash before anything runs (fail closed)
    moddir = _HERE / "mods"
    for info in list(data["modules"].values()) + [data["ui"]]:
        blob = (moddir / info["file"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != info["sha"]:
            _fail("BehavioralTracker cannot start: the program file '%s' is damaged or was changed.\n\n"
                  % info["file"] + _REPAIR)
    _INTEGRITY_OK = True


class _SecureLoader(Loader):
    def __init__(self, dotted, info):
        self.dotted = dotted
        self.info = info

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        blob = (_HERE / "mods" / self.info["file"]).read_bytes()
        code = marshal.loads(_fernet().decrypt(blob))
        pkg = self.dotted.rpartition(".")[0]
        module.__file__ = str(_APP / Path(*self.dotted.split(".")) /
                              ("__init__.py" if self.info["pkg"] else "")) if self.info["pkg"] \
            else str(_APP / Path(*self.dotted.split(".")).with_suffix(".py"))
        if self.info["pkg"]:
            module.__path__ = [str(_APP / Path(*self.dotted.split(".")))]
            module.__package__ = self.dotted
        else:
            module.__package__ = pkg
        exec(code, module.__dict__)


class _Finder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if _MANIFEST is None:
            return None
        info = _MANIFEST["modules"].get(fullname)
        if info is None:
            return None
        return ModuleSpec(fullname, _SecureLoader(fullname, info), is_package=info["pkg"])


def ui_bytes():
    """Decrypted bytes of the Qt .ui (used by pro_ui to build the window)."""
    info = _MANIFEST["ui"]
    blob = (_HERE / "mods" / info["file"]).read_bytes()
    return _fernet().decrypt(blob)


def build_info():
    return json.loads((_HERE / "build.json").read_text("utf-8"))


def public_key_pem():
    p = _HERE / "pubkey.pem"
    return p.read_bytes() if p.exists() else None


def install():
    _load_manifest()
    sys.meta_path.insert(0, _Finder())
'''


def split_key(key, n=6):
    """Break the key into n masked chunks (lightweight obfuscation in bytecode)."""
    size = (len(key) + n - 1) // n
    parts, masks = [], []
    for i in range(0, len(key), size):
        chunk = key[i:i + size]
        m = secrets.randbelow(255) + 1
        parts.append(bytes(b ^ m for b in chunk))
        masks.append(m)
    return parts, masks


BOOTSTRAP = r'''"""BehavioralTracker 1.0.0 -- start the application (protected build).

This is the only plain-text file; it just turns on the secure loader (which
decrypts the real program in memory) and then launches it.
"""
import datetime
import os
import sys


def _log_dir():
    if os.environ.get("BEHAVIORALTRACKER_HOME"):
        d = os.environ["BEHAVIORALTRACKER_HOME"]
    elif os.environ.get("APPDATA"):
        d = os.path.join(os.environ["APPDATA"], "BehavioralTracker")
    else:                                   # macOS / Linux
        d = os.path.join(os.path.expanduser("~"), ".behavioraltracker")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        d = os.path.expanduser("~")
    return d


def _streams():
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        f = open(os.path.join(_log_dir(), "BehavioralTracker.log"), "a", encoding="utf-8", buffering=1)
        f.write("\n===== started %s =====\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        sys.stdout = sys.stdout or f
        sys.stderr = sys.stderr or f
    except Exception:
        pass


def _fatal(text):
    path = os.path.join(_log_dir(), "crash.log")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n===== %s =====\n%s\n" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text))
    except Exception:
        pass
    msg = ("BehavioralTracker could not start.\n\n" + text.strip().splitlines()[-1] + "\n\nDetails: " + path
           + "\n\nPlease send this file to manojkumarmunusamy1101@outlook.com (Manojkumar Munusamy, "
           "phone 9080338669).")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "BehavioralTracker", msg)
        app.processEvents()
    except Exception:
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, msg, "BehavioralTracker", 0x10)
            except Exception:
                pass


def main():
    _streams()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import _sec.loader as loader
        loader.install()
        from btgate import run_guarded      # encrypted: licence gate + launch
        return run_guarded(loader)
    except SystemExit:
        raise
    except Exception:
        import traceback
        _fatal(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
'''

# the licence gate -- shipped encrypted as module "btgate"
GATE_SOURCE = r'''"""Licence gate (encrypted). Verifies the node-locked licence, then runs the app."""
import os
import sys
from pathlib import Path


def _app_dir():
    if os.environ.get("BEHAVIORALTRACKER_HOME"):
        d = Path(os.environ["BEHAVIORALTRACKER_HOME"])
    elif os.environ.get("APPDATA"):
        d = Path(os.environ["APPDATA"]) / "BehavioralTracker"
    else:                                   # macOS / Linux
        d = Path(os.path.expanduser("~")) / ".behavioraltracker"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path(os.path.expanduser("~"))
    return d


LICENSE_PATH = _app_dir() / "license.key"
SUPPORT_EMAIL = "manojkumarmunusamy1101@outlook.com"
SUPPORT_PHONE = "9080338669"


def _run_app():
    from pro_ui.ux_controller import run
    app, _ctrl = run()
    return app.exec()


def run_guarded(loader):
    if not loader.build_info().get("licensed"):
        return _run_app()
    import btlicense
    pub = loader.public_key_pem()
    mid = btlicense.machine_id()
    blob = LICENSE_PATH.read_bytes() if LICENSE_PATH.exists() else b""
    ok, _info = btlicense.verify_license_bytes(blob, pub) if blob else (False, "No licence installed yet.")
    while not ok:
        choice = _license_dialog(mid, _info if isinstance(_info, str) else "")
        if choice is None:
            return 0
        try:
            data = Path(choice).read_bytes()
        except Exception:
            data = b""
        ok, _info = btlicense.verify_license_bytes(data, pub)
        if ok:
            try:
                LICENSE_PATH.write_bytes(data)
            except Exception:
                pass
        else:
            _error_dialog(_info if isinstance(_info, str) else "This licence is not valid.")
    try:                                    # licence holder + number for the app (audit trail, sidebar)
        from tracking.project_file import set_licence
        set_licence(_info)
    except Exception:
        pass
    return _run_app()


def _license_dialog(machine_id, reason):
    """Returns a chosen licence-file path, or None to quit."""
    from PySide6.QtWidgets import (QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout,
                                   QPushButton, QFileDialog, QFrame)
    from PySide6.QtCore import Qt
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
    field = QLineEdit(machine_id)
    field.setReadOnly(True)
    field.setStyleSheet("font-weight:700; font-size:12pt;")
    copy = QPushButton("Copy")
    copy.clicked.connect(lambda: (QApplication.clipboard().setText(machine_id)))
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
    dlg.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
    return state["path"] if dlg.exec() == QDialog.Accepted else None


def _error_dialog(text):
    from PySide6.QtWidgets import QApplication, QMessageBox
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        app.setStyle("Fusion")
        from pro_ui import theme
        theme.install(app)
    except Exception:
        pass
    QMessageBox.warning(None, "BehavioralTracker — Activation", text)
'''


def main():
    ap = argparse.ArgumentParser(description="Build the protected BehavioralTracker bundle.")
    ap.add_argument("--src", required=True, help="plain application source directory")
    ap.add_argument("--out", required=True, help="output directory for the secured app")
    ap.add_argument("--licensed", action="store_true", help="require a node-locked licence to run")
    ap.add_argument("--pubkey", help="Ed25519 public key PEM (default: protect/keys/public.pem)")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sec = out / "_sec"
    (sec / "mods").mkdir(parents=True)

    key = Fernet.generate_key()
    fernet = Fernet(key)

    # --- gate module source (written to a temp file so it can be compiled like the others)
    gate_path = sec / "_gate_src.py"
    gate_path.write_text(GATE_SOURCE, "utf-8")
    mods = collect_modules(src)
    mods["btgate"] = (gate_path, False)

    manifest = {"product": "BehavioralTracker", "modules": {}, "licensed": bool(args.licensed)}
    for dotted, (path, is_pkg) in sorted(mods.items()):
        code = compile(path.read_text("utf-8"), f"<{dotted}>", "exec")
        blob = fernet.encrypt(marshal.dumps(code))
        fname = dotted + ".btx"
        (sec / "mods" / fname).write_bytes(blob)
        manifest["modules"][dotted] = {"file": fname, "pkg": is_pkg,
                                       "sha": hashlib.sha256(blob).hexdigest()}
    gate_path.unlink()

    # --- encrypt the .ui
    ui_src = src / "pro_ui" / "behavioral_tracker.ui"
    ui_blob = fernet.encrypt(ui_src.read_bytes())
    (sec / "mods" / "__ui__.btx").write_bytes(ui_blob)
    manifest["ui"] = {"file": "__ui__.btx", "sha": hashlib.sha256(ui_blob).hexdigest()}

    # --- HMAC over the manifest (tamper -> refuse)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["hmac"] = hmac.new(key, canonical, hashlib.sha256).hexdigest()
    (sec / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    (sec / "build.json").write_text(json.dumps({"licensed": bool(args.licensed)}), "utf-8")

    # --- public key for licence checks
    if args.licensed:
        pub = Path(args.pubkey) if args.pubkey else (HERE / "keys" / "public.pem")
        if not pub.exists():
            raise SystemExit(f"Licensed build needs a public key: {pub}\nRun  python protect/keygen.py init  first.")
        shutil.copyfile(pub, sec / "pubkey.pem")

    # --- loader.py with the split key, compiled to .pyc, source removed
    parts, masks = split_key(key)
    loader_src = (LOADER_TEMPLATE
                  .replace("__PARTS__", repr([bytes(p) for p in parts]))
                  .replace("__MASKS__", repr(masks)))
    loader_py = sec / "loader.py"
    loader_py.write_text(loader_src, "utf-8")
    py_compile.compile(str(loader_py), cfile=str(sec / "loader.pyc"), optimize=2, doraise=True)
    loader_py.unlink()
    (sec / "__init__.py").write_text("", "utf-8")   # make _sec importable; loader.pyc loaded explicitly

    # loader is imported as `_sec.loader`; put the .pyc where Python finds it for 3.12
    # (a bare .pyc next to __init__ is importable when named loader.pyc only via import machinery that
    #  accepts sourceless; ensure it is on the cache path)
    _place_sourceless(sec / "loader.pyc", sec, "loader")

    # --- plain bootstrap + keep images
    (out / "main_pro.py").write_text(BOOTSTRAP, "utf-8")
    for pkg in PACKAGES:
        for d in KEEP_DIRS:
            srcd = src / pkg / d
            if srcd.is_dir():
                shutil.copytree(srcd, out / pkg / d)
    if (src / "resources").is_dir():
        shutil.copytree(src / "resources", out / "resources")
    # README_UX.md is a DEVELOPER document (internal file names, how the engine works) -> never shipped

    n = len(manifest["modules"])
    print(f"Secured {n} modules + UI -> {out}")
    print(f"  licensed build: {bool(args.licensed)}")
    print("  plain files shipped: main_pro.py, _sec/__init__.py (everything else is encrypted)")


def _place_sourceless(pyc_path, pkg_dir, name):
    """Make `loader.pyc` importable as a sourceless module `_sec.loader`."""
    # Python imports `_sec.loader` from `_sec/loader.pyc` directly when no .py exists. Keep as-is.
    # (SourcelessFileLoader handles a bare .pyc beside the package __init__.)
    pass


if __name__ == "__main__":
    sys.exit(main() or 0)
