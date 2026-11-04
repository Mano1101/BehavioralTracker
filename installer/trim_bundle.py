"""Shrink the bundled Windows Python to what BehavioralTracker uses.

    python3 installer/trim_bundle.py <stage>/python

* Qt: keeps QtCore / QtGui / QtWidgets / QtUiTools / QtSvg, the plugins a
  widgets application needs on Windows, and every DLL they import (computed
  with objdump); drops the rest (QML / Quick / Lottie / Network / SQL ...).
* Python: drops tkinter + Tcl/Tk, pip / ensurepip, type stubs (*.pyi) and
  byte-code caches made by the Linux build machine.
Needs `x86_64-w64-mingw32-objdump` (package mingw-w64).
"""
import glob
import os
import re
import shutil
import subprocess
import sys

PY = os.path.abspath(sys.argv[1])
KEEP_PIP = "--keep-pip" in sys.argv[2:]          # the online Setup needs pip to fetch the packages
SP = os.path.join(PY, "Lib", "site-packages")
QT = os.path.join(SP, "PySide6")

KEEP_PYD = {"QtCore.pyd", "QtGui.pyd", "QtWidgets.pyd", "QtUiTools.pyd", "QtSvg.pyd"}
KEEP_PLUGINS = {
    "platforms": {"qwindows.dll", "qoffscreen.dll", "qminimal.dll"},
    "styles": None,                                   # None = keep all
    "imageformats": {"qsvg.dll", "qico.dll", "qjpeg.dll", "qgif.dll"},
    "iconengines": None,
}
ALWAYS = {"pyside6.abi3.dll"}                         # + the MSVC runtime below
MSVC = re.compile(r"^(msvcp140.*|vcruntime140.*|concrt140)\.dll$", re.I)


def imports(path):
    out = subprocess.run(["x86_64-w64-mingw32-objdump", "-p", path], capture_output=True, text=True).stdout
    return {m.lower() for m in re.findall(r"DLL Name: (\S+)", out)}


def rm(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
    elif os.path.exists(path):
        os.remove(path)


# ---- Qt tools / QML / translations / headers (never used by a widgets application)
if os.path.isdir(QT):
    for sub in ("qml", "translations", "metatypes", "include", "typesystems", "glue", "scripts", "doc", "lib",
                "resources", "examples"):
        rm(os.path.join(QT, sub))
    for pat in ("*.exe", "*.lib", "*.pyi"):
        for f in glob.glob(os.path.join(QT, pat)):
            rm(f)
    for f in glob.glob(os.path.join(SP, "*.whl")):
        rm(f)

# ---- Qt plugins
for sub in (os.listdir(os.path.join(QT, "plugins")) if os.path.isdir(QT) else []):
    d = os.path.join(QT, "plugins", sub)
    if sub not in KEEP_PLUGINS:
        rm(d)
        continue
    keep = KEEP_PLUGINS[sub]
    if keep is not None:
        for f in os.listdir(d):
            if f.lower() not in keep:
                rm(os.path.join(d, f))

# ---- Qt modules + DLL closure
for f in (glob.glob(os.path.join(QT, "*.pyd")) if os.path.isdir(QT) else []):
    if os.path.basename(f) not in KEEP_PYD:
        rm(f)
local = {f.lower(): os.path.join(QT, f) for f in (os.listdir(QT) if os.path.isdir(QT) else [])
         if f.lower().endswith(".dll")}
shib = os.path.join(SP, "shiboken6")
for f in (os.listdir(shib) if os.path.isdir(shib) else []):
    if f.lower().endswith(".dll"):
        local.setdefault(f.lower(), os.path.join(shib, f))
roots = [os.path.join(QT, f) for f in KEEP_PYD if os.path.exists(os.path.join(QT, f))]
roots += glob.glob(os.path.join(QT, "plugins", "*", "*.dll"))
needed, stack = set(), list(roots)
while stack:
    p = stack.pop()
    for d in imports(p):
        if d in local and d not in needed:
            needed.add(d)
            stack.append(local[d])
removed = []
for name, path in local.items():
    if os.path.dirname(path) != QT:
        continue
    if name in needed or name in ALWAYS or MSVC.match(name):
        continue
    rm(path)
    removed.append(name)
if removed:
    print("Qt DLLs removed:", ", ".join(sorted(removed)))

# ---- Python extras the app never uses
for p in (os.path.join(PY, "tcl"), os.path.join(PY, "Lib", "tkinter"), os.path.join(PY, "Lib", "turtledemo"),
          os.path.join(PY, "Lib", "ensurepip"), os.path.join(PY, "Lib", "turtle.py")):
    rm(p)
for p in glob.glob(os.path.join(PY, "DLLs", "_tkinter.pyd")) + glob.glob(os.path.join(PY, "DLLs", "t*86t.dll")):
    rm(p)
if not KEEP_PIP:
    rm(os.path.join(SP, "pip"))
    for p in glob.glob(os.path.join(SP, "pip-*.dist-info")):
        rm(p)
n_pyi = 0
for dp, dn, fn in os.walk(PY):
    for f in fn:
        if f.endswith(".pyi"):
            os.remove(os.path.join(dp, f))
            n_pyi += 1
    for d in list(dn):
        if d == "__pycache__":
            shutil.rmtree(os.path.join(dp, d), ignore_errors=True)
            dn.remove(d)
print(f"removed {n_pyi} type-stub files; bundle trimmed")
