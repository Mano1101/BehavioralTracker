"""Shrink the offline Linux runtime to what BehavioralTracker uses (trim_bundle.py does the same for Windows).

    python3 installer/unix/trim_linux.py <runtime>/python

* Python: removes the debug information from the Python program and libpython (about 190 MB).
* Qt: keeps QtCore / QtGui / QtWidgets / QtUiTools / QtSvg, the plugins a widgets application needs on X11
  and Wayland, and every Qt library they need (followed with readelf); drops the rest
  (QML / Quick / Designer / SQL / Network / printing ...).
Needs `strip` and `readelf` (package binutils). Run after slim.py.
"""
import glob
import os
import re
import shutil
import subprocess
import sys

PY = os.path.abspath(sys.argv[1])
SP = glob.glob(os.path.join(PY, "lib", "python3.*", "site-packages"))[0]
QT = os.path.join(SP, "PySide6")
QTLIB = os.path.join(QT, "Qt", "lib")
PLUG = os.path.join(QT, "Qt", "plugins")

KEEP_MODULES = {"QtCore", "QtGui", "QtWidgets", "QtUiTools", "QtSvg", "QtSvgWidgets"}
KEEP_PLUGINS = {                                   # None = keep the whole folder
    "platforms": re.compile(r"^libq(xcb|wayland|offscreen|minimal)\.so$"),
    "xcbglintegrations": None,
    "platformthemes": None,
    "platforminputcontexts": re.compile(r"^lib(compose|ibus)"),   # keyboard input methods
    "styles": None,
    "iconengines": None,
    "imageformats": re.compile(r"^libq(svg|ico|jpeg|gif)\.so$"),
    "wayland-shell-integration": None,
    "wayland-decoration-client": None,
    "wayland-graphics-integration-client": None,
}
freed = 0


def size(p):
    if os.path.isdir(p) and not os.path.islink(p):
        return sum(os.path.getsize(f) for f in glob.glob(os.path.join(p, "**"), recursive=True)
                   if os.path.isfile(f) and not os.path.islink(f))
    return os.path.getsize(p) if os.path.isfile(p) and not os.path.islink(p) else 0


def rm(p):
    global freed
    freed += size(p)
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p, ignore_errors=True)
    elif os.path.lexists(p):
        os.remove(p)


def needed(path):
    out = subprocess.run(["readelf", "-d", path], capture_output=True, text=True).stdout
    return re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


# ---- debug information of the Python program + libpython (and the standard-library modules)
for f in [os.path.join(PY, "lib", "libpython3.12.so.1.0"), os.path.join(PY, "bin", "python3.12")] + \
        glob.glob(os.path.join(PY, "lib", "python3.12", "lib-dynload", "*.so")):
    if os.path.isfile(f) and not os.path.islink(f):
        before = os.path.getsize(f)
        subprocess.run(["strip", "--strip-debug", f], check=True)
        freed += before - os.path.getsize(f)

# ---- Qt plugins
for sub in sorted(os.listdir(PLUG)) if os.path.isdir(PLUG) else []:
    d = os.path.join(PLUG, sub)
    if sub not in KEEP_PLUGINS:
        rm(d)
        continue
    pat = KEEP_PLUGINS[sub]
    if pat is not None:
        for f in os.listdir(d):
            if not pat.match(f):
                rm(os.path.join(d, f))

# ---- Qt modules (Python bindings)
for f in glob.glob(os.path.join(QT, "Qt*.abi3.so")):
    if os.path.basename(f).split(".")[0] not in KEEP_MODULES:
        rm(f)

for f in glob.glob(os.path.join(QT, "libpyside6qml*")) + [os.path.join(QT, "lib")]:   # PySide QML support, cmake
    rm(f)

# ---- Qt libraries: keep what the kept modules + plugins need, directly or indirectly
roots = glob.glob(os.path.join(QT, "*.so*")) + glob.glob(os.path.join(SP, "shiboken6", "*.so*")) + \
    glob.glob(os.path.join(PLUG, "**", "*.so"), recursive=True)
local = {os.path.basename(f): f for f in glob.glob(os.path.join(QTLIB, "*.so*"))}
keep, todo = set(), list(roots)
while todo:
    f = todo.pop()
    for name in needed(f):
        if name in local and name not in keep:
            keep.add(name)
            todo.append(local[name])
for name, f in local.items():
    if name.startswith("libQt6") and name not in keep:
        rm(f)
print(f"trimmed {freed / 1e6:.0f} MB; Qt libraries kept: {len([k for k in keep if k.startswith('libQt6')])}")
