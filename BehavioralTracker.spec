# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Animal Behaviour Tracker.
# Usage:
#   pip install pyinstaller
#   pyinstaller BehavioralTracker.spec
#
# Built artifacts:
#   Windows: dist/BehavioralTracker.exe
#   macOS:   dist/BehavioralTracker.app
#   Linux:   dist/BehavioralTracker

import sys
import os

block_cipher = None

IS_WINDOWS = sys.platform.startswith("win")
IS_MACOS = sys.platform == "darwin"

# Equivalent of --collect-all for the large scientific packages.  Using the
# PyInstaller hook helpers gives us the exact same behaviour as the CLI
# `--collect-all PKG` flag but embedded inside the spec file so both CI and
# local builds are identical.
from PyInstaller.utils.hooks import collect_all  # noqa: E402

extra_datas = []
extra_binaries = []
extra_hiddenimports = []
for pkg in ("cv2", "matplotlib", "scipy", "openpyxl"):
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        extra_datas.extend(pkg_datas)
        extra_binaries.extend(pkg_binaries)
        extra_hiddenimports.extend(pkg_hidden)
    except Exception:
        # Package not installed in the current build env; skip silently.
        pass

# Resources are bundled next to main.py in dev mode; PyInstaller must carry
# them along so resource_path() helpers inside the app still resolve at
# runtime (whether they're in sys._MEIPASS or alongside the frozen exe).
res_dir = os.path.join(os.getcwd(), "resources")
datas = list(extra_datas)
if os.path.isdir(res_dir):
    sep = ";" if IS_WINDOWS else ":"
    datas.append((res_dir, f"resources{sep}resources"))

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=list(extra_binaries),
    datas=datas,
    hiddenimports=[
        # openpyxl engine used by pandas for .xlsx output
        "openpyxl",
        # scipy submodules used indirectly by filtering/interpolation
        "scipy.signal",
        "scipy.spatial",
        "scipy.sparse",
        "scipy.ndimage",
        # matplotlib backends (try the common ones; matplotlib itself is
        # collected below with --collect-all)
        "matplotlib.backends.backend_agg",
        "matplotlib.backends.backend_tkagg",
        *extra_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Strip out heavy GUI toolkit bindings we definitely don't use,
        # keeping the generated single-file binary size smaller.
        "PyQt5",
        "PyQt6",
        "PySide2",
        "PySide6",
        "IPython",
        "notebook",
        "jupyter",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

kwargs = dict(
    name="BehavioralTracker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if IS_WINDOWS:
    icon_path = os.path.join("resources", "icon.ico")
    if os.path.isfile(icon_path):
        kwargs["icon"] = icon_path
elif IS_MACOS:
    icon_path = os.path.join("resources", "icon.icns")
    if os.path.isfile(icon_path):
        kwargs["icon"] = icon_path
    kwargs["bundle_identifier"] = "com.animalbehaviourtracker.app"

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    **kwargs,
)

if IS_MACOS:
    app = BUNDLE(
        exe,
        name="BehavioralTracker.app",
        icon=kwargs.get("icon"),
        bundle_identifier=kwargs.get("bundle_identifier"),
        info_plist={
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "LSApplicationCategoryType": "public.app-category.education",
            "CFBundleShortVersionString": "1.8.0",
            "CFBundleVersion": "1.8.0",
        },
    )
