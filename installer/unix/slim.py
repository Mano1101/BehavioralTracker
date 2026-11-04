"""Remove what the app never uses from the installed packages (test suites, developer files, Qt tools)."""
import shutil
import sys
from pathlib import Path

site = Path(sys.argv[1])
gone = 0


def rm(p):
    global gone
    try:
        if p.is_dir() and not p.is_symlink():
            gone += sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists() or p.is_symlink():
            gone += p.stat().st_size if p.is_file() else 0
            p.unlink()
    except OSError:
        pass


for d in list(site.rglob("tests")) + list(site.rglob("test")):          # package test suites
    if d.is_dir() and "PySide6" not in d.parts:
        rm(d)
for f in site.rglob("*.pyi"):                                              # type stubs (editor-only)
    rm(f)
ps = site / "PySide6"
for rel in ("include", "typesystems", "glue", "examples", "scripts", "doc", "Qt/qml", "Qt/translations",
            "Qt/metatypes", "Qt/mkspecs", "Qt/libexec", "Qt/plugins/designer", "Qt/plugins/qmltooling",
            "Qt/plugins/qmllint", "Qt/plugins/qmlls"):
    rm(ps / rel)
for exe in ("designer", "assistant", "linguist", "lrelease", "lupdate", "qmlformat", "qmllint", "qmlls",
            "Designer.app", "Assistant.app", "Linguist.app", "balsam", "balsamui", "qsb", "svgtoqml"):
    rm(ps / exe)
for mpl_data in (site / "matplotlib" / "mpl-data" / "sample_data",):
    rm(mpl_data)
print(f"slimmed {gone / 1e6:.0f} MB")
