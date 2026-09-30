"""
Regression test for a bug MM found from real screenshots: on the Results
page, the "Zone Occupancy" chart's longest bar (and its own "50.1s"-style
value label, printed just past the bar's tip) was getting silently cut off
-- invisible, not just hard to read -- whenever that chart rendered wider
than its 1/3 share of the Trajectory/Heatmap/Zone Occupancy row. Root
cause: save_zone_occupancy_chart's own figure is a FIXED 9in wide
regardless of how many zones it lists (see output/graphs.py), so a run
with few zones, or one dominant zone, renders proportionally WIDE; the
Results page was only scaling each chart image to a fixed HEIGHT
(scaledToHeight) and never capping its width, and a QLabel doesn't shrink
a pixmap bigger than its own rect -- it just clips it. Fixed in
qt_app/pages/results_page.py's show_standard by re-fitting each picture to
its label's REAL assigned width too, once the layout has actually run
(QTimer.singleShot(0, ...), same "fixed once, not resize-reactive"
trade-off already accepted for the original scaledToHeight-only version).

This reproduces the worst case directly (a 3-zone Zone Occupancy chart --
the fewer the zones, the WIDER the fixed-9in figure ends up relative to
its height, see save_zone_occupancy_chart's own figsize formula) rather
than running a full tracking pipeline, so it's fast and deterministic.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_results_chart_fit.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring/sets sys.path/cwd/QT_QPA_PLATFORM)

import os
import sys
import tempfile

from PySide6.QtGui import QPixmap, QColor
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtTest import QTest

import tracking.location as _loc  # import this first -- resolves the graphs.py<->location.py circular import
from output.graphs import save_zone_occupancy_chart
from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


PANEL_H = 300  # must match results_page.py's own panel_h

# QPixmap needs a QApplication to already exist, so build it before doing
# anything pixmap-related (the placeholders below, and the chart itself).
app = QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())

out_dir = tempfile.mkdtemp(prefix="results_chart_fit_")

# Trajectory/Heatmap stand-ins: plain square placeholders (aspect ~1) --
# real ones are roughly video-shaped, never anywhere near as wide,
# relative to their height, as a Zone Occupancy chart with few zones.
for fname in ("trajectory.png", "heatmap.png"):
    pm = QPixmap(400, 400)
    pm.fill(QColor(200, 200, 200))
    pm.save(os.path.join(out_dir, fname))

# The worst case from save_zone_occupancy_chart's own figsize formula:
# figsize=(9, max(3, 0.5*len(rows)+1.5)) -- FEWER zones means a SHORTER
# figure at the same fixed 9in width, i.e. a proportionally WIDER chart.
# 3 zones (the fewest that still sorts/labels meaningfully), one of them
# clearly dominant, mirrors exactly the screenshot MM sent (A/B/C, B the
# longest, its value label sitting past everything else).
summary = {"A_time_s": 18.8, "B_time_s": 50.1, "C_time_s": 28.9}
save_zone_occupancy_chart(summary, ["A", "B", "C"], out_dir)
zo_path = os.path.join(out_dir, "zone_occupancy.png")
check("zone_occupancy.png was generated for the test", os.path.exists(zo_path))

raw_pix = QPixmap(zo_path)
check("could load the generated chart as a QPixmap", not raw_pix.isNull())
naive_w = round(raw_pix.width() * (PANEL_H / raw_pix.height()))  # what scaledToHeight ALONE would give
check(f"sanity: this chart really IS wider than a typical single-panel "
      f"share once scaled to panel height (naive width={naive_w}px)",
      naive_w > 500)

win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()

full_summary = dict(summary, Output_folder=out_dir, Video="dummy.mp4",
                     Tracking_quality_percent=100.0, Total_transitions=5,
                     Total_distance_pixels=500.0)
win.results_page.show_standard(full_summary)
app.processEvents()
QTest.qWait(150)  # let the QTimer.singleShot(0, ...) re-fit actually fire
app.processEvents()


def picture_labels():
    # The picture panels (Trajectory/Heatmap/Zone Occupancy) share a
    # distinctive stylesheet not used anywhere else on this page (see
    # results_page.py's "border-radius: 6px" picture style) -- filter to
    # just those, then require an actual pixmap (excludes the "No result
    # images" empty-state label, which shares part of that same style but
    # never carries a pixmap).
    out = []
    for lbl in win.results_page.findChildren(QLabel):
        if "border-radius: 6px" in lbl.styleSheet() and lbl.pixmap() is not None and not lbl.pixmap().isNull():
            out.append(lbl)
    return out


pics = picture_labels()
check(f"all 3 picture panels present (Trajectory/Heatmap/Zone Occupancy), got {len(pics)}",
      len(pics) == 3)

for lbl in pics:
    pm = lbl.pixmap()
    check(f"a picture panel's final pixmap never exceeds its own label width "
          f"({pm.width()}px pixmap in a {lbl.width()}px label) -- nothing clipped",
          pm.width() <= lbl.width() + 1)
    check(f"a picture panel's final pixmap never exceeds the fixed panel height "
          f"({pm.height()}px pixmap vs {PANEL_H}px panel)",
          pm.height() <= PANEL_H + 1)

# Identify the wide one (Zone Occupancy) by aspect ratio and confirm the
# width-capping fix actually engaged for it specifically -- this is the
# one that was silently losing its longest bar's value label before.
wide = max(pics, key=lambda lbl: lbl.pixmap().width() / max(1, lbl.pixmap().height()))
wide_pm = wide.pixmap()
check(f"the wide (Zone Occupancy) panel's pixmap aspect ratio is clearly the widest of the 3 "
      f"({wide_pm.width()}x{wide_pm.height()})",
      wide_pm.width() / wide_pm.height() > 1.8)
check(f"the wide chart's final width was actually shrunk below the naive height-only scale "
      f"({wide_pm.width()}px vs naive {naive_w}px) -- proves the re-fit engaged, not just present",
      wide_pm.width() < naive_w)
check("the wide chart's final width fits its own label (the actual bug: it didn't, before this fix)",
      wide_pm.width() <= wide.width() + 1)

# The two near-square placeholders should NOT have been shrunk below their
# natural height-only scale (400x400 at panel_h=300 -> exactly 300x300) --
# confirms the fix doesn't over-shrink charts that already fit fine.
squares = [lbl for lbl in pics if lbl is not wide]
check("the two square placeholders were unaffected by the width cap (still exactly panel_h tall/wide)",
      all(lbl.pixmap().width() == PANEL_H and lbl.pixmap().height() == PANEL_H for lbl in squares))

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
