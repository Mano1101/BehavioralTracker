"""
Headless regression test for the "Quick Setup" apparatus-template flow.

Replaces test_maze_template_dialog.py (the old, now-deleted Maze Template
dialog -- see MainWindow.quick_setup_template's docstring for the full
story): MM asked for the old "auto-generate a default-sized shape per zone,
then drag its corners to align" approach to be replaced everywhere with the
SAME draw-it-yourself-then-name-it interaction Draw Zones already uses,
since tracing the real photographed maze by hand lines up better than
dragging an idealized default shape into place ever did, and a whole
separate dialog for picking an apparatus and tuning its size params is
redundant once nothing generates a shape from those params anymore.

Covers: the arena-needed guard, a Quick Setup tile starting a normal
'zones' op tagged with its template key, each drawn shape getting prompted
with that apparatus's next unused zone name IN ORDER (Center, Open Arm 1,
...), falling back to a plain 'Zone N' once a template's own suggestions
are all used, a bare 'Custom Arena' tile (no template) behaving exactly
like plain Draw Zones, and that the old dialog/button/module are really
gone -- not just unused.

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_maze_templates.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import sys

from PySide6.QtWidgets import QApplication, QMessageBox

mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

import qt_app.dialogs.zone_label_dialog as zone_label_dialog

# Accept whatever name is pre-filled (current_name) every time -- lets this
# test verify _next_auto_zone_name's suggestion ORDER by checking what
# actually ends up in pending_roi_points/accept_calls, rather than having
# to hand-simulate picking a specific dropdown item.
accept_calls = []


def _accept_current(parent, current_name, suggestions, existing_names):
    accept_calls.append((current_name, list(suggestions)))
    return current_name


zone_label_dialog.prompt_zone_label = _accept_current

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from tracking.maze_templates import TEMPLATES as MAZE_TEMPLATES
from _pathsetup import VIDEO

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def draw_line(win, x0, y0, x1, y1):
    win.on_canvas_press(x0, y0)
    win.on_canvas_drag(x1, y1)
    win.on_canvas_release()


app = QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()

entry = win._probe_video(VIDEO)
win.videos.append(entry)
win.active_index = 0
win.setup_page.refresh_video_list()
app.processEvents()

# ------------------------------------------------------------------
# 1) "Arena needed" guard -- no crop/no-crop set yet.
# ------------------------------------------------------------------
mb_calls.clear()
win.quick_setup_template("epm")
check("no arena -> critical dialog shown", any(k == "critical" for k, _ in mb_calls))
check("no arena -> no op started", win._op is None)

win.on_tool_no_crop()
app.processEvents()

# ------------------------------------------------------------------
# 2) Happy path: an EPM Quick Setup tile starts a normal 'zones' op (NOT
# the old drag-corners 'template_zones' op), tagged with the template key,
# with its first auto-named placeholder pre-named the template's own
# FIRST suggested zone label.
# ------------------------------------------------------------------
epm_labels = [label for label, _pts in MAZE_TEMPLATES["epm"]["generate"](
    win.pending_warp_w, win.pending_warp_h,
    {p[0]: p[2] for p in MAZE_TEMPLATES["epm"]["params"]})]

mb_calls.clear()
win.quick_setup_template("epm")
app.processEvents()
check("quick_setup_template starts a 'zones' op (not the old template_zones)",
      win._op is not None and win._op["kind"] == "zones")
check("op is tagged with the template key", win._op.get("template_key") == "epm")
check("first auto-named placeholder is the template's first suggested label",
      list(win._op["regions"].keys()) == [epm_labels[0]]
      and epm_labels[0] in win._op["auto_named"])

# ------------------------------------------------------------------
# 3) Draw each EPM zone in turn (Line/Arm tool), accepting whatever name
# is pre-filled each time -- should walk straight through epm_labels IN
# ORDER, one per shape, via _next_auto_zone_name.
# ------------------------------------------------------------------
win.op_set_draw_mode("line")
ww, wh = win.pending_warp_w, win.pending_warp_h
cx, cy = ww // 2, wh // 2
accept_calls.clear()
offsets = [(-80, 0), (80, 0), (0, -80), (0, 80), (0, 0)]
for dx, dy in offsets[:len(epm_labels)]:
    draw_line(win, cx + dx - 5, cy + dy - 15, cx + dx + 5, cy + dy + 15)
    app.processEvents()

check("prompted once per zone, in the template's own suggestion order",
      [c[0] for c in accept_calls] == epm_labels)
check("every prompt offered the full template suggestion list",
      all(set(c[1]) == set(epm_labels) for c in accept_calls))
check("all 5 EPM zones ended up drawn in the op's regions",
      set(win._op["regions"].keys()) >= set(epm_labels))
check("after all 5 suggestions are used, the next auto placeholder falls back to 'Zone 1'",
      "Zone 1" in win._op["regions"] and "Zone 1" in win._op["auto_named"]
      and win._op["active_region"] == "Zone 1")

win.finish_op()
app.processEvents()
check("Finish committed exactly the 5 EPM zones (the empty 'Zone 1' placeholder was dropped)",
      set(win.pending_roi_points.keys()) == set(epm_labels))
check("roi_names_entry synced to the same names",
      set(n.strip() for n in win.setup_page.roi_names_entry.text().split(",") if n.strip())
      == set(epm_labels))

# ------------------------------------------------------------------
# 4) A bare 'Custom Arena' tile (template_key=None, what that tile calls)
# behaves exactly like plain Draw Zones always has -- no suggestions,
# plain 'Zone 1'.
# ------------------------------------------------------------------
win.setup_page.roi_names_entry.setText("")  # otherwise step 3's names would carry over as typed names
win.start_op("zones")
check("plain Draw Zones op has no template_key", win._op.get("template_key") is None)
check("plain Draw Zones starts with a generic 'Zone 1' placeholder",
      list(win._op["regions"].keys()) == ["Zone 1"])
win.cancel_op()
app.processEvents()

# ------------------------------------------------------------------
# 5) The old dialog is really gone -- no leftover entry point, no
# importable module either.
# ------------------------------------------------------------------
check("on_open_maze_template_dialog no longer exists on MainWindow",
      not hasattr(win, "on_open_maze_template_dialog"))
check("start_template_zone_op no longer exists on MainWindow (dead code removed)",
      not hasattr(win, "start_template_zone_op"))
check("no 'Maze Template' tool button in the Setup page toolbar anymore",
      "maze" not in getattr(win.setup_page, "tool_buttons", {}))
try:
    import qt_app.dialogs.maze_template_dialog  # noqa: F401
    module_gone = False
except ModuleNotFoundError:
    module_gone = True
check("qt_app/dialogs/maze_template_dialog.py was actually deleted, not just unused", module_gone)

win.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
