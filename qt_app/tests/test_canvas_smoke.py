"""
Headless regression test for the Qt app's core UI: header, Quick Setup,
the interactive preview canvas (crop/mask/zones/objects/distance/maze-
template-rename), analysis-type switching, Results page navigation, and
Save/Open Project round-tripping. Covers the ground built in Task #23-25
of the PySide6 rewrite (see README.md's "Remaining roadmap").

Run directly (needs a virtual display -- see _pathsetup.py's docstring):
    xvfb-run -a python3.12 qt_app/tests/test_canvas_smoke.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd/QT_QPA_PLATFORM)

import sys, os, json, tempfile

from PySide6.QtWidgets import QApplication, QMessageBox, QLineEdit, QComboBox, QCheckBox, QLabel
from PySide6.QtCore import Qt

# Headless run: any real QMessageBox.exec() would block forever waiting for
# a click that never comes. Patch the static convenience methods (shared
# class object -- every module that imported QMessageBox sees this).
mb_calls = []
QMessageBox.information = staticmethod(lambda *a, **k: mb_calls.append(("information", a)) or QMessageBox.Ok)
QMessageBox.warning = staticmethod(lambda *a, **k: mb_calls.append(("warning", a)) or QMessageBox.Ok)
QMessageBox.critical = staticmethod(lambda *a, **k: mb_calls.append(("critical", a)) or QMessageBox.Ok)
QMessageBox.question = staticmethod(lambda *a, **k: mb_calls.append(("question", a)) or QMessageBox.Yes)

from qt_app.theme import build_stylesheet
from qt_app.main_window import MainWindow
from _pathsetup import VIDEO

PROJ_PATH = os.path.join(tempfile.mkdtemp(prefix="bt_qt_test_"), "qt_test_project.btproj")

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def widget_texts(widget):
    from PySide6.QtWidgets import QPushButton, QGroupBox
    out = []
    if isinstance(widget, QLabel):
        out.append(widget.text())
    if isinstance(widget, QPushButton):
        out.append(widget.text())
    if isinstance(widget, QGroupBox):
        out.append(widget.title())
    for child in widget.findChildren(QLabel):
        out.append(child.text())
    for child in widget.findChildren(QPushButton):
        out.append(child.text())
    for child in widget.findChildren(QGroupBox):
        out.append(child.title())
    return out


app = QApplication(sys.argv)
app.setStyleSheet(build_stylesheet())
win = MainWindow()
win.resize(1600, 980)
win.show()
app.processEvents()

# ------------------------------------------------------------------
# 1) Header
# ------------------------------------------------------------------
all_text = widget_texts(win)
check("Header has app brand text", any("BehavioralTracker" == t for t in all_text))
check("Header has Open Project button", "Open Project" in all_text)
check("Header has Save Project button", "Save Project" in all_text)
check("Header has Settings button", "Settings" in all_text)
check("Header has Help button", "Help" in all_text)
check("Header has Reset All button", "Reset All" in all_text)

check("analysis_type_buttons has 3 entries", len(win.analysis_type_buttons) == 3)
check("step_labels has setup+results", set(win.step_labels.keys()) == {"setup", "results"})
check("step tracker starts on 'setup' object name", win.step_labels["setup"].objectName() == "stepLabelActive")

# ------------------------------------------------------------------
# 2) Add a video (bypassing the file dialog) -> width/height captured
# ------------------------------------------------------------------
entry = win._probe_video(VIDEO)
check("_probe_video returns a dict", entry is not None)
check("_probe_video captured width/height", bool(entry and entry.get("width") and entry.get("height")))
win.videos.append(entry)
win.active_index = 0
win.setup_page.refresh_video_list()
app.processEvents()
info_text = win.setup_page.video_info_label.text()
print("video_info_label text:", info_text)
check("video info label shows resolution", "×" in info_text)
check("video info label shows fps", "fps" in info_text.lower())

# ------------------------------------------------------------------
# 3) Standard Tracking: Quick Setup grid present, No Crop + template works
# ------------------------------------------------------------------
check("analysis_type defaults to standard", win.analysis_type == "standard")
all_text = widget_texts(win.setup_page)
check("Quick Setup box present", any("Quick Setup" in t for t in all_text))
check("Custom Arena tile present", "Custom Arena" in all_text)
check("Animals in frame label present", any("Animals in frame" in t for t in all_text))

mb_calls.clear()
win.quick_setup_template("open_field")
check("quick_setup_template warns 'Arena needed' without a crop",
      any("Arena needed" in str(c[1]) for c in mb_calls))

win.on_tool_no_crop()
app.processEvents()
check("pending_matrix set after No Crop", win.pending_matrix is not None)

win.quick_setup_template("open_field")
app.processEvents()
check("quick_setup_template starts an interactive template_zones op (doesn't commit directly)",
      win._op is not None and win._op["kind"] == "template_zones")
check("pending_roi_points NOT yet set before Finish", win.pending_roi_points == {})
win.finish_op()
app.processEvents()
check("Finish commits quick_setup_template's zones", bool(win.pending_roi_points))
check("roi_names_entry synced with generated zone names",
      bool(win.setup_page.roi_names_entry.text().strip()))
check("_op cleared after Finish", win._op is None)

# ------------------------------------------------------------------
# 3b) Interactive Crop Arena: click 4 corners, drag one, Finish
# ------------------------------------------------------------------
win.on_reset_calibration_chamber()
app.processEvents()
check("Reset Calibration cleared _op", win._op is None)
check("Reset Calibration cleared pending_matrix", win.pending_matrix is None)

win.start_op("crop")
app.processEvents()
check("start_op('crop') creates a crop _op", win._op is not None and win._op["kind"] == "crop")
check("Crop tool button restyled active",
      win.setup_page.tool_buttons["crop"].objectName() == "toolBtnActive")

raw_w, raw_h = win.videos[0]["width"], win.videos[0]["height"]
corners = [(20, 20), (raw_w - 20, 20), (raw_w - 20, raw_h - 20), (20, raw_h - 20)]
for fx, fy in corners:
    win.on_canvas_press(fx, fy)
app.processEvents()
check("4 corner clicks recorded", win._op["shapes"][0] == corners)
check("op instructions mention 4/4 placed", "4/4" in win.op_instructions_text())

win.on_canvas_press(*corners[0])  # clicking an existing point starts a drag, not a 5th point
check("clicking an existing corner starts a drag instead of adding a 5th point",
      win._op["drag"] == ("shape", 0, 0) and len(win._op["shapes"][0]) == 4)
win.on_canvas_drag(30, 35)
check("drag moved the corner", win._op["shapes"][0][0] == (30, 35))
win.on_canvas_release()
check("release cleared the drag", win._op["drag"] is None)

mb_calls.clear()
win.finish_op()
app.processEvents()
check("finish_op('crop') committed a perspective matrix", win.pending_matrix is not None)
check("finish_op('crop') set pending_use_crop True", win.pending_use_crop is True)
check("finish_op('crop') recorded the (dragged) corners", win.pending_crop_corners[0] == (30, 35))
check("finish_op('crop') cleared _op", win._op is None)
check("finish_op('crop') restyled the crop button inactive",
      win.setup_page.tool_buttons["crop"].objectName() == "toolBtn")

# Back to a predictable identity-transform arena (same w/h as the raw
# frame) before the zones/objects/distance/mask tests below, so their
# hand-picked point coordinates stay safely in-bounds regardless of
# exactly where the perspective warp above put things.
win.on_tool_no_crop()
app.processEvents()

# ------------------------------------------------------------------
# 3c) Draw Zones: names required first; incomplete regions block Finish;
# region-switch buttons move the active region.
# ------------------------------------------------------------------
win.setup_page.roi_names_entry.setText("Light, Dark")
mb_calls.clear()
win.start_op("zones")
check("start_op('zones') builds one region per name",
      win._op is not None and set(win._op["regions"].keys()) == {"Light", "Dark"})
check("active_region defaults to the first name", win._op["active_region"] == "Light")

win.on_canvas_press(10, 10)
win.on_canvas_press(10, 50)
mb_calls.clear()
win.finish_op()
check("finish_op('zones') blocks a region with <3 points",
      any("still need 3+ points" in str(c[1]) for c in mb_calls))
check("_op stays open after a blocked Finish", win._op is not None)

win.on_canvas_press(50, 50)  # 'Light' now has 3 points
win.op_set_active_region("Dark")
check("op_set_active_region switched the active region", win._op["active_region"] == "Dark")
for fx, fy in [(200, 10), (240, 10), (240, 50)]:
    win.on_canvas_press(fx, fy)
mb_calls.clear()
win.finish_op()
check("finish_op('zones') committed both regions",
      set(win.pending_roi_points.keys()) == {"Light", "Dark"})
check("_op cleared after a successful Finish", win._op is None)

# ------------------------------------------------------------------
# 3d) Mask Zone: 'New Shape' starts a second masked region
# ------------------------------------------------------------------
win.start_op("mask")
for fx, fy in [(5, 5), (5, 30), (30, 30)]:
    win.on_canvas_press(fx, fy)
win.op_new_shape()
check("op_new_shape appended a second empty shape", len(win._op["shapes"]) == 2)
for fx, fy in [(100, 100), (100, 130), (130, 130)]:
    win.on_canvas_press(fx, fy)
win.finish_op()
check("finish_op('mask') committed 2 shapes (>=3 pts each)", len(win.pending_mask_points) == 2)

# ------------------------------------------------------------------
# 3e) Mark Objects: needs object names first
# ------------------------------------------------------------------
win.setup_page.object_names_entry.setText("Food")
win.start_op("objects")
for fx, fy in [(50, 50), (70, 50), (70, 70)]:
    win.on_canvas_press(fx, fy)
win.finish_op()
check("finish_op('objects') committed the object", "Food" in win.pending_object_points)

# ------------------------------------------------------------------
# 3f) Calibrate Distance
# ------------------------------------------------------------------
win.setup_page.real_distance_entry.setText("30")
win.setup_page.units_entry.setText("cm")
win.start_op("distance")
win.on_canvas_press(0, 0)
win.on_canvas_press(100, 0)
win.finish_op()
check("finish_op('distance') computed a scale factor", win.pending_scale_factor is not None)
check("finish_op('distance') scale factor is 30cm/100px = 0.3",
      win.pending_scale_factor is not None and abs(win.pending_scale_factor - 0.3) < 1e-6)
check("finish_op('distance') captured the units", win.pending_scale_unit == "cm")

# ------------------------------------------------------------------
# 3g) Cancel Op leaves pending_* untouched
# ------------------------------------------------------------------
saved_objects = dict(win.pending_object_points)
win.start_op("objects")
win.on_canvas_press(9, 9)
win.cancel_op()
check("cancel_op cleared _op", win._op is None)
check("cancel_op did not touch pending_object_points", win.pending_object_points == saved_objects)

# ------------------------------------------------------------------
# 3h) Maze Template zone rename via the explicit op-bar button (NOT a
# canvas click -- MM's video showed that on a real EPM recording, the
# arm/center zones overlap and are only a few px wide, so an ordinary
# alignment click easily landed inside a DIFFERENT zone and popped up its
# rename dialog by surprise. Canvas clicks are drag-only now; renaming
# only happens via rebuild_template_zone_buttons()'s per-zone button. The
# dialog itself is a blocking QDialog.exec(), so patch the module-level
# function headlessly.
# ------------------------------------------------------------------
import qt_app.dialogs.zone_label_dialog as zone_label_dialog
rename_calls = []


def fake_prompt(parent, current_name, suggestions, existing_names):
    rename_calls.append(current_name)
    return "Renamed Zone"


zone_label_dialog.prompt_zone_label = fake_prompt

win.quick_setup_template("open_field")
app.processEvents()
check("quick_setup_template (again) starts a template_zones op",
      win._op is not None and win._op["kind"] == "template_zones")
first_zone = next(iter(win._op["regions"].keys()))
first_pts = win._op["regions"][first_zone]
cx = sum(p[0] for p in first_pts) / len(first_pts)
cy = sum(p[1] for p in first_pts) / len(first_pts)
win.on_canvas_press(cx, cy)
check("clicking inside a generated zone's INTERIOR (away from any corner) does nothing now",
      rename_calls == [] and first_zone in win._op["regions"])
win.op_rename_template_zone(first_zone)
check("the per-zone op-bar button opened the rename dialog", rename_calls == [first_zone])
check("zone renamed inside the op",
      "Renamed Zone" in win._op["regions"] and first_zone not in win._op["regions"])
win.finish_op()
check("Finish committed the renamed zone", "Renamed Zone" in win.pending_roi_points)
check("roi_names_entry synced with the renamed zone",
      "Renamed Zone" in win.setup_page.roi_names_entry.text())

# ------------------------------------------------------------------
# 3h-2) Maze Template corner-dragging: a generated zone's geometry can be
# off from the real maze (camera angle/zoom per rig), so a click NEAR one
# of its corners should drag that corner -- same interaction as
# crop/zones/objects/distance. Start a fresh template_zones op (the
# previous one was already Finish()ed above).
# ------------------------------------------------------------------
rename_calls.clear()
win.quick_setup_template("open_field")
app.processEvents()
check("fresh template_zones op started for the drag test", win._op is not None and win._op["kind"] == "template_zones")
drag_zone = next(iter(win._op["regions"].keys()))
drag_pts_before = list(win._op["regions"][drag_zone])
corner_x, corner_y = drag_pts_before[0]
win.on_canvas_press(corner_x, corner_y)
check("clicking ON a generated zone's corner starts a drag",
      win._op["drag"] == ("region", drag_zone, 0) and rename_calls == [])
win.on_canvas_drag(corner_x + 15, corner_y - 15)
check("dragging moved that corner", win._op["regions"][drag_zone][0] == (corner_x + 15, corner_y - 15))
win.on_canvas_release()
check("release cleared the drag", win._op["drag"] is None)
win.finish_op()
check("Finish committed the dragged corner", win.pending_roi_points[drag_zone][0] == (corner_x + 15, corner_y - 15))

# ------------------------------------------------------------------
# 3h-3) Draw Zones: 'New Zone' adds an auto-named zone without needing to
# know all names up front, a click near ANY zone's corner drags it (not
# just the active one), and clicking INSIDE a different, already-finished
# zone opens the rename popup instead of adding a point to the active
# region -- the same "draw first, name after" capability Maze Template
# already had, now also in the manual Draw Zones tool.
# ------------------------------------------------------------------
win.setup_page.roi_names_entry.setText("A")
win.start_op("zones")
for fx, fy in [(300, 10), (340, 10), (340, 50), (300, 50)]:
    win.on_canvas_press(fx, fy)
check("'A' zone has 4 points before adding a second zone", len(win._op["regions"]["A"]) == 4)

win.op_new_zone()
check("op_new_zone added an auto-named 'Zone 1' and made it active",
      "Zone 1" in win._op["regions"] and win._op["active_region"] == "Zone 1")
for fx, fy in [(400, 10), (440, 10), (440, 50), (400, 50)]:
    win.on_canvas_press(fx, fy)
check("'Zone 1' now has 4 points", len(win._op["regions"]["Zone 1"]) == 4)

ax, ay = win._op["regions"]["A"][0]
win.on_canvas_press(ax, ay)
check("clicking A's corner while 'Zone 1' is active still starts a drag",
      win._op["drag"] == ("region", "A", 0))
win.on_canvas_drag(ax + 5, ay + 5)
win.on_canvas_release()
check("dragging moved A's corner even though it wasn't the active region",
      win._op["regions"]["A"][0] == (ax + 5, ay + 5))

rename_calls.clear()
a_pts = win._op["regions"]["A"]
acx = sum(p[0] for p in a_pts) / len(a_pts)
acy = sum(p[1] for p in a_pts) / len(a_pts)
win.on_canvas_press(acx, acy)
check("clicking inside a finished, non-active zone opens the rename popup",
      rename_calls == ["A"] and "Renamed Zone" in win._op["regions"])
check("renaming did NOT add a point to the still-active 'Zone 1'",
      len(win._op["regions"]["Zone 1"]) == 4)

win.finish_op()
check("Finish committed both freehand zones under their final names",
      set(win.pending_roi_points.keys()) == {"Renamed Zone", "Zone 1"})

# ------------------------------------------------------------------
# 3h-4) Draw Zones shape tools (Rectangle/Ellipse/Line) + snap-to-grid --
# MM asked for these after seeing ANY-maze's shape-drawing toolbar in his
# reference video. Existing corner-drag detection still takes priority
# over starting a new shape (so a committed shape can still be nudged),
# and once committed a shape is just an ordinary point list -- same
# mechanics as freehand.
# ------------------------------------------------------------------
win.setup_page.roi_names_entry.setText("Box, Oval, Arm")
win.start_op("zones")
check("a fresh zones op defaults to freehand, no snap",
      win._op["draw_mode"] == "freehand" and win._op["snap"] is False)

win.op_set_draw_mode("rectangle")
check("op_set_draw_mode switched to rectangle", win._op["draw_mode"] == "rectangle")
win.on_canvas_press(10, 10)
check("rectangle press starts a newshape drag, no point appended yet",
      win._op["drag"] == ("newshape", "Box") and win._op["regions"]["Box"] == [])
win.on_canvas_drag(60, 40)
check("rectangle drag live-builds a 4-point box",
      win._op["regions"]["Box"] == [(10, 10), (60, 10), (60, 40), (10, 40)])
win.on_canvas_release()
check("release clears the shape anchor and drag",
      win._op["drag"] is None and "_shape_anchor" not in win._op)

win.on_canvas_press(60, 40)
check("clicking ON a just-drawn rectangle's corner starts a corner-drag, not a new shape",
      win._op["drag"] == ("region", "Box", 2))
win.on_canvas_drag(65, 45)
win.on_canvas_release()
check("corner-drag still nudges the rectangle after it was committed",
      win._op["regions"]["Box"][2] == (65, 45))

win.op_set_active_region("Oval")
win.op_set_draw_mode("ellipse")
win.on_canvas_press(200, 10)
win.on_canvas_drag(240, 50)
win.on_canvas_release()
oval_pts = win._op["regions"]["Oval"]
check("ellipse drag produced a closed polygon inside its bounding box",
      len(oval_pts) >= 20 and all(200 <= p[0] <= 240 and 10 <= p[1] <= 50 for p in oval_pts))

win.op_set_active_region("Arm")
win.op_set_draw_mode("line")
win.on_canvas_press(300, 10)
win.on_canvas_drag(300, 110)
win.on_canvas_release()
arm_pts = win._op["regions"]["Arm"]
check("line/arm drag produced a thin 4-point corridor, not a bare 2-point line", len(arm_pts) == 4)
arm_xs = [p[0] for p in arm_pts]
check("the arm corridor has some width (not degenerate)", max(arm_xs) - min(arm_xs) > 1)
win.cancel_op()

win.start_op("zones")  # fresh op -- draw_mode/snap reset to defaults each time
check("draw_mode/snap reset to defaults on a fresh start_op",
      win._op["draw_mode"] == "freehand" and win._op["snap"] is False)
win.op_toggle_snap(True)
check("op_toggle_snap recorded snap on the op", win._op["snap"] is True)
spacing = win._op["grid_spacing"]
win.on_canvas_press(53, 47)
snapped_pt = win._op["regions"]["Box"][0]
expected = (round(53 / spacing) * spacing, round(47 / spacing) * spacing)
check(f"snap-to-grid rounded (53,47) to {expected}", snapped_pt == expected)
win.cancel_op()

# ------------------------------------------------------------------
# 3i) Real Qt mouse events (not direct on_canvas_press calls) through
# PreviewCanvas itself -- exercises the actual widget-pixel -> frame-pixel
# mapping (to_frame_xy) and mousePress/Move/ReleaseEvent handlers, i.e.
# what a real click in the app actually drives.
# ------------------------------------------------------------------
from PySide6.QtTest import QTest
from PySide6.QtCore import QPoint

canvas = win.setup_page.canvas
win.start_op("distance")
app.processEvents()
check("canvas has a positive scale after refresh", canvas.scale > 0)

# Click two widget points and check they land close to their expected
# frame-space positions (allowing for scale-factor rounding).
w1 = QPoint(int(canvas._off_x + 10 * canvas.scale), int(canvas._off_y + 10 * canvas.scale))
w2 = QPoint(int(canvas._off_x + 110 * canvas.scale), int(canvas._off_y + 10 * canvas.scale))
QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, w1)
QTest.mouseClick(canvas, Qt.LeftButton, Qt.NoModifier, w2)
app.processEvents()
check("2 real mouse clicks recorded 2 distance points", len(win._op["points"]) == 2)
p0, p1 = win._op["points"]
check("clicked frame points land near the intended (10,10)/(110,10)",
      abs(p0[0] - 10) < 2 and abs(p0[1] - 10) < 2 and abs(p1[0] - 110) < 2 and abs(p1[1] - 10) < 2)

# Drag via press/move/release at the QTest level too.
QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, w1)
w1_dragged = QPoint(w1.x() + int(5 * canvas.scale), w1.y())
QTest.mouseMove(canvas, w1_dragged)
QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, w1_dragged)
app.processEvents()
check("real mouse drag moved the first distance point", abs(win._op["points"][0][0] - 15) < 2)
check("real mouse release cleared the drag", win._op["drag"] is None)

win.cancel_op()
app.processEvents()

# ------------------------------------------------------------------
# 4) output_size_entry Combobox + memory persistence
# ------------------------------------------------------------------
check("output_size_entry is a QComboBox", isinstance(win.setup_page.output_size_entry, QComboBox))
check("output_size_entry defaults to 'Same as input'",
      win.setup_page.output_size_entry.currentText() == "Same as input")
win.setup_page.output_size_entry.setCurrentText("1280x720")
win.setup_page.save_all_to_memory(include_calibration=False)
check("Combobox value persisted to memory", win._memory.get("output_size_entry") == "1280x720")

# ------------------------------------------------------------------
# 5) Switch to Multi-Mouse and Behavior Classification, verify rebuild
# ------------------------------------------------------------------
win._set_analysis_type("multi_mouse")
app.processEvents()
check("analysis_type switched to multi_mouse", win.analysis_type == "multi_mouse")
check("multi_mouse button restyled active",
      win.analysis_type_buttons["multi_mouse"].objectName() == "analysisCardActive")
check("standard button restyled inactive",
      win.analysis_type_buttons["standard"].objectName() == "analysisCard")
all_text = widget_texts(win.setup_page)
check("Quick Setup still present for multi_mouse", any("Quick Setup" in t for t in all_text))

win._set_analysis_type("behavior")
app.processEvents()
all_text = widget_texts(win.setup_page)
check("No Quick Setup section in Behavior Classification", not any("Quick Setup" in t for t in all_text))
check("output_size_entry is a Combobox in behavior mode",
      isinstance(win.setup_page.output_size_entry, QComboBox))
check("ML classifier panel present in behavior mode",
      any("Deep Learning Classifier" in t for t in all_text))
check("Manual Scoring button present", any("MANUAL SCORING" in t for t in all_text))

# Interactive Crop Arena also works on Behavior Classification's smaller
# toolbar (crop/no-crop only) -- a fresh SetupPage.rebuild() means a fresh
# op_bar_layout too, so this also checks that wiring survives a rebuild.
win.start_op("crop")
app.processEvents()
check("start_op('crop') works after switching to Behavior Classification",
      win._op is not None and win._op["kind"] == "crop")
win.on_canvas_press(5, 5)
win.on_canvas_press(50, 5)
win.on_canvas_press(50, 50)
win.on_canvas_press(5, 50)
win.finish_op()
app.processEvents()
check("finish_op('crop') works in Behavior Classification too", win.pending_matrix is not None)
win.on_reset_calibration_chamber()
app.processEvents()

# ------------------------------------------------------------------
# 6) Results page navigation + step tracker
# ------------------------------------------------------------------
win.show_results_page()
app.processEvents()
check("stack shows results page", win.stack.currentWidget() is win.results_page)
check("step tracker highlights results", win.step_labels["results"].objectName() == "stepLabelActive")
check("step tracker un-highlights setup", win.step_labels["setup"].objectName() == "stepLabel")
win.show_setup_page()
app.processEvents()
check("step tracker back to setup", win.step_labels["setup"].objectName() == "stepLabelActive")

# ------------------------------------------------------------------
# 7) Save Project / Open Project round trip (patching file dialogs)
# ------------------------------------------------------------------
import PySide6.QtWidgets as QtWidgets

win._set_analysis_type("multi_mouse")
app.processEvents()
win.videos = [win._probe_video(VIDEO)]
win.active_index = 0
win.setup_page.refresh_video_list()
win.setup_page.on_num_animals(3)
win.batch_radio.setChecked(True)
win.setup_page.start_entry.setText("12.5")
app.processEvents()

QtWidgets.QFileDialog.getSaveFileName = staticmethod(lambda *a, **k: (PROJ_PATH, ""))
win.on_save_project()
check("project file written by on_save_project", os.path.exists(PROJ_PATH))
with open(PROJ_PATH) as f:
    saved = json.load(f)
check("saved file has num_animals=3", saved.get("memory", {}).get("num_animals") == 3)
check("saved file has start_entry=12.5", saved.get("memory", {}).get("start_entry") == "12.5")
check("saved file has mode=batch", saved.get("mode") == "batch")

win2 = MainWindow()
win2.show()
app.processEvents()
QtWidgets.QFileDialog.getOpenFileName = staticmethod(lambda *a, **k: (PROJ_PATH, ""))
win2.on_open_project()
app.processEvents()
check("Opened project restored analysis_type", win2.analysis_type == "multi_mouse")
check("Opened project restored mode", win2.get_mode() == "batch")
check("Opened project restored video queue", len(win2.videos) == 1 and win2.videos[0]["path"] == VIDEO)
check("Opened project restored num_animals via memory", win2.setup_page.num_animals == 3)
check("Opened project restored start_entry text", win2.setup_page.start_entry.text() == "12.5")

# ------------------------------------------------------------------
# 8) Reset handlers don't crash
# ------------------------------------------------------------------
try:
    win.on_reset_calibration_chamber()
    win.on_reset_video_chamber()
    win.on_reset_all()
    reset_ok = True
except Exception as exc:
    reset_ok = False
    import traceback; traceback.print_exc()
check("Reset handlers run without crashing", reset_ok)
check("Reset All cleared videos", win.videos == [])
check("Reset All cleared memory", win._memory == {})

win.close()
win2.close()

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
