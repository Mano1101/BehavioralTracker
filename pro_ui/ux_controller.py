"""BehavioralTracker 1.0.0 -- the UX file.

Loads the Qt-Designer layout `behavioral_tracker.ui` (unchanged, purely
visual) and gives every screen its behaviour on top of the tracking /
analysis / output engine:

    Dashboard   -> live stats + recent experiments (persisted history), test tiles
    Experiment  -> video / camera source, subjects, experimental design
    Video Setup -> interactive zone editor:
                     * draw zones: polygon (click), freehand trace (drag),
                       rectangle, ellipse / circle
                     * edit ANY zone incl. presets: drag points, add point
                       (double-click an edge), delete point (right-click),
                       move / resize / rotate one zone or the whole layout
                     * auto zone detection (one click) and "trace the
                       apparatus outline -> zones"
                     * auto-mask everything outside the zones
                     * crop (4 corners, zones are carried over), masks,
                       scale calibration, zoom (wheel) and pan (middle drag)
                     * undo / redo, save / load zone layouts
                     * chambers: up to 4 separate arenas in one video, tracked at
                       the same time (pro_ui/chambers_ui.py, tracking/chambers.py)
    Tracking    -> robust detection engine (or Classic), animal
                   colour, test-detection preview, live view, start / stop
    Analysis    -> KPIs per test (EPM open-arm %, Y-maze alternation, LD
                   transitions, OF centre time, NOR discrimination ...),
                   zone table, arm-entry sequence, charts, export
    Batch       -> queue, per-video zones and time windows, optional
                   auto-fit of zones per video, progress + ETA, summary
    Freezing    -> freezing analysis (pro_ui/freezing_page.py): Freezing - Individual and
                   Freezing - Batch pages; results on the Analysis page (freezing_results.py), .btf files
    Event Scoring -> manual keyboard scoring (always the last side-bar item)

Grooming / rearing / behaviour classification were removed on purpose.
"""
import copy
import datetime
import json
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # charts are rendered off-screen (worker-thread safe)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import (QDate, QDir, QEvent, QFile, QObject, QPointF, QRectF, Qt, QThread,  # noqa: E402
                            QSize, QTimer, QUrl, Signal, Slot)
from PySide6.QtGui import (QBrush, QColor, QCursor, QDesktopServices, QFont, QIcon, QImage, QKeySequence,  # noqa: E402
                           QPainter, QPainterPath, QPen, QPixmap, QShortcut)
from PySide6.QtUiTools import QUiLoader  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDateEdit,  # noqa: E402
                               QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                               QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressBar, QProgressDialog, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSpinBox,
                               QStackedWidget, QStyle, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextBrowser, QToolButton, QVBoxLayout, QWidget)

APP_VERSION = "1.0.0"
def _app_data_dir():
    """Settings, experiment history and zone layouts of this Windows user.

    Windows: %APPDATA%\\BehavioralTracker (kept when the program is updated; the
    uninstaller offers to delete it).  Other systems: ~/.behavioraltracker.
    BEHAVIORALTRACKER_HOME overrides both (portable use, tests)."""
    override = os.environ.get("BEHAVIORALTRACKER_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "BehavioralTracker"
    return Path.home() / ".behavioraltracker"


APP_DIR = _app_data_dir()
HISTORY_PATH = APP_DIR / "history.json"
SETTINGS_PATH = APP_DIR / "pro_settings.json"
LAYOUT_DIR = APP_DIR / "zone_layouts"
LEARNED_DIR = APP_DIR / "learned_layouts"       # set-ups ⚡ Auto has learned from your own videos
_LEARN_LOCK = threading.Lock()
INVENTORY_PATH = APP_DIR / "behavioral_inventory.json"   # last-used event-scoring behavioral inventory
INVENTORY_DIR = APP_DIR / "behavioral_inventories"
VIDEO_FILTER = "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg);;All files (*)"
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".mpg", ".mpeg"}

# Live tracking (delayed: the trial is tracked in pieces while it is recorded -- tracking/live_delayed.py).
# Set the environment variable BT_LIVE_TRACKING=0 to hide it.
LIVE_TRACKING = os.environ.get("BT_LIVE_TRACKING", "1").strip() != "0"

# test tile key -> (label, emoji, description, maze_templates key or None)
TESTS = {
    "open_field": ("Open Field", "🐭", "Locomotor activity, anxiety-like behaviour and exploration: "
                   "distance, speed, centre time and centre entries.", "open_field"),
    "epm": ("Elevated Plus Maze", "✚", "Anxiety-like behaviour: time and entries in open versus closed "
            "arms (open-arm %, open-arm entry %).", "epm"),
    "light_dark": ("Light / Dark Box", "◐", "Aversion to light: time in each compartment, latency to the "
                   "dark side and light-dark transitions.", None),
    "nor": ("Novel Object Recognition", "🧊", "Recognition memory: exploration time around each object "
            "and the discrimination index (name the zones 'Novel' / 'Familiar').", "open_field"),
    "y_maze": ("Y-Maze", "Y", "Spatial working memory: arm entries and spontaneous alternation "
               "(plus same-arm and alternate-arm returns).", "y_maze"),
    "t_maze": ("T-Maze", "T", "Choice behaviour: stem and goal-arm entries and goal-arm alternation.",
               "t_maze"),
    "social": ("Social Interaction", "👥", "Three-chamber sociability: time in each chamber and the "
               "sociability index (name the sides 'Social' / 'Object').", "social_3_chamber"),
    "custom": ("Custom Test", "✎", "Draw your own zones for any other apparatus; every zone gets time, "
               "entries, latency and distance.", None),
}

# name suggestions offered when a zone is drawn or renamed
ZONE_SUGGESTIONS = {
    "open_field": ["Center", "Periphery", "Corner 1", "Corner 2", "Corner 3", "Corner 4", "Wall zone"],
    "epm": ["Center", "Open Arm 1", "Open Arm 2", "Closed Arm 1", "Closed Arm 2"],
    "light_dark": ["Light", "Dark", "Door"],
    "nor": ["Novel Object", "Familiar Object", "Object A", "Object B", "Center"],
    "y_maze": ["Center", "Arm A", "Arm B", "Arm C"],
    "t_maze": ["Center", "Stem", "Goal Arm Left", "Goal Arm Right", "Start box"],
    "social": ["Social Chamber", "Center", "Object Chamber", "Social zone", "Object zone"],
    "custom": ["Zone 1", "Zone 2", "Center", "Periphery"],
}

ZONE_PALETTE = ["#f5b800", "#3b82f6", "#22c55e", "#ef4444", "#a855f7", "#06b6d4", "#f97316",
                "#ec4899", "#84cc16", "#14b8a6", "#8b5cf6", "#64748b"]
STATUS_COLORS = {"Completed": "#16a34a", "Processing": "#1f6feb", "Pending": "#64748b",
                 "Failed": "#dc2626", "Stopped": "#d97706"}
ENGINES = [("robust", "Robust (recommended)"), ("classic", "Classic")]
POLARITIES = [("auto", "Auto-detect"), ("dark", "Darker than floor (black mouse)"),
              ("light", "Lighter than floor (white mouse)"), ("any", "Either (absolute difference)")]
LOW_LIGHT_CHOICES = [("off", "Off"), ("auto", "Auto (dark zones)"), ("low", "Low"), ("medium", "Medium"),
                     ("high", "Strong (≈5 lux)")]
DARK_BOX_CHOICES = [("covered", "Covered (lid)"), ("open", "Open (dim light)")]


# ============================================================================= helpers
def find(win, cls, name):
    w = win.findChild(cls, name)
    if w is None:
        raise RuntimeError(f"Widget '{name}' ({cls.__name__}) not found in the .ui file")
    return w


def fmt_clock(sec, hours=False):
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if (hours or h) else f"{m:02d}:{s:02d}"


def fmt_clock_exact(sec):
    """00:01:02.50 -- exactly the text written on the live picture (tracking.location.clock_text)."""
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m = rem // 60
    return f"{h:02d}:{m:02d}:{sec - h * 3600 - m * 60:05.2f}"


TEST_BTN_TEXT = "🔍  Test detection (at the start time)"


def fmt_track_pos(t, frame, end_t, end_frame):
    """Clock under the Tracking picture: the same time + frame as written on the picture."""
    return f"{fmt_clock_exact(t)} · frame {int(frame)}   /   {fmt_clock_exact(end_t)} · frame {int(end_frame)}"


def fmt_duration(sec):
    """'2 min 10 s' / '45 s' / '1 h 05 min' for a remaining time."""
    sec = max(0, int(round(sec)))
    if sec < 60:
        return f"{sec} s"
    if sec < 3600:
        return f"{sec // 60} min {sec % 60:02d} s"
    return f"{sec // 3600} h {sec % 3600 // 60:02d} min"


def bgr_to_qimage(bgr):
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def save_json(path, data):
    try:
        os.makedirs(os.path.dirname(str(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def poly_area(pts):
    if len(pts) < 3:
        return 0.0
    a = np.asarray(pts, float)
    return 0.5 * abs(float(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1))))


def poly_centroid(pts):
    a = np.asarray(pts, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def point_in_poly(x, y, pts):
    if len(pts) < 3:
        return False
    return cv2.pointPolygonTest(np.asarray(pts, np.float32), (float(x), float(y)), False) >= 0


from tracking.fragment import label_point, outline_arrays, poly_rings  # noqa: E402,F401
from tracking.location import results_root  # noqa: E402


def seg_distance(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy), (qx, qy)


def unique_name(base, existing):
    if base not in existing:
        return base
    i = 2
    while f"{base} {i}" in existing:
        i += 1
    return f"{base} {i}"


def ellipse_points(x0, y0, x1, y1, n=40):
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    rx, ry = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
    return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]


SHAPE_TOOLS = ("rect", "square", "ellipse", "circle", "triangle")


def shape_points(tool, ax, ay, x1, y1, shift=False):
    """Standard shape dragged from (ax, ay) to (x1, y1).  Shift = regular (square / circle / equilateral)."""
    if tool in ("square", "circle") or (shift and tool in ("rect", "ellipse")):
        s = max(abs(x1 - ax), abs(y1 - ay))
        x1, y1 = ax + math.copysign(s, (x1 - ax) or 1.0), ay + math.copysign(s, (y1 - ay) or 1.0)
    if tool in ("rect", "square"):
        return [(ax, ay), (x1, ay), (x1, y1), (ax, y1)]
    if tool in ("ellipse", "circle"):
        return ellipse_points(ax, ay, x1, y1)
    # triangle: apex at the middle of the side where the drag started, base on the other side
    if shift:
        wdt = abs(x1 - ax)
        y1 = ay + math.copysign(wdt * math.sqrt(3) / 2.0, (y1 - ay) or 1.0)
    mx = (ax + x1) / 2.0
    return [(mx, ay), (x1, y1), (ax, y1)]


SHAPE_MENU = (("Rectangle — drag", "rect"), ("Square — drag", "square"), ("Circle — drag", "circle"),
              ("Ellipse — drag", "ellipse"), ("Triangle — drag (Shift = equilateral)", "triangle"),
              ("Polygon — click points", "polygon"), ("Freehand — trace with the mouse", "freehand"))


def default_zones(test_key, w, h):
    """Preset zone template per test, centred in a w x h frame (pixel coords)."""
    from tracking.maze_templates import TEMPLATES
    if test_key == "light_dark":
        half = w / 2.0
        return {"Light": [(0, 0), (half - 2, 0), (half - 2, h - 1), (0, h - 1)],
                "Dark": [(half + 2, 0), (w - 1, 0), (w - 1, h - 1), (half + 2, h - 1)]}
    if test_key == "custom":
        return {}
    tpl = TEMPLATES[TESTS[test_key][3]]
    scale = min(w, h) / 500.0
    params = {k: (d * scale if kind == "length" else d) for k, _lbl, d, kind in tpl["params"]}
    zones = {name: [(float(x), float(y)) for x, y in pts] for name, pts in tpl["generate"](w, h, params)}
    if test_key == "nor":
        r = min(w, h) * 0.08
        for name, fx in (("Object A", 0.3), ("Object B", 0.7)):
            zones[name] = ellipse_points(w * fx - r, h / 2 - r, w * fx + r, h / 2 + r, 32)
    return zones


def zone_suggestions(test_key, existing=()):
    sug = list(ZONE_SUGGESTIONS.get(test_key, ZONE_SUGGESTIONS["custom"]))
    free = [s for s in sug if s not in existing]
    return free + [s for s in sug if s in existing]


# ============================================================================= simple image view
class FrameView(QObject):
    """Aspect-correct image display inside a QLabel (charts, tracking view)."""
    pressed = Signal(float, float, bool)
    double_clicked = Signal()

    def __init__(self, label, parent=None):
        super().__init__(parent)
        self.label = label
        label.setMinimumSize(1, 1)
        label.setAlignment(Qt.AlignCenter)
        label.installEventFilter(self)
        self._img = None
        self._geo = None
        self._placeholder = label.text()

    def set_bgr(self, bgr):
        self._img = bgr_to_qimage(bgr)
        self._render()

    def set_qimage(self, img):
        self._img = img
        self._render()

    def set_file(self, path):
        img = QImage(str(path))
        if img.isNull():
            self.clear()
        else:
            self.set_qimage(img)

    def clear(self, text=None):
        self._img = None
        self._geo = None
        self.label.setPixmap(QPixmap())
        self.label.setText(self._placeholder if text is None else text)

    def _render(self):
        if self._img is None:
            return
        lw, lh = max(self.label.width(), 1), max(self.label.height(), 1)
        pm = QPixmap.fromImage(self._img).scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._geo = ((lw - pm.width()) / 2.0, (lh - pm.height()) / 2.0, pm.width() / max(1, self._img.width()))
        self.label.setPixmap(pm)

    def eventFilter(self, obj, ev):
        if obj is self.label:
            t = ev.type()
            if t == QEvent.Resize:
                QTimer.singleShot(0, self._render)
            elif t == QEvent.MouseButtonPress and self._geo and self._img is not None:
                ox, oy, s = self._geo
                p = ev.position()
                fx, fy = (p.x() - ox) / s, (p.y() - oy) / s
                if 0 <= fx < self._img.width() and 0 <= fy < self._img.height():
                    self.pressed.emit(fx, fy, ev.button() == Qt.RightButton)
            elif t == QEvent.MouseButtonDblClick:
                self.double_clicked.emit()
        return False


class Player:
    """Thin cv2.VideoCapture wrapper: sequential reads + (verified) random access."""

    def __init__(self):
        self.cap = None
        self.path = None
        self.fps, self.total, self.w, self.h, self.pos = 30.0, 0, 0, 0, 0

    def open(self, path):
        self.close()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        self.cap, self.path = cap, str(path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 0 < fps < 1000 else 30.0
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.w, self.h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = 0
        return self.total > 0 and self.w > 0

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.path = None, None

    @property
    def duration(self):
        return self.total / self.fps if self.fps else 0.0

    @property
    def current(self):
        return max(0, self.pos - 1)

    def seek(self, idx):
        if self.cap is None:
            return None
        idx = int(min(max(idx, 0), max(self.total - 1, 0)))
        if idx == self.pos:                       # already there: sequential read is exact
            return self.next()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok and idx > 0:                    # some files report one frame too many
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            ok, frame = self.cap.read()
            idx -= 1
        self.pos = idx + 1
        return frame if ok else None

    def next(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        return frame


# ============================================================================= zone canvas
class ZoneCanvas(QWidget):
    """Interactive video frame + zone editor.

    Owns the zones, masks and selection.  Everything is stored in frame
    (video / cropped-arena) pixel coordinates.

    Mouse (no tool active):  click a zone = select (Shift = add), drag =
    move, drag a point = reshape, double-click an edge = add point,
    right-click a point = delete point, handles = resize / rotate the
    selection (Shift keeps proportions / snaps 15 deg), wheel = zoom,
    middle-drag = pan, right-click = menu.
    Drawing tools: polygon (click points; double-click / Enter / click the
    first point to close), freehand (press and drag), rectangle and ellipse
    (drag; Shift = square / circle), points (crop corners, scale ends).
    """

    zonesEdited = Signal()
    selectionChanged = Signal()
    drawingFinished = Signal(str, object)        # purpose, points
    drawingCancelled = Signal()
    zoneRenameRequested = Signal(str)
    contextRequested = Signal(object, object)    # zone name or None, global QPoint
    hoverInfo = Signal(str)
    draftChanged = Signal()                      # a point was added / moved / removed while drawing

    HANDLE = 6.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(440, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._img = None
        self.placeholder = "Load a video on the Experiment page"
        self.zones = {}
        self.colors = {}
        self.masks = []
        self.auto_mask = True
        self.locked = False
        self.selected = []
        self.hover = None
        self.tool = None
        self.purpose = None
        self.max_points = None
        self.draft = []
        self._drag = None
        self._cursor_f = None
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.undo_stack, self.redo_stack = [], []
        self.banner = ""
        self.point_labels = None    # labels drawn next to clicked points (guided fit)
        self.covered_fn = None      # -> set of covered zone names (the animal is hidden inside them)
        self.point_prompts = None   # banner text for the next point to click
        self.auto_finish = False
        self.marker = None          # (x, y) e.g. detection preview point
        self._color_i = 0
        self.chamber_rects = []     # [x0, y0, x1, y1] of each chamber (chamber mode, see chambers_ui)
        self.extra_paint = None     # fn(painter): chamber frames drawn over the zones
        self.extra_inside = None    # fn() -> rectangles the auto-mask preview leaves undimmed

    # ------------------------------------------------------------------ data
    def set_frame(self, bgr):
        self._img = None if bgr is None else bgr_to_qimage(bgr)
        self.update()

    @property
    def frame_size(self):
        return (self._img.width(), self._img.height()) if self._img is not None else (0, 0)

    def color_of(self, name):
        if name not in self.colors:
            self.colors[name] = QColor(ZONE_PALETTE[self._color_i % len(ZONE_PALETTE)])
            self._color_i += 1
        return self.colors[name]

    def snapshot(self):
        return (copy.deepcopy(self.zones), copy.deepcopy(self.masks))

    def push_undo(self, snap=None):
        self.undo_stack.append(snap if snap is not None else self.snapshot())
        self.undo_stack = self.undo_stack[-80:]
        self.redo_stack.clear()

    def undo(self):
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.snapshot())
        self.zones, self.masks = self.undo_stack.pop()
        self.selected = [n for n in self.selected if n in self.zones]
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()
        return True

    def redo(self):
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.snapshot())
        self.zones, self.masks = self.redo_stack.pop()
        self.selected = [n for n in self.selected if n in self.zones]
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()
        return True

    def set_zones(self, zones, record=True, select_all=False):
        if record:
            self.push_undo()
        self.zones = {k: [(float(x), float(y)) for x, y in v] for k, v in zones.items() if len(v) >= 3}
        for n in self.zones:
            self.color_of(n)
        self.selected = list(self.zones) if select_all else [n for n in self.selected if n in self.zones]
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()

    def add_zone(self, name, pts, select=True):
        self.push_undo()
        self.zones[name] = [(float(x), float(y)) for x, y in pts]
        self.color_of(name)
        if select:
            self.selected = [name]
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()

    def rename_zone(self, old, new):
        if old not in self.zones or not new or new == old or new in self.zones:
            return False
        self.push_undo()
        self.zones = {(new if k == old else k): v for k, v in self.zones.items()}
        self.colors[new] = self.colors.pop(old, self.color_of(new))
        self.selected = [new if n == old else n for n in self.selected]
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()
        return True

    def delete_zones(self, names):
        names = [n for n in names if n in self.zones]
        if not names:
            return
        self.push_undo()
        for n in names:
            self.zones.pop(n, None)
        self.selected = [n for n in self.selected if n in self.zones]
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()

    def duplicate_selected(self):
        if not self.selected:
            return
        self.push_undo()
        new_sel = []
        off = 12.0 / max(self._view()[2], 1e-6)
        for n in list(self.selected):
            nn = unique_name(n + " copy", self.zones)
            self.zones[nn] = [(x + off, y + off) for x, y in self.zones[n]]
            self.color_of(nn)
            new_sel.append(nn)
        self.selected = new_sel
        self.update()
        self.zonesEdited.emit()
        self.selectionChanged.emit()

    def select(self, names):
        self.selected = [n for n in names if n in self.zones]
        self.update()
        self.selectionChanged.emit()

    def transform_points(self, fn):
        """Apply fn((x, y)) -> (x, y) to every zone and mask point (crop changes)."""
        self.zones = {k: [fn(p) for p in v] for k, v in self.zones.items()}
        self.masks = [[fn(p) for p in m] for m in self.masks]
        for i, (x0, y0, x1, y1) in enumerate(list(self.chamber_rects)):
            q = [fn(pt) for pt in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
            self.chamber_rects[i] = [min(a for a, _b in q), min(b for _a, b in q), max(a for a, _b in q),
                                     max(b for _a, b in q)]
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update()
        self.zonesEdited.emit()

    # ------------------------------------------------------------------ tools
    def start_tool(self, tool, purpose, max_points=None, banner="", labels=None, prompts=None,
                   auto_finish=False):
        self.tool, self.purpose, self.max_points = tool, purpose, max_points
        self.draft = []
        self._drag = None
        self.banner = banner
        self.point_labels, self.point_prompts, self.auto_finish = labels, prompts, bool(auto_finish)
        self.setCursor(Qt.CrossCursor)
        self.setFocus()
        self.update()

    def cancel_tool(self, silent=False):
        was = self.tool is not None
        self.tool, self.purpose, self.max_points = None, None, None
        self.draft = []
        self._drag = None
        self.banner = ""
        self.point_labels, self.point_prompts, self.auto_finish = None, None, False
        self.unsetCursor()
        self.update()
        if was and not silent:
            self.drawingCancelled.emit()

    def current_banner(self):
        """Banner text; in a guided points tool it names the next point to click."""
        if self.tool == "points" and self.point_prompts:
            k = len(self.draft)
            if self.max_points and k >= self.max_points:
                return "All points placed — drag a point to adjust it, then Enter / ✓ Finish."
            return self.point_prompts[min(k, len(self.point_prompts) - 1)]
        return self.banner

    def finish_tool(self):
        if self.tool is None:
            return False
        need = self.max_points if self.tool == "points" else 3
        if len(self.draft) < (need or 3):
            return False
        pts, purpose = list(self.draft), self.purpose
        self.cancel_tool(silent=True)
        self.drawingFinished.emit(purpose, pts)
        return True

    def undo_point(self):
        if self.draft:
            self.draft.pop()
            self.update()
            self.draftChanged.emit()

    # ------------------------------------------------------------------ view math
    def _view(self):
        iw, ih = self.frame_size
        if iw == 0:
            return 0.0, 0.0, 1.0
        W, H = self.width(), self.height()
        s0 = min(W / iw, H / ih)
        sc = s0 * self.zoom
        ox = (W - iw * sc) / 2.0 + self.pan.x()
        oy = (H - ih * sc) / 2.0 + self.pan.y()
        return ox, oy, sc

    def to_screen(self, x, y):
        ox, oy, sc = self._view()
        return QPointF(ox + x * sc, oy + y * sc)

    def to_frame(self, p):
        ox, oy, sc = self._view()
        return ((p.x() - ox) / sc, (p.y() - oy) / sc)

    def _clamp(self, x, y):
        iw, ih = self.frame_size
        return (min(max(x, 0.0), iw - 1.0), min(max(y, 0.0), ih - 1.0))

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    # ------------------------------------------------------------------ hit testing
    def _sel_bbox(self):
        pts = [p for n in self.selected for p in self.zones.get(n, [])]
        if not pts:
            return None
        a = np.asarray(pts, float)
        return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())

    def _handles(self):
        """Screen positions of the selection handles: dict name -> QPointF."""
        bb = self._sel_bbox()
        if bb is None or self.locked:
            return {}
        x0, y0, x1, y1 = bb
        p0, p1 = self.to_screen(x0, y0), self.to_screen(x1, y1)
        pad = 10.0
        L, T, R, B = p0.x() - pad, p0.y() - pad, p1.x() + pad, p1.y() + pad
        mx, my = (L + R) / 2, (T + B) / 2
        return {"nw": QPointF(L, T), "n": QPointF(mx, T), "ne": QPointF(R, T), "e": QPointF(R, my),
                "se": QPointF(R, B), "s": QPointF(mx, B), "sw": QPointF(L, B), "w": QPointF(L, my),
                "rot": QPointF(mx, T - 26)}

    def _hit_handle(self, pos):
        for k, hp in self._handles().items():
            if math.hypot(pos.x() - hp.x(), pos.y() - hp.y()) <= self.HANDLE + 4:
                return k
        return None

    def _hit_vertex(self, pos):
        if len(self.selected) != 1 or self.locked:
            return None
        n = self.selected[0]
        best, bd = None, 9.0
        for i, (x, y) in enumerate(self.zones.get(n, [])):
            sp = self.to_screen(x, y)
            d = math.hypot(sp.x() - pos.x(), sp.y() - pos.y())
            if d <= bd:
                best, bd = (n, i), d
        return best

    def _same_vertex(self, name, i):
        pts = self.zones.get(name, [])
        x, y = pts[i]
        return [k for k, (a, b) in enumerate(pts) if abs(a - x) < 1e-6 and abs(b - y) < 1e-6]

    def _hit_edge(self, pos):
        if len(self.selected) != 1:
            return None
        n = self.selected[0]
        pts = self.zones.get(n, [])
        best, bd = None, 7.0
        edges = {(tuple(pts[i]), tuple(pts[(i + 1) % len(pts)])) for i in range(len(pts))}
        for i in range(len(pts)):
            if (tuple(pts[(i + 1) % len(pts)]), tuple(pts[i])) in edges:
                continue                       # the hidden cut that joins a ring's two outlines
            a, b = self.to_screen(*pts[i]), self.to_screen(*pts[(i + 1) % len(pts)])
            d, q = seg_distance(pos.x(), pos.y(), a.x(), a.y(), b.x(), b.y())
            if d <= bd:
                best, bd = (n, i, q), d
        return best

    def zone_at(self, pos):
        fx, fy = self.to_frame(pos)
        hits = [(poly_area(p), n) for n, p in self.zones.items() if point_in_poly(fx, fy, p)]
        return min(hits)[1] if hits else None

    # ------------------------------------------------------------------ painting
    def _path(self, pts, closed=True):
        path = QPainterPath()
        if not pts:
            return path
        path.moveTo(self.to_screen(*pts[0]))
        for p in pts[1:]:
            path.lineTo(self.to_screen(*p))
        if closed:
            path.closeSubpath()
        return path

    def _outline(self, pts):
        """Outline for drawing: a ring zone (keyhole polygon from Fragment) is drawn as its two loops, without
        the zero-width cut that joins them."""
        rings = poly_rings(pts)
        if len(rings) == 1:
            return self._path(pts)
        path = QPainterPath()
        for r in rings:
            path.addPath(self._path(r))
        return path

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#1c212b"))
        if self._img is None:
            p.setPen(QColor("#8b95a7"))
            f = p.font()
            f.setPointSize(12)
            p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        ox, oy, sc = self._view()
        iw, ih = self.frame_size
        target = QRectF(ox, oy, iw * sc, ih * sc)
        p.drawImage(target, self._img)
        drawing_crop = self.purpose == "crop"

        if not drawing_crop:
            # auto-mask preview: everything outside the zones is dimmed
            if self.auto_mask and self.zones:
                outside = QPainterPath()
                outside.addRect(target)
                inside = QPainterPath()
                inside.setFillRule(Qt.WindingFill)
                for pts in self.zones.values():
                    inside.addPath(self._path(pts))
                try:
                    for x0, y0, x1, y1 in (self.extra_inside() if self.extra_inside is not None else ()):
                        inside.addRect(QRectF(self.to_screen(x0, y0), self.to_screen(x1, y1)))
                except Exception:
                    pass
                p.fillPath(outside.subtracted(inside.simplified()), QColor(5, 8, 15, 140))
            for m in self.masks:
                path = self._path(m)
                p.fillPath(path, QColor(15, 15, 15, 150))
                p.fillPath(path, QBrush(QColor(255, 255, 255, 70), Qt.BDiagPattern))
                p.setPen(QPen(QColor("#cbd5e1"), 1.5, Qt.DashLine))
                p.drawPath(path)
            try:
                covered = set(self.covered_fn() or ()) if self.covered_fn is not None else set()
            except Exception:
                covered = set()
            for name, pts in self.zones.items():
                col = QColor(self.color_of(name))
                sel = name in self.selected
                fill = QColor(col)
                fill.setAlpha(85 if sel else (70 if name == self.hover else 45))
                path = self._path(pts)
                p.fillPath(path, fill)
                if name in covered:                       # covered zone: hatched (the animal is hidden there)
                    hatch = QColor(col).lighter(150)
                    hatch.setAlpha(150)
                    p.fillPath(path, QBrush(hatch, Qt.BDiagPattern))
                p.setPen(QPen(col, 2.0 if sel else 1.0))
                p.setBrush(Qt.NoBrush)
                p.drawPath(self._outline(pts))
            # labels (on top of all fills)
            f = QFont(self.font())
            f.setPointSize(9)
            f.setBold(True)
            p.setFont(f)
            for name, pts in self.zones.items():
                cx, cy = label_point(pts)
                sp = self.to_screen(cx, cy)
                shown = name + ("  (covered)" if name in covered else "")
                r = p.fontMetrics().boundingRect(shown)
                box = QRectF(sp.x() - r.width() / 2 - 5, sp.y() - r.height() / 2 - 2, r.width() + 10, r.height() + 4)
                scr = [self.to_screen(x, y) for x, y in pts]
                if scr:
                    top = min(q.y() for q in scr)
                    zw = max(q.x() for q in scr) - min(q.x() for q in scr)
                    zh = max(q.y() for q in scr) - top
                    if zw < box.width() + 4 and zh < box.height() * 2.4:
                        # a small zone (e.g. the Y-maze centre triangle) would vanish under its own label:
                        # put the label just above it with a short leader line
                        box.moveBottom(top - 7)
                        p.setPen(QPen(QColor(self.color_of(name)), 1.4))
                        p.drawLine(QPointF(sp.x(), box.bottom()), QPointF(sp.x(), top + min(zh, 3.0)))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(15, 23, 42, 170))
                p.drawRoundedRect(box, 4, 4)
                p.setPen(QColor(self.color_of(name)).lighter(130))
                p.drawText(box, Qt.AlignCenter, shown)
            # selection decorations
            if self.selected and not self.locked:
                hs = self._handles()
                if hs:
                    p.setBrush(Qt.NoBrush)
                    p.setPen(QPen(QColor("#e2e8f0"), 1.0, Qt.DashLine))
                    p.drawRect(QRectF(hs["nw"], hs["se"]))
                    p.drawLine(hs["n"], hs["rot"])
                    p.setPen(QPen(QColor("#1f6feb"), 1.5))
                    p.setBrush(QColor("#ffffff"))
                    for k, hp in hs.items():
                        if k == "rot":
                            p.setBrush(QColor("#1f6feb"))
                            p.drawEllipse(hp, self.HANDLE, self.HANDLE)
                            p.setBrush(QColor("#ffffff"))
                        else:
                            p.drawRect(QRectF(hp.x() - self.HANDLE / 1.3, hp.y() - self.HANDLE / 1.3,
                                              2 * self.HANDLE / 1.3, 2 * self.HANDLE / 1.3))
                if len(self.selected) == 1:
                    vpts = self.zones.get(self.selected[0], [])
                    r = 4.0 if len(vpts) <= 16 else 2.6
                    p.setPen(QPen(QColor("#0f172a"), 1.0))
                    p.setBrush(QColor("#fde047"))
                    for x, y in vpts:
                        sp = self.to_screen(x, y)
                        p.drawRect(QRectF(sp.x() - r, sp.y() - r, 2 * r, 2 * r))
            if self.extra_paint is not None:
                try:
                    self.extra_paint(p)
                except Exception:
                    pass

        # draft shape
        if self.tool is not None:
            pen = QPen(QColor("#facc15"), 2.0)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            pts = list(self.draft)
            if self.tool == "polygon" and pts:
                if len(pts) >= 3:
                    fill = QColor("#facc15")
                    fill.setAlpha(30)
                    p.fillPath(self._path(pts), fill)
                p.drawPath(self._path(pts, closed=False))
                if self._cursor_f is not None:
                    p.setPen(QPen(QColor("#facc15"), 1.2, Qt.DashLine))
                    p.drawLine(self.to_screen(*pts[-1]), self.to_screen(*self._cursor_f))
                    if len(pts) >= 3:
                        p.setPen(QPen(QColor("#facc15"), 1.0, Qt.DotLine))
                        p.drawLine(self.to_screen(*self._cursor_f), self.to_screen(*pts[0]))
            elif self.tool in ("freehand",) + SHAPE_TOOLS and pts:
                fill = QColor("#facc15")
                fill.setAlpha(40)
                path = self._path(pts, closed=self.tool != "freehand")
                p.fillPath(path, fill)
                p.drawPath(path)
            elif self.tool == "points" and pts:
                if len(pts) >= 2:
                    p.drawPath(self._path(pts, closed=self.purpose == "crop" and len(pts) == 4))
            p.setBrush(QColor("#facc15"))
            p.setPen(QPen(QColor("#0f172a"), 1.0))
            for i, (x, y) in enumerate(pts if self.tool in ("polygon", "points") else pts[:1]):
                sp = self.to_screen(x, y)
                rr = 6.0 if (i == 0 and self.tool == "polygon" and len(pts) >= 3) else 4.5
                p.drawEllipse(sp, rr, rr)
            if self.tool == "points" and self.point_labels:
                f = p.font()
                f.setBold(True)
                p.setFont(f)
                for i, (x, y) in enumerate(pts):
                    lab = self.point_labels[min(i, len(self.point_labels) - 1)]
                    sp = self.to_screen(x, y) + QPointF(9, -9)
                    rect = p.fontMetrics().boundingRect(lab).adjusted(-4, -2, 4, 2).translated(sp.toPoint())
                    p.fillRect(rect, QColor(15, 23, 42, 200))
                    p.setPen(QColor("#facc15"))
                    p.drawText(sp, lab)
                p.setPen(QPen(QColor("#0f172a"), 1.0))
            if self.purpose == "scale" and len(pts) == 2:
                (x0, y0), (x1, y1) = pts
                mid = self.to_screen((x0 + x1) / 2, (y0 + y1) / 2)
                p.setPen(QColor("#facc15"))
                p.drawText(mid + QPointF(8, -8), f"{math.hypot(x1 - x0, y1 - y0):.1f} px")

        if self.marker is not None:
            sp = self.to_screen(*self.marker)
            p.setPen(QPen(QColor("#ff2d55"), 2.5))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(sp, 11, 11)
            p.drawLine(sp + QPointF(-16, 0), sp + QPointF(16, 0))
            p.drawLine(sp + QPointF(0, -16), sp + QPointF(0, 16))

        if self.tool is not None and self.zoom > 1.0:
            p.setPen(QColor("#94a3b8"))
            p.drawText(QRectF(10, self.height() - 24, 300, 20), Qt.AlignLeft, f"zoom {self.zoom:.1f}×  (0 = reset)")
        if self.locked:
            p.setPen(QColor("#fbbf24"))
            p.drawText(QRectF(0, self.height() - 24, self.width() - 10, 20), Qt.AlignRight, "🔒 zones locked")

    # ------------------------------------------------------------------ keyboard
    def event(self, ev):
        if ev.type() == QEvent.ShortcutOverride:
            k = ev.key()
            mods = ev.modifiers()
            wants = set()
            if self.tool is not None:
                wants |= {Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape, Qt.Key_Backspace, Qt.Key_Delete}
            elif self.selected:
                wants |= {Qt.Key_Escape, Qt.Key_Delete, Qt.Key_Backspace}
                if mods & Qt.ControlModifier:          # Ctrl+arrows nudge; plain arrows step frames
                    wants |= {Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down}
            if mods & Qt.ControlModifier and k in (Qt.Key_Z, Qt.Key_Y, Qt.Key_A, Qt.Key_D):
                wants.add(k)
            if k in wants:
                ev.accept()
                return True
        return super().event(ev)

    def keyPressEvent(self, ev):
        k, mods = ev.key(), ev.modifiers()
        if mods & Qt.ControlModifier:
            if k == Qt.Key_Z:
                (self.redo() if mods & Qt.ShiftModifier else self.undo())
                return
            if k == Qt.Key_Y:
                self.redo()
                return
            if k == Qt.Key_A and not self.locked:
                self.select(list(self.zones))
                return
            if k == Qt.Key_D and not self.locked:
                self.duplicate_selected()
                return
        if self.tool is not None:
            if k in (Qt.Key_Return, Qt.Key_Enter):
                if not self.finish_tool():
                    self.hoverInfo.emit("Need more points before finishing.")
                return
            if k == Qt.Key_Escape:
                self.cancel_tool()
                return
            if k in (Qt.Key_Backspace, Qt.Key_Delete):
                self.undo_point()
                return
        elif self.selected and not self.locked:
            if k == Qt.Key_Escape:
                self.select([])
                return
            if k in (Qt.Key_Delete, Qt.Key_Backspace):
                self.delete_zones(list(self.selected))
                return
            step = (10.0 if mods & Qt.ShiftModifier else 1.0) / max(self._view()[2], 1e-6)
            d = {Qt.Key_Left: (-step, 0), Qt.Key_Right: (step, 0), Qt.Key_Up: (0, -step), Qt.Key_Down: (0, step)}.get(k)
            if d and mods & Qt.ControlModifier:
                self.push_undo()
                for n in self.selected:
                    self.zones[n] = [(x + d[0], y + d[1]) for x, y in self.zones[n]]
                self.update()
                self.zonesEdited.emit()
                return
        if k in (Qt.Key_0, Qt.Key_F) and not mods:
            self.reset_view()
            return
        super().keyPressEvent(ev)

    # ------------------------------------------------------------------ mouse
    def wheelEvent(self, ev):
        if self._img is None:
            return
        pos = ev.position()
        before = self.to_frame(pos)
        factor = 1.0015 ** ev.angleDelta().y()
        self.zoom = min(12.0, max(1.0, self.zoom * factor))
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        else:
            after = self.to_screen(*before)
            self.pan += QPointF(pos.x() - after.x(), pos.y() - after.y())
        self.update()

    def mousePressEvent(self, ev):
        if self._img is None:
            return
        self.setFocus()
        pos = ev.position()
        btn = ev.button()
        if btn == Qt.MiddleButton or (btn == Qt.LeftButton and ev.modifiers() & Qt.AltModifier):
            self._drag = {"mode": "pan", "start": pos, "pan0": QPointF(self.pan)}
            self.setCursor(Qt.ClosedHandCursor)
            return
        fx, fy = self._clamp(*self.to_frame(pos))
        shift = bool(ev.modifiers() & Qt.ShiftModifier)
        if self.tool is not None:
            if btn == Qt.RightButton:
                self.undo_point()
                return
            if btn != Qt.LeftButton:
                return
            if self.tool == "polygon":
                if len(self.draft) >= 3:
                    sp0 = self.to_screen(*self.draft[0])
                    if math.hypot(sp0.x() - pos.x(), sp0.y() - pos.y()) <= 10:
                        self.finish_tool()
                        return
                self.draft.append((fx, fy))
            elif self.tool == "points":
                for i, (x, y) in enumerate(self.draft):
                    sp = self.to_screen(x, y)
                    if math.hypot(sp.x() - pos.x(), sp.y() - pos.y()) <= 9:
                        self._drag = {"mode": "draft_pt", "i": i}
                        return
                if self.max_points is None or len(self.draft) < self.max_points:
                    self.draft.append((fx, fy))
                    self.draftChanged.emit()
                    if (self.purpose == "scale" or self.auto_finish) and len(self.draft) == self.max_points:
                        self.update()
                        self.finish_tool()
                        return
            else:   # freehand / rect / ellipse
                self.draft = [(fx, fy)]
                self._drag = {"mode": "shape", "anchor": (fx, fy), "shift": shift}
            self.update()
            return
        # ---- editing (no tool)
        if btn == Qt.RightButton:
            v = self._hit_vertex(pos)
            if v is not None and len(self.zones[v[0]]) > 3 and len(self._same_vertex(*v)) == 1:
                self.push_undo()
                del self.zones[v[0]][v[1]]
                self.update()
                self.zonesEdited.emit()
                return
            name = self.zone_at(pos)
            if name and name not in self.selected:
                self.select([name])
            self.contextRequested.emit(name, ev.globalPosition().toPoint())
            return
        if btn != Qt.LeftButton:
            return
        if self.locked:
            name = self.zone_at(pos)
            self.select([name] if name else [])
            return
        h = self._hit_handle(pos)
        if h is not None:
            bb = self._sel_bbox()
            self._drag = {"mode": "rotate" if h == "rot" else "scale", "handle": h, "bbox": bb,
                          "start": (fx, fy), "snap": self.snapshot(),
                          "orig": {n: list(self.zones[n]) for n in self.selected}}
            return
        v = self._hit_vertex(pos)
        if v is not None:
            self._drag = {"mode": "vertex", "zone": v[0], "i": v[1], "same": self._same_vertex(*v),
                          "snap": self.snapshot()}
            return
        name = self.zone_at(pos)
        if name is not None:
            if shift:
                sel = [n for n in self.selected if n != name] if name in self.selected else self.selected + [name]
                self.select(sel)
            elif name not in self.selected:
                self.select([name])
            if self.selected:
                self._drag = {"mode": "move", "start": (fx, fy), "snap": self.snapshot(),
                              "orig": {n: list(self.zones[n]) for n in self.selected}}
                self.setCursor(Qt.SizeAllCursor)
        else:
            if not shift:
                self.select([])

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._img is None:
            return
        fx, fy = self._clamp(*self.to_frame(pos))
        self._cursor_f = (fx, fy)
        d = self._drag
        if d is None:
            if self.tool is None:
                name = self.zone_at(pos)
                if name != self.hover:
                    self.hover = name
                    self.update()
                h = self._hit_handle(pos) if not self.locked else None
                if h == "rot":
                    self.setCursor(Qt.CrossCursor)
                elif h in ("nw", "se"):
                    self.setCursor(Qt.SizeFDiagCursor)
                elif h in ("ne", "sw"):
                    self.setCursor(Qt.SizeBDiagCursor)
                elif h in ("n", "s"):
                    self.setCursor(Qt.SizeVerCursor)
                elif h in ("e", "w"):
                    self.setCursor(Qt.SizeHorCursor)
                elif self._hit_vertex(pos) is not None:
                    self.setCursor(Qt.PointingHandCursor)
                elif name in self.selected and not self.locked:
                    self.setCursor(Qt.SizeAllCursor)
                else:
                    self.unsetCursor()
            else:
                self.update()
            return
        mode = d["mode"]
        if mode == "pan":
            self.pan = d["pan0"] + (pos - d["start"])
        elif mode == "draft_pt":
            self.draft[d["i"]] = (fx, fy)
        elif mode == "shape":
            ax, ay = d["anchor"]
            if self.tool == "freehand":
                lx, ly = self.draft[-1]
                sp, lp = self.to_screen(fx, fy), self.to_screen(lx, ly)
                if math.hypot(sp.x() - lp.x(), sp.y() - lp.y()) >= 3.0:
                    self.draft.append((fx, fy))
            else:
                self.draft = shape_points(self.tool, ax, ay, fx, fy, bool(ev.modifiers() & Qt.ShiftModifier))
        elif mode == "vertex":
            for i in d.get("same") or [d["i"]]:       # a ring's joining points exist twice: move both
                self.zones[d["zone"]][i] = (fx, fy)
        elif mode == "move":
            dx, dy = fx - d["start"][0], fy - d["start"][1]
            for n, pts in d["orig"].items():
                self.zones[n] = [(x + dx, y + dy) for x, y in pts]
        elif mode == "scale":
            x0, y0, x1, y1 = d["bbox"]
            h = d["handle"]
            ax = x1 if "w" in h else (x0 if "e" in h else None)
            ay = y1 if "n" in h else (y0 if "s" in h else None)
            sx = sy = 1.0
            if ax is not None:
                ref = x0 if "w" in h else x1
                sx = (fx - ax) / (ref - ax) if abs(ref - ax) > 1e-6 else 1.0
            if ay is not None:
                ref = y0 if "n" in h else y1
                sy = (fy - ay) / (ref - ay) if abs(ref - ay) > 1e-6 else 1.0
            if ev.modifiers() & Qt.ShiftModifier and ax is not None and ay is not None:
                s = max(abs(sx), abs(sy))
                sx, sy = math.copysign(s, sx), math.copysign(s, sy)
            ax = ax if ax is not None else (x0 + x1) / 2
            ay = ay if ay is not None else (y0 + y1) / 2
            sx, sy = max(sx, 0.05), max(sy, 0.05)          # no mirroring / collapsing
            for n, pts in d["orig"].items():
                self.zones[n] = [(ax + (x - ax) * sx, ay + (y - ay) * sy) for x, y in pts]
        elif mode == "rotate":
            x0, y0, x1, y1 = d["bbox"]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            a0 = math.atan2(d["start"][1] - cy, d["start"][0] - cx)
            a1 = math.atan2(fy - cy, fx - cx)
            ang = a1 - a0
            if ev.modifiers() & Qt.ShiftModifier:
                ang = math.radians(round(math.degrees(ang) / 15.0) * 15.0)
            c, s = math.cos(ang), math.sin(ang)
            for n, pts in d["orig"].items():
                self.zones[n] = [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]
            self.hoverInfo.emit(f"Rotation {math.degrees(ang):+.1f}°")
        self.update()

    def mouseReleaseEvent(self, ev):
        d = self._drag
        self._drag = None
        if d is None:
            return
        mode = d["mode"]
        if mode == "pan":
            self.unsetCursor()
            return
        if mode == "shape":
            if self.tool in SHAPE_TOOLS and self._img is not None:
                fx, fy = self._clamp(*self.to_frame(ev.position()))
                ax, ay = d["anchor"]
                if abs(fx - ax) > 1 or abs(fy - ay) > 1:
                    self.draft = shape_points(self.tool, ax, ay, fx, fy, bool(ev.modifiers() & Qt.ShiftModifier))
            if self.tool == "freehand":
                pts = self.draft
                if len(pts) >= 3:
                    eps = max(0.8, 1.2 / max(self._view()[2], 1e-6))
                    arr = cv2.approxPolyDP(np.asarray(pts, np.float32).reshape(-1, 1, 2), eps, True).reshape(-1, 2)
                    self.draft = [(float(x), float(y)) for x, y in arr]
                if len(self.draft) >= 3 and poly_area(self.draft) > 4:
                    self.finish_tool()
                else:
                    self.draft = []
                    self.hoverInfo.emit("Trace: press and drag around the area, then release.")
            else:
                if len(self.draft) >= 3 and poly_area(self.draft) > 4:
                    self.finish_tool()
                else:
                    self.draft = []
            self.update()
            return
        if mode in ("vertex", "move", "scale", "rotate"):
            if self.snapshot() != d["snap"]:
                self.push_undo(d["snap"])
                self.zonesEdited.emit()
            self.unsetCursor()
        self.update()

    def mouseDoubleClickEvent(self, ev):
        pos = ev.position()
        if self.tool == "polygon":
            if len(self.draft) >= 3:
                self.finish_tool()
            else:
                fx, fy = self._clamp(*self.to_frame(pos))
                self.draft.append((fx, fy))
                self.update()
            return
        if self.tool is not None or self.locked:
            return
        e = self._hit_edge(pos)
        if e is not None and self._hit_vertex(pos) is None:
            n, i, q = e
            fx, fy = self._clamp(*self.to_frame(QPointF(*q)))
            self.push_undo()
            self.zones[n].insert(i + 1, (fx, fy))
            self.update()
            self.zonesEdited.emit()
            return
        name = self.zone_at(pos)
        if name:
            self.zoneRenameRequested.emit(name)

    def leaveEvent(self, _ev):
        self._cursor_f = None
        if self.hover is not None:
            self.hover = None
            self.update()


# ============================================================================= workers
class _Cancelled(Exception):
    pass


class _LiveSink:
    """frame_callback for the tracking loop: call it with a frame; `due()` says whether one is wanted."""

    def __init__(self, worker):
        self.worker = worker
        self.meta = None

    def set_meta(self, frame_number, t):
        self.meta = (frame_number, t)

    def __call__(self, bgr):
        self.worker._on_frame(bgr, self.meta)

    def due(self):
        if self.worker._stop:
            raise _Cancelled("Stopped by user")
        return self.worker._frame_due()


class TrackWorker(QObject):
    """Runs the tracking pipeline for one or many videos on a background thread."""
    frame = Signal(object)                 # QImage (live view)
    frame_pos = Signal(int, float)         # frame number + video time (s) of the next live picture
    job_started = Signal(int)
    job_progress = Signal(int, float)
    job_done = Signal(int, object)
    job_failed = Signal(int, str)
    job_note = Signal(int, str)
    all_done = Signal(bool)                # True if cancelled

    def __init__(self, jobs, emit_frames=True):
        super().__init__()
        self.jobs = jobs                   # list of (path, setup, mode)
        self.emit_frames = emit_frames
        self.frame_interval = 0.06         # seconds between live frames (the GUI slows this when nobody watches)
        self._stop = False
        self._idx = 0
        self._last_frame_t = 0.0

    def stop(self):
        self._stop = True

    def _on_progress(self, fraction):
        if self._stop:
            raise _Cancelled("Stopped by user")
        self.job_progress.emit(self._idx, float(fraction))

    def _on_frame(self, bgr, meta=None):
        if self._stop:
            raise _Cancelled("Stopped by user")
        now = time.time()
        if self.emit_frames and bgr is not None and now - self._last_frame_t > self.frame_interval:
            self._last_frame_t = now
            if meta is not None:
                self.frame_pos.emit(int(meta[0]), float(meta[1]))
            self.frame.emit(bgr_to_qimage(bgr))

    def _frame_due(self):
        """Asked by the tracking loop BEFORE it draws a live frame -- drawing is skipped when the
        live view would not show it anyway (saves a lot of time on large videos)."""
        return self.emit_frames and time.time() - self._last_frame_t > self.frame_interval

    def _auto_fit(self, i, path, setup):
        """Batch option: fit zones to THIS video's apparatus before tracking."""
        from tracking.apparatus import auto_detect_zones
        from tracking.robust_tracker import build_background, detection_scale
        cap = cv2.VideoCapture(path)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            W, H = int(cap.get(3)), int(cap.get(4))
            if (W, H) != (setup["warp_w"], setup["warp_h"]) and not setup.get("use_crop"):
                setup["warp_w"], setup["warp_h"] = W, H

            def read(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, f = cap.read()
                if not ok:
                    return None
                if setup.get("use_crop"):
                    f = cv2.warpPerspective(f, setup["matrix"], (setup["warp_w"], setup["warp_h"]))
                return f
            bg = build_background(read, np.linspace(0, max(0, n - 1), 30).astype(int), "gray", "auto",
                                  scale=detection_scale(setup["warp_w"], setup["warp_h"]))
            img = bg.full_res_bgr(setup["warp_w"], setup["warp_h"])
        finally:
            cap.release()
        zones, _outline, msg = auto_detect_zones(img, setup.get("test_type") or "custom",
                                                 ppc=(1.0 / setup["scale_factor"]) if setup.get("scale_factor") else None)
        if zones:
            setup["roi_points"] = zones
            setup["roi_names"] = list(zones)
            self.job_note.emit(i, "auto-fitted zones: " + msg)
            out = setup.get("output_dir_override") or results_root(path)
            try:
                vis = img.copy()
                for k, (name, pts) in enumerate(zones.items()):
                    col = QColor(ZONE_PALETTE[k % len(ZONE_PALETTE)])
                    c = (col.blue(), col.green(), col.red())
                    cv2.polylines(vis, outline_arrays(pts), True, c, max(1, int(round(max(vis.shape[:2]) / 1100.0))),
                                  cv2.LINE_AA)
                    cx, cy = label_point(pts)
                    cv2.putText(vis, name, (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2, cv2.LINE_AA)
                d = Path(out) / Path(path).stem
                d.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(d / "zones_check.png"), vis)
            except Exception:
                pass
        else:
            self.job_note.emit(i, "auto-fit failed, shared zones used: " + msg)

    def _run_chambers(self, i, path, setup, spec):
        """Chamber mode: every chamber of the video tracked at once (chambers found first when asked)."""
        from tracking.chambers import process_chamber_video, find_chambers
        if isinstance(spec, dict) and spec.get("find"):
            rects, by_motion = find_chambers(path, int(spec["find"]),
                                             matrix=setup["matrix"] if setup.get("use_crop") else None,
                                             size=(setup["warp_w"], setup["warp_h"]))
            ids = spec.get("ids") or []
            ovs = spec.get("overrides") or []
            spec = [{"rect": r, "id": ids[k] if k < len(ids) else f"M{k + 1}",
                     "overrides": ovs[k] if k < len(ovs) else {}} for k, r in enumerate(rects)]
            self.job_note.emit(i, f"{len(rects)} chambers found" + ("" if by_motion else
                                                                    " (could not tell them apart: even split)"))
        if not spec:
            raise RuntimeError("No chambers are set for this video.")
        if setup.pop("_auto_fit", False):
            try:
                self._auto_fit_chambers(i, path, setup, spec)
            except _Cancelled:
                raise
            except Exception as exc:
                self.job_note.emit(i, f"auto-fit skipped ({exc})")
        return process_chamber_video(path, setup, spec, progress_callback=self._on_progress,
                                     frame_callback=_LiveSink(self))

    def _auto_fit_chambers(self, i, path, setup, chambers):
        """Batch option 'auto-fit zones': the zones of the selected test fitted inside every chamber."""
        from tracking.apparatus import auto_detect_zones
        from tracking.robust_tracker import build_background, detection_scale
        from tracking.chambers import zone_prefix
        cap = cv2.VideoCapture(path)
        try:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            def read(idx):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, f = cap.read()
                if not ok:
                    return None
                if setup.get("use_crop"):
                    f = cv2.warpPerspective(f, setup["matrix"], (setup["warp_w"], setup["warp_h"]))
                return f
            bg = build_background(read, np.linspace(0, max(0, n - 1), 30).astype(int), "gray", "auto",
                                  scale=detection_scale(setup["warp_w"], setup["warp_h"]))
            img = bg.full_res_bgr(setup["warp_w"], setup["warp_h"])
        finally:
            cap.release()
        ppc = (1.0 / setup["scale_factor"]) if setup.get("scale_factor") else None
        zones, got = {}, 0
        for k, ch in enumerate(chambers):
            x0, y0, x1, y1 = [int(round(v)) for v in ch["rect"]]
            z, _o, _m = auto_detect_zones(np.ascontiguousarray(img[y0:y1, x0:x1]), setup.get("test_type") or "custom",
                                          ppc=ppc)
            got += bool(z)
            for name, pts in (z or {}).items():
                zones[zone_prefix(k + 1) + name] = [(float(x) + x0, float(y) + y0) for x, y in pts]
        if zones:
            setup["roi_points"] = zones
            setup["roi_names"] = list(zones)
            self.job_note.emit(i, f"auto-fitted zones in {got} of {len(chambers)} chambers")

    def run(self):
        from tracking.location import process_single_video
        from tracking.two_mouse import process_multi_mouse_video
        cancelled = False
        for i, job in enumerate(self.jobs):
            path, setup = job[0], dict(job[1])
            mode = job[2] if len(job) > 2 else "single"
            if self._stop:
                cancelled = True
                break
            self._idx = i
            self.job_started.emit(i)
            try:
                if mode == "chambers":
                    pass                                # zones are fitted per chamber (_run_chambers)
                elif setup.pop("_auto_fit", False):
                    try:
                        self._auto_fit(i, path, setup)
                    except _Cancelled:
                        raise
                    except Exception as exc:
                        self.job_note.emit(i, f"auto-fit skipped ({exc})")
                if mode == "chambers":
                    summary = self._run_chambers(i, path, setup, job[3] if len(job) > 3 else None)
                elif mode == "pose":
                    from tracking.location import process_pose_file
                    summary = process_pose_file(path, setup, progress_callback=self._on_progress)
                elif mode == "multi":
                    out_root = setup.get("output_dir_override") or results_root(path)
                    summary = process_multi_mouse_video(
                        path, str(Path(out_root) / Path(path).stem), num_animals=setup.get("num_animals", 2),
                        progress_callback=self._on_progress, frame_callback=self._on_frame,
                        start_time=setup.get("start_time", 0.0), end_time=setup.get("end_time"),
                        setup=setup, color_mode=setup.get("color_mode", "gray"))
                else:
                    summary = process_single_video(path, setup, progress_callback=self._on_progress,
                                                   frame_callback=_LiveSink(self))
                try:
                    out_dir = summary.get("Output_folder", "")
                    if out_dir and setup.get("experimental_design"):
                        Path(out_dir).mkdir(parents=True, exist_ok=True)
                        save_json(os.path.join(out_dir, "experimental_design.json"), {
                            "experiment_name": setup.get("experiment_name", ""),
                            "investigator": setup.get("investigator", ""),
                            "description": setup.get("description", ""),
                            "records": setup.get("experimental_design", []),
                            "created": datetime.datetime.now().isoformat(timespec="seconds")})
                        pd.DataFrame(setup.get("experimental_design", [])).to_csv(
                            Path(out_dir) / "experimental_design.csv", index=False)
                    if out_dir:
                        save_json(os.path.join(out_dir, "zones.json"), {
                            "zones": {k: [list(map(float, p)) for p in v] for k, v in setup.get("roi_points", {}).items()},
                            "masks": setup.get("mask_points", []), "test": setup.get("test_type"),
                            "warp_size": [setup.get("warp_w"), setup.get("warp_h")]})
                except Exception:
                    pass
                self.job_done.emit(i, summary)
            except _Cancelled:
                cancelled = True
                self.job_failed.emit(i, "Stopped")
                break
            except Exception as exc:
                self.job_failed.emit(i, f"{type(exc).__name__}: {exc}")
        self.all_done.emit(cancelled)


class LiveBatchDialog(QDialog):
    """Batch Processing ▸ watch one video of the queue being tracked, live.

    Non-modal: the batch keeps running while it is open.  Shows the selected
    video's live tracking frames (animal marker, zones, trajectory), its
    progress and the video time being tracked; a pending video shows its first
    frame until its turn comes, a finished one its last tracked frame."""

    def __init__(self, ctrl):
        super().__init__(ctrl.win)
        self.ctrl = ctrl
        self.path = None
        self._thumbs = {}
        self.setModal(False)
        self.setWindowTitle("Live tracking view")
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(960, 680)
        v = QVBoxLayout(self)
        v.setContentsMargins(14, 12, 14, 12)
        v.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(QLabel("Video:"))
        self.cmb = QComboBox()
        self.cmb.setMinimumWidth(320)
        self.cmb.activated.connect(self._picked)
        top.addWidget(self.cmb, 1)
        self.chk_follow = QCheckBox("Follow the video being tracked")
        self.chk_follow.setToolTip("Switch to each video automatically as soon as its tracking starts.")
        self.chk_follow.toggled.connect(self._follow_toggled)
        top.addWidget(self.chk_follow)
        v.addLayout(top)
        self.surface = QLabel("Live tracking")
        self.surface.setProperty("cssClass", "videoSurface")
        self.surface.setStyleSheet("background:#1e2230; color:#c8cdd8; border-radius:8px; font-size:14px;")
        self.surface.setMinimumSize(480, 300)
        self.view = FrameView(self.surface, self)
        v.addWidget(self.surface, 1)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.lbl_state = QLabel("")
        self.lbl_state.setStyleSheet("font-weight:600;")
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedWidth(220)
        self.lbl_pct = QLabel("")
        self.lbl_pct.setStyleSheet("color:#000000;")
        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet("color:#000000;")
        for wd in (self.lbl_state, self.bar, self.lbl_pct, self.lbl_time):
            row.addWidget(wd)
        row.addStretch(1)
        self.btn_results = _small_button("📂 Results", "Open this video's results folder.")
        self.btn_results.clicked.connect(self._open_results)
        b_close = _small_button("Close", "Close the live view (the batch keeps running).")
        b_close.clicked.connect(self.close)
        row.addWidget(self.btn_results)
        row.addWidget(b_close)
        v.addLayout(row)

    # ------------------------------------------------------------------ public
    def show_path(self, path):
        self.path = path
        self.fill_combo()
        self.refresh(full=True)

    def fill_combo(self):
        self.cmb.blockSignals(True)
        self.cmb.clear()
        for k, it in enumerate(self.ctrl.batch_items, start=1):
            self.cmb.addItem(self._entry_text(k, it), it["path"])
        i = self.cmb.findData(self.path)
        if i >= 0:
            self.cmb.setCurrentIndex(i)
        self.cmb.blockSignals(False)

    def on_frame(self, path, img):
        if path == self.path and self.isVisible():
            self.view.set_qimage(img)

    def on_job_started(self, path):
        if self.chk_follow.isChecked() and path != self.path:
            self.show_path(path)
        else:
            self.fill_combo()
            self.refresh(full=(path == self.path))

    def refresh(self, full=False):
        it = self._item()
        if it is None:
            self.setWindowTitle("Live tracking view")
            self.lbl_state.setText("")
            self.lbl_pct.setText("")
            self.lbl_time.setText("")
            self.bar.setValue(0)
            self.btn_results.setEnabled(False)
            self.view.clear("This video is no longer in the batch queue." if self.path else
                            "Add videos to the batch queue.")
            return
        name = Path(it["path"]).name
        self.setWindowTitle(f"Live tracking — {name}")
        st, pct = it["status"], int(it.get("pct") or 0)
        self.lbl_state.setText(st)
        self.lbl_state.setStyleSheet(f"font-weight:600; color:{STATUS_COLORS.get(st, '#334155')};")
        self.bar.setValue(pct)
        color = {"Completed": "#22c55e", "Processing": "#1f6feb", "Failed": "#ef4444",
                 "Stopped": "#f59e0b"}.get(st, "#cbd5e1")
        self.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        self.lbl_pct.setText(f"{pct}%")
        t0 = float(it.get("start") or 0.0)
        t1 = float(it["end"]) if it.get("end") is not None else float(it.get("duration") or 0.0)
        if st == "Processing":
            rel = min(1.0, max(0.0, (pct / 100.0 - 0.08) / 0.84))
            self.lbl_time.setText(f"Video time  {fmt_clock(t0 + rel * (t1 - t0), True)} / {fmt_clock(t1, True)}")
        elif st == "Completed":
            self.lbl_time.setText("Finished — last tracked frame shown")
        elif st in ("Failed", "Stopped"):
            self.lbl_time.setText(it.get("error") or "")
        else:
            self.lbl_time.setText(f"Window  {fmt_clock(t0, True)} – {fmt_clock(t1, True)}")
        self.btn_results.setEnabled(bool((it.get("summary") or {}).get("Output_folder")))
        k = self.cmb.findData(it["path"])
        if k >= 0:
            self.cmb.setItemText(k, self._entry_text(k + 1, it))
        if not full:
            return
        img = self.ctrl._batch_frames.get(it["path"])
        if img is not None:
            self.view.set_qimage(img)
            return
        if st == "Processing":
            msg = "Building the animal-free background — the live view starts in a moment…"
        elif st == "Pending":
            msg = ("Waiting in the queue — the live view starts when this video's turn comes."
                   if self.ctrl.tracking and self.ctrl.run_kind == "batch"
                   else "Not tracked yet — start the batch to watch it live.")
        elif st == "Completed":
            msg = "Tracked in an earlier session — open its results folder to see the charts."
        else:
            msg = f"{st}."
        thumb = self._thumb(it)
        if thumb is None:
            self.view.clear(msg)
        else:
            self.view.set_bgr(self._banner(thumb, msg))

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _entry_text(k, it):
        return f"{k}.  {Path(it['path']).name}   —   {it['status']}"

    def _item(self):
        return next((b for b in self.ctrl.batch_items if b["path"] == self.path), None)

    def _thumb(self, it):
        if it["path"] in self._thumbs:
            return self._thumbs[it["path"]]
        img = None
        cap = cv2.VideoCapture(it["path"])
        try:
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(it.get("start") or 0.0) * fps))
                ok, f = cap.read()
                img = f if ok else None
        finally:
            cap.release()
        self._thumbs[it["path"]] = img
        return img

    @staticmethod
    def _banner(bgr, text):
        out = (bgr.astype(np.float32) * 0.45).astype(np.uint8)
        h, w = out.shape[:2]
        fs = max(0.45, min(1.0, w / 1400.0))
        th = max(1, int(round(fs * 2)))
        words, lines, cur = text.split(), [], ""
        for wd in words:                                   # wrap to ~70 % of the width
            trial = (cur + " " + wd).strip()
            if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0] > w * 0.7 and cur:
                lines.append(cur)
                cur = wd
            else:
                cur = trial
        if cur:
            lines.append(cur)
        lh = int(cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][1] * 1.9)
        y = h // 2 - (len(lines) - 1) * lh // 2
        for ln in lines:
            tw = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0][0]
            cv2.putText(out, ln, ((w - tw) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
            y += lh
        return out

    def _picked(self, i):
        path = self.cmb.itemData(i)
        if path and path != self.path:
            if path != self.ctrl._batch_cur_path:
                self.chk_follow.blockSignals(True)
                self.chk_follow.setChecked(False)
                self.chk_follow.blockSignals(False)
            self.show_path(path)

    def _follow_toggled(self, on):
        cur = self.ctrl._batch_cur_path
        if on and cur and cur != self.path:
            self.show_path(cur)

    def _open_results(self):
        it = self._item()
        if it is not None:
            self.ctrl._open_folder((it.get("summary") or {}).get("Output_folder", ""))

    def showEvent(self, ev):
        super().showEvent(ev)
        self.ctrl._sync_live_rate()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.ctrl._sync_live_rate()


class _Relay(QObject):
    """Lives in the GUI thread: worker signals connected to its slot are queued,
    so the Python callback always runs on the GUI thread."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot(object)
    def call(self, value):
        self.fn(value)


class FuncWorker(QObject):
    """Run any function on a thread: done(result) / failed(message)."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ============================================================================= dialogs
class ZoneNamesDialog(QDialog):
    """Rename several zones at once (auto-detected / traced zones)."""

    def __init__(self, parent, names, suggestions, title="Name the zones", note="", epm_swap=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 120 + 40 * len(names))
        lay = QVBoxLayout(self)
        if note:
            lab = QLabel(note)
            lab.setWordWrap(True)
            lab.setStyleSheet("color:#475569;")
            lay.addWidget(lab)
        form = QFormLayout()
        self.combos = {}
        for n in names:
            cb = QComboBox()
            cb.setEditable(True)
            items = [n] + [s for s in suggestions if s != n]
            cb.addItems(items)
            cb.setCurrentText(n)
            self.combos[n] = cb
            form.addRow(QLabel(n), cb)
        lay.addLayout(form)
        self.swapped = False
        if epm_swap:
            b = QPushButton("⇄ Swap open / closed arms")
            b.clicked.connect(self._swap)
            lay.addWidget(b)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _swap(self):
        for cb in self.combos.values():
            t = cb.currentText()
            if "Open" in t:
                cb.setCurrentText(t.replace("Open", "Closed"))
            elif "Closed" in t:
                cb.setCurrentText(t.replace("Closed", "Open"))
        self.swapped = not self.swapped

    def _ok(self):
        new = [cb.currentText().strip() for cb in self.combos.values()]
        if any(not n for n in new):
            QMessageBox.warning(self, "Zone names", "Every zone needs a name.")
            return
        if len(set(new)) != len(new):
            QMessageBox.warning(self, "Zone names", "Zone names must be different.")
            return
        self.accept()

    def mapping(self):
        return {old: cb.currentText().strip() for old, cb in self.combos.items()}


class ExperimentalDesignDialog(QDialog):
    """Explicit experimental-unit metadata (never inferred).  The columns can be changed: add your own (e.g.
    Weight, Treatment, Dose), rename or remove them -- only Animal ID and Group are always kept."""
    HEADERS = ["Animal ID", "Group", "Sex", "Dam ID", "Cage ID", "Session", "Test", "Timepoint", "Measurement ID"]
    REQUIRED = ("Animal ID", "Group")

    def __init__(self, parent, rows=None, default_test="Open Field", headers=None):
        super().__init__(parent)
        self.setWindowTitle("Experimental Design")
        self.resize(1020, 540)
        self.default_test = default_test
        hdr = [h for h in (headers or self.HEADERS) if str(h).strip()]
        for r in rows or []:                              # columns used by saved records stay visible
            hdr += [k for k in r if k not in hdr]
        self.headers = list(self.REQUIRED) + [h for h in dict.fromkeys(hdr) if h not in self.REQUIRED]
        lay = QVBoxLayout(self)
        info = QLabel("Define the experimental unit and repeated-measure structure explicitly. Group, Dam ID, "
                      "Cage ID and repeated-measure relationships are never inferred.  Columns can be changed: "
                      "＋ Add column for anything else you record (e.g. Weight, Treatment, Dose); double-click "
                      "a column title to rename it; Animal ID and Group are always kept.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#475569;padding:4px 2px;")
        lay.addWidget(info)
        self.table = QTableWidget(0, len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.DoubleClicked | QTableWidget.EditKeyPressed |
                                   QTableWidget.AnyKeyPressed)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Stretch)
        hh.setMinimumSectionSize(118)                   # many columns -> scroll sideways instead of squeezing
        hh.setContextMenuPolicy(Qt.CustomContextMenu)
        hh.customContextMenuRequested.connect(self._header_menu)
        hh.sectionDoubleClicked.connect(self.rename_column)
        lay.addWidget(self.table, 1)
        bar = QHBoxLayout()
        for text, fn in (("+ Add Measurement", lambda: self.add_row(default_test=self.default_test)),
                         ("Duplicate Row", self.duplicate_row), ("Remove Row", self.remove_rows)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addSpacing(18)
        sep = QLabel("Columns:")
        sep.setStyleSheet("color:#475569;font-weight:600;")
        bar.addWidget(sep)
        for text, tip, fn in (("＋ Add column", "Add a column of your own (e.g. Weight, Treatment, Dose)",
                               self.add_column),
                              ("✕ Remove column", "Remove the column of the selected cell (Animal ID and Group "
                               "cannot be removed)", self.remove_column),
                              ("↺ Default columns", "Go back to the standard columns", self.default_columns)):
            b = QPushButton(text)
            b.setAutoDefault(False)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            bar.addWidget(b)
        bar.addStretch()
        lay.addLayout(bar)
        bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._validate_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for row in (rows or []):
            self.add_row(row=row)
        if not rows:
            self.add_row(default_test=default_test)

    # ------------------------------------------------------------------ rows
    def add_row(self, row=None, default_test="Open Field"):
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = row or {"Animal ID": "", "Group": "", "Sex": "", "Dam ID": "", "Cage ID": "", "Session": "1",
                       "Test": default_test, "Timepoint": "", "Measurement ID": f"M{r + 1}"}
        for c, h in enumerate(self.headers):
            self.table.setItem(r, c, QTableWidgetItem(str(vals.get(h, ""))))

    def duplicate_row(self):
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        for old in rows:
            vals = self._row_dict(old)
            if "Measurement ID" in vals:
                vals["Measurement ID"] = f"M{self.table.rowCount() + 1}"
            self.add_row(vals)

    def remove_rows(self):
        for r in sorted({i.row() for i in self.table.selectionModel().selectedRows()}, reverse=True):
            self.table.removeRow(r)

    # ------------------------------------------------------------------ columns
    def _col_from_selection(self):
        c = self.table.currentColumn()
        return c if 0 <= c < len(self.headers) else None

    def _header_menu(self, pos):
        c = self.table.horizontalHeader().logicalIndexAt(pos)
        menu = QMenu(self)
        menu.addAction("＋ Add column after this…", lambda: self.add_column(after=c))
        if 0 <= c < len(self.headers):
            menu.addAction(f"✎ Rename '{self.headers[c]}'…", lambda: self.rename_column(c))
            a = menu.addAction(f"✕ Remove '{self.headers[c]}'", lambda: self.remove_column(c))
            a.setEnabled(self.headers[c] not in self.REQUIRED)
        menu.exec(self.table.horizontalHeader().mapToGlobal(pos))

    def _ask_column_name(self, title, current=""):
        while True:
            name, ok = QInputDialog.getText(self, title, "Column name:", QLineEdit.Normal, current)
            if not ok:
                return None
            name = name.strip()
            if not name:
                continue
            if name in self.headers and name != current:
                QMessageBox.warning(self, title, f"There is already a column called '{name}'.")
                continue
            return name

    def add_column(self, after=None):
        name = self._ask_column_name("Add column")
        if not name:
            return
        if after is None or after is False:
            after = self._col_from_selection()
        pos = len(self.headers) if after is None or after < 0 else after + 1
        pos = max(pos, len(self.REQUIRED))
        self.table.insertColumn(pos)
        self.headers.insert(pos, name)
        self.table.setHorizontalHeaderLabels(self.headers)
        for r in range(self.table.rowCount()):
            self.table.setItem(r, pos, QTableWidgetItem(""))

    def rename_column(self, c=None):
        if c is None or c is False:
            c = self._col_from_selection()
        if c is None:
            QMessageBox.information(self, "Rename column", "Click a cell in the column you want to rename.")
            return
        old = self.headers[c]
        if old in self.REQUIRED:
            QMessageBox.information(self, "Rename column", f"'{old}' is needed by the analysis and keeps its "
                                    "name.")
            return
        name = self._ask_column_name("Rename column", old)
        if name and name != old:
            self.headers[c] = name
            self.table.setHorizontalHeaderLabels(self.headers)

    def remove_column(self, c=None):
        if c is None or c is False:
            c = self._col_from_selection()
        if c is None:
            QMessageBox.information(self, "Remove column", "Click a cell in the column you want to remove.")
            return
        name = self.headers[c]
        if name in self.REQUIRED:
            QMessageBox.information(self, "Remove column", f"'{name}' is needed by the analysis and cannot be "
                                    "removed.")
            return
        filled = sum(1 for r in range(self.table.rowCount())
                     if self.table.item(r, c) and self.table.item(r, c).text().strip())
        if filled and QMessageBox.question(
                self, "Remove column", f"Remove the column '{name}'? {filled} row(s) have a value in it.") \
                != QMessageBox.Yes:
            return
        self.table.removeColumn(c)
        self.headers.pop(c)
        self.table.setHorizontalHeaderLabels(self.headers)

    def default_columns(self):
        rows = [self._row_dict(r) for r in range(self.table.rowCount())]
        extra = [h for h in self.headers if h not in self.HEADERS]
        lost = [h for h in extra if any(r.get(h) for r in rows)]
        if lost and QMessageBox.question(
                self, "Default columns", "Go back to the standard columns? These columns and their values will "
                f"be removed: {', '.join(lost)}") != QMessageBox.Yes:
            return
        self.headers = list(self.HEADERS)
        self.table.setRowCount(0)
        self.table.setColumnCount(len(self.headers))
        self.table.setHorizontalHeaderLabels(self.headers)
        for r in rows:
            self.add_row(r)

    # ------------------------------------------------------------------ values
    def _vals(self, r):
        return [self.table.item(r, c).text().strip() if self.table.item(r, c) else ""
                for c in range(len(self.headers))]

    def _row_dict(self, r):
        return dict(zip(self.headers, self._vals(r)))

    def _validate_accept(self):
        title = "Experimental Design"
        if len(set(self.headers)) != len(self.headers):
            QMessageBox.warning(self, title, "Two columns have the same name.")
            return
        need = [h for h in ("Animal ID", "Group", "Session", "Test") if h in self.headers]
        key_cols = [h for h in ("Animal ID", "Session", "Test", "Measurement ID") if h in self.headers]
        fixed = [h for h in ("Group", "Sex", "Dam ID", "Cage ID") if h in self.headers]
        seen, identity = set(), {}
        for r in range(self.table.rowCount()):
            d = self._row_dict(r)
            if any(not d.get(h) for h in need):
                QMessageBox.warning(self, title, f"{', '.join(need)} are required (row {r + 1}).")
                return
            key = tuple(d.get(h) for h in key_cols)
            if key in seen:
                QMessageBox.warning(self, title, f"Row {r + 1}: the same {' / '.join(key_cols)} appears twice. "
                                    "Give repeated measurements a different Session, Test or Measurement ID.")
                return
            seen.add(key)
            meta = tuple(d.get(h) for h in fixed)
            aid = d.get("Animal ID")
            if aid in identity and identity[aid] != meta:
                QMessageBox.warning(self, title, f"{', '.join(fixed)} must stay the same for one Animal ID "
                                    f"('{aid}').")
                return
            identity[aid] = meta
        self.accept()

    def values(self):
        return [self._row_dict(r) for r in range(self.table.rowCount())]


class SubjectDialog(QDialog):
    def __init__(self, parent, next_id):
        super().__init__(parent)
        self.setWindowTitle("Add subject")
        form = QFormLayout(self)
        self.id_edit = QLineEdit(next_id)
        self.group = QComboBox()
        self.group.setEditable(True)
        self.group.addItems(["Control", "Treatment"])
        self.sex = QComboBox()
        self.sex.addItems(["M", "F"])
        self.weight = QDoubleSpinBox()
        self.weight.setRange(0, 500)
        self.weight.setDecimals(1)
        self.weight.setValue(25.0)
        for lbl, w in (("ID", self.id_edit), ("Group", self.group), ("Sex", self.sex), ("Weight (g)", self.weight)):
            form.addRow(lbl, w)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self):
        return (self.id_edit.text().strip(), self.group.currentText().strip(), self.sex.currentText(),
                f"{self.weight.value():.1f}")


class SettingsDialog(QDialog):
    """Advanced detection settings (the everyday ones live on the Tracking page)."""

    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.setWindowTitle("Detection settings")
        form = QFormLayout(self)
        self.bg = QSpinBox()
        self.bg.setRange(10, 500)
        self.bg.setValue(int(cfg["bg_samples"]))
        self.max_area = QSpinBox()
        self.max_area.setRange(0, 2000000)
        self.max_area.setSpecialValueText("Automatic (from animal size)")
        self.max_area.setValue(int(cfg["max_area"]))
        self.max_jump = QSpinBox()
        self.max_jump.setRange(5, 4000)
        self.max_jump.setValue(int(cfg["max_jump"]))
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0.0, 5.0)
        self.gap.setSingleStep(0.1)
        self.gap.setDecimals(2)
        self.gap.setValue(float(cfg["max_gap_s"]))
        self.gap.setSpecialValueText("Off")
        self.minmove = QDoubleSpinBox()
        self.minmove.setRange(0.0, 50.0)
        self.minmove.setDecimals(1)
        self.minmove.setValue(float(cfg["min_move_px"]))
        self.minmove.setSpecialValueText("Automatic")
        self.color = QComboBox()
        self.color.addItems(["gray", "rgb"])
        self.color.setCurrentText(cfg["color_mode"])
        self.bin_s = QSpinBox()
        self.bin_s.setRange(5, 3600)
        self.bin_s.setValue(int(cfg["bin_size_s"]))
        self.margin = QSpinBox()
        self.margin.setRange(0, 200)
        self.margin.setValue(int(cfg.get("auto_mask_margin", 0)))
        self.margin.setSpecialValueText("None (mask on the zone line)")
        self.previews = QSpinBox()
        self.previews.setRange(0, 100)
        self.previews.setValue(int(cfg.get("preview_samples", 3)))
        self.previews.setSpecialValueText("None")
        self.previews.setSuffix(" images")
        self.entry_rule = QComboBox()
        self.entry_rule.addItem("All four paws in the zone (recommended)", "paws")
        self.entry_rule.addItem("Body centre point", "centre")
        self.entry_rule.setCurrentIndex(0 if cfg.get("entry_rule", "paws") == "paws" else 1)
        self.entry_pct = QSpinBox()
        self.entry_pct.setRange(50, 100)
        self.entry_pct.setSuffix(" % of the body")
        self.entry_pct.setValue(int(cfg.get("entry_body_pct", 80)))
        self.fast = QCheckBox("Several parts of the video at once")
        self.fast.setChecked(bool(cfg.get("fast_tracking", False)))
        self.point = QComboBox()
        for key, text in (("centre", "Body centre (recommended)"), ("head", "Head (nose)"), ("tail", "Tail base")):
            self.point.addItem(text, key)
        self.point.setCurrentIndex(max(0, self.point.findData(cfg.get("tracking_point", "centre"))))
        self.hold = QCheckBox("Keep it where it was last seen (zone times add up to the whole time window)")
        self.hold.setChecked(bool(cfg.get("hold_lost", True)))
        self.adapt = QCheckBox("Follow light changes and a nudged camera (adaptive background)")
        self.adapt.setChecked(bool(cfg.get("adapt_background", True)))
        self.ldb_lid = QCheckBox("The dark box has a lid (the animal cannot be seen inside)")
        self.ldb_lid.setChecked(bool(cfg.get("ldb_dark_covered", True)))
        self.embed = QComboBox()
        for key, text in ((0, "No"), (480, "Yes — small (480 px)"), (640, "Yes — medium (640 px, recommended)"),
                          (960, "Yes — large (960 px)")):
            self.embed.addItem(text, key)
        self.embed.setCurrentIndex(max(0, self.embed.findData(int(cfg.get("embed_video_px", 640)))))
        self.explore = QComboBox()
        self.explore.addItem("Nose in the object zone (recommended)", "nose")
        self.explore.addItem("Body centre in the object zone", "centre")
        self.explore.setCurrentIndex(max(0, self.explore.findData(cfg.get("exploration_point", "nose"))))
        self.res_box = QWidget()
        rbx = QHBoxLayout(self.res_box)
        rbx.setContentsMargins(0, 0, 0, 0)
        self.res_mode = QComboBox()
        self.res_mode.addItem("Next to each video ('results' folder)", "")
        self.res_mode.addItem("One folder for everything", "one")
        self.res_path = QLineEdit(str(cfg.get("results_root") or ""))
        self.res_path.setPlaceholderText("choose a folder…")
        b_res = QPushButton("Browse…")

        def browse_res():
            d = QFileDialog.getExistingDirectory(self, "Folder for all results", self.res_path.text() or str(Path.home()))
            if d:
                self.res_path.setText(d)
                self.res_mode.setCurrentIndex(1)
        b_res.clicked.connect(browse_res)
        self.res_mode.setCurrentIndex(1 if cfg.get("results_root") else 0)
        self.res_mode.currentIndexChanged.connect(lambda i: (self.res_path.setEnabled(i == 1), b_res.setEnabled(i == 1)))
        self.res_path.setEnabled(bool(cfg.get("results_root")))
        b_res.setEnabled(bool(cfg.get("results_root")))
        rbx.addWidget(self.res_mode)
        rbx.addWidget(self.res_path, 1)
        rbx.addWidget(b_res)
        self.learn = QCheckBox("⚡ Auto remembers the zones of my set-ups")
        self.learn.setChecked(bool(cfg.get("learn_layouts", True)))
        self.learn_box = QWidget()
        lb = QHBoxLayout(self.learn_box)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.addWidget(self.learn)
        try:
            from tracking.layout_library import library
            n_known = len(library(str(LEARNED_DIR)))
        except Exception:
            n_known = 0
        self.btn_forget = QPushButton(f"Forget ({n_known})")
        self.btn_forget.setEnabled(n_known > 0)
        self.btn_forget.setToolTip("Forget every set-up ⚡ Auto has learned on this computer.")

        def forget():
            if QMessageBox.question(self, "Learned set-ups", "Forget every set-up ⚡ Auto has learned?") \
                    == QMessageBox.Yes:
                from tracking.layout_library import library as _lib
                _lib(str(LEARNED_DIR)).forget_all()
                self.btn_forget.setText("Forget (0)")
                self.btn_forget.setEnabled(False)
        self.btn_forget.clicked.connect(forget)
        lb.addWidget(self.btn_forget)
        lb.addStretch(1)
        self.entry_rule.currentIndexChanged.connect(lambda _i: self.entry_pct.setEnabled(
            self.entry_rule.currentData() == "paws"))
        self.entry_pct.setEnabled(self.entry_rule.currentData() == "paws")
        for lbl, w, tip in (
                ("Background samples", self.bg, "Frames sampled across the analysed window to build the animal-free background."),
                ("Max animal area (px)", self.max_area, "0 = automatic (6 x the measured animal size)."),
                ("Max movement / frame (px)", self.max_jump, "Search radius around the last position; it grows automatically while the animal is lost."),
                ("Fill tracking gaps up to (s)", self.gap, "Short gaps (occlusion, rearing) are interpolated; longer ones stay 'Lost'."),
                ("Min movement counted (px)", self.minmove, "Jitter filter for distance: smaller steps are not added (0 = automatic, 4 % of body length)."),
                ("Colour mode", self.color, "rgb helps when the animal differs from the floor in colour but not brightness."),
                ("Time-bin size (s)", self.bin_s, "Size of the time bins in the report."),
                ("Gap between zones and mask (px)", self.margin, "The 'mask outside the zones' follows the zone lines "
                 "tightly (0 = no gap). Give a few pixels if the animal leans over a zone edge and should not be cut."),
                ("Preview images per video", self.previews, "How many detection-check images are saved in the 'preview' "
                 "folder of each result (spread evenly over the analysed time). 0 = none."),
                ("Zone entry", self.entry_rule, "An ENTRY is counted when all four paws are inside the zone: the "
                 "animal's body (tail not counted) must be inside it. Time in a zone always uses the body centre."),
                ("Entry needs", self.entry_pct, "How much of the body must be inside the zone for 'all four paws' "
                 "(80 % lets the head stay outside)."),
                ("Fast tracking (one video)", self.fast, "Track several parts of a long video at the same time (2-3x "
                 "faster on computers with many cores, same results). The live picture then shows only the first "
                 "part, so it does not run with the clock. Off = frame by frame in order: the picture, frame number "
                 "and clock move together. Batch processing always uses all cores."),
                ("Light changes", self.adapt, "On (recommended): the empty-arena picture follows slow light drift, "
                 "and a place that has looked different for ~2 s without being the animal (the light changed, the "
                 "camera was nudged, a shadow or dropping that stays) becomes floor again. When large parts of "
                 "the arena change at once, the picture is renewed around the animal straight away."),
                ("When the animal is not found", self.hold, "On: when the animal cannot be found for a moment "
                 "(longer than the gap filling), it is taken to stay where it was last seen until it is found "
                 "again -- like ANY-maze -- so every frame of the time window you set is counted (Whole Arena = "
                 "the full window). The frames are marked 'Held' in the data. Off: those frames are left out."),
                ("Light / dark box", self.ldb_lid, "On (usual set-up): the animal disappears under the lid of the "
                 "dark box. Nothing under the lid is tracked and the time the animal is out of sight after going "
                 "in counts as time in the dark box. Off: the camera sees inside the dark box (infrared camera). "
                 "Any zone can be marked as covered with a right-click in the zone editor."),
                ("Learn my set-ups", self.learn_box, "Each time you track a video (or save zones for a batch "
                 "video) the apparatus picture and its zones are remembered on this computer. ⚡ Auto then finds "
                 "your set-up in new videos (even if the camera moved a little) and puts YOUR zones, with your "
                 "names, on it. Press ⚡ Auto twice for the built-in template."),
                ("Copy of the video in BT files", self.embed, "A smaller copy of the video is put inside the .bta "
                 "/ .btms / .btf file, so whoever opens it can check every frame with the animal visible, even "
                 "without the original video. Makes the file bigger (about 2-5 MB per minute at 640 px). "
                 "'No' keeps the files small."),
                ("Results folder", self.res_box, "Where results are saved: a 'results' folder next to each video, "
                 "or one folder for every result (a sub-folder per video)."),
                ("Object exploration (NOR)", self.explore, "Novel object test: exploration = the NOSE in an object "
                 "zone (the usual definition; the nose is the head end of the body, decided over the whole track "
                 "from the tail, the body shape and the walking direction), or the body centre in the zone (as in "
                 "BT 1.0). The nose and tail-base positions are added to the frame-by-frame data."),
        ):
            w.setToolTip(tip)
            form.addRow(lbl, w)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self):
        return {"bg_samples": self.bg.value(), "max_area": self.max_area.value(), "max_jump": self.max_jump.value(),
                "max_gap_s": self.gap.value(), "min_move_px": self.minmove.value(),
                "color_mode": self.color.currentText(), "bin_size_s": self.bin_s.value(),
                "auto_mask_margin": self.margin.value(), "preview_samples": self.previews.value(),
                "entry_rule": self.entry_rule.currentData(), "entry_body_pct": self.entry_pct.value(),
                "fast_tracking": self.fast.isChecked(), "tracking_point": "centre",
                "ldb_dark_covered": self.ldb_lid.isChecked(), "embed_video_px": int(self.embed.currentData()),
                "learn_layouts": self.learn.isChecked(), "hold_lost": self.hold.isChecked(),
                "adapt_background": self.adapt.isChecked(),
                "results_root": (self.res_path.text().strip() if self.res_mode.currentData() == "one" else ""),
                "exploration_point": self.explore.currentData()}


class ScaleDialog(QDialog):
    """Batch ▸ Measure: the same scale tool as Video Setup (yellow points on the zoomable zone view, the
    line and its length in px), then the real length -> pixels per cm."""

    def __init__(self, parent, frame_bgr, default_cm=40.0, zones=None, colors=None):
        super().__init__(parent)
        self.setWindowTitle("Measure the scale")
        self.resize(980, 700)
        self.ppc = None
        self.default_cm = float(default_cm or 40.0)
        v = QVBoxLayout(self)
        self.hint = QLabel()
        self.hint.setWordWrap(True)
        self.hint.setProperty("cssClass", "muted")
        v.addWidget(self.hint)
        self.canvas = ZoneCanvas(self)
        self.canvas.set_frame(frame_bgr)
        if zones:
            self.canvas.zones = {k: [tuple(p) for p in pts] for k, pts in zones.items()}
            for k, c in (colors or {}).items():
                self.canvas.colors[k] = QColor(c)
        self.canvas.auto_mask = False                 # zones are only shown here (for orientation)
        self.canvas.drawingFinished.connect(self._finished)
        self.canvas.draftChanged.connect(self._update)
        v.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.lbl = QLabel("")
        self.lbl.setStyleSheet("font-weight:600;")
        row.addWidget(self.lbl, 1)
        b_again = QPushButton("Click the points again")
        b_again.clicked.connect(self._start)
        row.addWidget(b_again)
        self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        row.addWidget(self.bb)
        v.addLayout(row)
        self._start()

    def _start(self):
        self.ppc = None
        self.canvas.start_tool("points", "scale", max_points=2,
                               banner="Click both ends of a distance you know (e.g. one wall of the arena). "
                                      "Mouse wheel = zoom.")
        self._update()

    def _update(self):
        self.hint.setText(self.canvas.current_banner() or "Click both ends of a distance you know.")
        n = len(self.canvas.draft)
        if self.ppc:
            self.lbl.setText(f"Scale: {self.ppc:.2f} px/cm")
        else:
            self.lbl.setText(f"Click point {min(n + 1, 2)} of 2")
        self.bb.button(QDialogButtonBox.Ok).setEnabled(self.ppc is not None)

    def _finished(self, purpose, pts):
        if purpose != "scale" or len(pts) < 2:
            return
        (x1, y1), (x2, y2) = pts[:2]
        dist = float(math.hypot(x2 - x1, y2 - y1))
        if dist < 5:
            self.lbl.setText("Those points are too close together — click again.")
            self._start()
            return
        cm, ok = QInputDialog.getDouble(self, "Set scale", f"Measured {dist:.1f} px.\nReal length (cm):",
                                        self.default_cm, 0.1, 10000.0, 2)
        if ok and cm > 0:
            self.ppc = dist / cm
            self.default_cm = cm
            self._update()
            self.accept()
        else:
            self._start()


class TimeWindowDialog(QDialog):
    def __init__(self, parent, start, end, duration):
        super().__init__(parent)
        self.setWindowTitle("Time window")
        form = QFormLayout(self)
        self.s = QDoubleSpinBox()
        self.e = QDoubleSpinBox()
        for w in (self.s, self.e):
            w.setRange(0.0, max(duration, 0.01))
            w.setDecimals(2)
            w.setSuffix(" s")
        self.s.setValue(start)
        self.e.setValue(end)
        form.addRow("Start", self.s)
        form.addRow("End", self.e)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        bb.button(QDialogButtonBox.Reset).setText("Whole video")
        bb.button(QDialogButtonBox.Reset).clicked.connect(lambda: (self.s.setValue(0), self.e.setValue(duration)))
        form.addRow(bb)


# ----------------------------------------------------------------------------- live camera
def _camera_backends():
    """OpenCV capture back-ends to try, best first, for this operating system."""
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]          # DirectShow: the same device list OBS shows
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION]
    return [cv2.CAP_V4L2]


def list_cameras():
    """[(name, index, backend)] of every camera connected now -- built-in AND USB cameras, with their
    real device names (e.g. 'Integrated Webcam', 'Logitech HD Pro Webcam C920')."""
    found = []
    try:
        from cv2_enumerate_cameras import enumerate_cameras
        for be in _camera_backends():
            try:
                cams = enumerate_cameras(be)
            except Exception:
                cams = []
            if cams:
                names = {}
                for c in cams:
                    name = (c.name or f"Camera {c.index}").strip()
                    names[name] = names.get(name, 0) + 1
                    label = name if names[name] == 1 else f"{name} ({names[name]})"
                    found.append((label, int(c.index), int(be)))
                break
    except Exception:
        found = []
    if not found:                                    # fallback: probe the first indices (quietly)
        try:
            prev = cv2.utils.logging.getLogLevel()
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        except Exception:
            prev = None
        try:
            for i in range(6):
                cap = cv2.VideoCapture(i)
                ok = cap.isOpened()
                cap.release()
                if ok:
                    found.append((f"Camera {i}", i, int(cv2.CAP_ANY)))
        finally:
            if prev is not None:
                cv2.utils.logging.setLogLevel(prev)
    return found


def _default_recordings_dir():
    base = Path.home() / ("Movies" if sys.platform == "darwin" else "Videos")
    if not base.is_dir():
        base = Path.home()
    return base / "BehavioralTracker Recordings"


class CameraWorker(QObject):
    """Reads the camera on its own thread (smooth preview at full camera speed) and, while recording,
    writes the ORIGINAL frames to a video file paced by the real clock (the file's length = real time).
    During a live trial every recorded frame is also handed to the delayed live tracker
    (tracking.live_delayed), which tracks the trial in pieces while it is being recorded."""
    frame = Signal(object)                 # QImage for the preview
    opened = Signal(int, int, float)       # width, height, measured fps
    failed = Signal(str)
    recorded = Signal(str, float)          # path, seconds
    empty_done = Signal(object)            # list of frames of the empty arena (background help)
    REC_FPS = (5, 10, 12, 15, 20, 24, 25, 30, 50, 60)

    def __init__(self, index, backend, size=None):
        super().__init__()
        self.index, self.backend, self.size = index, backend, size
        self._stop = False
        self._rec_path = None
        self._rec_stop = False
        self.fps = 0.0
        self.overlay = None                # tracking.live_delayed.ZoneOverlay: zones on the preview
        self.delayed = None                # tracking.live_delayed.DelayedLiveTracker during a live trial
        self._empty = None                 # empty-arena capture in progress
        self._cmds = []                    # commands from the window, applied on the next camera frame

    def set_overlay(self, overlay):
        self._cmds.append(("overlay", overlay))

    def capture_empty(self, seconds=2.0):
        self._cmds.append(("empty", float(seconds)))

    def start_trial(self, path, delayed=None):
        self._cmds.append(("start", str(path), delayed))

    def stop_trial(self):
        self._cmds.append(("stop",))

    def stop(self):
        self._stop = True

    def start_recording(self, path):
        self._cmds.append(("start", str(path), None))

    def stop_recording(self):
        self._cmds.append(("stop",))

    def _open(self):
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened() and self.backend != cv2.CAP_ANY:
            cap.release()
            cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            return None
        if self.size:
            w, h = self.size
            if w >= 1280:                               # USB cameras give 720p/1080p at full speed only as MJPG
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        return cap

    def run(self):
        from tracking.live_delayed import compose_view
        from tracking.location import draw_status_box
        cap = self._open()
        if cap is None:
            self.failed.emit("open")
            return
        writer, written, t_rec, rec_fps = None, 0, 0.0, 30
        t0, n, last_preview, announced, misses = time.time(), 0, 0.0, False, 0
        pace = None
        if isinstance(self.index, str):              # a video file standing in for a camera: play at its real speed
            pace = 1.0 / max(1.0, float(cap.get(cv2.CAP_PROP_FPS) or 30.0))
        try:
            while not self._stop:
                if pace is not None:
                    time.sleep(pace)
                ok, f = cap.read()
                now = time.time()
                if not ok or f is None:
                    misses += 1
                    if misses > 60:
                        self.failed.emit("lost")
                        break
                    time.sleep(0.02)
                    continue
                misses = 0
                n += 1
                if n >= 10:
                    self.fps = (n - 1) / max(1e-3, now - t0)
                if not announced and (now - t0) > 1.5:
                    announced = True
                    self.opened.emit(f.shape[1], f.shape[0], self.fps)
                # ---- commands, applied on THIS frame (a trial's recording and tracking start together)
                while self._cmds:
                    cmd = self._cmds.pop(0)
                    if cmd[0] == "overlay":
                        self.overlay = cmd[1]
                    elif cmd[0] == "empty":
                        self._empty = {"until": now + cmd[1], "every": cmd[1] / 20.0, "next": now, "frames": []}
                    elif cmd[0] == "start" and writer is None:
                        self._rec_stop = False
                        self._rec_path = cmd[1]
                        self.delayed = cmd[2]
                    elif cmd[0] == "stop" and self._rec_path:
                        self._rec_stop = True
                # ---- empty-arena pictures (about 20 over 2 s)
                if self._empty is not None:
                    e = self._empty
                    if now >= e["next"]:
                        e["frames"].append(f.copy())
                        e["next"] = now + e["every"]
                    if now >= e["until"] or len(e["frames"]) >= 20:
                        self._empty = None
                        self.empty_done.emit(e["frames"])
                # ---- recording (original frames, real-time pacing)
                if self._rec_path and writer is None and not self._rec_stop:
                    measured = self.fps or float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
                    rec_fps = min(self.REC_FPS, key=lambda v: abs(v - measured))
                    writer = cv2.VideoWriter(self._rec_path, cv2.VideoWriter_fourcc(*"mp4v"), float(rec_fps),
                                             (f.shape[1], f.shape[0]))
                    if not writer.isOpened():
                        writer = None
                        self._rec_path = None
                        if self.delayed is not None:
                            self.delayed.cancel()
                            self.delayed = None
                        self.failed.emit("write")
                    else:
                        written, t_rec = 0, now
                        if self.delayed is not None:
                            self.delayed.set_fps(rec_fps)
                            self.delayed.start()
                if writer is not None:
                    due = int((now - t_rec) * rec_fps) + 1
                    while written < due:                 # duplicate if the camera is slower than rec_fps
                        writer.write(f)
                        if self.delayed is not None:
                            self.delayed.add_frame(f, t_rec + written / float(rec_fps))
                        written += 1
                    if self._rec_stop:
                        writer.release()
                        writer = None
                        if self.delayed is not None:
                            self.delayed.finish()
                        self.recorded.emit(self._rec_path, written / float(rec_fps))
                        self._rec_path = None
                        self._rec_stop = False
                # ---- preview (max ~25 per second, scaled down -- the recording keeps full size)
                if now - last_preview > 0.04:
                    last_preview = now
                    p = self.overlay.draw(f) if self.overlay is not None else f
                    dl = self.delayed
                    if dl is not None:
                        if dl.phase in ("tracking", "finishing") and dl.image is not None:
                            p = compose_view(dl.image, p, dl.lag_s)
                        elif dl.phase in ("recording", "learning") and writer is not None:
                            st = dl.status()
                            txt = (f"RECORDING  -  tracked view starts in {st['first_in_s']:.0f} s"
                                   if dl.phase == "recording" else "RECORDING  -  learning the background...")
                            p = draw_status_box(p.copy(), txt, (0, 0, 220))
                        if dl.phase in ("analysing", "done", "failed", "cancelled") and writer is None:
                            self.delayed = None
                    if p.shape[1] > 1280:
                        s = 1280.0 / p.shape[1]
                        p = cv2.resize(p, (1280, int(p.shape[0] * s)), interpolation=cv2.INTER_AREA)
                    self.frame.emit(bgr_to_qimage(p))
        finally:
            if writer is not None:
                writer.release()
                if self.delayed is not None:
                    self.delayed.finish()
                self.recorded.emit(self._rec_path, written / float(rec_fps))
            cap.release()


class CameraDialog(QDialog):
    """Live camera: choose a camera by its real name (built-in or USB), preview it and record.
    Every recording is saved as an ORIGINAL video file in the recordings folder.  A live trial is also
    tracked while it is recorded (delayed live tracking: the tracked view runs ~30 s behind the camera and
    the results are ready a few seconds after the trial).  `result_paths` = the files recorded,
    `live_results` = [(video, summary)] of the trials tracked live."""

    RESOLUTIONS = [("Camera default", None), ("640 × 480", (640, 480)), ("1280 × 720 (HD)", (1280, 720)),
                   ("1920 × 1080 (Full HD)", (1920, 1080))]

    def __init__(self, parent, cfg=None, name_hint="", ctrl=None, autostart=False):
        super().__init__(parent)
        self.cfg = cfg if cfg is not None else {}
        self.ctrl = ctrl
        self.setWindowTitle("Live camera · live tracking" if LIVE_TRACKING else "Live camera")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(1340, 800)
        self.result_paths = []
        self.live_results = []             # [(video path, summary)] -- trials tracked live
        self.action = None                 # None | "zones" | "analyse" (track a trial afterwards) | "results"
        self.zones_path = None
        self.trial_path = None
        self._rec_mode = None              # "plain" | "trial" | "zones"
        self._frame_size = None
        self._fps = 30.0
        self._empty_frames = []
        self.delayed = None                # DelayedLiveTracker of the running / finishing trial
        self.thread = None
        self.worker = None
        self._recording = False
        self._rec_started = 0.0
        self.rec_dir = Path(self.cfg.get("camera_dir") or _default_recordings_dir())
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        top = QHBoxLayout()
        top.addWidget(QLabel("Camera:"))
        self.combo = QComboBox()
        self.combo.setMinimumWidth(320)
        top.addWidget(self.combo, 1)
        b_refresh = QToolButton()
        b_refresh.setText("↻")
        b_refresh.setToolTip("Look for cameras again (after plugging in a USB camera).")
        b_refresh.clicked.connect(self.refresh_cameras)
        top.addWidget(b_refresh)
        top.addWidget(QLabel("Resolution:"))
        self.cmb_res = QComboBox()
        for text, size in self.RESOLUTIONS:
            self.cmb_res.addItem(text, size)
        self.cmb_res.setCurrentIndex(int(self.cfg.get("camera_res", 0)))
        top.addWidget(self.cmb_res)
        self.btn_preview = QPushButton("Start preview")
        self.btn_preview.clicked.connect(self._toggle_preview)
        top.addWidget(self.btn_preview)
        lay.addLayout(top)
        self.label = QLabel("Choose a camera and press Start preview")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(420)
        self.label.setStyleSheet("background:#1c212b;color:#8b95a7;border-radius:8px;")
        mid = QHBoxLayout()
        mid.addWidget(self.label, 1)
        live_panel = self._build_live_panel()
        live_panel.setVisible(LIVE_TRACKING)
        mid.addWidget(live_panel)
        lay.addLayout(mid, 1)
        self.view = FrameView(self.label, self)
        # where the original videos go
        row = QHBoxLayout()
        row.addWidget(QLabel("Save original videos to:"))
        self.lbl_dir = QLabel()
        self.lbl_dir.setStyleSheet("color:#000000;")
        self.lbl_dir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.lbl_dir, 1)
        b_dir = _small_button("Change…", "Choose the folder for the recorded (original) videos.")
        b_dir.clicked.connect(self._choose_dir)
        b_open = _small_button("📂 Open", "Open the recordings folder.")
        b_open.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._ensure_dir()))))
        row.addWidget(b_dir)
        row.addWidget(b_open)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("File name:"))
        self.le_name = QLineEdit(name_hint or "recording")
        self.le_name.setToolTip("Start of the file name; date and time are added automatically.")
        self.le_name.setMaximumWidth(260)
        row2.addWidget(self.le_name)
        self.lbl_example = QLabel()
        self.lbl_example.setStyleSheet("color:#64748b;")
        row2.addWidget(self.lbl_example, 1)
        lay.addLayout(row2)
        self.le_name.textChanged.connect(self._update_example)
        bottom = QHBoxLayout()
        self.status = QLabel("")
        self.status.setWordWrap(True)
        bottom.addWidget(self.status, 1)
        self.btn_record = QPushButton("●  Start recording")
        self.btn_record.setEnabled(False)
        self.btn_record.setMinimumHeight(38)
        self.btn_record.setToolTip("Record only (no live tracking) -- the video is tracked afterwards.")
        self.btn_record.clicked.connect(self._toggle_record)
        self.btn_done = QPushButton("Close")
        self.btn_done.setMinimumHeight(38)
        self.btn_done.clicked.connect(self.accept)
        bottom.addWidget(self.btn_record)
        bottom.addWidget(self.btn_done)
        lay.addLayout(bottom)
        self.rec_timer = QTimer(self)
        self.rec_timer.timeout.connect(self._rec_tick)
        self.trial_timer = QTimer(self)
        self.trial_timer.timeout.connect(self._trial_tick)
        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._poll_live)
        self._set_dir(self.rec_dir)
        self._style_record(False)
        self._style_trial(False)
        self._live_enable(False)
        self.refresh_cameras()
        if autostart and self.btn_preview.isEnabled():
            QTimer.singleShot(200, self._toggle_preview)

    # ------------------------------------------------------------------ live tracking panel
    def _build_live_panel(self):
        panel = QFrame()
        panel.setObjectName("livePanel")
        panel.setFixedWidth(380)
        panel.setStyleSheet("QFrame#livePanel{background:#ffffff;border:1px solid #e3e8f0;border-radius:8px;}")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(1, 1, 1, 1)
        scroll = QScrollArea(panel)
        scroll.setObjectName("liveScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body.setObjectName("liveBody")
        scroll.setStyleSheet("QScrollArea#liveScroll{background:transparent;border:none;} "
                             "QWidget#liveBody{background:#ffffff;}")
        scroll.setWidget(body)
        outer.addWidget(scroll)
        self._live_scroll = scroll
        v = QVBoxLayout(body)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)
        t = QLabel("Live tracking")
        t.setStyleSheet("font-size:14pt;font-weight:700;color:#0f172a;")
        v.addWidget(t)
        self.lbl_how = QLabel()
        self.lbl_how.setWordWrap(True)
        self.lbl_how.setStyleSheet("color:#64748b;font-size:9pt;")
        v.addWidget(self.lbl_how)

        def head(text):
            h = QLabel(text)
            h.setStyleSheet("font-weight:700;color:#0f172a;margin-top:6px;")
            v.addWidget(h)
        head("1.  Zones")
        self.lbl_zones = QLabel()
        self.lbl_zones.setWordWrap(True)
        self.lbl_zones.setStyleSheet("color:#334155;")
        v.addWidget(self.lbl_zones)
        self.btn_zones = QPushButton("Set zones on this camera…")
        self.btn_zones.setToolTip("Records 3 seconds of the camera and opens it in Video Setup: fit the zones there "
                                  "(⚡ Auto / 🎯 Guided / ▦ Preset / ✂ Fragment), then open Live camera again.")
        self.btn_zones.clicked.connect(self._zones_from_camera)
        v.addWidget(self.btn_zones)
        head("2.  Empty arena (optional)")
        self.btn_bg_empty = QPushButton("Capture empty arena (2 s)")
        self.btn_bg_empty.setToolTip("Best before you put the animal in: 2 seconds of the empty arena help the "
                                     "background, above all if the animal sits still at the start of the trial.")
        self.btn_bg_empty.clicked.connect(self._capture_empty)
        v.addWidget(self.btn_bg_empty)
        self.lbl_bg = QLabel()
        self.lbl_bg.setWordWrap(True)
        self.lbl_bg.setStyleSheet("color:#64748b;")
        v.addWidget(self.lbl_bg)
        head("3.  Trial")
        g = QGridLayout()
        g.setHorizontalSpacing(6)
        g.addWidget(QLabel("Length:"), 0, 0)
        self.spin_min = QSpinBox()
        self.spin_min.setRange(0, 240)
        self.spin_min.setSuffix(" min")
        self.spin_min.setValue(int(self.cfg.get("trial_min", 5)))
        self.spin_sec = QSpinBox()
        self.spin_sec.setRange(0, 59)
        self.spin_sec.setSuffix(" s")
        self.spin_sec.setValue(int(self.cfg.get("trial_sec", 0)))
        for sp in (self.spin_min, self.spin_sec):
            sp.setToolTip("Trial length; 0 min 0 s = until you press Stop.")
        g.addWidget(self.spin_min, 0, 1)
        g.addWidget(self.spin_sec, 0, 2)
        g.addWidget(QLabel("Start delay:"), 1, 0)
        self.spin_delay = QSpinBox()
        self.spin_delay.setRange(0, 600)
        self.spin_delay.setSuffix(" s")
        self.spin_delay.setValue(int(self.cfg.get("trial_delay", 0)))
        self.spin_delay.setToolTip("Seconds to wait after pressing Start (time to put the animal in and step away).")
        g.addWidget(self.spin_delay, 1, 1)
        lbl_live = QLabel("Live delay:")
        g.addWidget(lbl_live, 2, 0)
        self.spin_live = QSpinBox()
        self.spin_live.setRange(10, 120)
        self.spin_live.setSingleStep(5)
        self.spin_live.setSuffix(" s")
        self.spin_live.setValue(int(self.cfg.get("live_delay", 30)))
        self.spin_live.setToolTip("The trial is tracked in pieces of this length: each piece is tracked while the "
                                  "next one is recorded, so the tracked view runs this far behind the camera. "
                                  "30 s suits most computers.")
        self.spin_live.valueChanged.connect(lambda _v: (self._update_how(), self._update_bg_label()))
        g.addWidget(self.spin_live, 2, 1)
        lbl_live.setVisible(False)                 # fixed at 30 s (explained in the text above), not a setting
        self.spin_live.setVisible(False)
        self.spin_live.setValue(30)
        v.addLayout(g)
        self.chk_show = QCheckBox("Show the results when the trial ends")
        self.chk_show.setChecked(bool(self.cfg.get("trial_show", True)))
        self.chk_show.setToolTip("Open the Analysis page with the results (Excel, CSV, charts and the BT analysis "
                                 "file are saved next to the recording either way).")
        v.addWidget(self.chk_show)
        self.btn_trial = QPushButton()
        self.btn_trial.setMinimumHeight(42)
        self.btn_trial.clicked.connect(self._toggle_trial)
        v.addWidget(self.btn_trial)
        self.lbl_clock = QLabel("00:00:00")
        self.lbl_clock.setAlignment(Qt.AlignCenter)
        self.lbl_clock.setStyleSheet("font-size:20pt;font-weight:700;color:#0f172a;")
        v.addWidget(self.lbl_clock)
        self.lbl_live = QLabel("")
        self.lbl_live.setWordWrap(True)
        self.lbl_live.setStyleSheet("color:#334155;")
        v.addWidget(self.lbl_live)
        self.tbl_live = QTableWidget(0, 4)
        self.tbl_live.setHorizontalHeaderLabels(["Zone", "Time (s)", "%", "Entries"])
        self.tbl_live.verticalHeader().setVisible(False)
        self.tbl_live.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_live.setSelectionMode(QAbstractItemView.NoSelection)
        hh = self.tbl_live.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.tbl_live.setMinimumHeight(170)
        v.addWidget(self.tbl_live, 1)
        self._update_zone_info()
        self._update_how()
        self._update_bg_label()
        return panel

    def _update_how(self):
        d = self.spin_live.value() if hasattr(self, "spin_live") else 30
        self.lbl_how.setText(f"The trial is recorded (original video) and tracked at the same time, in {d}-second "
                             f"pieces: while one piece is tracked the next one is recorded. The tracked view runs "
                             f"about {d} s behind the camera (the live camera is shown in the corner), and the full "
                             "results are ready a few seconds after the trial ends.")

    def _update_bg_label(self):
        d = self.spin_live.value() if hasattr(self, "spin_live") else 30
        if self._empty_frames:
            self.lbl_bg.setText(f"✓ {len(self._empty_frames)} pictures of the empty arena — they are used with "
                                f"the first {d} s of the trial for the background. Now put the animal in.")
        else:
            self.lbl_bg.setText(f"Not needed: the background is learned from the first {d} s of the trial (the "
                                "animal should move around). Capturing the empty arena first makes it even "
                                "safer.")

    def _update_zone_info(self):
        c = self.ctrl
        if c is None:
            self.lbl_zones.setText("Zones come from Video Setup.")
            return
        n = len(c.canvas.zones)
        test = TESTS.get(c.test_key, ("",))[0]
        src = Path(c.player.path).name if getattr(c.player, "path", None) else ""
        if n:
            self.lbl_zones.setText(f"<b>{n} zone(s)</b> · {test}, from Video Setup. Check that the outlines fit "
                                   "the apparatus on the camera picture.")
            self.lbl_zones.setToolTip(f"Zones drawn on: {src}" if src else "")
        else:
            self.lbl_zones.setText("No zones yet. Press 'Set zones on this camera…' first "
                                   "(without zones only distance and speed are measured).")

    def _busy(self):
        return self._rec_mode is not None or getattr(self, "_countdown", 0) > 0 or \
            (self.delayed is not None and self.delayed.phase not in ("done", "failed", "cancelled"))

    def _live_enable(self, on):
        busy = self._busy()
        for w in (self.btn_zones, self.btn_bg_empty, self.spin_live):
            w.setEnabled(on and not busy)
        trial_running = self._rec_mode == "trial" or getattr(self, "_countdown", 0) > 0
        finishing = self._rec_mode is None and self.delayed is not None and \
            self.delayed.phase not in ("done", "failed", "cancelled")
        self.btn_trial.setEnabled(on and not finishing and (trial_running or self._rec_mode is None))

    def _style_trial(self, running):
        if running:
            self.btn_trial.setText("■  Stop trial")
            self.btn_trial.setStyleSheet("background:#dc2626;color:white;font-weight:700;border:none;"
                                         "border-radius:6px;font-size:11pt;")
        else:
            self.btn_trial.setText("▶  Start live trial")
            self.btn_trial.setStyleSheet("QPushButton{background:#16a34a;color:white;font-weight:700;border:none;"
                                         "border-radius:6px;font-size:11pt;} QPushButton:disabled{background:#cbd5e1;}")

    def _live_setup(self):
        w, h = self._frame_size
        return self.ctrl._build_setup("live_camera", (int(w), int(h)))

    def _make_overlay(self, w, h):
        from tracking.live_delayed import ZoneOverlay
        if self.ctrl is None or self.worker is None:
            return
        self._frame_size = (int(w), int(h))
        try:
            self.worker.set_overlay(ZoneOverlay(self._live_setup(), int(w), int(h)))
        except Exception as exc:
            print(f"Zone overlay skipped: {exc}")

    def _capture_empty(self):
        if self.worker is None:
            return
        self.btn_bg_empty.setEnabled(False)
        self.lbl_bg.setText("Capturing the empty arena…")
        self.worker.capture_empty(2.0)

    def _empty_done(self, frames):
        self._empty_frames = list(frames or [])
        self._update_bg_label()
        self._live_enable(self.worker is not None)

    def _trial_len(self):
        return self.spin_min.value() * 60 + self.spin_sec.value()

    def _toggle_trial(self):
        if self._rec_mode == "trial" or getattr(self, "_countdown", 0) > 0:
            self._stop_trial()
            return
        if self.worker is None or self._frame_size is None:
            return
        self.cfg.update(trial_min=self.spin_min.value(), trial_sec=self.spin_sec.value(),
                        trial_delay=self.spin_delay.value(), trial_show=self.chk_show.isChecked(),
                        live_delay=self.spin_live.value())
        self._countdown = self.spin_delay.value()
        self._rec_mode = "trial"
        self._style_trial(True)
        self.btn_record.setEnabled(False)
        self.btn_done.setEnabled(False)
        self._live_enable(True)
        self.tbl_live.setRowCount(0)
        self.lbl_live.setText("")
        if self._countdown > 0:
            self.lbl_clock.setText(f"Starts in {self._countdown} s")
            self.trial_timer.start(1000)
        else:
            self._begin_trial()

    def _begin_trial(self):
        from tracking.live_delayed import DelayedLiveTracker
        self._countdown = 0
        path = self._ensure_dir() / f"{self._file_stem()}_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4"
        self.trial_path = str(path)
        self._trial_t0 = time.time()
        self.delayed = None
        try:
            w, h = self._frame_size
            self.delayed = DelayedLiveTracker(self._live_setup(), str(path), w, h, self.worker.fps or 30.0,
                                              piece_s=self.spin_live.value(), empty_frames=self._empty_frames)
        except Exception as exc:                      # still record the trial; it is tracked afterwards
            print(f"Live tracking unavailable: {exc}")
            self.lbl_live.setText(f"<span style='color:#dc2626;'>Live tracking unavailable ({exc}) — the trial is "
                                  "recorded and tracked afterwards.</span>")
        self._empty_frames = []
        self.worker.start_trial(path, self.delayed)
        self._live_scroll.ensureWidgetVisible(self.tbl_live, 0, 0)
        self.status.setText(f"<span style='color:#dc2626;font-weight:700;'>● LIVE TRIAL</span> — recording the "
                            f"original video to {path.name} and tracking it")
        self.trial_timer.start(250)
        self.live_timer.start(250)

    def _trial_tick(self):
        if getattr(self, "_countdown", 0) > 0:
            self._countdown -= 1
            if self._countdown > 0:
                self.lbl_clock.setText(f"Starts in {self._countdown} s")
            else:
                self.trial_timer.stop()
                self._begin_trial()
            return
        ln = self._trial_len()
        if self._rec_mode == "trial":
            el = time.time() - self._trial_t0
            self.lbl_clock.setText(fmt_clock(el, True) + (f" / {fmt_clock(ln, True)}" if ln else ""))
            if ln and el >= ln:
                self._stop_trial()

    def _stop_trial(self):
        self.trial_timer.stop()
        if getattr(self, "_countdown", 0) > 0:          # stopped during the start delay: nothing recorded
            self._countdown = 0
            self._rec_mode = None
            self._style_trial(False)
            self.btn_record.setEnabled(True)
            self.btn_done.setEnabled(True)
            self.lbl_clock.setText("00:00:00")
            self._live_enable(True)
            return
        if self.worker is not None:
            self.worker.stop_trial()
        self.status.setText("Trial finished — saving the original video…")
        self._style_trial(False)
        self.btn_trial.setEnabled(False)

    def _fill_live_table(self, s):
        from tracking.location import safe_col
        names = list(getattr(self.delayed, "roi_names", []) or [])
        rows = []
        for n in names:
            k = safe_col(n)
            if f"{k}_time_s" in s:
                rows.append((n, s.get(f"{k}_time_s", 0.0), s.get(f"{k}_percent", 0.0), s.get(f"{k}_entries", "")))
        self.tbl_live.setRowCount(len(rows))
        for r, (n, t, pct, e) in enumerate(rows):
            for c, val in enumerate((n, f"{float(t):.1f}", f"{float(pct):.0f}", "" if e == "" else str(int(e)))):
                it = QTableWidgetItem(val)
                if c:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_live.setItem(r, c, it)

    def _numbers_text(self, s):
        unit = "cm" if "Total_distance_cm" in s else "px"
        dist = float(s.get("Total_distance_cm", s.get("Total_distance_pixels", 0.0)) or 0.0)
        parts = [f"Distance: <b>{dist:.1f} {unit}</b>",
                 f"detected {float(s.get('Tracking_quality_percent') or 0):.0f} %"]
        test = s.get("Test_type") if s.get("Test_type") in TESTS else (self.ctrl.test_key if self.ctrl else None)
        for key, caption, fmt in UXController.KPI_BY_TEST.get(test, []):
            if key and s.get(key) is not None and str(s.get(key)) != "nan":
                try:
                    parts.append(f"{caption}: <b>{fmt.format(float(s[key]))}</b>")
                except (TypeError, ValueError):
                    pass
        return " · ".join(parts)

    def _poll_live(self):
        d = self.delayed
        if d is None:
            self.live_timer.stop()
            return
        st = d.status()
        ph = st["phase"]
        tracked = fmt_clock(st["processed_s"], True)
        if ph == "recording":
            head = f"Recording the first {d.piece_s:.0f} s — the tracked view starts in {st['first_in_s']:.0f} s."
        elif ph == "learning":
            head = "Learning the background from the first piece…"
        elif ph == "tracking":
            head = f"Tracked up to <b>{tracked}</b> · the tracked view is {st['lag_s']:.0f} s behind the camera."
        elif ph == "finishing":
            head = (f"Trial over — tracking the last {st['remaining_s']:.0f} s at full speed "
                    f"(tracked up to {tracked})…")
        elif ph == "analysing":
            head = "Writing the results (Excel, CSV, charts, BT analysis file)…"
        elif ph == "done":
            head = "✓ Results ready."
        elif ph == "failed":
            head = f"<span style='color:#dc2626;'>{st['error']}</span>"
        else:
            head = ""
        nums = st.get("numbers")
        txt = head
        if nums:
            txt += "<br>" + self._numbers_text(nums)
            self._fill_live_table(nums)
        if st.get("background") and ph in ("tracking", "finishing"):
            txt += f"<br><span style='color:#64748b;font-size:9pt;'>{st['background']}</span>"
        self.lbl_live.setText(txt)
        if self._rec_mode is None and ph in ("done", "failed", "cancelled"):
            self.live_timer.stop()
            self._trial_finished(d)

    def _trial_finished(self, d):
        """The recording is saved and the live tracking has ended (results written, or failed)."""
        self.delayed = None
        path = self.trial_path
        self.btn_record.setEnabled(self.worker is not None)
        self.btn_done.setEnabled(True)
        if d.phase == "done" and d.summary:
            self.live_results.append((path, d.summary))
            if path not in self.result_paths:
                self.result_paths.append(path)
            if self.chk_show.isChecked():
                self.action = "results"
                self.accept()
                return
            self.status.setText(f"✓ Trial tracked — results saved in {d.summary.get('Output_folder', '')}<br>"
                                "Start another trial, or press <b>Close</b>.")
            self.btn_done.setText(f"Close ({len(self.live_results)} trial(s) tracked)")
        elif d.phase == "failed":
            if path and os.path.exists(path):
                if self.chk_show.isChecked():
                    self.action = "analyse"           # track the saved recording like any video instead
                    self.accept()
                    return
                self.result_paths.append(path)
            self.status.setText("The recording is saved; it will be tracked like any video when you close this "
                                "window.")
        self._live_enable(self.worker is not None)

    def _zones_from_camera(self):
        if self.worker is None:
            return
        path = self._ensure_dir() / f"{self._file_stem()}_zones_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4"
        self._rec_mode = "zones"
        self.zones_path = str(path)
        self.worker.start_recording(path)
        self._live_enable(True)
        self.btn_record.setEnabled(False)
        self.status.setText("Recording 3 seconds of the camera for the zone setup…")
        QTimer.singleShot(3000, lambda: self.worker.stop_recording() if self.worker is not None else None)

    # ------------------------------------------------------------------ cameras
    def refresh_cameras(self):
        if self.worker is not None:
            return
        prev = self.combo.currentText() or str(self.cfg.get("camera_name") or "")
        self.combo.clear()
        self.status.setText("Looking for cameras…")
        QApplication.processEvents()
        cams = list_cameras()
        for name, idx, be in cams:
            self.combo.addItem(name, (idx, be))
        if cams:
            i = self.combo.findText(prev)
            self.combo.setCurrentIndex(i if i >= 0 else 0)
            self.status.setText(f"{len(cams)} camera(s) found. Plugged in a USB camera just now? Press ↻.")
        else:
            self.status.setText("No camera found. Connect a camera and press ↻. On Windows also check "
                                "Settings ▸ Privacy ▸ Camera ▸ 'Let desktop apps access your camera'.")
        self.btn_preview.setEnabled(bool(cams))

    def _toggle_preview(self):
        if self.worker is None:
            data = self.combo.currentData()
            if data is None:
                return
            self.cfg["camera_res"] = self.cmb_res.currentIndex()
            self.cfg["camera_name"] = self.combo.currentText()
            idx, be = data
            self.thread = QThread(self)
            self.worker = CameraWorker(idx, be, self.cmb_res.currentData())
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.frame.connect(self.view.set_qimage)
            self.worker.opened.connect(self._opened)
            self.worker.failed.connect(self._failed)
            self.worker.recorded.connect(self._recorded)
            self.worker.empty_done.connect(self._empty_done)
            self.thread.start()
            self.status.setText(f"Starting {self.combo.currentText()}…")
            self.btn_preview.setText("Stop preview")
            for w in (self.combo, self.cmb_res):
                w.setEnabled(False)
        else:
            if self._busy():
                QMessageBox.information(self, "Live camera", "Stop the trial / recording first.")
                return
            self._stop_camera()

    def _opened(self, w, h, fps):
        self.btn_record.setEnabled(True)
        self._fps = fps or 30.0
        self.status.setText(f"{self.combo.currentText()}  ·  {w} × {h}  ·  {fps:.0f} frames/s")
        self._make_overlay(w, h)
        self._live_enable(True)

    def _failed(self, why):
        msg = {"open": "Could not open this camera. Close other programs that use it (Zoom, Teams, OBS…) "
                       "and try again, or pick another camera.",
               "lost": "The camera stopped sending pictures (unplugged?).",
               "write": "Cannot write the video file in the chosen folder — choose another folder."}.get(why, why)
        self._stop_camera()
        QMessageBox.warning(self, "Camera", msg)

    def _stop_camera(self):
        if self.worker is not None:
            if self._recording:
                self.worker.stop_recording()
            self.worker.stop()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
        self.thread = None
        self.worker = None
        self._recording = False
        self.rec_timer.stop()
        if hasattr(self, "trial_timer"):
            self.trial_timer.stop()
        self._style_record(False)
        self.btn_record.setEnabled(False)
        self.btn_preview.setText("Start preview")
        if hasattr(self, "btn_trial"):
            self._style_trial(False)
            self._live_enable(False)
        for w in (self.combo, self.cmb_res):
            w.setEnabled(True)

    # ------------------------------------------------------------------ recording
    def _ensure_dir(self):
        try:
            self.rec_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return self.rec_dir

    def _set_dir(self, d):
        self.rec_dir = Path(d)
        txt = str(self.rec_dir)
        self.lbl_dir.setText(txt if len(txt) < 70 else "…" + txt[-68:])
        self.lbl_dir.setToolTip(txt)
        self.cfg["camera_dir"] = txt
        self._update_example()

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Save original videos to", str(self._ensure_dir()))
        if d:
            self._set_dir(d)

    def _file_stem(self):
        stem = "".join(ch if (ch.isalnum() or ch in "-_ ") else "_" for ch in self.le_name.text().strip())
        return (stem.strip() or "recording").replace(" ", "_")

    def _update_example(self):
        self.lbl_example.setText(f"→ {self._file_stem()}_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4")

    def _style_record(self, on):
        if on:
            self.btn_record.setText("■  Stop recording")
            self.btn_record.setStyleSheet("background:#dc2626;color:white;font-weight:700;border:none;"
                                          "border-radius:6px;padding:0 18px;")
        else:
            self.btn_record.setText("●  Start recording")
            self.btn_record.setStyleSheet("QPushButton{color:#dc2626;font-weight:700;padding:0 18px;} "
                                          "QPushButton:disabled{color:#b6bfcc;}")

    def _toggle_record(self):
        if self.worker is None:
            return
        if not self._recording:
            if self._busy():
                return
            path = self._ensure_dir() / f"{self._file_stem()}_{datetime.datetime.now():%Y%m%d_%H%M%S}.mp4"
            self._rec_mode = "plain"
            self._live_enable(True)
            self.worker.start_recording(path)
            self._recording = True
            self._rec_started = time.time()
            self._rec_path = str(path)
            self._style_record(True)
            self.rec_timer.start(500)
            self.btn_done.setEnabled(False)
            self._rec_tick()
        else:
            self.worker.stop_recording()
            self._recording = False
            self.rec_timer.stop()
            self._style_record(False)
            self.btn_done.setEnabled(True)

    def _rec_tick(self):
        s = int(time.time() - self._rec_started)
        self.status.setText(f"<span style='color:#dc2626;font-weight:700;'>● REC  {s // 3600:02d}:{s // 60 % 60:02d}:"
                            f"{s % 60:02d}</span>  —  recording the original video to {Path(self._rec_path).name}")

    def _recorded(self, path, secs):
        mode, self._rec_mode = self._rec_mode, None
        ok = bool(path and os.path.exists(path))
        if ok and path in self.result_paths:          # already kept (window closed during the recording)
            return
        if mode == "zones":
            if ok:
                self.action = "zones"
                self.accept()
            return
        if mode == "trial":
            if ok:
                self.trial_path = path
                self.status.setText(f"✓ Trial recorded ({fmt_clock(secs, True)}): {path}<br>Finishing the "
                                    "tracking…")
            if self.delayed is None:                  # live tracking was not available: track it afterwards
                self.btn_record.setEnabled(True)
                self.btn_done.setEnabled(True)
                if ok:
                    if self.chk_show.isChecked():
                        self.action = "analyse"
                        self.accept()
                        return
                    self.result_paths.append(path)
            self._live_enable(self.worker is not None)
            return
        self._live_enable(self.worker is not None)
        if ok:
            self.result_paths.append(path)
            self.status.setText(f"✓ Saved original video ({fmt_clock(secs, True)}): {path}<br>"
                                "Record another one, or press <b>Close</b> to track the recording(s).")
            self.btn_done.setText(f"Close and track ({len(self.result_paths)})")

    # ------------------------------------------------------------------ closing
    def accept(self):
        if self.delayed is not None and self.delayed.phase not in ("done", "failed", "cancelled"):
            self.delayed.cancel()
        self._stop_camera()
        super().accept()

    def reject(self):
        live_busy = self.delayed is not None and self.delayed.phase not in ("done", "failed", "cancelled")
        if self._rec_mode == "trial" or getattr(self, "_countdown", 0) > 0 or self._recording or live_busy:
            what = ("A live trial is running" if self._rec_mode == "trial" or getattr(self, "_countdown", 0) > 0
                    else "The live tracking is being finished" if live_busy else "A recording is running")
            if QMessageBox.question(self, "Live camera", f"{what}. Stop it and close this window?\n\n"
                                    "The video recorded so far is kept and can be tracked like any video.") \
                    != QMessageBox.Yes:
                return
            self.trial_timer.stop()
            self.live_timer.stop()
            self._countdown = 0
            keep = self.trial_path if (self._rec_mode == "trial" or live_busy) else getattr(self, "_rec_path", None)
            self._rec_mode = None
            if self.delayed is not None:
                self.delayed.cancel()
                self.delayed = None
            self._stop_camera()
            if keep and os.path.exists(keep) and keep not in self.result_paths:
                self.result_paths.append(keep)
        self._stop_camera()
        super().reject()

    def closeEvent(self, e):
        super().closeEvent(e)                   # -> reject(), which may keep the window open
        if e.isAccepted():
            self._stop_camera()



# ============================================================================= controller
def compute_video_background(path, crop=None, size=None, samples=40, start_s=None, end_s=None):
    """Animal-free background picture (uint8 BGR) of one video; crop = the
    Video-Setup crop dict or None.  Used for automatic zone detection."""
    from tracking.robust_tracker import build_background, detection_scale
    cap = cv2.VideoCapture(str(path))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W, H = int(cap.get(3)), int(cap.get(4))
        if size is None:
            size = (crop["w"], crop["h"]) if crop else (W, H)
        f0 = int(max(0, (start_s or 0.0) * fps))
        f1 = int(min(n - 1, end_s * fps)) if end_s else n - 1

        def read(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                return None
            return cv2.warpPerspective(f, crop["matrix"], (crop["w"], crop["h"])) if crop else f
        bg = build_background(read, np.linspace(f0, max(f0, f1), samples).astype(int), "gray", "auto",
                              scale=detection_scale(*size))
        return bg.full_res_bgr(*size)
    finally:
        cap.release()


# ----------------------------------------------------------------------------- Fragment / Merge (shared)
def canvas_fragment(parent, canvas):
    """PowerPoint-style Fragment on a ZoneCanvas.  Uses the selected zones (2 or more), otherwise all zones.
    Returns (number of source shapes, [new piece names]) or None (a message was shown)."""
    from tracking.fragment import fragment_shapes, letter, overlap_px
    title = "Fragment"
    sel = [n for n in canvas.selected if n in canvas.zones]
    src = sel if len(sel) >= 2 else list(canvas.zones)
    if len(src) < 2:
        QMessageBox.information(
            parent, title, "Draw two or more overlapping shapes first (＋ Draw ▸ Rectangle / Square / Circle / "
            "Triangle / Polygon), e.g. two crossing bars for a plus maze, or a small square inside a big one for "
            "an open field. Then press ✂ Fragment to cut them into pieces.")
        return None
    w, h = canvas.frame_size
    if w <= 0 or h <= 0:
        return None
    polys = [canvas.zones[n] for n in src]
    if overlap_px(polys, w, h) < 20:
        QMessageBox.information(parent, title, "These shapes do not overlap, so there is nothing to cut. Move "
                                "them so they overlap (like the parts of the apparatus), then press ✂ Fragment "
                                "again.")
        return None
    pieces = fragment_shapes(polys, w, h, eps=1.0)
    if not pieces:
        QMessageBox.warning(parent, title, "No pieces could be made from these shapes.")
        return None
    new = {n: p for n, p in canvas.zones.items() if n not in src}
    names, i = [], 0
    for poly in pieces:
        while letter(i) in new:
            i += 1
        nm = letter(i)
        i += 1
        new[nm] = poly
        names.append(nm)
    for nm in names:                           # fresh colours for the pieces
        canvas.colors.pop(nm, None)
    canvas.set_zones(new, record=True)
    canvas.select(names)
    return len(src), names


def ask_name_pieces(parent, n_src, names):
    return QMessageBox.question(
        parent, "Fragment", f"The {n_src} shapes were cut into {len(names)} pieces, named {', '.join(names)} for "
        "now.\n\nName them now? (You can also do it later: double-click a name in the zone list.)\n\nTip: "
        "pieces you do not need can be deleted, and pieces that belong together can be joined again with "
        "⊕ Merge (Shift+click to select several).", QMessageBox.Yes | QMessageBox.No,
        QMessageBox.Yes) == QMessageBox.Yes


def canvas_merge(parent, canvas):
    """Join the selected zones into one (PowerPoint 'Union').  Returns (new name, [merged names]) or None."""
    from tracking.fragment import merge_shapes
    sel = [n for n in canvas.selected if n in canvas.zones]
    if len(sel) < 2:
        QMessageBox.information(parent, "Merge", "Select two or more zones that touch (Shift+click them), then "
                                "choose ⊕ Merge.")
        return None
    w, h = canvas.frame_size
    poly = merge_shapes([canvas.zones[n] for n in sel], w, h, eps=1.0)
    if poly is None:
        QMessageBox.warning(parent, "Merge", "These zones do not touch, so they cannot become one zone.")
        return None
    new, done = {}, False
    for n, p in canvas.zones.items():
        if n in sel:
            if not done:
                new[sel[0]] = poly
                done = True
            continue
        new[n] = p
    canvas.set_zones(new, record=True)
    canvas.select([sel[0]])
    return sel[0], sel


class BatchZoneDialog(QDialog):
    """Zone editor for the batch queue: set the zones (and masks) of every
    video one after the other -- each video is shown with its own frame --
    then run the whole batch at once."""

    def __init__(self, ctrl, start_row=0):
        super().__init__(ctrl.win)
        self.ctrl = ctrl
        self.pos = int(start_row)
        self.cap = None
        self.n_frames = 0
        self.fps = 30.0
        self.dirty = False
        self.saved = 0
        self._loading = False
        self._busy = False
        self._alive = True
        self._last = None                     # (zones, masks, size) of the last video saved here
        self.setWindowTitle("Zones for each video")
        self.resize(1240, 800)
        self.setSizeGripEnabled(True)

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.lbl_title = QLabel()
        f = self.lbl_title.font()
        f.setPointSize(f.pointSize() + 2)
        f.setBold(True)
        self.lbl_title.setFont(f)
        left.addWidget(self.lbl_title)
        self.lbl_source = QLabel()
        self.lbl_source.setWordWrap(True)
        self.lbl_source.setStyleSheet("color:#475569;")
        left.addWidget(self.lbl_source)
        self.canvas = ZoneCanvas(self)
        self.canvas.covered_fn = lambda: self.ctrl.covered_names(self.canvas.zones)
        self.canvas.placeholder = "Could not read a frame from this video"
        left.addWidget(self.canvas, 1)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Frame:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(lambda _v: self._seek_timer.start(60))
        srow.addWidget(self.slider, 1)
        self.lbl_time = QLabel("00:00")
        self.lbl_time.setStyleSheet("color:#000000; font-weight:600;")
        srow.addWidget(self.lbl_time)
        left.addLayout(srow)
        self.lbl_hint = QLabel("")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("color:#1f6feb;")
        left.addWidget(self.lbl_hint)
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀  Previous video")
        self.btn_next = QPushButton("Save && next video  ▶")
        self.btn_next.setProperty("cssClass", "primary")
        self.btn_save = QPushButton("Save && close")
        self.btn_cancel = QPushButton("Close")
        for b in (self.btn_prev, self.btn_next, self.btn_save, self.btn_cancel):
            b.setAutoDefault(False)
            b.setDefault(False)
            b.setMinimumHeight(34)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addStretch(1)
        nav.addWidget(self.btn_cancel)
        nav.addWidget(self.btn_save)
        left.addLayout(nav)
        root.addLayout(left, 1)

        panel = QFrame()
        panel.setFixedWidth(300)
        pl = QVBoxLayout(panel)
        pl.setContentsMargins(6, 0, 0, 0)
        cap = QLabel(f"Zones  ·  {TESTS[ctrl.test_key][0]}")
        cap.setStyleSheet("font-weight:600;")
        pl.addWidget(cap)
        self.lst = QListWidget()
        self.lst.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst.itemSelectionChanged.connect(self._list_selected)
        self.lst.itemDoubleClicked.connect(lambda it: self._rename(it.data(Qt.UserRole)))
        pl.addWidget(self.lst, 1)
        grid = QGridLayout()
        self.btn_draw = QPushButton("＋ Draw zone")
        dm = QMenu(self.btn_draw)
        for text, tool in SHAPE_MENU:
            dm.addAction(text, lambda t=tool: self._draw(t, "zone"))
        self.btn_draw.setMenu(dm)
        self.btn_auto = QPushButton("⚡ Auto-detect")
        self.btn_auto.setToolTip("Find the apparatus in this video and fit the zones to it.")
        self.btn_auto.clicked.connect(self.auto_detect)
        self.btn_guided = QPushButton("🎯 Guided fit")
        self.btn_guided.setToolTip("Click the arm ends (mazes) or the floor corners (boxes): the zones are fitted "
                                   "exactly to your clicks, the arm width is measured on the video.")
        self.btn_guided.clicked.connect(self._guided_clicked)
        self.btn_preset = QPushButton("▦ Preset")
        self.btn_preset.setToolTip(f"Start from the {TESTS[ctrl.test_key][0]} preset, then drag / resize it.")
        self.btn_preset.clicked.connect(self._preset)
        self.btn_copy = QPushButton("⧉ Copy from…")
        self.copy_menu = QMenu(self.btn_copy)
        self.copy_menu.aboutToShow.connect(self._fill_copy_menu)
        self.btn_copy.setMenu(self.copy_menu)
        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(lambda: self._rename(self.canvas.selected[0] if self.canvas.selected else None))
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(lambda: self.canvas.delete_zones(list(self.canvas.selected)))
        self.btn_mask = QPushButton("Mask area")
        mm = QMenu(self.btn_mask)
        mm.addAction("Mask polygon — click points", lambda: self._draw("polygon", "mask"))
        mm.addAction("Mask rectangle — drag", lambda: self._draw("rect", "mask"))
        mm.addSeparator()
        mm.addAction("Clear all masks", self._clear_masks)
        self.btn_mask.setMenu(mm)
        self.btn_clear = QPushButton("Clear zones")
        self.btn_clear.clicked.connect(lambda: self.canvas.set_zones({}))
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.clicked.connect(self.canvas.undo)
        self.btn_redo = QPushButton("↷ Redo")
        self.btn_redo.clicked.connect(self.canvas.redo)
        self.btn_fragment = QPushButton("✂ Fragment")
        self.btn_fragment.setToolTip("Cut overlapping shapes into every separate piece (like PowerPoint ▸ "
                                     "Fragment). Pieces are named A, B, C… — rename them afterwards.")
        self.btn_fragment.clicked.connect(self._fragment)
        self.btn_merge = QPushButton("⊕ Merge")
        self.btn_merge.setToolTip("Join the selected zones (Shift+click) into one zone.")
        self.btn_merge.clicked.connect(lambda: canvas_merge(self, self.canvas))
        btns = [self.btn_auto, self.btn_guided, self.btn_draw, self.btn_preset, self.btn_fragment, self.btn_merge,
                self.btn_copy, self.btn_rename, self.btn_delete, self.btn_mask, self.btn_clear, self.btn_undo,
                self.btn_redo]
        for i, b in enumerate(btns):
            b.setAutoDefault(False)
            b.setProperty("cssClass", "secondary")
            grid.addWidget(b, i // 2, i % 2)
        pl.addLayout(grid)
        self.chk_automask = QCheckBox("Auto-mask outside zones")
        self.chk_automask.setToolTip("Ignore everything outside the zones (hands, reflections) for this video.")
        self.chk_automask.toggled.connect(self._automask_toggled)
        pl.addWidget(self.chk_automask)
        from pro_ui.chambers_ui import BatchChamberPanel
        self.chambers = BatchChamberPanel(self, pl, ctrl)     # this video's chambers (several animals)
        tip = QLabel("Click a zone to select it · drag points to reshape · double-click an edge = new point · "
                     "right-click a point = delete it · handles resize / rotate · wheel = zoom · Ctrl+Z = undo.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#64748b; font-size:9pt;")
        pl.addWidget(tip)
        root.addWidget(panel)

        self._seek_timer = QTimer(self)
        self._seek_timer.setSingleShot(True)
        self._seek_timer.timeout.connect(self._show_slider_frame)
        self.canvas.zonesEdited.connect(self._edited)
        self.canvas.selectionChanged.connect(self._sync_list_selection)
        self.canvas.drawingFinished.connect(self._drawing_finished)
        self.canvas.zoneRenameRequested.connect(self._rename)
        self.canvas.contextRequested.connect(self._context_menu)
        self.canvas.hoverInfo.connect(self.lbl_hint.setText)
        self.canvas.draftChanged.connect(lambda: self.lbl_hint.setText(self.canvas.current_banner()))
        self._bgs = {}                        # animal-free background per video (auto-detect / guided fit)
        self._bg_jobs = {}
        self._guided_shape = "rect"
        self.btn_prev.clicked.connect(lambda: self._go(self.pos - 1))
        self.btn_next.clicked.connect(self._save_next)
        self.btn_save.clicked.connect(self._save_close)
        self.btn_cancel.clicked.connect(self.reject)
        self._load(self.pos)

    # ---------------------------------------------------------------- video
    @property
    def item(self):
        return self.ctrl.batch_items[self.pos]

    def _load(self, pos):
        self._loading = True
        self.pos = pos
        it = self.item
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(it["path"])
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = it["size"]
        n = len(self.ctrl.batch_items)
        self.lbl_title.setText(f"Video {pos + 1} of {n}  —  {Path(it['path']).name}")
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(int(min(max(0, self.n_frames - 1), (it.get("start") or 0.0) * self.fps)))
        self.slider.blockSignals(False)
        self._show_slider_frame()
        self.canvas.cancel_tool(silent=True)
        self.canvas.undo_stack.clear()
        self.canvas.redo_stack.clear()
        self.canvas.colors.clear()
        self.canvas._color_i = 0
        if it.get("zones"):
            zones, masks = self.ctrl._item_zones_raw(it)
            src = "This video's own zones."
        elif self._last is not None:
            zones, masks = self._scaled(*self._last, (w, h))
            src = "Started from the previous video's zones — drag them onto this apparatus, then Save."
        elif self.ctrl.canvas.zones:
            zones, masks = self.ctrl._setup_zones_for_size((w, h))
            src = "Started from the Video-Setup layout — drag them onto this apparatus, then Save."
        else:
            zones, masks = default_zones(self.ctrl.test_key, w, h), []
            src = f"Started from the {TESTS[self.ctrl.test_key][0]} preset — fit it to the apparatus, then Save."
        self.canvas.masks = [list(m) for m in masks]
        self.canvas.set_zones(zones, record=False)
        self.canvas.select([])
        self.canvas.reset_view()
        self.chambers.load(it)
        am = it.get("automask")
        self.chk_automask.blockSignals(True)
        self.chk_automask.setChecked(self.ctrl.chk_automask.isChecked() if am is None else bool(am))
        self.chk_automask.blockSignals(False)
        self.canvas.auto_mask = self.chk_automask.isChecked()
        self.lbl_source.setText(src)
        self.lbl_hint.setText("")
        self.btn_prev.setEnabled(pos > 0)
        self.btn_next.setText("Save && next video  ▶" if pos < n - 1 else "Save (last video)")
        self.dirty = False
        self._loading = False
        self._refresh_list()

    @staticmethod
    def _scaled(zones, masks, src_size, dst_size):
        sx, sy = dst_size[0] / max(1, src_size[0]), dst_size[1] / max(1, src_size[1])
        return ({k: [(x * sx, y * sy) for x, y in v] for k, v in zones.items()},
                [[(x * sx, y * sy) for x, y in m] for m in masks])

    def _show_slider_frame(self):
        if self.cap is None:
            return
        idx = self.slider.value()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.canvas.set_frame(frame)
            self._frame = frame
        self.lbl_time.setText(f"{fmt_clock(idx / self.fps)} / {fmt_clock(self.n_frames / self.fps)}")

    # ---------------------------------------------------------------- editing
    def _edited(self):
        if not self._loading:
            self.dirty = True
        self._refresh_list()

    def _refresh_list(self):
        self.lst.blockSignals(True)
        self.lst.clear()
        for name in self.canvas.zones:
            it = QListWidgetItem(f"●  {name}")
            it.setForeground(QBrush(self.canvas.color_of(name)))
            it.setData(Qt.UserRole, name)
            self.lst.addItem(it)
            it.setSelected(name in self.canvas.selected)
        self.lst.blockSignals(False)
        self.btn_rename.setEnabled(len(self.canvas.selected) == 1)
        self.btn_delete.setEnabled(bool(self.canvas.selected))

    def _list_selected(self):
        self.canvas.select([it.data(Qt.UserRole) for it in self.lst.selectedItems()])

    def _sync_list_selection(self):
        self._refresh_list()

    def _draw(self, tool, purpose):
        self.canvas.select([])
        tips = {"polygon": "Click the corners; close it by clicking the first point, double-click or Enter. "
                           "Esc cancels.",
                "freehand": "Press and drag around the area, release to finish. Esc cancels.",
                "rect": "Drag a rectangle (Shift = square). Esc cancels.",
                "square": "Drag a square. Esc cancels.",
                "circle": "Drag a circle. Esc cancels.",
                "triangle": "Drag a triangle (Shift = equilateral). Esc cancels.",
                "ellipse": "Drag an ellipse (Shift = circle). Esc cancels."}
        self.canvas.start_tool(tool, purpose, banner=tips[tool])
        self.lbl_hint.setText(tips[tool])

    def _drawing_finished(self, purpose, pts):
        self.lbl_hint.setText("")
        if purpose == "chamber":
            self.chambers.rect_drawn(pts)
        elif purpose == "guided":
            self._apply_guided(pts)
        elif purpose == "zone":
            name = self._ask_name(title="New zone")
            if name:
                self.canvas.add_zone(self.chambers.zone_name(name, pts), pts)
        elif purpose == "mask":
            self.canvas.push_undo()
            self.canvas.masks.append([(float(x), float(y)) for x, y in pts])
            self.canvas.update()
            self._edited()

    def _ask_name(self, default=None, title="Zone name", exclude=()):
        existing = [n for n in self.canvas.zones if n not in exclude]
        sug = zone_suggestions(self.ctrl.test_key, existing)
        if default:
            sug = [default] + [s for s in sug if s != default]
        items = [s for s in sug if s not in existing] or [unique_name("Zone", existing)]
        while True:
            name, ok = QInputDialog.getItem(self, title, "Zone name (pick or type):", items, 0, True)
            if not ok:
                return None
            name = name.strip()
            if name and name not in existing:
                return name
            if name:
                QMessageBox.warning(self, title, f"A zone called '{name}' already exists.")

    def _rename(self, name):
        if not name or name not in self.canvas.zones:
            return
        new = self._ask_name(default=name, title=f"Rename '{name}'", exclude=(name,))
        if new and new != name:
            self.canvas.rename_zone(name, new)

    def _fragment(self):
        res = canvas_fragment(self, self.canvas)
        if res and ask_name_pieces(self, *res):
            dlg = ZoneNamesDialog(self, list(self.canvas.zones), zone_suggestions(self.ctrl.test_key),
                                  epm_swap=self.ctrl.test_key == "epm")
            if dlg.exec() == QDialog.Accepted:
                mp = dlg.mapping()
                if any(k != v for k, v in mp.items()):
                    self.canvas.push_undo()
                    cols = {mp.get(k, k): self.canvas.color_of(k) for k in self.canvas.zones}
                    self.canvas.zones = {mp.get(k, k): v for k, v in self.canvas.zones.items()}
                    self.canvas.colors.update(cols)
                    self.canvas.selected = [mp.get(n, n) for n in self.canvas.selected]
                    self.canvas.update()
                    self.canvas.zonesEdited.emit()

    def _context_menu(self, name, gpos):
        menu = QMenu(self)
        if name:
            menu.addAction(f"Rename '{name}'…", lambda: self._rename(name))
            menu.addAction("Duplicate", self.canvas.duplicate_selected)
            menu.addAction("Delete", lambda: self.canvas.delete_zones(list(self.canvas.selected)))
            self.ctrl.add_covered_action(menu, name, self.canvas)
            menu.addSeparator()
        sub = menu.addMenu("＋ Draw a zone")
        for text, tool in tuple((t.split(" — ")[0], k) for t, k in SHAPE_MENU):
            sub.addAction(text, lambda t=tool: self._draw(t, "zone"))
        menu.addAction("⛶ Select all zones", lambda: self.canvas.select(list(self.canvas.zones)))
        if len(self.canvas.zones) >= 2:
            menu.addAction("✂ Fragment → pieces A, B, C…", self._fragment)
            if len(self.canvas.selected) >= 2:
                menu.addAction("⊕ Merge selected zones", lambda: canvas_merge(self, self.canvas))
        menu.addAction("Reset zoom", self.canvas.reset_view)
        menu.exec(gpos)

    def _clear_masks(self):
        if self.canvas.masks:
            self.canvas.push_undo()
            self.canvas.masks = []
            self.canvas.update()
            self._edited()

    def _automask_toggled(self, on):
        self.canvas.auto_mask = bool(on)
        self.canvas.update()
        if not self._loading:
            self.dirty = True

    def _preset(self):
        w, h = self.item["size"]
        if self.canvas.zones and QMessageBox.question(
                self, "Preset", f"Replace the zones with the {TESTS[self.ctrl.test_key][0]} preset?") != QMessageBox.Yes:
            return
        self.canvas.colors.clear()
        self.canvas._color_i = 0
        if self.chambers.rects:                         # the preset in every chamber of this video
            self.canvas.set_zones(self.chambers.preset(self.ctrl.test_key))
        else:
            self.canvas.set_zones(default_zones(self.ctrl.test_key, w, h), select_all=True)
        self.lbl_hint.setText("Preset placed — drag it onto the apparatus; corner handles resize, the round "
                              "handle rotates.")

    def _fill_copy_menu(self):
        m = self.copy_menu
        m.clear()
        size = tuple(self.item["size"])
        a = m.addAction("Video Setup layout",
                        lambda: self._apply_copy(*self.ctrl._setup_zones_for_size(size), "the Video-Setup layout"))
        a.setEnabled(bool(self.ctrl.canvas.zones))
        if self._last is not None:
            m.addAction("Previous video (this session)",
                        lambda: self._apply_copy(*self._scaled(*self._last, size), "the previous video"))
        m.addSeparator()
        for i, other in enumerate(self.ctrl.batch_items):
            if i == self.pos or not other.get("zones"):
                continue
            z, mk = self.ctrl._item_zones_raw(other)
            zz, mm = self._scaled(z, mk, tuple(other["size"]), size)
            m.addAction(Path(other["path"]).name, lambda zz=zz, mm=mm, n=Path(other["path"]).name:
                        self._apply_copy(zz, mm, n))
        if len(m.actions()) <= 2 and not self.ctrl.canvas.zones and self._last is None:
            m.addAction("(no other zones yet)").setEnabled(False)

    def _apply_copy(self, zones, masks, what):
        self.canvas.push_undo()
        self.canvas.masks = [list(mk) for mk in masks]
        self.canvas.set_zones(zones, record=False, select_all=True)
        self.lbl_hint.setText(f"Copied the zones of {what} — drag them onto this apparatus if it moved.")

    def _with_background(self, on_ready):
        """This video's animal-free background (cached per video), delivered async."""
        it = self.item
        path, size = it["path"], tuple(it["size"])
        if path in self._bgs:
            on_ready(self._bgs[path])
            return
        waiters = self._bg_jobs.setdefault(path, [])
        waiters.append(on_ready)
        if len(waiters) > 1:
            return

        def work():
            return compute_video_background(path, None, size, 40, it.get("start"), it.get("end"))

        def done(img):
            self._bgs[path] = img
            for cb in self._bg_jobs.pop(path, []):
                if self._alive and self.item["path"] == path:
                    cb(img)

        def failed(msg):
            self._bg_jobs.pop(path, None)
            if self._alive:
                QMessageBox.warning(self, "Background", msg)
        self.ctrl._thread_run(work, done, failed)

    def _guided_clicked(self):
        from tracking.apparatus import guided_spec
        test = self.ctrl.test_key
        shape = "rect"
        if test in ("open_field", "nor", "custom"):
            menu = QMenu(self)
            menu.addAction("Square / rectangular floor — click its 4 corners")
            a_circ = menu.addAction("Round arena — click 3 or more points on its edge")
            chosen = menu.exec(self.btn_guided.mapToGlobal(self.btn_guided.rect().bottomLeft()))
            if chosen is None:
                return
            shape = "circle" if chosen is a_circ else "rect"
        n, labels, prompts = guided_spec(test, shape)
        self._guided_shape = shape
        self.canvas.select([])
        self.canvas.start_tool("points", "guided", max_points=n, banner=prompts[0], labels=labels,
                               prompts=prompts, auto_finish=n is not None)
        self.lbl_hint.setText(self.canvas.current_banner())
        self._with_background(lambda _img: None)

    def _apply_guided(self, pts):
        from tracking.apparatus import zones_from_clicks
        test, shape, path = self.ctrl.test_key, self._guided_shape, self.item["path"]
        ppc = self.ctrl.spin_ppc.value() or None
        if ppc and self.ctrl.player.w and self.item["size"][0] != self.ctrl.player.w:
            ppc = ppc * self.item["size"][0] / self.ctrl.player.w

        def ready(img):
            if self.item["path"] != path:
                return
            zones, msg = zones_from_clicks(test, pts, img, ppc=ppc, shape=shape)
            if not zones:
                QMessageBox.information(self, "Guided fit", msg)
                return
            self.canvas.colors.clear()
            self.canvas._color_i = 0
            self.canvas.set_zones(zones, select_all=True)
            self.lbl_hint.setText("🎯 " + msg + "  Drag to fine-tune, then Save.")
        self._with_background(ready)

    def auto_detect(self):
        if self._busy:
            return
        self._busy = True
        self.btn_auto.setEnabled(False)
        self.btn_auto.setText("⚡ Detecting…")
        it = self.item
        path, size = it["path"], tuple(it["size"])
        test = self.ctrl.test_key
        ppc = self.ctrl.spin_ppc.value() or None
        if ppc and self.ctrl.player.w and size[0] != self.ctrl.player.w:
            ppc = ppc * size[0] / self.ctrl.player.w

        again = getattr(self, "_auto_lib_video", None) == path
        self._auto_lib_video = None

        def work():
            from tracking.apparatus import auto_detect_zones, guess_test
            img = self._bgs.get(path)
            if img is None:
                img = compute_video_background(path, None, size, 40, it.get("start"), it.get("end"))
                self._bgs[path] = img
            if not again:
                with _LEARN_LOCK:
                    hit = self.ctrl._match_learned(img, test)
                if hit is not None:
                    return hit[0], "learned", hit[2], hit[1]
            kind = test if test != "custom" else (guess_test(img) or "custom")
            z, o, m = auto_detect_zones(img, kind, ppc=ppc)
            if kind != test and z:
                m = f"Looks like a {TESTS[kind][0]}. " + m
            return z, o, m, None

        def done(res):
            self._busy = False
            if not self._alive:
                return
            self.btn_auto.setEnabled(True)
            self.btn_auto.setText("⚡ Auto-detect")
            if self.item["path"] != path:
                return
            zones, _outline, msg, entry = res
            if entry is not None:
                self.canvas.colors.clear()
                self.canvas._color_i = 0
                self.canvas.set_zones(zones, select_all=True)
                if entry.get("covered"):
                    self.ctrl.covered_override = set(entry["covered"])
                self._auto_lib_video = path
                self.lbl_hint.setText(f"⚡ Auto: zones of {msg} — check them, then Save. Press ⚡ again for the "
                                      "built-in template.")
                return
            if not zones:
                QMessageBox.information(self, "Auto-detect zones",
                                        msg + "\n\nUse ▦ Preset or ＋ Draw zone and fit the zones by hand.")
                return
            self.canvas.colors.clear()
            self.canvas._color_i = 0
            self.canvas.set_zones(zones, select_all=True)
            self.lbl_hint.setText("Auto-detect: " + msg + "  — check the zones, then Save.")
            if test == "epm":
                dlg = ZoneNamesDialog(self, list(zones), zone_suggestions("epm"), title="Confirm the EPM arms",
                                      note="Open / closed arms were guessed from the wall edges. If they are the "
                                           "wrong way round press 'Swap'.", epm_swap=True)
                if dlg.exec() == QDialog.Accepted:
                    mp = dlg.mapping()
                    if any(k != v for k, v in mp.items()):
                        self.canvas.push_undo()
                        cols = {mp.get(k, k): self.canvas.color_of(k) for k in self.canvas.zones}
                        self.canvas.zones = {mp.get(k, k): v for k, v in self.canvas.zones.items()}
                        self.canvas.colors.update(cols)
                        self.canvas.selected = [mp.get(n, n) for n in self.canvas.selected]
                        self.canvas.update()
                        self.canvas.zonesEdited.emit()

        def failed(msg):
            self._busy = False
            if self._alive:
                self.btn_auto.setEnabled(True)
                self.btn_auto.setText("⚡ Auto-detect")
                QMessageBox.warning(self, "Auto-detect zones", msg)
        self.ctrl._thread_run(work, done, failed)

    # ---------------------------------------------------------------- save / navigate
    def save_current(self):
        if self.canvas.tool is not None:
            self.canvas.cancel_tool(silent=True)
        it = self.item
        if not self.canvas.zones:
            for k in ("zones", "masks", "crop", "ref_size"):
                it.pop(k, None)
        else:
            it["zones"] = copy.deepcopy(self.canvas.zones)
            it["masks"] = copy.deepcopy(self.canvas.masks)
            it["crop"] = None
            it["ref_size"] = tuple(it["size"])
            self._last = (copy.deepcopy(self.canvas.zones), copy.deepcopy(self.canvas.masks), tuple(it["size"]))
            img = self._bgs.get(it["path"]) if isinstance(getattr(self, "_bgs", None), dict) else None
            if img is None:
                img = getattr(self, "_frame", None)
            if not self.chambers.rects:
                self.ctrl._learn_layout(img, self.canvas.zones, Path(it["path"]).name)
        self.chambers.save(it)
        it["automask"] = bool(self.chk_automask.isChecked())
        self.dirty = False
        self.saved += 1
        self.ctrl._rebuild_batch_table()

    def _confirm_leave(self):
        if not self.dirty:
            return True
        r = QMessageBox.question(self, "Unsaved zones", f"Save the zones of {Path(self.item['path']).name}?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            self.save_current()
            return True
        return r == QMessageBox.Discard

    def _go(self, pos):
        if 0 <= pos < len(self.ctrl.batch_items) and self._confirm_leave():
            self._load(pos)

    def _save_next(self):
        self.save_current()
        if self.pos < len(self.ctrl.batch_items) - 1:
            self._load(self.pos + 1)
        else:
            self.lbl_hint.setText("All videos have zones now — press 'Save & close' and start the batch.")
            self.dirty = False

    def _save_close(self):
        self.save_current()
        self.accept()

    def reject(self):
        if self.canvas.tool is not None:
            self.canvas.cancel_tool()
            return
        if not self._confirm_leave():
            return
        super().reject()

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.canvas.tool is not None:
                self.canvas.finish_tool()
            return                              # never close the dialog with Enter
        if ev.key() == Qt.Key_Escape and self.canvas.tool is not None:
            self.canvas.cancel_tool()
            return
        super().keyPressEvent(ev)

    def done(self, r):
        self._alive = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        super().done(r)


def _small_button(text, tip="", css="secondary", parent=None):
    b = QPushButton(text, parent)
    b.setProperty("cssClass", css)
    b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet("padding:4px 8px;")
    return b


class BehaviourDialog(QDialog):
    """Add / edit one behaviour of the behavioral inventory (name, key, type)."""

    def __init__(self, parent, taken_keys, behaviour=None):
        super().__init__(parent)
        self.setWindowTitle("Behaviour" if behaviour is None else "Edit behaviour")
        self._taken = {k.lower() for k in taken_keys}
        form = QFormLayout(self)
        self.name = QLineEdit(behaviour["name"] if behaviour else "")
        self.name.setPlaceholderText("e.g. Freezing, Rearing, Grooming")
        self.key = QLineEdit(behaviour["key"] if behaviour else "")
        self.key.setMaxLength(1)
        self.key.setFixedWidth(48)
        self.key.setAlignment(Qt.AlignCenter)
        self.key.setPlaceholderText("F")
        self.kind = QComboBox()
        self.kind.addItem("State  (press to start, press again to stop)", "state")
        self.kind.addItem("Point  (one press = one occurrence)", "point")
        if behaviour:
            self.kind.setCurrentIndex(0 if behaviour.get("kind", "state") == "state" else 1)
        form.addRow("Name", self.name)
        form.addRow("Key", self.key)
        form.addRow("Type", self.kind)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _accept(self):
        name = self.name.text().strip()
        key = self.key.text().strip().lower()
        if not name:
            QMessageBox.warning(self, "Behaviour", "Give the behaviour a name.")
            return
        if not key or not (key.isalnum()):
            QMessageBox.warning(self, "Behaviour", "Give it a single letter or digit as its key.")
            return
        if key in self._taken:
            QMessageBox.warning(self, "Behaviour", f"The key '{key}' is already used by another behaviour.")
            return
        self.result_value = {"name": name, "key": key, "kind": self.kind.currentData()}
        self.accept()


class ExclusionMatrixDialog(QDialog):
    """BORIS-style exclusion matrix: tick two STATE behaviours that cannot happen at the same time -- starting
    one then stops the other automatically (e.g. Walking / Freezing / Grooming)."""

    def __init__(self, parent, names, pairs):
        super().__init__(parent)
        self.setWindowTitle("Exclusion matrix")
        self.names = list(names)
        self.resize(max(460, 150 + 110 * len(self.names)), max(320, 190 + 34 * len(self.names)))
        v = QVBoxLayout(self)
        info = QLabel("Tick the pairs of <b>state</b> behaviours that exclude each other: when you start one, a "
                      "running one that is ticked with it stops at the same moment. (Point behaviours are never "
                      "stopped.)")
        info.setWordWrap(True)
        v.addWidget(info)
        n = len(self.names)
        self.tbl = QTableWidget(n, n)
        self.tbl.setHorizontalHeaderLabels(self.names)
        self.tbl.setVerticalHeaderLabels(self.names)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._busy = True
        for i in range(n):
            for j in range(n):
                it = QTableWidgetItem()
                it.setTextAlignment(Qt.AlignCenter)
                if i == j:
                    it.setFlags(Qt.NoItemFlags)
                    it.setText("—")
                    from pro_ui import theme
                    it.setBackground(QBrush(QColor(theme.bg("#e2e8f0"))))
                else:
                    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    on = frozenset((self.names[i], self.names[j])) in pairs
                    it.setCheckState(Qt.Checked if on else Qt.Unchecked)
                self.tbl.setItem(i, j, it)
        self._busy = False
        self.tbl.itemChanged.connect(self._mirror)
        v.addWidget(self.tbl, 1)
        row = QHBoxLayout()
        b_all = QPushButton("All exclusive")
        b_all.setToolTip("Every state behaviour excludes every other one.")
        b_all.clicked.connect(lambda: self._set_all(True))
        b_none = QPushButton("None")
        b_none.clicked.connect(lambda: self._set_all(False))
        row.addWidget(b_all)
        row.addWidget(b_none)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        v.addLayout(row)

    def _mirror(self, it):
        if self._busy:
            return
        self._busy = True
        other = self.tbl.item(it.column(), it.row())
        if other is not None and other is not it:
            other.setCheckState(it.checkState())
        self._busy = False

    def _set_all(self, on):
        self._busy = True
        for i in range(self.tbl.rowCount()):
            for j in range(self.tbl.columnCount()):
                if i != j:
                    self.tbl.item(i, j).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._busy = False

    def pairs(self):
        out = set()
        for i in range(len(self.names)):
            for j in range(i + 1, len(self.names)):
                if self.tbl.item(i, j).checkState() == Qt.Checked:
                    out.add(frozenset((self.names[i], self.names[j])))
        return out


class EventAnalysisDialog(QDialog):
    """Event Scoring ▸ 📊 Plots & analysis -- the BORIS-style toolbox: events plot (ethogram), time budget
    (+ plot), time bins, transitions and inter-observer agreement (Cohen's kappa)."""

    def __init__(self, page):
        super().__init__(page)
        from analysis import event_analysis as EA
        self.EA = EA
        self.page = page
        self.setWindowTitle("Event plots & analysis")
        self.resize(1180, 760)
        self.tmp = tempfile.mkdtemp(prefix="bt_events_")
        self.other = None                      # the second observer's .btms for the agreement
        v = QVBoxLayout(self)
        top = QHBoxLayout()
        dur = (page.n_frames / page.fps) if page.fps and page.n_frames else max(
            [float(e.get("stop") or e["start"]) for e in page.events] + [1.0])
        self.dur = float(dur)
        top.addWidget(QLabel("From"))
        self.sp_from = QDoubleSpinBox()
        self.sp_to = QDoubleSpinBox()
        for sp in (self.sp_from, self.sp_to):
            sp.setRange(0.0, max(self.dur, 0.1))
            sp.setDecimals(1)
            sp.setSuffix(" s")
            sp.setMinimumWidth(90)
        self.sp_to.setValue(self.dur)
        top.addWidget(self.sp_from)
        top.addWidget(QLabel("to"))
        top.addWidget(self.sp_to)
        b_whole = _small_button("Whole video", "Analyse the whole video.")
        b_whole.clicked.connect(lambda: (self.sp_from.setValue(0.0), self.sp_to.setValue(self.dur)))
        top.addWidget(b_whole)
        top.addSpacing(16)
        top.addWidget(QLabel("Time bin"))
        self.sp_bin = QDoubleSpinBox()
        self.sp_bin.setRange(1.0, 3600.0)
        self.sp_bin.setValue(60.0)
        self.sp_bin.setSuffix(" s")
        top.addWidget(self.sp_bin)
        b_up = QPushButton("↻ Update")
        b_up.setProperty("cssClass", "primary")
        b_up.clicked.connect(self.refresh)
        top.addWidget(b_up)
        top.addStretch(1)
        self.lbl_note = QLabel("")
        self.lbl_note.setProperty("cssClass", "muted")
        top.addWidget(self.lbl_note)
        v.addLayout(top)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)
        # events plot
        self.lbl_plot = QLabel()
        self.lbl_plot.setStyleSheet("background:#ffffff;")
        self.view_plot = FrameView(self.lbl_plot, self)
        self.tabs.addTab(self.lbl_plot, "Events plot")
        # time budget: table + plot
        w = QWidget()
        lay = QVBoxLayout(w)
        self.tbl_budget = self._table()
        lay.addWidget(self.tbl_budget, 2)
        self.lbl_budget = QLabel()
        self.lbl_budget.setStyleSheet("background:#ffffff;")
        self.view_budget = FrameView(self.lbl_budget, self)
        lay.addWidget(self.lbl_budget, 3)
        self.tabs.addTab(w, "Time budget")
        self.tbl_bins = self._table()
        self.tabs.addTab(self.tbl_bins, "Time bins")
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("How often each behaviour is FOLLOWED by each other behaviour (by onset):"))
        self.tbl_tr = self._table()
        lay.addWidget(self.tbl_tr, 1)
        lay.addWidget(QLabel("Probabilities (each row adds up to 1):"))
        self.tbl_trp = self._table()
        lay.addWidget(self.tbl_trp, 1)
        self.tabs.addTab(w, "Transitions")
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        b_other = QPushButton("📂 Second observer's BT scoring file…")
        b_other.clicked.connect(self._pick_other)
        row.addWidget(b_other)
        row.addWidget(QLabel("Time slice"))
        self.sp_step = QDoubleSpinBox()
        self.sp_step.setRange(0.1, 60.0)
        self.sp_step.setValue(1.0)
        self.sp_step.setSuffix(" s")
        self.sp_step.setToolTip("Both codings are compared every this many seconds (BORIS uses the same method).")
        row.addWidget(self.sp_step)
        row.addStretch(1)
        lay.addLayout(row)
        self.lbl_kappa = QLabel("Two people score the same video; open the other person's .btms file here to see "
                                "how well they agree (Cohen's kappa).")
        self.lbl_kappa.setWordWrap(True)
        lay.addWidget(self.lbl_kappa)
        self.tbl_kappa = self._table()
        lay.addWidget(self.tbl_kappa, 1)
        self.tabs.addTab(w, "Agreement (kappa)")
        bb = QHBoxLayout()
        b_save = QPushButton("💾 Save plots + tables…")
        b_save.setToolTip("Excel workbook with every table + the plots as PNG (300 dpi) and SVG.")
        b_save.clicked.connect(self.save_all)
        bb.addWidget(b_save)
        bb.addStretch(1)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        bb.addWidget(b_close)
        v.addLayout(bb)
        self.refresh()

    @staticmethod
    def _table():
        t = QTableWidget()
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setAlternatingRowColors(True)
        return t

    def _colors(self):
        return {b["name"]: self.page._behaviour_color(b["name"]).name() for b in self.page.inventory}

    def _window(self):
        a, b = float(self.sp_from.value()), float(self.sp_to.value())
        return (a, b) if b > a else (0.0, self.dur)

    def _title(self):
        return Path(self.page.path).stem if self.page.path else "Observation"

    def results(self):
        EA = self.EA
        t0, t1 = self._window()
        ev, inv = self.page.events, self.page.inventory
        budget = EA.time_budget(ev, inv, t0, t1, close_open_at=self.dur)
        bins = EA.time_budget_bins(ev, inv, t0, t1, self.sp_bin.value(), close_open_at=self.dur)
        tr, trp = EA.transitions(ev, inv, close_open_at=self.dur)
        return budget, bins, tr, trp

    def refresh(self):
        EA = self.EA
        t0, t1 = self._window()
        ev, inv = self.page.events, self.page.inventory
        open_ = EA.unpaired(ev)
        self.lbl_note.setText(f"⚠ {len(open_)} state bout(s) never stopped — counted until the end of the video"
                              if open_ else "")
        cols = self._colors()
        p1 = EA.plot_events(ev, inv, t0, t1, os.path.join(self.tmp, "events_plot.png"), self._title(), cols,
                            close_open_at=self.dur)
        self.view_plot.set_file(p1)
        budget, bins, tr, trp = self.results()
        self.page.ctrl._fill_table(self.tbl_budget, budget)
        p2 = EA.plot_time_budget(budget, os.path.join(self.tmp, "time_budget.png"), self._title(), cols)
        self.view_budget.set_file(p2)
        self.page.ctrl._fill_table(self.tbl_bins, bins)
        self.page.ctrl._fill_table(self.tbl_tr, tr.reset_index())
        self.page.ctrl._fill_table(self.tbl_trp, trp.reset_index())
        if self.other is not None:
            self._kappa()

    def _pick_other(self):
        from tracking.project_file import EXT_SCORING, load_scoring
        start = str(Path(self.page.session_path).parent) if self.page.session_path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Second observer's BT scoring file", start,
                                              f"BT Manual Scoring file (*{EXT_SCORING})")
        if not path:
            return
        try:
            self.other = load_scoring(path)
        except Exception as exc:
            QMessageBox.warning(self, "Agreement", f"Could not open that file:\n{exc}")
            return
        self.other["_name"] = Path(path).name
        self._kappa()

    def _kappa(self):
        EA = self.EA
        t0, t1 = self._window()
        k = EA.cohen_kappa(self.page.events, self.other.get("events") or [], self.page.inventory, t0, t1,
                           self.sp_step.value(), close_open_at=self.dur)
        self.kappa = k
        same = (Path(str((self.other.get("video") or {}).get("name") or "")).name == Path(str(self.page.path or "")).name)
        warn = "" if same else "  ⚠ The two files are from different videos."
        self.lbl_kappa.setText(f"<b>Cohen's kappa = {k['kappa']}</b> ({EA.kappa_words(k['kappa'])} agreement) · "
                               f"{k['agreement_pct']} % of {k['slices']} slices of {k['step_s']:g} s agree · "
                               f"this session vs <b>{self.other['_name']}</b>.{warn}<br>"
                               "Landis &amp; Koch: &lt;0 poor, 0–0.2 slight, 0.2–0.4 fair, 0.4–0.6 moderate, "
                               "0.6–0.8 substantial, 0.8–1 almost perfect.")
        self.page.ctrl._fill_table(self.tbl_kappa, k["per_behaviour"])

    def save_all(self):
        EA = self.EA
        base = Path(self.page.path).with_suffix("") if self.page.path else Path.home() / "events"
        path, _ = QFileDialog.getSaveFileName(self, "Save plots + tables", f"{base}_event_analysis.xlsx",
                                              "Excel workbook (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        t0, t1 = self._window()
        budget, bins, tr, trp = self.results()
        stem = str(Path(path).with_suffix(""))
        cols = self._colors()
        try:
            with pd.ExcelWriter(path) as xl:
                pd.DataFrame([{"Video": self.page.path or "", "From_s": t0, "To_s": t1,
                               "Time_bin_s": self.sp_bin.value(), "Events": len(self.page.events)}]).to_excel(
                    xl, sheet_name="Settings", index=False)
                budget.to_excel(xl, sheet_name="Time budget", index=False)
                bins.to_excel(xl, sheet_name="Time bins", index=False)
                tr.to_excel(xl, sheet_name="Transitions (counts)")
                trp.to_excel(xl, sheet_name="Transitions (prob)")
                if getattr(self, "kappa", None):
                    k = self.kappa
                    pd.DataFrame([{"Cohen_kappa": k["kappa"], "Agreement_pct": k["agreement_pct"],
                                   "Slices": k["slices"], "Slice_s": k["step_s"],
                                   "Other_file": self.other.get("_name")}]).to_excel(xl, sheet_name="Agreement",
                                                                                   index=False)
                    k["per_behaviour"].to_excel(xl, sheet_name="Agreement", index=False, startrow=3)
            for ext, dpi in (("png", 300), ("svg", 100)):
                EA.plot_events(self.page.events, self.page.inventory, t0, t1, f"{stem}_events_plot.{ext}",
                               self._title(), cols, close_open_at=self.dur, dpi=dpi)
                EA.plot_time_budget(budget, f"{stem}_time_budget.{ext}", self._title(), cols, dpi=dpi)
        except Exception as exc:
            QMessageBox.warning(self, "Save", f"Could not save:\n{exc}")
            return
        self.page.ctrl.status.showMessage(f"Saved {Path(path).name} + events plot + time budget plot "
                                          "(PNG and SVG).", 8000)

    def done(self, r):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().done(r)


class EventScoringPage(QWidget):
    """Manual, keyboard-driven behaviour scoring.

    Define a behavioral inventory (behaviours, each with a key and a type), play the
    video, and press a behaviour's key to log it at the current video time:
    a STATE behaviour toggles on/off (start -> stop), a POINT behaviour logs
    one occurrence per press.  The events can be edited and exported, and a
    summary (bouts, total / mean duration, latency to first) is computed per
    behaviour.  Playback and scoring here are independent of the tracking
    pipeline.
    """
    PALETTE = ["#1f6feb", "#ef4444", "#22c55e", "#a855f7", "#f59e0b", "#06b6d4", "#ec4899", "#84cc16",
               "#14b8a6", "#f97316", "#8b5cf6", "#0ea5e9"]

    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.cap = None
        self.copy = None            # video copy from a .btms file (when the original video is not here)
        self.opened_data = None     # the .btms this session was opened from
        self.path = None
        self.fps = 30.0
        self.n_frames = 0
        self.frame_idx = 0
        self.speed = 1.0
        self.playing = False
        self.events = []            # {behaviour, kind, start, stop}
        self._open = {}             # behaviour name -> start time (ongoing state bouts)
        self.session_path = None    # the BT Manual Scoring file (.btms) of this session
        self.audit = []             # its audit trail
        self._changes = {"added": 0, "deleted": 0}
        self.inventory = []         # {name, key, kind}
        self.exclusive = set()      # exclusion matrix: frozenset({state A, state B}) -> starting one stops the other
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self._build()
        self._load_inventory(INVENTORY_PATH, silent=True)
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ UI
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(16)

        left = QVBoxLayout()
        head = QVBoxLayout()
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        badge = QLabel()
        badge.setFixedSize(40, 40)
        badge.setPixmap(QIcon(str(HERE / "icons" / "events_score.svg")).pixmap(40, 40))
        badge.setToolTip("Event Scoring")
        title = QLabel("Event Scoring")
        title.setProperty("cssClass", "pageTitle")
        title_row.addWidget(badge, 0, Qt.AlignVCenter)
        title_row.addWidget(title, 1, Qt.AlignVCenter)
        sub = QLabel("Score behaviours live with the keyboard — build your behavioral inventory, play the video and "
                     "press each behaviour's key.")
        sub.setProperty("cssClass", "pageSub")
        sub.setWordWrap(True)
        head.addLayout(title_row)
        head.addWidget(sub)
        left.addLayout(head)

        self.video_label = QLabel("Choose a video to score")
        self.video_label.setProperty("cssClass", "videoSurface")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(360)
        self.view = FrameView(self.video_label, self)
        self.video_label2 = QLabel("2nd camera")
        self.video_label2.setProperty("cssClass", "videoSurface")
        self.video_label2.setAlignment(Qt.AlignCenter)
        self.view2 = FrameView(self.video_label2, self)
        self.video_label2.hide()
        vrow2 = QHBoxLayout()
        vrow2.setSpacing(6)
        vrow2.addWidget(self.video_label, 1)
        vrow2.addWidget(self.video_label2, 1)
        left.addLayout(vrow2, 1)

        self.lbl_ongoing = QLabel("")
        self.lbl_ongoing.setStyleSheet("color:#1f6feb; font-weight:600;")
        left.addWidget(self.lbl_ongoing)

        ctl = QHBoxLayout()
        self.btn_play = QToolButton()
        self.btn_play.setText("▶")
        self.btn_play.setToolTip("Play / pause (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self._seek_to)
        self.slider.sliderPressed.connect(lambda: self.pause())
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color:#000000; font-weight:600;")
        b_back = QToolButton()
        b_back.setText("◀|")
        b_back.setToolTip("Step back one frame (←)")
        b_back.clicked.connect(lambda: self.step(-1))
        b_fwd = QToolButton()
        b_fwd.setText("|▶")
        b_fwd.setToolTip("Step forward one frame (→)")
        b_fwd.clicked.connect(lambda: self.step(1))
        self.cmb_speed = QComboBox()
        for s in (0.25, 0.5, 1.0, 1.5, 2.0, 4.0):
            self.cmb_speed.addItem(f"{s:g}×", s)
        self.cmb_speed.setCurrentIndex(2)
        self.cmb_speed.currentIndexChanged.connect(self._speed_changed)
        self.cmb_speed.setToolTip("Playback speed")
        ctl.addWidget(self.btn_play)
        ctl.addWidget(b_back)
        ctl.addWidget(b_fwd)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.lbl_time)
        ctl.addWidget(self.cmb_speed)
        left.addLayout(ctl)

        src = QHBoxLayout()
        src.addWidget(QLabel("Video:"))
        self.cmb_video = QComboBox()
        self.cmb_video.setMinimumWidth(240)
        self.cmb_video.activated.connect(self._pick_video)
        b_open = _small_button("Open video…", "Open a video file to score.")
        b_open.clicked.connect(self._open_video_dialog)
        self.b_cam2 = _small_button("＋ 2nd camera…", "Show a second video of the same trial (another camera "
                                    "angle) next to this one, played in step. Set its time offset if the two "
                                    "recordings did not start together.")
        self.b_cam2.clicked.connect(self._open_cam2)
        self.sp_off2 = QDoubleSpinBox()
        self.sp_off2.setRange(-3600, 3600)
        self.sp_off2.setDecimals(2)
        self.sp_off2.setSuffix(" s")
        self.sp_off2.setToolTip("2nd camera offset: its time = this video's time + offset.")
        self.sp_off2.valueChanged.connect(lambda _v: self._show_cam2())
        self.sp_off2.hide()
        src.addWidget(self.cmb_video, 1)
        src.addWidget(b_open)
        src.addWidget(self.b_cam2)
        src.addWidget(self.sp_off2)
        left.addLayout(src)
        root.addLayout(left, 3)

        # ---- right: behavioral inventory + events + summary
        panel = QFrame()
        panel.setProperty("cssClass", "card")
        panel.setMinimumWidth(430)
        panel.setMaximumWidth(560)
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(12, 12, 12, 12)
        pv.setSpacing(6)

        cap = QLabel("Behavioral Inventory  ·  behaviours and their keys")
        cap.setProperty("cssClass", "cardTitle")
        pv.addWidget(cap)
        syn = QLabel("Also called a behaviour catalogue, behavioural repertoire or coding scheme — "
                     "the list of behaviours you score, each with its own key.")
        syn.setWordWrap(True)
        syn.setStyleSheet("color:#64748b; font-size:11px;")
        pv.addWidget(syn)
        self.tbl_inventory = QTableWidget(0, 3)
        self.tbl_inventory.setHorizontalHeaderLabels(["Key", "Behaviour", "Type"])
        self.tbl_inventory.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_inventory.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_inventory.verticalHeader().setVisible(False)
        self.tbl_inventory.setMaximumHeight(170)
        eh = self.tbl_inventory.horizontalHeader()
        eh.setSectionResizeMode(0, QHeaderView.Fixed)
        eh.setSectionResizeMode(1, QHeaderView.Stretch)
        eh.setSectionResizeMode(2, QHeaderView.Fixed)
        self.tbl_inventory.setColumnWidth(0, 44)
        self.tbl_inventory.setColumnWidth(2, 64)
        self.tbl_inventory.cellDoubleClicked.connect(lambda r, _c: self._edit_behaviour(r))
        pv.addWidget(self.tbl_inventory)
        erow = QHBoxLayout()
        for text, fn, tip in (("＋ Add", self._add_behaviour, "Add a behaviour"),
                              ("Edit", lambda: self._edit_behaviour(self._inventory_row()), "Edit the selected behaviour"),
                              ("Remove", self._remove_behaviour, "Remove the selected behaviour"),
                              ("Save…", self._save_inventory, "Save this behavioral inventory to a file"),
                              ("Load…", lambda: self._load_inventory(None), "Load a behavioral inventory from a file")):
            b = _small_button(text, tip)
            b.clicked.connect(fn)
            erow.addWidget(b)
        erow.addStretch(1)
        pv.addLayout(erow)
        xrow = QHBoxLayout()
        b_imp = _small_button("⤒ Import events…", "Events from BORIS (tabular export), SimBA (machine results: "
                              "classifier bouts) or any table with behaviour + start + stop columns.")
        b_imp.clicked.connect(self._import_events)
        b_val = _small_button("≟ Validate BT freezing…", "Compare BT's freezing with your hand scoring of the same "
                              "videos: frame agreement, Cohen's kappa, Pearson r, ICC and Bland-Altman limits.")
        b_val.clicked.connect(self._validate_freezing)
        b_clf = _small_button("🧠 Classifier…", "Teach BT a behaviour (grooming, rearing ...) from videos you scored "
                              "here, test how well it agrees with you, then let it find the behaviour in new videos.")
        b_clf.clicked.connect(self._classifier)
        irow = QHBoxLayout()
        irow.addWidget(b_imp)
        irow.addWidget(b_val)
        irow.addWidget(b_clf)
        irow.addStretch(1)
        pv.addLayout(irow)
        b_x = _small_button("⊘ Exclusion matrix…", "Behaviours that cannot happen together (like BORIS): starting "
                            "one stops the other automatically.")
        b_x.clicked.connect(self._edit_exclusion)
        self.lbl_excl = QLabel("")
        self.lbl_excl.setWordWrap(True)
        self.lbl_excl.setStyleSheet("color:#64748b; font-size:11px;")
        xrow.addWidget(b_x)
        xrow.addWidget(self.lbl_excl, 1)
        pv.addLayout(xrow)

        cap2 = QLabel("Events")
        cap2.setProperty("cssClass", "cardTitle")
        pv.addWidget(cap2)
        self.tbl_events = QTableWidget(0, 5)
        self.tbl_events.setHorizontalHeaderLabels(["Behaviour", "Type", "Start", "Stop", "Duration"])
        self.tbl_events.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_events.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_events.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_events.verticalHeader().setVisible(False)
        self.tbl_events.setAlternatingRowColors(True)
        vh = self.tbl_events.horizontalHeader()
        vh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3, 4):
            vh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.tbl_events.cellDoubleClicked.connect(self._goto_event)
        pv.addWidget(self.tbl_events, 1)
        vrow = QHBoxLayout()
        b_del = _small_button("Delete event", "Delete the selected event(s) (Del).")
        b_del.clicked.connect(self._delete_events)
        b_clear = _small_button("Clear all", "Delete every scored event.")
        b_clear.clicked.connect(self._clear_events)
        b_exp = QPushButton("⤓ Export…")
        b_exp.setProperty("cssClass", "primary")
        b_exp.setToolTip("Save the events and the summary (Excel or CSV).")
        b_exp.clicked.connect(self._export)
        b_plot = QPushButton("📊 Plots && analysis…")
        b_plot.setProperty("cssClass", "secondary")
        b_plot.setToolTip("Events plot (ethogram), time budget, time bins, transitions and the agreement between "
                          "two observers (Cohen's kappa) — like BORIS.")
        b_plot.clicked.connect(self.open_analysis)
        vrow.addWidget(b_del)
        vrow.addWidget(b_clear)
        vrow.addStretch(1)
        vrow.addWidget(b_plot)
        vrow.addWidget(b_exp)
        pv.addLayout(vrow)
        srow = QHBoxLayout()
        b_ss = _small_button("💾 Save BT scoring file…", "Save this scoring session (video, behaviours, events and "
                             "audit trail) as a BT Manual Scoring file (.btms) — it opens only in BehavioralTracker.")
        b_ss.clicked.connect(self.save_session)
        b_os = _small_button("📂 Open BT scoring file…", "Open a saved BT Manual Scoring file (.btms).")
        b_os.clicked.connect(lambda: self.open_session(None))
        b_au = _small_button("🧾 Audit trail", "Who scored and changed what, and when.")
        b_au.clicked.connect(self.show_audit)
        srow.addWidget(b_ss)
        srow.addWidget(b_os)
        srow.addWidget(b_au)
        srow.addStretch(1)
        pv.addLayout(srow)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        self.lbl_summary.setStyleSheet("color:#475569; font-size:9pt;")
        pv.addWidget(self.lbl_summary)
        root.addWidget(panel, 2)

        self._refresh_inventory()
        self._refresh_events()

    # ------------------------------------------------------------------ video
    def refresh_sources(self):
        self.cmb_video.blockSignals(True)
        self.cmb_video.clear()
        for it in self.ctrl.exp_videos:
            self.cmb_video.addItem(Path(it["path"]).name, it["path"])
        if self.path and self.path not in [it["path"] for it in self.ctrl.exp_videos]:
            self.cmb_video.addItem(Path(self.path).name, self.path)
        if self.path:
            i = self.cmb_video.findData(self.path)
            if i >= 0:
                self.cmb_video.setCurrentIndex(i)
        self.cmb_video.blockSignals(False)

    def _pick_video(self, idx):
        path = self.cmb_video.itemData(idx)
        if path and path != self.path:
            self.open_video(path)

    def _open_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open video to score", str(Path.home()), VIDEO_FILTER)
        if path:
            self.open_video(path)

    def open_video(self, path):
        self.pause()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Video", f"Could not open:\n{path}")
            return
        if self.cap is not None:
            self.cap.release()
        self._close_copy()
        self.opened_data = None
        self.cap = cap
        self.path = path
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(0)
        self.refresh_sources()
        if self.events and QMessageBox.question(
                self, "New video", "Clear the events scored on the previous video?") == QMessageBox.Yes:
            self._clear_events(ask=False)
            self.session_path, self.audit, self._changes = None, [], {"added": 0, "deleted": 0}
        self._show_frame(0)

    def _close_copy(self):
        if getattr(self, "copy", None) is not None:
            self.copy.close()
        self.copy = None

    def _open_copy(self, data, segs):
        """Score on the video copy stored in a .btms file (the original video is not here)."""
        from tracking.review_video import CopyReader
        info = data.get("video_copy") or {}
        self.pause()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._close_copy()
        self.copy = CopyReader(segs, info.get("size") or (640, 480))
        vid = data.get("video") or {}
        self.path = vid.get("path") or ""
        self.fps = float(vid.get("fps") or info.get("fps") or 30.0)
        self.n_frames = int(vid.get("frames") or info.get("frames") or 0)
        self.frame_idx = 0
        self.slider.setRange(0, max(0, self.n_frames - 1))
        self.slider.setValue(0)
        self._show_frame(0)

    def _show_frame(self, idx):
        if self.cap is None:
            if getattr(self, "copy", None) is None:
                return
            idx = int(min(max(0, idx), max(0, self.n_frames - 1)))
            frame = self.copy.read(idx)
            if frame is not None:
                self.frame_idx = idx
                self.view.set_bgr(frame)
            self.slider.blockSignals(True)
            self.slider.setValue(self.frame_idx)
            self.slider.blockSignals(False)
            self.lbl_time.setText(f"{fmt_clock(self.frame_idx / self.fps)} / {fmt_clock(self.n_frames / self.fps)}"
                                  "   (video copy from the BT file)")
            return
        idx = int(min(max(0, idx), max(0, self.n_frames - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if ok:
            self.frame_idx = idx
            self.view.set_bgr(frame)
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)
        self.lbl_time.setText(f"{fmt_clock(self.frame_idx / self.fps)} / {fmt_clock(self.n_frames / self.fps)}")
        self._show_cam2()

    # ------------------------------------------------------------------ 2nd camera (synced)
    def _open_cam2(self):
        if getattr(self, "cap2", None) is not None:
            self.cap2.release()
            self.cap2 = None
            self.video_label2.hide()
            self.sp_off2.hide()
            self.b_cam2.setText("＋ 2nd camera…")
            return
        fn, _ = QFileDialog.getOpenFileName(self, "Second camera of the same trial",
                                            str(Path(self.path).parent if self.path else Path.home()), VIDEO_FILTER)
        if not fn:
            return
        cap = cv2.VideoCapture(fn)
        if not cap.isOpened():
            QMessageBox.warning(self, "2nd camera", f"Could not open:\n{fn}")
            return
        self.cap2, self.cam2_path = cap, fn
        self.fps2 = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n2 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._pos2 = -10
        self.video_label2.show()
        self.sp_off2.show()
        self.b_cam2.setText("✕ 2nd camera")
        self._show_cam2()

    def _show_cam2(self):
        cap = getattr(self, "cap2", None)
        if cap is None:
            return
        t = self.frame_idx / (self.fps or 30.0) + self.sp_off2.value()
        j = int(round(t * self.fps2))
        if not 0 <= j < max(1, self.n2):
            self.view2.clear(f"2nd camera: no picture at {fmt_clock(max(0.0, t))}")
            return
        if j != self._pos2 + 1:                     # playing reads on; a jump seeks
            cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok, fr = cap.read()
        if ok:
            self._pos2 = j
            self.view2.set_bgr(fr)

    # ------------------------------------------------------------------ events from other programs
    def _import_events(self):
        from analysis.event_import import read_events
        fn, _ = QFileDialog.getOpenFileName(self, "Import events (BORIS / SimBA / table)",
                                            str(Path(self.path).parent if self.path else Path.home()),
                                            "Tables (*.csv *.tsv *.txt *.xlsx *.xls);;All files (*)")
        if not fn:
            return
        try:
            events, src = read_events(fn, fps=self.fps or 30.0)
        except Exception as exc:
            QMessageBox.warning(self, "Import events", f"No events could be read:\n{exc}")
            return
        if not events:
            QMessageBox.information(self, "Import events", "The file has no events.")
            return
        known = {b["name"] for b in self.inventory}
        new = []
        for e in events:
            if e["behaviour"] not in known:
                known.add(e["behaviour"])
                kind = e["kind"]
                new.append({"name": e["behaviour"], "key": "", "kind": kind})
        self.inventory += new
        self.events += events
        self.events.sort(key=lambda e: e["start"])
        try:
            from tracking.project_file import audit_add
            audit_add(self.audit, "Events imported", {"file": Path(fn).name, "source": src, "events": len(events)})
        except Exception:
            pass
        self._changes["added"] += len(events)
        self._refresh_inventory()
        self._refresh_events()
        self.ctrl.status.showMessage(f"{len(events)} events imported from {Path(fn).name} ({src}); "
                                     f"{len(new)} new behaviour(s) added (give them a key to score more).", 8000)

    def _classifier(self):
        from pro_ui.classifier_dialog import ClassifierDialog
        cur = {"video": self.path, "events": list(self.events), "fps": self.fps} if self.path else None
        ClassifierDialog(self, cur, on_events=self._add_bt_events).exec()

    def _add_bt_events(self, events):
        """Bouts found by the classifier: added as '<behaviour> (BT)' (compare, correct, export)."""
        known = {b["name"] for b in self.inventory}
        self.events = [e for e in self.events if not str(e["behaviour"]).endswith(" (BT)")]   # replace earlier ones
        for e in events:
            if e["behaviour"] not in known:
                known.add(e["behaviour"])
                self.inventory.append({"name": e["behaviour"], "key": "", "kind": "state"})
        self.events += events
        self.events.sort(key=lambda e: e["start"])
        try:
            from tracking.project_file import audit_add
            audit_add(self.audit, "Behaviours found by the BT classifier", {"events": len(events)})
        except Exception:
            pass
        self._changes["added"] += len(events)
        self._refresh_inventory()
        self._refresh_events()

    def _validate_freezing(self):
        from pro_ui.validation_dialog import ValidationDialog
        cur = {"video": self.path, "events": list(self.events), "inventory": list(self.inventory), "fps": self.fps}
        ValidationDialog(self, cur).exec()

    @property
    def t(self):
        return self.frame_idx / self.fps if self.fps else 0.0

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if self.cap is None and getattr(self, "copy", None) is None:
            return
        self.playing = True
        self.btn_play.setText("⏸")
        self.timer.start(max(5, int(1000.0 / (self.fps * self.speed))))

    def pause(self):
        self.playing = False
        self.btn_play.setText("▶")
        self.timer.stop()

    def _advance(self):
        if self.frame_idx >= self.n_frames - 1:
            self.pause()
            return
        self._show_frame(self.frame_idx + 1)

    def step(self, d):
        self.pause()
        self._show_frame(self.frame_idx + d)

    def _seek_to(self, v):
        self._show_frame(v)

    def _speed_changed(self, _i):
        self.speed = float(self.cmb_speed.currentData())
        if self.playing:
            self.play()

    # ------------------------------------------------------------------ behavioral inventory
    def _inventory_row(self):
        rows = self.tbl_inventory.selectionModel().selectedRows() if self.tbl_inventory.selectionModel() else []
        return rows[0].row() if rows else -1

    def _add_behaviour(self):
        dlg = BehaviourDialog(self, [b["key"] for b in self.inventory])
        if dlg.exec() == QDialog.Accepted:
            self.inventory.append(dlg.result_value)
            self._refresh_inventory()
            self._persist_inventory()

    def _edit_behaviour(self, row):
        if not (0 <= row < len(self.inventory)):
            return
        b = self.inventory[row]
        dlg = BehaviourDialog(self, [x["key"] for i, x in enumerate(self.inventory) if i != row], b)
        if dlg.exec() == QDialog.Accepted:
            old = b["name"]
            self.inventory[row] = dlg.result_value
            new = dlg.result_value["name"]
            if old != new:
                for ev in self.events:
                    if ev["behaviour"] == old:
                        ev["behaviour"] = new
                if old in self._open:
                    self._open[new] = self._open.pop(old)
                self.exclusive = {frozenset(new if x == old else x for x in pr) for pr in self.exclusive}
            if dlg.result_value.get("kind") == "point":
                self.exclusive = {pr for pr in self.exclusive if new not in pr}
            self._refresh_inventory()
            self._refresh_events()
            self._persist_inventory()

    def _remove_behaviour(self):
        row = self._inventory_row()
        if not (0 <= row < len(self.inventory)):
            return
        gone = self.inventory[row]["name"]
        del self.inventory[row]
        self.exclusive = {pr for pr in self.exclusive if gone not in pr}
        self._refresh_inventory()
        self._persist_inventory()

    def _behaviour_color(self, name):
        names = [b["name"] for b in self.inventory]
        i = names.index(name) if name in names else len(names)
        return QColor(self.PALETTE[i % len(self.PALETTE)])

    def _refresh_inventory(self):
        t = self.tbl_inventory
        t.setRowCount(len(self.inventory))
        for r, b in enumerate(self.inventory):
            k = QTableWidgetItem(b["key"].upper())
            k.setTextAlignment(Qt.AlignCenter)
            f = k.font()
            f.setBold(True)
            k.setFont(f)
            k.setForeground(QBrush(self._behaviour_color(b["name"])))
            t.setItem(r, 0, k)
            t.setItem(r, 1, QTableWidgetItem(b["name"]))
            t.setItem(r, 2, QTableWidgetItem("State" if b.get("kind", "state") == "state" else "Point"))
        if not self.inventory:
            self.lbl_summary.setText("Add behaviours (＋ Add), then play the video and press each behaviour's key "
                                     "to score it. State = hold (start→stop); Point = one press.")
        self._show_exclusion()

    def _state_names(self):
        return [b["name"] for b in self.inventory if b.get("kind", "state") == "state"]

    def _show_exclusion(self):
        if not hasattr(self, "lbl_excl"):
            return
        names = set(self._state_names())
        self.exclusive = {pr for pr in self.exclusive if len(pr) == 2 and pr <= names}
        if not self.exclusive:
            self.lbl_excl.setText("No exclusive behaviours — any states can run at the same time.")
        else:
            txt = ", ".join(" ⟷ ".join(sorted(pr)) for pr in sorted(self.exclusive, key=lambda p: sorted(p)))
            self.lbl_excl.setText(f"Exclusive: {txt}")

    def _edit_exclusion(self):
        names = self._state_names()
        if len(names) < 2:
            QMessageBox.information(self, "Exclusion matrix", "Add at least two STATE behaviours first.")
            return
        dlg = ExclusionMatrixDialog(self, names, self.exclusive)
        if dlg.exec() == QDialog.Accepted:
            self.exclusive = dlg.pairs()
            self._show_exclusion()
            self._persist_inventory()

    def _exclusion_list(self):
        return [sorted(pr) for pr in sorted(self.exclusive, key=lambda p: sorted(p))]

    def _set_exclusion(self, pairs):
        self.exclusive = {frozenset(p) for p in (pairs or []) if isinstance(p, (list, tuple)) and len(set(p)) == 2}
        self._show_exclusion()

    def _persist_inventory(self):
        try:
            APP_DIR.mkdir(parents=True, exist_ok=True)
            save_json(INVENTORY_PATH, {"behaviours": self.inventory, "exclusion": self._exclusion_list()})
        except Exception:
            pass

    def _save_inventory(self):
        if not self.inventory:
            QMessageBox.information(self, "Behavioral Inventory", "Add some behaviours first.")
            return
        INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(self, "Save behavioral inventory",
                                              str(INVENTORY_DIR / "behavioral_inventory.json"),
                                              "Behavioral inventory (*.json)")
        if path:
            save_json(path, {"behaviours": self.inventory, "exclusion": self._exclusion_list()})
            self.ctrl.status.showMessage(f"Behavioral inventory saved to {path}", 5000)

    def _load_inventory(self, path, silent=False):
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Load behavioral inventory", str(INVENTORY_DIR),
                                                  "Behavioral inventory (*.json)")
            if not path:
                return
        data = load_json(path, None)
        if not data or "behaviours" not in data:
            if not silent:
                QMessageBox.warning(self, "Behavioral Inventory",
                                    "That file is not a behavioral inventory (behaviour list).")
            return
        clean = []
        seen = set()
        for b in data["behaviours"]:
            if not isinstance(b, dict) or "name" not in b or "key" not in b:
                continue
            k = str(b["key"]).lower()[:1]
            if not k or k in seen:
                continue
            seen.add(k)
            clean.append({"name": str(b["name"]), "key": k, "kind": b.get("kind", "state")})
        self.inventory = clean
        self._set_exclusion(data.get("exclusion"))
        self._refresh_inventory()
        if not silent:
            self.ctrl.status.showMessage(f"Loaded {len(clean)} behaviour(s).", 5000)

    # ------------------------------------------------------------------ scoring
    def _score_key(self, key):
        b = next((x for x in self.inventory if x["key"] == key.lower()), None)
        if b is None:
            return False
        name, kind = b["name"], b.get("kind", "state")
        if kind == "point":
            self.events.append({"behaviour": name, "kind": "point", "start": self.t, "stop": None})
            self._changes["added"] += 1
        else:
            if name in self._open:
                start = self._open.pop(name)
                self.events.append({"behaviour": name, "kind": "state", "start": start, "stop": self.t})
                self._changes["added"] += 1
            else:
                stopped = []
                for other in list(self._open):           # exclusion matrix: the excluded running states stop now
                    if frozenset((name, other)) in self.exclusive:
                        start = self._open.pop(other)
                        self.events.append({"behaviour": other, "kind": "state", "start": start, "stop": self.t})
                        self._changes["added"] += 1
                        stopped.append(other)
                self._open[name] = self.t
                if stopped:
                    self.ctrl.status.showMessage(f"{name} started — {', '.join(stopped)} stopped (exclusive).", 4000)
        self._refresh_events()
        return True

    def _refresh_events(self):
        rows = sorted(self.events, key=lambda e: e["start"])
        self._rows = rows
        t = self.tbl_events
        t.setRowCount(len(rows))
        for r, e in enumerate(rows):
            dur = "" if e["stop"] is None else f"{e['stop'] - e['start']:.2f}"
            stop = "" if e["stop"] is None else f"{e['stop']:.2f}"
            typ = "State" if e["kind"] == "state" else "Point"
            vals = [e["behaviour"], typ, f"{e['start']:.2f}", stop, dur]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 0:
                    it.setForeground(QBrush(self._behaviour_color(e["behaviour"])))
                t.setItem(r, c, it)
        # ongoing state bouts
        if self._open:
            self.lbl_ongoing.setText("● Ongoing: " + ", ".join(f"{n} ({self.t - s:.1f}s)"
                                                               for n, s in self._open.items()))
        else:
            self.lbl_ongoing.setText("")
        self._update_summary()

    def _goto_event(self, row, _c):
        if 0 <= row < len(getattr(self, "_rows", [])):
            self._show_frame(int(round(self._rows[row]["start"] * self.fps)))

    def _delete_events(self):
        rows = sorted({i.row() for i in self.tbl_events.selectionModel().selectedRows()}, reverse=True) \
            if self.tbl_events.selectionModel() else []
        if not rows:
            return
        for r in rows:
            if 0 <= r < len(self._rows):
                e = self._rows[r]
                if e in self.events:
                    self.events.remove(e)
                    self._changes["deleted"] += 1
        self._refresh_events()

    def _clear_events(self, ask=True):
        if ask and self.events and QMessageBox.question(
                self, "Clear events", "Delete every scored event?") != QMessageBox.Yes:
            return
        self._changes["deleted"] += len(self.events)
        self.events = []
        self._open = {}
        self._refresh_events()

    # ------------------------------------------------------------------ BT Manual Scoring file (.btms)
    def save_session(self):
        from tracking.project_file import EXT_SCORING, audit_add, save_scoring
        if not self.events and not self.inventory:
            QMessageBox.information(self, "BT scoring file", "Add behaviours and score some events first.")
            return
        if self.session_path:
            path = self.session_path
        else:
            base = Path(self.path).with_suffix("") if self.path else Path.home() / "scoring"
            path, _ = QFileDialog.getSaveFileName(self, "Save BT Manual Scoring file",
                                                  f"{base}_BT_Manual_Scoring{EXT_SCORING}",
                                                  f"BT Manual Scoring file (*{EXT_SCORING})")
            if not path:
                return
            if not path.lower().endswith(EXT_SCORING):
                path += EXT_SCORING
        if not self.audit:
            audit_add(self.audit, "Scoring session created", {
                "video": Path(self.path).name if self.path else "-",
                "behaviours": ", ".join(f"{b['name']} ({b['key'].upper()})" for b in self.inventory)})
        audit_add(self.audit, "Saved", {"events_total": len(self.events), "events_added": self._changes["added"],
                                        "events_deleted": self._changes["deleted"],
                                        "behaviours": len(self.inventory),
                                        "exclusive": "; ".join(" / ".join(p) for p in self._exclusion_list()) or "-",
                                        "ongoing_not_saved": ", ".join(self._open) or "-"})
        prev = self.opened_data if (self.opened_data and Path(str((self.opened_data.get("video") or {}).get(
            "path") or "")).name == Path(str(self.path or "")).name) else None
        embed = int(self.ctrl.cfg.get("embed_video_px", 640) or 0)
        need_copy = bool(embed and self.path and os.path.exists(self.path) and self.cap is not None
                         and not (prev and prev.get("video_copy")))
        dlg = None
        if need_copy:
            dlg = QProgressDialog("Putting a copy of the video in the file…", None, 0, 100, self)
            dlg.setWindowTitle("BT scoring file")
            dlg.setWindowModality(Qt.WindowModal)
            dlg.setMinimumDuration(400)
            dlg.setValue(0)

        def prog(f):
            if dlg is not None:
                dlg.setValue(int(100 * f))
                QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            save_scoring(path, self.path, self.inventory, self.events, self.fps, self.n_frames, self.audit,
                         extra={"exclusion": self._exclusion_list(),
                                "video_size": [int(self.cap.get(3)), int(self.cap.get(4))] if self.cap is not None
                                else None},
                         embed_px=embed if need_copy else 0, previous=prev, progress=prog)
        except Exception as exc:
            self.audit.pop()
            QMessageBox.warning(self, "BT scoring file", f"Could not save:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            if dlg is not None:
                dlg.close()
        self.session_path = path
        self._changes = {"added": 0, "deleted": 0}
        self.ctrl.status.showMessage(f"Saved {Path(path).name} — opens only in BehavioralTracker.", 6000)

    def open_session(self, path=None):
        from tracking.project_file import EXT_SCORING, ProjectFileError, audit_add, audit_verify, load_scoring
        if path is None:
            path, _ = QFileDialog.getOpenFileName(self, "Open BT Manual Scoring file",
                                                  str(Path(self.path).parent if self.path else Path.home()),
                                                  f"BT Manual Scoring file (*{EXT_SCORING})")
            if not path:
                return False
        try:
            data = load_scoring(path)
        except ProjectFileError as exc:
            QMessageBox.warning(self, "BT scoring file", str(exc))
            return False
        except Exception as exc:
            QMessageBox.warning(self, "BT scoring file", f"Could not open this file:\n{exc}")
            return False
        from tracking.project_file import extract_video_copy, has_video_copy
        vp = (data.get("video") or {}).get("path")
        self.events = []
        used_copy = False
        if vp and os.path.exists(vp):
            if self.path != vp or self.cap is None:
                self.open_video(vp)
        elif has_video_copy(data):
            try:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                segs = extract_video_copy(data)
            except Exception as exc:
                segs = None
                QMessageBox.warning(self, "BT scoring file", f"The video copy in this file could not be read:\n{exc}")
            finally:
                QApplication.restoreOverrideCursor()
            if segs:
                self._open_copy(data, segs)
                used_copy = True
        elif vp:
            QMessageBox.information(self, "BT scoring file", f"The video is not in its folder any more:\n{vp}\n\n"
                                    "The behaviours and events are loaded; open the video with 'Open video…'.")
        self.opened_data = data
        self.inventory = list(data.get("inventory") or [])
        self._set_exclusion(data.get("exclusion"))
        self.events = list(data.get("events") or [])
        self._open = {}
        self.audit = list(data.get("audit") or [])
        ok, bad = audit_verify(self.audit)
        audit_add(self.audit, "Opened", {"file": Path(path).name, "events": len(self.events)})
        self.session_path = path
        self._changes = {"added": 0, "deleted": 0}
        self._refresh_inventory()
        self._refresh_events()
        msg = f"Opened {Path(path).name} — {len(self.events)} events."
        if used_copy:
            msg += " The original video is not here: showing the video copy stored in the file."
        if not ok:
            msg += f" ⚠ Audit trail does not match from entry {bad + 1}."
        self.ctrl.status.showMessage(msg, 8000)
        return True

    def show_audit(self):
        from pro_ui.frame_review import AuditDialog
        data = {"format": "BT Manual Scoring", "_path": self.session_path or "(not saved yet)", "audit": self.audit,
                "video": {"name": Path(self.path).name if self.path else "", "fps": self.fps}}
        AuditDialog(self, data).exec()

    def _summary_rows(self):
        out = []
        for b in self.inventory:
            name, kind = b["name"], b.get("kind", "state")
            evs = [e for e in self.events if e["behaviour"] == name]
            if not evs:
                continue
            latency = min(e["start"] for e in evs)
            if kind == "state":
                closed = [e for e in evs if e["stop"] is not None]
                total = sum(e["stop"] - e["start"] for e in closed)
                n = len(closed)
                out.append({"Behaviour": name, "Type": "State", "Bouts": n,
                            "Total_duration_s": round(total, 2),
                            "Mean_duration_s": round(total / n, 2) if n else 0.0,
                            "Latency_s": round(latency, 2)})
            else:
                out.append({"Behaviour": name, "Type": "Point", "Bouts": len(evs),
                            "Total_duration_s": "", "Mean_duration_s": "",
                            "Latency_s": round(latency, 2)})
        return out

    def _update_summary(self):
        rows = self._summary_rows()
        if not rows:
            if self.inventory:
                self.lbl_summary.setText("No events yet — play the video and press a behaviour's key.")
            return
        parts = []
        for r in rows:
            if r["Type"] == "State":
                parts.append(f"<b>{r['Behaviour']}</b>: {r['Bouts']} bouts, {r['Total_duration_s']:.1f}s total")
            else:
                parts.append(f"<b>{r['Behaviour']}</b>: {r['Bouts']}×")
        self.lbl_summary.setText("   ·   ".join(parts))

    def open_analysis(self):
        if not self.events:
            QMessageBox.information(self, "Plots & analysis", "Score some events first.")
            return
        EventAnalysisDialog(self).exec()

    def _export(self):
        if not self.events:
            QMessageBox.information(self, "Export", "There are no events to export yet.")
            return
        base = Path(self.path).with_suffix("") if self.path else Path.home() / "events"
        path, flt = QFileDialog.getSaveFileName(self, "Export events", str(base) + "_events.xlsx",
                                                "Excel workbook (*.xlsx);;CSV files (*.csv)")
        if not path:
            return
        rows = sorted(self.events, key=lambda e: e["start"])
        ev_df = pd.DataFrame([{"Behaviour": e["behaviour"], "Type": "State" if e["kind"] == "state" else "Point",
                               "Start_s": round(e["start"], 3),
                               "Stop_s": None if e["stop"] is None else round(e["stop"], 3),
                               "Duration_s": None if e["stop"] is None else round(e["stop"] - e["start"], 3)}
                              for e in rows])
        sum_df = pd.DataFrame(self._summary_rows())
        from analysis import event_analysis as EA
        dur = (self.n_frames / self.fps) if self.fps and self.n_frames else max(
            [float(e.get("stop") or e["start"]) for e in self.events] + [1.0])
        budget = EA.time_budget(self.events, self.inventory, 0.0, dur, close_open_at=dur)
        bins = EA.time_budget_bins(self.events, self.inventory, 0.0, dur, 60.0, close_open_at=dur)
        trc, trp = EA.transitions(self.events, self.inventory, close_open_at=dur)
        try:
            stem = str(Path(path).with_suffix(""))
            cols = {b["name"]: self._behaviour_color(b["name"]).name() for b in self.inventory}
            title = Path(self.path).stem if self.path else "Observation"
            EA.plot_events(self.events, self.inventory, 0.0, dur, stem + "_events_plot.png", title, cols,
                           close_open_at=dur, dpi=300)
            EA.plot_time_budget(budget, stem + "_time_budget.png", title, cols, dpi=300)
            if path.lower().endswith(".csv"):
                ev_df.to_csv(path, index=False)
                sum_df.to_csv(stem + "_summary.csv", index=False)
                budget.to_csv(stem + "_time_budget.csv", index=False)
                bins.to_csv(stem + "_time_bins.csv", index=False)
            else:
                inv_df = pd.DataFrame([{"Key": b["key"].upper(), "Behaviour": b["name"],
                                        "Type": "State" if b.get("kind", "state") == "state" else "Point"}
                                       for b in self.inventory])
                with pd.ExcelWriter(path) as xl:
                    ev_df.to_excel(xl, sheet_name="Events", index=False)
                    if len(sum_df):
                        sum_df.to_excel(xl, sheet_name="Summary", index=False)
                    if len(budget):
                        budget.to_excel(xl, sheet_name="Time budget", index=False)
                        bins.to_excel(xl, sheet_name="Time bins (60 s)", index=False)
                        trc.to_excel(xl, sheet_name="Transitions")
                    if len(inv_df):
                        inv_df.to_excel(xl, sheet_name="Behavioral Inventory", index=False)
                    names = self._state_names()
                    if self.exclusive and names:
                        mx = pd.DataFrame([["—" if a == b else ("X" if frozenset((a, b)) in self.exclusive else "")
                                            for b in names] for a in names], index=names, columns=names)
                        mx.to_excel(xl, sheet_name="Exclusion matrix")
            self.ctrl.status.showMessage(f"Events exported to {path} (+ events plot and time-budget plot PNG)", 8000)
        except Exception as exc:
            QMessageBox.warning(self, "Export", f"Could not write the file:\n{exc}")

    # ------------------------------------------------------------------ keyboard
    def eventFilter(self, obj, ev):
        if ev.type() not in (QEvent.KeyPress, QEvent.ShortcutOverride):
            return False
        if not self.isVisible() or QApplication.activeModalWidget() is not None:
            return False
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit)) or \
                (isinstance(fw, QComboBox) and fw.isEditable()):
            return False
        key = ev.key()
        if key == Qt.Key_Space:
            handled = True
            if ev.type() == QEvent.KeyPress:
                self.toggle_play()
        elif key == Qt.Key_Left:
            handled = True
            if ev.type() == QEvent.KeyPress:
                self.step(-1)
        elif key == Qt.Key_Right:
            handled = True
            if ev.type() == QEvent.KeyPress:
                self.step(1)
        elif key in (Qt.Key_Delete, Qt.Key_Backspace) and self.tbl_events.hasFocus():
            handled = True
            if ev.type() == QEvent.KeyPress:
                self._delete_events()
        else:
            text = ev.text()
            match = bool(text) and any(b["key"] == text.lower() for b in self.inventory)
            handled = match
            if match and ev.type() == QEvent.KeyPress and not ev.isAutoRepeat():
                self._score_key(text)
        if handled:
            ev.accept()
            return True
        return False

    def closeEvent(self, ev):
        if self.cap is not None:
            self.cap.release()
        super().closeEvent(ev)


class UXController(QObject):
    DEFAULT_CFG = {"bg_samples": 100, "max_area": 0, "max_jump": 100, "color_mode": "gray", "bin_size_s": 60,
                   "max_gap_s": 0.5, "min_move_px": 0.0, "auto_mask_margin": 0, "engine": "robust",
                   "polarity": "auto", "min_contrast": 12, "preview_samples": 3, "entry_rule": "paws",
                   "entry_body_pct": 80, "fast_tracking": False, "tracking_point": "centre",
                   "ldb_dark_covered": True, "embed_video_px": 640, "learn_layouts": True,
                   "hold_lost": True, "low_light": "off", "body_length_px": 0, "adapt_background": True}

    def __init__(self):
        super().__init__()
        self.win = self._load_window()
        w = self.win
        self.stack = find(w, QStackedWidget, "stackedPages")
        self.status = w.statusBar()

        # ---- state
        self.test_key = "open_field"
        self.player = Player()
        self.raw_frame = None
        self.cur_frame = None
        self.crop = None                 # {"matrix", "inv", "w", "h", "corners"}
        self.last_summary = None
        self.last_output_dir = None
        self.exp_videos = []             # videos loaded for single / individual tracking (Experiment page)
        self.batch_items = []            # the Batch-Processing queue (kept separate from exp_videos)
        self.thread = None
        self.worker = None
        self.run_kind = None             # "single" | "batch"
        self._batch_frames = {}          # batch video path -> latest live tracking frame (QImage)
        self._batch_cur_path = None      # batch video being tracked right now
        self.live_dlg = None             # Batch ▸ live tracking window
        self.tracking = False
        self._row_map = []
        self._single_error = None
        self._single_subject = ""
        self._batch_t0 = 0.0
        self._scale_auto_done = False
        self._sync = False
        self._bg_threads = []
        self._preview = None             # cached detection-preview model
        self._preview_busy = False
        self._bg_cache = {}              # background image per (video, crop) for auto zones
        self._bg_waiters = {}            # callbacks waiting for a background that is being built
        self._shape_warned = set()       # (video, test) pairs already warned about a test-type mismatch
        self._guided_shape = "rect"
        self.history = load_json(HISTORY_PATH, [])
        self.cfg = dict(self.DEFAULT_CFG)
        self.cfg.update(load_json(SETTINGS_PATH, {}))
        self._apply_results_root()
        if self.cfg.get("engine") not in [k for k, _t in ENGINES]:
            self.cfg["engine"] = "classic" if self.cfg.get("engine") else "robust"   # older settings files
        self.experimental_design = []
        self.advanced_statistics = None
        self.covered_override = None     # None = automatic (light/dark box: the dark box); else set of zone names

        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._play_tick)
        style = QApplication.style()
        self.icon_play = style.standardIcon(QStyle.SP_MediaPlay)
        self.icon_pause = style.standardIcon(QStyle.SP_MediaPause)

        self._bind_navigation()
        self._bind_dashboard()
        self._bind_experiment()
        self._bind_setup()
        self._bind_tracking()
        from pro_ui.chambers_ui import ChamberUI
        self.chamber_ui = ChamberUI(self)        # chambers: several arenas in one video (off until switched on)
        self._bind_analysis()
        self._bind_batch()
        self._bind_shortcuts()
        # side panels scroll instead of squeezing their fields (small screens / display scaling)
        pinned = ((self.btn_live,) if LIVE_TRACKING else ()) + (self.btn_test, self.btn_start)
        self._scrollify_card(find(self.win, QFrame, "cardTrackingSettings"), pinned=pinned)
        self._scrollify_card(find(self.win, QFrame, "cardArenaSettings"))
        self.refresh_dashboard()
        self.select_test("open_field", navigate=False)
        self.status.showMessage("Ready — pick a test on the Dashboard or load a video on the Experiment page.", 8000)

    # ------------------------------------------------------------------ window
    def _load_window(self):
        loader = QUiLoader()
        loader.setWorkingDirectory(QDir(str(HERE)))
        ui_data = None
        try:                                   # protected build: the .ui is decrypted in memory
            import _sec.loader as _secloader
            ui_data = _secloader.ui_bytes()
        except Exception:
            ui_data = None
        if ui_data is not None:
            from PySide6.QtCore import QBuffer, QByteArray, QIODevice
            buf = QBuffer()
            buf.setData(QByteArray(ui_data))
            buf.open(QIODevice.ReadOnly)
            win = loader.load(buf)
            buf.close()
        else:
            f = QFile(str(HERE / "behavioral_tracker.ui"))
            f.open(QFile.ReadOnly)
            win = loader.load(f)
            f.close()
        if win is None:
            raise RuntimeError(loader.errorString())
        # the stylesheet uses url(icons/...) relative paths -> make them absolute
        # instead of changing the process working directory
        ss = win.styleSheet()
        if "url(icons/" in ss:
            ss = ss.replace("url(icons/", "url(" + (HERE / "icons").as_posix() + "/")
        # extra rules so the parts the main style sheet leaves to the palette look the same in both themes
        ss += ("\nQMenu { background: #ffffff; border: 1px solid #d5dce8; padding: 4px; }"
               "\nQMenu::item { padding: 5px 22px 5px 22px; border-radius: 4px; color: #1f2a44; }"
               "\nQMenu::item:selected { background: #eaf2ff; color: #1f2a44; }"
               "\nQMenu::item:disabled { color: #94a3b8; }"
               "\nQMenu::separator { height: 1px; background: #e3e8f0; margin: 4px 8px; }"
               "\nQScrollBar:vertical { background: #f3f6fb; width: 12px; margin: 0; }"
               "\nQScrollBar::handle:vertical { background: #c7d0de; min-height: 30px; border-radius: 5px; margin: 2px; }"
               "\nQScrollBar:horizontal { background: #f3f6fb; height: 12px; margin: 0; }"
               "\nQScrollBar::handle:horizontal { background: #c7d0de; min-width: 30px; border-radius: 5px; margin: 2px; }"
               "\nQScrollBar::add-line, QScrollBar::sub-line { width: 0px; height: 0px; }"
               "\nQGroupBox { border: 1px solid #e3e8f0; border-radius: 8px; margin-top: 10px; padding-top: 8px; }"
               "\nQGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #1b2a4a; }"
               "\nQListWidget, QTextBrowser, QTextEdit, QTreeWidget { background: #ffffff; border: 1px solid #e3e8f0; }"
               "\nQTableWidget QTableCornerButton::section { background: #f7f9fc; border: none; }"
               "\nQDialog, QMessageBox { background: #f3f6fb; }"
               "\nQStatusBar { background: #ffffff; border-top: 1px solid #e3e8f0; }"
               "\nQCheckBox::indicator, QTableView::indicator, QListView::indicator { width: 15px; height: 15px;"
               " border: 1px solid #b6bfcc; border-radius: 3px; background: #ffffff; }"
               "\nQCheckBox::indicator:hover, QTableView::indicator:hover { border-color: #1f6feb; }"
               "\nQCheckBox::indicator:checked, QTableView::indicator:checked, QListView::indicator:checked {"
               " background: #1f6feb; border-color: #1f6feb; image: url(" + (HERE / "icons").as_posix()
               + "/check_white.svg); }"
               "\nQCheckBox::indicator:disabled { background: #e9edf3; border-color: #d5dce8; }")
        win.setStyleSheet(ss)
        try:
            from pro_ui import theme
            theme.adopt_tree(win)
        except Exception:
            pass
        icon = HERE / "icons" / "app_icon.ico"
        if icon.exists():
            win.setWindowIcon(QIcon(str(icon)))
        win.setWindowTitle(f"{win.windowTitle() or 'BehavioralTracker'}  ·  v{APP_VERSION}")
        win.installEventFilter(self)
        return win

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if obj is getattr(self, "lst_videos", None):
                self.remove_videos(self._selected_list_paths())
                return True
            if obj is getattr(self, "tbl_batch", None):
                self.batch_remove_selected()
                return True
        if obj is self.win and ev.type() == QEvent.Close:
            if self.tracking and QMessageBox.question(
                    self.win, "Tracking in progress", "Tracking is still running. Stop it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return True
            if getattr(getattr(self, "freezing_page", None), "busy", None) and QMessageBox.question(
                    self.win, "Freezing analysis in progress",
                    "The freezing analysis is still running. Stop it and quit?") != QMessageBox.Yes:
                ev.ignore()
                return True
            self._stop_worker(wait=True)
            self.player.close()
            if hasattr(self, "freezing_page"):
                self.freezing_page.shutdown()
            if hasattr(self, "freeze_panel"):
                self.freeze_panel.shutdown()
        return False

    def show(self):
        self.win.show()

    def _thread_run(self, fn, on_done, on_fail=None):
        """Run fn() on a worker thread; callbacks always run on the GUI thread."""
        th = QThread()
        wk = FuncWorker(fn)
        wk.moveToThread(th)
        entry = {"th": th, "wk": wk}

        def finish():
            th.quit()
            th.wait(5000)
            self._bg_threads = [e for e in self._bg_threads if e["th"] is not th]
            th.deleteLater()

        def ok(value):
            try:
                on_done(value)
            finally:
                finish()

        def bad(msg):
            try:
                (on_fail or (lambda m: QMessageBox.warning(self.win, "Error", m)))(msg)
            finally:
                finish()
        entry["ok"], entry["bad"] = _Relay(ok), _Relay(bad)
        th.started.connect(wk.run)
        wk.done.connect(entry["ok"].call)
        wk.failed.connect(entry["bad"].call)
        self._bg_threads.append(entry)
        th.start()

    # ------------------------------------------------------------------ navigation
    def _bind_navigation(self):
        self.nav = {n: find(self.win, QPushButton, n) for n in
                    ("btnNavDashboard", "btnNavExperiment", "btnNavVideoSetup", "btnNavTracking",
                     "btnNavAnalysis", "btnNavResults", "btnNavBatch")}
        # Event Scoring: its own page (index 6) + a sidebar button under Batch Processing
        self.event_page = EventScoringPage(self)
        self.EVENT_PAGE_INDEX = self.stack.addWidget(self.event_page)
        nav_batch = self.nav["btnNavBatch"]
        self.btn_nav_events = QPushButton("  Event Scoring", nav_batch.parent())
        ev_icon = QIcon()
        ev_icon.addFile(str(HERE / "icons" / "nav_events.svg"), QSize(), QIcon.Normal, QIcon.Off)
        ev_icon.addFile(str(HERE / "icons" / "nav_events_on.svg"), QSize(), QIcon.Normal, QIcon.On)
        self.btn_nav_events.setIcon(ev_icon)
        self.btn_nav_events.setIconSize(nav_batch.iconSize())
        self.btn_nav_events.setMinimumHeight(nav_batch.minimumHeight())
        self.btn_nav_events.setSizePolicy(nav_batch.sizePolicy())
        self.btn_nav_events.setCheckable(True)
        self.btn_nav_events.setCursor(Qt.PointingHandCursor)
        self.btn_nav_events.setToolTip("Manual keyboard behaviour scoring (build a behavioral inventory, play the video, "
                                       "press each behaviour's key).")
        # Freezing Analysis: one page, two side-bar items (Individual / Batch) right under Tracking -- the flow is
        # track or freeze -> Analysis & Results.  Event Scoring (manual scoring) always stays the last item.
        from pro_ui.freezing_page import FreezingPage
        self.freezing_page = FreezingPage(self)
        self.FREEZE_PAGE_INDEX = self.stack.addWidget(self.freezing_page)

        def side_button(text, icon, icon_on, tip):
            b = QPushButton(text, nav_batch.parent())
            ic = QIcon()
            ic.addFile(str(HERE / "icons" / icon), QSize(), QIcon.Normal, QIcon.Off)
            ic.addFile(str(HERE / "icons" / icon_on), QSize(), QIcon.Normal, QIcon.On)
            b.setIcon(ic)
            b.setIconSize(nav_batch.iconSize())
            b.setMinimumHeight(nav_batch.minimumHeight())
            b.setSizePolicy(nav_batch.sizePolicy())
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(tip)
            return b
        self.btn_nav_freezing = side_button("  Freezing – Individual", "nav_freezing.svg", "nav_freezing_on.svg",
                                            "Freezing analysis of one video (fear conditioning): run, review and "
                                            "tune; the results open in Analysis & Results.")
        self.btn_nav_freezing_batch = side_button("  Freezing – Batch", "nav_freezing_batch.svg",
                                                  "nav_freezing_batch_on.svg",
                                                  "Freezing analysis of many videos with the same settings + one "
                                                  "summary workbook.")
        sb = self._layout_with(find(self.win, QFrame, "sidebar").layout(), nav_batch)
        if sb is not None:
            # Tracking → Batch Processing → Freezing – Individual → Freezing – Batch → Analysis → Results →
            # Event Scoring (always last)
            track_btn = self.nav["btnNavTracking"]
            sb.removeWidget(nav_batch)
            sb.insertWidget(sb.indexOf(track_btn) + 1, nav_batch)
            sb.insertWidget(sb.indexOf(nav_batch) + 1, self.btn_nav_freezing)
            sb.insertWidget(sb.indexOf(self.btn_nav_freezing) + 1, self.btn_nav_freezing_batch)
            sb.insertWidget(sb.indexOf(self.nav["btnNavResults"]) + 1, self.btn_nav_events)
        self.nav["btnNavFreezing"] = self.btn_nav_freezing
        self.nav["btnNavFreezingBatch"] = self.btn_nav_freezing_batch
        self.nav["btnNavEvents"] = self.btn_nav_events
        # manage the highlighted item ourselves (the extra button is not in the .ui autoexclusive set)
        for b in self.nav.values():
            b.setAutoExclusive(False)
        pages = {"btnNavDashboard": 0, "btnNavExperiment": 1, "btnNavVideoSetup": 2, "btnNavTracking": 3,
                 "btnNavAnalysis": 4, "btnNavResults": 4, "btnNavBatch": 5, "btnNavEvents": self.EVENT_PAGE_INDEX}
        for name, idx in pages.items():
            self.nav[name].clicked.connect(lambda _=False, i=idx, n=name: self.go(i, results=(n == "btnNavResults")))
        self.btn_nav_freezing.clicked.connect(lambda _=False: self.go_freezing("single"))
        self.btn_nav_freezing_batch.clicked.connect(lambda _=False: self.go_freezing("batch"))
        self.stack.currentChanged.connect(self._page_changed)
        self._add_help_bar()
        self._add_licence_badge()
        self._apply_simple_view(bool(self.cfg.get("simple_view", False)), save=False)

    # pages hidden by "Simple view" (the main path stays: Dashboard -> Experiment -> Video Setup -> Tracking /
    # Freezing -> Results)
    SIMPLE_HIDE = ("btnNavBatch", "btnNavFreezingBatch", "btnNavAnalysis", "btnNavEvents")

    def _apply_simple_view(self, on, save=True):
        on = bool(on)
        self.cfg["simple_view"] = on
        for name in self.SIMPLE_HIDE:
            b = self.nav.get(name)
            if b is not None:
                b.setVisible(not on)
        chk = getattr(self, "chk_simple", None)
        if chk is not None and chk.isChecked() != on:
            chk.blockSignals(True)
            chk.setChecked(on)
            chk.blockSignals(False)
        if on:
            idx = self.stack.currentIndex()
            fz_batch = (idx == getattr(self, "FREEZE_PAGE_INDEX", -1)
                        and getattr(self.freezing_page, "mode", "single") == "batch")
            if idx in (5, getattr(self, "EVENT_PAGE_INDEX", -1)) or fz_batch:
                self.go(0)
        if save:
            self.save_settings_now()

    def _add_help_bar(self):
        """Simple view · ❔ Help · ⓘ About · ✉ Contact and support at the bottom of the side bar."""
        side = find(self.win, QFrame, "sidebar")
        lay = side.layout() if side is not None else None
        if lay is None:
            return
        self.chk_simple = QCheckBox("Simple view", side)
        self.chk_simple.setToolTip("Show only the main pages: Dashboard, Experiment, Video Setup, Tracking, Freezing "
                                   "– Individual and Results. Batch Processing, Freezing – Batch, Analysis "
                                   "(statistics) and Event Scoring come back when you untick it.")
        self.chk_simple.setChecked(bool(self.cfg.get("simple_view", False)))
        self.chk_simple.setStyleSheet("QCheckBox{color:#475569; padding-left:8px;}")
        self.chk_simple.toggled.connect(lambda on: self._apply_simple_view(on))
        lay.addWidget(self.chk_simple)
        css = ("QToolButton{border:1px solid #d5dce8; border-radius:6px; padding:3px 8px; background:#ffffff;"
               " color:#1f2a44;} QToolButton:hover{background:#eef3fb;}")
        bar = QWidget(side)
        g = QGridLayout(bar)
        g.setContentsMargins(4, 6, 4, 2)
        g.setHorizontalSpacing(4)
        g.setVerticalSpacing(4)
        for (text, tip, fn), (r, c, cs) in zip(
                (("❔ Help", "How to use BehavioralTracker, step by step, and the how-to videos.", self.show_help),
                 ("ⓘ About", "The version of BehavioralTracker and your licence.", self.show_about),
                 ("✉ Contact and support", "Email and phone number of the BehavioralTracker contact person.",
                  self.show_contact)),
                ((0, 0, 1), (0, 1, 1), (1, 0, 2))):
            b = QToolButton(bar)
            b.setText(text)
            b.setToolTip(tip)
            b.setStyleSheet(css)
            b.setCursor(Qt.PointingHandCursor)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(fn)
            g.addWidget(b, r, c, 1, cs)
        lay.addWidget(bar)

    def save_settings_now(self):
        try:
            save_json(SETTINGS_PATH, self.cfg)
        except Exception:
            pass

    def _text_dialog(self, title, markdown, extra=None):
        dlg = QDialog(self.win)
        dlg.setWindowTitle(title)
        dlg.resize(860, 700)
        v = QVBoxLayout(dlg)
        if extra is not None:
            v.addWidget(extra)
        if len(markdown) == 1:
            tb = QTextBrowser(dlg)
            tb.setOpenExternalLinks(True)
            tb.setMarkdown(markdown[0][1])
            v.addWidget(tb, 1)
        else:
            tabs = QTabWidget(dlg)
            for name, md in markdown:
                tb = QTextBrowser()
                tb.setOpenExternalLinks(True)
                tb.setMarkdown(md)
                tabs.addTab(tb, name)
            v.addWidget(tabs, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.exec()

    def show_help(self):
        from pro_ui import help_text as HT
        videos = Path(__file__).resolve().parent.parent / "resources" / "docs" / "videos"
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        b = QPushButton("▶ How-to videos")
        b.setToolTip("Short videos: freezing in 2 minutes, clear / reflective boxes, teach BT a behaviour.")
        b.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(videos))))
        b.setEnabled(videos.exists())
        h.addWidget(b)
        b = QPushButton("✉ Contact and support")
        b.clicked.connect(self.show_contact)
        h.addWidget(b)
        h.addStretch(1)
        self._text_dialog("How to use BehavioralTracker", [("How to use", HT.MANUAL)], extra=row)

    def show_about(self):
        from pro_ui import help_text as HT
        from tracking.project_file import licence_info
        info = licence_info() or {}
        dlg = QDialog(self.win)
        dlg.setWindowTitle("About BehavioralTracker")
        v = QVBoxLayout(dlg)
        lab = QLabel(f"<b style='font-size:15pt'>BehavioralTracker</b><br>Version {HT.VERSION}"
                     + (f"<br><br>Licensed to {info.get('licensee')}<br>Licence No {info.get('licence_no')}"
                        if info else ""))
        lab.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(lab)
        bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        b = bb.addButton("✉ Contact and support", QDialogButtonBox.ActionRole)
        b.clicked.connect(self.show_contact)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.setMinimumWidth(380)
        dlg.exec()

    def show_contact(self):
        """Contact and support: the contact person's email and phone number."""
        from pro_ui import help_text as HT
        dlg = QDialog(self.win)
        dlg.setWindowTitle("Contact and support")
        v = QVBoxLayout(dlg)
        v.setSpacing(10)
        head = QLabel(f"<b style='font-size:13pt'>Contact and support</b><br>{HT.SUPPORT_NAME}")
        v.addWidget(head)
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        for r, (name, value, copy_tip) in enumerate((("Email", HT.SUPPORT_EMAIL, "Copy the email address"),
                                                     ("Phone", HT.SUPPORT_PHONE, "Copy the phone number"))):
            grid.addWidget(QLabel(f"<b>{name}</b>"), r, 0)
            val = QLabel(value)
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setStyleSheet("font-size:12pt;")
            grid.addWidget(val, r, 1)
            cp = QToolButton()
            cp.setText("Copy")
            cp.setToolTip(copy_tip)
            cp.clicked.connect(lambda _c=False, s=value, w=cp: (QApplication.clipboard().setText(s),
                                                                w.setText("Copied ✓")))
            grid.addWidget(cp, r, 2)
        grid.setColumnStretch(1, 1)
        v.addLayout(grid)
        note = QLabel("When you write about a problem, please say what you did, what you expected and what "
                      "happened, and attach the log file (crash.log or BehavioralTracker.log) from "
                      + ("%APPDATA%\\BehavioralTracker" if sys.platform == "win32" else "~/.behavioraltracker")
                      + " and, if possible, a short piece of the video.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#475569;")
        v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Close, dlg)
        b = bb.addButton("✉ Send an email", QDialogButtonBox.ActionRole)
        b.setToolTip(f"Opens an email to {HT.SUPPORT_EMAIL} with the BT version and your system filled in.")
        b.clicked.connect(lambda: self.email_support())
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.setMinimumWidth(460)
        dlg.exec()

    def email_support(self, subject=None):
        """mailto: the contact person, with the version / system / licence number filled in."""
        import platform
        from urllib.parse import quote
        from pro_ui import help_text as HT
        from tracking.project_file import licence_info
        info = licence_info() or {}
        body = ("Please describe what you did, what you expected and what happened.\n\n\n"
                f"---\nBehavioralTracker {HT.VERSION}\n"
                f"System: {platform.system()} {platform.release()} ({platform.machine()})\n"
                + (f"Licence No: {info.get('licence_no')}\n" if info else "")
                + "Logs: attach crash.log / BehavioralTracker.log from "
                + ("%APPDATA%\\BehavioralTracker" if sys.platform == "win32" else "~/.behavioraltracker") + "\n")
        subj = subject or f"BehavioralTracker {HT.VERSION} support"
        url = QUrl(f"mailto:{HT.SUPPORT_EMAIL}?subject={quote(subj)}&body={quote(body)}")
        if not QDesktopServices.openUrl(url):
            QApplication.clipboard().setText(HT.SUPPORT_EMAIL)
            QMessageBox.information(self.win, "Contact and support",
                                    f"No email program opened. The address {HT.SUPPORT_EMAIL} is copied: paste it "
                                    f"into your email, or call {HT.SUPPORT_PHONE}.")

    def _add_licence_badge(self):
        """Licensed edition: the licence holder and licence number at the bottom of the side bar."""
        from tracking.project_file import licence_info
        info = licence_info()
        side = find(self.win, QFrame, "sidebar")
        lay = side.layout() if side is not None else None
        if not info or lay is None:
            return
        box = QFrame(side)
        box.setObjectName("licenceBadge")
        box.setStyleSheet("QFrame#licenceBadge{background:#f3f6fb;border:1px solid #dbe3ef;border-radius:8px;}"
                          "QFrame#licenceBadge QLabel{background:transparent;border:none;}")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(1)
        cap = QLabel("🔑  Licensed to")
        cap.setStyleSheet("color:#64748b;font-size:8pt;")
        who = QLabel(info["licensee"])
        who.setWordWrap(True)
        who.setStyleSheet("color:#0f172a;font-weight:700;font-size:9pt;")
        no = QLabel(f"Licence No: <b>{info['licence_no']}</b>")
        no.setStyleSheet("color:#334155;font-size:8pt;")
        no.setTextInteractionFlags(Qt.TextSelectableByMouse)
        for wd in (cap, who, no):
            v.addWidget(wd)
        if info.get("expires"):
            ex = QLabel(f"Valid until {info['expires']}")
            ex.setStyleSheet("color:#64748b;font-size:8pt;")
            v.addWidget(ex)
        box.setToolTip(f"Licence No {info['licence_no']} · issued {info.get('issued') or '—'} · "
                       f"Machine ID {info.get('machine_id') or '—'}")
        lay.addWidget(box)
        self.licence_badge = box

    def go_freezing(self, mode="single"):
        """Freezing – Individual / Freezing – Batch (one page, two side-bar items)."""
        self.freezing_page.set_mode(mode)
        self.go(self.FREEZE_PAGE_INDEX)

    def go(self, index, results=False):
        self.stack.setCurrentIndex(index)
        fz = "btnNavFreezingBatch" if getattr(getattr(self, "freezing_page", None), "mode", "") == "batch" \
            else "btnNavFreezing"
        target = {0: "btnNavDashboard", 1: "btnNavExperiment", 2: "btnNavVideoSetup", 3: "btnNavTracking",
                  4: "btnNavResults" if results else "btnNavAnalysis", 5: "btnNavBatch",
                  getattr(self, "FREEZE_PAGE_INDEX", -2): fz,
                  getattr(self, "EVENT_PAGE_INDEX", -1): "btnNavEvents"}[index]
        for n, b in self.nav.items():
            b.setChecked(n == target)
        if index == 4 and hasattr(self, "analysis_tabs"):
            self.analysis_tabs.setCurrentIndex(self.analysis_tabs.indexOf(self.tab_events) if results else 0)
            if getattr(self, "results_mode", "tracking") == "freezing":
                self.freeze_panel.tabs.setCurrentIndex(2 if results else 0)

    def _page_changed(self, index):
        self._stop_playback()
        if index != 2 and hasattr(self, "canvas") and self.canvas.tool is not None:
            self.canvas.cancel_tool()
        if index == 3 and hasattr(self, "cmb_darkbox"):
            self._sync_darkbox()
        if index != 3 and getattr(self, "_measure_pts", None) is not None:
            self._end_measure()
        if hasattr(self, "chamber_ui"):
            if index != 3:
                self.chamber_ui.to_all()                # a chamber's own settings are only shown on Tracking
            elif self.chamber_ui.view is None:
                self.chamber_ui._refresh_tracking_row()
        if index in (2, 3):
            self._refresh_views()
        if index == 0:
            self.refresh_dashboard()
        if hasattr(self, "event_page"):
            if index == getattr(self, "EVENT_PAGE_INDEX", -1):
                self.event_page.refresh_sources()
            else:
                self.event_page.pause()
        if hasattr(self, "freezing_page"):
            if index == getattr(self, "FREEZE_PAGE_INDEX", -2):
                self.freezing_page.refresh_sources()
            else:
                self.freezing_page.pause()

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self.win, activated=self.add_video_files)
        QShortcut(QKeySequence("Space"), self.win, activated=self.toggle_play)
        QShortcut(QKeySequence("Escape"), self.win, activated=lambda: self.canvas.cancel_tool())
        QShortcut(QKeySequence("Return"), self.win, activated=lambda: self.canvas.finish_tool())
        QShortcut(QKeySequence("Left"), self.win, activated=lambda: self.step(-1))
        QShortcut(QKeySequence("Right"), self.win, activated=lambda: self.step(1))
        QShortcut(QKeySequence("Ctrl+Z"), self.win, activated=lambda: self.canvas.undo())
        QShortcut(QKeySequence("Ctrl+Y"), self.win, activated=lambda: self.canvas.redo())
        for i, key in enumerate(("Ctrl+1", "Ctrl+2", "Ctrl+3", "Ctrl+4", "Ctrl+5", "Ctrl+6")):
            QShortcut(QKeySequence(key), self.win, activated=lambda i=i: self.go(i))

    # ------------------------------------------------------------------ dashboard
    def _add_freezing_tile(self):
        """Dashboard ▸ Start a New Experiment: a Freezing tile (-> Freezing – Individual)."""
        ref = self.win.findChild(QToolButton, "dashTile_custom")
        card = self.win.findChild(QFrame, "cardStartNew")
        if ref is None or card is None:
            return
        grid = self._layout_with(card.layout(), ref)
        if grid is None:
            return
        tile = QToolButton(ref.parentWidget())
        tile.setObjectName("dashTile_freezing")
        tile.setText("Freezing")
        tile.setIcon(QIcon(str(HERE / "icons" / "nav_freezing.svg")))
        tile.setIconSize(ref.iconSize())
        tile.setToolButtonStyle(ref.toolButtonStyle())
        tile.setProperty("cssClass", ref.property("cssClass"))
        tile.setMinimumSize(ref.minimumSize())
        tile.setMaximumSize(ref.maximumSize())
        tile.setSizePolicy(ref.sizePolicy())
        tile.setStyleSheet(ref.styleSheet())
        tile.setCursor(Qt.PointingHandCursor)
        tile.setToolTip("Freezing analysis (fear conditioning, contextual / cued fear): motion and freezing scored "
                        "automatically.")
        tile.clicked.connect(lambda _=False: self.go_freezing("single"))
        try:
            r, _c, _rs, _cs = grid.getItemPosition(grid.indexOf(ref))
            grid.addWidget(tile, r + 1, 0)
        except Exception:
            grid.addWidget(tile)
        self.dash_tile_freezing = tile

    def _bind_dashboard(self):
        w = self.win
        for key in TESTS:
            btn = w.findChild(QToolButton, f"dashTile_{key}")
            if btn is not None:
                btn.clicked.connect(lambda _=False, k=key: self.select_test(k, navigate=True))
        self._add_freezing_tile()
        btn_all = find(w, QPushButton, "btnViewAll")
        btn_all.clicked.connect(self.view_all_experiments)
        self.btn_clear_history = QPushButton("Clear", btn_all.parentWidget())
        self.btn_clear_history.setProperty("cssClass", "link")
        self.btn_clear_history.setCursor(Qt.PointingHandCursor)
        self.btn_clear_history.setToolTip("Empty this list and the counters (start fresh). Result folders on disk "
                                          "are not touched.")
        self.btn_clear_history.clicked.connect(self.clear_history)
        hl = self._layout_with(btn_all.parentWidget().layout(), btn_all)
        if hl is not None:
            hl.insertWidget(hl.indexOf(btn_all), self.btn_clear_history)
        for lab in w.findChildren(QLabel):
            if lab.text().strip() == "Animals per video":
                lab.setText("Animals")
        self.tbl_recent = find(w, QTableWidget, "tblRecent")
        self.tbl_recent.setRowCount(0)
        self.tbl_recent.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_recent.setSelectionBehavior(QTableWidget.SelectRows)
        h = self.tbl_recent.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.Fixed)
        self.tbl_recent.setColumnWidth(2, 30)
        self.tbl_recent.cellClicked.connect(self._open_recent)
        search = w.findChild(QLineEdit, "searchEdit")
        if search is not None:
            search.setPlaceholderText("Search experiments…")
            search.textChanged.connect(self._filter_recent)

    def refresh_dashboard(self):
        w = self.win
        hist = self.history
        find(w, QLabel, "lblStatExperiments").setText(str(len(hist)))
        find(w, QLabel, "lblStatVideos").setText(str(sum(int(h.get("videos", 1)) for h in hist)))
        find(w, QLabel, "lblStatAnimals").setText(str(len({h.get("subject") for h in hist if h.get("subject")})))
        find(w, QLabel, "lblStatTestTypes").setText(str(len({h.get("test") for h in hist if h.get("test")})))
        self._fill_recent(hist[::-1][:5])

    def _fill_recent(self, items):
        t = self.tbl_recent
        t.clearSpans()
        if not items:
            t.setRowCount(1)
            it = QTableWidgetItem("No experiments yet — choose a test on the left to start.")
            it.setFlags(Qt.ItemIsEnabled)
            it.setForeground(QColor("#94a3b8"))
            t.setItem(0, 0, it)
            for c in (1, 2):
                t.setItem(0, c, QTableWidgetItem(""))
            t.setSpan(0, 0, 1, 3)
            if hasattr(self, "btn_clear_history"):
                self.btn_clear_history.setVisible(False)
            return
        if hasattr(self, "btn_clear_history"):
            self.btn_clear_history.setVisible(True)
        t.setRowCount(len(items))
        for r, h in enumerate(items):
            for c, text in enumerate((h.get("name", ""), h.get("date", ""), "›")):
                it = QTableWidgetItem(text)
                it.setData(Qt.UserRole, h.get("output_dir", ""))
                t.setItem(r, c, it)

    def _filter_recent(self, text):
        text = text.strip().lower()
        for r in range(self.tbl_recent.rowCount()):
            name = self.tbl_recent.item(r, 0).text().lower()
            self.tbl_recent.setRowHidden(r, bool(text) and text not in name)

    def _open_recent(self, row, _col):
        it = self.tbl_recent.item(row, 0)
        if it is not None and it.data(Qt.UserRole):
            self.open_results_folder(it.data(Qt.UserRole))

    def clear_history(self):
        if not self.history:
            return
        if QMessageBox.question(self.win, "Clear history",
                                f"Remove all {len(self.history)} experiment(s) from this list and reset the "
                                "counters?\n\nThe result folders and Excel files on disk are NOT deleted.") \
                != QMessageBox.Yes:
            return
        self.history = []
        save_json(HISTORY_PATH, [])
        self.refresh_dashboard()
        self.status.showMessage("History cleared — starting fresh.", 5000)

    def view_all_experiments(self):
        self._filter_recent("")
        self._fill_recent(self.history[::-1])
        self.status.showMessage(f"Showing all {len(self.history)} experiments", 4000)

    def open_results_folder(self, folder):
        if not folder or not os.path.isdir(folder):
            QMessageBox.information(self.win, "Results", "The results folder for this experiment no longer exists.")
            return
        bta = sorted(Path(folder).glob("*.bta"))
        if bta and self.open_bta(str(bta[0])):      # results + frame review + audit trail
            return
        summary = self._read_summary(folder)
        if summary is None:
            QMessageBox.information(self.win, "Results", "No analysis workbook found in that folder.")
            return
        self.show_results(summary)
        self.go(4)

    @staticmethod
    def _read_summary(folder):
        for f in sorted(Path(folder).glob("*_Analysis.xlsx")):
            try:
                try:
                    df = pd.read_excel(f, sheet_name="All values")
                except Exception:
                    df = pd.read_excel(f, sheet_name="Summary")          # older workbooks
                out = {k: (v.item() if hasattr(v, "item") else v) for k, v in df.iloc[0].to_dict().items()}
                out["Output_folder"] = str(folder)
                return out
            except Exception:
                continue
        return None

    def add_history(self, summary, video_path, subject=""):
        name = self.le_name.text().strip() or TESTS[self.test_key][0].replace(" ", "")
        if self.run_kind == "batch":
            name = f"{name}_{Path(video_path).stem}"
        entry = {"name": name, "date": datetime.date.today().isoformat(), "test": self.test_key,
                 "subject": subject, "video": video_path, "videos": 1,
                 "output_dir": summary.get("Output_folder", "")}
        self.history.append(entry)
        save_json(HISTORY_PATH, self.history[-500:])
        if entry["output_dir"] and os.path.isdir(entry["output_dir"]):
            save_json(os.path.join(entry["output_dir"], "experiment_info.json"), {
                **entry, "investigator": self.le_investigator.text(), "notes": self.le_notes.text(),
                "pixels_per_cm": self.spin_ppc.value(), "arena_cm": [self.spin_len.value(), self.spin_wid.value()],
                "arena_shape": self.cmb_shape.currentText(), "detection": self.cfg,
                "app_version": APP_VERSION, "created": datetime.datetime.now().isoformat(timespec="seconds")})

    # ------------------------------------------------------------------ experiment page
    def _bind_experiment(self):
        w = self.win
        find(w, QPushButton, "btnSourceFiles").clicked.connect(self.add_video_files)
        find(w, QPushButton, "btnSourceCamera").clicked.connect(self.open_camera)
        find(w, QPushButton, "btnAddSubject").clicked.connect(self.add_subject)
        self.tbl_subjects = find(w, QTableWidget, "tblSubjects")
        self.tbl_subjects.setRowCount(0)
        self.tbl_subjects.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tbl_subjects.customContextMenuRequested.connect(self._subject_menu)
        self.tbl_subjects.setEditTriggers(QTableWidget.DoubleClicked)
        self.le_name = find(w, QLineEdit, "leExperimentName")
        self.le_investigator = find(w, QLineEdit, "leInvestigator")
        self.le_notes = find(w, QLineEdit, "leNotes")
        # nothing is pre-filled: the user types what they need
        self.le_name.clear()
        self.le_name.setPlaceholderText("e.g. EPM_Cohort1_Day1")
        self.le_investigator.clear()
        self.le_investigator.setPlaceholderText("Your name")
        self.le_notes.clear()
        desc = w.findChild(QPlainTextEdit, "teDescription")
        if desc is not None:
            desc.clear()
            desc.setPlaceholderText("Short description of the experiment (optional)")
        date_edit = w.findChild(QDateEdit, "deDate")
        if date_edit is not None:
            date_edit.setDate(QDate.currentDate())
        self._build_video_list()
        self.btn_design = QPushButton("Experimental Design…", w)
        self.btn_design.setProperty("cssClass", "secondary")
        self.btn_design.setMinimumHeight(34)
        self.btn_design.setToolTip("Animal ID, Group, Sex, Dam/Cage, Session, Test and repeated measurements.")
        lay = find(w, QWidget, "cardExperimentInfo").layout()
        if lay is not None:
            lay.insertWidget(max(0, lay.count() - 1), self.btn_design)
        self.btn_design.clicked.connect(self.open_experimental_design)
        self.exp_tiles = {}
        for key in TESTS:
            t = w.findChild(QToolButton, f"expTile_{key}")
            if t is not None:
                self.exp_tiles[key] = t
                t.clicked.connect(lambda _=False, k=key: self.select_test(k, navigate=False))

    def open_experimental_design(self):
        dlg = ExperimentalDesignDialog(self.win, self.experimental_design, TESTS[self.test_key][0],
                                       headers=self.cfg.get("design_columns"))
        if dlg.exec() == QDialog.Accepted:
            self.experimental_design = dlg.values()
            self.cfg["design_columns"] = list(dlg.headers)     # your columns are kept for the next experiment
            save_json(SETTINGS_PATH, self.cfg)
            by_id = {}
            for row in self.experimental_design:
                by_id.setdefault(row["Animal ID"], row)
            self.tbl_subjects.setRowCount(0)
            for idx, (sid, row) in enumerate(by_id.items()):
                self._add_subject_row(sid, row.get("Group", ""), row.get("Sex", ""), "", checked=(idx == 0))
            self.status.showMessage(f"Experimental design saved: {len(self.experimental_design)} record(s).", 5000)

    def select_test(self, key, navigate=True):
        changed = key != self.test_key
        self.test_key = key
        label, emoji, text, _tpl = TESTS[key]
        if key in self.exp_tiles:
            self.exp_tiles[key].setChecked(True)
        find(self.win, QLabel, "lblPreviewImage").setText(emoji)
        find(self.win, QLabel, "lblPreviewTitle").setText(f"{label} Test" if "Test" not in label else label)
        find(self.win, QLabel, "lblPreviewText").setText(text)
        if self.player.cap is not None and changed:
            if not self.canvas.zones or QMessageBox.question(
                    self.win, "Test changed", f"Replace the current zones with the {label} preset?\n"
                    "(You can also use Zone ▸ Auto-detect to fit them to the apparatus.)") == QMessageBox.Yes:
                self.apply_template()
        self._refresh_batch_test_column()
        self._update_kpi_captions()
        if hasattr(self, "cmb_darkbox"):
            self._sync_darkbox()
        if navigate:
            self.go(1)

    def add_video_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self.win, "Select video file(s)", str(Path.home()), VIDEO_FILTER)
        if paths:
            self.load_videos(paths)

    @staticmethod
    def _probe_video(path):
        """{'path','duration','size'} for a video, or None if it cannot be opened."""
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if ok else 0
        size = (int(cap.get(3)), int(cap.get(4)))
        cap.release()
        return {"path": path, "duration": dur, "size": size} if ok else None

    def load_videos(self, paths):
        """Load videos for SINGLE / individual tracking (Experiment page).  These
        stay separate from the Batch-Processing queue."""
        paths = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTS]
        if not paths:
            return
        known = {it["path"] for it in self.exp_videos}
        added = 0
        for p in paths:
            if p in known:
                continue
            info = self._probe_video(p)
            if info is not None:
                self.exp_videos.append(info)
                added += 1
        if self.open_video(paths[0]):
            if added > 1:
                self.status.showMessage(f"{added} videos loaded. Track them one at a time here, or use the video "
                                        "list / the box above the tracking view to switch between them.", 8000)
            self.go(2)

    # --- loaded videos list (Experiment page) + removing videos
    def _build_video_list(self):
        w = self.win
        card = find(w, QFrame, "cardVideoSource")
        lay = card.layout()
        self.lbl_videos = QLabel("Loaded videos")
        self.lbl_videos.setProperty("cssClass", "fieldLabel")
        self.lst_videos = QListWidget()
        self.lst_videos.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lst_videos.setMinimumHeight(90)
        self.lst_videos.setMaximumHeight(190)
        self.lst_videos.setToolTip("Double-click = open in Video Setup.  Delete = remove the selected video(s).")
        self.lst_videos.itemDoubleClicked.connect(
            lambda it: (self.open_video(it.data(Qt.UserRole)) and self.go(2)) if not self.tracking else None)
        self.lst_videos.installEventFilter(self)
        self.lst_videos.setContextMenuPolicy(Qt.CustomContextMenu)
        self.lst_videos.customContextMenuRequested.connect(self._video_list_menu)
        row = QHBoxLayout()
        self.btn_rm_video = _small_button("✕  Remove selected", "Remove the selected video(s) from this experiment "
                                          "(the files on disk are not touched).")
        self.btn_rm_all_videos = _small_button("Remove all", "Remove every loaded video.")
        self.btn_rm_video.clicked.connect(lambda: self.remove_videos(self._selected_list_paths()))
        self.btn_rm_all_videos.clicked.connect(self._remove_all_videos)
        row.addWidget(self.btn_rm_video)
        row.addWidget(self.btn_rm_all_videos)
        row.addStretch(1)
        lay.addWidget(self.lbl_videos)
        lay.addWidget(self.lst_videos)
        lay.addLayout(row)
        self._refresh_video_list()

    def _selected_list_paths(self):
        return [it.data(Qt.UserRole) for it in self.lst_videos.selectedItems()]

    def _video_list_menu(self, pos):
        it = self.lst_videos.itemAt(pos)
        if it is None or self.tracking:
            return
        if not it.isSelected():
            self.lst_videos.clearSelection()
            it.setSelected(True)
        menu = QMenu(self.lst_videos)
        a_open = menu.addAction("Open in Video Setup")
        a_rm = menu.addAction("Remove")
        chosen = menu.exec(self.lst_videos.viewport().mapToGlobal(pos))
        if chosen is a_open:
            if self.open_video(it.data(Qt.UserRole)):
                self.go(2)
        elif chosen is a_rm:
            self.remove_videos(self._selected_list_paths())

    def _refresh_video_list(self):
        if not hasattr(self, "lst_videos"):
            return
        paths = [it["path"] for it in self.exp_videos]
        if self.player.path and self.player.path not in paths:
            paths.insert(0, self.player.path)
        self.lst_videos.clear()
        for p in paths:
            it = next((b for b in self.exp_videos if b["path"] == p), None)
            extra = f"   ·  {fmt_clock(it['duration'])}" if it else ""
            item = QListWidgetItem(("▶  " if p == self.player.path else "     ") + Path(p).name + extra)
            item.setData(Qt.UserRole, p)
            item.setToolTip(p)
            if p == self.player.path:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.lst_videos.addItem(item)
        self.lbl_videos.setText(f"Loaded videos ({len(paths)})")
        self.btn_rm_video.setEnabled(bool(paths))
        self.btn_rm_all_videos.setEnabled(bool(paths))

    def _remove_all_videos(self):
        paths = [self.lst_videos.item(i).data(Qt.UserRole) for i in range(self.lst_videos.count())]
        if paths and QMessageBox.question(self.win, "Remove videos",
                                          f"Remove all {len(paths)} video(s) from this experiment?") == QMessageBox.Yes:
            self.remove_videos(paths)

    def remove_videos(self, paths):
        """Take videos out of the experiment (single-tracking) list; files on disk are untouched.
        The Batch-Processing queue is separate and has its own Remove."""
        paths = [p for p in paths if p]
        if not paths:
            self.status.showMessage("Select a video first.", 4000)
            return
        if self.tracking:
            QMessageBox.information(self.win, "Remove videos", "Stop tracking before removing videos.")
            return
        gone = set(paths)
        self.exp_videos = [it for it in self.exp_videos if it["path"] not in gone]
        if self.player.path in gone:
            if self.exp_videos:
                self.open_video(self.exp_videos[0]["path"])
            else:
                self._unload_video()
        self._refresh_video_combo()
        self.status.showMessage(f"Removed {len(gone)} video(s) from this experiment.", 5000)

    def _unload_video(self):
        self._stop_playback()
        self.canvas.cancel_tool(silent=True)
        if self.crop:
            self.remove_crop(quiet=True)
        self.player.close()
        self.player.total, self.player.pos = 0, 0
        self.raw_frame = self.cur_frame = None
        self.canvas.set_frame(None)
        self.view_track.clear("Load a video on the Experiment page")
        for sl in (self.sl_setup, self.sl_track):
            sl.blockSignals(True)
            sl.setRange(0, 0)
            sl.blockSignals(False)
        self._preview = None
        self._update_time_labels()
        self._update_opbar()

    def open_camera(self):
        if self.tracking:
            QMessageBox.information(self.win, "Live camera", "Wait until tracking has finished (or stop it) first.")
            return
        if self.canvas.tool is not None:
            self.canvas.cancel_tool(silent=True)
        hint = self.le_name.text().strip() if hasattr(self, "le_name") else ""
        autostart, self._camera_return = bool(getattr(self, "_camera_return", False)), False
        dlg = CameraDialog(self.win, self.cfg, hint or (self.current_subject() or ""), ctrl=self,
                           autostart=autostart)
        dlg.exec()
        save_json(SETTINGS_PATH, self.cfg)                 # remembers the folder / resolution / camera
        paths = list(dict.fromkeys(p for p in dlg.result_paths if os.path.exists(p)))
        QTimer.singleShot(0, dlg.deleteLater)
        if dlg.action == "zones" and dlg.zones_path and os.path.exists(dlg.zones_path):
            if paths:
                self.load_videos(paths)
            if self.open_video(dlg.zones_path):
                self.go(2)
                self._camera_return = True
                QMessageBox.information(
                    self.win, "Zones for the live camera",
                    "This is a 3-second clip of your camera.\n\n"
                    "Fit the zones on the apparatus here (⚡ Auto, 🎯 Guided, or drag the preset), and "
                    "crop / mask if needed.\n\n"
                    "Then press 📷 Live tracking from a camera (Tracking page) or Live Camera (Experiment page) "
                    "again — the same camera starts by itself with these zones.")
            return
        if dlg.live_results:
            subject = self.current_subject()
            for vpath, summ in dlg.live_results:
                try:
                    self.add_history(summ, vpath, subject)
                except Exception as exc:
                    print(f"History not updated: {exc}")
        if dlg.action == "results" and dlg.live_results:
            vpath, summ = dlg.live_results[-1]
            todo = [p for p in paths if p != vpath] + [vpath]
            self.load_videos(todo)
            if self.player.path != vpath:
                self.open_video(vpath)
            self.show_results(summ)
            if summ.get("Project_file"):
                self.open_bta(summ["Project_file"], show=False)
            self.go(4)
            q = summ.get("Tracking_quality_percent")
            self.status.showMessage(f"Live trial tracked — {Path(vpath).name}: animal detected in {q:.1f} % of the "
                                    "frames. Check it frame by frame in the Frame Review tab."
                                    if isinstance(q, (int, float)) else "Live trial tracked.", 12000)
            return
        if dlg.action == "analyse" and dlg.trial_path and os.path.exists(dlg.trial_path):
            todo = [p for p in paths if p != dlg.trial_path] + [dlg.trial_path]
            self.load_videos(todo)
            if self.player.path != dlg.trial_path and not self.open_video(dlg.trial_path):
                return
            self.go(3)
            self.status.showMessage(f"Analysing the live trial {Path(dlg.trial_path).name} …", 8000)
            QTimer.singleShot(150, self.start_or_stop_tracking)
            return
        if paths:
            self.load_videos(paths)
            live = {p for p, _s in dlg.live_results}
            if live:
                self.status.showMessage(f"{len(live)} live trial(s) tracked — the results are saved next to the "
                                        "recordings (open one with Export ▸ Open BT analysis file).", 12000)
            else:
                self.status.showMessage(f"{len(paths)} recording(s) saved in {Path(paths[0]).parent} and added "
                                        "to the experiment — track them as usual.", 10000)

    def _add_subject_row(self, sid, group, sex, weight, checked=False):
        r = self.tbl_subjects.rowCount()
        self.tbl_subjects.insertRow(r)
        chk = QTableWidgetItem()
        chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        chk.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.tbl_subjects.setItem(r, 0, chk)
        for c, text in enumerate((sid, group, sex, weight), start=1):
            self.tbl_subjects.setItem(r, c, QTableWidgetItem(text))

    def add_subject(self):
        dlg = SubjectDialog(self.win, f"M{self.tbl_subjects.rowCount() + 1}")
        if dlg.exec() != QDialog.Accepted:
            return
        sid, group, sex, weight = dlg.values()
        if not sid:
            return
        for r in range(self.tbl_subjects.rowCount()):
            if self.tbl_subjects.item(r, 1) and self.tbl_subjects.item(r, 1).text() == sid:
                QMessageBox.warning(self.win, "Subjects", f"Subject '{sid}' already exists.")
                return
        self._add_subject_row(sid, group, sex, weight, checked=self.tbl_subjects.rowCount() == 0)

    def _subject_menu(self, pos):
        row = self.tbl_subjects.rowAt(pos.y())
        if row < 0:
            return
        menu = QMenu(self.tbl_subjects)
        act = menu.addAction("Remove subject")
        if menu.exec(self.tbl_subjects.viewport().mapToGlobal(pos)) is act:
            self.tbl_subjects.removeRow(row)

    def current_subject(self):
        for r in range(self.tbl_subjects.rowCount()):
            it = self.tbl_subjects.item(r, 0)
            if it is not None and it.checkState() == Qt.Checked and self.tbl_subjects.item(r, 1):
                return self.tbl_subjects.item(r, 1).text()
        return ""

    # ------------------------------------------------------------------ video setup page
    @staticmethod
    def _scrollify_card(card, pinned=()):
        """Put a card's content in a scroll area so nothing is ever squeezed or
        clipped (small screens, Windows display scaling, large fonts).  The
        card title stays on top, `pinned` widgets stay visible at the bottom."""
        lay = card.layout()
        if lay is None:
            return None
        items = []
        while lay.count():
            it = lay.itemAt(0)
            wd = it.widget()
            if wd is not None:
                # removeWidget deletes the old layout item: the widget then reports its size changes to its new
                # place (a taken-out item would keep a stale height cache and squeeze widgets that grow later)
                lay.removeWidget(wd)
                items.append(("w", wd))
            elif it.layout() is not None:
                sub = it.layout()
                lay.takeAt(0)
                items.append(("l", sub))
            else:
                lay.takeAt(0)
                items.append(("s", None))
        body = QWidget()
        body.setObjectName(card.objectName() + "Body")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 10, 4)
        bl.setSpacing(lay.spacing() if lay.spacing() >= 0 else 6)
        title = None
        for k, (kind, obj) in enumerate(items):
            if k == 0 and kind == "w" and isinstance(obj, QLabel):
                title = obj
            elif kind == "w" and obj in pinned:
                continue
            elif kind == "s":
                continue
            elif kind == "w":
                bl.addWidget(obj)
            elif kind == "l":
                bl.addLayout(obj)
        bl.addStretch(1)
        scroll = QScrollArea(card)
        scroll.setObjectName(card.objectName() + "Scroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"QScrollArea#{scroll.objectName()} {{ background: transparent; border: none; }} "
                             f"QWidget#{body.objectName()} {{ background: transparent; }}")
        scroll.viewport().setAutoFillBackground(False)
        scroll.setWidget(body)
        # never narrower than the content: the card widens instead of cutting text on the right
        scroll.setMinimumWidth(body.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4)
        if title is not None:
            lay.addWidget(title)
        lay.addWidget(scroll, 1)
        for wd in pinned:
            lay.addWidget(wd)
        return scroll

    @staticmethod
    def _layout_with(root_layout, widget):
        if root_layout is None:
            return None
        for i in range(root_layout.count()):
            it = root_layout.itemAt(i)
            if it.widget() is widget:
                return root_layout
            if it.layout() is not None:
                r = UXController._layout_with(it.layout(), widget)
                if r is not None:
                    return r
        return None

    def _bind_setup(self):
        w = self.win
        page = find(w, QWidget, "pageVideoSetup")
        label = find(w, QLabel, "videoFrameSetup")
        vlay = self._layout_with(page.layout(), label)
        self.canvas = ZoneCanvas(label.parentWidget())
        self.canvas.covered_fn = lambda: self.covered_names()
        self.canvas.setObjectName("zoneCanvas")
        self.canvas.setMinimumSize(label.minimumSize())
        vlay.replaceWidget(label, self.canvas)
        label.hide()
        self.canvas.zonesEdited.connect(self._on_zones_edited)
        self.canvas.selectionChanged.connect(self._on_selection_changed)
        self.canvas.drawingFinished.connect(self._on_drawing_finished)
        self.canvas.draftChanged.connect(self._update_opbar)
        self.canvas.drawingCancelled.connect(self._drawing_cancelled)
        self.canvas.zoneRenameRequested.connect(self.rename_zone)
        self.canvas.contextRequested.connect(self._canvas_menu)
        self.canvas.hoverInfo.connect(lambda t: self.status.showMessage(t, 3000))

        # ---- contextual instruction / action bar above the video
        self.opbar = QFrame()
        self.opbar.setStyleSheet("QFrame#opbar{background:#eef3fb;border:1px solid #d5e2f5;border-radius:8px;}")
        self.opbar.setObjectName("opbar")
        ob = QHBoxLayout(self.opbar)
        ob.setContentsMargins(10, 4, 6, 4)
        ob.setSpacing(6)
        self.op_text = QLabel("")
        self.op_text.setWordWrap(True)
        self.op_text.setStyleSheet("color:#1e3a8a;")
        ob.addWidget(self.op_text, 1)
        self.op_btns = {}
        for key, text, tip, css in (
                ("finish", "✓ Finish", "Close the shape (Enter / double-click)", "primary"),
                ("undo_pt", "↶ Point", "Remove the last point (right-click / Backspace)", "secondary"),
                ("cancel", "✕ Cancel", "Cancel drawing (Esc)", "secondary"),
                ("rename", "Rename", "Rename the selected zone", "secondary"),
                ("dup", "Duplicate", "Duplicate the selected zone(s) (Ctrl+D)", "secondary"),
                ("delete", "Delete", "Delete the selected zone(s) (Delete)", "danger"),
                ("undo", "↶", "Undo (Ctrl+Z)", "secondary"),
                ("redo", "↷", "Redo (Ctrl+Y)", "secondary"),
                ("fit", "⤢", "Reset zoom (0) — mouse wheel zooms, middle-drag pans", "secondary")):
            b = _small_button(text, tip, css)
            ob.addWidget(b)
            self.op_btns[key] = b
        self.op_btns["finish"].clicked.connect(lambda: self.canvas.finish_tool() or
                                               self.status.showMessage("Need at least 3 points.", 3000))
        self.op_btns["undo_pt"].clicked.connect(self.canvas.undo_point)
        self.op_btns["cancel"].clicked.connect(lambda: self.canvas.cancel_tool())
        self.op_btns["rename"].clicked.connect(lambda: self.rename_zone(self.canvas.selected[0])
                                               if self.canvas.selected else None)
        self.op_btns["dup"].clicked.connect(self.canvas.duplicate_selected)
        self.op_btns["delete"].clicked.connect(lambda: self.canvas.delete_zones(list(self.canvas.selected)))
        self.op_btns["undo"].clicked.connect(self.canvas.undo)
        self.op_btns["redo"].clicked.connect(self.canvas.redo)
        self.op_btns["fit"].clicked.connect(self.canvas.reset_view)
        vlay.insertWidget(0, self.opbar)

        # ---- playback
        self.btn_play_setup = find(w, QToolButton, "btnPlaySetup")
        self.sl_setup = find(w, QSlider, "sliderPositionSetup")
        self.lbl_time_setup = find(w, QLabel, "lblTimeSetup")
        self.lbl_time_setup.setStyleSheet("color: #000000; font-weight: 600;")
        self.btn_play_setup.setIcon(self.icon_play)
        self.btn_play_setup.clicked.connect(self.toggle_play)
        find(w, QToolButton, "btnPrevFrame").clicked.connect(lambda: self.step(-1))
        find(w, QToolButton, "btnNextFrame").clicked.connect(lambda: self.step(1))
        self.sl_setup.valueChanged.connect(self._slider_moved)

        # ---- arena + calibration
        self.spin_len = find(w, QSpinBox, "spinLength")
        self.spin_wid = find(w, QSpinBox, "spinWidth")
        self.spin_ppc = find(w, QDoubleSpinBox, "spinPixelPerCm")
        self.spin_ppc.setDecimals(2)
        self.view_arena = FrameView(find(w, QLabel, "lblArenaPreview"), self)
        self.cmb_shape = find(w, QComboBox, "cmbArenaShape")
        self.cmb_shape.currentTextChanged.connect(self._shape_changed)
        self.spin_len.valueChanged.connect(self._len_changed)
        self.spin_len.valueChanged.connect(lambda _=0: self._refresh_arena())
        self.spin_wid.valueChanged.connect(lambda _=0: self._refresh_arena())
        find(w, QPushButton, "btnSetScale").clicked.connect(self.begin_scale)
        self.spin_ppc.setToolTip("Pixels per centimetre. 'Set Scale' measures it on the frame (click both ends "
                                 "of a known distance).")

        # ---- zone panel (replaces the three fixed rows of the design)
        self._build_zone_panel()

        # ---- tool column
        self.tool_buttons = {n: find(w, QToolButton, n) for n in ("toolCrop", "toolZone", "toolMask", "toolCalibrate")}
        for b in self.tool_buttons.values():
            b.setAutoExclusive(False)
            b.setChecked(False)
        self.tool_buttons["toolCrop"].clicked.connect(self.tool_crop)
        self.tool_buttons["toolZone"].clicked.connect(self.tool_zone)
        self.tool_buttons["toolMask"].clicked.connect(self.tool_mask)
        self.tool_buttons["toolCalibrate"].clicked.connect(self.tool_calibrate)
        self.tool_buttons["toolCrop"].setToolTip("Crop / straighten the arena: click its 4 corners (zones are kept)")
        self.tool_buttons["toolZone"].setToolTip("Draw, trace, auto-detect, load or reset zones")
        self.tool_buttons["toolMask"].setToolTip("Areas the detector must ignore; auto-mask outside the zones")
        self.tool_buttons["toolCalibrate"].setToolTip("Measure a known distance to set pixels per cm")
        self._update_opbar()

    def _build_zone_panel(self):
        w = self.win
        card = find(w, QFrame, "cardArenaSettings")
        lay = card.layout()
        for n in ("dotCenterZone", "lblCenterZone", "btnEditCenterZone", "btnDeleteCenterZone",
                  "dotPeripheryZone", "lblPeripheryZone", "btnEditPeripheryZone", "btnDeletePeripheryZone",
                  "dotCustomZone", "lblCustomZone", "btnEditCustomZone", "btnDeleteCustomZone"):
            wd = w.findChild(QWidget, n)
            if wd is not None:
                wd.hide()
        lbl_zones = find(w, QLabel, "lblZones")
        idx = lay.indexOf(lbl_zones)
        panel = QWidget()
        v = QVBoxLayout(panel)
        v.setContentsMargins(0, 2, 0, 2)
        v.setSpacing(6)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.btn_zone_draw = _small_button("＋ Draw", "Draw a new zone")
        m = QMenu(self.btn_zone_draw)
        for text, tool in SHAPE_MENU:
            m.addAction(text, lambda t=tool: self.begin_draw(t, "zone"))
        self.btn_zone_draw.setMenu(m)
        self.btn_zone_auto = _small_button("⚡ Auto", "Detect the apparatus and fit the zones automatically")
        self.btn_zone_auto.clicked.connect(self.auto_detect_zones)
        self.btn_zone_guided = _small_button("🎯 Guided", "Guided fit: click the arm ends (EPM / Y / T maze) or the "
                                             "floor corners (boxes) — the zones are fitted exactly to your "
                                             "clicks and the real arm width is measured on the video.")
        self.btn_zone_guided.clicked.connect(self._guided_clicked)
        self.btn_zone_preset = _small_button("▦ Preset", "Apply the preset zones of the selected test, then move / "
                                             "resize / rotate them onto the apparatus")
        self.btn_zone_preset.clicked.connect(lambda: self.apply_template(ask=True))
        self.btn_zone_all = _small_button("⛶ All", "Select every zone to move / resize / rotate the whole layout")
        self.btn_zone_all.clicked.connect(lambda: self.canvas.select(list(self.canvas.zones)))
        self.btn_zone_fragment = _small_button(
            "✂ Fragment", "Like PowerPoint ▸ Merge Shapes ▸ Fragment: draw overlapping shapes for the apparatus, "
            "then cut them into every separate piece (e.g. two crossing bars → 4 arms + centre). The pieces get "
            "temporary names A, B, C… — rename them afterwards. Uses the selected zones (2 or more), otherwise all "
            "zones.")
        self.btn_zone_fragment.clicked.connect(self.fragment_zones)
        grid = QGridLayout()
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        for i, b in enumerate((self.btn_zone_draw, self.btn_zone_auto, self.btn_zone_guided, self.btn_zone_preset,
                               self.btn_zone_all, self.btn_zone_fragment)):
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            grid.addWidget(b, i // 3, i % 3)
        bar.addLayout(grid, 1)
        v.addLayout(bar)
        self.zone_scroll = QScrollArea()
        self.zone_scroll.setWidgetResizable(True)
        self.zone_scroll.setFrameShape(QFrame.NoFrame)
        self.zone_scroll.setMinimumHeight(96)
        self.zone_scroll.setMaximumHeight(210)
        self.zone_list = QWidget()
        self.zone_list_lay = QVBoxLayout(self.zone_list)
        self.zone_list_lay.setContentsMargins(0, 0, 4, 0)
        self.zone_list_lay.setSpacing(2)
        self.zone_scroll.setWidget(self.zone_list)
        v.addWidget(self.zone_scroll)
        self.chk_automask = QCheckBox("Auto-mask outside zones")
        self.chk_automask.setChecked(True)
        self.chk_automask.setToolTip("Auto-mask: once the zones are set, reflections, hands or shadows outside them "
                                     "can never be taken for the animal (dimmed area on the video).")
        self.chk_automask.toggled.connect(self._automask_toggled)
        self.chk_lock = QCheckBox("Lock zones")
        self.chk_lock.setToolTip("Protect the zones from accidental edits while you scrub the video.")
        self.chk_lock.toggled.connect(self._lock_toggled)
        row = QHBoxLayout()
        row.addWidget(self.chk_automask)
        row.addStretch()
        row.addWidget(self.chk_lock)
        v.addLayout(row)
        lay.insertWidget(idx + 1, panel)
        self._zone_rows = {}

    # --- zone list
    def _refresh_zone_list(self):
        lay = self.zone_list_lay
        while lay.count():
            it = lay.takeAt(0)
            if it.widget() is not None:
                it.widget().hide()
                it.widget().deleteLater()
        self._zone_rows = {}
        if not self.canvas.zones:
            lab = QLabel("No zones yet — use ⚡ Auto, 🎯 Guided, ▦ Preset or ＋ Draw.")
            lab.setProperty("cssClass", "muted")
            lab.setWordWrap(True)
            lay.addWidget(lab)
            lay.addStretch()
            return
        for name, pts in self.canvas.zones.items():
            roww = QFrame()
            roww.setObjectName("zoneRow")
            h = QHBoxLayout(roww)
            h.setContentsMargins(4, 1, 0, 1)
            h.setSpacing(6)
            dot = QLabel()
            dot.setFixedSize(14, 14)
            dot.setStyleSheet(f"background:{self.canvas.color_of(name).name()};border-radius:7px;")
            h.addWidget(dot)
            lab = QLabel(name)
            lab.setToolTip(f"{len(pts)} points · area {poly_area(pts):.0f} px² — click to select, "
                           "double-click to rename")
            lab.setCursor(Qt.PointingHandCursor)
            lab.mousePressEvent = (lambda e, n=name: self._zone_row_clicked(n, e))
            lab.mouseDoubleClickEvent = (lambda _e, n=name: self.rename_zone(n))
            h.addWidget(lab, 1)
            for text, tip, fn in (("✎", "Edit (select)", lambda _=False, n=name: self._edit_zone(n)),
                                  ("🗑", "Delete", lambda _=False, n=name: self.canvas.delete_zones([n]))):
                b = QToolButton()
                b.setText(text)
                b.setProperty("cssClass", "iconBtn")
                b.setToolTip(tip)
                b.clicked.connect(fn)
                h.addWidget(b)
            lay.addWidget(roww)
            self._zone_rows[name] = roww
        lay.addStretch()
        self._mark_selected_rows()

    def _zone_row_clicked(self, name, ev):
        if ev.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier):
            sel = list(self.canvas.selected)
            self.canvas.select([n for n in sel if n != name] if name in sel else sel + [name])
        else:
            self.canvas.select([name])

    def _mark_selected_rows(self):
        for n, roww in self._zone_rows.items():
            roww.setStyleSheet("QFrame#zoneRow{background:#eaf2ff;border-radius:6px;}" if n in self.canvas.selected
                               else "QFrame#zoneRow{background:transparent;}")

    def _edit_zone(self, name):
        if self.chk_lock.isChecked():
            self.chk_lock.setChecked(False)
        self.canvas.select([name])
        self.go(2)

    def _on_zones_edited(self):
        self._refresh_zone_list()
        self._refresh_arena()
        self._preview = None                  # zones / masks changed -> detection preview is stale
        self._update_opbar()
        self._refresh_batch_test_column()     # batch rows that follow the Video-Setup zones

    def _on_selection_changed(self):
        self._mark_selected_rows()
        self._update_opbar()

    def _automask_toggled(self, on):
        self.canvas.auto_mask = bool(on)
        self.canvas.update()
        self._preview = None
        if hasattr(self, "act_automask"):
            self.act_automask.setChecked(bool(on))

    def _lock_toggled(self, on):
        self.canvas.locked = bool(on)
        if on:
            self.canvas.cancel_tool(silent=True)
        self.canvas.update()
        self._update_opbar()

    def _uncheck_tools(self):
        for b in getattr(self, "tool_buttons", {}).values():
            b.setChecked(False)

    def _update_opbar(self):
        c = self.canvas
        drawing = c.tool is not None
        sel = list(c.selected)
        for k in ("finish", "undo_pt", "cancel"):
            self.op_btns[k].setVisible(drawing)
        self.op_btns["finish"].setVisible(drawing and not (c.tool == "points" and c.purpose == "scale"))
        for k in ("rename", "dup", "delete"):
            self.op_btns[k].setVisible(not drawing and bool(sel) and not c.locked)
        self.op_btns["rename"].setVisible(not drawing and len(sel) == 1 and not c.locked)
        if self.player.cap is None:
            text = "Load a video (Experiment page or Ctrl+O), then draw or auto-detect the zones."
        elif drawing:
            text = c.current_banner() or "Drawing…"
        elif c.locked:
            text = "🔒 Zones are locked. Untick 'Lock zones' to edit them."
        elif len(sel) == 1:
            text = (f"<b>{sel[0]}</b> — drag to move · drag yellow points to reshape · double-click edge = add point · "
                    "right-click point = delete · handles resize / rotate (Shift = proportional / 15°) · Ctrl+arrows nudge")
        elif len(sel) > 1:
            text = (f"<b>{len(sel)} zones selected</b> — move / resize / rotate them together to fit the apparatus "
                    "(Shift keeps proportions). Esc to deselect.")
        elif c.zones:
            text = ("Click a zone to edit it (⛶ All = the whole layout). Wheel zooms, middle-drag pans. "
                    "Right-click for more.")
        else:
            text = ("No zones yet: ⚡ Auto detects them, 🎯 Guided fits them to a few clicks, ▦ Preset adds the "
                    "test template, ＋ Draw lets you draw your own.")
        self.op_text.setText(text)

    # --- video
    def open_video(self, path):
        self._stop_playback()
        self.canvas.cancel_tool(silent=True)
        prev_size = (self.player.w, self.player.h) if self.player.cap is not None else None
        prev_arena = self.arena_size if self.player.cap is not None else None
        if not self.player.open(path):
            QMessageBox.critical(self.win, "Video", f"Could not open video:\n{path}")
            return False
        new_size = (self.player.w, self.player.h)
        if self.crop and prev_size != new_size:
            self.remove_crop(quiet=True)
            prev_arena = prev_size
        frame = self.player.seek(0)
        self.raw_frame, self.cur_frame = frame, self._warp(frame)
        for sl in (self.sl_setup, self.sl_track):
            sl.blockSignals(True)
            sl.setRange(0, max(self.player.total - 1, 0))
            sl.setValue(0)
            sl.blockSignals(False)
        if not self._scale_auto_done and self.spin_len.value() > 0:
            self.spin_ppc.setValue(round(self.arena_size[0] / self.spin_len.value(), 2))
            self._scale_auto_done = True
        self.canvas.set_frame(self.cur_frame)
        if not self.canvas.zones:
            self.apply_template()
        elif prev_arena and prev_arena != self.arena_size:
            sx, sy = self.arena_size[0] / prev_arena[0], self.arena_size[1] / prev_arena[1]
            self.canvas.transform_points(lambda p: (p[0] * sx, p[1] * sy))
            self.status.showMessage("New video has a different size — zones were rescaled; check their position.", 8000)
        self.spin_start.setRange(0.0, max(0.0, self.player.duration))
        self.spin_end.setRange(0.0, max(0.0, self.player.duration))
        self.spin_start.setValue(0.0)
        self.spin_end.setValue(max(0.0, self.player.duration))
        self._preview = None
        self._update_time_labels()
        self._refresh_video_combo()
        self._refresh_arena()
        self._update_opbar()
        self.status.showMessage(f"Loaded {Path(path).name}  —  {self.player.w}×{self.player.h}, "
                                f"{self.player.fps:.1f} fps, {fmt_clock(self.player.duration)}", 8000)
        return True

    @property
    def arena_size(self):
        if self.crop:
            return self.crop["w"], self.crop["h"]
        return self.player.w, self.player.h

    def _warp(self, frame):
        if self.crop and frame is not None:
            return cv2.warpPerspective(frame, self.crop["matrix"], (self.crop["w"], self.crop["h"]))
        return frame

    def apply_template(self, ask=False):
        if self.player.cap is None:
            self.status.showMessage("Load a video first.", 4000)
            return
        if getattr(self, "chamber_ui", None) is not None and self.chamber_ui.enabled:
            self.chamber_ui.apply_preset(ask=ask)          # the preset fitted into the chamber(s)
            return
        if ask and self.canvas.zones and QMessageBox.question(
                self.win, "Preset zones", f"Replace the current zones with the {TESTS[self.test_key][0]} preset?"
        ) != QMessageBox.Yes:
            return
        try:
            zones = default_zones(self.test_key, *self.arena_size)
        except Exception as exc:
            zones = {}
            self.status.showMessage(f"Could not build the zone template: {exc}", 6000)
        self.canvas.colors.clear()
        self.canvas._color_i = 0
        self.canvas.set_zones(zones, record=bool(self.canvas.zones), select_all=bool(zones))
        if zones:
            self.status.showMessage("Preset applied — drag it onto the apparatus; handles resize / rotate the whole "
                                    "layout, points reshape single zones. ⚡ Auto can fit it for you.", 10000)

    # --- frames / playback
    def _refresh_views(self):
        if self.cur_frame is None:
            return
        page = self.stack.currentIndex()
        if page == 2:
            self.canvas.set_frame(self.cur_frame)
        elif page == 3 and not self.tracking:
            self._show_tracking_frame()

    def _show_frame(self, frame):
        if frame is None:
            return
        self.raw_frame, self.cur_frame = frame, self._warp(frame)
        self._refresh_views()
        idx = self.player.current
        if not self.tracking:
            for sl in (self.sl_setup, self.sl_track):
                sl.blockSignals(True)
                sl.setValue(idx)
                sl.blockSignals(False)
        self._update_time_labels()

    def _update_time_labels(self):
        t = self.player.current / self.player.fps if self.player.fps else 0
        d = self.player.duration
        self.lbl_time_setup.setText(f"{fmt_clock(t)} / {fmt_clock(d)}")
        if not self.tracking:
            self.lbl_time_track.setText(fmt_track_pos(t, self.player.current, d, max(0, self.player.total - 1)))

    def _slider_moved(self, value):
        if self.player.cap is None or self.tracking:
            return
        self._show_frame(self.player.seek(value))

    def toggle_play(self):
        if self.player.cap is None or self.tracking or self.stack.currentIndex() not in (2, 3):
            return
        if self.canvas.tool is not None:
            return
        if self.play_timer.isActive():
            self._stop_playback()
        else:
            if self.player.pos >= self.player.total - 1:
                self.player.seek(0)
            self.play_timer.start(max(int(1000 / self.player.fps), 10))
            for b in (self.btn_play_setup, self.btn_play_track):
                b.setIcon(self.icon_pause)

    def _stop_playback(self):
        self.play_timer.stop()
        if hasattr(self, "btn_play_track"):
            for b in (self.btn_play_setup, self.btn_play_track):
                b.setIcon(self.icon_play)

    def _play_tick(self):
        frame = self.player.next()
        if frame is None:
            self._stop_playback()
            return
        self._show_frame(frame)
        if (self._preview is not None and self.stack.currentIndex() == 3
                and self.player.current >= int(round(float(self.spin_end.value()) * (self.player.fps or 30.0)))):
            self._stop_playback()                 # detection preview: stop at the End time

    def step(self, delta):
        if self.player.cap is None or self.tracking or self.stack.currentIndex() not in (2, 3):
            return
        self._stop_playback()
        self._show_frame(self.player.seek(self.player.current + delta))

    # --- arena settings
    def _shape_changed(self, text):
        locked = text in ("Square", "Circle")
        self.spin_wid.setEnabled(not locked)
        if locked:
            self.spin_wid.setValue(self.spin_len.value())
        self._refresh_arena()

    def _len_changed(self, v):
        if self.cmb_shape.currentText() in ("Square", "Circle"):
            self.spin_wid.setValue(v)

    def _refresh_arena(self):
        if self.player.cap is None:
            return
        w, h = self.arena_size
        img = np.full((h, w, 3), 255, np.uint8)
        ov = img.copy()
        for name, pts in self.canvas.zones.items():
            c = self.canvas.color_of(name)
            cv2.fillPoly(ov, [np.round(np.asarray(pts)).astype(np.int32)], (c.blue(), c.green(), c.red()))
        img = cv2.addWeighted(ov, 0.55, img, 0.45, 0)
        if self.cmb_shape.currentText() == "Circle":
            cv2.ellipse(img, (w // 2, h // 2), (w // 2 - 2, h // 2 - 2), 0, 0, 360, (90, 90, 90), max(2, w // 120))
        else:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (90, 90, 90), max(2, w // 120))
        self.view_arena.set_bgr(img)

    def _need_video(self):
        if self.player.cap is None:
            self.status.showMessage("Load a video first (Experiment page or Ctrl+O).", 5000)
            return False
        return True

    # --- tool column menus
    def _menu_at(self, button, menu):
        chosen = menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        button.setChecked(False)
        return chosen

    def tool_crop(self):
        btn = self.tool_buttons["toolCrop"]
        if not self._need_video():
            btn.setChecked(False)
            return
        menu = QMenu(self.win)
        a_new = menu.addAction("Crop / straighten arena — click its 4 corners")
        a_rm = menu.addAction("Remove crop (use full frame)")
        a_rm.setEnabled(bool(self.crop))
        menu.addSeparator()
        a_view = menu.addAction("Reset zoom")
        chosen = self._menu_at(btn, menu)
        if chosen is a_new:
            self.begin_crop()
        elif chosen is a_rm:
            self.remove_crop()
        elif chosen is a_view:
            self.canvas.reset_view()

    def tool_zone(self):
        btn = self.tool_buttons["toolZone"]
        if not self._need_video():
            btn.setChecked(False)
            return
        menu = QMenu(self.win)
        sub = menu.addMenu("＋ Draw a zone")
        for text, tool in SHAPE_MENU:
            sub.addAction(text, lambda t=tool: self.begin_draw(t, "zone"))
        menu.addAction("⚡ Auto-detect zones from the video", self.auto_detect_zones)
        menu.addAction("🎯 Guided fit — click the arm ends / floor corners", self._guided_clicked)
        sub2 = menu.addMenu("✎ Trace the apparatus outline → zones")
        sub2.addAction("Polygon — click around the outline", lambda: self.begin_draw("polygon", "outline"))
        sub2.addAction("Freehand — trace the outline", lambda: self.begin_draw("freehand", "outline"))
        menu.addAction(f"▦ Apply {TESTS[self.test_key][0]} preset", lambda: self.apply_template(ask=True))
        menu.addSeparator()
        menu.addAction("⛶ Edit whole layout (select all zones)", lambda: self.canvas.select(list(self.canvas.zones)))
        menu.addAction("✂ Fragment overlapping shapes → pieces A, B, C…", self.fragment_zones)
        menu.addAction("⊕ Merge the selected zones into one", self.merge_zones)
        menu.addAction("Rename zones…", self.rename_all_zones)
        menu.addSeparator()
        menu.addAction("💾 Save zone layout…", self.save_layout)
        menu.addAction("📂 Load zone layout…", self.load_layout)
        menu.addSeparator()
        a_clear = menu.addAction("Clear all zones")
        chosen = self._menu_at(btn, menu)
        if chosen is a_clear and self.canvas.zones:
            self.canvas.set_zones({})

    def tool_mask(self):
        btn = self.tool_buttons["toolMask"]
        if not self._need_video():
            btn.setChecked(False)
            return
        menu = QMenu(self.win)
        menu.addAction("Mask polygon — click points", lambda: self.begin_draw("polygon", "mask"))
        menu.addAction("Mask freehand — trace", lambda: self.begin_draw("freehand", "mask"))
        menu.addAction("Mask rectangle — drag", lambda: self.begin_draw("rect", "mask"))
        menu.addSeparator()
        self.act_automask = menu.addAction("Auto-mask everything outside the zones")
        self.act_automask.setCheckable(True)
        self.act_automask.setChecked(self.chk_automask.isChecked())
        self.act_automask.toggled.connect(self.chk_automask.setChecked)
        menu.addSeparator()
        a_clear = menu.addAction("Clear all masks")
        a_clear.setEnabled(bool(self.canvas.masks))
        chosen = self._menu_at(btn, menu)
        if chosen is a_clear:
            self.canvas.push_undo()
            self.canvas.masks = []
            self.canvas.update()
            self._on_zones_edited()

    def tool_calibrate(self):
        btn = self.tool_buttons["toolCalibrate"]
        if not self._need_video():
            btn.setChecked(False)
            return
        menu = QMenu(self.win)
        a_measure = menu.addAction("Measure a known distance (2 clicks)")
        a_arena = menu.addAction(f"Use arena length ({self.spin_len.value()} cm) = frame / crop width")
        chosen = self._menu_at(btn, menu)
        if chosen is a_measure:
            self.begin_scale()
        elif chosen is a_arena and self.spin_len.value() > 0:
            self.spin_ppc.setValue(round(self.arena_size[0] / self.spin_len.value(), 2))
            self.status.showMessage(f"Scale set to {self.spin_ppc.value():.2f} px/cm.", 5000)

    # --- learned set-ups ("learn my layouts")
    def _learned(self):
        from tracking.layout_library import library
        return library(str(LEARNED_DIR))

    def _match_learned(self, img, test_key=None):
        """(zones, entry, message) of a learned set-up found in `img`, or None."""
        if not self.cfg.get("learn_layouts", True):
            return None
        try:
            from tracking.layout_library import describe, zone_overlap_ok
            lib = self._learned()
            if not len(lib):
                return None
            tk = test_key or self.test_key
            zones, entry, n_in = lib.match(img, tk, any_test=(tk == "custom"))
            if not zones or not zone_overlap_ok(zones, img.shape[1], img.shape[0]):
                return None
            return zones, entry, describe(entry, n_in)
        except Exception as exc:
            print(f"Learned layouts skipped: {exc}")
            return None

    def _learn_layout(self, img, zones, video_name, test_key=None, covered=None):
        """Remember this set-up (in the background; never slows the work down)."""
        if not self.cfg.get("learn_layouts", True) or img is None or not zones:
            return
        tk = test_key or self.test_key
        zz = {k: [tuple(map(float, p)) for p in v] for k, v in zones.items() if len(v) >= 3}
        cov = sorted(covered if covered is not None else self.covered_names(zz))
        img = img.copy()

        def work():
            try:
                with _LEARN_LOCK:
                    self._learned().learn(img, tk, zz, video_name, covered=cov)
            except Exception as exc:
                print(f"Layout not learned: {exc}")
        threading.Thread(target=work, name="bt-learn", daemon=True).start()

    # --- covered zones (the animal is hidden inside: lid of the dark box, a shelter...)
    def covered_names(self, zones=None):
        """Covered zones among `zones` (default: the Video-Setup zones)."""
        zones = self.canvas.zones if zones is None else zones
        if self.covered_override is not None:
            return {n for n in zones if n in self.covered_override}
        if self.test_key == "light_dark" and self.cfg.get("ldb_dark_covered", True):
            return {n for n in zones if any(w in str(n).lower() for w in ("dark", "black"))}
        return set()

    def add_covered_action(self, menu, name, canvas=None):
        canvas = canvas or self.canvas
        act = menu.addAction("Covered zone — the animal is hidden inside (e.g. lid of the dark box)")
        act.setCheckable(True)
        act.setChecked(name in self.covered_names(canvas.zones))
        act.setToolTip("Nothing is tracked inside a covered zone. When the animal goes out of sight next to it, "
                       "that time counts as time in this zone.")
        act.toggled.connect(lambda on, n=name, c=canvas: self.set_covered(n, on, c))

    def set_covered(self, name, on, canvas=None):
        canvas = canvas or self.canvas
        cur = set(self.covered_names(canvas.zones))
        if self.covered_override is not None:
            cur |= set(self.covered_override)
        (cur.add if on else cur.discard)(name)
        self.covered_override = cur
        canvas.update()
        self.canvas.update()
        self._preview = None
        self.status.showMessage(f"'{name}' is {'now a covered zone (animal hidden inside)' if on else 'no longer covered'}.",
                                6000)
        if hasattr(self, "cmb_darkbox"):
            self._sync_darkbox()

    def _canvas_menu(self, name, gpos):
        if self.player.cap is None:
            return
        menu = QMenu(self.win)
        if name and not self.canvas.locked:
            menu.addAction(f"Rename '{name}'…", lambda: self.rename_zone(name))
            menu.addAction("Duplicate", self.canvas.duplicate_selected)
            menu.addAction("Delete", lambda: self.canvas.delete_zones(list(self.canvas.selected)))
            self.add_covered_action(menu, name)
            menu.addSeparator()
        sub = menu.addMenu("＋ Draw a zone")
        for text, tool in tuple((t.split(" — ")[0], k) for t, k in SHAPE_MENU):
            sub.addAction(text, lambda t=tool: self.begin_draw(t, "zone"))
        menu.addAction("⚡ Auto-detect zones", self.auto_detect_zones)
        menu.addAction("🎯 Guided fit (click arm ends / corners)", self._guided_clicked)
        menu.addAction("⛶ Select all zones", lambda: self.canvas.select(list(self.canvas.zones)))
        if len(self.canvas.zones) >= 2 and not self.canvas.locked:
            menu.addAction("✂ Fragment → pieces A, B, C…", self.fragment_zones)
            if len(self.canvas.selected) >= 2:
                menu.addAction("⊕ Merge selected zones", self.merge_zones)
            menu.addAction("Rename zones…", self.rename_all_zones)
        menu.addSeparator()
        a_lock = menu.addAction("Lock zones")
        a_lock.setCheckable(True)
        a_lock.setChecked(self.chk_lock.isChecked())
        a_lock.toggled.connect(self.chk_lock.setChecked)
        menu.addAction("Reset zoom", self.canvas.reset_view)
        menu.exec(gpos)

    # --- drawing
    def begin_draw(self, tool, purpose):
        if not self._need_video():
            return
        if self.chk_lock.isChecked():
            self.chk_lock.setChecked(False)
        self._stop_playback()
        if self.stack.currentIndex() != 2:
            self.go(2)
        what = {"zone": "zone", "mask": "mask area", "outline": "apparatus outline"}[purpose]
        tips = {"polygon": f"Click the corners of the {what}. Close it by clicking the first point, double-click or "
                           "Enter. Right-click / Backspace removes the last point, Esc cancels.",
                "freehand": f"Press and drag around the {what}, release to finish. Esc cancels.",
                "rect": f"Drag a rectangle for the {what} (Shift = square). Esc cancels.",
                "square": f"Drag a square for the {what}. Esc cancels.",
                "circle": f"Drag a circle for the {what}. Esc cancels.",
                "triangle": f"Drag a triangle for the {what} (Shift = equilateral). Esc cancels.",
                "ellipse": f"Drag an ellipse for the {what} (Shift = circle). Esc cancels."}
        self.canvas.select([])
        self.canvas.start_tool(tool, purpose, banner=tips[tool])
        btn = {"zone": "toolZone", "outline": "toolZone", "mask": "toolMask"}[purpose]
        self.tool_buttons[btn].setChecked(True)
        self._update_opbar()

    def begin_crop(self):
        if not self._need_video():
            return
        self._stop_playback()
        self._crop_backup = None
        if self.crop:
            # show the raw frame; keep zones in raw space while picking corners
            self._crop_backup = list(self.crop["corners"])
            self.remove_crop(quiet=True)
        self.canvas.start_tool("points", "crop", max_points=4,
                               banner="Click the 4 corners of the arena floor (any order). Drag a corner to adjust, "
                                      "then Enter / ✓ Finish. The view is straightened; zones are carried over.")
        self.tool_buttons["toolCrop"].setChecked(True)
        self._update_opbar()

    def begin_scale(self):
        if not self._need_video():
            return
        self._stop_playback()
        if self.stack.currentIndex() != 2:
            self.go(2)
        self.canvas.start_tool("points", "scale", max_points=2,
                               banner="Click both ends of a distance you know (e.g. one wall of the arena).")
        self.tool_buttons["toolCalibrate"].setChecked(True)
        self._update_opbar()

    def _drawing_cancelled(self):
        self._uncheck_tools()
        backup = getattr(self, "_crop_backup", None)
        if backup and not self.crop:
            self._crop_backup = None
            self._apply_crop(backup)
        self._update_opbar()

    def _on_drawing_finished(self, purpose, pts):
        self._uncheck_tools()
        if purpose == "crop":
            self._crop_backup = None
        if purpose == "chamber":
            self.chamber_ui.rect_drawn(pts)
        if purpose == "zone":
            name = self._ask_zone_name(title="New zone")
            if name:
                name = self.chamber_ui.zone_name(name, pts)
                self.canvas.add_zone(name, pts)
                self.status.showMessage(f"Zone '{name}' added — drag its points to refine it.", 5000)
        elif purpose == "mask":
            self.canvas.push_undo()
            self.canvas.masks.append([(float(x), float(y)) for x, y in pts])
            self.canvas.update()
            self._on_zones_edited()
            self.status.showMessage("Mask added — the detector ignores this area.", 4000)
        elif purpose == "outline":
            self._zones_from_outline(pts)
        elif purpose == "guided":
            self._apply_guided(pts)
        elif purpose == "crop":
            self._apply_crop(pts)
        elif purpose == "scale":
            self._apply_scale(pts)
        self._update_opbar()

    def _ask_zone_name(self, default=None, title="Zone name", exclude=()):
        existing = [n for n in self.canvas.zones if n not in exclude]
        sug = zone_suggestions(self.test_key, existing)
        if default:
            sug = [default] + [s for s in sug if s != default]
        free = [s for s in sug if s not in existing]
        items = free or [unique_name("Zone", existing)]
        items += [s for s in sug if s not in items]
        while True:
            name, ok = QInputDialog.getItem(self.win, title, "Zone name (pick or type):", items, 0, True)
            if not ok:
                return None
            name = name.strip()
            if not name:
                continue
            if name in existing:
                QMessageBox.warning(self.win, title, f"A zone called '{name}' already exists.")
                continue
            return name

    def rename_zone(self, name):
        if name not in self.canvas.zones:
            return
        new = self._ask_zone_name(default=name, title=f"Rename '{name}'", exclude=(name,))
        if new and new != name:
            self.canvas.rename_zone(name, new)
            if self.covered_override is not None and name in self.covered_override:
                self.covered_override.discard(name)
                self.covered_override.add(new)

    def rename_all_zones(self):
        if not self.canvas.zones:
            return
        dlg = ZoneNamesDialog(self.win, list(self.canvas.zones), zone_suggestions(self.test_key),
                              epm_swap=self.test_key == "epm")
        if dlg.exec() == QDialog.Accepted:
            self._apply_name_mapping(dlg.mapping())

    # --- Fragment / Merge (PowerPoint 'Merge Shapes')
    def fragment_zones(self):
        """Cut the overlapping shapes into every separate piece; the pieces are named A, B, C… for now."""
        if not self._need_video():
            return
        if self.chk_lock.isChecked():
            self.chk_lock.setChecked(False)
        res = canvas_fragment(self.win, self.canvas)
        if not res:
            return
        n_src, names = res
        self.status.showMessage(f"✂ {n_src} shapes cut into {len(names)} pieces ({', '.join(names)}). "
                                "Rename them: double-click a name in the list, or Zone ▸ Rename zones…  "
                                "(Ctrl+Z undoes the fragment.)", 12000)
        if ask_name_pieces(self.win, n_src, names):
            self.rename_all_zones()

    def merge_zones(self):
        """Join the selected zones into one (PowerPoint 'Union')."""
        if self.chk_lock.isChecked():
            self.chk_lock.setChecked(False)
        res = canvas_merge(self.win, self.canvas)
        if res:
            self.status.showMessage(f"⊕ {', '.join(res[1])} merged into '{res[0]}'. (Ctrl+Z undoes it.)", 8000)

    def _apply_name_mapping(self, mapping):
        if all(k == v for k, v in mapping.items()):
            return
        self.canvas.push_undo()
        new = {mapping.get(k, k): v for k, v in self.canvas.zones.items()}
        cols = {mapping.get(k, k): self.canvas.color_of(k) for k in self.canvas.zones}
        self.canvas.zones = new
        self.canvas.colors.update(cols)
        self.canvas.selected = [mapping.get(n, n) for n in self.canvas.selected]
        self.canvas.update()
        self.canvas.zonesEdited.emit()

    # --- crop + scale
    def _apply_crop(self, pts):
        from tracking.location import compute_perspective_transform, order_points
        matrix, w, h = compute_perspective_transform(pts)
        M = np.asarray(matrix, np.float64)
        Minv = np.linalg.inv(M)

        def fwd(p):
            v = M @ np.array([p[0], p[1], 1.0])
            return (float(v[0] / v[2]), float(v[1] / v[2]))
        self.crop = {"matrix": matrix, "inv": Minv, "w": int(w), "h": int(h),
                     "corners": [tuple(map(float, c)) for c in order_points(pts)]}
        self.canvas.transform_points(fwd)
        self.cur_frame = self._warp(self.raw_frame)
        self.canvas.set_frame(self.cur_frame)
        self.canvas.reset_view()
        if self.spin_len.value() > 0:
            self.spin_ppc.setValue(round(w / self.spin_len.value(), 2))
        self._on_zones_edited()
        self.status.showMessage(f"Arena straightened to {w}×{h}px; zones carried over and scale re-estimated "
                                f"({self.spin_ppc.value():.2f} px/cm).", 8000)

    def remove_crop(self, quiet=False):
        if not self.crop:
            return
        Minv = self.crop["inv"]

        def back(p):
            v = Minv @ np.array([p[0], p[1], 1.0])
            return (float(v[0] / v[2]), float(v[1] / v[2]))
        self.crop = None
        self.canvas.transform_points(back)
        self.cur_frame = self.raw_frame
        self.canvas.set_frame(self.cur_frame)
        self.canvas.reset_view()
        self._on_zones_edited()
        if not quiet:
            self.status.showMessage("Crop removed — zones mapped back onto the full frame.", 5000)

    def _apply_scale(self, pts):
        (x1, y1), (x2, y2) = pts[:2]
        dist = float(math.hypot(x2 - x1, y2 - y1))
        if dist < 5:
            self.status.showMessage("Those points are too close together.", 5000)
            return
        cm, ok = QInputDialog.getDouble(self.win, "Set scale", f"Measured {dist:.1f} px.\nReal length (cm):",
                                        float(self.spin_len.value()), 0.1, 10000.0, 2)
        if ok and cm > 0:
            self.spin_ppc.setValue(round(dist / cm, 2))
            self.status.showMessage(f"Scale set: {dist / cm:.2f} px/cm", 6000)

    # --- automatic zones
    def _bg_key(self):
        return (self.player.path, None if not self.crop else tuple(np.round(np.asarray(self.crop["matrix"]), 6).ravel()))

    def _background_image(self, on_ready, samples=40, quiet=False):
        """Animal-free background of the current video (cached), delivered async.
        quiet=True: build it in the background without a wait cursor (guided fit)."""
        if self.player.cap is None:
            return
        key = self._bg_key()
        if key in self._bg_cache:
            on_ready(self._bg_cache[key])
            return
        waiters = self._bg_waiters.setdefault(key, [])
        waiters.append((on_ready, quiet))
        if len(waiters) > 1:                 # already being built
            if not quiet:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            return
        path, crop = self.player.path, (dict(self.crop) if self.crop else None)
        aw, ah = self.arena_size

        def work():
            return compute_video_background(path, crop, (aw, ah), samples)

        def restore():
            for _cb, q in self._bg_waiters.get(key, []):
                if not q:
                    QApplication.restoreOverrideCursor()

        def done(img):
            restore()
            self._bg_cache[key] = img
            for cb, _q in self._bg_waiters.pop(key, []):
                cb(img)

        def failed(msg):
            restore()
            loud = any(not q for _cb, q in self._bg_waiters.pop(key, []))
            if loud:
                QMessageBox.warning(self.win, "Background", msg)
        if not quiet:
            self.status.showMessage("Building an animal-free background image…")
            QApplication.setOverrideCursor(Qt.WaitCursor)
        self._thread_run(work, done, failed)

    def auto_detect_zones(self):
        if not self._need_video():
            return
        if self.canvas.tool is not None:
            self.canvas.cancel_tool(silent=True)
        if self.chamber_ui.enabled:
            self.chamber_ui.auto_zones()                    # fitted inside each chamber
            return

        def ready(img):
            from tracking.apparatus import auto_detect_zones, guess_test
            # 1) one of YOUR set-ups (learned from the videos you analysed) -- pressing ⚡ again on the same
            #    video skips this and uses the built-in template
            again = getattr(self, "_auto_lib_video", None) == self.player.path
            self._auto_lib_video = None
            if not again:
                hit = self._match_learned(img)
                if hit is not None:
                    zones, entry, msg = hit
                    self.canvas.colors.clear()
                    self.canvas._color_i = 0
                    self.canvas.set_zones(zones, record=True, select_all=True)
                    if entry.get("covered"):
                        self.covered_override = set(entry["covered"])
                    self._auto_lib_video = self.player.path
                    self.status.showMessage(f"⚡ Auto: zones of {msg}. Press ⚡ Auto again for the built-in "
                                            "template.", 15000)
                    return
            if not self._check_test_type(img):
                return
            ppc = self.spin_ppc.value() if self.spin_ppc.value() > 0 else None
            kind = self.test_key
            if kind == "custom":                       # no test chosen: fit what the picture looks like
                kind = guess_test(img) or "custom"
            zones, _outline, msg = auto_detect_zones(img, kind, ppc=ppc)
            if kind != self.test_key and zones:
                msg = f"Looks like a {TESTS[kind][0]}. " + msg
            if not zones:
                box = QMessageBox(QMessageBox.Information, "Auto-detect zones",
                                  msg + "\n\n🎯 Guided fit: click the arm ends (mazes) or the floor corners "
                                        "(boxes) and the zones are fitted exactly to your clicks.", parent=self.win)
                b_g = box.addButton("🎯 Guided fit", QMessageBox.AcceptRole)
                box.addButton("Close", QMessageBox.RejectRole)
                box.exec()
                if box.clickedButton() is b_g:
                    self._guided_clicked()
                return
            self.canvas.colors.clear()
            self.canvas._color_i = 0
            self.canvas.set_zones(zones, record=True, select_all=True)
            self.status.showMessage("Auto-detect: " + msg + "  Not right? Ctrl+Z, then 🎯 Guided (click the arm "
                                    "ends / corners).", 15000)
            if self.test_key == "epm":
                dlg = ZoneNamesDialog(self.win, list(zones), zone_suggestions("epm"), title="Confirm the EPM arms",
                                      note="Open / closed arms were guessed from the wall edges (look at the "
                                           "coloured labels on the video). If they are the wrong way round press "
                                           "'Swap'.", epm_swap=True)
                if dlg.exec() == QDialog.Accepted:
                    self._apply_name_mapping(dlg.mapping())
        self._background_image(ready)

    def _check_test_type(self, img):
        """Warn when the video's apparatus does not look like the selected test (e.g. a light/dark box while
        'Open Field' is selected) and offer the test it looks like.  False = stop."""
        from tracking.apparatus import guess_test
        if self.test_key == "custom":
            return True
        key = (self.player.path, self.test_key)
        if key in self._shape_warned:
            return True
        try:
            guess = guess_test(img)
        except Exception:
            guess = None
        same = {"nor": "open_field"}
        if guess is None or same.get(self.test_key, self.test_key) == same.get(guess, guess):
            return True
        if self.test_key == "nor" and guess == "open_field":
            return True
        self._shape_warned.add(key)
        box = QMessageBox(QMessageBox.Question, "Check the test type",
                          f"The apparatus in this video looks like a <b>{TESTS[guess][0]}</b>, but the selected "
                          f"test is <b>{TESTS[self.test_key][0]}</b>.<br><br>Zones are fitted for the selected "
                          "test.", parent=self.win)
        b_use = box.addButton(f"Use {TESTS[guess][0]}", QMessageBox.AcceptRole)
        b_change = box.addButton("Other test…", QMessageBox.ActionRole)
        b_guided = box.addButton("🎯 Guided fit", QMessageBox.ActionRole)
        box.addButton("Continue anyway", QMessageBox.RejectRole)
        box.exec()
        if box.clickedButton() is b_use:
            self.select_test(guess, navigate=False)
            self._shape_warned.add((self.player.path, self.test_key))
            QTimer.singleShot(0, self.auto_detect_zones)
            return False
        if box.clickedButton() is b_change:
            keys = [k for k in TESTS if k != "custom"]
            labels = [TESTS[k][0] for k in keys]
            choice, ok = QInputDialog.getItem(self.win, "Test type", "Test type of this video:", labels, 0, False)
            if ok and choice:
                self.select_test(keys[labels.index(choice)], navigate=False)
                self._shape_warned.add((self.player.path, self.test_key))
                QTimer.singleShot(0, self.auto_detect_zones)
            return False
        if box.clickedButton() is b_guided:
            QTimer.singleShot(0, self._guided_clicked)
            return False
        return True

    # --- guided fit: a few clicks -> exact zones
    def _guided_clicked(self):
        if not self._need_video():
            return
        if self.test_key in ("open_field", "nor", "custom"):
            menu = QMenu(self.win)
            a_rect = menu.addAction("Square / rectangular floor — click its 4 corners")
            a_circ = menu.addAction("Round arena — click 3 or more points on its edge")
            btn = getattr(self, "btn_zone_guided", None)
            pos = btn.mapToGlobal(btn.rect().bottomLeft()) if btn is not None and btn.isVisible() else QCursor.pos()
            chosen = menu.exec(pos)
            if chosen is a_rect:
                self.begin_guided("rect")
            elif chosen is a_circ:
                self.begin_guided("circle")
        else:
            self.begin_guided("rect")

    def begin_guided(self, shape="rect"):
        from tracking.apparatus import guided_spec
        if not self._need_video():
            return
        if self.chk_lock.isChecked():
            self.chk_lock.setChecked(False)
        self._stop_playback()
        if self.stack.currentIndex() != 2:
            self.go(2)
        n, labels, prompts = guided_spec(self.test_key, shape)
        self._guided_shape = shape
        self.canvas.select([])
        self.canvas.start_tool("points", "guided", max_points=n, banner=prompts[0], labels=labels,
                               prompts=prompts, auto_finish=n is not None)
        self.tool_buttons["toolZone"].setChecked(True)
        self._update_opbar()
        self._background_image(lambda _img: None, quiet=True)      # build it while the user clicks

    def _place_zones(self, zones):
        """Zones from a fit (guided / traced outline): in chamber mode they go into their chamber (the other
        chambers keep their zones), otherwise they replace the layout."""
        cu = getattr(self, "chamber_ui", None)
        if cu is not None and cu.enabled:
            merged, new = cu.merge(zones, target=cu.active)
            self.canvas.set_zones(merged, record=True)
            self.canvas.select(new)
            return
        self.canvas.set_zones(zones, record=True, select_all=True)

    def _apply_guided(self, pts):
        shape = self._guided_shape
        test = self.test_key

        def ready(img):
            from tracking.apparatus import zones_from_clicks
            ppc = self.spin_ppc.value() if self.spin_ppc.value() > 0 else None
            zones, msg = zones_from_clicks(test, pts, img, ppc=ppc, shape=shape)
            if not zones:
                QMessageBox.information(self.win, "Guided fit", msg)
                return
            self.canvas.colors.clear()
            self.canvas._color_i = 0
            self._place_zones(zones)
            self.status.showMessage("🎯 " + msg + "  Drag zones or points to fine-tune (Ctrl+Z undoes).", 15000)
            self._update_opbar()
        self._background_image(ready)

    def _zones_from_outline(self, pts):
        from tracking.apparatus import zones_from_outline
        w, h = self.arena_size
        key = (self.player.path, None if not self.crop else tuple(np.round(np.asarray(self.crop["matrix"]), 6).ravel()))
        bg = self._bg_cache.get(key)
        if bg is None and self.cur_frame is not None:
            bg = self.cur_frame
        zones, msg = zones_from_outline(pts, w, h, self.test_key, background_bgr=bg,
                                        ppc=self.spin_ppc.value() or None)
        if not zones:
            QMessageBox.information(self.win, "Trace outline", msg)
            return
        if self.canvas.zones and QMessageBox.question(self.win, "Trace outline",
                                                      f"{msg}\n\nReplace the current zones?") != QMessageBox.Yes:
            return
        self.canvas.colors.clear()
        self.canvas._color_i = 0
        self._place_zones(zones)
        self.status.showMessage(msg, 10000)
        if all(len(n) <= 2 and n.isalpha() and n.isupper() for n in zones):
            dlg = ZoneNamesDialog(self.win, list(zones), zone_suggestions(self.test_key),
                                  note="The outline was split into these parts. Give each one its real name.")
            if dlg.exec() == QDialog.Accepted:
                self._apply_name_mapping(dlg.mapping())

    # --- layouts
    def save_layout(self):
        if not self.canvas.zones and not self.chamber_ui.enabled:
            self.status.showMessage("No zones to save.", 4000)
            return
        LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
        default = LAYOUT_DIR / f"{self.test_key}_{datetime.date.today():%Y%m%d}.json"
        path, _ = QFileDialog.getSaveFileName(self.win, "Save zone layout", str(default), "Zone layout (*.json)")
        if not path:
            return
        save_json(path, {"version": APP_VERSION, "test": self.test_key, "frame_size": [self.player.w, self.player.h],
                         "crop": None if not self.crop else {"corners": self.crop["corners"]},
                         "zones": {k: [list(p) for p in v] for k, v in self.canvas.zones.items()},
                         "masks": [[list(p) for p in m] for m in self.canvas.masks],
                         "pixels_per_cm": self.spin_ppc.value(),
                         "arena_cm": [self.spin_len.value(), self.spin_wid.value()],
                         "auto_mask": self.chk_automask.isChecked(),
                         "covered_zones": (None if self.covered_override is None
                                           else sorted(self.covered_names())),
                         "chambers": self.chamber_ui.layout_data()})
        self.status.showMessage(f"Zone layout saved to {path}", 6000)

    def load_layout(self):
        path, _ = QFileDialog.getOpenFileName(self.win, "Load zone layout", str(LAYOUT_DIR), "Zone layout (*.json)")
        if not path:
            return
        data = load_json(path, None)
        if not data or "zones" not in data:
            QMessageBox.warning(self.win, "Load layout", "That file is not a zone layout.")
            return
        if self.crop:
            self.remove_crop(quiet=True)
        fw, fh = data.get("frame_size") or [self.player.w, self.player.h]
        sx, sy = self.player.w / max(1, fw), self.player.h / max(1, fh)
        zones = {k: [(x, y) for x, y in v] for k, v in data["zones"].items()}
        masks = [[(x, y) for x, y in m] for m in data.get("masks", [])]
        if data.get("crop") and data["crop"].get("corners"):
            from tracking.location import compute_perspective_transform
            corners = [(x * sx, y * sy) for x, y in data["crop"]["corners"]]
            matrix, w, h = compute_perspective_transform(corners)
            self.crop = {"matrix": matrix, "inv": np.linalg.inv(np.asarray(matrix, np.float64)), "w": int(w),
                         "h": int(h), "corners": corners}
            self.cur_frame = self._warp(self.raw_frame)
            self.canvas.set_frame(self.cur_frame)
            if (fw, fh) != (self.player.w, self.player.h):
                self.status.showMessage("Layout saved on a different video size — please check the crop.", 8000)
        else:
            zones = {k: [(x * sx, y * sy) for x, y in v] for k, v in zones.items()}
            masks = [[(x * sx, y * sy) for x, y in m] for m in masks]
        self.canvas.push_undo()
        self.canvas.masks = masks
        self.canvas.set_zones(zones, record=False, select_all=True)
        if "chambers" in data:
            crop_used = bool(data.get("crop") and data["crop"].get("corners"))
            self.chamber_ui.load_layout_data(data["chambers"], 1.0 if crop_used else sx, 1.0 if crop_used else sy)
        if data.get("pixels_per_cm"):
            self.spin_ppc.setValue(float(data["pixels_per_cm"]))
        self.chk_automask.setChecked(bool(data.get("auto_mask", True)))
        if "covered_zones" in data:
            self.covered_override = None if data["covered_zones"] is None else set(data["covered_zones"])
        if data.get("test") in TESTS and data["test"] != self.test_key:
            self.test_key = data["test"]
            if data["test"] in self.exp_tiles:
                self.exp_tiles[data["test"]].setChecked(True)
            self._update_kpi_captions()
        self.status.showMessage(f"Zone layout loaded from {Path(path).name}", 6000)

    # ------------------------------------------------------------------ tracking page
    def _bind_tracking(self):
        w = self.win
        self.view_track = FrameView(find(w, QLabel, "videoFrameTracking"), self)
        self.btn_play_track = find(w, QToolButton, "btnPlayTracking")
        self.btn_play_track.setIcon(self.icon_play)
        self.btn_play_track.clicked.connect(self.toggle_play)
        self.sl_track = find(w, QSlider, "sliderPositionTracking")
        self.sl_track.valueChanged.connect(self._slider_moved)
        self.lbl_time_track = find(w, QLabel, "lblTimeTracking")
        self.lbl_time_track.setStyleSheet("color: #000000; font-weight: 600;")
        self.sl_thr = find(w, QSlider, "sliderThreshold")
        self.sl_smooth = find(w, QSlider, "sliderSmooth")
        self.sl_min = find(w, QSlider, "sliderMinSize")
        self._track_defaults = {"smooth": self.sl_smooth.value(), "min_size": self.sl_min.value()}
        self.lbl_thr_cap = find(w, QLabel, "qlabel_24")
        self.lbl_thr_val = find(w, QLabel, "lblThresholdValue")
        self.spin_start = find(w, QDoubleSpinBox, "spinTrackStart")
        self.spin_end = find(w, QDoubleSpinBox, "spinTrackEnd")
        self.cmb_color_mode = find(w, QComboBox, "cmbColorMode")
        self.cmb_color_mode.setCurrentText("RGB" if str(self.cfg.get("color_mode", "gray")).lower() == "rgb" else "Gray")
        self.cmb_color_mode.currentTextChanged.connect(self._color_mode_changed)
        self.chk_window = find(w, QCheckBox, "chkUseWindow")
        self.spin_window = find(w, QSpinBox, "spinWindowSize")
        self.spin_window_weight = find(w, QDoubleSpinBox, "spinWindowWeight")
        self._window_widgets = [self.chk_window, self.spin_window, self.spin_window_weight,
                                find(w, QLabel, "lblWindowSize"), find(w, QLabel, "lblWindowWeight")]
        card_lay = find(w, QFrame, "cardTrackingSettings").layout()
        # animal colour (polarity) selector, right after "Detection mode"
        self.lbl_pol = QLabel("Animal colour")
        self.lbl_pol.setProperty("cssClass", "fieldLabel")
        self.cmb_pol = QComboBox()
        for key, text in POLARITIES:
            self.cmb_pol.addItem(text, key)
        self.cmb_pol.setCurrentIndex(max(0, [k for k, _t in POLARITIES].index(self.cfg.get("polarity", "auto"))))
        self.cmb_pol.setToolTip("Black mouse on a light floor = 'Darker'. Auto-detect decides from the video. "
                                "Knowing the colour makes the background immune to an animal that rests in one "
                                "place for a long time.")
        self.cmb_pol.currentIndexChanged.connect(lambda _i: self._settings_changed())
        i = card_lay.indexOf(self.cmb_color_mode)
        card_lay.insertWidget(i + 1, self.lbl_pol)
        card_lay.insertWidget(i + 2, self.cmb_pol)
        # low-light boost (dim videos, an open dark box at a few lux)
        self.lbl_low = QLabel("Low-light boost")
        self.lbl_low.setProperty("cssClass", "fieldLabel")
        self.cmb_low = QComboBox()
        for key, text in LOW_LIGHT_CHOICES:
            self.cmb_low.addItem(text, key)
        cur = str(self.cfg.get("low_light", "off")).lower()
        self.cmb_low.setCurrentIndex(max(0, self.cmb_low.findData(cur)))
        self.cmb_low.setToolTip("Brightens the dark grey levels of the video (for tracking and on the screen): a "
                                "mouse in a dim zone (open dark box, dark corner, 5 lux and up) gets several times "
                                "more contrast. The noise of the dark areas is measured too, so they do not give "
                                "false detections.\nAuto = only when a zone is dark (its floor below ~70 grey "
                                "levels), as much as needed. Low / Medium / Strong = always, fixed.")
        self.cmb_low.currentIndexChanged.connect(self._low_light_changed)
        # light/dark box: the dark compartment covered by a lid, or open
        self.lbl_darkbox = QLabel("Dark box")
        self.lbl_darkbox.setProperty("cssClass", "fieldLabel")
        self.cmb_darkbox = QComboBox()
        for key, text in DARK_BOX_CHOICES:
            self.cmb_darkbox.addItem(text, key)
        self.cmb_darkbox.setToolTip("Covered = a lid hides the mouse inside the dark box: nothing is tracked there "
                                    "and the time is counted as 'in the dark box' (hidden).\nOpen = the mouse can "
                                    "be seen inside (dim light): it is tracked there too -- turn on the Low-light "
                                    "boost if it is hard to see.")
        self.cmb_darkbox.currentIndexChanged.connect(self._darkbox_changed)
        card_lay.insertWidget(i + 3, self.lbl_low)
        card_lay.insertWidget(i + 4, self.cmb_low)
        card_lay.insertWidget(i + 5, self.lbl_darkbox)
        card_lay.insertWidget(i + 6, self.cmb_darkbox)
        # tracking point row is not applicable (body centre is always tracked)
        for n in ("qlabel_23", "chkHead", "chkCenter", "chkTail"):
            wd = w.findChild(QWidget, n)
            if wd is not None:
                wd.hide()
        for slider, lbl, fmt in ((self.sl_smooth, "lblSmoothValue", lambda v: "off" if v < 2 else str(v)),
                                 (self.sl_min, "lblMinSizeValue", str)):
            label = find(w, QLabel, lbl)
            slider.valueChanged.connect(lambda v, l=label, f=fmt: l.setText(f(v)))
            label.setText(fmt(slider.value()))
        self.sl_thr.valueChanged.connect(self._thr_label)
        self.sl_thr.valueChanged.connect(lambda _v: self._settings_changed())
        self.sl_min.valueChanged.connect(lambda _v: self._settings_changed())
        self.sl_min.setMaximum(2000)
        self.sl_smooth.setToolTip("Centred smoothing window (frames) for the trajectory, distance and zones "
                                  "(below 2 = off). No lag is introduced.")
        self.sl_min.setToolTip("Smallest spot (in pixels) that can be the animal. Easiest: press 'Measure the "
                               "mouse' below and click its nose and tail base -- the size is set for you "
                               "(about a quarter of the mouse's body area). The size is also measured "
                               "automatically from the video.")
        find(w, QLabel, "qlabel_26").setText("Minimum animal size (px)")
        find(w, QLabel, "qlabel_25").setText("Smoothing (frames)")
        # how many detection-check preview images are saved with each result
        self.lbl_prev = QLabel("Preview images per video")
        self.lbl_prev.setProperty("cssClass", "fieldLabel")
        self.spin_prev = QSpinBox()
        self.spin_prev.setRange(0, 100)
        self.spin_prev.setSpecialValueText("None")
        self.spin_prev.setSuffix(" images")
        self.spin_prev.setValue(int(self.cfg.get("preview_samples", 3)))
        self.spin_prev.setToolTip("How many preview images (video frame + detected animal + zones) are saved in "
                                  "the 'preview' folder of every result, spread evenly over the analysed time. "
                                  "0 = none.")
        self.spin_prev.valueChanged.connect(self._preview_count_changed)
        idx = -1
        for i in range(card_lay.count()):
            it = card_lay.itemAt(i)
            if it.widget() is self.sl_min or (it.layout() is not None and
                                              self._layout_with(it.layout(), self.sl_min) is not None):
                idx = i
        if idx < 0:
            idx = card_lay.indexOf(self.sl_min)
        # measure the mouse: 2 clicks on the video (nose tip, tail base) -> body length -> minimum size
        self.btn_measure = QPushButton("📏  Measure the mouse")
        self.btn_measure.setProperty("cssClass", "secondary")
        self.btn_measure.setMinimumHeight(30)
        self.btn_measure.setToolTip("Pause on a frame where the mouse is stretched out, press this button, then click "
                                    "the tip of its NOSE and the BASE of its TAIL on the video. The mouse length "
                                    "(pixels) sets the minimum animal size, and helps the tracker know how big "
                                    "the animal is. Right-click = cancel.")
        self.btn_measure.clicked.connect(self.start_measure_animal)
        self.lbl_measure = QLabel("")
        self.lbl_measure.setWordWrap(True)
        self.lbl_measure.setStyleSheet("color:#475569; font-size:9pt;")
        self._measure_pts = None
        self._track_disp = None
        self._disp_gamma_cache = None
        bl = float(self.cfg.get("body_length_px") or 0)
        if bl > 0:
            self.lbl_measure.setText(f"Mouse length set: {bl:.0f} px (nose → tail base).")
        self.view_track.pressed.connect(self._track_view_pressed)
        card_lay.insertWidget(idx + 1, self.btn_measure)
        card_lay.insertWidget(idx + 2, self.lbl_measure)
        card_lay.insertWidget(idx + 3, self.lbl_prev)
        card_lay.insertWidget(idx + 4, self.spin_prev)
        # minimum contrast: measured from the video
        self.btn_auto_thr = QPushButton("🎯  Auto-set contrast")
        self.btn_auto_thr.setProperty("cssClass", "secondary")
        self.btn_auto_thr.setMinimumHeight(30)
        self.btn_auto_thr.setToolTip("Checks 40 frames between the start and end time: how different the mouse is "
                                     "from the empty floor (also in its darkest moments) and how noisy the video "
                                     "is, and recommends the minimum contrast.")
        self.btn_auto_thr.clicked.connect(self.auto_contrast)
        self.lbl_thr_hint = QLabel("")
        self.lbl_thr_hint.setWordWrap(True)
        self.lbl_thr_hint.setStyleSheet("color:#475569; font-size:9pt;")
        jdx = -1
        for k in range(card_lay.count()):
            it = card_lay.itemAt(k)
            if it.widget() is self.sl_thr or (it.layout() is not None and
                                              self._layout_with(it.layout(), self.sl_thr) is not None):
                jdx = k
        if jdx < 0:
            jdx = card_lay.indexOf(self.sl_thr)
        card_lay.insertWidget(jdx + 1, self.btn_auto_thr)
        card_lay.insertWidget(jdx + 2, self.lbl_thr_hint)
        self.cmb_video = find(w, QComboBox, "cmbVideoFile")
        self.cmb_video.clear()
        self.cmb_video.activated.connect(self._video_combo_changed)
        self.cmb_method = find(w, QComboBox, "cmbTrackingMethod")
        self.cmb_method.clear()
        for key, text in ENGINES:
            self.cmb_method.addItem(text, key)
        self.cmb_method.setEnabled(True)
        self.cmb_method.setCurrentIndex(0 if self.cfg.get("engine", "robust") == "robust" else 1)
        self.cmb_method.currentIndexChanged.connect(self._engine_changed)
        self.cmb_animals = find(w, QComboBox, "cmbNumAnimals")
        while self.cmb_animals.count() > 3:
            self.cmb_animals.removeItem(self.cmb_animals.count() - 1)
        self.cmb_animals.activated.connect(self._animals_changed)
        # 1 = one animal per arena / chamber (separate cages: Video Setup ▸ Chambers).  2-3 = several animals
        # TOGETHER in one arena (social tests): the multi-animal engine keeps their identities by motion and,
        # when they look different (coat colour / a marked back), by their look after every contact.
        self.cmb_animals.setCurrentIndex(0)
        self.cmb_animals.setToolTip("Animals together in ONE arena (social test).\n1 = one animal (separate "
                                    "cages: use Video Setup ▸ Chambers).\n2–3 = identities kept by motion and, "
                                    "after the animals touch, by their look — mark the animals (a dyed back / "
                                    "different coat) so they can be told apart; check the preview video.")
        lab_an = w.findChild(QLabel, "qlabel_22")
        if lab_an is not None:
            lab_an.setText("Animals in one arena")
        self.btn_test = QPushButton(TEST_BTN_TEXT)
        self.btn_test.setProperty("cssClass", "secondary")
        self.btn_test.setMinimumHeight(34)
        self.btn_test.setToolTip("Builds the background and shows a still picture of the Start time with the animal "
                                 "the tracking will find there (yellow = detected animal). Nothing is analysed. "
                                 "Press play or scrub to check other moments.")
        self.btn_test.clicked.connect(self.test_detection)
        self.btn_live = QPushButton("📷  Live tracking from a camera")
        self.btn_live.setProperty("cssClass", "secondary")
        self.btn_live.setMinimumHeight(34)
        self.btn_live.setToolTip("Track the animal live on a camera (built-in or USB) with these zones and settings: "
                                 "the trial is recorded and tracked at the same time (the tracked view runs about "
                                 "30 s behind) and the results are ready right after the trial.")
        self.btn_live.clicked.connect(self.open_camera)
        self.btn_start = find(w, QPushButton, "btnStartTracking")
        if LIVE_TRACKING:
            card_lay.insertWidget(card_lay.indexOf(self.btn_start), self.btn_live)
        else:
            self.btn_live.setParent(None)
            self.btn_live.hide()
        card_lay.insertWidget(card_lay.indexOf(self.btn_start), self.btn_test)
        self.btn_pose = QPushButton("⤒  Use DeepLabCut / SLEAP positions…")
        self.btn_pose.setProperty("cssClass", "secondary")
        self.btn_pose.setMinimumHeight(30)
        self.btn_pose.setToolTip("Already tracked this video with a pose-estimation program? Choose its CSV file "
                                 "(DeepLabCut or SLEAP 'analysis CSV'): BT uses its body points instead of its own "
                                 "tracking and writes the usual results with these zones (the nose point counts "
                                 "object exploration).")
        self.btn_pose.clicked.connect(self.import_pose_file)
        card_lay.insertWidget(card_lay.indexOf(self.btn_start), self.btn_pose)
        self.btn_start.clicked.connect(self.start_or_stop_tracking)
        gear = find(w, QToolButton, "btnTrackingSettings")
        gear.clicked.connect(self.open_settings)
        self.btn_reset_track = QToolButton(gear.parentWidget())
        self.btn_reset_track.setText("↻")
        self.btn_reset_track.setToolTip("Reset: clear the results and the tracking view, and/or put the tracking "
                                        "settings back to their defaults (saved files are kept).")
        self.btn_reset_track.setCursor(Qt.PointingHandCursor)
        self.btn_reset_track.setStyleSheet("QToolButton { font-size: 15pt; border: 1px solid #d5dce8; "
                                           "border-radius: 6px; padding: 2px 8px; background: #ffffff; } "
                                           "QToolButton:hover { background: #f3f6fb; }")
        self.btn_reset_track.clicked.connect(lambda: self.reset_tracking_results(settings=True))
        gl = self._layout_with(find(w, QWidget, "pageTracking").layout(), gear)
        if gl is not None:
            gl.insertWidget(gl.indexOf(gear), self.btn_reset_track)
        find(w, QToolButton, "btnFullscreen").clicked.connect(self.toggle_fullscreen)
        self.spin_start.valueChanged.connect(lambda _v: self._settings_changed())
        self.spin_end.valueChanged.connect(lambda _v: self._settings_changed())
        self._engine_changed()

    @property
    def engine(self):
        return self.cmb_method.currentData() or "robust"

    def _engine_changed(self, *_a):
        robust = self.engine == "robust"
        self.sl_thr.blockSignals(True)
        if robust:
            self.lbl_thr_cap.setText("Minimum contrast (grey levels)")
            self.sl_thr.setRange(4, 60)
            self.sl_thr.setValue(int(self.cfg.get("min_contrast", 12)))
            self.sl_thr.setToolTip("How many grey levels darker/lighter than the empty floor a pixel must be to "
                                   "count as 'mouse'. Too high: the mouse is missed in dim places (lost frames, "
                                   "wobbly track). Too low: shadows and noise join the mouse. A good value is "
                                   "about half of the mouse's weakest contrast -- press 'Auto-set' to measure "
                                   "it. 12 suits most videos.")
        else:
            self.lbl_thr_cap.setText("Location threshold (percentile)")
            self.sl_thr.setRange(980, 999)
            self.sl_thr.setValue(int(round(float(self.cfg.get("loc_thresh", 99.5)) * 10)))
            self.sl_thr.setToolTip("Classic engine percentile: 99.5 keeps the strongest 0.5 % of pixel differences.")
        self.sl_thr.blockSignals(False)
        self._thr_label(self.sl_thr.value())
        for wd in self._window_widgets:
            wd.setVisible(not robust)
        self.cmb_pol.setEnabled(robust)
        self.lbl_pol.setEnabled(robust)
        if hasattr(self, "btn_auto_thr"):
            self.btn_auto_thr.setEnabled(robust)
        self.cmb_pol.setToolTip(self.cmb_pol.toolTip() if robust else
                                "The Classic engine always uses the absolute difference.")
        self.cfg["engine"] = self.engine
        self._settings_changed()

    def _chamber_view(self):
        """True while the Tracking page shows ONE chamber's own settings (they are not the program settings)."""
        cu = getattr(self, "chamber_ui", None)
        return cu is not None and cu.view is not None

    def _thr_label(self, v):
        own = self._chamber_view()
        if self.engine == "robust":
            self.lbl_thr_val.setText(str(v))
            if not own:
                self.cfg["min_contrast"] = int(v)
        else:
            self.lbl_thr_val.setText(f"{v / 10:.1f}")
            if not own:
                self.cfg["loc_thresh"] = v / 10.0

    def _color_mode_changed(self, text):
        if not self._chamber_view():
            self.cfg["color_mode"] = text.lower()
        self._settings_changed()

    def _preview_count_changed(self, v):
        self.cfg["preview_samples"] = int(v)
        save_json(SETTINGS_PATH, self.cfg)

    def _settings_changed(self):
        self._preview = None
        if self._chamber_view():                 # one chamber's own value: kept by the chamber, not the program
            return
        if hasattr(self, "cmb_pol"):
            self.cfg["polarity"] = self.cmb_pol.currentData() or "auto"
        save_json(SETTINGS_PATH, self.cfg)

    def _refresh_video_combo(self):
        self._refresh_video_list()
        paths = [it["path"] for it in self.exp_videos]
        if self.player.path and self.player.path not in paths:
            paths.insert(0, self.player.path)
        self.cmb_video.blockSignals(True)
        self.cmb_video.clear()
        for p in paths:
            self.cmb_video.addItem(Path(p).name, p)
        if self.player.path in paths:
            self.cmb_video.setCurrentIndex(paths.index(self.player.path))
        self.cmb_video.blockSignals(False)

    def _video_combo_changed(self, idx):
        path = self.cmb_video.itemData(idx)
        if path and path != self.player.path and not self.tracking:
            self.open_video(path)

    def _animals_changed(self, _idx):
        n = int(self.cmb_animals.currentText() or "1")
        self._preview = None
        if hasattr(self, "cmb_low"):
            self.cmb_low.setEnabled(n <= 1)
            self.lbl_low.setEnabled(n <= 1)
        if n <= 1:
            self.status.showMessage("Single-animal tracking (robust engine).", 3000)
        else:
            self.status.showMessage(f"{n} animals together in one arena — identities kept by motion and, after "
                                    "contacts, by their look (mark the animals). Check the preview video.", 8000)

    def _apply_results_root(self):
        import tracking.location as _loc
        root = str(self.cfg.get("results_root") or "").strip()
        _loc.RESULTS_ROOT = root or None

    def ensure_results_folder(self):
        """Asked once: results next to each video (a 'results' folder) or all in one folder.  False = cancelled."""
        if self.cfg.get("results_root_asked"):
            return True
        box = QMessageBox(self.win)
        box.setWindowTitle("Where should results go?")
        box.setText("Where should BehavioralTracker save the results (Excel, charts, PDF, BT files)?\n\n"
                    "• Next to each video: a 'results' folder beside the video (handy on your own disk).\n"
                    "• One folder: everything in one place you choose (best when the videos are on a slow "
                    "network drive).\n\nYou can change this any time in Settings.")
        b_next = box.addButton("Next to each video", QMessageBox.AcceptRole)
        b_one = box.addButton("One folder…", QMessageBox.ActionRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        if box.clickedButton() is b_one:
            d = QFileDialog.getExistingDirectory(self.win, "Folder for all results", str(Path.home()))
            if not d:
                return False
            self.cfg["results_root"] = d
        elif box.clickedButton() is b_next:
            self.cfg["results_root"] = ""
        else:
            return False
        self.cfg["results_root_asked"] = True
        self._apply_results_root()
        try:
            save_json(SETTINGS_PATH, self.cfg)
        except Exception:
            pass
        return True

    def open_settings(self):
        dlg = SettingsDialog(self.win, self.cfg)
        if dlg.exec() == QDialog.Accepted:
            self.cfg.update(dlg.values())
            self.cfg["results_root_asked"] = True
            self._apply_results_root()
            self.cmb_color_mode.setCurrentText("RGB" if self.cfg["color_mode"] == "rgb" else "Gray")
            self.spin_prev.blockSignals(True)
            self.spin_prev.setValue(int(self.cfg.get("preview_samples", 3)))
            self.spin_prev.blockSignals(False)
            self._settings_changed()
            self.status.showMessage("Detection settings saved.", 4000)

    def toggle_fullscreen(self):
        self.win.showNormal() if self.win.isFullScreen() else self.win.showFullScreen()

    def _overlay_zones_bgr(self, frame):
        """Zones + dimmed auto-mask area drawn on a BGR frame (tracking page view)."""
        from tracking.robust_tracker import brighten
        out = brighten(frame, self._display_gamma(frame))
        if out is frame:
            out = frame.copy()
        zones = self.canvas.zones
        covered = sorted(self.covered_names(zones)) if zones else []
        ch_on = self.chamber_ui.enabled
        if (zones and (self.chk_automask.isChecked() or covered)) or ch_on:
            from tracking.location import build_detection_exclusion
            ex = build_detection_exclusion(out.shape[1], out.shape[0], self.canvas.masks, zones,
                                           self.chk_automask.isChecked() and bool(zones),
                                           margin=self.cfg.get("auto_mask_margin") or None, covered=covered)
            if ch_on:
                ex = self.chamber_ui.adjust_exclusion(ex)
            out[ex] = (out[ex] * 0.35).astype(np.uint8)
        for m in self.canvas.masks:
            cv2.polylines(out, [np.round(np.asarray(m)).astype(np.int32)], True, (200, 200, 200), 1, cv2.LINE_AA)
        for name, pts in zones.items():
            c = self.canvas.color_of(name)
            col = (c.blue(), c.green(), c.red())
            cv2.polylines(out, outline_arrays(pts), True, col, max(1, int(round(max(out.shape[:2]) / 1100.0))),
                          cv2.LINE_AA)
            cx, cy = label_point(pts)
            fs = max(0.45, min(out.shape[:2]) / 1100.0)
            cv2.putText(out, name, (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(out, name, (int(cx) - 30, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, fs, col, 1, cv2.LINE_AA)
        return self.chamber_ui.overlay(out)

    def _show_tracking_frame(self):
        if self.cur_frame is None:
            return
        if self._preview is not None and self._preview.get("video") == self.player.path:
            self._run_preview_detection()
        else:
            self._set_track_view(self._overlay_zones_bgr(self.cur_frame))

    # --- detection preview
    def _preview_key(self, setup):
        return json.dumps({k: (np.asarray(v).round(4).tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in setup.items()
                           if k in ("video_path", "matrix", "warp_w", "warp_h", "roi_points", "mask_points",
                                    "auto_mask_outside_zones", "color_mode", "animal_polarity",
                                    "tracking_algorithm", "start_time", "end_time", "auto_mask_margin",
                                    "low_light", "covered_zones", "ldb_dark_covered", "test_type")},
                          sort_keys=True, default=str)

    def test_detection(self):
        if self.chamber_ui.enabled:
            self.chamber_ui.test_detection()            # every chamber at once
            return
        if not self._validate_ready(quiet_zones=True):
            return
        setup = self._build_setup(self.player.path, (self.player.w, self.player.h),
                                  start=float(self.spin_start.value()), end=float(self.spin_end.value()))
        key = self._preview_key(setup)
        if self._preview is not None and self._preview.get("key") == key:
            self._preview_from_start()
            return
        if self._preview_busy:
            return
        self._preview_busy = True
        self.btn_test.setEnabled(False)
        self.btn_test.setText("Building background…")
        path = self.player.path

        def work():
            return self._build_preview(setup, key, path)

        def done(res):
            self._preview_busy = False
            self.btn_test.setEnabled(True)
            self.btn_test.setText(TEST_BTN_TEXT)
            self._preview = res
            self._preview_from_start()
            self._check_polarity(res.get("bgm"))

        def failed(msg):
            self._preview_busy = False
            self.btn_test.setEnabled(True)
            self.btn_test.setText(TEST_BTN_TEXT)
            QMessageBox.warning(self.win, "Test detection", msg)
        self._thread_run(work, done, failed)

    @staticmethod
    def _build_preview(setup, key, path):
        """Background model of the analysis window, as 'Test detection' uses it (worker thread, no Qt)."""
        from tracking.location import make_tracking_engine, build_detection_exclusion, covered_zone_names
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            s = dict(setup)
            s["_fps"] = fps
            s["background_samples"] = min(60, int(s.get("background_samples", 100)))
            exclusion = None
            covered = covered_zone_names(s)
            if s.get("mask_points") or (s.get("auto_mask_outside_zones") and s.get("roi_points")) or covered:
                exclusion = build_detection_exclusion(s["warp_w"], s["warp_h"], s.get("mask_points"),
                                                      s.get("roi_points"), s.get("auto_mask_outside_zones"),
                                                      margin=s.get("auto_mask_margin"), covered=covered)
                if exclusion.all():
                    exclusion = None
            M = s["matrix"]
            warp = s.get("use_crop")

            def read(i):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if not ok:
                    return None
                return cv2.warpPerspective(f, M, (s["warp_w"], s["warp_h"])) if warp else f
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            f0 = int(round(s["start_time"] * fps))
            f1 = min(n - 1, int(round(min(s["end_time"], n / fps) * fps)))
            _tr, bgm = make_tracking_engine(s, s["warp_w"], s["warp_h"], read, f0, max(f0, f1), exclusion)
            return {"key": key, "bgm": bgm, "exclusion": exclusion, "setup": s, "video": path,
                    "f0": f0, "f1": max(f0, f1), "read": None}
        finally:
            cap.release()

    def _keep_preview(self, pv, **changes):
        """A slider moved because of a measurement: keep the background (it does not depend on it)."""
        if pv is None:
            return
        pv = dict(pv)
        pv["setup"] = dict(pv["setup"], **changes)
        pv.pop("tracker", None)
        self._preview = pv

    def _set_track_view(self, bgr):
        self._track_disp = bgr
        if self._measure_pts:
            bgr = self._draw_measure(bgr.copy(), self._measure_pts)
        self.view_track.set_bgr(bgr)

    # --- low light / dark box
    def _low_light_changed(self, *_a):
        if not self._chamber_view():
            self.cfg["low_light"] = self.cmb_low.currentData() or "off"
        self._disp_gamma_cache = None
        self._settings_changed()
        if self.cur_frame is not None and self.stack.currentIndex() == 3 and not self.tracking:
            self._show_tracking_frame()

    def _display_gamma(self, frame=None):
        """Low-light boost for the picture on the Tracking page (the same curve the tracker uses)."""
        from tracking.robust_tracker import resolve_gamma
        key = self.cmb_low.currentData() if hasattr(self, "cmb_low") else "off"
        if not key or key == "off":
            return 1.0
        pv = self._preview
        if pv is not None and pv.get("bgm") is not None and pv.get("video") == self.player.path:
            return float(getattr(pv["bgm"], "gamma", 1.0) or 1.0)
        if key != "auto":
            return resolve_gamma(key)
        if frame is None:
            return 1.0
        zones = {n: p for n, p in self.canvas.zones.items() if len(p) >= 3}
        cov = sorted(self.covered_names(zones))
        ck = (self.player.path, json.dumps({n: [list(map(float, q)) for q in p] for n, p in zones.items()},
                                           sort_keys=True), tuple(cov), frame.shape[:2])
        if self._disp_gamma_cache is not None and self._disp_gamma_cache[0] == ck:
            return self._disp_gamma_cache[1]
        h, w = frame.shape[:2]
        sc = min(1.0, 480.0 / max(h, w))
        small = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        masks = []
        for n, pts in zones.items():
            if n in cov:
                continue
            m = np.zeros(gray.shape, np.uint8)
            cv2.fillPoly(m, [np.round(np.asarray(pts, np.float64) * sc).astype(np.int32)], 1)
            masks.append(m > 0)
        g = resolve_gamma("auto", gray, None, masks)
        self._disp_gamma_cache = (ck, g)
        return g

    def _dark_zone_names(self, zones=None):
        zones = self.canvas.zones if zones is None else zones
        return [n for n in zones if any(w in str(n).lower() for w in ("dark", "black"))]

    def _sync_darkbox(self):
        """The 'Dark box' choice is shown for the light/dark box and follows the covered zones."""
        dark = self._dark_zone_names()
        show = self.test_key == "light_dark"          # only for the light / dark box test
        self.lbl_darkbox.setVisible(show)
        self.cmb_darkbox.setVisible(show)
        cov = self.covered_names()
        covered = bool(dark) and all(n in cov for n in dark) if dark else bool(self.cfg.get("ldb_dark_covered", True))
        self.cmb_darkbox.blockSignals(True)
        self.cmb_darkbox.setCurrentIndex(max(0, self.cmb_darkbox.findData("covered" if covered else "open")))
        self.cmb_darkbox.blockSignals(False)

    def _darkbox_changed(self, *_a):
        covered = self.cmb_darkbox.currentData() == "covered"
        self.cfg["ldb_dark_covered"] = covered
        if self.covered_override is not None:
            cur = set(self.covered_override)
            for n in self._dark_zone_names():
                (cur.add if covered else cur.discard)(n)
            self.covered_override = cur
        self.canvas.update()
        self._disp_gamma_cache = None
        self._settings_changed()
        msg = ("Dark box covered: the mouse is not tracked inside; its time there counts as 'in the dark box'."
               if covered else "Dark box open: the mouse is tracked inside the dark box too.")
        if not covered and (self.cmb_low.currentData() or "off") == "off":
            self.cmb_low.setCurrentIndex(max(0, self.cmb_low.findData("auto")))
            msg += " Low-light boost set to Auto (brightens the dark box only if it is dark)."
        self.status.showMessage(msg, 9000)
        if self.cur_frame is not None and self.stack.currentIndex() == 3 and not self.tracking:
            self._show_tracking_frame()

    # --- measure the mouse: 2 clicks (nose tip, tail base)
    def start_measure_animal(self):
        if self.player.cap is None or self.cur_frame is None:
            QMessageBox.information(self.win, "Measure the mouse", "Load a video first.")
            return
        if self.tracking:
            return
        self._stop_playback()
        self._measure_pts = []
        self.btn_measure.setText("👆  Click the NOSE tip…")
        self.status.showMessage("📏 Click the tip of the mouse's NOSE on the video, then the BASE of its TAIL. "
                                "Right-click = cancel.", 20000)
        self._show_tracking_frame()

    def _end_measure(self):
        self._measure_pts = None
        self.btn_measure.setText("📏  Measure the mouse")

    @staticmethod
    def _draw_measure(img, pts):
        s = max(1.0, min(img.shape[:2]) / 500.0)
        r = int(round(5 * s))
        if len(pts) == 2:
            a, b = (tuple(int(round(v)) for v in p) for p in pts)
            cv2.line(img, a, b, (0, 0, 0), int(round(4 * s)), cv2.LINE_AA)
            cv2.line(img, a, b, (0, 230, 255), max(1, int(round(2 * s))), cv2.LINE_AA)
        for k, p in enumerate(pts):
            c = (int(round(p[0])), int(round(p[1])))
            col = (60, 200, 60) if k == 0 else (0, 140, 255)
            cv2.circle(img, c, r + 2, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, c, r, col, -1, cv2.LINE_AA)
            txt = "Nose" if k == 0 else "Tail base"
            cv2.putText(img, txt, (c[0] + r + 4, c[1] - r), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, (0, 0, 0),
                        int(round(3 * s)), cv2.LINE_AA)
            cv2.putText(img, txt, (c[0] + r + 4, c[1] - r), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s, col,
                        max(1, int(round(1 * s))), cv2.LINE_AA)
        return img

    def _track_view_pressed(self, x, y, right):
        if self._measure_pts is None:
            return
        if right:
            self._end_measure()
            self.status.showMessage("Measuring cancelled.", 4000)
            self._show_tracking_frame()
            return
        self._measure_pts.append((float(x), float(y)))
        if len(self._measure_pts) == 1:
            self.btn_measure.setText("👆  …now the TAIL base")
            if self._track_disp is not None:
                self._set_track_view(self._track_disp)
            return
        pts = list(self._measure_pts)
        if self._track_disp is not None:
            self._set_track_view(self._track_disp)
        self._finish_measure(pts)

    def _finish_measure(self, pts):
        (x0, y0), (x1, y1) = pts
        L = math.hypot(x1 - x0, y1 - y0)
        self._end_measure()
        if L < 4:
            QMessageBox.information(self.win, "Measure the mouse", "The two points are too close. Press the button "
                                    "again and click the nose tip, then the tail base.")
            return
        pv = self._preview
        self.cfg["body_length_px"] = round(L, 1)
        body = 0.35 * L * L
        min_size = int(max(10, round(0.25 * body)))
        if min_size > self.sl_min.maximum():
            self.sl_min.setMaximum(int(min_size * 1.5))
        self.sl_min.setValue(min_size)
        save_json(SETTINGS_PATH, self.cfg)
        ppc = float(self.spin_ppc.value() or 0)
        cm = f" ≈ {L / ppc:.1f} cm" if ppc > 0.5 else ""
        txt = (f"Mouse length {L:.0f} px{cm} → body ≈ {body:.0f} px → minimum animal size set to {min_size} px "
               "(a quarter of the body).")
        con = self._contrast_on_segment(pv, pts) if pv is not None and pv.get("video") == self.player.path else None
        if con is not None:
            rec = int(round(min(40.0, max(6.0, 0.45 * con))))
            txt += (f" Here the mouse differs from the floor by ≈{con:.0f} grey levels → a minimum contrast of "
                    f"about {rec} suits this frame (🎯 Auto-set checks the whole video).")
        self.lbl_measure.setText(txt)
        self._keep_preview(pv, min_area=float(min_size), body_length_px=float(L))
        self.status.showMessage(f"📏 Mouse length {L:.0f} px — minimum animal size {min_size} px.", 10000)

    def _contrast_on_segment(self, pv, pts):
        """Median contrast of the mouse along the clicked nose -> tail-base line (current frame)."""
        try:
            from tracking.location import new_tracker
            tr = new_tracker(pv["setup"], pv["bgm"], pv["exclusion"])
            if not hasattr(tr, "strength") or self.cur_frame is None:
                return None
            d = tr.strength(self.cur_frame)
            s = float(tr.s)
            (x0, y0), (x1, y1) = pts
            vals = []
            for t in np.linspace(0.15, 0.85, 15):
                xi = int(round((x0 + t * (x1 - x0)) * s))
                yi = int(round((y0 + t * (y1 - y0)) * s))
                if 0 <= yi < d.shape[0] and 0 <= xi < d.shape[1]:
                    vals.append(float(d[yi, xi]))
            return float(np.median(vals)) if vals else None
        except Exception:
            return None

    # --- minimum contrast measured from the video
    def auto_contrast(self):
        if self.chamber_ui.enabled:
            self.chamber_ui.auto_contrast()             # measured in each chamber
            return
        if self.engine != "robust" or self._preview_busy:
            return
        if not self._validate_ready(quiet_zones=True):
            return
        setup = self._build_setup(self.player.path, (self.player.w, self.player.h),
                                  start=float(self.spin_start.value()), end=float(self.spin_end.value()))
        key = self._preview_key(setup)
        pv = self._preview if (self._preview is not None and self._preview.get("key") == key) else None
        path = self.player.path
        cur = float(setup.get("min_contrast", 12))
        self._preview_busy = True
        self.btn_auto_thr.setEnabled(False)
        self.btn_test.setEnabled(False)
        self.btn_auto_thr.setText("Checking the video…")

        def work():
            from tracking.location import measure_animal_contrast
            res = pv if pv is not None else self._build_preview(setup, key, path)
            s = dict(res["setup"], min_contrast=cur)
            cap = cv2.VideoCapture(path)
            frames = []
            try:
                M, warp = s["matrix"], s.get("use_crop")
                for i in np.linspace(res.get("f0", 0), res.get("f1", 0), 40).astype(int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                    ok, f = cap.read()
                    if ok:
                        frames.append(cv2.warpPerspective(f, M, (s["warp_w"], s["warp_h"])) if warp else f)
            finally:
                cap.release()
            m = measure_animal_contrast(s, res["bgm"], res["exclusion"], frames)
            return res, m

        def restore():
            self._preview_busy = False
            self.btn_auto_thr.setEnabled(self.engine == "robust")
            self.btn_test.setEnabled(True)
            self.btn_auto_thr.setText("🎯  Auto-set contrast")

        def done(out):
            restore()
            res, m = out
            self._preview = res
            rec = m.get("recommended")
            if rec is None:
                self.lbl_thr_hint.setText("The mouse could not be found clearly in the checked frames.")
                QMessageBox.information(
                    self.win, "Minimum contrast",
                    f"The mouse could not be found clearly in the checked frames (found in {m.get('found', 0)} of "
                    f"{m.get('frames', 0)}).\n\nCheck that the zones cover the arena, set the animal colour "
                    "(black mouse = 'Darker than floor'), and if part of the arena is dark turn on the "
                    "Low-light boost. Then press Auto-set again.")
                return
            n = int(m["frames"])
            boost = (f"\n• Low-light boost: gamma {m['gamma']:.1f} is on." if m.get("gamma", 1.0) > 1.001 else "")
            weak = ""
            if m.get("weak"):
                weak = ("\n\n⚠ At times the mouse is only a little different from the floor (a dark place?). "
                        + ("Turn on 'Low-light boost' (Auto) to brighten the dark zones."
                           if m.get("gamma", 1.0) <= 1.001 else "Try the 'Strong' low-light boost or more light."))
            msg = (f"Minimum contrast = how many grey levels a pixel must differ from the empty floor to count "
                   f"as 'mouse'.\n\nIn this video ({n} frames checked between the start and end time):\n"
                   f"• the mouse differs from the floor by ≈{m['peak_median']:.0f} grey levels, "
                   f"≈{m['peak_low']:.0f} in its weakest moments (dim places, shadow);\n"
                   f"• the video noise is ≈{m['sigma']:.1f} grey levels (below {m['noise_floor']} only noise "
                   f"would be picked up).{boost}\n\n"
                   f"Recommended: {rec}  (about half of the weakest mouse contrast, well above the noise).\n"
                   f"With {rec} the mouse is found in {m['found_with_rec']} of {n} checked frames; with your "
                   f"current {cur:.0f}: {m['found_with_current']} of {n}.{weak}")
            self.lbl_thr_hint.setText(f"Mouse contrast ≈{m['peak_median']:.0f} (weakest ≈{m['peak_low']:.0f}), "
                                      f"noise ≈{m['sigma']:.1f} → recommended {rec}.")
            box = QMessageBox(self.win)
            box.setWindowTitle("Minimum contrast")
            box.setIcon(QMessageBox.Information)
            box.setText(msg)
            use = box.addButton(f"Use {rec}", QMessageBox.AcceptRole)
            box.addButton(f"Keep {cur:.0f}", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is use:
                self.sl_thr.setValue(int(rec))
                self._keep_preview(res, min_contrast=float(rec))
                self.status.showMessage(f"Minimum contrast set to {rec}. Press 'Test detection' to check.", 8000)

        def failed(msg):
            restore()
            QMessageBox.warning(self.win, "Minimum contrast", msg)
        self._thread_run(work, done, failed)

    def _preview_from_start(self):
        """Test detection: a still picture of the Start time with the animal the run will find there (nothing is
        played or analysed; press play or scrub to check other moments)."""
        if self._preview is None or self.player.cap is None:
            return
        self._stop_playback()
        f0 = int(round(float(self.spin_start.value()) * (self.player.fps or 30.0)))
        f0 = min(max(0, f0), max(0, self.player.total - 1))
        self._preview.pop("tracker", None)
        self._preview["start_frame"] = f0
        if self.stack.currentIndex() != 3:
            self.go(3)
        self._show_frame(self.player.seek(f0))

    def _check_polarity(self, bgm):
        """The animal colour chosen by hand does not match the video: say so and offer the right one."""
        looks = getattr(bgm, "polarity_warning", None) if bgm is not None else None
        chosen = self.cmb_pol.currentData() if hasattr(self, "cmb_pol") else None
        if not looks or chosen not in ("dark", "light", "any") or looks == chosen:
            return
        from tracking.location import polarity_warning_text
        want = "Darker than floor (black mouse)" if looks == "dark" else "Lighter than floor (white mouse)"
        r = QMessageBox.question(self.win, "Animal colour looks wrong",
                                 polarity_warning_text(chosen, looks) + f"\n\nSwitch Animal colour to '{want}' now?")
        if r == QMessageBox.Yes:
            self.cmb_pol.setCurrentIndex(max(0, self.cmb_pol.findData(looks)))
            self.status.showMessage(f"Animal colour set to '{want}'. Press 'Test detection' to check again.", 10000)

    def _start_still(self, pv, idx):
        """The animal at the start frame exactly as the real run finds it: a fresh tracker looks ~1 s ahead
        from the start (like the run), the position found is put back on the start frame."""
        from tracking.location import new_tracker, apply_backfill, detection_still
        s = pv["setup"]
        tr = new_tracker(s, pv["bgm"], pv["exclusion"])
        n = int(getattr(tr, "first_window", 30) or 30) + int(round(1.5 * (self.player.fps or 30.0)))
        xs, ys, areas, status = [], [], [], []
        try:
            cap = pv.get("_warm_cap")
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.player.path)
                pv["_warm_cap"] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            for _k in range(n):
                ok, f = cap.read()
                if not ok:
                    break
                det = tr.update(self._warp(f))
                apply_backfill(det, xs, ys, areas, status)
                xs.append(det["x"]), ys.append(det["y"]), areas.append(det["area"]), status.append(det["status"])
                if status[0] == "Tracked":
                    break
        except Exception:
            pass
        x0, y0, st0 = (xs[0], ys[0], status[0]) if status else (float("nan"), float("nan"), "Lost")
        t = idx / self.player.fps if self.player.fps else 0.0
        disp = detection_still(s, pv["bgm"], pv["exclusion"], self.cur_frame, x0, y0, st0, t, idx,
                               s.get("roi_points", {}), s.get("mask_points", []), lost_label="NOT FOUND")
        return disp, {"x": x0, "y": y0, "status": st0, "area": areas[0] if areas else float("nan")}

    def _run_preview_detection(self):
        pv = self._preview
        if pv is None or self.cur_frame is None:
            return
        if pv.get("chambers") is not None:
            self.chamber_ui.preview_detection(pv)
            return
        from tracking.location import new_tracker, _draw_live
        s = pv["setup"]
        idx = self.player.current
        # playing on: keep following the animal like the real run; a jump (scrubbing) starts a fresh search
        tr = pv.get("tracker") if pv.get("tracker_at") == idx - 1 else None
        if tr is None and idx == pv.get("start_frame"):
            # the start time: a still picture of the animal as the run finds it there
            disp, det = self._start_still(pv, idx)
            pv["tracker"], pv["tracker_at"] = None, None
            self._set_track_view(disp)
            bgm = pv["bgm"]
            if det["status"] == "Tracked":
                self.status.showMessage(f"✓ Start time {fmt_clock_exact(idx / (self.player.fps or 30.0))}: animal found "
                                        f"at ({det['x']:.0f}, {det['y']:.0f}) (yellow = what is detected as the animal). "
                                        f"Background: {bgm.describe()}.{self._dark_hint(bgm)} Press play or scrub "
                                        "to check other moments.", 20000)
            else:
                self.status.showMessage("✗ No animal found at the start time. Try 🎯 Auto-set contrast, set the "
                                        f"animal colour, or check the zones / masks.{self._dark_hint(bgm)} "
                                        f"Background: {bgm.describe()}", 20000)
            return
        if tr is None:
            tr = new_tracker(s, pv["bgm"], pv["exclusion"])
            self._warm_up_tracker(pv, tr, idx)
        pv["tracker"], pv["tracker_at"] = tr, idx
        frame = self.cur_frame
        det = tr.update(frame)
        t = self.player.current / self.player.fps if self.player.fps else 0.0
        disp = _draw_live(frame, s.get("roi_points", {}), s.get("mask_points", []), pv["exclusion"], det, tr, t,
                          self.player.current)
        self._set_track_view(disp)
        bgm = pv["bgm"]
        if det["status"] == "Tracked":
            exp = (f", expected ≈{bgm.animal_area / bgm.scale ** 2:.0f}" if getattr(bgm, "animal_area", None) else "")
            con = (f", contrast {det['peak']:.0f} (minimum {float(s.get('min_contrast', 12)):.0f})"
                   if det.get("peak") is not None and self.engine == "robust" else "")
            self.status.showMessage(f"✓ Animal found at ({det['x']:.0f}, {det['y']:.0f}) — size {det['area']:.0f} px"
                                    f"{exp}{con}. Background: {bgm.describe()}.{self._dark_hint(bgm)} Scrub the "
                                    "slider to check other frames.", 15000)
        else:
            self.status.showMessage("✗ No animal found on this frame. Try a lower minimum contrast (🎯 Auto-set), "
                                    "set the animal colour, or check the zones / masks."
                                    f"{self._dark_hint(bgm)} Background: {bgm.describe()}", 15000)

    @staticmethod
    def _dark_hint(bgm):
        lvl = getattr(bgm, "dark_level", None)
        if lvl is not None and lvl < 45 and float(getattr(bgm, "gamma", 1.0) or 1.0) <= 1.001:
            return f" ⚠ A zone is dark (floor ≈{lvl:.0f} grey levels): try Low-light boost = Auto."
        return ""

    def _warm_up_tracker(self, pv, tr, idx):
        """A fresh tracker looks ~1 s ahead before it takes the first lock (so a still shadow cannot win):
        when the preview jumps to a frame, the frames just before it are run first, like the real run does."""
        n = int(getattr(tr, "first_window", 0) or 0) + 6
        f0 = int(round(float(pv["setup"].get("start_time", 0.0)) * (self.player.fps or 30.0)))
        a = max(0, f0, idx - n)
        if n <= 6 or a >= idx or self.player.path is None:
            return
        try:
            cap = pv.get("_warm_cap")
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.player.path)
                pv["_warm_cap"] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            for _i in range(a, idx):
                ok, f = cap.read()
                if not ok:
                    break
                tr.update(self._warp(f))
        except Exception:
            pass

    # --- setup dict for the engine
    def _build_setup(self, video_path, size, start=0.0, end=None, item=None):
        """Backend `setup` dict for one video (size = that video's (w, h)).

        A batch item may carry its own zones / masks / crop; otherwise the
        Video-Setup layout is used.  The crop applies only to videos with the
        same resolution as the one it was drawn on; otherwise the zones are
        rescaled onto the full frame."""
        vw, vh = size
        own = item or {}
        zones = own.get("zones") if own.get("zones") else self.canvas.zones
        masks = own.get("masks") if own.get("zones") else self.canvas.masks
        crop = own.get("crop") if own.get("zones") else self.crop
        ref_size = tuple(own.get("ref_size") or (self.player.w, self.player.h))
        if crop and (vw, vh) == ref_size:
            matrix, w, h, use_crop, sx, sy = crop["matrix"], crop["w"], crop["h"], True, 1.0, 1.0
        else:
            if crop:          # crop drawn on another resolution: map zones back to the raw frame, then rescale
                Minv = np.asarray(crop["inv"], np.float64)

                def back(p):
                    v = Minv @ np.array([p[0], p[1], 1.0])
                    return (v[0] / v[2], v[1] / v[2])
                zones = {k: [back(p) for p in v] for k, v in zones.items()}
                masks = [[back(p) for p in m] for m in masks]
            matrix, w, h, use_crop = np.eye(3, dtype=np.float32), vw, vh, False
            sx, sy = vw / max(1, ref_size[0]), vh / max(1, ref_size[1])
        roi_points = {n: [(float(x * sx), float(y * sy)) for x, y in pts] for n, pts in zones.items()}
        mask_points = [[(float(x * sx), float(y * sy)) for x, y in m] for m in masks]
        ppc = max(float(own.get("ppc") or 0.0) or self.spin_ppc.value(), 1e-6)
        smooth = self.sl_smooth.value()
        span = max((end if end is not None else 0.0) - start, 1.0)
        bin_s = float(self.cfg["bin_size_s"])
        while span / bin_s < 3 and bin_s > 5:
            bin_s = max(5.0, bin_s / 2)
        eff = self._effective(own)            # a batch video may have its own engine / detection settings
        n_an = max(1, min(3, int(eff["animals"])))
        automask = self.chk_automask.isChecked() if own.get("automask") is None else bool(own["automask"])
        desc_w = self.win.findChild(QPlainTextEdit, "teDescription")
        date_w = self.win.findChild(QDateEdit, "deDate")
        return {
            "video_path": video_path, "use_crop": use_crop, "matrix": matrix,
            "warp_w": int(w), "warp_h": int(h), "arena": (0, 0, int(w), int(h)),
            "roi_names": list(roi_points), "roi_points": roi_points, "mask_points": mask_points,
            "auto_mask_outside_zones": bool(automask) and bool(roi_points),
            "auto_mask_margin": int(self.cfg.get("auto_mask_margin") or 0) or None,
            "object_names": [], "object_points": {}, "interaction_margin_px": 0.0,
            "background_samples": int(self.cfg["bg_samples"]),
            "tracking_algorithm": eff["engine"],
            "animal_polarity": eff["polarity"],
            "min_contrast": float(self.cfg.get("min_contrast", 12)),
            "loc_thresh": float(self.cfg.get("loc_thresh", 99.5)),
            "use_window": bool(self.chk_window.isChecked()),
            "window_size": int(self.spin_window.value()),
            "window_weight": float(self.spin_window_weight.value()),
            "min_area": float(self.sl_min.value()), "max_area": float(self.cfg.get("max_area") or 0),
            "max_jump": float(self.cfg["max_jump"]),
            "max_gap_s": float(self.cfg.get("max_gap_s", 0.5)),
            "min_move_px": (float(self.cfg["min_move_px"]) if float(self.cfg.get("min_move_px") or 0) > 0 else None),
            "entry_rule": str(self.cfg.get("entry_rule", "paws")),
            "tracking_point": "centre",                     # body centre only
            "hold_lost": bool(self.cfg.get("hold_lost", True)),
            "adapt_background": bool(self.cfg.get("adapt_background", True)),
            "low_light": str(self.cfg.get("low_light", "off") or "off"),
            "body_length_px": (float(self.cfg.get("body_length_px") or 0) * math.sqrt(sx * sy)) or None,
            "ldb_dark_covered": bool(self.cfg.get("ldb_dark_covered", True)),
            "covered_zones": (None if self.covered_override is None
                              else [n for n in roi_points if n in self.covered_override]),
            "embed_video_px": int(self.cfg.get("embed_video_px", 640)),
            "entry_body_pct": float(self.cfg.get("entry_body_pct", 80)),
            "scale_factor": 1.0 / ppc, "scale_unit": "cm", "preview_samples": int(self.cfg.get("preview_samples", 3)),
            "color_mode": eff["color_mode"],
            "start_time": float(start), "end_time": float(end if end is not None else 1e9),
            "output_dir_override": None, "test_type": self.test_key,
            "track_nose": self.test_key == "nor" and self.cfg.get("exploration_point", "nose") == "nose",
            "compute_arm_entries": self.test_key in ("epm", "y_maze", "t_maze"),
            "compute_alternation": self.test_key in ("y_maze", "t_maze"),
            "stop_condition": None, "stop_value": None, "stop_zone_name": "", "zone_groups": {},
            "smooth_window": smooth if smooth >= 2 else None, "custom_variables": [],
            "bin_size_s": bin_s,
            "tracking_mode": "multi" if n_an >= 2 else "single", "num_animals": n_an,
            "experimental_design": list(self.experimental_design),
            "experiment_name": self.le_name.text().strip(),
            "investigator": self.le_investigator.text().strip(),
            "description": desc_w.toPlainText().strip() if desc_w is not None else "",
            "notes": self.le_notes.text().strip(),
            "experiment_date": date_w.date().toString("yyyy-MM-dd") if date_w is not None else "",
            "subject": (own.get("subject") or "") if item is not None else self.current_subject(),
        }

    def _validate_ready(self, quiet_zones=False):
        if self.player.cap is None:
            QMessageBox.information(self.win, "Tracking", "Load a video first (Experiment → Video File(s)).")
            self.go(1)
            return False
        if self.canvas.tool is not None:
            self.canvas.cancel_tool(silent=True)
        if not self.canvas.zones and not quiet_zones:
            if QMessageBox.question(self.win, "No zones", "No zones are defined. Track anyway (distance and speed "
                                    "only)?") != QMessageBox.Yes:
                self.go(2)
                return False
        return True

    def import_pose_file(self):
        """Analyse this video with the positions of a DeepLabCut / SLEAP CSV file (no BT tracking)."""
        if self.tracking:
            return
        if not self._validate_ready(quiet_zones=self.chamber_ui.enabled):
            return
        if self.chamber_ui.enabled:
            QMessageBox.information(self.win, "Pose file", "Pose files are used for one animal per video "
                                                          "(turn Chambers off).")
            return
        fn, _ = QFileDialog.getOpenFileName(self.win, "Positions from DeepLabCut / SLEAP",
                                            str(Path(self.player.path).parent), "Pose CSV (*.csv);;All files (*)")
        if not fn:
            return
        try:
            from tracking.pose_import import read_pose_csv
            pose = read_pose_csv(fn)
        except Exception as exc:
            QMessageBox.warning(self.win, "Pose file", f"This file could not be read:\n{exc}")
            return
        n = len(next(iter(pose["parts"].values()))[0]) if pose["parts"] else 0
        if n and self.player.total and abs(n - self.player.total) > max(5, 0.02 * self.player.total):
            if QMessageBox.question(self.win, "Pose file", f"The pose file has {n} frames, the video "
                                    f"{self.player.total}. Use it anyway?") != QMessageBox.Yes:
                return
        self.start_or_stop_tracking(pose_file=fn)

    def start_or_stop_tracking(self, pose_file=None):
        if self.tracking:
            self._stop_worker()
            self.btn_start.setEnabled(False)
            self.btn_start.setText("Stopping…")
            return
        self.chamber_ui.to_all()                     # the All settings (each chamber adds its own on top)
        if not self._validate_ready(quiet_zones=self.chamber_ui.enabled):
            return
        self._stop_playback()
        start_t, end_t = float(self.spin_start.value()), float(self.spin_end.value())
        if end_t <= start_t:
            QMessageBox.warning(self.win, "Tracking", "End time must be greater than Start time.")
            return
        if not self.ensure_results_folder():
            return
        setup = self._build_setup(self.player.path, (self.player.w, self.player.h), start=start_t, end=end_t)
        if not self.cfg.get("fast_tracking"):
            # frame by frame in order: the live picture, its frame number and the clock below move together
            setup["workers"] = 1
        if self.canvas.zones and not self.chamber_ui.enabled:     # ⚡ Auto learns this set-up for the next videos
            key = self._bg_key()
            img = self._bg_cache.get(key) if isinstance(getattr(self, "_bg_cache", None), dict) else None
            self._learn_layout(img if img is not None else self.cur_frame, self.canvas.zones,
                               Path(self.player.path).name)
        self.run_kind = "single"
        self._single_subject = self.current_subject()
        self._single_error = None
        self.last_summary = None
        self.tracking = True
        self.btn_start.setText("■   Stop Tracking")
        self.btn_test.setEnabled(False)
        self.sl_track.setEnabled(False)
        self.sl_track.blockSignals(True)
        self.sl_track.setRange(0, max(self.player.total - 1, 0))
        self.sl_track.setValue(int(round(start_t * (self.player.fps or 30.0))))
        self.sl_track.blockSignals(False)
        fps_ = self.player.fps or 30.0
        self.lbl_time_track.setText(fmt_track_pos(start_t, int(round(start_t * fps_)), end_t,
                                                  min(int(round(end_t * fps_)), max(0, self.player.total - 1))))
        self._lock_ui(True)
        self.go(3)
        if self.chamber_ui.enabled:
            job = self.chamber_ui.tracking_job(self.player.path, setup)
            self.status.showMessage(f"Tracking {len(job[3])} chambers at once… building each chamber's background, "
                                    "then following every animal.")
            self._start_worker([job], emit_frames=True)
            return
        if pose_file:
            setup["pose_file"] = str(pose_file)
            setup["track_nose"] = True
            self.status.showMessage(f"Analysing with the positions of {Path(pose_file).name}…")
            self._start_worker([(self.player.path, setup, "pose")], emit_frames=False)
            return
        self.status.showMessage("Tracking… building the animal-free background, then following the animal.")
        self._start_worker([(self.player.path, setup, setup.get("tracking_mode", "single"))], emit_frames=True)

    def _lock_ui(self, busy):
        for n in ("btnNavExperiment", "btnNavVideoSetup", "btnNavDashboard"):
            self.nav[n].setEnabled(not busy)
        if hasattr(self, "btn_live"):
            self.btn_live.setEnabled(not busy)

    # --- worker plumbing
    def _start_worker(self, jobs, emit_frames):
        self.thread = QThread()
        self.worker = TrackWorker(jobs, emit_frames)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.frame.connect(self._on_live_frame)
        self.worker.frame_pos.connect(self._on_frame_pos)
        self.worker.job_started.connect(self._on_job_started)
        self.worker.job_progress.connect(self._on_job_progress)
        self.worker.job_done.connect(self._on_job_done)
        self.worker.job_failed.connect(self._on_job_failed)
        self.worker.job_note.connect(self._on_job_note)
        self.worker.all_done.connect(self._on_all_done)
        self.worker.all_done.connect(self.thread.quit)
        self.thread.finished.connect(self._cleanup_thread)
        self.thread.start()

    def _stop_worker(self, wait=False):
        if self.worker is not None:
            self.worker.stop()
        if wait and self.thread is not None:
            self.thread.quit()
            self.thread.wait(8000)

    def _cleanup_thread(self):
        if self.thread is not None:
            self.thread.deleteLater()
        self.thread = None
        self.worker = None

    def _on_frame_pos(self, frame_number, t):
        """The player under the live picture shows the same frame and time as the picture."""
        if self.run_kind != "single":
            return
        self.sl_track.blockSignals(True)
        self.sl_track.setValue(int(frame_number))
        self.sl_track.blockSignals(False)
        end_t = float(self.spin_end.value())
        end_f = min(int(round(end_t * (self.player.fps or 30.0))), max(0, self.player.total - 1))
        self.lbl_time_track.setText(fmt_track_pos(t, frame_number, end_t, end_f))

    def _on_live_frame(self, img):
        if self.run_kind == "single":
            self.view_track.set_qimage(img)
        elif self.run_kind == "batch" and self._batch_cur_path:
            self._batch_frames[self._batch_cur_path] = img
            if self.live_dlg is not None:
                self.live_dlg.on_frame(self._batch_cur_path, img)

    def _live_visible(self):
        return self.live_dlg is not None and self.live_dlg.isVisible()

    def _sync_live_rate(self):
        """Fast live frames while the batch live view is open, a slow trickle (last frame per video) otherwise."""
        if self.worker is not None and self.run_kind == "batch":
            self.worker.frame_interval = 0.06 if self._live_visible() else 0.5

    def open_batch_live(self, row=None):
        """Batch ▸ watch a video's tracking live (row = batch table row; None = selected / running video)."""
        if not self.batch_items:
            self.status.showMessage("Add videos to the batch queue first.", 4000)
            return
        if row is None:
            rows = self._batch_selected_rows()
            path = (self.batch_items[rows[0]]["path"] if rows and rows[0] < len(self.batch_items)
                    else self._batch_cur_path or self.batch_items[0]["path"])
        else:
            path = self.batch_items[int(min(max(0, row), len(self.batch_items) - 1))]["path"]
        if self.live_dlg is None:
            self.live_dlg = LiveBatchDialog(self)
        self.live_dlg.chk_follow.blockSignals(True)
        self.live_dlg.chk_follow.setChecked(bool(self._batch_cur_path) and path == self._batch_cur_path)
        self.live_dlg.chk_follow.blockSignals(False)
        self.live_dlg.show_path(path)
        self.live_dlg.show()
        self.live_dlg.raise_()
        self.live_dlg.activateWindow()
        self._sync_live_rate()

    def _batch_cell_clicked(self, r, c):
        """While a batch runs, clicking a video opens its live view; afterwards the Status cell does."""
        if not (0 <= r < len(self.batch_items)):
            return
        running = self.tracking and self.run_kind == "batch"
        if running or (c == 6 and self.batch_items[r]["path"] in self._batch_frames):
            self.open_batch_live(r)

    def _row(self, i):
        return self._row_map[i] if self.run_kind == "batch" else i

    def _on_job_started(self, i):
        self._job_t0, self._job_frames_t0, self._job_msg_t = time.time(), None, 0.0
        if self.run_kind == "batch":
            r = self._row(i)
            self._set_batch_row(r, "Processing", 0)
            self._batch_cur_path = self.batch_items[r]["path"]
            if self._live_visible():
                self.live_dlg.on_job_started(self._batch_cur_path)

    def _on_job_progress(self, i, frac):
        if self.run_kind == "single":
            # the slider + clock follow the live picture (_on_frame_pos); here only the status line
            t0, t1 = float(self.spin_start.value()), float(self.spin_end.value())
            span = max(0.0, t1 - t0)
            # 0-8 % = background model, 8-92 % = frames, 92-100 % = saving
            rel = min(1.0, max(0.0, (frac - 0.08) / 0.84))
            self._speed_status(frac, rel, span)
        else:
            r = self._row(i)
            self._set_batch_row(r, "Processing", int(frac * 100))
            self._update_batch_overall(r, frac)

    def _speed_status(self, frac, rel, span):
        """Status line while one video is tracked: phase, speed (x real time) and time left."""
        now = time.time()
        if now - getattr(self, "_job_msg_t", 0.0) < 1.0:
            return
        self._job_msg_t = now
        if frac < 0.08:
            self.status.showMessage(f"Preparing — building the animal-free background ({frac / 0.08 * 100:.0f} %)…")
            return
        if getattr(self, "_job_frames_t0", None) is None:
            self._job_frames_t0, self._job_rel0 = now, rel
            return
        el = now - self._job_frames_t0
        done = rel - self._job_rel0
        if frac >= 0.92:
            self.status.showMessage("Saving the results (Excel, CSV, charts)…")
        elif el > 2 and done > 0:
            left = (1.0 - rel) * el / done
            speed = done * span / el if span else 0.0
            self.status.showMessage(f"Tracking {rel * 100:.0f} %  ·  {speed:.1f}× real time  ·  about "
                                    f"{fmt_duration(left)} left")

    def _on_job_note(self, i, msg):
        if self.run_kind == "batch":
            r = self._row(i)
            if 0 <= r < len(self.batch_items):
                self.batch_items[r]["note"] = msg
                it = self.tbl_batch.item(r, 2)
                if it is not None:
                    it.setToolTip(msg)
        self.status.showMessage(msg, 8000)

    def _add_history_all(self, summary, video_path, subject):
        """History: one entry per animal (a chamber run has one per chamber)."""
        if summary.get("Chambers"):
            for sm in summary["Chambers"]:
                self.add_history(sm, video_path, sm.get("Animal_ID") or subject)
        else:
            self.add_history(summary, video_path, subject)

    def _on_job_done(self, i, summary):
        if self.run_kind == "single":
            self.last_summary = summary
            self._add_history_all(summary, self.player.path, self._single_subject)
        else:
            r = self._row(i)
            summary["Video_name"] = Path(self.batch_items[r]["path"]).name
            self.batch_items[r]["summary"] = summary
            warn = summary.get("Polarity_warning") or next(
                (f"C{sm.get('Chamber')}: {sm['Polarity_warning']}" for sm in summary.get("Chambers") or []
                 if sm.get("Polarity_warning")), None)
            if warn:
                self.batch_items[r]["note"] = "⚠ " + str(warn)
                it = self.tbl_batch.item(r, 2)
                if it is not None:
                    it.setToolTip(self.batch_items[r]["note"])
            self._set_batch_row(r, "Completed", 100)      # also shows this video's 📂 Open / Export buttons
            self._add_history_all(summary, self.batch_items[r]["path"], self.batch_items[r].get("subject", ""))
            self._update_batch_overall(r, 1.0, done=True)

    def _on_job_failed(self, i, msg):
        if self.run_kind == "batch":
            r = self._row(i)
            self._set_batch_row(r, "Stopped" if msg == "Stopped" else "Failed", None)
            self.batch_items[r]["error"] = msg
            it = self.tbl_batch.item(r, 6)
            if it is not None:
                it.setToolTip(msg)
        else:
            self._single_error = msg

    def _on_all_done(self, cancelled):
        if self.run_kind == "single":
            self.tracking = False
            self._lock_ui(False)
            self.btn_start.setEnabled(True)
            self.btn_test.setEnabled(True)
            self.btn_start.setText("▶   Start Tracking")
            self.sl_track.setEnabled(True)
            self.sl_track.blockSignals(True)
            self.sl_track.setRange(0, max(self.player.total - 1, 0))
            self.sl_track.setValue(self.player.current)
            self.sl_track.blockSignals(False)
            self._update_time_labels()
            err = self._single_error
            self._single_error = None
            if cancelled:
                self.status.showMessage("Tracking stopped — no results were saved.", 8000)
                self._refresh_views()
            elif err or self.last_summary is None:
                QMessageBox.critical(self.win, "Tracking failed", err or "No tracking data were produced.")
                self.status.showMessage("Tracking failed.", 8000)
            elif self.last_summary.get("Chambers"):
                run = self.last_summary
                self.chamber_ui.show_run(run, self.player.path)       # the Analysis page shows chamber 1
                parts = []
                for sm in run["Chambers"]:
                    q = sm.get("Tracking_quality_percent")
                    parts.append(f"C{sm.get('Chamber')}{' · ' + sm['Animal_ID'] if sm.get('Animal_ID') else ''} "
                                 + (f"{q:.1f} %" if isinstance(q, (int, float)) else "done"))
                warn = [f"C{sm.get('Chamber')}" for sm in run["Chambers"] if sm.get("Polarity_warning")]
                self.status.showMessage(
                    f"Tracking complete — {len(parts)} chambers (animal found: " + ", ".join(parts) + "). "
                    "Pick a chamber under 'Showing'. Summary of all animals: "
                    f"{Path(run.get('Chamber_summary_xlsx') or '').name or '—'}"
                    + (f"  ⚠ Animal colour looks wrong in {', '.join(warn)} — set it for that chamber." if warn
                       else ""), 20000)
            else:
                self.chamber_ui.results = []              # a one-animal run: the chamber list of an earlier run goes
                self.show_results(self.last_summary)
                if self.last_summary.get("Project_file"):
                    self.open_bta(self.last_summary["Project_file"], show=False)
                self._refresh_showing()
                self.go(4)
                q = self.last_summary.get("Tracking_quality_percent")
                self.status.showMessage(f"Tracking complete — animal detected in {q:.1f} % of frames"
                                        f" ({self.last_summary.get('Interpolated_frames', 0)} short gaps filled)."
                                        if isinstance(q, (int, float)) else "Tracking complete.", 12000)
                warn = self.last_summary.get("Polarity_warning")
                if warn:
                    looks = "dark" if "looks DARKER" in str(warn) else "light"
                    want = "Darker than floor (black mouse)" if looks == "dark" else "Lighter than floor (white mouse)"
                    if QMessageBox.question(self.win, "Animal colour looks wrong",
                                            f"{warn}\n\nSwitch Animal colour to '{want}' for the next run?") \
                            == QMessageBox.Yes:
                        self.cmb_pol.setCurrentIndex(max(0, self.cmb_pol.findData(looks)))
        else:
            self._finish_batch(cancelled)

    # ------------------------------------------------------------------ analysis page
    KPI_BY_TEST = {
        "open_field": [("Center_time_pct", "Time in Centre", "{:.1f} %"), ("Anxiety_index", "Anxiety Index", "{:.3f}")],
        "nor": [("Discrimination_index", "Discrimination Index", "{:+.3f}"), ("Discrimination_ratio", "Discrimination Ratio", "{:.3f}")],
        "epm": [("Open_arm_time_pct", "Open-Arm Time", "{:.1f} %"), ("Anxiety_index", "Anxiety Index", "{:.3f}")],
        "light_dark": [("Light_time_pct", "Time in Light", "{:.1f} %"), ("Anxiety_index", "Anxiety Index", "{:.3f}")],
        "y_maze": [("Spontaneous_alternation_pct", "Spontaneous Alternation", "{:.1f} %"), ("Total_arm_entries", "Arm Entries", "{:.0f}")],
        "t_maze": [("Spontaneous_alternation_pct", "Spontaneous Alternation", "{:.1f} %"), ("Total_arm_entries", "Arm Entries", "{:.0f}")],
        "social": [("Sociability_index", "Sociability Index", "{:+.2f}"), ("Chamber_entries", "Chamber Entries", "{:.0f}")],
        "custom": [(None, "Time in first zone", "{:.1f} %"), ("Total_transitions", "Zone Transitions", "{:.0f}")],
    }

    def _bind_analysis(self):
        w = self.win
        self.analysis_tabs = find(w, QTabWidget, "analysisTabs")
        self.views = {n: FrameView(find(w, QLabel, n), self) for n in
                      ("plotTrajectory", "plotHeatmap", "plotActivity", "plotTabTrackPlot", "plotTabHeatmap",
                       "plotTabZoneAnalysis", "plotTabTimeBins")}
        # the design's "Behavior Events" tab becomes "Zone Events": metrics + zone table + entries + transitions
        self.tab_events = find(w, QWidget, "tabBehaviorEvents")
        self.analysis_tabs.setTabText(self.analysis_tabs.indexOf(self.tab_events), "Zone Events")
        lbl = find(w, QLabel, "plotTabBehaviorEvents")
        self.event_tabs = QTabWidget()
        self.event_tables = {}
        for key, title in (("metrics", "Test measures"), ("zones", "Zone summary"), ("entries", "Arm / zone entries"),
                           ("transitions", "Transitions"), ("quality", "Tracking quality")):
            t = QTableWidget()
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            t.setAlternatingRowColors(True)
            t.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.event_tables[key] = t
            self.event_tabs.addTab(t, title)
        lay = lbl.parentWidget().layout()
        lay.replaceWidget(lbl, self.event_tabs)
        lbl.hide()
        from pro_ui.frame_review import FrameReviewTab
        self.review = FrameReviewTab(self)
        self.analysis_tabs.insertTab(self.analysis_tabs.indexOf(self.tab_events) + 1, self.review, "Frame Review")
        self.bta = None
        self._build_statistics_tab()
        self.btn_export = find(w, QPushButton, "btnExport")
        menu = QMenu(self.btn_export)
        menu.addAction("📂 Open BT file (.bta / .btf)…", self.open_bt_file_dialog)
        menu.addAction("🧾 Audit trail of these results…", lambda: self.review.show_audit())
        menu.addSeparator()
        for text, fn in (("Excel report (.xlsx)", lambda: self.export("xlsx")),
                         ("Zone summary + entries (.csv)", lambda: self.export("zones")),
                         ("Raw tracking data (.csv)", lambda: self.export("csv")),
                         ("All charts (.png)", lambda: self.export("png")),
                         ("Everything (.zip)", lambda: self.export("zip")),
                         (None, None),
                         ("Advanced statistics (.xlsx)", self.export_advanced_statistics),
                         ("Open results folder", self.open_output_folder)):
            if text is None:
                menu.addSeparator()
            else:
                menu.addAction(text, fn)
        self.btn_export.setMenu(menu)
        self.export_menu_tracking = menu
        self.btn_reset_results = QToolButton(self.btn_export.parentWidget())
        self.btn_reset_results.setText("↻")
        self.btn_reset_results.setToolTip("Clear these results and the tracking view, ready for a new run "
                                          "(saved files are kept).")
        self.btn_reset_results.setCursor(Qt.PointingHandCursor)
        self.btn_reset_results.setStyleSheet("QToolButton { font-size: 15pt; border: 1px solid #d5dce8; "
                                             "border-radius: 6px; padding: 2px 10px; background: #ffffff; } "
                                             "QToolButton:hover { background: #f3f6fb; }")
        self.btn_reset_results.clicked.connect(lambda: self.reset_tracking_results(settings=False))
        self.btn_open_bta = QPushButton("📂 Open BT file", self.btn_export.parentWidget())
        self.btn_open_bta.setProperty("cssClass", "secondary")
        self.btn_open_bta.setMinimumHeight(self.btn_export.sizeHint().height())
        self.btn_open_bta.setToolTip("Open a saved BT file: a tracking analysis (.bta) or a freezing analysis "
                                     "(.btf) — its results, the frame-by-frame review and the audit trail.")
        self.btn_open_bta.clicked.connect(self.open_bt_file_dialog)
        el = self._layout_with(find(w, QWidget, "pageAnalysis").layout(), self.btn_export)
        if el is not None:
            el.insertWidget(el.indexOf(self.btn_export), self.btn_reset_results)
            el.insertWidget(el.indexOf(self.btn_reset_results), self.btn_open_bta)
        self._bind_freezing_results(el)
        self._reset_analysis()

    # ------------------------------------------------------------------ freezing results (Analysis & Results)
    def _bind_freezing_results(self, header):
        """A freezing result is shown in its own panel in place of the tracking tabs; 'Showing' switches."""
        from collections import OrderedDict
        from pro_ui.freezing_results import FreezingResultsPanel
        self.results_mode = "tracking"
        self.freeze_results = OrderedDict()           # key -> entry (the latest 12)
        self.freeze_key = None
        self.freeze_panel = FreezingResultsPanel(self)
        self.freeze_panel.hide()
        pl = self._layout_with(find(self.win, QWidget, "pageAnalysis").layout(), self.analysis_tabs)
        if pl is not None:
            pl.insertWidget(pl.indexOf(self.analysis_tabs) + 1, self.freeze_panel, 1)
        box = QFrame()
        box.setObjectName("showingBox")
        box.setStyleSheet("QFrame#showingBox{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;}"
                          "QFrame#showingBox QLabel{border:none;background:transparent;}")
        bl = QHBoxLayout(box)
        bl.setContentsMargins(8, 2, 8, 2)
        bl.addWidget(QLabel("Showing:"))
        self.cmb_showing = QComboBox()
        self.cmb_showing.setMinimumWidth(280)
        self.cmb_showing.setToolTip("Which result this page shows: the tracking result, or a freezing result.")
        self.cmb_showing.activated.connect(self._showing_picked)
        bl.addWidget(self.cmb_showing)
        box.hide()
        self.showing_box = box
        if header is not None:
            header.insertWidget(header.indexOf(self.btn_open_bta), box)
        m = QMenu(self.btn_export)
        m.addAction("📂 Open BT file (.bta / .btf)…", self.open_bt_file_dialog)
        m.addAction("🧾 Audit trail of this result…", self.show_freezing_audit)
        m.addSeparator()
        for text, kind in (("Excel report (.xlsx)", "xlsx"), ("One-page summary (.pdf)", "pdf"),
                           ("Frame-by-frame table (.csv)", "csv"),
                           ("All charts (.png)", "png"), ("Everything (.zip)", "zip"),
                           ("BT Freezing file (.btf) — save a copy…", "btf")):
            m.addAction(text, lambda k=kind: self.export_freezing(k))
        m.addSeparator()
        m.addAction("Open results folder", self.open_freezing_folder)
        self.export_menu_freezing = m

    def set_results_mode(self, mode):
        if not hasattr(self, "freeze_panel") or mode == self.results_mode:
            return
        if mode == "freezing":
            i = self.analysis_tabs.indexOf(self.stats_page)
            if i >= 0:
                self.analysis_tabs.removeTab(i)
            self.freeze_panel.take_statistics(self.stats_page)
            self.analysis_tabs.hide()
            self.freeze_panel.show()
            self.btn_export.setMenu(self.export_menu_freezing)
            self.btn_reset_results.setToolTip("Close this freezing result (saved files are kept).")
        else:
            page = self.freeze_panel.give_back_statistics()
            if page is not None and self.analysis_tabs.indexOf(page) < 0:
                self.analysis_tabs.addTab(page, "Statistics")
            self.freeze_panel.hide()
            self.freeze_panel.review.pause()
            self.analysis_tabs.show()
            self.btn_export.setMenu(self.export_menu_tracking)
            self.btn_reset_results.setToolTip("Clear these results and the tracking view, ready for a new run "
                                              "(saved files are kept).")
        self.results_mode = mode
        self._refresh_showing()

    def _refresh_showing(self):
        c = self.cmb_showing
        c.blockSignals(True)
        c.clear()
        chs = self.chamber_ui.results if hasattr(self, "chamber_ui") else []
        cur_ch = self.chamber_ui.current_index() if chs else None
        if chs:
            vn = Path(chs[0].get("video") or "").name
            for k, e in enumerate(chs):
                c.addItem(f"◎ {e['label']}" + (f"  —  {vn}" if vn else ""), ("chamber", k))
        if self.last_summary is not None and cur_ch is None and not self.last_summary.get("Chambers"):
            vn = self.last_summary.get("Video_name") or os.path.basename(str(self.last_summary.get("Video_file", "")))
            c.addItem(f"◎ Tracking — {vn or 'current result'}", ("tracking", None))
        for key, e in reversed(list(self.freeze_results.items())):
            who = f"  ·  {e['subject']}" if e.get("subject") else ""
            c.addItem(f"❄ Freezing — {e.get('name', '')}{who}", ("freezing", key))
        if self.results_mode == "freezing":
            want = ("freezing", self.freeze_key)
        else:
            want = ("chamber", cur_ch) if cur_ch is not None else ("tracking", None)
        for i in range(c.count()):
            if c.itemData(i) == want:
                c.setCurrentIndex(i)
        c.blockSignals(False)
        self.showing_box.setVisible(bool(self.freeze_results) or len(chs) > 1)

    def _showing_picked(self, i):
        kind, key = self.cmb_showing.itemData(i) or ("tracking", None)
        if kind == "freezing" and key in self.freeze_results:
            self.show_freezing_result(self.freeze_results[key], navigate=False)
        elif kind == "chamber":
            self.chamber_ui.open_result(int(key))
        else:
            self.set_results_mode("tracking")

    def _register_freezing(self, entry):
        key = entry["key"]
        old = self.freeze_results.pop(key, None)
        if old is not None and old.get("_tmp") and old.get("_tmp") != entry.get("_tmp"):
            shutil.rmtree(old["_tmp"], ignore_errors=True)
        self.freeze_results[key] = entry
        while len(self.freeze_results) > 12:
            _k, gone = self.freeze_results.popitem(last=False)
            if gone.get("_tmp"):
                shutil.rmtree(gone["_tmp"], ignore_errors=True)

    def freezing_result_ready(self, entry, show=False):
        """Freezing – Individual has a (new / re-tuned / saved) result: keep the Analysis page in step."""
        if entry is None or not hasattr(self, "freeze_panel"):
            return
        prev_key = self.freeze_key
        # the same video analysed again under a new .btf name replaces its older entry
        for k, e in list(self.freeze_results.items()):
            if k != entry["key"] and e.get("video") == entry.get("video") and e.get("md") is not entry.get("md") \
                    and e.get("from_page"):
                self.freeze_results.pop(k, None)
                if prev_key == k:
                    prev_key = entry["key"]
        entry["from_page"] = True
        self._register_freezing(entry)
        if show:
            self.show_freezing_result(entry)
        elif self.results_mode == "freezing" and prev_key == entry["key"]:
            self.freeze_key = entry["key"]
            self.freeze_panel.load(entry)
        self._refresh_showing()

    def show_freezing_result(self, entry, navigate=True):
        self._register_freezing(entry)
        self.freeze_key = entry["key"]
        self.set_results_mode("freezing")
        self.freeze_panel.load(entry, keep_tab=False)
        self._refresh_showing()
        if navigate:
            self.go(4)
            self.freeze_panel.tabs.setCurrentIndex(0)

    def open_bt_file_dialog(self):
        start = (os.path.dirname(self.bta["_path"]) if self.bta else (self.last_output_dir or str(Path.home())))
        path, _ = QFileDialog.getOpenFileName(self.win, "Open BT file", start,
                                              "BT files (*.bta *.btf);;BT analysis file (*.bta);;"
                                              "BT Freezing file (*.btf)")
        if not path:
            return
        if path.lower().endswith(".btf"):
            self.open_btf(path)
        else:
            self.open_bta(path, review=True)

    def open_btf(self, path, data=None):
        """A BT Freezing file -> its result on the Analysis & Results page."""
        from tracking import freezing as FZ
        from tracking.project_file import ProjectFileError, audit_verify, load_freezing
        if data is None:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                data = load_freezing(path)
            except ProjectFileError as exc:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self.win, "BT Freezing file", str(exc))
                return False
            except Exception as exc:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self.win, "BT Freezing file", f"Could not open this file:\n{exc}")
                return False
            QApplication.restoreOverrideCursor()
        md = data["md"]
        vp = (data.get("video") or {}).get("path") or ""
        md.path = vp
        result = FZ.analyse(md, data.get("settings") or {})
        folder = os.path.dirname(os.path.abspath(path))
        name = Path(vp).stem if vp else Path(path).stem
        charts = {}
        from output.freezing_graphs import CHARTS
        for k in CHARTS:
            p = os.path.join(folder, f"{name}_Freezing_{k}.png")
            if os.path.exists(p):
                charts[k] = p
        exp = data.get("experiment") or {}
        entry = {"key": os.path.normcase(os.path.abspath(path)), "video": vp, "name": name, "md": md,
                 "result": result, "region": data.get("region") or None,
                 "folder": folder, "files": {"folder": folder, "charts": charts} if len(charts) == len(CHARTS) else None,
                 "btf": data, "subject": exp.get("subject", ""), "group": exp.get("group", "")}
        self.show_freezing_result(entry)
        ok, bad = audit_verify(data.get("audit") or [])
        msg = f"Opened {Path(path).name}"
        if not ok:
            msg += f"  ⚠ Audit trail does not match from entry {bad + 1}."
        from tracking.project_file import APP
        made = str(data.get("app") or "")
        if made and made != APP:
            msg += (f"  ⚠ Made with {made}; this is {APP}. The movement was measured by the older version: "
                    f"re-analyse the video to use this version, and do not mix versions in one study.")
        self.status.showMessage(msg, 15000)
        return True

    def _freezing_entry(self):
        return self.freeze_results.get(self.freeze_key) if self.results_mode == "freezing" else None

    def show_freezing_audit(self):
        e = self._freezing_entry()
        trail = ((e or {}).get("btf") or {}).get("audit") or []
        if not trail:
            QMessageBox.information(self.win, "Audit trail", "This result has no audit trail yet (it is saved with "
                                    "the .btf after the analysis).")
            return
        from tracking.project_file import audit_verify
        ok, bad = audit_verify(trail)
        dlg = QDialog(self.win)
        dlg.setWindowTitle("Audit trail — " + (e.get("name") or ""))
        dlg.resize(900, 420)
        v = QVBoxLayout(dlg)
        lab = QLabel("✓ The audit trail is complete and unchanged." if ok else
                     f"⚠ The audit trail does not match from entry {bad + 1}.")
        lab.setStyleSheet("color:#15803d;font-weight:600;" if ok else "color:#b91c1c;font-weight:600;")
        v.addWidget(lab)
        t = QTableWidget(len(trail), 5)
        t.setHorizontalHeaderLabels(["#", "Time", "User", "Action", "Details"])
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        for r, en in enumerate(trail):
            det = ", ".join(f"{k}: {v}" for k, v in (en.get("details") or {}).items())
            for c_, val in enumerate((str(en.get("n")), str(en.get("time")), str(en.get("user")),
                                      str(en.get("action")), det)):
                t.setItem(r, c_, QTableWidgetItem(val))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)
        v.addWidget(t, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        dlg.exec()

    def _freezing_files(self, e):
        """Folder with up-to-date result files of entry e (a fresh report in a temporary folder when the
        result was tuned but not saved yet)."""
        if e.get("unsaved") or not e.get("files"):
            from output.freezing_report import write_freezing_report
            tmp = tempfile.mkdtemp(prefix="bt_freeze_export_")
            write_freezing_report(e["md"], e["result"], e.get("video") or e.get("name", "result"), out_dir=tmp,
                                  subject=e.get("subject", ""), group=e.get("group", ""))
            return tmp
        from pro_ui.freezing_results import result_files_folder
        return result_files_folder(e)

    def export_freezing(self, kind):
        e = self._freezing_entry()
        if e is None:
            return
        name = e.get("name") or "result"
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            folder = self._freezing_files(e)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self.win, "Export", f"Could not prepare the files:\n{exc}")
            return
        QApplication.restoreOverrideCursor()
        start = str(Path(e.get("video") or Path.home()).parent) if e.get("video") else str(Path.home())
        try:
            if kind in ("xlsx", "csv", "pdf"):
                src = os.path.join(folder, {"xlsx": f"{name}_Freezing.xlsx", "csv": f"{name}_Freezing_frames.csv",
                                            "pdf": f"{name}_Freezing_Summary.pdf"}[kind])
                if kind == "pdf" and not os.path.exists(src):
                    hits = sorted(Path(folder).glob("*_Freezing_Summary.pdf"))
                    src = str(hits[0]) if hits else src
                if not os.path.exists(src):
                    QMessageBox.information(self.win, "Export", "This result has no such file (made by an older "
                                                                "version: save the result again).")
                    return
                dst, _ = QFileDialog.getSaveFileName(self.win, "Save as", os.path.join(start, os.path.basename(src)),
                                                     {"xlsx": "Excel (*.xlsx)", "csv": "CSV (*.csv)",
                                                      "pdf": "PDF (*.pdf)"}[kind])
                if dst:
                    shutil.copy(src, dst)
            elif kind == "png":
                d = QFileDialog.getExistingDirectory(self.win, "Folder for the charts", start)
                if d:
                    for f in Path(folder).glob(f"{name}_Freezing_*.png"):
                        shutil.copy(str(f), d)
            elif kind == "zip":
                dst, _ = QFileDialog.getSaveFileName(self.win, "Save as", os.path.join(start, f"{name}_Freezing.zip"),
                                                     "Zip (*.zip)")
                if dst:
                    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
                        for f in sorted(Path(folder).iterdir()):
                            if f.is_file() and not f.name.endswith(".tmp"):
                                z.write(str(f), f.name)
            elif kind == "btf":
                src = ((e.get("btf") or {}).get("_path"))
                if not src or not os.path.exists(src):
                    QMessageBox.information(self.win, "BT Freezing file", "Save the result first (⤓ Save results "
                                            "on Freezing – Individual).")
                    return
                dst, _ = QFileDialog.getSaveFileName(self.win, "Save a copy of the BT Freezing file",
                                                     os.path.join(start, os.path.basename(src)),
                                                     "BT Freezing file (*.btf)")
                if dst:
                    shutil.copy(src, dst)
            else:
                return
        except Exception as exc:
            QMessageBox.warning(self.win, "Export", f"Export failed:\n{exc}")
            return
        self.status.showMessage("Exported.", 4000)

    def open_freezing_folder(self):
        e = self._freezing_entry()
        folder = (e or {}).get("folder") or (os.path.dirname(((e or {}).get("btf") or {}).get("_path") or "") or None)
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        else:
            QMessageBox.information(self.win, "Results", "No results folder for this result.")

    def close_freezing_result(self):
        """↻ on a freezing result: take it off this page (files stay)."""
        key = self.freeze_key
        gone = self.freeze_results.pop(key, None)
        if gone and gone.get("_tmp"):
            shutil.rmtree(gone["_tmp"], ignore_errors=True)
        self.freeze_key = None
        if self.freeze_results:
            self.show_freezing_result(list(self.freeze_results.values())[-1], navigate=False)
        else:
            self.freeze_panel.clear()
            self.set_results_mode("tracking")

    def _build_statistics_tab(self):
        page = QWidget()
        v = QVBoxLayout(page)
        note = QLabel("Group statistics work on a summary table with one row per video (Batch Processing writes "
                      "BatchSummary_*.xlsx with Subject / Group columns when subjects are assigned). The animal is the "
                      "experimental unit; repeated rows of one animal are averaged before between-group tests; "
                      "Benjamini–Hochberg FDR across measures. Missing design columns are reported, never inferred.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#475569;padding:4px;")
        v.addWidget(note)
        bar = QHBoxLayout()
        b1 = QPushButton("Load summary table…")
        b1.setProperty("cssClass", "secondary")
        b1.clicked.connect(self.load_statistics_table)
        bar.addWidget(b1)
        self.lbl_stats_src = QLabel("")
        self.lbl_stats_src.setProperty("cssClass", "muted")
        bar.addWidget(self.lbl_stats_src, 1)
        v.addLayout(bar)
        self.stat_tabs = QTabWidget()
        self.stat_tables = {}
        for key, title in (("between", "Between groups"), ("audit", "Design audit"), ("subject", "Subject level"),
                           ("repeated", "Repeated measures"), ("mixed", "Mixed effects")):
            t = QTableWidget()
            t.setEditTriggers(QTableWidget.NoEditTriggers)
            t.setAlternatingRowColors(True)
            self.stat_tables[key] = t
            self.stat_tabs.addTab(t, title)
        v.addWidget(self.stat_tabs, 1)
        self.analysis_tabs.addTab(page, "Statistics")
        self.stats_page = page

    @staticmethod
    def _fill_table(t, df):
        t.clear()
        t.setRowCount(0)
        t.setColumnCount(0)
        if df is None or len(df) == 0:
            return
        cols = list(df.columns)
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels([str(c) for c in cols])
        t.setRowCount(len(df))
        for r, row in enumerate(df.itertuples(index=False, name=None)):
            for c, v in enumerate(row):
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    txt = ""
                elif isinstance(v, (float, np.floating)):
                    txt = f"{float(v):.4g}" if abs(float(v)) >= 1e-3 or v == 0 else f"{float(v):.3e}"
                else:
                    txt = str(v)
                t.setItem(r, c, QTableWidgetItem(txt))
        t.resizeColumnsToContents()
        t.horizontalHeader().setStretchLastSection(True)
        t.verticalHeader().setVisible(False)

    def _update_kpi_captions(self):
        if not hasattr(self, "views"):
            return
        k = self.KPI_BY_TEST.get(self.test_key, self.KPI_BY_TEST["custom"])
        find(self.win, QLabel, "lblCaptionCenter").setText(k[0][1])
        find(self.win, QLabel, "lblCaptionRearing").setText(k[1][1])

    def _reset_analysis(self):
        for n in ("lblKpiDistance", "lblKpiSpeed", "lblKpiCenter", "lblKpiRearing"):
            find(self.win, QLabel, n).setText("—")
        self._update_kpi_captions()
        for v in self.views.values():
            v.clear("No results yet — run tracking first")
        for t in self.event_tables.values():
            self._fill_table(t, None)

    def reset_tracking_results(self, ask=True, settings=False):
        """↻ Reset.  Clears the results (Analysis page, frame review) and the tracking view, and -- from the
        Tracking page -- can also put the tracking settings back to their defaults.  Files already saved on disk
        are NOT deleted."""
        if getattr(self, "results_mode", "tracking") == "freezing" and self.stack.currentIndex() == 4:
            self.close_freezing_result()
            return
        if self.tracking:
            self.status.showMessage("Stop tracking before resetting.", 4000)
            return
        do_results, do_settings = True, bool(settings)
        if ask:
            dlg = QDialog(self.win)
            dlg.setWindowTitle("Reset")
            v = QVBoxLayout(dlg)
            c1 = QCheckBox("Clear the results and the tracking view")
            c1.setChecked(True)
            c2 = QCheckBox("Reset the tracking settings to their defaults\n(start / end time = whole video, engine, "
                           "detection mode, animal colour,\ncontrast, smoothing, animal size, preview images)")
            c2.setChecked(bool(settings))
            v.addWidget(c1)
            v.addWidget(c2)
            note = QLabel("Files already saved on disk (Excel, CSV, charts, .bta) are not deleted.")
            note.setStyleSheet("color:#64748b;")
            v.addWidget(note)
            bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            bb.button(QDialogButtonBox.Ok).setText("Reset")
            bb.accepted.connect(dlg.accept)
            bb.rejected.connect(dlg.reject)
            v.addWidget(bb)
            if dlg.exec() != QDialog.Accepted:
                return
            do_results, do_settings = c1.isChecked(), c2.isChecked()
        done = []
        if do_settings:
            self.chamber_ui.to_all()
            self._reset_tracking_settings()
            for it in self.chamber_ui.items:                # chambers follow the defaults again too
                it["overrides"] = {}
            self.chamber_ui._refresh_tracking_row()
            done.append("tracking settings back to defaults")
        if do_results:
            self.last_summary, self.last_output_dir = None, None
            self.bta = None
            self.chamber_ui.results = []
            if hasattr(self, "review"):
                self.review.clear()
            self._reset_analysis()
            self._preview = None                     # no detection-preview overlay any more
            done.append("results and tracking view cleared")
        # tracking view back to the current frame (with zones) or its placeholder
        if self.player.cap is not None:
            self.sl_track.setEnabled(True)
            self.sl_track.blockSignals(True)
            self.sl_track.setRange(0, max(self.player.total - 1, 0))
            self.sl_track.setValue(self.player.current)
            self.sl_track.blockSignals(False)
            self._update_time_labels()
            self._show_tracking_frame()
        else:
            self.view_track.clear("Load a video on the Experiment page")
        self.btn_start.setEnabled(True)
        self.btn_start.setText("▶   Start Tracking")
        self.btn_test.setEnabled(True)
        if done:
            self.status.showMessage("Reset — " + ", ".join(done) + ". Ready for a new tracking run.", 6000)

    def _reset_tracking_settings(self):
        d = self.DEFAULT_CFG
        self.cfg["min_contrast"] = d["min_contrast"]
        self.cfg["loc_thresh"] = 99.5
        self.cfg["engine"] = d["engine"]
        i = self.cmb_method.findData(d["engine"])
        self.cmb_method.setCurrentIndex(max(0, i))   # -> _engine_changed puts the threshold slider back
        self._engine_changed()
        if self.cmb_animals.count():
            self.cmb_animals.setCurrentIndex(0)
        self.cmb_color_mode.setCurrentText("Gray")
        self.cmb_pol.setCurrentIndex(max(0, self.cmb_pol.findData(d["polarity"])))
        self.sl_smooth.setValue(int(self._track_defaults.get("smooth", 5)))
        self.sl_min.setValue(int(self._track_defaults.get("min_size", 15)))
        self.cfg["body_length_px"] = 0
        self.lbl_measure.setText("")
        self.lbl_thr_hint.setText("")
        self.cmb_low.setCurrentIndex(max(0, self.cmb_low.findData(d["low_light"])))
        self.spin_prev.setValue(int(d["preview_samples"]))
        self.spin_start.setValue(0.0)
        self.spin_end.setValue(max(0.0, self.player.duration) if self.player.cap is not None else
                               self.spin_end.value())
        self._preview = None
        self._settings_changed()

    # ------------------------------------------------------------------ protected BT analysis files
    def open_bta_dialog(self, review=False):
        start = os.path.dirname(self.bta["_path"]) if self.bta else (self.last_output_dir or str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(self.win, "Open BT analysis file", start,
                                              "BT analysis file (*.bta)")
        if path:
            self.open_bta(path, review=review)

    def open_file_arg(self, path):
        """A .bta / .btms given on the command line (double-click in Windows Explorer)."""
        low = str(path).lower()
        if low.endswith(".btf"):
            self.open_btf(path)
        elif low.endswith(".bta"):
            self.open_bta(path, review=True)
        elif low.endswith(".btms"):
            self.go(self.EVENT_PAGE_INDEX)
            self.event_page.open_session(path)

    def open_bta(self, path, show=True, review=False):
        """Load a .bta: results on the Analysis page + the frame review.  When the result files are not next
        to it any more (copied to another computer), the copies stored inside the file are written next to
        it (older files without them: the results are computed again from the file)."""
        from tracking.project_file import load_analysis, ProjectFileError, audit_verify, extract_results
        try:
            data = load_analysis(path)
        except ProjectFileError as exc:
            QMessageBox.warning(self.win, "BT analysis file", str(exc))
            return False
        except Exception as exc:
            QMessageBox.warning(self.win, "BT analysis file", f"Could not open this file:\n{exc}")
            return False
        ok, bad = audit_verify(data.get("audit") or [])
        summary = data.get("summary") or {}
        out = summary.get("Output_folder") or ""
        here = os.path.dirname(os.path.abspath(path))
        if show and (not out or not os.path.exists(os.path.join(out, "trajectory.png"))):
            data["raw"]["output_dir"] = here
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                n = 0
                if data.get("_results"):
                    try:
                        n = extract_results(data, here)
                    except Exception as exc:
                        print(f"Stored result files not written: {exc}")
                if n and os.path.exists(os.path.join(here, "trajectory.png")):
                    summary = dict(summary)
                    summary["Output_folder"] = here
                else:
                    from tracking.location import analyse_track
                    summary = analyse_track(data["raw"], data.get("setup") or {},
                                            corrections=data.get("corrections"))
            except Exception as exc:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(self.win, "BT analysis file", f"The results could not be shown:\n{exc}")
                return False
            QApplication.restoreOverrideCursor()
            summary["Project_file"] = path
            data["summary"] = summary
        summary["Project_file"] = path
        self.bta = data
        self.review.load(data)
        if show:
            self.show_results(summary)
            self.go(4)
            if review:
                self.analysis_tabs.setCurrentWidget(self.review)
            msg = f"Opened {Path(path).name} — {len(data.get('corrections') or [])} corrected frame(s)."
            if not ok:
                msg += f"  ⚠ Audit trail does not match from entry {bad + 1}."
            self.status.showMessage(msg, 10000)
        return True

    def commit_corrections(self, pending, reason):
        """Frame review ▸ Commit: apply the corrections, recompute + rewrite the results, log the audit trail."""
        from tracking.location import analyse_track
        from tracking.project_file import audit_add, save_analysis, _key_results
        from pro_ui.frame_review import _ranges
        data = self.bta
        if data is None:
            return
        corr = {int(c["frame"]): c for c in (data.get("corrections") or [])}
        set_f, absent_f, restored_f = [], [], []
        for f, c in pending.items():
            if c.get("kind") == "restore":
                corr.pop(int(f), None)
                restored_f.append(f)
            else:
                corr[int(f)] = {k: v for k, v in c.items() if k in ("frame", "kind", "x", "y")}
                (absent_f if c.get("kind") == "absent" else set_f).append(f)
        new_corr = [corr[f] for f in sorted(corr)]
        raw = data["raw"]
        if not os.path.isdir(raw.get("output_dir") or ""):
            raw["output_dir"] = os.path.dirname(data["_path"])
        before = _key_results(data.get("summary") or {})
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            summary = analyse_track(raw, data.get("setup") or {}, corrections=new_corr)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self.win, "Commit changes", f"The results could not be recomputed:\n{exc}")
            return
        summary["Project_file"] = data["_path"]
        audit_add(data.setdefault("audit", []), "Manual correction committed", {
            "reason": reason, "positions_set": _ranges(set_f) or "-", "not_visible": _ranges(absent_f) or "-",
            "corrections_removed": _ranges(restored_f) or "-", "frames_changed": len(pending),
            "corrected_frames_total": len(new_corr), "before": before, "after": _key_results(summary)})
        data["corrections"] = new_corr
        data["summary"] = summary
        try:
            save_analysis(data)
        finally:
            QApplication.restoreOverrideCursor()
        self.bta = data
        self.show_results(summary)
        self.review.committed(data)
        self.status.showMessage(f"{len(pending)} change(s) committed — results, files and the audit trail are "
                                "updated.", 10000)

    def show_results(self, summary):
        from analysis.test_metrics import pretty
        w = self.win
        if hasattr(self, "freeze_panel"):
            self.set_results_mode("tracking")
        out = summary.get("Output_folder", "")
        self.last_summary, self.last_output_dir = summary, out
        unit = "cm" if "Total_distance_cm" in summary else "px"
        dist = float(summary.get("Total_distance_cm", summary.get("Total_distance_pixels", 0.0)) or 0.0)
        fps = float(summary.get("FPS") or 30.0)
        tracked_s = float(summary.get("Tracked_frames", 0) or 0) / fps
        find(w, QLabel, "lblCaptionDistance").setText("Total Distance")
        find(w, QLabel, "lblCaptionSpeed").setText("Mean Speed")
        find(w, QLabel, "lblKpiDistance").setText(f"{dist:.1f} {unit}")
        find(w, QLabel, "lblKpiSpeed").setText(f"{dist / tracked_s:.2f} {unit}/s" if tracked_s > 0 else "—")
        test = summary.get("Test_type") if summary.get("Test_type") in TESTS else self.test_key
        kp = self.KPI_BY_TEST.get(test, self.KPI_BY_TEST["custom"])
        if summary.get("Mode") == "Multi-Mouse":
            self._show_multi_kpis(summary, unit)
            kp = []
        for (key, caption, fmt), (cap_name, val_name) in zip(kp, (("lblCaptionCenter", "lblKpiCenter"),
                                                                   ("lblCaptionRearing", "lblKpiRearing"))):
            if key is None:          # custom: first zone percentage
                zk = [k for k in summary if k.endswith("_percent") and not k.startswith(("Tracking", "Thigmo"))]
                key = zk[0] if zk else None
                if key:
                    caption = "Time in " + key[:-8].replace("_", " ")
            val = summary.get(key) if key else None
            find(w, QLabel, cap_name).setText(caption)
            try:
                txt = fmt.format(float(val)) if val is not None and str(val) != "nan" else "—"
            except (TypeError, ValueError):
                txt = str(val)
            find(w, QLabel, val_name).setText(txt)

        def img(view, name):
            p = os.path.join(out, name)
            view.set_file(p) if os.path.exists(p) else view.clear("Not available for this run")
        img(self.views["plotTrajectory"], "trajectory.png")
        img(self.views["plotHeatmap"], "heatmap.png")
        img(self.views["plotTabTrackPlot"], "trajectory.png")
        img(self.views["plotTabHeatmap"], "heatmap.png")
        img(self.views["plotTabZoneAnalysis"], "zone_analysis.png"
            if os.path.exists(os.path.join(out, "zone_analysis.png")) else "zone_occupancy.png")
        self._fill_time_bins(summary, out, unit)

        # zone-events tables
        test_keys = ["Test_type", "Open_arm_time_s", "Closed_arm_time_s", "Center_time_s", "Open_arm_time_pct",
                     "Open_arm_time_ratio_pct", "Open_arm_entries", "Closed_arm_entries", "Open_arm_entries_pct",
                     "Total_arm_entries", "Latency_first_open_arm_s", "Spontaneous_alternation_pct", "Alternations",
                     "Possible_alternations", "Same_arm_returns", "Alternate_arm_returns", "Goal_arm_entries",
                     "Goal_arm_alternations", "Goal_arm_alternation_pct", "First_goal_choice", "Light_time_s",
                     "Dark_time_s", "Light_time_pct", "Transitions", "Light_entries", "Dark_entries",
                     "Start_compartment", "Latency_to_dark_s", "Latency_to_light_s", "Center_time_pct",
                     "Center_entries", "Periphery_time_s", "Center_distance", "Center_distance_pct",
                     "Latency_first_center_s", "Total_exploration_s", "Discrimination_index", "Discrimination_ratio",
                     "Recognition_index_pct", "Sociability_index", "Chamber_entries", "Periphery_entries",
                     "Anxiety_index"]
        rows = [(pretty(k), TESTS[summary[k]][0] if k == "Test_type" and summary[k] in TESTS else summary[k])
                for k in test_keys if k in summary and str(summary[k]) not in ("nan", "None")]
        rows += [(pretty(k), v) for k, v in summary.items()
                 if (k.endswith("_arm_entries") or k.startswith("Discrimination_index_") or k.startswith("Preference_index_")
                     or k.endswith("_exploration_s")) and k not in test_keys]
        rows += [("Total distance", f"{dist:.2f} {unit}"), ("Mean speed", f"{dist / tracked_s:.3f} {unit}/s" if tracked_s else "—")]
        self._fill_table(self.event_tables["metrics"], pd.DataFrame(rows, columns=["Measure", "Value"]))
        zpath = Path(out) / "zone_summary.csv"
        self._fill_table(self.event_tables["zones"], pd.read_csv(zpath) if zpath.exists() else None)
        epath = Path(out) / "arm_entries.csv"
        self._fill_table(self.event_tables["entries"], pd.read_csv(epath) if epath.exists() else None)
        qrows = [(pretty(k), summary.get(k)) for k in
                 ("Window_note", "Analysed_time_s", "Detection_engine", "Animal_polarity", "Background_info",
                  "Frames_processed", "Detected_frames", "Interpolated_frames", "Held_time_s", "Hidden_time_s",
                  "Lost_frames", "Corrected_frames", "Tracking_quality_percent", "Min_move_px",
                  "Entry_rule", "Time_in_zone_rule", "Auto_mask_outside_zones", "Trajectory_smoothing_window_frames",
                  "Project_file") if k in summary]
        self._fill_table(self.event_tables["quality"], pd.DataFrame(qrows, columns=["Item", "Value"]))
        try:
            from tracking.location import calculate_transitions
            csv_path = os.path.join(out, "raw_tracking.csv")
            if os.path.exists(csv_path) and summary.get("Mode") != "Multi-Mouse":
                tdf = pd.read_csv(csv_path, usecols=lambda c: c in ("Tracking_Status", "Time_seconds", "ROI",
                                                                       "Zone_4paw"))
                tr = (calculate_transitions(tdf, confirmation_frames=1, roi_col="Zone_4paw")
                      if "Zone_4paw" in tdf.columns else calculate_transitions(tdf))
                self._fill_table(self.event_tables["transitions"], tr)
            else:
                self._fill_table(self.event_tables["transitions"], None)
        except Exception:
            self._fill_table(self.event_tables["transitions"], None)

    def _show_multi_kpis(self, summary, unit):
        w = self.win
        mice = sorted({k.split("_")[1] for k in summary if k.startswith("mouse_") and k.endswith("_Tracked_frames")})
        fps = float(summary.get("FPS") or 30.0)

        def per(key_tail, fmt):
            vals = []
            for m in mice:
                v = summary.get(f"mouse_{m}_{key_tail}")
                vals.append("—" if v is None or str(v) == "nan" else fmt.format(float(v)))
            return " / ".join(vals)
        dist_key = f"Total_distance_{unit}" if unit == "cm" else "Total_distance_pixels"
        find(w, QLabel, "lblCaptionDistance").setText("Distance " + " / ".join(mice))
        find(w, QLabel, "lblKpiDistance").setText(per(dist_key, "{:.0f}") + f" {unit}")
        speeds = []
        for m in mice:
            d = summary.get(f"mouse_{m}_{dist_key}")
            t = (summary.get(f"mouse_{m}_Tracked_frames") or 0) / fps
            speeds.append(f"{float(d) / t:.1f}" if d is not None and t > 0 else "—")
        find(w, QLabel, "lblCaptionSpeed").setText("Speed " + " / ".join(mice))
        find(w, QLabel, "lblKpiSpeed").setText(" / ".join(speeds) + f" {unit}/s")
        zone = next((k[len("mouse_A_"):-len("_percent")] for k in summary
                     if k.startswith("mouse_A_") and k.endswith("_percent") and "quality" not in k.lower()), None)
        find(w, QLabel, "lblCaptionCenter").setText(f"Time in {zone.replace('_', ' ')} " + " / ".join(mice)
                                                    if zone else "Zone time")
        find(w, QLabel, "lblKpiCenter").setText(per(f"{zone}_percent", "{:.0f} %") if zone else "—")
        find(w, QLabel, "lblCaptionRearing").setText("Tracking quality " + " / ".join(mice))
        find(w, QLabel, "lblKpiRearing").setText(per("Tracking_quality_percent", "{:.0f} %"))

    def _fill_time_bins(self, summary, out, unit):
        csv_path = os.path.join(out, "raw_tracking.csv")
        if os.path.exists(os.path.join(out, "timebins.png")) and summary.get("Mode") == "Multi-Mouse":
            self.views["plotTabTimeBins"].set_file(os.path.join(out, "timebins.png"))
            self.views["plotActivity"].set_file(os.path.join(out, "timebins.png"))
            return
        if not os.path.exists(csv_path):
            self.views["plotTabTimeBins"].clear("Raw tracking data not found")
            self.views["plotActivity"].clear("Raw tracking data not found")
            return
        try:
            import matplotlib.pyplot as plt
            df = pd.read_csv(csv_path)
            bin_s = float(summary.get("Bin_size_s") or 60.0)
            ok = df[df["Tracking_Status"] == "Tracked"].copy()
            dcol = "Distance_cm" if (unit == "cm" and "Distance_cm" in ok.columns) else "Distance_pixels"
            t0 = float(df["Time_seconds"].min())
            ok["bin"] = ((ok["Time_seconds"] - t0) // bin_s).astype(int)
            per_bin = ok.groupby("bin")[dcol].sum()
            labels = [f"{int(b * bin_s)}–{int((b + 1) * bin_s)}s" for b in per_bin.index]
            fps = float(summary.get("FPS") or 30.0)
            tracked_t = ok.groupby("bin").size().reindex(per_bin.index) / fps
            fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
            ax.bar(labels, per_bin.values, color="#1f6feb")
            ax.set_ylabel(f"Distance ({unit})")
            ax.set_title(f"Distance per {int(bin_s)} s time bin")
            ax2.plot(labels, per_bin.values / tracked_t.values, marker="o", color="#ff006e")
            ax2.set_ylabel(f"Mean speed ({unit}/s)")
            ax2.set_title("Mean speed per time bin")
            for a in (ax, ax2):
                a.grid(alpha=0.25)
            plt.setp(ax2.get_xticklabels(), rotation=30, ha="right")
            fig.tight_layout()
            path = os.path.join(out, "timebins.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)
            self.views["plotTabTimeBins"].set_file(path)
            self.views["plotActivity"].set_file(path)
        except Exception as exc:
            self.views["plotTabTimeBins"].clear(f"Time-bin chart unavailable: {exc}")

    # --- statistics
    def load_statistics_table(self):
        start = str(Path(self.last_output_dir).parent) if self.last_output_dir else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self.win, "Load summary table", start, "Tables (*.xlsx *.csv)")
        if not path:
            return
        try:
            from analysis.advanced_statistics import run_advanced_statistics
            out = Path(path).parent / f"{Path(path).stem}_statistics"
            res = run_advanced_statistics(path, str(out))
            self.advanced_statistics = res
            for k in ("between", "audit", "subject", "repeated", "mixed"):
                self._fill_table(self.stat_tables[k], res.get(k))
            self.lbl_stats_src.setText(f"{Path(path).name}  →  {Path(res['excel_path']).name}")
            self.status.showMessage(f"Statistics written to {res['excel_path']}", 10000)
        except Exception as exc:
            QMessageBox.warning(self.win, "Statistics", f"Could not analyse that table:\n{exc}")

    def export_advanced_statistics(self):
        res = self.advanced_statistics
        if not res or not os.path.exists(res.get("excel_path", "")):
            QMessageBox.information(self.win, "Export", "Run Statistics ▸ Load summary table first.")
            return
        folder = QFileDialog.getExistingDirectory(self.win, "Choose export folder", str(Path(res["excel_path"]).parent))
        if folder:
            shutil.copy2(res["excel_path"], folder)
            self.status.showMessage(f"Exported {Path(res['excel_path']).name} to {folder}", 8000)

    def open_output_folder(self):
        if not self.last_output_dir or not os.path.isdir(self.last_output_dir):
            self.status.showMessage("No results folder yet.", 4000)
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.last_output_dir))

    def export(self, kind):
        src = self.last_output_dir
        if not src or not os.path.isdir(src):
            QMessageBox.information(self.win, "Export", "Run tracking (or open a past experiment) first.")
            return
        patterns = {"xlsx": ["*_Analysis.xlsx"], "csv": ["raw_tracking.csv"], "png": ["*.png"],
                    "zones": ["zone_summary.csv", "arm_entries.csv"]}
        if kind == "zip":
            target, _ = QFileDialog.getSaveFileName(self.win, "Export everything", str(Path(src).with_suffix(".zip")),
                                                    "Zip (*.zip)")
            if not target:
                return
            with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
                for f in Path(src).rglob("*"):
                    if f.is_file():
                        z.write(f, f.relative_to(src))
            self.status.showMessage(f"Exported to {target}", 8000)
            return
        files = [f for pat in patterns[kind] for f in Path(src).glob(pat)]
        if not files:
            QMessageBox.information(self.win, "Export", "Nothing to export for that option.")
            return
        folder = QFileDialog.getExistingDirectory(self.win, "Choose export folder", str(Path(src).parent))
        if not folder:
            return
        for f in files:
            shutil.copy2(f, folder)
        self.status.showMessage(f"Exported {len(files)} file(s) to {folder}", 8000)

    # ------------------------------------------------------------------ batch page
    BATCH_COLS = ["", "#", "Video Name", "Zones", "Time window", "Tracking settings", "Status", "Progress", "Results"]

    def _bind_batch(self):
        w = self.win
        self.tbl_batch = find(w, QTableWidget, "tblBatch")
        t = self.tbl_batch
        t.setRowCount(0)
        t.setColumnCount(len(self.BATCH_COLS))
        t.setHorizontalHeaderLabels(self.BATCH_COLS)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setSelectionBehavior(QTableWidget.SelectRows)
        t.setSelectionMode(QAbstractItemView.ExtendedSelection)
        h = t.horizontalHeader()
        for c in range(t.columnCount()):
            h.setSectionResizeMode(c, QHeaderView.Stretch if c in (2, 5) else QHeaderView.Fixed)
        for c, width in {0: 34, 1: 34, 3: 130, 4: 110, 6: 112, 7: 116, 8: 270}.items():
            t.setColumnWidth(c, width)
        t.setContextMenuPolicy(Qt.CustomContextMenu)
        t.customContextMenuRequested.connect(self._batch_menu)
        t.cellDoubleClicked.connect(lambda r, _c: self.open_batch_live(r) if self.tracking
                                    else self.open_batch_zone_editor(r))
        t.cellClicked.connect(self._batch_cell_clicked)
        t.itemSelectionChanged.connect(self._batch_selection_changed)
        t.itemChanged.connect(lambda it: self._update_start_text() if it.column() == 0 else None)
        h.sectionClicked.connect(lambda c: self._batch_tick_all() if c == 0 else None)
        hdr = QTableWidgetItem("☑")
        hdr.setToolTip("Tick = this video is processed when you press Start.\nClick here to tick / untick all.")
        t.setHorizontalHeaderItem(0, hdr)
        t.installEventFilter(self)
        self.prog_batch = find(w, QProgressBar, "progressBatch")
        self.lbl_batch_done = find(w, QLabel, "lblBatchCompleted")
        self.lbl_batch_eta = find(w, QLabel, "lblBatchEta")
        self.btn_start_batch = find(w, QPushButton, "btnStartBatch")
        find(w, QPushButton, "btnAddVideos").clicked.connect(self.batch_add_videos_dialog)
        find(w, QPushButton, "btnAddFolder").clicked.connect(self.batch_add_folder)
        find(w, QPushButton, "btnClear").clicked.connect(self.batch_clear)
        self.btn_start_batch.clicked.connect(self.start_or_stop_batch)
        self.btn_batch_live = QPushButton("👁   Live view")
        self.btn_batch_live.setProperty("cssClass", "secondary")
        self.btn_batch_live.setMinimumHeight(self.btn_start_batch.minimumHeight())
        self.btn_batch_live.setMinimumWidth(150)
        self.btn_batch_live.setCursor(Qt.PointingHandCursor)
        self.btn_batch_live.setToolTip("Watch the tracking of the selected video (or the one running now) live.")
        self.btn_batch_live.clicked.connect(lambda: self.open_batch_live(None))
        pc0 = find(w, QFrame, "cardBatchProgress").layout()
        pc0.insertWidget(pc0.indexOf(self.btn_start_batch), self.btn_batch_live)
        self.btn_start_batch.setMinimumWidth(220)
        # stack "n / N videos completed" + ETA UNDER the bar instead of beside it, so the progress card
        # fits narrower windows / 125 % display scaling without the text being drawn over the bar
        bar_row = self._layout_with(pc0, self.prog_batch)
        lbl_box = self._layout_with(pc0, self.lbl_batch_done)
        col = self._layout_with(pc0, find(w, QLabel, "lblBatchProgress"))
        if None not in (bar_row, lbl_box, col) and lbl_box is not bar_row:
            for lb in (self.lbl_batch_done, self.lbl_batch_eta):
                lbl_box.removeWidget(lb)
                col.addWidget(lb)
            col.setSpacing(6)
            self.prog_batch.setMinimumWidth(240)
            for lb in (self.lbl_batch_done, self.lbl_batch_eta):
                lb.setMinimumWidth(lb.fontMetrics().horizontalAdvance("Estimated time remaining: 00:00:00") + 8)
        hl = self._layout_with(find(w, QWidget, "pageBatch").layout(), find(w, QPushButton, "btnClear"))
        self.btn_batch_remove = QPushButton("✕  Remove")
        self.btn_batch_remove.setProperty("cssClass", "secondary")
        self.btn_batch_remove.setToolTip("Remove the selected video(s) from the queue (Delete key).")
        self.btn_batch_remove.clicked.connect(self.batch_remove_selected)
        self.btn_batch_reset = QPushButton("↺  Reset")
        self.btn_batch_reset.setProperty("cssClass", "secondary")
        self.btn_batch_reset.setToolTip("Keep the videos in the queue but clear every analysis (status, progress, "
                                        "results), so the batch can run again from the start.")
        self.btn_batch_reset.clicked.connect(self.batch_reset)
        self.chk_autofit = QCheckBox("Auto-fit zones to videos without zones")
        self.chk_autofit.setToolTip("Before tracking, detect the apparatus in every video that has no zones of its "
                                    "own and fit the zones to it. A zones_check.png is saved in each results "
                                    "folder — review it.")
        if hl is not None:
            hl.insertWidget(hl.indexOf(find(w, QPushButton, "btnAddVideos")), self.chk_autofit)
            hl.insertWidget(hl.indexOf(find(w, QPushButton, "btnClear")), self.btn_batch_remove)
            hl.insertWidget(hl.indexOf(find(w, QPushButton, "btnClear")), self.btn_batch_reset)
        self.chk_autofit.toggled.connect(lambda _on: self._refresh_batch_test_column())
        self.lbl_batch_hint = QLabel("1. Add videos   2. Select a video and set its zones, time window and tracking "
                                     "settings below (double-click = zone editor)   3. Tick ☑ the videos to "
                                     "process (click the ☑ heading to tick / untick all) and press Start — every "
                                     "video gets its own results folder.   While it runs, click any video to watch "
                                     "its tracking live (👁).")
        self.lbl_batch_hint.setProperty("cssClass", "muted")
        self.lbl_batch_hint.setWordWrap(True)
        tl = find(w, QFrame, "cardBatchTable").layout()
        tl.insertWidget(0, self.lbl_batch_hint)
        tl.addWidget(self._build_batch_panel())
        # results folder (in the progress card)
        self.batch_out_dir = self.cfg.get("batch_out_dir") or ""
        if self.batch_out_dir and not os.path.isdir(self.batch_out_dir):
            self.batch_out_dir = ""
        pc = find(w, QFrame, "cardBatchProgress").layout()
        box = QVBoxLayout()
        box.setSpacing(4)
        lbl = QLabel("Save results to")
        lbl.setProperty("cssClass", "fieldLabel")
        box.addWidget(lbl)
        self.lbl_batch_out = QLabel()
        self.lbl_batch_out.setStyleSheet("color:#000000;")
        self.lbl_batch_out.setMinimumWidth(220)
        self.lbl_batch_out.setMaximumWidth(420)
        box.addWidget(self.lbl_batch_out)
        brow = QHBoxLayout()
        b1 = _small_button("Change…", "Choose one folder for all results (each video gets its own sub-folder).")
        b1.clicked.connect(self._choose_batch_out)
        b2 = _small_button("Next to each video", "Default: a 'results' folder next to every video.")
        b2.clicked.connect(lambda: self._set_batch_out(""))
        b3 = _small_button("📂 Open", "Open the results folder.")
        b3.clicked.connect(self._open_batch_out)
        for b in (b1, b2, b3):
            brow.addWidget(b)
        brow.addStretch(1)
        box.addLayout(brow)
        pc.insertLayout(1, box)
        self._set_batch_out(self.batch_out_dir, save=False)
        self._batch_selection_changed()
        self._update_batch_overall(None)

    # --- settings panel for the selected video(s)
    def _build_batch_panel(self):
        panel = QFrame()
        panel.setObjectName("batchPanel")
        panel.setStyleSheet("QFrame#batchPanel { border-top: 1px solid #e3e8f0; }")
        v = QVBoxLayout(panel)
        v.setContentsMargins(8, 10, 8, 4)
        v.setSpacing(6)
        top = QHBoxLayout()
        self.lbl_bp_title = QLabel("Select a video above to change its settings")
        self.lbl_bp_title.setStyleSheet("font-weight:600; color:#000000;")
        top.addWidget(self.lbl_bp_title, 1)
        self.btn_bp_apply = _small_button("Apply to all videos ▾", "Copy the selected video's settings to every "
                                          "video in the queue.")
        am = QMenu(self.btn_bp_apply)
        am.addAction("Zones (fitted to each video's size)", lambda: self._batch_apply_all("zones"))
        am.addAction("Time window", lambda: self._batch_apply_all("window"))
        am.addAction("Tracking settings", lambda: self._batch_apply_all("settings"))
        am.addAction("Scale (pixels per cm)", lambda: self._batch_apply_all("scale"))
        am.addAction("Chambers (+ animal IDs)", lambda: self._batch_apply_all("chambers"))
        am.addSeparator()
        am.addAction("Everything", lambda: self._batch_apply_all("all"))
        self.btn_bp_apply.setMenu(am)
        top.addWidget(self.btn_bp_apply)
        v.addLayout(top)

        g = QGridLayout()
        g.setHorizontalSpacing(10)
        g.setVerticalSpacing(2)

        def cap(text):
            lab = QLabel(text)
            lab.setProperty("cssClass", "fieldLabel")
            return lab
        # zones
        self.lbl_bp_zones = QLabel("—")
        self.lbl_bp_zones.setStyleSheet("color:#000000;")
        self.lbl_bp_zones.setMinimumWidth(130)
        self.btn_bp_zones = QPushButton("✏  Set zones…")
        self.btn_bp_zones.setProperty("cssClass", "primary")
        self.btn_bp_zones.setToolTip("Open the zone editor with this video's own frame (draw, auto-detect, preset, "
                                     "copy). 'Save & next' walks through all videos.")
        self.btn_bp_zones.clicked.connect(lambda: self.open_batch_zone_editor(self._batch_selected_rows()[0]
                                                                              if self._batch_selected_rows() else 0))
        g.addWidget(cap("Zones"), 0, 0)
        zrow = QHBoxLayout()
        zrow.addWidget(self.lbl_bp_zones)
        zrow.addWidget(self.btn_bp_zones)
        g.addLayout(zrow, 1, 0)
        # time window
        self.sp_bp_start = QDoubleSpinBox()
        self.sp_bp_end = QDoubleSpinBox()
        for sp in (self.sp_bp_start, self.sp_bp_end):
            sp.setDecimals(1)
            sp.setSuffix(" s")
            sp.setRange(0.0, 1e6)
            sp.setMinimumWidth(86)
            sp.valueChanged.connect(self._batch_window_edited)
        self.btn_bp_whole = _small_button("Whole video", "Analyse the full length of the selected video(s).")
        self.btn_bp_whole.clicked.connect(self._batch_whole_video)
        g.addWidget(cap("Start"), 0, 1)
        g.addWidget(self.sp_bp_start, 1, 1)
        g.addWidget(cap("End"), 0, 2)
        g.addWidget(self.sp_bp_end, 1, 2)
        g.addWidget(self.btn_bp_whole, 1, 3)
        # distance calibration of this video
        self.sp_bp_ppc = QDoubleSpinBox()
        self.sp_bp_ppc.setDecimals(2)
        self.sp_bp_ppc.setRange(0.0, 10000.0)
        self.sp_bp_ppc.setSuffix(" px/cm")
        self.sp_bp_ppc.setSpecialValueText("Video Setup")
        self.sp_bp_ppc.setMinimumWidth(110)
        self.sp_bp_ppc.setToolTip("Pixels per cm of this video (0 = use the Video Setup scale). Use 📏 to measure it "
                                  "on the video.")
        self.sp_bp_ppc.valueChanged.connect(self._batch_ppc_edited)
        self.btn_bp_measure = _small_button("📏 Measure…", "Click both ends of a known distance on this video's frame.")
        self.btn_bp_measure.clicked.connect(self._batch_measure_scale)
        g.addWidget(cap("Scale"), 0, 4)
        srow = QHBoxLayout()
        srow.addWidget(self.sp_bp_ppc)
        srow.addWidget(self.btn_bp_measure)
        g.addLayout(srow, 1, 4)
        # chambers of this video (below Scale)
        self._bp_chamber_widgets = self.chamber_ui.build_batch_row(g, cap, 2, 4)
        g.setColumnStretch(5, 1)                       # everything stays together on the left (no wide gap)
        # tracking settings
        self.cmb_bp_engine = QComboBox()
        for key, text in ENGINES:
            self.cmb_bp_engine.addItem(text.replace(" (recommended)", ""), key)
        self.cmb_bp_mode = QComboBox()
        self.cmb_bp_mode.addItem("Gray", "gray")
        self.cmb_bp_mode.addItem("RGB", "rgb")
        self.cmb_bp_pol = QComboBox()
        for key, text in POLARITIES:
            self.cmb_bp_pol.addItem(text.split(" (")[0], key)
        self.cmb_bp_animals = QComboBox()
        for n in ("1", "2", "3"):
            self.cmb_bp_animals.addItem(n, int(n))
        self.cmb_bp_animals.hide()                     # one animal per video / chamber
        for col, (text, cmb, key) in enumerate((("Tracking engine", self.cmb_bp_engine, "engine"),
                                                ("Detection mode", self.cmb_bp_mode, "color_mode"),
                                                ("Animal colour", self.cmb_bp_pol, "polarity")), start=0):
            cmb.setMinimumWidth(60 if key == "animals" else 112)
            cmb.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cmb.setMinimumContentsLength(3 if key == "animals" else 9)
            cmb.currentIndexChanged.connect(lambda _i, c=cmb, k=key: self._batch_setting_edited(k, c))
            g.addWidget(cap(text), 2, col)
            if col == 0:
                g.addWidget(cmb, 3, col, Qt.AlignLeft)     # not as wide as the zones cell above it
            else:
                g.addWidget(cmb, 3, col)
        self.btn_bp_reset = _small_button("Use Tracking-page settings", "Forget this video's own engine / detection "
                                          "settings and follow the Tracking page.")
        self.btn_bp_reset.clicked.connect(self._batch_reset_settings)
        g.addWidget(self.btn_bp_reset, 3, 3)
        g.setVerticalSpacing(3)
        v.addLayout(g)
        self._bp_widgets = [self.btn_bp_zones, self.sp_bp_start, self.sp_bp_end, self.btn_bp_whole,
                            self.cmb_bp_engine, self.cmb_bp_mode, self.cmb_bp_pol, self.cmb_bp_animals,
                            self.btn_bp_reset, self.btn_bp_apply, self.sp_bp_ppc, self.btn_bp_measure] + \
            list(self._bp_chamber_widgets)
        return panel

    def _batch_selected_rows(self):
        sm = self.tbl_batch.selectionModel()
        return sorted({i.row() for i in sm.selectedRows()}) if sm is not None else []

    def _effective(self, it):
        """The tracking settings a batch video will really use (own value or Tracking page)."""
        g = self.chamber_ui.global_values() if self._chamber_view() else None   # All, not the chamber shown
        return {"engine": it.get("engine") or self.engine,
                "color_mode": it.get("color_mode") or (g["color_mode"] if g else
                                                       self.cmb_color_mode.currentText().lower()),
                "polarity": it.get("polarity") or (g["animal_polarity"] if g else
                                                   (self.cmb_pol.currentData() or "auto")),
                "animals": int(it.get("animals") or (self.cmb_animals.currentText() if hasattr(self, "cmb_animals")
                                                      and not getattr(self.chamber_ui, "enabled", False)
                                                      else 1) or 1)}

    def _batch_selection_changed(self):
        rows = [r for r in self._batch_selected_rows() if r < len(self.batch_items)]
        on = bool(rows) and not self.tracking
        for wd in self._bp_widgets:
            wd.setEnabled(on)
        if not rows:
            self.lbl_bp_title.setText("Select a video above to change its zones, time window and tracking settings"
                                      if self.batch_items else "Add videos to start")
            self.lbl_bp_zones.setText("—")
            return
        it = self.batch_items[rows[0]]
        self.lbl_bp_title.setText(f"Settings for {Path(it['path']).name}" if len(rows) == 1 else
                                  f"Settings for {len(rows)} selected videos (changes apply to all of them)")
        self._sync = True
        try:
            self.lbl_bp_zones.setText(self._batch_zone_text(it))
            dur = max(self.batch_items[r]["duration"] for r in rows)
            for sp in (self.sp_bp_start, self.sp_bp_end):
                sp.setRange(0.0, max(dur, 0.1))
            self.sp_bp_start.setValue(float(it.get("start") or 0.0))
            self.sp_bp_end.setValue(float(it["end"]) if it.get("end") is not None else float(it["duration"]))
            self.sp_bp_ppc.setValue(float(it.get("ppc") or 0.0))
            self.sp_bp_ppc.setStyleSheet("font-weight:600;" if it.get("ppc") else "")
            self.chamber_ui.batch_selected(it)
            eff = self._effective(it)
            for cmb, key in ((self.cmb_bp_engine, "engine"), (self.cmb_bp_mode, "color_mode"),
                             (self.cmb_bp_pol, "polarity"), (self.cmb_bp_animals, "animals")):
                i = cmb.findData(eff[key])
                cmb.setCurrentIndex(max(0, i))
                own = it.get(key) is not None
                cmb.setStyleSheet("font-weight:600;" if own else "")
                cmb.setToolTip("Own setting for this video" if own else "Follows the Tracking page")
        finally:
            self._sync = False

    def _batch_ppc_edited(self, _v=None):
        if self._sync:
            return
        v = float(self.sp_bp_ppc.value())
        for r in self._batch_selected_rows():
            if v > 0:
                self.batch_items[r]["ppc"] = round(v, 4)
            else:
                self.batch_items[r].pop("ppc", None)
            self._refresh_batch_row(r)
        self.sp_bp_ppc.setStyleSheet("font-weight:600;" if v > 0 else "")

    def _batch_measure_scale(self):
        rows = self._batch_selected_rows()
        if not rows:
            return
        it = self.batch_items[rows[0]]
        cap = cv2.VideoCapture(it["path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(float(it.get("start") or 0.0) * float(cap.get(cv2.CAP_PROP_FPS) or 30.0)))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            QMessageBox.warning(self.win, "Scale", "Could not read a frame of this video.")
            return
        crop = it.get("crop")
        if crop:
            frame = cv2.warpPerspective(frame, np.asarray(crop["matrix"], np.float32), (int(crop["w"]), int(crop["h"])))
        zones, _m = self._item_zones_raw(it) if it.get("zones") else self._setup_zones_for_size(it["size"])
        if crop:
            M = np.asarray(crop["matrix"], np.float64)

            def fwd(p):
                q = M @ np.array([p[0], p[1], 1.0])
                return (float(q[0] / q[2]), float(q[1] / q[2]))
            zones = {k: [fwd(p) for p in v] for k, v in zones.items()}
        dlg = ScaleDialog(self.win, frame, float(self.spin_len.value() or 40), zones=zones,
                          colors={k: self.canvas.color_of(k).name() for k in zones})
        if dlg.exec() == QDialog.Accepted and dlg.ppc:
            for r in rows:
                self.batch_items[r]["ppc"] = round(dlg.ppc, 4)
                self._refresh_batch_row(r)
            self._sync = True
            self.sp_bp_ppc.setValue(dlg.ppc)
            self._sync = False
            self.sp_bp_ppc.setStyleSheet("font-weight:600;")
            self.status.showMessage(f"Scale set: {dlg.ppc:.2f} px/cm for {len(rows)} video(s).", 6000)

    def _batch_window_edited(self, _v=None):
        if self._sync:
            return
        s, e = float(self.sp_bp_start.value()), float(self.sp_bp_end.value())
        for r in self._batch_selected_rows():
            it = self.batch_items[r]
            dur = float(it["duration"])
            ss, ee = min(s, dur), min(e, dur)
            if ee <= ss:
                continue
            it["start"] = ss if ss > 0 else None
            it["end"] = ee if ee < dur - 1e-6 else None
            self._refresh_batch_row(r)

    def _batch_whole_video(self):
        for r in self._batch_selected_rows():
            self.batch_items[r].pop("start", None)
            self.batch_items[r].pop("end", None)
            self._refresh_batch_row(r)
        self._batch_selection_changed()

    def _batch_setting_edited(self, key, cmb):
        if self._sync:
            return
        val = cmb.currentData()
        for r in self._batch_selected_rows():
            self.batch_items[r][key] = val
            self._refresh_batch_row(r)
        self._batch_selection_changed()

    def _batch_reset_settings(self):
        for r in self._batch_selected_rows():
            for k in ("engine", "color_mode", "polarity", "animals"):
                self.batch_items[r].pop(k, None)
            self._refresh_batch_row(r)
        self._batch_selection_changed()

    def _batch_apply_all(self, what):
        rows = self._batch_selected_rows()
        if not rows:
            return
        src = self.batch_items[rows[0]]
        n = 0
        for i, it in enumerate(self.batch_items):
            if i == rows[0]:
                continue
            if what in ("zones", "all"):
                if src.get("zones"):
                    z, m = self._item_zones_raw(src)
                    sx = it["size"][0] / max(1, src["size"][0])
                    sy = it["size"][1] / max(1, src["size"][1])
                    it["zones"] = {k: [(x * sx, y * sy) for x, y in v] for k, v in z.items()}
                    it["masks"] = [[(x * sx, y * sy) for x, y in mk] for mk in m]
                    it["crop"], it["ref_size"] = None, tuple(it["size"])
                    it["automask"] = src.get("automask")
                else:
                    for k in ("zones", "masks", "crop", "ref_size", "automask"):
                        it.pop(k, None)
            if what in ("window", "all"):
                dur = float(it["duration"])
                s = src.get("start")
                e = src.get("end")
                it["start"] = min(s, dur) if s else None
                it["end"] = min(e, dur) if e is not None and e < dur else None
                if it["end"] is not None and it["start"] is not None and it["end"] <= it["start"]:
                    it["start"] = it["end"] = None
            if what in ("settings", "all"):
                for k in ("engine", "color_mode", "polarity", "animals"):
                    if src.get(k) is None:
                        it.pop(k, None)
                    else:
                        it[k] = src[k]
            if what in ("scale", "all"):
                if src.get("ppc"):
                    it["ppc"] = src["ppc"]
                else:
                    it.pop("ppc", None)
            if what in ("chambers", "all"):
                self.chamber_ui.batch_apply(it, src)
            n += 1
        self._rebuild_batch_table()
        label = {"zones": "zones", "window": "time window", "settings": "tracking settings", "scale": "scale",
                 "chambers": "chambers", "all": "all settings"}[what]
        self.status.showMessage(f"Copied the {label} of {Path(src['path']).name} to {n} other video(s).", 6000)

    # --- zones per video
    def _item_zones_raw(self, it):
        """A batch video's own zones / masks in that video's raw frame coordinates."""
        zones = copy.deepcopy(it.get("zones") or {})
        masks = copy.deepcopy(it.get("masks") or [])
        crop = it.get("crop")
        if crop:
            Minv = np.asarray(crop["inv"], np.float64)

            def back(p):
                v = Minv @ np.array([p[0], p[1], 1.0])
                return (float(v[0] / v[2]), float(v[1] / v[2]))
            zones = {k: [back(p) for p in v] for k, v in zones.items()}
            masks = [[back(p) for p in m] for m in masks]
        ref = tuple(it.get("ref_size") or it["size"])
        sx, sy = it["size"][0] / max(1, ref[0]), it["size"][1] / max(1, ref[1])
        return ({k: [(x * sx, y * sy) for x, y in v] for k, v in zones.items()},
                [[(x * sx, y * sy) for x, y in m] for m in masks])

    def _setup_zones_for_size(self, size):
        """The Video-Setup zones / masks mapped onto the raw frame of a video of `size`."""
        return self._item_zones_raw({"zones": self.canvas.zones, "masks": self.canvas.masks, "crop": self.crop,
                                     "ref_size": (self.player.w, self.player.h) if self.player.w else size,
                                     "size": tuple(size)})

    def open_batch_zone_editor(self, row=0):
        if self.tracking:
            return
        if not self.batch_items:
            QMessageBox.information(self.win, "Zones", "Add videos to the batch first.")
            return
        row = int(min(max(0, row), len(self.batch_items) - 1))
        if self.canvas.tool is not None:
            self.canvas.cancel_tool(silent=True)
        dlg = BatchZoneDialog(self, row)
        dlg.exec()
        self._rebuild_batch_table()
        last = dlg.pos if dlg.pos < self.tbl_batch.rowCount() else row
        if 0 <= last < self.tbl_batch.rowCount():
            self.tbl_batch.selectRow(last)
        if dlg.saved:
            self.status.showMessage(f"Zones saved for {dlg.saved} video(s).", 6000)

    # --- queue
    def batch_add_videos_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self.win, "Add videos", str(Path.home()), VIDEO_FILTER)
        if paths:
            self.add_batch_videos(paths)

    def batch_add_folder(self):
        folder = QFileDialog.getExistingDirectory(self.win, "Add all videos from folder", str(Path.home()))
        if folder:
            paths = sorted(str(p) for p in Path(folder).iterdir() if p.suffix.lower() in VIDEO_EXTS)
            if not paths:
                self.status.showMessage("No video files found in that folder.", 5000)
                return
            self.add_batch_videos(paths)

    def add_batch_videos(self, paths, quiet=False):
        known = {it["path"] for it in self.batch_items}
        added = 0
        for p in paths:
            if p in known or Path(p).suffix.lower() not in VIDEO_EXTS:
                continue
            cap = cv2.VideoCapture(p)
            ok = cap.isOpened()
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps) if ok else 0
            size = (int(cap.get(3)), int(cap.get(4)))
            cap.release()
            if not ok:
                continue
            self.batch_items.append({"path": p, "duration": dur, "size": size, "status": "Pending", "pct": 0})
            added += 1
        self._rebuild_batch_table()
        if added and not quiet:
            self.status.showMessage(f"Added {added} video(s) to the batch queue.", 5000)

    def batch_remove_selected(self):
        if self.tracking:
            QMessageBox.information(self.win, "Remove", "Stop the batch before removing videos.")
            return
        rows = self._batch_selected_rows()
        if not rows:
            self.status.showMessage("Select the video(s) to remove first.", 4000)
            return
        gone = {self.batch_items[r]["path"] for r in rows if r < len(self.batch_items)}
        self.batch_items = [it for it in self.batch_items if it["path"] not in gone]
        self._rebuild_batch_table()
        self.status.showMessage(f"Removed {len(gone)} video(s) from the batch queue.", 5000)

    def _batch_zone_text(self, it):
        return self.chamber_ui.batch_text(it) + self._batch_zone_text_plain(it)

    def _batch_zone_text_plain(self, it):
        if it.get("zones"):
            return f"✓ Own zones ({len(it['zones'])})"
        if self.chk_autofit.isChecked():
            return "Auto-fit"
        if self.canvas.zones:
            return f"Video Setup ({len(self.canvas.zones)})"
        return "No zones"

    def _batch_window_text(self, it):
        dur = float(it["duration"])
        s = float(it.get("start") or 0.0)
        e = float(it["end"]) if it.get("end") is not None else dur
        if s <= 0 and abs(e - dur) < 1e-6:
            return f"Whole ({fmt_clock(dur)})"
        return f"{fmt_clock(s)} – {fmt_clock(e)}"

    def _batch_settings_text(self, it):
        eff = self._effective(it)
        eng = dict(ENGINES).get(eff["engine"], eff["engine"]).replace(" (recommended)", "")
        pol = {"auto": "Auto colour", "dark": "Dark animal", "light": "Light animal", "any": "Either"}[eff["polarity"]]
        txt = f"{eng} · {'RGB' if eff['color_mode'] == 'rgb' else 'Gray'} · {pol}"
        txt += f" · {float(it['ppc']):.2f} px/cm" if it.get("ppc") else ""
        own = any(it.get(k) is not None for k in ("engine", "color_mode", "polarity", "ppc"))
        return ("● " if own else "") + txt

    def _refresh_batch_row(self, r):
        t = self.tbl_batch
        if not (0 <= r < len(self.batch_items)) or t.item(r, 3) is None:
            return
        it = self.batch_items[r]
        t.item(r, 3).setText(self._batch_zone_text(it))
        t.item(r, 4).setText(self._batch_window_text(it))
        t.item(r, 5).setText(self._batch_settings_text(it))
        t.item(r, 5).setToolTip("● = this video has its own settings" if t.item(r, 5).text().startswith("●")
                                else "Follows the Tracking page settings")
        t.item(r, 2).setText(Path(it["path"]).name + (f"   ·  {it['subject']}" if it.get("subject") else ""))

    def _rebuild_batch_table(self):
        t = self.tbl_batch
        checks = {}
        for r in range(t.rowCount()):
            it0, it2 = t.item(r, 0), t.item(r, 2)
            if it0 is not None and it2 is not None and it2.data(Qt.UserRole):
                checks[it2.data(Qt.UserRole)] = it0.checkState()
        sel = {t.item(r, 2).data(Qt.UserRole) for r in self._batch_selected_rows() if t.item(r, 2) is not None}
        t.blockSignals(True)
        t.setRowCount(len(self.batch_items))
        for r, it in enumerate(self.batch_items):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(checks.get(it["path"], Qt.Checked))
            t.setItem(r, 0, chk)
            for c, text in ((1, str(r + 1)), (2, Path(it["path"]).name), (3, ""), (4, ""), (5, "")):
                t.setItem(r, c, QTableWidgetItem(text))
            t.item(r, 2).setData(Qt.UserRole, it["path"])
            t.item(r, 2).setToolTip(it["path"] + ("\n" + it["note"] if it.get("note") else ""))
            t.setItem(r, 6, QTableWidgetItem(""))
            t.setItem(r, 7, QTableWidgetItem(""))
            t.setItem(r, 8, QTableWidgetItem(""))
            t.removeCellWidget(r, 8)
            self._refresh_batch_row(r)
            self._set_batch_row(r, it["status"], it["pct"])
        t.clearSelection()
        sm = t.selectionModel()
        for r, it in enumerate(self.batch_items):
            if it["path"] in sel:
                sm.select(t.model().index(r, 0), sm.SelectionFlag.Select | sm.SelectionFlag.Rows)
        t.blockSignals(False)
        self._batch_selection_changed()
        self._update_batch_overall(None)
        self._update_start_text()
        if self._live_visible():
            self.live_dlg.fill_combo()
            self.live_dlg.refresh(full=True)

    def _refresh_batch_test_column(self):
        if not hasattr(self, "tbl_batch"):
            return
        for r in range(len(self.batch_items)):
            self._refresh_batch_row(r)
        if hasattr(self, "lbl_bp_zones"):
            self._batch_selection_changed()

    def _set_batch_row(self, r, status, pct):
        it = self.batch_items[r]
        changed = it.get("status") != status
        it["status"] = status
        if pct is not None:
            it["pct"] = pct
        watchable = status == "Processing" or it["path"] in self._batch_frames
        q = (it.get("summary") or {}).get("Tracking_quality_percent") if status == "Completed" else None
        qtxt = ""
        if q is not None:
            qtxt = f" · {q:.0f}%" + ("" if q >= 95 else (" ⚠ check" if q >= 85 else " ⚠ poor"))
        st = QTableWidgetItem(status + qtxt + ("  👁" if watchable else ""))
        from pro_ui import theme
        st.setForeground(QColor(theme.text(STATUS_COLORS.get(status, "#334155") if q is None or q >= 95
                                           else ("#d97706" if q >= 85 else "#dc2626"))))
        if it.get("error"):
            st.setToolTip(it["error"])
        elif q is not None:
            st.setToolTip(f"Tracking quality {q:.1f} %: the animal was found in {q:.1f} % of the frames in which it "
                          "could be seen (time hidden in a covered zone not counted).\n"
                          + ("Good." if q >= 95 else "Open 📊 Analysis ▸ Frame review and check the problem frames."))
        elif watchable:
            st.setToolTip("Click to watch this video's tracking (live view).")
        self.tbl_batch.setItem(r, 6, st)
        if self._live_visible() and self.live_dlg.path == it["path"]:
            self.live_dlg.refresh(full=changed)
        box = self.tbl_batch.cellWidget(r, 7)
        if box is None:
            box = QWidget()
            lay = QHBoxLayout(box)
            lay.setContentsMargins(6, 0, 6, 0)
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setTextVisible(False)
            lbl = QLabel()
            lbl.setMinimumWidth(38)
            lay.addWidget(bar)
            lay.addWidget(lbl)
            box.bar, box.lbl = bar, lbl
            self.tbl_batch.setCellWidget(r, 7, box)
        box.bar.setValue(it["pct"])
        box.lbl.setText(f"{it['pct']}%")
        color = {"Completed": "#22c55e", "Processing": "#1f6feb", "Failed": "#ef4444", "Stopped": "#f59e0b"}.get(status, "#cbd5e1")
        box.bar.setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
        # results: this video's own folder (written as soon as the video is finished)
        folder = (it.get("summary") or {}).get("Output_folder", "")
        if status == "Completed" and folder:
            if getattr(self.tbl_batch.cellWidget(r, 8), "folder", None) != folder:
                cell = QWidget()
                cell.folder = folder
                lay = QHBoxLayout(cell)
                lay.setContentsMargins(2, 0, 2, 0)
                lay.setSpacing(2)
                b_open = QToolButton()
                b_open.setText("📂 Open")
                b_open.setToolTip(f"Open this video's results folder:\n{folder}")
                b_open.clicked.connect(lambda _=False, f=folder: self._open_folder(f))
                b_exp = QToolButton()
                b_exp.setText("⤓ Export")
                b_exp.setToolTip("Copy this video's results (Excel, CSV, charts) to another folder.")
                b_exp.clicked.connect(lambda _=False, p=it["path"]: self._export_results_for(p))
                b_an = QToolButton()
                b_an.setText("📊 Analysis")
                b_an.setToolTip("Show this video's results on the Analysis page (plots, tables, frame review).")
                b_an.clicked.connect(lambda _=False, rr=r: self.open_batch_analysis(rr))
                for b in (b_an, b_open, b_exp):
                    b.setCursor(Qt.PointingHandCursor)
                    b.setAutoRaise(True)
                    lay.addWidget(b)
                lay.addStretch(1)
                self.tbl_batch.setCellWidget(r, 8, cell)
        elif self.tbl_batch.cellWidget(r, 8) is not None:
            self.tbl_batch.removeCellWidget(r, 8)

    def open_batch_analysis(self, r):
        """Batch ▸ 📊 Analysis: this video's results on the Analysis page, like an individual tracking run."""
        if not (0 <= r < len(self.batch_items)):
            return
        summ = self.batch_items[r].get("summary") or {}
        if summ.get("Chambers"):
            self.chamber_ui.show_run(summ, self.batch_items[r]["path"])   # every chamber under 'Showing'
        elif summ.get("Project_file") and os.path.exists(summ["Project_file"]):
            self.open_bta(summ["Project_file"])
        elif summ:
            self.show_results(summ)
            self.go(4)
        else:
            self.status.showMessage("This video has no results yet.", 4000)

    def _open_folder(self, folder):
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
        else:
            self.status.showMessage("That folder does not exist (yet).", 5000)

    # --- results folder
    def _set_batch_out(self, folder, save=True):
        self.batch_out_dir = folder or ""
        if self.batch_out_dir:
            txt = self.batch_out_dir
            if len(txt) > 60:
                txt = "…" + txt[-58:]
            self.lbl_batch_out.setText(txt)
            self.lbl_batch_out.setToolTip(self.batch_out_dir + "\nEach video gets its own sub-folder.")
        else:
            self.lbl_batch_out.setText("A 'results' folder next to each video")
            self.lbl_batch_out.setToolTip("<video folder>/results/<video name>/")
        if save:
            self.cfg["batch_out_dir"] = self.batch_out_dir
            save_json(SETTINGS_PATH, self.cfg)

    def _choose_batch_out(self):
        start = self.batch_out_dir or (str(Path(self.batch_items[0]["path"]).parent) if self.batch_items
                                       else str(Path.home()))
        folder = QFileDialog.getExistingDirectory(self.win, "Save batch results to", start)
        if folder:
            self._set_batch_out(folder)

    def _open_batch_out(self):
        folder = self.batch_out_dir or (str(Path(self.batch_items[0]["path"]).parent / "results")
                                        if self.batch_items else "")
        self._open_folder(folder)

    def _export_results_for(self, path):
        it = next((b for b in self.batch_items if b["path"] == path), None)
        if it is not None:
            self._export_results(it)

    def _export_results(self, it):
        src = (it.get("summary") or {}).get("Output_folder", "")
        if not src or not os.path.isdir(src):
            QMessageBox.information(self.win, "Export results", "This video has no results yet.")
            return
        dest_root = QFileDialog.getExistingDirectory(self.win, f"Export results of {Path(it['path']).name} to",
                                                     str(Path(src).parent))
        if not dest_root:
            return
        dest = Path(dest_root) / Path(src).name
        if dest.resolve() == Path(src).resolve():
            self.status.showMessage("That is already the results folder.", 5000)
            return
        try:
            shutil.copytree(src, dest, dirs_exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self.win, "Export results", f"Could not copy the results:\n{exc}")
            return
        self.status.showMessage(f"Results of {Path(it['path']).name} exported to {dest}", 8000)

    # --- context menu
    def _batch_menu(self, pos):
        row = self.tbl_batch.rowAt(pos.y())
        if row < 0:
            return
        if self.tracking:
            menu = QMenu(self.tbl_batch)
            a_live = menu.addAction("👁  Watch tracking live")
            if menu.exec(self.tbl_batch.viewport().mapToGlobal(pos)) is a_live:
                self.open_batch_live(row)
            return
        if row not in self._batch_selected_rows():
            self.tbl_batch.selectRow(row)
        it = self.batch_items[row]
        menu = QMenu(self.tbl_batch)
        a_live = menu.addAction("👁  Live view / last tracked frame")
        menu.addSeparator()
        a_zones = menu.addAction("✏  Set zones for this video…")
        a_pin = menu.addAction("Use current Video-Setup zones for this video")
        a_unpin = menu.addAction("Forget this video's own zones")
        a_unpin.setEnabled(bool(it.get("zones")))
        a_load = menu.addAction("Load in Video Setup")
        menu.addSeparator()
        a_win = menu.addAction("Set time window…")
        a_subj = menu.addAction("Assign subject…")
        a_reset = menu.addAction("Mark as pending (re-run)")
        menu.addSeparator()
        a_res = menu.addAction("Show results in Analysis")
        a_open = menu.addAction("Open results folder")
        a_exp = menu.addAction("Export results…")
        for a in (a_res, a_open, a_exp):
            a.setEnabled(bool(it.get("summary")))
        menu.addSeparator()
        a_rm = menu.addAction("Remove from queue")
        chosen = menu.exec(self.tbl_batch.viewport().mapToGlobal(pos))
        if chosen is a_live:
            self.open_batch_live(row)
        elif chosen is a_zones:
            self.open_batch_zone_editor(row)
        elif chosen is a_load:
            self._batch_load_row(row)
        elif chosen is a_pin:
            if self.player.path != it["path"] and QMessageBox.question(
                    self.win, "Zones", "The video loaded in Video Setup is a different file. Use its zones for "
                    f"{Path(it['path']).name} anyway?") != QMessageBox.Yes:
                return
            it["zones"] = copy.deepcopy(self.canvas.zones)
            it["masks"] = copy.deepcopy(self.canvas.masks)
            it["crop"] = copy.deepcopy(self.crop)
            it["ref_size"] = (self.player.w, self.player.h)
            it["automask"] = self.chk_automask.isChecked()
            self._rebuild_batch_table()
        elif chosen is a_unpin:
            for k in ("zones", "masks", "crop", "ref_size", "automask"):
                it.pop(k, None)
            self._rebuild_batch_table()
        elif chosen is a_win:
            dlg = TimeWindowDialog(self.win, it.get("start") or 0.0,
                                   it.get("end") if it.get("end") is not None else it["duration"], it["duration"])
            if dlg.exec() == QDialog.Accepted:
                s, e = dlg.s.value(), dlg.e.value()
                if e <= s:
                    QMessageBox.warning(self.win, "Time window", "End must be after start.")
                    return
                it["start"] = s if s > 0 else None
                it["end"] = e if e < it["duration"] - 1e-6 else None
                self._rebuild_batch_table()
        elif chosen is a_subj:
            ids = []
            for r in range(self.tbl_subjects.rowCount()):
                if self.tbl_subjects.item(r, 1):
                    ids.append(self.tbl_subjects.item(r, 1).text())
            ids += [d["Animal ID"] for d in self.experimental_design if d.get("Animal ID") not in ids]
            sid, ok = QInputDialog.getItem(self.win, "Assign subject", f"Subject for {Path(it['path']).name}:",
                                           ids or [""], 0, True)
            if ok:
                it["subject"] = sid.strip()
                self._rebuild_batch_table()
        elif chosen is a_reset:
            it.pop("summary", None)
            it.pop("error", None)
            self._set_batch_row(row, "Pending", 0)
        elif chosen is a_res:
            self.show_results(it["summary"])
            self.go(4)
        elif chosen is a_open:
            self._open_folder(it["summary"].get("Output_folder", ""))
        elif chosen is a_exp:
            self._export_results(it)
        elif chosen is a_rm:
            self.batch_remove_selected()

    def _batch_load_row(self, row):
        if self.tracking or not (0 <= row < len(self.batch_items)):
            return
        it = self.batch_items[row]
        if self.open_video(it["path"]):
            if it.get("zones"):
                if self.crop:
                    self.remove_crop(quiet=True)
                self.crop = copy.deepcopy(it.get("crop"))
                self.cur_frame = self._warp(self.raw_frame)
                self.canvas.set_frame(self.cur_frame)
                self.canvas.masks = copy.deepcopy(it.get("masks") or [])
                self.canvas.set_zones(copy.deepcopy(it["zones"]), record=True)
                self.status.showMessage(f"Loaded {Path(it['path']).name} with its own zones. After editing, "
                                        "right-click it in Batch ▸ 'Use current zones for this video'.", 10000)
            self.go(2)

    def _ticked_rows(self):
        t = self.tbl_batch
        return [r for r in range(min(t.rowCount(), len(self.batch_items)))
                if t.item(r, 0) is not None and t.item(r, 0).checkState() == Qt.Checked]

    def _batch_tick_all(self):
        if self.tracking or not self.batch_items:
            return
        t = self.tbl_batch
        on = len(self._ticked_rows()) < len(self.batch_items)
        t.blockSignals(True)
        for r in range(t.rowCount()):
            if t.item(r, 0) is not None:
                t.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)
        t.blockSignals(False)
        self._update_start_text()

    def _update_start_text(self):
        if self.tracking or not hasattr(self, "btn_start_batch"):
            return
        n, k = len(self.batch_items), len(self._ticked_rows())
        todo = sum(1 for r in self._ticked_rows() if self.batch_items[r]["status"] != "Completed")
        if n and k < n:
            self.btn_start_batch.setText(f"▶   Start Batch ({todo} ticked)")
        else:
            self.btn_start_batch.setText("▶   Start Batch Processing")
        self.btn_start_batch.setToolTip(f"Processes the ticked videos that are not completed yet ({todo} of {n}).")

    def batch_reset(self):
        """Keep the videos, clear every analysis (status, progress, results) -- ready to run again."""
        if self.tracking or not self.batch_items:
            return
        box = QMessageBox(self.win)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Reset batch")
        box.setText(f"Clear the analysis of all {len(self.batch_items)} video(s)?\n\nThe videos stay in the queue; "
                    "their status, progress and results in this list are cleared (the result files already "
                    "written on disk are not deleted).")
        chk = QCheckBox("Also forget each video's own zones, time window, scale and tracking settings")
        box.setCheckBox(chk)
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        if box.exec() != QMessageBox.Yes:
            return
        own = ("zones", "masks", "crop", "ref_size", "automask", "start", "end", "ppc", "engine", "color_mode",
               "polarity", "animals", "subject", "chambers", "chamber_ids")
        for it in self.batch_items:
            for k in ("summary", "error", "note"):
                it.pop(k, None)
            it["status"], it["pct"] = "Pending", 0
            if chk.isChecked():
                for k in own:
                    it.pop(k, None)
        self._batch_frames.clear()
        self._batch_cur_path = None
        self._rebuild_batch_table()
        self._update_start_text()
        self.status.showMessage("Batch reset — the videos are kept, all analyses cleared"
                                + (" (and their own settings)." if chk.isChecked() else "."), 6000)

    def batch_clear(self):
        if self.tracking or not self.batch_items:
            return
        if QMessageBox.question(self.win, "Clear queue", f"Remove all {len(self.batch_items)} video(s) from the "
                                "queue?") != QMessageBox.Yes:
            return
        self.batch_items = []
        self._rebuild_batch_table()
        self.status.showMessage("Batch queue cleared.", 5000)

    def _update_batch_overall(self, current, frac=0.0, done=False):
        n = len(self.batch_items)
        completed = sum(1 for it in self.batch_items if it["status"] == "Completed")
        total_frac = (completed + (0 if done or current is None else frac)) / n if n else 0
        pct = int(round(total_frac * 100))
        self.prog_batch.setValue(pct)
        self.lbl_batch_done.setText(f"{completed} / {n} videos completed ({pct}%)")
        if self.tracking and self.run_kind == "batch" and total_frac > 0.01:
            elapsed = time.time() - self._batch_t0
            self.lbl_batch_eta.setText(f"Estimated time remaining: {fmt_clock(elapsed * (1 - total_frac) / total_frac, True)}")
        elif not self.tracking:
            self.lbl_batch_eta.setText("Estimated time remaining: --:--:--")

    def start_or_stop_batch(self):
        if self.tracking and self.run_kind == "batch":
            self._stop_worker()
            self.btn_start_batch.setEnabled(False)
            self.btn_start_batch.setText("Stopping…")
            return
        if self.tracking:
            return
        if not self.canvas.zones and not self.chk_autofit.isChecked() and not any(it.get("zones") for it in self.batch_items):
            if QMessageBox.question(self.win, "Batch", "No zones are defined (Video Setup). Process the videos "
                                    "anyway (distance / speed only)?") != QMessageBox.Yes:
                return
        if self.canvas.tool is not None:
            self.canvas.cancel_tool(silent=True)
        if not self.batch_out_dir and not self.ensure_results_folder():
            return
        if self.batch_out_dir:
            try:
                Path(self.batch_out_dir).mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                QMessageBox.warning(self.win, "Batch", f"Cannot use the results folder:\n{exc}")
                return
        jobs, rows = [], []
        for r in range(self.tbl_batch.rowCount()):
            it = self.batch_items[r]
            if self.tbl_batch.item(r, 0).checkState() != Qt.Checked or it["status"] == "Completed":
                continue
            start = float(it.get("start") or 0.0)
            end = float(it["end"]) if it.get("end") is not None else float(it["duration"])
            setup = self._build_setup(it["path"], it["size"], start=start, end=end, item=it)
            if self.batch_out_dir:
                setup["output_dir_override"] = self.batch_out_dir
            if self.chk_autofit.isChecked() and not it.get("zones"):
                setup["_auto_fit"] = True
            job = self.chamber_ui.batch_job(it, setup)    # a chamber video: every chamber at once
            jobs.append(job or (it["path"], setup, setup.get("tracking_mode", "single")))
            rows.append(r)
        if not jobs:
            if self.batch_items and not self._ticked_rows():
                msg = "No video is ticked — tick ☑ the videos to process (or click the ☑ heading to tick all)."
            elif self.batch_items:
                msg = "All ticked videos are already completed — press ↺ Reset to run them again."
            else:
                msg = "Nothing to process — add videos first."
            self.status.showMessage(msg, 8000)
            return
        if self.canvas.zones and self.player.path and self.cur_frame is not None and not self.chamber_ui.enabled:
            self._learn_layout(self.cur_frame, self.canvas.zones, Path(self.player.path).name)
        self._row_map = rows
        self.run_kind = "batch"
        self.tracking = True
        self._batch_t0 = time.time()
        self._lock_ui(True)
        self._batch_selection_changed()
        self.btn_start_batch.setText("■   Stop Batch")
        for r in rows:
            self.batch_items[r].pop("error", None)
            self._batch_frames.pop(self.batch_items[r]["path"], None)
            self._set_batch_row(r, "Pending", 0)
        self._batch_cur_path = None
        self._start_worker(jobs, emit_frames=True)
        self._sync_live_rate()
        if self._live_visible():
            self.live_dlg.fill_combo()
            self.live_dlg.refresh(full=True)

    def _finish_batch(self, cancelled):
        self.tracking = False
        self._batch_cur_path = None
        self._lock_ui(False)
        self._batch_selection_changed()
        self.btn_start_batch.setEnabled(True)
        self.btn_start_batch.setText("▶   Start Batch Processing")
        self._update_start_text()
        done_items = [it for it in self.batch_items if it.get("summary") and it["status"] == "Completed"]
        failed = sum(1 for it in self.batch_items if it["status"] == "Failed")
        self._update_batch_overall(None)
        summary_path = None
        if done_items:
            try:
                design = {d.get("Animal ID"): d for d in self.experimental_design}
                rows = []
                for it in done_items:
                    # a chamber video gives one row per animal (chamber); its Animal ID is the Subject
                    parts = it["summary"].get("Chambers") or [it["summary"]]
                    for sm in parts:
                        s = {k: v for k, v in sm.items() if not isinstance(v, (list, dict))}
                        subj = (sm.get("Animal_ID") or it.get("subject", "")) if "Chamber" in sm else \
                            it.get("subject", "")
                        row = {"Video_name": Path(it["path"]).name, "Subject": subj,
                               "Results_folder": sm.get("Output_folder", "")}
                        if "Chamber" in sm:
                            row["Chamber"] = sm["Chamber"]
                        meta = design.get(subj, {})
                        for k, v in meta.items():           # every design column, your own ones too
                            if k != "Animal ID" and v:
                                row[k] = v
                        s.pop("Subject", None)
                        row.update(s)
                        rows.append(row)
                folder = (Path(self.batch_out_dir) if self.batch_out_dir
                          else Path(self.batch_items[0]["path"]).parent / "results")
                folder.mkdir(parents=True, exist_ok=True)
                summary_path = folder / f"BatchSummary_{datetime.datetime.now():%Y%m%d_%H%M%S}.xlsx"
                from output.batch_summary import write_batch_summary
                design_cols = []
                for d in self.experimental_design:
                    for k in d:
                        if k != "Animal ID" and k not in design_cols:
                            design_cols.append(k)
                write_batch_summary(rows, summary_path, design_cols)
            except Exception as exc:
                self.status.showMessage(f"Batch finished, but the summary workbook failed: {exc}", 10000)
                summary_path = None
        msg = "Batch stopped." if cancelled else "Batch finished — every video has its own results folder (📂 Open)."
        if summary_path is not None:
            msg += f"  Summary: {summary_path}"
        if failed:
            msg += f"  {failed} video(s) failed — hover the Status cell for the reason."
        if done_items and not cancelled:
            last = done_items[-1]["summary"]
            if last.get("Chambers"):
                self.chamber_ui.show_run(last, done_items[-1]["path"])
                self.go(5)                                # stay on the batch page (like a one-animal batch)
            else:
                self.show_results(last)
        self.status.showMessage(msg, 15000)


def run():
    if "--cli" in sys.argv[1:]:                # command line (no window): pro_ui/cli.py
        from pro_ui.cli import main as cli_main
        sys.exit(cli_main(sys.argv[sys.argv.index("--cli") + 1:]))
    # Taskbar identity: the installed app runs inside BehavioralTracker.exe itself (the launcher embeds
    # Python), so Windows takes the name + icon from that exe and pinning pins BehavioralTracker.exe.
    # Only when started from source with python.exe do we set an explicit ID (else it groups under Python).
    if sys.platform == "win32" and Path(sys.executable).stem.lower().startswith("python"):
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("BehavioralTracker.app.1")
        except Exception:
            pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BehavioralTracker")
    if sys.platform.startswith("linux"):     # Ubuntu dock: match behavioraltracker.desktop (name + icon)
        app.setDesktopFileName("behavioraltracker")
    for icon_file in (HERE / "icons" / "app_icon.ico", HERE / "icons" / "app_logo.png"):
        if icon_file.exists():
            app.setWindowIcon(QIcon(str(icon_file)))
            break
    # Fusion = the same control sizes on every Windows version (the native Windows 11 style ignores parts
    # of the style sheet and makes fields taller, which clipped their text)
    app.setStyle("Fusion")
    from pro_ui import theme
    theme.install(app)                     # light / dark like the computer (no setting in the program)
    ctrl = UXController()
    ctrl.show()
    # a BT file double-clicked in the file manager (Windows: the installer registers .bta and .btms)
    for arg in sys.argv[1:]:
        if str(arg).lower().endswith((".bta", ".btms", ".btf")) and os.path.isfile(arg):
            QTimer.singleShot(500, lambda a=arg: ctrl.open_file_arg(a))
            break
    return app, ctrl


if __name__ == "__main__":
    app, ctrl = run()
    sys.exit(app.exec())
