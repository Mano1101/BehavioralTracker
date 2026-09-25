"""
PreviewCanvas -- shows the active video's frame (warped through the
current crop, if any) with zone/mask/object overlays drawn on top using
the same OpenCV drawing primitives the rest of the pipeline uses, so what
you see here matches the annotated video the real analysis produces.

Mouse-driven editing (dragging crop corners, clicking zone/mask/object
vertices, the distance calibration line, clicking a Maze Template zone to
rename it) is handled here too, mirroring the canvas event handlers in
gui/main_window.py (_on_canvas_press/_on_canvas_drag/_on_canvas_release):
this widget converts a click from WIDGET pixels to FRAME pixels (the same
space pending_roi_points etc. are stored in) and hands off to the _op
state machine on MainWindow (app.on_canvas_press/drag/release), which owns
all the actual point data. This widget owns only the coordinate mapping,
the overlay rendering, and the QLabel that displays it.

Overlay drawing happens with the SAME cv2 primitives used elsewhere in the
frame, in frame-pixel space, BEFORE the frame is scaled down to fit the
widget -- this avoids having to duplicate the fit-scale/offset math for
every drawn shape (unlike the Tkinter canvas, which draws overlay items
directly in canvas/widget space on top of a separately-placed base image).
"""

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from tracking.location import ROI_COLORS, OBJECT_COLORS
from analysis.calculations import point_distance

MASK_COLOR = (0, 0, 0)
ZONE_COLOR = (0, 0, 0)
OBJECT_COLOR = (30, 105, 210)       # BGR for #d2691e (chocolate/orange)
TEMPLATE_ZONE_COLOR = (204, 119, 0)  # BGR for #0077cc
DISTANCE_COLOR = (255, 0, 255)       # magenta
GRID_COLOR = (222, 222, 222)         # faint light gray, snap-to-grid aid


def _pt(p):
    return int(round(p[0])), int(round(p[1]))


def _draw_dashed_line(frame, p1, p2, color, thickness=2, dash_len=8, gap_len=5):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    dist = float(np.linalg.norm(p2 - p1))
    if dist < 1e-6:
        return
    direction = (p2 - p1) / dist
    pos = 0.0
    while pos < dist:
        start = p1 + direction * pos
        end = p1 + direction * min(pos + dash_len, dist)
        cv2.line(frame, _pt(start), _pt(end), color, thickness, cv2.LINE_AA)
        pos += dash_len + gap_len


def _draw_polygon_set(frame, shapes, color, dashed=False, point_radius=4, active_index=None):
    """Unnamed shapes (crop corners, mask regions). Mirrors
    TrackerApp._draw_polygon_set."""
    for i, pts in enumerate(shapes):
        width = 3 if i == active_index else 2
        if len(pts) >= 2:
            for j in range(len(pts) - 1):
                if dashed:
                    _draw_dashed_line(frame, pts[j], pts[j + 1], color, width)
                else:
                    cv2.line(frame, _pt(pts[j]), _pt(pts[j + 1]), color, width, cv2.LINE_AA)
            if len(pts) >= 3:
                if dashed:
                    _draw_dashed_line(frame, pts[-1], pts[0], color, width)
                else:
                    cv2.line(frame, _pt(pts[-1]), _pt(pts[0]), color, width, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, _pt(p), point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, _pt(p), point_radius, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_grid(frame, spacing):
    """Faint reference grid shown while 'Snap to grid' is on for a
    zones/objects op (see op_toggle_snap in main_window.py) -- purely a
    visual aid, drawn BEHIND the shapes/points below it."""
    if spacing <= 0:
        return
    h, w = frame.shape[:2]
    x = 0
    while x < w:
        cv2.line(frame, (int(x), 0), (int(x), h), GRID_COLOR, 1, cv2.LINE_AA)
        x += spacing
    y = 0
    while y < h:
        cv2.line(frame, (0, int(y)), (w, int(y)), GRID_COLOR, 1, cv2.LINE_AA)
        y += spacing


def _draw_named_polygon_set(frame, regions, color, active_region=None, point_radius=4):
    """Named shapes (zones, objects, template zones). Mirrors
    TrackerApp._draw_named_polygon_set."""
    for name, pts in regions.items():
        width = 3 if name == active_region else 2
        if len(pts) >= 2:
            for j in range(len(pts) - 1):
                cv2.line(frame, _pt(pts[j]), _pt(pts[j + 1]), color, width, cv2.LINE_AA)
            if len(pts) >= 3:
                cv2.line(frame, _pt(pts[-1]), _pt(pts[0]), color, width, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, _pt(p), point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(frame, _pt(p), point_radius, (255, 255, 255), 1, cv2.LINE_AA)
        if pts:
            mx = sum(p[0] for p in pts) / len(pts)
            my = sum(p[1] for p in pts) / len(pts)
            cv2.putText(frame, name, (int(mx) - 10, int(my)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2, cv2.LINE_AA)


def draw_overlays(frame_bgr, app):
    """Pure function: returns a NEW annotated frame with the STATIC
    (non-editing) pending_* overlays, matching what the real analysis run
    will draw. Used whenever no interactive op is in progress."""
    disp = frame_bgr.copy()

    if app.pending_mask_points:
        _draw_polygon_set(disp, app.pending_mask_points, MASK_COLOR, dashed=True)

    if app.pending_roi_points:
        for i, (name, pts) in enumerate(sorted(app.pending_roi_points.items())):
            if len(pts) < 2:
                continue
            color = ROI_COLORS[i % len(ROI_COLORS)]
            arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(disp, [arr], True, color, 2, lineType=cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in pts]))
            cy = int(np.mean([p[1] for p in pts]))
            cv2.putText(disp, name, (cx - 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    if app.pending_object_points:
        for i, (name, pts) in enumerate(sorted(app.pending_object_points.items())):
            if not pts:
                continue
            color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
            x, y = pts[0]
            cv2.circle(disp, (int(x), int(y)), 6, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(disp, name, (int(x) + 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return disp


def draw_op_overlay(frame_bgr, op):
    """Pure function: returns a NEW frame with the LIVE, in-progress
    editing overlay for the given _op dict. Mirrors
    TrackerApp._redraw_op. Caller passes a frame it's fine to draw on
    directly (already copied)."""
    disp = frame_bgr
    kind = op["kind"]

    if op.get("snap") and op.get("grid_spacing"):
        _draw_grid(disp, op["grid_spacing"])

    if kind == "crop":
        _draw_polygon_set(disp, op["shapes"], (0, 0, 0))
    elif kind == "mask":
        _draw_polygon_set(disp, op["shapes"], MASK_COLOR, dashed=True, active_index=len(op["shapes"]) - 1)
    elif kind == "zones":
        _draw_named_polygon_set(disp, op["regions"], ZONE_COLOR, active_region=op["active_region"])
    elif kind == "objects":
        _draw_named_polygon_set(disp, op["regions"], OBJECT_COLOR, active_region=op["active_region"])
    elif kind == "template_zones":
        _draw_named_polygon_set(disp, op["regions"], TEMPLATE_ZONE_COLOR)
    elif kind == "distance":
        for p in op["points"]:
            cv2.circle(disp, _pt(p), 4, DISTANCE_COLOR, -1, cv2.LINE_AA)
        if len(op["points"]) == 2:
            p0, p1 = op["points"]
            cv2.line(disp, _pt(p0), _pt(p1), DISTANCE_COLOR, 2, cv2.LINE_AA)
            px_d = point_distance(p0, p1)
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            cv2.putText(disp, f"{px_d:.1f} px", (int(mx) - 25, int(my) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, DISTANCE_COLOR, 2, cv2.LINE_AA)

    return disp


class PreviewCanvas(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setObjectName("previewCanvas")
        self.setMinimumSize(480, 360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.image_label = QLabel("No video selected")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: #8a93a3; font-size: 11px;")
        # Let clicks fall through to THIS widget (PreviewCanvas), which is
        # what actually handles crop/zone/mask/object/distance drawing --
        # QLabel itself doesn't need or want mouse events.
        self.image_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.image_label)

        # Frame-pixel <-> widget-pixel mapping for the CURRENTLY displayed
        # image, recomputed whenever refresh() rebuilds the pixmap (mirrors
        # TrackerApp._canvas_scale/_canvas_off_x/_canvas_off_y).
        self.scale = 1.0
        self._off_x = 0
        self._off_y = 0
        self._frame_w = 0
        self._frame_h = 0

    def refresh(self):
        app = self.app
        op = getattr(app, "_op", None)
        frame = op["_base_frame"] if op is not None else app.get_warped_active_frame()

        if frame is None:
            if app.active_index is None or not app.videos:
                self.image_label.setText("No video selected")
            else:
                self.image_label.setText("Could not read a frame from this video")
            self.image_label.setPixmap(QPixmap())
            self.scale, self._off_x, self._off_y = 1.0, 0, 0
            self._frame_w = self._frame_h = 0
            return

        annotated = draw_op_overlay(frame.copy(), op) if op is not None else draw_overlays(frame, app)
        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        h, w, _ = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)

        target = self.size()
        if target.width() > 10 and target.height() > 10 and w > 0 and h > 0:
            scaled = pix.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled)
            self.scale = scaled.width() / w if w else 1.0
            self._off_x = max(0, (target.width() - scaled.width()) // 2)
            self._off_y = max(0, (target.height() - scaled.height()) // 2)
        else:
            self.image_label.setPixmap(pix)
            self.scale, self._off_x, self._off_y = 1.0, 0, 0
        self._frame_w, self._frame_h = w, h
        self.image_label.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh()

    # ------------------------------------------------------------------
    # Coordinate mapping (mirrors TrackerApp._to_frame_xy/_to_canvas_xy)
    # ------------------------------------------------------------------

    def to_frame_xy(self, wx, wy):
        if self.scale <= 0:
            return 0.0, 0.0
        return (wx - self._off_x) / self.scale, (wy - self._off_y) / self.scale

    # ------------------------------------------------------------------
    # Mouse events -> the _op state machine on MainWindow
    # (mirrors TrackerApp._on_canvas_press/_on_canvas_drag/_on_canvas_release)
    # ------------------------------------------------------------------

    def _event_frame_xy(self, event):
        pos = event.position() if hasattr(event, "position") else event.pos()
        return self.to_frame_xy(pos.x(), pos.y())

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or getattr(self.app, "_op", None) is None:
            super().mousePressEvent(event)
            return
        fx, fy = self._event_frame_xy(event)
        self.app.on_canvas_press(fx, fy)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or getattr(self.app, "_op", None) is None:
            super().mouseMoveEvent(event)
            return
        fx, fy = self._event_frame_xy(event)
        self.app.on_canvas_drag(fx, fy)

    def mouseReleaseEvent(self, event):
        if getattr(self.app, "_op", None) is None:
            super().mouseReleaseEvent(event)
            return
        self.app.on_canvas_release()
