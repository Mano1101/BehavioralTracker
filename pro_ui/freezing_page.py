"""Freezing Analysis page (BehavioralTracker 1.0.0).

Single video:  open a video, (optionally) draw the region polygon / masks and measure the animal, press Analyse; the motion
               graph, the freezing bouts and the review video (changed pixels in red, FREEZING / ACTIVE) appear.
               The threshold, durations, epochs and bins can then be changed and the result follows at once.
Batch:         many videos with the same settings, one results folder per video + one summary workbook.

The page sits in the side bar under Batch Processing and above Event Scoring (manual scoring stays last).
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QTableWidget,
                               QTableWidgetItem, QTabWidget, QToolButton, QVBoxLayout, QWidget)

from pro_ui.ux_controller import (APP_DIR, HERE, POLARITIES, VIDEO_EXTS, VIDEO_FILTER, Player, _small_button,
                                  bgr_to_qimage, fmt_clock, load_json, save_json)
from tracking import freezing as FZ

SETTINGS_PATH = APP_DIR / "freezing_settings.json"
PRESETS_PATH = APP_DIR / "freezing_presets.json"
# what a preset holds: the method settings (not the video's own body length / region)
PRESET_KEYS = ("motion_area", "colour", "cutoff_auto", "cutoff", "threshold_pct", "min_freeze_s", "ignore_blips_s",
               "light_guard", "trim_lights", "polarity", "min_contrast", "bin_s", "pre_s")
PRESET_PROTOCOL_KEYS = ("start_time", "end_time", "epochs")
BUILTIN_PRESETS = {
    "BT defaults — whole region (Classic)": {k: FZ.DEFAULTS[k] for k in PRESET_KEYS},
    "BT defaults — around the animal": dict({k: FZ.DEFAULTS[k] for k in PRESET_KEYS}, motion_area="animal"),
}
INSTANT_KEYS = ("cutoff_auto", "cutoff", "threshold_pct", "min_freeze_s", "ignore_blips_s", "light_guard",
                "bin_s", "epochs", "motion_area", "body_length_px", "trim_lights")


# ============================================================================= small widgets
class FrameCanvas(QWidget):
    """The video picture (aspect kept) + drawing a polygon (region / mask), a box (drag) or measuring the animal
    (2 clicks)."""
    box_done = Signal(object)          # (x0, y0, x1, y1) video pixels
    measure_done = Signal(float)       # length in video pixels
    poly_done = Signal(object)         # [(x, y), ...] video pixels
    poly_edited = Signal(str, int, object)   # kind ("region" / "mask"), mask index, new points (a corner dragged)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.img = None
        self.mode = None
        self._a = self._b = None
        self._pts = []
        self.poly_hint = "Click points · double-click = done · right-click = undo · Esc"
        self.poly_kind = "region"
        self.placeholder = "Open a video to analyse freezing"
        self.status = None              # (big text, small text, freezing?)
        self.body = None                # the body dots (FZ.body_overlay), drawn sharp at screen resolution
        self.edit_polys = []            # [(kind, index, [(x, y), ...])] region / masks whose corners can be dragged
        self.editable = True
        self._drag = None               # (kind, index, corner) being dragged
        self._pan = None                # (start widget pos, start centre) while moving a zoomed picture
        self.zoom = 1.0
        self.centre = None              # picture point at the middle of the view (zoomed)

    def set_bgr(self, bgr, status=None, body=None):
        """status = (big, small, freezing?) or the live (big, small, freezing?, body dots)."""
        self.img = bgr_to_qimage(bgr) if bgr is not None else None
        if status is not None and len(status) > 3:
            body = status[3] if body is None else body
            status = tuple(status[:3])
        self.status = status
        self.body = body
        self.update()

    def _paint_body(self, p, s):
        """Small dots joined like a mouse (green = still, red = that part moved), on top of the picture."""
        b = self.body
        P = b["pts"]
        moved = b.get("moved") or {}
        L = float(b.get("len") or 0.0)
        r = min(5.0, max(2.2, 0.035 * L * s)) if L > 0 else 3.0
        p.setRenderHint(QPainter.Antialiasing, True)
        pts = {j: self._to_w(float(P[j, 0]), float(P[j, 1])) for j in range(len(P)) if np.isfinite(P[j, 0])}
        lines = [(pts[a], pts[c]) for a, c in FZ.body_chain(P) if a in pts and c in pts]
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor(0, 0, 0, 140), 3.0, Qt.SolidLine, Qt.RoundCap))
        for a, c in lines:
            p.drawLine(a, c)
        p.setPen(QPen(QColor(240, 240, 240), 1.3, Qt.SolidLine, Qt.RoundCap))
        for a, c in lines:
            p.drawLine(a, c)
        p.setPen(QPen(QColor(15, 15, 15), 1.0))
        for j, q in pts.items():
            part = FZ.BODY_PART_OF.get(j, "tail")
            if moved.get(part):          # moved: red diamond (shape + colour, readable for colour-blind users)
                d = 1.45 * r
                p.setBrush(QColor(235, 60, 60))
                p.drawPolygon([QPointF(q.x(), q.y() - d), QPointF(q.x() + d, q.y()), QPointF(q.x(), q.y() + d),
                               QPointF(q.x() - d, q.y())])
            else:                        # still: green dot
                p.setBrush(QColor(90, 205, 90))
                p.drawEllipse(q, r, r)

    def set_mode(self, mode):
        self.mode = mode
        self._a = self._b = None
        self._pts = []
        self.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)
        self.update()

    def _geo(self):
        if self.img is None:
            return None
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih)
        if self.zoom <= 1.0001 or self.centre is None:
            return (self.width() - iw * s) / 2.0, (self.height() - ih * s) / 2.0, s
        s *= self.zoom
        cx, cy = self.centre
        return self.width() / 2.0 - cx * s, self.height() / 2.0 - cy * s, s

    def _clamp_centre(self):
        if self.img is None or self.centre is None:
            return
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih) * self.zoom
        hw, hh = self.width() / (2.0 * s), self.height() / (2.0 * s)
        cx = min(max(self.centre[0], min(hw, iw / 2.0)), max(iw - hw, iw / 2.0))
        cy = min(max(self.centre[1], min(hh, ih / 2.0)), max(ih - hh, ih / 2.0))
        self.centre = (cx, cy)

    def reset_zoom(self):
        self.zoom, self.centre = 1.0, None
        self.update()

    def wheelEvent(self, ev):
        """Scroll = zoom at the mouse (to place corners exactly on small animals / boxes)."""
        if self.img is None:
            return
        pos = ev.position()
        before = self._to_img(pos, clamp=False)
        step = 1.25 if ev.angleDelta().y() > 0 else 1 / 1.25
        z = min(8.0, max(1.0, self.zoom * step))
        if abs(z - self.zoom) < 1e-6:
            return
        if self.centre is None:
            self.centre = (self.img.width() / 2.0, self.img.height() / 2.0)
        self.zoom = z
        if z <= 1.0001:
            self.reset_zoom()
            return
        # keep the picture point under the mouse where it is
        iw, ih = self.img.width(), self.img.height()
        s = min(self.width() / iw, self.height() / ih) * z
        self.centre = (before[0] - (pos.x() - self.width() / 2.0) / s, before[1] - (pos.y() - self.height() / 2.0) / s)
        self._clamp_centre()
        self.update()

    def _corner_at(self, pos, radius=8.0):
        """(kind, index, corner) of the region / mask corner under the mouse, or None."""
        if not self.editable or self.img is None:
            return None
        best = None
        for kind, idx, pts in self.edit_polys:
            for k, (x, y) in enumerate(pts):
                q = self._to_w(x, y)
                d = math.hypot(q.x() - pos.x(), q.y() - pos.y())
                if d <= radius and (best is None or d < best[0]):
                    best = (d, (kind, idx, k))
        return best[1] if best else None

    def _to_img(self, p, clamp=True):
        g = self._geo()
        if g is None:
            return None
        ox, oy, s = g
        x, y = (p.x() - ox) / s, (p.y() - oy) / s
        if not clamp:
            return x, y
        return (min(max(x, 0.0), self.img.width() - 1.0), min(max(y, 0.0), self.img.height() - 1.0))

    def _to_w(self, x, y):
        ox, oy, s = self._geo()
        return QPointF(ox + x * s, oy + y * s)

    def _finish_poly(self):
        pts = list(self._pts)
        while len(pts) >= 2 and math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < 1.0:
            pts.pop()                                      # a double-click adds the last point twice
        self.set_mode(None)
        if len(pts) >= 3:
            self.poly_done.emit(pts)

    def mouseDoubleClickEvent(self, ev):
        if self.mode == "poly" and ev.button() == Qt.LeftButton:
            self._finish_poly()
        elif self.mode is None and self.zoom > 1.0001:
            self.reset_zoom()

    def event(self, ev):
        # while drawing, Enter / Esc / Backspace are for the drawing (not the window's shortcuts)
        if ev.type() == QEvent.ShortcutOverride and self.mode and \
                ev.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Escape, Qt.Key_Backspace):
            ev.accept()
            return True
        return super().event(ev)

    def mousePressEvent(self, ev):
        if self.mode == "poly":
            if ev.button() == Qt.RightButton:
                if self._pts:
                    self._pts.pop()
                else:
                    self.set_mode(None)
                self.update()
                return
            p = self._to_img(ev.position())
            if p is None or ev.button() != Qt.LeftButton:
                return
            if len(self._pts) >= 3:                        # click on the first point = close
                a = self._to_w(*self._pts[0])
                if math.hypot(a.x() - ev.position().x(), a.y() - ev.position().y()) <= 9:
                    self._finish_poly()
                    return
            self._pts.append(p)
            self.update()
            return
        if self.mode is None:
            if ev.button() == Qt.LeftButton:
                hit = self._corner_at(ev.position())
                if hit is not None:
                    self._drag = hit
                    self.setCursor(Qt.ClosedHandCursor)
                    return
            if self.zoom > 1.0001 and ev.button() in (Qt.LeftButton, Qt.MiddleButton):
                self._pan = (ev.position(), self.centre)
                self.setCursor(Qt.ClosedHandCursor)
            return
        if ev.button() != Qt.LeftButton:
            return
        p = self._to_img(ev.position())
        if p is None:
            return
        if self.mode == "box":
            self._a = self._b = p
        elif self.mode == "measure":
            self._pts.append(p)
            if len(self._pts) == 2:
                (x0, y0), (x1, y1) = self._pts
                self.set_mode(None)
                self.measure_done.emit(math.hypot(x1 - x0, y1 - y0))
        self.update()

    def mouseMoveEvent(self, ev):
        if self.mode is None:
            if self._drag is not None:
                kind, idx, k = self._drag
                p = self._to_img(ev.position())
                for j, (kd, ix, pts) in enumerate(self.edit_polys):
                    if kd == kind and ix == idx and p is not None:
                        pts = list(pts)
                        pts[k] = p
                        self.edit_polys[j] = (kd, ix, pts)
                self.update()
                return
            if self._pan is not None:
                (p0, c0) = self._pan
                ox, oy, s = self._geo()
                self.centre = (c0[0] - (ev.position().x() - p0.x()) / s, c0[1] - (ev.position().y() - p0.y()) / s)
                self._clamp_centre()
                self.update()
                return
            hit = self._corner_at(ev.position())
            self.setCursor(Qt.OpenHandCursor if hit is not None else
                           (Qt.SizeAllCursor if self.zoom > 1.0001 else Qt.ArrowCursor))
            return
        if self.mode == "poly":
            self._b = self._to_img(ev.position())
            self.update()
            return
        if self.mode == "box" and self._a is not None:
            self._b = self._to_img(ev.position())
            self.update()
        elif self.mode == "measure" and self._pts:
            self._b = self._to_img(ev.position())
            self.update()

    def mouseReleaseEvent(self, ev):
        if self.mode is None and self._drag is not None:
            kind, idx, _k = self._drag
            self._drag = None
            self.setCursor(Qt.OpenHandCursor)
            for kd, ix, pts in self.edit_polys:
                if kd == kind and ix == idx:
                    self.poly_edited.emit(kind, idx, [tuple(map(float, q)) for q in pts])
            return
        if self.mode is None and self._pan is not None:
            self._pan = None
            self.setCursor(Qt.SizeAllCursor)
            return
        if self.mode == "box" and self._a is not None:
            self._b = self._to_img(ev.position()) or self._b
            (x0, y0), (x1, y1) = self._a, self._b
            self.set_mode(None)
            if abs(x1 - x0) >= 8 and abs(y1 - y0) >= 8:
                self.box_done.emit((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)))

    def keyPressEvent(self, ev):
        if self.mode == "poly" and ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._finish_poly()
            return
        if self.mode == "poly" and ev.key() == Qt.Key_Backspace and self._pts:
            self._pts.pop()
            self.update()
            return
        if ev.key() == Qt.Key_Escape and self.mode:
            self.set_mode(None)
        elif ev.key() == Qt.Key_0 and not self.mode:
            self.reset_zoom()
        else:
            super().keyPressEvent(ev)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#1c212b"))
        if self.img is None:
            p.setPen(QColor("#8b95a7"))
            p.drawText(self.rect(), Qt.AlignCenter, self.placeholder)
            return
        ox, oy, s = self._geo()
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(QRectF(ox, oy, self.img.width() * s, self.img.height() * s), self.img)
        if self.body is not None and not self.mode:
            self._paint_body(p, s)
        if self.status is not None and not self.mode:
            big, small, frz = self.status
            f = p.font()
            f.setPointSizeF(13)
            f.setBold(True)
            p.setFont(f)
            fm = p.fontMetrics()
            wbig = fm.horizontalAdvance(big)
            f2 = p.font()
            f2.setPointSizeF(8.5)
            f2.setBold(False)
            p.setFont(f2)
            wsmall = p.fontMetrics().horizontalAdvance(small)
            bw = max(wbig, wsmall) + 18
            x0, y0 = ox + 8, oy + 8
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 190))
            p.drawRoundedRect(QRectF(x0, y0, bw, 46), 6, 6)
            p.setFont(f)
            p.setPen(QColor(frz) if isinstance(frz, str) else (QColor("#38bdf8") if frz else QColor("#4ade80")))
            p.drawText(QRectF(x0 + 9, y0 + 3, bw, 24), Qt.AlignVCenter | Qt.AlignLeft, big)
            p.setFont(f2)
            p.setPen(QColor("#e5e7eb"))
            p.drawText(QRectF(x0 + 9, y0 + 25, bw, 18), Qt.AlignVCenter | Qt.AlignLeft, small)
        if not self.mode and self.editable and self.edit_polys:
            for kind, _idx, pts in self.edit_polys:     # corners you can drag
                col = QColor("#f87171") if kind == "mask" else QColor("#fb923c")
                if self._drag is not None:              # the shape follows the corner while it is dragged
                    p.setPen(QPen(col, 1.4, Qt.DashLine))
                    ws = [self._to_w(*q) for q in pts]
                    for a, b in zip(ws, ws[1:] + ws[:1]):
                        p.drawLine(a, b)
                p.setPen(QPen(QColor(0, 0, 0, 200), 1.0))
                p.setBrush(col)
                for q in pts:
                    w = self._to_w(*q)
                    p.drawRect(QRectF(w.x() - 3.5, w.y() - 3.5, 7, 7))
        if self.zoom > 1.0001 and not self.mode:
            ztxt = f"Zoom {self.zoom:.1f}×  ·  scroll = zoom  ·  drag = move  ·  0 / double-click = whole picture"
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(QRectF(8, self.height() - 30, 16 + 6.3 * len(ztxt), 22), 5, 5)
            p.setPen(QColor("#ffffff"))
            p.drawText(QRectF(16, self.height() - 30, 6.3 * len(ztxt) + 8, 22), Qt.AlignVCenter | Qt.AlignLeft, ztxt)
        pen = QPen(QColor("#22d3ee"), 1.5, Qt.DashLine)
        p.setPen(pen)
        if self.mode == "box" and self._a is not None and self._b is not None:
            a, b = self._to_w(*self._a), self._to_w(*self._b)
            p.drawRect(QRectF(a, b).normalized())
        if self.mode == "measure":
            p.setPen(QPen(QColor("#facc15"), 2))
            for q in self._pts:
                p.drawEllipse(self._to_w(*q), 4, 4)
            if self._pts and self._b is not None:
                p.drawLine(self._to_w(*self._pts[0]), self._to_w(*self._b))
        if self.mode == "poly" and self._pts:
            col = QColor("#f87171") if self.poly_kind == "mask" else QColor("#22d3ee")
            p.setPen(QPen(col, 1.6, Qt.DashLine))
            ws = [self._to_w(*q) for q in self._pts]
            for a, b in zip(ws, ws[1:]):
                p.drawLine(a, b)
            if self._b is not None:
                p.drawLine(ws[-1], self._to_w(*self._b))
            p.setPen(QPen(col, 1.5))
            p.setBrush(Qt.NoBrush)
            for q in ws:
                p.drawEllipse(q, 3, 3)
            p.drawEllipse(ws[0], 7, 7)                     # click here to close
        if self.mode:
            txt = {"box": "Drag a box (Esc = cancel)", "poly": self.poly_hint}.get(
                self.mode, "Click the nose …" if not self._pts else "… then the tail base (Esc = cancel)")
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            p.drawRoundedRect(QRectF(8, 8, 8 + 7 * len(txt), 24), 5, 5)
            p.setPen(QColor("#ffffff"))
            p.drawText(QRectF(14, 8, 7 * len(txt), 24), Qt.AlignVCenter, txt)


class FreezeTimeline(QWidget):
    """Motion over time, freezing shaded, threshold line, epochs on top, the playhead.  Click / drag = seek."""
    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(88)
        self.setMaximumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.times = None
        self.pos_t = None
        self._cache = None

    def set_result(self, result):
        if result is None:
            self.times = None
        else:
            self.times = result["times"]
            self.motion = result["motion"]
            self.freezing = result["freezing"]
            self.light = result["light"]
            self.thr = float(result["settings"]["threshold_pct"])
            self.epochs = result["epochs"]
        self._cache = None
        self.update()

    def set_position(self, t):
        self.pos_t = t
        self.update()

    def _x(self, t, w):
        t0, t1 = float(self.times[0]), float(self.times[-1])
        return 40 + (w - 48) * (t - t0) / max(1e-9, t1 - t0)

    def _t(self, x, w):
        t0, t1 = float(self.times[0]), float(self.times[-1])
        return t0 + (x - 40) / max(1.0, w - 48) * (t1 - t0)

    def mousePressEvent(self, ev):
        if self.times is not None:
            self.seek.emit(self._t(ev.position().x(), self.width()))

    def mouseMoveEvent(self, ev):
        if self.times is not None and ev.buttons() & Qt.LeftButton:
            self.seek.emit(self._t(ev.position().x(), self.width()))

    def resizeEvent(self, ev):
        self._cache = None
        super().resizeEvent(ev)

    def _render(self):
        from pro_ui import theme
        dark = theme.is_dark()
        col = {"bg": "#1a2131" if dark else "#ffffff", "border": "#334058" if dark else "#e3e8f0",
               "muted": "#8592a8" if dark else "#94a3b8", "label": "#97a3b6" if dark else "#64748b",
               "frz": "#243a5c" if dark else "#cfe0f5", "line": "#6aa5ff" if dark else "#1f6feb"}
        self._dark = dark
        w, h = self.width(), self.height()
        pm = QPixmap(w, h)
        pm.fill(QColor(col["bg"]))
        p = QPainter(pm)
        p.setPen(QColor(col["border"]))
        p.drawRect(0, 0, w - 1, h - 1)
        if self.times is None or len(self.times) < 2:
            p.setPen(QColor(col["muted"]))
            p.drawText(pm.rect(), Qt.AlignCenter, "Motion graph appears here after the analysis")
            p.end()
            return pm
        top, bot = 16, h - 16
        mot = self.motion
        fin = np.isfinite(mot)
        ymax = max(float(np.nanpercentile(mot[fin], 99)) * 1.15 if fin.any() else 1.0, self.thr * 2.0, 1.0)

        def y(v):
            return bot - (bot - top) * min(v, ymax) / ymax
        n = len(self.times)
        # freezing spans
        m = np.concatenate([[False], self.freezing.astype(bool), [False]])
        d = np.diff(m.astype(np.int8))
        for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
            xa, xb = self._x(self.times[a], w), self._x(self.times[b - 1], w)
            p.fillRect(QRectF(xa, top, max(1.0, xb - xa), bot - top), QColor(col["frz"]))
        # epochs
        for i, e in enumerate(self.epochs):
            xa, xb = self._x(e["Start_s"], w), self._x(e["End_s"], w)
            p.fillRect(QRectF(xa, 2, max(1.0, xb - xa - 1), 11), QColor("#f59e0b" if i % 2 == 0 else "#fb923c"))
            p.setPen(QColor("#ffffff"))
            p.drawText(QRectF(xa + 2, 1, max(1.0, xb - xa - 4), 13), Qt.AlignVCenter | Qt.AlignLeft,
                       str(e["Epoch"]))
        # motion: max per pixel column
        cols = max(2, w - 48)
        idx = np.linspace(0, n, cols + 1).astype(int)
        p.setPen(QPen(QColor(col["line"]), 1))
        prev = None
        for c in range(cols):
            seg = mot[idx[c]:max(idx[c] + 1, idx[c + 1])]
            seg = seg[np.isfinite(seg)]
            if not seg.size:
                prev = None
                continue
            x = 40 + c
            pt_hi, pt_lo = QPointF(x, y(float(seg.max()))), QPointF(x, y(float(seg.min())))
            p.drawLine(pt_lo, pt_hi)
            if prev is not None:
                p.drawLine(prev, pt_lo)
            prev = pt_hi
        # light changes
        if self.light.any():
            p.setPen(QPen(QColor("#a855f7"), 1))
            for i in np.where(self.light)[0][:400]:
                x = self._x(self.times[i], w)
                p.drawLine(QPointF(x, top), QPointF(x, top + 6))
        # threshold
        p.setPen(QPen(QColor("#dc2626"), 1, Qt.DashLine))
        p.drawLine(QPointF(40, y(self.thr)), QPointF(w - 8, y(self.thr)))
        p.setPen(QColor(col["label"]))
        p.drawText(QRectF(0, top - 6, 38, 14), Qt.AlignRight | Qt.AlignVCenter, f"{ymax:.0f}%")
        p.drawText(QRectF(0, y(self.thr) - 7, 38, 14), Qt.AlignRight | Qt.AlignVCenter, f"{self.thr:g}%")
        p.drawText(QRectF(40, bot + 1, 80, 14), Qt.AlignLeft, fmt_clock(self.times[0]))
        p.drawText(QRectF(w - 88, bot + 1, 80, 14), Qt.AlignRight, fmt_clock(self.times[-1]))
        p.end()
        return pm

    def paintEvent(self, _ev):
        from pro_ui import theme
        if self._cache is None or self._cache.size() != self.size() or getattr(self, "_dark", None) != theme.is_dark():
            self._cache = self._render()
        p = QPainter(self)
        p.drawPixmap(0, 0, self._cache)
        if self.times is not None and self.pos_t is not None:
            x = self._x(self.pos_t, self.width())
            p.setPen(QPen(QColor("#e6eaf2" if self._dark else "#0f172a"), 1.5))
            p.drawLine(QPointF(x, 2), QPointF(x, self.height() - 2))


class _Relay(QObject):
    """Lives in the GUI thread: worker signals connected to its slots are queued, so the callbacks always run on
    the GUI thread."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    @Slot(object)
    def one(self, a):
        self.fn(a)

    @Slot(str)
    def text(self, a):
        self.fn(a)

    @Slot(float, str)
    def prog(self, a, b):
        self.fn(a, b)

    @Slot(int, object)
    def item(self, a, b):
        self.fn(a, b)

    @Slot(object, str)
    def live(self, a, b):
        self.fn(a, b)


class _Worker(QObject):
    progress = Signal(float, str)
    item = Signal(int, object)
    live = Signal(object, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancel = threading.Event()

    def run(self):
        try:
            self.done.emit(self.fn(self))
        except FZ.Cancelled:
            self.failed.emit("__cancelled__")
        except Exception as exc:                           # shown to the user
            import traceback
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)


class TonesDialog(QDialog):
    """Fear conditioning: tones (CS) at regular times -> epochs (+ baseline and the gaps between them)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add tones (CS)")
        f = QFormLayout(self)
        self.name = QLineEdit("Tone")
        self.first = QDoubleSpinBox(); self.first.setRange(0, 86400); self.first.setValue(120); self.first.setSuffix(" s")
        self.dur = QDoubleSpinBox(); self.dur.setRange(0.1, 3600); self.dur.setValue(30); self.dur.setSuffix(" s")
        self.every = QDoubleSpinBox(); self.every.setRange(0.1, 86400); self.every.setValue(120); self.every.setSuffix(" s")
        self.count = QSpinBox(); self.count.setRange(1, 200); self.count.setValue(3)
        self.gaps = QCheckBox("Also add the baseline and the gaps between tones (ITI)")
        self.gaps.setChecked(True)
        f.addRow("Name", self.name)
        f.addRow("First tone starts at", self.first)
        f.addRow("Each tone lasts", self.dur)
        f.addRow("Tone starts every", self.every)
        f.addRow("Number of tones", self.count)
        f.addRow(self.gaps)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def epochs(self):
        nm = self.name.text().strip() or "Tone"
        t, d, ev, n = self.first.value(), self.dur.value(), self.every.value(), self.count.value()
        out = []
        if self.gaps.isChecked() and t > 0:
            out.append({"name": "Baseline", "start": 0.0, "end": t})
        for i in range(n):
            a = t + i * ev
            out.append({"name": f"{nm} {i + 1}", "start": a, "end": a + d})
            if self.gaps.isChecked() and i < n - 1 and ev > d:
                out.append({"name": f"ITI {i + 1}", "start": a + d, "end": a + ev})
        return out


class ImportEpochsDialog(QDialog):
    """Epochs from a stimulus / protocol log: a table with start + end (or duration), a list of event times or
    a MED-PC data file (tracking.epoch_import)."""

    UNITS = [("seconds", 1.0), ("minutes", 60.0), ("milliseconds", 0.001), ("MED-PC ticks (10 ms)", 0.01)]

    def __init__(self, data, file_name, parent=None):
        super().__init__(parent)
        from tracking import epoch_import as EI
        self.EI = EI
        self.data = data
        self.setWindowTitle("Import epochs")
        self.setMinimumWidth(520)
        f = QFormLayout(self)
        kind = data["kind"]
        f.addRow(QLabel(f"<b>{os.path.basename(file_name)}</b> — " +
                        ("a table of epochs (start + end / duration)" if kind == "table" else
                         "event times (" + data.get("source", "") + ")")))
        self.cmb_series = QComboBox()
        self.name = QLineEdit("Tone")
        self.length = QDoubleSpinBox(); self.length.setRange(0.1, 86400); self.length.setValue(30)
        self.length.setSuffix(" s")
        if kind == "times":
            for k, v in data["series"].items():
                self.cmb_series.addItem(f"{k}  ({len(v)} values: {', '.join(f'{x:g}' for x in v[:4])}"
                                        f"{' …' if len(v) > 4 else ''})", k)
            f.addRow("Event times from", self.cmb_series)
            f.addRow("Each event lasts", self.length)
            f.addRow("Name", self.name)
        self.offset = QDoubleSpinBox(); self.offset.setRange(-86400, 86400); self.offset.setDecimals(2)
        self.offset.setSuffix(" s")
        self.offset.setToolTip("Video time of the log's 0 s (the video started before the protocol: positive).")
        f.addRow("Video time of the log's 0 s", self.offset)
        self.unit = QComboBox()
        for lab, k in self.UNITS:
            self.unit.addItem(lab, k)
        if data.get("source") == "MED-PC":
            self.unit.setCurrentIndex(0)
        f.addRow("Times in the file are in", self.unit)
        self.replace = QCheckBox("Replace the current epochs")
        self.replace.setChecked(True)
        f.addRow("", self.replace)
        self.preview = QLabel("")
        self.preview.setWordWrap(True)
        self.preview.setStyleSheet("color:#334155;")
        f.addRow("Result", self.preview)
        for w in (self.cmb_series, self.unit):
            w.currentIndexChanged.connect(lambda _i: self._update())
        for w in (self.length, self.offset):
            w.valueChanged.connect(lambda _v: self._update())
        self.name.textChanged.connect(lambda _t: self._update())
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)
        self._update()

    def epochs(self):
        sc, off = float(self.unit.currentData()), self.offset.value()
        if self.data["kind"] == "table":
            rows = self.EI.shift_rows(self.data["rows"], off, sc)
        else:
            times = self.data["series"].get(self.cmb_series.currentData()) or []
            rows = self.EI.epochs_from_times(times, self.length.value(), self.name.text().strip() or "Event", off, sc)
        seen = {}
        for r in rows:                                  # the same name twice -> "Tone 1", "Tone 2" …
            seen[r["name"]] = seen.get(r["name"], 0) + 1
        if any(n > 1 for n in seen.values()):
            k = {}
            for r in rows:
                if seen[r["name"]] > 1:
                    k[r["name"]] = k.get(r["name"], 0) + 1
                    r["name"] = f"{r['name']} {k[r['name']]}"
        return rows

    def _update(self):
        rows = self.epochs()
        txt = "; ".join(f"{r['name']} {r['start']:g}–{r['end']:g} s" for r in rows[:6])
        self.preview.setText(f"{len(rows)} epochs: {txt}{' …' if len(rows) > 6 else ''}")


def _save_all(md, res, path, region, settings, subject, group, embed_px, previous, progress, out_dir=None):
    """Excel + CSV + charts + the .btf of one result -> (files, btf data).  Runs in a worker thread.
    out_dir: the chamber's own folder (videos with chambers), else the video's results folder."""
    from output.freezing_report import write_freezing_report
    from tracking import project_file as PF
    files = write_freezing_report(md, res, path, out_dir=out_dir, subject=subject, group=group)
    prev = previous if previous and os.path.normcase(os.path.dirname(previous.get("_path") or "")) == \
        os.path.normcase(files["folder"]) else None
    btf_path = PF.save_freezing(md, res, path, region=region, settings=settings, subject=subject, group=group,
                                results_folder=files["folder"], embed_px=embed_px, previous=prev, progress=progress)
    data = prev if prev is not None else PF.load_freezing(btf_path)
    if prev is None:
        data.pop("md", None)
    data["_path"] = btf_path
    files["btf"] = btf_path
    return files, data


def chamber_folder(video_path, name):
    """results/<video>/freezing/<chamber name>/"""
    safe = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in str(name)).strip() or "Chamber"
    return os.path.join(FZ.default_output_dir(video_path), safe)


MAX_CHAMBERS = 4


def _kpi(title):
    box = QFrame()
    box.setProperty("cssClass", "card")
    v = QVBoxLayout(box)
    v.setContentsMargins(10, 4, 10, 4)
    v.setSpacing(0)
    t = QLabel(title)
    t.setProperty("cssClass", "fieldLabel")
    val = QLabel("—")
    val.setProperty("cssClass", "kpiValue")
    v.addWidget(t)
    v.addWidget(val)
    return box, val


def _num(v, nd=1, unit=""):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(f):
        return "—"
    return f"{f:.{nd}f}{unit}"


class FreezeLiveDialog(QDialog):
    """Freezing – Batch ▸ the video being analysed, live (non-modal; the batch keeps running)."""

    def __init__(self, page):
        super().__init__(page)
        self.setWindowTitle("Freezing – live view")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.resize(900, 640)
        v = QVBoxLayout(self)
        self.canvas = FrameCanvas()
        self.canvas.placeholder = "Waiting for the batch to start…"
        v.addWidget(self.canvas, 1)
        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("font-weight:600;")
        v.addWidget(self.lbl)
        note = QLabel("Red = moving pixels · dots: green still, red moved")
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748b;")
        v.addWidget(note)

    def set_frame(self, bgr, text):
        status = None
        if isinstance(bgr, tuple):
            bgr, status = bgr
        self.canvas.set_bgr(bgr, status)
        self.lbl.setText(text)

    def set_text(self, text):
        self.lbl.setText(text)

    def set_idle(self):
        self.lbl.setText("Nothing running. ▶ Run batch, or double-click a finished video.")

    def showEvent(self, ev):
        self.parent()._live_visible = True
        super().showEvent(ev)

    def hideEvent(self, ev):
        self.parent()._live_visible = False
        super().hideEvent(ev)


# ============================================================================= the page
class FreezingPage(QWidget):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.player = Player()
        self.path = None
        self.regions = {}             # video path -> region (the chamber shown, when the video has chambers)
        self.chambers = {}            # video path -> [{"name", "subject", "spec"}]  (2-4 animals in one video)
        self.ch_cur = {}              # video path -> index of the chamber shown
        self.ch_results = {}          # video path -> {chamber index: stashed result state}
        self.md = None                # MotionData of self.path
        self.md_key = None            # (video, region box, start, end) md was measured on
        self.md_pol = None            # animal colour md was tracked with
        self.md_con = None            # min contrast md was tracked with
        self.md_len = 0.0             # measured body length md was tracked with (0 = automatic)
        self._det_model = None        # background model kept for 🎯 Test detection / ✨ Suggest (FZ.detection_model)
        self.result = None
        self.saved = None             # paths of the last saved report
        self.btf = None               # the BT Freezing file (.btf) of this result (data as loaded / saved)
        self.dirty = False
        self.mode = "single"          # "single" (Freezing – Individual) | "batch" (Freezing – Batch)
        self.live_dlg = None
        self.cur = 0                  # current frame
        self._ring = []
        self.playing = False
        self.speed = 1.0
        self.batch = []               # {path, subject, group, status, summary, folder}
        self.batch_out = None
        self.thread = self.worker = None
        self.busy = None              # "single" | "batch"
        self._sync = False
        self.cfg = FZ.merged_settings(load_json(SETTINGS_PATH, {}))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._advance)
        self._recalc = QTimer(self)
        self._recalc.setSingleShot(True)
        self._recalc.setInterval(120)
        self._recalc.timeout.connect(self._reanalyse)
        self._build()
        self._load_cfg_into_ui()
        QApplication.instance().installEventFilter(self)

    # ------------------------------------------------------------------ layout
    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 16)
        root.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(8)
        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        badge = QLabel()
        badge.setFixedSize(40, 40)
        badge.setPixmap(QIcon(str(HERE / "icons" / "freezing_badge.svg")).pixmap(40, 40))
        title = QLabel("Freezing Analysis — Individual")
        title.setProperty("cssClass", "pageTitle")
        self.title_lbl = title
        title_row.addWidget(badge, 0, Qt.AlignVCenter)
        title_row.addWidget(title, 1, Qt.AlignVCenter)
        left.addLayout(title_row)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._tab_changed)
        self.tabs.addTab(self._build_single(), "Single video")
        self.tabs.addTab(self._build_batch(), "Batch (many videos)")
        self.tabs.tabBar().hide()                       # the side bar chooses: Individual / Batch
        left.addWidget(self.tabs, 1)
        root.addLayout(left, 3)
        root.addWidget(self._build_settings(), 0)

    def _build_single(self):
        outer = QScrollArea()
        outer.setWidgetResizable(True)
        outer.setFrameShape(QFrame.NoFrame)
        outer.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        w = QWidget()
        outer.setWidget(w)
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 4, 0)
        v.setSpacing(6)
        src = QHBoxLayout()
        src.addWidget(QLabel("Video:"))
        self.cmb_video = QComboBox()
        self.cmb_video.setMinimumWidth(150)
        self.cmb_video.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_video.setMinimumContentsLength(12)
        self.cmb_video.activated.connect(lambda i: self.open_video(self.cmb_video.itemData(i)))
        b_open = _small_button("Open…", "Open a video file.")
        b_open.clicked.connect(self._open_dialog)
        self.btn_box = _small_button("⬠ Region", "Draw the area to analyse (click the corners). Only motion "
                                     "inside counts; tracking stays inside.\nDrag a corner to move it · scroll on "
                                     "the picture = zoom.")
        self.btn_box.clicked.connect(lambda: self._start_poly("region"))
        self.btn_mask = _small_button("▨ Mask", "Draw an area to leave out: reflections, a lamp, a cable. "
                                      "Several masks are fine.")
        self.btn_mask.clicked.connect(lambda: self._start_poly("mask"))
        self.btn_nobox = QToolButton()
        self.btn_nobox.setText("Clear ▾")
        self.btn_nobox.setPopupMode(QToolButton.InstantPopup)
        self.btn_nobox.setToolTip("Remove the region / the masks.")
        clear_menu = QMenu(self.btn_nobox)
        clear_menu.addAction("Clear region", lambda: self._clear_region("region"))
        clear_menu.addAction("Clear last mask", lambda: self._clear_region("last mask"))
        clear_menu.addAction("Clear all masks", lambda: self._clear_region("masks"))
        clear_menu.addAction("Clear all", lambda: self._clear_region("all"))
        self.btn_nobox.setMenu(clear_menu)
        self.btn_measure = _small_button("📏 Measure", "Click the nose, then the tail base of the mouse (not its "
                                         "reflection): tracking looks for a mouse of this size; also the threshold.")
        self.btn_measure.clicked.connect(lambda: self._start_canvas("measure"))
        b_btf = _small_button("📂 .btf", "Open a saved freezing result (.btf).")
        b_btf.clicked.connect(self._open_btf_dialog)
        self.btn_ch = QToolButton()
        self.btn_ch.setText("▦ Chambers ▾")
        self.btn_ch.setPopupMode(QToolButton.InstantPopup)
        self.btn_ch.setToolTip("2–4 animals in separate chambers of one video: each chamber gets its own region, "
                               "masks and Animal ID, and its own result.")
        ch_menu = QMenu(self.btn_ch)
        ch_menu.addAction("＋ Add a chamber (draw its floor)", self._ch_add)
        ch_menu.addAction("✎ Animal ID of this chamber…", self._ch_rename)
        ch_menu.addAction("Remove this chamber", self._ch_remove)
        ch_menu.addAction("Remove all chambers", self._ch_clear)
        self.btn_ch.setMenu(ch_menu)
        self.cmb_ch = QComboBox()
        self.cmb_ch.setToolTip("The chamber shown (its region, masks and result).")
        self.cmb_ch.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.cmb_ch.activated.connect(lambda i: self._ch_select(i))
        self.cmb_ch.hide()
        src.addWidget(self.cmb_video, 1)
        src.addWidget(self.cmb_ch)
        src.addWidget(b_open)
        src.addWidget(b_btf)
        v.addLayout(src)
        tools = QHBoxLayout()                       # second row: drawing tools (the first row stays short)
        for wd in (self.btn_box, self.btn_mask, self.btn_nobox, self.btn_ch, self.btn_measure):
            tools.addWidget(wd)
        tools.addStretch(1)
        v.addLayout(tools)

        self.canvas = FrameCanvas()
        self.canvas.setMinimumSize(320, 220)
        self.canvas.box_done.connect(self._box_drawn)
        self.canvas.poly_done.connect(self._poly_drawn)
        self.canvas.poly_edited.connect(self._poly_edited)
        self.canvas.measure_done.connect(self._measured)
        v.addWidget(self.canvas, 1)

        ctl = QHBoxLayout()
        self.btn_play = QToolButton()
        self.btn_play.setText("▶")
        self.btn_play.setToolTip("Play / pause (Space)")
        self.btn_play.clicked.connect(self.toggle_play)
        b_back = QToolButton()
        b_back.setText("◀|")
        b_back.setToolTip("One frame back (←)")
        b_back.clicked.connect(lambda: self.step(-1))
        b_fwd = QToolButton()
        b_fwd.setText("|▶")
        b_fwd.setToolTip("One frame forward (→)")
        b_fwd.clicked.connect(lambda: self.step(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(lambda val: self.show_frame(val))
        self.slider.sliderPressed.connect(self.pause)
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("color:#000000; font-weight:600;")
        self.cmb_speed = QComboBox()
        for s in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            self.cmb_speed.addItem(f"{s:g}×", s)
        self.cmb_speed.setCurrentIndex(2)
        self.cmb_speed.currentIndexChanged.connect(lambda _i: self._speed_changed())
        self.chk_overlay = QCheckBox("Moving pixels")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setToolTip("Red = pixels that changed more than the motion cutoff (what is counted as "
                                    "movement).")
        self.chk_overlay.toggled.connect(lambda _b: self.show_frame(self.cur))
        for wd in (self.btn_play, b_back, b_fwd):
            ctl.addWidget(wd)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.lbl_time)
        ctl.addWidget(self.cmb_speed)
        ctl.addWidget(self.chk_overlay)
        v.addLayout(ctl)

        self.timeline = FreezeTimeline()
        self.timeline.seek.connect(self._seek_time)
        v.addWidget(self.timeline)

        self.result_bar = QFrame()
        self.result_bar.setProperty("cssClass", "card")
        rb = QHBoxLayout(self.result_bar)
        rb.setContentsMargins(12, 8, 12, 8)
        self.lbl_result = QLabel("")
        self.lbl_result.setStyleSheet("color:#15803d; font-size:10.5pt;")
        self.lbl_result.setWordWrap(True)
        self.btn_to_results = QPushButton("📊  Analysis && Results")
        self.btn_to_results.setProperty("cssClass", "primary")
        self.btn_to_results.setCursor(Qt.PointingHandCursor)
        self.btn_to_results.setToolTip("Graphs, tables, epochs, bouts, frame review and statistics of this result.")
        self.btn_to_results.clicked.connect(self.open_in_results)
        self.btn_body = _small_button("👁 Body size", "See the body size the threshold is based on, drawn on the "
                                                     "video.")
        self.btn_body.clicked.connect(self.show_body_size)
        rb.addWidget(self.lbl_result, 1)
        rb.addWidget(self.btn_body)
        rb.addWidget(self.btn_to_results)
        self.result_bar.hide()
        v.addWidget(self.result_bar)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setStyleSheet("color:#475569; font-size:9pt;")
        v.addWidget(self.lbl_info)
        self.lbl_warn = QLabel("")
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setStyleSheet("color:#b45309; font-size:9pt; font-weight:600;")
        self.lbl_warn.hide()
        v.addWidget(self.lbl_warn)
        return outer

    def _build_batch(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 8, 0, 0)
        v.setSpacing(6)
        row = QHBoxLayout()
        for text, fn, tip in (("＋ Add videos…", self._batch_add_files, "Add video files."),
                              ("＋ Add folder…", self._batch_add_folder, "Add every video of a folder."),
                              ("＋ Experiment videos", self._batch_add_experiment,
                               "Add the videos loaded on the Experiment page."),
                              ("Remove", self._batch_remove, "Remove the selected videos (Del)."),
                              ("Clear", self._batch_clear, "Empty the list.")):
            b = _small_button(text, tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        row.addStretch(1)
        b_live = _small_button("👁  Live view", "Watch the video being analysed.")
        b_live.clicked.connect(self.show_live)
        row.addWidget(b_live)
        v.addLayout(row)
        self.tbl_batch = QTableWidget(0, 8)
        self.tbl_batch.setHorizontalHeaderLabels(["Video", "Subject", "Group", "Region", "Status",
                                                  "Freezing (%)", "Bouts", "Check"])
        self.tbl_batch.verticalHeader().setVisible(False)
        self.tbl_batch.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_batch.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_batch.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
                                       | QAbstractItemView.SelectedClicked)
        hh = self.tbl_batch.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 8):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.tbl_batch.itemChanged.connect(self._batch_item_changed)
        self.tbl_batch.cellDoubleClicked.connect(self._batch_open_row)
        v.addWidget(self.tbl_batch, 1)
        hint = QLabel("Type Subject / Group in the table. Double-click: done → result · running → live · "
                      "waiting → set its region in Individual.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#64748b; font-size:9pt;")
        v.addWidget(hint)
        self.chk_first_box = QCheckBox("No own region → use the first video's region / masks")
        self.chk_first_box.setChecked(True)
        self.chk_first_box.toggled.connect(lambda _b: self._batch_refresh())
        v.addWidget(self.chk_first_box)
        orow = QHBoxLayout()
        orow.addWidget(QLabel("Summary folder:"))
        self.lbl_out = QLabel("(the results folder next to the first video)")
        self.lbl_out.setStyleSheet("color:#475569;")
        b_out = _small_button("Change…", "Where the batch summary workbook is saved.")
        b_out.clicked.connect(self._batch_choose_out)
        orow.addWidget(self.lbl_out, 1)
        orow.addWidget(b_out)
        v.addLayout(orow)
        self.lbl_batch = QLabel("")
        self.lbl_batch.setWordWrap(True)
        self.lbl_batch.setStyleSheet("font-weight:600; color:#1f2a44;")
        v.addWidget(self.lbl_batch)
        self.prog_batch = QProgressBar()
        self.prog_batch.setRange(0, 1000)
        self.prog_batch.setValue(0)
        self.prog_batch.setTextVisible(False)
        self.prog_batch.setFixedHeight(10)
        v.addWidget(self.prog_batch)
        return w

    def _build_settings(self):
        panel = QFrame()
        panel.setProperty("cssClass", "card")
        panel.setMinimumWidth(400)
        panel.setMaximumWidth(450)
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        inner = QWidget()
        pv = QVBoxLayout(inner)
        pv.setContentsMargins(14, 12, 14, 12)
        pv.setSpacing(6)
        cap = QLabel("Settings")
        cap.setProperty("cssClass", "cardTitle")
        pv.addWidget(cap)
        f = QFormLayout()
        f.setLabelAlignment(Qt.AlignLeft)
        f.setHorizontalSpacing(10)
        f.setVerticalSpacing(6)

        self.sp_start = QDoubleSpinBox()
        self.sp_start.setRange(0, 86400)
        self.sp_start.setDecimals(1)
        self.sp_start.setSuffix(" s")
        self.sp_start.setToolTip("Start of the analysis (video time).")
        self.sp_end = QDoubleSpinBox()
        self.sp_end.setRange(0, 86400)
        self.sp_end.setDecimals(1)
        self.sp_end.setSuffix(" s")
        self.sp_end.setSpecialValueText("end of video")
        self.sp_end.setToolTip("End of the analysis (0 = end of video).")
        for sp in (self.sp_start, self.sp_end):
            sp.setKeyboardTracking(False)               # type the time, Enter / Tab = done
        f.addRow("Start (s)", self.sp_start)
        f.addRow("End (s)", self.sp_end)

        self.cmb_area = QComboBox()
        for k, lab in FZ.MOTION_AREAS:
            self.cmb_area.addItem(lab, k)
        self.cmb_area.setToolTip("Whole region: every pixel of the region (side or top view).\n"
                                 "Around the animal: a circle around the tracked animal.\n"
                                 "Body outline points: head, legs and body only — the tail is ignored (top view).")
        f.addRow("Count motion", self.cmb_area)
        self.sp_thr = QDoubleSpinBox()
        self.sp_thr.setRange(0.5, 500)
        self.sp_thr.setDecimals(1)
        self.sp_thr.setSingleStep(0.5)
        self.sp_thr.setSuffix(" % of body")
        self.sp_thr.setToolTip("Motion below this = still.  The same threshold in changed pixels is shown below "
                               "(after an analysis).")
        f.addRow("Threshold", self.sp_thr)
        self.lbl_thr_px = QLabel("")
        self.lbl_thr_px.setStyleSheet("color:#64748b; font-size:8.5pt;")
        f.addRow("", self.lbl_thr_px)
        self.sp_min = QDoubleSpinBox()
        self.sp_min.setRange(0.1, 30)
        self.sp_min.setDecimals(2)
        self.sp_min.setSingleStep(0.25)
        self.sp_min.setSuffix(" s")
        self.sp_min.setToolTip("Still at least this long = freezing.")
        f.addRow("Min freeze", self.sp_min)
        brow = QHBoxLayout()
        self.lbl_body = QLabel("Auto (from tracking)")
        self.lbl_body.setStyleSheet("color:#334155;")
        b_auto_body = _small_button("Auto", "Forget the measured length: size from tracking.")
        b_auto_body.clicked.connect(self._body_auto)
        brow.addWidget(self.lbl_body, 1)
        brow.addWidget(b_auto_body)
        f.addRow("Body size", brow)
        self.chk_light = QCheckBox("Light-change guard")
        self.chk_light.setToolTip("Skip frames where the whole picture changes (light on / off).")
        f.addRow("", self.chk_light)
        pv.addLayout(f)

        # advanced settings: folded away (the defaults are right for most videos)
        self.btn_adv = QToolButton()
        self.btn_adv.setCheckable(True)
        self.btn_adv.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_adv.setArrowType(Qt.RightArrow)
        self.btn_adv.setText("Advanced (colour, cutoff, blips)")
        self.btn_adv.setStyleSheet("QToolButton{border:none; font-weight:600; color:#334155;}")
        self.btn_adv.toggled.connect(self._toggle_advanced)
        pv.addWidget(self.btn_adv)
        self.adv_box = QWidget()
        fa = QFormLayout(self.adv_box)
        fa.setContentsMargins(0, 0, 0, 0)
        fa.setLabelAlignment(Qt.AlignLeft)
        fa.setHorizontalSpacing(10)
        fa.setVerticalSpacing(6)
        self.cmb_colour = QComboBox()
        for k, lab in FZ.COLOURS:
            self.cmb_colour.addItem(lab, k)
        self.cmb_colour.setToolTip("Gray (classic, less noise — recommended) or RGB (any colour channel).")
        fa.addRow("Colour", self.cmb_colour)
        crow = QHBoxLayout()
        self.chk_cut_auto = QCheckBox("Auto")
        self.chk_cut_auto.setToolTip("Auto = 2 × this video's camera noise (spiky webcam noise is limited).")
        self.sp_cut = QSpinBox()
        self.sp_cut.setRange(FZ.CUTOFF_MIN, FZ.CUTOFF_MAX)
        self.sp_cut.setToolTip("Change (grey levels) that counts as a moving pixel.")
        b_cal = _small_button("Empty video…", "Calibrate with a video of the empty chamber.")
        b_cal.clicked.connect(self._calibrate_empty)
        crow.addWidget(self.chk_cut_auto)
        crow.addWidget(self.sp_cut)
        crow.addWidget(b_cal)
        fa.addRow("Cutoff", crow)
        self.sp_blip = QDoubleSpinBox()
        self.sp_blip.setRange(0, 2)
        self.sp_blip.setDecimals(2)
        self.sp_blip.setSingleStep(0.05)
        self.sp_blip.setSuffix(" s")
        self.sp_blip.setSpecialValueText("off")
        self.sp_blip.setToolTip("Shorter movements do not break a freeze (off = classic).")
        fa.addRow("Ignore blips", self.sp_blip)
        self.chk_trim = QCheckBox("Leave out light off / on at start / end")
        self.chk_trim.setToolTip("When the room light is switched off at the end (or on late at the start), those "
                                 "frames are not analysed: the animal cannot be seen there.")
        fa.addRow("", self.chk_trim)
        pv.addWidget(self.adv_box)
        self.adv_box.setVisible(False)

        cap_t = QLabel("Tracking")
        cap_t.setProperty("cssClass", "cardTitle")
        cap_t.setToolTip("Used by 'Around the animal', 'Body outline points' and the automatic body size.")
        pv.addWidget(cap_t)
        ft = QFormLayout()
        ft.setLabelAlignment(Qt.AlignLeft)
        ft.setHorizontalSpacing(10)
        ft.setVerticalSpacing(6)
        self.cmb_pol = QComboBox()
        for k, lab in POLARITIES:
            self.cmb_pol.addItem(lab, k)
        self.cmb_pol.setToolTip("Darker or lighter than the floor.")
        ft.addRow("Animal colour", self.cmb_pol)
        self.sp_con = QDoubleSpinBox()
        self.sp_con.setRange(2, 120)
        self.sp_con.setDecimals(0)
        self.sp_con.setSingleStep(1)
        self.sp_con.setToolTip("How much darker / lighter than the floor a pixel must be to count as animal (grey "
                               "levels).\nToo HIGH = the animal is lost (it often differs by only 20–40 levels);\n"
                               "too low = shadows / reflections are picked up.\n"
                               "✨ Suggest measures the animal and picks about half of its weakest contrast.")
        ft.addRow("Min contrast", self.sp_con)
        trow = QHBoxLayout()
        self.btn_suggest = _small_button("✨ Suggest", "Measure the animal colour, contrast and size from this video "
                                                      "and fill in the tracking settings.")
        self.btn_suggest.clicked.connect(self.suggest_settings)
        self.btn_test = _small_button("🎯 Test detection", "Track ~1 s up to the shown frame and mark the animal.")
        self.btn_test.clicked.connect(self.test_detection)
        trow.addWidget(self.btn_suggest)
        trow.addWidget(self.btn_test)
        trow.addStretch(1)
        ft.addRow("", trow)
        pv.addLayout(ft)
        for wd in (self.cmb_area, self.cmb_pol, self.cmb_colour):
            wd.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            wd.setMinimumContentsLength(14)
        for wd in (self.sp_start, self.sp_end, self.sp_cut, self.sp_thr, self.sp_min, self.sp_blip):
            wd.setMinimumWidth(70)

        cap2 = QLabel("Epochs")
        cap2.setProperty("cssClass", "cardTitle")
        pv.addWidget(cap2)
        self.tbl_ep = QTableWidget(0, 3)
        self.tbl_ep.setHorizontalHeaderLabels(["Name", "Start (s)", "End (s)"])
        self.tbl_ep.verticalHeader().setVisible(False)
        self.tbl_ep.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_ep.verticalHeader().setDefaultSectionSize(max(28, self.tbl_ep.fontMetrics().height() + 14))
        self.tbl_ep.setMinimumHeight(84)                 # both button rows visible on a 900-px screen
        self.tbl_ep.setMaximumHeight(170)
        self.tbl_ep.itemChanged.connect(lambda _it: self._settings_changed())
        pv.addWidget(self.tbl_ep)
        erow = QHBoxLayout()
        erow2 = QHBoxLayout()
        for text, fn, tip in (("＋ Add", self._ep_add, "Add an epoch (video seconds)."),
                              ("＋ Tones…", self._ep_tones, "Tones at regular times (+ baseline and ITI)."),
                              ("⤒ Import…", self._ep_import, "Epochs from a stimulus / protocol log: a CSV or Excel "
                                                            "table (start + end or duration), a list of event "
                                                            "times, or a MED-PC data file."),
                              ("Remove", self._ep_remove, "Remove the selected epoch."),
                              ("Clear", self._ep_clear, "Remove every epoch.")):
            b = _small_button(text, tip)
            b.clicked.connect(fn)
            (erow2 if text in ("Remove", "Clear") else erow).addWidget(b)
        erow.addStretch(1)
        erow2.addStretch(1)
        pv.addLayout(erow)
        pv.addLayout(erow2)
        brow2 = QHBoxLayout()
        brow2.addWidget(QLabel("Time bins"))
        self.sp_bin = QDoubleSpinBox()
        self.sp_bin.setRange(0, 86400)
        self.sp_bin.setDecimals(0)
        self.sp_bin.setSuffix(" s")
        self.sp_bin.setSpecialValueText("none")
        self.sp_bin.setToolTip("Freezing per time bin.")
        brow2.addWidget(self.sp_bin, 1)
        pv.addLayout(brow2)

        prow = QHBoxLayout()
        self.btn_presets = QToolButton()
        self.btn_presets.setText("☰ Presets")
        self.btn_presets.setToolTip("Named settings: your lab's protocol (threshold, min freeze, cutoff, colour ... "
                                    "and, if you want, the epochs) in one click. Save, share (export / import) and "
                                    "reuse them.")
        self.btn_presets.setPopupMode(QToolButton.InstantPopup)
        self.menu_presets = QMenu(self.btn_presets)
        self.menu_presets.aboutToShow.connect(self._fill_preset_menu)
        self.btn_presets.setMenu(self.menu_presets)
        prow.addWidget(self.btn_presets)
        b_reset = QPushButton("Reset to defaults")
        b_reset.setProperty("cssClass", "link")
        b_reset.setCursor(Qt.PointingHandCursor)
        b_reset.clicked.connect(self._reset_defaults)
        prow.addWidget(b_reset)
        prow.addStretch(1)
        pv.addLayout(prow)
        pv.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll, 1)
        act = QWidget()
        pv = QVBoxLayout(act)
        pv.setContentsMargins(14, 6, 14, 12)
        pv.setSpacing(6)

        self.lbl_stale = QLabel("")
        self.lbl_stale.setWordWrap(True)
        self.lbl_stale.setStyleSheet("color:#b45309; font-size:9pt;")
        pv.addWidget(self.lbl_stale)
        self.lbl_prog = QLabel("")
        self.lbl_prog.setWordWrap(True)
        self.lbl_prog.setStyleSheet("font-weight:600; color:#1f2a44;")
        self.lbl_prog.hide()
        pv.addWidget(self.lbl_prog)
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setTextVisible(False)               # the text is in the label above (readable over any colour)
        self.prog.setFixedHeight(10)
        self.prog.hide()
        pv.addWidget(self.prog)
        self.btn_run = QPushButton("❄  Analyse this video")
        self.btn_run.setProperty("cssClass", "primary")
        self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self._run_clicked)
        pv.addWidget(self.btn_run)
        arow = QHBoxLayout()
        self.btn_save = _small_button("⤓ Save results", "Save Excel, CSV and graphs with the current settings.")
        self.btn_save.clicked.connect(self.save_results)
        self.btn_folder = _small_button("📂 Results folder", "Open the results folder.")
        self.btn_folder.clicked.connect(self._open_folder)
        arow.addWidget(self.btn_save)
        arow.addWidget(self.btn_folder)
        pv.addLayout(arow)
        b_val = QPushButton("≟ Compare with hand scoring…")
        b_val.setProperty("cssClass", "link")
        b_val.setCursor(Qt.PointingHandCursor)
        b_val.setToolTip("Validation: BT's freezing (.btf) against your hand scoring (Event Scoring) of the same "
                         "videos — frame agreement, Cohen's kappa, Pearson r, ICC, Bland-Altman.")
        b_val.clicked.connect(self._validate)
        pv.addWidget(b_val, 0, Qt.AlignLeft)
        outer.addWidget(act, 0)

        for wd in (self.sp_thr, self.sp_min, self.sp_blip, self.sp_bin, self.sp_cut):
            wd.valueChanged.connect(lambda _v: self._settings_changed())
        for wd in (self.chk_cut_auto, self.chk_light, self.chk_trim):
            wd.toggled.connect(lambda _b: self._settings_changed())
        for wd in (self.sp_start, self.sp_end):
            wd.valueChanged.connect(lambda _v: self._settings_changed())
        self.cmb_area.currentIndexChanged.connect(lambda _i: self._settings_changed())
        self.cmb_colour.currentIndexChanged.connect(lambda _i: self._settings_changed())
        self.cmb_pol.currentIndexChanged.connect(lambda _i: self._settings_changed())
        self.sp_con.valueChanged.connect(lambda _v: self._settings_changed())
        self.chk_cut_auto.toggled.connect(lambda b: self.sp_cut.setEnabled(not b))
        return panel

    # ------------------------------------------------------------------ settings
    def _load_cfg_into_ui(self):
        c = self.cfg
        self._sync = True
        try:
            self.sp_start.setValue(float(c.get("start_time") or 0))
            self.sp_end.setValue(float(c.get("end_time") or 0))
            self.cmb_area.setCurrentIndex(max(0, self.cmb_area.findData(c.get("motion_area", "region"))))
            self.cmb_colour.setCurrentIndex(max(0, self.cmb_colour.findData(c.get("colour", "gray"))))
            self.cmb_pol.setCurrentIndex(max(0, self.cmb_pol.findData(c.get("polarity", "auto"))))
            self.sp_con.setValue(float(c.get("min_contrast") or 12))
            self.chk_cut_auto.setChecked(bool(c.get("cutoff_auto", True)))
            self.sp_cut.setValue(int(c.get("cutoff") or 10))
            self.sp_cut.setEnabled(not self.chk_cut_auto.isChecked())
            self.sp_thr.setValue(float(c.get("threshold_pct") or 20))
            self.sp_min.setValue(float(c.get("min_freeze_s") or 1.0))
            self.sp_blip.setValue(float(c.get("ignore_blips_s") or 0))
            self.chk_light.setChecked(bool(c.get("light_guard", True)))
            self.chk_trim.setChecked(bool(c.get("trim_lights", True)))
            self.btn_adv.setChecked(bool(c.get("_advanced_open", False)))
            self.sp_bin.setValue(float(c.get("bin_s") or 0))
            self.tbl_ep.setRowCount(0)
            for e in c.get("epochs") or []:
                self._ep_row(e.get("name", ""), e.get("start", 0), e.get("end", 0))
            self._body_label()
        finally:
            self._sync = False

    def settings(self):
        eps = []
        for r in range(self.tbl_ep.rowCount()):
            def cell(c):
                it = self.tbl_ep.item(r, c)
                return it.text().strip() if it is not None else ""
            try:
                a, b = float(cell(1).replace(",", ".")), float(cell(2).replace(",", "."))
            except ValueError:
                continue
            if b > a:
                eps.append({"name": cell(0) or f"{a:g}-{b:g} s", "start": a, "end": b})
        end = self.sp_end.value()
        return {"start_time": self.sp_start.value(), "end_time": end if end > 0 else None,
                "motion_area": self.cmb_area.currentData(), "polarity": self.cmb_pol.currentData(),
                "min_contrast": self.sp_con.value(),
                "colour": self.cmb_colour.currentData() or "gray",
                "cutoff_auto": self.chk_cut_auto.isChecked(), "cutoff": self.sp_cut.value(),
                "threshold_pct": self.sp_thr.value(), "min_freeze_s": self.sp_min.value(),
                "ignore_blips_s": self.sp_blip.value(), "light_guard": self.chk_light.isChecked(),
                "trim_lights": self.chk_trim.isChecked(),
                "bin_s": self.sp_bin.value(), "epochs": eps,
                "body_length_px": float(self.cfg.get("body_length_px") or 0.0)}

    def _save_cfg(self):
        self.cfg.update(self.settings())
        try:
            save_json(SETTINGS_PATH, self.cfg)
        except Exception:
            pass

    def _key(self, s, path=None):
        """The video + the part of it the stored motion was measured on."""
        path = path or self.path
        reg = json.dumps(FZ.region_json(self.regions.get(path)), sort_keys=True)
        return (path, reg, round(float(s["start_time"] or 0), 2), s["end_time"])

    def _compatible(self, s):
        """True when the stored motion can show these settings without reading the video again."""
        md = self.md
        if md is None or self.md_key is None or self.md_key != self._key(s):
            return False
        if FZ.needs_tracking(s) and (not md.tracked or s["polarity"] != self.md_pol
                                     or float(s.get("min_contrast") or 12) != float(self.md_con or 12)
                                     or float(s.get("body_length_px") or 0) != float(self.md_len or 0)):
            return False
        if (s.get("colour") or "gray") != (getattr(md, "colour", "gray") or "gray"):
            return False
        if s["motion_area"] == "body" and not md.has_body():
            return False
        if s["motion_area"] == "animal":
            if not md.has_disc() or getattr(md, "disc_provisional", False):
                return False                        # (the Classic one-pass circle is only for the result check)
            want = float(s["body_length_px"] or 0) or md.body_len_full
            if abs(want - md.disc_len) > 0.5:
                return False
        return True

    def _settings_changed(self):
        if self._sync:
            return
        self._save_cfg()
        if self.md is not None:
            if self._compatible(self.settings()):
                self.lbl_stale.setText("")
                self._recalc.start()
            else:
                self.lbl_stale.setText("Changed — press ❄ Analyse again.")

    def _toggle_advanced(self, on):
        self.adv_box.setVisible(bool(on))
        self.btn_adv.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.cfg["_advanced_open"] = bool(on)
        try:
            save_json(SETTINGS_PATH, self.cfg)
        except Exception:
            pass

    # presets
    def _presets(self):
        d = load_json(PRESETS_PATH, {})
        return d if isinstance(d, dict) else {}

    def _fill_preset_menu(self):
        m = self.menu_presets
        m.clear()
        m.addSection("Built in")
        for name, vals in BUILTIN_PRESETS.items():
            m.addAction(name).triggered.connect(lambda _c=False, n=name, v=vals: self._apply_preset(n, v))
        mine = self._presets()
        if mine:
            m.addSection("Yours")
            for name in sorted(mine, key=str.lower):
                m.addAction(name).triggered.connect(lambda _c=False, n=name: self._apply_preset(n, self._presets()[n]))
        m.addSeparator()
        m.addAction("Save the current settings as a preset…").triggered.connect(self._save_preset)
        m.addAction("Import a preset file…").triggered.connect(self._import_preset)
        a = m.addAction("Export a preset…")
        a.triggered.connect(self._export_preset)
        a.setEnabled(bool(mine))
        a = m.addAction("Delete a preset…")
        a.triggered.connect(self._delete_preset)
        a.setEnabled(bool(mine))

    def _apply_preset(self, name, vals):
        vals = {k: v for k, v in dict(vals).items() if k in PRESET_KEYS + PRESET_PROTOCOL_KEYS}
        self.cfg.update(vals)
        self._load_cfg_into_ui()
        self._settings_changed()
        extra = " (with its epochs)" if "epochs" in vals else ""
        self.ctrl.status.showMessage(f"Preset '{name}' applied{extra}.", 8000)

    def _save_preset(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save preset", "Name of the preset (e.g. 'Lab protocol — context "
                                        "test, 2 s min freeze'):")
        name = (name or "").strip()
        if not ok or not name:
            return
        if name in BUILTIN_PRESETS:
            QMessageBox.warning(self, "Save preset", "That name is a built-in preset: choose another name.")
            return
        s = self.settings()
        vals = {k: s[k] for k in PRESET_KEYS if k in s}
        if s.get("epochs") and QMessageBox.question(
                self, "Save preset", f"Also keep the {len(s['epochs'])} epochs and the time window in the preset "
                "(the protocol of the session)?") == QMessageBox.Yes:
            vals.update({k: s.get(k) for k in PRESET_PROTOCOL_KEYS})
        mine = self._presets()
        if name in mine and QMessageBox.question(self, "Save preset", f"Replace the preset '{name}'?") \
                != QMessageBox.Yes:
            return
        mine[name] = vals
        save_json(PRESETS_PATH, mine)
        self.ctrl.status.showMessage(f"Preset '{name}' saved.", 6000)

    def _pick_preset(self, title):
        from PySide6.QtWidgets import QInputDialog
        mine = self._presets()
        if not mine:
            return None
        name, ok = QInputDialog.getItem(self, title, "Preset:", sorted(mine, key=str.lower), 0, False)
        return name if ok and name in mine else None

    def _export_preset(self):
        name = self._pick_preset("Export a preset")
        if not name:
            return
        safe = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in name).strip() or "preset"
        fn, _ = QFileDialog.getSaveFileName(self, "Export the preset", str(Path.home() / f"{safe}.btpreset"),
                                            "BT freezing preset (*.btpreset)")
        if fn:
            save_json(fn, {"format": "BT freezing preset", "version": 1, "name": name,
                           "settings": self._presets()[name]})

    def _import_preset(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Import a preset", str(Path.home()),
                                            "BT freezing preset (*.btpreset *.json)")
        if not fn:
            return
        d = load_json(fn, None)
        if not isinstance(d, dict) or d.get("format") != "BT freezing preset" or not isinstance(d.get("settings"), dict):
            QMessageBox.warning(self, "Import preset", "This is not a BT freezing preset file.")
            return
        name = str(d.get("name") or Path(fn).stem)
        vals = {k: v for k, v in d["settings"].items() if k in PRESET_KEYS + PRESET_PROTOCOL_KEYS}
        mine = self._presets()
        while name in mine or name in BUILTIN_PRESETS:
            name += " (imported)"
        mine[name] = vals
        save_json(PRESETS_PATH, mine)
        self._apply_preset(name, vals)

    def _delete_preset(self):
        name = self._pick_preset("Delete a preset")
        if name and QMessageBox.question(self, "Delete preset", f"Delete the preset '{name}'?") == QMessageBox.Yes:
            mine = self._presets()
            mine.pop(name, None)
            save_json(PRESETS_PATH, mine)

    def _reset_defaults(self):
        keep = float(self.cfg.get("body_length_px") or 0)
        self.cfg = FZ.merged_settings({})
        self.cfg["body_length_px"] = keep
        self._load_cfg_into_ui()
        self._settings_changed()

    def _body_label(self):
        L = float(self.cfg.get("body_length_px") or 0)
        dots_len = float(getattr(self.md, "body_len_dots", 0.0) or 0.0) if self.md is not None else 0.0
        if L > 0:
            self.lbl_body.setText(f"Measured: {L:.0f} px long")
        elif self.cfg.get("motion_area") == "body" and dots_len > 0:
            self.lbl_body.setText(f"{dots_len:.0f} px (body outline)")
        elif self.md is not None and self.md.body_len_full > 0:
            self.lbl_body.setText(f"{self.md.body_len_full:.0f} px (tracking)")
        else:
            self.lbl_body.setText("Auto (from tracking)")

    def _body_auto(self):
        self.cfg["body_length_px"] = 0.0
        self._body_label()
        self._settings_changed()

    def _measured(self, length):
        if length < 4:
            return
        self.cfg["body_length_px"] = float(length)
        self._body_label()
        self.ctrl.status.showMessage(f"Animal length {length:.0f} px — used for tracking and the threshold.", 6000)
        self._settings_changed()

    # epochs
    def _ep_row(self, name, a, b):
        r = self.tbl_ep.rowCount()
        self.tbl_ep.blockSignals(True)
        self.tbl_ep.insertRow(r)
        for c, v in enumerate((str(name), f"{float(a):g}", f"{float(b):g}")):
            self.tbl_ep.setItem(r, c, QTableWidgetItem(v))
        self.tbl_ep.blockSignals(False)

    def _ep_add(self):
        t = self.cur / max(1e-6, self.player.fps) if self.player.cap is not None else 0.0
        self._ep_row(f"Epoch {self.tbl_ep.rowCount() + 1}", round(t, 1), round(t + 30, 1))
        self._settings_changed()

    def _ep_tones(self):
        dlg = TonesDialog(self)
        if dlg.exec() == QDialog.Accepted:
            for e in dlg.epochs():
                self._ep_row(e["name"], e["start"], e["end"])
            self._settings_changed()

    def _ep_import(self):
        from tracking.epoch_import import read_epoch_file
        fn, _ = QFileDialog.getOpenFileName(self, "Import epochs (stimulus / protocol log)",
                                            os.path.dirname(self.path or ""),
                                            "Logs (*.csv *.tsv *.txt *.xlsx *.xls *.dat);;All files (*)")
        if not fn:
            return
        try:
            data = read_epoch_file(fn)
        except Exception as exc:
            QMessageBox.warning(self, "Import epochs", f"Could not read epochs from this file:\n{exc}")
            return
        dlg = ImportEpochsDialog(data, fn, self)
        if dlg.exec() != QDialog.Accepted:
            return
        rows = dlg.epochs()
        if dlg.replace.isChecked():
            self.tbl_ep.setRowCount(0)
        for e in rows:
            self._ep_row(e["name"], e["start"], e["end"])
        self._settings_changed()
        self.ctrl.status.showMessage(f"{len(rows)} epochs imported from {os.path.basename(fn)}.", 6000)

    def _ep_remove(self):
        rows = sorted({i.row() for i in self.tbl_ep.selectedIndexes()}, reverse=True)
        for r in rows:
            self.tbl_ep.removeRow(r)
        self._settings_changed()

    def _ep_clear(self):
        self.tbl_ep.setRowCount(0)
        self._settings_changed()

    def _calibrate_empty(self):
        path, _ = QFileDialog.getOpenFileName(self, "Video of the empty chamber (no animal)",
                                              str(Path(self.path).parent if self.path else Path.home()),
                                              VIDEO_FILTER)
        if not path:
            return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            r = FZ.calibrate_empty_video(path, self.regions.get(self.path))
        except Exception as exc:
            QMessageBox.warning(self, "Calibration", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.chk_cut_auto.setChecked(False)
        self.sp_cut.setValue(int(r["cutoff"]))
        QMessageBox.information(self, "Calibration", f"Camera noise (99.99th percentile): {r['p9999']:.2f} grey "
                                f"levels over {r['frames']} frames.\nMotion cutoff = 2 × that = {r['cutoff']}.")

    # ------------------------------------------------------------------ video
    def refresh_sources(self):
        self.cmb_video.blockSignals(True)
        self.cmb_video.clear()
        seen = set()
        for it in getattr(self.ctrl, "exp_videos", []):
            self.cmb_video.addItem(Path(it["path"]).name, it["path"])
            seen.add(it["path"])
        for it in self.batch:
            if it["path"] not in seen:
                self.cmb_video.addItem(Path(it["path"]).name, it["path"])
                seen.add(it["path"])
        if self.path and self.path not in seen:
            self.cmb_video.addItem(Path(self.path).name, self.path)
        if self.path:
            i = self.cmb_video.findData(self.path)
            if i >= 0:
                self.cmb_video.setCurrentIndex(i)
        self.cmb_video.blockSignals(False)
        if self.path is None and self.cmb_video.count():
            self.open_video(self.cmb_video.itemData(0))

    def _open_dialog(self):
        start = str(Path(self.path).parent) if self.path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open video", start, VIDEO_FILTER)
        if path:
            self.open_video(path)

    def open_video(self, path):
        if not path or path == self.path:
            return
        if self.busy == "single":
            return
        self.pause()
        if not self.player.open(path):
            QMessageBox.critical(self, "Video", f"Could not open:\n{path}")
            return
        self._ch_stash()
        self.path = str(path)
        self.md = self.md_key = self.result = self.saved = None
        self.dirty = False
        chs = self.chambers.get(self.path)
        if chs:                                          # back to a video with chambers: the chamber shown
            k = self.ch_cur.get(self.path, 0)
            if chs[k].get("spec"):
                self.regions[self.path] = chs[k]["spec"]
            self._ch_unstash(k)
        self._ch_refresh()
        self._ring = []
        self.canvas.reset_zoom()
        self.slider.setRange(0, max(0, self.player.total - 1))
        self.refresh_sources()
        self._show_result()
        self.show_frame(int(round(self.sp_start.value() * self.player.fps)))
        self.lbl_stale.setText("")

    def _start_canvas(self, mode):
        if self.player.cap is None:
            return
        self.pause()
        self.canvas.set_mode(mode)
        self.canvas.setFocus()

    def _box_drawn(self, box):
        if not self.path:
            return
        r = FZ.clean_region(box, self.player.w, self.player.h)
        if r is None:
            self.regions.pop(self.path, None)
        else:
            self.regions[self.path] = r
        self._region_changed()

    def _start_poly(self, kind):
        self.canvas.poly_kind = kind
        self.canvas.poly_hint = ("Mask: click points · double-click = done · right-click = undo · Esc" if kind == "mask"
                                 else "Region: click points · double-click = done · right-click = undo · Esc")
        self._start_canvas("poly")

    def _region_spec(self, path=None):
        """{"poly": [...] or None, "masks": [...]} of a video (an old box becomes its 4 corners)."""
        poly, masks = FZ.region_parts(self.regions.get(path or self.path))
        return {"poly": [list(q) for q in poly] if poly else None, "masks": [[list(q) for q in m] for m in masks]}

    def _edit_polys(self):
        """The region / mask corners the canvas lets you drag."""
        if not self.path:
            return []
        spec = self._region_spec()
        out = []
        if spec["poly"]:
            out.append(("region", 0, [tuple(q) for q in spec["poly"]]))
        for k, m in enumerate(spec["masks"]):
            out.append(("mask", k, [tuple(q) for q in m]))
        return out

    def _poly_edited(self, kind, idx, pts):
        """A corner of the region / a mask was dragged."""
        if not self.path:
            return
        w, h = self.player.w, self.player.h
        pts = [[min(max(0.0, float(x)), w), min(max(0.0, float(y)), h)] for x, y in pts]
        spec = self._region_spec()
        if kind == "mask" and 0 <= idx < len(spec["masks"]):
            spec["masks"][idx] = pts
        elif kind == "region":
            spec["poly"] = pts
        else:
            return
        self._set_spec(spec)

    def _set_spec(self, spec):
        if spec["poly"] is None and not spec["masks"]:
            self.regions.pop(self.path, None)
        else:
            self.regions[self.path] = FZ.region_json(spec)
        chs = self.chambers.get(self.path)
        if chs:
            chs[self.ch_cur.get(self.path, 0)]["spec"] = self.regions.get(self.path)
            self._ch_refresh()
        self._region_changed()

    # ------------------------------------------------------------------ chambers (2-4 animals in one video)
    _STASH = ("md", "md_key", "md_pol", "md_con", "md_len", "result", "saved", "btf", "dirty")

    def _ch_refresh(self):
        chs = self.chambers.get(self.path) or []
        self.cmb_ch.blockSignals(True)
        self.cmb_ch.clear()
        for c in chs:
            self.cmb_ch.addItem(c["name"] + (f" · {c['subject']}" if c.get("subject") else "")
                                + ("" if c.get("spec") else "  (no region)"))
        self.cmb_ch.setCurrentIndex(self.ch_cur.get(self.path, 0) if chs else -1)
        self.cmb_ch.blockSignals(False)
        self.cmb_ch.setVisible(bool(chs))
        self.btn_box.setText("⬠ Chamber floor" if chs else "⬠ Region")

    def _ch_stash(self):
        chs = self.chambers.get(self.path)
        if not chs:
            return
        k = self.ch_cur.get(self.path, 0)
        self.ch_results.setdefault(self.path, {})[k] = {a: getattr(self, a) for a in self._STASH}

    def _ch_unstash(self, k):
        st = (self.ch_results.get(self.path) or {}).get(k)
        if st is None:
            self.md = self.md_key = self.result = self.saved = self.btf = None
            self.md_pol = self.md_con = None
            self.md_len = 0.0
            self.dirty = False
        else:
            for a, v in st.items():
                setattr(self, a, v)
        self.btn_save.setText("⤓ Save results *" if self.dirty else "⤓ Save results")

    def _ch_select(self, k):
        chs = self.chambers.get(self.path) or []
        if not 0 <= k < len(chs) or self.busy:
            self._ch_refresh()
            return
        self._ch_stash()
        self.ch_cur[self.path] = k
        spec = chs[k].get("spec")
        if spec:
            self.regions[self.path] = spec
        else:
            self.regions.pop(self.path, None)
        self._ch_unstash(k)
        self._ch_refresh()
        self._body_label()
        self._show_result()
        self.lbl_stale.setText("")
        self.show_frame(self.cur)
        if self.result is not None:
            self.ctrl.freezing_result_ready(self.entry(), show=False)

    def _ch_add(self):
        if not self.path or self.busy:
            return
        chs = self.chambers.setdefault(self.path, [])
        if len(chs) >= MAX_CHAMBERS:
            QMessageBox.information(self, "Chambers", f"Up to {MAX_CHAMBERS} chambers per video.")
            return
        if not chs:                                    # the region drawn so far becomes chamber 1
            subj, _g = self._subject_group(self.path)
            chs.append({"name": "Chamber 1", "subject": subj or "", "spec": self.regions.get(self.path)})
            self.ch_cur[self.path] = 0
            if self.regions.get(self.path) is not None:
                self._ch_stash()
        if chs[-1].get("spec") is not None:            # the last chamber has its floor: a new chamber
            self._ch_stash()
            chs.append({"name": f"Chamber {len(chs) + 1}", "subject": "", "spec": None})
            self.ch_cur[self.path] = len(chs) - 1
            self.regions.pop(self.path, None)
            self._ch_unstash(len(chs) - 1)
        self._ch_refresh()
        self._show_result()
        self.show_frame(self.cur)
        self._start_poly("region")
        self.canvas.poly_hint = (f"{chs[self.ch_cur[self.path]]['name']} floor: click points · double-click = done "
                                 "· right-click = undo · Esc")

    def _ch_rename(self):
        chs = self.chambers.get(self.path) or []
        if not chs:
            QMessageBox.information(self, "Chambers", "Add a chamber first (▦ Chambers → Add a chamber).")
            return
        from PySide6.QtWidgets import QInputDialog
        k = self.ch_cur.get(self.path, 0)
        txt, ok = QInputDialog.getText(self, "Animal ID", f"Animal ID in {chs[k]['name']}:",
                                       text=chs[k].get("subject") or "")
        if ok:
            chs[k]["subject"] = txt.strip()
            self._ch_refresh()
            self._batch_refresh()

    def _ch_remove(self):
        chs = self.chambers.get(self.path) or []
        if not chs or self.busy:
            return
        k = self.ch_cur.get(self.path, 0)
        chs.pop(k)
        res = self.ch_results.get(self.path) or {}
        self.ch_results[self.path] = {(i if i < k else i - 1): v for i, v in res.items() if i != k}
        for i, c in enumerate(chs):
            if c["name"].startswith("Chamber "):
                c["name"] = f"Chamber {i + 1}"
        if len(chs) <= 1:                              # one left: an ordinary region again
            spec = chs[0].get("spec") if chs else None
            st = (self.ch_results.get(self.path) or {}).get(0)
            self.chambers.pop(self.path, None)
            self.ch_results.pop(self.path, None)
            self.ch_cur.pop(self.path, None)
            if spec:
                self.regions[self.path] = spec
            else:
                self.regions.pop(self.path, None)
            if st:
                for a, v in st.items():
                    setattr(self, a, v)
            else:
                self._ch_unstash(-1)
            self._ch_refresh()
            self._show_result()
            self._region_changed()
            return
        kk = min(k, len(chs) - 1)
        spec = chs[kk].get("spec")
        self.ch_cur[self.path] = kk
        if spec:
            self.regions[self.path] = spec
        else:
            self.regions.pop(self.path, None)
        self._ch_unstash(kk)
        self._ch_refresh()
        self._show_result()
        self._region_changed()

    def _ch_clear(self):
        if not self.chambers.get(self.path) or self.busy:
            return
        if QMessageBox.question(self, "Chambers", "Remove every chamber of this video (their regions and "
                                                  "results on this page)?") != QMessageBox.Yes:
            return
        self.chambers.pop(self.path, None)
        self.ch_results.pop(self.path, None)
        self.ch_cur.pop(self.path, None)
        self.regions.pop(self.path, None)
        self._ch_unstash(-1)
        self._ch_refresh()
        self._show_result()
        self._region_changed()

    def _chamber_jobs(self, path, subject):
        """[(chamber name, subject, region, out folder)] of a video, or None (no chambers)."""
        chs = self.chambers.get(path) or []
        if len(chs) < 2:
            return None
        stem = Path(path).stem
        out = []
        for c in chs:
            subj = c.get("subject") or (f"{subject} · {c['name']}" if subject else f"{stem} · {c['name']}")
            out.append((c["name"], subj, c.get("spec"), chamber_folder(path, c["name"])))
        return out

    def _poly_drawn(self, pts):
        if not self.path:
            return
        w, h = self.player.w, self.player.h
        pts = [[min(max(0.0, float(x)), w), min(max(0.0, float(y)), h)] for x, y in pts]
        spec = self._region_spec()
        if self.canvas.poly_kind == "mask":
            spec["masks"].append(pts)
        else:
            spec["poly"] = pts
        self._set_spec(spec)

    def _clear_region(self, what):
        if not self.path:
            return
        spec = self._region_spec()
        if what in ("region", "all"):
            spec["poly"] = None
        if what in ("masks", "all"):
            spec["masks"] = []
        if what == "last mask" and spec["masks"]:
            spec["masks"].pop()
        self._set_spec(spec)

    def _clear_box(self):
        self._clear_region("all")

    def _region_changed(self):
        self._batch_refresh()
        self.show_frame(self.cur)
        self._settings_changed()

    def _frames_at(self, idx):
        """(frame idx, frame idx - lag) -- read forward when playing, otherwise seek."""
        lag = FZ.frame_lag(self.player.fps)
        if self._ring and self._ring[-1][0] == idx:
            pass
        elif self._ring and self._ring[-1][0] == idx - 1:
            f = self.player.next()
            if f is not None:
                self._ring.append((idx, f))
        else:
            self._ring = []
            a = max(0, idx - lag)
            f = self.player.seek(a)
            if f is not None:
                self._ring.append((a, f))
                while self._ring[-1][0] < idx:
                    g = self.player.next()
                    if g is None:
                        break
                    self._ring.append((self._ring[-1][0] + 1, g))
        self._ring = self._ring[-(lag + 1):]
        cur = self._ring[-1][1] if self._ring and self._ring[-1][0] == idx else None
        prev = next((f for i, f in self._ring if i == idx - lag), None)
        return cur, prev

    def show_frame(self, idx):
        if self.player.cap is None:
            return
        idx = int(min(max(0, idx), max(0, self.player.total - 1)))
        cur, prev = self._frames_at(idx)
        if cur is None:
            self.pause()
            return
        self.cur = idx
        md = self.md
        status = None
        if md is not None and md.frames[0] <= idx <= md.frames[-1]:
            i = idx - int(md.frames[0])
            disp = FZ.review_image(md, cur, prev, i, self.result, overlay=self.chk_overlay.isChecked(), text=False,
                                   dots=False)
            status = FZ.review_status(self.result, i)
            body = FZ.review_body(md, i, self.result)
        else:
            disp = cur.copy()
            body = None
            FZ.draw_region(disp, region=self.regions.get(self.path))
            if md is not None:
                cv2.putText(disp, "outside the analysed time", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1, cv2.LINE_AA)
        self.canvas.edit_polys = self._edit_polys()
        self.canvas.set_bgr(disp, status, body)
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
        fps = self.player.fps or 30.0
        self.lbl_time.setText(f"{fmt_clock(idx / fps)} / {fmt_clock(self.player.duration)}")
        self.timeline.set_position(idx / fps)

    def _seek_time(self, t):
        self.pause()
        self.show_frame(int(round(t * (self.player.fps or 30.0))))

    def step(self, d):
        self.pause()
        self.show_frame(self.cur + d)

    def toggle_play(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if self.player.cap is None or self.canvas.mode:
            return
        if self.cur >= self.player.total - 1:
            self.show_frame(0)
        self.playing = True
        self.btn_play.setText("⏸")
        self._speed_changed()

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.btn_play.setText("▶")

    def _speed_changed(self):
        self.speed = float(self.cmb_speed.currentData() or 1.0)
        if self.playing:
            self.timer.start(max(5, int(1000.0 / ((self.player.fps or 30.0) * self.speed))))

    def _advance(self):
        if not self.playing:
            return
        skip = max(1, int(round(self.speed / 2.0))) if self.speed > 2 else 1
        nxt = self.cur + skip
        if nxt >= self.player.total:
            self.pause()
            return
        if skip > 1:                             # fast playback: read the skipped frames, show every n-th
            for _ in range(skip - 1):
                self._frames_at(self.cur + 1)
                self.cur += 1
        self.show_frame(nxt)

    def _goto_epoch(self, r, _c):
        if self.result and r < len(self.result["epochs"]):
            self._seek_time(self.result["epochs"][r]["Start_s"])

    # ------------------------------------------------------------------ mode (side bar: Individual / Batch)
    def set_mode(self, mode):
        self.mode = "batch" if mode == "batch" else "single"
        self.tabs.setCurrentIndex(1 if self.mode == "batch" else 0)
        self.title_lbl.setText("Freezing Analysis — Batch" if self.mode == "batch" else
                               "Freezing Analysis — Individual")

    def _tab_changed(self, i):
        self.pause()
        self._update_run_button()
        if hasattr(self, "btn_save"):
            self.btn_save.setVisible(i == 0)

    def test_detection(self):
        """Track ~1 s up to the shown frame (the real run's background) and mark the animal on the canvas."""
        if not self.path or self.busy or self.player.cap is None:
            return
        self.pause()
        s, path, region, frame = self.settings(), self.path, self.regions.get(self.path), self.cur
        model = self._det_model
        cached = model is not None and model.get("key") == FZ.model_key(path, s, region)
        self.lbl_prog.setText("Testing detection…" if cached else
                              "Testing detection… (first test: learning the empty floor)")
        self.lbl_prog.show()

        def fn(_wk):
            return FZ.test_detection(path, s, region, frame, model=model if cached else None)

        def done(r):
            self.lbl_prog.hide()
            if r.get("model") is not None:
                self._det_model = r["model"]
            if path != self.path or r.get("bgr") is None:
                return
            if r["found"]:
                big, col = "ANIMAL FOUND", "#4ade80"
                small = f"seen in {r['found_pct']:.0f} % of the last second"
            else:
                big, col = "NOT FOUND", "#f87171"
                small = "try another animal colour / min contrast, or mask reflections"
            if r.get("polarity_warning"):
                small += f" · looks {'darker' if r['polarity_warning'] == 'dark' else 'lighter'} than the floor"
            self.canvas.set_bgr(r["bgr"], (big, small, col))

        def failed(msg):
            self.lbl_prog.hide()
            QMessageBox.warning(self, "Test detection", msg)
        self._start_worker(fn, "test", done, failed, lambda _f, _t: None)

    def suggest_settings(self):
        """✨ Suggest: animal colour + min contrast measured from 40 frames of the analysis window."""
        if not self.path or self.busy or self.player.cap is None:
            return
        self.pause()
        s, path, region = self.settings(), self.path, self.regions.get(self.path)
        model = self._det_model
        self.lbl_prog.setText("Checking the video (animal colour, contrast, size)…")
        self.lbl_prog.show()
        self.prog.setValue(0)
        self.prog.show()

        def fn(wk):
            return FZ.suggest_settings(path, s, region, model=model,
                                       progress=lambda f: wk.progress.emit(f, "Checking the video…"))

        def done(r):
            self.lbl_prog.hide()
            self.prog.hide()
            if r.get("model") is not None and r["model"].get("key") == FZ.model_key(path, s, region):
                self._det_model = r["model"]
            if path != self.path:
                return
            lines = []
            if r.get("polarity"):
                lines.append("• Animal colour: " + ("darker" if r["polarity"] == "dark" else "lighter")
                             + " than the floor")
            if r.get("min_contrast"):
                lines.append(f"• Min contrast: {r['min_contrast']}  (the animal differs from the floor by "
                             f"≈{r['peak_median']:.0f}, at its weakest ≈{r['peak_low']:.0f}; noise ≈{r['noise']:.1f})")
                lines.append(f"• Found in {r['found']} of {r['frames']} checked frames with it")
            if r.get("length_px"):
                lines.append(f"• Tracked size ≈ {r['length_px']:.0f} px long — 📏 measure to be sure")
            lines.append("• Colour: Gray · Cutoff: Auto (measured during the analysis)")
            lines += r.get("notes") or []
            if not r.get("min_contrast") and not r.get("polarity"):
                QMessageBox.information(self, "Suggest settings", "\n".join(lines))
                return
            box = QMessageBox(self)
            box.setWindowTitle("Suggest settings")
            box.setText("Measured from this video:\n\n" + "\n".join(lines) + "\n\nUse these settings?")
            box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            if box.exec() != QMessageBox.Yes:
                return
            if r.get("polarity"):
                i = self.cmb_pol.findData(r["polarity"])
                if i >= 0:
                    self.cmb_pol.setCurrentIndex(i)
            if r.get("min_contrast"):
                self.sp_con.setValue(float(r["min_contrast"]))
            i = self.cmb_colour.findData("gray")
            if i >= 0:
                self.cmb_colour.setCurrentIndex(i)
            self.chk_cut_auto.setChecked(True)
            self.ctrl.status.showMessage("Suggested settings applied — 🎯 Test detection to check.", 6000)

        def failed(msg):
            self.lbl_prog.hide()
            self.prog.hide()
            QMessageBox.warning(self, "Suggest settings", msg)

        def progress(f, txt):
            self.prog.setValue(int(1000 * f))
            self.lbl_prog.setText(f"{txt}  {100 * f:.0f} %")
        self._start_worker(fn, "test", done, failed, progress)

    def _update_run_button(self):
        if not hasattr(self, "btn_run"):
            return
        if self.busy:
            self.btn_run.setText("■  Stop")
            return
        self.btn_run.setText("❄  Analyse this video" if self.tabs.currentIndex() == 0 else "▶  Run batch")

    def _run_clicked(self):
        if self.busy:
            if self.worker is not None:
                self.worker.cancel.set()
            self.btn_run.setEnabled(False)
            return
        if self.tabs.currentIndex() == 0:
            self.analyse_current()
        else:
            self.run_batch()

    def _start_worker(self, fn, kind, on_done, on_failed, on_progress, on_item=None, on_live=None):
        self.busy = kind
        self.worker = _Worker(fn)
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        r_prog = _Relay(on_progress)
        self.worker.progress.connect(r_prog.prog)
        self._relays = [r_prog]
        if on_item is not None:
            r_item = _Relay(on_item)
            self.worker.item.connect(r_item.item)
            self._relays.append(r_item)
        if on_live is not None:
            r_live = _Relay(on_live)
            self.worker.live.connect(r_live.live)
            self._relays.append(r_live)

        def finish():
            self.thread.quit()
            self.thread.wait(5000)
            self.thread = self.worker = None
            self.busy = None
            self.btn_run.setEnabled(True)
            self._update_run_button()
            self._lock(False)

        def ok(v):
            try:
                on_done(v)
            finally:
                finish()

        def bad(m):
            try:
                on_failed(m)
            finally:
                finish()
        r_ok, r_bad = _Relay(ok), _Relay(bad)
        self.worker.done.connect(r_ok.one)
        self.worker.failed.connect(r_bad.text)
        self._relays += [r_ok, r_bad]
        self._lock(True)
        self._update_run_button()
        self.thread.start()

    def _lock(self, on):
        for wd in (self.cmb_video, self.btn_box, self.btn_mask, self.btn_nobox, self.btn_measure, self.sp_start,
                   self.sp_end, self.cmb_pol, self.sp_con, self.btn_test, self.btn_suggest):
            wd.setEnabled(not on)
        self.canvas.editable = not on
        self.canvas.update()

    def _embed_px(self):
        try:
            return int(self.ctrl.cfg.get("embed_video_px", 640) or 0)
        except Exception:
            return 640

    def analyse_current(self):
        if not self.path:
            QMessageBox.information(self, "Freezing", "Open a video first.")
            return
        if hasattr(self.ctrl, "ensure_results_folder") and not self.ctrl.ensure_results_folder():
            return
        self.pause()
        s = self.settings()
        path, region = self.path, self.regions.get(self.path)
        subject, group = self._subject_group(path)
        embed = self._embed_px()
        jobs = self._chamber_jobs(path, subject)
        if jobs is not None:
            empty = [n for n, _s, r, _o in jobs if r is None]
            if empty:
                QMessageBox.information(self, "Chambers", "Draw the floor of every chamber first: "
                                        + ", ".join(empty) + " has no region.")
                return
            self._ch_stash()
        else:
            jobs = [(None, subject, region, None)]
        self.prog.show()
        self.prog.setValue(0)
        self.lbl_prog.setText("Starting…")
        self.lbl_prog.show()
        # all cores (the video is read in pieces at once; the live picture follows the first piece)
        self._live_single = True

        def fn(wk):
            outs = []
            nj = len(jobs)
            for j, (cname, subj, reg, folder) in enumerate(jobs):
                pre = f"{cname}: " if cname else ""

                def prog(f, j=j, pre=pre):
                    wk.progress.emit((j + 0.92 * f) / nj, pre + ("Tracking the animal…" if FZ.needs_tracking(s)
                                                                 and f < 0.6 else "Measuring motion…"))

                def emit(bgr, text, pre=pre):
                    wk.live.emit(bgr, pre + text)
                feed = FZ.LiveFeed(emit, on=lambda: self._live_single, rate=12.0)
                md = FZ.run_freezing(path, s, region=reg, progress=prog, cancel=wk.cancel.is_set, live=feed)
                res = FZ.analyse(md, s)
                wk.progress.emit((j + 0.93) / nj, pre + "Saving results…")
                files, btf = _save_all(md, res, path, reg, s, subj, group, embed, None,
                                       lambda f, j=j, pre=pre: wk.progress.emit(
                                           (j + 0.93 + 0.07 * f) / nj, pre + "Saving results (video copy)…"),
                                       out_dir=folder)
                outs.append({"md": md, "result": res, "files": files, "btf": btf, "region": reg})
            return outs

        def done(outs):
            self.prog.hide()
            self.lbl_prog.hide()
            self._live_single = False
            if path != self.path:
                return
            if self.chambers.get(path) and len(outs) > 1:
                key_s = dict(s)
                for k, o in enumerate(outs):
                    self.ch_results.setdefault(path, {})[k] = {
                        "md": o["md"], "result": o["result"], "saved": o["files"], "btf": o["btf"],
                        "md_key": (path, json.dumps(FZ.region_json(o["region"]), sort_keys=True),
                                   round(float(key_s["start_time"] or 0), 2), key_s["end_time"]),
                        "md_pol": s["polarity"], "md_con": s.get("min_contrast"),
                        "md_len": float(s.get("body_length_px") or 0.0), "dirty": False}
                    if k != self.ch_cur.get(path, 0):
                        self.ctrl.freezing_result_ready(self._entry_of(k), show=False)
                out = outs[self.ch_cur.get(path, 0)]
            else:
                out = outs[0]
            self.md, self.result, self.saved, self.btf = out["md"], out["result"], out["files"], out["btf"]
            self.md_key = self._key(s, path)
            self.md_pol = s["polarity"]
            self.md_con = s.get("min_contrast")
            self.md_len = float(s.get("body_length_px") or 0.0)
            self.dirty = False
            self.btn_save.setText("⤓ Save results")
            self.lbl_stale.setText("")
            self._body_label()
            self._after_result(saved=True)
            md = self.md
            self.show_frame(max(self.cur, int(md.frames[0])) if self.cur <= md.frames[-1] else int(md.frames[0]))
            self.ctrl.status.showMessage(f"Freezing results saved: {self.saved['xlsx']}  ·  "
                                         f"{os.path.basename(self.btf['_path'])}", 8000)

        def failed(msg):
            self.prog.hide()
            self.lbl_prog.hide()
            self._live_single = False
            if msg == "__cancelled__":
                self.ctrl.status.showMessage("Freezing analysis stopped.", 5000)
            else:
                QMessageBox.warning(self, "Freezing analysis", f"The analysis failed:\n{msg}")

        def progress(f, txt):
            self.prog.setValue(int(1000 * f))
            self.lbl_prog.setText(f"{txt}  {100 * f:.0f} %")

        def live(bgr, text):
            """The analysis live on the video (like the tracking page): the animal while it is tracked, then the
            moving pixels, the body dots and FREEZING / ACTIVE."""
            status = None
            if isinstance(bgr, tuple):
                bgr, status = bgr
            if path == self.path and self.busy == "single":
                self.canvas.set_bgr(bgr, status)
                self.lbl_info.setText(text)
        self._start_worker(fn, "single", done, failed, progress, on_live=live)

    def _reanalyse(self, save=False):
        if self.md is None:
            return
        try:
            self.result = FZ.analyse(self.md, self.settings())
        except Exception as exc:
            self.lbl_warn.setText(f"Could not compute the result: {exc}")
            self.lbl_warn.show()
            return
        self.dirty = True
        self.btn_save.setText("⤓ Save results *")
        self._after_result(saved=False)
        if save:
            self.save_results(quiet=True)
        self.show_frame(self.cur)

    def _after_result(self, saved):
        if self.chk_cut_auto.isChecked():               # show the cutoff Auto found (greyed out)
            self.sp_cut.blockSignals(True)
            self.sp_cut.setValue(int(self.result["cutoff"]))
            self.sp_cut.blockSignals(False)
        self._show_result()
        self.ctrl.freezing_result_ready(self.entry(), show=False)

    def _entry_of(self, k):
        """The Analysis & Results entry of chamber k (stashed result) of the current video."""
        st = (self.ch_results.get(self.path) or {}).get(k)
        if not st or st.get("result") is None:
            return None
        cur = {a: getattr(self, a) for a in self._STASH}
        cur_k = self.ch_cur.get(self.path, 0)
        try:
            for a, v in st.items():
                setattr(self, a, v)
            self.ch_cur[self.path] = k
            return self.entry()
        finally:
            for a, v in cur.items():
                setattr(self, a, v)
            self.ch_cur[self.path] = cur_k

    def entry(self):
        """This page's result for the Analysis & Results page."""
        if self.md is None or self.result is None:
            return None
        subject, group = self._subject_group(self.path)
        btf_path = (self.btf or {}).get("_path")
        name = Path(self.path).stem
        chs = self.chambers.get(self.path) or []
        if len(chs) > 1:
            c = chs[self.ch_cur.get(self.path, 0)]
            name += f" · {c['name']}"
            subject = c.get("subject") or subject
        return {"key": os.path.normcase(os.path.abspath(btf_path or self.path)), "video": self.path,
                "name": name, "md": self.md, "result": self.result,
                "region": self.regions.get(self.path), "folder": (self.saved or {}).get("folder"),
                "files": None if self.dirty else self.saved, "btf": self.btf, "subject": subject, "group": group,
                "unsaved": self.dirty}

    def open_in_results(self):
        e = self.entry()
        if e is None:
            QMessageBox.information(self, "Freezing", "Analyse a video first.")
            return
        if self.dirty:
            self.save_results(quiet=True, then=lambda: self.ctrl.show_freezing_result(self.entry()))
            return
        self.ctrl.show_freezing_result(e)

    def _show_result(self):
        r = self.result
        self.timeline.set_result(r)
        if r is None:
            self.result_bar.hide()
            self.lbl_info.setText("⬠ Region · ▨ Mask (optional) → 🎯 Test detection → ❄ Analyse")
            self.lbl_warn.hide()
            return
        s = r["summary"]
        lat = _num(s["Latency_first_freeze_s"], 1, " s")
        checks = r.get("checks") or []
        errors = [c for c in checks if c["level"] == "error"]
        head = ("⚠ CHECK before using: " if errors else "✓ ")
        self.lbl_result.setText(f"{head}Freezing <b>{_num(s['Freezing_pct'], 1, ' %')}</b>  ·  "
                                f"{_num(s['Freezing_time_s'], 1, ' s')}  ·  {s['Freezing_bouts']} bouts  ·  "
                                f"latency {lat}" + ("  <span style='color:#b45309'>(not saved yet)</span>"
                                                    if self.dirty else "  (saved)"))
        self.lbl_result.setStyleSheet("color:%s; font-size:10.5pt;" % ("#b91c1c" if errors else "#15803d"))
        self.result_bar.show()
        self.lbl_info.setText(
            f"{fmt_clock(s['Start_time_s'])}–{fmt_clock(s['End_time_s'])} · {s['Motion_counted']} · "
            f"cutoff {s['Motion_cutoff']} · body {s['Body_length_px']:.0f} px ({s['Body_size_from']})")
        self.lbl_thr_px.setText(f"= {s.get('Freezing_threshold_px', 0):.0f} changed px "
                                f"(body {s['Body_area_px']:.0f} px²)")
        warn = [("⛔ " if c["level"] == "error" else "⚠ " if c["level"] == "warn" else "ℹ ") + c["text"]
                for c in checks]
        self.lbl_warn.setText("\n".join(warn))
        self.lbl_warn.setStyleSheet("color:%s; font-size:9pt; font-weight:600;"
                                    % ("#b91c1c" if errors else "#b45309"))
        self.lbl_warn.setVisible(bool(warn))

    def show_body_size(self):
        """👁 The body size the threshold uses, drawn on a frame where the animal was seen."""
        if self.md is None or self.result is None:
            return
        i = FZ.body_size_frame(self.md)
        fr = self._frames_at(int(self.md.frames[i]))
        img = fr[0] if isinstance(fr, tuple) else fr
        if img is None:
            QMessageBox.information(self, "Body size", "The video frame could not be read.")
            return
        bgr = FZ.body_size_image(self.md, self.result, img, i)
        dlg = QDialog(self)
        dlg.setWindowTitle("Body size used for the threshold")
        lay = QVBoxLayout(dlg)
        lab = QLabel()
        pm = QPixmap.fromImage(bgr_to_qimage(bgr))
        scr = QApplication.primaryScreen()
        av = scr.availableGeometry() if scr is not None else None
        mw, mh = (int(av.width() * 0.8), int(av.height() * 0.75)) if av is not None else (1200, 800)
        lab.setPixmap(pm.scaled(mw, mh, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        lay.addWidget(lab)
        tip = QLabel("Wrong size?  📏 Measure the mouse (nose → tail base) and analyse again.")
        tip.setStyleSheet("color:#334155;")
        lay.addWidget(tip)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        dlg.exec()

    def save_results(self, quiet=False, then=None):
        """Excel + CSV + charts + the .btf with the current settings (in the background)."""
        if self.md is None or self.result is None:
            if not quiet:
                QMessageBox.information(self, "Freezing", "Analyse a video first.")
            return
        if getattr(self, "_saving", False):
            return
        self._saving = True
        md, res, path, region = self.md, self.result, self.path, self.regions.get(self.path)
        s = dict(res["settings"])
        subject, group = self._subject_group(path)
        folder = None
        jobs = self._chamber_jobs(path, subject)
        if jobs is not None:
            _n, subject, _r, folder = jobs[self.ch_cur.get(path, 0)]
        prev, embed = self.btf, self._embed_px()
        self.btn_save.setEnabled(False)
        self.btn_save.setText("Saving…")

        def job():
            return _save_all(md, res, path, region, s, subject, group, embed, prev, None, out_dir=folder)

        def ok(out):
            self._saving = False
            self.btn_save.setEnabled(True)
            if res is not self.result:                   # tuned again while saving
                self.btn_save.setText("⤓ Save results *")
                return
            self.saved, self.btf = out
            self.dirty = False
            self.btn_save.setText("⤓ Save results")
            self._show_result()
            self.ctrl.freezing_result_ready(self.entry(), show=False)
            self.ctrl.status.showMessage(f"Freezing results saved: {self.saved['xlsx']}", 8000)
            if then is not None:
                then()

        def bad(msg):
            self._saving = False
            self.btn_save.setEnabled(True)
            self.btn_save.setText("⤓ Save results *")
            QMessageBox.warning(self, "Save results", f"Could not save the results:\n{msg}")
        self.ctrl._thread_run(job, ok, bad)

    def _validate(self):
        from pro_ui.validation_dialog import ValidationDialog
        ep = getattr(self.ctrl, "event_page", None)
        cur = None
        if ep is not None and ep.events:
            cur = {"video": ep.path, "events": list(ep.events), "inventory": list(ep.inventory), "fps": ep.fps}
        ValidationDialog(self, cur).exec()

    def _open_folder(self):
        folder = None
        if self.mode == "batch" and self.batch_out and os.path.isdir(self.batch_out):
            folder = self.batch_out
        elif self.saved:
            folder = self.saved["folder"]
        elif self.path:
            folder = os.path.join(os.path.dirname(self.path), "results", Path(self.path).stem, "freezing")
        if folder and os.path.isdir(folder):
            QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
        else:
            QMessageBox.information(self, "Results", "No results saved yet.")

    # ------------------------------------------------------------------ BT Freezing file (.btf)
    def _open_btf_dialog(self):
        start = str(Path(self.path).parent) if self.path else str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open BT Freezing file", start, "BT Freezing file (*.btf)")
        if path:
            self.load_btf(path)

    def load_btf(self, path):
        """Settings + result of a .btf back on this page (needs the video to tune; else -> Analysis & Results)."""
        from tracking import project_file as PF
        try:
            data = PF.load_freezing(path)
        except Exception as exc:
            QMessageBox.warning(self, "BT Freezing file", str(exc))
            return
        vp = (data.get("video") or {}).get("path")
        if not vp or not os.path.exists(vp):
            QMessageBox.information(self, "BT Freezing file",
                                    "The video of this file is not here, so it opens in Analysis & Results "
                                    "(graphs, tables and the frame review from the copy inside the file).")
            self.ctrl.open_btf(path, data=data)
            return
        self.open_video(vp) if vp != self.path else None
        if vp != self.path:
            return
        region = data.get("region") or None
        if region:
            self.regions[vp] = region
        else:
            self.regions.pop(vp, None)
        self.cfg.update({k: v for k, v in (data.get("settings") or {}).items()})
        self._load_cfg_into_ui()
        self._save_cfg()
        self.md = data["md"]
        self.md.path = vp
        self.md_key = self._key(self.settings(), vp)
        self.md_pol = (data.get("settings") or {}).get("polarity", "auto")
        self.md_con = (data.get("settings") or {}).get("min_contrast", 12.0)
        self.md_len = float((data.get("settings") or {}).get("body_length_px") or 0.0)
        self.btf = data
        self.saved = {"folder": os.path.dirname(path)}
        self.result = FZ.analyse(self.md, self.settings())
        self.dirty = False
        self.btn_save.setText("⤓ Save results")
        self._after_result(saved=True)
        self.show_frame(int(self.md.frames[0]))
        self.ctrl.status.showMessage(f"Opened {os.path.basename(path)}", 6000)

    def _subject_group(self, path):
        for it in self.batch:
            if it["path"] == path:
                return it.get("subject", ""), it.get("group", "")
        for it in getattr(self.ctrl, "batch_items", []):
            if it.get("path") == path and it.get("subject"):
                return it["subject"], self._design_group(it["subject"])
        return "", ""

    def _design_group(self, subject):
        for d in getattr(self.ctrl, "experimental_design", []) or []:
            if subject and d.get("Animal ID") == subject:
                for k in ("Group", "group", "Treatment"):
                    if d.get(k):
                        return str(d[k])
        return ""

    # ------------------------------------------------------------------ batch
    def _batch_add(self, paths):
        known = {it["path"] for it in self.batch}
        for p in paths:
            p = str(p)
            if Path(p).suffix.lower() not in VIDEO_EXTS or p in known:
                continue
            subj = ""
            for it in getattr(self.ctrl, "batch_items", []):
                if it.get("path") == p:
                    subj = it.get("subject", "") or ""
            self.batch.append({"path": p, "subject": subj, "group": self._design_group(subj), "status": "Pending",
                               "summary": None, "folder": None})
            known.add(p)
        self._batch_refresh()
        self.refresh_sources()

    def _batch_add_files(self):
        start = str(Path(self.batch[-1]["path"]).parent) if self.batch else str(Path.home())
        paths, _ = QFileDialog.getOpenFileNames(self, "Add videos", start, VIDEO_FILTER)
        self._batch_add(paths)

    def _batch_add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with videos", str(Path.home()))
        if d:
            self._batch_add(sorted(str(p) for p in Path(d).iterdir() if p.suffix.lower() in VIDEO_EXTS))

    def _batch_add_experiment(self):
        self._batch_add([it["path"] for it in getattr(self.ctrl, "exp_videos", [])])

    def _batch_selected(self):
        return sorted({i.row() for i in self.tbl_batch.selectedIndexes()})

    def _batch_remove(self):
        if self.busy == "batch":
            return
        for r in reversed(self._batch_selected()):
            if r < len(self.batch):
                self.batch.pop(r)
        self._batch_refresh()

    def _batch_clear(self):
        if self.busy == "batch":
            return
        self.batch = []
        self._batch_refresh()

    def _batch_chambers(self, i):
        """[(chamber, subject, region, folder)] for batch video i (its own chambers, or the first video's when it
        has no region of its own), or None."""
        it = self.batch[i]
        p, subj = it["path"], it.get("subject", "")
        own = self._chamber_jobs(p, subj)
        if own is not None:
            return own
        if p in self.regions or i == 0 or not self.chk_first_box.isChecked():
            return None
        chs = self.chambers.get(self.batch[0]["path"]) or []
        if len(chs) < 2:
            return None
        stem = Path(p).stem
        return [(c["name"], f"{subj or stem} · {c['name']}", c.get("spec"), chamber_folder(p, c["name"]))
                for c in chs]

    def _batch_region(self, i):
        p = self.batch[i]["path"]
        ch = self._batch_chambers(i)
        if ch is not None:
            own = len(self.chambers.get(p) or []) > 1
            return ch[0][2], f"{'Own' if own else 'First video'}'s {len(ch)} chambers".replace("Own's", "Own")
        if p in self.regions:
            return self.regions[p], "Own"
        if self.chk_first_box.isChecked() and self.batch and self.batch[0]["path"] in self.regions and i > 0:
            return self.regions[self.batch[0]["path"]], "First video's"
        return None, "Whole frame"

    def _batch_refresh(self):
        t = self.tbl_batch
        t.blockSignals(True)
        t.setRowCount(len(self.batch))
        colors = {"Completed": "#16a34a", "Failed": "#dc2626", "Processing": "#1f6feb", "Pending": "#64748b",
                  "Stopped": "#b45309"}
        for r, it in enumerate(self.batch):
            s = it.get("summary") or {}
            sums = it.get("summaries") or ([s] if s else [])
            chks = [str(x.get("Result_check") or "") for x in sums]
            chk = "" if not chks else ("OK" if all(c == "OK" for c in chks) else
                                       "; ".join(c for c in chks if c != "OK"))
            vals = (Path(it["path"]).name, it.get("subject", ""), it.get("group", ""), self._batch_region(r)[1],
                    it["status"], " / ".join(_num(x.get("Freezing_pct"), 1) for x in sums),
                    " / ".join(str(x.get("Freezing_bouts", "")) for x in sums),
                    ("OK" if chk == "OK" else "⚠ check") if chk else "")
            for c, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if c not in (1, 2):
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if c == 0:
                    item.setToolTip(it["path"])
                if c == 4:
                    item.setForeground(QColor(colors.get(it["status"].split(" (")[0], "#334155")))
                    if it.get("error"):
                        item.setToolTip(it["error"])
                if c == 7 and chk:
                    item.setForeground(QColor("#16a34a" if chk == "OK" else "#b91c1c"))
                    item.setToolTip(chk.replace("CHECK: ", "").replace("; ", "\n"))
                t.setItem(r, c, item)
        t.blockSignals(False)

    def _batch_item_changed(self, item):
        r, c = item.row(), item.column()
        if r < len(self.batch) and c in (1, 2):
            self.batch[r]["subject" if c == 1 else "group"] = item.text().strip()

    def _batch_open_row(self, r, c):
        """Double-click: a finished video -> its result in Analysis & Results; the video being analysed -> the
        live view; a waiting one -> Freezing – Individual (its own region box, a test run)."""
        if c in (1, 2) or r >= len(self.batch):
            return
        it = self.batch[r]
        if it["status"] == "Completed" and it.get("btf") and os.path.exists(it["btf"]):
            self.ctrl.open_btf(it["btf"])
            return
        if it["status"].startswith("Processing"):
            self.show_live()
            return
        if self.busy:
            self.ctrl.status.showMessage("This video waits for its turn — double-click it when it is done to see "
                                         "its result, or watch the video being analysed (👁 Live view).", 6000)
            return
        self.ctrl.go_freezing("single")
        self.open_video(it["path"])

    def _batch_choose_out(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for the batch summary workbook",
                                             self.batch_out or str(Path.home()))
        if d:
            self.batch_out = d
            self.lbl_out.setText(d)

    # ------------------------------------------------------------------ live view (batch)
    def show_live(self):
        if self.live_dlg is None:
            self.live_dlg = FreezeLiveDialog(self)
        self.live_dlg.show()
        self.live_dlg.raise_()
        self.live_dlg.activateWindow()
        if self.busy != "batch":
            self.live_dlg.set_idle()

    def _live_on(self):
        return bool(getattr(self, "_live_visible", False))

    def _live_frame(self, bgr, text):
        if self._live_on():
            self.live_dlg.set_frame(bgr, text)

    def run_batch(self):
        if not self.batch:
            QMessageBox.information(self, "Batch", "Add videos first.")
            return
        if hasattr(self.ctrl, "ensure_results_folder") and not self.ctrl.ensure_results_folder():
            return
        s = self.settings()
        embed = self._embed_px()
        jobs = []
        first_with_region = next((it["path"] for it in self.batch if self.regions.get(it["path"]) is not None), None)
        for i, it in enumerate(self.batch):
            reg, src = self._batch_region(i)
            it["status"] = "Pending"
            it["summary"] = None
            it["summaries"] = []
            it.pop("error", None)
            it.pop("btf", None)
            own = self.regions.get(it["path"]) is not None or len(self.chambers.get(it["path"]) or []) > 1
            borrowed = (first_with_region or (self.batch[0]["path"] if self.chambers.get(self.batch[0]["path"])
                                              else None)) if (reg is not None and not own) else None
            chs = self._batch_chambers(i)
            if chs:
                for j, (cname, csubj, creg, folder) in enumerate(chs):
                    jobs.append((i, it["path"], creg, csubj, it.get("group", ""), borrowed, cname, folder,
                                 j == len(chs) - 1))
            else:
                jobs.append((i, it["path"], reg, it.get("subject", ""), it.get("group", ""), borrowed, None, None,
                             True))
        self._batch_refresh()
        from tracking.location import results_root
        out_dir = self.batch_out or results_root(self.batch[0]["path"])
        self.prog_batch.setValue(0)
        t0 = time.time()
        live_on = self._live_on            # read from the worker thread: only a bool from the GUI side

        def fn(wk):
            from output.freezing_report import batch_row, write_freezing_batch
            rows = []
            n = len(jobs)
            sums = {}
            for k, (i, path, reg, subj, grp, borrowed, cname, folder, last) in enumerate(jobs):
                if wk.cancel.is_set():
                    break
                wk.item.emit(i, {"status": f"Processing ({cname})" if cname else "Processing"})
                name = Path(path).name + (f" · {cname}" if cname else "")

                def prog(f, k=k, name=name):
                    wk.progress.emit((k + 0.9 * f) / n, name)

                def emit(bgr, text, name=name):
                    wk.live.emit(bgr, f"{name}  ·  {text}")
                feed = FZ.LiveFeed(emit, on=live_on)
                try:
                    md = FZ.run_freezing(path, s, region=reg, progress=prog, cancel=wk.cancel.is_set, live=feed)
                    res = FZ.analyse(md, s)
                    if borrowed and borrowed != path:
                        try:
                            sh = FZ.camera_shift(borrowed, path)
                        except Exception:
                            sh = None
                        if sh is not None and sh[0] > 0.02 * max(md.frame_w, md.frame_h):
                            FZ.add_check(res, "warn", "moved",
                                         f"The picture is shifted by ≈{sh[0]:.0f} px against "
                                         f"{Path(borrowed).name}, whose region is used — the camera or box moved: "
                                         f"give this video its own region")
                    wk.progress.emit((k + 0.92) / n, name + " (saving)")
                    files, btf = _save_all(md, res, path, reg, s, subj, grp, embed, None, None, out_dir=folder)
                    rows.append(batch_row(path, res, subject=subj, group=grp, folder=files["folder"],
                                          extra={"Chamber": cname} if cname else None))
                    sums.setdefault(i, []).append(res["summary"])
                    wk.item.emit(i, {"status": "Completed" if last else f"Processing ({cname} done)",
                                     "summary": res["summary"], "summaries": list(sums[i]),
                                     "folder": files["folder"], "btf": files["btf"]})
                except FZ.Cancelled:
                    wk.item.emit(i, {"status": "Stopped"})
                    break
                except Exception as exc:
                    wk.item.emit(i, {"status": "Failed", "error": str(exc)})
            summary_path = None
            if rows:
                os.makedirs(out_dir, exist_ok=True)
                summary_path = os.path.join(out_dir, f"Freezing_BatchSummary_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
                write_freezing_batch(rows, summary_path)
            return {"summary_path": summary_path, "n_ok": len(rows), "cancelled": wk.cancel.is_set()}

        def on_item(i, upd):
            if i < len(self.batch):
                self.batch[i].update(upd)
                self._batch_refresh()
                if self._live_on() and upd.get("status") == "Completed" and upd.get("summary"):
                    self.live_dlg.set_text(f"{Path(self.batch[i]['path']).name} finished — freezing "
                                           f"{_num(upd['summary']['Freezing_pct'], 1, ' %')}. Double-click it in the "
                                           "list to see its result.")

        def progress(f, name):
            self.prog_batch.setValue(int(1000 * f))
            el = time.time() - t0
            eta = el / f - el if f > 0.02 else None
            self.lbl_batch.setText(f"{100 * f:.0f} %  ·  Analysing {name} …" +
                                   (f"  about {fmt_clock(eta)} left" if eta else ""))

        def done(r):
            self.prog_batch.setValue(1000 if not r["cancelled"] else self.prog_batch.value())
            if r["summary_path"]:
                self.batch_out = os.path.dirname(r["summary_path"])
                self.lbl_out.setText(self.batch_out)
                self.lbl_batch.setText(("Batch stopped. " if r["cancelled"] else "Batch finished. ") +
                                       f"{r['n_ok']} video(s) analysed — summary: {r['summary_path']}.  "
                                       "Double-click a video to see its result.")
            else:
                self.lbl_batch.setText("Batch stopped." if r["cancelled"] else "No video could be analysed.")
            self._batch_refresh()
            if self._live_on():
                self.live_dlg.set_idle()

        def failed(msg):
            self.lbl_batch.setText("Batch stopped." if msg == "__cancelled__" else f"Batch failed: {msg}")
            self._batch_refresh()
        self._start_worker(fn, "batch", done, failed, progress, on_item, on_live=self._live_frame)

    # ------------------------------------------------------------------ keys / lifecycle
    def eventFilter(self, obj, ev):
        if obj is getattr(self, "tbl_batch", None) and ev.type() == QEvent.KeyPress and \
                ev.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self.tbl_batch.state() != \
                QAbstractItemView.EditingState:
            self._batch_remove()
            return True
        if ev.type() not in (QEvent.KeyPress, QEvent.ShortcutOverride):
            return False
        if not self.isVisible() or QApplication.activeModalWidget() is not None or self.mode != "single":
            return False
        fw = QApplication.focusWidget()
        if isinstance(fw, (QLineEdit, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget)) or \
                (isinstance(fw, QComboBox) and fw.isEditable()):
            return False
        key = ev.key()
        if key not in (Qt.Key_Space, Qt.Key_Left, Qt.Key_Right):
            return False
        if ev.type() == QEvent.KeyPress:
            if key == Qt.Key_Space:
                self.toggle_play()
            else:
                self.step(-1 if key == Qt.Key_Left else 1)
        ev.accept()
        return True

    def shutdown(self):
        self.pause()
        if self.live_dlg is not None:
            self.live_dlg.close()
        if self.worker is not None:
            self.worker.cancel.set()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(5000)
        self.player.close()
