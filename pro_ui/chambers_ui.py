"""Chambers: several separate arenas (one animal each) in one video -- the screens.

Video Setup  : 'Chambers' tool tile (on/off).  When on: chamber tabs above the video, a 'Chambers in this video'
               card (how many, ⚡ Find, ▭ Draw, Animal ID per chamber, copy zones to all chambers) and dashed
               colour frames on the video.  Zones named 'C2 · …' belong to chamber 2; a zone drawn while a
               chamber tab is selected gets its name prefix automatically.
Tracking     : 'Settings for: All | C1 | C2 …' -- the start / end time, detection mode, animal colour, low-light
               boost and minimum contrast can differ per chamber.  Start Tracking tracks every chamber at once.
Analysis     : 'Showing' lists every chamber's result.
Batch        : a 'Chambers' choice per video (below Scale).
When the tile is off, nothing here is shown and tracking is exactly as before.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QIcon, QPen
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QDialog, QDialogButtonBox,
                               QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
                               QSpinBox, QToolButton, QVBoxLayout, QWidget)

from tracking.chambers import (CHAMBER_HEX, MAX_CHAMBERS, chamber_label, chamber_of_zone, chamber_setups,
                               short_label, split_zone_name, zone_prefix)

HERE = Path(__file__).resolve().parent
OVERRIDE_TEXT = {"start_time": "start", "end_time": "end", "color_mode": "detection mode",
                 "animal_polarity": "animal colour", "low_light": "low-light boost", "min_contrast": "contrast",
                 "loc_thresh": "threshold"}


def _btn(text, tip="", css="secondary"):
    b = QPushButton(text)
    b.setProperty("cssClass", css)
    b.setToolTip(tip)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet("padding:4px 8px;")
    return b


def _seg(text):
    b = QPushButton(text)
    b.setProperty("cssClass", "segment")
    b.setCheckable(True)
    b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet("padding:4px 10px;")
    return b


def _rect_of(pts):
    a = np.asarray(pts, float)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


class ChamberUI(QObject):
    def __init__(self, ctrl):
        super().__init__(ctrl.win)
        self.c = ctrl
        self.items = []            # [{"id": str, "overrides": {}}] -- rectangles live in canvas.chamber_rects
        self.active = None         # Video Setup tab: None = all chambers, else chamber index
        self.view = None           # Tracking 'Settings for': None = all, else chamber index
        self._global = None        # the All values of the per-chamber settings while a chamber is shown
        self._loading = False
        self._draw_idx = None
        self.results = []          # chamber results of the last run: [{"label", "bta", "summary", "video"}]
        self._build_setup_page()
        self._build_tracking_row()
        ctrl.canvas.extra_paint = self.paint
        ctrl.canvas.extra_inside = self.undimmed_rects
        self.refresh()

    # ================================================================== state
    @property
    def rects(self):
        return self.c.canvas.chamber_rects

    @property
    def enabled(self):
        """Chamber mode on and at least one chamber marked."""
        return self.tile.isChecked() and bool(self.rects)

    def _sync_items(self):
        while len(self.items) < len(self.rects):
            self.items.append({"id": f"M{len(self.items) + 1}", "overrides": {}})
        del self.items[len(self.rects):]
        if self.active is not None and self.active >= len(self.rects):
            self.active = None
        if self.view is not None and self.view >= len(self.rects):
            self.to_all()

    def label(self, i, short=False):
        cid = self.items[i].get("id", "") if i < len(self.items) else ""
        return short_label(i + 1, cid) if short else chamber_label(i + 1, cid)

    # ================================================================== Video Setup
    def _build_setup_page(self):
        c = self.c
        cal = c.tool_buttons["toolCalibrate"]
        tile = QToolButton(cal.parentWidget())
        tile.setObjectName("toolChambers")
        tile.setText("Chambers")
        tile.setIcon(QIcon(str(HERE / "icons" / "tile_chambers.svg")))
        tile.setIconSize(cal.iconSize())
        tile.setToolButtonStyle(cal.toolButtonStyle())
        tile.setProperty("cssClass", cal.property("cssClass"))
        tile.setMinimumSize(cal.minimumSize())
        tile.setMaximumSize(cal.maximumSize())
        tile.setSizePolicy(cal.sizePolicy())
        tile.setStyleSheet(cal.styleSheet())
        tile.setCheckable(True)
        tile.setAutoExclusive(False)
        tile.setChecked(False)
        tile.setCursor(Qt.PointingHandCursor)
        tile.setToolTip("Several chambers in one video (one animal each, up to 4): every chamber is tracked "
                        "separately — its own zones, settings and results — all at the same time.\n"
                        "Click to switch chamber mode on / off.")
        tile.toggled.connect(self._tile_toggled)
        lay = c._layout_with(c.win.findChild(QWidget, "pageVideoSetup").layout(), cal)
        lay.insertWidget(lay.indexOf(cal) + 1, tile)
        self.tile = tile

        # chamber tabs above the video
        self.tabs = QWidget()
        th = QHBoxLayout(self.tabs)
        th.setContentsMargins(0, 0, 0, 0)
        th.setSpacing(6)
        self.tabs_lay = th
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        vlay = c._layout_with(c.win.findChild(QWidget, "pageVideoSetup").layout(), c.canvas)
        vlay.insertWidget(vlay.indexOf(c.canvas), self.tabs)
        self.tabs.hide()

        # 'Chambers in this video' card on top of the right panel
        card = QFrame()
        card.setObjectName("chamberCard")
        card.setStyleSheet("QFrame#chamberCard { background: #f7f9fc; border: 1px solid #d5dce8; border-radius: 8px; }"
                           " QFrame#chamberCard QLabel { background: transparent; border: none; }")
        v = QVBoxLayout(card)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(6)
        title = QLabel("Chambers in this video")
        title.setStyleSheet("font-weight: 700; color: #1f2a44;")
        v.addWidget(title)
        top = QHBoxLayout()
        top.setSpacing(6)
        top.addWidget(QLabel("Chambers"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, MAX_CHAMBERS)
        self.spin_n.setValue(2)
        self.spin_n.setToolTip("How many chambers (animals) this video has — up to 4.")
        self.spin_n.valueChanged.connect(self._count_changed)
        top.addWidget(self.spin_n)
        self.btn_find = _btn("⚡ Find", "Find the chambers automatically (from where the animals move and the "
                                       "floor around them). Check the frames and redraw any that are not right.")
        self.btn_find.clicked.connect(lambda: self.find())
        self.btn_draw = _btn("▭ Draw", "Draw a chamber: drag a rectangle around it on the video.")
        self.btn_draw.clicked.connect(self._draw_clicked)
        top.addWidget(self.btn_find)
        top.addWidget(self.btn_draw)
        top.addStretch(1)
        v.addLayout(top)
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(4)
        v.addLayout(self.rows_box)
        self.btn_copy = _btn("⧉ Copy zones to all chambers", "Copy the zones of the selected chamber tab (or chamber "
                                                             "1) into every other chamber, fitted to its frame.")
        self.btn_copy.clicked.connect(self.copy_zones)
        v.addWidget(self.btn_copy)
        hint = QLabel("Zones 'C1 · …' belong to chamber 1. Pick a chamber tab above the video, then add its "
                      "zones (draw, ⚡ Auto or ▦ Preset).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #64748b; font-size: 9pt;")
        v.addWidget(hint)
        arena = c.win.findChild(QFrame, "cardArenaSettings")
        alay = arena.layout()
        alay.insertWidget(1, card)
        card.hide()
        self.card = card

    def _tile_toggled(self, on):
        c = self.c
        if on and c.player.cap is None:
            self.tile.blockSignals(True)
            self.tile.setChecked(False)
            self.tile.blockSignals(False)
            c.status.showMessage("Load a video first (Experiment → Video File(s)).", 5000)
            return
        if on and not self.rects:
            self.find(self.spin_n.value(), quiet=True)
        if not on:
            self.to_all()
            self.active = None
        self.refresh()
        c._preview = None
        if on:
            c.status.showMessage("Chamber mode: every chamber is tracked separately (own zones, settings, results), "
                                 "all at the same time. Check the coloured frames; redraw with ▭ if needed.", 10000)
        else:
            c.status.showMessage("Chamber mode off — one animal in the whole video (as before).", 6000)

    def refresh(self):
        """Rebuild the tabs, card rows, tracking row and repaint."""
        self._sync_items()
        on = self.tile.isChecked()
        self.card.setVisible(on)
        self.tabs.setVisible(on and bool(self.rects))
        self.spin_n.blockSignals(True)
        self.spin_n.setValue(max(1, len(self.rects) or self.spin_n.value()))
        self.spin_n.blockSignals(False)
        # tabs
        while self.tabs_lay.count():
            it = self.tabs_lay.takeAt(0)
            if it.widget() is not None:
                self.tab_group.removeButton(it.widget()) if isinstance(it.widget(), QPushButton) else None
                it.widget().deleteLater()
        b = _seg("All chambers")
        b.setProperty("chamber", -1)
        b.setChecked(self.active is None)
        b.clicked.connect(lambda: self.set_active(None))
        self.tab_group.addButton(b)
        self.tabs_lay.addWidget(b)
        for i in range(len(self.rects)):
            b = _seg(self.label(i))
            b.setProperty("chamber", i)
            b.setChecked(self.active == i)
            col = CHAMBER_HEX[i % len(CHAMBER_HEX)]
            b.setStyleSheet(f"padding:4px 10px; border-left: 4px solid {col};")
            b.clicked.connect(lambda _=False, k=i: self.set_active(k))
            self.tab_group.addButton(b)
            self.tabs_lay.addWidget(b)
        if len(self.rects) < MAX_CHAMBERS:
            add = _btn("＋ Chamber", "Add a chamber: drag a rectangle around it on the video.")
            add.clicked.connect(lambda: self.begin_draw(len(self.rects)))
            self.tabs_lay.addWidget(add)
        self.tabs_lay.addStretch(1)
        # card rows
        while self.rows_box.count():
            it = self.rows_box.takeAt(0)
            w = it.widget()
            if w is not None:
                w.hide()
                w.deleteLater()
        for i in range(len(self.rects)):
            roww = QWidget()
            h = QHBoxLayout(roww)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            sw = QLabel()
            sw.setFixedSize(14, 14)
            sw.setStyleSheet(f"background: {CHAMBER_HEX[i % len(CHAMBER_HEX)]}; border-radius: 3px;")
            h.addWidget(sw)
            h.addWidget(QLabel(f"Chamber {i + 1}  ID:"))
            le = QLineEdit(self.items[i].get("id", ""))
            le.setPlaceholderText("Animal ID")
            le.setMinimumWidth(60)
            le.setMinimumHeight(26)
            le.setToolTip("The animal in this chamber: names its results folder / file and its row in the "
                          "summary (the Subject).")
            le.editingFinished.connect(lambda k=i, e=le: self._id_edited(k, e.text()))
            h.addWidget(le, 1)
            rd = QToolButton()
            rd.setFixedSize(28, 26)
            rd.setText("▭")
            rd.setToolTip(f"Redraw chamber {i + 1}")
            rd.setCursor(Qt.PointingHandCursor)
            rd.clicked.connect(lambda _=False, k=i: self.begin_draw(k))
            h.addWidget(rd)
            rm = QToolButton()
            rm.setFixedSize(28, 26)
            rm.setText("✕")
            rm.setToolTip(f"Remove chamber {i + 1}")
            rm.setCursor(Qt.PointingHandCursor)
            rm.clicked.connect(lambda _=False, k=i: self.remove(k))
            h.addWidget(rm)
            self.rows_box.addWidget(roww)
        self.btn_copy.setEnabled(len(self.rects) >= 2)
        _relayout(self.card)
        self._refresh_tracking_row()
        self.c.canvas.update()
        if hasattr(self.c, "_update_opbar"):
            self.c._update_opbar()

    def set_active(self, k):
        self.active = k
        for b in self.tab_group.buttons():
            b.setChecked(b.property("chamber") == (-1 if k is None else k))
        self.c.canvas.update()
        if k is None:
            self.c.status.showMessage("All chambers: new zones go to the chamber they are drawn in; ▦ Preset / "
                                      "⚡ Auto fit zones in every chamber.", 6000)
        else:
            self.c.status.showMessage(f"{self.label(k)}: new zones, ▦ Preset and ⚡ Auto now work in this chamber "
                                      f"only (named 'C{k + 1} · …').", 6000)

    def _id_edited(self, k, text):
        if k < len(self.items):
            text = text.strip()
            if text != self.items[k].get("id"):
                self.items[k]["id"] = text
                QTimer.singleShot(0, self.refresh)

    def _count_changed(self, n):
        if not self.tile.isChecked() or n == len(self.rects):
            return
        if n < len(self.rects):
            for k in range(len(self.rects) - 1, n - 1, -1):
                self.remove(k, ask=False)
            return
        self.find(n)

    def _draw_clicked(self):
        k = self.active if self.active is not None else (len(self.rects) if len(self.rects) < MAX_CHAMBERS else 0)
        self.begin_draw(k)

    def begin_draw(self, k):
        c = self.c
        if not c._need_video():
            return
        if c.canvas.tool is not None:
            c.canvas.cancel_tool(silent=True)
        self._draw_idx = int(k)
        c.canvas.start_tool("rect", "chamber",
                            banner=f"Drag a rectangle around chamber {k + 1} (include its walls). Esc cancels.")
        c._update_opbar()

    def rect_drawn(self, pts):
        """Canvas ▸ a chamber rectangle was drawn."""
        k = self._draw_idx if self._draw_idx is not None else len(self.rects)
        self._draw_idx = None
        iw, ih = self.c.canvas.frame_size
        x0, y0, x1, y1 = _rect_of(pts)
        x0, y0 = max(0.0, x0), max(0.0, y0)
        x1, y1 = min(float(iw), x1), min(float(ih), y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            self.c.status.showMessage("That chamber is too small — drag a larger rectangle.", 5000)
            return
        r = [x0, y0, x1, y1]
        if k < len(self.rects):
            self.rects[k] = r
        elif len(self.rects) < MAX_CHAMBERS:
            self.rects.append(r)
            k = len(self.rects) - 1
        if not self.tile.isChecked():
            self.tile.blockSignals(True)
            self.tile.setChecked(True)
            self.tile.blockSignals(False)
        self.c._preview = None
        self.refresh()
        self.c.status.showMessage(f"Chamber {k + 1} set. Now its zones: pick its tab and draw / ⚡ Auto / ▦ Preset.",
                                  8000)

    def remove(self, k, ask=True):
        c = self.c
        if not (0 <= k < len(self.rects)):
            return
        rects = [list(r) for r in self.rects]
        mine = [n for n, p in c.canvas.zones.items() if chamber_of_zone(n, p, rects) == k]
        if ask and QMessageBox.question(
                c.win, "Remove chamber",
                f"Remove {self.label(k)}?" + (f"\n\nIts {len(mine)} zone(s) are removed too." if mine else "")
        ) != QMessageBox.Yes:
            return
        # zones: this chamber's go, the next chambers' prefixes move down by one
        new = {}
        for n, p in c.canvas.zones.items():
            i = chamber_of_zone(n, p, rects)
            if i == k:
                continue
            num, bare = split_zone_name(n)
            if num is not None and num - 1 > k:
                n = zone_prefix(num - 1) + bare
            new[n] = p
        del self.rects[k]
        del self.items[k]
        if new != c.canvas.zones:
            c.canvas.set_zones(new, record=True)
        if self.active is not None:
            self.active = None if self.active == k else (self.active - 1 if self.active > k else self.active)
        if not self.rects:
            self.tile.setChecked(False)
        c._preview = None
        self.refresh()

    def find(self, n=None, quiet=False):
        """⚡ Find: chambers from the video (worker thread)."""
        c = self.c
        if not c._need_video():
            return
        n = int(n or self.spin_n.value())
        path = c.player.path
        crop = c.crop
        matrix = crop["matrix"] if crop else None
        size = (crop["w"], crop["h"]) if crop else (c.player.w, c.player.h)
        self.btn_find.setEnabled(False)
        self.btn_find.setText("Finding…")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        def work():
            from tracking.chambers import find_chambers
            return find_chambers(path, n, matrix=matrix, size=size)

        def done(res):
            QApplication.restoreOverrideCursor()
            self.btn_find.setEnabled(True)
            self.btn_find.setText("⚡ Find")
            rects, by_motion = res
            self.rects[:] = [list(map(float, r)) for r in rects]
            if not self.tile.isChecked():
                self.tile.blockSignals(True)
                self.tile.setChecked(True)
                self.tile.blockSignals(False)
            c._preview = None
            self.refresh()
            c.status.showMessage(
                (f"⚡ {len(rects)} chamber(s) found from where the animals move — check the coloured frames "
                 "(▭ redraws one)." if by_motion else
                 f"⚡ The animals' areas could not be told apart: the picture was split into {len(rects)} equal "
                 "chambers — please check them and redraw (▭) where needed."), 12000)
            self._offer_preset()

        def failed(msg):
            QApplication.restoreOverrideCursor()
            self.btn_find.setEnabled(True)
            self.btn_find.setText("⚡ Find")
            QMessageBox.warning(c.win, "Find chambers", msg)
        c._thread_run(work, done, failed)

    def _offer_preset(self):
        """Zones drawn for the whole video (no chamber prefix) when the chambers were marked: offer the test's
        zones fitted into every chamber instead (whole-video zones would be cut between the chambers)."""
        from pro_ui.ux_controller import TESTS
        c = self.c
        plain = [n for n in c.canvas.zones if split_zone_name(n)[0] is None]
        if not plain or not self.enabled:
            return
        if QMessageBox.question(
                c.win, "Zones for the chambers",
                f"The current {len(plain)} zone(s) were drawn for the whole video.\n\nPut the "
                f"{TESTS[c.test_key][0]} preset zones into every chamber instead? (You can adjust them, or draw "
                "your own zones in each chamber.)") == QMessageBox.Yes:
            c.canvas.set_zones({n: p for n, p in c.canvas.zones.items() if n not in plain}, record=True)
            self.apply_preset(ask=False)

    # ------------------------------------------------------------------ zones
    def zone_name(self, name, pts):
        """Name for a newly drawn zone: with the chamber prefix (active tab, else where it was drawn)."""
        if not self.enabled:
            return name
        num, bare = split_zone_name(name)
        if num is not None:
            return name
        k = self.active if self.active is not None else chamber_of_zone(bare, pts, [list(r) for r in self.rects])
        if k is None:
            return name
        new = zone_prefix(k + 1) + bare
        base, i = new, 2
        while new in self.c.canvas.zones:
            new = f"{base} {i}"
            i += 1
        return new

    def merge(self, zones, target=None, base=None):
        """New zones (whole-video coordinates) into the layout `base` (default: the current zones): each goes to
        `target` (or the chamber it lies in) with its prefix; that chamber's old zones are replaced, the other
        chambers keep theirs.  Returns (merged zones, names of the new ones)."""
        base = self.c.canvas.zones if base is None else base
        rects = [list(r) for r in self.rects]
        new, hit = {}, set()
        for n, p in zones.items():
            k = target if target is not None else chamber_of_zone(n, p, rects)
            bare = split_zone_name(n)[1]
            if k is None:
                new[n] = p
                continue
            hit.add(k)
            new[zone_prefix(k + 1) + bare] = p
        keep = {n: p for n, p in base.items() if chamber_of_zone(n, p, rects) not in hit and n not in new}
        out = dict(keep)
        out.update(new)
        return out, list(new)

    def _targets(self):
        return [self.active] if self.active is not None else list(range(len(self.rects)))

    def apply_preset(self, ask=False):
        """▦ Preset in chamber mode: the test's template fitted into the chamber(s)."""
        from pro_ui.ux_controller import default_zones, TESTS
        c = self.c
        targets = self._targets()
        if ask and any(chamber_of_zone(n, p, [list(r) for r in self.rects]) in targets
                       for n, p in c.canvas.zones.items()):
            which = self.label(targets[0]) if len(targets) == 1 else "every chamber"
            if QMessageBox.question(c.win, "Preset zones", f"Replace the zones of {which} with the "
                                    f"{TESTS[c.test_key][0]} preset?") != QMessageBox.Yes:
                return
        zones = dict(c.canvas.zones)
        added = []
        for k in targets:
            x0, y0, x1, y1 = self.rects[k]
            try:
                tpl = default_zones(c.test_key, x1 - x0, y1 - y0)
            except Exception:
                tpl = {}
            tpl = {n: [(x + x0, y + y0) for x, y in p] for n, p in tpl.items()}
            zones, new = self.merge(tpl, target=k, base=zones)
            added += new
        c.canvas.colors.clear()
        c.canvas._color_i = 0
        c.canvas.set_zones(zones, record=True)
        if len(targets) == 1:
            c.canvas.select(added)                     # ready to be moved / resized onto that apparatus
        c.status.showMessage("Preset placed in " + (self.label(targets[0]) if len(targets) == 1 else "every chamber")
                             + " — drag / resize it onto the apparatus.", 10000)

    def auto_zones(self):
        """⚡ Auto in chamber mode: zones fitted to the apparatus inside each chamber."""
        c = self.c
        targets = self._targets()

        def ready(img):
            from tracking.apparatus import auto_detect_zones, guess_test
            ppc = c.spin_ppc.value() if c.spin_ppc.value() > 0 else None
            zones = dict(c.canvas.zones)
            added, failed = [], []
            for k in targets:
                x0, y0, x1, y1 = [int(round(v)) for v in self.rects[k]]
                sub = np.ascontiguousarray(img[y0:y1, x0:x1])
                kind = c.test_key
                if kind == "custom":
                    try:
                        kind = guess_test(sub) or "custom"
                    except Exception:
                        kind = "custom"
                try:
                    z, _o, _m = auto_detect_zones(sub, kind, ppc=ppc)
                except Exception:
                    z = {}
                if not z:
                    failed.append(self.label(k, short=True))
                    continue
                z = {n: [(x + x0, y + y0) for x, y in p] for n, p in z.items()}
                zones, new = self.merge(z, target=k, base=zones)
                added += new
            if added:
                c.canvas.colors.clear()
                c.canvas._color_i = 0
                c.canvas.set_zones(zones, record=True)
                if len(targets) == 1:
                    c.canvas.select(added)
            msg = (f"⚡ Zones fitted in {len(targets) - len(failed)} chamber(s)." if added else "⚡ No zones found.")
            if failed:
                msg += f"  Not found in {', '.join(failed)} — draw or ▦ Preset them there."
            c.status.showMessage(msg + "  Not right? Ctrl+Z.", 12000)
        c._background_image(ready)

    def copy_zones(self):
        c = self.c
        if len(self.rects) < 2:
            return
        src = self.active if self.active is not None else 0
        rects = [list(r) for r in self.rects]
        mine = {n: p for n, p in c.canvas.zones.items() if chamber_of_zone(n, p, rects) == src}
        if not mine:
            QMessageBox.information(c.win, "Copy zones", f"{self.label(src)} has no zones yet — draw them first "
                                    "(pick its tab above the video).")
            return
        others = [k for k in range(len(rects)) if k != src]
        if any(chamber_of_zone(n, p, rects) in others for n, p in c.canvas.zones.items()):
            if QMessageBox.question(c.win, "Copy zones", f"Replace the zones of the other chambers with the zones of "
                                    f"{self.label(src)}?") != QMessageBox.Yes:
                return
        sx0, sy0, sx1, sy1 = rects[src]
        zones = dict(c.canvas.zones)
        for k in others:
            tx0, ty0, tx1, ty1 = rects[k]
            fx, fy = (tx1 - tx0) / max(1e-6, sx1 - sx0), (ty1 - ty0) / max(1e-6, sy1 - sy0)
            moved = {split_zone_name(n)[1]: [(tx0 + (x - sx0) * fx, ty0 + (y - sy0) * fy) for x, y in p]
                     for n, p in mine.items()}
            zones, _new = self.merge(moved, target=k, base=zones)
        c.canvas.set_zones(zones, record=True)
        c.status.showMessage(f"Zones of {self.label(src)} copied to {len(others)} other chamber(s) — check them.",
                             8000)

    # ------------------------------------------------------------------ canvas painting
    def undimmed_rects(self):
        """Auto-mask preview: chambers without zones are not dimmed (they are tracked in full)."""
        if not self.enabled:
            return []
        rects = [list(r) for r in self.rects]
        have = {chamber_of_zone(n, p, rects) for n, p in self.c.canvas.zones.items()}
        return [r for i, r in enumerate(rects) if i not in have]

    def paint(self, p):
        if not self.tile.isChecked() or not self.rects:
            return
        cv = self.c.canvas
        f = QFont(cv.font())
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        for i, (x0, y0, x1, y1) in enumerate(self.rects):
            col = QColor(CHAMBER_HEX[i % len(CHAMBER_HEX)])
            a, b = cv.to_screen(x0, y0), cv.to_screen(x1, y1)
            r = QRectF(a, b)
            if self.active is not None and self.active != i:
                p.fillRect(r, QColor(5, 8, 15, 90))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 3.0 if self.active == i else 2.0, Qt.DashLine))
            p.drawRect(r)
            text = self.label(i)
            fm = p.fontMetrics()
            tw, tht = fm.horizontalAdvance(text) + 12, fm.height() + 4
            tag = QRectF(r.left(), r.top() - tht if r.top() - tht >= 0 else r.top(), tw, tht)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(tag, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(tag, Qt.AlignCenter, text)

    # ------------------------------------------------------------------ layout files
    def layout_data(self):
        return {"on": self.tile.isChecked(), "items": [{"rect": [float(v) for v in r], "id": it.get("id", ""),
                                                        "overrides": dict(it.get("overrides") or {})}
                                                       for r, it in zip(self.rects, self.items)]}

    def load_layout_data(self, data, sx=1.0, sy=1.0):
        items = (data or {}).get("items") or []
        self.rects[:] = [[r["rect"][0] * sx, r["rect"][1] * sy, r["rect"][2] * sx, r["rect"][3] * sy] for r in items]
        self.items = [{"id": r.get("id", ""), "overrides": dict(r.get("overrides") or {})} for r in items]
        self.tile.blockSignals(True)
        self.tile.setChecked(bool((data or {}).get("on")) and bool(self.rects))
        self.tile.blockSignals(False)
        self.refresh()

    # ================================================================== Tracking page
    def _build_tracking_row(self):
        c = self.c
        box = QFrame()
        box.setObjectName("chamberSettingsFor")
        box.setStyleSheet("QFrame#chamberSettingsFor { background: #f7f9fc; border: 1px solid #d5dce8; "
                          "border-radius: 8px; } QFrame#chamberSettingsFor QLabel { background: transparent; "
                          "border: none; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(5)
        t = QLabel("Settings for")
        t.setStyleSheet("font-weight: 700; color: #1f2a44;")
        v.addWidget(t)
        self.view_btns = QWidget()
        self.view_lay = QGridLayout(self.view_btns)
        self.view_lay.setContentsMargins(0, 0, 0, 0)
        self.view_lay.setHorizontalSpacing(4)
        self.view_lay.setVerticalSpacing(4)
        v.addWidget(self.view_btns)
        self.view_group = QButtonGroup(self)
        self.view_group.setExclusive(True)
        self.view_hint = QLabel("")
        self.view_hint.setWordWrap(True)
        self.view_hint.setStyleSheet("color: #64748b; font-size: 9pt;")
        v.addWidget(self.view_hint)
        card_lay = c.win.findChild(QFrame, "cardTrackingSettings").layout()
        card_lay.insertWidget(1, box)
        box.hide()
        self.view_box = box
        # the per-chamber settings: their changes are kept for the chamber shown
        c.spin_start.valueChanged.connect(lambda _v: self._captured())
        c.spin_end.valueChanged.connect(lambda _v: self._captured())
        c.cmb_color_mode.currentTextChanged.connect(lambda _t: self._captured())
        c.cmb_pol.currentIndexChanged.connect(lambda _i: self._captured())
        c.cmb_low.currentIndexChanged.connect(lambda _i: self._captured())
        c.sl_thr.valueChanged.connect(lambda _v: self._captured())

    def _shared_widgets(self):
        c = self.c
        out = [c.cmb_method, c.cmb_animals, c.sl_smooth, c.sl_min, c.btn_measure]
        for n in ("spin_prev", "cmb_darkbox"):
            if hasattr(c, n):
                out.append(getattr(c, n))
        out += list(getattr(c, "_window_widgets", []))
        return out

    def _refresh_tracking_row(self):
        show = self.enabled
        self.view_box.setVisible(show)
        while self.view_lay.count():
            it = self.view_lay.takeAt(0)
            w = it.widget()
            if w is not None:
                self.view_group.removeButton(w)
                w.hide()
                w.deleteLater()
        if not show:
            return
        names = ["All"] + [self.label(i, short=True) + (" •" if self.items[i].get("overrides") else "")
                           for i in range(len(self.rects))]
        for j, text in enumerate(names):
            b = _seg(text)
            b.setStyleSheet("padding:4px 6px;" + (f" border-left: 4px solid {CHAMBER_HEX[(j - 1) % 4]};" if j else ""))
            k = None if j == 0 else j - 1
            b.setChecked(self.view == k)
            b.clicked.connect(lambda _=False, kk=k: self.set_view(kk))
            self.view_group.addButton(b)
            self.view_lay.addWidget(b, j // 3, j % 3)
        self._update_hint()

    def _update_hint(self):
        if self.view is None:
            own = [f"{self.label(i, short=True)}: " + ", ".join(OVERRIDE_TEXT.get(k, k) for k in it["overrides"])
                   for i, it in enumerate(self.items) if it.get("overrides")]
            self.view_hint.setText("Settings below: every chamber." +
                                   (" Own: " + "; ".join(own) + "." if own else
                                    " Pick a chamber for its own time, colour or contrast."))
        else:
            self.view_hint.setText(f"Time, detection, colour, low light and contrast below: {self.label(self.view)} "
                                   "only (bold = own value).")
        self.view_hint.setToolTip("Per chamber: start / end time, detection mode, animal colour, low-light boost "
                                  "and minimum contrast. Everything else (engine, smoothing, animal size…) is the "
                                  "same for all chambers — pick All to change it.")
        _relayout(self.view_box)

    def _thr_key(self):
        return "min_contrast" if self.c.engine == "robust" else "loc_thresh"

    def _read(self):
        c = self.c
        thr = c.sl_thr.value() if self.c.engine == "robust" else c.sl_thr.value() / 10.0
        return {"start_time": float(c.spin_start.value()), "end_time": float(c.spin_end.value()),
                "color_mode": c.cmb_color_mode.currentText().lower(),
                "animal_polarity": c.cmb_pol.currentData() or "auto",
                "low_light": c.cmb_low.currentData() or "off", self._thr_key(): float(thr)}

    def _write(self, vals):
        c = self.c
        self._loading = True
        try:
            c.spin_start.setValue(float(vals["start_time"]))
            c.spin_end.setValue(float(vals["end_time"]))
            c.cmb_color_mode.setCurrentText("RGB" if vals["color_mode"] == "rgb" else "Gray")
            c.cmb_pol.setCurrentIndex(max(0, c.cmb_pol.findData(vals["animal_polarity"])))
            c.cmb_low.setCurrentIndex(max(0, c.cmb_low.findData(vals["low_light"])))
            k = self._thr_key()
            if k in vals:
                c.sl_thr.setValue(int(round(vals[k] if k == "min_contrast" else vals[k] * 10)))
        finally:
            self._loading = False

    def global_values(self):
        """The All values of the per-chamber settings (what the tracking widgets show in All)."""
        return dict(self._global) if self._global is not None else self._read()

    def set_view(self, k):
        if k == self.view:
            return
        if k is None:
            self.to_all()
            return
        if self.view is None:
            self._global = self._read()
        self.view = k
        ov = {kk: v for kk, v in (self.items[k].get("overrides") or {}).items() if kk in self._global}
        self._write(dict(self._global, **ov))
        for w in self._shared_widgets():
            w.setEnabled(False)
        self._style_own()
        self._update_hint()
        self.c._preview = None
        for b in self.view_group.buttons():
            b.setChecked(False)
        bs = self.view_group.buttons()
        if k + 1 < len(bs):
            bs[k + 1].setChecked(True)

    def to_all(self):
        """Back to the All view (before tracking / testing / leaving the page)."""
        if self.view is None:
            return
        self.view = None
        g, self._global = self._global, None
        if g is not None:
            self._write(g)
        for w in self._shared_widgets():
            w.setEnabled(True)
        if hasattr(self.c, "_engine_changed"):
            self.c.cmb_pol.setEnabled(self.c.engine == "robust")
            self.c.btn_auto_thr.setEnabled(self.c.engine == "robust")
        self._style_own()
        self._refresh_tracking_row()

    def _captured(self):
        if self._loading or self.view is None or self._global is None:
            return
        vals = self._read()
        ov = self.items[self.view].setdefault("overrides", {})
        for k, v in vals.items():
            g = self._global.get(k)
            if g is None or (isinstance(v, float) and abs(v - float(g)) < 1e-9) or v == g:
                ov.pop(k, None)
            else:
                ov[k] = v
        self._style_own()
        bs = self.view_group.buttons()
        if self.view + 1 < len(bs):
            bs[self.view + 1].setText(self.label(self.view, short=True) + (" •" if ov else ""))

    def _style_own(self):
        c = self.c
        ov = (self.items[self.view].get("overrides") or {}) if self.view is not None else {}
        pairs = ((c.spin_start, "start_time"), (c.spin_end, "end_time"), (c.cmb_color_mode, "color_mode"),
                 (c.cmb_pol, "animal_polarity"), (c.cmb_low, "low_light"))
        for w, k in pairs:
            f = w.font()
            f.setBold(k in ov)
            w.setFont(f)
        f = c.lbl_thr_val.font()
        f.setBold(self._thr_key() in ov)
        c.lbl_thr_val.setFont(f)

    # ------------------------------------------------------------------ setups for the engine
    def chambers_for(self, size, item=None):
        """The chambers in the coordinates of a setup built for a video of `size` (as _build_setup maps the
        zones): [{"rect", "id", "overrides"}]."""
        c = self.c
        own = item or {}
        crop = own.get("crop") if own.get("zones") else c.crop
        ref_size = tuple(own.get("ref_size") or (c.player.w, c.player.h))
        vw, vh = size
        out = []
        for r, it in zip(self.rects, self.items):
            x0, y0, x1, y1 = r
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if crop and (vw, vh) == ref_size:
                pts = corners
            else:
                if crop:
                    Minv = np.asarray(crop["inv"], np.float64)
                    back = []
                    for px, py in corners:
                        v = Minv @ np.array([px, py, 1.0])
                        back.append((v[0] / v[2], v[1] / v[2]))
                    corners = back
                sx, sy = vw / max(1, ref_size[0]), vh / max(1, ref_size[1])
                pts = [(px * sx, py * sy) for px, py in corners]
            ids = own.get("chamber_ids") or []
            k = len(out)
            cid = ids[k] if k < len(ids) and ids[k] else it.get("id", "")
            out.append({"rect": _rect_of(pts), "id": cid, "overrides": dict(it.get("overrides") or {})})
        return out

    def tracking_job(self, path, setup):
        size = (self.c.player.w, self.c.player.h)
        return (path, setup, "chambers", self.chambers_for(size))

    # ------------------------------------------------------------------ results
    def show_run(self, summary, video=None):
        """A chamber run finished (or its batch row is opened): every chamber in 'Showing', the first one shown."""
        c = self.c
        video = video or summary.get("Video_file") or ""
        self.results = []
        for sm in summary.get("Chambers") or []:
            n, cid = int(sm.get("Chamber") or len(self.results) + 1), sm.get("Animal_ID") or ""
            self.results.append({"label": chamber_label(n, cid), "bta": sm.get("Project_file"), "summary": sm,
                                 "video": video})
        if not self.results:
            return False
        first = self.results[0]
        if first.get("bta") and os.path.exists(first["bta"]):
            c.open_bta(first["bta"], show=True)
        else:
            c.show_results(first["summary"])
            c.go(4)
        c.set_results_mode("tracking")
        c._refresh_showing()
        return True

    def current_index(self):
        bta = (self.c.bta or {}).get("_path") if getattr(self.c, "bta", None) else None
        for i, e in enumerate(self.results):
            if bta and e.get("bta") and os.path.normcase(os.path.abspath(e["bta"])) == \
                    os.path.normcase(os.path.abspath(bta)):
                return i
        return None

    def open_result(self, i):
        c = self.c
        if not (0 <= i < len(self.results)):
            return
        e = self.results[i]
        c.set_results_mode("tracking")
        if self.current_index() == i:
            return
        if e.get("bta") and os.path.exists(e["bta"]):
            c.open_bta(e["bta"], show=True)
        else:
            c.show_results(e["summary"])
        c._refresh_showing()

    # ------------------------------------------------------------------ tracking page picture
    def overlay(self, out):
        """Chamber frames + names on a Tracking-page picture (in place)."""
        if not self.enabled:
            return out
        from tracking.chambers import CHAMBER_BGR
        fs = max(0.4, min(out.shape[:2]) / 1100.0)
        th = max(1, int(round(1.4 * fs)))
        for i, (x0, y0, x1, y1) in enumerate(self.rects):
            col = CHAMBER_BGR[i % len(CHAMBER_BGR)]
            p0, p1 = (int(round(x0)), int(round(y0))), (int(round(x1)) - 1, int(round(y1)) - 1)
            _dashed_rect(out, p0, p1, col, max(2, th))
            text = self.label(i)
            (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ty0 = p0[1] - tht - base - 6 if p0[1] - tht - base - 6 >= 0 else p0[1]
            cv2.rectangle(out, (p0[0], ty0), (p0[0] + tw + 8, ty0 + tht + base + 6), col, -1)
            cv2.putText(out, text, (p0[0] + 4, ty0 + tht + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th,
                        cv2.LINE_AA)
        return out

    def adjust_exclusion(self, ex):
        """Tracking-page dimming in chamber mode: outside the chambers dimmed, a chamber without zones not."""
        if not self.enabled or ex is None:
            return ex
        rects = [list(r) for r in self.rects]
        have = {chamber_of_zone(n, p, rects) for n, p in self.c.canvas.zones.items()}
        out = np.ones_like(ex, bool)
        for i, (x0, y0, x1, y1) in enumerate(rects):
            a, b = int(round(y0)), int(round(y1))
            l, r = int(round(x0)), int(round(x1))
            out[a:b, l:r] = ex[a:b, l:r] if i in have else False
        return out

    # ------------------------------------------------------------------ test detection
    def test_detection(self):
        c = self.c
        self.to_all()
        if not c._validate_ready(quiet_zones=True):
            return
        setup = c._build_setup(c.player.path, (c.player.w, c.player.h), start=float(c.spin_start.value()),
                               end=float(c.spin_end.value()))
        chambers = self.chambers_for((c.player.w, c.player.h))
        key = "chambers|" + c._preview_key(setup) + json.dumps(chambers, sort_keys=True, default=str)
        if c._preview is not None and c._preview.get("key") == key:
            c._preview_from_start()
            return
        if c._preview_busy:
            return
        c._preview_busy = True
        c.btn_test.setEnabled(False)
        c.btn_test.setText("Building backgrounds…")
        path = c.player.path
        setups = chamber_setups(setup, chambers)

        def work():
            from tracking.chambers import preview_chambers
            return preview_chambers(path, setups)

        def restore():
            from pro_ui.ux_controller import TEST_BTN_TEXT
            c._preview_busy = False
            c.btn_test.setEnabled(True)
            c.btn_test.setText(TEST_BTN_TEXT)

        def done(chs):
            restore()
            c._preview = {"key": key, "video": path, "chambers": chs}
            c._preview_from_start()

        def failed(msg):
            restore()
            QMessageBox.warning(c.win, "Test detection", msg)
        c._thread_run(work, done, failed)

    def preview_detection(self, pv):
        """Test detection picture of the current frame: every chamber's animal (start time: as the run finds it)."""
        from tracking.chambers import ChamberLive
        from tracking.location import new_tracker, apply_backfill
        c = self.c
        chs = pv["chambers"]
        frame = c.cur_frame
        idx = c.player.current
        fps = c.player.fps or 30.0
        if pv.get("live") is None:
            h, w = frame.shape[:2]
            pv["live"] = ChamberLive(w, h, [{"rect": ch["rect"], "roi_points": ch["roi_points"],
                                             "exclusion": ch["exclusion"], "label": ch["label"], "tag": ch["tag"]}
                                            for ch in chs])
        trs = pv.get("trackers") if pv.get("tracker_at") == idx - 1 else None
        start = idx == pv.get("start_frame")
        dets = [None] * len(chs)
        if trs is None and start:
            # start time: a fresh tracker per chamber looks ~1.5 s ahead (like the run); the position found is put
            # back on the start frame
            trs = [new_tracker(ch["setup"], ch["bgm"], ch["exclusion"]) for ch in chs]
            n = max(int(getattr(t, "first_window", 30) or 30) for t in trs) + int(round(1.5 * fps))
            cols = [([], [], [], []) for _ in chs]
            cap = cv2.VideoCapture(c.player.path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                for _k in range(n):
                    ok, f = cap.read()
                    if not ok:
                        break
                    full = c._warp(f)
                    for j, ch in enumerate(chs):
                        xs, ys, ar, st = cols[j]
                        if st and st[0] == "Tracked":
                            continue
                        x0, y0, x1, y1 = ch["rect"]
                        d = trs[j].update(np.ascontiguousarray(full[y0:y1, x0:x1]))
                        apply_backfill(d, xs, ys, ar, st)
                        xs.append(d["x"]), ys.append(d["y"]), ar.append(d["area"]), st.append(d["status"])
                    if all(cl[3] and cl[3][0] == "Tracked" for cl in cols):
                        break
            finally:
                cap.release()
            probes = []
            for j, ch in enumerate(chs):
                xs, ys, ar, st = cols[j]
                ok = bool(st) and st[0] == "Tracked"
                det, probe = _still(ch, frame, xs[0] if ok else None, ys[0] if ok else None, ok)
                dets[j] = det
                probes.append(probe)
            pv["trackers"], pv["tracker_at"] = None, None
            disp = pv["live"].draw(frame, dets, probes, idx / fps, idx, lost_label="NOT FOUND")
        else:
            if trs is None:
                trs = [new_tracker(ch["setup"], ch["bgm"], ch["exclusion"]) for ch in chs]
                self._warm_up(pv, trs, idx)
            for j, ch in enumerate(chs):
                x0, y0, x1, y1 = ch["rect"]
                dets[j] = trs[j].update(np.ascontiguousarray(frame[y0:y1, x0:x1]))
            pv["trackers"], pv["tracker_at"] = trs, idx
            disp = pv["live"].draw(frame, dets, trs, idx / fps, idx)
        c._set_track_view(disp)
        parts = [f"{'✓' if d and d.get('status') == 'Tracked' else '✗'} {ch['tag']}" for d, ch in zip(dets, chs)]
        ok_all = all(d and d.get("status") == "Tracked" for d in dets)
        c.status.showMessage(("Start time: " if start else "") + "   ".join(parts) + (
            "  — every animal found. Press play or scrub to check other moments." if ok_all else
            "  — ✗ = not found: pick that chamber under 'Settings for' and try 🎯 Auto-set contrast or its animal "
            "colour, or check its frame / zones."), 20000)

    def _warm_up(self, pv, trs, idx):
        c = self.c
        n = max(int(getattr(t, "first_window", 0) or 0) for t in trs) + 6
        f0 = min(ch["start_frame"] for ch in pv["chambers"])
        a = max(0, f0, idx - n)
        if n <= 6 or a >= idx:
            return
        cap = cv2.VideoCapture(c.player.path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            for _i in range(a, idx):
                ok, f = cap.read()
                if not ok:
                    break
                full = c._warp(f)
                for t, ch in zip(trs, pv["chambers"]):
                    x0, y0, x1, y1 = ch["rect"]
                    t.update(np.ascontiguousarray(full[y0:y1, x0:x1]))
        finally:
            cap.release()

    # ------------------------------------------------------------------ minimum contrast per chamber
    def auto_contrast(self):
        """🎯 Auto-set contrast in chamber mode: measured in every chamber (or the chamber shown)."""
        c = self.c
        if c.engine != "robust" or c._preview_busy:
            return
        if not c._validate_ready(quiet_zones=True):
            return
        g = self.global_values()
        setup = c._build_setup(c.player.path, (c.player.w, c.player.h), start=g["start_time"], end=g["end_time"])
        chambers = self.chambers_for((c.player.w, c.player.h))
        which = [self.view] if self.view is not None else list(range(len(chambers)))
        setups = [chamber_setups(setup, chambers)[k] for k in which]
        path = c.player.path
        c._preview_busy = True
        c.btn_auto_thr.setEnabled(False)
        c.btn_test.setEnabled(False)
        c.btn_auto_thr.setText("Checking the video…")

        def work():
            from tracking.chambers import preview_chambers
            from tracking.location import measure_animal_contrast
            res = preview_chambers(path, setups)
            out = []
            cap = cv2.VideoCapture(path)
            try:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                for r in res:
                    s = r["setup"]
                    f0 = r["start_frame"]
                    f1 = int(round(min(float(s.get("end_time") or 1e9), c.player.duration) * fps))
                    frames = []
                    x0, y0, x1, y1 = r["rect"]
                    for i in np.linspace(f0, max(f0, f1 - 1), 40).astype(int):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                        ok, f = cap.read()
                        if ok:
                            frames.append(np.ascontiguousarray(c._warp(f)[y0:y1, x0:x1]))
                    out.append(measure_animal_contrast(s, r["bgm"], r["exclusion"], frames))
            finally:
                cap.release()
            return out

        def restore():
            c._preview_busy = False
            c.btn_auto_thr.setEnabled(c.engine == "robust")
            c.btn_test.setEnabled(True)
            c.btn_auto_thr.setText("🎯  Auto-set contrast")

        def done(ms):
            restore()
            lines, recs = [], {}
            for k, m in zip(which, ms):
                rec = m.get("recommended")
                if rec is None:
                    lines.append(f"• {self.label(k)}: the animal could not be found clearly "
                                 f"({m.get('found', 0)} of {m.get('frames', 0)} frames) — check its frame, zones "
                                 "and animal colour.")
                    continue
                recs[k] = int(rec)
                lines.append(f"• {self.label(k)}: animal ≈{m['peak_median']:.0f} grey levels from the floor "
                             f"(weakest ≈{m['peak_low']:.0f}), noise ≈{m['sigma']:.1f} → recommended {rec} "
                             f"(found in {m['found_with_rec']} of {m['frames']} frames).")
            box = QMessageBox(c.win)
            box.setWindowTitle("Minimum contrast")
            box.setIcon(QMessageBox.Information)
            box.setText("Minimum contrast measured in each chamber:\n\n" + "\n".join(lines))
            use = None
            if recs:
                use = box.addButton("Use for each chamber" if len(recs) > 1 else f"Use {list(recs.values())[0]}",
                                    QMessageBox.AcceptRole)
            box.addButton("Keep", QMessageBox.RejectRole)
            box.exec()
            if use is not None and box.clickedButton() is use:
                self._apply_contrast(recs)
                c.status.showMessage("Minimum contrast set per chamber. Press 'Test detection' to check.", 8000)

        def failed(msg):
            restore()
            QMessageBox.warning(c.win, "Minimum contrast", msg)
        c._thread_run(work, done, failed)

    def _apply_contrast(self, recs):
        c = self.c
        if self.view is not None and list(recs) == [self.view]:
            c.sl_thr.setValue(recs[self.view])          # recorded as this chamber's own value
            return
        g = self.global_values()
        vals = set(recs.values())
        if len(vals) == 1 and len(recs) == len(self.rects):
            if self.view is None:
                c.sl_thr.setValue(vals.pop())
            for it in self.items:
                it.get("overrides", {}).pop("min_contrast", None)
        else:
            for k, v in recs.items():
                ov = self.items[k].setdefault("overrides", {})
                if abs(float(v) - float(g.get("min_contrast", 12))) < 1e-9:
                    ov.pop("min_contrast", None)
                else:
                    ov["min_contrast"] = float(v)
            if self.view is not None and self.view in recs:
                self._loading = True
                c.sl_thr.setValue(int(recs[self.view]))
                self._loading = False
        c._preview = None
        self._style_own()
        self._refresh_tracking_row()

    # ================================================================== Batch Processing
    BATCH_CHOICES = (("Video Setup", None), ("Off", "off"), ("Find 1", 1), ("Find 2", 2), ("Find 3", 3),
                     ("Find 4", 4), ("Own (Set zones…)", "own"))

    def build_batch_row(self, grid, cap, row, col):
        """'Chambers' below 'Scale' in the batch panel."""
        grid.addWidget(cap("Chambers"), row, col)
        h = QHBoxLayout()
        self.cmb_bp = QComboBox()
        for text, data in self.BATCH_CHOICES:
            self.cmb_bp.addItem(text, data)
        self.cmb_bp.setMinimumWidth(110)
        self.cmb_bp.setToolTip("Chambers of this video: 'Video Setup' = the chambers marked on the Video Setup page "
                               "(when chamber mode is on there), 'Off' = one animal in the whole video, 'Find n' = "
                               "find n chambers in this video automatically, 'Own' = the chambers drawn for this "
                               "video in ✏ Set zones….")
        self.cmb_bp.currentIndexChanged.connect(self._bp_changed)
        self.btn_bp_ids = _btn("Chambers…", "Animal ID and start / end time of every chamber of this video "
                                            "(empty time = the video's time window).")
        self.btn_bp_ids.clicked.connect(self._bp_ids)
        h.addWidget(self.cmb_bp)
        h.addWidget(self.btn_bp_ids)
        grid.addLayout(h, row + 1, col)
        return [self.cmb_bp, self.btn_bp_ids]

    def raw_rects_for(self, size):
        """The Video Setup chambers on the raw frame of a video of `size` (as the batch zone editor shows it)."""
        c = self.c
        ref = (c.player.w, c.player.h) if c.player.w else tuple(size)
        sx, sy = size[0] / max(1, ref[0]), size[1] / max(1, ref[1])
        out = []
        for x0, y0, x1, y1 in self.rects:
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if c.crop:
                Minv = np.asarray(c.crop["inv"], np.float64)
                pts = [((Minv @ np.array([px, py, 1.0]))[:2] / (Minv @ np.array([px, py, 1.0]))[2]) for px, py in pts]
            out.append(_rect_of([(px * sx, py * sy) for px, py in pts]))
        return out

    def batch_mode(self, it):
        """None (no chambers) | ("layout" | "find" | "own", n) for a batch video."""
        m = it.get("chambers")
        if m == "off":
            return None
        if m == "own":
            return ("own", len(it["chamber_rects"])) if it.get("chamber_rects") else None
        if isinstance(m, int) and m > 0:
            return ("find", m)
        if self.enabled:
            if it.get("zones") and it.get("crop"):
                return ("find", len(self.rects))       # its own straightening: chambers found in it
            return ("layout", len(self.rects))
        return None

    def batch_ids(self, it, n):
        ids = list(it.get("chamber_ids") or [])
        out = []
        for k in range(n):
            if k < len(ids) and ids[k]:
                out.append(ids[k])
            elif k < len(self.items) and self.items[k].get("id") and not it.get("subject"):
                out.append(self.items[k]["id"])
            elif it.get("subject"):
                out.append(f"{it['subject']}_C{k + 1}")
            else:
                out.append(f"M{k + 1}")
        return out

    @staticmethod
    def batch_times(it, n):
        t = list(it.get("chamber_times") or [])
        return [(t[k] if k < len(t) and t[k] else None) for k in range(n)]

    def batch_text(self, it):
        m = self.batch_mode(it)
        if m is None:
            return ""
        ids = self.batch_ids(it, m[1])
        own_t = any(self.batch_times(it, m[1]))
        return f"▦ {m[1]} chamber{'s' if m[1] > 1 else ''} ({', '.join(ids)}){' ⏱' if own_t else ''} · "

    def batch_job(self, it, setup):
        """The batch job of one video: chambers or not.  Each chamber's own start / end (Chambers…) is kept; the
        start / end set per chamber on the Tracking page belongs to that video and is not used here."""
        m = self.batch_mode(it)
        if m is None:
            return None
        n = m[1]
        ids = self.batch_ids(it, n)
        times = self.batch_times(it, n)

        def own_time(k):
            ov = {}
            if times[k]:
                a, b = times[k]
                if a is not None:
                    ov["start_time"] = float(a)
                if b is not None:
                    ov["end_time"] = float(b)
            return ov
        if m[0] in ("layout", "own"):
            if m[0] == "layout":
                ch = self.chambers_for(tuple(it["size"]), item=it)
            else:
                ch = [{"rect": list(r), "id": "", "overrides": {}} for r in it["chamber_rects"]]
            for k, cc in enumerate(ch):
                cc["id"] = ids[k]
                ov = {kk: v for kk, v in (cc.get("overrides") or {}).items() if kk not in ("start_time", "end_time")}
                ov.update(own_time(k))
                cc["overrides"] = ov
            return (it["path"], setup, "chambers", ch)
        return (it["path"], setup, "chambers", {"find": n, "ids": ids,
                                                "overrides": [own_time(k) for k in range(n)]})

    def batch_selected(self, it):
        self.cmb_bp.blockSignals(True)
        m = it.get("chambers")
        i = self.cmb_bp.findData(m) if m is not None else 0
        self.cmb_bp.setCurrentIndex(max(0, i))
        self.cmb_bp.setStyleSheet("font-weight:600;" if m is not None else "")
        self.cmb_bp.blockSignals(False)

    def _bp_rows(self):
        c = self.c
        return [r for r in c._batch_selected_rows() if r < len(c.batch_items)]

    def _bp_changed(self, _i):
        c = self.c
        if getattr(c, "_sync", False):
            return
        data = self.cmb_bp.currentData()
        rows = self._bp_rows()
        if data == "own" and rows and not c.batch_items[rows[0]].get("chamber_rects"):
            QMessageBox.information(c.win, "Chambers", "Draw (▭) or find (⚡) this video's chambers in ✏ Set zones… "
                                    "— they are used as soon as you save there.")
            self.batch_selected(c.batch_items[rows[0]])
            c.open_batch_zone_editor(rows[0])
            return
        for r in rows:
            if data is None:
                c.batch_items[r].pop("chambers", None)
            else:
                c.batch_items[r]["chambers"] = data
            c._refresh_batch_row(r)
        self.cmb_bp.setStyleSheet("font-weight:600;" if data is not None else "")

    def _bp_ids(self):
        c = self.c
        rows = self._bp_rows()
        if not rows:
            return
        it = c.batch_items[rows[0]]
        m = self.batch_mode(it)
        n = m[1] if m else (len(self.rects) or 2)
        res = chamber_table_dialog(c.win, f"Chambers — {Path(it['path']).name}", n, self.batch_ids(it, n),
                                   self.batch_times(it, n), float(it.get("duration") or 0))
        if res is None:
            return
        ids, times = res
        for r in rows:
            c.batch_items[r]["chamber_ids"] = list(ids)
            c.batch_items[r]["chamber_times"] = [list(t) if t else None for t in times]
            c._refresh_batch_row(r)

    @staticmethod
    def batch_apply(it, src):
        """Batch ▸ Apply to all videos ▸ Chambers: the chamber choice, IDs, times (and own chambers, fitted to the
        video's size) of `src` on `it`."""
        for k in ("chambers", "chamber_ids", "chamber_times", "chamber_rects"):
            it.pop(k, None)
        if src.get("chambers") is not None:
            it["chambers"] = src["chambers"]
        if src.get("chamber_ids"):
            it["chamber_ids"] = list(src["chamber_ids"])
        if src.get("chamber_times"):
            it["chamber_times"] = [list(t) if t else None for t in src["chamber_times"]]
        if src.get("chamber_rects"):
            sx = it["size"][0] / max(1, src["size"][0])
            sy = it["size"][1] / max(1, src["size"][1])
            it["chamber_rects"] = [[r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy] for r in src["chamber_rects"]]


def chamber_table_dialog(parent, title, n, ids, times, duration=0.0):
    """Animal ID + start / end (s) of every chamber.  Returns (ids, [(start, end) | None]) or None."""
    from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    v = QVBoxLayout(dlg)
    note = QLabel("Animal ID and time window of each chamber. Leave Start / End empty to use the video's time "
                  "window (e.g. when the animals were put in one after the other, give each chamber its own "
                  "start).")
    note.setWordWrap(True)
    note.setStyleSheet("color:#475569;")
    v.addWidget(note)
    t = QTableWidget(n, 3)
    t.setHorizontalHeaderLabels(["Animal ID", "Start (s)", "End (s)"])
    t.setVerticalHeaderLabels([f"Chamber {k + 1}" for k in range(n)])
    t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    for k in range(n):
        t.setItem(k, 0, QTableWidgetItem(ids[k] if k < len(ids) else ""))
        tm = times[k] if k < len(times) else None
        t.setItem(k, 1, QTableWidgetItem("" if not tm or tm[0] is None else f"{float(tm[0]):g}"))
        t.setItem(k, 2, QTableWidgetItem("" if not tm or tm[1] is None else f"{float(tm[1]):g}"))
    t.setMinimumHeight(60 + 30 * n)
    v.addWidget(t)
    bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    bb.accepted.connect(dlg.accept)
    bb.rejected.connect(dlg.reject)
    v.addWidget(bb)
    dlg.resize(460, 140 + 34 * n)
    if dlg.exec() != QDialog.Accepted:
        return None

    def num(item):
        txt = (item.text() if item is not None else "").strip().replace(",", ".")
        if not txt:
            return None
        try:
            if ":" in txt:                                   # mm:ss
                m_, s_ = txt.split(":", 1)
                return max(0.0, float(m_) * 60 + float(s_))
            return max(0.0, float(txt))
        except ValueError:
            return None
    out_ids, out_t = [], []
    for k in range(n):
        it0 = t.item(k, 0)
        out_ids.append((it0.text() if it0 is not None else "").strip())
        a, b = num(t.item(k, 1)), num(t.item(k, 2))
        if a is not None and b is not None and b <= a:
            b = None
        out_t.append((a, b) if (a is not None or b is not None) else None)
    return out_ids, out_t


class BatchChamberPanel(QObject):
    """Batch ▸ ✏ Set zones…: the chambers of THIS video (drawn / found on its own frame), their animal IDs and
    times.  Saved with the video's zones; the batch then uses them ('Own')."""

    def __init__(self, dlg, layout, ctrl):
        super().__init__(dlg)
        self.dlg, self.ctrl = dlg, ctrl
        self.rects = dlg.canvas.chamber_rects
        self.ids, self.times = [], []
        self.active = None
        self._draw_idx = None
        self.changed = False
        box = QFrame()
        box.setObjectName("bzChambers")
        box.setStyleSheet("QFrame#bzChambers { border: 1px solid #d5dce8; border-radius: 8px; background: #f7f9fc; } "
                          "QFrame#bzChambers QLabel { border: none; background: transparent; }")
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)
        t = QLabel("Chambers in this video")
        t.setStyleSheet("font-weight:700;")
        v.addWidget(t)
        row = QHBoxLayout()
        self.spin = QSpinBox()
        self.spin.setRange(0, MAX_CHAMBERS)
        self.spin.setSpecialValueText("none")
        self.spin.setToolTip("How many chambers (animals) this video has (none = one animal in the whole video).")
        self.spin.valueChanged.connect(self._count)
        b_find = _btn("⚡ Find", "Find the chambers in this video.")
        b_find.clicked.connect(self.find)
        b_draw = _btn("▭ Draw", "Draw a chamber (drag a rectangle).")
        b_draw.clicked.connect(lambda: self.begin_draw(len(self.rects) if len(self.rects) < MAX_CHAMBERS else 0))
        row.addWidget(self.spin)
        row.addWidget(b_find)
        row.addWidget(b_draw)
        v.addLayout(row)
        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setStyleSheet("color:#475569; font-size:9pt;")
        v.addWidget(self.lbl)
        row2 = QHBoxLayout()
        self.b_ids = _btn("IDs && times…", "Animal ID and start / end time of every chamber.")
        self.b_ids.clicked.connect(self.edit_table)
        self.b_copy = _btn("⧉ C1 zones → all", "Copy the zones of chamber 1 into the other chambers.")
        self.b_copy.clicked.connect(self.copy_zones)
        row2.addWidget(self.b_ids)
        row2.addWidget(self.b_copy)
        v.addLayout(row2)
        layout.addWidget(box)
        dlg.canvas.extra_paint = self.paint
        dlg.canvas.extra_inside = lambda: [list(r) for i, r in enumerate(self.rects)
                                           if not any(chamber_of_zone(n, p, [list(q) for q in self.rects]) == i
                                                      for n, p in dlg.canvas.zones.items())]

    # ---------------------------------------------------------------- state
    def load(self, it):
        c = self.ctrl.chamber_ui
        self.rects.clear()
        mode = c.batch_mode(it)
        src = ""
        if it.get("chamber_rects"):
            self.rects.extend([list(r) for r in it["chamber_rects"]])
            src = "This video's own chambers."
        elif mode is not None and mode[0] == "layout":
            self.rects.extend(c.raw_rects_for(tuple(it["size"])))
            src = "The Video Setup chambers (fitted to this video) — move them by redrawing (▭) if needed."
        elif mode is not None and mode[0] == "find":
            src = f"{mode[1]} chambers will be found when the batch runs — or press ⚡ Find to see / fix them now."
        n = len(self.rects) or (mode[1] if mode else 0)
        self.ids = c.batch_ids(it, n) if n else []
        self.times = c.batch_times(it, n) if n else []
        self.changed = False
        self._src = src
        self._refresh()

    def save(self, it):
        if not self.changed:
            return
        if self.rects:
            it["chambers"] = "own"
            it["chamber_rects"] = [list(map(float, r)) for r in self.rects]
            it["chamber_ids"] = list(self.ids[:len(self.rects)])
            it["chamber_times"] = [list(t) if t else None for t in self.times[:len(self.rects)]]
        else:
            it["chambers"] = "off"
            it.pop("chamber_rects", None)
        self.changed = False

    def _mark(self):
        self.changed = True
        self.dlg.dirty = True
        self._refresh()

    def _refresh(self):
        n = len(self.rects)
        while len(self.ids) < n:
            self.ids.append(f"M{len(self.ids) + 1}")
        while len(self.times) < n:
            self.times.append(None)
        self.spin.blockSignals(True)
        self.spin.setValue(n)
        self.spin.blockSignals(False)
        if n:
            lab = ", ".join(f"C{k + 1} {self.ids[k]}" + (" ⏱" if self.times[k] else "") for k in range(n))
            self.lbl.setText(lab + ("  —  " + self._src if getattr(self, "_src", "") and not self.changed else ""))
        else:
            self.lbl.setText(getattr(self, "_src", "") or "One animal in the whole video. Press ⚡ Find or ▭ Draw "
                                                        "for chambers.")
        self.b_ids.setEnabled(n > 0)
        self.b_copy.setEnabled(n > 1)
        self.dlg.canvas.update()

    def _count(self, n):
        if n == len(self.rects):
            return
        if n < len(self.rects):
            del self.rects[n:]
            self._mark()
        else:
            self.find(n)

    # ---------------------------------------------------------------- drawing / finding
    def begin_draw(self, k):
        self._draw_idx = int(k)
        self.dlg.canvas.start_tool("rect", "chamber", banner=f"Drag a rectangle around chamber {k + 1}.")
        self.dlg.lbl_hint.setText(f"Drag a rectangle around chamber {k + 1} (include its walls). Esc cancels.")

    def rect_drawn(self, pts):
        k = self._draw_idx if self._draw_idx is not None else len(self.rects)
        self._draw_idx = None
        iw, ih = self.dlg.canvas.frame_size
        x0, y0, x1, y1 = _rect_of(pts)
        r = [max(0.0, x0), max(0.0, y0), min(float(iw), x1), min(float(ih), y1)]
        if r[2] - r[0] < 8 or r[3] - r[1] < 8:
            return
        if k < len(self.rects):
            self.rects[k] = r
        elif len(self.rects) < MAX_CHAMBERS:
            self.rects.append(r)
        self._mark()

    def find(self, n=None):
        it = self.dlg.item
        n = int(n or self.spin.value() or max(2, len(self.rects)))
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from tracking.chambers import find_chambers
            rects, by_motion = find_chambers(it["path"], n)
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self.dlg, "Find chambers", str(exc))
            return
        QApplication.restoreOverrideCursor()
        self.rects[:] = [list(map(float, r)) for r in rects]
        self._src = "" if by_motion else "Could not tell the animals apart: even split — please check."
        self._mark()

    def edit_table(self):
        n = len(self.rects)
        res = chamber_table_dialog(self.dlg, "Chambers of this video", n, self.ids, self.times,
                                   float(self.dlg.item.get("duration") or 0))
        if res is not None:
            self.ids, self.times = list(res[0]), list(res[1])
            self._mark()

    # ---------------------------------------------------------------- zones
    def zone_name(self, name, pts):
        if not self.rects:
            return name
        num, bare = split_zone_name(name)
        if num is not None:
            return name
        k = chamber_of_zone(bare, pts, [list(r) for r in self.rects])
        if k is None:
            return name
        new = zone_prefix(k + 1) + bare
        base, i = new, 2
        while new in self.dlg.canvas.zones:
            new = f"{base} {i}"
            i += 1
        return new

    def preset(self, test_key):
        """The test's preset zones in every chamber (prefixed)."""
        from pro_ui.ux_controller import default_zones
        zones = {}
        for k, (x0, y0, x1, y1) in enumerate(self.rects):
            try:
                tpl = default_zones(test_key, x1 - x0, y1 - y0)
            except Exception:
                tpl = {}
            for n, p in tpl.items():
                zones[zone_prefix(k + 1) + n] = [(x + x0, y + y0) for x, y in p]
        return zones

    def copy_zones(self):
        cv = self.dlg.canvas
        rects = [list(r) for r in self.rects]
        mine = {n: p for n, p in cv.zones.items() if chamber_of_zone(n, p, rects) == 0}
        if not mine:
            QMessageBox.information(self.dlg, "Copy zones", "Chamber 1 has no zones yet.")
            return
        sx0, sy0, sx1, sy1 = rects[0]
        zones = {n: p for n, p in cv.zones.items() if chamber_of_zone(n, p, rects) in (0, None)}
        for k in range(1, len(rects)):
            tx0, ty0, tx1, ty1 = rects[k]
            fx, fy = (tx1 - tx0) / max(1e-6, sx1 - sx0), (ty1 - ty0) / max(1e-6, sy1 - sy0)
            for n, p in mine.items():
                zones[zone_prefix(k + 1) + split_zone_name(n)[1]] = [(tx0 + (x - sx0) * fx, ty0 + (y - sy0) * fy)
                                                                     for x, y in p]
        cv.set_zones(zones, record=True)

    # ---------------------------------------------------------------- painting
    def paint(self, p):
        cv = self.dlg.canvas
        if not self.rects:
            return
        f = QFont(cv.font())
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        for i, (x0, y0, x1, y1) in enumerate(self.rects):
            col = QColor(CHAMBER_HEX[i % len(CHAMBER_HEX)])
            r = QRectF(cv.to_screen(x0, y0), cv.to_screen(x1, y1))
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 2.0, Qt.DashLine))
            p.drawRect(r)
            text = chamber_label(i + 1, self.ids[i] if i < len(self.ids) else "")
            fm = p.fontMetrics()
            tw, th = fm.horizontalAdvance(text) + 12, fm.height() + 4
            tag = QRectF(r.left(), r.top() - th if r.top() - th >= 0 else r.top(), tw, th)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(tag, 3, 3)
            p.setPen(QColor("#ffffff"))
            p.drawText(tag, Qt.AlignCenter, text)


def _relayout(w):
    """Rows were added / text changed inside a card that sits in a scroll area: make its parent layout measure it
    again (otherwise Qt keeps the old height and squeezes the rows)."""
    def go():
        try:
            lay = w.layout()
            if lay is not None:
                lay.invalidate()
                lay.activate()
            w.updateGeometry()
            p = w.parentWidget()
            if p is not None and p.layout() is not None:
                p.layout().invalidate()
                p.layout().activate()
        except RuntimeError:                     # deleted meanwhile
            pass
    go()
    QTimer.singleShot(0, go)


def _still(ch, frame, x, y, ok):
    """(det, probe) of one chamber on a frame with the animal at (x, y) (chamber coordinates) -- its body tinted."""
    from tracking.location import new_tracker
    x0, y0, x1, y1 = ch["rect"]
    sub = np.ascontiguousarray(frame[y0:y1, x0:x1])
    probe = new_tracker(ch["setup"], ch["bgm"], ch["exclusion"])
    det = {"x": float(x) if ok else np.nan, "y": float(y) if ok else np.nan, "area": np.nan,
           "status": "Tracked" if ok else "Lost"}
    if ok and hasattr(probe, "candidates"):
        d = probe.strength(sub)
        s = float(probe.s)
        px, py = (float(x) + 0.5) * s - 0.5, (float(y) + 0.5) * s - 0.5
        best, bd = None, max(float(probe.body_l), 6.0)
        for cnd in probe.candidates(d):
            dist = float(np.hypot(cnd["cx"] - px, cnd["cy"] - py))
            if dist <= bd:
                best, bd = cnd, dist
        m = np.zeros(d.shape[:2], np.uint8)
        if best is not None:
            bx, by, bw, bh = best["bbox"]
            m[by:by + bh, bx:bx + bw][best["full"]] = 255
        probe.last_mask = m
    elif ok:
        probe.update(sub)
    return det, probe


def _dashed_rect(img, p0, p1, col, th, dash=10, gap=6):
    (x0, y0), (x1, y1) = p0, p1
    for (ax, ay, bx, by) in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        length = max(abs(bx - ax), abs(by - ay))
        if length <= 0:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        t = 0
        while t < length:
            e = min(length, t + dash)
            cv2.line(img, (int(ax + dx * t), int(ay + dy * t)), (int(ax + dx * e), int(ay + dy * e)), col, th,
                     cv2.LINE_AA)
            t += dash + gap
