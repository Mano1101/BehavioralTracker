"""Freezing results on the Analysis & Results page (BehavioralTracker 1.0.0).

The page shows either a tracking result (its own tabs, unchanged) or a freezing result -- this panel:

  Overview      freezing per epoch · freezing per time bin · motion + bouts + cumulative freezing · 4 key numbers
  Motion Trace  the motion of every frame, freezing shaded, the threshold, the epochs
  Epochs        trial by trial per kind of epoch, before vs during, change; the epoch and "trials by kind" tables
  Bouts         bouts over time, bout durations, intervals between bouts; the bout table
  Time Bins     freezing and motion per bin, cumulative freezing; the bin table
  Frame Review  the video with the moving pixels in red and FREEZING / ACTIVE (original video or the copy
                inside the .btf)
  Statistics    the same Statistics tab as for tracking results (moved here while a freezing result is shown)
"""
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QSlider, QTableWidget, QTableWidgetItem, QTabWidget, QToolButton,
                               QVBoxLayout, QWidget)

from pro_ui.freezing_page import FrameCanvas, FreezeTimeline, _kpi, _num
from pro_ui.ux_controller import FrameView, Player, fmt_clock
from tracking import freezing as FZ


def _table(rows, cols, labels=None):
    t = QTableWidget(len(rows), len(cols))
    t.setHorizontalHeaderLabels(labels or cols)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    for r, row in enumerate(rows):
        for c, k in enumerate(cols):
            v = row.get(k)
            if isinstance(v, (float, np.floating)):
                txt = "" if not np.isfinite(v) else f"{float(v):.4g}"
            elif isinstance(v, list):
                txt = ", ".join(f"{x:.1f}" for x in v)
            else:
                txt = "" if v is None else str(v)
            t.setItem(r, c, QTableWidgetItem(txt))
    hh = t.horizontalHeader()
    hh.setSectionResizeMode(QHeaderView.ResizeToContents)      # the header text counts too
    hh.setStretchLastSection(True)
    return t


def _card(title, widget):
    f = QFrame()
    f.setProperty("cssClass", "card")
    v = QVBoxLayout(f)
    v.setContentsMargins(12, 10, 12, 10)
    lab = QLabel(title)
    lab.setProperty("cssClass", "cardTitle")
    v.addWidget(lab)
    v.addWidget(widget, 1)
    return f, lab


def _plot_label(text="No freezing result yet"):
    lab = QLabel(text)
    lab.setProperty("cssClass", "plotSurface")
    lab.setAlignment(Qt.AlignCenter)
    lab.setMinimumHeight(120)
    return lab, None


class FreezeReview(QWidget):
    """Frame review of a freezing result: original video, or the copy inside the .btf."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.entry = None
        self.player = None
        self.copy = None
        self.cur = 0
        self.playing = False
        self._cache = {}
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 6, 0, 0)
        self.lbl_src = QLabel("")
        self.lbl_src.setProperty("cssClass", "muted")
        v.addWidget(self.lbl_src)
        self.canvas = FrameCanvas()
        self.canvas.placeholder = "Frame review appears here"
        v.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.btn_play = QToolButton()
        self.btn_play.setText("▶")
        self.btn_play.clicked.connect(self.toggle)
        b1 = QToolButton()
        b1.setText("◀|")
        b1.clicked.connect(lambda: self.step(-1))
        b2 = QToolButton()
        b2.setText("|▶")
        b2.clicked.connect(lambda: self.step(1))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.show)
        self.slider.sliderPressed.connect(self.pause)
        self.lbl_t = QLabel("")
        self.lbl_t.setStyleSheet("font-weight:600;")
        self.chk = QCheckBox("Moving pixels")
        self.chk.setChecked(True)
        self.chk.toggled.connect(lambda _b: self.show(self.cur))
        self.cmb_speed = QComboBox()
        for s in (0.5, 1.0, 2.0, 4.0):
            self.cmb_speed.addItem(f"{s:g}×", s)
        self.cmb_speed.setCurrentIndex(1)
        for wd in (self.btn_play, b1, b2):
            row.addWidget(wd)
        row.addWidget(self.slider, 1)
        row.addWidget(self.lbl_t)
        row.addWidget(self.cmb_speed)
        row.addWidget(self.chk)
        v.addLayout(row)
        self.timeline = FreezeTimeline()
        self.timeline.seek.connect(lambda t: (self.pause(), self.show(int(round(t * self.fps())))))
        v.addWidget(self.timeline)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def fps(self):
        return float(self.entry["md"].fps) if self.entry else 30.0

    def close_source(self):
        self.pause()
        if self.player is not None:
            self.player.close()
        if self.copy is not None:
            self.copy.close()
        self.player = self.copy = None
        self._cache = {}

    def load(self, entry):
        self.close_source()
        self.entry = entry
        if entry is None:
            self.canvas.set_bgr(None)
            self.timeline.set_result(None)
            self.lbl_src.setText("")
            return
        md = entry["md"]
        vp = entry.get("video")
        if vp and os.path.exists(vp):
            self.player = Player()
            if not self.player.open(vp):
                self.player = None
            else:
                self.lbl_src.setText(f"Video: {vp}")
        if self.player is None and entry.get("btf"):
            from tracking import project_file as PF
            from tracking.review_video import CopyReader
            try:
                segs = PF.extract_video_copy(entry["btf"])
            except Exception:
                segs = None
            if segs:
                self.copy = CopyReader(segs, (md.frame_w, md.frame_h))
                self.lbl_src.setText("Video: the copy inside the BT Freezing file (the original video is not here)")
        if self.player is None and self.copy is None:
            self.lbl_src.setText("The video is not here and the BT Freezing file holds no copy — only the graphs "
                                 "can be shown.")
        self.slider.setRange(int(md.frames[0]), int(md.frames[-1]))
        self.timeline.set_result(entry["result"])
        self.show(int(md.frames[0]))

    def refresh(self):
        if self.entry:
            self.timeline.set_result(self.entry["result"])
            self.show(self.cur)

    def _frame(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        f = None
        if self.player is not None:
            f = self.player.next() if self.player.pos == idx else self.player.seek(idx)
        elif self.copy is not None:
            f = self.copy.read(idx)
        if len(self._cache) > 6:
            self._cache.pop(next(iter(self._cache)))
        self._cache[idx] = f
        return f

    def show(self, idx):
        if not self.entry:
            return
        md, res = self.entry["md"], self.entry["result"]
        idx = int(min(max(idx, md.frames[0]), md.frames[-1]))
        lag = md.lag
        prev = self._frame(idx - lag) if idx - lag >= md.frames[0] else None
        cur = self._frame(idx)
        self.cur = idx
        i = idx - int(md.frames[0])
        if cur is None:
            self.canvas.set_bgr(None)
        else:
            disp = FZ.review_image(md, cur, prev, i, res, overlay=self.chk.isChecked(), text=False, dots=False)
            self.canvas.set_bgr(disp, FZ.review_status(res, i),
                                FZ.review_body(md, i, res))
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.lbl_t.setText(f"{fmt_clock(idx / md.fps)} / {fmt_clock((md.frames[-1] + 1) / md.fps)}")
        self.timeline.set_position(idx / md.fps)

    def step(self, d):
        self.pause()
        self.show(self.cur + d)

    def toggle(self):
        self.pause() if self.playing else self.play()

    def play(self):
        if not self.entry or (self.player is None and self.copy is None):
            return
        self.playing = True
        self.btn_play.setText("⏸")
        sp = float(self.cmb_speed.currentData() or 1.0)
        self.timer.start(max(5, int(1000.0 / (self.fps() * sp))))

    def pause(self):
        self.playing = False
        self.timer.stop()
        self.btn_play.setText("▶")

    def _tick(self):
        if not self.entry or self.cur >= self.entry["md"].frames[-1]:
            self.pause()
            return
        self.show(self.cur + 1)


class FreezingResultsPanel(QWidget):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.entry = None
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        v.addWidget(self.tabs, 1)
        self.views = {}
        # ---- overview
        ov = QWidget()
        g = QGridLayout(ov)
        g.setContentsMargins(0, 8, 0, 0)
        g.setSpacing(12)
        self.cards = {}
        for col, (key, title, stretch) in enumerate((("overview_epochs", "Freezing per epoch", 1),
                                                     ("overview_bins", "Freezing per time bin", 1),
                                                     ("overview_trace", "Motion & freezing over time", 2))):
            lab, _ = _plot_label()
            card, tl = _card(title, lab)
            self.views[key] = FrameView(lab, self)
            self.cards[key] = tl
            g.addWidget(card, 0, col)
            g.setColumnStretch(col, stretch)
        g.setRowStretch(0, 1)
        kp = QHBoxLayout()
        kp.setSpacing(12)
        self.kpi = {}
        for key, title in (("Freezing_pct", "Freezing"), ("Freezing_time_s", "Freezing time"),
                           ("Freezing_bouts", "Freezing bouts"), ("Latency_first_freeze_s", "Latency to 1st freeze")):
            box, val = _kpi(title)
            box.setMinimumHeight(64)
            self.kpi[key] = val
            kp.addWidget(box)
        g.addLayout(kp, 1, 0, 1, 3)
        self.lbl_checks = QLabel("")
        self.lbl_checks.setWordWrap(True)
        self.lbl_checks.hide()
        g.addWidget(self.lbl_checks, 2, 0, 1, 3)
        self.tabs.addTab(ov, "Overview")
        # ---- motion trace
        mt = QWidget()
        mv = QVBoxLayout(mt)
        mv.setContentsMargins(0, 8, 0, 0)
        lab, _ = _plot_label()
        self.views["motion_trace"] = FrameView(lab, self)
        mv.addWidget(lab, 1)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setProperty("cssClass", "muted")
        mv.addWidget(self.lbl_info)
        self.tabs.addTab(mt, "Motion Trace")
        # ---- epochs / bouts / bins: chart + table(s)
        for key, title in (("epochs_tab", "Epochs"), ("bouts_tab", "Bouts"), ("bins_tab", "Time Bins")):
            w = QWidget()
            wv = QVBoxLayout(w)
            wv.setContentsMargins(0, 8, 0, 0)
            lab, _ = _plot_label()
            self.views[key] = FrameView(lab, self)
            wv.addWidget(lab, 3)
            holder = QTabWidget()
            holder.setDocumentMode(True)
            wv.addWidget(holder, 2)
            setattr(self, f"tables_{key}", holder)
            self.tabs.addTab(w, title)
        # ---- frame review
        self.review = FreezeReview()
        self.tabs.addTab(self.review, "Frame Review")
        self._stats_page = None
        self.clear()

    # statistics tab borrowed from the tracking tabs while this panel is shown
    def take_statistics(self, page):
        if page is not None and self.tabs.indexOf(page) < 0:
            self.tabs.addTab(page, "Statistics")
            self._stats_page = page

    def give_back_statistics(self):
        page = self._stats_page
        if page is not None:
            i = self.tabs.indexOf(page)
            if i >= 0:
                self.tabs.removeTab(i)
        self._stats_page = None
        return page

    def clear(self):
        self.entry = None
        for v in self.views.values():
            v.clear("No freezing result yet — analyse a video on Freezing – Individual")
        for k in self.kpi.values():
            k.setText("—")
        for key in ("epochs_tab", "bouts_tab", "bins_tab"):
            h = getattr(self, f"tables_{key}")
            while h.count():
                h.removeTab(0)
        self.lbl_info.setText("")
        self.lbl_checks.hide()
        self.review.load(None)

    def _charts(self, entry):
        """Chart files of the result (made again into a temporary folder when they are missing)."""
        charts = dict((entry.get("files") or {}).get("charts") or {})
        if charts and all(os.path.exists(p) for p in charts.values()):
            return charts
        from output.freezing_graphs import make_all
        tmp = entry.get("_tmp") or tempfile.mkdtemp(prefix="bt_freeze_charts_")
        entry["_tmp"] = tmp
        charts = make_all(entry["result"], entry["md"].fps, tmp, entry.get("name") or "result")
        if not entry.get("files"):
            entry["files"] = {}
        entry["files"]["charts"] = charts
        return charts

    def load(self, entry, keep_tab=True):
        if entry is None:
            self.clear()
            return
        same = self.entry is not None and self.entry.get("key") == entry.get("key")
        self.entry = entry
        res = entry["result"]
        s = res["summary"]
        charts = self._charts(entry)
        for key, view in self.views.items():
            p = charts.get(key)
            if p and os.path.exists(p):
                view.set_file(p)
            else:
                view.clear("—")
        self.cards["overview_epochs"].setText("Freezing per epoch" if res["epochs"] else "Freezing vs active")
        if res["bins"]:
            bs = float((res.get("settings") or {}).get("bin_s") or (res["bins"][0]["End_s"] - res["bins"][0]["Start_s"]))
            self.cards["overview_bins"].setText("Freezing per minute" if abs(bs - 60) < 0.5 else
                                                f"Freezing per {bs:.4g} s bin")
        checks = [c for c in (res.get("checks") or []) if c["level"] in ("error", "warn")]
        errors = any(c["level"] == "error" for c in checks)
        if checks:
            self.lbl_checks.setText(("⛔ Check before using this result:\n" if errors else "⚠ Check:\n")
                                    + "\n".join("• " + c["text"] for c in checks))
            self.lbl_checks.setStyleSheet("color:%s; font-weight:600; background:%s; border-radius:6px; "
                                          "padding:6px 10px;" % (("#991b1b", "#fee2e2") if errors
                                                                 else ("#92400e", "#fef3c7")))
        self.lbl_checks.setVisible(bool(checks))
        self.kpi["Freezing_pct"].setText(_num(s["Freezing_pct"], 1, " %"))
        self.kpi["Freezing_time_s"].setText(_num(s["Freezing_time_s"], 1, " s"))
        self.kpi["Freezing_bouts"].setText(str(s["Freezing_bouts"]))
        self.kpi["Latency_first_freeze_s"].setText(_num(s["Latency_first_freeze_s"], 1, " s"))
        self.lbl_info.setText(
            f"{entry.get('name', '')}  ·  analysed {fmt_clock(s['Start_time_s'])} – {fmt_clock(s['End_time_s'])}  ·  "
            f"{s['Motion_counted']}  ·  motion cutoff {s['Motion_cutoff']}  ·  threshold "
            f"{s['Freezing_threshold_pct']:g} % of body (= {s.get('Freezing_threshold_px', 0):.0f} px)  ·  "
            f"body {s['Body_length_px']:.0f} px ({s['Body_size_from']})  ·  minimum freeze {s['Min_freeze_s']:g} s  ·  "
            f"mean bout {_num(s['Mean_bout_s'], 2, ' s')}  ·  longest {_num(s['Longest_bout_s'], 1, ' s')}  ·  "
            f"{_num(s.get('Bout_rate_per_min'), 2)} bouts / min")
        from output.freezing_report import EPOCH_EXTRA, GROUP_COLS, PERIOD_COLS
        ep_cols = ["Epoch", "Start_s", "End_s"] + list(PERIOD_COLS) + list(EPOCH_EXTRA)
        ep_lab = ["Epoch", "Start (s)", "End (s)"] + list(PERIOD_COLS.values()) + [v[0] for v in EPOCH_EXTRA.values()]
        self._tables("epochs_tab", [
            ("Epochs", res["epochs"], ep_cols, ep_lab),
            ("Trials by kind", res.get("groups") or [], list(GROUP_COLS) + ["Trial_values"],
             list(GROUP_COLS.values()) + ["Trials (%)"])])
        self._tables("bouts_tab", [
            ("Freezing bouts", res["bouts"], ["Bout", "Start_s", "End_s", "Duration_s"],
             ["Bout", "Start (s)", "End (s)", "Duration (s)"])])
        self._tables("bins_tab", [
            ("Time bins", res["bins"], ["Bin", "Start_s", "End_s"] + list(PERIOD_COLS),
             ["Bin", "Start (s)", "End (s)"] + list(PERIOD_COLS.values()))])
        if same and keep_tab:
            self.review.entry = entry
            self.review.refresh()
        else:
            self.review.load(entry)

    def _tables(self, key, specs):
        h = getattr(self, f"tables_{key}")
        while h.count():
            h.removeTab(0)
        for title, rows, cols, labels in specs:
            if rows:
                h.addTab(_table(rows, cols, labels), title)
        h.setVisible(h.count() > 0)

    def shutdown(self):
        self.review.close_source()
        if self.entry and self.entry.get("_tmp"):
            shutil.rmtree(self.entry["_tmp"], ignore_errors=True)


def result_files_folder(entry):
    """A folder with every result file of the entry: its results folder, or (opened .btf, folder gone) the files
    stored inside the .btf written to a temporary folder."""
    folder = entry.get("folder")
    if folder and os.path.isdir(folder) and any(Path(folder).glob("*_Freezing.xlsx")):
        return folder
    btf = entry.get("btf")
    if btf and btf.get("_results"):
        from tracking import project_file as PF
        tmp = tempfile.mkdtemp(prefix="bt_freeze_files_")
        PF.extract_results(btf, tmp)
        return tmp
    return folder
