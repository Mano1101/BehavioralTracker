"""Frame Review (Analysis page): see where the animal was detected in every frame, correct wrong positions by
clicking on the picture, and commit -- the results are recomputed, the files rewritten, and every change is
written to the audit trail of the protected BT analysis file (.bta)."""
import math
import os

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
                               QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSlider, QTableWidget, QTableWidgetItem, QTextBrowser, QToolButton,
                               QVBoxLayout, QWidget)

STATUS_COLORS = {"Detected": (40, 190, 40), "Interpolated": (0, 165, 255), "Lost": (60, 60, 230),
                 "Corrected": (220, 120, 0), "Not visible": (150, 150, 150), "Hidden": (180, 120, 160)}


def _ranges(frames):
    """[3,4,5,9] -> '3-5, 9'."""
    out, fr = [], sorted(set(int(f) for f in frames))
    i = 0
    while i < len(fr):
        j = i
        while j + 1 < len(fr) and fr[j + 1] == fr[j] + 1:
            j += 1
        out.append(f"{fr[i]}" if i == j else f"{fr[i]}-{fr[j]}")
        i = j + 1
    return ", ".join(out)


class AuditDialog(QDialog):
    def __init__(self, parent, data):
        super().__init__(parent)
        from tracking.project_file import audit_verify
        self.setWindowTitle("Audit trail")
        self.resize(980, 560)
        v = QVBoxLayout(self)
        trail = data.get("audit") or []
        ok, bad = audit_verify(trail)
        vid = data.get("video") or {}
        head = QLabel(f"<b>{data.get('format', 'BT file')}</b> · {os.path.basename(data.get('_path', ''))}<br>"
                      f"Video: {vid.get('name', '')} · {vid.get('width', '')}×{vid.get('height', '')} · "
                      f"{float(vid.get('fps') or 0):.2f} frames/s · created {data.get('created', data.get('saved', ''))} "
                      f"with {data.get('app', '')}")
        head.setWordWrap(True)
        v.addWidget(head)
        state = QLabel("✓ Audit trail intact — no entry was changed or removed." if ok else
                       f"⚠ The audit trail does not match from entry {bad + 1}: it was changed outside BehavioralTracker.")
        state.setStyleSheet(f"font-weight:700;color:{'#15803d' if ok else '#b91c1c'};padding:4px 0;")
        v.addWidget(state)
        t = QTableWidget(len(trail), 5)
        t.setHorizontalHeaderLabels(["#", "Time", "User", "Action", "Details"])
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.NoEditTriggers)
        t.setWordWrap(True)
        for r, e in enumerate(trail):
            det = "; ".join(f"{k}: {v}" for k, v in (e.get("details") or {}).items())
            who = f"{e.get('user', '')} @ {e.get('computer', '')}"
            if e.get("licence"):
                who += f"\nLicence No: {e['licence']}"
            for c, val in enumerate((str(e.get("n", r + 1)), str(e.get("time", "")), who,
                                     str(e.get("action", "")), det)):
                t.setItem(r, c, QTableWidgetItem(val))
        hh = t.horizontalHeader()
        for c in range(4):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.Stretch)
        t.resizeRowsToContents()
        v.addWidget(t, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        v.addWidget(bb)


def _esc(v):
    return str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_size(n):
    n = float(n or 0)
    for unit in ("bytes", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "bytes" else f"{n:.1f} {unit}"
        n /= 1024.0


def _copy_text(data):
    vc = data.get("video_copy")
    if not vc:
        return "no (turn it on in ⚙ Detection settings ▸ Copy of the video in BT files)"
    size = sum(int(sg.get("length") or 0) for sg in vc.get("segments") or [])
    return (f"<span style='color:#15803d'>✓ yes</span> — {vc.get('size', ['?', '?'])[0]} × "
            f"{vc.get('size', ['?', '?'])[1]} px, {vc.get('frames')} frames (every analysed frame), {_fmt_size(size)}")


class ContentsDialog(QDialog):
    """Everything a BT analysis file holds, in plain words (+ save its result files to a folder)."""

    def __init__(self, parent, data):
        super().__init__(parent)
        from analysis.test_metrics import pretty
        from tracking.project_file import audit_verify
        self.data = data
        self.setWindowTitle("What is inside this BT analysis file")
        self.resize(820, 680)
        v = QVBoxLayout(self)
        raw = data.get("raw") or {}
        setup = data.get("setup") or {}
        summ = data.get("summary") or {}
        vid = data.get("video") or {}
        exp = data.get("experiment") or {}
        vp = raw.get("video_path") or vid.get("path") or ""
        found = bool(vp) and os.path.exists(vp)
        fps = float(raw.get("fps") or vid.get("fps") or 30.0)

        def rows(items):
            return "".join(f"<tr><td style='color:#475569;padding:2px 14px 2px 0;'>{_esc(k)}</td>"
                           f"<td style='padding:2px 0;'>{v}</td></tr>" for k, v in items if v not in (None, ""))

        def sec(title, items):
            body = rows(items)
            return f"<h3 style='margin:12px 0 4px 0;color:#0f172a;'>{title}</h3><table>{body}</table>" if body else ""
        ok, bad = audit_verify(data.get("audit") or [])
        lic = data.get("licence") or {}
        html = sec("File", [
            ("File", _esc(os.path.basename(data.get("_path", "")))),
            ("Folder", _esc(os.path.dirname(data.get("_path", "")))),
            ("Created", _esc(data.get("created", ""))), ("Last saved", _esc(data.get("saved", ""))),
            ("Made with", _esc(data.get("app", ""))),
            ("Licence", _esc(f"{lic.get('licensee')} ({lic.get('licence_no')})") if lic else ""),
            ("Audit trail", (f"<span style='color:#15803d'>✓ {len(data.get('audit') or [])} entries, intact</span>"
                             if ok else f"<span style='color:#b91c1c'>⚠ changed from entry {bad + 1}</span>"))])
        dur = (len(raw.get("frames") or []) / fps) if fps else 0
        html += sec("Video", [
            ("Name", _esc(vid.get("name") or os.path.basename(vp))),
            ("Location", _esc(vp) + (" <span style='color:#15803d'>✓ found</span>" if found else
                                     " <span style='color:#b91c1c'>✗ not found — use 🎞 Locate the video</span>")),
            ("Picture", f"{raw.get('frame_w', vid.get('width', ''))} × {raw.get('frame_h', vid.get('height', ''))} px, "
                        f"{fps:.2f} frames/s"),
            ("Frames in the video", _esc(raw.get("total_frames") or vid.get("frames") or "")),
            ("Size", _fmt_size(vid.get("size_bytes")) if vid.get("size_bytes") else ""),
            ("Modified", _esc(vid.get("modified", ""))),
            ("Copy of the video inside", _copy_text(data))])
        design = "; ".join(", ".join(f"{k}: {v}" for k, v in r.items() if v) for r in (exp.get("design") or []))
        html += sec("Experiment", [
            ("Experiment", _esc(exp.get("name") or setup.get("experiment_name") or "")),
            ("Investigator", _esc(exp.get("investigator") or setup.get("investigator") or "")),
            ("Date", _esc(exp.get("date", ""))), ("Animal / subject", _esc(exp.get("subject", ""))),
            ("Experimental design", _esc(design)), ("Description", _esc(exp.get("description", ""))),
            ("Notes", _esc(exp.get("notes", "")))])
        sf = float(setup.get("scale_factor") or 0)
        zones = list((setup.get("roi_points") or {}).keys())
        test = summ.get("Test_type") or setup.get("test_type") or ""
        try:
            from pro_ui.ux_controller import TESTS
            test = TESTS.get(test, (test,))[0]
        except Exception:
            pass
        html += sec("Test, zones and scale", [
            ("Test", _esc(test)), ("Zones", _esc(", ".join(zones)) if zones else "none"),
            ("Masked areas", str(len(setup.get("mask_points") or [])) if setup.get("mask_points") else ""),
            ("Arena crop", "yes (straightened)" if raw.get("need_warp") else "no"),
            ("Scale", f"{1 / sf:.2f} px/cm" if sf else ""),
            ("Analysed time", f"{float(raw.get('start_time') or 0):.2f} s – {float(raw.get('end_time') or 0):.2f} s "
                              f"({dur:.1f} s, {len(raw.get('frames') or [])} frames)"),
            ("Tracking", _esc(raw.get("live_mode") or "video file"))])
        html += sec("Detection settings", [
            ("Engine", _esc(raw.get("algorithm") or setup.get("tracking_algorithm") or "")),
            ("Detection mode", _esc(setup.get("color_mode", ""))), ("Animal colour", _esc(raw.get("polarity", ""))),
            ("Minimum contrast", _esc(setup.get("min_contrast", ""))),
            ("Minimum animal size", _esc(setup.get("min_area", ""))),
            ("Smoothing", f"{setup.get('smooth_window')} frames" if setup.get("smooth_window") else "off"),
            ("Zone entry", _esc(summ.get("Entry_rule", ""))),
            ("Time in a zone", _esc(summ.get("Time_in_zone_rule", ""))),
            ("Covered zones", _esc(summ.get("Covered_zones", "")) +
             (f" (hidden {float(summ.get('Hidden_time_s') or 0):.1f} s)" if summ.get("Covered_zones") else "")),
            ("Background", _esc(raw.get("background_info", "")))])
        keys = ("Tracking_quality_percent", "Total_distance_cm", "Total_distance_pixels", "Center_time_pct",
                "Open_arm_time_pct", "Light_time_pct", "Anxiety_index", "Spontaneous_alternation_pct",
                "Discrimination_index", "Discrimination_ratio", "Total_arm_entries", "Corrected_frames")
        res = []
        for k in keys:
            if k in summ and str(summ[k]) not in ("nan", "None"):
                val = summ[k]
                res.append((pretty(k), f"{val:.3f}" if isinstance(val, float) else _esc(val)))
        files = data.get("_results") or {}
        if files:
            res.append(("All measures", f"in the Excel file inside ({len(summ)} values)"))
        html += sec("Results", res)
        html += sec("Corrections", [("Corrected frames", str(len(data.get("corrections") or [])))])
        if files:
            html += sec(f"Result files inside ({len(files)})",
                        [(name, _fmt_size(len(b))) for name, b in sorted(files.items())])
        tb = QTextBrowser()
        tb.setHtml(f"<div style='font-size:10pt;'>{html}</div>")
        v.addWidget(tb, 1)
        row = QHBoxLayout()
        b = QPushButton("💾 Save the result files to a folder…")
        b.setEnabled(bool(files))
        b.setToolTip("Writes the Excel / CSV / chart files stored in this .bta into a folder of your choice.")
        b.clicked.connect(self._save_files)
        row.addWidget(b)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        row.addWidget(bb)
        v.addLayout(row)

    def _save_files(self):
        from tracking.project_file import extract_results
        d = QFileDialog.getExistingDirectory(self, "Save the result files to", os.path.dirname(self.data.get("_path", "")))
        if not d:
            return
        name = os.path.splitext(os.path.basename(self.data.get("_path", "analysis")))[0]
        target = os.path.join(d, f"{name}_results")
        try:
            n = extract_results(self.data, target)
        except Exception as exc:
            QMessageBox.warning(self, "Save result files", f"Could not write the files:\n{exc}")
            return
        QMessageBox.information(self, "Save result files", f"{n} file(s) saved in:\n{target}")


class FrameReviewTab(QWidget):
    def __init__(self, ctrl):
        super().__init__()
        self.ctrl = ctrl
        self.data = None
        self.cap = None
        self.copy = None
        self.using_copy = False
        self._cap_pos = -1
        self.k = 0                       # index into the frame list
        self.pending = {}                # frame -> correction dict (not committed yet)
        self._order = []                 # pending frames in the order they were made (undo)
        self._build()
        self._set_enabled(False)

    # ------------------------------------------------------------------ UI
    def _build(self):
        from pro_ui.ux_controller import FrameView
        h = QHBoxLayout(self)
        left = QVBoxLayout()
        self.lbl_view = QLabel("Run tracking, or press 📂 Open BT analysis file… (right) to review a saved "
                               "analysis frame by frame.")
        self.lbl_view.setMinimumSize(480, 320)
        self.lbl_view.setStyleSheet("background:#1c212b;color:#8b95a7;border-radius:8px;")
        self.lbl_view.setWordWrap(True)
        left.addWidget(self.lbl_view, 1)
        self.view = FrameView(self.lbl_view, self)
        self.view.pressed.connect(self._clicked)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(lambda v: self.goto(v))
        left.addWidget(self.slider)
        nav = QHBoxLayout()
        for text, d, tip in (("|◀", "first", "First frame"), ("◀◀", -10, "10 frames back"), ("◀", -1, "Previous frame (←)"),
                             ("▶", 1, "Next frame (→)"), ("▶▶", 10, "10 frames forward"), ("▶|", "last", "Last frame")):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _c=False, d=d: self.step(d))
            nav.addWidget(b)
        self.lbl_frame = QLabel("")
        self.lbl_frame.setStyleSheet("font-weight:600;")
        nav.addWidget(self.lbl_frame, 1)
        self.btn_prev_prob = QPushButton("◀ Problem")
        self.btn_prev_prob.setToolTip("Previous lost / filled-in / jumping frame")
        self.btn_prev_prob.clicked.connect(lambda: self.next_problem(-1))
        self.btn_next_prob = QPushButton("Problem ▶")
        self.btn_next_prob.setToolTip("Next lost / filled-in / jumping frame")
        self.btn_next_prob.clicked.connect(lambda: self.next_problem(1))
        nav.addWidget(self.btn_prev_prob)
        nav.addWidget(self.btn_next_prob)
        left.addLayout(nav)
        legend = QLabel("<span style='color:#28be28'>●</span> detected &nbsp; <span style='color:#ffa500'>●</span> filled in "
                        "&nbsp; <span style='color:#e63c3c'>✕</span> lost &nbsp; <span style='color:#0078dc'>●</span> "
                        "corrected &nbsp; <span style='color:#969696'>✕</span> not visible &nbsp; <span style='color:#a0789a'>▒</span> hidden "
                        "in a covered zone &nbsp; <span style='color:#00c8ff'>●</span> head &nbsp; "
                        "<span style='color:#ff3cc8'>●</span> tail base &nbsp; — the line is the last 2 s")
        legend.setStyleSheet("color:#475569;font-size:9pt;")
        legend.setWordWrap(True)
        left.addWidget(legend)
        h.addLayout(left, 1)

        panel = QFrame()
        panel.setFixedWidth(370)
        panel.setObjectName("reviewPanel")
        panel.setStyleSheet("QFrame#reviewPanel{background:#ffffff;border:1px solid #e3e8f0;border-radius:8px;}")
        v = QVBoxLayout(panel)
        v.setContentsMargins(12, 10, 12, 10)
        title = QLabel("Frame review")
        title.setStyleSheet("font-size:13pt;font-weight:700;color:#0f172a;")
        v.addWidget(title)
        top = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open BT analysis file…")
        self.btn_open.setToolTip("Open a saved BT analysis file (.bta) here to review it frame by frame.")
        self.btn_open.clicked.connect(lambda: self.ctrl.open_bta_dialog(review=True))
        self.btn_contents = QPushButton("ℹ What's inside")
        self.btn_contents.setToolTip("Everything this analysis file holds: video, experiment, zones, settings, "
                                     "results, result files and the audit trail.")
        self.btn_contents.clicked.connect(self.show_contents)
        top.addWidget(self.btn_open, 1)
        top.addWidget(self.btn_contents)
        v.addLayout(top)
        self.btn_locate = QPushButton("🎞 Locate the video…")
        self.btn_locate.setToolTip("The video is not where it was analysed (moved, or another computer): show "
                                   "BehavioralTracker where it is now.")
        self.btn_locate.setStyleSheet("QPushButton{color:#b45309;font-weight:600;}")
        self.btn_locate.clicked.connect(self.locate_video)
        self.btn_locate.hide()
        v.addWidget(self.btn_locate)
        self.lbl_copy = QLabel("Showing the video copy stored in this file (the original video is not here).")
        self.lbl_copy.setWordWrap(True)
        self.lbl_copy.setStyleSheet("color:#15803d;font-size:8.5pt;")
        self.lbl_copy.hide()
        v.addWidget(self.lbl_copy)
        self.lbl_info = QLabel("")
        self.lbl_info.setWordWrap(True)
        self.lbl_info.setTextFormat(Qt.RichText)
        v.addWidget(self.lbl_info)
        self.chk_click = QCheckBox("Click on the picture = the animal is here")
        self.chk_click.setChecked(True)
        self.chk_click.setToolTip("Click the centre of the animal's body to correct this frame.")
        v.addWidget(self.chk_click)
        self.chk_advance = QCheckBox("Go to the next frame after a click")
        self.chk_advance.setChecked(True)
        v.addWidget(self.chk_advance)
        g = QGridLayout()
        self.btn_absent = QPushButton("Animal not visible")
        self.btn_absent.setToolTip("The animal cannot be seen in this frame (hidden compartment, out of view).")
        self.btn_absent.clicked.connect(self.mark_absent)
        self.btn_fill = QPushButton("Fill between")
        self.btn_fill.setToolTip("Straight line between the last two frames you corrected.")
        self.btn_fill.clicked.connect(self.fill_between)
        self.btn_undo = QPushButton("↶ Undo")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_keep = QPushButton("Restore detection")
        self.btn_keep.setToolTip("Remove the correction of this frame (also an already committed one).")
        self.btn_keep.clicked.connect(self.restore_frame)
        g.addWidget(self.btn_absent, 0, 0)
        g.addWidget(self.btn_fill, 0, 1)
        g.addWidget(self.btn_undo, 1, 0)
        g.addWidget(self.btn_keep, 1, 1)
        v.addLayout(g)
        v.addWidget(QLabel("<b>Problem frames</b> (click to go there)"))
        self.tbl_prob = QTableWidget(0, 3)
        self.tbl_prob.setHorizontalHeaderLabels(["Frames", "Time (s)", "What"])
        self.tbl_prob.verticalHeader().setVisible(False)
        self.tbl_prob.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_prob.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_prob.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_prob.cellClicked.connect(self._prob_clicked)
        v.addWidget(self.tbl_prob, 2)
        self.lbl_pending = QLabel("No changes yet.")
        self.lbl_pending.setWordWrap(True)
        self.lbl_pending.setStyleSheet("color:#334155;")
        v.addWidget(self.lbl_pending)
        self.le_reason = QLineEdit()
        self.le_reason.setPlaceholderText("Reason for the correction (required)")
        v.addWidget(self.le_reason)
        self.btn_commit = QPushButton("✓  Commit changes")
        self.btn_commit.setMinimumHeight(38)
        self.btn_commit.setStyleSheet("QPushButton{background:#1f6feb;color:white;font-weight:700;border:none;"
                                      "border-radius:6px;} QPushButton:disabled{background:#cbd5e1;}")
        self.btn_commit.clicked.connect(self.commit)
        v.addWidget(self.btn_commit)
        row = QHBoxLayout()
        self.btn_audit = QPushButton("Audit trail…")
        self.btn_audit.clicked.connect(self.show_audit)
        self.btn_discard = QPushButton("Discard changes")
        self.btn_discard.clicked.connect(self.discard)
        row.addWidget(self.btn_audit)
        row.addWidget(self.btn_discard)
        v.addLayout(row)
        h.addWidget(panel)

    def _set_enabled(self, on):
        for w in (self.slider, self.btn_absent, self.btn_fill, self.btn_undo, self.btn_keep, self.btn_commit,
                  self.btn_audit, self.btn_discard, self.btn_prev_prob, self.btn_next_prob, self.le_reason,
                  self.btn_contents):
            w.setEnabled(on)

    # ------------------------------------------------------------------ data
    def load(self, data):
        """data = tracking.project_file.load_analysis(...) result."""
        self.close_video()
        self.data = data
        self.pending, self._order = {}, []
        raw = data["raw"]
        self.frames = list(map(int, raw["frames"]))
        self.fps = float(raw["fps"])
        self.setup = data.get("setup") or {}
        self.roi = {k: [tuple(p) for p in v] for k, v in (self.setup.get("roi_points") or {}).items() if len(v) >= 3}
        m = self.setup.get("matrix")
        self.matrix = np.eye(3, dtype=np.float32) if m is None else np.asarray(m, np.float32)
        self.warp_w, self.warp_h = int(raw["warp_w"]), int(raw["warp_h"])
        self.need_warp = bool(raw.get("need_warp"))
        a = raw.get("animal_area_full")
        self.body_r = max(6.0, 0.6 * math.sqrt(a)) if a else 12.0
        self._cov_masks = None
        if hasattr(self, "_masks"):
            del self._masks                    # zone masks of the previous file
        self._recompute()
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(0, len(self.frames) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_enabled(True)
        vp = raw.get("video_path")
        missing = not (vp and os.path.exists(vp))
        self.btn_locate.setVisible(missing)
        self.using_copy = bool(missing and self._open_copy())
        self.lbl_copy.setVisible(self.using_copy)
        self.k = 0
        self._fill_problems()
        self._update_pending()
        self.goto(0)

    def show_contents(self):
        if self.data is not None:
            ContentsDialog(self, self.data).exec()

    def locate_video(self):
        """Point the analysis file to the video's new place (checked: same picture size and length)."""
        from tracking.project_file import audit_add, save_analysis
        if self.data is None:
            return
        raw = self.data["raw"]
        old = raw.get("video_path") or ""
        name = os.path.basename(old)
        start = os.path.dirname(self.data.get("_path", ""))
        path, _f = QFileDialog.getOpenFileName(self, f"Where is {name}?", os.path.join(start, name),
                                               "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.mpg *.mpeg);;"
                                               "All files (*)")
        if not path:
            return
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened()
        w, h, n = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), \
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if not ok:
            QMessageBox.warning(self, "Locate the video", "This video cannot be opened.")
            return
        want_n = int(raw.get("total_frames") or 0)
        if (w, h) != (int(raw.get("frame_w") or w), int(raw.get("frame_h") or h)) or \
                (want_n and abs(n - want_n) > max(2, 0.01 * want_n)):
            if QMessageBox.question(self, "Locate the video",
                                    f"This video does not look like the analysed one ({w}×{h}, {n} frames; the "
                                    f"analysis was made on {raw.get('frame_w')}×{raw.get('frame_h')}, {want_n} "
                                    "frames).\n\nUse it anyway?") != QMessageBox.Yes:
                return
        raw["video_path"] = path
        (self.data.setdefault("video", {}))["path"] = path
        audit_add(self.data.setdefault("audit", []), "Video location updated",
                  {"old": old, "new": path, "video": os.path.basename(path)})
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            save_analysis(self.data)
        except Exception as exc:
            QMessageBox.warning(self, "Locate the video", f"The file could not be saved:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()
        self.close_video()
        self.btn_locate.hide()
        self.lbl_copy.hide()
        self.using_copy = False
        self.goto(self.k)

    def close_video(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self._cap_pos = None, -1
        if getattr(self, "copy", None) is not None:
            self.copy.close()
        self.copy = None

    def _open_copy(self):
        """Original video missing: use the smaller copy kept inside the .bta (if there is one)."""
        from tracking.project_file import extract_video_copy, has_video_copy
        if self.data is None or not has_video_copy(self.data):
            return False
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            segs = extract_video_copy(self.data)
        except Exception as exc:
            QMessageBox.warning(self, "Frame review", f"The video copy in this file could not be read:\n{exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        if not segs:
            return False
        from tracking.review_video import CopyReader
        self.copy = CopyReader(segs, (self.warp_w, self.warp_h))
        return True

    def clear(self):
        """Back to the empty state (↻ Clear results)."""
        self.close_video()
        self.data = None
        self.frames = []
        self.pending, self._order = {}, []
        self.k = 0
        self.tbl_prob.setRowCount(0)
        self.lbl_info.setText("")
        self.lbl_frame.setText("")
        self.lbl_pending.setText("No changes yet.")
        self.le_reason.clear()
        self.view.clear("Run tracking, or press 📂 Open BT analysis file… (right) to review a saved analysis "
                        "frame by frame.")
        self._set_enabled(False)
        self.btn_locate.hide()
        self.lbl_copy.hide()

    def _committed(self):
        return {int(c["frame"]): c for c in (self.data.get("corrections") or [])}

    def _recompute(self):
        """Effective per-frame positions + status (committed + pending corrections, gaps filled)."""
        from tracking.robust_tracker import gap_fill
        raw = self.data["raw"]
        xs = np.array(raw["xs"], float)
        ys = np.array(raw["ys"], float)
        st = np.array([s == "Tracked" for s in raw["status"]])
        corr = self._committed()
        for f, c in self.pending.items():
            if c.get("kind") == "restore":
                corr.pop(f, None)
            else:
                corr[f] = c
        index = {f: i for i, f in enumerate(self.frames)}
        corrected = np.zeros(len(xs), bool)
        absent = np.zeros(len(xs), bool)
        for f, c in corr.items():
            i = index.get(f)
            if i is None:
                continue
            if c.get("kind") == "absent":
                xs[i] = ys[i] = np.nan
                st[i] = False
                absent[i] = True
            else:
                xs[i], ys[i], st[i] = float(c["x"]), float(c["y"]), True
                corrected[i] = True
        gap = int(round(float(self.setup.get("max_gap_s", 0.5) or 0.0) * self.fps))
        filled = gap_fill(xs, ys, st, gap) & ~absent
        # out of sight next to a covered zone (lid of the dark box) = hidden inside it, not a problem
        hidden = np.zeros(len(xs), bool)
        self.hidden_zone = np.empty(len(xs), dtype=object)
        cov = [n for n in (raw.get("covered_zones") or []) if n in self.roi]
        if cov:
            from tracking.location import build_roi_masks, hidden_in_covered
            if getattr(self, "_cov_masks", None) is None or set(self._cov_masks) != set(cov):
                self._cov_masks = build_roi_masks((self.warp_h, self.warp_w), {n: self.roi[n] for n in cov})
            hidden, self.hidden_zone = hidden_in_covered(xs, ys, st | filled, self._cov_masks,
                                                         float(raw.get("body_len_full") or 0) or 2 * self.body_r)
        status = np.where(corrected, "Corrected", np.where(st, "Detected", np.where(filled, "Interpolated",
                          np.where(hidden, "Hidden", np.where(absent, "Not visible", "Lost")))))
        self.xs, self.ys, self.status = xs, ys, status
        # head / tail-base points (not for frames set by hand)
        self.hx = self.hy = self.tx = self.ty = None
        if False and raw.get("ends"):                 # head / tail points are not shown (body centre only)
            try:
                from tracking.body_points import resolve_head_tail
                self.hx, self.hy, self.tx, self.ty = resolve_head_tail(raw, xs, ys, st & ~corrected)
            except Exception:
                pass

    def _problems(self):
        """[(first_k, last_k, what)] -- lost runs, filled-in runs and sudden jumps."""
        out = []
        st = self.status
        n = len(st)
        i = 0
        while i < n:
            if st[i] in ("Lost", "Interpolated"):
                j = i
                while j + 1 < n and st[j + 1] == st[i]:
                    j += 1
                out.append((i, j, "lost" if st[i] == "Lost" else "filled in"))
                i = j + 1
            else:
                i += 1
        ok = np.isfinite(self.xs) & np.isfinite(self.ys)
        steps = np.full(n, np.nan)
        idx = np.where(ok)[0]
        if len(idx) > 2:
            d = np.hypot(np.diff(self.xs[idx]), np.diff(self.ys[idx]))
            steps[idx[1:]] = d
            med = float(np.nanmedian(d)) if np.isfinite(d).any() else 0.0
            thr = max(4.0 * self.body_r, 8.0 * med)
            for kk in np.where(steps > thr)[0]:
                out.append((int(kk), int(kk), "jump"))
        out.sort()
        return out

    def _fill_problems(self):
        probs = self._problems()
        self._probs = probs
        t = self.tbl_prob
        t.setRowCount(len(probs))
        for r, (a, b, what) in enumerate(probs):
            fa, fb = self.frames[a], self.frames[b]
            for c, val in enumerate((f"{fa}" if a == b else f"{fa}-{fb}", f"{fa / self.fps:.2f}", what)):
                t.setItem(r, c, QTableWidgetItem(val))

    def _prob_clicked(self, r, _c):
        if 0 <= r < len(self._probs):
            self.goto(self._probs[r][0])

    def next_problem(self, direction):
        if not getattr(self, "_probs", None):
            return
        if direction > 0:
            nxt = [a for a, _b, _w in self._probs if a > self.k]
            if nxt:
                self.goto(nxt[0])
        else:
            prv = [a for a, _b, _w in self._probs if a < self.k]
            if prv:
                self.goto(prv[-1])

    # ------------------------------------------------------------------ frames
    def _read(self, frame_no):
        vp = self.data["raw"].get("video_path")
        if self.cap is None and vp and os.path.exists(vp):
            self.cap = cv2.VideoCapture(vp)
            self._cap_pos = -1
        if self.cap is None and self.copy is not None:          # the video copy stored in the .bta
            f = self.copy.read(frame_no)
            if f is not None:
                return f
        if self.cap is None:
            bg = self.data["raw"].get("background_bgr")
            return None if bg is None else bg.copy()
        if frame_no != self._cap_pos + 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_no))
        ok, f = self.cap.read()
        self._cap_pos = frame_no if ok else -1
        if not ok:
            return None
        return cv2.warpPerspective(f, self.matrix, (self.warp_w, self.warp_h)) if self.need_warp else f

    def step(self, d):
        if self.data is None:
            return
        if d == "first":
            self.goto(0)
        elif d == "last":
            self.goto(len(self.frames) - 1)
        else:
            self.goto(self.k + int(d))

    def goto(self, k):
        if self.data is None or not self.frames:
            return
        k = int(min(max(0, k), len(self.frames) - 1))
        self.k = k
        if self.slider.value() != k:
            self.slider.blockSignals(True)
            self.slider.setValue(k)
            self.slider.blockSignals(False)
        f = self._read(self.frames[k])
        if f is None:
            self.view.clear("This frame cannot be read (is the video still in its folder?)")
        else:
            self.view.set_bgr(self._draw(f, k))
        self._update_info()

    def _draw(self, f, k):
        disp = f.copy()
        from tracking.colors import ROI_COLORS
        from tracking.fragment import outline_arrays
        for i, pts in enumerate(self.roi.values()):
            cv2.polylines(disp, outline_arrays(pts), True, ROI_COLORS[i % len(ROI_COLORS)],
                          max(1, int(round(max(disp.shape[:2]) / 1100.0))), cv2.LINE_AA)
        a = max(0, k - int(2 * self.fps))
        seg = [(self.xs[i], self.ys[i]) for i in range(a, k + 1) if np.isfinite(self.xs[i])]
        if len(seg) >= 2:
            cv2.polylines(disp, [np.array(seg, np.int32)], False, (255, 120, 0), 2, cv2.LINE_AA)
        st = str(self.status[k])
        col = STATUS_COLORS.get(st, (255, 255, 255))
        size = max(8, int(min(disp.shape[:2]) * 0.022))
        if np.isfinite(self.xs[k]):
            x, y = int(round(self.xs[k])), int(round(self.ys[k]))
            cv2.circle(disp, (x, y), int(self.body_r), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(disp, (x, y), int(self.body_r) - 2, col, 2, cv2.LINE_AA)
            cv2.drawMarker(disp, (x, y), col, cv2.MARKER_CROSS, size, 2, cv2.LINE_AA)
            if self.hx is not None:
                from tracking.location import draw_head_tail
                draw_head_tail(disp, (self.hx[k], self.hy[k]), (self.tx[k], self.ty[k]), size)
        elif st == "Hidden":
            z = str(self.hidden_zone[k])
            pts = self.roi.get(z)
            if pts:
                c = np.mean(np.asarray(pts, float), axis=0)
                txt = f"hidden in {z}"
                fs = max(0.5, min(disp.shape[:2]) / 700.0)
                (tw, th), _b = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
                org = (int(c[0] - tw / 2), int(c[1] + th / 2))
                cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 4, cv2.LINE_AA)
                cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, 2, cv2.LINE_AA)
        else:
            cx, cy = disp.shape[1] // 2, disp.shape[0] // 2
            cv2.drawMarker(disp, (cx, cy), col, cv2.MARKER_TILTED_CROSS, size * 2, 3, cv2.LINE_AA)
        return disp

    def _update_info(self):
        k = self.k
        fno = self.frames[k]
        st = str(self.status[k])
        sf = (self.setup.get("scale_factor") or 0.0)
        pos = "—"
        zone = "—"
        if np.isfinite(self.xs[k]):
            pos = f"x {self.xs[k]:.1f}, y {self.ys[k]:.1f} px"
            from tracking.location import build_roi_masks
            if not hasattr(self, "_masks"):
                self._masks = build_roi_masks((self.warp_h, self.warp_w), self.roi)
            xi, yi = int(round(self.xs[k])), int(round(self.ys[k]))
            inz = [n for n, m in self._masks.items() if 0 <= yi < m.shape[0] and 0 <= xi < m.shape[1] and m[yi, xi]]
            zone = ", ".join(inz) or "outside the zones"
        mark = " · <b style='color:#d97706'>changed (not committed)</b>" if fno in self.pending else (
            " · corrected earlier" if fno in self._committed() else "")
        col = {"Detected": "#15803d", "Interpolated": "#d97706", "Lost": "#dc2626", "Corrected": "#1d4ed8",
               "Not visible": "#64748b", "Hidden": "#9d4e8f"}.get(st, "#0f172a")
        if st == "Hidden":
            zone = f"{self.hidden_zone[k]} (covered zone — out of sight)"
        from tracking.location import clock_text
        self.lbl_frame.setText(f"{clock_text(fno / self.fps)} · frame {fno}")
        self.lbl_info.setText(f"<b>{clock_text(fno / self.fps)} · frame {fno}</b>{mark}<br>"
                              f"Status: <b style='color:{col}'>{st}</b><br>Position: {pos}<br>"
                              f"Body centre in: {zone}" + (f"<br><span style='color:#64748b'>scale {1 / sf:.2f} px/cm"
                                                           f"</span>" if sf else ""))

    # ------------------------------------------------------------------ corrections
    def _add(self, corr):
        f = int(corr["frame"])
        self.pending[f] = corr
        if f in self._order:
            self._order.remove(f)
        self._order.append(f)
        self._recompute()
        self._update_pending()

    def _clicked(self, fx, fy, right):
        if self.data is None or right or not self.chk_click.isChecked():
            return
        self._add({"frame": self.frames[self.k], "kind": "set", "x": round(float(fx), 2), "y": round(float(fy), 2)})
        if self.chk_advance.isChecked() and self.k < len(self.frames) - 1:
            self.goto(self.k + 1)
        else:
            self.goto(self.k)

    def mark_absent(self):
        if self.data is None:
            return
        self._add({"frame": self.frames[self.k], "kind": "absent"})
        self.goto(self.k)

    def fill_between(self):
        sets = [f for f in reversed(self._order) if self.pending.get(f, {}).get("kind") == "set"][:2]
        if len(sets) < 2:
            QMessageBox.information(self, "Fill between", "Correct two frames first (click the animal in both); "
                                    "the frames between them are then filled with a straight line.")
            return
        a, b = sorted(sets)
        ca, cb = self.pending[a], self.pending[b]
        for f in range(a + 1, b):
            t = (f - a) / float(b - a)
            self.pending[f] = {"frame": f, "kind": "set", "x": round(ca["x"] + (cb["x"] - ca["x"]) * t, 2),
                               "y": round(ca["y"] + (cb["y"] - ca["y"]) * t, 2), "filled": True}
        self._order.append(b)
        self._recompute()
        self._update_pending()
        self.goto(self.k)

    def undo(self):
        if not self._order:
            return
        f = self._order.pop()
        self.pending.pop(f, None)
        self._recompute()
        self._update_pending()
        self.goto(self.k)

    def restore_frame(self):
        if self.data is None:
            return
        f = self.frames[self.k]
        if f in self.pending:
            self.pending.pop(f)
            if f in self._order:
                self._order.remove(f)
        elif f in self._committed():
            self.pending[f] = {"frame": f, "kind": "restore"}
            self._order.append(f)
        self._recompute()
        self._update_pending()
        self.goto(self.k)

    def discard(self):
        if not self.pending:
            return
        if QMessageBox.question(self, "Discard changes", f"Discard {len(self.pending)} uncommitted change(s)?") \
                != QMessageBox.Yes:
            return
        self.pending, self._order = {}, []
        self._recompute()
        self._update_pending()
        self.goto(self.k)

    def _update_pending(self):
        n = len(self.pending)
        done = len(self.data.get("corrections") or []) if self.data else 0
        if n:
            self.lbl_pending.setText(f"<b>{n} change(s) not committed</b> (frames {_ranges(self.pending)})."
                                     f" {done} frame(s) corrected earlier.")
        else:
            self.lbl_pending.setText(f"No changes yet. {done} frame(s) corrected earlier." if done else
                                     "No changes yet.")
        self.btn_commit.setEnabled(bool(n))

    # ------------------------------------------------------------------ commit
    def commit(self):
        if not self.pending or self.data is None:
            return
        reason = self.le_reason.text().strip()
        if not reason:
            QMessageBox.information(self, "Commit changes", "Type the reason for the correction first "
                                    "(it is written to the audit trail).")
            self.le_reason.setFocus()
            return
        self.ctrl.commit_corrections(dict(self.pending), reason)

    def committed(self, data):
        """Called by the controller after a successful commit."""
        self.le_reason.clear()
        k = self.k
        self.load(data)
        self.goto(k)

    def show_audit(self):
        if self.data is not None:
            AuditDialog(self, self.data).exec()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Left:
            self.step(-1)
        elif e.key() == Qt.Key_Right:
            self.step(1)
        else:
            super().keyPressEvent(e)

    def showEvent(self, e):
        super().showEvent(e)
        if self.data is not None:
            QTimer.singleShot(0, lambda: self.goto(self.k))
