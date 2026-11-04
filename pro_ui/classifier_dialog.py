"""🧠 Behaviour classifier: teach BT a behaviour from videos you scored by hand on the Event Scoring page, check how
well it learned it (tested on videos / parts it did not learn from), and let it find the behaviour in new videos.
The logic lives in analysis.classifier."""
import math
import os
import threading
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QHeaderView, QLabel, QListWidget, QListWidgetItem, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QApplication)

from analysis import classifier as C


def _f(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{v:.{nd}f}"


def _state_behaviours(events):
    tot = {}
    for e in events or []:
        if (e.get("kind") or "state") == "state" and e.get("stop") is not None:
            tot[e["behaviour"]] = tot.get(e["behaviour"], 0.0) + float(e["stop"]) - float(e["start"])
    return tot


class _Job:
    """A background job: fn(progress) -> result; polled by a QTimer."""

    def __init__(self, fn):
        self.progress, self.text, self.result, self.error, self.done = 0.0, "", None, None, False
        self.cancelled = False

        def run():
            try:
                self.result = fn(self)
            except Exception as exc:          # shown in the dialog
                import traceback
                traceback.print_exc()
                self.error = str(exc)
            self.done = True
        self.thread = threading.Thread(target=run, daemon=True, name="bt-classifier")
        self.thread.start()


class ClassifierDialog(QDialog):
    def __init__(self, parent=None, current=None, on_events=None):
        """current = dict(video, events, fps) of the Event Scoring page; on_events(events) adds predicted events to
        that page."""
        super().__init__(parent)
        self.setWindowTitle("Behaviour classifier")
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self.resize(980, 760)
        self.current, self.on_events = current, on_events
        self.train_rows = []            # dict(video, events, fps, source)
        self.model = None
        self.job = None
        v = QVBoxLayout(self)
        intro = QLabel("Teach BT a behaviour from your hand scoring: add videos you scored on the Event Scoring page "
                       "(state behaviours: grooming, rearing, freezing ...), tick the behaviours, press Train. BT tests "
                       "itself on videos (or parts) it did not learn from and shows how well it agrees with you. "
                       "Score at least 2–3 videos with every behaviour several times; more videos = better.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        # ---- training videos
        h = QLabel("1.  Videos scored by hand")
        h.setStyleSheet("font-weight:700;color:#0f172a;")
        v.addWidget(h)
        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(["Video", "Scored behaviours", "Scored time", "From"])
        self.tbl.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setMaximumHeight(160)
        v.addWidget(self.tbl)
        r = QHBoxLayout()
        self.btn_cur = QPushButton("＋ The video open here")
        self.btn_cur.setEnabled(bool(current and current.get("video") and current.get("events")))
        self.btn_cur.clicked.connect(self._add_current)
        r.addWidget(self.btn_cur)
        b = QPushButton("＋ BT scoring files (.btms)…")
        b.clicked.connect(self._add_files)
        r.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(self._remove)
        r.addWidget(b)
        r.addStretch(1)
        v.addLayout(r)
        # ---- behaviours + options
        h = QLabel("2.  Behaviours to learn")
        h.setStyleSheet("font-weight:700;color:#0f172a;")
        v.addWidget(h)
        self.lst = QListWidget()
        self.lst.setMaximumHeight(110)
        v.addWidget(self.lst)
        o = QHBoxLayout()
        o.addWidget(QLabel("Trees:"))
        self.sp_trees = QSpinBox()
        self.sp_trees.setRange(10, 200)
        self.sp_trees.setValue(40)
        self.sp_trees.setToolTip("More trees = a little steadier, slower to train (40 is plenty).")
        o.addWidget(self.sp_trees)
        o.addWidget(QLabel("Shortest bout:"))
        self.sp_min = QDoubleSpinBox()
        self.sp_min.setRange(0.0, 30.0)
        self.sp_min.setDecimals(2)
        self.sp_min.setSingleStep(0.1)
        self.sp_min.setValue(0.3)
        self.sp_min.setSuffix(" s")
        self.sp_min.setToolTip("Bouts shorter than this are dropped; gaps shorter than 0.2 s are closed.")
        o.addWidget(self.sp_min)
        o.addStretch(1)
        self.btn_train = QPushButton("Train and test")
        self.btn_train.setProperty("cssClass", "primary")
        self.btn_train.clicked.connect(self._train)
        o.addWidget(self.btn_train)
        v.addLayout(o)
        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setVisible(False)
        v.addWidget(self.bar)
        self.lbl_state = QLabel("")
        self.lbl_state.setWordWrap(True)
        self.lbl_state.setStyleSheet("color:#475569;")
        v.addWidget(self.lbl_state)
        # ---- report
        h = QLabel("3.  How well BT agrees with your scoring (tested on data it did not learn from)")
        h.setStyleSheet("font-weight:700;color:#0f172a;")
        v.addWidget(h)
        self.rep = QTableWidget(0, 9)
        self.rep.setHorizontalHeaderLabels(["Behaviour", "Scored (s)", "F1", "Precision", "Recall", "Kappa",
                                            "Hand (%)", "BT (%)", "Tested by"])
        self.rep.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.rep.verticalHeader().setVisible(False)
        self.rep.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rep.setMaximumHeight(150)
        v.addWidget(self.rep)
        note = QLabel("F1 / kappa above 0.8 = very good agreement, 0.6–0.8 = good, below 0.6 = score more videos or "
                      "check that the behaviour is visible from this camera. Always look at BT's bouts on a few "
                      "videos before you use the numbers.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#64748b;font-size:9pt;")
        v.addWidget(note)
        # ---- use
        h = QLabel("4.  Use the model")
        h.setStyleSheet("font-weight:700;color:#0f172a;")
        v.addWidget(h)
        u = QHBoxLayout()
        self.btn_save = QPushButton("Save model…")
        self.btn_save.clicked.connect(self._save)
        b = QPushButton("Load model…")
        b.clicked.connect(self._load)
        self.btn_here = QPushButton("Find in the video open here")
        self.btn_here.setToolTip("BT's bouts are added to this video's events as '<behaviour> (BT)' — compare them "
                                 "with yours, correct them, export them.")
        self.btn_here.clicked.connect(self._apply_here)
        self.btn_many = QPushButton("Find in videos…")
        self.btn_many.setToolTip("Every chosen video: BT's bouts as CSV and as a BT scoring file (.btms, open it here "
                                 "to check / correct), and one Excel time budget for all of them.")
        self.btn_many.clicked.connect(self._apply_many)
        self.btn_prep = QPushButton("Prepare videos…")
        self.btn_prep.setToolTip("Analyse many videos now (tracking + body motion) and keep the result: training "
                                 "on them and finding behaviours in them later takes seconds. Start it before you "
                                 "go home; it needs no model.")
        self.btn_prep.clicked.connect(self._prepare)
        for w in (self.btn_save, b, self.btn_here, self.btn_many, self.btn_prep):
            u.addWidget(w)
        u.addStretch(1)
        v.addLayout(u)
        self.lbl_model = QLabel("No model yet.")
        self.lbl_model.setWordWrap(True)
        self.lbl_model.setStyleSheet("color:#475569;")
        v.addWidget(self.lbl_model)
        bb = QHBoxLayout()
        bb.addStretch(1)
        b = QPushButton("Close")
        b.clicked.connect(self.reject)
        bb.addWidget(b)
        v.addLayout(bb)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll)
        self._update_buttons()

    # ------------------------------------------------------------------ training data
    def _add_row(self, video, events, fps, source):
        if not video or not os.path.exists(video):
            QMessageBox.warning(self, "Classifier", f"The video is not here:\n{video}")
            return
        tot = _state_behaviours(events)
        if not tot:
            QMessageBox.warning(self, "Classifier", f"{Path(video).name}: no state behaviours (with start and stop) "
                                "were scored.")
            return
        self.train_rows.append({"video": video, "events": list(events), "fps": fps, "source": source})
        self._fill()

    def _add_current(self):
        c = self.current or {}
        self._add_row(c.get("video"), c.get("events"), c.get("fps"), "this session")

    def _add_files(self):
        from tracking.project_file import load_scoring
        fns, _ = QFileDialog.getOpenFileNames(self, "BT scoring files", str(Path.home()), "BT scoring (*.btms)")
        for fn in fns:
            try:
                d = load_scoring(fn)
            except Exception as exc:
                QMessageBox.warning(self, "Classifier", f"{Path(fn).name}: {exc}")
                continue
            vid = (d.get("video") or {}).get("path")
            if vid and not os.path.exists(vid):          # the video moved: look next to the scoring file
                alt = os.path.join(os.path.dirname(fn), os.path.basename(vid))
                vid = alt if os.path.exists(alt) else vid
            self._add_row(vid, d.get("events") or [], (d.get("video") or {}).get("fps"), Path(fn).name)

    def _remove(self):
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rows:
            del self.train_rows[r]
        self._fill()

    def _fill(self):
        self.tbl.setRowCount(0)
        allb = {}
        for d in self.train_rows:
            r = self.tbl.rowCount()
            self.tbl.insertRow(r)
            tot = _state_behaviours(d["events"])
            for b, s in tot.items():
                allb.setdefault(b, [0.0, 0])
                allb[b][0] += s
                allb[b][1] += 1
            for c, t in enumerate((Path(d["video"]).name, ", ".join(sorted(tot)),
                                   f"{sum(tot.values()):.0f} s", d["source"])):
                self.tbl.setItem(r, c, QTableWidgetItem(t))
        checked = {self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())
                   if self.lst.item(i).checkState() == Qt.Checked}
        self.lst.clear()
        for b in sorted(allb):
            s, nv = allb[b]
            it = QListWidgetItem(f"{b}  —  {s:.0f} s scored in {nv} video(s)")
            it.setData(Qt.UserRole, b)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if (b in checked or not checked) else Qt.Unchecked)
            self.lst.addItem(it)
        self._update_buttons()

    def _chosen(self):
        return [self.lst.item(i).data(Qt.UserRole) for i in range(self.lst.count())
                if self.lst.item(i).checkState() == Qt.Checked]

    def _update_buttons(self):
        busy = self.job is not None and not self.job.done
        self.btn_train.setEnabled(not busy and bool(self.train_rows))
        has = self.model is not None and bool(self.model.forests)
        self.btn_save.setEnabled(has and not busy)
        self.btn_here.setEnabled(has and not busy and bool(self.current and self.current.get("video")))
        self.btn_many.setEnabled(has and not busy)
        if hasattr(self, "btn_prep"):
            self.btn_prep.setEnabled(not busy)

    # ------------------------------------------------------------------ jobs
    def _start(self, fn, on_done, text):
        self.job = _Job(fn)
        self.job.text = text
        self._on_done = on_done
        self.bar.setValue(0)
        self.bar.setVisible(True)
        self.lbl_state.setText(text)
        self._update_buttons()
        self.timer.start(200)

    def _poll(self):
        j = self.job
        if j is None:
            return
        self.bar.setValue(int(1000 * max(0.0, min(1.0, j.progress))))
        if j.text:
            self.lbl_state.setText(j.text)
        if j.done:
            self.timer.stop()
            self.bar.setVisible(False)
            self.job = None
            self._update_buttons()
            if j.error:
                self.lbl_state.setText(f"<span style='color:#b91c1c;'>⚠ {j.error}</span>")
                return
            self._on_done(j.result)

    @staticmethod
    def _features(job, rows, k0, k_all, label):
        """Features of every video (BT analysis; cached in each video's results folder)."""
        from tracking.location import results_root
        feats = []
        for k, d in enumerate(rows):
            job.text = f"{label} {k0 + k + 1}/{k_all}: analysing {Path(d['video']).name} (tracking + body motion)…"

            def prog(f, k=k):
                job.progress = (k0 + k + 0.9 * f) / max(1, k_all)
            f = C.video_features(d["video"], progress=prog, cache_dir=os.path.join(results_root(d["video"]),
                                                                                   "BT features"))
            f.update(events=d.get("events") or [], video=d["video"])
            feats.append(f)
        return feats

    def _train(self):
        beh = self._chosen()
        if not beh:
            QMessageBox.warning(self, "Classifier", "Tick at least one behaviour to learn.")
            return
        rows = list(self.train_rows)
        trees, min_bout = self.sp_trees.value(), self.sp_min.value()

        def fn(job):
            feats = self._features(job, rows, 0, len(rows) + 1, "Video")
            m = C.BehaviourModel(beh)
            for b in beh:
                m.post[b]["min_bout_s"] = float(min_bout)
            job.text = "Training and testing…"

            def prog(f):
                job.progress = (len(rows) + f) / (len(rows) + 1)
            m.train(feats, n_trees=trees, progress=prog)
            m.training_videos = [Path(d["video"]).name for d in rows]
            return m
        self._start(fn, self._trained, "Analysing the videos…")

    def _trained(self, m):
        self.model = m
        self._show_report()
        self._update_buttons()
        self.lbl_state.setText(f"✓ Trained on {len(self.train_rows)} video(s). Save the model to use it again.")

    def _show_report(self):
        m = self.model
        self.rep.setRowCount(0)
        for b in m.behaviours:
            r = m.report.get(b, {})
            row = self.rep.rowCount()
            self.rep.insertRow(row)
            vals = [b, _f(r.get("hand_s"), 0), _f(r.get("f1")), _f(r.get("precision")), _f(r.get("recall")),
                    _f(r.get("kappa")), _f(r.get("hand_pct"), 1), _f(r.get("bt_pct"), 1),
                    r.get("error") or r.get("scheme", "")]
            for c, t in enumerate(vals):
                it = QTableWidgetItem(str(t))
                if c == 0 and r.get("top_features"):
                    it.setToolTip("Most useful measures: " + ", ".join(r["top_features"]))
                if c == 2 and isinstance(r.get("f1"), float):
                    it.setForeground(Qt.darkGreen if r["f1"] >= 0.8 else (Qt.darkYellow if r["f1"] >= 0.6 else Qt.red))
                self.rep.setItem(row, c, it)
        info = f"Model: {', '.join(m.forests)}"
        if m.trained:
            info += f" · trained {m.trained}"
        self.lbl_model.setText(info)

    # ------------------------------------------------------------------ files
    def _save(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Save the behaviour model", str(Path.home() / "behaviours.btclf"),
                                            "BT behaviour model (*.btclf)")
        if fn:
            self.model.save(fn)
            self.lbl_state.setText(f"✓ Saved {Path(fn).name}")

    def _load(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Load a behaviour model", str(Path.home()),
                                            "BT behaviour model (*.btclf)")
        if not fn:
            return
        try:
            self.model = C.BehaviourModel.load(fn)
        except Exception as exc:
            QMessageBox.warning(self, "Classifier", str(exc))
            return
        self._show_report()
        self._update_buttons()
        self.lbl_state.setText(f"✓ Loaded {Path(fn).name}")

    # ------------------------------------------------------------------ apply
    def _apply_here(self):
        video = self.current.get("video")
        m = self.model

        def fn(job):
            f = self._features(job, [{"video": video}], 0, 1, "Video")[0]
            job.text = "Finding the behaviours…"
            return m.predict(f)
        self._start(fn, self._applied_here, "Analysing the video…")

    def _applied_here(self, pred):
        events = []
        for b, r in pred.items():
            for e in r["events"]:
                events.append(dict(e, behaviour=f"{b} (BT)"))
        if self.on_events is not None:
            self.on_events(events)
        per = ", ".join(f"{b}: {len(r['events'])} bouts" for b, r in pred.items())
        self.lbl_state.setText(f"✓ {per} — added to the events as '… (BT)'.")

    def _pick_videos(self, title):
        fns, _ = QFileDialog.getOpenFileNames(self, title, str(Path.home()),
                                              "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.mpg *.mpeg *.m4v);;All (*)")
        return fns

    def _apply_many(self):
        fns = self._pick_videos("Videos to analyse")
        if not fns:
            return
        m = self.model

        def fn(job):
            def prog(f):
                job.progress = f

            def log(t):
                job.text = t
            out, status = C.classify_videos(m, fns, progress=prog, log=log)
            bad = [f"{Path(v).name}: {st}" for v, st in status.items() if st != "done"]
            if out is None:
                raise RuntimeError("No video could be analysed. " + "; ".join(bad[:3]))
            return out, bad
        self._start(fn, lambda r: self.lbl_state.setText(
            f"✓ Done. Time budget: {r[0]} — each video's bouts are in its results folder (CSV + .btms)."
            + (f" Not done: {'; '.join(r[1][:3])}" if r[1] else "")), "Analysing the videos…")

    def _prepare(self):
        fns = self._pick_videos("Videos to prepare (analysed now, e.g. overnight)")
        if not fns:
            return

        def fn(job):
            def prog(f):
                job.progress = f

            def log(t):
                job.text = t
            return C.prepare_features(fns, progress=prog, log=log)

        def done(st):
            ok = sum(1 for v in st.values() if v in ("ready", "cached"))
            bad = [f"{Path(v).name}: {x}" for v, x in st.items() if x not in ("ready", "cached")]
            self.lbl_state.setText(f"✓ {ok} of {len(st)} videos prepared: training on them and finding behaviours in "
                                   "them is now quick." + (f" Not done: {'; '.join(bad[:3])}" if bad else ""))
        self._start(fn, done, "Preparing the videos…")

    def reject(self):
        if self.job is not None and not self.job.done:
            if QMessageBox.question(self, "Classifier", "The analysis is still running. Close anyway? (It stops "
                                    "when it has finished the current video.)") != QMessageBox.Yes:
                return
        super().reject()
