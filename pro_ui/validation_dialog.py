"""≟ Validate BT freezing against hand scoring (one or many videos).

Each row = one video: a BT Freezing file (.btf) + the hand scoring of the same video (a BT Manual Scoring file
.btms, or the session open on the Event Scoring page) and the hand behaviour that means freezing.  The table shows
per video hand % / BT % / frame agreement / Cohen's kappa; below it the study numbers (Pearson r, ICC, bias and
limits of agreement).  Export writes an Excel workbook + the Bland-Altman and scatter plots for a paper.
"""
import math
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout)

COLS = ["Video", "Hand behaviour", "Hand (%)", "BT (%)", "Frame agreement (%)", "Kappa", "Bin r", "Bins"]


def _f(v, nd=1):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    return "—" if not math.isfinite(v) else f"{v:.{nd}f}"


class _PairDialog(QDialog):
    def __init__(self, parent, inventory):
        super().__init__(parent)
        self.setWindowTitle("Hand behaviour")
        f = QFormLayout(self)
        f.addRow(QLabel("Which hand-scored behaviour is freezing?"))
        self.cmb = QComboBox()
        names = [b.get("name") for b in inventory if (b.get("kind") or "state") == "state"]
        names.sort(key=lambda n: (0 if "freez" in str(n).lower() else 1, str(n)))
        for n in names:
            self.cmb.addItem(str(n))
        f.addRow("Behaviour", self.cmb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


class ValidationDialog(QDialog):
    def __init__(self, parent=None, current_session=None):
        """current_session = dict(video, events, inventory, fps) of the Event Scoring page (optional)."""
        super().__init__(parent)
        self.setWindowTitle("Validate BT freezing against hand scoring")
        self.resize(980, 560)
        self.current = current_session
        self.rows = []
        v = QVBoxLayout(self)
        intro = QLabel("Add each video you scored by hand (Event Scoring, a 'Freezing' state behaviour) together "
                       "with BT's freezing result (.btf) of the same video. Score 20–30 videos for a validation "
                       "study.")
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.tbl = QTableWidget(0, len(COLS))
        self.tbl.setHorizontalHeaderLabels(COLS)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.setSelectionBehavior(QTableWidget.SelectRows)
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(COLS)):
            hh.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        v.addWidget(self.tbl, 1)
        row = QHBoxLayout()
        b_add = QPushButton("＋ Add a video…")
        b_add.clicked.connect(self.add_pair)
        b_rm = QPushButton("Remove")
        b_rm.clicked.connect(self.remove)
        self.b_exp = QPushButton("⤓ Export (Excel + plots)…")
        self.b_exp.clicked.connect(self.export)
        row.addWidget(b_add)
        row.addWidget(b_rm)
        row.addStretch(1)
        row.addWidget(self.b_exp)
        v.addLayout(row)
        self.lbl = QLabel("")
        self.lbl.setWordWrap(True)
        self.lbl.setTextFormat(Qt.RichText)
        v.addWidget(self.lbl)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)
        self._refresh()

    # ------------------------------------------------------------------ rows
    def add_pair(self):
        from tracking import freezing as FZ
        from tracking import project_file as PF
        from analysis.validation import compare_video
        btf, _ = QFileDialog.getOpenFileName(self, "BT's freezing result (.btf) of the video",
                                             str(Path.home()), f"BT Freezing file (*{PF.EXT_FREEZING})")
        if not btf:
            return
        try:
            data = PF.load_freezing(btf)
        except Exception as exc:
            QMessageBox.warning(self, "Validation", f"Could not open the .btf:\n{exc}")
            return
        md = data["md"]
        res = FZ.analyse(md, data.get("settings") or {})
        vname = (data.get("video") or {}).get("name") or Path(btf).stem
        hand = None
        cur = self.current
        if cur and cur.get("events") and Path(str(cur.get("video") or "")).name == vname:
            if QMessageBox.question(self, "Validation", f"Use the hand scoring open on the Event Scoring page "
                                    f"({len(cur['events'])} events) for {vname}?") == QMessageBox.Yes:
                hand = {"events": cur["events"], "inventory": cur.get("inventory") or [], "source": "Event Scoring"}
        if hand is None:
            btms, _ = QFileDialog.getOpenFileName(self, f"Hand scoring of {vname} (.btms)", os.path.dirname(btf),
                                                  f"BT Manual Scoring file (*{PF.EXT_SCORING})")
            if not btms:
                return
            try:
                sd = PF.load_scoring(btms)
            except Exception as exc:
                QMessageBox.warning(self, "Validation", f"Could not open the scoring file:\n{exc}")
                return
            hand = {"events": sd.get("events") or [], "inventory": sd.get("inventory") or [],
                    "source": Path(btms).name}
        dlg = _PairDialog(self, hand["inventory"])
        if dlg.cmb.count() == 0:
            QMessageBox.information(self, "Validation", "The hand scoring has no state behaviour (a behaviour with "
                                                        "a start and a stop, like Freezing).")
            return
        if dlg.exec() != QDialog.Accepted:
            return
        beh = dlg.cmb.currentText()
        r = compare_video(res["freezing"], res["times"], res.get("keep"), hand["events"], beh, md.fps)
        r.update({"Video": vname, "Hand behaviour": beh, "Hand source": hand["source"], "BT file": Path(btf).name,
                  "BT threshold (% body)": res["summary"]["Freezing_threshold_pct"],
                  "BT motion counted": res["summary"]["Motion_counted"],
                  "BT result check": res["summary"].get("Result_check", "")})
        self.rows.append(r)
        self._refresh()

    def remove(self):
        rs = sorted({i.row() for i in self.tbl.selectedIndexes()}, reverse=True)
        for r in rs:
            if 0 <= r < len(self.rows):
                self.rows.pop(r)
        self._refresh()

    def _refresh(self):
        from analysis.validation import across, kappa_words
        self.tbl.setRowCount(len(self.rows))
        for i, r in enumerate(self.rows):
            vals = [r["Video"], r["Hand behaviour"], _f(r["Hand_pct"]), _f(r["BT_pct"]),
                    _f(r["Frame_agreement_pct"]), _f(r["Kappa"], 2), _f(r.get("Bin_r"), 2), str(r.get("Bins", ""))]
            for c, t in enumerate(vals):
                it = QTableWidgetItem(t)
                if c == 0:
                    it.setToolTip(f"{r['BT file']} · hand: {r['Hand source']}")
                self.tbl.setItem(i, c, it)
        self.b_exp.setEnabled(bool(self.rows))
        if not self.rows:
            self.lbl.setText("No video yet.")
            return
        a = across(self.rows)
        k = a["Mean_kappa"]
        txt = (f"<b>{a['Videos']} video(s)</b> · mean frame agreement <b>{_f(a['Mean_frame_agreement_pct'])} %</b>"
               f" · mean kappa <b>{_f(k, 2)}</b> ({kappa_words(k)})")
        if a["Videos"] >= 3:
            txt += (f"<br>Freezing % per video: Pearson r <b>{_f(a['Pearson_r'], 3)}</b> · ICC(A,1) "
                    f"<b>{_f(a['ICC_A1'], 3)}</b> · bias (BT − hand) <b>{_f(a['Bias_BT_minus_hand'])}</b> % points, "
                    f"95 % limits of agreement {_f(a['LoA_low'])} to {_f(a['LoA_high'])}")
        else:
            txt += "<br>Add at least 3 videos for r, ICC and the limits of agreement."
        self.lbl.setText(txt)

    # ------------------------------------------------------------------ export
    def export(self):
        import pandas as pd
        from analysis.validation import across
        fn, _ = QFileDialog.getSaveFileName(self, "Save the validation", str(Path.home() / "BT_validation.xlsx"),
                                            "Excel (*.xlsx)")
        if not fn:
            return
        if not fn.lower().endswith(".xlsx"):
            fn += ".xlsx"
        per = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in self.rows])
        a = across(self.rows)
        study = pd.DataFrame([{"Measure": k.replace("_", " "), "Value": v} for k, v in a.items()])
        bins = pd.DataFrame([{"Video": r["Video"], "Bin start (s)": b[0], "BT (%)": b[1], "Hand (%)": b[2]}
                             for r in self.rows for b in r.get("_bins") or []])
        methods = pd.DataFrame({"Item": [
            "Frame agreement", "Cohen's kappa", "Per-bin values", "Pearson r", "ICC(A,1)", "Bias / limits"],
            "How": [
            "Share of analysed frames where BT and the hand scoring agree (freezing / not freezing).",
            "Agreement beyond chance (two classes), per video.",
            "Freezing % of BT and of the hand scoring in consecutive 10 s bins of each video.",
            "Correlation of the video-level freezing % (BT vs hand).",
            "Two-way random effects, absolute agreement, single rater (Shrout & Fleiss ICC(2,1)).",
            "Bland-Altman: mean of (BT − hand) and mean ± 1.96 SD."]})
        try:
            with pd.ExcelWriter(fn, engine="openpyxl") as w:
                study.to_excel(w, sheet_name="Study", index=False)
                per.to_excel(w, sheet_name="Per video", index=False)
                if not bins.empty:
                    bins.to_excel(w, sheet_name="Bins", index=False)
                methods.to_excel(w, sheet_name="Methods", index=False)
            self._plots(os.path.splitext(fn)[0])
        except Exception as exc:
            QMessageBox.warning(self, "Validation", f"Could not save:\n{exc}")
            return
        QMessageBox.information(self, "Validation", f"Saved:\n{fn}\n+ the Bland-Altman and scatter plots next to it.")

    def _plots(self, base):
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        bt = np.array([r["BT_pct"] for r in self.rows], float)
        hd = np.array([r["Hand_pct"] for r in self.rows], float)
        fig, ax = plt.subplots(figsize=(4.2, 4.0))
        ax.scatter(hd, bt, s=26, color="#1f6feb")
        lim = [0, max(100.0, float(np.nanmax(np.r_[bt, hd])) if len(bt) else 100.0)]
        ax.plot(lim, lim, ls="--", color="#94a3b8", lw=1)
        ax.set_xlabel("Hand scoring: freezing (%)")
        ax.set_ylabel("BT: freezing (%)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        fig.tight_layout()
        fig.savefig(base + "_scatter.png", dpi=200)
        plt.close(fig)
        if len(bt) >= 2:
            m, d = (bt + hd) / 2, bt - hd
            bias, sd = float(np.mean(d)), float(np.std(d, ddof=1))
            fig, ax = plt.subplots(figsize=(4.6, 3.6))
            ax.scatter(m, d, s=26, color="#1f6feb")
            for y, ls in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
                ax.axhline(y, ls=ls, color="#64748b", lw=1)
            ax.set_xlabel("Mean of BT and hand (%)")
            ax.set_ylabel("BT − hand (% points)")
            fig.tight_layout()
            fig.savefig(base + "_bland_altman.png", dpi=200)
            plt.close(fig)
