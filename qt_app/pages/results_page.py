"""
ResultsPage -- the Qt equivalent of TrackerApp._show_results_standard /
_show_results_multi_mouse / _show_results_behavior in gui/main_window.py.
Shown once Start Tracking finishes; MainWindow calls show_standard(),
show_multi_mouse() or show_behavior() with the same result dict shapes the
Tkinter app builds (see _run_standard_flow/_run_multi_mouse_flow/
_run_behavior_flow there). Before any run, or right after "New Analysis",
a plain placeholder is shown instead.
"""

import os

import cv2
import pandas as pd
from PySide6.QtCore import Qt, QUrl, QTimer
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QFrame, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView, QSizePolicy,
    QMessageBox,
)

from qt_app.theme import PALETTE
from qt_app.widgets.results_charts import TrajectoryWidget, EthogramWidget, MOUSE_COLORS, BEHAVIOR_COLORS
from tracking.behavior import behavior_summary_table

# See the identical note in setup_page.py: PALETTE is one fixed dict (no
# runtime theme switch anymore), but call sites below still read
# PALETTE['MUTED'] etc. fresh rather than caching it in a module-level
# constant, for consistency.


def _hint(text):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9.5px;")
    lbl.setWordWrap(True)
    return lbl


def _stat_row(parent_layout, label_text, value_text):
    lbl = QLabel(label_text)
    lbl.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9px;")
    val = QLabel(value_text)
    val.setStyleSheet("font-size: 13px; font-weight: 700;")
    parent_layout.addWidget(lbl)
    parent_layout.addWidget(val)


class ResultsPage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.output_dir = None
        self.setObjectName("pageBody")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(0)

        self.header_row = QHBoxLayout()
        outer.addLayout(self.header_row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {PALETTE['BORDER']};")
        outer.addWidget(sep)
        outer.addSpacing(10)

        self.body_widget = QWidget()
        self.body_widget.setObjectName("pageBody")
        self.body_layout = QHBoxLayout(self.body_widget)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(14)
        outer.addWidget(self.body_widget, 1)

        self._show_placeholder()

    # ------------------------------------------------------------------
    # Shared chrome (mirrors TrackerApp._results_header)
    # ------------------------------------------------------------------

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            else:
                child = item.layout()
                if child is not None:
                    self._clear_layout(child)

    def _build_header(self, title_suffix):
        self._clear_layout(self.header_row)
        title = QLabel("Results: " + title_suffix if title_suffix else "Results")
        title.setStyleSheet("font-size: 15px; font-weight: 700;")
        self.header_row.addWidget(title)
        self.header_row.addStretch()
        if self.output_dir:
            open_btn = QPushButton("Open Results Folder")
            open_btn.setObjectName("toolBtn")
            open_btn.clicked.connect(lambda: self._open_folder(self.output_dir))
            self.header_row.addWidget(open_btn)
            export_btn = QPushButton("Export to Excel")
            export_btn.setObjectName("toolBtn")
            export_btn.clicked.connect(self.on_export_excel)
            self.header_row.addWidget(export_btn)
            # Custom Report builder (Upgrade Plan Tier 2 #10) -- only shown
            # once this run's _report_full_df (set by show_standard/
            # show_multi_mouse/show_behavior, BEFORE they call _build_header)
            # actually has at least one optional stat column to pick from;
            # e.g. a Multi-Mouse run with both Location and Interaction
            # Tracking off has an empty per_mouse_summary and nothing worth
            # building a report from.
            from analysis.custom_report import reportable_columns
            full_df = getattr(self, "_report_full_df", None)
            id_cols = getattr(self, "_report_id_columns", [])
            if full_df is not None and reportable_columns(full_df, id_columns=id_cols):
                report_btn = QPushButton("Custom Report")
                report_btn.setObjectName("toolBtn")
                report_btn.clicked.connect(self.on_custom_report)
                self.header_row.addWidget(report_btn)
        new_btn = QPushButton("New Analysis")
        new_btn.setObjectName("accentBtn")
        new_btn.clicked.connect(self.app.show_setup_page)
        self.header_row.addWidget(new_btn)

    def _open_folder(self, path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def on_export_excel(self):
        """Builds/refreshes a single Results_Export.xlsx in the output
        folder (one sheet per result CSV -- see
        MainWindow.export_results_excel) and reveals it, giving a direct
        Excel export for every analysis type, not just the 'standard'
        pipeline's own internal per-video xlsx report."""
        if not self.output_dir:
            return
        try:
            path = self.app.export_results_excel(self.output_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        QMessageBox.information(self, "Exported", f"Saved:\n{path}")
        self._open_folder(self.output_dir)

    def on_custom_report(self):
        """Opens the Custom Report builder (Upgrade Plan Tier 2 #10) on
        whichever "full" stats table the current show_standard/
        show_multi_mouse/show_behavior call left in self._report_full_df --
        see report_builder_dialog.py for the picker itself and
        analysis/custom_report.py for the (analysis-type-agnostic)
        column-narrowing it saves."""
        full_df = getattr(self, "_report_full_df", None)
        id_cols = getattr(self, "_report_id_columns", [])
        if full_df is None or not self.output_dir:
            return
        from qt_app.dialogs.report_builder_dialog import open_report_builder_dialog
        open_report_builder_dialog(self, full_df, id_cols, self.output_dir)

    def _scroll_column(self, fixed_width):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(fixed_width)
        content = QWidget()
        content.setObjectName("scrollContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        scroll.setWidget(content)
        return scroll, layout

    def _output_files_box(self, parent_layout, output_dir, extra_notes=None):
        from PySide6.QtWidgets import QGroupBox
        box = QGroupBox("Output Files")
        v = QVBoxLayout(box)
        if os.path.isdir(output_dir):
            for f in sorted(os.listdir(output_dir)):
                if os.path.isfile(os.path.join(output_dir, f)):
                    note = (extra_notes or {}).get(f, "")
                    lbl = QLabel(f"- {f}{note}")
                    lbl.setStyleSheet("font-size: 9px;")
                    lbl.setWordWrap(True)
                    v.addWidget(lbl)
        parent_layout.addWidget(box)

    def _reference_frames_gallery(self, parent_layout, preview_paths, thumb_w=150, max_height=230):
        if not preview_paths:
            return
        title = QLabel("Reference Frames")
        title.setStyleSheet("font-size: 10.5px; font-weight: 700;")
        parent_layout.addWidget(title)
        grid = QGridLayout()
        cols = 4
        shown = 0
        for path in preview_paths:
            if not path or not os.path.isfile(path):
                continue
            pix = QPixmap(path)
            if pix.isNull():
                continue
            pix = pix.scaledToWidth(thumb_w, Qt.SmoothTransformation)
            lbl = QLabel()
            lbl.setPixmap(pix)
            lbl.setStyleSheet(f"border: 1px solid {PALETTE['BORDER']};")
            r, c = divmod(shown, cols)
            grid.addWidget(lbl, r, c)
            shown += 1
        if not shown:
            return
        # Capped-height scroll area, not an unbounded grid -- a big
        # Background samples setting can produce far more than the ~6
        # reference frames shown in a quick run, and without a scrollbar
        # that would just push the rest of the Results page down/off
        # screen instead of staying contained.
        content = QWidget()
        content.setObjectName("scrollContent")
        content.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        # Only cap the height (and thus only show a scrollbar) once there's
        # enough thumbnails to need one -- a short gallery just sizes to fit.
        rows = (shown + cols - 1) // cols
        row_h = thumb_w * 3 // 4 + 6  # thumbnails aren't square; +6 for the border
        content_h = rows * row_h
        if content_h > max_height:
            scroll.setFixedHeight(max_height)
        else:
            scroll.setFixedHeight(content_h)
        parent_layout.addWidget(scroll)

    def _show_placeholder(self):
        self.output_dir = None
        self._report_full_df = None
        self._report_id_columns = []
        self._build_header("")
        self._clear_layout(self.body_layout)
        lbl = QLabel("Run Start Tracking from the Setup screen to see results here.")
        lbl.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 11px;")
        self.body_layout.addWidget(lbl)
        self.body_layout.addStretch()

    # ------------------------------------------------------------------
    # Standard Tracking (mirrors TrackerApp._show_results_standard)
    # ------------------------------------------------------------------

    def show_standard(self, summary):
        self.output_dir = summary.get("Output_folder", "")
        # Custom Report builder (Upgrade Plan Tier 2 #10) -- the exact same
        # one-row shape output/excel.py's write_excel_report already turns
        # into the "Summary" sheet (pd.DataFrame([summary])), so picking a
        # subset of its columns here really does replace "the fixed Summary
        # sheet shape" the plan calls out, not a different computation.
        self._report_full_df = pd.DataFrame([summary]) if summary else pd.DataFrame()
        self._report_id_columns = []
        self._build_header(os.path.basename(summary.get("Video", "")))
        self._clear_layout(self.body_layout)

        left_scroll, left = self._scroll_column(300)
        from PySide6.QtWidgets import QGroupBox
        box = QGroupBox("Summary")
        bl = QVBoxLayout(box)
        _stat_row(bl, "Tracking quality", f"{summary.get('Tracking_quality_percent', 0):.1f} %")
        _stat_row(bl, "Total transitions", str(summary.get("Total_transitions", 0)))
        _stat_row(bl, "Total distance (px)", f"{summary.get('Total_distance_pixels', 0):.1f}")
        if summary.get("Stop_Reason"):
            _stat_row(bl, "Stopped because", str(summary["Stop_Reason"]))
        if summary.get("Trajectory_smoothing_applied"):
            _stat_row(bl, "Trajectory smoothing",
                      f"on ({summary.get('Trajectory_smoothing_window_frames', '?')}-frame window)")
        if summary.get("Custom_variables_added"):
            _stat_row(bl, "Custom variables", ", ".join(summary["Custom_variables_added"]))
        if summary.get("Custom_variables_warnings"):
            _stat_row(bl, "Custom variable issues",
                      f"{len(summary['Custom_variables_warnings'])} skipped (see console)")
        for k, v in summary.items():
            # Skip the per-zone full/half/semi 3-point entry-depth breakdown
            # here (Entry_<zone> in the raw CSV) -- like arm-entries/
            # alternation, that's a detailed derived metric meant for the
            # exported CSV/Excel report, not this at-a-glance panel.
            if k.endswith("_time_s") and not k.endswith(("_full_time_s", "_half_time_s", "_semi_time_s")):
                _stat_row(bl, k.replace("_time_s", " time (s)"), f"{v:.1f}")
        left.addWidget(box)
        left.addStretch()

        center = QVBoxLayout()
        header_row = QHBoxLayout()
        rv_lbl = QLabel("Results view")
        rv_lbl.setStyleSheet("font-size: 10.5px; font-weight: 700;")
        header_row.addWidget(rv_lbl)
        header_row.addStretch()
        center.addLayout(header_row)

        views = [("Trajectory", "trajectory.png"), ("Heatmap", "heatmap.png"),
                 ("Zone Occupancy", "zone_occupancy.png")]
        available_views = [(label, fname) for label, fname in views
                            if os.path.exists(os.path.join(self.output_dir, fname))]

        # Trajectory/Heatmap/Zone Occupancy shown side by side, all at
        # once -- no click-to-switch needed to see the heatmap next to the
        # trajectory.
        views_row = QHBoxLayout()
        panel_h = 300
        for label, fname in available_views:
            panel = QVBoxLayout()
            title = QLabel(label)
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(f"font-size: 9.5px; font-weight: 700; color: {PALETTE['MUTED']};")
            panel.addWidget(title)
            pic = QLabel()
            pic.setAlignment(Qt.AlignCenter)
            pic.setStyleSheet(f"background: {PALETTE['SURFACE']}; border: 1px solid {PALETTE['BORDER']}; "
                               "border-radius: 6px;")
            pic.setFixedHeight(panel_h)
            pic.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            pix = QPixmap(os.path.join(self.output_dir, fname))
            if not pix.isNull():
                # Scale to the one dimension we actually know at this point
                # (panel_h, just fixed above) rather than pic.size() -- the
                # label hasn't been through a layout pass yet here, so its
                # width is still whatever a fresh QLabel defaults to, not
                # the real panel width the Expanding size policy will later
                # give it. Scaling against that unknown produced a
                # wrong-sized (often tiny) frozen pixmap that never
                # corrected itself, since a QLabel doesn't auto-rescale a
                # pixmap you set on it. Scaling to the fixed height and
                # centering (AlignCenter above) sidesteps the problem for
                # the common case (never a blank panel while waiting).
                pic.setPixmap(pix.scaledToHeight(panel_h, Qt.SmoothTransformation))

                # Height-only scaling isn't enough on its own though: Zone
                # Occupancy's chart is a fixed 9in WIDE regardless of how
                # many zones it lists (see save_zone_occupancy_chart), so a
                # run with few zones, or one dominant zone whose value
                # label sits far to the right of its bar, renders
                # proportionally wide -- at panel_h alone that can end up
                # wider than this panel's real ~1/3 share of the row, and
                # a QLabel doesn't shrink a pixmap that's bigger than its
                # own rect, it just silently clips it (this is exactly
                # what was cutting off the longest zone's own bar and its
                # "50.1s"-style value label). Once this event loop tick
                # finishes and the real layout has run, pic.width() is the
                # label's true assigned width -- re-scale to fit within
                # THAT too (still preserving aspect ratio), so nothing
                # ever renders wider than its panel. A one-shot re-fit
                # (rather than a resize-aware subclass) is enough since
                # this page's charts don't need to keep re-fitting on
                # every later manual window resize, same trade-off
                # Trajectory/Heatmap already accepted here.
                def _refit_to_panel_width(pic=pic, pix=pix):
                    try:
                        avail_w = pic.width()
                        if avail_w > 0:
                            pic.setPixmap(pix.scaled(
                                avail_w, panel_h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    except RuntimeError:
                        pass  # pic was already deleted (page rebuilt before this fired)

                QTimer.singleShot(0, _refit_to_panel_width)
            panel.addWidget(pic)
            views_row.addLayout(panel)
        center.addLayout(views_row)
        if not available_views:
            empty_lbl = QLabel("No result images were generated for this run.")
            empty_lbl.setAlignment(Qt.AlignCenter)
            empty_lbl.setStyleSheet(f"background: {PALETTE['SURFACE']}; color: {PALETTE['MUTED']}; "
                                      f"border: 1px solid {PALETTE['BORDER']}; border-radius: 6px; padding: 20px;")
            center.addWidget(empty_lbl)

        preview_dir = os.path.join(self.output_dir, "preview")
        preview_paths = []
        if os.path.isdir(preview_dir):
            preview_paths = [os.path.join(preview_dir, f) for f in sorted(os.listdir(preview_dir))
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self._reference_frames_gallery(center, preview_paths)
        center.addStretch()

        right_scroll, right = self._scroll_column(280)
        self._output_files_box(right, self.output_dir)
        right.addStretch()

        self.body_layout.addWidget(left_scroll)
        self.body_layout.addLayout(center, 1)
        self.body_layout.addWidget(right_scroll)

    # ------------------------------------------------------------------
    # Multi-Mouse Tracking (mirrors TrackerApp._show_results_multi_mouse)
    # ------------------------------------------------------------------

    def show_multi_mouse(self, results):
        self.output_dir = results["output_dir"]
        df = results["tracks_df"]
        per_mouse_summary = results.get("per_mouse_summary", {})

        # Custom Report builder (Upgrade Plan Tier 2 #10) -- the exact same
        # shape already written to per_mouse_summary.csv (one row per
        # mouse_id), so picking a subset of columns here narrows that same
        # fixed shape rather than computing something new.
        if per_mouse_summary:
            self._report_full_df = (
                pd.DataFrame.from_dict(per_mouse_summary, orient="index")
                .reset_index().rename(columns={"index": "mouse_id"})
            )
        else:
            self._report_full_df = pd.DataFrame()
        self._report_id_columns = ["mouse_id"]

        self._build_header(os.path.basename(results["video_path"]))
        self._clear_layout(self.body_layout)

        left_scroll, left = self._scroll_column(300)
        from PySide6.QtWidgets import QGroupBox
        box = QGroupBox("Summary")
        bl = QVBoxLayout(box)
        duration_s = df["time_s"].max() if len(df) else 0
        _stat_row(bl, "Duration analyzed", f"{duration_s:.1f} s")
        left.addWidget(box)

        mouse_ids = sorted(df["mouse_id"].unique()) if "mouse_id" in df.columns and len(df) else []

        tbox = QGroupBox("Per-Mouse Tracking")
        tl = QVBoxLayout(tbox)
        for i, mid in enumerate(mouse_ids):
            sub = df[df["mouse_id"] == mid]
            pct_ok = 100 * (sub["status"] == "ok").sum() / len(sub) if len(sub) and "status" in sub.columns else 0
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background: {MOUSE_COLORS[i % len(MOUSE_COLORS)]}; border-radius: 2px;")
            row.addWidget(swatch)
            lbl = QLabel(f" {mid} -- {pct_ok:.1f}% tracked normally")
            lbl.setStyleSheet("font-size: 9px;")
            row.addWidget(lbl)
            row.addStretch()
            tl.addLayout(row)
        left.addWidget(tbox)

        if per_mouse_summary:
            zbox = QGroupBox("Per-Mouse Zones / Distance")
            zl = QVBoxLayout(zbox)
            for i, mid in enumerate(mouse_ids):
                summ = per_mouse_summary.get(mid, {})
                if not summ:
                    continue
                mid_lbl = QLabel(mid)
                mid_lbl.setStyleSheet(f"font-size: 9px; font-weight: 700; color: {MOUSE_COLORS[i % len(MOUSE_COLORS)]};")
                zl.addWidget(mid_lbl)
                for k, v in summ.items():
                    if k.endswith("_time_s"):
                        zone_name = k[: -len("_time_s")]
                        zl.addWidget(_hint(f"  {zone_name}: {v:.1f} s"))
                    elif k.startswith("Total_distance_") and k != "Total_distance_pixels":
                        unit = k[len("Total_distance_"):]
                        zl.addWidget(_hint(f"  distance: {v:.1f} {unit}"))
                    elif k == "Total_distance_pixels" and not any(
                            kk.startswith("Total_distance_") and kk != "Total_distance_pixels" for kk in summ):
                        zl.addWidget(_hint(f"  distance: {v:.1f} px"))
                    elif k.endswith("_bout_count"):
                        obj_name = k[: -len("_bout_count")]
                        zl.addWidget(_hint(f"  near {obj_name}: {int(v)} bout(s)"))
            left.addWidget(zbox)
        left.addStretch()

        center = QVBoxLayout()
        traj_lbl = QLabel("Trajectory (per mouse)")
        traj_lbl.setStyleSheet("font-size: 10.5px; font-weight: 700;")
        center.addWidget(traj_lbl)
        traj_widget = TrajectoryWidget()
        traj_widget.set_data(df)
        center.addWidget(traj_widget)
        self._reference_frames_gallery(center, results.get("preview_paths"))
        center.addStretch()

        right_scroll, right = self._scroll_column(280)
        notes = {os.path.basename(results["annotate_path"]): "  (watch for ID swaps)"}
        self._output_files_box(right, self.output_dir, extra_notes=notes)
        right.addStretch()

        self.body_layout.addWidget(left_scroll)
        self.body_layout.addLayout(center, 1)
        self.body_layout.addWidget(right_scroll)

    # ------------------------------------------------------------------
    # Behavior Classification (mirrors TrackerApp._show_results_behavior)
    # ------------------------------------------------------------------

    def show_behavior(self, results):
        self.output_dir = results["output_dir"]
        bouts_df = results["bouts_df"]
        selected = results["selected_behaviors"]

        # Custom Report builder (Upgrade Plan Tier 2 #10) -- a per-subject
        # pivot of the exact same (subject, behavior) -> count/sum/mean
        # aggregation the "Behavior Bouts" table below already computes,
        # covering all three ways a bouts_df can be produced (rule-based
        # classify_behaviors, the trained-ML path, and manual scoring --
        # see behavior_summary_table's own docstring).
        self._report_full_df = behavior_summary_table(bouts_df, selected)
        self._report_id_columns = ["subject"]

        self._build_header(os.path.basename(results["video_path"]))
        self._clear_layout(self.body_layout)

        labeled_df = results["labeled_df"]
        duration_s = labeled_df["time_s"].max() if len(labeled_df) else 0

        left_scroll, left = self._scroll_column(280)
        from PySide6.QtWidgets import QGroupBox
        box = QGroupBox("Summary")
        bl = QVBoxLayout(box)
        _stat_row(bl, "Duration analyzed", f"{duration_s:.1f} s")
        _stat_row(bl, "Total bouts", str(len(bouts_df)))
        left.addWidget(box)
        notes = {}
        self._output_files_box(left, self.output_dir)
        left.addStretch()

        center = QVBoxLayout()
        eth_lbl = QLabel("Behavior Timeline (Ethogram)")
        eth_lbl.setStyleSheet("font-size: 10.5px; font-weight: 700;")
        center.addWidget(eth_lbl)
        eth_widget = EthogramWidget()
        eth_widget.set_data(bouts_df, duration_s)
        center.addWidget(eth_widget)

        bouts_lbl = QLabel("Behavior Bouts")
        bouts_lbl.setStyleSheet("font-size: 10.5px; font-weight: 700; margin-top: 10px;")
        center.addWidget(bouts_lbl)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Subject", "Behavior", "Count", "Total s", "Mean s"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)

        if len(bouts_df):
            keep = [b for b, v in selected.items() if v] + ["other", "undetermined"]
            shown = bouts_df[bouts_df["behavior"].isin(keep)]
            summary_tbl = shown.groupby(["subject", "behavior"])["duration_s"].agg(
                ["count", "sum", "mean"]).reset_index()
            table.setRowCount(len(summary_tbl))
            for i, (_, r) in enumerate(summary_tbl.iterrows()):
                vals = [r["subject"], r["behavior"], str(int(r["count"])), f"{r['sum']:.1f}", f"{r['mean']:.1f}"]
                for col, val in enumerate(vals):
                    table.setItem(i, col, QTableWidgetItem(val))
            table.setFixedHeight(min(28 * (len(summary_tbl) + 1) + 4, 260))
        else:
            table.setRowCount(0)
            no_bouts = _hint("No bouts detected.")
            center.addWidget(no_bouts)
        center.addWidget(table)

        self._reference_frames_gallery(center, results.get("preview_paths"))
        center.addStretch()

        right_scroll, right = self._scroll_column(200)
        legend_box = QGroupBox("Legend")
        ll = QVBoxLayout(legend_box)
        for name, color in BEHAVIOR_COLORS.items():
            if name in ("other", "undetermined"):
                continue
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(f"background: {color}; border-radius: 2px;")
            row.addWidget(swatch)
            lbl = QLabel(" " + name)
            lbl.setStyleSheet("font-size: 9px;")
            row.addWidget(lbl)
            row.addStretch()
            ll.addLayout(row)
        right.addWidget(legend_box)
        right.addStretch()

        self.body_layout.addWidget(left_scroll)
        self.body_layout.addLayout(center, 1)
        self.body_layout.addWidget(right_scroll)
