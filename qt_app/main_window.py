"""
MainWindow -- the PySide6 rewrite of gui/main_window.py's TrackerApp
(README's "Step 2"). Owns all shared state (video queue, analysis type,
mode, calibration, the persistent _memory dict) and the cross-cutting
actions (add/remove video, Reset handlers, Open/Save Project, Settings,
Help, analysis-type switching, the Setup/Results step tracker, and
eventually Start Tracking). The Setup and Results pages are views that
read this state back via the `app` reference passed into them.

tracking/, analysis/, and output/ are untouched by this migration -- this
file (and setup_page.py / results_page.py / the preview canvas / dialogs
under it) is the only thing being rewritten.
"""

import os
import json
import math
import glob
import tempfile

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QFrame, QStackedWidget, QMessageBox, QFileDialog,
    QApplication,
)

from qt_app import theme
from qt_app.theme import PALETTE, build_stylesheet

APP_SETTINGS_PATH = os.path.expanduser("~/.behavioraltracker_settings.json")
ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "resources", "icon.png")


def load_app_settings():
    """Small persisted app-level prefs (currently just dark mode) -- separate
    from a project's .btproj file and from the in-memory self._memory dict,
    since this should survive across different videos/projects and even a
    fresh app restart, not just a Reset All."""
    try:
        with open(APP_SETTINGS_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_app_settings(data):
    try:
        with open(APP_SETTINGS_PATH, "w") as f:
            json.dump(data, f)
    except Exception:
        pass
from tracking.location import (
    identity_transform, compute_perspective_transform, process_single_video,
    compute_output_dir, compute_zone_interaction_stats,
)
from tracking.two_mouse import (
    track_video, save_preview_frames as save_preview_frames_multi_mouse, choose_color_mode,
)
from tracking.behavior import (
    extract_features, classify_behaviors, save_preview_frames as save_preview_frames_behavior,
)
from tracking.maze_templates import TEMPLATES as MAZE_TEMPLATES
from analysis.calculations import point_distance


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Dark mode: a small persisted app-level prefs file (see
        # load_app_settings), not the project .btproj or the in-memory
        # _memory dict -- restored before any widget is built so the very
        # first paint already uses the right theme, matching theme.PALETTE
        # (mutated in place by theme.set_dark) rather than rebuilding a
        # separate palette object here.
        self._app_settings = load_app_settings()
        self.dark_mode = bool(self._app_settings.get("dark_mode", False))
        theme.set_dark(self.dark_mode)
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.setStyleSheet(build_stylesheet(PALETTE))
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
            if app_instance is not None:
                app_instance.setWindowIcon(QIcon(ICON_PATH))

        # ---- shared state (mirrors TrackerApp.__init__ in gui/main_window.py) ----
        self.videos = []
        self.active_index = None
        self.analysis_type = "standard"
        self.mode = "individual"
        self._memory = {}

        self.pending_use_crop = None
        self.pending_matrix = None
        self.pending_warp_w = None
        self.pending_warp_h = None
        self.pending_crop_corners = None
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        self.pending_scale_factor = None
        self.pending_scale_unit = None

        # Subject database (optional): rows imported from an Excel file via
        # on_import_subjects, e.g. [{"ID": "M1", "Sex": "M", ...}, ...] --
        # schema is whatever columns the researcher's spreadsheet has, not
        # fixed. Videos in the queue can be assigned a subject_id (see
        # on_assign_subject) so a batch run/export can be traced back to
        # which animal each video belongs to -- this is the lightweight
        # "plan sessions ahead" feature (MM's "Subject database + scheduler"
        # ask), not a calendar/full scheduler.
        self.subjects = []

        # Set by Prepare Training Data after a successful build, read back
        # by Train Model as its default dataset folder (mirrors
        # TrackerApp's getattr(self, "_last_ml_dataset_dir", "") pattern).
        self._last_ml_dataset_dir = None

        # Interactive drawing state (mirrors TrackerApp._op / .active_tool):
        # a dict while a crop/mask/zones/objects/distance/template_zones
        # operation is being drawn on the preview canvas, else None.
        self._op = None
        self.active_tool = None

        self.setWindowTitle(PALETTE["APP_BRAND"])
        self.resize(1550, 980)
        self.setMinimumSize(1200, 760)

        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        main_layout.addWidget(self._build_header())
        main_layout.addWidget(self._build_mode_row())
        main_layout.addWidget(self._build_analysis_row())

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {PALETTE['BORDER']};")
        sep.setFixedHeight(1)
        main_layout.addWidget(sep)
        self._top_sep = sep

        # Local imports to avoid a circular import (pages import MainWindow
        # only for type-hinting purposes / not at all).
        from qt_app.pages.setup_page import SetupPage
        from qt_app.pages.results_page import ResultsPage

        self.stack = QStackedWidget()
        self.setup_page = SetupPage(self)
        self.results_page = ResultsPage(self)
        self.stack.addWidget(self.setup_page)
        self.stack.addWidget(self.results_page)
        main_layout.addWidget(self.stack, 1)

        main_layout.addWidget(self._build_footer())

        self._set_step("setup")

    # ------------------------------------------------------------------
    # Header / navigation chrome
    # ------------------------------------------------------------------

    def _build_header(self):
        header = QWidget()
        header.setObjectName("headerBar")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 10, 16, 10)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(10)
        badge = QLabel("B")
        badge.setObjectName("brandBadge")
        badge.setFixedSize(36, 36)
        badge.setAlignment(Qt.AlignCenter)
        if os.path.exists(ICON_PATH):
            # The icon file already has its own rounded gradient background
            # baked in, so fill the badge with it directly instead of
            # layering it on top of the QSS accent-colored square.
            pix = QPixmap(ICON_PATH).scaled(
                36, 36, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            badge.setPixmap(pix)
            badge.setStyleSheet("background: transparent; border-radius: 6px;")
        brand_row.addWidget(badge)
        self.brand_badge = badge

        title_col = QVBoxLayout()
        title_col.setSpacing(1)
        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        title = QLabel(PALETTE["APP_BRAND"])
        title.setObjectName("brandTitle")
        version = QLabel(PALETTE["APP_VERSION"])
        version.setObjectName("brandVersion")
        title_row.addWidget(title)
        title_row.addWidget(version)
        title_row.addStretch()
        subtitle = QLabel("Track • Analyze • Understand Behavior")
        subtitle.setObjectName("brandSubtitle")
        title_col.addLayout(title_row)
        title_col.addWidget(subtitle)
        brand_row.addLayout(title_col)

        layout.addLayout(brand_row)
        layout.addStretch()

        actions = QHBoxLayout()
        actions.setSpacing(8)
        open_btn = QPushButton("Open Project")
        open_btn.setObjectName("headerBtn")
        open_btn.clicked.connect(self.on_open_project)
        save_btn = QPushButton("Save Project")
        save_btn.setObjectName("headerBtn")
        save_btn.clicked.connect(self.on_save_project)
        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("headerBtn")
        settings_btn.clicked.connect(self.on_open_settings)
        help_btn = QPushButton("Help")
        help_btn.setObjectName("headerBtn")
        help_btn.clicked.connect(self.on_open_help)
        self.theme_toggle_btn = QPushButton("☀ Light" if self.dark_mode else "🌙 Dark")
        self.theme_toggle_btn.setObjectName("themeToggleBtn")
        self.theme_toggle_btn.setToolTip("Switch between light and dark mode")
        self.theme_toggle_btn.clicked.connect(self.on_toggle_theme)
        reset_btn = QPushButton("Reset All")
        reset_btn.setObjectName("resetAllBtn")
        reset_btn.clicked.connect(self.on_reset_all)
        for b in (open_btn, save_btn, settings_btn, self.theme_toggle_btn, help_btn, reset_btn):
            actions.addWidget(b)
        layout.addLayout(actions)

        return header

    def _build_mode_row(self):
        row = QWidget()
        row.setObjectName("modeRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(16, 8, 16, 8)

        self.individual_radio = QRadioButton("Individual")
        self.batch_radio = QRadioButton("Batch")
        self.individual_radio.setChecked(True)
        self.mode_group = QButtonGroup(row)
        self.mode_group.addButton(self.individual_radio)
        self.mode_group.addButton(self.batch_radio)
        self.individual_radio.toggled.connect(self._on_mode_change)
        layout.addWidget(self.individual_radio)
        layout.addWidget(self.batch_radio)
        layout.addStretch()

        self.status_label = QLabel("Idle.")
        self.status_label.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 11px;")
        layout.addWidget(self.status_label)

        return row

    def _build_analysis_row(self):
        row = QWidget()
        row.setObjectName("analysisRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(16, 8, 16, 8)

        label = QLabel("Analysis Type:")
        label.setStyleSheet("font-weight: 700; font-size: 11px;")
        layout.addWidget(label)

        self.analysis_type_buttons = {}
        at_defs = [
            ("standard", "Standard Tracking", "Single mouse - zones & distance"),
            ("multi_mouse", "Multi-Mouse Tracking", "2-3 mice, ID-matched tracks"),
            ("behavior", "Behavior Classification", "Grooming / rearing / locomotion"),
        ]
        for val, text, subtitle in at_defs:
            cell = QVBoxLayout()
            cell.setSpacing(2)
            btn = QPushButton(text)
            btn.setObjectName("analysisCardActive" if val == "standard" else "analysisCard")
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked=False, v=val: self._set_analysis_type(v))
            sub = QLabel(subtitle)
            sub.setObjectName("analysisSubtitle")
            cell.addWidget(btn)
            cell.addWidget(sub)
            wrapper = QWidget()
            wrapper.setLayout(cell)
            layout.addWidget(wrapper)
            self.analysis_type_buttons[val] = btn

        layout.addStretch()

        self.step_labels = {}
        self._step_arrows = []
        step_defs = [("setup", "1  Setup"), ("results", "2  Results")]
        for i, (key, text) in enumerate(step_defs):
            if i > 0:
                arrow = QLabel("→")
                arrow.setStyleSheet(f"color: {PALETTE['MUTED']};")
                layout.addWidget(arrow)
                self._step_arrows.append(arrow)
            lbl = QLabel(text)
            lbl.setObjectName("stepLabelActive" if key == "setup" else "stepLabel")
            layout.addWidget(lbl)
            self.step_labels[key] = lbl

        return row

    def _build_footer(self):
        footer = QWidget()
        footer.setObjectName("footerBar")
        footer.setFixedHeight(26)
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(16, 0, 16, 0)
        copyright_lbl = QLabel(f"© 2025 {PALETTE['APP_BRAND']}")
        copyright_lbl.setObjectName("footerText")
        version_lbl = QLabel(PALETTE["APP_VERSION"])
        version_lbl.setObjectName("footerVersion")
        layout.addWidget(copyright_lbl)
        layout.addStretch()
        layout.addWidget(version_lbl)
        self._footer_copyright = copyright_lbl
        return footer

    def on_toggle_theme(self):
        """Flip light/dark. The bulk of the chrome (header, buttons, cards,
        the Setup/Results pages' QGroupBoxes, etc.) is styled purely through
        object-name QSS selectors in theme.build_stylesheet, so re-applying
        the app-wide stylesheet after theme.set_dark() repaints all of that
        automatically. A handful of labels set an explicit color inline at
        construction time (status_label, the step-arrow labels, the top
        separator) and need re-styling by hand here; the Setup page's own
        inline-styled hint labels are refreshed by rebuilding it."""
        self.dark_mode = not self.dark_mode
        theme.set_dark(self.dark_mode)
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.setStyleSheet(build_stylesheet(PALETTE))
        self.theme_toggle_btn.setText("☀ Light" if self.dark_mode else "🌙 Dark")
        self._top_sep.setStyleSheet(f"color: {PALETTE['BORDER']};")
        self.status_label.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 11px;")
        for arrow in getattr(self, "_step_arrows", []):
            arrow.setStyleSheet(f"color: {PALETTE['MUTED']};")
        try:
            # resave=True (the default) so whatever the researcher has
            # already typed into the current fields (ROI names, thresholds,
            # etc.) is preserved into _memory before the page is torn down
            # and rebuilt with the new theme's colors -- resave=False is
            # only for right after Open Project/Reset All, where _memory
            # was JUST written fresh and resaving would overwrite it with
            # stale on-screen values (see rebuild()'s own docstring).
            self.setup_page.rebuild()
        except Exception:
            pass
        self._app_settings["dark_mode"] = self.dark_mode
        save_app_settings(self._app_settings)

    # ------------------------------------------------------------------
    # Step tracker + analysis-type switching
    # ------------------------------------------------------------------

    def _set_step(self, active):
        for key, lbl in self.step_labels.items():
            lbl.setObjectName("stepLabelActive" if key == active else "stepLabel")
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def _restyle_analysis_buttons(self):
        for val, btn in self.analysis_type_buttons.items():
            btn.setObjectName("analysisCardActive" if val == self.analysis_type else "analysisCard")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _set_analysis_type(self, value):
        if value == self.analysis_type:
            return
        self.setup_page.save_all_to_memory(include_calibration=True)
        self.analysis_type = value
        self._restyle_analysis_buttons()
        self._reset_calibration_state()
        self.setup_page.rebuild()
        self.show_setup_page()

    def show_setup_page(self):
        self.stack.setCurrentWidget(self.setup_page)
        self._set_step("setup")

    def show_results_page(self):
        self.stack.setCurrentWidget(self.results_page)
        self._set_step("results")

    # ------------------------------------------------------------------
    # Mode (Individual/Batch)
    # ------------------------------------------------------------------

    def get_mode(self):
        return "individual" if self.individual_radio.isChecked() else "batch"

    def _on_mode_change(self, checked):
        if not checked:
            return
        self.mode = "individual"
        if len(self.videos) > 1:
            keep = self.active_index if self.active_index is not None else 0
            kept = self.videos[keep]
            removed = len(self.videos) - 1
            self.videos = [kept]
            self.active_index = 0
            self.setup_page.refresh_video_list()
            QMessageBox.information(
                self, "Individual mode",
                f"Individual mode allows only 1 video -- kept 1, removed {removed} other(s) from the queue."
            )

    def _on_batch_selected(self):
        self.mode = "batch"

    # ------------------------------------------------------------------
    # Video queue (shared across all analysis types)
    # ------------------------------------------------------------------

    def _probe_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {"path": path, "fps": fps, "frame_count": frame_count, "duration": duration,
                "width": width, "height": height}

    def on_add_videos(self):
        if self.get_mode() == "individual" and self.videos:
            QMessageBox.warning(
                self, "Individual mode",
                "Individual mode allows only 1 video. Remove the current one first "
                "(click its x), or switch to Batch mode."
            )
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select video(s)", "", "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*.*)"
        )
        if not paths:
            return

        if self.get_mode() == "individual" and len(paths) > 1:
            QMessageBox.information(self, "Individual mode",
                                     "Individual mode allows only 1 video -- using the first one selected.")
            paths = paths[:1]

        for path in paths:
            entry = self._probe_video(path)
            if entry is None:
                continue
            self.videos.append(entry)

        if self.active_index is None and self.videos:
            self.active_index = 0

        self.setup_page.refresh_video_list()

    def on_remove_video(self, index):
        del self.videos[index]
        if not self.videos:
            self.active_index = None
        elif self.active_index is not None and self.active_index >= len(self.videos):
            self.active_index = len(self.videos) - 1
        self.setup_page.refresh_video_list()

    def on_select_video_row(self, index):
        # Same rule as the Tkinter app: switching the active video does NOT
        # clear calibration (crop/zones/objects) -- only an explicit Reset
        # does that. Otherwise redrawing zones for every video in a batch
        # queue would defeat the point of a queue.
        self.active_index = index
        self.setup_page.refresh_video_list()

    def on_align_video(self, index):
        """Batch mode's per-video zone alignment (MM asked for this after
        his EPM alignment video): starts from the SAME zone shape already
        defined on the active/template video (or this video's own
        previously-saved alignment, if it has one) and lets the researcher
        drag corners to fit THIS specific video's framing -- reusing the
        drag-only template_zones op machinery entirely (see
        on_canvas_press's template_zones branch), just tagged with
        `_align_target_index` so finish_op saves the result into this
        video's own roi_points_override instead of overwriting the shared
        pending_roi_points template used for every other video."""
        if not self.pending_roi_points:
            QMessageBox.critical(
                self, "Set up zones first",
                "Define zones once (Draw Zones or a Maze Template) on the active video before "
                "aligning individual videos in the queue -- each video then starts from that "
                "same shape and you just nudge its corners to fit."
            )
            return
        self.active_index = index
        video = self.videos[index]
        starting = video.get("roi_points_override") or self.pending_roi_points
        base_frame = self.get_warped_active_frame()
        if base_frame is None:
            QMessageBox.critical(self, "No video", "Could not read a frame from this video.")
            return
        self.set_active_tool("template_zones")
        self._op = {
            "kind": "template_zones", "drag": None, "_base_frame": base_frame,
            "regions": {n: list(pts) for n, pts in starting.items()},
            "_align_target_index": index,
        }
        self.setup_page.show_op_bar("template_zones")
        self.setup_page.refresh_canvas()

    # ------------------------------------------------------------------
    # Subject database (optional Excel import) -- see self.subjects' comment
    # in __init__ for scope.
    # ------------------------------------------------------------------

    def _subject_id_key(self, row):
        """Which column of an imported subject row to treat as the ID --
        prefers a column literally named id/subject/animal (case
        insensitive), else just the first column, since a researcher's
        spreadsheet layout can't be assumed."""
        if not row:
            return None
        for key in row.keys():
            if key.strip().lower() in ("id", "subject", "subject_id", "animal", "animal_id"):
                return key
        return next(iter(row.keys()), None)

    def subject_ids(self):
        out = []
        for row in self.subjects:
            key = self._subject_id_key(row)
            if key and row.get(key):
                out.append(row[key])
        return out

    def on_import_subjects(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Subject List", "", "Excel files (*.xlsx *.xls)")
        if not path:
            return
        try:
            df = pd.read_excel(path)
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))
            return
        if df.empty:
            QMessageBox.warning(self, "Empty file", "That Excel file has no rows.")
            return
        df = df.dropna(how="all")
        clean = []
        for row in df.to_dict("records"):
            clean_row = {str(k).strip(): ("" if pd.isna(v) else str(v).strip()) for k, v in row.items()}
            if any(clean_row.values()):
                clean.append(clean_row)
        if not clean:
            QMessageBox.warning(self, "Empty file", "That Excel file has no usable rows.")
            return
        self.subjects = clean
        self.status_label.setText(f"Imported {len(clean)} subject(s) from {os.path.basename(path)}.")
        self.setup_page.refresh_subjects_panel()
        self.setup_page.refresh_video_list()

    def on_clear_subjects(self):
        self.subjects = []
        for v in self.videos:
            v.pop("subject_id", None)
        self.setup_page.refresh_subjects_panel()
        self.setup_page.refresh_video_list()

    def on_assign_subject(self, video_index, subject_id):
        if not (0 <= video_index < len(self.videos)):
            return
        if subject_id:
            self.videos[video_index]["subject_id"] = subject_id
        else:
            self.videos[video_index].pop("subject_id", None)
        self.setup_page.refresh_subjects_panel()

    # ------------------------------------------------------------------
    # Frame access + calibration tools
    # ------------------------------------------------------------------

    def get_active_raw_frame(self):
        if self.active_index is None:
            return None
        video = self.videos[self.active_index]
        cap = cv2.VideoCapture(video["path"])
        if not cap.isOpened():
            return None
        try:
            start_time = float(self.setup_page.start_entry.text() or 0)
        except (ValueError, RuntimeError, AttributeError):
            start_time = 0
        start_frame = int(max(0, start_time) * (video["fps"] or 30))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def get_warped_active_frame(self):
        raw = self.get_active_raw_frame()
        if raw is None or self.pending_matrix is None:
            return raw
        return cv2.warpPerspective(raw, self.pending_matrix, (self.pending_warp_w, self.pending_warp_h))

    def set_active_tool(self, key):
        """Style-only: highlights the given toolbar button. Mirrors
        TrackerApp._set_active_tool -- does NOT start an operation (see
        start_op for that); used for 'nocrop'/'maze', which have their own
        one-shot handlers rather than an interactive _op."""
        self.active_tool = key
        self.setup_page._restyle_tool_buttons(key)

    def on_tool_no_crop(self):
        raw = self.get_active_raw_frame()
        if raw is None:
            QMessageBox.critical(self, "No video", "Add and select a video first.")
            return
        self.set_active_tool("nocrop")
        h, w = raw.shape[:2]
        self.pending_crop_corners = None
        self.pending_use_crop = False
        self.pending_matrix, self.pending_warp_w, self.pending_warp_h = identity_transform(w, h)
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        self.setup_page.refresh_canvas()

    def quick_setup_template(self, key):
        """One click from an arena template straight to its zones, using
        the template's own default sizes. Needs the arena already set
        (Crop Arena or No Crop). Generates the zones then hands off to the
        SAME interactive template_zones op the Maze Template dialog uses
        (start_template_zone_op), so the researcher can drag a generated
        zone's corners to fit the real maze on the canvas, and use the
        per-zone button in the op bar to rename one (e.g. swap which arm
        is Open/Closed) before Finish commits it -- rather than silently
        committing whatever the template guessed."""
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena needed", "Use 'Crop Arena' or 'No Crop' first.")
            return
        try:
            params = {pkey: default for pkey, _label, default, _kind in MAZE_TEMPLATES[key]["params"]}
            zones = MAZE_TEMPLATES[key]["generate"](self.pending_warp_w, self.pending_warp_h, params)
        except Exception as exc:
            QMessageBox.critical(self, "Could not generate zones", str(exc))
            return
        self.start_template_zone_op(key, zones)

    # ------------------------------------------------------------------
    # Embedded operations: Crop / Mask / Zones / Objects / Distance
    # (mirrors TrackerApp._start_op/_cancel_op/_finish_op and the canvas
    # mouse-event handlers in gui/main_window.py). The _op dict's points
    # are always stored in FRAME-pixel space (the same space as
    # pending_roi_points etc.) -- PreviewCanvas converts widget clicks to
    # frame coordinates before calling on_canvas_press/drag/release.
    # ------------------------------------------------------------------

    def start_op(self, kind):
        if kind == "crop":
            base_frame = self.get_active_raw_frame()
            if base_frame is None:
                QMessageBox.critical(self, "No video", "Add and select a video first.")
                return
        else:
            if self.pending_matrix is None:
                QMessageBox.critical(self, "Arena needed", "Use 'Crop Arena' or 'No Crop' first.")
                return
            base_frame = self.get_warped_active_frame()
            if base_frame is None:
                QMessageBox.critical(self, "No video", "Add and select a video first.")
                return

        names = []
        if kind == "zones":
            names = [n.strip() for n in self.setup_page.roi_names_entry.text().split(",") if n.strip()]
            if not names:
                QMessageBox.critical(self, "Zone names needed", "Enter zone names first (e.g. Light,Dark).")
                return
        if kind == "objects":
            names = [n.strip() for n in self.setup_page.object_names_entry.text().split(",") if n.strip()]
            if not names:
                QMessageBox.critical(self, "Object names needed", "Enter object names first in the left panel.")
                return

        self.set_active_tool(kind)

        op = {"kind": kind, "drag": None, "_base_frame": base_frame}
        if kind == "crop":
            saved = self.pending_crop_corners
            op["shapes"] = [list(saved)] if saved and len(saved) == 4 else [[]]
        elif kind == "mask":
            op["shapes"] = [list(p) for p in self.pending_mask_points] if self.pending_mask_points else [[]]
        elif kind in ("zones", "objects"):
            existing = self.pending_roi_points if kind == "zones" else self.pending_object_points
            op["regions"] = {n: list(existing.get(n, [])) for n in names}
            op["active_region"] = names[0]
            # Shape-drawing mode (Freehand/Rectangle/Ellipse/Line) + optional
            # snap-to-grid, both reset to defaults each time an op starts --
            # see op_set_draw_mode/op_toggle_snap and on_canvas_press/drag.
            op["draw_mode"] = "freehand"
            op["snap"] = False
            op["grid_spacing"] = 20
        elif kind == "distance":
            op["points"] = []

        self._op = op
        self.setup_page.show_op_bar(kind)
        self.setup_page.refresh_canvas()

    def cancel_op(self):
        self._op = None
        self.set_active_tool(None)
        self.setup_page.hide_op_bar()
        self.setup_page.refresh_canvas()

    def finish_op(self):
        op = self._op
        if op is None:
            return

        if op["kind"] == "crop":
            if len(op["shapes"][0]) != 4:
                QMessageBox.critical(self, "Not done", "Click all 4 corners first.")
                return
            matrix, w, h = compute_perspective_transform(op["shapes"][0])
            self.pending_crop_corners = list(op["shapes"][0])
            self.pending_use_crop = True
            self.pending_matrix, self.pending_warp_w, self.pending_warp_h = matrix, w, h
            self.pending_roi_points = {}
            self.pending_object_points = {}
            self.pending_mask_points = []

        elif op["kind"] == "mask":
            self.pending_mask_points = [s for s in op["shapes"] if len(s) >= 3]

        elif op["kind"] in ("zones", "objects"):
            incomplete = [n for n, pts in op["regions"].items() if len(pts) < 3]
            if incomplete:
                noun = "zones" if op["kind"] == "zones" else "objects"
                QMessageBox.critical(self, "Not done", f"These {noun} still need 3+ points: {', '.join(incomplete)}")
                return
            if op["kind"] == "zones":
                self.pending_roi_points = dict(op["regions"])
            else:
                self.pending_object_points = dict(op["regions"])

        elif op["kind"] == "distance":
            if len(op["points"]) != 2:
                QMessageBox.critical(self, "Not done", "Click 2 points first.")
                return
            px_distance = point_distance(op["points"][0], op["points"][1])
            try:
                true_distance = float(self.setup_page.real_distance_entry.text())
            except (ValueError, RuntimeError, AttributeError):
                QMessageBox.critical(self, "Invalid distance", "Enter a valid real-world distance number first.")
                return
            if px_distance > 0:
                self.pending_scale_factor = true_distance / px_distance
                self.pending_scale_unit = self.setup_page.units_entry.text().strip() or "cm"

        elif op["kind"] == "template_zones":
            target_idx = op.get("_align_target_index")
            if target_idx is not None:
                # Per-video batch alignment (on_align_video) -- save into
                # THIS video's own override, and leave the shared
                # pending_roi_points template (used by every other video)
                # untouched.
                self.videos[target_idx]["roi_points_override"] = dict(op["regions"])
                vname = os.path.basename(self.videos[target_idx]["path"])
                self.status_label.setText(f"Zone alignment saved for '{vname}'.")
                self.setup_page.refresh_video_list()
            else:
                self.pending_roi_points = dict(op["regions"])
                # roi_names_entry is read elsewhere (CSV column naming, arm
                # entries/alternation) rather than from pending_roi_points
                # directly -- keep it in sync with whatever the researcher
                # ended up naming each generated zone.
                self.setup_page.set_roi_names_text(", ".join(op["regions"].keys()))

        self._op = None
        self.set_active_tool(None)
        self.setup_page.hide_op_bar()
        self.setup_page.refresh_canvas()

    def op_new_shape(self):
        if self._op and self._op["kind"] == "mask":
            self._op["shapes"].append([])
            self.setup_page.update_op_instructions()
            self.setup_page.refresh_canvas()

    def op_new_zone(self):
        """Draw Zones: add another empty, auto-named zone (Zone 1, Zone 2,
        ...) without needing to type its real name in the ROI names field
        first -- trace the outline now, then click INSIDE it afterwards to
        assign the real name (same click-to-rename popup Maze Template
        uses). Lets you draw all N shapes for a maze first and only decide
        which is which once you can see them next to each other, instead
        of committing to names before anything is on screen."""
        if not (self._op and self._op["kind"] == "zones"):
            return
        n = 1
        while f"Zone {n}" in self._op["regions"]:
            n += 1
        new_name = f"Zone {n}"
        self._op["regions"][new_name] = []
        self._op["active_region"] = new_name
        self.setup_page.rebuild_region_buttons()
        self.setup_page.update_op_instructions()
        self.setup_page.refresh_canvas()

    def op_set_active_region(self, name):
        if self._op is None:
            return
        self._op["active_region"] = name
        self.setup_page.rebuild_region_buttons()
        self.setup_page.update_op_instructions()
        self.setup_page.refresh_canvas()

    def op_set_draw_mode(self, mode):
        """Zones/Objects op bar: switch between Freehand (click-to-add-point,
        the original behavior) and the Rectangle/Ellipse/Line shape tools
        MM asked for after seeing them in the ANY-maze reference video --
        see on_canvas_press/on_canvas_drag's use of op['draw_mode'] and
        _compute_shape_points."""
        if self._op is None or self._op["kind"] not in ("zones", "objects"):
            return
        self._op["draw_mode"] = mode
        self._op.pop("_shape_anchor", None)
        self._op["drag"] = None
        self.setup_page.rebuild_draw_mode_buttons()
        self.setup_page.update_op_instructions()

    def op_toggle_snap(self, checked):
        if self._op is None:
            return
        self._op["snap"] = bool(checked)
        self.setup_page.update_op_instructions()
        self.setup_page.refresh_canvas()

    def _snap_point(self, op, fx, fy):
        if not op.get("snap"):
            return fx, fy
        spacing = op.get("grid_spacing") or 20
        return round(fx / spacing) * spacing, round(fy / spacing) * spacing

    def _compute_shape_points(self, op, cur_fx, cur_fy):
        """Builds the live point list for the in-progress Rectangle/Ellipse/
        Line shape, from op['_shape_anchor'] (set on press) to the current
        drag position -- called every drag event, so the ordinary zone/
        object polygon renderer (draw_op_overlay -> _draw_named_polygon_set)
        just draws whatever this returns, with no new rendering code
        needed. Once committed this is indistinguishable from a freehand
        polygon, so the existing corner-drag-to-adjust behavior keeps
        working on it afterward."""
        anchor = op.get("_shape_anchor")
        if anchor is None:
            return op["regions"].get(op["active_region"], [])
        x0, y0 = anchor
        x1, y1 = self._snap_point(op, cur_fx, cur_fy)
        mode = op.get("draw_mode", "freehand")
        if mode == "rectangle":
            return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if mode == "ellipse":
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
            if rx < 1 or ry < 1:
                return [(x0, y0)]
            n = 24
            return [(cx + rx * math.cos(2 * math.pi * i / n), cy + ry * math.sin(2 * math.pi * i / n))
                    for i in range(n)]
        if mode == "line":
            dx, dy = x1 - x0, y1 - y0
            length = math.hypot(dx, dy)
            if length < 1:
                return [(x0, y0)]
            # Perpendicular unit vector, scaled to a default arm/corridor
            # width -- a bare 2-point line can't be a zone polygon (finish_op
            # requires 3+ points), and a thin rectangle is exactly what a
            # maze arm (e.g. EPM open/closed arms) looks like anyway.
            ux, uy = -dy / length, dx / length
            half_w = max(10.0, 0.08 * length)
            return [
                (x0 + ux * half_w, y0 + uy * half_w),
                (x1 + ux * half_w, y1 + uy * half_w),
                (x1 - ux * half_w, y1 - uy * half_w),
                (x0 - ux * half_w, y0 - uy * half_w),
            ]
        return op["regions"].get(op["active_region"], [])

    def op_instructions_text(self):
        op = self._op
        if op is None:
            return ""
        if op["kind"] == "crop":
            n = len(op["shapes"][0])
            return f"Click the 4 arena corners ({n}/4 placed). Drag a corner to adjust it."
        elif op["kind"] == "mask":
            n_shapes = len(op["shapes"])
            n_pts = len(op["shapes"][-1])
            return (f"Shape {n_shapes} ({n_pts} pts so far). Click to add points, "
                    "'New Shape' to start another masked region.")
        elif op["kind"] in ("zones", "objects"):
            noun = "zone" if op["kind"] == "zones" else "object"
            name = op["active_region"]
            n = len(op["regions"][name])
            mode = op.get("draw_mode", "freehand")
            mode_hint = {
                "freehand": "click to add points, tracing its outline",
                "rectangle": "drag corner-to-corner for a box",
                "ellipse": "drag to set the oval's bounding box",
                "line": "drag from one end to the other for a thin arm/corridor",
            }.get(mode, "click to add points")
            snap_hint = " (snap to grid on)" if op.get("snap") else ""
            tail = (" Click a name above to switch, 'New Zone' to add another, or click "
                    "inside a finished zone to rename it." if op["kind"] == "zones"
                    else " Click a name above to switch.")
            return f"Drawing {noun} '{name}' ({n} pts, need 3+) -- {mode_hint}{snap_hint}.{tail}"
        elif op["kind"] == "distance":
            return f"Click 2 points of known real-world distance ({len(op['points'])}/2 placed)."
        elif op["kind"] == "template_zones":
            if op.get("_align_target_index") is not None:
                return (f"{len(op['regions'])} zone(s) from the template. Drag corners to fit THIS video's "
                        "own framing, then Finish to save just this video's alignment (other queued videos "
                        "are unaffected).")
            return (f"{len(op['regions'])} zone(s) generated. Drag a corner on the canvas to fix alignment "
                    "with the maze. To rename a zone (e.g. swap which arm is Open/Closed), click its name "
                    "below -- clicking the canvas never pops up a rename prompt by itself. Then Finish.")
        return ""

    # -- maze templates: geometry-driven zone generation, then click-to-confirm/rename --

    def start_template_zone_op(self, template_key, zones):
        base_frame = self.get_warped_active_frame()
        if base_frame is None:
            QMessageBox.critical(self, "No video", "Add and select a video first.")
            return

        self.set_active_tool("maze")

        # Suggested labels from the template can collide with each other in
        # freak cases -- de-dupe by appending a counter rather than
        # silently dropping a zone.
        regions = {}
        for label, pts in zones:
            name = label
            n = 2
            while name in regions:
                name = f"{label} ({n})"
                n += 1
            regions[name] = [tuple(p) for p in pts]

        self._op = {"kind": "template_zones", "drag": None, "_base_frame": base_frame,
                    "regions": regions, "template_key": template_key}
        self.setup_page.show_op_bar("template_zones")
        self.setup_page.refresh_canvas()

    def prompt_zone_label(self, op, current_name):
        """Popup shown after clicking a template-generated zone on the
        canvas: confirm the geometry's guess, pick a different one of the
        template's suggested names (e.g. swap which arm is actually
        'Open'), or type a custom name."""
        template_key = op.get("template_key")
        suggestions = []
        if template_key:
            try:
                suggestions = [label for label, _ in MAZE_TEMPLATES[template_key]["generate"](
                    self.pending_warp_w, self.pending_warp_h,
                    {p[0]: p[2] for p in MAZE_TEMPLATES[template_key]["params"]})]
            except Exception:
                suggestions = []
        existing_names = set(op["regions"].keys()) - {current_name}
        from qt_app.dialogs.zone_label_dialog import prompt_zone_label as _prompt
        return _prompt(self, current_name, suggestions, existing_names)

    def op_rename_template_zone(self, current_name):
        """Called from the explicit per-zone button in the op bar (not a
        canvas click -- see the comment in on_canvas_press's template_zones
        branch for why)."""
        op = self._op
        if op is None or op["kind"] != "template_zones" or current_name not in op["regions"]:
            return
        new_name = self.prompt_zone_label(op, current_name)
        if new_name and new_name != current_name:
            op["regions"][new_name] = op["regions"].pop(current_name)
            self.setup_page.rebuild_template_zone_buttons()
        self.setup_page.refresh_canvas()

    # -- canvas mouse events (fx, fy are already in frame-pixel space) --

    def _find_nearby_point(self, points, fx, fy, hit_radius_px=10):
        scale = getattr(self.setup_page.canvas, "scale", 1.0) if self.setup_page.canvas else 1.0
        hit_radius_frame = hit_radius_px / max(scale, 1e-6)
        best = None
        best_d = hit_radius_frame
        for i, (px, py) in enumerate(points):
            d = math.hypot(px - fx, py - fy)
            if d <= best_d:
                best_d = d
                best = i
        return best

    def _point_in_polygon(self, pt, polygon):
        if len(polygon) < 3:
            return False
        contour = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0

    def on_canvas_press(self, fx, fy):
        op = self._op
        if op is None:
            return

        if op["kind"] == "crop":
            idx = self._find_nearby_point(op["shapes"][0], fx, fy)
            if idx is not None:
                op["drag"] = ("shape", 0, idx)
            elif len(op["shapes"][0]) < 4:
                op["shapes"][0].append((fx, fy))

        elif op["kind"] == "mask":
            found = False
            for si, shape in enumerate(op["shapes"]):
                idx = self._find_nearby_point(shape, fx, fy)
                if idx is not None:
                    op["drag"] = ("shape", si, idx)
                    found = True
                    break
            if not found:
                op["shapes"][-1].append((fx, fy))

        elif op["kind"] == "zones":
            draw_mode = op.get("draw_mode", "freehand")
            # 1) A click near ANY zone's existing corner drags it -- not
            # just the active one -- so a zone can be fixed without first
            # switching to it. This always takes priority, whatever the
            # current shape-drawing mode, so a shape can still be nudged
            # after it's been committed.
            drag_target = None
            for name, pts in op["regions"].items():
                idx = self._find_nearby_point(pts, fx, fy)
                if idx is not None:
                    drag_target = (name, idx)
                    break
            if drag_target is not None:
                name, idx = drag_target
                op["drag"] = ("region", name, idx)
            elif draw_mode in ("rectangle", "ellipse", "line"):
                # 2) Rectangle/Ellipse/Line: start a new shape from this
                # corner -- see _compute_shape_points, called live from
                # on_canvas_drag.
                op["_shape_anchor"] = self._snap_point(op, fx, fy)
                op["drag"] = ("newshape", op["active_region"])
            else:
                # 3) Freehand: a click INSIDE a different, already-completed
                # zone opens the confirm/rename popup for that zone (same as
                # Maze Template) instead of adding a point to the active
                # one -- draw all the shapes first, then click each to
                # name it, rather than typing every name up front.
                clicked_other = None
                for name, pts in op["regions"].items():
                    if name != op["active_region"] and len(pts) >= 3 and self._point_in_polygon((fx, fy), pts):
                        clicked_other = name
                        break
                if clicked_other is not None:
                    new_name = self.prompt_zone_label(op, clicked_other)
                    if new_name and new_name != clicked_other:
                        op["regions"][new_name] = op["regions"].pop(clicked_other)
                        if op["active_region"] == clicked_other:
                            op["active_region"] = new_name
                        self.setup_page.rebuild_region_buttons()
                else:
                    fx, fy = self._snap_point(op, fx, fy)
                    op["regions"][op["active_region"]].append((fx, fy))

        elif op["kind"] == "objects":
            draw_mode = op.get("draw_mode", "freehand")
            active_pts = op["regions"][op["active_region"]]
            idx = self._find_nearby_point(active_pts, fx, fy)
            if idx is not None:
                op["drag"] = ("region", op["active_region"], idx)
            elif draw_mode in ("rectangle", "ellipse", "line"):
                op["_shape_anchor"] = self._snap_point(op, fx, fy)
                op["drag"] = ("newshape", op["active_region"])
            else:
                fx, fy = self._snap_point(op, fx, fy)
                op["regions"][op["active_region"]].append((fx, fy))

        elif op["kind"] == "distance":
            idx = self._find_nearby_point(op["points"], fx, fy)
            if idx is not None:
                op["drag"] = ("distance", idx)
            elif len(op["points"]) < 2:
                op["points"].append((fx, fy))

        elif op["kind"] == "template_zones":
            # Canvas clicks are drag-only here -- a click near an existing
            # corner drags that corner, exactly like crop/zones/objects/
            # distance, so the researcher can nudge each generated zone's
            # points to fit the real maze (camera angle/zoom differs per
            # rig). Renaming used to also live on the canvas (click inside
            # a zone's outline), but for an EPM the arm/center zones
            # overlap and are only a few px wide, so an ordinary alignment
            # click easily landed inside a DIFFERENT zone's outline and
            # popped up its rename dialog instead of dragging -- annoying
            # and not what was clicked for. Renaming now only happens via
            # the explicit per-zone buttons in the op bar (see
            # rebuild_region_buttons/op_rename_template_zone), so a canvas
            # click can never surprise you with a popup. A click that
            # isn't near any corner is simply ignored.
            drag_target = None
            for name, pts in op["regions"].items():
                idx = self._find_nearby_point(pts, fx, fy, hit_radius_px=14)
                if idx is not None:
                    drag_target = (name, idx)
                    break
            if drag_target is not None:
                name, idx = drag_target
                op["drag"] = ("region", name, idx)

        self.setup_page.update_op_instructions()
        self.setup_page.refresh_canvas()

    def on_canvas_drag(self, fx, fy):
        op = self._op
        if op is None or not op.get("drag"):
            return
        kind = op["drag"][0]
        if kind == "shape":
            _, si, idx = op["drag"]
            op["shapes"][si][idx] = (fx, fy)
        elif kind == "region":
            _, name, idx = op["drag"]
            fx, fy = self._snap_point(op, fx, fy)
            op["regions"][name][idx] = (fx, fy)
        elif kind == "distance":
            _, idx = op["drag"]
            op["points"][idx] = (fx, fy)
        elif kind == "newshape":
            _, name = op["drag"]
            op["regions"][name] = self._compute_shape_points(op, fx, fy)
        self.setup_page.refresh_canvas()

    def on_canvas_release(self):
        op = self._op
        if op is None:
            return
        if op.get("drag") and op["drag"][0] == "newshape":
            op.pop("_shape_anchor", None)
        op["drag"] = None

    # ------------------------------------------------------------------
    # Reset handlers
    # ------------------------------------------------------------------

    def _reset_calibration_state(self):
        self.pending_use_crop = None
        self.pending_matrix = None
        self.pending_warp_w = None
        self.pending_warp_h = None
        self.pending_crop_corners = None
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        self.pending_scale_factor = None
        self.pending_scale_unit = None
        self._op = None
        self.active_tool = None
        self.setup_page.reset_canvas_tool_state()

    def on_reset_video_chamber(self):
        if QMessageBox.question(self, "Reset video panel",
                                 "Clear the video queue and time/zone/analysis fields?") != QMessageBox.Yes:
            return
        self.videos = []
        self.active_index = None
        self.setup_page.reset_video_chamber_fields()

    def on_reset_calibration_chamber(self):
        self._reset_calibration_state()
        self.setup_page.refresh_canvas()

    def on_reset_settings_chamber(self):
        self.setup_page.reset_settings_fields()

    def on_reset_all(self):
        if QMessageBox.question(
                self, "Reset everything",
                "Reset ALL back to defaults?\n\nThis clears: videos, calibration, "
                "settings, and all remembered values.") != QMessageBox.Yes:
            return
        self._memory.clear()
        self.videos = []
        self.active_index = None
        self.subjects = []
        self._reset_calibration_state()
        # resave=False: rebuild()'s default would otherwise read the OLD
        # (about-to-be-destroyed) widgets' current text and write it right
        # back into _memory, undoing the clear() above before the fresh,
        # empty widgets ever get built.
        self.setup_page.rebuild(resave=False)
        self.status_label.setText("Idle.")
        self.show_setup_page()

    # ------------------------------------------------------------------
    # Open/Save Project, Settings, Help
    # ------------------------------------------------------------------

    def on_save_project(self):
        if not self.videos:
            QMessageBox.warning(self, "Nothing to save", "Add at least one video before saving a project.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "BehavioralTracker project (*.btproj)")
        if not path:
            return
        self.setup_page.save_all_to_memory(include_calibration=False)
        safe_memory = {k: v for k, v in self._memory.items() if not k.startswith("pending_")}
        data = {
            "app": PALETTE["APP_BRAND"], "version": PALETTE["APP_VERSION"],
            "analysis_type": self.analysis_type, "mode": self.get_mode(),
            "videos": [
                {"path": v["path"], "subject_id": v.get("subject_id", ""),
                 "roi_points_override": v.get("roi_points_override")}
                for v in self.videos
            ],
            "subjects": self.subjects,
            "memory": safe_memory,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        self.status_label.setText(f"Saved project: {os.path.basename(path)}")

    def on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "BehavioralTracker project (*.btproj)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return

        missing = []
        loaded_videos = []
        for entry in data.get("videos", []):
            vpath = entry.get("path") if isinstance(entry, dict) else None
            probed = self._probe_video(vpath) if vpath else None
            if probed is None:
                missing.append(vpath or "(unknown path)")
                continue
            if isinstance(entry, dict):
                if entry.get("subject_id"):
                    probed["subject_id"] = entry["subject_id"]
                override = entry.get("roi_points_override")
                if override:
                    probed["roi_points_override"] = {k: [tuple(p) for p in pts] for k, pts in override.items()}
            loaded_videos.append(probed)

        self.videos = loaded_videos
        self.active_index = 0 if self.videos else None
        self.subjects = data.get("subjects", [])

        analysis_type = data.get("analysis_type", "standard")
        if analysis_type not in self.analysis_type_buttons:
            analysis_type = "standard"
        mode = data.get("mode", "individual")
        (self.individual_radio if mode == "individual" else self.batch_radio).setChecked(True)

        self.analysis_type = analysis_type
        self._restyle_analysis_buttons()
        self._memory.update(data.get("memory", {}))
        self._reset_calibration_state()
        self.setup_page.rebuild(resave=False)
        self.show_setup_page()

        if missing:
            QMessageBox.warning(
                self, "Some videos missing",
                "Project settings loaded, but these video files couldn't be found and were "
                "skipped:\n\n" + "\n".join(missing),
            )
        self.status_label.setText(f"Opened project: {os.path.basename(path)}")

    def on_open_settings(self):
        QMessageBox.information(
            self, "Settings",
            "A dedicated Settings screen is on the way. For now, everything is on the setup "
            "screen itself: Detection Settings (right column) apply to the current Analysis "
            "Type, Save Project keeps a video queue and its fields for later, and Reset All "
            "clears everything back to defaults."
        )

    def on_open_help(self):
        QMessageBox.information(
            self, "Quick start",
            "1) Pick an Analysis Type above.\n"
            "2) Add your video(s) with the + button.\n"
            "3) Crop the arena (or click No Crop), then draw zones/objects, or use a Quick "
            "Setup template if your analysis type has one.\n"
            "4) Adjust Detection Settings on the right if the default tracking looks off.\n"
            "5) Click Start Tracking -- Results open automatically when it finishes.\n\n"
            "Open Project / Save Project let you reload a video queue and its setup fields "
            "later (calibration -- crop/zones/objects -- is redrawn per video, not saved)."
        )

    # ------------------------------------------------------------------
    # Start Tracking: builds the setup dict, runs the SAME synchronous
    # tracking.location/tracking.two_mouse/tracking.behavior pipeline the
    # Tkinter app uses (unchanged by this migration), and shows the
    # Results page. Mirrors TrackerApp.on_start/_build_setup/
    # _run_standard_flow/_run_multi_mouse_flow/_run_behavior_flow.
    #
    # Kept synchronous (no QThread), same as the Tkinter version -- the
    # progress callback below calls QApplication.processEvents() to pump
    # the UI during the blocking loop, matching Tkinter's
    # root.update_idletasks() in on_progress.
    # ------------------------------------------------------------------

    def on_progress(self, fraction):
        self.progress_bar.setValue(max(0, min(100, int(fraction * 100))))
        self.progress_label.setText(f"{fraction * 100:.0f}%")
        QApplication.processEvents()

    def _qt_confirm(self, title, prompt):
        """Passed to process_single_video as confirm_callback -- replaces
        its old hardcoded dependency on gui.main_window.ask_yes_no (a
        Tkinter dialog) with a native QMessageBox, so a Tkinter window
        never pops up inside this all-Qt app. See the confirm_callback
        docstring in tracking/location.py for why that mattered."""
        return QMessageBox.question(self, title, prompt) == QMessageBox.Yes

    def export_results_excel(self, output_dir):
        """Results page 'Export to Excel' button: builds ONE .xlsx workbook
        for a completed run, one sheet per result CSV found in output_dir
        (tracks/features/bouts/labeled_frames/manual_bouts/raw_tracking/
        etc.) -- so there's always a single direct Excel export regardless
        of which analysis type produced this output_dir. (The 'standard'
        pipeline already writes its own richer per-video
        {video}_Analysis.xlsx internally via output/excel.py, but that
        doesn't exist for multi_mouse/behavior/manual-scoring runs -- this
        covers all of them uniformly.) Raises on failure so the caller can
        show its own error dialog."""
        csvs = sorted(glob.glob(os.path.join(output_dir, "*.csv")))
        if not csvs:
            raise ValueError("No result CSVs found in this results folder yet.")
        xlsx_path = os.path.join(output_dir, "Results_Export.xlsx")
        used_names = set()
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for csv_path in csvs:
                base = os.path.splitext(os.path.basename(csv_path))[0][:31]  # Excel sheet-name limit
                sheet = base
                n = 2
                while sheet in used_names:
                    suffix = f"_{n}"
                    sheet = base[:31 - len(suffix)] + suffix
                    n += 1
                used_names.add(sheet)
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                df.to_excel(writer, sheet_name=sheet, index=False)
        return xlsx_path

    def _build_setup(self, video_path):
        sp = self.setup_page
        try:
            start_raw = sp.start_entry.text().strip()
            end_raw = sp.end_entry.text().strip()
            full_duration = 0.0
            if not start_raw or not end_raw:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                full_duration = total_frames / fps if fps else 0.0
            start_time = float(start_raw) if start_raw else 0.0
            end_time = float(end_raw) if end_raw else full_duration

            setup = {
                "video_path": video_path,
                "use_crop": self.pending_use_crop,
                "matrix": self.pending_matrix,
                "warp_w": self.pending_warp_w,
                "warp_h": self.pending_warp_h,
                "arena": (0, 0, self.pending_warp_w, self.pending_warp_h),
                "roi_names": [n.strip() for n in sp.roi_names_entry.text().split(",") if n.strip()],
                "roi_points": self.pending_roi_points,
                "mask_points": self.pending_mask_points,
                "object_names": [n.strip() for n in sp.object_names_entry.text().split(",") if n.strip()]
                    if sp.interact_var.isChecked() else [],
                "object_points": self.pending_object_points if sp.interact_var.isChecked() else {},
                "interaction_margin_px": float(sp.interaction_margin_entry.text())
                    if getattr(sp, "interaction_margin_entry", None) is not None else 0.0,
                "behavior_names": [],
                "background_samples": int(float(sp.bg_samples_entry.text())),
                "threshold": float(sp.threshold_entry.text()),
                "min_area": float(sp.min_area_entry.text()),
                "max_area": float(sp.max_area_entry.text()),
                "max_jump": float(sp.max_jump_entry.text()),
                "use_zone_threshold": sp.use_zone_threshold_var.isChecked(),
                "reject_shadows": sp.reject_shadows_var.isChecked() if getattr(sp, "reject_shadows_var", None) else False,
                "use_window": sp.use_window_var.isChecked(),
                "window_size": float(sp.window_size_entry.text()) if getattr(sp, "window_size_entry", None) is not None else 120.0,
                "window_weight": float(sp.window_weight_entry.text()) if getattr(sp, "window_weight_entry", None) is not None else 0.5,
                "scale_factor": self.pending_scale_factor,
                "scale_unit": self.pending_scale_unit,
                "preview_samples": int(float(sp.preview_samples_entry.text())),
                "color_mode": sp.get_color_mode(),
                "start_time": start_time,
                "end_time": end_time,
                "output_dir_override": None,
                "compute_arm_entries": sp.entries_var.isChecked(),
                "compute_alternation": sp.altern_var.isChecked(),
            }
        except ValueError as exc:
            QMessageBox.critical(self, "Invalid setting", f"Check the Detection Settings values -- {exc}")
            return None

        if setup["color_mode"] == "auto":
            setup["color_mode"] = self._resolve_auto_color_mode(
                video_path, num_animals=1, min_area=setup["min_area"], max_area=setup["max_area"],
                diff_threshold=setup["threshold"], n_background_samples=setup["background_samples"],
            )
        return setup

    def _resolve_auto_color_mode(self, video_path, num_animals, min_area, max_area,
                                  diff_threshold, n_background_samples):
        self.status_label.setText("Checking best color mode for this video...")
        QApplication.processEvents()
        try:
            resolved, _diagnostics = choose_color_mode(
                video_path, num_animals=num_animals, min_area=min_area, max_area=max_area,
                diff_threshold=diff_threshold,
                n_background_samples=min(int(n_background_samples), 20),
            )
        except Exception as exc:
            print(f"[warn] Auto color-mode selection failed ({exc}); defaulting to grayscale.")
            resolved = "gray"
        self.status_label.setText(f"Auto color mode picked: {resolved.upper()}")
        QApplication.processEvents()
        return resolved

    def on_start(self):
        if not self.videos:
            QMessageBox.critical(self, "No video", "Add a video first.")
            return
        if self.active_index is None:
            QMessageBox.critical(self, "No video selected", "Click a video in the list to select it.")
            return

        sp = self.setup_page
        try:
            min_area = int(float(sp.min_area_entry.text())) if getattr(sp, "min_area_entry", None) else 15
            max_area = int(float(sp.max_area_entry.text())) if getattr(sp, "max_area_entry", None) else 5000
            if min_area >= max_area:
                QMessageBox.critical(
                    self, "Invalid area settings",
                    f"Min mouse area ({min_area} px) must be LESS THAN Max object area ({max_area} px)."
                )
                return
        except (ValueError, AttributeError, RuntimeError):
            pass

        try:
            start_raw = sp.start_entry.text() if getattr(sp, "start_entry", None) else ""
            end_raw = sp.end_entry.text() if getattr(sp, "end_entry", None) else ""
            if start_raw.strip() and end_raw.strip():
                start_s = float(start_raw)
                end_s = float(end_raw)
                if start_s >= end_s:
                    QMessageBox.critical(
                        self, "Invalid time window",
                        f"Start time ({start_s}s) must be LESS THAN end time ({end_s}s)."
                    )
                    return
                if start_s < 0:
                    QMessageBox.critical(self, "Invalid time window", "Start time cannot be negative.")
                    return
        except (ValueError, AttributeError, RuntimeError):
            pass

        if self.analysis_type == "standard":
            self._run_standard_flow()
        elif self.analysis_type == "multi_mouse":
            self._run_multi_mouse_flow()
        elif self.analysis_type == "behavior":
            self._run_behavior_flow()

    def _run_standard_flow(self):
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return
        if not self.pending_roi_points:
            QMessageBox.critical(self, "Zones not set", "Use 'Draw Zones' first.")
            return

        active_path = self.videos[self.active_index]["path"]
        base_setup = self._build_setup(active_path)
        if base_setup is None:
            return

        self.progress_bar.setValue(0)
        self.status_label.setText("Tracking...")
        QApplication.processEvents()

        if self.get_mode() == "individual":
            try:
                # show_display=False here (Tkinter uses True for Individual
                # mode) -- this build's OpenCV is built against Qt5
                # (cv2.imshow/waitKey open a Qt5 window), and running that
                # alongside this PySide6 (Qt6) app in the same process can
                # hard-crash it (confirmed: cv2.imshow aborts the process
                # outright rather than raising a catchable exception, the
                # moment both toolkits are active together). The interactive
                # "confirm the background"/"press a key to advance the
                # detection preview" steps that show_display=True used to
                # gate are skipped, but nothing is lost: run_detection_preview
                # still saves every preview frame to disk unconditionally,
                # and the Results page's Reference Frames gallery below is
                # exactly those same images, so sanity-checking detection is
                # still available -- just after the run instead of during it,
                # the same way Batch mode has always worked here.
                summary = process_single_video(active_path, base_setup, show_display=False,
                                               progress_callback=self.on_progress,
                                               confirm_callback=self._qt_confirm)
            except SystemExit as exc:
                self.status_label.setText(f"Stopped: {exc}")
                cv2.destroyAllWindows()
                return
            except Exception as exc:
                self.status_label.setText(f"Error: {type(exc).__name__}")
                cv2.destroyAllWindows()
                QMessageBox.critical(self, "Tracking Error",
                                      f"An error occurred during tracking:\n\n{type(exc).__name__}: {exc}")
                return
            self.progress_bar.setValue(100)
            self.status_label.setText("Done.")
            try:
                self.results_page.show_standard(summary)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results Error", f"Could not display results:\n\n{type(exc).__name__}: {exc}")
            return

        # Per-video zone alignment (on_align_video/'Align' button in the
        # video queue -- MM's "multiple apparatus, separate video files"
        # ask): if any queued video has its own saved roi_points_override,
        # use it for that video and skip the old all-or-nothing question;
        # videos without one still fall back to the shared template below.
        has_overrides = any(v.get("roi_points_override") for v in self.videos)
        if has_overrides:
            aligned = sum(1 for v in self.videos if v.get("roi_points_override"))
            proceed = self._qt_confirm(
                "Per-video alignment",
                f"{aligned}/{len(self.videos)} video(s) have their own saved zone alignment "
                "(from the 'Align' button in the video list) -- those will be used as-is. "
                "The rest will reuse the currently on-screen zones.\n\nContinue?"
            )
            if not proceed:
                self.status_label.setText("Batch cancelled.")
                return
        else:
            same_camera = self._qt_confirm(
                "Camera position",
                f"Found {len(self.videos)} videos.\n\n"
                "Was the camera in EXACTLY the same position for all of them?\n\n"
                "Yes = reuse this calibration for every video (fast)\n"
                "No = click 'Align' next to each video in the queue first to nudge its own "
                "zones to fit that video's framing, then Start again -- videos with a saved "
                "alignment are used automatically."
            )
            if not same_camera:
                self.status_label.setText("Batch cancelled -- use 'Align' on each video in the queue, then Start again.")
                return

        summaries = []
        errors = []
        for i, v in enumerate(self.videos):
            print(f"\n[{i + 1}/{len(self.videos)}] Processing: {v['path']}")
            self.status_label.setText(f"Batch: video {i + 1}/{len(self.videos)} -- {os.path.basename(v['path'])}")
            self.progress_bar.setValue(0)
            self.progress_label.setText("")
            QApplication.processEvents()
            setup = dict(base_setup)
            setup["video_path"] = v["path"]
            if v.get("roi_points_override"):
                setup["roi_points"] = v["roi_points_override"]
            try:
                # show_display=False -- same Qt5/Qt6 crash-avoidance reason
                # as the Individual-mode branch above (a cv2.imshow window
                # would hard-crash this Qt6 app), and it also matches the
                # Tkinter app's own batch behavior: with show_display=True,
                # every video in the batch would pop up its own blocking
                # "press a key to continue" step, stalling an unattended
                # batch run N times over.
                summary = process_single_video(v["path"], setup, show_display=False,
                                               progress_callback=self.on_progress,
                                               confirm_callback=self._qt_confirm)
                summary["subject_id"] = v.get("subject_id", "")
                summaries.append(summary)
            except SystemExit as exc:
                print(f"  Skipped: {exc}")
                errors.append((os.path.basename(v["path"]), str(exc)))
                continue
            except Exception as exc:
                print(f"  Error: {type(exc).__name__}: {exc}")
                errors.append((os.path.basename(v["path"]), f"{type(exc).__name__}: {exc}"))
                continue

        cv2.destroyAllWindows()
        self.progress_bar.setValue(100)

        if summaries:
            batch_df = pd.DataFrame(summaries)
            folder = os.path.dirname(self.videos[0]["path"])
            batch_path = os.path.join(folder, "BatchSummary.csv")
            batch_xlsx_path = os.path.join(folder, "BatchSummary.xlsx")
            try:
                batch_df.to_csv(batch_path, index=False)
                batch_df.to_excel(batch_xlsx_path, index=False, engine="openpyxl")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Could not save batch summary:\n\n{exc}")
            self.status_label.setText("Batch complete.")
            msg = (f"Processed {len(summaries)}/{len(self.videos)} videos.\n\n"
                   f"Batch summary:\n{batch_path}\n{batch_xlsx_path}")
            if errors:
                msg += f"\n\n{len(errors)} video(s) had issues:\n"
                for name, err in errors[:10]:
                    msg += f"  - {name}: {err}\n"
            QMessageBox.information(self, "Batch Complete", msg)
        else:
            self.status_label.setText("Batch: no videos processed.")
            if errors:
                msg = f"No videos were successfully processed ({len(errors)} had issues):\n\n"
                for name, err in errors[:15]:
                    msg += f"  - {name}: {err}\n"
                QMessageBox.warning(self, "Batch Complete", msg)

    # ------------------------------------------------------------------
    # Multi-Mouse Tracking / Behavior Classification: shared video prep
    # (mirrors TrackerApp._prepare_source_video)
    # ------------------------------------------------------------------

    def _prepare_source_video(self, video_path):
        """two_mouse.track_video() and behavior.extract_features() both
        take a plain video path and process it top to bottom -- neither
        knows about arena cropping, a start/end trim window, or a masked-
        out region. To still support Crop Arena, the time window, and Mask
        Zone for these analysis types, this pre-warps the requested frame
        range into a temporary video file first (painting any masked
        polygons a flat color in every frame), deleting it afterward.
        Returns (path_to_use, temp_path_or_None, fps)."""
        sp = self.setup_page
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            start_time = float(sp.start_entry.text() or 0)
        except ValueError:
            start_time = 0
        try:
            end_time = float(sp.end_entry.text() or (total_frames / fps))
        except ValueError:
            end_time = total_frames / fps

        start_frame = max(0, int(start_time * fps))
        end_frame = min(total_frames, int(end_time * fps))

        full_range = start_frame == 0 and end_frame >= total_frames
        if full_range and not self.pending_use_crop and not self.pending_mask_points:
            cap.release()
            return video_path, None, fps

        warp_w, warp_h = self.pending_warp_w, self.pending_warp_h
        matrix = self.pending_matrix
        mask_polygons = [np.array(pts, dtype=np.int32) for pts in self.pending_mask_points]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".avi")
        os.close(tmp_fd)
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (warp_w, warp_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_number = start_frame
        while frame_number < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            warped = cv2.warpPerspective(frame, matrix, (warp_w, warp_h))
            for pts in mask_polygons:
                cv2.fillPoly(warped, [pts], (128, 128, 128))
            writer.write(warped)
            frame_number += 1
        cap.release()
        writer.release()
        return tmp_path, tmp_path, fps

    # ------------------------------------------------------------------
    # Multi-Mouse Tracking (mirrors TrackerApp._run_multi_mouse_flow)
    # ------------------------------------------------------------------

    def _run_multi_mouse_flow(self):
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        sp = self.setup_page
        active_path = self.videos[self.active_index]["path"]
        tmp_path = None

        try:
            try:
                min_area = int(float(sp.min_area_entry.text()))
                max_area = int(float(sp.max_area_entry.text()))
                threshold = int(float(sp.threshold_entry.text()))
                bg_samples = int(float(sp.bg_samples_entry.text()))
            except ValueError:
                QMessageBox.critical(self, "Invalid setting", "Check the Detection Settings values.")
                return

            self.progress_bar.setValue(0)
            self.status_label.setText("Preparing video...")
            QApplication.processEvents()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            tracks_csv = os.path.join(output_dir, "tracks.csv")
            annotate_path = os.path.join(output_dir, "preview.mp4")

            try:
                preview_n = int(float(sp.preview_samples_entry.text()))
            except ValueError:
                preview_n = 6

            color_mode = sp.get_color_mode()
            if color_mode == "auto":
                color_mode = self._resolve_auto_color_mode(
                    source_path, num_animals=sp.num_animals, min_area=min_area, max_area=max_area,
                    diff_threshold=threshold, n_background_samples=bg_samples,
                )

            self.status_label.setText("Saving reference frames...")
            QApplication.processEvents()
            try:
                preview_paths = save_preview_frames_multi_mouse(
                    source_path, output_dir, n_samples=preview_n,
                    num_animals=sp.num_animals, min_area=min_area, max_area=max_area,
                    diff_threshold=threshold, n_background_samples=bg_samples,
                    color_mode=color_mode,
                )
            except Exception as exc:
                print(f"[warn] Reference frames: {exc}")
                preview_paths = []

            self.status_label.setText("Tracking (multi-mouse)...")
            QApplication.processEvents()

            # show_display=False (Tkinter uses True, a live "Q to stop"
            # preview window) -- see the matching comment in
            # _run_standard_flow: this build's cv2 is Qt5-based, and a
            # cv2.imshow window alongside this Qt6 app can hard-crash the
            # process. annotate_path (preview.mp4) already gives a full
            # annotated video to review afterward instead.
            tracks_df = track_video(
                source_path, tracks_csv, annotate_path=annotate_path,
                num_animals=sp.num_animals, min_area=min_area, max_area=max_area,
                diff_threshold=threshold, n_background_samples=bg_samples,
                progress_callback=self.on_progress, show_display=False,
                color_mode=color_mode,
            )
            cv2.destroyAllWindows()

            object_names = [n.strip() for n in sp.object_names_entry.text().split(",") if n.strip()] \
                if sp.interact_var.isChecked() else []
            active_object_points = {k: v for k, v in self.pending_object_points.items() if k in object_names} \
                if object_names else {}
            try:
                min_bout_s = float(sp.min_bout_entry.text()) if getattr(sp, "min_bout_entry", None) else 0.3
            except (ValueError, RuntimeError, AttributeError):
                min_bout_s = 0.3
            try:
                interaction_margin_px = float(sp.interaction_margin_entry.text()) \
                    if getattr(sp, "interaction_margin_entry", None) is not None else 0.0
            except (ValueError, RuntimeError, AttributeError):
                interaction_margin_px = 0.0

            per_mouse_summary = {}
            per_mouse_bouts = {}
            if tracks_df is not None and len(tracks_df) > 0:
                roi_to_use = self.pending_roi_points if sp.loc_var.isChecked() else {}
                for mouse_id, sub in tracks_df.groupby("mouse_id"):
                    sub = sub.sort_values("frame").reset_index(drop=True)
                    try:
                        if "x" in sub.columns and "y" in sub.columns and "frame" in sub.columns:
                            summary, bouts = compute_zone_interaction_stats(
                                sub, roi_to_use, active_object_points,
                                self.pending_warp_w, self.pending_warp_h, fps, min_bout_s=min_bout_s,
                                scale_factor=self.pending_scale_factor, scale_unit=self.pending_scale_unit,
                                interaction_margin_px=interaction_margin_px,
                            )
                            per_mouse_summary[mouse_id] = summary
                            per_mouse_bouts[mouse_id] = bouts
                    except Exception as exc:
                        print(f"[warn] zone stats for {mouse_id}: {exc}")

            summary_csv = None
            bouts_csv = None
            try:
                if per_mouse_summary:
                    summary_df = pd.DataFrame.from_dict(per_mouse_summary, orient="index")
                    summary_df.index.name = "mouse_id"
                    summary_csv = os.path.join(output_dir, "per_mouse_summary.csv")
                    summary_df.to_csv(summary_csv)
                all_bouts_rows = []
                for mouse_id, bouts in per_mouse_bouts.items():
                    for b in bouts:
                        row = dict(b)
                        row["mouse_id"] = mouse_id
                        all_bouts_rows.append(row)
                if all_bouts_rows:
                    bouts_csv = os.path.join(output_dir, "per_mouse_object_bouts.csv")
                    pd.DataFrame(all_bouts_rows).to_csv(bouts_csv, index=False)
            except Exception as exc:
                print(f"[warn] saving per-mouse CSV: {exc}")

            self.progress_bar.setValue(100)
            self.status_label.setText("Done.")

            results = {
                "analysis_type": "multi_mouse", "video_path": active_path, "output_dir": output_dir,
                "tracks_df": tracks_df if tracks_df is not None else pd.DataFrame(),
                "tracks_csv": tracks_csv, "annotate_path": annotate_path,
                "fps": fps, "per_mouse_summary": per_mouse_summary, "per_mouse_bouts": per_mouse_bouts,
                "summary_csv": summary_csv, "bouts_csv": bouts_csv, "preview_paths": preview_paths,
            }
            try:
                self.results_page.show_multi_mouse(results)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results Error", f"Could not display results:\n\n{type(exc).__name__}: {exc}")

        except SystemExit as exc:
            self.status_label.setText(f"Stopped: {exc}")
            cv2.destroyAllWindows()
            return
        except Exception as exc:
            self.status_label.setText(f"Error: {type(exc).__name__}")
            cv2.destroyAllWindows()
            QMessageBox.critical(self, "Multi-Mouse Error", f"An error occurred:\n\n{type(exc).__name__}: {exc}")
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Behavior Classification (rule-based; mirrors
    # TrackerApp._run_behavior_flow -- the ML-model branch is a separate,
    # not-yet-wired-up path, same as the Prepare/Train buttons)
    # ------------------------------------------------------------------

    def _run_behavior_flow(self):
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        sp = self.setup_page
        active_path = self.videos[self.active_index]["path"]

        if getattr(sp, "ml_mode_var", None) is not None and sp.ml_mode_var.isChecked():
            self._run_behavior_flow_ml(active_path)
            return

        tmp_path = None
        try:
            try:
                min_area = int(float(sp.min_area_entry.text()))
                max_area = int(float(sp.max_area_entry.text()))
                threshold = int(float(sp.threshold_entry.text()))
                bg_samples = int(float(sp.bg_samples_entry.text()))
                max_jump = float(sp.max_jump_entry.text())
                loco_thresh = float(sp.loco_thresh_entry.text())
                rear_thresh = float(sp.rear_thresh_entry.text())
                groom_thresh = float(sp.groom_thresh_entry.text())
                persistence_frames = int(float(sp.immobile_thresh_entry.text()))
                min_bout_s = float(sp.min_bout_entry.text())
            except (ValueError, AttributeError, RuntimeError):
                QMessageBox.critical(self, "Invalid setting", "Check the Detection Settings values.")
                return

            self.progress_bar.setValue(0)
            self.status_label.setText("Preparing video...")
            QApplication.processEvents()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            features_csv = os.path.join(output_dir, "features.csv")
            bouts_csv = os.path.join(output_dir, "bouts.csv")
            labeled_csv = os.path.join(output_dir, "labeled_frames.csv")

            try:
                preview_n = int(float(sp.preview_samples_entry.text()))
            except (ValueError, AttributeError, RuntimeError):
                preview_n = 6

            color_mode = sp.get_color_mode()
            if color_mode == "auto":
                color_mode = self._resolve_auto_color_mode(
                    source_path, num_animals=sp.num_animals, min_area=min_area, max_area=max_area,
                    diff_threshold=threshold, n_background_samples=bg_samples,
                )

            self.status_label.setText("Saving reference frames...")
            QApplication.processEvents()
            try:
                preview_paths = save_preview_frames_behavior(
                    source_path, output_dir, n_samples=preview_n, num_animals=sp.num_animals,
                    min_area=min_area, max_area=max_area, diff_threshold=threshold,
                    n_background_samples=bg_samples, color_mode=color_mode,
                    max_jump_px=max_jump,
                )
            except Exception as exc:
                QMessageBox.critical(self, "Reference frames failed", str(exc))
                preview_paths = []

            self.status_label.setText("Extracting features...")
            QApplication.processEvents()

            extract_features(
                source_path, features_csv, num_animals=sp.num_animals,
                min_area=min_area, max_area=max_area, diff_threshold=threshold,
                n_background_samples=bg_samples, color_mode=color_mode,
                max_jump_px=max_jump, progress_callback=self.on_progress,
            )
            self.status_label.setText("Classifying behaviors...")
            QApplication.processEvents()
            labeled_df, bouts_df = classify_behaviors(
                features_csv, bouts_csv, labeled_output_csv=labeled_csv,
                loco_body_move_high=loco_thresh, rear_score_threshold=rear_thresh,
                groom_score_threshold=groom_thresh,
                groom_min_persistence=persistence_frames, rear_min_persistence=persistence_frames,
                min_bout_s=min_bout_s,
            )

            if labeled_df is None or len(labeled_df) == 0:
                QMessageBox.warning(
                    self, "No behavior data",
                    "No frames could be classified. Try adjusting the detection threshold or area settings."
                )

            selected = {
                "rearing": sp.behavior_rearing_var.isChecked(),
                "grooming": sp.behavior_grooming_var.isChecked(),
                "locomotion": sp.behavior_locomotion_var.isChecked(),
                "immobile": sp.behavior_immobile_var.isChecked(),
            }
            if (not any(selected.values())) or sp.behavior_all_var.isChecked():
                selected = {k: True for k in selected}

            self.progress_bar.setValue(100)
            self.status_label.setText("Done.")

            results = {
                "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df, "selected_behaviors": selected,
                "bouts_csv": bouts_csv, "labeled_csv": labeled_csv, "fps": fps, "preview_paths": preview_paths,
            }
            try:
                self.results_page.show_behavior(results)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results display failed",
                                      f"Results were saved but could not be shown: {exc}")

        except SystemExit:
            self.progress_bar.setValue(0)
            self.status_label.setText("Tracking stopped.")
            cv2.destroyAllWindows()
            return
        except Exception as exc:
            self.progress_bar.setValue(0)
            self.status_label.setText("Error occurred.")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            QMessageBox.critical(self, "Tracking failed", str(exc))
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _run_behavior_flow_ml(self, active_path):
        """The optional-mode counterpart to _run_behavior_flow(): instead
        of extract_features()+classify_behaviors(), runs the trained
        deep-learning model (tracking.ml_infer.classify_video_ml) over the
        video and builds the exact same results dict shape
        ResultsPage.show_behavior() expects -- same idea as
        _run_behavior_manual_flow() reusing that screen for manual
        scoring. Mirrors TrackerApp._run_behavior_flow_ml field-for-field.
        v1 limitation, carried over from the Tkinter app: the model looks
        at the whole frame, not a per-animal crop, so with more than one
        animal in frame its bouts describe the SCENE, not a specific
        mouse -- warned about below."""
        sp = self.setup_page
        checkpoint_path = sp.ml_checkpoint_entry.text().strip()
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            QMessageBox.critical(
                self, "No trained model",
                "Choose a trained model (.pt) file above, or use 'Train Model...' to make one first."
            )
            return
        from tracking.ml_train import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            QMessageBox.critical(
                self, "PyTorch not installed",
                "Install PyTorch (see pytorch.org) to use a trained model here, or uncheck "
                "'Use trained model' to use the automatic rules instead."
            )
            return
        if sp.num_animals > 1:
            if QMessageBox.question(
                self, "Single-animal model",
                "The trained model looks at the whole frame, not one animal at a time -- with "
                f"{sp.num_animals} animals in frame its bouts describe the scene as a whole, "
                "not a specific mouse. Continue anyway?"
            ) != QMessageBox.Yes:
                return

        from tracking.ml_infer import classify_video_ml

        tmp_path = None
        try:
            self.progress_bar.setValue(0)
            self.status_label.setText("Preparing video...")
            QApplication.processEvents()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            bouts_csv = os.path.join(output_dir, "ml_bouts.csv")

            self.status_label.setText("Running trained model...")
            QApplication.processEvents()

            def on_window(done, total):
                self.on_progress(done / total if total else 1.0)

            subject_name = "mouse_A"
            bouts_df = classify_video_ml(
                source_path, checkpoint_path, subject=subject_name,
                fps_override=fps, progress_callback=on_window,
            )
            os.makedirs(output_dir, exist_ok=True)
            bouts_df.to_csv(bouts_csv, index=False)

            if len(bouts_df) == 0:
                QMessageBox.warning(
                    self, "No behavior data",
                    "The trained model didn't confidently detect any of its trained behaviors in "
                    "this video."
                )

            cap = cv2.VideoCapture(source_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
            cap.release()
            duration_s = (frame_count / vid_fps) if vid_fps else 0.0
            labeled_df = pd.DataFrame({"time_s": [duration_s]})

            selected = {b: True for b in bouts_df["behavior"].unique()} if len(bouts_df) else \
                {"grooming": True, "rearing": True}

            self.progress_bar.setValue(100)
            self.status_label.setText(f"Done -- {len(bouts_df)} bouts from the trained model.")

            results = {
                "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df, "selected_behaviors": selected,
                "bouts_csv": bouts_csv, "labeled_csv": bouts_csv, "fps": vid_fps, "preview_paths": [],
            }
            try:
                self.results_page.show_behavior(results)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results display failed",
                                      f"Results were saved but could not be shown: {exc}")

        except Exception as exc:
            self.progress_bar.setValue(0)
            self.status_label.setText("Error occurred.")
            QMessageBox.critical(self, "ML classification failed", str(exc))
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _default_ml_dataset_dir(self):
        if self.active_index is not None and self.videos:
            active_path = self.videos[self.active_index]["path"]
            return os.path.join(os.path.dirname(active_path), "ml_dataset")
        return ""

    # ------------------------------------------------------------------
    # Dialogs (Maze Template, Manual Scoring, ML classifier)
    # ------------------------------------------------------------------

    def on_open_maze_template_dialog(self):
        from qt_app.dialogs.maze_template_dialog import open_maze_template_dialog
        open_maze_template_dialog(self, self)

    def on_manual_behavior_scoring(self):
        if not self.videos:
            QMessageBox.critical(self, "No video", "Add a video first.")
            return
        if self.active_index is None:
            QMessageBox.critical(self, "No video selected", "Click a video in the list to select it.")
            return
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return
        self._run_behavior_manual_flow()

    def _run_behavior_manual_flow(self):
        """Same idea as _run_behavior_flow(), but instead of the automatic
        classifier this opens the (cropped/masked/time-windowed, same as
        the automatic path) video in the ManualScoringDialog and lets the
        researcher mark bout start/stop themselves. Builds the exact same
        results dict shape ResultsPage.show_behavior() expects, so the
        Results page and export don't need to know which one ran. Mirrors
        TrackerApp._run_behavior_manual_flow field-for-field, aside from
        the dialog itself being Qt-native (see manual_scoring_dialog.py's
        module docstring for why manual_score_video()'s cv2 window
        couldn't be reused as-is)."""
        sp = self.setup_page
        active_path = self.videos[self.active_index]["path"]
        tmp_path = None
        try:
            self.status_label.setText("Preparing video...")
            QApplication.processEvents()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            bouts_csv = os.path.join(output_dir, "manual_bouts.csv")

            subject_names = [f"mouse_{chr(65 + i)}" for i in range(sp.num_animals)]

            self.status_label.setText("Manual scoring -- window open. "
                                       "SPACE=play/pause, 1-4=behaviors, Q=finish.")
            QApplication.processEvents()

            from qt_app.dialogs.manual_scoring_dialog import ManualScoringDialog
            dialog = ManualScoringDialog(self, source_path, subject_names)
            dialog.exec()
            bouts_df = dialog.bouts_df
            duration_s = dialog.duration_s
            vid_fps = dialog.fps or fps or 30.0

            os.makedirs(output_dir, exist_ok=True)
            bouts_df.to_csv(bouts_csv, index=False)
            labeled_df = pd.DataFrame({"time_s": [duration_s]})

            self.progress_bar.setValue(100)
            self.status_label.setText(f"Manual scoring saved -- {len(bouts_df)} bouts.")

            results = {
                "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df,
                "selected_behaviors": {"rearing": True, "grooming": True, "locomotion": True, "immobile": True},
                "bouts_csv": bouts_csv, "labeled_csv": bouts_csv, "fps": vid_fps, "preview_paths": [],
            }
            try:
                self.results_page.show_behavior(results)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results display failed",
                                      f"Results were saved but could not be shown: {exc}")

        except Exception as exc:
            self.progress_bar.setValue(0)
            self.status_label.setText("Error occurred.")
            QMessageBox.critical(self, "Manual scoring failed", str(exc))
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def on_browse_ml_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select a trained model (.pt)",
                                               "", "PyTorch checkpoint (*.pt);;All files (*.*)")
        if path:
            self.setup_page.ml_checkpoint_entry.setText(path)

    def on_prepare_ml_dataset(self):
        from qt_app.dialogs.ml_dataset_dialog import open_prepare_ml_dataset_dialog
        open_prepare_ml_dataset_dialog(self, self)

    def on_train_ml_model(self):
        from qt_app.dialogs.ml_train_dialog import open_train_ml_model_dialog
        open_train_ml_model_dialog(self, self)
