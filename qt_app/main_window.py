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

from qt_app.theme import PALETTE, build_stylesheet
from qt_app.app_settings import load_settings
from analysis.custom_variables import parse_custom_variables

ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "resources", "icon.png")

# Setup page's "Stop condition (optional)" combo box labels -> the internal
# codes tracking.location.process_single_video understands. Shared between
# _build_setup (reads the combo) and on_start (validates it) so the two
# never drift out of sync.
STOP_CONDITION_CODES = {
    "None": None,
    "N seconds of immobility": "immobility",
    "N entries into a zone": "zone_entries",
    "N pixels of total distance": "distance",
}


def parse_partition_formula(text):
    """Setup page's Zone Formula box (Draw Zone Outline's auto-detected
    A/B/C/... arms/zones): "A = Center; B+C = Left Arm" -> ({"Center": ["A"],
    "Left Arm": ["B", "C"]}, None), or ({}, "<message>") for the first
    malformed piece found. Blank text is valid (no renaming applied -- the
    raw letters are used as-is). Same shape/style as parse_zone_associations
    just above, except a single member ('A = Center') is allowed here,
    since naming ONE partition is the common case, not just merging
    several."""
    text = text.strip()
    if not text:
        return {}, None
    groups = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            return {}, f"'{chunk}' is missing '=' -- expected 'A = Zone Name' or 'A+B = Zone Name'."
        name, members_raw = chunk.split("=", 1)
        name = name.strip()
        members = [m.strip() for m in members_raw.split("+") if m.strip()]
        if not name:
            return {}, f"A zone formula entry is missing its name in '{chunk}'."
        if not members:
            return {}, f"'{name}' needs at least one partition letter (e.g. 'A')."
        groups[name] = members
    return groups, None


def parse_zone_associations(text):
    """Setup page's Zone Associations box: "Group = Zone A + Zone B; Group2
    = Zone C + Zone D" -> ({"Group": ["Zone A", "Zone B"], ...}, None), or
    ({}, "<message>") for the first malformed piece found. Blank text is
    valid and parses to no groups at all -- this is an optional feature.
    Pure/Qt-free so it's easy to unit test on its own (see
    test_zone_associations.py)."""
    text = text.strip()
    if not text:
        return {}, None
    groups = {}
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            return {}, f"'{chunk}' is missing '=' -- expected 'Group Name = Zone A + Zone B'."
        name, members_raw = chunk.split("=", 1)
        name = name.strip()
        members = [m.strip() for m in members_raw.split("+") if m.strip()]
        if not name:
            return {}, f"A zone association is missing its group name in '{chunk}'."
        if len(members) < 2:
            return {}, f"'{name}' needs at least 2 zones joined with '+' (found {len(members)})."
        groups[name] = members
    return groups, None

from tracking.location import (
    identity_transform, compute_perspective_transform, process_single_video,
    compute_output_dir, compute_zone_interaction_stats,
    make_background, read_and_warp, detect_apparatus_partitions,
)
from tracking.two_mouse import (
    track_video, save_preview_frames as save_preview_frames_multi_mouse, choose_color_mode,
    calculate_time_bins,
)
from tracking.behavior import (
    extract_features, classify_behaviors, save_preview_frames as save_preview_frames_behavior,
    calculate_behavior_time_bins,
)
from analysis.calculations import point_distance
from qt_app.widgets.preview_canvas import draw_overlays


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

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
        # Real Settings screen (Upgrade Plan Tier 2 #8) -- Detection
        # Settings defaults and a couple of app-wide preferences (default
        # color mode, default distance-calibration units) used to reset to
        # their hardcoded fallbacks every launch since _memory always
        # started empty; now they're loaded straight into it here, so
        # every Setup page field that already reads _memory (see
        # setup_page.py's entry_or_default/var_or_default/_setting_field/
        # _color_mode_group) picks them up with no separate plumbing.
        # Nothing persisted yet (first run, or before this feature
        # existed) -> load_settings() returns {} -> no behavior change.
        self._memory.update(load_settings())

        self.pending_use_crop = None
        self.pending_matrix = None
        self.pending_warp_w = None
        self.pending_warp_h = None
        self.pending_crop_corners = None
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        # The traced apparatus outline(s) from the "Draw Zone Outline" tool
        # (see start_op/finish_op's "zone_lines" kind) -- one closed shape
        # per traced outline (usually just one; "New Shape" allows more,
        # e.g. an apparatus split across the frame). Kept around so
        # reopening the tool starts from what was last drawn, same as
        # pending_mask_points does for Mask Zone.
        self.pending_zone_lines = []
        # Which tool last WROTE pending_roi_points -- "lines" (Draw Zone
        # Outline's auto-detected arms/zones) or "manual" (Draw Zones,
        # hand-drawn/edited). Draw Zones and Draw Zone Outline are two
        # alternative ways to define the same zones, not two tools meant
        # to be mixed: start_op reads this so opening Draw Zones right
        # after a Draw Zone Outline run starts from a blank slate instead
        # of silently pre-loading the auto-detected (many-point, not
        # meant to be hand-edited) shapes under their auto-picked A/B/C
        # names.
        self._roi_points_source = None
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
        reset_btn = QPushButton("Reset All")
        reset_btn.setObjectName("resetAllBtn")
        reset_btn.clicked.connect(self.on_reset_all)
        for b in (open_btn, save_btn, settings_btn, help_btn, reset_btn):
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

    def on_open_camera(self):
        """'Camera' video source (Upgrade Plan Tier 2 #6) -- opens the live
        acquisition dialog. Recording is handled entirely inside that
        dialog; it hands the saved file back via on_camera_recorded below
        once the researcher stops recording, rather than this method
        blocking on it."""
        from qt_app.dialogs.camera_dialog import open_camera_dialog
        open_camera_dialog(self, self)

    def on_camera_recorded(self, path):
        """Called by the Camera dialog once a recording is stopped -- adds
        the saved file to the video queue exactly like Add Video(s) does,
        so every existing feature (zones, Subject Database, batch mode,
        Start Tracking, ...) picks it up completely unchanged."""
        if self.get_mode() == "individual" and self.videos:
            QMessageBox.warning(
                self, "Individual mode",
                f"Individual mode allows only 1 video. The recording was saved to {path}, "
                "but remove the current queued video first (or switch to Batch mode) to add it."
            )
            return
        entry = self._probe_video(path)
        if entry is None:
            QMessageBox.critical(
                self, "Could not read recording",
                f"The recording was saved to {path} but couldn't be read back -- it may be corrupt."
            )
            return
        self.videos.append(entry)
        if self.active_index is None:
            self.active_index = len(self.videos) - 1
        self.setup_page.refresh_video_list()
        QMessageBox.information(
            self, "Recording added",
            f"Saved {os.path.basename(path)} and added it to the video queue."
        )

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
                "Define zones once (Draw Zones, or a Quick Setup apparatus tile) on the active "
                "video before aligning individual videos in the queue -- each video then starts "
                "from that same shape and you just nudge its corners to fit."
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

    def get_subject_row(self, subject_id):
        """The full imported subject-database row for subject_id (matching
        on whichever column _subject_id_key treats as the ID), or {} if no
        subject_id was given or it isn't in self.subjects. self.subjects'
        schema isn't fixed (see its comment in __init__) -- a researcher's
        spreadsheet might have Code/Group/Color/Gender/Age/Genotype/
        Phenotype/Treatment/Dose (SMART's own optional columns), some
        subset of those, or entirely different columns of their own choosing
        -- so this returns whatever that row actually has, unfiltered, and
        callers (see the Batch Standard Tracking summary-building loop) fold
        it into their own output generically rather than only ever reading
        a fixed list of expected column names."""
        if not subject_id:
            return {}
        for row in self.subjects:
            key = self._subject_id_key(row)
            if key and row.get(key) == subject_id:
                return row
        return {}

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
        self.pending_zone_lines = []
        self._roi_points_source = None
        self.setup_page.refresh_canvas()

    # ------------------------------------------------------------------
    # Embedded operations: Crop / Mask / Zones / Zone Outline / Objects /
    # Distance (mirrors TrackerApp._start_op/_cancel_op/_finish_op and the
    # canvas mouse-event handlers in gui/main_window.py). The _op dict's
    # points are always stored in FRAME-pixel space (the same space as
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
            # Typing names first is still supported (backward compatible),
            # but no longer required: leave the field blank and Draw Zones
            # starts with one ready-to-draw zone instead -- select its line
            # (or box/oval outline), and you're asked to name it the moment
            # you finish drawing it (see _maybe_prompt_new_zone_name), one
            # zone at a time, instead of typing every name up front.
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
        elif kind == "zone_lines":
            op["shapes"] = [list(p) for p in self.pending_zone_lines] if self.pending_zone_lines else [[]]
        elif kind in ("zones", "objects"):
            # Draw Zones only resumes PREVIOUSLY HAND-DRAWN zones. If the
            # current pending_roi_points instead came from Draw Zone Outline's
            # auto-detected arms/zones (self._roi_points_source == "lines"),
            # treat this as switching tools to start fresh by hand rather
            # than silently loading those many-point auto-traced shapes in
            # under their auto-picked A/B/C names -- see _roi_points_source's
            # own comment in __init__. roi_names_entry (and so `names`
            # above) is left alone either way, so the same names can just
            # be redrawn by hand if that's what's wanted.
            if kind == "zones" and self._roi_points_source == "lines":
                existing = {}
            else:
                existing = self.pending_roi_points if kind == "zones" else self.pending_object_points
            op["regions"] = {n: list(existing.get(n, [])) for n in names}
            # auto_named: zones that don't have a real, typed name yet --
            # either because roi_names_entry was left blank (below) or
            # because op_new_zone added one -- get asked for their name the
            # instant their shape is finished (see _maybe_prompt_new_zone_name).
            # A zone typed into roi_names_entry already has its real name,
            # so it's never in this set and is left alone, same as before.
            op["auto_named"] = set()
            if kind == "zones" and not op["regions"]:
                first_name = self._next_auto_zone_name(op)
                op["regions"][first_name] = []
                op["auto_named"].add(first_name)
            op["active_region"] = next(iter(op["regions"]), None)
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
        self.setup_page.reset_canvas_title()
        self.setup_page.refresh_canvas()

    def cancel_op(self):
        self._op = None
        self.set_active_tool(None)
        self.setup_page.hide_op_bar()
        self.setup_page.reset_canvas_title()
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

        elif op["kind"] == "zone_lines":
            # Trace the WHOLE apparatus's outer outline as one closed shape
            # (occasionally a couple of disconnected ones -- "New Shape" --
            # for an apparatus split across the frame) and let
            # detect_apparatus_partitions work out on its own how many
            # arms/partitions it naturally divides into -- see that
            # function's own docstring. Each shape needs 3+ points to
            # enclose any area at all, same as every other closed
            # hand-drawn outline in this app (Mask Zone, Draw Zones).
            outline_shapes = [s for s in op["shapes"] if len(s) >= 3]
            if not outline_shapes:
                QMessageBox.critical(self, "Not done", "Trace the apparatus's outer outline first.")
                return
            partitions, err = detect_apparatus_partitions(outline_shapes, self.pending_warp_w, self.pending_warp_h)
            if err:
                QMessageBox.critical(self, "Could not detect zones", err)
                return
            self.pending_zone_lines = outline_shapes
            self.pending_roi_points = partitions
            self._roi_points_source = "lines"
            self.setup_page.set_roi_names_text(", ".join(partitions.keys()))
            self.status_label.setText(
                f"Detected {len(partitions)} zone(s): {', '.join(partitions.keys())} -- "
                "name them in the Zone Formula box."
            )

        elif op["kind"] in ("zones", "objects"):
            # Drop any auto-named placeholder that was never actually drawn
            # into -- e.g. the empty "Zone 3" left ready-to-draw after the
            # last real one was finished (see _maybe_prompt_new_zone_name,
            # which advances to a fresh blank zone after every completed
            # shape). That's just bookkeeping for "ready in case you want
            # another one", not a half-finished zone to warn about.
            auto_named = op.get("auto_named", set())
            for empty_name in [n for n, pts in op["regions"].items() if not pts and n in auto_named]:
                del op["regions"][empty_name]
            incomplete = [n for n, pts in op["regions"].items() if len(pts) < 3]
            if incomplete:
                noun = "zones" if op["kind"] == "zones" else "objects"
                QMessageBox.critical(self, "Not done", f"These {noun} still need 3+ points: {', '.join(incomplete)}")
                return
            if op["kind"] == "zones":
                self.pending_roi_points = dict(op["regions"])
                self._roi_points_source = "manual"
                # roi_names_entry is read elsewhere (CSV column naming, arm
                # entries/alternation) rather than from pending_roi_points
                # directly -- keep it in sync with whatever names were
                # actually used, same as the template_zones branch below.
                # Matters most for the blank-ROI-names-entry flow (draw
                # first, name each zone as you finish it): without this,
                # the field would stay empty even though real zones exist.
                self.setup_page.set_roi_names_text(", ".join(op["regions"].keys()))
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
                self._roi_points_source = "manual"
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
        if self._op and self._op["kind"] in ("mask", "zone_lines"):
            self._op["shapes"].append([])
            self.setup_page.update_op_instructions()
            self.setup_page.refresh_canvas()

    def _next_auto_zone_name(self, op):
        """The name to pre-fill for the next not-yet-drawn zone -- a plain
        'Zone N', counting past whatever's already in this op's regions."""
        n = 1
        while f"Zone {n}" in op["regions"]:
            n += 1
        return f"Zone {n}"

    def op_new_zone(self):
        """Draw Zones: add another empty, auto-named zone (see
        _next_auto_zone_name -- either a template's next suggested name, or
        a plain 'Zone N') without needing to type its real name in the ROI
        names field first -- trace its outline (Rectangle/Ellipse/Line
        finish it and ask for the real name right away -- see
        _maybe_prompt_new_zone_name -- freehand instead waits for a click
        INSIDE it afterwards, or Finish) instead of committing to a name
        before anything is on screen. Also called automatically after each
        shape drawn with the Rectangle/Ellipse/Line tools finishes, so a
        row of lines/boxes can be drawn back-to-back without reaching for
        this button every time."""
        if not (self._op and self._op["kind"] == "zones"):
            return
        new_name = self._next_auto_zone_name(self._op)
        self._op["regions"][new_name] = []
        self._op["active_region"] = new_name
        self._op.setdefault("auto_named", set()).add(new_name)
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
        elif op["kind"] == "zone_lines":
            n_shapes = len(op["shapes"])
            n_pts = len(op["shapes"][-1])
            return (f"Shape {n_shapes} ({n_pts} pts, need 3+). Click to add points, tracing the "
                    "WHOLE apparatus's own outer outline -- 'New Shape' only if it's split into "
                    "separate pieces on screen, then Finish to auto-detect and letter its arms/zones.")
        elif op["kind"] == "distance":
            return f"Click 2 points of known real-world distance ({len(op['points'])}/2 placed)."
        elif op["kind"] == "template_zones":
            # Only reachable from on_align_video (Batch queue's per-video
            # "Align" button).
            return (f"{len(op['regions'])} zone(s) from the template. Drag corners to fit THIS video's "
                    "own framing, then Finish to save just this video's alignment (other queued videos "
                    "are unaffected).")
        return ""

    # -- "template_zones": a drag-corners-only op kind, used by
    # on_align_video (Batch queue's per-video "Align" button) to nudge an
    # ALREADY-drawn/named set of zones to fit a different video's framing. --

    def prompt_zone_label(self, op, current_name):
        """Popup shown after clicking a just-finished zone on the canvas
        (or a template_zones align target): confirm the auto-suggested
        name, pick a different existing one, or type a custom name."""
        existing_names = set(op["regions"].keys()) - {current_name}
        from qt_app.dialogs.zone_label_dialog import prompt_zone_label as _prompt
        return _prompt(self, current_name, [], existing_names)

    def _maybe_prompt_new_zone_name(self, op, name):
        """Draw Zones, Rectangle/Ellipse/Line tools only: called right after
        a shape finishes (mouse released -- see on_canvas_release), this is
        what makes 'select the shape's 1-2 points, then name it on the
        spot' work -- MM asked for this after sketching a maze frame with
        lines drawn straight onto it, each meant to be named the moment
        it's placed, rather than typing every zone's name up front or
        clicking back into a finished shape later to rename it.

        Only fires for a zone that doesn't have a real name yet (still in
        op['auto_named'] -- see start_op/op_new_zone); a zone typed into
        roi_names_entry already has its real name and is left alone.
        Freehand zones are a multi-click shape with no single 'done'
        moment, so they keep the older click-inside/Finish-then-rename
        flow instead of auto-prompting after every click."""
        pts = op["regions"].get(name, [])
        if len(pts) < 3 or name not in op.get("auto_named", set()):
            return
        new_name = self.prompt_zone_label(op, name)
        if new_name and new_name != name:
            op["regions"][new_name] = op["regions"].pop(name)
            if op["active_region"] == name:
                op["active_region"] = new_name
            op["auto_named"].discard(name)
        self.setup_page.rebuild_region_buttons()
        # Ready for the next line/box/oval immediately -- drawing several
        # zones in a row (an EPM's arms + center, say) shouldn't need a
        # fresh "New Zone" click after every single one.
        self.op_new_zone()

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

        elif op["kind"] in ("mask", "zone_lines"):
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
                # zone opens the confirm/rename popup for that zone instead
                # of adding a point to the active one -- draw all the
                # shapes first, then click each to name it, rather than
                # typing every name up front.
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
        finished_shape_name = None
        if op.get("drag") and op["drag"][0] == "newshape":
            finished_shape_name = op["drag"][1]
            op.pop("_shape_anchor", None)
        op["drag"] = None
        if finished_shape_name is not None and op["kind"] == "zones":
            # Rectangle/Ellipse/Line only (freehand never sets a "newshape"
            # drag -- on_canvas_press adds points one at a time for it
            # instead) -- see _maybe_prompt_new_zone_name.
            self._maybe_prompt_new_zone_name(op, finished_shape_name)

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
        self.pending_zone_lines = []
        self._roi_points_source = None
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
        # Re-seed from the persisted Settings screen (if the researcher has
        # ever saved one) rather than leaving _memory truly empty -- "reset
        # to defaults" means back to THEIR configured defaults when they
        # have any, same as a fresh launch does (see __init__); {} when
        # they never have, so this is a no-op exactly like before this
        # feature existed.
        self._memory.update(load_settings())
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
        from qt_app.dialogs.settings_dialog import open_settings_dialog
        open_settings_dialog(self, self)

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

    def on_live_tracking_frame(self, frame_bgr):
        """frame_callback for process_single_video (see the comment above
        its call in _run_standard_flow for why this replaces cv2.imshow):
        draws the same annotated AUTO TRACK frame straight into the Setup
        page's own preview canvas every few frames while tracking runs, so
        MM can watch the dot follow the animal live instead of only a
        percentage bar (his "i need to see the screen of tracking" ask).
        Safe to no-op if the canvas isn't around for some reason (e.g. this
        ran from a script/test with no Setup page built)."""
        sp = getattr(self, "setup_page", None)
        if sp is None or getattr(sp, "canvas", None) is None:
            return
        sp.canvas_title_label.setText("Preview / Calibration -- Live tracking")
        sp.canvas.display_frame(frame_bgr)
        QApplication.processEvents()

    def on_preview_background(self):
        """'Preview Background' toolbar button (Draw Zones toolbar): MM's
        "i need reference image to display and empty frame image to see
        and confirm" ask. Computes the SAME median background image
        process_single_video will use for motion detection, and grabs one
        real frame (with the animal in it, from partway through the chosen
        time window) as a reference alongside it, then shows both side by
        side -- with the current zones/mask/objects drawn on top of each,
        exactly like the real run -- in the preview canvas, so a bad
        background (e.g. a mouse-shaped smudge baked in because it sat
        still too long) is obvious before spending time on a full run,
        without ever needing the cv2.imshow confirm step that used to gate
        this and crashes this Qt build (see the comment above the
        Individual-mode process_single_video call for why).
        """
        if self.active_index is None or not self.videos:
            QMessageBox.critical(self, "No video", "Add and select a video first.")
            return
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        video_path = self.videos[self.active_index]["path"]
        setup = self._build_setup(video_path)
        if setup is None:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Could not open video", f"Could not open:\n{video_path}")
            return
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps else 0.0
            start_time = max(0.0, min(setup["start_time"], duration))
            end_time = max(start_time, min(setup["end_time"], duration))
            start_frame = int(start_time * fps)
            end_frame = max(start_frame, int(end_time * fps) - 1)

            self.status_label.setText("Computing background...")
            QApplication.processEvents()
            background = make_background(
                cap, start_frame, end_frame, setup["matrix"], setup["warp_w"], setup["warp_h"],
                setup["background_samples"], color_mode=setup["color_mode"],
            )

            mid_frame = (start_frame + end_frame) // 2
            reference = read_and_warp(cap, mid_frame, setup["matrix"], setup["warp_w"], setup["warp_h"])
        finally:
            cap.release()

        if background is None or reference is None:
            QMessageBox.critical(self, "Preview failed",
                                  "Couldn't read frames from this video to build a preview.")
            self.status_label.setText("Idle.")
            return

        # Reuse the exact same overlay drawing the live canvas uses, so the
        # zones/mask/objects shown here line up pixel-for-pixel with what
        # the real run will see -- draw_overlays reads pending_roi_points
        # etc. straight off self (MainWindow), same as PreviewCanvas.refresh().
        # make_background() returns a single-channel image in "gray" color
        # mode (the common/default case) -- draw_overlays/cv2 drawing calls
        # and the RGB conversion in display_frame all expect 3 channels.
        if background.ndim == 2:
            background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        ref_annotated = draw_overlays(reference, self)
        bg_annotated = draw_overlays(background, self)
        composed = self._side_by_side_preview(
            ref_annotated, "REFERENCE (with animal)", bg_annotated, "BACKGROUND (should be empty)"
        )

        self.setup_page.canvas_title_label.setText(
            "Preview / Calibration -- Reference vs. background (click any tool to go back)"
        )
        self.setup_page.canvas.display_frame(composed)
        self.status_label.setText(
            "Background preview ready. If a mouse-shaped smudge shows on the right, raise "
            "'Background samples' or pick a time window where the animal moves more, then "
            "preview again."
        )

    @staticmethod
    def _side_by_side_preview(left_bgr, left_label, right_bgr, right_label):
        """Stacks two same-sized BGR frames horizontally with a small
        caption baked into each half, for on_preview_background -- lets the
        existing single-frame preview canvas show two images to compare at
        once without adding a whole second widget just for this."""
        h = max(left_bgr.shape[0], right_bgr.shape[0])
        w = max(left_bgr.shape[1], right_bgr.shape[1])

        def _pad(frame):
            if frame.shape[:2] == (h, w):
                return frame.copy()
            canvas = np.zeros((h, w, 3), dtype=frame.dtype)
            fh, fw = frame.shape[:2]
            canvas[:fh, :fw] = frame
            return canvas

        left = _pad(left_bgr)
        right = _pad(right_bgr)
        gap = np.full((h, 6, 3), 40, dtype=np.uint8)
        combined = np.hstack([left, gap, right])

        for label, x_off in ((left_label, 10), (right_label, w + gap.shape[1] + 10)):
            cv2.putText(combined, label, (x_off, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(combined, label, (x_off, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return combined

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

            # Stop condition (optional) -- only present in the Standard-
            # Tracking Setup body (see setup_page._build_zone_body), so
            # falls back to "no stop condition" for multi-mouse callers of
            # this same method.
            stop_combo = getattr(sp, "stop_condition_combo", None)
            stop_label = stop_combo.currentText() if stop_combo is not None else "None"
            stop_condition = STOP_CONDITION_CODES.get(stop_label)
            stop_value_raw = sp.stop_value_entry.text().strip() \
                if getattr(sp, "stop_value_entry", None) is not None else ""
            stop_value = float(stop_value_raw) if (stop_condition is not None and stop_value_raw) else None
            stop_zone_name = sp.stop_zone_entry.text().strip() \
                if getattr(sp, "stop_zone_entry", None) is not None else ""
            setup["stop_condition"] = stop_condition
            setup["stop_value"] = stop_value
            setup["stop_zone_name"] = stop_zone_name

            # Zone Associations (optional) -- same "only in the Standard-
            # Tracking Setup body" caveat as stop_condition above. A
            # malformed box is treated as no groups here (on_start's own
            # validation is what actually blocks Start with a message);
            # this keeps _build_setup itself infallible for callers like
            # on_preview_background that never run that validation.
            zone_assoc_entry = getattr(sp, "zone_associations_entry", None)
            zone_groups, _zone_assoc_err = parse_zone_associations(
                zone_assoc_entry.text() if zone_assoc_entry is not None else ""
            )
            # Zone Formula (optional) -- Draw Zone Outline's auto-detected
            # A/B/C/... arms/zones, named/merged here ("A = Center", "B+C =
            # Left Arm"). Feeds into the SAME zone_groups reporting
            # machinery as Zone Associations above -- a formula name is
            # just a group made of one or more lettered zones instead of
            # hand-drawn zones. Same "infallible here, validated in
            # on_start" treatment as Zone Associations.
            zone_formula_entry = getattr(sp, "zone_formula_entry", None)
            partition_groups, _zone_formula_err = parse_partition_formula(
                zone_formula_entry.text() if zone_formula_entry is not None else ""
            )
            zone_groups = {**zone_groups, **partition_groups}
            setup["zone_groups"] = zone_groups

            # Trajectory smoothing (optional) -- same "Standard-Tracking-
            # only Setup body" caveat as the two above.
            smooth_var = getattr(sp, "smooth_trajectory_var", None)
            smooth_window = None
            if smooth_var is not None and smooth_var.isChecked():
                smooth_window_raw = sp.smooth_window_entry.text().strip() \
                    if getattr(sp, "smooth_window_entry", None) is not None else ""
                smooth_window = int(float(smooth_window_raw)) if smooth_window_raw else None
            setup["smooth_window"] = smooth_window

            # Custom Variables (optional) -- same "Standard-Tracking-only
            # Setup body" caveat as the three above. A malformed box is
            # treated as no definitions here (on_start's own validation is
            # what actually blocks Start with a message), same reasoning
            # as Zone Associations above.
            custom_vars_entry = getattr(sp, "custom_variables_entry", None)
            custom_var_defs, _custom_var_err = parse_custom_variables(
                custom_vars_entry.toPlainText() if custom_vars_entry is not None else ""
            )
            setup["custom_variables"] = custom_var_defs

            # Time Bins (optional bin size, Upgrade Plan Tier 2 #9) --
            # this field lives in _detection_settings (shared by all three
            # analysis types, see setup_page.py), so it's always present
            # here, unlike the Standard-Tracking-only fields above.
            bin_entry = getattr(sp, "bin_size_entry", None)
            bin_raw = bin_entry.text().strip() if bin_entry is not None else ""
            setup["bin_size_s"] = float(bin_raw) if bin_raw else 60.0
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

        stop_combo = getattr(sp, "stop_condition_combo", None)
        if stop_combo is not None:
            stop_label = stop_combo.currentText()
            stop_condition = STOP_CONDITION_CODES.get(stop_label)
            if stop_condition is not None:
                value_raw = sp.stop_value_entry.text().strip() if getattr(sp, "stop_value_entry", None) else ""
                try:
                    value_ok = value_raw != "" and float(value_raw) > 0
                except ValueError:
                    value_ok = False
                if not value_ok:
                    QMessageBox.critical(
                        self, "Invalid stop condition",
                        f"Enter a positive number for the '{stop_label}' stop condition's value."
                    )
                    return
                if stop_condition == "zone_entries":
                    zone_name = sp.stop_zone_entry.text().strip() if getattr(sp, "stop_zone_entry", None) else ""
                    known_zones = [n.strip() for n in sp.roi_names_entry.text().split(",") if n.strip()] \
                        if getattr(sp, "roi_names_entry", None) else []
                    if not zone_name:
                        QMessageBox.critical(
                            self, "Invalid stop condition",
                            "Enter the zone name to count entries into (or draw/name a zone first)."
                        )
                        return
                    if known_zones and zone_name not in known_zones:
                        QMessageBox.critical(
                            self, "Invalid stop condition",
                            f"'{zone_name}' isn't one of the current zone names ({', '.join(known_zones)})."
                        )
                        return

        zone_assoc_entry = getattr(sp, "zone_associations_entry", None)
        if zone_assoc_entry is not None and zone_assoc_entry.text().strip():
            zone_groups, zone_assoc_err = parse_zone_associations(zone_assoc_entry.text())
            if zone_assoc_err is not None:
                QMessageBox.critical(self, "Invalid Zone Associations", zone_assoc_err)
                return
            known_zones = [n.strip() for n in sp.roi_names_entry.text().split(",") if n.strip()] \
                if getattr(sp, "roi_names_entry", None) else []
            for group_name, members in zone_groups.items():
                if known_zones and group_name in known_zones:
                    QMessageBox.critical(
                        self, "Invalid Zone Associations",
                        f"'{group_name}' is already one of your real zone names -- pick a different "
                        "name for the combined zone."
                    )
                    return
                unknown = [m for m in members if known_zones and m not in known_zones]
                if unknown:
                    QMessageBox.critical(
                        self, "Invalid Zone Associations",
                        f"'{group_name}' references zone(s) that don't exist: {', '.join(unknown)} "
                        f"(current zones: {', '.join(known_zones) if known_zones else '(none)'})."
                    )
                    return

        zone_formula_entry = getattr(sp, "zone_formula_entry", None)
        if zone_formula_entry is not None and zone_formula_entry.text().strip():
            partition_groups, zone_formula_err = parse_partition_formula(zone_formula_entry.text())
            if zone_formula_err is not None:
                QMessageBox.critical(self, "Invalid Zone Formula", zone_formula_err)
                return
            known_zones = [n.strip() for n in sp.roi_names_entry.text().split(",") if n.strip()] \
                if getattr(sp, "roi_names_entry", None) else []
            for group_name, members in partition_groups.items():
                unknown = [m for m in members if known_zones and m not in known_zones]
                if unknown:
                    QMessageBox.critical(
                        self, "Invalid Zone Formula",
                        f"'{group_name}' references a lettered zone that wasn't detected: "
                        f"{', '.join(unknown)} (current zones: "
                        f"{', '.join(known_zones) if known_zones else '(none)'}). "
                        "Draw Zone Outline first, then name the lettered zones here."
                    )
                    return

        smooth_var = getattr(sp, "smooth_trajectory_var", None)
        if smooth_var is not None and smooth_var.isChecked():
            window_raw = sp.smooth_window_entry.text().strip() if getattr(sp, "smooth_window_entry", None) else ""
            try:
                window_ok = window_raw != "" and int(float(window_raw)) >= 2
            except ValueError:
                window_ok = False
            if not window_ok:
                QMessageBox.critical(
                    self, "Invalid trajectory smoothing window",
                    "Enter a whole number of 2 or more frames for the smoothing window."
                )
                return

        custom_vars_entry = getattr(sp, "custom_variables_entry", None)
        if custom_vars_entry is not None and custom_vars_entry.toPlainText().strip():
            _custom_var_defs, custom_var_err = parse_custom_variables(custom_vars_entry.toPlainText())
            if custom_var_err is not None:
                QMessageBox.critical(self, "Invalid Custom Variables", custom_var_err)
                return

        bin_entry = getattr(sp, "bin_size_entry", None)
        if bin_entry is not None and bin_entry.text().strip():
            try:
                bin_ok = float(bin_entry.text()) > 0
            except ValueError:
                bin_ok = False
            if not bin_ok:
                QMessageBox.critical(self, "Invalid time bin size",
                                      "Enter a positive number of seconds for the time bin size "
                                      "(or leave it blank for the default 1-minute bins).")
                return

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
                # cv2.imshow "confirm the background"/"press a key to
                # advance the detection preview" steps that show_display=True
                # used to gate are skipped -- but not lost: "Preview
                # Background" (on_preview_background, below) covers the
                # pre-run confirmation in a plain Qt image instead, and
                # frame_callback=self.on_live_tracking_frame draws the same
                # live AUTO TRACK overlay this cv2 window would have shown,
                # straight into the preview canvas, without ever touching
                # cv2.imshow. run_detection_preview also still saves every
                # preview frame to disk unconditionally, and the Results
                # page's Reference Frames gallery below is exactly those.
                summary = process_single_video(active_path, base_setup, show_display=False,
                                               progress_callback=self.on_progress,
                                               confirm_callback=self._qt_confirm,
                                               frame_callback=self.on_live_tracking_frame)
            except SystemExit as exc:
                self.status_label.setText(f"Stopped: {exc}")
                cv2.destroyAllWindows()
                self.setup_page.reset_canvas_title()
                return
            except Exception as exc:
                self.status_label.setText(f"Error: {type(exc).__name__}")
                cv2.destroyAllWindows()
                self.setup_page.reset_canvas_title()
                QMessageBox.critical(self, "Tracking Error",
                                      f"An error occurred during tracking:\n\n{type(exc).__name__}: {exc}")
                return
            self.progress_bar.setValue(100)
            self.status_label.setText("Done.")
            self.setup_page.reset_canvas_title()
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
                # batch run N times over. frame_callback still draws a live
                # view into the preview canvas for whichever video is
                # currently processing, same as Individual mode.
                summary = process_single_video(v["path"], setup, show_display=False,
                                               progress_callback=self.on_progress,
                                               confirm_callback=self._qt_confirm,
                                               frame_callback=self.on_live_tracking_frame)
                summary["subject_id"] = v.get("subject_id", "")
                # Richer Subject Database fields (Upgrade Plan Tier 1 #2,
                # mirrors SMART's Code/Group/Color/Gender/Age/Genotype/
                # Phenotype/Treatment/Dose columns) -- carry every OTHER
                # column this subject's imported row has (whatever they
                # are; see get_subject_row's own comment) into the batch
                # summary too, "Subject_"-prefixed so they can't collide
                # with an existing summary column, instead of just the
                # bare subject_id string as before.
                subject_row = self.get_subject_row(summary["subject_id"])
                id_key = self._subject_id_key(subject_row) if subject_row else None
                for col, val in subject_row.items():
                    if col == id_key:
                        continue  # already carried above as subject_id -- don't duplicate it
                    summary[f"Subject_{col}"] = val
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
        self.setup_page.reset_canvas_title()
        self._finish_batch_run(summaries, errors, "BatchSummary")

    def _finish_batch_run(self, summaries, errors, base_filename, processed_count=None):
        """Shared tail end of every analysis type's batch loop (Standard
        Tracking/Multi-Mouse Tracking/Behavior Classification -- Upgrade
        Plan Tier 1 #5 extended batch mode to the latter two, which used to
        only ever process the active video regardless of the Individual/
        Batch radio). Writes `summaries` (a list of flat per-row dicts --
        one row per video for Standard/Behavior, one row per (video,
        mouse) for Multi-Mouse) to <base_filename>.csv/.xlsx next to the
        first queued video, and shows the same 'Batch Complete' dialog
        shape all three used individually before this was extracted.
        processed_count is the number of VIDEOS (not summary ROWS --
        Multi-Mouse can have several rows per video) successfully
        processed; defaults to len(summaries), correct whenever each video
        contributes exactly one row."""
        if processed_count is None:
            processed_count = len(summaries)
        if summaries:
            batch_df = pd.DataFrame(summaries)
            folder = os.path.dirname(self.videos[0]["path"])
            batch_path = os.path.join(folder, f"{base_filename}.csv")
            batch_xlsx_path = os.path.join(folder, f"{base_filename}.xlsx")
            try:
                batch_df.to_csv(batch_path, index=False)
                batch_df.to_excel(batch_xlsx_path, index=False, engine="openpyxl")
            except Exception as exc:
                QMessageBox.critical(self, "Save Error", f"Could not save batch summary:\n\n{exc}")
            self.status_label.setText("Batch complete.")
            msg = (f"Processed {processed_count}/{len(self.videos)} videos.\n\n"
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

    def _process_one_multi_mouse(self, video_path, roi_points_override=None, status_prefix=""):
        """Runs the full Multi-Mouse Tracking pipeline for ONE video and
        returns its results dict -- the exact body _run_multi_mouse_flow
        used to run inline for only ever the active video, now shared by
        Individual mode (one call) and Batch mode (one call per queued
        video, see _run_multi_mouse_flow below). roi_points_override, when
        given, is a video's own saved 'Align' override (see
        on_align_video); otherwise falls back to the shared
        pending_roi_points template, same as before. Raises ValueError for
        a bad Detection Settings value, or whatever track_video/
        compute_zone_interaction_stats itself raises otherwise -- the
        caller decides how to report it (a dialog for a single video, a
        per-video error list for a batch)."""
        sp = self.setup_page
        tmp_path = None
        try:
            try:
                min_area = int(float(sp.min_area_entry.text()))
                max_area = int(float(sp.max_area_entry.text()))
                threshold = int(float(sp.threshold_entry.text()))
                bg_samples = int(float(sp.bg_samples_entry.text()))
                bin_size_s = float(sp.bin_size_entry.text()) if getattr(sp, "bin_size_entry", None) \
                    and sp.bin_size_entry.text().strip() else 60.0
            except ValueError:
                raise ValueError("Check the Detection Settings values.")

            self.status_label.setText(f"{status_prefix}Preparing video...")
            QApplication.processEvents()

            source_path, tmp_path, fps = self._prepare_source_video(video_path)
            output_dir = compute_output_dir(video_path)
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

            self.status_label.setText(f"{status_prefix}Saving reference frames...")
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

            self.status_label.setText(f"{status_prefix}Tracking (multi-mouse)...")
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
                if not sp.loc_var.isChecked():
                    roi_to_use = {}
                elif roi_points_override is not None:
                    roi_to_use = roi_points_override
                else:
                    roi_to_use = self.pending_roi_points
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

            # Time Bins (optional bin size, Upgrade Plan Tier 2 #9) --
            # extends Standard Tracking's own configurable time-bin report
            # to Multi-Mouse: per (mouse_id, bin), distance traveled and
            # frames tracked (see tracking.two_mouse.calculate_time_bins's
            # own docstring for why distance rather than time-in-zone).
            time_bins_csv = None
            try:
                time_bins_df = calculate_time_bins(tracks_df, fps, bin_size_s=bin_size_s)
                if len(time_bins_df):
                    time_bins_csv = os.path.join(output_dir, "time_bins.csv")
                    time_bins_df.to_csv(time_bins_csv, index=False)
            except Exception as exc:
                print(f"[warn] time bins: {exc}")

            return {
                "analysis_type": "multi_mouse", "video_path": video_path, "output_dir": output_dir,
                "tracks_df": tracks_df if tracks_df is not None else pd.DataFrame(),
                "tracks_csv": tracks_csv, "annotate_path": annotate_path,
                "fps": fps, "per_mouse_summary": per_mouse_summary, "per_mouse_bouts": per_mouse_bouts,
                "summary_csv": summary_csv, "bouts_csv": bouts_csv, "preview_paths": preview_paths,
                "time_bins_csv": time_bins_csv,
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _run_multi_mouse_flow(self):
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        self.progress_bar.setValue(0)
        QApplication.processEvents()

        if self.get_mode() == "individual":
            active_path = self.videos[self.active_index]["path"]
            try:
                results = self._process_one_multi_mouse(active_path)
            except SystemExit as exc:
                self.status_label.setText(f"Stopped: {exc}")
                cv2.destroyAllWindows()
                return
            except Exception as exc:
                self.status_label.setText(f"Error: {type(exc).__name__}")
                cv2.destroyAllWindows()
                QMessageBox.critical(self, "Multi-Mouse Error", f"An error occurred:\n\n{type(exc).__name__}: {exc}")
                return
            self.progress_bar.setValue(100)
            self.status_label.setText("Done.")
            try:
                self.results_page.show_multi_mouse(results)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results Error", f"Could not display results:\n\n{type(exc).__name__}: {exc}")
            return

        # ---- Batch mode (Upgrade Plan Tier 1 #5) -- same per-video
        # override confirmation _run_standard_flow's batch already used,
        # reused verbatim since roi_points_override lives on the video
        # dict regardless of analysis type.
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

        summaries = []
        errors = []
        processed_count = 0
        for i, v in enumerate(self.videos):
            print(f"\n[{i + 1}/{len(self.videos)}] Processing: {v['path']}")
            prefix = f"Batch {i + 1}/{len(self.videos)}: "
            self.status_label.setText(f"{prefix}{os.path.basename(v['path'])}")
            self.progress_bar.setValue(0)
            self.progress_label.setText("")
            QApplication.processEvents()
            try:
                results = self._process_one_multi_mouse(
                    v["path"], roi_points_override=v.get("roi_points_override"), status_prefix=prefix
                )
                subject_id = v.get("subject_id", "")
                subject_row = self.get_subject_row(subject_id)
                id_key = self._subject_id_key(subject_row) if subject_row else None
                base_row = {"video": os.path.basename(v["path"]), "subject_id": subject_id}
                for col, val in subject_row.items():
                    if col == id_key:
                        continue
                    base_row[f"Subject_{col}"] = val
                per_mouse_summary = results.get("per_mouse_summary", {})
                if per_mouse_summary:
                    # One row per (video, mouse) -- a queued video can hold
                    # several animals, so BatchSummary_MultiMouse can't
                    # follow Standard/Behavior's one-row-per-video shape.
                    for mouse_id, mouse_summary in per_mouse_summary.items():
                        row = dict(base_row)
                        row["mouse_id"] = mouse_id
                        row.update(mouse_summary)
                        summaries.append(row)
                else:
                    # Still record that this video ran even with no zone/
                    # object stats (Location Tracking and Interaction
                    # Tracking both off) -- otherwise a successfully-
                    # tracked video would silently vanish from the batch
                    # summary entirely.
                    row = dict(base_row)
                    row["mouse_id"] = ""
                    row["Tracked_rows"] = len(results.get("tracks_df", []))
                    summaries.append(row)
                processed_count += 1
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
        self._finish_batch_run(summaries, errors, "BatchSummary_MultiMouse", processed_count=processed_count)

    # ------------------------------------------------------------------
    # Behavior Classification (rule-based; mirrors
    # TrackerApp._run_behavior_flow -- the ML-model branch is a separate,
    # not-yet-wired-up path, same as the Prepare/Train buttons)
    # ------------------------------------------------------------------

    def _process_one_behavior(self, video_path, status_prefix="", warn_if_empty=True):
        """Runs the full rule-based Behavior Classification pipeline for
        ONE video and returns its results dict -- the exact body
        _run_behavior_flow used to run inline for only ever the active
        video, now shared by Individual mode (one call) and Batch mode
        (one call per queued video, see _run_behavior_flow below). The
        ML-model path (_run_behavior_flow_ml) is untouched/out of scope --
        see its own docstring. warn_if_empty=False (batch mode) prints a
        console note instead of popping a blocking 'No behavior data'
        dialog once per empty video. Raises ValueError for a bad Detection
        Settings value, or whatever extract_features/classify_behaviors
        itself raises otherwise -- the caller decides how to report it."""
        sp = self.setup_page
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
                bin_size_s = float(sp.bin_size_entry.text()) if getattr(sp, "bin_size_entry", None) \
                    and sp.bin_size_entry.text().strip() else 60.0
            except (ValueError, AttributeError, RuntimeError):
                raise ValueError("Check the Detection Settings values.")

            self.status_label.setText(f"{status_prefix}Preparing video...")
            QApplication.processEvents()

            source_path, tmp_path, fps = self._prepare_source_video(video_path)
            output_dir = compute_output_dir(video_path)
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

            self.status_label.setText(f"{status_prefix}Saving reference frames...")
            QApplication.processEvents()
            try:
                preview_paths = save_preview_frames_behavior(
                    source_path, output_dir, n_samples=preview_n, num_animals=sp.num_animals,
                    min_area=min_area, max_area=max_area, diff_threshold=threshold,
                    n_background_samples=bg_samples, color_mode=color_mode,
                    max_jump_px=max_jump,
                )
            except Exception as exc:
                print(f"[warn] Reference frames: {exc}")
                preview_paths = []

            self.status_label.setText(f"{status_prefix}Extracting features...")
            QApplication.processEvents()

            extract_features(
                source_path, features_csv, num_animals=sp.num_animals,
                min_area=min_area, max_area=max_area, diff_threshold=threshold,
                n_background_samples=bg_samples, color_mode=color_mode,
                max_jump_px=max_jump, progress_callback=self.on_progress,
            )
            self.status_label.setText(f"{status_prefix}Classifying behaviors...")
            QApplication.processEvents()
            labeled_df, bouts_df = classify_behaviors(
                features_csv, bouts_csv, labeled_output_csv=labeled_csv,
                loco_body_move_high=loco_thresh, rear_score_threshold=rear_thresh,
                groom_score_threshold=groom_thresh,
                groom_min_persistence=persistence_frames, rear_min_persistence=persistence_frames,
                min_bout_s=min_bout_s,
            )

            if labeled_df is None or len(labeled_df) == 0:
                if warn_if_empty:
                    QMessageBox.warning(
                        self, "No behavior data",
                        "No frames could be classified. Try adjusting the detection threshold or area settings."
                    )
                else:
                    print(f"[warn] No frames could be classified for {os.path.basename(video_path)}.")

            selected = {
                "rearing": sp.behavior_rearing_var.isChecked(),
                "grooming": sp.behavior_grooming_var.isChecked(),
                "locomotion": sp.behavior_locomotion_var.isChecked(),
                "immobile": sp.behavior_immobile_var.isChecked(),
            }
            if (not any(selected.values())) or sp.behavior_all_var.isChecked():
                selected = {k: True for k in selected}

            # Time Bins (optional bin size, Upgrade Plan Tier 2 #9) --
            # extends Standard Tracking's own configurable time-bin report
            # to Behavior Classification: per (mouse_id, bin), the time/
            # percent spent in each classified behavior.
            time_bins_csv = None
            try:
                time_bins_df = calculate_behavior_time_bins(labeled_df, fps, bin_size_s=bin_size_s)
                if len(time_bins_df):
                    time_bins_csv = os.path.join(output_dir, "time_bins.csv")
                    time_bins_df.to_csv(time_bins_csv, index=False)
            except Exception as exc:
                print(f"[warn] time bins: {exc}")

            return {
                "analysis_type": "behavior", "video_path": video_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df, "selected_behaviors": selected,
                "bouts_csv": bouts_csv, "labeled_csv": labeled_csv, "fps": fps, "preview_paths": preview_paths,
                "time_bins_csv": time_bins_csv,
            }
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

    def _run_behavior_flow(self):
        if self.pending_matrix is None:
            QMessageBox.critical(self, "Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        sp = self.setup_page
        if getattr(sp, "ml_mode_var", None) is not None and sp.ml_mode_var.isChecked():
            # The ML-model path stays single-video/Individual-mode only --
            # see _run_behavior_flow_ml's own docstring; batch mode isn't
            # wired up for it (out of scope for Upgrade Plan Tier 1 #5,
            # which named the rule-based flow's batch loop specifically).
            active_path = self.videos[self.active_index]["path"]
            self._run_behavior_flow_ml(active_path)
            return

        self.progress_bar.setValue(0)
        QApplication.processEvents()

        if self.get_mode() == "individual":
            active_path = self.videos[self.active_index]["path"]
            try:
                results = self._process_one_behavior(active_path)
            except SystemExit:
                self.progress_bar.setValue(0)
                self.status_label.setText("Tracking stopped.")
                cv2.destroyAllWindows()
                return
            except Exception as exc:
                self.progress_bar.setValue(0)
                self.status_label.setText("Error occurred.")
                cv2.destroyAllWindows()
                QMessageBox.critical(self, "Tracking failed", str(exc))
                return
            self.progress_bar.setValue(100)
            self.status_label.setText("Done.")
            try:
                self.results_page.show_behavior(results)
                self.show_results_page()
            except Exception as exc:
                QMessageBox.critical(self, "Results display failed",
                                      f"Results were saved but could not be shown: {exc}")
            return

        # ---- Batch mode (Upgrade Plan Tier 1 #5) ----
        summaries = []
        errors = []
        processed_count = 0
        for i, v in enumerate(self.videos):
            print(f"\n[{i + 1}/{len(self.videos)}] Processing: {v['path']}")
            prefix = f"Batch {i + 1}/{len(self.videos)}: "
            self.status_label.setText(f"{prefix}{os.path.basename(v['path'])}")
            self.progress_bar.setValue(0)
            self.progress_label.setText("")
            QApplication.processEvents()
            try:
                results = self._process_one_behavior(v["path"], status_prefix=prefix, warn_if_empty=False)
                subject_id = v.get("subject_id", "")
                subject_row = self.get_subject_row(subject_id)
                id_key = self._subject_id_key(subject_row) if subject_row else None
                row = {"video": os.path.basename(v["path"]), "subject_id": subject_id}
                for col, val in subject_row.items():
                    if col == id_key:
                        continue
                    row[f"Subject_{col}"] = val
                labeled_df = results.get("labeled_df")
                bouts_df = results.get("bouts_df")
                row["Duration_s"] = float(labeled_df["time_s"].max()) \
                    if labeled_df is not None and len(labeled_df) else 0.0
                row["Total_bouts"] = int(len(bouts_df)) if bouts_df is not None else 0
                if bouts_df is not None and len(bouts_df) and "behavior" in bouts_df.columns:
                    agg = bouts_df.groupby("behavior")["duration_s"].agg(["count", "sum"])
                    for behavior, stats in agg.iterrows():
                        row[f"{behavior}_count"] = int(stats["count"])
                        row[f"{behavior}_total_s"] = float(stats["sum"])
                summaries.append(row)
                processed_count += 1
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
        self._finish_batch_run(summaries, errors, "BatchSummary_Behavior", processed_count=processed_count)

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
    # Dialogs (Manual Scoring, ML classifier)
    # ------------------------------------------------------------------

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
