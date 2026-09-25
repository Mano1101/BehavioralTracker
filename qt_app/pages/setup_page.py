"""
SetupPage -- the Qt equivalent of _build_zone_setup / _build_behavior_setup
in gui/main_window.py. Rebuilt whenever the analysis type changes, same as
the Tkinter body_container destroy/rebuild pattern; MainWindow.app holds
all the actual state (videos, pending_* calibration, _memory), this page
is just the view + the widgets that read/write it.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QCheckBox, QRadioButton, QGroupBox, QScrollArea,
    QFrame, QProgressBar, QSizePolicy, QMessageBox,
)

from qt_app.theme import PALETTE
from qt_app.widgets.preview_canvas import PreviewCanvas
from tracking.maze_templates import TEMPLATES as MAZE_TEMPLATES

# NOTE on theme reactivity: PALETTE is a single dict object that
# theme.set_dark()/set_light() mutate IN PLACE (clear+update), so every
# `PALETTE['MUTED']` lookup below always reflects the CURRENT theme as
# long as it's a fresh dict read at call time. Do NOT pre-extract a color
# into a module-level constant (e.g. `MUTED = PALETTE["MUTED"]`) -- that
# snapshots today's string once at import time and then never changes,
# which is exactly what used to make dark mode's hint-text stay
# light-mode-gray forever. Read PALETTE['MUTED'] fresh every time instead.


def _hint(text, wrap=True):
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9.5px;")
    if wrap:
        lbl.setWordWrap(True)
    return lbl


def _info_icon(text):
    """A small 'ⓘ' that shows a longer explanation as a hover tooltip,
    instead of a permanent paragraph of hint text taking up screen space.
    Used for the setup-page notes MM asked to declutter (still available
    on hover/tap -- not deleted, just not shown by default)."""
    lbl = QLabel("ⓘ")
    lbl.setToolTip(text)
    lbl.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 11px; font-weight: 700;")
    lbl.setCursor(Qt.WhatsThisCursor)
    return lbl


class SetupPage(QWidget):

    ENTRY_ATTRS = [
        "start_entry", "end_entry", "roi_names_entry", "object_names_entry",
        "interaction_margin_entry", "bg_samples_entry", "threshold_entry",
        "min_area_entry", "max_area_entry", "max_jump_entry",
        "window_size_entry", "window_weight_entry", "real_distance_entry",
        "units_entry", "preview_samples_entry",
        "loco_thresh_entry", "rear_thresh_entry", "groom_thresh_entry",
        "immobile_thresh_entry", "min_bout_entry",
    ]
    COMBO_ATTRS = ["output_size_entry"]
    CHECK_ATTRS = [
        "use_window_var", "use_zone_threshold_var", "reject_shadows_var",
        "loc_var", "interact_var", "entries_var", "altern_var", "all_var",
        "behavior_rearing_var", "behavior_grooming_var", "behavior_locomotion_var",
        "behavior_immobile_var", "behavior_all_var", "ml_mode_var",
    ]

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.setObjectName("pageBody")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.body_widget = QWidget()
        self.body_widget.setObjectName("pageBody")
        self.body_layout = QHBoxLayout(self.body_widget)
        self.body_layout.setContentsMargins(10, 10, 10, 10)
        self.body_layout.setSpacing(10)
        outer.addWidget(self.body_widget, 1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {PALETTE['BORDER']};")
        sep.setFixedHeight(1)
        outer.addWidget(sep)

        self.bottom_widget = QWidget()
        self.bottom_widget.setObjectName("pageBody")
        self.bottom_layout = QHBoxLayout(self.bottom_widget)
        self.bottom_layout.setContentsMargins(16, 12, 16, 12)
        outer.addWidget(self.bottom_widget)

        self.canvas = None
        self.rebuild(resave=False)

    # ------------------------------------------------------------------
    # Memory (mirrors TrackerApp._save_all_to_memory / _entry_or_default)
    # ------------------------------------------------------------------

    def save_all_to_memory(self, include_calibration=True):
        app = self.app
        for name in self.ENTRY_ATTRS:
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                app._memory[name] = w.text()
            except RuntimeError:
                pass
        for name in self.COMBO_ATTRS:
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                app._memory[name] = w.currentText()
            except RuntimeError:
                pass
        for name in self.CHECK_ATTRS:
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                app._memory[name] = w.isChecked()
            except RuntimeError:
                pass
        try:
            app._memory["color_mode_var"] = self.get_color_mode()
        except RuntimeError:
            pass
        if hasattr(self, "num_animals"):
            app._memory["num_animals"] = self.num_animals
        if include_calibration:
            for k in ("pending_use_crop", "pending_matrix", "pending_warp_w", "pending_warp_h",
                      "pending_crop_corners", "pending_roi_points", "pending_object_points",
                      "pending_mask_points", "pending_scale_factor", "pending_scale_unit"):
                if hasattr(app, k):
                    app._memory[k] = getattr(app, k)

    def entry_or_default(self, name, default):
        val = self.app._memory.get(name, None)
        if val not in (None, ""):
            return str(val)
        return str(default)

    def var_or_default(self, name, default):
        val = self.app._memory.get(name, None)
        if val is not None:
            return bool(val)
        return default

    def get_color_mode(self):
        for value, radio in getattr(self, "_color_mode_radios", {}).items():
            try:
                if radio.isChecked():
                    return value
            except RuntimeError:
                continue
        return "auto"

    # ------------------------------------------------------------------
    # Rebuild
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

    def rebuild(self, resave=True):
        # resave=False is used right after Open Project / Reset All have
        # just written fresh values into app._memory -- resaving here
        # would read the OLD (about-to-be-replaced) widgets and overwrite
        # those fresh values with whatever was on screen a moment ago,
        # same footgun as the Tkinter version's _build_setup_body(resave=).
        if resave:
            self.save_all_to_memory(include_calibration=False)

        self._clear_layout(self.body_layout)
        self._clear_layout(self.bottom_layout)
        self.canvas = None

        if self.app.analysis_type == "behavior":
            self._build_behavior_body()
        else:
            self._build_zone_body(self.app.analysis_type)

    def refresh_canvas(self):
        if self.canvas is not None:
            self.canvas.refresh()

    def reset_canvas_tool_state(self):
        self.hide_op_bar()
        self._restyle_tool_buttons(None)
        self.refresh_canvas()

    def _restyle_tool_buttons(self, active_key):
        for key, btn in getattr(self, "tool_buttons", {}).items():
            try:
                btn.setObjectName("toolBtnActive" if key == active_key else "toolBtn")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
            except RuntimeError:
                pass

    # ------------------------------------------------------------------
    # Op bar (mirrors TrackerApp._build_op_bar/_rebuild_zone_region_buttons):
    # shown between the toolbar and the canvas while a crop/mask/zones/
    # objects/distance/template_zones operation is being drawn.
    # ------------------------------------------------------------------

    def show_op_bar(self, kind):
        if not hasattr(self, "op_bar_layout"):
            return
        self._clear_layout(self.op_bar_layout)
        if hasattr(self, "op_region_row"):
            del self.op_region_row
        if hasattr(self, "op_draw_mode_row"):
            del self.op_draw_mode_row

        # op_bar_layout is a VBox (see its 2 creation sites): the
        # instructions label gets its own full-width line so it wraps
        # legibly instead of being squeezed into a narrow column next to
        # a growing set of buttons (region names, shape tools, Cancel/
        # Finish all used to compete with it in one QHBoxLayout) -- those
        # buttons live in controls_row below it instead, and zones/objects'
        # shape-tool row gets a THIRD line of its own for the same reason.
        self.op_instructions_label = QLabel("")
        self.op_instructions_label.setStyleSheet("font-size: 10px;")
        self.op_instructions_label.setWordWrap(True)
        self.op_bar_layout.addWidget(self.op_instructions_label)

        controls_row = QHBoxLayout()
        self.op_bar_layout.addLayout(controls_row)

        if kind in ("zones", "objects"):
            self.op_region_row = QHBoxLayout()
            controls_row.addLayout(self.op_region_row)
            self.rebuild_region_buttons()

        if kind == "template_zones":
            # One button per generated zone -- click to rename it. Kept
            # separate from canvas clicks (which are drag-only for this op)
            # so a rename prompt never pops up by surprise while aligning
            # corners; see on_canvas_press's template_zones branch.
            self.op_region_row = QHBoxLayout()
            controls_row.addLayout(self.op_region_row)
            self.rebuild_template_zone_buttons()

        if kind == "mask":
            new_shape_btn = QPushButton("New Shape")
            new_shape_btn.setObjectName("toolBtn")
            new_shape_btn.clicked.connect(self.app.op_new_shape)
            controls_row.addWidget(new_shape_btn)

        if kind == "zones":
            # Draw shapes first, name them after -- see op_new_zone()'s
            # docstring. Kept separate from the pre-named ROI-list flow
            # (still the default/backward-compatible way to start), this
            # is just an escape hatch for "I don't know all the names yet".
            new_zone_btn = QPushButton("New Zone")
            new_zone_btn.setObjectName("toolBtn")
            new_zone_btn.clicked.connect(self.app.op_new_zone)
            controls_row.addWidget(new_zone_btn)

        controls_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("toolBtn")
        cancel_btn.clicked.connect(self.app.cancel_op)
        controls_row.addWidget(cancel_btn)

        finish_btn = QPushButton("Finish")
        finish_btn.setObjectName("accentBtn")
        finish_btn.clicked.connect(self.app.finish_op)
        controls_row.addWidget(finish_btn)

        if kind in ("zones", "objects"):
            self.op_draw_mode_row = QHBoxLayout()
            self.op_bar_layout.addLayout(self.op_draw_mode_row)
            self.rebuild_draw_mode_buttons()

        self.update_op_instructions()

    def hide_op_bar(self):
        if hasattr(self, "op_bar_layout"):
            self._clear_layout(self.op_bar_layout)
        if hasattr(self, "op_region_row"):
            del self.op_region_row
        if hasattr(self, "op_draw_mode_row"):
            del self.op_draw_mode_row

    def rebuild_region_buttons(self):
        if not hasattr(self, "op_region_row"):
            return
        self._clear_layout(self.op_region_row)
        op = self.app._op
        if op is None or "regions" not in op:
            return
        for name in op["regions"].keys():
            is_active = name == op["active_region"]
            b = QPushButton(name)
            b.setObjectName("toolBtnActive" if is_active else "toolBtn")
            b.clicked.connect(lambda checked=False, n=name: self.app.op_set_active_region(n))
            self.op_region_row.addWidget(b)

    def rebuild_template_zone_buttons(self):
        if not hasattr(self, "op_region_row"):
            return
        self._clear_layout(self.op_region_row)
        op = self.app._op
        if op is None or "regions" not in op:
            return
        if op.get("_align_target_index") is not None:
            # Per-video batch alignment (on_align_video): zone NAMES must
            # stay exactly what the template used (batch processing keys
            # results by name), so renaming isn't offered here -- only
            # dragging corners is.
            note = QLabel("Aligning this video's zones -- names stay fixed, drag corners only.")
            note.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9.5px;")
            self.op_region_row.addWidget(note)
            return
        for name in op["regions"].keys():
            b = QPushButton(name)
            b.setObjectName("toolBtn")
            b.setToolTip(f"Rename '{name}'")
            b.clicked.connect(lambda checked=False, n=name: self.app.op_rename_template_zone(n))
            self.op_region_row.addWidget(b)

    def rebuild_draw_mode_buttons(self):
        """Zones/Objects op bar: Freehand/Rectangle/Ellipse/Line shape-tool
        buttons + a Snap to grid checkbox -- see op_set_draw_mode/
        op_toggle_snap in main_window.py."""
        if not hasattr(self, "op_draw_mode_row"):
            return
        self._clear_layout(self.op_draw_mode_row)
        op = self.app._op
        if op is None or "draw_mode" not in op:
            return
        label = QLabel("Shape:")
        label.setStyleSheet("font-size: 10px;")
        self.op_draw_mode_row.addWidget(label)
        for mode, text in (("freehand", "Freehand"), ("rectangle", "Rectangle"),
                           ("ellipse", "Ellipse"), ("line", "Line/Arm")):
            is_active = op.get("draw_mode", "freehand") == mode
            b = QPushButton(text)
            b.setObjectName("toolBtnActive" if is_active else "toolBtn")
            b.clicked.connect(lambda checked=False, m=mode: self.app.op_set_draw_mode(m))
            self.op_draw_mode_row.addWidget(b)
        snap_box = QCheckBox("Snap to grid")
        snap_box.setChecked(bool(op.get("snap")))
        snap_box.setStyleSheet("font-size: 10px;")
        snap_box.toggled.connect(self.app.op_toggle_snap)
        self.op_draw_mode_row.addWidget(snap_box)

    def update_op_instructions(self):
        if not hasattr(self, "op_instructions_label"):
            return
        try:
            self.op_instructions_label.setText(self.app.op_instructions_text())
        except RuntimeError:
            pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _scroll_column(self, fixed_width=None):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if fixed_width:
            scroll.setFixedWidth(fixed_width)
        content = QWidget()
        content.setObjectName("scrollContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)
        scroll.setWidget(content)
        return scroll, layout

    def _video_header_row(self, parent_layout):
        row = QHBoxLayout()
        title = QLabel("Video")
        title.setStyleSheet("font-weight: 700; font-size: 11.5px;")
        row.addWidget(title)
        row.addStretch()
        reset_btn = QPushButton("Reset")
        reset_btn.setObjectName("resetSmallBtn")
        reset_btn.clicked.connect(self.app.on_reset_video_chamber)
        add_btn = QPushButton("+")
        add_btn.setObjectName("accentBtn")
        add_btn.setFixedWidth(30)
        add_btn.clicked.connect(self.app.on_add_videos)
        row.addWidget(reset_btn)
        row.addWidget(add_btn)
        parent_layout.addLayout(row)

        self.video_list_container = QWidget()
        self.video_list_container.setStyleSheet(
            f"background: {PALETTE['SURFACE']}; border: 1px solid {PALETTE['BORDER']}; border-radius: 5px;")
        # Usage hint moved to a hover tooltip on the list itself, rather than
        # a permanent extra line under it (decluttering -- still there when
        # you hover the list, just not taking up screen space by default).
        self.video_list_container.setToolTip("Click a video to make it active for the preview.")
        self.video_list_layout = QVBoxLayout(self.video_list_container)
        self.video_list_layout.setContentsMargins(4, 4, 4, 4)
        self.video_list_layout.setSpacing(2)
        parent_layout.addWidget(self.video_list_container)

        mode_row = QHBoxLayout()
        self.video_mode_note = _hint("")
        mode_row.addWidget(self.video_mode_note)
        mode_row.addStretch()
        parent_layout.addLayout(mode_row)
        self.video_info_label = QLabel("")
        self.video_info_label.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9.5px; font-weight: 700;")
        parent_layout.addWidget(self.video_info_label)

        self._subjects_panel(parent_layout)

        self.refresh_video_list()

    def _subjects_panel(self, parent_layout):
        """Optional Subject database (Excel import) -- see self.app.subjects'
        comment in MainWindow.__init__. Each queued video row (below) gets a
        subject dropdown once at least one subject is imported."""
        row = QHBoxLayout()
        import_btn = QPushButton("Import Subject List (Excel)")
        import_btn.setObjectName("toolBtn")
        import_btn.clicked.connect(self.app.on_import_subjects)
        row.addWidget(import_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("toolBtn")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self.app.on_clear_subjects)
        row.addWidget(clear_btn)
        parent_layout.addLayout(row)
        self.subjects_status_label = _hint("")
        parent_layout.addWidget(self.subjects_status_label)
        self.refresh_subjects_panel()

    def refresh_subjects_panel(self):
        if not hasattr(self, "subjects_status_label"):
            return
        try:
            n = len(self.app.subjects)
            if n == 0:
                self.subjects_status_label.setText(
                    "No subject list imported (optional). Import an Excel file, one row per "
                    "animal, to assign each queued video to a subject and plan sessions ahead.")
            else:
                assigned = sum(1 for v in self.app.videos if v.get("subject_id"))
                self.subjects_status_label.setText(
                    f"{n} subject(s) loaded -- {assigned}/{len(self.app.videos)} queued video(s) assigned.")
        except RuntimeError:
            pass

    def refresh_video_list(self):
        app = self.app
        if hasattr(self, "video_mode_note"):
            try:
                self.video_mode_note.setText(
                    "Individual mode: only 1 video allowed." if app.get_mode() == "individual"
                    else "Batch mode: add as many videos as you like.")
            except RuntimeError:
                pass
        if hasattr(self, "video_list_layout"):
            try:
                self._clear_layout(self.video_list_layout)
                for i, v in enumerate(app.videos):
                    # A queued video now optionally carries a subject
                    # assignment and/or a per-video zone alignment (batch
                    # mode) alongside its name -- rather than crowd all of
                    # that into one horizontal row (which squeezed the
                    # remove button off a narrow 330px panel), each entry
                    # is its own small VBox: the clickable name+remove row
                    # on top, an optional subject/Align row underneath.
                    entry_col = QVBoxLayout()
                    entry_col.setSpacing(1)

                    top_row = QHBoxLayout()
                    name = os.path.basename(v["path"])
                    text = f"{name}   {v['fps']:.2f}fps   {v['duration']:.0f}s"
                    lbl = QLabel(text)
                    lbl.setCursor(Qt.PointingHandCursor)
                    is_active = (i == app.active_index)
                    lbl.setStyleSheet(
                        f"font-size: 10px; font-weight: {'700' if is_active else '400'}; "
                        f"color: {PALETTE['TEXT']}; "
                        f"background: {PALETTE['ACCENT_LIGHT'] if is_active else 'transparent'}; padding: 2px;")
                    lbl.mousePressEvent = (lambda ev, i=i: self.app.on_select_video_row(i))
                    top_row.addWidget(lbl, 1)
                    remove_btn = QPushButton("x")
                    remove_btn.setObjectName("dangerBtn")
                    remove_btn.setFixedWidth(22)
                    remove_btn.clicked.connect(lambda checked=False, i=i: self.app.on_remove_video(i))
                    top_row.addWidget(remove_btn)
                    entry_col.addLayout(top_row)

                    if app.subjects or app.get_mode() == "batch":
                        extra_row = QHBoxLayout()
                        if app.subjects:
                            ids = app.subject_ids()
                            subj_combo = QComboBox()
                            subj_combo.addItem("(no subject)")
                            subj_combo.addItems(ids)
                            current = v.get("subject_id", "")
                            subj_combo.setCurrentText(current if current in ids else "(no subject)")
                            subj_combo.currentTextChanged.connect(
                                lambda text, i=i: self.app.on_assign_subject(i, "" if text == "(no subject)" else text))
                            extra_row.addWidget(subj_combo, 1)
                        else:
                            extra_row.addStretch()
                        if app.get_mode() == "batch":
                            align_btn = QPushButton("Re-align" if v.get("roi_points_override") else "Align")
                            align_btn.setObjectName("toolBtn")
                            align_btn.setToolTip("Nudge this video's own zone alignment (per-video batch alignment)")
                            align_btn.clicked.connect(lambda checked=False, i=i: self.app.on_align_video(i))
                            extra_row.addWidget(align_btn)
                        entry_col.addLayout(extra_row)

                    self.video_list_layout.addLayout(entry_col)
                if not app.videos:
                    empty = _hint("(no videos added -- click + above)")
                    self.video_list_layout.addWidget(empty)
            except RuntimeError:
                pass
        self._update_video_info_label()
        self.refresh_subjects_panel()
        self.refresh_canvas()

    def _update_video_info_label(self):
        if not hasattr(self, "video_info_label"):
            return
        app = self.app
        try:
            if app.active_index is None or not app.videos:
                self.video_info_label.setText("No video selected")
                return
            v = app.videos[app.active_index]
        except (IndexError, RuntimeError):
            return
        w = v.get("width") or 0
        h = v.get("height") or 0
        fps = v.get("fps") or 0
        duration = v.get("duration") or 0
        mins, secs = divmod(int(duration), 60)
        size_txt = f"{w}×{h}" if w and h else "size unknown"
        try:
            self.video_info_label.setText(f"{size_txt}   {fps:.2f} fps   {mins:d}:{secs:02d}")
        except RuntimeError:
            pass

    def set_roi_names_text(self, text):
        if hasattr(self, "roi_names_entry"):
            try:
                self.roi_names_entry.setText(text)
            except RuntimeError:
                pass

    def reset_video_chamber_fields(self):
        for name in ("start_entry", "end_entry", "roi_names_entry", "object_names_entry"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setText("")
                except RuntimeError:
                    pass
        if hasattr(self, "output_size_entry"):
            try:
                self.output_size_entry.setCurrentText("Same as input")
            except RuntimeError:
                pass
        for name in ("interact_var", "entries_var", "altern_var", "all_var"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setChecked(False)
                except RuntimeError:
                    pass
        for name, val in (("behavior_rearing_var", True), ("behavior_grooming_var", True),
                           ("behavior_locomotion_var", False), ("behavior_immobile_var", False)):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.setChecked(val)
                except RuntimeError:
                    pass
        self.refresh_video_list()

    def reset_settings_fields(self):
        self.rebuild(resave=False)

    def on_num_animals(self, n):
        app = self.app
        if app.analysis_type == "standard" and n != 1:
            QMessageBox.information(
                self, "Switch analysis type",
                f"Standard Tracking only handles 1 animal (it needs zones/objects, which the "
                f"multi-animal engine doesn't support yet). Switch to 'Multi-Mouse Tracking' or "
                f"'Behavior Classification' above to track {n} animals."
            )
            return
        self.num_animals = n
        for k, btn in getattr(self, "num_animals_buttons", {}).items():
            btn.setObjectName("analysisCardActive" if k == n else "toolBtn")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _animals_row(self, parent_layout, mode):
        wrap = QVBoxLayout()
        lbl = QLabel("Animals in frame:")
        lbl.setStyleSheet("font-weight: 700; font-size: 10.5px;")
        wrap.addWidget(lbl)
        row = QHBoxLayout()
        default_n = 2 if mode == "multi_mouse" else 1
        self.num_animals = int(self.app._memory.get("num_animals", default_n))
        self.num_animals_buttons = {}
        for n in (1, 2, 3):
            b = QPushButton(str(n))
            b.setFixedWidth(30)
            b.setObjectName("analysisCardActive" if n == self.num_animals else "toolBtn")
            b.clicked.connect(lambda checked=False, n=n: self.on_num_animals(n))
            row.addWidget(b)
            self.num_animals_buttons[n] = b
        row.addStretch()
        wrap.addLayout(row)
        if mode == "multi_mouse":
            wrap.addWidget(_hint("Uses the multi-mouse tracker (background subtract + split + "
                                  "ID match) -- works with 2 or 3 animals"))
        elif mode == "standard":
            wrap.addWidget(_hint("(2/3 animals: switch to Multi-Mouse Tracking above)"))
        else:
            wrap.addWidget(_hint("2 = classifies both mice"))
        parent_layout.addLayout(wrap)

    def _setting_field(self, parent_layout, label_text, attr_name, default):
        lbl = QLabel(label_text)
        lbl.setStyleSheet("font-size: 9.5px;")
        lbl.setWordWrap(True)
        parent_layout.addWidget(lbl)
        e = QLineEdit(self.entry_or_default(attr_name, default))
        parent_layout.addWidget(e)
        setattr(self, attr_name, e)
        return e

    def _color_mode_group(self, parent_layout):
        box = QGroupBox("Analysis Color Mode")
        v = QVBoxLayout(box)
        current = self.app._memory.get("color_mode_var", "auto")
        self._color_mode_radios = {}
        auto_hint = ("Auto checks a few sample frames first and switches to RGB only when "
                     "the animal's color contrasts with the floor but its brightness "
                     "doesn't -- otherwise it uses the faster grayscale mode.")
        for value, text in (("auto", "Auto (recommended)"), ("gray", "Grayscale (faster)"),
                             ("rgb", "RGB / Color")):
            r = QRadioButton(text)
            r.setChecked(value == current)
            self._color_mode_radios[value] = r
            if value == "auto":
                # The explanation lives on a hover ⓘ next to this option now,
                # instead of a permanent paragraph under the whole group.
                row = QHBoxLayout()
                row.addWidget(r)
                row.addWidget(_info_icon(auto_hint))
                row.addStretch()
                v.addLayout(row)
            else:
                v.addWidget(r)
        parent_layout.addWidget(box)

    def _detection_settings(self, parent_layout, mode):
        self._color_mode_group(parent_layout)

        default_min_area = 15 if mode == "standard" else 150
        default_max_area = 5000 if mode == "standard" else 1200
        box = QGroupBox("Detection Settings")
        v = QVBoxLayout(box)
        self._setting_field(v, "Background samples (default 100)", "bg_samples_entry", 100)
        self._setting_field(v, "Difference threshold (start 25)", "threshold_entry", 25)
        self._setting_field(v, "Min mouse area, px (keep LOW)", "min_area_entry", default_min_area)
        self._setting_field(
            v, "Max object area, px" if mode == "standard" else "Max object area, px (~1 mouse; tune this)",
            "max_area_entry", default_max_area)
        self._setting_field(v, "Max movement / frame, px", "max_jump_entry", 100)

        self.use_window_var = QCheckBox("Prior-position weighting")
        self.use_zone_threshold_var = QCheckBox("Per-zone adaptive threshold")
        self.reject_shadows_var = QCheckBox("Reject shadows")
        self.window_size_entry = None
        self.window_weight_entry = None

        if mode == "standard":
            self.use_window_var.setChecked(self.var_or_default("use_window_var", False))
            v.addWidget(self.use_window_var)
            self._setting_field(v, "  window size, px", "window_size_entry", 120)
            self._setting_field(v, "  window weight (0-1)", "window_weight_entry", 0.5)

            self.use_zone_threshold_var.setChecked(self.var_or_default("use_zone_threshold_var", False))
            v.addWidget(self.use_zone_threshold_var)

            self.reject_shadows_var.setChecked(self.var_or_default("reject_shadows_var", False))
            v.addWidget(self.reject_shadows_var)
            v.addWidget(_hint("(if the tracker keeps grabbing the animal's shadow instead of "
                               "its body, try this)"))
        else:
            self.use_window_var.setChecked(self.var_or_default("use_window_var", False))
            self.use_zone_threshold_var.setChecked(self.var_or_default("use_zone_threshold_var", False))
            self.use_window_var.hide()
            self.use_zone_threshold_var.hide()
            self.reject_shadows_var.setChecked(False)
            self.reject_shadows_var.hide()

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        v.addWidget(sep)
        v.addWidget(QLabel("Distance calibration (optional)"))
        v.addWidget(_hint("Click 'Calibrate Distance' above, then fill in:"))
        self._setting_field(v, "Real-world distance", "real_distance_entry", 30)
        self._setting_field(v, "Units (e.g. cm, mm, in)", "units_entry", "cm")
        self._setting_field(v, "Preview frames to save/check", "preview_samples_entry", 6)

        parent_layout.addWidget(box)

    # ------------------------------------------------------------------
    # Standard / Multi-Mouse ("zone") setup body
    # ------------------------------------------------------------------

    def _build_zone_body(self, mode):
        left_scroll, left = self._scroll_column(fixed_width=330)
        self._video_header_row(left)

        time_box = QGroupBox("Time window & zone names")
        tv = QVBoxLayout(time_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Start (s):"))
        self.start_entry = QLineEdit(self.app._memory.get("start_entry", ""))
        self.start_entry.setFixedWidth(60)
        row1.addWidget(self.start_entry)
        row1.addWidget(QLabel("End (s):"))
        self.end_entry = QLineEdit(self.app._memory.get("end_entry", ""))
        self.end_entry.setFixedWidth(60)
        row1.addWidget(self.end_entry)
        row1.addStretch()
        tv.addLayout(row1)
        tv.addWidget(QLabel("ROI names (comma-separated):"))
        self.roi_names_entry = QLineEdit(self.app._memory.get("roi_names_entry", ""))
        tv.addWidget(self.roi_names_entry)
        tv.addWidget(_hint("Object names (only used if Interaction Tracking is on):"))
        self.object_names_entry = QLineEdit(self.entry_or_default("object_names_entry", ""))
        tv.addWidget(self.object_names_entry)
        tv.addWidget(_hint("Interaction margin around objects, px (0 = only counts when the "
                            "tracked point is strictly inside the object's outline; raise this "
                            "to also count approaching/sniffing from just outside it):"))
        self.interaction_margin_entry = QLineEdit(self.entry_or_default("interaction_margin_entry", "20"))
        self.interaction_margin_entry.setFixedWidth(60)
        tv.addWidget(self.interaction_margin_entry)
        left.addWidget(time_box)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Video output size:"))
        self.output_size_entry = QComboBox()
        self.output_size_entry.addItems(["Same as input", "1920x1080", "1280x720", "854x480", "640x480"])
        self.output_size_entry.setCurrentText(self.entry_or_default("output_size_entry", "Same as input"))
        out_row.addWidget(self.output_size_entry)
        out_row.addStretch()
        left.addLayout(out_row)

        analysis_box = QGroupBox("Analysis")
        av = QVBoxLayout(analysis_box)
        self.loc_var = QCheckBox("Location Tracking (always on)" if mode == "standard" else "Location Tracking")
        self.loc_var.setChecked(True if mode == "standard" else self.var_or_default("loc_var", True))
        if mode == "standard":
            self.loc_var.setEnabled(False)
        av.addWidget(self.loc_var)

        self.interact_var = QCheckBox("Interaction Tracking")
        self.interact_var.setChecked(self.var_or_default("interact_var", False))
        av.addWidget(self.interact_var)
        entries_row = QHBoxLayout()
        self.entries_var = QCheckBox("Arm Entries")
        self.entries_var.setChecked(self.var_or_default("entries_var", False))
        entries_row.addWidget(self.entries_var)
        if mode == "standard":
            # Explanation moved to a hover ⓘ (was a permanent paragraph
            # under the checkboxes) -- still one hover away, not clutter.
            entries_row.addWidget(_info_icon(
                "Standard Tracking also estimates nose/center/tail-base each frame (no extra "
                "setup) and reports full/half/semi zone-entry depth (how much of the body "
                "crossed in) in the exported CSV/Excel -- Arm Entries adds a stricter "
                "whole-body '_full_entries' count alongside the usual one."))
        entries_row.addStretch()
        av.addLayout(entries_row)
        self.altern_var = QCheckBox("Arm Alternation")
        self.altern_var.setChecked(self.var_or_default("altern_var", False))
        av.addWidget(self.altern_var)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        av.addWidget(sep)
        self.all_var = QCheckBox("All Behaviours")
        self.all_var.setChecked(self.var_or_default("all_var", False))
        av.addWidget(self.all_var)
        left.addWidget(analysis_box)

        # ---- Quick Setup: one click from an arena template straight to
        # zone drawing, reusing the same real templates as Maze Template.
        quick_box = QGroupBox("Quick Setup (Arena Templates)")
        qv = QVBoxLayout(quick_box)
        qv.addWidget(_hint("Crop or set the arena above first, then pick a shape here to draw "
                            "its zones instantly with default sizes."))
        grid = QGridLayout()
        tiles = [(k, MAZE_TEMPLATES[k]["label"]) for k in MAZE_TEMPLATES.keys()]
        tiles.append((None, "Custom Arena"))
        self.quick_setup_buttons = {}
        for i, (key, label) in enumerate(tiles):
            r, c = divmod(i, 2)
            b = QPushButton(label)
            b.setObjectName("toolBtn")
            if key:
                b.clicked.connect(lambda checked=False, k=key: self.app.quick_setup_template(k))
            else:
                b.clicked.connect(lambda checked=False: self.app.start_op("zones"))
            grid.addWidget(b, r, c)
            self.quick_setup_buttons[key] = b
        qv.addLayout(grid)
        left.addWidget(quick_box)

        self._animals_row(left, mode)
        left.addStretch()

        # ---- CENTER: preview canvas + toolbar ----
        center = QVBoxLayout()
        center_hdr = QHBoxLayout()
        center_hdr.addWidget(QLabel("Preview / Calibration"))
        center_hdr.addStretch()
        reset_calib_btn = QPushButton("Reset")
        reset_calib_btn.setObjectName("resetSmallBtn")
        reset_calib_btn.clicked.connect(self.app.on_reset_calibration_chamber)
        center_hdr.addWidget(reset_calib_btn)
        center.addLayout(center_hdr)

        toolbar = QHBoxLayout()
        self.tool_buttons = {}
        tools = [
            ("crop", "Crop Arena", lambda: self.app.start_op("crop")),
            ("mask", "Mask Zone", lambda: self.app.start_op("mask")),
            ("zones", "Draw Zones", lambda: self.app.start_op("zones")),
            ("maze", "Maze Template", self.app.on_open_maze_template_dialog),
            ("objects", "Mark Objects", lambda: self.app.start_op("objects")),
            ("distance", "Calibrate Distance", lambda: self.app.start_op("distance")),
            ("nocrop", "No Crop", self.app.on_tool_no_crop),
        ]
        for key, text, cmd in tools:
            b = QPushButton(text)
            b.setObjectName("toolBtn")
            b.clicked.connect(cmd)
            toolbar.addWidget(b)
            self.tool_buttons[key] = b
        toolbar.addStretch()
        center.addLayout(toolbar)

        self.op_bar_layout = QVBoxLayout()
        center.addLayout(self.op_bar_layout)

        self.canvas = PreviewCanvas(self.app)
        center.addWidget(self.canvas, 1)

        # ---- RIGHT: detection settings ----
        right_scroll, right = self._scroll_column(fixed_width=320)
        right_hdr = QHBoxLayout()
        right_hdr.addWidget(QLabel("Detection Settings"))
        right_hdr.addStretch()
        reset_settings_btn = QPushButton("Reset")
        reset_settings_btn.setObjectName("resetSmallBtn")
        reset_settings_btn.clicked.connect(self.app.on_reset_settings_chamber)
        right_hdr.addWidget(reset_settings_btn)
        right.addLayout(right_hdr)
        self._detection_settings(right, mode)
        right.addStretch()

        self.body_layout.addWidget(left_scroll)
        self.body_layout.addLayout(center, 1)
        self.body_layout.addWidget(right_scroll)

        # ---- BOTTOM: start + progress ----
        left_cta = QVBoxLayout()
        ready = QLabel("Ready to begin analysis")
        ready.setStyleSheet(f"color: {PALETTE['SUCCESS']}; font-weight: 700; font-size: 10.5px;")
        left_cta.addWidget(ready)
        start_btn = QPushButton("▶  START TRACKING")
        start_btn.setObjectName("startBtn")
        start_btn.clicked.connect(self.app.on_start)
        left_cta.addWidget(start_btn)

        progress_col = QVBoxLayout()
        progress_col.addWidget(QLabel("Progress"))
        self.app.progress_bar = QProgressBar()
        self.app.progress_bar.setValue(0)
        progress_col.addWidget(self.app.progress_bar)
        self.app.progress_label = QLabel("")
        self.app.progress_label.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9.5px;")
        progress_col.addWidget(self.app.progress_label)

        self.bottom_layout.addLayout(left_cta, 1)
        self.bottom_layout.addLayout(progress_col, 1)

        self.refresh_canvas()

    # ------------------------------------------------------------------
    # Behavior Classification setup body
    # ------------------------------------------------------------------

    def _build_behavior_body(self):
        left_scroll, left = self._scroll_column(fixed_width=330)
        self._video_header_row(left)

        time_box = QGroupBox("Time window")
        tv = QVBoxLayout(time_box)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Start (s):"))
        self.start_entry = QLineEdit(self.app._memory.get("start_entry", ""))
        self.start_entry.setFixedWidth(60)
        row1.addWidget(self.start_entry)
        row1.addWidget(QLabel("End (s):"))
        self.end_entry = QLineEdit(self.app._memory.get("end_entry", ""))
        self.end_entry.setFixedWidth(60)
        row1.addWidget(self.end_entry)
        row1.addStretch()
        tv.addLayout(row1)
        left.addWidget(time_box)

        # Not used by this mode, but on_start()/_build_setup reference
        # these names generically -- harmless empty stand-ins.
        self.roi_names_entry = QLineEdit("")
        self.object_names_entry = QLineEdit("")
        self.loc_var = QCheckBox(); self.loc_var.setChecked(False)
        self.interact_var = QCheckBox(); self.interact_var.setChecked(False)
        self.entries_var = QCheckBox(); self.entries_var.setChecked(False)
        self.altern_var = QCheckBox(); self.altern_var.setChecked(False)
        self.all_var = QCheckBox(); self.all_var.setChecked(False)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Video output size:"))
        self.output_size_entry = QComboBox()
        self.output_size_entry.addItems(["Same as input", "1920x1080", "1280x720", "854x480", "640x480"])
        self.output_size_entry.setCurrentText(self.entry_or_default("output_size_entry", "Same as input"))
        out_row.addWidget(self.output_size_entry)
        out_row.addStretch()
        left.addLayout(out_row)

        beh_box = QGroupBox("Behaviors to detect")
        bv = QVBoxLayout(beh_box)
        self.behavior_rearing_var = QCheckBox("Rearing")
        self.behavior_rearing_var.setChecked(self.var_or_default("behavior_rearing_var", True))
        self.behavior_grooming_var = QCheckBox("Grooming")
        self.behavior_grooming_var.setChecked(self.var_or_default("behavior_grooming_var", True))
        self.behavior_locomotion_var = QCheckBox("Locomotion")
        self.behavior_locomotion_var.setChecked(self.var_or_default("behavior_locomotion_var", False))
        self.behavior_immobile_var = QCheckBox("Immobile")
        self.behavior_immobile_var.setChecked(self.var_or_default("behavior_immobile_var", False))
        for w in (self.behavior_rearing_var, self.behavior_grooming_var,
                  self.behavior_locomotion_var, self.behavior_immobile_var):
            bv.addWidget(w)
        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        bv.addWidget(sep)
        self.behavior_all_var = QCheckBox("All")
        self.behavior_all_var.setChecked(self.var_or_default("behavior_all_var", False))
        self.behavior_all_var.toggled.connect(self._on_behavior_all_toggle)
        bv.addWidget(self.behavior_all_var)
        left.addWidget(beh_box)

        self._animals_row(left, "behavior")
        left.addStretch()

        # ---- CENTER ----
        center = QVBoxLayout()
        center_hdr = QHBoxLayout()
        center_hdr.addWidget(QLabel("Preview / Calibration"))
        center_hdr.addStretch()
        reset_calib_btn = QPushButton("Reset")
        reset_calib_btn.setObjectName("resetSmallBtn")
        reset_calib_btn.clicked.connect(self.app.on_reset_calibration_chamber)
        center_hdr.addWidget(reset_calib_btn)
        center.addLayout(center_hdr)

        toolbar = QHBoxLayout()
        self.tool_buttons = {}
        for key, text, cmd in [("crop", "Crop Arena", lambda: self.app.start_op("crop")),
                                ("nocrop", "No Crop", self.app.on_tool_no_crop)]:
            b = QPushButton(text)
            b.setObjectName("toolBtn")
            b.clicked.connect(cmd)
            toolbar.addWidget(b)
            self.tool_buttons[key] = b
        toolbar.addWidget(_hint("  (zones/objects/distance not used by this analysis type)"))
        toolbar.addStretch()
        center.addLayout(toolbar)

        self.op_bar_layout = QVBoxLayout()
        center.addLayout(self.op_bar_layout)

        self.canvas = PreviewCanvas(self.app)
        center.addWidget(self.canvas, 1)

        # ---- RIGHT ----
        right_scroll, right = self._scroll_column(fixed_width=320)
        right_hdr = QHBoxLayout()
        right_hdr.addWidget(QLabel("Detection Settings"))
        right_hdr.addStretch()
        reset_settings_btn = QPushButton("Reset")
        reset_settings_btn.setObjectName("resetSmallBtn")
        reset_settings_btn.clicked.connect(self.app.on_reset_settings_chamber)
        right_hdr.addWidget(reset_settings_btn)
        right.addLayout(right_hdr)
        self._detection_settings(right, "behavior")

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        right.addWidget(sep2)
        beh_settings = QGroupBox("Behavior Classification thresholds")
        bsv = QVBoxLayout(beh_settings)
        bsv.addWidget(_hint("Each frame gets a pseudo-pose (nose/paws/tail) from the detected "
                             "silhouette, then grooming/rearing/locomotion are scored from "
                             "several combined cues, not one threshold."))
        self._setting_field(bsv, "Locomotion speed threshold (px/s)", "loco_thresh_entry", 30)
        self._setting_field(bsv, "Rearing sensitivity 0-1 (lower = more sensitive)", "rear_thresh_entry", 0.25)
        self._setting_field(bsv, "Grooming sensitivity 0-1 (lower = more sensitive)", "groom_thresh_entry", 0.50)
        self._setting_field(bsv, "Frames to confirm a behavior (persistence)", "immobile_thresh_entry", 6)
        self._setting_field(bsv, "Min bout duration (s)", "min_bout_entry", 0.3)
        bsv.addWidget(_hint("Grooming is the hardest of these to detect without a trained pose "
                             "model -- treat it as a starting point to validate by eye, not "
                             "ground truth."))
        right.addWidget(beh_settings)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.HLine)
        right.addWidget(sep3)
        self._build_ml_classifier_panel(right)
        right.addStretch()

        self.body_layout.addWidget(left_scroll)
        self.body_layout.addLayout(center, 1)
        self.body_layout.addWidget(right_scroll)

        # ---- BOTTOM ----
        left_cta = QVBoxLayout()
        ready = QLabel("Ready to begin behavior classification")
        ready.setStyleSheet(f"color: {PALETTE['SUCCESS']}; font-weight: 700; font-size: 10.5px;")
        left_cta.addWidget(ready)
        start_row = QHBoxLayout()
        start_btn = QPushButton("▶  START TRACKING (Automatic)")
        start_btn.setObjectName("startBtn")
        start_btn.clicked.connect(self.app.on_start)
        manual_btn = QPushButton("✎  MANUAL SCORING")
        manual_btn.setObjectName("toolBtn")
        manual_btn.clicked.connect(self.app.on_manual_behavior_scoring)
        start_row.addWidget(start_btn, 1)
        start_row.addWidget(manual_btn, 1)
        left_cta.addLayout(start_row)
        left_cta.addWidget(_hint("Manual scoring opens the video in its own window -- you watch "
                                  "it and press keys to mark when each behavior starts/stops "
                                  "yourself. Use it for grooming especially, since automatic "
                                  "detection is weakest there."))

        progress_col = QVBoxLayout()
        progress_col.addWidget(QLabel("Progress"))
        self.app.progress_bar = QProgressBar()
        self.app.progress_bar.setValue(0)
        progress_col.addWidget(self.app.progress_bar)
        self.app.progress_label = QLabel("")
        self.app.progress_label.setStyleSheet(f"color: {PALETTE['MUTED']}; font-size: 9.5px;")
        progress_col.addWidget(self.app.progress_label)

        self.bottom_layout.addLayout(left_cta, 2)
        self.bottom_layout.addLayout(progress_col, 1)

        self.refresh_canvas()

    def _on_behavior_all_toggle(self, checked):
        for name in ("behavior_rearing_var", "behavior_grooming_var",
                     "behavior_locomotion_var", "behavior_immobile_var"):
            w = getattr(self, name, None)
            if w is not None:
                w.setChecked(checked)

    # ------------------------------------------------------------------
    # Deep-learning classifier panel (optional; tracking/ml_* modules)
    # ------------------------------------------------------------------

    def _build_ml_classifier_panel(self, parent_layout):
        box = QGroupBox("Deep Learning Classifier (optional)")
        v = QVBoxLayout(box)
        v.addWidget(_hint("A small neural network you train yourself on clips you've labeled "
                           "with Manual Scoring -- an alternative to the automatic rules above "
                           "for grooming/rearing, not a replacement for it. 1) Manual Score a "
                           "few videos, 2) Prepare Training Data, 3) Train Model, 4) turn this on."))
        self.ml_mode_var = QCheckBox("Use trained model instead of automatic rules")
        self.ml_mode_var.setChecked(self.var_or_default("ml_mode_var", False))
        v.addWidget(self.ml_mode_var)

        ckpt_row = QHBoxLayout()
        self.ml_checkpoint_entry = QLineEdit(self.entry_or_default("ml_checkpoint_entry", ""))
        ckpt_row.addWidget(self.ml_checkpoint_entry, 1)
        browse_btn = QPushButton("Browse...")
        browse_btn.setObjectName("toolBtn")
        browse_btn.clicked.connect(self.app.on_browse_ml_checkpoint)
        ckpt_row.addWidget(browse_btn)
        v.addLayout(ckpt_row)
        v.addWidget(_hint("Trained model file (.pt)"))

        self.ml_status_label = _hint("")
        v.addWidget(self.ml_status_label)

        prep_btn = QPushButton("Prepare Training Data...")
        prep_btn.setObjectName("toolBtn")
        prep_btn.clicked.connect(self.app.on_prepare_ml_dataset)
        train_btn = QPushButton("Train Model...")
        train_btn.setObjectName("toolBtn")
        train_btn.clicked.connect(self.app.on_train_ml_model)
        v.addWidget(prep_btn)
        v.addWidget(train_btn)

        from tracking.ml_train import TORCH_AVAILABLE
        if not TORCH_AVAILABLE:
            warn = QLabel("PyTorch isn't installed on this machine -- install it (see "
                           "pytorch.org) to train or use a model here. Preparing training "
                           "data doesn't need it, and everything else in the app works "
                           "without it.")
            warn.setWordWrap(True)
            warn.setStyleSheet(f"color: {PALETTE['DANGER']}; font-size: 9.5px;")
            v.addWidget(warn)

        parent_layout.addWidget(box)
