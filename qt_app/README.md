# Animal Behaviour Tracker

Automatic rodent tracking for behavioral video, with configurable zones,
optional interaction objects, multi-mouse ID-matched tracking, a
grooming/rearing/locomotion behavior classifier (rule-based, or an optional
trained deep-learning model), manual behavior scoring, maze/arena templates,
and batch processing (all three analysis types). No pose estimation, no
manual per-frame animal selection.

Also includes a set of features brought in from EthoVision XT and SMART
3.0 (see "Upgrade Plan" in the roadmap below): configurable stop
conditions, a Subject Database with real experimental metadata, Zone
Associations (combining zones for reporting without redrawing), a
trajectory smoothing/outlier filter, live camera acquisition, sandboxed
custom variables/derived columns, a real Settings screen, configurable
time-bin reporting for every analysis type, and a custom report builder
on the Results screen.

## Two GUIs, one unchanged engine

The app was rebuilt from Tkinter to PySide6 (Qt) one piece at a time; that
migration is done. **`qt_app/` is the app -- `main.py` runs it** and it
covers every screen the old one did (Setup, the interactive preview
canvas, Start Tracking + Results for all three analysis types, the Quick
Setup apparatus templates, Manual Scoring, and the Deep Learning
Classifier panel).
`gui/` (Tkinter) is kept working as an unmaintained fallback/reference,
run via `main_legacy_tkinter.py` -- not the recommended way to run the
app, and not what the PyInstaller build (`BehavioralTracker.spec`)
packages.

> `main.py` used to be the Tkinter entry point, back when the Qt GUI was
> the new, parallel, not-yet-primary one (then named `main_qt.py`). The
> names were swapped on 2026-09-24 once Qt became the actual app, since
> the old naming led someone to run the unmaintained GUI by mistake, not
> realizing `main_qt.py` was the one to use -- and separately, until the
> same date, `BehavioralTracker.spec` had `PySide6` in its PyInstaller
> `excludes` list (a leftover from when it targeted the Tkinter app),
> which meant every packaged .exe/.app silently could not have run the
> Qt GUI at all. Both are fixed now; see git history / this README's
> roadmap item 5 for the details if this ever needs untangling again.

Neither GUI touches `tracking/`, `analysis/`, or `output/` -- those are the
same detection/analysis engine and file-writing code underneath both.

```
AnimalBehaviourTracker/
├── main.py                   entry point -- run this (PySide6/Qt GUI)
├── main_legacy_tkinter.py    old entry point (Tkinter GUI, unmaintained fallback)
├── BehavioralTracker.spec    PyInstaller build spec (packages main.py, i.e. the Qt GUI)
├── qt_app/                   current GUI
│   ├── main_window.py        shared state + cross-cutting actions
│   ├── theme.py               palette + stylesheet
│   ├── pages/                 setup_page.py, results_page.py
│   ├── widgets/                preview_canvas.py, results_charts.py
│   ├── dialogs/                maze_template_dialog.py, manual_scoring_dialog.py,
│   │                            ml_dataset_dialog.py, ml_train_dialog.py,
│   │                            zone_label_dialog.py, camera_dialog.py (live
│   │                            camera acquisition), settings_dialog.py (the
│   │                            Settings screen), report_builder_dialog.py
│   │                            (Custom Report builder)
│   ├── app_settings.py        persisted Detection Settings defaults + app-wide
│   │                            preferences (~/.behavioraltracker/settings.json)
│   └── tests/                  headless regression suite (run_all.py runs it all)
├── gui/
│   └── main_window.py        Tkinter dashboard window + every setup dialog
├── tracking/
│   ├── location.py           detection engine + the main per-video (single-mouse) pipeline
│   ├── two_mouse.py           multi-mouse background-subtract + ID-matched tracking
│   │                            (also calculate_time_bins -- Multi-Mouse's
│   │                            configurable time-bin report)
│   ├── behavior.py            grooming/rearing/locomotion feature extraction + classifier
│   │                            + manual scoring (also calculate_behavior_time_bins and
│   │                            behavior_summary_table -- Behavior Classification's
│   │                            time-bin report and Custom Report stats table)
│   ├── behaviour.py           (older, British spelling) interaction-bout tagging/binning
│   │                            helpers used by location.py, incl. the shared bin-edge/
│   │                            label helpers (format_bin_label/compute_bin_edges/
│   │                            bin_row_label) every analysis type's time-bin report uses
│   ├── interaction.py         object-proximity bout extraction
│   ├── epm.py                  arm entries / spontaneous alternation
│   ├── maze_templates.py      built-in maze/arena zone-geometry templates
│   ├── ml_dataset.py          slices labeled video into training clips (no torch dependency)
│   ├── ml_model.py             the clip-classifier network definition
│   ├── ml_train.py             training loop (needs torch; everything else works without it)
│   └── ml_infer.py             runs a trained model over a video
├── analysis/
│   ├── calculations.py       generic math helpers (distance, column naming)
│   ├── custom_variables.py   sandboxed "name = expression" derived-column engine
│   │                            (Setup page's Custom Variables panel, Standard Tracking)
│   ├── custom_report.py      analysis-type-agnostic column-picking for the Custom
│   │                            Report builder (Results page, all three analysis types)
│   └── statistics.py         placeholder -- cross-group stats, not built yet
├── output/
│   ├── csv.py                 raw_tracking.csv
│   ├── excel.py                the multi-sheet .xlsx workbook (Summary/Raw Tracking/
│   │                            <bin> Individual+Cumulative/Transitions/etc.)
│   └── graphs.py               trajectory.png / heatmap.png / zone_occupancy.png
├── resources/                  app icon (icon.png/.ico/.icns), wired into the window/taskbar icon
├── models/                     reserved for future ML/pose models -- unused
└── requirements.txt
```

## Install

```
pip install -r requirements.txt
```

PySide6 (the current GUI's framework) is a regular pip package, included
above. `tkinter` is only needed for the old GUI
(`python main_legacy_tkinter.py`) -- it's part of the Python standard
library, not a pip package, and ships with the standard Windows/macOS
installers. On Linux, if `import tkinter` fails:
`sudo apt-get install python3-tk`.

PyTorch is optional and only needed for the Deep Learning Classifier panel's
"Train Model" / "Use trained model" (Prepare Training Data works without
it). See the comment in `requirements.txt` for the install command, since
it differs by machine (CPU-only vs. GPU/CUDA).

## Run

```
python main.py
```

`python main_legacy_tkinter.py` still runs the older Tkinter GUI
unchanged, for comparison or as a fallback.

## Remaining roadmap

1. ~~Rebuild the GUI in PySide6~~ -- done: Setup, interactive canvas, Start
   Tracking + Results (all 3 analysis types), Quick Setup apparatus
   templates, Manual Scoring, Deep Learning Classifier panel are all
   wired up in `qt_app/`.
2. ~~A dedicated Settings screen~~ -- done (Upgrade Plan Tier 2 #8):
   `qt_app/dialogs/settings_dialog.py` persists Detection Settings
   defaults and a couple of app-wide preferences (default units, default
   color mode) to `~/.behavioraltracker/settings.json`
   (`qt_app/app_settings.py`), loaded straight into the same `_memory`
   dict the Setup page's own fields already read -- so Save takes effect
   on the current Setup page immediately, and a brand-new session starts
   from these instead of the hardcoded fallbacks every time. Deliberately
   excludes a theme preference -- dark/light mode was removed on purpose
   (see `qt_app/theme.py`'s own docstring), not just hidden.
3. ~~Full headless regression pass consolidated into a permanent test
   suite~~ -- done: `qt_app/tests/` (16 suites -- core UI, Quick Setup
   apparatus templates, Start Tracking/Results for all 3 analysis types,
   Manual Scoring, the batch/subject/Excel-export features below, the
   Deep Learning Classifier pipeline end to end, and every Upgrade Plan
   feature below). Run with `xvfb-run -a python3.12 qt_app/tests/run_all.py`;
   see `qt_app/tests/README.md`.
4. ~~Batch mode per-video zone alignment, Subject database, Excel
   export~~ -- done, built after comparing against commercial competitors
   (SMART 3.0, ANY-maze): each queued video in Batch mode can have its own
   zone alignment (the "Align" button next to it in the video list, saved
   independently of the shared template -- see `on_align_video` in
   `qt_app/main_window.py`); an optional Subject database imports an animal
   list from Excel and assigns a subject to each queued video, tagging
   batch summary rows; and a "Export to Excel" button on the Results page
   (plus an automatic `BatchSummary.xlsx` alongside `BatchSummary.csv`)
   gives a direct `.xlsx` for every analysis type. Draw Zones/Mark Objects
   also gained Rectangle/Ellipse/Line shape-drawing tools and a snap-to-grid
   toggle, alongside the original freehand click-to-add-point drawing.
5. ~~PyInstaller build of the multi-module app~~ -- `BehavioralTracker.spec`
   exists and CI (`.github/workflows/build.yml`) runs it per-OS, but it was
   silently broken from the start: it still pointed at the old Tkinter
   `main.py` and (worse) had `PySide6` in its `excludes` list, a leftover
   from before the Qt GUI existed -- meaning every packaged .exe/.app ever
   built from this repo shipped the unmaintained Tkinter GUI, or would have
   flat-out failed to start if pointed at Qt. Fixed 2026-09-24: `main.py`
   is now the Qt entry point (see the note above), the spec's
   `collect_all(...)` loop now includes `"PySide6"`, and `"PySide6"` was
   removed from `excludes`. Not yet re-verified end to end by actually
   running a built .exe/.app on a real Windows/macOS machine -- do that
   before relying on a downloaded build; running from source
   (`python main.py`) is unaffected by any of this and has always used the
   real Qt GUI.
6. Wrap the built exe in a proper installer.
7. ~~Upgrade Plan: features brought in from EthoVision XT & SMART 3.0~~ --
   done (Tier 1 + Tier 2 of the plan; Tier 3's hardware-trigger output and
   true multi-point pose tracking were scoped separately and are still
   out of scope). Ten features, each with its own regression suite in
   `qt_app/tests/`:
   1. Configurable stop conditions (Setup page, Standard Tracking) -- stop
      after N seconds of continuous immobility, N zone entries, or a
      total-distance threshold, in addition to the fixed End (s) time.
   2. Richer Subject Database fields -- Code/Group/Color/Sex/Age/Genotype/
      Phenotype/Treatment/Dose columns (when present in the imported
      Excel file) carry into `BatchSummary.csv`/`.xlsx`, not just an ID.
   3. Zone Associations -- combine 2+ existing zones into one named
      reporting group (e.g. "Left Side" = Zone A + Zone B) without
      redrawing anything.
   4. Trajectory smoothing / outlier filter -- an optional post-processing
      pass on `raw_tracking.csv` before the Trajectory/Heatmap charts, to
      reduce single-frame detection jitter.
   5. Batch mode for Multi-Mouse Tracking & Behavior Classification --
      the batch loop that used to exist only for Standard Tracking now
      covers all three analysis types.
   6. Live camera acquisition -- a "Camera" button next to the video list
      opens a live OpenCV `VideoCapture(index)` preview with a Record
      button (`qt_app/dialogs/camera_dialog.py`), alongside file upload.
   7. Custom variables / derived columns -- a sandboxed "name = expression"
      panel (`analysis/custom_variables.py`, a hand-rolled AST whitelist
      evaluator, not `eval()`/`df.eval()`) adds derived columns to the
      exported CSV/Excel, e.g. `distance_cm = Distance_pixels / scale_factor`.
   8. A real Settings screen -- see item 2 above.
   9. Configurable time-bin reporting for every analysis type -- the bin
      size behind Standard Tracking's "1min Individual"/"1min Cumulative"
      sheets is now configurable (Setup page's shared "Time Bins"
      panel, `bin_size_entry`), and the same idea was extended to
      Multi-Mouse (distance/frames-tracked per mouse per bin) and
      Behavior Classification (time-in-each-behavior per mouse per bin),
      both of which had no time-bin report before.
   10. Custom report builder -- a "Custom Report" button on the Results
       page (all three analysis types) opens a checkbox picker
       (`qt_app/dialogs/report_builder_dialog.py`) over whichever
       computed-stats table that analysis type already builds, and saves
       just the checked columns to `Custom_Report.csv`/`.xlsx`, instead of
       always exporting the fixed Summary sheet shape.
