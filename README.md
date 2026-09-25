# Animal Behaviour Tracker

Automatic rodent tracking for behavioral video, with configurable zones,
optional interaction objects, multi-mouse ID-matched tracking, a
grooming/rearing/locomotion behavior classifier (rule-based, or an optional
trained deep-learning model), manual behavior scoring, maze/arena templates,
and batch processing. No pose estimation, no manual per-frame animal
selection.

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
│   │                            zone_label_dialog.py
│   └── tests/                  headless regression suite (run_all.py runs it all)
├── gui/
│   └── main_window.py        Tkinter dashboard window + every setup dialog
├── tracking/
│   ├── location.py           detection engine + the main per-video (single-mouse) pipeline
│   ├── two_mouse.py           multi-mouse background-subtract + ID-matched tracking
│   ├── behavior.py            grooming/rearing/locomotion feature extraction + classifier + manual scoring
│   ├── behaviour.py           (older, British spelling) interaction-bout tagging/binning helpers used by location.py
│   ├── interaction.py         object-proximity bout extraction
│   ├── epm.py                  arm entries / spontaneous alternation
│   ├── maze_templates.py      built-in maze/arena zone-geometry templates
│   ├── ml_dataset.py          slices labeled video into training clips (no torch dependency)
│   ├── ml_model.py             the clip-classifier network definition
│   ├── ml_train.py             training loop (needs torch; everything else works without it)
│   └── ml_infer.py             runs a trained model over a video
├── analysis/
│   ├── calculations.py       generic math helpers (distance, column naming)
│   └── statistics.py         placeholder -- cross-group stats, not built yet
├── output/
│   ├── csv.py                 raw_tracking.csv
│   ├── excel.py                the multi-sheet .xlsx workbook
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
2. A dedicated Settings screen (Detection Settings currently live on the
   Setup page itself, same as before).
3. ~~Full headless regression pass consolidated into a permanent test
   suite~~ -- done: `qt_app/tests/` (6 suites -- core UI, Quick Setup
   apparatus templates, Start Tracking/Results for all 3 analysis types,
   Manual Scoring, the batch/subject/Excel-export features below, and the
   Deep Learning Classifier pipeline end to end). Run with
   `xvfb-run -a python3.12 qt_app/tests/run_all.py`; see
   `qt_app/tests/README.md`.
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
