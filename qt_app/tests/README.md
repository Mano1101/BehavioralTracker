# qt_app/tests -- headless regression suite

Headless regression tests for the PySide6 GUI (`qt_app/`), covering every
screen against the real, unchanged `tracking/`/`analysis/`/`output/`
engine -- not mocked. Each file builds a real `QApplication` + `MainWindow`,
drives it (including real `QTest` mouse/keyboard events where the
interaction itself matters, e.g. the Manual Scoring dialog's keyboard
shortcuts), and asserts on the resulting app state, generated files, and
Results page contents.

## Running

Every test needs a virtual display -- `QT_QPA_PLATFORM=offscreen` alone is
not enough in this environment, so run everything under `xvfb-run -a`.

All tests share the same `dummy_behavior_test.mp4` at the repo root, and
the app writes real output next to the video it's given (`results/`,
`ml_dataset/`, `BatchSummary.csv`, `BatchSummary.xlsx`) -- same as it would
for any real video, since that's what these tests are checking. That's
expected and safe to delete afterwards; it isn't test fixture state
anything else depends on (`test_ml_pipeline.py` and `test_batch_features.py`
clear their own output dir/batch-summary files at the top of the file so
their guard checks don't depend on what ran before them).

Run the whole suite:

```
xvfb-run -a python3.12 qt_app/tests/run_all.py
```

Run one file directly:

```
xvfb-run -a python3.12 qt_app/tests/test_start_tracking.py
```

Run a subset through the runner:

```
xvfb-run -a python3.12 qt_app/tests/run_all.py test_maze_template_dialog test_ml_pipeline
```

Each file is a standalone script, not a pytest module -- only one
`QApplication` can live in a process, and several tests intentionally drive
a full, real `MainWindow` end to end. Each prints a `[PASS]`/`[FAIL]` line
per check, a final `ALL CHECKS PASSED` or a numbered failure list, and
exits `0`/`1` accordingly. `run_all.py` runs them as subprocesses (in the
order below) and prints a combined summary.

## Files

- `_pathsetup.py` -- shared setup, imported first by every test module:
  puts the repo root on `sys.path`, `chdir()`s into it (tests use relative
  paths like `dummy_behavior_test.mp4`), and sets
  `QT_QPA_PLATFORM=offscreen` before PySide6 is ever imported (it reads
  that env var at first-import/init time, so this import must come before
  any `PySide6`/`qt_app`/`tracking` import in a test file).
- `test_canvas_smoke.py` -- header, video probing, Standard Tracking Quick
  Setup/crop/zones/mask/objects/distance ops (direct calls and real
  `QTest` mouse events), analysis-type switching, Results page navigation,
  Save/Open Project round-trip, Reset handlers.
- `test_maze_template_dialog.py` -- arena-needed guard, happy-path
  default-params generation, invalid-numeric-input guard, real-world-unit
  labeling of length params once a distance calibration is active.
- `test_start_tracking.py` -- Start Tracking -> Results for all three
  analysis types (Standard Individual, Batch, Multi-Mouse, Behavior
  Classification) plus the ML-mode-no-checkpoint guard, run against the
  real tracking pipeline.
- `test_manual_scoring.py` -- bout open/close/undo/subject-switching/
  finish() on the dialog directly, then a full keyboard-driven session
  through `MainWindow` with real `QTest` key events (this is what caught
  the Tab-focus-chain bug -- see `focusNextPrevChild()` in
  `qt_app/dialogs/manual_scoring_dialog.py`).
- `test_batch_features.py` -- the 4 features built from MM's SMART 3.0/
  ANY-maze reference videos: Subject database (Excel import via a mocked
  `QFileDialog`, per-video assignment, tagging batch summary rows), batch
  mode's per-video zone alignment (`on_align_video`'s drag-only
  `template_zones` op, saved as a `roi_points_override` per video without
  touching the shared template, and used automatically by a batch run),
  Export to Excel on the Results page (single run + `BatchSummary.xlsx`),
  and a Save/Open Project round-trip of subjects + overrides. The
  Rectangle/Ellipse/Line shape tools + snap-to-grid (the other item from
  that same feature set) are covered in `test_canvas_smoke.py` instead,
  alongside the rest of the canvas op machinery they extend.
- `test_ml_pipeline.py` -- the Deep Learning Classifier panel end to end:
  Prepare Training Data -> Train Model (real 1-epoch PyTorch training) ->
  Start Tracking with "Use trained model" (real inference), plus the
  multi-animal confirmation guard. Needs `TORCH_AVAILABLE`.

## Adding a new test

Follow the existing files: `import _pathsetup` (bare, not
`qt_app.tests._pathsetup` -- see that module's docstring for why) as the
very first import, patch the `QMessageBox` statics before importing
`qt_app` anything (they're shared class objects), build a real
`QApplication` + `MainWindow`, use the module-level `check(name, cond)`
helper, and end with a `[FAIL]` summary + `sys.exit(1)`, or `sys.exit(0)`
(the default) on success. Add the new filename to `ORDERED_TESTS` in
`run_all.py`.
