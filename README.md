# Animal Behaviour Tracker

Automatic rodent tracking for behavioral video, with configurable zones,
optional interaction objects, and batch processing. No pose estimation, no
manual per-frame animal selection.

## Step 1 of 10 -- what changed in this update

This is the same tool as before, split into a proper package instead of one
big file, with nothing functionally changed (verified with the same test
suite that's been run at every step of building this). The point: Step 2
(a PySide6 GUI) will only touch `gui/main_window.py` -- every other folder
below is untouched by that migration.

```
AnimalBehaviourTracker/
├── main.py                  entry point -- run this
├── gui/
│   └── main_window.py       dashboard window + every setup dialog
├── tracking/
│   ├── location.py          detection engine + the main per-video pipeline
│   ├── interaction.py       object-proximity bout extraction
│   ├── epm.py                arm entries / spontaneous alternation
│   └── behaviour.py          time-budget binning + bout behavior tagging
├── analysis/
│   ├── calculations.py      generic math helpers (distance, column naming)
│   └── statistics.py        placeholder -- cross-group stats, not built yet
├── output/
│   ├── csv.py                raw_tracking.csv
│   ├── excel.py              the multi-sheet .xlsx workbook
│   └── graphs.py             trajectory.png / heatmap.png
├── resources/                 app icon goes here -- Step 7, not yet
├── models/                    reserved for future ML/pose models -- unused
└── requirements.txt
```

## Install

```
pip install -r requirements.txt
```

`tkinter` is part of the Python standard library, not a pip package. It
ships with the standard Windows/macOS installers. On Linux, if
`import tkinter` fails: `sudo apt-get install python3-tk`.

## Run

```
python main.py
```

Everything about how the app looks and behaves is identical to before --
same dashboard window, same calibration wizard, same save/load setup
feature, same outputs.

## The next 9 steps (for reference)

1. ~~Restructure into this package, confirm nothing broke~~ -- done here
2. Rebuild `gui/main_window.py` in PySide6
3. Connect that new GUI to the (unchanged) `tracking/` pipeline
4. Wire up results display in the new GUI
5. Excel/CSV output polish (the modules already exist as of Step 1)
6. Add distance/velocity/zone-time/alternation graphs to `output/graphs.py`
7. ~~Application icon~~ -- done, out of order, once you had one ready
8. Full pass of testing everything together
9. PyInstaller build of the multi-module app (entry point becomes `main.py`)
10. Wrap the built exe in a proper installer

Each of these is its own real chunk of work -- we're doing them one at a
time, the same way everything so far has been built and tested before
being handed over.
