# BehavioralTracker Pro — Backend Integration Audit

## Fixed in this build

1. **Results navigation**
   - The UI has one combined `Analysis & Results` page, not a separate Results page.
   - Results navigation now intentionally opens stack index 4 and selects the Overview tab.

2. **Test-specific analysis flags**
   - EPM now sends `compute_arm_entries=True` to the existing tracking backend.
   - Y-Maze now sends `compute_alternation=True` to the existing tracking backend.

3. **Analysis tab binding**
   - The controller explicitly binds the existing `analysisTabs` widget so Results navigation can select Overview safely.

## Verified connections

- UI → `UXController` → `tracking.location.process_single_video`
- Tracking progress/frame callbacks → Qt signals → Tracking page
- `tracking.location` → `analysis.calculations`
- `tracking.location` → `tracking.epm`
- `tracking.location` → `output.csv`
- `tracking.location` → `output.excel`
- `tracking.location` → `output.graphs`
- Results UI reads the generated Analysis workbook and raw CSV.
- Batch processing uses the same worker/backend pipeline.

## Verified functional test

The bundled `dummy_behavior_test.mp4` was processed headlessly through `process_single_video` for a 5-second run.

Result: **PASS** — 151/151 frames tracked (100% tracking quality), Excel/CSV/plot outputs generated, and a summary dictionary was returned.

## Remaining integration scope

- Multi-animal selector currently remains intentionally blocked in the main tracking UI; the backend has a separate `tracking/two_mouse.py` implementation but it is not yet wired into this controller's single-animal result pipeline.
- DeepLabCut selectors are intentionally informational; no pose model is bundled.
- Behaviour classification/ML remains a separate backend flow and is not automatically run by the centroid tracking button.

These are next-stage integrations, not UI navigation failures.

## Multi-Mouse integration completed in this build

- The Tracking page's Number of Animals selector now supports **1, 2 and 3** directly; the unsupported 4-animal option is removed from the runtime UI.
- Selecting 2 or 3 switches the worker to `tracking.two_mouse.process_multi_mouse_video` instead of the single-animal `tracking.location` engine.
- The existing OpenCV + Hungarian identity matching + Kalman smoothing engine is now connected to the Qt background worker.
- Annotated tracking frames are streamed back to the Tracking page during processing.
- Multi-Mouse outputs include:
  - `multi_mouse_tracking.csv` — identity-resolved frame-by-frame tracks
  - `raw_tracking.csv` — compatibility bridge for the existing Analysis & Results page
  - `multi_mouse_summary.csv` — per-animal metrics
  - `multi_mouse_timebins.csv` — per-animal time-bin distances
  - `trajectory.png`
  - `heatmap.png`
  - `timebins.png`
  - `MultiMouse_Analysis.xlsx`
  - `multi_mouse_preview.mp4`
- Batch Processing now passes the selected 1/2/3-animal tracking mode to the same worker pipeline.

## Multi-Mouse functional verification

A synthetic two-mouse video was processed through the application-facing `process_multi_mouse_video` API.
Result: **PASS** — 100 tracking rows for 2 animals across 50 frames, 100% tracking quality, CSV/Excel/trajectory/heatmap/time-bin outputs generated, and annotated preview generated.

## Known scientific limitation

This is **unmarked classical multi-animal tracking**. Identity is inferred from motion continuity. During prolonged physical overlap/huddling, a true identity swap cannot be ruled out without visual markers or a pose/re-identification model. The annotated preview should therefore be treated as the quality-control record for identity continuity.
