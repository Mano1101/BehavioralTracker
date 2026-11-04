# Multi-Mouse Core Integration & Pipeline Audit

## Scope
Validated the application path:

`Tracking UI → multi_mouse tracker → identity-preserved raw_tracking.csv → Behavior layer → Professional Analysis → Results/exports`

## Integration checks

- Tracking page `cmbNumAnimals` exposes 1–3 animals.
- Selecting 2 or 3 animals sets `tracking_mode=multi` in the controller setup.
- `TrackWorker` routes multi jobs to `tracking.two_mouse.process_multi_mouse_video`.
- The worker then sends the resulting `raw_tracking.csv` to `tracking.behavior_events.classify_tracking_output`.
- Behavior receives `num_animals` from the same tracking setup, preserving the selected subject count.
- Professional Analysis runs after behavior and reads the same output folder.
- Multi-mouse bridge retains `mouse_id`, frame/time, coordinates and tracking status.
- Per-mouse distance and tracking-quality outputs are retained in `multi_mouse_summary.csv` and `MultiMouse_Analysis.xlsx`.
- The Professional Analysis layer can calculate per-mouse statistics and behavior outputs without collapsing mouse identities.
- Partial tracking status is preserved as `Partial` rather than being silently converted to `Tracked`.

## Validation

A synthetic two-animal video was used to exercise the controller-facing backend API. The test checked that:

1. two mouse identities are emitted;
2. `raw_tracking.csv` contains both IDs;
3. behavior output is generated for the multi-animal tracking result;
4. professional analysis consumes the same result folder;
5. per-mouse analysis rows remain separated by `mouse_id`.

## Scientific boundary

The classical multi-mouse tracker is identity-matching/centroid based. It is not equivalent to trained pose estimation such as DeepLabCut. Identity confidence should therefore be visually QC-checked, especially during crossing, occlusion, close contact and merged silhouettes.

## Result

**PASS** for the controller-facing multi-mouse pipeline on the validated synthetic case.
