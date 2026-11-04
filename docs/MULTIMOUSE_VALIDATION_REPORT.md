# Multi-Mouse Validation & Bug-Fix Report

## Fixed in this build

1. **Merged-blob identity stability**
   - Overlapping animals are no longer split with unconstrained k-means alone when Kalman predictions are available.
   - The merged foreground pixels are refined from each animal's predicted position, preserving mouse-A/B/C ordering before Hungarian matching.

2. **Start/end time window**
   - Multi-mouse tracking now seeks to `start_time` and processes only the requested frame interval up to `end_time`.
   - Progress is calculated against the selected interval rather than the full video.

3. **Results frame count**
   - `Frames_processed` now reports the actual number of processed frames in the selected interval.

## Stress test

A generated 3-mouse crossing/near-contact video was processed from 2 s to 9 s.

- Requested interval: 210 frames
- Animals: 3
- Tracking rows: 630
- Lost frames: 0
- Partial frames: 48
- OK rows: 582
- Tracking quality: 100%
- Mean spatial assignment error: ~1.13 px
- Identity changes observed against the known synthetic ground truth: **0** for mouse_A, mouse_B and mouse_C over the tested interval
- Processing time: ~2.42 s for 210 frames (~86.8 processed frames/s)

## Interpretation

This validates the repaired multi-mouse engine on a controlled stress case. It does **not** prove zero identity swaps on arbitrary laboratory footage. Real recordings with long complete occlusion, severe lighting changes, reflections, bedding movement, or identical animals can still require manual QC or appearance/pose-based tracking.
