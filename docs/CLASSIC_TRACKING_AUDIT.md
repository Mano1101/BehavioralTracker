# Classic Tracking Integration Audit

- Removed DeepLabCut selectors from the Tracking UI; the bundled engine is now the classic reference-difference tracker.
- Added visible Start time / End time controls.
- Added Gray / RGB detection mode.
- Added percentile location threshold, previous-location window, window size and window weight.
- Adapted the supplied LocationTracking_Functions.py concepts: median reference, percentile threshold, prior-location weighting, and gray/RGB handling.
- Kept the existing multi-mouse backend separate; the classic engine is the primary single-animal tracker.
- Existing Behavior, Event Editor, Experimental Design, Professional Analysis and Advanced Statistics pipeline remains connected.
- GUI rendering must still be run on the user's Windows/PySide6 environment.
