"""Shared color palettes for ROI zones and interaction objects.

Split out of tracking/location.py so that output/graphs.py (and anything
else) can import these WITHOUT importing the whole detection engine --
previously graphs.py imported them from tracking.location.py, which in
turn imports output.graphs, a genuine circular import that only worked
when tracking.location happened to be imported first.
"""

ROI_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (180, 105, 255),
]

OBJECT_COLORS = [
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (0, 200, 120),
    (200, 0, 120),
]
