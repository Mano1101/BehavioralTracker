"""
Regression test for the "Draw Zone Outline" apparatus auto-decompose
feature (tracking/location.py's detect_apparatus_partitions / _partition_
one_blob) -- the one MM flagged after everything else was settled: "okay
fine mathathu ellamey oralayuku okay than ipo ROI Marking mattom than
enaku innam statisfy aagala" (everything else is fine, only the ROI
Marking/auto zone-outline splitting still isn't right), narrowed down via
follow-up questions to: the zone COUNT coming out right, but the zone
BOUNDARIES landing in the wrong place.

Two real, independent bugs were found and fixed here, both in how
_partition_one_blob grows each arm's zone outward from its skeleton seed
via cv2.watershed:

1. (The big one -- this is almost certainly what MM was actually seeing.)
   "Outside the traced outline" used to be seeded as its own watershed
   label (1), competing for interior pixels same as every arm's seed. But
   each arm's seed is only a thin line down its OWN corridor's centre, so
   for any interior pixel within roughly half the corridor's width of a
   side wall, that wall sits closer (in raw pixel distance) than the
   arm's own seed line does -- so "outside" was WINNING a wide strip along
   both walls of every arm, shrinking each zone to a narrow central
   stripe of its true traced width (confirmed on the synthetic EPM below:
   covered under half the true per-arm area before the fix). Fixed by
   leaving "outside" unseeded and instead giving cv2.watershed a barrier
   IMAGE (flat/0 inside the traced outline, high/255 outside it) so the
   interior is always fully resolved, arm vs arm, before the exterior is
   ever touched -- see the long comment above `markers = np.zeros(...)`
   in _partition_one_blob.

2. (Smaller, more subtle -- a real but much less visually dramatic
   defect.) At the hub where 2+ arms actually meet, each arm's seed is
   deliberately cut short of the true centre by `junction_halo_px` (so
   branch-finding doesn't see one pixel soup instead of separate arms).
   Right at that crossing, cv2.watershed's nearest-seed-PIXEL metric then
   quietly flips from "distance to a long line" (a clean straight
   bisector) to "distance to that line's cut-short endpoint" (a curved
   boundary) for whichever arm's seed is still line-like there, visibly
   bulging the boundary instead of radiating straight out of the
   crossing. Fixed by deciding those specific crossing pixels by plain
   angle around the crossing's own centre instead, before watershed ever
   runs -- see the angle-split block in _partition_one_blob.

Pure tracking/location.py logic, no Qt/video needed -- lives here anyway,
same as test_auto_mask_outside_zones.py, so run_all.py picks it up
automatically.

Run directly:
    python3.12 qt_app/tests/test_apparatus_outline_split.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import numpy as np
import cv2

from tracking.location import detect_apparatus_partitions

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def polygon_coverage(zones, comp_mask):
    """-> (gap_px, overlap_px, inside_px): rasterize every returned zone
    polygon and compare against the traced shape's own true filled mask,
    to catch a zone polygon that doesn't actually reach the real traced
    wall (gap) as well as two zones double-claiming the same pixels
    (overlap) -- both symptoms of a bad watershed/polygon-extraction, not
    just a cosmetic rendering issue."""
    h, w = comp_mask.shape
    coverage = np.zeros((h, w), dtype=np.int32)
    for _name, pts in zones.items():
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 1)
        coverage += m.astype(np.int32)
    inside = comp_mask == 1
    gap_px = int((inside & (coverage == 0)).sum())
    overlap_px = int((inside & (coverage >= 2)).sum())
    return gap_px, overlap_px, int(inside.sum())


def filled_mask(pts_list, width, height):
    filled = np.zeros((height, width), dtype=np.uint8)
    for pts in pts_list:
        cv2.fillPoly(filled, [np.array(pts, dtype=np.int32)], 255)
    return (filled > 0).astype(np.uint8)


# ------------------------------------------------------------------
# 1) Symmetric EPM-style plus-maze: 4 equal arms off a square hub -- the
# exact shape used to originally diagnose both bugs above.
# ------------------------------------------------------------------
W, H = 500, 500
cx, cy = W // 2, H // 2
arm_len, arm_half_w = 180, 35
epm_pts = [
    (cx - arm_half_w, cy - arm_half_w - arm_len), (cx + arm_half_w, cy - arm_half_w - arm_len),
    (cx + arm_half_w, cy - arm_half_w), (cx + arm_half_w + arm_len, cy - arm_half_w),
    (cx + arm_half_w + arm_len, cy + arm_half_w), (cx + arm_half_w, cy + arm_half_w),
    (cx + arm_half_w, cy + arm_half_w + arm_len), (cx - arm_half_w, cy + arm_half_w + arm_len),
    (cx - arm_half_w, cy + arm_half_w), (cx - arm_half_w - arm_len, cy + arm_half_w),
    (cx - arm_half_w - arm_len, cy - arm_half_w), (cx - arm_half_w, cy - arm_half_w),
]
epm_mask = filled_mask([epm_pts], W, H)
epm_area = int(epm_mask.sum())

zones, err = detect_apparatus_partitions([epm_pts], W, H)
check("EPM: no error", err is None)
check("EPM: exactly 4 zones found (4 arms, hub too small to get its own)", len(zones) == 4)

expected_per_zone = epm_area / 4.0
if len(zones) == 4:
    for name, pts in zones.items():
        area = cv2.contourArea(np.array(pts, dtype=np.int32))
        # Before the "outside competes as a seed" fix, each zone covered
        # under HALF its true share of the traced area (a narrow central
        # stripe of the corridor instead of its full width) -- so this
        # tolerance band is tight enough to catch that regression again
        # while still allowing for ordinary polygon-simplification slop.
        check(f"EPM: zone {name} area ({area:.0f}) is within 15% of the true "
              f"even 1/4 share ({expected_per_zone:.0f})",
              abs(area - expected_per_zone) <= 0.15 * expected_per_zone)

total_zone_area = sum(cv2.contourArea(np.array(pts, dtype=np.int32)) for pts in zones.values())
check(f"EPM: zones' total area ({total_zone_area:.0f}) covers >=97% of the traced "
      f"outline's true area ({epm_area})",
      total_zone_area >= 0.97 * epm_area)

gap_px, overlap_px, inside_px = polygon_coverage(zones, epm_mask)
check(f"EPM: per-pixel gap between zone polygons and the true traced shape "
      f"({gap_px}px) is well under 1% of the shape ({inside_px}px)",
      gap_px < 0.01 * inside_px)
check(f"EPM: per-pixel overlap between adjacent zone polygons ({overlap_px}px) "
      f"is well under 1% of the shape ({inside_px}px)",
      overlap_px < 0.01 * inside_px)

# Each arm's zone should reach all the way to BOTH its true side walls,
# not stop partway and leave a dead strip along them (the core symptom of
# the "outside outcompetes the arm seed near a wall" bug). Checked on the
# north arm, deep enough into its corridor to be far from any hub effect.
north_zone_pts = None
for name, pts in zones.items():
    cyy = np.mean([p[1] for p in pts])
    if cyy < cy - arm_half_w - arm_len * 0.3:  # clearly the north arm's zone
        north_zone_pts = pts
        break
if north_zone_pts is not None:
    m = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(m, [np.array(north_zone_pts, dtype=np.int32)], 1)
    row = m[cy - arm_half_w - int(arm_len * 0.6)]  # a row deep in the north arm
    xs = np.where(row > 0)[0]
    true_left, true_right = cx - arm_half_w, cx + arm_half_w
    check("EPM: north arm's zone reaches within 3px of the true LEFT wall "
          f"(true={true_left}, got={xs.min() if xs.size else None})",
          xs.size > 0 and xs.min() <= true_left + 3)
    check("EPM: north arm's zone reaches within 3px of the true RIGHT wall "
          f"(true={true_right}, got={xs.max() if xs.size else None})",
          xs.size > 0 and xs.max() >= true_right - 3)
else:
    check("EPM: could identify the north arm's own zone to check its wall-to-wall width", False)

print()

# ------------------------------------------------------------------
# 2) T-maze: one stem + a top crossbar (2 arms) -- same per-blob code
# path, different arm count/angles (perpendicular, but asymmetric: 1 vs 2
# arms on each side of the junction, unlike the EPM's 4-way symmetry).
# ------------------------------------------------------------------
tcx, tcy = 300, 300
stem = [(tcx - 30, tcy), (tcx + 30, tcy), (tcx + 30, tcy + 220), (tcx - 30, tcy + 220)]
top = [(tcx - 220, tcy - 30), (tcx + 220, tcy - 30), (tcx + 220, tcy + 30), (tcx - 220, tcy + 30)]
filled_t = np.zeros((600, 600), dtype=np.uint8)
cv2.fillPoly(filled_t, [np.array(stem, dtype=np.int32)], 255)
cv2.fillPoly(filled_t, [np.array(top, dtype=np.int32)], 255)
contours_t, _ = cv2.findContours(filled_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
outline_t = max(contours_t, key=cv2.contourArea)
eps_t = 0.003 * cv2.arcLength(outline_t, True)
approx_t = cv2.approxPolyDP(outline_t, eps_t, True)
t_pts = [(int(p[0][0]), int(p[0][1])) for p in approx_t]
t_mask = filled_mask([t_pts], 600, 600)
t_area = int(t_mask.sum())

t_zones, t_err = detect_apparatus_partitions([t_pts], 600, 600)
check("T-maze: no error", t_err is None)
check("T-maze: exactly 3 zones found (2 top arms + 1 stem)", len(t_zones) == 3)

t_total_area = sum(cv2.contourArea(np.array(pts, dtype=np.int32)) for pts in t_zones.values())
check(f"T-maze: zones' total area ({t_total_area:.0f}) covers >=95% of the traced "
      f"outline's true area ({t_area})",
      t_total_area >= 0.95 * t_area)

t_gap_px, t_overlap_px, t_inside_px = polygon_coverage(t_zones, t_mask)
check(f"T-maze: per-pixel gap ({t_gap_px}px) is well under 1% of the shape ({t_inside_px}px)",
      t_gap_px < 0.01 * t_inside_px)

print()

# ------------------------------------------------------------------
# 3) A non-branching shape (plain rectangle, e.g. an open-field arena) is
# untouched by either fix -- still comes back as ONE whole zone.
# ------------------------------------------------------------------
rect_pts = [(50, 50), (450, 50), (450, 350), (50, 350)]
rect_zones, rect_err = detect_apparatus_partitions([rect_pts], 500, 400)
check("Open field (no branching): no error", rect_err is None)
check("Open field (no branching): comes back as exactly ONE zone", len(rect_zones) == 1)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    import sys
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
