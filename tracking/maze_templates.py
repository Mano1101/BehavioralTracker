"""Maze/apparatus geometry templates.

Generates the standard named zone polygons for common behavioral paradigms
(Elevated Plus Maze, Y-Maze, T-Maze, Open Field, 3-Chamber Social Test) from
a handful of MEASURED dimensions (arm length, arm width, center size,
rotation...) instead of the researcher having to freehand-trace every arm
by eye with the mouse.

Zones come back in the SAME coordinate space as the calibrated/cropped
arena (0..warp_w, 0..warp_h) that tracking.location.build_roi_masks and
every other zone-based function already consumes -- so template-generated
zones flow through the whole existing analysis pipeline (interaction
stats, arm entries, alternation, heatmaps, CSV/Excel export...) with zero
special-casing anywhere else in the app. As far as the rest of the
pipeline is concerned, a template-generated zone is indistinguishable from
one the researcher drew by hand.

Each template returns an ORDERED list of (suggested_label, polygon)
pairs. The suggested label is a starting guess, not a commitment -- e.g.
generate_epm() has no way to know which physical pair of arms is actually
"open" on a given rig, so the GUI lets the researcher click any generated
zone on the canvas and confirm or rename it before the layout is
finalized. That's apparatus knowledge no geometry alone can infer.
"""

import math


def _rect(cx, cy, length, width, angle_deg):
    """An axis-agnostic rectangle centered at (cx, cy): `length` long along
    angle_deg (0 = +x axis, increasing clockwise since image y grows
    downward), `width` across. Returns 4 corner points, in order."""
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)   # unit vector along the long axis
    nx, ny = -dy, dx                              # perpendicular (width) direction
    hl, hw = length / 2.0, width / 2.0
    return [
        (cx + dx * hl + nx * hw, cy + dy * hl + ny * hw),
        (cx + dx * hl - nx * hw, cy + dy * hl - ny * hw),
        (cx - dx * hl - nx * hw, cy - dy * hl - ny * hw),
        (cx - dx * hl + nx * hw, cy - dy * hl + ny * hw),
    ]


def _regular_polygon(cx, cy, radius, n_sides=16):
    """A regular n_sides-gon approximating a circle of the given radius --
    used as the maze's central hub zone instead of a square. A circle (as
    opposed to a square) is EXACTLY `radius` away from its center in every
    direction, so an arm placed at any angle -- including an arbitrary
    rotation_deg, or Y-Maze's 120-degree spacing, neither of which lines up
    with a square's 4-fold symmetry -- starts flush against the hub with no
    geometric overlap and no gap, regardless of angle."""
    pts = []
    for i in range(n_sides):
        theta = math.radians(360.0 * i / n_sides)
        pts.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return pts


def _arm_rect(cx, cy, center_radius, arm_length, arm_width, angle_deg, gap_px=2.0):
    """One maze arm: a rectangle running OUTWARD from the edge of the
    (circular) center hub -- not from the center point -- so it doesn't
    overlap the hub. `center_radius` is the hub's radius (see
    _regular_polygon): because the hub is circular, this is the correct
    offset distance for an arm at ANY angle, not just ones aligned to a
    square's sides.

    gap_px leaves a small deliberate gap between the arm and the hub --
    with zero gap, a rasterized pixel mask right at that shared boundary
    can end up "inside" both zones at once for the same pixel, which would
    occasionally mislabel the exact moment of an arm entry/exit as a
    nonsense combined zone. gap_px=2 is negligible next to any real
    arm/hub measurement but keeps every zone's mask cleanly disjoint."""
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    mid_dist = center_radius + gap_px + arm_length / 2.0
    mx, my = cx + dx * mid_dist, cy + dy * mid_dist
    return _rect(mx, my, arm_length, arm_width, angle_deg)


def generate_epm(warp_w, warp_h, arm_length, arm_width, center_size, rotation_deg=0.0):
    """Elevated Plus Maze: a circular center hub with 4 arms at 90 degrees
    to each other. By convention, the pair at rotation_deg/+180 is
    suggested as "Open" and the perpendicular pair at +90/+270 as "Closed"
    -- swap these on the canvas if your rig is oriented the other way."""
    cx, cy = warp_w / 2.0, warp_h / 2.0
    radius = center_size / 2.0
    return [
        ("Center", _regular_polygon(cx, cy, radius)),
        ("Open Arm 1", _arm_rect(cx, cy, radius, arm_length, arm_width, rotation_deg)),
        ("Open Arm 2", _arm_rect(cx, cy, radius, arm_length, arm_width, rotation_deg + 180)),
        ("Closed Arm 1", _arm_rect(cx, cy, radius, arm_length, arm_width, rotation_deg + 90)),
        ("Closed Arm 2", _arm_rect(cx, cy, radius, arm_length, arm_width, rotation_deg + 270)),
    ]


def generate_y_maze(warp_w, warp_h, arm_length, arm_width, center_size, rotation_deg=0.0):
    """Y-Maze: a circular central hub with 3 arms 120 degrees apart."""
    cx, cy = warp_w / 2.0, warp_h / 2.0
    radius = center_size / 2.0
    zones = [("Center", _regular_polygon(cx, cy, radius))]
    for i, label in enumerate(["Arm A", "Arm B", "Arm C"]):
        angle = rotation_deg + i * 120.0
        zones.append((label, _arm_rect(cx, cy, radius, arm_length, arm_width, angle)))
    return zones


def generate_t_maze(warp_w, warp_h, stem_length, goal_length, arm_width, center_size, rotation_deg=0.0):
    """T-Maze: one stem arm (the start-box end) plus two goal arms at 90
    degrees to the stem, meeting at a circular center hub. rotation_deg=0
    points the stem toward -x and the two goal arms run along +/-y --
    rotate to match your rig."""
    cx, cy = warp_w / 2.0, warp_h / 2.0
    radius = center_size / 2.0
    return [
        ("Center", _regular_polygon(cx, cy, radius)),
        ("Stem", _arm_rect(cx, cy, radius, stem_length, arm_width, rotation_deg + 180)),
        ("Goal Arm 1", _arm_rect(cx, cy, radius, goal_length, arm_width, rotation_deg + 90)),
        ("Goal Arm 2", _arm_rect(cx, cy, radius, goal_length, arm_width, rotation_deg + 270)),
    ]


def generate_open_field(warp_w, warp_h, center_fraction=0.5, gap_px=2.0):
    """Open Field: the whole calibrated arena split into a rectangular
    "Center" zone (center_fraction of each side -- 0.5 is the common OFT
    convention: a center square half the width/height) plus 4 border
    strips covering everything else. Kept as 4 separate periphery zones
    (Top/Bottom/Left/Right) rather than one polygon-with-a-hole, since the
    app's zone masks are simple filled polygons -- as a side benefit this
    also lets a researcher look at wall-hugging bias per wall if they
    want, not just an aggregate "periphery".

    The periphery strips are inset by gap_px from Center's own boundary
    (see _arm_rect's docstring for why touching, unshrunk zones can cause
    a rasterized pixel to register as "inside" two zones at once) --
    leaves a hairline (gap_px-wide) unassigned ring around Center,
    negligible next to any real arena size."""
    cw, ch = warp_w * center_fraction, warp_h * center_fraction
    cx, cy = warp_w / 2.0, warp_h / 2.0
    top_y, bot_y = cy - ch / 2, cy + ch / 2
    left_x, right_x = cx - cw / 2, cx + cw / 2
    gt, gb = top_y - gap_px, bot_y + gap_px
    gl, gr = left_x - gap_px, right_x + gap_px
    # Left/Right are additionally inset a further gap_px from Top/Bottom's
    # own edges (gt/gb), not just from Center -- Top and Bottom already
    # span the FULL width, so without this, Left/Right's top/bottom edges
    # would land exactly on Top/Bottom's edges and double-count that shared
    # line too, the same touching-boundary issue as Center vs Periphery
    # above. Leaves a tiny (gap_px-square) unassigned notch at each of the
    # 4 inner corners -- negligible next to any real arena size.
    lt, lb = gt + gap_px, gb - gap_px
    return [
        ("Center", _rect(cx, cy, cw, ch, 0)),
        ("Periphery Top", [(0, 0), (warp_w, 0), (warp_w, gt), (0, gt)]),
        ("Periphery Bottom", [(0, gb), (warp_w, gb), (warp_w, warp_h), (0, warp_h)]),
        ("Periphery Left", [(0, lt), (gl, lt), (gl, lb), (0, lb)]),
        ("Periphery Right", [(gr, lt), (warp_w, lt), (warp_w, lb), (gr, lb)]),
    ]


def generate_social_3_chamber(warp_w, warp_h, side_fraction=0.32, gap_px=2.0):
    """3-Chamber Social Test: the arena split into 3 equal-height strips
    left to right -- Chamber 1 (often the stranger-mouse side), Center,
    Chamber 2 (often the object/empty side). side_fraction is each side
    chamber's share of the total width (defaults to just under a third
    each, leaving the doorway margins out of every zone by design).
    gap_px inset between adjacent chambers, same reasoning as
    generate_open_field's gap_px."""
    sw = warp_w * side_fraction
    half_gap = gap_px / 2.0
    return [
        ("Chamber 1", [(0, 0), (sw - half_gap, 0), (sw - half_gap, warp_h), (0, warp_h)]),
        ("Center", [(sw + half_gap, 0), (warp_w - sw - half_gap, 0),
                    (warp_w - sw - half_gap, warp_h), (sw + half_gap, warp_h)]),
        ("Chamber 2", [(warp_w - sw + half_gap, 0), (warp_w, 0),
                       (warp_w, warp_h), (warp_w - sw + half_gap, warp_h)]),
    ]


# Each param tuple is (key, label, default, kind). kind == "length" means
# the value is a real physical measurement, so the GUI converts it through
# the video's real-world scale calibration (px per cm) when one has been
# set, and treats it as raw pixels otherwise; "fraction" and "angle" are
# never unit-converted.
TEMPLATES = {
    "epm": {
        "label": "Elevated Plus Maze",
        "params": [
            ("arm_length", "Arm length", 150.0, "length"),
            ("arm_width", "Arm width", 40.0, "length"),
            ("center_size", "Center size", 40.0, "length"),
            ("rotation_deg", "Rotation (deg)", 0.0, "angle"),
        ],
        "generate": lambda w, h, p: generate_epm(
            w, h, p["arm_length"], p["arm_width"], p["center_size"], p["rotation_deg"]),
    },
    "y_maze": {
        "label": "Y-Maze",
        "params": [
            ("arm_length", "Arm length", 150.0, "length"),
            ("arm_width", "Arm width", 40.0, "length"),
            ("center_size", "Center size", 50.0, "length"),
            ("rotation_deg", "Rotation (deg)", 0.0, "angle"),
        ],
        "generate": lambda w, h, p: generate_y_maze(
            w, h, p["arm_length"], p["arm_width"], p["center_size"], p["rotation_deg"]),
    },
    "t_maze": {
        "label": "T-Maze",
        "params": [
            ("stem_length", "Stem length", 150.0, "length"),
            ("goal_length", "Goal arm length", 100.0, "length"),
            ("arm_width", "Arm width", 40.0, "length"),
            ("center_size", "Center size", 40.0, "length"),
            ("rotation_deg", "Rotation (deg)", 0.0, "angle"),
        ],
        "generate": lambda w, h, p: generate_t_maze(
            w, h, p["stem_length"], p["goal_length"], p["arm_width"], p["center_size"], p["rotation_deg"]),
    },
    "open_field": {
        "label": "Open Field",
        "params": [
            ("center_fraction", "Center zone size (fraction of arena, e.g. 0.5)", 0.5, "fraction"),
        ],
        "generate": lambda w, h, p: generate_open_field(w, h, p["center_fraction"]),
    },
    "social_3_chamber": {
        "label": "3-Chamber Social Test",
        "params": [
            ("side_fraction", "Side chamber width (fraction of arena, e.g. 0.32)", 0.32, "fraction"),
        ],
        "generate": lambda w, h, p: generate_social_3_chamber(w, h, p["side_fraction"]),
    },
}
