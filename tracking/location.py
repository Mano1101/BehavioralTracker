"""
Core tracking engine: perspective rectification, zone/object masks,
background modeling, per-frame animal detection, zone transitions, the
calibration-preview renderer, and the main per-video pipeline
(process_single_video) that ties everything -- including the other
tracking/analysis/output modules -- together.
"""

import cv2
import numpy as np
import pandas as pd
import os
import glob
import math

from analysis.calculations import point_distance, safe_col
from analysis.custom_variables import apply_custom_variables
from tracking.interaction import extract_bouts
from tracking.epm import calculate_arm_entries
from output.csv import write_csv_report
from output.excel import write_excel_report

# Color palettes live in tracking/colors.py (shared with output/graphs.py
# without a circular import -- see that module's docstring).
from tracking.colors import ROI_COLORS, OBJECT_COLORS  # noqa: F401  (re-exported for existing importers)
from tracking.fragment import label_point, outline_arrays

from output.graphs import (
    save_plots, save_zone_occupancy_chart, save_zone_latency_chart,
    save_timecourse_plots, save_thigmotaxis_chart,
)


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def compute_perspective_transform(pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 2)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 2)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return matrix, max_width, max_height


def identity_transform(width, height):
    """
    A perspective 'transform' that changes nothing -- used when the user
    opts out of the 4-point crop. Every downstream step (make_background,
    detect_mouse, the tracking loop, etc.) always expects a matrix plus a
    warp width/height, so this lets "no cropping" mean "warp with an
    identity matrix at the video's native size" instead of needing a
    separate code path everywhere a matrix is used.
    """
    return np.eye(3, dtype=np.float32), int(width), int(height)


def read_and_warp(cap, frame_number, matrix, out_w, out_h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()

    if not ok:
        return None

    return cv2.warpPerspective(frame, matrix, (out_w, out_h))


# -----------------------------
# Scale-bar overlay (25 / 50 / 75 / 100% on both axes)
# -----------------------------


def draw_scale_overlay(frame):
    disp = frame.copy()
    h, w = disp.shape[:2]
    color = (0, 200, 255)

    for frac in (0.25, 0.50, 0.75, 1.00):
        x = max(0, min(w - 1, int(w * frac) - 1))
        cv2.line(disp, (x, 0), (x, h), color, 1)
        cv2.putText(disp, f"{int(frac * 100)}%", (max(2, x - 24), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        y = max(0, min(h - 1, int(h * frac) - 1))
        cv2.line(disp, (0, y), (w, y), color, 1)
        cv2.putText(disp, f"{int(frac * 100)}%", (4, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return disp


# -----------------------------
# Named polygon zones
# -----------------------------


def build_roi_masks(shape_hw, roi_points):
    h, w = shape_hw
    masks = {}

    for name, pts in roi_points.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        masks[name] = mask > 0

    return masks


def dilate_masks(masks, margin_px):
    """Grow each named boolean mask outward by margin_px pixels in every
    direction. Used to turn an object's own drawn outline into an "approach
    zone" around it, so interaction can be counted when the animal's tracked
    point is near the object -- e.g. sniffing it from just outside its
    footprint -- not only when the point falls strictly inside the outline
    the user drew. A strict polygon-only check systematically undercounts
    real interaction: the tracked point is normally the animal's centroid or
    nose, and an animal investigating an object leads with its nose right up
    to (but not necessarily past) the object's edge.

    margin_px <= 0 returns the masks unchanged -- this keeps every existing
    caller's behavior identical unless it explicitly opts in to a margin."""
    if margin_px is None or margin_px <= 0:
        return masks
    k = int(round(margin_px)) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = {}
    for name, mask in masks.items():
        m8 = mask.astype(np.uint8) * 255
        m8 = cv2.dilate(m8, kernel)
        dilated[name] = m8 > 0
    return dilated


def roi_membership(masks, x, y):
    xi, yi = int(round(x)), int(round(y))
    membership = {}

    for name, mask in masks.items():
        h, w = mask.shape
        membership[name] = bool(0 <= yi < h and 0 <= xi < w and mask[yi, xi])

    return membership


def linearize_roi(membership, null_name="None"):
    active = [name for name, in_roi in membership.items() if in_roi]
    return "_".join(active) if active else null_name


# -----------------------------
# Line-based zone drawing (auto-detected partitions)
# -----------------------------


def _partition_letter(index):
    """0->A, 1->B, ..., 25->Z, 26->AA, 27->AB, ... (spreadsheet-column
    style), so an apparatus with more than 26 partitions still gets
    distinct, orderly labels instead of running out of letters."""
    index += 1
    letters = ""
    while index > 0:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _mask_to_polygon(region_mask, epsilon_frac=0.01):
    """A binary (0/255) uint8 mask -> a simplified (x, y) polygon outline
    (cv2.findContours + approxPolyDP), in exactly the shape
    build_roi_masks()/roi_membership() already expect from a hand-drawn
    zone. Returns None if the mask has no usable contour (empty, or the
    simplified outline collapses to fewer than 3 points)."""
    contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    return pts if len(pts) >= 3 else None


def _order_reading_order(found, height):
    """found: [(cy, cx, pts), ...] -> the same entries reordered top-to-
    bottom, then left-to-right -- bucketing into "rows" of roughly-level
    partitions first (within row_tol of each other) so e.g. two side-by-
    side arms read left-then-right before dropping to the next row,
    rather than being ordered by exact pixel y (which would interleave
    rows on the slightest camera tilt)."""
    found = sorted(found, key=lambda p: p[0])
    row_tol = max(20.0, height * 0.08)
    rows = []
    for cy, cx, pts in found:
        placed_row = next((r for r in rows if abs(r[0] - cy) <= row_tol), None)
        if placed_row is None:
            rows.append([cy, [(cy, cx, pts)]])
        else:
            placed_row[1].append((cy, cx, pts))
    ordered = []
    for _row_cy, items in rows:
        items.sort(key=lambda p: p[1])
        ordered.extend(items)
    return ordered


# -----------------------------
# Apparatus-outline auto-decomposition -- the "trace the whole apparatus"
# Zone Drawing workflow. There is no ready-made skeletonize() available in
# this build (scikit-image isn't installed, and plain opencv-python here
# has no cv2.ximgproc thinning either), so _zhang_suen_thin hand-rolls the
# classic 1984 two-subiteration thinning algorithm, vectorized with NumPy
# array shifts instead of a per-pixel Python loop.
# -----------------------------


def _shift_neighbors(img):
    """The 8 neighbors of every pixel in `img` (a 0/1 array), clockwise
    from north (P2..P9 in the classic Zhang-Suen naming), each as a
    same-shape array aligned to img -- neighbors[k][r, c] is img's pixel
    in direction k from (r, c). A pixel off the array's edge reads as 0
    (padding), which is exactly "not foreground" -- the right answer for
    a skeleton that never wraps around the frame border."""
    h, w = img.shape
    padded = np.zeros((h + 2, w + 2), dtype=img.dtype)
    padded[1:-1, 1:-1] = img
    p2 = padded[0:h, 1:w + 1]      # N
    p3 = padded[0:h, 2:w + 2]      # NE
    p4 = padded[1:h + 1, 2:w + 2]  # E
    p5 = padded[2:h + 2, 2:w + 2]  # SE
    p6 = padded[2:h + 2, 1:w + 1]  # S
    p7 = padded[2:h + 2, 0:w]      # SW
    p8 = padded[1:h + 1, 0:w]      # W
    p9 = padded[0:h, 0:w]          # NW
    return p2, p3, p4, p5, p6, p7, p8, p9


def _zhang_suen_thin(mask01, max_iterations=500):
    """Zhang-Suen thinning: reduces a filled binary shape (mask01: 0/1
    uint8 array, foreground=1) down to its 1-pixel-wide medial-axis
    skeleton. This is what detect_apparatus_partitions reads the traced
    apparatus's branching structure from -- a plain blob (open field)
    thins down to one simple strand with no junctions; a star/plus shape
    (EPM, Y-maze, radial-arm maze, ...) thins down to one strand per arm,
    all meeting at a junction cluster in the middle."""
    img = mask01.astype(np.uint8).copy()
    for _ in range(max_iterations):
        changed = False
        for step in (1, 2):
            p2, p3, p4, p5, p6, p7, p8, p9 = _shift_neighbors(img)
            b = (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9).astype(np.uint8)
            a = (
                ((p2 == 0) & (p3 == 1)).astype(np.uint8)
                + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
                + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
                + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
                + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
                + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
                + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
                + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
            )
            if step == 1:
                cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            to_remove = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if np.any(to_remove):
                img[to_remove] = 0
                changed = True
        if not changed:
            break
    return img


def _skeleton_neighbor_counts(skel01):
    p2, p3, p4, p5, p6, p7, p8, p9 = _shift_neighbors(skel01.astype(np.uint8))
    return p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9


def _build_redundant_neighborhood_lut():
    """256-entry lookup, indexed by an 8-bit pattern of which of a
    pixel's 8 ring neighbors (bit k = P(2+k) in the classic Zhang-Suen
    ordering N, NE, E, SE, S, SW, W, NW, k=0..7) are foreground: True
    when those neighbors are ALL already mutually reachable from one
    another directly, without going through the shared center pixel --
    i.e. the center pixel is topologically redundant for connectivity.

    Two ring positions are themselves directly (8-)adjacent to each
    other exactly when they're at most 2 apart in this cyclic order (N &
    NE are next-door; N & E are also directly diagonal-adjacent to each
    other even though NE sits between them in the ring; N & SE, or N &
    S opposite, are not). Zhang-Suen's own "A(P)==1" transition test
    (used inside _zhang_suen_thin) only checks ring-CONSECUTIVE runs, so
    it misses exactly this ring-distance-2 case -- e.g. a pixel whose
    only two foreground neighbors are due-east and due-south (ring
    positions E, S -- distance 2 apart the short way around, through the
    empty SE between them) reads as "2 separate arcs" to that test even
    though E and S are already directly diagonal-adjacent to each other.
    This is exactly the "elbow" a diagonal thinned run characteristically
    leaves behind (see _remove_redundant_junction_pixels), and this LUT
    correctly marks that pattern as redundant where the simpler
    consecutive-arc test would not."""
    def _adjacent(i, j):
        d = abs(i - j) % 8
        d = min(d, 8 - d)
        return 0 < d <= 2

    lut = np.zeros(256, dtype=bool)
    for pattern in range(256):
        present = [k for k in range(8) if (pattern >> k) & 1]
        if len(present) < 2:
            continue
        visited = {present[0]}
        frontier = [present[0]]
        while frontier:
            cur = frontier.pop()
            for k in present:
                if k not in visited and _adjacent(cur, k):
                    visited.add(k)
                    frontier.append(k)
        lut[pattern] = len(visited) == len(present)
    return lut


_REDUNDANT_NEIGHBORHOOD_LUT = _build_redundant_neighborhood_lut()


def _remove_redundant_junction_pixels(skel01, max_iterations=50):
    """Removes a foreground pixel whose foreground neighbors are all
    ALREADY directly connected to each other without it (see
    _build_redundant_neighborhood_lut) -- by definition safe, since no
    neighbor relies on this pixel to reach any other. Thinning's own
    per-sub-iteration conditions (there to stop a 1px-wide line from
    being erased outright) occasionally leave one of these behind on a
    diagonal run anyway, reading as a spurious degree>=3 "junction" where
    no real branch point exists.

    This sweeps up exactly that pattern (degree >= 3 AND topologically
    redundant) and nothing else: a GENUINE junction -- three or four arms
    meeting at one pixel -- has its neighbors sitting in 2+ truly
    SEPARATE groups (there's no way to walk from one arm's direction to
    another's without crossing the junction pixel itself), so the LUT
    reads False there and this leaves it untouched.

    Redundant-for-ITS-OWN-neighbors doesn't mean safe to remove several
    such pixels all AT ONCE, though: two ADJACENT "redundant" pixels can
    each individually check out fine (each one's own neighbors stay
    connected without IT) while secretly relying on EACH OTHER -- remove
    both together and the line between them still breaks. So removal is
    done in 4 interleaved sub-passes by (row%2, col%2) class, recomputing
    which pixels still qualify before each one: any two pixels in the
    same class are always at least 2 apart in row or column, so they're
    never 8-adjacent to each other and a whole class can safely be
    removed together in one vectorized pass."""
    skel = skel01.astype(np.uint8).copy()
    h, w = skel.shape
    row_parity = (np.arange(h) % 2).reshape(-1, 1)
    col_parity = (np.arange(w) % 2).reshape(1, -1)
    class_masks = [(row_parity == rp) & (col_parity == cp) for rp in (0, 1) for cp in (0, 1)]

    for _ in range(max_iterations):
        any_removed = False
        for cmask in class_masks:
            p2, p3, p4, p5, p6, p7, p8, p9 = _shift_neighbors(skel)
            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            pattern = (
                p2.astype(np.int32) | (p3.astype(np.int32) << 1) | (p4.astype(np.int32) << 2)
                | (p5.astype(np.int32) << 3) | (p6.astype(np.int32) << 4) | (p7.astype(np.int32) << 5)
                | (p8.astype(np.int32) << 6) | (p9.astype(np.int32) << 7)
            )
            redundant = (skel == 1) & (b >= 3) & _REDUNDANT_NEIGHBORHOOD_LUT[pattern] & cmask
            if np.any(redundant):
                skel[redundant] = 0
                any_removed = True
        if not any_removed:
            break
    return skel


def _prune_skeleton_spurs(skel01, iterations):
    """Erode away short spurs from a skeleton by repeatedly deleting
    endpoint pixels (skeleton pixels with exactly 1 skeleton neighbor),
    `iterations` times. A spur shorter than `iterations` pixels
    disappears entirely -- this is what a straight/rectangular arm TIP's
    medial axis characteristically produces (the axis forks into two
    short branches approaching each corner of the flat end before
    merging into the main strand), a corner artifact rather than a real
    branch. A real arm strand is far longer than that, so it just loses
    a few pixels off its tip and otherwise survives, separately labeled.
    Only used to decide which branches are real (see
    _partition_one_blob) -- never to build the final zone shapes, which
    come from watershed-growing the untouched filled mask outward from
    each surviving branch's own seed."""
    skel = skel01.astype(np.uint8).copy()
    for _ in range(iterations):
        neighbor_count = _skeleton_neighbor_counts(skel)
        endpoints = (skel == 1) & (neighbor_count == 1)
        if not np.any(endpoints):
            break
        skel[endpoints] = 0
    return skel


def _partition_one_blob(comp_mask, offset, min_area_fraction):
    """One connected filled blob (comp_mask: 0/1 uint8, cropped to its own
    bounding box) -> a list of (cy, cx, pts) in FULL-FRAME coordinates
    (offset = the crop's (x0, y0) in the full frame, added back onto
    every point/centroid before returning). See detect_apparatus_partitions
    for the overall algorithm this implements."""
    ox, oy = offset
    ch, cw = comp_mask.shape
    comp_area = int(comp_mask.sum())
    min_dim = max(1, min(cw, ch))

    def _whole_blob_as_one_zone():
        pts = _mask_to_polygon((comp_mask * 255).astype(np.uint8))
        if pts is None:
            return []
        pts_full = [(x + ox, y + oy) for x, y in pts]
        cx = float(np.mean([p[0] for p in pts_full]))
        cy = float(np.mean([p[1] for p in pts_full]))
        return [(cy, cx, pts_full)]

    # A hand-traced outline is never pixel-perfectly smooth -- clicking a
    # curve or a straight edge as a handful of points instead of tracing
    # every pixel leaves small in/out wiggles along the boundary, and a
    # medial-axis skeleton is notoriously sensitive to exactly that kind
    # of small-scale boundary noise (each little wiggle can throw off its
    # own short spurious branch). Skeletonize a lightly SMOOTHED copy of
    # the filled mask instead of the raw one -- opening then closing with
    # a kernel much smaller than any real arm should be, so it rounds off
    # small trace jitter without eating into (or bridging across) an
    # actual narrow arm. The untouched, un-smoothed comp_mask is still
    # what the final zones are grown over below (via watershed), so this
    # only affects which branches are FOUND, never the shape of a zone.
    smooth_px = max(2, round(0.012 * min_dim))
    smooth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (smooth_px * 2 + 1, smooth_px * 2 + 1))
    smoothed = cv2.morphologyEx((comp_mask * 255).astype(np.uint8), cv2.MORPH_OPEN, smooth_kernel)
    smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, smooth_kernel)
    smoothed = (smoothed > 0).astype(np.uint8)
    if not np.any(smoothed):
        smoothed = comp_mask  # a shape thinner than the smoothing kernel itself -- don't erase it away

    skel = _zhang_suen_thin(smoothed)
    if not np.any(skel):
        return _whole_blob_as_one_zone()
    skel = _remove_redundant_junction_pixels(skel)

    # A straight/rectangular arm TIP's own medial axis characteristically
    # forks into two short branches approaching each corner of the flat
    # end before merging into the main strand -- a shape artifact, not a
    # real fork, and the sharper the corner the more of these show up
    # (e.g. every corner along a jittery hand-drawn trace). A LIGHT fixed
    # prune first soaks up single-pixel jaggies from thinning a rasterized
    # polygon; telling a genuine short arm apart from a corner-fork spur
    # then happens below by comparing each candidate branch's length to
    # the corridor's own local width (distance transform), NOT a fixed
    # pixel count -- a fixed threshold can't work across arms of very
    # different widths (a maze's 20px-wide arms vs. its 80px-wide ones)
    # or across very different overall apparatus sizes.
    skel_pruned = _prune_skeleton_spurs(skel, max(3, round(0.01 * min_dim)))
    dt = cv2.distanceTransform((comp_mask * 255).astype(np.uint8), cv2.DIST_L2, 5)

    neighbor_count = _skeleton_neighbor_counts(skel_pruned)
    junctions = (skel_pruned == 1) & (neighbor_count >= 3)
    if not np.any(junctions):
        # A single strand, no branching at all -- e.g. an open-field
        # arena or a round chamber. Nothing to decompose.
        return _whole_blob_as_one_zone()

    # A little breathing room around each junction so an arm's own
    # branch-component doesn't include a stray junction-adjacent pixel,
    # and so the hub-area check below reflects the junction's real
    # footprint rather than a single skeleton pixel.
    junction_halo_px = max(2, round(0.02 * min_dim))
    halo_kernel = np.ones((junction_halo_px * 2 + 1, junction_halo_px * 2 + 1), dtype=np.uint8)
    junctions_dilated = cv2.dilate(junctions.astype(np.uint8), halo_kernel)

    arm_skel = (skel_pruned == 1) & (junctions_dilated == 0)
    n_arm_labels, arm_labels, arm_stats, _ = cv2.connectedComponentsWithStats(
        arm_skel.astype(np.uint8), connectivity=8
    )
    real_arms = []
    for lbl in range(1, n_arm_labels):
        branch_len_px = int(arm_stats[lbl, cv2.CC_STAT_AREA])  # skeleton is 1px wide -> area ~= path length
        if branch_len_px < 3:
            continue
        branch_pixels = arm_labels == lbl
        local_half_width = float(dt[branch_pixels].mean())
        # A real arm reads as clearly LONGER than the corridor is wide; a
        # corner-fork spur is on the order of the corridor's own
        # half-width, rarely more -- the multiplier just needs to sit
        # comfortably between those two cases.
        min_len_for_real_arm = max(8.0, 2.5 * local_half_width)
        if branch_len_px >= min_len_for_real_arm:
            real_arms.append(lbl)

    if len(real_arms) < 2:
        # Not enough surviving branches to call this "branching" -- (0:
        # skeletonization noise; 1: a single strand plus a stray junction
        # pixel from a slightly lumpy trace) -- treat it the same as a
        # non-branching shape.
        return _whole_blob_as_one_zone()

    # Seed markers for cv2.watershed: 0 = unassigned (grown into from
    # whichever numbered seed reaches it first BY SHORTEST PATH THROUGH THE
    # SHAPE, not a straight line -- watershed on a flat/gradient-less image
    # is exactly a geodesic-nearest-seed partition, so it correctly follows
    # a bent or branching corridor instead of jumping across a concave
    # inner corner the way straight-line nearest-neighbor would), 2.. = one
    # per arm (+ the junction hub, if it's more than just a crossing point
    # -- see below).
    #
    # Deliberately NOT marking "outside the traced outline" as its own
    # seed (as an earlier version of this did, with value 1): every arm's
    # seed is only a thin 1-3px-wide line down its own corridor's centre,
    # so anywhere within roughly half the corridor's width of a wall, that
    # wall sits CLOSER (by raw pixel distance) than the arm's own seed
    # line does. With "outside" competing as a real seed, cv2.watershed's
    # geodesic nearest-seed metric then handed a wide strip along BOTH
    # walls of every arm to "outside" instead of to the arm -- shrinking
    # each zone to little more than a thin central stripe of its true
    # corridor (confirmed by measuring a cross-section of a symmetric test
    # corridor: the zone covered under half the traced width). Leaving
    # "outside" unseeded and relying on the barrier image below (so the
    # flood never prefers it over a same-cost interior route) instead lets
    # each arm's seed fill its ENTIRE corridor, wall to wall, contested
    # only by neighboring arms -- which is the only competition that
    # should exist.
    markers = np.zeros((ch, cw), dtype=np.int32)

    seed_kernel = np.ones((3, 3), dtype=np.uint8)
    next_id = 2
    arm_marker_id = {}
    for lbl in real_arms:
        arm_mask = (arm_labels == lbl).astype(np.uint8)
        arm_seed = cv2.dilate(arm_mask, seed_kernel) & comp_mask
        markers[arm_seed == 1] = next_id
        arm_marker_id[lbl] = next_id
        next_id += 1

    hub_area_threshold = max(min_area_fraction * comp_area, 1)
    n_hub_labels, hub_labels, hub_stats, hub_centroids = cv2.connectedComponentsWithStats(
        (junctions_dilated & comp_mask), connectivity=8
    )
    # How close an arm has to pass by a crossing point to be considered one
    # of the arms meeting THERE (see the angle-split below) -- generous
    # relative to the halo itself, since every arm attached to a crossing
    # gets cut by this exact halo and so naturally ends just outside it.
    hub_attach_radius = junction_halo_px * 4
    for lbl in range(1, n_hub_labels):
        if hub_stats[lbl, cv2.CC_STAT_AREA] < hub_area_threshold:
            # Just a crossing point, not a hub zone of its own: left for
            # cv2.watershed's geodesic "nearest seed PIXEL" metric to decide
            # how the arms split it between them. But every arm's seed was
            # cut short of this exact spot by junction_halo_px above, so
            # right here -- where it matters most -- that metric quietly
            # flips from "distance to a long line" (a clean straight
            # bisector, exactly what a crossing of 2+ arms should look
            # like) to "distance to that line's cut-short ENDPOINT" (a
            # curved/parabolic boundary) for whichever neighboring arm's
            # seed happens to still be line-like at that point. The result
            # is boundaries that visibly bulge instead of radiating
            # straight out of the crossing, even for a perfectly symmetric
            # hub. Deciding these specific pixels by plain angle around the
            # crossing's own centroid sidesteps that mismatch entirely --
            # hard-assign them straight into their nearest-angle arm's seed
            # here, before cv2.watershed even runs, so there is nothing
            # left right at the crossing for it to get wrong.
            hcx, hcy = hub_centroids[lbl]
            arm_angle = {}
            for albl in real_arms:
                ys_a, xs_a = np.where(arm_labels == albl)
                d2 = (xs_a - hcx) ** 2 + (ys_a - hcy) ** 2
                near_i = int(np.argmin(d2))
                if d2[near_i] ** 0.5 > hub_attach_radius:
                    continue  # this arm isn't one of the ones meeting at this crossing
                far_i = int(np.argmax(d2))  # the arm's far end sets its outward direction
                arm_angle[albl] = np.arctan2(ys_a[far_i] - hcy, xs_a[far_i] - hcx)

            crossing_mask = (hub_labels == lbl) & (markers == 0)
            ys_c, xs_c = np.where(crossing_mask)
            if ys_c.size and arm_angle:
                pixel_angle = np.arctan2(ys_c - hcy, xs_c - hcx)
                arm_ids = list(arm_angle.keys())
                # Wrapped angular distance from each pixel to each
                # candidate arm's outward direction (handles the +-pi
                # wraparound correctly, unlike a plain subtraction).
                diffs = np.stack(
                    [np.abs(np.angle(np.exp(1j * (pixel_angle - arm_angle[a])))) for a in arm_ids],
                    axis=0,
                )
                nearest = np.argmin(diffs, axis=0)
                for k, albl in enumerate(arm_ids):
                    sel = nearest == k
                    if np.any(sel):
                        markers[ys_c[sel], xs_c[sel]] = arm_marker_id[albl]
            continue  # not promoted to a zone of its own
        hub_seed = (hub_labels == lbl) & (markers == 0)
        if np.any(hub_seed):
            markers[hub_seed] = next_id
            next_id += 1

    # The flood-priority image for cv2.watershed: flat (0) everywhere
    # inside the traced outline, so arm/hub seeds compete there purely by
    # geodesic pixel distance (see above) -- but a solid high value (255)
    # OUTSIDE it, well above anything a seed ever produces, so that region
    # is only ever flooded into AFTER the entire interior is already
    # spoken for. That keeps "outside" from ever being used as a cheap
    # detour between two points that are close in raw pixel space but far
    # apart along the apparatus's actual corridors (e.g. two arm tips that
    # pass near each other on a tightly folded layout) -- the interior-only
    # route always wins first, exactly as if "outside" didn't exist until
    # there's nothing left inside to assign. Whatever label the exterior
    # eventually ends up with from this trailing, arbitrary flood is
    # discarded below (every region is intersected back with comp_mask),
    # so it never matters.
    barrier_img = np.zeros((ch, cw, 3), dtype=np.uint8)
    barrier_img[comp_mask == 0] = 255
    cv2.watershed(barrier_img, markers)

    found = []
    for region_id in range(2, next_id):
        region_mask = ((markers == region_id) & (comp_mask == 1)).astype(np.uint8) * 255
        pts = _mask_to_polygon(region_mask)
        if pts is None:
            continue
        pts_full = [(x + ox, y + oy) for x, y in pts]
        cx = float(np.mean([p[0] for p in pts_full]))
        cy = float(np.mean([p[1] for p in pts_full]))
        found.append((cy, cx, pts_full))
    return found if found else _whole_blob_as_one_zone()


def detect_apparatus_partitions(outline_shapes, width, height, min_area_fraction=0.01):
    """The "trace the whole apparatus" Zone Drawing workflow: the user
    traces the apparatus's own OUTER outline once, as a single closed
    shape (occasionally a couple of disconnected ones -- see "New Shape"
    -- for an apparatus split across the frame), instead of drawing its
    internal walls/dividers by hand or tracing each zone one at a time.
    From that one traced outline, this works out on its own how many
    arms/partitions the shape naturally divides into (an EPM's 4 arms +
    center, a Y-maze's 3 arms, a T-maze's 3, a radial-arm maze's N, ... or,
    for a shape with no branching at all -- a plain open-field arena, a
    round chamber -- just the ONE zone the outline already is) and
    auto-labels each one A, B, C, ... directly on the image; the
    researcher assigns the real arm/zone names afterward in the Zone
    Formula box, same as with the old divider-lines workflow.

    How (see _partition_one_blob for the per-blob detail): fill the
    traced outline(s) into one binary mask; for each disconnected filled
    blob, thin it to its medial-axis skeleton, find where it branches
    (junctions) vs. its individual arm strands, and -- with 2 or more real
    arms -- geodesically grow each arm (plus the junction hub itself, AS
    ITS OWN zone, if it has real area of its own rather than being just a
    crossing point) out to fill the whole blob via watershed. A blob with
    no branching is returned whole, as ONE zone.

    Because the user's own trace is always ONE ALREADY-CLOSED shape (its
    last point auto-connects back to the first, same as every other
    hand-drawn zone/mask outline in this app), there's no equivalent of
    the old divider-lines workflow's "lines didn't quite meet, so open
    space leaked through and silently merged what should've been separate
    zones" failure mode -- an imprecise click just makes the traced
    outline a little lumpy, never a gap in a wall.

    Returns ({letter: [(x, y), ...], ...}, None) on success, or
    ({}, "<message>") if no usable outline was traced.
    """
    shapes = [s for s in outline_shapes if len(s) >= 3]
    if not shapes:
        return {}, "Trace the apparatus's outer outline first (click to add points, then Finish)."

    # "Exactly what I draw": when the apparatus is traced as MULTIPLE
    # shapes (one shape per part, using 'New Shape' between parts), each
    # traced shape becomes exactly ONE zone, labeled A, B, C ... in the
    # order drawn. No merging into a single mask, no skeleton
    # re-interpretation -- the zone geometry IS the traced geometry, so
    # N traced parts -> N zones. A SINGLE traced shape keeps the skeleton
    # auto-partition below (trace the whole apparatus once -> its arms
    # are detected automatically).
    if len(shapes) > 1:
        return {_partition_letter(i): [(float(x), float(y)) for x, y in shape]
                for i, shape in enumerate(shapes)}, None

    filled = np.zeros((height, width), dtype=np.uint8)
    for shape in shapes:
        cv2.fillPoly(filled, [np.array(shape, dtype=np.int32)], 255)

    if not np.any(filled):
        return {}, "The traced outline is empty -- trace the apparatus's outer outline first."

    min_area = min_area_fraction * width * height
    n_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(filled, connectivity=8)

    found = []  # (cy, cx, pts), pooled across every disconnected traced blob
    for comp_label in range(1, n_labels):
        if stats[comp_label, cv2.CC_STAT_AREA] < min_area:
            continue
        x0 = stats[comp_label, cv2.CC_STAT_LEFT]
        y0 = stats[comp_label, cv2.CC_STAT_TOP]
        cw = stats[comp_label, cv2.CC_STAT_WIDTH]
        ch = stats[comp_label, cv2.CC_STAT_HEIGHT]
        pad = 2
        cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
        cx1, cy1 = min(width, x0 + cw + pad), min(height, y0 + ch + pad)
        comp_mask = (labels[cy0:cy1, cx0:cx1] == comp_label).astype(np.uint8)

        found.extend(_partition_one_blob(comp_mask, offset=(cx0, cy0), min_area_fraction=min_area_fraction))

    if not found:
        return {}, "No usable zone was found in the traced outline -- trace the apparatus's outer outline again."

    ordered = _order_reading_order(found, height)
    return {_partition_letter(i): pts for i, (_cy, _cx, pts) in enumerate(ordered)}, None


def draw_roi_polygons(display, roi_points):
    for name, pts in roi_points.items():
        cv2.polylines(display, outline_arrays(pts), True, (0, 0, 0), 1, cv2.LINE_AA)
        cx, cy = (int(v) for v in label_point(pts))
        cv2.putText(display, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# -----------------------------
# Interaction objects (named point + radius) -- new in V6
# -----------------------------


def draw_object_markers(display, object_points):
    """object_points[name] is a polygon (list of points), same shape as
    roi_points -- objects are marked by tracing their actual outline now,
    not a circle that may not match the object's real shape."""
    for i, (name, pts) in enumerate(object_points.items()):
        color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(display, [pts_arr], True, color, 2, cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(display, name, (cx - 15, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def compute_zone_interaction_stats(positions_df, roi_points, object_points,
                                    warp_w, warp_h, fps, min_bout_s=0.3,
                                    scale_factor=None, scale_unit=None,
                                    interaction_margin_px=0):
    """
    Given ONE subject's trajectory (columns: frame, x, y -- e.g. one
    mouse_id's rows from a multi-animal tracker), compute per-zone occupied
    time, per-object interaction bouts, and total distance -- reusing the
    exact same mask/membership/bout machinery the single-animal pipeline
    uses, so multi-animal analysis types get the same zone/object/distance
    capability as Standard Tracking, not a separate re-implementation.

    interaction_margin_px grows each object's own drawn outline outward by
    this many pixels before checking proximity (see dilate_masks) -- so an
    animal approaching/sniffing an object from just outside it still counts
    as "near" that object, not only when its tracked point crosses inside
    the outline. 0 (default) keeps the strict, outline-only behavior.

    Returns (summary_dict, bouts_list). bouts_list entries have the same
    shape as tracking.interaction.extract_bouts's output (Object,
    Frame_start/end, Start/End_seconds, Duration_seconds).
    """
    summary = {}
    xs = positions_df["x"].to_numpy(dtype=float)
    ys = positions_df["y"].to_numpy(dtype=float)
    frames = positions_df["frame"].to_numpy()

    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.hypot(dx, dy)
    seg = seg[~np.isnan(seg)]
    dist_px = float(seg.sum())
    summary["Total_distance_pixels"] = dist_px
    if scale_factor:
        summary[f"Total_distance_{scale_unit or 'units'}"] = dist_px * scale_factor

    if roi_points:
        roi_masks = build_roi_masks((warp_h, warp_w), roi_points)
        for name in roi_points:
            in_zone = np.zeros(len(xs), dtype=bool)
            for i, (x, y) in enumerate(zip(xs, ys)):
                if not (np.isnan(x) or np.isnan(y)):
                    in_zone[i] = roi_membership(roi_masks, x, y).get(name, False)
            time_s = in_zone.sum() / fps
            summary[f"{safe_col(name)}_time_s"] = time_s
            total_s = len(xs) / fps
            summary[f"{safe_col(name)}_percent"] = (time_s / total_s * 100) if total_s > 0 else 0

    bouts = []
    if object_points:
        object_masks = build_roi_masks((warp_h, warp_w), object_points)
        object_masks = dilate_masks(object_masks, interaction_margin_px)
        near_cols = {name: np.zeros(len(xs), dtype=bool) for name in object_points}
        for i, (x, y) in enumerate(zip(xs, ys)):
            if not (np.isnan(x) or np.isnan(y)):
                membership = roi_membership(object_masks, x, y)
                for name in object_points:
                    near_cols[name][i] = membership.get(name, False)
        bout_df = pd.DataFrame({"Frame": frames})
        for name in object_points:
            bout_df[f"Near_{safe_col(name)}"] = near_cols[name]
        bouts = extract_bouts(bout_df, list(object_points.keys()), fps, min_duration=min_bout_s)
        for name in object_points:
            near_time = near_cols[name].sum() / fps
            summary[f"{safe_col(name)}_near_time_s"] = near_time
            summary[f"{safe_col(name)}_bout_count"] = sum(1 for b in bouts if b["Object"] == name)

    return summary, bouts


def draw_crosshair(display, center, size=9, color=(0, 0, 255), thickness=2):
    """A medium '+' marker for the tracked point, instead of a filled circle
    that can hide exactly where the center is."""
    x, y = int(center[0]), int(center[1])
    cv2.line(display, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(display, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def _pack_segments(parts, sep, font, font_scale, thickness, max_width):
    """Greedily pack `parts`, joined by sep, onto as few lines as fit
    within max_width px (measured with cv2.getTextSize). Always makes
    progress -- a single part wider than max_width still gets its own
    line rather than being dropped or looping forever -- so every part
    ends up somewhere in the result even when nothing fits cleanly."""
    lines, current = [], ""
    for part in parts:
        candidate = part if not current else current + sep + part
        if not current or cv2.getTextSize(candidate, font, font_scale, thickness)[0][0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = part
    if current:
        lines.append(current)
    return lines


def wrap_status_line(segments, font, font_scale, thickness, max_width):
    """Wrap ["ROI: ...", "Near: ...", "Entry: ..."]-style segments (Near/
    Entry optional) to fit within a frame max_width px wide, instead of
    drawing them all on one fixed-position cv2.putText line -- which used
    to run straight off a narrower video's right edge unreadably as soon
    as the Entry: list had more than a zone or two in it (a live EPM/
    Y-maze/T-maze run routinely lists 2-3 zones there at "semi"/"half"
    entry depth during a transition). Segments are packed onto as few
    lines as fit, joined by "  | " the same way the single line used to
    be; a segment that's STILL too wide alone on its own line (many zones
    in one Entry:/Near: list) is further split at its own ", ' boundaries
    across as many lines as it needs. Returns the list of lines to draw,
    top to bottom."""
    lines = _pack_segments(segments, "  | ", font, font_scale, thickness, max_width)
    wrapped = []
    for line in lines:
        if cv2.getTextSize(line, font, font_scale, thickness)[0][0] <= max_width:
            wrapped.append(line)
            continue
        label, sep, rest = line.partition(": ")
        if not sep:
            wrapped.append(line)  # nothing to split on -- draw as-is (rare)
            continue
        # Split on the comma but KEEP the comma on every item except the
        # last, and pack with a plain space: the list punctuates itself,
        # so no information is lost no matter how the lines break -- and
        # the result is identical to the input when everything fits on
        # one line. (Splitting on ", " and joining with ", " consumed
        # the comma on line breaks; and OpenCV 4.12's font metrics are
        # slightly wider than the build this was first written against,
        # which is why this only surfaces now.)
        parts = [p.strip() for p in rest.split(",")]
        items = [p + "," for p in parts[:-1]] + parts[-1:]
        items[0] = f"{label}: {items[0]}"
        # Word-level fallback: a single comma-item (usually the first one,
        # carrying its "Label:" prefix) can still exceed max_width -- split
        # it at spaces so every emitted line fits, no matter the OpenCV
        # version's font metrics or how long zone names get.
        final_items = []
        for it in items:
            if cv2.getTextSize(it, font, font_scale, thickness)[0][0] <= max_width:
                final_items.append(it)
            else:
                final_items.extend(it.split(" "))
        wrapped.extend(_pack_segments(final_items, " ", font, font_scale, thickness, max_width))
    return wrapped


def draw_mask_polygons(display, mask_polygons, color=(0, 0, 0)):
    """Draw excluded (masked-out) regions as black outlines with a light
    hatch fill, so they're visibly distinct from zones/objects."""
    overlay = display.copy()
    for pts in mask_polygons:
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_arr], (60, 60, 60))
        cv2.polylines(display, [pts_arr], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, dst=display)


def build_exclusion_mask(shape_hw, mask_polygons, zone_polygons=None, mask_outside_zones=False):
    """Boolean array, True where pixels should be excluded from detection
    entirely (e.g. a food hopper, cage wire, reflection).

    zone_polygons/mask_outside_zones add the "auto-mask everything outside
    my zones" setting: once the zones are drawn/finalized, MM wants
    everything OUTSIDE them excluded automatically, on top of whatever
    Mask Zone shapes were drawn by hand -- so a stray reflection or a
    passing shadow just outside the maze can never be mistaken for the
    animal. This is a no-op unless BOTH mask_outside_zones is True AND
    zone_polygons is non-empty (no zones yet == nothing to mask outside
    of), so it never blanks the whole frame by accident."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in mask_polygons:
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    if mask_outside_zones and zone_polygons:
        outside = np.full((h, w), 255, dtype=np.uint8)
        for pts in zone_polygons:
            if len(pts) >= 3:
                cv2.fillPoly(outside, [np.array(pts, dtype=np.int32)], 0)
        mask = np.maximum(mask, outside)
    return mask > 0


# -----------------------------
# Real-world distance calibration
# -----------------------------


def make_background(cap, start_frame, end_frame, matrix, warp_w, warp_h, sample_count=100, color_mode="gray"):
    sample_count = max(1, int(sample_count))

    if end_frame <= start_frame:
        sample_count = 1

    positions = np.linspace(start_frame, max(start_frame, end_frame - 1), sample_count).astype(int)
    samples = []

    print("\nBuilding automatic background model...")
    print(f"Sampling {len(positions)} frames (median-based). This may take a little time.")

    for frame_number in positions:
        frame = read_and_warp(cap, frame_number, matrix, warp_w, warp_h)

        if frame is None:
            continue

        if color_mode == "rgb":
            sample = cv2.GaussianBlur(frame, (5, 5), 0)
        else:
            sample = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sample = cv2.GaussianBlur(sample, (5, 5), 0)
        samples.append(sample)

    if not samples:
        raise RuntimeError("Could not build background model.")

    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


# -----------------------------
# Automatic mouse detection
# -----------------------------


def detect_mouse(
    frame, background, arena, previous_point, previous_area,
    threshold, min_area, max_area, max_jump,
    roi_masks=None, exclusion_mask=None, color_mode="gray",
    tracking_algorithm="classical", loc_thresh=99.5, use_window=True,
    window_size=200, window_weight=0.9
):
    """Core per-frame detection: background-diff -> threshold (global, or
    per-zone adaptive Otsu when roi_masks is given) -> morphological
    clean-up -> contour candidates -> best-candidate scoring. Deliberately
    kept to this proven, minimal pipeline (matching the light-dark box
    tracker this was benchmarked against) -- no prior-position weighting,
    shadow rejection, or motion-history debris filtering, all of which
    turned out to cause more localized tracking loss (e.g. a dark-zone/
    dark-furred mouse getting misread as a shadow) than they prevented."""
    x1, y1, x2, y2 = arena

    if color_mode == "rgb":
        current = cv2.GaussianBlur(frame, (5, 5), 0)
    else:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(current, (5, 5), 0)

    bg_crop = background[y1:y2, x1:x2]
    current_crop = current[y1:y2, x1:x2]
    diff = cv2.absdiff(current_crop, bg_crop).astype(np.float32)
    if color_mode == "rgb" and diff.ndim == 3:
        # Max across B/G/R rather than mean: catches an animal that stands
        # out strongly in just one channel (e.g. reddish fur on a green
        # floor) even when its overall grayscale brightness barely differs
        # from the background.
        diff = diff.max(axis=2)

    if exclusion_mask is not None:
        # Zero out masked-out regions (e.g. a food hopper, cage wire,
        # reflection) so they can never register as a detection candidate --
        # same technique as the classic engine (dif[mask] = 0).
        diff[exclusion_mask[y1:y2, x1:x2]] = 0

    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)

    if roi_masks:
        mask = np.zeros(diff_u8.shape, dtype=np.uint8)
        assigned = np.zeros(diff_u8.shape, dtype=bool)

        for full_mask in roi_masks.values():
            zone_mask = full_mask[y1:y2, x1:x2] & (~assigned)
            pixels = diff_u8[zone_mask]

            if pixels.size == 0:
                continue

            otsu_val, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            zone_threshold = max(float(threshold), float(otsu_val))
            zone_binary = (diff_u8 >= zone_threshold).astype(np.uint8) * 255
            mask[zone_mask] = zone_binary[zone_mask]
            assigned |= zone_mask

        leftover = ~assigned
        leftover_pixels = diff_u8[leftover]

        if leftover_pixels.size:
            otsu_val, _ = cv2.threshold(leftover_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            zone_threshold = max(float(threshold), float(otsu_val))
            zone_binary = (diff_u8 >= zone_threshold).astype(np.uint8) * 255
            mask[leftover] = zone_binary[leftover]
    else:
        _, mask = cv2.threshold(diff_u8, int(threshold), 255, cv2.THRESH_BINARY)

    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = (moments["m10"] / moments["m00"]) + x1
        cy = (moments["m01"] / moments["m00"]) + y1

        bx, by, bw, bh = cv2.boundingRect(contour)
        bx += x1
        by += y1

        if bw <= 0 or bh <= 0:
            continue

        aspect = max(bw, bh) / max(1, min(bw, bh))
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        candidates.append({
            "center": (cx, cy),
            "area": float(area),
            "bbox": (bx, by, bw, bh),
            "aspect": float(aspect),
            "circularity": float(circularity),
            "distance": point_distance((cx, cy), previous_point),
            "contour": contour
        })

    if not candidates:
        return None, mask

    best = None
    best_score = float("inf")

    for c in candidates:
        d = c["distance"]

        if previous_point is not None:
            if d > max_jump:
                continue

            score = d

            if previous_area is not None and previous_area > 0:
                ratio = c["area"] / previous_area
                score += 30 * abs(math.log(max(ratio, 1e-6)))

            if c["aspect"] > 8:
                score += 20
            if c["circularity"] < 0.02:
                score += 10
        else:
            score = 0
            if c["aspect"] > 8:
                score += 30
            score -= min(c["area"], 1000) * 0.01

        if score < best_score:
            best_score = score
            best = c

    return best, mask


# -----------------------------
# Recovery using local search
# -----------------------------


def local_recovery(frame, background, previous_point, arena, threshold, min_area, max_area,
                    exclusion_mask=None, color_mode="gray"):
    if previous_point is None:
        return None

    cx, cy = map(int, previous_point)
    radius = 100

    x1 = max(arena[0], cx - radius)
    y1 = max(arena[1], cy - radius)
    x2 = min(arena[2], cx + radius)
    y2 = min(arena[3], cy + radius)

    if x2 <= x1 or y2 <= y1:
        return None

    if color_mode == "rgb":
        current = frame[y1:y2, x1:x2]
    else:
        current = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    bg = background[y1:y2, x1:x2]
    diff = cv2.absdiff(current, bg)
    if color_mode == "rgb" and diff.ndim == 3:
        diff = diff.max(axis=2)

    if exclusion_mask is not None:
        diff[exclusion_mask[y1:y2, x1:x2]] = 0

    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_d = float("inf")

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area or area > max_area:
            continue

        m = cv2.moments(c)

        if m["m00"] == 0:
            continue

        px = (m["m10"] / m["m00"]) + x1
        py = (m["m01"] / m["m00"]) + y1
        d = point_distance((px, py), previous_point)

        if d < best_d:
            best_d = d
            bx, by, bw, bh = cv2.boundingRect(c)
            best = {"center": (px, py), "area": float(area), "bbox": (bx + x1, by + y1, bw, bh)}

    return best


# -----------------------------
# Zone transitions / binning
# -----------------------------


def calculate_transitions(df, confirmation_frames=3, roi_col="ROI"):
    """Turn a per-frame zone-label column into a From/To transitions table.
    roi_col defaults to "ROI" (the centroid-based zone label) but can be
    pointed at any other linearize_roi()-style column -- e.g. a Zone
    Associations group label -- for a different zone grouping."""
    valid = df[df["Tracking_Status"].isin(PRESENT)]
    columns = ["From_ROI", "To_ROI", "Transition_seconds"]

    if valid.empty:
        return pd.DataFrame(columns=columns)

    events = []
    current_roi = None
    candidate_roi = None
    candidate_count = 0

    for _, row in valid.iterrows():
        roi = row[roi_col]
        t = float(row["Time_seconds"])

        if current_roi is None:
            current_roi = roi
            continue

        if roi == current_roi:
            candidate_roi = None
            candidate_count = 0
            continue

        if candidate_roi != roi:
            candidate_roi = roi
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= confirmation_frames:
            events.append({"From_ROI": current_roi, "To_ROI": roi, "Transition_seconds": t})
            current_roi = roi
            candidate_roi = None
            candidate_count = 0

    return pd.DataFrame(events, columns=columns)


def run_detection_preview(
    cap, matrix, warp_w, warp_h, start_frame, end_frame,
    background, arena, roi_masks, roi_points, object_points,
    threshold, min_area, max_area, n_samples, output_dir, show_live=False,
    mask_polygons=None, exclusion_mask=None, color_mode="gray", tracking_algorithm="classical", loc_thresh=99.5, use_window=True, window_size=200, window_weight=0.9
):
    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(preview_dir, "preview_*.png")):
        os.remove(stale)

    n_samples = max(1, int(n_samples))
    last_valid_frame = max(start_frame, end_frame - 1)
    positions = np.linspace(start_frame, last_valid_frame, n_samples).astype(int)

    print(f"\nSaving {len(positions)} calibration preview frame(s) to:\n{preview_dir}")
    if show_live:
        print("Press any key to advance, ESC to cancel and retune.")

    for i, pframe in enumerate(positions, start=1):
        frame = read_and_warp(cap, pframe, matrix, warp_w, warp_h)

        if frame is None:
            continue

        candidate, _ = detect_mouse(
            frame, background, arena, None, None,
            threshold, max(1, int(min_area)), max(int(max_area), int(min_area) + 1),
            float("inf"), roi_masks=roi_masks, exclusion_mask=exclusion_mask, color_mode=color_mode,
            tracking_algorithm=tracking_algorithm, loc_thresh=loc_thresh, use_window=use_window,
            window_size=window_size, window_weight=window_weight
        )

        disp = frame.copy()
        cv2.rectangle(disp, (arena[0], arena[1]), (arena[2], arena[3]), (255, 255, 0), 1)
        if mask_polygons:
            draw_mask_polygons(disp, mask_polygons)
        draw_roi_polygons(disp, roi_points)
        draw_object_markers(disp, object_points)

        if candidate is not None:
            cx, cy = candidate["center"]
            draw_crosshair(disp, (cx, cy), size=9, color=(0, 0, 255), thickness=2)
            cv2.putText(
                disp, f"area={candidate['area']:.0f}",
                (max(5, int(cx) - 40), max(20, int(cy) - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        else:
            cv2.putText(disp, "NO DETECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(
            disp, f"Preview {i}/{len(positions)}  frame {int(pframe)}",
            (20, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        fname = os.path.join(preview_dir, f"preview_{i:02d}_frame{int(pframe)}.png")
        cv2.imwrite(fname, disp)

        if show_live:
            cv2.imshow("Detection preview", disp)
            key = cv2.waitKey(0) & 0xFF

            if key == 27:
                cv2.destroyWindow("Detection preview")
                raise SystemExit("Cancelled during preview. Rerun with adjusted threshold/min_area/max_area.")

    if show_live:
        cv2.destroyWindow("Detection preview")


# -----------------------------
# Calibration (step-back wizard)
# -----------------------------


RESULTS_ROOT = None        # one folder for every result (Settings ▸ Results folder); None = next to each video


def results_root(video_path):
    """Where the results of a video go: the chosen results folder, or a 'results' folder next to the video."""
    if RESULTS_ROOT:
        return str(RESULTS_ROOT)
    return os.path.join(os.path.dirname(os.path.abspath(str(video_path))), "results")


def compute_output_dir(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(results_root(video_path), video_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# -----------------------------
# Save/load a reusable setup (zones, objects, thresholds -- not the video
# itself, not per-video state like fps/duration/first_frame)
# -----------------------------


def smooth_trajectory(df, window, x_col="Mouse_X", y_col="Mouse_Y",
                       out_x_col="Mouse_X_smooth", out_y_col="Mouse_Y_smooth",
                       outlier_mad_multiple=3.5):
    """Optional post-processing pass (Upgrade Plan Tier 1 #4, mirrors
    SMART's Anti-Vibration/Anti-Artifact/LOWESS filters) -- adds
    <x_col>_smooth/<y_col>_smooth columns WITHOUT touching the original
    x_col/y_col, so raw_tracking.csv keeps the real, unmodified per-frame
    detections; process_single_video only points the Trajectory/Heatmap
    charts at the smoothed columns when this was actually requested. Two
    steps, applied independently to each column and only across TRACKED
    frames (a Lost frame has no position to smooth either way):

    1. Anti-artifact: a single frame whose position sits far from its own
       local rolling-median neighborhood (more than outlier_mad_multiple
       times the local median absolute deviation) is treated as one-frame
       detection noise -- a momentary snap onto a shadow/reflection/debris
       blob -- and replaced by that local median before smoothing, rather
       than being allowed to drag the average off just for its own window.
    2. Anti-vibration / LOWESS-style smoothing: a centered rolling mean
       over `window` frames evens out the remaining frame-to-frame jitter
       that real background-subtraction detection always has a little of.

    window < 2 (or fewer than 2 tracked frames) is a no-op: both new
    columns are just copies of the originals.
    """
    df = df.copy()
    tracked_mask = df["Tracking_Status"] == "Tracked"
    df[out_x_col] = df[x_col]
    df[out_y_col] = df[y_col]
    if not window or window < 2 or tracked_mask.sum() < 2:
        return df

    for col, out_col in ((x_col, out_x_col), (y_col, out_y_col)):
        series = df.loc[tracked_mask, col].astype(float)

        # 1) Anti-artifact: flag/replace single-frame outliers first, so
        # step 2's averaging isn't itself dragged off by them.
        rolling_median = series.rolling(window, center=True, min_periods=1).median()
        mad = (series - rolling_median).abs().rolling(window, center=True, min_periods=1).median()
        # A near-zero MAD (a genuinely still animal) would make even
        # sub-pixel noise look like a huge multiple of it -- floor it so a
        # still animal's own tiny jitter is never misflagged as an outlier.
        mad_floor = mad.clip(lower=0.5)
        is_outlier = (series - rolling_median).abs() > (outlier_mad_multiple * mad_floor)
        cleaned = series.where(~is_outlier, rolling_median)

        # 2) Smooth what's left with a centered rolling mean.
        smoothed = cleaned.rolling(window, center=True, min_periods=1).mean()
        df.loc[tracked_mask, out_col] = smoothed

    return df


# =============================================================================
# tracking pipeline: robust detection engine + sequential frame reading
# =============================================================================


def auto_mask_margin(warp_w, warp_h):
    """How far (px) the zones are grown before 'mask everything outside the zones' is applied.  The mask
    follows the zone lines tightly (1 px: only the line itself stays visible); a wider gap can be set in
    the settings ('Gap between the zones and the mask')."""
    return 1


def polarity_warning_text(chosen, looks):
    """Plain-words warning: the animal colour chosen by hand does not match the video."""
    words = {"dark": "DARKER than the floor (black mouse)", "light": "LIGHTER than the floor (white mouse)"}
    if chosen == "any":
        return (f"The animal looks {words.get(looks, looks)}, but 'Animal colour' is 'Either (absolute difference)', "
                "which also follows light changes, shadows and reflections. Set Animal colour to "
                f"'{'Darker than floor' if looks == 'dark' else 'Lighter than floor'}' for a steadier track.")
    return (f"The animal looks {words.get(looks, looks)}, but 'Animal colour' was set to "
            f"'{words.get(chosen, chosen)}'. With the wrong colour the tracker can follow an empty spot "
            "(where the animal sat) instead of the animal -- set Animal colour to "
            f"'{'Darker than floor' if looks == 'dark' else 'Lighter than floor'}' (or Auto-detect) and track again.")


def low_light_zone_polys(setup):
    """{name: polygon} of the zones the animal can be seen in (covered zones left out) -- what the 'Auto'
    low-light boost measures."""
    cov = set(covered_zone_names(setup))
    return {n: p for n, p in (setup.get("roi_points") or {}).items()
            if n not in cov and p is not None and len(p) >= 3}


def build_detection_exclusion(warp_w, warp_h, mask_polygons=None, roi_points=None,
                              auto_mask_outside_zones=False, margin=None, covered=None):
    """Boolean (H, W) array, True = ignored by the detector.  Hand-drawn mask
    polygons plus -- when auto_mask_outside_zones is on and zones exist --
    everything outside the (slightly grown) union of the zones.  `covered` = names of covered zones
    (the animal cannot be seen inside them, e.g. the lid of a light/dark box): ignored as well."""
    h, w = int(warp_h), int(warp_w)
    mask = np.zeros((h, w), np.uint8)
    for pts in mask_polygons or []:
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    covered = [n for n in (covered or []) if roi_points and n in roi_points]
    if auto_mask_outside_zones and roi_points:
        inside = np.zeros((h, w), np.uint8)
        for name, pts in roi_points.items():
            if len(pts) >= 3 and name not in covered:
                cv2.fillPoly(inside, [np.array(pts, dtype=np.int32)], 255)
        if inside.any():
            r = auto_mask_margin(w, h) if margin is None else int(margin)
            if r > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                inside = cv2.dilate(inside, k)
            mask[inside == 0] = 255
    for name in covered:
        pts = roi_points[name]
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask > 0


def covered_zone_names(setup):
    """Zones the animal is hidden in (a lid over the dark box...): set per zone in the zone editor; for the
    light/dark box the dark compartment counts as covered unless 'dark box has a lid' is switched off."""
    roi = setup.get("roi_points") or {}
    names = [n for n in (setup.get("covered_zones") or []) if n in roi]
    if setup.get("covered_zones") is None and setup.get("test_type") == "light_dark" \
            and setup.get("ldb_dark_covered", True):
        names = [n for n in roi if any(w in str(n).lower() for w in ("dark", "black"))]
    return names


def detection_guides(setup, warp_w, warp_h):
    """(core, covered) full-resolution bool masks for the tracker, or (None, None).
    core    = where the animal's CENTRE may be: the open zones + a small tolerance (only with the
              'mask outside the zones' option on);
    covered = the covered zones (the animal disappears into them and comes back out next to them)."""
    roi = {k: v for k, v in (setup.get("roi_points") or {}).items() if len(v) >= 3}
    if not roi:
        return None, None
    h, w = int(warp_h), int(warp_w)
    cov_names = covered_zone_names(setup)
    covered = None
    if cov_names:
        covered = np.zeros((h, w), np.uint8)
        for n in cov_names:
            cv2.fillPoly(covered, [np.array(roi[n], dtype=np.int32)], 255)
        covered = covered > 0
    core = None
    if setup.get("auto_mask_outside_zones"):
        inside = np.zeros((h, w), np.uint8)
        for n, pts in roi.items():
            if n not in cov_names:
                cv2.fillPoly(inside, [np.array(pts, dtype=np.int32)], 255)
        if inside.any():
            m = setup.get("auto_mask_margin")
            r = max(3, int(round(0.5 * (auto_mask_margin(w, h) if m is None else int(m)))))
            inside = cv2.dilate(inside, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            core = inside > 0
            if covered is not None:
                core &= ~covered
    return core, covered


def _is_identity_warp(matrix, frame_w, frame_h, warp_w, warp_h):
    try:
        m = np.asarray(matrix, dtype=np.float64)
        return (m.shape == (3, 3) and np.allclose(m, np.eye(3), atol=1e-9)
                and int(frame_w) == int(warp_w) and int(frame_h) == int(warp_h))
    except Exception:
        return False


def path_distance(xs, ys, valid, min_move_px=1.0):
    """Per-frame distance (px) with a jitter filter: movement is accumulated
    only once the animal has moved at least `min_move_px` away from the
    last counted position (EthoVision's 'minimal distance moved').  Without
    it, sub-pixel centroid jitter of a resting animal adds metres of fake
    distance over a long session."""
    n = len(xs)
    out = np.zeros(n, dtype=float)
    ax = ay = None
    for i in range(n):
        if not valid[i] or not np.isfinite(xs[i]) or not np.isfinite(ys[i]):
            continue
        if ax is None:
            ax, ay = xs[i], ys[i]
            continue
        d = math.hypot(xs[i] - ax, ys[i] - ay)
        if d >= min_move_px:
            out[i] = d
            ax, ay = xs[i], ys[i]
    return out


def zone_flags(masks, xs, ys, valid):
    """{name: bool array} -- membership of every frame's point in every mask."""
    out = {}
    n = len(xs)
    if not masks:
        return out
    any_mask = next(iter(masks.values()))
    h, w = any_mask.shape
    ok = valid & np.isfinite(xs) & np.isfinite(ys)
    xi = np.clip(np.round(np.nan_to_num(xs)).astype(int), 0, w - 1)
    yi = np.clip(np.round(np.nan_to_num(ys)).astype(int), 0, h - 1)
    inside_frame = ok & (np.round(np.nan_to_num(xs)) >= 0) & (np.round(np.nan_to_num(xs)) < w) \
        & (np.round(np.nan_to_num(ys)) >= 0) & (np.round(np.nan_to_num(ys)) < h)
    for name, m in masks.items():
        f = np.zeros(n, bool)
        f[inside_frame] = m[yi[inside_frame], xi[inside_frame]]
        out[name] = f
    return out


PRESENT = ("Tracked", "Hidden")     # statuses that count as 'the animal is in the apparatus' for zone times


def hold_last_position(xs, ys, tracked, skip):
    """Frames without a position (and not 'hidden' / marked 'not visible' = `skip`): the animal is taken
    to be where it was last seen (at the start of the window: where it is first seen).  xs / ys are filled
    in place.  Returns the bool array of the frames filled this way."""
    n = len(xs)
    held = np.zeros(n, bool)
    ok = np.asarray(tracked, bool) & np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return held
    skip = np.asarray(skip, bool)
    first = int(np.argmax(ok))
    last_x, last_y = xs[first], ys[first]
    for i in range(n):
        if ok[i]:
            last_x, last_y = xs[i], ys[i]
        elif not skip[i]:
            xs[i], ys[i] = last_x, last_y
            held[i] = True
    return held


def hidden_in_covered(xs, ys, tracked, covered_masks, body_len):
    """Frames the animal spends inside a covered zone (it cannot be seen there).  A run of frames without
    the animal counts as hidden in a covered zone when the animal was last seen next to that zone before
    the run AND is seen next to it again after it (or the run is at the start / end of the video).
    Returns (bool array, zone-name array)."""
    n = len(xs)
    hidden = np.zeros(n, bool)
    zone = np.empty(n, dtype=object)
    zone[:] = ""
    if not covered_masks or n == 0:
        return hidden, zone
    any_mask = next(iter(covered_masks.values()))
    h, w = any_mask.shape
    near_px = max(1.5 * float(body_len or 30.0), 0.05 * min(h, w))
    dist = {k: cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3) for k, m in covered_masks.items()}

    def nearest(i):
        if i is None or not (np.isfinite(xs[i]) and np.isfinite(ys[i])):
            return None
        xi = int(min(max(round(xs[i]), 0), w - 1))
        yi = int(min(max(round(ys[i]), 0), h - 1))
        best = min(dist, key=lambda k: dist[k][yi, xi])
        return best if dist[best][yi, xi] <= near_px else None

    seen = np.asarray(tracked, bool)
    i = 0
    while i < n:
        if seen[i]:
            i += 1
            continue
        j = i
        while j < n and not seen[j]:
            j += 1
        prev = i - 1 if i > 0 else None
        nxt = j if j < n else None
        zp, zn = nearest(prev), nearest(nxt)
        z = None
        if prev is None and nxt is not None:
            z = zn
        elif nxt is None and prev is not None:
            z = zp
        elif prev is not None and nxt is not None and zp is not None and zp == zn:
            z = zp
        if z is not None:
            hidden[i:j] = True
            zone[i:j] = z
        i = j
    return hidden, zone


ENTRY_BODY_PCT = 80.0     # zone entry = this % of the animal's body (tail removed) inside the zone ~ all four paws


class BodyZoneFractions:
    """Per frame: the fraction of the detected body (tail removed) that lies inside each zone.  Used for the
    'all four paws in the zone' entry rule.  Works on the tracker's detection-scale body mask."""

    def __init__(self, roi_masks, names):
        self.names = [n for n in names if n in roi_masks]
        self.full = [roi_masks[n] for n in self.names]
        self._scaled, self._shape = None, None

    def _masks(self, shape):
        if self._scaled is None or self._shape != shape:
            h, w = shape
            self._scaled = [cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                            for m in self.full]
            self._shape = shape
        return self._scaled

    def __call__(self, tracker, det):
        n = len(self.names)
        nan = (float("nan"),) * n
        if not n or det.get("status") != "Tracked":
            return nan
        body = getattr(tracker, "last_body", None)
        valid = getattr(tracker, "valid", None)
        if body is None or valid is None:
            return nan
        mask, x0, y0 = body
        tot = float(mask.sum())
        if tot <= 0:
            return nan
        out = []
        for m in self._masks(valid.shape):
            sub = m[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]]
            mm = mask[:sub.shape[0], :sub.shape[1]]
            out.append(round(float((sub & mm).sum()) / tot, 3))
        return tuple(out)


def disc_zone_fractions(roi_masks, names, x, y, radius):
    """Body fractions for a position set by hand: the body is taken as a disc of the animal's size."""
    out = []
    r = max(2, int(round(radius)))
    for n in names:
        m = roi_masks.get(n)
        if m is None or not np.isfinite(x) or not np.isfinite(y):
            out.append(float("nan"))
            continue
        h, w = m.shape
        x0, x1 = max(0, int(x) - r), min(w, int(x) + r + 1)
        y0, y1 = max(0, int(y) - r), min(h, int(y) + r + 1)
        if x1 <= x0 or y1 <= y0:
            out.append(0.0)
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        disc = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
        tot = float(disc.sum()) or 1.0
        out.append(round(float((m[y0:y1, x0:x1] & disc).sum()) / tot, 3))
    return tuple(out)


class _LiveDrawer:
    """Live-view frames drawn on a small copy (<= max_w wide): the screen shows them small anyway, and
    drawing on a 1080p / 4K frame costs more than tracking it.  Saved images are not affected."""

    def __init__(self, w, h, roi_points, mask_polygons, exclusion, max_w=960, show_mask=True):
        self.show_mask = bool(show_mask)
        self.s = min(1.0, float(max_w) / float(max(w, 1)))
        s = self.s
        self.size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
        self.roi = {k: [(float(x) * s, float(y) * s) for x, y in pts] for k, pts in roi_points.items()}
        self.masks = [[(float(x) * s, float(y) * s) for x, y in pts] for pts in (mask_polygons or [])]
        self.excl = exclusion
        if exclusion is not None and s < 1.0:
            self.excl = cv2.resize(exclusion.astype(np.uint8), self.size, interpolation=cv2.INTER_NEAREST) > 0

    def draw(self, frame, det, tracker, t, frame_number):
        if self.s < 1.0:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            det = dict(det)
            if det.get("status") == "Tracked":
                det["x"], det["y"] = det["x"] * self.s, det["y"] * self.s
                for key in ("head", "tail"):
                    if det.get(key):
                        det[key] = (det[key][0] * self.s, det[key][1] * self.s)
        return _draw_live(frame, self.roi, self.masks, self.excl, det, tracker, t, frame_number,
                          show_mask=self.show_mask)


def _draw_live(frame, roi_points, mask_polygons, exclusion_dim, det, tracker, t, frame_number, show_mask=True,
               lost_label="SEARCHING..."):
    from tracking.robust_tracker import brighten
    g = float(getattr(getattr(tracker, "bg", None), "gamma", 1.0) or 1.0)
    disp = brighten(frame, g) if g > 1.001 else frame.copy()      # low-light boost: shown brightened too
    if exclusion_dim is not None:
        disp[exclusion_dim] = (disp[exclusion_dim] * 0.35).astype(np.uint8)
    if show_mask and tracker is not None and det["status"] == "Tracked":
        disp = tracker.overlay(disp, color=(0, 215, 255))
    lw = zone_line_width(disp)
    for i, (name, pts) in enumerate(roi_points.items()):
        col = ROI_COLORS[i % len(ROI_COLORS)]
        cv2.polylines(disp, outline_arrays(pts), True, col, lw, cv2.LINE_AA)
    if mask_polygons:
        draw_mask_polygons(disp, mask_polygons)
    size = max(8, int(min(disp.shape[:2]) * 0.02))
    if det["status"] == "Tracked":
        x, y = int(round(det["x"])), int(round(det["y"]))
        cv2.circle(disp, (x, y), size + 3, (255, 255, 255), 2, cv2.LINE_AA)
        draw_crosshair(disp, (x, y), size=size, color=(0, 0, 255), thickness=2)
        draw_head_tail(disp, det.get("head"), det.get("tail"), size)
        label = "TRACKED"
        col = (40, 200, 40)
    else:
        label = lost_label
        col = (0, 0, 255)
    return draw_status_box(disp, f"{label}   {clock_text(t)}   frame {frame_number}", col)


def detection_still(setup, bgm, exclusion, frame, x, y, status, t, frame_number, roi_points, mask_polygons,
                    lost_label="NOT SEEN"):
    """One frame drawn with the animal found there (by a whole tracking run, or by the start-time look-ahead):
    its body tinted yellow, the marker, zones and masks."""
    probe = new_tracker(setup, bgm, exclusion)
    ok = status == "Tracked" and x is not None and np.isfinite(x) and np.isfinite(y)
    det = {"x": float(x) if ok else np.nan, "y": float(y) if ok else np.nan, "area": np.nan,
           "status": "Tracked" if ok else "Lost", "n_candidates": 0}
    if ok and hasattr(probe, "candidates"):
        d = probe.strength(frame)
        cands = probe.candidates(d)
        s = float(probe.s)
        px, py = (float(x) + 0.5) * s - 0.5, (float(y) + 0.5) * s - 0.5
        best, bd = None, max(float(probe.body_l), 6.0)
        for c in cands:
            dist = math.hypot(c["cx"] - px, c["cy"] - py)
            if dist <= bd:
                best, bd = c, dist
        m = np.zeros(d.shape[:2], np.uint8)
        if best is not None:
            bx, by, bw, bh = best["bbox"]
            m[by:by + bh, bx:bx + bw][best["full"]] = 255
            det["area"] = best["area"] / (s * s)
        probe.last_mask = m
    elif ok:
        probe.update(frame)
    return _draw_live(frame, roi_points, mask_polygons, exclusion, det, probe, t, frame_number, lost_label=lost_label)


def zone_line_width(img):
    """Zone outlines as thin as a thread: 1 px (2 px only on very large pictures, which are shown smaller)."""
    return max(1, int(round(max(img.shape[:2]) / 1100.0)))


HEAD_BGR = (255, 200, 0)        # head (nose): light blue
TAIL_BGR = (200, 60, 255)       # tail base: pink


def draw_head_tail(disp, head, tail, size=10):
    """Head = blue dot, tail base = pink dot (BGR picture, drawn in place)."""
    r = max(3, int(round(size * 0.45)))
    for p, col in ((tail, TAIL_BGR), (head, HEAD_BGR)):
        if p is not None and np.isfinite(p[0]) and np.isfinite(p[1]):
            c = (int(round(p[0])), int(round(p[1])))
            cv2.circle(disp, c, r + 2, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(disp, c, r, col, -1, cv2.LINE_AA)


def clock_text(t):
    """hh:mm:ss.ss -- the same text on the live picture and on the clock under it."""
    t = max(0.0, float(t))
    h, rem = divmod(int(t), 3600)
    m = rem // 60
    return f"{h:02d}:{m:02d}:{t - h * 3600 - m * 60:05.2f}"


def draw_status_box(disp, txt, col=(40, 200, 40)):
    """Status line drawn in place: black text on a white box (readable on any
    video) with a small coloured status dot."""
    fs = max(0.5, min(disp.shape[:2]) / 900.0)
    th = max(1, int(round(1.6 * fs)))
    (tw, tht), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(8 * fs)
    dot = int(7 * fs)
    x0, y0 = 10, 10
    x1, y1 = x0 + pad * 3 + 2 * dot + tw, y0 + tht + base + 2 * pad
    box = disp[y0:y1, x0:x1]
    if box.size:
        disp[y0:y1, x0:x1] = (box * 0.15 + 255 * 0.85).astype(np.uint8)
    cy = (y0 + y1) // 2
    cv2.circle(disp, (x0 + pad + dot, cy), dot, col, -1, cv2.LINE_AA)
    cv2.putText(disp, txt, (x0 + 2 * pad + 2 * dot, y0 + pad + tht), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th,
                cv2.LINE_AA)
    return disp


def original_view_image(video_path, frame_indices, warp=None, animal_xy=None, animal_radius=None):
    """Colour, full-resolution, UN-masked picture of the arena for the
    trajectory / heat-map charts -- it looks like the original video (crop
    applied, no detection masks, no auto-mask dimming, no downscaling).

    Per-pixel median of a few frames.  animal_xy[i] (or None) is where the
    animal was in frame_indices[i]; that spot of that frame is left out of
    the median so an animal that rests in one place does not show up.
    Returns uint8 BGR or None."""
    import warnings
    got = read_frames_parallel(str(video_path), list(frame_indices), (warp if warp is not None else (lambda f: f)))
    frames, masks = [], []
    try:
        for k, f in enumerate(got):
            if f is None:
                continue
            frames.append(f)
            m = None
            if animal_xy is not None and animal_radius and k < len(animal_xy) and animal_xy[k] is not None:
                x, y = animal_xy[k]
                if np.isfinite(x) and np.isfinite(y):
                    m = np.zeros(f.shape[:2], np.uint8)
                    cv2.circle(m, (int(round(x)), int(round(y))), int(round(animal_radius)), 1, -1)
                    m = m.astype(bool)
            masks.append(m)
    finally:
        got = None
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    h, w = frames[0].shape[:2]
    frames = [f if f.shape[:2] == (h, w) else cv2.resize(f, (w, h)) for f in frames]
    stack = np.stack(frames)                              # K,H,W,3 uint8
    any_mask = any(m is not None for m in masks)
    mstack = (np.stack([m if m is not None and m.shape == (h, w) else np.zeros((h, w), bool) for m in masks])
              if any_mask else None)
    out = np.empty((h, w, 3), np.uint8)
    step = 96
    for y0 in range(0, h, step):
        chunk = stack[:, y0:y0 + step]
        med = np.median(chunk, axis=0)
        if mstack is not None:
            mc = mstack[:, y0:y0 + step]
            hit = mc.any(axis=0)
            if hit.any():
                vals = chunk[:, hit].astype(np.float32)   # K,n,3
                vals[mc[:, hit]] = np.nan
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    alt = np.nanmedian(vals, axis=0)
                good = np.isfinite(alt).all(axis=1)
                sub = med[hit]
                sub[good] = alt[good]
                med[hit] = sub
        out[y0:y0 + step] = np.clip(np.round(med), 0, 255).astype(np.uint8)
    return out


# ----------------------------------------------------------------------------- speed: parallel video reading
SEQ_GAP = 50           # reach a frame this close by reading forward (one seek costs about as much as ~50 reads)


def worker_count(n_frames=None, min_frames_per_worker=1800, cap=4):
    """How many decoders/trackers to run at once: about one per 2 CPU cores (one uses ~1.6),
    at most `cap`, and only when every piece gets >= min_frames_per_worker frames."""
    cores = os.cpu_count() or 2
    w = 1 if cores < 2 else max(2, min(cap, cores // 2))
    if n_frames is not None:
        w = max(1, min(w, int(n_frames) // int(min_frames_per_worker)))
    return w


def read_samples_parallel(video_path, positions, warp, color_mode, scale, progress=None, workers=None):
    """prepare_raw()d frames at `positions` (same order; None where unreadable), read by several decoders at
    once.  Close positions are reached by reading forward, far ones by seeking.  Identical pictures to
    seeking to every position one by one -- only faster."""
    from tracking.robust_tracker import prepare_raw
    return read_frames_parallel(video_path, positions, lambda f: prepare_raw(warp(f), color_mode, scale),
                                progress=progress, workers=workers)


def read_frames_parallel(video_path, positions, fn, progress=None, workers=None):
    """fn(frame) for the frames at `positions` (same order; None where unreadable), several decoders at once."""
    import threading
    positions = [int(p) for p in positions]
    n = len(positions)
    out = [None] * n
    if n == 0:
        return out
    nw = max(1, min(workers or min(4, os.cpu_count() or 2), n))
    chunks = [range(k * n // nw, (k + 1) * n // nw) for k in range(nw)]
    done = [0] * nw
    errors = []
    stop = threading.Event()

    def work(k):
        cap = cv2.VideoCapture(video_path)
        try:
            cur = None
            for i in chunks[k]:
                if stop.is_set():
                    return
                pos = positions[i]
                if cur is not None and 0 <= pos - cur - 1 <= SEQ_GAP:
                    for _ in range(pos - cur - 1):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ok, f = cap.read()
                cur = pos
                if ok and f is not None:
                    out[i] = fn(f)
                done[k] += 1
        except Exception as exc:
            errors.append(exc)
        finally:
            cap.release()

    threads = [threading.Thread(target=work, args=(k,), name=f"bt-bg{k}", daemon=True) for k in range(nw)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
            if progress is not None:
                progress(0.55 * sum(done) / n)
    except BaseException:
        stop.set()
        for t in threads:
            t.join(timeout=5)
        raise
    if errors:
        raise errors[0]
    return out


def apply_backfill(det, xs, ys, areas, status, ends=None, fracs=None):
    """The tracker confirmed the animal after a few frames reported 'Lost' while it was being checked: put
    those frames' positions in (lists are changed in place).  With 'backfill_override' the frames were
    tracked on a wrong spot and are replaced (their body ends / body fractions are dropped)."""
    bf = det.get("backfill") if isinstance(det, dict) else None
    if not bf:
        return
    n = len(bf)
    force = bool(det.get("backfill_override"))
    for q, (bx, by, ba) in enumerate(bf):
        i = len(xs) - n + q
        if 0 <= i < len(xs) and (force or status[i] != "Tracked"):
            xs[i], ys[i], areas[i], status[i] = float(bx), float(by), float(ba), "Tracked"
            if force and ends is not None and i < len(ends):
                ends[i] = None
            if force and fracs is not None and i < len(fracs) and fracs[i]:
                fracs[i] = tuple(float("nan") for _ in fracs[i])


def _live_meta(frame_callback, frame_number, t):
    """Tell the live view which frame / video time the next picture shows (so the player's clock and slider
    show exactly the same moment as the picture)."""
    fn = getattr(frame_callback, "set_meta", None)
    if fn is not None:
        fn(int(frame_number), float(t))


def _track_parallel(video_path, setup, bgm, exclusion, warp, start_frame, end_frame, fps, workers, prog,
                    frame_callback=None, live_due=None, live=None, bodyfrac=None):
    """Track [start_frame, end_frame] in `workers` pieces at the same time (own decoder + tracker each).
    Every piece starts a few seconds early (warm-up) so its tracker is locked on before its first frame:
    the positions equal the frame-by-frame run.  Returns (frames, times, xs, ys, areas, status,
    last_tracker), or None when the video cannot be positioned exactly (the caller then tracks frame by
    frame as usual)."""
    import threading
    warm = max(90, int(round(3.0 * fps)))
    n = end_frame - start_frame + 1
    bounds = [start_frame + k * n // workers for k in range(workers)] + [end_frame + 1]
    results = [None] * workers
    trackers = [None] * workers
    done = [0] * workers
    stop = threading.Event()
    errors = []
    live_k = [0]                     # the live picture follows the first piece that is still running
    finished = [False] * workers

    def work(k):
        a, b = bounds[k], bounds[k + 1]
        s0 = start_frame if k == 0 else max(start_frame, a - warm)
        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
            pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or s0))
            while pos < s0:                                  # some codecs land a bit early
                if not cap.grab():
                    break
                pos += 1
            if pos != s0:
                errors.append("seek")
                stop.set()
                return
            tr = new_tracker(setup, bgm, exclusion)
            prep_fn = getattr(tr, "prepare_frame", None)
            hook = setup.get("_frame_hook")
            cols = ([], [], [], [], [], [], [], [])
            fn = s0
            while fn < b and not stop.is_set():
                ok, raw = cap.read()
                if not ok:
                    break
                fr = warp(raw)
                det = tr.update(fr, prep_fn(fr)) if prep_fn is not None else tr.update(fr)
                if hook is not None:                          # one-pass freezing: motion from the same frames
                    hook(k, a, b, fn, raw, det)
                if fn >= a:
                    apply_backfill(det, cols[2], cols[3], cols[4], cols[5], ends=cols[7], fracs=cols[6])
                    bf = bodyfrac(tr, det) if bodyfrac is not None else ()
                    for col, v in zip(cols, (fn, fn / fps, det["x"], det["y"], det["area"], det["status"], bf,
                                             det.get("ends"))):
                        col.append(v)
                    done[k] += 1
                    if (k == live_k[0] and frame_callback is not None
                            and (live_due is not None or (fn - start_frame) % 5 == 0)
                            and (live_due is None or live_due())):
                        _live_meta(frame_callback, fn, fn / fps)
                        frame_callback(live.draw(fr, det, tr, fn / fps, fn))
                fn += 1
            results[k] = cols
            trackers[k] = tr
        except BaseException as exc:                         # incl. "stopped by user" from the live view
            errors.append(exc)
            stop.set()
        finally:
            cap.release()
            finished[k] = True     # the live picture stays on the first piece (plays in order, never jumps)

    threads = [threading.Thread(target=work, args=(k,), name=f"bt-track{k}", daemon=True) for k in range(workers)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
            prog(0.08 + 0.84 * sum(done) / n)
    except BaseException:
        stop.set()
        for t in threads:
            t.join(timeout=10)
        raise
    real = [e for e in errors if isinstance(e, BaseException)]
    if real:
        raise real[0]
    if errors:                                               # could not position a piece exactly
        return None
    merged = ([], [], [], [], [], [], [], [])
    last = None
    for k in range(workers):
        cols = results[k]
        if cols is None:
            break
        for m, c in zip(merged, cols):
            m.extend(c)
        last = trackers[k]
        if len(cols[0]) < bounds[k + 1] - bounds[k]:         # the video ended early here: stop, as frame by frame
            break
    if not merged[0]:
        return None
    return merged + (last,)



def make_tracking_engine(setup, warp_w, warp_h, read_warped, start_frame, end_frame, exclusion,
                         progress=None, video_path=None, warp_fn=None):
    """Build the background model + tracker described by `setup`.

    Returns (tracker, background_model).  Shared by the full pipeline and
    the UI's 'Test detection' preview so both behave identically."""
    from tracking.robust_tracker import build_background, detection_scale
    algorithm = str(setup.get("tracking_algorithm", "robust")).lower()
    color_mode = str(setup.get("color_mode", "gray")).lower()
    polarity = str(setup.get("animal_polarity", "auto")).lower()
    if algorithm == "classic":
        polarity = "any"           # the classic engine uses the absolute difference
    scale = detection_scale(warp_w, warp_h)
    n_bg = max(10, int(setup.get("background_samples", 100)))
    end_frame = max(start_frame, end_frame)
    b0, b1 = start_frame, end_frame
    wide = setup.get("background_window_frames")     # freezing: a wider stretch, so an animal that never moves
    if wide:                                          # in the time window is not part of the background
        b0, b1 = min(start_frame, int(wide[0])), max(end_frame, int(wide[1]))
    positions = np.linspace(b0, b1, min(n_bg, max(1, b1 - b0 + 1))).astype(int)
    valid = None if exclusion is None else ~exclusion
    prepared = None
    if video_path is not None and warp_fn is not None and len(positions) >= 8:
        prepared = read_samples_parallel(video_path, positions, warp_fn, color_mode, scale, progress=progress)
    bgm = build_background(read_warped, positions, color_mode, polarity, valid_full=valid, scale=scale,
                           progress=progress, raw_samples=prepared, low_light=setup.get("low_light", "off"),
                           zone_polys=low_light_zone_polys(setup), rest_pct=setup.get("background_rest_pct"))
    tracker = new_tracker(setup, bgm, exclusion)
    return tracker, bgm


def new_tracker(setup, bgm, exclusion):
    """A fresh tracker (no position lock) for an existing background model."""
    from tracking.robust_tracker import RobustTracker, ClassicTracker
    if str(setup.get("tracking_algorithm", "robust")).lower() == "classic":
        return ClassicTracker(bgm, loc_thresh=float(setup.get("loc_thresh", 99.5)),
                              use_window=bool(setup.get("use_window", True)),
                              window_size=int(setup.get("window_size", 200)),
                              window_weight=float(setup.get("window_weight", 0.9)),
                              exclusion_full=exclusion)
    fps = float(setup.get("_fps", 30.0) or 30.0)
    core = covered = None
    if setup.get("warp_w") and setup.get("warp_h"):
        core, covered = detection_guides(setup, setup["warp_w"], setup["warp_h"])
    return RobustTracker(bgm, min_contrast=float(setup.get("min_contrast", 12.0)),
                         min_area=float(setup.get("min_area", 15)),
                         max_area=float(setup.get("max_area", 0) or 0),
                         max_jump=float(setup.get("max_jump", 100)),
                         exclusion_full=exclusion,
                         reacquire_frames=max(3, int(round(float(setup.get("reacquire_s", 0.33)) * fps))),
                         core_full=core, covered_full=covered, fps=fps,
                         body_length=float(setup.get("body_length_px") or 0) or None,
                         adapt_background=bool(setup.get("adapt_background", True)),
                         want_ends=bool(setup.get("track_nose")))


def measure_animal_contrast(setup, bgm, exclusion, frames, progress=None):
    """How different the animal is from the empty floor, measured on `frames` (warped BGR frames spread over
    the time window) -> what 'Minimum contrast' should be.

    Every frame is searched with a very low threshold; spots that sit in the same place in many frames
    (labels, shadows of the apparatus) are not the animal.  Returns a dict: peak / body contrast of the
    animal per frame, the video noise, the recommended minimum contrast and how often the animal would
    be found with it."""
    from tracking.robust_tracker import RobustTracker
    probe = dict(setup)
    probe["min_contrast"] = 4.0
    tr = new_tracker(probe, bgm, exclusion)
    if not isinstance(tr, RobustTracker):
        raise RuntimeError("The contrast check needs the Robust engine.")
    per = []
    n = len(frames)
    for i, fr in enumerate(frames):
        if fr is None:
            continue
        d = tr.strength(fr)
        out = []
        for c in tr.candidates(d):
            x, y, bw, bh = c["bbox"]
            vals = d[y:y + bh, x:x + bw][c["full"]]
            out.append({"x": c["cx"], "y": c["cy"], "area": c["area"], "peak": c["peak"], "mass": c["mass"],
                        "body": float(np.median(vals)) if vals.size else 0.0})
        per.append(out)
        if progress is not None:
            progress((i + 1) / max(1, n))
    # spots seen at the same place in many frames are fixed (not the animal)
    rad = max(3.0, 0.5 * tr.body_l)
    n_f = len(per)

    def fixed(c):
        hits = sum(1 for o in per if any(math.hypot(c["x"] - c2["x"], c["y"] - c2["y"]) <= rad for c2 in o))
        return n_f >= 8 and hits >= max(4, 0.4 * n_f)
    peaks, bodies, areas = [], [], []
    for o in per:
        moving = [c for c in o if not fixed(c)]
        if not moving:
            continue
        best = max(moving, key=lambda c: c["mass"] * (0.15 + tr._area_like(c["area"])))
        peaks.append(best["peak"])
        bodies.append(best["body"])
        areas.append(best["area"])
    sigma = float(bgm.sigma)
    floor_c = max(6.0, math.ceil(5.0 * sigma))
    res = {"frames": n_f, "found": len(peaks), "sigma": round(sigma, 2), "noise_floor": int(floor_c),
           "area_median_full": float(np.median(areas)) / (tr.s * tr.s) if areas else 0.0,
           "gamma": float(getattr(bgm, "gamma", 1.0) or 1.0), "dark_level": getattr(bgm, "dark_level", None)}
    if len(peaks) < max(3, 0.2 * max(1, n_f)):
        res.update({"recommended": None, "peak_median": None, "peak_low": None, "body_median": None})
        return res
    peaks, bodies = np.array(peaks), np.array(bodies)
    p_low = float(np.percentile(peaks, 10))
    b_low = float(np.percentile(bodies, 10))
    rec = min(0.45 * p_low, 0.8 * b_low)
    rec = int(round(min(40.0, max(floor_c, rec))))
    res.update({"recommended": rec, "peak_median": round(float(np.median(peaks)), 1), "peak_low": round(p_low, 1),
                "body_median": round(float(np.median(bodies)), 1), "body_low": round(b_low, 1),
                "found_with_rec": int((peaks >= max(rec, 7.0 * sigma)).sum()),
                "found_with_current": int((peaks >= max(float(setup.get("min_contrast", 12)), 7.0 * sigma)).sum()),
                "weak": bool(p_low < 2.0 * floor_c)})
    return res


def track_video(video_path, setup, progress_callback=None, frame_callback=None):
    """Track one animal in one video: background, detection, positions (no results yet -- see analyse_track).

    setup keys (all optional unless noted): matrix/warp_w/warp_h (required),
    roi_points/roi_names, mask_points, auto_mask_outside_zones,
    tracking_algorithm ('robust' | 'classic'), animal_polarity ('auto' |
    'dark' | 'light' | 'any'), color_mode, min_contrast, min_area,
    max_area (0 = automatic), max_jump, background_samples, max_gap_s,
    smooth_window, min_move_px, scale_factor/scale_unit, start_time,
    end_time, test_type, bin_size_s, zone_groups, stop_condition/... .

    frame_callback(bgr) receives an annotated frame every 5th frame for a
    live view; progress_callback(fraction) reports progress.  show_display
    and confirm_callback are accepted for backwards compatibility only (no
    OpenCV windows are ever opened -- they crash next to a Qt GUI)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        fps = 30.0
    if setup.get("fps_override"):
        fps = float(setup["fps_override"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0.0

    start_time = max(0.0, min(float(setup.get("start_time", 0.0) or 0.0), duration))
    end_req = setup.get("end_time")
    end_time = duration if end_req is None else max(start_time, min(float(end_req), duration))
    start_frame = int(round(start_time * fps))
    end_frame = min(max(total_frames - 1, 0), int(round(end_time * fps)))

    matrix = setup.get("matrix")
    if matrix is None:
        matrix = np.eye(3, dtype=np.float32)
    warp_w = int(setup.get("warp_w") or frame_w)
    warp_h = int(setup.get("warp_h") or frame_h)
    need_warp = not _is_identity_warp(matrix, frame_w, frame_h, warp_w, warp_h)

    def warp(frame):
        return cv2.warpPerspective(frame, matrix, (warp_w, warp_h)) if need_warp else frame

    def read_warped(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        return warp(f) if ok else None

    roi_points = {k: [tuple(map(float, p)) for p in v] for k, v in (setup.get("roi_points") or {}).items()
                  if len(v) >= 3}
    roi_names = [n for n in (setup.get("roi_names") or list(roi_points)) if n in roi_points]
    for n in roi_points:
        if n not in roi_names:
            roi_names.append(n)
    roi_masks = build_roi_masks((warp_h, warp_w), roi_points)

    mask_polygons = setup.get("mask_points", []) or []
    auto_mask_outside_zones = bool(setup.get("auto_mask_outside_zones")) and bool(roi_points)
    covered_names = covered_zone_names(setup)
    exclusion = None
    if mask_polygons or auto_mask_outside_zones or covered_names:
        exclusion = build_detection_exclusion(warp_w, warp_h, mask_polygons, roi_points, auto_mask_outside_zones,
                                              margin=setup.get("auto_mask_margin"), covered=covered_names)
        if exclusion.all():
            exclusion = None

    stop_condition = setup.get("stop_condition")
    stop_value = setup.get("stop_value")
    stop_zone_name = setup.get("stop_zone_name")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if setup.get("output_dir_override"):
        output_dir = os.path.join(setup["output_dir_override"], video_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = compute_output_dir(video_path)

    def prog(f):
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, f)))

    # ---- background + tracker -------------------------------------------------
    setup = dict(setup)
    setup["_fps"] = fps
    setup["warp_w"], setup["warp_h"] = warp_w, warp_h
    tracker, bgm = make_tracking_engine(setup, warp_w, warp_h, read_warped, start_frame, end_frame, exclusion,
                                        progress=lambda f: prog(0.08 * f), video_path=video_path, warp_fn=warp)
    background_bgr = bgm.full_res_bgr(warp_w, warp_h)
    cv2.imwrite(os.path.join(output_dir, "background.png"), background_bgr)
    algorithm = "classic" if tracker.__class__.__name__ == "ClassicTracker" else "robust"

    # body-in-zone fractions (four-paw entry rule); not for object zones (NOR) or when the rule is off
    entry_rule = str(setup.get("entry_rule", "paws")).lower()
    frac_names = []
    if entry_rule == "paws" and setup.get("test_type") != "nor" and roi_names:
        frac_names = list(roi_names)
    bodyfrac = BodyZoneFractions(roi_masks, frac_names) if frac_names else None

    # ---- tracking: long videos in several pieces at once on multi-core computers (identical positions,
    #      2-4x faster), otherwise frame by frame.  Stop conditions always run frame by frame.
    n_expected = max(1, end_frame - start_frame + 1)
    workers = int(setup.get("workers") or worker_count(n_expected))
    if covered_names:
        workers = 1           # the animal hides for long stretches: each piece must know where it was last seen
    par = None
    if workers > 1 and not stop_condition:
        live_p = (_LiveDrawer(warp_w, warp_h, roi_points, mask_polygons, exclusion,
                              show_mask=setup.get("live_show_mask", True))
                  if frame_callback is not None else None)
        if setup.get("_frame_hook") is not None:
            setup["_frame_hook"].begin(workers)
        par = _track_parallel(video_path, setup, bgm, exclusion, warp, start_frame, end_frame, fps, workers, prog,
                              frame_callback=frame_callback, live_due=getattr(frame_callback, "due", None),
                              live=live_p, bodyfrac=bodyfrac)
    if par is not None:
        frames, times, xs, ys, areas, status, fracs, ends, tracker = par
        stop_reason = None
        cap.release()
    if par is None:
        # ---- tracking loop (sequential reads -- no per-frame seeking) ---------------
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        pos_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or start_frame)
        while pos_now < start_frame:              # some codecs land a bit early
            if not cap.grab():
                break
            pos_now += 1

        n_expected = max(1, end_frame - start_frame + 1)
        frames, times, xs, ys, areas, status, fracs, ends = [], [], [], [], [], [], [], []
        stop_reason = None
        stop_immobile_seconds = 0.0
        stop_zone_entry_count = 0
        stop_zone_was_in = False
        stop_cum_px = 0.0
        last_xy = None
        STOP_IMMOBILITY_PIXEL_THRESHOLD = 2.0

        # Reading + straightening + preparing the next frames runs in a second thread while this one is tracked
        # (OpenCV releases the GIL): the same per-frame work as before, so the results are identical -- only faster.
        import queue
        import threading
        prep_fn = getattr(tracker, "prepare_frame", None)
        q = queue.Queue(maxsize=24)
        reader_stop = threading.Event()

        def _reader():
            fn = start_frame
            try:
                while fn <= end_frame and not reader_stop.is_set():
                    ok_r, raw_r = cap.read()
                    if not ok_r:
                        break
                    fr = warp(raw_r)
                    item = (fn, fr, prep_fn(fr) if prep_fn is not None else None)
                    while not reader_stop.is_set():
                        try:
                            q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    fn += 1
            except Exception as exc:                        # surfaced in the tracking loop below
                q.put(("error", exc, None))
            finally:
                q.put(None)

        reader = threading.Thread(target=_reader, name="bt-reader", daemon=True)
        reader.start()
        live_due = getattr(frame_callback, "due", None)       # skip drawing when nobody is looking
        live = (_LiveDrawer(warp_w, warp_h, roi_points, mask_polygons, exclusion,
                            show_mask=setup.get("live_show_mask", True)) if frame_callback is not None else None)

        frame_number = start_frame
        hook = setup.get("_frame_hook")
        if hook is not None:
            hook.begin(1)
        try:
            while frame_number <= end_frame:
                item = q.get()
                if item is None:
                    break
                if item[0] == "error":
                    raise item[1]
                frame_number, frame, prepared = item
                t = frame_number / fps
                det = tracker.update(frame, prepared) if prepared is not None else tracker.update(frame)
                if hook is not None:                          # one-pass freezing: motion from the same frames
                    hook(0, start_frame, end_frame + 1, frame_number, frame, det)
                apply_backfill(det, xs, ys, areas, status, ends=ends, fracs=fracs)
                frames.append(frame_number)
                times.append(t)
                xs.append(det["x"])
                ys.append(det["y"])
                areas.append(det["area"])
                status.append(det["status"])
                fracs.append(bodyfrac(tracker, det) if bodyfrac is not None else ())
                ends.append(det.get("ends"))

                if stop_condition and stop_value is not None:
                    moved = 0.0
                    if det["status"] == "Tracked":
                        if last_xy is not None:
                            moved = math.hypot(det["x"] - last_xy[0], det["y"] - last_xy[1])
                        last_xy = (det["x"], det["y"])
                    if stop_condition == "immobility":
                        if det["status"] == "Tracked" and moved < STOP_IMMOBILITY_PIXEL_THRESHOLD:
                            stop_immobile_seconds += 1.0 / fps
                        else:
                            stop_immobile_seconds = 0.0
                        if stop_immobile_seconds >= stop_value:
                            stop_reason = f"Immobile for {stop_value:g}s"
                    elif stop_condition == "zone_entries" and stop_zone_name in roi_masks:
                        now_in = det["status"] == "Tracked" and roi_membership(
                            {stop_zone_name: roi_masks[stop_zone_name]}, det["x"], det["y"])[stop_zone_name]
                        if now_in and not stop_zone_was_in:
                            stop_zone_entry_count += 1
                        stop_zone_was_in = now_in
                        if stop_zone_entry_count >= stop_value:
                            stop_reason = f"{int(stop_value)} entries into '{stop_zone_name}'"
                    elif stop_condition == "distance":
                        stop_cum_px += moved
                        if stop_cum_px >= stop_value:
                            stop_reason = f"Reached {stop_value:g}px total distance"

                if (frame_callback is not None and (live_due is not None or (frame_number - start_frame) % 5 == 0)
                        and (live_due is None or live_due())):
                    _live_meta(frame_callback, frame_number, t)
                    frame_callback(live.draw(frame, det, tracker, t, frame_number))
                if (frame_number - start_frame) % 5 == 0:
                    prog(0.08 + 0.84 * (frame_number - start_frame + 1) / n_expected)
                frame_number += 1
                if stop_reason is not None:
                    break
        finally:
            reader_stop.set()
            while reader.is_alive():                          # unblock + let the reader finish its frame
                try:
                    q.get(timeout=0.1)
                except queue.Empty:
                    pass
            reader.join(timeout=5)
            cap.release()

    if not frames:
        raise RuntimeError("No frames could be read in the selected time window.")

    # ---- a few preview pictures: the animal exactly as this run found it on those frames
    try:
        preview_dir = os.path.join(output_dir, "preview")
        os.makedirs(preview_dir, exist_ok=True)
        for stale in glob.glob(os.path.join(preview_dir, "preview_*.png")):
            os.remove(stale)
        n_prev = max(0, int(setup.get("preview_samples", 3) if setup.get("preview_samples") is not None else 3))
        if n_prev:
            pcap = cv2.VideoCapture(video_path)
            try:
                for i, k in enumerate(np.linspace(0, len(frames) - 1, n_prev + 2).astype(int)[1:-1], start=1):
                    pf = int(frames[k])
                    pcap.set(cv2.CAP_PROP_POS_FRAMES, pf)
                    ok, raw_f = pcap.read()
                    if not ok:
                        continue
                    disp = detection_still(setup, bgm, exclusion, warp(raw_f), xs[k], ys[k], status[k],
                                           pf / fps, pf, roi_points, mask_polygons,
                                           lost_label="NOT SEEN (hidden?)" if covered_names else "NOT SEEN")
                    cv2.imwrite(os.path.join(preview_dir, f"preview_{i:02d}_frame{pf}.png"), disp)
            finally:
                pcap.release()
    except Exception as exc:  # previews are a convenience only
        print(f"Preview images skipped: {exc}")

    # ---- everything the analysis (and a later re-analysis after corrections) needs
    body = getattr(tracker, "body_l", None)
    body_full = (body / tracker.s) if (body and getattr(tracker, "s", None)) else None
    a_full = (bgm.animal_area / (bgm.scale ** 2)) if getattr(bgm, "animal_area", None) else None
    if not a_full:
        good = np.asarray(areas, float)
        good = good[np.isfinite(good) & (good > 0)]
        a_full = float(np.median(good)) if good.size else None
    return {
        "video_path": video_path, "output_dir": output_dir, "video_name": video_name,
        "fps": fps, "frame_w": frame_w, "frame_h": frame_h, "total_frames": total_frames,
        "start_time": start_time, "end_time": end_time, "start_frame": start_frame, "end_frame": end_frame,
        "requested_end_time": None if end_req is None else float(end_req),
        "warp_w": warp_w, "warp_h": warp_h, "need_warp": bool(need_warp),
        "frames": list(map(int, frames)), "times": list(map(float, times)),
        "xs": list(map(float, xs)), "ys": list(map(float, ys)), "areas": list(map(float, areas)),
        "status": list(status), "frac_names": list(frac_names),
        "fracs": [list(f) for f in fracs] if frac_names else [],
        "ends": ([None if e is None else [round(float(v), 2) for v in e] for e in ends]
                 if any(e is not None for e in ends) else []),
        "covered_zones": list(covered_names),
        "stop_reason": stop_reason, "algorithm": algorithm, "polarity": bgm.polarity,
        "background_info": bgm.describe(), "background_bgr": background_bgr,
        "animal_area_full": a_full, "body_len_full": body_full,
        "low_light_gamma": float(getattr(bgm, "gamma", 1.0) or 1.0),
        "polarity_warning": getattr(bgm, "polarity_warning", None),
    }


def process_pose_file(video_path, setup, progress_callback=None):
    """The usual results of one video from the positions of a DeepLabCut / SLEAP CSV (setup['pose_file'])."""
    from tracking.pose_import import read_pose_csv, pose_to_raw
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if progress_callback is not None:
        progress_callback(0.1)
    pose = read_pose_csv(setup["pose_file"])
    raw = pose_to_raw(pose, video_path, setup, fps, fw, fh, total, p_cut=float(setup.get("pose_p_cut", 0.6)))
    summary = analyse_track(raw, setup, progress_callback=progress_callback, write_files=True)
    summary["Detection_engine"] = raw["algorithm"]
    summary["Background_info"] = raw["background_info"]
    summary["Pose_file"] = os.path.basename(str(setup["pose_file"]))
    return summary


def process_single_video(video_path, setup, show_display=False, progress_callback=None, confirm_callback=None,
                         frame_callback=None, save_project=True):
    """Track one animal in one video and write every result file (+ the protected BT analysis file).

    show_display and confirm_callback are accepted for backwards compatibility only."""
    raw = track_video(video_path, setup, progress_callback=progress_callback, frame_callback=frame_callback)
    embed = save_project and int(setup.get("embed_video_px", 640) or 0) > 0
    top = 0.95 if embed else 1.0

    def p_an(f):
        if progress_callback is not None:
            progress_callback(min(f, top))
    summary = analyse_track(raw, setup, progress_callback=p_an)
    if save_project:
        try:
            from tracking.project_file import save_new_analysis

            def p_copy(f):                             # the video copy for the .bta (last few %)
                if progress_callback is not None:
                    progress_callback(top + (1.0 - top) * f)
            summary["Project_file"] = save_new_analysis(raw, setup, summary, progress=p_copy)
        except Exception as exc:                      # the result files are already written
            print(f"BT analysis file not saved: {exc}")
    if progress_callback is not None:
        progress_callback(1.0)
    return summary


def nose_exploration(df, raw, xs, ys, roi_names, roi_masks, fps, summary, test_type=None):
    """Nose (head) and tail-base points from the body ends of every frame (tracking.body_points); for the novel
    object test the object exploration is then counted with the NOSE in the object zone (the usual definition)
    instead of the body centre.  Adds Nose_X / Nose_Y / Tail_base_X / Tail_base_Y to df and to the summary:
    Nose_found_pct, Exploration_point and the exploration measures."""
    from tracking.body_points import resolve_head_tail
    from analysis.test_metrics import _has
    tracked = df["Tracking_Status"].to_numpy() == "Tracked"
    hx, hy, tx, ty = resolve_head_tail(raw, xs, ys, valid=tracked)
    if len(hx) != len(df):
        return
    df["Nose_X"], df["Nose_Y"] = np.round(hx, 2), np.round(hy, 2)
    df["Tail_base_X"], df["Tail_base_Y"] = np.round(tx, 2), np.round(ty, 2)
    ok = np.isfinite(hx) & np.isfinite(hy)
    summary["Nose_found_pct"] = float(100.0 * ok[tracked].mean()) if tracked.any() else 0.0
    if test_type != "nor":
        return
    objs = [z for z in roi_names if _has(z, "object", "novel", "familiar", "obj")]
    if not objs:
        return
    h, w = next(iter(roi_masks.values())).shape
    xi = np.clip(np.round(np.nan_to_num(hx)).astype(int), 0, w - 1)
    yi = np.clip(np.round(np.nan_to_num(hy)).astype(int), 0, h - 1)
    times = {}
    for z in objs:
        m = roi_masks.get(z)
        if m is None:
            continue
        inside = ok & m[yi, xi]
        df[f"Nose_in_{safe_col(z)}"] = inside
        times[z] = float(inside.sum()) / fps
        summary[f"{safe_col(z)}_exploration_s"] = round(times[z], 3)
    summary["Total_exploration_s"] = round(sum(times.values()), 3)
    novel = [z for z in times if _has(z, "novel", "new")]
    fam = [z for z in times if _has(z, "familiar", "old")]
    if novel and fam:
        n_t, f_t = sum(times[z] for z in novel), sum(times[z] for z in fam)
        tot = n_t + f_t
        summary["Discrimination_index"] = round((n_t - f_t) / tot, 4) if tot > 0 else None
        summary["Discrimination_ratio"] = round(n_t / tot, 4) if tot > 0 else None
        summary["Recognition_index_pct"] = round(100 * n_t / tot, 2) if tot > 0 else None
    elif len(times) >= 2:
        a, b = list(times)[:2]
        ta, tb = times[a], times[b]
        summary[f"Discrimination_index_{safe_col(b)}_vs_{safe_col(a)}"] = round((tb - ta) / (ta + tb), 4) if (ta + tb) > 0 else None
        summary[f"Discrimination_ratio_{safe_col(b)}_vs_{safe_col(a)}"] = round(tb / (ta + tb), 4) if (ta + tb) > 0 else None
    summary["Exploration_point"] = "Nose (head end of the body)"


def analyse_track(raw, setup, corrections=None, progress_callback=None, write_files=True):
    """Results from tracked positions: gap filling, smoothing, distance, zones, test measures, time bins and
    every result file.  `corrections` = manual fixes from the frame review:
    [{"frame": n, "kind": "set", "x": .., "y": ..} | {"frame": n, "kind": "absent"}]."""
    from tracking.robust_tracker import gap_fill
    fps = float(raw["fps"])
    video_path = raw["video_path"]
    output_dir = raw["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    video_name = raw.get("video_name") or os.path.splitext(os.path.basename(video_path))[0]
    start_time = float(raw["start_time"])
    warp_w, warp_h = int(raw["warp_w"]), int(raw["warp_h"])
    need_warp = bool(raw.get("need_warp"))
    matrix = setup.get("matrix")
    matrix = np.eye(3, dtype=np.float32) if matrix is None else np.asarray(matrix, np.float32)

    def warp(frame):
        return cv2.warpPerspective(frame, matrix, (warp_w, warp_h)) if need_warp else frame

    def prog(f):
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, f)))

    roi_points = {k: [tuple(map(float, p)) for p in v] for k, v in (setup.get("roi_points") or {}).items()
                  if len(v) >= 3}
    roi_names = [n for n in (setup.get("roi_names") or list(roi_points)) if n in roi_points]
    for n in roi_points:
        if n not in roi_names:
            roi_names.append(n)
    roi_masks = build_roi_masks((warp_h, warp_w), roi_points)
    zone_groups = setup.get("zone_groups", {}) or {}
    smooth_window = setup.get("smooth_window")
    custom_variable_defs = setup.get("custom_variables") or []
    object_points = setup.get("object_points", {}) or {}
    object_names = [n for n in (setup.get("object_names", []) or []) if n in object_points]
    interaction_margin_px = setup.get("interaction_margin_px", 0) or 0
    object_masks = build_roi_masks((warp_h, warp_w), object_points) if object_points else {}
    object_masks = dilate_masks(object_masks, interaction_margin_px)
    auto_mask_outside_zones = bool(setup.get("auto_mask_outside_zones")) and bool(roi_points)

    frames = list(raw["frames"])
    times = list(raw["times"])
    xs = np.array(raw["xs"], dtype=float)
    ys = np.array(raw["ys"], dtype=float)
    areas = list(raw["areas"])
    status = list(raw["status"])
    frac_names = list(raw.get("frac_names") or [])
    fracs = np.array(raw.get("fracs") or [], dtype=float).reshape(len(frames), len(frac_names)) \
        if frac_names else np.zeros((len(frames), 0))
    stop_reason = raw.get("stop_reason")
    algorithm = raw.get("algorithm", "robust")
    a_full = raw.get("animal_area_full")
    body_r = max(4.0, 0.6 * math.sqrt(a_full)) if a_full else 10.0

    # ---- manual corrections (frame review) -----------------------------------------
    corrected = np.zeros(len(frames), bool)
    absent = np.zeros(len(frames), bool)
    if corrections:
        index = {f: i for i, f in enumerate(frames)}
        for c in corrections:
            i = index.get(int(c.get("frame", -1)))
            if i is None:
                continue
            if c.get("kind") == "absent":
                xs[i], ys[i], status[i] = np.nan, np.nan, "Lost"
                absent[i] = True
                if frac_names:
                    fracs[i, :] = np.nan
            else:
                xs[i], ys[i], status[i] = float(c["x"]), float(c["y"]), "Tracked"
                corrected[i] = True
                if frac_names:
                    fracs[i, :] = disc_zone_fractions(roi_masks, frac_names, xs[i], ys[i], body_r)

    # ---- post-processing: gap fill, smoothing, distance, zones -------------------
    tracked_raw = np.array([s == "Tracked" for s in status])
    max_gap_frames = int(round(float(setup.get("max_gap_s", 0.5) or 0.0) * fps))
    filled = gap_fill(xs, ys, tracked_raw, max_gap_frames) & ~absent
    if absent.any():                                  # 'animal not visible' frames are never filled in
        xs[absent], ys[absent] = np.nan, np.nan
    tracked = tracked_raw | filled
    # covered zones (lid of a light/dark box...): not seen after it was last seen next to one = inside it
    covered = [n for n in (raw.get("covered_zones") if raw.get("covered_zones") is not None
                           else covered_zone_names(setup)) if n in roi_masks]
    body_len_full = float(raw.get("body_len_full") or 0.0) or (2.4 * math.sqrt(a_full / 2.4) if a_full else 30.0)
    hidden, hidden_zone = hidden_in_covered(xs, ys, tracked, {n: roi_masks[n] for n in covered}, body_len_full)
    # not found for longer than the gap filling: the animal stays where it was last seen until it is found
    # again (as ANY-maze does) -- so the zone times add up to the whole time window you set
    held = np.zeros(len(xs), bool)
    if setup.get("hold_lost", True):
        held = hold_last_position(xs, ys, tracked, hidden | absent)
    tracked = tracked | held

    df = pd.DataFrame({
        "Frame": frames,
        "Time_seconds": times,
        "Mouse_X": xs,
        "Mouse_Y": ys,
        "Mouse_Present": tracked | hidden,
        "ROI": "None",
        "Detected_area": areas,
        "Distance_pixels": 0.0,
        "Velocity_pixels_s": 0.0,
        "Tracking_Status": np.where(tracked, "Tracked", np.where(hidden, "Hidden", "Lost")),
        "Interpolated": filled,
        "Held": held,
        "Corrected": corrected,
    })

    traj_x_col, traj_y_col = "Mouse_X", "Mouse_Y"
    smoothing_applied = bool(smooth_window and int(smooth_window) >= 2)
    if smoothing_applied:
        df = smooth_trajectory(df, int(smooth_window))
        traj_x_col, traj_y_col = "Mouse_X_smooth", "Mouse_Y_smooth"
    px = df[traj_x_col].to_numpy(float)
    py = df[traj_y_col].to_numpy(float)

    if setup.get("min_move_px") is not None:
        min_move_px = float(setup["min_move_px"])
    else:
        body_full = raw.get("body_len_full") or 40.0
        min_move_px = max(1.0, 0.04 * body_full)
    dist = path_distance(px, py, tracked, min_move_px)
    df["Distance_pixels"] = dist
    df["Velocity_pixels_s"] = dist * fps

    # the body centre is the tracking point (head / tail points are not tracked)
    from tracking.body_points import POINT_NAMES
    point = "centre"
    flags = zone_flags(roi_masks, px, py, tracked)
    for name in covered:                              # hidden in a covered zone
        if name in flags:
            flags[name] = flags[name] | (hidden & (hidden_zone == name))
    for name in roi_names:
        df[f"In_{safe_col(name)}"] = flags.get(name, np.zeros(len(df), bool))
    if roi_names:
        labels = []
        for i in range(len(df)):
            if not (tracked[i] or hidden[i]):
                labels.append("None")
                continue
            act = [n for n in roi_names if flags[n][i]]
            labels.append("_".join(act) if act else "None")
        df["ROI"] = labels
    # body-in-zone fractions (four-paw entry rule) for zones big enough to hold the whole animal
    entry_pct = float(setup.get("entry_body_pct") or ENTRY_BODY_PCT)
    paw_zones = []
    if frac_names:
        for j, name in enumerate(frac_names):
            if name not in roi_masks:
                continue
            if a_full and float(roi_masks[name].sum()) < 1.25 * a_full:
                continue                                 # too small for the whole animal: centre-point rule
            col = fracs[:, j].copy()
            col[~tracked_raw] = np.nan                   # filled-in frames have no body
            if hidden.any():                             # hidden in a covered zone = the whole body is in it
                col[hidden] = np.where(hidden_zone[hidden] == name, 1.0, 0.0)
            df[f"Body_{safe_col(name)}"] = col
            paw_zones.append(name)
    if object_names:
        oflags = zone_flags(object_masks, px, py, tracked)
        for name in object_names:
            df[f"Near_{safe_col(name)}"] = oflags.get(name, np.zeros(len(df), bool))

    # ---- summary ------------------------------------------------------------
    from analysis.test_metrics import paw_state
    paw = paw_state(df, paw_zones, frac=entry_pct / 100.0) if paw_zones else None
    if paw is not None:
        df["Zone_4paw"] = paw[0]
        transitions = calculate_transitions(df, confirmation_frames=1, roi_col="Zone_4paw")
    else:
        transitions = calculate_transitions(df)
    tracked_df = df[df["Tracking_Status"].isin(PRESENT)]          # seen, or hidden in a covered zone
    tracked_time = len(tracked_df) / fps
    visible_time = float((df["Tracking_Status"] == "Tracked").sum()) / fps
    n_hidden = int(hidden.sum())
    quality = 100.0 * float(tracked_raw.sum()) / (len(df) - n_hidden) if len(df) > n_hidden else 0.0

    summary = {
        "Video": video_path,
        "Output_folder": output_dir,
        "Start_s": start_time,
        "End_s": float(times[-1]) + 1.0 / fps,
        "FPS": fps,
        "Frames_processed": len(df),
        "Tracked_frames": int(tracked.sum()),
        "Detected_frames": int(tracked_raw.sum()),
        "Interpolated_frames": int(filled.sum()),
        "Lost_frames": int(len(df) - tracked.sum() - hidden.sum()),
        "Held_frames": int(held.sum()),
        "Held_time_s": round(float(held.sum()) / fps, 3),
        "Tracking_quality_percent": quality,
        "Total_distance_pixels": float(df["Distance_pixels"].sum()),
        "Stop_Reason": stop_reason if stop_reason is not None else "Reached End (s)",
        "Detection_engine": "Robust" if algorithm == "robust" else "Classic",
        "Animal_polarity": raw.get("polarity"),
        "Background_info": raw.get("background_info"),
        "Corrected_frames": int(corrected.sum() + absent.sum()),
        "Entry_rule": (f"All four paws ({entry_pct:.0f} % of the body inside the zone)" if paw_zones
                       else "Body centre point"),
        "Time_in_zone_rule": "Body centre point",
        "Min_move_px": round(min_move_px, 2),
        "Auto_mask_outside_zones": bool(auto_mask_outside_zones),
        "Tracking_point": POINT_NAMES[point],
        "Low_light_boost": (f"gamma {float(raw.get('low_light_gamma') or 1.0):.1f} ({setup.get('low_light', 'off')})"
                            if float(raw.get("low_light_gamma") or 1.0) > 1.001 else "Off"),
    }
    if raw.get("polarity_warning"):
        summary["Polarity_warning"] = polarity_warning_text(raw.get("polarity"), raw["polarity_warning"])
    # was the whole time window you set analysed?
    req_end = raw.get("requested_end_time")
    want_end = min(float(req_end), 1e8) if req_end is not None else None
    if want_end is not None and want_end < 1e7 and summary["End_s"] < want_end - 1.5 / fps:
        summary["Window_note"] = (f"The video ends at {summary['End_s']:.1f} s, before the end time you set "
                                  f"({want_end:.1f} s): {want_end - summary['End_s']:.1f} s could not be analysed.")
    summary["Analysed_time_s"] = round(len(df) / fps, 3)
    if covered:
        summary["Covered_zones"] = ", ".join(covered)
        summary["Hidden_frames"] = n_hidden
        summary["Hidden_time_s"] = round(n_hidden / fps, 3)
    if point != "centre":
        summary["Time_in_zone_rule"] = POINT_NAMES[point] + " point"
    for name in roi_names:
        col = f"In_{safe_col(name)}"
        z_t = tracked_df[col].sum() / fps if col in tracked_df else 0.0
        summary[f"{safe_col(name)}_time_s"] = float(z_t)
        summary[f"{safe_col(name)}_percent"] = float(z_t / tracked_time * 100) if tracked_time > 0 else 0.0

    if zone_groups:
        for group_name, members in zone_groups.items():
            valid_cols = [f"In_{safe_col(m)}" for m in members if f"In_{safe_col(m)}" in df.columns]
            if valid_cols:
                df[f"In_{safe_col(group_name)}"] = df[valid_cols].any(axis=1)
        tracked_df = df[df["Tracking_Status"].isin(PRESENT)]
        for group_name in zone_groups:
            gcol = f"In_{safe_col(group_name)}"
            if gcol not in df.columns:
                continue
            g_t = tracked_df[gcol].sum() / fps
            summary[f"{safe_col(group_name)}_time_s"] = float(g_t)
            summary[f"{safe_col(group_name)}_percent"] = float(g_t / tracked_time * 100) if tracked_time > 0 else 0.0
            label_col = f"_group_label_{safe_col(group_name)}"
            df[label_col] = np.where(df[gcol], group_name, "Outside")
            gtr = calculate_transitions(df, roi_col=label_col)
            summary[f"{safe_col(group_name)}_entries"] = calculate_arm_entries(gtr, [group_name])[group_name]
            df.drop(columns=[label_col], inplace=True)

    unit = setup.get("scale_unit") or "cm"
    distance_col = "Distance_pixels"
    if setup.get("scale_factor"):
        factor = float(setup["scale_factor"])
        df[f"Distance_{unit}"] = df["Distance_pixels"] * factor
        df[f"Velocity_{unit}_s"] = df["Velocity_pixels_s"] * factor
        summary[f"Total_distance_{unit}"] = summary["Total_distance_pixels"] * factor
        distance_col = f"Distance_{unit}"
    summary["Mean_speed_" + (f"{unit}_s" if setup.get("scale_factor") else "pixels_s")] = (
        float(df[distance_col].sum()) / visible_time if visible_time > 0 else 0.0)

    if not transitions.empty:
        for (frm, to), cnt in transitions.groupby(["From_ROI", "To_ROI"]).size().items():
            summary[f"Trans_{safe_col(frm)}_to_{safe_col(to)}"] = int(cnt)
    summary["Total_transitions"] = len(transitions)

    # ---- zone table + test-specific measures (EPM / Y / T / LD / OF / NOR / social)
    from analysis.test_metrics import compute_test_metrics
    zone_tbl, entries_df = pd.DataFrame(), pd.DataFrame()
    if roi_names:
        try:
            metrics, zone_tbl, entries_df = compute_test_metrics(
                df, roi_names, fps, start_time, test_key=setup.get("test_type"), distance_col=distance_col,
                entry_frac=entry_pct / 100.0)
            summary.update(metrics)
            for r in zone_tbl.itertuples(index=False):
                summary[f"{safe_col(r.Zone)}_entries"] = int(r.Entries)
                summary[f"{safe_col(r.Zone)}_latency_s"] = r.Latency_first_entry_s
        except Exception as exc:
            print(f"Test-specific metrics skipped: {exc}")
    if setup.get("compute_alternation") and "Spontaneous_alternation_pct" in summary:
        summary["Alternation_percent"] = summary["Spontaneous_alternation_pct"]

    # ---- nose point (object exploration): head end of the body, decided over the whole track
    if setup.get("track_nose") and raw.get("ends"):
        try:
            nose_exploration(df, raw, xs, ys, roi_names, roi_masks, fps, summary, setup.get("test_type"))
        except Exception as exc:
            print(f"Nose point skipped: {exc}")

    # ---- object proximity bouts (kept for object zones; no manual tagging) -----
    from tracking.time_bins import calculate_bins, format_bin_label
    bouts = extract_bouts(df, object_names, fps) if object_names else []
    tracked_df = df[df["Tracking_Status"].isin(PRESENT)]
    for name in object_names:
        col = f"Near_{safe_col(name)}"
        near_t = tracked_df[col].sum() / fps if col in tracked_df else 0.0
        summary[f"{safe_col(name)}_near_time_s"] = float(near_t)
        summary[f"{safe_col(name)}_near_percent"] = float(near_t / tracked_time * 100) if tracked_time > 0 else 0.0
        summary[f"{safe_col(name)}_bout_count"] = sum(1 for b in bouts if b["Object"] == name)
    summary["Total_interaction_bouts"] = len(bouts)

    # ---- time bins --------------------------------------------------------------
    bin_size_s = float(setup.get("bin_size_s") or 60.0)
    bin_label = format_bin_label(bin_size_s)
    summary["Bin_size_s"] = bin_size_s
    individual, cumulative = calculate_bins(df, fps, start_time, summary["End_s"], transitions, roi_names,
                                            bin_size_s=bin_size_s)
    if setup.get("scale_factor"):
        factor = float(setup["scale_factor"])
        individual[f"Distance_{unit}"] = individual["Distance_pixels"] * factor
        cumulative[f"Distance_{unit}"] = cumulative["Distance_pixels"] * factor

    summary["Trajectory_smoothing_applied"] = smoothing_applied
    if smoothing_applied:
        summary["Trajectory_smoothing_window_frames"] = int(smooth_window)

    if custom_variable_defs:
        df, custom_var_warnings = apply_custom_variables(
            df, custom_variable_defs,
            scalar_context={"scale_factor": setup.get("scale_factor") or 1.0, "FPS": fps, "fps": fps})
        for warning in custom_var_warnings:
            print(f"[custom variable] {warning}")
        summary["Custom_variables_added"] = ", ".join(n for n, _e in custom_variable_defs if n in df.columns)

    thig_margin = setup.get("thigmotaxis_margin_px")
    if thig_margin is None:
        thig_margin = int(round(min(warp_w, warp_h) * 0.10))
    if thig_margin and int(thig_margin) > 0 and setup.get("test_type") in (None, "open_field", "custom", "nor"):
        try:
            from analysis.thigmotaxis import thigmotaxis_metrics, flatten_for_summary
            thig = thigmotaxis_metrics(df, warp_w, warp_h, int(thig_margin), fps)
            summary.update(flatten_for_summary(thig))
            if write_files:
                save_thigmotaxis_chart(thig, output_dir)
        except Exception as exc:
            print(f"Thigmotaxis analysis skipped: {exc}")

    if not write_files:
        prog(1.0)
        return summary
    # ---- output files -------------------------------------------------------------
    write_csv_report(df, output_dir)
    roi_coords = pd.DataFrame(
        [{"ROI": name, "Vertices_xy": str([(round(x, 1), round(y, 1)) for x, y in pts])}
         for name, pts in roi_points.items()]
        + [{"ROI": name, "Vertices_xy": str(pts)} for name, pts in object_points.items()])
    interactions_df = pd.DataFrame(bouts) if bouts else pd.DataFrame(
        columns=["Object", "Frame_start", "Frame_end", "Start_seconds", "End_seconds", "Duration_seconds"])
    extra = {}
    if not zone_tbl.empty:
        extra["Zone Summary"] = zone_tbl
        zone_tbl.to_csv(os.path.join(output_dir, "zone_summary.csv"), index=False)
    if not entries_df.empty:
        extra["Arm Entries"] = entries_df
        entries_df.to_csv(os.path.join(output_dir, "arm_entries.csv"), index=False)
    write_excel_report(output_dir, video_name, df, summary, individual, cumulative,
                       transitions, roi_coords, interactions_df, bin_label=bin_label, extra_sheets=extra)

    # charts are drawn on a picture that looks like the original video (no masks / dimming)
    chart_bg = None
    try:
        if os.path.exists(video_path):
            n_s = int(min(15, max(1, len(frames))))
            pick = np.unique(np.linspace(0, len(frames) - 1, n_s).astype(int))
            tr_now = df["Tracking_Status"].to_numpy() == "Tracked"
            ax_, ay_ = df["Mouse_X"].to_numpy(float), df["Mouse_Y"].to_numpy(float)
            radius = max(8.0, 1.3 * math.sqrt(a_full)) if a_full else None
            xy = [(ax_[i], ay_[i]) if tr_now[i] else None for i in pick]
            chart_bg = original_view_image(video_path, [frames[i] for i in pick], warp=warp if need_warp else None,
                                           animal_xy=xy, animal_radius=radius)
    except Exception as exc:
        print(f"Original-view chart background skipped: {exc}")
    if chart_bg is None:
        chart_bg = raw.get("background_bgr")
    save_plots(df, output_dir, roi_points, object_points, warp_w, warp_h, background=chart_bg,
               x_col=traj_x_col, y_col=traj_y_col)
    save_zone_occupancy_chart(summary, roi_names, output_dir)
    try:
        save_zone_latency_chart(summary, roi_names, output_dir)
    except Exception as exc:
        print(f"Latency chart skipped: {exc}")
    try:
        save_timecourse_plots(df, fps, output_dir, scale_factor=setup.get("scale_factor"),
                              x_col=traj_x_col, y_col=traj_y_col)
    except Exception as exc:
        print(f"Time-course plots skipped: {exc}")
    try:                                    # one-page PDF for lab meetings / lab books
        from output.pdf_report import tracking_pdf
        tracking_pdf(output_dir, video_name, summary)
    except Exception as exc:
        print(f"PDF summary skipped: {exc}")
    try:                                    # a 16:9 picture for a lab-meeting slide
        from output.pdf_report import tracking_slide
        tracking_slide(output_dir, video_name, summary)
    except Exception as exc:
        print(f"Slide picture skipped: {exc}")
    prog(1.0)
    return summary
