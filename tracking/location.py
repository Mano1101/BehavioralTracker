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
from collections import Counter

from analysis.calculations import point_distance, safe_col
from analysis.custom_variables import apply_custom_variables
from tracking.interaction import extract_bouts
from tracking.epm import calculate_arm_entries, calculate_alternation
from output.csv import write_csv_report
from output.excel import write_excel_report

# Color palettes live in tracking/colors.py (shared with output/graphs.py
# without a circular import -- see that module's docstring).
from tracking.colors import ROI_COLORS, OBJECT_COLORS  # noqa: F401  (re-exported for existing importers)

from output.graphs import (
    save_plots, save_zone_occupancy_chart,
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
        cv2.polylines(display, [np.array(pts, dtype=np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
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
    roi_masks=None, exclusion_mask=None, color_mode="gray"
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
        # same technique ezTrack uses (dif[mask] = 0).
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
    valid = df[df["Tracking_Status"] == "Tracked"]
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
    threshold, min_area, max_area, n_samples, output_dir, show_live=True,
    mask_polygons=None, exclusion_mask=None, color_mode="gray"
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
            float("inf"), roi_masks=roi_masks, exclusion_mask=exclusion_mask, color_mode=color_mode
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


def compute_output_dir(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), "results", video_name)
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


def process_single_video(video_path, setup, show_display=True, progress_callback=None, confirm_callback=None,
                          frame_callback=None):
    """frame_callback(display_bgr), if given, is handed the SAME annotated
    frame that show_display=True would otherwise put in a cv2.imshow
    window -- used by the Qt app to draw a live tracking view inside its
    own preview canvas instead (see the show_display branch below for why
    cv2.imshow itself can't be used there: it hard-crashes when a Qt5
    OpenCV build and this PySide6/Qt6 app are both active in the same
    process). Called on the same throttled cadence as progress_callback,
    independent of show_display."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    if setup.get("fps_override"):
        fps = float(setup["fps_override"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_time = max(0, min(setup["start_time"], duration))
    end_time = max(start_time, min(setup["end_time"], duration))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    matrix = setup["matrix"]
    warp_w = setup["warp_w"]
    warp_h = setup["warp_h"]
    arena = setup["arena"]
    roi_points = setup["roi_points"]
    roi_names = setup["roi_names"]
    roi_masks = build_roi_masks((warp_h, warp_w), roi_points)

    # Zone Associations (optional) -- report 2+ existing zones as one named
    # combined zone too, without redrawing anything (mirrors SMART's zone
    # grouping). {"Left Side": ["Left Arm", "Left Corner"], ...}; built from
    # setup_page's own Zone Associations box, see qt_app/main_window.py's
    # _build_setup. Used further below, once roi_names' own per-zone stats
    # (which member zones borrow their In_<zone> columns from) are computed.
    zone_groups = setup.get("zone_groups", {})

    # Trajectory smoothing / outlier filter (optional) -- mirrors SMART's
    # Anti-Vibration/Anti-Artifact/LOWESS filters. None/0 (the default)
    # means "off, exactly as before this feature existed"; an integer N
    # applies smooth_trajectory() (below) with an N-frame window once the
    # full tracking loop has produced its raw per-frame positions.
    smooth_window = setup.get("smooth_window")

    # Custom Variables (optional) -- EthoVision-style user-written derived
    # columns, applied further below once every other column (including
    # zone/smoothing columns) already exists to reference. A list of
    # (name, expression) pairs from setup_page's Custom Variables box, see
    # qt_app/main_window.py's _build_setup / analysis/custom_variables.py.
    custom_variable_defs = setup.get("custom_variables") or []

    # roi_masks (above) is ALWAYS used for location/zone membership -- which
    # zone(s) the animal is in, regardless of this setting. detection_roi_masks
    # is what actually reaches detect_mouse()'s per-zone Otsu split, and is
    # only non-None when the zones represent genuinely different lighting
    # (see step_detection_settings). Decoupling these two avoids an
    # overlapping analysis-only region (e.g. a "Whole_Arena" zone) from
    # starving the per-zone threshold split of pixels.
    use_zone_threshold = setup.get("use_zone_threshold", False)
    detection_roi_masks = roi_masks if use_zone_threshold else None

    object_points = setup.get("object_points", {})
    object_names = setup.get("object_names", [])
    # interaction_margin_px extends each object's own drawn outline outward
    # by this many pixels before checking "is the animal near this object" --
    # see dilate_masks() -- so approaching/sniffing an object from just
    # outside it still counts as interaction, not only stepping inside it.
    interaction_margin_px = setup.get("interaction_margin_px", 0)
    object_masks = build_roi_masks((warp_h, warp_w), object_points) if object_points else {}
    object_masks = dilate_masks(object_masks, interaction_margin_px)
    behavior_names = setup.get("behavior_names", [])

    mask_polygons = setup.get("mask_points", [])
    # Auto-mask everything outside my zones (optional, on by default) --
    # once zones are drawn/finalized (roi_points non-empty), anything
    # outside their union is excluded from detection too, same pipeline
    # as a hand-drawn Mask Zone shape. See build_exclusion_mask's
    # docstring: a no-op when there are no zones yet.
    auto_mask_outside_zones = bool(setup.get("auto_mask_outside_zones")) and bool(roi_points)
    zone_polygons = list(roi_points.values()) if auto_mask_outside_zones else None
    exclusion_mask = None
    if mask_polygons or auto_mask_outside_zones:
        exclusion_mask = build_exclusion_mask(
            (warp_h, warp_w), mask_polygons,
            zone_polygons=zone_polygons, mask_outside_zones=auto_mask_outside_zones,
        )

    threshold = setup["threshold"]
    min_area = setup["min_area"]
    max_area = setup["max_area"]
    max_jump = setup["max_jump"]

    # Optional early-stop rule -- inspired by EthoVision's Trial Control
    # rules and SMART's Status Rules, but simplified to the three cases
    # most apparatus protocols actually need. None/None/None means "run to
    # the End (s) time above, as before" (the default, unchanged behavior).
    stop_condition = setup.get("stop_condition")  # None, "immobility", "zone_entries", or "distance"
    stop_value = setup.get("stop_value")
    stop_zone_name = setup.get("stop_zone_name")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if setup.get("output_dir_override"):
        output_dir = os.path.join(setup["output_dir_override"], video_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = compute_output_dir(video_path)

    color_mode = setup.get("color_mode", "gray")

    background = make_background(
        cap, start_frame, end_frame, matrix, warp_w, warp_h, setup["background_samples"],
        color_mode=color_mode
    )

    if show_display:
        cv2.imshow("Automatic background - press ENTER", background)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32):
                break
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit("Cancelled.")
        cv2.destroyWindow("Automatic background - press ENTER")

    run_detection_preview(
        cap, matrix, warp_w, warp_h, start_frame, end_frame,
        background, arena, detection_roi_masks, roi_points, object_points,
        threshold, min_area, max_area, setup["preview_samples"], output_dir, show_live=show_display,
        mask_polygons=mask_polygons, exclusion_mask=exclusion_mask, color_mode=color_mode
    )

    # -----------------------------------------
    # Tracking loop
    # -----------------------------------------

    previous_point = None
    previous_area = None
    previous_time = start_time

    rows = []
    frame_number = start_frame

    max_recovery_streak = 10
    recovery_streak = 0

    # Bookkeeping for the optional early-stop rule (see stop_condition
    # above) -- only the branch that matches stop_condition is ever
    # updated, but all three are initialized unconditionally since that's
    # cheap and keeps the per-frame block below simple.
    stop_reason = None
    stop_immobile_seconds = 0.0
    stop_zone_entry_count = 0
    stop_zone_was_in = False
    stop_cumulative_distance_px = 0.0
    STOP_IMMOBILITY_PIXEL_THRESHOLD = 2.0  # px/frame below this counts as "not moving"

    print("\nStarting automatic tracking...")
    if show_display:
        print("Press Q during tracking to stop.")

    while frame_number <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, raw_frame = cap.read()

        if not ok:
            break

        frame = cv2.warpPerspective(raw_frame, matrix, (warp_w, warp_h))
        current_time = frame_number / fps

        candidate, mask = detect_mouse(
            frame, background, arena, previous_point, previous_area,
            threshold, max(1, int(min_area)), max(int(max_area), int(min_area) + 1), max_jump,
            roi_masks=detection_roi_masks, exclusion_mask=exclusion_mask, color_mode=color_mode
        )

        used_recovery = False

        if candidate is None and previous_point is not None:
            candidate = local_recovery(
                frame, background, previous_point, arena,
                threshold, max(1, int(min_area)), max(int(max_area), int(min_area) + 1),
                exclusion_mask=exclusion_mask, color_mode=color_mode
            )
            used_recovery = candidate is not None

            if candidate is not None and point_distance(candidate["center"], previous_point) > max_jump:
                candidate = None
                used_recovery = False

        if used_recovery and candidate is not None:
            recovery_streak += 1
            if recovery_streak > max_recovery_streak:
                candidate = None
                previous_point = None
                previous_area = None
        elif candidate is not None:
            recovery_streak = 0

        x = np.nan
        y = np.nan
        area = np.nan
        distance_px = 0.0
        velocity = 0.0
        zone_membership = {name: False for name in roi_names}
        object_membership = {name: False for name in object_names}

        if candidate is not None:
            x, y = candidate["center"]
            area = candidate["area"]

            smoothing_alpha = 0.6
            if previous_point is not None:
                x = smoothing_alpha * x + (1 - smoothing_alpha) * previous_point[0]
                y = smoothing_alpha * y + (1 - smoothing_alpha) * previous_point[1]
                distance_px = point_distance((x, y), previous_point)

            dt = max(1.0 / fps, current_time - previous_time)
            velocity = distance_px / dt

            previous_point = (x, y)
            previous_area = area

            zone_membership = roi_membership(roi_masks, x, y)
            roi_label = linearize_roi(zone_membership)
            if object_masks:
                object_membership = roi_membership(object_masks, x, y)
            status = "Tracked"
        else:
            roi_label = "None"
            status = "Lost"

        row = {
            "Frame": frame_number,
            "Time_seconds": current_time,
            "Mouse_X": x,
            "Mouse_Y": y,
            "Mouse_Present": status == "Tracked",
            "ROI": roi_label,
            "Detected_area": area,
            "Distance_pixels": distance_px,
            "Velocity_pixels_s": velocity,
            "Tracking_Status": status
        }

        for name in roi_names:
            row[f"In_{safe_col(name)}"] = zone_membership.get(name, False)
        for name in object_names:
            row[f"Near_{safe_col(name)}"] = object_membership.get(name, False)

        rows.append(row)

        # Early-stop rule check -- updates whichever counter matches
        # stop_condition and sets stop_reason once the target is reached.
        # The actual `break` happens further below (after the progress/
        # frame callbacks for this frame), so the run still reports/paints
        # its final frame before stopping.
        if stop_condition and stop_value is not None:
            frame_dt = max(0.0, current_time - previous_time)
            if stop_condition == "immobility":
                if status == "Tracked" and distance_px < STOP_IMMOBILITY_PIXEL_THRESHOLD:
                    stop_immobile_seconds += frame_dt
                else:
                    stop_immobile_seconds = 0.0
                if stop_immobile_seconds >= stop_value:
                    stop_reason = f"Immobile for {stop_value:g}s"
            elif stop_condition == "zone_entries" and stop_zone_name:
                now_in_zone = zone_membership.get(stop_zone_name, False)
                if now_in_zone and not stop_zone_was_in:
                    stop_zone_entry_count += 1
                stop_zone_was_in = now_in_zone
                if stop_zone_entry_count >= stop_value:
                    stop_reason = f"{int(stop_value)} entries into '{stop_zone_name}'"
            elif stop_condition == "distance":
                stop_cumulative_distance_px += distance_px
                if stop_cumulative_distance_px >= stop_value:
                    stop_reason = f"Reached {stop_value:g}px total distance"

        # Built every frame when show_display=True (a live cv2 window needs
        # each frame redrawn), or every 5th frame -- matching
        # progress_callback's own cadence below -- when only frame_callback
        # wants an occasional snapshot for the Qt live tracking view.
        build_display = show_display or (frame_callback is not None and (frame_number + 1) % 5 == 0)
        if build_display:
            display = frame.copy()
            cv2.rectangle(display, (arena[0], arena[1]), (arena[2], arena[3]), (255, 255, 0), 1)
            if mask_polygons:
                draw_mask_polygons(display, mask_polygons)
            draw_roi_polygons(display, roi_points)
            draw_object_markers(display, object_points)

            # Fixed 35px-per-line rhythm below the status text -- matches
            # this block's own original single-line layout (Time: used to
            # sit at a hardcoded y=75, exactly 35px under the y=40 status
            # line; Frame: another 30px under that), so the common one-line
            # case looks pixel-identical to before. status_bottom is
            # advanced dynamically so a WRAPPED, multi-line status (see
            # below) pushes Time:/Frame: down instead of overlapping it.
            status_line_h = 35
            status_bottom = 40 + status_line_h

            if status == "Tracked":
                draw_crosshair(display, (int(x), int(y)), size=9, color=(0, 0, 255), thickness=2)
                cv2.putText(
                    display, "AUTO TRACK", (max(5, int(x) - 50), max(20, int(y) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
                near_list = [n for n, v in object_membership.items() if v]
                segments = ["ROI: " + roi_label]
                if near_list:
                    segments.append("Near: " + ", ".join(near_list))

                # Wrapped instead of one fixed cv2.putText call -- on a
                # narrower video (EPM/Y-maze/T-maze clips are often
                # cropped down close to the apparatus itself) a status line
                # with a couple of zones in the Entry: list easily ran past
                # the frame's right edge and got silently clipped off-
                # screen, unreadable (MM's report). wrap_status_line keeps
                # it fully on-screen by stacking onto extra lines instead.
                status_font, status_scale, status_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                status_max_w = max(60, display.shape[1] - 40)  # 20px margin each side, matching the x=20 draw origin
                status_lines = wrap_status_line(segments, status_font, status_scale, status_thick, status_max_w)
                y = 40
                for line in status_lines:
                    cv2.putText(display, line, (20, y), status_font, status_scale, (0, 255, 255), status_thick)
                    y += status_line_h
                status_bottom = y
            else:
                cv2.putText(display, "TRACKING LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.putText(display, f"Time: {current_time:.2f} s", (20, status_bottom),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display, f"Frame: {frame_number}", (20, status_bottom + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            if show_display:
                cv2.imshow("BEHAVIORAL TRACKING - Q to stop", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("\nTracking stopped by user.")
                    break

        frame_number += 1
        previous_time = current_time

        if progress_callback is not None and frame_number % 5 == 0:
            span = max(1, end_frame - start_frame)
            progress_callback(min(1.0, (frame_number - start_frame) / span))
        if frame_callback is not None and build_display and frame_number % 5 == 0:
            frame_callback(display)

        if stop_reason is not None:
            break
    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    # -----------------------------------------
    # Data
    # -----------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit("No tracking data were produced.")

    transitions = calculate_transitions(df)
    tracked = df[df["Tracking_Status"] == "Tracked"]
    tracked_time = len(tracked) / fps
    tracking_quality = 100 * len(tracked) / len(df) if len(df) > 0 else 0

    summary = {
        "Video": video_path,
        "Output_folder": output_dir,
        "Start_s": start_time,
        "End_s": end_time,
        "FPS": fps,
        "Frames_processed": len(df),
        "Tracked_frames": len(tracked),
        "Lost_frames": len(df) - len(tracked),
        "Tracking_quality_percent": tracking_quality,
        "Total_distance_pixels": tracked["Distance_pixels"].sum(),
        "Stop_Reason": stop_reason if stop_reason is not None else "Reached End (s)",
    }

    for name in roi_names:
        col = f"In_{safe_col(name)}"
        name_time = tracked[col].sum() / fps if col in tracked else 0.0
        summary[f"{safe_col(name)}_time_s"] = name_time
        summary[f"{safe_col(name)}_percent"] = (name_time / tracked_time * 100) if tracked_time > 0 else 0

    # Zone Associations -- fold each named group's member zones into one
    # OR-combined "zone" for reporting: In_<Group> is true whenever the
    # animal is in ANY member zone, then time/percent are computed exactly
    # like a real zone above, and entries reuse the same debounced
    # calculate_transitions/calculate_arm_entries machinery real zones use
    # (via a throwaway per-frame label column) so a brief single-frame
    # flicker across a member zone's own boundary isn't double-counted as a
    # false entry into the group.
    if zone_groups:
        for group_name, members in zone_groups.items():
            valid_cols = [f"In_{safe_col(m)}" for m in members if f"In_{safe_col(m)}" in df.columns]
            if valid_cols:
                df[f"In_{safe_col(group_name)}"] = df[valid_cols].any(axis=1)

        tracked = df[df["Tracking_Status"] == "Tracked"]  # refresh -- pick up the new In_<Group> columns

        for group_name in zone_groups:
            gcol = f"In_{safe_col(group_name)}"
            if gcol not in df.columns:
                continue  # none of this group's member zones exist in this run -- skip it silently
            g_time = tracked[gcol].sum() / fps if gcol in tracked else 0.0
            summary[f"{safe_col(group_name)}_time_s"] = g_time
            summary[f"{safe_col(group_name)}_percent"] = (g_time / tracked_time * 100) if tracked_time > 0 else 0

            label_col = f"_group_label_{safe_col(group_name)}"
            df[label_col] = np.where(df[gcol], group_name, "Outside")
            group_transitions = calculate_transitions(df, roi_col=label_col)
            summary[f"{safe_col(group_name)}_entries"] = calculate_arm_entries(
                group_transitions, [group_name]
            )[group_name]
            df.drop(columns=[label_col], inplace=True)

    if setup.get("scale_factor"):
        factor = setup["scale_factor"]
        unit = setup["scale_unit"]
        df[f"Distance_{unit}"] = df["Distance_pixels"] * factor
        summary[f"Total_distance_{unit}"] = summary["Total_distance_pixels"] * factor

    if not transitions.empty:
        pair_counts = transitions.groupby(["From_ROI", "To_ROI"]).size()
        for (frm, to), cnt in pair_counts.items():
            summary[f"Trans_{safe_col(frm)}_to_{safe_col(to)}"] = int(cnt)

    summary["Total_transitions"] = len(transitions)

    if setup.get("compute_arm_entries"):
        for name, count in calculate_arm_entries(transitions, roi_names).items():
            summary[f"{safe_col(name)}_entries"] = count

    if setup.get("compute_alternation"):
        summary.update(calculate_alternation(transitions))

    # -----------------------------------------
    # Interaction bouts
    # -----------------------------------------

    # Local import: tracking.behaviour imports FROM this module at its own
    # top level (for read_and_warp/draw_roi_polygons/draw_object_markers),
    # so importing it back at this module's top level would be circular.
    #
    # The yes/no confirmation below used to import gui.main_window.ask_yes_no
    # directly, which quietly made this GUI-agnostic module depend on the
    # Tkinter GUI (an architecture violation neither GUI's own code caught,
    # since it's a runtime import, not a module-level one). confirm_callback
    # lets each GUI supply its own yes/no prompt (a real Tk dialog, a Qt
    # QMessageBox, or `lambda *a: False` for a fully unattended run) instead;
    # when the caller doesn't pass one, this still falls back to the
    # Tkinter dialog so the existing Tkinter app's behavior is unchanged.
    from tracking.behaviour import tag_interaction_bouts, calculate_bins, format_bin_label
    if confirm_callback is None:
        from gui.main_window import ask_yes_no as confirm_callback
    bouts = extract_bouts(df, object_names, fps) if object_names else []

    if bouts and show_display:
        if confirm_callback(
            "Tag interactions",
            f"{len(bouts)} interaction bout(s) detected near your objects.\nReview and tag each one now?"
        ):
            bouts = tag_interaction_bouts(video_path, matrix, warp_w, warp_h, bouts, behavior_names,
                                           roi_points, object_points)
        else:
            for b in bouts:
                b["Behavior"] = "Unclassified"
    else:
        for b in bouts:
            b["Behavior"] = "Unclassified"

    for name in object_names:
        col = f"Near_{safe_col(name)}"
        near_time = tracked[col].sum() / fps if col in tracked else 0.0
        summary[f"{safe_col(name)}_near_time_s"] = near_time
        summary[f"{safe_col(name)}_near_percent"] = (near_time / tracked_time * 100) if tracked_time > 0 else 0
        summary[f"{safe_col(name)}_bout_count"] = sum(1 for b in bouts if b["Object"] == name)

    summary["Total_interaction_bouts"] = len(bouts)

    if behavior_names and bouts:
        counts = Counter((b["Object"], b["Behavior"]) for b in bouts if b.get("Behavior") not in (None, "Discarded"))
        for (obj, beh), cnt in counts.items():
            summary[f"Interact_{safe_col(obj)}_{safe_col(beh)}_count"] = cnt

    # -----------------------------------------
    # Bins -- configurable size (Upgrade Plan Tier 2 #9, EthoVision's Time
    # Bins/Nesting); setup["bin_size_s"] unset/0 keeps the original fixed
    # 1-minute behavior (see calculate_bins/format_bin_label's own docs).
    # -----------------------------------------

    bin_size_s = setup.get("bin_size_s") or 60.0
    bin_label = format_bin_label(bin_size_s)
    summary["Bin_size_s"] = bin_size_s
    individual, cumulative = calculate_bins(df, fps, start_time, end_time, transitions, roi_names,
                                             bin_size_s=bin_size_s)

    if setup.get("scale_factor"):
        factor = setup["scale_factor"]
        unit = setup["scale_unit"]
        individual[f"Distance_{unit}"] = individual["Distance_pixels"] * factor
        cumulative[f"Distance_{unit}"] = cumulative["Distance_pixels"] * factor

    # -----------------------------------------
    # Trajectory smoothing (optional) -- adds Mouse_X_smooth/Mouse_Y_smooth
    # columns (raw_tracking.csv still keeps the real Mouse_X/Mouse_Y
    # untouched); the Trajectory/Heatmap charts below are pointed at
    # whichever pair was actually requested.
    # -----------------------------------------

    traj_x_col, traj_y_col = "Mouse_X", "Mouse_Y"
    summary["Trajectory_smoothing_applied"] = False
    if smooth_window and smooth_window >= 2:
        df = smooth_trajectory(df, int(smooth_window))
        traj_x_col, traj_y_col = "Mouse_X_smooth", "Mouse_Y_smooth"
        summary["Trajectory_smoothing_applied"] = True
        summary["Trajectory_smoothing_window_frames"] = int(smooth_window)

    # -----------------------------------------
    # Custom variables (optional) -- Upgrade Plan Tier 2 #7, EthoVision's
    # own custom variables: user-written expressions against whatever
    # columns df already has at this point (including the zone/smoothing
    # columns added above), safely sandboxed (analysis/custom_variables.py
    # -- a small hand-rolled AST whitelist, not eval()/exec()/df.eval()).
    # A failing expression is skipped with a console warning rather than
    # aborting an otherwise-successful run.
    # -----------------------------------------
    if custom_variable_defs:
        df, custom_var_warnings = apply_custom_variables(
            df, custom_variable_defs,
            scalar_context={"scale_factor": setup.get("scale_factor") or 1.0, "FPS": fps, "fps": fps},
        )
        for warning in custom_var_warnings:
            print(f"[custom variable] {warning}")
        summary["Custom_variables_added"] = [name for name, _expr in custom_variable_defs
                                              if name in df.columns]
        if custom_var_warnings:
            summary["Custom_variables_warnings"] = custom_var_warnings

    # -----------------------------------------
    # Thigmotaxis (open-field border/center occupancy) -- computed BEFORE
    # the Summary sheet is written so its columns land in the Excel/CSV
    # output. Border margin defaults to 10% of the smaller arena side;
    # override with setup["thigmotaxis_margin_px"], disable with 0.
    # -----------------------------------------
    thig_margin = setup.get("thigmotaxis_margin_px")
    if thig_margin is None:
        thig_margin = int(round(min(warp_w, warp_h) * 0.10))
    if thig_margin and int(thig_margin) > 0:
        try:
            from analysis.thigmotaxis import thigmotaxis_metrics, flatten_for_summary
            thig = thigmotaxis_metrics(df, warp_w, warp_h, int(thig_margin), fps)
            summary.update(flatten_for_summary(thig))
            save_thigmotaxis_chart(thig, output_dir)
        except Exception as exc:
            print(f"Thigmotaxis analysis skipped: {exc}")

    # -----------------------------------------
    # Output files -- see output/csv.py, output/excel.py, output/graphs.py
    # -----------------------------------------

    write_csv_report(df, output_dir)

    roi_coords = pd.DataFrame(
        [{"ROI": name, "Vertices_xy": str(pts)} for name, pts in roi_points.items()]
        + [{"ROI": name, "Vertices_xy": str(pts)} for name, pts in object_points.items()]
    )

    interactions_df = pd.DataFrame(bouts) if bouts else pd.DataFrame(
        columns=["Object", "Frame_start", "Frame_end", "Start_seconds", "End_seconds",
                 "Duration_seconds", "Behavior"]
    )

    write_excel_report(
        output_dir, video_name, df, summary, individual, cumulative,
        transitions, roi_coords, interactions_df, bin_label=bin_label,
    )

    save_plots(df, output_dir, roi_points, object_points, warp_w, warp_h, background=background,
               x_col=traj_x_col, y_col=traj_y_col)
    save_zone_occupancy_chart(summary, roi_names, output_dir)

    # Distance/speed time-course charts (EthoVision/ANY-maze parity) --
    # plotted from the same possibly-smoothed trajectory columns the
    # trajectory chart uses.
    try:
        save_timecourse_plots(df, fps, output_dir,
                              scale_factor=setup.get("scale_factor"),
                              x_col=traj_x_col, y_col=traj_y_col)
    except Exception as exc:
        print(f"Time-course plots skipped: {exc}")

    # -----------------------------------------
    # Final report
    # -----------------------------------------

    print("\n" + "=" * 65)
    print("TRACKING COMPLETE")
    print("=" * 65)
    print(f"Tracking quality: {tracking_quality:.2f}%")

    for name in roi_names:
        print(f"{name} time: {summary.get(f'{safe_col(name)}_time_s', 0):.2f} s")

    print(f"Total transitions: {len(transitions)}")

    if object_names:
        print(f"Interaction bouts: {len(bouts)}")

    print("\nResults folder:")
    print(output_dir)

    return summary


# ============================================================
# MAIN
# ============================================================
