"""
tracking/behavior.py
-----------------------
Automated grooming / rearing / locomotion behavior classification, built on
top of the same background-subtraction detection used by two_mouse.py.
Outputs bout-level events (subject, behavior, start, stop, duration).

v2 ENGINE (rewritten from a single-cue heuristic)
--------------------------------------------------
The previous version decided a frame's behavior from ONE feature at a time
(raw velocity, contour-area-drop, or a local-pixel-motion score). That is
easy to fool: a light reflection can spike "motion energy" and get called
grooming; a hunched, motionless posture can shrink the silhouette and get
called rearing.

This version instead derives an approximate MULTI-POINT POSE per animal
every frame -- nose, ears, neck, body center, both forepaws, tail base,
both hindlimbs -- purely from the shape of its foreground silhouette (PCA
on the contour's pixel coordinates, no deep learning, no training data,
no GPU). Geometric features are computed FROM that pose (paw-to-nose
distance, head/forelimb elevation relative to the animal's own resting
baseline, repetitive-motion score, body-movement, posture-change-rate...),
and each behavior is scored as a WEIGHTED COMBINATION of several of those
features -- never a single threshold -- so no single noisy cue can flip
the label by itself. A frame is only confirmed as a new behavior after the
score holds for several consecutive frames (temporal persistence /
hysteresis), and frames where the pose itself is unreliable (silhouette
too small/fragmented) are marked "undetermined" instead of being guessed.

This design -- the pose-from-silhouette approach, the specific per-behavior
feature set, and the weighted-scoring + hysteresis classifier -- is adapted
from a more complete reference implementation the user supplied, kept
self-contained here (no new package, no ML dependency) so it drops
straight into this app's existing per-file tracking/ layout.

Detection/tracking itself -- background model, foreground mask, per-animal
blob splitting for touching mice, Hungarian-algorithm identity matching
across frames -- is UNCHANGED from two_mouse.py, so this behaves
consistently with Multi-Mouse Tracking for 1-3 animals.

WHY TWO STAGES (extract, then classify)
----------------------------------------
Feature extraction (reading every video frame) is slow. Threshold tuning is
something you'll do many times. So this is split into:

  1. extract_features - run once per video. Walks the video, tracks each
                 animal, derives its pseudo-pose each frame, and computes
                 the full geometric feature set. Saves these to a CSV.

  2. classify_behaviors - reads the features CSV (fast, no video needed)
                 and turns per-frame features into confirmed behavior
                 labels, then into bouts with durations. Re-run this as
                 many times as you like with different thresholds without
                 re-processing video.

THIS IS A PROXY, NOT GROUND TRUTH. Precision comes from validation against
manual (e.g. BORIS) scoring, not from the algorithm alone.

USAGE
-----
    python -m tracking.behavior extract --video v.mp4 --output features.csv
    python -m tracking.behavior classify --features features.csv \
        --output bouts.csv --labeled-output labeled_frames.csv
"""

import argparse
import glob
import math
import os
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking.two_mouse import (
    build_background, foreground_mask, MotionHistory, reject_static_debris,
    DEFAULTS as TRACKER_DEFAULTS,
)

DEFAULTS = dict(
    **TRACKER_DEFAULTS,
    # bout rules
    min_bout_s=0.3,          # bouts shorter than this (after gap-bridging) are dropped
    merge_gap_s=0.5,         # a short differing run sandwiched between two matching
                              # runs (e.g. a 1-frame blip mid-grooming) is bridged
    # pose smoothing / quality gating
    pose_min_confidence=0.4,     # below this, a keypoint is treated as not-confident
    pose_smoothing_alpha=0.35,   # EMA smoothing factor for pose keypoints (0=frozen, 1=no smoothing)
    history_window_frames=15,    # rolling window for velocity/repetition features
    pixels_per_cm=0.0,           # 0 => features stay in pixel units
    max_jump_px=100.0,           # max plausible per-frame centroid/nose movement, px
    # grooming (weighted score: stillness + local head/paw motion energy + repetitiveness)
    #
    # NOTE on why this uses raw pixel motion near the head, not a "forepaw
    # keypoint velocity": the pseudo-pose below derives forepaw/hindlimb
    # positions from a FIXED template along the body's principal axis --
    # it has no learned model of mouse anatomy, so it cannot resolve a paw
    # moving independently of the head/body (this is a real, inherent
    # limitation of any classical-CV/no-deep-learning pose backend, not a
    # bug to be tuned away -- see the module docstring). What a plain
    # pixel-difference window CAN see, though, is genuine local motion:
    # grooming involves repeated small limb/head movements IN PLACE, which
    # shows up as real frame-to-frame pixel change in a small window
    # around the head even while the body centroid isn't translating.
    groom_body_move_low=15.0,      # px/s, below this the body is considered stationary
    groom_roi_size=40,             # px, window size for local head-motion energy
    groom_head_motion_high=4.0,    # mean abs pixel diff above this (while body is still) = active grooming motion
    groom_repetitive_high=0.30,    # normalized repetitiveness of that local motion (0-1)
    groom_posture_wobble_high=3.0, # deg/s of body-axis wobble above this = active in-place postural movement
                                    # (a real, contour-derived signal -- distinct from the fixed pose template's
                                    # forepaw landmarks -- that a hunched/grooming posture shows and true stillness doesn't)
    groom_min_persistence=7,       # frames the score must hold before confirming
    groom_score_threshold=0.50,    # weighted score >= this => grooming
    # rearing (weighted score: head/forelimb elevation + hindlimb stability + vertical onset).
    # Thresholds are calibrated for THIS classical silhouette-based backend's
    # own achievable elevation range, which reads lower than a trained
    # pose model would for the same physical rear (see module docstring).
    rear_upright_elevation_high=0.22,  # normalized head elevation proxy (0-1)
    rear_forelimb_elevated=0.18,       # normalized forelimb elevation proxy
    rear_hindlimb_stability_high=0.75, # normalized stability (0-1, higher = stiller hindlimbs)
    rear_vertical_move_min=8.0,        # px/s, minimum vertical head speed to start a rear
    rear_min_persistence=4,
    rear_score_threshold=0.25,
    # locomotion / resting (simple complement of windowed body movement)
    loco_body_move_high=30.0,  # px/s (windowed average) => moving through the arena
)

_LABEL_MAP = {
    "GROOMING": "grooming",
    "REARING": "rearing",
    "LOCOMOTION": "locomotion",
    "RESTING": "immobile",
    "UNKNOWN": "undetermined",
}

BODY_PARTS = [
    "nose", "left_ear", "right_ear", "neck", "body_center",
    "left_forepaw", "right_forepaw", "tail_base",
    "left_hindlimb", "right_hindlimb",
]

_SKELETON_EDGES = [
    ("nose", "left_ear"), ("nose", "right_ear"), ("nose", "neck"),
    ("left_ear", "neck"), ("right_ear", "neck"), ("neck", "body_center"),
    ("neck", "left_forepaw"), ("neck", "right_forepaw"),
    ("body_center", "left_forepaw"), ("body_center", "right_forepaw"),
    ("body_center", "tail_base"),
    ("body_center", "left_hindlimb"), ("body_center", "right_hindlimb"),
    ("tail_base", "left_hindlimb"), ("tail_base", "right_hindlimb"),
]


# --------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------

def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _perp(v):
    return (-v[1], v[0])


def _angle_deg(a, b):
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _norm01(value, lo, hi):
    if hi <= lo:
        return 0.0
    return _clamp((value - lo) / (hi - lo), 0.0, 1.0)


def _bucket(value, low_hi, high_lo, low="LOW", medium="MEDIUM", high="HIGH"):
    if value < low_hi:
        return low
    if value >= high_lo:
        return high
    return medium


# --------------------------------------------------------------------------
# Lightweight pose representation (no external pose package -- kept local
# to this file, same idea as pose/base.py in the reference implementation)
# --------------------------------------------------------------------------

class Keypoint:
    __slots__ = ("x", "y", "confidence")

    def __init__(self, x, y, confidence=0.0):
        self.x = x
        self.y = y
        self.confidence = confidence

    def xy(self):
        return (self.x, self.y)


class PoseFrame:
    __slots__ = ("frame_idx", "time", "keypoints", "track_quality")

    def __init__(self, frame_idx, time, keypoints=None, track_quality=0.0):
        self.frame_idx = frame_idx
        self.time = time
        self.keypoints = keypoints if keypoints is not None else {}
        self.track_quality = track_quality

    def get(self, part):
        return self.keypoints.get(part)

    def xy(self, part):
        kp = self.keypoints.get(part)
        return kp.xy() if kp else None


# --------------------------------------------------------------------------
# STAGE 0: SEGMENTATION + PSEUDO-POSE (classical CV, no deep learning)
# --------------------------------------------------------------------------

def _solidity_from_contour(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 1e-6:
        return 0.0
    return float(np.clip(area / hull_area, 0.0, 1.0))


def _solidity_from_points(pts):
    """Confidence proxy for a raw point cloud (used when we only have a
    k-means-split cluster from a merged blob, not a clean single contour)."""
    if pts is None or len(pts) < 3:
        return 0.3
    hull = cv2.convexHull(pts.astype(np.float32))
    hull_area = cv2.contourArea(hull)
    x0, y0, w, h = cv2.boundingRect(pts.astype(np.int32))
    bbox_area = max(w * h, 1.0)
    return float(np.clip(hull_area / bbox_area, 0.0, 1.0)) * 0.9  # slightly discounted: less certain than a real contour


def _disambiguate_ends(end_a, end_b, centroid, pts, proj_major, proj_minor, prev_pose, max_jump_px):
    """Decide which contour extreme is the nose vs. tail-base end: prefer
    continuity with the previous frame's nose position; otherwise assume
    the narrower end (in local cross-section) is the nose.

    The local-cross-section width is measured along proj_minor -- the
    point cloud's own PCA-derived perpendicular axis -- rather than the
    raw image x/y axes, so it stays correct regardless of which way the
    animal happens to be lying (not just near-axis-aligned headings).

    Which points count as "near this end" is chosen by DISTANCE along the
    major axis (the outermost slice_frac of the animal's own length),
    not by how many of the contour's points happen to fall there. A
    count-based band (e.g. "the nearest N points by index") is at the
    mercy of the contour approximation's point density, which is very
    uneven for an elongated, part-pointed silhouette: a sharp tip needs
    only 1-2 vertices to represent under CHAIN_APPROX_SIMPLE, while a
    comparable-length flatter stretch nearby can contribute many more --
    so a fixed point-count band can end up mixing in points from well
    past the actual tip (or missing the tip's own points), corrupting the
    width estimate at many headings. A distance-based slice always
    measures the same physical extent of the animal regardless of how
    densely that stretch happened to be sampled."""
    prev_nose = prev_pose.xy("nose") if prev_pose else None
    if prev_nose and prev_pose.get("nose") and prev_pose.get("nose").confidence > 0.05:
        da, db = _distance(end_a, prev_nose), _distance(end_b, prev_nose)
        if min(da, db) <= max_jump_px:
            return (end_a, end_b) if da <= db else (end_b, end_a)

    span = float(proj_major.max() - proj_major.min())
    slice_len = max(2.0, span * 0.15)
    top_idx = np.where(proj_major >= proj_major.max() - slice_len)[0]
    bot_idx = np.where(proj_major <= proj_major.min() + slice_len)[0]

    def local_width(idxs):
        return float(np.ptp(proj_minor[idxs])) if len(idxs) else 0.0

    width_a, width_b = local_width(top_idx), local_width(bot_idx)
    return (end_a, end_b) if width_a <= width_b else (end_b, end_a)


def _pose_from_points(points, centroid, frame_idx, time, prev_pose, max_jump_px, base_conf, split=False):
    """Derive 10 approximate body-part landmarks from the foreground pixel
    coordinates belonging to ONE animal this frame, via PCA on the point
    cloud (principal axis = head-tail axis) plus a fixed landmark template
    positioned along and across that axis. Works for a clean contour or a
    k-means-split cluster of a merged/touching-animals blob alike
    (split=True lowers confidence, since a split cluster boundary is less
    certain than a clean silhouette). Returns None if there aren't enough
    points to fit a principal axis."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or len(pts) < 5:
        return None

    mean, eigenvectors = cv2.PCACompute(pts, mean=np.array([centroid], dtype=np.float64))
    axis_dir = (float(eigenvectors[0][0]), float(eigenvectors[0][1]))
    perp_dir = _perp(axis_dir)

    rel = pts - np.array(centroid)
    proj_major = rel @ np.array(axis_dir)
    proj_minor = rel @ np.array(perp_dir)

    end_a = tuple(pts[int(np.argmax(proj_major))])
    end_b = tuple(pts[int(np.argmin(proj_major))])
    minor_half_width = float((proj_minor.max() - proj_minor.min()) / 2.0)
    minor_half_width = max(minor_half_width, 4.0)

    nose_pt, tail_pt = _disambiguate_ends(end_a, end_b, centroid, pts, proj_major, proj_minor, prev_pose, max_jump_px)

    conf = float(np.clip(base_conf, 0.0, 1.0)) * (0.8 if split else 1.0)

    neck = _lerp(nose_pt, centroid, 0.42)
    ear_anchor = _lerp(nose_pt, neck, 0.55)
    ear_w = minor_half_width * 0.75
    left_ear = (ear_anchor[0] + perp_dir[0] * ear_w, ear_anchor[1] + perp_dir[1] * ear_w)
    right_ear = (ear_anchor[0] - perp_dir[0] * ear_w, ear_anchor[1] - perp_dir[1] * ear_w)

    forepaw_anchor = _lerp(neck, centroid, 0.55)
    paw_w = minor_half_width * 0.95
    left_forepaw = (forepaw_anchor[0] + perp_dir[0] * paw_w, forepaw_anchor[1] + perp_dir[1] * paw_w)
    right_forepaw = (forepaw_anchor[0] - perp_dir[0] * paw_w, forepaw_anchor[1] - perp_dir[1] * paw_w)

    hind_anchor = _lerp(centroid, tail_pt, 0.45)
    hind_w = minor_half_width * 0.9
    left_hindlimb = (hind_anchor[0] + perp_dir[0] * hind_w, hind_anchor[1] + perp_dir[1] * hind_w)
    right_hindlimb = (hind_anchor[0] - perp_dir[0] * hind_w, hind_anchor[1] - perp_dir[1] * hind_w)

    def kp(pt, scale):
        return Keypoint(pt[0], pt[1], conf * scale)

    keypoints = {
        "nose": kp(nose_pt, 1.0),
        "left_ear": kp(left_ear, 0.6),
        "right_ear": kp(right_ear, 0.6),
        "neck": kp(neck, 0.9),
        "body_center": kp(centroid, 1.0),
        "left_forepaw": kp(left_forepaw, 0.6),
        "right_forepaw": kp(right_forepaw, 0.6),
        "tail_base": kp(tail_pt, 0.9),
        "left_hindlimb": kp(left_hindlimb, 0.6),
        "right_hindlimb": kp(right_hindlimb, 0.6),
    }
    return PoseFrame(frame_idx, time, keypoints, track_quality=conf)


def _hold_last(prev_pose, frame_idx, t):
    """No usable detection this frame: hold the prior pose with decaying
    confidence rather than snapping to nothing, so a brief occlusion/miss
    doesn't immediately break tracking continuity -- but a sustained miss
    quickly decays track_quality below the classification gate."""
    if prev_pose is None:
        return PoseFrame(frame_idx, t, {}, track_quality=0.0)
    decay = 0.5
    pose = PoseFrame(frame_idx, t, {}, track_quality=max(0.0, prev_pose.track_quality * decay))
    for part, prev_kp in prev_pose.keypoints.items():
        pose.keypoints[part] = Keypoint(prev_kp.x, prev_kp.y, prev_kp.confidence * decay)
    return pose


def _draw_skeleton(img, pose, color):
    for a, b in _SKELETON_EDGES:
        pa, pb = pose.xy(a), pose.xy(b)
        if pa and pb:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 1, cv2.LINE_AA)
    for kp in pose.keypoints.values():
        cv2.circle(img, (int(kp.x), int(kp.y)), 2, color, -1)


# --------------------------------------------------------------------------
# STAGE 1a: per-animal blob detection + splitting + identity matching
# (same three-stage idea as two_mouse.py, extended to also carry the raw
# point cloud each animal needs for pose derivation)
# --------------------------------------------------------------------------

def _split_blob(contour, mask_shape, k):
    blob_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(blob_mask == 255)
    pts_full = np.column_stack([xs, ys]).astype(np.float32)

    if len(pts_full) < k:
        cx, cy = (pts_full.mean(axis=0) if len(pts_full) else (np.nan, np.nan))
        return [((cx, cy), len(pts_full) / k, pts_full, None, True)] * k

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pts_full, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    out = []
    for i in range(k):
        cluster_pts = pts_full[labels.flatten() == i]
        out.append((tuple(centers[i]), float(len(cluster_pts)), cluster_pts, None, True))
    return out


def _detect_animals(mask, min_area, max_area, num_animals, motion_mask=None, debris_motion_fraction=0.02):
    """Returns a list of up to num_animals tuples
    (centroid, area, points, contour_or_None, was_split), best-effort, or
    None if nothing was detected at all.

    motion_mask (optional, from a MotionHistory instance) rejects static
    debris -- e.g. droppings -- before ranking blobs by area; see
    two_mouse.reject_static_debris for why this matters (without it, a
    dropped pellet that outsizes the animal at some point permanently
    hijacks that identity's tracked position instead of the real mouse)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = reject_static_debris(contours, mask.shape, motion_mask, num_animals, debris_motion_fraction)
    contours.sort(key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        return None

    if len(contours) == 1:
        area = cv2.contourArea(contours[0])
        if area > max_area:
            return _split_blob(contours[0], mask.shape, num_animals)
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return None
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        pts = contours[0].reshape(-1, 2).astype(np.float64)
        return [(centroid, area, pts, contours[0], False)]

    out = []
    for c in contours[:num_animals]:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        centroid = (M["m10"] / M["m00"], M["m01"] / M["m00"])
        pts = c.reshape(-1, 2).astype(np.float64)
        out.append((centroid, cv2.contourArea(c), pts, c, False))
    return out if out else None


def _match_identities(prev_items, curr_items):
    """Hungarian matching on centroid position, carrying the rest of each
    tuple along with whichever centroid gets assigned to which identity."""
    prev_pos = [p[0] for p in prev_items]
    curr_pos = [c[0] for c in curr_items]
    cost = np.zeros((len(prev_pos), len(curr_pos)))
    for i, p in enumerate(prev_pos):
        for j, c in enumerate(curr_pos):
            cost[i, j] = math.hypot(p[0] - c[0], p[1] - c[1])
    row_ind, col_ind = linear_sum_assignment(cost)
    ordered = list(prev_items)
    for r, c in zip(row_ind, col_ind):
        ordered[r] = curr_items[c]
    return ordered


def _local_motion_energy(gray_prev, gray_curr, center, roi_size):
    """Mean absolute pixel change in a roi_size x roi_size window centered
    on `center`, between two consecutive grayscale frames. High while an
    animal moves limbs/head in place; near zero while truly still; also
    high during locomotion (but locomotion is already identified via
    body-translation velocity, so that's not a source of confusion for the
    grooming score, which additionally requires the body to be still)."""
    if gray_prev is None or center is None:
        return 0.0
    x, y = center
    if x is None or y is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    half = roi_size // 2
    h, w = gray_curr.shape
    x0, x1 = max(0, int(x - half)), min(w, int(x + half))
    y0, y1 = max(0, int(y - half)), min(h, int(y + half))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    prev_patch = gray_prev[y0:y1, x0:x1].astype(np.float32)
    curr_patch = gray_curr[y0:y1, x0:x1].astype(np.float32)
    return float(np.abs(curr_patch - prev_patch).mean())


def _load_background(video_path, background_source, n_background_samples, color_mode="gray"):
    """Background estimation is only reliable if the animal moves around
    enough during sampling that no pixel is 'mouse' most of the time. That
    assumption breaks for low-mobility footage -- e.g. a session with long
    grooming/rearing/freezing bouts. If you have a short clip or photo of
    the EMPTY arena (recommended: grab a frame before the animal is placed
    in), pass it as background_source for a much more reliable background
    than auto-sampling from the animal's own video."""
    if background_source is None:
        cap = cv2.VideoCapture(str(video_path))
        bg = build_background(cap, n_background_samples, color_mode=color_mode)
        cap.release()
        return bg

    ext = str(background_source).lower()
    if ext.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
        img = cv2.imread(str(background_source))
        if img is None:
            raise RuntimeError(f"Could not read background image: {background_source}")
        return img if color_mode == "rgb" else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cap = cv2.VideoCapture(str(background_source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open background source video: {background_source}")
    bg = build_background(cap, n_background_samples, color_mode=color_mode)
    cap.release()
    return bg


# --------------------------------------------------------------------------
# STAGE 1b: geometric features derived from the pose (velocities, distances,
# elevation/stability proxies, repetitive-motion score)
# --------------------------------------------------------------------------

@dataclass
class FeatureSet:
    frame_idx: int = 0
    time: float = 0.0

    nose_velocity: float = 0.0
    head_velocity: float = 0.0          # neck
    body_velocity: float = 0.0
    left_forepaw_velocity: float = 0.0
    right_forepaw_velocity: float = 0.0
    hindlimb_velocity: float = 0.0

    paw_to_nose_distance: float = 0.0
    paw_to_head_distance: float = 0.0
    head_to_body_distance: float = 0.0

    body_angle_deg: float = 0.0
    elongation_ratio: float = 0.0
    posture_change_rate: float = 0.0

    body_movement: float = 0.0          # windowed average body_velocity
    forelimb_movement: float = 0.0
    hindlimb_movement: float = 0.0
    relative_forepaw_movement: float = 0.0  # paw movement independent of body translation

    head_motion_energy: float = 0.0     # mean abs pixel diff in a window around the head (grooming cue)
    repetitive_score: float = 0.0       # 0..1, periodicity of head_motion_energy (grooming cue)

    vertical_movement: float = 0.0      # px/s, +up
    head_elevation: float = 0.0         # 0..1, relative to the animal's own resting baseline
    forelimb_elevation: float = 0.0
    hindlimb_stability: float = 0.0

    track_quality: float = 0.0

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


class FeatureExtractor:
    HINDLIMB_STABILITY_SCALE = 60.0
    REPETITIVE_RATE_SCALE_HZ = 6.0
    ELEVATION_LENGTH_FRACTION = 0.6
    REPETITIVE_MIN_AMPLITUDE = 1.5
    REPETITIVE_FULL_AMPLITUDE = 8.0

    def __init__(self, cfg):
        self.min_conf = cfg["pose_min_confidence"]
        self.window = max(4, int(cfg["history_window_frames"]))
        self.px_per_cm = cfg.get("pixels_per_cm", 0.0) or 0.0

        self._prev_pose = None
        self._prev_feature = None
        self._history = deque(maxlen=self.window)
        self._neck_baseline_y = None
        self._paw_baseline_y = None

    @staticmethod
    def _update_baseline(baseline, raw_y, grounding_alpha=0.2, elevating_alpha=0.0015):
        if baseline is None:
            return raw_y
        alpha = grounding_alpha if raw_y >= baseline else elevating_alpha
        return baseline * (1 - alpha) + raw_y * alpha

    def _unit(self, value):
        return value / self.px_per_cm if self.px_per_cm > 0 else value

    def _pt(self, pose, part):
        if pose is None:
            return None
        kp = pose.get(part)
        if kp is None or kp.confidence < self.min_conf:
            return None
        return kp.xy()

    def _velocity(self, pose, part, dt):
        cur = self._pt(pose, part)
        prev = self._pt(self._prev_pose, part) if self._prev_pose else None
        if cur is None or prev is None or dt <= 0:
            return 0.0, (0.0, 0.0)
        vec = ((cur[0] - prev[0]) / dt, (cur[1] - prev[1]) / dt)
        speed = float(math.hypot(*vec))
        return self._unit(speed), vec

    def update(self, pose, head_motion_energy=0.0):
        dt = (pose.time - self._prev_pose.time) if self._prev_pose else (1.0 / 30.0)
        dt = dt if dt > 1e-4 else (1.0 / 30.0)

        fs = FeatureSet(frame_idx=pose.frame_idx, time=pose.time, track_quality=pose.track_quality)
        fs.head_motion_energy = float(head_motion_energy)

        fs.nose_velocity, _ = self._velocity(pose, "nose", dt)
        fs.head_velocity, _ = self._velocity(pose, "neck", dt)
        fs.body_velocity, body_vec = self._velocity(pose, "body_center", dt)
        fs.left_forepaw_velocity, lpaw_vec = self._velocity(pose, "left_forepaw", dt)
        fs.right_forepaw_velocity, rpaw_vec = self._velocity(pose, "right_forepaw", dt)
        lh_v, _ = self._velocity(pose, "left_hindlimb", dt)
        rh_v, _ = self._velocity(pose, "right_hindlimb", dt)
        fs.hindlimb_velocity = (lh_v + rh_v) / 2.0
        fs._body_vel_vec = body_vec
        fs._left_paw_vel_vec = lpaw_vec
        fs._right_paw_vel_vec = rpaw_vec

        nose = self._pt(pose, "nose")
        neck = self._pt(pose, "neck")
        body_c = self._pt(pose, "body_center")
        tail = self._pt(pose, "tail_base")
        lpaw = self._pt(pose, "left_forepaw")
        rpaw = self._pt(pose, "right_forepaw")
        lhind = self._pt(pose, "left_hindlimb")
        rhind = self._pt(pose, "right_hindlimb")

        paw_nose = [_distance(p, nose) for p in (lpaw, rpaw) if p and nose]
        fs.paw_to_nose_distance = self._unit(min(paw_nose)) if paw_nose else (
            self._prev_feature.paw_to_nose_distance if self._prev_feature else 0.0)

        paw_head = [_distance(p, neck) for p in (lpaw, rpaw) if p and neck]
        fs.paw_to_head_distance = self._unit(min(paw_head)) if paw_head else (
            self._prev_feature.paw_to_head_distance if self._prev_feature else 0.0)

        fs.head_to_body_distance = self._unit(_distance(neck, body_c)) if (neck and body_c) else (
            self._prev_feature.head_to_body_distance if self._prev_feature else 0.0)

        if tail and neck:
            fs.body_angle_deg = _angle_deg(tail, neck)
        elif self._prev_feature:
            fs.body_angle_deg = self._prev_feature.body_angle_deg

        length_proxy = _distance(nose, tail) if (nose and tail) else None
        widths = [d for d in [
            _distance(lpaw, rpaw) if lpaw and rpaw else None,
            _distance(lhind, rhind) if lhind and rhind else None,
        ] if d is not None]
        width_proxy = float(np.mean(widths)) if widths else None
        if width_proxy and length_proxy and length_proxy > 1e-6:
            fs.elongation_ratio = float(width_proxy / length_proxy)
        elif self._prev_feature:
            fs.elongation_ratio = self._prev_feature.elongation_ratio

        if self._prev_feature is not None:
            dtheta = abs(_angle_diff_deg(fs.body_angle_deg, self._prev_feature.body_angle_deg))
            fs.posture_change_rate = dtheta / dt

        length_ref = max(length_proxy or 0.0, 1.0)
        if neck:
            self._neck_baseline_y = self._update_baseline(self._neck_baseline_y, neck[1])
            elevation = (self._neck_baseline_y - neck[1]) / length_ref
            fs.head_elevation = _norm01(elevation, 0.0, self.ELEVATION_LENGTH_FRACTION)
            prev_neck = self._pt(self._prev_pose, "neck") if self._prev_pose else None
            fs.vertical_movement = self._unit(-((neck[1] - (prev_neck[1] if prev_neck else neck[1])) / dt))
        if lpaw or rpaw:
            cur_py = float(np.mean([p[1] for p in (lpaw, rpaw) if p]))
            self._paw_baseline_y = self._update_baseline(self._paw_baseline_y, cur_py)
            elevation_p = (self._paw_baseline_y - cur_py) / length_ref
            fs.forelimb_elevation = _norm01(elevation_p, 0.0, self.ELEVATION_LENGTH_FRACTION)

        self._history.append(fs)
        window = list(self._history)
        fs.body_movement = self._unit(float(np.mean([f.body_velocity for f in window])))
        fs.forelimb_movement = float(np.mean(
            [(f.left_forepaw_velocity + f.right_forepaw_velocity) / 2.0 for f in window]))
        fs.hindlimb_movement = float(np.mean([f.hindlimb_velocity for f in window]))
        fs.hindlimb_stability = 1.0 - _norm01(fs.hindlimb_movement, 0.0, self.HINDLIMB_STABILITY_SCALE)

        rel = []
        for f in window:
            bv = getattr(f, "_body_vel_vec", (0.0, 0.0))
            lv = getattr(f, "_left_paw_vel_vec", (0.0, 0.0))
            rv = getattr(f, "_right_paw_vel_vec", (0.0, 0.0))
            rel.append(math.hypot(lv[0] - bv[0], lv[1] - bv[1]))
            rel.append(math.hypot(rv[0] - bv[0], rv[1] - bv[1]))
        fs.relative_forepaw_movement = self._unit(float(np.mean(rel))) if rel else 0.0

        fs.repetitive_score = self._repetitive_score(window, dt)

        self._prev_pose = pose
        self._prev_feature = fs
        return fs

    def _repetitive_score(self, window, dt):
        # Periodicity of the local head/paw motion-energy signal -- a real
        # grooming stroke pulses this value up and down at ~4-8 Hz, which
        # shows up as repeated crossings of its own windowed mean (a pulse
        # train crosses its mean just as reliably as a clean sine wave
        # would), whereas steady stillness or steady locomotion doesn't.
        series = np.array([f.head_motion_energy for f in window], dtype=float)
        if len(series) < 4 or np.allclose(series, series[0]):
            return 0.0
        detrended = series - series.mean()
        amplitude = float(detrended.std())
        amp_c = _norm01(amplitude, self.REPETITIVE_MIN_AMPLITUDE, self.REPETITIVE_FULL_AMPLITUDE)
        if amp_c <= 0.0:
            return 0.0
        signs = np.sign(detrended)
        signs[signs == 0] = 1
        crossings = int(np.sum(signs[1:] != signs[:-1]))
        duration = max((len(series) - 1) * dt, 1e-6)
        rate_hz = crossings / duration
        rate_c = _norm01(rate_hz, 0.0, self.REPETITIVE_RATE_SCALE_HZ)
        return rate_c * amp_c


_FS_FIELDS = [f.name for f in FeatureSet.__dataclass_fields__.values() if f.name not in ("frame_idx", "time")]


def _row_to_feature_set(row):
    kwargs = {}
    for f in _FS_FIELDS:
        v = row.get(f, 0.0)
        kwargs[f] = float(v) if pd.notna(v) else 0.0
    return FeatureSet(frame_idx=int(row.get("frame", 0)), time=float(row.get("time_s", 0.0)), **kwargs)


# --------------------------------------------------------------------------
# STAGE 2: explainable, weighted-cue scorers (never a single feature alone)
# --------------------------------------------------------------------------

def _score_grooming(fs, cfg):
    t_body_low = cfg["groom_body_move_low"]
    t_motion_high = cfg["groom_head_motion_high"]
    t_rep = cfg["groom_repetitive_high"]
    t_wobble = cfg["groom_posture_wobble_high"]

    stillness = 1.0 - _norm01(fs.body_movement, 0.0, t_body_low * 2.5)
    motion_activity = _norm01(fs.head_motion_energy, t_motion_high * 0.3, t_motion_high)
    repetitive = _norm01(fs.repetitive_score, t_rep * 0.4, t_rep * 1.3)
    wobble = _norm01(fs.posture_change_rate, t_wobble * 0.2, t_wobble)

    # No single cue alone (see module docstring on why paw *keypoints*
    # can't be trusted independently on this backend): a still body that's
    # also producing local head-region pixel motion, some periodicity to
    # that motion, and small body-axis wobble together are a much better
    # proxy than any one of those alone.
    score = 0.30 * stillness + 0.30 * motion_activity + 0.20 * repetitive + 0.20 * wobble
    # Hard gate: the body must actually be near-stationary on its own, so
    # real locomotion can't be scored as grooming just because moving
    # through the arena also produces local pixel motion and body-axis
    # wobble (both of those are large during genuine locomotion too, and
    # are not by themselves evidence of in-place grooming).
    if fs.body_movement > t_body_low * 2.0:
        score *= 0.15
    explain = {
        "body_movement": _bucket(fs.body_movement, t_body_low, t_body_low * 2),
        "head_motion": _bucket(fs.head_motion_energy, t_motion_high * 0.5, t_motion_high),
        "repetitive": _bucket(fs.repetitive_score, t_rep * 0.5, t_rep),
        "posture_wobble": _bucket(fs.posture_change_rate, t_wobble * 0.3, t_wobble),
    }
    return float(score), explain


def _score_rearing(fs, cfg):
    t_elev = cfg["rear_upright_elevation_high"]
    t_fl = cfg["rear_forelimb_elevated"]
    t_stab = cfg["rear_hindlimb_stability_high"]
    t_vert = cfg["rear_vertical_move_min"]

    vertical = _norm01(fs.vertical_movement, t_vert * 0.3, t_vert * 2)
    # Elevation/stability are weighted most heavily since they hold up for
    # the whole duration of a held rear; vertical *velocity* is naturally
    # near zero once the animal has settled upright, so it's kept as a
    # smaller onset bonus rather than something a sustained rear must keep
    # re-earning every frame.
    score = (0.40 * fs.head_elevation + 0.30 * fs.forelimb_elevation
             + 0.20 * fs.hindlimb_stability + 0.10 * vertical)
    # Hard gate: elevation must clear the "upright" bar on its own, so a
    # brief bounce or locomotion stride can't be scored as rearing purely
    # by combining with the other three softer cues.
    if fs.head_elevation < t_elev * 0.6:
        score *= 0.5

    explain = {
        "head_elevation": _bucket(fs.head_elevation, t_elev * 0.5, t_elev),
        "forelimb_elevation": _bucket(fs.forelimb_elevation, t_fl * 0.5, t_fl),
        "hindlimb_stability": _bucket(fs.hindlimb_stability, t_stab * 0.6, t_stab),
    }
    return float(score), explain


def _score_locomotion_resting(fs, cfg):
    t_high = cfg["loco_body_move_high"]
    loco = _norm01(fs.body_movement, t_high * 0.4, t_high)
    rest = 1.0 - _norm01(fs.body_movement, 0.0, t_high * 0.5)
    explain = {"body_movement": _bucket(fs.body_movement, t_high * 0.3, t_high)}
    return float(loco), float(rest), explain


class BehaviorClassifier:
    """Combines the weighted scorers into one confirmed behavior label per
    frame, with hysteresis (temporal persistence) so a single noisy frame
    can't flip the label, and a pose-quality gate so an unreliable frame is
    marked 'undetermined' rather than guessed. One instance per animal --
    holds that animal's persistence state across the video."""

    MIN_POSE_QUALITY = 0.15
    DEFAULT_PERSISTENCE = 2

    def __init__(self, cfg):
        self.cfg = cfg
        self._pending_label = None
        self._pending_count = 0
        self._confirmed_label = "UNKNOWN"
        self._confirmed_since_time = None

    def _persistence_for(self, label):
        if label == "GROOMING":
            return max(1, int(self.cfg["groom_min_persistence"]))
        if label == "REARING":
            return max(1, int(self.cfg["rear_min_persistence"]))
        return self.DEFAULT_PERSISTENCE

    def classify(self, fs):
        cfg = self.cfg
        quality_flag = fs.track_quality < self.MIN_POSE_QUALITY

        groom_score, groom_explain = _score_grooming(fs, cfg)
        rear_score, rear_explain = _score_rearing(fs, cfg)
        loco_score, rest_score, lr_explain = _score_locomotion_resting(fs, cfg)
        scores = {"GROOMING": groom_score, "REARING": rear_score,
                  "LOCOMOTION": loco_score, "RESTING": rest_score}

        candidate = "UNKNOWN"
        if not quality_flag:
            if groom_score >= cfg["groom_score_threshold"] and groom_score >= rear_score:
                candidate = "GROOMING"
            elif rear_score >= cfg["rear_score_threshold"]:
                candidate = "REARING"
            elif loco_score > rest_score and loco_score > 0.4:
                candidate = "LOCOMOTION"
            else:
                candidate = "RESTING"

        if candidate == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = candidate
            self._pending_count = 1

        if self._pending_count >= self._persistence_for(candidate):
            if self._confirmed_label != candidate:
                self._confirmed_label = candidate
                self._confirmed_since_time = fs.time

        confirmed = self._confirmed_label
        if self._confirmed_since_time is None:
            self._confirmed_since_time = fs.time

        if confirmed == "UNKNOWN":
            confidence = float(np.clip((1.0 - fs.track_quality) * 100.0, 0.0, 100.0))
            reason = "Pose tracking quality too low this frame"
        else:
            confidence = float(np.clip(scores.get(confirmed, 0.0) * 100.0, 0.0, 100.0))
            explain = {"GROOMING": groom_explain, "REARING": rear_explain,
                       "LOCOMOTION": lr_explain, "RESTING": lr_explain}.get(confirmed, {})
            reason = "; ".join(f"{k}={v}" for k, v in explain.items())

        return dict(behavior=confirmed, confidence=confidence, quality_flag=quality_flag, reason=reason)


# --------------------------------------------------------------------------
# save_preview_frames -- reference-frame gallery with skeleton overlay
# --------------------------------------------------------------------------

def save_preview_frames(video_path, output_dir, n_samples=6, background_source=None, **overrides):
    """Save a handful of sample frames evenly spaced across the video, each
    annotated with the detected animal(s) and derived pseudo-skeleton at
    that frame -- lets you sanity-check detection/area settings before
    running the full feature extraction."""
    cfg = {**DEFAULTS, **overrides}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_animals = cfg["num_animals"]
    cap.release()

    background = _load_background(video_path, background_source, cfg["n_background_samples"],
                                   color_mode=cfg["color_mode"])

    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(preview_dir, "preview_*.png")):
        os.remove(stale)

    n_samples = max(1, int(n_samples))
    positions = np.linspace(0, max(0, n_frames - 1), n_samples).astype(int)
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]  # BGR
    saved_paths = []

    cap = cv2.VideoCapture(str(video_path))
    for i, frame_idx in enumerate(positions, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_for_diff = frame if cfg["color_mode"] == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(frame_for_diff, background, cfg["diff_threshold"], cfg["morph_kernel"],
                                color_mode=cfg["color_mode"])
        detected = _detect_animals(mask, cfg["min_area"], cfg["max_area"], n_animals)

        disp = frame.copy()
        if detected:
            for j, (centroid, area, pts, contour, split) in enumerate(detected[:n_animals]):
                x, y = centroid
                if isinstance(x, float) and np.isnan(x):
                    continue
                color = colors[j % len(colors)]
                if pts is not None and len(pts) >= 5:
                    solidity = _solidity_from_contour(contour) if contour is not None else _solidity_from_points(pts)
                    pose = _pose_from_points(pts, centroid, frame_idx, frame_idx, None,
                                              cfg["max_jump_px"], solidity, split=split)
                    if pose is not None:
                        _draw_skeleton(disp, pose, color)
                pt = (int(x), int(y))
                cv2.circle(disp, pt, 4, color, -1)
                cv2.putText(disp, f"{chr(65 + j)} (area={area:.0f})", (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        else:
            cv2.putText(disp, "NO DETECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(disp, f"frame {frame_idx}", (16, disp.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        path = os.path.join(preview_dir, f"preview_{i:02d}_frame{frame_idx}.png")
        cv2.imwrite(path, disp)
        saved_paths.append(path)

    cap.release()
    print(f"Saved {len(saved_paths)} reference frame(s) -> {preview_dir}")
    return saved_paths


# --------------------------------------------------------------------------
# STAGE 1: EXTRACT (video -> per-frame pseudo-pose features CSV)
# --------------------------------------------------------------------------

def extract_features(video_path, output_csv, background_source=None, progress_callback=None,
                      show_display=False, **overrides):
    cfg = {**DEFAULTS, **overrides}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_animals = cfg["num_animals"]

    print("Building background model...")
    background = _load_background(video_path, background_source, cfg["n_background_samples"],
                                   color_mode=cfg["color_mode"])

    smoothers_alpha = cfg["pose_smoothing_alpha"]
    smoothers_min_conf = cfg["pose_min_confidence"]
    extractors = [FeatureExtractor(cfg) for _ in range(n_animals)]
    prev_poses = [None] * n_animals
    smoother_state = [dict() for _ in range(n_animals)]  # part -> Keypoint, confidence-gated EMA

    def smooth(i, raw_pose):
        state = smoother_state[i]
        out = PoseFrame(raw_pose.frame_idx, raw_pose.time, {}, raw_pose.track_quality)
        for part in BODY_PARTS:
            raw_kp = raw_pose.keypoints.get(part)
            prev = state.get(part)
            if raw_kp is None:
                if prev is not None:
                    out.keypoints[part] = prev
                continue
            if prev is None:
                new_kp = Keypoint(raw_kp.x, raw_kp.y, raw_kp.confidence)
            elif raw_kp.confidence < smoothers_min_conf:
                new_kp = Keypoint(prev.x, prev.y, prev.confidence * 0.85 + raw_kp.confidence * 0.15)
            else:
                a = smoothers_alpha * raw_kp.confidence
                new_kp = Keypoint(prev.x * (1 - a) + raw_kp.x * a, prev.y * (1 - a) + raw_kp.y * a,
                                   prev.confidence * (1 - smoothers_alpha) + raw_kp.confidence * smoothers_alpha)
            state[part] = new_kp
            out.keypoints[part] = new_kp
        return out

    records = []
    prev_items = None
    prev_gray = None
    stopped_early = False
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]  # BGR
    roi_size = int(cfg["groom_roi_size"])

    motion_history = None
    if cfg["reject_static_debris"]:
        motion_history = MotionHistory(fps * cfg["debris_motion_window_s"],
                                        pixel_threshold=cfg["debris_motion_pixel_threshold"])

    frame_iter = range(n_frames) if (show_display or progress_callback) \
        else tqdm(range(n_frames), desc="Extracting features")

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_for_diff = frame if cfg["color_mode"] == "rgb" else gray
        mask = foreground_mask(frame_for_diff, background, cfg["diff_threshold"], cfg["morph_kernel"],
                                color_mode=cfg["color_mode"])

        if motion_history is not None:
            motion_history.update(gray)

        detected = _detect_animals(mask, cfg["min_area"], cfg["max_area"], n_animals,
                                    motion_mask=(motion_history.mask if motion_history else None),
                                    debris_motion_fraction=cfg["debris_motion_fraction"])

        if detected is None:
            curr = prev_items if prev_items else [
                ((np.nan, np.nan), np.nan, np.zeros((0, 2)), None, False)] * n_animals
            status = "lost"
        elif len(detected) < n_animals:
            curr = (detected + [detected[0]])[:n_animals] if prev_items is None \
                else _match_identities(prev_items, detected)
            status = "partial"
        else:
            if prev_items is None:
                detected = sorted(detected, key=lambda item: item[0][0])  # seed left-to-right
                curr = detected[:n_animals]
            else:
                curr = _match_identities(prev_items, detected)
            status = "ok"
        prev_items = curr

        t = frame_idx / fps
        for i, (centroid, area, pts, contour, split) in enumerate(curr):
            x, y = centroid
            raw_pose = None
            if pts is not None and len(pts) >= 5 and not (isinstance(x, float) and np.isnan(x)):
                solidity = _solidity_from_contour(contour) if contour is not None else _solidity_from_points(pts)
                raw_pose = _pose_from_points(pts, centroid, frame_idx, t, prev_poses[i],
                                              cfg["max_jump_px"], solidity, split=split)
            if raw_pose is None:
                raw_pose = _hold_last(prev_poses[i], frame_idx, t)

            smoothed = smooth(i, raw_pose)
            prev_poses[i] = smoothed

            # Real (non-template) grooming signal: local pixel motion in a
            # window around the head, between this frame and the last --
            # see the note on groom_head_motion_high in DEFAULTS.
            head_center = smoothed.xy("nose") or smoothed.xy("body_center") or (x, y)
            head_motion = _local_motion_energy(prev_gray, gray, head_center, roi_size)

            fs = extractors[i].update(smoothed, head_motion_energy=head_motion)

            rec = dict(frame=frame_idx, time_s=round(t, 4), mouse_id=f"mouse_{chr(65 + i)}",
                       x=x, y=y, area=area, status=status)
            rec.update(fs.as_dict())
            records.append(rec)

        prev_gray = gray

        if show_display:
            annotated = frame.copy()
            for i, (centroid, _area, pts, contour, split) in enumerate(curr):
                x, y = centroid
                if not (isinstance(x, float) and np.isnan(x)):
                    pt = (int(x), int(y))
                    cv2.circle(annotated, pt, 6, colors[i % len(colors)], -1)
                    cv2.putText(annotated, chr(65 + i), (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)
                    if prev_poses[i] is not None:
                        _draw_skeleton(annotated, prev_poses[i], colors[i % len(colors)])
            cv2.putText(annotated, f"Frame: {frame_idx}/{n_frames}  (extracting features)", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.imshow("BEHAVIOR FEATURE EXTRACTION - Q to stop", annotated)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                stopped_early = True

        if progress_callback and (frame_idx % 5 == 0 or frame_idx == n_frames - 1):
            progress_callback(min(1.0, (frame_idx + 1) / max(1, n_frames)))

        if stopped_early:
            break

    cap.release()
    if show_display:
        cv2.destroyAllWindows()
    if progress_callback:
        progress_callback(1.0)

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows -> {output_csv}")
    return df


# --------------------------------------------------------------------------
# DESCRIBE (feature distributions, to help pick thresholds)
# --------------------------------------------------------------------------

def describe_features(features_csv, **overrides):
    df = pd.read_csv(features_csv)
    for mouse_id, sub in df.groupby("mouse_id"):
        print(f"\n=== {mouse_id} ===")
        cols = [c for c in ["body_movement", "head_elevation", "forelimb_elevation",
                             "hindlimb_stability", "paw_to_nose_distance", "repetitive_score",
                             "track_quality"] if c in sub.columns]
        print(sub[cols].describe().round(3))
    print(
        "\nPick groom_score_threshold / rear_score_threshold a bit above where the "
        "weighted score sits for frames you know are NOT that behavior, and "
        "loco_body_move_high a bit below 'body_movement' for genuine locomotion frames."
    )


# --------------------------------------------------------------------------
# STAGE 2: CLASSIFY (features -> per-frame confirmed labels -> bouts)
# --------------------------------------------------------------------------

def _group_runs(sub):
    labels = sub["label"].tolist()
    times = sub["time_s"].tolist()
    frames = sub["frame"].tolist()
    confs = sub["confidence"].tolist() if "confidence" in sub else [0.0] * len(labels)
    runs = []
    for lab, t, fr, c in zip(labels, times, frames, confs):
        if runs and runs[-1]["behavior"] == lab:
            runs[-1]["stop_s"] = t
            runs[-1]["stop_frame"] = fr
            runs[-1]["confidences"].append(c)
        else:
            runs.append(dict(behavior=lab, start_s=t, stop_s=t, start_frame=fr, stop_frame=fr,
                              confidences=[c]))
    return runs


def _merge_short_gaps(runs, merge_gap_s):
    """Bridge a short differing-behavior run sandwiched between two runs of
    the SAME behavior (a brief pause mid-grooming, etc.) into one run. Only
    the sandwiched run's duration (or the gap it creates) is checked
    against the threshold -- a genuinely long, distinct bout in between is
    never absorbed just because it sits next to a matching behavior."""
    runs = list(runs)
    changed = True
    while changed and len(runs) >= 3:
        changed = False
        for i in range(len(runs) - 2):
            a, b, c = runs[i], runs[i + 1], runs[i + 2]
            b_dur = b["stop_s"] - b["start_s"]
            gap = c["start_s"] - a["stop_s"]
            if a["behavior"] == c["behavior"] and min(b_dur, gap) <= merge_gap_s:
                merged = dict(behavior=a["behavior"], start_s=a["start_s"], start_frame=a["start_frame"],
                              stop_s=c["stop_s"], stop_frame=c["stop_frame"],
                              confidences=a["confidences"] + c["confidences"])
                runs = runs[:i] + [merged] + runs[i + 3:]
                changed = True
                break
    return runs


def labels_to_bouts(sub, mouse_id, merge_gap_s=0.5, min_bout_s=0.3):
    if len(sub) == 0:
        return []
    raw_runs = _group_runs(sub)
    merged = _merge_short_gaps(raw_runs, merge_gap_s)
    kept = [r for r in merged if (r["stop_s"] - r["start_s"]) >= min_bout_s]

    bouts = []
    for r in kept:
        conf_list = r["confidences"] or [0.0]
        bouts.append(dict(
            subject=mouse_id, behavior=r["behavior"],
            start_frame=r["start_frame"], stop_frame=r["stop_frame"],
            start_s=r["start_s"], stop_s=r["stop_s"],
            duration_s=round(r["stop_s"] - r["start_s"], 3),
            confidence=round(float(np.mean(conf_list)), 1),
        ))
    return bouts


def classify_behaviors(features_csv, output_bouts_csv, labeled_output_csv=None, **overrides):
    cfg = {**DEFAULTS, **overrides}
    df = pd.read_csv(features_csv)

    labeled_all, bouts_all = [], []
    for mouse_id, sub in df.groupby("mouse_id"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        classifier = BehaviorClassifier(cfg)

        labels, confidences, quality_flags, reasons = [], [], [], []
        for _, row in sub.iterrows():
            fs = _row_to_feature_set(row)
            result = classifier.classify(fs)
            labels.append(_LABEL_MAP.get(result["behavior"], "undetermined"))
            confidences.append(round(result["confidence"], 1))
            quality_flags.append(result["quality_flag"])
            reasons.append(result["reason"])

        sub["label"] = labels
        sub["confidence"] = confidences
        sub["quality_flag"] = quality_flags
        sub["explain"] = reasons

        labeled_all.append(sub)
        bouts_all.extend(labels_to_bouts(sub, mouse_id, cfg["merge_gap_s"], cfg["min_bout_s"]))

    labeled_df = pd.concat(labeled_all, ignore_index=True) if labeled_all else pd.DataFrame()
    bouts_df = pd.DataFrame(bouts_all)

    if labeled_output_csv:
        labeled_df.to_csv(labeled_output_csv, index=False)
        print(f"Saved per-frame labels -> {labeled_output_csv}")
    bouts_df.to_csv(output_bouts_csv, index=False)
    print(f"Saved {len(bouts_df)} bouts -> {output_bouts_csv}")

    if len(bouts_df):
        summary = bouts_df.groupby(["subject", "behavior"])["duration_s"].agg(["count", "sum", "mean"])
        print("\nSummary (count, total_s, mean_s):")
        print(summary.round(2))

    return labeled_df, bouts_df


def render_annotated(video_path, labeled_df, output_path):
    frame_lookup = {}
    for row in labeled_df.itertuples():
        frame_lookup.setdefault(row.frame, []).append(row)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    colors = {"locomotion": (255, 0, 0), "rearing": (0, 165, 255), "grooming": (0, 255, 0),
              "immobile": (128, 128, 128), "undetermined": (0, 0, 200)}

    for frame_idx in tqdm(range(n_frames), desc="Rendering annotated preview"):
        ok, frame = cap.read()
        if not ok:
            break
        for row in frame_lookup.get(frame_idx, []):
            if np.isnan(row.x):
                continue
            pt = (int(row.x), int(row.y))
            color = colors.get(row.label, (255, 255, 255))
            cv2.circle(frame, pt, 6, color, -1)
            cv2.putText(frame, f"{row.mouse_id}: {row.label}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated preview -> {output_path}")


# --------------------------------------------------------------------------
# Manual (keyboard-driven) behavior scoring
# --------------------------------------------------------------------------
# The automatic classifier above is a proxy -- see the module docstring. For
# whichever behaviors it handles poorly (grooming especially, since a fixed
# pose template can't resolve genuine limb articulation), the reliable
# fallback is what a human would do with BORIS: watch the video and mark
# start/stop of each behavior by hand. manual_score_video() is that tool --
# it produces the exact same bout shape classify_behaviors() does, so the
# results screen and export downstream don't need to know which one ran.

MANUAL_BEHAVIOR_KEYS = [
    (ord("1"), "grooming", (0, 140, 255)),     # orange
    (ord("2"), "rearing", (210, 90, 220)),     # pink/purple
    (ord("3"), "locomotion", (60, 200, 60)),   # green
    (ord("4"), "immobile", (150, 150, 150)),   # gray
]

_MANUAL_BOUT_COLUMNS = ["subject", "behavior", "start_frame", "stop_frame",
                         "start_s", "stop_s", "duration_s", "confidence", "source"]


def manual_score_video(video_path, subject_names=None, num_animals=1,
                        window_name="Manual Behavior Scoring", max_display_width=1280):
    """
    Interactive, keyboard-driven behavior annotation. You watch the video
    and mark bout boundaries yourself, instead of trusting the automatic
    classifier -- the same idea as manual BORIS event-logging.

    Controls
    --------
    SPACE       play / pause (starts paused)
    , / .       step one frame back / forward (while paused)
    [ / ]       slower / faster playback
    1           toggle GROOMING for the active subject
    2           toggle REARING for the active subject
    3           toggle LOCOMOTION for the active subject
    4           toggle IMMOBILE for the active subject
    TAB         switch the active subject (only matters with >1 animal)
    u           undo the last completed bout
    q / ESC     finish and save

    Pressing a behavior key opens a bout for the active subject if none is
    open yet; pressing that SAME key again closes it. Pressing a DIFFERENT
    behavior key while one is open closes the current bout and opens the
    new one -- a subject is only ever doing one of these at a time. Any
    bout still open when you quit is closed at the frame you quit on.

    Returns a DataFrame with the same bout columns classify_behaviors()
    produces (subject, behavior, start_frame, stop_frame, start_s, stop_s,
    duration_s, confidence) plus source="Manual", so it plugs directly into
    the same results screen / export as the automatic engine.
    """
    if subject_names is None:
        subject_names = [f"mouse_{chr(65 + i)}" for i in range(max(1, num_animals))]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    key_to_behavior = {k: name for k, name, _ in MANUAL_BEHAVIOR_KEYS}
    color_by_behavior = {name: color for _, name, color in MANUAL_BEHAVIOR_KEYS}

    open_bouts = {s: None for s in subject_names}  # subject -> dict(behavior, start_frame, start_s) | None
    completed = []
    active_subject = subject_names[0]

    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]
    speed_idx = 2  # 1.0x

    cur_idx = -1
    frame = None
    playing = False

    def time_s(idx):
        return round(idx / fps, 3) if idx is not None and idx >= 0 else 0.0

    def read_next():
        nonlocal cur_idx, frame
        ok, f = cap.read()
        if ok:
            cur_idx += 1
            frame = f
        return ok

    def seek_to(idx):
        nonlocal cur_idx, frame
        idx = max(0, idx if n_frames is None else min(idx, n_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            cur_idx = idx
            frame = f
        return ok

    def close_bout(subject, stop_idx):
        b = open_bouts.get(subject)
        if b is None:
            return
        stop_time = time_s(stop_idx)
        completed.append(dict(
            subject=subject, behavior=b["behavior"],
            start_frame=b["start_frame"], stop_frame=stop_idx,
            start_s=b["start_s"], stop_s=stop_time,
            duration_s=round(stop_time - b["start_s"], 3),
            confidence=100.0, source="Manual",
        ))
        open_bouts[subject] = None

    def open_bout(subject, behavior, start_idx):
        open_bouts[subject] = dict(behavior=behavior, start_frame=start_idx, start_s=time_s(start_idx))

    def toggle(subject, behavior, idx):
        b = open_bouts.get(subject)
        if b is not None and b["behavior"] == behavior:
            close_bout(subject, idx)
        else:
            if b is not None:
                close_bout(subject, idx)
            open_bout(subject, behavior, idx)

    cv2.namedWindow(window_name)
    if not read_next():
        cap.release()
        cv2.destroyWindow(window_name)
        raise RuntimeError("Could not read any frames from the video.")

    print("\nManual behavior scoring -- SPACE=play/pause  ,/.=step  [/]=speed  "
          "1=grooming 2=rearing 3=locomotion 4=immobile (toggle)  TAB=switch subject  "
          "u=undo  q/ESC=finish & save.\n")

    while True:
        disp = frame.copy()
        h, w = disp.shape[:2]
        if w > max_display_width:
            scale = max_display_width / w
            disp = cv2.resize(disp, (max_display_width, int(h * scale)))
        dh, dw = disp.shape[:2]

        bar_h = 30 + 20 * len(subject_names)
        overlay = disp.copy()
        cv2.rectangle(overlay, (0, 0), (dw, bar_h), (0, 0, 0), -1)
        disp = cv2.addWeighted(overlay, 0.55, disp, 0.45, 0)

        status = "PLAYING" if playing else "PAUSED"
        cv2.putText(
            disp, f"{status}  frame {cur_idx}{'/' + str(n_frames) if n_frames else ''}  "
                  f"t={time_s(cur_idx):.2f}s  speed={speeds[speed_idx]:.2f}x",
            (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        for i, subj in enumerate(subject_names):
            b = open_bouts.get(subj)
            text = f"{subj}: " + ((b["behavior"].upper() + " (open)") if b else "--")
            color = color_by_behavior.get(b["behavior"], (255, 255, 255)) if b else (200, 200, 200)
            marker = "> " if subj == active_subject else "  "
            cv2.putText(disp, marker + text, (8, 36 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        legend = "  ".join(f"{chr(k)}={name}" for k, name, _ in MANUAL_BEHAVIOR_KEYS)
        cv2.putText(disp, legend + f"   bouts saved: {len(completed)}", (8, dh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        cv2.imshow(window_name, disp)

        base_delay = max(1, int(1000 / fps))
        delay = max(1, int(base_delay / speeds[speed_idx])) if playing else 30
        key = cv2.waitKey(delay) & 0xFF

        if key == 32:  # SPACE
            playing = not playing
        elif key == ord(","):
            playing = False
            seek_to(cur_idx - 1)
        elif key == ord("."):
            playing = False
            read_next()
        elif key == ord("["):
            speed_idx = max(0, speed_idx - 1)
        elif key == ord("]"):
            speed_idx = min(len(speeds) - 1, speed_idx + 1)
        elif key == 9 and len(subject_names) > 1:  # TAB
            i = subject_names.index(active_subject)
            active_subject = subject_names[(i + 1) % len(subject_names)]
        elif key in key_to_behavior:
            toggle(active_subject, key_to_behavior[key], cur_idx)
        elif key == ord("u"):
            if completed:
                completed.pop()
        elif key in (27, ord("q")):
            break

        if playing:
            if not read_next():
                playing = False  # reached the end -- stay open so trailing bouts can still be closed

    for subj in subject_names:
        if open_bouts.get(subj) is not None:
            close_bout(subj, cur_idx)

    cap.release()
    cv2.destroyWindow(window_name)

    if not completed:
        return pd.DataFrame(columns=_MANUAL_BOUT_COLUMNS)
    out = pd.DataFrame(completed)[_MANUAL_BOUT_COLUMNS]
    return out.sort_values(["start_s", "subject"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Automated grooming/rearing/locomotion bout detection.")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("extract", help="Video -> per-frame feature CSV")
    pe.add_argument("--video", required=True)
    pe.add_argument("--output", default="features.csv")
    pe.add_argument("--num-animals", type=int, default=DEFAULTS["num_animals"])
    pe.add_argument("--min-area", type=int, default=DEFAULTS["min_area"])
    pe.add_argument("--max-area", type=int, default=DEFAULTS["max_area"])
    pe.add_argument("--diff-threshold", type=int, default=DEFAULTS["diff_threshold"])
    pe.add_argument("--bg-samples", type=int, default=DEFAULTS["n_background_samples"])
    pe.add_argument("--background-source", default=None)
    pe.add_argument("--show-display", action="store_true")

    pdesc = sub.add_parser("describe", help="Print feature distributions to help pick thresholds")
    pdesc.add_argument("--features", required=True)

    pc = sub.add_parser("classify", help="Feature CSV -> behavior bouts")
    pc.add_argument("--features", required=True)
    pc.add_argument("--output", default="bouts.csv")
    pc.add_argument("--labeled-output", default=None)
    pc.add_argument("--video", default=None, help="Needed only if --annotate is given")
    pc.add_argument("--annotate", default=None)
    pc.add_argument("--min-bout-s", type=float, default=DEFAULTS["min_bout_s"])
    pc.add_argument("--merge-gap-s", type=float, default=DEFAULTS["merge_gap_s"])
    pc.add_argument("--loco-body-move-high", type=float, default=DEFAULTS["loco_body_move_high"])
    pc.add_argument("--groom-score-threshold", type=float, default=DEFAULTS["groom_score_threshold"])
    pc.add_argument("--rear-score-threshold", type=float, default=DEFAULTS["rear_score_threshold"])

    pm = sub.add_parser("manual", help="Interactive keyboard-driven behavior scoring (BORIS-style)")
    pm.add_argument("--video", required=True)
    pm.add_argument("--output", default="manual_bouts.csv")
    pm.add_argument("--num-animals", type=int, default=1)

    args = p.parse_args()

    if args.command == "extract":
        extract_features(
            args.video, args.output, background_source=args.background_source,
            num_animals=args.num_animals, min_area=args.min_area, max_area=args.max_area,
            diff_threshold=args.diff_threshold, n_background_samples=args.bg_samples,
            show_display=args.show_display,
        )
    elif args.command == "describe":
        describe_features(args.features)
    elif args.command == "classify":
        labeled_df, _ = classify_behaviors(
            args.features, args.output, labeled_output_csv=args.labeled_output,
            min_bout_s=args.min_bout_s, merge_gap_s=args.merge_gap_s,
            loco_body_move_high=args.loco_body_move_high,
            groom_score_threshold=args.groom_score_threshold,
            rear_score_threshold=args.rear_score_threshold,
        )
        if args.video and args.annotate:
            render_annotated(args.video, labeled_df, args.annotate)
    elif args.command == "manual":
        bouts_df = manual_score_video(args.video, num_animals=args.num_animals)
        bouts_df.to_csv(args.output, index=False)
        print(f"Saved {len(bouts_df)} manually-scored bouts -> {args.output}")


if __name__ == "__main__":
    main()
