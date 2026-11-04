"""
two_mouse_tracker.py
---------------------
A free, dependency-light, from-scratch tracker for TWO unmarked mice in one
arena. Built with plain OpenCV + NumPy + SciPy (Hungarian algorithm) --
no paid software, no deep learning, no GPU required.

HOW IT WORKS (three stages, same idea used by classical multi-animal
tracking tools like Tracktor):

  1. DETECT   - build a static background image (median of sampled frames),
                subtract it from each frame, threshold the difference, and
                clean up the resulting mask with morphology. This gives
                blobs = "things that moved" = the mice.

  2. SPLIT    - most of the time this gives you two separate blobs (one per
                mouse). When the mice touch/overlap, OpenCV sees ONE big
                blob. We split that blob into two using k-means clustering
                on its foreground pixel coordinates.

  3. IDENTIFY - blob position alone doesn't tell you WHICH mouse is which
                from frame to frame. We solve that with the Hungarian
                algorithm (scipy.optimize.linear_sum_assignment): match this
                frame's centroids against each mouse's PREDICTED position
                (not just its raw last-seen spot) so that total movement is
                minimized. This is what keeps "mouse_A" referring to the
                same physical animal across the video.

                The prediction comes from a small constant-velocity Kalman
                filter per animal (CentroidKalman, below) -- the same core
                idea AlphaTracker's own PoseFlow/utils_kalman.py uses to
                track multi-animal identity through occlusion (there, per
                pose-keypoint with the `filterpy` package; here, per
                centroid, hand-rolled to keep this file's "no extra heavy
                deps" design). It buys two things over matching against a
                raw last-seen position: (a) during a brief occlusion/merge
                (status="lost"/"partial"), the estimate keeps moving along
                each animal's last known velocity instead of freezing in
                place, so re-identification on reappearance has a far
                better anchor to match against; and (b) the reported
                trail itself comes out smoother, since a Kalman update
                blends the noisy raw detection with the motion model
                rather than reporting it raw.

LIMITATION (be aware of this - it applies to any unmarked tracker, not just
this script): if the two mice are IDENTICAL in appearance, identity can only
be inferred from motion continuity. During a long, static huddle there is no
visual information to tell them apart, so on separation the assignment is a
best guess. Two things fix this almost completely if it matters for your
paradigm: (a) mark one mouse (a small dot of non-toxic animal marker on the
tail base/back) and extend get_two_centroids() to split by color instead of
just position, or (b) accept the small risk and hand-correct swaps by eye
using the annotated preview video (--annotate) -- swaps are easy to spot
because the trail jumps.

USAGE
-----
    python two_mouse_tracker.py --video myvideo.mp4 --output tracks.csv \
        --annotate preview.mp4

Then open tracks.csv (columns: frame, time_s, mouse_id, x, y, status) and
preview.mp4 to check quality, and adjust --min-area / --max-area /
--diff-threshold for your lighting/arena (see TUNING notes below).

TUNING
------
  --diff-threshold  Raise if background noise (flicker, shadows) is being
                     picked up as a blob. Lower if the mice are faint /
                     low-contrast against the floor.
  --min-area         Should be a bit smaller than one mouse's blob area in
                     pixels. Filters out speckle noise.
  --max-area         Should be a bit larger than one mouse's blob area.
                     Anything bigger is treated as two overlapping mice and
                     gets split.
  --bg-samples       More samples = more robust background, but slower to
                     build. 30-60 is plenty for most videos.

Print the area of a normal single-mouse blob once (there's a small helper,
inspect_areas(), at the bottom of this file) to set --min-area/--max-area
sensibly for your own footage.
"""

import argparse
import os
import glob
from collections import deque

import cv2
import math

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover -- tqdm is in requirements.txt; fallback keeps engine importable without it
    def tqdm(iterable=None, **kwargs):
        """No-op fallback so the tracking engine works without tqdm
        (e.g. minimal/headless installs and frozen PyInstaller builds
        where the console progress bar is unwanted anyway)."""
        return iterable if iterable is not None else iter(())

DEFAULTS = dict(
    n_background_samples=40,   # frames used to build the background model
    diff_threshold=25,         # 0-255, intensity difference to call "foreground"
    min_area=150,              # px^2, blobs smaller than this are noise
    max_area=3000,             # px^2, roughly one mouse's footprint -- TUNE THIS
    morph_kernel=5,            # size of morphological open/close kernel
    num_animals=2,
    color_mode="gray",         # "gray" (default) or "rgb" -- see build_background/foreground_mask
    reject_static_debris=True,     # see MotionHistory / reject_static_debris below
    debris_motion_window_s=1.2,    # how far back to look for "this blob ever moved" --
                                    # short on purpose: long enough to ride out a moment
                                    # of real stillness, short enough that a freshly-
                                    # dropped object's one-time appearance-diff (itself a
                                    # "moved" pixel, the instant it lands) ages back out of
                                    # the window quickly instead of masking it as "alive"
                                    # for as long as the window is
    debris_motion_pixel_threshold=12,  # per-pixel intensity change to count as "moved"
    debris_motion_fraction=0.02,   # min fraction of a blob's own pixels that must have
                                    # moved at some point in the window to count as alive
    kalman_process_var=4.0,      # px^2/frame^2, how much each animal's velocity is
                                    # allowed to drift frame-to-frame -- higher = the
                                    # predicted position trusts/follows recent motion more
                                    # eagerly (more responsive to real turns, but also to
                                    # detection noise); lower = smoother/steadier but slower
                                    # to react to a genuine direction change. See
                                    # CentroidKalman.
    kalman_measurement_var=25.0,   # px^2, how noisy a raw detected centroid is assumed to
                                    # be (e.g. from the k-means split during a blob merge).
                                    # Higher = trust the motion-model prediction more than
                                    # this frame's raw detection when blending the two.
    kalman_max_coast_frames=15,     # after this many CONSECUTIVE frames with no real
                                    # detection for a given animal, stop extrapolating its
                                    # position along its last known velocity and just hold
                                    # steady instead -- a real mouse doesn't move in a
                                    # straight line forever, so unbounded extrapolation
                                    # through a long occlusion (it stopped, turned around,
                                    # etc.) would drift further from the truth the longer it
                                    # coasts, not closer. Short gaps (a quick pass-through/
                                    # merge) stay within this window, so they still benefit
                                    # from motion-aware prediction; only occlusions longer
                                    # than this degrade to the old "freeze in place" behavior.
)


# --------------------------------------------------------------------------
# STAGE 1: DETECT
# --------------------------------------------------------------------------

def build_background(cap, n_samples=40, color_mode="gray"):
    """Estimate a static background image as the per-pixel MEDIAN of frames
    sampled evenly through the video. This works because the mice move
    around: at any given pixel, most sampled frames show empty arena floor,
    so the median at that pixel is "floor", not "mouse".

    color_mode="gray" (default) converts each sampled frame to grayscale
    first, same as before. color_mode="rgb" keeps all 3 color channels,
    producing a color background -- useful when the animal's color stands
    out from the floor but its grayscale brightness doesn't."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_samples = max(1, min(n_samples, max(total, 1)))
    idxs = np.linspace(0, max(total - 1, 0), num=n_samples, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if ok:
            if color_mode == "rgb":
                frames.append(frame.astype(np.float32))
            else:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not frames:
        raise RuntimeError("Could not read any frames to build the background model.")
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)


def foreground_mask(frame, background, diff_threshold, morph_kernel, color_mode="gray", exclusion=None):
    """Absolute-difference from background -> threshold -> clean with
    morphology. Returns a binary (0/255) mask of "moved" pixels.

    In RGB mode, frame/background are 3-channel BGR images; the per-pixel
    difference is the MAX across channels rather than a single grayscale
    value, so an animal that differs in color but not brightness still
    registers as foreground."""
    diff = cv2.absdiff(frame, background)
    if color_mode == "rgb" and diff.ndim == 3:
        diff = diff.max(axis=2)
    if exclusion is not None:
        diff[exclusion] = 0
    _, mask = cv2.threshold(diff, max(1, int(diff_threshold)), 255, cv2.THRESH_BINARY)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove speckle noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    return mask


def _probe_color_mode(cap, color_mode, probe_idxs, background,
                       min_area, max_area, diff_threshold, morph_kernel, num_animals):
    """Run foreground detection on a handful of sample frames under one
    color_mode and score how "clean" the result looks -- used by
    choose_color_mode() to compare "gray" vs "rgb" before committing to one
    for the full run. Returns (hit_rate, mean_quality):

      hit_rate     fraction of probe frames with at least one plausible
                   (min_area-or-bigger) blob at all -- catches the case
                   where a mode misses the animal completely (e.g. gray
                   mode when the animal matches the floor's brightness but
                   not its color).
      mean_quality per-frame score in [0, 1] of how close the blob COUNT
                   is to num_animals -- catches the noisier failure where
                   the animal IS detected but drowned in extra blobs
                   (shadow/flicker noise), which is common in whichever
                   mode is the worse fit for this footage."""
    hits = 0
    quality_total = 0.0
    n = 0
    for idx in probe_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        n += 1
        frame_proc = frame if color_mode == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(frame_proc, background, diff_threshold, morph_kernel, color_mode=color_mode)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = [c for c in contours if cv2.contourArea(c) >= min_area]
        blob_count = len(blobs)
        if blob_count >= 1:
            hits += 1
        closeness = 1.0 - min(abs(blob_count - num_animals), num_animals) / num_animals
        quality_total += closeness
    if n == 0:
        return 0.0, 0.0
    return hits / n, quality_total / n


def choose_color_mode(video_path, num_animals=2, min_area=150, max_area=3000,
                       diff_threshold=25, morph_kernel=5,
                       n_background_samples=20, n_probe_frames=8, rgb_margin=0.05):
    """Decide whether "gray" or "rgb" background-subtraction gives cleaner
    detection for THIS video, instead of the user having to guess and
    manually flip the setting.

    Why this matters: grayscale mode is cheaper and is the right choice
    most of the time, but it collapses color to brightness -- an animal
    that's a different COLOR from the floor but similar BRIGHTNESS (e.g. a
    grey-brown mouse on a mid-grey liner) can be nearly invisible in gray
    mode while standing out clearly once color is kept (rgb mode).

    Builds a background model under each mode, probes a handful of sample
    frames spread across the video, and scores each mode by (a) how often
    it detects at least one plausible blob at all, and (b) how close the
    blob count tends to be to num_animals (fewer stray/noise blobs).
    Switches to "rgb" only if it scores meaningfully better than "gray" by
    rgb_margin -- a tie, or any result too close to call, favors "gray"
    since it's the cheaper, simpler default and rgb mode also makes actual
    lighting flicker/shadows register as foreground more readily (color
    noise, not just brightness noise), so it isn't a free upgrade.

    Returns (chosen_mode, diagnostics) where diagnostics is a dict with the
    raw hit-rate/quality scores for both modes, useful for showing the user
    why a mode was picked."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for color-mode selection: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_probe = max(1, min(n_probe_frames, max(total, 1)))
        probe_idxs = np.linspace(0, max(total - 1, 0), num=n_probe, dtype=int)

        scores = {}
        for color_mode in ("gray", "rgb"):
            background = build_background(cap, n_samples=n_background_samples, color_mode=color_mode)
            hit_rate, mean_quality = _probe_color_mode(
                cap, color_mode, probe_idxs, background,
                min_area, max_area, diff_threshold, morph_kernel, num_animals)
            # equally weight "did it see anything" and "was what it saw clean"
            combined = 0.5 * hit_rate + 0.5 * mean_quality
            scores[color_mode] = dict(hit_rate=hit_rate, mean_quality=mean_quality, combined=combined)
    finally:
        cap.release()

    chosen = "rgb" if scores["rgb"]["combined"] > scores["gray"]["combined"] + rgb_margin else "gray"
    diagnostics = dict(scores)
    diagnostics["chosen"] = chosen
    return chosen, diagnostics


# --------------------------------------------------------------------------
# STAGE 2: SPLIT (merged-blob handling)
# --------------------------------------------------------------------------

def split_merged_blob(contour, mask_shape, k=2, seed_centroids=None):
    """Split one merged foreground blob into ``k`` centroids.

    When predicted identity positions are available, use a small seeded
    nearest-centre refinement instead of unconstrained k-means. OpenCV k-means
    is free to label clusters in arbitrary order on every frame, which can
    create avoidable ID swaps during contact/occlusion. Prediction-seeded
    refinement keeps the split spatially anchored to the existing identities.
    """

    blob_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(blob_mask == 255)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    if len(pts) < k:
        cx, cy = (pts.mean(axis=0) if len(pts) else (np.nan, np.nan))
        return [(cx, cy)] * k

    if seed_centroids is not None and len(seed_centroids) == k and all(np.isfinite(c).all() for c in seed_centroids):
        centers = np.asarray(seed_centroids, dtype=np.float32).copy()
        # A few deterministic Lloyd iterations.  This deliberately preserves
        # centre order: centre 0 remains mouse_A, centre 1 mouse_B, etc.
        for _ in range(8):
            d2 = ((pts[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(d2, axis=1)
            new_centers = centers.copy()
            for j in range(k):
                members = pts[labels == j]
                if len(members):
                    new_centers[j] = members.mean(axis=0)
            if np.allclose(new_centers, centers, atol=0.25):
                centers = new_centers
                break
            centers = new_centers
        return [tuple(c) for c in centers]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(pts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return [tuple(c) for c in centers]


class MotionHistory:
    """Rolling OR of per-frame "did this pixel change meaningfully" masks,
    covering the last `window_frames` frames. Used to tell a living,
    moving animal apart from static debris -- see reject_static_debris().
    Call update(gray_frame) once per frame with a grayscale uint8 frame
    (any color_mode; this is independent of the background-diff color
    mode), then read .mask."""

    def __init__(self, window_frames, pixel_threshold=12):
        self.window_frames = max(1, int(window_frames))
        self.pixel_threshold = pixel_threshold
        self._history = deque(maxlen=self.window_frames)
        self._prev_gray = None
        self.mask = None

    def update(self, gray_frame):
        if self._prev_gray is not None and self._prev_gray.shape == gray_frame.shape:
            diff = cv2.absdiff(gray_frame, self._prev_gray)
            moved = (diff > self.pixel_threshold).astype(np.uint8) * 255
            self._history.append(moved)
        self._prev_gray = gray_frame
        self.mask = np.bitwise_or.reduce(np.stack(self._history, axis=0), axis=0) if self._history else None


def _motion_overlap_fraction(contour, mask_shape, motion_mask):
    """Fraction of this contour's own pixels that showed real (above-noise)
    frame-to-frame motion at some point in the recent window."""
    blob_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
    blob_px = int(np.count_nonzero(blob_mask))
    if blob_px == 0:
        return 0.0
    overlap = int(np.count_nonzero((blob_mask > 0) & (motion_mask > 0)))
    return overlap / blob_px


def reject_static_debris(contours, mask_shape, motion_mask, num_animals, debris_motion_fraction=0.02):
    """Drop candidate blobs that never showed real pixel-level motion
    anywhere in the recent window -- e.g. a dropped stool pellet, which
    differs from the background FOREVER once it lands (so it competes with
    the real animal for a "top area" slot every single frame after that)
    but never itself moves, unlike a living animal, which shows at least
    occasional breathing/fur micro-motion even while behaviorally still.

    Only ever filters when there are MORE candidates than needed AND
    enough of them are clearly "alive" after filtering -- it never drops
    the candidate pool below num_animals, so a genuine long freezing bout
    (a real behavior, not debris) is never mistaken for debris and
    silently dropped: with only as many blobs as animals, or with every
    blob equally motionless (nothing else moving to compare against
    either), nothing is filtered and behavior is unchanged."""
    if motion_mask is None or len(contours) <= num_animals:
        return contours
    alive = [c for c in contours
             if _motion_overlap_fraction(c, mask_shape, motion_mask) >= debris_motion_fraction]
    if len(alive) < min(num_animals, len(contours)):
        return contours
    return alive


def get_centroids(mask, min_area, max_area, num_animals=2, motion_mask=None, debris_motion_fraction=0.02, expected_centroids=None):
    """Always tries to return `num_animals` (x, y) centroids for this frame.
    Handles three cases: two clean blobs, one merged blob (split it), or
    zero/partial detections (caller decides how to fill the gap).

    motion_mask (optional, from MotionHistory.mask) is used to reject
    static debris (see reject_static_debris) before ranking blobs by
    area -- without it, a dropped stool pellet that happens to be bigger
    than the animal at some point would permanently hijack that identity's
    tracked position."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours = reject_static_debris(contours, mask.shape, motion_mask, num_animals, debris_motion_fraction)
    contours.sort(key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        return None  # nothing detected this frame

    if len(contours) == 1:
        area = cv2.contourArea(contours[0])
        if area > max_area:
            return split_merged_blob(contours[0], mask.shape, k=num_animals, seed_centroids=expected_centroids)
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return None
        return [(M["m10"] / M["m00"], M["m01"] / M["m00"])]  # only one mouse visible

    centroids = []
    for c in contours[:num_animals]:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return centroids if centroids else None


# --------------------------------------------------------------------------
# STAGE 3: IDENTIFY (Hungarian algorithm + per-animal Kalman prediction)
# --------------------------------------------------------------------------

def _hungarian_assignment(prev_centroids, curr_centroids):
    """Core of match_identities: Hungarian-match curr_centroids against
    prev_centroids by minimizing total displacement. Returns (ordered,
    matched) where `ordered` is curr_centroids reordered/held to line up
    with prev_centroids' identities (exactly match_identities' own
    contract), and `matched` is the set of prev-indices that actually got
    paired with a real detection this round -- the rest fell back to
    holding their prev_centroids value, same as match_identities, but the
    caller needs to know WHICH indices those were (e.g. to decide which
    per-animal Kalman filter should absorb a real measurement this frame
    vs. just coast on its prediction)."""
    cost = np.zeros((len(prev_centroids), len(curr_centroids)))
    for i, p in enumerate(prev_centroids):
        for j, c in enumerate(curr_centroids):
            cost[i, j] = np.hypot(p[0] - c[0], p[1] - c[1])
    row_ind, col_ind = linear_sum_assignment(cost)
    ordered = list(prev_centroids)  # fallback: hold last position if unmatched
    matched = set()
    for r, c in zip(row_ind, col_ind):
        ordered[r] = curr_centroids[c]
        matched.add(r)
    return ordered, matched


def disc_mean(gray, xy, r):
    """Mean grey value in a disc of radius r around xy (the animal's look: coat darkness / brightness)."""
    x, y = xy
    if not (np.isfinite(x) and np.isfinite(y)):
        return np.nan
    h, w = gray.shape[:2]
    x0, x1 = max(0, int(x - r)), min(w, int(x + r) + 1)
    y0, y1 = max(0, int(y - r)), min(h, int(y + r) + 1)
    if x1 <= x0 or y1 <= y0:
        return np.nan
    sub = gray[y0:y1, x0:x1].astype(np.float32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    m = (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    return float(sub[m].mean()) if m.any() else np.nan


class Appearance:
    """What each animal looks like (mean grey in a small disc on its body), learned on frames where the animals
    are clearly apart.  After the animals have touched (a merged blob, positions only guessed), the first clear
    frame is matched by LOOK, so a swap made during the contact is undone -- when the animals look different
    (another coat colour, a marked / dyed back).  Identical-looking animals: positions only (and a note)."""

    def __init__(self, n, min_samples=15):
        self.n = n
        self.min = min_samples
        self.vals = [[] for _ in range(n)]

    def add(self, feats):
        for k, f in enumerate(feats):
            if np.isfinite(f):
                self.vals[k] = (self.vals[k] + [float(f)])[-300:]

    def model(self):
        if any(len(v) < self.min for v in self.vals):
            return None
        mu = np.array([np.median(v) for v in self.vals])
        sd = np.array([max(1.5, 1.4826 * np.median(np.abs(np.asarray(v) - np.median(v)))) for v in self.vals])
        return mu, sd

    def distinct(self):
        """True when every pair of animals differs clearly in look (> 4 pooled SD)."""
        m = self.model()
        if m is None:
            return False
        mu, sd = m
        for a in range(self.n):
            for b in range(a + 1, self.n):
                if abs(mu[a] - mu[b]) < 4.0 * math.sqrt(0.5 * (sd[a] ** 2 + sd[b] ** 2)):
                    return False
        return True

    def best_order(self, feats):
        """Permutation p (identity r <- detection p[r]) that best matches the learned looks, or None."""
        import itertools
        m = self.model()
        if m is None or not all(np.isfinite(feats)):
            return None
        mu, sd = m
        best, arg = None, None
        for p in itertools.permutations(range(self.n)):
            c = sum(((feats[p[r]] - mu[r]) / sd[r]) ** 2 for r in range(self.n))
            if best is None or c < best:
                best, arg = c, p
        return list(arg)


def match_identities(prev_centroids, curr_centroids):
    """Reorder curr_centroids to best match prev_centroids' identities by
    minimizing total displacement (Hungarian / linear sum assignment)."""
    ordered, _matched = _hungarian_assignment(prev_centroids, curr_centroids)
    return ordered


class CentroidKalman:
    """A minimal constant-velocity Kalman filter tracking ONE animal's
    (x, y) position + velocity, run independently per axis. Same core
    per-axis constant-velocity design as AlphaTracker's own
    PoseFlow/utils_kalman.py pos_vel_filter() -- ported here from N pose
    keypoints (which needs the `filterpy` package) down to a single
    centroid, hand-rolled with plain 2x2 numpy matrices instead, to keep
    this tracker's "no extra heavy deps" design (see module docstring).

    Usage each frame, per animal:
        x_pred, y_pred = kf.predict()   # ALWAYS -- advances the estimate
                                         # along the last known velocity,
                                         # occlusion or not
        kf.update(x_meas, y_meas)       # ONLY when a real detection was
                                         # matched to this animal this
                                         # frame; skip it otherwise and the
                                         # prediction stands as-is

    predict() must be called before update() each frame (matches this
    file's own main loop: predict first to get this frame's expected
    position for Hungarian matching, THEN update whichever animals a real
    detection was actually matched to)."""

    def __init__(self, x, y, process_var=4.0, measurement_var=25.0):
        self.pos = np.array([float(x), float(y)])
        self.vel = np.array([0.0, 0.0])
        # One 2x2 state covariance per axis: [[var_pos, cov], [cov, var_vel]].
        # Start confident in position (we're seeded from a real detection)
        # and deliberately unsure about velocity (no motion history yet).
        self._P = [np.array([[1.0, 0.0], [0.0, 50.0]]) for _ in range(2)]
        self._process_var = process_var
        self._measurement_var = measurement_var

    def predict(self):
        """Advance the estimate by one frame (dt=1) under the constant-
        velocity model alone, with no new measurement. Returns the
        predicted (x, y). Safe/expected to call every frame regardless of
        whether a detection exists that frame -- see class docstring."""
        F = np.array([[1.0, 1.0], [0.0, 1.0]])  # dt = 1 frame
        Q = np.array([[0.0, 0.0], [0.0, self._process_var]])  # process noise on velocity
        for axis in range(2):
            state = F @ np.array([self.pos[axis], self.vel[axis]])
            self._P[axis] = F @ self._P[axis] @ F.T + Q
            self.pos[axis], self.vel[axis] = state
        return float(self.pos[0]), float(self.pos[1])

    def update(self, x, y):
        """Fold in a real measurement (x, y), blending it with the
        current (already-predicted) estimate by their relative
        uncertainties. Call once per frame, AFTER predict()."""
        H = np.array([1.0, 0.0])
        for axis, z in enumerate((x, y)):
            P = self._P[axis]
            S = float(H @ P @ H.T) + self._measurement_var
            K = (P @ H) / S  # Kalman gain, one per state component [pos, vel]
            state = np.array([self.pos[axis], self.vel[axis]])
            residual = float(z) - float(H @ state)
            state = state + K * residual
            self._P[axis] = P - np.outer(K, H) @ P
            self.pos[axis], self.vel[axis] = state


# --------------------------------------------------------------------------
# REFERENCE / CALIBRATION PREVIEW FRAMES
# --------------------------------------------------------------------------

def save_preview_frames(video_path, output_dir, n_samples=6, **overrides):
    """Save a handful of sample frames evenly spaced across the video, each
    annotated with the detected animal(s) at that frame -- lets you
    sanity-check min/max area and the difference threshold before (or
    after) running the full tracker, same idea as Standard Tracking's
    calibration preview. Returns the list of saved file paths."""
    cfg = {**DEFAULTS, **overrides}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_animals = cfg["num_animals"]

    background = build_background(cap, cfg["n_background_samples"], color_mode=cfg["color_mode"])

    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(preview_dir, "preview_*.png")):
        os.remove(stale)

    n_samples = max(1, int(n_samples))
    positions = np.linspace(0, max(0, n_frames - 1), n_samples).astype(int)
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]  # BGR
    saved_paths = []

    for i, frame_idx in enumerate(positions, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_for_diff = frame if cfg["color_mode"] == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(frame_for_diff, background, cfg["diff_threshold"], cfg["morph_kernel"],
                                color_mode=cfg["color_mode"])
        detected = get_centroids(mask, cfg["min_area"], cfg["max_area"], n_animals)

        disp = frame.copy()
        if detected:
            for j, (x, y) in enumerate(detected[:n_animals]):
                if np.isnan(x):
                    continue
                pt = (int(x), int(y))
                color = colors[j % len(colors)]
                cv2.circle(disp, pt, 8, color, 2)
                cv2.putText(disp, chr(65 + j), (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
# MAIN TRACKING LOOP
# --------------------------------------------------------------------------

def track_video(video_path, output_csv, annotate_path=None, progress_callback=None,
                 show_display=False, frame_callback=None, warp=None, exclusion_mask=None,
                 background_override=None, **overrides):
    """warp(frame)->frame (optional crop/perspective), exclusion_mask (bool,
    warped size; True = ignore) and background_override (uint8 image in the
    warped space) let the application apply its crop / masks / ghost-free
    background to the multi-animal engine too."""
    cfg = {**DEFAULTS, **overrides}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if warp is not None:
        ok0, f0 = cap.read()
        if ok0:
            h, w = warp(f0).shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_animals = cfg["num_animals"]
    start_time = max(0.0, float(cfg.get("start_time", 0.0) or 0.0))
    requested_end = cfg.get("end_time")
    end_time = (float(requested_end) if requested_end is not None else n_frames / fps)
    start_frame = min(max(int(round(start_time * fps)), 0), max(n_frames - 1, 0))
    end_frame = min(max(int(round(end_time * fps)), start_frame + 1), n_frames)
    selected_frames = max(1, end_frame - start_frame)

    print("Building background model...")
    if background_override is not None:
        background = background_override
    else:
        background = build_background(cap, cfg["n_background_samples"], color_mode=cfg["color_mode"])

    writer = None
    if annotate_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotate_path), fourcc, fps, (w, h))

    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]  # BGR
    trails = {i: [] for i in range(n_animals)}

    records = []
    appearance = Appearance(n_animals) if cfg.get("appearance_id", True) else None
    look_r = max(2.0, 0.25 * math.sqrt(max(1.0, float(cfg["min_area"]) * 3.0)))   # disc inside the body
    since_clear = 0          # frames since the animals were last clearly apart
    reid_swaps = 0
    prev_centroids = None    # used only for the initial seeding, before Kalman filters exist
    kalmans = None           # list[CentroidKalman], one per animal, created once seeded
    frames_since_update = None  # consecutive no-detection frames per animal, for coasting
    stopped_early = False

    motion_history = None
    if cfg["reject_static_debris"]:
        motion_history = MotionHistory(fps * cfg["debris_motion_window_s"],
                                        pixel_threshold=cfg["debris_motion_pixel_threshold"])

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_iter = range(start_frame, end_frame) if (show_display or progress_callback) else tqdm(range(start_frame, end_frame), desc="Tracking")

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break
        if warp is not None:
            frame = warp(frame)
        frame_for_diff = frame if cfg["color_mode"] == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(frame_for_diff, background, cfg["diff_threshold"], cfg["morph_kernel"],
                                color_mode=cfg["color_mode"], exclusion=exclusion_mask)

        if motion_history is not None:
            gray_for_motion = frame_for_diff if cfg["color_mode"] != "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion_history.update(gray_for_motion)

        # Predict BEFORE splitting a merged blob so overlap handling can use
        # identity-specific expected positions. This is the key anti-swap
        # improvement: the split itself is identity-aware, not merely a
        # post-hoc Hungarian reorder of arbitrary k-means labels.
        predicted = None
        if kalmans is not None:
            for r in range(n_animals):
                if frames_since_update[r] > cfg["kalman_max_coast_frames"]:
                    kalmans[r].vel[:] = 0.0
            predicted = [kf.predict() for kf in kalmans]

        detected = get_centroids(mask, cfg["min_area"], cfg["max_area"], n_animals,
                                  motion_mask=(motion_history.mask if motion_history else None),
                                  debris_motion_fraction=cfg["debris_motion_fraction"],
                                  expected_centroids=predicted)

        if kalmans is None:
            # --- SEEDING: no Kalman filters yet (either this is frame 0,
            # or nothing confident enough has been seen yet to start
            # them from) -- same logic this file has always used, purely
            # matching against the last raw position. ---
            if detected is None:
                curr = prev_centroids if prev_centroids else [(np.nan, np.nan)] * n_animals
                flag = "lost"
            elif len(detected) < n_animals:
                if prev_centroids is None:
                    curr = (detected + [detected[0]])[:n_animals]
                else:
                    curr = match_identities(prev_centroids, detected)
                flag = "partial"
            else:
                if prev_centroids is None:
                    detected = sorted(detected, key=lambda p: p[0])  # seed left-to-right
                    curr = detected[:n_animals]
                else:
                    curr = match_identities(prev_centroids, detected)
                flag = "ok"

            prev_centroids = curr
            # Once every animal has a real, non-NaN fix, start a Kalman
            # filter for each from here on -- see CentroidKalman.
            if flag == "ok" and not any(np.isnan(c[0]) for c in curr):
                kalmans = [CentroidKalman(x, y, process_var=cfg["kalman_process_var"],
                                           measurement_var=cfg["kalman_measurement_var"])
                           for x, y in curr]
                frames_since_update = [0] * n_animals
        else:
            # --- STEADY STATE: predict each animal's expected position
            # for THIS frame first (motion-aware -- keeps extrapolating
            # through an occlusion instead of freezing, unless it's been
            # coasting too long, see kalman_max_coast_frames), THEN match
            # detections against that expectation rather than the raw
            # last-measured position, and only update the filters that
            # actually got a real detection this round. ---
            if detected is None:
                curr = predicted
                matched = set()
                flag = "lost"
            elif len(detected) < n_animals:
                curr, matched = _hungarian_assignment(predicted, detected)
                flag = "partial"
            else:
                curr, matched = _hungarian_assignment(predicted, detected)
                flag = "ok"

            for r in range(n_animals):
                if r in matched:
                    kalmans[r].update(*curr[r])
                    curr[r] = tuple(kalmans[r].pos)  # report the smoothed (not raw) position
                    frames_since_update[r] = 0
                else:
                    frames_since_update[r] += 1

            # identity by LOOK after a contact (see Appearance)
            if appearance is not None:
                pts = [c for c in curr if np.isfinite(c[0])]
                apart = flag == "ok" and len(pts) == n_animals and len(detected or []) >= n_animals and all(
                    math.hypot(pts[a][0] - pts[b][0], pts[a][1] - pts[b][1]) > 6.0 * look_r
                    for a in range(n_animals) for b in range(a + 1, n_animals))
                if apart:
                    gray = frame_for_diff if frame_for_diff.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    feats = [disc_mean(gray, c, look_r) for c in curr]
                    if since_clear > 0 and appearance.distinct():
                        p = appearance.best_order(feats)
                        if p is not None and p != list(range(n_animals)):
                            curr = [curr[p[r]] for r in range(n_animals)]
                            kalmans = [kalmans[p[r]] for r in range(n_animals)]
                            frames_since_update = [frames_since_update[p[r]] for r in range(n_animals)]
                            feats = [feats[p[r]] for r in range(n_animals)]
                            reid_swaps += 1
                    appearance.add(feats)
                    since_clear = 0
                else:
                    since_clear += 1

            prev_centroids = curr

        for i, (x, y) in enumerate(curr):
            records.append(dict(
                frame=frame_idx, time_s=round(frame_idx / fps, 3),
                mouse_id=f"mouse_{chr(65 + i)}", x=x, y=y, status=flag,
            ))

        if writer is not None or show_display:
            annotated = frame.copy()
            for i, (x, y) in enumerate(curr):
                if not np.isnan(x):
                    pt = (int(x), int(y))
                    trails[i] = (trails[i] + [pt])[-100:]
                    for j in range(1, len(trails[i])):
                        cv2.line(annotated, trails[i][j - 1], trails[i][j], colors[i % len(colors)], 1)
                    cv2.circle(annotated, pt, 6, colors[i % len(colors)], -1)
                    cv2.putText(annotated, chr(65 + i), (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)
            try:
                from tracking.location import draw_status_box
                n_ok = sum(1 for (x, _y) in curr if not np.isnan(x))
                draw_status_box(annotated, f"{n_ok}/{len(curr)} TRACKED   t = {frame_idx / fps:.2f} s   "
                                           f"frame {frame_idx}",
                                (40, 200, 40) if n_ok == len(curr) else (0, 165, 255) if n_ok else (0, 0, 255))
            except Exception:
                pass
            if writer is not None:
                writer.write(annotated)
            if frame_callback is not None and (frame_idx - start_frame) % 5 == 0:
                frame_callback(annotated)
            if show_display:
                cv2.putText(annotated, f"Frame: {frame_idx}/{n_frames}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow("MULTI-MOUSE TRACKING - Q to stop", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stopped_early = True

        if progress_callback and (frame_idx % 5 == 0 or frame_idx == n_frames - 1):
            progress_callback(min(1.0, (frame_idx - start_frame + 1) / max(1, selected_frames)))

        if stopped_early:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if show_display:
        cv2.destroyAllWindows()
    if progress_callback:
        progress_callback(1.0)

    df = pd.DataFrame.from_records(records)
    df.to_csv(output_csv, index=False)

    if appearance is not None:
        print(f"  identity by look: animals {'look different' if appearance.distinct() else 'look alike'}; "
              f"{reid_swaps} identity correction(s) after contacts")
        df.attrs["look_distinct"] = bool(appearance.distinct())
        df.attrs["identity_corrections"] = int(reid_swaps)
    n_lost = (df["status"] == "lost").sum()
    n_partial = (df["status"] == "partial").sum()
    print(f"Saved {len(df)} rows -> {output_csv}")
    print(f"  status counts: ok={(df['status'] == 'ok').sum()}, "
          f"partial={n_partial}, lost={n_lost}")
    if annotate_path:
        print(f"Annotated preview -> {annotate_path}  (watch this for ID swaps)")
    return df


# --------------------------------------------------------------------------
# Time bins (Upgrade Plan Tier 2 #9) -- Standard Tracking's own
# configurable-size time-bin report (tracking.behaviour.calculate_bins),
# extended to Multi-Mouse: per (mouse_id, bin), how far that mouse
# traveled and how much of the bin it was tracked for. Multi-Mouse has no
# zones of its own the way Standard Tracking's Setup body does, so
# distance-per-bin is the natural per-bin measure here instead of
# time-in-zone.
# --------------------------------------------------------------------------

def calculate_time_bins(tracks_df, fps, start_time=None, end_time=None, bin_size_s=60.0):
    """tracks_df is exactly what track_video (above) returns/writes to
    tracks.csv -- one row per (frame, mouse_id) with x/y/status columns.
    start_time/end_time default to the track's own first/last timestamp
    when not given (main_window.py doesn't carry an explicit Start/End (s)
    window for Multi-Mouse the way Standard Tracking's Setup body does).
    Returns one row per (mouse_id, bin); an empty DataFrame if there's
    nothing to bin."""
    from tracking.time_bins import compute_bin_edges, bin_row_label

    if tracks_df is None or len(tracks_df) == 0:
        return pd.DataFrame()
    if start_time is None:
        start_time = float(tracks_df["time_s"].min())
    if end_time is None:
        end_time = float(tracks_df["time_s"].max())
    edges, unit_divisor, unit_name = compute_bin_edges(start_time, end_time, bin_size_s)

    rows = []
    for mouse_id, sub in tracks_df.groupby("mouse_id"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        # Frame-to-frame displacement -- NaN on the very first row (no
        # previous position yet) reads as 0 traveled, same convention
        # Standard Tracking's own Distance_pixels column uses.
        step_distance = np.hypot(sub["x"].diff(), sub["y"].diff()).fillna(0.0)
        sub = sub.assign(_step_distance=step_distance)

        for i in range(len(edges) - 1):
            a, b = float(edges[i]), float(edges[i + 1])
            seg = sub[(sub["time_s"] >= a) & (sub["time_s"] < b)]
            tracked = int((seg["status"] != "lost").sum()) if "status" in seg.columns else len(seg)
            rows.append({
                "mouse_id": mouse_id,
                "Bin": bin_row_label(a, b, start_time, unit_divisor, unit_name),
                "Start_seconds": a, "End_seconds": b,
                "Distance_pixels": float(seg["_step_distance"].sum()),
                "Tracked_frames": tracked,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# OPTIONAL HELPER: check typical blob area to set --min-area/--max-area
# --------------------------------------------------------------------------

def inspect_areas(video_path, n_frames_to_check=10, color_mode="gray"):
    """Prints the area (in pixels) of the largest blob in a handful of
    frames, so you can pick sensible --min-area / --max-area values before
    running the full tracker."""
    cap = cv2.VideoCapture(str(video_path))
    background = build_background(cap, 40, color_mode=color_mode)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, max(total - 1, 0), num=n_frames_to_check, dtype=int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        frame_for_diff = frame if color_mode == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(frame_for_diff, background, DEFAULTS["diff_threshold"], DEFAULTS["morph_kernel"],
                                color_mode=color_mode)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)[:3]
        print(f"frame {i}: top blob areas = {areas}")
    cap.release()


def main():
    p = argparse.ArgumentParser(description="Free from-scratch two-mouse identity tracker.")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--output", default="tracks.csv", help="Path to output CSV")
    p.add_argument("--annotate", default=None, help="Optional path to save an annotated preview video (.mp4)")
    p.add_argument("--inspect-areas", action="store_true",
                   help="Just print blob areas for a few frames, then exit (use this first to tune --min-area/--max-area)")
    p.add_argument("--num-animals", type=int, default=DEFAULTS["num_animals"])
    p.add_argument("--min-area", type=int, default=DEFAULTS["min_area"])
    p.add_argument("--max-area", type=int, default=DEFAULTS["max_area"])
    p.add_argument("--diff-threshold", type=int, default=DEFAULTS["diff_threshold"])
    p.add_argument("--bg-samples", type=int, default=DEFAULTS["n_background_samples"])
    args = p.parse_args()

    if args.inspect_areas:
        inspect_areas(args.video)
        return

    track_video(
        args.video, args.output, annotate_path=args.annotate,
        num_animals=args.num_animals, min_area=args.min_area,
        max_area=args.max_area, diff_threshold=args.diff_threshold,
        n_background_samples=args.bg_samples,
    )


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# APPLICATION INTEGRATION
# --------------------------------------------------------------------------

def process_multi_mouse_video(video_path, output_dir, num_animals=2,
                              progress_callback=None, frame_callback=None,
                              start_time=0.0, end_time=None, setup=None, **overrides):
    """Run the multi-animal engine as a BehavioralTracker application job.

    `setup` (the same dict the single-animal pipeline receives) adds the
    application's crop / masks / auto-mask-outside-zones, a ghost-free
    background from the robust engine, automatic animal-size limits and
    per-mouse zone statistics."""
    from pathlib import Path
    from analysis.calculations import safe_col

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_csv = out / "multi_mouse_tracking.csv"
    preview = out / "multi_mouse_preview.mp4"
    setup = dict(setup or {})

    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    cfg["num_animals"] = int(max(2, min(4, num_animals)))
    cfg["start_time"] = float(start_time or 0.0)
    cfg["end_time"] = float(end_time) if end_time is not None else None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    matrix = setup.get("matrix")
    warp_w = int(setup.get("warp_w") or fw)
    warp_h = int(setup.get("warp_h") or fh)
    need_warp = matrix is not None and not (np.allclose(np.asarray(matrix, float), np.eye(3)) and
                                            (warp_w, warp_h) == (fw, fh))
    warp = (lambda f: cv2.warpPerspective(f, matrix, (warp_w, warp_h))) if need_warp else None

    roi_points = {k: v for k, v in (setup.get("roi_points") or {}).items() if len(v) >= 3}
    exclusion = None
    try:
        from tracking.location import build_detection_exclusion
        if setup.get("mask_points") or (setup.get("auto_mask_outside_zones") and roi_points):
            exclusion = build_detection_exclusion(warp_w, warp_h, setup.get("mask_points") or [], roi_points,
                                                  bool(setup.get("auto_mask_outside_zones")))
            if exclusion.all():
                exclusion = None
    except Exception:
        exclusion = None

    # ghost-free, polarity-aware background + automatic animal size (robust engine)
    background = None
    try:
        from tracking.robust_tracker import build_background as robust_bg, detection_scale
        s0 = int(round(cfg["start_time"] * fps))
        s1 = n_frames - 1 if cfg["end_time"] is None else min(n_frames - 1, int(round(cfg["end_time"] * fps)))

        def read_warped(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if not ok:
                return None
            return warp(f) if warp else f
        positions = np.linspace(s0, max(s0, s1), max(10, int(setup.get("background_samples", 60)))).astype(int)
        bgm = robust_bg(read_warped, positions, cfg.get("color_mode", "gray"), "auto",
                        valid_full=None if exclusion is None else ~exclusion,
                        scale=detection_scale(warp_w, warp_h))
        img = bgm.full_res_bgr(warp_w, warp_h)
        background = img if cfg.get("color_mode") == "rgb" else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if bgm.animal_area and not setup.get("max_area"):
            a = bgm.animal_area / (bgm.scale ** 2)
            cfg["min_area"] = max(int(setup.get("min_area", 15) or 15), int(0.25 * a))
            cfg["max_area"] = int(1.6 * a)
        cfg["diff_threshold"] = max(10, int(2 * float(setup.get("min_contrast", 12) or 12)))
        if progress_callback:
            progress_callback(0.06)
    except Exception as exc:
        print(f"[multi-mouse] robust background unavailable ({exc}); using the classic median.")
    cap.release()

    def prog(f):
        if progress_callback:
            progress_callback(0.06 + 0.88 * float(f))

    df = track_video(
        str(video_path), str(raw_csv), annotate_path=str(preview),
        progress_callback=prog, show_display=False, frame_callback=frame_callback,
        warp=warp, exclusion_mask=exclusion, background_override=background, **cfg
    )
    if df.empty:
        raise RuntimeError("Multi-animal tracker produced no tracking rows.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scale_factor = setup.get("scale_factor")
    unit = setup.get("scale_unit") or "cm"
    per_animal = []
    zone_rows = []
    for mouse_id, g in df.groupby("mouse_id", sort=True):
        g = g.sort_values("frame")
        xy = g[["x", "y"]].to_numpy(dtype=float)
        valid = np.isfinite(xy).all(axis=1) & (g["status"].to_numpy() != "lost")
        dist_px = float(np.linalg.norm(np.diff(xy[valid], axis=0), axis=1).sum()) if valid.sum() > 1 else 0.0
        row = {"mouse_id": mouse_id, "Total_distance_pixels": dist_px,
               "Tracked_frames": int(valid.sum()), "Lost_frames": int((g["status"] == "lost").sum()),
               "Partial_frames": int((g["status"] == "partial").sum()),
               "Tracking_quality_percent": 100.0 * float(valid.sum()) / max(1, len(g))}
        if scale_factor:
            row[f"Total_distance_{unit}"] = dist_px * float(scale_factor)
        if roi_points:
            try:
                from tracking.location import build_roi_masks, zone_flags
                from analysis.test_metrics import zone_table
                masks = build_roi_masks((warp_h, warp_w), roi_points)
                fl = zone_flags(masks, xy[:, 0], xy[:, 1], valid)
                zdf = pd.DataFrame({"Time_seconds": g["time_s"].to_numpy(float),
                                    "Tracking_Status": np.where(valid, "Tracked", "Lost")})
                for z in roi_points:
                    zdf[f"In_{safe_col(z)}"] = fl[z]
                zt = zone_table(zdf, list(roi_points), fps, cfg["start_time"])
                zone_rows.append(zt.assign(Mouse=mouse_id)[["Mouse"] + list(zt.columns)])
                for r in zt.itertuples(index=False):
                    row[f"{safe_col(r.Zone)}_time_s"] = r.Time_s
                    row[f"{safe_col(r.Zone)}_percent"] = r.Percent_time
                    row[f"{safe_col(r.Zone)}_entries"] = r.Entries
            except Exception as exc:
                print(f"[multi-mouse] zone statistics skipped: {exc}")
        per_animal.append(row)

    n_an = max(1, cfg["num_animals"])
    summary = {
        "Mode": "Multi-Mouse",
        "Video": str(video_path),
        "Video_name": Path(video_path).name,
        "FPS": float(fps),
        "Frames_processed": int(len(df) / n_an) if len(df) else 0,
        "Tracked_frames": int(sum(r["Tracked_frames"] for r in per_animal) / n_an),
        "Lost_frames": int(sum(r["Lost_frames"] for r in per_animal) / n_an),
        "Tracking_quality_percent": float(np.mean([r["Tracking_quality_percent"] for r in per_animal])),
        "Num_animals": int(cfg["num_animals"]),
        "Tracking_mode": "multi",
        "Total_distance_pixels": float(sum(r["Total_distance_pixels"] for r in per_animal)),
        "Total_transitions": 0,
        "Output_folder": str(out),
    }
    if scale_factor:
        summary[f"Total_distance_{unit}"] = summary["Total_distance_pixels"] * float(scale_factor)
    for r in per_animal:
        for k, v in r.items():
            if k != "mouse_id":
                summary[f"{r['mouse_id']}_{k}"] = v

    bridge = df.copy()
    bridge["Time_seconds"] = bridge["time_s"]
    bridge["Tracking_Status"] = bridge["status"].map({"lost": "Lost", "partial": "Partial", "ok": "Tracked",
                                                      "tracked": "Tracked"}).fillna("Tracked")
    bridge["Distance_pixels"] = 0.0
    for mouse_id, idx in bridge.groupby("mouse_id", sort=False).groups.items():
        g = bridge.loc[idx].sort_values("frame")
        xy = g[["x", "y"]].to_numpy(dtype=float)
        d = np.zeros(len(g), dtype=float)
        if len(g) > 1:
            valid_pair = np.isfinite(xy[1:]).all(axis=1) & np.isfinite(xy[:-1]).all(axis=1)
            d[1:][valid_pair] = np.linalg.norm(xy[1:][valid_pair] - xy[:-1][valid_pair], axis=1)
        bridge.loc[g.index, "Distance_pixels"] = d
    bridge[["frame", "mouse_id", "Time_seconds", "x", "y", "Tracking_Status", "Distance_pixels"]].to_csv(
        out / "raw_tracking.csv", index=False)
    pd.DataFrame(per_animal).to_csv(out / "multi_mouse_summary.csv", index=False)
    if zone_rows:
        pd.concat(zone_rows, ignore_index=True).to_csv(out / "zone_summary.csv", index=False)

    # charts are drawn on a picture that looks like the original video (no masks / dimming)
    ref = None
    try:
        from tracking.location import original_view_image
        fr = np.sort(df["frame"].unique())
        if len(fr):
            pick = fr[np.unique(np.linspace(0, len(fr) - 1, min(15, len(fr))).astype(int))]
            img = original_view_image(video_path, pick, warp=warp)
            if img is not None:
                ref = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        print(f"[multi-mouse] original-view chart background skipped: {exc}")
    if ref is None and background is not None:
        ref = background if background.ndim == 3 else cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(8, 6))
    if ref is not None:
        ax.imshow(ref, extent=(0, warp_w, warp_h, 0))
    for mouse_id, g in df.groupby("mouse_id", sort=True):
        ax.plot(g["x"], g["y"], linewidth=1.0, label=mouse_id)
    for name, pts in roi_points.items():
        arr = np.array(list(pts) + [pts[0]], float)
        ax.plot(arr[:, 0], arr[:, 1], "--", linewidth=1.0, color="white")
    ax.set_xlim(0, warp_w); ax.set_ylim(warp_h, 0)
    ax.set_title("Multi-Mouse Trajectories"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(out / "trajectory.png", dpi=160); plt.close(fig)

    from scipy.ndimage import gaussian_filter
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = df[np.isfinite(df["x"]) & np.isfinite(df["y"])]
    if ref is not None:
        ax.imshow(ref, extent=(0, warp_w, warp_h, 0))
    if len(valid):
        heat, _xe, _ye = np.histogram2d(valid["x"], valid["y"], bins=50, range=[[0, warp_w], [0, warp_h]])
        heat = gaussian_filter(heat, sigma=1.2).T
        hn = heat / heat.max() if heat.max() > 0 else heat
        ax.imshow(hn, extent=(0, warp_w, warp_h, 0), origin="upper", cmap="inferno",
                  alpha=np.clip(hn * 1.3, 0, 0.85), interpolation="bilinear")
    ax.set_xlim(0, warp_w); ax.set_ylim(warp_h, 0)
    ax.set_title("Multi-Mouse Occupancy Heatmap (all animals)")
    fig.tight_layout(); fig.savefig(out / "heatmap.png", dpi=160); plt.close(fig)

    bins = calculate_time_bins(df, fps=fps, start_time=start_time, end_time=end_time,
                               bin_size_s=float(setup.get("bin_size_s", overrides.get("bin_size_s", 60.0)) or 60.0))
    bins.to_csv(out / "multi_mouse_timebins.csv", index=False)
    if not bins.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for mouse_id, g in bins.groupby("mouse_id", sort=True):
            ax.plot(g["Start_seconds"], g["Distance_pixels"], marker="o", label=mouse_id)
        ax.set_title("Distance per Time Bin")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Distance (px)"); ax.legend(loc="best")
        fig.tight_layout(); fig.savefig(out / "timebins.png", dpi=160); plt.close(fig)

    try:
        with pd.ExcelWriter(out / "MultiMouse_Analysis.xlsx", engine="openpyxl") as xw:
            pd.DataFrame([{k: v for k, v in summary.items()}]).to_excel(xw, sheet_name="Summary", index=False)
            pd.DataFrame(per_animal).to_excel(xw, sheet_name="Per Mouse", index=False)
            if not bins.empty:
                bins.to_excel(xw, sheet_name="Time Bins", index=False)
            df.to_excel(xw, sheet_name="Tracking", index=False)
    except Exception as exc:
        print(f"[multi-mouse] Excel export skipped: {exc}")
    if progress_callback:
        progress_callback(1.0)
    return summary
