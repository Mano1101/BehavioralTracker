"""
Regression test for the Multi-Mouse Tracking identity-tracking upgrade MM
asked for after sharing AlphaTracker (https://github.com/MVIG-SJTU/
AlphaTracker) and asking to use it for multi-mouse tracking. AlphaTracker
itself is a heavy deep-learning pipeline (YOLO detection + SPPE pose
estimation + PoseFlow tracking, GPU/CUDA + a trained model required) that
doesn't fit this app's "no GPU, no paid software" design, and it ships no
mouse-specific trained weights of its own -- so per MM's own choice, only
the IDEA behind its PoseFlow tracker's per-animal Kalman filter is ported
here, adapted to this tracker's classical centroid-based approach (no
deep learning, no `filterpy` dependency -- see CentroidKalman's own
docstring in tracking/two_mouse.py).

WHAT CHANGED: tracking/two_mouse.py's identify stage used to match this
frame's detections against each animal's raw LAST-SEEN position, and simply
FROZE that position outright on any frame with a missing/partial detection
(status="lost"/"partial") -- e.g. a mouse briefly hidden behind its
cage-mate or under bedding. On reappearing several frames later, the
tracker had a stale, wrong anchor to match against, and the reported
trail had a dead flat spot then a sudden jump.

Now each animal has its own constant-velocity Kalman filter
(CentroidKalman): every frame, it PREDICTS an expected position first
(extrapolating along the animal's last known velocity, occlusion or not),
matching happens against that prediction instead of the raw last position,
and only animals that got a real detection this frame have their filter
updated -- an unmatched animal's reported position keeps moving smoothly
along its last trend instead of freezing, until it's been unmatched for
longer than kalman_max_coast_frames, at which point it gracefully degrades
back to holding still (a mouse doesn't move in a straight line forever, so
unbounded extrapolation through a very long occlusion would just drift
further from the truth, not closer).

This is pure tracking/two_mouse.py logic (CentroidKalman + a synthetic
video through track_video), no Qt needed -- run the same way as
test_auto_mask_outside_zones.py.

Run directly:
    python3.12 qt_app/tests/test_multi_mouse_kalman_tracking.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import os
import tempfile

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.two_mouse import (
    CentroidKalman, DEFAULTS, build_background, foreground_mask,
    get_centroids, track_video, calculate_time_bins,
)

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# --------------------------------------------------------------------------
# 1) CentroidKalman -- pure math, no video needed.
# --------------------------------------------------------------------------
kf = CentroidKalman(0.0, 0.0, process_var=4.0, measurement_var=25.0)
true_x = 0.0
for _ in range(40):
    true_x += 5.0
    kf.predict()
    kf.update(true_x, 0.0)
check("noiseless constant-velocity motion: filter converges to near-zero position error",
      abs(kf.pos[0] - true_x) < 1.0)
check("...and recovers the true velocity (5.0 px/frame)", abs(kf.vel[0] - 5.0) < 0.5)

kf2 = CentroidKalman(0.0, 0.0)
for t in range(5):
    kf2.predict()
    kf2.update(5.0 * (t + 1), 0.0)
coasted = [kf2.predict() for _ in range(5)]  # no updates -- pure extrapolation
steps = [coasted[i + 1][0] - coasted[i][0] for i in range(len(coasted) - 1)]
check("predict()-only coasting advances by the last known velocity each frame (no freeze)",
      all(abs(s - 5.0) < 0.5 for s in steps))

rng = np.random.default_rng(0)
kf3 = CentroidKalman(0.0, 0.0, process_var=4.0, measurement_var=25.0)
true_pos = 0.0
raw_errs, filt_errs = [], []
for t in range(60):
    true_pos += 3.0
    noisy = true_pos + rng.normal(0, 5.0)
    kf3.predict()
    kf3.update(noisy, 0.0)
    if t >= 10:  # skip warmup
        raw_errs.append(abs(noisy - true_pos))
        filt_errs.append(abs(kf3.pos[0] - true_pos))
check("filtering noisy measurements gives lower mean error than the raw measurements themselves "
      f"(raw={np.mean(raw_errs):.1f}px vs filtered={np.mean(filt_errs):.1f}px)",
      np.mean(filt_errs) < np.mean(raw_errs))

print()

# --------------------------------------------------------------------------
# 2) End-to-end: a mouse occluded for longer than a couple of frames while
# it keeps moving -- the actual scenario the upgrade targets. Compares
# track_video's CURRENT behavior against a reimplementation of the OLD
# freeze-on-partial logic (kept only here, for comparison -- the shipped
# module has already moved to the new behavior).
# --------------------------------------------------------------------------
VW, VH, FPS = 240, 160, 30
BLOB_R = 8
N_FRAMES = 90
OCCLUDE_START, OCCLUDE_END = 35, 55  # mouse B hidden for 20 frames

true_B = []
frames = []
for t in range(N_FRAMES):
    ax, ay = 20 + 2.0 * t, 30
    bx, by = 20 + 2.0 * t, 130
    true_B.append((bx, by))
    frame = np.full((VH, VW), 200, dtype=np.uint8)
    cv2.circle(frame, (int(ax), int(ay)), BLOB_R, 40, -1)
    if not (OCCLUDE_START <= t < OCCLUDE_END):
        cv2.circle(frame, (int(bx), int(by)), BLOB_R, 40, -1)
    frames.append(frame)

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "kalman_tracking_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW, VH))
for f in frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
writer.release()


def run_old_freeze_logic():
    """The tracker's identify stage exactly as it worked BEFORE this
    upgrade: Hungarian-match against the raw last position, and freeze
    that position outright whenever the detected count is short."""
    def match_identities_old(prev_centroids, curr_centroids):
        cost = np.zeros((len(prev_centroids), len(curr_centroids)))
        for i, p in enumerate(prev_centroids):
            for j, c in enumerate(curr_centroids):
                cost[i, j] = np.hypot(p[0] - c[0], p[1] - c[1])
        row_ind, col_ind = linear_sum_assignment(cost)
        ordered = list(prev_centroids)
        for r, c in zip(row_ind, col_ind):
            ordered[r] = curr_centroids[c]
        return ordered

    cap = cv2.VideoCapture(VIDEO_PATH)
    background = build_background(cap, 40, color_mode="gray")
    n_animals = 2
    prev_centroids = None
    reported_B = []
    for _t in range(N_FRAMES):
        ok, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(gray, background, 25, 5, color_mode="gray")
        detected = get_centroids(mask, 150, 3000, n_animals)
        if detected is None:
            curr = prev_centroids if prev_centroids else [(np.nan, np.nan)] * n_animals
        elif len(detected) < n_animals:
            curr = (detected + [detected[0]])[:n_animals] if prev_centroids is None \
                else match_identities_old(prev_centroids, detected)
        else:
            if prev_centroids is None:
                detected = sorted(detected, key=lambda p: p[0])
                curr = detected[:n_animals]
            else:
                curr = match_identities_old(prev_centroids, detected)
        prev_centroids = curr
        b_idx = 0 if curr[0][1] > 80 else 1  # mouse B is whichever sits at the bottom
        reported_B.append(curr[b_idx])
    cap.release()
    return reported_B


old_B = run_old_freeze_logic()

tracks_csv = os.path.join(tempfile.gettempdir(), "kalman_tracking_test_tracks.csv")
new_df = track_video(VIDEO_PATH, tracks_csv, num_animals=2, show_display=False)

check("track_video's output schema is unchanged (frame/time_s/mouse_id/x/y/status)",
      list(new_df.columns) == ["frame", "time_s", "mouse_id", "x", "y", "status"])
check("status still includes 'partial' for the occluded stretch (mouse B genuinely missing)",
      "partial" in set(new_df["status"]))

mean_y = new_df.groupby("mouse_id")["y"].mean()
b_id = mean_y.idxmax()
b_rows = new_df[new_df["mouse_id"] == b_id].sort_values("frame")
new_B = list(zip(b_rows["x"].to_numpy(), b_rows["y"].to_numpy()))

true_arr = np.array(true_B)
old_arr = np.array(old_B)
new_arr = np.array(new_B)
check(f"sanity: both approaches recovered {N_FRAMES} frames for mouse B",
      len(old_arr) == N_FRAMES and len(new_arr) == N_FRAMES)

old_occl_err = np.hypot(*(old_arr[OCCLUDE_START:OCCLUDE_END] - true_arr[OCCLUDE_START:OCCLUDE_END]).T).mean()
new_occl_err = np.hypot(*(new_arr[OCCLUDE_START:OCCLUDE_END] - true_arr[OCCLUDE_START:OCCLUDE_END]).T).mean()
check(f"OLD (freeze-on-partial) drifts far from the truth while occluded (mean error={old_occl_err:.1f}px)",
      old_occl_err > 10.0)
check(f"NEW (Kalman-predicted) tracks much closer to the truth through the SAME occlusion "
      f"(mean error={new_occl_err:.1f}px vs OLD's {old_occl_err:.1f}px)",
      new_occl_err < 3.0 and new_occl_err < old_occl_err / 5)

# The old behavior's tell-tale signature: a dead-flat trajectory (zero
# movement reported) for the whole occluded stretch, because it was
# frozen. The new one should show real, non-zero motion throughout,
# since it's extrapolating rather than holding still (until the frame
# count exceeds kalman_max_coast_frames, which this 20-frame occlusion
# does not).
old_occl_motion = np.hypot(*np.diff(old_arr[OCCLUDE_START:OCCLUDE_END], axis=0).T).sum()
new_occl_motion = np.hypot(*np.diff(new_arr[OCCLUDE_START:OCCLUDE_END], axis=0).T).sum()
check(f"OLD's reported position is frozen (near-zero total movement={old_occl_motion:.1f}px) while occluded",
      old_occl_motion < 1.0)
check(f"NEW's reported position keeps moving smoothly while occluded (total movement={new_occl_motion:.1f}px), "
      "instead of freezing",
      new_occl_motion > 20.0)

print()

# --------------------------------------------------------------------------
# 3) Coasting degrades gracefully for occlusions LONGER than
# kalman_max_coast_frames -- confirms this isn't unbounded extrapolation.
# --------------------------------------------------------------------------
LONG_OCCLUDE_START, LONG_OCCLUDE_END = 20, 70  # 50 frames, well past the default coast limit
frames_long = []
for t in range(N_FRAMES):
    ax, ay = 20 + 2.0 * t, 30
    bx, by = 20 + 2.0 * t, 130
    frame = np.full((VH, VW), 200, dtype=np.uint8)
    cv2.circle(frame, (int(ax), int(ay)), BLOB_R, 40, -1)
    if not (LONG_OCCLUDE_START <= t < LONG_OCCLUDE_END):
        cv2.circle(frame, (int(bx), int(by)), BLOB_R, 40, -1)
    frames_long.append(frame)

VIDEO_PATH_LONG = os.path.join(tempfile.gettempdir(), "kalman_tracking_test_video_long.mp4")
writer = cv2.VideoWriter(VIDEO_PATH_LONG, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (VW, VH))
for f in frames_long:
    writer.write(cv2.cvtColor(f, cv2.COLOR_GRAY2BGR))
writer.release()

long_df = track_video(VIDEO_PATH_LONG, os.path.join(tempfile.gettempdir(), "kalman_long_tracks.csv"),
                       num_animals=2, show_display=False,
                       kalman_max_coast_frames=DEFAULTS["kalman_max_coast_frames"])
mean_y_long = long_df.groupby("mouse_id")["y"].mean()
b_id_long = mean_y_long.idxmax()
b_rows_long = long_df[long_df["mouse_id"] == b_id_long].sort_values("frame")
b_xs_long = b_rows_long["x"].to_numpy()

coast_limit = DEFAULTS["kalman_max_coast_frames"]
# Once coasting has run past the limit, the reported x should stop
# advancing (velocity was zeroed) -- i.e. the last few frames of the long
# occlusion should show near-zero movement, unlike the short-occlusion
# case above which kept moving throughout.
tail_motion = np.abs(np.diff(b_xs_long[LONG_OCCLUDE_START + coast_limit + 3:LONG_OCCLUDE_END])).sum()
check(f"an occlusion much longer than kalman_max_coast_frames ({coast_limit}) eventually "
      f"stops extrapolating and holds still (tail movement={tail_motion:.1f}px)",
      tail_motion < 1.0)

# ...but once the mouse is visible again (frame >= LONG_OCCLUDE_END), a
# real detection snaps the filter back onto the truth -- it isn't stuck
# drifted forever just because it coasted for a while.
true_x_at_last_frame = 20 + 2.0 * (N_FRAMES - 1)
reported_x_at_last_frame = b_rows_long.iloc[-1]["x"]
check(f"...but tracking recovers and matches the truth again once re-detected "
      f"(reported x={reported_x_at_last_frame:.1f} vs true x={true_x_at_last_frame:.1f})",
      abs(reported_x_at_last_frame - true_x_at_last_frame) < 5.0)

print()

# --------------------------------------------------------------------------
# 4) Backward compatibility: an existing caller that doesn't know about
# the new kalman_* settings at all must still work, using the defaults.
# --------------------------------------------------------------------------
plain_df = track_video(VIDEO_PATH, os.path.join(tempfile.gettempdir(), "kalman_plain_tracks.csv"),
                        num_animals=2, show_display=False)
check("track_video runs fine with no kalman_* overrides at all (uses DEFAULTS)",
      plain_df is not None and len(plain_df) > 0)

bins_df = calculate_time_bins(new_df, fps=FPS, bin_size_s=1.0)
check("calculate_time_bins still works against the new (Kalman-smoothed) tracks_df output",
      len(bins_df) > 0 and "Distance_pixels" in bins_df.columns)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    import sys
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
