"""
Regression test for "Trajectory smoothing / outlier filter" (Setup page,
Standard Tracking only) -- an optional post-processing pass that adds
Mouse_X_smooth/Mouse_Y_smooth columns (raw_tracking.csv keeps the real,
unfiltered Mouse_X/Mouse_Y too) and points the Trajectory/Heatmap charts at
whichever pair was requested. Mirrors SMART's Anti-Vibration/Anti-Artifact/
LOWESS filters (see the Upgrade Plan doc, Tier 1 #4).

Two things are covered:

1. tracking.location.smooth_trajectory() itself -- pure pandas, no video
   needed: a single-frame outlier gets pulled back toward its local
   neighborhood, overall jitter goes down, Lost frames stay NaN, and a
   window < 2 is a no-op.

2. process_single_video's wiring -- with setup["smooth_window"] unset, no
   *_smooth columns/flags appear at all (unchanged from before this
   feature existed); set, they do, and the chart files still get written.

Run directly (part 2 uses OpenCV video I/O like test_stop_conditions.py):
    python3.12 qt_app/tests/test_trajectory_smoothing.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

from tracking.location import process_single_video, identity_transform, smooth_trajectory

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ------------------------------------------------------------------
# 1) smooth_trajectory() -- pure function, synthetic DataFrame.
# ------------------------------------------------------------------
N = 40
rng = np.random.default_rng(0)
true_x = np.linspace(0, 100, N)  # a smooth, known underlying trend
noisy_x = true_x + rng.normal(0, 0.4, N)  # small realistic per-frame jitter
OUTLIER_IDX = 20
noisy_x[OUTLIER_IDX] += 40  # one big single-frame detection-noise spike

df = pd.DataFrame({
    "Tracking_Status": ["Tracked"] * N,
    "Mouse_X": noisy_x,
    "Mouse_Y": np.zeros(N),
})
# A couple of genuinely Lost frames in the middle -- must stay NaN/untouched.
df.loc[10:11, "Tracking_Status"] = "Lost"
df.loc[10:11, "Mouse_X"] = np.nan

smoothed_df = smooth_trajectory(df, window=5)

check("original Mouse_X column is left completely untouched",
      smoothed_df["Mouse_X"].equals(df["Mouse_X"]))
check("Mouse_X_smooth column was added", "Mouse_X_smooth" in smoothed_df.columns)
check("Lost frames stay NaN in the smoothed column too",
      smoothed_df.loc[10:11, "Mouse_X_smooth"].isna().all())

outlier_raw_error = abs(noisy_x[OUTLIER_IDX] - true_x[OUTLIER_IDX])
outlier_smooth_error = abs(smoothed_df.loc[OUTLIER_IDX, "Mouse_X_smooth"] - true_x[OUTLIER_IDX])
check("the single-frame outlier is pulled MUCH closer to the true underlying trend "
      "than the raw, noisy detection was",
      outlier_smooth_error < outlier_raw_error / 3)

tracked = df["Tracking_Status"] == "Tracked"
raw_jitter = np.diff(df.loc[tracked, "Mouse_X"].to_numpy(), n=2)
smooth_jitter = np.diff(smoothed_df.loc[tracked, "Mouse_X_smooth"].to_numpy(), n=2)
check("overall frame-to-frame jitter (2nd-difference energy) goes down after smoothing",
      np.nansum(smooth_jitter ** 2) < np.nansum(raw_jitter ** 2))

noop_df = smooth_trajectory(df, window=1)
check("window < 2 is a no-op -- the smoothed column just mirrors the raw one",
      noop_df["Mouse_X_smooth"].equals(noop_df["Mouse_X"]))

none_window_df = smooth_trajectory(df, window=None)
check("window=None is also a no-op",
      none_window_df["Mouse_X_smooth"].equals(none_window_df["Mouse_X"]))

# ------------------------------------------------------------------
# 2) process_single_video wiring -- a short synthetic video, run twice:
# once with smoothing off (default), once on.
# ------------------------------------------------------------------
W, H, FPS = 200, 160, 30
BG_VAL, BLOB_VAL, BLOB_R = 200, 40, 8

# A short, gently zig-zagging path (never revisiting the same pixel twice,
# same reasoning as test_stop_conditions.py's module docstring) so there's
# some real per-frame jitter for smoothing to act on, without tripping
# make_background's own median-background-ghosting edge case.
rng2 = np.random.default_rng(1)
xs = np.linspace(30, 170, 60) + rng2.normal(0, 2.0, 60)
ys = 80 + 15 * np.sin(np.linspace(0, 6, 60))
POSITIONS = list(zip(xs, ys))
TOTAL_FRAMES = len(POSITIONS)
DURATION_S = TOTAL_FRAMES / FPS

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "smoothing_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for (x, y) in POSITIONS:
    frame = np.full((H, W), BG_VAL, dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), BLOB_R, BLOB_VAL, -1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

matrix, warp_w, warp_h = identity_transform(W, H)


def make_setup(**overrides):
    setup = {
        "start_time": 0.0, "end_time": DURATION_S,
        "matrix": matrix, "warp_w": warp_w, "warp_h": warp_h,
        "arena": (0, 0, W, H),
        "roi_points": {}, "roi_names": [],
        "object_points": {}, "object_names": [],
        "interaction_margin_px": 0, "behavior_names": [],
        "background_samples": 30, "threshold": 30,
        "min_area": 20, "max_area": 2000, "max_jump": 400,
        "use_zone_threshold": False, "reject_shadows": False,
        "use_window": False, "window_size": 120, "window_weight": 0.5,
        "scale_factor": None, "scale_unit": None,
        "preview_samples": 3, "color_mode": "gray",
        "output_dir_override": None,
        "compute_arm_entries": False, "compute_alternation": False,
        "mask_points": [],
        "stop_condition": None, "stop_value": None, "stop_zone_name": "",
        "zone_groups": {}, "smooth_window": None,
    }
    setup.update(overrides)
    return setup


def run(label, **overrides):
    out_dir = tempfile.mkdtemp(prefix=f"smoothing_{label}_")
    setup = make_setup(output_dir_override=out_dir, **overrides)
    summary = process_single_video(
        VIDEO_PATH, setup, show_display=False, confirm_callback=lambda *a, **k: False
    )
    return summary, summary["Output_folder"]  # process_single_video nests video_name under output_dir_override


summary_off, out_off = run("off")
check("smoothing off -> Trajectory_smoothing_applied is False",
      summary_off["Trajectory_smoothing_applied"] is False)
raw_csv_off = pd.read_csv(os.path.join(out_off, "raw_tracking.csv"))
check("smoothing off -> raw_tracking.csv has no *_smooth columns at all",
      "Mouse_X_smooth" not in raw_csv_off.columns and "Mouse_Y_smooth" not in raw_csv_off.columns)
check("smoothing off -> trajectory.png still gets written",
      os.path.exists(os.path.join(out_off, "trajectory.png")))

summary_on, out_on = run("on", smooth_window=5)
check("smoothing on -> Trajectory_smoothing_applied is True",
      summary_on["Trajectory_smoothing_applied"] is True)
check("smoothing on -> the window is echoed back in the summary",
      summary_on.get("Trajectory_smoothing_window_frames") == 5)
raw_csv_on = pd.read_csv(os.path.join(out_on, "raw_tracking.csv"))
check("smoothing on -> raw_tracking.csv carries the *_smooth columns too",
      "Mouse_X_smooth" in raw_csv_on.columns and "Mouse_Y_smooth" in raw_csv_on.columns)
check("smoothing on -> raw_tracking.csv's ORIGINAL Mouse_X is unaffected (same tracked count)",
      raw_csv_on["Mouse_X"].notna().sum() == raw_csv_off["Mouse_X"].notna().sum())
check("smoothing on -> trajectory.png/heatmap.png still get written",
      os.path.exists(os.path.join(out_on, "trajectory.png"))
      and os.path.exists(os.path.join(out_on, "heatmap.png")))

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
