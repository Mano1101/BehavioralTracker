"""
behavior_classifier.py
-----------------------
Automated grooming / rearing detection, built on top of the same
background-subtraction tracking as two_mouse_tracker.py. Outputs bout-level
events (subject, behavior, start, stop, duration) in the same shape as a
BORIS state-event export, so you can compare directly against manual scoring.

WHY TWO STAGES (extract, then classify)
----------------------------------------
Feature extraction (reading every video frame) is slow. Threshold tuning is
something you'll do many times. So this is split into:

  1. extract   - run once per video. Walks the video, tracks each animal,
                 and computes three per-frame features: velocity, contour
                 area (relative to that animal's own recent baseline), and
                 "local motion energy" (how much the pixels right around the
                 animal changed frame-to-frame). Saves these to a CSV.

  2. describe  - prints the distribution of those features so you can pick
                 sensible thresholds for stage 3, without re-processing video.

  3. classify  - reads the features CSV (fast, no video needed unless you
                 also want an annotated preview) and turns per-frame features
                 into behavior labels, then into bouts with durations.

THE HEURISTIC (read this before trusting the output)
------------------------------------------------------
This is a classical, feature-threshold classifier, not a trained model. It
labels each frame using this logic, per animal:

    velocity high                              -> locomotion
    velocity low  + area much below baseline   -> rearing
    velocity low  + high local motion energy   -> grooming
    velocity low  + low local motion energy    -> immobile
    (anything else)                            -> other

Why area for rearing: from a top-down camera, a mouse standing up on its
hind legs presents a smaller floor-projected silhouette than its normal
resting posture, so contour area drops.

Why local motion energy for grooming: grooming involves repeated small
limb/head movements IN PLACE. A small window of pixels centered on the
animal will show real frame-to-frame change even though the centroid isn't
translating -- that's different from true immobility (no change) and
different from locomotion (the whole window moves).

THIS IS A PROXY, NOT GROUND TRUTH. Known failure modes: sniffing in place or
scratching can look like grooming to this feature set; a hunched, motionless
posture can look like rearing if it shrinks the silhouette; lighting changes
shift the area/motion baselines. Precision comes from validation, not from
the algorithm alone -- see the README section on validating against BORIS.

USAGE
-----
    python behavior_classifier.py extract --video v.mp4 --output features.csv
    python behavior_classifier.py describe --features features.csv
    python behavior_classifier.py classify --features features.csv \
        --output bouts.csv --labeled-output labeled_frames.csv \
        --video v.mp4 --annotate preview.mp4 \
        --loco-thresh 40 --rear-area-drop 0.7 --groom-motion 3.0 --immobile-motion 1.0
"""

import argparse
import os
import glob

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from tracking.two_mouse import build_background, foreground_mask, DEFAULTS as TRACKER_DEFAULTS

DEFAULTS = dict(
    **TRACKER_DEFAULTS,
    roi_size=50,             # px, window size for local motion energy
    baseline_window_s=60.0,  # s, trailing window for each animal's "normal" area --
                             # keep this several times longer than the longest single
                             # bout you expect, or the baseline "catches up" to a
                             # sustained rearing/grooming bout and stops seeing it as
                             # different from normal (see README)
    loco_thresh=40.0,        # px/s, above this = locomotion
    rear_area_drop=0.7,      # area / baseline below this = candidate rearing
    groom_motion_thresh=3.0, # local motion energy above this (while slow) = grooming
    immobile_motion_thresh=1.0,  # local motion energy below this (while slow) = immobile
    min_bout_s=0.3,          # bouts shorter than this get merged into a neighbor
)


# --------------------------------------------------------------------------
# STAGE 1: EXTRACT (video -> per-frame features)
# --------------------------------------------------------------------------

def split_merged_blob_with_area(contour, mask_shape, k=2):
    """Like two_mouse_tracker's blob split, but also returns an area estimate
    per animal (the pixel count of its k-means cluster)."""
    blob_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(blob_mask == 255)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    if len(pts) < k:
        cx, cy = (pts.mean(axis=0) if len(pts) else (np.nan, np.nan))
        return [((cx, cy), len(pts) / k)] * k

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    return [(tuple(centers[i]), float(counts[i])) for i in range(k)]


def get_centroids_with_area(mask, min_area, max_area, num_animals=2):
    """Same detection logic as two_mouse_tracker.get_centroids, but returns
    (centroid, area) pairs instead of just centroids."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        return None

    if len(contours) == 1:
        area = cv2.contourArea(contours[0])
        if area > max_area:
            return split_merged_blob_with_area(contours[0], mask.shape, k=num_animals)
        M = cv2.moments(contours[0])
        if M["m00"] == 0:
            return None
        return [((M["m10"] / M["m00"], M["m01"] / M["m00"]), area)]

    out = []
    for c in contours[:num_animals]:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        out.append(((M["m10"] / M["m00"], M["m01"] / M["m00"]), cv2.contourArea(c)))
    return out if out else None


def match_identities_with_area(prev_items, curr_items):
    """Hungarian matching on position, carrying the area value along with
    whichever centroid gets assigned to which previous identity."""
    prev_pos = [p for p, _ in prev_items]
    curr_pos = [p for p, _ in curr_items]
    cost = np.zeros((len(prev_pos), len(curr_pos)))
    for i, p in enumerate(prev_pos):
        for j, c in enumerate(curr_pos):
            cost[i, j] = np.hypot(p[0] - c[0], p[1] - c[1])
    row_ind, col_ind = linear_sum_assignment(cost)
    ordered = list(prev_items)
    for r, c in zip(row_ind, col_ind):
        ordered[r] = curr_items[c]
    return ordered


def local_motion_energy(gray_prev, gray_curr, center, roi_size):
    """Mean absolute pixel change in a roi_size x roi_size window centered on
    `center`, between two consecutive grayscale frames. High while an animal
    moves limbs/head in place; near zero while truly still; also high during
    locomotion (but locomotion is already identified by velocity, so that's
    not a source of confusion in the classifier)."""
    x, y = center
    if np.isnan(x) or np.isnan(y):
        return np.nan
    half = roi_size // 2
    h, w = gray_curr.shape
    x0, x1 = max(0, int(x - half)), min(w, int(x + half))
    y0, y1 = max(0, int(y - half)), min(h, int(y + half))
    if x1 <= x0 or y1 <= y0:
        return np.nan
    prev_patch = gray_prev[y0:y1, x0:x1].astype(np.float32)
    curr_patch = gray_curr[y0:y1, x0:x1].astype(np.float32)
    return float(np.abs(curr_patch - prev_patch).mean())


def _load_background(video_path, background_source, n_background_samples, color_mode="gray"):
    """Background estimation is only reliable if the animal moves around
    enough during sampling that no pixel is 'mouse' most of the time. That
    assumption breaks for low-mobility footage -- e.g. a session with long
    grooming/rearing/freezing bouts, where the animal can occupy the same
    spot for a large share of the video. If you have a short clip or photo
    of the EMPTY arena (recommended: grab a frame before the animal is
    placed in), pass it as --background-source for a much more reliable
    background than auto-sampling from the animal's own video."""
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


def save_preview_frames(video_path, output_dir, n_samples=6, background_source=None, **overrides):
    """Save a handful of sample frames evenly spaced across the video, each
    annotated with the detected animal(s) at that frame -- lets you
    sanity-check min/max area and the difference threshold before running
    the full feature extraction, same idea as Standard Tracking's
    calibration preview. Returns the list of saved file paths."""
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
        detected = get_centroids_with_area(mask, cfg["min_area"], cfg["max_area"], n_animals)

        disp = frame.copy()
        if detected:
            for j, (pos, area) in enumerate(detected[:n_animals]):
                x, y = pos
                if np.isnan(x):
                    continue
                pt = (int(x), int(y))
                color = colors[j % len(colors)]
                cv2.circle(disp, pt, 8, color, 2)
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

    records = []
    prev_items = None
    prev_gray = None
    stopped_early = False
    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]  # BGR

    frame_iter = range(n_frames) if (show_display or progress_callback) \
        else tqdm(range(n_frames), desc="Extracting features")

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break
        # Position detection respects color_mode; local_motion_energy (used
        # only for the grooming feature) always works on grayscale -- it's
        # measuring local pixel-change magnitude, not color, so there's no
        # benefit to color there and it keeps that function simple.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_for_diff = frame if cfg["color_mode"] == "rgb" else gray
        mask = foreground_mask(frame_for_diff, background, cfg["diff_threshold"], cfg["morph_kernel"],
                                color_mode=cfg["color_mode"])
        detected = get_centroids_with_area(mask, cfg["min_area"], cfg["max_area"], n_animals)

        if detected is None:
            curr = prev_items if prev_items else [((np.nan, np.nan), np.nan)] * n_animals
            status = "lost"
        elif len(detected) < n_animals:
            curr = (detected + [detected[0]])[:n_animals] if prev_items is None \
                else match_identities_with_area(prev_items, detected)
            status = "partial"
        else:
            if prev_items is None:
                detected = sorted(detected, key=lambda item: item[0][0])  # seed left-to-right
                curr = detected[:n_animals]
            else:
                curr = match_identities_with_area(prev_items, detected)
            status = "ok"

        prev_items = curr

        for i, (pos, area) in enumerate(curr):
            x, y = pos
            me = local_motion_energy(prev_gray, gray, (x, y), cfg["roi_size"]) if prev_gray is not None else np.nan
            records.append(dict(
                frame=frame_idx, time_s=round(frame_idx / fps, 4),
                mouse_id=f"mouse_{chr(65 + i)}", x=x, y=y, area=area,
                motion_energy=me, status=status,
            ))
        prev_gray = gray

        if show_display:
            annotated = frame.copy()
            for i, (pos, _) in enumerate(curr):
                x, y = pos
                if not np.isnan(x):
                    pt = (int(x), int(y))
                    cv2.circle(annotated, pt, 6, colors[i % len(colors)], -1)
                    cv2.putText(annotated, chr(65 + i), (pt[0] + 8, pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i % len(colors)], 2)
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
    cfg = {**DEFAULTS, **overrides}
    df = pd.read_csv(features_csv)
    df = _add_derived_features(df, cfg["baseline_window_s"])
    for mouse_id, sub in df.groupby("mouse_id"):
        print(f"\n=== {mouse_id} ===")
        print(sub[["velocity", "area_ratio", "motion_energy"]].describe().round(2))
    print(
        "\nPick --loco-thresh a bit below the top of the 'velocity' range for "
        "genuine locomotion frames, --rear-area-drop a bit above the low end of "
        "'area_ratio' seen when you know the animal is upright/resting, and "
        "--groom-motion / --immobile-motion to split 'motion_energy' between "
        "clearly-still and clearly-moving-in-place frames."
    )


# --------------------------------------------------------------------------
# STAGE 2: CLASSIFY (features -> per-frame labels -> bouts)
# --------------------------------------------------------------------------

def _add_derived_features(df, baseline_window_s):
    out = []
    for mouse_id, sub in df.groupby("mouse_id"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        dt = sub["time_s"].diff().median()
        fps = 1.0 / dt if dt and dt > 0 else 30.0
        baseline_frames = max(1, int(baseline_window_s * fps))

        dx, dy = sub["x"].diff(), sub["y"].diff()
        dtimes = sub["time_s"].diff().replace(0, np.nan)
        sub["velocity"] = (np.hypot(dx, dy) / dtimes).fillna(0)

        sub["area_baseline"] = sub["area"].rolling(baseline_frames, min_periods=1).median()
        sub["area_ratio"] = sub["area"] / sub["area_baseline"]
        out.append(sub)
    return pd.concat(out, ignore_index=True)


def _label_frame(row, cfg):
    if row["status"] != "ok":
        return "undetermined"
    if row["velocity"] > cfg["loco_thresh"]:
        return "locomotion"
    if pd.notna(row["area_ratio"]) and row["area_ratio"] < cfg["rear_area_drop"]:
        return "rearing"
    if pd.notna(row["motion_energy"]) and row["motion_energy"] > cfg["groom_motion_thresh"]:
        return "grooming"
    if pd.notna(row["motion_energy"]) and row["motion_energy"] < cfg["immobile_motion_thresh"]:
        return "immobile"
    return "other"


def smooth_labels(labels, min_run):
    """Merge any run of identical labels shorter than min_run frames into a
    neighboring run, to remove single-frame flicker before counting bouts."""
    if not labels:
        return labels
    runs = []
    for lab in labels:
        if runs and runs[-1][0] == lab:
            runs[-1][1] += 1
        else:
            runs.append([lab, 1])

    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, (lab, length) in enumerate(runs):
            if length < min_run:
                target = i - 1 if i > 0 else i + 1
                runs[target][1] += length
                del runs[i]
                changed = True
                break

    out = []
    for lab, length in runs:
        out.extend([lab] * length)
    return out


def labels_to_bouts(sub, mouse_id):
    labels = sub["label_smoothed"].tolist()
    times = sub["time_s"].tolist()
    frames = sub["frame"].tolist()
    bouts = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            bouts.append(dict(
                subject=mouse_id,
                behavior=labels[start_idx],
                start_frame=frames[start_idx],
                stop_frame=frames[i - 1],
                start_s=times[start_idx],
                stop_s=times[i - 1],
                duration_s=round(times[i - 1] - times[start_idx], 3),
            ))
            start_idx = i
    return bouts


def classify_behaviors(features_csv, output_bouts_csv, labeled_output_csv=None, **overrides):
    cfg = {**DEFAULTS, **overrides}
    df = pd.read_csv(features_csv)
    df = _add_derived_features(df, cfg["baseline_window_s"])

    labeled_all, bouts_all = [], []
    for mouse_id, sub in df.groupby("mouse_id"):
        sub = sub.sort_values("frame").reset_index(drop=True)
        dt = sub["time_s"].diff().median()
        fps = 1.0 / dt if dt and dt > 0 else 30.0
        min_bout_frames = max(1, int(cfg["min_bout_s"] * fps))

        sub["label"] = sub.apply(lambda r: _label_frame(r, cfg), axis=1)
        sub["label_smoothed"] = smooth_labels(sub["label"].tolist(), min_bout_frames)

        labeled_all.append(sub)
        bouts_all.extend(labels_to_bouts(sub, mouse_id))

    labeled_df = pd.concat(labeled_all, ignore_index=True)
    bouts_df = pd.DataFrame(bouts_all)

    if labeled_output_csv:
        labeled_df.to_csv(labeled_output_csv, index=False)
        print(f"Saved per-frame labels -> {labeled_output_csv}")
    bouts_df.to_csv(output_bouts_csv, index=False)
    print(f"Saved {len(bouts_df)} bouts -> {output_bouts_csv}")

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
              "immobile": (128, 128, 128), "other": (0, 0, 0), "undetermined": (0, 0, 200)}

    for frame_idx in tqdm(range(n_frames), desc="Rendering annotated preview"):
        ok, frame = cap.read()
        if not ok:
            break
        for row in frame_lookup.get(frame_idx, []):
            if np.isnan(row.x):
                continue
            pt = (int(row.x), int(row.y))
            color = colors.get(row.label_smoothed, (255, 255, 255))
            cv2.circle(frame, pt, 6, color, -1)
            cv2.putText(frame, f"{row.mouse_id}: {row.label_smoothed}", (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"Annotated preview -> {output_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Automated grooming/rearing bout detection.")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("extract", help="Video -> per-frame feature CSV")
    pe.add_argument("--video", required=True)
    pe.add_argument("--output", default="features.csv")
    pe.add_argument("--num-animals", type=int, default=DEFAULTS["num_animals"])
    pe.add_argument("--min-area", type=int, default=DEFAULTS["min_area"])
    pe.add_argument("--max-area", type=int, default=DEFAULTS["max_area"])
    pe.add_argument("--diff-threshold", type=int, default=DEFAULTS["diff_threshold"])
    pe.add_argument("--bg-samples", type=int, default=DEFAULTS["n_background_samples"])
    pe.add_argument("--roi-size", type=int, default=DEFAULTS["roi_size"])
    pe.add_argument("--background-source", default=None,
                     help="Optional image or video of the EMPTY arena, for a background "
                          "model that isn't at risk of the animal's own low-mobility bouts "
                          "contaminating it. Strongly recommended if you have one.")

    pd_ = sub.add_parser("describe", help="Print feature distributions to help pick thresholds")
    pd_.add_argument("--features", required=True)
    pd_.add_argument("--baseline-window-s", type=float, default=DEFAULTS["baseline_window_s"])

    pc = sub.add_parser("classify", help="Feature CSV -> behavior bouts")
    pc.add_argument("--features", required=True)
    pc.add_argument("--output", default="bouts.csv")
    pc.add_argument("--labeled-output", default=None)
    pc.add_argument("--video", default=None, help="Needed only if --annotate is given")
    pc.add_argument("--annotate", default=None)
    pc.add_argument("--baseline-window-s", type=float, default=DEFAULTS["baseline_window_s"])
    pc.add_argument("--loco-thresh", type=float, default=DEFAULTS["loco_thresh"])
    pc.add_argument("--rear-area-drop", type=float, default=DEFAULTS["rear_area_drop"])
    pc.add_argument("--groom-motion", type=float, default=DEFAULTS["groom_motion_thresh"])
    pc.add_argument("--immobile-motion", type=float, default=DEFAULTS["immobile_motion_thresh"])
    pc.add_argument("--min-bout-s", type=float, default=DEFAULTS["min_bout_s"])

    args = p.parse_args()

    if args.command == "extract":
        extract_features(
            args.video, args.output, background_source=args.background_source,
            num_animals=args.num_animals, min_area=args.min_area,
            max_area=args.max_area, diff_threshold=args.diff_threshold,
            n_background_samples=args.bg_samples, roi_size=args.roi_size,
        )
    elif args.command == "describe":
        describe_features(args.features, baseline_window_s=args.baseline_window_s)
    elif args.command == "classify":
        labeled_df, _ = classify_behaviors(
            args.features, args.output, labeled_output_csv=args.labeled_output,
            baseline_window_s=args.baseline_window_s, loco_thresh=args.loco_thresh,
            rear_area_drop=args.rear_area_drop, groom_motion_thresh=args.groom_motion,
            immobile_motion_thresh=args.immobile_motion, min_bout_s=args.min_bout_s,
        )
        if args.video and args.annotate:
            render_annotated(args.video, labeled_df, args.annotate)


if __name__ == "__main__":
    main()
