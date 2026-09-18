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
                frame's two centroids to last frame's two centroids so that
                total movement is minimized. This is what keeps "mouse_A"
                referring to the same physical animal across the video.

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

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

DEFAULTS = dict(
    n_background_samples=40,   # frames used to build the background model
    diff_threshold=25,         # 0-255, intensity difference to call "foreground"
    min_area=150,              # px^2, blobs smaller than this are noise
    max_area=3000,             # px^2, roughly one mouse's footprint -- TUNE THIS
    morph_kernel=5,            # size of morphological open/close kernel
    num_animals=2,
    color_mode="gray",         # "gray" (default) or "rgb" -- see build_background/foreground_mask
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


def foreground_mask(frame, background, diff_threshold, morph_kernel, color_mode="gray"):
    """Absolute-difference from background -> threshold -> clean with
    morphology. Returns a binary (0/255) mask of "moved" pixels.

    In RGB mode, frame/background are 3-channel BGR images; the per-pixel
    difference is the MAX across channels rather than a single grayscale
    value, so an animal that differs in color but not brightness still
    registers as foreground."""
    diff = cv2.absdiff(frame, background)
    if color_mode == "rgb" and diff.ndim == 3:
        diff = diff.max(axis=2)
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove speckle noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    return mask


# --------------------------------------------------------------------------
# STAGE 2: SPLIT (merged-blob handling)
# --------------------------------------------------------------------------

def split_merged_blob(contour, mask_shape, k=2):
    """A single contour that's too big to be one mouse is assumed to be `k`
    overlapping mice. Cluster its foreground pixel coordinates into k groups
    with k-means and return their centers as the k centroids."""
    blob_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(blob_mask, [contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(blob_mask == 255)
    pts = np.column_stack([xs, ys]).astype(np.float32)

    if len(pts) < k:
        cx, cy = (pts.mean(axis=0) if len(pts) else (np.nan, np.nan))
        return [(cx, cy)] * k

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, _, centers = cv2.kmeans(pts, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return [tuple(c) for c in centers]


def get_centroids(mask, min_area, max_area, num_animals=2):
    """Always tries to return `num_animals` (x, y) centroids for this frame.
    Handles three cases: two clean blobs, one merged blob (split it), or
    zero/partial detections (caller decides how to fill the gap)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    contours.sort(key=cv2.contourArea, reverse=True)

    if len(contours) == 0:
        return None  # nothing detected this frame

    if len(contours) == 1:
        area = cv2.contourArea(contours[0])
        if area > max_area:
            return split_merged_blob(contours[0], mask.shape, k=num_animals)
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
# STAGE 3: IDENTIFY (Hungarian algorithm)
# --------------------------------------------------------------------------

def match_identities(prev_centroids, curr_centroids):
    """Reorder curr_centroids to best match prev_centroids' identities by
    minimizing total displacement (Hungarian / linear sum assignment)."""
    cost = np.zeros((len(prev_centroids), len(curr_centroids)))
    for i, p in enumerate(prev_centroids):
        for j, c in enumerate(curr_centroids):
            cost[i, j] = np.hypot(p[0] - c[0], p[1] - c[1])
    row_ind, col_ind = linear_sum_assignment(cost)
    ordered = list(prev_centroids)  # fallback: hold last position if unmatched
    for r, c in zip(row_ind, col_ind):
        ordered[r] = curr_centroids[c]
    return ordered


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
                 show_display=False, **overrides):
    cfg = {**DEFAULTS, **overrides}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_animals = cfg["num_animals"]

    print("Building background model...")
    background = build_background(cap, cfg["n_background_samples"], color_mode=cfg["color_mode"])

    writer = None
    if annotate_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotate_path), fourcc, fps, (w, h))

    colors = [(0, 0, 255), (255, 0, 0), (0, 200, 0), (0, 200, 200)]  # BGR
    trails = {i: [] for i in range(n_animals)}

    records = []
    prev_centroids = None
    stopped_early = False

    frame_iter = range(n_frames) if (show_display or progress_callback) else tqdm(range(n_frames), desc="Tracking")

    for frame_idx in frame_iter:
        ok, frame = cap.read()
        if not ok:
            break
        frame_for_diff = frame if cfg["color_mode"] == "rgb" else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = foreground_mask(frame_for_diff, background, cfg["diff_threshold"], cfg["morph_kernel"],
                                color_mode=cfg["color_mode"])
        detected = get_centroids(mask, cfg["min_area"], cfg["max_area"], n_animals)

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
            if writer is not None:
                writer.write(annotated)
            if show_display:
                cv2.putText(annotated, f"Frame: {frame_idx}/{n_frames}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.imshow("MULTI-MOUSE TRACKING - Q to stop", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    stopped_early = True

        if progress_callback and (frame_idx % 5 == 0 or frame_idx == n_frames - 1):
            progress_callback(min(1.0, (frame_idx + 1) / max(1, n_frames)))

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

    n_lost = (df["status"] == "lost").sum()
    n_partial = (df["status"] == "partial").sum()
    print(f"Saved {len(df)} rows -> {output_csv}")
    print(f"  status counts: ok={(df['status'] == 'ok').sum()}, "
          f"partial={n_partial}, lost={n_lost}")
    if annotate_path:
        print(f"Annotated preview -> {annotate_path}  (watch this for ID swaps)")
    return df


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
