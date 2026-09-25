"""
Turns manually- or automatically-scored behavior bouts (see
tracking.behavior.classify_behaviors / manual_score_video, which both
write a bouts CSV with subject/behavior/start_frame/stop_frame/...
columns) into a labeled short-clip dataset for training the OPTIONAL
deep-learning behavior classifier (tracking.ml_model / tracking.ml_train).

This is the entry point into the one deep-learning-touching corner of the
app -- everywhere else is classical computer vision (no training data, no
GPU) by design. This module itself needs no extra dependency (just
opencv/pandas, already required) -- only tracking.ml_train and
tracking.ml_infer need PyTorch, and only when you actually train or run a
trained model.

Dataset layout (one folder per class, plain mp4 clips -- portable and easy
to spot-check by eye before trusting it to train on):

    <dataset_dir>/
        grooming/clip_000001.mp4
        grooming/clip_000002.mp4
        rearing/clip_000001.mp4
        ...
        other/clip_000001.mp4      <- sampled from time NOT covered by any
                                        labeled bout, so the model also
                                        learns what "neither" looks like

Re-running this on a newly-scored video ADDS to an existing dataset_dir
rather than overwriting it, so a dataset can be built up video by video as
you label more footage.
"""

import os
import glob

import cv2
import numpy as np
import pandas as pd


def _write_clip(cap, start_frame, n_frames, fps, out_path, crop=None):
    """Write n_frames starting at start_frame to out_path as an mp4.
    Returns True only if the full n_frames were written -- a clip that ran
    off the end of the video is deleted rather than kept short, since a
    fixed-length model input can't use a partial clip."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = None
    written = 0
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if crop is not None:
            x1, y1, x2, y2 = crop
            frame = frame[y1:y2, x1:x2]
        if writer is None:
            h, w = frame.shape[:2]
            if h <= 0 or w <= 0:
                break
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        writer.write(frame)
        written += 1
    if writer is not None:
        writer.release()
    if written < n_frames and os.path.exists(out_path):
        os.remove(out_path)
    return written == n_frames


def _next_clip_index(cls_dir):
    existing = glob.glob(os.path.join(cls_dir, "clip_*.mp4"))
    return len(existing) + 1


def build_clip_dataset(
    video_path, bouts_csv, dataset_dir, behaviors,
    window_frames=45, stride_frames=20, fps_override=None,
    other_label="other", other_per_bout=1.0, crop=None,
    progress_callback=None,
):
    """Slice video_path into fixed-length (window_frames) clips using the
    bout intervals in bouts_csv, one clip subfolder per requested behavior
    in `behaviors` (matched case-insensitively against the bouts'
    "behavior" column), plus an `other_label` folder sampled from
    everywhere NOT covered by any bout of interest -- negative examples,
    so the model learns what "neither" looks like instead of only ever
    seeing positive examples.

    window_frames/stride_frames: a behavior bout is usually longer than
    one training clip, so it's sliced into several overlapping
    (stride_frames < window_frames) or adjacent windows; a bout SHORTER
    than window_frames still gets exactly one centered window (clamped to
    stay inside the video) rather than being skipped.

    Returns {class_name: clip_count_added}. Raises FileNotFoundError if
    bouts_csv doesn't exist, and returns {} (no-op) if it has no rows
    matching `behaviors`.
    """
    if not os.path.exists(bouts_csv):
        raise FileNotFoundError(f"Bouts file not found: {bouts_csv}")
    bouts = pd.read_csv(bouts_csv)
    if bouts.empty:
        return {}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    wanted = {b.lower() for b in behaviors}
    bouts = bouts[bouts["behavior"].astype(str).str.lower().isin(wanted)].reset_index(drop=True)
    if bouts.empty:
        cap.release()
        return {}

    os.makedirs(dataset_dir, exist_ok=True)
    counts = {}
    covered = np.zeros(max(total_frames, 1), dtype=bool)

    for _, row in bouts.iterrows():
        behavior = str(row["behavior"]).lower()
        start_f, stop_f = int(row["start_frame"]), int(row["stop_frame"])
        covered[max(0, start_f):min(total_frames, stop_f + 1)] = True

        cls_dir = os.path.join(dataset_dir, behavior)
        os.makedirs(cls_dir, exist_ok=True)
        idx = _next_clip_index(cls_dir)
        added = 0

        if stop_f - start_f + 1 < window_frames:
            center = (start_f + stop_f) // 2
            f0 = max(0, min(max(0, total_frames - window_frames), center - window_frames // 2))
            out_path = os.path.join(cls_dir, f"clip_{idx:06d}.mp4")
            if _write_clip(cap, f0, window_frames, fps, out_path, crop=crop):
                added += 1
                idx += 1
        else:
            f = start_f
            while f + window_frames <= stop_f + 1:
                out_path = os.path.join(cls_dir, f"clip_{idx:06d}.mp4")
                if _write_clip(cap, f, window_frames, fps, out_path, crop=crop):
                    added += 1
                    idx += 1
                f += stride_frames

        counts[behavior] = counts.get(behavior, 0) + added
        if progress_callback:
            progress_callback(behavior, added)

    if other_label and other_per_bout > 0:
        n_other_target = int(round(sum(counts.values()) * other_per_bout))
        cls_dir = os.path.join(dataset_dir, other_label)
        os.makedirs(cls_dir, exist_ok=True)
        idx = _next_clip_index(cls_dir)
        added = 0
        rng = np.random.default_rng(0)
        candidates = list(range(0, max(1, total_frames - window_frames), stride_frames))
        rng.shuffle(candidates)
        for f0 in candidates:
            if added >= n_other_target:
                break
            if covered[f0:f0 + window_frames].any():
                continue
            out_path = os.path.join(cls_dir, f"clip_{idx:06d}.mp4")
            if _write_clip(cap, f0, window_frames, fps, out_path, crop=crop):
                added += 1
                idx += 1
        counts[other_label] = counts.get(other_label, 0) + added
        if progress_callback:
            progress_callback(other_label, added)

    cap.release()
    return counts


def dataset_summary(dataset_dir):
    """{class_name: clip_count} for an existing dataset_dir -- used by the
    GUI to show what's been collected so far before training."""
    if not os.path.isdir(dataset_dir):
        return {}
    out = {}
    for name in sorted(os.listdir(dataset_dir)):
        cls_dir = os.path.join(dataset_dir, name)
        if os.path.isdir(cls_dir):
            out[name] = len(glob.glob(os.path.join(cls_dir, "clip_*.mp4")))
    return out
