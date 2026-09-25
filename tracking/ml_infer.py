"""
Runs a trained tracking.ml_model.BehaviorClipClassifier over a video and
emits a bouts DataFrame in the exact shape
tracking.behavior.classify_behaviors()/manual_score_video() already
produce: subject, behavior, start_frame, stop_frame, start_s, stop_s,
duration_s, confidence[, source]. The results screen, CSV export, and
downstream analysis (calculate_bins, tag_interaction_bouts, ...) don't
need to know whether a bout came from the heuristic classifier, manual
scoring, or this optional deep-learning model -- that's the whole point
of matching the schema.

Needs PyTorch -- import is guarded, see tracking.ml_train for why.
"""

try:
    import torch
    from tracking.ml_model import BehaviorClipClassifier  # itself needs torch -- keep inside the guard
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import cv2
import numpy as np
import pandas as pd

from tracking.ml_train import _load_clip_frames  # same decode/resize path used in training, no torch needed

BOUTS_COLUMNS = ["subject", "behavior", "start_frame", "stop_frame",
                 "start_s", "stop_s", "duration_s", "confidence", "source"]


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. Install it (see https://pytorch.org) "
            "to use the deep-learning behavior classifier, or use the "
            "automatic/manual classifier instead -- everything else in this "
            "app works without PyTorch."
        )


def load_model(checkpoint_path, device=None):
    """Loads a .pt file saved by tracking.ml_train.train_model() and
    returns (model, device) ready for classify_video_ml() or predict_clip()."""
    _require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = BehaviorClipClassifier.from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    return model, device


def predict_clip(model, device, clip_path, resize=(64, 64)):
    """Runs one already-cut mp4 clip (e.g. one of tracking.ml_dataset's
    output files, or anything the same fixed length) through the model.
    Returns (label, confidence, {label: probability, ...}) -- handy for
    spot-checking a trained model against a few known clips before
    trusting it on a full video."""
    _require_torch()
    frames = _load_clip_frames(clip_path, resize=resize)
    if len(frames) == 0:
        raise RuntimeError(f"Could not read any frames from clip: {clip_path}")
    clip = frames.astype(np.float32) / 255.0
    clip = np.ascontiguousarray(np.transpose(clip, (0, 3, 1, 2)))  # T,H,W,C -> T,C,H,W
    x = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1,T,C,H,W)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_idx = int(torch.argmax(probs).item())
    prob_map = {name: float(p) for name, p in zip(model.class_names, probs.tolist())}
    return model.class_names[top_idx], prob_map[model.class_names[top_idx]], prob_map


def classify_video_ml(
    video_path, checkpoint_path, subject="mouse1",
    window_frames=45, stride_frames=None, resize=(64, 64),
    crop=None, min_confidence=0.0, other_label="other",
    device=None, fps_override=None, progress_callback=None,
):
    """Slides a window_frames-long window over the whole video (default
    non-overlapping: stride_frames=window_frames when not given),
    classifies each window with the trained model, and merges consecutive
    windows sharing the same predicted label into bouts -- the same idea
    as tracking.behavior.labels_to_bouts turning per-frame labels into
    bouts, just at window granularity, since a clip classifier has no
    finer time resolution than its own window.

    Windows predicted as `other_label`, and windows whose top prediction
    falls below `min_confidence`, are dropped and also break a run in
    progress -- a model isn't asked to grade regions with nothing to
    say, and a dropped window between two same-label windows correctly
    prevents them from being merged into one falsely-continuous bout.

    Returns a DataFrame with tracking.behavior's bout columns plus
    source="ml" (an empty, correctly-columned one if the video has fewer
    than window_frames frames, or nothing survives min_confidence) --
    this never raises for "no detections found", only for real errors
    (bad paths, missing PyTorch, an unreadable checkpoint).
    """
    _require_torch()
    model, device = load_model(checkpoint_path, device=device)
    class_names = model.class_names
    stride_frames = stride_frames or window_frames

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames < window_frames:
        return pd.DataFrame(columns=BOUTS_COLUMNS)

    starts = list(range(0, total_frames - window_frames + 1, stride_frames))

    # (window_index, start_frame, stop_frame, label, confidence) for every
    # window that survives the other_label/min_confidence filter -- kept
    # as a flat list so the merge pass below can detect a gap (a filtered-
    # out window in between) by window_index, not just frame adjacency.
    kept = []
    cap = cv2.VideoCapture(video_path)
    for w_idx, f0 in enumerate(starts):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        frames = []
        for _ in range(window_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if crop is not None:
                x1, y1, x2, y2 = crop
                frame = frame[y1:y2, x1:x2]
            frames.append(cv2.resize(frame, resize, interpolation=cv2.INTER_AREA))
        if len(frames) < window_frames:
            break  # ran off the end mid-window

        clip = np.stack(frames, axis=0).astype(np.float32) / 255.0
        clip = np.ascontiguousarray(np.transpose(clip, (0, 3, 1, 2)))
        x = torch.from_numpy(clip).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0]
        top_idx = int(torch.argmax(probs).item())
        top_label = class_names[top_idx]
        top_conf = float(probs[top_idx].item())

        if progress_callback:
            progress_callback(w_idx + 1, len(starts))

        if top_label == other_label or top_conf < min_confidence:
            continue
        kept.append((w_idx, f0, f0 + window_frames - 1, top_label, top_conf))
    cap.release()

    if not kept:
        return pd.DataFrame(columns=BOUTS_COLUMNS)

    bouts = []
    run_idx, run_start, run_stop, run_label, run_confs = kept[0][0], kept[0][1], kept[0][2], kept[0][3], [kept[0][4]]
    for (w_idx, f0, f1, label, conf) in kept[1:]:
        if label == run_label and w_idx == run_idx + 1:
            run_stop = f1
            run_confs.append(conf)
        else:
            bouts.append((run_start, run_stop, run_label, run_confs))
            run_start, run_stop, run_label, run_confs = f0, f1, label, [conf]
        run_idx = w_idx
    bouts.append((run_start, run_stop, run_label, run_confs))

    rows = []
    for (start_frame, stop_frame, label, confs) in bouts:
        start_s = start_frame / fps
        stop_s = (stop_frame + 1) / fps
        rows.append(dict(
            subject=subject, behavior=label,
            start_frame=start_frame, stop_frame=stop_frame,
            start_s=round(start_s, 3), stop_s=round(stop_s, 3),
            duration_s=round(stop_s - start_s, 3),
            confidence=round(float(np.mean(confs)), 2),
            source="ml",
        ))

    return pd.DataFrame(rows, columns=BOUTS_COLUMNS)
