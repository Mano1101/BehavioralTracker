"""
Trains tracking.ml_model.BehaviorClipClassifier on a clip dataset built by
tracking.ml_dataset.build_clip_dataset.

This is the one place PyTorch training actually happens. Needs PyTorch --
the import is guarded (TORCH_AVAILABLE) so the rest of the app keeps
working with PyTorch not installed at all; train_model() raises a clear,
actionable error instead of an ImportError traceback if it's missing.

Meant to be run on the researcher's OWN machine (ideally with a GPU) once
they've labeled enough footage with the app's existing manual-scoring
tool and built a dataset from it with tracking.ml_dataset. This sandbox
has no GPU -- CPU-only PyTorch here is only for verifying the code runs
correctly end-to-end, not for real training speed.
"""

import os
import glob
import random

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, Subset
    from tracking.ml_model import BehaviorClipClassifier  # itself needs torch -- keep inside the guard
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import cv2
import numpy as np


def _require_torch():
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed. The deep-learning behavior classifier "
            "is optional -- install it with e.g. `pip install torch` (see "
            "https://pytorch.org for the right command for your machine/GPU) "
            "to use this feature. Everything else in this app works without it."
        )


def _load_clip_frames(path, resize=(64, 64)):
    """mp4 clip -> (T, H, W, 3) uint8 array, BGR (matches how ml_dataset
    wrote the clip and how classify_video_ml reads live video -- OpenCV is
    BGR throughout this app, so nothing needs converting)."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames, axis=0) if frames else np.zeros((0, *resize, 3), dtype=np.uint8)


if TORCH_AVAILABLE:

    class ClipDataset(Dataset):
        """One folder-of-mp4-clips-per-class dataset dir (see
        tracking.ml_dataset), decoded lazily on __getitem__ rather than
        all loaded into memory up front -- fine for the few hundred short
        clips a real lab dataset is likely to be."""

        def __init__(self, dataset_dir, class_names, resize=(64, 64), augment=False):
            self.resize = resize
            self.augment = augment
            self.class_names = list(class_names)
            self.samples = []  # (path, class_idx)
            for idx, cls in enumerate(self.class_names):
                cls_dir = os.path.join(dataset_dir, cls)
                for path in sorted(glob.glob(os.path.join(cls_dir, "clip_*.mp4"))):
                    self.samples.append((path, idx))
            if not self.samples:
                raise ValueError(f"No clips found under {dataset_dir} for classes {self.class_names}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            path, label = self.samples[i]
            frames = _load_clip_frames(path, resize=self.resize)
            if self.augment and random.random() < 0.5:
                frames = frames[:, :, ::-1, :]  # horizontal flip
            frames = frames.astype(np.float32) / 255.0
            frames = np.ascontiguousarray(np.transpose(frames, (0, 3, 1, 2)))  # T,H,W,C -> T,C,H,W
            return torch.from_numpy(frames), label

    def _collate(batch):
        # Every clip shares T (ml_dataset's fixed window_frames) and H,W
        # (ClipDataset's resize), so a plain stack is enough.
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys


def train_model(
    dataset_dir, class_names, output_path,
    epochs=15, batch_size=8, lr=1e-3, val_fraction=0.2,
    resize=(64, 64), device=None, seed=0,
    progress_callback=None,
):
    """Trains a BehaviorClipClassifier on dataset_dir (built by
    tracking.ml_dataset.build_clip_dataset) and saves the best
    (lowest validation loss) checkpoint to output_path.

    class_names fixes the class order baked into the saved model -- and
    therefore into every later inference call -- so pass the same list
    used to build the dataset (e.g. ["grooming", "rearing", "other"]).

    progress_callback(epoch, epochs, train_loss, val_loss, val_acc), if
    given, is called after every epoch -- e.g. to drive a GUI progress
    bar, since a multi-minute-or-longer training run has no other way to
    report progress mid-run.

    With too few clips for a meaningful held-out split (< 5 total),
    training falls back to using the training loss/accuracy itself as
    the checkpoint criterion, since there's nothing to hold out -- this
    is clearly optimistic and is meant for a quick sanity-check run, not
    a dataset that small being genuinely trustworthy.

    Returns (output_path, best_val_acc).
    """
    _require_torch()
    random.seed(seed)
    torch.manual_seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    full = ClipDataset(dataset_dir, class_names, resize=resize, augment=False)
    n_val = max(1, int(round(len(full) * val_fraction))) if len(full) >= 5 else 0
    n_train = len(full) - n_val

    indices = list(range(len(full)))
    random.Random(seed).shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_ds = ClipDataset(dataset_dir, class_names, resize=resize, augment=True)
    train_subset = Subset(train_ds, train_idx)
    val_subset = Subset(full, val_idx) if val_idx else None

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=_collate) if val_subset else None

    model = BehaviorClipClassifier(class_names).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)
            train_n += xb.size(0)
        train_loss = train_loss_sum / max(1, train_n)

        if val_loader is not None:
            model.eval()
            val_loss_sum, val_n, val_correct = 0.0, 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    val_loss_sum += loss.item() * xb.size(0)
                    val_n += xb.size(0)
                    val_correct += (logits.argmax(dim=1) == yb).sum().item()
            val_loss = val_loss_sum / max(1, val_n)
            val_acc = val_correct / max(1, val_n)
        else:
            val_loss = train_loss
            model.eval()
            with torch.no_grad():
                correct, total = 0, 0
                for xb, yb in DataLoader(train_subset, batch_size=batch_size, collate_fn=_collate):
                    xb, yb = xb.to(device), yb.to(device)
                    correct += (model(xb).argmax(dim=1) == yb).sum().item()
                    total += xb.size(0)
            val_acc = correct / max(1, total)

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.to_checkpoint(), output_path)

        if progress_callback:
            progress_callback(epoch, epochs, train_loss, val_loss, val_acc)

    return output_path, best_val_acc
