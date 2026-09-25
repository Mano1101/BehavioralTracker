"""
The OPTIONAL deep-learning behavior classifier's model definition.

Everywhere else in this app is classical computer vision -- no training
data, no GPU, deterministic geometry. This is the one corner that isn't:
a small per-frame CNN feeds a GRU over time, so the model sees both what a
single frame looks like (a hunched, curled-up posture; a reared-up
silhouette) and how it changes over the clip (grooming's small repeated
paw motion; rearing's brief vertical hold) -- either cue alone is
ambiguous, together they're what actually tells the behaviors apart.

Deliberately small: a per-frame CNN + GRU (not a heavier 3D-CNN or video
transformer) fits a realistic lab amount of labeled data -- tens to a few
hundred scored bouts, sliced into clips by tracking.ml_dataset -- without
overfitting immediately, and trains in a reasonable time on a single GPU
(or even CPU, slowly, for a smoke test).

Importing this module requires PyTorch. Nothing else in the app imports it
eagerly -- only tracking.ml_train (training) and tracking.ml_infer
(running a trained model) do, and both guard the import so the rest of
the app keeps working with PyTorch not installed at all.
"""

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU -> 2x2 max-pool, halving spatial size."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(torch.relu(self.bn(self.conv(x))))


class BehaviorClipClassifier(nn.Module):
    """Classifies a short fixed-length video clip (as produced by
    tracking.ml_dataset.build_clip_dataset) into one of `class_names`.

    Input:  (B, T, C, H, W) float tensor, pixel values in [0, 1], C=3 (BGR
            or RGB, doesn't matter as long as train/infer agree -- see
            tracking.ml_infer for the exact preprocessing it must match).
    Output: (B, num_classes) raw logits (use softmax/argmax to interpret).

    Each of the T frames is run through the same small CNN (weights
    shared across time -- a grooming posture looks the same whether it's
    frame 3 or frame 30 of the clip), pooled down to one feature vector
    per frame, then a GRU reads the T-length sequence of those vectors so
    the model can use motion over time, not just a single frame's shape.
    Only the GRU's final hidden state feeds the classification head --
    it's a summary of the whole clip, not any one frame.
    """

    def __init__(self, class_names, in_channels=3, cnn_channels=(16, 32, 64, 64),
                 gru_hidden=128, dropout=0.3):
        super().__init__()
        self.class_names = list(class_names)

        chans = [in_channels] + list(cnn_channels)
        self.cnn = nn.Sequential(*[
            _ConvBlock(chans[i], chans[i + 1]) for i in range(len(chans) - 1)
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> (B*T, cnn_channels[-1], 1, 1)
        feat_dim = cnn_channels[-1]

        self.gru = nn.GRU(input_size=feat_dim, hidden_size=gru_hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gru_hidden, len(self.class_names))

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        frames = x.reshape(b * t, c, h, w)
        feats = self.cnn(frames)              # (B*T, feat_dim, H', W')
        feats = self.pool(feats).reshape(b, t, -1)  # (B, T, feat_dim)

        _, h_n = self.gru(feats)               # h_n: (1, B, gru_hidden)
        clip_feat = self.dropout(h_n.squeeze(0))  # (B, gru_hidden)
        return self.head(clip_feat)            # (B, num_classes)

    def predict_proba(self, x):
        """Convenience for inference: (B, T, C, H, W) -> (B, num_classes)
        softmax probabilities, no gradient tracking."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)

    def to_checkpoint(self):
        """Everything ml_infer.py needs to reconstruct this exact model
        later, bundled with the trained weights -- so a saved .pt file is
        self-describing and doesn't depend on remembering training-time
        settings (class order and CNN width both affect the weight
        shapes, so they must round-trip exactly)."""
        return {
            "class_names": self.class_names,
            "in_channels": self.cnn[0].conv.in_channels,
            "cnn_channels": tuple(block.conv.out_channels for block in self.cnn),
            "gru_hidden": self.gru.hidden_size,
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        model = cls(
            class_names=checkpoint["class_names"],
            in_channels=checkpoint.get("in_channels", 3),
            cnn_channels=checkpoint.get("cnn_channels", (16, 32, 64, 64)),
            gru_hidden=checkpoint.get("gru_hidden", 128),
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model
