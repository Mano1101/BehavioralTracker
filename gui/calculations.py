"""Generic, paradigm-agnostic calculation helpers shared across modules."""

import math
import re


def point_distance(a, b):
    if a is None or b is None:
        return float("inf")
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def safe_col(name):
    """Sanitize an ROI/object/behavior name into a safe column header."""
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(name))
    cleaned = re.sub("_+", "_", cleaned).strip("_")
    return cleaned or "X"


# -----------------------------
# 4-point perspective crop
# -----------------------------
