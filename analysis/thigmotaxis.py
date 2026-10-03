"""Thigmotaxis (wall-proximity / border preference) analysis for open-field
tests -- a standard readout the commercial packages (EthoVision XT,
ANY-maze) report by default: how much of the session the animal spent in a
border band along the arena wall versus the interior center.

Works directly on the raw per-frame tracking table (raw_tracking.csv shape)
so it can run either inside the per-video pipeline or as a re-analysis on
an already-exported CSV. The border band is an axis-aligned rectangle of
`margin_px` pixels inside the (post-perspective-warp) arena bounds.
"""

import numpy as np


def thigmotaxis_metrics(df, arena_w, arena_h, margin_px, fps,
                        x_col="Mouse_X", y_col="Mouse_Y", status_col="Tracking_Status"):
    """Return border/center occupancy metrics for one tracked session.

    Parameters
    ----------
    df : per-frame tracking DataFrame (must contain x/y, Time, status cols)
    arena_w, arena_h : post-warp arena dimensions in pixels
    margin_px : border band thickness in pixels
    fps : frames per second of the source video
    """
    valid = df[df[status_col] == "Tracked"]
    x = valid[x_col].to_numpy(dtype=float)
    y = valid[y_col].to_numpy(dtype=float)
    if len(x) == 0:
        return {}

    border = ((x < margin_px) | (x > arena_w - margin_px) |
              (y < margin_px) | (y > arena_h - margin_px))

    # Distance is only counted on runs that stay inside the same region, so
    # a segment straddling the border/center line is not double-counted.
    seg = np.hypot(np.diff(x), np.diff(y))
    both_border = border[:-1] & border[1:]
    border_dist = float(seg[both_border].sum())
    total_dist = float(seg.sum())

    return {
        "margin_px": int(margin_px),
        "border": {
            "time_s": float(border.sum() / fps),
            "percent": float(border.mean() * 100),
            "distance_px": border_dist,
            "distance_percent": float(border_dist / total_dist * 100) if total_dist > 0 else 0.0,
            "entries": int(np.count_nonzero(np.diff(border.astype(int)) == 1)),
        },
        "center": {
            "time_s": float((~border).sum() / fps),
            "percent": float((~border).mean() * 100),
            "distance_px": float(total_dist - border_dist),
            "distance_percent": float((total_dist - border_dist) / total_dist * 100) if total_dist > 0 else 0.0,
        },
    }


def flatten_for_summary(metrics, prefix="Thigmotaxis"):
    """Flatten thigmotaxis_metrics() output into summary-dict columns
    (border_time_s, border_percent, center_time_s, ...) so a per-video
    pipeline can merge it straight into its Summary sheet."""
    if not metrics:
        return {}
    b, c = metrics["border"], metrics["center"]
    return {
        f"{prefix}_border_time_s": b["time_s"],
        f"{prefix}_border_percent": b["percent"],
        f"{prefix}_border_distance_px": b["distance_px"],
        f"{prefix}_border_entries": b["entries"],
        f"{prefix}_center_time_s": c["time_s"],
        f"{prefix}_center_percent": c["percent"],
    }
