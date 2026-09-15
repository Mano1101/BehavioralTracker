"""Trajectory and occupancy-heatmap plots. Distance/velocity/zone-time/
alternation charts are planned for a later step (Step 6), not this one."""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle

from tracking.location import ROI_COLORS, OBJECT_COLORS


def save_plots(df, output_dir, roi_points, object_points, warp_w, warp_h):
    valid = df[df["Tracking_Status"] == "Tracked"].copy()
    if valid.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(valid["Mouse_X"], valid["Mouse_Y"], linewidth=0.8)

    for i, (name, pts) in enumerate(roi_points.items()):
        color = np.array(ROI_COLORS[i % len(ROI_COLORS)][::-1]) / 255.0
        pts_arr = np.array(list(pts) + [pts[0]])
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], linestyle="--", linewidth=1.2, label=name, color=color)

    for i, (name, spec) in enumerate(object_points.items()):
        color = np.array(OBJECT_COLORS[i % len(OBJECT_COLORS)][::-1]) / 255.0
        circle = MplCircle(spec["center"], spec["radius"], fill=False, linestyle=":", linewidth=1.4,
                            edgecolor=color, label=name)
        ax.add_patch(circle)

    if roi_points or object_points:
        ax.legend(loc="upper right", fontsize=8)

    ax.invert_yaxis()
    ax.set_xlim(0, warp_w)
    ax.set_ylim(warp_h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Mouse Trajectory")

    fig.savefig(os.path.join(output_dir, "trajectory.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    h = ax.hist2d(valid["Mouse_X"], valid["Mouse_Y"], bins=50)

    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Mouse Occupancy Heatmap")
    fig.colorbar(h[3], ax=ax, label="Frame count")

    fig.savefig(os.path.join(output_dir, "heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Calibration preview
# -----------------------------
