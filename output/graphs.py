"""Trajectory and occupancy-heatmap plots, drawn on top of the actual
reference (background) frame -- matching ezTrack's convention of showing
results overlaid on the arena rather than a bare axis plot, so it's
immediately clear WHERE in the arena the animal spent its time."""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from tracking.location import ROI_COLORS, OBJECT_COLORS


def _reference_frame_rgb(background):
    """background may be grayscale (H,W) or color (H,W,3) BGR, depending on
    which Analysis Color Mode was used -- normalize either to an RGB image
    matplotlib can imshow."""
    if background is None:
        return None
    if background.ndim == 2:
        return cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


def save_plots(df, output_dir, roi_points, object_points, warp_w, warp_h, background=None):
    valid = df[df["Tracking_Status"] == "Tracked"].copy()
    if valid.empty:
        return

    ref_rgb = _reference_frame_rgb(background)

    fig, ax = plt.subplots(figsize=(10, 7))
    if ref_rgb is not None:
        ax.imshow(ref_rgb, extent=(0, warp_w, warp_h, 0))
    ax.plot(valid["Mouse_X"], valid["Mouse_Y"], linewidth=0.9, color="#00e5ff" if ref_rgb is not None else None)

    for i, (name, pts) in enumerate(roi_points.items()):
        color = np.array(ROI_COLORS[i % len(ROI_COLORS)][::-1]) / 255.0
        pts_arr = np.array(list(pts) + [pts[0]])
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], linestyle="--", linewidth=1.2, label=name, color=color)

    for i, (name, pts) in enumerate(object_points.items()):
        color = np.array(OBJECT_COLORS[i % len(OBJECT_COLORS)][::-1]) / 255.0
        pts_arr = np.array(list(pts) + [pts[0]])
        ax.plot(pts_arr[:, 0], pts_arr[:, 1], linestyle=":", linewidth=1.4, label=name, color=color)

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
    if ref_rgb is not None:
        ax.imshow(ref_rgb, extent=(0, warp_w, warp_h, 0))

    # A smoothed 2D occupancy density, overlaid semi-transparently so the
    # arena is still visible underneath -- low-occupancy cells fade to
    # fully transparent instead of covering the reference frame with a
    # solid low-value color.
    heat, xedges, yedges = np.histogram2d(
        valid["Mouse_X"], valid["Mouse_Y"], bins=50, range=[[0, warp_w], [0, warp_h]]
    )
    heat = gaussian_filter(heat, sigma=1.2).T
    heat_norm = heat / heat.max() if heat.max() > 0 else heat
    cmap = plt.get_cmap("inferno").copy()
    im = ax.imshow(heat_norm, extent=(0, warp_w, warp_h, 0), origin="upper",
                    cmap=cmap, alpha=np.clip(heat_norm * 1.3, 0, 0.85), interpolation="bilinear")

    ax.invert_yaxis()
    ax.set_xlim(0, warp_w)
    ax.set_ylim(warp_h, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title("Mouse Occupancy Heatmap")
    fig.colorbar(im, ax=ax, label="Relative occupancy")

    fig.savefig(os.path.join(output_dir, "heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Calibration preview
# -----------------------------
