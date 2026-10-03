"""Trajectory and occupancy-heatmap plots, drawn on top of the actual
reference (background) frame -- matching ezTrack's convention of showing
results overlaid on the arena rather than a bare axis plot, so it's
immediately clear WHERE in the arena the animal spent its time."""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import gaussian_filter

from tracking.colors import ROI_COLORS, OBJECT_COLORS


def _reference_frame_rgb(background):
    """background may be grayscale (H,W) or color (H,W,3) BGR, depending on
    which Analysis Color Mode was used -- normalize either to an RGB image
    matplotlib can imshow."""
    if background is None:
        return None
    if background.ndim == 2:
        return cv2.cvtColor(background, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)


def save_plots(df, output_dir, roi_points, object_points, warp_w, warp_h, background=None,
               x_col="Mouse_X", y_col="Mouse_Y"):
    valid = df[df["Tracking_Status"] == "Tracked"].copy()
    if valid.empty:
        return

    ref_rgb = _reference_frame_rgb(background)

    fig, ax = plt.subplots(figsize=(10, 7))
    if ref_rgb is not None:
        ax.imshow(ref_rgb, extent=(0, warp_w, warp_h, 0))

    # Color the path by elapsed time (ezTrack's own signature look for a
    # location trace) rather than one flat color, so direction of travel
    # and where-the-animal-was-when are both visible at a glance -- a flat
    # line only shows the SHAPE of the path, not its time course.
    # x_col/y_col default to the raw Mouse_X/Mouse_Y columns, but the caller
    # points these at Mouse_X_smooth/Mouse_Y_smooth instead when trajectory
    # smoothing was requested (see tracking/location.py).
    xy = valid[[x_col, y_col]].to_numpy()
    t = valid["Time_seconds"].to_numpy()
    if len(xy) >= 2:
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        lc = LineCollection(segments, cmap="viridis", linewidth=1.1)
        lc.set_array(t[:-1])
        line = ax.add_collection(lc)
        fig.colorbar(line, ax=ax, label="Time (s)", shrink=0.8)
    elif len(xy) == 1:
        ax.plot(xy[:, 0], xy[:, 1], "o", color="#00e5ff", markersize=3)

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
        valid[x_col], valid[y_col], bins=50, range=[[0, warp_w], [0, warp_h]]
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


def save_zone_occupancy_chart(summary, roi_names, output_dir):
    """Horizontal bar chart of time spent (s) in each named zone --
    ezTrack/ANY-maze-style at-a-glance summary of WHERE the session's time
    went, sorted by time descending so the most-visited zone reads first.
    When the 3-point full/half/semi entry-depth breakdown is present in
    summary (see tracking.location.classify_entry_type), each bar is
    stacked into its "full" (whole body committed) portion and the rest
    of its "any part touching" time, rather than only showing the
    looser centroid-based total -- so a bar's own two segments show at a
    glance how much of that zone's time was a genuine full-body visit.
    No-op (writes nothing) if there are no named zones to chart."""
    if not roi_names:
        return

    from analysis.calculations import safe_col

    rows = []
    for i, name in enumerate(roi_names):
        total = summary.get(f"{safe_col(name)}_time_s")
        if total is None:
            continue
        full = summary.get(f"{safe_col(name)}_full_time_s")
        color = np.array(ROI_COLORS[i % len(ROI_COLORS)][::-1]) / 255.0
        rows.append((name, float(total), float(full) if full is not None else None, color))

    if not rows:
        return

    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    totals = [r[1] for r in rows]
    fulls = [r[2] for r in rows]
    colors = [r[3] for r in rows]
    has_full = any(f is not None for f in fulls)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.5 * len(rows) + 1.5)))
    y_pos = np.arange(len(rows))

    if has_full:
        full_vals = [f if f is not None else 0.0 for f in fulls]
        rest_vals = [max(0.0, t - f) for t, f in zip(totals, full_vals)]
        ax.barh(y_pos, full_vals, color=colors, edgecolor="black", linewidth=0.4, label="Full entry (all 3 points)")
        ax.barh(y_pos, rest_vals, left=full_vals, color=colors, alpha=0.35,
                edgecolor="black", linewidth=0.4, label="Partial (centroid-only)")
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.barh(y_pos, totals, color=colors, edgecolor="black", linewidth=0.4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # largest at top
    ax.set_xlabel("Time (s)")
    ax.set_title("Zone Occupancy")
    for y, total in zip(y_pos, totals):
        ax.text(total, y, f"  {total:.1f}s", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "zone_occupancy.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Calibration preview
# -----------------------------


# -----------------------------
# Cross-group comparison charts
# -----------------------------


def save_group_boxplots(df, group_col, metric_cols, output_dir, prefix="group"):
    """One PNG per metric: boxplot per group with jittered per-subject
    points overlaid (ANY-maze / EthoVision comparison style). Groups with
    fewer than 2 unique values are skipped; non-numeric junk is coerced."""
    for metric in metric_cols:
        try:
            sub = df[[group_col, metric]].copy()
        except KeyError:
            continue
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna()
        if sub[group_col].nunique() < 2 or len(sub) < 4:
            continue

        grouped = [(str(g), part[metric].to_numpy()) for g, part in sub.groupby(group_col)]
        labels = [g for g, _ in grouped]
        data = [d for _, d in grouped]

        rng = np.random.default_rng(42)
        fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(labels) + 2.0), 4.5))
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor("#3a86ff")
            patch.set_alpha(0.35)
        for i, d in enumerate(data):
            ax.scatter(i + 1 + rng.uniform(-0.12, 0.12, len(d)), d,
                       s=22, color="#10243e", zorder=3)
        ax.set_ylabel(metric)
        ax.set_xlabel(group_col)
        ax.set_title(f"{metric} by {group_col}")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in metric).strip("_") or "metric"
        fig.savefig(os.path.join(output_dir, f"{prefix}_{safe}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)


def save_timecourse_plots(df, fps, output_dir, scale_factor=None,
                          x_col="Mouse_X", y_col="Mouse_Y",
                          time_col="Time_seconds", status_col="Tracking_Status"):
    """Cumulative distance and instantaneous speed over time -- the
    time-course charts EthoVision/ANY-maze show alongside the trajectory,
    which the app previously only had as session TOTALS."""
    valid = df[df[status_col] == "Tracked"].copy()
    if len(valid) < 2:
        return
    xy = valid[[x_col, y_col]].to_numpy(dtype=float)
    t = valid[time_col].to_numpy(dtype=float)

    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    unit = "units" if scale_factor else "pixels"
    if scale_factor:
        seg = seg * scale_factor
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    dt = np.diff(t).astype(float)
    dt[dt <= 0] = np.nan
    speed = np.concatenate([[np.nan], seg / dt])  # units per second

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, cum, color="#3a86ff", lw=1.4)
    axes[0].set_ylabel(f"Cumulative distance ({unit})")
    axes[0].set_title("Distance time course")
    axes[1].plot(t, speed, color="#ff006e", lw=0.9)
    axes[1].set_ylabel(f"Speed ({unit}/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Speed time course")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "distance_timecourse.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_thigmotaxis_chart(metrics, output_dir):
    """Border vs center occupancy bar chart for an open-field session."""
    if not metrics or "border" not in metrics:
        return
    labels = [k for k in ("border", "center") if k in metrics]
    vals = [metrics[k]["time_s"] for k in labels]
    colors = ["#ff9f1c" if k == "border" else "#2ec4b6" for k in labels]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Thigmotaxis (border margin {metrics.get('margin_px', '?')} px)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "thigmotaxis.png"),
                dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Group statistics export helper
# -----------------------------


def save_group_statistics(df, group_col, output_dir, input_path=None, alpha=0.05):
    """Run analysis.statistics.compare_groups on a batch summary table and
    export statistics.csv/.xlsx plus one boxplot PNG per metric. Returns
    (results_df, paths_dict). No-op-safe when the group column is absent
    or every metric is degenerate."""
    from analysis.statistics import compare_groups, export_statistics

    if group_col not in df.columns:
        return None, {}
    results = compare_groups(df, group_col=group_col, alpha=alpha)
    if results.empty:
        return results, {}
    csv_path, xlsx_path = export_statistics(results, input_path or os.path.join(output_dir, "BatchSummary.csv"))
    save_group_boxplots(df, group_col,
                        [m for m in results["Metric"] if m in df.columns],
                        output_dir)
    return results, {"csv": csv_path, "xlsx": xlsx_path}
