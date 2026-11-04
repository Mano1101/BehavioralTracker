"""Freezing charts (BehavioralTracker 1.0.0) -- shown on the Analysis & Results page and saved with the results.

  overview_epochs   freezing per epoch (bars, coloured by kind: tone / ITI / baseline ...)
  overview_bins     freezing per time bin
  overview_trace    motion over time + freezing raster + cumulative freezing
  motion_trace      the big motion trace (Motion Trace tab)
  epochs_tab        trial by trial per kind, epoch vs the time before it, change vs before
  bouts_tab         freezing raster, bout durations, intervals between bouts
  bins_tab          freezing and motion per time bin + cumulative freezing
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GREY, DARK, GREEN, RED = "#1f6feb", "#cbd5e1", "#334155", "#16a34a", "#dc2626"
KIND_COLOURS = ["#f59e0b", "#94a3b8", "#64748b", "#8b5cf6", "#06b6d4", "#ef4444", "#22c55e", "#ec4899"]


def _style(ax, size=8):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=size)


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def kind_colours(result):
    from tracking.freezing import group_name
    kinds = []
    for e in result["epochs"]:
        g = group_name(e["Epoch"])
        if g not in kinds:
            kinds.append(g)
    tone_first = sorted(kinds, key=lambda k: 0 if k.lower().startswith(("tone", "cs")) else 1)
    return {k: KIND_COLOURS[i % len(KIND_COLOURS)] for i, k in enumerate(tone_first)}


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


def _empty(ax, text):
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=9, color="#64748b", wrap=True,
            transform=ax.transAxes)


def overview_epochs(result, path, size=(3.2, 5.4)):
    from tracking.freezing import group_name
    fig, ax = plt.subplots(figsize=size, dpi=110)
    _style(ax)
    eps = [e for e in result["epochs"] if np.isfinite(e.get("Freezing_pct", np.nan))]
    if not eps:
        s = result["summary"]
        f = float(s["Freezing_pct"])
        ax.bar(["Freezing", "Active"], [f, 100 - f], color=[BLUE, GREY])
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of the analysed time", fontsize=8)
        ax.set_title("Freezing vs active\n(add epochs for tone / ITI bars)", fontsize=9)
        for i, v in enumerate((f, 100 - f)):
            ax.text(i, v + 1.5, f"{v:.0f}", ha="center", fontsize=8)
    else:
        col = kind_colours(result)
        names = [e["Epoch"] for e in eps][::-1]
        vals = [e["Freezing_pct"] for e in eps][::-1]
        ax.barh(names, vals, color=[col[group_name(n)] for n in names])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Freezing (%)", fontsize=8)
        for i, v in enumerate(vals):
            ax.text(min(v + 1.5, 92), i, f"{v:.0f}", va="center", fontsize=7)
        ax.set_title("Freezing per epoch", fontsize=9)
        if len(names) > 14:
            ax.tick_params(axis="y", labelsize=6)
    fig.tight_layout()
    return _save(fig, path)


def overview_bins(result, path, size=(3.2, 5.4)):
    fig, ax = plt.subplots(figsize=size, dpi=110)
    _style(ax)
    bins = result["bins"]
    if not bins:
        _empty(ax, "No time bins\n(set 'Time bins' on the Freezing page)")
    else:
        lab = [f"{b['Start_s']:.0f}–{b['End_s']:.0f}" for b in bins]
        v = [b["Freezing_pct"] for b in bins]
        ax.bar(range(len(v)), v, color=BLUE)
        ax.set_xticks(range(len(v)))
        ax.set_xticklabels(lab, rotation=45, ha="right", fontsize=7 if len(v) < 12 else 5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Freezing (%)", fontsize=8)
        bs = float((result.get("settings") or {}).get("bin_s") or (bins[0]["End_s"] - bins[0]["Start_s"]))
        ax.set_title(f"Freezing per {'minute' if abs(bs - 60) < 0.5 else f'{bs:g} s bin'}", fontsize=9)
        if len(v) <= 16:
            for i, x in enumerate(v):
                ax.text(i, x + 1.5, f"{x:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    return _save(fig, path)


def _trace(ax, result, fps, thr, top=None, tones=True):
    t, mot, frz = result["times"], result["motion"], result["freezing"]
    fin = np.isfinite(mot)
    top = top or max((np.nanpercentile(mot[fin], 99) * 1.1) if fin.any() else 1.0, 2 * thr, 1.0)
    for a, b in _runs(frz):
        ax.axvspan(t[a], t[b - 1] + 1.0 / fps, color="#dbe7f7", lw=0)
    if tones:
        col = kind_colours(result)
        from tracking.freezing import group_name
        for e in result["epochs"]:
            ax.axvspan(e["Start_s"], e["End_s"], ymin=0.94, ymax=1, color=col[group_name(e["Epoch"])], lw=0)
    ax.plot(t, np.minimum(mot, top), color=BLUE, lw=0.5)
    ax.axhline(thr, color=RED, lw=0.8, ls="--")
    ax.set_ylim(0, top)
    ax.set_xlim(t[0], t[-1])
    return top


def overview_trace(result, fps, path, size=(6.6, 5.4)):
    thr = float(result["settings"]["threshold_pct"])
    t, frz = result["times"], result["freezing"]
    fig = plt.figure(figsize=size, dpi=110)
    ax = fig.add_axes([0.1, 0.42, 0.86, 0.5])
    _style(ax)
    _trace(ax, result, fps, thr)
    ax.set_ylabel("Motion (% of body)", fontsize=8)
    ax.set_title("Motion over time (blue shade = freezing, red line = threshold)", fontsize=9)
    ax2 = fig.add_axes([0.1, 0.27, 0.86, 0.07])
    _style(ax2)
    for a, b in _runs(frz):
        ax2.axvspan(t[a], t[b - 1] + 1.0 / fps, color=DARK, lw=0)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_yticks([])
    ax2.set_title("Freezing bouts", fontsize=8, loc="left")
    ax3 = fig.add_axes([0.1, 0.07, 0.86, 0.14])
    _style(ax3)
    ax3.plot(t, np.cumsum(frz) / fps, color=GREEN, lw=1.2)
    ax3.set_xlim(t[0], t[-1])
    ax3.set_xlabel("Time (s)", fontsize=8)
    ax3.set_ylabel("Cum. freezing (s)", fontsize=7)
    return _save(fig, path)


def motion_trace(result, fps, path, title="", size=(12, 3.6)):
    thr = float(result["settings"]["threshold_pct"])
    fig, ax = plt.subplots(figsize=size, dpi=110)
    _style(ax)
    _trace(ax, result, fps, thr)
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Motion (% of body)", fontsize=9)
    s = result["summary"]
    ax.set_title(f"{title}   ·   freezing {s['Freezing_pct']:.1f} %  ·  {s['Freezing_bouts']} bouts   "
                 f"(blue shade = freezing, red line = threshold {thr:g} %, top bar = epochs)", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def epochs_tab(result, path, size=(13, 4.6)):
    fig, axs = plt.subplots(1, 3, figsize=size, dpi=110)
    for ax in axs:
        _style(ax)
    groups = [g for g in result.get("groups") or [] if g["Trials"] >= 1]
    col = kind_colours(result)
    ax = axs[0]
    if not groups:
        _empty(ax, "No epochs — add tones / epochs on the Freezing page")
    else:
        for g in groups:
            v = g["Trial_values"]
            ax.plot(range(1, len(v) + 1), v, "o-" if len(v) > 1 else "o", color=col.get(g["Group"], BLUE), lw=2,
                    label=f"{g['Group']} (n={len(v)})")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Freezing (%)")
        mx = max(len(g["Trial_values"]) for g in groups)
        ax.set_xticks(range(1, mx + 1))
        ax.set_title("Trial by trial (per kind of epoch)")
        ax.legend(fontsize=7)
    eps = [e for e in result["epochs"] if np.isfinite(e.get("Pre_freezing_pct", np.nan))]
    ax = axs[1]
    if not eps:
        _empty(ax, f"No 'before' window\n(epochs need {result['settings'].get('pre_s', 30):g} s of video before them)")
    else:
        x = np.arange(len(eps))
        ax.bar(x - 0.2, [e["Pre_freezing_pct"] for e in eps], 0.4, color="#64748b",
               label=f"{result['settings'].get('pre_s', 30):g} s before")
        ax.bar(x + 0.2, [e["Freezing_pct"] for e in eps], 0.4, color="#f59e0b", label="Epoch")
        ax.set_xticks(x)
        ax.set_xticklabels([e["Epoch"] for e in eps], rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Freezing (%)")
        ax.set_title("Before vs during each epoch")
        ax.legend(fontsize=7)
    ax = axs[2]
    if not eps:
        _empty(ax, "")
    else:
        ch = [e["Change_vs_pre_pct"] for e in eps]
        ax.bar(range(len(eps)), ch, color=[GREEN if c >= 0 else RED for c in ch])
        ax.axhline(0, color="#475569", lw=0.8)
        ax.set_xticks(range(len(eps)))
        ax.set_xticklabels([e["Epoch"] for e in eps], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Freezing change (% points)")
        ax.set_title("Change vs before (epoch − before)")
    fig.tight_layout()
    return _save(fig, path)


def bouts_tab(result, fps, path, size=(13, 4.6)):
    t, frz = result["times"], result["freezing"]
    bouts = result["bouts"]
    fig = plt.figure(figsize=size, dpi=110)
    ax = fig.add_axes([0.05, 0.62, 0.92, 0.28])
    _style(ax)
    for a, b in _runs(frz):
        ax.axvspan(t[a], t[b - 1] + 1.0 / fps, color=DARK, lw=0)
    col = kind_colours(result)
    from tracking.freezing import group_name
    for e in result["epochs"]:
        ax.axvspan(e["Start_s"], e["End_s"], ymin=1.02, ymax=1.12, color=col[group_name(e["Epoch"])], lw=0,
                   clip_on=False)
    ax.set_xlim(t[0], t[-1])
    ax.set_yticks([])
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_title(f"Freezing bouts over time ({len(bouts)} bouts; coloured bar = epochs)", fontsize=9, pad=14)
    d = [b["Duration_s"] for b in bouts]
    ax2 = fig.add_axes([0.05, 0.1, 0.42, 0.38])
    _style(ax2)
    if d:
        ax2.hist(d, bins=np.arange(0, max(d) + 1.0, max(0.5, round(max(d) / 20, 1))), color=BLUE)
    ax2.set_xlabel("Bout duration (s)", fontsize=8)
    ax2.set_ylabel("Bouts", fontsize=8)
    ax2.set_title("Bout durations", fontsize=9)
    ibi = [bouts[i + 1]["Start_s"] - bouts[i]["End_s"] for i in range(len(bouts) - 1)]
    ax3 = fig.add_axes([0.55, 0.1, 0.42, 0.38])
    _style(ax3)
    if ibi:
        ax3.hist(ibi, bins=np.arange(0, max(ibi) + 1.0, max(0.5, round(max(ibi) / 20, 1))), color="#94a3b8")
    ax3.set_xlabel("Time between bouts (s)", fontsize=8)
    ax3.set_ylabel("Count", fontsize=8)
    ax3.set_title("Intervals between bouts", fontsize=9)
    return _save(fig, path)


def bins_tab(result, fps, path, size=(13, 4.6)):
    fig, axs = plt.subplots(1, 2, figsize=size, dpi=110)
    for ax in axs:
        _style(ax)
    bins = result["bins"]
    ax = axs[0]
    if not bins:
        _empty(ax, "No time bins")
    else:
        x = np.arange(len(bins))
        ax.bar(x, [b["Freezing_pct"] for b in bins], color=BLUE, label="Freezing (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{b['Start_s']:.0f}–{b['End_s']:.0f}" for b in bins], rotation=45, ha="right",
                           fontsize=7)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Freezing (%)")
        ax2 = ax.twinx()
        ax2.plot(x, [b["Mean_motion_pct"] for b in bins], "o-", color="#f59e0b", lw=1.5, label="Mean motion")
        ax2.set_ylabel("Mean motion (% of body)", color="#b45309")
        ax2.spines["top"].set_visible(False)
        ax.set_title("Freezing and motion per time bin")
    ax = axs[1]
    t, frz = result["times"], result["freezing"]
    ax.plot(t, np.cumsum(frz) / fps, color=GREEN, lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative freezing (s)")
    ax.set_title("Cumulative freezing")
    fig.tight_layout()
    return _save(fig, path)


CHARTS = ("overview_epochs", "overview_bins", "overview_trace", "motion_trace", "epochs_tab", "bouts_tab",
          "bins_tab")


def make_all(result, fps, folder, name):
    """Every chart as <folder>/<name>_Freezing_<chart>.png -> {chart: path}."""
    import os
    os.makedirs(folder, exist_ok=True)

    def p(k):
        return os.path.join(folder, f"{name}_Freezing_{k}.png")
    out = {}
    jobs = (("overview_epochs", lambda: overview_epochs(result, p("overview_epochs"))),
            ("overview_bins", lambda: overview_bins(result, p("overview_bins"))),
            ("overview_trace", lambda: overview_trace(result, fps, p("overview_trace"))),
            ("motion_trace", lambda: motion_trace(result, fps, p("motion_trace"), title=name)),
            ("epochs_tab", lambda: epochs_tab(result, p("epochs_tab"))),
            ("bouts_tab", lambda: bouts_tab(result, fps, p("bouts_tab"))),
            ("bins_tab", lambda: bins_tab(result, fps, p("bins_tab"))))
    for k, fn in jobs:
        try:
            out[k] = fn()
        except Exception as exc:                       # a chart is a convenience only
            print(f"Freezing chart {k} skipped: {exc}")
            plt.close("all")
    return out
