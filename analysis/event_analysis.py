"""Analysis of manually scored events (Event Scoring / .btms) -- the BORIS-style toolbox.

  plot_events          ethogram timeline: one row per behaviour, state bouts as bars, point events as ticks
  plot_time_budget     duration of every state behaviour + number of occurrences of every behaviour
  time_budget          per behaviour: occurrences, total / mean / SD / min / max duration, inter-event interval
                       mean / SD, % of the observation, latency, rate per minute
  time_budget_bins     the same, cut in time bins
  transitions          which behaviour follows which (counts + probabilities)
  cohen_kappa          agreement of two observers who scored the same video (time-sliced, as in BORIS)

Events are dicts {behaviour, kind: 'state'|'point', start (s), stop (s or None)}.
"""
import math

import numpy as np
import pandas as pd

PALETTE = ["#1f6feb", "#f59e0b", "#16a34a", "#dc2626", "#7c3aed", "#0ea5e9", "#ec4899", "#84cc16", "#14b8a6",
           "#f97316", "#8b5cf6", "#64748b"]


def _kind(inventory, name, default="state"):
    for b in inventory or []:
        if b.get("name") == name:
            return b.get("kind", "state")
    return default


def behaviour_names(inventory, events):
    names = [b["name"] for b in (inventory or [])]
    for e in events or []:
        if e["behaviour"] not in names:
            names.append(e["behaviour"])
    return names


def clip_events(events, t0, t1, close_open_at=None):
    """Events inside [t0, t1]; state bouts are cut at the window edges.  A state bout that was never
    stopped is closed at `close_open_at` (the end of the video) or dropped (BORIS calls it UNPAIRED)."""
    out = []
    for e in events or []:
        s, p = float(e["start"]), e.get("stop")
        if e.get("kind", "state") == "state":
            if p is None:
                if close_open_at is None:
                    continue
                p = float(close_open_at)
            p = float(p)
            if p <= t0 or s >= t1:
                continue
            out.append(dict(e, start=max(s, t0), stop=min(p, t1)))
        else:
            if t0 <= s <= t1:
                out.append(dict(e, stop=None))
    return sorted(out, key=lambda d: d["start"])


def unpaired(events):
    return [e for e in events or [] if e.get("kind", "state") == "state" and e.get("stop") is None]


def time_budget(events, inventory, t0, t1, close_open_at=None):
    """DataFrame, one row per behaviour (BORIS 'time budget' columns + latency and rate)."""
    span = max(1e-9, float(t1) - float(t0))
    ev = clip_events(events, t0, t1, close_open_at)
    rows = []
    for name in behaviour_names(inventory, events):
        kind = _kind(inventory, name)
        mine = [e for e in ev if e["behaviour"] == name]
        starts = np.array([e["start"] for e in mine], float)
        row = {"Behaviour": name, "Type": "State" if kind == "state" else "Point", "Occurrences": len(mine)}
        if kind == "state":
            d = np.array([e["stop"] - e["start"] for e in mine], float)
            row.update({"Total_duration_s": round(float(d.sum()), 3) if d.size else 0.0,
                        "Duration_mean_s": round(float(d.mean()), 3) if d.size else None,
                        "Duration_SD_s": round(float(d.std(ddof=1)), 3) if d.size > 1 else None,
                        "Duration_min_s": round(float(d.min()), 3) if d.size else None,
                        "Duration_max_s": round(float(d.max()), 3) if d.size else None,
                        "Percent_of_time": round(100.0 * float(d.sum()) / span, 2) if d.size else 0.0})
            stops = np.array([e["stop"] for e in mine], float)
            iei = (starts[1:] - stops[:-1]) if len(mine) > 1 else np.array([])
        else:
            row.update({"Total_duration_s": None, "Duration_mean_s": None, "Duration_SD_s": None,
                        "Duration_min_s": None, "Duration_max_s": None, "Percent_of_time": None})
            iei = np.diff(starts) if len(mine) > 1 else np.array([])
        row.update({"Inter_event_interval_mean_s": round(float(iei.mean()), 3) if iei.size else None,
                    "Inter_event_interval_SD_s": round(float(iei.std(ddof=1)), 3) if iei.size > 1 else None,
                    "Latency_s": round(float(starts.min()) - float(t0), 3) if starts.size else None,
                    "Rate_per_min": round(60.0 * len(mine) / span, 3)})
        rows.append(row)
    return pd.DataFrame(rows)


def time_budget_bins(events, inventory, t0, t1, bin_s=60.0, close_open_at=None):
    """Long table: one row per time bin and behaviour (occurrences that START in the bin, time spent in it)."""
    bin_s = max(0.1, float(bin_s))
    edges = list(np.arange(float(t0), float(t1), bin_s)) + [float(t1)]
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a <= 1e-9:
            continue
        ev = clip_events(events, a, b, close_open_at)
        for name in behaviour_names(inventory, events):
            kind = _kind(inventory, name)
            mine = [e for e in ev if e["behaviour"] == name]
            last = b >= float(t1) - 1e-9
            n_start = sum(1 for e in (events or []) if e["behaviour"] == name and a <= float(e["start"]) and
                          (float(e["start"]) < b or (last and float(e["start"]) <= b)))
            dur = sum(e["stop"] - e["start"] for e in mine) if kind == "state" else None
            rows.append({"Bin": f"{a - t0:.0f}-{b - t0:.0f} s", "Bin_start_s": round(a, 3), "Bin_end_s": round(b, 3),
                         "Behaviour": name, "Occurrences": n_start,
                         "Duration_s": None if dur is None else round(dur, 3),
                         "Percent_of_bin": None if dur is None else round(100.0 * dur / (b - a), 2)})
    return pd.DataFrame(rows)


def transitions(events, inventory=None, close_open_at=None):
    """Behaviour sequence by onset -> (counts DataFrame, probabilities DataFrame (rows sum to 1))."""
    ev = sorted(clip_events(events, -1e12, 1e12, close_open_at), key=lambda e: e["start"])
    seq = [e["behaviour"] for e in ev]
    names = behaviour_names(inventory, events)
    counts = pd.DataFrame(0, index=names, columns=names, dtype=int)
    for a, b in zip(seq[:-1], seq[1:]):
        counts.loc[a, b] += 1
    tot = counts.sum(axis=1).replace(0, np.nan)
    probs = counts.div(tot, axis=0).round(3)
    counts.index.name = "From \\ To"
    probs.index.name = "From \\ To"
    return counts, probs


def _labels(events, inventory, t0, t1, step, point_window, close_open_at=None):
    """What the observer had coded at every time slice (sorted tuple of behaviours, () = nothing)."""
    ev = clip_events(events, t0, t1, close_open_at)
    times = np.arange(float(t0), float(t1), float(step))
    out = []
    for t in times:
        cur = []
        for e in ev:
            if e.get("kind", "state") == "state":
                if e["start"] <= t < e["stop"]:
                    cur.append(e["behaviour"])
            elif abs(e["start"] - t) <= point_window / 2.0:
                cur.append(e["behaviour"])
        out.append(tuple(sorted(set(cur))))
    return times, out


def _kappa(a, b):
    cats = sorted(set(a) | set(b), key=str)
    if not cats:
        return None, None
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return None, None
    po = np.trace(m) / n
    pe = float((m.sum(axis=0) * m.sum(axis=1)).sum()) / (n * n)
    if abs(1 - pe) < 1e-12:
        return (1.0 if po == 1 else 0.0), po
    return (po - pe) / (1 - pe), po


def cohen_kappa(events_a, events_b, inventory, t0, t1, step=1.0, point_window=None, close_open_at=None):
    """Agreement between two observers (BORIS method: both codings are looked at every `step` seconds;
    a point event counts at the slices within +-step/2 of it).  Returns dict with overall kappa,
    % agreement, number of slices and a per-behaviour table (presence / absence kappa)."""
    pw = float(point_window if point_window is not None else step)
    _t, la = _labels(events_a, inventory, t0, t1, step, pw, close_open_at)
    _t, lb = _labels(events_b, inventory, t0, t1, step, pw, close_open_at)
    k, po = _kappa(la, lb)
    rows = []
    for name in behaviour_names(inventory, list(events_a or []) + list(events_b or [])):
        xa = [name in x for x in la]
        xb = [name in x for x in lb]
        kk, pp = _kappa(xa, xb)
        rows.append({"Behaviour": name, "Kappa": None if kk is None else round(kk, 3),
                     "Agreement_pct": None if pp is None else round(100 * pp, 1),
                     "Coded_by_A_s": round(sum(xa) * step, 2), "Coded_by_B_s": round(sum(xb) * step, 2)})
    return {"kappa": None if k is None else round(k, 3), "agreement_pct": None if po is None else round(100 * po, 1),
            "slices": len(la), "step_s": float(step), "per_behaviour": pd.DataFrame(rows)}


def kappa_words(k):
    """Landis & Koch (1977) wording."""
    if k is None or (isinstance(k, float) and math.isnan(k)):
        return "-"
    for lim, w in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= lim:
            return w
    return "almost perfect"


# ----------------------------------------------------------------------------- plots
def _hms(x, _pos=None):
    x = max(0.0, float(x))
    h, r = divmod(int(x), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def plot_events(events, inventory, t0, t1, path, title="", colors=None, close_open_at=None, dpi=150):
    """Ethogram (BORIS 'plot events'): one row per behaviour, state bouts as bars, point events as ticks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    names = behaviour_names(inventory, events)
    ev = clip_events(events, t0, t1, close_open_at)
    span = max(1e-6, float(t1) - float(t0))
    tick_w = max(0.25, span * 0.002)
    fig_h = 1.4 + 0.62 * max(1, len(names))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    for i, name in enumerate(names):
        y = len(names) - 1 - i
        col = (colors or {}).get(name) or PALETTE[i % len(PALETTE)]
        bars = [(e["start"], e["stop"] - e["start"]) for e in ev
                if e["behaviour"] == name and e.get("kind", "state") == "state"]
        pts = [(e["start"] - tick_w / 2, tick_w) for e in ev if e["behaviour"] == name and e.get("kind") == "point"]
        if bars:
            ax.broken_barh(bars, (y - 0.3, 0.6), facecolors=col)
        if pts:
            ax.broken_barh(pts, (y - 0.3, 0.6), facecolors=col)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(list(reversed(names)))
    ax.set_ylim(-0.8, len(names) - 0.2)
    ax.set_xlim(float(t0), float(t1))
    ax.xaxis.set_major_formatter(FuncFormatter(_hms))
    ax.grid(True, axis="both", linestyle=":", color="#16a34a", alpha=0.6)
    ax.set_xlabel("Time (HH:MM:SS)")
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plot_time_budget(budget, path, title="", colors=None, dpi=150):
    """Two bar charts: total duration of the state behaviours, number of occurrences of every behaviour."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    b = budget.copy()
    names = list(b["Behaviour"])
    cols = [(colors or {}).get(n) or PALETTE[i % len(PALETTE)] for i, n in enumerate(names)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 + 0.45 * max(3, len(names))))
    st = b[b["Type"] == "State"]
    ax = axes[0]
    if len(st):
        idx = [names.index(n) for n in st["Behaviour"]]
        ax.barh(list(st["Behaviour"]), list(st["Total_duration_s"]), color=[cols[i] for i in idx])
        for y, v in enumerate(st["Total_duration_s"]):
            ax.text(v, y, f" {v:.1f} s", va="center", fontsize=8)
    ax.invert_yaxis()
    if len(st):
        ax.set_xlim(0, max(1.0, float(st["Total_duration_s"].max())) * 1.22)
    ax.set_xlabel("Total duration (s)")
    ax.set_title("Duration (state behaviours)")
    ax = axes[1]
    ax.barh(names, list(b["Occurrences"]), color=cols)
    for y, v in enumerate(b["Occurrences"]):
        ax.text(v, y, f" {int(v)}", va="center", fontsize=8)
    ax.invert_yaxis()
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(0, max(1, int(b["Occurrences"].max() if len(b) else 1)) * 1.15)
    ax.set_xlabel("Number of occurrences")
    ax.set_title("Occurrences (all behaviours)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def zone_bouts(df, zones, label_col=None, fps=30.0):
    """Zone occupancy of a tracking run as state 'events' (for the zone timeline).  Uses the four-paw zone
    label when it exists, else the In_<zone> flags of the body centre (or the chosen point)."""
    from analysis.calculations import safe_col
    t = df["Time_seconds"].to_numpy(float)
    dt = 1.0 / float(fps or 30.0)
    out = []
    present = df["Tracking_Status"].isin(("Tracked", "Hidden")).to_numpy() if "Tracking_Status" in df else None
    for z in zones:
        if label_col and label_col in df.columns:
            flag = (df[label_col].astype(str) == str(z)).to_numpy()
        else:
            col = f"In_{safe_col(z)}"
            if col not in df.columns:
                continue
            flag = df[col].fillna(False).to_numpy(bool)
        if present is not None:
            flag = flag & present
        i, n = 0, len(flag)
        while i < n:
            if flag[i]:
                j = i
                while j < n and flag[j]:
                    j += 1
                out.append({"behaviour": z, "kind": "state", "start": float(t[i]), "stop": float(t[j - 1]) + dt})
                i = j
            else:
                i += 1
    return out
