"""One-page PDF summary per video (A4 landscape) -- for lab meetings and lab books.

freezing_pdf(): freezing %, time, bouts, latency, the result checks, the motion / freezing chart, the epoch table
                and the settings.
tracking_pdf(): the test's main measures (time and % in every zone, entries, distance, indices), the trajectory,
                the heat map and the zone chart.
Both are drawn with matplotlib only (no extra package)."""
import math
import os
import time

import numpy as np

A4_LANDSCAPE = (11.69, 8.27)


def _fmt(v, nd=2):
    if v is None:
        return "—"
    if isinstance(v, (float, np.floating)):
        if not math.isfinite(float(v)):
            return "—"
        return f"{float(v):.{nd}f}"
    return str(v)


def _clean(t):
    """Text for the PDF fonts: no emoji / symbols the standard font does not have."""
    t = str(t)
    for a, b in (("⛔", ""), ("⚠", ""), ("ℹ", ""), ("📏", ""), ("🎯", ""), ("✨", ""), ("👁", ""), ("▨", ""),
                 ("⬠", ""), ("▦", "")):
        t = t.replace(a, b)
    return "".join(ch for ch in t if ord(ch) <= 0xFFFF).replace("  ", " ").strip()


def _figure():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=A4_LANDSCAPE)
    return plt, fig


def _header(fig, title, sub):
    title, sub = _clean(title), _clean(sub)
    fig.text(0.04, 0.955, title, fontsize=16, fontweight="bold", color="#0f172a", va="top")
    fig.text(0.04, 0.915, sub, fontsize=9, color="#475569", va="top")
    from tracking.project_file import APP
    fig.text(0.96, 0.955, APP, fontsize=9, color="#64748b", ha="right", va="top")      # name + version
    fig.text(0.96, 0.93, time.strftime("%Y-%m-%d %H:%M"), fontsize=8, color="#94a3b8", ha="right", va="top")


def _table(ax, rows, col_labels=None, font=8.5, col_widths=None):
    ax.axis("off")
    if not rows:
        return
    rows = [[_clean(c) for c in r] for r in rows]
    t = ax.table(cellText=rows, colLabels=col_labels, loc="upper left", cellLoc="left", colLoc="left",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(font)
    t.scale(1, 1.25)
    for (r, _c), cell in t.get_celld().items():
        cell.set_edgecolor("#dbe3ef")
        if r == 0 and col_labels:
            cell.set_facecolor("#eef3fb")
            cell.set_text_props(fontweight="bold", color="#1f2a44")


def _image(ax, path, title=None):
    ax.axis("off")
    if not path or not os.path.exists(path):
        ax.text(0.5, 0.5, "—", ha="center", va="center", color="#94a3b8")
        return
    import matplotlib.image as mpimg
    try:
        ax.imshow(mpimg.imread(path))
    except Exception:
        ax.text(0.5, 0.5, "—", ha="center", va="center", color="#94a3b8")
        return
    if title:
        ax.set_title(title, fontsize=9, color="#1f2a44", loc="left")


def freezing_pdf(md, result, charts, path, name, subject="", group=""):
    """The freezing result of one video on one page.  Returns the path."""
    plt, fig = _figure()
    s = result["summary"]
    who = " · ".join(x for x in (f"Subject {subject}" if subject else "", f"Group {group}" if group else "") if x)
    _header(fig, f"Freezing — {name}",
            (who + "  ·  " if who else "") +
            f"analysed {_fmt(s.get('Start_time_s'), 1)}–{_fmt(s.get('End_time_s'), 1)} s  ·  "
            f"{s.get('Motion_counted', '')}  ·  threshold {_fmt(s.get('Freezing_threshold_pct'), 1)} % of body "
            f"(= {_fmt(s.get('Freezing_threshold_px'), 0)} px)  ·  min freeze {_fmt(s.get('Min_freeze_s'), 2)} s")
    kpis = [("Freezing", f"{_fmt(s.get('Freezing_pct'), 1)} %"), ("Freezing time", f"{_fmt(s.get('Freezing_time_s'), 1)} s"),
            ("Bouts", str(s.get("Freezing_bouts", "—"))),
            ("Latency to 1st freeze", f"{_fmt(s.get('Latency_first_freeze_s'), 1)} s"),
            ("Mean bout", f"{_fmt(s.get('Mean_bout_s'), 2)} s")]
    for k, (lab, val) in enumerate(kpis):
        x = 0.04 + k * 0.186
        fig.patches.append(plt.Rectangle((x, 0.80), 0.17, 0.085, transform=fig.transFigure, facecolor="#f3f6fb",
                                         edgecolor="#dbe3ef"))
        fig.text(x + 0.01, 0.868, lab, fontsize=8, color="#64748b")
        fig.text(x + 0.01, 0.815, val, fontsize=15, fontweight="bold", color="#1f2a44")
    checks = [c for c in (result.get("checks") or []) if c["level"] in ("error", "warn")]
    if checks:
        err = any(c["level"] == "error" for c in checks)
        txt = ("CHECK BEFORE USING: " if err else "Check: ") + "  ·  ".join(_clean(c["text"]) for c in checks)
        fig.text(0.04, 0.775, txt[:330] + ("…" if len(txt) > 330 else ""), fontsize=8,
                 color="#991b1b" if err else "#92400e", va="top", wrap=True)
    else:
        fig.text(0.04, 0.775, "Result check: OK", fontsize=8, color="#15803d", va="top")
    ax = fig.add_axes([0.04, 0.33, 0.56, 0.40])
    _image(ax, (charts or {}).get("overview_trace") or (charts or {}).get("motion_trace"),
           "Motion and freezing over time")
    ax = fig.add_axes([0.62, 0.33, 0.34, 0.40])
    _image(ax, (charts or {}).get("body_size"), "Body size used (yellow outline)")
    rows = [[e["Epoch"], _fmt(e["Start_s"], 1), _fmt(e["End_s"], 1), _fmt(e.get("Freezing_pct"), 1),
             _fmt(e.get("Pre_freezing_pct"), 1)] for e in result.get("epochs") or []]
    if not rows:
        rows = [[f"Bin {b['Bin']}", _fmt(b["Start_s"], 1), _fmt(b["End_s"], 1), _fmt(b.get("Freezing_pct"), 1), "—"]
                for b in result.get("bins") or []]
    ax = fig.add_axes([0.04, 0.04, 0.56, 0.26])
    if rows:
        more = len(rows) > 12
        rows = rows[:12] + ([["…", "", "", "", ""]] if more else [])
        _table(ax, rows, ["Epoch / bin", "Start (s)", "End (s)", "Freezing (%)", "Before (%)"],
               col_widths=[0.36, 0.14, 0.14, 0.18, 0.18])
    else:
        ax.axis("off")
    st = [["Cutoff", f"{s.get('Motion_cutoff')} ({s.get('Motion_cutoff_mode', '')})"],
          ["Body size", f"{_fmt(s.get('Body_length_px'), 0)} px long ({s.get('Body_size_from', '')})"],
          ["Animal found", f"{_fmt(s.get('Animal_found_pct'), 0)} % of frames"],
          ["Colour / light guard", f"{s.get('Motion_colour', '')} / {s.get('Light_guard', '')}"],
          ["Region", str(s.get("Region", ""))[:60]],
          ["Video", os.path.basename(str(getattr(md, "path", "") or ""))[:60]]]
    ax = fig.add_axes([0.62, 0.04, 0.34, 0.26])
    _table(ax, st, None, font=8, col_widths=[0.33, 0.67])
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


def tracking_pdf(output_dir, video_name, summary, test_name=""):
    """The tracking result of one video on one page (main measures + trajectory + heat map + zones)."""
    from output import report_layout as RL
    plt, fig = _figure()
    s = {k: v for k, v in summary.items()}
    zones = RL.zones_of(list(s))
    test = s.get("Test_type") if s.get("Test_type") in RL.TEST_NAMES else "custom"
    _header(fig, f"{test_name or RL.TEST_NAMES.get(test, 'Tracking')} — {video_name}",
            f"subject {s.get('Subject') or '—'}  ·  analysed {_fmt(s.get('Start_s'), 1)}–{_fmt(s.get('End_s'), 1)} s"
            f"  ·  tracking quality {_fmt(s.get('Tracking_quality_percent'), 1)} %")
    rows = []
    for z in zones:
        t, p, e = s.get(f"{z}_time_s"), s.get(f"{z}_percent"), s.get(f"{z}_entries")
        if t is not None or p is not None:
            rows.append([str(z), _fmt(t, 1), _fmt(p, 1), _fmt(e, 0)])
    ax = fig.add_axes([0.04, 0.50, 0.42, 0.38])
    if rows:
        more = len(rows) > 14
        rows = rows[:14] + ([["…", "", "", ""]] if more else [])
        _table(ax, rows, ["Zone", "Time (s)", "Time (%)", "Entries"], col_widths=[0.4, 0.2, 0.2, 0.2])
    else:
        ax.axis("off")
    g = RL.groups(list(s), zones, test=test)
    main = []
    for k in g.get("indices", []) + g.get("distance", []) + g.get("time", []) + g.get("entries", []):
        if RL.zone_field(k, zones)[0] is not None or s.get(k) is None:
            continue
        v = s[k]
        if isinstance(v, float) and not math.isfinite(v):
            continue
        try:
            lab = RL.label_of(k)
        except Exception:
            lab = k.replace("_", " ")
        main.append([str(lab)[:46], _fmt(v, 2)])
    ax = fig.add_axes([0.04, 0.05, 0.42, 0.40])
    _table(ax, main[:14], ["Measure", "Value"], col_widths=[0.72, 0.28])
    ax = fig.add_axes([0.50, 0.50, 0.23, 0.38])
    _image(ax, os.path.join(output_dir, "trajectory.png"), "Trajectory")
    ax = fig.add_axes([0.75, 0.50, 0.23, 0.38])
    _image(ax, os.path.join(output_dir, "heatmap.png"), "Heat map")
    ax = fig.add_axes([0.50, 0.05, 0.48, 0.40])
    _image(ax, os.path.join(output_dir, "zone_occupancy.png"), "Time in the zones")
    path = os.path.join(output_dir, f"{video_name}_Summary.pdf")
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- slides (16:9 pictures)
SLIDE_SIZE = (13.333, 7.5)          # inches; saved at 144 dpi = 1920 x 1080 px


def _slide():
    import matplotlib
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    return plt, plt.figure(figsize=SLIDE_SIZE)


def _slide_head(fig, title, sub):
    fig.text(0.035, 0.94, _clean(title), fontsize=24, fontweight="bold", color="#0f172a", va="top")
    fig.text(0.035, 0.875, _clean(sub), fontsize=12, color="#475569", va="top")
    from tracking.project_file import APP
    fig.text(0.965, 0.035, APP, fontsize=9, color="#94a3b8", ha="right")


def _kpis(plt, fig, kpis, y=0.69, h=0.13):
    n = max(1, len(kpis))
    w = (0.93 - 0.02 * (n - 1)) / n
    for k, (lab, val) in enumerate(kpis):
        x = 0.035 + k * (w + 0.02)
        fig.patches.append(plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor="#f3f6fb",
                                         edgecolor="#dbe3ef"))
        fig.text(x + 0.012, y + h - 0.035, lab, fontsize=12, color="#64748b")
        fig.text(x + 0.012, y + 0.02, val, fontsize=26 if len(kpis) > 4 else 30, fontweight="bold",
                 color="#1f2a44")


def freezing_slide(md, result, charts, path, name, subject="", group=""):
    """One 16:9 picture (1920 x 1080) of a freezing result for a lab-meeting slide.  Returns the path."""
    plt, fig = _slide()
    s = result["summary"]
    who = " · ".join(x for x in (f"Subject {subject}" if subject else "", f"Group {group}" if group else "") if x)
    _slide_head(fig, f"Freezing — {name}", (who + "  ·  " if who else "") + f"{s.get('Motion_counted', '')}  ·  "
                f"threshold {_fmt(s.get('Freezing_threshold_pct'), 1)} % of body  ·  min freeze "
                f"{_fmt(s.get('Min_freeze_s'), 2)} s")
    _kpis(plt, fig, [("Freezing", f"{_fmt(s.get('Freezing_pct'), 1)} %"),
                     ("Freezing time", f"{_fmt(s.get('Freezing_time_s'), 0)} s"),
                     ("Bouts", str(s.get("Freezing_bouts", "—"))),
                     ("Latency to 1st freeze", f"{_fmt(s.get('Latency_first_freeze_s'), 1)} s")])
    ax = fig.add_axes([0.035, 0.07, 0.6, 0.58])
    _image(ax, (charts or {}).get("overview_trace") or (charts or {}).get("motion_trace"))
    ax = fig.add_axes([0.665, 0.07, 0.3, 0.58])
    side = (charts or {}).get("overview_epochs") if result.get("epochs") else (charts or {}).get("overview_bins")
    _image(ax, side or (charts or {}).get("body_size"))
    checks = [c for c in (result.get("checks") or []) if c["level"] == "error"]
    if checks:
        fig.text(0.035, 0.66, "CHECK BEFORE USING: " + _clean(checks[0]["text"])[:160], fontsize=10,
                 color="#991b1b")
    fig.savefig(path, dpi=144)
    plt.close(fig)
    return path


def tracking_slide(output_dir, video_name, summary, test_name=""):
    """One 16:9 picture (1920 x 1080) of a tracking result: trajectory, heat map, time in the zones."""
    from output import report_layout as RL
    plt, fig = _slide()
    s = dict(summary)
    test = s.get("Test_type") if s.get("Test_type") in RL.TEST_NAMES else "custom"
    _slide_head(fig, f"{test_name or RL.TEST_NAMES.get(test, 'Tracking')} — {video_name}",
                f"subject {s.get('Subject') or '—'}  ·  analysed {_fmt(s.get('Start_s'), 1)}–"
                f"{_fmt(s.get('End_s'), 1)} s  ·  tracking quality {_fmt(s.get('Tracking_quality_percent'), 1)} %")
    zones = RL.zones_of(list(s))
    g = RL.groups(list(s), zones, test=test)
    kp = []
    for k in g.get("indices", []) + g.get("time", []) + g.get("entries", []) + g.get("distance", []):
        if len(kp) >= 4 or RL.zone_field(k, zones)[0] is not None:
            continue
        v = s.get(k)
        if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v)):
            continue
        try:
            lab = str(RL.label_of(k))
        except Exception:
            lab = k.replace("_", " ")
        kp.append((lab[:34], _fmt(v, 1)))
    if kp:
        _kpis(plt, fig, kp, y=0.70, h=0.12)
    top = 0.64 if kp else 0.83
    ax = fig.add_axes([0.035, 0.07, 0.29, top - 0.07])
    _image(ax, os.path.join(output_dir, "trajectory.png"), "Trajectory")
    ax = fig.add_axes([0.345, 0.07, 0.29, top - 0.07])
    _image(ax, os.path.join(output_dir, "heatmap.png"), "Heat map")
    ax = fig.add_axes([0.655, 0.07, 0.31, top - 0.07])
    _image(ax, os.path.join(output_dir, "zone_occupancy.png"), "Time in the zones")
    path = os.path.join(output_dir, f"{video_name}_Slide.png")
    fig.savefig(path, dpi=144)
    plt.close(fig)
    return path
