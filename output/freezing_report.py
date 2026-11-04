"""Freezing analysis output (BehavioralTracker 1.0.0).

Per video (in <video folder>/results/<video>/freezing/):
  <video>_Freezing.xlsx       Freezing results · Epochs · Time bins · Freezing bouts · Settings · Frame by frame ·
                              Column guide
  <video>_Freezing_frames.csv the frame-by-frame table
  <video>_Freezing_<chart>.png the charts of the Analysis & Results page (output/freezing_graphs.py)
  <video>.btf                 the BT Freezing file (tracking/project_file.py) -- written by the caller

Batch (one row per video):
  Freezing_BatchSummary_<time>.xlsx   Freezing summary (ready for Statistics) · Epochs · Time bins ·
                                      Video & tracking info · Settings · Column guide
"""
import math
import os

import numpy as np
import pandas as pd

from output.excel import style_sheet, _auto_widths, _Sheet
from tracking.project_file import APP as BT_APP

# key -> (label, formula, what it means)
MEASURES = {
    "Freezing_pct": ("Freezing (%)", "Freezing time ÷ analysed time × 100",
                     "Share of the analysed time the animal was freezing."),
    "Freezing_time_s": ("Freezing time (s)", "Sum of all freezing bouts",
                        "Total time spent freezing."),
    "Active_time_s": ("Active time (s)", "Analysed time − freezing time", "Time not freezing."),
    "Freezing_bouts": ("Freezing bouts", "Count of freezing periods",
                       "How many separate times the animal froze (each at least the minimum freeze duration)."),
    "Mean_bout_s": ("Mean bout duration (s)", "Freezing time ÷ freezing bouts", "Average length of one freeze."),
    "Median_bout_s": ("Median bout duration (s)", "Middle value of the bout durations",
                      "Typical length of one freeze (not pulled up by one very long freeze)."),
    "Longest_bout_s": ("Longest bout (s)", "Longest freezing period", "The longest single freeze."),
    "Latency_first_freeze_s": ("Latency to first freeze (s)", "Start of the first bout − start of the analysis",
                               "How long until the animal froze for the first time (empty = never froze)."),
    "Mean_motion_pct": ("Mean motion (% of body)", "Average of the frame motion values",
                        "Average amount of movement (changed pixels as % of the animal's body area)."),
    "Bout_rate_per_min": ("Freezing bouts per minute", "Freezing bouts ÷ analysed minutes",
                          "How often the animal started to freeze."),
    "Mean_interbout_s": ("Mean time between bouts (s)", "Average of (start of a bout − end of the bout before)",
                         "How long the animal moved between two freezes."),
    "Median_interbout_s": ("Median time between bouts (s)", "Middle value of the times between bouts",
                           "Typical time between two freezes."),
    "Analysed_time_s": ("Analysed time (s)", "End time − start time", "Length of the analysed time window."),
    "Start_time_s": ("Start time (s)", "", "Where the analysis starts in the video."),
    "End_time_s": ("End time (s)", "", "Where the analysis ends in the video."),
    "Animal_found_pct": ("Animal found by tracking (%)", "Tracked frames ÷ all frames × 100",
                         "How often BT tracking saw the animal (empty = not tracked: body size measured by you and "
                         "motion counted in the whole region)."),
    "Movement_outside_pct": ("Freezing frames with movement elsewhere (%)",
                             "Freezing frames whose whole-region motion is above the threshold ÷ freezing frames × 100",
                             "Only for 'Around the animal': movement away from the animal while it froze (a swaying "
                             "cable, a hand — fine — or the circle is not on the animal; check the review video)."),
    "Light_change_s": ("Light changes (s)", "Frames where the typical tile changed > 25 % ÷ FPS",
                       "Time when the whole picture changed at once (light on / off). These frames keep the state "
                       "of the frame before."),
    "Lights_off_trimmed_s": ("Not analysed: light off / on (s)",
                             "Frames at the start or end whose brightness is < 0.55 × or > 1.8 × the session's ÷ FPS",
                             "The light was switched off at the end (or on late at the start): the animal cannot be "
                             "seen and a dark, still picture would count as freezing, so these frames are left out "
                             "(Light-change guard on)."),
    "Result_check": ("Result check", "Automatic checks of this result",
                     "OK, or what to check before using the number (animal hardly found, cutoff at its maximum, "
                     "body size not found / far from your measurement, animal colour, light changes)."),
    # body dots (when BT tracking found the animal's outline)
    "Tail_only_movement_s": ("Tail-only movement (s)", "Frames where the tail moved but the body stayed below the "
                             "freezing threshold ÷ FPS",
                             "Time when ONLY the tail moved. With 'Body outline points' this time can still be "
                             "freezing."),
    "Head_moving_s": ("Head moving (s)", "Frames where the head moved ÷ FPS",
                      "Head = a circle of 0.3 body length around the nose dot."),
    "Legs_moving_s": ("Leg area moving (s)", "Frames where the leg areas moved ÷ FPS",
                      "Leg areas = small circles around 4 points placed on the body outline (front / hind, left / "
                      "right). They are estimated from the outline's shape — not detected paws."),
    "Tail_moving_s": ("Tail moving (s)", "Frames where the tail moved ÷ FPS", "Tail = the thin part behind the "
                                                                              "tail base."),
    "Mean_head_motion_pct": ("Mean head motion (% of body)", "Average changed pixels of the head ÷ body area",
                             "How much the head moved on average."),
    "Mean_legs_motion_pct": ("Mean leg-area motion (% of body)", "Average changed pixels of the leg areas ÷ body area",
                             "How much the legs moved on average."),
    "Mean_tail_motion_pct": ("Mean tail motion (% of body)", "Average changed pixels of the tail ÷ body area",
                             "How much the tail moved on average."),
    "Body_dots_found_pct": ("Body outline found (%)", "Frames with the body outline ÷ all frames × 100",
                            "How often the outline (and its dots) was found."),
}
SETTINGS = {
    "Motion_counted": ("Motion counted in", "Whole region: every pixel of the region box. Around the animal: a circle "
                                            "of 1 body length around the tracked animal. Body outline points: the animal's "
                                            "body (head, legs, trunk) — the tail is not counted."),
    "Motion_colour": ("Motion colour", "Gray: a pixel changes when its grey value changes. RGB: when any colour "
                                       "channel changes."),
    "Motion_cutoff": ("Motion cutoff (grey levels)", "A pixel 'moved' when its grey value changed by at least this "
                                                     "much in 1/30 s."),
    "Motion_cutoff_mode": ("Motion cutoff set by", "Auto = 2 × the 99.99th percentile of the camera noise of this "
                                                   "video (classic calibration rule; with spiky webcam noise the "
                                                   "99.99th percentile is limited to 1.6 × the 99.9th), or your own "
                                                   "value."),
    "Noise_level": ("Camera noise (grey levels)", "99.99th percentile of the frame-to-frame change of the quiet part "
                                                   "of the picture (limited to 1.6 × the 99.9th percentile)."),
    "Freezing_threshold_pct": ("Freezing threshold (% of body)", "Motion below this = still."),
    "Freezing_threshold_px": ("Freezing threshold (changed px)", "Threshold % × body area: the same threshold in "
                                                                 "changed pixels of the video (for comparison with "
                                                                 "pixel-based tools)."),
    "Min_freeze_s": ("Minimum freeze duration (s)", "Still for at least this long = freezing."),
    "Ignore_blips_s": ("Ignore motion blips shorter than (s)", "Short movements inside a freeze that do not break "
                                                              "it (0 = off)."),
    "Light_guard": ("Light-change guard", "On: frames where the whole picture changes at once are not used."),
    "FPS": ("Frames per second", "From the video file."),
    "Frame_lag": ("Frames compared apart", "1 at 25–45 fps, 2 at 60 fps (= 1/30 s): same numbers at any frame rate."),
    "Body_size_from": ("Body size from", "Measured by you (2 clicks), the body outline (Body outline points mode: outline "
                                         "area without the tail, measured on 16 frames) or BT tracking."),
    "Body_area_px": ("Body area (px²)", "Body length² ÷ 2.4 when measured; 90th percentile of the tracked size "
                                       "otherwise."),
    "Body_length_px": ("Body length (px)", "Nose to tail base."),
    "Region": ("Region", "The box the motion is counted in (video pixels)."),
    "Animal_colour": ("Animal colour (tracking)", "What BT tracking found / used."),
}
PERIOD_COLS = {
    "Freezing_pct": "Freezing (%)", "Freezing_time_s": "Freezing time (s)", "Freezing_bouts": "Freezing bouts",
    "Mean_bout_s": "Mean bout (s)", "Mean_motion_pct": "Mean motion (% of body)", "Duration_s": "Duration (s)",
}
EPOCH_EXTRA = {
    "Pre_freezing_pct": ("Before (%)", "Freezing in the window just before the epoch (pre-tone baseline)"),
    "Change_vs_pre_pct": ("Change vs before (% points)", "Epoch freezing − before freezing"),
    "Suppression_ratio": ("Suppression ratio", "Epoch motion ÷ (epoch motion + before motion); 0.5 = no change, "
                                               "below 0.5 = less movement during the epoch"),
}
GROUP_COLS = {
    "Group": "Kind of epoch", "Trials": "Trials", "Mean_freezing_pct": "Mean freezing (%)",
    "SD_freezing_pct": "SD (%)", "First_pct": "First trial (%)", "Last_pct": "Last trial (%)",
    "Change_first_last_pct": "Last − first (% points)", "Slope_per_trial_pct": "Slope (% points per trial)",
    "Mean_change_vs_pre_pct": "Mean change vs before (% points)", "Mean_suppression_ratio": "Mean suppression ratio",
}
GROUP_GUIDE = {
    "Kind of epoch": "Epochs with the same name apart from the number (Tone 1, Tone 2 … = Tone).",
    "Mean freezing (%)": "Average freezing over the trials of this kind.",
    "Last − first (% points)": "Last trial − first trial: negative = less freezing over the session (extinction).",
    "Slope (% points per trial)": "Straight-line trend over the trials (least squares).",
    "Mean change vs before (% points)": "Average of (trial − the window before it).",
    "Mean suppression ratio": "Average suppression ratio of the trials.",
}
BODY_KEYS = ["Tail_only_movement_s", "Head_moving_s", "Legs_moving_s", "Tail_moving_s", "Mean_head_motion_pct",
             "Mean_legs_motion_pct", "Mean_tail_motion_pct", "Body_dots_found_pct"]
RESULT_KEYS = ["Freezing_pct", "Freezing_time_s", "Active_time_s", "Freezing_bouts", "Mean_bout_s", "Median_bout_s",
               "Longest_bout_s", "Latency_first_freeze_s", "Bout_rate_per_min", "Mean_interbout_s",
               "Median_interbout_s", "Mean_motion_pct"] + BODY_KEYS
ANALYSED_WITH = "Analysed with"
ANALYSED_WITH_TIP = ("The BehavioralTracker version that computed these numbers. Analyse every video of one study "
                     "with the same version: detection and defaults can change between versions.")
INFO_KEYS = ["Result_check", "Start_time_s", "End_time_s", "Analysed_time_s", "Animal_found_pct",
             "Movement_outside_pct", "Light_change_s", "Lights_off_trimmed_s"]
SETTING_KEYS = list(SETTINGS)
SUMMARY_KEYS = ["Freezing_pct", "Freezing_time_s", "Freezing_bouts", "Mean_bout_s", "Longest_bout_s",
                "Latency_first_freeze_s", "Bout_rate_per_min", "Mean_motion_pct"]


def _v(x, nd=3):
    if isinstance(x, (float, np.floating)):
        if not math.isfinite(float(x)):
            return None
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _period_table(rows, first, extra=False):
    if not rows:
        return pd.DataFrame()
    cols = [first, "Start_s", "End_s"] + list(PERIOD_COLS) + (list(EPOCH_EXTRA) if extra else [])
    df = pd.DataFrame([{c: _v(r.get(c)) for c in cols} for r in rows])
    return df.rename(columns={"Start_s": "Start (s)", "End_s": "End (s)", **PERIOD_COLS,
                              **{k: v[0] for k, v in EPOCH_EXTRA.items()}})


def group_table(groups):
    if not groups:
        return pd.DataFrame()
    df = pd.DataFrame([{k: _v(g.get(k)) for k in GROUP_COLS} for g in groups])
    df["Trials (%)"] = [", ".join(f"{x:.1f}" for x in g.get("Trial_values") or []) for g in groups]
    return df.rename(columns=GROUP_COLS)


def frame_table(md, result):
    d = {"Frame": md.frames, "Time_s": np.round(result["times"], 4),
         "Motion_pct": np.round(result["motion"], 3), "Freezing": result["freezing"].astype(int),
         "Light_change": result["light"].astype(int)}
    if result.get("keep") is not None and not np.all(result["keep"]):
        d["Analysed"] = np.asarray(result["keep"]).astype(int)
    if md.xs is not None:
        d["Animal_X"] = np.round(md.xs, 2)
        d["Animal_Y"] = np.round(md.ys, 2)
        d["Animal_seen"] = md.seen.astype(int)
    parts = result.get("parts")
    if parts is not None:
        for name in ("body", "head", "legs", "tail"):
            d[f"{name.capitalize()}_motion_pct"] = np.round(parts["pct"][name], 3)
        mv = parts["moved"]
        d["Moving_parts"] = [", ".join(n for n in ("head", "legs", "tail") if mv[n][i]) or "—"
                             for i in range(len(md.frames))]
        kp = getattr(md, "kp", None)
        if kp is not None:
            geo = md.geo
            for j, name in ((0, "Nose"), (3, "Body_centre"), (6, "Tail_base")):
                d[f"{name}_X"] = np.round(kp[:, j, 0] / geo.s + geo.x0, 2)
                d[f"{name}_Y"] = np.round(kp[:, j, 1] / geo.s + geo.y0, 2)
    return pd.DataFrame(d)


FRAME_GUIDE = [
    ("Frame", "Frame number in the video (0 = first frame)."),
    ("Time_s", "Video time of the frame."),
    ("Motion_pct", "Changed pixels (minus the camera noise of the frame) as % of the body area. Empty = not used "
                   "(first frame / light change)."),
    ("Freezing", "1 = freezing, 0 = active."),
    ("Light_change", "1 = the whole picture changed at once (frame keeps the state of the frame before)."),
    ("Analysed", "0 = left out: the light was off / on at the start or end (only when that happened)."),
    ("Animal_X / Animal_Y", "Tracked animal centre (video pixels); held at the last place when not seen."),
    ("Animal_seen", "1 = tracking saw the animal on this frame."),
    ("Body / Head / Legs / Tail_motion_pct", "Changed pixels of that body part as % of the body area (body = head + "
                                             "legs + trunk, without the tail)."),
    ("Moving_parts", "Parts that moved on this frame ('tail' alone = only the tail moved)."),
    ("Nose / Body_centre / Tail_base _X _Y", "The body dots (video pixels)."),
]


def body_size_png(md, result, path):
    """The body size the threshold is based on, drawn on a frame of the video (tracking.freezing.body_size_image).
    None when the video cannot be read."""
    import cv2
    from tracking.freezing import body_size_frame, body_size_image
    vp = getattr(md, "path", None)
    if not vp or not os.path.exists(str(vp)):
        return None
    i = body_size_frame(md)
    cap = cv2.VideoCapture(str(vp))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(md.frames[i]))
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    img = body_size_image(md, result, img, i)
    h, w = img.shape[:2]
    if max(h, w) > 1600:
        f = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return path if cv2.imwrite(path, img) else None


def save_plot(md, result, path, title=""):
    """The big motion trace (kept for older callers)."""
    from output.freezing_graphs import motion_trace
    return motion_trace(result, md.fps, path, title=title)


def _info_rows(md, s, video_path, out_dir, subject="", group=""):
    rows = [("Video", os.path.basename(str(video_path)), "", "The video file."),
            ("Video file", str(video_path), "", "Full path of the video."),
            ("Results folder", str(out_dir), "", "Where these files are saved.")]
    if subject:
        rows.append(("Subject", subject, "", "Animal ID."))
    if group:
        rows.append(("Group", group, "", "Experimental group."))
    for k in INFO_KEYS:
        lab, f, mean = MEASURES[k]
        rows.append((lab, _v(s.get(k)), f, mean))
    rows.append((ANALYSED_WITH, BT_APP, "", ANALYSED_WITH_TIP))
    return pd.DataFrame(rows, columns=["Measure", "Value", "Formula", "What it means"])


def write_freezing_report(md, result, video_path, out_dir=None, subject="", group=""):
    """Write the per-video files.  Returns dict(xlsx, csv, png, charts, folder)."""
    from tracking.freezing import default_output_dir
    out_dir = out_dir or default_output_dir(video_path)
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(str(video_path)))[0]
    s = result["summary"]
    xlsx = os.path.join(out_dir, f"{name}_Freezing.xlsx")
    csv = os.path.join(out_dir, f"{name}_Freezing_frames.csv")
    frames = frame_table(md, result)
    frames.to_csv(csv, index=False)
    old = os.path.join(out_dir, f"{name}_Freezing_plot.png")       # name of the first version
    if os.path.exists(old):
        try:
            os.remove(old)
        except OSError:
            pass
    from output.freezing_graphs import make_all
    charts = make_all(result, md.fps, out_dir, name)
    png = charts.get("motion_trace")
    bs = body_size_png(md, result, os.path.join(out_dir, f"{name}_Freezing_body_size.png"))
    if bs:
        charts["body_size"] = bs
    pdf = None
    try:
        from output.pdf_report import freezing_pdf
        pdf = freezing_pdf(md, result, charts, os.path.join(out_dir, f"{name}_Freezing_Summary.pdf"), name,
                           subject, group)
    except Exception as exc:
        print(f"PDF summary skipped: {exc}")
    slide = None
    try:                                         # a 16:9 picture for a lab-meeting slide
        from output.pdf_report import freezing_slide
        slide = freezing_slide(md, result, charts, os.path.join(out_dir, f"{name}_Freezing_slide.png"), name,
                               subject, group)
    except Exception as exc:
        print(f"Slide picture skipped: {exc}")
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        sh = _Sheet(w, "Freezing results")
        sh.title(f"Freezing analysis — {name}")
        sh.table(pd.DataFrame([{"Measure": MEASURES[k][0], "Value": _v(s.get(k)), "Formula": MEASURES[k][1],
                                "What it means": MEASURES[k][2]} for k in RESULT_KEYS
                               if k not in BODY_KEYS or k in s]), "Freezing")
        sh.table(_info_rows(md, s, video_path, out_dir, subject, group), "Video & tracking info")
        sh.finish({"A": 34, "B": 22, "C": 46, "D": 90}, wrap=("C", "D"))
        ep = _period_table(result["epochs"], "Epoch", extra=True)
        if not ep.empty:
            ep.to_excel(w, sheet_name="Epochs", index=False)
            style_sheet(w.sheets["Epochs"])
            _auto_widths(w.sheets["Epochs"], ep)
        gt = group_table(result.get("groups"))
        if not gt.empty:
            gt.to_excel(w, sheet_name="Trials by kind", index=False)
            style_sheet(w.sheets["Trials by kind"])
            _auto_widths(w.sheets["Trials by kind"], gt, max_w=46)
        bn = _period_table(result["bins"], "Bin")
        if not bn.empty:
            bn.to_excel(w, sheet_name="Time bins", index=False)
            style_sheet(w.sheets["Time bins"])
            _auto_widths(w.sheets["Time bins"], bn)
        bt = pd.DataFrame([{"Bout": b["Bout"], "Start (s)": _v(b["Start_s"]), "End (s)": _v(b["End_s"]),
                            "Duration (s)": _v(b["Duration_s"])} for b in result["bouts"]],
                          columns=["Bout", "Start (s)", "End (s)", "Duration (s)"])
        bt.to_excel(w, sheet_name="Freezing bouts", index=False)
        style_sheet(w.sheets["Freezing bouts"])
        _auto_widths(w.sheets["Freezing bouts"], bt)
        st = pd.DataFrame([{"Setting": SETTINGS[k][0], "Value": _v(s.get(k)), "What it means": SETTINGS[k][1]}
                           for k in SETTING_KEYS])
        st.to_excel(w, sheet_name="Settings", index=False)
        style_sheet(w.sheets["Settings"], {"A": 36, "B": 30, "C": 100}, wrap=("C",))
        frames.to_excel(w, sheet_name="Frame by frame", index=False)
        style_sheet(w.sheets["Frame by frame"])
        _auto_widths(w.sheets["Frame by frame"], frames)
        guide = [{"Sheet": "Freezing results", "Column": MEASURES[k][0], "Formula": MEASURES[k][1],
                  "What it means": MEASURES[k][2]} for k in RESULT_KEYS + INFO_KEYS]
        guide += [{"Sheet": "Epochs / Time bins", "Column": lab, "Formula": "", "What it means":
                   "The same measure inside that epoch / time bin."} for lab in PERIOD_COLS.values()]
        pre_s = float(result["settings"].get("pre_s") or 0)
        guide += [{"Sheet": "Epochs", "Column": lab, "Formula": f, "What it means":
                   f"'Before' = the {pre_s:g} s just before the epoch starts."} for lab, f in EPOCH_EXTRA.values()]
        guide += [{"Sheet": "Trials by kind", "Column": k, "Formula": "", "What it means": m}
                  for k, m in GROUP_GUIDE.items()]
        guide += [{"Sheet": "Frame by frame", "Column": c, "Formula": "", "What it means": m} for c, m in FRAME_GUIDE]
        guide += [{"Sheet": "Settings", "Column": SETTINGS[k][0], "Formula": "", "What it means": SETTINGS[k][1]}
                  for k in SETTING_KEYS]
        gd = pd.DataFrame(guide)
        gd.to_excel(w, sheet_name="Column guide", index=False)
        style_sheet(w.sheets["Column guide"], {"A": 22, "B": 40, "C": 60, "D": 100}, wrap=("C", "D"))
    return {"xlsx": xlsx, "csv": csv, "png": png, "charts": charts, "folder": out_dir, "pdf": pdf, "slide": slide}


# ============================================================================= batch
def batch_row(video_path, result, subject="", group="", folder="", extra=None):
    """One flat row for the batch summary."""
    s = result["summary"]
    row = {"Video": os.path.basename(str(video_path)), "Subject": subject or "", "Group": group or "",
           "Video file": str(video_path), "Results folder": folder}
    for k, v in (extra or {}).items():
        row.setdefault(k, v)
    for k in RESULT_KEYS + INFO_KEYS + SETTING_KEYS:
        row[k] = _v(s.get(k))
    row["_epochs"] = result["epochs"]
    row["_bins"] = result["bins"]
    row["_groups"] = result.get("groups") or []
    return row


def write_freezing_batch(rows, path):
    """rows = batch_row() per video -> the batch workbook.  Returns the path."""
    ident = ["Video"] + [c for c in ("Subject", "Group") if any(str(r.get(c) or "").strip() for r in rows)]
    extra = []
    for r in rows:
        for k in r:
            if k not in extra and not k.startswith("_") and k not in RESULT_KEYS + INFO_KEYS + SETTING_KEYS \
                    and k not in ("Video", "Subject", "Group", "Video file", "Results folder") \
                    and any(str(x.get(k) or "").strip() for x in rows):
                extra.append(k)
    ident += extra
    ep_names = []
    for r in rows:
        for e in r["_epochs"]:
            if e["Epoch"] not in ep_names:
                ep_names.append(e["Epoch"])
    n_bins = max([len(r["_bins"]) for r in rows] + [0])
    kinds = []
    for r in rows:
        for g in r.get("_groups") or []:
            if g["Group"] not in kinds:
                kinds.append(g["Group"])

    multi_kinds = [k for k in kinds if any(g["Group"] == k and g["Trials"] > 1 for r in rows
                                           for g in r.get("_groups") or [])]

    def base(r):
        return {c: r.get(c, "") for c in ident}

    summ, epw, binw, info, sett, grw = [], [], [], [], [], []
    for r in rows:
        a = base(r)
        for k in SUMMARY_KEYS:
            a[MEASURES[k][0]] = r.get(k)
        a[MEASURES["Result_check"][0]] = r.get("Result_check") or ""
        gs = {g["Group"]: g for g in r.get("_groups") or []}
        for kd in multi_kinds:
            a[f"{kd} – mean freezing (%)"] = _v(gs[kd]["Mean_freezing_pct"]) if kd in gs else None
        eps = {e["Epoch"]: e for e in r["_epochs"]}
        for n in ep_names:
            a[f"{n} – freezing (%)"] = _v(eps[n]["Freezing_pct"]) if n in eps else None
        summ.append(a)
        g_ = base(r)
        for kd in kinds:
            gg = gs.get(kd) or {}
            for k in ("Trials", "Mean_freezing_pct", "First_pct", "Last_pct", "Change_first_last_pct",
                      "Slope_per_trial_pct", "Mean_change_vs_pre_pct", "Mean_suppression_ratio"):
                g_[f"{kd} – {GROUP_COLS[k].lower()}"] = _v(gg.get(k)) if gg else None
        grw.append(g_)
        b = base(r)
        for n in ep_names:
            e = eps.get(n)
            b[f"{n} – start (s)"] = _v(e["Start_s"]) if e else None
            b[f"{n} – end (s)"] = _v(e["End_s"]) if e else None
            for k in ("Freezing_pct", "Freezing_time_s", "Freezing_bouts", "Mean_motion_pct"):
                b[f"{n} – {PERIOD_COLS[k].lower()}"] = _v(e[k]) if e else None
            for k, (lab, _f) in EPOCH_EXTRA.items():
                b[f"{n} – {lab.lower()}"] = _v(e.get(k)) if e else None
        epw.append(b)
        c = base(r)
        for bi in r["_bins"]:
            c[f"Bin {bi['Bin']} ({bi['Start_s']:g}–{bi['End_s']:g} s) – freezing (%)"] = _v(bi["Freezing_pct"])
        binw.append(c)
        d = base(r)
        d["Video file"] = r.get("Video file", "")
        d["Results folder"] = r.get("Results folder", "")
        for k in INFO_KEYS:
            d[MEASURES[k][0]] = r.get(k)
        d[ANALYSED_WITH] = BT_APP
        info.append(d)
        e = base(r)
        for k in SETTING_KEYS:
            e[SETTINGS[k][0]] = r.get(k)
        sett.append(e)
    guide = [{"Sheet": "Freezing summary", "Column": MEASURES[k][0], "Formula": MEASURES[k][1],
              "What it means": MEASURES[k][2]} for k in SUMMARY_KEYS + ["Result_check"]]
    guide += [{"Sheet": "Freezing summary / Epochs", "Column": "<epoch> – …", "Formula": "",
               "What it means": "The same measure inside that epoch (start / end in video seconds)."},
              {"Sheet": "Time bins", "Column": "Bin n (a–b s) – freezing (%)", "Formula": "",
               "What it means": "Freezing (%) inside that time bin (seconds from the start of the analysis)."}]
    guide += [{"Sheet": "Freezing summary", "Column": "<kind> – mean freezing (%)", "Formula": "",
               "What it means": "Average over the trials of that kind of epoch (Tone 1, Tone 2 … = Tone)."}]
    guide += [{"Sheet": "Epochs", "Column": f"<epoch> – {lab.lower()}", "Formula": f, "What it means": ""}
              for lab, f in EPOCH_EXTRA.values()]
    guide += [{"Sheet": "Trials by kind", "Column": f"<kind> – {k.lower()}", "Formula": "", "What it means": m}
              for k, m in GROUP_GUIDE.items() if k != "Kind of epoch"]
    guide += [{"Sheet": "Video & tracking info", "Column": MEASURES[k][0], "Formula": MEASURES[k][1],
               "What it means": MEASURES[k][2]} for k in INFO_KEYS]
    guide += [{"Sheet": "Video & tracking info", "Column": ANALYSED_WITH, "Formula": "", "What it means": ANALYSED_WITH_TIP}]
    guide += [{"Sheet": "Settings", "Column": SETTINGS[k][0], "Formula": "", "What it means": SETTINGS[k][1]}
              for k in SETTING_KEYS]
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, data in (("Freezing summary", summ), ("Epochs", epw if ep_names else None),
                           ("Trials by kind", grw if kinds else None),
                           ("Time bins", binw if n_bins else None), ("Video & tracking info", info),
                           ("Settings", sett)):
            if not data:
                continue
            df = pd.DataFrame(data)
            df.to_excel(w, sheet_name=name, index=False)
            style_sheet(w.sheets[name], freeze="B2")
            _auto_widths(w.sheets[name], df, max_w=34)
        from output.tidy import prism_sheet
        sm = pd.DataFrame(summ)
        pm = [(MEASURES[k][0], MEASURES[k][0]) for k in ("Freezing_pct", "Freezing_time_s", "Freezing_bouts",
                                                         "Latency_first_freeze_s", "Mean_bout_s")]
        pm += [(c, c) for c in sm.columns if c.endswith("– mean freezing (%)") or c.endswith("– freezing (%)")]
        if "Group" in sm.columns and "Subject" in sm.columns and prism_sheet(w, sm, "Group", "Subject", pm):
            guide.append({"Sheet": "For Prism", "Column": "(every block)", "Formula": "",
                          "What it means": "One block per measure: the columns are the groups, the rows the "
                                           "animals. Select a block, copy and paste it into a GraphPad Prism "
                                           "Column table."})
        gd = pd.DataFrame(guide)
        gd.to_excel(w, sheet_name="Column guide", index=False)
        style_sheet(w.sheets["Column guide"], {"A": 26, "B": 40, "C": 60, "D": 100}, wrap=("C", "D"))
    try:
        write_freezing_tidy(rows, ident, path.replace(".xlsx", "_tidy.csv"))
    except Exception as exc:
        print(f"Tidy CSV skipped: {exc}")
    return path


def write_freezing_tidy(rows, ident, path):
    """The batch as one long table (one row per animal x measure x epoch / bin) -> path."""
    from output.tidy import tidy_csv
    out = []
    for r in rows:
        base = {c: r.get(c, "") for c in ident}
        for k in SUMMARY_KEYS + ["Analysed_time_s", "Animal_found_pct"]:
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)):
                lab = MEASURES[k][0]
                out.append(dict(base, Period="Whole analysis", Start_s=None, End_s=None,
                                Measure=lab.rsplit(" (", 1)[0], Unit=(lab.rsplit(" (", 1)[1].rstrip(")")
                                                                      if " (" in lab else ""), Value=float(v)))
        for e in r.get("_epochs") or []:
            for k in ("Freezing_pct", "Freezing_time_s", "Freezing_bouts", "Mean_motion_pct"):
                v = e.get(k)
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    lab = PERIOD_COLS[k]
                    out.append(dict(base, Period=f"Epoch: {e['Epoch']}", Start_s=e["Start_s"], End_s=e["End_s"],
                                    Measure=lab.rsplit(" (", 1)[0],
                                    Unit=lab.rsplit(" (", 1)[1].rstrip(")") if " (" in lab else "", Value=float(v)))
        for b in r.get("_bins") or []:
            v = b.get("Freezing_pct")
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                out.append(dict(base, Period=f"Bin {b['Bin']}", Start_s=b["Start_s"], End_s=b["End_s"],
                                Measure="Freezing", Unit="%", Value=float(v)))
    return tidy_csv(out, path)
