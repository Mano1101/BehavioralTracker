"""Test-specific behavioural measures computed from the per-frame zone flags.

Everything here works on the tracking DataFrame produced by
tracking.location.process_single_video (columns Time_seconds, Tracking_Status,
In_<zone>, Distance_*) and plain zone names, so it is independent of how the
zones were drawn (template, auto-detected, hand-drawn).

Zone roles are recognised from the zone NAMES (case-insensitive):
  centre / center / hub        -> hub zone of a maze
  open / closed                -> EPM arm type
  light / dark                 -> light-dark box compartments
  goal, left, right / stem     -> T-maze goal arms / stem
  novel, familiar, object      -> NOR objects
  social, stranger / object, empty -> 3-chamber sides
So renaming a zone in the zone editor is all that is needed to tell the
analysis what a zone is.
"""
import re

import numpy as np
import pandas as pd

from analysis.calculations import safe_col


# ----------------------------------------------------------------------------- generic helpers
def _has(name, *words):
    n = str(name).lower()
    return any(w in n for w in words)


def is_hub(name):
    return _has(name, "center", "centre", "hub", "middle")


def debounced_state(flags, min_frames):
    """Boolean per-frame series -> debounced boolean array.  A run of the
    opposite value shorter than `min_frames` is ignored (keeps the previous
    state); a long enough run switches the state for its whole length."""
    flags = np.asarray(flags, bool)
    n = len(flags)
    out = np.zeros(n, bool)
    if n == 0:
        return out
    state = bool(flags[0])
    i = 0
    while i < n:
        v = bool(flags[i])
        j = i
        while j < n and bool(flags[j]) == v:
            j += 1
        if v != state and (j - i) >= min_frames:
            state = v
        out[i:j] = state
        i = j
    return out


PRESENT = ("Tracked", "Hidden")      # seen, or hidden in a covered zone (e.g. under the lid of the dark box)
ENTRY_FRACTION = 0.80     # four-paw entry: this fraction of the body (tail removed) inside the zone


def has_body_columns(df, zones):
    return bool(zones) and all(f"Body_{safe_col(z)}" in df.columns for z in zones)


def paw_state(df, zones, min_frames=3, frac=ENTRY_FRACTION):
    """'All four paws' zone state.  The animal ENTERS a zone when >= `frac` of its body is inside it for
    `min_frames` frames in a row; it stays in that zone until it enters another one this way (a head poked
    into the next zone, or a lost frame, does not count as leaving).
    Returns (per-frame zone label array, ["None" before the first entry], entries [(time_s, zone), ...])
    or None when the body columns are missing."""
    zones = list(zones)
    if not has_body_columns(df, zones):
        return None
    n = len(df)
    F = np.column_stack([df[f"Body_{safe_col(z)}"].to_numpy(float) for z in zones]) if zones else np.zeros((n, 0))
    t = df["Time_seconds"].to_numpy(float)
    Fz = np.where(np.isfinite(F), F, -1.0)
    best = Fz.argmax(axis=1) if zones else np.zeros(n, int)
    bestv = Fz[np.arange(n), best] if zones else np.full(n, -1.0)
    cand = np.where(bestv >= frac - 1e-9, best, -1)
    labels = np.empty(n, dtype=object)
    labels[:] = "None"
    state, c, cnt, c_start = -1, -1, 0, 0
    entries = []
    for i in range(n):
        k = int(cand[i])
        if k >= 0 and k != state:
            if k == c:
                cnt += 1
            else:
                c, cnt, c_start = k, 1, i
            if cnt >= min_frames:
                state = k
                entries.append((float(t[c_start]), zones[k]))
                labels[c_start:i] = zones[k]
                c, cnt = -1, 0
        elif k == state and k >= 0:
            c, cnt = -1, 0
        if state >= 0:
            labels[i] = zones[state]
    return labels, entries


def zone_visits(df, zone, fps, min_frames=3):
    """List of (start_s, end_s) visits for one zone (tracked frames only)."""
    col = f"In_{safe_col(zone)}"
    if col not in df.columns:
        return []
    tracked = np.isin(df["Tracking_Status"].to_numpy(), PRESENT)
    flags = df[col].fillna(False).to_numpy(bool)
    # untracked frames: keep the previous known state (no fake exits)
    f2 = flags.copy()
    last = False
    for i in range(len(f2)):
        if tracked[i]:
            last = f2[i]
        else:
            f2[i] = last
    st = debounced_state(f2, min_frames)
    t = df["Time_seconds"].to_numpy(float)
    visits, i, n = [], 0, len(st)
    while i < n:
        if st[i]:
            j = i
            while j < n and st[j]:
                j += 1
            visits.append((float(t[i]), float(t[j - 1]) + 1.0 / fps))
            i = j
        else:
            i += 1
    return visits


def zone_table(df, zones, fps, start_time, distance_col=None, min_frames=3, entry_frac=ENTRY_FRACTION):
    """Per-zone time / % / entries / latency / mean visit / distance.
    Time = body centre point in the zone.  Entries + latency = all four paws in the zone (when the body
    columns exist for that zone), otherwise the centre point."""
    tracked = df[df["Tracking_Status"].isin(PRESENT)]
    tracked_time = len(tracked) / fps if fps else 0.0
    paw_z = [z for z in zones if f"Body_{safe_col(z)}" in df.columns]
    paw = paw_state(df, paw_z, min_frames, entry_frac) if paw_z else None
    rows = []
    for z in zones:
        col = f"In_{safe_col(z)}"
        inside = tracked[col].fillna(False).astype(bool) if col in tracked.columns else pd.Series(dtype=bool)
        t_in = float(inside.sum()) / fps if fps else 0.0
        if paw is not None and z in paw_z:
            ent_t = [t for t, zz in paw[1] if zz == z]
            entries = len(ent_t)
            first = ent_t[0] if ent_t else None
        else:
            visits = zone_visits(df, z, fps, min_frames)
            entries = len(visits)
            first = visits[0][0] if visits else None
        lat = (max(0.0, first - start_time) if first is not None else None)
        dist = None
        if distance_col and distance_col in tracked.columns and len(inside):
            dist = float(tracked.loc[inside.values, distance_col].sum()) if inside.any() else 0.0
        rows.append({"Zone": z, "Time_s": round(t_in, 3),
                     "Percent_time": round(100.0 * t_in / tracked_time, 2) if tracked_time > 0 else 0.0,
                     "Entries": entries,
                     "Latency_first_entry_s": None if lat is None else round(lat, 3),
                     "Mean_visit_s": round(t_in / entries, 3) if entries else 0.0,
                     "Distance_in_zone": None if dist is None else round(dist, 3)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- maze arm sequences
def arm_entry_sequence(df, arms, hubs, fps, min_frames=3, entry_frac=ENTRY_FRACTION):
    """Ordered list of (time_s, arm) arm ENTRIES -- only arms, the centre is never an entry.

    With body columns: an arm entry = all four paws in the arm; the animal must fully enter the centre or
    another arm before the same arm counts again."""
    if arms and has_body_columns(df, list(arms) + list(hubs)):
        res = paw_state(df, list(arms) + list(hubs), min_frames, entry_frac)
        if res is not None:
            return [(t, z) for t, z in res[1] if z in arms]
    return _arm_entry_sequence_centroid(df, arms, hubs, fps, min_frames)


def _arm_entry_sequence_centroid(df, arms, hubs, fps, min_frames=3):
    """Ordered list of (time_s, arm) arm ENTRIES (body centre point).

    A re-entry into the same arm only counts when the animal has visited the
    hub (or another arm) in between -- briefly losing the track or crossing a
    tiny gap between zones never creates a fake entry."""
    if not arms:
        return []
    tracked = np.isin(df["Tracking_Status"].to_numpy(), PRESENT)
    t = df["Time_seconds"].to_numpy(float)
    arm_cols = [(a, f"In_{safe_col(a)}") for a in arms]
    hub_cols = [f"In_{safe_col(h)}" for h in hubs]
    labels = []
    for i in range(len(df)):
        lab = None
        if tracked[i]:
            for a, c in arm_cols:
                if c in df.columns and bool(df[c].iat[i]):
                    lab = a
                    break
            if lab is None:
                for c in hub_cols:
                    if c in df.columns and bool(df[c].iat[i]):
                        lab = "__HUB__"
                        break
            if lab is None and not hub_cols:
                lab = "__HUB__"        # no centre zone drawn: 'outside every arm' acts as the centre
        labels.append(lab)
    # debounce: a new label is adopted after min_frames consecutive frames
    state, cand, cnt = None, None, 0
    seq, last_arm, left_since = [], None, True
    for i, lab in enumerate(labels):
        if lab is None or lab == state:
            cand, cnt = None, 0
            continue
        if lab == cand:
            cnt += 1
        else:
            cand, cnt = lab, 1
        if cnt >= min_frames:
            state = lab
            start_i = i - cnt + 1
            if state == "__HUB__":
                left_since = True
            else:
                if state != last_arm or left_since:
                    seq.append((float(t[start_i]), state))
                last_arm = state
                left_since = False
            cand, cnt = None, 0
    return seq


def alternation_metrics(seq_names):
    """Spontaneous alternation (Y-maze): overlapping triplets of three
    different arms / (entries - 2) x 100, plus same-arm (SAR) and
    alternate-arm (AAR) returns."""
    n = len(seq_names)
    alt = sum(1 for i in range(n - 2) if len({seq_names[i], seq_names[i + 1], seq_names[i + 2]}) == 3)
    sar = sum(1 for i in range(n - 1) if seq_names[i] == seq_names[i + 1])
    aar = sum(1 for i in range(n - 2) if seq_names[i] == seq_names[i + 2] and seq_names[i] != seq_names[i + 1])
    return {"Total_arm_entries": n, "Alternations": alt, "Possible_alternations": max(0, n - 2),
            "Spontaneous_alternation_pct": round(100.0 * alt / (n - 2), 2) if n >= 3 else None,
            "Same_arm_returns": sar, "Alternate_arm_returns": aar}


# ----------------------------------------------------------------------------- per test
def classify_test(test_key, zones):
    if test_key:
        return test_key
    names = " ".join(z.lower() for z in zones)
    if "open" in names and "closed" in names:
        return "epm"
    if "light" in names and "dark" in names:
        return "light_dark"
    return "custom"


def _two_part_entries(df, part_a, part_b, fps, min_frames=3, entry_frac=ENTRY_FRACTION):
    """Entries into zone group A and zone group B (e.g. centre vs the whole periphery of an open field)."""
    cols_a = [f"Body_{safe_col(z)}" for z in part_a]
    cols_b = [f"Body_{safe_col(z)}" for z in part_b]
    if part_a and part_b and all(c in df.columns for c in cols_a + cols_b):
        tmp = pd.DataFrame({"Time_seconds": df["Time_seconds"].to_numpy(float),
                            f"Body_{safe_col('GroupA')}": np.clip(df[cols_a].sum(axis=1, min_count=1).to_numpy(float), 0, 1),
                            f"Body_{safe_col('GroupB')}": np.clip(df[cols_b].sum(axis=1, min_count=1).to_numpy(float), 0, 1)})
        res = paw_state(tmp, ["GroupA", "GroupB"], min_frames, entry_frac)
        if res is not None:
            ent = [z for _t, z in res[1]]
            return ent.count("GroupA"), ent.count("GroupB")
    # centre point: visits of A and of the union of B
    tmp = df.copy()
    ca = [f"In_{safe_col(z)}" for z in part_a if f"In_{safe_col(z)}" in df.columns]
    cb = [f"In_{safe_col(z)}" for z in part_b if f"In_{safe_col(z)}" in df.columns]
    tmp[f"In_{safe_col('GroupA')}"] = tmp[ca].any(axis=1) if ca else False
    tmp[f"In_{safe_col('GroupB')}"] = tmp[cb].any(axis=1) if cb else False
    return len(zone_visits(tmp, "GroupA", fps, min_frames)), len(zone_visits(tmp, "GroupB", fps, min_frames))


def anxiety_index(zone_time, total_time, zone_entries, total_entries):
    """Anxiety index = 1 - ((time in open arm / light zone / centre  / total time)
                           + (entries into it / total entries)) / 2."""
    if not total_time or not total_entries:
        return None
    return round(1.0 - ((zone_time / total_time) + (zone_entries / total_entries)) / 2.0, 4)


def compute_test_metrics(df, zones, fps, start_time, test_key=None, distance_col=None, min_frames=3,
                         entry_frac=ENTRY_FRACTION):
    """Returns (metrics_dict, zone_table_df, entries_df)."""
    zones = list(zones)
    test = classify_test(test_key, zones)
    zt = zone_table(df, zones, fps, start_time, distance_col, min_frames, entry_frac)
    total_time = len(df) / fps if fps else 0.0
    by = {r.Zone: r for r in zt.itertuples(index=False)}
    m = {"Test_type": test}
    entries_rows = []

    def tsum(names, field="Time_s"):
        return float(sum(getattr(by[n], field) for n in names if n in by))

    tracked_time = float(df["Tracking_Status"].isin(PRESENT).sum()) / fps if fps else 0.0

    if test == "epm":
        oa = [z for z in zones if _has(z, "open")]
        ca = [z for z in zones if _has(z, "closed", "close")]
        hub = [z for z in zones if is_hub(z)]
        oa_t, ca_t, c_t = tsum(oa), tsum(ca), tsum(hub)
        oa_e, ca_e = int(tsum(oa, "Entries")), int(tsum(ca, "Entries"))
        lat = [by[z].Latency_first_entry_s for z in oa if z in by and by[z].Latency_first_entry_s is not None]
        m.update({"Open_arm_time_s": round(oa_t, 3), "Closed_arm_time_s": round(ca_t, 3), "Center_time_s": round(c_t, 3),
                  "Open_arm_time_pct": round(100 * oa_t / tracked_time, 2) if tracked_time else None,
                  "Open_arm_time_ratio_pct": round(100 * oa_t / (oa_t + ca_t), 2) if (oa_t + ca_t) > 0 else None,
                  "Open_arm_entries": oa_e, "Closed_arm_entries": ca_e, "Total_arm_entries": oa_e + ca_e,
                  "Open_arm_entries_pct": round(100 * oa_e / (oa_e + ca_e), 2) if (oa_e + ca_e) > 0 else None,
                  "Latency_first_open_arm_s": min(lat) if lat else None})
        m["Anxiety_index"] = anxiety_index(oa_t, total_time, oa_e, oa_e + ca_e)
        seq = arm_entry_sequence(df, oa + ca, hub, fps, min_frames, entry_frac)
        entries_rows = [{"Entry": i + 1, "Time_s": round(t, 3), "Arm": a,
                         "Arm_type": "Open" if a in oa else "Closed"} for i, (t, a) in enumerate(seq)]
    elif test in ("y_maze", "t_maze"):
        hub = [z for z in zones if is_hub(z)]
        arms = [z for z in zones if z not in hub]
        seq = arm_entry_sequence(df, arms, hub, fps, min_frames, entry_frac)
        names = [a for _t, a in seq]
        m.update({f"{safe_col(a)}_arm_entries": names.count(a) for a in arms})
        if test == "y_maze":
            m.update(alternation_metrics(names))
        else:
            goals = [a for a in arms if _has(a, "goal", "left", "right")] or \
                    [a for a in arms if not _has(a, "stem", "start")]
            gseq = [a for a in names if a in goals]
            n = len(gseq)
            alt = sum(1 for i in range(1, n) if gseq[i] != gseq[i - 1])
            m.update({"Total_arm_entries": len(names), "Goal_arm_entries": n, "Goal_arm_alternations": alt,
                      "Goal_arm_alternation_pct": round(100.0 * alt / (n - 1), 2) if n >= 2 else None,
                      "First_goal_choice": gseq[0] if gseq else None})
            yl = alternation_metrics(names)
            m.update({"Spontaneous_alternation_pct": yl["Spontaneous_alternation_pct"],
                      "Alternations": yl["Alternations"], "Possible_alternations": yl["Possible_alternations"]})
        entries_rows = [{"Entry": i + 1, "Time_s": round(t, 3), "Arm": a} for i, (t, a) in enumerate(seq)]
    elif test == "light_dark":
        light = [z for z in zones if _has(z, "light", "bright", "white")]
        dark = [z for z in zones if _has(z, "dark", "black")]
        lt, dt = tsum(light), tsum(dark)
        seq = arm_entry_sequence(df, light + dark, [], fps, min_frames, entry_frac)
        kinds = ["Light" if a in light else "Dark" for _t, a in seq]
        trans = sum(1 for i in range(1, len(kinds)) if kinds[i] != kinds[i - 1])
        first_dark = next((t for t, a in seq if a in dark), None)
        first_light = next((t for t, a in seq if a in light), None)
        start_side = kinds[0] if kinds else None
        m.update({"Light_time_s": round(lt, 3), "Dark_time_s": round(dt, 3),
                  "Light_time_pct": round(100 * lt / (lt + dt), 2) if (lt + dt) > 0 else None,
                  "Transitions": trans, "Light_entries": int(tsum(light, "Entries")),
                  "Dark_entries": int(tsum(dark, "Entries")), "Start_compartment": start_side,
                  "Latency_to_dark_s": round(first_dark - start_time, 3) if (first_dark is not None and start_side != "Dark") else (0.0 if start_side == "Dark" else None),
                  "Latency_to_light_s": round(first_light - start_time, 3) if (first_light is not None and start_side != "Light") else (0.0 if start_side == "Light" else None)})
        le, de = int(tsum(light, "Entries")), int(tsum(dark, "Entries"))
        m["Anxiety_index"] = anxiety_index(lt, total_time, le, le + de)
        entries_rows = [{"Entry": i + 1, "Time_s": round(t, 3), "Compartment": k} for i, ((t, _a), k) in enumerate(zip(seq, kinds))]
    elif test == "open_field":
        hub = [z for z in zones if is_hub(z)]
        rest = [z for z in zones if z not in hub]
        c_t = tsum(hub)
        m.update({"Center_time_s": round(c_t, 3),
                  "Center_time_pct": round(100 * c_t / tracked_time, 2) if tracked_time else None,
                  "Center_entries": int(tsum(hub, "Entries")),
                  "Periphery_time_s": round(tsum(rest), 3)})
        if distance_col and hub:
            cd = sum((by[z].Distance_in_zone or 0) for z in hub if z in by)
            tot = float(df.loc[df["Tracking_Status"] == "Tracked", distance_col].sum())
            m["Center_distance"] = round(cd, 3)
            m["Center_distance_pct"] = round(100 * cd / tot, 2) if tot > 0 else None
        lat = [by[z].Latency_first_entry_s for z in hub if z in by and by[z].Latency_first_entry_s is not None]
        m["Latency_first_center_s"] = min(lat) if lat else None
        # anxiety index: centre vs the whole periphery (one zone), entries by the same rule as above
        c_e, p_e = _two_part_entries(df, hub, rest, fps, min_frames, entry_frac)
        m["Periphery_entries"] = p_e
        m["Anxiety_index"] = anxiety_index(c_t, total_time, c_e, c_e + p_e)
    elif test == "nor":
        objs = [z for z in zones if _has(z, "object", "novel", "familiar", "obj")]
        novel = [z for z in objs if _has(z, "novel", "new")]
        fam = [z for z in objs if _has(z, "familiar", "old")]
        times = {z: by[z].Time_s for z in objs if z in by}
        m.update({f"{safe_col(z)}_exploration_s": round(v, 3) for z, v in times.items()})
        m["Total_exploration_s"] = round(sum(times.values()), 3)
        if novel and fam:
            n_t, f_t = tsum(novel), tsum(fam)
            m["Discrimination_index"] = round((n_t - f_t) / (n_t + f_t), 4) if (n_t + f_t) > 0 else None
            m["Discrimination_ratio"] = round(n_t / (n_t + f_t), 4) if (n_t + f_t) > 0 else None
            m["Recognition_index_pct"] = round(100 * n_t / (n_t + f_t), 2) if (n_t + f_t) > 0 else None
        elif len(objs) >= 2:
            a, b = objs[0], objs[1]
            ta, tb = times.get(a, 0.0), times.get(b, 0.0)
            m[f"Discrimination_index_{safe_col(b)}_vs_{safe_col(a)}"] = round((tb - ta) / (ta + tb), 4) if (ta + tb) > 0 else None
            m[f"Discrimination_ratio_{safe_col(b)}_vs_{safe_col(a)}"] = round(tb / (ta + tb), 4) if (ta + tb) > 0 else None
    elif test == "social":
        soc = [z for z in zones if _has(z, "social", "stranger", "mouse", "novel")]
        obj = [z for z in zones if _has(z, "object", "empty", "cup")]
        sides = [z for z in zones if not is_hub(z)]
        if soc and obj:
            s_t, o_t = tsum(soc), tsum(obj)
            m.update({"Social_side_time_s": round(s_t, 3), "Object_side_time_s": round(o_t, 3),
                      "Sociability_index": round((s_t - o_t) / (s_t + o_t), 4) if (s_t + o_t) > 0 else None})
        elif len(sides) >= 2:
            a, b = sides[0], sides[-1]
            ta, tb = by[a].Time_s, by[b].Time_s
            m[f"Preference_index_{safe_col(a)}_vs_{safe_col(b)}"] = round((ta - tb) / (ta + tb), 4) if (ta + tb) > 0 else None
        m["Center_time_s"] = round(tsum([z for z in zones if is_hub(z)]), 3)
        m["Chamber_entries"] = int(tsum(sides, "Entries"))

    entries_df = pd.DataFrame(entries_rows)
    return m, zt, entries_df


# labels for the Results page (pretty names); anything missing is shown as-is
PRETTY = {
    "Open_arm_time_pct": "Open-arm time (%)", "Open_arm_time_ratio_pct": "Open / (open+closed) time (%)",
    "Open_arm_entries_pct": "Open-arm entries (%)", "Open_arm_entries": "Open-arm entries",
    "Closed_arm_entries": "Closed-arm entries", "Total_arm_entries": "Total arm entries",
    "Spontaneous_alternation_pct": "Spontaneous alternation (%)", "Alternations": "Alternations",
    "Same_arm_returns": "Same-arm returns", "Alternate_arm_returns": "Alternate-arm returns",
    "Goal_arm_alternation_pct": "Goal-arm alternation (%)", "Light_time_pct": "Time in light (%)",
    "Transitions": "Light-dark transitions", "Latency_to_dark_s": "Latency to dark (s)",
    "Center_time_pct": "Time in centre (%)", "Center_entries": "Centre entries",
    "Discrimination_index": "Discrimination index", "Sociability_index": "Sociability index",
    "Total_exploration_s": "Total object exploration (s)",
    "Anxiety_index": "Anxiety index", "Discrimination_ratio": "Discrimination ratio",
    "Periphery_entries": "Periphery entries", "Possible_alternations": "Possible alternations",
}


def pretty(key):
    if key in PRETTY:
        return PRETTY[key]
    k = str(key)
    for suf, rep in (("_pct", " (%)"), ("_percent", " (%)"), ("_s", " (s)"), ("_px", " (px)")):
        if k.endswith(suf):
            k = k[: -len(suf)] + rep
            break
    return re.sub(r"_+", " ", k).strip()
