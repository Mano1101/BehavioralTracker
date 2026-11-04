"""Behaviour events from other programs -> Event Scoring events ({behaviour, kind, start, stop}).

  * BORIS  -- tabular events export (Time, Behavior, Status START / STOP / POINT) or the aggregated export
              (Behavior, Start (s), Stop (s))
  * SimBA  -- machine results (one row per frame; a 0 / 1 column per classifier next to its Probability_<name>
              column): every run of 1s becomes a bout
  * any table with a behaviour (or name / label) column and start + stop (or duration) columns
"""
import os

import numpy as np
import pandas as pd


def _read(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        head = f.read(4096)
    sep = "\t" if head.count("\t") > head.count(",") else ("," if head.count(",") >= head.count(";") else ";")
    return pd.read_csv(path, sep=sep)


def _col(df, *names):
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in low:
            return low[n]
    for n in names:
        for k, c in low.items():
            if k.startswith(n):
                return c
    return None


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def read_events(path, fps=30.0, offset_s=0.0):
    """-> (events, source).  Times in video seconds (+ offset_s)."""
    df = _read(path)
    if df.empty:
        raise ValueError("The file is empty.")
    off = float(offset_s or 0.0)
    # --- SimBA machine results
    probs = [c for c in df.columns if str(c).startswith("Probability_")]
    if probs:
        events = []
        frame_col = _col(df, "frame", "frames", "unnamed: 0")
        frames = pd.to_numeric(df[frame_col], errors="coerce").to_numpy() if frame_col else np.arange(len(df))
        for pc in probs:
            name = str(pc)[len("Probability_"):]
            if name not in df.columns:
                continue
            on = pd.to_numeric(df[name], errors="coerce").fillna(0).to_numpy() > 0.5
            for a, b in _runs(on):
                events.append({"behaviour": name, "kind": "state", "start": round(off + frames[a] / fps, 4),
                               "stop": round(off + (frames[b - 1] + 1) / fps, 4)})
        return events, "SimBA"
    # --- BORIS tabular events
    t, beh, st = _col(df, "time"), _col(df, "behavior", "behaviour"), _col(df, "status", "behavior type")
    if t is not None and beh is not None and st is not None:
        events, open_ = [], {}
        for _i, r in df.sort_values(t).iterrows():
            name, s = str(r[beh]), str(r[st]).strip().upper()
            tv = float(r[t]) + off
            if s == "POINT":
                events.append({"behaviour": name, "kind": "point", "start": round(tv, 4), "stop": None})
            elif s == "START":
                open_[name] = tv
            elif s == "STOP" and name in open_:
                events.append({"behaviour": name, "kind": "state", "start": round(open_.pop(name), 4),
                               "stop": round(tv, 4)})
        return events, "BORIS"
    # --- BORIS aggregated / any table with start + stop / duration
    beh = beh or _col(df, "name", "label", "event", "class")
    a = _col(df, "start (s)", "start", "onset", "begin")
    b = _col(df, "stop (s)", "stop", "end", "offset")
    d = _col(df, "duration (s)", "duration", "length")
    if beh is not None and a is not None and (b is not None or d is not None):
        events = []
        for _i, r in df.iterrows():
            s0 = pd.to_numeric(r[a], errors="coerce")
            s1 = pd.to_numeric(r[b], errors="coerce") if b is not None else s0 + pd.to_numeric(r[d], errors="coerce")
            if not np.isfinite(s0):
                continue
            if np.isfinite(s1) and s1 > s0:
                events.append({"behaviour": str(r[beh]), "kind": "state", "start": round(off + float(s0), 4),
                               "stop": round(off + float(s1), 4)})
            else:
                events.append({"behaviour": str(r[beh]), "kind": "point", "start": round(off + float(s0), 4),
                               "stop": None})
        return events, "table"
    raise ValueError("No events found: expected a BORIS export (Time / Behavior / Status), a SimBA machine-results "
                     "file (Probability_<behaviour> columns) or a table with behaviour + start + stop columns.")
