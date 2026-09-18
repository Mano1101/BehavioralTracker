"""BORIS-style event generation from tracking data.

BORIS (Behavioral Observation Research Interactive Software) is a widely
used open-source tool for coding behavioral events; many labs' downstream
analysis expects a simple subject/behavior/start/end event table like the
one this module produces. Events are generated from the frame-level CSVs
this app already writes (raw_tracking.csv, tracks.csv, labeled_frames.csv)
and are always meant to be reviewed, not treated as ground truth --
automated behavioral labels require validation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import csv
import math


@dataclass
class BehaviorEvent:
    subject: str
    behavior: str
    target: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    source: str = "automatic"
    confidence: float = 1.0

    def to_dict(self):
        d = asdict(self)
        d["start_time"] = round(self.start_time, 3)
        d["end_time"] = round(self.end_time, 3)
        d["duration"] = round(self.duration, 3)
        return d


def _flush(events, subject, behavior, start, end, target="", source="automatic", confidence=1.0, min_duration=0.0):
    if start is None or end <= start or end - start < min_duration:
        return
    events.append(BehaviorEvent(subject, behavior, target, start, end, end - start, source, confidence))


def _cols(rows):
    """Match this app's actual column names across all three analysis
    engines (Standard Tracking's raw_tracking.csv, Multi-Mouse's
    tracks.csv, Behavior Classification's labeled_frames.csv), not just
    generic/BORIS-typical names."""
    if not rows:
        return None
    keys = set(rows[0])

    def pick(*names):
        for n in names:
            if n in keys:
                return n
        return None

    return {
        "time": pick("time", "Time", "timestamp", "Timestamp", "Time_seconds", "time_s"),
        "x": pick("x", "X", "centroid_x", "center_x", "Mouse_X"),
        "y": pick("y", "Y", "centroid_y", "center_y", "Mouse_Y"),
        "subject": pick("subject", "Subject", "animal", "Animal", "mouse", "Mouse", "mouse_id"),
        "label": pick("label_smoothed", "label", "Behavior"),
        "near": [k for k in keys if k.lower().startswith("near_") or k.lower().startswith("in_")],
    }


def read_tracking_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def generate_events(rows, movement_threshold=2.0, freeze_threshold=0.8, min_event_duration=0.25):
    """Generate BORIS-like events for each subject from a frame-level
    tracking table (Standard Tracking or Multi-Mouse Tracking output).

    Required columns: a time column and x/y position columns (several
    naming variants recognized -- see _cols). Subject is optional and
    defaults to Mouse_1. Any In_<Zone> or Near_<Object> boolean-like
    column emits zone/interaction events too.

    If a classified-behavior column is present (label_smoothed, from
    Behavior Classification), those labels are used directly instead of
    re-deriving Moving/Freezing from speed -- an actual classification is
    more informative than a generic speed threshold."""
    c = _cols(rows)
    if not c or not c["time"] or not c["x"] or not c["y"]:
        raise ValueError("Tracking CSV needs time, x and y columns for automatic coding.")
    groups = {}
    for r in rows:
        subject = r.get(c["subject"], "Mouse_1") if c["subject"] else "Mouse_1"
        groups.setdefault(subject, []).append(r)
    events = []
    for subject, rs in groups.items():
        rs.sort(key=lambda r: float(r[c["time"]] or 0))

        if c["label"]:
            # A real classification exists -- use it verbatim instead of
            # a speed-threshold guess.
            active_label = None
            start = None
            for r in rs:
                t = float(r[c["time"]] or 0)
                label = (r.get(c["label"]) or "").strip()
                if label != active_label:
                    if active_label:
                        _flush(events, subject, active_label, start, t, min_duration=min_event_duration)
                    active_label, start = label, t
            if active_label and rs:
                _flush(events, subject, active_label, start, float(rs[-1][c["time"]] or 0),
                       min_duration=min_event_duration)
        else:
            moving_start = freeze_start = None
            prev = None
            for r in rs:
                try:
                    t = float(r[c["time"]] or 0)
                    x = float(r[c["x"]] or 0)
                    y = float(r[c["y"]] or 0)
                except ValueError:
                    continue
                speed = 0.0 if prev is None else math.hypot(x - prev[1], y - prev[2]) / max(1e-9, t - prev[0])
                is_moving = speed >= movement_threshold
                if is_moving:
                    if moving_start is None:
                        moving_start = t
                    if freeze_start is not None:
                        _flush(events, subject, "Freezing", freeze_start, t, min_duration=min_event_duration)
                        freeze_start = None
                else:
                    if speed <= freeze_threshold and freeze_start is None:
                        freeze_start = t
                    if moving_start is not None:
                        _flush(events, subject, "Moving", moving_start, t, min_duration=min_event_duration)
                        moving_start = None
                prev = (t, x, y)
            if prev:
                if moving_start is not None:
                    _flush(events, subject, "Moving", moving_start, prev[0], min_duration=min_event_duration)
                if freeze_start is not None:
                    _flush(events, subject, "Freezing", freeze_start, prev[0], min_duration=min_event_duration)

        # Zone/object boolean columns, regardless of whether a
        # classification label was also present.
        for col in c["near"]:
            prefix_len = 5 if col.lower().startswith("near_") else 3  # "near_" vs "in_"
            label = col[prefix_len:].replace("_", " ").strip() or "Zone"
            active = None
            start = None
            for r in rs:
                t = float(r[c["time"]] or 0)
                val = str(r.get(col, "")).strip().lower() in ("1", "true", "yes", "y")
                if val and not active:
                    active, start = True, t
                elif not val and active:
                    _flush(events, subject, "Interaction" if col.lower().startswith("near_") else "In Zone",
                           start, t, label, min_duration=min_event_duration)
                    active = None
                    start = None
            if active and rs:
                _flush(events, subject, "Interaction" if col.lower().startswith("near_") else "In Zone",
                       start, float(rs[-1][c["time"]] or 0), label, min_duration=min_event_duration)

    events.sort(key=lambda e: (e.start_time, e.subject, e.behavior))
    return events


def save_events(events, path):
    fields = ["subject", "behavior", "target", "start_time", "end_time", "duration", "source", "confidence"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for e in events:
            w.writerow(e.to_dict())
