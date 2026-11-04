"""The per-video Excel workbook (<video>_Analysis.xlsx), organised by kind of value:

  Video & tracking info        video, time window, how well the animal was tracked
  Time & percentage            every zone's time (s) / % / mean visit + the test's time measures
  Entries & transitions        every zone's entries + the test's entry measures + zone -> zone transitions
  Distance, speed & latency    distance, speed, distance in every zone, latency to every zone
  Indices                      the test's indices, each with its formula
  Settings                     how the video was analysed
  Entry sequence               every arm / compartment entry in order
  Time bins (+ cumulative)     the measures per time bin
  Zone coordinates             the zones you drew
  Other values                 anything else
  Raw tracking                 the position on every frame
  Column guide                 what every column means and how it is calculated
  All values                   every value in one row (read back by the program)
"""

import os

import pandas as pd

from output import report_layout as RL


def _clean_summary(summary):
    out = {}
    for k, v in summary.items():
        if isinstance(v, (list, tuple, set, dict)):
            v = ", ".join(map(str, v)) if not isinstance(v, dict) else str(v)
        out[k] = v
    return out


def _value(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def style_sheet(ws, widths=None, wrap=(), header_row=1, freeze="A2"):
    """Bold header, column widths, wrapped long text, frozen header row."""
    try:
        from openpyxl.styles import Alignment, Font, PatternFill
        fill = PatternFill("solid", fgColor="E8EEF8")
        for cell in ws[header_row]:
            if cell.value is not None:
                cell.font = Font(bold=True)
                cell.fill = fill
                cell.alignment = Alignment(vertical="center", wrap_text=True)
        for col, w in (widths or {}).items():
            ws.column_dimensions[col].width = w
        for col in wrap:
            for cell in ws[col][header_row:]:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        if freeze:
            ws.freeze_panes = freeze
    except Exception:
        pass


def _auto_widths(ws, df, start=1, max_w=40):
    try:
        from openpyxl.utils import get_column_letter
        for i, c in enumerate(df.columns, start=start):
            vals = [len(str(c))] + [len(str(v)) for v in df[c].head(200).tolist()]
            ws.column_dimensions[get_column_letter(i)].width = max(9, min(max_w, max(vals) + 2))
    except Exception:
        pass


def _header(ws, row, ncols):
    try:
        from openpyxl.styles import Font, PatternFill, Alignment
        fill = PatternFill("solid", fgColor="E8EEF8")
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.font = Font(bold=True)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
    except Exception:
        pass


def _text(ws, row, txt, bold=False, size=11):
    try:
        from openpyxl.styles import Font
        ws.cell(row=row, column=1, value=txt).font = Font(bold=bold, size=size)
    except Exception:
        pass


class _Sheet:
    """Writes titled blocks (tables) one below the other on one sheet."""

    def __init__(self, writer, name):
        self.w, self.name, self.row = writer, name, 0        # row = next free row (0-based for pandas)

    def title(self, txt, bold=True, size=13):
        self._ensure()
        _text(self.w.sheets[self.name], self.row + 1, txt, bold, size)
        self.row += 1

    def table(self, df, title=None):
        if df is None or df.empty:
            return
        if title:
            self.title(title, True, 11)
        df.to_excel(self.w, sheet_name=self.name, index=False, startrow=self.row)
        _header(self.w.sheets[self.name], self.row + 1, len(df.columns))
        self.row += len(df) + 2

    def _ensure(self):
        if self.name not in self.w.sheets:
            pd.DataFrame().to_excel(self.w, sheet_name=self.name, index=False)

    def finish(self, widths=None, wrap=()):
        if self.name not in self.w.sheets:
            return
        ws = self.w.sheets[self.name]
        try:
            from openpyxl.styles import Alignment
            for col, wd in (widths or {}).items():
                ws.column_dimensions[col].width = wd
            for col in wrap:
                for cell in ws[col]:
                    if cell.value is not None and cell.row > 1:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
        except Exception:
            pass


def _vertical(s, keys, test, zones, label=None):
    label = label or RL.label_of
    return pd.DataFrame([{"Measure": label(k), "Value": _value(s.get(k)), "Formula": RL.formula_of(k, test, zones),
                          "What it means": RL.meaning_of(k)} for k in keys])


def _transition_tables(transitions):
    """(count matrix From x To, list of moves) from the From_ROI / To_ROI / Transition_seconds table."""
    if transitions is None or transitions.empty or "From_ROI" not in transitions.columns:
        return None, None
    t = transitions.copy()
    empty = {"", "none", "nan"}
    first = t["From_ROI"].isna() | t["From_ROI"].astype(str).str.strip().str.lower().isin(empty)
    t = t[~first]                    # the first placement into a zone at the start is not a move between zones
    if t.empty:
        return None, None
    t["From_ROI"] = t["From_ROI"].astype(str).str.replace("_", " ")
    t["To_ROI"] = t["To_ROI"].fillna("(outside the zones)").astype(str).str.replace("_", " ")
    t.loc[t["To_ROI"].str.strip().str.lower().isin(empty), "To_ROI"] = "(outside the zones)"
    mat = pd.crosstab(t["From_ROI"], t["To_ROI"])
    mat.index.name = "From \\ To"
    mat["Total"] = mat.sum(axis=1)
    moves = t.rename(columns={"From_ROI": "From", "To_ROI": "To", "Transition_seconds": "Time (s)"})
    moves.insert(0, "Move", range(1, len(moves) + 1))
    return mat.reset_index(), moves


def _bin_columns(df, zones):
    ren = {"Bin": "Time bin", "Start_seconds": "From (s)", "End_seconds": "To (s)",
           "Total_transitions": "Transitions", "Distance_pixels": "Distance (px)", "Distance_cm": "Distance (cm)",
           "Mean_velocity_pixels_s": "Mean speed (px/s)", "Mean_velocity_cm_s": "Mean speed (cm/s)",
           "Tracked_frames": "Frames with a position"}
    for c in df.columns:
        if c not in ren:
            ren[c] = RL.zone_label(c, zones)
    return df.rename(columns=ren)


RAW_GUIDE = {
    "Frame": "Video frame number.", "Time_seconds": "Video time of the frame (s).",
    "Mouse_X": "Body-centre x position (pixels).", "Mouse_Y": "Body-centre y position (pixels).",
    "Mouse_X_smooth": "Smoothed x position (used for distance and zones).",
    "Mouse_Y_smooth": "Smoothed y position.", "Mouse_Present": "A position exists for this frame.",
    "ROI": "Zone the body centre is in (None = outside every zone).", "Detected_area": "Size of the detected animal (px).",
    "Distance_pixels": "Distance moved since the previous frame (px).", "Distance_cm": "The same in cm.",
    "Velocity_pixels_s": "Speed (px/s).", "Velocity_cm_s": "Speed (cm/s).",
    "Tracking_Status": "Tracked = seen; Interpolated = short gap filled; Held = not found, kept at the last place; "
                       "Hidden = inside a covered zone; Lost = not found.",
    "Interpolated": "Short gap filled.", "Held": "Kept at the last seen place.", "Corrected": "Corrected by hand.",
    "Zone_4paw": "Zone by the four-paw entry rule (80 % of the body inside) -- count the changes of this column "
                 "to check the entries by hand.",
}


def _raw_meaning(c):
    if c.startswith("In_"):
        return f"Body centre inside zone '{c[3:].replace('_', ' ')}'."
    if c.startswith("Body_"):
        return f"Fraction of the body inside zone '{c[5:].replace('_', ' ')}' (four-paw rule)."
    if c.startswith("Near_"):
        return f"Animal near object '{c[5:].replace('_', ' ')}'."
    return ""


def _zone_rows(zone_tbl, s, zones, fields, unit):
    """One row per zone with the chosen fields of the zone table (time / entries / distance ...)."""
    if zone_tbl is None or zone_tbl.empty:
        return None
    names = {"Time_s": "Time (s)", "Percent_time": "Time (%)", "Entries": "Entries",
             "Latency_first_entry_s": "Latency to first entry (s)", "Mean_visit_s": "Mean visit (s)",
             "Distance_in_zone": f"Distance in zone ({unit})"}
    cols = ["Zone"] + [f for f in fields if f in zone_tbl.columns]
    t = zone_tbl[cols].copy()
    t["Zone"] = t["Zone"].astype(str).str.replace("_", " ")
    return t.rename(columns=names)


def write_excel_report(output_dir, video_name, df, summary, individual, cumulative,
                       transitions, roi_coords, interactions_df, bin_label="1min", extra_sheets=None):
    excel_path = os.path.join(output_dir, f"{video_name}_Analysis.xlsx")
    s = _clean_summary(summary)
    test = s.get("Test_type") if s.get("Test_type") in RL.TEST_NAMES else "custom"
    zones = RL.zones_of(list(s))
    g = RL.groups(list(s), zones, test=test)
    extra = dict(extra_sheets or {})
    zone_tbl = extra.pop("Zone Summary", None)
    entries = extra.pop("Arm Entries", None)
    mat, moves = _transition_tables(transitions)
    unit = "cm" if "Total_distance_cm" in s else "px"
    not_zone = lambda keys: [k for k in keys if RL.zone_field(k, zones)[0] is None]   # noqa: E731
    W_V = {"A": 36, "B": 18, "C": 48, "D": 70}

    with pd.ExcelWriter(excel_path, engine="openpyxl") as w:
        # 1. video & tracking info
        sh = _Sheet(w, "Video & tracking info")
        sh.title(f"{RL.TEST_NAMES.get(test, test)} — {video_name}")
        sh.title(RL.GROUP_HELP["info"], False, 11)
        sh.row += 1
        sh.table(_vertical(s, [k for k in g["info"] if k != "Video_name"], test, zones))
        sh.finish(W_V, wrap=("C", "D"))
        # 2. time & percentage
        sh = _Sheet(w, "Time & percentage")
        sh.title("Time & percentage")
        sh.title(RL.GROUP_HELP["time"], False, 11)
        sh.row += 1
        sh.table(_zone_rows(zone_tbl, s, zones, ["Time_s", "Percent_time", "Mean_visit_s"], unit), "Every zone")
        sh.table(_vertical(s, not_zone(g["time"]), test, zones), "Test measures")
        sh.finish(W_V, wrap=("C", "D"))
        # 3. entries & transitions
        sh = _Sheet(w, "Entries & transitions")
        sh.title("Entries & transitions")
        sh.title(RL.GROUP_HELP["entries"], False, 11)
        sh.row += 1
        sh.table(_zone_rows(zone_tbl, s, zones, ["Entries"], unit), "Entries into every zone")
        test_entries = [k for k in not_zone(g["entries"]) if not k.startswith("Trans_") and k != "Total_transitions"]
        sh.table(_vertical(s, test_entries, test, zones), "Test measures")
        if mat is not None:
            sh.table(mat, "Transitions: moves from a zone (row) straight into another zone (column)")
            sh.table(moves, "Every move, in order")
        sh.finish(W_V, wrap=("C", "D"))
        # 4. distance, speed & latency
        sh = _Sheet(w, "Distance, speed & latency")
        sh.title("Distance, speed & latency")
        sh.title(RL.GROUP_HELP["distance"], False, 11)
        sh.row += 1
        dist_keys = [k for k in not_zone(g["distance"]) if k not in RL.LATENCY]
        sh.table(_vertical(s, dist_keys, test, zones), "Distance and speed")
        sh.table(_zone_rows(zone_tbl, s, zones, ["Distance_in_zone", "Latency_first_entry_s"], unit),
                 "Every zone: distance inside it and latency to the first entry")
        sh.table(_vertical(s, [k for k in not_zone(g["distance"]) if k in RL.LATENCY], test, zones),
                 "Test latencies")
        sh.finish(W_V, wrap=("C", "D"))
        # 5. indices (formula next to every index)
        if g["indices"]:
            sh = _Sheet(w, "Indices")
            sh.title("Indices")
            sh.title(RL.GROUP_HELP["indices"], False, 11)
            sh.row += 1
            ind = pd.DataFrame([{"Index": RL.label_of(k), "Value": _value(s.get(k)),
                                 "Formula": RL.formula_of(k, test, zones), "What it means": RL.meaning_of(k)}
                                for k in g["indices"]])
            sh.table(ind)
            sh.finish({"A": 30, "B": 14, "C": 70, "D": 70}, wrap=("C", "D"))
        # 6. settings
        sh = _Sheet(w, "Settings")
        sh.title("Settings")
        sh.title(RL.GROUP_HELP["settings"], False, 11)
        sh.row += 1
        sh.table(_vertical(s, g["settings"], test, zones)[["Measure", "Value", "What it means"]]
                 .rename(columns={"Measure": "Setting"}))
        sh.finish({"A": 32, "B": 46, "C": 70}, wrap=("B", "C"))
        # 7. entry sequence
        if entries is not None and not entries.empty:
            en = entries.rename(columns={"Time_s": "Time (s)", "Arm_type": "Arm type"}).copy()
            for c in ("Arm", "Compartment"):
                if c in en.columns:
                    en[c] = en[c].astype(str).str.replace("_", " ")
            en.to_excel(w, sheet_name="Entry sequence", index=False)
            style_sheet(w.sheets["Entry sequence"])
            _auto_widths(w.sheets["Entry sequence"], en)
        # 8. time bins
        if individual is not None and not individual.empty:
            ib = _bin_columns(individual, zones)
            ib.to_excel(w, sheet_name="Time bins", index=False)
            style_sheet(w.sheets["Time bins"])
            _auto_widths(w.sheets["Time bins"], ib, max_w=24)
        if cumulative is not None and not cumulative.empty:
            cb = _bin_columns(cumulative, zones)
            cb.to_excel(w, sheet_name="Time bins (cumulative)", index=False)
            style_sheet(w.sheets["Time bins (cumulative)"])
            _auto_widths(w.sheets["Time bins (cumulative)"], cb, max_w=24)
        # 9. zones drawn, object interactions, other values, other tables
        if roi_coords is not None and not roi_coords.empty:
            rc = roi_coords.rename(columns={"ROI": "Zone", "Vertices_xy": "Corner points (x, y) in pixels"})
            rc.to_excel(w, sheet_name="Zone coordinates", index=False)
            style_sheet(w.sheets["Zone coordinates"], {"A": 22, "B": 100})
        if interactions_df is not None and not interactions_df.empty:
            interactions_df.to_excel(w, sheet_name="Object interactions", index=False)
            style_sheet(w.sheets["Object interactions"])
        if g["other"]:
            _vertical(s, g["other"], test, zones).to_excel(w, sheet_name="Other values", index=False)
            style_sheet(w.sheets["Other values"], W_V, wrap=("C", "D"))
        for name, sheet in extra.items():
            if sheet is not None and not sheet.empty:
                sheet.to_excel(w, sheet_name=str(name)[:31], index=False)
        # 10. raw data
        df.to_excel(w, sheet_name="Raw tracking", index=False)
        style_sheet(w.sheets["Raw tracking"])
        # 11. column guide
        guide = []
        for key, name in RL.GROUPS:
            for k in g[key]:
                if k == "Video_name":
                    continue
                guide.append({"Sheet": name, "Column": RL.zone_label(k, zones), "Formula": RL.formula_of(k, test, zones),
                              "What it means": RL.meaning_of(k) or _zone_meaning(k, zones)})
        guide += [{"Sheet": "Raw tracking", "Column": c, "Formula": "",
                   "What it means": RAW_GUIDE.get(c, _raw_meaning(c))} for c in df.columns]
        guide.append({"Sheet": "Time bins", "Column": "<zone> – time (s) / (%)", "Formula": "",
                      "What it means": "Time in each zone within that time bin (cumulative: from the start)."})
        gd = pd.DataFrame(guide)
        gd.to_excel(w, sheet_name="Column guide", index=False)
        style_sheet(w.sheets["Column guide"], {"A": 24, "B": 34, "C": 60, "D": 80}, wrap=("C", "D"))
        # 12. every value in one row (read back by the program; also handy for copying into other tables)
        pd.DataFrame([s]).to_excel(w, sheet_name="All values", index=False)

    return excel_path


def _zone_meaning(key, zones):
    z, f = RL.zone_field(key, zones)
    if z is None:
        return ""
    return dict((a, c) for a, _b, c in RL.ZONE_FIELDS)[f]
