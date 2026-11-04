"""BatchSummary workbook (BehavioralTracker 1.0.0) -- one row per video, organised by kind of value:

  Video & tracking info        video, subject / groups, time window, how well the animal was tracked
  Time & percentage            every zone's time (s), then every zone's %, then the test's time measures
  Entries & transitions        every zone's entries, the test's entry measures, zone -> zone transitions
  Distance, speed & latency    distance, speed, latency to every zone, the test's latencies
  Indices                      every index with its formula in the column next to it
  Settings                     how each video was analysed
  Other values                 anything else
  Column guide                 what every column means and how it is calculated

Every sheet starts with Video, Subject and the group columns.  Zone columns whose names differ only in upper /
lower case (Whole_Arena / Whole_arena in two videos) are merged.  read_batch_summary() gives the Statistics page
one table again (older summaries are read as before).
"""
import re

import pandas as pd

from output import report_layout as RL
from output.excel import style_sheet, _auto_widths

DATA_GROUPS = ("time", "entries", "distance", "indices")
# older layouts
INFO_SHEET = "Batch information"
DATA_SHEET = "Summary data"


def _merge_case_duplicates(df):
    """Columns that differ only in case and never both hold a value in one row -> one column."""
    seen = {}
    for c in list(df.columns):
        key = str(c).lower()
        if key not in seen:
            seen[key] = c
            continue
        keep = seen[key]
        both = df[keep].notna() & df[c].notna()
        if both.any():
            continue                      # two different zones after all: keep both
        df[keep] = df[keep].where(df[keep].notna(), df[c])
        df = df.drop(columns=[c])
    return df


def _filled(series):
    return series.astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any()


def write_batch_summary(rows, path, design_cols=()):
    """rows = one flat summary dict per video -> the organised BatchSummary workbook.  Returns the path."""
    df = pd.DataFrame(rows)
    df = _merge_case_duplicates(df)
    keys = list(df.columns)
    tests = [t for t in df.get("Test_type", pd.Series(dtype=object)).dropna().unique().tolist() if t]
    test = tests[0] if len(tests) == 1 else None
    zones = RL.zones_of(keys)
    design = [d for d in design_cols if d in keys and _filled(df[d])]
    g = RL.groups(keys, zones, design, test=test)
    for c in g["entries"]:
        if RL.is_transition(c):
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    ident = ["Video_name"] + [c for c in ["Chamber", "Subject"] + design if c in keys and _filled(df[c])]
    if len(tests) > 1:
        ident.append("Test_type")

    def label(c):
        if c == "Video_name":
            return "Video"
        if c in design:
            return c
        z, _f = RL.zone_field(c, zones)
        return RL.zone_label(c, zones) if z is not None else RL.label_of(c)

    guide = []
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for key, name in RL.GROUPS:
            cols = [c for c in g[key] if c not in ident]
            if key == "info":
                cols = [c for c in g["info"] if c not in ident]
            if not cols:
                continue
            t = pd.DataFrame(index=df.index)
            for c in ident + cols:
                t[label(c)] = df[c]
                if key == "indices" and c not in ident:
                    # the formula right next to the index (per video: the anxiety index depends on the test)
                    tt = df["Test_type"] if "Test_type" in df.columns else pd.Series([test] * len(df), index=df.index)
                    t[f"{label(c)} – formula"] = [RL.formula_of(c, x, zones) for x in tt]
            t.to_excel(w, sheet_name=name, index=False)
            ws = w.sheets[name]
            style_sheet(ws, freeze="B2")
            _auto_widths(ws, t, max_w=34 if key != "indices" else 60)
            for c in cols:
                guide.append({"Sheet": name, "Column": label(c), "Formula": RL.formula_of(c, test, zones),
                              "What it means": RL.meaning_of(c) or _zone_meaning(c, zones)})
        intro = [{"Sheet": name, "Column": "(the whole sheet)", "Formula": "", "What it means": RL.GROUP_HELP[key]}
                 for key, name in RL.GROUPS if name in w.sheets]
        # GraphPad Prism: groups as columns, one block per main measure
        from output.tidy import prism_sheet, tidy_rows, tidy_csv
        main = [c for k in DATA_GROUPS for c in g[k] if c not in ident and RL.zone_field(c, zones)[0] is None
                and not RL.is_transition(c) and pd.to_numeric(df[c], errors="coerce").notna().any()]
        gcol = next((d for d in design if d.lower() in ("group", "treatment", "condition")), design[0] if design else None)
        if gcol and "Subject" in df.columns and prism_sheet(w, df, gcol, "Subject", [(c, label(c)) for c in main]):
            intro.append({"Sheet": "For Prism", "Column": "(every block)", "Formula": "",
                          "What it means": f"One block per measure: the columns are the {gcol} values, the rows the "
                                           "animals. Select a block, copy and paste it into a GraphPad Prism Column "
                                           "table."})
        gd = pd.DataFrame(intro + guide)
        gd.to_excel(w, sheet_name="Column guide", index=False)
        style_sheet(w.sheets["Column guide"], {"A": 26, "B": 40, "C": 60, "D": 90}, wrap=("C", "D"))
    try:                                   # the same data as one long table (R, Python, SPSS, pivot tables)
        num = [c for k in DATA_GROUPS for c in g[k] if c not in ident
               and pd.to_numeric(df[c], errors="coerce").notna().any()]
        idn = [(c, "Video" if c == "Video_name" else c) for c in ident]

        def dims(c):
            z, f = RL.zone_field(c, zones)
            return {"Zone": z or ""}
        tidy_csv(tidy_rows(df, idn, [(c, label(c)) for c in num], dims), str(path).replace(".xlsx", "_tidy.csv"))
    except Exception as exc:
        print(f"Tidy CSV skipped: {exc}")
    return path


def _zone_meaning(key, zones):
    z, f = RL.zone_field(key, zones)
    if z is None:
        return ""
    return dict((a, c) for a, _b, c in RL.ZONE_FIELDS)[f]


def read_batch_summary(path):
    """One table for statistics (one row per video): the data sheets of the organised workbook merged on the
    video (formula columns are text and are skipped by the statistics), the previous layouts, or any other
    table as it is."""
    p = str(path)
    if not p.lower().endswith((".xlsx", ".xls")):
        return pd.read_csv(p)
    sheets = pd.read_excel(p, sheet_name=None)
    names = dict(RL.GROUPS)
    data_sheets = [names[k] for k in DATA_GROUPS if names[k] in sheets]
    if data_sheets:
        info = sheets.get(names["info"])
        data = None
        for n in ([names["info"]] if info is not None else []) + data_sheets:
            t = sheets[n]
            if "Video" not in t.columns:
                continue
            key = ["Video"] + (["Chamber"] if "Chamber" in t.columns else [])   # chamber videos: a row per animal
            if n == names["info"]:
                keep = [c for c in t.columns if c in key or not pd.api.types.is_numeric_dtype(t[c])
                        and c not in ("Video file", "Results folder", "BT analysis file (.bta)", "Stopped because",
                                      "Time-window note", "Animal colour warning")]
                t = t[keep]
            if data is not None:
                key = [k for k in key if k in data.columns]
            data = t if data is None else data.merge(t[key + [c for c in t.columns if c not in data.columns]]
                                                     .drop_duplicates(key), on=key, how="left")
        if data is not None:
            return data
    if "Main results" in sheets and "Video" in sheets["Main results"].columns:      # layout of 22 Oct (first)
        data = sheets["Main results"]
        for name in ("Zone details", "Transitions"):
            t = sheets.get(name)
            if t is not None and "Video" in t.columns:
                extra = [c for c in t.columns if c not in data.columns]
                if extra:
                    data = data.merge(t[["Video"] + extra].drop_duplicates("Video"), on="Video", how="left")
        return data
    data = sheets.get(DATA_SHEET)
    if data is None:
        return next(iter(sheets.values()))
    data = data.rename(columns=lambda c: "Video_name" if re.sub(r"[\s_]+", "_", str(c).strip()).lower()
                       == "video_name" else c)
    info = sheets.get(INFO_SHEET)
    if info is not None and "Video_name" in info.columns and "Video_name" in data.columns:
        extra = [c for c in info.columns if c not in data.columns and c in ("Subject", "Group", "Sex", "Treatment",
                                                                             "Cohort", "Dam", "Cage", "Day")]
        if extra:
            data = data.merge(info[["Video_name"] + extra].drop_duplicates("Video_name"), on="Video_name",
                              how="left")
    return data
