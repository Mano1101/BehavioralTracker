"""Freezing epochs from a file (stimulus / protocol logs).

Understood:
  * a table (.csv / .tsv / .txt / .xlsx) with a start column and an end or duration column, and optionally a
    name column -- e.g. exported from a protocol program or typed in Excel:
        name,start,end          Tone 1,120,150          (header words are matched loosely: onset, offset,
        label,onset,duration    CS+,180,30               begin, stop, time, length, event, trial ...)
  * a list of event times (one column of numbers, or MED-PC data-file arrays such as "C:" / "D:"): every time is
    the start of an epoch; the length and the name are chosen when importing.
  * MED-PC data files ("File:", "Subject:", arrays "A:" ... with rows "     0:     120.000   180.000 ...").

Times in the file are in seconds of the LOG; `offset_s` = the video time of the log's time 0 (the video started
before / after the protocol).  Values in minutes or in MED-PC ticks are converted with `scale` (0.01 for 10 ms
ticks, 60 for minutes).
"""
import csv
import os
import re

START_WORDS = ("start", "onset", "begin", "from", "time", "on")
END_WORDS = ("end", "offset", "stop", "to", "off")
DUR_WORDS = ("duration", "length", "dur", "len")
NAME_WORDS = ("name", "label", "epoch", "event", "trial", "stimulus", "cs", "type", "condition")

_NUM = re.compile(r"^[+-]?(\d+([.,]\d*)?|[.,]\d+)([eE][+-]?\d+)?$")


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    t = str(v).strip().replace(" ", "")
    if not t or not _NUM.match(t):
        return None
    return float(t.replace(",", "."))


def _find(cols, words):
    low = [str(c).strip().lower() for c in cols]
    for w in words:                                  # exact word first, then "contains"
        for i, c in enumerate(low):
            if c == w or c.replace("_", " ").split(" ")[0] == w:
                return i
    for w in words:
        for i, c in enumerate(low):
            if w in c:
                return i
    return None


def _rows_from_file(path):
    ext = os.path.splitext(str(path))[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        import pandas as pd
        df = pd.read_excel(path, header=None)
        return [[None if (isinstance(v, float) and v != v) else v for v in r] for r in df.values.tolist()]
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t ")
        delim = dialect.delimiter
    except csv.Error:
        delim = "\t" if "\t" in sample else ("," if "," in sample else None)
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in (line.split(delim) if delim and delim != " " else line.split())]
        rows.append(parts)
    return rows


def read_medpc(text):
    """{array letter: [values]} from a MED-PC data file (several subjects: the first one)."""
    arrays = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^\s*([A-Z]):\s*(.*)$", line)
        if m and not re.match(r"^\s*\d+:", line):
            letter, rest = m.group(1), m.group(2).strip()
            if letter in arrays and cur is None:
                break
            cur = letter
            arrays[cur] = []
            if rest:                                   # a scalar ("A: 3.000")
                v = _num(rest.split()[0])
                if v is not None:
                    arrays[cur].append(v)
                cur = None if v is not None else cur
            continue
        m = re.match(r"^\s*\d+:\s*(.*)$", line)
        if m and cur is not None:
            for t in m.group(1).split():
                v = _num(t)
                if v is not None:
                    arrays[cur].append(v)
            continue
        if line.strip().lower().startswith(("file:", "start date", "subject", "experiment", "group", "box",
                                            "start time", "end date", "end time", "msn")):
            if arrays and line.strip().lower().startswith("file:"):
                break                                  # the next subject
            cur = None
    return {k: v for k, v in arrays.items() if v}


def read_epoch_file(path):
    """-> dict(kind="table", rows=[{name, start, end}], columns=...) or dict(kind="times", series={label: [t]})."""
    with open(path, "rb") as f:
        head = f.read(4096)
    text = None
    if os.path.splitext(str(path))[1].lower() not in (".xlsx", ".xlsm", ".xls"):
        text = head.decode("utf-8", "replace")
    if text is not None and re.search(r"^\s*(File|Subject|MSN):", text, re.M) \
            and re.search(r"^\s*[A-Z]:\s*$", text, re.M):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            arrays = read_medpc(f.read())
        series = {}
        for k, vals in arrays.items():
            vals = list(vals)
            while len(vals) > 1 and vals[-1] == 0:      # arrays are padded with zeros
                vals.pop()
            if len(vals) >= 2 or (vals and vals[0] > 0):
                series[f"MED-PC array {k}"] = vals
        if not series:
            raise ValueError("No time arrays were found in this MED-PC file.")
        return {"kind": "times", "series": series, "source": "MED-PC"}
    rows = _rows_from_file(path)
    rows = [r for r in rows if any(str(c).strip() for c in r if c is not None)]
    if not rows:
        raise ValueError("The file is empty.")
    header = None
    if not all(_num(c) is not None for c in rows[0] if c is not None and str(c).strip()):
        header, rows = [str(c or "").strip() for c in rows[0]], rows[1:]
    ncol = max(len(r) for r in rows) if rows else 0
    if header is not None:
        i_s = _find(header, START_WORDS)
        i_e = _find(header, END_WORDS)
        i_d = _find(header, DUR_WORDS)
        i_n = _find(header, NAME_WORDS)
        if i_e is not None and i_e == i_s:
            i_e = None
        if i_s is not None and (i_e is not None or i_d is not None):
            out = []
            for k, r in enumerate(rows, start=1):
                a = _num(r[i_s]) if i_s < len(r) else None
                b = _num(r[i_e]) if (i_e is not None and i_e < len(r)) else None
                d = _num(r[i_d]) if (i_d is not None and i_d < len(r)) else None
                if a is None:
                    continue
                if b is None and d is not None:
                    b = a + d
                if b is None or b <= a:
                    continue
                name = str(r[i_n]).strip() if (i_n is not None and i_n < len(r) and r[i_n] is not None) else ""
                if i_n is not None and _num(name) is not None and i_n in (i_s, i_e, i_d):
                    name = ""
                out.append({"name": name or f"Epoch {k}", "start": a, "end": b})
            if out:
                return {"kind": "table", "rows": out, "columns": header}
    # numbers only: every numeric column is a list of event times
    series = {}
    for c in range(ncol):
        vals = [_num(r[c]) for r in rows if c < len(r)]
        vals = [v for v in vals if v is not None]
        if vals:
            label = (header[c] if header and c < len(header) and header[c] else f"Column {c + 1}")
            series[label] = vals
    if not series:
        raise ValueError("No start / end columns and no event times were found in this file.")
    return {"kind": "times", "series": series, "source": "table"}


def epochs_from_times(times, length_s, name="Tone", offset_s=0.0, scale=1.0, numbered=True):
    """Every event time -> an epoch of `length_s` seconds named 'Tone 1', 'Tone 2' …"""
    out = []
    for k, t in enumerate(sorted(float(x) for x in times), start=1):
        a = float(offset_s) + float(t) * float(scale)
        out.append({"name": f"{name} {k}" if numbered else name, "start": round(a, 4),
                    "end": round(a + float(length_s), 4)})
    return out


def shift_rows(rows, offset_s=0.0, scale=1.0):
    return [{"name": r["name"], "start": round(float(offset_s) + float(r["start"]) * scale, 4),
             "end": round(float(offset_s) + float(r["end"]) * scale, 4)} for r in rows]
