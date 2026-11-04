"""Exports for other programs.

tidy_csv():      one long table per experiment: one row per animal x measure (x epoch / time bin), with Video,
                 Subject, Group(s), Measure, Unit, Value -- R (ggplot / lme4), Python (pandas / seaborn), SPSS and
                 Excel pivot tables read it directly.
prism_sheet():   the "For Prism" sheet: for every main measure a block whose columns are the groups and whose rows
                 are the animals -- select a block, copy, and paste into a GraphPad Prism *Column* table.
"""
import math
import re

import pandas as pd

_UNIT = re.compile(r"\(([^()]*)\)\s*$")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def unit_of(label):
    m = _UNIT.search(str(label))
    return m.group(1) if m else ""


def tidy_rows(df, ident, measures, extra_dims=None):
    """df = one row per animal; ident = identity columns kept on every row (names, or (column, label) pairs);
    measures = [(column, label)].  extra_dims(column) -> dict of extra columns (e.g. Zone) for that measure."""
    out = []
    ident = [(c, c) if isinstance(c, str) else tuple(c) for c in ident]
    cols = list(df.columns)
    for k in range(len(df)):
        def val(c):
            j = cols.index(c) if c in cols else None
            return "" if j is None else df.iat[k, j]
        r = {c: val(c) for c, _l in ident}
        r.update({c: val(c) for c, _l in measures})
        base = {lab: r[c] for c, lab in ident}
        for col, lab in measures:
            v = _num(r.get(col))
            if v is None:
                continue
            row = dict(base)
            dims = extra_dims(col) if extra_dims else None
            row.update(dims or {})
            row["Measure"] = re.sub(r"\s*\([^()]*\)\s*$", "", str(lab)).strip()
            row["Unit"] = unit_of(lab)
            row["Value"] = v
            out.append(row)
    return out


def tidy_csv(rows, path):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    front = [c for c in df.columns if c not in ("Measure", "Unit", "Value")]
    df = df[front + ["Measure", "Unit", "Value"]]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def prism_sheet(writer, df, group_col, subject_col, measures, sheet="For Prism"):
    """Blocks of group columns (Prism column tables), one block per measure, side by side down the sheet."""
    if group_col not in df.columns or subject_col not in df.columns:
        return False
    groups = [g for g in pd.unique(df[group_col].astype(str)) if g and g.lower() not in ("nan", "none")]
    if not groups:
        return False
    blocks = []
    for col, lab in measures:
        if col not in df.columns:
            continue
        vals = {g: [(_num(v), s) for v, s, gg in zip(df[col], df[subject_col], df[group_col].astype(str))
                    if gg == g and _num(v) is not None] for g in groups}
        if not any(vals.values()):
            continue
        n = max(len(v) for v in vals.values())
        t = pd.DataFrame({g: [vals[g][k][0] if k < len(vals[g]) else None for k in range(n)] for g in groups})
        blocks.append((lab, t))
    if not blocks:
        return False
    row = 0
    from output.excel import style_sheet
    for lab, t in blocks:
        pd.DataFrame([[lab]]).to_excel(writer, sheet_name=sheet, startrow=row, index=False, header=False)
        t.to_excel(writer, sheet_name=sheet, startrow=row + 1, index=False)
        row += len(t) + 4
    ws = writer.sheets[sheet]
    style_sheet(ws, freeze=None)
    for i in range(1, len(groups) + 2):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = 22
    return True
