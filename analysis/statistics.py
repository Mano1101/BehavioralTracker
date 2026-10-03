"""
Cross-video / cross-group statistical comparisons (EthoVision XT /
ANY-maze "compare groups" parity -- Upgrade Plan follow-up).

Consumes the per-video rows that batch mode writes to BatchSummary*.csv/.xlsx
(or any list of flat row dicts), groups them by a categorical column
(typically "Group" imported through the Subject Database), and for every
numeric metric column computes:
  - descriptive statistics per group (n, mean, SD, SEM, median)
  - Shapiro-Wilk normality per group (with sample-size guards)
  - 2 groups: Welch's t-test, or Mann-Whitney U when any group is non-normal
  - 3+ groups: one-way ANOVA, or Kruskal-Wallis when any group is non-normal
  - effect size: Cohen's d (2 groups) / eta-squared (3+ groups)
Exports a GraphPad-friendly long table (one row per metric) to
statistics.csv/.xlsx next to the input table and returns the DataFrame.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

# Substrings that mark a column as an identifier/metadata rather than a
# measurable metric -- never fed to the tests, never box-plotted.
_NON_METRIC_HINTS = ("video", "subject", "file", "path", "folder", "group", "status", "date")


def load_summary_table(path):
    """Load a BatchSummary .csv or .xlsx into a DataFrame."""
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def numeric_metric_columns(df, group_col="Group"):
    """Numeric columns that are actual metrics, not identifiers/paths/group."""
    out = []
    for c in df.columns:
        if c == group_col:
            continue
        if any(h in c.lower() for h in _NON_METRIC_HINTS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() >= 2:
            out.append(c)
    return out


def _is_normal(x, alpha=0.05):
    """Shapiro-Wilk normality test with sample-size guards.

    Groups with <3 samples (normality is undefined), >5000 (Shapiro's
    documented ceiling) or zero variance default to "normal" so the
    parametric test is still attempted rather than crashing the batch."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or len(x) > 5000 or np.std(x) == 0:
        return True
    try:
        return stats.shapiro(x).pvalue >= alpha
    except Exception:
        return True


def _cohen_d(a, b):
    """Pooled-SD Cohen's d for two independent groups."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def _eta_squared(group_arrays):
    """Eta-squared for k independent groups."""
    vals = np.concatenate(group_arrays)
    grand = vals.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in group_arrays)
    ss_t = ((vals - grand) ** 2).sum()
    return ss_b / ss_t if ss_t > 0 else np.nan


def compare_groups(df, group_col="Group", metric_cols=None, alpha=0.05):
    """Compare every metric column across groups.

    df may be a DataFrame or a list of flat row dicts (batch summaries).
    Returns a long results DataFrame, one row per metric."""
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found. Have: {list(df.columns)}")
    if metric_cols is None:
        metric_cols = numeric_metric_columns(df, group_col)

    sub = df[df[group_col].notna() & (df[group_col].astype(str).str.strip() != "")]
    group_names = sorted(sub[group_col].unique())
    rows = []

    for metric in metric_cols:
        samples = []
        for g in group_names:
            vals = pd.to_numeric(sub.loc[sub[group_col] == g, metric], errors="coerce")
            vals = vals.dropna().to_numpy(dtype=float)
            samples.append(vals)

        row = {"Metric": metric, "Groups": ", ".join(map(str, group_names)),
               "Test": "insufficient data", "Statistic": np.nan, "p_value": np.nan,
               "Significant": "", "Effect_size": np.nan}
        for g, s in zip(group_names, samples):
            row[f"{g}_n"] = len(s)
            row[f"{g}_mean"] = float(s.mean()) if len(s) else np.nan
            row[f"{g}_SD"] = float(s.std(ddof=1)) if len(s) > 1 else np.nan
            row[f"{g}_SEM"] = float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan
            row[f"{g}_median"] = float(np.median(s)) if len(s) else np.nan

        usable = [s for s in samples if len(s) >= 2]
        if len(usable) < 2:
            rows.append(row)
            continue

        normal = all(_is_normal(s, alpha) for s in usable)
        try:
            if len(usable) == 2:
                if normal:
                    stat, p = stats.ttest_ind(usable[0], usable[1], equal_var=False)
                    row["Test"] = "Welch t-test"
                else:
                    stat, p = stats.mannwhitneyu(usable[0], usable[1], alternative="two-sided")
                    row["Test"] = "Mann-Whitney U"
                row["Effect_size"] = _cohen_d(usable[0], usable[1])
            else:
                if normal:
                    stat, p = stats.f_oneway(*usable)
                    row["Test"] = "One-way ANOVA"
                else:
                    stat, p = stats.kruskal(*usable)
                    row["Test"] = "Kruskal-Wallis"
                row["Effect_size"] = _eta_squared(usable)
            row["Statistic"] = float(stat)
            row["p_value"] = float(p)
            row["Significant"] = "yes" if p < alpha else "no"
        except Exception as exc:
            row["Test"] = f"error: {exc}"
        rows.append(row)

    return pd.DataFrame(rows)


def export_statistics(results_df, input_path):
    """Write statistics.csv/.xlsx next to the input summary table."""
    folder = os.path.dirname(os.path.abspath(str(input_path))) if input_path else "."
    csv_path = os.path.join(folder, "statistics.csv")
    xlsx_path = os.path.join(folder, "statistics.xlsx")
    results_df.to_csv(csv_path, index=False)
    results_df.to_excel(xlsx_path, index=False, engine="openpyxl")
    return csv_path, xlsx_path
