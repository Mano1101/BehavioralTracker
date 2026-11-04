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
    """Load a BatchSummary .csv or .xlsx into a DataFrame (two-sheet workbooks: 'Summary data' + groups)."""
    from output.batch_summary import read_batch_summary
    return read_batch_summary(path)


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


def _subject_col(df):
    """Return the best available experimental-unit column."""
    for c in ("Subject", "subject", "Subject_ID", "subject_id", "Animal", "animal", "Animal_ID", "animal_id"):
        if c in df.columns:
            return c
    return None


def _bh_adjust(pvals):
    """Benjamini-Hochberg FDR adjustment, preserving NaNs."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan)
    valid = np.isfinite(p)
    if not valid.any():
        return out
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    q = np.empty(m, dtype=float)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        val = p[order[rank]] * m / (rank + 1)
        running = min(running, val)
        q[rank] = running
    out[order] = np.minimum(q, 1.0)
    return out


def compare_groups(df, group_col="Group", metric_cols=None, alpha=0.05,
                   subject_col=None, aggregate_repeated=True, correction="fdr_bh"):
    """Design-aware between-group comparison.

    The experimental unit is the subject/animal when a subject identifier is
    available. If several rows belong to the same subject, they are averaged
    within subject before the between-group test. This prevents pseudoreplication
    when a batch contains repeated videos/trials from the same animal.

    If no subject identifier exists, the function falls back to row/video-level
    analysis and explicitly reports that limitation. No group assignment is
    invented. P values are optionally adjusted across metrics using Benjamini-
    Hochberg FDR.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found. Have: {list(df.columns)}")
    if metric_cols is None:
        metric_cols = numeric_metric_columns(df, group_col)

    subject_col = subject_col or _subject_col(df)
    sub = df[df[group_col].notna() & (df[group_col].astype(str).str.strip() != "")].copy()
    if sub.empty:
        return pd.DataFrame()

    analysis_unit = "subject" if subject_col and subject_col in sub.columns else "row/video"
    if analysis_unit == "subject" and aggregate_repeated:
        # Keep group attached to each subject. Conflicting group labels are not
        # silently resolved; they are flagged and excluded from inference.
        sub["__subject"] = sub[subject_col].astype(str)
        conflicts = sub.groupby("__subject")[group_col].nunique(dropna=True)
        bad_subjects = set(conflicts[conflicts > 1].index)
        if bad_subjects:
            sub = sub[~sub["__subject"].isin(bad_subjects)].copy()
        agg_cols = {}
        for c in metric_cols:
            agg_cols[c] = "mean"
        if agg_cols:
            grouped = sub.groupby(["__subject", group_col], as_index=False)[metric_cols].mean()
        else:
            grouped = sub[["__subject", group_col]].drop_duplicates()
        grouped = grouped.rename(columns={"__subject": subject_col})
    else:
        grouped = sub.copy()
        bad_subjects = set()

    group_names = sorted(grouped[group_col].dropna().unique(), key=lambda x: str(x))
    rows = []
    raw_p = []
    for metric in metric_cols:
        samples = []
        names_used = []
        for g in group_names:
            vals = pd.to_numeric(grouped.loc[grouped[group_col] == g, metric], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) >= 2:
                samples.append(vals)
                names_used.append(g)

        row = {
            "Metric": metric,
            "Groups": ", ".join(map(str, group_names)),
            "Analysis_unit": analysis_unit,
            "Subject_column": subject_col or "",
            "Repeated_rows_aggregated": "yes" if analysis_unit == "subject" and aggregate_repeated else "no",
            "Excluded_conflicting_subjects": len(bad_subjects),
            "Test": "insufficient data",
            "Statistic": np.nan,
            "p_value": np.nan,
            "p_adjusted": np.nan,
            "Significant": "",
            "Effect_size": np.nan,
        }
        for g in group_names:
            vals = pd.to_numeric(grouped.loc[grouped[group_col] == g, metric], errors="coerce").dropna().to_numpy(dtype=float)
            key = str(g)
            row[f"{key}_n"] = len(vals)
            row[f"{key}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{key}_SD"] = float(vals.std(ddof=1)) if len(vals) > 1 else np.nan
            row[f"{key}_SEM"] = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else np.nan
            row[f"{key}_median"] = float(np.median(vals)) if len(vals) else np.nan

        if len(samples) < 2:
            rows.append(row); raw_p.append(np.nan); continue
        normal = all(_is_normal(s, alpha) for s in samples)
        # the assumptions behind the choice of test, shown next to the result
        sw = []
        for g, sm in zip(names_used, samples):
            try:
                sw.append(f"{g}: {stats.shapiro(sm).pvalue:.3f}" if 3 <= len(sm) <= 5000 else f"{g}: n<3")
            except Exception:
                sw.append(f"{g}: —")
        row["Normality_Shapiro_p"] = "; ".join(sw)
        try:
            row["Equal_variance_Levene_p"] = float(stats.levene(*samples).pvalue)
        except Exception:
            row["Equal_variance_Levene_p"] = np.nan
        row["Test_choice"] = ("every group looked normal (Shapiro-Wilk p ≥ %.2f) → parametric" % alpha if normal else
                              "a group did not look normal (Shapiro-Wilk p < %.2f) or n < 3 → rank-based" % alpha)
        try:
            if len(samples) == 2:
                if normal:
                    stat, p = stats.ttest_ind(samples[0], samples[1], equal_var=False)
                    row["Test"] = "Welch t-test"
                else:
                    stat, p = stats.mannwhitneyu(samples[0], samples[1], alternative="two-sided")
                    row["Test"] = "Mann-Whitney U"
                row["Effect_size"] = _cohen_d(samples[0], samples[1])
            else:
                if normal:
                    stat, p = stats.f_oneway(*samples)
                    row["Test"] = "One-way ANOVA"
                else:
                    stat, p = stats.kruskal(*samples)
                    row["Test"] = "Kruskal-Wallis"
                row["Effect_size"] = _eta_squared(samples)
            row["Statistic"] = float(stat)
            row["p_value"] = float(p)
            raw_p.append(float(p))
        except Exception:
            raw_p.append(np.nan)
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        n_tests = int(result["p_value"].notna().sum())
        result["Correction"] = (f"Benjamini–Hochberg FDR over the {n_tests} metrics tested (alpha {alpha:g})"
                                if correction == "fdr_bh" else "none")
        result["Effect_size_type"] = np.where(result["Test"].isin(["Welch t-test", "Mann-Whitney U"]),
                                              "Cohen's d (pooled SD)",
                                              np.where(result["Test"].isin(["One-way ANOVA", "Kruskal-Wallis"]),
                                                       "eta squared", ""))
    if not result.empty and correction == "fdr_bh":
        result["p_adjusted"] = _bh_adjust(result["p_value"].to_numpy())
        result["Significant"] = np.where(
            result["p_adjusted"].notna(),
            np.where(result["p_adjusted"] < alpha, "yes", "no"), "")
    else:
        result["Significant"] = np.where(
            result["p_value"].notna(), np.where(result["p_value"] < alpha, "yes", "no"), "")
    return result


def export_statistics(results_df, input_path):
    """Write statistics.csv/.xlsx next to the input summary table."""
    folder = os.path.dirname(os.path.abspath(str(input_path))) if input_path else "."
    csv_path = os.path.join(folder, "statistics.csv")
    xlsx_path = os.path.join(folder, "statistics.xlsx")
    results_df.to_csv(csv_path, index=False)
    results_df.to_excel(xlsx_path, index=False, engine="openpyxl")
    return csv_path, xlsx_path
