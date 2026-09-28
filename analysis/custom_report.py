"""Custom report builder -- Upgrade Plan Tier 2 #10 ("SMART's Report
Definitions Manager"): "a simple 'pick which computed stats go in this
sheet' UI on the Results screen, instead of the fixed Summary sheet
shape."

This module is deliberately generic and analysis-type-agnostic: it never
computes a single stat itself, and never knows what "mouse_id" or
"subject" means. Each caller builds its own "full" stats table in
whichever shape that analysis type already uses today --

  - Standard Tracking:      pd.DataFrame([summary])            (one row)
  - Multi-Mouse Tracking:   per_mouse_summary, one row per mouse_id
  - Behavior Classification: tracking.behavior.behavior_summary_table(),
                             one row per subject

-- and this module just narrows that table's COLUMNS down to whichever
subset the user picked in qt_app/dialogs/report_builder_dialog.py, always
keeping the identifying column(s) (id_columns) regardless of the
selection. Keeping the narrowing logic here in one place, rather than
duplicated per analysis type, is what lets the same dialog and the same
"Custom Report" button on ResultsPage serve all three.
"""

import pandas as pd


def reportable_columns(full_df, id_columns=()):
    """The stat columns available to check/uncheck in the picker --
    every column of full_df except the identifying one(s) (those are
    always kept, never offered as optional), in their existing order.
    Returns [] for an empty/columnless/None full_df."""
    if full_df is None or len(full_df.columns) == 0:
        return []
    return [c for c in full_df.columns if c not in id_columns]


def build_custom_report(full_df, selected_columns, id_columns=()):
    """Narrows full_df to id_columns + whichever of selected_columns
    actually exist in it, preserving full_df's own column order (so the
    exported sheet's column order reflects the data, not click order).
    A selected name that isn't an actual column (e.g. a remembered
    selection from a run whose data didn't have it) is silently dropped
    rather than raising. Returns an empty, columnless DataFrame if
    full_df is empty/None or nothing survives the narrowing (including
    when id_columns themselves aren't present)."""
    if full_df is None or len(full_df.columns) == 0:
        return pd.DataFrame()
    selected = set(selected_columns or [])
    keep = [c for c in full_df.columns if c in id_columns or c in selected]
    if not keep:
        return pd.DataFrame()
    return full_df[keep].copy()
