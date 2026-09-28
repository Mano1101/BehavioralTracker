"""Custom report builder (Upgrade Plan Tier 2 #10, SMART's Report
Definitions Manager) -- a simple "pick which computed stats go in this
sheet" narrowing of whichever "full" stats table an analysis type already
built (Standard Tracking's summary dict as a one-row DataFrame, Multi-Mouse's
per-mouse table, Behavior Classification's behavior_summary_table), used by
qt_app/dialogs/report_builder_dialog.py and ResultsPage.on_custom_report.

Both functions here are pure column-narrowing -- no stat computation of any
kind happens in this module.
"""

import pandas as pd


def reportable_columns(full_df, id_columns=None):
    """The columns of full_df that can be picked for a Custom Report --
    everything except the id column(s) that identify each row (e.g.
    'mouse_id' for Multi-Mouse, 'subject' for Behavior Classification, none
    at all for Standard Tracking's single-row summary). Column order is
    preserved from full_df. Returns [] for an empty or missing table."""
    if full_df is None or len(full_df.columns) == 0:
        return []
    id_columns = set(id_columns or [])
    return [c for c in full_df.columns if c not in id_columns]


def build_custom_report(full_df, selected_columns, id_columns=None):
    """Narrow full_df down to just the id column(s) -- always kept, even
    though they're never offered as a pickable column -- plus whichever of
    selected_columns was actually checked, in full_df's ORIGINAL column
    order rather than selection order. A stale/unknown selected column name
    (e.g. left over in a re-opened project whose analysis type changed) is
    silently dropped rather than raising. Returns an empty DataFrame if
    there's nothing left to keep, or if full_df itself is empty/missing."""
    if full_df is None or len(full_df.columns) == 0:
        return pd.DataFrame()

    id_columns = list(id_columns or [])
    selected = set(selected_columns or [])
    keep = [c for c in full_df.columns if c in id_columns or c in selected]
    if not keep:
        return pd.DataFrame()
    return full_df[keep].copy()
