"""The multi-sheet Excel workbook: Raw Tracking, Summary, <bin> Individual,
<bin> Cumulative, Transitions, ROI Coordinates, Interactions."""

import os
import pandas as pd


def write_excel_report(output_dir, video_name, df, summary, individual, cumulative,
                        transitions, roi_coords, interactions_df, bin_label="1min"):
    """bin_label names the two time-bin sheets (default "1min", matching
    this report's behavior before time-bin size became configurable --
    Upgrade Plan Tier 2 #9; see tracking.behaviour.calculate_bins/
    format_bin_label). Excel sheet names are capped at 31 characters, so a
    bin_label longer than "Cumulative"'s own share of that is truncated
    defensively rather than raising."""
    excel_path = os.path.join(output_dir, f"{video_name}_Analysis.xlsx")

    def sheet(suffix):
        return f"{bin_label} {suffix}"[:31]

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Tracking", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        individual.to_excel(writer, sheet_name=sheet("Individual"), index=False)
        cumulative.to_excel(writer, sheet_name=sheet("Cumulative"), index=False)
        transitions.to_excel(writer, sheet_name="Transitions", index=False)
        roi_coords.to_excel(writer, sheet_name="ROI Coordinates", index=False)
        interactions_df.to_excel(writer, sheet_name="Interactions", index=False)

    return excel_path
