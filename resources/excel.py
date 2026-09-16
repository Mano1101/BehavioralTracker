"""The multi-sheet Excel workbook: Raw Tracking, Summary, 1min Individual,
1min Cumulative, Transitions, ROI Coordinates, Interactions."""

import os
import pandas as pd


def write_excel_report(output_dir, video_name, df, summary, individual, cumulative,
                        transitions, roi_coords, interactions_df):
    excel_path = os.path.join(output_dir, f"{video_name}_Analysis.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Raw Tracking", index=False)
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        individual.to_excel(writer, sheet_name="1min Individual", index=False)
        cumulative.to_excel(writer, sheet_name="1min Cumulative", index=False)
        transitions.to_excel(writer, sheet_name="Transitions", index=False)
        roi_coords.to_excel(writer, sheet_name="ROI Coordinates", index=False)
        interactions_df.to_excel(writer, sheet_name="Interactions", index=False)

    return excel_path
