"""Raw per-frame tracking data, as CSV."""

import os


def write_csv_report(df, output_dir, filename="raw_tracking.csv"):
    """Write the per-frame raw tracking DataFrame to a CSV file."""
    df.to_csv(os.path.join(output_dir, filename), index=False)
