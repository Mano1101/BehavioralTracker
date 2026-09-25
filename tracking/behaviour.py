"""
Time-budget binning (1-minute individual/cumulative summaries) and the
manual interaction-bout behavior-tagging review step -- the "what behavior
was this" classification layered on top of tracking.interaction's
"when/where" proximity detection.
"""

import cv2
import numpy as np
import pandas as pd

from analysis.calculations import safe_col
from tracking.location import read_and_warp, draw_roi_polygons, draw_object_markers


def calculate_bins(df, fps, start_time, end_time, transitions, roi_names):
    valid = df[df["Tracking_Status"] == "Tracked"]

    edges = list(np.arange(start_time, end_time, 60.0))
    if not edges or edges[-1] < end_time:
        edges.append(end_time)

    def build_rows(cumulative):
        rows = []
        span = range(1, len(edges)) if cumulative else range(len(edges) - 1)

        for i in span:
            a = start_time if cumulative else float(edges[i])
            b = float(edges[i]) if cumulative else float(edges[i + 1])

            s = valid[(valid["Time_seconds"] >= a) & (valid["Time_seconds"] < b)]
            duration = b - a

            label = (
                f"0-{(b - start_time) / 60:.0f} min" if cumulative
                else f"{(a - start_time) / 60:.0f}-{(b - start_time) / 60:.0f} min"
            )

            row = {"Bin": label, "Start_seconds": a, "End_seconds": b}

            for name in roi_names:
                col = f"In_{safe_col(name)}"
                name_time = (s[col].sum() / fps) if col in s else 0.0
                row[f"{safe_col(name)}_time_s"] = name_time
                row[f"{safe_col(name)}_percent"] = (name_time / duration * 100) if duration > 0 else 0

            transition_count = int(
                ((transitions["Transition_seconds"] >= a) & (transitions["Transition_seconds"] < b)).sum()
            )

            row["Total_transitions"] = transition_count
            row["Distance_pixels"] = s["Distance_pixels"].sum()

            if not cumulative:
                row["Mean_velocity_pixels_s"] = s["Velocity_pixels_s"].mean() if len(s) else 0

            row["Tracked_frames"] = len(s)
            rows.append(row)

        return pd.DataFrame(rows)

    return build_rows(cumulative=False), build_rows(cumulative=True)


def tag_interaction_bouts(video_path, matrix, warp_w, warp_h, bouts, behavior_names, roi_points, object_points):
    if not bouts or not behavior_names:
        for b in bouts:
            b["Behavior"] = "Unclassified"
        return bouts

    cap = cv2.VideoCapture(video_path)
    key_map = {str(i + 1): name for i, name in enumerate(behavior_names[:9])}

    print("\n--- Interaction bout review ---")
    print("Each detected approach replays on a loop. Press the matching number key,")
    print("'s' to discard this bout as a false positive, or just let it replay to rewatch.")
    for k, v in key_map.items():
        print(f"  {k} = {v}")

    for idx, bout in enumerate(bouts, start=1):
        label = None

        while label is None:
            for f in range(bout["Frame_start"], bout["Frame_end"] + 1):
                frame = read_and_warp(cap, f, matrix, warp_w, warp_h)
                if frame is None:
                    continue

                disp = frame.copy()
                draw_roi_polygons(disp, roi_points)
                draw_object_markers(disp, object_points)

                cv2.putText(
                    disp, f"Bout {idx}/{len(bouts)}  Object: {bout['Object']}  {bout['Duration_seconds']:.2f}s",
                    (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                )

                y0 = 50
                for k, v in key_map.items():
                    cv2.putText(disp, f"[{k}] {v}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y0 += 20
                cv2.putText(disp, "[s] discard", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Tag interaction", disp)
                key = cv2.waitKey(max(1, int(1000 / 15))) & 0xFF
                key_char = chr(key) if 0 <= key < 256 else ""

                if key_char in key_map:
                    label = key_map[key_char]
                    break
                if key_char == "s":
                    label = "Discarded"
                    break
            # falling through the for-loop with no keypress just replays the bout

        bout["Behavior"] = label

    cv2.destroyWindow("Tag interaction")
    cap.release()

    return [b for b in bouts if b["Behavior"] != "Discarded"]


# -----------------------------
# Plot outputs
# -----------------------------
