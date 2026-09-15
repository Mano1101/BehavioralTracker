"""Object-interaction proximity: turning per-frame Near_<object> flags
(computed by tracking.location during the main loop) into discrete bouts."""

from analysis.calculations import safe_col


def extract_bouts(df, object_names, fps, min_duration=0.3):
    bouts = []
    min_frames = max(1, int(round(min_duration * fps)))

    for name in object_names:
        col = f"Near_{safe_col(name)}"
        if col not in df:
            continue

        active = df[col].fillna(False).to_numpy()
        frames = df["Frame"].to_numpy()
        n = len(active)
        i = 0

        while i < n:
            if active[i]:
                j = i
                while j < n and active[j]:
                    j += 1
                if (j - i) >= min_frames:
                    bouts.append({
                        "Object": name,
                        "Frame_start": int(frames[i]),
                        "Frame_end": int(frames[j - 1]),
                        "Start_seconds": float(frames[i]) / fps,
                        "End_seconds": float(frames[j - 1]) / fps,
                        "Duration_seconds": (j - i) / fps,
                        "Behavior": None,
                    })
                i = j
            else:
                i += 1

    bouts.sort(key=lambda b: b["Frame_start"])
    return bouts
