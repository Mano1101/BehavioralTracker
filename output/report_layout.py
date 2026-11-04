"""How the results are laid out in the Excel workbooks (BehavioralTracker 1.0.0).

Every value of the analysis belongs to one group (one sheet each):

  Video & tracking info        which video / subject, the time window, how well the animal was tracked
  Time & percentage            time (s) and % time in every zone + the test's time measures
  Entries & transitions        entries into every zone, the test's entry measures (alternation...),
                               zone -> zone transitions
  Distance, speed & latency    distance, speed, distance in zones, latency to every zone / test latencies
  Indices                      anxiety / discrimination / sociability / preference indices, each with its formula
  Settings                     how the video was analysed (methods section)
  Other values                 anything else (e.g. your own custom variables)

LABELS gives every value a plain name with its unit and its meaning, FORMULAS how it is calculated (the
'Column guide' sheet is made from them).
"""
import math
import re

TEST_NAMES = {"open_field": "Open field", "epm": "Elevated plus maze", "light_dark": "Light / dark box",
              "y_maze": "Y-maze", "t_maze": "T-maze", "nor": "Novel object recognition",
              "social": "Social interaction (three chambers)", "custom": "Custom zones"}

# key -> (label with unit, what it means / what it is used for)
LABELS = {
    # identity
    "Video_name": ("Video", "The video file."),
    "Subject": ("Subject", "Animal ID given on the Experiment page / in the batch list."),
    # locomotion
    "Total_distance_cm": ("Total distance (cm)", "How far the animal walked in the analysed time -- general "
                          "activity. A drug or diet that makes the animal slow also changes the other measures."),
    "Total_distance_pixels": ("Total distance (px)", "The same distance in video pixels (before the cm scale)."),
    "Mean_speed_cm_s": ("Mean speed (cm/s)", "Average walking speed -- activity."),
    "Anxiety_index": ("Anxiety index (0-1)", "1 - (time ratio + entry ratio) / 2 for the exposed area (open arms "
                      "/ light box / centre). Higher = more anxiety-like. One number combining time and entries."),
    # elevated plus maze
    "Open_arm_time_pct": ("Open-arm time (%)", "% of the analysed time spent on the open arms. Lower = more "
                          "anxiety-like. The main EPM measure."),
    "Open_arm_time_ratio_pct": ("Open-arm time of arm time (%)", "Open-arm time / (open + closed arm time); "
                                "the centre is left out. Some papers use this one."),
    "Open_arm_time_s": ("Open-arm time (s)", "Seconds on the open arms."),
    "Closed_arm_time_s": ("Closed-arm time (s)", "Seconds in the closed arms."),
    "Center_time_s": ("Centre time (s)", "Seconds in the centre zone."),
    "Open_arm_entries": ("Open-arm entries", "How many times the animal entered an open arm (80 % of the body in)."),
    "Closed_arm_entries": ("Closed-arm entries", "Entries into the closed arms -- a measure of activity, not anxiety."),
    "Total_arm_entries": ("Total arm entries", "All arm entries -- activity / exploration."),
    "Open_arm_entries_pct": ("Open-arm entries (%)", "Open-arm entries / all arm entries. Lower = more "
                             "anxiety-like. The second main EPM measure."),
    "Latency_first_open_arm_s": ("Latency to first open-arm entry (s)", "Time from the start until the first "
                                 "open-arm entry."),
    # light / dark box
    "Light_time_pct": ("Time in light (%)", "% of time in the light box. Lower = more anxiety-like. The main "
                       "light/dark box measure."),
    "Light_time_s": ("Time in light (s)", "Seconds in the light box."),
    "Dark_time_s": ("Time in dark (s)", "Seconds in the dark box (also while hidden under its lid)."),
    "Transitions": ("Light-dark transitions", "How many times the animal crossed between the boxes -- "
                    "exploration. For the light/dark box use this (not the zone entries)."),
    "Light_entries": ("Light-box entries", "Entries into the light box."),
    "Dark_entries": ("Dark-box entries", "Entries into the dark box."),
    "Latency_to_dark_s": ("Latency to enter the dark box (s)", "Time until the first entry into the dark box."),
    "Latency_to_light_s": ("Latency to enter the light box (s)", "Time until the first entry into the light box."),
    "Start_compartment": ("Start compartment", "Where the animal was when the analysis started."),
    # open field
    "Center_time_pct": ("Time in centre (%)", "% of time in the centre zone. Lower = more anxiety-like. The main "
                        "open-field anxiety measure."),
    "Center_entries": ("Centre entries", "Entries into the centre zone."),
    "Latency_first_center_s": ("Latency to first centre entry (s)", "Time until the first centre entry."),
    "Periphery_time_s": ("Time in periphery (s)", "Seconds outside the centre zone (along the walls)."),
    "Periphery_entries": ("Periphery entries", "Entries into the periphery."),
    "Center_distance": ("Distance in centre", "Distance walked inside the centre zone."),
    "Center_distance_pct": ("Distance in centre (%)", "Centre distance / total distance."),
    "Thigmotaxis_border_percent": ("Time near the walls (%)", "Thigmotaxis (wall hugging): % of time within the "
                                   "border band along the walls. Open field only."),
    "Thigmotaxis_border_time_s": ("Time near the walls (s)", "Seconds in the border band (open field)."),
    "Thigmotaxis_border_entries": ("Entries to the wall band", "Entries into the border band (open field)."),
    "Thigmotaxis_border_distance_px": ("Distance near the walls (px)", "Distance walked in the border band."),
    "Thigmotaxis_center_time_s": ("Time away from the walls (s)", "Seconds outside the border band."),
    "Thigmotaxis_center_percent": ("Time away from the walls (%)", "% of time outside the border band."),
    # Y / T maze
    "Spontaneous_alternation_pct": ("Spontaneous alternation (%)", "Alternations / possible alternations x 100 "
                                    "(three different arms in a row). Working memory; ~50-70 % in normal mice."),
    "Alternations": ("Alternations", "Triplets of entries into three different arms."),
    "Possible_alternations": ("Possible alternations", "Total arm entries - 2."),
    "Same_arm_returns": ("Same-arm returns", "Re-entries into the arm just left (a perseveration measure)."),
    "Alternate_arm_returns": ("Alternate-arm returns", "Returns to the arm visited two entries before."),
    "Goal_arm_alternation_pct": ("Goal-arm alternation (%)", "T-maze: % of goal-arm choices that differ from the "
                                 "previous choice."),
    "Goal_arm_alternations": ("Goal-arm alternations", "T-maze: number of alternating goal-arm choices."),
    "Goal_arm_entries": ("Goal-arm entries", "T-maze: entries into the goal arms."),
    "First_goal_choice": ("First goal-arm choice", "T-maze: the first goal arm entered."),
    "Alternation_percent": ("Alternation (%)", "Alternation percentage."),
    # NOR
    "Total_exploration_s": ("Total object exploration (s)", "Seconds near / at both objects (with the nose point: "
                                                            "seconds the NOSE was in the object zones)."),
    "Exploration_point": ("Exploration counted by", "Nose = the head end of the body (decided over the whole "
                                                    "track from the tail, the body shape and the walking "
                                                    "direction); body centre otherwise."),
    "Nose_found_pct": ("Nose found (%)", "Tracked frames in which the body had a clear long axis, so the nose "
                                         "point could be placed (curled-up frames have none)."),
    "Discrimination_index": ("Discrimination index", "(novel - familiar) / (novel + familiar) exploration time. "
                             "> 0 = the animal prefers the novel object (remembers the familiar one)."),
    "Discrimination_ratio": ("Discrimination ratio", "novel / (novel + familiar) exploration time (0.5 = no "
                             "preference)."),
    "Recognition_index_pct": ("Recognition index (%)", "Discrimination ratio x 100."),
    "Total_interaction_bouts": ("Object interaction bouts", "Number of separate visits to the objects."),
    # social
    "Sociability_index": ("Sociability index", "(social - object side time) / (social + object side time). "
                          "> 0 = prefers the social side."),
    "Social_side_time_s": ("Time on the social side (s)", "Seconds in the chamber with the stranger mouse."),
    "Object_side_time_s": ("Time on the object side (s)", "Seconds in the chamber with the empty cup / object."),
    "Chamber_entries": ("Chamber entries", "Entries into the side chambers -- activity."),
    # transitions
    "Total_transitions": ("Total transitions", "All moves from one zone straight into another."),
    # tracking quality
    "Start_s": ("Analysed from (s)", "Start of the time window."),
    "End_s": ("Analysed to (s)", "End of the time window."),
    "Analysed_time_s": ("Analysed time (s)", "Length of the time window that was analysed."),
    "Tracking_quality_percent": ("Animal seen (%)", "% of frames where the animal was really found. Above 95 % "
                                 "= reliable data."),
    "Frames_processed": ("Frames analysed", "Number of video frames in the time window."),
    "Tracked_frames": ("Frames with a position", "Frames with a position (seen + short gaps filled)."),
    "Detected_frames": ("Frames animal seen", "Frames where the animal was found."),
    "Interpolated_frames": ("Short gaps filled", "Frames of short gaps filled in between two good positions."),
    "Lost_frames": ("Frames not found", "Frames where the animal could not be found."),
    "Held_frames": ("Frames held at last position", "Not-found frames counted at the last seen place, so the zone "
                    "times add up to the whole time window."),
    "Held_time_s": ("Time held at last position (s)", "The same in seconds."),
    "Hidden_frames": ("Frames hidden in a covered zone", "Frames inside a covered zone (lid of the dark box)."),
    "Hidden_time_s": ("Time hidden in a covered zone (s)", "The same in seconds."),
    "Corrected_frames": ("Frames corrected by hand", "Frames you corrected in the Frame Review."),
    "Stop_Reason": ("Stopped because", "Why the tracking stopped (normally: reached the end time)."),
    "Window_note": ("Time-window note", "Shown when the video ended before the end time you set."),
    "Polarity_warning": ("Animal colour warning", "The chosen animal colour did not look right for this video."),
    # settings
    "Test_type": ("Test", "Behaviour test the measures were computed for."),
    "Detection_engine": ("Detection engine", "Tracking engine used."),
    "Animal_polarity": ("Animal colour", "dark = animal darker than the floor; light = lighter; any = either."),
    "Background_info": ("Background", "Measured noise, animal size and corrections of the empty-arena picture."),
    "Low_light_boost": ("Low-light boost", "Brightening of dark zones used for tracking."),
    "Entry_rule": ("Entry rule", "When a zone entry is counted."),
    "Time_in_zone_rule": ("Time-in-zone rule", "Which point of the animal decides the zone (body centre)."),
    "Tracking_point": ("Tracking point", "The point that is tracked (body centre)."),
    "Min_move_px": ("Minimum movement (px)", "Smaller steps are treated as standing still (removes jitter from "
                    "the distance)."),
    "Auto_mask_outside_zones": ("Area outside the zones masked", "Everything outside the zones was ignored."),
    "Covered_zones": ("Covered zones", "Zones the animal cannot be seen in (e.g. the lid of the dark box)."),
    "Bin_size_s": ("Time-bin length (s)", "Length of the time bins on the Time bins sheet."),
    "Trajectory_smoothing_applied": ("Track smoothed", "Whether the track was smoothed before measuring."),
    "Trajectory_smoothing_window_frames": ("Smoothing window (frames)", "Frames used for smoothing."),
    "FPS": ("Frames per second", "Frame rate of the video."),
    "Live_tracking": ("Live tracking", "Recorded and tracked live from a camera."),
    "Custom_variables_added": ("Custom variables", "Your own variables added to the results."),
    "Video": ("Video file", "Full path of the video."),
    "Output_folder": ("Results folder", "Folder with all result files of this video."),
    "Results_folder": ("Results folder", "Folder with all result files of this video."),
    "Project_file": ("BT analysis file (.bta)", "The protected analysis file (open it to see / re-check the "
                     "results)."),
}

GROUPS = [("info", "Video & tracking info"), ("time", "Time & percentage"), ("entries", "Entries & transitions"),
          ("distance", "Distance, speed & latency"), ("indices", "Indices"), ("settings", "Settings"),
          ("other", "Other values")]
GROUP_HELP = {
    "info": "Which video and animal, the time window that was analysed and how well the animal was tracked "
            "(Tracking quality above 95 % = reliable data).",
    "time": "How long the animal stayed in every zone (seconds and % of the analysed time), and the test's time "
            "measures.",
    "entries": "How many times the animal entered every zone (four paws = 80 % of the body inside), the test's "
               "entry measures, and the zone -> zone transitions. Use entries for every test; the light/dark box "
               "also uses transitions (light <-> dark crossings).",
    "distance": "How far and how fast the animal moved, distance inside zones, and how long it took to reach "
                "every zone (latency).",
    "indices": "Single-number scores of the test. The formula is written next to every index.",
    "settings": "How the video was analysed -- for the methods section.",
    "other": "Anything else that was computed.",
}
INFO = ["Video_name", "Subject", "Test_type", "Start_s", "End_s", "Analysed_time_s", "FPS", "Frames_processed",
        "Tracking_quality_percent", "Detected_frames", "Interpolated_frames", "Tracked_frames", "Lost_frames",
        "Held_frames", "Held_time_s", "Hidden_frames", "Hidden_time_s", "Corrected_frames", "Stop_Reason",
        "Window_note", "Polarity_warning", "Video", "Output_folder", "Results_folder", "Project_file"]
TIME = ["Open_arm_time_s", "Open_arm_time_pct", "Open_arm_time_ratio_pct", "Closed_arm_time_s", "Center_time_s",
        "Center_time_pct", "Periphery_time_s", "Light_time_s", "Light_time_pct", "Dark_time_s",
        "Thigmotaxis_border_time_s", "Thigmotaxis_border_percent", "Thigmotaxis_center_time_s",
        "Thigmotaxis_center_percent", "*_exploration_s", "Total_exploration_s", "Social_side_time_s",
        "Object_side_time_s"]
ENTRIES = ["Open_arm_entries", "Closed_arm_entries", "Total_arm_entries", "Open_arm_entries_pct", "Light_entries",
           "Dark_entries", "Transitions", "Center_entries", "Periphery_entries", "Thigmotaxis_border_entries",
           "*_arm_entries", "Spontaneous_alternation_pct", "Alternations", "Possible_alternations",
           "Same_arm_returns", "Alternate_arm_returns", "Goal_arm_entries", "Goal_arm_alternations",
           "Goal_arm_alternation_pct", "First_goal_choice", "Alternation_percent", "Chamber_entries",
           "Total_interaction_bouts"]
DISTANCE = ["Total_distance_cm", "Total_distance_pixels", "Mean_speed_cm_s", "Mean_speed_pixels_s", "Center_distance",
            "Center_distance_pct", "Thigmotaxis_border_distance_px"]
LATENCY = ["Latency_first_open_arm_s", "Latency_to_dark_s", "Latency_to_light_s", "Latency_first_center_s",
           "Start_compartment"]
INDICES = ["Anxiety_index", "Discrimination_index", "Discrimination_ratio", "Recognition_index_pct",
           "Discrimination_index_*", "Discrimination_ratio_*", "Sociability_index", "Preference_index_*"]
SETTINGS = ["Detection_engine", "Animal_polarity", "Background_info", "Low_light_boost", "Entry_rule",
            "Time_in_zone_rule", "Tracking_point", "Min_move_px", "Auto_mask_outside_zones", "Covered_zones",
            "Bin_size_s", "Trajectory_smoothing_applied", "Trajectory_smoothing_window_frames", "Live_tracking",
            "Custom_variables_added"]
ZONE_FIELDS = [("time_s", "time (s)", "Seconds the body centre was in the zone."),
               ("percent", "time (%)", "% of the analysed time in the zone."),
               ("entries", "entries", "How many times the animal entered the zone (80 % of the body in)."),
               ("latency_s", "latency (s)", "Time from the start until the first entry into the zone.")]
ZONE_GROUP = {"time_s": "time", "percent": "time", "entries": "entries", "latency_s": "distance"}

# how every derived value is calculated
FORMULAS = {
    "Open_arm_time_pct": "open-arm time ÷ analysed time × 100",
    "Open_arm_time_ratio_pct": "open-arm time ÷ (open-arm time + closed-arm time) × 100",
    "Open_arm_entries_pct": "open-arm entries ÷ (open-arm entries + closed-arm entries) × 100",
    "Total_arm_entries": "open-arm entries + closed-arm entries (Y / T maze: all arm entries)",
    "Light_time_pct": "light time ÷ (light time + dark time) × 100",
    "Transitions": "light entries + dark entries − 1 (every crossing between the boxes)",
    "Center_time_pct": "centre time ÷ analysed time × 100",
    "Center_distance_pct": "distance in the centre ÷ total distance × 100",
    "Thigmotaxis_border_percent": "time in the wall band ÷ analysed time × 100",
    "Thigmotaxis_center_percent": "time outside the wall band ÷ analysed time × 100",
    "Spontaneous_alternation_pct": "alternations ÷ (total arm entries − 2) × 100",
    "Alternations": "number of 3 entries in a row into 3 different arms (overlapping: ABC, BCA ...)",
    "Possible_alternations": "total arm entries − 2",
    "Same_arm_returns": "entries into the arm just left (A → A)",
    "Alternate_arm_returns": "entries into the arm visited two entries before (A B A)",
    "Goal_arm_alternation_pct": "goal-arm alternations ÷ (goal-arm entries − 1) × 100",
    "Total_exploration_s": "exploration time of all objects added up",
    "Discrimination_index": "(novel − familiar) ÷ (novel + familiar) exploration time",
    "Discrimination_ratio": "novel ÷ (novel + familiar) exploration time",
    "Recognition_index_pct": "novel ÷ (novel + familiar) exploration time × 100",
    "Sociability_index": "(social side time − object side time) ÷ (social side time + object side time)",
    "Mean_speed_cm_s": "total distance ÷ time the animal was seen",
    "Mean_speed_pixels_s": "total distance (px) ÷ time the animal was seen",
    "Total_distance_cm": "sum of the steps of the body centre × cm per pixel (steps smaller than the minimum "
                         "movement are left out)",
    "Total_distance_pixels": "sum of the steps of the body centre (pixels)",
    "Tracking_quality_percent": "frames the animal was seen ÷ (frames analysed − frames hidden in a covered zone) × 100",
    "Analysed_time_s": "frames analysed ÷ FPS",
    "Total_transitions": "all zone → zone moves added up",
    "Latency_first_open_arm_s": "time of the first open-arm entry − start time",
    "Latency_to_dark_s": "time of the first dark-box entry − start time",
    "Latency_to_light_s": "time of the first light-box entry − start time",
    "Latency_first_center_s": "time of the first centre entry − start time",
}
ANXIETY_FORMULA = {
    "epm": "1 − [(open-arm time ÷ total time) + (open-arm entries ÷ total arm entries)] ÷ 2",
    "light_dark": "1 − [(light time ÷ total time) + (light entries ÷ (light + dark entries))] ÷ 2",
    "open_field": "1 − [(centre time ÷ total time) + (centre entries ÷ (centre + periphery entries))] ÷ 2",
}
ZONE_FORMULAS = {"time_s": "frames with the body centre in the zone ÷ FPS",
                 "percent": "zone time ÷ analysed time × 100",
                 "entries": "four-paw entries: ≥ 80 % of the body inside for ≥ 3 frames in a row",
                 "latency_s": "time of the first entry − start time"}


def nice(name):
    return str(name).replace("_", " ").strip()


def _missing(v):
    if v is None:
        return True
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def zones_of(keys):
    """Zone names in order of first appearance (a zone has <name>_time_s and <name>_percent)."""
    keys = list(keys)
    have = set(keys)
    out = []
    for k in keys:
        if k.endswith("_time_s"):
            z = k[:-len("_time_s")]
            if f"{z}_percent" in have and not z.startswith("Thigmotaxis_") and z not in out:
                out.append(z)
    return out


def label_of(key):
    if key in LABELS:
        return LABELS[key][0]
    m = re.match(r"Trans_(.+)_to_(.+)$", key)
    if m:
        a, b = m.group(1), m.group(2)
        a = "(start / outside the zones)" if a == "None" else nice(a)
        b = "(outside the zones)" if b == "None" else nice(b)
        return f"{a} → {b}"
    for suf, unit in (("_arm_entries", " arm entries"), ("_exploration_s", " exploration (s)")):
        if key.endswith(suf):
            return nice(key[:-len(suf)]) + unit
    if key.startswith(("Preference_index_", "Discrimination_index_", "Discrimination_ratio_")):
        return nice(key).replace(" vs ", " vs ")
    for suf, unit in (("_pct", " (%)"), ("_percent", " (%)"), ("_time_s", " time (s)"), ("_s", " (s)"),
                      ("_px", " (px)"), ("_cm", " (cm)")):
        if key.endswith(suf):
            return nice(key[:-len(suf)]) + unit
    return nice(key)


def meaning_of(key):
    if key in LABELS:
        return LABELS[key][1]
    if key.startswith("Trans_"):
        return "How many times the animal moved from the first zone straight into the second."
    if key.endswith("_arm_entries"):
        return "Entries into this arm."
    if key.endswith("_exploration_s"):
        return "Seconds exploring this object."
    if key.startswith("Preference_index_"):
        return "(time in the first - time in the second) / (sum): > 0 = prefers the first."
    for suf, txt in (("_time_s", "Seconds the body centre was in the zone."), ("_percent", "% of time in the zone."),
                     ("_entries", "Entries into the zone."), ("_latency_s", "Time until the first entry.")):
        if key.endswith(suf):
            return txt
    return ""


def _match(pattern, key):
    if "*" not in pattern:
        return pattern == key
    rx = "^" + re.escape(pattern).replace("\\*", ".+") + "$"
    return re.match(rx, key) is not None


def zone_field(key, zones):
    """(zone, field) when key is <zone>_<time_s|percent|entries|latency_s>, else (None, None)."""
    for z in sorted(zones, key=len, reverse=True):
        for suffix, _lab, _m in ZONE_FIELDS:
            if key == f"{z}_{suffix}":
                return z, suffix
    return None, None


def zone_label(key, zones):
    z, f = zone_field(key, zones)
    if z is not None:
        lab = dict((a, b) for a, b, _c in ZONE_FIELDS)[f]
        return f"{nice(z)} – {lab}"
    return label_of(key)


def formula_of(key, test=None, zones=()):
    if key == "Anxiety_index":
        return ANXIETY_FORMULA.get(test, "1 − [(time ratio + entry ratio) of the exposed area] ÷ 2")
    if key in FORMULAS:
        return FORMULAS[key]
    z, f = zone_field(key, zones)
    if z is not None:
        return ZONE_FORMULAS[f]
    if key.startswith("Preference_index_"):
        m = re.match(r"Preference_index_(.+)_vs_(.+)$", key)
        if m:
            return f"({nice(m.group(1))} time − {nice(m.group(2))} time) ÷ (sum of both)"
    if key.startswith("Discrimination_index_"):
        m = re.match(r"Discrimination_index_(.+)_vs_(.+)$", key)
        if m:
            return f"({nice(m.group(1))} − {nice(m.group(2))}) ÷ ({nice(m.group(1))} + {nice(m.group(2))}) exploration"
    if key.startswith("Discrimination_ratio_"):
        m = re.match(r"Discrimination_ratio_(.+)_vs_(.+)$", key)
        if m:
            return f"{nice(m.group(1))} ÷ ({nice(m.group(1))} + {nice(m.group(2))}) exploration"
    if key.startswith("Trans_"):
        return "count of moves from the first zone straight into the second"
    return ""


def is_transition(key):
    return key.startswith("Trans_") and not re.match(r"Trans_None_to_|Trans_.+_to_None$", key)


def groups(keys, zones=None, design=(), test=None):
    """Split summary keys into the sheet groups -> {group: ordered keys} (zone blocks measure by measure:
    every zone's time, then every zone's %, ...)."""
    keys = list(keys)
    zones = zones if zones is not None else zones_of(keys)
    used = set()
    out = {}

    def take(names):
        lst = []
        for pat in names:
            for k in keys:
                if k not in used and _match(pat, k):
                    lst.append(k)
                    used.add(k)
        return lst

    def zone_block(fields):
        lst = []
        for f in fields:
            for z in zones:
                k = f"{z}_{f}"
                if k in keys and k not in used:
                    lst.append(k)
                    used.add(k)
        return lst
    out["info"] = take(INFO[:2]) + take(list(design)) + take(INFO[2:])
    out["time"] = zone_block(["time_s", "percent"]) + take(TIME)
    zone_entries = zone_block(["entries"])
    for k in keys:                     # '<arm>_arm_entries' repeats the arm's zone entries: shown once
        if k.endswith("_arm_entries") and f"{k[:-len('_arm_entries')]}_entries" in zone_entries:
            used.add(k)
    out["entries"] = (zone_entries + take(ENTRIES)
                      + take(sorted(k for k in keys if is_transition(k))) + take(["Total_transitions"]))
    out["distance"] = take(DISTANCE) + zone_block(["latency_s"]) + take(LATENCY)
    out["indices"] = take(INDICES)
    out["settings"] = take(SETTINGS)
    used.update(k for k in keys if k.startswith("Trans_"))          # start placements: not moves
    if test and test != "nor":                                        # object bouts belong to the NOR test only
        out["entries"] = [k for k in out["entries"] if k != "Total_interaction_bouts"]
    out["other"] = [k for k in keys if k not in used]
    return out
