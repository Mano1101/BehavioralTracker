"""
Main dashboard window (TrackerApp) and every interactive setup dialog:
tkinter forms/dialogs plus the OpenCV mouse-driven pickers (arena corners,
zone/object drawing, distance calibration). This is the module Step 2
(PySide6 migration) will actually rewrite -- everything it imports from
tracking/, analysis/, and output/ stays untouched by that migration.
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
import sys
import math
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox

from analysis.calculations import point_distance
from tracking.location import (
    compute_perspective_transform, identity_transform, draw_scale_overlay, ROI_COLORS,
    process_single_video, compute_output_dir, compute_zone_interaction_stats,
)
from tracking.two_mouse import (
    track_video, save_preview_frames as save_preview_frames_multi_mouse, choose_color_mode,
)
from tracking.maze_templates import TEMPLATES as MAZE_TEMPLATES
from tracking.behavior import (
    extract_features, classify_behaviors, manual_score_video,
    save_preview_frames as save_preview_frames_behavior,
)
from tracking.ml_dataset import build_clip_dataset, dataset_summary
from tracking.ml_train import TORCH_AVAILABLE, train_model
from tracking.ml_infer import classify_video_ml


def resource_path(relative_path):
    """
    Resolve a path to a bundled resource (e.g. resources/icon.png) that
    works both when running normally (python main_legacy_tkinter.py) and
    when packaged by PyInstaller, which unpacks --add-data files into a
    temporary sys._MEIPASS folder at runtime instead of the real project
    layout. (This GUI itself isn't what BehavioralTracker.spec packages
    any more -- see the README -- but this helper still works standalone.)
    """
    if hasattr(sys, "_MEIPASS"):
        base_path = sys._MEIPASS
    else:
        # this file lives at <project_root>/gui/main_window.py
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")



def ask_number(title, prompt, default):
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askstring(title, prompt, initialvalue=str(default))
    root.destroy()

    if value is None:
        raise SystemExit("Cancelled.")

    try:
        return float(value)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number.")
        raise SystemExit



def ask_text(title, prompt, default=""):
    root = tk.Tk()
    root.withdraw()
    value = simpledialog.askstring(title, prompt, initialvalue=default)
    root.destroy()

    if value is None or value.strip() == "":
        return default

    return value.strip()



def ask_yes_no(title, prompt):
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(title, prompt)
    root.destroy()
    return bool(result)



def select_video(title="Select behavioral video"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title=title,
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
    )
    root.destroy()
    return path



def select_folder(title="Select folder containing videos"):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title)
    root.destroy()
    return path



def select_mode():
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesno(
        "Mode",
        "Run in BATCH mode and process every video in a folder?\n\n"
        "Yes = Batch folder\nNo = Single video"
    )
    root.destroy()
    return "batch" if result else "single"



def list_videos(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS))
    return [os.path.join(folder, f) for f in files]


# -----------------------------
# Step-back wizard
# -----------------------------
# Drives calibration as a sequence of (label, function) steps. Each
# function is interactive and mutates a shared `state` dict. After
# each step you choose: Yes = keep & continue, No = redo this step,
# Cancel = go back to the previous step (repeatable -> rewind as
# many steps as you need). Because steps read `state` fresh every
# time they run, redoing an earlier step and moving forward again
# naturally re-runs everything after it with the corrected values.



def confirm_step(step_label):
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askyesnocancel(
        "Confirm: " + step_label,
        f"'{step_label}' is done.\n\n"
        "Yes = keep it and continue\n"
        "No = redo this step\n"
        "Cancel = go back to the previous step"
    )
    root.destroy()

    if result is True:
        return "accept"
    if result is False:
        return "redo"
    return "back"



def run_wizard(steps, state):
    i = 0
    n = len(steps)

    while i < n:
        label, fn = steps[i]
        print(f"\n--- Step {i + 1}/{n}: {label} ---")
        fn(state)
        choice = confirm_step(label)

        if choice == "accept":
            i += 1
        elif choice == "redo":
            continue
        else:
            i = max(0, i - 1)

    return state


# -----------------------------
# Geometry
# -----------------------------



def select_four_points(frame, window_title="Click the 4 arena corners"):
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_mouse)

    print("\nClick the 4 corners of the arena, in any order.")
    print("r = reset, ENTER/SPACE = confirm once 4 points are placed, ESC = cancel.")

    while True:
        disp = frame.copy()

        for i, p in enumerate(points):
            cv2.circle(disp, p, 6, (0, 0, 255), -1)
            cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if len(points) >= 2:
            for i in range(len(points)):
                cv2.line(disp, points[i], points[(i + 1) % len(points)], (0, 255, 255), 1)

        cv2.putText(
            disp, f"Points: {len(points)}/4  (r=reset, ENTER=confirm, ESC=cancel)",
            (20, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points = []
        elif key in (13, 32) and len(points) == 4:
            break
        elif key == 27:
            cv2.destroyWindow(window_title)
            raise SystemExit("Cancelled during arena corner selection.")

    cv2.destroyWindow(window_title)
    return points



def edit_regions_interactive(frame, roi_names):
    """
    ezTrack-style multi-region editor: every named region is drawn on
    ONE shared view at once, labeled and semi-transparently filled,
    with draggable vertices -- click near an existing point (any
    region) to grab and drag it, click empty space to add a new point
    to whichever region is currently "active".

    Controls:
      1-9    switch the active region (only changes where new clicks
             land; dragging works on any region's points regardless)
      click  add a point to the active region, OR grab+drag an
             existing point if the click lands on one
      z      undo the active region's last point
      ENTER  finish (every region needs >= 3 points)
      ESC    cancel
    """

    regions = {name: [] for name in roi_names}
    active = [roi_names[0]]
    drag = {"region": None, "index": None}
    hit_radius = 10

    window_title = "Draw Regions"
    key_to_name = {str(i + 1): name for i, name in enumerate(roi_names[:9])}

    def find_vertex_near(x, y):
        best = None
        best_d = hit_radius
        for name, pts in regions.items():
            for i, p in enumerate(pts):
                d = point_distance((x, y), p)
                if d <= best_d:
                    best_d = d
                    best = (name, i)
        return best if best else (None, None)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            name, idx = find_vertex_near(x, y)
            if name is not None:
                drag["region"], drag["index"] = name, idx
            else:
                regions[active[0]].append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if drag["region"] is not None:
                regions[drag["region"]][drag["index"]] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drag["region"], drag["index"] = None, None

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_mouse)

    print("\nDraw ALL regions on one view:")
    print("  1-9 = pick which region new clicks add to")
    print("  click = add a point to the active region, or drag an existing point")
    print("  z = undo active region's last point, ENTER = finish, ESC = cancel")

    while True:
        base = frame.copy()
        overlay = frame.copy()

        for i, name in enumerate(roi_names):
            pts = regions[name]
            color = ROI_COLORS[i % len(ROI_COLORS)]
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], color)

        disp = cv2.addWeighted(overlay, 0.25, base, 0.75, 0)

        for i, name in enumerate(roi_names):
            pts = regions[name]
            color = ROI_COLORS[i % len(ROI_COLORS)]
            is_active = name == active[0]

            if len(pts) >= 2:
                cv2.polylines(
                    disp, [np.array(pts, dtype=np.int32)], len(pts) >= 3,
                    color, 3 if is_active else 2
                )

            for p in pts:
                r = 6 if is_active else 4
                cv2.circle(disp, p, r, color, -1)
                cv2.circle(disp, p, r, (255, 255, 255), 1)

            if pts:
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))
                cv2.putText(disp, name, (cx - 25, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        title = "Draw Regions: " + ", ".join(roi_names)
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 24), (30, 30, 30), -1)
        cv2.putText(disp, title[:100], (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        key_hint = "  ".join(f"[{k}]{v}" for k, v in key_to_name.items())
        status = f"Active: {active[0]} ({len(regions[active[0]])} pts)   {key_hint}"
        cv2.putText(disp, status, (10, disp.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(
            disp, "z=undo  ENTER=finish (all regions need 3+ pts)  ESC=cancel",
            (10, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(20) & 0xFF
        key_char = chr(key) if 0 <= key < 256 else ""

        if key_char in key_to_name:
            active[0] = key_to_name[key_char]
        elif key_char == "z" and regions[active[0]]:
            regions[active[0]].pop()
        elif key in (13, 32):
            incomplete = [n for n, p in regions.items() if len(p) < 3]
            if incomplete:
                print(f"Still need at least 3 points for: {', '.join(incomplete)}")
            else:
                break
        elif key == 27:
            cv2.destroyWindow(window_title)
            raise SystemExit("Cancelled during region drawing.")

    cv2.destroyWindow(window_title)
    return regions



def select_object_point(frame, name):
    picked = []
    window_title = f"Click center of OBJECT: {name}  (click, r=reset, ENTER=confirm, ESC=cancel)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            picked[:] = [(x, y)]

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_mouse)

    while True:
        disp = frame.copy()

        if picked:
            cv2.circle(disp, picked[0], 6, (255, 128, 0), -1)

        cv2.putText(
            disp, f"Object '{name}' (r=reset, ENTER=confirm, ESC=cancel)",
            (20, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2
        )

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            picked = []
        elif key in (13, 32) and picked:
            break
        elif key == 27:
            cv2.destroyWindow(window_title)
            raise SystemExit(f"Cancelled while marking object '{name}'.")

    cv2.destroyWindow(window_title)
    return picked[0]



def select_two_points(frame, window_title="Click 2 points of known distance"):
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_mouse)

    print("\nClick 2 points of known real-world distance apart.")
    print("r = reset, ENTER = confirm, ESC = skip calibration.")

    result = None

    while True:
        disp = frame.copy()

        for p in points:
            cv2.circle(disp, p, 6, (255, 0, 255), -1)

        if len(points) == 2:
            cv2.line(disp, points[0], points[1], (255, 0, 255), 2)
            d = point_distance(points[0], points[1])
            cv2.putText(disp, f"{d:.1f} px", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.putText(
            disp, "r=reset  ENTER=confirm  ESC=skip",
            (20, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        cv2.imshow(window_title, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points = []
        elif key in (13, 32) and len(points) == 2:
            result = points
            break
        elif key == 27:
            result = None
            break

    cv2.destroyWindow(window_title)
    return result



def get_scale_factor(frame):
    if not ask_yes_no(
        "Distance calibration",
        "Calibrate pixel-to-real-world distance now?\n(needed for a Distance_<unit> column, e.g. cm)"
    ):
        return None, None

    pts = select_two_points(frame)

    if pts is None:
        return None, None

    px_distance = point_distance(pts[0], pts[1])

    if px_distance <= 0:
        return None, None

    true_distance = ask_number("Real-world distance", "Real-world distance between those 2 points:", 30)
    unit = ask_text("Units", "Units for that distance (e.g. cm, mm, in):", "cm")

    return true_distance / px_distance, unit


# -----------------------------
# Background estimation
# -----------------------------



def calibrate(video_path, include_interactions=True):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print("\nVideo:", video_path)
    print("FPS:", fps)
    print(f"Duration: {duration:.2f} seconds")

    # ---- step functions ----

    def step_time_window(state):
        start_time = ask_number(
            "Start time", f"START time in seconds (0-{duration:.2f}):", state.get("start_time", 0)
        )
        end_time = ask_number(
            "End time", f"END time in seconds ({start_time:.2f}-{duration:.2f}):",
            state.get("end_time", min(600, duration))
        )
        start_time = max(0, min(start_time, duration))
        end_time = max(start_time, min(end_time, duration))

        if end_time <= start_time:
            raise SystemExit("End time must be greater than start time.")

        state["start_time"] = start_time
        state["end_time"] = end_time

    def step_arena(state):
        start_frame = int(state["start_time"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame_raw = cap.read()

        if not ok:
            raise RuntimeError("Could not read start frame.")

        use_crop = ask_yes_no(
            "Arena cropping",
            "Crop/rectify the arena using a 4-point perspective correction?\n\n"
            "Yes = click the 4 arena corners (fixes camera tilt/angle)\n"
            "No = use the full video frame as-is, uncropped"
        )

        if use_crop:
            print("\nSelect the 4 ARENA CORNERS.")
            corner_pts = select_four_points(first_frame_raw)
            matrix, warp_w, warp_h = compute_perspective_transform(corner_pts)
            preview_title = "Rectified arena with scale reference - press any key"
        else:
            h, w = first_frame_raw.shape[:2]
            matrix, warp_w, warp_h = identity_transform(w, h)
            preview_title = "Full frame (uncropped) with scale reference - press any key"

        first_frame = cv2.warpPerspective(first_frame_raw, matrix, (warp_w, warp_h))

        overlay = draw_scale_overlay(first_frame)
        cv2.imshow(preview_title, overlay)
        cv2.waitKey(0)
        cv2.destroyWindow(preview_title)

        state["use_crop"] = use_crop
        state["matrix"] = matrix
        state["warp_w"] = warp_w
        state["warp_h"] = warp_h
        state["arena"] = (0, 0, warp_w, warp_h)
        state["first_frame"] = first_frame

    def step_zone_names(state):
        default = ",".join(state.get("roi_names", []))
        names_raw = ask_text(
            "Zone names",
            "Enter zone/ROI names, comma-separated -- as many as this paradigm needs\n"
            "(e.g. Light,Dark  /  Open,Closed  /  Center,Left,Right for EPM/Y-maze):",
            default
        )
        state["roi_names"] = [n.strip() for n in names_raw.split(",") if n.strip()]

        if not state["roi_names"]:
            raise SystemExit("At least one zone name is required.")

    def step_zone_polygons(state):
        state["roi_points"] = edit_regions_interactive(state["first_frame"], state["roi_names"])

    def step_object_names(state):
        default = ",".join(state.get("object_names", []))
        names_raw = ask_text(
            "Interaction objects",
            "Names of objects/stimuli to track interaction with, comma-separated.\n"
            "Leave blank if this paradigm has none:",
            default
        )
        state["object_names"] = [n.strip() for n in names_raw.split(",") if n.strip()]

    def step_object_points(state):
        state["object_points"] = edit_regions_interactive(
            state["first_frame"], state["object_names"]
        ) if state["object_names"] else {}

    def step_behavior_names(state):
        if not state.get("object_names"):
            state["behavior_names"] = []
            return

        default = ",".join(state.get("behavior_names", []))
        names_raw = ask_text(
            "Interaction/behavior types",
            "Behavior labels to choose from when tagging an interaction bout,\n"
            "comma-separated, up to 9 (e.g. Sniffing,Touching,Climbing):",
            default
        )
        state["behavior_names"] = [n.strip() for n in names_raw.split(",") if n.strip()][:9]

    def step_detection_settings(state):
        state["background_samples"] = int(ask_number(
            "Background samples",
            "Number of frames to median-average for the background model (default 100):",
            state.get("background_samples", 100)
        ))
        state["threshold"] = ask_number(
            "Background threshold", "Difference threshold.\nStart with 25 for this video:",
            state.get("threshold", 25)
        )
        state["min_area"] = ask_number(
            "Minimum mouse area", "Minimum detected mouse area in pixels.\nKeep this LOW:",
            state.get("min_area", 15)
        )
        state["max_area"] = ask_number(
            "Maximum mouse area", "Maximum detected object area in pixels:", state.get("max_area", 5000)
        )
        state["max_jump"] = ask_number(
            "Maximum movement per frame", "Maximum expected movement between frames, pixels:",
            state.get("max_jump", 100)
        )
        state["use_zone_threshold"] = ask_yes_no(
            "Per-zone adaptive threshold",
            "Use a separate auto (Otsu) threshold PER ZONE during detection?\n\n"
            "Turn this ON only if your zones have genuinely different brightness "
            "(e.g. a Light zone vs a Dark zone).\n\n"
            "Leave it OFF for same-lit zones (EPM arms, Y-maze, three-chamber, "
            "open field, etc.) -- one global threshold is more correct there, "
            "and an overlapping region like 'Whole_Arena' would otherwise starve "
            "the per-zone split of pixels to work with."
        )

    def step_window_weighting(state):
        state["use_window"] = ask_yes_no(
            "Prior-position weighting",
            "Also down-weight detections far from the last known position?\n"
            "(soft window layered on top of the max-jump gate and local recovery)"
        )
        if state["use_window"]:
            state["window_size"] = ask_number("Window size", "Window side length in pixels:",
                                               state.get("window_size", 120))
            state["window_weight"] = ask_number(
                "Window weight", "Window weight, 0-1 (1 = fully suppress outside the window):",
                state.get("window_weight", 0.5)
            )
        else:
            state["window_size"] = state.get("window_size", 120)
            state["window_weight"] = state.get("window_weight", 0.5)

    def step_scale_calibration(state):
        factor, unit = get_scale_factor(state["first_frame"])
        state["scale_factor"] = factor
        state["scale_unit"] = unit

    def step_preview_samples(state):
        state["preview_samples"] = int(ask_number(
            "Preview samples", "How many calibration preview frames to save/check?",
            state.get("preview_samples", 6)
        ))

    steps = [
        ("Time window", step_time_window),
        ("Arena corners", step_arena),
        ("Zone names", step_zone_names),
        ("Zone polygons", step_zone_polygons),
    ]

    if include_interactions:
        steps += [
            ("Interaction object names", step_object_names),
            ("Interaction object points", step_object_points),
            ("Interaction behavior types", step_behavior_names),
        ]
    # When interactions are off, object_names/object_points/behavior_names are
    # simply never set on `state` -- process_single_video already reads all
    # three with .get(..., default), so no further change is needed there.

    steps += [
        ("Detection settings", step_detection_settings),
        ("Prior-position weighting", step_window_weighting),
        ("Distance calibration", step_scale_calibration),
        ("Preview sample count", step_preview_samples),
    ]

    state = {"video_path": video_path}
    run_wizard(steps, state)

    cap.release()

    state["fps"] = fps
    state["duration"] = duration
    return state



def recalibrate_spatial_only(video_path, base_setup):
    """
    Batch mode, camera NOT constant across videos: re-run the spatial
    steps only (arena corners, zone polygons, object points) for this
    video, reusing every other setting from base_setup. Still uses the
    same accept/redo/back wizard.
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_time = min(base_setup["start_time"], duration)
    end_time = min(base_setup["end_time"], duration)

    def step_arena(state):
        start_frame = int(start_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, first_frame_raw = cap.read()

        if not ok:
            raise RuntimeError("Could not read start frame.")

        print(f"\nRe-calibrating arena for: {video_path}")

        # Reuse the crop/no-crop choice from base_setup rather than asking
        # again for every video in the batch -- if cropping wasn't used
        # before, "the camera moved" doesn't imply that's changed.
        if base_setup.get("use_crop", True):
            corner_pts = select_four_points(first_frame_raw, "Click the 4 arena corners for THIS video")
            matrix, warp_w, warp_h = compute_perspective_transform(corner_pts)
        else:
            h, w = first_frame_raw.shape[:2]
            matrix, warp_w, warp_h = identity_transform(w, h)

        first_frame = cv2.warpPerspective(first_frame_raw, matrix, (warp_w, warp_h))

        overlay = draw_scale_overlay(first_frame)
        cv2.imshow("Arena preview - press any key", overlay)
        cv2.waitKey(0)
        cv2.destroyWindow("Arena preview - press any key")

        state["matrix"] = matrix
        state["warp_w"] = warp_w
        state["warp_h"] = warp_h
        state["arena"] = (0, 0, warp_w, warp_h)
        state["first_frame"] = first_frame

    def step_zone_polygons(state):
        state["roi_points"] = edit_regions_interactive(state["first_frame"], base_setup["roi_names"])

    def step_object_points(state):
        names = base_setup.get("object_names", [])
        state["object_points"] = edit_regions_interactive(state["first_frame"], names) if names else {}

    steps = [
        ("Arena corners", step_arena),
        ("Zone polygons", step_zone_polygons),
        ("Interaction object points", step_object_points),
    ]

    state = {}
    run_wizard(steps, state)
    cap.release()

    setup = dict(base_setup)
    setup.update({
        "video_path": video_path,
        "fps": fps,
        "duration": duration,
        "start_time": start_time,
        "end_time": end_time,
        "matrix": state["matrix"],
        "warp_w": state["warp_w"],
        "warp_h": state["warp_h"],
        "arena": state["arena"],
        "roi_points": state["roi_points"],
        "object_points": state["object_points"],
    })
    return setup


# -----------------------------
# Output folder
# -----------------------------



SETUP_SAVE_KEYS = [
    "use_crop", "matrix", "warp_w", "warp_h", "arena",
    "roi_names", "roi_points", "mask_points",
    "object_names", "object_points", "behavior_names",
    "background_samples", "threshold", "min_area", "max_area", "max_jump",
    "use_zone_threshold", "use_window", "window_size", "window_weight",
    "scale_factor", "scale_unit", "preview_samples",
    "start_time", "end_time",
    "compute_arm_entries", "compute_alternation",
]



def save_setup(setup, path):
    data = {}
    for key in SETUP_SAVE_KEYS:
        if key not in setup:
            continue
        value = setup[key]
        if key == "matrix":
            value = value.tolist()
        data[key] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=2)



def load_setup(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "matrix" in data:
        data["matrix"] = np.array(data["matrix"], dtype=np.float32)
    # roi_points/object_points/arena come back as lists instead of tuples --
    # every consumer already indexes into them rather than requiring a tuple,
    # so no further conversion is needed.
    return data


# -----------------------------
# Per-video pipeline
# -----------------------------



def messagebox_summary(summary):
    lines = [
        f"Tracking quality: {summary.get('Tracking_quality_percent', 0):.2f}%",
        f"Total transitions: {summary.get('Total_transitions', 0)}",
    ]
    if summary.get("Alternation_percent") is not None:
        lines.append(f"Alternation: {summary['Alternation_percent']:.1f}%")
    if summary.get("Total_interaction_bouts") is not None:
        lines.append(f"Interaction bouts: {summary.get('Total_interaction_bouts', 0)}")
    lines.append(f"\nResults saved in:\n{summary.get('Output_folder', '')}")
    messagebox.showinfo("Tracking Complete", "\n".join(lines))



def run_single():
    video_path = select_video()

    if not video_path:
        return

    setup = calibrate(video_path)
    summary = process_single_video(video_path, setup, show_display=True)
    messagebox_summary(summary)



def run_batch():
    folder = select_folder()

    if not folder:
        return

    videos = list_videos(folder)

    if not videos:
        raise SystemExit("No video files found in that folder.")

    print(f"\nFound {len(videos)} video(s) in: {folder}")

    same_camera = ask_yes_no(
        "Camera position",
        f"Found {len(videos)} videos.\n\n"
        "Was the camera in EXACTLY the same position/angle for all of them?\n\n"
        "Yes = calibrate once and reuse for every video (fast)\n"
        "No = re-select the arena corners, zones, and objects for EACH video"
    )

    show_display = ask_yes_no(
        "Live display",
        "Show the live tracking window (and interaction-tagging review) while processing each video?\n"
        "(No is much faster for large batches -- bouts still get detected, just left Unclassified)"
    )

    use_saved_setup = ask_yes_no(
        "Setup file",
        "Load a previously saved setup (zones/objects/thresholds) instead of\n"
        "calibrating the first video from scratch?"
    )

    if use_saved_setup:
        setup_path = filedialog.askopenfilename(
            title="Load setup", filetypes=[("Tracker setup", "*.json"), ("All files", "*.*")]
        )
        if not setup_path:
            raise SystemExit("No setup file selected.")
        loaded = load_setup(setup_path)
        if same_camera:
            base_setup = dict(loaded)
            base_setup["video_path"] = videos[0]
        else:
            base_setup = recalibrate_spatial_only(videos[0], loaded)
    else:
        base_setup = calibrate(videos[0])
        if ask_yes_no(
            "Save setup",
            "Save this setup (zones/objects/thresholds) so you can reuse it\n"
            "next time without recalibrating?"
        ):
            save_path = filedialog.asksaveasfilename(
                title="Save setup", defaultextension=".json", filetypes=[("Tracker setup", "*.json")]
            )
            if save_path:
                save_setup(base_setup, save_path)

    summaries = []

    for i, video_path in enumerate(videos, start=1):
        print(f"\n[{i}/{len(videos)}] Processing: {video_path}")

        if video_path == videos[0]:
            setup = base_setup
        elif same_camera:
            setup = dict(base_setup)
            setup["video_path"] = video_path
        else:
            setup = recalibrate_spatial_only(video_path, base_setup)

        try:
            summary = process_single_video(video_path, setup, show_display=show_display)
            summaries.append(summary)
        except SystemExit as exc:
            print(f"  Skipped: {exc}")
            continue

    if summaries:
        batch_df = pd.DataFrame(summaries)
        batch_path = os.path.join(folder, "BatchSummary.csv")
        batch_df.to_csv(batch_path, index=False)

        print("\n" + "=" * 65)
        print("BATCH COMPLETE")
        print("=" * 65)
        print(f"Processed {len(summaries)}/{len(videos)} videos.")
        print(f"Batch summary: {batch_path}")

        messagebox.showinfo(
            "Batch Complete",
            f"Processed {len(summaries)}/{len(videos)} videos.\n\nBatch summary:\n{batch_path}"
        )
    else:
        print("\nNo videos were successfully processed.")



def main():
    print("=" * 65)
    print("ANIMAL BEHAVIOUR TRACKER")
    print("=" * 65)

    launch_gui()


class TrackerApp:
    """
    Main dashboard: a video queue, an always-visible settings sidebar, and
    a center preview panel where Crop Arena / Mask Zone / Draw Zones /
    Mark Objects / Calibrate Distance are all done DIRECTLY on an embedded
    canvas (click to add a point, drag to move one, Finish/Cancel to end
    the operation) -- no separate popup windows for these anymore.

    Re-clicking a tool at any point re-opens it pre-loaded with whatever
    was drawn before (crop/mask/zones), so correcting an earlier mistake
    doesn't mean starting over from scratch.

    Honest scope note: the live TRACKING display (while a video is
    actually being analyzed, after you hit Start) still uses its own
    OpenCV window rather than this same embedded canvas -- embedding
    continuous video playback here is a separate, larger piece of work
    than embedding the (comparatively static, point-and-click) setup
    steps, and is not part of this pass.
    """

    def __init__(self, root):
        self.root = root

        self.videos = []
        self.active_index = None

        self.pending_use_crop = None
        self.pending_matrix = None
        self.pending_warp_w = None
        self.pending_warp_h = None
        self.pending_crop_corners = None
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        self.pending_scale_factor = None
        self.pending_scale_unit = None

        self._canvas_photo = None
        self._canvas_scale = 1.0
        self._canvas_off_x = 0
        self._canvas_off_y = 0
        self._op = None
        self.active_tool = None

        # ---- Persistent memory (cleared ONLY by Reset All) ----
        # Stores every user-entered value so switching analysis types,
        # adding/removing videos, re-editing calibration steps, etc.
        # never loses user progress. Only "Reset All" clears this dict.
        self._memory = {}

        self._last_built_mode = None

        APP_VERSION = "v1.8"
        APP_BRAND = "BehavioralTracker"
        APP_COPYRIGHT = f"© 2025 {APP_BRAND}"

        root.title(APP_BRAND)
        root.geometry("1550x980")
        root.minsize(1200, 760)
        root.resizable(True, True)

        try:
            icon_path = resource_path(os.path.join("resources", "icon.png"))
            if os.path.exists(icon_path):
                self._icon_image = tk.PhotoImage(file=icon_path)
                root.iconphoto(True, self._icon_image)
        except Exception:
            pass

        # ---- color palette + ttk styling (commercial, clean) ----
        BG = "#f7f7f7"
        SURFACE = "#ffffff"
        HEADER_BG = "#1f2937"
        ACCENT = "#2f6fb0"
        ACCENT_DARK = "#24557f"
        ACCENT_LIGHT = "#e8f0fb"
        SUCCESS = "#4caf7d"
        DANGER = "#b0392f"
        DANGER_DARK = "#8f2e26"
        TEXT = "#1a1a1a"
        TEXT_SOFT = "#4b5563"
        MUTED = "#666666"
        BORDER = "#c9c9c9"

        root.configure(bg=BG)

        style = ttk.Style()
        for theme in ("clam", "vista", "aqua"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("TCheckbutton", background=BG, foreground=TEXT, focuscolor="none")
        style.configure("TRadiobutton", background=BG, foreground=TEXT, font=("Segoe UI", 10), focuscolor="none")
        style.map("TRadiobutton", foreground=[("selected", ACCENT)],
                  font=[("selected", ("Segoe UI", 10, "bold"))])
        style.configure("TLabelframe", background=BG, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG, foreground=TEXT, font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground=SURFACE)

        style.configure("Header.TFrame", background=HEADER_BG)
        style.configure("HeaderTitle.TLabel", background=HEADER_BG, foreground="white",
                        font=("Segoe UI", 15, "bold"))
        style.configure("HeaderVersion.TLabel", background=HEADER_BG, foreground="#bcd4f5",
                        font=("Segoe UI", 8, "bold"))
        style.configure("HeaderSubtitle.TLabel", background=HEADER_BG, foreground="#93a6c2",
                        font=("Segoe UI", 8))
        style.configure("HeaderBadge.TLabel", background=ACCENT, foreground="white",
                        font=("Segoe UI", 11, "bold"), anchor="center")
        style.configure("HeaderBtn.TButton", background=HEADER_BG, foreground="white",
                        font=("Segoe UI", 8, "bold"), padding=5, borderwidth=1, relief="solid")
        style.map("HeaderBtn.TButton", background=[("active", ACCENT_DARK)],
                  foreground=[("active", "white")])

        style.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)], foreground=[("active", "white")])

        style.configure("Start.TButton", background=SUCCESS, foreground="white",
                        font=("Segoe UI", 12, "bold"))
        style.map("Start.TButton", background=[("active", "#3f9468")], foreground=[("active", "white")])

        style.configure("Tool.TButton", padding=6, font=("Segoe UI", 9))
        style.configure("ToolActive.TButton", padding=6, font=("Segoe UI", 9, "bold"),
                        background=ACCENT, foreground="white")
        style.map("ToolActive.TButton", background=[("active", ACCENT_DARK)], foreground=[("active", "white")])

        style.configure("Reset.TButton", padding=2, font=("Segoe UI", 8))
        style.configure("Danger.TButton", padding=4, font=("Segoe UI", 9, "bold"),
                        background=DANGER, foreground="white")
        style.map("Danger.TButton", background=[("active", DANGER_DARK)], foreground=[("active", "white")])

        style.configure("Footer.TFrame", background="#ececec")
        style.configure("Footer.TLabel", background="#ececec", foreground=TEXT_SOFT, font=("Segoe UI", 8))
        style.configure("FooterAccent.TLabel", background="#ececec", foreground=ACCENT, font=("Segoe UI", 8, "bold"))

        style.configure("Green.Horizontal.TProgressbar", background=SUCCESS, troughcolor="#e2e2e2")

        self._palette = {"BG": BG, "SURFACE": SURFACE, "HEADER_BG": HEADER_BG, "ACCENT": ACCENT,
                         "ACCENT_DARK": ACCENT_DARK, "ACCENT_LIGHT": ACCENT_LIGHT,
                         "SUCCESS": SUCCESS, "DANGER": DANGER, "TEXT": TEXT, "TEXT_SOFT": TEXT_SOFT,
                         "MUTED": MUTED, "BORDER": BORDER, "APP_BRAND": APP_BRAND,
                         "APP_VERSION": APP_VERSION}

        # ---- header bar (B badge + title/version/subtitle + Open/Save Project,
        # Settings, Help, Reset All) ----
        header = ttk.Frame(root, style="Header.TFrame", padding=(14, 10))
        header.pack(fill="x")

        brand_box = ttk.Frame(header, style="Header.TFrame")
        brand_box.pack(side="left")
        ttk.Label(brand_box, text="B", style="HeaderBadge.TLabel", width=2).pack(side="left")
        title_col = ttk.Frame(brand_box, style="Header.TFrame")
        title_col.pack(side="left", padx=(10, 0))
        title_row = ttk.Frame(title_col, style="Header.TFrame")
        title_row.pack(anchor="w")
        ttk.Label(title_row, text=APP_BRAND, style="HeaderTitle.TLabel").pack(side="left")
        ttk.Label(title_row, text=f" {APP_VERSION}", style="HeaderVersion.TLabel").pack(side="left", padx=(8, 0))
        ttk.Label(title_col, text="Track • Analyze • Understand Behavior",
                  style="HeaderSubtitle.TLabel").pack(anchor="w")

        header_actions = ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side="right")
        ttk.Button(header_actions, text="Reset All", style="Danger.TButton",
                  command=self.on_reset_all).pack(side="right")
        ttk.Button(header_actions, text="Help", style="HeaderBtn.TButton",
                  command=self.on_open_help).pack(side="right", padx=(0, 8))
        ttk.Button(header_actions, text="Settings", style="HeaderBtn.TButton",
                  command=self.on_open_settings).pack(side="right", padx=(0, 8))
        ttk.Button(header_actions, text="Save Project", style="HeaderBtn.TButton",
                  command=self.on_save_project).pack(side="right", padx=(0, 8))
        ttk.Button(header_actions, text="Open Project", style="HeaderBtn.TButton",
                  command=self.on_open_project).pack(side="right", padx=(0, 8))

        # ---- mode row ----
        mode_row = ttk.Frame(root, padding=(14, 8))
        mode_row.pack(fill="x")
        self.mode_var = tk.StringVar(value="individual")
        self.mode_var.trace_add("write", self._on_mode_change)
        ttk.Radiobutton(mode_row, text="Individual", value="individual", variable=self.mode_var).pack(side="left")
        ttk.Radiobutton(mode_row, text="Batch", value="batch", variable=self.mode_var).pack(side="left", padx=(20, 0))
        self.status_label = ttk.Label(mode_row, text="Idle.", foreground=MUTED)
        self.status_label.pack(side="right")

        # ---- analysis type row (cards with a subtitle + a Setup/Results
        # step tracker on the right) ----
        at_row = ttk.Frame(root, padding=(14, 8))
        at_row.pack(fill="x")

        at_left = ttk.Frame(at_row)
        at_left.pack(side="left")
        ttk.Label(at_left, text="Analysis Type:", font=("Segoe UI", 9, "bold")).pack(side="left", pady=(4, 0))

        self.analysis_type = "standard"
        self.analysis_type_buttons = {}
        at_defs = [
            ("standard", "Standard Tracking", "Single mouse - zones & distance"),
            ("multi_mouse", "Multi-Mouse Tracking", "2-3 mice, ID-matched tracks"),
            ("behavior", "Behavior Classification", "Grooming / rearing / locomotion"),
        ]
        for val, text, subtitle in at_defs:
            cell = ttk.Frame(at_left)
            cell.pack(side="left", padx=(10, 0))
            b = ttk.Button(cell, text=text, style=("Accent.TButton" if val == "standard" else "Tool.TButton"),
                          command=lambda v=val: self._set_analysis_type(v))
            b.pack(fill="x")
            ttk.Label(cell, text=subtitle, font=("Segoe UI", 7), foreground=MUTED).pack(anchor="w", pady=(2, 0))
            self.analysis_type_buttons[val] = b

        step_frame = ttk.Frame(at_row)
        step_frame.pack(side="right", anchor="n")
        self.step_labels = {}
        step_defs = [("setup", "1  Setup"), ("results", "2  Results")]
        for i, (key, text) in enumerate(step_defs):
            if i > 0:
                ttk.Label(step_frame, text="→", foreground=MUTED).pack(side="left")
            lbl = ttk.Label(step_frame, text=text, font=("Segoe UI", 8, "bold"), foreground=MUTED,
                            padding=(8, 3))
            lbl.pack(side="left")
            self.step_labels[key] = lbl

        ttk.Separator(root).pack(fill="x")

        self.body_container = ttk.Frame(root)
        self.body_container.pack(fill="both", expand=True)

        self._build_setup_body()

        # ---- footer status bar (subtle) ----
        footer = tk.Frame(root, background="#ececec", height=24)
        footer.pack(fill="x", side="bottom")
        footer.pack_propagate(False)
        tk.Label(footer, text=APP_COPYRIGHT, background="#ececec",
                 foreground=TEXT_SOFT, font=("Segoe UI", 8), padx=14).pack(side="left")
        tk.Label(footer, text=APP_VERSION, background="#ececec",
                 foreground=ACCENT, font=("Segoe UI", 8, "bold"), padx=14).pack(side="right")

    # ------------------------------------------------------------------
    # Analysis type switching + setup-body construction
    # ------------------------------------------------------------------

    def _set_analysis_type(self, value):
        if value == self.analysis_type:
            return
        self.analysis_type = value
        for val, btn in self.analysis_type_buttons.items():
            btn.configure(style="Accent.TButton" if val == value else "Tool.TButton")
        self._save_all_to_memory(include_calibration=True)
        self._reset_calibration_state()
        self._build_setup_body()

    def _set_step(self, active):
        """Highlights the current stage in the Setup/Results step tracker
        next to Analysis Type. Only those two stages are real right now --
        there's no Run or Export screen in the app yet, so the tracker
        deliberately only has the two steps it can actually reflect."""
        if not hasattr(self, "step_labels"):
            return
        ACCENT = self._palette["ACCENT"]
        MUTED = self._palette["MUTED"]
        for key, lbl in self.step_labels.items():
            try:
                lbl.configure(foreground=(ACCENT if key == active else MUTED))
            except tk.TclError:
                pass

    # ---------------- Memory helpers (cleared ONLY by Reset All) ----------------
    def _mem_get(self, key, default=""):
        return self._memory.get(key, default)

    def _mem_save_entry(self, attr_name, mem_key=None):
        if mem_key is None:
            mem_key = attr_name
        widget = getattr(self, attr_name, None)
        if widget is None:
            return
        try:
            self._memory[mem_key] = widget.get()
        except (tk.TclError, AttributeError):
            pass

    def _mem_save_var(self, attr_name, mem_key=None):
        if mem_key is None:
            mem_key = attr_name
        var = getattr(self, attr_name, None)
        if var is None:
            return
        try:
            self._memory[mem_key] = var.get()
        except (tk.TclError, AttributeError):
            pass

    def _save_all_to_memory(self, include_calibration=True):
        for attr_name in (
            "start_entry", "end_entry", "roi_names_entry", "object_names_entry",
            "interaction_margin_entry",
            "output_size_entry", "bg_samples_entry", "threshold_entry",
            "min_area_entry", "max_area_entry", "max_jump_entry",
            "window_size_entry", "window_weight_entry",
            "real_distance_entry", "units_entry", "preview_samples_entry",
            "color_mode_var", "use_window_var", "use_zone_threshold_var",
            "reject_shadows_var", "loc_var", "interact_var", "entries_var",
            "altern_var", "all_var", "num_animals",
            "loco_thresh_entry", "rear_thresh_entry", "groom_thresh_entry",
            "immobile_thresh_entry", "min_bout_entry",
            "behavior_rearing_var", "behavior_grooming_var",
            "behavior_locomotion_var", "behavior_immobile_var",
            "behavior_all_var",
        ):
            if attr_name == "num_animals":
                if hasattr(self, "num_animals"):
                    self._memory["num_animals"] = self.num_animals
            elif attr_name.endswith("_var"):
                self._mem_save_var(attr_name)
            else:
                self._mem_save_entry(attr_name)
        if include_calibration:
            for k in ("pending_use_crop", "pending_matrix", "pending_warp_w",
                      "pending_warp_h", "pending_crop_corners", "pending_roi_points",
                      "pending_object_points", "pending_mask_points",
                      "pending_scale_factor", "pending_scale_unit"):
                if hasattr(self, k):
                    self._memory[k] = getattr(self, k)

    @property
    def _remembered_start(self):
        return self._mem_get("start_entry", "")

    @_remembered_start.setter
    def _remembered_start(self, value):
        self._memory["start_entry"] = value

    @property
    def _remembered_end(self):
        return self._mem_get("end_entry", "")

    @_remembered_end.setter
    def _remembered_end(self, value):
        self._memory["end_entry"] = value

    @property
    def _remembered_roi_names(self):
        return self._mem_get("roi_names_entry", "")

    @_remembered_roi_names.setter
    def _remembered_roi_names(self, value):
        self._memory["roi_names_entry"] = value

    def _entry_or_default(self, attr_name, default):
        mem_val = self._mem_get(attr_name, None)
        if mem_val not in (None, ""):
            return str(mem_val)
        return str(default)

    def _var_or_default(self, attr_name, default):
        mem_val = self._mem_get(attr_name, None)
        if mem_val is not None:
            return bool(mem_val)
        return default

    def _build_setup_body(self, resave=True):
        # resave=False is only used by on_open_project: right after it has
        # just written freshly-loaded values into self._memory, re-saving
        # here would read the OLD (about-to-be-replaced) widgets and
        # immediately overwrite those loaded values with whatever text
        # happened to still be on screen from before the load.
        if resave:
            self._save_all_to_memory(include_calibration=False)

        for child in self.body_container.winfo_children():
            child.destroy()

        # Restore calibration state from memory (the drawing tools populate
        # pending_* fields directly, so they survive any rebuild).
        for k in ("pending_use_crop", "pending_matrix", "pending_warp_w",
                  "pending_warp_h", "pending_crop_corners", "pending_roi_points",
                  "pending_object_points", "pending_mask_points",
                  "pending_scale_factor", "pending_scale_unit"):
            if k in self._memory:
                setattr(self, k, self._memory[k])

        if self.analysis_type == "behavior":
            self._build_behavior_setup(self.body_container)
        else:
            self._build_zone_setup(self.body_container, self.analysis_type)

        self._render_canvas()
        self._set_step("setup")

    def _build_zone_setup(self, container, mode):
        """Standard Tracking and Multi-Mouse Tracking share this layout --
        both get the full toolbar (crop/mask/zones/objects/distance), since
        zone occupancy, object interaction, and distance calibration are
        all computed from tracked (x, y) points regardless of which engine
        produced them. The only real differences are: Location Tracking
        locked vs. unlocked, and the detection-engine area thresholds."""
        MUTED = self._palette["MUTED"]

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True, side="top")

        # ================================================================
        # LEFT chamber: video queue + time/zones + analysis
        # ================================================================
        left = ttk.Frame(body, width=330, padding=10)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        left_hdr = ttk.Frame(left)
        left_hdr.pack(fill="x")
        ttk.Label(left_hdr, text="Video", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(left_hdr, text="Reset", style="Reset.TButton",
                  command=self.on_reset_video_chamber).pack(side="right")
        ttk.Button(left_hdr, text="+", width=3, style="Accent.TButton",
                  command=self.on_add_videos).pack(side="right", padx=(0, 4))

        # Everything below the header scrolls -- this column has grown a lot
        # of fields (time window, ROI/object names, analysis checkboxes,
        # animal count) and a shorter window was clipping the bottom of it
        # with no way to reach the rest, same problem the right-hand
        # Detection Settings column already solves with a scrollbar.
        left_scroll = self._make_scrollable_panel(left)

        list_frame = ttk.Frame(left_scroll, relief="sunken", borderwidth=1)
        list_frame.pack(fill="x", pady=(6, 4))
        self.video_list_inner = ttk.Frame(list_frame)
        self.video_list_inner.pack(fill="x")

        self.video_mode_note = ttk.Label(left_scroll, text="", font=("Segoe UI", 7), foreground=MUTED)
        self.video_mode_note.pack(anchor="w")
        ttk.Label(left_scroll, text="(click a video to make it active for the preview)",
                  font=("Segoe UI", 7), foreground="#888888").pack(anchor="w")
        self.video_info_label = ttk.Label(left_scroll, text="", font=("Segoe UI", 7, "bold"), foreground=MUTED)
        self.video_info_label.pack(anchor="w", pady=(2, 0))
        self._refresh_video_list()

        time_frame = ttk.LabelFrame(left_scroll, text="Time window & zone names", padding=8)
        time_frame.pack(fill="x", pady=(10, 0))
        row1 = ttk.Frame(time_frame)
        row1.pack(fill="x")
        ttk.Label(row1, text="Start (s):").pack(side="left")
        self.start_entry = ttk.Entry(row1, width=8)
        self.start_entry.insert(0, self._remembered_start)
        self.start_entry.pack(side="left", padx=(4, 12))
        ttk.Label(row1, text="End (s):").pack(side="left")
        self.end_entry = ttk.Entry(row1, width=8)
        self.end_entry.insert(0, self._remembered_end)
        self.end_entry.pack(side="left", padx=(4, 0))
        ttk.Label(time_frame, text="ROI names (comma-separated):").pack(anchor="w", pady=(8, 0))
        self.roi_names_entry = ttk.Entry(time_frame)
        self.roi_names_entry.insert(0, self._remembered_roi_names)
        self.roi_names_entry.pack(fill="x")
        ttk.Label(time_frame, text="Object names (only used if Interaction Tracking is on):",
                  font=("Segoe UI", 7)).pack(anchor="w", pady=(6, 0))
        self.object_names_entry = ttk.Entry(time_frame)
        self.object_names_entry.insert(0, self._entry_or_default("object_names_entry", ""))
        self.object_names_entry.pack(fill="x")
        ttk.Label(time_frame, text="Interaction margin around objects, px (0 = only counts "
                  "when the tracked point is strictly inside the object's outline; raise "
                  "this to also count approaching/sniffing from just outside it):",
                  font=("Segoe UI", 7), wraplength=280, justify="left").pack(anchor="w", pady=(6, 0))
        self.interaction_margin_entry = ttk.Entry(time_frame, width=8)
        self.interaction_margin_entry.insert(0, self._entry_or_default("interaction_margin_entry", "20"))
        self.interaction_margin_entry.pack(anchor="w")

        out_frame = ttk.Frame(left_scroll, padding=(0, 8, 0, 0))
        out_frame.pack(fill="x")
        ttk.Label(out_frame, text="Video output size:").pack(side="left")
        self.output_size_entry = ttk.Combobox(
            out_frame, width=13, state="readonly",
            values=["Same as input", "1920x1080", "1280x720", "854x480", "640x480"])
        self.output_size_entry.set(self._entry_or_default("output_size_entry", "Same as input"))
        self.output_size_entry.pack(side="left", padx=(6, 0))

        analysis_frame = ttk.LabelFrame(left_scroll, text="Analysis", padding=8)
        analysis_frame.pack(fill="x", pady=(10, 0))
        self.loc_var = tk.BooleanVar(value=self._var_or_default("loc_var", True))
        self.interact_var = tk.BooleanVar(value=self._var_or_default("interact_var", False))
        self.entries_var = tk.BooleanVar(value=self._var_or_default("entries_var", False))
        self.altern_var = tk.BooleanVar(value=self._var_or_default("altern_var", False))
        self.all_var = tk.BooleanVar(value=self._var_or_default("all_var", False))

        if mode == "standard":
            ttk.Checkbutton(analysis_frame, text="Location Tracking (always on)", variable=self.loc_var,
                            state="disabled").pack(anchor="w")
        else:
            ttk.Checkbutton(analysis_frame, text="Location Tracking", variable=self.loc_var).pack(anchor="w")

        ttk.Checkbutton(analysis_frame, text="Interaction Tracking", variable=self.interact_var,
                        command=self.on_individual_toggle).pack(anchor="w")
        ttk.Checkbutton(analysis_frame, text="Arm Entries", variable=self.entries_var,
                        command=self.on_individual_toggle).pack(anchor="w")
        ttk.Checkbutton(analysis_frame, text="Arm Alternation", variable=self.altern_var,
                        command=self.on_individual_toggle).pack(anchor="w")
        if mode == "standard":
            ttk.Label(analysis_frame,
                      text="Standard Tracking also estimates nose/center/tail-base each frame "
                      "(no extra setup) and reports full/half/semi zone-entry depth (how much of "
                      "the body crossed in) in the exported CSV/Excel -- Arm Entries above adds a "
                      "stricter whole-body '_full_entries' count alongside the usual one.",
                      font=("Segoe UI", 7), foreground=MUTED, wraplength=290, justify="left").pack(
                anchor="w", pady=(4, 0))
        ttk.Separator(analysis_frame).pack(fill="x", pady=4)
        ttk.Checkbutton(analysis_frame, text="All Behaviours", variable=self.all_var,
                        command=self.on_all_toggle).pack(anchor="w")

        # ---- Quick Setup: one click from arena template straight to
        # zone-drawing, using the same real templates as the Maze Template
        # toolbar button (just with default sizes instead of a dialog).
        # "Custom Arena" is just a shortcut into the normal Draw Zones tool.
        quick_frame = ttk.LabelFrame(left_scroll, text="Quick Setup (Arena Templates)", padding=8)
        quick_frame.pack(fill="x", pady=(10, 0))
        ttk.Label(quick_frame, text="Crop or set the arena above first, then pick a shape here to "
                  "draw its zones instantly with default sizes -- or fine-tune sizes first with "
                  "'Maze Template' above.", font=("Segoe UI", 7), foreground=MUTED,
                  wraplength=290, justify="left").pack(anchor="w", pady=(0, 6))
        quick_grid = ttk.Frame(quick_frame)
        quick_grid.pack(fill="x")
        quick_tiles = [(k, MAZE_TEMPLATES[k]["label"]) for k in MAZE_TEMPLATES.keys()]
        quick_tiles.append((None, "Custom Arena"))
        self.quick_setup_buttons = {}
        for i, (key, label) in enumerate(quick_tiles):
            r, c = divmod(i, 2)
            cmd = (lambda k=key: self._quick_setup_template(k)) if key else (lambda: self._start_op("zones"))
            b = ttk.Button(quick_grid, text=label, style="Tool.TButton", command=cmd)
            b.grid(row=r, column=c, sticky="ew", padx=3, pady=3)
            self.quick_setup_buttons[key] = b
        quick_grid.columnconfigure(0, weight=1)
        quick_grid.columnconfigure(1, weight=1)

        animals_frame = ttk.Frame(left_scroll, padding=(0, 10, 0, 0))
        animals_frame.pack(fill="x")
        ttk.Label(animals_frame, text="Animals in frame:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        animals_row = ttk.Frame(animals_frame)
        animals_row.pack(fill="x", pady=(6, 0))
        default_n = 2 if mode == "multi_mouse" else 1
        self.num_animals = int(self._mem_get("num_animals", default_n))
        self.num_animals_buttons = {}
        for n in (1, 2, 3):
            b = ttk.Button(animals_row, text=str(n), width=3,
                           style=("Accent.TButton" if n == self.num_animals else "Tool.TButton"),
                           command=lambda n=n: self.on_num_animals(n))
            b.pack(side="left", padx=(0, 6))
            self.num_animals_buttons[n] = b
        if mode == "multi_mouse":
            ttk.Label(animals_frame, text="Uses the multi-mouse tracker (background subtract + split + ID match) "
                      "-- works with 2 or 3 animals",
                      font=("Segoe UI", 7), foreground=MUTED, wraplength=290, justify="left").pack(
                anchor="w", pady=(4, 0))
        else:
            ttk.Label(animals_frame, text="(2/3 animals: switch to Multi-Mouse Tracking above)",
                      font=("Segoe UI", 7), foreground="#888888").pack(anchor="w", pady=(4, 0))

        # ================================================================
        # CENTER chamber: embedded preview canvas + toolbar
        # ================================================================
        center = ttk.Frame(body, padding=10)
        center.pack(side="left", fill="both", expand=True)

        center_hdr = ttk.Frame(center)
        center_hdr.pack(fill="x")
        ttk.Label(center_hdr, text="Preview / Calibration", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(center_hdr, text="Reset", style="Reset.TButton",
                  command=self.on_reset_calibration_chamber).pack(side="right")

        toolbar = ttk.Frame(center)
        toolbar.pack(fill="x", pady=(4, 0))
        self.tool_buttons = {}
        tools = [
            ("crop", "Crop Arena", lambda: self._start_op("crop")),
            ("mask", "Mask Zone", lambda: self._start_op("mask")),
            ("zones", "Draw Zones", lambda: self._start_op("zones")),
            ("maze", "Maze Template", self._open_maze_template_dialog),
            ("objects", "Mark Objects", lambda: self._start_op("objects")),
            ("distance", "Calibrate Distance", lambda: self._start_op("distance")),
            ("nocrop", "No Crop", self.on_tool_no_crop),
        ]
        for key, text, cmd in tools:
            b = ttk.Button(toolbar, text=text, style="Tool.TButton", command=cmd)
            b.pack(side="left", padx=(0, 6))
            self.tool_buttons[key] = b

        self.op_bar = ttk.Frame(center, padding=(0, 6))
        self.op_bar.pack(fill="x")

        self._build_preview_canvas(center)

        # ================================================================
        # RIGHT chamber: detection settings
        # ================================================================
        right = ttk.Frame(body, width=320, padding=10)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        right_hdr = ttk.Frame(right)
        right_hdr.pack(fill="x")
        ttk.Label(right_hdr, text="Detection Settings", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(right_hdr, text="Reset", style="Reset.TButton",
                  command=self.on_reset_settings_chamber).pack(side="right")

        settings_frame = self._make_scrollable_panel(right)

        color_frame = ttk.LabelFrame(settings_frame, text="Analysis Color Mode", padding=8)
        color_frame.pack(fill="x", pady=(0, 10))
        self.color_mode_var = tk.StringVar(value=self._entry_or_default("color_mode_var", "auto"))
        ttk.Radiobutton(color_frame, text="Auto (recommended)",
                        variable=self.color_mode_var, value="auto").pack(anchor="w")
        ttk.Radiobutton(color_frame, text="Grayscale (faster)",
                        variable=self.color_mode_var, value="gray").pack(anchor="w")
        ttk.Radiobutton(color_frame, text="RGB / Color", variable=self.color_mode_var,
                        value="rgb").pack(anchor="w")
        ttk.Label(color_frame, text="Auto checks a few sample frames first and switches to "
                  "RGB only when the animal's color contrasts with the floor but its "
                  "brightness doesn't -- otherwise it uses the faster grayscale mode.",
                  font=("Segoe UI", 7), foreground=MUTED,
                  wraplength=255, justify="left").pack(anchor="w", padx=(18, 0))

        def setting_field(parent, label_text, attr_name, default):
            ttk.Label(parent, text=label_text, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))
            e = ttk.Entry(parent)
            e.insert(0, self._entry_or_default(attr_name, default))
            e.pack(fill="x")
            return e

        default_min_area = 15 if mode == "standard" else 150
        default_max_area = 5000 if mode == "standard" else 1200

        self.bg_samples_entry = setting_field(settings_frame, "Background samples (default 100)", "bg_samples_entry", 100)
        self.threshold_entry = setting_field(settings_frame, "Difference threshold (start 25)", "threshold_entry", 25)
        self.min_area_entry = setting_field(settings_frame, "Min mouse area, px (keep LOW)", "min_area_entry", default_min_area)
        self.max_area_entry = setting_field(
            settings_frame,
            "Max object area, px" if mode == "standard" else "Max object area, px (~1 mouse; tune this)",
            "max_area_entry", default_max_area
        )
        self.max_jump_entry = setting_field(settings_frame, "Max movement / frame, px", "max_jump_entry", 100)

        self.use_window_var = tk.BooleanVar(value=self._var_or_default("use_window_var", False))
        self.use_zone_threshold_var = tk.BooleanVar(value=self._var_or_default("use_zone_threshold_var", False))
        self.reject_shadows_var = tk.BooleanVar(value=self._var_or_default("reject_shadows_var", False))
        self.real_distance_entry = None
        self.units_entry = None

        if mode == "standard":
            ttk.Checkbutton(settings_frame, text="Prior-position weighting", variable=self.use_window_var
                            ).pack(anchor="w", pady=(8, 0))
            self.window_size_entry = setting_field(settings_frame, "  window size, px", "window_size_entry", 120)
            self.window_weight_entry = setting_field(settings_frame, "  window weight (0-1)", "window_weight_entry", 0.5)

            ttk.Checkbutton(settings_frame, text="Per-zone adaptive threshold", variable=self.use_zone_threshold_var
                            ).pack(anchor="w", pady=(8, 0))

            ttk.Checkbutton(settings_frame, text="Reject shadows", variable=self.reject_shadows_var
                            ).pack(anchor="w", pady=(8, 0))
            ttk.Label(settings_frame, text="(if the tracker keeps grabbing the animal's shadow "
                      "instead of its body, try this)", font=("Segoe UI", 7),
                      foreground=MUTED, wraplength=255, justify="left").pack(anchor="w", padx=(18, 0))
        else:
            self.window_size_entry = None
            self.window_weight_entry = None

        ttk.Separator(settings_frame).pack(fill="x", pady=8)
        ttk.Label(settings_frame, text="Distance calibration (optional)", font=("Segoe UI", 9, "bold")
                  ).pack(anchor="w")
        ttk.Label(settings_frame, text="Click 'Calibrate Distance' above, then fill in:",
                  font=("Segoe UI", 8), foreground="#777777").pack(anchor="w")
        ttk.Label(settings_frame, text="Real-world distance").pack(anchor="w", pady=(4, 0))
        self.real_distance_entry = ttk.Entry(settings_frame)
        self.real_distance_entry.insert(0, self._entry_or_default("real_distance_entry", "30"))
        self.real_distance_entry.pack(fill="x")
        ttk.Label(settings_frame, text="Units (e.g. cm, mm, in)", font=("Segoe UI", 8)
                  ).pack(anchor="w", pady=(6, 0))
        self.units_entry = ttk.Entry(settings_frame)
        self.units_entry.insert(0, self._entry_or_default("units_entry", "cm"))
        self.units_entry.pack(fill="x")

        self.preview_samples_entry = setting_field(settings_frame, "Preview frames to save/check", "preview_samples_entry", 6)

        # ---- BOTTOM: start + progress ----
        ttk.Separator(container).pack(fill="x", side="bottom")
        bottom = ttk.Frame(container, padding=(16, 12))
        bottom.pack(fill="x", side="bottom")

        left_cta = ttk.Frame(bottom)
        left_cta.pack(side="left", fill="x", expand=True)
        ttk.Label(left_cta, text="  Ready to begin analysis",
                  font=("Segoe UI", 8, "bold"), foreground=self._palette["SUCCESS"]).pack(anchor="w", pady=(0, 4))
        ttk.Button(left_cta, text="▶  START TRACKING", style="Start.TButton",
                   command=self.on_start).pack(fill="x", ipady=8)

        progress_col = ttk.Frame(bottom)
        progress_col.pack(side="left", fill="x", expand=True, padx=(18, 0))
        ttk.Label(progress_col, text="Progress",
                  font=("Segoe UI", 8, "bold"), foreground=self._palette["TEXT_SOFT"]).pack(anchor="w", pady=(0, 4))
        self.progress = ttk.Progressbar(progress_col, maximum=100, style="Green.Horizontal.TProgressbar")
        self.progress.pack(fill="x", ipady=2)
        self.progress_label = ttk.Label(progress_col, text="", font=("Segoe UI", 8), foreground=MUTED)
        self.progress_label.pack(anchor="w", pady=(2, 0))

        self._last_built_mode = mode

    def _build_behavior_setup(self, container):
        """Dedicated Behavior Classification layout -- no zones/ROIs, since
        behavior_classifier.py has no concept of them."""
        MUTED = self._palette["MUTED"]

        body = ttk.Frame(container)
        body.pack(fill="both", expand=True, side="top")

        left = ttk.Frame(body, width=330, padding=10)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        left_hdr = ttk.Frame(left)
        left_hdr.pack(fill="x")
        ttk.Label(left_hdr, text="Video", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(left_hdr, text="Reset", style="Reset.TButton",
                  command=self.on_reset_video_chamber).pack(side="right")
        ttk.Button(left_hdr, text="+", width=3, style="Accent.TButton",
                  command=self.on_add_videos).pack(side="right", padx=(0, 4))

        # Everything below the header scrolls -- same fix as the other
        # setup screens' left column, so the Deep Learning panel and
        # everything above it never gets clipped by a shorter window.
        left_scroll = self._make_scrollable_panel(left)

        list_frame = ttk.Frame(left_scroll, relief="sunken", borderwidth=1)
        list_frame.pack(fill="x", pady=(6, 4))
        self.video_list_inner = ttk.Frame(list_frame)
        self.video_list_inner.pack(fill="x")
        self.video_mode_note = ttk.Label(left_scroll, text="", font=("Segoe UI", 7), foreground=MUTED)
        self.video_mode_note.pack(anchor="w")
        ttk.Label(left_scroll, text="(click a video to make it active for the preview)",
                  font=("Segoe UI", 7), foreground="#888888").pack(anchor="w")
        self.video_info_label = ttk.Label(left_scroll, text="", font=("Segoe UI", 7, "bold"), foreground=MUTED)
        self.video_info_label.pack(anchor="w", pady=(2, 0))
        self._refresh_video_list()

        time_frame = ttk.LabelFrame(left_scroll, text="Time window", padding=8)
        time_frame.pack(fill="x", pady=(10, 0))
        row1 = ttk.Frame(time_frame)
        row1.pack(fill="x")
        ttk.Label(row1, text="Start (s):").pack(side="left")
        self.start_entry = ttk.Entry(row1, width=8)
        self.start_entry.insert(0, self._remembered_start)
        self.start_entry.pack(side="left", padx=(4, 12))
        ttk.Label(row1, text="End (s):").pack(side="left")
        self.end_entry = ttk.Entry(row1, width=8)
        self.end_entry.insert(0, self._remembered_end)
        self.end_entry.pack(side="left")

        # not used by this mode, but on_start()/_build_setup reference these
        # names generically -- keep harmless empty stand-ins
        self.roi_names_entry = tk.Entry(left_scroll)
        self.object_names_entry = tk.Entry(left_scroll)
        self.loc_var = tk.BooleanVar(value=False)
        self.interact_var = tk.BooleanVar(value=False)
        self.entries_var = tk.BooleanVar(value=False)
        self.altern_var = tk.BooleanVar(value=False)
        self.all_var = tk.BooleanVar(value=False)

        out_frame = ttk.Frame(left_scroll, padding=(0, 8, 0, 0))
        out_frame.pack(fill="x")
        ttk.Label(out_frame, text="Video output size:").pack(side="left")
        self.output_size_entry = ttk.Combobox(
            out_frame, width=13, state="readonly",
            values=["Same as input", "1920x1080", "1280x720", "854x480", "640x480"])
        self.output_size_entry.set(self._entry_or_default("output_size_entry", "Same as input"))
        self.output_size_entry.pack(side="left", padx=(6, 0))

        beh_frame = ttk.LabelFrame(left_scroll, text="Behaviors to detect", padding=8)
        beh_frame.pack(fill="x", pady=(10, 0))
        self.behavior_rearing_var = tk.BooleanVar(value=self._var_or_default("behavior_rearing_var", True))
        self.behavior_grooming_var = tk.BooleanVar(value=self._var_or_default("behavior_grooming_var", True))
        self.behavior_locomotion_var = tk.BooleanVar(value=self._var_or_default("behavior_locomotion_var", False))
        self.behavior_immobile_var = tk.BooleanVar(value=self._var_or_default("behavior_immobile_var", False))
        self.behavior_all_var = tk.BooleanVar(value=self._var_or_default("behavior_all_var", False))
        ttk.Checkbutton(beh_frame, text="Rearing", variable=self.behavior_rearing_var).pack(anchor="w")
        ttk.Checkbutton(beh_frame, text="Grooming", variable=self.behavior_grooming_var).pack(anchor="w")
        ttk.Checkbutton(beh_frame, text="Locomotion", variable=self.behavior_locomotion_var).pack(anchor="w")
        ttk.Checkbutton(beh_frame, text="Immobile", variable=self.behavior_immobile_var).pack(anchor="w")
        ttk.Separator(beh_frame).pack(fill="x", pady=4)
        ttk.Checkbutton(beh_frame, text="All", variable=self.behavior_all_var,
                        command=self._on_behavior_all_toggle).pack(anchor="w")

        animals_frame = ttk.Frame(left_scroll, padding=(0, 10, 0, 0))
        animals_frame.pack(fill="x")
        ttk.Label(animals_frame, text="Animals in frame:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        animals_row = ttk.Frame(animals_frame)
        animals_row.pack(fill="x", pady=(6, 0))
        self.num_animals = int(self._mem_get("num_animals", 1))
        self.num_animals_buttons = {}
        for n in (1, 2, 3):
            b = ttk.Button(animals_row, text=str(n), width=3,
                           style=("Accent.TButton" if n == self.num_animals else "Tool.TButton"),
                           command=lambda n=n: self.on_num_animals(n))
            b.pack(side="left", padx=(0, 6))
            self.num_animals_buttons[n] = b
        ttk.Label(animals_frame, text="2 = classifies both mice",
                  font=("Segoe UI", 7), foreground=MUTED, wraplength=300, justify="left").pack(
            anchor="w", pady=(4, 0))

        center = ttk.Frame(body, padding=10)
        center.pack(side="left", fill="both", expand=True)
        center_hdr = ttk.Frame(center)
        center_hdr.pack(fill="x")
        ttk.Label(center_hdr, text="Preview / Calibration", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(center_hdr, text="Reset", style="Reset.TButton",
                  command=self.on_reset_calibration_chamber).pack(side="right")
        toolbar = ttk.Frame(center)
        toolbar.pack(fill="x", pady=(4, 0))
        self.tool_buttons = {}
        for key, text, cmd in [("crop", "Crop Arena", lambda: self._start_op("crop")),
                                ("nocrop", "No Crop", self.on_tool_no_crop)]:
            b = ttk.Button(toolbar, text=text, style="Tool.TButton", command=cmd)
            b.pack(side="left", padx=(0, 6))
            self.tool_buttons[key] = b
        ttk.Label(toolbar, text="  (zones/objects/distance not used by this analysis type)",
                  font=("Segoe UI", 7), foreground="#888888").pack(side="left")

        self.op_bar = ttk.Frame(center, padding=(0, 6))
        self.op_bar.pack(fill="x")
        self._build_preview_canvas(center)

        right = ttk.Frame(body, width=320, padding=10)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        right_hdr = ttk.Frame(right)
        right_hdr.pack(fill="x")
        ttk.Label(right_hdr, text="Detection Settings", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(right_hdr, text="Reset", style="Reset.TButton",
                  command=self.on_reset_settings_chamber).pack(side="right")
        settings_frame = self._make_scrollable_panel(right)

        color_frame = ttk.LabelFrame(settings_frame, text="Analysis Color Mode", padding=8)
        color_frame.pack(fill="x", pady=(0, 10))
        self.color_mode_var = tk.StringVar(value=self._entry_or_default("color_mode_var", "auto"))
        ttk.Radiobutton(color_frame, text="Auto (recommended)",
                        variable=self.color_mode_var, value="auto").pack(anchor="w")
        ttk.Radiobutton(color_frame, text="Grayscale (faster)",
                        variable=self.color_mode_var, value="gray").pack(anchor="w")
        ttk.Radiobutton(color_frame, text="RGB / Color", variable=self.color_mode_var,
                        value="rgb").pack(anchor="w")
        ttk.Label(color_frame, text="Auto checks a few sample frames first and switches to "
                  "RGB only when the animal's color contrasts with the floor but its "
                  "brightness doesn't -- otherwise it uses the faster grayscale mode.",
                  font=("Segoe UI", 7), foreground=MUTED,
                  wraplength=255, justify="left").pack(anchor="w", padx=(18, 0))

        def setting_field(parent, label_text, attr_name, default):
            ttk.Label(parent, text=label_text, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))
            e = ttk.Entry(parent)
            e.insert(0, self._entry_or_default(attr_name, default))
            e.pack(fill="x")
            return e

        self.bg_samples_entry = setting_field(settings_frame, "Background samples (default 100)", "bg_samples_entry", 100)
        self.threshold_entry = setting_field(settings_frame, "Difference threshold (start 25)", "threshold_entry", 25)
        self.min_area_entry = setting_field(settings_frame, "Min mouse area, px (keep LOW)", "min_area_entry", 15)
        self.max_area_entry = setting_field(settings_frame, "Max object area, px", "max_area_entry", 5000)
        self.max_jump_entry = setting_field(settings_frame, "Max movement / frame, px", "max_jump_entry", 100)
        self.use_window_var = tk.BooleanVar(value=self._var_or_default("use_window_var", False))
        self.use_zone_threshold_var = tk.BooleanVar(value=self._var_or_default("use_zone_threshold_var", False))
        self.window_size_entry = None
        self.window_weight_entry = None
        self.real_distance_entry = None
        self.units_entry = None
        self.preview_samples_entry = setting_field(settings_frame, "Preview frames to save/check", "preview_samples_entry", 6)

        ttk.Separator(settings_frame).pack(fill="x", pady=8)
        beh_settings = ttk.LabelFrame(settings_frame, text="Behavior Classification thresholds", padding=8)
        beh_settings.pack(fill="x")
        ttk.Label(beh_settings, text="Each frame gets a pseudo-pose (nose/paws/tail) from the "
                  "detected silhouette, then grooming/rearing/locomotion are scored from several "
                  "combined cues, not one threshold.", font=("Segoe UI", 7), foreground=MUTED,
                  wraplength=255, justify="left").pack(anchor="w", pady=(0, 6))
        self.loco_thresh_entry = setting_field(beh_settings, "Locomotion speed threshold (px/s)", "loco_thresh_entry", 30)
        self.rear_thresh_entry = setting_field(beh_settings, "Rearing sensitivity 0-1 (lower = more sensitive)", "rear_thresh_entry", 0.25)
        self.groom_thresh_entry = setting_field(beh_settings, "Grooming sensitivity 0-1 (lower = more sensitive)", "groom_thresh_entry", 0.50)
        self.immobile_thresh_entry = setting_field(beh_settings, "Frames to confirm a behavior (persistence)", "immobile_thresh_entry", 6)
        self.min_bout_entry = setting_field(beh_settings, "Min bout duration (s)", "min_bout_entry", 0.3)
        ttk.Label(beh_settings, text="Grooming is the hardest of these to detect without a trained "
                  "pose model -- treat it as a starting point to validate by eye, not ground truth.",
                  font=("Segoe UI", 7), foreground=MUTED, wraplength=255, justify="left").pack(
            anchor="w", pady=(6, 0))

        ttk.Separator(settings_frame).pack(fill="x", pady=8)
        self._build_ml_classifier_panel(settings_frame, MUTED)

        # ---- BOTTOM: start + progress ----
        ttk.Separator(container).pack(fill="x", side="bottom")
        bottom = ttk.Frame(container, padding=(16, 12))
        bottom.pack(fill="x", side="bottom")

        left_cta = ttk.Frame(bottom)
        left_cta.pack(side="left", fill="x", expand=True)
        ttk.Label(left_cta, text="  Ready to begin behavior classification",
                  font=("Segoe UI", 8, "bold"), foreground=self._palette["SUCCESS"]).pack(anchor="w", pady=(0, 4))
        start_row = ttk.Frame(left_cta)
        start_row.pack(fill="x")
        ttk.Button(start_row, text="▶  START TRACKING (Automatic)", style="Start.TButton",
                   command=self.on_start).pack(side="left", fill="x", expand=True, ipady=8)
        ttk.Button(start_row, text="✎  MANUAL SCORING", style="Tool.TButton",
                   command=self.on_manual_behavior_scoring).pack(side="left", fill="x", expand=True,
                                                                  ipady=8, padx=(8, 0))
        ttk.Label(left_cta, text="Manual scoring opens the video in its own window -- you watch it "
                  "and press keys to mark when each behavior starts/stops yourself. Use it for "
                  "grooming especially, since automatic detection is weakest there.",
                  font=("Segoe UI", 7), foreground=MUTED, wraplength=420, justify="left").pack(
            anchor="w", pady=(4, 0))

        progress_col = ttk.Frame(bottom)
        progress_col.pack(side="left", fill="x", expand=True, padx=(18, 0))
        ttk.Label(progress_col, text="Progress",
                  font=("Segoe UI", 8, "bold"), foreground=self._palette["TEXT_SOFT"]).pack(anchor="w", pady=(0, 4))
        self.progress = ttk.Progressbar(progress_col, maximum=100, style="Green.Horizontal.TProgressbar")
        self.progress.pack(fill="x", ipady=2)
        self.progress_label = ttk.Label(progress_col, text="", font=("Segoe UI", 8), foreground=MUTED)
        self.progress_label.pack(anchor="w", pady=(2, 0))

        self._last_built_mode = "behavior"

    def _on_behavior_all_toggle(self):
        val = self.behavior_all_var.get()
        for var in (self.behavior_rearing_var, self.behavior_grooming_var,
                    self.behavior_locomotion_var, self.behavior_immobile_var):
            var.set(val)

    # ------------------------------------------------------------------
    # Optional deep-learning behavior classifier (tracking/ml_dataset.py,
    # ml_model.py, ml_train.py, ml_infer.py). An alternative to the rule-
    # based classifier above, not a replacement for it -- OFF by default,
    # and everything here (except actually training/running a model)
    # works without PyTorch installed, since tracking.ml_dataset has no
    # torch dependency at all.
    # ------------------------------------------------------------------

    def _build_ml_classifier_panel(self, parent, MUTED):
        ml_frame = ttk.LabelFrame(parent, text="Deep Learning Classifier (optional)", padding=8)
        ml_frame.pack(fill="x")
        ttk.Label(ml_frame, text="A small neural network you train yourself on clips you've "
                  "labeled with Manual Scoring -- an alternative to the automatic rules above "
                  "for grooming/rearing, not a replacement for it. 1) Manual Score a few videos, "
                  "2) Prepare Training Data, 3) Train Model, 4) turn this on.",
                  font=("Segoe UI", 7), foreground=MUTED, wraplength=255, justify="left").pack(
            anchor="w", pady=(0, 6))

        self.ml_mode_var = tk.BooleanVar(value=self._var_or_default("ml_mode_var", False))
        ttk.Checkbutton(ml_frame, text="Use trained model instead of automatic rules",
                        variable=self.ml_mode_var).pack(anchor="w")

        ckpt_row = ttk.Frame(ml_frame, padding=(0, 6, 0, 0))
        ckpt_row.pack(fill="x")
        self.ml_checkpoint_entry = ttk.Entry(ckpt_row)
        self.ml_checkpoint_entry.insert(0, self._entry_or_default("ml_checkpoint_entry", ""))
        self.ml_checkpoint_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(ckpt_row, text="Browse...", width=9,
                   command=self._on_browse_ml_checkpoint).pack(side="left", padx=(4, 0))
        ttk.Label(ml_frame, text="Trained model file (.pt)", font=("Segoe UI", 7),
                  foreground=MUTED).pack(anchor="w")

        self.ml_status_label = ttk.Label(ml_frame, text="", font=("Segoe UI", 7), foreground=MUTED,
                                          wraplength=255, justify="left")
        self.ml_status_label.pack(anchor="w", pady=(4, 0))

        btn_row = ttk.Frame(ml_frame, padding=(0, 8, 0, 0))
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Prepare Training Data...", style="Tool.TButton",
                   command=self._on_prepare_ml_dataset).pack(fill="x")
        ttk.Button(btn_row, text="Train Model...", style="Tool.TButton",
                   command=self._on_train_ml_model).pack(fill="x", pady=(4, 0))

        if not TORCH_AVAILABLE:
            ttk.Label(ml_frame, text="PyTorch isn't installed on this machine -- install it "
                      "(see pytorch.org) to train or use a model here. Preparing training data "
                      "doesn't need it, and everything else in the app works without it.",
                      font=("Segoe UI", 7), foreground=self._palette["DANGER"],
                      wraplength=255, justify="left").pack(anchor="w", pady=(6, 0))

    def _on_browse_ml_checkpoint(self):
        path = filedialog.askopenfilename(title="Select a trained model (.pt)",
                                           filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.ml_checkpoint_entry.delete(0, "end")
            self.ml_checkpoint_entry.insert(0, path)

    def _default_ml_dataset_dir(self):
        if self.active_index is not None and self.videos:
            active_path = self.videos[self.active_index]["path"]
            return os.path.join(os.path.dirname(active_path), "ml_dataset")
        return ""

    def _on_prepare_ml_dataset(self):
        if not self.videos:
            messagebox.showerror("No video", "Add a video first.")
            return
        if self.active_index is None:
            messagebox.showerror("No video selected", "Click a video in the list to select it.")
            return
        if self.pending_matrix is None:
            messagebox.showerror("Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        active_path = self.videos[self.active_index]["path"]
        output_dir = compute_output_dir(active_path)
        candidates = []
        for fname, src_label in [("bouts.csv", "Automatic classification"),
                                  ("manual_bouts.csv", "Manual scoring")]:
            fpath = os.path.join(output_dir, fname)
            if os.path.exists(fpath):
                candidates.append((f"{src_label} ({fname})", fpath))
        if not candidates:
            messagebox.showerror(
                "No labeled bouts found",
                "Run 'START TRACKING (Automatic)' or 'MANUAL SCORING' on this video first -- "
                "training data comes from a bouts file those produce."
            )
            return

        cap = cv2.VideoCapture(active_path)
        fps_guess = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        dialog = tk.Toplevel(self.root)
        dialog.title("Prepare Training Data")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        pad = {"padx": 10, "pady": (8, 2)}
        ttk.Label(dialog, text="Bouts source:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", **pad)
        bouts_labels = [c[0] for c in candidates]
        bouts_var = tk.StringVar(value=bouts_labels[0])
        ttk.Combobox(dialog, textvariable=bouts_var, values=bouts_labels, state="readonly",
                     width=32).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(dialog, text="Include behaviors:", font=("Segoe UI", 9, "bold")).grid(
            row=1, column=0, sticky="w", **pad)
        beh_frame = ttk.Frame(dialog)
        beh_frame.grid(row=1, column=1, sticky="w", **pad)
        groom_var = tk.BooleanVar(value=True)
        rear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(beh_frame, text="Grooming", variable=groom_var).pack(side="left")
        ttk.Checkbutton(beh_frame, text="Rearing", variable=rear_var).pack(side="left", padx=(10, 0))

        ttk.Label(dialog, text="Clip length (s):", font=("Segoe UI", 9, "bold")).grid(
            row=2, column=0, sticky="w", **pad)
        clip_len_entry = ttk.Entry(dialog, width=10)
        clip_len_entry.insert(0, "1.5")
        clip_len_entry.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(dialog, text="Dataset folder:", font=("Segoe UI", 9, "bold")).grid(
            row=3, column=0, sticky="w", **pad)
        ds_row = ttk.Frame(dialog)
        ds_row.grid(row=3, column=1, sticky="w", **pad)
        ds_entry = ttk.Entry(ds_row, width=28)
        ds_entry.insert(0, self._entry_or_default("ml_dataset_dir_entry", self._default_ml_dataset_dir()))
        ds_entry.pack(side="left")

        def browse_ds():
            path = filedialog.askdirectory(title="Choose (or create) a dataset folder")
            if path:
                ds_entry.delete(0, "end")
                ds_entry.insert(0, path)

        ttk.Button(ds_row, text="...", width=3, command=browse_ds).pack(side="left", padx=(4, 0))

        ttk.Label(dialog, text="Re-running this on more videos adds to the same folder -- "
                  "point every video at the SAME dataset folder as you label more footage. "
                  "Keep the Crop/Time-window settings the same as when you made the bouts file.",
                  font=("Segoe UI", 7), foreground=self._palette["MUTED"], wraplength=300,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 0))

        status_label = ttk.Label(dialog, text="", font=("Segoe UI", 8), wraplength=300, justify="left")
        status_label.grid(row=5, column=0, columnspan=2, sticky="w", padx=10, pady=(6, 0))

        btn_row = ttk.Frame(dialog, padding=(10, 10))
        btn_row.grid(row=6, column=0, columnspan=2, sticky="e")
        ttk.Button(btn_row, text="Close", command=dialog.destroy).pack(side="right", padx=(6, 0))
        build_btn = ttk.Button(btn_row, text="Build Dataset", style="Accent.TButton")
        build_btn.pack(side="right")

        def do_build():
            behaviors = [b for b, v in [("grooming", groom_var.get()), ("rearing", rear_var.get())] if v]
            if not behaviors:
                messagebox.showerror("Nothing selected", "Choose at least one behavior.", parent=dialog)
                return
            try:
                clip_len_s = float(clip_len_entry.get())
                window_frames = max(1, int(round(clip_len_s * fps_guess)))
            except ValueError:
                messagebox.showerror("Invalid value", "Clip length must be a number.", parent=dialog)
                return
            dataset_dir = ds_entry.get().strip()
            if not dataset_dir:
                messagebox.showerror("No dataset folder", "Choose a dataset folder.", parent=dialog)
                return
            bouts_csv = dict(candidates)[bouts_var.get()]

            build_btn.config(state="disabled")
            status_label.config(text="Preparing video and slicing clips...")
            dialog.update_idletasks()

            tmp_path = None
            try:
                source_path, tmp_path, _fps = self._prepare_source_video(active_path)
                counts = build_clip_dataset(
                    source_path, bouts_csv, dataset_dir, behaviors=behaviors,
                    window_frames=window_frames, stride_frames=max(1, window_frames // 2),
                )
            except Exception as exc:
                status_label.config(text="")
                messagebox.showerror("Could not build dataset", str(exc), parent=dialog)
                build_btn.config(state="normal")
                return
            finally:
                try:
                    if tmp_path and os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

            self._last_ml_dataset_dir = dataset_dir
            summary = dataset_summary(dataset_dir)
            lines = [f"  {cls}: {n} clips" for cls, n in summary.items()]
            status_label.config(text="Added " + ", ".join(f"{k}: {v}" for k, v in counts.items()) +
                                 ".\n\nDataset now has:\n" + "\n".join(lines))
            self.ml_status_label.config(text=f"Dataset: {dataset_dir}\n" + "\n".join(lines))
            build_btn.config(state="normal")

        build_btn.config(command=do_build)
        dialog.wait_window()

    def _on_train_ml_model(self):
        if not TORCH_AVAILABLE:
            messagebox.showerror(
                "PyTorch not installed",
                "Training needs PyTorch. Install it (see pytorch.org for the right command "
                "for your machine/GPU), then try again."
            )
            return

        dataset_dir_default = getattr(self, "_last_ml_dataset_dir", "") or self._default_ml_dataset_dir()

        dialog = tk.Toplevel(self.root)
        dialog.title("Train Model")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        pad = {"padx": 10, "pady": (8, 2)}
        ttk.Label(dialog, text="Dataset folder:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", **pad)
        ds_row = ttk.Frame(dialog)
        ds_row.grid(row=0, column=1, sticky="w", **pad)
        ds_entry = ttk.Entry(ds_row, width=28)
        ds_entry.insert(0, dataset_dir_default)
        ds_entry.pack(side="left")

        summary_label = ttk.Label(dialog, text="", font=("Segoe UI", 8), foreground=self._palette["MUTED"],
                                   wraplength=300, justify="left")
        summary_label.grid(row=1, column=0, columnspan=2, sticky="w", padx=10)

        def refresh_summary():
            d = ds_entry.get().strip()
            summary = dataset_summary(d) if d else {}
            if summary:
                lines = ", ".join(f"{k}: {v}" for k, v in summary.items())
                summary_label.config(text=f"Found -- {lines}")
            else:
                summary_label.config(text="No clips found in that folder yet -- use "
                                           "'Prepare Training Data' first.")
            return summary

        def browse_ds():
            path = filedialog.askdirectory(title="Choose a dataset folder", initialdir=ds_entry.get() or ".")
            if path:
                ds_entry.delete(0, "end")
                ds_entry.insert(0, path)
                refresh_summary()

        ttk.Button(ds_row, text="...", width=3, command=browse_ds).pack(side="left", padx=(4, 0))
        refresh_summary()

        ttk.Label(dialog, text="Training passes (epochs):", font=("Segoe UI", 9, "bold")).grid(
            row=2, column=0, sticky="w", **pad)
        epochs_entry = ttk.Entry(dialog, width=10)
        epochs_entry.insert(0, "15")
        epochs_entry.grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(dialog, text="Save trained model as:", font=("Segoe UI", 9, "bold")).grid(
            row=3, column=0, sticky="w", **pad)
        out_row = ttk.Frame(dialog)
        out_row.grid(row=3, column=1, sticky="w", **pad)
        out_entry = ttk.Entry(out_row, width=28)
        default_out = os.path.join(dataset_dir_default, "model.pt") if dataset_dir_default else ""
        out_entry.insert(0, default_out)
        out_entry.pack(side="left")

        def browse_out():
            path = filedialog.asksaveasfilename(title="Save trained model as", defaultextension=".pt",
                                                 filetypes=[("PyTorch checkpoint", "*.pt")])
            if path:
                out_entry.delete(0, "end")
                out_entry.insert(0, path)

        ttk.Button(out_row, text="...", width=3, command=browse_out).pack(side="left", padx=(4, 0))

        ttk.Label(dialog, text="More epochs and more labeled clips both help, but each epoch takes "
                  "longer without a GPU -- this runs on the CPU here, which is much slower than a "
                  "GPU. A small first run is a good way to check everything works.",
                  font=("Segoe UI", 7), foreground=self._palette["MUTED"], wraplength=300,
                  justify="left").grid(row=4, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 0))

        progress = ttk.Progressbar(dialog, maximum=100, style="Green.Horizontal.TProgressbar")
        progress.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=(8, 0))
        status_label = ttk.Label(dialog, text="", font=("Segoe UI", 8), wraplength=300, justify="left")
        status_label.grid(row=6, column=0, columnspan=2, sticky="w", padx=10, pady=(4, 0))

        btn_row = ttk.Frame(dialog, padding=(10, 10))
        btn_row.grid(row=7, column=0, columnspan=2, sticky="e")
        ttk.Button(btn_row, text="Close", command=dialog.destroy).pack(side="right", padx=(6, 0))
        train_btn = ttk.Button(btn_row, text="Start Training", style="Accent.TButton")
        train_btn.pack(side="right")

        def do_train():
            dataset_dir = ds_entry.get().strip()
            summary = refresh_summary()
            if not summary:
                messagebox.showerror("No training data", "That folder has no clips yet.", parent=dialog)
                return
            class_names = sorted(summary.keys(), key=lambda k: (k == "other", k))
            try:
                epochs = max(1, int(float(epochs_entry.get())))
            except ValueError:
                messagebox.showerror("Invalid value", "Training passes must be a whole number.", parent=dialog)
                return
            output_path = out_entry.get().strip()
            if not output_path:
                messagebox.showerror("No save location", "Choose where to save the trained model.", parent=dialog)
                return

            train_btn.config(state="disabled")

            def on_epoch(epoch, total_epochs, train_loss, val_loss, val_acc):
                progress["value"] = 100 * epoch / total_epochs
                status_label.config(text=f"Epoch {epoch}/{total_epochs} -- "
                                          f"train loss {train_loss:.3f}, val loss {val_loss:.3f}, "
                                          f"val accuracy {val_acc * 100:.0f}%")
                dialog.update_idletasks()

            try:
                _out_path, best_val_acc = train_model(
                    dataset_dir, class_names, output_path,
                    epochs=epochs, progress_callback=on_epoch,
                )
            except Exception as exc:
                status_label.config(text="")
                messagebox.showerror("Training failed", str(exc), parent=dialog)
                train_btn.config(state="normal")
                return

            status_label.config(text=f"Done. Best validation accuracy: {best_val_acc * 100:.0f}%. "
                                      f"Saved to {output_path}")
            self.ml_checkpoint_entry.delete(0, "end")
            self.ml_checkpoint_entry.insert(0, output_path)
            self.ml_status_label.config(
                text=f"Trained model ready: {os.path.basename(output_path)} "
                     f"(val accuracy {best_val_acc * 100:.0f}%). Check 'Use trained model' above to use it."
            )
            train_btn.config(state="normal")

        train_btn.config(command=do_train)
        dialog.wait_window()

    # ------------------------------------------------------------------
    # Video queue
    # ------------------------------------------------------------------

    def _on_mode_change(self, *args):
        if self.mode_var.get() == "individual" and len(self.videos) > 1:
            keep = self.active_index if self.active_index is not None else 0
            kept = self.videos[keep]
            removed = len(self.videos) - 1
            self.videos = [kept]
            self.active_index = 0
            self._refresh_video_list()
            self._render_canvas()
            messagebox.showinfo(
                "Individual mode",
                f"Individual mode allows only 1 video -- kept 1, removed {removed} other(s) from the queue."
            )
        self._update_video_mode_note()

    def _update_video_mode_note(self):
        if self.mode_var.get() == "individual":
            self.video_mode_note.config(text="Individual mode: only 1 video allowed.")
        else:
            self.video_mode_note.config(text="Batch mode: add as many videos as you like.")

    def _probe_video(self, path):
        """Opens a video just long enough to read its fps/frame-count/size,
        used both by on_add_videos (picking new files) and on_open_project
        (re-linking videos from a saved project, which only stores paths).
        Returns None if the file can't be opened."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = (frame_count / fps) if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        cap.release()
        return {"path": path, "fps": fps, "frame_count": frame_count, "duration": duration,
                "width": width, "height": height}

    def on_add_videos(self):
        if self.mode_var.get() == "individual" and self.videos:
            messagebox.showwarning(
                "Individual mode",
                "Individual mode allows only 1 video. Remove the current one first "
                "(click its x), or switch to Batch mode."
            )
            return

        paths = filedialog.askopenfilenames(
            title="Select video(s)",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if not paths:
            return

        if self.mode_var.get() == "individual" and len(paths) > 1:
            messagebox.showinfo("Individual mode", "Individual mode allows only 1 video -- using the first one selected.")
            paths = paths[:1]

        for path in paths:
            entry = self._probe_video(path)
            if entry is None:
                continue
            self.videos.append(entry)

        if self.active_index is None and self.videos:
            self.active_index = 0

        self._refresh_video_list()
        self._render_canvas()

    def on_remove_video(self, index):
        del self.videos[index]
        if not self.videos:
            self.active_index = None
        elif self.active_index is not None and self.active_index >= len(self.videos):
            self.active_index = len(self.videos) - 1
        self._refresh_video_list()
        self._render_canvas()

    def on_select_video_row(self, index):
        # Deliberately does NOT call _reset_calibration_state(): clicking a
        # different video in the queue is just changing which one is shown
        # in the preview/calibration canvas, not a request to throw away
        # the zones/crop/objects/settings already set up. Those should only
        # ever be cleared by an explicit Reset action (on_reset_* below) --
        # otherwise moving through a multi-video queue means redoing the
        # same zone drawing from scratch for every single video.
        self.active_index = index
        self._refresh_video_list()
        self._render_canvas()

    def _update_video_info_label(self):
        """Keeps the WxH / fps / duration line in sync with whichever video
        is active. Harmless no-op on setup screens that don't have the
        label (there aren't any right now, but this stays defensive since
        _refresh_video_list is shared by every mode)."""
        if not hasattr(self, "video_info_label"):
            return
        try:
            if self.active_index is None or not self.videos:
                self.video_info_label.configure(text="No video selected")
                return
            v = self.videos[self.active_index]
        except (IndexError, AttributeError, tk.TclError):
            return
        w = v.get("width") or 0
        h = v.get("height") or 0
        fps = v.get("fps") or 0
        duration = v.get("duration") or 0
        mins, secs = divmod(int(duration), 60)
        size_txt = f"{w}×{h}" if w and h else "size unknown"
        try:
            self.video_info_label.configure(text=f"{size_txt}   {fps:.2f} fps   {mins:d}:{secs:02d}")
        except tk.TclError:
            pass

    def _refresh_video_list(self):
        self._update_video_mode_note()
        self._update_video_info_label()
        for child in self.video_list_inner.winfo_children():
            child.destroy()

        for i, v in enumerate(self.videos):
            row = ttk.Frame(self.video_list_inner)
            row.pack(fill="x", pady=1)
            name = os.path.basename(v["path"])
            label_text = f"{name}   {v['fps']:.2f}fps   {v['duration']:.0f}s"
            fg = "#000000" if i == self.active_index else "#333333"
            font = ("Segoe UI", 8, "bold") if i == self.active_index else ("Segoe UI", 8)
            lbl = tk.Label(row, text=label_text, anchor="w", fg=fg, font=font,
                           bg="#dce6f0" if i == self.active_index else "#fafafa", cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True)
            lbl.bind("<Button-1>", lambda e, i=i: self.on_select_video_row(i))
            ttk.Button(row, text="x", width=2, style="Danger.TButton",
                      command=lambda i=i: self.on_remove_video(i)).pack(side="right")

        if not self.videos:
            ttk.Label(self.video_list_inner, text="(no videos added -- click + above)",
                      foreground="#888888", font=("Segoe UI", 8)).pack(anchor="w", pady=4)

    # ------------------------------------------------------------------
    # Reset handlers
    # ------------------------------------------------------------------

    def _reset_calibration_state(self):
        self.pending_use_crop = None
        self.pending_matrix = None
        self.pending_warp_w = None
        self.pending_warp_h = None
        self.pending_crop_corners = None
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        self.pending_scale_factor = None
        self.pending_scale_unit = None
        self._op = None
        self.active_tool = None
        if hasattr(self, "tool_buttons"):
            for btn in self.tool_buttons.values():
                btn.configure(style="Tool.TButton")
        if hasattr(self, "op_bar"):
            for child in self.op_bar.winfo_children():
                child.destroy()

    def on_reset_video_chamber(self):
        if not messagebox.askyesno("Reset video panel", "Clear the video queue and time/zone/analysis fields?"):
            return
        self.videos = []
        self.active_index = None
        self._remembered_start = ""
        self._remembered_end = ""
        self._remembered_roi_names = ""
        try:
            self.start_entry.delete(0, tk.END)
            self.end_entry.delete(0, tk.END)
            self.roi_names_entry.delete(0, tk.END)
            self.object_names_entry.delete(0, tk.END)
        except (tk.TclError, AttributeError):
            pass
        try:
            self.output_size_entry.set("Same as input")
        except (tk.TclError, AttributeError):
            pass
        try:
            self.interact_var.set(False)
            self.entries_var.set(False)
            self.altern_var.set(False)
            self.all_var.set(False)
        except (tk.TclError, AttributeError):
            pass
        try:
            if hasattr(self, "behavior_rearing_var"): self.behavior_rearing_var.set(True)
            if hasattr(self, "behavior_grooming_var"): self.behavior_grooming_var.set(True)
            if hasattr(self, "behavior_locomotion_var"): self.behavior_locomotion_var.set(False)
            if hasattr(self, "behavior_immobile_var"): self.behavior_immobile_var.set(False)
            if hasattr(self, "behavior_all_var"): self.behavior_all_var.set(False)
        except (tk.TclError, AttributeError):
            pass
        self._refresh_video_list()
        self._reset_calibration_state()
        self._render_canvas()

    def on_reset_calibration_chamber(self):
        if not messagebox.askyesno("Reset preview panel", "Clear the arena crop, mask, zones, objects, and distance calibration?"):
            return
        self._reset_calibration_state()
        self._render_canvas()

    def on_reset_settings_chamber(self):
        if not messagebox.askyesno("Reset settings panel", "Reset all Detection Settings back to defaults?"):
            return
        defaults = [
            (self.bg_samples_entry, "100"), (self.threshold_entry, "25"),
            (self.min_area_entry, "15"), (self.max_area_entry, "5000"),
            (self.max_jump_entry, "100"), (self.window_size_entry, "120"),
            (self.window_weight_entry, "0.5"), (self.real_distance_entry, "30"),
            (self.units_entry, "cm"), (self.preview_samples_entry, "6"),
        ]
        for entry, default in defaults:
            if entry is not None:
                try:
                    entry.delete(0, tk.END)
                    entry.insert(0, default)
                except (tk.TclError, AttributeError):
                    pass
        self.use_window_var.set(False)
        self.use_zone_threshold_var.set(False)
        if hasattr(self, "reject_shadows_var") and self.reject_shadows_var is not None:
            self.reject_shadows_var.set(False)
        if hasattr(self, "loco_thresh_entry") and self.loco_thresh_entry is not None:
            try:
                self.loco_thresh_entry.delete(0, tk.END)
                self.loco_thresh_entry.insert(0, "40")
            except (tk.TclError, AttributeError):
                pass
        if hasattr(self, "rear_thresh_entry") and self.rear_thresh_entry is not None:
            try:
                self.rear_thresh_entry.delete(0, tk.END)
                self.rear_thresh_entry.insert(0, "0.7")
            except (tk.TclError, AttributeError):
                pass
        if hasattr(self, "groom_thresh_entry") and self.groom_thresh_entry is not None:
            try:
                self.groom_thresh_entry.delete(0, tk.END)
                self.groom_thresh_entry.insert(0, "3.0")
            except (tk.TclError, AttributeError):
                pass
        if hasattr(self, "immobile_thresh_entry") and self.immobile_thresh_entry is not None:
            try:
                self.immobile_thresh_entry.delete(0, tk.END)
                self.immobile_thresh_entry.insert(0, "1.0")
            except (tk.TclError, AttributeError):
                pass
        if hasattr(self, "min_bout_entry") and self.min_bout_entry is not None:
            try:
                self.min_bout_entry.delete(0, tk.END)
                self.min_bout_entry.insert(0, "0.3")
            except (tk.TclError, AttributeError):
                pass

    def on_reset_all(self):
        if not messagebox.askyesno("Reset everything",
                                   "Reset ALL back to defaults?\n\nThis clears: videos, calibration, "
                                   "settings, and all remembered values."):
            return

        # THE KEY LINE: this is the ONLY place that clears persistent memory
        self._memory.clear()

        self.videos = []
        self.active_index = None

        self._reset_calibration_state()

        # Rebuild the entire body (will see empty memory → all fields at empty defaults)
        self._build_setup_body()

        try:
            self.status_label.config(text="Idle.")
            self.progress["value"] = 0
            self.progress_label.config(text="")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Header actions: Open/Save Project, Settings, Help
    # ------------------------------------------------------------------

    def on_save_project(self):
        """Saves the video queue + analysis type/mode + the setup fields
        (everything _save_all_to_memory tracks) to a JSON file, so a
        session can be picked back up later with 'Open Project'.
        Deliberately excludes calibration (pending_* crop/zones/objects/
        distance state) -- that's drawn interactively per-video and
        re-serializing it safely is a bigger job than this pass, so for
        now a reopened project starts from a clean calibration, same as
        picking the video(s) fresh."""
        if not self.videos:
            messagebox.showwarning("Nothing to save", "Add at least one video before saving a project.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".btproj",
            filetypes=[("BehavioralTracker project", "*.btproj"), ("All files", "*.*")],
        )
        if not path:
            return

        self._save_all_to_memory(include_calibration=False)
        safe_memory = {k: v for k, v in self._memory.items() if not k.startswith("pending_")}
        data = {
            "app": self._palette["APP_BRAND"],
            "version": self._palette["APP_VERSION"],
            "analysis_type": self.analysis_type,
            "mode": self.mode_var.get(),
            "videos": [{"path": v["path"]} for v in self.videos],
            "memory": safe_memory,
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))
            return
        try:
            self.status_label.config(text=f"Saved project: {os.path.basename(path)}")
        except Exception:
            pass

    def on_open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("BehavioralTracker project", "*.btproj"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Open failed", str(exc))
            return

        missing = []
        loaded_videos = []
        for entry in data.get("videos", []):
            vpath = entry.get("path") if isinstance(entry, dict) else None
            probed = self._probe_video(vpath) if vpath else None
            if probed is None:
                missing.append(vpath or "(unknown path)")
                continue
            loaded_videos.append(probed)

        self.videos = loaded_videos
        self.active_index = 0 if self.videos else None

        self._memory.update(data.get("memory", {}))
        analysis_type = data.get("analysis_type", "standard")
        if analysis_type not in self.analysis_type_buttons:
            analysis_type = "standard"
        mode = data.get("mode", "individual")

        self.mode_var.set(mode)
        self.analysis_type = analysis_type
        for val, btn in self.analysis_type_buttons.items():
            btn.configure(style="Accent.TButton" if val == analysis_type else "Tool.TButton")
        self._reset_calibration_state()
        self._build_setup_body(resave=False)

        if missing:
            messagebox.showwarning(
                "Some videos missing",
                "Project settings loaded, but these video files couldn't be found and were "
                "skipped:\n\n" + "\n".join(missing),
            )
        try:
            self.status_label.config(text=f"Opened project: {os.path.basename(path)}")
        except Exception:
            pass

    def on_open_settings(self):
        messagebox.showinfo(
            "Settings",
            "A dedicated Settings screen is on the way. For now, everything is on the setup "
            "screen itself: Detection Settings (right column) apply to the current Analysis "
            "Type, Save Project keeps a video queue and its fields for later, and Reset All "
            "clears everything back to defaults."
        )

    def on_open_help(self):
        messagebox.showinfo(
            "Quick start",
            "1) Pick an Analysis Type above.\n"
            "2) Add your video(s) with the + button.\n"
            "3) Crop the arena (or click No Crop), then draw zones/objects, or use a Quick "
            "Setup template if your analysis type has one.\n"
            "4) Adjust Detection Settings on the right if the default tracking looks off.\n"
            "5) Click Start Tracking -- Results open automatically when it finishes.\n\n"
            "Open Project / Save Project let you reload a video queue and its setup fields "
            "later (calibration -- crop/zones/objects -- is redrawn per video, not saved)."
        )

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def _get_active_raw_frame(self):
        if self.active_index is None:
            return None
        video = self.videos[self.active_index]
        cap = cv2.VideoCapture(video["path"])
        if not cap.isOpened():
            return None
        try:
            start_time = float(self.start_entry.get() or 0)
        except ValueError:
            start_time = 0
        start_frame = int(max(0, start_time) * (video["fps"] or 30))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None

    def _get_warped_active_frame(self):
        raw = self._get_active_raw_frame()
        if raw is None or self.pending_matrix is None:
            return raw
        return cv2.warpPerspective(raw, self.pending_matrix, (self.pending_warp_w, self.pending_warp_h))

    # ------------------------------------------------------------------
    # Canvas <-> frame coordinate mapping + rendering
    # ------------------------------------------------------------------

    def _bgr_to_photoimage(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        header = f"P6 {w} {h} 255 ".encode()
        data = header + rgb.tobytes()
        return tk.PhotoImage(width=w, height=h, data=data, format="PPM")

    def _load_png_into_canvas(self, canvas, path):
        """Load an image file into a Tk canvas, scaled down to fit (never
        upscaled) -- shared by the Standard Tracking results view switcher
        (Trajectory / Heatmap / Zone Occupancy) so switching views is just
        swapping which file this loads, not three separately-built canvases.
        Keeps self._results_photo alive (Tk drops a PhotoImage with no live
        reference) and clears the canvas first so stale content never shows
        through if the new image fails to load."""
        canvas.delete("all")
        self._results_photo = None
        if not os.path.exists(path):
            return
        frame = cv2.imread(path)
        if frame is None:
            return
        canvas.update_idletasks()
        cw, ch = max(300, canvas.winfo_width()), max(200, canvas.winfo_height())
        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh, 1.0)
        disp = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))))
        self._results_photo = self._bgr_to_photoimage(disp)
        canvas.create_image(0, 0, anchor="nw", image=self._results_photo)

    def _build_preview_canvas(self, center):
        """The Preview/Calibration canvas, with a zoom slider and
        scrollbars -- a small-resolution source video no longer displays at
        a tiny native pixel size with wasted space around it (auto-fit now
        upscales), and the slider lets you zoom in further and scroll
        around for a closer look at any video."""
        MUTED = self._palette["MUTED"]

        zoom_row = ttk.Frame(center)
        zoom_row.pack(fill="x", pady=(4, 0))
        ttk.Label(zoom_row, text="Zoom:", font=("Segoe UI", 8)).pack(side="left")
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale_widget = ttk.Scale(zoom_row, from_=0.25, to=3.0, orient="horizontal",
                                           variable=self.zoom_var, command=lambda v: self._on_zoom_change())
        self.zoom_scale_widget.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.zoom_label = ttk.Label(zoom_row, text="100%", font=("Segoe UI", 8), width=5)
        self.zoom_label.pack(side="left")
        ttk.Button(zoom_row, text="Fit", style="Reset.TButton",
                  command=self._on_zoom_reset).pack(side="left", padx=(6, 0))

        canvas_frame = ttk.Frame(center)
        canvas_frame.pack(fill="both", expand=True, pady=(4, 4))
        v_scroll = ttk.Scrollbar(canvas_frame, orient="vertical")
        h_scroll = ttk.Scrollbar(canvas_frame, orient="horizontal")
        self.canvas = tk.Canvas(canvas_frame, bg="#dddddd", highlightthickness=0,
                                xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        v_scroll.config(command=self.canvas.yview)
        h_scroll.config(command=self.canvas.xview)

        # Fixed-size scrollbars packed FIRST, expanding canvas LAST --
        # packing the other way around lets the expand=True canvas claim
        # the scrollbars' space first, collapsing them to 1px (the same
        # packing-order mistake hit twice before elsewhere in this app).
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.preview_caption = ttk.Label(center, text="", foreground=MUTED, font=("Segoe UI", 8))
        self.preview_caption.pack(fill="x")

    def _on_zoom_change(self):
        self.zoom_label.config(text=f"{int(round(self.zoom_var.get() * 100))}%")
        if self._op is not None:
            self._redraw_current_frame_only()
        else:
            self._render_canvas()

    def _on_zoom_reset(self):
        self.zoom_var.set(1.0)
        self._on_zoom_change()

    def _redraw_current_frame_only(self):
        """Re-fit + redraw the base image and overlays for the operation in
        progress, without restarting it -- used when the zoom slider moves
        mid-operation so points already placed stay exactly where they are
        (in frame coordinates) even though the on-screen scale just
        changed."""
        op = self._op
        if op is None:
            return
        base_frame = op.get("_base_frame")
        if base_frame is None:
            return
        photo, scale, off_x, off_y = self._canvas_fit(base_frame)
        self._canvas_photo = photo
        self._canvas_scale, self._canvas_off_x, self._canvas_off_y = scale, off_x, off_y
        self.canvas.delete("base")
        self.canvas.create_image(off_x, off_y, anchor="nw", image=photo, tags="base")
        self._redraw_op()

    def _canvas_fit(self, frame_bgr):
        self.canvas.update_idletasks()
        cw = max(300, self.canvas.winfo_width())
        ch = max(200, self.canvas.winfo_height())
        fh, fw = frame_bgr.shape[:2]
        # Fit-to-canvas as before, but no longer capped at native (1.0x)
        # resolution -- a small source video (e.g. 640x360) used to display
        # at that tiny native pixel size with empty space around it even in
        # a huge window. Auto-fit now upscales too, capped at 3x so it
        # doesn't get uselessly blurry by default; the zoom slider can go
        # well past that deliberately.
        fit_scale = min(cw / fw, ch / fh)
        fit_scale = min(fit_scale, 3.0) if fit_scale > 0 else 1.0
        zoom_factor = self.zoom_var.get() if getattr(self, "zoom_var", None) is not None else 1.0
        scale = max(0.05, fit_scale * zoom_factor)
        disp_w, disp_h = max(1, int(fw * scale)), max(1, int(fh * scale))
        disp = frame_bgr if (disp_w, disp_h) == (fw, fh) else cv2.resize(frame_bgr, (disp_w, disp_h))
        off_x = max(0, (cw - disp_w) // 2)
        off_y = max(0, (ch - disp_h) // 2)
        photo = self._bgr_to_photoimage(disp)
        self.canvas.configure(scrollregion=(0, 0, max(cw, disp_w), max(ch, disp_h)))
        return photo, scale, off_x, off_y

    def _to_canvas_xy(self, fx, fy):
        return fx * self._canvas_scale + self._canvas_off_x, fy * self._canvas_scale + self._canvas_off_y

    def _to_frame_xy(self, cx, cy):
        return (cx - self._canvas_off_x) / self._canvas_scale, (cy - self._canvas_off_y) / self._canvas_scale

    def _render_canvas(self):
        # Auto-fit whenever a DIFFERENT video becomes active (newly added,
        # or clicked in the list) -- reset the zoom slider back to its
        # fit-to-canvas baseline (100% = _canvas_fit's own fit_scale, see
        # there) rather than carrying over whatever zoom the previous
        # video was left at. Re-renders of the SAME still-active video
        # (after drawing a zone, changing a setting, etc.) leave the zoom
        # exactly as the person set it.
        active_path = (self.videos[self.active_index]["path"]
                       if self.active_index is not None and self.active_index < len(self.videos) else None)
        if active_path != getattr(self, "_zoom_fit_for_path", None):
            self._zoom_fit_for_path = active_path
            if getattr(self, "zoom_var", None) is not None:
                self.zoom_var.set(1.0)
                self.zoom_label.config(text="100%")

        frame = self._get_warped_active_frame() if self.pending_matrix is not None else self._get_active_raw_frame()
        self.canvas.delete("all")

        if frame is None:
            self.canvas.update_idletasks()
            cw = max(300, self.canvas.winfo_width())
            ch = max(200, self.canvas.winfo_height())
            self.canvas.create_text(cw // 2, ch // 2, text="Add a video to begin",
                                    fill="#777777", font=("Segoe UI", 12))
            self.preview_caption.config(text="")
            return

        photo, scale, off_x, off_y = self._canvas_fit(frame)
        self._canvas_photo = photo
        self._canvas_scale, self._canvas_off_x, self._canvas_off_y = scale, off_x, off_y
        self.canvas.create_image(off_x, off_y, anchor="nw", image=photo, tags="base")

        if self._op is not None:
            self._redraw_op()
        else:
            self._redraw_static_overlays()
            parts = []
            parts.append("Arena: cropped" if self.pending_use_crop else
                         ("Arena: full frame (no crop)" if self.pending_use_crop is False else "Arena: not set"))
            if self.pending_mask_points:
                parts.append(f"Masked shapes: {len(self.pending_mask_points)}")
            parts.append(f"Zones: {', '.join(self.pending_roi_points.keys())}" if self.pending_roi_points
                         else "Zones: not drawn")
            if self.pending_object_points:
                parts.append(f"Objects: {', '.join(self.pending_object_points.keys())}")
            self.preview_caption.config(text="   |   ".join(parts))

    def _redraw_static_overlays(self):
        if self.pending_mask_points:
            self._draw_polygon_set(self.pending_mask_points, color="black", dashed=True)
        if self.pending_roi_points:
            self._draw_named_polygon_set(self.pending_roi_points, color="black")
        if self.pending_object_points:
            self._draw_named_polygon_set(self.pending_object_points, color="#d2691e")

    def _draw_polygon_set(self, shapes, color="black", dashed=False, point_radius=3, active_shape=None):
        dash = (4, 2) if dashed else None
        for i, pts in enumerate(shapes):
            width = 3 if i == active_shape else 2
            if len(pts) >= 2:
                flat = []
                for pt in pts:
                    flat.extend(self._to_canvas_xy(*pt))
                self.canvas.create_line(*flat, fill=color, width=width, dash=dash, tags="overlay")
                if len(pts) >= 3:
                    x0, y0 = self._to_canvas_xy(*pts[-1])
                    x1, y1 = self._to_canvas_xy(*pts[0])
                    self.canvas.create_line(x0, y0, x1, y1, fill=color, width=width, dash=dash, tags="overlay")
            for pt in pts:
                cx, cy = self._to_canvas_xy(*pt)
                self.canvas.create_oval(cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius,
                                        fill=color, outline="white", tags="overlay")

    def _draw_named_polygon_set(self, regions, color="black", active_region=None, point_radius=3):
        for name, pts in regions.items():
            width = 3 if name == active_region else 2
            if len(pts) >= 2:
                flat = []
                for pt in pts:
                    flat.extend(self._to_canvas_xy(*pt))
                self.canvas.create_line(*flat, fill=color, width=width, tags="overlay")
                if len(pts) >= 3:
                    x0, y0 = self._to_canvas_xy(*pts[-1])
                    x1, y1 = self._to_canvas_xy(*pts[0])
                    self.canvas.create_line(x0, y0, x1, y1, fill=color, width=width, tags="overlay")
            for pt in pts:
                cx, cy = self._to_canvas_xy(*pt)
                self.canvas.create_oval(cx - point_radius, cy - point_radius, cx + point_radius, cy + point_radius,
                                        fill=color, outline="white", tags="overlay")
            if pts:
                mx = sum(p[0] for p in pts) / len(pts)
                my = sum(p[1] for p in pts) / len(pts)
                lx, ly = self._to_canvas_xy(mx, my)
                self.canvas.create_text(lx, ly, text=name, fill=color, font=("Segoe UI", 9, "bold"), tags="overlay")

    # ------------------------------------------------------------------
    # Embedded operations: Crop / Mask / Zones / Objects / Distance
    # ------------------------------------------------------------------

    def _set_active_tool(self, key):
        self.active_tool = key
        for k, btn in self.tool_buttons.items():
            btn.configure(style="ToolActive.TButton" if k == key else "Tool.TButton")

    def _build_op_bar(self, kind):
        for child in self.op_bar.winfo_children():
            child.destroy()

        self.op_instructions = ttk.Label(self.op_bar, text="", font=("Segoe UI", 9))
        self.op_instructions.pack(side="left")

        if kind in ("zones", "objects"):
            self.op_region_row = ttk.Frame(self.op_bar)
            self.op_region_row.pack(side="left", padx=(12, 0))
            self._rebuild_zone_region_buttons()

        if kind == "mask":
            ttk.Button(self.op_bar, text="New Shape", command=self._op_new_shape).pack(side="right", padx=(4, 0))

        ttk.Button(self.op_bar, text="Cancel", command=self._cancel_op).pack(side="right", padx=(4, 0))
        ttk.Button(self.op_bar, text="Finish", style="Accent.TButton", command=self._finish_op).pack(
            side="right", padx=(4, 0))

    def _rebuild_zone_region_buttons(self):
        for child in self.op_region_row.winfo_children():
            child.destroy()
        for name in self._op["regions"].keys():
            is_active = name == self._op["active_region"]
            b = ttk.Button(self.op_region_row, text=name,
                          style="ToolActive.TButton" if is_active else "Tool.TButton",
                          command=lambda n=name: self._op_set_active_region(n))
            b.pack(side="left", padx=(0, 4))

    def _op_set_active_region(self, name):
        self._op["active_region"] = name
        self._rebuild_zone_region_buttons()
        self._update_op_instructions()

    def _op_new_shape(self):
        if self._op and self._op["kind"] == "mask":
            self._op["shapes"].append([])
            self._update_op_instructions()

    def _update_op_instructions(self):
        op = self._op
        if op is None:
            return
        if op["kind"] == "crop":
            n = len(op["shapes"][0])
            text = f"Click the 4 arena corners ({n}/4 placed). Drag a corner to adjust it."
        elif op["kind"] == "mask":
            n_shapes = len(op["shapes"])
            n_pts = len(op["shapes"][-1])
            text = (f"Shape {n_shapes} ({n_pts} pts so far). Click to add points, "
                    "'New Shape' to start another masked region.")
        elif op["kind"] in ("zones", "objects"):
            name = op["active_region"]
            n = len(op["regions"][name])
            noun = "zone" if op["kind"] == "zones" else "object"
            text = f"Drawing {noun} '{name}' ({n} pts, need 3+ -- trace its outline). Click a name above to switch."
        elif op["kind"] == "distance":
            text = f"Click 2 points of known real-world distance ({len(op['points'])}/2 placed)."
        elif op["kind"] == "template_zones":
            text = (f"{len(op['regions'])} zone(s) generated. Click a zone on the canvas to confirm or "
                    "rename it (e.g. swap which arm is Open/Closed), then Finish.")
        else:
            text = ""
        self.op_instructions.config(text=text)

    def _start_op(self, kind):
        if kind == "crop":
            base_frame = self._get_active_raw_frame()
            if base_frame is None:
                messagebox.showerror("No video", "Add and select a video first.")
                return
        else:
            if self.pending_matrix is None:
                messagebox.showerror("Arena needed", "Use 'Crop Arena' or 'No Crop' first.")
                return
            base_frame = self._get_warped_active_frame()
            if base_frame is None:
                messagebox.showerror("No video", "Add and select a video first.")
                return

        names = []
        if kind == "zones":
            names = [n.strip() for n in self.roi_names_entry.get().split(",") if n.strip()]
            if not names:
                messagebox.showerror("Zone names needed", "Enter zone names first (e.g. Light,Dark).")
                return
        if kind == "objects":
            names = [n.strip() for n in self.object_names_entry.get().split(",") if n.strip()]
            if not names:
                messagebox.showerror("Object names needed", "Enter object names first in the left panel.")
                return

        self._set_active_tool(kind)

        photo, scale, off_x, off_y = self._canvas_fit(base_frame)
        self._canvas_photo = photo
        self._canvas_scale, self._canvas_off_x, self._canvas_off_y = scale, off_x, off_y
        self.canvas.delete("all")
        self.canvas.create_image(off_x, off_y, anchor="nw", image=photo, tags="base")

        op = {"kind": kind, "drag": None, "_base_frame": base_frame}

        if kind == "crop":
            saved = getattr(self, "pending_crop_corners", None)
            if saved and len(saved) == 4:
                op["shapes"] = [list(saved)]
            else:
                op["shapes"] = [[]]
        elif kind == "mask":
            op["shapes"] = [list(p) for p in self.pending_mask_points] if self.pending_mask_points else [[]]
        elif kind in ("zones", "objects"):
            existing = self.pending_roi_points if kind == "zones" else self.pending_object_points
            op["regions"] = {n: list(existing.get(n, [])) for n in names}
            op["active_region"] = names[0]
        elif kind == "distance":
            op["points"] = []

        self._op = op
        self._build_op_bar(kind)
        self._update_op_instructions()
        self._redraw_op()

    def _cancel_op(self):
        self._op = None
        for child in self.op_bar.winfo_children():
            child.destroy()
        for k, btn in self.tool_buttons.items():
            btn.configure(style="Tool.TButton")
        self.active_tool = None
        self._render_canvas()

    def _finish_op(self):
        op = self._op
        if op is None:
            return

        if op["kind"] == "crop":
            if len(op["shapes"][0]) != 4:
                messagebox.showerror("Not done", "Click all 4 corners first.")
                return
            matrix, w, h = compute_perspective_transform(op["shapes"][0])
            self.pending_crop_corners = list(op["shapes"][0])
            self.pending_use_crop = True
            self.pending_matrix, self.pending_warp_w, self.pending_warp_h = matrix, w, h
            self.pending_roi_points = {}
            self.pending_object_points = {}
            self.pending_mask_points = []

        elif op["kind"] == "mask":
            self.pending_mask_points = [s for s in op["shapes"] if len(s) >= 3]

        elif op["kind"] in ("zones", "objects"):
            incomplete = [n for n, pts in op["regions"].items() if len(pts) < 3]
            if incomplete:
                noun = "zones" if op["kind"] == "zones" else "objects"
                messagebox.showerror("Not done", f"These {noun} still need 3+ points: {', '.join(incomplete)}")
                return
            if op["kind"] == "zones":
                self.pending_roi_points = dict(op["regions"])
            else:
                self.pending_object_points = dict(op["regions"])

        elif op["kind"] == "distance":
            if len(op["points"]) != 2:
                messagebox.showerror("Not done", "Click 2 points first.")
                return
            px_distance = point_distance(op["points"][0], op["points"][1])
            try:
                true_distance = float(self.real_distance_entry.get())
            except ValueError:
                messagebox.showerror("Invalid distance", "Enter a valid real-world distance number first.")
                return
            if px_distance > 0:
                self.pending_scale_factor = true_distance / px_distance
                self.pending_scale_unit = self.units_entry.get().strip() or "cm"

        elif op["kind"] == "template_zones":
            self.pending_roi_points = dict(op["regions"])
            # roi_names is read from this text field elsewhere (_build_setup,
            # arm entries/alternation, CSV column naming) rather than from
            # pending_roi_points directly -- keep it in sync with whatever
            # the researcher ended up naming each generated zone.
            self.roi_names_entry.delete(0, tk.END)
            self.roi_names_entry.insert(0, ", ".join(op["regions"].keys()))

        self._op = None
        for child in self.op_bar.winfo_children():
            child.destroy()
        self._render_canvas()

    # ------------------------------------------------------------------
    # Maze protocol templates: geometry-driven zone generation, then
    # click-on-canvas to confirm/rename each generated zone (rather than
    # freehand-tracing every arm by hand).
    # ------------------------------------------------------------------

    def _open_maze_template_dialog(self):
        if self.pending_matrix is None:
            messagebox.showerror("Arena needed", "Use 'Crop Arena' or 'No Crop' first.")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Maze Template")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text="Template:", font=("Segoe UI", 9, "bold")).grid(
            row=0, column=0, sticky="w", padx=10, pady=(10, 2))
        key_list = list(MAZE_TEMPLATES.keys())
        label_list = [MAZE_TEMPLATES[k]["label"] for k in key_list]
        combo_var = tk.StringVar(value=label_list[0])
        combo = ttk.Combobox(dialog, textvariable=combo_var, values=label_list, state="readonly", width=28)
        combo.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 2))

        param_frame = ttk.Frame(dialog, padding=(10, 6))
        param_frame.grid(row=1, column=0, columnspan=2, sticky="w")
        entries = {}

        def current_key():
            idx = label_list.index(combo_var.get())
            return key_list[idx]

        def rebuild_params():
            for child in param_frame.winfo_children():
                child.destroy()
            entries.clear()
            key = current_key()
            unit_suffix = f" ({self.pending_scale_unit})" if getattr(self, "pending_scale_factor", None) else " (px)"
            for r, (pkey, label_text, default, kind) in enumerate(MAZE_TEMPLATES[key]["params"]):
                suffix = unit_suffix if kind == "length" else (" (deg)" if kind == "angle" else "")
                ttk.Label(param_frame, text=label_text + suffix, font=("Segoe UI", 8)).grid(
                    row=r, column=0, sticky="w", pady=2)
                e = ttk.Entry(param_frame, width=10)
                e.insert(0, str(default))
                e.grid(row=r, column=1, sticky="w", padx=(6, 0), pady=2)
                entries[pkey] = (e, kind)

        combo.bind("<<ComboboxSelected>>", lambda ev: rebuild_params())
        rebuild_params()

        btn_row = ttk.Frame(dialog, padding=(10, 10))
        btn_row.grid(row=2, column=0, columnspan=2, sticky="e")
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="Generate Zones", style="Accent.TButton",
                  command=lambda: self._maze_dialog_generate(dialog, current_key, entries)).pack(side="right")

        dialog.wait_window()

    def _maze_dialog_generate(self, dialog, current_key, entries):
        key = current_key()
        scale_factor = getattr(self, "pending_scale_factor", None)
        params = {}
        try:
            for pkey, (entry, kind) in entries.items():
                value = float(entry.get())
                if kind == "length" and scale_factor:
                    value = value / scale_factor  # real-world units -> px
                params[pkey] = value
        except ValueError:
            messagebox.showerror("Invalid value", "Every field needs a plain number.")
            return

        try:
            zones = MAZE_TEMPLATES[key]["generate"](self.pending_warp_w, self.pending_warp_h, params)
        except Exception as exc:
            messagebox.showerror("Could not generate zones", str(exc))
            return

        dialog.destroy()
        self._start_template_zone_op(key, zones)

    def _start_template_zone_op(self, template_key, zones):
        base_frame = self._get_warped_active_frame()
        if base_frame is None:
            messagebox.showerror("No video", "Add and select a video first.")
            return

        self._set_active_tool("maze")

        photo, scale, off_x, off_y = self._canvas_fit(base_frame)
        self._canvas_photo = photo
        self._canvas_scale, self._canvas_off_x, self._canvas_off_y = scale, off_x, off_y
        self.canvas.delete("all")
        self.canvas.create_image(off_x, off_y, anchor="nw", image=photo, tags="base")

        # Suggested labels from the template can collide with each other in
        # freak cases (e.g. a researcher re-running the dialog after already
        # renaming one zone to match another template's name) -- de-dupe by
        # appending a counter rather than silently dropping a zone.
        regions = {}
        for label, pts in zones:
            name = label
            n = 2
            while name in regions:
                name = f"{label} ({n})"
                n += 1
            regions[name] = [tuple(p) for p in pts]

        op = {"kind": "template_zones", "drag": None, "_base_frame": base_frame,
              "regions": regions, "template_key": template_key}
        self._op = op
        self._build_op_bar("template_zones")
        self._update_op_instructions()
        self._redraw_op()

    def _quick_setup_template(self, key):
        """One-click version of the Maze Template dialog for the Quick Setup
        grid: generates zones straight from the template's own default
        sizes, skipping the parameter dialog. Reuses the exact same
        generate()/_start_template_zone_op() pipeline as the full dialog
        above, so it carries the same behavior and the same requirement
        that the arena already be cropped/set."""
        if self.pending_matrix is None:
            messagebox.showerror("Arena needed", "Use 'Crop Arena' or 'No Crop' first.")
            return
        try:
            params = {pkey: default for pkey, _label, default, _kind in MAZE_TEMPLATES[key]["params"]}
            zones = MAZE_TEMPLATES[key]["generate"](self.pending_warp_w, self.pending_warp_h, params)
        except Exception as exc:
            messagebox.showerror("Could not generate zones", str(exc))
            return
        self._start_template_zone_op(key, zones)

    def _point_in_polygon(self, pt, polygon):
        if len(polygon) < 3:
            return False
        contour = np.array(polygon, dtype=np.float32)
        return cv2.pointPolygonTest(contour, (float(pt[0]), float(pt[1])), False) >= 0

    def _prompt_zone_label(self, op, current_name):
        """Popup shown after clicking a template-generated zone on the
        canvas: confirm the geometry's guess, pick a different one of the
        template's suggested names (e.g. swap which arm is actually
        "Open"), or type a custom name -- the click-to-select flow the
        maze templates use instead of freehand zone drawing."""
        template_key = op.get("template_key")
        suggestions = [label for label, _ in MAZE_TEMPLATES[template_key]["generate"](
            self.pending_warp_w, self.pending_warp_h,
            {p[0]: p[2] for p in MAZE_TEMPLATES[template_key]["params"]})] if template_key else []

        dialog = tk.Toplevel(self.root)
        dialog.title("Name this zone")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        ttk.Label(dialog, text=f"What is this zone? (currently \"{current_name}\")",
                  font=("Segoe UI", 9), wraplength=260, justify="left").pack(padx=10, pady=(10, 6), anchor="w")
        name_var = tk.StringVar(value=current_name)
        combo = ttk.Combobox(dialog, textvariable=name_var, values=suggestions, width=26)
        combo.pack(padx=10, pady=(0, 10), anchor="w")

        result = {"name": None}

        def confirm():
            new_name = name_var.get().strip()
            if not new_name:
                messagebox.showerror("Name needed", "Enter or pick a zone name.")
                return
            if new_name != current_name and new_name in op["regions"]:
                messagebox.showerror("Name in use", f"'{new_name}' is already used by another zone.")
                return
            result["name"] = new_name
            dialog.destroy()

        btn_row = ttk.Frame(dialog, padding=(10, 0, 10, 10))
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="right", padx=(6, 0))
        ttk.Button(btn_row, text="OK", style="Accent.TButton", command=confirm).pack(side="right")

        dialog.wait_window()
        return result["name"]

    # -- canvas mouse events --

    def _find_nearby_point(self, points, fx, fy, hit_radius_px=10):
        hit_radius_frame = hit_radius_px / max(self._canvas_scale, 1e-6)
        best = None
        best_d = hit_radius_frame
        for i, (px, py) in enumerate(points):
            d = math.hypot(px - fx, py - fy)
            if d <= best_d:
                best_d = d
                best = i
        return best

    def _on_canvas_press(self, event):
        op = self._op
        if op is None:
            return
        fx, fy = self._to_frame_xy(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

        if op["kind"] == "crop":
            idx = self._find_nearby_point(op["shapes"][0], fx, fy)
            if idx is not None:
                op["drag"] = ("shape", 0, idx)
            elif len(op["shapes"][0]) < 4:
                op["shapes"][0].append((fx, fy))

        elif op["kind"] == "mask":
            for si, shape in enumerate(op["shapes"]):
                idx = self._find_nearby_point(shape, fx, fy)
                if idx is not None:
                    op["drag"] = ("shape", si, idx)
                    break
            else:
                op["shapes"][-1].append((fx, fy))

        elif op["kind"] in ("zones", "objects"):
            # Only hit-test the ACTIVE region's own points, not every region --
            # adjacent regions sharing a border (e.g. Light/Dark touching at
            # x=100) can have points a click or two apart, which would
            # otherwise grab a point from the wrong region entirely.
            active_pts = op["regions"][op["active_region"]]
            idx = self._find_nearby_point(active_pts, fx, fy)
            if idx is not None:
                op["drag"] = ("region", op["active_region"], idx)
            else:
                op["regions"][op["active_region"]].append((fx, fy))

        elif op["kind"] == "distance":
            idx = self._find_nearby_point(op["points"], fx, fy)
            if idx is not None:
                op["drag"] = ("distance", idx)
            elif len(op["points"]) < 2:
                op["points"].append((fx, fy))

        elif op["kind"] == "template_zones":
            # No dragging/point-editing here -- clicking INSIDE a generated
            # zone's outline opens the confirm/rename popup for that one
            # zone (the "select instead of draw" flow), rather than adding
            # or moving a point.
            clicked_name = None
            for name, pts in op["regions"].items():
                if self._point_in_polygon((fx, fy), pts):
                    clicked_name = name
                    break
            if clicked_name is not None:
                new_name = self._prompt_zone_label(op, clicked_name)
                if new_name and new_name != clicked_name:
                    op["regions"][new_name] = op["regions"].pop(clicked_name)

        self._update_op_instructions()
        self._redraw_op()

    def _on_canvas_drag(self, event):
        op = self._op
        if op is None:
            return
        fx, fy = self._to_frame_xy(self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

        if op.get("drag"):
            kind = op["drag"][0]
            if kind == "shape":
                _, si, idx = op["drag"]
                op["shapes"][si][idx] = (fx, fy)
            elif kind == "region":
                _, name, idx = op["drag"]
                op["regions"][name][idx] = (fx, fy)
            elif kind == "distance":
                _, idx = op["drag"]
                op["points"][idx] = (fx, fy)
            self._redraw_op()

    def _on_canvas_release(self, event):
        op = self._op
        if op is None:
            return

        if op.get("drag"):
            op["drag"] = None
            return

    def _redraw_op(self):
        self.canvas.delete("overlay")
        op = self._op
        if op is None:
            return

        if op["kind"] == "crop":
            self._draw_polygon_set(op["shapes"], color="black")
        elif op["kind"] == "mask":
            self._draw_polygon_set(op["shapes"], color="black", dashed=True,
                                   active_shape=len(op["shapes"]) - 1)
        elif op["kind"] == "zones":
            self._draw_named_polygon_set(op["regions"], color="black", active_region=op["active_region"])
        elif op["kind"] == "objects":
            self._draw_named_polygon_set(op["regions"], color="#d2691e", active_region=op["active_region"])
        elif op["kind"] == "template_zones":
            self._draw_named_polygon_set(op["regions"], color="#0077cc")
        elif op["kind"] == "distance":
            for pt in op["points"]:
                cx, cy = self._to_canvas_xy(*pt)
                self.canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="magenta", tags="overlay")
            if len(op["points"]) == 2:
                x0, y0 = self._to_canvas_xy(*op["points"][0])
                x1, y1 = self._to_canvas_xy(*op["points"][1])
                self.canvas.create_line(x0, y0, x1, y1, fill="magenta", width=2, tags="overlay")
                px_d = point_distance(op["points"][0], op["points"][1])
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                self.canvas.create_text(mx, my - 10, text=f"{px_d:.1f} px", fill="magenta",
                                        font=("Segoe UI", 9, "bold"), tags="overlay")

    def on_tool_no_crop(self):
        raw = self._get_active_raw_frame()
        if raw is None:
            messagebox.showerror("No video", "Add and select a video first.")
            return
        self._set_active_tool("nocrop")
        h, w = raw.shape[:2]
        self.pending_crop_corners = None
        self.pending_use_crop = False
        self.pending_matrix, self.pending_warp_w, self.pending_warp_h = identity_transform(w, h)
        self.pending_roi_points = {}
        self.pending_object_points = {}
        self.pending_mask_points = []
        self._render_canvas()

    # ------------------------------------------------------------------
    # Analysis checkboxes / animal count
    # ------------------------------------------------------------------

    def on_all_toggle(self):
        value = self.all_var.get()
        for var in (self.interact_var, self.entries_var, self.altern_var):
            var.set(value)

    def on_individual_toggle(self):
        if not (self.interact_var.get() and self.entries_var.get() and self.altern_var.get()):
            self.all_var.set(False)

    def on_num_animals(self, n):
        if self.analysis_type == "standard" and n != 1:
            messagebox.showinfo(
                "Switch analysis type",
                f"Standard Tracking only handles 1 animal (it needs zones/objects, which the "
                f"multi-animal engine doesn't support yet). Switch to 'Multi-Mouse Tracking' or "
                f"'Behavior Classification' above to track {n} animals."
            )
            return
        self.num_animals = n
        for k, btn in self.num_animals_buttons.items():
            btn.configure(style="Accent.TButton" if k == n else "Tool.TButton")

    # ------------------------------------------------------------------
    # Progress + tracking
    # ------------------------------------------------------------------

    def on_progress(self, fraction):
        self.progress["value"] = max(0, min(100, fraction * 100))
        self.progress_label.config(text=f"{fraction * 100:.0f}%")
        self.root.update_idletasks()

    def _build_setup(self, video_path):
        try:
            start_raw = self.start_entry.get().strip()
            end_raw = self.end_entry.get().strip()
            if not start_raw or not end_raw:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                full_duration = total_frames / fps if fps else 0.0
            start_time = float(start_raw) if start_raw else 0.0
            end_time = float(end_raw) if end_raw else full_duration

            setup = {
                "video_path": video_path,
                "use_crop": self.pending_use_crop,
                "matrix": self.pending_matrix,
                "warp_w": self.pending_warp_w,
                "warp_h": self.pending_warp_h,
                "arena": (0, 0, self.pending_warp_w, self.pending_warp_h),
                "roi_names": [n.strip() for n in self.roi_names_entry.get().split(",") if n.strip()],
                "roi_points": self.pending_roi_points,
                "mask_points": self.pending_mask_points,
                "object_names": [n.strip() for n in self.object_names_entry.get().split(",") if n.strip()] \
                    if self.interact_var.get() else [],
                "object_points": self.pending_object_points if self.interact_var.get() else {},
                "interaction_margin_px": float(getattr(self, "interaction_margin_entry", None).get())
                    if getattr(self, "interaction_margin_entry", None) is not None else 0.0,
                "behavior_names": [],
                "background_samples": int(float(self.bg_samples_entry.get())),
                "threshold": float(self.threshold_entry.get()),
                "min_area": float(self.min_area_entry.get()),
                "max_area": float(self.max_area_entry.get()),
                "max_jump": float(self.max_jump_entry.get()),
                "use_zone_threshold": self.use_zone_threshold_var.get(),
                "reject_shadows": getattr(self, "reject_shadows_var", tk.BooleanVar(value=False)).get(),
                "use_window": self.use_window_var.get(),
                "window_size": float(self.window_size_entry.get()) if getattr(self, "window_size_entry", None) is not None else 120.0,
                "window_weight": float(self.window_weight_entry.get()) if getattr(self, "window_weight_entry", None) is not None else 0.5,
                "scale_factor": getattr(self, "pending_scale_factor", None),
                "scale_unit": getattr(self, "pending_scale_unit", None),
                "preview_samples": int(float(self.preview_samples_entry.get())),
                "color_mode": self.color_mode_var.get(),
                "start_time": start_time,
                "end_time": end_time,
                "output_dir_override": None,
                "compute_arm_entries": self.entries_var.get(),
                "compute_alternation": self.altern_var.get(),
            }
        except ValueError as exc:
            messagebox.showerror("Invalid setting", f"Check the Detection Settings values -- {exc}")
            return None

        if setup["color_mode"] == "auto":
            setup["color_mode"] = self._resolve_auto_color_mode(
                video_path, num_animals=1, min_area=setup["min_area"], max_area=setup["max_area"],
                diff_threshold=setup["threshold"], n_background_samples=setup["background_samples"],
            )
        return setup

    def _resolve_auto_color_mode(self, video_path, num_animals, min_area, max_area,
                                  diff_threshold, n_background_samples):
        """Turn the "Auto" color-mode choice into a concrete "gray"/"rgb"
        decision for this specific video, via tracking.two_mouse.choose_color_mode.
        Called once per video (not per frame) -- a couple of seconds of extra
        probing up front, in exchange for not needing the user to guess and
        manually flip Grayscale/RGB themselves. Falls back to "gray" (the
        safe, cheap default) if the probe itself errors out for any reason,
        rather than letting that abort the whole run."""
        self.status_label.config(text="Checking best color mode for this video...")
        self.root.update_idletasks()
        try:
            resolved, _diagnostics = choose_color_mode(
                video_path, num_animals=num_animals, min_area=min_area, max_area=max_area,
                diff_threshold=diff_threshold,
                n_background_samples=min(int(n_background_samples), 20),
            )
        except Exception as exc:
            print(f"[warn] Auto color-mode selection failed ({exc}); defaulting to grayscale.")
            resolved = "gray"
        self.status_label.config(text=f"Auto color mode picked: {resolved.upper()}")
        self.root.update_idletasks()
        return resolved

    def on_start(self):
        if not self.videos:
            messagebox.showerror("No video", "Add a video first.")
            return
        if self.active_index is None:
            messagebox.showerror("No video selected", "Click a video in the list to select it.")
            return

        try:
            min_area = int(float(getattr(self, "min_area_entry", None).get())) if getattr(self, "min_area_entry", None) else 15
            max_area = int(float(getattr(self, "max_area_entry", None).get())) if getattr(self, "max_area_entry", None) else 5000
            if min_area >= max_area:
                messagebox.showerror(
                    "Invalid area settings",
                    f"Min mouse area ({min_area} px) must be LESS THAN Max object area ({max_area} px)."
                )
                return
        except (ValueError, tk.TclError, AttributeError):
            pass

        try:
            start_raw = getattr(self, "start_entry", None).get() if getattr(self, "start_entry", None) else ""
            end_raw = getattr(self, "end_entry", None).get() if getattr(self, "end_entry", None) else ""
            if start_raw.strip() and end_raw.strip():
                start_s = float(start_raw)
                end_s = float(end_raw)
                if start_s >= end_s:
                    messagebox.showerror(
                        "Invalid time window",
                        f"Start time ({start_s}s) must be LESS THAN end time ({end_s}s)."
                    )
                    return
                if start_s < 0:
                    messagebox.showerror("Invalid time window", "Start time cannot be negative.")
                    return
        except (ValueError, tk.TclError, AttributeError):
            pass

        if self.analysis_type == "standard":
            self._run_standard_flow()
        elif self.analysis_type == "multi_mouse":
            self._run_multi_mouse_flow()
        elif self.analysis_type == "behavior":
            self._run_behavior_flow()

    def _run_standard_flow(self):
        if self.pending_matrix is None:
            messagebox.showerror("Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return
        if not self.pending_roi_points:
            messagebox.showerror("Zones not set", "Use 'Draw Zones' first.")
            return

        active_path = self.videos[self.active_index]["path"]
        base_setup = self._build_setup(active_path)
        if base_setup is None:
            return

        self.progress["value"] = 0
        self.status_label.config(text="Tracking...")
        self.root.update_idletasks()

        if self.mode_var.get() == "individual":
            try:
                summary = process_single_video(active_path, base_setup, show_display=True,
                                               progress_callback=self.on_progress)
            except SystemExit as exc:
                self.status_label.config(text=f"Stopped: {exc}")
                cv2.destroyAllWindows()
                return
            except Exception as exc:
                self.status_label.config(text=f"Error: {type(exc).__name__}")
                cv2.destroyAllWindows()
                messagebox.showerror("Tracking Error", f"An error occurred during tracking:\n\n{type(exc).__name__}: {exc}")
                return
            self.progress["value"] = 100
            self.status_label.config(text="Done.")
            try:
                self._show_results_standard(summary)
            except Exception as exc:
                messagebox.showerror("Results Error", f"Could not display results:\n\n{type(exc).__name__}: {exc}")
            return

        same_camera = ask_yes_no(
            "Camera position",
            f"Found {len(self.videos)} videos.\n\n"
            "Was the camera in EXACTLY the same position for all of them?\n\n"
            "Yes = reuse this calibration for every video (fast)\n"
            "No = re-select the arena/zones/objects for each one"
        )

        summaries = []
        errors = []
        for i, v in enumerate(self.videos):
            print(f"\n[{i + 1}/{len(self.videos)}] Processing: {v['path']}")
            self.status_label.config(
                text=f"Batch: video {i + 1}/{len(self.videos)} -- {os.path.basename(v['path'])}")
            self.progress["value"] = 0
            self.progress_label.config(text="")
            self.root.update_idletasks()
            try:
                if i == 0 or same_camera:
                    setup = dict(base_setup)
                    setup["video_path"] = v["path"]
                else:
                    try:
                        setup = recalibrate_spatial_only(v["path"], base_setup)
                    except SystemExit:
                        print(f"  Skipped: calibration cancelled for this video")
                        errors.append((os.path.basename(v["path"]), "Calibration cancelled"))
                        continue
                # show_display=False here on purpose -- with it True, every
                # video in the batch would pop up its own "press ENTER to
                # confirm the background" step AND up to
                # preview_samples-many "press any key to advance" detection-
                # preview steps (see process_single_video/run_detection_preview
                # in tracking/location.py), each one BLOCKING until someone
                # is sitting there to press a key. For a batch of N videos
                # that means the run stalls, looking hung, N*(1+preview
                # samples) separate times unless babysat the whole way
                # through -- exactly what "batch process" should not require.
                # Individual (single-video) mode keeps show_display=True
                # above, since there a person is actively watching that one
                # run and the sanity-check preview is genuinely useful.
                summary = process_single_video(v["path"], setup, show_display=False,
                                               progress_callback=self.on_progress)
                summaries.append(summary)
            except SystemExit as exc:
                print(f"  Skipped: {exc}")
                errors.append((os.path.basename(v["path"]), str(exc)))
                continue
            except Exception as exc:
                print(f"  Error: {type(exc).__name__}: {exc}")
                errors.append((os.path.basename(v["path"]), f"{type(exc).__name__}: {exc}"))
                continue

        cv2.destroyAllWindows()
        self.progress["value"] = 100

        if summaries:
            batch_df = pd.DataFrame(summaries)
            folder = os.path.dirname(self.videos[0]["path"])
            batch_path = os.path.join(folder, "BatchSummary.csv")
            try:
                batch_df.to_csv(batch_path, index=False)
            except Exception as exc:
                messagebox.showerror("Save Error", f"Could not save batch summary CSV:\n\n{exc}")
            self.status_label.config(text="Batch complete.")
            msg = f"Processed {len(summaries)}/{len(self.videos)} videos.\n\nBatch summary:\n{batch_path}"
            if errors:
                msg += f"\n\n{len(errors)} video(s) had issues:\n"
                for name, err in errors[:10]:
                    msg += f"  - {name}: {err}\n"
            messagebox.showinfo("Batch Complete", msg)
        else:
            self.status_label.config(text="Batch: no videos processed.")
            if errors:
                msg = f"No videos were successfully processed ({len(errors)} had issues):\n\n"
                for name, err in errors[:15]:
                    msg += f"  - {name}: {err}\n"
                messagebox.showwarning("Batch Complete", msg)

    # ------------------------------------------------------------------
    # Multi-Mouse Tracking / Behavior Classification: shared video prep
    # ------------------------------------------------------------------

    def _prepare_source_video(self, video_path):
        """
        two_mouse.track_video() and behavior.extract_features() both take a
        plain video path and process it top to bottom -- neither knows about
        arena cropping, a start/end trim window, or a masked-out region. To
        still support Crop Arena, the time window, and Mask Zone for these
        analysis types (without touching the two_mouse.py/behavior.py files
        themselves), this pre-warps the requested frame range into a
        temporary video file -- painting any masked polygons a flat color in
        every frame first, so they're identical frame to frame and can never
        register as a difference-based detection -- and hands THAT to the
        algorithm, deleting it afterward.

        Returns (path_to_use, temp_path_or_None, fps).
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            start_time = float(self.start_entry.get() or 0)
        except ValueError:
            start_time = 0
        try:
            end_time = float(self.end_entry.get() or (total_frames / fps))
        except ValueError:
            end_time = total_frames / fps

        start_frame = max(0, int(start_time * fps))
        end_frame = min(total_frames, int(end_time * fps))

        full_range = start_frame == 0 and end_frame >= total_frames
        if full_range and not self.pending_use_crop and not self.pending_mask_points:
            cap.release()
            return video_path, None, fps

        warp_w, warp_h = self.pending_warp_w, self.pending_warp_h
        matrix = self.pending_matrix
        mask_polygons = [np.array(pts, dtype=np.int32) for pts in self.pending_mask_points]
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".avi")
        os.close(tmp_fd)
        writer = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (warp_w, warp_h))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_number = start_frame
        while frame_number < end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            warped = cv2.warpPerspective(frame, matrix, (warp_w, warp_h))
            for pts in mask_polygons:
                cv2.fillPoly(warped, [pts], (128, 128, 128))
            writer.write(warped)
            frame_number += 1
        cap.release()
        writer.release()
        return tmp_path, tmp_path, fps

    # ------------------------------------------------------------------
    # Multi-Mouse Tracking
    # ------------------------------------------------------------------

    def _run_multi_mouse_flow(self):
        if self.pending_matrix is None:
            messagebox.showerror("Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        active_path = self.videos[self.active_index]["path"]
        tmp_path = None

        try:
            try:
                min_area = int(float(self.min_area_entry.get()))
                max_area = int(float(self.max_area_entry.get()))
                threshold = int(float(self.threshold_entry.get()))
                bg_samples = int(float(self.bg_samples_entry.get()))
            except ValueError:
                messagebox.showerror("Invalid setting", "Check the Detection Settings values.")
                return

            self.progress["value"] = 0
            self.status_label.config(text="Preparing video...")
            self.root.update_idletasks()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            tracks_csv = os.path.join(output_dir, "tracks.csv")
            annotate_path = os.path.join(output_dir, "preview.mp4")

            try:
                preview_n = int(float(self.preview_samples_entry.get()))
            except ValueError:
                preview_n = 6

            color_mode = self.color_mode_var.get()
            if color_mode == "auto":
                color_mode = self._resolve_auto_color_mode(
                    source_path, num_animals=self.num_animals, min_area=min_area, max_area=max_area,
                    diff_threshold=threshold, n_background_samples=bg_samples,
                )

            self.status_label.config(text="Saving reference frames...")
            self.root.update_idletasks()
            try:
                preview_paths = save_preview_frames_multi_mouse(
                    source_path, output_dir, n_samples=preview_n,
                    num_animals=self.num_animals, min_area=min_area, max_area=max_area,
                    diff_threshold=threshold, n_background_samples=bg_samples,
                    color_mode=color_mode,
                )
            except Exception as exc:
                print(f"[warn] Reference frames: {exc}")
                preview_paths = []

            self.status_label.config(text="Tracking (multi-mouse)...")
            self.root.update_idletasks()

            tracks_df = track_video(
                source_path, tracks_csv, annotate_path=annotate_path,
                num_animals=self.num_animals, min_area=min_area, max_area=max_area,
                diff_threshold=threshold, n_background_samples=bg_samples,
                progress_callback=self.on_progress, show_display=True,
                color_mode=color_mode,
            )
            cv2.destroyAllWindows()

            # Per-mouse zone/object/distance stats
            object_names = [n.strip() for n in self.object_names_entry.get().split(",") if n.strip()] \
                if self.interact_var.get() else []
            active_object_points = {k: v for k, v in self.pending_object_points.items() if k in object_names} \
                if object_names else {}
            try:
                min_bout_s = float(self.min_bout_entry.get()) if getattr(self, "min_bout_entry", None) else 0.3
            except (ValueError, tk.TclError):
                min_bout_s = 0.3
            try:
                interaction_margin_px = float(self.interaction_margin_entry.get()) \
                    if getattr(self, "interaction_margin_entry", None) is not None else 0.0
            except (ValueError, tk.TclError):
                interaction_margin_px = 0.0

            per_mouse_summary = {}
            per_mouse_bouts = {}
            if tracks_df is not None and len(tracks_df) > 0:
                roi_to_use = self.pending_roi_points if self.loc_var.get() else {}
                for mouse_id, sub in tracks_df.groupby("mouse_id"):
                    sub = sub.sort_values("frame").reset_index(drop=True)
                    try:
                        if "x" in sub.columns and "y" in sub.columns and "frame" in sub.columns:
                            summary, bouts = compute_zone_interaction_stats(
                                sub, roi_to_use, active_object_points,
                                self.pending_warp_w, self.pending_warp_h, fps, min_bout_s=min_bout_s,
                                scale_factor=self.pending_scale_factor, scale_unit=self.pending_scale_unit,
                                interaction_margin_px=interaction_margin_px,
                            )
                            per_mouse_summary[mouse_id] = summary
                            per_mouse_bouts[mouse_id] = bouts
                    except Exception as exc:
                        print(f"[warn] zone stats for {mouse_id}: {exc}")

            summary_csv = None
            bouts_csv = None
            try:
                if per_mouse_summary:
                    summary_df = pd.DataFrame.from_dict(per_mouse_summary, orient="index")
                    summary_df.index.name = "mouse_id"
                    summary_csv = os.path.join(output_dir, "per_mouse_summary.csv")
                    summary_df.to_csv(summary_csv)
                all_bouts_rows = []
                for mouse_id, bouts in per_mouse_bouts.items():
                    for b in bouts:
                        row = dict(b)
                        row["mouse_id"] = mouse_id
                        all_bouts_rows.append(row)
                if all_bouts_rows:
                    bouts_csv = os.path.join(output_dir, "per_mouse_object_bouts.csv")
                    pd.DataFrame(all_bouts_rows).to_csv(bouts_csv, index=False)
            except Exception as exc:
                print(f"[warn] saving per-mouse CSV: {exc}")

            self.progress["value"] = 100
            self.status_label.config(text="Done.")

            results = {
                "analysis_type": "multi_mouse", "video_path": active_path, "output_dir": output_dir,
                "tracks_df": tracks_df if tracks_df is not None else pd.DataFrame(),
                "tracks_csv": tracks_csv, "annotate_path": annotate_path,
                "fps": fps, "per_mouse_summary": per_mouse_summary, "per_mouse_bouts": per_mouse_bouts,
                "summary_csv": summary_csv, "bouts_csv": bouts_csv, "preview_paths": preview_paths,
            }
            try:
                self._show_results_multi_mouse(results)
            except Exception as exc:
                messagebox.showerror("Results Error", f"Could not display results:\n\n{type(exc).__name__}: {exc}")

        except SystemExit as exc:
            self.status_label.config(text=f"Stopped: {exc}")
            cv2.destroyAllWindows()
            return
        except Exception as exc:
            self.status_label.config(text=f"Error: {type(exc).__name__}")
            cv2.destroyAllWindows()
            messagebox.showerror("Multi-Mouse Error", f"An error occurred:\n\n{type(exc).__name__}: {exc}")
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Behavior Classification
    # ------------------------------------------------------------------

    def _run_behavior_flow(self):
        if self.pending_matrix is None:
            messagebox.showerror("Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        active_path = self.videos[self.active_index]["path"]

        if getattr(self, "ml_mode_var", None) is not None and self.ml_mode_var.get():
            self._run_behavior_flow_ml(active_path)
            return

        tmp_path = None
        try:
            try:
                min_area = int(float(self.min_area_entry.get()))
                max_area = int(float(self.max_area_entry.get()))
                threshold = int(float(self.threshold_entry.get()))
                bg_samples = int(float(self.bg_samples_entry.get()))
                max_jump = float(self.max_jump_entry.get())
                loco_thresh = float(self.loco_thresh_entry.get())
                rear_thresh = float(self.rear_thresh_entry.get())
                groom_thresh = float(self.groom_thresh_entry.get())
                persistence_frames = int(float(self.immobile_thresh_entry.get()))
                min_bout_s = float(self.min_bout_entry.get())
            except (ValueError, tk.TclError, AttributeError):
                messagebox.showerror("Invalid setting", "Check the Detection Settings values.")
                return

            self.progress["value"] = 0
            self.status_label.config(text="Preparing video...")
            self.root.update_idletasks()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            self._last_output_dir = output_dir
            features_csv = os.path.join(output_dir, "features.csv")
            bouts_csv = os.path.join(output_dir, "bouts.csv")
            labeled_csv = os.path.join(output_dir, "labeled_frames.csv")

            try:
                preview_n = int(float(self.preview_samples_entry.get()))
            except (ValueError, tk.TclError, AttributeError):
                preview_n = 6

            color_mode = self.color_mode_var.get()
            if color_mode == "auto":
                color_mode = self._resolve_auto_color_mode(
                    source_path, num_animals=self.num_animals, min_area=min_area, max_area=max_area,
                    diff_threshold=threshold, n_background_samples=bg_samples,
                )

            self.status_label.config(text="Saving reference frames...")
            self.root.update_idletasks()
            try:
                preview_paths = save_preview_frames_behavior(
                    source_path, output_dir, n_samples=preview_n, num_animals=self.num_animals,
                    min_area=min_area, max_area=max_area, diff_threshold=threshold,
                    n_background_samples=bg_samples, color_mode=color_mode,
                    max_jump_px=max_jump,
                )
            except Exception as exc:
                messagebox.showerror("Reference frames failed", str(exc))
                preview_paths = []

            self.status_label.config(text="Extracting features...")
            self.root.update_idletasks()

            extract_features(
                source_path, features_csv, num_animals=self.num_animals,
                min_area=min_area, max_area=max_area, diff_threshold=threshold,
                n_background_samples=bg_samples, color_mode=color_mode,
                max_jump_px=max_jump, progress_callback=self.on_progress,
            )
            self.status_label.config(text="Classifying behaviors...")
            self.root.update_idletasks()
            labeled_df, bouts_df = classify_behaviors(
                features_csv, bouts_csv, labeled_output_csv=labeled_csv,
                loco_body_move_high=loco_thresh, rear_score_threshold=rear_thresh,
                groom_score_threshold=groom_thresh,
                groom_min_persistence=persistence_frames, rear_min_persistence=persistence_frames,
                min_bout_s=min_bout_s,
            )

            if labeled_df is None or len(labeled_df) == 0:
                messagebox.showwarning(
                    "No behavior data",
                    "No frames could be classified. Try adjusting the detection threshold or area settings."
                )

            selected = {
                "rearing": getattr(self, "behavior_rearing_var", tk.BooleanVar(value=True)).get(),
                "grooming": getattr(self, "behavior_grooming_var", tk.BooleanVar(value=True)).get(),
                "locomotion": getattr(self, "behavior_locomotion_var", tk.BooleanVar(value=True)).get(),
                "immobile": getattr(self, "behavior_immobile_var", tk.BooleanVar(value=True)).get(),
            }
            if (not any(selected.values())) or getattr(self, "behavior_all_var", tk.BooleanVar(value=True)).get():
                selected = {k: True for k in selected}

            self.progress["value"] = 100
            self.status_label.config(text="Done.")

            results = {
                "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df, "selected_behaviors": selected,
                "bouts_csv": bouts_csv, "labeled_csv": labeled_csv, "fps": fps, "preview_paths": preview_paths,
            }
            try:
                self._show_results_behavior(results)
            except Exception as exc:
                messagebox.showerror("Results display failed", f"Results were saved but could not be shown: {exc}")

        except SystemExit:
            self.progress["value"] = 0
            self.status_label.config(text="Tracking stopped.")
            cv2.destroyAllWindows()
            self.root.update_idletasks()
            raise
        except Exception as exc:
            self.progress["value"] = 0
            self.status_label.config(text="Error occurred.")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.root.update_idletasks()
            messagebox.showerror("Tracking failed", str(exc))
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _run_behavior_flow_ml(self, active_path):
        """The optional-mode counterpart to _run_behavior_flow(): instead
        of extract_features()+classify_behaviors(), runs the trained
        deep-learning model (tracking.ml_infer.classify_video_ml) over the
        video and builds the exact same results dict shape
        _show_results_behavior() expects -- same idea as
        _run_behavior_manual_flow() reusing that screen for manual
        scoring. v1 limitation: the model looks at the whole frame, not a
        per-animal crop, so with more than one animal in frame its bouts
        describe the SCENE, not a specific mouse -- warned about below."""
        checkpoint_path = self.ml_checkpoint_entry.get().strip()
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            messagebox.showerror(
                "No trained model",
                "Choose a trained model (.pt) file above, or use 'Train Model...' to make one first."
            )
            return
        if not TORCH_AVAILABLE:
            messagebox.showerror(
                "PyTorch not installed",
                "Install PyTorch (see pytorch.org) to use a trained model here, or uncheck "
                "'Use trained model' to use the automatic rules instead."
            )
            return
        if self.num_animals > 1:
            if not messagebox.askyesno(
                "Single-animal model",
                "The trained model looks at the whole frame, not one animal at a time -- with "
                f"{self.num_animals} animals in frame its bouts describe the scene as a whole, "
                "not a specific mouse. Continue anyway?"
            ):
                return

        tmp_path = None
        try:
            self.progress["value"] = 0
            self.status_label.config(text="Preparing video...")
            self.root.update_idletasks()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            self._last_output_dir = output_dir
            bouts_csv = os.path.join(output_dir, "ml_bouts.csv")

            self.status_label.config(text="Running trained model...")
            self.root.update_idletasks()

            def on_window(done, total):
                self.on_progress(done / total if total else 1.0)

            subject_name = f"mouse_{chr(65)}"
            bouts_df = classify_video_ml(
                source_path, checkpoint_path, subject=subject_name,
                fps_override=fps, progress_callback=on_window,
            )
            os.makedirs(output_dir, exist_ok=True)
            bouts_df.to_csv(bouts_csv, index=False)

            if len(bouts_df) == 0:
                messagebox.showwarning(
                    "No behavior data",
                    "The trained model didn't confidently detect any of its trained behaviors in "
                    "this video."
                )

            cap = cv2.VideoCapture(source_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
            cap.release()
            duration_s = (frame_count / vid_fps) if vid_fps else 0.0
            labeled_df = pd.DataFrame({"time_s": [duration_s]})

            selected = {b: True for b in bouts_df["behavior"].unique()} if len(bouts_df) else \
                {"grooming": True, "rearing": True}

            self.progress["value"] = 100
            self.status_label.config(text=f"Done -- {len(bouts_df)} bouts from the trained model.")

            results = {
                "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df, "selected_behaviors": selected,
                "bouts_csv": bouts_csv, "labeled_csv": bouts_csv, "fps": vid_fps, "preview_paths": [],
            }
            try:
                self._show_results_behavior(results)
            except Exception as exc:
                messagebox.showerror("Results display failed", f"Results were saved but could not be shown: {exc}")

        except Exception as exc:
            self.progress["value"] = 0
            self.status_label.config(text="Error occurred.")
            self.root.update_idletasks()
            messagebox.showerror("ML classification failed", str(exc))
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def on_manual_behavior_scoring(self):
        if not self.videos:
            messagebox.showerror("No video", "Add a video first.")
            return
        if self.active_index is None:
            messagebox.showerror("No video selected", "Click a video in the list to select it.")
            return
        if self.pending_matrix is None:
            messagebox.showerror("Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return
        self._run_behavior_manual_flow()

    def _run_behavior_manual_flow(self):
        """Same idea as _run_behavior_flow(), but instead of the automatic
        classifier this opens the (cropped/masked/time-windowed, same as
        the automatic path) video in its own window and lets the person
        mark bout start/stop themselves -- see manual_score_video(). Builds
        the exact same results dict shape _show_results_behavior() expects,
        so the results screen and export don't need to know which one ran."""
        active_path = self.videos[self.active_index]["path"]
        tmp_path = None
        try:
            self.status_label.config(text="Preparing video...")
            self.root.update_idletasks()

            source_path, tmp_path, fps = self._prepare_source_video(active_path)
            output_dir = compute_output_dir(active_path)
            self._last_output_dir = output_dir
            bouts_csv = os.path.join(output_dir, "manual_bouts.csv")

            subject_names = [f"mouse_{chr(65 + i)}" for i in range(self.num_animals)]

            self.status_label.config(text="Manual scoring -- video window open. "
                                           "SPACE=play/pause, 1-4=behaviors, q=finish.")
            self.root.update_idletasks()

            bouts_df = manual_score_video(source_path, subject_names=subject_names)

            os.makedirs(output_dir, exist_ok=True)
            bouts_df.to_csv(bouts_csv, index=False)

            cap = cv2.VideoCapture(source_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
            cap.release()
            duration_s = (frame_count / vid_fps) if vid_fps else 0.0
            labeled_df = pd.DataFrame({"time_s": [duration_s]})

            self.progress["value"] = 100
            self.status_label.config(text=f"Manual scoring saved -- {len(bouts_df)} bouts.")

            results = {
                "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
                "labeled_df": labeled_df, "bouts_df": bouts_df,
                "selected_behaviors": {"rearing": True, "grooming": True, "locomotion": True, "immobile": True},
                "bouts_csv": bouts_csv, "labeled_csv": bouts_csv, "fps": vid_fps, "preview_paths": [],
            }
            try:
                self._show_results_behavior(results)
            except Exception as exc:
                messagebox.showerror("Results display failed", f"Results were saved but could not be shown: {exc}")

        except Exception as exc:
            self.progress["value"] = 0
            self.status_label.config(text="Error occurred.")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            self.root.update_idletasks()
            messagebox.showerror("Manual scoring failed", str(exc))
            return
        finally:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Results screens
    # ------------------------------------------------------------------

    def _results_header(self, title_suffix):
        self._set_step("results")
        for child in self.body_container.winfo_children():
            child.destroy()

        top = ttk.Frame(self.body_container, padding=(0, 0, 0, 10))
        top.pack(fill="x")
        ttk.Label(top, text="Results: " + title_suffix, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Open Results Folder",
                  command=lambda: self._open_folder(self._last_output_dir)).pack(side="right")
        ttk.Button(top, text="New Analysis", style="Accent.TButton",
                  command=self._back_to_setup).pack(side="right", padx=(0, 8))
        ttk.Separator(self.body_container).pack(fill="x", pady=(0, 10))

    def _open_folder(self, path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception:
            messagebox.showinfo("Results folder", path)

    def _back_to_setup(self):
        self._build_setup_body()

    def _make_scrollable_panel(self, outer):
        """Turns the rest of `outer` (below whatever's already packed in it,
        e.g. a header) into a vertically scrollable area, so a longer
        settings list never silently gets cut off by the window's height --
        which analysis type has more fields than the window is tall stops
        mattering, since you can always scroll to reach the rest."""
        canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(canvas_window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the fixed-width scrollbar FIRST, then let the canvas expand
        # into whatever's left -- packing them the other way around let the
        # expand=True canvas claim the scrollbar's space first (same class
        # of bug as the earlier bottom-bar issue), collapsing it to 1px.
        scrollbar.pack(side="right", fill="y", pady=(6, 0))
        canvas.pack(side="left", fill="both", expand=True, pady=(6, 0))

        def _on_mousewheel(event):
            delta = -1 * int(event.delta / 120) if event.delta else (-1 if event.num == 4 else 1)
            canvas.yview_scroll(delta, "units")

        def _bind_wheel(_e):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_e):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        canvas.bind("<Enter>", _bind_wheel)
        canvas.bind("<Leave>", _unbind_wheel)
        return inner

    def _add_reference_frames_gallery(self, parent, preview_paths, thumb_w=150):
        """Small thumbnail gallery of the calibration/reference frames saved
        by save_preview_frames() -- lets you sanity-check detection at a
        few sample points across the video, same idea as Standard
        Tracking's own preview-frame check."""
        if not preview_paths:
            return
        ttk.Label(parent, text="Reference Frames", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10, 4))
        gallery = ttk.Frame(parent)
        gallery.pack(fill="x")
        self._reference_thumbs = []  # keep strong refs so PhotoImages aren't garbage-collected
        cols = 4
        shown = 0
        for path in preview_paths:
            if not os.path.isfile(path):
                continue
            frame = cv2.imread(path)
            if frame is None:
                continue
            h, w = frame.shape[:2]
            scale = thumb_w / w
            disp = cv2.resize(frame, (thumb_w, max(1, int(h * scale))))
            photo = self._bgr_to_photoimage(disp)
            self._reference_thumbs.append(photo)
            lbl = tk.Label(gallery, image=photo, bd=1, relief="solid")
            r, c = divmod(shown, cols)
            lbl.grid(row=r, column=c, padx=3, pady=3)
            shown += 1

    def _show_results_standard(self, summary):
        self._last_output_dir = summary.get("Output_folder", "")
        self._results_header(os.path.basename(summary.get("Video", "")))

        body = ttk.Frame(self.body_container)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        # Scrollable -- with several zones/objects the per-zone stat list
        # below can run longer than a shorter window's height.
        left_scroll = self._make_scrollable_panel(left)
        lf = ttk.LabelFrame(left_scroll, text="Summary", padding=10)
        lf.pack(fill="x")
        stats = [("Tracking quality", f"{summary.get('Tracking_quality_percent', 0):.1f} %"),
                 ("Total transitions", str(summary.get("Total_transitions", 0))),
                 ("Total distance (px)", f"{summary.get('Total_distance_pixels', 0):.1f}")]
        for k, v in summary.items():
            # Skip the per-zone full/half/semi 3-point entry-depth
            # breakdown here (Entry_<zone> in the raw CSV, "_full_time_s"/
            # "_half_time_s"/"_semi_time_s" in this same summary dict) --
            # like the existing arm-entries/alternation numbers, it's a
            # detailed derived metric meant for the exported CSV/Excel
            # report, not this at-a-glance panel, which stays limited to
            # one headline "time in zone" row per zone/object.
            if k.endswith("_time_s") and not k.endswith(("_full_time_s", "_half_time_s", "_semi_time_s")):
                stats.append((k.replace("_time_s", " time (s)"), f"{v:.1f}"))
        for k, v in stats:
            ttk.Label(lf, text=k, foreground=self._palette["MUTED"], font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
            ttk.Label(lf, text=v, font=("Segoe UI", 13, "bold")).pack(anchor="w")

        center = ttk.Frame(body, padding=(14, 0))
        center.pack(side="left", fill="both", expand=True)

        # View switcher -- Trajectory (path colored by elapsed time) /
        # Heatmap (occupancy density) / Zone Occupancy (per-zone time bar
        # chart, when zones were used) -- all pre-rendered PNGs from
        # output.graphs, swapped into the same canvas rather than shown as
        # three separate static images, so the panel stays compact.
        header_row = ttk.Frame(center)
        header_row.pack(fill="x")
        ttk.Label(header_row, text="Results view", font=("Segoe UI", 10, "bold")).pack(side="left")

        views = [("Trajectory", "trajectory.png"), ("Heatmap", "heatmap.png"),
                 ("Zone Occupancy", "zone_occupancy.png")]
        available_views = [(label, fname) for label, fname in views
                            if os.path.exists(os.path.join(self._last_output_dir, fname))]

        img_canvas = tk.Canvas(center, bg="#dddddd", highlightthickness=0, height=340)
        self._results_photo = None

        def show_view(fname):
            for lbl, btn in view_buttons.items():
                btn.configure(style="Accent.TButton" if fname == views_by_label[lbl] else "Tool.TButton")
            self._load_png_into_canvas(img_canvas, os.path.join(self._last_output_dir, fname))

        views_by_label = dict(available_views)
        view_buttons = {}
        button_row = ttk.Frame(header_row)
        button_row.pack(side="right")
        for label, fname in available_views:
            b = ttk.Button(button_row, text=label, style="Tool.TButton",
                            command=lambda f=fname: show_view(f))
            b.pack(side="left", padx=(4, 0))
            view_buttons[label] = b

        img_canvas.pack(fill="x", pady=(6, 0))
        if available_views:
            show_view(available_views[0][1])

        preview_dir = os.path.join(self._last_output_dir, "preview")
        preview_paths = []
        if os.path.isdir(preview_dir):
            preview_paths = [os.path.join(preview_dir, f) for f in sorted(os.listdir(preview_dir))
                              if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self._add_reference_frames_gallery(center, preview_paths)

        right = ttk.Frame(body, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        rf = ttk.LabelFrame(right, text="Output Files", padding=10)
        rf.pack(fill="x")
        if os.path.isdir(self._last_output_dir):
            for f in sorted(os.listdir(self._last_output_dir)):
                if os.path.isfile(os.path.join(self._last_output_dir, f)):
                    ttk.Label(rf, text="- " + f, font=("Segoe UI", 8)).pack(anchor="w", pady=1)

    def _show_results_multi_mouse(self, results):
        self._last_output_dir = results["output_dir"]
        self._results_header(os.path.basename(results["video_path"]))

        df = results["tracks_df"]
        per_mouse_summary = results.get("per_mouse_summary", {})
        body = ttk.Frame(self.body_container)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=300)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        # Scrollable -- with 2-3 mice and several zones/objects, the
        # Per-Mouse Zones/Distance list below can run longer than a
        # shorter window's height.
        left_scroll = self._make_scrollable_panel(left)
        lf = ttk.LabelFrame(left_scroll, text="Summary", padding=10)
        lf.pack(fill="x")
        duration_s = df["time_s"].max() if len(df) else 0
        ttk.Label(lf, text="Duration analyzed", foreground=self._palette["MUTED"], font=("Segoe UI", 9)).pack(
            anchor="w", pady=(6, 0))
        ttk.Label(lf, text=f"{duration_s:.1f} s", font=("Segoe UI", 13, "bold")).pack(anchor="w")

        colors = ["#c0392b", "#2f6fb0", "#27ae60"]
        mouse_ids = sorted(df["mouse_id"].unique())

        tf = ttk.LabelFrame(left_scroll, text="Per-Mouse Tracking", padding=10)
        tf.pack(fill="x", pady=(10, 0))
        for i, mid in enumerate(mouse_ids):
            sub = df[df["mouse_id"] == mid]
            pct_ok = 100 * (sub["status"] == "ok").sum() / len(sub) if len(sub) else 0
            row = ttk.Frame(tf)
            row.pack(fill="x", pady=2)
            tk.Canvas(row, width=12, height=12, bg=colors[i % len(colors)], highlightthickness=0).pack(side="left")
            ttk.Label(row, text=f" {mid} -- {pct_ok:.1f}% tracked normally", font=("Segoe UI", 8)).pack(side="left")

        # Per-mouse zone time / distance / object bouts, if zones/objects/
        # distance calibration were configured on the Preview panel.
        if per_mouse_summary:
            zf = ttk.LabelFrame(left_scroll, text="Per-Mouse Zones / Distance", padding=10)
            zf.pack(fill="x", pady=(10, 0))
            for i, mid in enumerate(mouse_ids):
                summ = per_mouse_summary.get(mid, {})
                if not summ:
                    continue
                ttk.Label(zf, text=mid, font=("Segoe UI", 8, "bold"),
                          foreground=colors[i % len(colors)]).pack(anchor="w", pady=(6 if i else 0, 0))
                for k, v in summ.items():
                    if k.endswith("_time_s"):
                        zone_name = k[:-len("_time_s")]
                        ttk.Label(zf, text=f"  {zone_name}: {v:.1f} s", font=("Segoe UI", 8)).pack(anchor="w")
                    elif k.startswith("Total_distance_") and k != "Total_distance_pixels":
                        unit = k[len("Total_distance_"):]
                        ttk.Label(zf, text=f"  distance: {v:.1f} {unit}", font=("Segoe UI", 8)).pack(anchor="w")
                    elif k == "Total_distance_pixels" and not any(
                            kk.startswith("Total_distance_") and kk != "Total_distance_pixels" for kk in summ):
                        ttk.Label(zf, text=f"  distance: {v:.1f} px", font=("Segoe UI", 8)).pack(anchor="w")
                    elif k.endswith("_bout_count"):
                        obj_name = k[:-len("_bout_count")]
                        ttk.Label(zf, text=f"  near {obj_name}: {int(v)} bout(s)", font=("Segoe UI", 8)).pack(
                            anchor="w")

        center = ttk.Frame(body, padding=(14, 0))
        center.pack(side="left", fill="both", expand=True)
        ttk.Label(center, text="Trajectory (per mouse)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        traj_canvas = tk.Canvas(center, bg="#dddddd", highlightthickness=0, height=340)
        traj_canvas.pack(fill="x", pady=(6, 0))
        traj_canvas.update_idletasks()
        cw, ch = max(300, traj_canvas.winfo_width()), max(200, traj_canvas.winfo_height())
        xs_all = df["x"].dropna()
        ys_all = df["y"].dropna()
        if len(xs_all) and len(ys_all):
            x_min, x_max = xs_all.min(), xs_all.max()
            y_min, y_max = ys_all.min(), ys_all.max()
            pad = 20
            sx = (cw - 2 * pad) / max(1.0, (x_max - x_min))
            sy = (ch - 2 * pad) / max(1.0, (y_max - y_min))
            scale = min(sx, sy)
            for i, mid in enumerate(mouse_ids):
                sub = df[df["mouse_id"] == mid].dropna(subset=["x", "y"])
                pts = [(pad + (x - x_min) * scale, pad + (y - y_min) * scale) for x, y in zip(sub["x"], sub["y"])]
                for j in range(len(pts) - 1):
                    traj_canvas.create_line(*pts[j], *pts[j + 1], fill=colors[i % len(colors)], width=1)

        self._add_reference_frames_gallery(center, results.get("preview_paths"))

        right = ttk.Frame(body, width=280)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        rf = ttk.LabelFrame(right, text="Output Files", padding=10)
        rf.pack(fill="x")
        file_list = [os.path.basename(results["tracks_csv"]),
                     os.path.basename(results["annotate_path"]) + "  (watch for ID swaps)"]
        if results.get("summary_csv"):
            file_list.append(os.path.basename(results["summary_csv"]))
        if results.get("bouts_csv"):
            file_list.append(os.path.basename(results["bouts_csv"]))
        for f in file_list:
            ttk.Label(rf, text="- " + f, font=("Segoe UI", 8)).pack(anchor="w", pady=1)

    def _show_results_behavior(self, results):
        self._last_output_dir = results["output_dir"]
        self._results_header(os.path.basename(results["video_path"]))

        bouts_df = results["bouts_df"]
        selected = results["selected_behaviors"]

        body = ttk.Frame(self.body_container)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, width=280)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        lf = ttk.LabelFrame(left, text="Summary", padding=10)
        lf.pack(fill="x")
        labeled_df = results["labeled_df"]
        duration_s = labeled_df["time_s"].max() if len(labeled_df) else 0
        for k, v in [("Duration analyzed", f"{duration_s:.1f} s"), ("Total bouts", str(len(bouts_df)))]:
            ttk.Label(lf, text=k, foreground=self._palette["MUTED"], font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
            ttk.Label(lf, text=v, font=("Segoe UI", 13, "bold")).pack(anchor="w")

        rf = ttk.LabelFrame(left, text="Output Files", padding=10)
        rf.pack(fill="x", pady=(10, 0))
        for f in dict.fromkeys([os.path.basename(results["bouts_csv"]), os.path.basename(results["labeled_csv"])]):
            ttk.Label(rf, text="- " + f, font=("Segoe UI", 8)).pack(anchor="w", pady=1)

        center = ttk.Frame(body, padding=(14, 0))
        center.pack(side="left", fill="both", expand=True)

        behavior_colors = {"locomotion": "#9e9e9e", "rearing": "#8e44ad", "grooming": "#d35400",
                          "immobile": "#2c3e50", "other": "#bdbdbd", "undetermined": "#e0e0e0"}

        ttk.Label(center, text="Behavior Timeline (Ethogram)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        subjects = sorted(bouts_df["subject"].unique()) if len(bouts_df) else []
        eth_h = max(80, 40 * len(subjects) + 40)
        eth_canvas = tk.Canvas(center, bg="white", highlightthickness=1, highlightbackground="#c9c9c9", height=eth_h)
        eth_canvas.pack(fill="x", pady=(6, 10))
        eth_canvas.update_idletasks()
        cw = max(300, eth_canvas.winfo_width())
        total_s = duration_s if duration_s > 0 else 1
        label_w = 70
        track_w = cw - label_w - 20
        for i, subj in enumerate(subjects):
            y0 = 10 + i * 34
            eth_canvas.create_text(8, y0 + 10, text=subj, anchor="w", font=("Segoe UI", 8, "bold"))
            sub_bouts = bouts_df[bouts_df["subject"] == subj]
            for _, r in sub_bouts.iterrows():
                x0 = label_w + (r["start_s"] / total_s) * track_w
                x1 = label_w + (r["stop_s"] / total_s) * track_w
                color = behavior_colors.get(r["behavior"], "#e0e0e0")
                eth_canvas.create_rectangle(x0, y0, max(x1, x0 + 1), y0 + 20, fill=color, outline="")
        legend_y = 10 + max(1, len(subjects)) * 34 + 6
        lx = label_w
        for name in ["locomotion", "rearing", "grooming", "immobile"]:
            eth_canvas.create_rectangle(lx, legend_y, lx + 12, legend_y + 12, fill=behavior_colors[name], outline="")
            eth_canvas.create_text(lx + 16, legend_y + 6, text=name, anchor="w", font=("Segoe UI", 8))
            lx += 95

        ttk.Label(center, text="Behavior Bouts", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        table_frame = ttk.Frame(center)
        table_frame.pack(fill="x", pady=(6, 10))
        cols = ["Subject", "Behavior", "Count", "Total s", "Mean s"]
        cw = [90, 100, 70, 80, 80]
        hdr = ttk.Frame(table_frame)
        hdr.pack(fill="x")
        for c, w in zip(cols, cw):
            ttk.Label(hdr, text=c, font=("Segoe UI", 8, "bold"), width=int(w / 8)).pack(side="left")

        if len(bouts_df):
            shown = bouts_df[bouts_df["behavior"].isin([b for b, v in selected.items() if v] + ["other", "undetermined"])]
            summary_tbl = shown.groupby(["subject", "behavior"])["duration_s"].agg(["count", "sum", "mean"]).reset_index()
            for _, r in summary_tbl.iterrows():
                rowf = ttk.Frame(table_frame)
                rowf.pack(fill="x")
                vals = [r["subject"], r["behavior"], str(int(r["count"])), f"{r['sum']:.1f}", f"{r['mean']:.1f}"]
                for val, w in zip(vals, cw):
                    ttk.Label(rowf, text=val, font=("Segoe UI", 8), width=int(w / 8)).pack(side="left")
        else:
            ttk.Label(center, text="No bouts detected.", foreground=self._palette["MUTED"]).pack(anchor="w")

        self._add_reference_frames_gallery(center, results.get("preview_paths"))

        right = ttk.Frame(body, width=200)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        legend = ttk.LabelFrame(right, text="Legend", padding=10)
        legend.pack(fill="x")
        for name, color in behavior_colors.items():
            if name in ("other", "undetermined"):
                continue
            row = ttk.Frame(legend)
            row.pack(fill="x", pady=1)
            tk.Canvas(row, width=12, height=12, bg=color, highlightthickness=0).pack(side="left")
            ttk.Label(row, text=" " + name, font=("Segoe UI", 8)).pack(side="left")



def launch_gui():
    root = tk.Tk()
    TrackerApp(root)
    root.mainloop()
