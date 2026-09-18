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
from tracking.two_mouse import track_video, save_preview_frames as save_preview_frames_multi_mouse
from tracking.behavior import extract_features, classify_behaviors, save_preview_frames as save_preview_frames_behavior
from tracking.boris_events import read_tracking_csv, generate_events, save_events


def resource_path(relative_path):
    """
    Resolve a path to a bundled resource (e.g. resources/icon.png) that
    works both when running normally (python main.py) and when packaged
    by PyInstaller, which unpacks --add-data files into a temporary
    sys._MEIPASS folder at runtime instead of the real project layout.
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
        default = ",".join(state.get("roi_names", ["Light", "Dark"]))
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

        default = ",".join(state.get("behavior_names", ["Sniffing", "Touching", "Climbing"]))
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

        # Remembered field values -- carried over whenever the setup panel
        # rebuilds (switching Analysis Type, adding a video, etc.) so
        # typing in a custom time window or zone names doesn't get wiped
        # back to the hardcoded defaults. Only the "Reset" buttons clear
        # these back to the defaults below.
        self._remembered_start = ""
        self._remembered_end = ""
        self._remembered_roi_names = ""
        self._last_built_mode = None

        root.title("BehavioralTracker")
        root.geometry("1550x980")
        root.minsize(1200, 760)
        root.resizable(True, True)
        try:
            root.state("zoomed")  # Windows: opens maximized by default
        except tk.TclError:
            try:
                root.attributes("-zoomed", True)  # some Linux window managers
            except tk.TclError:
                root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")

        try:
            icon_path = resource_path(os.path.join("resources", "icon.png"))
            if os.path.exists(icon_path):
                self._icon_image = tk.PhotoImage(file=icon_path)
                root.iconphoto(True, self._icon_image)
        except Exception:
            pass

        # ---- color palette + ttk styling ----
        BG = "#f7f7f7"
        HEADER_BG = "#1f2937"
        ACCENT = "#2f6fb0"
        ACCENT_DARK = "#24557f"
        GREEN = "#4caf7d"
        TEXT = "#1a1a1a"
        MUTED = "#666666"

        root.configure(bg=BG)

        style = ttk.Style()
        for theme in ("clam", "vista", "aqua"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break

        style.configure(".", background=BG, foreground=TEXT, font=("Segoe UI", 9))
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("TCheckbutton", background=BG, foreground=TEXT)
        style.configure("TRadiobutton", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.map("TRadiobutton", foreground=[("selected", ACCENT)],
                  font=[("selected", ("Segoe UI", 10, "bold"))])
        style.configure("TLabelframe", background=BG, bordercolor="#c9c9c9")
        style.configure("TLabelframe.Label", background=BG, foreground=TEXT, font=("Segoe UI", 10, "bold"))
        style.configure("TEntry", fieldbackground="white")

        style.configure("Header.TFrame", background=HEADER_BG)
        style.configure("HeaderTitle.TLabel", background=HEADER_BG, foreground="white",
                        font=("Segoe UI", 15, "bold"))
        style.configure("HeaderBadge.TLabel", background=ACCENT, foreground="white",
                        font=("Segoe UI", 11, "bold"), anchor="center")

        style.configure("Accent.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)], foreground=[("active", "white")])

        style.configure("Start.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 12, "bold"))
        style.map("Start.TButton", background=[("active", ACCENT_DARK)], foreground=[("active", "white")])

        style.configure("Tool.TButton", padding=6, font=("Segoe UI", 9))
        style.configure("ToolActive.TButton", padding=6, font=("Segoe UI", 9, "bold"),
                        background=ACCENT, foreground="white")
        style.map("ToolActive.TButton", background=[("active", ACCENT_DARK)], foreground=[("active", "white")])

        style.configure("Reset.TButton", padding=2, font=("Segoe UI", 8))
        style.configure("Danger.TButton", padding=4, font=("Segoe UI", 9, "bold"),
                        background="#b0392f", foreground="white")
        style.map("Danger.TButton", background=[("active", "#8f2e26")], foreground=[("active", "white")])

        style.configure("Green.Horizontal.TProgressbar", background=GREEN, troughcolor="#e2e2e2")

        self._palette = {"BG": BG, "HEADER_BG": HEADER_BG, "ACCENT": ACCENT,
                         "ACCENT_DARK": ACCENT_DARK, "GREEN": GREEN, "TEXT": TEXT, "MUTED": MUTED}

        # ---- header bar ----
        header = ttk.Frame(root, style="Header.TFrame", padding=(14, 10))
        header.pack(fill="x")
        ttk.Label(header, text="B", style="HeaderBadge.TLabel", width=2).pack(side="left")
        ttk.Label(header, text="BehavioralTracker", style="HeaderTitle.TLabel").pack(side="left", padx=(10, 0))
        ttk.Button(header, text="Reset All", style="Danger.TButton",
                  command=self.on_reset_all).pack(side="right")

        # ---- mode row ----
        mode_row = ttk.Frame(root, padding=(14, 8))
        mode_row.pack(fill="x")
        self.mode_var = tk.StringVar(value="individual")
        self.mode_var.trace_add("write", self._on_mode_change)
        ttk.Radiobutton(mode_row, text="Individual", value="individual", variable=self.mode_var).pack(side="left")
        ttk.Radiobutton(mode_row, text="Batch", value="batch", variable=self.mode_var).pack(side="left", padx=(20, 0))
        self.status_label = ttk.Label(mode_row, text="Idle.", foreground=MUTED)
        self.status_label.pack(side="right")

        # ---- analysis type row ----
        at_row = ttk.Frame(root, padding=(14, 8))
        at_row.pack(fill="x")
        ttk.Label(at_row, text="Analysis Type:", font=("Segoe UI", 9, "bold")).pack(side="left")
        self.analysis_type = "standard"
        self.analysis_type_buttons = {}
        for val, text in [("standard", "Standard Tracking"), ("multi_mouse", "Multi-Mouse Tracking"),
                           ("behavior", "Behavior Classification (Grooming/Rearing)")]:
            b = ttk.Button(at_row, text=text, style=("Accent.TButton" if val == "standard" else "Tool.TButton"),
                          command=lambda v=val: self._set_analysis_type(v))
            b.pack(side="left", padx=(10, 0))
            self.analysis_type_buttons[val] = b

        ttk.Separator(root).pack(fill="x")

        self.body_container = ttk.Frame(root)
        self.body_container.pack(fill="both", expand=True)

        self._build_setup_body()

    # ------------------------------------------------------------------
    # Analysis type switching + setup-body construction
    # ------------------------------------------------------------------

    def _set_analysis_type(self, value):
        if value == self.analysis_type:
            return
        self.analysis_type = value
        for val, btn in self.analysis_type_buttons.items():
            btn.configure(style="Accent.TButton" if val == value else "Tool.TButton")
        self._reset_calibration_state()
        self._build_setup_body()

    def _build_setup_body(self):
        # Capture whatever's currently in these fields before the rebuild
        # destroys the widgets, so switching Analysis Type (or re-adding a
        # video) doesn't silently wipe custom values back to the defaults.
        # This can be called when the setup panel ISN'T currently showing
        # (e.g. "New Analysis" from a results screen, which already
        # destroyed these widgets) -- self.start_entry etc. then point to a
        # dead Tk widget, so guard every read against that.
        try:
            if hasattr(self, "start_entry"):
                self._remembered_start = self.start_entry.get()
        except tk.TclError:
            pass
        try:
            if hasattr(self, "end_entry"):
                self._remembered_end = self.end_entry.get()
        except tk.TclError:
            pass
        # roi_names_entry exists even in Behavior mode (a harmless unpacked
        # stand-in for code that references it generically) -- only capture
        # it when the panel we're LEAVING actually had a real zone-names
        # field, or leaving Behavior mode would overwrite a good remembered
        # value with that stand-in's empty content.
        try:
            if getattr(self, "roi_names_entry", None) is not None and self._last_built_mode != "behavior":
                self._remembered_roi_names = self.roi_names_entry.get()
        except tk.TclError:
            pass

        for child in self.body_container.winfo_children():
            child.destroy()

        if self.analysis_type == "behavior":
            self._build_behavior_setup(self.body_container)
        else:
            self._build_zone_setup(self.body_container, self.analysis_type)

        self._render_canvas()

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

        list_frame = ttk.Frame(left, relief="sunken", borderwidth=1)
        list_frame.pack(fill="x", pady=(6, 4))
        self.video_list_inner = ttk.Frame(list_frame)
        self.video_list_inner.pack(fill="x")

        self.video_mode_note = ttk.Label(left, text="", font=("Segoe UI", 7), foreground=MUTED)
        self.video_mode_note.pack(anchor="w")
        ttk.Label(left, text="(click a video to make it active for the preview)",
                  font=("Segoe UI", 7), foreground="#888888").pack(anchor="w")
        self._refresh_video_list()

        time_frame = ttk.LabelFrame(left, text="Time window & zone names", padding=8)
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
        self.object_names_entry.pack(fill="x")

        out_frame = ttk.Frame(left, padding=(0, 8, 0, 0))
        out_frame.pack(fill="x")
        ttk.Label(out_frame, text="Video output size:").pack(side="left")
        self.output_size_entry = ttk.Entry(out_frame, width=12)
        self.output_size_entry.pack(side="left", padx=(6, 0))

        analysis_frame = ttk.LabelFrame(left, text="Analysis", padding=8)
        analysis_frame.pack(fill="x", pady=(10, 0))
        self.loc_var = tk.BooleanVar(value=True)
        self.interact_var = tk.BooleanVar(value=False)
        self.entries_var = tk.BooleanVar(value=False)
        self.altern_var = tk.BooleanVar(value=False)
        self.all_var = tk.BooleanVar(value=False)

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
        ttk.Separator(analysis_frame).pack(fill="x", pady=4)
        ttk.Checkbutton(analysis_frame, text="All Behaviours", variable=self.all_var,
                        command=self.on_all_toggle).pack(anchor="w")

        animals_row = ttk.Frame(analysis_frame)
        animals_row.pack(fill="x", pady=(8, 0))
        ttk.Label(animals_row, text="Animals in frame:").pack(side="left")
        default_n = 2 if mode == "multi_mouse" else 1
        self.num_animals = default_n
        self.num_animals_buttons = {}
        for n in (1, 2, 3):
            b = ttk.Button(animals_row, text=str(n), width=3,
                           style=("Accent.TButton" if n == default_n else "Tool.TButton"),
                           command=lambda n=n: self.on_num_animals(n))
            b.pack(side="left", padx=(6, 0))
            self.num_animals_buttons[n] = b
        if mode == "multi_mouse":
            ttk.Label(analysis_frame, text="Uses the multi-mouse tracker (background subtract + split + ID match) "
                      "-- works with 2 or 3 animals",
                      font=("Segoe UI", 7), foreground=MUTED, wraplength=290, justify="left").pack(
                anchor="w", pady=(4, 0))
        else:
            ttk.Label(analysis_frame, text="(2/3 animals: switch to Multi-Mouse Tracking above)",
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
        self.color_mode_var = tk.StringVar(value="gray")
        ttk.Radiobutton(color_frame, text="Grayscale (default -- faster)",
                        variable=self.color_mode_var, value="gray").pack(anchor="w")
        ttk.Radiobutton(color_frame, text="RGB / Color", variable=self.color_mode_var,
                        value="rgb").pack(anchor="w")
        ttk.Label(color_frame, text="(better when the animal's color contrasts with the "
                  "floor but brightness doesn't)", font=("Segoe UI", 7), foreground=MUTED,
                  wraplength=255, justify="left").pack(anchor="w", padx=(18, 0))

        def setting_field(parent, label_text, default):
            ttk.Label(parent, text=label_text, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))
            e = ttk.Entry(parent)
            e.insert(0, str(default))
            e.pack(fill="x")
            return e

        default_min_area = 15 if mode == "standard" else 150
        default_max_area = 5000 if mode == "standard" else 1200

        self.bg_samples_entry = setting_field(settings_frame, "Background samples (default 100)", 100)
        self.threshold_entry = setting_field(settings_frame, "Difference threshold (start 25)", 25)
        self.min_area_entry = setting_field(settings_frame, "Min mouse area, px (keep LOW)", default_min_area)
        self.max_area_entry = setting_field(
            settings_frame,
            "Max object area, px" if mode == "standard" else "Max object area, px (~1 mouse; tune this)",
            default_max_area
        )
        self.max_jump_entry = setting_field(settings_frame, "Max movement / frame, px", 100)

        self.use_window_var = tk.BooleanVar(value=False)
        self.use_zone_threshold_var = tk.BooleanVar(value=False)
        self.reject_shadows_var = tk.BooleanVar(value=False)
        self.exclude_objects_var = tk.BooleanVar(value=True)
        self.real_distance_entry = None
        self.units_entry = None

        if mode == "standard":
            ttk.Checkbutton(settings_frame, text="Prior-position weighting", variable=self.use_window_var
                            ).pack(anchor="w", pady=(8, 0))
            self.window_size_entry = setting_field(settings_frame, "  window size, px", 120)
            self.window_weight_entry = setting_field(settings_frame, "  window weight (0-1)", 0.5)

            ttk.Checkbutton(settings_frame, text="Per-zone adaptive threshold", variable=self.use_zone_threshold_var
                            ).pack(anchor="w", pady=(8, 0))

            ttk.Checkbutton(settings_frame, text="Reject shadows", variable=self.reject_shadows_var
                            ).pack(anchor="w", pady=(8, 0))
            ttk.Label(settings_frame, text="(if the tracker keeps grabbing the animal's shadow "
                      "instead of its body, try this)", font=("Segoe UI", 7),
                      foreground=MUTED, wraplength=255, justify="left").pack(anchor="w", padx=(18, 0))

            polarity_frame = ttk.LabelFrame(settings_frame, text="Animal is...", padding=8)
            polarity_frame.pack(fill="x", pady=(10, 0))
            self.polarity_var = tk.StringVar(value="either")
            ttk.Radiobutton(polarity_frame, text="Either (default)", variable=self.polarity_var,
                            value="either").pack(anchor="w")
            ttk.Radiobutton(polarity_frame, text="Darker than the floor", variable=self.polarity_var,
                            value="darker").pack(anchor="w")
            ttk.Radiobutton(polarity_frame, text="Lighter than the floor", variable=self.polarity_var,
                            value="lighter").pack(anchor="w")
            ttk.Label(polarity_frame, text="If glare or a reflection off a glass/acrylic wall keeps "
                      "getting mistaken for the animal, picking the correct direction here makes that "
                      "artifact invisible to detection instead of competing with the real animal.",
                      font=("Segoe UI", 7), foreground=MUTED, wraplength=255,
                      justify="left").pack(anchor="w", pady=(4, 0))

            ttk.Checkbutton(settings_frame, text="Exclude marked objects from position detection",
                            variable=self.exclude_objects_var).pack(anchor="w", pady=(10, 0))
            ttk.Label(settings_frame, text="A marked object (e.g. a wire food-hopper cup) is a solid, "
                      "static obstacle -- excluding its own outline stops it from ever being mistaken "
                      "for the animal, without affecting whether the animal is detected as near it. "
                      "Turn this off only if you drew a larger buffer zone around the object rather "
                      "than tracing the object itself.",
                      font=("Segoe UI", 7), foreground=MUTED, wraplength=255,
                      justify="left").pack(anchor="w", padx=(18, 0))
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
        self.real_distance_entry.insert(0, "30")
        self.real_distance_entry.pack(fill="x")
        ttk.Label(settings_frame, text="Units (e.g. cm, mm, in)", font=("Segoe UI", 8)
                  ).pack(anchor="w", pady=(6, 0))
        self.units_entry = ttk.Entry(settings_frame)
        self.units_entry.insert(0, "cm")
        self.units_entry.pack(fill="x")

        self.preview_samples_entry = setting_field(settings_frame, "Preview frames to save/check", 6)

        # ---- BOTTOM: start + progress ----
        bottom = ttk.Frame(container, padding=10)
        bottom.pack(fill="x", side="bottom")
        ttk.Button(bottom, text="START TRACKING", style="Start.TButton",
                   command=self.on_start).pack(side="left", fill="x", expand=True, ipady=8)
        progress_col = ttk.Frame(bottom)
        progress_col.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.progress = ttk.Progressbar(progress_col, maximum=100, style="Green.Horizontal.TProgressbar")
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(progress_col, text="", font=("Segoe UI", 8), foreground=MUTED)
        self.progress_label.pack(anchor="w")

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

        list_frame = ttk.Frame(left, relief="sunken", borderwidth=1)
        list_frame.pack(fill="x", pady=(6, 4))
        self.video_list_inner = ttk.Frame(list_frame)
        self.video_list_inner.pack(fill="x")
        self.video_mode_note = ttk.Label(left, text="", font=("Segoe UI", 7), foreground=MUTED)
        self.video_mode_note.pack(anchor="w")
        ttk.Label(left, text="(click a video to make it active for the preview)",
                  font=("Segoe UI", 7), foreground="#888888").pack(anchor="w")
        self._refresh_video_list()

        time_frame = ttk.LabelFrame(left, text="Time window", padding=8)
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
        self.roi_names_entry = tk.Entry(left)
        self.object_names_entry = tk.Entry(left)
        self.loc_var = tk.BooleanVar(value=False)
        self.interact_var = tk.BooleanVar(value=False)
        self.entries_var = tk.BooleanVar(value=False)
        self.altern_var = tk.BooleanVar(value=False)
        self.all_var = tk.BooleanVar(value=False)

        out_frame = ttk.Frame(left, padding=(0, 8, 0, 0))
        out_frame.pack(fill="x")
        ttk.Label(out_frame, text="Video output size:").pack(side="left")
        self.output_size_entry = ttk.Entry(out_frame, width=12)
        self.output_size_entry.pack(side="left", padx=(6, 0))

        beh_frame = ttk.LabelFrame(left, text="Behaviors to detect", padding=8)
        beh_frame.pack(fill="x", pady=(10, 0))
        self.behavior_rearing_var = tk.BooleanVar(value=True)
        self.behavior_grooming_var = tk.BooleanVar(value=True)
        self.behavior_locomotion_var = tk.BooleanVar(value=False)
        self.behavior_immobile_var = tk.BooleanVar(value=False)
        self.behavior_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(beh_frame, text="Rearing", variable=self.behavior_rearing_var).pack(anchor="w")
        ttk.Checkbutton(beh_frame, text="Grooming", variable=self.behavior_grooming_var).pack(anchor="w")
        ttk.Checkbutton(beh_frame, text="Locomotion", variable=self.behavior_locomotion_var).pack(anchor="w")
        ttk.Checkbutton(beh_frame, text="Immobile", variable=self.behavior_immobile_var).pack(anchor="w")
        ttk.Separator(beh_frame).pack(fill="x", pady=4)
        ttk.Checkbutton(beh_frame, text="All", variable=self.behavior_all_var,
                        command=self._on_behavior_all_toggle).pack(anchor="w")

        animals_frame = ttk.Frame(left, padding=(0, 10, 0, 0))
        animals_frame.pack(fill="x")
        ttk.Label(animals_frame, text="Animals in frame:", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        animals_row = ttk.Frame(animals_frame)
        animals_row.pack(fill="x", pady=(6, 0))
        self.num_animals = 1
        self.num_animals_buttons = {}
        for n in (1, 2, 3):
            b = ttk.Button(animals_row, text=str(n), width=3,
                           style=("Accent.TButton" if n == 1 else "Tool.TButton"),
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
        self.color_mode_var = tk.StringVar(value="gray")
        ttk.Radiobutton(color_frame, text="Grayscale (default -- faster)",
                        variable=self.color_mode_var, value="gray").pack(anchor="w")
        ttk.Radiobutton(color_frame, text="RGB / Color", variable=self.color_mode_var,
                        value="rgb").pack(anchor="w")
        ttk.Label(color_frame, text="(better when the animal's color contrasts with the "
                  "floor but brightness doesn't)", font=("Segoe UI", 7), foreground=MUTED,
                  wraplength=255, justify="left").pack(anchor="w", padx=(18, 0))

        def setting_field(parent, label_text, default):
            ttk.Label(parent, text=label_text, font=("Segoe UI", 8)).pack(anchor="w", pady=(6, 0))
            e = ttk.Entry(parent)
            e.insert(0, str(default))
            e.pack(fill="x")
            return e

        self.bg_samples_entry = setting_field(settings_frame, "Background samples (default 100)", 100)
        self.threshold_entry = setting_field(settings_frame, "Difference threshold (start 25)", 25)
        self.min_area_entry = setting_field(settings_frame, "Min mouse area, px (keep LOW)", 15)
        self.max_area_entry = setting_field(settings_frame, "Max object area, px", 5000)
        self.max_jump_entry = setting_field(settings_frame, "Max movement / frame, px", 100)
        self.use_window_var = tk.BooleanVar(value=False)
        self.use_zone_threshold_var = tk.BooleanVar(value=False)
        self.window_size_entry = None
        self.window_weight_entry = None
        self.real_distance_entry = None
        self.units_entry = None
        self.preview_samples_entry = setting_field(settings_frame, "Preview frames to save/check", 6)

        ttk.Separator(settings_frame).pack(fill="x", pady=8)
        beh_settings = ttk.LabelFrame(settings_frame, text="Behavior Classification thresholds", padding=8)
        beh_settings.pack(fill="x")
        self.loco_thresh_entry = setting_field(beh_settings, "Locomotion threshold (px/s)", 40)
        self.rear_thresh_entry = setting_field(beh_settings, "Rearing area-drop threshold", 0.7)
        self.groom_thresh_entry = setting_field(beh_settings, "Grooming motion threshold", 3.0)
        self.immobile_thresh_entry = setting_field(beh_settings, "Immobile motion threshold", 1.0)
        self.min_bout_entry = setting_field(beh_settings, "Min bout duration (s)", 0.3)

        # ---- BOTTOM: start + progress ----
        bottom = ttk.Frame(container, padding=10)
        bottom.pack(fill="x", side="bottom")
        ttk.Button(bottom, text="START TRACKING", style="Start.TButton",
                   command=self.on_start).pack(side="left", fill="x", expand=True, ipady=8)
        progress_col = ttk.Frame(bottom)
        progress_col.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.progress = ttk.Progressbar(progress_col, maximum=100, style="Green.Horizontal.TProgressbar")
        self.progress.pack(fill="x")
        self.progress_label = ttk.Label(progress_col, text="", font=("Segoe UI", 8), foreground=MUTED)
        self.progress_label.pack(anchor="w")

        self._last_built_mode = "behavior"

    def _on_behavior_all_toggle(self):
        val = self.behavior_all_var.get()
        for var in (self.behavior_rearing_var, self.behavior_grooming_var,
                    self.behavior_locomotion_var, self.behavior_immobile_var):
            var.set(val)

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
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = (frame_count / fps) if fps > 0 else 0
            cap.release()
            self.videos.append({"path": path, "fps": fps, "frame_count": frame_count, "duration": duration})

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
        self.active_index = index
        self._reset_calibration_state()
        self._refresh_video_list()
        self._render_canvas()

    def _refresh_video_list(self):
        self._update_video_mode_note()
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
        self.start_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)
        self.roi_names_entry.delete(0, tk.END)
        self.object_names_entry.delete(0, tk.END)
        self.output_size_entry.delete(0, tk.END)
        self.interact_var.set(False); self.entries_var.set(False)
        self.altern_var.set(False); self.all_var.set(False)
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
            entry.delete(0, tk.END); entry.insert(0, default)
        self.use_window_var.set(False)
        self.use_zone_threshold_var.set(False)
        self.reject_shadows_var.set(False)
        self.polarity_var.set("either")
        self.exclude_objects_var.set(True)

    def on_reset_all(self):
        if not messagebox.askyesno("Reset everything",
                                   "Reset all 3 panels back to defaults? This clears the video "
                                   "queue, calibration, and settings."):
            return
        self.videos = []
        self.active_index = None
        self._remembered_start = ""
        self._remembered_end = ""
        self._remembered_roi_names = ""
        self.start_entry.delete(0, tk.END)
        self.end_entry.delete(0, tk.END)
        self.roi_names_entry.delete(0, tk.END)
        self.object_names_entry.delete(0, tk.END)
        self.output_size_entry.delete(0, tk.END)
        self.interact_var.set(False); self.entries_var.set(False)
        self.altern_var.set(False); self.all_var.set(False)
        self._reset_calibration_state()
        defaults = [
            (self.bg_samples_entry, "100"), (self.threshold_entry, "25"),
            (self.min_area_entry, "15"), (self.max_area_entry, "5000"),
            (self.max_jump_entry, "100"), (self.window_size_entry, "120"),
            (self.window_weight_entry, "0.5"), (self.real_distance_entry, "30"),
            (self.units_entry, "cm"), (self.preview_samples_entry, "6"),
        ]
        for entry, default in defaults:
            entry.delete(0, tk.END); entry.insert(0, default)
        self.use_window_var.set(False)
        self.use_zone_threshold_var.set(False)
        self.reject_shadows_var.set(False)
        self.polarity_var.set("either")
        self.exclude_objects_var.set(True)
        self._refresh_video_list()
        self._render_canvas()
        self.status_label.config(text="Idle.")
        self.progress["value"] = 0
        self.progress_label.config(text="")

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

        self._op = None
        for child in self.op_bar.winfo_children():
            child.destroy()
        self._render_canvas()

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
                "behavior_names": [],
                "background_samples": int(float(self.bg_samples_entry.get())),
                "threshold": float(self.threshold_entry.get()),
                "min_area": float(self.min_area_entry.get()),
                "max_area": float(self.max_area_entry.get()),
                "max_jump": float(self.max_jump_entry.get()),
                "use_zone_threshold": self.use_zone_threshold_var.get(),
                "reject_shadows": self.reject_shadows_var.get(),
                "polarity": self.polarity_var.get(),
                "exclude_objects": self.exclude_objects_var.get(),
                "use_window": self.use_window_var.get(),
                "window_size": float(self.window_size_entry.get()),
                "window_weight": float(self.window_weight_entry.get()),
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
        return setup

    def on_start(self):
        if not self.videos:
            messagebox.showerror("No video", "Add a video first.")
            return
        if self.active_index is None:
            messagebox.showerror("No video selected", "Click a video in the list to select it.")
            return

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
                return
            self.progress["value"] = 100
            self.status_label.config(text="Done.")
            self._show_results_standard(summary)
            return

        same_camera = ask_yes_no(
            "Camera position",
            f"Found {len(self.videos)} videos.\n\n"
            "Was the camera in EXACTLY the same position for all of them?\n\n"
            "Yes = reuse this calibration for every video (fast)\n"
            "No = re-select the arena/zones/objects for each one"
        )

        summaries = []
        for i, v in enumerate(self.videos):
            print(f"\n[{i + 1}/{len(self.videos)}] Processing: {v['path']}")
            if i == 0 or same_camera:
                setup = dict(base_setup)
                setup["video_path"] = v["path"]
            else:
                setup = recalibrate_spatial_only(v["path"], base_setup)

            try:
                summary = process_single_video(v["path"], setup, show_display=True,
                                               progress_callback=self.on_progress)
                summaries.append(summary)
            except SystemExit as exc:
                print(f"  Skipped: {exc}")
                continue

        self.progress["value"] = 100

        if summaries:
            batch_df = pd.DataFrame(summaries)
            folder = os.path.dirname(self.videos[0]["path"])
            batch_path = os.path.join(folder, "BatchSummary.csv")
            batch_df.to_csv(batch_path, index=False)
            self.status_label.config(text="Batch complete.")
            messagebox.showinfo(
                "Batch Complete",
                f"Processed {len(summaries)}/{len(self.videos)} videos.\n\nBatch summary:\n{batch_path}"
            )
        else:
            self.status_label.config(text="Batch: no videos processed.")

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

        self.status_label.config(text="Saving reference frames...")
        self.root.update_idletasks()
        try:
            preview_paths = save_preview_frames_multi_mouse(
                source_path, output_dir, n_samples=preview_n,
                num_animals=self.num_animals, min_area=min_area, max_area=max_area,
                diff_threshold=threshold, n_background_samples=bg_samples,
                color_mode=self.color_mode_var.get(), progress_callback=self.on_progress,
            )
        except Exception as exc:
            messagebox.showerror("Reference frames failed", str(exc))
            preview_paths = []

        self.status_label.config(text="Tracking (multi-mouse)...")
        self.root.update_idletasks()

        try:
            tracks_df = track_video(
                source_path, tracks_csv, annotate_path=annotate_path,
                num_animals=self.num_animals, min_area=min_area, max_area=max_area,
                diff_threshold=threshold, n_background_samples=bg_samples,
                progress_callback=self.on_progress, show_display=True,
                color_mode=self.color_mode_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Tracking failed", str(exc))
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        # Per-mouse zone/object/distance stats -- reuses the exact same
        # mask/membership/bout machinery Standard Tracking uses, applied to
        # each mouse_id's own (x, y) trajectory in turn, so zones/objects/
        # distance calibration drawn on the Preview panel actually work here
        # too, not just for single-animal tracking.
        object_names = [n.strip() for n in self.object_names_entry.get().split(",") if n.strip()] \
            if self.interact_var.get() else []
        active_object_points = {k: v for k, v in self.pending_object_points.items() if k in object_names} \
            if object_names else {}
        try:
            min_bout_s = float(self.min_bout_entry.get()) if getattr(self, "min_bout_entry", None) else 0.3
        except ValueError:
            min_bout_s = 0.3

        per_mouse_summary = {}
        per_mouse_bouts = {}
        for mouse_id, sub in tracks_df.groupby("mouse_id"):
            sub = sub.sort_values("frame").reset_index(drop=True)
            summary, bouts = compute_zone_interaction_stats(
                sub, self.pending_roi_points if self.loc_var.get() else {}, active_object_points,
                self.pending_warp_w, self.pending_warp_h, fps, min_bout_s=min_bout_s,
                scale_factor=self.pending_scale_factor, scale_unit=self.pending_scale_unit,
            )
            per_mouse_summary[mouse_id] = summary
            per_mouse_bouts[mouse_id] = bouts

        summary_csv = None
        bouts_csv = None
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

        self.progress["value"] = 100
        self.status_label.config(text="Done.")

        results = {
            "analysis_type": "multi_mouse", "video_path": active_path, "output_dir": output_dir,
            "tracks_df": tracks_df, "tracks_csv": tracks_csv, "annotate_path": annotate_path,
            "fps": fps, "per_mouse_summary": per_mouse_summary, "per_mouse_bouts": per_mouse_bouts,
            "summary_csv": summary_csv, "bouts_csv": bouts_csv, "preview_paths": preview_paths,
        }
        self._show_results_multi_mouse(results)

    # ------------------------------------------------------------------
    # Behavior Classification
    # ------------------------------------------------------------------

    def _run_behavior_flow(self):
        if self.pending_matrix is None:
            messagebox.showerror("Arena not set", "Use 'Crop Arena' or 'No Crop' first.")
            return

        active_path = self.videos[self.active_index]["path"]

        try:
            min_area = int(float(self.min_area_entry.get()))
            max_area = int(float(self.max_area_entry.get()))
            threshold = int(float(self.threshold_entry.get()))
            bg_samples = int(float(self.bg_samples_entry.get()))
            loco_thresh = float(self.loco_thresh_entry.get())
            rear_thresh = float(self.rear_thresh_entry.get())
            groom_thresh = float(self.groom_thresh_entry.get())
            immobile_thresh = float(self.immobile_thresh_entry.get())
            min_bout_s = float(self.min_bout_entry.get())
        except ValueError:
            messagebox.showerror("Invalid setting", "Check the Detection Settings values.")
            return

        self.progress["value"] = 0
        self.status_label.config(text="Preparing video...")
        self.root.update_idletasks()

        source_path, tmp_path, fps = self._prepare_source_video(active_path)
        output_dir = compute_output_dir(active_path)
        features_csv = os.path.join(output_dir, "features.csv")
        bouts_csv = os.path.join(output_dir, "bouts.csv")
        labeled_csv = os.path.join(output_dir, "labeled_frames.csv")

        try:
            preview_n = int(float(self.preview_samples_entry.get()))
        except ValueError:
            preview_n = 6

        self.status_label.config(text="Saving reference frames...")
        self.root.update_idletasks()
        try:
            preview_paths = save_preview_frames_behavior(
                source_path, output_dir, n_samples=preview_n, num_animals=self.num_animals,
                min_area=min_area, max_area=max_area, diff_threshold=threshold,
                n_background_samples=bg_samples, color_mode=self.color_mode_var.get(),
                progress_callback=self.on_progress,
            )
        except Exception as exc:
            messagebox.showerror("Reference frames failed", str(exc))
            preview_paths = []

        self.status_label.config(text="Extracting features...")
        self.root.update_idletasks()

        try:
            extract_features(
                source_path, features_csv, num_animals=self.num_animals,
                min_area=min_area, max_area=max_area, diff_threshold=threshold,
                n_background_samples=bg_samples, color_mode=self.color_mode_var.get(),
                progress_callback=self.on_progress,
            )
            self.status_label.config(text="Classifying behaviors...")
            self.root.update_idletasks()
            labeled_df, bouts_df = classify_behaviors(
                features_csv, bouts_csv, labeled_output_csv=labeled_csv,
                loco_thresh=loco_thresh, rear_area_drop=rear_thresh,
                groom_motion_thresh=groom_thresh, immobile_motion_thresh=immobile_thresh,
                min_bout_s=min_bout_s,
            )
        except Exception as exc:
            messagebox.showerror("Classification failed", str(exc))
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        selected = {
            "rearing": self.behavior_rearing_var.get(), "grooming": self.behavior_grooming_var.get(),
            "locomotion": self.behavior_locomotion_var.get(), "immobile": self.behavior_immobile_var.get(),
        }
        if not any(selected.values()) or self.behavior_all_var.get():
            selected = {k: True for k in selected}

        self.progress["value"] = 100
        self.status_label.config(text="Done.")

        results = {
            "analysis_type": "behavior", "video_path": active_path, "output_dir": output_dir,
            "labeled_df": labeled_df, "bouts_df": bouts_df, "selected_behaviors": selected,
            "bouts_csv": bouts_csv, "labeled_csv": labeled_csv, "fps": fps, "preview_paths": preview_paths,
        }
        self._show_results_behavior(results)

    # ------------------------------------------------------------------
    # Results screens
    # ------------------------------------------------------------------

    def _results_header(self, title_suffix):
        for child in self.body_container.winfo_children():
            child.destroy()

        top = ttk.Frame(self.body_container, padding=(0, 0, 0, 10))
        top.pack(fill="x")
        ttk.Label(top, text="Results: " + title_suffix, font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Button(top, text="Open Results Folder",
                  command=lambda: self._open_folder(self._last_output_dir)).pack(side="right")
        ttk.Button(top, text="New Analysis", style="Accent.TButton",
                  command=self._back_to_setup).pack(side="right", padx=(0, 8))
        ttk.Button(top, text="Generate BORIS Events",
                  command=self._on_generate_boris_events).pack(side="right", padx=(0, 8))
        ttk.Separator(self.body_container).pack(fill="x", pady=(0, 10))

    def _on_generate_boris_events(self):
        """BORIS (Behavioral Observation Research Interactive Software) is
        a widely used tool for coding behavioral events; this writes a
        simple subject/behavior/start/end event table many labs' further
        analysis expects, generated from whichever tracking CSV this run
        produced -- using the classified behavior label directly when one
        exists (Behavior Classification), or movement speed and zone/
        object proximity otherwise (Standard/Multi-Mouse Tracking)."""
        output_dir = self._last_output_dir
        candidates = ["labeled_frames.csv", "tracks.csv", "raw_tracking.csv"]
        csv_path = None
        for name in candidates:
            p = os.path.join(output_dir, name)
            if os.path.isfile(p):
                csv_path = p
                break
        if csv_path is None:
            messagebox.showerror("No tracking data", "Could not find a tracking CSV to generate events from.")
            return
        try:
            rows = read_tracking_csv(csv_path)
            events = generate_events(rows)
            if not events:
                messagebox.showinfo(
                    "BORIS Events",
                    "No events were generated -- the video may be too short, or nothing crossed the "
                    "movement/freezing thresholds."
                )
                return
            out_path = os.path.join(output_dir, "boris_events.csv")
            save_events(events, out_path)
            messagebox.showinfo(
                "BORIS Events",
                f"Saved {len(events)} events to:\n{out_path}\n\n"
                "These are a starting point, not a final coding -- review them in BORIS or a "
                "spreadsheet before treating them as ground truth."
            )
        except Exception as exc:
            messagebox.showerror("BORIS Events failed", str(exc))

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
        lf = ttk.LabelFrame(left, text="Summary", padding=10)
        lf.pack(fill="x")
        stats = [("Tracking quality", f"{summary.get('Tracking_quality_percent', 0):.1f} %"),
                 ("Total transitions", str(summary.get("Total_transitions", 0))),
                 ("Total distance (px)", f"{summary.get('Total_distance_pixels', 0):.1f}")]
        for k, v in summary.items():
            if k.endswith("_time_s"):
                stats.append((k.replace("_time_s", " time (s)"), f"{v:.1f}"))
        for k, v in stats:
            ttk.Label(lf, text=k, foreground=self._palette["MUTED"], font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 0))
            ttk.Label(lf, text=v, font=("Segoe UI", 13, "bold")).pack(anchor="w")

        center = ttk.Frame(body, padding=(14, 0))
        center.pack(side="left", fill="both", expand=True)
        ttk.Label(center, text="Trajectory", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        img_canvas = tk.Canvas(center, bg="#dddddd", highlightthickness=0, height=340)
        img_canvas.pack(fill="x", pady=(6, 0))
        traj_path = os.path.join(self._last_output_dir, "trajectory.png")
        self._results_photo = None
        if os.path.exists(traj_path):
            frame = cv2.imread(traj_path)
            if frame is not None:
                img_canvas.update_idletasks()
                cw, ch = max(300, img_canvas.winfo_width()), max(200, img_canvas.winfo_height())
                fh, fw = frame.shape[:2]
                scale = min(cw / fw, ch / fh, 1.0)
                disp = cv2.resize(frame, (max(1, int(fw * scale)), max(1, int(fh * scale))))
                self._results_photo = self._bgr_to_photoimage(disp)
                img_canvas.create_image(0, 0, anchor="nw", image=self._results_photo)

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
        lf = ttk.LabelFrame(left, text="Summary", padding=10)
        lf.pack(fill="x")
        duration_s = df["time_s"].max() if len(df) else 0
        ttk.Label(lf, text="Duration analyzed", foreground=self._palette["MUTED"], font=("Segoe UI", 9)).pack(
            anchor="w", pady=(6, 0))
        ttk.Label(lf, text=f"{duration_s:.1f} s", font=("Segoe UI", 13, "bold")).pack(anchor="w")

        colors = ["#c0392b", "#2f6fb0", "#27ae60"]
        mouse_ids = sorted(df["mouse_id"].unique())

        tf = ttk.LabelFrame(left, text="Per-Mouse Tracking", padding=10)
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
            zf = ttk.LabelFrame(left, text="Per-Mouse Zones / Distance", padding=10)
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
        for f in [os.path.basename(results["bouts_csv"]), os.path.basename(results["labeled_csv"])]:
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
