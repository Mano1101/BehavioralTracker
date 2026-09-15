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
    result = messagebox.askyesnocancel(
        "Mode",
        "Run in BATCH mode and process every video in a folder?\n\n"
        "Yes = Batch folder\n"
        "No = Single video"
    )
    root.destroy()
    return "batch" if result else "single"


def list_videos(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS))
    return [os.path.join(folder, f) for f in files]


# -------- -------- --------
# Step-back wizard
# -------- -------- --------
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

        result = confirm_step(label)

        if result == "accept":
            i += 1
        elif result == "redo":
            pass
        elif result == "back":
            i -= 1
            if i < 0:
                i = 0


def select_four_points(frame, window_title="Click the 4 arena corners"):
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_title, display)

    display = frame.copy()
    cv2.imshow(window_title, display)
    cv2.setMouseCallback(window_title, on_mouse)

    print("Click the 4 corners of the arena (top-left, top-right, bottom-right, bottom-left).")
    while len(points) < 4:
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window_title)
            return None

    cv2.destroyWindow(window_title)
    return np.array(points, dtype=np.float32)


def edit_regions_interactive(frame, roi_names):
    """
    Let user draw/edit polygons for each ROI name. Returns a dict:
      {name: [(x0, y0), (x1, y1), ...], ...}
    """
    regions = {name: [] for name in roi_names}
    current_region = [roi_names[0]] if roi_names else []
    drawing = [False]
    temp_points = []

    def find_vertex_near(x, y):
        for name, pts in regions.items():
            for j, (px, py) in enumerate(pts):
                if math.hypot(x - px, y - py) < 10:
                    return (name, j)
        return None

    def on_mouse(event, x, y, flags, param):
        nonlocal temp_points

        if not current_region:
            return

        name = current_region[0]
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
            temp_points = []
            vertex = find_vertex_near(x, y)
            if vertex:
                # Start dragging a vertex
                temp_points = [vertex]
            else:
                # Add new point
                regions[name].append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and drawing[0]:
            if temp_points and isinstance(temp_points[0], tuple):
                name_v, j = temp_points[0]
                regions[name_v][j] = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
            temp_points = []

    display = frame.copy()
    cv2.imshow("Edit Regions", display)
    cv2.setMouseCallback("Edit Regions", on_mouse)

    print(f"Draw polygons for: {', '.join(roi_names)}")
    print("  - Left-click to add/drag vertices")
    print("  - Keys: [1-9] switch region, [d] delete last vertex, [c] clear region, [Enter] done")

    while True:
        display = frame.copy()
        color_idx = 0
        for name, pts in regions.items():
            if pts:
                color = tuple(ROI_COLORS[color_idx % len(ROI_COLORS)])
                pts_arr = np.array(pts, dtype=np.int32)
                cv2.polylines(display, [pts_arr], False, color, 2)
                for px, py in pts:
                    cv2.circle(display, (px, py), 3, color, -1)
            color_idx += 1

        if current_region:
            cv2.putText(
                display, f"Current: {current_region[0]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

        cv2.imshow("Edit Regions", display)
        key = cv2.waitKey(100) & 0xFF

        if key == 13:  # Enter
            break
        elif 49 <= key <= 57:  # 1-9
            idx = key - 49
            if idx < len(roi_names):
                current_region = [roi_names[idx]]
        elif key == ord("d"):
            if current_region and regions[current_region[0]]:
                regions[current_region[0]].pop()
        elif key == ord("c"):
            if current_region:
                regions[current_region[0]] = []
        elif key == 27:  # ESC
            cv2.destroyWindow("Edit Regions")
            return None

    cv2.destroyWindow("Edit Regions")
    return regions


def select_object_point(frame, name):
    """Let user click once to mark an object location."""
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) == 0:
            points.append([x, y])
            cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
            cv2.imshow(f"Select {name}", display)

    display = frame.copy()
    cv2.imshow(f"Select {name}", display)
    cv2.setMouseCallback(f"Select {name}", on_mouse)

    print(f"Click to mark the center of '{name}'.")
    while len(points) == 0:
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(f"Select {name}")
            return None

    cv2.destroyWindow(f"Select {name}")
    return tuple(points[0])


def select_two_points(frame, window_title="Click 2 points of known distance"):
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append([x, y])
            cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(display, str(len(points)), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow(window_title, display)

    display = frame.copy()
    cv2.imshow(window_title, display)
    cv2.setMouseCallback(window_title, on_mouse)

    print("Click 2 points of known distance (e.g., two ends of a ruler, or opposite walls).")
    while len(points) < 2:
        key = cv2.waitKey(100) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(window_title)
            return None

    cv2.destroyWindow(window_title)
    return np.array(points, dtype=np.float32)


def get_scale_factor(frame):
    """
    User marks 2 points on frame, then enters the known physical distance
    between them (in cm or whatever unit). Returns pixels/unit.
    """
    points = select_two_points(frame)
    if points is None:
        return None

    pixel_distance = point_distance(points[0], points[1])
    print(f"Pixel distance: {pixel_distance:.1f}")

    try:
        physical_distance = ask_number("Scale calibration", "Enter the known distance (in cm):", 10.0)
    except SystemExit:
        return None

    if physical_distance <= 0:
        messagebox.showerror("Error", "Distance must be > 0.")
        return None

    scale = pixel_distance / physical_distance
    print(f"Scale factor: {scale:.2f} pixels/cm")
    return scale


def calibrate(video_path, include_interactions=True):
    """
    Run the full calibration wizard. Returns a setup dict.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame.")

    h, w = frame.shape[:2]

    state = {
        "video_path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "first_frame": frame,
        "start_time": None,
        "end_time": None,
        "arena_pts": None,
        "roi_names": [],
        "roi_points": {},
        "object_names": [],
        "object_points": {},
        "behavior_names": [],
        "scale_factor": None,
        "window_baseline_s": 10.0,
        "detection_threshold": 20,
        "min_area": 50,
        "max_area": 5000,
        "morph_kernel": 5,
        "matrix": identity_transform(w, h),
        "warp_w": w,
        "warp_h": h,
    }

    def step_time_window(state):
        """Ask user for start and end times (no prefilled defaults)."""
        try:
            state["start_time"] = ask_number("Time Window", "Start time (seconds):", "")
        except SystemExit:
            return

        try:
            state["end_time"] = ask_number("Time Window", "End time (seconds):", "")
        except SystemExit:
            return

        if state["end_time"] <= state["start_time"]:
            messagebox.showerror("Error", "End time must be after start time.")
            state["start_time"] = None
            state["end_time"] = None

    def step_arena(state):
        """Select the 4 corners of the arena."""
        pts = select_four_points(state["first_frame"], "Arena corners")
        if pts is not None:
            state["arena_pts"] = pts
            state["matrix"] = compute_perspective_transform(pts)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, frame = cap.read()
            state["first_frame"] = frame

    def step_zone_names(state):
        """Ask user for zone (ROI) names."""
        prompt = "Enter ROI names (comma-separated, e.g., 'zone1, zone2'):"
        names_str = ask_text("Zones", prompt, "")
        if names_str.strip():
            state["roi_names"] = [n.strip() for n in names_str.split(",") if n.strip()]
        else:
            messagebox.showwarning("Warning", "No ROI names entered.")

    def step_zone_polygons(state):
        """Draw polygons for each zone."""
        if not state["roi_names"]:
            messagebox.showwarning("Warning", "No zones defined yet.")
            return
        state["roi_points"] = edit_regions_interactive(state["first_frame"], state["roi_names"])

    def step_object_names(state):
        """Ask user for object names."""
        prompt = "Enter object names (comma-separated, e.g., 'object1, object2'):"
        names_str = ask_text("Objects", prompt, "")
        if names_str.strip():
            state["object_names"] = [n.strip() for n in names_str.split(",") if n.strip()]
        else:
            state["object_names"] = []

    def step_object_points(state):
        """Click the center point of each object."""
        if not state["object_names"]:
            return
        for name in state["object_names"]:
            pt = select_object_point(state["first_frame"], name)
            if pt:
                state["object_points"][name] = {"center": pt, "radius": 50}

    def step_behavior_names(state):
        """Ask user for behavior names (if applicable)."""
        prompt = "Enter behavior names (comma-separated, or leave blank):"
        names_str = ask_text("Behaviors", prompt, "")
        if names_str.strip():
            state["behavior_names"] = [n.strip() for n in names_str.split(",") if n.strip()]
        else:
            state["behavior_names"] = []

    def step_detection_settings(state):
        """Tune detection parameters."""
        state["detection_threshold"] = ask_number("Detection", "Diff threshold (0-255):", 20)
        state["min_area"] = ask_number("Detection", "Min blob area (pixels):", 50)
        state["max_area"] = ask_number("Detection", "Max blob area (pixels):", 5000)
        state["morph_kernel"] = int(ask_number("Detection", "Morphology kernel (odd):", 5))

    def step_window_weighting(state):
        """Ask for baseline window."""
        state["window_baseline_s"] = ask_number("Baseline", "Baseline window (seconds):", 10.0)

    def step_scale_calibration(state):
        """Optional: calibrate pixel -> cm conversion."""
        do_scale = ask_yes_no("Scale", "Calibrate scale factor (pixels to cm)?")
        if do_scale:
            scale = get_scale_factor(state["first_frame"])
            if scale:
                state["scale_factor"] = scale

    def step_preview_samples(state):
        """Show sample frames with detected objects."""
        if not state["arena_pts"] is None:
            print("Running preview detection...")
            # This would call a preview function

    steps = [
        ("Time Window", step_time_window),
        ("Arena", step_arena),
        ("Zone Names", step_zone_names),
        ("Zone Polygons", step_zone_polygons),
        ("Object Names", step_object_names),
        ("Object Locations", step_object_points),
        ("Behavior Names", step_behavior_names),
        ("Detection Settings", step_detection_settings),
        ("Baseline Window", step_window_weighting),
        ("Scale Calibration", step_scale_calibration),
    ]

    run_wizard(steps, state)

    cap.release()

    return {
        "start_time": state["start_time"],
        "end_time": state["end_time"],
        "arena_pts": state["arena_pts"],
        "roi_names": state["roi_names"],
        "roi_points": state["roi_points"],
        "object_names": state["object_names"],
        "object_points": state["object_points"],
        "behavior_names": state["behavior_names"],
        "fps": fps,
        "total_frames": total_frames,
        "detection_threshold": state["detection_threshold"],
        "min_area": state["min_area"],
        "max_area": state["max_area"],
        "morph_kernel": state["morph_kernel"],
        "window_baseline_s": state["window_baseline_s"],
        "scale_factor": state["scale_factor"],
        "matrix": state["matrix"],
        "warp_w": state["warp_w"],
        "warp_h": state["warp_h"],
    }


def recalibrate_spatial_only(video_path, base_setup):
    """
    Re-run just the spatial calibration (arena + zones) keeping detection/behavior settings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    ret, frame = cap.read()
    if not ret:
        raise ValueError("Cannot read first frame.")

    state = {
        "first_frame": frame,
        "arena_pts": base_setup.get("arena_pts"),
        "roi_names": base_setup.get("roi_names", []),
        "roi_points": base_setup.get("roi_points", {}),
        "object_names": base_setup.get("object_names", []),
        "object_points": base_setup.get("object_points", {}),
        "matrix": base_setup.get("matrix", identity_transform(frame.shape[1], frame.shape[0])),
        "warp_w": base_setup.get("warp_w", frame.shape[1]),
        "warp_h": base_setup.get("warp_h", frame.shape[0]),
    }

    def step_arena(state):
        pts = select_four_points(state["first_frame"], "Arena corners")
        if pts is not None:
            state["arena_pts"] = pts
            state["matrix"] = compute_perspective_transform(pts)

    def step_zone_polygons(state):
        if not state["roi_names"]:
            messagebox.showwarning("Warning", "No zones defined.")
            return
        state["roi_points"] = edit_regions_interactive(state["first_frame"], state["roi_names"])

    def step_object_points(state):
        if not state["object_names"]:
            return
        state["object_points"] = {}
        for name in state["object_names"]:
            pt = select_object_point(state["first_frame"], name)
            if pt:
                state["object_points"][name] = {"center": pt, "radius": 50}

    steps = [
        ("Arena", step_arena),
        ("Zone Polygons", step_zone_polygons),
        ("Object Locations", step_object_points),
    ]

    run_wizard(steps, state)
    cap.release()

    return {
        **base_setup,
        "arena_pts": state["arena_pts"],
        "roi_points": state["roi_points"],
        "object_points": state["object_points"],
        "matrix": state["matrix"],
        "warp_w": state["warp_w"],
        "warp_h": state["warp_h"],
    }


SETUP_SAVE_KEYS = [
    "start_time", "end_time", "arena_pts", "roi_names", "roi_points",
    "object_names", "object_points", "behavior_names", "fps", "total_frames",
    "detection_threshold", "min_area", "max_area", "morph_kernel",
    "window_baseline_s", "scale_factor", "matrix", "warp_w", "warp_h",
]


def save_setup(setup, path):
    """Save setup to JSON, converting numpy arrays to lists."""
    data = {}
    for key in SETUP_SAVE_KEYS:
        val = setup.get(key)
        if isinstance(val, np.ndarray):
            data[key] = val.tolist()
        else:
            data[key] = val
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Setup saved to {path}")


def load_setup(path):
    """Load setup from JSON, converting lists back to numpy arrays."""
    with open(path) as f:
        data = json.load(f)
    if data.get("arena_pts"):
        data["arena_pts"] = np.array(data["arena_pts"], dtype=np.float32)
    if data.get("matrix"):
        data["matrix"] = np.array(data["matrix"], dtype=np.float32)
    print(f"Setup loaded from {path}")
    return data


def messagebox_summary(summary):
    """Display summary stats in a messagebox."""
    text = "\n".join(f"{k}: {v}" for k, v in sorted(summary.items()))
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("Summary", text)
    root.destroy()


def run_single():
    """Single video processing."""
    video_path = select_video()
    if not video_path:
        return

    print(f"\nSelected video: {video_path}")

    setup = calibrate(video_path)
    if not setup or setup.get("start_time") is None:
        print("Calibration incomplete or cancelled.")
        return

    output_dir = compute_output_dir(video_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing: {video_path}")
    summary = process_single_video(video_path, setup)
    messagebox_summary(summary)


def run_batch():
    """Batch video processing."""
    folder = select_folder()
    if not folder:
        return

    videos = list_videos(folder)
    if not videos:
        messagebox.showwarning("No videos", f"No video files found in {folder}")
        return

    print(f"\nFound {len(videos)} video(s) in {folder}")

    # Calibrate on the first video
    setup = calibrate(videos[0])
    if not setup or setup.get("start_time") is None:
        print("Calibration incomplete or cancelled.")
        return

    for i, video_path in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] Processing: {video_path}")
        output_dir = compute_output_dir(video_path)
        os.makedirs(output_dir, exist_ok=True)
        summary = process_single_video(video_path, setup)
        print(f"  Summary: {summary}")


def main():
    """Entry point: ask user for mode and run."""
    root = tk.Tk()
    root.withdraw()
    root.destroy()

    while True:
        mode = select_mode()
        if mode == "batch":
            run_batch()
        else:
            run_single()

        if not ask_yes_no("Continue", "Process another video?"):
            break


class TrackerApp:
    """Placeholder for the PySide6 GUI app (Step 2)."""
    pass


def launch_gui():
    """Placeholder for PySide6 launch."""
    print("GUI mode not yet implemented (Step 2).")
    main()


if __name__ == "__main__":
    launch_gui()
