"""
Core tracking engine: perspective rectification, zone/object masks,
background modeling, per-frame animal detection, zone transitions, the
calibration-preview renderer, and the main per-video pipeline
(process_single_video) that ties everything -- including the other
tracking/analysis/output modules -- together.
"""

import cv2
import numpy as np
import pandas as pd
import os
import glob
import math
from collections import Counter

from analysis.calculations import point_distance, safe_col
from tracking.interaction import extract_bouts
from tracking.epm import calculate_arm_entries, calculate_alternation
from output.csv import write_csv_report
from output.excel import write_excel_report

# ROI_COLORS/OBJECT_COLORS must be defined BEFORE the output.graphs import
# below: output/graphs.py imports them back from this module, and since
# Python runs a module top-to-bottom, they need to already exist on this
# (still-loading) module by the time that reverse import happens.
ROI_COLORS = [
    (0, 255, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 0),
    (180, 105, 255),
]

OBJECT_COLORS = [
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (0, 200, 120),
    (200, 0, 120),
]

from output.graphs import save_plots


def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype="float32")


def compute_perspective_transform(pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b), 2)

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b), 2)

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    return matrix, max_width, max_height


def identity_transform(width, height):
    """
    A perspective 'transform' that changes nothing -- used when the user
    opts out of the 4-point crop. Every downstream step (make_background,
    detect_mouse, the tracking loop, etc.) always expects a matrix plus a
    warp width/height, so this lets "no cropping" mean "warp with an
    identity matrix at the video's native size" instead of needing a
    separate code path everywhere a matrix is used.
    """
    return np.eye(3, dtype=np.float32), int(width), int(height)


def read_and_warp(cap, frame_number, matrix, out_w, out_h):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = cap.read()

    if not ok:
        return None

    return cv2.warpPerspective(frame, matrix, (out_w, out_h))


# -----------------------------
# Scale-bar overlay (25 / 50 / 75 / 100% on both axes)
# -----------------------------


def draw_scale_overlay(frame):
    disp = frame.copy()
    h, w = disp.shape[:2]
    color = (0, 200, 255)

    for frac in (0.25, 0.50, 0.75, 1.00):
        x = max(0, min(w - 1, int(w * frac) - 1))
        cv2.line(disp, (x, 0), (x, h), color, 1)
        cv2.putText(disp, f"{int(frac * 100)}%", (max(2, x - 24), 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        y = max(0, min(h - 1, int(h * frac) - 1))
        cv2.line(disp, (0, y), (w, y), color, 1)
        cv2.putText(disp, f"{int(frac * 100)}%", (4, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    return disp


# -----------------------------
# Named polygon zones
# -----------------------------


def build_roi_masks(shape_hw, roi_points):
    h, w = shape_hw
    masks = {}

    for name, pts in roi_points.items():
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
        masks[name] = mask > 0

    return masks


def roi_membership(masks, x, y):
    xi, yi = int(round(x)), int(round(y))
    membership = {}

    for name, mask in masks.items():
        h, w = mask.shape
        membership[name] = bool(0 <= yi < h and 0 <= xi < w and mask[yi, xi])

    return membership


def linearize_roi(membership, null_name="None"):
    active = [name for name, in_roi in membership.items() if in_roi]
    return "_".join(active) if active else null_name


def draw_roi_polygons(display, roi_points):
    for name, pts in roi_points.items():
        cv2.polylines(display, [np.array(pts, dtype=np.int32)], True, (0, 0, 0), 1, cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(display, name, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


# -----------------------------
# Interaction objects (named point + radius) -- new in V6
# -----------------------------


def draw_object_markers(display, object_points):
    """object_points[name] is a polygon (list of points), same shape as
    roi_points -- objects are marked by tracing their actual outline now,
    not a circle that may not match the object's real shape."""
    for i, (name, pts) in enumerate(object_points.items()):
        color = OBJECT_COLORS[i % len(OBJECT_COLORS)]
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.polylines(display, [pts_arr], True, color, 2, cv2.LINE_AA)
        cx = int(np.mean([p[0] for p in pts]))
        cy = int(np.mean([p[1] for p in pts]))
        cv2.putText(display, name, (cx - 15, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def compute_zone_interaction_stats(positions_df, roi_points, object_points,
                                    warp_w, warp_h, fps, min_bout_s=0.3,
                                    scale_factor=None, scale_unit=None):
    """
    Given ONE subject's trajectory (columns: frame, x, y -- e.g. one
    mouse_id's rows from a multi-animal tracker), compute per-zone occupied
    time, per-object interaction bouts, and total distance -- reusing the
    exact same mask/membership/bout machinery the single-animal pipeline
    uses, so multi-animal analysis types get the same zone/object/distance
    capability as Standard Tracking, not a separate re-implementation.

    Returns (summary_dict, bouts_list). bouts_list entries have the same
    shape as tracking.interaction.extract_bouts's output (Object,
    Frame_start/end, Start/End_seconds, Duration_seconds).
    """
    summary = {}
    xs = positions_df["x"].to_numpy(dtype=float)
    ys = positions_df["y"].to_numpy(dtype=float)
    frames = positions_df["frame"].to_numpy()

    dx = np.diff(xs)
    dy = np.diff(ys)
    seg = np.hypot(dx, dy)
    seg = seg[~np.isnan(seg)]
    dist_px = float(seg.sum())
    summary["Total_distance_pixels"] = dist_px
    if scale_factor:
        summary[f"Total_distance_{scale_unit or 'units'}"] = dist_px * scale_factor

    if roi_points:
        roi_masks = build_roi_masks((warp_h, warp_w), roi_points)
        for name in roi_points:
            in_zone = np.zeros(len(xs), dtype=bool)
            for i, (x, y) in enumerate(zip(xs, ys)):
                if not (np.isnan(x) or np.isnan(y)):
                    in_zone[i] = roi_membership(roi_masks, x, y).get(name, False)
            time_s = in_zone.sum() / fps
            summary[f"{safe_col(name)}_time_s"] = time_s
            total_s = len(xs) / fps
            summary[f"{safe_col(name)}_percent"] = (time_s / total_s * 100) if total_s > 0 else 0

    bouts = []
    if object_points:
        object_masks = build_roi_masks((warp_h, warp_w), object_points)
        near_cols = {name: np.zeros(len(xs), dtype=bool) for name in object_points}
        for i, (x, y) in enumerate(zip(xs, ys)):
            if not (np.isnan(x) or np.isnan(y)):
                membership = roi_membership(object_masks, x, y)
                for name in object_points:
                    near_cols[name][i] = membership.get(name, False)
        bout_df = pd.DataFrame({"Frame": frames})
        for name in object_points:
            bout_df[f"Near_{safe_col(name)}"] = near_cols[name]
        bouts = extract_bouts(bout_df, list(object_points.keys()), fps, min_duration=min_bout_s)
        for name in object_points:
            near_time = near_cols[name].sum() / fps
            summary[f"{safe_col(name)}_near_time_s"] = near_time
            summary[f"{safe_col(name)}_bout_count"] = sum(1 for b in bouts if b["Object"] == name)

    return summary, bouts


def draw_crosshair(display, center, size=9, color=(0, 0, 255), thickness=2):
    """A medium '+' marker for the tracked point, instead of a filled circle
    that can hide exactly where the center is."""
    x, y = int(center[0]), int(center[1])
    cv2.line(display, (x - size, y), (x + size, y), color, thickness, cv2.LINE_AA)
    cv2.line(display, (x, y - size), (x, y + size), color, thickness, cv2.LINE_AA)


def draw_mask_polygons(display, mask_polygons, color=(0, 0, 0)):
    """Draw excluded (masked-out) regions as black outlines with a light
    hatch fill, so they're visibly distinct from zones/objects."""
    overlay = display.copy()
    for pts in mask_polygons:
        pts_arr = np.array(pts, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_arr], (60, 60, 60))
        cv2.polylines(display, [pts_arr], True, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, display, 0.65, 0, dst=display)


def build_exclusion_mask(shape_hw, mask_polygons):
    """Boolean array, True where pixels should be excluded from detection
    entirely (e.g. a food hopper, cage wire, reflection)."""
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in mask_polygons:
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    return mask > 0


# -----------------------------
# Real-world distance calibration
# -----------------------------


def make_background(cap, start_frame, end_frame, matrix, warp_w, warp_h, sample_count=100, color_mode="gray"):
    sample_count = max(1, int(sample_count))

    if end_frame <= start_frame:
        sample_count = 1

    positions = np.linspace(start_frame, max(start_frame, end_frame - 1), sample_count).astype(int)
    samples = []

    print("\nBuilding automatic background model...")
    print(f"Sampling {len(positions)} frames (median-based). This may take a little time.")

    for frame_number in positions:
        frame = read_and_warp(cap, frame_number, matrix, warp_w, warp_h)

        if frame is None:
            continue

        if color_mode == "rgb":
            sample = cv2.GaussianBlur(frame, (5, 5), 0)
        else:
            sample = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sample = cv2.GaussianBlur(sample, (5, 5), 0)
        samples.append(sample)

    if not samples:
        raise RuntimeError("Could not build background model.")

    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


# -----------------------------
# Automatic mouse detection
# -----------------------------


def _shadow_mask(current_bgr, background_bgr, v_ratio_range=(0.25, 0.92), hue_tol=25, sat_tol=60):
    """True where a pixel looks like a SHADOW rather than a genuine change:
    similar hue and saturation to the background at that spot, but
    consistently darker (lower brightness/V) within the ratio range real
    shadows typically fall in. This is the same hue/saturation/value-ratio
    idea OpenCV's own MOG2 background subtractor uses for shadow detection
    -- a shadow darkens the existing surface without changing its color,
    while a genuine animal usually differs in hue/saturation too, or is
    darkened well outside a shadow's typical ratio."""
    cur_hsv = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    bg_hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    v_bg = bg_hsv[..., 2]
    v_cur = cur_hsv[..., 2]
    v_ratio = np.divide(v_cur, v_bg, out=np.ones_like(v_cur), where=v_bg > 5)

    hue_diff = np.abs(cur_hsv[..., 0] - bg_hsv[..., 0])
    hue_diff = np.minimum(hue_diff, 180 - hue_diff)  # hue wraps at 180 in OpenCV's 0-179 range

    sat_diff = np.abs(cur_hsv[..., 1] - bg_hsv[..., 1])

    return (
        (v_ratio >= v_ratio_range[0]) & (v_ratio <= v_ratio_range[1])
        & (hue_diff <= hue_tol) & (sat_diff <= sat_tol)
    )


def _polarity_diff(current, background, polarity="either"):
    """Difference-from-background, but aware of which DIRECTION the animal
    is expected to differ in. A common source of false detections -- glare
    or a reflection off a glass/acrylic wall -- is BRIGHTER than the floor,
    while a dark-furred animal is always DARKER than it (and vice versa
    for a light-furred animal on a dark floor). Restricting to the
    direction the real animal is known to go makes an opposite-direction
    artifact invisible to detection entirely, instead of it competing with
    the real animal on equal footing the way an absolute difference does.
    polarity="either" reproduces the previous (direction-blind) behavior."""
    if polarity == "either":
        return cv2.absdiff(current, background)
    cur = current.astype(np.int16)
    bg = background.astype(np.int16)
    diff = (bg - cur) if polarity == "darker" else (cur - bg)
    return np.clip(diff, 0, 255).astype(np.uint8)


def detect_mouse(
    frame, background, arena, previous_point, previous_area,
    threshold, min_area, max_area, max_jump,
    roi_masks=None, use_window=False, window_size=120, window_weight=0.5,
    exclusion_mask=None, color_mode="gray", color_background=None, reject_shadows=False,
    polarity="either"
):
    x1, y1, x2, y2 = arena

    if color_mode == "rgb":
        current = cv2.GaussianBlur(frame, (5, 5), 0)
    else:
        current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(current, (5, 5), 0)

    bg_crop = background[y1:y2, x1:x2]
    current_crop = current[y1:y2, x1:x2]
    diff = _polarity_diff(current_crop, bg_crop, polarity).astype(np.float32)
    if color_mode == "rgb" and diff.ndim == 3:
        # Max across B/G/R rather than mean: catches an animal that stands
        # out strongly in just one channel (e.g. reddish fur on a green
        # floor) even when its overall grayscale brightness barely differs
        # from the background.
        diff = diff.max(axis=2)

    if exclusion_mask is not None:
        # Zero out masked-out regions (e.g. a food hopper, cage wire,
        # reflection) so they can never register as a detection candidate --
        # same technique ezTrack uses (dif[mask] = 0).
        diff[exclusion_mask[y1:y2, x1:x2]] = 0

    if use_window and previous_point is not None:
        px = previous_point[0] - x1
        py = previous_point[1] - y1
        half = max(1.0, window_size / 2.0)

        weights = np.full(diff.shape, max(0.0, 1.0 - window_weight), dtype=np.float32)

        wy1 = max(0, int(py - half))
        wy2 = min(diff.shape[0], int(py + half))
        wx1 = max(0, int(px - half))
        wx2 = min(diff.shape[1], int(px + half))

        if wy2 > wy1 and wx2 > wx1:
            weights[wy1:wy2, wx1:wx2] = 1.0

        diff = diff * weights

    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)

    if reject_shadows and color_background is not None:
        shadow = _shadow_mask(frame[y1:y2, x1:x2], color_background[y1:y2, x1:x2])
        diff_u8[shadow] = 0

    if roi_masks:
        mask = np.zeros(diff_u8.shape, dtype=np.uint8)
        assigned = np.zeros(diff_u8.shape, dtype=bool)

        for full_mask in roi_masks.values():
            zone_mask = full_mask[y1:y2, x1:x2] & (~assigned)
            pixels = diff_u8[zone_mask]

            if pixels.size == 0:
                continue

            otsu_val, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            zone_threshold = max(float(threshold), float(otsu_val))
            zone_binary = (diff_u8 >= zone_threshold).astype(np.uint8) * 255
            mask[zone_mask] = zone_binary[zone_mask]
            assigned |= zone_mask

        leftover = ~assigned
        leftover_pixels = diff_u8[leftover]

        if leftover_pixels.size:
            otsu_val, _ = cv2.threshold(leftover_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            zone_threshold = max(float(threshold), float(otsu_val))
            zone_binary = (diff_u8 >= zone_threshold).astype(np.uint8) * 255
            mask[leftover] = zone_binary[leftover]
    else:
        _, mask = cv2.threshold(diff_u8, int(threshold), 255, cv2.THRESH_BINARY)

    kernel_small = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < min_area or area > max_area:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        cx = (moments["m10"] / moments["m00"]) + x1
        cy = (moments["m01"] / moments["m00"]) + y1

        bx, by, bw, bh = cv2.boundingRect(contour)
        bx += x1
        by += y1

        if bw <= 0 or bh <= 0:
            continue

        aspect = max(bw, bh) / max(1, min(bw, bh))
        perimeter = cv2.arcLength(contour, True)
        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

        # Mean strength of the difference-from-background signal within
        # this candidate's own contour. A genuine, directly-lit animal
        # typically produces a much stronger, more solid signal than a
        # reflection (e.g. in a glass/acrylic wall or glossy floor), which
        # is dimmer and partially blended with whatever's behind the glass
        # -- even when the reflection's area and shape look similar.
        contour_mask = np.zeros(diff_u8.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        mean_intensity = float(cv2.mean(diff_u8, mask=contour_mask)[0])

        candidates.append({
            "center": (cx, cy),
            "area": float(area),
            "bbox": (bx, by, bw, bh),
            "aspect": float(aspect),
            "circularity": float(circularity),
            "distance": point_distance((cx, cy), previous_point),
            "mean_intensity": mean_intensity,
            "contour": contour
        })

    if not candidates:
        return None, mask

    best = None
    best_score = float("inf")

    for c in candidates:
        d = c["distance"]
        # Reward a stronger signal with a lower (better) score -- weighted
        # gently enough that spatial continuity still dominates once
        # tracking is established, but it breaks ties in favor of the
        # more solid detection when candidates are otherwise close.
        intensity_bonus = c["mean_intensity"] * 0.15

        if previous_point is not None:
            if d > max_jump:
                continue

            score = d - intensity_bonus

            if previous_area is not None and previous_area > 0:
                ratio = c["area"] / previous_area
                score += 30 * abs(math.log(max(ratio, 1e-6)))

            if c["aspect"] > 8:
                score += 20
            if c["circularity"] < 0.02:
                score += 10
        else:
            score = -intensity_bonus
            if c["aspect"] > 8:
                score += 30
            score -= min(c["area"], 1000) * 0.01

        if score < best_score:
            best_score = score
            best = c

    return best, mask


# -----------------------------
# Recovery using local search
# -----------------------------


def local_recovery(frame, background, previous_point, arena, threshold, min_area, max_area,
                    exclusion_mask=None, color_mode="gray", color_background=None, reject_shadows=False,
                    polarity="either"):
    if previous_point is None:
        return None

    cx, cy = map(int, previous_point)
    radius = 100

    x1 = max(arena[0], cx - radius)
    y1 = max(arena[1], cy - radius)
    x2 = min(arena[2], cx + radius)
    y2 = min(arena[3], cy + radius)

    if x2 <= x1 or y2 <= y1:
        return None

    if color_mode == "rgb":
        current = frame[y1:y2, x1:x2]
    else:
        current = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    bg = background[y1:y2, x1:x2]
    diff = _polarity_diff(current, bg, polarity)
    if color_mode == "rgb" and diff.ndim == 3:
        diff = diff.max(axis=2)

    if exclusion_mask is not None:
        diff[exclusion_mask[y1:y2, x1:x2]] = 0

    if reject_shadows and color_background is not None:
        shadow = _shadow_mask(frame[y1:y2, x1:x2], color_background[y1:y2, x1:x2])
        diff[shadow] = 0

    _, mask = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_d = float("inf")

    for c in contours:
        area = cv2.contourArea(c)

        if area < min_area or area > max_area:
            continue

        m = cv2.moments(c)

        if m["m00"] == 0:
            continue

        px = (m["m10"] / m["m00"]) + x1
        py = (m["m01"] / m["m00"]) + y1
        d = point_distance((px, py), previous_point)

        if d < best_d:
            best_d = d
            bx, by, bw, bh = cv2.boundingRect(c)
            best = {"center": (px, py), "area": float(area), "bbox": (bx + x1, by + y1, bw, bh)}

    return best


# -----------------------------
# Zone transitions / binning
# -----------------------------


def calculate_transitions(df, confirmation_frames=3):
    valid = df[df["Tracking_Status"] == "Tracked"]
    columns = ["From_ROI", "To_ROI", "Transition_seconds"]

    if valid.empty:
        return pd.DataFrame(columns=columns)

    events = []
    current_roi = None
    candidate_roi = None
    candidate_count = 0

    for _, row in valid.iterrows():
        roi = row["ROI"]
        t = float(row["Time_seconds"])

        if current_roi is None:
            current_roi = roi
            continue

        if roi == current_roi:
            candidate_roi = None
            candidate_count = 0
            continue

        if candidate_roi != roi:
            candidate_roi = roi
            candidate_count = 1
        else:
            candidate_count += 1

        if candidate_count >= confirmation_frames:
            events.append({"From_ROI": current_roi, "To_ROI": roi, "Transition_seconds": t})
            current_roi = roi
            candidate_roi = None
            candidate_count = 0

    return pd.DataFrame(events, columns=columns)


def run_detection_preview(
    cap, matrix, warp_w, warp_h, start_frame, end_frame,
    background, arena, roi_masks, roi_points, object_points,
    threshold, min_area, max_area, n_samples, output_dir, show_live=True,
    mask_polygons=None, exclusion_mask=None, color_mode="gray",
    color_background=None, reject_shadows=False, polarity="either"
):
    preview_dir = os.path.join(output_dir, "preview")
    os.makedirs(preview_dir, exist_ok=True)
    for stale in glob.glob(os.path.join(preview_dir, "preview_*.png")):
        os.remove(stale)

    n_samples = max(1, int(n_samples))
    last_valid_frame = max(start_frame, end_frame - 1)
    positions = np.linspace(start_frame, last_valid_frame, n_samples).astype(int)

    print(f"\nSaving {len(positions)} calibration preview frame(s) to:\n{preview_dir}")
    if show_live:
        print("Press any key to advance, ESC to cancel and retune.")

    for i, pframe in enumerate(positions, start=1):
        frame = read_and_warp(cap, pframe, matrix, warp_w, warp_h)

        if frame is None:
            continue

        candidate, _ = detect_mouse(
            frame, background, arena, None, None,
            threshold, max(1, int(min_area)), max(int(max_area), int(min_area) + 1),
            float("inf"), roi_masks=roi_masks, exclusion_mask=exclusion_mask, color_mode=color_mode,
            color_background=color_background, reject_shadows=reject_shadows, polarity=polarity
        )

        disp = frame.copy()
        cv2.rectangle(disp, (arena[0], arena[1]), (arena[2], arena[3]), (255, 255, 0), 1)
        if mask_polygons:
            draw_mask_polygons(disp, mask_polygons)
        draw_roi_polygons(disp, roi_points)
        draw_object_markers(disp, object_points)

        if candidate is not None:
            cx, cy = candidate["center"]
            draw_crosshair(disp, (cx, cy), size=9, color=(0, 0, 255), thickness=2)
            cv2.putText(
                disp, f"area={candidate['area']:.0f}",
                (max(5, int(cx) - 40), max(20, int(cy) - 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        else:
            cv2.putText(disp, "NO DETECTION", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.putText(
            disp, f"Preview {i}/{len(positions)}  frame {int(pframe)}",
            (20, disp.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )

        fname = os.path.join(preview_dir, f"preview_{i:02d}_frame{int(pframe)}.png")
        cv2.imwrite(fname, disp)

        if show_live:
            cv2.imshow("Detection preview", disp)
            key = cv2.waitKey(0) & 0xFF

            if key == 27:
                cv2.destroyWindow("Detection preview")
                raise SystemExit("Cancelled during preview. Rerun with adjusted threshold/min_area/max_area.")

    if show_live:
        cv2.destroyWindow("Detection preview")


# -----------------------------
# Calibration (step-back wizard)
# -----------------------------


def compute_output_dir(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), "results", video_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# -----------------------------
# Save/load a reusable setup (zones, objects, thresholds -- not the video
# itself, not per-video state like fps/duration/first_frame)
# -----------------------------


def process_single_video(video_path, setup, show_display=True, progress_callback=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    if setup.get("fps_override"):
        fps = float(setup["fps_override"])

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    start_time = max(0, min(setup["start_time"], duration))
    end_time = max(start_time, min(setup["end_time"], duration))
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    matrix = setup["matrix"]
    warp_w = setup["warp_w"]
    warp_h = setup["warp_h"]
    arena = setup["arena"]
    roi_points = setup["roi_points"]
    roi_names = setup["roi_names"]
    roi_masks = build_roi_masks((warp_h, warp_w), roi_points)

    # roi_masks (above) is ALWAYS used for location/zone membership -- which
    # zone(s) the animal is in, regardless of this setting. detection_roi_masks
    # is what actually reaches detect_mouse()'s per-zone Otsu split, and is
    # only non-None when the zones represent genuinely different lighting
    # (see step_detection_settings). Decoupling these two avoids an
    # overlapping analysis-only region (e.g. a "Whole_Arena" zone) from
    # starving the per-zone threshold split of pixels.
    use_zone_threshold = setup.get("use_zone_threshold", False)
    detection_roi_masks = roi_masks if use_zone_threshold else None

    object_points = setup.get("object_points", {})
    object_names = setup.get("object_names", [])
    object_masks = build_roi_masks((warp_h, warp_w), object_points) if object_points else {}
    behavior_names = setup.get("behavior_names", [])

    mask_polygons = setup.get("mask_points", [])
    exclusion_mask = build_exclusion_mask((warp_h, warp_w), mask_polygons) if mask_polygons else None

    threshold = setup["threshold"]
    min_area = setup["min_area"]
    max_area = setup["max_area"]
    max_jump = setup["max_jump"]
    use_window = setup["use_window"]
    window_size = setup["window_size"]
    window_weight = setup["window_weight"]

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if setup.get("output_dir_override"):
        output_dir = os.path.join(setup["output_dir_override"], video_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = compute_output_dir(video_path)

    color_mode = setup.get("color_mode", "gray")
    reject_shadows = setup.get("reject_shadows", False)
    polarity = setup.get("polarity", "either")

    background = make_background(
        cap, start_frame, end_frame, matrix, warp_w, warp_h, setup["background_samples"],
        color_mode=color_mode
    )

    # Shadow rejection needs a COLOR reference regardless of color_mode --
    # only build the extra background model when the feature is actually
    # turned on, since it's otherwise wasted work.
    color_background = background if color_mode == "rgb" else None
    if reject_shadows and color_background is None:
        color_background = make_background(
            cap, start_frame, end_frame, matrix, warp_w, warp_h, setup["background_samples"],
            color_mode="rgb"
        )

    if show_display:
        cv2.imshow("Automatic background - press ENTER", background)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key in (13, 32):
                break
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit("Cancelled.")
        cv2.destroyWindow("Automatic background - press ENTER")

    run_detection_preview(
        cap, matrix, warp_w, warp_h, start_frame, end_frame,
        background, arena, detection_roi_masks, roi_points, object_points,
        threshold, min_area, max_area, setup["preview_samples"], output_dir, show_live=show_display,
        mask_polygons=mask_polygons, exclusion_mask=exclusion_mask, color_mode=color_mode,
        color_background=color_background, reject_shadows=reject_shadows, polarity=polarity
    )

    # -----------------------------------------
    # Tracking loop
    # -----------------------------------------

    previous_point = None
    previous_area = None
    previous_time = start_time

    rows = []
    frame_number = start_frame

    max_recovery_streak = 10
    recovery_streak = 0

    print("\nStarting automatic tracking...")
    if show_display:
        print("Press Q during tracking to stop.")

    while frame_number <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, raw_frame = cap.read()

        if not ok:
            break

        frame = cv2.warpPerspective(raw_frame, matrix, (warp_w, warp_h))
        current_time = frame_number / fps

        candidate, mask = detect_mouse(
            frame, background, arena, previous_point, previous_area,
            threshold, max(1, int(min_area)), max(int(max_area), int(min_area) + 1), max_jump,
            roi_masks=detection_roi_masks, use_window=use_window,
            window_size=window_size, window_weight=window_weight,
            exclusion_mask=exclusion_mask, color_mode=color_mode,
            color_background=color_background, reject_shadows=reject_shadows, polarity=polarity
        )

        used_recovery = False

        if candidate is None and previous_point is not None:
            candidate = local_recovery(
                frame, background, previous_point, arena,
                threshold, max(1, int(min_area)), max(int(max_area), int(min_area) + 1),
                exclusion_mask=exclusion_mask, color_mode=color_mode,
                color_background=color_background, reject_shadows=reject_shadows, polarity=polarity
            )
            used_recovery = candidate is not None

            if candidate is not None and point_distance(candidate["center"], previous_point) > max_jump:
                candidate = None
                used_recovery = False

        if used_recovery and candidate is not None:
            recovery_streak += 1
            if recovery_streak > max_recovery_streak:
                candidate = None
                previous_point = None
                previous_area = None
        elif candidate is not None:
            recovery_streak = 0

        x = np.nan
        y = np.nan
        area = np.nan
        distance_px = 0.0
        velocity = 0.0
        zone_membership = {name: False for name in roi_names}
        object_membership = {name: False for name in object_names}

        if candidate is not None:
            x, y = candidate["center"]
            area = candidate["area"]

            smoothing_alpha = 0.6
            if previous_point is not None:
                x = smoothing_alpha * x + (1 - smoothing_alpha) * previous_point[0]
                y = smoothing_alpha * y + (1 - smoothing_alpha) * previous_point[1]
                distance_px = point_distance((x, y), previous_point)

            dt = max(1.0 / fps, current_time - previous_time)
            velocity = distance_px / dt

            previous_point = (x, y)
            previous_area = area

            zone_membership = roi_membership(roi_masks, x, y)
            roi_label = linearize_roi(zone_membership)
            if object_masks:
                object_membership = roi_membership(object_masks, x, y)
            status = "Tracked"
        else:
            roi_label = "None"
            status = "Lost"

        row = {
            "Frame": frame_number,
            "Time_seconds": current_time,
            "Mouse_X": x,
            "Mouse_Y": y,
            "Mouse_Present": status == "Tracked",
            "ROI": roi_label,
            "Detected_area": area,
            "Distance_pixels": distance_px,
            "Velocity_pixels_s": velocity,
            "Tracking_Status": status
        }

        for name in roi_names:
            row[f"In_{safe_col(name)}"] = zone_membership.get(name, False)
        for name in object_names:
            row[f"Near_{safe_col(name)}"] = object_membership.get(name, False)

        rows.append(row)

        if show_display:
            display = frame.copy()
            cv2.rectangle(display, (arena[0], arena[1]), (arena[2], arena[3]), (255, 255, 0), 1)
            if mask_polygons:
                draw_mask_polygons(display, mask_polygons)
            draw_roi_polygons(display, roi_points)
            draw_object_markers(display, object_points)

            if status == "Tracked":
                draw_crosshair(display, (int(x), int(y)), size=9, color=(0, 0, 255), thickness=2)
                cv2.putText(
                    display, "AUTO TRACK", (max(5, int(x) - 50), max(20, int(y) - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                )
                near_list = [n for n, v in object_membership.items() if v]
                status_line = "ROI: " + roi_label + (("  | Near: " + ", ".join(near_list)) if near_list else "")
                cv2.putText(display, status_line, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(display, "TRACKING LOST", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.putText(display, f"Time: {current_time:.2f} s", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(display, f"Frame: {frame_number}", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow("BEHAVIORAL TRACKING - Q to stop", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nTracking stopped by user.")
                break

        frame_number += 1
        previous_time = current_time

        if progress_callback is not None and frame_number % 5 == 0:
            span = max(1, end_frame - start_frame)
            progress_callback(min(1.0, (frame_number - start_frame) / span))
    cap.release()
    if show_display:
        cv2.destroyAllWindows()

    # -----------------------------------------
    # Data
    # -----------------------------------------

    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit("No tracking data were produced.")

    transitions = calculate_transitions(df)
    tracked = df[df["Tracking_Status"] == "Tracked"]
    tracked_time = len(tracked) / fps
    tracking_quality = 100 * len(tracked) / len(df) if len(df) > 0 else 0

    summary = {
        "Video": video_path,
        "Output_folder": output_dir,
        "Start_s": start_time,
        "End_s": end_time,
        "FPS": fps,
        "Frames_processed": len(df),
        "Tracked_frames": len(tracked),
        "Lost_frames": len(df) - len(tracked),
        "Tracking_quality_percent": tracking_quality,
        "Total_distance_pixels": tracked["Distance_pixels"].sum(),
    }

    for name in roi_names:
        col = f"In_{safe_col(name)}"
        name_time = tracked[col].sum() / fps if col in tracked else 0.0
        summary[f"{safe_col(name)}_time_s"] = name_time
        summary[f"{safe_col(name)}_percent"] = (name_time / tracked_time * 100) if tracked_time > 0 else 0

    if setup.get("scale_factor"):
        factor = setup["scale_factor"]
        unit = setup["scale_unit"]
        df[f"Distance_{unit}"] = df["Distance_pixels"] * factor
        summary[f"Total_distance_{unit}"] = summary["Total_distance_pixels"] * factor

    if not transitions.empty:
        pair_counts = transitions.groupby(["From_ROI", "To_ROI"]).size()
        for (frm, to), cnt in pair_counts.items():
            summary[f"Trans_{safe_col(frm)}_to_{safe_col(to)}"] = int(cnt)

    summary["Total_transitions"] = len(transitions)

    if setup.get("compute_arm_entries"):
        for name, count in calculate_arm_entries(transitions, roi_names).items():
            summary[f"{safe_col(name)}_entries"] = count

    if setup.get("compute_alternation"):
        summary.update(calculate_alternation(transitions))

    # -----------------------------------------
    # Interaction bouts
    # -----------------------------------------

    # Local imports: tracking.behaviour imports FROM this module at its own
    # top level (for read_and_warp/draw_roi_polygons/draw_object_markers),
    # so importing it back at this module's top level would be circular.
    # ask_yes_no is a GUI dialog helper (gui.main_window) -- same reasoning:
    # gui.main_window imports process_single_video FROM this module at its
    # top level, so importing ask_yes_no back at this module's top level
    # would also be circular.
    from tracking.behaviour import tag_interaction_bouts, calculate_bins
    from gui.main_window import ask_yes_no
    bouts = extract_bouts(df, object_names, fps) if object_names else []

    if bouts and show_display:
        if ask_yes_no(
            "Tag interactions",
            f"{len(bouts)} interaction bout(s) detected near your objects.\nReview and tag each one now?"
        ):
            bouts = tag_interaction_bouts(video_path, matrix, warp_w, warp_h, bouts, behavior_names,
                                           roi_points, object_points)
        else:
            for b in bouts:
                b["Behavior"] = "Unclassified"
    else:
        for b in bouts:
            b["Behavior"] = "Unclassified"

    for name in object_names:
        col = f"Near_{safe_col(name)}"
        near_time = tracked[col].sum() / fps if col in tracked else 0.0
        summary[f"{safe_col(name)}_near_time_s"] = near_time
        summary[f"{safe_col(name)}_near_percent"] = (near_time / tracked_time * 100) if tracked_time > 0 else 0
        summary[f"{safe_col(name)}_bout_count"] = sum(1 for b in bouts if b["Object"] == name)

    summary["Total_interaction_bouts"] = len(bouts)

    if behavior_names and bouts:
        counts = Counter((b["Object"], b["Behavior"]) for b in bouts if b.get("Behavior") not in (None, "Discarded"))
        for (obj, beh), cnt in counts.items():
            summary[f"Interact_{safe_col(obj)}_{safe_col(beh)}_count"] = cnt

    # -----------------------------------------
    # Bins
    # -----------------------------------------

    individual, cumulative = calculate_bins(df, fps, start_time, end_time, transitions, roi_names)

    if setup.get("scale_factor"):
        factor = setup["scale_factor"]
        unit = setup["scale_unit"]
        individual[f"Distance_{unit}"] = individual["Distance_pixels"] * factor
        cumulative[f"Distance_{unit}"] = cumulative["Distance_pixels"] * factor

    # -----------------------------------------
    # Output files -- see output/csv.py, output/excel.py, output/graphs.py
    # -----------------------------------------

    write_csv_report(df, output_dir)

    roi_coords = pd.DataFrame(
        [{"ROI": name, "Vertices_xy": str(pts)} for name, pts in roi_points.items()]
        + [{"ROI": name, "Vertices_xy": str(pts)} for name, pts in object_points.items()]
    )

    interactions_df = pd.DataFrame(bouts) if bouts else pd.DataFrame(
        columns=["Object", "Frame_start", "Frame_end", "Start_seconds", "End_seconds",
                 "Duration_seconds", "Behavior"]
    )

    write_excel_report(
        output_dir, video_name, df, summary, individual, cumulative,
        transitions, roi_coords, interactions_df
    )

    save_plots(df, output_dir, roi_points, object_points, warp_w, warp_h, background=background)

    # -----------------------------------------
    # Final report
    # -----------------------------------------

    print("\n" + "=" * 65)
    print("TRACKING COMPLETE")
    print("=" * 65)
    print(f"Tracking quality: {tracking_quality:.2f}%")

    for name in roi_names:
        print(f"{name} time: {summary.get(f'{safe_col(name)}_time_s', 0):.2f} s")

    print(f"Total transitions: {len(transitions)}")

    if object_names:
        print(f"Interaction bouts: {len(bouts)}")

    print("\nResults folder:")
    print(output_dir)

    return summary


# ============================================================
# MAIN
# ============================================================
