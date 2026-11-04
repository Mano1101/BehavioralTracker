"""Freezing analysis (BehavioralTracker 1.0.0) -- fear conditioning and other tests where the animal "freezes".

How it works (the classic frame-difference motion method, with a few BT additions):

1.  Motion = the number of pixels whose grey value changes by at least the *motion cutoff* between a frame and
    the frame 1/30 s before it (two frames apart at 60 fps, so the numbers do not depend on the frame rate).
    The picture is cut into small tiles; the change of a typical (median) tile is the camera noise / flicker /
    compression change of that frame and is subtracted, so light flicker and compression "pulses" are not
    counted as movement.
      * "Whole region" (default, Classic): every pixel inside the region box is counted.
      * "Around the animal": BT tracks the animal and only pixels inside a circle around it are counted
        (cables, hands or shadows far away are ignored).  Best for top-view videos that BT tracks well.
      * "Body dots - tail ignored": the animal's outline is found in every frame and marked with dots (nose, 4
        legs, body centre, tail base and dots along the tail).  Only the body (head, legs, trunk) is counted:
        when only the tail moves, the frame says "tail moved", not "body moved".
    Gray (default) or RGB: in RGB a pixel changes when any colour channel changes (colour-on-colour videos).
2.  Motion is given as a percentage of the animal's body area, so the same threshold works for any camera
    distance or video size.  The body size comes from BT tracking (automatic) or from your own 2-click
    measurement (nose -> tail base).
3.  Freezing = motion below the *freezing threshold* for at least the *minimum freeze duration* (seconds).
    Motion blips shorter than "ignore blips" inside a freeze do not break it.  Frames where most of the picture
    changes at once (light switched on / off, camera exposure jump) are not used: the animal keeps the state it
    had just before.

The expensive part (reading the video) is done once: for every frame a small histogram of the grey-value
changes is kept, so the cutoff, threshold, durations, epochs and time bins can be changed afterwards instantly.
"""
import hashlib
import math
import json
import os
import shutil
import tempfile
import threading

import cv2
import numpy as np

NBINS = 64                    # grey-value changes 0..63 are kept per frame (63 = "63 or more")
WORK_MAX_SIDE = 640           # the region is analysed at most this big (speed; thresholds are relative anyway)
SIGMA = 1.0                   # Gaussian blur before the difference (classic method)
LIGHT_FRACTION = 0.25         # a typical tile changed this much -> the light changed (frame not used)
DISC_FACTOR = 1.0             # "around the animal": circle radius = 1 x body length
REST_RETRY_PCT = 60.0         # animal found in fewer frames: tracked again with the whole video's background
TILE_TARGET = 40              # the picture is cut into tiles of about 40 x 40 px (work scale) ...
NOISE_KEEP = 0.75             # ... the 75 % quietest tiles give the camera noise of the frame
ACTIVITY_LEVEL = 6            # grey change used to rank the tiles from quiet to busy
CUTOFF_MIN, CUTOFF_MAX = 3, 40
TAIL_RATIO = 1.6              # auto cutoff: 99.99th percentile at most 1.6 x the 99.9th (spiky webcam noise)
AREA_PER_LEN2 = 1.0 / 2.4     # body area = length^2 / 2.4 (BT's body model: length = 2.4 x width)

DEFAULTS = {
    "motion_area": "region",      # "region" | "animal" | "body"
    "colour": "gray",             # "gray" | "rgb" (motion and tracking)
    "cutoff_auto": True,
    "cutoff": 10,                 # grey levels (used when cutoff_auto is False)
    "threshold_pct": 20.0,        # motion below this % of the body area = still
    "min_freeze_s": 1.0,
    "ignore_blips_s": 0.0,        # 0 = off (classic rule)
    "light_guard": True,
    "polarity": "auto",
    "min_contrast": 12.0,
    "body_length_px": 0.0,        # 0 = from BT tracking; > 0 = measured (video pixels)
    "start_time": 0.0,
    "end_time": None,
    "bin_s": 60.0,
    "pre_s": 30.0,                # "before" window of every epoch (pre-tone baseline)
    "epochs": [],                 # [{"name", "start", "end"}] in video seconds
    "trim_lights": True,          # light switched off / on at the start or end: those frames are not analysed
}

MOTION_AREAS = [("region", "Whole region (Classic)"), ("animal", "Around the animal"),
                ("body", "Body outline points (tail ignored)")]
COLOURS = [("gray", "Gray"), ("rgb", "RGB (colour)")]
# body dots: index -> name
KP_NAMES = ("Nose", "Front left leg", "Front right leg", "Body centre", "Hind left leg", "Hind right leg",
            "Tail base") + tuple(f"Tail {k}" for k in range(1, 7))
N_KP = len(KP_NAMES)
PARTS = ("body", "head", "legs", "tail")                       # motion counted per part
PART_HIST = {"body": ("h_body", "n_body"), "head": ("h_head", "n_head"), "legs": ("h_legs", "n_legs"),
             "tail": ("h_tail", "n_tail")}


class Cancelled(Exception):
    pass


def frame_lag(fps):
    """Frames between the two pictures that are compared: 1/30 s (1 frame at 25-45 fps, 2 at 60 fps)."""
    return max(1, int(round(float(fps or 30.0) / 30.0)))


def region_parts(region):
    """(polygon or None, [mask polygons]) of a region: None (whole frame), an (x0, y0, x1, y1) box (older files) or
    {"poly": [[x, y], ...] or None, "masks": [[[x, y], ...], ...]} (video px)."""
    if not region:
        return None, []
    if isinstance(region, dict):
        poly = region.get("poly")
        poly = [(float(x), float(y)) for x, y in poly] if poly and len(poly) >= 3 else None
        masks = [[(float(x), float(y)) for x, y in m] for m in (region.get("masks") or []) if m and len(m) >= 3]
        return poly, masks
    x0, y0, x1, y1 = [float(v) for v in region]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], []


def _is_box(poly):
    """An axis-aligned rectangle (4 corners): analysed exactly like the classic region box."""
    if poly is None or len(poly) != 4:
        return False
    xs = sorted({round(x, 3) for x, _y in poly})
    ys = sorted({round(y, 3) for _x, y in poly})
    return len(xs) == 2 and len(ys) == 2


def region_json(region):
    """For saving (settings, .btf): a list for a box, a dict for a polygon / masks, None for the whole frame."""
    if not region:
        return None
    if isinstance(region, dict):
        poly, masks = region_parts(region)
        if poly is None and not masks:
            return None
        return {"poly": [list(p) for p in poly] if poly else None, "masks": [[list(p) for p in m] for m in masks]}
    return [float(v) for v in region]


def region_from_json(v):
    if not v or v == "whole frame":
        return None
    if isinstance(v, dict):
        return region_json(v)
    return tuple(float(x) for x in v)


def region_text(region):
    """'Whole frame' / 'Box x0 y0 - x1 y1' / 'Polygon (n points)', plus the number of masks."""
    poly, masks = region_parts(region)
    if poly is None:
        txt = "Whole frame"
    elif _is_box(poly):
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        txt = "Box x{:.0f} y{:.0f} - x{:.0f} y{:.0f}".format(min(xs), min(ys), max(xs), max(ys))
    else:
        txt = f"Polygon ({len(poly)} points)"
    if masks:
        txt += f" · {len(masks)} mask{'s' if len(masks) > 1 else ''}"
    return txt


def clean_region(region, frame_w, frame_h):
    """The region's bounding box (x0, y0, x1, y1) inside the frame, at least 8 px each way, or None (= the whole
    frame).  `region` = None, a box or a polygon / masks dict (region_parts)."""
    if not region:
        return None
    poly, _masks = region_parts(region)
    if poly is None:
        return None
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, x1 = sorted((max(0.0, min(frame_w, x0)), max(0.0, min(frame_w, x1))))
    y0, y1 = sorted((max(0.0, min(frame_h, y0)), max(0.0, min(frame_h, y1))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    if x0 <= 0 and y0 <= 0 and x1 >= frame_w and y1 >= frame_h and _is_box(poly):
        return None
    return (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 1000:
            fps = 30.0
        return {"fps": float(fps), "total": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    finally:
        cap.release()


def time_window(info, start_time=0.0, end_time=None):
    """(start_frame, end_frame) inclusive, as BT tracking uses them."""
    fps, total = info["fps"], info["total"]
    duration = total / fps if fps else 0.0
    t0 = max(0.0, min(float(start_time or 0.0), duration))
    t1 = duration if end_time in (None, "") else max(t0, min(float(end_time), duration))
    return int(round(t0 * fps)), min(max(total - 1, 0), int(round(t1 * fps)))


class Geometry:
    """Region (box, or polygon + masks) in the video + the scale it is analysed at + the tiles.  A box without
    masks is analysed exactly as before (every pixel of the box); a polygon / masks: only the pixels inside the
    polygon and outside the masks count (`inside`, work scale)."""

    def __init__(self, frame_w, frame_h, region=None):
        self.frame_w, self.frame_h = int(frame_w), int(frame_h)
        r = clean_region(region, frame_w, frame_h)
        self.region = r
        self.spec = region_json(region)
        poly, masks = region_parts(region)
        self.poly = poly if (poly is not None and not _is_box(poly)) else None
        self.masks = masks
        self.x0, self.y0, self.x1, self.y1 = r if r else (0, 0, int(frame_w), int(frame_h))
        rw, rh = self.x1 - self.x0, self.y1 - self.y0
        self.s = min(1.0, WORK_MAX_SIDE / float(max(rw, rh)))
        self.ww = max(8, int(round(rw * self.s)))
        self.wh = max(8, int(round(rh * self.s)))
        tile = max(12, min(TILE_TARGET, min(self.ww, self.wh) // 4))
        nx, ny = max(1, self.ww // tile), max(1, self.wh // tile)
        tx = np.minimum(np.arange(self.ww) * nx // self.ww, nx - 1)
        ty = np.minimum(np.arange(self.wh) * ny // self.wh, ny - 1)
        self.n_tiles = nx * ny
        self.tile_idx = (ty[:, None] * nx + tx[None, :]).astype(np.int32) * NBINS
        self.tile_px = np.bincount((self.tile_idx // NBINS).ravel(), minlength=self.n_tiles).astype(np.int64)
        self.inside = None                      # None = every pixel of the box counts
        if self.poly is not None or self.masks:
            m = np.full((self.y1 - self.y0, self.x1 - self.x0), 255 if self.poly is None else 0, np.uint8)
            off = np.array([self.x0, self.y0], np.float64)
            if self.poly is not None:
                cv2.fillPoly(m, [np.round(np.asarray(self.poly) - off).astype(np.int32)], 255)
            for mk in self.masks:
                cv2.fillPoly(m, [np.round(np.asarray(mk) - off).astype(np.int32)], 0)
            if (m.shape[1], m.shape[0]) != (self.ww, self.wh):
                m = cv2.resize(m, (self.ww, self.wh), interpolation=cv2.INTER_AREA)
            ins = m >= 128
            if not ins.all():
                if ins.sum() < 64:
                    raise RuntimeError("The region left after the masks is too small.")
                self.inside = ins
        full = self.tile_px.copy()
        if self.inside is not None:
            self.tile_px = np.bincount((self.tile_idx // NBINS)[self.inside],
                                       minlength=self.n_tiles).astype(np.int64)
        # tiles mostly inside the region give the typical change / the camera noise
        self.tile_ok = self.tile_px >= np.maximum(16, 0.3 * full)
        if not self.tile_ok.any():
            self.tile_ok = self.tile_px > 0
        self.n_px = int(self.inside.sum()) if self.inside is not None else self.ww * self.wh
        self.mask_u8 = None if self.inside is None else self.inside.astype(np.uint8) * 255

    def brightness(self, img):
        """Mean grey of the region (inside the polygon, outside the masks) of a prep() picture."""
        m = cv2.mean(img, mask=self.mask_u8)
        return float(np.mean(m[:3])) if img.ndim == 3 else float(m[0])

    def video_polys(self):
        """(region polygon or None, [masks]) in video px, for drawing."""
        poly = self.poly
        if poly is None and self.region is not None:
            x0, y0, x1, y1 = self.region
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        return poly, self.masks

    def prep(self, frame_bgr, colour="gray"):
        """The region at work scale, blurred: grey (float32 H x W) or colour (float32 H x W x 3)."""
        sub = frame_bgr[self.y0:self.y1, self.x0:self.x1]
        if colour == "rgb" and sub.ndim == 3:
            g = sub
        else:
            g = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY) if sub.ndim == 3 else sub
        if g.shape[1] != self.ww or g.shape[0] != self.wh:
            g = cv2.resize(g, (self.ww, self.wh), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g.astype(np.float32), (0, 0), SIGMA)

    def to_work(self, x, y):
        return (np.asarray(x, float) - self.x0) * self.s, (np.asarray(y, float) - self.y0) * self.s


def quantise(diff):
    return np.minimum(diff, NBINS - 1).astype(np.uint8)


def frame_diff(prev, cur):
    """|change| per pixel; colour pictures: the largest change of the three channels."""
    d = cv2.absdiff(cur, prev)
    return d.max(axis=2) if d.ndim == 3 else d


def as_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def change_mask(prev, cur, cutoff, inside=None):
    """True where the grey (colour) value changed by at least `cutoff` (work scale), inside the region."""
    m = frame_diff(prev, cur) >= float(cutoff)
    return m if inside is None else (m & inside)


def _rcum(h, axis=-1):
    """cum[..., c] = count of values >= c."""
    return np.flip(np.cumsum(np.flip(h, axis=axis), axis=axis), axis=axis)


def _percentile_from_cum(cum, n, frac):
    """Grey level reached by `frac` of n pixels (cum per row, linear inside a level) -> array."""
    target = frac * np.asarray(n, float)
    p = np.full(len(cum), np.nan)
    for c in range(NBINS - 1, -1, -1):
        hit = (cum[:, c] >= target) & np.isnan(p)
        if hit.any():
            nxt = cum[hit, c + 1] if c + 1 < NBINS else np.zeros(int(hit.sum()))
            span = np.maximum(cum[hit, c] - nxt, 1)
            p[hit] = c + np.clip((cum[hit, c] - target[hit]) / span, 0, 1)
    return p


# ============================================================================= calibration (empty video)
def calibrate_empty_video(path, region=None, frames=300, pixels=10000, seed=0):
    """Classic calibration on a video WITHOUT an animal: 2 x the 99.99th percentile of the frame-to-frame grey
    changes of random pixels.  Returns dict(cutoff, p9999, mean_change, frames)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        geo = Geometry(fw, fh, region)
        lag = frame_lag(fps)
        rng = np.random.default_rng(seed)
        if geo.inside is None:
            ys = (rng.random(pixels) * geo.wh).astype(int)
            xs = (rng.random(pixels) * geo.ww).astype(int)
        else:
            iy, ix = np.nonzero(geo.inside)
            k = rng.integers(0, len(ix), pixels)
            ys, xs = iy[k], ix[k]
        ring, rows = [], []
        for _ in range(int(frames) + lag):
            ok, f = cap.read()
            if not ok:
                break
            img = geo.prep(f)
            ring.append(img)
            if len(ring) > lag:
                prev = ring.pop(0)
                rows.append(np.abs(img[ys, xs] - prev[ys, xs]))
    finally:
        cap.release()
    if len(rows) < 10:
        raise RuntimeError("The calibration video is too short (fewer than 10 frames could be read).")
    d = np.vstack(rows)
    p = float(np.percentile(d, 99.99))
    cut = int(min(CUTOFF_MAX, max(CUTOFF_MIN, round(2.0 * p))))
    return {"cutoff": cut, "p9999": p, "mean_change": float(d.mean()), "frames": len(rows)}


# ============================================================================= step 1: where is the animal
def _track_setup(path, settings, region, info, tmp=None):
    """The tracking setup the freezing analysis uses (shared by the real run, Test detection and Suggest)."""
    fw, fh = info["w"], info["h"]
    r = clean_region(region, fw, fh)
    setup = {"matrix": np.eye(3, dtype=np.float32), "warp_w": fw, "warp_h": fh,
             "tracking_algorithm": "robust", "animal_polarity": settings.get("polarity", "auto"),
             "color_mode": "rgb" if settings.get("colour") == "rgb" else "gray", "background_samples": 80,
             "min_area": 15, "max_area": 0, "max_jump": 100, "live_show_mask": False,
             "min_contrast": float(settings.get("min_contrast", 12.0) or 12.0),
             "start_time": float(settings.get("start_time") or 0.0), "end_time": settings.get("end_time"),
             "output_dir_override": tmp, "entry_rule": "centre", "preview_samples": 0, "test_type": "freezing",
             "adapt_background": True, "low_light": settings.get("low_light", "off"), "_fps": info["fps"]}
    if settings.get("_workers"):
        setup["workers"] = int(settings["_workers"])
    measured = float(settings.get("body_length_px") or 0.0)
    if measured > 0:                               # 📏 measured (nose -> tail base): the tracker expects this size
        setup["body_length_px"] = measured
    poly, masks = region_parts(region)
    if r is not None:
        x0, y0, x1, y1 = r
        setup["roi_points"] = {"Region": poly if (poly is not None and not _is_box(poly))
                               else [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]}
        setup["roi_names"] = ["Region"]
        setup["auto_mask_outside_zones"] = True
        setup["auto_mask_margin"] = 0
    if masks:
        setup["mask_points"] = masks               # reflections, a lamp, a cable ...: never the animal
    return setup


def locate_animal(path, settings, region=None, progress=None, live=None, bg_frames=None, fused=None):
    """BT tracking of the time window.  Returns dict(frames, xs, ys, found_pct, area, body_len, ...).
    xs / ys are video pixels, NaN where the animal was not seen.  area = the animal's full body area (the 90th
    percentile of the tracked sizes: a still animal is often only partly separated from the floor)."""
    from tracking.location import track_video
    info = video_info(path)
    tmp = tempfile.mkdtemp(prefix="bt_freeze_")
    setup = _track_setup(path, settings, region, info, tmp)
    if bg_frames is not None:                      # test detection: the background of the real run's window
        setup["background_window_frames"] = [int(bg_frames[0]), int(bg_frames[1])]
    if fused is not None:                          # one pass: the motion is measured from the same frames
        setup["_frame_hook"] = fused
    cb = None
    if live is not None:
        def cb(frame):
            live(frame, "Tracking the animal (body size / position)…")
        cb.due = live.due

    def run(st_, prog):
        try:
            return track_video(str(path), st_, progress_callback=prog, frame_callback=cb)
        finally:
            shutil.rmtree(st_["output_dir_override"], ignore_errors=True)

    raw = run(setup, progress)
    loc = _loc_from(raw, settings)
    if loc["found_pct"] < REST_RETRY_PCT:
        # hardly found: the animal may have frozen (or slept) through the whole time window, so it is part of a
        # background taken from that window.  Again with the background from the whole video, where the floor under
        # a resting animal is found from the few samples without it there.
        wide = dict(setup, output_dir_override=tempfile.mkdtemp(prefix="bt_freeze_"),
                    background_window_frames=[0, max(0, info["total"] - 1)], background_samples=120,
                    background_rest_pct=97)
        wide.pop("_frame_hook", None)              # the motion was measured in the first run already
        loc2 = _loc_from(run(wide, progress), settings)
        # kept only when it clearly finds more and the size is an animal's (not floor noise inside a box the
        # animal is not in): at least half the size seen in the first run, when that run saw it at all
        size_ok = loc2["area"] > 0 and (loc["area"] <= 0 or loc2["area"] >= 0.5 * loc["area"])
        if loc2["found_pct"] >= loc["found_pct"] + 20.0 and size_ok:
            loc = loc2
    return loc


def _loc_from(raw, settings):
    st = np.array([s == "Tracked" for s in raw["status"]], bool)
    xs = np.where(st, np.asarray(raw["xs"], float), np.nan)
    ys = np.where(st, np.asarray(raw["ys"], float), np.nan)
    good = np.asarray(raw["areas"], float)[st]
    good = good[np.isfinite(good) & (good > 0)]
    area = float(np.percentile(good, 90)) if good.size >= 10 else float(raw.get("animal_area_full") or 0.0)
    return {"frames": np.asarray(raw["frames"], np.int64), "xs": xs, "ys": ys,
            "found_pct": float(100.0 * st.mean()) if st.size else 0.0,
            "area": area, "body_len": math.sqrt(area / AREA_PER_LEN2) if area > 0 else 0.0,
            "polarity": raw.get("polarity"), "polarity_warning": raw.get("polarity_warning"),
            "background": raw.get("background_bgr"),
            "min_contrast": float(settings.get("min_contrast", 12.0) or 12.0)}


def model_key(path, settings, region):
    """What the background model of the analysis window depends on (not min contrast / body size)."""
    s = merged_settings(settings)
    return (os.path.normcase(os.path.abspath(str(path))), json.dumps(region_json(region), sort_keys=True),
            s.get("polarity"), s.get("colour"), s.get("low_light", "off"), float(s.get("start_time") or 0.0),
            None if s.get("end_time") is None else float(s["end_time"]))


def detection_model(path, settings, region=None, progress=None):
    """The background model of the analysis window, built as the real run builds it.  Kept by the page and reused
    by Test detection and Suggest settings (building it is the slow part).  -> dict."""
    from tracking.location import make_tracking_engine, build_detection_exclusion
    info = video_info(path)
    a0, a1 = time_window(info, settings.get("start_time"), settings.get("end_time"))
    st = _track_setup(path, settings, region, info)
    fw, fh = info["w"], info["h"]
    exclusion = None
    if st.get("mask_points") or st.get("roi_points"):
        exclusion = build_detection_exclusion(fw, fh, st.get("mask_points"), st.get("roi_points"),
                                              st.get("auto_mask_outside_zones"), margin=st.get("auto_mask_margin"))
        if exclusion.all():
            exclusion = None
    cap = cv2.VideoCapture(str(path))
    try:
        def read(i):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            return f if ok else None
        _tr, bgm = make_tracking_engine(st, fw, fh, read, a0, max(a0, a1), exclusion, progress=progress,
                                        video_path=str(path), warp_fn=lambda f: f)
    finally:
        cap.release()
    return {"key": model_key(path, settings, region), "bgm": bgm, "exclusion": exclusion, "setup": st,
            "a0": a0, "a1": a1, "fps": info["fps"], "total": info["total"]}


def _tracker_for(model, settings):
    from tracking.location import new_tracker
    st = dict(model["setup"], min_contrast=float(settings.get("min_contrast", 12.0) or 12.0))
    st.pop("body_length_px", None)
    if float(settings.get("body_length_px") or 0) > 0:
        st["body_length_px"] = float(settings["body_length_px"])
    return new_tracker(st, model["bgm"], model["exclusion"])


def test_detection(path, settings, region, frame, before_s=1.0, model=None):
    """BT tracking of about 1 s up to `frame`, with the background of the analysis time window (as in the real run).
    `model` = detection_model() to reuse (built when None or for other settings).
    -> dict(frame, x, y, found, found_pct, body_len, polarity, polarity_warning, bgr = the frame marked, model)."""
    if model is None or model.get("key") != model_key(path, settings, region):
        model = detection_model(path, settings, region)
    fps = model["fps"]
    f1 = int(min(max(0, frame), max(0, model["total"] - 1)))
    f0 = max(0, f1 - max(2, int(round(before_s * fps))))
    tr = _tracker_for(model, settings)
    cap = cv2.VideoCapture(str(path))
    xs, ys, areas = [], [], []
    img = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
        for _fn in range(f0, f1 + 1):
            ok, f = cap.read()
            if not ok:
                break
            img = f
            d = tr.update(f)
            okd = d.get("status") == "Tracked"
            for k, (bx, by, ba) in enumerate(reversed(d.get("backfill") or [])):
                j = len(xs) - 1 - k                 # frames reported lost while the lock was being confirmed
                if j >= 0:
                    xs[j], ys[j] = float(bx), float(by)
                    areas.append(float(ba or 0))
            xs.append(d["x"] if okd else float("nan"))
            ys.append(d["y"] if okd else float("nan"))
            if okd:
                areas.append(float(d.get("area") or 0))
    finally:
        cap.release()
    x = y = float("nan")
    if xs:
        x, y = float(xs[-1]), float(ys[-1])
    found = bool(np.isfinite(x) and np.isfinite(y))
    bgm = model["bgm"]
    area = float(np.median(areas)) if areas else 0.0
    body_len = math.sqrt(area / AREA_PER_LEN2) if area > 0 else 0.0
    if img is not None:
        img = img.copy()
        draw_region(img, region=region)
        if found:
            r = max(6, int(round(0.45 * (body_len or 30.0))))
            lw = max(2, int(round(max(img.shape[:2]) / 500)))
            c = (int(round(x)), int(round(y)))
            cv2.circle(img, c, r, (20, 20, 20), lw + 2, cv2.LINE_AA)
            cv2.circle(img, c, r, (80, 220, 80), lw, cv2.LINE_AA)
            cv2.drawMarker(img, c, (80, 220, 80), cv2.MARKER_CROSS, r, lw, cv2.LINE_AA)
    n = len(xs)
    return {"frame": f1, "x": x, "y": y, "found": found,
            "found_pct": 100.0 * float(np.isfinite(np.asarray(xs, float)).mean()) if n else 0.0,
            "body_len": body_len, "polarity": getattr(bgm, "polarity", None),
            "polarity_warning": getattr(bgm, "polarity_warning", None), "bgr": img, "model": model}


def suggest_settings(path, settings, region=None, progress=None, model=None):
    """Tracking settings measured from the video (Suggest settings): the animal colour the background vote found
    and the minimum contrast from 40 frames of the analysis window (half the animal's weakest contrast, above the
    noise).  -> dict(polarity, min_contrast, found, frames, length_px, noise, peak_median, peak_low, notes, model)."""
    from tracking.location import measure_animal_contrast
    s = dict(settings, polarity="auto")
    if model is None or model.get("key") != model_key(path, s, region):
        model = detection_model(path, s, region,
                                progress=None if progress is None else (lambda f: progress(0.6 * f)))
    bgm = model["bgm"]
    pol = getattr(bgm, "polarity", None)
    if getattr(bgm, "polarity_warning", None) in ("dark", "light"):
        pol = bgm.polarity_warning
    frames = []
    cap = cv2.VideoCapture(str(path))
    try:
        for i in np.linspace(model["a0"], model["a1"], 40).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            frames.append(f if ok else None)
    finally:
        cap.release()
    st = dict(model["setup"], min_contrast=float(settings.get("min_contrast", 12.0) or 12.0))
    m = measure_animal_contrast(st, bgm, model["exclusion"], frames,
                                progress=None if progress is None else (lambda f: progress(0.6 + 0.4 * f)))
    notes = []
    rec = m.get("recommended")
    if rec is None:
        notes.append(f"The animal was found clearly in only {m.get('found', 0)} of {m.get('frames', 0)} checked "
                     f"frames: check the region and the masks (reflections), or 📏 measure the animal.")
    elif m.get("weak"):
        notes.append("At times the animal is only a little different from the floor (dim place / shadow).")
    area = m.get("area_median_full")
    length = math.sqrt(area / AREA_PER_LEN2) if area else 0.0
    return {"polarity": pol if pol in ("dark", "light") else None, "min_contrast": rec,
            "found": int(m.get("found_with_rec") or m.get("found") or 0), "frames": int(m.get("frames") or 0),
            "length_px": length, "noise": m.get("sigma"), "peak_median": m.get("peak_median"),
            "peak_low": m.get("peak_low"), "notes": notes, "model": model}


def hold_positions(xs, ys):
    """Fill frames where the animal was not seen with the last (or, at the start, the first) known position."""
    xs, ys = np.asarray(xs, float).copy(), np.asarray(ys, float).copy()
    ok = np.isfinite(xs) & np.isfinite(ys)
    if not ok.any():
        return None, None
    idx = np.where(ok, np.arange(len(xs)), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[idx < 0] = int(np.argmax(ok))
    return xs[idx], ys[idx]


# ============================================================================= step 2: motion per frame
class _Disc:
    """Pixel histogram inside the circle(s) around the animal (now and 1/30 s ago)."""

    def __init__(self, ww, wh, r, inside=None):
        self.ww, self.wh = ww, wh
        self.r2 = float(r) ** 2
        self.pad = int(math.ceil(r)) + 1
        self.inside = inside                     # the region's pixels (polygon / masks) or None

    def hist(self, q, p1, p2):
        bx0 = max(0, int(math.floor(min(p1[0], p2[0]))) - self.pad)
        bx1 = min(self.ww, int(math.ceil(max(p1[0], p2[0]))) + self.pad + 1)
        by0 = max(0, int(math.floor(min(p1[1], p2[1]))) - self.pad)
        by1 = min(self.wh, int(math.ceil(max(p1[1], p2[1]))) + self.pad + 1)
        if bx1 <= bx0 or by1 <= by0:
            return np.zeros(NBINS, np.int64), 0
        gx = np.arange(bx0, bx1, dtype=np.float32)[None, :]
        gy = np.arange(by0, by1, dtype=np.float32)[:, None]
        dm = np.minimum((gx - p1[0]) ** 2 + (gy - p1[1]) ** 2, (gx - p2[0]) ** 2 + (gy - p2[1]) ** 2)
        disc = dm <= self.r2
        if self.inside is not None:
            disc &= self.inside[by0:by1, bx0:bx1]
        return np.bincount(q[by0:by1, bx0:bx1][disc], minlength=NBINS), int(disc.sum())


K3 = np.ones((3, 3), np.uint8)
# the 8 neighbours of a pixel (a walk along the tail goes from pixel to pixel)
_NB = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_NBU = [(dx / math.hypot(dx, dy), dy / math.hypot(dx, dy)) for dx, dy in _NB]
_NBS = [math.hypot(dx, dy) for dx, dy in _NB]
BODY_EDGE = 0.4               # the outline is where the picture is 40 % of the way from floor to animal
TAIL_STEP = 0.2               # a tail dot every 0.2 body length


def _signed(g, b, pol):
    """How much darker (dark animal) / lighter (light animal) / different (either) than the floor."""
    return (b - g) if pol == "dark" else ((g - b) if pol == "light" else np.abs(g - b))


def _odd(v):
    return max(3, int(round(v)) | 1)


def _ridge_kernels(w):
    """Thin-line detectors in 8 directions: the centre minus the mean of the two sides w px away."""
    ks = []
    size = 2 * int(math.ceil(w)) + 3
    c = size // 2
    for a in np.deg2rad(np.arange(0.0, 180.0, 22.5)):
        nx, ny = -math.sin(a), math.cos(a)
        k = np.zeros((size, size), np.float32)
        k[c, c] += 1.0
        for sg in (1, -1):
            x, y = c + sg * w * nx, c + sg * w * ny
            x0, y0 = int(math.floor(x)), int(math.floor(y))
            fx, fy = x - x0, y - y0
            for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy),
                               (1, 1, fx * fy)):
                k[y0 + dy, x0 + dx] -= 0.5 * wt
        ks.append(k)
    return ks


def _ridge_map(d, ks):
    out = None
    for k in ks:
        r = cv2.filter2D(d, -1, k, borderType=cv2.BORDER_REPLICATE)
        out = r if out is None else np.maximum(out, r)
    return out


def _blob_at(m, px, py, max_dist):
    """The blob under (px, py), else the biggest one within max_dist of it; holes filled."""
    n, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    ix, iy = int(round(px)), int(round(py))
    if 0 <= iy < lab.shape[0] and 0 <= ix < lab.shape[1] and lab[iy, ix]:
        j = int(lab[iy, ix])
    else:
        bx, by, bw, bh = st[1:, 0], st[1:, 1], st[1:, 2], st[1:, 3]
        dx = np.maximum(0, np.maximum(bx - px, px - (bx + bw - 1)))
        dy = np.maximum(0, np.maximum(by - py, py - (by + bh - 1)))
        ok = np.hypot(dx, dy) <= max_dist
        if not ok.any():
            return None
        j = 1 + int(np.argmax(np.where(ok, st[1:, cv2.CC_STAT_AREA], -1)))
    sil = (lab == j).astype(np.uint8)
    cnts, _h = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(sil, cnts, -1, 1, -1)
    return sil


def _silhouette(d, px, py, T, L):
    m = cv2.morphologyEx((d >= T).astype(np.uint8), cv2.MORPH_CLOSE, K3)
    return _blob_at(m, px, py, max(3.0, 0.25 * L))


def _core_contrast(d, px, py, rc):
    """The animal's contrast: 95th percentile inside a small disc at the tracked body centre."""
    h, w = d.shape
    x0, x1 = max(0, int(px - rc)), min(w, int(px + rc) + 1)
    y0, y1 = max(0, int(py - rc)), min(h, int(py + rc) + 1)
    if x1 <= x0 or y1 <= y0:
        return float("nan")
    yy, xx = np.ogrid[y0:y1, x0:x1]
    v = d[y0:y1, x0:x1][(xx - px) ** 2 + (yy - py) ** 2 <= rc * rc]
    return float(np.percentile(v, 95)) if v.size else float("nan")


class BodyCal:
    """Measured on a few frames before the motion pass (work-scale px): body length (nose -> tail base) and area,
    the floor's noise, the animal's contrast and the floor's thin-line level (for the tail)."""

    def __init__(self, L, area, noise, contrast, rfloor, ok):
        self.L, self.area, self.noise, self.contrast, self.rfloor, self.ok = L, area, noise, contrast, rfloor, ok


def calibrate_body(path, geo, frames, wx, wy, background_bgr, polarity, min_contrast, len_guess, n=16):
    """BodyCal from up to n frames spread over the window (frames = video frame numbers, wx / wy = the tracked
    body centre at work scale).  The outline is where the picture is 40 % of the way from floor to animal, so a
    textured floor (bedding), the animal's shadow or the wall do not make it bigger than the animal."""
    pol = polarity if polarity in ("dark", "light") else "any"
    bg = geo.prep(background_bgr, "gray")
    ok_idx = [i for i in range(len(frames)) if np.isfinite(wx[i]) and np.isfinite(wy[i])]
    pick = [ok_idx[int(k)] for k in np.linspace(0, len(ok_idx) - 1, min(n, len(ok_idx)))] if ok_idx else []
    areas, cons, noises, sils = [], [], [], []
    cap = cv2.VideoCapture(str(path))
    try:
        for i in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[i]))
            ok, f = cap.read()
            if not ok:
                continue
            g = geo.prep(f, "gray")
            R = int(round(1.3 * len_guess)) + 4
            x, y = float(wx[i]), float(wy[i])
            x0, y0 = max(0, int(x) - R), max(0, int(y) - R)
            x1, y1 = min(geo.ww, int(x) + R + 1), min(geo.wh, int(y) + R + 1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            d = _signed(g[y0:y1, x0:x1], bg[y0:y1, x0:x1], pol).astype(np.float32)
            if geo.inside is not None:
                d[~geo.inside[y0:y1, x0:x1]] = 0.0
            px, py = x - x0, y - y0
            c = _core_contrast(d, px, py, max(2.0, 0.12 * len_guess))
            med = float(np.median(d))
            sig = 1.4826 * float(np.median(np.abs(d - med)))
            lo = max(float(min_contrast), 3.0 * sig)
            if not np.isfinite(c) or c < 1.5 * lo:
                continue
            sil = _silhouette(d, px, py, max(lo, BODY_EDGE * c), len_guess)
            if sil is None:
                continue
            areas.append(float(sil.sum()))
            cons.append(c)
            noises.append(sig)
            sils.append((sil, d))
    finally:
        cap.release()
    if len(areas) < 3:
        return BodyCal(len_guess, AREA_PER_LEN2 * len_guess ** 2, 3.0, 2.0 * min_contrast, 0.5 * min_contrast,
                       False)
    a0 = float(np.median(areas))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.18 * math.sqrt(a0 / AREA_PER_LEN2)),) * 2)
    opened = [float(cv2.morphologyEx(s, cv2.MORPH_OPEN, k).sum()) for s, _d in sils]   # without the tail
    area = float(np.median([o for o in opened if o > 0] or [a0]))
    L = math.sqrt(area / AREA_PER_LEN2)
    ks = _ridge_kernels(max(1.5, 0.07 * L))
    rf = []
    for s, d in sils:
        far = cv2.dilate(s, K3, iterations=max(3, int(0.2 * L))) == 0
        if far.sum() > 50:
            rf.append(float(np.percentile(_ridge_map(d, ks)[far], 90)))
    return BodyCal(L, area, float(np.median(noises)), float(np.median(cons)),
                   float(np.median(rf)) if rf else 0.5 * min_contrast, True)


class _Body:
    """The animal's outline in one frame (work scale) and its dots: nose, 4 legs, body centre, tail base and dots
    along the tail.

    * Outline: where the picture is 40 % of the way from the floor to the animal (its own contrast, measured in
      every frame at the tracked body centre) -- textured bedding, the shadow or the wall stay outside.  An outline
      much bigger than the animal (it touches something dark) is cut again with a higher threshold.
    * Body = the outline without thin parts (an opening ~1/5 body length); legs are placed on the full outline,
      so a paw that sticks out pulls its dot.
    * Tail: a thin dark line followed pixel by pixel from the body's edge near one end (thin-line filter, floor
      level measured on the video).  The end with the tail is the back; the nose is the outline point farthest
      from the tail base.  Without a visible tail the head stays where it was in the frame before.
    * A dot keeps its place while it moves less than the picture's jitter, so still = still and a part that
      moves moves its dot."""

    def __init__(self, geo, background_bgr, polarity, contrast, cal):
        self.W, self.H = geo.ww, geo.wh
        self.bg = geo.prep(background_bgr, "gray")
        self.pol = polarity if polarity in ("dark", "light") else "any"
        self.inside = geo.inside                         # polygon / masks (work scale) or None
        self.cal = cal
        self.Lw = float(cal.L)                           # body length at work scale
        self.ds = min(1.0, 60.0 / self.Lw)              # found on a smaller copy when the animal is big (speed)
        self.L = self.Lw * self.ds
        self.T_lo = max(float(contrast), 3.0 * cal.noise)
        self.k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(0.2 * self.L),) * 2)
        self.k3 = K3
        self.win = int(round(1.9 * self.Lw)) + 4
        self.area_ds = cal.area * self.ds * self.ds
        self.ks = _ridge_kernels(max(1.5, 0.07 * self.L))
        self.strong = max(2.2 * cal.rfloor, 0.35 * self.T_lo)
        self.weak = max(1.4 * cal.rfloor, 0.25 * self.T_lo)
        self.prev = None
        self._hint = None
        self._turn = 0                                  # frames in a row that say "turned round"
        self.r_head = max(2.0, 0.3 * self.Lw)
        self.r_leg = max(1.5, 0.16 * self.Lw)

    # ---- the tail
    def _follow(self, R, blocked, p, v, max_len):
        """Walk along the thin-line map from p (outwards, direction v), pixel by pixel: the best forward neighbour
        (looking one step further on); four weak pixels in a row end the tail."""
        h, w = R.shape
        Ri, Bi = R.item, blocked.item
        x, y = int(p[0]), int(p[1])
        vx, vy = float(v[0]), float(v[1])
        weak = self.weak
        pts = [(x, y)]
        good, miss, length, tot, nval = 1, 0, 0.0, 0.0, 0
        while length < max_len:
            bj, bs = -1, -1e18
            for j in range(8):
                ux, uy = _NBU[j]
                cosv = ux * vx + uy * vy
                if cosv < 0.38:                                  # within ~67 degrees of the way it was going
                    continue
                nx, ny = x + _NB[j][0], y + _NB[j][1]
                if nx < 0 or ny < 0 or nx >= w or ny >= h or Bi(ny, nx):
                    continue
                s2 = -1e18
                for k in (j - 1, j, j + 1):
                    dx, dy = _NB[k % 8]
                    mx, my = nx + dx, ny + dy
                    if 0 <= mx < w and 0 <= my < h:
                        r2 = Ri(my, mx)
                        if r2 > s2:
                            s2 = r2
                sc = Ri(ny, nx) + 0.5 * s2 + 2.0 * cosv
                if sc > bs:
                    bj, bs = j, sc
            if bj < 0:
                break
            x, y = x + _NB[bj][0], y + _NB[bj][1]
            r = Ri(y, x)
            pts.append((x, y))
            length += _NBS[bj]
            if r >= weak:
                miss = 0
                tot += r
                nval += 1
                good = len(pts)
            else:
                miss += 1
                if miss > 3:
                    break
            ux, uy = _NBU[bj]
            vx, vy = 0.7 * vx + 0.3 * ux, 0.7 * vy + 0.3 * uy
            nv = math.hypot(vx, vy)
            vx, vy = vx / nv, vy / nv
        pts = np.array(pts[:good], np.float32)
        ln = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
        return pts, ln, (tot / nval if nval else 0.0)

    def _tail(self, d, body, c, u, span):
        """The best tail at each end of the body: {-1: (points, length, line strength, start) or None, +1: ...}.
        The walk keeps a little away from the body (a band of 0.1 body length, open only where it starts), so it
        cannot creep along the body's own dark edge."""
        out = {-1: None, 1: None}
        R = _ridge_map(d, self.ks)
        ring = (cv2.dilate(body, K3, iterations=2) > 0) & (cv2.dilate(body, K3, iterations=1) == 0)
        ys, xs = np.nonzero(ring & (R >= self.weak))
        if not len(xs):
            return out
        P = np.column_stack([xs, ys]).astype(np.float32)
        t = ((P - c) @ u) / max(1.0, 0.5 * span)                 # -1 .. 1 along the body
        keep = np.abs(t) >= 0.45                                   # near an end, not on the flank
        if not keep.any():
            return out
        P, t = P[keep], t[keep]
        order = np.argsort(-R[P[:, 1].astype(int), P[:, 0].astype(int)])
        band = max(2.0, 0.1 * self.L)
        kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(band)) + 1,) * 2)
        near_body = cv2.dilate(body, kb)
        starts = []
        sep = max(2.0, 0.15 * self.L)
        for o in order:                                            # the strongest few starts, a little apart
            p = P[o]
            if any(abs(p[0] - q[0]) + abs(p[1] - q[1]) < sep for q in starts):
                continue
            starts.append(p)
            if len(starts) > 5:
                break
            sg = 1 if t[o] > 0 else -1
            blocked = near_body.copy()
            cv2.circle(blocked, (int(p[0]), int(p[1])), int(round(band)) + 2, 0, -1)
            blocked |= cv2.dilate(body, K3)
            v = p - c
            v /= max(1e-6, float(np.hypot(*v)))
            pts, ln, mv = self._follow(R, blocked, p, v, 1.5 * self.L)
            if mv < 0.9 * self.strong or ln < 0.25 * self.L:
                continue
            if out[sg] is None or ln > out[sg][1]:
                out[sg] = (pts, ln, mv, p)
        return out

    # ---- one frame
    def find(self, gray, x, y):
        """-> (x0, y0, body uint8, tail uint8, dots (N_KP, 2) work coords) or None."""
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        r = self.win
        x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
        x1, y1 = min(self.W, int(x) + r + 1), min(self.H, int(y) + r + 1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        g, b = gray[y0:y1, x0:x1], self.bg[y0:y1, x0:x1]
        d = _signed(g, b, self.pol).astype(np.float32)
        if self.inside is not None:                     # masked parts (reflections, lamp ...) are not the animal
            d[~self.inside[y0:y1, x0:x1]] = 0.0
        ds = self.ds
        if ds < 1.0:
            size = (max(4, int(round((x1 - x0) * ds))), max(4, int(round((y1 - y0) * ds))))
            d = cv2.resize(d, size, interpolation=cv2.INTER_AREA)
        px, py = (x - x0 + 0.5) * ds - 0.5, (y - y0 + 0.5) * ds - 0.5
        c = _core_contrast(d, px, py, max(2.0, 0.12 * self.L))
        if not np.isfinite(c):
            c = self.cal.contrast
        c = min(max(c, 0.5 * self.cal.contrast), 1.5 * self.cal.contrast)
        T = max(self.T_lo, BODY_EDGE * c)
        sil = None
        for _k in range(4):                     # much bigger than the animal (shadow / wall joined): cut higher
            sil = _silhouette(d, px, py, T, self.L)
            if sil is None or sil.sum() <= 1.8 * self.area_ds:
                break
            T *= 1.2
        if sil is None:
            return None
        body = cv2.morphologyEx(sil, cv2.MORPH_OPEN, self.k_open)
        nb, lb, sb, _cb = cv2.connectedComponentsWithStats(body, 8)
        if nb > 1:
            body = (lb == 1 + int(np.argmax(sb[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        if body.sum() < 0.25 * sil.sum():
            body = sil.copy()
        dots, tail_pts = self._dots(d, sil, body)
        tail = np.zeros_like(body)
        if tail_pts is not None and len(tail_pts) > 1:
            cv2.polylines(tail, [np.round(tail_pts).astype(np.int32)], False, 1, max(1, int(round(0.08 * self.L))))
            tail &= (1 - body)
        if ds < 1.0:
            full = (x1 - x0, y1 - y0)
            body = cv2.resize(body, full, interpolation=cv2.INTER_NEAREST)
            tail = cv2.resize(tail, full, interpolation=cv2.INTER_NEAREST)
            if dots is not None:
                dots = (dots + 0.5) / ds - 0.5
        if dots is not None:
            dots[:, 0] += x0
            dots[:, 1] += y0
            dots = self._steady(dots)
        return (x0, y0, body, tail, dots)

    def _steady(self, dots):
        """A dot keeps its place while it moves less than the picture's jitter (still = still)."""
        prev = self.prev
        if prev is not None:
            ok = np.isfinite(prev[:, 0]) & np.isfinite(dots[:, 0])
            keep = ok & (np.hypot(*(dots - prev).T) < max(0.8, 0.04 * self.Lw))
            dots[keep] = prev[keep]
        self.prev = dots.copy()
        return dots

    def _dots(self, d, sil, body):
        mo = cv2.moments(body, True)
        if mo["m00"] < 6:
            return None, None
        c = np.array([mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]], np.float32)
        ang = 0.5 * math.atan2(2.0 * mo["mu11"], mo["mu20"] - mo["mu02"])
        u = np.array([math.cos(ang), math.sin(ang)], np.float32)
        cnts, _h = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None, None
        P = max(cnts, key=len).reshape(-1, 2).astype(np.float32)          # the body outline
        proj = (P - c) @ u
        lo, hi = proj.min(), proj.max()
        span = max(1.0, hi - lo)
        e1 = P[proj <= lo + 0.03 * span].mean(axis=0)
        e2 = P[proj >= hi - 0.03 * span].mean(axis=0)
        tails = self._tail(d, body, c, u, span)
        hint = None if self._hint is None else np.asarray(self._hint, np.float32)
        ends = {-1: e1, 1: e2}
        prior = None                                  # the end the tail was at (the nose was at the other one)
        if hint is not None:
            prior = 1 if math.hypot(*(hint - e1)) < math.hypot(*(hint - e2)) else -1
        vote = False
        if prior is None:
            ok = [sg for sg in (-1, 1) if tails[sg] is not None]
            side = max(ok, key=lambda sg: tails[sg][1]) if ok else None
        elif tails[prior] is not None:
            side = prior
        elif tails[-prior] is not None and tails[-prior][1] >= 0.6 * self.L:
            # turned round?  Only on a clearly long tail at the other end, seen in 3 frames in a row (an animal
            # does not turn round from one frame to the next; a line on the floor near its head comes and goes)
            vote = True
            self._turn += 1
            side = -prior if self._turn >= 3 else None
        else:
            side = None
        if not vote or side is not None:
            self._turn = 0
        tail_pts = None
        if side is not None:
            tl = tails[side]
            tb = P[int(np.argmin(np.hypot(*(P - tl[3]).T)))]              # where the tail leaves the body
            far = np.hypot(*(P - tb).T)
            nose = P[far >= far.max() - 0.04 * span].mean(axis=0)
            tail_pts = np.vstack([tb[None, :], tl[0]])
        else:                                         # no tail seen: the head stays where it was
            tsg = prior if prior is not None else -1
            nose, tb = ends[-tsg], ends[tsg]
        out = np.full((N_KP, 2), np.nan, np.float32)
        out[0], out[3], out[6] = nose, c, tb
        f = nose - tb
        flen = float(math.hypot(f[0], f[1]))
        if flen > 1e-3:
            fu = f / flen
            left = np.array([fu[1], -fu[0]], np.float32)
            cs, _h = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            Q = max(cs, key=len).reshape(-1, 2).astype(np.float32) if cs else P   # full outline (paws stick out)
            t = ((nose - Q) @ fu) / flen                              # 0 at the nose, 1 at the tail base
            side = (Q - nose) @ left
            inward = max(0.6, 0.03 * self.L)
            for il, ir, a0, a1 in ((1, 2, 0.15, 0.42), (4, 5, 0.58, 0.85)):
                sel = (t >= a0) & (t <= a1)
                if sel.sum() >= 2:
                    Qs, ss = Q[sel], side[sel]
                    out[il] = Qs[int(np.argmax(ss))] - left * inward
                    out[ir] = Qs[int(np.argmin(ss))] + left * inward
        if tail_pts is not None and len(tail_pts) > 1:
            seg = np.hypot(*np.diff(tail_pts, axis=0).T)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            step = TAIL_STEP * self.L
            k, dd = 0, step
            while dd <= s[-1] + 0.25 * step and k < N_KP - 7:
                q = min(dd, s[-1])
                out[7 + k] = [np.interp(q, s, tail_pts[:, 0]), np.interp(q, s, tail_pts[:, 1])]
                k += 1
                dd += step
        return out, tail_pts

    def locate(self, gray, x, y):
        """find() with the previous nose (keeps the head where it was when no tail is seen)."""
        self._hint = None
        if self.prev is not None and np.isfinite(self.prev[0, 0]) and np.isfinite(x):
            r = self.win
            self._hint = ((self.prev[0, 0] - max(0, int(x) - r) + 0.5) * self.ds - 0.5,
                          (self.prev[0, 1] - max(0, int(y) - r) + 0.5) * self.ds - 0.5)
        return self.find(gray, x, y)


def part_hists(q, cur, prev, body_model):
    """Changed-pixel histograms of the body parts between two frames ((x0, y0, body, tail, dots) each): body
    (head + legs + trunk), head, legs, tail.  Both outlines count, so the moving edges are inside."""
    if cur is None:
        return None
    items = [cur] + ([prev] if prev is not None else [])
    X0 = min(it[0] for it in items)
    Y0 = min(it[1] for it in items)
    X1 = max(it[0] + it[2].shape[1] for it in items)
    Y1 = max(it[1] + it[2].shape[0] for it in items)
    B = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    Tm = np.zeros_like(B)
    for x0, y0, body, tail, _d in items:
        h, w = body.shape
        B[y0 - Y0:y0 - Y0 + h, x0 - X0:x0 - X0 + w] |= body
        Tm[y0 - Y0:y0 - Y0 + h, x0 - X0:x0 - X0 + w] |= tail
    k3 = body_model.k3
    B = cv2.dilate(B, k3)
    Tm = cv2.dilate(Tm, k3) & (1 - B)
    dots = cur[4]
    H = np.zeros_like(B)
    Lg = np.zeros_like(B)
    if dots is not None:
        if np.isfinite(dots[0, 0]):
            cv2.circle(H, (int(round(dots[0, 0] - X0)), int(round(dots[0, 1] - Y0))),
                       int(round(body_model.r_head)), 1, -1)
        for j in (1, 2, 4, 5):
            if np.isfinite(dots[j, 0]):
                cv2.circle(Lg, (int(round(dots[j, 0] - X0)), int(round(dots[j, 1] - Y0))),
                           int(round(body_model.r_leg)), 1, -1)
    H &= B
    Lg &= B
    ins = getattr(body_model, "inside", None)
    if ins is not None:                                   # polygon / masks: only the region's pixels
        ins = ins[Y0:Y1, X0:X1].astype(np.uint8)
        B &= ins
        Tm &= ins
        H &= ins
        Lg &= ins
    qq = q[Y0:Y1, X0:X1]
    out = {}
    for name, m in (("body", B), ("head", H), ("legs", Lg), ("tail", Tm)):
        sel = m.astype(bool)
        out[name] = (np.bincount(qq[sel], minlength=NBINS), int(sel.sum()))
    return out


def frame_stats(geo, q):
    """Per frame: histogram of the region, the noise histogram (quiet tiles) and the typical tile's change."""
    if geo.inside is None:
        th = np.bincount((geo.tile_idx + q).ravel(), minlength=geo.n_tiles * NBINS).reshape(geo.n_tiles, NBINS)
        tc = _rcum(th, axis=1) / geo.tile_px[:, None]
        rate = np.median(tc, axis=0)
        keep = np.argsort(tc[:, ACTIVITY_LEVEL], kind="stable")[:max(1, int(round(NOISE_KEEP * geo.n_tiles)))]
        return th.sum(axis=0), th[keep].sum(axis=0), int(geo.tile_px[keep].sum()), rate
    # polygon / masks: only the pixels inside; tiles mostly inside give the noise and the typical change
    th = np.bincount((geo.tile_idx + q)[geo.inside], minlength=geo.n_tiles * NBINS).reshape(geo.n_tiles, NBINS)
    ok = np.flatnonzero(geo.tile_ok)
    tc = _rcum(th[ok], axis=1) / geo.tile_px[ok][:, None]
    rate = np.median(tc, axis=0)
    keep = ok[np.argsort(tc[:, ACTIVITY_LEVEL], kind="stable")[:max(1, int(round(NOISE_KEEP * len(ok))))]]
    return th.sum(axis=0), th[keep].sum(axis=0), int(geo.tile_px[keep].sum()), rate


OVERLAP = 8                   # frames read by two neighbouring pieces: their pictures must be identical


def _frame_key(frame):
    return hashlib.blake2b(frame.tobytes(), digest_size=12).digest()


class _Feeder:
    """The per-frame motion statistics of one piece of the time window, frames [a, b): fed the decoded frames in
    order (from any frame <= a - lag on).  Used by _process_range (its own reading) and by the one-pass analysis
    (the frames the tracking reads anyway, see _FusedMotion)."""

    def __init__(self, geo, lag, a, b, out, k, done, keys, colour="gray", wx=None, wy=None, disc=None,
                 make_body=None, live_k=None, mark=None):
        self.geo, self.lag, self.a, self.b, self.out, self.k = geo, lag, a, b, out, k
        self.done, self.keys, self.colour = done, keys, colour
        self.wx, self.wy, self.disc = wx, wy, disc
        self.mark = mark                       # frames whose circle histogram was measured (one pass, see below)
        self.body = make_body() if make_body is not None else None
        self.live_k = live_k
        self.ring = []
        self.base = out["start_frame"]
        self.last_fn = None

    def feed(self, fn, f):
        """Frame number fn (consecutive calls: fn, fn + 1, ...) and its BGR picture."""
        geo, lag, a, b, out = self.geo, self.lag, self.a, self.b, self.out
        if self.last_fn is not None and fn != self.last_fn + 1:
            raise RuntimeError("frames out of order")
        self.last_fn = fn
        if fn >= b:
            return
        if a - OVERLAP <= fn < a or b - OVERLAP <= fn < b:
            self.keys[fn] = _frame_key(f)
        if fn < a - lag - OVERLAP:
            return
        img = geo.prep(f, self.colour)
        i = fn - self.base
        seg = None
        body = self.body
        if body is not None and 0 <= i < len(self.wx):
            seg = body.locate(as_gray(img), self.wx[i], self.wy[i])
        ring = self.ring
        ring.append((img, seg))
        if len(ring) > lag + 1:
            ring.pop(0)
        if fn >= a:
            if 0 <= i < len(out["bright"]):
                out["bright"][i] = geo.brightness(img)
            if len(ring) == lag + 1:
                q = quantise(frame_diff(ring[0][0], img))
                hr, hb, nb, rate = frame_stats(geo, q)
                out["h_reg"][i], out["h_bg"][i], out["n_bg"][i], out["rate"][i] = hr, hb, nb, rate
                disc = self.disc
                if disc is not None:
                    j = max(0, i - lag)
                    out["h_disc"][i], out["n_disc"][i] = disc.hist(q, (self.wx[i], self.wy[i]),
                                                                    (self.wx[j], self.wy[j]))
                    if self.mark is not None:
                        self.mark[i] = True
                if body is not None:
                    ph = part_hists(q, seg, ring[0][1], body)
                    if ph is not None:
                        for name in PARTS:
                            hk, nk = PART_HIST[name]
                            out[hk][i], out[nk][i] = ph[name]
                    if seg is not None and seg[4] is not None:
                        out["kp"][i] = seg[4]
                out["valid"][i] = True
                lv = out.get("live")
                if lv is not None and (self.live_k is None or self.k == self.live_k[0]):
                    lv.frame(out, i, fn, f, ring[0][0], img, seg)
            pr = out.get("probe")
            disc = self.disc
            if pr is not None and disc is not None and i in pr and pr[i] is None and self.wx is not None:
                x, y = float(self.wx[i]), float(self.wy[i])
                if math.isfinite(x) and math.isfinite(y) and x > FAR / 2:
                    r = int(math.ceil(PROBE_R * math.sqrt(disc.r2))) + 2
                    g = img if img.ndim == 2 else img.mean(axis=2)
                    x0, y0 = max(0, int(x) - r), max(0, int(y) - r)
                    crop = g[y0:min(g.shape[0], int(y) + r + 1), x0:min(g.shape[1], int(x) + r + 1)]
                    if crop.size:
                        pr[i] = (crop.copy(), x - x0, y - y0, x0, y0)
            out["read"][i] = True
            self.done[self.k] += 1


def _process_range(path, geo, lag, a, b, s0, wx, wy, disc, out, done, k, stop, cancel, keys, colour="gray",
                   make_body=None, live_k=None):
    """Frames [a, b) (reading starts at s0 <= a so the first compared picture exists).  The pictures of the frames
    just before a and just before b are fingerprinted (keys) to check that seeking landed exactly."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
        pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or s0))
        while pos < s0:                                   # some codecs land a bit early
            if not cap.grab():
                break
            pos += 1
        if pos != s0:
            return "seek"
        feeder = _Feeder(geo, lag, a, b, out, k, done, keys, colour, wx, wy, disc, make_body, live_k)
        fn = s0
        while fn < b:
            if stop.is_set():
                return "stopped"
            if cancel is not None and k == 0 and (fn - s0) % 16 == 0 and cancel():
                stop.set()
                return "cancel"
            ok, f = cap.read()
            if not ok:
                break
            feeder.feed(fn, f)
            fn += 1
        return "ok"
    finally:
        cap.release()


class MotionData:
    """Everything the freezing result is computed from (one video, one time window)."""

    def has_disc(self):
        return self.h_disc is not None

    def has_body(self):
        return getattr(self, "h_body", None) is not None

    def body_area_work(self, body_length_px=0.0, area="region"):
        """Body area at work scale: measured length > (body dots) the body outline > tracking > (unknown) 2.5 % of
        the region."""
        if body_length_px and body_length_px > 0:
            return AREA_PER_LEN2 * float(body_length_px) ** 2 * self.geo.s ** 2
        if area in PART_HIST and (getattr(self, "body_area_dots", 0.0) or 0.0) > 0:
            return float(self.body_area_dots) * self.geo.s ** 2
        if self.body_area_full > 0:
            return self.body_area_full * self.geo.s ** 2
        return 0.025 * self.n_reg

    def auto_cutoff(self):
        """(cutoff, noise level) -- see auto_cutoff_info()."""
        a = self.auto_cutoff_info()
        return a["cutoff"], a["level"]

    def auto_cutoff_info(self):
        """2 x the 99.99th percentile of the grey changes of the quiet part of each frame (= the classic
        calibration, done on the video itself: no empty video is needed).  The 75th percentile over the frames
        keeps compression pulses in, but not the odd frame with a hand or a light switch.
        Spiky (compressed / webcam) noise: when the 99.99th percentile is more than TAIL_RATIO x the 99.9th, the
        rare spikes -- not the noise -- set it; it is then limited to TAIL_RATIO x the 99.9th percentile (normal
        camera noise is about 1.2 x: unchanged).  -> dict(cutoff, level, p9999, p999, spiky, at_limit)."""
        nan = float("nan")
        out = {"cutoff": 10, "level": nan, "p9999": nan, "p999": nan, "spiky": False, "at_limit": False}
        ok = self.valid & (self.n_bg > 200)
        if not ok.any():
            return out
        cum = _rcum(self.h_bg[ok].astype(np.int64), axis=1)
        p4 = _percentile_from_cum(cum, self.n_bg[ok], 1e-4)
        p3 = _percentile_from_cum(cum, self.n_bg[ok], 1e-3)
        good = np.isfinite(p4) & np.isfinite(p3)
        if not good.any():
            return out
        l4, l3 = float(np.percentile(p4[good], 75)), float(np.percentile(p3[good], 75))
        level = min(l4, TAIL_RATIO * l3) if l3 > 0 else l4
        raw = round(2.0 * level)
        out.update(cutoff=int(min(CUTOFF_MAX, max(CUTOFF_MIN, raw))), level=level, p9999=l4, p999=l3,
                   spiky=bool(l3 > 0 and l4 > TAIL_RATIO * l3), at_limit=bool(raw >= CUTOFF_MAX))
        return out

    def motion(self, cutoff, area="region", body_length_px=0.0):
        """(motion as % of the body area per frame, the typical tile's share of changed pixels per frame)."""
        c = int(min(NBINS - 1, max(1, int(cutoff))))
        rate = self.rate[:, c].astype(float)
        if area == "animal" and self.has_disc():
            px = _rcum(self.h_disc[:, c:].astype(np.int64), axis=1)[:, 0] - self.n_disc * rate
        elif area in PART_HIST and self.has_body():
            hk, nk = PART_HIST[area]
            px = _rcum(getattr(self, hk)[:, c:].astype(np.int64), axis=1)[:, 0] - getattr(self, nk) * rate
        else:
            px = _rcum(self.h_reg[:, c:].astype(np.int64), axis=1)[:, 0] - self.n_reg * rate
        px = np.maximum(0.0, px)
        pct = 100.0 * px / max(1e-6, self.body_area_work(body_length_px, area))
        pct[~self.valid] = np.nan
        return pct, rate


def part_threshold(threshold_pct):
    """A body part 'moved' when its changed pixels reach this % of the body area."""
    return max(1.0, 0.5 * float(threshold_pct))


def draw_region(disp, region=None, geo=None):
    """The region (orange outline: box or polygon) and the masks (dimmed, red edge) on a video-size picture."""
    if geo is not None:
        poly, masks = geo.video_polys()
    else:
        poly, masks = region_parts(region)
    lw = max(1, int(round(max(disp.shape[:2]) / 900)))
    for mk in masks:
        pts = np.round(np.asarray(mk)).astype(np.int32)
        m = np.zeros(disp.shape[:2], np.uint8)
        cv2.fillPoly(m, [pts], 1)
        sel = m.astype(bool)
        disp[sel] = (0.35 * disp[sel]).astype(np.uint8)
        cv2.polylines(disp, [pts], True, (70, 70, 230), lw, cv2.LINE_AA)
    if poly is not None:
        cv2.polylines(disp, [np.round(np.asarray(poly)).astype(np.int32)], True, (0, 200, 255), lw, cv2.LINE_AA)
    return disp


BODY_PART_OF = {0: "head", 1: "legs", 2: "legs", 4: "legs", 5: "legs", 3: "body", 6: "body"}   # else tail
BODY_LINKS = [(1, 2), (4, 5)]                     # a bar through each pair of legs


def body_chain(P):
    """Nose - body centre - tail base - tail dots, then the leg bars: the lines that join the dots."""
    chain = [0, 3, 6] + [j for j in range(7, N_KP) if np.isfinite(P[j, 0])]
    return list(zip(chain, chain[1:])) + BODY_LINKS


def body_overlay(geo, dots, moved=None, body_len_full=0.0):
    """The body dots for a picture viewer, in video px: {"pts": (N_KP, 2), "moved": {part: bool}, "len": body
    length}.  The viewer draws them at screen resolution (small and sharp at any zoom)."""
    if dots is None or not np.isfinite(dots[0, 0]):
        return None
    P = np.asarray(dots, float) / geo.s + np.array([geo.x0, geo.y0])
    return {"pts": P, "moved": dict(moved or {}), "len": float(body_len_full or 0.0)}


def draw_body(disp, geo, dots, moved=None, body_len_full=0.0):
    """The body dots drawn into a picture (exports): joined like a mouse; green = still, red = that part moved
    (moved = {"head", "legs", "body", "tail"} -> bool).  Small dots, placed to 1/16 px."""
    ov = body_overlay(geo, dots, moved, body_len_full)
    if ov is None:
        return disp
    P, moved = ov["pts"], ov["moved"]
    still, mov, line = (90, 205, 90), (60, 60, 235), (235, 235, 235)
    r = max(1.5, 0.03 * body_len_full) if body_len_full > 0 else max(1.5, max(disp.shape[:2]) / 400.0)
    sh = 4

    def pt(j):
        return (int(round(P[j, 0] * 16)), int(round(P[j, 1] * 16))) if np.isfinite(P[j, 0]) else None
    for a, b in body_chain(P):
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(disp, pa, pb, line, 1, cv2.LINE_AA, sh)
    for j in range(N_KP):
        p = pt(j)
        if p is None:
            continue
        if moved.get(BODY_PART_OF.get(j, "tail")):       # moved: a red diamond (shape + colour: colour-blind safe)
            q = 1.45 * (r + 0.8) * 16
            dia = np.array([[p[0], p[1] - q], [p[0] + q, p[1]], [p[0], p[1] + q], [p[0] - q, p[1]]], np.int32)
            cv2.fillConvexPoly(disp, dia, (20, 20, 20), cv2.LINE_AA, sh)
            q = 1.45 * r * 16
            dia = np.array([[p[0], p[1] - q], [p[0] + q, p[1]], [p[0], p[1] + q], [p[0] - q, p[1]]], np.int32)
            cv2.fillConvexPoly(disp, dia, mov, cv2.LINE_AA, sh)
        else:                                            # still: a green dot
            cv2.circle(disp, p, int(round((r + 0.8) * 16)), (20, 20, 20), -1, cv2.LINE_AA, sh)
            cv2.circle(disp, p, int(round(r * 16)), still, -1, cv2.LINE_AA, sh)
    return disp


class _LiveState:
    """The live picture while the motion is measured: moving pixels in red, the body dots (red = that part
    moved) and FREEZING / ACTIVE as it happens (the cutoff is estimated from the frames seen so far)."""

    def __init__(self, live, fps, settings, geo, body_area_work, body_len_full):
        s = merged_settings(settings)
        self.live, self.fps, self.geo = live, float(fps), geo
        self.auto = bool(s.get("cutoff_auto", True))
        self.cut = int(s.get("cutoff") or 10)
        self.thr = float(s["threshold_pct"])
        self.part_thr = part_threshold(self.thr)
        self.min_f = max(1, int(round(float(s["min_freeze_s"]) * self.fps)))
        self.area = s.get("motion_area", "region")
        self.body_area = max(1e-6, float(body_area_work))
        self.body_len = float(body_len_full or 0.0)
        self.still_run = 0
        self.levels = []
        self.n_seen = 0

    def _cutoff(self):
        if not self.auto:
            return self.cut
        if len(self.levels) < 5:
            return 10
        return int(min(CUTOFF_MAX, max(CUTOFF_MIN, round(2.0 * float(np.percentile(self.levels, 75))))))

    def _pct(self, h, n, rate_c, c):
        px = float(h[c:].sum()) - float(n) * rate_c
        return 100.0 * max(0.0, px) / self.body_area

    def frame(self, out, i, fn, frame, prev_img, img, seg):
        self.n_seen += 1
        if self.auto and self.n_seen % 10 == 1 and out["n_bg"][i] > 200:
            cum = _rcum(out["h_bg"][i:i + 1].astype(np.int64), axis=1)
            p4 = _percentile_from_cum(cum, out["n_bg"][i:i + 1], 1e-4)
            p3 = _percentile_from_cum(cum, out["n_bg"][i:i + 1], 1e-3)
            if np.isfinite(p4[0]):
                self.levels.append(min(float(p4[0]), TAIL_RATIO * float(p3[0])) if np.isfinite(p3[0])
                                   and p3[0] > 0 else float(p4[0]))
        c = int(min(NBINS - 1, max(1, self._cutoff())))
        rate_c = float(out["rate"][i][c])
        if self.area in PART_HIST and out.get("h_body") is not None and out["n_body"][i] > 0:
            mot = self._pct(out["h_body"][i], out["n_body"][i], rate_c, c)
        elif self.area == "animal" and out.get("h_disc") is not None:
            mot = self._pct(out["h_disc"][i], out["n_disc"][i], rate_c, c)
        else:
            mot = self._pct(out["h_reg"][i], self.geo.n_px, rate_c, c)
        light = rate_c > LIGHT_FRACTION
        if not light:
            self.still_run = self.still_run + 1 if mot < self.thr else 0
        moved = {}
        if out.get("h_body") is not None:
            for name in PARTS:
                hk, nk = PART_HIST[name]
                if out[nk][i] > 0:
                    moved[name] = self._pct(out[hk][i], out[nk][i], rate_c, c) >= self.part_thr
        if not self.live.due():
            return
        disp = frame.copy()
        geo = self.geo
        m = change_mask(prev_img, img, c, geo.inside).astype(np.uint8)
        if geo.s < 1.0:
            m = cv2.resize(m, (geo.x1 - geo.x0, geo.y1 - geo.y0), interpolation=cv2.INTER_NEAREST)
        sub = disp[geo.y0:geo.y1, geo.x0:geo.x1]
        mm = m.astype(bool)
        sub[mm] = (0.35 * sub[mm] + np.array([14, 14, 152])).astype(np.uint8)
        draw_region(disp, geo=geo)
        body = body_overlay(geo, seg[4], moved, self.body_len) if seg is not None else None
        frz = self.still_run >= self.min_f
        big = "LIGHT CHANGE" if light else ("FREEZING" if frz else ("STILL" if self.still_run else "ACTIVE"))
        parts = [n for n in ("head", "legs", "tail") if moved.get(n)]
        small = f"motion {mot:.1f} % of body" + (f"  ·  moved: {', '.join(parts)}" if parts else "")
        t = fn / self.fps
        self.live(disp, f"Measuring motion…  {int(t // 60):02d}:{t % 60:04.1f}  (red = moving pixels)",
                  status=(big, small, frz, body))


def measure_motion(path, settings, region=None, loc=None, progress=None, cancel=None, workers=None, live=None):
    """Grey-change statistics for every frame of the time window (`loc` = locate_animal() or None)."""
    from tracking.location import worker_count
    info = video_info(path)
    fps = info["fps"]
    lag = frame_lag(fps)
    geo = Geometry(info["w"], info["h"], region)
    if loc is not None and len(loc["frames"]):
        start, end = int(loc["frames"][0]), int(loc["frames"][-1])
    else:
        start, end = time_window(info, settings.get("start_time"), settings.get("end_time"))
    n = end - start + 1
    if n < lag + 2:
        raise RuntimeError("The time window is too short.")
    hx = hy = wx = wy = disc = None
    radius = 0.0
    if loc is not None:
        hx, hy = hold_positions(loc["xs"], loc["ys"])
        if hx is not None and len(hx) < n:                 # tracking stopped early: hold its last position
            hx = np.concatenate([hx, np.full(n - len(hx), hx[-1])])
            hy = np.concatenate([hy, np.full(n - len(hy), hy[-1])])
    body_len = float(settings.get("body_length_px") or 0.0) or (loc["body_len"] if loc else 0.0)
    colour = "rgb" if settings.get("colour") == "rgb" else "gray"
    make_body = None
    cal = None
    if hx is not None and body_len > 0:
        wx, wy = geo.to_work(hx[:n], hy[:n])
        radius = DISC_FACTOR * body_len * geo.s
        disc = _Disc(geo.ww, geo.wh, max(3.0, radius), geo.inside)
        # body dots only when they are used (Body dots mode): they cost time and mean nothing in the other modes
        if (settings.get("motion_area") == "body" and loc is not None and loc.get("background") is not None
                and body_len * geo.s >= 6):
            bg, pol, con = loc["background"], loc.get("polarity"), loc.get("min_contrast", 12.0)
            if bg.shape[0] == info["h"] and bg.shape[1] == info["w"]:
                tx, ty = geo.to_work(np.asarray(loc["xs"], float), np.asarray(loc["ys"], float))
                cal = calibrate_body(path, geo, loc["frames"], tx, ty, bg, pol, con,
                                     float(settings.get("body_length_px") or loc["body_len"] or body_len) * geo.s)
                if cal.L >= 6:
                    def make_body():
                        return _Body(geo, bg, pol, con, cal)
                else:
                    cal = None
    body = make_body is not None
    dots_area = cal.area if (cal is not None and cal.ok) else 0.0          # work px
    out = _motion_arrays(n, start, disc is not None, body)
    if live is not None:
        if float(settings.get("body_length_px") or 0) > 0:
            ba = AREA_PER_LEN2 * float(settings["body_length_px"]) ** 2 * geo.s ** 2
        elif settings.get("motion_area") == "body" and dots_area > 0:
            ba = dots_area
        elif loc is not None and loc.get("area", 0) > 0:
            ba = float(loc["area"]) * geo.s ** 2
        else:
            ba = 0.025 * geo.n_px
        out["live"] = _LiveState(live, fps, settings, geo, ba,
                                 (cal.L / geo.s) if (cal is not None and cal.ok) else body_len)
    keys = MOTION_KEYS

    def run(nw):
        bounds = [start + k * n // nw for k in range(nw)] + [end + 1]
        done = [0] * nw
        stop = threading.Event()
        res = [None] * nw
        errors = []
        live_k = [0]                 # the live picture shows the first piece (in order); the others run unseen
        finished = [False] * nw

        fkeys = [dict() for _ in range(nw)]

        def work(k):
            a, b = bounds[k], bounds[k + 1]
            s0 = start if k == 0 else max(start, a - lag - OVERLAP)
            try:
                res[k] = _process_range(path, geo, lag, a, b, s0, wx, wy, disc, out, done, k, stop, cancel,
                                        fkeys[k], colour=colour, make_body=make_body, live_k=live_k)
            except BaseException as exc:
                errors.append(exc)
                stop.set()
            finally:
                finished[k] = True     # the live picture stays on the first piece: it plays in order, never jumps
        threads = [threading.Thread(target=work, args=(k,), name=f"bt-freeze{k}", daemon=True)
                   for k in range(nw)]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                for t in threads:
                    t.join(timeout=0.2)
                if progress is not None:
                    progress(sum(done) / max(1, n))
        except BaseException:
            stop.set()
            for t in threads:
                t.join(timeout=10)
            raise
        if errors:
            raise errors[0]
        if "cancel" in res:
            raise Cancelled()
        for k in range(1, nw):                            # seeking must land exactly (MPEG-1 / WMV often do not)
            a = bounds[k]
            for fn in range(max(start, a - OVERLAP), a):
                if fkeys[k - 1].get(fn) is None or fkeys[k - 1].get(fn) != fkeys[k].get(fn):
                    res[k] = "seek"
                    break
        return res

    nw = int(workers or settings.get("_workers") or worker_count(n))
    res = run(nw)
    if nw > 1 and "seek" in res:                          # the video cannot be positioned exactly: one piece
        for key in keys:
            if out[key] is not None:
                out[key][:] = 0
        res = run(1)
    return _assemble(out, path, info, geo, lag, start, loc, hx, hy, radius, body_len if disc is not None else 0.0,
                     dots_area, colour)


MOTION_KEYS = ("h_reg", "h_bg", "n_bg", "rate", "h_disc", "n_disc", "valid", "read", "bright", "kp") + \
    tuple(k for name in PARTS for k in PART_HIST[name])


def _motion_arrays(n, start, disc, body):
    out = {"start_frame": start,
           "h_reg": np.zeros((n, NBINS), np.uint32), "h_bg": np.zeros((n, NBINS), np.uint32),
           "n_bg": np.zeros(n, np.int32), "rate": np.zeros((n, NBINS), np.float32),
           "h_disc": np.zeros((n, NBINS), np.uint32) if disc else None,
           "n_disc": np.zeros(n, np.int32) if disc else None,
           "valid": np.zeros(n, bool), "read": np.zeros(n, bool),
           "bright": np.full(n, np.nan, np.float32),
           "kp": np.full((n, N_KP, 2), np.nan, np.float32) if body else None}
    for name in PARTS:
        hk, nk = PART_HIST[name]
        out[hk] = np.zeros((n, NBINS), np.uint32) if body else None
        out[nk] = np.zeros(n, np.int32) if body else None
    # a few pictures around the animal, spread over the window: is the tracked body bigger than its solid part?
    out["probe"] = dict.fromkeys(np.unique(np.linspace(min(n - 1, max(0, n // 50)), n - 1, PROBE_N).astype(int)).tolist())
    return out


def _core_stats(geo, probes, loc):
    """(solid share of the tracked blob, solid length / tracked body length), medians over the probe pictures, or
    (nan, nan).  A reflection in a clear wall or an attached shadow is weaker than the animal itself: when it is
    counted with the animal the tracked body is bigger than its solid part (the strong half of the contrast)."""
    nan = float("nan")
    try:
        if not probes or loc is None or loc.get("background") is None or not loc.get("body_len"):
            return nan, nan
        bg = geo.prep(loc["background"], "gray")
        pol, con = loc.get("polarity"), float(loc.get("min_contrast") or 12.0)
        fr, rl = [], []
        for v in probes.values():
            if v is None:
                continue
            crop, cx, cy, x0, y0 = v
            b = bg[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]]
            if b.shape != crop.shape:
                continue
            d = _signed(crop, b, pol)
            m = (d > con).astype(np.uint8)
            nlab, lab = cv2.connectedComponents(m)
            if nlab <= 1:
                continue
            c = lab[min(int(round(cy)), lab.shape[0] - 1), min(int(round(cx)), lab.shape[1] - 1)]
            if c == 0:
                c = 1 + int(np.argmax(np.bincount(lab.ravel(), minlength=nlab)[1:]))
            blob = lab == c
            vals = d[blob]
            if vals.size < 12:
                continue
            strong = blob & (d > 0.5 * float(np.percentile(vals, 95)))
            fr.append(float(strong.sum()) / float(blob.sum()))
            rl.append(math.sqrt(float(strong.sum()) / AREA_PER_LEN2) / geo.s)
        if len(fr) < 8:
            return nan, nan
        return float(np.median(fr)), float(np.median(rl)) / float(loc["body_len"])
    except Exception:
        return nan, nan


def _assemble(out, path, info, geo, lag, start, loc, hx, hy, radius, disc_len, dots_area, colour):
    """The MotionData of a filled set of motion arrays."""
    if not out["read"].any():
        raise RuntimeError("No frames could be read in the selected time window.")
    m = int(np.max(np.where(out["read"])[0])) + 1          # the video may end before its header says
    md = MotionData()
    for key in MOTION_KEYS:
        setattr(md, key, None if out[key] is None else out[key][:m])
    md.path = str(path)
    md.fps, md.lag, md.geo = float(info["fps"]), lag, geo
    md.start_frame = start
    md.frames = np.arange(start, start + m, dtype=np.int64)
    md.n_reg = geo.n_px
    md.xs = None if hx is None else hx[:m]
    md.ys = None if hy is None else hy[:m]
    md.seen = (np.isfinite(np.asarray(loc["xs"], float))[:m] if loc is not None
               else np.zeros(m, bool))
    if len(md.seen) < m:
        md.seen = np.concatenate([md.seen, np.zeros(m - len(md.seen), bool)])
    md.radius_work = radius
    md.disc_len = disc_len
    md.tracked = loc is not None
    md.body_area_full = float(loc["area"]) if loc else 0.0
    md.body_len_full = float(loc["body_len"]) if loc else 0.0
    md.body_area_dots = dots_area / geo.s ** 2                  # from the body outline (body dots), video px
    md.body_len_dots = math.sqrt(md.body_area_dots / AREA_PER_LEN2) if dots_area > 0 else 0.0
    md.found_pct = float(loc["found_pct"]) if loc else float("nan")
    md.polarity = loc.get("polarity") if loc else None
    md.polarity_warning = loc.get("polarity_warning") if loc else None
    md.frame_w, md.frame_h, md.total_frames = info["w"], info["h"], info["total"]
    md.colour = colour
    md.core_frac, md.core_ratio = _core_stats(geo, out.get("probe"), loc)
    md.one_pass = False
    md.disc_provisional = False
    return md


FAR = -1.0e6                  # a disc 'position' outside every picture: an empty disc (no position known yet)
PROBE_N = 32                  # pictures kept to compare the tracked body with its solid part
PROBE_R = 1.5                 # ... a square of +- 1.5 body lengths around the animal
CORE_RATIO_MAX = 0.80         # warn: the solid part is shorter than this x the tracked body length ...
CORE_FRAC_MAX = 0.50          # ... and less than this share of the tracked blob is solid
PROV_N = 90                   # Classic, one pass: tracked frames that fix the provisional circle's size
REDO_MERGE = 300              # frames measured again: gaps shorter than this are read through (no extra seek)


def _light_key(frame):
    """A cheap fingerprint of a decoded frame (every 16th pixel): the same frame read twice must match."""
    return hashlib.blake2b(np.ascontiguousarray(frame[::16, ::16]).tobytes(), digest_size=8).digest()


class _FusedMotion:
    """One-pass analysis: the motion statistics are measured from the frames the tracking reads anyway, so the
    video is decoded once instead of twice.  The result is identical to measure_motion():
      * 'Whole region' (Classic): the motion does not depend on the tracking at all;
      * 'Around the animal' with a measured body length (the circle's size is known before the tracking): every
        frame uses the position known at that frame; the few frames whose final position differs (found again
        later, or before the animal was first seen) are measured again from the video.
    Anything unusual (a piece boundary decoded differently, too many frames to measure again, an error) -> finish()
    returns None and the caller measures the motion in a second pass as before."""

    def __init__(self, path, settings, region, info):
        self.path = str(path)
        self.info = info
        self.lag = frame_lag(info["fps"])
        self.geo = Geometry(info["w"], info["h"], region)
        self.start, self.end = time_window(info, settings.get("start_time"), settings.get("end_time"))
        self.n = self.end - self.start + 1
        self.colour = "rgb" if settings.get("colour") == "rgb" else "gray"
        measured = float(settings.get("body_length_px") or 0.0)
        self.disc, self.radius, self.body_len = None, 0.0, 0.0
        self.prov = False
        if settings.get("motion_area") in ("region", "animal") and measured > 0:    # (Classic: for a quick switch)
            self.body_len = measured
            self.radius = DISC_FACTOR * measured * self.geo.s
            self.disc = _Disc(self.geo.ww, self.geo.wh, max(3.0, self.radius), self.geo.inside)
        elif settings.get("motion_area", "region") == "region":
            # Classic with the body size from the tracking: the result does not depend on the circle; the circle
            # only feeds the 'movement elsewhere' check.  Its size is taken from the first PROV_N tracked frames
            # (the final size is known only at the end) -- md.disc_provisional tells the page so.
            self.prov = True
            self.prov_areas = []
        self.failed = None
        self.begin(1)

    def _set_provisional(self):
        """Enough tracked frames: fix the circle's size and hand it to every piece (called under the lock)."""
        a = np.asarray(self.prov_areas, float)
        area = float(np.percentile(a, 90))
        self.body_len = math.sqrt(area / AREA_PER_LEN2)
        self.radius = DISC_FACTOR * self.body_len * self.geo.s
        self.disc = _Disc(self.geo.ww, self.geo.wh, max(3.0, self.radius), self.geo.inside)
        for fd in self.feeders.values():
            fd.disc = self.disc

    def begin(self, workers):
        """A (new) tracking run starts: workers pieces."""
        self.out = _motion_arrays(self.n, self.start, self.disc is not None or self.prov, False)
        self.mark = np.zeros(self.n, bool) if self.prov else None
        self.feeders, self.keys, self.bounds, self.pos, self.last = {}, {}, {}, {}, {}
        self.lkeys = [None] * self.n
        self.done = [0] * max(1, int(workers))
        self.lock = threading.Lock()

    def __call__(self, k, a, b, fn, frame, det):
        if self.failed is not None:
            return
        try:
            fd = self.feeders.get(k)
            if fd is None:
                with self.lock:
                    a2, b2 = max(int(a), self.start), min(int(b), self.end + 1)
                    keys = {}
                    wxk = wyk = None
                    if self.disc is not None or self.prov:
                        wxk, wyk = np.full(self.n, FAR), np.full(self.n, FAR)
                        self.pos[k] = (wxk, wyk)
                    if k >= len(self.done):
                        self.done.extend([0] * (k + 1 - len(self.done)))
                    fd = _Feeder(self.geo, self.lag, a2, b2, self.out, k, self.done, keys, self.colour, wxk, wyk,
                                 self.disc, mark=self.mark)
                    self.keys[k], self.bounds[k], self.feeders[k] = keys, (a2, b2), fd
            i = fn - self.start
            if 0 <= i < self.n:
                if fn >= self.bounds[k][0] and fn < self.bounds[k][1]:
                    self.lkeys[i] = _light_key(frame)
                if self.disc is not None or self.prov:
                    wxk, wyk = self.pos[k]
                    x, y = det.get("x"), det.get("y")
                    if det.get("status") == "Tracked" and x is not None and math.isfinite(x) and math.isfinite(y):
                        p = self.geo.to_work(x, y)
                        self.last[k] = (float(p[0]), float(p[1]))
                        if self.prov and self.disc is None:
                            ar = det.get("area")
                            if ar is not None and math.isfinite(ar) and ar > 0:
                                with self.lock:
                                    if self.disc is None:
                                        self.prov_areas.append(float(ar))
                                        if len(self.prov_areas) >= PROV_N:
                                            self._set_provisional()
                    if k in self.last:
                        wxk[i], wyk[i] = self.last[k]
            fd.feed(fn, frame)
        except Exception as exc:
            self.failed = f"{type(exc).__name__}: {exc}"

    def finish(self, loc, progress=None):
        """-> MotionData (identical to a separate measure_motion pass) or None (measure in a second pass)."""
        self.why = None
        if self.failed is not None or not self.feeders:
            self.why = self.failed or "no frames"
            return None
        fr = np.asarray(loc["frames"], np.int64)
        if not len(fr) or int(fr[0]) != self.start or int(fr[-1]) > self.end:
            self.why = f"time window {fr[0] if len(fr) else None}-{fr[-1] if len(fr) else None} vs {self.start}-{self.end}"
            return None
        ks = sorted(self.bounds)
        for k1, k2 in zip(ks, ks[1:]):                    # neighbouring pieces must have decoded the same pictures
            a = self.bounds[k2][0]
            for fn in range(max(self.start, a - OVERLAP), a):
                if self.keys[k1].get(fn) is None or self.keys[k1].get(fn) != self.keys[k2].get(fn):
                    self.why = "piece boundary"
                    return None
        out, n, geo, lag = self.out, self.n, self.geo, self.lag
        hx = hy = None
        if self.prov and self.disc is None:
            self.why = "animal tracked in too few frames for the circle"
            return None
        if self.disc is not None:
            hx, hy = hold_positions(loc["xs"], loc["ys"])
            if hx is None:
                return None
            if len(hx) < n:
                hx = np.concatenate([hx, np.full(n - len(hx), hx[-1])])
                hy = np.concatenate([hy, np.full(n - len(hy), hy[-1])])
            wx, wy = geo.to_work(hx[:n], hy[:n])
            redo = np.zeros(n, bool)
            for k, (a2, b2) in self.bounds.items():
                wxk, wyk = self.pos[k]
                i = np.arange(a2 - self.start, b2 - self.start)
                j = np.maximum(0, i - lag)
                same = (wxk[i] == wx[i]) & (wyk[i] == wy[i]) & (wxk[j] == wx[j]) & (wyk[j] == wy[j])
                redo[i] = ~same
            if self.mark is not None:                    # measured before the provisional circle was known
                redo |= out["valid"] & ~self.mark
            redo &= out["read"]
            self.n_redo = int(redo.sum())
            if redo.any() and not self._redo(redo, wx, wy, progress):
                self.why = self.why or f"{self.n_redo} frames to measure again"
                return None
        md = _assemble(out, self.path, self.info, geo, lag, self.start, loc, hx, hy, self.radius,
                       self.body_len if self.disc is not None else 0.0, 0.0, self.colour)
        md.one_pass = True
        md.disc_provisional = bool(self.prov)
        return md

    def _redo(self, redo, wx, wy, progress=None):
        """Measure the frames whose circle position changed after they were measured (the circle around the final
        position).  False = too many / the video could not be positioned exactly."""
        idx = np.flatnonzero(redo)
        segs, a = [], int(idx[0])
        prev = a
        for i in idx[1:]:
            i = int(i)
            if i - prev > REDO_MERGE:
                segs.append((a, prev + 1))
                a = i
            prev = i
        segs.append((a, prev + 1))
        cost = sum(b - a + self.lag + OVERLAP + 250 for a, b in segs)      # + a keyframe interval per seek
        if cost > max(0.3 * self.n, 900):
            self.why = f"{len(idx)} frames in {len(segs)} pieces to measure again (cost {cost})"
            return False
        done = [0]
        for a, b in segs:
            fa, fb = self.start + a, self.start + b
            s0 = max(self.start, fa - self.lag - OVERLAP)
            cap = cv2.VideoCapture(self.path)
            try:
                cap.set(cv2.CAP_PROP_POS_FRAMES, s0)
                pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or s0))
                while pos < s0:
                    if not cap.grab():
                        break
                    pos += 1
                if pos != s0:
                    return False
                fd = _Feeder(self.geo, self.lag, fa, fb, self.out, 0, done, {}, self.colour, wx, wy, self.disc)
                fn = s0
                while fn < fb:
                    ok, f = cap.read()
                    if not ok:
                        return False
                    i = fn - self.start
                    if self.lkeys[i] is not None and self.lkeys[i] != _light_key(f):
                        return False                      # not the same picture as in the tracking pass
                    fd.feed(fn, f)
                    fn += 1
            finally:
                cap.release()
        return True


def needs_tracking(settings):
    """Tracking is needed for "around the animal" and when the body size is not measured."""
    return (settings.get("motion_area", "region") in ("animal", "body")
            or not float(settings.get("body_length_px") or 0.0) > 0)


class LiveFeed:
    """Live pictures for a viewer: `emit(bgr, text)` is called at most `rate` times a second and only while
    `on()` is true."""

    def __init__(self, emit, on=None, rate=8.0):
        import time as _t
        self._t = _t
        self.emit, self.on, self.gap, self.last = emit, on, 1.0 / rate, 0.0

    def due(self):
        if self.on is not None and not self.on():
            return False
        now = self._t.monotonic()
        if now - self.last >= self.gap:
            self.last = now
            return True
        return False

    def __call__(self, bgr, text="", status=None):
        try:
            self.emit(bgr if status is None else (bgr, status), text)
        except Exception:
            pass


def run_freezing(path, settings, region=None, progress=None, cancel=None, live=None, workers=None):
    """(Track +) measure motion for one video -> MotionData (then call analyse()).  `live` = LiveFeed or None.
    workers = 1: frame by frame in order (the live picture plays like a video)."""
    if workers:
        settings = dict(settings, _workers=int(workers))
    track = needs_tracking(settings)
    split = 0.6 if track else 0.0

    def p_track(f):
        if cancel is not None and cancel():
            raise Cancelled()
        if progress is not None:
            progress(split * f)

    def p_motion(f):
        if progress is not None:
            progress(split + (1.0 - split) * f)
    fused = None
    if track and one_pass_possible(settings):
        try:
            fused = _FusedMotion(path, settings, region, video_info(path))
        except Exception:
            fused = None
    if fused is not None:
        split = 0.97                                   # one pass: the tracking reads every frame once

        def p_track(f):                                # noqa: F811 (the same, with the new split)
            if cancel is not None and cancel():
                raise Cancelled()
            if progress is not None:
                progress(split * f)
    loc = locate_animal(path, settings, region=region, progress=p_track, live=live, fused=fused) if track else None
    md = fused.finish(loc) if fused is not None else None
    if fused is not None and md is None and os.environ.get("BT_DEBUG"):
        print("one pass not used:", fused.why)
    if md is None:
        split = 0.6 if track else 0.0
        if fused is not None and fused.why == "piece boundary" and not settings.get("_workers"):
            settings = dict(settings, _workers=1)      # this video cannot be positioned exactly: in one piece
        md = measure_motion(path, settings, region=region, loc=loc, progress=p_motion, cancel=cancel, live=live)
    if progress is not None:
        progress(1.0)
    return md


def one_pass_possible(settings):
    """The motion can be measured while tracking (one reading of the video): 'Whole region' (Classic) always, and
    'Around the animal' with a 📏 measured body length.  'Around the animal' without a measured length (the
    circle's size comes from the whole tracking) and 'Body outline points' (they need the tracking's body
    outline) read the video twice."""
    if settings.get("_two_pass"):
        return False
    area = settings.get("motion_area", "region")
    # with a measured length the circle's size is known before the tracking, so 'Around the animal' is measured in
    # the same pass for both modes (a quick switch between them, and the Classic 'movement elsewhere' check)
    return area == "region" or (area == "animal" and float(settings.get("body_length_px") or 0.0) > 0)


# ============================================================================= step 3: freezing
def _runs(mask):
    """[(start, stop)) runs of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def freezing_states(motion_pct, usable, fps, threshold_pct, min_freeze_s, ignore_blips_s):
    """Frame-by-frame freezing (bool) from motion.  Unusable frames keep the state of the frame before."""
    n = len(motion_pct)
    below = np.zeros(n, bool)
    ok = np.asarray(usable, bool) & np.isfinite(motion_pct)
    below[ok] = motion_pct[ok] < float(threshold_pct)
    if ok.any() and not ok.all():
        idx = np.where(ok, np.arange(n), -1)
        np.maximum.accumulate(idx, out=idx)
        idx[idx < 0] = int(np.argmax(ok))
        below = below[idx]
    blip = int(round(float(ignore_blips_s or 0) * fps))
    if blip > 0:
        for a, b in _runs(~below):
            if a > 0 and b < n and b - a <= blip:
                below[a:b] = True
    min_f = max(1, int(round(float(min_freeze_s) * fps)))
    frz = np.zeros(n, bool)
    for a, b in _runs(below):
        if b - a >= min_f:
            frz[a:b] = True
    return frz


def bin_edges(t0, t1, bin_s):
    if not bin_s or bin_s <= 0 or t1 <= t0:
        return []
    edges = list(np.arange(t0, t1, float(bin_s))) + [t1]
    if len(edges) > 2 and (edges[-1] - edges[-2]) < max(0.5, 0.02 * bin_s):
        edges.pop(-2)                                  # a tiny last piece joins the bin before it
    return [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]


def _period(frz, mot, times, fps, a_t, b_t, keep=None):
    sel = (times >= a_t - 1e-9) & (times < b_t - 1e-9)
    if keep is not None:
        sel &= keep
    if not sel.any():
        return None
    f = frz[sel]
    m = mot[sel]
    runs = _runs(f)
    dur = [(b - a) / fps for a, b in runs]
    return {"Freezing_pct": 100.0 * f.mean(), "Freezing_time_s": f.sum() / fps, "Freezing_bouts": len(runs),
            "Mean_bout_s": float(np.mean(dur)) if dur else 0.0,
            "Mean_motion_pct": float(np.nanmean(m)) if np.isfinite(m).any() else float("nan"),
            "Duration_s": sel.sum() / fps}


def merged_settings(settings):
    s = dict(DEFAULTS)
    for k, v in (settings or {}).items():
        if v is not None or k == "end_time":
            s[k] = v
    return s


LIGHTS_OFF_DARK, LIGHTS_OFF_BRIGHT = 0.55, 1.8   # x the session's typical brightness = the light was off / on


def lights_trim(md, min_s=1.0):
    """(i0, i1): frames [i0, i1) are analysed.  A run of frames at the START or the END of the time window whose
    region brightness is far from the session's (light switched off at the end, or on late) is left out -- the
    animal cannot be seen there and the dark, still picture would count as freezing.  Older .btf files (no
    brightness): (0, n)."""
    n = len(md.frames)
    b = getattr(md, "bright", None)
    if b is None or len(b) != n or not np.isfinite(b).any():
        return 0, n
    b = np.asarray(b, float)
    med = float(np.nanmedian(b))
    if not med > 1.0:
        return 0, n
    odd = np.isfinite(b) & ((b < LIGHTS_OFF_DARK * med) | (b > LIGHTS_OFF_BRIGHT * med))
    need = max(2, int(round(min_s * md.fps)))
    i0, i1 = 0, n
    k = 0
    while k < n and (odd[k] or not np.isfinite(b[k])):
        k += 1
    if k >= need and odd[:k].sum() >= need:
        i0 = min(n, k + md.lag)                     # + the frames compared across the switch
    k = n
    while k > i0 and (odd[k - 1] or not np.isfinite(b[k - 1])):
        k -= 1
    if n - k >= need and odd[k:].sum() >= need:
        i1 = max(i0, k - md.lag)
    if i1 - i0 < max(2, md.lag + 2):                # (nearly) everything dark: keep it all, say so in the checks
        return 0, n
    return i0, i1


def threshold_px(threshold_pct, body_area_px):
    """The freezing threshold in changed pixels (video scale): threshold % x body area."""
    return float(threshold_pct) / 100.0 * float(body_area_px)


def analyse(md, settings):
    """The freezing result for the current settings (fast: no video reading)."""
    s = merged_settings(settings)
    fps = md.fps
    if s["motion_area"] == "animal" and md.has_disc():
        area = "animal"
    elif s["motion_area"] == "body" and md.has_body():
        area = "body"
    else:
        area = "region"
    cut_info = None
    if s.get("cutoff_auto", True):
        cut_info = md.auto_cutoff_info()
        cutoff, level = cut_info["cutoff"], cut_info["level"]
    else:
        cutoff, level = int(s.get("cutoff") or 10), float("nan")
    body_len = measured = float(s.get("body_length_px") or 0.0)
    mot, rate = md.motion(cutoff, area, measured)
    n = len(mot)
    light = ((rate > LIGHT_FRACTION) & md.valid) if s.get("light_guard", True) else np.zeros(n, bool)
    i0, i1 = lights_trim(md) if (s.get("light_guard", True) and s.get("trim_lights", True)) else (0, n)
    keep = np.zeros(n, bool)
    keep[i0:i1] = True
    usable = md.valid & ~light & keep
    frz = freezing_states(mot, usable, fps, s["threshold_pct"], s["min_freeze_s"], s["ignore_blips_s"])
    frz &= keep
    times = md.frames / fps
    nk = int(keep.sum())
    t0 = float(times[i0])
    t1 = float(times[i1 - 1] + 1.0 / fps)
    runs = _runs(frz)
    bouts = [{"Bout": i + 1, "Start_s": float(times[a]), "End_s": float(times[b - 1] + 1.0 / fps),
              "Duration_s": (b - a) / fps} for i, (a, b) in enumerate(runs)]
    dur = [b["Duration_s"] for b in bouts]
    ibi = [bouts[i + 1]["Start_s"] - bouts[i]["End_s"] for i in range(len(bouts) - 1)]
    dots_area = float(getattr(md, "body_area_dots", 0.0) or 0.0)
    if body_len > 0:
        size_src, body_area = "Measured by you", AREA_PER_LEN2 * body_len ** 2
    elif area == "body" and dots_area > 0:
        size_src, body_area = "Body outline", dots_area
        body_len = math.sqrt(dots_area / AREA_PER_LEN2)
    elif md.body_area_full > 0:
        size_src, body_area = "From BT tracking", md.body_area_full
        body_len = md.body_len_full
    else:
        size_src, body_area = "Not found (2.5 % of the region assumed)", md.body_area_work() / md.geo.s ** 2
        body_len = math.sqrt(body_area / AREA_PER_LEN2)
    parts = None
    if md.has_body():
        pthr = part_threshold(s["threshold_pct"])
        pct = {name: md.motion(cutoff, name, measured)[0] for name in PARTS}
        moved = {name: np.isfinite(v) & (v >= pthr) & keep for name, v in pct.items()}
        dots_ok = np.isfinite(md.kp[:, 0, 0])
        body_still = np.isfinite(pct["body"]) & (pct["body"] < float(s["threshold_pct"]))
        parts = {"pct": pct, "moved": moved, "dots": dots_ok,
                 "tail_only": body_still & moved["tail"] & dots_ok & keep}
    outside = float("nan")
    if area == "animal":
        # freezing frames in which the whole region still shows movement: a swaying cable / a hand (fine) or the
        # circle is not on the animal (then use the whole region) -- the review video shows which
        mot_r, _ = md.motion(cutoff, "region", body_len)
        busy = np.isfinite(mot_r) & (mot_r >= float(s["threshold_pct"]))
        outside = 100.0 * float((frz & busy).sum()) / max(1, int(frz.sum()))
    mk = mot[keep]
    summ = {
        "Analysed_time_s": nk / fps, "Start_time_s": t0, "End_time_s": t1,
        "Freezing_pct": 100.0 * frz.sum() / nk if nk else 0.0,
        "Freezing_time_s": frz.sum() / fps, "Active_time_s": (nk - frz.sum()) / fps,
        "Freezing_bouts": len(bouts), "Mean_bout_s": float(np.mean(dur)) if dur else 0.0,
        "Median_bout_s": float(np.median(dur)) if dur else 0.0, "Longest_bout_s": max(dur) if dur else 0.0,
        "Latency_first_freeze_s": (bouts[0]["Start_s"] - t0) if bouts else float("nan"),
        "Bout_rate_per_min": len(bouts) / (nk / fps / 60.0) if nk else 0.0,
        "Mean_interbout_s": float(np.mean(ibi)) if ibi else float("nan"),
        "Median_interbout_s": float(np.median(ibi)) if ibi else float("nan"),
        "Mean_motion_pct": float(np.nanmean(mk)) if np.isfinite(mk).any() else float("nan"),
        "Animal_found_pct": md.found_pct,
        "Movement_outside_pct": outside,
        "Light_change_s": float((light & keep).sum()) / fps,
        "Lights_off_trimmed_s": float(n - nk) / fps,
        "Motion_colour": "RGB" if getattr(md, "colour", "gray") == "rgb" else "Gray",
        "Motion_counted": dict(MOTION_AREAS)[area],
        "Motion_cutoff": int(cutoff),
        "Motion_cutoff_mode": ("Set by you" if cut_info is None else
                               "Auto (from this video; noise spikes limited)" if cut_info["spiky"] else
                               "Auto (from this video)"),
        "Noise_level": level,
        "Freezing_threshold_pct": float(s["threshold_pct"]),
        "Freezing_threshold_px": threshold_px(s["threshold_pct"], body_area),
        "Min_freeze_s": float(s["min_freeze_s"]),
        "Ignore_blips_s": float(s["ignore_blips_s"] or 0),
        "Light_guard": "On" if s.get("light_guard", True) else "Off",
        "FPS": fps, "Frame_lag": md.lag,
        "Body_size_from": size_src, "Body_area_px": float(body_area), "Body_length_px": float(body_len),
        "Region": region_text(md.geo.spec if md.geo.spec is not None else md.geo.region),
        "Animal_colour": {"dark": "Darker than floor", "light": "Lighter than floor", "any": "Either"}.get(
            md.polarity, "—") if md.tracked else "— (not tracked)",
    }
    if parts is not None:
        summ["Body_dots_found_pct"] = 100.0 * float(parts["dots"][keep].mean()) if nk else 0.0
        summ["Tail_only_movement_s"] = float(parts["tail_only"].sum()) / fps
        for name in ("head", "legs", "tail"):
            v = parts["pct"][name][keep]
            summ[f"Mean_{name}_motion_pct"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            summ[f"{name.capitalize()}_moving_s"] = float(parts["moved"][name].sum()) / fps
    epochs = []
    for e in s.get("epochs") or []:
        try:
            a_t, b_t = float(e["start"]), float(e["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b_t <= a_t:
            continue
        r = _period(frz, mot, times, fps, a_t, b_t, keep)
        row = {"Epoch": str(e.get("name") or f"{a_t:g}-{b_t:g} s"), "Start_s": a_t, "End_s": b_t}
        if r is None:
            row.update({"Freezing_pct": float("nan"), "Freezing_time_s": float("nan"), "Freezing_bouts": 0,
                        "Mean_bout_s": float("nan"), "Mean_motion_pct": float("nan"), "Duration_s": 0.0})
        else:
            row.update(r)
        _pre_epoch(row, frz, mot, times, fps, t0, float(s.get("pre_s") or 0), keep)
        epochs.append(row)
    groups = epoch_groups(epochs)
    bins = []
    for i, (a_t, b_t) in enumerate(bin_edges(t0, t1, float(s.get("bin_s") or 0)), start=1):
        r = _period(frz, mot, times, fps, a_t, b_t, keep)
        if r is not None:
            bins.append({"Bin": i, "Start_s": a_t, "End_s": b_t, **r})
    for g in groups:
        summ[f"Group_{g['Group']}_mean_freezing_pct"] = g["Mean_freezing_pct"]
    res = {"summary": summ, "bouts": bouts, "epochs": epochs, "groups": groups, "bins": bins, "freezing": frz,
           "motion": mot, "light": light, "usable": usable, "times": times, "cutoff": int(cutoff), "area": area,
           "settings": s, "parts": parts, "keep": keep, "cutoff_info": cut_info}
    checks = quality_checks(md, res)
    res["checks"] = checks
    bad = [c["text"] for c in checks if c["level"] in ("error", "warn")]
    summ["Result_check"] = "OK" if not bad else "CHECK: " + "; ".join(bad)
    return res


def add_check(res, level, code, text):
    """Add a check to a result (and its summary's Result_check)."""
    res.setdefault("checks", []).append({"level": level, "code": code, "text": text})
    bad = [c["text"] for c in res["checks"] if c["level"] in ("error", "warn")]
    res["summary"]["Result_check"] = "OK" if not bad else "CHECK: " + "; ".join(bad)


def _small_gray(path, t=1.0, width=320):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, f = cap.read()
    finally:
        cap.release()
    if not ok:
        return None, 1.0
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f
    k = width / float(g.shape[1])
    g = cv2.resize(g, (width, max(8, int(round(g.shape[0] * k)))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5), k


def camera_shift(path_ref, path, t=1.0):
    """How far the picture of `path` is shifted against `path_ref` (video px of `path`), or None when the two
    cannot be compared (other size / unreadable).  Used by Batch when a video borrows the first video's region:
    a nudged camera or box makes that region wrong."""
    a, ka = _small_gray(path_ref, t)
    b, kb = _small_gray(path, t)
    if a is None or b is None or a.shape != b.shape:
        return None
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return math.hypot(dx, dy) / kb, float(resp)


def quality_checks(md, res):
    """What could make this result wrong, worst first: [{level: error|warn|info, code, text}].  'error' = do
    not use the number before fixing it."""
    s = res["summary"]
    out = []
    found = md.found_pct if md.tracked else float("nan")
    if md.tracked and math.isfinite(found):
        if found < REST_RETRY_PCT:
            out.append({"level": "error", "code": "found",
                        "text": f"Animal found in only {found:.0f} % of frames (tracking may follow reflections) "
                                f"— 🎯 test, mask reflections, check Animal colour / Min contrast"})
        elif found < 90:
            out.append({"level": "warn", "code": "found",
                        "text": f"Animal found in {found:.0f} % of frames — 🎯 test, mask reflections"})
    ci = res.get("cutoff_info")
    if ci is not None and ci.get("at_limit"):
        out.append({"level": "error", "code": "cutoff",
                    "text": f"Auto cutoff hit its maximum ({CUTOFF_MAX}): the noise is too strong for Auto, so "
                            f"real movement may be missed and freezing comes out too high — mask flickering "
                            f"parts, use Gray, or set the cutoff yourself"})
    if getattr(md, "h_reg", None) is not None and md.valid.any():
        still = md.h_reg[:, 1:].sum(axis=1) == 0              # not one pixel changed at all
        frac = float(still[md.valid].mean())
        if frac > 0.9:
            out.append({"level": "error", "code": "static",
                        "text": f"The picture does not change at all in {100 * frac:.0f} % of the frames (a frozen "
                                f"camera or a still image?) — freezing cannot be measured"})
        elif frac > 0.3:
            out.append({"level": "warn", "code": "static",
                        "text": f"{100 * frac:.0f} % of the frames are exact copies of the frame before (the camera "
                                f"recorded fewer frames per second than the file says): movement is under-counted "
                                f"— check the camera's frame rate"})
    mot, use = np.asarray(res.get("motion"), float), np.asarray(res.get("usable"), bool)
    ok = use & np.isfinite(mot)
    if ok.sum() >= 30:
        floor = float(np.percentile(mot[ok], 5))
        if floor >= float(s["Freezing_threshold_pct"]):
            out.append({"level": "warn", "code": "noise",
                        "text": f"The picture is never still (even its quietest moments show {floor:.0f} % motion, "
                                f"above the {s['Freezing_threshold_pct']:g} % threshold): either the animal never "
                                f"stopped, or camera noise / flicker counts as movement — check the moving pixels; "
                                f"mask flickering parts, use 'Around the animal' or raise the cutoff"})
    if s["Body_size_from"].startswith("Not found"):
        out.append({"level": "error", "code": "size", "text": "Animal not found: the body size is a guess — "
                                                              "📏 measure the animal"})
    if md.tracked and md.body_len_full > 0 and s["Body_size_from"] == "Measured by you":
        r = s["Body_length_px"] / md.body_len_full
        if not 0.6 <= r <= 1.6:
            out.append({"level": "warn", "code": "size",
                        "text": f"Your measured length ({s['Body_length_px']:.0f} px) and the tracked size "
                                f"({md.body_len_full:.0f} px) differ a lot — check the 📏 measurement"})
    cr, cf = getattr(md, "core_ratio", float("nan")), getattr(md, "core_frac", float("nan"))
    if (md.tracked and s["Body_size_from"] != "Measured by you" and not s["Body_size_from"].startswith("Not found")
            and cr is not None and cf is not None and math.isfinite(cr) and math.isfinite(cf)
            and cr < CORE_RATIO_MAX and cf < CORE_FRAC_MAX):
        out.append({"level": "warn", "code": "bodysize",
                    "text": f"The tracked body looks bigger than the animal: its solid part is about "
                            f"{cr * md.body_len_full:.0f} px long, the tracked body {md.body_len_full:.0f} px (a "
                            f"reflection in a clear wall or a shadow is counted with it) — 📏 measure the animal so "
                            f"the threshold uses its real size"})
    if getattr(md, "polarity_warning", None):
        look = {"dark": "darker", "light": "lighter"}.get(md.polarity_warning, md.polarity_warning)
        out.append({"level": "warn", "code": "colour",
                    "text": f"The animal looks {look} than the floor — set Animal colour"})
    if res["area"] == "region" and md.has_disc() and ok.sum() >= 30:
        # Classic counts every movement in the region: when the animal itself is still in many of the frames
        # Classic calls 'moving', something else moves (a cue light, a cable, a hand, the next cage)
        try:
            cut = int(res["cutoff"])
            d_mot = md.motion(cut, "animal", float(res["settings"].get("body_length_px") or 0.0))[0]
            thr = float(s["Freezing_threshold_pct"])
            moving = ok & ~np.asarray(res["freezing"], bool) & (mot >= thr)
            st = res["settings"]
            keep = np.asarray(res["keep"], bool)
            f_an = freezing_states(d_mot, use, md.fps, thr, st["min_freeze_s"], st["ignore_blips_s"]) & keep
            pct_an = 100.0 * float(f_an.sum()) / max(1, int(keep.sum()))
            frac = 0.0
            if moving.sum() >= max(30, 0.05 * ok.sum()):
                frac = float((moving & np.isfinite(d_mot) & (d_mot < 0.5 * thr)).sum()) / float(moving.sum())
            if frac > 0.2 or pct_an - s["Freezing_pct"] > 15:
                out.append({"level": "warn", "code": "elsewhere",
                            "text": f"Movement away from the animal is counted: around the animal it is {pct_an:.0f} % "
                                    f"freezing, in the whole region {s['Freezing_pct']:.0f} % — something else in the "
                                    f"region moves (a cue light, a cable, a hand?): mask it, or use 'Around the "
                                    f"animal'"})
        except Exception:
            pass
    if res["area"] == "animal" and math.isfinite(s.get("Movement_outside_pct") or float("nan")) \
            and s["Movement_outside_pct"] > 25:
        out.append({"level": "warn", "code": "outside",
                    "text": f"{s['Movement_outside_pct']:.0f} % of freezing frames move elsewhere — check that "
                            f"the circle follows the animal"})
    if s["Analysed_time_s"] > 0 and s["Light_change_s"] > 0.1 * s["Analysed_time_s"]:
        out.append({"level": "warn", "code": "light",
                    "text": f"Light changes in {s['Light_change_s']:.0f} s of the video — flicker or a moving "
                            f"lamp; mask it"})
    elif s["Light_change_s"] >= 0.1:
        out.append({"level": "info", "code": "light", "text": f"Light changes: {s['Light_change_s']:.1f} s "
                                                              f"skipped"})
    if s.get("Lights_off_trimmed_s", 0) > 0:
        out.append({"level": "info", "code": "trim",
                    "text": f"Light off / on at the start or end: {s['Lights_off_trimmed_s']:.1f} s not analysed"})
    return out


def _pre_epoch(row, frz, mot, times, fps, t0, pre_s, keep=None):
    """The `pre_s` seconds before an epoch (pre-tone baseline): freezing there, the change and the suppression
    ratio (epoch motion / (epoch + before motion); 0.5 = no change, < 0.5 = less movement in the epoch)."""
    nan = float("nan")
    row.update({"Pre_freezing_pct": nan, "Change_vs_pre_pct": nan, "Suppression_ratio": nan})
    a = float(row["Start_s"])
    if pre_s <= 0 or a - pre_s < t0 - 1e-9 or not np.isfinite(row.get("Freezing_pct", nan)):
        return
    pre = _period(frz, mot, times, fps, a - pre_s, a, keep)
    if pre is None:
        return
    row["Pre_freezing_pct"] = pre["Freezing_pct"]
    row["Change_vs_pre_pct"] = row["Freezing_pct"] - pre["Freezing_pct"]
    m_e, m_p = row.get("Mean_motion_pct", nan), pre["Mean_motion_pct"]
    if np.isfinite(m_e) and np.isfinite(m_p) and m_e + m_p > 0:
        row["Suppression_ratio"] = m_e / (m_e + m_p)


def group_name(name):
    """'Tone 3' -> 'Tone', 'ITI_2' -> 'ITI', 'CS+ 1' -> 'CS+', 'Baseline' -> 'Baseline'."""
    import re
    g = re.sub(r"[\s_\-#]*\d+\s*$", "", str(name)).strip()
    return g or str(name)


def epoch_groups(epochs):
    """Epochs of the same kind (Tone 1, Tone 2 ... -> Tone): trial-by-trial summary per kind."""
    order, by = [], {}
    for e in epochs:
        g = group_name(e["Epoch"])
        if g not in by:
            order.append(g)
            by[g] = []
        by[g].append(e)
    out = []
    for g in order:
        rows = [e for e in by[g] if np.isfinite(e.get("Freezing_pct", float("nan")))]
        v = np.array([e["Freezing_pct"] for e in rows], float)
        ch = np.array([e.get("Change_vs_pre_pct", float("nan")) for e in rows], float)
        sup = np.array([e.get("Suppression_ratio", float("nan")) for e in rows], float)
        n = len(v)
        out.append({
            "Group": g, "Trials": n,
            "Mean_freezing_pct": float(v.mean()) if n else float("nan"),
            "SD_freezing_pct": float(v.std(ddof=1)) if n > 1 else float("nan"),
            "First_pct": float(v[0]) if n else float("nan"), "Last_pct": float(v[-1]) if n else float("nan"),
            "Change_first_last_pct": float(v[-1] - v[0]) if n > 1 else float("nan"),
            "Slope_per_trial_pct": float(np.polyfit(np.arange(n), v, 1)[0]) if n > 2 else
            (float(v[-1] - v[0]) if n == 2 else float("nan")),
            "Mean_change_vs_pre_pct": float(np.nanmean(ch)) if np.isfinite(ch).any() else float("nan"),
            "Mean_suppression_ratio": float(np.nanmean(sup)) if np.isfinite(sup).any() else float("nan"),
            "Trial_values": [float(x) for x in v],
        })
    return out


# ============================================================================= review picture
def review_status(result, i):
    """("FREEZING" / "ACTIVE" [+ light change], motion text, freezing?) for frame i of the result."""
    if result is None or not 0 <= i < len(result["freezing"]):
        return None
    frz = bool(result["freezing"][i])
    keep = result.get("keep")
    if keep is not None and i < len(keep) and not keep[i]:
        return "NOT ANALYSED", "light off / on at the start or end of the video", False
    txt = "FREEZING" if frz else "ACTIVE"
    if result["light"][i]:
        txt += " (light change)"
    mv = result["motion"][i]
    small = "motion --" if not np.isfinite(mv) else f"motion {mv:.1f} % of body"
    parts = result.get("parts")
    if parts is not None and i < len(parts["dots"]) and parts["dots"][i]:
        moved = [n for n in ("head", "legs", "tail") if parts["moved"][n][i]]
        if moved == ["tail"] and not parts["moved"]["body"][i]:
            small += "  ·  tail moved (body still)"
        elif moved:
            small += "  ·  moved: " + ", ".join(moved)
    return txt, small, frz


def review_body(md, i, result):
    """The body dots of frame i for a picture viewer (body_overlay), or None."""
    kp = getattr(md, "kp", None)
    if kp is None or not 0 <= i < len(kp):
        return None
    moved = None
    if result and result.get("parts") is not None:
        moved = {n: bool(result["parts"]["moved"][n][i]) for n in PARTS}
    return body_overlay(md.geo, kp[i], moved, body_length_shown(md))


def body_length_shown(md):
    """Body length (video px) for drawing the dots: the body outline's when known, else tracking's."""
    return float(getattr(md, "body_len_dots", 0.0) or 0.0) or float(getattr(md, "body_len_full", 0.0) or 0.0)


def review_image(md, frame_now, frame_prev, i, result, overlay=True, text=True, dots=True):
    """The frame with the changed pixels in red, the region box, the circle around the animal (when used), the body
    dots (dots=False: a viewer draws them, review_body) and FREEZING / ACTIVE."""
    disp = frame_now.copy()
    geo = md.geo
    cutoff = result["cutoff"] if result else 10
    colour = getattr(md, "colour", "gray") or "gray"
    if overlay and frame_prev is not None:
        m = change_mask(geo.prep(frame_prev, colour), geo.prep(frame_now, colour), cutoff,
                        geo.inside).astype(np.uint8)
        if geo.s < 1.0:
            m = cv2.resize(m, (geo.x1 - geo.x0, geo.y1 - geo.y0), interpolation=cv2.INTER_NEAREST)
        sub = disp[geo.y0:geo.y1, geo.x0:geo.x1]
        mm = m.astype(bool)
        sub[mm] = (0.35 * sub[mm] + np.array([0.65 * 40, 0.65 * 40, 0.65 * 235])).astype(np.uint8)
    lw = max(1, int(round(max(disp.shape[:2]) / 900)))
    draw_region(disp, geo=geo)
    if md.xs is not None and 0 <= i < len(md.xs) and result and result["area"] == "animal":
        r = md.radius_work / geo.s
        col = (255, 200, 0) if md.seen[i] else (160, 160, 160)
        cv2.circle(disp, (int(round(md.xs[i])), int(round(md.ys[i]))), max(2, int(round(r))), col, lw,
                   cv2.LINE_AA)
    kp = getattr(md, "kp", None)
    if dots and kp is not None and 0 <= i < len(kp):
        moved = None
        if result and result.get("parts") is not None:
            moved = {n: bool(result["parts"]["moved"][n][i]) for n in PARTS}
        draw_body(disp, geo, kp[i], moved, body_length_shown(md))
    st = review_status(result, i) if text else None
    if st is not None:
        txt, txt2, frz = st
        fs = max(0.45, disp.shape[0] / 600.0)
        th = max(1, int(round(fs * 2)))
        col = (255, 170, 40) if frz else (60, 200, 60)
        (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        (tw2, _), _ = cv2.getTextSize(txt2, cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, max(1, th // 2))
        cv2.rectangle(disp, (6, 6), (16 + max(tw, tw2), 16 + tht + int(22 * fs)), (0, 0, 0), -1)
        cv2.putText(disp, txt, (11, 11 + tht), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
        cv2.putText(disp, txt2, (11, 11 + tht + int(20 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6,
                    (230, 230, 230), max(1, th // 2), cv2.LINE_AA)
    return disp


def body_size_frame(md):
    """Index (into md.frames) of a frame to show the body size on: the middle of the frames where the animal was
    seen (else the middle of the window)."""
    n = len(md.frames)
    seen = np.flatnonzero(np.asarray(getattr(md, "seen", np.zeros(n, bool)), bool)[:n])
    return int(seen[len(seen) // 2]) if seen.size else n // 2


def _heading(md, i, k=6):
    """Direction of the animal at frame i (radians): nose -> tail base of the body dots, else the path."""
    kp = getattr(md, "kp", None)
    if kp is not None and 0 <= i < len(kp) and np.isfinite(kp[i, 0, 0]) and np.isfinite(kp[i, 6, 0]):
        return math.atan2(kp[i, 0, 1] - kp[i, 6, 1], kp[i, 0, 0] - kp[i, 6, 0])
    if md.xs is not None:
        a, b = max(0, i - k), min(len(md.xs) - 1, i + k)
        dx, dy = md.xs[b] - md.xs[a], md.ys[b] - md.ys[a]
        if math.hypot(dx, dy) > 2:
            return math.atan2(dy, dx)
    return None


def body_size_image(md, result, frame_bgr, i):
    """The body size the freezing threshold is based on, drawn on frame i for checking by eye: an outline of that
    size on the animal (length = nose to tail base) and a bar of one body length."""
    disp = frame_bgr.copy()
    s = result["summary"]
    L = float(s.get("Body_length_px") or 0.0)
    draw_region(disp, geo=md.geo)
    h, w = disp.shape[:2]
    lw = max(2, int(round(max(h, w) / 500)))
    fs = max(0.5, h / 900.0)
    if L > 0:
        if md.xs is not None and 0 <= i < len(md.xs) and np.isfinite(md.xs[i]):
            c = (int(round(md.xs[i])), int(round(md.ys[i])))
            ang = _heading(md, i)
            ax = (max(2, int(round(L / 2))), max(2, int(round(L / 2 / 2.4 * 1.27))))  # area = L^2 / 2.4
            if ang is None:
                r = max(2, int(round(math.sqrt(L * L / 2.4 / math.pi))))
                cv2.circle(disp, c, r, (20, 20, 20), lw + 2, cv2.LINE_AA)
                cv2.circle(disp, c, r, (0, 215, 255), lw, cv2.LINE_AA)
            else:
                cv2.ellipse(disp, c, ax, math.degrees(ang), 0, 360, (20, 20, 20), lw + 2, cv2.LINE_AA)
                cv2.ellipse(disp, c, ax, math.degrees(ang), 0, 360, (0, 215, 255), lw, cv2.LINE_AA)
        geo = md.geo
        x0, y1 = geo.x0 + 12, geo.y1 - 14
        x1 = int(round(x0 + L))
        cv2.line(disp, (x0, y1), (x1, y1), (20, 20, 20), lw + 4, cv2.LINE_AA)
        cv2.line(disp, (x0, y1), (x1, y1), (0, 215, 255), lw + 1, cv2.LINE_AA)
        for x in (x0, x1):
            cv2.line(disp, (x, y1 - 8), (x, y1 + 8), (0, 215, 255), lw, cv2.LINE_AA)
    lines = [f"Body size used: {L:.0f} px long ({s.get('Body_size_from', '')})",
             f"Freezing threshold: {s.get('Freezing_threshold_pct', 0):g} % of the body = "
             f"{s.get('Freezing_threshold_px', 0):.0f} changed px",
             "Yellow outline = that size. It should match the mouse (nose to tail base)."]
    th = max(1, int(round(fs * 1.6)))
    y = 10
    for k, t in enumerate(lines):
        f = fs * (0.75 if k else 0.85)
        (tw, tht), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, f, th)
        cv2.rectangle(disp, (6, y - 4), (16 + tw, y + tht + 8), (0, 0, 0), -1)
        cv2.putText(disp, t, (11, y + tht + 2), cv2.FONT_HERSHEY_SIMPLEX, f,
                    (0, 215, 255) if k == 0 else (235, 235, 235), th, cv2.LINE_AA)
        y += tht + 14
    return disp


def default_output_dir(video_path):
    from tracking.location import compute_output_dir
    d = os.path.join(compute_output_dir(str(video_path)), "freezing")
    os.makedirs(d, exist_ok=True)
    return d
