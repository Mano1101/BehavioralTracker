"""Live tracking from a camera.

The SAME detector, zones, masks, crop and detection settings as video-file tracking
(the `setup` dict built by the UI for a video of the camera's size), applied to each
camera frame as it arrives.  Used for the live view and the running numbers during a
trial; the official results of a trial are computed afterwards from the recorded
original video by the normal pipeline (so live and recorded trials are comparable).

States: idle -> learning (background) -> ready (tracking shown, not counted)
        -> running (trial: numbers counted) -> done.
"""
import math
import threading

import cv2
import numpy as np

from tracking.location import (ROI_COLORS, _LiveDrawer, _is_identity_warp, build_detection_exclusion,
                               build_roi_masks, draw_mask_polygons, draw_status_box, new_tracker, roi_membership,
                               covered_zone_names)
from tracking.robust_tracker import build_background, detection_scale, prepare_raw
from tracking.fragment import outline_arrays

TRAIL_POINTS = 120          # last positions drawn as a trail on the live view


class LiveSession:
    EMPTY_SAMPLES, EMPTY_EVERY = 30, 0.07      # empty arena: ~2 s
    LEARN_SAMPLES, LEARN_EVERY = 40, 0.25      # animal present (must move): ~10 s

    def __init__(self, setup, frame_w, frame_h, fps_hint=30.0):
        self.setup = dict(setup)
        self.setup["_fps"] = float(fps_hint or 30.0)
        m = self.setup.get("matrix")
        self.matrix = np.eye(3, dtype=np.float32) if m is None else np.asarray(m, np.float32)
        self.w = int(self.setup.get("warp_w") or frame_w)
        self.h = int(self.setup.get("warp_h") or frame_h)
        self.need_warp = not _is_identity_warp(self.matrix, frame_w, frame_h, self.w, self.h)
        rp = {k: [tuple(map(float, p)) for p in v] for k, v in (self.setup.get("roi_points") or {}).items()
              if len(v) >= 3}
        self.roi_names = [n for n in (self.setup.get("roi_names") or list(rp)) if n in rp]
        self.roi_names += [n for n in rp if n not in self.roi_names]
        self.roi_points = {n: rp[n] for n in self.roi_names}
        self.masks = build_roi_masks((self.h, self.w), self.roi_points) if self.roi_points else {}
        mask_polygons = self.setup.get("mask_points", []) or []
        auto = bool(self.setup.get("auto_mask_outside_zones")) and bool(self.roi_points)
        self.exclusion = None
        covered = covered_zone_names(self.setup)
        if mask_polygons or auto or covered:
            ex = build_detection_exclusion(self.w, self.h, mask_polygons, self.roi_points, auto,
                                           margin=self.setup.get("auto_mask_margin"), covered=covered)
            self.exclusion = None if ex.all() else ex
        algo = str(self.setup.get("tracking_algorithm", "robust")).lower()
        self.color_mode = str(self.setup.get("color_mode", "gray")).lower()
        self.polarity = "any" if algo == "classic" else str(self.setup.get("animal_polarity", "auto")).lower()
        self.scale = detection_scale(self.w, self.h)
        self.cm_per_px = float(self.setup.get("scale_factor") or 0.0)
        self.drawer = _LiveDrawer(self.w, self.h, self.roi_points, mask_polygons, self.exclusion)
        self.state = "idle"
        self.samples, self.bgm, self.tracker = [], None, None
        self.learn_target, self.learn_every, self._last_sample = 0, 0.0, -1e9
        self.n_frames = 0
        self._last_feed = None
        self._rate = []                     # times of the frames tracked in the last 2 s
        self.empty_arena = True
        self._reset_stats()

    # ------------------------------------------------------------------ background
    def start_learning(self, empty_arena, now):
        """empty_arena=True: the arena is empty now (~2 s).  False: the animal is in and moving (~10 s)."""
        self.samples, self.bgm, self.tracker = [], None, None
        self.learn_target = self.EMPTY_SAMPLES if empty_arena else self.LEARN_SAMPLES
        self.learn_every = self.EMPTY_EVERY if empty_arena else self.LEARN_EVERY
        self._last_sample = -1e9
        self.empty_arena = bool(empty_arena)
        self.state = "learning"

    def _build(self):
        valid = None if self.exclusion is None else ~self.exclusion
        from tracking.location import low_light_zone_polys
        self.bgm = build_background(None, range(len(self.samples)), self.color_mode, self.polarity,
                                    valid_full=valid, scale=self.scale, raw_samples=self.samples,
                                    low_light=self.setup.get("low_light", "off"),
                                    zone_polys=low_light_zone_polys(self.setup))
        self.tracker = new_tracker(self.setup, self.bgm, self.exclusion)
        self.samples = []
        self._last_feed = None
        self.state = "ready"

    @property
    def learn_progress(self):
        return (len(self.samples) / float(self.learn_target)) if self.learn_target else 0.0

    # ------------------------------------------------------------------ trial
    def _reset_stats(self):
        self.t0 = None
        self.elapsed = 0.0
        self._last_now = None
        self.zone_time = {n: 0.0 for n in self.roi_names}
        self.zone_entries = {n: 0 for n in self.roi_names}
        self._inside = {n: False for n in self.roi_names}
        self.current_zones = []
        self.distance_px = 0.0
        self._last_xy = None
        self.frames_counted = 0
        self.frames_detected = 0
        self.trail = []

    def start_trial(self, now):
        self._reset_stats()
        self.t0 = now
        self._last_now = now
        self.state = "running"

    def stop_trial(self):
        if self.state == "running":
            self.state = "done"

    # ------------------------------------------------------------------ per frame
    def warp(self, frame):
        return cv2.warpPerspective(frame, self.matrix, (self.w, self.h)) if self.need_warp else frame

    def feed(self, frame_bgr, now):
        """Process one camera frame -> preview image (BGR, <= 960 px wide)."""
        fr = self.warp(frame_bgr)
        self.n_frames += 1
        self._rate.append(now)
        while self._rate and now - self._rate[0] > 2.0:
            del self._rate[0]
        if self.state == "learning":
            if now - self._last_sample >= self.learn_every:
                self._last_sample = now
                self.samples.append(prepare_raw(fr, self.color_mode, self.scale))
                if len(self.samples) >= self.learn_target:
                    self._build()
            disp = self._plain(self._bright(fr))
            what = "empty arena" if self.empty_arena else "keep the animal moving"
            return draw_status_box(disp, f"LEARNING BACKGROUND ({what})  {int(self.learn_progress * 100)} %",
                                   (0, 165, 255))
        if self.tracker is None:
            return draw_status_box(self._plain(self._bright(fr)), "LIVE  -  no background yet", (150, 150, 150))
        # frames the camera delivered while the previous one was being tracked (slow PC): widen the
        # movement gate for this step so the animal's longer jump between tracked frames is accepted
        k = 1.0
        if self._last_feed is not None:
            k = min(4.0, max(1.0, (now - self._last_feed) * self.setup["_fps"]))
        self._last_feed = now
        gate0 = getattr(self.tracker, "gate", None)
        if gate0 is not None and k > 1.3:
            self.tracker.gate = gate0 * k
        try:
            prep_fn = getattr(self.tracker, "prepare_frame", None)
            det = self.tracker.update(fr, prep_fn(fr)) if prep_fn is not None else self.tracker.update(fr)
        finally:
            if gate0 is not None:
                self.tracker.gate = gate0
        t = (now - self.t0) if (self.state == "running" and self.t0 is not None) else \
            (self.elapsed if self.state == "done" else 0.0)
        if self.state == "running":
            self._count(det, now)
        disp = self.drawer.draw(fr, det, self.tracker, t, self.n_frames)      # (boost applied inside)
        if self.state == "running" and len(self.trail) >= 2:
            s = self.drawer.s
            pts = np.array([(x * s, y * s) for x, y in self.trail], np.int32)
            cv2.polylines(disp, [pts], False, (255, 120, 0), 2, cv2.LINE_AA)
        return disp

    def _bright(self, fr):
        """The camera picture as shown on screen: with the low-light boost when it is on."""
        from tracking.robust_tracker import brighten, resolve_gamma
        g = getattr(self.bgm, "gamma", None) if self.bgm is not None else None
        if g is None:
            setting = str(self.setup.get("low_light", "off")).lower()
            g = 1.0 if setting == "auto" else resolve_gamma(setting)
        return brighten(fr, g)

    def _plain(self, fr):
        """Zones (and the dimmed masked area) on the frame, without a detection."""
        d = self.drawer
        disp = cv2.resize(fr, d.size, interpolation=cv2.INTER_AREA) if d.s < 1.0 else fr.copy()
        if d.excl is not None:
            disp[d.excl] = (disp[d.excl] * 0.35).astype(np.uint8)
        for i, pts in enumerate(d.roi.values()):
            cv2.polylines(disp, outline_arrays(pts), True, ROI_COLORS[i % len(ROI_COLORS)],
                          max(1, int(round(max(disp.shape[:2]) / 1100.0))), cv2.LINE_AA)
        if d.masks:
            draw_mask_polygons(disp, d.masks)
        return disp

    def _count(self, det, now):
        dt = max(0.0, now - (self._last_now if self._last_now is not None else now))
        self._last_now = now
        self.elapsed = now - self.t0
        self.frames_counted += 1
        tracked = det.get("status") == "Tracked" and np.isfinite(det.get("x", np.nan))
        if not tracked:
            for n in self.current_zones:                 # a short loss keeps the last zone (like the analysis)
                self.zone_time[n] += dt
            return
        self.frames_detected += 1
        x, y = float(det["x"]), float(det["y"])
        member = roi_membership(self.masks, x, y) if self.masks else {}
        now_in = [n for n in self.roi_names if member.get(n)]
        for n in self.roi_names:
            if member.get(n) and not self._inside[n]:
                self.zone_entries[n] += 1
            self._inside[n] = bool(member.get(n))
        for n in now_in:
            self.zone_time[n] += dt
        self.current_zones = now_in
        if self._last_xy is not None:
            step = math.hypot(x - self._last_xy[0], y - self._last_xy[1])
            if step >= self._min_move():
                self.distance_px += step
                self._last_xy = (x, y)
        else:
            self._last_xy = (x, y)
        self.trail.append((x, y))
        if len(self.trail) > TRAIL_POINTS:
            del self.trail[0]

    def _min_move(self):
        mm = self.setup.get("min_move_px")
        if mm:
            return float(mm)
        body = getattr(self.tracker, "body_l", None)
        s = getattr(self.tracker, "s", None)
        return max(1.0, 0.04 * ((body / s) if (body and s) else 40.0))

    # ------------------------------------------------------------------ numbers for the UI
    def stats(self):
        dist = self.distance_px * self.cm_per_px if self.cm_per_px else self.distance_px
        zones = []
        for n in self.roi_names:
            tz = self.zone_time.get(n, 0.0)
            zones.append({"zone": n, "time_s": tz, "percent": (100.0 * tz / self.elapsed) if self.elapsed > 0 else 0.0,
                          "entries": self.zone_entries.get(n, 0)})
        return {"state": self.state, "elapsed": self.elapsed, "distance": dist,
                "distance_unit": "cm" if self.cm_per_px else "px",
                "detected_pct": (100.0 * self.frames_detected / self.frames_counted) if self.frames_counted else 0.0,
                "current": ", ".join(self.current_zones) or "—", "zones": zones,
                "learn_progress": self.learn_progress, "track_fps": self.track_fps}

    @property
    def track_fps(self):
        r = self._rate
        return (len(r) - 1) / (r[-1] - r[0]) if len(r) >= 3 and r[-1] > r[0] else 0.0


class LiveRunner:
    """Runs a LiveSession on its own thread so the camera keeps reading and recording at full speed:
    the camera thread hands over the NEWEST frame (older ones are skipped when tracking is slower than
    the camera).  Commands carry the camera time of the frame they belong to, so a trial starts on the
    same frame as its recording."""

    def __init__(self):
        self._thread = threading.Thread(target=self._loop, name="BT-live-tracking", daemon=True)
        self._cv = threading.Condition()
        self._item = None
        self._cmds = []
        self._quit = False
        self.session = None
        self.image = None                   # newest annotated preview (BGR)
        self.latest = None                  # newest stats() dict
        self.error = None

    def push(self, frame, now):
        with self._cv:
            self._item = (frame, now)
            self._cv.notify()

    def command(self, cmd, now, arg=None):
        with self._cv:
            self._cmds.append((cmd, now, arg))
            self._cv.notify()

    def start(self):
        self._thread.start()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def stop(self):
        with self._cv:
            self._quit = True
            self._cv.notify()

    def _loop(self):
        while True:
            with self._cv:
                while not self._quit and self._item is None and not self._cmds:
                    self._cv.wait(0.5)
                if self._quit:
                    return
                cmds, self._cmds = self._cmds, []
                item, self._item = self._item, None
            for cmd, now, arg in cmds:
                if cmd == "session":
                    self.session, self.image = arg, None
                elif self.session is None:
                    continue
                elif cmd == "learn":
                    self.session.start_learning(bool(arg), now)
                elif cmd == "start":
                    self.session.start_trial(now)
                elif cmd == "stop":
                    self.session.stop_trial()
            if self.session is not None:
                if item is not None:
                    try:
                        self.image = self.session.feed(item[0], item[1])
                        self.error = None
                    except Exception as exc:          # never stop the camera / recording for a tracking error
                        self.error = str(exc)
                        print(f"Live tracking error: {exc}")
                self.latest = self.session.stats()
