"""Delayed live tracking from a camera.

The trial is recorded as usual (the ORIGINAL video, full size and real-time paced).  At the same time the
same frames are cut into pieces of `piece_s` seconds (default 30 s) in a temporary folder:

    1. frames      the camera thread hands every recorded frame over; a writer thread stores the pieces
    2. detection   as soon as a piece is complete it is tracked (background, animal, body-in-zone) on an
                   analyser thread, while the next piece is being recorded; positions are kept in memory
    3. live view   the tracked frames are shown paced at their recording time + the delay, so the user
                   watches the tracked trial running about `piece_s` seconds behind the camera

When the trial ends only the last piece is left: it is tracked at full speed and the normal analysis
(tracking.location.analyse_track -- the same code as a video file) writes every result file and the
protected .bta file.  So the results are ready a few seconds after the trial, not after a full re-run.

The background is learned from the first piece (plus an optional empty-arena capture before the trial).
"""
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback

import cv2
import numpy as np

from tracking.fragment import outline_arrays
from tracking.location import (ROI_COLORS, BodyZoneFractions, _LiveDrawer, _is_identity_warp,
                               build_detection_exclusion, build_roi_masks, compute_output_dir, draw_mask_polygons,
                               new_tracker, apply_backfill, covered_zone_names)
from tracking.robust_tracker import build_background, detection_scale, prepare_raw

PIECE_S = 30.0


def _setup_geometry(setup, frame_w, frame_h):
    m = setup.get("matrix")
    matrix = np.eye(3, dtype=np.float32) if m is None else np.asarray(m, np.float32)
    w = int(setup.get("warp_w") or frame_w)
    h = int(setup.get("warp_h") or frame_h)
    need_warp = not _is_identity_warp(matrix, frame_w, frame_h, w, h)
    rp = {k: [tuple(map(float, p)) for p in v] for k, v in (setup.get("roi_points") or {}).items() if len(v) >= 3}
    names = [n for n in (setup.get("roi_names") or list(rp)) if n in rp]
    names += [n for n in rp if n not in names]
    roi_points = {n: rp[n] for n in names}
    mask_polygons = setup.get("mask_points", []) or []
    auto = bool(setup.get("auto_mask_outside_zones")) and bool(roi_points)
    exclusion = None
    covered = covered_zone_names(setup)
    if mask_polygons or auto or covered:
        ex = build_detection_exclusion(w, h, mask_polygons, roi_points, auto, margin=setup.get("auto_mask_margin"),
                                       covered=covered)
        exclusion = None if ex.all() else ex
    return matrix, w, h, need_warp, roi_points, names, mask_polygons, exclusion


class ZoneOverlay:
    """Camera preview with the zones drawn on it (straightened like the analysis when a crop is set)."""

    def __init__(self, setup, frame_w, frame_h):
        (self.matrix, self.w, self.h, self.need_warp, roi_points, _names, masks,
         exclusion) = _setup_geometry(setup, frame_w, frame_h)
        self.drawer = _LiveDrawer(self.w, self.h, roi_points, masks, exclusion)
        self.frame_size = (int(frame_w), int(frame_h))

    def draw(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.frame_size:
            return frame
        fr = cv2.warpPerspective(frame, self.matrix, (self.w, self.h)) if self.need_warp else frame
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


def compose_view(tracked, live, lag_s):
    """The tracked (delayed) picture, with the live camera as a small inset in the top-right corner."""
    disp = tracked.copy()
    H, W = disp.shape[:2]
    if live is not None:
        iw = max(80, int(W * 0.27))
        ih = max(1, int(round(live.shape[0] * iw / float(live.shape[1]))))
        if ih < H - 20 and iw < W - 20:
            small = cv2.resize(live, (iw, ih), interpolation=cv2.INTER_AREA)
            x0, y0 = W - iw - 10, 10
            disp[y0:y0 + ih, x0:x0 + iw] = small
            cv2.rectangle(disp, (x0 - 1, y0 - 1), (x0 + iw, y0 + ih), (255, 255, 255), 2)
            fs = max(0.4, iw / 420.0)
            cv2.rectangle(disp, (x0, y0), (x0 + int(70 * fs) + 8, y0 + int(26 * fs)), (0, 0, 200), -1)
            cv2.putText(disp, "LIVE", (x0 + 5, y0 + int(20 * fs)), cv2.FONT_HERSHEY_SIMPLEX, 0.75 * fs,
                        (255, 255, 255), max(1, int(2 * fs)), cv2.LINE_AA)
    txt = f"TRACKED VIEW  -  {max(0.0, lag_s):.0f} s behind the camera"
    fs = max(0.45, W / 1500.0)
    (tw, th), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
    y1 = H - 8
    cv2.rectangle(disp, (8, y1 - th - base - 10), (8 + tw + 16, y1), (20, 20, 20), -1)
    cv2.putText(disp, txt, (16, y1 - base - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return disp


class DelayedLiveTracker:
    """See the module text.  Thread use: add_frame()/finish() from the camera thread; status(), image,
    summary from the UI; everything else runs on the tracker's own two threads."""

    def __init__(self, setup, video_path, frame_w, frame_h, fps, piece_s=PIECE_S, empty_frames=None,
                 work_dir=None, pace=True):
        self.setup = dict(setup)
        self.fps = float(fps or 30.0)
        self.setup["_fps"] = self.fps
        self.video_path = str(video_path)
        self.frame_w, self.frame_h = int(frame_w), int(frame_h)
        self.piece_s = max(2.0, float(piece_s or PIECE_S))
        self.piece_n = max(2, int(round(self.piece_s * self.fps)))
        self.delay_s = self.piece_s
        self.pace = bool(pace)
        (self.matrix, self.w, self.h, self.need_warp, self.roi_points, self.roi_names, self.mask_polygons,
         self.exclusion) = _setup_geometry(self.setup, frame_w, frame_h)
        self.roi_masks = build_roi_masks((self.h, self.w), self.roi_points)
        entry_rule = str(self.setup.get("entry_rule", "paws")).lower()
        self.frac_names = (list(self.roi_names) if entry_rule == "paws" and self.setup.get("test_type") != "nor"
                           and self.roi_names else [])
        self._empty = list(empty_frames or [])
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="bt_live_")
        self._own_dir = work_dir is None
        os.makedirs(self.work_dir, exist_ok=True)
        # hand-over between the threads
        self._wq = queue.Queue(maxsize=64)
        self._pieces = queue.Queue()
        self._finishing = threading.Event()
        self._cancel = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, name="BT-live-pieces", daemon=True)
        self._analyser = threading.Thread(target=self._analyse_loop, name="BT-live-analysis", daemon=True)
        # state (read by the UI)
        self.t0 = None                       # wall-clock time of the first recorded frame
        self.n_recorded = 0
        self.n_processed = 0
        self.phase = "recording"             # recording -> learning -> tracking -> finishing -> analysing -> done
        self.image = None                    # newest tracked picture (BGR, <= 960 px wide)
        self.image_t = 0.0                   # trial time of that picture
        self.numbers = None                  # summary of the part tracked so far (same method as the results)
        self.summary = None                  # final results (written to disk)
        self.error = None
        self.background_note = ""
        self.frames, self.times, self.xs, self.ys, self.areas, self.status_list, self.fracs, self.ends = (
            [] for _ in range(8))
        self.tracker = None
        self.bgm = None
        self._numbers_due = 0.0
        self._numbers_cost = 0.0
        self._writer_ok = True

    # ------------------------------------------------------------------ camera thread side
    def set_fps(self, fps):
        """The frame rate of the recording file (chosen when it is opened, before the first frame)."""
        self.fps = float(fps or self.fps)
        self.setup["_fps"] = self.fps
        self.piece_n = max(2, int(round(self.piece_s * self.fps)))

    def start(self):
        self._writer.start()
        self._analyser.start()

    def add_frame(self, frame, now=None):
        """Every frame written to the recording, in the same order (duplicates included)."""
        if self._cancel.is_set() or not self._writer_ok:
            return
        if self.t0 is None:
            self.t0 = time.time() if now is None else float(now)
        try:
            self._wq.put(frame, timeout=2.0)
            self.n_recorded += 1
        except queue.Full:                    # the disk cannot keep up: tracking would lose its place
            self._fail("The computer could not store the frames for live tracking fast enough. The recording "
                       "is complete -- track it like any video.")

    def finish(self):
        """The recording has ended (the video file is closed)."""
        self._finishing.set()
        try:
            self._wq.put(None, timeout=5.0)
        except queue.Full:
            pass

    def cancel(self):
        self._cancel.set()
        self._finishing.set()
        for q in (self._wq, self._pieces):
            try:
                q.put_nowait(None)
            except queue.Full:
                pass

    def wait(self, timeout=None):
        self._analyser.join(timeout)
        return not self._analyser.is_alive()

    @property
    def recorded_s(self):
        return self.n_recorded / self.fps

    @property
    def processed_s(self):
        return self.n_processed / self.fps

    @property
    def lag_s(self):
        if self.t0 is None:
            return 0.0
        return max(0.0, (time.time() - self.t0) - self.image_t) if not self._finishing.is_set() else \
            max(0.0, self.recorded_s - self.image_t)

    def status(self):
        left = 0.0
        if self.phase == "recording" and self.t0 is not None:
            left = max(0.0, self.piece_s - (time.time() - self.t0))
        return {"phase": self.phase, "recorded_s": self.recorded_s, "processed_s": self.processed_s,
                "lag_s": self.lag_s, "first_in_s": left, "numbers": self.numbers, "error": self.error,
                "summary": self.summary, "background": self.background_note,
                "remaining_s": max(0.0, self.recorded_s - self.processed_s)}

    def _fail(self, msg):
        if self.error is None:
            self.error = msg
        self.phase = "failed"
        self._writer_ok = False
        self._cancel.set()
        try:
            self._pieces.put_nowait(None)
        except queue.Full:
            pass

    # ------------------------------------------------------------------ 1. frames -> pieces
    def _writer_loop(self):
        vw, path, first, n_in, idx, k = None, None, 0, 0, 0, 0
        try:
            while True:
                f = self._wq.get()
                if f is None:
                    break
                if self._cancel.is_set():
                    continue
                if vw is None:
                    path = os.path.join(self.work_dir, f"piece_{k:04d}.avi")
                    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), self.fps,
                                         (f.shape[1], f.shape[0]))
                    if not vw.isOpened():
                        vw = None
                        raise RuntimeError("cannot write the tracking pieces in the temporary folder")
                    first, n_in = idx, 0
                vw.write(f)
                n_in += 1
                idx += 1
                if n_in >= self.piece_n:
                    vw.release()
                    vw = None
                    self._pieces.put((path, first, n_in))
                    k += 1
        except Exception as exc:
            self._fail(f"Live tracking stopped: {exc}. The recording is complete -- track it like any video.")
        finally:
            if vw is not None:
                vw.release()
                if n_in:
                    self._pieces.put((path, first, n_in))
            self._pieces.put(None)

    # ------------------------------------------------------------------ 2. detection  3. paced live view
    def _warp(self, frame):
        return cv2.warpPerspective(frame, self.matrix, (self.w, self.h)) if self.need_warp else frame

    def _build_engine(self, path, n):
        """Background from the first piece (+ the empty-arena frames) -> tracker, like a video file."""
        self.phase = "learning"
        color_mode = str(self.setup.get("color_mode", "gray")).lower()
        algo = str(self.setup.get("tracking_algorithm", "robust")).lower()
        polarity = "any" if algo == "classic" else str(self.setup.get("animal_polarity", "auto")).lower()
        scale = detection_scale(self.w, self.h)
        n_bg = max(10, int(self.setup.get("background_samples", 100)))
        want = set(np.linspace(0, n - 1, min(n_bg, n)).astype(int).tolist())
        samples = [prepare_raw(self._warp(f), color_mode, scale) for f in self._empty
                   if f is not None and (f.shape[1], f.shape[0]) == (self.frame_w, self.frame_h)]
        n_empty = len(samples)
        cap = cv2.VideoCapture(path)
        i = 0
        while i < n:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                samples.append(prepare_raw(self._warp(f), color_mode, scale))
            i += 1
        cap.release()
        valid = None if self.exclusion is None else ~self.exclusion
        from tracking.location import low_light_zone_polys
        self.bgm = build_background(None, range(len(samples)), color_mode, polarity, valid_full=valid, scale=scale,
                                    raw_samples=samples, low_light=self.setup.get("low_light", "off"),
                                    zone_polys=low_light_zone_polys(self.setup))
        self.tracker = new_tracker(self.setup, self.bgm, self.exclusion)
        self.bodyfrac = BodyZoneFractions(self.roi_masks, self.frac_names) if self.frac_names else None
        self.drawer = _LiveDrawer(self.w, self.h, self.roi_points, self.mask_polygons, self.exclusion)
        self.background_note = (f"Background: first {n / self.fps:.0f} s of the trial"
                                + (f" + {n_empty} empty-arena pictures" if n_empty else ""))
        self._empty = []                      # free the memory
        self.phase = "tracking"

    def _sleep_until(self, due):
        while not self._finishing.is_set() and not self._cancel.is_set():
            d = due - time.time()
            if d <= 0:
                return
            time.sleep(min(d, 0.05))

    def _analyse_loop(self):
        try:
            last_draw = 0.0
            while not self._cancel.is_set():
                item = self._pieces.get()
                if item is None:
                    break
                path, first, n = item
                if self.tracker is None:
                    self._build_engine(path, n)
                prep_fn = getattr(self.tracker, "prepare_frame", None)
                cap = cv2.VideoCapture(path)
                for i in range(n):
                    if self._cancel.is_set():
                        break
                    ok, raw_f = cap.read()
                    if not ok:
                        break
                    fn = first + i
                    t = fn / self.fps
                    fr = self._warp(raw_f)
                    det = self.tracker.update(fr, prep_fn(fr)) if prep_fn is not None else self.tracker.update(fr)
                    apply_backfill(det, self.xs, self.ys, self.areas, self.status_list, ends=self.ends,
                                   fracs=self.fracs)
                    self.ends.append(det.get("ends"))
                    self.frames.append(fn)
                    self.times.append(t)
                    self.xs.append(float(det["x"]))
                    self.ys.append(float(det["y"]))
                    self.areas.append(float(det["area"]))
                    self.status_list.append(det["status"])
                    self.fracs.append(self.bodyfrac(self.tracker, det) if self.bodyfrac is not None else ())
                    self.n_processed = fn + 1
                    finishing = self._finishing.is_set()
                    if self.pace and not finishing and self.t0 is not None:
                        self._sleep_until(self.t0 + t + self.delay_s)
                    now = time.time()
                    if now - last_draw >= (0.15 if self._finishing.is_set() else 0.033):
                        last_draw = now
                        img = self.drawer.draw(fr, det, self.tracker, t, fn)
                        self._trail(img)
                        self.image = img
                        self.image_t = t
                    if now >= self._numbers_due and not self._finishing.is_set():
                        self._update_numbers()
                    if self._finishing.is_set() and self.phase == "tracking":
                        self.phase = "finishing"
                cap.release()
                try:
                    os.remove(path)
                except OSError:
                    pass
            if self._cancel.is_set():
                if self.phase != "failed":
                    self.phase = "cancelled"
                return
            if not self.frames:
                raise RuntimeError("no frames were tracked")
            self.phase = "analysing"
            self._final()
            self.phase = "done"
        except Exception as exc:
            traceback.print_exc()
            self.error = (f"Live tracking stopped: {exc}. The recording is complete -- track it like any "
                          "video.")
            self.phase = "failed"
        finally:
            if self._own_dir:
                shutil.rmtree(self.work_dir, ignore_errors=True)

    def _trail(self, img, seconds=4.0):
        """The path of the last few seconds on the tracked view."""
        k = int(seconds * self.fps)
        xs = np.asarray(self.xs[-k:], float)
        ys = np.asarray(self.ys[-k:], float)
        ok = np.isfinite(xs) & np.isfinite(ys)
        if ok.sum() >= 2:
            s = self.drawer.s
            pts = np.round(np.stack([xs[ok] * s, ys[ok] * s], axis=1)).astype(np.int32)
            cv2.polylines(img, [pts], False, (255, 120, 0), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------ results
    def _output_dir(self):
        name = os.path.splitext(os.path.basename(self.video_path))[0]
        if self.setup.get("output_dir_override"):
            d = os.path.join(self.setup["output_dir_override"], name)
            os.makedirs(d, exist_ok=True)
            return d
        return compute_output_dir(self.video_path)

    def raw(self, background=True):
        """The same dict tracking.location.track_video returns (for analyse_track and the .bta file)."""
        n = len(self.frames)
        body = getattr(self.tracker, "body_l", None)
        body_full = (body / self.tracker.s) if (body and getattr(self.tracker, "s", None)) else None
        a_full = (self.bgm.animal_area / (self.bgm.scale ** 2)) if getattr(self.bgm, "animal_area", None) else None
        if not a_full:
            good = np.asarray(self.areas[:n], float)
            good = good[np.isfinite(good) & (good > 0)]
            a_full = float(np.median(good)) if good.size else None
        name = os.path.splitext(os.path.basename(self.video_path))[0]
        total = max(self.n_recorded, n)
        return {
            "video_path": self.video_path, "output_dir": getattr(self, "_out_dir", None) or self._output_dir(),
            "video_name": name, "fps": self.fps, "frame_w": self.frame_w, "frame_h": self.frame_h,
            "total_frames": total, "start_time": 0.0, "end_time": (total - 1) / self.fps if total else 0.0,
            "start_frame": 0, "end_frame": max(0, total - 1), "warp_w": self.w, "warp_h": self.h,
            "need_warp": bool(self.need_warp), "frames": list(map(int, self.frames[:n])),
            "times": list(map(float, self.times[:n])), "xs": list(self.xs[:n]), "ys": list(self.ys[:n]),
            "areas": list(self.areas[:n]), "status": list(self.status_list[:n]), "frac_names": list(self.frac_names),
            "fracs": [list(f) for f in self.fracs[:n]] if self.frac_names else [], "stop_reason": None,
            "ends": ([None if e is None else [round(float(v), 2) for v in e] for e in self.ends[:n]]
                     if any(e is not None for e in self.ends[:n]) else []),
            "covered_zones": covered_zone_names(self.setup),
            "algorithm": "classic" if self.tracker.__class__.__name__ == "ClassicTracker" else "robust",
            "polarity": self.bgm.polarity, "background_info": self.bgm.describe() + "; " + self.background_note,
            "background_bgr": self.bgm.full_res_bgr(self.w, self.h) if background else None,
            "animal_area_full": a_full,
            "body_len_full": body_full, "live_mode": f"delayed live tracking ({self.piece_s:.0f} s pieces)",
        }

    def _update_numbers(self):
        """Results of the part tracked so far, by the same analysis as the final results (no files)."""
        from tracking.location import analyse_track
        t = time.time()
        try:
            if len(self.frames) >= max(10, int(self.fps)):
                if not hasattr(self, "_out_dir"):
                    self._out_dir = self._output_dir()
                self.numbers = analyse_track(self.raw(background=False), self.setup, write_files=False)
        except Exception as exc:              # numbers are a convenience during the trial
            print(f"Live numbers skipped: {exc}")
        self._numbers_cost = time.time() - t
        self._numbers_due = time.time() + max(2.0, 6.0 * self._numbers_cost)

    def _recheck_gaps(self, min_s=0.5, max_total_s=900.0):
        """After the trial: stretches where the animal was not found (typically the start, when it rested while
        the background was learned and so became part of it) are looked at again in the recording, with the
        floor as it is now -- the animal has walked away from that place since.  The animal is followed back in
        time from where it was found after the stretch and forward from where it was lost before it; a spot
        counts only when it links up, frame by frame, with a position that was tracked.  Returns the number of
        frames found again."""
        import copy
        import math
        import types
        st = self.status_list
        n = len(st)
        if not n or self.tracker is None or not os.path.isfile(self.video_path) or \
                not hasattr(self.tracker, "candidates"):
            return 0
        min_n = max(2, int(round(min_s * self.fps)))
        spans, i = [], 0
        while i < n:
            if st[i] == "Tracked":
                i += 1
                continue
            j = i
            while j < n and st[j] != "Tracked":
                j += 1
            if j - i >= min_n and (i > 0 or j < n):
                spans.append((i, j))
            i = j
        budget, kept = int(max_total_s * self.fps), []
        for sp in spans:                      # a time limit: the results must not wait long
            budget -= sp[1] - sp[0]
            if budget >= 0:
                kept.append(sp)
        spans = kept
        if not spans:
            return 0
        try:
            t = copy.deepcopy(self.tracker)
        except Exception:
            t = self.tracker
        s = t.s
        a_ref = t.model_area or t.a_ref
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return 0
        found = 0
        try:
            for a, b in spans:
                fa, fb = int(self.frames[a]), int(self.frames[b - 1])
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, fa) or int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != fa:
                    continue
                cands = []                        # pass 1: the animal-like spots of every frame of the stretch
                for fn in range(fa, fb + 1):
                    ok, f = cap.read()
                    if not ok or (f.shape[1], f.shape[0]) != (self.frame_w, self.frame_h):
                        break
                    fr = self._warp(f)
                    d = t.strength(fr, t.prepare_frame(fr))
                    keep = []
                    for c in t.candidates(d):
                        if t.core is not None and not t._in_core(c):
                            continue
                        if a_ref and not (0.3 * a_ref <= c["area"] <= 3.0 * a_ref):
                            continue
                        keep.append(c)
                    cands.append(keep)
                m = len(cands)
                if not m:
                    continue
                pick = [None] * m

                def follow(order, p):             # pass 2: link spot to spot from a tracked position
                    miss = 0
                    for k in order:
                        if pick[k] is not None:
                            return
                        radius = min(t.gate * (1.0 + 0.5 * miss), t.gate + 2.0 * t.body_l)
                        best, cost = None, None
                        for c in cands[k]:
                            dd = math.hypot(c["cx"] - p[0], c["cy"] - p[1])
                            if dd > radius:
                                continue
                            q = dd / radius + (0.5 * abs(math.log(max(c["area"], 1e-6) / a_ref)) if a_ref else 0.0)
                            if cost is None or q < cost:
                                best, cost = c, q
                        if best is None:
                            miss += 1
                            if miss > self.fps:
                                return
                            continue
                        miss = 0
                        pick[k] = best
                        p = (best["cx"], best["cy"])

                def det_px(idx):
                    return ((self.xs[idx] + 0.5) * s - 0.5, (self.ys[idx] + 0.5) * s - 0.5)

                if b < n and st[b] == "Tracked" and np.isfinite(self.xs[b]):
                    follow(range(m - 1, -1, -1), det_px(b))
                if a > 0 and st[a - 1] == "Tracked" and np.isfinite(self.xs[a - 1]):
                    follow(range(m), det_px(a - 1))
                for k, c in enumerate(pick):
                    if c is None:
                        continue
                    idx = a + k
                    self.xs[idx] = float((c["cx"] + 0.5) / s - 0.5)
                    self.ys[idx] = float((c["cy"] + 0.5) / s - 0.5)
                    self.areas[idx] = float(c["area"] / (s * s))
                    self.status_list[idx] = "Tracked"
                    if self.bodyfrac is not None and c.get("body") is not None:
                        bx0, by0 = c["bbox"][0], c["bbox"][1]
                        fake = types.SimpleNamespace(last_body=(c["body"], int(bx0), int(by0)), valid=t.valid)
                        self.fracs[idx] = self.bodyfrac(fake, {"status": "Tracked"})
                    found += 1
        finally:
            cap.release()
        if found:
            self.background_note += f"; {found / self.fps:.1f} s found again after the trial"
        return found

    def _final(self):
        from tracking.location import analyse_track
        from tracking.project_file import save_new_analysis
        self._out_dir = self._output_dir()
        try:
            self._recheck_gaps()
        except Exception as exc:              # a convenience: the results stand without it
            print(f"Live tracking: second look at the lost stretches skipped: {exc}")
        raw = self.raw()
        cv2.imwrite(os.path.join(self._out_dir, "background.png"), raw["background_bgr"])
        summary = analyse_track(raw, self.setup)
        summary["Live_tracking"] = raw["live_mode"]
        try:
            summary["Project_file"] = save_new_analysis(raw, self.setup, summary)
        except Exception as exc:
            print(f"BT analysis file not saved: {exc}")
        self.summary = summary
        self.numbers = summary
