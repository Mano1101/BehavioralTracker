#!/usr/bin/env python3
"""BehavioralTracker -- open reference implementation of the freezing measure ("Whole region (Classic)" mode).

This short, readable program computes freezing from a video exactly as BehavioralTracker does in its Classic mode,
so that anyone (a reviewer, a colleague, a sceptic) can see and check the method without the BT program itself.
It needs only Python 3, numpy and OpenCV (pip install numpy opencv-python).

    python reference_freezing.py VIDEO --body-length 62.5 [--region region.json] [--start 0] [--end 300]
                                 [--threshold 20] [--min-freeze 1.0] [--cutoff auto|N] [--no-light-guard]
                                 [--csv frames.csv] [--compare VIDEO_Freezing_frames.csv]

--body-length  the animal's length in video pixels (nose to tail base): BT's "Body length (px)" in the result,
               or your own 📏 measurement.  The freezing threshold is a % of the body area (length^2 / 2.4).
--region       BT's region: {"poly": [[x, y], ...], "masks": [[[x, y], ...], ...]} in video pixels, or a box
               [x0, y0, x1, y1]; default the whole frame.
--compare      BT's frame table (<video>_Freezing_frames.csv): prints the frame-by-frame agreement.

The method, step by step (the same constants as BT):
  1. The region is cut out, made grey, scaled so its longer side is at most 640 px, and blurred (Gaussian, 1 px).
  2. Each frame is compared with the frame 1/30 s before: |grey change| per pixel (0-63, 63 = 63 or more).
  3. Camera noise: the region is cut into ~40 px tiles; the 75 % quietest tiles of each frame give that frame's
     noise histogram, and the typical (median) tile gives the share of pixels a quiet tile changes by chance.
  4. Cutoff (Auto): 2 x the grey change reached by 1 pixel in 10 000 of the quiet tiles (75th percentile over the
     frames), limited to 1.6 x the 1-in-1000 level (rare compression spikes), kept between 3 and 40.
  5. Motion of a frame = pixels changed by >= cutoff in the region, minus the chance changes of a quiet tile of
     that size, as % of the body area.
  6. Light guard: a frame in which the typical tile changed by > 25 % (a light switched on / off) is not used
     (the frame keeps the state of the frame before); a dark / bright run at the very start or end (light off,
     > 1 s) is left out of the analysis.
  7. Freezing: motion below the threshold (default 20 % of the body) for at least the minimum time (1 s).
"""
import argparse
import csv
import json
import math
import sys

import cv2
import numpy as np

NBINS = 64
WORK_MAX_SIDE = 640
SIGMA = 1.0
LIGHT_FRACTION = 0.25
TILE_TARGET = 40
NOISE_KEEP = 0.75
ACTIVITY_LEVEL = 6
CUTOFF_MIN, CUTOFF_MAX = 3, 40
TAIL_RATIO = 1.6
AREA_PER_LEN2 = 1.0 / 2.4
LIGHTS_OFF_DARK, LIGHTS_OFF_BRIGHT = 0.55, 1.8


# ----------------------------------------------------------------------------------------------- region / scale
def region_parts(region):
    if not region:
        return None, []
    if isinstance(region, dict):
        poly = region.get("poly")
        poly = [(float(x), float(y)) for x, y in poly] if poly and len(poly) >= 3 else None
        masks = [[(float(x), float(y)) for x, y in m] for m in (region.get("masks") or []) if m and len(m) >= 3]
        return poly, masks
    x0, y0, x1, y1 = [float(v) for v in region]
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], []


def is_box(poly):
    if poly is None or len(poly) != 4:
        return False
    return len({round(x, 3) for x, _ in poly}) == 2 and len({round(y, 3) for _, y in poly}) == 2


def region_box(region, w, h):
    poly, _m = region_parts(region)
    if poly is None:
        return None
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    x0, x1 = sorted((max(0.0, min(w, min(xs))), max(0.0, min(w, max(xs)))))
    y0, y1 = sorted((max(0.0, min(h, min(ys))), max(0.0, min(h, max(ys)))))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    if x0 <= 0 and y0 <= 0 and x1 >= w and y1 >= h and is_box(poly):
        return None
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


class Geometry:
    def __init__(self, w, h, region):
        r = region_box(region, w, h)
        poly, masks = region_parts(region)
        poly = poly if (poly is not None and not is_box(poly)) else None
        self.x0, self.y0, self.x1, self.y1 = r if r else (0, 0, w, h)
        rw, rh = self.x1 - self.x0, self.y1 - self.y0
        self.s = min(1.0, WORK_MAX_SIDE / float(max(rw, rh)))
        self.ww, self.wh = max(8, int(round(rw * self.s))), max(8, int(round(rh * self.s)))
        tile = max(12, min(TILE_TARGET, min(self.ww, self.wh) // 4))
        nx, ny = max(1, self.ww // tile), max(1, self.wh // tile)
        tx = np.minimum(np.arange(self.ww) * nx // self.ww, nx - 1)
        ty = np.minimum(np.arange(self.wh) * ny // self.wh, ny - 1)
        self.n_tiles = nx * ny
        self.tile_idx = (ty[:, None] * nx + tx[None, :]).astype(np.int32) * NBINS
        self.tile_px = np.bincount((self.tile_idx // NBINS).ravel(), minlength=self.n_tiles).astype(np.int64)
        self.inside = None
        if poly is not None or masks:
            m = np.full((self.y1 - self.y0, self.x1 - self.x0), 255 if poly is None else 0, np.uint8)
            off = np.array([self.x0, self.y0], np.float64)
            if poly is not None:
                cv2.fillPoly(m, [np.round(np.asarray(poly) - off).astype(np.int32)], 255)
            for mk in masks:
                cv2.fillPoly(m, [np.round(np.asarray(mk) - off).astype(np.int32)], 0)
            if (m.shape[1], m.shape[0]) != (self.ww, self.wh):
                m = cv2.resize(m, (self.ww, self.wh), interpolation=cv2.INTER_AREA)
            ins = m >= 128
            if not ins.all():
                self.inside = ins
        full = self.tile_px.copy()
        if self.inside is not None:
            self.tile_px = np.bincount((self.tile_idx // NBINS)[self.inside], minlength=self.n_tiles).astype(np.int64)
        self.tile_ok = self.tile_px >= np.maximum(16, 0.3 * full)
        if not self.tile_ok.any():
            self.tile_ok = self.tile_px > 0
        self.n_px = int(self.inside.sum()) if self.inside is not None else self.ww * self.wh
        self.mask_u8 = None if self.inside is None else self.inside.astype(np.uint8) * 255

    def prep(self, frame):
        g = cv2.cvtColor(frame[self.y0:self.y1, self.x0:self.x1], cv2.COLOR_BGR2GRAY)
        if g.shape[1] != self.ww or g.shape[0] != self.wh:
            g = cv2.resize(g, (self.ww, self.wh), interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(g.astype(np.float32), (0, 0), SIGMA)


# ----------------------------------------------------------------------------------------------- per frame
def rcum(h, axis=-1):
    """cum[..., c] = count of values >= c."""
    return np.flip(np.cumsum(np.flip(h, axis=axis), axis=axis), axis=axis)


def frame_stats(geo, q):
    """(region histogram, quiet-tiles histogram, quiet-tiles pixels, typical tile's changed share per level)."""
    if geo.inside is None:
        th = np.bincount((geo.tile_idx + q).ravel(), minlength=geo.n_tiles * NBINS).reshape(geo.n_tiles, NBINS)
        tc = rcum(th, axis=1) / geo.tile_px[:, None]
        rate = np.median(tc, axis=0)
        keep = np.argsort(tc[:, ACTIVITY_LEVEL], kind="stable")[:max(1, int(round(NOISE_KEEP * geo.n_tiles)))]
        return th.sum(axis=0), th[keep].sum(axis=0), int(geo.tile_px[keep].sum()), rate
    th = np.bincount((geo.tile_idx + q)[geo.inside], minlength=geo.n_tiles * NBINS).reshape(geo.n_tiles, NBINS)
    ok = np.flatnonzero(geo.tile_ok)
    tc = rcum(th[ok], axis=1) / geo.tile_px[ok][:, None]
    rate = np.median(tc, axis=0)
    keep = ok[np.argsort(tc[:, ACTIVITY_LEVEL], kind="stable")[:max(1, int(round(NOISE_KEEP * len(ok))))]]
    return th.sum(axis=0), th[keep].sum(axis=0), int(geo.tile_px[keep].sum()), rate


def percentile_from_cum(cum, n, frac):
    target = frac * np.asarray(n, float)
    p = np.full(len(cum), np.nan)
    for c in range(NBINS - 1, -1, -1):
        hit = (cum[:, c] >= target) & np.isnan(p)
        if hit.any():
            nxt = cum[hit, c + 1] if c + 1 < NBINS else np.zeros(int(hit.sum()))
            span = np.maximum(cum[hit, c] - nxt, 1)
            p[hit] = c + np.clip((cum[hit, c] - target[hit]) / span, 0, 1)
    return p


def measure(path, region=None, start_s=0.0, end_s=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30.0 if not fps or fps <= 0 or fps > 1000 else float(fps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = total / fps
    t0 = max(0.0, min(float(start_s or 0.0), dur))
    t1 = dur if end_s is None else max(t0, min(float(end_s), dur))
    a, b = int(round(t0 * fps)), min(max(total - 1, 0), int(round(t1 * fps)))
    n = b - a + 1
    lag = max(1, int(round(fps / 30.0)))
    geo = Geometry(w, h, region)
    h_reg = np.zeros((n, NBINS), np.uint32)
    h_bg = np.zeros((n, NBINS), np.uint32)
    n_bg = np.zeros(n, np.int32)
    rate = np.zeros((n, NBINS), np.float32)
    valid = np.zeros(n, bool)
    read = np.zeros(n, bool)
    bright = np.full(n, np.nan, np.float32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    ring = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        img = geo.prep(f)
        ring.append(img)
        if len(ring) > lag + 1:
            ring.pop(0)
        bright[i] = float(cv2.mean(img, mask=geo.mask_u8)[0])
        if len(ring) == lag + 1:
            q = np.minimum(cv2.absdiff(img, ring[0]), NBINS - 1).astype(np.uint8)
            h_reg[i], h_bg[i], n_bg[i], rate[i] = frame_stats(geo, q)
            valid[i] = True
        read[i] = True
    cap.release()
    m = int(np.max(np.where(read)[0])) + 1 if read.any() else 0
    return {"fps": fps, "lag": lag, "geo": geo, "frames": np.arange(a, a + m), "h_reg": h_reg[:m],
            "h_bg": h_bg[:m], "n_bg": n_bg[:m], "rate": rate[:m], "valid": valid[:m], "bright": bright[:m]}


# ----------------------------------------------------------------------------------------------- the result
def auto_cutoff(d):
    ok = d["valid"] & (d["n_bg"] > 200)
    if not ok.any():
        return 10
    cum = rcum(d["h_bg"][ok].astype(np.int64), axis=1)
    p4 = percentile_from_cum(cum, d["n_bg"][ok], 1e-4)
    p3 = percentile_from_cum(cum, d["n_bg"][ok], 1e-3)
    good = np.isfinite(p4) & np.isfinite(p3)
    if not good.any():
        return 10
    l4, l3 = float(np.percentile(p4[good], 75)), float(np.percentile(p3[good], 75))
    level = min(l4, TAIL_RATIO * l3) if l3 > 0 else l4
    return int(min(CUTOFF_MAX, max(CUTOFF_MIN, round(2.0 * level))))


def runs(mask):
    mm = np.concatenate([[False], np.asarray(mask, bool), [False]])
    dd = np.diff(mm.astype(np.int8))
    return list(zip(np.where(dd == 1)[0].tolist(), np.where(dd == -1)[0].tolist()))


def lights_trim(d, min_s=1.0):
    b = np.asarray(d["bright"], float)
    n = len(b)
    med = float(np.nanmedian(b)) if np.isfinite(b).any() else 0.0
    if not med > 1.0:
        return 0, n
    odd = np.isfinite(b) & ((b < LIGHTS_OFF_DARK * med) | (b > LIGHTS_OFF_BRIGHT * med))
    need = max(2, int(round(min_s * d["fps"])))
    i0, i1, k = 0, n, 0
    while k < n and (odd[k] or not np.isfinite(b[k])):
        k += 1
    if k >= need and odd[:k].sum() >= need:
        i0 = min(n, k + d["lag"])
    k = n
    while k > i0 and (odd[k - 1] or not np.isfinite(b[k - 1])):
        k -= 1
    if n - k >= need and odd[k:].sum() >= need:
        i1 = max(i0, k - d["lag"])
    if i1 - i0 < max(2, d["lag"] + 2):
        return 0, n
    return i0, i1


def freezing(d, body_length_px, threshold_pct=20.0, min_freeze_s=1.0, cutoff=None, light_guard=True):
    geo, fps = d["geo"], d["fps"]
    c = auto_cutoff(d) if cutoff is None else int(cutoff)
    c = int(min(NBINS - 1, max(1, c)))
    r = d["rate"][:, c].astype(float)
    px = np.maximum(0.0, rcum(d["h_reg"][:, c:].astype(np.int64), axis=1)[:, 0] - geo.n_px * r)
    body_area_work = AREA_PER_LEN2 * float(body_length_px) ** 2 * geo.s ** 2
    mot = 100.0 * px / body_area_work
    mot[~d["valid"]] = np.nan
    n = len(mot)
    light = (r > LIGHT_FRACTION) & d["valid"] if light_guard else np.zeros(n, bool)
    i0, i1 = lights_trim(d) if light_guard else (0, n)
    keep = np.zeros(n, bool)
    keep[i0:i1] = True
    usable = d["valid"] & ~light & keep
    below = np.zeros(n, bool)
    ok = usable & np.isfinite(mot)
    below[ok] = mot[ok] < float(threshold_pct)
    if ok.any() and not ok.all():                     # unusable frames keep the state of the frame before
        idx = np.where(ok, np.arange(n), -1)
        np.maximum.accumulate(idx, out=idx)
        idx[idx < 0] = int(np.argmax(ok))
        below = below[idx]
    min_f = max(1, int(round(float(min_freeze_s) * fps)))
    frz = np.zeros(n, bool)
    for a, b in runs(below):
        if b - a >= min_f:
            frz[a:b] = True
    frz &= keep
    nk = int(keep.sum())
    return {"cutoff": c, "motion_pct": mot, "freezing": frz, "light": light, "keep": keep,
            "freezing_pct": 100.0 * frz.sum() / nk if nk else 0.0, "bouts": len(runs(frz)),
            "threshold_px": float(threshold_pct) / 100.0 * AREA_PER_LEN2 * float(body_length_px) ** 2}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--body-length", type=float, required=True, help="animal length in video pixels")
    ap.add_argument("--region", default=None, help="region.json (BT format) or x0,y0,x1,y1")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=20.0, help="% of the body area (default 20)")
    ap.add_argument("--min-freeze", type=float, default=1.0, help="seconds (default 1.0)")
    ap.add_argument("--cutoff", default="auto")
    ap.add_argument("--no-light-guard", action="store_true")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--compare", default=None, help="BT's <video>_Freezing_frames.csv")
    a = ap.parse_args(argv)
    region = None
    if a.region:
        region = json.load(open(a.region)) if a.region.endswith(".json") else [float(v) for v in a.region.split(",")]
    d = measure(a.video, region, a.start, a.end)
    r = freezing(d, a.body_length, a.threshold, a.min_freeze, None if a.cutoff == "auto" else int(a.cutoff),
                 not a.no_light_guard)
    print(f"Frames {len(d['frames'])} | cutoff {r['cutoff']} | threshold {a.threshold:g} % of the body "
          f"(= {r['threshold_px']:.0f} px) | freezing {r['freezing_pct']:.2f} % | bouts {r['bouts']}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Frame", "Time_s", "Motion_pct", "Freezing", "Light_change"])
            for i, fn in enumerate(d["frames"]):
                m = r["motion_pct"][i]
                w.writerow([int(fn), f"{fn / d['fps']:.4f}", "" if not math.isfinite(m) else f"{m:.3f}",
                            int(r["freezing"][i]), int(r["light"][i])])
    if a.compare:
        bt = {}
        with open(a.compare, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                bt[int(row["Frame"])] = int(row["Freezing"])
        same = [int(r["freezing"][i]) == bt[int(fn)] for i, fn in enumerate(d["frames"]) if int(fn) in bt]
        print(f"Agreement with BT's frame table: {100.0 * sum(same) / max(1, len(same)):.3f} % of {len(same)} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
