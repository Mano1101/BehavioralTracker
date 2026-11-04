"""Several chambers (separate arenas, one animal each) in ONE video, tracked at the same time.

A chamber is a rectangle of the (straightened) video.  Every chamber is tracked exactly as if the video had
been cropped to it: its own background model, tracker, zones, settings and result folder / .bta file.  The
video is read once for all chambers; long videos in several pieces at once (like a single animal).

Zones belong to a chamber by their name prefix ("C2 · Centre" -> chamber 2, shown as "Centre" in that
chamber's results) or, without a prefix, by where their centre lies.
"""
import math
import os
import re

import cv2
import numpy as np

MAX_CHAMBERS = 4
CHAMBER_HEX = ("#f97316", "#a855f7", "#10b981", "#ec4899")
CHAMBER_BGR = tuple((int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)) for h in CHAMBER_HEX)
_PREFIX = re.compile(r"^\s*C([1-9])\s*[·\-:.]\s*", re.I)
OVERRIDE_KEYS = ("animal_polarity", "color_mode", "min_contrast", "low_light", "start_time", "end_time")


# --------------------------------------------------------------------------------------------- names / zones
def zone_prefix(n):
    return f"C{int(n)} · "


def split_zone_name(name):
    """'C2 · Centre' -> (2, 'Centre'); 'Centre' -> (None, 'Centre')."""
    m = _PREFIX.match(str(name))
    if not m:
        return None, str(name)
    return int(m.group(1)), str(name)[m.end():].strip() or str(name)


def chamber_label(n, animal_id=""):
    return f"Chamber {int(n)}" + (f" · {animal_id}" if animal_id else "")


def short_label(n, animal_id=""):
    return f"C{int(n)}" + (f" · {animal_id}" if animal_id else "")


def safe_name(text):
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text or "")).strip("_")
    return s[:40]


def _centre(pts):
    a = np.asarray(pts, float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def chamber_of_zone(name, pts, rects):
    """Index (0-based) of the chamber a zone belongs to, or None."""
    n, _bare = split_zone_name(name)
    if n is not None:
        return n - 1 if n - 1 < len(rects) else None
    if pts is None or len(pts) < 3:
        return None
    cx, cy = _centre(pts)
    best, bd = None, None
    for i, (x0, y0, x1, y1) in enumerate(rects):
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            d = min(cx - x0, x1 - cx, cy - y0, y1 - cy)
            if bd is None or d > bd:
                best, bd = i, d
    return best


def zones_by_chamber(zones, rects):
    """{chamber index: {bare name: points}} (zones outside every chamber are left out)."""
    out = {i: {} for i in range(len(rects))}
    for name, pts in (zones or {}).items():
        i = chamber_of_zone(name, pts, rects)
        if i is None:
            continue
        bare = split_zone_name(name)[1]
        while bare in out[i]:
            bare += "'"
        out[i][bare] = pts
    return out


def clip_rect(rect, w, h, min_side=8):
    x0, y0, x1, y1 = [float(v) for v in rect]
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0 = int(max(0, min(math.floor(x0), w - min_side)))
    y0 = int(max(0, min(math.floor(y0), h - min_side)))
    x1 = int(min(w, max(math.ceil(x1), x0 + min_side)))
    y1 = int(min(h, max(math.ceil(y1), y0 + min_side)))
    return x0, y0, x1, y1


# --------------------------------------------------------------------------------------------- per-chamber setup
def chamber_setups(base, chambers):
    """Per-chamber setups from the whole-video `base` setup (as built for one animal) and
    chambers = [{"rect": (x0, y0, x1, y1) in base (straightened) coordinates, "id": "M1",
                 "overrides": {setup key: value}}].
    Each chamber setup crops the video to its rectangle (matrix = crop x straightening)."""
    W, H = int(base["warp_w"]), int(base["warp_h"])
    M = np.asarray(base.get("matrix") if base.get("matrix") is not None else np.eye(3), np.float64)
    rects = [clip_rect(c["rect"], W, H) for c in chambers]
    zones = base.get("roi_points") or {}
    order = [n for n in (base.get("roi_names") or list(zones)) if n in zones]
    order += [n for n in zones if n not in order]
    by = zones_by_chamber({n: zones[n] for n in order}, rects)
    cov_all = base.get("covered_zones")
    stem = os.path.splitext(os.path.basename(str(base.get("video_path") or "video")))[0]
    out = []
    for i, (c, r) in enumerate(zip(chambers, rects)):
        x0, y0, x1, y1 = r
        T = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], np.float64)
        s = dict(base)
        s["matrix"] = (T @ M).astype(np.float32)
        s["warp_w"], s["warp_h"] = x1 - x0, y1 - y0
        s["use_crop"] = True
        s["arena"] = (0, 0, x1 - x0, y1 - y0)
        z = {bare: [(float(px) - x0, float(py) - y0) for px, py in pts] for bare, pts in by[i].items()}
        s["roi_points"] = z
        s["roi_names"] = list(z)
        s["mask_points"] = [[(float(px) - x0, float(py) - y0) for px, py in m] for m in (base.get("mask_points") or [])]
        s["auto_mask_outside_zones"] = bool(base.get("auto_mask_outside_zones")) and bool(z)
        if cov_all is not None:
            names = []
            for n in cov_all:
                if n in zones and chamber_of_zone(n, zones[n], rects) == i:
                    names.append(split_zone_name(n)[1])
            s["covered_zones"] = [n for n in names if n in z]
        s["zone_groups"] = {}
        for k, v in (c.get("overrides") or {}).items():
            if k in OVERRIDE_KEYS and v is not None:
                s[k] = v
        cid = str(c.get("id") or "").strip()
        s["chamber"] = i + 1
        s["chamber_id"] = cid
        s["chamber_count"] = len(chambers)
        s["chamber_rect"] = [int(x0), int(y0), int(x1), int(y1)]
        s["chamber_base"] = {"matrix": M.tolist(), "warp_w": W, "warp_h": H}
        if cid:
            s["subject"] = cid
        elif base.get("subject"):
            s["subject"] = f"{base['subject']} C{i + 1}"
        s["_out_name"] = f"{stem}_Chamber{i + 1}" + (f"_{safe_name(cid)}" if safe_name(cid) else "")
        out.append(s)
    return out


def results_root(video_path, base):
    from tracking.location import results_root
    root = base.get("output_dir_override") or results_root(video_path)
    return os.path.join(root, os.path.splitext(os.path.basename(video_path))[0])


# --------------------------------------------------------------------------------------------- finding chambers
def find_chambers(video_path, n, matrix=None, size=None, samples=48, progress=None):
    """Rectangles (x0, y0, x1, y1) of `n` chambers in straightened-video coordinates.

    Where the animals move during the video gives one area per chamber; around each, the floor of the same
    colour (bedding, the box floor) is followed out to the chamber's edge, a small margin is added for the
    walls.  Falls back to an even split of the picture when fewer moving areas are found.
    Returns (rects, found_by_motion)."""
    from tracking.location import read_frames_parallel
    n = int(max(1, min(MAX_CHAMBERS, n)))
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    W, H = (int(size[0]), int(size[1])) if size else (fw, fh)
    M = None if matrix is None else np.asarray(matrix, np.float32)
    ident = M is None or (np.allclose(M, np.eye(3), atol=1e-9) and (W, H) == (fw, fh))
    sc = min(1.0, 320.0 / max(W, H, 1))
    sw, sh = max(8, int(round(W * sc))), max(8, int(round(H * sc)))

    def small(f):
        if not ident:
            f = cv2.warpPerspective(f, M, (W, H))
        if f.ndim == 2:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        return cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA)
    pos = np.linspace(0, max(0, total - 2), max(8, int(samples))).astype(int) if total > 1 else [0]
    frames = [f for f in read_frames_parallel(video_path, pos, small, progress=progress) if f is not None]
    rects = []
    by_motion = False
    if len(frames) >= 8:
        st = np.stack(frames)
        gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]).astype(np.float32)
        med = np.median(gray, axis=0)
        energy = np.percentile(np.abs(gray - med[None]), 90, axis=0)
        energy = cv2.GaussianBlur(energy, (0, 0), 1.2)
        thr = max(12.0, float(np.median(energy)) * 4.0, float(np.percentile(energy, 75)))
        m = (energy > thr).astype(np.uint8)
        k = max(3, int(round(min(sw, sh) * 0.06)) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        num, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
        comps = sorted(range(1, num), key=lambda j: -stats[j, cv2.CC_STAT_AREA])
        comps = [j for j in comps if stats[j, cv2.CC_STAT_AREA] >= 0.004 * sw * sh][:n]
        if len(comps) == n:
            by_motion = True
            bg = np.median(st, axis=0).astype(np.uint8)
            labc = cv2.cvtColor(cv2.GaussianBlur(bg, (0, 0), 2.0), cv2.COLOR_BGR2LAB).astype(np.float32)
            centres = np.array([cents[j] for j in comps], float)
            yy, xx = np.mgrid[0:sh, 0:sw]
            dist_c = np.stack([np.hypot(xx - cx, yy - cy) for cx, cy in centres])
            owner = np.argmin(dist_c, axis=0)
            for q, j in enumerate(comps):
                seed = lab == j
                x, y, bw, bh = stats[j, :4]
                box = [float(x), float(y), float(x + bw), float(y + bh)]
                floor = _floor_box(labc, seed, owner == q)
                if floor is not None and (floor[2] - floor[0]) * (floor[3] - floor[1]) <= 12.0 * bw * bh:
                    box = [min(box[0], floor[0]), min(box[1], floor[1]), max(box[2], floor[2]), max(box[3], floor[3])]
                    pad = 0.06
                else:
                    pad = 0.15
                px, py = max(2.0, pad * (box[2] - box[0])), max(2.0, pad * (box[3] - box[1]))
                rects.append([(box[0] - px) / sc, (box[1] - py) / sc, (box[2] + px) / sc, (box[3] + py) / sc])
            rects = _separate(rects)
    if not rects:
        rects = even_split(W, H, n)
    rects = [clip_rect(r, W, H) for r in rects]
    rects.sort(key=lambda r: (round(((r[1] + r[3]) / 2) / max(1.0, H / 2.0)), r[0]))
    return rects, by_motion


def _floor_box(labc, seed, region):
    """Bounding box of the floor around a moving area: the pixels coloured like the background where the
    animal moved (colour counts more than brightness; split from the surroundings by Otsu), connected to
    it, inside the chamber's own part of the picture."""
    px = labc[seed]
    if px.shape[0] < 12 or not region.any():
        return None
    ref = np.median(px, axis=0)
    d = np.sqrt(((labc[..., 1:] - ref[1:]) ** 2).sum(-1) + 0.25 * (labc[..., 0] - ref[0]) ** 2)
    dd = np.clip(d, 0, 255).astype(np.uint8)
    t, _m = cv2.threshold(dd[region].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = ((d < max(8.0, float(t))) & region).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    num, lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    best, bo = None, 0
    for j in range(1, num):
        ov = int((lab[seed] == j).sum())
        if ov > bo:
            best, bo = j, ov
    if best is None or bo < 0.3 * int(seed.sum()):
        return None
    x, y, bw, bh = st[best, :4]
    ys, xs = np.nonzero(region)
    reg_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    if st[best, cv2.CC_STAT_AREA] < 0.5 * bw * bh or bw * bh > 0.9 * reg_area:
        return None                                       # leaked into the surroundings: not a floor
    return [float(x), float(y), float(x + bw), float(y + bh)]


def even_split(W, H, n):
    if n == 4:
        return [[0, 0, W / 2, H / 2], [W / 2, 0, W, H / 2], [0, H / 2, W / 2, H], [W / 2, H / 2, W, H]]
    if W >= H:
        return [[W * i / n, 0, W * (i + 1) / n, H] for i in range(n)]
    return [[0, H * i / n, W, H * (i + 1) / n] for i in range(n)]


def _separate(rects):
    """Overlapping rectangles (generous margins) are cut at the middle of their overlap."""
    r = [list(map(float, x)) for x in rects]
    for _ in range(3):
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                a, b = r[i], r[j]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oy = min(a[3], b[3]) - max(a[1], b[1])
                if ox <= 0 or oy <= 0:
                    continue
                if ox <= oy:                       # side by side: cut vertically
                    left, right = (a, b) if a[0] + a[2] < b[0] + b[2] else (b, a)
                    mid = (left[2] + right[0]) / 2.0
                    left[2], right[0] = mid, mid
                else:
                    top, bot = (a, b) if a[1] + a[3] < b[1] + b[3] else (b, a)
                    mid = (top[3] + bot[1]) / 2.0
                    top[3], bot[1] = mid, mid
    return r


# --------------------------------------------------------------------------------------------- live picture
class ChamberLive:
    """One live picture for all chambers: the whole (straightened) frame, outside the chambers dimmed, each
    chamber with its zones, its colour frame + name and its animal."""

    def __init__(self, W, H, infos, max_w=960):
        self.s = s = min(1.0, float(max_w) / float(max(W, 1)))
        self.size = (max(1, int(round(W * s))), max(1, int(round(H * s))))
        self.ch = []
        outside = np.ones((self.size[1], self.size[0]), bool)
        for i, inf in enumerate(infos):
            x0, y0, x1, y1 = inf["rect"]
            rx0, ry0 = int(round(x0 * s)), int(round(y0 * s))
            rx1, ry1 = max(rx0 + 1, int(round(x1 * s))), max(ry0 + 1, int(round(y1 * s)))
            outside[ry0:ry1, rx0:rx1] = False
            excl = inf.get("exclusion")
            if excl is not None:
                excl = cv2.resize(excl.astype(np.uint8), (rx1 - rx0, ry1 - ry0), interpolation=cv2.INTER_NEAREST) > 0
            roi = {k: [((float(px) + x0) * s, (float(py) + y0) * s) for px, py in pts]
                   for k, pts in (inf.get("roi_points") or {}).items()}
            self.ch.append({"rect": (x0, y0, x1, y1), "r": (rx0, ry0, rx1, ry1), "excl": excl, "roi": roi,
                            "label": inf.get("label") or f"C{i + 1}", "tag": inf.get("tag") or f"C{i + 1}",
                            "col": CHAMBER_BGR[i % len(CHAMBER_BGR)]})
        self.outside = outside

    def draw(self, full, dets, trackers, t, frame_number, lost_label="SEARCHING"):
        from tracking.location import (draw_crosshair, draw_status_box, clock_text, zone_line_width,
                                       outline_arrays, ROI_COLORS)
        from tracking.robust_tracker import brighten
        s = self.s
        disp = cv2.resize(full, self.size, interpolation=cv2.INTER_AREA) if s < 1.0 else full.copy()
        if disp.ndim == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        disp[self.outside] = (disp[self.outside] * 0.45).astype(np.uint8)
        lw = zone_line_width(disp)
        size = max(6, int(min(disp.shape[:2]) * 0.018))
        fs = max(0.38, min(disp.shape[:2]) / 1100.0)
        th = max(1, int(round(1.4 * fs)))
        states = []
        for c, det, tr in zip(self.ch, dets, trackers):
            rx0, ry0, rx1, ry1 = c["r"]
            sub = disp[ry0:ry1, rx0:rx1]
            g = float(getattr(getattr(tr, "bg", None), "gamma", 1.0) or 1.0) if tr is not None else 1.0
            if g > 1.001:
                sub[:] = brighten(sub, g)
            if c["excl"] is not None and c["excl"].shape == sub.shape[:2]:
                sub[c["excl"]] = (sub[c["excl"]] * 0.35).astype(np.uint8)
            tracked = det is not None and det.get("status") == "Tracked"
            if tracked and tr is not None and hasattr(tr, "overlay"):
                sub[:] = tr.overlay(sub, color=(0, 215, 255))
            for k, (_n, pts) in enumerate(c["roi"].items()):
                cv2.polylines(disp, outline_arrays(pts), True, ROI_COLORS[k % len(ROI_COLORS)], lw, cv2.LINE_AA)
            cv2.rectangle(disp, (rx0, ry0), (rx1 - 1, ry1 - 1), c["col"], max(2, lw + 1), cv2.LINE_AA)
            (tw, tht), base = cv2.getTextSize(c["label"], cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ty0 = ry0 - tht - base - 6 if ry0 - tht - base - 6 >= 0 else ry0
            cv2.rectangle(disp, (rx0, ty0), (rx0 + tw + 8, ty0 + tht + base + 6), c["col"], -1)
            cv2.putText(disp, c["label"], (rx0 + 4, ty0 + tht + 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255),
                        th, cv2.LINE_AA)
            if tracked:
                x0, y0 = c["rect"][:2]
                x, y = int(round((det["x"] + x0) * s)), int(round((det["y"] + y0) * s))
                cv2.circle(disp, (x, y), size + 3, (255, 255, 255), 2, cv2.LINE_AA)
                draw_crosshair(disp, (x, y), size=size, color=c["col"], thickness=2)
            states.append(f"{c['tag']} {'OK' if tracked else lost_label}")
        ok_all = all(d is not None and d.get("status") == "Tracked" for d in dets)
        return draw_status_box(disp, f"{clock_text(t)}   frame {frame_number}   " + "   ".join(states),
                               (40, 200, 40) if ok_all else (0, 140, 255))


# --------------------------------------------------------------------------------------------- tracking
def _prep_chamber(setup, fps, total_frames, duration):
    """roi masks, exclusion, body-in-zone fractions and the time window of one chamber setup."""
    from tracking.location import (build_roi_masks, build_detection_exclusion, covered_zone_names,
                                   BodyZoneFractions)
    w, h = int(setup["warp_w"]), int(setup["warp_h"])
    roi_points = {k: [tuple(map(float, p)) for p in v] for k, v in (setup.get("roi_points") or {}).items()
                  if len(v) >= 3}
    roi_names = [n for n in (setup.get("roi_names") or list(roi_points)) if n in roi_points]
    roi_names += [n for n in roi_points if n not in roi_names]
    roi_masks = build_roi_masks((h, w), roi_points)
    mask_polygons = setup.get("mask_points", []) or []
    auto = bool(setup.get("auto_mask_outside_zones")) and bool(roi_points)
    covered = covered_zone_names(setup)
    exclusion = None
    if mask_polygons or auto or covered:
        exclusion = build_detection_exclusion(w, h, mask_polygons, roi_points, auto,
                                              margin=setup.get("auto_mask_margin"), covered=covered)
        if exclusion.all():
            exclusion = None
    start_time = max(0.0, min(float(setup.get("start_time", 0.0) or 0.0), duration))
    end_req = setup.get("end_time")
    end_time = duration if end_req is None else max(start_time, min(float(end_req), duration))
    start_frame = int(round(start_time * fps))
    end_frame = min(max(total_frames - 1, 0), int(round(end_time * fps)))
    entry_rule = str(setup.get("entry_rule", "paws")).lower()
    frac_names = list(roi_names) if (entry_rule == "paws" and setup.get("test_type") != "nor" and roi_names) else []
    return {"w": w, "h": h, "roi_points": roi_points, "roi_names": roi_names, "roi_masks": roi_masks,
            "mask_polygons": mask_polygons, "covered": covered, "exclusion": exclusion,
            "start_time": start_time, "end_time": end_time, "end_req": end_req,
            "start_frame": start_frame, "end_frame": max(start_frame, end_frame),
            "frac_names": frac_names,
            "bodyfrac": BodyZoneFractions(roi_masks, frac_names) if frac_names else None}


class _StopCheck:
    """A chamber's stop condition (immobility / zone entries / distance), checked frame by frame."""

    def __init__(self, setup, roi_masks, fps):
        self.cond = setup.get("stop_condition")
        self.value = setup.get("stop_value")
        self.zone = setup.get("stop_zone_name")
        self.masks = roi_masks
        self.fps = fps
        self.immobile = 0.0
        self.entries = 0
        self.was_in = False
        self.cum = 0.0
        self.last = None

    def __call__(self, det):
        from tracking.location import roi_membership
        if not self.cond or self.value is None:
            return None
        moved = 0.0
        if det["status"] == "Tracked":
            if self.last is not None:
                moved = math.hypot(det["x"] - self.last[0], det["y"] - self.last[1])
            self.last = (det["x"], det["y"])
        if self.cond == "immobility":
            self.immobile = self.immobile + 1.0 / self.fps if (det["status"] == "Tracked" and moved < 2.0) else 0.0
            if self.immobile >= self.value:
                return f"Immobile for {self.value:g}s"
        elif self.cond == "zone_entries" and self.zone in self.masks:
            now_in = det["status"] == "Tracked" and roi_membership({self.zone: self.masks[self.zone]},
                                                                   det["x"], det["y"])[self.zone]
            if now_in and not self.was_in:
                self.entries += 1
            self.was_in = now_in
            if self.entries >= self.value:
                return f"{int(self.value)} entries into '{self.zone}'"
        elif self.cond == "distance":
            self.cum += moved
            if self.cum >= self.value:
                return f"Reached {self.value:g}px total distance"
        return None


def _cols():
    return {"frames": [], "times": [], "xs": [], "ys": [], "areas": [], "status": [], "fracs": [], "ends": []}


def _record(cols, fn, fps, det, bf):
    from tracking.location import apply_backfill
    apply_backfill(det, cols["xs"], cols["ys"], cols["areas"], cols["status"], ends=cols["ends"], fracs=cols["fracs"])
    cols["frames"].append(fn)
    cols["times"].append(fn / fps)
    cols["xs"].append(det["x"])
    cols["ys"].append(det["y"])
    cols["areas"].append(det["area"])
    cols["status"].append(det["status"])
    cols["fracs"].append(bf)
    cols["ends"].append(det.get("ends"))


def _track_parallel_multi(video_path, chs, base_warp, F0, F1, fps, workers, prog, frame_callback=None,
                          live=None):
    """All chambers in `workers` pieces at the same time (one decoder per piece, every chamber's tracker in
    it).  Each piece starts a few seconds early (warm-up).  Returns [cols per chamber] + trackers, or None when
    the video cannot be positioned exactly."""
    import threading
    from tracking.location import new_tracker, _live_meta
    warm = max(90, int(round(3.0 * fps)))
    n = F1 - F0 + 1
    bounds = [F0 + k * n // workers for k in range(workers)] + [F1 + 1]
    results = [None] * workers
    trackers = [None] * workers
    done = [0] * workers
    stop = threading.Event()
    errors = []
    live_due = getattr(frame_callback, "due", None)
    live_k = [0]                     # the live picture follows the first piece that is still running
    finished = [False] * workers

    def work(k):
        a, b = bounds[k], bounds[k + 1]
        s0 = F0 if k == 0 else max(F0, a - warm)
        # where each chamber starts / stops in this piece
        plan = []
        for c in chs:
            cs0 = max(s0, c["start_frame"])
            ce = min(b - 1, c["end_frame"])
            plan.append((cs0, max(a, c["start_frame"]), ce))
        active = [p[0] <= p[2] for p in plan]
        if not any(active):
            results[k] = [_cols() for _ in chs]
            trackers[k] = [None] * len(chs)
            finished[k] = True
            return
        first = min(p[0] for p, on in zip(plan, active) if on)
        last = max(p[2] for p, on in zip(plan, active) if on)
        cap = cv2.VideoCapture(video_path)
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            pos = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES) or first))
            while pos < first:                               # some codecs land a bit early
                if not cap.grab():
                    break
                pos += 1
            if pos != first:
                errors.append("seek")
                stop.set()
                return
            trs = [None] * len(chs)
            cols = [_cols() for _ in chs]
            dets = [None] * len(chs)
            fn = first
            while fn <= last and not stop.is_set():
                ok, raw = cap.read()
                if not ok:
                    break
                full = base_warp(raw)
                for j, c in enumerate(chs):
                    cs0, rec0, ce = plan[j]
                    if not active[j] or fn < cs0 or fn > ce:
                        continue
                    if trs[j] is None:
                        trs[j] = new_tracker(c["setup"], c["bgm"], c["exclusion"])
                    tr = trs[j]
                    x0, y0, x1, y1 = c["rect"]
                    fr = full[y0:y1, x0:x1]
                    prep_fn = getattr(tr, "prepare_frame", None)
                    det = tr.update(fr, prep_fn(fr)) if prep_fn is not None else tr.update(fr)
                    dets[j] = det
                    if fn >= rec0:
                        bf = c["bodyfrac"](tr, det) if c["bodyfrac"] is not None else ()
                        _record(cols[j], fn, fps, det, bf)
                if fn >= a:
                    done[k] += 1
                    if (k == live_k[0] and frame_callback is not None and live is not None
                            and (live_due is not None or (fn - F0) % 5 == 0) and (live_due is None or live_due())):
                        _live_meta(frame_callback, fn, fn / fps)
                        frame_callback(live.draw(full, dets, trs, fn / fps, fn))
                fn += 1
            results[k] = cols
            trackers[k] = trs
        except BaseException as exc:                         # incl. "stopped by user" from the live view
            errors.append(exc)
            stop.set()
        finally:
            cap.release()
            finished[k] = True
            while live_k[0] < workers - 1 and finished[live_k[0]]:
                live_k[0] += 1

    threads = [threading.Thread(target=work, args=(k,), name=f"bt-chtrack{k}", daemon=True) for k in range(workers)]
    for t in threads:
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(timeout=0.2)
            prog(0.08 + 0.84 * min(1.0, sum(done) / n))
    except BaseException:
        stop.set()
        for t in threads:
            t.join(timeout=10)
        raise
    real = [e for e in errors if isinstance(e, BaseException)]
    if real:
        raise real[0]
    if errors:
        return None
    merged = [_cols() for _ in chs]
    last_tr = [None] * len(chs)
    ended = [False] * len(chs)
    for k in range(workers):
        cols = results[k]
        if cols is None:
            break
        a, b = bounds[k], bounds[k + 1]
        for j, c in enumerate(chs):
            if ended[j]:
                continue
            for key in merged[j]:
                merged[j][key].extend(cols[j][key])
            if trackers[k][j] is not None:
                last_tr[j] = trackers[k][j]
            want = max(0, min(b - 1, c["end_frame"]) - max(a, c["start_frame"]) + 1)
            if len(cols[j]["frames"]) < want:            # the video ended early here: stop, as frame by frame
                ended[j] = True
    if not any(m["frames"] for m in merged):
        return None
    return merged, last_tr


def track_chambers(video_path, setups, progress_callback=None, frame_callback=None):
    """Track every chamber of one video (setups from chamber_setups()).  Returns one `raw` dict per chamber,
    exactly what track_video() returns for a video cropped to that chamber."""
    import glob
    import queue
    import threading
    from tracking.location import (make_tracking_engine, worker_count, _is_identity_warp, detection_still,
                                   _live_meta)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:
        fps = 30.0
    if setups[0].get("fps_override"):
        fps = float(setups[0]["fps_override"])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w, frame_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps else 0.0
    base = setups[0]["chamber_base"]
    BM = np.asarray(base["matrix"], np.float32)
    BW, BH = int(base["warp_w"]), int(base["warp_h"])
    base_ident = _is_identity_warp(BM, frame_w, frame_h, BW, BH)

    def base_warp(frame):
        return frame if base_ident else cv2.warpPerspective(frame, BM, (BW, BH))

    def prog(f):
        if progress_callback is not None:
            progress_callback(min(1.0, max(0.0, f)))

    root = results_root(video_path, setups[0])
    chs = []
    for s in setups:
        c = _prep_chamber(s, fps, total_frames, duration)
        c["rect"] = tuple(int(v) for v in s["chamber_rect"])
        x0, y0, x1, y1 = c["rect"]
        c["need_warp"] = True
        c["output_dir"] = os.path.join(root, s["_out_name"])
        os.makedirs(c["output_dir"], exist_ok=True)
        s = dict(s)
        s["_fps"] = fps
        c["setup"] = s
        chs.append(c)

    # ---- background model of every chamber (each from its own time window, as a single run)
    n_ch = len(chs)
    for i, c in enumerate(chs):
        x0, y0, x1, y1 = c["rect"]

        def cwarp(frame, r=(x0, y0, x1, y1)):
            return np.ascontiguousarray(base_warp(frame)[r[1]:r[3], r[0]:r[2]])

        def read_warped(idx, w=cwarp):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            return w(f) if ok else None
        tr, bgm = make_tracking_engine(c["setup"], c["w"], c["h"], read_warped, c["start_frame"], c["end_frame"],
                                       c["exclusion"], progress=lambda f, i=i: prog(0.08 * (i + f) / n_ch),
                                       video_path=video_path, warp_fn=cwarp)
        c["bgm"], c["tracker0"] = bgm, tr
        c["background_bgr"] = bgm.full_res_bgr(c["w"], c["h"])
        cv2.imwrite(os.path.join(c["output_dir"], "background.png"), c["background_bgr"])
        c["algorithm"] = "classic" if tr.__class__.__name__ == "ClassicTracker" else "robust"

    F0 = min(c["start_frame"] for c in chs)
    F1 = max(c["end_frame"] for c in chs)
    live = None
    if frame_callback is not None:
        live = ChamberLive(BW, BH, [{"rect": c["rect"], "roi_points": c["roi_points"], "exclusion": c["exclusion"],
                                     "label": chamber_label(c["setup"]["chamber"], c["setup"].get("chamber_id")),
                                     "tag": short_label(c["setup"]["chamber"], c["setup"].get("chamber_id"))}
                                    for c in chs])
    stopping = any(c["setup"].get("stop_condition") for c in chs)
    workers = int(setups[0].get("workers") or worker_count(F1 - F0 + 1))
    if any(c["covered"] for c in chs):
        workers = 1          # an animal hides for long stretches: each piece must know where it was last seen
    par = None
    if workers > 1 and not stopping:
        par = _track_parallel_multi(video_path, chs, base_warp, F0, F1, fps, workers, prog,
                                    frame_callback=frame_callback, live=live)
    stop_reasons = [None] * n_ch
    if par is not None:
        cols_all, trackers = par
        cap.release()
    else:
        cols_all = [_cols() for _ in chs]
        trackers = [c["tracker0"] for c in chs]
        checks = [_StopCheck(c["setup"], c["roi_masks"], fps) for c in chs]
        cap.set(cv2.CAP_PROP_POS_FRAMES, F0)
        pos_now = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or F0)
        while pos_now < F0:
            if not cap.grab():
                break
            pos_now += 1
        q = queue.Queue(maxsize=24)
        reader_stop = threading.Event()

        def _reader():
            fn = F0
            try:
                while fn <= F1 and not reader_stop.is_set():
                    ok_r, raw_r = cap.read()
                    if not ok_r:
                        break
                    full = base_warp(raw_r)
                    subs = []
                    for j, c in enumerate(chs):
                        if c["start_frame"] <= fn <= c["end_frame"]:
                            x0, y0, x1, y1 = c["rect"]
                            fr = full[y0:y1, x0:x1]
                            pf = getattr(trackers[j], "prepare_frame", None)
                            subs.append((fr, pf(fr) if pf is not None else None))
                        else:
                            subs.append(None)
                    item = (fn, full, subs)
                    while not reader_stop.is_set():
                        try:
                            q.put(item, timeout=0.2)
                            break
                        except queue.Full:
                            continue
                    fn += 1
            except Exception as exc:
                q.put(("error", exc, None))
            finally:
                q.put(None)

        reader = threading.Thread(target=_reader, name="bt-chreader", daemon=True)
        reader.start()
        live_due = getattr(frame_callback, "due", None)
        n_exp = max(1, F1 - F0 + 1)
        dets = [None] * n_ch
        stopped = [False] * n_ch
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                if item[0] == "error":
                    raise item[1]
                fn, full, subs = item
                t = fn / fps
                for j, c in enumerate(chs):
                    if subs[j] is None or stopped[j]:
                        continue
                    fr, prepared = subs[j]
                    tr = trackers[j]
                    det = tr.update(fr, prepared) if prepared is not None else tr.update(fr)
                    dets[j] = det
                    bf = c["bodyfrac"](tr, det) if c["bodyfrac"] is not None else ()
                    _record(cols_all[j], fn, fps, det, bf)
                    why = checks[j](det)
                    if why:
                        stop_reasons[j] = why
                        stopped[j] = True
                if (frame_callback is not None and live is not None
                        and (live_due is not None or (fn - F0) % 5 == 0) and (live_due is None or live_due())):
                    _live_meta(frame_callback, fn, t)
                    frame_callback(live.draw(full, dets, trackers, t, fn))
                if (fn - F0) % 5 == 0:
                    prog(0.08 + 0.84 * (fn - F0 + 1) / n_exp)
                if all(stopped[j] or fn >= c["end_frame"] for j, c in enumerate(chs)):
                    break
        finally:
            reader_stop.set()
            while reader.is_alive():
                try:
                    q.get(timeout=0.1)
                except queue.Empty:
                    pass
            reader.join(timeout=5)
            cap.release()

    raws = []
    for j, c in enumerate(chs):
        cols = cols_all[j]
        if not cols["frames"]:
            raise RuntimeError(f"No frames could be read for chamber {j + 1} in its time window.")
        tracker = trackers[j] if trackers[j] is not None else c["tracker0"]
        s = c["setup"]
        x0, y0, x1, y1 = c["rect"]
        # preview pictures of this chamber
        try:
            preview_dir = os.path.join(c["output_dir"], "preview")
            os.makedirs(preview_dir, exist_ok=True)
            for stale in glob.glob(os.path.join(preview_dir, "preview_*.png")):
                os.remove(stale)
            n_prev = max(0, int(s.get("preview_samples", 3) if s.get("preview_samples") is not None else 3))
            if n_prev:
                pcap = cv2.VideoCapture(video_path)
                try:
                    frames = cols["frames"]
                    for i, k in enumerate(np.linspace(0, len(frames) - 1, n_prev + 2).astype(int)[1:-1], start=1):
                        pf = int(frames[k])
                        pcap.set(cv2.CAP_PROP_POS_FRAMES, pf)
                        ok, raw_f = pcap.read()
                        if not ok:
                            continue
                        sub = np.ascontiguousarray(base_warp(raw_f)[y0:y1, x0:x1])
                        disp = detection_still(s, c["bgm"], c["exclusion"], sub, cols["xs"][k], cols["ys"][k],
                                               cols["status"][k], pf / fps, pf, c["roi_points"], c["mask_polygons"],
                                               lost_label="NOT SEEN (hidden?)" if c["covered"] else "NOT SEEN")
                        cv2.imwrite(os.path.join(preview_dir, f"preview_{i:02d}_frame{pf}.png"), disp)
                finally:
                    pcap.release()
        except Exception as exc:
            print(f"Preview images skipped: {exc}")
        bgm = c["bgm"]
        body = getattr(tracker, "body_l", None)
        body_full = (body / tracker.s) if (body and getattr(tracker, "s", None)) else None
        a_full = (bgm.animal_area / (bgm.scale ** 2)) if getattr(bgm, "animal_area", None) else None
        if not a_full:
            good = np.asarray(cols["areas"], float)
            good = good[np.isfinite(good) & (good > 0)]
            a_full = float(np.median(good)) if good.size else None
        ends = cols["ends"]
        raws.append({
            "video_path": video_path, "output_dir": c["output_dir"], "video_name": s["_out_name"],
            "fps": fps, "frame_w": frame_w, "frame_h": frame_h, "total_frames": total_frames,
            "start_time": c["start_time"], "end_time": c["end_time"],
            "start_frame": c["start_frame"], "end_frame": c["end_frame"],
            "requested_end_time": None if c["end_req"] is None else float(c["end_req"]),
            "warp_w": c["w"], "warp_h": c["h"], "need_warp": True,
            "frames": list(map(int, cols["frames"])), "times": list(map(float, cols["times"])),
            "xs": list(map(float, cols["xs"])), "ys": list(map(float, cols["ys"])),
            "areas": list(map(float, cols["areas"])), "status": list(cols["status"]),
            "frac_names": list(c["frac_names"]),
            "fracs": [list(f) for f in cols["fracs"]] if c["frac_names"] else [],
            "ends": ([None if e is None else [round(float(v), 2) for v in e] for e in ends]
                     if any(e is not None for e in ends) else []),
            "covered_zones": list(c["covered"]),
            "stop_reason": stop_reasons[j], "algorithm": c["algorithm"], "polarity": bgm.polarity,
            "background_info": bgm.describe(), "background_bgr": c["background_bgr"],
            "animal_area_full": a_full, "body_len_full": body_full,
            "low_light_gamma": float(getattr(bgm, "gamma", 1.0) or 1.0),
            "polarity_warning": getattr(bgm, "polarity_warning", None),
            "chamber": int(s["chamber"]), "chamber_id": s.get("chamber_id") or "",
            "chamber_rect": [int(x0), int(y0), int(x1), int(y1)],
        })
    return raws


# --------------------------------------------------------------------------------------------- whole run
SUMMARY_FIRST = ("Video_file", "Chamber", "Animal_ID", "Subject", "Test_type", "Duration_s", "Total_distance_cm",
                 "Mean_speed_cm_s", "Tracking_quality_percent")


def _flat(summary):
    row = {}
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool, np.integer, np.floating)) or v is None:
            row[k] = v.item() if isinstance(v, (np.integer, np.floating)) else v
    return row


def write_chamber_summary(video_path, summaries, folder):
    """One row per chamber (animal): <video>_chambers_summary.xlsx + .csv in the video's results folder."""
    import pandas as pd
    rows = [_flat(s) for s in summaries]
    if not rows:
        return None
    df = pd.DataFrame(rows)
    first = [c for c in SUMMARY_FIRST if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]
    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(folder, exist_ok=True)
    xlsx = os.path.join(folder, f"{stem}_chambers_summary.xlsx")
    csv = os.path.join(folder, f"{stem}_chambers_summary.csv")
    df.to_csv(csv, index=False)
    try:
        with pd.ExcelWriter(xlsx) as xw:
            df.to_excel(xw, sheet_name="Chambers", index=False)
            ws = xw.sheets["Chambers"]
            for i, col in enumerate(df.columns, start=1):
                width = min(48, max(10, len(str(col)) + 2))
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
            ws.freeze_panes = "D2"
    except Exception as exc:
        print(f"Chamber summary workbook skipped: {exc}")
        xlsx = None
    return {"xlsx": xlsx, "csv": csv}


def process_chamber_video(video_path, setup, chambers, progress_callback=None, frame_callback=None,
                          save_project=True):
    """Track all chambers of one video at once and write every chamber's results (+ its .bta) and a summary
    with one row per animal.  Returns {"Chambers": [summary per chamber], "Output_folder", ...}."""
    from tracking.location import analyse_track
    if not chambers:
        raise RuntimeError("No chambers are set for this video.")
    setups = chamber_setups(setup, chambers)
    for s in setups:
        s["video_path"] = video_path

    def p(a, b):
        def f(x):
            if progress_callback is not None:
                progress_callback(min(1.0, a + (b - a) * max(0.0, min(1.0, x))))
        return f
    # 0-8 % backgrounds, 8-92 % tracking (as for one animal), then results and the video copies
    raws = track_chambers(video_path, setups, progress_callback=p(0.0, 1.0), frame_callback=frame_callback)
    summaries = []
    n = len(raws)
    for i, (raw, s) in enumerate(zip(raws, setups)):
        sm = analyse_track(raw, s, progress_callback=p(0.92 + 0.03 * i / n, 0.92 + 0.03 * (i + 1) / n))
        sm["Chamber"] = int(s["chamber"])
        sm["Animal_ID"] = s.get("chamber_id") or ""
        if s.get("subject"):
            sm.setdefault("Subject", s["subject"])
        summaries.append(sm)
        try:
            import json
            with open(os.path.join(raw["output_dir"], "zones.json"), "w", encoding="utf-8") as f:
                json.dump({"zones": {k: [list(map(float, q)) for q in v] for k, v in (s.get("roi_points") or {}).items()},
                           "masks": [[list(map(float, q)) for q in m] for m in (s.get("mask_points") or [])],
                           "test": s.get("test_type"), "warp_size": [s.get("warp_w"), s.get("warp_h")],
                           "chamber": int(s["chamber"]), "animal_id": s.get("chamber_id") or "",
                           "chamber_rect": s.get("chamber_rect"),
                           "own_settings": {k: s.get(k) for k in OVERRIDE_KEYS
                                            if k in (chambers[i].get("overrides") or {})}}, f, indent=1)
        except Exception:
            pass
    if save_project:
        copies = [None] * n
        px = int(setup.get("embed_video_px", 640) or 0) if setup.get("embed_video_px") is not None else 640
        if px > 0:
            try:
                from tracking.review_video import encode_copies
                base = setups[0]["chamber_base"]
                BM = np.asarray(base["matrix"], np.float32)
                BW, BH = int(base["warp_w"]), int(base["warp_h"])
                ident = (np.allclose(BM, np.eye(3), atol=1e-9) and (BW, BH) == (raws[0]["frame_w"], raws[0]["frame_h"]))
                warp = None if ident else (lambda f: cv2.warpPerspective(f, BM, (BW, BH)))
                specs = [{"first": r["frames"][0], "last": r["frames"][-1], "rect": r["chamber_rect"]} for r in raws]
                copies = encode_copies(video_path, specs, warp=warp, max_side=px, fps=float(raws[0]["fps"]),
                                       progress=p(0.95, 0.99))
            except Exception as exc:
                print(f"Video copies skipped: {exc}")
                copies = [None] * n
        from tracking.project_file import save_new_analysis
        for i, (raw, s, sm) in enumerate(zip(raws, setups, summaries)):
            try:
                sm["Project_file"] = save_new_analysis(raw, s, sm, copy=copies[i])
            except Exception as exc:
                print(f"BT analysis file not saved (chamber {i + 1}): {exc}")
                if copies[i]:
                    import shutil
                    shutil.rmtree(copies[i].get("dir") or "", ignore_errors=True)
    folder = results_root(video_path, setup)
    files = write_chamber_summary(video_path, summaries, folder) or {}
    if progress_callback is not None:
        progress_callback(1.0)
    q = [s.get("Tracking_quality_percent") for s in summaries if isinstance(s.get("Tracking_quality_percent"), (int, float))]
    return {"Chambers": summaries, "Output_folder": folder, "Video_file": video_path,
            "Video_name": os.path.basename(video_path), "Chamber_summary_xlsx": files.get("xlsx"),
            "Chamber_summary_csv": files.get("csv"), "Chamber_count": n,
            "Tracking_quality_percent": float(np.mean(q)) if q else None,
            "Project_file": summaries[0].get("Project_file") if summaries else None}


# --------------------------------------------------------------------------------------------- test detection
def preview_chambers(video_path, setups, progress=None):
    """Background model of every chamber (as 'Test detection' uses it).  Returns a list of
    {"setup", "bgm", "exclusion", "rect", "start_frame"} (worker thread, no Qt)."""
    from tracking.location import make_tracking_engine, _is_identity_warp
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base = setups[0]["chamber_base"]
        BM = np.asarray(base["matrix"], np.float32)
        BW, BH = int(base["warp_w"]), int(base["warp_h"])
        ident = _is_identity_warp(BM, fw, fh, BW, BH)
        out = []
        n = len(setups)
        for i, s0 in enumerate(setups):
            s = dict(s0)
            s["_fps"] = fps
            s["background_samples"] = min(60, int(s.get("background_samples", 100)))
            c = _prep_chamber(s, fps, total, total / fps if fps else 0.0)
            x0, y0, x1, y1 = [int(v) for v in s["chamber_rect"]]

            def read(idx, r=(x0, y0, x1, y1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, f = cap.read()
                if not ok:
                    return None
                if not ident:
                    f = cv2.warpPerspective(f, BM, (BW, BH))
                return np.ascontiguousarray(f[r[1]:r[3], r[0]:r[2]])
            _tr, bgm = make_tracking_engine(s, c["w"], c["h"], read, c["start_frame"], c["end_frame"], c["exclusion"],
                                            progress=(lambda f, i=i: progress((i + f) / n)) if progress else None)
            out.append({"setup": s, "bgm": bgm, "exclusion": c["exclusion"], "rect": (x0, y0, x1, y1),
                        "roi_points": c["roi_points"], "start_frame": c["start_frame"],
                        "label": chamber_label(s["chamber"], s.get("chamber_id")),
                        "tag": short_label(s["chamber"], s.get("chamber_id"))})
        return out
    finally:
        cap.release()
