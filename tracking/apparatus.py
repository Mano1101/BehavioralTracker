"""Automatic apparatus / zone detection (BehavioralTracker 1.0.0).

Two entry points used by the Video-Setup page:

* auto_detect_zones(background_bgr, test_key)
      Finds the apparatus in an (animal-free) background image and returns
      named zones fitted to it: EPM / Y-maze / T-maze are fitted with the
      parametric maze templates (centre + arms, any rotation), open field /
      NOR / light-dark / 3-chamber are fitted to the arena rectangle (or
      circle) and split the usual way.  NOR also finds the two objects.

* zones_from_outline(outline_pts, w, h, test_key)
      The user traced the apparatus outline by hand; the same fitting is run
      on the traced shape (mazes), or the shape is split into its natural
      arms with the skeleton method (anything else).

Every function returns plain {name: [(x, y), ...]} dictionaries in the
warped-video pixel space, i.e. exactly what the tracking pipeline consumes.
"""
import math

import cv2
import numpy as np

MAZES = ("epm", "y_maze", "t_maze")
ARENAS = ("open_field", "nor", "light_dark", "social", "custom")


# ----------------------------------------------------------------------------- geometry helpers
def _rect(cx, cy, length, width, angle_deg):
    t = math.radians(angle_deg)
    dx, dy = math.cos(t), math.sin(t)
    nx, ny = -dy, dx
    hl, hw = length / 2.0, width / 2.0
    return [(cx + dx * hl + nx * hw, cy + dy * hl + ny * hw), (cx + dx * hl - nx * hw, cy + dy * hl - ny * hw),
            (cx - dx * hl - nx * hw, cy - dy * hl - ny * hw), (cx - dx * hl + nx * hw, cy - dy * hl + ny * hw)]


def _circle(cx, cy, r, n=24):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def _triangle(cx, cy, side, rot):
    """Equilateral triangle centred at (cx, cy) whose three sides face the
    directions rot, rot+120, rot+240 (the Y-maze arms)."""
    R = side / math.sqrt(3.0)
    return [(cx + R * math.cos(math.radians(rot + 60.0 + 120.0 * k)),
             cy + R * math.sin(math.radians(rot + 60.0 + 120.0 * k))) for k in range(3)]


def _arm(cx, cy, hub_r, length, width, angle_deg, gap=2.0):
    t = math.radians(angle_deg)
    mid = hub_r + gap + length / 2.0
    return _rect(cx + math.cos(t) * mid, cy + math.sin(t) * mid, length, width, angle_deg)


def _mask_of(polys, w, h):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.array(p)).astype(np.int32)], 255)
    return m


def polygon_area(pts):
    a = np.array(pts, float)
    return 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1)))


def _pts(arr):
    return [(float(x), float(y)) for x, y in arr]


# ----------------------------------------------------------------------------- maze generators (centre given)
def maze_zones(kind, cx, cy, rot, arm_len, arm_w, hub, goal_len=None):
    """Named zone polygons of a maze centred at (cx, cy), first arm at `rot` degrees."""
    if kind == "epm":
        return {"Center": _rect(cx, cy, hub, hub, rot),
                "Open Arm 1": _arm(cx, cy, hub / 2.0, arm_len, arm_w, rot),
                "Open Arm 2": _arm(cx, cy, hub / 2.0, arm_len, arm_w, rot + 180),
                "Closed Arm 1": _arm(cx, cy, hub / 2.0, arm_len, arm_w, rot + 90),
                "Closed Arm 2": _arm(cx, cy, hub / 2.0, arm_len, arm_w, rot + 270)}
    if kind == "y_maze":                       # triangular centre, hub = triangle side
        r_in = hub / (2.0 * math.sqrt(3.0))
        z = {"Center": _triangle(cx, cy, hub, rot)}
        for i, nm in enumerate(("Arm A", "Arm B", "Arm C")):
            z[nm] = _arm(cx, cy, r_in, arm_len, arm_w, rot + 120 * i)
        return z
    if kind == "t_maze":
        gl = goal_len if goal_len else arm_len
        return {"Center": _rect(cx, cy, hub, hub, rot),
                "Stem": _arm(cx, cy, hub / 2.0, arm_len, arm_w, rot),
                "Goal Arm Left": _arm(cx, cy, hub / 2.0, gl, arm_w, rot + 90),
                "Goal Arm Right": _arm(cx, cy, hub / 2.0, gl, arm_w, rot - 90)}
    raise ValueError(kind)


# ----------------------------------------------------------------------------- segmentation
def _fill_holes(m):
    h, w = m.shape
    ff = m.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for sx, sy in seeds:
        if ff[sy, sx] == 0:
            cv2.floodFill(ff, mask, (sx, sy), 128)
    return np.where(ff == 128, 0, 255).astype(np.uint8)


def segment_apparatus(bg_bgr, kind=None):
    """Binary mask (uint8 0/255) of the apparatus floor in a background image.

    Both Otsu classes are tried; the connected component that best looks
    like 'the apparatus' (large, central, not hugging the image border,
    shape matching the test type) wins."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if bg_bgr.ndim == 3 else bg_bgr.copy()
    h, w = g.shape
    g = cv2.GaussianBlur(g, (5, 5), 0)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best, best_s = None, -1.0
    k = max(3, int(round(min(h, w) * 0.012))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    for t_off in (0, -12, 12):
        for pol in (0, 1):
            m = ((g > otsu + t_off) if pol == 0 else (g <= otsu + t_off)).astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
            n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
            for i in range(1, n):
                a = stats[i, cv2.CC_STAT_AREA] / float(h * w)
                if a < 0.03 or a > 0.97:
                    continue
                comp = (lab == i).astype(np.uint8) * 255
                comp = _fill_holes(comp)
                cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                c = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(c)
                hull = cv2.contourArea(cv2.convexHull(c))
                solidity = area / hull if hull > 0 else 0
                x, y, bw, bh = cv2.boundingRect(c)
                touch = ((x <= 1) + (y <= 1) + (x + bw >= w - 1) + (y + bh >= h - 1)) / 4.0
                cx, cy = cents[i]
                centr = 1.0 - min(1.0, math.hypot(cx - w / 2, cy - h / 2) / (0.5 * math.hypot(w, h)))
                s = (0.5 + a) * (0.4 + centr) * (1.0 - 0.6 * touch)
                if kind in MAZES:
                    s *= 1.6 if solidity < 0.8 else 0.6
                elif kind in ARENAS:
                    s *= 1.4 if solidity > 0.85 else 0.7
                if s > best_s:
                    best_s, best = s, comp
    return best


# ----------------------------------------------------------------------------- maze fitting
def _radial_extent(mask, cx, cy, n=360):
    h, w = mask.shape
    rmax = int(math.hypot(w, h))
    ext = np.zeros(n)
    for i in range(n):
        t = 2 * math.pi * i / n
        dx, dy = math.cos(t), math.sin(t)
        r_last = 0
        miss = 0
        for r in range(0, rmax, 2):
            x, y = int(round(cx + dx * r)), int(round(cy + dy * r))
            if x < 0 or y < 0 or x >= w or y >= h:
                break
            if mask[y, x]:
                r_last = r
                miss = 0
            else:
                miss += 1
                if miss > 6 and r_last > 0:
                    break
        ext[i] = r_last
    return ext


def _hub_center(mask):
    """Centre of a maze = point of maximum 'multi-directional' extent; start
    from the distance-transform maximum near the centroid."""
    m = (mask > 0).astype(np.uint8)
    mom = cv2.moments(m, binaryImage=True)
    if mom["m00"] == 0:
        return None
    cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
    dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    h, w = m.shape
    rad = int(0.15 * min(h, w))
    x0, x1 = max(0, int(cx) - rad), min(w, int(cx) + rad)
    y0, y1 = max(0, int(cy) - rad), min(h, int(cy) + rad)
    sub = dt[y0:y1, x0:x1]
    if sub.size == 0 or sub.max() <= 0:
        return cx, cy
    yy, xx = np.unravel_index(np.argmax(sub), sub.shape)
    return float(x0 + xx), float(y0 + yy)


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    uni = np.logical_or(a, b).sum()
    return inter / uni if uni else 0.0


def _maze_init(m, kind, cx, cy, cand_rots_out=None):
    """Initial maze parameters for a centre guess (radial-extent harmonics)."""
    h, w = m.shape
    ext = _radial_extent(m, cx, cy, n=180)
    ang = np.deg2rad(np.arange(0, 360, 2))
    n_arms = {"epm": 4, "y_maze": 3, "t_maze": 3}[kind]
    if kind in ("epm", "y_maze"):
        z = np.sum(ext * np.exp(1j * n_arms * ang))
        rot = (math.degrees(np.angle(z)) / n_arms) % (360.0 / n_arms)
        rots = [rot + k * 360.0 / n_arms for k in range(n_arms)]
        steps = [360.0 / n_arms] * n_arms
    else:
        z = np.sum(ext * np.exp(2j * ang))
        goal_axis = (math.degrees(np.angle(z)) / 2.0) % 180.0
        rots = [goal_axis + 90, goal_axis - 90]
        steps = [90.0] * 3
    lengths = []
    for k in range(n_arms):
        a = rots[0] + sum(steps[:k])
        lengths.append(ext[int(round(a / 2.0)) % 180])
    dt = cv2.distanceTransform(np.pad((m > 0).astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    vals = []
    for k in range(n_arms):
        a = math.radians(rots[0] + sum(steps[:k]))
        for f in (0.45, 0.6, 0.75):
            r = f * max(lengths[k], 1)
            x, y = int(round(cx + math.cos(a) * r)), int(round(cy + math.sin(a) * r))
            if 0 <= x < w and 0 <= y < h and dt[y, x] > 0:
                vals.append(dt[y, x])
    arm_w = 2.0 * float(np.median(vals)) if vals else 0.1 * min(h, w)
    reach = float(np.median(lengths)) if lengths else 0.35 * min(h, w)
    hub = arm_w                                   # square side (EPM / T) or triangle side (Y)
    arm_len = max(6.0, reach - (hub / (2.0 * math.sqrt(3.0)) if kind == "y_maze" else hub / 2.0))
    out = []
    for r0 in rots:
        p = {"cx": cx, "cy": cy, "rot": r0 % 360, "arm_len": arm_len, "arm_w": max(3.0, arm_w), "hub": max(3.0, hub)}
        if kind == "t_maze":
            p["goal_len"] = arm_len
        out.append(p)
    return out


def fit_maze(mask, kind):
    """Fit a parametric maze to a binary apparatus mask (any rotation, any
    position).  Works on a down-scaled copy, tries several centre guesses and
    refines the best ones by coordinate descent on the overlap (IoU).
    Returns (zones, params, iou) in full-resolution pixels."""
    H, W = mask.shape
    f = min(1.0, 360.0 / max(H, W))
    m = cv2.resize((mask > 0).astype(np.uint8), (max(8, int(W * f)), max(8, int(H * f))),
                   interpolation=cv2.INTER_NEAREST)
    h, w = m.shape
    target = m > 0
    if target.sum() < 20:
        return None, None, 0.0

    def zones_for(p, scale=1.0):
        q = {k: (v / scale if k not in ("rot",) else v) for k, v in p.items()}
        return maze_zones(kind, q["cx"], q["cy"], q["rot"], q["arm_len"], q["arm_w"], q["hub"], q.get("goal_len"))

    def score(p):
        vals = [v for k, v in p.items() if k != "rot"]
        if not all(np.isfinite(v) for v in vals) or max(vals) > 20 * max(w, h):
            return 0.0
        mm = _mask_of(zones_for(p).values(), w, h) > 0
        return _iou(mm, target)

    # centre candidates: distance-transform maximum, centroid, and a grid around them
    cands = []
    c0 = _hub_center(m)
    if c0 is not None:
        cands.append(c0)
    ys, xs = np.nonzero(target)
    cands.append((float(xs.mean()), float(ys.mean())))
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    for gy in np.linspace(y0, y1, 7)[1:-1]:
        for gx in np.linspace(x0, x1, 7)[1:-1]:
            if target[int(gy), int(gx)]:
                cands.append((float(gx), float(gy)))
    starts = []
    for cx, cy in cands:
        for p in _maze_init(m, kind, cx, cy):
            starts.append((score(p), p))
    starts.sort(key=lambda t: -t[0])
    best_p, best_s = None, -1.0
    for s0, p in starts[:4]:
        steps = {"cx": 0.03 * w, "cy": 0.03 * h, "rot": 5.0, "arm_len": 0.1 * p["arm_len"],
                 "arm_w": 0.15 * p["arm_w"], "hub": 0.15 * p["hub"]}
        if kind == "t_maze":
            steps["goal_len"] = 0.1 * p["arm_len"]
        cur, cur_s = dict(p), s0
        for _ in range(7):
            improved = False
            for key in list(steps):
                for sgn in (1, -1):
                    q = dict(cur)
                    q[key] = cur[key] + sgn * steps[key]
                    if key in ("arm_len", "arm_w", "hub", "goal_len") and q[key] <= 2:
                        continue
                    sq = score(q)
                    if sq > cur_s + 1e-4:
                        cur, cur_s, improved = q, sq, True
            if not improved:
                steps = {k: v * 0.5 for k, v in steps.items()}
        if cur_s > best_s:
            best_p, best_s = cur, cur_s
    if best_p is None:
        return None, None, 0.0
    full = {k: (v / f if k != "rot" else v) for k, v in best_p.items()}
    zones = maze_zones(kind, full["cx"], full["cy"], full["rot"], full["arm_len"], full["arm_w"], full["hub"],
                       full.get("goal_len"))
    return zones, full, best_s


def segment_candidates(bg_bgr, kind=None):
    """Several plausible apparatus masks (different thresholds / polarities)."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if bg_bgr.ndim == 3 else bg_bgr.copy()
    h, w = g.shape
    g = cv2.GaussianBlur(g, (5, 5), 0)
    otsu, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = max(3, int(round(min(h, w) * 0.008))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    hi = float(np.percentile(g, 99))
    lo = float(np.percentile(g, 1))
    thresholds = sorted(set([float(otsu)] + [otsu + d for d in (-25, 20, 40)] +
                            [otsu + (hi - otsu) * f for f in (0.3, 0.5, 0.7)] +
                            [otsu - (otsu - lo) * f for f in (0.3, 0.5)]))
    out = []
    for t in thresholds:
        for pol in (0, 1):
            mm = ((g > t) if pol == 0 else (g <= t)).astype(np.uint8) * 255
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, ker)
            n, lab, stats, _c = cv2.connectedComponentsWithStats(mm, 8)
            if n <= 1:
                continue
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            a = stats[i, cv2.CC_STAT_AREA] / float(h * w)
            if 0.02 <= a <= 0.9:
                filled = _fill_holes((lab == i).astype(np.uint8) * 255)
                if 0.02 <= (filled > 0).mean() <= 0.85:        # the surroundings filled in = not an apparatus
                    out.append(filled)
    return out


def _arm_axis(pts):
    """(centre, unit vector along the arm, length, width) of a 4-point arm rectangle."""
    a = np.asarray(pts, float)
    c = a.mean(axis=0)
    e1, e2 = a[1] - a[0], a[2] - a[1]
    l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
    if l1 >= l2:
        return c, e1 / max(l1, 1e-6), l1, l2
    return c, e2 / max(l2, 1e-6), l2, l1


def epm_wall_score(bg_bgr, zones):
    """Which arm pair has walls?  Perpendicular intensity profiles: next to an
    OPEN arm the floor drops straight to the surroundings; next to a CLOSED arm
    there is a band (the wall) that differs from the far surroundings.
    Returns (current_closed_labels_are_right: bool, confidence_ratio)."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) if bg_bgr.ndim == 3 else bg_bgr.astype(np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    h, w = g.shape

    def sample(x, y):
        xi, yi = int(round(x)), int(round(y))
        return float(g[yi, xi]) if 0 <= xi < w and 0 <= yi < h else np.nan

    def nm(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    def wall_score(names):
        scores = []
        for n in names:
            if n not in zones or len(zones[n]) != 4:
                continue
            c, u, L, W = _arm_axis(zones[n])
            v = np.array([-u[1], u[0]])
            for t in np.linspace(-0.3, 0.4, 8):          # along the arm, away from the hub end
                p0 = c + u * t * L
                for side in (1, -1):
                    floor = nm([sample(*(p0 + side * v * W * f)) for f in (0.0, 0.15, 0.3)])
                    band = nm([sample(*(p0 + side * v * W * f)) for f in (0.62, 0.72, 0.82)])
                    far = nm([sample(*(p0 + side * v * W * f)) for f in (1.3, 1.5, 1.7)])
                    if np.isfinite(floor) and np.isfinite(band) and np.isfinite(far):
                        scores.append(abs(band - far) / (abs(floor - far) + 8.0))
        return float(np.median(scores)) if scores else 0.0
    s_closed = wall_score(["Closed Arm 1", "Closed Arm 2"])
    s_open = wall_score(["Open Arm 1", "Open Arm 2"])
    ratio = (max(s_open, s_closed) + 0.02) / (min(s_open, s_closed) + 0.02)
    return s_closed >= s_open, ratio


def swap_epm_arms(zones):
    """Open <-> closed relabelling (the geometry stays)."""
    out = {}
    for k, v in zones.items():
        if k.startswith("Open Arm"):
            out[k.replace("Open", "Closed")] = v
        elif k.startswith("Closed Arm"):
            out[k.replace("Closed", "Open")] = v
        else:
            out[k] = v
    order = ["Center", "Open Arm 1", "Open Arm 2", "Closed Arm 1", "Closed Arm 2"]
    return {k: out[k] for k in order if k in out} | {k: v for k, v in out.items() if k not in order}


# ----------------------------------------------------------------------------- arena fitting
def _norm_rect(rx, ry, rw, rh, ang):
    """Keep the rotation in (-45, 45] so 'top' stays on top for square-ish arenas."""
    while ang > 45:
        ang -= 90
        rw, rh = rh, rw
    while ang <= -45:
        ang += 90
        rw, rh = rh, rw
    return rx, ry, rw, rh, ang


def find_arena_quad(bg_bgr, min_frac=0.2, prefer="inner"):
    """Convex quadrilateral outlined by edges (arena walls).  Works even when
    the floor has two very different brightnesses (light/dark box).
    prefer='outer' -> the whole box (outer walls); 'inner' -> the floor
    inside the walls (smallest quad >= 60 % of the largest one)."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if bg_bgr.ndim == 3 else bg_bgr
    h, w = g.shape
    g = cv2.GaussianBlur(g, (5, 5), 0)
    # absolute (not brightness-relative) thresholds: a background is a median of
    # many frames, so it is nearly noise-free and weak walls in a dark
    # compartment must still count
    edges = cv2.Canny(g, 12, 36)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    quads = []
    for mode in (cv2.RETR_EXTERNAL, cv2.RETR_LIST):
        cnts, _ = cv2.findContours(edges, mode, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_frac * w * h or a > 0.995 * w * h:
                continue
            per = cv2.arcLength(c, True)
            for eps in (0.02, 0.035):
                ap = cv2.approxPolyDP(c, eps * per, True)
                if len(ap) == 4 and cv2.isContourConvex(ap):
                    quads.append((cv2.contourArea(ap), ap.reshape(-1, 2)))
                    break
    if not quads:
        return None
    quads.sort(key=lambda q: -q[0])
    if prefer == "outer":
        chosen = quads[0][1]
    else:
        big = quads[0][0]
        chosen = min((q for q in quads if q[0] >= 0.6 * big), key=lambda q: q[0])[1]
    rect = cv2.minAreaRect(chosen.astype(np.float32))
    (rx, ry), (rw, rh), ang = rect
    shrink = 0.985 if prefer != "outer" else 0.97
    return _norm_rect(rx, ry, rw * shrink, rh * shrink, ang)


def fit_arena(mask, bg_bgr=None, kind=None):
    """('rect', box_pts, (cx, cy, w, h, angle)) or ('circle', pts, (cx, cy, r))."""
    if mask is not None:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    else:
        cnts = []
    c = max(cnts, key=cv2.contourArea) if cnts else None
    if c is not None:
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        circ = 4 * math.pi * area / (per * per) if per > 0 else 0
        (ccx, ccy), cr = cv2.minEnclosingCircle(c)
        rect = cv2.minAreaRect(c)
        rect_area = rect[1][0] * rect[1][1]
        if circ > 0.84 and area / (math.pi * cr * cr) > 0.82 and area / max(rect_area, 1) < 0.86:
            return "circle", _circle(ccx, ccy, cr * 0.98, 48), (ccx, ccy, cr * 0.98)
    quad = (find_arena_quad(bg_bgr, prefer="outer" if kind in ("light_dark", "social") else "inner")
            if bg_bgr is not None else None)
    if quad is not None:
        cand = quad
        if c is not None and kind not in ("light_dark", "social"):
            (rx, ry), (rw, rh), ang = cv2.minAreaRect(c)
            seg = _norm_rect(rx, ry, rw, rh, ang)
            # prefer the edge quad unless the intensity blob is clearly bigger
            if seg[2] * seg[3] > 1.25 * quad[2] * quad[3]:
                cand = seg
        params = cand
    elif c is not None:
        (rx, ry), (rw, rh), ang = cv2.minAreaRect(c)
        params = _norm_rect(rx, ry, rw, rh, ang)
    else:
        return None
    box = _rect(params[0], params[1], params[2], params[3], params[4])
    return "rect", box, params


def _rect_frame(params):
    """Map canonical arena coords (u in [0,W], v in [0,H]) to image coords."""
    cx, cy, rw, rh, ang = params
    t = math.radians(ang)
    ux, uy = math.cos(t), math.sin(t)
    vx, vy = -uy, ux

    def to_img(u, v):
        du, dv = u - rw / 2.0, v - rh / 2.0
        return (cx + ux * du + vx * dv, cy + uy * du + vy * dv)
    return to_img, rw, rh


def _profile_along(bg_gray, params, n=200):
    to_img, rw, rh = _rect_frame(params)
    prof = []
    for i in range(n):
        u = (i + 0.5) / n * rw
        vals = []
        for j in range(15):
            v = (0.15 + 0.7 * j / 14.0) * rh
            x, y = to_img(u, v)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < bg_gray.shape[0] and 0 <= xi < bg_gray.shape[1]:
                vals.append(float(bg_gray[yi, xi]))
        prof.append(np.median(vals) if vals else 0.0)
    return np.array(prof)


def arena_zones(kind, shape, params, bg_bgr=None, center_fraction=0.5, ppc=None):
    """Zones for an open-field-like test fitted to a rect/circle arena."""
    g = None
    if bg_bgr is not None:
        g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if bg_bgr.ndim == 3 else bg_bgr
    zones = {}
    if shape == "circle":
        cx, cy, r = params
        if kind in ("open_field", "nor", "custom"):
            zones["Center"] = _circle(cx, cy, r * center_fraction, 36)
            zones["Periphery"] = keyhole_ring_circle(cx, cy, r, r * center_fraction + 2.0)
        else:
            zones["Arena"] = _circle(cx, cy, r, 48)
        if kind == "nor" and g is not None:
            zones.update(find_objects(g, _mask_of([_circle(cx, cy, r, 48)], g.shape[1], g.shape[0]), ppc))
        return zones
    to_img, rw, rh = _rect_frame(params)

    def poly(u0, v0, u1, v1):
        return [to_img(u0, v0), to_img(u1, v0), to_img(u1, v1), to_img(u0, v1)]
    gap = 2.0
    if kind in ("open_field", "nor", "custom"):
        f = center_fraction
        u0, u1 = rw * (1 - f) / 2, rw * (1 + f) / 2
        v0, v1 = rh * (1 - f) / 2, rh * (1 + f) / 2
        zones["Center"] = poly(u0, v0, u1, v1)
        zones["Periphery Top"] = poly(0, 0, rw, v0 - gap)
        zones["Periphery Bottom"] = poly(0, v1 + gap, rw, rh)
        zones["Periphery Left"] = poly(0, v0, u0 - gap, v1)
        zones["Periphery Right"] = poly(u1 + gap, v0, rw, v1)
        if kind == "nor" and g is not None:
            zones.update(find_objects(g, _mask_of([poly(0, 0, rw, rh)], g.shape[1], g.shape[0]), ppc))
    elif kind == "light_dark":
        cx, cy, rw0, rh0, ang0 = params
        best = None
        for p2 in ((cx, cy, rw0, rh0, ang0), (cx, cy, rh0, rw0, ang0 + 90)):
            if g is None:
                break
            prof = _profile_along(g, p2)
            sm = np.convolve(prof, np.ones(9) / 9, mode="same")
            grad = np.abs(np.diff(sm))
            lo, hi = int(0.2 * len(grad)), int(0.8 * len(grad))
            if hi <= lo:
                continue
            k = lo + int(np.argmax(grad[lo:hi]))
            step = abs(float(np.mean(sm[max(0, k - 30):k])) - float(np.mean(sm[k + 1:k + 31])))
            if best is None or step > best[0]:
                best = (step, p2, k, prof)
        if best is None:
            to_img, rw, rh = _rect_frame(params)
            split, left_bright = rw / 2.0, True
        else:
            _step, p2, k, prof = best
            to_img, rw, rh = _rect_frame(p2)
            split = (k + 1) / len(prof) * rw
            ks = max(1, min(len(prof) - 1, k + 1))
            left_bright = float(np.mean(prof[:ks])) >= float(np.mean(prof[ks:]))

        def poly(u0, v0, u1, v1):
            return [to_img(u0, v0), to_img(u1, v0), to_img(u1, v1), to_img(u0, v1)]
        a = poly(0, 0, split - gap, rh)
        b = poly(split + gap, 0, rw, rh)
        zones["Light"], zones["Dark"] = (a, b) if left_bright else (b, a)
    elif kind == "social":
        cx, cy, rw0, rh0, ang0 = params
        if rh0 > rw0:                                    # chambers run along the long side
            params = (cx, cy, rh0, rw0, ang0 + 90)
        to_img, rw, rh = _rect_frame(params)

        def poly(u0, v0, u1, v1):
            return [to_img(u0, v0), to_img(u1, v0), to_img(u1, v1), to_img(u0, v1)]
        cuts = [rw / 3.0, 2 * rw / 3.0]
        if g is not None:
            prof = _profile_along(g, params)
            sm = np.convolve(prof, np.ones(5) / 5, mode="same")
            dev = np.abs(sm - np.median(sm))
            n = len(sm)
            found = []
            for lo, hi in ((int(0.2 * n), int(0.45 * n)), (int(0.55 * n), int(0.8 * n))):
                seg = dev[lo:hi]
                if seg.size and seg.max() > 12:
                    found.append((lo + int(np.argmax(seg)) + 0.5) / n * rw)
            if len(found) == 2:
                cuts = found
        zones["Chamber 1"] = poly(0, 0, cuts[0] - gap * 3, rh)
        zones["Center"] = poly(cuts[0] + gap * 3, 0, cuts[1] - gap * 3, rh)
        zones["Chamber 2"] = poly(cuts[1] + gap * 3, 0, rw, rh)
    else:
        zones["Arena"] = poly(0, 0, rw, rh)
    return zones


def keyhole_ring_circle(cx, cy, r_out, r_in, n=48):
    """Single polygon covering a ring (outer circle minus inner circle) with a
    zero-width bridge -- works with fillPoly-based zone masks."""
    outer = _circle(cx, cy, r_out, n)
    inner = _circle(cx, cy, r_in, n)
    return outer + [outer[0], inner[0]] + inner[:0:-1] + [inner[0], outer[0]]


def find_objects(gray, arena_mask, ppc=None, max_objects=2):
    """NOR: the two objects = the two largest compact blobs that differ from
    the floor inside the arena.  Zones are drawn ~2 cm around each object
    (exploration zone)."""
    m = arena_mask > 0
    if m.sum() < 100:
        return {}
    k = max(3, int(0.04 * math.sqrt(m.sum()))) | 1
    inner = cv2.erode(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    floor = float(np.median(gray[inner]))
    dev = np.abs(gray.astype(np.float32) - floor)
    thr = max(25.0, 3.0 * float(np.median(np.abs(gray[inner].astype(np.float32) - floor))) * 1.4826)
    cand = ((dev > thr) & inner).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, 8)
    blobs = []
    tot = float(m.sum())
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if 0.002 * tot <= a <= 0.08 * tot:
            bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if max(bw, bh) / max(1, min(bw, bh)) < 3:
                blobs.append((a, cents[i], max(bw, bh) / 2.0))
    blobs.sort(key=lambda b: -b[0])
    blobs = sorted(blobs[:max_objects], key=lambda b: (b[1][0], b[1][1]))
    margin = (2.0 * ppc) if ppc else 0.04 * math.sqrt(tot)
    out = {}
    for i, (_a, (x, y), r) in enumerate(blobs):
        out[f"Object {'AB'[i] if i < 2 else i + 1}"] = _circle(x, y, r + margin, 32)
    return out


# ----------------------------------------------------------------------------- public API
def auto_detect_zones(background_bgr, test_key, ppc=None, center_fraction=0.5):
    """Returns (zones, outline, message).  zones == {} when nothing usable was found."""
    kind = test_key if test_key in MAZES + ARENAS else "custom"
    mask = segment_apparatus(background_bgr, kind)
    if mask is None and kind in MAZES:
        return {}, None, "Could not find the apparatus in the image (low contrast between floor and surroundings)."
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if mask is not None else []
    outline = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        eps = 0.004 * cv2.arcLength(c, True)
        outline = _pts(cv2.approxPolyDP(c, eps, True).reshape(-1, 2))
    if kind in MAZES:
        best = (None, None, -1.0, None)
        masks = segment_candidates(background_bgr, kind)
        if mask is not None:
            masks.insert(0, mask)
        Hh, Ww = background_bgr.shape[:2]
        for mk in masks:
            z_, p_, iou_ = fit_maze(mk, kind)
            if not z_:
                continue
            plausible = (p_["arm_w"] < 0.3 * min(Hh, Ww) and p_["hub"] < 0.4 * min(Hh, Ww)
                         and p_["arm_len"] > 1.2 * p_["arm_w"])
            if plausible and iou_ > best[2]:
                best = (z_, p_, iou_, mk)
        zones, params, iou, mask = best
        if not zones:
            return {}, outline, "Maze could not be fitted."
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            outline = _pts(cv2.approxPolyDP(c, 0.004 * cv2.arcLength(c, True), True).reshape(-1, 2))
        msg = f"Fitted {kind.replace('_', '-').upper()} template (overlap {iou * 100:.0f}%)."
        if kind == "epm":
            closed_ok, ratio = epm_wall_score(background_bgr, zones)
            if not closed_ok:
                zones = swap_epm_arms(zones)
            msg += (" Open/closed arms chosen from the wall edges" +
                    (" — please check the labels." if ratio < 1.15 else "."))
        if iou < 0.55:
            msg += " Low overlap: adjust the zones by hand."
        return zones, outline, msg
    fit = fit_arena(mask, background_bgr, kind)
    if fit is None:
        return {}, outline, "Arena outline not found."
    shape, pts, params = fit
    zones = arena_zones(kind, shape, params, background_bgr, center_fraction, ppc)
    msg = f"Found {'circular' if shape == 'circle' else 'rectangular'} arena."
    if kind == "nor":
        n_obj = sum(1 for k in zones if k.startswith("Object"))
        msg += f" {n_obj} object(s) found." if n_obj else " No objects found — add them with the circle tool."
    return zones, pts, msg


def zones_from_outline(outline_pts, w, h, test_key, background_bgr=None, ppc=None):
    """Zones from a hand-traced apparatus outline (or several outlines)."""
    from tracking.location import detect_apparatus_partitions
    shapes = outline_pts if outline_pts and isinstance(outline_pts[0][0], (list, tuple)) else [outline_pts]
    shapes = [s for s in shapes if len(s) >= 3]
    if not shapes:
        return {}, "Trace the apparatus outline first."
    kind = test_key
    mask = _mask_of(shapes, w, h)
    if kind in MAZES and len(shapes) == 1:
        zones, params, iou = fit_maze(mask, kind)
        if zones and iou > 0.5:
            msg = f"Traced outline fitted with the {kind.replace('_', '-')} template (overlap {iou * 100:.0f}%)."
            if kind == "epm" and background_bgr is not None:
                ok, ratio = epm_wall_score(background_bgr, zones)
                if not ok:
                    zones = swap_epm_arms(zones)
            return zones, msg
    if kind in ("open_field", "nor", "light_dark", "social") and len(shapes) == 1:
        fit = fit_arena(mask, None, kind)
        if fit is not None:
            shape, _pts_, params = fit
            return arena_zones(kind, shape, params, background_bgr, 0.5, ppc), "Arena zones fitted to the traced outline."
    parts, err = detect_apparatus_partitions(shapes, w, h)
    if err:
        return {}, err
    return parts, f"Outline split into {len(parts)} zone(s): {', '.join(parts)} — rename them as needed."


# ----------------------------------------------------------------------------- guided fit (user clicks)
GUIDED_POINTS = {
    "epm": (["Open arm 1", "Open arm 2", "Closed arm 1", "Closed arm 2"],
            ["Click the far END of an OPEN arm (1/4)",
             "Click the far END of the opposite OPEN arm (2/4)",
             "Click the far END of a CLOSED arm (3/4)",
             "Click the far END of the opposite CLOSED arm (4/4)"]),
    "y_maze": (["Arm A", "Arm B", "Arm C"],
               ["Click the far END of Arm A (1/3)", "Click the far END of Arm B (2/3)",
                "Click the far END of Arm C (3/3)"]),
    "t_maze": (["Stem", "Left goal", "Right goal"],
               ["Click the far END of the STEM / start arm (1/3)",
                "Click the far END of the LEFT goal arm (2/3)",
                "Click the far END of the RIGHT goal arm (3/3)"]),
    "rect": (["Corner 1", "Corner 2", "Corner 3", "Corner 4"],
             ["Click a corner of the arena FLOOR (1/4)", "Click the next corner of the floor (2/4)",
              "Click the next corner (3/4)", "Click the last corner (4/4)"]),
    "circle": (["Edge"],
               ["Click 3 or more points on the EDGE of the round arena floor, then press Enter / ✓ Finish"]),
}


def guided_spec(test_key, shape="rect"):
    """(number of clicks or None for 'any >= 3', point labels, prompts) for the guided fit."""
    if test_key in MAZES:
        labels, prompts = GUIDED_POINTS[test_key]
        return len(labels), labels, prompts
    if shape == "circle":
        labels, prompts = GUIDED_POINTS["circle"]
        return None, labels, prompts
    labels, prompts = GUIDED_POINTS["rect"]
    return 4, labels, prompts


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def _line_intersection(p1, d1, p2, d2):
    """Intersection of p1 + s d1 and p2 + t d2 (None when parallel)."""
    a = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]], float)
    if abs(np.linalg.det(a)) < 1e-6:
        return None
    s, _t = np.linalg.solve(a, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + s * np.asarray(d1, float)


def _fermat_point(pts, iters=200):
    """Point with the smallest summed distance to `pts` (Weiszfeld) -- the
    hub of a Y-maze whose arms are 120 deg apart, whatever the arm lengths."""
    p = np.asarray(pts, float)
    x = p.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(p - x, axis=1)
        if np.any(d < 1e-6):
            return p[int(np.argmin(d))]
        w = 1.0 / d
        nx = (p * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(nx - x) < 1e-4:
            return nx
        x = nx
    return x


def _gray_f(bg_bgr):
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if bg_bgr.ndim == 3 else bg_bgr
    return cv2.GaussianBlur(g.astype(np.float32), (5, 5), 0)


def _arm_samples(g, center, tip, fracs=np.linspace(0.35, 0.88, 8)):
    """Cross-sections of one arm: list of (width, mid-point) where the floor
    edges (first clear intensity step on both sides of the line centre->tip)
    were found."""
    h, w = g.shape
    c = np.asarray(center, float)
    t = np.asarray(tip, float)
    L = float(np.linalg.norm(t - c))
    out = []
    if L < 10:
        return out
    d = _unit(t - c)
    nrm = np.array([-d[1], d[0]])
    span = min(0.45 * L, 0.3 * min(h, w))
    s = np.arange(-span, span + 0.25, 0.5, dtype=np.float32)
    mid = int(np.argmin(np.abs(s)))
    for f in fracs:
        p0 = c + d * f * L
        xs = (p0[0] + nrm[0] * s).astype(np.float32).reshape(1, -1)
        ys = (p0[1] + nrm[1] * s).astype(np.float32).reshape(1, -1)
        prof = cv2.remap(g, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)[0]
        inside = (xs[0] >= 0) & (xs[0] <= w - 1) & (ys[0] >= 0) & (ys[0] <= h - 1)
        if not inside[max(0, mid - 3):mid + 4].all():
            continue
        floor = float(np.median(prof[max(0, mid - 3):mid + 4]))
        dev = np.abs(prof - floor)
        dev[~inside] = np.nan
        peak = np.nanmax(dev) if np.isfinite(dev).any() else 0.0
        if not np.isfinite(peak) or peak < 10:
            continue
        thr = max(8.0, 0.35 * peak)
        right = np.where(np.nan_to_num(dev[mid:], nan=0.0) > thr)[0]
        left = np.where(np.nan_to_num(dev[:mid + 1][::-1], nan=0.0) > thr)[0]
        if not len(right) or not len(left):
            continue
        # the edge sits half-way up the step
        kr = mid + int(right[0])
        while kr > mid and dev[kr - 1] > 0.5 * thr:
            kr -= 1
        kl = mid - int(left[0])
        while kl < mid and dev[kl + 1] > 0.5 * thr:
            kl += 1
        width = float(s[kr] - s[kl])
        out.append((width, p0 + nrm * 0.5 * float(s[kr] + s[kl])))
    return out


def _robust_median(ws):
    ws = np.asarray(ws, float)
    if len(ws) < 3:
        return None
    med = float(np.median(ws))
    good = ws[np.abs(ws - med) <= 0.35 * med]
    return float(np.median(good)) if len(good) >= 3 else med


def estimate_arm_width(bg_bgr, center, tips, prefer=None):
    """Width of maze arms measured on the image (cross-sections between 35 and
    88 % of each arm).  `prefer` = indices of arms to trust first (EPM: the
    open arms -- no walls, no wall shadows).  None when the image gives no
    consistent answer."""
    if bg_bgr is None:
        return None
    g = _gray_f(bg_bgr)
    per_arm = [[wd for wd, _m in _arm_samples(g, center, t)] for t in tips]
    if prefer:
        pref = _robust_median([x for i in prefer if i < len(per_arm) for x in per_arm[i]])
        if pref is not None:
            return pref
    arm_w = [x for x in (_robust_median(ws) for ws in per_arm) if x is not None]
    if not arm_w:
        return _robust_median([x for ws in per_arm for x in ws])
    return float(np.median(arm_w))


def _fit_line(points):
    """(point on line, unit direction) through 2-D points (total least squares)."""
    p = np.asarray(points, float)
    c = p.mean(axis=0)
    _u, _s, vt = np.linalg.svd(p - c)
    return c, _unit(vt[0])


def _refine_axes(bg_bgr, center, tips, groups, width_guess=None):
    """Snap the arm centre lines to the real arms (mid-points between the floor
    edges), so a click a few pixels off the arm's middle does not tilt or
    shift the zones.  groups = lists of arm indices lying on one straight
    line (EPM: [[0, 1], [2, 3]]).  Returns {group_index: (point, direction)}."""
    if bg_bgr is None:
        return {}
    g = _gray_f(bg_bgr)
    out = {}
    for gi, grp in enumerate(groups):
        mids, wds = [], []
        for i in grp:
            for wd, m in _arm_samples(g, center, tips[i]):
                wds.append(wd)
                mids.append(m)
        if len(mids) < 4:
            continue
        med = float(np.median(wds))
        if width_guess:
            # only cross-sections showing the full arm width locate the arm's middle (a wall
            # shadow on one side makes the section narrower and off-centre)
            med = float(width_guess)
            keep = [m for wd, m in zip(wds, mids) if abs(wd - med) <= 0.15 * med]
        else:
            keep = [m for wd, m in zip(wds, mids) if abs(wd - med) <= 0.35 * med]
        if len(keep) < 4:
            continue
        p, d = _fit_line(keep)
        # residuals must be small compared with the arm width, else keep the clicks
        res = np.abs((np.asarray(keep) - p) @ np.array([-d[1], d[0]]))
        if float(np.median(res)) > 0.25 * med:
            continue
        out[gi] = (p, d)
    return out


def _project(pt, line):
    p, d = line
    return p + d * float(np.dot(np.asarray(pt, float) - p, d))


def star_maze_zones(center, tips, width, names, center_name="Center", gap=1.5):
    """Zones of a maze whose arms run from `center` to the clicked `tips`
    (any angles).  The centre polygon is where the arm strips meet, every arm
    is a strip of `width` from there to its tip."""
    c = np.asarray(center, float)
    hw = width / 2.0
    dirs = [_unit(np.asarray(t, float) - c) for t in tips]
    lens = [float(np.linalg.norm(np.asarray(t, float) - c)) for t in tips]
    ang = [math.atan2(d[1], d[0]) for d in dirs]
    order = sorted(range(len(tips)), key=lambda i: ang[i])
    n = len(order)

    def n_next(d):                     # side of the arm facing the next arm (increasing angle)
        return np.array([-d[1], d[0]])

    def n_prev(d):
        return np.array([d[1], -d[0]])
    prev_pt, next_pt = {}, {}          # hub-end corners of each arm
    centre_poly = []
    for k in range(n):
        i, j = order[k], order[(k + 1) % n]
        di, dj = dirs[i], dirs[j]
        sweep = (ang[j] - ang[i]) % (2 * math.pi) if n > 1 else 2 * math.pi
        pi_ = c + n_next(di) * hw
        pj_ = c + n_prev(dj) * hw
        cross = _line_intersection(pi_, di, pj_, dj) if sweep < math.radians(170) else None
        if cross is not None and np.linalg.norm(cross - c) < 3.0 * width:
            next_pt[i] = cross
            prev_pt[j] = cross
            centre_poly.append(cross)
        else:                          # (nearly) straight continuation, e.g. the two goal arms of a T
            a = pi_ + di * hw
            b = pj_ + dj * hw
            next_pt[i] = a
            prev_pt[j] = b
            centre_poly.extend([a, b])
    zones = {center_name: _pts(np.asarray(centre_poly))}
    for i, name in enumerate(names):
        d = dirs[i]
        tip = c + d * lens[i]
        a = next_pt[i] + d * gap
        b = prev_pt[i] + d * gap
        if float(np.dot(tip - a, d)) <= 2 or float(np.dot(tip - b, d)) <= 2:
            continue                   # tip inside the centre: arm too short
        zones[name] = _pts([b, a, tip + n_next(d) * hw, tip + n_prev(d) * hw])
    return zones


def _order_quad(pts):
    """4 points -> convex order starting at the top-left corner (clockwise on screen)."""
    p = np.asarray(pts, float)
    c = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    start = int(np.argmin(p[:, 0] + p[:, 1]))
    return np.roll(p, -start, axis=0)


def _fit_circle(pts):
    p = np.asarray(pts, float)
    a = np.column_stack([p[:, 0], p[:, 1], np.ones(len(p))])
    b = -(p[:, 0] ** 2 + p[:, 1] ** 2)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = -sol[0] / 2.0, -sol[1] / 2.0
    r = math.sqrt(max(cx * cx + cy * cy - sol[2], 1e-6))
    return cx, cy, r


def zones_from_clicks(test_key, pts, background_bgr=None, ppc=None, shape="rect", center_fraction=0.5):
    """Guided fit: zones from a few clicked points (see guided_spec for the
    order).  Returns (zones, message)."""
    pts = [(float(x), float(y)) for x, y in pts]
    kind = test_key if test_key in MAZES + ARENAS else "custom"
    if kind in MAZES:
        need = len(GUIDED_POINTS[kind][0])
        if len(pts) < need:
            return {}, f"Click all {need} arm ends."
        tips = [np.asarray(p, float) for p in pts[:need]]

        def centre_of(tp):
            if kind == "epm":
                cc = _line_intersection(tp[0], tp[1] - tp[0], tp[2], tp[3] - tp[2])
                return cc if cc is not None else np.mean(tp, axis=0)
            if kind == "y_maze":
                return _fermat_point(tp)
            st, lf, rt = tp                                # T: stem, left, right
            u = _unit(rt - lf)
            return lf + u * float(np.dot(st - lf, u))
        c = centre_of(tips)
        names = {"epm": ["Open Arm 1", "Open Arm 2", "Closed Arm 1", "Closed Arm 2"],
                 "y_maze": ["Arm A", "Arm B", "Arm C"],
                 "t_maze": ["Stem", "Goal Arm Left", "Goal Arm Right"]}[kind]
        # snap the arm axes to the arms seen in the video (clicks need not be exact)
        groups = {"epm": [[0, 1], [2, 3]], "y_maze": [[0], [1], [2]], "t_maze": [[0], [1, 2]]}[kind]
        w0 = estimate_arm_width(background_bgr, c, tips, prefer=[0, 1] if kind == "epm" else None)
        axes = _refine_axes(background_bgr, c, tips, groups, width_guess=w0)
        if kind == "epm" and 0 in axes and 1 not in axes:
            # closed arms (walls, shadows) could not be measured: they are perpendicular to the
            # open arms on a plus maze -> keep only the position of the clicks, not their angle
            d_open = axes[0][1]
            d_cl = np.array([-d_open[1], d_open[0]])
            click_dir = _unit(tips[3] - tips[2])
            if abs(float(np.dot(click_dir, d_cl))) > math.cos(math.radians(15)):
                axes[1] = ((tips[2] + tips[3]) / 2.0, d_cl)
        if axes:
            line_of = {i: axes[gi] for gi, grp in enumerate(groups) if gi in axes for i in grp}
            if kind == "y_maze" and len(axes) == 3:
                # hub = point closest to the three refined arm lines
                a_m = np.zeros((2, 2))
                b_v = np.zeros(2)
                for i in range(3):
                    p_, d_ = line_of[i]
                    pr = np.eye(2) - np.outer(d_, d_)
                    a_m += pr
                    b_v += pr @ p_
                c_new = np.linalg.solve(a_m, b_v)
                tips = [_project(tips[i], line_of[i]) for i in range(3)]
                c = c_new
            else:
                tips = [_project(t, line_of[i]) if i in line_of else t for i, t in enumerate(tips)]
                if kind == "epm" and len(axes) == 2:
                    c2 = _line_intersection(axes[0][0], axes[0][1], axes[1][0], axes[1][1])
                    c = c2 if c2 is not None else centre_of(tips)
                elif kind == "t_maze" and 1 in axes:
                    c = _project(tips[0], axes[1])
                    if 0 in axes:
                        c2 = _line_intersection(axes[0][0], axes[0][1], axes[1][0], axes[1][1])
                        if c2 is not None:
                            c = c2
                else:
                    c = centre_of(tips)
        lens = [float(np.linalg.norm(t - c)) for t in tips]
        if min(lens) < 8:
            return {}, "Two clicks are too close to the centre — click the far ends of the arms."
        width = estimate_arm_width(background_bgr, c, tips, prefer=[0, 1] if kind == "epm" else None)
        how = "measured on the video"
        lmed = float(np.median(lens))
        if width is None or not (0.05 * lmed <= width <= 0.7 * lmed):
            width = 0.17 * lmed
            how = "estimated — adjust it if needed"
        zones = star_maze_zones(c, tips, width, names)
        order = ["Center"] + names
        zones = {k: zones[k] for k in order if k in zones}
        msg = f"Zones fitted to your clicks (arm width {width:.0f} px, {how})."
        if ppc:
            msg = msg[:-2] + f", ≈{width / ppc:.1f} cm)."
        return zones, msg
    # arenas
    if shape == "circle":
        if len(pts) < 3:
            return {}, "Click at least 3 points on the edge of the arena."
        cx, cy, r = _fit_circle(pts)
        return arena_zones(kind, "circle", (cx, cy, r), background_bgr, center_fraction, ppc), \
            f"Round arena fitted (radius {r:.0f} px)."
    if len(pts) < 4:
        return {}, "Click the 4 corners of the arena floor."
    q = _order_quad(pts[:4]).astype(np.float32)
    rw = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    rh = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if rw < 10 or rh < 10:
        return {}, "The corners are too close together."
    canon = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]])
    m_to_img = cv2.getPerspectiveTransform(canon, q)
    flat = None
    if background_bgr is not None:
        m_to_flat = cv2.getPerspectiveTransform(q, canon)
        flat = cv2.warpPerspective(background_bgr, m_to_flat, (int(round(rw)), int(round(rh))))
    msg = "Zones fitted to the 4 corners (perspective corrected)."
    if kind == "custom" and flat is not None:
        # no test chosen: a box with a strong light / dark split is a light/dark box
        if light_dark_step(flat, (rw / 2.0, rh / 2.0, rw, rh, 0.0)) >= 38:
            kind = "light_dark"
            msg = "Looks like a light/dark box — " + msg[0].lower() + msg[1:]
    zc = arena_zones(kind, "rect", (rw / 2.0, rh / 2.0, rw, rh, 0.0), flat, center_fraction, ppc)
    zones = {}
    for name, poly in zc.items():
        arr = cv2.perspectiveTransform(np.asarray(poly, np.float32).reshape(-1, 1, 2), m_to_img).reshape(-1, 2)
        zones[name] = _pts(arr)
    return zones, msg


def apparatus_shape(background_bgr):
    """'maze' (plus / Y / T shape), 'arena' (box or round floor) or None, from
    the solidity of the segmented apparatus -- used to warn when the selected
    test type does not match the video."""
    try:
        cands = segment_candidates(background_bgr, None)
        m0 = segment_apparatus(background_bgr, None)
        if m0 is not None:
            cands.insert(0, m0)
    except Exception:
        return None
    h, w = background_bgr.shape[:2]
    votes = []
    for m in cands[:4]:
        if m is None:
            continue
        cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        a = cv2.contourArea(c)
        if a < 0.03 * h * w or a > 0.92 * h * w:
            continue
        hull = cv2.contourArea(cv2.convexHull(c))
        votes.append(a / max(hull, 1.0))
    if not votes:
        return None
    sol = float(np.median(votes))
    if sol < 0.62:
        return "maze"
    if sol > 0.85:
        return "arena"
    return None


# ----------------------------------------------------------------------------- what kind of apparatus is this?
def light_dark_step(bg_bgr, params):
    """Brightness step (grey levels) between the two halves of a box, along its better axis: a light/dark
    box has one strong step (bright open part vs dark / covered part), an open field has none."""
    g = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY) if bg_bgr.ndim == 3 else bg_bgr
    cx, cy, rw0, rh0, ang0 = params
    best = 0.0
    for p2 in ((cx, cy, rw0, rh0, ang0), (cx, cy, rh0, rw0, ang0 + 90)):
        prof = _profile_along(g, p2)
        sm = np.convolve(prof, np.ones(9) / 9, mode="same")
        grad = np.abs(np.diff(sm))
        lo, hi = int(0.25 * len(grad)), int(0.75 * len(grad))
        if hi <= lo:
            continue
        k = lo + int(np.argmax(grad[lo:hi]))
        a, b = sm[max(5, k - 40):k - 3], sm[k + 4:min(len(sm) - 5, k + 44)]
        if a.size and b.size:
            best = max(best, abs(float(np.median(a)) - float(np.median(b))))
    return best


def guess_test(background_bgr):
    """Best guess of the test from the apparatus picture: 'epm', 'y_maze', 't_maze', 'light_dark',
    'social', 'open_field' -- or None when the picture says too little."""
    try:
        shape = apparatus_shape(background_bgr)
    except Exception:
        shape = None
    if shape == "maze":
        best, best_iou = None, 0.0
        for kind in MAZES:
            masks = list(segment_candidates(background_bgr, kind))
            m0 = segment_apparatus(background_bgr, kind)
            if m0 is not None:
                masks.insert(0, m0)
            for mk in masks[:3]:
                z, p, iou = fit_maze(mk, kind)
                if z and iou > best_iou:
                    best, best_iou = kind, iou
        return best if best_iou >= 0.5 else None
    mask = segment_apparatus(background_bgr, "light_dark")
    fit = fit_arena(mask, background_bgr, "light_dark")
    if fit is None:
        return None
    kind, _pts, params = fit
    if kind == "circle":
        return "open_field"
    if light_dark_step(background_bgr, params) >= 38:
        return "light_dark"
    g = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy, rw0, rh0, ang0 = params
    p = (cx, cy, rh0, rw0, ang0 + 90) if rh0 > rw0 else params
    sm = np.convolve(_profile_along(g, p), np.ones(5) / 5, mode="same")
    dev = np.abs(sm - np.median(sm))
    n = len(sm)
    peaks = [dev[int(a * n):int(b * n)].max() if int(b * n) > int(a * n) else 0 for a, b in ((0.2, 0.45), (0.55, 0.8))]
    if min(peaks) > 25 and dev[int(0.45 * n):int(0.55 * n)].max() < 0.6 * min(peaks):
        return "social"
    return "open_field"
