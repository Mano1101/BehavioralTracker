"""Head (nose), body centre and tail-base points of the tracked animal.

Per frame the tracker measures the two ends of the body along its long axis (tail removed) and three
clues about which end is the head:
  * tail      -- the thin tail sticks out of one end (strongest clue when the tail is visible)
  * shape     -- seen from above a mouse / rat is narrower at the snout than at the hips
  * movement  -- an animal walks forwards (used here, in the analysis, from the centre track)
Which end is the head is then decided for the WHOLE track at once (dynamic programming over the two
possible orientations): an end does not jump by a body length from one frame to the next, so the
choice stays consistent while the animal rests or turns on the spot, and is set right by the clues
whenever they are clear.
"""
import math

import cv2
import numpy as np

POINTS = ("centre", "head", "tail")
POINT_NAMES = {"centre": "Body centre", "head": "Head (nose)", "tail": "Tail base"}


def body_ends(full_mask, body_mask, x0=0, y0=0):
    """(ax, ay, bx, by, tail_ev, shape_ev) -- the two body ends (input pixel coords + offset) and the clues:
    tail_ev > 0: the tail sticks out at end A (so A is the rear); shape_ev > 0: end A is the narrower one
    (head-like).  None when the body is too small / round to have a direction."""
    ys, xs = np.nonzero(body_mask)
    if xs.size < 12:
        return None
    xs = xs.astype(np.float32)
    ys = ys.astype(np.float32)
    mx, my = float(xs.mean()), float(ys.mean())
    dx, dy = xs - mx, ys - my
    cxx, cyy, cxy = float((dx * dx).mean()), float((dy * dy).mean()), float((dx * dy).mean())
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = math.sqrt(max(0.0, tr * tr / 4 - det))
    l1, l2 = tr / 2 + disc, max(1e-6, tr / 2 - disc)
    if l1 / l2 < 1.4:                                 # nearly round (curled up): no reliable direction
        return None
    ang = 0.5 * math.atan2(2 * cxy, cxx - cyy)
    vx, vy = math.cos(ang), math.sin(ang)
    proj = dx * vx + dy * vy
    perp = -dx * vy + dy * vx
    lo, hi = np.percentile(proj, [1.0, 99.0])
    length = float(hi - lo)
    if length < 4:
        return None
    end_a = proj > hi - 0.22 * length
    end_b = proj < lo + 0.22 * length
    wa = float(np.ptp(perp[end_a])) if end_a.sum() > 2 else 0.0
    wb = float(np.ptp(perp[end_b])) if end_b.sum() > 2 else 0.0
    shape_ev = (wb - wa) / max(wa, wb, 1e-6)
    tail_ev = 0.0
    if full_mask is not None:
        grown = cv2.dilate(body_mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        tail = full_mask & ~grown
        tys, txs = np.nonzero(tail)
        if txs.size >= 4:
            tp = (txs - mx) * vx + (tys - my) * vy
            na = float((tp > hi - 0.1 * length).sum())
            nb = float((tp < lo + 0.1 * length).sum())
            if na + nb >= 4:
                tail_ev = (na - nb) / (na + nb)
    ax, ay = mx + vx * hi + x0, my + vy * hi + y0
    bx, by = mx + vx * lo + x0, my + vy * lo + y0
    return (float(ax), float(ay), float(bx), float(by), float(tail_ev), float(shape_ev))


def head_evidence(ends, vel=None, speed_ref=1.0):
    """How strongly the frame says 'end A is the head' (> 0) or 'end B is the head' (< 0)."""
    ax, ay, bx, by, tail_ev, shape_ev = ends
    e = -1.5 * tail_ev + 0.5 * shape_ev
    if vel is not None:
        ux, uy = ax - bx, ay - by
        n = math.hypot(ux, uy)
        sp = math.hypot(vel[0], vel[1])
        if n > 0 and sp > 0:
            cosv = (ux * vel[0] + uy * vel[1]) / (n * sp)
            e += 1.2 * cosv * min(1.0, sp / max(speed_ref, 1e-6))
    return e


def resolve_head_tail(raw, xs=None, ys=None, valid=None):
    """Head and tail-base positions for every frame of a tracking run (full-resolution px).
    Returns (head_x, head_y, tail_x, tail_y) float arrays (NaN where unknown)."""
    ends = raw.get("ends") or []
    n = len(raw.get("frames") or [])
    hx = np.full(n, np.nan)
    hy = np.full(n, np.nan)
    tx = np.full(n, np.nan)
    ty = np.full(n, np.nan)
    if not ends or len(ends) != n:
        return hx, hy, tx, ty
    fps = float(raw.get("fps") or 30.0)
    cx = np.asarray(raw["xs"] if xs is None else xs, float)
    cy = np.asarray(raw["ys"] if ys is None else ys, float)
    body_len = float(raw.get("body_len_full") or 0.0)
    if body_len <= 0:
        lens = [math.hypot(e[0] - e[2], e[1] - e[3]) for e in ends if e]
        body_len = float(np.median(lens)) if lens else 30.0
    body_len = max(body_len, 4.0)
    # centre velocity over +-0.2 s (smooth enough to ignore jitter)
    k = max(1, int(round(0.2 * fps)))
    vel = [None] * n
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if b > a and np.isfinite(cx[a]) and np.isfinite(cx[b]):
            vel[i] = ((cx[b] - cx[a]) / ((b - a) / fps), (cy[b] - cy[a]) / ((b - a) / fps))
    speed_ref = 1.5 * body_len                       # >= 1.5 body lengths per second = clearly walking
    idx = [i for i in range(n) if ends[i]]
    if not idx:
        return hx, hy, tx, ty
    # dynamic programming over the orientation s (0: head = A, 1: head = B)
    w_cont = 2.0
    cost = np.zeros((len(idx), 2))
    back = np.zeros((len(idx), 2), int)
    prev = None
    for j, i in enumerate(idx):
        e = ends[i]
        ev = head_evidence(e, vel[i], speed_ref)
        unary = np.array([-ev, ev])
        if prev is None:
            cost[j] = unary
        else:
            pi, pe = prev
            gap = i - pi
            wc = w_cont / (1.0 + 0.5 * max(0, gap - 1))          # weaker link across lost frames
            heads_p = ((pe[0], pe[1]), (pe[2], pe[3]))
            heads_c = ((e[0], e[1]), (e[2], e[3]))
            for s in (0, 1):
                best, arg = None, 0
                for sp in (0, 1):
                    hp, tp = heads_p[sp], heads_p[1 - sp]
                    hc, tc = heads_c[s], heads_c[1 - s]
                    jump = (math.hypot(hc[0] - hp[0], hc[1] - hp[1]) +
                            math.hypot(tc[0] - tp[0], tc[1] - tp[1])) / body_len
                    c = cost[j - 1, sp] + wc * jump
                    if best is None or c < best:
                        best, arg = c, sp
                cost[j, s] = best + unary[s]
                back[j, s] = arg
        prev = (i, e)
    s = int(np.argmin(cost[-1]))
    for j in range(len(idx) - 1, -1, -1):
        i = idx[j]
        e = ends[i]
        if s == 0:
            hx[i], hy[i], tx[i], ty[i] = e[0], e[1], e[2], e[3]
        else:
            hx[i], hy[i], tx[i], ty[i] = e[2], e[3], e[0], e[1]
        s = back[j, s]
    if valid is not None:
        bad = ~np.asarray(valid, bool)
        hx[bad] = hy[bad] = tx[bad] = ty[bad] = np.nan
    return hx, hy, tx, ty


class OnlineOrientation:
    """Head / tail for the LIVE view (frame by frame; the saved results use resolve_head_tail)."""

    def __init__(self):
        self.head = None
        self.prev_c = None

    def update(self, ends, cx, cy):
        if not ends:
            self.prev_c = None if cx is None else (cx, cy)
            return None
        ax, ay, bx, by = ends[:4]
        vel = None
        if self.prev_c is not None and cx is not None:
            vel = (cx - self.prev_c[0], cy - self.prev_c[1])
        self.prev_c = (cx, cy)
        ev = head_evidence(ends, vel, speed_ref=2.0)
        if self.head is not None:
            da = math.hypot(ax - self.head[0], ay - self.head[1])
            db = math.hypot(bx - self.head[0], by - self.head[1])
            ln = max(1.0, math.hypot(ax - bx, ay - by))
            ev += 2.0 * (db - da) / ln
        if ev >= 0:
            self.head = (ax, ay)
            return (ax, ay, bx, by)
        self.head = (bx, by)
        return (bx, by, ax, ay)
