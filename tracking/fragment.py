"""'Fragment' for zones -- like PowerPoint's Merge Shapes ▸ Fragment: overlapping shapes are cut into every
separate piece formed by their outlines.  Two crossing bars (a plus maze) give 4 arms + 1 centre; a square
inside a square (an open field) gives the centre + the border ring.

Works on pixel masks (exact at the frame's resolution) and returns plain polygons.  A piece with a hole (a
ring) is returned as one 'keyhole' polygon: the hole is joined to the outline by a zero-width cut, so it
still fills as a ring."""
import cv2
import numpy as np


def _poly_mask(pts, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(pts, np.float64)).astype(np.int32)], 1)
    return m


def _keyhole(outer, holes):
    """Join every hole to the outer outline at their closest points -> one polygon that fills as a ring."""
    poly = [tuple(p) for p in outer]
    for hole in holes:
        hole = [tuple(p) for p in hole]
        if len(hole) < 3:
            continue
        P = np.asarray(poly, float)
        Hh = np.asarray(hole, float)
        d = ((P[:, None, :] - Hh[None, :, :]) ** 2).sum(axis=2)
        i, j = np.unravel_index(int(d.argmin()), d.shape)
        loop = hole[j:] + hole[:j + 1]
        poly = poly[:i + 1] + loop + poly[i:]
    return poly


def _raster(pts, w, h):
    """Exactly how BehavioralTracker rasterises a zone (tracking.location.build_roi_masks)."""
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.array(pts, dtype=np.int32)], 255)
    return m > 0


def _trace(comp, eps, min_area_px):
    """One connected piece (uint8 mask) -> (polygon, has_hole) or None."""
    contours, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hier = hier[0]
    outer_i = max((i for i in range(len(contours)) if hier[i][3] < 0), key=lambda i: cv2.contourArea(contours[i]))

    def simp(cnt):
        return [(float(x), float(y)) for x, y in cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)]
    outer = simp(contours[outer_i])
    holes = [simp(contours[i]) for i in range(len(contours))
             if hier[i][3] == outer_i and cv2.contourArea(contours[i]) >= min_area_px]
    poly = _keyhole(outer, holes) if holes else outer
    return (poly, bool(holes)) if len(poly) >= 3 else None


def fragment_shapes(polys, w, h, min_area_px=None, eps=0.8):
    """polys: list of polygons [(x, y), ...] in frame pixels.  Returns the pieces as polygons, in reading
    order (top to bottom, left to right).  Pieces smaller than min_area_px (default 0.4 % of the shapes'
    total area) are dropped -- they are slivers where outlines almost meet."""
    polys = [p for p in polys if len(p) >= 3]
    if len(polys) < 1 or len(polys) > 30:
        return []
    sig = np.zeros((h, w), np.int64)
    for k, p in enumerate(polys):
        sig |= (_poly_mask(p, w, h).astype(np.int64) << k)
    total = int((sig > 0).sum())
    if min_area_px is None:
        min_area_px = max(30, int(0.004 * total))
    comps = []
    for val in np.unique(sig):
        if val == 0:
            continue
        m = (sig == val).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=4)
        for c in range(1, n):
            if stats[c, cv2.CC_STAT_AREA] >= min_area_px:
                comps.append(((lab == c).astype(np.uint8), float(cents[c][1]), float(cents[c][0])))
    # The pieces must not share a single pixel once BehavioralTracker turns them back into masks (a point on a
    # shared edge would count in two zones).  Plain pieces are placed first; a piece that would overlap one
    # already placed (the border ring around a centre) is traced again 1 px smaller there.
    traced = [(comp, cy, cx, _trace(comp, eps, min_area_px)) for comp, cy, cx in comps]
    traced.sort(key=lambda t: (t[3] is not None and t[3][1], -int(t[0].sum())))
    claimed = np.zeros((h, w), bool)
    pieces = []
    for comp, cy, cx, tr in traced:
        if tr is None:
            continue
        poly = tr[0]
        for _ in range(3):
            r = _raster(poly, w, h)
            if not (r & claimed).any():
                break
            comp = (comp.astype(bool) & ~cv2.dilate(claimed.astype(np.uint8), np.ones((3, 3), np.uint8))
                    .astype(bool)).astype(np.uint8)
            tr2 = _trace(comp, eps, min_area_px)
            if tr2 is None:
                break
            poly = tr2[0]
        r = _raster(poly, w, h)
        if len(poly) < 3 or int(r.sum()) < min_area_px:
            continue
        claimed |= r
        pieces.append((cy, cx, poly))
    if not pieces:
        return []
    # reading order: rows of similar height, then left to right
    ys = sorted(p[0] for p in pieces)
    band = max(10.0, 0.15 * (ys[-1] - ys[0] + 1))
    pieces.sort(key=lambda p: (round(p[0] / band), p[1]))
    return [p[2] for p in pieces]


def overlap_px(polys, w, h):
    """Number of pixels covered by two or more of the shapes (0 = they do not overlap)."""
    cnt = np.zeros((h, w), np.uint8)
    for p in polys:
        if len(p) >= 3:
            cnt += _poly_mask(p, w, h)
    return int((cnt >= 2).sum())


def merge_shapes(polys, w, h, eps=0.8):
    """Union of the shapes (PowerPoint 'Union').  Returns one polygon, or None when the shapes do not touch
    (they would give two separate pieces)."""
    polys = [p for p in polys if len(p) >= 3]
    if not polys:
        return None
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        m |= _poly_mask(p, w, h)
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    hier = hier[0]
    outers = [i for i in range(len(contours)) if hier[i][3] < 0 and cv2.contourArea(contours[i]) >= 20]
    if len(outers) != 1:
        return None
    oi = outers[0]
    simp = lambda cnt: [(float(x), float(y)) for x, y in  # noqa: E731
                        cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)]
    holes = [simp(contours[i]) for i in range(len(contours))
             if hier[i][3] == oi and cv2.contourArea(contours[i]) >= 20]
    outer = simp(contours[oi])
    return _keyhole(outer, holes) if holes else outer


def poly_rings(pts):
    """Split a polygon that touches itself (a 'keyhole' ring made by Fragment) into its separate loops, so its
    outline can be drawn without the zero-width cut.  A normal polygon comes back as [pts]."""
    stack, idx, rings = [], {}, []
    for p in pts:
        key = (round(float(p[0]), 3), round(float(p[1]), 3))
        k = idx.get(key)
        if k is None:
            idx[key] = len(stack)
            stack.append(p)
            continue
        loop = stack[k:]
        for q in loop[1:]:
            idx.pop((round(float(q[0]), 3), round(float(q[1]), 3)), None)
        stack = stack[:k + 1]
        if len(loop) >= 3:
            rings.append(loop)
    if len(stack) >= 3:
        rings.append(stack)
    return rings or [list(pts)]


def outline_arrays(pts):
    """cv2.polylines-ready int32 arrays for a zone outline (one per loop)."""
    return [np.round(np.asarray(r, np.float64)).astype(np.int32) for r in poly_rings(pts)]


_LABEL_CACHE = {}


def label_point(pts):
    """Where to write a zone's name: its centre, or -- for a ring / L-shape whose centre lies outside the zone
    (e.g. an open-field border made with Fragment) -- the point deepest inside it."""
    a = np.asarray(pts, np.float64)
    cx, cy = float(a[:, 0].mean()), float(a[:, 1].mean())
    if len(a) < 3 or cv2.pointPolygonTest(a.astype(np.float32), (cx, cy), False) >= 0:
        return cx, cy
    key = tuple((round(float(x), 1), round(float(y), 1)) for x, y in a)
    hit = _LABEL_CACHE.get(key)
    if hit is not None:
        return hit
    x0, y0 = a.min(axis=0)
    span = max(float((a.max(axis=0) - a.min(axis=0)).max()), 1.0)
    s = 200.0 / span
    w = int(np.ceil((a[:, 0].max() - x0) * s)) + 3
    h = int(np.ceil((a[:, 1].max() - y0) * s)) + 3
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round((a - (x0, y0)) * s + 1).astype(np.int32)], 255)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    iy, ix = np.unravel_index(int(d.argmax()), d.shape)
    res = (float((ix - 1) / s + x0), float((iy - 1) / s + y0))
    if len(_LABEL_CACHE) > 400:
        _LABEL_CACHE.clear()
    _LABEL_CACHE[key] = res
    return res


def letter(i):
    """0 -> A, 25 -> Z, 26 -> AA ..."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s
