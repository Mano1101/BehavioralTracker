"""Robust single-animal detection engine (BehavioralTracker 1.0.0).

Why this exists
---------------
The earlier "classic" loop lost or mis-tracked the animal in a few very
common situations.  Each one is handled here:

* Animal resting in one place for a long time -> the plain median
  background contains the animal ("ghost"); when it walks away the empty
  spot looks like an animal.  -> Polarity-aware background (dark animal =
  bright percentile, light animal = dark percentile) followed by an
  animal-removal refinement pass.
* Tracking lost for a moment (occlusion, dark corner, rearing) and the
  animal re-appears further than "max movement" away -> the old loop kept
  the stale last position for ever.  -> The search gate grows while lost
  and a global re-acquisition runs after a short time.
* A hand / reflection / shadow outside the apparatus grabbed the lock at
  the start and never let go.  -> Everything outside the zones is masked
  (when enabled), the first lock prefers an animal-sized blob, and the
  polarity vote ignores rare/huge blobs.
* HD videos: the animal is bigger than the fixed 5000 px "max area".  ->
  Animal size is measured automatically from the background samples and
  the size limits follow it.
* Camera shake / vibration -> global shift estimated by phase correlation
  and compensated; thresholds are raised a little on strong background
  edges; thin edge-like blobs are rejected.
* Lighting flicker / auto-exposure drift / passing shadows -> per-frame
  global gain compensation and splitting of oversized blobs.
* Frame-by-frame seeking (slow and inaccurate on many codecs) -> the
  caller reads frames sequentially; this module never seeks per frame.

The engine works on a (possibly down-scaled) blurred copy of every frame;
all public coordinates are returned in full-resolution (warped) pixels.
"""
import math
import warnings

import cv2
import numpy as np

DET_MAX_SIDE = 960          # frames larger than this are down-scaled for detection only


# ----------------------------------------------------------------------------- helpers
def detection_scale(w, h, max_side=DET_MAX_SIDE):
    m = max(int(w), int(h))
    return 1.0 if m <= max_side else max_side / float(m)


# ----------------------------------------------------------------------------- low light
LOW_LIGHT_LEVELS = {"low": 1.6, "medium": 2.2, "high": 3.0}
_LUTS = {}


def gamma_lut(gamma):
    """256-entry brightening curve (gamma > 1 lifts the dark grey levels: a dim corner / an open dark box at a
    few lux gets several times more contrast, bright areas lose a little).  Cached."""
    g = round(float(gamma), 2)
    lut = _LUTS.get(g)
    if lut is None:
        x = np.arange(256, dtype=np.float64) / 255.0
        lut = np.clip(np.round(255.0 * np.power(x, 1.0 / max(g, 1e-6))), 0, 255).astype(np.uint8)
        _LUTS[g] = lut
    return lut


def brighten(img_u8, gamma):
    """The same brightening on any uint8 image (grey or BGR) -- for the screen."""
    if not gamma or gamma <= 1.001:
        return img_u8
    return cv2.LUT(img_u8, gamma_lut(gamma))


def auto_gamma(level):
    """Brightening for an area whose floor is `level` grey levels: none from 70 up, otherwise enough to lift
    it to ~110 (at most gamma 3)."""
    level = float(level)
    if level >= 70.0:
        return 1.0
    level = max(level, 2.0)
    g = math.log(level / 255.0) / math.log(110.0 / 255.0)
    return float(round(min(3.0, max(1.0, g)), 1))


def darkest_level(gray, valid=None, zone_masks=None):
    """Floor brightness of the darkest place the animal can be: the darkest zone (median), or the 20th
    percentile of the open area when there are no zones."""
    g = gray.astype(np.float32)
    if g.ndim == 3:
        g = g.mean(axis=2)
    ok = np.ones(g.shape, bool) if valid is None else valid
    total = max(1, int(ok.sum()))
    levels = []
    for m in zone_masks or []:
        sel = m & ok
        if sel.sum() >= max(50, 0.005 * total):
            levels.append(float(np.median(g[sel])))
    if levels:
        return min(levels)
    v = g[ok]
    return float(np.percentile(v, 20)) if v.size else 128.0


def resolve_gamma(setting, gray=None, valid=None, zone_masks=None):
    """'off' | 'auto' | 'low' | 'medium' | 'high' | a number -> gamma (1.0 = no brightening)."""
    s = str(setting if setting is not None else "off").strip().lower()
    if s in ("", "off", "none", "0", "1", "1.0", "false"):
        return 1.0
    if s in LOW_LIGHT_LEVELS:
        return LOW_LIGHT_LEVELS[s]
    if s == "auto":
        if gray is None:
            return 1.0
        return auto_gamma(darkest_level(gray, valid, zone_masks))
    try:
        return float(min(3.0, max(1.0, float(s))))
    except ValueError:
        return 1.0


def prepare_raw(frame_bgr, color_mode="gray", scale=1.0):
    """BGR frame -> down-scaled uint8 grey (or colour) image, not yet blurred (see finish())."""
    img = frame_bgr
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if str(color_mode).lower() != "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def finish(img, gamma=1.0):
    """prepare_raw() image -> working image: low-light boost (before the blur, so the brightened dark levels
    stay smooth instead of jumping in steps) + 5x5 blur."""
    if gamma and gamma > 1.001:
        img = cv2.LUT(img, gamma_lut(gamma))
    return cv2.GaussianBlur(img, (5, 5), 0)


def prepare(frame_bgr, color_mode="gray", scale=1.0, gamma=1.0):
    """BGR frame -> blurred uint8 working image (gray HxW or colour HxWx3); gamma > 1 = low-light boost."""
    return finish(prepare_raw(frame_bgr, color_mode, scale), gamma)


def to_gray_f(img):
    img = img.astype(np.float32)
    return img.mean(axis=2) if img.ndim == 3 else img


def polar_diff(img, bg, polarity):
    """Foreground strength (float32 HxW, >= 0 for dark/light).

    dark  -> how much darker than the background (black mouse on light floor)
    light -> how much brighter (white mouse on dark floor)
    any   -> absolute difference
    Colour images use the strongest channel."""
    d = img.astype(np.float32) - bg
    if d.ndim == 3:
        if polarity == "dark":
            return np.max(-d, axis=2)
        if polarity == "light":
            return np.max(d, axis=2)
        return np.max(np.abs(d), axis=2)
    if polarity == "dark":
        return -d
    if polarity == "light":
        return d
    return np.abs(d)


def _polar_diff_u8(img_u8, bg, polarity):
    """polar_diff() for a grey uint8 frame, computed by OpenCV directly from the 8-bit image: exactly the
    same float32 values (x - y == -(y - x) in IEEE arithmetic) at a fraction of the cost."""
    if polarity == "dark":
        return cv2.subtract(bg, img_u8, dtype=cv2.CV_32F)
    d = cv2.subtract(img_u8, bg, dtype=cv2.CV_32F)
    if polarity == "light":
        return d
    return np.abs(d, out=d)


def _banded(stack, fn, band=48):
    """Apply a per-pixel reduction over axis 0 in horizontal bands (bounded memory)."""
    h = stack.shape[1]
    out = None
    for y0 in range(0, h, band):
        part = fn(stack[:, y0:y0 + band])
        if out is None:
            out = np.empty((h,) + part.shape[1:], np.float32)
        out[y0:y0 + band] = part
    return out


def shift_image(img, dx, dy):
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return img
    m = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class _ShiftEstimator:
    """Global (camera-shake) translation of a frame relative to the background.

    Gauss-Newton / Lucas-Kanade on the strongest background edges, with
    robust rejection of pixels covered by the animal or a shadow.  Accurate
    for the small shifts typical of vibration (< ~6 px); a shift is only
    accepted when it really makes the frame match the background better."""

    MAX_SHIFT = 8.0

    def __init__(self, ref_gray):
        self.ref = ref_gray.astype(np.float32)
        hh, ww = self.ref.shape
        m = max(6, int(0.05 * min(hh, ww)))
        gx = cv2.Sobel(self.ref, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(self.ref, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gm = np.sqrt(gx * gx + gy * gy)
        inner = np.zeros_like(gm, bool)
        inner[m:hh - m, m:ww - m] = True
        thr = max(2.0, float(np.percentile(gm[inner], 96))) if inner.any() else 1e9
        sel = inner & (gm >= thr)
        iy, ix = np.nonzero(sel)
        if iy.size > 6000:
            k = np.linspace(0, iy.size - 1, 6000).astype(int)
            iy, ix = iy[k], ix[k]
        self.iy, self.ix = iy, ix
        self.gx, self.gy = gx[iy, ix], gy[iy, ix]
        self.r = self.ref[iy, ix]
        gy0, gx0 = np.mgrid[m:hh - m:7, m:ww - m:7]
        self.flat = (gy0.ravel(), gx0.ravel())
        self.flat_ref = self.ref[self.flat]
        self.textured = iy.size >= 150 and float(np.linalg.det(np.array(
            [[np.sum(self.gx ** 2), np.sum(self.gx * self.gy)],
             [np.sum(self.gx * self.gy), np.sum(self.gy ** 2)]]))) > 1e-3

    def _sample(self, g, dx, dy):
        mx = (self.ix + dx).astype(np.float32).reshape(1, -1)
        my = (self.iy + dy).astype(np.float32).reshape(1, -1)
        return cv2.remap(g, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()

    def __call__(self, gray):
        if not self.textured:
            return 0.0, 0.0
        g = gray.astype(np.float32)
        ok = self.flat_ref > 15
        if ok.sum() > 50:                       # remove flicker / exposure change first
            gain = float(np.median(g[self.flat][ok] / self.flat_ref[ok]))
            if 0.5 < gain < 2.0:
                g = g / gain
        dx = dy = 0.0
        res0 = None
        for _ in range(4):
            it = self._sample(g, dx, dy) - self.r
            a = np.abs(it)
            if res0 is None:
                res0 = float(np.median(a))
            keep = a <= 3.0 * float(np.median(a)) + 6.0
            gx, gy, itk = self.gx[keep], self.gy[keep], it[keep]
            A = np.array([[np.sum(gx * gx), np.sum(gx * gy)], [np.sum(gx * gy), np.sum(gy * gy)]])
            b = np.array([np.sum(gx * itk), np.sum(gy * itk)])
            try:
                step = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                return 0.0, 0.0
            dx -= float(step[0])
            dy -= float(step[1])
            if abs(dx) > self.MAX_SHIFT or abs(dy) > self.MAX_SHIFT:
                return 0.0, 0.0
            if abs(step[0]) < 0.02 and abs(step[1]) < 0.02:
                break
        res1 = float(np.median(np.abs(self._sample(g, dx, dy) - self.r)))
        if res1 > 0.85 * res0 or (abs(dx) < 0.1 and abs(dy) < 0.1):
            return 0.0, 0.0
        return dx, dy


def blob_shape_ok(area, bw, bh, body_w=None):
    """Reject thin, edge-like components (wall edges from camera shake, cables...)."""
    if bw <= 0 or bh <= 0:
        return False
    long_side = max(bw, bh)
    thickness = area / float(long_side)
    extent = area / float(bw * bh)
    if long_side / max(1.0, min(bw, bh)) > 8 and extent < 0.25:
        return False
    if thickness < 2.5:
        return False
    if body_w and thickness < 0.3 * body_w:
        return False
    return True


def gap_fill(xs, ys, status_tracked, max_gap):
    """Linear interpolation over runs of untracked frames no longer than
    `max_gap` that have a tracked frame on both sides.  Returns a boolean
    array marking the filled frames (xs/ys are modified in place)."""
    n = len(xs)
    filled = np.zeros(n, bool)
    if max_gap <= 0 or n < 3:
        return filled
    i = 0
    while i < n:
        if status_tracked[i]:
            i += 1
            continue
        j = i
        while j < n and not status_tracked[j]:
            j += 1
        if i > 0 and j < n and (j - i) <= max_gap:
            x0, y0, x1, y1 = xs[i - 1], ys[i - 1], xs[j], ys[j]
            for k in range(i, j):
                t = (k - i + 1) / float(j - i + 1)
                xs[k] = x0 + (x1 - x0) * t
                ys[k] = y0 + (y1 - y0) * t
                filled[k] = True
        i = j
    return filled


# ----------------------------------------------------------------------------- background
class BackgroundModel:
    def __init__(self, image, polarity, sigma, animal_area, scale, color_mode, valid, samples_used,
                 votes=None, stabilize=False, shake_px=0.0, gamma=1.0, dark_level=None, sigma_map=None):
        self.gamma = float(gamma or 1.0)               # low-light boost applied to every frame (1 = none)
        self.dark_level = dark_level                   # floor brightness of the darkest zone (before the boost)
        self.sigma_map = sigma_map                     # per-pixel noise (with the boost: dark areas are noisier)
        self.polarity_warning = None                   # the chosen animal colour looks wrong ('dark' / 'light')
        self.image = image.astype(np.float32)          # detection-scale working image
        self.polarity = polarity
        self.sigma = float(sigma)
        self.animal_area = animal_area                 # detection-scale px, or None
        self.scale = float(scale)
        self.color_mode = color_mode
        self.valid = valid                             # bool HxW (detection scale) -- not excluded
        self.samples_used = samples_used
        self.votes = votes or {}
        self.stabilize = bool(stabilize)
        self.shake_px = float(shake_px)
        g = to_gray_f(self.image)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        self.edges = np.sqrt(gx * gx + gy * gy)        # background gradient (levels / px)
        self.gray = g

    def full_res_bgr(self, w, h):
        """uint8 BGR background at full (warped) resolution, for charts/previews."""
        img = np.clip(self.image, 0, 255).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
        return img

    def describe(self):
        pol = {"dark": "animal darker than floor", "light": "animal lighter than floor",
               "any": "absolute difference"}.get(self.polarity, self.polarity)
        size = (f"{self.animal_area / (self.scale ** 2):.0f} px" if self.animal_area else "unknown")
        txt = f"{pol}; noise σ={self.sigma:.1f}; animal size ≈ {size}"
        if self.stabilize:
            txt += f"; camera shake compensated (≈{self.shake_px:.1f} px)"
        if self.gamma > 1.001:
            txt += f"; low-light boost ×{self.gamma:.1f}"
        return txt


def estimate_sigma(stack, ref, valid):
    """Robust per-pixel noise sigma from a few samples (MAD)."""
    idx = np.linspace(0, len(stack) - 1, min(8, len(stack))).astype(int)
    vals = []
    for i in idx:
        d = np.abs(stack[i].astype(np.float32) - ref)
        if d.ndim == 3:
            d = d.max(axis=2)
        v = d[valid]
        if v.size:
            vals.append(v[:: max(1, v.size // 20000)])
    if not vals:
        return 1.0
    allv = np.concatenate(vals)
    return max(0.6, 1.4826 * float(np.median(allv)))


def _components(d, thr, valid, a_min, body_w=None, edges=None, kappa=0.0):
    """Animal-shaped components of d > thr (+ edge-aware raise)."""
    t = thr if edges is None or kappa <= 0 else thr + kappa * edges
    m = ((d > t) & valid).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        a = float(stats[k, cv2.CC_STAT_AREA])
        if a < a_min:
            continue
        bw, bh = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
        if not blob_shape_ok(a, bw, bh, body_w):
            continue
        out.append((k, a, cents[k]))
    return out, lab


def _vote_polarity(stack, b0, valid, sigma, scale, edges):
    """Decide whether the animal is darker or lighter than the floor.

    For each sign the largest animal-shaped blob in every sample is
    collected.  The real animal is seen in many samples AND moves around; a
    hand/reflection is huge but rare; a "ghost" (the place where the animal
    rested while the median was taken) has the opposite sign but never
    moves.  Score = count x typical mass x (spatial spread + body size)."""
    thr = max(10.0, 6.0 * sigma)
    a_min = max(12.0 * scale * scale, 0.00004 * float(valid.sum()))
    idx = np.linspace(0, len(stack) - 1, min(60, len(stack))).astype(int)
    res = {}
    for pol in ("dark", "light"):
        masses, cents, areas = [], [], []
        for i in idx:
            d = polar_diff(stack[i], b0, pol)
            comps, lab = _components(d, thr, valid, a_min, edges=edges, kappa=0.35)
            if not comps:
                continue
            k, a, c = max(comps, key=lambda t: t[1])
            masses.append(float(d[lab == k].sum()))
            cents.append(c)
            areas.append(a)
        if masses:
            c = np.array(cents)
            med = np.median(c, axis=0)
            # 80th percentile: an animal that rests most of the time still moves in some samples, while a
            # flickering label / lamp / shadow never leaves its place
            spread = float(np.percentile(np.hypot(c[:, 0] - med[0], c[:, 1] - med[1]), 80))
            ar = np.asarray(areas, float)
            body = math.sqrt(float(np.median(ar)))
            # the animal has about the same size in every sample; flicker of a grid floor, compression noise, a
            # hand or a light streak give blobs of any size (scattered, so their 'spread' alone would win)
            cv = float(np.median(np.abs(ar - np.median(ar))) / max(1.0, np.median(ar)))
            score = len(masses) * float(np.median(masses)) * (spread + 0.3 * body) / (1.0 + 2.0 * cv)
        else:
            score, spread, body, cv = 0.0, 0.0, 0.0, 0.0
        res[pol] = {"score": score, "spread": spread, "body": body, "found": len(masses), "size_cv": cv}
    sd, sl = res["dark"]["score"], res["light"]["score"]
    if sd <= 0 and sl <= 0:
        floor = float(np.mean(to_gray_f(b0)[valid])) if valid.any() else 128.0
        choice = "dark" if floor >= 100 else "light"
    else:
        choice = "dark" if sd >= sl else "light"
    res["choice"] = choice
    return choice, res


def _even_brightness(stack, valid):
    """A light switched on / off for part of the video (a light cue, a house light) makes some samples much
    brighter or darker than the rest; the background would then be built from a mix.  Such samples are brought
    to the usual brightness first (shift of their median grey level).  Samples within +-3 grey levels of the
    usual level -- every sample of an ordinary video -- are not touched.  Returns (stack, biggest shift)."""
    v = valid if valid is not None and valid.any() else np.ones(stack.shape[1:3], bool)
    step = max(1, int(math.sqrt(v.sum() / 20000.0)))
    vv = v[::step, ::step]
    levels = np.array([float(np.median((s_[::step, ::step] if s_.ndim == 2
                                         else s_[::step, ::step].mean(axis=2))[vv])) for s_ in stack])
    off = levels - float(np.median(levels))
    if len(stack) < 5 or float(np.max(np.abs(off))) < 6.0:
        return stack, 0.0
    out = stack.copy()
    for i, o in enumerate(off):
        if abs(o) >= 3.0:
            out[i] = np.clip(stack[i].astype(np.float32) - o, 0, 255).astype(np.uint8)
    return out, float(np.max(np.abs(off)))


def build_background(read_frame, positions, color_mode="gray", polarity="auto",
                     valid_full=None, scale=1.0, progress=None, refine_samples=60, stabilize="auto",
                     prepared=None, low_light="off", zone_polys=None, raw_samples=None, rest_pct=None):
    """read_frame(frame_index) -> warped full-res BGR frame or None.

    Returns a BackgroundModel.  `valid_full` (bool, full res) marks pixels
    that may contain the animal (exclusion masks removed).  `prepared` = the
    samples already read and prepared (prepare(frame, color_mode, scale)) by a
    faster parallel reader; then read_frame is not used.  `raw_samples` = the same, prepare_raw()d (not
    blurred yet): needed for the low-light boost, which is applied before the blur.
    low_light = 'off' | 'auto' | 'low' | 'medium' | 'high': brightening of the dark grey levels (the same
    curve is then applied to every frame by the tracker); 'auto' measures the darkest zone (`zone_polys`
    = {name: full-res polygon} of the zones the animal can be seen in).
    rest_pct (freezing): the percentile of the first floor estimate for a dark animal (100 - it for a light one;
    default 80).  97 finds the floor where the animal rested in all but a few samples, so it can be masked out of
    the samples there too (the final background is still the median of the samples without the animal)."""
    samples = []
    n = len(positions)
    raw = None
    if raw_samples is not None:
        raw = [p for p in raw_samples if p is not None]
        if progress is not None:
            progress(0.55)
    elif prepared is not None:
        samples = [p for p in prepared if p is not None]
        if progress is not None:
            progress(0.55)
    else:
        raw = []
        for i, pos in enumerate(positions):
            f = read_frame(int(pos))
            if f is not None:
                raw.append(prepare_raw(f, color_mode, scale))
            if progress is not None and (i % 5 == 0 or i == n - 1):
                progress(0.55 * (i + 1) / max(1, n))
    first = raw if raw is not None else samples
    if not first:
        raise RuntimeError("Could not read any frame to build the background.")
    h, w = first[0].shape[0], first[0].shape[1]
    if valid_full is not None:
        valid = cv2.resize(valid_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        if valid.sum() < 50:
            valid = np.ones((h, w), bool)
    else:
        valid = np.ones((h, w), bool)
    # ---- low-light boost: how dark is the darkest zone (median of a few samples), which curve to use
    zmasks = []
    for pts in (zone_polys or {}).values():
        if pts is not None and len(pts) >= 3:
            zm = np.zeros((h, w), np.uint8)
            cv2.fillPoly(zm, [np.round(np.asarray(pts, np.float64) * scale).astype(np.int32)], 1)
            zmasks.append(zm > 0)
    pick = np.linspace(0, len(first) - 1, min(15, len(first))).astype(int)
    level_img = np.median(np.stack([first[i] for i in pick]), axis=0)
    dark_level = darkest_level(level_img, valid, zmasks)
    gamma = resolve_gamma(low_light, level_img, valid, zmasks)
    if raw is not None:
        samples = [finish(r, gamma) for r in raw]
        raw = first = None
    elif gamma > 1.001:                  # already blurred samples: the curve after the blur (close enough)
        lut = gamma_lut(gamma)
        samples = [cv2.LUT(s_, lut) for s_ in samples]
    stack = np.stack(samples, axis=0)                  # uint8 N,H,W[,3]
    del samples
    stack, level_shift = _even_brightness(stack, valid)

    b0 = _banded(stack, lambda s: np.median(s, axis=0))

    # ---- camera shake: estimate every sample's shift against the median, re-align if needed
    shake, use_stab = 0.0, False
    if stabilize in ("auto", True) and len(stack) >= 5:
        est = _ShiftEstimator(to_gray_f(b0))
        shifts = np.array([est(to_gray_f(s)) for s in stack])
        mags = np.hypot(shifts[:, 0], shifts[:, 1])
        shake = float(np.percentile(mags, 90))
        accepted = float(np.mean(mags > 0))
        use_stab = stabilize is True or (shake > 0.7 and accepted >= 0.4)
        if use_stab:
            stack = np.stack([np.clip(shift_image(s, dx, dy), 0, 255).astype(np.uint8)
                              for s, (dx, dy) in zip(stack, shifts)])
            b0 = _banded(stack, lambda s: np.median(s, axis=0))
    sigma = estimate_sigma(stack, b0, valid)
    g0 = to_gray_f(b0)
    gx = cv2.Sobel(g0, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(g0, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    edges = np.sqrt(gx * gx + gy * gy)

    votes = {}
    pol = str(polarity or "auto").lower()
    polarity_warning = None
    if pol not in ("dark", "light", "any"):
        pol, votes = _vote_polarity(stack, b0, valid, sigma, scale, edges)
    elif pol == "any":
        # 'either' also follows light changes, shadows and reflections: say so when the animal clearly has one colour
        try:
            choice, votes = _vote_polarity(stack, b0, valid, sigma, scale, edges)
            other = "light" if choice == "dark" else "dark"
            vc, vo = votes[choice], votes[other]
            if (vc["score"] > 0 and vc["score"] >= 3.0 * max(vo["score"], 0.0) and vc["found"] >= 0.3 * min(60, len(stack))
                    and vc["spread"] >= 0.5 * max(vc["body"], 1.0)):
                polarity_warning = choice
        except Exception:
            votes = {}
    elif pol in ("dark", "light"):
        # the animal colour was chosen by hand: check it -- with the wrong colour the tracker follows the
        # empty place where the animal sat (a 'ghost') instead of the animal
        try:
            choice, votes = _vote_polarity(stack, b0, valid, sigma, scale, edges)
            vm, vo = votes[pol], votes[choice]
            mine, other = vm["score"], vo["score"]
            n_s = min(60, len(stack))
            mine_static = vm["found"] == 0 or vm["spread"] < 0.5 * max(vm["body"], 1.0)
            other_moves = vo["found"] >= 0.3 * n_s and vo["spread"] >= 0.5 * max(vo["body"], 1.0)
            if choice != pol and other > 0 and (mine <= 0 or other >= 3.0 * mine or (mine_static and other_moves)):
                polarity_warning = choice
        except Exception:
            votes = {}
    if progress is not None:
        progress(0.7)

    hi = float(rest_pct) if rest_pct else 80.0
    if pol == "dark":
        bp = _banded(stack, lambda s: np.percentile(s, hi, axis=0))
    elif pol == "light":
        bp = _banded(stack, lambda s: np.percentile(s, 100.0 - hi, axis=0))
    else:
        bp = b0

    # ---- animal-removal refinement: mask the animal in every sample, re-take the median
    thr = max(8.0, 4.0 * sigma)
    a_min = max(12.0 * scale * scale, 0.00004 * float(valid.sum()))
    ridx = np.linspace(0, len(stack) - 1, min(refine_samples, len(stack))).astype(int)
    masks = np.zeros((len(ridx), h, w), bool)
    per_sample = []                 # [(area, cx, cy), ...] of every blob, per sample
    for j, i in enumerate(ridx):
        d = polar_diff(stack[i], bp, pol)
        comps, lab = _components(d, thr, valid, a_min, edges=edges, kappa=0.35)
        if not comps:
            continue
        keep = np.zeros((h, w), np.uint8)
        for k, a, _c in comps:
            keep[lab == k] = 1
        best = max(a for _k, a, _c in comps)
        per_sample.append([(a, float(c[0]), float(c[1])) for _k, a, c in comps])
        r = int(max(3, round(0.6 * math.sqrt(best))))
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        masks[j] = keep > 0
    # animal size = the biggest blob of each sample that is NOT a fixed spot (a flickering label, a lamp, a
    # lid edge sits in the same place in many samples; the animal moves around)
    areas = []
    if per_sample:
        allc = [(a, x, y) for s in per_sample for a, x, y in s]
        n_s = len(per_sample)

        def is_fixed(x, y, a):
            rad = max(3.0, 0.15 * math.sqrt(a))
            hits = sum(1 for s in per_sample if any(math.hypot(x - x2, y - y2) <= rad for _a2, x2, y2 in s))
            return n_s >= 8 and hits >= max(4, 0.25 * n_s)
        fixed = {(a, x, y): is_fixed(x, y, a) for a, x, y in allc} if len(allc) <= 2000 else {}
        for s in per_sample:
            moving = [a for a, x, y in s if not fixed.get((a, x, y), False)]
            if moving:
                areas.append(max(moving))
        if len(areas) < max(3, 0.15 * len(ridx)):           # the animal hardly moved: as before
            areas = [max(a for a, _x, _y in s) for s in per_sample]
    sub = stack[ridx]

    bg = np.empty_like(bp, dtype=np.float32)
    for y0 in range(0, h, 32):
        sl = slice(y0, min(h, y0 + 32))
        s = sub[:, sl].astype(np.float32)
        mk = masks[:, sl]
        if s.ndim == 4:
            mk = np.repeat(mk[..., None], s.shape[3], axis=3)
        s[mk] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(s, axis=0)
        fb = bp[sl]
        bad = ~np.isfinite(med)
        med[bad] = fb[bad]
        bg[sl] = med
    if progress is not None:
        progress(0.95)

    animal_area = None
    if len(areas) >= max(3, 0.15 * len(ridx)):
        animal_area = float(np.median(areas))
    sigma = estimate_sigma(stack, bg, valid)
    sigma_map = None
    if gamma > 1.001:
        # the boost multiplies the noise of the dark areas too: measure the noise of every place (the animal
        # masked out) so a dark, noisy floor needs a stronger signal than a bright, clean one
        acc = np.zeros((h, w), np.float64)
        cnt = np.zeros((h, w), np.float64)
        for j in range(len(ridx)):
            dd = np.abs(sub[j].astype(np.float32) - bg)
            if dd.ndim == 3:
                dd = dd.max(axis=2)
            ok_ = ~masks[j]
            acc[ok_] += dd[ok_]
            cnt[ok_] += 1
        mad = (acc / np.maximum(cnt, 1)).astype(np.float32) * 1.2533
        mad[cnt < 3] = sigma
        k = int(max(5, round(0.02 * max(h, w)))) | 1
        sigma_map = np.maximum(cv2.GaussianBlur(mad, (k, k), 0), sigma).astype(np.float32)
    model = BackgroundModel(bg, pol, sigma, animal_area, scale, str(color_mode).lower(), valid,
                            len(stack), votes, stabilize=use_stab, shake_px=shake, gamma=gamma,
                            dark_level=dark_level, sigma_map=sigma_map)
    model.polarity_warning = polarity_warning       # 'dark' / 'light': what the animal really looks like
    return model


# ----------------------------------------------------------------------------- tracker
class RobustTracker:
    """Frame-by-frame animal localisation on top of a BackgroundModel.

    Parameters are given in FULL-resolution pixels (as the user sees them)."""

    EDGE_KAPPA = 0.35
    JUMP_RULES = ("artefact", "size", "contrast", "see")
    LEARN_GUARD = True        # learn the animal's look only from a lock that moves now and then, at a plausible size
    ADAPT_MERGED = 3.0        # a blob this many times the animal's size is not protected from the background update

    def __init__(self, bg, min_contrast=12.0, min_area=15.0, max_area=0.0, max_jump=100.0,
                 exclusion_full=None, reacquire_frames=10, gain_compensation=True, core_full=None,
                 covered_full=None, fps=30.0, body_length=None, adapt_background=True, want_ends=False):
        self.bg = bg
        self.want_ends = bool(want_ends)    # also measure the two body ends (nose / tail base, tracking.body_points)
        # this tracker's own copy of the background: it follows slow light changes / a nudged camera
        # (adapt_background) without touching the shared model
        self.adapt = bool(adapt_background)
        self.bg_img = bg.image.copy() if self.adapt else bg.image
        self._last_img = None
        self._n_upd = 0
        s = bg.scale
        self.s = s
        self.min_contrast = float(min_contrast)
        self.t_core = max(float(min_contrast), 7.0 * bg.sigma)
        self.t_low = max(0.5 * float(min_contrast), 3.5 * bg.sigma, 4.0)
        self.t_low_map = self.t_low + self.EDGE_KAPPA * bg.edges
        self.t_core_map = None
        smap = getattr(bg, "sigma_map", None)
        if smap is not None:                       # low-light boost: thresholds follow the local noise
            self.t_low_map = np.maximum(self.t_low, 3.5 * smap) + self.EDGE_KAPPA * bg.edges
            self.t_core_map = np.maximum(self.t_core, 6.0 * smap).astype(np.float32)
        self.user_min_area = float(min_area) * s * s
        self.user_max_area = float(max_area) * s * s if max_area and max_area > 0 else 0.0
        self.max_jump = float(max_jump) * s
        self.reacquire_frames = int(reacquire_frames)
        self.gain_compensation = gain_compensation
        h, w = bg.image.shape[:2]
        if exclusion_full is not None:
            ex = cv2.resize(exclusion_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        else:
            ex = np.zeros((h, w), bool)
        self.excluded = ex
        self.valid = bg.valid & ~ex
        if self.valid.sum() < 50:
            self.valid = ~ex if (~ex).sum() >= 50 else np.ones((h, w), bool)
        vy, vx = np.nonzero(self.valid[::4, ::4])
        self._gain_idx = (vy * 4, vx * 4)
        self._shift = _ShiftEstimator(bg.gray) if bg.stabilize else None
        self.a_ref = bg.animal_area
        # body length measured by the user (nose -> tail base, full-res px): expected body area ~0.35 L^2.
        # A measured size always wins over the size guessed from the background samples (a partly hidden or
        # resting animal, or a reflection, makes that guess wrong).
        self.a_user = None
        self.a_guess = self.a_ref
        if body_length and float(body_length) > 2:
            self.a_user = 0.35 * (float(body_length) * s) ** 2
            self.a_ref = self.a_user
        self._update_size_params()
        # state
        self.pos = None
        self.vel = np.zeros(2)
        self.area = None
        self.n_miss = 0
        self.last_mask = None
        self.last_candidates = []
        self.last_shift = (0.0, 0.0)
        self.last_body = None          # (bool mask, x0, y0) of the chosen animal's body (tail removed), detection scale
        self.last_ends = None          # (ax, ay, bx, by, tail_ev, shape_ev) full-res, see tracking.body_points
        # ---- false-detection guards (one animal: a static spot that is not the animal never takes the lock)
        # core: the animal's centre must lie inside the zones (+ a small tolerance), not just its edge
        self.core = (cv2.resize(core_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
                     if core_full is not None else None)
        # covered zones (e.g. the lid of a light/dark box): the animal disappears there and comes back
        # next to them -- distance of every pixel to the nearest covered zone (detection px)
        self.covered_dist = None
        if covered_full is not None and np.any(covered_full):
            cov = cv2.resize(covered_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            self.covered_dist = cv2.distanceTransform((~cov).astype(np.uint8), cv2.DIST_L2, 3)
        self.fps = float(fps or 30.0)
        self.confirm_max = max(10, int(round(2.0 * self.fps)))     # a still spot is judged after ~2 s
        self.ever_locked = False
        self.lock_travel = 0.0         # distance moved since the current lock (a real animal moves)
        self.model_area = None         # typical area / contrast of the animal while it was tracked
        self.model_peak = None
        self.model_mean = None         # typical mean contrast of the animal
        self.n_model = 0
        self.last_seen = None          # last position the animal was seen at (detection px)
        self.pending = None            # candidate spots being followed before one may take the lock
        self.artefacts = []            # [x, y, frames_to_live] spots known not to be the animal
        self.first_window = max(6, int(round(1.0 * self.fps)))   # the first lock looks ~1 s ahead
        self._anchor, self._still, self._rivals = None, 0, []     # lock watchdog
        self.watch_still = max(30, int(round(3.0 * self.fps)))
        self.watch_min = max(8, int(round(0.5 * self.fps)))
        self.switches = 0
        # reflections / shadows in clear walls (see _mirror_twin)
        self.twin_frames = max(6, int(round(0.6 * self.fps)))     # a mirror pair lasts; parts of one animal flicker
        self._twin = None
        self.model_res = None           # how much floor light comes through the animal (solid: low)
        self.mirror_switches = 0
        self.jump_blocked = {}          # jumps not taken, by reason (diagnostics)
        # adaptive background: every ~1/6 s the floor away from the animal moves a little towards the
        # current picture -- slowly (~40 s) normally, quickly (~3 s) when large parts of the arena changed
        # at once (light switched / camera nudged)
        self._adapt_every = max(1, int(round(self.fps / 6.0)))
        self.tau_slow, self.tau_fast = 40.0, 3.0
        self._keep = (self.valid.astype(np.uint8) * 255)
        self._n_valid = int(np.count_nonzero(self._keep))
        self.adapt_events = 0
        self._big_count = 0
        self.rebaselines = 0
        self._static = np.zeros(self.valid.shape, np.uint16)
        self._sure = False
        self._last_rebase = -10 ** 9
        self._static_n = max(6, int(round(2.0 * self.fps / self._adapt_every)))   # ~2 s
        self._ghost = np.zeros(self.valid.shape, np.uint16)
        self._ghost_n = max(3, int(round(1.0 * self.fps / self._adapt_every)))     # ~1 s
        self.ghost_events = 0
        self.last_blob = None           # (bool mask, x0, y0) of the animal's whole blob (detection scale)

    # -- geometry derived from the expected animal size
    def _update_size_params(self):
        a = self.a_ref if self.a_ref else max(self.user_min_area * 20, 200.0 * self.s * self.s)
        self.body_w = max(3.0, math.sqrt(a / 2.4))
        self.body_l = max(6.0, 2.4 * self.body_w)
        k = int(max(3, round(0.25 * self.body_w))) | 1
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        kt = int(max(3, round(0.45 * self.body_w))) | 1
        self.k_tail = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kt, kt))
        self.min_area = max(self.user_min_area, 0.12 * self.a_ref) if self.a_ref else max(self.user_min_area, 4.0)
        if self.user_max_area > 0:
            self.max_area = self.user_max_area
        elif self.a_ref:
            self.max_area = 6.0 * self.a_ref
        else:
            h, w = self.bg.image.shape[:2]
            self.max_area = 0.25 * h * w
        self.gate = max(self.max_jump, 0.8 * self.body_l)

    # -- one frame
    def prepare_frame(self, frame_bgr):
        """Down-scale + grey/blur one frame (thread-safe; the reader thread does this ahead of time)."""
        return prepare(frame_bgr, self.bg.color_mode, self.s, getattr(self.bg, "gamma", 1.0))

    def strength(self, frame_bgr, prepared=None):
        base = prepared if prepared is not None else self.prepare_frame(frame_bgr)
        img = None                 # float copy -- made only when the frame must be shifted / gain-corrected
        if self._shift is not None:
            img = base.astype(np.float32)
            dx, dy = self._shift(to_gray_f(img))
            self.last_shift = (dx, dy)
            img = shift_image(img, dx, dy)
        if self.gain_compensation and self._gain_idx[0].size > 50:
            iy, ix = self._gain_idx
            src = img if img is not None else base
            chans = [None] if src.ndim == 2 else list(range(src.shape[2]))
            for c in chans:
                src = img if img is not None else base
                b = self.bg_img[iy, ix] if c is None else self.bg_img[iy, ix, c]
                f = (src[iy, ix] if c is None else src[iy, ix, c]).astype(np.float32)
                ok = b > 20
                if ok.sum() > 50:
                    g = float(np.median(f[ok] / b[ok]))
                    if 0.6 < g < 1.6 and abs(g - 1.0) > 0.004:
                        if img is None:
                            img = base.astype(np.float32)
                        if c is None:
                            img /= g
                        else:
                            img[..., c] /= g
        self._last_img = img if img is not None else base
        if img is None and base.ndim == 2:
            d = _polar_diff_u8(base, self.bg_img, self.bg.polarity)     # same values, without a float copy
        else:
            d = polar_diff(img if img is not None else base.astype(np.float32), self.bg_img, self.bg.polarity)
        if getattr(self, "_invalid_of", None) is not self.valid:
            self._invalid, self._invalid_of = ~self.valid, self.valid
        np.copyto(d, 0, where=self._invalid)
        return d

    def _describe(self, sub_lab, sub_d, x, y):
        """Stats of one component (cropped label mask + strength)."""
        vals = sub_d[sub_lab]
        a = float(sub_lab.sum())
        # tail removed: an opening with a disc a little thinner than THIS blob's body (its widest part)
        # takes away the thin tail but keeps the body, whatever the animal's size
        u8 = sub_lab.astype(np.uint8)
        r_in = float(cv2.distanceTransform(np.pad(u8, 1), cv2.DIST_L2, 3).max()) if a >= 20 else 0.0
        if r_in >= 2.5:
            kr = max(1, int(round(0.55 * r_in)))
            body = cv2.morphologyEx(u8, cv2.MORPH_OPEN,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1))).astype(bool)
        else:
            body = cv2.morphologyEx(u8, cv2.MORPH_OPEN, self.k_tail).astype(bool)
        if body.sum() < 0.3 * a:
            body = sub_lab
        wts = sub_d * body
        tot = float(wts.sum())
        if tot <= 0 or a <= 0:
            return None
        bh, bw = sub_lab.shape
        yy, xx = np.mgrid[0:bh, 0:bw]
        return {"cx": float((wts * xx).sum() / tot) + x, "cy": float((wts * yy).sum() / tot) + y,
                "area": a, "peak": float(vals.max()), "mass": float(vals.sum()), "bbox": (x, y, bw, bh),
                "body": body, "full": sub_lab}

    def _split_oversized(self, sub_lab, sub_d, x, y):
        """A blob too big to be the animal (merged with a shadow / reflection):
        look for animal-sized cores inside it, strongest contrast first (Otsu,
        then a descending fraction of the peak)."""
        vals = sub_d[sub_lab]
        if vals.size == 0:
            return []
        peak = float(vals.max())
        thresholds = []
        v8 = np.clip(vals, 0, 255).astype(np.uint8).reshape(-1, 1)
        otsu, _ = cv2.threshold(v8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if otsu > self.t_low:
            thresholds.append(float(otsu))
        thresholds += [peak * f for f in (0.7, 0.55, 0.4, 0.3)]
        for t in thresholds:
            if t <= self.t_low:
                continue
            m = ((sub_d > t) & sub_lab).astype(np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
            out = []
            for k in range(1, n):
                a = float(stats[k, cv2.CC_STAT_AREA])
                if not (self.min_area <= a <= self.max_area):
                    continue
                bx, by, bw, bh = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                             cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
                if not blob_shape_ok(a, bw, bh, self.body_w if self.a_ref else None):
                    continue
                desc = self._describe(lab[by:by + bh, bx:bx + bw] == k, sub_d[by:by + bh, bx:bx + bw],
                                      x + bx, y + by)
                if desc is not None and desc["peak"] >= self._t_core_at(desc):
                    desc["split"] = True
                    out.append(desc)
            if out:
                return out
        return []

    def candidates(self, d):
        m = cv2.compare(d, self.t_low_map, cv2.CMP_GT)          # 0 / 255 (same blobs as 0 / 1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.k_close)
        self.last_mask = m
        n, lab, stats, _c = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = []
        if n <= 1:
            return out
        areas = stats[1:, cv2.CC_STAT_AREA]
        order = np.argsort(-areas)[:12] + 1
        for k in order:
            a = float(stats[k, cv2.CC_STAT_AREA])
            if a < self.min_area * 0.5:
                break
            x, y, bw, bh = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                       cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            sub_lab = lab[y:y + bh, x:x + bw] == k
            sub_d = d[y:y + bh, x:x + bw]
            if a > self.max_area:
                out.extend(self._split_oversized(sub_lab, sub_d, x, y))
                continue
            if a < self.min_area:
                continue
            if not blob_shape_ok(a, bw, bh, self.body_w if self.a_ref else None):
                continue
            desc = self._describe(sub_lab, sub_d, x, y)
            if desc is None or desc["peak"] < self._t_core_at(desc):
                continue
            out.append(desc)
        return out

    def _t_core_at(self, c):
        """Peak contrast a spot needs here (the local noise counts when the low-light boost is on)."""
        if self.t_core_map is None:
            return self.t_core
        yi = int(min(max(round(c["cy"]), 0), self.t_core_map.shape[0] - 1))
        xi = int(min(max(round(c["cx"]), 0), self.t_core_map.shape[1] - 1))
        return float(self.t_core_map[yi, xi])

    def _area_like(self, a):
        if not self.a_ref:
            return 1.0
        z = math.log(max(a, 1e-6) / self.a_ref) / 0.8
        return math.exp(-0.5 * z * z)

    def _pick_global(self, cands):
        best, best_s = None, -1.0
        for c in cands:
            s = c["mass"] * (0.15 + self._area_like(c["area"]))
            if s > best_s:
                best, best_s = c, s
        return best

    # -- false-detection guards ------------------------------------------------------------------------
    def _in_core(self, c):
        if self.core is None:
            return True
        yi = int(min(max(round(c["cy"]), 0), self.core.shape[0] - 1))
        xi = int(min(max(round(c["cx"]), 0), self.core.shape[1] - 1))
        return bool(self.core[yi, xi])

    def _plausible(self, c):
        """Looks enough like the animal seen so far (size and contrast)."""
        if not self.model_area:
            return self._area_like(c["area"]) > 0.25 or not self.a_ref
        return (0.3 * self.model_area <= c["area"] <= 3.5 * self.model_area
                and c["peak"] >= 0.4 * self.model_peak)

    def _looks_like_animal(self, c):
        if not self.model_area:
            return bool(self.a_ref) and 0.5 * self.a_ref <= c["area"] <= 2.0 * self.a_ref
        return (0.55 * self.model_area <= c["area"] <= 1.8 * self.model_area
                and c["peak"] >= 0.6 * self.model_peak)

    def _near_covered(self, c):
        if self.covered_dist is None:
            return False
        yi = int(min(max(round(c["cy"]), 0), self.covered_dist.shape[0] - 1))
        xi = int(min(max(round(c["cx"]), 0), self.covered_dist.shape[1] - 1))
        return float(self.covered_dist[yi, xi]) <= 1.2 * self.body_l

    def _is_artefact(self, c):
        r = max(4.0, 0.7 * self.body_w)
        return any(t > 0 and math.hypot(c["cx"] - x, c["cy"] - y) <= r for x, y, t in self.artefacts)

    def _remember_artefact(self, c, frames):
        r = max(4.0, 0.7 * self.body_w)
        for a in self.artefacts:
            if math.hypot(c["cx"] - a[0], c["cy"] - a[1]) <= r:
                a[0], a[1], a[2] = c["cx"], c["cy"], max(a[2], frames)
                return
        if len(self.artefacts) < 64:
            self.artefacts.append([c["cx"], c["cy"], frames])

    def _follow(self, tracks, cands, max_tracks=6):
        """Follow candidate spots from frame to frame (short tracklets).  A tracklet that is not seen in a
        frame is dropped, so its 'track' always holds the positions of the frames just before this one."""
        new_tracks, used = [], set()
        for tr in tracks:
            best, bd = None, None
            for k, c in enumerate(cands):
                if k in used:
                    continue
                dd = math.hypot(c["cx"] - tr["pos"][0], c["cy"] - tr["pos"][1])
                if dd <= self.gate and (bd is None or dd < bd):
                    best, bd = k, dd
            if best is None:
                continue
            used.add(best)
            c = cands[best]
            tr["track"].append(tr["pos"] + (tr["c"]["area"],))
            tr["pos"] = (c["cx"], c["cy"])
            tr["c"] = c
            tr["mass"] = 0.8 * tr["mass"] + 0.2 * c["mass"]
            tr["moved"] = max(tr["moved"], math.hypot(c["cx"] - tr["p0"][0], c["cy"] - tr["p0"][1]))
            new_tracks.append(tr)
        for k, c in enumerate(cands):
            if k not in used and len(new_tracks) < max_tracks:
                new_tracks.append({"p0": (c["cx"], c["cy"]), "pos": (c["cx"], c["cy"]), "moved": 0.0,
                                   "track": [], "c": c, "mass": c["mass"]})
        return new_tracks

    def _score(self, tr):
        """Animal-likeness of a tracklet: strong, animal-sized AND moving (a still spot scores far lower)."""
        motion = 0.15 + min(1.0, tr["moved"] / max(1e-6, 0.5 * self.body_l))
        contrast = tr["mass"] / max(1.0, tr["c"]["area"])          # mean contrast (not just 'bigger')
        return contrast * (0.1 + self._area_like(tr["c"]["area"]) ** 2) * motion

    def _reacquire(self, cands):
        """Find the animal (at the start, or again after it was lost).
        Next to where it was last seen, or next to a covered zone it can come out of: taken at once.
        Otherwise the candidate spots are followed for a short while and the one that MOVES wins -- a
        still spot (a shadow, a label, a lid edge, a lamp reflection) never takes the lock just because it
        is stronger than the animal on the first frame.  Returns (chosen | None, backfill list)."""
        plaus = [c for c in cands if self._plausible(c) and not self._is_artefact(c)]
        quick = []
        if self.ever_locked:
            for c in plaus:
                near_last = (self.last_seen is not None and
                             math.hypot(c["cx"] - self.last_seen[0], c["cy"] - self.last_seen[1])
                             <= max(2.0 * self.body_l, self.gate))
                if near_last or self._near_covered(c):
                    quick.append(c)
        if quick:
            self.pending = None
            return self._pick_global(quick), []
        tracks = self._follow(self.pending or [], plaus if (plaus or self.ever_locked) else cands)
        if not tracks:
            self.pending = None
            return None, []
        if not self.ever_locked:
            # the first lock: wait up to ~1 s; the first spot that clearly moves is the animal, otherwise
            # the best by strength x size x movement (as strong as before when nothing moves)
            oldest = max(len(t["track"]) for t in tracks)
            pick = None
            ranked = sorted(tracks, key=self._score, reverse=True)
            lead = ranked[0]
            # early: the best spot is clearly moving AND clearly the best (a swinging cable or a shadow
            # moves too, but is weaker than the animal)
            if (lead["moved"] >= 0.6 * self.body_l and len(lead["track"]) >= 3 and
                    (len(ranked) == 1 or self._score(lead) >= 2.0 * self._score(ranked[1]))):
                pick = lead
            elif oldest >= self.first_window:
                ripe = [t for t in tracks if len(t["track"]) >= max(2, self.first_window // 3)] or tracks
                pick = max(ripe, key=self._score)
            if pick is not None:
                self.pending = None
                return pick["c"], list(pick["track"])
            self.pending = tracks
            return None, []
        confirmed, keep = [], []
        for tr in tracks:
            c = tr["c"]
            if tr["moved"] >= 0.35 * self.body_l or (len(tr["track"]) >= self.confirm_max and
                                                       self._looks_like_animal(c)):
                confirmed.append(tr)
            elif len(tr["track"]) >= self.confirm_max:
                self._remember_artefact(c, int(10 * self.fps))      # still for 2 s and not like the animal
            else:
                keep.append(tr)
        if confirmed:
            best = max(confirmed, key=self._score)
            self.pending = None
            return best["c"], list(best["track"])
        self.pending = keep or None
        return None, []

    def _watchdog(self, chosen, cands):
        """While locked: if the lock has not moved for a few seconds but another animal-like spot keeps
        moving, the lock is on something that is not the animal (it can happen when the animal was hard
        to see at the start).  Returns the spot to switch to (+ its recent positions) or (None, [])."""
        p = (chosen["cx"], chosen["cy"])
        if self._anchor is None or math.hypot(p[0] - self._anchor[0], p[1] - self._anchor[1]) > 0.3 * self.body_l:
            self._anchor, self._still = p, 0
        else:
            self._still += 1
        long_still = self._still >= self.watch_still and self.a_ref
        rivals = [c for c in cands if c is not chosen and not self._is_artefact(c)
                  and (self._plausible(c) or (long_still and 0.45 * self.a_ref <= c["area"] <= 2.2 * self.a_ref))
                  and math.hypot(c["cx"] - p[0], c["cy"] - p[1]) > 1.5 * self.body_l]
        self._rivals = self._follow(self._rivals, rivals, max_tracks=4)
        if self._still < self.watch_still:
            return None, []
        best = None
        # the lock can have slipped onto a still spot while the animal walked over it (a new shadow, a dark
        # object, a stain): the size / contrast learned since then are that spot's, so a rival is also
        # accepted when it looks like the animal's size from the background (a_ref) and its mean contrast is
        # at least half the lock's
        lock_mean = chosen["mass"] / max(1.0, chosen["area"])
        for tr in self._rivals:
            c = tr["c"]
            if tr["moved"] < 1.2 * self.body_l or len(tr["track"]) < self.watch_min:
                continue
            like_model = self._looks_like_animal(c) and tr["mass"] >= 0.5 * chosen["mass"]
            like_ref = (self.a_ref is not None and 0.45 * self.a_ref <= c["area"] <= 2.2 * self.a_ref
                        and c["mass"] / max(1.0, c["area"]) >= 0.5 * lock_mean
                        and self._still >= 2 * self.watch_still)
            if like_model or like_ref:
                if best is None or self._score(tr) > self._score(best):
                    best = tr
        if best is None:
            return None, []
        self._remember_artefact(chosen, int(10 * self.fps))
        self._rivals, self._anchor, self._still = [], None, 0
        self.switches += 1
        return best["c"], list(best["track"])

    # -- reflections and shadows in clear (acrylic) walls --------------------------------------------------
    def _residual(self, c):
        """How much of the floor's light still comes through a spot: dark animal = its grey / the floor's grey
        there (light animal: the same, counted from white).  A solid animal blocks the floor (about 0.05-0.1);
        a reflection or a shadow in a clear wall lets the floor through (the grid bars show), about 2-3x more.
        None when it cannot be measured."""
        img = self._last_img
        if img is None or self.bg.polarity not in ("dark", "light") or c.get("body") is None:
            return None
        if "res" in c:
            return c["res"]
        x, y, bw, bh = c["bbox"]
        core = c["body"]
        if core.sum() >= 40:
            e = cv2.erode(core.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if e.sum() >= 0.3 * core.sum():
                core = e
        im = img[y:y + bh, x:x + bw]
        bg = self.bg_img[y:y + bh, x:x + bw]
        if im.ndim == 3:
            im, bg = im.mean(axis=2), bg.mean(axis=2)
        if im.shape != core.shape or not core.any():
            return None
        a, b = im[core].astype(np.float32), bg[core].astype(np.float32)
        rr = (a + 2.0) / (b + 2.0) if self.bg.polarity == "dark" else (257.0 - a) / np.maximum(1.0, 257.0 - b)
        # the most solid part of the spot (a quarter of it): a shadow or the floor next to the animal that
        # is part of the blob does not make a solid animal look see-through
        r = float(np.percentile(rr, 25))
        c["res"] = r
        return r

    @staticmethod
    def _covers(c, p, pad=2):
        """True when the blob of candidate c covers the point p (detection px): the same animal that changed its
        shape (turned towards the camera, stretched, merged with its shadow), not a jump to another spot."""
        if p is None or c.get("full") is None:
            return False
        x, y, bw, bh = c["bbox"]
        px, py = int(round(p[0])) - x, int(round(p[1])) - y
        if not (-pad <= px < bw + pad and -pad <= py < bh + pad):
            return False
        y0, y1 = max(0, py - pad), min(bh, py + pad + 1)
        x0, x1 = max(0, px - pad), min(bw, px + pad + 1)
        return bool(y1 > y0 and x1 > x0 and c["full"][y0:y1, x0:x1].any())

    def _overlaps_last(self, c):
        """True when candidate c overlaps the animal's blob of the frame before (same animal, new shape)."""
        lb = self.last_blob
        if lb is None or c.get("full") is None:
            return False
        bm, bx, by = lb
        x, y, bw, bh = c["bbox"]
        x0, y0 = max(x, bx), max(y, by)
        x1, y1 = min(x + bw, bx + bm.shape[1]), min(y + bh, by + bm.shape[0])
        if x1 <= x0 or y1 <= y0:
            return False
        return bool((c["full"][y0 - y:y1 - y, x0 - x:x1 - x] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).any())

    def _mirror_twin(self, chosen, cands):
        """A reflection (or a shadow) in a clear wall moves exactly with the animal and has its size, so the lock
        can slip onto it when the two touch and part again.  When a clearly more solid spot of about the same
        size stays next to the lock for a few frames, that one is the animal -> it is returned (else None)."""
        self._twin_near = False
        if self.a_ref and not chosen["area"] <= 2.5 * self.a_ref:
            return None                  # the lock is not animal-sized right now (a light change, a hand ...)
        near = [c for c in cands if c is not chosen and not self._is_artefact(c)
                and (not self.a_ref or 0.4 * self.a_ref <= c["area"] <= 2.0 * self.a_ref)
                and math.hypot(c["cx"] - chosen["cx"], c["cy"] - chosen["cy"]) <= 3.0 * self.body_l
                and 0.65 * chosen["area"] <= c["area"] <= 1.5 * chosen["area"]
                and (self.model_area is None or 0.6 * self.model_area <= c["area"] <= 1.6 * self.model_area)]
        if not near:
            self._twin = None
            return None
        self._twin_near = True
        r0 = self._residual(chosen)
        if r0 is None:
            return None
        best = None
        for c in near:
            r = self._residual(c)
            if r is not None and r <= 0.75 * r0 and r0 - r >= 0.04 and r <= 0.45:
                if best is None or r < best[1]:
                    best = (c, r)
        if best is None:
            self._twin = None
            return None
        c = best[0]
        if self._twin is not None and math.hypot(c["cx"] - self._twin[0], c["cy"] - self._twin[1]) <= self.gate:
            self._twin = (c["cx"], c["cy"], self._twin[2] + 1)
        else:
            self._twin = (c["cx"], c["cy"], 1)
        if self._twin[2] < self.twin_frames:
            return None
        self._twin = None
        self.mirror_switches += 1
        return c

    def update(self, frame_bgr, prepared=None):
        """Returns dict(x, y, area, status, n_candidates[, backfill, ends]) in full-res pixels (see _update);
        then the background follows slow changes of the arena (adapt_background)."""
        out = self._update(frame_bgr, prepared)
        if self.adapt:
            try:
                self._adapt_background(out)
            except Exception:
                pass
        return out

    def _adapt_background(self, out):
        """Every ~1/6 s: (1) the floor follows slow light drift (only where nothing stands out yet, ~40 s),
        (2) a place that has stood out for ~2 s without being the animal (the light changed there, the camera
        was nudged, a shadow / dropping that stays) becomes floor."""
        self._n_upd += 1
        if self._n_upd % self._adapt_every or self._last_img is None:
            return
        m = self.last_mask
        if m is None:
            return
        tracked = out.get("status") == "Tracked" and self.pos is not None
        if tracked and self.lock_travel >= 2.0 * self.body_l:
            self._sure = True            # this lock has moved: it is the animal
        elif tracked:
            self._sure = False           # a new lock that has not moved yet (could be a still spot)
        if not self._sure:
            return
        if tracked:
            animal = np.zeros(m.shape, np.uint8)
            blob = self.last_blob
            ref = self.model_area or self.a_ref
            if blob is not None and ref and self.ADAPT_MERGED and \
                    float(blob[0].sum()) > self.ADAPT_MERGED * max(ref, self.a_ref or 0.0):
                blob = None              # merged with a patch of changed light / a shadow: only the animal is kept
            if blob is not None:
                bm, bx, by = blob
                animal[by:by + bm.shape[0], bx:bx + bm.shape[1]][bm] = 255
                k = int(max(3, round(0.3 * self.body_w))) | 1
                animal = cv2.dilate(animal, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            cv2.circle(animal, (int(round(self.pos[0])), int(round(self.pos[1]))), int(round(0.6 * self.body_l)),
                       255, -1)
            limit = self._static_n
        elif self.last_seen is not None:
            animal = np.zeros(m.shape, np.uint8)  # not seen right now: keep away from where it was last seen
            c = self.last_seen
            cv2.circle(animal, (int(round(c[0])), int(round(c[1]))), int(round(2.0 * self.body_l)), 255, -1)
            limit = 2 * self._static_n
        else:
            return                       # the animal was never found yet: leave the floor as it is
        img = self._last_img
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        fg = (m > 0) & self.valid
        an = animal > 0
        stand = fg & ~an
        cnt = self._static
        cnt[stand] += 1
        cnt[~stand] = 0
        rep = cnt >= limit
        if rep.any():
            self.adapt_events += 1
            np.copyto(self.bg_img, img, where=rep if img.ndim == 2 else rep[..., None])
            cnt[rep] = 0
        # a place that now differs the OTHER way (lighter than the floor for a dark animal): something that was
        # there when the floor was learned has gone -- usually the animal itself, resting while the background
        # was learned (its "ghost").  The animal never looks like that, so after ~1 s it becomes floor, away from
        # the animal and its shadow.  Without this the ghost fades only slowly (~40 s) and an animal coming back
        # to the same place stays hard to see.
        pol = getattr(self.bg, "polarity", "any")
        c = self.pos if tracked else self.last_seen
        if pol in ("dark", "light") and c is not None:
            od = polar_diff(img, self.bg_img, "light" if pol == "dark" else "dark")
            ghost = (od > self.t_low_map) & self.valid
            near = np.zeros(m.shape, np.uint8)
            cv2.circle(near, (int(round(c[0])), int(round(c[1]))), int(round(2.0 * self.body_l)), 255, -1)
            ghost &= near == 0
            ref = self.model_area or self.a_ref
            if not ref or int(np.count_nonzero(ghost)) > 3.0 * ref:
                ghost[:] = False             # no animal size yet, or the light changed: the slow ways above
            elif ghost.any():                # only animal-sized places
                n_g, lab, stats, _c = cv2.connectedComponentsWithStats(ghost.astype(np.uint8), connectivity=8)
                keep_g = stats[:, cv2.CC_STAT_AREA] <= 3.0 * ref
                keep_g[0] = False
                ghost = keep_g[lab]
            gc = self._ghost
            gc[ghost] += 1
            gc[~ghost] = 0
            rg = gc >= self._ghost_n
            if rg.any():
                self.ghost_events += 1
                np.copyto(self.bg_img, img, where=rg if img.ndim == 2 else rg[..., None])
                gc[rg] = 0
        # slow drift
        keep = self._keep.copy()
        keep[fg | an] = 0
        alpha = float(min(0.5, self._adapt_every / (self.tau_slow * self.fps)))
        cv2.accumulateWeighted(img, self.bg_img, alpha, mask=keep)

    def _rebaseline(self):
        """Large parts of the arena changed at once (light switched on/off, camera nudged): take the current
        picture as the floor everywhere except around where the animal is (it stays visible there); the
        rest of the old floor around it is replaced as soon as the animal walks on."""
        m = self.last_mask
        if m is None or self._last_img is None or not self._sure:
            return False                 # only when it is known where the animal is (a lock that has moved)
        if self._n_upd - self._last_rebase < self.fps:
            return False                 # at most once a second
        c = self.pos if self.pos is not None else self.last_seen
        if c is None:
            return False
        keep = self._keep.copy()
        cv2.circle(keep, (int(round(c[0])), int(round(c[1]))), int(round(2.5 * self.body_l)), 0, -1)
        out_fg = int(cv2.countNonZero(cv2.bitwise_and(m, keep)))
        if out_fg <= 0.15 * max(1, int(cv2.countNonZero(keep))):
            return False
        self._last_rebase = self._n_upd
        img = self._last_img
        if img.dtype != np.float32:
            img = img.astype(np.float32)
        np.copyto(self.bg_img, img, where=(keep > 0) if img.ndim == 2 else (keep > 0)[..., None])
        self.rebaselines += 1
        return True

    def _update(self, frame_bgr, prepared=None):
        """Returns dict(x, y, area, status, n_candidates[, backfill, ends]) in full-res pixels.
        backfill = [(x, y, area), ...] for the frames just before this one, reported 'Lost' while the
        animal was being confirmed -- the caller may fill them in."""
        d = self.strength(frame_bgr, prepared)
        cands = self.candidates(d)
        if self.adapt and self._rebaseline():
            d = self.strength(frame_bgr, prepared)      # the light changed at once: look again
            cands = self.candidates(d)
        self._twin_near = False
        if self.core is not None:
            cands = [c for c in cands if self._in_core(c)]
        self.last_candidates = cands
        if self.artefacts:
            for a in self.artefacts:
                a[2] -= 1
            self.artefacts = [a for a in self.artefacts if a[2] > 0]
        chosen, backfill, override = None, [], False
        if self.pos is None:
            chosen, backfill = self._reacquire(cands)
        elif self.n_miss >= self.reacquire_frames:
            chosen, backfill = self._reacquire(cands)
        else:
            steps = self.n_miss + 1
            pred = self.pos + self.vel * steps
            # the search circle grows while the animal is missing, but never beyond what it can walk; it
            # does not grow at all when it vanished next to a covered zone (it is in there)
            radius = min(self.gate * (1.0 + 0.5 * self.n_miss), self.gate + 2.0 * self.body_l)
            if self.n_miss and self.covered_dist is not None:
                yi = int(min(max(round(self.pos[1]), 0), self.covered_dist.shape[0] - 1))
                xi = int(min(max(round(self.pos[0]), 0), self.covered_dist.shape[1] - 1))
                if float(self.covered_dist[yi, xi]) <= 1.2 * self.body_l:
                    radius = self.gate
            inside = []
            for c in cands:
                dist = math.hypot(c["cx"] - pred[0], c["cy"] - pred[1])
                dist0 = math.hypot(c["cx"] - self.pos[0], c["cy"] - self.pos[1])
                dd = min(dist, dist0)
                if dd <= radius:
                    cost = dd / radius
                    if self.area:
                        cost += 0.5 * abs(math.log(max(c["area"], 1e-6) / self.area))
                    cost -= 0.25 * self._area_like(c["area"])
                    if dd > 1.5 * self.body_l and not self._covers(c, self.pos) and not self._overlaps_last(c):
                        # a jump onto a spot known not to be the animal, onto a see-through one (a reflection,
                        # a shadow, a changed patch of floor) while the animal is solid, or onto one clearly
                        # unlike the animal (twice its size, or far stronger -- a hand just leaving, a lid): not
                        # taken
                        why = None
                        G = self.JUMP_RULES
                        if "artefact" in G and self._is_artefact(c):
                            why = "artefact"
                        elif "size" in G and self.model_area and c["area"] > 1.8 * self.model_area:
                            why = "size"
                        elif "contrast" in G and self.model_mean and \
                                c["mass"] / max(1.0, c["area"]) > 2.2 * self.model_mean:
                            why = "contrast"
                        elif "see" in G and self.model_res is not None and self.model_res <= 0.35:
                            r = self._residual(c)
                            if r is not None and r > 1.8 * self.model_res + 0.08:
                                why = "see-through"
                        if why is not None:
                            self.jump_blocked[why] = self.jump_blocked.get(why, 0) + 1
                            continue
                    inside.append((cost, c))
            if inside:
                chosen = min(inside, key=lambda t: t[0])[1]
                if self.n_miss == 0:
                    sw, sw_back = self._watchdog(chosen, cands)
                    if sw is not None:                # the lock was on a still spot: take the moving animal
                        chosen, backfill, override = sw, sw_back, True
                        self.pos = None               # a new lock (no velocity carried over)
                    else:
                        tw = self._mirror_twin(chosen, cands)
                        if tw is not None:            # the lock was on the see-through copy: take the solid one
                            chosen = tw
                            self.pos = None
        if chosen is None:
            self.last_body = None
            self.last_blob = None
            self.last_ends = None
            self.n_miss += 1
            if self.n_miss > self.reacquire_frames * 6:
                self.pos = None            # fully re-acquire from scratch
                self.vel[:] = 0
            return {"x": np.nan, "y": np.nan, "area": np.nan, "status": "Lost", "n_candidates": len(cands)}
        p = np.array([chosen["cx"], chosen["cy"]])
        relock = self.pos is None or self.n_miss >= self.reacquire_frames
        if self.pos is not None and self.n_miss < self.reacquire_frames:
            v = (p - self.pos) / float(self.n_miss + 1)
            vmax = self.gate
            nv = float(np.hypot(*v))
            if nv > vmax:
                v *= vmax / nv
            self.vel = 0.6 * self.vel + 0.4 * v
            self.lock_travel += float(np.hypot(*(p - self.pos)))
        else:
            self.vel[:] = 0
        if relock:
            self.lock_travel = 0.0
        self.pending = None
        self.pos = p
        self.last_seen = (float(p[0]), float(p[1]))
        self.ever_locked = True
        bx0, by0 = chosen["bbox"][0], chosen["bbox"][1]
        self.last_body = (chosen["body"], int(bx0), int(by0)) if chosen.get("body") is not None else None
        self.last_blob = (chosen["full"], int(bx0), int(by0)) if chosen.get("full") is not None else None
        if not chosen.get("split"):
            self.area = chosen["area"] if self.area is None else 0.8 * self.area + 0.2 * chosen["area"]
        moving_lock = self.lock_travel >= 2.0 * self.body_l
        # the look of the animal is learned only while the lock moves now and then: a lock that has sat still for
        # a few seconds may have slipped onto a still spot (a stain, a changed patch of floor) -- its size and
        # contrast must not become 'the animal'
        learn = moving_lock and (not self.LEARN_GUARD or (self._still < self.watch_still and (
            not self.a_ref or 0.3 * self.a_ref <= chosen["area"] <= 3.0 * self.a_ref)))
        if self.n_miss == 0 and not chosen.get("split") and learn and not self._twin_near:
            r = self._residual(chosen)
            if r is not None and (self.model_area is None or 0.6 * self.model_area <= chosen["area"]
                                  <= 1.5 * self.model_area):
                self.model_res = r if self.model_res is None else 0.95 * self.model_res + 0.05 * r
        if self.n_miss == 0 and not chosen.get("split") and learn:
            # what the animal looks like (for re-acquisition); only from a lock that has really moved
            a = 0.1 if self.n_model < 30 else 0.02
            self.model_area = chosen["area"] if self.model_area is None else (1 - a) * self.model_area + a * chosen["area"]
            self.model_peak = chosen["peak"] if self.model_peak is None else (1 - a) * self.model_peak + a * chosen["peak"]
            mc = chosen["mass"] / max(1.0, chosen["area"])
            self.model_mean = mc if self.model_mean is None else (1 - a) * self.model_mean + a * mc
            self.n_model += 1
        if moving_lock and self.n_miss == 0:
            # one animal: a clearly weaker spot seen while the animal is HERE is not the animal
            lock_c = chosen["mass"] / max(1.0, chosen["area"])
            for c in cands:
                if c is not chosen and math.hypot(c["cx"] - p[0], c["cy"] - p[1]) > 1.5 * self.body_l \
                        and self._plausible(c) and c["mass"] / max(1.0, c["area"]) < 0.6 * lock_c:
                    self._remember_artefact(c, int(5 * self.fps))
        self.n_miss = 0
        s = self.s
        x = (chosen["cx"] + 0.5) / s - 0.5
        y = (chosen["cy"] + 0.5) / s - 0.5
        self.last_ends = None          # body centre only, unless the nose is wanted (object exploration)
        if self.want_ends and chosen.get("body") is not None:
            try:
                from tracking.body_points import body_ends
                e = body_ends(chosen.get("full"), chosen["body"], bx0, by0)
                if e is not None:
                    ax, ay, bx_, by_, t_ev, s_ev = e
                    self.last_ends = ((ax + 0.5) / s - 0.5, (ay + 0.5) / s - 0.5, (bx_ + 0.5) / s - 0.5,
                                      (by_ + 0.5) / s - 0.5, t_ev, s_ev)
            except Exception:
                self.last_ends = None
        # (with camera-shake compensation the position stays in the apparatus-fixed
        #  background frame, so shake does not add fake distance)
        out = {"x": float(x), "y": float(y), "area": float(chosen["area"] / (s * s)), "status": "Tracked",
               "n_candidates": len(cands), "ends": self.last_ends, "peak": float(chosen["peak"]),
               "contrast": float(chosen["mass"] / max(1.0, chosen["area"]))}
        if backfill:
            out["backfill"] = [((bx + 0.5) / s - 0.5, (by + 0.5) / s - 0.5, ba / (s * s)) for bx, by, ba in backfill]
            if override:
                out["backfill_override"] = True      # these frames were on the wrong spot: replace them
        return out

    # -- for live views / detection preview
    def overlay(self, frame_bgr, color=(0, 200, 255)):
        """Tint the current foreground mask onto a full-res frame copy."""
        out = frame_bgr.copy()
        if self.last_mask is None:
            return out
        m = self.last_mask
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = m > 0
        out[sel] = (0.45 * out[sel] + 0.55 * np.array(color, np.float32)).astype(np.uint8)
        return out


class ClassicTracker:
    """Classic centre-of-mass algorithm: reference image,
    optional prior-location window weighting, percentile threshold and
    centre of mass of the remaining difference -- with one fix: the window
    is dropped when there is no signal so it can re-acquire."""

    def __init__(self, bg, loc_thresh=99.5, use_window=True, window_size=200, window_weight=0.9,
                 exclusion_full=None):
        self.bg = bg
        self.s = bg.scale
        self.loc_thresh = float(np.clip(loc_thresh, 50.0, 99.99))
        self.use_window = bool(use_window)
        self.window = int(window_size * self.s)
        self.weight = float(window_weight)
        h, w = bg.image.shape[:2]
        ex = (cv2.resize(exclusion_full.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
              if exclusion_full is not None else np.zeros((h, w), bool))
        self.valid = bg.valid & ~ex
        self.pos = None
        self.last_mask = None
        self.last_candidates = []
        self.last_body = None

    def prepare_frame(self, frame_bgr):
        return prepare(frame_bgr, self.bg.color_mode, self.s, getattr(self.bg, "gamma", 1.0))

    def update(self, frame_bgr, prepared=None):
        img = prepared if prepared is not None else self.prepare_frame(frame_bgr)
        d = polar_diff(img, self.bg.image, "any")
        d[~self.valid] = 0
        if self.pos is not None and self.use_window:
            px, py = int(self.pos[0]), int(self.pos[1])
            hw = self.window // 2
            wts = np.full(d.shape, 1.0 - self.weight, np.float32)
            wts[max(0, py - hw):py + hw, max(0, px - hw):px + hw] = 1.0
            d = d * wts
        vals = d[self.valid]
        if vals.size == 0 or float(vals.max()) <= 0:
            self.pos = None
            self.last_body = None
            return {"x": np.nan, "y": np.nan, "area": np.nan, "status": "Lost", "n_candidates": 0}
        thr = float(np.percentile(vals, self.loc_thresh))
        sel = (d > thr) & self.valid
        self.last_mask = sel.astype(np.uint8)
        tot = float(d[sel].sum())
        if tot <= 0:
            self.pos = None
            self.last_body = None
            return {"x": np.nan, "y": np.nan, "area": np.nan, "status": "Lost", "n_candidates": 0}
        ys, xs = np.nonzero(sel)
        w_ = d[ys, xs]
        cx, cy = float((xs * w_).sum() / tot), float((ys * w_).sum() / tot)
        self.pos = np.array([cx, cy])
        r = int(max(4, 1.5 * math.sqrt(float(sel.sum()))))          # the selected pixels around the animal
        y0, x0 = max(0, int(cy) - r), max(0, int(cx) - r)
        self.last_body = (sel[y0:int(cy) + r + 1, x0:int(cx) + r + 1].copy(), x0, y0)
        s = self.s
        return {"x": (cx + 0.5) / s - 0.5, "y": (cy + 0.5) / s - 0.5, "area": float(sel.sum() / (s * s)),
                "status": "Tracked", "n_candidates": 1}

    def overlay(self, frame_bgr, color=(0, 200, 255)):
        out = frame_bgr.copy()
        if self.last_mask is None:
            return out
        m = self.last_mask
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
        sel = m > 0
        out[sel] = (0.45 * out[sel] + 0.55 * np.array(color, np.float32)).astype(np.uint8)
        return out
