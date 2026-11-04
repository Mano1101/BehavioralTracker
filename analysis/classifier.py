"""Trainable behaviour classifier: learns behaviours (grooming, rearing, freezing, digging ...) from videos scored by
hand on the Event Scoring page, then finds them in new videos.

Features (per frame, from BT's own analysis of the video -- no GPU, no pose model needed):
  speed of the body centre, motion of the whole body / head / legs / tail (body-outline points), motion around the
  animal and in the whole picture, visible body size, nose-tail distance and turning -- each also averaged over
  0.33 s, 1 s and 3 s windows (mean, spread, min, max), so a frame is judged with its context.
Model: one random forest per behaviour (this behaviour vs everything else), on binned features (pure numpy).
Output: per-frame probability -> smoothed -> bouts (minimum bout length, short gaps closed) -> events.
Validation: leave-one-video-out (2+ videos) or 5 contiguous time blocks (1 video), per behaviour precision, recall,
F1, Cohen's kappa and hand vs BT time.

Files: a model is saved as JSON (*.btclf) -- readable, versioned, with its training report.
"""
import json
import math
import os
import time

import numpy as np

MODEL_FORMAT = "BehavioralTracker behaviour classifier"
BASE = ("speed", "motion_body", "motion_head", "motion_legs", "motion_tail", "motion_around", "motion_region",
        "size", "length", "nose_dist", "turning", "seen")
WINDOWS_S = (0.33, 1.0, 3.0)


# ============================================================================= features
def _roll(x, k, fn):
    """Centred rolling statistic over k frames (edges: the window shrinks).  fn in mean / std / min / max."""
    x = np.asarray(x, float)
    n = len(x)
    if k <= 1 or n == 0:
        return x.copy() if fn != "std" else np.zeros(n)
    h = k // 2
    pad = np.concatenate([np.full(h, np.nan), x, np.full(k - 1 - h, np.nan)])
    from numpy.lib.stride_tricks import sliding_window_view
    w = sliding_window_view(pad, k)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if fn == "mean":
                r = np.nanmean(w, axis=1)
            elif fn == "std":
                r = np.nanstd(w, axis=1)
            elif fn == "min":
                r = np.nanmin(w, axis=1)
            else:
                r = np.nanmax(w, axis=1)
    return r


def base_features(md):
    """Per-frame base features of one analysed video (tracking.freezing.MotionData in 'body' mode) -> (n, len(BASE))."""
    fps = float(md.fps)
    n = len(md.frames)
    out = np.full((n, len(BASE)), np.nan, np.float32)
    try:
        cut = int(md.auto_cutoff()[0])
    except Exception:
        cut = 10
    body_len = float(getattr(md, "body_len_dots", 0.0) or getattr(md, "body_len_full", 0.0) or 0.0)
    s = md.geo.s
    L = max(1.0, body_len * s)                         # body length, work px
    # speed of the body centre (body lengths per second), 3-frame smoothing
    if md.xs is not None:
        x, y = md.geo.to_work(md.xs, md.ys)
        x, y = _roll(x, 3, "mean"), _roll(y, 3, "mean")
        v = np.hypot(np.gradient(x), np.gradient(y)) * fps / L
        out[:, 0] = v
    for j, area in ((1, "body"), (2, "head"), (3, "legs"), (4, "tail"), (5, "animal"), (6, "region")):
        try:
            if area in ("body", "head", "legs", "tail") and not md.has_body():
                continue
            if area == "animal" and not md.has_disc():
                continue
            out[:, j] = md.motion(cut, area)[0]
        except Exception:
            pass
    kp = getattr(md, "kp", None)
    if kp is not None:
        nose, ctr, tb = kp[:, 0], kp[:, 3], kp[:, 6]
        length = np.hypot(nose[:, 0] - tb[:, 0], nose[:, 1] - tb[:, 1]) / L
        nd = np.hypot(nose[:, 0] - ctr[:, 0], nose[:, 1] - ctr[:, 1]) / L
        out[:, 8], out[:, 9] = length, nd
        hd = np.arctan2(nose[:, 1] - ctr[:, 1], nose[:, 0] - ctr[:, 0])
        dh = np.abs(np.angle(np.exp(1j * np.diff(hd, prepend=hd[:1])))) * fps
        out[:, 10] = dh
        # visible size: the outline's extent (points spread), relative
        pts = kp[:, :7]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spread = np.nanmax(pts[:, :, 0], axis=1) - np.nanmin(pts[:, :, 0], axis=1) + \
                    np.nanmax(pts[:, :, 1], axis=1) - np.nanmin(pts[:, :, 1], axis=1)
        out[:, 7] = spread / L
    seen = getattr(md, "seen", None)
    out[:, 11] = np.asarray(seen, float)[:n] if seen is not None and len(seen) >= n else 1.0
    return out


def window_features(base, fps):
    """Base features + rolling mean / spread over 0.33 / 1 / 3 s + min / max over 1 s -> (n, F), names."""
    cols, names = [], []
    for j, nm in enumerate(BASE):
        x = base[:, j].astype(float)
        cols.append(x)
        names.append(nm)
        if nm == "seen":
            continue
        for w in WINDOWS_S:
            k = max(1, int(round(w * fps)))
            cols.append(_roll(x, k, "mean"))
            names.append(f"{nm}_mean{w:g}s")
            cols.append(_roll(x, k, "std"))
            names.append(f"{nm}_sd{w:g}s")
        k = max(1, int(round(fps)))
        cols.append(_roll(x, k, "min"))
        names.append(f"{nm}_min1s")
        cols.append(_roll(x, k, "max"))
        names.append(f"{nm}_max1s")
    X = np.stack(cols, axis=1).astype(np.float32)
    return X, names


def video_features(path, settings=None, region=None, progress=None, cancel=None, cache_dir=None):
    """Analyse one video (BT tracking + body-outline motion) -> dict(X, names, fps, frames, start_frame).
    Cached next to the results (the analysis is the slow part)."""
    from tracking import freezing as FZ
    st = FZ.merged_settings(dict(settings or {}, motion_area="body"))
    key = None
    if cache_dir:
        try:
            stt = os.stat(path)
            key = json.dumps([os.path.basename(str(path)), stt.st_size, int(stt.st_mtime), FZ.region_json(region),
                              st.get("polarity"), st.get("min_contrast"), st.get("body_length_px"),
                              st.get("start_time"), st.get("end_time"), 2], sort_keys=True)
            cf = os.path.join(cache_dir, os.path.splitext(os.path.basename(str(path)))[0] + "_BT_features.npz")
            if os.path.exists(cf):
                z = np.load(cf, allow_pickle=False)
                if str(z["key"]) == key:
                    return {"X": z["X"], "names": [str(s) for s in z["names"]], "fps": float(z["fps"]),
                            "frames": z["frames"], "start_frame": int(z["frames"][0])}
        except Exception:
            key = None
    md = FZ.run_freezing(path, st, region=region, progress=progress, cancel=cancel)
    base = base_features(md)
    X, names = window_features(base, md.fps)
    res = {"X": X, "names": names, "fps": float(md.fps), "frames": np.asarray(md.frames), "start_frame":
           int(md.frames[0])}
    if cache_dir and key:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez_compressed(cf, X=X, names=np.array(names), fps=md.fps, frames=res["frames"], key=key)
        except Exception:
            pass
    return res


def labels_for(events, behaviour, frames, fps):
    """Frames (video frame numbers) inside a state event of `behaviour` -> bool array."""
    t = np.asarray(frames, float) / float(fps)
    y = np.zeros(len(t), bool)
    for e in events or []:
        if e.get("behaviour") != behaviour or (e.get("kind") or "state") != "state" or e.get("stop") is None:
            continue
        a, b = float(e["start"]), float(e["stop"])
        y |= (t >= a) & (t < b)
    return y


# ============================================================================= model
class Forest:
    """Random forest for one yes / no question, on features binned to n_bins quantile levels (fast in numpy).
    Balanced: both classes weigh the same in every tree (rare behaviours are not ignored)."""

    def __init__(self, n_trees=40, max_depth=10, min_leaf=15, n_bins=32, seed=0):
        self.n_trees, self.max_depth, self.min_leaf, self.n_bins, self.seed = n_trees, max_depth, min_leaf, n_bins, seed
        self.edges = None
        self.trees = []
        self.importance = None

    # ---- binning
    def _fit_bins(self, X):
        edges = []
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[np.isfinite(v)]
            if v.size == 0:
                edges.append(np.zeros(0, np.float32))
                continue
            q = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1)[1:-1]))
            edges.append(q.astype(np.float32))
        self.edges = edges

    def _bin(self, X):
        B = np.zeros(X.shape, np.uint8)
        for j, e in enumerate(self.edges):
            col = X[:, j]
            b = np.searchsorted(e, col, side="right").astype(np.int32) + 1     # 1..n_bins
            b[~np.isfinite(col)] = 0                                            # 0 = missing
            B[:, j] = b
        return B

    # ---- training
    def fit(self, X, y, progress=None):
        rng = np.random.default_rng(self.seed)
        self._fit_bins(X)
        Xb = self._bin(X)
        y = np.asarray(y, bool)
        n, F = Xb.shape
        nb = self.n_bins + 1
        pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        if pos.size == 0 or neg.size == 0:
            raise ValueError("The behaviour needs frames with AND without it.")
        mtry = max(2, int(round(math.sqrt(F))))
        self.trees = []
        imp = np.zeros(F)
        for t in range(self.n_trees):
            m = min(max(pos.size, neg.size), 20000)       # balanced bootstrap
            idx = np.concatenate([rng.choice(pos, m), rng.choice(neg, m)])
            tree = self._grow(Xb, y, idx, rng, mtry, nb, imp)
            self.trees.append(tree)
            if progress is not None:
                progress((t + 1) / self.n_trees)
        self.importance = imp / max(1e-12, imp.sum())
        return self

    def _grow(self, Xb, y, idx, rng, mtry, nb, imp):
        feat, thr, left, right, val = [], [], [], [], []

        def new_node():
            feat.append(-1)
            thr.append(0)
            left.append(-1)
            right.append(-1)
            val.append(0.0)
            return len(feat) - 1
        root = new_node()
        stack = [(root, idx, 0)]
        F = Xb.shape[1]
        while stack:
            node, ii, depth = stack.pop()
            yy = y[ii]
            p = float(yy.mean())
            val[node] = p
            if depth >= self.max_depth or len(ii) < 2 * self.min_leaf or p in (0.0, 1.0):
                continue
            fs = rng.choice(F, mtry, replace=False)
            sub = Xb[np.ix_(ii, fs)].astype(np.int64) + (np.arange(len(fs)) * nb)[None, :]
            cnt = np.bincount(sub.ravel(), minlength=len(fs) * nb).reshape(len(fs), nb)
            cpos = np.bincount(sub[yy].ravel(), minlength=len(fs) * nb).reshape(len(fs), nb)
            cl, pl = np.cumsum(cnt, axis=1)[:, :-1], np.cumsum(cpos, axis=1)[:, :-1]
            N, P = len(ii), float(yy.sum())
            cr, pr = N - cl, P - pl
            ok = (cl >= self.min_leaf) & (cr >= self.min_leaf)
            with np.errstate(divide="ignore", invalid="ignore"):
                gl = 1.0 - (pl / cl) ** 2 - (1.0 - pl / cl) ** 2
                gr = 1.0 - (pr / cr) ** 2 - (1.0 - pr / cr) ** 2
                score = (cl * gl + cr * gr) / N
            score = np.where(ok, score, np.inf)
            k = int(np.argmin(score))
            fi, b = divmod(k, nb - 1)
            best = score[fi, b]
            g0 = 1.0 - p ** 2 - (1.0 - p) ** 2
            if not np.isfinite(best) or best >= g0 - 1e-9:
                continue
            f = int(fs[fi])
            imp[f] += (g0 - best) * N
            col = Xb[ii, f]
            mask = col <= b
            feat[node], thr[node] = f, int(b)
            ln, rn = new_node(), new_node()
            left[node], right[node] = ln, rn
            stack.append((ln, ii[mask], depth + 1))
            stack.append((rn, ii[~mask], depth + 1))
        return {"f": np.array(feat, np.int16), "t": np.array(thr, np.uint8), "l": np.array(left, np.int32),
                "r": np.array(right, np.int32), "v": np.array(val, np.float32)}

    # ---- use
    def predict_proba(self, X):
        Xb = self._bin(X)
        n = Xb.shape[0]
        acc = np.zeros(n)
        rows = np.arange(n)
        for tr in self.trees:
            node = np.zeros(n, np.int64)
            for _d in range(self.max_depth + 1):
                f = tr["f"][node]
                inner = f >= 0
                if not inner.any():
                    break
                go_left = np.zeros(n, bool)
                fi = f[inner].astype(np.int64)
                go_left[inner] = Xb[rows[inner], fi] <= tr["t"][node[inner]]
                node = np.where(inner, np.where(go_left, tr["l"][node], tr["r"][node]), node)
            acc += tr["v"][node]
        return acc / max(1, len(self.trees))

    def to_json(self):
        return {"n_trees": self.n_trees, "max_depth": self.max_depth, "min_leaf": self.min_leaf, "n_bins": self.n_bins,
                "edges": [e.tolist() for e in self.edges],
                "trees": [{k: v.tolist() for k, v in t.items()} for t in self.trees],
                "importance": None if self.importance is None else self.importance.tolist()}

    @classmethod
    def from_json(cls, d):
        f = cls(d["n_trees"], d["max_depth"], d["min_leaf"], d["n_bins"])
        f.edges = [np.asarray(e, np.float32) for e in d["edges"]]
        dt = {"f": np.int16, "t": np.uint8, "l": np.int32, "r": np.int32, "v": np.float32}
        f.trees = [{k: np.asarray(v, dt[k]) for k, v in t.items()} for t in d["trees"]]
        f.importance = None if d.get("importance") is None else np.asarray(d["importance"])
        return f


# ============================================================================= bouts
def to_bouts(prob, fps, threshold=0.5, smooth_s=0.2, min_bout_s=0.3, max_gap_s=0.2):
    """Per-frame probability -> bool per frame (smoothed, short gaps closed, short bouts dropped)."""
    p = np.asarray(prob, float)
    k = max(1, int(round(smooth_s * fps)))
    if k > 1:
        p = _roll(p, k, "mean")
    on = p >= threshold
    g = int(round(max_gap_s * fps))
    if g > 0:
        r = _runs(~on)
        for a, b in r:
            if a > 0 and b < len(on) and b - a <= g:
                on[a:b] = True
    m = int(round(min_bout_s * fps))
    for a, b in _runs(on):
        if b - a < m:
            on[a:b] = False
    return on


def _runs(mask):
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(np.int8))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def events_from(on, frames, fps, behaviour):
    out = []
    fr = np.asarray(frames)
    for a, b in _runs(on):
        out.append({"behaviour": behaviour, "kind": "state", "start": round(float(fr[a]) / fps, 4),
                    "stop": round(float(fr[b - 1] + 1) / fps, 4)})
    return out


# ============================================================================= metrics
def scores(y_true, y_pred):
    y, p = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    n = max(1, tp + fp + fn + tn)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (tp and (prec + rec) > 0) else (0.0 if tp + fp + fn else float("nan"))
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "kappa": kappa, "agreement": po,
            "hand_pct": 100.0 * (tp + fn) / n, "bt_pct": 100.0 * (tp + fp) / n}


# ============================================================================= training / applying
class BehaviourModel:
    """Several behaviours, one forest each, with the post-processing settings and the training report."""

    def __init__(self, behaviours=None, settings=None):
        self.behaviours = list(behaviours or [])
        self.forests = {}
        self.names = []
        self.post = {b: {"threshold": 0.5, "min_bout_s": 0.3, "max_gap_s": 0.2, "smooth_s": 0.2}
                     for b in self.behaviours}
        self.settings = dict(settings or {})          # the analysis settings the features were made with
        self.report = {}
        self.trained = None

    def train(self, data, n_trees=40, progress=None, validate=True):
        """data = [dict(X, names, fps, frames, events, video)].  Cross-validated report first, then the final
        model on all data."""
        if not data:
            raise ValueError("No training videos.")
        self.names = list(data[0]["names"])
        steps = len(self.behaviours) * ((self._n_folds(data) if validate else 0) + 1)
        step = [0]

        def prog(f):
            if progress is not None:
                progress((step[0] + f) / max(1, steps))
        report = {}
        for b in self.behaviours:
            ys = [labels_for(d["events"], b, d["frames"], d["fps"]) for d in data]
            tot = sum(int(y.sum()) for y in ys)
            if tot < 10:
                report[b] = {"error": "fewer than 10 frames scored"}
                step[0] += (self._n_folds(data) if validate else 0) + 1
                continue
            if validate:
                report[b] = self._validate(data, ys, b, n_trees, prog, step)
            X = np.concatenate([d["X"] for d in data])
            y = np.concatenate(ys)
            self.forests[b] = Forest(n_trees=n_trees, seed=7).fit(X, y, progress=prog)
            step[0] += 1
            imp = self.forests[b].importance
            top = np.argsort(-imp)[:5]
            report.setdefault(b, {})["top_features"] = [self.names[i] for i in top]
            report[b]["hand_s"] = round(sum(int(y.sum()) / d["fps"] for y, d in zip(ys, data)), 1)
        self.report = report
        self.trained = time.strftime("%Y-%m-%d %H:%M")
        return report

    @staticmethod
    def _n_folds(data):
        return len(data) if len(data) >= 2 else 5

    def _validate(self, data, ys, b, n_trees, prog, step):
        """Leave one video out (2+ videos) or 5 contiguous blocks of the one video."""
        preds, truths = [], []
        folds = []
        if len(data) >= 2:
            for k in range(len(data)):
                folds.append(([i for i in range(len(data)) if i != k], [k], None))
            scheme = "leave one video out"
        else:
            n = len(ys[0])
            edges = np.linspace(0, n, 6).astype(int)
            for k in range(5):
                folds.append(([0], [0], (edges[k], edges[k + 1])))
            scheme = "5 time blocks"
        post = self.post[b]
        for tr_ids, te_ids, block in folds:
            if block is None:
                Xtr = np.concatenate([data[i]["X"] for i in tr_ids])
                ytr = np.concatenate([ys[i] for i in tr_ids])
                Xte, yte, fps, frames = data[te_ids[0]]["X"], ys[te_ids[0]], data[te_ids[0]]["fps"], None
            else:
                a, c = block
                gap = int(round(3 * data[0]["fps"]))          # 3 s guard: no neighbouring frames in training
                keep = np.ones(len(ys[0]), bool)
                keep[max(0, a - gap):min(len(keep), c + gap)] = False
                Xtr, ytr = data[0]["X"][keep], ys[0][keep]
                Xte, yte, fps = data[0]["X"][a:c], ys[0][a:c], data[0]["fps"]
            if ytr.sum() < 5 or (~ytr).sum() < 5:
                step[0] += 1
                continue
            f = Forest(n_trees=max(10, n_trees // 2), seed=3).fit(Xtr, ytr, progress=prog)
            step[0] += 1
            p = to_bouts(f.predict_proba(Xte), fps, post["threshold"], post["smooth_s"], post["min_bout_s"],
                         post["max_gap_s"])
            preds.append(p)
            truths.append(yte)
        if not preds:
            return {"error": "not enough scored frames to test", "scheme": scheme}
        r = scores(np.concatenate(truths), np.concatenate(preds))
        r["scheme"] = scheme
        r["per_fold_f1"] = [round(scores(t, p)["f1"], 3) for t, p in zip(truths, preds)]
        return {k: (round(v, 4) if isinstance(v, float) and math.isfinite(v) else v) for k, v in r.items()}

    def predict(self, feats):
        """feats = video_features() -> {behaviour: dict(prob, on, events)}."""
        X, fps, frames = feats["X"], feats["fps"], feats["frames"]
        if list(feats["names"]) != self.names:
            raise ValueError("These features do not match the model (made with another BT version).")
        out = {}
        for b, f in self.forests.items():
            pr = f.predict_proba(X)
            po = self.post[b]
            on = to_bouts(pr, fps, po["threshold"], po["smooth_s"], po["min_bout_s"], po["max_gap_s"])
            out[b] = {"prob": pr, "on": on, "events": events_from(on, frames, fps, f"{b}")}
        return out

    # ---- files
    def save(self, path):
        d = {"format": MODEL_FORMAT, "version": 1, "behaviours": self.behaviours, "names": self.names,
             "post": self.post, "settings": self.settings, "report": self.report, "trained": self.trained,
             "forests": {b: f.to_json() for b, f in self.forests.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, separators=(",", ":"))

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("format") != MODEL_FORMAT:
            raise ValueError("Not a BT behaviour classifier file.")
        m = cls(d["behaviours"], d.get("settings"))
        m.names, m.post, m.report, m.trained = d["names"], d["post"], d.get("report") or {}, d.get("trained")
        m.forests = {b: Forest.from_json(v) for b, v in d["forests"].items()}
        return m


def write_predictions(path_base, video, pred, fps, frames=None):
    """<base>_BT_behaviours.csv (bouts) + <base>_BT_behaviour_frames.csv (per frame: probability and yes / no)."""
    import csv
    ev_p, fr_p = path_base + "_BT_behaviours.csv", path_base + "_BT_behaviour_frames.csv"
    with open(ev_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Video", "Behaviour", "Start_s", "Stop_s", "Duration_s"])
        for b, r in pred.items():
            for e in r["events"]:
                w.writerow([os.path.basename(str(video)), b, f"{e['start']:.3f}", f"{e['stop']:.3f}",
                            f"{e['stop'] - e['start']:.3f}"])
    names = list(pred)
    n = len(next(iter(pred.values()))["prob"]) if pred else 0
    fr = np.arange(n) if frames is None else np.asarray(frames)
    with open(fr_p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Frame", "Time_s"] + [f"P_{b}" for b in names] + [b for b in names])
        for i in range(n):
            w.writerow([int(fr[i]), f"{fr[i] / fps:.3f}"] + [f"{pred[b]['prob'][i]:.3f}" for b in names] +
                       [int(pred[b]["on"][i]) for b in names])
    return ev_p, fr_p


# ============================================================================= many videos (dialog + command line)
def features_dir(video):
    """Where a video's features are cached (in its results folder)."""
    from tracking.location import results_root
    return os.path.join(results_root(video), "BT features")


def has_features(video, settings=None, region=None):
    """True when the video's features are cached and still valid (the next use is instant)."""
    from tracking import freezing as FZ
    try:
        st = FZ.merged_settings(dict(settings or {}, motion_area="body"))
        stt = os.stat(video)
        key = json.dumps([os.path.basename(str(video)), stt.st_size, int(stt.st_mtime), FZ.region_json(region),
                          st.get("polarity"), st.get("min_contrast"), st.get("body_length_px"),
                          st.get("start_time"), st.get("end_time"), 2], sort_keys=True)
        cf = os.path.join(features_dir(video), os.path.splitext(os.path.basename(str(video)))[0] + "_BT_features.npz")
        if not os.path.exists(cf):
            return False
        with np.load(cf, allow_pickle=False) as z:
            return str(z["key"]) == key
    except Exception:
        return False


def prepare_features(videos, progress=None, cancel=None, log=None):
    """Analyse the videos now (BT tracking + body motion) and cache their features, so that training on them and
    finding behaviours in them later is quick -- e.g. overnight.  -> {video: 'ready' | 'cached' | error text}."""
    out = {}
    n = max(1, len(videos))
    for k, v in enumerate(videos):
        if cancel is not None and cancel():
            break
        if has_features(v):
            out[v] = "cached"
        else:
            try:
                video_features(v, progress=(lambda f, k=k: progress((k + f) / n)) if progress else None,
                               cancel=cancel, cache_dir=features_dir(v))
                out[v] = "ready"
            except Exception as exc:
                out[v] = f"failed: {exc}"
        if log is not None:
            log(f"{os.path.basename(v)}: {out[v]}")
        if progress is not None:
            progress((k + 1) / n)
    return out


def classify_videos(model, videos, out_xlsx=None, progress=None, cancel=None, log=None):
    """Find the model's behaviours in every video: <results>/<video>/<video>_BT_behaviours.csv + _frames.csv +
    .btms per video, and one Excel time budget (+ the model's test).  -> (xlsx path, {video: 'done' | error})."""
    import pandas as pd
    from tracking.location import results_root
    from tracking.project_file import save_scoring, audit_add
    rows, status = [], {}
    n = max(1, len(videos))
    for k, v in enumerate(videos):
        if cancel is not None and cancel():
            break
        try:
            f = video_features(v, progress=(lambda q, k=k: progress((k + 0.9 * q) / n)) if progress else None,
                               cancel=cancel, cache_dir=features_dir(v))
            pred = model.predict(f)
            outdir = os.path.join(results_root(v), os.path.splitext(os.path.basename(v))[0])
            os.makedirs(outdir, exist_ok=True)
            base = os.path.join(outdir, os.path.splitext(os.path.basename(v))[0])
            write_predictions(base, v, pred, f["fps"], f["frames"])
            events = [dict(e) for r in pred.values() for e in r["events"]]
            inv = [{"name": b, "key": "", "kind": "state"} for b in pred]
            trail = []
            audit_add(trail, "Behaviours found by the BT classifier", {"model behaviours": list(pred),
                                                                       "events": len(events)})
            try:
                save_scoring(base + "_BT_behaviours.btms", v, inv, events, f["fps"], len(f["frames"]), trail,
                             embed_px=0)
            except Exception:
                pass
            dur = len(f["frames"]) / f["fps"]
            row = {"Video": os.path.basename(v), "Analysed_s": round(dur, 1)}
            for b, r in pred.items():
                t = sum(e["stop"] - e["start"] for e in r["events"])
                row[f"{b}_time_s"] = round(t, 2)
                row[f"{b}_pct"] = round(100 * t / max(1e-9, dur), 2)
                row[f"{b}_bouts"] = len(r["events"])
                row[f"{b}_latency_s"] = round(r["events"][0]["start"] - f["frames"][0] / f["fps"], 2) \
                    if r["events"] else None
            rows.append(row)
            status[v] = "done"
        except Exception as exc:
            status[v] = f"failed: {exc}"
        if log is not None:
            log(f"{os.path.basename(v)}: {status[v]}")
    out = None
    if rows:
        out = out_xlsx or os.path.join(results_root(videos[0]), f"BT_behaviours_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with pd.ExcelWriter(out) as w:
            pd.DataFrame(rows).to_excel(w, sheet_name="Time budget", index=False)
            rep = [{"Behaviour": b, **{k: v for k, v in (model.report.get(b) or {}).items()
                                       if not isinstance(v, (list, dict))}} for b in model.behaviours]
            pd.DataFrame(rep).to_excel(w, sheet_name="Model test", index=False)
    if progress is not None:
        progress(1.0)
    return out, status
