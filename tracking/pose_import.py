"""Positions from a pose-estimation program -> BT's normal zone analysis.

Reads the CSV files of
  * DeepLabCut  (3 header rows: scorer / bodyparts / coords x, y, likelihood; multi-animal files have an extra
                 'individuals' row -- the first individual is used)
  * SLEAP       ("Export analysis CSV": track, frame_idx, instance.score, <part>.x, <part>.y, <part>.score)
and turns them into the same `raw` result BT's own tracker gives (tracking.location.track_video), so
analyse_track() writes the usual Excel / charts / PDF.  The body centre = the mean of the trunk points (or of
every point), the nose and the tail base (when the file has them) give the nose point for object exploration.
Points below the likelihood cut-off count as not seen.
"""
import math
import os

import numpy as np
import pandas as pd

NOSE_NAMES = ("nose", "snout", "head", "nosetip", "nose_tip")
TAIL_NAMES = ("tailbase", "tail_base", "tail base", "tail1", "tail_1", "tailroot", "tail_root", "base_of_tail")
CENTRE_NAMES = ("centre", "center", "body", "bodycenter", "body_center", "centroid", "spine", "spine2", "mid",
                "midbody", "back", "torso", "thorax")


def _norm(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def read_pose_csv(path):
    """-> dict(source, parts={name: (x, y, p)}, frames=array) (arrays per frame, NaN where missing)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        head = [f.readline() for _ in range(4)]
    first = [c.strip().lower() for c in head[0].split(",")]
    if first and first[0] == "scorer":                       # DeepLabCut
        rows = 4 if head[1].lower().startswith("individuals") else 3
        df = pd.read_csv(path, header=list(range(rows)), index_col=0)
        cols = df.columns
        lv = cols.nlevels
        parts = {}
        ind0 = cols.get_level_values(1)[0] if lv == 4 else None
        for col in cols:
            if lv == 4:
                _sc, ind, bp, co = col
                if ind != ind0:
                    continue
            else:
                _sc, bp, co = col
            parts.setdefault(bp, {})[co] = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        out = {}
        for bp, d in parts.items():
            if "x" in d and "y" in d:
                out[bp] = (d["x"], d["y"], d.get("likelihood", np.ones_like(d["x"])))
        frames = np.asarray(pd.to_numeric(pd.Series(df.index), errors="coerce").fillna(-1).to_numpy(), int)
        if (frames < 0).any():
            frames = np.arange(len(df))
        return {"source": "DeepLabCut", "parts": out, "frames": frames}
    df = pd.read_csv(path)
    low = {c.lower(): c for c in df.columns}
    if "frame_idx" in low and any(c.endswith(".x") for c in df.columns):   # SLEAP
        if "track" in low:
            tracks = [t for t in df[low["track"]].dropna().unique()]
            if tracks:
                df = df[df[low["track"]] == tracks[0]]
        fr = pd.to_numeric(df[low["frame_idx"]], errors="coerce").astype(int).to_numpy()
        n = int(fr.max()) + 1 if len(fr) else 0
        out = {}
        for c in df.columns:
            if c.endswith(".x"):
                bp = c[:-2]
                xc, yc, sc = c, f"{bp}.y", f"{bp}.score"
                x = np.full(n, np.nan)
                y = np.full(n, np.nan)
                p = np.full(n, np.nan)
                x[fr] = pd.to_numeric(df[xc], errors="coerce").to_numpy(float)
                y[fr] = pd.to_numeric(df[yc], errors="coerce").to_numpy(float) if yc in df else np.nan
                p[fr] = pd.to_numeric(df[sc], errors="coerce").to_numpy(float) if sc in df else 1.0
                out[bp] = (x, y, p)
        return {"source": "SLEAP", "parts": out, "frames": np.arange(n)}
    raise ValueError("This is not a DeepLabCut or SLEAP CSV file (DeepLabCut: 3 header rows starting with "
                     "'scorer'; SLEAP: 'Export analysis CSV' with frame_idx and <part>.x columns).")


def _pick(parts, names):
    for n in names:
        for bp in parts:
            if _norm(bp) == n:
                return bp
    for n in names:
        for bp in parts:
            if n in _norm(bp):
                return bp
    return None


def pose_to_raw(pose, video_path, setup, fps, frame_w, frame_h, total_frames, p_cut=0.6, output_dir=None):
    """The pose file as BT's tracking result (see module doc).  Positions are video pixels; when the setup has a
    perspective crop (matrix) they are mapped into the cropped picture like BT's own tracking."""
    parts = pose["parts"]
    if not parts:
        raise ValueError("The pose file has no body points.")
    n = len(next(iter(parts.values()))[0])

    def good(bp):
        x, y, p = parts[bp]
        ok = np.isfinite(x) & np.isfinite(y) & (np.nan_to_num(p, nan=0.0) >= p_cut)
        return np.where(ok, x, np.nan), np.where(ok, y, np.nan)
    nose, tail, centre = _pick(parts, NOSE_NAMES), _pick(parts, TAIL_NAMES), _pick(parts, CENTRE_NAMES)
    trunk = [bp for bp in parts if bp not in (nose,) and not _norm(bp).startswith(("tail_", "tailtip", "tail_end",
                                                                                     "tail2", "tail3", "tail4"))]
    if centre is not None:
        cx, cy = good(centre)
    else:
        xs = np.vstack([good(bp)[0] for bp in trunk])
        ys = np.vstack([good(bp)[1] for bp in trunk])
        with np.errstate(all="ignore"):
            cx, cy = np.nanmean(xs, axis=0), np.nanmean(ys, axis=0)
    M = np.asarray(setup.get("matrix") if setup.get("matrix") is not None else np.eye(3), np.float64)

    def warp(x, y):
        if np.allclose(M, np.eye(3)):
            return x, y
        v = np.stack([x, y, np.ones_like(x)])
        w = M @ np.nan_to_num(v)
        bad = ~np.isfinite(x) | ~np.isfinite(y)
        wx, wy = w[0] / w[2], w[1] / w[2]
        wx[bad] = np.nan
        wy[bad] = np.nan
        return wx, wy
    cx, cy = warp(cx, cy)
    fps = float(fps or 30.0)
    start_t = float(setup.get("start_time") or 0.0)
    end_req = setup.get("end_time")
    f0 = int(round(start_t * fps))
    f1 = min(n - 1, int(round(float(end_req) * fps)) if end_req is not None else n - 1)
    sel = np.arange(max(0, f0), max(f0, f1) + 1)
    status = ["Tracked" if (np.isfinite(cx[i]) and np.isfinite(cy[i])) else "Lost" for i in sel]
    ends = []
    if nose is not None and tail is not None:
        nx, ny = warp(*good(nose))
        tx, ty = warp(*good(tail))
        for i in sel:
            ok = np.isfinite(nx[i]) and np.isfinite(tx[i])
            # end A = nose, end B = tail base; strong head evidence for A (the file says which end is which)
            ends.append([float(nx[i]), float(ny[i]), float(tx[i]), float(ty[i]), -1.0, 1.0] if ok else None)
    L = float("nan")
    if nose is not None and tail is not None and ends:
        ls = [math.hypot(e[0] - e[2], e[1] - e[3]) for e in ends if e]
        L = float(np.median(ls)) if ls else float("nan")
    a_full = (L * L / 2.4) if math.isfinite(L) else None
    name = os.path.splitext(os.path.basename(str(video_path)))[0]
    if output_dir is None:
        from tracking.location import compute_output_dir
        output_dir = (os.path.join(setup["output_dir_override"], name) if setup.get("output_dir_override")
                      else compute_output_dir(str(video_path)))
    os.makedirs(output_dir, exist_ok=True)
    return {
        "video_path": str(video_path), "output_dir": output_dir, "video_name": name, "fps": fps,
        "frame_w": int(frame_w), "frame_h": int(frame_h), "total_frames": int(total_frames),
        "start_time": sel[0] / fps, "end_time": (sel[-1] + 1) / fps, "start_frame": int(sel[0]),
        "end_frame": int(sel[-1]), "requested_end_time": None if end_req is None else float(end_req),
        "warp_w": int(setup.get("warp_w") or frame_w), "warp_h": int(setup.get("warp_h") or frame_h),
        "need_warp": False, "frames": [int(i) for i in sel], "times": [float(i / fps) for i in sel],
        "xs": [float(cx[i]) for i in sel], "ys": [float(cy[i]) for i in sel],
        "areas": [float(a_full or 0.0)] * len(sel), "status": status, "frac_names": [], "fracs": [],
        "ends": ends if any(e is not None for e in ends) else [], "covered_zones": [],
        "stop_reason": None, "algorithm": f"imported ({pose['source']})", "polarity": None,
        "background_info": f"positions imported from {pose['source']} (likelihood ≥ {p_cut:g})",
        "background_bgr": None, "animal_area_full": a_full, "body_len_full": L if math.isfinite(L) else None,
        "low_light_gamma": 1.0, "polarity_warning": None,
        "pose_points": {"centre": centre or "mean of the trunk points", "nose": nose, "tail_base": tail},
    }
