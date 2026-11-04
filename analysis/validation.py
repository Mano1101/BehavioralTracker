"""Agreement between BT and hand scoring -- the numbers a methods section / a reviewer asks for.

Per video (a hand-scored behaviour, e.g. "Freezing" scored in Event Scoring, against BT's freezing of the same
video):
  * frame-by-frame agreement (%) and Cohen's kappa
  * hand % vs BT % of the analysed time, and per time bin (default 10 s)
  * per-bin Pearson r, mean difference (bias, BT - hand) and its 95 % limits of agreement (Bland-Altman)
Across videos (the validation study):
  * Pearson r and ICC(A,1) (two-way random effects, absolute agreement, single rater -- Shrout & Fleiss ICC(2,1))
    of the video-level percentages, the bias and its 95 % limits of agreement, the mean kappa / agreement.
"""
import math

import numpy as np


def hand_mask(events, behaviour, times, fps):
    """Frames (video times `times`) inside the hand-scored bouts of `behaviour` (state events with a stop)."""
    m = np.zeros(len(times), bool)
    half = 0.5 / float(fps or 30.0)
    for e in events or []:
        if str(e.get("behaviour")) != str(behaviour):
            continue
        a, b = e.get("start"), e.get("stop")
        if a is None or b is None:
            continue
        m |= (times + half > float(a)) & (times + half <= float(b))
    return m


def cohen_kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float((po - pe) / (1 - pe)) if pe < 1 else (1.0 if po == 1 else float("nan"))


def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def bland_altman(auto, hand):
    d = np.asarray(auto, float) - np.asarray(hand, float)
    d = d[np.isfinite(d)]
    if d.size < 2:
        return float("nan"), float("nan"), float("nan")
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def icc_a1(x, y):
    """ICC(A,1) / Shrout-Fleiss ICC(2,1): two raters (BT, hand) x n videos, absolute agreement."""
    Y = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    Y = Y[np.all(np.isfinite(Y), axis=1)]
    n, k = Y.shape
    if n < 3:
        return float("nan")
    gm = Y.mean()
    ssr = k * ((Y.mean(axis=1) - gm) ** 2).sum()
    ssc = n * ((Y.mean(axis=0) - gm) ** 2).sum()
    sst = ((Y - gm) ** 2).sum()
    sse = sst - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else float("nan")


def compare_video(auto_frz, times, keep, events, behaviour, fps, bin_s=10.0):
    """One video: BT's frame-by-frame freezing (bool, aligned with `times`) vs the hand bouts of `behaviour`."""
    times = np.asarray(times, float)
    keep = np.ones(len(times), bool) if keep is None else np.asarray(keep, bool)
    auto = np.asarray(auto_frz, bool) & keep
    hand = hand_mask(events, behaviour, times, fps) & keep
    a, h = auto[keep], hand[keep]
    out = {"Frames": int(keep.sum()), "BT_pct": 100.0 * a.mean() if a.size else float("nan"),
           "Hand_pct": 100.0 * h.mean() if h.size else float("nan"),
           "Frame_agreement_pct": 100.0 * float((a == h).mean()) if a.size else float("nan"),
           "Kappa": cohen_kappa(a, h)}
    bins = []
    if bin_s and bin_s > 0 and keep.any():
        t0, t1 = times[keep][0], times[keep][-1] + 1.0 / fps
        edges = np.arange(t0, t1, bin_s)
        for e0 in edges:
            sel = keep & (times >= e0) & (times < min(t1, e0 + bin_s))
            if sel.sum() >= max(2, 0.5 * bin_s * fps):
                bins.append((float(e0), 100.0 * auto[sel].mean(), 100.0 * hand[sel].mean()))
    if bins:
        b = np.array(bins)
        out["Bins"] = len(bins)
        out["Bin_r"] = _pearson(b[:, 1], b[:, 2])
        out["Bin_bias"], out["Bin_LoA_low"], out["Bin_LoA_high"] = bland_altman(b[:, 1], b[:, 2])
    out["_bins"] = bins
    return out


def across(rows):
    """The validation study over videos (rows = compare_video() results)."""
    bt = np.array([r["BT_pct"] for r in rows], float)
    hd = np.array([r["Hand_pct"] for r in rows], float)
    bias, lo, hi = bland_altman(bt, hd)
    k = np.array([r["Kappa"] for r in rows], float)
    ag = np.array([r["Frame_agreement_pct"] for r in rows], float)
    return {"Videos": len(rows), "Pearson_r": _pearson(bt, hd), "ICC_A1": icc_a1(bt, hd),
            "Bias_BT_minus_hand": bias, "LoA_low": lo, "LoA_high": hi,
            "Mean_kappa": float(np.nanmean(k)) if np.isfinite(k).any() else float("nan"),
            "Mean_frame_agreement_pct": float(np.nanmean(ag)) if np.isfinite(ag).any() else float("nan")}


def kappa_words(k):
    if not math.isfinite(k):
        return "—"
    return ("almost perfect" if k > 0.8 else "substantial" if k > 0.6 else "moderate" if k > 0.4 else
            "fair" if k > 0.2 else "slight" if k > 0 else "poor")
