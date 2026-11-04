"""A smaller copy of the analysed video, stored inside the BT files (.bta / .btms).

Whoever opens a BT file -- a colleague checking the analysis, a reviewer -- then sees the animal in every
frame (frame review, event scoring) even without the original video.  The copy keeps EVERY frame of the
analysed time window (frame N of the copy = frame N of the analysis), only the picture is smaller.

The copy is written in a few pieces at the same time (one decoder + encoder each), MPEG-4 Part 2 inside
.mp4 -- an encoder that OpenCV has on every computer.
"""
import math
import os
import shutil
import tempfile
import threading

import cv2


def copy_size(w, h, max_side=640):
    s = min(1.0, float(max_side) / float(max(w, h, 1)))
    return max(2, int(round(w * s)) // 2 * 2), max(2, int(round(h * s)) // 2 * 2)


def _workers(n_frames, cap=4):
    cpu = os.cpu_count() or 2
    return int(max(1, min(cap, cpu, math.ceil(n_frames / 1500.0))))


def encode_copy(video_path, first, last, warp=None, max_side=640, fps=30.0, progress=None, workers=None,
                out_dir=None):
    """Write frames [first, last] of `video_path` (straightened by `warp(frame)` when given) as a smaller
    video.  Returns {"segments": [{"first", "last", "file"}], "size": [w, h], "fps", "frames", "dir"} or
    None when the video cannot be read.  The caller deletes "dir" when done."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        return None
    if warp is not None:
        f0 = warp(f0)
    h0, w0 = f0.shape[:2]
    size = copy_size(w0, h0, max_side)
    first, last = int(first), int(last)
    n = max(0, last - first + 1)
    if n == 0:
        return None
    k = int(workers or _workers(n))
    bounds = [first + i * n // k for i in range(k)] + [last + 1]
    tmp = out_dir or tempfile.mkdtemp(prefix="bt_copy_")
    os.makedirs(tmp, exist_ok=True)
    done = [0] * k
    results = [None] * k
    errors = []

    def work(i):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            results[i] = None
            return
        fn = os.path.join(tmp, f"part{i:02d}.mp4")
        c = cv2.VideoCapture(video_path)
        vw = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 30.0), size)
        try:
            if not vw.isOpened():
                errors.append("no encoder")
                return
            c.set(cv2.CAP_PROP_POS_FRAMES, a)
            pos = int(round(c.get(cv2.CAP_PROP_POS_FRAMES) or a))
            while pos < a:                                  # some codecs land a bit early
                if not c.grab():
                    break
                pos += 1
            last_ok = None
            for j in range(a, b):
                okr, fr = c.read()
                if not okr:
                    if last_ok is None:
                        break
                    fr = last_ok                            # keep the frame count exact
                else:
                    last_ok = fr
                if warp is not None and okr:
                    fr = warp(fr)
                if fr.shape[1] != size[0] or fr.shape[0] != size[1]:
                    fr = cv2.resize(fr, size, interpolation=cv2.INTER_AREA)
                vw.write(fr)
                done[i] += 1
            results[i] = {"first": a, "last": a + done[i] - 1, "file": fn}
        except Exception as exc:                            # a copy is a convenience: never break the analysis
            errors.append(str(exc))
        finally:
            vw.release()
            c.release()

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(k)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.25)
        if progress is not None:
            try:
                progress(min(1.0, sum(done) / float(n)))
            except Exception:
                pass
    segs = [r for r in results if r and r["last"] >= r["first"] and os.path.exists(r["file"])]
    if errors or not segs:
        if out_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)
        return None
    return {"segments": segs, "size": [int(size[0]), int(size[1])], "fps": float(fps or 30.0),
            "frames": int(sum(s["last"] - s["first"] + 1 for s in segs)), "dir": tmp}



def encode_copies(video_path, specs, warp=None, max_side=640, fps=30.0, progress=None, workers=None):
    """Several copies from ONE reading of the video (chambers: one copy per chamber).

    specs = [{"first": f, "last": l, "rect": (x0, y0, x1, y1) or None}]: frames [first, last] of the video,
    straightened by `warp(frame)` when given, then cut to `rect`.  Returns one result per spec, each like
    encode_copy() (or None when that copy could not be written).  The caller deletes every "dir"."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [None] * len(specs)
    ok, f0 = cap.read()
    cap.release()
    if not ok:
        return [None] * len(specs)
    if warp is not None:
        f0 = warp(f0)
    H0, W0 = f0.shape[:2]
    sp = []
    for s in specs:
        x0, y0, x1, y1 = [int(v) for v in (s.get("rect") or (0, 0, W0, H0))]
        x0, y0 = max(0, min(x0, W0 - 2)), max(0, min(y0, H0 - 2))
        x1, y1 = max(x0 + 2, min(x1, W0)), max(y0 + 2, min(y1, H0))
        sp.append({"first": int(s["first"]), "last": int(s["last"]), "rect": (x0, y0, x1, y1),
                   "size": copy_size(x1 - x0, y1 - y0, max_side), "dir": tempfile.mkdtemp(prefix="bt_copy_")})
    first = min(s["first"] for s in sp)
    last = max(s["last"] for s in sp)
    n = max(0, last - first + 1)
    if n == 0:
        for s in sp:
            shutil.rmtree(s["dir"], ignore_errors=True)
        return [None] * len(specs)
    k = int(workers or _workers(n))
    bounds = [first + i * n // k for i in range(k)] + [last + 1]
    done = [0] * k
    results = [[None] * len(sp) for _ in range(k)]
    errors = []

    def work(i):
        a, b = bounds[i], bounds[i + 1]
        if b <= a:
            return
        c = cv2.VideoCapture(video_path)
        writers = {}
        counts = {}
        try:
            for j, s in enumerate(sp):
                if max(a, s["first"]) <= min(b - 1, s["last"]):
                    fn = os.path.join(s["dir"], f"part{i:02d}.mp4")
                    vw = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"), float(fps or 30.0), s["size"])
                    if not vw.isOpened():
                        errors.append("no encoder")
                        return
                    writers[j] = (vw, fn)
                    counts[j] = 0
            if not writers:
                return
            a0 = max(a, min(s["first"] for j, s in enumerate(sp) if j in writers))
            c.set(cv2.CAP_PROP_POS_FRAMES, a0)
            pos = int(round(c.get(cv2.CAP_PROP_POS_FRAMES) or a0))
            while pos < a0:                                  # some codecs land a bit early
                if not c.grab():
                    break
                pos += 1
            b0 = min(b - 1, max(s["last"] for j, s in enumerate(sp) if j in writers))
            last_ok = None
            for fnum in range(a0, b0 + 1):
                okr, fr = c.read()
                if not okr:
                    if last_ok is None:
                        break
                    fr = last_ok                            # keep the frame count exact
                else:
                    if warp is not None:
                        fr = warp(fr)
                    last_ok = fr
                for j, (vw, _fn) in writers.items():
                    s = sp[j]
                    if s["first"] <= fnum <= s["last"]:
                        x0, y0, x1, y1 = s["rect"]
                        sub = fr[y0:y1, x0:x1]
                        if sub.shape[1] != s["size"][0] or sub.shape[0] != s["size"][1]:
                            sub = cv2.resize(sub, s["size"], interpolation=cv2.INTER_AREA)
                        vw.write(sub)
                        counts[j] += 1
                done[i] += 1
            for j, (_vw, fn) in writers.items():
                if counts[j]:
                    f_a = max(a, sp[j]["first"])
                    results[i][j] = {"first": f_a, "last": f_a + counts[j] - 1, "file": fn}
        except Exception as exc:                            # a copy is a convenience: never break the analysis
            errors.append(str(exc))
        finally:
            for vw, _fn in writers.values():
                vw.release()
            c.release()

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(k)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=0.25)
        if progress is not None:
            try:
                progress(min(1.0, sum(done) / float(n)))
            except Exception:
                pass
    out = []
    for j, s in enumerate(sp):
        segs = [results[i][j] for i in range(k) if results[i][j] and results[i][j]["last"] >= results[i][j]["first"]
                and os.path.exists(results[i][j]["file"])]
        if errors or not segs:
            shutil.rmtree(s["dir"], ignore_errors=True)
            out.append(None)
            continue
        out.append({"segments": segs, "size": [int(s["size"][0]), int(s["size"][1])], "fps": float(fps or 30.0),
                    "frames": int(sum(g["last"] - g["first"] + 1 for g in segs)), "dir": s["dir"]})
    return out


class CopyReader:
    """Frames of an extracted copy by ORIGINAL frame number, scaled back to the analysis size."""

    def __init__(self, segments, full_size):
        self.segs = sorted(segments, key=lambda s: s["first"])
        self.full = (int(full_size[0]), int(full_size[1]))
        self.cap = None
        self.cur = None            # segment index open now
        self.pos = -1              # next frame number the open capture will return (original numbering)

    def covers(self, frame_no):
        return any(s["first"] <= frame_no <= s["last"] for s in self.segs)

    def read(self, frame_no):
        frame_no = int(frame_no)
        for i, s in enumerate(self.segs):
            if s["first"] <= frame_no <= s["last"]:
                break
        else:
            return None
        if self.cur != i:
            self.close()
            self.cap = cv2.VideoCapture(s["file"])
            self.cur, self.pos = i, -1
        if frame_no != self.pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no - s["first"])
        ok, f = self.cap.read()
        if not ok:
            self.pos = -1
            return None
        self.pos = frame_no + 1
        if (f.shape[1], f.shape[0]) != self.full:
            f = cv2.resize(f, self.full, interpolation=cv2.INTER_LINEAR)
        return f

    def close(self):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.cur, self.pos = None, None, -1
