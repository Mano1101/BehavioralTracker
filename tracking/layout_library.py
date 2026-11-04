"""'Learn my layouts': BehavioralTracker remembers the zones you used on your own set-ups.

Every time you track a video (or save zones for a batch video) the picture of the apparatus and its zones
are kept here (a small grey picture + the zones, nothing else).  ⚡ Auto first looks for one of YOUR saved
set-ups in the new video: the apparatus picture is matched point by point (ORB features + a perspective
fit, so a camera that moved a little or zoomed is fine) and your zones -- with your names -- are moved onto
the new video.  Only when none of your set-ups is found does Auto fall back to the built-in templates.

The more videos you analyse, the more set-ups it knows.  Everything stays on this computer.
"""
import json
import os
import time

import cv2
import numpy as np

MAX_ENTRIES = 80
SIDE = 480                         # pictures are kept at this size (longest side)
_MIN_INLIERS = 22


def _gray_small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = g.shape[:2]
    s = min(1.0, SIDE / float(max(h, w)))
    if s < 1.0:
        g = cv2.resize(g, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return g, s


def _features(gray):
    make = getattr(getattr(cv2, "ORB", None), "create", None) or cv2.ORB_create      # OpenCV 5 / 4
    orb = make(nfeatures=1500, scaleFactor=1.2, nlevels=8, fastThreshold=12)
    g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return orb.detectAndCompute(g, None)


class LayoutLibrary:
    def __init__(self, folder):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)
        self.index_path = os.path.join(folder, "index.json")
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)
        except Exception:
            self.entries = []
        self._feat = {}

    def _save_index(self):
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.entries, f)
        os.replace(tmp, self.index_path)

    def _entry_features(self, e):
        if e["id"] not in self._feat:
            g = cv2.imread(os.path.join(self.folder, e["image"]), cv2.IMREAD_GRAYSCALE)
            self._feat[e["id"]] = _features(g) if g is not None else (None, None)
        return self._feat[e["id"]]

    @staticmethod
    def _homography(kq, dq, ke, de):
        """Entry -> query homography and its inlier count (or None, 0)."""
        if dq is None or de is None or len(kq) < 12 or len(ke) < 12:
            return None, 0
        make = getattr(getattr(cv2, "BFMatcher", None), "create", None) or cv2.BFMatcher
        bf = make(cv2.NORM_HAMMING)
        try:
            pairs = bf.knnMatch(de, dq, k=2)
        except cv2.error:
            return None, 0
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.78 * n.distance]
        if len(good) < 12:
            return None, 0
        src = np.float32([ke[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if H is None or inl is None:
            return None, 0
        n_in = int(inl.sum())
        # a sensible camera change: no mirror, no extreme scale / perspective
        a = H[:2, :2] / H[2, 2]
        det = float(np.linalg.det(a))
        if not (0.35 < det < 3.0) or abs(H[2, 0] / H[2, 2]) > 0.003 or abs(H[2, 1] / H[2, 2]) > 0.003:
            return None, 0
        return H, n_in

    def match(self, image, test_key=None, any_test=False):
        """Best saved set-up found in `image` -> (zones in image px, entry, inliers) or (None, None, 0)."""
        if not self.entries or image is None:
            return None, None, 0
        gq, sq = _gray_small(image)
        kq, dq = _features(gq)
        best = (None, None, 0)
        for e in self.entries:
            if not any_test and test_key and e.get("test") != test_key:
                continue
            ke, de = self._entry_features(e)
            H, n_in = self._homography(kq, dq, ke, de)
            if H is None or n_in < _MIN_INLIERS:
                continue
            if n_in > best[2]:
                best = (H, e, n_in)
        H, e, n_in = best
        if e is None:
            return None, None, 0
        zones = {}
        for name, pts in e["zones"].items():
            p = np.float32(pts).reshape(-1, 1, 2)
            q = cv2.perspectiveTransform(p, H).reshape(-1, 2) / sq
            zones[name] = [(float(x), float(y)) for x, y in q]
        # the moved zones must still be inside the picture
        h, w = image.shape[:2]
        allp = np.vstack([np.asarray(v) for v in zones.values()]) if zones else np.zeros((0, 2))
        if allp.size and (allp[:, 0].min() < -0.1 * w or allp[:, 0].max() > 1.1 * w or
                          allp[:, 1].min() < -0.1 * h or allp[:, 1].max() > 1.1 * h):
            return None, None, 0
        return zones, e, n_in

    def learn(self, image, test_key, zones, video_name="", covered=None):
        """Remember (or update) one set-up.  zones in `image` px."""
        if image is None or not zones or not all(len(v) >= 3 for v in zones.values()):
            return None
        gq, sq = _gray_small(image)
        kq, dq = _features(gq)
        if dq is None or len(kq) < 40:
            return None                           # too plain a picture to recognise again
        small = {k: [[float(x) * sq, float(y) * sq] for x, y in v] for k, v in zones.items()}
        # same set-up already known (same test + zone names, picture matches): update it
        for e in self.entries:
            if e.get("test") != test_key or sorted(e["zones"]) != sorted(small):
                continue
            ke, de = self._entry_features(e)
            H, n_in = self._homography(kq, dq, ke, de)
            if H is not None and n_in >= 60:
                e.update(zones=small, video=video_name, time=time.time(), covered=list(covered or []),
                         uses=int(e.get("uses", 1)) + 1)
                cv2.imwrite(os.path.join(self.folder, e["image"]), gq)
                self._feat.pop(e["id"], None)
                self._save_index()
                return e
        eid = f"L{int(time.time() * 1000)}"
        e = {"id": eid, "image": eid + ".png", "test": test_key, "zones": small, "video": video_name,
             "size": [int(gq.shape[1]), int(gq.shape[0])], "time": time.time(), "uses": 1,
             "covered": list(covered or [])}
        cv2.imwrite(os.path.join(self.folder, e["image"]), gq)
        self.entries.append(e)
        if len(self.entries) > MAX_ENTRIES:              # forget the least used, oldest set-ups
            self.entries.sort(key=lambda d: (d.get("uses", 1), d.get("time", 0)))
            for old in self.entries[:len(self.entries) - MAX_ENTRIES]:
                try:
                    os.remove(os.path.join(self.folder, old["image"]))
                except OSError:
                    pass
                self._feat.pop(old["id"], None)
            self.entries = self.entries[len(self.entries) - MAX_ENTRIES:]
        self._save_index()
        return e

    def forget_all(self):
        for e in self.entries:
            try:
                os.remove(os.path.join(self.folder, e["image"]))
            except OSError:
                pass
        self.entries, self._feat = [], {}
        self._save_index()

    def __len__(self):
        return len(self.entries)


def zone_overlap_ok(zones, w, h):
    """Sanity check: zones cover a reasonable part of the picture."""
    tot = 0.0
    for pts in zones.values():
        a = np.asarray(pts, float)
        if len(a) >= 3:
            tot += abs(0.5 * np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]))
    return 0.02 * w * h <= tot <= 1.5 * w * h if w and h else False


def describe(entry, n_in):
    when = time.strftime("%d %b %Y", time.localtime(entry.get("time", 0))) if entry.get("time") else ""
    src = f" of {entry['video']}" if entry.get("video") else ""
    return f"your saved set-up{src} ({when}, {n_in} matching points)"


_LIB = {}


def library(folder):
    if folder not in _LIB:
        _LIB[folder] = LayoutLibrary(folder)
    return _LIB[folder]
