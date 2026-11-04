"""Camera discovery with the cameras' real names (like the source list in OBS Studio).

list_cameras() -> [Camera(label, index, backend)], one entry per camera the system knows,
labelled with the device's own name ("Integrated Webcam", "Logitech C920", ...).
open_camera()  -> an opened cv2.VideoCapture for one of them, at the requested resolution.

Windows : DirectShow device list (the order OpenCV's CAP_DSHOW uses) -- USB / external
          cameras appear next to the laptop camera.
Linux   : /sys/class/video4linux (capture nodes only), opened with V4L2.
macOS   : system camera list, opened with AVFoundation.
When no name can be read, cameras are found by trying them and labelled "Camera 1, 2, ...".
"""
import glob
import json
import os
import subprocess
import sys
from collections import namedtuple

import cv2

Camera = namedtuple("Camera", "label index backend")

RESOLUTIONS = [("Camera default", None), ("640 × 480", (640, 480)), ("1280 × 720 (HD)", (1280, 720)),
               ("1920 × 1080 (Full HD)", (1920, 1080))]


def _unique(names):
    """Two identical cameras -> 'Name' and 'Name (2)'."""
    seen, out = {}, []
    for n in names:
        n = (n or "").strip() or "Camera"
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


# --------------------------------------------------------------------------- Windows
def _windows_names():
    """DirectShow video-input names, in DirectShow order (= the index cv2.CAP_DSHOW opens)."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        return list(FilterGraph().get_input_devices())
    except Exception:
        return []


# --------------------------------------------------------------------------- Linux
def _linux_cameras():
    cams = []
    for dev in sorted(glob.glob("/sys/class/video4linux/video*"), key=lambda p: int(p.rsplit("video", 1)[1])):
        num = int(dev.rsplit("video", 1)[1])
        try:
            with open(os.path.join(dev, "index")) as f:
                if f.read().strip() not in ("", "0"):       # metadata node of a camera, not a picture source
                    continue
        except OSError:
            pass
        try:
            with open(os.path.join(dev, "name")) as f:
                name = f.read().strip()
        except OSError:
            name = f"Camera {num}"
        cams.append((name, num))
    labels = _unique([n for n, _ in cams])
    return [Camera(lab, num, cv2.CAP_V4L2) for lab, (_n, num) in zip(labels, cams)]


# --------------------------------------------------------------------------- macOS
def _mac_names():
    try:
        out = subprocess.run(["/usr/sbin/system_profiler", "SPCameraDataType", "-json"], capture_output=True,
                             text=True, timeout=10).stdout
        return [c.get("_name", "") for c in json.loads(out).get("SPCameraDataType", [])]
    except Exception:
        return []


# --------------------------------------------------------------------------- common
def _probe(backend, max_probe=6):
    """Try camera numbers 0..max_probe-1 and keep the ones that open and deliver a picture."""
    found = []
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, backend)
        try:
            if cap.isOpened():
                ok, _f = cap.read()
                if ok:
                    found.append(i)
        finally:
            cap.release()
    return found


def list_cameras(max_probe=6):
    if sys.platform == "win32":
        names = _windows_names()
        if names:
            return [Camera(lab, i, cv2.CAP_DSHOW) for i, lab in enumerate(_unique(names))]
        return [Camera(f"Camera {k + 1}", i, cv2.CAP_DSHOW) for k, i in enumerate(_probe(cv2.CAP_DSHOW, max_probe))]
    if sys.platform.startswith("linux"):
        cams = _linux_cameras()
        if cams:
            return cams
        return [Camera(f"Camera {k + 1}", i, cv2.CAP_V4L2) for k, i in enumerate(_probe(cv2.CAP_V4L2, max_probe))]
    if sys.platform == "darwin":
        names = _unique(_mac_names())
        idx = _probe(cv2.CAP_AVFOUNDATION, max(max_probe, len(names)))
        if names and len(names) == len(idx):
            return [Camera(n, i, cv2.CAP_AVFOUNDATION) for n, i in zip(names, idx)]
        return [Camera(f"Camera {k + 1}", i, cv2.CAP_AVFOUNDATION) for k, i in enumerate(idx)]
    return [Camera(f"Camera {k + 1}", i, cv2.CAP_ANY) for k, i in enumerate(_probe(cv2.CAP_ANY, max_probe))]


def open_camera(cam, size=None, fps=None):
    """Open one Camera; size=(w, h) asks for that resolution (the camera picks the nearest it supports)."""
    cap = cv2.VideoCapture(int(cam.index), int(cam.backend))
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(cam.index))                # last resort: the system's default backend
    if not cap.isOpened():
        return None
    if size:
        if size[0] >= 1280:                                    # USB cameras need MJPG for HD at full speed
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(size[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(size[1]))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def unique_path(folder, stem, ext=".mp4"):
    """folder/stem.mp4, or folder/stem_2.mp4 ... if that already exists (never overwrites a recording)."""
    stem = "".join(c if (c.isalnum() or c in " ._-()") else "_" for c in stem).strip() or "Recording"
    path = os.path.join(folder, stem + ext)
    k = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}_{k}{ext}")
        k += 1
    return path
