"""Protected BehavioralTracker files.

  .bta   BT Analysis file        one video: tracking data, results, run information and the audit trail
  .btms  BT Manual Scoring file  one event-scoring session (behaviours, events) and its audit trail
  .btf   BT Freezing file        one freezing analysis: settings, motion data, results, audit trail (+ video copy)

Only BehavioralTracker can open them: the content is encrypted with a key inside the program (Fernet =
AES-128-CBC + HMAC-SHA256).  A file changed outside the program fails the HMAC check and is refused.
The audit trail is a hash chain -- every entry stores the hash of the one before it -- so entries cannot be
removed or edited without it showing.
"""
import base64
import datetime
import getpass
import hashlib
import io
import json
import math
import os
import platform
import shutil
import tempfile
import time
import zipfile

import numpy as np

APP = "BehavioralTracker 1.0.0"
MAGIC_ANALYSIS = b"BTA1\n"
MAGIC_SCORING = b"BTMS1\n"
MAGIC_FREEZING = b"BTF1\n"
EXT_ANALYSIS = ".bta"
EXT_SCORING = ".btms"
EXT_FREEZING = ".btf"
_SECRET = b"BehavioralTracker/project-files/v1/9f4c2e7a5b13d8e6"


class ProjectFileError(Exception):
    pass


def _fernet():
    from cryptography.fernet import Fernet
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(_SECRET).digest()))


# ----------------------------------------------------------------------------- json helpers
def jsonable(o):
    """numpy / tuples / sets -> plain JSON types (NaN kept as NaN; BehavioralTracker reads it back)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return o
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    return str(o)


def _now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# The licence of this copy (licensed edition only; set by the licence check before the window opens).
_LICENCE = {}


def licence_number(payload):
    """The licence number printed on a licence; licences issued before numbers existed get a fixed code
    made from the licence itself."""
    no = str(payload.get("licence_no") or "").strip()
    if no:
        return no
    h = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return f"BT-{h[:4]}-{h[4:8]}".upper()


def set_licence(payload):
    """Called once by the licence check with the verified licence (dict)."""
    _LICENCE.clear()
    if isinstance(payload, dict) and payload.get("licensee"):
        _LICENCE.update(licensee=str(payload["licensee"]), licence_no=licence_number(payload),
                        issued=payload.get("issued"), expires=payload.get("expires"),
                        machine_id=payload.get("machine_id"))


def licence_info():
    """{} for the unlicensed edition, else licensee / licence_no / issued / expires / machine_id."""
    return dict(_LICENCE)


def _user():
    """Licensed edition: the licence holder's name.  Unlicensed edition: the computer's user name."""
    if _LICENCE.get("licensee"):
        return _LICENCE["licensee"]
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------- container
# File = magic + Fernet token (zip: payload.json + blobs) [+ VIDEO_MARK + encrypted video-copy pieces].
# The pieces are AES-CTR encrypted; their length, nonce and SHA-256 are kept in the (authenticated) payload,
# so a changed or swapped video copy is refused just like a changed payload.
VIDEO_MARK = b"\n#BTVIDEO1\n"          # cannot occur inside a Fernet token (url-safe base64)


def _video_cipher(nonce):
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    key = hashlib.sha256(_SECRET + b"/video-copy").digest()
    return Cipher(algorithms.AES(key), modes.CTR(nonce))


def _encrypt_copy(copy):
    """encode_copy() result -> (payload entry, [encrypted bytes, ...])."""
    segs, chunks = [], []
    for sg in copy["segments"]:
        with open(sg["file"], "rb") as f:
            plain = f.read()
        nonce = os.urandom(16)
        enc = _video_cipher(nonce).encryptor()
        ct = enc.update(plain) + enc.finalize()
        segs.append({"first": int(sg["first"]), "last": int(sg["last"]), "length": len(ct),
                     "nonce": base64.b64encode(nonce).decode("ascii"), "sha256": hashlib.sha256(ct).hexdigest()})
        chunks.append(ct)
    info = {"size": list(copy["size"]), "fps": float(copy["fps"]), "frames": int(copy["frames"]),
            "codec": "mp4v", "segments": segs, "made": _now()}
    return info, chunks


def _read_copy_chunks(ref, info):
    """The encrypted pieces of an existing file (to write them again unchanged)."""
    out = []
    with open(ref["file"], "rb") as f:
        off = int(ref["base"])
        for sg in info["segments"]:
            f.seek(off)
            ct = f.read(int(sg["length"]))
            if len(ct) != int(sg["length"]) or hashlib.sha256(ct).hexdigest() != sg["sha256"]:
                raise ProjectFileError("The video copy inside this file is damaged.")
            out.append(ct)
            off += len(ct)
    return out


def write_container(path, magic, payload, blobs=None, video_chunks=None):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("payload.json", json.dumps(jsonable(payload), allow_nan=True))
        for name, data in (blobs or {}).items():
            z.writestr(f"blobs/{name}", data)
    token = _fernet().encrypt(buf.getvalue())
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(magic + token)
        if video_chunks:
            f.write(VIDEO_MARK)
            for ct in video_chunks:
                f.write(ct)
    os.replace(tmp, path)
    return str(path)


def read_container(path, magic):
    from cryptography.fernet import InvalidToken
    with open(path, "rb") as f:
        data = f.read()
    if not data.startswith(magic):
        raise ProjectFileError("This is not a BehavioralTracker file of this type.")
    cut = data.find(VIDEO_MARK, len(magic))
    token = data[len(magic):cut] if cut >= 0 else data[len(magic):]
    try:
        raw = _fernet().decrypt(token)
    except InvalidToken:
        raise ProjectFileError("This file was changed outside BehavioralTracker or is damaged, so it cannot be "
                               "opened.")
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        payload = json.loads(z.read("payload.json"))
        blobs = {n[len("blobs/"):]: z.read(n) for n in z.namelist() if n.startswith("blobs/")}
    if cut >= 0 and payload.get("video_copy"):
        payload["_video_ref"] = {"file": str(path), "base": cut + len(VIDEO_MARK),
                                 "total": len(data) - cut - len(VIDEO_MARK)}
    return payload, blobs


def has_video_copy(data):
    return bool(data.get("video_copy") and data.get("_video_ref"))


def video_copy_dir():
    d = os.path.join(tempfile.gettempdir(), "BehavioralTracker_video_copies")
    os.makedirs(d, exist_ok=True)
    return d


def _clean_copy_cache(keep=None, max_age_s=2 * 86400):
    root = video_copy_dir()
    now = time.time()
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name != keep and os.path.isdir(p):
            try:
                if now - os.path.getmtime(p) > max_age_s:
                    shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass


def extract_video_copy(data):
    """Decrypt the video copy of an opened BT file into a temporary folder (once).
    Returns [{"first", "last", "file"}] or None."""
    info, ref = data.get("video_copy"), data.get("_video_ref")
    if not info or not ref:
        return None
    key = hashlib.sha256("".join(s["sha256"] for s in info["segments"]).encode()).hexdigest()[:24]
    folder = os.path.join(video_copy_dir(), key)
    out = []
    files = [os.path.join(folder, f"part{i:02d}.mp4") for i in range(len(info["segments"]))]
    if not all(os.path.exists(f) for f in files):
        chunks = _read_copy_chunks(ref, info)
        os.makedirs(folder, exist_ok=True)
        for sg, ct, fn in zip(info["segments"], chunks, files):
            dec = _video_cipher(base64.b64decode(sg["nonce"])).decryptor()
            with open(fn + ".tmp", "wb") as f:
                f.write(dec.update(ct) + dec.finalize())
            os.replace(fn + ".tmp", fn)
    try:
        os.utime(folder, None)
    except OSError:
        pass
    _clean_copy_cache(keep=key)
    for sg, fn in zip(info["segments"], files):
        out.append({"first": int(sg["first"]), "last": int(sg["last"]), "file": fn})
    return out


# ----------------------------------------------------------------------------- audit trail
def _entry_hash(entry):
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(jsonable(body), sort_keys=True, allow_nan=True).encode("utf-8")).hexdigest()


def audit_add(trail, action, details=None, user=None):
    """Append one entry (returns the trail).  action = short text; details = dict shown in the trail."""
    prev = trail[-1]["hash"] if trail else ""
    e = {"n": len(trail) + 1, "time": _now(), "user": user or _user(), "computer": platform.node(),
         "app": APP, "action": action, "details": jsonable(details or {}), "prev": prev}
    if _LICENCE.get("licence_no"):
        e["licence"] = _LICENCE["licence_no"]
    e["hash"] = _entry_hash(e)
    trail.append(e)
    return trail


def audit_verify(trail):
    """(ok, first_bad_index | None)."""
    prev = ""
    for i, e in enumerate(trail):
        if e.get("prev") != prev or _entry_hash(e) != e.get("hash"):
            return False, i
        prev = e["hash"]
    return True, None


# ----------------------------------------------------------------------------- BT Analysis file
def _video_info(raw):
    p = raw["video_path"]
    info = {"path": p, "name": os.path.basename(p), "fps": raw["fps"], "width": raw["frame_w"],
            "height": raw["frame_h"], "frames": raw.get("total_frames")}
    try:
        st = os.stat(p)
        info["size_bytes"] = st.st_size
        info["modified"] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    except OSError:
        pass
    return info


def _setup_for_file(setup):
    s = {k: v for k, v in setup.items() if not str(k).startswith("_") and k not in ("stop_condition_cb",)}
    if s.get("matrix") is not None:
        s["matrix"] = np.asarray(s["matrix"], float).tolist()
    return jsonable(s)


def _key_results(summary):
    keep = ("Total_distance_cm", "Total_distance_pixels", "Tracking_quality_percent", "Anxiety_index",
            "Spontaneous_alternation_pct", "Discrimination_index", "Discrimination_ratio", "Open_arm_time_pct",
            "Center_time_pct", "Light_time_pct", "Corrected_frames")
    return {k: summary[k] for k in keep if k in summary}


def analysis_path(raw):
    return os.path.join(raw["output_dir"], f"{raw.get('video_name') or 'analysis'}{EXT_ANALYSIS}")


RESULT_EXTS = {".xlsx", ".csv", ".png", ".jpg", ".json", ".txt", ".pdf"}


def collect_results(folder, max_file=60 * 2 ** 20, max_total=400 * 2 ** 20):
    """Every result file of one analysis (Excel, CSV, charts, previews...) as {relative path: bytes} -- they
    travel inside the .bta, so the file alone holds the complete analysis."""
    out, total = {}, 0
    if not folder or not os.path.isdir(folder):
        return out
    for root, dirs, files in os.walk(folder):
        if os.path.relpath(root, folder).count(os.sep) >= 1:      # the folder + one level (preview/)
            dirs[:] = []
        for name in sorted(files):
            ext = os.path.splitext(name)[1].lower()
            if ext not in RESULT_EXTS or name.endswith(".tmp"):
                continue
            fp = os.path.join(root, name)
            try:
                size = os.path.getsize(fp)
                if size > max_file or total + size > max_total:
                    continue
                with open(fp, "rb") as f:
                    out[os.path.relpath(fp, folder).replace(os.sep, "/")] = f.read()
                total += size
            except OSError:
                continue
    return out


def extract_results(data, folder):
    """Write the result files stored in an opened .bta into `folder`; returns how many were written."""
    n = 0
    for rel, blob in (data.get("_results") or {}).items():
        rel = rel.replace("\\", "/")
        if rel.startswith("/") or ".." in rel.split("/"):
            continue
        fp = os.path.join(folder, *rel.split("/"))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "wb") as f:
            f.write(blob)
        n += 1
    return n


def _experiment(setup):
    """Who / what / when of the experiment, from the run's setup."""
    subj = setup.get("subject") or ""
    design = [r for r in (setup.get("experimental_design") or []) if subj and str(r.get("Animal ID")) == str(subj)]
    return {"name": setup.get("experiment_name") or "", "investigator": setup.get("investigator") or "",
            "description": setup.get("description") or "", "notes": setup.get("notes") or "",
            "date": setup.get("experiment_date") or "", "subject": subj, "design": design}


def make_analysis_copy(raw, setup, progress=None):
    """A smaller copy of the analysed frames (straightened like the analysis) -- or None."""
    px = int(setup.get("embed_video_px", 640) or 0) if setup.get("embed_video_px") is not None else 640
    vp = raw.get("video_path")
    frames = raw.get("frames") or []
    if px <= 0 or not vp or not os.path.exists(vp) or not frames:
        return None
    from tracking.review_video import encode_copy
    warp = None
    if raw.get("need_warp") and setup.get("matrix") is not None:
        import cv2
        m = np.asarray(setup["matrix"], np.float32)
        ww, wh = int(raw["warp_w"]), int(raw["warp_h"])

        def warp(f):
            return cv2.warpPerspective(f, m, (ww, wh))
    return encode_copy(vp, int(frames[0]), int(frames[-1]), warp=warp, max_side=px, fps=float(raw["fps"]),
                       progress=progress)


def save_new_analysis(raw, setup, summary, path=None, progress=None, copy="make"):
    """Write the .bta file right after a tracking run (also used by batch processing).  `copy` = a video copy
    made already (chambers: all copies come from one reading of the video) or None for no copy."""
    path = path or analysis_path(raw)
    if isinstance(copy, str):
        copy = None
        try:
            copy = make_analysis_copy(raw, setup, progress)
        except Exception as exc:                              # the copy is optional
            print(f"Video copy skipped: {exc}")
    raw_store = {k: v for k, v in raw.items() if k != "background_bgr"}
    trail = []
    audit_add(trail, "Tracking run", {
        "video": os.path.basename(raw["video_path"]), "test": setup.get("test_type"),
        "engine": raw.get("algorithm"), "time_window_s": [raw.get("start_time"), raw.get("end_time")],
        "zones": list((setup.get("roi_points") or {}).keys()), "entry_rule": summary.get("Entry_rule"),
        "frames": len(raw.get("frames") or []), "mode": raw.get("live_mode") or "video file",
        "experiment": setup.get("experiment_name"),
        "investigator": setup.get("investigator")})
    audit_add(trail, "Results computed", _key_results(summary))
    payload = {"format": "BT Analysis", "version": 2, "app": APP, "created": _now(),
               "video": _video_info(raw), "experiment": _experiment(setup), "setup": _setup_for_file(setup),
               "raw": raw_store, "summary": summary, "corrections": [], "audit": trail}
    if _LICENCE:
        payload["licence"] = {"licensee": _LICENCE.get("licensee"), "licence_no": _LICENCE.get("licence_no")}
    blobs = {}
    bg = raw.get("background_bgr")
    if bg is not None:
        import cv2
        ok, png = cv2.imencode(".png", bg)
        if ok:
            blobs["background.png"] = png.tobytes()
    for rel, data in collect_results(raw.get("output_dir")).items():
        blobs["results/" + rel] = data
    chunks = None
    if copy:
        try:
            payload["video_copy"], chunks = _encrypt_copy(copy)
        finally:
            shutil.rmtree(copy.get("dir") or "", ignore_errors=True)
    return write_container(path, MAGIC_ANALYSIS, payload, blobs, video_chunks=chunks)


def load_analysis(path):
    """dict with format/video/setup/raw/summary/corrections/audit (+ raw['background_bgr'] decoded)."""
    payload, blobs = read_container(path, MAGIC_ANALYSIS)
    raw = payload.get("raw") or {}
    if "background.png" in blobs:
        import cv2
        raw["background_bgr"] = cv2.imdecode(np.frombuffer(blobs["background.png"], np.uint8), cv2.IMREAD_COLOR)
    payload["raw"] = raw
    payload["_results"] = {k[len("results/"):]: v for k, v in blobs.items() if k.startswith("results/")}
    payload["_path"] = str(path)
    return payload


def save_analysis(data, path=None):
    """Write an opened .bta again (after corrections); keeps every audit entry."""
    path = path or data.get("_path")
    raw = dict(data.get("raw") or {})
    bg = raw.pop("background_bgr", None)
    payload = {k: v for k, v in data.items() if not k.startswith("_")}
    payload["raw"] = raw
    payload["saved"] = _now()
    blobs = {}
    if bg is not None:
        import cv2
        ok, png = cv2.imencode(".png", bg)
        if ok:
            blobs["background.png"] = png.tobytes()
    fresh = collect_results((data.get("summary") or {}).get("Output_folder") or raw.get("output_dir"))
    results = fresh or data.get("_results") or {}
    data["_results"] = results
    for rel, b in results.items():
        blobs["results/" + rel] = b
    chunks = None
    if data.get("video_copy") and data.get("_video_ref"):          # the video copy goes along unchanged
        chunks = _read_copy_chunks(data["_video_ref"], data["video_copy"])
    elif "video_copy" in payload:
        payload.pop("video_copy", None)
    out = write_container(path, MAGIC_ANALYSIS, payload, blobs, video_chunks=chunks)
    if chunks:
        data["_video_ref"] = {"file": str(out), "base": _video_base(out)}
    return out


def _video_base(path):
    """Where the video pieces start in a written file."""
    with open(path, "rb") as f:
        data = f.read()
    cut = data.find(VIDEO_MARK)
    return cut + len(VIDEO_MARK) if cut >= 0 else None


# ----------------------------------------------------------------------------- BT Manual Scoring file
def save_scoring(path, video_path, inventory, events, fps, n_frames, trail, extra=None, embed_px=640,
                 previous=None, progress=None):
    """Write a .btms file.  embed_px > 0: a smaller copy of the whole video goes inside (made once; when the
    file was opened from a .btms that already holds one, `previous` = that opened file and it is kept)."""
    payload = {"format": "BT Manual Scoring", "version": 1, "app": APP, "saved": _now(),
               "video": {"path": video_path, "name": os.path.basename(video_path or ""), "fps": fps,
                         "frames": n_frames},
               "inventory": inventory, "events": events, "audit": trail}
    if extra:
        payload.update(extra)
    chunks = None
    if previous and previous.get("video_copy") and previous.get("_video_ref") and \
            int(previous["video_copy"].get("frames") or 0) >= int(n_frames or 0) - 1:
        payload["video_copy"] = previous["video_copy"]
        chunks = _read_copy_chunks(previous["_video_ref"], previous["video_copy"])
    elif embed_px and video_path and os.path.exists(video_path) and n_frames:
        try:
            from tracking.review_video import encode_copy
            copy = encode_copy(video_path, 0, int(n_frames) - 1, max_side=int(embed_px), fps=float(fps or 30.0),
                               progress=progress)
        except Exception as exc:
            print(f"Video copy skipped: {exc}")
            copy = None
        if copy:
            try:
                payload["video_copy"], chunks = _encrypt_copy(copy)
            finally:
                shutil.rmtree(copy.get("dir") or "", ignore_errors=True)
    out = write_container(path, MAGIC_SCORING, payload, video_chunks=chunks)
    if previous is not None and chunks:
        previous["video_copy"] = payload["video_copy"]
        previous["_video_ref"] = {"file": str(out), "base": _video_base(out)}
    return out


def load_scoring(path):
    payload, _blobs = read_container(path, MAGIC_SCORING)
    payload["_path"] = str(path)
    return payload


# ----------------------------------------------------------------------------- BT Freezing file
MD_ARRAYS = ("h_reg", "h_bg", "n_bg", "rate", "h_disc", "n_disc", "valid", "read", "frames", "xs", "ys", "seen",
             "kp", "h_body", "n_body", "h_head", "n_head", "h_legs", "n_legs", "h_tail", "n_tail", "bright")
MD_META = ("path", "fps", "lag", "start_frame", "n_reg", "radius_work", "disc_len", "tracked", "body_area_full",
           "body_len_full", "found_pct", "polarity", "polarity_warning", "frame_w", "frame_h", "total_frames",
           "colour", "body_area_dots", "body_len_dots", "disc_provisional", "core_frac", "core_ratio")
FREEZE_KEY_SETTINGS = ("motion_area", "colour", "cutoff_auto", "cutoff", "threshold_pct", "min_freeze_s",
                       "ignore_blips_s",
                       "light_guard", "polarity", "min_contrast", "body_length_px", "start_time", "end_time",
                       "bin_s", "pre_s")


def freezing_path(video_path, folder=None):
    from tracking.freezing import default_output_dir
    name = os.path.splitext(os.path.basename(str(video_path)))[0]
    return os.path.join(folder or default_output_dir(video_path), f"{name}{EXT_FREEZING}")


def _md_blob(md):
    buf = io.BytesIO()
    arrays = {k: getattr(md, k) for k in MD_ARRAYS if getattr(md, k, None) is not None}
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def md_from_file(meta, blob, region):
    """MotionData back from a .btf (no video reading)."""
    from tracking.freezing import Geometry, MotionData
    md = MotionData()
    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
        for k in MD_ARRAYS:
            setattr(md, k, z[k] if k in z.files else None)
    for k in MD_META:
        setattr(md, k, meta.get(k))
    md.geo = Geometry(int(meta["frame_w"]), int(meta["frame_h"]), region)
    md.fps = float(md.fps)
    md.lag, md.start_frame, md.n_reg = int(md.lag), int(md.start_frame), int(md.n_reg)
    md.radius_work = float(md.radius_work or 0.0)
    md.disc_len = float(md.disc_len or 0.0)
    md.body_area_full = float(md.body_area_full or 0.0)
    md.body_len_full = float(md.body_len_full or 0.0)
    md.body_area_dots = float(getattr(md, "body_area_dots", None) or 0.0)    # older .btf files: not saved
    md.body_len_dots = float(getattr(md, "body_len_dots", None) or 0.0)
    md.found_pct = float("nan") if md.found_pct is None else float(md.found_pct)
    md.tracked = bool(md.tracked)
    md.colour = md.colour or "gray"
    md.disc_provisional = bool(getattr(md, "disc_provisional", None) or False)
    for k in ("core_frac", "core_ratio"):                    # older .btf files: not saved
        v = getattr(md, k, None)
        setattr(md, k, float("nan") if v is None else float(v))
    if md.seen is None:
        md.seen = np.zeros(len(md.frames), bool)
    return md


def _freeze_results(result):
    s = result["summary"]
    return {k: s.get(k) for k in ("Freezing_pct", "Freezing_time_s", "Freezing_bouts", "Latency_first_freeze_s",
                                  "Motion_cutoff", "Freezing_threshold_pct", "Min_freeze_s")}


def _fz_region_json(region):
    from tracking.freezing import region_json
    return region_json(region)


def save_freezing(md, result, video_path, region=None, settings=None, subject="", group="", path=None,
                  results_folder=None, embed_px=640, previous=None, progress=None, extra=None):
    """Write the .btf of one freezing analysis.  `previous` = the opened .btf this one replaces (its audit trail
    goes on, its video copy is kept when it covers the same frames).  Returns the path."""
    path = path or freezing_path(video_path, results_folder)
    s = {k: v for k, v in (settings or result.get("settings") or {}).items()}
    trail = list((previous or {}).get("audit") or [])
    if not trail:
        audit_add(trail, "Freezing analysis run", {
            "video": os.path.basename(str(video_path)), "frames": len(md.frames),
            "time_window_s": [float(md.frames[0] / md.fps), float((md.frames[-1] + 1) / md.fps)],
            "region": _fz_region_json(region) or "whole frame", "tracked": bool(md.tracked),
            "settings": {k: s.get(k) for k in FREEZE_KEY_SETTINGS}})
    else:
        old = (previous or {}).get("settings") or {}
        changed = {k: [old.get(k), s.get(k)] for k in FREEZE_KEY_SETTINGS if old.get(k) != s.get(k)}
        old_eps = [(e.get("name"), e.get("start"), e.get("end")) for e in old.get("epochs") or []]
        new_eps = [(e.get("name"), e.get("start"), e.get("end")) for e in s.get("epochs") or []]
        if old_eps != new_eps:
            changed["epochs"] = [len(old_eps), len(new_eps)]
        if changed:
            audit_add(trail, "Settings changed", changed)
    audit_add(trail, "Results computed", _freeze_results(result))
    info = {"path": str(video_path), "name": os.path.basename(str(video_path)), "fps": md.fps,
            "width": md.frame_w, "height": md.frame_h, "frames": md.total_frames}
    try:
        st = os.stat(video_path)
        info["size_bytes"] = st.st_size
        info["modified"] = datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    except OSError:
        pass
    meta = {k: getattr(md, k, None) for k in MD_META}
    payload = {"format": "BT Freezing", "version": 1, "app": APP, "created": (previous or {}).get("created") or _now(),
               "saved": _now(), "video": info, "experiment": {"subject": subject or "", "group": group or ""},
               "settings": s, "region": _fz_region_json(region), "md_meta": meta,
               "summary": result["summary"], "epochs": result["epochs"], "groups": result.get("groups") or [],
               "bins": result["bins"], "bouts": result["bouts"], "audit": trail}
    if extra:
        payload.update(extra)
    if _LICENCE:
        payload["licence"] = {"licensee": _LICENCE.get("licensee"), "licence_no": _LICENCE.get("licence_no")}
    blobs = {"motion.npz": _md_blob(md)}
    for rel, data in collect_results(results_folder or os.path.dirname(path)).items():
        if not rel.lower().endswith(EXT_FREEZING):
            blobs["results/" + rel] = data
    first, last = int(md.frames[0]), int(md.frames[-1])
    chunks = None
    pv = previous or {}
    if pv.get("video_copy") and pv.get("_video_ref") and pv.get("_copy_span") == [first, last]:
        payload["video_copy"] = pv["video_copy"]
        payload["copy_span"] = [first, last]
        chunks = _read_copy_chunks(pv["_video_ref"], pv["video_copy"])
    elif embed_px and video_path and os.path.exists(video_path):
        try:
            from tracking.review_video import encode_copy
            copy = encode_copy(str(video_path), first, last, max_side=int(embed_px), fps=float(md.fps),
                               progress=progress)
        except Exception as exc:                              # the copy is optional
            print(f"Video copy skipped: {exc}")
            copy = None
        if copy:
            try:
                payload["video_copy"], chunks = _encrypt_copy(copy)
                payload["copy_span"] = [first, last]
            finally:
                shutil.rmtree(copy.get("dir") or "", ignore_errors=True)
    out = write_container(path, MAGIC_FREEZING, payload, blobs, video_chunks=chunks)
    if previous is not None:
        previous.update({k: v for k, v in payload.items()})
        previous["_path"] = str(out)
        previous["_copy_span"] = payload.get("copy_span")
        if chunks:
            previous["_video_ref"] = {"file": str(out), "base": _video_base(out)}
    return out


def load_freezing(path):
    """dict with format/video/settings/region/summary/epochs/bins/bouts/audit + 'md' (MotionData) + '_results'."""
    payload, blobs = read_container(path, MAGIC_FREEZING)
    if "motion.npz" not in blobs:
        raise ProjectFileError("This BT Freezing file has no motion data.")
    from tracking.freezing import region_from_json
    region = region_from_json(payload.get("region"))
    payload["region"] = region
    payload["md"] = md_from_file(payload.get("md_meta") or {}, blobs["motion.npz"], region)
    payload["_results"] = {k[len("results/"):]: v for k, v in blobs.items() if k.startswith("results/")}
    payload["_path"] = str(path)
    payload["_copy_span"] = payload.get("copy_span")
    return payload
