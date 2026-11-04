"""BehavioralTracker from the command line (no window) -- for scheduled / scripted pipelines.

    BehavioralTracker --cli track    --zones zones.json [--test epm] [--ppc 10.5] [--start 0] [--end 300]
                                     [--out DIR] VIDEO_OR_FOLDER ...
    BehavioralTracker --cli freezing [--settings freezing_settings.json] [--region region.json]
                                     [--out DIR] VIDEO_OR_FOLDER ...
    BehavioralTracker --cli pose     --zones zones.json --pose-suffix DLC_resnet50.csv [--test nor] VIDEO ...
    BehavioralTracker --cli classify --model grooming.btclf [--out time_budget.xlsx] VIDEO_OR_FOLDER ...
    BehavioralTracker --cli features VIDEO_OR_FOLDER ...      (prepare videos for the classifier, e.g. overnight)
    BehavioralTracker --cli --help

Windows: run it with the console Python of the installation, e.g.
    "%LOCALAPPDATA%\\Programs\\BehavioralTracker\\python\\python.exe" "%LOCALAPPDATA%\\Programs\\BehavioralTracker\\app\\main_pro.py" --cli ...
Linux: ~/.local/share/BehavioralTracker/BehavioralTracker --cli ...   (or python3 main_pro.py --cli ... from the source)
macOS: ~/Applications/BehavioralTracker.app/Contents/Resources/python/bin/python3 \\
       ~/Applications/BehavioralTracker.app/Contents/Resources/app/main_pro.py --cli ...

zones.json   = the zones.json BT writes into every tracking results folder ({"zones": {name: [[x, y], ...]},
               "masks": [...], "test": "...", "warp_size": [w, h]}) -- draw the zones once in the app, reuse here.
freezing_settings.json = the settings file of the Freezing page (or any subset of its keys).
region.json  = {"poly": [[x, y], ...], "masks": [[[x, y], ...], ...]} in video pixels (optional).

Every video gets the same files as in the app; one BatchSummary / Freezing_BatchSummary workbook (+ tidy CSV) is
written to --out (default: the results folder of the first video).  Exit code 0 = all videos done, 1 = some failed,
2 = bad arguments.
"""
import argparse
import glob
import json
import os
import sys
import time

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".mpg", ".mpeg", ".m4v", ".flv", ".webm")


def _videos(items):
    out = []
    for it in items:
        if os.path.isdir(it):
            for f in sorted(os.listdir(it)):
                if f.lower().endswith(VIDEO_EXT):
                    out.append(os.path.join(it, f))
        else:
            out.extend(sorted(glob.glob(it)) or [it])
    return [v for v in out if os.path.isfile(v)]


def _track_setup(video, zones_file, test, ppc, start, end, out):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    z = json.load(open(zones_file, encoding="utf-8")) if zones_file else {}
    zones = z.get("zones") or {}
    masks = z.get("masks") or []
    ws = z.get("warp_size") or [w, h]
    sx, sy = w / float(ws[0] or w), h / float(ws[1] or h)
    roi = {n: [(float(x) * sx, float(y) * sy) for x, y in pts] for n, pts in zones.items()}
    msk = [[(float(x) * sx, float(y) * sy) for x, y in m] for m in masks]
    test = test or z.get("test") or "open_field"
    return {"video_path": video, "use_crop": False, "matrix": np.eye(3, dtype=np.float32), "warp_w": w, "warp_h": h,
            "arena": (0, 0, w, h), "roi_names": list(roi), "roi_points": roi, "mask_points": msk,
            "auto_mask_outside_zones": bool(roi), "auto_mask_margin": None, "object_names": [], "object_points": {},
            "interaction_margin_px": 0.0, "background_samples": 100, "tracking_algorithm": "robust",
            "animal_polarity": "auto", "min_contrast": 12.0, "min_area": 15.0, "max_area": 0.0, "max_jump": 100.0,
            "max_gap_s": 0.5, "min_move_px": None, "entry_rule": "paws", "tracking_point": "centre",
            "hold_lost": True, "adapt_background": True, "low_light": "off", "body_length_px": None,
            "ldb_dark_covered": True, "covered_zones": None, "embed_video_px": 0, "entry_body_pct": 80.0,
            "scale_factor": (1.0 / ppc) if ppc else None, "scale_unit": "cm" if ppc else "px",
            "preview_samples": 0, "color_mode": "gray", "start_time": float(start or 0.0),
            "end_time": float(end) if end else 1e9, "output_dir_override": out, "test_type": test,
            "track_nose": test == "nor", "compute_arm_entries": test in ("epm", "y_maze", "t_maze"),
            "compute_alternation": test in ("y_maze", "t_maze"), "stop_condition": None, "stop_value": None,
            "stop_zone_name": "", "zone_groups": {}, "smooth_window": None, "custom_variables": [],
            "bin_size_s": 60.0, "tracking_mode": "single", "num_animals": 1, "workers": None}


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def cmd_track(a, pose=False):
    from tracking.location import process_single_video, process_pose_file, results_root
    from output.batch_summary import write_batch_summary
    vids = _videos(a.videos)
    if not vids:
        _log("No videos found.")
        return 2
    out = a.out or results_root(vids[0])
    os.makedirs(out, exist_ok=True)
    rows, failed = [], 0
    for k, v in enumerate(vids, 1):
        _log(f"[{k}/{len(vids)}] {os.path.basename(v)}")
        try:
            st = _track_setup(v, a.zones, a.test, a.ppc, a.start, a.end, out)
            if pose:
                st["pose_file"] = os.path.splitext(v)[0] + a.pose_suffix if not os.path.isabs(a.pose_suffix) \
                    else a.pose_suffix
                st["track_nose"] = True
                s = process_pose_file(v, st)
            else:
                s = process_single_video(v, st, save_project=False)
            s["Video_name"] = os.path.basename(v)
            rows.append(s)
            _log(f"    done: {s.get('Output_folder', '')}")
        except Exception as exc:
            failed += 1
            _log(f"    FAILED: {exc}")
    if rows:
        p = os.path.join(out, f"BatchSummary_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        write_batch_summary(rows, p)
        _log(f"Summary: {p}")
    return 1 if failed else 0


def cmd_freezing(a):
    from tracking import freezing as FZ
    from output.freezing_report import write_freezing_report, batch_row, write_freezing_batch
    from tracking.location import results_root
    vids = _videos(a.videos)
    if not vids:
        _log("No videos found.")
        return 2
    s = FZ.merged_settings(json.load(open(a.settings, encoding="utf-8")) if a.settings else {})
    region = FZ.region_from_json(json.load(open(a.region, encoding="utf-8"))) if a.region else None
    out = a.out or results_root(vids[0])
    os.makedirs(out, exist_ok=True)
    rows, failed = [], 0
    for k, v in enumerate(vids, 1):
        _log(f"[{k}/{len(vids)}] {os.path.basename(v)}")
        try:
            md = FZ.run_freezing(v, s, region=region)
            res = FZ.analyse(md, s)
            files = write_freezing_report(md, res, v)
            rows.append(batch_row(v, res, folder=files["folder"]))
            _log(f"    freezing {res['summary']['Freezing_pct']:.1f} %  ·  {res['summary']['Result_check']}")
        except Exception as exc:
            failed += 1
            _log(f"    FAILED: {exc}")
    if rows:
        p = os.path.join(out, f"Freezing_BatchSummary_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        write_freezing_batch(rows, p)
        _log(f"Summary: {p}")
    return 1 if failed else 0


def cmd_classify(a):
    from analysis import classifier as C
    vids = _videos(a.videos)
    if not vids:
        _log("No videos found.")
        return 2
    try:
        m = C.BehaviourModel.load(a.model)
    except Exception as exc:
        _log(f"Model not loaded: {exc}")
        return 2
    _log(f"Model: {', '.join(m.behaviours)} ({os.path.basename(a.model)})")
    out, status = C.classify_videos(m, vids, out_xlsx=a.out, log=lambda t: _log("    " + t))
    if out:
        _log(f"Time budget: {out}")
    return 1 if any(st != "done" for st in status.values()) else 0


def cmd_features(a):
    from analysis import classifier as C
    vids = _videos(a.videos)
    if not vids:
        _log("No videos found.")
        return 2
    st = C.prepare_features(vids, log=lambda t: _log("    " + t))
    return 1 if any(x not in ("ready", "cached") for x in st.values()) else 0


def main(argv):
    ap = argparse.ArgumentParser(prog="BehavioralTracker --cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("track", "pose"):
        t = sub.add_parser(name, help="zone tracking" if name == "track" else "zone analysis from pose CSV files")
        t.add_argument("videos", nargs="+")
        t.add_argument("--zones", required=True, help="zones.json (written by BT into every results folder)")
        t.add_argument("--test", default=None, help="open_field, epm, y_maze, t_maze, light_dark, nor, social, custom")
        t.add_argument("--ppc", type=float, default=None, help="pixels per cm (distances in cm)")
        t.add_argument("--start", type=float, default=0.0)
        t.add_argument("--end", type=float, default=None)
        t.add_argument("--out", default=None)
        if name == "pose":
            t.add_argument("--pose-suffix", default="DLC.csv",
                           help="pose CSV next to each video: <video name><suffix> (or a full path)")
    f = sub.add_parser("freezing", help="freezing analysis")
    f.add_argument("videos", nargs="+")
    f.add_argument("--settings", default=None)
    f.add_argument("--region", default=None)
    f.add_argument("--out", default=None)
    c = sub.add_parser("classify", help="find behaviours with a classifier model (.btclf from Event Scoring)")
    c.add_argument("videos", nargs="+")
    c.add_argument("--model", required=True, help="the .btclf model saved in Event Scoring ▸ Classifier")
    c.add_argument("--out", default=None, help="the time-budget workbook (.xlsx); default: the results folder")
    e = sub.add_parser("features", help="prepare videos for the classifier (analyse and cache, e.g. overnight)")
    e.add_argument("videos", nargs="+")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    if a.cmd is None:
        ap.print_help()
        return 2
    # the launchers start the program in its own folder: relative paths are the user's folder (BT_START_DIR)
    base = os.environ.get("BT_START_DIR") or os.getcwd()

    def ab(p):
        return p if (p is None or os.path.isabs(p)) else os.path.join(base, p)
    a.videos = [ab(v) for v in a.videos]
    for k in ("zones", "settings", "region", "out", "model"):
        if getattr(a, k, None):
            setattr(a, k, ab(getattr(a, k)))
    for k in ("zones", "settings", "region", "model"):
        if getattr(a, k, None) and not os.path.isfile(getattr(a, k)):
            _log(f"File not found: {getattr(a, k)}")
            return 2
    import matplotlib
    matplotlib.use("Agg")
    if a.cmd == "track":
        return cmd_track(a)
    if a.cmd == "pose":
        return cmd_track(a, pose=True)
    if a.cmd == "classify":
        return cmd_classify(a)
    if a.cmd == "features":
        return cmd_features(a)
    return cmd_freezing(a)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
