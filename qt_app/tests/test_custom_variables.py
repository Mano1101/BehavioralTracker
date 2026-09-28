"""
Regression test for "Custom variables / derived columns" (Setup page,
Standard Tracking only) -- Upgrade Plan Tier 2 #7, EthoVision's own custom
variables: one or more user-written "name = expression" lines, evaluated
against the columns already in raw_tracking.csv (plus a couple of known
scalars like scale_factor/FPS), and added as new columns to that CSV (and
the exported Excel) -- e.g. "distance_cm = Distance_pixels / scale_factor".

Three things are covered:

1. parse_custom_variables() -- pure text parsing, no video/DataFrame
   needed: valid lines, comments/blank lines skipped, and the various
   malformed-line error messages.

2. safe_eval_expr()/apply_custom_variables() -- the actual (sandboxed)
   expression engine: correct arithmetic/vectorized results on both plain
   scalars and pandas Series, chaining (a later definition using an
   earlier one's new column), collision/unknown-name handling that skips
   just the one bad definition with a warning rather than raising, and --
   since the whole point of hand-rolling this instead of calling eval()/
   df.eval() is to keep it safely sandboxed -- that the classic escape
   hatches (attribute access, subscripting, arbitrary calls, the
   __class__/__subclasses__ trick) are all rejected.

3. process_single_video's wiring -- with setup["custom_variables"] unset/
   empty, raw_tracking.csv is completely unchanged (no new columns, same
   as before this feature existed); set, the new columns appear with the
   right values, chaining works end to end, and a bad definition mixed in
   with good ones is skipped (with a warning surfaced in the summary)
   without taking down the good ones or the run itself.

Run directly (part 3 uses OpenCV video I/O like test_trajectory_smoothing.py):
    python3.12 qt_app/tests/test_custom_variables.py
"""
import _pathsetup  # noqa: F401 (bare import -- see its docstring; sets sys.path/cwd)

import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

from analysis.custom_variables import parse_custom_variables, safe_eval_expr, apply_custom_variables
from tracking.location import process_single_video, identity_transform

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


def raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return True
    except Exception:
        return False
    return False


# ------------------------------------------------------------------
# 1) parse_custom_variables() -- pure text parsing.
# ------------------------------------------------------------------
check("blank text parses to no definitions", parse_custom_variables("") == ([], None))
check("whitespace-only text parses to no definitions", parse_custom_variables("   \n  ") == ([], None))

defs, err = parse_custom_variables("distance_cm = Distance_pixels / scale_factor")
check("a single valid line parses correctly",
      err is None and defs == [("distance_cm", "Distance_pixels / scale_factor")])

defs, err = parse_custom_variables(
    "# a comment line\n"
    "distance_cm = Distance_pixels / scale_factor\n"
    "\n"
    "speed_cm_s = distance_cm * FPS\n"
)
check("comments and blank lines are skipped, multiple lines both parse",
      err is None and defs == [("distance_cm", "Distance_pixels / scale_factor"),
                                ("speed_cm_s", "distance_cm * FPS")])

defs, err = parse_custom_variables("distance_cm Distance_pixels / scale_factor")
check("a line missing '=' is a parse error", defs == [] and err is not None and "=" in err)

defs, err = parse_custom_variables("123abc = Distance_pixels")
check("a name starting with a digit is a parse error", defs == [] and err is not None)

defs, err = parse_custom_variables("distance_cm = ")
check("an empty expression is a parse error", defs == [] and err is not None)

defs, err = parse_custom_variables("distance_cm = Distance_pixels\ndistance_cm = Mouse_X")
check("a name defined twice is a parse error", defs == [] and err is not None)

# ------------------------------------------------------------------
# 2) safe_eval_expr() / apply_custom_variables() -- the expression engine.
# ------------------------------------------------------------------
check("basic scalar arithmetic", safe_eval_expr("a + b * 2", {"a": 3, "b": 4}) == 11)
check("whitelisted function call", safe_eval_expr("sqrt(x)", {"x": 16.0}) == 4.0)

series_result = safe_eval_expr("x / 2", {"x": pd.Series([2, 4, 6])})
check("vectorized arithmetic over a pandas Series",
      list(series_result) == [1.0, 2.0, 3.0])

cmp_result = safe_eval_expr("x > 2", {"x": pd.Series([1, 2, 3])})
check("comparison over a pandas Series produces the expected boolean mask",
      list(cmp_result) == [False, False, True])

check("an unknown name is rejected", raises_value_error(lambda: safe_eval_expr("y + 1", {"x": 1})))
check("attribute access is rejected (no '.__class__' escape)",
      raises_value_error(lambda: safe_eval_expr("x.__class__", {"x": 5})))
check("subscripting is rejected", raises_value_error(lambda: safe_eval_expr("x[0]", {"x": [1, 2, 3]})))
check("calling a non-whitelisted function is rejected",
      raises_value_error(lambda: safe_eval_expr("open(x)", {"x": "f"})))
check("a lambda call is rejected", raises_value_error(lambda: safe_eval_expr("(lambda: 1)()", {})))
check("the classic __class__/__subclasses__ sandbox-escape chain is rejected",
      raises_value_error(lambda: safe_eval_expr("().__class__.__bases__[0].__subclasses__()", {})))
check("a plain syntax error is rejected cleanly (no crash)",
      raises_value_error(lambda: safe_eval_expr("distance_cm = =", {})))

df = pd.DataFrame({"Distance_pixels": [10.0, 20.0, 30.0]})
df2, warnings = apply_custom_variables(
    df.copy(),
    [("distance_cm", "Distance_pixels / scale_factor"), ("speed_cm_s", "distance_cm * FPS")],
    scalar_context={"scale_factor": 2.0, "FPS": 30},
)
check("apply_custom_variables computes a simple derived column",
      list(df2["distance_cm"]) == [5.0, 10.0, 15.0])
check("apply_custom_variables lets a later definition use an earlier custom column (chaining)",
      list(df2["speed_cm_s"]) == [150.0, 300.0, 450.0])
check("no warnings when every definition succeeds", warnings == [])

df3, warnings3 = apply_custom_variables(
    df.copy(), [("Distance_pixels", "1")], scalar_context={},
)
check("a name colliding with an existing column is skipped, not overwritten",
      list(df3["Distance_pixels"]) == [10.0, 20.0, 30.0])
check("the collision produces exactly one warning", len(warnings3) == 1)

df4, warnings4 = apply_custom_variables(
    df.copy(),
    [("bad", "Nonexistent_Column * 2"), ("good", "Distance_pixels * 2")],
    scalar_context={},
)
check("a definition referencing an unknown column is skipped (one warning)", len(warnings4) == 1)
check("a good definition alongside a bad one still gets computed",
      "good" in df4.columns and list(df4["good"]) == [20.0, 40.0, 60.0])
check("the bad definition's column is simply absent, not a crash", "bad" not in df4.columns)

# ------------------------------------------------------------------
# 3) process_single_video wiring -- a short synthetic video, run 3 times:
# custom variables unset, set with all-valid definitions, and set with one
# good + one bad definition mixed together.
# ------------------------------------------------------------------
W, H, FPS = 200, 160, 30
rng = np.random.default_rng(2)
xs = np.linspace(30, 170, 50) + rng.normal(0, 2.0, 50)
ys = 80 + 15 * np.sin(np.linspace(0, 6, 50))
POSITIONS = list(zip(xs, ys))
TOTAL_FRAMES = len(POSITIONS)
DURATION_S = TOTAL_FRAMES / FPS

VIDEO_PATH = os.path.join(tempfile.gettempdir(), "custom_variables_test_video.mp4")
writer = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
for (x, y) in POSITIONS:
    frame = np.full((H, W), 200, dtype=np.uint8)
    cv2.circle(frame, (int(x), int(y)), 8, 40, -1)
    writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
writer.release()

matrix, warp_w, warp_h = identity_transform(W, H)


def make_setup(**overrides):
    setup = {
        "start_time": 0.0, "end_time": DURATION_S,
        "matrix": matrix, "warp_w": warp_w, "warp_h": warp_h,
        "arena": (0, 0, W, H),
        "roi_points": {}, "roi_names": [],
        "object_points": {}, "object_names": [],
        "interaction_margin_px": 0, "behavior_names": [],
        "background_samples": 30, "threshold": 30,
        "min_area": 20, "max_area": 2000, "max_jump": 400,
        "use_zone_threshold": False, "reject_shadows": False,
        "use_window": False, "window_size": 120, "window_weight": 0.5,
        "scale_factor": 2.0, "scale_unit": "cm",
        "preview_samples": 3, "color_mode": "gray",
        "output_dir_override": None,
        "compute_arm_entries": False, "compute_alternation": False,
        "mask_points": [],
        "stop_condition": None, "stop_value": None, "stop_zone_name": "",
        "zone_groups": {}, "smooth_window": None,
        "custom_variables": [],
    }
    setup.update(overrides)
    return setup


def run(label, **overrides):
    out_dir = tempfile.mkdtemp(prefix=f"customvars_{label}_")
    setup = make_setup(output_dir_override=out_dir, **overrides)
    summary = process_single_video(
        VIDEO_PATH, setup, show_display=False, confirm_callback=lambda *a, **k: False
    )
    return summary, summary["Output_folder"]


summary_off, out_off = run("off")
check("no custom variables -> summary carries no 'Custom_variables_added' key",
      "Custom_variables_added" not in summary_off)
raw_off = pd.read_csv(os.path.join(out_off, "raw_tracking.csv"))
check("no custom variables -> raw_tracking.csv is unaffected (no distance_cm/speed_cm_s)",
      "distance_cm" not in raw_off.columns and "speed_cm_s" not in raw_off.columns)

summary_on, out_on = run(
    "on",
    custom_variables=[("distance_cm", "Distance_pixels / scale_factor"),
                       ("speed_cm_s", "distance_cm * FPS")],
)
check("custom variables set -> summary lists both as added",
      summary_on.get("Custom_variables_added") == ["distance_cm", "speed_cm_s"])
check("custom variables set -> no warnings for an all-valid setup",
      "Custom_variables_warnings" not in summary_on)
raw_on = pd.read_csv(os.path.join(out_on, "raw_tracking.csv"))
check("raw_tracking.csv carries the new distance_cm column",
      "distance_cm" in raw_on.columns)
check("raw_tracking.csv carries the chained speed_cm_s column",
      "speed_cm_s" in raw_on.columns)
check("distance_cm matches Distance_pixels / scale_factor exactly",
      np.allclose(raw_on["distance_cm"], raw_on["Distance_pixels"] / 2.0))
check("speed_cm_s matches distance_cm * FPS (the chained reference) exactly",
      np.allclose(raw_on["speed_cm_s"], raw_on["distance_cm"] * FPS))

summary_mixed, out_mixed = run(
    "mixed",
    custom_variables=[("distance_cm", "Distance_pixels / scale_factor"),
                       ("broken", "Nonexistent_Column * 2")],
)
check("one bad definition alongside a good one doesn't stop the run",
      summary_mixed.get("Custom_variables_added") == ["distance_cm"])
check("the bad definition produced exactly one warning in the summary",
      len(summary_mixed.get("Custom_variables_warnings", [])) == 1)
raw_mixed = pd.read_csv(os.path.join(out_mixed, "raw_tracking.csv"))
check("the good column made it into raw_tracking.csv despite the bad one",
      "distance_cm" in raw_mixed.columns)
check("the bad column is simply absent from raw_tracking.csv",
      "broken" not in raw_mixed.columns)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
