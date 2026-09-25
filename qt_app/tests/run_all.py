"""
Runs the whole qt_app headless regression suite and prints a combined
pass/fail summary.

Each test_*.py file is a standalone script (real QApplication + MainWindow,
sys.exit(0) on success / sys.exit(1) on failure) rather than a pytest
module, because only one QApplication can live in a process and several of
these tests intentionally drive a full, real MainWindow end to end (real
video decoding, real Qt mouse/keyboard events, real PyTorch training in
test_ml_pipeline.py). So this runner launches each one as its own
subprocess and collects their exit codes -- it does not import them.

Run it exactly like a single test file (needs a virtual display, same as
every file in this suite -- see _pathsetup.py's docstring):

    xvfb-run -a python3.12 qt_app/tests/run_all.py

Pass one or more names (with or without .py) to run a subset, e.g.:

    xvfb-run -a python3.12 qt_app/tests/run_all.py test_maze_template_dialog test_ml_pipeline
"""
import os
import subprocess
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))

# Fastest/most-isolated first, most expensive (real PyTorch training) last,
# so a quick regression in an earlier area fails fast.
ORDERED_TESTS = [
    "test_canvas_smoke.py",
    "test_maze_template_dialog.py",
    "test_start_tracking.py",
    "test_manual_scoring.py",
    "test_batch_features.py",
    "test_ml_pipeline.py",
]


def resolve_selection(argv):
    if not argv:
        return list(ORDERED_TESTS)
    selected = []
    for name in argv:
        fname = name if name.endswith(".py") else name + ".py"
        if fname not in ORDERED_TESTS:
            print(f"Unknown test '{name}' -- choices are: "
                  + ", ".join(t[:-3] for t in ORDERED_TESTS))
            sys.exit(2)
        selected.append(fname)
    return selected


def main():
    selection = resolve_selection(sys.argv[1:])
    results = []  # (name, passed, elapsed_seconds)

    for fname in selection:
        path = os.path.join(TESTS_DIR, fname)
        banner = f" {fname} "
        print("\n" + banner.center(70, "="))
        start = time.time()
        proc = subprocess.run([sys.executable, path], cwd=REPO_ROOT)
        elapsed = time.time() - start
        passed = proc.returncode == 0
        results.append((fname, passed, elapsed))

    print("\n" + " SUMMARY ".center(70, "="))
    all_passed = True
    for fname, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"[{status}] {fname}  ({elapsed:.1f}s)")

    total = sum(e for _n, _p, e in results)
    print(f"\n{len(results)} suite(s) run in {total:.1f}s total.")
    if all_passed:
        print("ALL SUITES PASSED")
        sys.exit(0)
    else:
        failed = [n for n, p, _e in results if not p]
        print(f"{len(failed)} SUITE(S) FAILED: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
