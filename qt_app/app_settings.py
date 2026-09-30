"""
Persisted app-level Settings (Upgrade Plan Tier 2 #8, "a real Settings
screen") -- Detection Settings defaults and a couple of app-wide
preferences (default color mode, default distance-calibration units)
that used to reset back to their hardcoded fallbacks every time the app
was relaunched, since qt_app/main_window.py's self._memory always started
completely empty. Saved as a small JSON file in the user's home
directory, loaded once at startup straight into that same _memory dict --
every Setup page field that already reads _memory (see setup_page.py's
entry_or_default/var_or_default, _setting_field, _color_mode_group) picks
these up automatically, with no separate plumbing needed for the fields
themselves; see qt_app/dialogs/settings_dialog.py for the screen that
edits them.

Deliberately excludes a theme preference (a dark/light toggle used to
live in qt_app/theme.py -- see that module's own docstring: it was
removed entirely, on purpose, rather than just hidden, so re-adding a
theme setting here would undo that decision).
"""

import json
import os

SETTINGS_DIR = os.path.join(os.path.expanduser("~"), ".behavioraltracker")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

# Every _memory key this Settings screen manages -- Detection Settings'
# own numeric/checkbox fields plus the app-wide preferences the Upgrade
# Plan calls out (default units, default color mode). Kept as one list so
# load/save and the dialog can't drift out of sync with each other.
SETTINGS_KEYS = [
    "bg_samples_entry", "threshold_entry", "min_area_entry", "max_area_entry",
    "max_jump_entry", "use_zone_threshold_var",
    "real_distance_entry", "units_entry", "preview_samples_entry",
    "color_mode_var", "bin_size_entry",
]


def load_settings():
    """Returns the persisted {_memory key: value} dict, or {} if there's
    no settings file yet (first run, or every run before this feature
    existed) or it can't be read/parsed -- corrupt or missing settings
    should never stop the app from starting."""
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if k in SETTINGS_KEYS}


def save_settings(values):
    """Writes {_memory key: value} (only recognized SETTINGS_KEYS -- any
    other key in `values` is silently ignored) to the settings file,
    creating its folder first if needed. Raises OSError on a genuine
    write failure (e.g. a read-only home directory); callers decide how
    to surface that."""
    data = {k: v for k, v in values.items() if k in SETTINGS_KEYS}
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
