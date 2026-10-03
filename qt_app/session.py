"""Session/project save-load: serialize a full experiment configuration
to one portable .abtsession JSON file so an experiment can be closed and
reopened later without re-entering videos, zones, scale, subjects, and
detection settings (project-file parity with EthoVision/ANY-maze).

The state dict is collected by MainWindow (video queue + per-video
alignment/subject, the Setup page's _memory detection-settings dict, ROI /
object geometry, scale, time window, time-bin size, stop conditions, and
custom-variable definitions) and passed here verbatim; this module only
owns the file format. Paths are stored as given (relative paths stay
relative so a project folder can be moved as a whole)."""

import json
import os

SESSION_VERSION = 1
SESSION_EXT = ".abtsession"


def default_session_path(folder):
    return os.path.join(folder, f"session{SESSION_EXT}")


def save_session(path, state):
    """Write state dict to an .abtsession JSON file. `path` may omit the
    extension. Returns the path actually written."""
    if not str(path).endswith(SESSION_EXT):
        path = str(path) + SESSION_EXT
    payload = {"app": "AnimalBehaviourTracker", "version": SESSION_VERSION, "state": state}
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)  # atomic: a crash mid-save can't truncate a good session
    return path


def load_session(path):
    """Read an .abtsession file back into the state dict. Raises a clear
    error for wrong app, newer versions, or corrupt JSON."""
    if not str(path).endswith(SESSION_EXT) and not os.path.exists(str(path)):
        path = str(path) + SESSION_EXT
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("app") != "AnimalBehaviourTracker":
        raise ValueError("Not an AnimalBehaviourTracker session file.")
    if payload.get("version", 0) > SESSION_VERSION:
        raise ValueError(
            f"Session version {payload['version']} is newer than this app supports "
            f"({SESSION_VERSION}). Update the app to open it.")
    state = payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("Session file is missing its state block.")
    return state
