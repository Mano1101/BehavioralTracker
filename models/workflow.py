"""Dependency-aware workflow state for BehavioralTracker.

Editing a step invalidates only that step and everything downstream. Earlier
completed steps remain valid, so the user can correct an error at step 3 while
working at step 9 and continue from the corrected point.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class Step:
    key: str
    title: str
    status: str = "pending"  # pending, valid, stale, running, error
    data: Dict[str, Any] = field(default_factory=dict)
    message: str = ""

class Workflow:
    def __init__(self):
        self.steps: List[Step] = [
            Step("video", "1. Video / Assay"),
            Step("arena", "2. Arena Detection / Perspective"),
            Step("roi", "3. ROI / Objects"),
            Step("calibration", "4. Scale Calibration"),
            Step("tracking", "5. Tracking Configuration"),
            Step("preview", "6. Tracking Preview / QC"),
            Step("run", "7. Full Tracking"),
            Step("analysis", "8. Assay Analysis"),
            Step("events", "9. Behavior / BORIS Coding"),
            Step("results", "10. Review / Export"),
        ]
        self.index = {s.key: i for i, s in enumerate(self.steps)}

    def set_valid(self, key, data=None, message=""):
        s = self.steps[self.index[key]]
        s.status = "valid"
        if data is not None: s.data = data
        s.message = message

    def set_error(self, key, message):
        s = self.steps[self.index[key]]
        s.status = "error"; s.message = message

    def invalidate_from(self, key):
        start = self.index[key]
        for s in self.steps[start:]:
            if s.status != "pending": s.status = "stale"
            s.message = "Needs rerun because an upstream step changed."
        self.steps[start].status = "pending"

    def invalidate_downstream(self, key):
        start = self.index[key] + 1
        for s in self.steps[start:]:
            if s.status == "valid": s.status = "stale"
            s.message = "Needs rerun because an upstream step changed."

    def can_run(self, key):
        i = self.index[key]
        return all(s.status == "valid" for s in self.steps[:i])

    def next_step(self):
        for s in self.steps:
            if s.status != "valid": return s
        return self.steps[-1]

    def snapshot(self):
        return [{"key":s.key,"title":s.title,"status":s.status,"data":s.data,"message":s.message} for s in self.steps]
