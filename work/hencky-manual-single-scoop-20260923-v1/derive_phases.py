"""Derive immutable post-hoc action phases for the D-129 manual episode."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path("/home/qunwang/wajueji")
RUN = ROOT / "runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-204445-912095"
OUT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = [json.loads(line) for line in
        (RUN / "physics.ndjson").read_text(encoding="utf-8").splitlines()]
report = json.loads((RUN / "report.json").read_text(encoding="utf-8"))
commands = np.asarray([row["applied_command_si"] for row in rows])


def first_window(start: int, width: int, predicate) -> int:
    for index in range(start, len(rows) - width + 1):
        if predicate(commands[index:index + width]):
            return index + 1  # one-based frame
    raise RuntimeError("derived_phase_boundary_not_found")


first_motion = first_window(
    0, 3, lambda value: np.all(np.linalg.norm(value, axis=1) > 1.0e-4))
curl_start = first_window(
    first_motion, 3, lambda value: np.all(value[:, 3] > 0.01))
lift_start = first_window(
    curl_start, 30,
    lambda value: (float(value[:, 2].mean()) > 0.015
                   and float(np.abs(value[:, 3]).mean()) < 0.01))
transport_start = first_window(
    lift_start, 30,
    lambda value: (float(value[:, 0].mean()) < -0.005
                   and float(value[:, 1].mean()) > 0.015))
terminal = [index + 1 for index, row in enumerate(rows)
            if row["phase"] == "terminal_zero_hold"]
if terminal != list(range(len(rows) - 29, len(rows) + 1)):
    raise RuntimeError("terminal_hold_identity")
terminal_start = terminal[0]

boundaries = [
    ("precontact_positioning_hold", 1, first_motion - 1,
     "before first sustained nonzero command"),
    ("penetrate_deepen", first_motion, curl_start - 1,
     "sustained excavation motion before positive bucket curl"),
    ("curl_load", curl_start, lift_start - 1,
     "first sustained positive bucket command through load consolidation"),
    ("lift_clear", lift_start, transport_start - 1,
     "first sustained positive arm command with bucket command near neutral"),
    ("off_soil_transport", transport_start, terminal_start - 1,
     "first sustained negative swing plus positive boom transport command"),
    ("terminal_zero_hold", terminal_start, len(rows),
     "runner-generated final 30-frame zero-command hold"),
]
if any(start > end for _, start, end, _ in boundaries):
    raise RuntimeError("derived_phase_empty")
if [frame for _, start, end, _ in boundaries
        for frame in range(start, end + 1)] != list(range(1, len(rows) + 1)):
    raise RuntimeError("derived_phase_partition")

result = {
    "schema": "layered_47_manual_contact_posthoc_phases_v1",
    "status": "posthoc_action_phases_derived",
    "source_run": str(RUN),
    "source_report_sha256": sha256(RUN / "report.json"),
    "source_physics_sha256": sha256(RUN / "physics.ndjson"),
    "frames": len(rows),
    "original_operator_markers_preserved": True,
    "original_operator_markers": report["phase_events"],
    "operator_reported_marker_timing_inaccurate": True,
    "method": {
        "basis": "applied commands plus runner-owned terminal hold identity",
        "command_window_frames": {"general": 3, "lift": 30, "transport": 30},
        "physical_contact_or_fill_claim_from_labels": False,
        "manual_review_required_before_training": True,
    },
    "intervals": [
        {"label": label, "start_frame": start, "end_frame": end,
         "frame_count": end - start + 1, "basis": basis}
        for label, start, end, basis in boundaries
    ],
    "checks": {
        "all_frames_partitioned_once": True,
        "source_report_still_qualified": (
            report["status"] == "manual_contact_pilot_recorded"
            and all(report["checks"].values())),
        "source_files_not_modified": True,
        "terminal_hold_exactly_30_frames": len(terminal) == 30,
        "training_eligible": False,
    },
}
(OUT / "derived-phases-v1.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, ensure_ascii=False))
