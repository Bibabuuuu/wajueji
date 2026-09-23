#!/usr/bin/env python3
"""Execute the audited 65-rock non-contact bucket approach."""

from __future__ import annotations

import json

import run_layered_47_excavation_approach as shared
import run_layered_65_excavation_preflight as preflight


PROFILE = {
    **preflight.PROFILE,
    "temporary_directory_prefix": "layered-65-excavation-approach-",
    "restored_runtime_label": "approach-source-restored-65",
    "progress_label": "PROGRESS_65_APPROACH",
    "report_schema": "excavation_layered_65_excavation_approach_report_v1",
    "passed_status": "layered_65_excavation_approach_passed",
    "failed_status": "layered_65_excavation_approach_failed",
    "passed_next_gate": "contact_handoff_from_verified_noncontact_prepoint",
    "failed_next_gate": "fix_noncontact_approach",
}


def run(args) -> dict:
    return shared.run(
        args, config_loader=preflight.load_config, profile=PROFILE)


def main() -> int:
    report = run(shared._parser(default_device="cuda:1").parse_args())
    print(json.dumps(shared._summary(report), sort_keys=True, allow_nan=False))
    return 0 if report["status"] == PROFILE["passed_status"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
