#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys


report_path = Path(sys.argv[1])
source_checkpoint = Path(sys.argv[2])
code = int(sys.argv[3])
result = {
    "runner_exit_code": code,
    "report": str(report_path),
    "report_exists": report_path.is_file(),
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_exists": source_checkpoint.is_file(),
    "checkpoint_written": False,
}
if source_checkpoint.is_file():
    result["source_checkpoint_sha256_recomputed"] = hashlib.sha256(
        source_checkpoint.read_bytes()).hexdigest()
if report_path.is_file():
    report = json.loads(report_path.read_text())
    result.update({
        "status": report.get("status"),
        "checks": report.get("checks"),
        "hold_executed_frames": report.get("hold", {}).get("executed_frames"),
        "source_checkpoint_unchanged": report.get("checks", {}).get(
            "source_checkpoint_unchanged"),
        "wall_seconds": report.get("execution", {}).get("wall_seconds"),
        "next_gate": report.get("next_gate"),
    })
else:
    result["status"] = "NO_REPORT"
print("COPY_RESULT_BEGIN")
print(json.dumps(result, indent=2, sort_keys=True))
print("COPY_RESULT_END")
