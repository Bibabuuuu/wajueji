#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path
import sys


report_path = Path(sys.argv[1])
output_checkpoint = Path(sys.argv[2])
source_checkpoint = Path(sys.argv[3])
code = int(sys.argv[4])
result = {
    "runner_exit_code": code,
    "report": str(report_path),
    "report_exists": report_path.is_file(),
    "source_checkpoint": str(source_checkpoint),
    "source_checkpoint_exists": source_checkpoint.is_file(),
    "output_checkpoint": str(output_checkpoint),
    "checkpoint_exists": output_checkpoint.is_file(),
}
if source_checkpoint.is_file():
    result["source_checkpoint_sha256_recomputed"] = hashlib.sha256(
        source_checkpoint.read_bytes()).hexdigest()
if report_path.is_file():
    report = json.loads(report_path.read_text())
    checkpoint = report.get("checkpoint") or {}
    result.update({
        "status": report.get("status"),
        "checks": report.get("checks"),
        "particle_count": checkpoint.get("particle_count"),
        "rock_count": checkpoint.get("rock_count"),
        "simulation_time_s": checkpoint.get("simulation_time_s"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "wall_seconds": report.get("execution", {}).get(
            "wall_seconds_with_checkpoint"),
        "next_gate": report.get("next_gate"),
    })
else:
    result["status"] = "NO_REPORT"
if output_checkpoint.is_file():
    result["checkpoint_size_bytes"] = output_checkpoint.stat().st_size
    result["checkpoint_file_sha256_recomputed"] = hashlib.sha256(
        output_checkpoint.read_bytes()).hexdigest()
print("COPY_RESULT_BEGIN")
print(json.dumps(result, indent=2, sort_keys=True))
print("COPY_RESULT_END")
