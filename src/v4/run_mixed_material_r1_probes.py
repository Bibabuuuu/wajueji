"""Compile and run the short R1 MuJoCo fixtures, then record the results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value) -> None:
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run(repository_root: str | Path, report_dir: str | Path,
        source: str | Path, include_dir: str | Path,
        library_dir: str | Path, cxx: str = "g++") -> dict:
    root = Path(repository_root).resolve()
    reports = Path(report_dir).resolve()
    source_path = Path(source).resolve()
    report_path = reports / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixtures = report["probe_fixtures"]
    results = []
    with tempfile.TemporaryDirectory(prefix="mixed-material-r1-probe-") as temporary:
        binary = Path(temporary) / "mixed_material_r1_probe"
        command = [
            cxx, "-O2", "-std=c++17", "-Wall", "-Wextra",
            "-I" + str(Path(include_dir).resolve()), str(source_path),
            "-L" + str(Path(library_dir).resolve()),
            "-Wl,-rpath," + str(Path(library_dir).resolve()),
            "-l:libmujoco.so.2.3.8", "-o", str(binary),
        ]
        build = subprocess.run(command, text=True, capture_output=True, check=False)
        if build.returncode != 0:
            raise RuntimeError("probe_build_failed:" + build.stderr)
        for fixture in fixtures:
            fixture_path = root / fixture["path"]
            if sha256(fixture_path) != fixture["sha256"]:
                raise ValueError("fixture_sha256_mismatch:" + str(fixture_path))
            completed = subprocess.run(
                [str(binary), str(fixture_path)], text=True,
                capture_output=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(
                    f"probe_failed:{fixture_path}:{completed.stderr}:{completed.stdout}")
            result = json.loads(completed.stdout)
            if not result.get("passed"):
                raise RuntimeError("probe_reported_failure:" + str(fixture_path))
            result["fixture_path"] = fixture["path"]
            result["fixture_sha256"] = fixture["sha256"]
            results.append(result)
        compiler = subprocess.run(
            [cxx, "--version"], text=True, capture_output=True, check=True)

    record = {
        "schema": "excavation_open_bucket_probe_run_v1",
        "status": "passed",
        "source_path": "src/v4/mixed_material_r1_probe.cc",
        "source_sha256": sha256(source_path),
        "compiler_first_line": compiler.stdout.splitlines()[0],
        "mujoco_contract": "2.3.8, zero plugins",
        "probe_count": len(results),
        "all_passed": all(item["passed"] for item in results),
        "results": results,
        "scope_limit": (
            "proves short standalone entry through the open proxy and bucket "
            "contact only; does not prove mechanism-attached hold or dump"
        ),
    }
    result_path = reports / "probe-results.json"
    write_json(result_path, record)
    report["status"] = (
        "r1_full_rock_single_chain_passed_r2_pending"
        if report["admission"].get("rigid_bridge_mechanism_dump_passed") else
        "mechanism_hold_passed_dump_pending"
        if report["admission"].get("rigid_bridge_mechanism_hold_passed") else
        "mechanism_entry_passed_hold_dump_pending"
        if report["admission"].get("rigid_bridge_mechanism_entry_passed") else
        "standalone_probe_rigid_units_through_bucket_push_passed_"
        "entry_hold_dump_pending"
        if report["admission"].get("rigid_bridge_bucket_push_comparison_passed") else
        "bucket_push_pair_passed_comparison_pending"
        if (report["admission"].get("rigid_bridge_bucket_push_passed") and
            report["admission"].get("rigid_bridge_bucket_push_clear_passed")) else
        "standalone_probe_mechanism_freefall_slope_roll_rock_collision_"
        "passed_bucket_tests_pending"
        if report["admission"].get("rigid_bridge_rock_collision_passed") else
        "standalone_probe_mechanism_freefall_slope_roll_passed_"
        "remaining_bucket_tests_pending"
        if report["admission"].get("rigid_bridge_slope_roll_passed") else
        "standalone_probe_and_mechanism_freefall_passed_bucket_tests_pending"
        if report["admission"].get("rigid_bridge_freefall_passed") else
        "standalone_probe_passed_mechanism_attachment_pending"
    )
    report["probe_results_sha256"] = sha256(result_path)
    report["admission"]["standalone_mujoco_probe_passed"] = True
    write_json(report_path, report)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--include-dir", required=True)
    parser.add_argument("--library-dir", required=True)
    parser.add_argument("--cxx", default="g++")
    args = parser.parse_args()
    result = run(args.repository_root, args.report_dir, args.source,
                 args.include_dir, args.library_dir, args.cxx)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
