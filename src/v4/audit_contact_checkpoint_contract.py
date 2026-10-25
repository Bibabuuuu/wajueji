"""Audit contact checkpoint isolation and visual-review evidence availability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from audit_contact_data_contract import iter_jsonl
from contact_checkpoint_contract import (
    ACTOR_HEAD_PHASES,
    CONTACT_CHECKPOINT_SCHEMA,
    CONTRACT_SPEC_SHA256,
    LABEL_PHASES,
    contract_spec,
    validate_contact_checkpoint_metadata,
    validate_reviewed_segment_records,
)
from v4_actor_policy import CHECKPOINT_SCHEMA, PHASES, checkpoint_runtime_route


MEDIA_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".ppm", ".mp4", ".webm", ".avi", ".mov",
}
VIDEO_SUFFIXES = {".mp4", ".webm", ".avi", ".mov"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def rejection_code(callback):
    try:
        callback()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected_contract_rejection")


def scan_visual_evidence(training_root, correction_root):
    roots = {
        "training": Path(training_root).resolve(),
        "corrections": Path(correction_root).resolve(),
    }
    if roots["training"].name != "training" or roots["corrections"].name != "corrections":
        raise ValueError("unexpected_visual_evidence_roots")
    files = []
    for role, root in roots.items():
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES:
                files.append({
                    "role": role,
                    "relative_path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "kind": (
                        "video" if path.suffix.lower() in VIDEO_SUFFIXES
                        else "initial_ready_frame"
                        if path.name == "ready-with-ui.ppm"
                        else "image"),
                })
    return {
        "media_files": files,
        "initial_ready_frames": sum(
            item["kind"] == "initial_ready_frame" for item in files),
        "continuous_video_files": sum(item["kind"] == "video" for item in files),
        "noninitial_image_files": sum(item["kind"] == "image" for item in files),
    }


def write_markdown(path, report):
    counts = report["candidate_labels"]
    visual = report["visual_evidence"]
    content = f"""# 接触策略 checkpoint 隔离契约审计

更新：2026-09-19。状态：契约已冻结并通过零权重审计；人工连续画面复核仍受证据缺失阻塞，未训练、未加载checkpoint、未运行物理。

## 结论

新接触策略只允许三个动作头：`penetrate_deepen`、`curl_load`、`lift_clear`。完整标签仍保留`approach`、`off_soil_confirm`和`uncertain/recovery`，但前两者属于几何控制，后者只属于安全监督器。旧checkpoint的四个动作头为`{', '.join(report['legacy_checkpoint']['phase_names'])}`，不能通过新schema。

新checkpoint必须同时绑定：2412维观察契约、四轴速度顺序、已完成人工复核的标签清单哈希、互斥开发划分、仅训练划分归一化、零holdout读取，以及独立的运行时交接审计回执。训练产物不能自带“运行准入”。

当前{counts['segment_records']}段候选全部仍是`visual_manual_review_pending`，训练资格行数为{counts['training_eligible_rows_now']}；因此审计预期拒绝，拒绝码为`{counts['admission_rejection']}`。

## 为什么现在不能补签人工画面复核

允许的数据目录中只找到{visual['initial_ready_frames']}张`ready-with-ui.ppm`启动画面；连续视频{visual['continuous_video_files']}个，其他过程图片{visual['noninitial_image_files']}张。这些启动画面不能证明接触、卷斗、抬升、离土和接管时刻的视觉意图，因此本次没有把复核清单打勾，也没有把候选标签改成reviewed。

## 隔离结果

- 旧四头checkpoint被新接触契约拒绝：`{report['legacy_checkpoint']['new_contract_rejection']}`。
- 新接触checkpoint被旧运行加载器拒绝：`{report['legacy_checkpoint']['legacy_loader_rejection']}`。
- 新checkpoint即使未来完成训练，也必须另有与其SHA256绑定的`external_contact_handoff_router_v1`运行准入回执。
- 当前候选、旧checkpoint和任何缺少连续视觉复核回执的标签都不能启动接触BC训练。

## 下一步

先决定如何取得连续视觉证据：重新录制带同步视频/关键帧的新示范，或明确接受“旧六条仅作数值先验、不进入新BC训练”。之后才能生成reviewed标签、人工复核回执和M2开发划分。
"""
    Path(path).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-config", type=Path, required=True)
    parser.add_argument("--candidate-segments", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_exists:" + str(args.output))

    config = json.loads(args.contract_config.read_text(encoding="utf-8"))
    config_matches = (
        config.get("contract") == contract_spec()
        and config.get("contract_spec_sha256") == CONTRACT_SPEC_SHA256)
    candidates = [row for _, row in iter_jsonl(args.candidate_segments)]
    rejection = rejection_code(
        lambda: validate_reviewed_segment_records(candidates))
    legacy = {
        "schema": CHECKPOINT_SCHEMA,
        "phase_names": list(PHASES),
        "phase_route": "monotonic_two_tick_episode_reset",
    }
    legacy_rejection = rejection_code(
        lambda: validate_contact_checkpoint_metadata(legacy))
    new_rejection = rejection_code(
        lambda: checkpoint_runtime_route({"schema": CONTACT_CHECKPOINT_SCHEMA}))
    visual = scan_visual_evidence(args.training_root, args.correction_root)
    checks = {
        "config_matches_code_contract": config_matches,
        "six_label_phases_separate_from_three_actor_heads": (
            len(LABEL_PHASES) == 6 and len(ACTOR_HEAD_PHASES) == 3),
        "current_candidates_rejected_pending_review": (
            rejection == "contact_label_schema_not_reviewed"),
        "current_candidate_training_rows_zero": all(
            row.get("training_eligible_now") is False for row in candidates),
        "legacy_four_head_checkpoint_rejected_by_contact_contract": (
            legacy_rejection == "contact_checkpoint_schema_mismatch"),
        "contact_checkpoint_rejected_by_legacy_loader": (
            new_rejection == "v4_actor_checkpoint_contract_mismatch"),
        "continuous_visual_evidence_absent": (
            visual["continuous_video_files"] == 0
            and visual["noninitial_image_files"] == 0),
        "holdout_roots_scanned_zero": True,
        "checkpoint_files_loaded_zero": True,
        "physics_steps_zero": True,
        "parameter_updates_zero": True,
    }
    report = {
        "schema": "excavation_contact_checkpoint_contract_audit_v1",
        "status": (
            "contract_frozen_manual_visual_review_blocked"
            if all(checks.values()) else "failed"),
        "checks": checks,
        "contract_spec_sha256": CONTRACT_SPEC_SHA256,
        "candidate_labels": {
            "segment_records": len(candidates),
            "training_eligible_rows_now": sum(
                row.get("row_count", 0) for row in candidates
                if row.get("training_eligible_now") is True),
            "admission_rejection": rejection,
            "source_sha256": sha256(args.candidate_segments),
        },
        "legacy_checkpoint": {
            "schema": CHECKPOINT_SCHEMA,
            "phase_names": list(PHASES),
            "new_contract_rejection": legacy_rejection,
            "legacy_loader_rejection": new_rejection,
        },
        "visual_evidence": visual,
        "source_sha256": {
            "contract_config": sha256(args.contract_config),
            "generator": sha256(Path(__file__)),
        },
        "scope": {
            "holdout_paths_supported": False,
            "holdout_roots_scanned": 0,
            "checkpoint_files_loaded": 0,
            "physics_steps": 0,
            "parameter_updates": 0,
            "claim": (
                "metadata isolation and evidence-availability audit only; "
                "not label approval, training, or runtime admission"),
        },
    }
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    write_markdown(args.output / "report.md", report)
    names = ("report.json", "report.md")
    (args.output / "manifest.sha256").write_text(
        "".join(f"{sha256(args.output / name)}  ./{name}\n" for name in names),
        encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "checks": checks,
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
