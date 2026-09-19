"""Fixed-log diagnosis of entry-height drift and post-curl action reversal.

Only the six explicit training episodes, one already-recorded automatic run,
and small prior audit reports are accepted.  The tool does not support a
holdout path, load a checkpoint, run physics, or update parameters.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics

from audit_contact_data_contract import (
    TRAINING_EPISODE_ALLOWLIST,
    iter_jsonl,
    load_actor_index,
    require_safe_actor_index,
    validate_observation_by_schema,
)
from mpm_smoke import MOUTH_LOCAL_M, TIP_LOCAL_M


REFERENCES = {
    "bucket_origin": (0.0, 0.0, 0.0),
    "cutting_tip": tuple(float(value) for value in TIP_LOCAL_M),
    "bucket_mouth": tuple(float(value) for value in MOUTH_LOCAL_M),
}
ORIGIN_THRESHOLDS_M = (-1.25, -1.0, -0.75, -0.50, -0.45, -0.40, -0.30)
TOOL_THRESHOLDS_M = (-1.0, -0.75, -0.50, -0.45, -0.40, -0.30)
ENTRY_X_MIN_M = -0.30
ENTRY_Z_RANGE_M = (0.85, 1.05)
REVERSE_THRESHOLD = -0.10
TOLERANCE = 1e-7


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quaternion_rotate_wxyz(quaternion, vector):
    if len(quaternion) != 4 or len(vector) != 3:
        raise ValueError("invalid_quaternion_or_vector_shape")
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid_quaternion_norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    vx, vy, vz = (float(value) for value in vector)
    return (
        (1.0 - 2.0 * (y * y + z * z)) * vx
        + 2.0 * (x * y - z * w) * vy
        + 2.0 * (x * z + y * w) * vz,
        2.0 * (x * y + z * w) * vx
        + (1.0 - 2.0 * (x * x + z * z)) * vy
        + 2.0 * (y * z - x * w) * vz,
        2.0 * (x * z - y * w) * vx
        + 2.0 * (y * z + x * w) * vy
        + (1.0 - 2.0 * (x * x + y * y)) * vz,
    )


def add(left, right):
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def subtract(left, right):
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def maximum_abs(values):
    return max((abs(float(value)) for value in values), default=0.0)


def derive_observation(observation):
    validate_observation_by_schema(observation)
    bucket = observation["bucket"]
    target = tuple(float(value) for value in observation["target_world_m"])
    origin = tuple(float(value) for value in bucket["position_world_m"])
    quaternion = tuple(float(value) for value in
                       bucket["orientation_quaternion_wxyz"])
    references = {}
    for name, local in REFERENCES.items():
        world = add(origin, quaternion_rotate_wxyz(quaternion, local))
        references[name] = {
            "world_m": world,
            "relative_to_target_m": subtract(world, target),
        }
    tip = references["cutting_tip"]["world_m"]
    mouth = references["bucket_mouth"]["world_m"]
    mouth_line = subtract(mouth, tip)
    derived_mouth_angle = math.degrees(math.atan2(
        mouth_line[2], math.hypot(mouth_line[0], mouth_line[1])))
    logged_relative = tuple(float(value) for value in
                            bucket["relative_to_target_m"])
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    return {
        "references": references,
        "target_world_m": target,
        "quaternion_norm_error": abs(quaternion_norm - 1.0),
        "origin_relative_error_m": maximum_abs(
            a - b for a, b in zip(
                references["bucket_origin"]["relative_to_target_m"],
                logged_relative)),
        "mouth_angle_error_deg": abs(
            derived_mouth_angle - float(bucket["mouth_angle_deg"])),
    }


def load_trace(path):
    rows = []
    source_counts = Counter()
    schemas = Counter()
    maximum_errors = {
        "quaternion_norm": 0.0,
        "origin_relative_m": 0.0,
        "mouth_angle_deg": 0.0,
    }
    last_next_observation = None
    for _, policy in iter_jsonl(path):
        observation = policy["observation"]
        derived = derive_observation(observation)
        values = tuple(float(value) for value in
                       policy["action"]["normalized_velocity"])
        row = {
            "decision_step": int(policy["decision_step"]),
            "sim_time_s": float(observation["sim_time_s"]),
            "action": values,
            "references": derived["references"],
            "target_world_m": derived["target_world_m"],
            "cylinder_position_normalized": tuple(
                float(value) for value in observation["proprioception"]
                ["cylinder_position_normalized"]),
            "axis_velocity_normalized": tuple(
                float(value) for value in observation["proprioception"]
                ["axis_velocity_normalized"]),
        }
        rows.append(row)
        source_counts[policy.get("action_source", "<missing>")] += 1
        schemas[observation["schema"]] += 1
        last_next_observation = policy["next_observation"]
        maximum_errors["quaternion_norm"] = max(
            maximum_errors["quaternion_norm"],
            derived["quaternion_norm_error"])
        maximum_errors["origin_relative_m"] = max(
            maximum_errors["origin_relative_m"],
            derived["origin_relative_error_m"])
        maximum_errors["mouth_angle_deg"] = max(
            maximum_errors["mouth_angle_deg"],
            derived["mouth_angle_error_deg"])
    terminal_state = None
    if last_next_observation is not None:
        terminal_derived = derive_observation(last_next_observation)
        terminal_state = {
            "decision_step": int(last_next_observation["decision_step"]),
            "sim_time_s": float(last_next_observation["sim_time_s"]),
            "action": None,
            "references": terminal_derived["references"],
            "target_world_m": terminal_derived["target_world_m"],
            "cylinder_position_normalized": tuple(
                float(value) for value in last_next_observation["proprioception"]
                ["cylinder_position_normalized"]),
            "axis_velocity_normalized": tuple(
                float(value) for value in last_next_observation["proprioception"]
                ["axis_velocity_normalized"]),
        }
        maximum_errors["quaternion_norm"] = max(
            maximum_errors["quaternion_norm"],
            terminal_derived["quaternion_norm_error"])
        maximum_errors["origin_relative_m"] = max(
            maximum_errors["origin_relative_m"],
            terminal_derived["origin_relative_error_m"])
        maximum_errors["mouth_angle_deg"] = max(
            maximum_errors["mouth_angle_deg"],
            terminal_derived["mouth_angle_error_deg"])
    targets = {row["target_world_m"] for row in rows}
    if terminal_state is not None:
        targets.add(terminal_state["target_world_m"])
    return {
        "rows": rows,
        "terminal_state": terminal_state,
        "row_count": len(rows),
        "target_constant": len(targets) == 1,
        "target_world_m": list(next(iter(targets))) if len(targets) == 1 else None,
        "observation_schema_counts": dict(sorted(schemas.items())),
        "action_source_counts": dict(sorted(source_counts.items())),
        "maximum_reconstruction_error": maximum_errors,
        "source_sha256": sha256(path),
    }


def interpolate(left, right, fraction):
    return float(left) + fraction * (float(right) - float(left))


def first_forward_crossing(rows, reference, threshold):
    previous = None
    for current in rows:
        relative = current["references"][reference]["relative_to_target_m"]
        x_value = relative[0]
        if x_value >= threshold and (previous is None or previous["references"]
                                     [reference]["relative_to_target_m"][0]
                                     < threshold):
            if previous is None:
                fraction = 0.0
                lower = current
            else:
                lower_x = previous["references"][reference][
                    "relative_to_target_m"][0]
                denominator = x_value - lower_x
                fraction = 0.0 if abs(denominator) <= 1e-12 else (
                    threshold - lower_x) / denominator
                fraction = min(1.0, max(0.0, fraction))
                lower = previous
            lower_relative = lower["references"][reference]["relative_to_target_m"]
            action = [interpolate(a, b, fraction)
                      for a, b in zip(lower["action"], current["action"])]
            cylinders = [interpolate(a, b, fraction) for a, b in zip(
                lower["cylinder_position_normalized"],
                current["cylinder_position_normalized"])]
            return {
                "relative_x_m": threshold,
                "relative_y_m": interpolate(lower_relative[1], relative[1], fraction),
                "relative_z_m": interpolate(lower_relative[2], relative[2], fraction),
                "decision_step_interpolated": interpolate(
                    lower["decision_step"], current["decision_step"], fraction),
                "sim_time_s": interpolate(
                    lower["sim_time_s"], current["sim_time_s"], fraction),
                "bracket_steps": [lower["decision_step"], current["decision_step"]],
                "action_normalized": action,
                "cylinder_position_normalized": cylinders,
            }
        previous = current
    return None


def percentile(values, fraction):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return interpolate(ordered[lower], ordered[upper], position - lower)


def summarize_crossings(training, automatic):
    output = {}
    for reference in REFERENCES:
        thresholds = (ORIGIN_THRESHOLDS_M if reference == "bucket_origin"
                      else TOOL_THRESHOLDS_M)
        summaries = []
        for threshold in thresholds:
            manual = {
                episode_id: first_forward_crossing(
                    trace["rows"], reference, threshold)
                for episode_id, trace in training.items()
            }
            automatic_crossing = first_forward_crossing(
                automatic["rows"], reference, threshold)
            manual_z = [item["relative_z_m"] for item in manual.values()
                        if item is not None]
            median = statistics.median(manual_z) if manual_z else None
            summaries.append({
                "relative_x_threshold_m": threshold,
                "automatic": automatic_crossing,
                "manual_by_episode": manual,
                "manual_available_count": len(manual_z),
                "manual_relative_z_m": {
                    "minimum": min(manual_z) if manual_z else None,
                    "median": median,
                    "maximum": max(manual_z) if manual_z else None,
                },
                "automatic_minus_manual_median_z_m": (
                    automatic_crossing["relative_z_m"] - median
                    if automatic_crossing is not None and median is not None
                    else None),
            })
        output[reference] = summaries
    return output


def entry_gate_summary(rows, terminal_state=None):
    first_z = None
    first_x = None
    first_ready = None
    first_below_after_z = None
    scan_rows = list(rows)
    if (terminal_state is not None
            and (not scan_rows
                 or terminal_state["decision_step"] != scan_rows[-1]["decision_step"])):
        scan_rows.append(terminal_state)
    for row in scan_rows:
        x_value, _, z_value = row["references"]["bucket_origin"][
            "relative_to_target_m"]
        compact = {
            "decision_step": row["decision_step"],
            "sim_time_s": row["sim_time_s"],
            "relative_x_m": x_value,
            "relative_z_m": z_value,
            "action_normalized": (
                list(row["action"]) if row["action"] is not None else None),
        }
        in_z = ENTRY_Z_RANGE_M[0] <= z_value <= ENTRY_Z_RANGE_M[1]
        if first_z is None and in_z:
            first_z = compact
        if first_x is None and x_value >= ENTRY_X_MIN_M:
            first_x = compact
        if first_ready is None and x_value >= ENTRY_X_MIN_M and in_z:
            first_ready = compact
        if (first_z is not None and first_below_after_z is None
                and z_value < ENTRY_Z_RANGE_M[0]):
            first_below_after_z = compact
    terminal = scan_rows[-1]
    terminal_relative = terminal["references"]["bucket_origin"][
        "relative_to_target_m"]
    return {
        "first_vertical_window": first_z,
        "first_horizontal_ready": first_x,
        "first_joint_ready": first_ready,
        "first_below_after_entering_vertical_window": first_below_after_z,
        "vertical_window_exited_before_horizontal_ready": (
            first_below_after_z is not None
            and (first_x is None or first_below_after_z["decision_step"]
                 < first_x["decision_step"])),
        "terminal": {
            "decision_step": terminal["decision_step"],
            "relative_x_m": terminal_relative[0],
            "relative_z_m": terminal_relative[2],
            "horizontal_shortfall_to_entry_m": max(
                0.0, ENTRY_X_MIN_M - terminal_relative[0]),
            "vertical_below_entry_min_m": max(
                0.0, ENTRY_Z_RANGE_M[0] - terminal_relative[2]),
        },
    }


def action_summary_until_crossing(rows, reference="bucket_origin", threshold=-0.40):
    crossing = first_forward_crossing(rows, reference, threshold)
    if crossing is None:
        return None
    last_step = crossing["bracket_steps"][1]
    selected = [row for row in rows if row["decision_step"] <= last_step]
    return {
        "rows": len(selected),
        "crossing": crossing,
        "mean_action_normalized": [
            statistics.fmean(row["action"][axis] for row in selected)
            for axis in range(4)],
        "strong_negative_boom_rows": sum(
            row["action"][1] <= -0.8 for row in selected),
        "strong_negative_boom_fraction": sum(
            row["action"][1] <= -0.8 for row in selected) / len(selected),
        "zero_boom_rows": sum(abs(row["action"][1]) <= 1e-12 for row in selected),
        "positive_boom_rows": sum(row["action"][1] > 0.1 for row in selected),
    }


def load_index_records(path):
    index = load_actor_index(path)
    require_safe_actor_index(index)
    records = defaultdict(dict)
    phase_axis_sign_counts = defaultdict(
        lambda: {axis: Counter() for axis in ("swing", "boom", "arm", "bucket")})
    supervised_counts = defaultdict(Counter)
    for line_number, row in iter_jsonl(path):
        episode_id = row["episode_id"]
        if (episode_id not in TRAINING_EPISODE_ALLOWLIST
                or row.get("holdout") is not False
                or row.get("training_eligible") is not True):
            raise ValueError(f"unsafe_index_row:{line_number}")
        step = int(row["decision_step"])
        if step in records[episode_id]:
            raise ValueError("duplicate_index_step")
        records[episode_id][step] = row
        phase = row["subphase"]
        values = row["action"]["normalized_velocity"]
        mask = row["axis_mask"]
        for axis, value, supervised in zip(
                ("swing", "boom", "arm", "bucket"), values, mask):
            if supervised != 1.0:
                continue
            supervised_counts[phase][axis] += 1
            sign = ("positive" if value > 0.1 else
                    "negative" if value < -0.1 else "deadband")
            phase_axis_sign_counts[phase][axis][sign] += 1
    return index, records, {
        phase: {
            axis: {
                "supervised_rows": supervised_counts[phase][axis],
                "sign_counts_deadband_0p1": dict(
                    phase_axis_sign_counts[phase][axis]),
            }
            for axis in ("swing", "boom", "arm", "bucket")
        }
        for phase in sorted(phase_axis_sign_counts)
    }


def prior_hash_matches(report, episode_id, actual_hash):
    candidates = [digest for path, digest in report.get("source_sha256", {}).items()
                  if episode_id in path and path.endswith("policy-steps.ndjson")]
    return candidates == [actual_hash]


def write_crossings_csv(path, crossings):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([
            "reference", "relative_x_threshold_m", "automatic_relative_z_m",
            "manual_min_relative_z_m", "manual_median_relative_z_m",
            "manual_max_relative_z_m", "automatic_minus_manual_median_z_m",
            "manual_available_count",
        ])
        for reference, rows in crossings.items():
            for row in rows:
                automatic = row["automatic"]
                manual = row["manual_relative_z_m"]
                writer.writerow([
                    reference,
                    row["relative_x_threshold_m"],
                    automatic["relative_z_m"] if automatic else "",
                    manual["minimum"], manual["median"], manual["maximum"],
                    row["automatic_minus_manual_median_z_m"],
                    row["manual_available_count"],
                ])


def _prefix_to_x(rows, reference, maximum_x=-0.30):
    selected = []
    for row in rows:
        selected.append(row)
        if row["references"][reference]["relative_to_target_m"][0] >= maximum_x:
            break
    return selected


def write_svg(path, training, automatic):
    width, height = 1200, 390
    panel_width, panel_height = 360, 300
    margin_x, margin_top = 34, 42
    x_min, x_max = -1.55, -0.20
    z_min, z_max = -0.35, 1.75

    def sx(value, panel):
        return margin_x + panel * 400 + (value - x_min) / (x_max - x_min) * panel_width

    def sy(value):
        return margin_top + (z_max - value) / (z_max - z_min) * panel_height

    def points(rows, reference, panel):
        return " ".join(
            f"{sx(row['references'][reference]['relative_to_target_m'][0], panel):.2f},"
            f"{sy(row['references'][reference]['relative_to_target_m'][2]):.2f}"
            for row in _prefix_to_x(rows, reference))

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font:12px sans-serif;fill:#222}.axis{stroke:#777;stroke-width:1}'
        '.manual{fill:none;stroke:#a0a0a0;stroke-width:1.1;opacity:.72}'
        '.auto{fill:none;stroke:#d62728;stroke-width:2.6}.gate{fill:#d7f0d2;opacity:.8}'
        '.grid{stroke:#e5e5e5;stroke-width:1}</style>',
    ]
    titles = {
        "bucket_origin": "Bucket origin",
        "cutting_tip": "Cutting tip",
        "bucket_mouth": "Bucket mouth",
    }
    for panel, reference in enumerate(REFERENCES):
        left = margin_x + panel * 400
        if reference == "bucket_origin":
            gate_left = sx(ENTRY_X_MIN_M, panel)
            gate_top = sy(ENTRY_Z_RANGE_M[1])
            gate_bottom = sy(ENTRY_Z_RANGE_M[0])
            lines.append(
                f'<rect class="gate" x="{gate_left:.2f}" y="{gate_top:.2f}" '
                f'width="{sx(x_max, panel)-gate_left:.2f}" '
                f'height="{gate_bottom-gate_top:.2f}"/>')
        for x_tick in (-1.5, -1.0, -0.5):
            x_pos = sx(x_tick, panel)
            lines.append(f'<line class="grid" x1="{x_pos:.2f}" y1="{margin_top}" '
                         f'x2="{x_pos:.2f}" y2="{margin_top+panel_height}"/>')
            lines.append(f'<text x="{x_pos-12:.2f}" y="{margin_top+panel_height+18}">'
                         f'{x_tick:.1f}</text>')
        for z_tick in (0.0, 0.5, 1.0, 1.5):
            y_pos = sy(z_tick)
            lines.append(f'<line class="grid" x1="{left}" y1="{y_pos:.2f}" '
                         f'x2="{left+panel_width}" y2="{y_pos:.2f}"/>')
            lines.append(f'<text x="{left-30}" y="{y_pos+4:.2f}">{z_tick:.1f}</text>')
        lines.append(f'<line class="axis" x1="{left}" y1="{margin_top+panel_height}" '
                     f'x2="{left+panel_width}" y2="{margin_top+panel_height}"/>')
        lines.append(f'<line class="axis" x1="{left}" y1="{margin_top}" '
                     f'x2="{left}" y2="{margin_top+panel_height}"/>')
        for episode_id, trace in training.items():
            lines.append(f'<polyline class="manual" points="{points(trace["rows"], reference, panel)}">'
                         f'<title>{html.escape(episode_id)}</title></polyline>')
        lines.append(f'<polyline class="auto" points="{points(automatic["rows"], reference, panel)}"/>')
        lines.append(f'<text x="{left+125}" y="24" font-weight="bold">'
                     f'{titles[reference]}</text>')
        lines.append(f'<text x="{left+105}" y="378">relative x (m)</text>')
    lines.append('<text x="5" y="205" transform="rotate(-90 5 205)">relative z (m)</text>')
    lines.append('<line class="manual" x1="905" y1="52" x2="935" y2="52"/>'
                 '<text x="941" y="56">six demonstrations</text>')
    lines.append('<line class="auto" x1="1050" y1="52" x2="1080" y2="52"/>'
                 '<text x="1086" y="56">automatic</text>')
    lines.append('</svg>')
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _at_x(crossings, reference, threshold):
    return next(row for row in crossings[reference]
                if abs(row["relative_x_threshold_m"] - threshold) < 1e-12)


def write_markdown(path, report):
    rows = []
    for reference in REFERENCES:
        item = _at_x(report["reference_point_crossings"], reference, -0.40)
        manual = item["manual_relative_z_m"]
        automatic = item["automatic"]
        rows.append(
            f"| {reference} | {automatic['relative_z_m']:.3f} | "
            f"{manual['minimum']:.3f}–{manual['maximum']:.3f} | "
            f"{item['automatic_minus_manual_median_z_m']:.3f} |")
    auto_gate = report["entry_gate_diagnosis"]["automatic"]
    reverse = report["reverse_action_diagnosis"]
    content = f"""# 进场高度漂移与卷斗反向诊断

更新：2026-09-19。状态：固定日志归因完成；未运行物理、未训练、未读取封存轨迹。

## 结论

进场高度漂移不是日志坐标减法、四元数或铲斗参考点混用造成的假象。自动轨迹在铲斗原点、斗齿尖和斗口三个参考点上都明显低于6条人工示范。直接原因是自动策略在接近阶段长时间输出接近饱和的负动臂动作；入口高度窗口在横向条件满足之前就被向下穿过，路由继续停留在`approach`，从而继续放大偏差。

卷斗反向也不是液压缸正负号或专家标签写反。当前机构正卷斗映射已经由日志验证，训练索引的`curl_load`监督行中负卷斗标签为0；但失败固定轨迹在已路由到`curl_load`后仍输出{reverse['event']['raw_action'][3]:.3f}，说明问题位于闭环状态覆盖、阶段内泛化或路由契约，而不是简单翻转动作符号。现有保护在物理推进前暂停是正确的。

## 同一参考点比较

在各自参考点的目标相对x = -0.40 m处：

| 参考点 | 自动相对z（m） | 人工范围（m） | 自动减人工中位数（m） |
|---|---:|---:|---:|
{chr(10).join(rows)}

三种点均得到同方向差异，所以不能通过把`bucket_origin`改名为斗齿尖来消除问题。后续应保留原点字段用于机构状态，同时明确新增斗齿尖/斗口几何字段；碰撞、接触和进场间隙使用真实工具几何，不静默改变旧字段语义。

## 入口路由死锁

- 自动轨迹首次进入0.85–1.05 m高度窗：step {auto_gate['first_vertical_window']['decision_step']}，此时相对x={auto_gate['first_vertical_window']['relative_x_m']:.3f} m。
- 自动轨迹随后从高度窗下方退出：step {auto_gate['first_below_after_entering_vertical_window']['decision_step']}，此时相对x={auto_gate['first_below_after_entering_vertical_window']['relative_x_m']:.3f} m，距当前x门槛仍有{ENTRY_X_MIN_M-auto_gate['first_below_after_entering_vertical_window']['relative_x_m']:.3f} m。
- 终止时相对x={auto_gate['terminal']['relative_x_m']:.3f} m、相对z={auto_gate['terminal']['relative_z_m']:.3f} m；从未同时满足当前入口窗口。
- 到原点x=-0.40 m为止，自动轨迹强负动臂占比为{report['approach_action_diagnosis']['automatic_to_origin_x_minus_0p4']['strong_negative_boom_fraction']:.1%}。

这说明当前入口条件本身没有坐标计算错误，但它与学习式接近组合后会形成不可恢复状态。既然总体路线已经决定由几何闭环负责接近，后续不再让接触策略自行寻找这个窗口。

## 卷斗反向

- 训练索引`curl_load`的有效卷斗监督行：{reverse['training_curl_bucket_support']['supervised_rows']}；负向行：{reverse['training_curl_bucket_support']['sign_counts_deadband_0p1'].get('negative', 0)}。
- 失败事件：step {reverse['event']['first_reverse_decision_step']}开始反向，step {reverse['event']['decision_step']}确认；路由阶段为`{reverse['event']['routed_phase']}`。
- 当时原始四轴动作：`{reverse['event']['raw_action']}`。
- 机构日志中缸速—铲斗关节速度主符号一致率：{reverse['mechanism_bucket_mapping_dominant_sign_fraction']:.2%}。

因此已排除“控制接口方向写反”和“curl专家标签整体写反”。尚不能从当前归档确定网络内部是哪一个特征触发反向；失败原始逐步轨迹未作为本报告输入。可以确定的是：路由一旦进入`curl_load`不会因卷斗进展不足自动恢复，而策略在闭环偏离训练状态后能够输出训练标签中没有的强负动作。

## 对下一工作包的约束

1. 几何闭环把机构送到明确的近土起姿，并在交接时显式初始化`penetrate_deepen`，不再由接触策略完成`approach`。
2. 交接契约同时记录铲斗原点、斗齿尖和斗口；当前`relative_to_target_m`继续表示原点，避免破坏旧模型schema。
3. 阶段切换不能只依赖策略自己的动作；至少加入工具几何进展、接触历史和超时/回退状态。
4. 保留两拍反向卷斗保护。主动开斗脱困以后作为独立恢复状态设计，不能取消保护来掩盖策略错误。
5. 工作包C从近土交接状态重新切片和标注；在这之前不重训BC、不启动RL。

完整逐阈值结果见`crossings.csv`，轨迹图见`entry-reference-paths.svg`，机器可读证据见`report.json`。
"""
    Path(path).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-index", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--automatic-policy-log", type=Path, required=True)
    parser.add_argument("--action-audit-report", type=Path, required=True)
    parser.add_argument("--legacy-drift-report", type=Path, required=True)
    parser.add_argument("--reverse-source-report", type=Path, required=True)
    parser.add_argument("--reverse-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("output_exists:" + str(args.output))

    index, _, phase_support = load_index_records(args.actor_index)
    training = {
        episode_id: load_trace(
            args.training_root / episode_id / "policy-steps.ndjson")
        for episode_id in TRAINING_EPISODE_ALLOWLIST
    }
    automatic = load_trace(args.automatic_policy_log)
    action_audit = read_json(args.action_audit_report)
    legacy_drift = read_json(args.legacy_drift_report)
    reverse_source = read_json(args.reverse_source_report)
    reverse_audit = read_json(args.reverse_audit)

    crossings = summarize_crossings(training, automatic)
    training_gates = {
        episode_id: entry_gate_summary(trace["rows"], trace["terminal_state"])
        for episode_id, trace in training.items()
    }
    automatic_gate = entry_gate_summary(
        automatic["rows"], automatic["terminal_state"])
    approach_actions = {
        "automatic_to_origin_x_minus_0p4": action_summary_until_crossing(
            automatic["rows"]),
        "training_to_origin_x_minus_0p4": {
            episode_id: action_summary_until_crossing(trace["rows"])
            for episode_id, trace in training.items()
        },
    }
    teacher_stats = legacy_drift["teacher_forced_zero_boom_stats"]
    teacher_total = sum(item["zero_boom_target_rows"] for item in teacher_stats)
    teacher_weighted_mean = sum(
        item["mean_predicted_boom"] * item["zero_boom_target_rows"]
        for item in teacher_stats) / teacher_total
    teacher_strong_down = sum(item["strong_down_predictions"]
                              for item in teacher_stats)
    reverse_event = reverse_audit["known_reverse_risk_event"]
    bucket_mapping = action_audit["model_kinematics"][
        "cylinder_to_relative_joint_mapping"]["bucket"]
    reverse_diagnosis = {
        "event": reverse_event,
        "training_curl_bucket_support": phase_support["curl_load"]["bucket"],
        "mechanism_bucket_mapping_dominant_sign_fraction":
            bucket_mapping["dominant_sign_fraction"],
        "mechanism_bucket_mapping_ratio_sign_counts":
            bucket_mapping["ratio_sign_counts"],
        "direction_inversion_supported": False,
        "expert_label_sign_error_supported": False,
        "closed_loop_state_or_phase_generalization_failure_supported": True,
        "exact_network_internal_trigger": "unknown_from_available_fixed_logs",
    }

    reference_gap_checks = {}
    for reference in REFERENCES:
        at_x = _at_x(crossings, reference, -0.40)
        reference_gap_checks[reference] = (
            at_x["automatic"] is not None
            and at_x["manual_available_count"] == len(TRAINING_EPISODE_ALLOWLIST)
            and at_x["automatic"]["relative_z_m"]
            < at_x["manual_relative_z_m"]["minimum"])
    all_traces = [*training.values(), automatic]
    checks = {
        "explicit_six_training_episodes_only": (
            set(training) == set(TRAINING_EPISODE_ALLOWLIST)
            and len(index["episode_ids"]) == 6),
        "coordinate_reconstruction_exact": all(
            trace["maximum_reconstruction_error"]["origin_relative_m"]
            <= TOLERANCE for trace in all_traces),
        "quaternion_normalization_exact": all(
            trace["maximum_reconstruction_error"]["quaternion_norm"]
            <= TOLERANCE for trace in all_traces),
        "mouth_angle_reconstruction_exact": all(
            trace["maximum_reconstruction_error"]["mouth_angle_deg"]
            <= 1e-6 for trace in all_traces),
        "targets_constant_per_trace": all(
            trace["target_constant"] for trace in all_traces),
        "target_common_across_all_traces": len({
            tuple(trace["target_world_m"]) for trace in all_traces}) == 1,
        "prior_drift_sources_match_current_logs": (
            all(prior_hash_matches(
                legacy_drift, episode_id, trace["source_sha256"])
                for episode_id, trace in training.items())
            and prior_hash_matches(
                legacy_drift, args.automatic_policy_log.parent.name,
                automatic["source_sha256"])),
        "drift_persists_for_origin_tip_and_mouth_at_x_minus_0p4":
            all(reference_gap_checks.values()),
        "automatic_exits_vertical_window_before_horizontal_gate":
            automatic_gate["vertical_window_exited_before_horizontal_ready"],
        "automatic_never_reaches_joint_entry_gate":
            automatic_gate["first_joint_ready"] is None,
        "teacher_forced_zero_boom_bias_report_has_no_holdout": (
            legacy_drift["scope"].get("holdouts_opened") == 0),
        "action_audit_passed_without_holdout": (
            action_audit["status"] == "passed"
            and action_audit["scope"].get("holdout_files_opened") == 0),
        "curl_training_labels_have_no_negative_bucket_supervision": (
            phase_support["curl_load"]["bucket"]
            ["sign_counts_deadband_0p1"].get("negative", 0) == 0),
        "reverse_event_is_post_curl_negative_bucket": (
            reverse_event["routed_phase"] == "curl_load"
            and reverse_event["raw_action"][3] < REVERSE_THRESHOLD),
        "reverse_source_used_no_holdout_or_physics": (
            reverse_source["scope"].get("holdout_opened") is False
            and reverse_source["scope"].get("environment_steps") == 0),
        "physics_steps_zero": True,
        "parameter_updates_zero": True,
        "holdout_files_opened_zero": True,
        "checkpoint_files_loaded_zero": True,
    }
    report = {
        "schema": "excavation_entry_drift_and_reverse_diagnosis_v1",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "reference_point_contract": {
            "logged_bucket_position_world_m": "bucket body origin",
            "logged_bucket_relative_to_target_m": "bucket body origin minus target",
            "derived_local_points_m": {
                name: list(value) for name, value in REFERENCES.items()},
            "field_change": "none; derived points are audit-only",
        },
        "trace_contract": {
            "training_episode_ids": list(TRAINING_EPISODE_ALLOWLIST),
            "training": {episode_id: {
                key: value for key, value in trace.items()
                if key not in ("rows", "terminal_state")}
                for episode_id, trace in training.items()},
            "automatic_episode_id": args.automatic_policy_log.parent.name,
            "automatic": {
                key: value for key, value in automatic.items()
                if key not in ("rows", "terminal_state")},
        },
        "reference_point_crossings": crossings,
        "entry_gate_diagnosis": {
            "contract": {
                "reference": "bucket_origin",
                "relative_x_min_m": ENTRY_X_MIN_M,
                "relative_z_range_m": list(ENTRY_Z_RANGE_M),
                "confirmation_ticks": 2,
            },
            "training": training_gates,
            "automatic": automatic_gate,
            "root_cause": (
                "automatic approach action accumulates excessive descent; the "
                "vertical window is crossed before horizontal readiness and the "
                "approach-only route cannot recover"),
            "coordinate_bug_supported": False,
            "reference_point_mixup_supported": False,
            "policy_action_accumulation_supported": True,
            "route_deadlock_amplification_supported": True,
        },
        "approach_action_diagnosis": {
            **approach_actions,
            "teacher_forced_expert_zero_boom_rows": teacher_total,
            "teacher_forced_weighted_mean_predicted_boom": teacher_weighted_mean,
            "teacher_forced_strong_down_predictions": teacher_strong_down,
            "teacher_forced_source": str(args.legacy_drift_report),
            "boundary": (
                "teacher-forced predictions support a negative boom bias but are "
                "not a closed-loop counterfactual"),
        },
        "training_phase_action_support": phase_support,
        "reverse_action_diagnosis": reverse_diagnosis,
        "conclusions": {
            "entry_height_drift": "policy_action_accumulation_amplified_by_entry_gate",
            "entry_coordinate_chain": "verified_not_primary_cause",
            "curl_reverse": "policy_generalization_or_route_state_failure_not_sign_inversion",
            "exact_network_internal_trigger": "unresolved",
            "deployment_decision": "keep_protection_and_do_not_retrain_before_handoff_resegmentation",
        },
        "scope": {
            "generator_sha256": sha256(Path(__file__)),
            "input_sha256": {
                "actor_index": sha256(args.actor_index),
                "automatic_policy_log": sha256(args.automatic_policy_log),
                "action_audit_report": sha256(args.action_audit_report),
                "legacy_drift_report": sha256(args.legacy_drift_report),
                "reverse_source_report": sha256(args.reverse_source_report),
                "reverse_audit": sha256(args.reverse_audit),
            },
            "holdout_path_argument_supported": False,
            "holdout_files_opened": 0,
            "preexisting_holdout_derived_reports_used": False,
            "physics_steps": 0,
            "parameter_updates": 0,
            "checkpoint_files_loaded": 0,
            "claim": "fixed-log diagnosis only; no counterfactual physics or success claim",
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    write_crossings_csv(args.output / "crossings.csv", crossings)
    write_svg(args.output / "entry-reference-paths.svg", training, automatic)
    write_markdown(args.output / "report.md", report)
    artifacts = ("report.json", "report.md", "crossings.csv",
                 "entry-reference-paths.svg")
    (args.output / "manifest.sha256").write_text(
        "".join(f"{sha256(args.output / name)}  ./{name}\n" for name in artifacts),
        encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "output": str(args.output),
        "checks": checks,
        "conclusions": report["conclusions"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
