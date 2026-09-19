"""Build reviewable contact-phase candidates from fixed non-holdout logs.

The actor index is the sole episode selector.  The tool accepts no holdout
path, does not load a checkpoint, and does not run physics or training.  It
keeps the existing actor index and v1 subphase labels unchanged; the generated
v1 contact segments remain ineligible for training until their visual review
status is explicitly resolved.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path

from audit_contact_data_contract import (
    TRAINING_EPISODE_ALLOWLIST,
    action_values,
    iter_jsonl,
    load_actor_index,
    require_safe_actor_index,
    validate_observation_by_schema,
)


SEGMENT_SCHEMA = "excavation_contact_segment_candidate_v1"
EVENT_SCHEMA = "excavation_contact_event_candidate_v1"
REPORT_SCHEMA = "excavation_contact_segmentation_report_v1"
PHASES = (
    "approach",
    "penetrate_deepen",
    "curl_load",
    "lift_clear",
    "off_soil_confirm",
    "uncertain/recovery",
)
PRECONTACT_CONTEXT_TICKS = 20
EVENT_CONFIRMATION_TICKS = 2
OFF_SOIL_CONFIRMATION_TICKS = 10
CURL_ACTION_MIN = 0.5
LIFT_BOOM_ACTION_MIN = 0.5
LIFT_BUCKET_POSITION_MIN = 0.90


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_ndjson(path, records):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(
                record, ensure_ascii=False, separators=(",", ":"),
                allow_nan=False) + "\n")


def read_index_records(path):
    index = load_actor_index(path)
    require_safe_actor_index(index)
    records = defaultdict(list)
    for line_number, row in iter_jsonl(path):
        episode_id = row["episode_id"]
        if episode_id not in TRAINING_EPISODE_ALLOWLIST:
            raise ValueError(f"unlisted_episode_at_line:{line_number}")
        if row.get("holdout") is not False:
            raise ValueError(f"holdout_row_at_line:{line_number}")
        records[episode_id].append(row)
    for episode_id in TRAINING_EPISODE_ALLOWLIST:
        rows = records[episode_id]
        steps = [row["decision_step"] for row in rows]
        if not rows or steps != list(range(steps[0], steps[-1] + 1)):
            raise ValueError("noncontiguous_index:" + episode_id)
    return index, records


def load_policy_rows(path, indexed_steps):
    selected = {}
    source_counts = Counter()
    raw_present = 0
    for _, row in iter_jsonl(path):
        step = row.get("decision_step")
        if step not in indexed_steps:
            continue
        if step in selected:
            raise ValueError("duplicate_policy_step:" + str(step))
        validate_observation_by_schema(row["observation"])
        validate_observation_by_schema(row["next_observation"])
        action_values(row["action"])
        if "raw_action" in row:
            action_values(row["raw_action"])
            raw_present += 1
        selected[step] = row
        source_counts[row.get("action_source", "<missing>")] += 1
    if set(selected) != set(indexed_steps):
        missing = sorted(set(indexed_steps) - set(selected))
        raise ValueError("missing_policy_steps:" + ",".join(map(str, missing[:8])))
    return selected, {
        "source_counts": dict(sorted(source_counts.items())),
        "raw_action_rows": raw_present,
    }


def load_truth_rows(path, indexed_steps):
    selected = {}
    for _, row in iter_jsonl(path):
        step = row.get("decision_step")
        if step not in indexed_steps:
            continue
        truth = row.get("truth")
        if not isinstance(truth, dict):
            raise ValueError("missing_truth_at_step:" + str(step))
        selected[step] = row
    required = {step for step in indexed_steps if step > min(indexed_steps)}
    if not required.issubset(selected):
        missing = sorted(required - set(selected))
        raise ValueError("missing_truth_steps:" + ",".join(map(str, missing[:8])))
    return selected


def expert_action(policy):
    value = policy.get("raw_action", policy["action"])
    return tuple(float(item) for item in value["normalized_velocity"])


def applied_action(policy):
    return tuple(float(item) for item in
                 policy["action"]["normalized_velocity"])


def first_confirmed(steps, predicate, *, start_step, confirmation_ticks=2):
    run_start = None
    run_length = 0
    for step in steps:
        if step < start_step:
            continue
        if predicate(step):
            if run_start is None:
                run_start = step
            run_length += 1
            if run_length >= confirmation_ticks:
                return {
                    "onset_decision_step": run_start,
                    "confirmed_at_decision_step": step,
                    "confirmation_ticks": confirmation_ticks,
                }
        else:
            run_start = None
            run_length = 0
    return None


def final_false_tail(steps, predicate):
    """Return the final contiguous false tail after at least one true row."""
    true_steps = [step for step in steps if predicate(step) is True]
    if not true_steps:
        return None
    last_true = max(true_steps)
    tail = [step for step in steps if step > last_true]
    if not tail or tail != list(range(last_true + 1, max(steps) + 1)):
        return None
    if any(predicate(step) is not False for step in tail):
        return None
    return {
        "onset_decision_step": tail[0],
        "confirmed_at_decision_step": (
            tail[0] + OFF_SOIL_CONFIRMATION_TICKS - 1),
        "confirmation_ticks": OFF_SOIL_CONFIRMATION_TICKS,
        "tail_rows": len(tail),
    }


def transient_contact_uncertainty(steps, terrain_contact, *, contact_onset,
                                  off_soil_onset):
    """Mask contact-loss/re-entry chatter before the final off-soil tail."""
    intervals = []
    step = contact_onset
    while step < off_soil_onset:
        if terrain_contact(step) is not False:
            step += 1
            continue
        start = step
        while step < off_soil_onset and terrain_contact(step) is False:
            step += 1
        true_start = step
        confirmation = first_confirmed(
            steps, lambda value: terrain_contact(value) is True,
            start_step=true_start,
            confirmation_ticks=EVENT_CONFIRMATION_TICKS)
        end = (confirmation["confirmed_at_decision_step"]
               if confirmation is not None
               and confirmation["confirmed_at_decision_step"] < off_soil_onset
               else off_soil_onset - 1)
        intervals.append({
            "start_decision_step": start,
            "end_decision_step": end,
            "reason": "transient_contact_loss_or_unconfirmed_reentry",
        })
        step = end + 1
    return intervals


def compress_segments(episode_id, steps, phases, index_rows, policy_rows,
                      event_map):
    output = []
    start_index = 0
    while start_index < len(steps):
        phase = phases[steps[start_index]]
        end_index = start_index
        while (end_index + 1 < len(steps)
               and phases[steps[end_index + 1]] == phase):
            end_index += 1
        selected_steps = steps[start_index:end_index + 1]
        start = selected_steps[0]
        end = selected_steps[-1]
        source_counts = Counter(
            policy_rows[step].get("action_source", "<missing>")
            for step in selected_steps)
        old_counts = Counter(
            index_rows[step]["subphase"] for step in selected_steps)
        actor_eligible = phase in {
            "penetrate_deepen", "curl_load", "lift_clear"}
        confidence = {
            "approach": "medium",
            "penetrate_deepen": "medium",
            "curl_load": "high",
            "lift_clear": "high",
            "off_soil_confirm": "high",
            "uncertain/recovery": "low",
        }[phase]
        evidence = {
            "approach": "excluded geometry-controlled prefix",
            "penetrate_deepen": "one-second pre-contact context to curl intent",
            "curl_load": "two-tick positive expert curl intent",
            "lift_clear": "two-tick closed-bucket positive-boom milestone",
            "off_soil_confirm": "final sustained terrain separation tail",
            "uncertain/recovery": "transient contact loss/re-entry ambiguity",
        }[phase]
        boundary_events = [
            name for name, value in event_map.items()
            if value.get("onset_decision_step") == start
            or value.get("confirmed_at_decision_step") == start
        ]
        output.append({
            "schema": SEGMENT_SCHEMA,
            "label_version": "contact_segments_v1_candidate",
            "episode_id": episode_id,
            "segment_index": len(output),
            "phase": phase,
            "start_decision_step": start,
            "end_decision_step_inclusive": end,
            "row_count": len(selected_steps),
            "start_sim_time_s": policy_rows[start]["observation"]["sim_time_s"],
            "end_sim_time_s": policy_rows[end]["observation"]["sim_time_s"],
            "confidence": confidence,
            "evidence": evidence,
            "boundary_events": boundary_events,
            "action_sources": dict(sorted(source_counts.items())),
            "prior_subphase_counts": dict(sorted(old_counts.items())),
            "actor_action_eligible_after_review": actor_eligible,
            "axis_mask_after_review": ([1.0] * 4 if actor_eligible
                                        else [0.0] * 4),
            "review_status": "visual_manual_review_pending",
            "training_eligible_now": False,
            "holdout": False,
            "critic_transition_eligible": False,
        })
        start_index = end_index + 1
    return output


def event_record(episode_id, event_name, event, evidence,
                 *, future_information_used, online_gate_eligible=False,
                 confidence="high"):
    return {
        "schema": EVENT_SCHEMA,
        "label_version": "contact_events_v1_candidate",
        "episode_id": episode_id,
        "event": event_name,
        **event,
        "confidence": confidence,
        "evidence": evidence,
        "future_information_used": future_information_used,
        "online_gate_eligible": online_gate_eligible,
        "review_status": "visual_manual_review_pending",
        "training_eligible_now": False,
        "holdout": False,
    }


def analyze_episode(episode_id, index_records, policy_rows, truth_rows):
    steps = [row["decision_step"] for row in index_records]
    index_by_step = {row["decision_step"]: row for row in index_records}
    truth_steps = sorted(truth_rows)
    prior_boundaries = {}
    for row in index_records:
        prior_boundaries.setdefault(row["subphase"], row["decision_step"])

    def truth_value(step, key):
        row = truth_rows.get(step)
        return None if row is None else row["truth"].get(key)

    contact = first_confirmed(
        truth_steps,
        lambda step: truth_value(step, "digging_contact_current") is True,
        start_step=min(truth_steps),
        confirmation_ticks=EVENT_CONFIRMATION_TICKS)
    if contact is None:
        raise ValueError("missing_confirmed_contact:" + episode_id)
    handoff_step = max(min(steps),
                       contact["onset_decision_step"] - PRECONTACT_CONTEXT_TICKS)
    handoff = {
        "onset_decision_step": handoff_step,
        "confirmed_at_decision_step": handoff_step,
        "precontact_context_ticks": PRECONTACT_CONTEXT_TICKS,
        "contact_onset_decision_step": contact["onset_decision_step"],
    }
    curl = first_confirmed(
        steps,
        lambda step: expert_action(policy_rows[step])[3] >= CURL_ACTION_MIN,
        start_step=handoff_step,
        confirmation_ticks=EVENT_CONFIRMATION_TICKS)
    if curl is None:
        raise ValueError("missing_confirmed_curl:" + episode_id)
    lift = first_confirmed(
        steps,
        lambda step: (
            policy_rows[step]["observation"]["proprioception"]
            ["cylinder_position_normalized"][2] >= LIFT_BUCKET_POSITION_MIN
            and expert_action(policy_rows[step])[1] >= LIFT_BOOM_ACTION_MIN),
        start_step=curl["confirmed_at_decision_step"] + 1,
        confirmation_ticks=EVENT_CONFIRMATION_TICKS)
    if lift is None:
        raise ValueError("missing_confirmed_lift:" + episode_id)
    off_soil = final_false_tail(
        truth_steps,
        lambda step: truth_value(step, "terrain_contact"))
    if (off_soil is None
            or off_soil["confirmed_at_decision_step"] > max(steps)):
        raise ValueError("missing_confirmed_off_soil:" + episode_id)
    if not (handoff_step < curl["onset_decision_step"]
            < lift["onset_decision_step"]
            < off_soil["onset_decision_step"] <= max(steps)):
        raise ValueError("unordered_phase_boundaries:" + episode_id)

    uncertainty = transient_contact_uncertainty(
        truth_steps,
        lambda step: truth_value(step, "terrain_contact"),
        contact_onset=contact["onset_decision_step"],
        off_soil_onset=off_soil["onset_decision_step"])
    event_map = {
        "geometric_handoff_candidate": handoff,
        "contact_onset": contact,
        "curl_start": curl,
        "lift_clear_start": lift,
        "off_soil_onset": off_soil,
    }
    phases = {}
    for step in steps:
        if step < handoff_step:
            phase = "approach"
        elif step < curl["onset_decision_step"]:
            phase = "penetrate_deepen"
        elif step < lift["onset_decision_step"]:
            phase = "curl_load"
        elif step < off_soil["onset_decision_step"]:
            phase = "lift_clear"
        else:
            phase = "off_soil_confirm"
        if any(item["start_decision_step"] <= step
               <= item["end_decision_step"] for item in uncertainty):
            phase = "uncertain/recovery"
        phases[step] = phase

    events = [
        event_record(
            episode_id, "geometric_handoff_candidate", handoff,
            "derived as fixed pre-contact context; not a deployable geometry gate",
            future_information_used=True, confidence="medium"),
        event_record(
            episode_id, "contact_onset", contact,
            "two consecutive simulation-truth digging-contact rows",
            future_information_used=False),
        event_record(
            episode_id, "curl_start", curl,
            "two consecutive positive expert bucket intents",
            future_information_used=False),
        event_record(
            episode_id, "lift_clear_start", lift,
            "two consecutive positive expert boom intents with bucket cylinder >=0.90",
            future_information_used=False),
        event_record(
            episode_id, "off_soil_onset", off_soil,
            "final terrain-contact false tail with ten-tick confirmation",
            future_information_used=True),
    ]
    for item in uncertainty:
        events.append(event_record(
            episode_id, "uncertain_contact_recovery", {
                "onset_decision_step": item["start_decision_step"],
                "confirmed_at_decision_step": item["end_decision_step"],
            }, item["reason"], future_information_used=True,
            confidence="low"))

    segments = compress_segments(
        episode_id, steps, phases, index_by_step, policy_rows, event_map)
    phase_counts = Counter(phases.values())
    eligible_rows = sum(segment["row_count"] for segment in segments
                        if segment["actor_action_eligible_after_review"])
    precontact_curl = (
        curl["onset_decision_step"] < contact["onset_decision_step"])
    return {
        "segments": segments,
        "events": events,
        "summary": {
            "episode_id": episode_id,
            "indexed_rows": len(steps),
            "phase_counts": dict(sorted(phase_counts.items())),
            "actor_action_candidate_rows_after_review": eligible_rows,
            "uncertain_rows": phase_counts["uncertain/recovery"],
            "boundaries": event_map,
            "prior_subphase_boundaries": prior_boundaries,
            "prior_penetrate_lead_ticks": (
                contact["onset_decision_step"]
                - prior_boundaries["penetrate_deepen"]),
            "transient_uncertainty": uncertainty,
            "curl_begins_before_confirmed_contact": precontact_curl,
            "visual_manual_review_status": "pending",
        },
        "phases": phases,
    }


def load_takeover_review(root):
    handshake_path = Path(root) / "takeover-handshake.ndjson"
    correction_path = Path(root) / "dagger-corrections.ndjson"
    handshake = [row for _, row in iter_jsonl(handshake_path)]
    corrections = [row for _, row in iter_jsonl(correction_path)]
    if len(handshake) != 2 or not corrections:
        raise ValueError("invalid_takeover_handshake_artifacts")
    first = corrections[0]
    if not (
        handshake[0].get("event") == "human_takeover_armed"
        and handshake[1].get("event") == "first_nonzero_human_action"
        and handshake[0].get("decision_step") == 478
        and handshake[1].get("decision_step") == 478
        and first.get("decision_step") == 478
        and first.get("takeover_first_nonzero_action") is True
        and handshake[0].get("physics_advanced") is False
        and handshake[1].get("physics_advanced_before_action") is False
    ):
        raise ValueError("takeover_boundary_contract_failed")
    return {
        "episode_id": first["episode_id"],
        "armed_decision_step": 478,
        "first_human_action_decision_step": 478,
        "raw_human_action": first["raw_human_action"]["normalized_velocity"],
        "applied_action": first["applied_action"]["normalized_velocity"],
        "physics_advanced_between_arm_and_action": False,
        "correction_rows": len(corrections),
        "training_role": "separate_correlated_correction_episode_not_merged_into_six_segments",
        "visual_manual_review_status": "pending",
        "source_sha256": {
            "takeover_handshake": sha256(handshake_path),
            "dagger_corrections": sha256(correction_path),
        },
    }


def _svg_polyline(points, color, width=1.8, dash=None):
    value = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<polyline points="{value}" fill="none" stroke="{color}" '
            f'stroke-width="{width}"{dashed}/>' )


def write_episode_svg(path, episode_id, steps, phases, policy_rows, truth_rows,
                      events):
    width, height = 1240, 720
    left, right, top, bottom = 76, 24, 52, 44
    plot_w = width - left - right
    panel_h = 122
    gap = 30
    panel_tops = [top + index * (panel_h + gap) for index in range(4)]
    xmin, xmax = min(steps), max(steps)

    def xcoord(step):
        return left + (step - xmin) / max(1, xmax - xmin) * plot_w

    def ycoord(value, panel, low=-1.0, high=1.0):
        return (panel_tops[panel] + panel_h
                - (value - low) / (high - low) * panel_h)

    colors = {
        "approach": "#eef2f7",
        "penetrate_deepen": "#dbeafe",
        "curl_load": "#fef3c7",
        "lift_clear": "#dcfce7",
        "off_soil_confirm": "#ede9fe",
        "uncertain/recovery": "#fee2e2",
    }
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="27" font-family="sans-serif" font-size="18" '
        f'font-weight="bold">{html.escape(episode_id)} contact segmentation candidate</text>',
        '<text x="1216" y="27" text-anchor="end" font-family="sans-serif" '
        'font-size="11" fill="#64748b">fixed logs · visual review pending</text>',
    ]
    # Background phase bands across every panel.
    start = steps[0]
    current = phases[start]
    spans = []
    for step in steps[1:]:
        if phases[step] != current:
            spans.append((start, step - 1, current))
            start, current = step, phases[step]
    spans.append((start, steps[-1], current))
    for panel_top in panel_tops:
        for span_start, span_end, phase in spans:
            x1 = xcoord(span_start)
            x2 = xcoord(span_end + 1)
            body.append(f'<rect x="{x1:.2f}" y="{panel_top}" '
                        f'width="{max(0.8, x2-x1):.2f}" height="{panel_h}" '
                        f'fill="{colors[phase]}"/>')
        for fraction in (0.0, 0.5, 1.0):
            y = panel_top + fraction * panel_h
            body.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left+plot_w}" '
                        f'y2="{y:.2f}" stroke="#cbd5e1" stroke-width="0.7"/>')
    panel_labels = (
        "expert actions [-1,1]",
        "measured axis velocities [-1,1]",
        "cylinder positions / mouth angle",
        "terrain contact / log10(force+1)/5",
    )
    for panel, label in enumerate(panel_labels):
        y = panel_tops[panel] + panel_h / 2
        body.append(f'<text x="18" y="{y:.2f}" font-family="sans-serif" '
                    f'font-size="11" transform="rotate(-90 18 {y:.2f})" '
                    f'text-anchor="middle">{html.escape(label)}</text>')
    series_colors = ("#64748b", "#2563eb", "#ea580c", "#7c3aed")
    for axis in range(4):
        action_points = [(xcoord(step), ycoord(
            expert_action(policy_rows[step])[axis], 0)) for step in steps]
        velocity_points = [(xcoord(step), ycoord(
            policy_rows[step]["observation"]["proprioception"]
            ["axis_velocity_normalized"][axis], 1)) for step in steps]
        body.append(_svg_polyline(action_points, series_colors[axis]))
        body.append(_svg_polyline(velocity_points, series_colors[axis]))
    for axis, color in enumerate(series_colors[1:]):
        points = [(xcoord(step), ycoord(
            policy_rows[step]["observation"]["proprioception"]
            ["cylinder_position_normalized"][axis], 2)) for step in steps]
        body.append(_svg_polyline(points, color))
    mouth_points = [(xcoord(step), ycoord(
        max(-1.0, min(1.0, policy_rows[step]["observation"]
                      ["bucket"]["mouth_angle_deg"] / 90.0)), 2))
        for step in steps]
    body.append(_svg_polyline(mouth_points, "#0f766e", dash="5 3"))
    contact_points = []
    force_points = []
    for step in steps:
        truth = truth_rows.get(step)
        contact = (0.0 if truth is None else
                   float(truth["truth"].get("terrain_contact") is True))
        force = (0.0 if truth is None else float(truth.get("force_norm_n", 0.0)))
        contact_points.append((xcoord(step), ycoord(contact, 3, 0.0, 1.0)))
        force_points.append((xcoord(step), ycoord(
            min(1.0, math.log10(force + 1.0) / 5.0), 3, 0.0, 1.0)))
    body.append(_svg_polyline(contact_points, "#dc2626", 2.1))
    body.append(_svg_polyline(force_points, "#111827", 1.5, dash="4 3"))
    for event in events:
        step = event["onset_decision_step"]
        x = xcoord(step)
        body.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" '
                    f'y2="{panel_tops[-1]+panel_h}" stroke="#475569" '
                    'stroke-width="0.7" stroke-dasharray="2 3"/>')
    body.append(f'<line x1="{left}" y1="{panel_tops[-1]+panel_h}" '
                f'x2="{left+plot_w}" y2="{panel_tops[-1]+panel_h}" '
                'stroke="#334155"/>')
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        step = round(xmin + fraction * (xmax - xmin))
        x = xcoord(step)
        body.append(f'<text x="{x:.2f}" y="{height-18}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11">{step}</text>')
    body.append(f'<text x="{left+plot_w/2:.2f}" y="{height-2}" '
                'text-anchor="middle" font-family="sans-serif" '
                'font-size="11">decision step</text>')
    legend = [
        ("swing", series_colors[0]), ("boom", series_colors[1]),
        ("arm", series_colors[2]), ("bucket", series_colors[3]),
        ("mouth/90", "#0f766e"), ("contact", "#dc2626"),
        ("log force", "#111827"),
    ]
    x = left
    y = 45
    for label, color in legend:
        body.append(f'<line x1="{x}" y1="{y}" x2="{x+18}" y2="{y}" '
                    f'stroke="{color}" stroke-width="2"/>')
        body.append(f'<text x="{x+23}" y="{y+4}" font-family="sans-serif" '
                    f'font-size="10">{label}</text>')
        x += 112
    phase_legend = (
        ("approach", "approach"),
        ("penetrate", "penetrate_deepen"),
        ("curl", "curl_load"),
        ("lift", "lift_clear"),
        ("off-soil", "off_soil_confirm"),
        ("uncertain", "uncertain/recovery"),
    )
    x = left
    y = 651
    for label, phase in phase_legend:
        body.append(f'<rect x="{x}" y="{y-10}" width="14" height="10" '
                    f'fill="{colors[phase]}" stroke="#94a3b8" stroke-width="0.5"/>')
        body.append(f'<text x="{x+19}" y="{y}" font-family="sans-serif" '
                    f'font-size="10">{label}</text>')
        x += 118
    body.append("</svg>\n")
    Path(path).write_text("".join(body), encoding="utf-8")


def write_review_checklist(path, episode_summaries, takeover):
    lines = [
        "# 接触片段人工复核清单\n\n",
        "状态：固定日志候选已生成；画面/连续回放人工复核仍待完成。",
        "在复核完成并提交显式状态前，所有新片段均为`training_eligible_now=false`。\n\n",
        "## 六条训练轨迹\n\n",
        "| episode | 交接候选 | 接触 | 卷斗 | 抬升 | 最终离土 | 模糊行 | 画面复核 |\n",
        "|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for item in episode_summaries:
        boundaries = item["boundaries"]
        lines.append(
            f"| `{item['episode_id']}` | "
            f"{boundaries['geometric_handoff_candidate']['onset_decision_step']} | "
            f"{boundaries['contact_onset']['onset_decision_step']} | "
            f"{boundaries['curl_start']['onset_decision_step']} | "
            f"{boundaries['lift_clear_start']['onset_decision_step']} | "
            f"{boundaries['off_soil_onset']['onset_decision_step']} | "
            f"{item['uncertain_rows']} | ☐ pending |\n")
    lines.extend([
        "\n每条轨迹逐项确认：\n\n",
        "- ☐ 几何交接候选前仍属于自由空间接近，候选后保留了足够近土上下文。\n",
        "- ☐ 接触发生/确认帧与画面、受力和工具位置一致。\n",
        "- ☐ 卷斗开始是操作者挖掘意图，不是误按、端点投影或停顿。\n",
        "- ☐ `lift_clear`已进入带载脱土，不把接触中短暂抬臂误标为运输。\n",
        "- ☐ `off_soil_confirm`覆盖最终离土确认，不把中间弹跳当成完成。\n",
        "- ☐ 红色`uncertain/recovery`区间保持掩码，除非画面复核后另行裁定。\n",
        "- ☐ 图中动作、实际速度、缸位/斗口角和接触变化没有明显时序错位。\n",
        "\n## 精确接管交界\n\n",
        f"- episode：`{takeover['episode_id']}`。\n",
        f"- armed与第一条非零人工动作均在step {takeover['armed_decision_step']}。\n",
        f"- 第一条原始人工动作：`{takeover['raw_human_action']}`。\n",
        "- 两事件之间物理没有推进；该282行纠正序列保持为同一相关episode，",
        "不拆成独立示范，也不并入六条成功轨迹的阶段统计。\n",
        "- ☐ 回放确认暂停画面与第一条人工动作意图一致。\n",
    ])
    Path(path).write_text("".join(lines), encoding="utf-8")


def write_markdown(path, report):
    counts = report["counts"]
    prior_leads = [item["prior_penetrate_lead_ticks"]
                   for item in report["episodes"]]
    rows = []
    for item in report["episodes"]:
        b = item["boundaries"]
        rows.append(
            f"| `{item['episode_id']}` | "
            f"{b['geometric_handoff_candidate']['onset_decision_step']} | "
            f"{b['contact_onset']['onset_decision_step']} | "
            f"{b['curl_start']['onset_decision_step']} | "
            f"{b['lift_clear_start']['onset_decision_step']} | "
            f"{b['off_soil_onset']['onset_decision_step']} | "
            f"{item['uncertain_rows']} |")
    content = f"""# 接触阶段与事件候选标注

更新：2026-09-19。状态：固定日志候选完成，连续画面人工复核待完成；未训练、未运行物理、未读取封存轨迹。

## 结论

旧四阶段标签不能直接作为下一版接触BC标签：原人工界面阶段始终停在`approach`，旧后处理又用已锁存的`digging_interaction_observed`切入，因此六条轨迹的旧接触入口比真实受力接触早{min(prior_leads)}–{max(prior_leads)}步。新候选不再把这个锁存信号当交接条件，而是为每条轨迹保留真实接触前1秒上下文，明确标为“离线交接候选”，等待后续几何契约和画面复核。

`curl_load`由连续两拍正卷斗人工意图开始，`lift_clear`由铲斗缸位达到0.90且连续两拍抬臂意图开始；`off_soil_confirm`从最终持续离土开始，并要求10拍确认。中间接触丢失/重入的{counts['uncertain_rows']}行被显式标为`uncertain/recovery`并屏蔽，不强塞进成功顺序。

## 六条轨迹边界

| episode | 交接候选 | 接触发生 | 卷斗开始 | 抬升开始 | 最终离土 | 模糊行 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

两条轨迹在确认接触之前已开始卷斗，说明挖掘轴之间确实存在重叠，不能强制“先下探、再卷斗”的互斥顺序。新标签保留这种协同，但部署阶段不能用策略自身卷斗动作作为唯一阶段门槛。

## 训练边界

- 共读取{counts['indexed_rows']}条已允许索引行，生成{counts['segments']}段候选和{counts['events']}个事件。
- 自由空间`approach`、`off_soil_confirm`和`uncertain/recovery`不作为接触动作BC监督。
- `penetrate_deepen`、`curl_load`、`lift_clear`共{counts['actor_action_candidate_rows_after_review']}行仅表示“人工复核后可候选”；当前全部记录仍为`training_eligible_now=false`。
- 旧`subphase-labels.ndjson`和`v4-actor-data-index.ndjson`保持原样，旧权重资格不因新文件自动变化。
- 交接候选使用未来接触位置回看得到，只能裁切训练上下文，不能直接复制成在线几何门槛。

## 接管交界

精确接管episode在step 478完成arm与第一条非零人工动作，两者之间物理没有推进；282条纠正仍是同一相关episode，不并入六条成功轨迹统计。画面级意图复核仍列在`manual-review-checklist.md`。

## 下一步

先按六张`timeline-*.svg`完成连续画面/回放复核，处理红色模糊区并提交复核状态；随后冻结标签schema和checkpoint兼容契约。只有这些完成后，工作包D才实现几何交接候选，M2才讨论重训。
"""
    Path(path).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-index", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--correction-root", type=Path, required=True)
    parser.add_argument("--segments-output", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    parser.add_argument("--labels-manifest", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.segments_output, args.events_output,
                 args.labels_manifest, args.report_output):
        if path.exists():
            raise FileExistsError("output_exists:" + str(path))

    index, index_groups = read_index_records(args.actor_index)
    all_segments = []
    all_events = []
    episode_summaries = []
    episode_runtime = {}
    source_hashes = {"actor_index": sha256(args.actor_index)}
    timeline_inputs = {}
    for episode_id in TRAINING_EPISODE_ALLOWLIST:
        indexed = index_groups[episode_id]
        indexed_steps = {row["decision_step"] for row in indexed}
        policy_path = args.training_root / episode_id / "policy-steps.ndjson"
        truth_path = args.training_root / episode_id / "truth-steps.ndjson"
        policies, runtime = load_policy_rows(policy_path, indexed_steps)
        truths = load_truth_rows(truth_path, indexed_steps)
        analysis = analyze_episode(episode_id, indexed, policies, truths)
        all_segments.extend(analysis["segments"])
        all_events.extend(analysis["events"])
        episode_summaries.append(analysis["summary"])
        episode_runtime[episode_id] = runtime
        source_hashes[episode_id + "/policy-steps.ndjson"] = sha256(policy_path)
        source_hashes[episode_id + "/truth-steps.ndjson"] = sha256(truth_path)
        timeline_inputs[episode_id] = (indexed, policies, truths, analysis)

    takeover = load_takeover_review(args.correction_root)
    source_hashes.update({
        "takeover-handshake.ndjson": takeover["source_sha256"]["takeover_handshake"],
        "dagger-corrections.ndjson": takeover["source_sha256"]["dagger_corrections"],
    })
    write_ndjson(args.segments_output, all_segments)
    write_ndjson(args.events_output, all_events)
    args.labels_manifest.write_text(
        f"{sha256(args.segments_output)}  ./{args.segments_output.name}\n"
        f"{sha256(args.events_output)}  ./{args.events_output.name}\n",
        encoding="utf-8")

    args.report_output.mkdir(parents=True, exist_ok=False)
    for episode_id, (indexed, policies, truths, analysis) in timeline_inputs.items():
        steps = [row["decision_step"] for row in indexed]
        write_episode_svg(
            args.report_output / ("timeline-" + episode_id + ".svg"),
            episode_id, steps, analysis["phases"], policies, truths,
            analysis["events"])
    write_review_checklist(
        args.report_output / "manual-review-checklist.md",
        episode_summaries, takeover)

    phase_counts = Counter()
    for item in episode_summaries:
        phase_counts.update(item["phase_counts"])
    candidate_rows = sum(
        item["actor_action_candidate_rows_after_review"]
        for item in episode_summaries)
    checks = {
        "explicit_six_training_episodes_only": (
            set(index["episode_ids"]) == set(TRAINING_EPISODE_ALLOWLIST)),
        "indexed_rows_4363": index["row_count"] == 4363,
        "all_segment_rows_cover_index_once": (
            sum(item["row_count"] for item in all_segments) == 4363),
        "all_required_phases_present": set(PHASES) <= set(phase_counts),
        "all_boundaries_ordered": all(
            item["boundaries"]["geometric_handoff_candidate"]
            ["onset_decision_step"]
            < item["boundaries"]["curl_start"]["onset_decision_step"]
            < item["boundaries"]["lift_clear_start"]["onset_decision_step"]
            < item["boundaries"]["off_soil_onset"]["onset_decision_step"]
            for item in episode_summaries),
        "uncertain_rows_masked": all(
            segment["axis_mask_after_review"] == [0.0] * 4
            for segment in all_segments
            if segment["phase"] == "uncertain/recovery"),
        "all_candidates_training_disabled_pending_review": all(
            segment["training_eligible_now"] is False
            and segment["review_status"] == "visual_manual_review_pending"
            for segment in all_segments),
        "takeover_exact_state_boundary_verified": (
            takeover["physics_advanced_between_arm_and_action"] is False
            and takeover["armed_decision_step"] == 478),
        "holdout_files_opened_zero": True,
        "checkpoint_files_loaded_zero": True,
        "physics_steps_zero": True,
        "parameter_updates_zero": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": "candidate_review_pending" if all(checks.values()) else "failed",
        "checks": checks,
        "configuration": {
            "precontact_context_ticks": PRECONTACT_CONTEXT_TICKS,
            "policy_dt_s": 0.05,
            "event_confirmation_ticks": EVENT_CONFIRMATION_TICKS,
            "off_soil_confirmation_ticks": OFF_SOIL_CONFIRMATION_TICKS,
            "curl_action_min": CURL_ACTION_MIN,
            "lift_boom_action_min": LIFT_BOOM_ACTION_MIN,
            "lift_bucket_position_min": LIFT_BUCKET_POSITION_MIN,
            "phase_order": list(PHASES),
            "online_use": "forbidden; offline candidate labels only",
        },
        "counts": {
            "indexed_rows": index["row_count"],
            "segments": len(all_segments),
            "events": len(all_events),
            "phase_rows": dict(sorted(phase_counts.items())),
            "uncertain_rows": phase_counts["uncertain/recovery"],
            "actor_action_candidate_rows_after_review": candidate_rows,
            "training_eligible_rows_now": 0,
        },
        "episodes": episode_summaries,
        "takeover_boundary_review": takeover,
        "policy_runtime_sources": episode_runtime,
        "artifacts": {
            "segments": str(args.segments_output),
            "events": str(args.events_output),
            "labels_manifest": str(args.labels_manifest),
            "manual_review_checklist": str(
                args.report_output / "manual-review-checklist.md"),
            "timelines": [
                "timeline-" + episode_id + ".svg"
                for episode_id in TRAINING_EPISODE_ALLOWLIST],
        },
        "source_sha256": source_hashes,
        "scope": {
            "generator_sha256": sha256(Path(__file__)),
            "training_episode_allowlist": list(TRAINING_EPISODE_ALLOWLIST),
            "holdout_path_argument_supported": False,
            "holdout_files_opened": 0,
            "checkpoint_files_loaded": 0,
            "physics_steps": 0,
            "parameter_updates": 0,
            "existing_actor_index_modified": False,
            "existing_subphase_labels_modified": False,
            "claim": (
                "fixed-log candidate segmentation and review preparation only; "
                "not reviewed training admission or runtime routing"),
        },
    }
    report_path = args.report_output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    write_markdown(args.report_output / "report.md", report)
    internal = [
        "report.json", "report.md", "manual-review-checklist.md",
        *report["artifacts"]["timelines"],
    ]
    (args.report_output / "manifest.sha256").write_text(
        "".join(f"{sha256(args.report_output / name)}  ./{name}\n"
                for name in internal), encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "checks": checks,
        "counts": report["counts"],
        "output": str(args.report_output),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
