"""Canonical mixed-material state used at continuous scoop boundaries.

This module is deliberately passive: it validates and hashes snapshots, but it
never owns a selector and never mutates a simulator.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable


STATE_SCHEMA = "excavation_mixed_three_cycle_state_v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STATE_KEYS = {
    "schema", "physics_instance_id", "cycle_id", "sim_time_s",
    "physics_step", "mechanism", "mpm", "rocks", "bucket", "truck",
    "ledger", "contacts",
}


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"mixed_state_number_required:{name}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"mixed_state_nonfinite:{name}")
    if minimum is not None and result < minimum:
        raise ValueError(f"mixed_state_below_minimum:{name}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"mixed_state_integer_required:{name}")
    return value


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"mixed_state_vector_shape:{name}")
    return tuple(_finite_number(item, f"{name}[{index}]")
                 for index, item in enumerate(value))


def _string_list(value: Any, name: str) -> list[str]:
    if (not isinstance(value, list) or
            any(not isinstance(item, str) or not item for item in value)):
        raise ValueError(f"mixed_state_string_list:{name}")
    if len(value) != len(set(value)):
        raise ValueError(f"mixed_state_duplicate_id:{name}")
    return value


def _exact_keys(value: Any, expected: Iterable[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"mixed_state_keys:{name}")


def validate_state(snapshot: dict) -> None:
    """Reject incomplete, non-finite, non-conservative boundary snapshots."""

    _exact_keys(snapshot, STATE_KEYS, "root")
    if snapshot["schema"] != STATE_SCHEMA:
        raise ValueError("mixed_state_schema")
    if (not isinstance(snapshot["physics_instance_id"], str) or
            not snapshot["physics_instance_id"]):
        raise ValueError("mixed_state_physics_instance_id")
    cycle_id = _integer(snapshot["cycle_id"], "cycle_id")
    if cycle_id > 3:
        raise ValueError("mixed_state_cycle_id")
    _finite_number(snapshot["sim_time_s"], "sim_time_s", minimum=0.0)
    _integer(snapshot["physics_step"], "physics_step")

    mechanism = snapshot["mechanism"]
    _exact_keys(mechanism, {"qpos", "qvel", "command"}, "mechanism")
    _vector(mechanism["qpos"], 19, "mechanism.qpos")
    _vector(mechanism["qvel"], 18, "mechanism.qvel")
    _vector(mechanism["command"], 4, "mechanism.command")

    mpm = snapshot["mpm"]
    _exact_keys(
        mpm,
        {"particle_count", "total_mass_kg", "state_sha256", "history_step"},
        "mpm",
    )
    _integer(mpm["particle_count"], "mpm.particle_count", minimum=1)
    _finite_number(mpm["total_mass_kg"], "mpm.total_mass_kg", minimum=0.0)
    if (float(mpm["total_mass_kg"]) <= 0.0 or
            not isinstance(mpm["state_sha256"], str) or
            not SHA256_RE.fullmatch(mpm["state_sha256"])):
        raise ValueError("mixed_state_mpm_contract")
    _integer(mpm["history_step"], "mpm.history_step")

    rocks = snapshot["rocks"]
    if not isinstance(rocks, list) or not rocks:
        raise ValueError("mixed_state_rocks_required")
    rock_ids: list[str] = []
    rock_masses: dict[str, float] = {}
    rock_keys = {
        "id", "mass_kg", "position_world_m", "quaternion_wxyz",
        "linear_velocity_world_m_s", "angular_velocity_world_rad_s",
        "contact_objects",
    }
    for index, rock in enumerate(rocks):
        _exact_keys(rock, rock_keys, f"rocks[{index}]")
        rock_id = rock["id"]
        if not isinstance(rock_id, str) or not rock_id:
            raise ValueError("mixed_state_rock_id")
        rock_ids.append(rock_id)
        mass = _finite_number(rock["mass_kg"], f"rocks[{index}].mass_kg")
        if mass <= 0.0:
            raise ValueError("mixed_state_rock_mass")
        rock_masses[rock_id] = mass
        _vector(rock["position_world_m"], 3, f"rocks[{index}].position")
        quaternion = _vector(
            rock["quaternion_wxyz"], 4, f"rocks[{index}].quaternion")
        if abs(math.sqrt(sum(item * item for item in quaternion)) - 1.0) > 1e-6:
            raise ValueError("mixed_state_rock_quaternion_not_unit")
        _vector(
            rock["linear_velocity_world_m_s"], 3,
            f"rocks[{index}].linear_velocity",
        )
        _vector(
            rock["angular_velocity_world_rad_s"], 3,
            f"rocks[{index}].angular_velocity",
        )
        _string_list(rock["contact_objects"], f"rocks[{index}].contacts")
    if rock_ids != sorted(rock_ids) or len(rock_ids) != len(set(rock_ids)):
        raise ValueError("mixed_state_rock_ids_not_stable_sorted_unique")

    bucket = snapshot["bucket"]
    _exact_keys(
        bucket,
        {"state_sha256", "residual_mpm_mass_kg", "carried_rock_ids"},
        "bucket",
    )
    if (not isinstance(bucket["state_sha256"], str) or
            not SHA256_RE.fullmatch(bucket["state_sha256"])):
        raise ValueError("mixed_state_bucket_hash")
    _finite_number(
        bucket["residual_mpm_mass_kg"], "bucket.residual_mpm_mass_kg",
        minimum=0.0,
    )
    carried = _string_list(bucket["carried_rock_ids"], "bucket.carried_rock_ids")

    truck = snapshot["truck"]
    _exact_keys(truck, {"stable_mpm_mass_kg", "stable_rock_ids"}, "truck")
    _finite_number(
        truck["stable_mpm_mass_kg"], "truck.stable_mpm_mass_kg", minimum=0.0,
    )
    truck_ids = _string_list(truck["stable_rock_ids"], "truck.stable_rock_ids")

    ledger = snapshot["ledger"]
    ledger_keys = {
        "source_mpm_mass_kg", "bucket_mpm_mass_kg", "truck_mpm_mass_kg",
        "unassigned_mpm_mass_kg", "source_rock_ids", "bucket_rock_ids",
        "truck_rock_ids", "unassigned_rock_ids", "mpm_mass_residual_kg",
        "rock_mass_residual_kg",
    }
    _exact_keys(ledger, ledger_keys, "ledger")
    mpm_parts = []
    for key in (
        "source_mpm_mass_kg", "bucket_mpm_mass_kg", "truck_mpm_mass_kg",
        "unassigned_mpm_mass_kg",
    ):
        mpm_parts.append(_finite_number(ledger[key], f"ledger.{key}", minimum=0.0))
    mpm_residual = _finite_number(
        ledger["mpm_mass_residual_kg"], "ledger.mpm_mass_residual_kg")
    rock_residual = _finite_number(
        ledger["rock_mass_residual_kg"], "ledger.rock_mass_residual_kg")
    if (abs(sum(mpm_parts) - float(mpm["total_mass_kg"])) > 1e-8 or
            abs(mpm_residual) > 1e-8 or abs(rock_residual) > 1e-8):
        raise ValueError("mixed_state_mass_not_conserved")

    category_lists = []
    for key in (
        "source_rock_ids", "bucket_rock_ids", "truck_rock_ids",
        "unassigned_rock_ids",
    ):
        category_lists.append(_string_list(ledger[key], f"ledger.{key}"))
    flattened = [item for items in category_lists for item in items]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(rock_ids):
        raise ValueError("mixed_state_rock_ledger_not_conserved")
    if set(carried) != set(ledger["bucket_rock_ids"]):
        raise ValueError("mixed_state_bucket_ledger_mismatch")
    if set(truck_ids) != set(ledger["truck_rock_ids"]):
        raise ValueError("mixed_state_truck_ledger_mismatch")
    if abs(float(bucket["residual_mpm_mass_kg"]) -
           float(ledger["bucket_mpm_mass_kg"])) > 1e-8:
        raise ValueError("mixed_state_bucket_mpm_ledger_mismatch")
    if abs(float(truck["stable_mpm_mass_kg"]) -
           float(ledger["truck_mpm_mass_kg"])) > 1e-8:
        raise ValueError("mixed_state_truck_mpm_ledger_mismatch")
    if abs(sum(rock_masses.values()) - sum(
            rock_masses[item] for item in flattened)) > 1e-12:
        raise ValueError("mixed_state_rock_mass_not_conserved")

    contacts = snapshot["contacts"]
    _exact_keys(contacts, {"forbidden_contact_count"}, "contacts")
    _integer(contacts["forbidden_contact_count"], "forbidden_contact_count")


def state_sha256(snapshot: dict) -> str:
    validate_state(snapshot)
    return canonical_sha256(snapshot)


def build_state_snapshot(
        *, physics_instance_id: str, cycle_id: int, physics_step: int,
        bridge_state: dict, command: list[float], mpm: dict,
        rock_mass_kg: dict[str, float], rock_contact_objects: dict[str, list[str]],
        bucket: dict, truck: dict, ledger: dict,
) -> dict:
    """Adapt the live R3-C bridge and ledger outputs without changing them.

    The bridge keeps the 19/18 mechanism state separate from its per-rock
    free-joint state.  MPM hashes and ownership ledgers are supplied by the
    caller because reading them must remain an explicit observation operation.
    """

    bridge_keys = {
        "time_s", "mechanism_qpos", "mechanism_qvel", "rocks",
        "forbidden_contact_count",
    }
    if not isinstance(bridge_state, dict) or not bridge_keys.issubset(bridge_state):
        raise ValueError("mixed_state_bridge_fields")
    mpm_keys = {"particle_count", "total_mass_kg", "state_sha256", "history_step"}
    if not isinstance(mpm, dict) or set(mpm) != mpm_keys:
        raise ValueError("mixed_state_mpm_adapter_fields")
    rocks = []
    raw_rocks = bridge_state["rocks"]
    if not isinstance(raw_rocks, list):
        raise ValueError("mixed_state_bridge_rocks")
    for raw in raw_rocks:
        rock_id = raw.get("rock_id") if isinstance(raw, dict) else None
        if (not isinstance(rock_id, str) or rock_id not in rock_mass_kg or
                rock_id not in rock_contact_objects):
            raise ValueError("mixed_state_bridge_rock_binding")
        qpos = raw.get("qpos")
        if not isinstance(qpos, list) or len(qpos) != 7:
            raise ValueError("mixed_state_bridge_rock_qpos")
        rocks.append({
            "id": rock_id,
            "mass_kg": rock_mass_kg[rock_id],
            "position_world_m": list(raw.get("position", qpos[:3])),
            "quaternion_wxyz": list(qpos[3:7]),
            "linear_velocity_world_m_s": list(
                raw.get("linear_velocity_world_m_s", [])),
            "angular_velocity_world_rad_s": list(
                raw.get("angular_velocity_world_rad_s", [])),
            "contact_objects": list(rock_contact_objects[rock_id]),
        })
    snapshot = {
        "schema": STATE_SCHEMA,
        "physics_instance_id": physics_instance_id,
        "cycle_id": cycle_id,
        "sim_time_s": bridge_state["time_s"],
        "physics_step": physics_step,
        "mechanism": {
            "qpos": list(bridge_state["mechanism_qpos"]),
            "qvel": list(bridge_state["mechanism_qvel"]),
            "command": list(command),
        },
        "mpm": dict(mpm),
        "rocks": rocks,
        "bucket": dict(bucket),
        "truck": dict(truck),
        "ledger": dict(ledger),
        "contacts": {
            "forbidden_contact_count": bridge_state["forbidden_contact_count"],
        },
    }
    validate_state(snapshot)
    return snapshot


def validate_cycle_boundary(previous_end: dict, next_start: dict) -> None:
    """Require an exact, no-reset hand-off between two consecutive scoops."""

    validate_state(previous_end)
    validate_state(next_start)
    if next_start["cycle_id"] != previous_end["cycle_id"] + 1:
        raise ValueError("mixed_cycle_boundary_cycle_order")
    if next_start["physics_instance_id"] != previous_end["physics_instance_id"]:
        raise ValueError("mixed_cycle_boundary_physics_instance_changed")
    for key in ("sim_time_s", "physics_step", "mechanism", "mpm", "rocks",
                "bucket", "truck", "ledger", "contacts"):
        if next_start[key] != previous_end[key]:
            raise ValueError(f"mixed_cycle_boundary_state_reset:{key}")
