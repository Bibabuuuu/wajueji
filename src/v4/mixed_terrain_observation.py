"""Deterministic height-grid fusion for MPM surface and rigid rocks."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from mixed_three_cycle_state import SHA256_RE, canonical_sha256


OBSERVATION_SCHEMA = "excavation_mixed_terrain_observation_v1"
MATERIAL_CODES = {"unknown", "mpm", "rigid_rock", "mixed"}


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"mixed_terrain_number_required:{name}")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"mixed_terrain_positive_number_required:{name}")
    return result


def _point(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"mixed_terrain_point_shape:{name}")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"mixed_terrain_nonfinite_point:{name}")
    return result


def _empty_grid(rows: int, columns: int, value: Any) -> list[list[Any]]:
    return [[value() if callable(value) else value for _ in range(columns)]
            for _ in range(rows)]


def build_mixed_terrain_observation(
        *,
        mpm_surface_points_world_m: Sequence[Sequence[float]],
        rock_surface_points_world_m: Mapping[str, Sequence[Sequence[float]]],
        origin_xy_m: Sequence[float],
        resolution_m: float,
        shape_yx: Sequence[int],
        physics_instance_id: str,
        cycle_id: int,
        observation_sequence: int,
        capture_physics_step: int,
        capture_sim_time_s: float,
        source_state_sha256: str,
        source_mpm_state_sha256: str,
        source_rocks_state_sha256: str,
        previous_cycle_end_state_sha256: str | None,
        bucket_mouth_width_m: float,
        _skip_validation: bool = False,
) -> dict:
    """Project current MPM and rigid surfaces into one canonical observation."""

    if (not isinstance(origin_xy_m, (list, tuple)) or len(origin_xy_m) != 2 or
            not all(isinstance(item, (int, float)) and
                    not isinstance(item, bool) and math.isfinite(float(item))
                    for item in origin_xy_m)):
        raise ValueError("mixed_terrain_origin")
    if (not isinstance(shape_yx, (list, tuple)) or len(shape_yx) != 2 or
            any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in shape_yx)):
        raise ValueError("mixed_terrain_shape")
    resolution = _positive_number(resolution_m, "resolution_m")
    mouth_width = _positive_number(bucket_mouth_width_m, "bucket_mouth_width_m")
    if not isinstance(physics_instance_id, str) or not physics_instance_id:
        raise ValueError("mixed_terrain_physics_instance_id")
    for value, name, minimum in (
        (cycle_id, "cycle_id", 1),
        (observation_sequence, "observation_sequence", 1),
        (capture_physics_step, "capture_physics_step", 0),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"mixed_terrain_integer_required:{name}")
    if cycle_id > 3:
        raise ValueError("mixed_terrain_cycle_id")
    if (isinstance(capture_sim_time_s, bool) or
            not isinstance(capture_sim_time_s, (int, float)) or
            not math.isfinite(float(capture_sim_time_s)) or
            float(capture_sim_time_s) < 0.0):
        raise ValueError("mixed_terrain_capture_time")
    if not isinstance(source_state_sha256, str) or not SHA256_RE.fullmatch(
            source_state_sha256):
        raise ValueError("mixed_terrain_source_state_hash")
    if (not isinstance(source_mpm_state_sha256, str) or
            not SHA256_RE.fullmatch(source_mpm_state_sha256) or
            not isinstance(source_rocks_state_sha256, str) or
            not SHA256_RE.fullmatch(source_rocks_state_sha256)):
        raise ValueError("mixed_terrain_source_component_hash")
    if (previous_cycle_end_state_sha256 is not None and
            (not isinstance(previous_cycle_end_state_sha256, str) or
             not SHA256_RE.fullmatch(previous_cycle_end_state_sha256))):
        raise ValueError("mixed_terrain_previous_end_hash")

    normalized_mpm_points = [
        _point(raw_point, f"mpm[{index}]")
        for index, raw_point in enumerate(mpm_surface_points_world_m)
    ]
    if not isinstance(rock_surface_points_world_m, Mapping):
        raise ValueError("mixed_terrain_rock_mapping")
    normalized_rock_points = {
        rock_id: [
            _point(item, f"rocks.{rock_id}[{index}]")
            for index, item in enumerate(rock_surface_points_world_m[rock_id])
        ]
        for rock_id in sorted(rock_surface_points_world_m)
    }
    if any(not isinstance(rock_id, str) or not rock_id
           for rock_id in normalized_rock_points):
        raise ValueError("mixed_terrain_rock_id")

    rows, columns = shape_yx
    origin_x, origin_y = (float(item) for item in origin_xy_m)
    mpm_height = _empty_grid(rows, columns, None)
    rock_height = _empty_grid(rows, columns, None)
    mpm_count = _empty_grid(rows, columns, 0)
    rock_count = _empty_grid(rows, columns, 0)
    rock_ids_by_cell = _empty_grid(rows, columns, list)

    def cell(point: tuple[float, float, float]) -> tuple[int, int] | None:
        column = math.floor((point[0] - origin_x) / resolution)
        row = math.floor((point[1] - origin_y) / resolution)
        if 0 <= row < rows and 0 <= column < columns:
            return row, column
        return None

    for point in normalized_mpm_points:
        location = cell(point)
        if location is None:
            continue
        row, column = location
        current = mpm_height[row][column]
        mpm_height[row][column] = point[2] if current is None else max(current, point[2])
        mpm_count[row][column] += 1

    rock_diameters: dict[str, float] = {}
    rock_passable: dict[str, bool] = {}
    for rock_id, points in normalized_rock_points.items():
        if not points:
            raise ValueError("mixed_terrain_empty_rock_surface")
        spans = [max(point[axis] for point in points) -
                 min(point[axis] for point in points) for axis in range(3)]
        diameter = math.sqrt(sum(span * span for span in spans))
        rock_diameters[rock_id] = diameter
        rock_passable[rock_id] = diameter <= mouth_width
        for point in points:
            location = cell(point)
            if location is None:
                continue
            row, column = location
            current = rock_height[row][column]
            rock_height[row][column] = (
                point[2] if current is None else max(current, point[2]))
            rock_count[row][column] += 1
            if rock_id not in rock_ids_by_cell[row][column]:
                rock_ids_by_cell[row][column].append(rock_id)

    height = _empty_grid(rows, columns, None)
    known = _empty_grid(rows, columns, False)
    material = _empty_grid(rows, columns, "unknown")
    for row in range(rows):
        for column in range(columns):
            mpm_value = mpm_height[row][column]
            rock_value = rock_height[row][column]
            values = [item for item in (mpm_value, rock_value) if item is not None]
            if values:
                height[row][column] = max(values)
                known[row][column] = True
            if mpm_value is not None and rock_value is not None:
                material[row][column] = "mixed"
            elif mpm_value is not None:
                material[row][column] = "mpm"
            elif rock_value is not None:
                material[row][column] = "rigid_rock"
            rock_ids_by_cell[row][column].sort()

    visible_rock_ids = sorted({
        rock_id
        for row in rock_ids_by_cell
        for cell_ids in row
        for rock_id in cell_ids
    })
    observation = {
        "schema": OBSERVATION_SCHEMA,
        "physics_instance_id": physics_instance_id,
        "cycle_id": cycle_id,
        "observation_sequence": observation_sequence,
        "capture_physics_step": capture_physics_step,
        "capture_sim_time_s": float(capture_sim_time_s),
        "source_state_sha256": source_state_sha256,
        "previous_cycle_end_state_sha256": previous_cycle_end_state_sha256,
        "sampling_evidence": {
            "mpm_surface_points_world_m": [list(point)
                                            for point in normalized_mpm_points],
            "rock_surface_points_world_m": {
                rock_id: [list(point) for point in points]
                for rock_id, points in normalized_rock_points.items()
            },
            "bucket_mouth_width_m": mouth_width,
            "source_mpm_state_sha256": source_mpm_state_sha256,
            "source_rocks_state_sha256": source_rocks_state_sha256,
        },
        "grid": {
            "origin_xy_m": [origin_x, origin_y],
            "resolution_m": resolution,
            "shape_yx": [rows, columns],
            "height_m": height,
            "known": known,
            "material": material,
            "mpm_height_m": mpm_height,
            "rock_height_m": rock_height,
            "mpm_point_count": mpm_count,
            "rock_point_count": rock_count,
            "rock_ids_by_cell": rock_ids_by_cell,
        },
        "rocks": {
            "visible_ids": visible_rock_ids,
            "equivalent_diameter_m": {
                rock_id: rock_diameters[rock_id] for rock_id in visible_rock_ids
            },
            "bucket_mouth_passable": {
                rock_id: rock_passable[rock_id] for rock_id in visible_rock_ids
            },
        },
    }
    observation["content_sha256"] = canonical_sha256({
        "grid": observation["grid"],
        "rocks": observation["rocks"],
    })
    observation["observation_sha256"] = canonical_sha256(observation)
    if not _skip_validation:
        validate_mixed_terrain_observation(observation)
    return observation


def validate_mixed_terrain_observation(
        observation: dict, *, minimum_capture_physics_step: int | None = None,
) -> None:
    expected_root = {
        "schema", "physics_instance_id", "cycle_id", "observation_sequence",
        "capture_physics_step", "capture_sim_time_s", "source_state_sha256",
        "previous_cycle_end_state_sha256", "sampling_evidence", "grid", "rocks",
        "content_sha256", "observation_sha256",
    }
    if not isinstance(observation, dict) or set(observation) != expected_root:
        raise ValueError("mixed_terrain_root_keys")
    if observation["schema"] != OBSERVATION_SCHEMA:
        raise ValueError("mixed_terrain_schema")
    if (not isinstance(observation["physics_instance_id"], str) or
            not observation["physics_instance_id"]):
        raise ValueError("mixed_terrain_physics_instance_id")
    for key, minimum in (
        ("cycle_id", 1), ("observation_sequence", 1),
        ("capture_physics_step", 0),
    ):
        value = observation[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"mixed_terrain_integer_required:{key}")
    if observation["cycle_id"] > 3:
        raise ValueError("mixed_terrain_cycle_id")
    if (minimum_capture_physics_step is not None and
            observation["capture_physics_step"] <= minimum_capture_physics_step):
        raise ValueError("mixed_terrain_stale_physics_step")
    if (isinstance(observation["capture_sim_time_s"], bool) or
            not isinstance(observation["capture_sim_time_s"], (int, float)) or
            not math.isfinite(float(observation["capture_sim_time_s"])) or
            float(observation["capture_sim_time_s"]) < 0.0):
        raise ValueError("mixed_terrain_capture_time")
    if not SHA256_RE.fullmatch(str(observation["source_state_sha256"])):
        raise ValueError("mixed_terrain_source_state_hash")
    previous_hash = observation["previous_cycle_end_state_sha256"]
    if previous_hash is not None and not SHA256_RE.fullmatch(str(previous_hash)):
        raise ValueError("mixed_terrain_previous_end_hash")
    sampling = observation["sampling_evidence"]
    if (not isinstance(sampling, dict) or set(sampling) != {
            "mpm_surface_points_world_m", "rock_surface_points_world_m",
            "bucket_mouth_width_m", "source_mpm_state_sha256",
            "source_rocks_state_sha256"} or
            not isinstance(sampling["source_mpm_state_sha256"], str) or
            not SHA256_RE.fullmatch(sampling["source_mpm_state_sha256"]) or
            not isinstance(sampling["source_rocks_state_sha256"], str) or
            not SHA256_RE.fullmatch(sampling["source_rocks_state_sha256"])):
        raise ValueError("mixed_terrain_sampling_evidence")

    grid = observation["grid"]
    expected_grid = {
        "origin_xy_m", "resolution_m", "shape_yx", "height_m", "known",
        "material", "mpm_height_m", "rock_height_m", "mpm_point_count",
        "rock_point_count", "rock_ids_by_cell",
    }
    if not isinstance(grid, dict) or set(grid) != expected_grid:
        raise ValueError("mixed_terrain_grid_keys")
    origin = grid["origin_xy_m"]
    if (not isinstance(origin, list) or len(origin) != 2 or
            any(isinstance(item, bool) or not isinstance(item, (int, float)) or
                not math.isfinite(float(item)) for item in origin)):
        raise ValueError("mixed_terrain_origin")
    rows, columns = grid["shape_yx"]
    if (not isinstance(rows, int) or isinstance(rows, bool) or rows <= 0 or
            not isinstance(columns, int) or isinstance(columns, bool) or columns <= 0):
        raise ValueError("mixed_terrain_shape")
    _positive_number(grid["resolution_m"], "resolution_m")
    for key in (
        "height_m", "known", "material", "mpm_height_m", "rock_height_m",
        "mpm_point_count", "rock_point_count", "rock_ids_by_cell",
    ):
        value = grid[key]
        if (not isinstance(value, list) or len(value) != rows or
                any(not isinstance(row, list) or len(row) != columns
                    for row in value)):
            raise ValueError(f"mixed_terrain_grid_shape:{key}")
    for row in range(rows):
        for column in range(columns):
            material = grid["material"][row][column]
            known = grid["known"][row][column]
            height = grid["height_m"][row][column]
            if material not in MATERIAL_CODES or not isinstance(known, bool):
                raise ValueError("mixed_terrain_cell_contract")
            if (height is not None and
                    (isinstance(height, bool) or not isinstance(height, (int, float)) or
                     not math.isfinite(float(height)))):
                raise ValueError("mixed_terrain_cell_height")
            if known != (height is not None) or (material == "unknown") == known:
                raise ValueError("mixed_terrain_known_material_mismatch")
            layer_values = []
            for layer_key in ("mpm_height_m", "rock_height_m"):
                layer = grid[layer_key][row][column]
                if (layer is not None and
                        (isinstance(layer, bool) or
                         not isinstance(layer, (int, float)) or
                         not math.isfinite(float(layer)))):
                    raise ValueError("mixed_terrain_layer_height")
                layer_values.append(layer)
            counts = []
            for count_key in ("mpm_point_count", "rock_point_count"):
                count = grid[count_key][row][column]
                if (isinstance(count, bool) or not isinstance(count, int) or
                        count < 0):
                    raise ValueError("mixed_terrain_point_count")
                counts.append(count)
            ids = grid["rock_ids_by_cell"][row][column]
            if (not isinstance(ids, list) or ids != sorted(ids) or
                    len(ids) != len(set(ids)) or
                    any(not isinstance(item, str) or not item for item in ids)):
                raise ValueError("mixed_terrain_cell_rock_ids")
            mpm_present = layer_values[0] is not None
            rock_present = layer_values[1] is not None
            if (mpm_present != (counts[0] > 0) or
                    rock_present != (counts[1] > 0) or
                    rock_present != bool(ids)):
                raise ValueError("mixed_terrain_layer_count_mismatch")
            expected_material = (
                "mixed" if mpm_present and rock_present else
                "mpm" if mpm_present else
                "rigid_rock" if rock_present else "unknown")
            expected_height = (
                max(item for item in layer_values if item is not None)
                if mpm_present or rock_present else None)
            if (material != expected_material or known != (expected_height is not None) or
                    height != expected_height):
                raise ValueError("mixed_terrain_derived_cell_mismatch")

    rocks = observation["rocks"]
    if (not isinstance(rocks, dict) or set(rocks) != {
            "visible_ids", "equivalent_diameter_m", "bucket_mouth_passable"}):
        raise ValueError("mixed_terrain_rocks_keys")
    visible = rocks["visible_ids"]
    if (not isinstance(visible, list) or visible != sorted(visible) or
            len(visible) != len(set(visible))):
        raise ValueError("mixed_terrain_visible_rock_ids")
    if (set(rocks["equivalent_diameter_m"]) != set(visible) or
            set(rocks["bucket_mouth_passable"]) != set(visible)):
        raise ValueError("mixed_terrain_rock_metadata_mismatch")
    cell_visible = sorted({
        rock_id
        for row in grid["rock_ids_by_cell"]
        for ids in row
        for rock_id in ids
    })
    if visible != cell_visible:
        raise ValueError("mixed_terrain_visible_rock_cell_mismatch")
    for rock_id in visible:
        diameter = rocks["equivalent_diameter_m"][rock_id]
        if (isinstance(diameter, bool) or not isinstance(diameter, (int, float)) or
                not math.isfinite(float(diameter)) or float(diameter) < 0.0 or
                not isinstance(rocks["bucket_mouth_passable"][rock_id], bool)):
            raise ValueError("mixed_terrain_rock_metadata")

    if (not isinstance(observation["content_sha256"], str) or
            observation["content_sha256"] != canonical_sha256({
                "grid": grid, "rocks": rocks})):
        raise ValueError("mixed_terrain_content_hash_mismatch")

    claimed = observation["observation_sha256"]
    payload = dict(observation)
    del payload["observation_sha256"]
    if not isinstance(claimed, str) or claimed != canonical_sha256(payload):
        raise ValueError("mixed_terrain_observation_hash_mismatch")
    rebuilt = build_mixed_terrain_observation(
        mpm_surface_points_world_m=sampling["mpm_surface_points_world_m"],
        rock_surface_points_world_m=sampling["rock_surface_points_world_m"],
        origin_xy_m=grid["origin_xy_m"],
        resolution_m=grid["resolution_m"],
        shape_yx=grid["shape_yx"],
        physics_instance_id=observation["physics_instance_id"],
        cycle_id=observation["cycle_id"],
        observation_sequence=observation["observation_sequence"],
        capture_physics_step=observation["capture_physics_step"],
        capture_sim_time_s=observation["capture_sim_time_s"],
        source_state_sha256=observation["source_state_sha256"],
        source_mpm_state_sha256=sampling["source_mpm_state_sha256"],
        source_rocks_state_sha256=sampling["source_rocks_state_sha256"],
        previous_cycle_end_state_sha256=previous_hash,
        bucket_mouth_width_m=sampling["bucket_mouth_width_m"],
        _skip_validation=True,
    )
    if rebuilt != observation:
        raise ValueError("mixed_terrain_sampling_reconstruction_mismatch")
