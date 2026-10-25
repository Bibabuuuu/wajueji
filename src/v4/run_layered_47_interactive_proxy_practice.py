#!/usr/bin/env python3
"""Keyboard/gamepad practice on the recoverable non-formal 47-rock proxy.

This entry deliberately writes diagnostic journals rather than an episode.
The source proxy remains read-only and its rejected diagnostic ancestry means
that neither the input stream nor the resulting physics is training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import signal
import sys
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

from build_mixed_material_capacity_preflight import sha256
from layered_excavation_approach_planner import COMMAND_LIMITS_M_S, _read_obj
from layered_mound_checkpoint import checkpoint_identity, restore_fixture_prefix
from layered_mound_checkpoint_runtime import (
    RestoredRuntimeBridge,
    build_model_and_bridge,
    records_at_qpos,
)
import layered_47_interactive_proxy_checkpoint as proxy_checkpoint
from manual_action_input import (
    GamepadFourAxisState,
    KeyboardFourAxisState,
    ManualInputSample,
    load_manual_input_config,
)
from mpm_rigid_free_velocity import predict_free_linear_velocity
import run_layered_47_excavation_preflight as preflight
from run_layered_47_excavation_approach import _bridge_step
import run_layered_47_interactive_proxy_resume_audit as resume_audit
from run_layered_mound_single_rock_press_smoke import _finite_tree
from run_layered_mound_three_rock_adaptive_press import RuntimeBridge
from run_mixed_material_r3_a_probe import _make_transform, _write_json
from soil_surface import SoilSurface


CONFIG_SCHEMA = "excavation_layered_47_interactive_proxy_practice_config_v1"
REPORT_SCHEMA = "excavation_layered_47_interactive_proxy_practice_report_v1"
PASSED_STATUS = "layered_47_interactive_proxy_practice_completed"
FAILED_STATUS = "layered_47_interactive_proxy_practice_failed"
AXIS_NAMES = ("swing", "boom", "arm", "bucket")


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(path: str | Path, repository_root: str | Path) -> tuple[dict, dict]:
    root = Path(repository_root).resolve()
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if (config.get("schema") != CONFIG_SCHEMA or config.get("status") !=
            "hardware_practice_only_no_training_data"):
        raise ValueError("interactive_proxy_practice_config_schema_or_status")
    audit_binding = config.get("resume_audit_config", {})
    if set(audit_binding) != {"path", "sha256"}:
        raise ValueError("interactive_proxy_practice_audit_binding_fields")
    audit_path = _resolve(root, audit_binding["path"])
    if (not audit_path.is_file()
            or sha256(audit_path) != audit_binding["sha256"]):
        raise ValueError("interactive_proxy_practice_audit_binding")
    audit_config, loaded = resume_audit.load_config(audit_path, root)
    manual_binding = config.get("manual_input_config", {})
    if set(manual_binding) != {"path", "sha256"}:
        raise ValueError("interactive_proxy_practice_manual_binding_fields")
    manual_path = _resolve(root, manual_binding["path"])
    if (not manual_path.is_file()
            or sha256(manual_path) != manual_binding["sha256"]):
        raise ValueError("interactive_proxy_practice_manual_binding")
    manual_config = load_manual_input_config(manual_path)
    expected_control = {
        "axis_order": list(AXIS_NAMES),
        "command_limits_si": COMMAND_LIMITS_M_S.tolist(),
        "command_scale": 1.0,
        "start_paused": True,
        "maximum_wall_seconds": 1800.0,
        "maximum_simulation_seconds": 60.0,
    }
    if config.get("control") != expected_control:
        raise ValueError("interactive_proxy_practice_control")
    display = config.get("display", {})
    if (set(display) != {
            "backend_root", "width", "height", "soil_render_mode",
            "surface_voxel_m", "surface_update_interval_frames",
            "camera_position_m", "camera_pitch_deg", "camera_yaw_deg"}
            or not Path(display["backend_root"]).is_dir()
            or display["width"] != 1440 or display["height"] != 900
            or display["soil_render_mode"] != "gpu_density_surface_with_loose_particles"
            or display["surface_voxel_m"] != 0.04
            or display["surface_update_interval_frames"] != 1
            or len(display["camera_position_m"]) != 3):
        raise ValueError("interactive_proxy_practice_display")
    expected_scope = {
        "practice_only": True,
        "physics_enabled": True,
        "checkpoint_written": False,
        "formal_a30_equivalent": False,
        "manual_collection_eligible": False,
        "training_eligible": False,
        "writes_training_episode": False,
    }
    if config.get("scope") != expected_scope:
        raise ValueError("interactive_proxy_practice_scope")
    return config, {
        "audit_path": audit_path,
        "audit_config": audit_config,
        "manual_path": manual_path,
        "manual_config": manual_config,
        **loaded,
    }


def normalized_to_command(action, control: dict) -> np.ndarray:
    values = np.asarray(action, dtype=np.float64)
    if (values.shape != (4,) or not np.isfinite(values).all()
            or np.any(np.abs(values) > 1.0 + 1e-12)):
        raise ValueError("interactive_proxy_practice_action")
    limits = np.asarray(control["command_limits_si"], dtype=np.float64)
    return np.clip(values, -1.0, 1.0) * limits * float(
        control["command_scale"])


def _append(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False))
        stream.write("\n")
        stream.flush()


def _controlled_frame(fixture, bridge, base: dict, command: np.ndarray,
                      frame: int, *, bridge_step=None) -> dict:
    dt = float(base["probe"]["output_dt_s"])
    gravity = np.asarray(base["mpm"]["gravity_m_s2"], dtype=np.float64)
    started = time.perf_counter()
    prior = bridge.state
    rock_origins = {
        rock["rock_id"]: np.asarray(rock["position"], dtype=np.float64)
        for rock in prior["rocks"]}
    origins = {**rock_origins, "bucket": np.asarray(
        prior["bucket_origin"], dtype=np.float64)}
    transforms = [
        _make_transform(rock["qpos"][:3], rock["qpos"][3:7])
        for rock in prior["rocks"]] + [preflight._bucket_transform(prior)]
    velocities = [wp.spatial_vector(
        *[float(value) for value in predict_free_linear_velocity(
            np.asarray(rock["qvel"][:3], dtype=np.float64), gravity, dt)],
        *[float(value) for value in rock[
            "angular_velocity_world_rad_s"]]) for rock in prior["rocks"]]
    velocities.append(wp.spatial_vector(
        *[float(value) for value in prior[
            "bucket_linear_velocity_world_m_s"]],
        *[float(value) for value in prior[
            "bucket_angular_velocity_world_rad_s"]]))
    fixture.state.body_q.assign(wp.array(
        transforms, dtype=wp.transform, device=fixture.device))
    fixture.state.body_qd.assign(wp.array(
        velocities, dtype=wp.spatial_vector, device=fixture.device))
    assigned = time.perf_counter()
    measured, particles = fixture.step(dt, origins)
    after_mpm = time.perf_counter()
    after = (bridge_step or _bridge_step)(
        bridge, dt, int(base["probe"]["mujoco_internal_substeps"]),
        command, measured)
    finished = time.perf_counter()
    contacts = after.get("contact_substeps", {})
    rock_linear = max(float(np.linalg.norm(rock["qvel"][:3]))
                      for rock in after["rocks"])
    rock_angular = max(float(np.linalg.norm(rock["qvel"][3:]))
                       for rock in after["rocks"])
    row = {
        "frame": frame,
        "sim_time_s": float(after["time_s"]),
        "normalized_action": (
            command / np.asarray(COMMAND_LIMITS_M_S)).tolist(),
        "physical_command_si": command.tolist(),
        "wall_seconds": finished - started,
        "assignment_wall_seconds": assigned - started,
        "mpm_wall_seconds": after_mpm - assigned,
        "mujoco_wall_seconds": finished - after_mpm,
        "finite": bool(particles["finite"] and _finite_tree(after)),
        "maximum_rock_linear_speed_m_s": rock_linear,
        "maximum_rock_angular_speed_rad_s": rock_angular,
        "bucket_mpm_contact_samples": int(
            particles["contact_samples_by_body"]["bucket"]),
        "rock_bucket_contact_substeps": int(contacts.get("rock_bucket", 0)),
        "external_rigid_support_substeps": sum(int(
            contacts.get(name, 0)) for name in (
                "rock_ground", "rock_ramp", "rock_truck")),
        "forbidden_contact_count": int(after.get("forbidden_contact_count", 0)),
        "closure_m": float(after.get("closure_m", float("inf"))),
        **particles,
    }
    row["safe"] = bool(
        row["finite"] and row["forbidden_contact_count"] == 0
        and row["external_rigid_support_substeps"] == 0
        and row["unmapped_active_collider_sample_count"] == 0
        and row["nonfinite_collider_sample_count"] == 0
        and row["closure_m"] <= 0.001)
    return row


def _select_controller(mode, controllers, index):
    if mode == "keyboard":
        return None
    if index < 0:
        raise ValueError("invalid_gamepad_index")
    if mode == "gamepad":
        if index >= len(controllers):
            raise RuntimeError("requested_gamepad_not_found")
        return controllers[index]
    if mode == "auto":
        if not controllers:
            return None
        if len(controllers) != 1:
            raise RuntimeError("ambiguous_gamepad_auto_selection")
        return controllers[0]
    raise ValueError("unsupported_manual_input_mode")


def _display_runtime(display: dict):
    backend = Path(display["backend_root"])
    if str(backend) not in sys.path:
        sys.path.insert(0, str(backend))
    import common
    import bank_runtime
    from moving_scene import to_pose
    assembly, _, _, _, _, _, _, _ = common.setup()
    model, _, _, _, names = bank_runtime.build(
        backend.parent / "small-soil-20260910-v17-performance" / "scene",
        backend.parent / "manual-comparison-20260908-v1",
        particle_subdivision=2)
    return assembly, model, model.state(), names, to_pose


def _prepare_rock_visuals(viewer, records, root: Path, device: str) -> None:
    for prototype in sorted({record["prototype_id"] for record in records}):
        vertices, faces = _read_obj(
            root / "assets/v4/mixed-material-r0" / f"{prototype}.obj")
        viewer.log_mesh(
            f"/practice/prototype/{prototype}",
            wp.array(vertices, dtype=wp.vec3, device=device),
            wp.array(faces.reshape(-1), dtype=wp.int32, device=device),
            color=(0.40, 0.37, 0.32), roughness=0.92, metallic=0.0)


def _draw(viewer, display_state, fixture, bridge_state, records,
          assembly, names, to_pose, device: str, display: dict, frame: int,
          soil_surface: SoilSurface):
    poses = assembly.assemble(
        np.asarray(bridge_state["mechanism_qpos"], dtype=np.float64),
        include_meshes=False)[1]
    if np.linalg.norm(np.asarray(bridge_state["bucket_origin"])
                      - poses["bucket"][:3, 3]) > 1e-7:
        raise RuntimeError("interactive_proxy_practice_display_alignment")
    display_state.body_q.assign(wp.array(
        [to_pose(poses[name]) for name in names], dtype=wp.transform,
        device=device))
    viewer.begin_frame(float(bridge_state["time_s"]))
    viewer.log_state(display_state)
    update_token = frame // int(display["surface_update_interval_frames"])
    soil_surface.update(fixture.state.particle_q, update_token)
    soil_surface.draw(viewer, fixture.state.particle_q)
    for prototype in sorted({record["prototype_id"] for record in records}):
        indices = [index for index, record in enumerate(records)
                   if record["prototype_id"] == prototype]
        xforms = [_make_transform(
            bridge_state["rocks"][index]["qpos"][:3],
            bridge_state["rocks"][index]["qpos"][3:7]) for index in indices]
        scales = [wp.vec3(float(records[index]["equivalent_diameter_m"]))
                  for index in indices]
        viewer.log_instances(
            f"/practice/rocks/{prototype}",
            f"/practice/prototype/{prototype}",
            wp.array(xforms, dtype=wp.transform, device=device),
            wp.array(scales, dtype=wp.vec3, device=device),
            wp.full(
                len(indices), wp.vec3(0.40, 0.37, 0.32),
                dtype=wp.vec3, device=device),
            wp.full(
                len(indices), wp.vec4(0.92, 0.0, 0.0, 1.0),
                dtype=wp.vec4, device=device))
    viewer.end_frame()


def run(args) -> dict:
    started = time.perf_counter()
    root = Path(args.repository_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    input_log = output / "practice-input.ndjson"
    physics_log = output / "practice-physics.ndjson"
    config, loaded = load_config(config_path, root)
    source = loaded["audit_config"]["input_checkpoint"]
    checkpoint = _resolve(root, source["path"])
    metadata_path = _resolve(root, source["metadata"])
    arrays, source_metadata, before = (
        proxy_checkpoint.load_checkpoint_readonly(checkpoint, metadata_path))
    base = loaded["base"]
    profile_name = loaded["audit_config"]["benchmark_config"]["profile"]
    solver_profile = loaded["profiles"][profile_name]
    manifest_path = _resolve(root, base["physical_geometry"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [rock for rock in manifest["rocks"]
               if rock["layer_id"] in preflight.LAYERS]
    _, proxy = preflight._load_bound_json(
        root, base["open_bucket_proxy"], "open_bucket_proxy")

    newton.use_coord_layout_targets = True
    wp.init()
    wp.set_device(args.device)
    rows: list[dict] = []
    stop_reason = "probe_completed" if args.probe_frames else None
    device_info = {
        "mode_requested": args.manual_input,
        "mode_actual": "headless_probe" if args.probe_frames else None,
        "name": None, "index": None,
    }
    restore = None
    bridge = None
    viewer = None
    controller = None
    with tempfile.TemporaryDirectory(
            prefix="layered-47-interactive-practice-") as raw:
        work = Path(raw)
        restored_records = records_at_qpos(records, arrays["rigid_qpos"])
        model, binary, _, _, _ = build_model_and_bridge(
            args, root, root / base["capacity_bridge"]["config_path"],
            restored_records, work, "interactive-practice")
        default_bridge = RuntimeBridge(binary, model, len(records))
        default_bridge.close()
        bridge = RestoredRuntimeBridge(
            binary, model, len(records), arrays["mechanism_qpos"].tolist(),
            arrays["mechanism_qvel"].tolist(), arrays["rigid_qpos"],
            arrays["rigid_qvel"])
        fixture = preflight.ExcavationFixture(
            base, root, records, arrays["grid_reference_particle_q"], proxy,
            bridge.state, args.device, solver_profile=solver_profile)
        particle_restore = restore_fixture_prefix(fixture, arrays)
        fixture.solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        fixture.assign(bridge.state)
        solver_restore = proxy_checkpoint.restore_solver_continuation(
            fixture, arrays)
        restore = {
            "particle_fields_exact": all(
                particle_restore["field_readback_exact"].values()),
            "grid_exact": all(solver_restore["grid_readback_exact"].values()),
            "warmstart_exact": all(solver_restore[
                "solver_field_readback_exact"].values()),
        }
        if args.probe_frames:
            for frame in range(args.probe_frames):
                row = _controlled_frame(
                    fixture, bridge, base, np.zeros(4), frame)
                rows.append(row)
                _append(physics_log, row)
                if not row["safe"]:
                    stop_reason = "physics_fault"
                    break
        else:
            import pyglet
            import pyglet.display.xlib as pyglet_xlib
            from pyglet.window import key
            if not hasattr(pyglet_xlib.XlibScreen, "is_primary"):
                pyglet_xlib.XlibScreen.is_primary = False
            assembly, display_model, display_state, names, to_pose = (
                _display_runtime(config["display"]))
            viewer = newton.viewer.ViewerGL(
                width=int(config["display"]["width"]),
                height=int(config["display"]["height"]), vsync=False,
                paused=bool(config["control"]["start_paused"]))
            viewer.set_model(display_model)
            viewer.picking_enabled = False
            viewer.renderer.set_title(
                "47-ROCK PROXY PRACTICE - NOT TRAINING DATA")
            viewer.set_camera(
                wp.vec3(*config["display"]["camera_position_m"]),
                pitch=float(config["display"]["camera_pitch_deg"]),
                yaw=float(config["display"]["camera_yaw_deg"]))
            soil_surface = SoilSurface(
                fixture.state.particle_q,
                particle_spacing_m=float(base["mpm"]["particle_spacing_m"]),
                voxel_size_m=float(config["display"]["surface_voxel_m"]))
            _prepare_rock_visuals(viewer, records, root, args.device)
            keyboard = KeyboardFourAxisState()
            manual_state = keyboard
            focus = {"active": True}
            manager = None
            if args.manual_input in ("gamepad", "auto"):
                manager = pyglet.input.ControllerManager()
                controllers = manager.get_controllers()
                controller = _select_controller(
                    args.manual_input, controllers, args.gamepad_index)
                if controller is not None:
                    controller.open()
                    gamepad = GamepadFourAxisState(loaded["manual_config"])
                    manual_state = gamepad
                    device_info.update(
                        mode_actual="gamepad", name=controller.name,
                        index=(args.gamepad_index if args.manual_input ==
                               "gamepad" else 0))

                    def stick_motion(selected, stick, vector):
                        if selected is controller:
                            gamepad.set_stick(stick, vector.x, vector.y)

                    def button_press(selected, button):
                        nonlocal stop_reason
                        if selected is controller:
                            gamepad.press_button(button)
                            if gamepad.emergency_stop_latched:
                                stop_reason = "manual_emergency_stop"

                    def disconnected(selected):
                        nonlocal stop_reason
                        if selected is controller:
                            gamepad.disconnect()
                            stop_reason = "gamepad_disconnect"

                    controller.push_handlers(
                        on_stick_motion=stick_motion,
                        on_button_press=button_press)
                    manager.push_handlers(on_disconnect=disconnected)
                else:
                    device_info["mode_actual"] = "keyboard"
            else:
                device_info["mode_actual"] = "keyboard"

            motion_keys = {
                key.J: "J", key.L: "L", key.I: "I", key.K: "K",
                key.U: "U", key.O: "O", key.N: "N", key.M: "M"}

            def press(symbol, _modifiers):
                nonlocal stop_reason
                if symbol in motion_keys:
                    keyboard.press(motion_keys[symbol])
                if symbol in (key.X, key.ESCAPE):
                    stop_reason = "user_finished"
                if symbol == key.SPACE:
                    keyboard.clear()

            def release(symbol, _modifiers):
                if symbol in motion_keys:
                    keyboard.release(motion_keys[symbol])

            def activated():
                focus["active"] = True

            def deactivated():
                focus["active"] = False
                keyboard.clear()

            viewer.renderer.window.push_handlers(
                on_key_press=press, on_key_release=release,
                on_activate=activated, on_deactivate=deactivated)
            latest = {"action": [0.0] * 4, "fps": None, "row": None}

            def ui(imgui):
                imgui.text("PRACTICE ONLY - NOT TRAINING DATA")
                imgui.text("Space: run/pause | X/Esc: finish | Back/View: E-stop")
                imgui.text("Gamepad: left=(swing,boom), right=(bucket,arm)")
                imgui.text("Keyboard: J/L I/K U/O N/M")
                imgui.text(f"Input: {device_info['mode_actual']} {device_info['name'] or ''}")
                imgui.text("Action: " + str([
                    round(value, 3) for value in latest["action"]]))
                if latest["fps"] is not None:
                    imgui.text(f"Physics FPS: {latest['fps']:.2f}")
                if latest["row"] is not None:
                    imgui.text(
                        "bucket soil/rock contacts: %d / %d" % (
                            latest["row"]["bucket_mpm_contact_samples"],
                            latest["row"]["rock_bucket_contact_substeps"]))

            viewer.register_ui_callback(ui, position="side")

            def signal_stop(*_):
                nonlocal stop_reason
                stop_reason = "signal_requested"

            signal.signal(signal.SIGTERM, signal_stop)
            while viewer.is_running() and stop_reason is None:
                elapsed = time.perf_counter() - started
                if elapsed >= float(config["control"]["maximum_wall_seconds"]):
                    stop_reason = "wall_budget"
                    break
                if (float(bridge.state["time_s"])
                        - float(arrays["simulation_time_s"][0]) >= float(
                            config["control"]["maximum_simulation_seconds"])):
                    stop_reason = "simulation_budget"
                    break
                if viewer.should_step():
                    action = manual_state.action(window_focused=focus["active"])
                    command = normalized_to_command(action, config["control"])
                    input_row = ManualInputSample(
                        episode_id="practice-only",
                        decision_step=len(rows),
                        sim_time_s=float(bridge.state["time_s"]),
                        sample_wall_monotonic_s=time.monotonic(),
                        acquisition_domain="local_47_rejected_ancestry_practice_only",
                        input_device=device_info["mode_actual"],
                        device_name=device_info["name"],
                        device_index=device_info["index"],
                        config_sha256=sha256(loaded["manual_path"]),
                        window_focused=focus["active"],
                        connected=getattr(manual_state, "connected", True),
                        raw_controls=manual_state.raw_controls(),
                        normalized_before_projection=tuple(action),
                        emergency_stop_latched=getattr(
                            manual_state, "emergency_stop_latched", False),
                    ).as_dict()
                    _append(input_log, {
                        "practice_only": True, "training_eligible": False,
                        "sample": input_row,
                    })
                    row = _controlled_frame(
                        fixture, bridge, base, command, len(rows))
                    rows.append(row)
                    _append(physics_log, row)
                    latest.update(
                        action=list(action),
                        fps=(1.0 / row["wall_seconds"]
                             if row["wall_seconds"] else None), row=row)
                    if not row["safe"]:
                        stop_reason = "physics_fault"
                _draw(
                    viewer, display_state, fixture, bridge.state, records,
                    assembly, names, to_pose, args.device, config["display"],
                    len(rows), soil_surface)
            if stop_reason is None:
                stop_reason = "window_closed"
            if viewer is not None:
                viewer.close()
                viewer = None
            if controller is not None:
                controller.close()
                controller = None
        bridge.close(kill=not rows or not rows[-1]["finite"])
        bridge = None

    after = checkpoint_identity(checkpoint)
    checks = {
        "source_checkpoint_sha256_verified": before["sha256"] == source["sha256"],
        "source_checkpoint_metadata_verified": (
            source_metadata["checkpoint_sha256"] == before["sha256"]),
        "source_checkpoint_unchanged": after == before,
        "particle_fields_restored_exactly": restore["particle_fields_exact"],
        "solver_grid_restored_exactly": restore["grid_exact"],
        "solver_warmstart_restored_exactly": restore["warmstart_exact"],
        "all_executed_states_finite": bool(rows) and all(
            row["finite"] for row in rows),
        "no_checkpoint_written": True,
        "training_data_not_written": True,
        "manual_collection_not_granted": True,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "status": PASSED_STATUS if all(checks.values()) else FAILED_STATUS,
        "checks": checks,
        "stop_reason": stop_reason,
        "device": device_info,
        "metrics": {
            "executed_frames": len(rows),
            "mean_physics_fps": (1.0 / float(np.mean([
                row["wall_seconds"] for row in rows])) if rows else None),
            "maximum_bucket_mpm_contact_samples": max((
                row["bucket_mpm_contact_samples"] for row in rows), default=0),
            "maximum_rock_bucket_contact_substeps": max((
                row["rock_bucket_contact_substeps"] for row in rows), default=0),
        },
        "source_checkpoint": before,
        "source_checkpoint_after": after,
        "logs": {
            "input": str(input_log), "physics": str(physics_log),
            "training_episode": None,
        },
        "inputs": {
            "config_sha256": sha256(config_path),
            "resume_audit_config_sha256": sha256(loaded["audit_path"]),
            "manual_input_config_sha256": sha256(loaded["manual_path"]),
            "soil_surface_source_sha256": sha256(
                Path(__file__).with_name("soil_surface.py")),
            "fixture_source_sha256": sha256(Path(preflight.__file__)),
        },
        "runtime_baseline": {
            "soil_rendering": {
                "mode": config["display"]["soil_render_mode"],
                "surface_voxel_m": config["display"]["surface_voxel_m"],
                "surface_update_interval_frames": config["display"][
                    "surface_update_interval_frames"],
                "display_only": True,
            },
            "postsolve_projection": dict(
                preflight.POSTSOLVE_PROJECTION_POLICY),
        },
        "environment": {
            "python": platform.python_version(), "newton": newton.__version__,
            "warp": wp.__version__, "numpy": np.__version__,
            "device": str(wp.get_device()),
        },
        "execution": {"wall_seconds": time.perf_counter() - started},
        "scope": config["scope"],
        "next_gate": "operator_assess_latency_without_admitting_training_data",
    }
    _write_json(output / "report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manual-input", choices=("keyboard", "gamepad", "auto"),
                        default="keyboard")
    parser.add_argument("--gamepad-index", type=int, default=0)
    parser.add_argument("--probe-frames", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--mujoco-include", required=True)
    parser.add_argument("--mujoco-library-dir", required=True)
    parser.add_argument("--mechanism-header-dir", required=True)
    parser.add_argument("--controller-header-dir", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    output = Path(args.output).resolve()
    try:
        if args.probe_frames < 0 or args.probe_frames > 120:
            raise ValueError("invalid_probe_frames")
        report = run(args)
    except (Exception, KeyboardInterrupt) as exc:
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": REPORT_SCHEMA, "status": FAILED_STATUS,
            "failure": {
                "type": type(exc).__name__, "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "scope": {
                "practice_only": True, "manual_collection_eligible": False,
                "training_eligible": False, "checkpoint_written": False,
            },
            "next_gate": "fix_practice_wiring_without_using_partial_data",
        }
        _write_json(output / "failure.json", failure)
        print(json.dumps({
            "status": FAILED_STATUS, "failure_type": type(exc).__name__,
            "failure_message": str(exc), "output": str(output),
        }, sort_keys=True), flush=True)
        return 1
    print(json.dumps({
        "status": report["status"], "checks": report["checks"],
        "stop_reason": report["stop_reason"], "device": report["device"],
        "metrics": report["metrics"], "output": str(output),
    }, sort_keys=True, allow_nan=False), flush=True)
    return 0 if report["status"] == PASSED_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
