"""Acquire one human-operated demonstration in the qualified fast-tier runtime.

The keyboard drives the same normalized four-axis action contract used by BC
and SAC.  In ordinary manual mode the predictive candidates remain shadow
diagnostics; the confirmed bucket end-stop projector may suppress only an
ineffective positive bucket command.  With a policy checkpoint, the DAgger
mode pauses on predictive risk and waits for an explicit human takeover.  No
acquisition episode enters critic replay or performs an optimizer update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

from contracts import ACTION_SCHEMA, AXES, OBSERVATION_SCHEMA_V3
from dagger_supervisor import (
    BucketEndStopProjector, ClosedBucketBoomPauseSupervisor,
    CurlReversePauseSupervisor, HumanTakeoverHandshake, TakeoverSupervisor,
    augment_observation_v3, safety_state as make_safety_state,
    task_phase_from_termination_state,
)
from mpm_episode import _write_json, advance_termination
from manual_action_input import (
    GamepadFourAxisState, KeyboardFourAxisState, ManualInputSample,
    load_manual_input_config,
)
from mpm_smoke import LiveFastTierBackend, _read, actor_observation
from runtime_adapter import POLICY_DT_S, RESET_SCHEMA, FourAxisRuntime, ResetRequest, step_request
from termination import (FILL_SAMPLE_COUNT,
                         LOAD_CORRECTION_REENTRY_CONFIRM_TICKS,
                         MIN_BUCKET_CLOSURE_NORMALIZED,
                         SingleScoopTermination, classify_fast_tier_window)


SCHEMA = "excavation_four_axis_manual_demo_v1"
PHASES = ("approach", "dig", "lift", "hold", "turn", "dump")


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_line(stream, value) -> None:
    stream.write(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")


def _shadow_row(*, decision, frame, next_frame, action, phase) -> dict:
    clearance_before = float(frame["backend_state"]["bucket_min_z_m"])
    clearance_after = float(next_frame["backend_state"]["bucket_min_z_m"])
    descent_rate = max(0.0, (clearance_before - clearance_after) / POLICY_DT_S)
    ttc_s = clearance_after / descent_rate if descent_rate > 1e-9 else None
    return {
        "schema": "excavation_actor_risk_shadow_v1",
        "decision_step": decision,
        "sim_time_s": float(next_frame["sim_time_s"]),
        "phase": phase,
        "action": list(action),
        "bucket_min_z_m": clearance_after,
        "bucket_descent_rate_mps": descent_rate,
        "predicted_ground_ttc_s": ttc_s,
        "candidate_flags": {
            "clearance_lt_0p10": clearance_after < 0.10,
            "clearance_lt_0p50": clearance_after < 0.50,
            "descending": descent_rate > 1e-6,
            "ttc_lt_2p0": ttc_s is not None and ttc_s < 2.0,
            "negative_boom_and_positive_bucket": action[1] < 0.0 and action[3] > 0.0,
        },
        "intervention": False,
    }


def _select_control_action(*, dagger_mode, control_mode, actor_action,
                           human_action, takeover_handshake,
                           human_source="human_keyboard"):
    """Select one action and always define whether it is the handshake first."""
    if dagger_mode and control_mode == "actor":
        return list(actor_action), "actor_prefix", False
    if dagger_mode:
        action, first = takeover_handshake.action_for_step(human_action)
        return action, "human_correction", first
    return list(human_action), human_source, False


def _select_gamepad_controller(*, mode, controllers, index):
    """Resolve a controller only after the caller has created a display."""
    if mode == "keyboard":
        return None
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
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


def _requires_controller_enumeration(mode):
    if mode == "keyboard":
        return False
    if mode in ("gamepad", "auto"):
        return True
    raise ValueError("unsupported_manual_input_mode")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-report", type=Path, required=True)
    parser.add_argument("--empty-report", type=Path, required=True)
    parser.add_argument("--fast-reference-report", type=Path, required=True)
    parser.add_argument("--fill-root", type=Path, required=True)
    parser.add_argument("--fill-calibration-report", type=Path, required=True)
    parser.add_argument("--fill-held-out-report", type=Path, required=True)
    parser.add_argument("--episode-id", default="human-demo-001")
    parser.add_argument("--session-label")
    parser.add_argument("--manual-input", choices=("keyboard", "gamepad", "auto"),
                        default="keyboard")
    parser.add_argument("--manual-input-config", type=Path)
    parser.add_argument("--gamepad-index", type=int, default=0)
    parser.add_argument("--acquisition-domain", default="unspecified")
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--policy-site-packages", type=Path)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument(
        "--policy-observation-version", choices=("v1", "v3", "v4"), default="v1")
    parser.add_argument("--policy-seed", type=int, default=20260917)
    parser.add_argument("--takeover-ttc-s", type=float, default=2.0)
    parser.add_argument("--takeover-clearance-m", type=float, default=0.10)
    parser.add_argument("--max-wall", type=float, default=3600.0)
    parser.add_argument("--max-sim", type=float, default=60.0)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError("output_directory_exists:" + str(args.output))
    if not math.isfinite(args.max_wall) or args.max_wall <= 0.0:
        raise ValueError("invalid_max_wall")
    if not math.isfinite(args.max_sim) or not 0.0 < args.max_sim <= 60.0:
        raise ValueError("invalid_max_sim")
    if args.session_label is not None:
        args.session_label = args.session_label.strip()
        if not args.session_label or len(args.session_label) > 80:
            raise ValueError("invalid_session_label")
    if not args.acquisition_domain.strip() or len(args.acquisition_domain) > 120:
        raise ValueError("invalid_acquisition_domain")
    args.acquisition_domain = args.acquisition_domain.strip()
    if args.gamepad_index < 0:
        raise ValueError("invalid_gamepad_index")
    if args.manual_input in ("gamepad", "auto") and args.manual_input_config is None:
        raise ValueError("manual_input_config_required")
    manual_input_config = None
    manual_input_config_path = None
    manual_input_config_sha256 = None
    if args.manual_input_config is not None:
        manual_input_config_path = args.manual_input_config.resolve()
        if not manual_input_config_path.is_file():
            raise FileNotFoundError("manual_input_config_missing:" + str(manual_input_config_path))
        manual_input_config = load_manual_input_config(manual_input_config_path)
        manual_input_config_sha256 = sha256(manual_input_config_path)
    if args.extra_site_packages is not None:
        extra_site_packages = args.extra_site_packages.resolve()
        if not extra_site_packages.is_dir():
            raise ValueError("extra_site_packages_not_directory")
        # Append, do not prepend: the qualified physics environment keeps its
        # own Newton/Warp/Pyglet, while missing analysis dependencies such as
        # SciPy can be resolved from the shared permanent environment.
        sys.path.append(str(extra_site_packages))
    if args.policy_checkpoint is not None:
        if not args.policy_checkpoint.is_file():
            raise FileNotFoundError(
                "policy_checkpoint_missing:" + str(args.policy_checkpoint))
        if args.policy_site_packages is None:
            raise ValueError("policy_site_packages_required_for_dagger")
        policy_site_packages = args.policy_site_packages.resolve()
        if not policy_site_packages.is_dir():
            raise ValueError("policy_site_packages_not_directory")
        sys.path.append(str(policy_site_packages))
    elif args.policy_site_packages is not None:
        raise ValueError("policy_site_packages_without_checkpoint")
    if args.policy_checkpoint is None and args.policy_observation_version != "v1":
        raise ValueError("policy_observation_version_without_checkpoint")
    if (not math.isfinite(args.takeover_ttc_s) or args.takeover_ttc_s <= 0.0
            or not math.isfinite(args.takeover_clearance_m)
            or args.takeover_clearance_m <= 0.0):
        raise ValueError("invalid_takeover_threshold")

    source_root = Path(__file__).resolve().parent
    source_names = [
        "mpm_manual_demo.py", "mpm_smoke.py", "mpm_episode.py",
        "online_truth.py", "runtime_adapter.py", "contracts.py", "termination.py",
        "dagger_supervisor.py", "observation_vector_v2.py", "manual_action_input.py",
    ]
    if args.policy_checkpoint is not None:
        source_names.extend(("observation_vector.py",))
        if args.policy_observation_version == "v3":
            source_names.extend((
                "v3_actor_policy.py", "observation_vector_v2.py",
                "observation_vector_v3.py"))
        elif args.policy_observation_version == "v4":
            source_names.extend((
                "v4_actor_policy.py", "observation_vector_v2.py",
                "observation_vector_v3.py", "observation_vector_v4.py"))
        else:
            source_names.extend(("sac_agent.py",))
    source_paths = tuple(source_root / name for name in source_names)
    if manual_input_config_path is not None:
        source_paths = source_paths + (manual_input_config_path,)
    sources_before = {str(path): sha256(path) for path in source_paths}
    geometry = _read(args.geometry_report)
    target = geometry["selection"]["entry_xy_m"] + [geometry["selection"]["surface_z_m"]]
    reset_value = {
        "schema": RESET_SCHEMA,
        "episode_id": args.episode_id,
        "target_world_m": target,
    }
    ResetRequest.parse(reset_value)

    made = []

    def factory(request):
        backend = LiveFastTierBackend(
            request,
            backend_root=args.backend_root,
            output=args.output,
            geometry_report=args.geometry_report,
            empty_report=args.empty_report,
            fast_reference_report=args.fast_reference_report,
            fill_root=args.fill_root,
            fill_calibration_report=args.fill_calibration_report,
            fill_held_out_report=args.fill_held_out_report,
        )
        made.append(backend)
        return backend

    runtime = FourAxisRuntime(factory)
    viewer = None
    surface = None
    streams = []
    manual_input_actual_mode = "keyboard"
    manual_input_device_name = None
    manual_input_device_index = None
    gamepad_disconnect_count = 0
    gamepad_emergency_stop_triggered = False
    started = time.perf_counter()
    executed = 0
    phase = "approach"
    stop = {"value": False, "reason": "window_closed"}
    failure = None
    initial_truth = None
    final_truth = None
    last_truth = None
    termination = SingleScoopTermination()
    last_termination = None
    evaluation_samples = []
    completed_evaluation_windows = []
    current_evaluation_hold_s = 0.0
    peak_force_n = 0.0
    dagger_mode = args.policy_checkpoint is not None
    protected_v4_mode = dagger_mode and args.policy_observation_version == "v4"
    policy_actor = None
    control_mode = "actor" if dagger_mode else "human"
    safety_pause = False
    takeover_events = []
    takeover_handshake = HumanTakeoverHandshake()
    correction_samples = 0
    actor_prefix_steps = 0
    end_stop_projector = BucketEndStopProjector()
    takeover_supervisor = TakeoverSupervisor(
        ttc_threshold_s=args.takeover_ttc_s,
        clearance_threshold_m=args.takeover_clearance_m,
    )
    closed_bucket_boom_supervisor = ClosedBucketBoomPauseSupervisor()
    curl_reverse_supervisor = CurlReversePauseSupervisor()
    current_safety_state = None
    current_task_phase = "digging"
    shadow_counts = {
        "clearance_lt_0p10": 0,
        "clearance_lt_0p50": 0,
        "descending": 0,
        "ttc_lt_2p0": 0,
        "negative_boom_and_positive_bucket": 0,
    }
    status = {
        "schema": "excavation_four_axis_manual_demo_status_v1",
        "status": "initializing",
        "episode_id": args.episode_id,
        "pid": os.getpid(),
        "task_ready": False,
        "acquisition_mode": (
            "protected_v4_baseline_evaluation" if protected_v4_mode else
            "dagger_takeover" if dagger_mode else "manual_demo"),
        "control_mode": control_mode,
        "safety_pause": False,
        "training_started": False,
        "actor_parameter_updates": 0,
        "critic_parameter_updates": 0,
        "sac_parameter_updates": 0,
    }

    def save_status(frame=None):
        status.update(
            wall_s=time.perf_counter() - started,
            executed_decisions=executed,
            phase=phase,
            failure=failure,
            control_mode=control_mode,
            safety_pause=safety_pause,
            end_stop_projection_count=end_stop_projector.projection_count,
            curl_reverse_candidate_ticks=(
                curl_reverse_supervisor.candidate_ticks),
            dagger_correction_samples=correction_samples,
            takeover_handshake_state=(
                takeover_handshake.state if dagger_mode else None),
            first_human_action_buffered=(
                takeover_handshake.pending_first_action is not None
                if dagger_mode else None),
        )
        if frame is not None:
            status.update(
                sim_time_s=float(frame["sim_time_s"]),
                bucket_min_z_m=float(frame["backend_state"]["bucket_min_z_m"]),
                force_norm_n=float(np.linalg.norm(made[0].force)) if made else 0.0,
                mouth_angle_deg=(
                    last_truth.get("mouth_angle_deg") if last_truth is not None else None),
                evaluation_hold_sim_s=current_evaluation_hold_s,
                evaluation_sample_count=len(evaluation_samples),
            )
        if args.output.exists():
            _write_json(args.output / "status.json", status)

    try:
        frame = runtime.reset(reset_value)
        backend = made[0]
        if backend.collider_model is None:
            raise RuntimeError("interactive_renderer_model_missing")

        current_safety_state = make_safety_state(
            bucket_clearance_m=frame["backend_state"]["bucket_min_z_m"],
            previous_bucket_clearance_m=None,
            dt_s=POLICY_DT_S,
            bucket_end_stop_confirmed=False,
        )
        if dagger_mode:
            if args.policy_observation_version == "v3":
                from v3_actor_policy import LoadedV3ActorPolicy
                policy_actor = LoadedV3ActorPolicy(
                    checkpoint_path=args.policy_checkpoint)
            elif args.policy_observation_version == "v4":
                from v4_actor_policy import LoadedV4ActorPolicy
                policy_actor = LoadedV4ActorPolicy(
                    checkpoint_path=args.policy_checkpoint)
            else:
                from sac_agent import LoadedSacPolicy
                policy_actor = LoadedSacPolicy(
                    checkpoint_path=args.policy_checkpoint,
                    exploration_std=0.0,
                    seed=args.policy_seed,
                )

        initial_truth = backend.termination_inputs(require_fill=True)
        last_truth = initial_truth
        _write_json(args.output / "initial-truth.json", initial_truth)

        policy_log = (args.output / "policy-steps.ndjson").open("w", encoding="utf-8", newline="\n")
        truth_log = (args.output / "truth-steps.ndjson").open("w", encoding="utf-8", newline="\n")
        terrain_log = (args.output / "terrain-frames.ndjson").open("w", encoding="utf-8", newline="\n")
        shadow_log = (args.output / "shadow-risk.ndjson").open("w", encoding="utf-8", newline="\n")
        phase_log = (args.output / "phase-events.ndjson").open("w", encoding="utf-8", newline="\n")
        evaluation_log = (args.output / "manual-fill-samples.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        manual_input_log = (args.output / "manual-input.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        manual_input_event_log = (args.output / "manual-input-events.ndjson").open(
            "w", encoding="utf-8", newline="\n")
        streams = [
            policy_log, truth_log, terrain_log, shadow_log, phase_log, evaluation_log,
            manual_input_log, manual_input_event_log]
        correction_log = None
        intervention_log = None
        handshake_log = None
        if dagger_mode:
            correction_log = (args.output / "dagger-corrections.ndjson").open(
                "w", encoding="utf-8", newline="\n")
            intervention_log = (args.output / "intervention-events.ndjson").open(
                "w", encoding="utf-8", newline="\n")
            handshake_log = (args.output / "takeover-handshake.ndjson").open(
                "w", encoding="utf-8", newline="\n")
            streams.extend((correction_log, intervention_log, handshake_log))

        import newton
        import warp as wp
        import pyglet.display.xlib as pyglet_xlib

        # Pyglet 2.1.16 may fall back to a plain XlibScreen when XRandR is
        # present but reports no outputs in a remote desktop session.  Its
        # default-screen path still reads ``is_primary`` unconditionally.
        # Supply the missing fallback value in this process only.
        if not hasattr(pyglet_xlib.XlibScreen, "is_primary"):
            pyglet_xlib.XlibScreen.is_primary = False
        from pyglet.window import key

        try:
            from soil_surface import SoilSurface
        except ModuleNotFoundError as error:
            if error.name != "scipy":
                raise
            SoilSurface = None

        viewer = newton.viewer.ViewerGL(width=1440, height=900, vsync=False, paused=True)
        viewer.set_model(backend.collider_model)
        viewer.picking_enabled = False
        viewer.renderer.set_title(
            args.session_label
            if args.session_label is not None else
            "V4 BASELINE PROTECTED ACTOR / HUMAN TAKEOVER"
            if protected_v4_mode else
            "V3 HIGH-FILL PROTECTED ACTOR / HUMAN TAKEOVER"
            if dagger_mode and args.policy_observation_version == "v3"
            else "RL HUMAN DEMO - current 20Hz/120Hz/60Hz fast tier"
        )
        viewer.set_camera(wp.vec3(11, -12, 8), pitch=-23, yaw=125)
        surface = (SoilSurface(backend.soil.state.particle_q)
                   if SoilSurface is not None else None)
        surface_enabled = {"value": surface is not None}
        keyboard_input = KeyboardFourAxisState()
        gamepad_input = None
        manual_input_state = keyboard_input
        manual_gamepad = None
        gamepad_manager = None
        focus = {"active": True}
        motion_keys = {
            key.J: "J", key.L: "L", key.I: "I", key.K: "K",
            key.U: "U", key.O: "O", key.N: "N", key.M: "M",
        }

        def human_action_from_held():
            return manual_input_state.action(window_focused=focus["active"])

        def stop_signal(*_):
            stop.update(value=True, reason="signal_requested")

        signal.signal(signal.SIGTERM, stop_signal)

        def activated():
            focus["active"] = True

        def deactivated():
            focus["active"] = False
            keyboard_input.clear()

        def activate_human_correction_if_armed():
            nonlocal control_mode, safety_pause
            if not (dagger_mode and control_mode == "human_correction_armed"):
                return
            first_action = human_action_from_held()
            if not any(abs(value) > 0.0 for value in first_action):
                return
            activation_event = takeover_handshake.activate(
                raw_action=first_action,
                decision_step=executed,
                sim_time_s=frame["sim_time_s"],
            )
            if activation_event is not None:
                _write_line(handshake_log, activation_event)
                control_mode = "human_correction"
                safety_pause = False
                # ViewerGL exposes pause through the same private field its
                # own Space-key GUI handler uses.  Clear it here so the first
                # keyboard or gamepad motion command resumes safely.
                viewer._paused = False
                save_status(frame)

        def press(symbol, modifiers):
            nonlocal phase, control_mode, safety_pause
            del modifiers
            if symbol in motion_keys:
                focus["active"] = True
                keyboard_input.press(motion_keys[symbol])
            if (dagger_mode and symbol == key.ENTER
                    and control_mode == "takeover_pending"):
                keyboard_input.clear()
                armed_event = takeover_handshake.arm(
                    decision_step=executed,
                    sim_time_s=frame["sim_time_s"],
                )
                _write_line(handshake_log, armed_event)
                control_mode = "human_correction_armed"
                safety_pause = True
                save_status(frame)
                return
            if (dagger_mode and symbol in motion_keys
                    and control_mode == "human_correction_armed"):
                activate_human_correction_if_armed()
            if symbol == key.SPACE:
                keyboard_input.clear()
            if symbol == key.T and surface is not None:
                surface_enabled["value"] = not surface_enabled["value"]
            if symbol == key.X:
                stop.update(value=True, reason="user_finished")
            for phase_key, name in zip((key._1, key._2, key._3, key._4, key._5, key._6), PHASES):
                if symbol == phase_key:
                    phase = name
                    _write_line(phase_log, {
                        "decision_step": executed,
                        "sim_time_s": float(frame["sim_time_s"]),
                        "phase": phase,
                    })

        def release(symbol, modifiers):
            del modifiers
            if symbol in motion_keys:
                keyboard_input.release(motion_keys[symbol])

        viewer.renderer.window.push_handlers(
            on_activate=activated,
            on_deactivate=deactivated,
            on_key_press=press,
            on_key_release=release,
        )

        # Import and enumerate controllers only after ViewerGL created a display.
        # The default keyboard route never reaches this branch, preserving the
        # headless A30 fallback without a controller dependency.
        if _requires_controller_enumeration(args.manual_input):
            import pyglet.input

            gamepad_manager = pyglet.input.ControllerManager()
            controllers = gamepad_manager.get_controllers()
            manual_gamepad = _select_gamepad_controller(
                mode=args.manual_input,
                controllers=controllers,
                index=args.gamepad_index,
            )
            if manual_gamepad is not None:
                manual_gamepad.open()
                gamepad_input = GamepadFourAxisState(manual_input_config)
                manual_input_state = gamepad_input
                manual_input_actual_mode = "gamepad"
                manual_input_device_name = manual_gamepad.name
                manual_input_device_index = (
                    args.gamepad_index if args.manual_input == "gamepad" else 0)

                def gamepad_stick_motion(controller, stick, vector):
                    if controller is manual_gamepad:
                        gamepad_input.set_stick(stick, vector.x, vector.y)
                        activate_human_correction_if_armed()

                def gamepad_button_press(controller, button):
                    nonlocal gamepad_emergency_stop_triggered
                    if controller is manual_gamepad:
                        gamepad_input.press_button(button)
                        if gamepad_input.emergency_stop_latched:
                            gamepad_emergency_stop_triggered = True
                            _write_line(manual_input_event_log, {
                                "schema": "excavation_manual_input_event_v1",
                                "event": "manual_emergency_stop",
                                "decision_step": executed,
                                "sim_time_s": float(frame["sim_time_s"]),
                            })
                            stop.update(value=True, reason="manual_emergency_stop")

                def gamepad_disconnect(controller):
                    nonlocal gamepad_disconnect_count
                    if controller is manual_gamepad:
                        gamepad_input.disconnect()
                        gamepad_disconnect_count += 1
                        _write_line(manual_input_event_log, {
                            "schema": "excavation_manual_input_event_v1",
                            "event": "gamepad_disconnect",
                            "decision_step": executed,
                            "sim_time_s": float(frame["sim_time_s"]),
                        })

                manual_gamepad.push_handlers(
                    on_stick_motion=gamepad_stick_motion,
                    on_button_press=gamepad_button_press,
                )
                gamepad_manager.push_handlers(on_disconnect=gamepad_disconnect)

        def ui(imgui):
            imgui.text(
                args.session_label
                if args.session_label is not None else
                "V4 BASELINE PROTECTED / HUMAN TAKEOVER"
                if protected_v4_mode else
                "V3 HIGH-FILL PROTECTED / HUMAN TAKEOVER"
                if dagger_mode and args.policy_observation_version == "v3" else
                "CURRENT RL / DAgger TAKEOVER" if dagger_mode else
                "CURRENT RL FAST-TIER / HUMAN DEMO")
            imgui.text("Space: start/pause | X: save and finish")
            imgui.text("J/L swing -/+ | I/K boom +/-")
            imgui.text("U/O arm -/+ | N/M bucket -/+")
            imgui.text(
                "Manual input: %s (requested %s)" % (
                    manual_input_actual_mode, args.manual_input))
            if manual_input_actual_mode == "gamepad":
                imgui.text("Gamepad BACK: emergency stop and save")
            if dagger_mode:
                imgui.text(
                    "Actor drives first; ENTER arms takeover, first motion key resumes")
            imgui.text("1 approach | 2 dig | 3 lift | 4 hold | 5 turn | 6 dump")
            imgui.text("T: surface/particles | WASDQE: camera only")
            if surface is None:
                imgui.text("Particle display active (surface renderer unavailable)")
            imgui.separator()
            imgui.text(f"Phase: {phase} | Step: {executed}")
            imgui.text(f"Control: {control_mode}")
            if safety_pause:
                if control_mode == "takeover_pending":
                    imgui.text_wrapped(
                        "SAFETY PAUSE: physics is frozen. Press ENTER to arm "
                        "human correction; ENTER does not advance physics.")
                elif control_mode == "human_correction_armed":
                    imgui.text_wrapped(
                        "TAKEOVER ARMED: physics is still frozen. The first "
                        "J/L, I/K, U/O or N/M command resumes from this exact state.")
            imgui.text(f"Sim: {frame['sim_time_s']:.2f}/{args.max_sim:.0f} s")
            imgui.text(f"Bucket clearance: {frame['backend_state']['bucket_min_z_m']:.3f} m")
            if current_safety_state is not None:
                imgui.text(
                    "Descent: %.3f m/s | TTC: %s" % (
                        current_safety_state["bucket_descent_rate_mps"],
                        ("%.2f s" % current_safety_state["predicted_ground_ttc_s"]
                         if current_safety_state["predicted_ground_ttc_valid"]
                         else "inactive"),
                    ))
                imgui.text(
                    "Bucket end stop: %s | projected commands: %d" % (
                        "confirmed" if end_stop_projector.confirmed else "not confirmed",
                        end_stop_projector.projection_count,
                    ))
            imgui.text(f"MPM force: {np.linalg.norm(backend.force)/1000:.2f} kN")
            mouth_angle = (last_truth.get("mouth_angle_deg")
                           if last_truth is not None else None)
            if mouth_angle is not None:
                imgui.text(f"Mouth angle: {mouth_angle:.2f} deg | diagnostic only")
            closure = actor_observation(frame, target)[
                "proprioception"]["cylinder_position_normalized"][2]
            imgui.text(
                f"Bucket closure: {closure * 100.0:.1f}% | trigger >= "
                f"{MIN_BUCKET_CLOSURE_NORMALIZED * 100.0:.0f}%")
            state = last_termination.state if last_termination is not None else "DIGGING"
            imgui.text(f"Ending state: {state} | off-soil confirm 0.50 sim s")
            if last_termination is not None and state == "LOAD_CORRECTION":
                imgui.text(
                    "Load-correction soil re-entry: %d/3 consecutive ticks" %
                    last_termination.load_correction_reentry_ticks)
            imgui.text(
                f"Detached evaluation hold: {current_evaluation_hold_s:.2f}/3.00 sim s")
            if evaluation_samples:
                latest = evaluation_samples[-1]
                if latest["fill_percent"] is None:
                    imgui.text(
                        f"Latest 2cm fill: invalid | samples {len(evaluation_samples)}/13")
                else:
                    imgui.text(
                        f"Latest 2cm fill: {latest['fill_percent']:.2f}% | "
                        f"samples {len(evaluation_samples)}/13")
            if completed_evaluation_windows:
                latest_window = completed_evaluation_windows[-1]
                label = latest_window["fast_tier_classification"]
                imgui.text(
                    f"3s window: {label} | min {latest_window['minimum_fill_percent']:.2f}%")
            else:
                imgui.text("Press X after a sealed 13-sample result is shown")
            if dagger_mode:
                imgui.text(
                    "Intervention episode: actor prefix is never autonomous success")
                imgui.text(
                    "Human corrections -> actor BC only; no critic/SAC replay")
            else:
                imgui.text("Risk monitor is shadow-only; end-stop projection is active")
                imgui.text("No BC/SAC parameter update; no critic replay write")
            if failure:
                imgui.text_wrapped("STOPPED: " + failure)

        viewer.register_ui_callback(ui, position="side")

        def evaluation_ui(imgui):
            imgui.set_next_window_pos(
                imgui.ImVec2(320, 10), imgui.Cond_.first_use_ever)
            imgui.set_next_window_size(
                imgui.ImVec2(520, 285), imgui.Cond_.first_use_ever)
            expanded = imgui.begin(
                "MANUAL DEMO STATUS - ENDING CHECK",
                None,
                imgui.WindowFlags_.no_focus_on_appearing,
            )[0]
            if expanded:
                mouth_angle = (last_truth.get("mouth_angle_deg")
                               if last_truth is not None else None)
                closure = actor_observation(frame, target)[
                    "proprioception"]["cylinder_position_normalized"][2]
                imgui.text("ENDING TRIGGER: dug + detached 0.50 s + bucket near closed")
                imgui.text(
                    f"Bucket closure: {closure * 100.0:.1f}% / "
                    f"{MIN_BUCKET_CLOSURE_NORMALIZED * 100.0:.0f}%")
                if mouth_angle is None:
                    imgui.text("Current mouth angle: unavailable")
                else:
                    imgui.text(f"Mouth angle (diagnostic only): {mouth_angle:.2f} deg")
                imgui.separator()
                imgui.text(
                    f"Hold time (SIMULATION): {current_evaluation_hold_s:.2f} / 3.00 s")
                imgui.text(f"2 cm samples: {len(evaluation_samples)} / 13")
                if (last_termination is not None
                        and last_termination.state == "LOAD_CORRECTION"):
                    correction_samples_ui = last_termination.load_correction_samples
                    if correction_samples_ui:
                        imgui.text(
                            "Pre-window fill: %.2f%% / 95%%; formal count stays 0/13" %
                            (correction_samples_ui[-1][2] * 100.0))
                    imgui.text(
                        "Soil re-entry confirm: %d/3 consecutive ticks" %
                        last_termination.load_correction_reentry_ticks)
                elif (last_termination is not None
                      and last_termination.terminated
                      and not last_termination.success):
                    imgui.text_wrapped(
                        "SEALED FAILURE: " + str(last_termination.reason))
                if evaluation_samples:
                    latest = evaluation_samples[-1]
                    if latest["fill_percent"] is None:
                        imgui.text("Latest 2 cm fill: INVALID")
                    else:
                        imgui.text(f"Latest 2 cm fill: {latest['fill_percent']:.2f}%")
                else:
                    imgui.text("Latest 2 cm fill: waiting for shared ending trigger")
                if completed_evaluation_windows:
                    latest_window = completed_evaluation_windows[-1]
                    label = latest_window["fast_tier_classification"]
                    imgui.separator()
                    imgui.text(
                        f"13-SAMPLE RESULT: {label} | "
                        f"minimum {latest_window['minimum_fill_percent']:.2f}%")
                else:
                    imgui.separator()
                    imgui.text("DO NOT PRESS X until a sealed 13-sample result appears")
                    imgui.text("Space pauses; the timer uses simulation time, not wall time")
            imgui.end()

        viewer.register_ui_callback(evaluation_ui, position="free")

        if dagger_mode:
            def dagger_ui(imgui):
                imgui.set_next_window_pos(
                    imgui.ImVec2(850, 10), imgui.Cond_.first_use_ever)
                imgui.set_next_window_size(
                    imgui.ImVec2(540, 260), imgui.Cond_.first_use_ever)
                expanded = imgui.begin(
                    "DAGGER SAFETY / HUMAN TAKEOVER",
                    None,
                    imgui.WindowFlags_.no_focus_on_appearing,
                )[0]
                if expanded:
                    imgui.text(f"Control mode: {control_mode}")
                    imgui.text(f"Task phase: {current_task_phase}")
                    imgui.text(
                        f"Clearance: {current_safety_state['bucket_clearance_m']:.3f} m")
                    imgui.text(
                        f"Descent: {current_safety_state['bucket_descent_rate_mps']:.3f} m/s")
                    if current_safety_state["predicted_ground_ttc_valid"]:
                        imgui.text(
                            "Predicted ground TTC: %.2f s / pause below %.2f s" % (
                                current_safety_state["predicted_ground_ttc_s"],
                                args.takeover_ttc_s,
                            ))
                    else:
                        imgui.text("Predicted ground TTC: inactive")
                    imgui.text(
                        "Bucket end stop: %s | projected commands: %d" % (
                            "confirmed" if end_stop_projector.confirmed else
                            "not confirmed",
                            end_stop_projector.projection_count,
                        ))
                    imgui.separator()
                    if control_mode == "takeover_pending":
                        imgui.text_wrapped(
                            "SAFETY PAUSE ACTIVE. Physics is frozen. Press ENTER "
                            "once to arm human correction; no zero-action step "
                            "will be executed.")
                    elif control_mode == "human_correction_armed":
                        imgui.text_wrapped(
                            "TAKEOVER ARMED. Physics remains frozen until the "
                            "first non-zero J/L, I/K, U/O or N/M command. That "
                            "command is recorded as the first expert label.")
                    elif control_mode == "actor":
                        imgui.text_wrapped(
                            "Actor is driving. Milestone routing blocks premature "
                            "lift; closed-bucket boom-down, two-tick reverse curl, "
                            "or predictive risk pauses the full action before physics.")
                    else:
                        imgui.text_wrapped(
                            "Human correction is active. Finish normally and press "
                            "X only when you want to save the correction episode.")
                    imgui.text_wrapped(
                        "A sealed success with zero takeover can count toward the "
                        "baseline gate; this run never enters critic replay."
                        if protected_v4_mode else
                        "This run cannot become autonomous success or critic replay.")
                    if (protected_v4_mode and policy_actor is not None
                            and policy_actor.last_inference is not None):
                        inference = policy_actor.last_inference
                        imgui.text(
                            "V4 classifier: %s | milestone route: %s | "
                            "confirm %d/2 | history %d/3" % (
                                inference["raw_phase"],
                                inference["routed_phase"],
                                inference["advance_confirmation_ticks"],
                                inference["history_valid_count"],
                            ))
                        imgui.text(
                            "Reverse-curl confirm: %d/2" %
                            curl_reverse_supervisor.candidate_ticks)
                imgui.end()

            viewer.register_ui_callback(dagger_ui, position="free")
        last_terrain_sequence = -1
        last_status_wall = 0.0
        rendered_frames = 0
        status.update(status="paused_ready_for_user", task_ready=True)
        save_status(frame)

        while viewer.is_running() and not stop["value"]:
            now = time.perf_counter()
            if now - started >= args.max_wall:
                stop.update(value=True, reason="wall_budget")
                break
            if frame["sim_time_s"] >= args.max_sim - 1e-9:
                stop.update(value=True, reason="simulation_budget")
                break
            if not safety_pause and viewer.should_step() and failure is None:
                human_action = human_action_from_held()
                legacy_observation = actor_observation(frame, target)
                observation = (augment_observation_v3(
                    legacy_observation, current_safety_state,
                    current_task_phase)
                    if dagger_mode else legacy_observation)
                actor_action = None
                if dagger_mode and control_mode == "actor":
                    actor_action = policy_actor.action(
                        observation if args.policy_observation_version in ("v3", "v4")
                        else legacy_observation)
                raw_action, action_source, takeover_first_action = (
                    _select_control_action(
                        dagger_mode=dagger_mode,
                        control_mode=control_mode,
                        actor_action=actor_action,
                        human_action=human_action,
                        takeover_handshake=takeover_handshake,
                        human_source=(
                            "human_gamepad" if manual_input_actual_mode == "gamepad"
                            else "human_keyboard"),
                    ))
                manual_action_before_projection = (
                    list(human_action) if action_source == "actor_prefix"
                    else list(raw_action))
                action, end_stop = end_stop_projector.project(
                    raw_action, legacy_observation, record_projection=False)
                if dagger_mode and control_mode == "actor":
                    event = closed_bucket_boom_supervisor.evaluate(
                        raw_action=raw_action,
                        end_stop=end_stop,
                        decision_step=executed,
                        sim_time_s=frame["sim_time_s"],
                        safety_state=current_safety_state,
                    )
                    if event is None and protected_v4_mode:
                        inference = policy_actor.last_inference
                        event = curl_reverse_supervisor.evaluate(
                            raw_action=raw_action,
                            curl_started=bool(inference["curl_started"]),
                            decision_step=executed,
                            sim_time_s=frame["sim_time_s"],
                            routed_phase=inference["routed_phase"],
                            safety_state=current_safety_state,
                        )
                    if event is not None and not takeover_events:
                        takeover_events.append(event)
                        _write_line(intervention_log, event)
                        takeover_handshake.request(
                            decision_step=executed,
                            sim_time_s=frame["sim_time_s"],
                        )
                        control_mode = "takeover_pending"
                        safety_pause = True
                        keyboard_input.clear()
                        save_status(frame)
                        continue
                try:
                    result = runtime.step(step_request(frame, action))
                    if takeover_first_action:
                        takeover_handshake.mark_first_action_executed()
                    if end_stop["bucket_action_projected"]:
                        end_stop_projector.projection_count += 1
                    if dagger_mode and action_source == "actor_prefix":
                        actor_prefix_steps += 1
                    next_frame = result["observation"]
                    next_legacy_observation = actor_observation(next_frame, target)
                    next_safety_state = make_safety_state(
                        bucket_clearance_m=next_frame["backend_state"][
                            "bucket_min_z_m"],
                        previous_bucket_clearance_m=frame["backend_state"][
                            "bucket_min_z_m"],
                        dt_s=POLICY_DT_S,
                        bucket_end_stop_confirmed=end_stop_projector.confirmed,
                    )
                    last_truth = backend.termination_inputs()
                    previous_sample_count = len(termination.samples)
                    bucket_closure = next_legacy_observation[
                        "proprioception"]["cylinder_position_normalized"][2]
                    last_truth, last_termination, fill_required = advance_termination(
                        termination,
                        float(next_frame["sim_time_s"]),
                        last_truth,
                        bucket_closure,
                        lambda: backend.termination_inputs(require_fill=True),
                    )
                    next_task_phase = task_phase_from_termination_state(
                        last_termination.state)
                    next_observation = (augment_observation_v3(
                        next_legacy_observation, next_safety_state,
                        next_task_phase)
                        if dagger_mode else next_legacy_observation)
                    if len(termination.samples) > previous_sample_count:
                        index, hold_sim_s, fill_fraction = termination.samples[-1]
                        sample = {
                            "schema": "excavation_manual_fill_sample_v2",
                            "window_index": 0,
                            "sample_index": index,
                            "decision_step": executed + 1,
                            "sim_time_s": float(next_frame["sim_time_s"]),
                            "hold_sim_s": hold_sim_s,
                            "mouth_angle_deg": float(last_truth["mouth_angle_deg"]),
                            "bucket_closure_normalized": float(bucket_closure),
                            "valid": True,
                            "fill_fraction": fill_fraction,
                            "fill_percent": fill_fraction * 100.0,
                        }
                        evaluation_samples.append(sample)
                        _write_line(evaluation_log, sample)
                    current_evaluation_hold_s = (
                        0.0 if termination.evaluation_start_s is None else
                        max(0.0, float(next_frame["sim_time_s"])
                            - termination.evaluation_start_s))
                    if (last_termination.terminated
                            and len(termination.samples) == FILL_SAMPLE_COUNT
                            and not completed_evaluation_windows):
                        values = [item[2] for item in termination.samples]
                        completed_evaluation_windows.append({
                            "window_index": 0,
                            "start_sim_time_s": termination.evaluation_start_s,
                            "end_sim_time_s": float(last_termination.sim_time_s),
                            "sample_count": len(values),
                            "all_samples_valid": True,
                            "minimum_fill_percent": min(values) * 100.0,
                            "maximum_fill_percent": max(values) * 100.0,
                            "all_samples_ge_95": min(values) >= 0.95,
                            "fast_tier_classification": classify_fast_tier_window(values),
                            "formal_reference_qualified": False,
                            "result_sealed": True,
                        })
                    force_n = float(np.linalg.norm(backend.force))
                    peak_force_n = max(peak_force_n, force_n)
                    shadow = _shadow_row(
                        decision=executed,
                        frame=frame,
                        next_frame=next_frame,
                        action=action,
                        phase=phase,
                    )
                    for name, active in shadow["candidate_flags"].items():
                        shadow_counts[name] += int(active)
                    policy_row = {
                        "decision_step": executed,
                        "observation": observation,
                        "raw_action": {
                            "schema": ACTION_SCHEMA,
                            "axis_order": list(AXES),
                            "normalized_velocity": list(raw_action),
                        },
                        "action": {
                            "schema": ACTION_SCHEMA,
                            "axis_order": list(AXES),
                            "normalized_velocity": action,
                        },
                        "action_source": action_source,
                        "policy_diagnostics": (
                            policy_actor.last_inference
                            if protected_v4_mode and action_source == "actor_prefix"
                            else None),
                        "end_stop_projection": end_stop,
                        "phase": phase,
                        "plan": result["plan"],
                        "next_observation": next_observation,
                    }
                    _write_line(policy_log, policy_row)
                    if action_source != "actor_prefix" and (
                            policy_row["raw_action"]["normalized_velocity"]
                            != manual_action_before_projection):
                        raise RuntimeError("manual_input_policy_raw_action_mismatch")
                    _write_line(manual_input_log, ManualInputSample(
                        episode_id=args.episode_id,
                        decision_step=executed,
                        sim_time_s=float(frame["sim_time_s"]),
                        sample_wall_monotonic_s=time.perf_counter(),
                        acquisition_domain=args.acquisition_domain,
                        input_device=manual_input_actual_mode,
                        device_name=manual_input_device_name,
                        device_index=manual_input_device_index,
                        config_sha256=manual_input_config_sha256,
                        window_focused=focus["active"],
                        connected=(
                            gamepad_input.connected if gamepad_input is not None else True),
                        raw_controls=manual_input_state.raw_controls(),
                        normalized_before_projection=tuple(manual_action_before_projection),
                        emergency_stop_latched=(
                            gamepad_input.emergency_stop_latched
                            if gamepad_input is not None else False),
                    ).as_dict())
                    if dagger_mode and action_source == "human_correction":
                        correction_samples += 1
                        _write_line(correction_log, {
                            "schema": "excavation_dagger_correction_v1",
                            "episode_id": args.episode_id,
                            "decision_step": executed,
                            "observation": observation,
                            "raw_human_action": policy_row["raw_action"],
                            "applied_action": policy_row["action"],
                            "end_stop_projection": end_stop,
                            "takeover_first_nonzero_action": takeover_first_action,
                            "takeover_handshake_event": (
                                takeover_handshake.activation_event
                                if takeover_first_action else None),
                            "critic_eligible": False,
                            "autonomous_success_eligible": False,
                        })
                    _write_line(truth_log, {
                        "decision_step": executed + 1,
                        "sim_time_s": float(next_frame["sim_time_s"]),
                        "force_norm_n": force_n,
                        "fill_sample_required": fill_required,
                        "truth": last_truth,
                        "termination": last_termination.as_dict(),
                    })
                    _write_line(shadow_log, shadow)
                    if dagger_mode and control_mode == "actor":
                        event = takeover_supervisor.evaluate(
                            state=next_safety_state,
                            digging_interaction_observed=next_legacy_observation[
                                "digging_interaction_observed"],
                            decision_step=executed + 1,
                            sim_time_s=next_frame["sim_time_s"],
                        )
                        if event is not None and not takeover_events:
                            takeover_events.append(event)
                            _write_line(intervention_log, event)
                            takeover_handshake.request(
                                decision_step=executed + 1,
                                sim_time_s=next_frame["sim_time_s"],
                            )
                            control_mode = "takeover_pending"
                            safety_pause = True
                            keyboard_input.clear()
                    if next_frame["terrain_sequence"] != last_terrain_sequence:
                        _write_line(terrain_log, {
                            "terrain_sequence": next_frame["terrain_sequence"],
                            "capture_sim_s": next_frame["terrain_capture_sim_s"],
                            "terrain": next_frame["terrain"],
                        })
                        last_terrain_sequence = next_frame["terrain_sequence"]
                    executed += 1
                    frame = next_frame
                    current_safety_state = next_safety_state
                    current_task_phase = next_task_phase
                    if executed % 20 == 0:
                        for stream in streams:
                            stream.flush()
                except BaseException as error:
                    failure = repr(error)
                    stop.update(value=True, reason="physics_fault")

            viewer.begin_frame(float(frame["sim_time_s"]))
            viewer.log_state(backend.soil.state)
            if surface is not None and surface_enabled["value"]:
                surface.update(backend.soil.state.particle_q, backend.soil.steps)
                surface.draw(viewer, backend.soil.state.particle_q, backend.soil.colors)
                obj = viewer.objects.get(viewer._qualify("/soil_surface"))
                if obj is not None:
                    obj.hidden = False
            else:
                obj = viewer.objects.get(viewer._qualify("/soil_surface"))
                if obj is not None:
                    obj.hidden = True
                viewer.log_points(
                    "/soil", backend.soil.state.particle_q,
                    backend.soil.radius, backend.soil.colors)
            viewer.end_frame()
            rendered_frames += 1
            if rendered_frames == 30:
                pixels = viewer.get_frame(render_ui=True).numpy()
                with (args.output / "ready-with-ui.ppm").open("wb") as stream:
                    stream.write(
                        f"P6\n{pixels.shape[1]} {pixels.shape[0]}\n255\n".encode())
                    stream.write(pixels.tobytes())
            if now - last_status_wall >= 0.5:
                status["status"] = (
                    "takeover_armed_waiting_for_nonzero_action"
                    if control_mode == "human_correction_armed" else
                    "safety_paused_waiting_for_enter" if safety_pause else
                    "paused" if viewer.is_paused() else "running")
                save_status(frame)
                last_status_wall = now
            time.sleep(0.005)

        if not viewer.is_running() and not stop["value"]:
            stop.update(value=True, reason="window_closed")
        for stream in streams:
            stream.flush()
        try:
            final_truth = backend.termination_inputs(require_fill=True)
            _write_json(args.output / "final-truth.json", final_truth)
        except BaseException as error:
            failure = failure or ("final_truth:" + repr(error))
    except BaseException as error:
        failure = repr(error)
        stop.update(value=True, reason="exception")
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except BaseException as error:
                failure = failure or ("viewer_close:" + repr(error))
        for stream in streams:
            try:
                stream.close()
            except BaseException:
                pass
        try:
            runtime.close()
        except BaseException as error:
            failure = failure or ("runtime_close:" + repr(error))

        sources_after = {
            str(path): sha256(path) for path in source_paths if path.exists()}
        user_completed = stop["reason"] == "user_finished"
        separation_valid = bool(
            final_truth is not None
            and final_truth.get("soil_body_separation_valid") is True)
        fill_valid = bool(
            final_truth is not None
            and final_truth.get("fill_measurement_valid") is True)
        digging_observed = bool(
            last_truth is not None
            and last_truth.get("digging_interaction_observed_from_mpm_wrench") is True)
        autonomous_baseline_completion = bool(
            protected_v4_mode
            and not takeover_events
            and last_termination is not None
            and last_termination.terminated
            and last_termination.success)
        if failure is None and executed:
            report_status = "recorded_pending_audit"
        elif failure is None and executed == 0 and stop["reason"] in (
                "signal_requested", "user_finished", "window_closed"):
            report_status = "cancelled_no_steps"
        else:
            report_status = "failed"
        report = {
            "schema": SCHEMA,
            "status": report_status,
            "episode_id": args.episode_id,
            "session_label": args.session_label,
            "manual_input": {
                "requested_mode": args.manual_input,
                "actual_mode": manual_input_actual_mode,
                "device_name": manual_input_device_name,
                "device_index": manual_input_device_index,
                "config_sha256": manual_input_config_sha256,
                "acquisition_domain": args.acquisition_domain,
                "disconnect_count": gamepad_disconnect_count,
                "emergency_stop_triggered": gamepad_emergency_stop_triggered,
            },
            "acquisition_mode": (
                "protected_v4_baseline_evaluation" if protected_v4_mode else
                "dagger_takeover" if dagger_mode else "manual_demo"),
            "stop_reason": stop["reason"],
            "failure": failure,
            "wall_s": time.perf_counter() - started,
            "executed_decisions": executed,
            "executed_sim_time_s": executed * POLICY_DT_S,
            "phase_at_stop": phase,
            "peak_force_n": peak_force_n,
            "initial_truth": initial_truth,
            "final_truth": final_truth,
            "manual_evaluation": {
                "mouth_angle_role": "diagnostic_only",
                "minimum_bucket_closure_normalized": MIN_BUCKET_CLOSURE_NORMALIZED,
                "off_soil_confirmation_sim_s": 0.5,
                "required_hold_sim_s": 3.0,
                "sample_period_sim_s": 0.25,
                "load_correction_reentry_confirm_ticks": (
                    LOAD_CORRECTION_REENTRY_CONFIRM_TICKS),
                "completed_windows": completed_evaluation_windows,
                "current_hold_sim_s_at_stop": current_evaluation_hold_s,
                "current_sample_count_at_stop": len(evaluation_samples),
                "sealed_termination": (
                    termination.final.as_dict() if termination.final is not None else None),
            },
            "shadow_candidate_counts": shadow_counts,
            "shadow_intervention_count": len(takeover_events),
            "end_stop_projection": {
                "enabled": True,
                "projection_count": end_stop_projector.projection_count,
                "confirmed_at_stop": end_stop_projector.confirmed,
                "operator_may_hold_positive_bucket_key": True,
            },
            "dagger_takeover": {
                "enabled": dagger_mode,
                "policy_checkpoint_sha256": (
                    sha256(args.policy_checkpoint) if dagger_mode else None),
                "policy_checkpoint_metadata": (
                    policy_actor.metadata if policy_actor is not None else None),
                "policy_exploration_std": 0.0 if dagger_mode else None,
                "policy_observation_version": (
                    args.policy_observation_version if dagger_mode else None),
                "logged_observation_schema": (
                    OBSERVATION_SCHEMA_V3 if dagger_mode else None),
                "task_phase_at_stop": (
                    current_task_phase if dagger_mode else None),
                "task_phase_fill_blind": True if dagger_mode else None,
                "closed_bucket_negative_boom_pause_enabled": (
                    True if dagger_mode else None),
                "closed_bucket_negative_boom_pause_execution": (
                    "pause_full_action_before_physics" if dagger_mode else None),
                "milestone_phase_route_enabled": (
                    True if protected_v4_mode else None),
                "milestone_phase_route": (
                    "causal_milestone_two_tick_episode_reset"
                    if protected_v4_mode else None),
                "curl_reverse_pause_enabled": (
                    True if protected_v4_mode else None),
                "curl_reverse_threshold": (
                    curl_reverse_supervisor.reverse_threshold
                    if protected_v4_mode else None),
                "curl_reverse_confirmation_ticks": (
                    curl_reverse_supervisor.confirmation_ticks
                    if protected_v4_mode else None),
                "curl_reverse_pause_execution": (
                    "pause_full_action_before_physics"
                    if protected_v4_mode else None),
                "ttc_pause_threshold_s": (
                    args.takeover_ttc_s if dagger_mode else None),
                "clearance_pause_threshold_m": (
                    args.takeover_clearance_m if dagger_mode else None),
                "intervention_events": takeover_events,
                "takeover_handshake": (
                    takeover_handshake.as_dict() if dagger_mode else None),
                "actor_prefix_steps": actor_prefix_steps,
                "human_correction_samples": correction_samples,
                "control_mode_at_stop": control_mode,
                "autonomous_success_eligible": autonomous_baseline_completion,
                "autonomous_baseline_completion_observed": (
                    autonomous_baseline_completion),
                "critic_eligible": False,
                "claim": (
                    "protected v4 baseline evaluation; autonomous completion "
                    "requires a successful sealed termination and zero takeover, "
                    "and the episode never enters critic replay"
                    if protected_v4_mode else
                    "human-correction acquisition only; an intervention can "
                    "never become autonomous success or critic replay"
                    if dagger_mode else None),
            },
            "training_started": False,
            "persistent_replay_write_performed": False,
            "actor_parameter_updates": 0,
            "critic_parameter_updates": 0,
            "sac_parameter_updates": 0,
            "bc_candidate": {
                "status": (
                    "protected_actor_evaluation_only" if protected_v4_mode else
                    "dagger_corrections_pending_audit" if dagger_mode else
                    "pending_post_run_audit"),
                "user_completed_with_x": user_completed,
                "digging_interaction_observed": digging_observed,
                "final_separation_valid": separation_valid,
                "final_fill_measurement_valid": fill_valid,
                "automatic_admission": False,
            },
            "critic_eligible": False,
            "critic_exclusion_reason": (
                "protected_actor_evaluation_never_enters_replay"
                if protected_v4_mode else
                "human_intervention_episode_excluded" if dagger_mode else
                "human_demo_requires_terminal_reward_audit"),
            "runtime_counters": runtime.counters if made else None,
            "backend_qualification": made[0].qualification if made else None,
            "backend_sources_unchanged": bool(
                made and made[0].source_before is not None
                and made[0].source_before == made[0].source_after),
            "truth_sources_unchanged": bool(
                made and made[0].truth_sources_before is not None
                and made[0].truth_sources_before == made[0].truth_sources_after),
            "execution_sources_before": sources_before,
            "execution_sources_unchanged": sources_before == sources_after,
        }
        if args.output.exists():
            _write_json(args.output / "replay-admission.json", {
                "episode_id": args.episode_id,
                "persistent_replay_write_performed": False,
                "critic_eligible_transition_count": 0,
                "human_intervention_episode": bool(takeover_events),
                "dagger_correction_sample_count": correction_samples,
                "takeover_handshake": (
                    takeover_handshake.as_dict() if dagger_mode else None),
                "autonomous_success_eligible": autonomous_baseline_completion,
                "autonomous_baseline_completion_observed": (
                    autonomous_baseline_completion),
                "actor_parameter_updates": 0,
                "critic_parameter_updates": 0,
                "sac_parameter_updates": 0,
            })
            _write_json(args.output / "report.json", report)
            status.update(
                status="closed",
                task_ready=False,
                stop_reason=stop["reason"],
                failure=failure,
                executed_decisions=executed,
            )
            save_status()
        print(json.dumps({
            "status": report["status"],
            "stop_reason": stop["reason"],
            "failure": failure,
            "executed_decisions": executed,
            "bc_candidate": report["bc_candidate"],
            "dagger_takeover": report["dagger_takeover"],
        }, indent=2, allow_nan=False), flush=True)

    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
