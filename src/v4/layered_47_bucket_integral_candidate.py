"""Opt-in D-120 bucket integral for a user-authorized manual pilot.

The loaded curl improvement is verified; stopped/reverse qualification is not.
This context leaves the default bridge and frozen experiment sources unchanged.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

HEADER = Path(__file__).with_name("bucket_integral_candidate.h")
HEADER_SHA256 = "226e9421fcae393a6084edb54c43061378d73101f88a29839bbb4be5656ebecc"


def instrument_bucket_integral(source, header):
    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("bucket_integral_candidate_anchor")
        source = source.replace(old, new)

    replace_once(
        "    std::array<double, 4> unloaded_feedforward{};",
        "    std::array<double, 4> unloaded_feedforward{};\n"
        "    std::array<double,4> bucket_integral_effort{};\n"
        "    BucketIntegral bucket_integrator;\n"
        "    std::array<double,4> restored_controller_target{};\n"
        "    std::array<double,4> restored_controller_target_velocity{};\n"
        "    double restored_bucket_integral_effort = 0.0;")
    replace_once(
        '      array("unloaded_feedforward", unloaded_feedforward.data(), 4);',
        '      array("unloaded_feedforward", unloaded_feedforward.data(), 4);\n'
        '      array("bucket_integral_effort", bucket_integral_effort.data(), 4);')
    replace_once(
        '      if (operation == "init_default" || operation == "init") {\n'
        '        if (ready) throw std::runtime_error("init once only");\n'
        '        if (operation == "init") {\n'
        '          for (int index = 0; index < model->nq; ++index) data->qpos[index] = number();\n'
        '          for (int index = 0; index < model->nv; ++index) data->qvel[index] = number();\n'
        '        }\n'
        '        std::string extra;\n',
        '      if (operation == "init_default" || operation == "init" ||\n'
        '          operation == "init_loaded") {\n'
        '        if (ready) throw std::runtime_error("init once only");\n'
        '        const bool restore_loaded = operation == "init_loaded";\n'
        '        if (operation == "init" || restore_loaded) {\n'
        '          for (int index = 0; index < model->nq; ++index) data->qpos[index] = number();\n'
        '          for (int index = 0; index < model->nv; ++index) data->qvel[index] = number();\n'
        '        }\n'
        '        if (restore_loaded) {\n'
        '          for (double& value : restored_controller_target) value = number();\n'
        '          for (double& value : restored_controller_target_velocity) value = number();\n'
        '          restored_bucket_integral_effort = number();\n'
        '          if (std::abs(restored_bucket_integral_effort) > BucketIntegral::cap)\n'
        '            throw std::runtime_error("loaded integral outside envelope");\n'
        '        }\n'
        '        std::string extra;\n')
    replace_once(
        "        ConfigureCandidate(model.get(), controller, 1, 1);\n"
        "        const auto initial = feedforward.Evaluate(data->qpos, controller, {}, {});",
        "        ConfigureCandidate(model.get(), controller, 1, 1);\n"
        "        if (restore_loaded) {\n"
        "          controller.target = restored_controller_target;\n"
        "          controller.target_velocity = restored_controller_target_velocity;\n"
        "          controller.effective_velocity = restored_controller_target_velocity;\n"
        "          for (int axis = 0; axis < 4; ++axis) {\n"
        "            controller.reference[axis].Reset(restored_controller_target[axis]);\n"
        "            controller.reference[axis].velocity =\n"
        "                restored_controller_target_velocity[axis];\n"
        "          }\n"
        "          bucket_integrator.effort = restored_bucket_integral_effort;\n"
        "          bucket_integral_effort[3] = restored_bucket_integral_effort;\n"
        "        }\n"
        "        const auto initial = feedforward.Evaluate(data->qpos, controller, {}, {});")
    replace_once(
        "        const double before = data->time;\n"
        "        mj_step(model.get(), data.get());",
        "        const double bucket_error = controller.target[3] -\n"
        "            data->qpos[controller.qpos_address[3]];\n"
        "        bucket_integral_effort[3] = bucket_integrator.Step(\n"
        "            command[3], bucket_error, model->opt.timestep);\n"
        "        const int bucket_hold = controller.hold_actuator[3];\n"
        "        data->ctrl[bucket_hold] += bucket_integral_effort[3] /\n"
        "            model->actuator_gainprm[mjNGAIN * bucket_hold];\n"
        "        const double before = data->time;\n"
        "        mj_step(model.get(), data.get());")
    return "#include " + json.dumps(str(Path(header).resolve())) + "\n" + source


@contextmanager
def bucket_integral_bridge(enabled, build_dir):
    policy = {"enabled": bool(enabled), "default_enabled": False}
    if not enabled:
        yield policy
        return
    import layered_mound_checkpoint_runtime as runtime
    code = HEADER.read_bytes()
    if hashlib.sha256(code).hexdigest() != HEADER_SHA256:
        raise ValueError("unvalidated_bucket_integral_header")
    destination = Path(build_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    header = destination / HEADER.name
    header.write_bytes(code)
    policy.update(
        version="D120_conditional_integral_manual_pilot_v1",
        ki_N_per_m_s=1500000, cap_N=120000, slew_N_per_s=30000,
        deadband_m=0.0005, initial_effort_N=0,
        activation="from_first_step; growth_requires_active_bucket_command",
        neutral="preserve_support; opposite_error_may_unwind",
        reverse="D120_original; no_D121_forced_release",
        physical_hydraulic_limit=False, stopped_qualified=False,
        reverse_qualified=False, training_eligible=False,
        loaded_state_restore=True,
        loaded_state_restore_fields=(
            "controller_target", "controller_target_velocity",
            "bucket_integral_effort[3]"),
        header_sha256=HEADER_SHA256)
    original = runtime.render_capacity_bridge

    def render(*args):
        source, manifest = original(*args)
        source = instrument_bucket_integral(source, header)
        policy["generated_bridge_sha256"] = hashlib.sha256(
            source.encode()).hexdigest()
        (destination / "capacity-bridge.cc").write_text(source)
        (destination / "policy.json").write_text(json.dumps(policy, indent=2))
        return source, dict(manifest, bucket_integral_candidate=policy.copy())

    runtime.render_capacity_bridge = render
    try:
        yield policy
    finally:
        runtime.render_capacity_bridge = original
