"""Opt-in active curl experiment composed after the frozen D120 bridge.

No default manual deployment. Delayed activation supports matched-state tests.
Supplemental force is diagnostic state, not a qualified resumable controller.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path

HEADER = Path(__file__).with_name("bucket_active_curl_candidate.h")


def instrument_active_curl(source, header, activation_time_s=0.0,
                           initialize_from_d120=False):
    if not math.isfinite(activation_time_s) or activation_time_s < 0:
        raise ValueError("active_curl_activation_time")

    def replace_once(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("active_curl_candidate_anchor")
        source = source.replace(old, new)

    replace_once(
        "    BucketIntegral bucket_integrator;",
        "    BucketIntegral bucket_integrator;\n"
        "    BucketActiveCurl active_curl;\n"
        "    std::array<double,4> bucket_active_curl_effort{};")
    replace_once(
        '      array("bucket_integral_effort", bucket_integral_effort.data(), 4);',
        '      array("bucket_integral_effort", bucket_integral_effort.data(), 4);\n'
        '      array("bucket_active_curl_effort", bucket_active_curl_effort.data(), 4);')
    replace_once(
        "        const double before = data->time;\n"
        "        mj_step(model.get(), data.get());",
        f"        if (data->time >= {activation_time_s:.17g} - 1e-9) {{\n"
        "          bucket_active_curl_effort[3] = active_curl.Step(\n"
        "              command[3], bucket_error, model->opt.timestep);\n"
        "          data->ctrl[bucket_hold] += bucket_active_curl_effort[3] /\n"
        "              model->actuator_gainprm[mjNGAIN * bucket_hold];\n"
        "        }\n"
        "        const double before = data->time;\n"
        "        mj_step(model.get(), data.get());")
    # Loaded restart must not silently discard a nonzero supplemental effort.
    if not initialize_from_d120:
        replace_once(
            '        const bool restore_loaded = operation == "init_loaded";',
            '        const bool restore_loaded = operation == "init_loaded";\n'
            '        if (restore_loaded) throw std::runtime_error("active curl loaded restore unqualified");')
    return '#include ' + json.dumps(str(Path(header).resolve())) + '\n' + source


@contextmanager
def active_curl_bridge(enabled, build_dir, *, activation_time_s=0.0,
                       initialize_from_d120=False):
    policy = {"enabled": bool(enabled), "default_enabled": False}
    if not enabled:
        yield policy
        return
    if not math.isfinite(activation_time_s) or activation_time_s < 0:
        raise ValueError("active_curl_activation_time")
    import layered_mound_checkpoint_runtime as runtime
    destination = Path(build_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    header = destination / HEADER.name
    header.write_bytes(HEADER.read_bytes())
    policy.update(
        version="D136_active_curl_v1", gain_N_per_m=6000000,
        cap_N=60000, deadband_m=.001, rise_N_per_s=120000,
        release_N_per_s=600000, activation_time_s=activation_time_s,
        activation="positive_applied_bucket_command_and_positive_tracking_error",
        neutral_reverse="release_supplement_within_0.1s; D120_support_unchanged",
        physical_hydraulic_limit=False, training_eligible=False,
        loaded_restore_qualified=False,
        initialize_from_d120_with_zero_new_effort=bool(initialize_from_d120),
        header_sha256=hashlib.sha256(header.read_bytes()).hexdigest())
    original = runtime.render_capacity_bridge

    def render(*args):
        source, manifest = original(*args)
        source = instrument_active_curl(source, header, activation_time_s,
                                        initialize_from_d120)
        policy["generated_bridge_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        (destination / "capacity-bridge.cc").write_text(source)
        (destination / "policy.json").write_text(json.dumps(policy, indent=2) + '\n')
        return source, dict(manifest, active_curl_candidate=policy.copy())

    runtime.render_capacity_bridge = render
    try:
        yield policy
    finally:
        runtime.render_capacity_bridge = original
