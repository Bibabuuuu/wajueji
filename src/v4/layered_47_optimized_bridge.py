"""Scoped, opt-in bridge compilation using the D-107 tested header."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from build_sparse_inertia_candidate import render_header

CANDIDATE_SHA256 = 'f3faf5fcc3416666faef0c8a084d1d9a2cd264806a0594dcd508a4f3d3c63b64'


def _replace_once(source, old, new, label):
    if source.count(old) != 1:
        raise ValueError('drive_load_observation_anchor:' + label)
    return source.replace(old, new)


def instrument_drive_load(source):
    """Expose, but never compensate on, the projected external bucket load."""
    source = _replace_once(
        source,
        '    std::array<double, 3> bucket_interval_reference_point{};\n',
        '    std::array<double, 3> bucket_interval_reference_point{};\n'
        '    std::array<double, 4> equivalent_drive_load{};\n'
        '    std::array<double, 4> unloaded_feedforward{};\n',
        'declaration')
    source = _replace_once(
        source,
        '      array("controller_target_velocity", '
        'controller.target_velocity.data(), 4);\n',
        '      array("controller_target_velocity", '
        'controller.target_velocity.data(), 4);\n'
        '      array("actuator_force", data->actuator_force, model->nu);\n'
        '      array("equivalent_drive_load", '
        'equivalent_drive_load.data(), 4);\n'
        '      array("unloaded_feedforward", '
        'unloaded_feedforward.data(), 4);\n',
        'emit')
    source = _replace_once(
        source,
        '        const auto result = feedforward.Evaluate(\n'
        '            data->qpos, controller, controller.target_velocity, '
        'acceleration);\n',
        '        const auto result = feedforward.Evaluate(\n'
        '            data->qpos, controller, controller.target_velocity, '
        'acceleration);\n'
        '        unloaded_feedforward = result.force;\n',
        'feedforward')
    source = _replace_once(
        source,
        '        const double before = data->time;\n'
        '        mj_step(model.get(), data.get());\n',
        '        // Observation only: project the already-applied external '
        'wrench through\n'
        '        // the closed-chain tangent.  Do not add compensation or '
        'change ctrl.\n'
        '        equivalent_drive_load.fill(0);\n'
        '        for (int axis = 0; axis < 4; ++axis) {\n'
        '          for (int dof = 0; dof < model->nv; ++dof) {\n'
        '            equivalent_drive_load[axis] +=\n'
        '                result.tangent[dof * 4 + axis] * '
        'data->qfrc_applied[dof];\n'
        '          }\n'
        '        }\n'
        '        const double before = data->time;\n'
        '        mj_step(model.get(), data.get());\n',
        'projection')
    return source


@contextmanager
def optimized_bridge(header_dir, build_dir):
    # Imported here so static integrity tests need no physics runtime.
    import layered_mound_checkpoint_runtime as runtime
    destination = Path(build_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    code, provenance = render_header(Path(header_dir)/'inertia_controller.h')
    if provenance['candidate_header_sha256'] != CANDIDATE_SHA256:
        raise ValueError('unvalidated_inertia_candidate')
    header = destination/'inertia_candidate.h'
    header.write_text(code)
    original = runtime.render_capacity_bridge

    def render(*args):
        source, manifest = original(*args)
        anchor = '#include "inertia_controller.h"'
        if source.count(anchor) != 1:
            raise ValueError('inertia_include_anchor')
        source = source.replace(anchor, '#include "'+str(header)+'"')
        source = instrument_drive_load(source)
        (destination/'capacity-bridge.cc').write_text(source)
        provenance['drive_load_observation_only'] = True
        provenance['external_load_compensation_applied'] = False
        provenance['generated_bridge_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        (destination/'provenance.json').write_text(json.dumps(provenance, indent=2))
        return source, dict(manifest, optimized_inertia=provenance.copy())

    runtime.render_capacity_bridge = render
    try:
        yield provenance
    finally:
        runtime.render_capacity_bridge = original


def parser():
    import argparse
    result = argparse.ArgumentParser()
    for name in ('config','repository-root','output','candidate-report','resume-report',
                 'checkpoint-sha256','device','mujoco-include','mujoco-library-dir',
                 'mechanism-header-dir','controller-header-dir'):
        result.add_argument('--'+name, required=True)
    result.add_argument('--cxx', default='g++')
    return result
