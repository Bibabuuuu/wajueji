"""Versioned full-frame continuation envelope for the local three-scoop pilot.

Historical diagnostic imports retain an explicit cold-native-history boundary.
A saved envelope alone never admits demonstrations to training.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np

import layered_47_interactive_proxy_checkpoint as sparse
from layered_mound_checkpoint import checkpoint_identity
from layered_47_cycle_bridge import native_restore_tokens

SCHEMA = 'layered_47_cycle_checkpoint_v1'
PHYSICS_FILES = ('bucket_integral_candidate.h', 'bucket_active_curl_candidate.h',
                 'soil_hencky_update_candidate.py', 'moving_bucket_sweep_candidate.py',
                 'moving_bucket_sweep_runtime.py')
STAGES = ('loaded_manual_endpoint', 'hold', 'transport', 'unloaded', 'returned', 'prepoint', 'failure')
CONTROLLER_KEYS = ('controller_target', 'controller_target_velocity',
                   'bucket_integral_effort', 'bucket_active_curl_effort')


def validate_sweep(state):
    state = deepcopy(state)
    vectors = []
    for key in ('total_generated_reaction_ns', 'total_routed_reaction_ns'):
        x = np.asarray(state[key], dtype=float)
        if x.shape != (3,) or not np.isfinite(x).all():
            raise ValueError('cycle_sweep:' + key)
        vectors.append(x)
    pending = state['pending_velocity_reaction']
    impulse = np.zeros(3)
    if pending is not None:
        for key in ('reaction_impulse_ns', 'reaction_angular_impulse_nms', 'point_world_m'):
            x = np.asarray(pending[key], dtype=float)
            if x.shape != (3,) or not np.isfinite(x).all():
                raise ValueError('cycle_sweep_pending:' + key)
        impulse = np.asarray(pending['reaction_impulse_ns'])
    if np.max(np.abs(vectors[0] - vectors[1] - impulse)) > 1e-7:
        raise ValueError('cycle_sweep_impulse_balance')
    return state


def restore_sweep(runtime, state):
    state = validate_sweep(state)
    if runtime.old_pose is not None:
        raise ValueError('cycle_sweep_restore_mid_frame')
    runtime.pending = state['pending_velocity_reaction']
    runtime.generated = np.array(state['total_generated_reaction_ns'], dtype=float)
    runtime.routed = np.array(state['total_routed_reaction_ns'], dtype=float)


def validate_controller(state, count):
    for key in CONTROLLER_KEYS:
        v = np.asarray(state[key], dtype=float)
        if v.shape != (4,) or not np.isfinite(v).all():
            raise ValueError('cycle_controller:' + key)
    for key in ('bucket_linear_velocity_world_m_s', 'bucket_angular_velocity_world_rad_s'):
        v = np.asarray(state[key], dtype=float)
        if v.shape != (3,) or not np.isfinite(v).all():
            raise ValueError('cycle_reported_bucket_velocity:' + key)
    integral = np.asarray(state['bucket_integral_effort'])
    if np.any(integral[:3] != 0) or abs(integral[3]) > 120000:
        raise ValueError('cycle_integral_envelope')
    native_restore_tokens(state, count)


def load(path):
    path = Path(path).resolve()
    identity = checkpoint_identity(path)
    sidecar = Path(str(path) + '.json')
    metadata = json.loads(sidecar.read_text())
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    sparse._validate(arrays, 256441, 47)
    if metadata.get('schema') == SCHEMA:
        if metadata.get('stage') == 'failure':
            raise ValueError('cycle_failure_checkpoint_not_resumable')
        if metadata['checkpoint_sha256'] != identity['sha256']:
            raise ValueError('cycle_checkpoint_sha256')
        if metadata.get('stage') not in STAGES or metadata.get('cycle_index') not in (1,2,3):
            raise ValueError('cycle_metadata_stage_or_index')
        if not isinstance(metadata.get('truck_enabled'), bool):
            raise ValueError('cycle_metadata_truck_flag')
        state = metadata['controller_state']
    elif metadata.get('schema') == 'layered_47_manual_contact_diagnostic_snapshot_v1':
        if metadata['sha256'] != identity['sha256'] or metadata['runtime_state'] == 'fault_frozen':
            raise ValueError('cycle_diagnostic_identity_or_fault')
        report_path = path.parent/'report.json'
        report = json.loads(report_path.read_text())
        if (report.get('status') != 'manual_contact_pilot_recorded'
                or not report.get('checks') or not all(report['checks'].values())
                or not any(item['sha256'] == identity['sha256'] for item in report.get('diagnostic_snapshots', []))):
            raise ValueError('cycle_legacy_source_requires_normal_reviewable_episode')
        old = metadata['controller_state_diagnostic_only']
        state = {k: old[k] for k in (*CONTROLLER_KEYS,
            'bucket_linear_velocity_world_m_s', 'bucket_angular_velocity_world_rad_s')}
        state.update(native_state_kind='legacy_diagnostic_cold_acceleration',
                     native_time_s=old['time_s'], native_last_time_s=max(0, old['time_s']-1/4800),
                     native_timestep_s=1/4800)
        metadata = dict(metadata, controller_state=state, cycle_index=1,
                        stage='loaded_manual_endpoint', truck_enabled=False,
                        lineage_root=identity['sha256'],
                        historical_native_history_missing=True, runtime_bindings=report['input_bindings'])
    else:
        raise ValueError('cycle_checkpoint_schema')
    bindings = {Path(key).name: value for key,value in metadata['runtime_bindings'].items()}
    for name in PHYSICS_FILES:
        if bindings.get(name) != checkpoint_identity(Path(__file__).with_name(name))['sha256']:
            raise ValueError('cycle_physics_version_mismatch:' + name)
    validate_controller(state, 47)
    validate_sweep(metadata['moving_bucket_sweep_state'])
    metadata['source_identity'] = identity
    metadata['source_metadata_identity'] = checkpoint_identity(sidecar)
    return arrays, metadata


def save(path, arrays, controller, sweep, *, parent, stage, cycle_index,
         truck_enabled, solver_profile, runtime_bindings, selected_cut=None):
    validate_controller(controller, 47)
    sweep = validate_sweep(sweep)
    if controller['native_state_kind'] != 'full':
        raise ValueError('cycle_save_requires_full_native_state')
    if stage not in ('loaded_manual_endpoint', 'hold', 'transport', 'unloaded',
                     'returned', 'prepoint', 'failure') or cycle_index not in (1,2,3):
        raise ValueError('cycle_stage_or_index')
    info = sparse.write_checkpoint_atomic(path, arrays)
    document = dict(schema=SCHEMA, checkpoint_sha256=info['sha256'],
        controller_state=controller, moving_bucket_sweep_state=sweep,
        parent_checkpoint=parent['source_identity'],
        lineage_root=parent['lineage_root'],
        historical_native_history_missing=parent.get('historical_native_history_missing', False),
        cycle_index=cycle_index, stage=stage, truck_enabled=bool(truck_enabled),
        solver_profile=solver_profile, runtime_bindings=runtime_bindings,
        selected_cut=selected_cut, training_eligible=False,
        independent_restore_qualified=False,
        simulation_time_s=float(arrays['simulation_time_s'][0]))
    # The NPZ is never overwritten. A missing sidecar is an invalid checkpoint.
    with Path(str(path)+'.json').open('x') as out:
        json.dump(document, out, indent=2, allow_nan=False)
        out.write('\n')
    return info
