"""Causal native-60Hz contact tensors; no normalization, split, or training grant.

This staging schema retains both operator intent and executed velocity. It does
not choose a network architecture or use posthoc phases as online features.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

SCHEMA = 'contact_bc_native60_structured_v1'
LIMITS = np.array([.12, .0225, .0225, .0675])
STATE_FIELDS = {
    'mechanism_qpos': 19, 'mechanism_qvel': 18,
    'controller_target': 4, 'controller_target_velocity': 4,
    'bucket_integral_effort': 4, 'bucket_active_curl_effort': 4,
    'bucket_origin_world_m': 3, 'bucket_linear_velocity_world_m_s': 3,
    'bucket_angular_velocity_world_rad_s': 3,
}
FEATURE_KEYS = [*STATE_FIELDS, 'terrain_height_m', 'terrain_known',
                'terrain_origin_xy_m', 'terrain_resolution_m', 'terrain_age_frames',
                'goal_world_m', 'previous_applied_action_si',
                'previous_drive_load', 'previous_drive_load_valid']
TARGET_KEYS = ['requested_action_normalized', 'applied_action_normalized']
AUDIT_KEYS = ['frame', 'sequence_id', 'reset_learning_history', 'terrain_anchor_frame']

def require(value, reason):
    if not value:
        raise ValueError(reason)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def terrain_arrays(terrain):
    require('height_m' in terrain, 'terrain_grid_missing')
    h = np.asarray(terrain['height_m'], dtype=np.float64)
    require(h.shape == tuple(terrain['shape_yx']), 'terrain_shape')
    require(not np.isinf(h).any(), 'infinite_terrain')
    known = np.isfinite(h)
    require(np.count_nonzero(known) == terrain['known_cell_count'], 'terrain_known_count')
    return np.where(known, h, 0).astype(np.float32), known

def encode_row(row, physics, observations, *, previous_frame, sequence_id):
    """Missing control or terrain starts a new history sequence, never a join."""
    f = row['frame']; anchor = row['terrain_anchor_frame']
    require(row['tensor_candidate'] and row['terrain_grid_available'], 'not_tensor_candidate')
    require(0 < anchor < f, 'terrain_not_prior')
    require(anchor == max(k for k in observations if k < f), 'terrain_not_latest_prior')
    require(row['terrain_age_frames'] == f - 1 - anchor <= 2, 'terrain_age')
    p = physics[f-1]; require(p['frame'] == f, 'physics_frame')
    reset = previous_frame is None or f != previous_frame + 1
    sequence_id += int(reset)
    state = p['before_state']
    encoded = {}
    for name, size in STATE_FIELDS.items():
        values = np.asarray(state[name], dtype=np.float64)
        require(values.shape == (size,) and np.isfinite(values).all(), 'invalid_state:'+name)
        encoded[name] = values.astype(np.float32)
    terrain = observations[anchor]['terrain']
    encoded['terrain_height_m'], encoded['terrain_known'] = terrain_arrays(terrain)
    encoded['terrain_origin_xy_m'] = np.asarray(terrain['origin_xy_m'], dtype=np.float32)
    encoded['terrain_resolution_m'] = np.float32(terrain['resolution_m'])
    encoded['terrain_age_frames'] = np.int64(f-1-anchor)
    goal = row['recorded_goal']
    encoded['goal_world_m'] = np.asarray([*goal['entry_xy_m'], goal['surface_z_m']], dtype=np.float32)
    prior = physics[f-2]['applied_command_si'] if f > 1 else [0.]*4
    require(row['previous_applied_action_si'] == prior, 'previous_action_mismatch')
    encoded['previous_applied_action_si'] = np.asarray(prior, dtype=np.float32)
    # Reset only learned history. The previous executed command remains known.
    encoded['previous_drive_load'] = np.asarray([0.]*4 if reset else physics[f-2]['equivalent_drive_load'], dtype=np.float32)
    encoded['previous_drive_load_valid'] = np.bool_(not reset)
    for source, target in [('requested_command_si','requested_action_normalized'),
                           ('applied_command_si','applied_action_normalized')]:
        action = np.asarray(p[source], dtype=np.float64) / LIMITS
        require(action.shape == (4,) and np.isfinite(action).all() and (np.abs(action)<=1+1e-8).all(), 'action_bounds')
        require(np.allclose(action, row[target], rtol=0, atol=1e-12), 'indexed_action_mismatch')
        encoded[target] = action.astype(np.float32)
    encoded.update(frame=np.int64(f), sequence_id=np.int64(sequence_id),
                   reset_learning_history=np.bool_(reset), terrain_anchor_frame=np.int64(anchor))
    require(all(np.isfinite(v).all() for v in encoded.values()), 'nonfinite_tensor')
    return encoded, sequence_id

def export(manifest_path, output):
    manifest_path = Path(manifest_path); manifest = read(manifest_path)
    require(manifest['action_rate_decision']=='user_confirmed_native60Hz_four_axes', 'action_rate_not_confirmed')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    spec = dict(schema=SCHEMA, action_hz=60, action_order=['swing','boom','arm','bucket'],
                action_units=['rad/s','m/s','m/s','m/s'], limits_si=LIMITS.tolist(),
                feature_keys=FEATURE_KEYS, target_keys=TARGET_KEYS, audit_keys=AUDIT_KEYS,
                normalization='none; later fit only on assigned development data',
                history='sequence-local; frame gaps or missing terrain reset; no chosen temporal window',
                provenance='simulator-derived terrain and state; not a real-sensor guarantee',
                excluded_features=['operator_phase','posthoc_phase','final_fill','future_state','particle_Jp','scoop_index','lineage'])
    report = dict(schema=SCHEMA, spec=spec, spec_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
                  source_manifest_sha256=sha(manifest_path), training_eligible=False,
                  split_assignment='unassigned', normalization_fitted=False, episodes=[])
    for e in manifest['episodes']:
        path = Path(e['index_path']); require(sha(path)==e['index_sha256'], 'index_changed')
        run = manifest_path.parents[2]/'runs/layered-47-sparse-manual-contact-v1'/e['run']
        require(all(sha(run/n)==digest for n,digest in e['source_bindings'].items()), 'source_changed')
        physics = rows(run/'physics.ndjson'); observations={o['frame']:o for o in rows(run/'observation.ndjson')}
        index = rows(path); values=[]; previous=None; sequence_id=0
        for row in index:
            if not row['tensor_candidate']:
                previous=None
                continue
            encoded,sequence_id=encode_row(row, physics, observations, previous_frame=previous, sequence_id=sequence_id)
            values.append(encoded);previous=row['frame']
        require(len(values)==e['tensor_candidate_frames'] and len(values)>0,'tensor_count')
        arrays={k:np.stack([v[k] for v in values]) for k in values[0]}
        dest=out/(e['run']+'.npz');np.savez_compressed(dest,**arrays)
        require(all(sha(run/n)==digest for n,digest in e['source_bindings'].items()), 'source_changed_during_export')
        report['episodes'].append(dict(run=e['run'],lineage=e['lineage'],scoop_index=e['scoop_index'],
                                      rows=len(values),sequence_count=sequence_id,path=str(dest.resolve()),sha256=sha(dest),
                                      reset_mode=e['reset_mode'],material_history_policy=e['material_history_policy'],
                                      operator_retargeting_review=e['operator_retargeting_review']))
    report['rows']=sum(e['rows'] for e in report['episodes'])
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=export(args.manifest,args.output)
    print(json.dumps(dict(schema=SCHEMA,rows=result['rows'],episodes=len(result['episodes']),training_eligible=False)))
