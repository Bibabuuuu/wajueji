"""Independent source-to-array verification, without importing the exporter."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
DATA=ROOT/'local-data/contact-bc-native-tensors-20260924-v1'
def read(p):return json.loads(p.read_text())
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
staging=read(OUT/'report.json');tensors=read(DATA/'manifest.json');total=0;results=[]
assert tensors['source_manifest_sha256']==sha(OUT/'report.json')
assert not tensors['training_eligible'] and not tensors['normalization_fitted'] and tensors['split_assignment']=='unassigned'
for e,t in zip(staging['episodes'],tensors['episodes'],strict=True):
    run=ROOT/'runs/layered-47-sparse-manual-contact-v1'/e['run'];r=read(run/'report.json')
    actions=rows(run/'input.ndjson');physics=rows(run/'physics.ndjson');obs={x['frame']:x for x in rows(run/'observation.ndjson')}
    index=rows(Path(e['index_path']));assert sha(Path(e['index_path']))==e['index_sha256']
    manual=[a for a in actions if a['input_device'] in ('gamepad','keyboard') and a['armed'] and a['connected'] and a['window_focused'] and not a.get('emergency_stop_latched',False)]
    assert [x['frame'] for x in index]==[x['frame'] for x in manual]
    assert all(x['tensor_candidate']==(x['frame']>3) for x in index)
    assert all(x['terrain_reference'] is None for x in index if x['frame']<=3)
    expected=[x['frame'] for x in manual if x['frame']>3]
    assert sha(Path(t['path']))==t['sha256'] and t['lineage']==e['lineage']
    with np.load(t['path'],allow_pickle=False) as archive:
        a={key:archive[key] for key in archive.files}
        assert a['frame'].tolist()==expected
        assert set(a)==set(tensors['spec']['feature_keys']+tensors['spec']['target_keys']+tensors['spec']['audit_keys'])
        assert all(np.isfinite(a[k]).all() for k in a)
        prev=None;seq=0
        for i,f in enumerate(expected):
            p=physics[f-1];anchor=max(k for k in obs if k<f);terrain=obs[anchor]['terrain']
            assert a['terrain_anchor_frame'][i]==anchor and a['terrain_age_frames'][i]==f-1-anchor
            h=np.asarray(terrain['height_m'],dtype=float);mask=np.isfinite(h)
            np.testing.assert_array_equal(a['terrain_known'][i],mask)
            np.testing.assert_array_equal(a['terrain_height_m'][i],np.where(mask,h,0).astype(np.float32))
            for key in ('mechanism_qpos','mechanism_qvel','controller_target','controller_target_velocity','bucket_integral_effort','bucket_active_curl_effort','bucket_origin_world_m','bucket_linear_velocity_world_m_s','bucket_angular_velocity_world_rad_s'):
                np.testing.assert_array_equal(a[key][i],np.asarray(p['before_state'][key],dtype=np.float32))
            goal=r['selected_cut'];np.testing.assert_array_equal(a['goal_world_m'][i],np.asarray([*goal['entry_xy_m'],goal['surface_z_m']],dtype=np.float32))
            np.testing.assert_array_equal(a['terrain_origin_xy_m'][i],np.asarray(terrain['origin_xy_m'],dtype=np.float32))
            assert a['terrain_resolution_m'][i]==np.float32(terrain['resolution_m'])
            for name,key in [('requested_command_si','requested_action_normalized'),('applied_command_si','applied_action_normalized')]:
                np.testing.assert_array_equal(a[key][i],(np.asarray(actions[f-1][name])/r['protocol']['maximum_command_si']).astype(np.float32))
            np.testing.assert_array_equal(a['previous_applied_action_si'][i],np.asarray(actions[f-2]['applied_command_si'],dtype=np.float32))
            reset=prev is None or f!=prev+1;seq+=int(reset)
            assert a['reset_learning_history'][i]==reset and a['sequence_id'][i]==seq
            assert a['previous_drive_load_valid'][i]==(not reset)
            np.testing.assert_array_equal(a['previous_drive_load'][i],np.asarray([0.]*4 if reset else physics[f-2]['equivalent_drive_load'],dtype=np.float32))
            prev=f
        assert seq==t['sequence_count']
    assert all(sha(run/n)==digest for n,digest in e['source_bindings'].items())
    phases=read(Path(e['phase_path']));assert phases['original_events']==r['phase_events'] and phases['not_online_features']
    results.append(dict(run=e['run'],passed=True,index_frames=len(index),tensor_frames=len(expected),sequences=seq))
    total+=len(expected)
assert total==tensors['rows']==staging['tensor_candidate_frames']==13728
result=dict(passed=True,episodes=results,tensor_rows=total,excluded_missing_initial_terrain=30,normalized=False,training_eligible=False,
            spec_sha256=tensors['spec_sha256'],manifest_sha256=sha(DATA/'manifest.json'),scope='Source-to-tensor integrity and causality; not expert selection, fitted normalization, model or closed-loop qualification.')
(OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
