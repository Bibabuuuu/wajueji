"""Read-only review of the fast-collection third scoop; no simulation steps."""
import json, hashlib
from pathlib import Path
import numpy as np
from live_bucket_fill import MixedFillEstimator, load_config
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
from audit_layered_47_loaded_handoff import cavity_mask

ROOT=Path('/home/qunwang/wajueji')
RUN=ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-115812-1458627'
FINAL=RUN/'final-001490.cycle.npz'
BUNDLE=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1/repo')
K=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
def read(name):return json.loads((RUN/name).read_text())
def rows(name):return [json.loads(x) for x in (RUN/name).read_text().splitlines()]
def identity(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
r=read('report.json');ph=rows('physics.ndjson');actions=rows('input.ndjson');obs=rows('observation.ndjson')
a=np.load(FINAL);initial=np.load(r['handoff_checkpoint']['path'])
assembly,_=_load_kinematics(K,K/'scene/input/model/A.xml',K/'scene/input/mesh')
pose=assembly.assemble(a['mechanism_qpos'],include_meshes=False)[1]['bucket']
records=json.loads((BUNDLE/'results/layered-mound-physical-preflight-v1/scene-manifest.json').read_text())['rocks']
records=[v for v in records if v['layer_id'] in ('lower','lower_middle','middle','upper')]
config=r['live_bucket_fill']['config']
assert config['occupancy_level_override']==.2
e=MixedFillEstimator(config,[v['volume_m3'] for v in records],_cache_rock_vertices(BUNDLE,records))
fill=e.measure_snapshot(dict(frame=len(ph),soil=a['particle_q'],rocks=a['rigid_qpos'],pose=pose))
soil=e.measure(a['particle_q'],pose,e.contract,e.front,.025**3,e.calibration)
local=(a['particle_q']-pose[:3,3])@pose[:3,:3]
inside=cavity_mask(local,np.asarray(e.contract['bucket']['polygon_xz_m']),np.asarray(e.contract['bucket']['y_limits_m']),e.front)
state=ph[-1]['after_state'];velocity=np.asarray(state['bucket_linear_velocity_world_m_s']);omega=np.asarray(state['bucket_angular_velocity_world_rad_s'])
speed=np.linalg.norm(a['particle_qd']-velocity-np.cross(omega,a['particle_q']-pose[:3,3]),axis=1)
n=len(ph)
checks=dict(normal_user_finish=r['status']=='manual_contact_pilot_recorded' and r['stop_reason']=='user_finished',
 all_runtime_checks=all(r['checks'].values()),all_hard_safe=all(v['hard_guard'] for v in ph),
 all_material_safe=all(v['hencky_material_guard']['safe'] for v in ph),all_sweep_safe=all(v['moving_bucket_sweep']['safe'] for v in ph),
 input_frames_complete=[v['frame'] for v in actions]==list(range(1,n+1)),
 observation_anchors_complete=[v['frame'] for v in obs]==list(range(3,n+1,3)),
 actions_match_physics=len(actions)==n and all(x['applied_command_si']==y['applied_command_si'] for x,y in zip(actions,ph)),
 observations_match_post_step=all(x['state']==ph[x['frame']-1]['after_state'] for x in obs),
 final_30_system_zero=all(v['input_device']=='system_safety_hold' and v['applied_command_si']==[0.,0.,0.,0.] for v in actions[-30:]),
 fast_third_prepoint='fast-contact-collection/20260924-115633-1457317/prepare' in r['handoff_checkpoint']['path'],
 training_reset_preserved=json.loads(Path(str(FINAL)+'.json').read_text())['training_reset']==r['training_reset'],
 third_scoop=json.loads(Path(str(FINAL)+'.json').read_text())['cycle_index']==3,
 snapshot_identity_matches=identity(FINAL)['sha256']==r['cycle_checkpoints'][0]['sha256'],
 finite_final=bool(np.isfinite(a['particle_q']).all()),same_particle_count=len(a['particle_q'])==len(initial['particle_q'])==256441,
 same_rock_count=len(a['rigid_qpos'])==len(initial['rigid_qpos'])==47)
result=dict(schema='third_chain_third_scoop_review_v1',run=str(RUN),checks=checks,recording_review_pass=all(checks.values()),
 frames=n,simulation_seconds=n/60,input_frames=len(actions),observation_anchors=len(obs),
 live_off_soil_fill=r['live_bucket_fill']['last_display_result'],final_mixed_fill=fill,final_soil_only_fill=soil,
 cavity_soil_particles=int(inside.sum()),cavity_soil_relative_speed_quantiles_m_s=np.quantile(speed[inside],[0,.5,.9,1]).tolist(),
 cavity_soil_vertical_change_from_initial_prepoint_m=np.quantile((a['particle_q']-initial['particle_q'])[inside,2],[0,.5,1]).tolist(),
 phase_markers=[dict(frame=v['next_frame'],label=v['label']) for v in r['phase_events']],
 final_controller_axis_error_mm=((np.array(state['controller_target'])[1:]-np.array(ph[-1]['axis_positions'])[1:])*1000).tolist(),
 lineage=json.loads(Path(str(FINAL)+'.json').read_text())['lineage_root'],snapshot=identity(FINAL),metadata=identity(Path(str(FINAL)+'.json')),
 scope='Geometric in-bucket mixed occupancy, including heap; not stable delivered truck volume or automatic training admission.',
 bindings=[identity(RUN/x) for x in ('report.json','physics.ndjson','input.ndjson','observation.ndjson')]+[identity(Path(__file__))])
Path(__file__).with_name('report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
assert all(checks.values()) and fill['valid_estimate']
