"""CPU development ranking; working retained volume is not formal truck volume."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(Path(p).read_text())
def summarize(contact,tail):
 contact=Path(contact);tail=Path(tail);c=read(contact/'report.json');t=read(tail/'report.json')
 result=dict(eligible=False,formal_fill_qualified=False,contact=str(contact),tail=str(tail),status=t['status'])
 if c['status']!='bounded_probe_completed' or t['status'] not in ('dig_lift_completed','second_scoop_unload_completed'):return result
 crows=[json.loads(x) for x in (contact/'physics/frames.ndjson').read_text().splitlines()];trows=[json.loads(x) for x in (tail/'physics/frames.ndjson').read_text().splitlines()]
 inv=t.get('unload_inventory');checks=dict(all_physics_guards=all(r['hard_guard'] for r in crows+trows),
  frame_counts_match=bool(crows) and bool(trows) and len(crows)==c['frames_completed'] and len(trows)==t['frames'],
  contact_exact=all(c['zero_step_arrays_exact'].values()),tail_exact=all(t['zero_arrays_exact'].values()),
  source_unchanged=c['source_unchanged'] and t['source_unchanged'],
  positive_development_receipt=inv is None or inv['received_source_particles']>0,
  handoff=bool(t.get('handoff',{}).get('ready',False)))
 result['checks']=checks
 if not all(checks.values()):return result
 clear=next(s['frame'] for s in t['stages'] if s['name']=='CLEAR_PILE')
 if clear<4 or clear>len(trows):result['invalid_reason']='invalid_clear_pile_frame';return result
 scene=read(tail/'display/scene.json');states=read(tail/'display/states.json');particles=np.load(tail/'display/particles.npy',mmap_mode='r')
 from live_bucket_fill import MixedFillEstimator,load_config
 from layered_excavation_approach_planner import _load_kinematics
 from run_layered_47_sparse_manual_contact_episode import _cache_rock_vertices
 k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,_=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
 records=scene['records'];cfg=load_config(ROOT/'configs/v4/live-bucket-fill-local-v1.json');assert cfg['occupancy_level_override']==.2
 estimator=MixedFillEstimator(cfg,[r['volume_m3'] for r in records],_cache_rock_vertices(Path(scene['root']),records))
 values=[]
 for f in range(clear-4,clear+1):
  st=states[f];pose=a.assemble(st['mechanism_qpos'],include_meshes=False)[1]['bucket']
  values.append(estimator.measure_snapshot(dict(frame=f,pose=pose,soil=particles[f],rocks=[r['qpos'] for r in st['rocks']])))
 result['retained_fill_measurements']=values
 if not all(v['valid_estimate'] for v in values):result['invalid_reason']='working_volume_boundary_or_estimator_invalid';return result
 acts=[json.loads(x) for x in (contact/'actions.ndjson').read_text().splitlines()];tails=[json.loads(x) for x in (tail/'actions.ndjson').read_text().splitlines()]
 if len(acts)!=len(crows) or len(tails)!=len(trows):result['invalid_reason']='action_count_mismatch';return result
 commands=np.array([x['applied_command_si'] for x in acts]+[x['command'] for x in tails[:clear]]);from contact_bc_inference import LIMITS
 limits=LIMITS
 result.update(eligible=True,working_fill_percent=float(np.median([v['fill_percent'] for v in values])),
  seconds=(len(crows)+clear)/60,command_total_variation=float(np.abs(np.diff(commands/limits,axis=0)).sum()),
  receipt_fraction=None if inv is None else inv['received_source_particles']/max(1,inv['source_particles']),
  unload_validated=inv is not None,
  retained_volume_scope='five CLEAR_PILE endpoint snapshots; soil+rock+heap occupancy0.2; not stable delivered volume',
  endpoint_sha256=hashlib.sha256((tail/'final.npz').read_bytes()).hexdigest())
 return result

def better(candidate,current):
 if not candidate.get('eligible'):return False
 if not current.get('eligible'):return True
 # Do not exchange a substantial receipt or fill loss for speed.
 if candidate.get('receipt_fraction') is not None and current.get('receipt_fraction') is not None and candidate['receipt_fraction']<current['receipt_fraction']-.02:return False
 delta=candidate['working_fill_percent']-current['working_fill_percent']
 if abs(delta)>2.:return delta>0
 if candidate['seconds']<current['seconds']*.98:return True
 if abs(candidate['seconds']-current['seconds'])<=current['seconds']*.02:return candidate['command_total_variation']<current['command_total_variation']
 return False
if __name__=='__main__':
 result=summarize(sys.argv[1],sys.argv[2]);Path(sys.argv[3]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
