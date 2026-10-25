"""Recheck saved D194 paths and provenance without simulator entry."""
from pathlib import Path
import ast,json,sys,time
import numpy as np
W=Path('/home/qunwang/wajueji/work/motion-review-20261015-v1');sys.path.insert(0,str(W))
from motion_candidate import *
checks=[]
def check(name,passed):checks.append(dict(check=name,passed=bool(passed)));assert passed,name
def main():
 t=time.monotonic();O=W/'audit-v1';O.mkdir(exist_ok=False);report=read(W/'return-candidate-v3/report.json');carry=read(W/'candidate-v1/report.json')['carry_reference'];source=Path(report['source'])
 check('candidate geometry only',report['status']=='cpu_return_geometry_passed' and not report['physical_admission'])
 check('source checkpoint unchanged',sha(source)==report['source_sha256'])
 check('all original data identities preserved',all(sha(Path(p))==h for p,h in read(R/'runs/roadshow-two-scoop-media-20261015-v1/data-manifest.json')['sources_sha256'].items()))
 protocol=read(R/'work/edge-adjacent-59um-20261015-v1/live-protocol-continuation.json')
 check('original physics policy controller bindings unchanged',all(sha(Path(p))==h for p,h in protocol['bindings_sha256'].items()))
 r=CpuRuntime(source,O/'runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 spec=importlib.util.spec_from_file_location('audit_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
 terrain,observation=observer.stationary_terrain(r,r.arrays)
 first=read(source.parent/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy'];ranked,_=rank_adjacent(terrain,first,inward_sign=-1);selected=next(x for x in ranked if abs(x['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m']))
 check('fresh post-unload candidate recomputed',selected==report['selected'])
 check('fresh ground-connected observation exact',observation==report['fresh_observation'])
 with np.load(W/'return-candidate-v3/return-plan.npz') as z:q=z['qpos'].copy();v=z['velocity'].copy();reference=z['reference_q'].copy()
 axes=r.chain.axes;lengths=[p['steps'] for p in report['segments']];cuts=np.cumsum([0,*lengths]);last_start=cuts[-2]
 check('complete saved path',q.shape==(len(v)+1,19) and v.shape==(sum(lengths),4) and np.isfinite(q).all() and np.isfinite(v).all())
 check('unchanged actual driven start',np.array_equal(q[0,axes],r.arrays['mechanism_qpos'][axes]))
 check('position integrates recorded reference velocities',np.max(abs(np.diff(q[:,axes],axis=0)*60-v))<1e-8)
 check('original speed ceilings',np.max(abs(v)/SPEED)<=1+1e-8)
 check('reference slew within original command limit',np.max(abs(np.diff(np.vstack([np.zeros(4),v,np.zeros(4)]),axis=0))*60/(2*COMMAND_LIMITS))<=1+1e-8)
 checker=ReturnClearance(r,terrain);minimum={};full_roi_samples=0
 for i,pose in enumerate(q):
  d=checker.check(pose)
  for key,value in d.items():
   if value is not None:minimum[key]=min(minimum.get(key,float('inf')),value)
  if i>=last_start:
   _,b=r.assembly.assemble(pose,include_meshes=False);p=checker.samples@b['bucket'][:3,:3].T+b['bucket'][:3,3]
   gate=sampled_surface_clearance(p,checker.terrain)
   if gate['clearance_m']<.32:raise ValueError('full_roi_final_approach32cm')
   full_roi_samples+=1
 check('all stored poses original CAD bounds closure terrain gates',True)
 check('full ROI original32cm final approach',full_roi_samples==len(q)-last_start)
 check('terminal original positive margin',checker.full_roi_admission(q[-1])['clearance_m']>=.325)
 poses=[r.assembly.assemble(p,include_meshes=False)[1]['bucket'] for p in reference];alignment=ProgressAlignment(reference,poses).observe(q[-1],r.assembly.assemble(q[-1],include_meshes=False)[1]['bucket'],0.)
 check('frozen policy original reference corridor',alignment['trusted'])
 check('only two intermediate waypoints',report['intermediate_waypoints']==2 and len(lengths)==3)
 check('no old fixed safe-return loop',report['segment_labels']==['unloaded_exit','coordinated_outward','reference_entry'])
 check('time derives from saved reference frames',abs(report['nominal_reference_seconds']-len(v)/60)<1e-12)
 with np.load(W/'candidate-v1/carry-reference.npz') as z:cq=z['qpos'].copy();cv=z['velocity'].copy()
 check('carry held cylinder and zero active command',np.max(abs(cq[:,15]-carry['cylinder_m']))<1e-8 and np.count_nonzero(cv[:,3])==0)
 check('carry speed and acceleration bounded',np.max(abs(cv)/SPEED)<=1+1e-8 and np.max(abs(np.diff(np.vstack([np.zeros(4),cv,np.zeros(4)]),axis=0))*60/ACCEL)<=1+1e-8)
 control=read(W/'control-handoff-cpu-v2/report.json');check('nine handoff checks passed',control['status']=='cpu_control_handoff_passed' and len(control['checks'])==9 and all(control['checks'].values()))
 for name in ['motion_candidate.py','return_candidate_v3.py','control_handoff_cpu.py']:
  tree=ast.parse((W/name).read_text());attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
  check(name+'no native or learning calls',not set(attrs)&{'step','predict','reset','train','learn','__enter__'})
 check('no native entry',r.bridge is None and r.frames==0)
 check('source unchanged after audit',sha(source)==report['source_sha256'])
 old_v=np.load(source.parent/'transport-plan/plan.npz')['CLEAR_PILE_velocity'];accel=lambda x:np.max(abs(np.diff(np.vstack([np.zeros(4),x,np.zeros(4)]),axis=0))*60,axis=0)
 result=dict(status='cpu_candidate_audit_passed',checks=checks,checks_passed=len(checks),saved_return_poses=len(q),final_full_roi_poses=full_roi_samples,minimum_clearance=minimum,nominal_return_s=len(v)/60,old_actual_after_unload_to_next_dig_s=report['original_after_unload_to_next_dig_s'],time_comparison='Reference duration versus old physical duration, not a measured speedup.',carry_old_reference_s=len(old_v)/60,carry_new_reference_s=len(cv)/60,carry_old_max_discrete_acceleration_si=accel(old_v).tolist(),carry_new_max_discrete_acceleration_si=accel(cv).tolist(),handoff_checks=control['checks'],physics_steps=0,native_initializations=0,inference_calls=0,training_updates=0,physical_admission=False,wall_seconds=time.monotonic()-t)
 write(O/'report.json',result);print(json.dumps({k:result[k] for k in ['status','checks_passed','nominal_return_s','carry_old_reference_s','carry_new_reference_s','minimum_clearance','wall_seconds']},indent=2))

if __name__=='__main__':main()
