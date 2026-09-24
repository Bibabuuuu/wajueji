"""Bounded six-frame zero-command continuation; no policy or digging action."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from contact_phase_candidate import ContactPhaseConfig
from contact_phase_completion import CompletionConfig, FivePhaseCandidate
from contact_phase_geometry import SolidBucketGap
from layered_47_cycle_runtime import CycleRuntime
from run_layered_47_sparse_contact_handoff import wrench_summary

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'work/contact-phase-labels-20260924-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=int,choices=[1,3],required=True)
 p.add_argument('--output',required=True);p.add_argument('--frames',type=int,choices=[5,6],default=6);args=p.parse_args()
 prior_path=OLD/'phase-completion-replay-v1.json';prior=json.loads(prior_path.read_text());e=prior['episodes'][args.episode]
 source=Path(e['endpoint']['snapshot']['path']);feature=Path(e['feature_file']['path'])
 for item in e['source_inputs']+[e['feature_file'],e['endpoint']['snapshot']]:assert sha(Path(item['path']))==item['sha256']
 trace=json.loads(feature.read_text())['trace']
 cfg=CompletionConfig(**json.loads((OLD/'phase-completion-v1.json').read_text())['parameters'])
 router=FivePhaseCandidate(ContactPhaseConfig(**json.loads((OLD/'phase-candidate-v1.json').read_text())['parameters']),cfg,
          initial_mouth_angle_deg=trace[0]['mouth_chord_from_down_deg'])
 # Replay phase memory from the same episode, withholding all historical gaps.
 # This is a diagnostic history reconstruction, not arbitrary state restore.
 for f in trace[1:]:router.observe_end(frame_end=f['frame'],soil_force_n=f['soil_force_n'],rock_contact_substeps=f['rock_substeps'],
    mouth_angle_deg=f['mouth_chord_from_down_deg'],actual_boom_cylinder_m=f['q_boom'],bucket_origin_z_m=f['origin_z_m'])
 assert router.phase=='lift' and router.gap_streak==0
 base=trace[-1]['frame'];out=Path(args.output).resolve()
 contract_path=Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json')
 poly=np.array(json.loads(contract_path.read_text())['bucket']['polygon_xz_m']);local=np.c_[poly[:,0],np.zeros(len(poly)),poly[:,1]]
 report=dict(schema='phase_continuity_probe_v1',status='initializing',source=e['endpoint']['snapshot'],source_run=e['run'],
   feature_file=e['feature_file'],source_inputs=e['source_inputs'],prior_report_sha256=sha(prior_path),
   frames_requested=args.frames,source_frame_end=base,command_si=[0,0,0,0],training_eligible=False,
   qualification='Fresh end-state phase observer on short hold; not autonomous digging or earliest off-soil label',
   implementation_inputs=[dict(path=str(x),sha256=sha(x)) for x in (Path(__file__),ROOT/'src/v4/contact_phase_geometry.py',
    ROOT/'src/v4/contact_phase_completion.py',ROOT/'src/v4/contact_phase_candidate.py',ROOT/'src/v4/run_layered_47_sparse_contact_handoff.py',contract_path,OLD/'phase-completion-v1.json',OLD/'phase-candidate-v1.json')],frames=[])
 start=time.perf_counter()
 try:
  with CycleRuntime(source,out,allow_training_reset=True) as r:
   report['runtime_bindings']=r.bindings
   captured=r.capture();exact={k:bool(np.array_equal(captured[k],v)) for k,v in r.arrays.items()}
   report['zero_step_arrays_exact']=exact;report['restore']=r.restored
   assert all(exact.values()),'zero_step_arrays_inexact'
   assert max(r.restored['cycle_errors'].values())<=1e-12
   observer=SolidBucketGap(r.proxy)
   npz=out/'frame-000.npz';np.savez_compressed(npz,particle_q=r.arrays['particle_q'],mechanism_qpos=r.arrays['mechanism_qpos'])
   report['initial_observation_snapshot']=dict(path=str(npz),sha256=sha(npz))
   report['status']='running';dump(out/'phase-audit.json',report)
   print('RESTORE_EXACT',e['run'],len(exact),'arrays',flush=True)
   for step in range(1,args.frames+1):
    if time.perf_counter()-start>240:raise RuntimeError('bounded_probe_wallclock_budget')
    t=time.perf_counter();row=r.step([0,0,0,0]);physics_s=time.perf_counter()-t
    t=time.perf_counter();q=r.fixture.state.particle_q.numpy();mech=np.array(r.bridge.state['mechanism_qpos'])
    pose=r.assembly.assemble(mech,include_meshes=False)[1]['bucket']
    before=hashlib.sha256(q.tobytes()).hexdigest();observation=observer.observe(q,pose)
    assert hashlib.sha256(q.tobytes()).hexdigest()==before
    observe_s=time.perf_counter()-t
    world=local@pose[:3,:3].T+pose[:3,3];radial=np.array([pose[1,1],-pose[0,1]]);radial/=np.linalg.norm(radial)
    chord=world[-1]-world[0];angle=float(np.rad2deg(np.arctan2(chord[:2]@radial,-chord[2])))
    n=base+step;gap=observation['remaining_soil_gap_m'];old_phase=router.phase_for_decision(n)
    force=wrench_summary(r.captured['wrenches']['bucket'])['bucket_mpm_force_norm_n']
    phase=router.observe_end(frame_end=n,soil_force_n=force,rock_contact_substeps=row['rock_bucket_contact_substeps'],
      mouth_angle_deg=angle,actual_boom_cylinder_m=float(mech[9]),bucket_origin_z_m=float(pose[2,3]),
      remaining_soil_gap_m=gap,gap_frame_end=n if gap is not None else None)
    assert phase['phase_before_observation']==old_phase
    snap=out/f'frame-{step:03d}.npz';np.savez_compressed(snap,particle_q=q,mechanism_qpos=mech)
    report['frames'].append(dict(local_frame=step,frame_end=n,simulation_time_s=float(r.arrays['simulation_time_s'][0])+step/60,
      snapshot=dict(path=str(snap),sha256=sha(snap)),observation=observation,phase=phase,
      bucket_mpm_force_norm_n=force,hard_guard=row['hard_guard'],physics_wall_s=physics_s,observer_including_copy_pose_hash_wall_s=observe_s))
    dump(out/'phase-audit.json',report)
    print('FRAME',step,'gap',gap,'next phase',phase['phase_for_next_decision'],'observer_s',round(observe_s,4),flush=True)
   report['final_checkpoint']=r.save(out/'final.npz',stage='hold')
   report.update(status='bounded_hold_completed',frames_completed=r.frames,final_phase=router.phase)
 except BaseException as error:
  report.update(status='failed',error=repr(error))
  raise
 finally:
  report['wall_s']=time.perf_counter()-start
  report['source_unchanged']=sha(source)==e['endpoint']['snapshot']['sha256']
  if out.exists():dump(out/'phase-audit.json',report)

if __name__=='__main__':main()
