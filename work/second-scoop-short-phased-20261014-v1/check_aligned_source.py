"""Read-only qualification candidate for a later second-dig restore proposal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-short-phased-20261014-v1';sys.path.insert(0,str(W))
import run_phased_short as m
import numpy as np

def main():
 D=m.OUT;p=m.read(D/'report.json');checks={}
 def check(n,v):checks[n]=bool(v)
 phase=next(x for x in p['phases'] if x['name']=='REFERENCE_ALIGN');frame=phase['end_frame'];source=D/'reference_align.npz';meta=m.read(str(source)+'.json');ctrl=m.read(D/'reference_align.control.json')
 states=m.read(D/'display/states.json');cloud=np.load(D/'display/particles.npy',mmap_mode='r')
 with np.load(source) as z:a={k:z[k].copy() for k in z.files}
 check('source24_arrays',len(a)==24)
 check('saved_checkpoint_sha',m.sha(source)==meta['checkpoint_sha256']==ctrl['checkpoint_sha256'])
 check('prepoint_cycle2_not_training_reset',meta['stage']=='prepoint' and meta['cycle_index']==2 and not meta.get('training_reset'))
 check('actual_alignment_frame',ctrl['frame']==frame and ctrl['phase']=='REFERENCE_ALIGN')
 check('all_saved_particles_equal_cache',np.array_equal(a['particle_q'],cloud[frame]))
 check('mechanism_qpos_and_qvel_equal_cache',np.array_equal(a['mechanism_qpos'],states[frame]['mechanism_qpos']) and np.array_equal(a['mechanism_qvel'],states[frame]['mechanism_qvel']))
 check('rocks_equal_cache',np.array_equal(a['rigid_qpos'],[rock['qpos'] for rock in states[frame]['rocks']]))
 check('physical_time_preserved',abs(float(a['simulation_time_s'][0])-p['initial_simulation_time_s']-frame/60)<1e-10)
 check('native_targets_equal_cache',np.array_equal(meta['controller_state']['controller_target'],states[frame]['controller_target']) and np.array_equal(meta['controller_state']['controller_target_velocity'],states[frame]['controller_target_velocity']))
 d=json.loads((D/'policy-decisions.ndjson').read_text().splitlines()[0]);check('actual_frozen_policy_entry_same_frame',d['global_frame_before']==frame and d['local_frame_before']==0 and d['observation'][85]==0)
 check('not_automatically_admitted_for_restore',ctrl['continuation_qualified'] is False)
 result=dict(status='cpu_source_candidate_passed' if all(checks.values()) else 'failed',checks=checks,check_count=len(checks),source=str(source),source_sha256=m.sha(source),metadata_sha256=m.sha(str(source)+'.json'),control_sha256=m.sha(D/'reference_align.control.json'),
  actual_frame=frame,simulation_time_s=float(a['simulation_time_s'][0]),yaw_offset_rad=p['yaw_offset_rad'],
  physics_steps=0,policy_predictions=0,training_updates=0,physical_restore_verified=False,
  limitation='Only snapshot/cache identity checked. A new native zero restore and first3-frame comparison to the saved SAC prefix require separate physical authorization. This is not automatic resumption of the partial120-frame endpoint.')
 m.write(W/'aligned-source-candidate.json',result);print(m.encode(result),flush=True);assert all(checks.values())
if __name__=='__main__':main()
