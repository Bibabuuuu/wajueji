"""D196 bounded physical pair; no policy load, training, reset or automatic retry."""
from pathlib import Path
import importlib.util,json,sys,time,traceback,numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/motion-physical-short-20261015-v2';C=R/'work/motion-review-20261015-v1';OUT=R/'runs/motion-physical-short-20261015-v2'
sys.path[:0]=[str(C),str(R/'work/continuous-single-scoop-20261014-v1'),str(R/'work/edge-adjacent-59um-20261015-v1')]
from continuous_scoop import *
from motion_candidate import smooth_path,ReturnClearance,locked_lift_pose,ContactGeometryFeedback,FeedbackConfig,SPEED,COMMAND_LIMITS
from inplace_cache import InPlaceDisplayCache
from layered_47_cycle_geometry import bucket_mask
W=R/'work/motion-physical-short-20261015-v2';C=R/'work/motion-review-20261015-v1';OUT=R/'runs/motion-physical-short-20261015-v2'

class Budget:
 def __init__(self):
  self.data=read(W/'budget.json');assert not (W/'closure.json').exists();assert self.data['seconds']==1800 and self.data['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 def remaining(self):return min(self.data['epoch_deadline']-time.time(),self.data['monotonic_deadline']-time.monotonic())
 def check(self):
  if self.remaining()<=300:raise TimeoutError('save_audit_reserve')

class BoundRuntime(CycleRuntime):
 def __init__(self,case,budget,out):
  budget.check();self.budget=budget;self.case=case;self.authorized=False;self.prefix_verified=False
  assert sha(case['source'])==case['source_sha256'] and sha(case['source']+'.json')==case['metadata_sha256']
  super().__init__(case['source'],out,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
 def step(self,command):
  self.budget.check()
  if not self.authorized or (self.frames>=3 and not self.prefix_verified):raise ValueError('native_restore_gate')
  if self.frames>=self.case['max_frames']:raise ValueError('case_frame_ceiling')
  return super().step(command)

def main():
 budget=Budget();budget.check();protocol=read(W/'preflight.json');assert protocol['status']=='preflight_passed' and sha(W/'budget.json')==protocol['budget_sha256']
 for p,h in protocol['bindings_sha256'].items():assert sha(p)==h,('binding_changed',p)
 OUT.mkdir(exist_ok=False);report=dict(status='starting',cases=[],physical_frames=0,native_initializations=0,inference_calls=0,training_updates=0,budget_sha256=sha(W/'budget.json'));write(OUT/'report.json',report)
 def persist():write(OUT/'report.json',report)
 try:
  for case in protocol['cases']:
   budget.check();folder=OUT/case['name'];folder.mkdir();r=None;cache=None;phase='RESTORE';previous=np.zeros(4);labels=[];baseline=None
   item=dict(name=case['name'],source=case['source'],source_sha256=case['source_sha256'],status='restoring',frames=0,phases=[],prefix_checks=[]);report['cases'].append(item);persist()
   try:
    with BoundRuntime(case,budget,folder/'physics') as r:
     report['native_initializations']+=1
     actual=r.capture();exact={k:np.array_equal(v,r.arrays[k]) for k,v in actual.items()};dt=abs(float(actual['simulation_time_s'][0])-float(r.arrays['simulation_time_s'][0]))
     assert all(v for k,v in exact.items() if k!='simulation_time_s') and dt<=1e-12 and max(r.restored['cycle_errors'].values())<=1e-12,'native_zero_restore_mismatch'
     item.update(zero_arrays_exact=exact,zero_time_error_s=dt,native_restore_errors=r.restored['cycle_errors'])
     expected_states=read(Path(case['prefix'])/'states.json');expected_cloud=np.load(Path(case['prefix'])/'particles.npy',mmap_mode='r')
     assert manual.compact_state(r.bridge.state)==expected_states[0] and np.array_equal(r.fixture.state.particle_q.numpy(),expected_cloud[0]),'source_zero_cache_mismatch'
     cache=InPlaceDisplayCache(folder/'display',case['max_frames'],r.arrays['particle_q'].shape,dict(records=r.records,root=str(r.root),selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m']))
     cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy());labels.append('INITIAL');raw=r.step
     def tracked(command):
      nonlocal previous
      manual.append_json(folder/'dispatch.ndjson',dict(frame=r.frames+1,phase=phase,command=np.asarray(command)))
      try:row=raw(command)
      finally:item['frames']=r.frames;report['physical_frames']=sum(x['frames'] for x in report['cases'])
      previous=np.array(command,copy=True);cloud=r.fixture.state.particle_q.numpy();cache.capture(r.frames,row['after_state'],cloud);labels.append(phase)
      if not row['hencky_material_guard']['safe'] or not row['moving_bucket_sweep']['safe']:raise ValueError('original_material_or_sweep_guard')
      if r.frames<=3:
       gate=dict(frame=r.frames,command_exact=np.array_equal(command,case['commands'][r.frames-1]),particles_exact=np.array_equal(cloud,expected_cloud[r.frames]),state_exact=row['after_state']==expected_states[r.frames]);item['prefix_checks'].append(gate)
       assert all(gate[k] for k in ['command_exact','particles_exact','state_exact']),'original_prefix3_mismatch'
       if r.frames==3:r.prefix_verified=True
      if r.frames%120==0:print(case['name'],r.frames,phase,'remaining',round(budget.remaining()),flush=True);persist()
      return row
     r.step=tracked;r.authorized=True;phase='RESTORE_PREFIX3'
     for cmd in case['commands']:r.step(cmd)
     assert r.prefix_verified;item['prefix_checkpoint']=r.save(folder/'prefix3.npz',stage='transport');persist();print('PREFIX3_EXACT',case['name'],flush=True)
     q=np.asarray(r.bridge.state['mechanism_qpos']);view=PhaseView(r);segments=[]
     if case['name']=='loaded':
      fixed=FixedRouteClearance(view);start=project(view,q);held=float(start[15]);ground=BucketGroundGuard(r.assembly,q);mouth,_=tool_pose(r.assembly,q,MOUTH)
      fb=ContactGeometryFeedback(r.assembly,r.chain,ground,dict(entry_xy_m=mouth[:2],surface_z_m=mouth[2]),q,FeedbackConfig(late_curl_no_down=True))
      target=read(Path(case['original_run'])/'report.json')['transport_plan']['endpoint_solver_calls'][0]['candidate_target_m'];goal,_=locked_lift_pose(r.assembly,r.chain,start,target,MOUTH)
      class CarryCheck:
       def check(self,pose,old=None):
        d,_=fixed.measure_clearance(pose)
        if d['bucket_truck_m']<.04 or d['mechanism_truck_m']<.02 or d['mechanism_ground_m']<.02 or d['closure_m']>.0001:raise ValueError('original_carry_static_gate')
        if abs(pose[15]-held)>1e-8:raise ValueError('held_cylinder_changed')
        if old is not None and np.mean(fb.body_heights(pose)-fb.body_heights(old))<-1e-6:raise ValueError('original_carry_direction')
        return d
      lift=smooth_path(view,start,goal,CarryCheck(),max_steps=2400);segments.append(('CLEAR_PILE',lift,30));checker=fixed
      old_align=np.load(Path(case['original_run'])/'transport-plan/plan.npz');swing=goal.copy();swing[7]=old_align['ALIGN_qpos'][-1,7];swing=project(view,swing);align=smooth_path(view,goal,swing,fixed,max_steps=2400)
      align=dict(align,qpos=align['qpos'][:121],velocity=align['velocity'][:120],steps=120);segments.append(('ALIGN',align,0))
      pose=r.assembly.assemble(q,include_meshes=False)[1]['bucket'];baseline=bucket_mask(r.fixture.state.particle_q.numpy(),pose);item['initial_cavity_particles']=int(baseline.sum())
     else:
      spec=importlib.util.spec_from_file_location('physical_return_observer',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)
      terrain,obs=observer.stationary_terrain(view,view.arrays);old=read(C/'return-candidate-v3/report.json');first=read(Path(case['original_run'])/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy']
      from adjacent_selection import rank_adjacent
      ranked,_=rank_adjacent(terrain,first,inward_sign=-1);selected=next(x for x in ranked if abs(x['adjacency']['nominal_entry_boundary_gap_m'])<=float(terrain['resolution_m']))
      assert selected['entry_xy_m']==old['selected']['entry_xy_m'],'prefix_changed_selected_candidate'
      checker=ReturnClearance(view,terrain);data=np.load(C/'return-candidate-v3/return-plan.npz');item.update(fresh_observation=obs,selected=selected);reference=data['reference_q'].copy();offset=0
      for i,s in enumerate(old['segments']):
       n=s['steps'];goal=data['qpos'][offset+n].copy();start=project(view,q) if i==0 else data['qpos'][offset].copy();path=smooth_path(view,start,goal,checker) if i==0 else dict(qpos=data['qpos'][offset:offset+n+1],velocity=data['velocity'][offset:offset+n],steps=n)
       segments.append((['RETURN_EXIT','RETURN_COORDINATED','RETURN_REFERENCE'][i],path,30));offset+=n
     item['planned_frames_with_holds']=sum(len(p['velocity'])+hold for _,p,hold in segments)+3;assert item['planned_frames_with_holds']<=case['max_frames']
     np.savez(folder/'executed-reference.npz',**{name+'_qpos':path['qpos'] for name,path,hold in segments},**{name+'_velocity':path['velocity'] for name,path,hold in segments})
     controller=TransportController(PhaseView(r),checker,previous)
     for name,path,hold in segments:
      phase=name;begin=r.frames;n=len(path['velocity'])
      for desired,velocity in list(zip(path['qpos'][1:],path['velocity'],strict=True))+[(path['qpos'][-1],np.zeros(4))]*hold:
       before=np.asarray(r.bridge.state['mechanism_qpos']);controller.check(before,name,physical=True)
       if case['name']=='return':checker.check(before,physical=True)
       cmd,detail=controller.command(name,before,r.bridge.state['controller_target'],desired,velocity)
       manual.append_json(folder/'actions.ndjson',dict(frame=r.frames+1,phase=name,before_state=manual.compact_state(r.bridge.state),desired=desired,velocity=velocity,command=cmd,detail=detail))
       row=r.step(cmd);actual=np.asarray(row['after_state']['mechanism_qpos']);controller.check(actual,name,physical=True)
       measured=checker.check(actual,before,physical=True) if case['name']=='return' else controller.check(actual,name,physical=True)
       error=float(np.max(abs(actual[r.chain.axes]-desired[r.chain.axes])));manual.append_json(folder/'clearance.ndjson',dict(frame=r.frames,phase=name,axis_error=error,**measured))
       if error>.025:raise ValueError('original_tracking25mm:'+str(error))
      endpoint=float(np.max(abs(np.asarray(r.bridge.state['mechanism_qpos'])[r.chain.axes]-path['qpos'][-1][r.chain.axes])))
      if hold and endpoint>.015:raise ValueError('original_endpoint15mm:'+str(endpoint))
      item['phases'].append(dict(name=name,start_frame=begin,end_frame=r.frames,reference_frames=n,hold_frames=hold,endpoint_axis_error=endpoint));item['latest_checkpoint']=r.save(folder/(name.lower()+'.npz'),stage='transport');persist()
     if case['name']=='return':
      q=np.asarray(r.bridge.state['mechanism_qpos']);poses=[r.assembly.assemble(p,include_meshes=False)[1]['bucket'] for p in reference];admit=ProgressAlignment(reference,poses).observe(q,r.assembly.assemble(q,include_meshes=False)[1]['bucket'],0.);item['final_alignment']=admit;item['full_roi_admission']=checker.full_roi_admission(q);assert admit['trusted'],'actual_reference_corridor_failed'
     else:
      pose=r.assembly.assemble(np.asarray(r.bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket'];mask=bucket_mask(r.fixture.state.particle_q.numpy(),pose);item.update(final_cavity_particles=int(mask.sum()),initial_cohort_still_in_cavity=int((mask&baseline).sum()),particle_measurement_scope='Cavity particle cohort diagnostic only; not calibrated retained fill or proof of no spill.')
     item.update(status='completed',final_checkpoint=r.save(folder/'final.npz',stage='transport'));persist()
   except BaseException as exc:
    item.update(status='failed',failure=repr(exc),traceback=traceback.format_exc());raise
   finally:
    if cache is not None:
     try:item['display']=cache.finish(item['status']);write(folder/'display/phases.json',labels)
     except BaseException as e:item['display_error']=repr(e)
    item['frames']=r.frames if r else 0;report['physical_frames']=sum(x['frames'] for x in report['cases']);persist()
  report['status']='both_short_cases_completed'
 except BaseException as exc:report.update(status='stopped',failure=repr(exc),traceback=traceback.format_exc())
 finally:
  report.update(finished_epoch=time.time(),remaining_budget_seconds=budget.remaining());persist();write(W/'physical-finished.json',dict(status=report['status'],frames=report['physical_frames'],finished_epoch=time.time()));print(encode({k:report.get(k) for k in ['status','failure','physical_frames','remaining_budget_seconds']}),flush=True)

if __name__=='__main__':main()
