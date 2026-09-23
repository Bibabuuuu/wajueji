"""D-127 bounded opt-in candidate: original prepoint and 600 actual commands."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import warp as wp
import warp.fem as fem
import newton
import run_layered_47_sparse_manual_contact_episode as manual
import layered_mound_checkpoint_runtime as runtime
from soil_hencky_update_candidate import installed_candidate
from observer import observe

root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1');repo=bundle/'repo'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((out/'protocol.json').read_text())
assert all(sha(p)==h for p,h in protocol['bindings'].items()),'binding_changed'
assert all(json.loads((out/'preflight-report.json').read_text())['checks'].values())
trace=[json.loads(l) for l in (run/'physics.ndjson').read_text().splitlines()][:600]
arrays=dict(np.load(start/'approach-handoff.npz'))
base=json.loads((repo/'configs/v4/layered-47-excavation-preflight-v1.json').read_text())
manifest=json.loads((repo/base['physical_geometry']['manifest_path']).read_text())
_,identity=manual.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
proxy=json.loads((repo/base['open_bucket_proxy']['path']).read_text())
approach=json.loads((start/'report.json').read_text())
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
args=argparse.Namespace(device='cuda:0',cxx='g++',mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),mechanism_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),controller_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
_,chain=manual._load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
projector=manual.ContactCommandProjector(chain)
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
work=out/f'build-{os.getpid()}';work.mkdir(exist_ok=False)
frame=0;bridge=None;result={'status':'initializing','frames':0};clock_start=time.monotonic()
with manual.optimized_bridge(args.mechanism_header_dir,out/f'optimization-{os.getpid()}'):
    original_render=runtime.render_capacity_bridge
    frozen=(run/'bucket-integral-build/capacity-bridge.cc').read_text()
    def render_frozen(*args):
        _,metadata=original_render(*args)
        return frozen,{**metadata,'diagnostic_frozen_original_bridge':True}
    runtime.render_capacity_bridge=render_frozen
    try:
        fixture,held,restored=manual.motion.audit.restore_candidate_runtime(args,base,repo,records,arrays,proxy,approach['solver_profile'],work)
        bridge=manual.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
        n=len(arrays['particle_q']);data=wp.zeros((n,28),dtype=float);rejection=wp.zeros(n,dtype=int)
        def capture():return manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+frame/60)
        def save(label):
            a=capture()
            path=out/f'{label}-{frame:06d}.npz'
            manual.motion.probe.snapshots.write_rejected_diagnostic_atomic(path,a,'D127 experimental algorithm; diagnostic only, never resume/train')
            np.savez_compressed(out/f'{label}-material-input-{frame:06d}.npz',data=data.numpy(),rejection=rejection.numpy())
            return str(path)
        # Check complete continuation before installing the opt-in candidate.
        initial=capture();exact={k:bool(np.array_equal(initial[k],v)) for k,v in arrays.items() if k in initial}
        assert all(exact.values()),exact
        (out/'restore-verification.json').write_text(json.dumps({'restore_exact':restored['exact'],'captured_fields_exact':exact},indent=2)+'\n')
        real_update=fixture.solver._update_particles
        def observed_update(state_in,state_out,dt,pic,scratch):
            data.zero_();rejection.zero_();mpm=fixture.solver._mpm_model
            fem.interpolate(observe,at=pic,fields={'elastic':scratch.elastic_strain_delta_field,'velocity':scratch.velocity_field},
                values={'dt':dt,'flags':mpm.material_particle_flags,'density':mpm.particle_density,
                    'volume':mpm.material_particle_volume,'previous':state_in.mpm.particle_elastic_strain,'data':data,'rejection':rejection},temporary_store=fixture.solver.temporary_store)
            return real_update(state_in,state_out,dt,pic,scratch)
        fixture.solver._update_particles=observed_update
        material_peaks={'max_F_frobenius':0.,'max_Jp':0.,'rejected_particles':0}
        rejection_reason=None;last=None;physics_start=time.monotonic()
        with installed_candidate(), (out/'trace.ndjson').open('w') as log:
            for source in trace:
                frame=source['frame'];bridge.command=np.asarray(source['applied_command_si'])
                row,_=manual.motion.probe.bench._advance_frame(fixture,bridge,base,frame)
                a=capture();codes=rejection.numpy()
                nonfinite={k:int((~np.isfinite(v)).sum()) for k,v in a.items() if v.dtype.kind in 'fc' and not np.isfinite(v).all()}
                F=a['particle_elastic_strain'];Fnorm=np.linalg.norm(F,axis=(1,2));Jp=a['particle_Jp']
                material={'nonfinite_fields':nonfinite,'rejection_count':int(np.count_nonzero(codes)),
                    'rejection_codes':{str(k):int((codes==k).sum()) for k in [1,2,3]},'max_F_frobenius':float(np.max(Fnorm)),
                    'F_frobenius_over1000':int((Fnorm>1000).sum()),'Jp_nonpositive':int((Jp<=0).sum()),
                    'max_Jp':float(np.max(Jp)) if np.isfinite(Jp).all() else None,'min_Jp':float(np.min(Jp)) if np.isfinite(Jp).all() else None}
                guards={'all_arrays_finite':not nonfinite,'no_candidate_update_rejection':material['rejection_count']==0,
                    'no_extreme_elastic_F':material['max_F_frobenius']<=1000,'Jp_positive':material['Jp_nonpositive']==0,
                    'position_velocity_and_bridge_finite':bool(row['finite'] and manual.motion.probe.bench._finite_tree(bridge.state)),
                    'mechanism_hard_bounds':projector.physical_bounds_ok(bridge.state['mechanism_qpos']),
                    'closure':row['closure_m']<=.001,'no_forbidden_contacts':row['forbidden_contact_count']==0,
                    'no_unmapped_colliders':row.get('unmapped_collider_sample_count',0)==0,
                    'finite_colliders':row.get('nonfinite_collider_sample_count',0)==0,
                    'particle_flags_unchanged':bool(np.array_equal(a['particle_flags'],arrays['particle_flags']))}
                if frame==1 or frame%30==0 or not all(guards.values()):
                    d=data.numpy();sv=np.linalg.svd(F.astype(float),compute_uv=False) if np.isfinite(F).all() else None
                    material['F_singular_min_max']=None if sv is None else [float(sv.min()),float(sv.max())]
                    material['F_singular_max_over2']=None if sv is None else int((sv[:,0]>2).sum())
                    material['weight_min_max']=[float(d[:,0].min()),float(d[:,0].max())]
                    H=d[:,19:28].reshape(n,3,3);S=a['particle_stress'].astype(float)
                    compliance=((1+.3)*S-.3*np.trace(S,axis1=1,axis2=2)[:,None,None]*np.eye(3))/2e6
                    residual=np.linalg.norm(H+compliance,axis=(1,2))
                    material['local_H_stress_closure_norm_p50_p99_max']=np.quantile(residual,[.5,.99,1]).tolist()
                row.update(material_audit=material,guards=guards,after_state=manual.compact_state(bridge.state),applied_command_si=source['applied_command_si'])
                log.write(json.dumps(row)+'\n');log.flush();last=row
                material_peaks['max_F_frobenius']=max(material_peaks['max_F_frobenius'],material['max_F_frobenius'])
                material_peaks['max_Jp']=max(material_peaks['max_Jp'],material['max_Jp'] or 0.)
                material_peaks['rejected_particles']=max(material_peaks['rejected_particles'],material['rejection_count'])
                if not all(guards.values()):
                    rejection_reason=[k for k,v in guards.items() if not v];save('rejected');break
                if frame in (277,540,600):save('candidate')
                if frame%30==0:print('FRAME',frame,'Fmaxnorm',material['max_F_frobenius'],'Jpmax',material['max_Jp'],flush=True)
        result={'status':'candidate_short_prefix_passed' if frame==600 and rejection_reason is None else 'candidate_rejected',
            'frames':frame,'rejection_reason':rejection_reason,'restore_exact':restored['exact'],'initial_arrays_exact':exact,
            'material_peaks':material_peaks,'last_material_audit':last['material_audit'],'last_guards':last['guards'],
            'source_commands':'original actual commands; full coupled mechanics feedback; no original-state equality expected after frame1',
            'wall_seconds':time.monotonic()-clock_start,'physics_and_audit_wall_seconds':time.monotonic()-physics_start,
            'soil_loss_fixed':False,'manual_deployed':False,'training_eligible':False,'resume_qualified':False}
    except BaseException as error:
        failure_snapshot=None
        if bridge is not None:
            try:failure_snapshot=save('failure')
            except BaseException as save_error:result['save_failure']=repr(save_error)
        result.update(status='execution_failed',frames=frame,error=repr(error),traceback=traceback.format_exc(),failure_snapshot=failure_snapshot)
        print(result['traceback'],flush=True)
    finally:
        if bridge is not None:bridge.close()
        runtime.render_capacity_bridge=original_render
        result['bindings_unchanged']=all(sha(p)==h for p,h in protocol['bindings'].items())
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print('DONE',result['status'],frame,flush=True)
        if result['status']=='execution_failed':raise SystemExit(1)
