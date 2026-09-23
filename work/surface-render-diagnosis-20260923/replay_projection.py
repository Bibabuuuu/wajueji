"""Bounded 65-frame diagnostic replay; frozen inputs stay read-only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import warp as wp
import newton
import run_layered_47_sparse_manual_contact_episode as manual
from newton._src.solvers.implicit_mpm.rasterized_collisions import Collider, collision_sdf

@wp.kernel
def query(points:wp.array[wp.vec3], ids:wp.array[int], collider:Collider,
          body_q:wp.array[wp.transform], body_qd:wp.array[wp.spatial_vector], body_prev:wp.array[wp.transform],
          dt:float, results:wp.array(dtype=float,ndim=2), colliders:wp.array[int]):
    i=wp.tid()
    sdf,n,v,cid,mid=collision_sdf(points[ids[i]],-2,collider,body_q,body_qd,body_prev,dt)
    results[i,0]=sdf
    results[i,1]=n[0];results[i,2]=n[1];results[i,3]=n[2]
    results[i,4]=v[0];results[i,5]=v[1];results[i,6]=v[2]
    colliders[i]=cid

root=Path('/home/qunwang/wajueji')
out=root/'work/surface-render-diagnosis-20260923'
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
repo=bundle/'repo'
start=root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'
run=root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-161451-732607'
trace=[json.loads(x) for x in (run/'physics.ndjson').read_text().splitlines()]
assert len(trace)==65
arrays=dict(np.load(start/'approach-handoff.npz'))
saved=dict(np.load(run/'final-000065.diagnostic.npz'))
selected=np.load(out/'patch-particles.npz')['ids'].astype('i')
base=json.loads((repo/'configs/v4/layered-47-excavation-preflight-v1.json').read_text())
manifest=json.loads((repo/base['physical_geometry']['manifest_path']).read_text())
_,identity=manual.motion.probe.select(47)
records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
proxy=json.loads((repo/base['open_bucket_proxy']['path']).read_text())
approach=json.loads((start/'report.json').read_text())
base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
args=argparse.Namespace(device='cuda:0',cxx='g++',
    mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),
    mechanism_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),
    controller_header_dir=str(repo/'evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))
wp.init();wp.set_device('cuda:0');newton.use_coord_layout_targets=True
events=[]
work=out/f'replay-build-{os.getpid()}';work.mkdir(exist_ok=False)
with manual.optimized_bridge(args.mechanism_header_dir,out/f'replay-optimization-{os.getpid()}'):
    fixture,held,restored=manual.motion.audit.restore_candidate_runtime(args,base,repo,records,arrays,proxy,approach['solver_profile'],work)
    bridge=manual.rigid_frequency.RigidFrequencyCommandBridge(held.bridge,4800)
    ids_wp=wp.array(selected,dtype=int)
    queried=wp.zeros((len(selected),7),dtype=float)
    collider_ids=wp.zeros(len(selected),dtype=int)
    original=fixture.solver.project_outside
    frame=[0]
    def instrument(state_in,state_out,dt,gap=None):
        before=state_in.particle_q.numpy()
        collider=fixture.solver._mpm_model.collider
        old_gap=collider.query_max_dist
        if gap is not None:collider.query_max_dist=gap
        wp.launch(query,dim=len(selected),inputs=[state_in.particle_q,ids_wp,collider,state_in.body_q,state_in.body_qd,None,dt,queried,collider_ids])
        query_values=queried.numpy();query_ids=collider_ids.numpy()
        collider.query_max_dist=old_gap
        original(state_in,state_out,dt,gap=gap)
        after=state_out.particle_q.numpy()
        delta=np.linalg.norm(after-before,axis=1)
        events.append({'frame':frame[0],'selected_before':before[selected].tolist(),'selected_after':after[selected].tolist(),'query':query_values.tolist(),'collider_ids':query_ids.tolist(),'max_delta':float(delta.max()),'max_selected_delta':float(delta[selected].max()),'selected_moved_gt_1cm':int((delta[selected]>.01).sum())})
    fixture.solver.project_outside=instrument
    try:
        for i,row in enumerate(trace,1):
            frame[0]=i
            bridge.command=np.asarray(row['applied_command_si'])
            manual.motion.probe.bench._advance_frame(fixture,bridge,base,i)
            if i%10==0:print('REPLAY',i,events[-1]['selected_moved_gt_1cm'],events[-1]['max_selected_delta'],flush=True)
        final=manual.motion.probe.snapshots.capture_arrays(fixture,bridge.state,float(arrays['simulation_time_s'][0])+65/60)
        comparisons={k:{'exact':bool(np.array_equal(v,saved[k])),'max_abs_difference':float(np.max(np.abs(v-saved[k])))} for k,v in final.items() if k in saved and v.dtype.kind in 'fiu' and v.size}
        result={'frames':65,'source_unchanged':manual.motion.probe.bench.sha256(start/'approach-handoff.npz')==approach['handoff_checkpoint']['sha256'],'selected_particle_ids':selected.tolist(),'bucket_collider_ids':fixture.bucket_collider_ids.tolist(),'comparisons':comparisons,'events':events}
        (out/'projection-replay.json').write_text(json.dumps(result))
        print(json.dumps({'frames':65,'source_unchanged':result['source_unchanged'],'comparisons':comparisons}),flush=True)
    finally:
        bridge.close()
