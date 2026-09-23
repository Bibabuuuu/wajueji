#!/usr/bin/env python3
"""Bounded physical refinement until a saved prepoint has 30 zero-contact frames.

No contact threshold or physical state is edited. The original candidate remains
immutable. Small outward/upward moves address sparse-grid lateral support.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from layered_47_cycle_runtime import CycleRuntime, REPO
from layered_47_cycle_geometry import stationary_terrain
from planning.cycle_geometry import TIP, solve_pose
from planning.approach import sampled_approach
from planning.rear_lower_prepoint import (proxy_surface_samples, ground_completed_terrain,
    sampled_surface_clearance, mechanism_and_truck_clearance)
from run_layered_47_sparse_manual_contact_episode import _clean


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output).resolve()
    r=CycleRuntime(args.source,out)
    if r.metadata['stage']!='prepoint' or r.metadata['cycle_index'] not in (2,3):
        raise ValueError('qualification_requires_continuous_prepoint')
    report=dict(schema='layered_47_cycle_prepoint_qualification_v1',status='running',
        source=r.metadata['source_identity'],source_metadata=r.metadata['source_metadata_identity'],
        training_eligible=False,maximum_frames=1200,maximum_wall_seconds=360,attempts=[],
        maximum_refinement_bucket_force_n=0.)
    started=time.monotonic()
    def write():(out/'report.json').write_text(json.dumps(_clean(report),indent=2,allow_nan=False)+'\n')
    write()
    try:
        terrain,observation=stationary_terrain(r)
        terrain=ground_completed_terrain(terrain)
        samples=proxy_surface_samples(r.proxy)
        geometries,bodies=r.assembly.assemble(r.arrays['mechanism_qpos'])
        pose=bodies['bucket'];initial_tip=pose[:3,:3]@TIP+pose[:3,3]
        radial=initial_tip[:2]/np.linalg.norm(initial_tip[:2])
        with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:
            truck=data['world_vertices'].copy()
        def check(q,previous=None):
            bodies=r.assembly.assemble(q,include_meshes=False)[1];pose=bodies['bucket']
            bucket=sampled_surface_clearance(samples@pose[:3,:3].T+pose[:3,3],terrain)['clearance_m']
            links=mechanism_and_truck_clearance(geometries,bodies,terrain,truck)
            if bucket<.32 or any(value<.02 for value in links.values()):
                raise ValueError('prepoint_refinement_clearance:'+str((bucket,links)))
            return dict(bucket_terrain_clearance_m=bucket,**links)
        with r:
            def step(command,reference):
                if r.frames>=1200 or time.monotonic()-started>360:
                    raise RuntimeError('prepoint_qualification_budget')
                row=r.step(command)
                actual=np.asarray(r.bridge.state['mechanism_qpos'])
                if np.max(np.abs(actual[r.chain.axes]-reference[r.chain.axes]))>.025:
                    raise RuntimeError('prepoint_qualification_tracking')
                check(actual)
                row['qualification_bucket_force_norm_n']=float(np.linalg.norm(r.captured['wrenches']['bucket']['force']))
                report['maximum_refinement_bucket_force_n']=max(report['maximum_refinement_bucket_force_n'],row['qualification_bucket_force_norm_n'])
                # Keep the existing physical hard guards. This added 1 N
                # development bound detects meaningful load during retreat;
                # it never substitutes for the exact zero-sample endpoint.
                if row['qualification_bucket_force_norm_n']>1. or row['rock_bucket_contact_substeps']:
                    raise RuntimeError('prepoint_refinement_nonzero_load:'+str(row['qualification_bucket_force_norm_n']))
                return row
            # Try the existing point first. If needed, use two modest bounded
            # offsets from it, never shift the cut target or alter terrain.
            for outward,upward in ((0.,0.),(.05,.10),(.10,.15)):
                if outward:
                    start=np.asarray(r.bridge.state['mechanism_qpos']).copy()
                    target=initial_tip+np.r_[outward*radial,upward]
                    goal=solve_pose(r.assembly,r.chain,start,target,-.32,TIP)
                    def advance(q,v,dt):
                        r.chain.q=q.copy();return r.chain.advance(v,dt)[0]
                    path=sampled_approach(start,goal,axes=r.chain.axes,
                        speed_limits=np.array([.12,.0225,.0225,.0225]),advance=advance,
                        check_pose=check,dt=1/60,max_steps=600,validation_substeps=2)
                    for desired,velocity in zip(path['qpos'][1:],path['velocity'],strict=True):
                        controller=np.asarray(r.bridge.state['controller_target'])
                        command=np.clip(velocity+4*(desired[r.chain.axes]-controller),
                            [-.12,-.0225,-.0225,-.0675],[.12,.0225,.0225,.0675])
                        step(command,desired)
                reference=np.asarray(r.bridge.state['mechanism_qpos']).copy()
                consecutive=0;counts=[];forces=[]
                for _ in range(90):
                    row=step(np.zeros(4),reference)
                    counts.append(row['bucket_mpm_contact_samples']);forces.append(row['qualification_bucket_force_norm_n'])
                    consecutive=consecutive+1 if counts[-1]==0 and row['rock_bucket_contact_substeps']==0 else 0
                    if consecutive>=30:break
                    if not outward and len(counts)>=3 and any(counts):break
                report['attempts'].append(dict(outward_offset_from_initial_m=outward,upward_offset_from_initial_m=upward,
                    hold_contact_samples=counts,maximum_hold_bucket_force_n=max(forces),trailing_zero_contact_frames=consecutive))
                if consecutive>=30:
                    actual=np.asarray(r.bridge.state['mechanism_qpos']);pose=r.assembly.assemble(actual,include_meshes=False)[1]['bucket']
                    fresh,obs=stationary_terrain(r,r.capture())
                    clearance=sampled_surface_clearance(samples@pose[:3,:3].T+pose[:3,3],ground_completed_terrain(fresh))['clearance_m']
                    if clearance<.32:raise RuntimeError('qualified_prepoint_fresh_clearance')
                    report.update(status='zero_contact_prepoint_saved_requires_restore_audit',
                        checkpoint=r.save(out/'prepoint.npz',stage='prepoint'),frames=r.frames,
                        prepoint_tip_world_m=pose[:3,:3]@TIP+pose[:3,3],fresh_clearance_m=clearance,
                        fresh_observation=obs,trailing_zero_contact_frames=consecutive)
                    write();return
            raise RuntimeError('prepoint_zero_contact_not_qualified')
    except BaseException as error:
        report.update(status='failed',failure=repr(error),frames=r.frames);write();raise
    finally:
        report['wall_seconds']=time.monotonic()-started;write()


if __name__=='__main__':main()
