#!/usr/bin/env python3
"""Observe the returned scene and physically prepare the next manual scoop."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from layered_47_cycle_runtime import CycleRuntime, REPO
from layered_47_cycle_selection import plan_next_prepoint
from layered_excavation_approach_planner import build_fused_terrain
from planning.rear_lower_prepoint import (ground_completed_terrain, proxy_surface_samples,
    sampled_surface_clearance, mechanism_and_truck_clearance)
from run_layered_47_sparse_manual_contact_episode import _clean


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--plan-only',action='store_true');p.add_argument('--max-wall-s',type=float,default=2400)
    args=p.parse_args();out=Path(args.output).resolve();r=CycleRuntime(args.source,out)
    if r.metadata['stage']!='returned' or not r.truck_enabled:raise ValueError('next_prepoint_requires_returned_scene')
    report=dict(schema='layered_47_cycle_prepoint_v1',status='planning',source=r.metadata['source_identity'],
                training_eligible=False,source_cycle_index=r.metadata['cycle_index'])
    def write():(out/'report.json').write_text(json.dumps(_clean(report),indent=2,allow_nan=False)+'\n')
    started=time.monotonic();write()
    try:
        plan=plan_next_prepoint(r);path=plan['path']
        report.update(selected=plan['selected'],rejected_pairs=plan['rejected_pairs'],observation=plan['observation'],
                      planned_frames=path['total_frames'],next_cycle_index=plan['cycle_index'])
        np.savez(out/'plan.npz',qpos=path['qpos'],velocity=path['velocity'])
        if path['total_frames']>4800:raise ValueError('cycle_prepoint_frame_budget')
        report['status']='geometry_passed';write()
        if args.plan_only:return
        with r:
            # No motion occurred between observation and exact restoration.
            samples=proxy_surface_samples(r.proxy)
            collision_terrain=ground_completed_terrain(plan['terrain'])
            geometries,_=r.assembly.assemble(r.arrays['mechanism_qpos'])
            with np.load(REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-mesh.npz') as data:truck=data['world_vertices'].copy()
            min_clearance=float('inf')
            def advance(command,reference):
                nonlocal min_clearance
                if time.monotonic()-started>args.max_wall_s:raise RuntimeError('cycle_prepoint_wall_budget')
                row=r.step(command);q=np.array(r.bridge.state['mechanism_qpos'])
                error=float(np.max(np.abs(q[r.chain.axes]-reference[r.chain.axes])))
                if error>.025:raise RuntimeError('cycle_prepoint_tracking:'+str(error))
                _,bodies=r.assembly.assemble(q,include_meshes=False);pose=bodies['bucket']
                bucket=sampled_surface_clearance(samples@pose[:3,:3].T+pose[:3,3],collision_terrain)['clearance_m']
                mechanism=mechanism_and_truck_clearance(geometries,bodies,collision_terrain,truck)
                if bucket<.30 or mechanism['mechanism_terrain_clearance_m']<.02 or mechanism['static_truck_aabb_separation_m']<.02:
                    raise RuntimeError('cycle_prepoint_physical_clearance:'+str((bucket,mechanism)))
                min_clearance=min(min_clearance,bucket)
                if r.frames%120==0:print('PREPOINT',r.frames,'clearance',bucket,flush=True)
            for desired,velocity in zip(path['qpos'][1:],path['velocity'],strict=True):
                target=np.array(r.bridge.state['controller_target'])
                command=np.clip(velocity+4*(desired[r.chain.axes]-target),[-.12,-.0225,-.0225,-.0675],[.12,.0225,.0225,.0675])
                advance(command,desired)
            for _ in range(30):advance(np.zeros(4),path['qpos'][-1])
            actual=np.array(r.bridge.state['mechanism_qpos']);endpoint_error=float(np.max(np.abs(actual[r.chain.axes]-path['qpos'][-1][r.chain.axes])))
            if endpoint_error>.015:raise RuntimeError('cycle_prepoint_endpoint_tracking')
            arrays=r.capture();fresh,observation=build_fused_terrain(r.root,r.records,arrays['particle_q'],arrays['rigid_qpos'])
            pose=r.assembly.assemble(actual,include_meshes=False)[1]['bucket']
            final_clearance=sampled_surface_clearance(samples@pose[:3,:3].T+pose[:3,3],ground_completed_terrain(fresh))['clearance_m']
            if final_clearance<.30:raise RuntimeError('cycle_prepoint_fresh_endpoint_clearance')
            saved=r.save(out/'prepoint.npz',stage='prepoint',cycle_index=plan['cycle_index'],selected_cut=plan['selected'])
            report.update(status='prepoint_saved_requires_independent_audit',checkpoint=saved,frames=r.frames,
                endpoint_axis_error=endpoint_error,minimum_physical_bucket_clearance_m=min_clearance,
                fresh_endpoint_clearance_m=final_clearance,fresh_observation=observation)
            write()
    except BaseException as error:
        report.update(status='failed',failure=repr(error),frames=r.frames);write();raise
    finally:report['wall_seconds']=time.monotonic()-started;write()


if __name__=='__main__':main()
