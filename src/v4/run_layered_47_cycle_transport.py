#!/usr/bin/env python3
"""Bounded physical loaded transport/unload/return, never expert training data."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from layered_47_cycle_runtime import CycleRuntime, REPO
from layered_47_cycle_geometry import transport_plan, bucket_mask
from run_layered_47_sparse_manual_contact_episode import _clean
from terrain.truck_receiver import TruckReceiver, CAVITY


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--max-frames',type=int,default=5400)
    parser.add_argument('--max-wall-s',type=float,default=2400);args=parser.parse_args()
    if not 1<=args.max_frames<=6000 or not 0<args.max_wall_s<=3600:raise ValueError('cycle_transport_budget')
    out=Path(args.output).resolve();r=CycleRuntime(args.source,out,enable_truck=True)
    report=dict(schema='layered_47_cycle_transport_v1',status='planning',source=r.metadata['source_identity'],
                training_eligible=False,full_three_scoop_qualified=False,stages=[])
    started=time.monotonic()
    def write(): (out/'report.json').write_text(json.dumps(_clean(report),indent=2,allow_nan=False)+'\n')
    write()
    try:
        segments,checker=transport_plan(r)
        np.savez(out/'plan.npz',**{s['name']+'_qpos':s['path']['qpos'] for s in segments},
                  **{s['name']+'_velocity':s['path']['velocity'] for s in segments})
        report['plan']=[dict(name=s['name'],frames=s['path']['steps'],minimum_clearance=s['minimum_clearance']) for s in segments]
        report['observation']=checker.observation
        report['status']='geometry_passed';write()
        if args.plan_only:return
        initial_pose=r.assembly.assemble(r.arrays['mechanism_qpos'],include_meshes=False)[1]['bucket']
        source_payload=bucket_mask(r.arrays['particle_q'],initial_pose)
        if source_payload.sum()<100:raise ValueError('cycle_transport_requires_observed_load')
        receiver=TruckReceiver(json.loads((REPO/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json').read_text()))
        with r:
            def advance(command,reference=None):
                if r.frames>=args.max_frames or time.monotonic()-started>args.max_wall_s:raise RuntimeError('cycle_transport_budget_exceeded')
                row=r.step(command)
                q=np.array(r.bridge.state['mechanism_qpos'])
                checker.check(q,physical=True)
                if reference is not None:
                    error=float(np.max(np.abs(q[r.chain.axes]-reference[r.chain.axes])))
                    if error>.025:raise RuntimeError('cycle_transport_tracking:'+str(error))
                if r.frames%120==0:print('PHYSICS',r.frames,flush=True)
                return row
            for segment in segments:
                name=segment['name'];print('EXECUTE',name,flush=True)
                report['status']='executing_'+name;write()
                path=segment['path'];goal=segment['goal']
                for desired,velocity in zip(path['qpos'][1:],path['velocity'],strict=True):
                    target=np.array(r.bridge.state['controller_target'])
                    command=np.clip(velocity+4*(desired[r.chain.axes]-target),[-.12,-.0225,-.0225,-.0675],[.12,.0225,.0225,.0675])
                    advance(command,desired)
                for _ in range(30):advance(np.zeros(4),goal)
                actual=np.array(r.bridge.state['mechanism_qpos'])
                error=float(np.max(np.abs(actual[r.chain.axes]-goal[r.chain.axes])))
                if error>.015:raise RuntimeError('cycle_transport_endpoint:'+name+':'+str(error))
                if name=='DUMP':
                    for _ in range(180):advance(np.zeros(4),goal)
                    q=r.fixture.state.particle_q.numpy();pose=r.assembly.assemble(actual,include_meshes=False)[1]['bucket']
                    bucket=bucket_mask(q,pose);labels=receiver.classify(q)
                    received=source_payload&(labels==CAVITY);remaining=int(np.count_nonzero(source_payload&bucket))
                    inventory=dict(source_cavity_particles=int(source_payload.sum()),source_particles_in_truck_core=int(received.sum()),
                        received_source_fraction=float(received.sum()/source_payload.sum()),source_particles_remaining_in_bucket=remaining,
                        total_particles_in_bucket=int(bucket.sum()),total_particles_in_truck_core=int(np.sum(labels==CAVITY)),
                        scope='soil point occupancy, not stable delivered mixed volume; rock poses preserved and logged')
                    report['unload_inventory']=inventory;write()
                    if remaining>max(50,.1*source_payload.sum()) or received.sum()<.5*source_payload.sum():
                        raise RuntimeError('cycle_unload_development_gate:'+str(inventory))
                stage='returned' if name=='RETURN_HIGH' else 'unloaded' if name=='DUMP' else 'transport'
                saved=r.save(out/(name.lower()+'.npz'),stage=stage)
                report['stages'].append(dict(name=name,frame=r.frames,endpoint_axis_error=error,checkpoint=saved));write()
            report.update(status='transport_unload_return_passed',final_checkpoint=report['stages'][-1]['checkpoint'],frames=r.frames)
            write()
    except BaseException as error:
        report.update(status='failed',failure=repr(error),frames=r.frames)
        # The inner context may already be closed; every completed segment is
        # immutable and independently recoverable. Hard-fault frames stay logged.
        if r.bridge is not None:
            try:report['failure_checkpoint']=r.save(out/'failure.npz',stage='failure')
            except Exception as nested:report['snapshot_error']=repr(nested)
        write();raise
    finally:
        report['wall_seconds']=time.monotonic()-started;write()


if __name__=='__main__':main()
