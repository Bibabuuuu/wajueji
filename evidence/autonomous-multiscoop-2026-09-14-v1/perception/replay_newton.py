"""Read-only sensor/reconstruction replay of verified v17 scene snapshots.

No Soil instance is created and no physics step is called. The adapter alone
reads simulator internals; reconstruction receives depth, calibration and
proprioception. Evaluation truth is written separately from observations.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import warp as wp
from .core import Camera, Grid, StateBuilder, look_at, noisy_depth, terrain_from_points
from .virtual_depth import VirtualDepth
from .online import CAMERA_POSES


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--truck-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--views',type=int,choices=(1,2),default=1)
    args=parser.parse_args();runtime=args.runtime_root.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(runtime))
    from scene_assembly import Assembler
    from fast_kinematics import install
    from moving_scene import truck_parts
    from closed_chain import ClosedChain
    from inventory import classify,summarize
    wp.init();wp.set_device('cuda:0')
    assembly=Assembler(runtime/'scene/input/model/A.xml',runtime/'scene/input/mesh');install(assembly)
    q0=np.array(json.loads((runtime/'operator-start.json').read_text())['qpos'])
    geoms,poses=assembly.assemble(q0);axes=ClosedChain(assembly,q0).axes
    geometry=[(g['body_local_vertices'],g['faces']) for g in geoms]
    body_names=[g['body'] for g in geoms]
    truck_geometry=list(truck_parts(args.truck_root));geometry.extend((v,f) for _,v,f,_ in truck_geometry)
    camera=Camera();grid=Grid();pose_camera=CAMERA_POSES[0]
    sensor=VirtualDepth(camera,geometry,robot_count=len(geoms))
    builder=StateBuilder(camera,grid)
    contract=json.loads((runtime/'inventory-contract.json').read_text())
    material=json.loads((runtime/'cohesive-fine01/soil-config.json').read_text())
    paths=[('initial',0.,runtime/'cohesive-fine01/settle-state.npz'),
           ('loaded',1900/60,runtime/'replay-full1/carry-1900.npz'),
           ('carry',2423/60,runtime/'replay-full1/carry-2423.npz'),
           ('before_dump',52.75,runtime/'replay-full1/recorded-end.npz'),
           ('after_dump',69.75,runtime/'replay-full1/final-state.npz')]
    contacts=[json.loads(line) for line in (runtime/'replay-full1/contact.jsonl').read_text().splitlines()]
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for _,_,p in paths}
    evaluations=[];all_timings=[]
    for sequence,(label,sim_s,path) in enumerate(paths):
        data=np.load(path,allow_pickle=False)
        q=data['state__particle_q'] if label=='initial' else data['q']
        mechanism=q0 if label=='initial' else data['mechanism_q']
        _,poses=assembly.assemble(mechanism,include_meshes=False)
        mesh_poses=[poses[name] for name in body_names]+[np.eye(4)]*len(truck_geometry)
        contact=next((r for r in reversed(contacts) if r['time_after_s'] <= sim_s+1e-7),None)
        wrench=None if contact is None else contact['bucket_force_n']+contact['bucket_torque_about_end_origin_nm']
        # Warm up this geometry/camera path; benchmark includes capture, readback,
        # noise, self-filter and reconstruction, but excludes input file loading.
        sensor.capture(q,pose_camera,mesh_poses)
        timings=[]
        for repetition in range(6):
            start=time.perf_counter()
            ideal,robot_depth=sensor.capture(q,pose_camera,mesh_poses)
            captured=time.perf_counter()
            noisy=noisy_depth(ideal,camera,seed=sequence*100+repetition)
            self_mask=np.isfinite(noisy)&np.isfinite(robot_depth)&(np.abs(noisy-robot_depth)<.03)
            extra_views=[];extra_arrays={}
            if args.views==2:
                ideal2,robot2=sensor.capture(q,CAMERA_POSES[1],mesh_poses)
                depth2=noisy_depth(ideal2,camera,seed=100000+sequence*100+repetition)
                mask2=np.isfinite(depth2)&np.isfinite(robot2)&(np.abs(depth2-robot2)<.03)
                extra_views=[(camera,depth2,CAMERA_POSES[1],mask2)]
                extra_arrays={'depth_second_m':depth2,'second_self_mask':mask2}
            processed=time.perf_counter()
            packet,height,count=builder.build(episode='verified-v17-offline',sequence=sequence*6+repetition,
                capture_sim_s=sim_s,now_sim_s=sim_s,capture_wall_s=start,now_wall_s=processed,
                depth=noisy,world_from_camera=pose_camera,bucket_pose=poses['bucket'],joints=mechanism,
                wrench=wrench,truck_prior=contract['truck'],self_mask=self_mask,replay=True,extra_views=extra_views)
            done=time.perf_counter()
            packet['time']['processed_monotonic_s']=done;packet['time']['age_wall_s']=done-start
            packet['fresh']=done-start<=builder.max_age_wall_s
            packet['terrain']['valid']=bool(packet['fresh'] and count.any())
            packet['load']['valid']=bool(packet['fresh'] and wrench is not None)
            packet['proprioception'].update(commanded_coordinate_indices=axes,commanded_coordinates=mechanism[axes].tolist(),
                commanded_coordinate_names=['upper_yaw','boom_stroke','arm_stroke','bucket_stroke'],
                commanded_coordinate_units=['rad','m','m','m'],configuration_source='ideal_simulated_full_configuration; four-encoder-only estimator pending')
            packet['load']['sample_sim_s']=None if contact is None else contact['time_after_s']
            packet['load']['torque_origin_world_m']=poses['bucket'][:3,3].tolist()
            packet['sensor']={'camera':asdict(camera),'world_from_camera':pose_camera.tolist(),
                'mount':'fixed_site_cameras','views':args.views,'world_from_cameras':[p.tolist() for p in CAMERA_POSES[:args.views]],'noise_sigma_m':.005,'dropout_probability':.02,
                'self_mask':'known_robot_CAD_and_ideal_proprioception_3cm_band','seed':sequence*100+repetition}
            timings.append({'first_camera_capture_ms':(captured-start)*1000,'noise_mask_and_extra_camera_ms':(processed-captured)*1000,
                'reconstruct_packet_ms':(done-processed)*1000,'total_ms':(done-start)*1000})
        assert all(hashlib.sha256(p.read_bytes()).hexdigest()==source_hashes[str(p)] for _,_,p in paths)
        folder=output/label;folder.mkdir()
        (folder/'observation.json').write_text(json.dumps(packet,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(folder/'sensor-reconstruction.npz',depth_ideal_m=ideal,depth_noisy_m=noisy,
            robot_predicted_depth_m=robot_depth,self_mask=self_mask,height_m=height,point_count=count,world_from_camera=pose_camera,**extra_arrays)
        # Evaluation-only particle-centre surface proxy. It is not used to fill
        # missing depth or select observations. Loaded particles can lie in ROI.
        gt,_=terrain_from_points(q+np.array([0.,0.,sensor.radius]),grid)
        gt=np.maximum(np.nan_to_num(gt,nan=0.),0.)
        observed=count>0;errors=height[observed]-gt[observed]
        truth=summarize(classify(q,poses['bucket'],contract,([2.,-2.5,-.05],[7.5,2.5,2.2])),
            np.full(len(q),material['initial_mass_kg']/len(q)),1670.)
        evaluation={'label':label,'sim_s':sim_s,'source':str(path),'source_sha256':source_hashes[str(path)],
            'coverage':float(observed.mean()),'observed_cells':int(observed.sum()),'self_mask_pixels':int(self_mask.sum()),
            'reference_surface':'particle centre + rendering radius cell maximum, ground=0; proxy only, not calibrated terrain truth',
            'observed_proxy_mae_m':float(np.abs(errors).mean()) if len(errors) else None,
            'observed_proxy_p95_abs_error_m':float(np.quantile(np.abs(errors),.95)) if len(errors) else None,
            'reference_inventory':truth,'timings':timings,'physics_steps_advanced':0,'snapshot_files_unchanged':True}
        (folder/'evaluation-only.json').write_text(json.dumps(evaluation,indent=2,allow_nan=False)+'\n')
        np.savez_compressed(folder/'evaluation-only.npz',reference_height_proxy_m=gt)
        evaluations.append(evaluation);all_timings.extend(timings)
        print(json.dumps({k:v for k,v in evaluation.items() if k not in ('timings','reference_inventory')}),flush=True)
    summary={'status':'read_only_perception_replay_complete','frames':len(paths),'measurements':len(all_timings),
        'camera':asdict(camera),'views':args.views,'grid':asdict(grid),'physics_steps_advanced':0,'source_files_unchanged':True,
        'timing_ms':{key:{'median':float(np.median([r[key] for r in all_timings])),
                        'p95':float(np.quantile([r[key] for r in all_timings],.95))} for key in all_timings[0]},
        'states':[{k:v for k,v in e.items() if k not in ('timings','reference_inventory')} for e in evaluations],
        'limitations':['offline snapshot replay; online frame scheduling and input-to-control latency not tested',
            f'{args.views} fixed virtual camera(s), synthetic Gaussian depth noise and dropout; no real sensor calibration',
            'CAD surfaces and particle spheres approximate visible material; no dust or lighting model',
            'unknown terrain is not filled; no temporal fusion; multiple views observe the same physical state',
            'truck location is a configured prior; load mass/fill estimator and material recognition not implemented',
            'ideal full mechanism configuration supplied; encoder-noise closed-chain state estimation not implemented']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
