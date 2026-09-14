"""Bounded, synchronous read-only observer for the manual Newton scene.

Sampling uses wall time. A slow physics loop skips deadlines rather than queues
stale work. Contact telemetry is tapped without changing returned force values.
"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from .core import Camera, Grid, LatestSchedule, StateBuilder, look_at, noisy_depth, observation_fresh
from .virtual_depth import VirtualDepth


CAMERA_POSES=(look_at([8.,-8.,7.],[3.5,1.,.7]),look_at([7.,6.,6.],[5.,0.,.5]))


class OnlinePerception:
    def __init__(self, assembly, initial_q, truck_root, contract, soil, folder, *, views=2, period_s=.2, truck_parts_provider=None):
        from moving_scene import truck_parts
        if views not in (1,2):raise ValueError('invalid_camera_count')
        self.folder=Path(folder);self.folder.mkdir(exist_ok=False)
        self.soil=soil;self.contract=contract;self.camera=Camera();self.grid=Grid()
        self.poses=CAMERA_POSES[:views];self.schedule=LatestSchedule(period_s)
        self.builder=StateBuilder(self.camera,self.grid);self.sequence=0;self.latest=None
        self.error='';self.failures=0;self.timings=[];self.max_age_before_poll=0.;self.stale_polls=0
        self._contact=None;self._last_arrays={};self._closed=False
        geoms,_=assembly.assemble(initial_q)
        self.names=[g['body'] for g in geoms]
        geometry=[(g['body_local_vertices'],g['faces']) for g in geoms]
        parts=list((truck_parts_provider or truck_parts)(truck_root));geometry.extend((v,f) for _,v,f,_ in parts)
        self.static_poses=[np.eye(4)]*len(parts)
        self.sensor=VirtualDepth(self.camera,geometry,robot_count=len(geoms))
        self.trace=(self.folder/'timing.jsonl').open('w')
        self.original_force=soil.diagnostics.force
        def tapped_force(*args,**kwargs):
            force,torque=self.original_force(*args,**kwargs)
            self._contact={'sim_s':soil.time,'wall_s':time.perf_counter(),
                           'wrench':np.concatenate((force,torque)).copy(),
                           'origin_world_m':np.asarray(args[-2]).copy()}
            return force,torque
        soil.diagnostics.force=tapped_force

    def _write_latest(self,packet):
        tmp=self.folder/'latest.tmp'
        tmp.write_text(json.dumps(packet,allow_nan=False)+'\n')
        tmp.replace(self.folder/'latest.json')

    def poll(self, qpos, body_poses):
        """Caller supplies the accepted post-step pose; this never steps soil."""
        start=time.perf_counter()
        if self.latest:
            self.max_age_before_poll=max(self.max_age_before_poll,start-self.latest['time']['capture_monotonic_s'])
            self.stale_polls+=int(not observation_fresh(self.latest,self.soil.time,start))
        if not self.schedule.due(start):return False
        try:
            sim_s=self.soil.time;particles=self.soil.state.particle_q.numpy()
            mesh_poses=[body_poses[n] for n in self.names]+self.static_poses
            captured=[];arrays={}
            for index,pose in enumerate(self.poses):
                ideal,robot=self.sensor.capture(particles,pose,mesh_poses)
                seed=self.sequence*10+index
                depth=noisy_depth(ideal,self.camera,seed=seed)
                mask=np.isfinite(depth)&np.isfinite(robot)&(np.abs(depth-robot)<.03)
                captured.append((self.camera,depth,pose,mask))
                arrays[f'depth_{index}_m']=depth;arrays[f'self_mask_{index}']=mask
            contact=self._contact
            packet,height,count=self.builder.build(episode=self.folder.parent.name,sequence=self.sequence,
                capture_sim_s=sim_s,now_sim_s=sim_s,capture_wall_s=start,now_wall_s=time.perf_counter(),
                depth=captured[0][1],world_from_camera=captured[0][2],self_mask=captured[0][3],
                extra_views=captured[1:],bucket_pose=body_poses['bucket'],joints=qpos,
                wrench=None if contact is None else contact['wrench'],truck_prior=self.contract['truck'])
            done=time.perf_counter()
            packet['time'].update(processed_monotonic_s=done,age_wall_s=done-start)
            packet['fresh']=done-start<=self.builder.max_age_wall_s
            packet['terrain']['valid']=bool(packet['fresh'] and count.any())
            packet['sensor']={'camera':asdict(self.camera),'world_from_cameras':[p.tolist() for p in self.poses],
                'mount':'two_fixed_site_cameras' if len(self.poses)==2 else 'fixed_site_camera',
                'same_physics_state':True,'capture_skew_sim_s':0.,'period_wall_s':self.schedule.period,
                'noise_sigma_m':.005,'dropout_probability':.02,'no_catchup_queue':True}
            if contact:
                load_age=done-contact['wall_s'];load_age_sim=sim_s-contact['sim_s']
                packet['load'].update(sample_sim_s=contact['sim_s'],sample_monotonic_s=contact['wall_s'],
                    age_wall_s=load_age,age_sim_s=load_age_sim,torque_origin_world_m=contact['origin_world_m'].tolist(),
                    valid=bool(packet['fresh'] and 0<=load_age<=self.builder.max_age_wall_s and 0<=load_age_sim<=self.builder.max_age_sim_s))
            self.sequence+=1;self.latest=packet;self.error=''
            self._last_arrays={**arrays,'height_m':height,'point_count':count}
            self._write_latest(packet)
            finished=time.perf_counter()
            record={'sequence':packet['sequence'],'sim_s':sim_s,'wall_capture_s':start,
                'compute_ms':(done-start)*1000,'publish_ms':(finished-done)*1000,'total_ms':(finished-start)*1000,
                'coverage':packet['terrain']['coverage'],'fresh_at_publish':observation_fresh(packet,sim_s,finished),
                'load_valid':packet['load']['valid'],'skipped_deadlines':self.schedule.skipped}
            self.timings.append(record);self.trace.write(json.dumps(record)+'\n');self.trace.flush()
            return True
        except Exception as error:
            self.error=repr(error);self.failures+=1
            try:
                self._write_latest({'schema':'excavation_observation_v1','fresh':False,'error':self.error,
                                    'sim_s':self.soil.time,'wall_s':time.perf_counter()})
                self.trace.write(json.dumps({'error':self.error,'sim_s':self.soil.time})+'\n');self.trace.flush()
            except OSError:
                # Keep the in-memory error visible even when storage has failed.
                # Persisted packets carry absolute expiry for external readers.
                pass
            return False

    def status(self):
        now=time.perf_counter();p=self.latest
        if p is None:return {'valid':False,'error':self.error,'frames':0,'views':len(self.poses)}
        return {'valid':bool(not self.error and observation_fresh(p,self.soil.time,now)),
                'error':self.error,'frames':self.sequence,'views':len(self.poses),'coverage':p['terrain']['coverage'],
                'age_wall_s':now-p['time']['capture_monotonic_s'],
                'latency_ms':self.timings[-1]['total_ms'] if self.timings else None,
                'skipped_deadlines':self.schedule.skipped}

    def close(self):
        if self._closed:return
        self._closed=True;self.soil.diagnostics.force=self.original_force
        self.trace.close()
        if self._last_arrays:np.savez_compressed(self.folder/'last-frame.npz',**self._last_arrays)
        report={'frames':self.sequence,'failures':self.failures,'last_error':self.error,'views':len(self.poses),
                'skipped_deadlines':self.schedule.skipped,'max_age_before_poll_ms':self.max_age_before_poll*1000,
                'stale_polls':self.stale_polls,'period_wall_s':self.schedule.period,
                'timings_ms':{k:{'median':float(np.median([r[k] for r in self.timings])),
                                 'p95':float(np.quantile([r[k] for r in self.timings],.95))}
                             for k in ('compute_ms','publish_ms','total_ms')} if self.timings else {},
                'scope':'online manual observer, no control output or material changes; no end-to-end competition acceptance'}
        (self.folder/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
