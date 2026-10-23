"""Reconstruct a bounded, recorded BC probe once, then display cached frames.

The viewer never advances physics. Replay reconstruction must match every
recorded mechanism state and all final physics arrays before viewing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
from contact_bc_inference import sha


def cache_digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


class ProbeDisplayCache:
    """Capture display state during a live probe, without a second simulation."""
    def __init__(self,output,maximum_frames,particle_shape,scene):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.states=[]
        self.positions=np.lib.format.open_memmap(self.output/'particles.npy',mode='w+',dtype=np.float32,
            shape=(maximum_frames+1,*particle_shape))
        (self.output/'scene.json').write_text(json.dumps(scene,allow_nan=False))
    def capture(self,frame,state,particles):
        if frame!=len(self.states) or frame>=len(self.positions):raise ValueError('display_cache_frame_order')
        if not np.isfinite(particles).all():raise ValueError('display_cache_nonfinite_particles')
        copied=json.loads(json.dumps(state,allow_nan=False))
        self.positions[frame]=particles;self.states.append(copied)
    def finish(self,probe_status):
        self.positions.flush()
        used_frames=len(self.states)
        if used_frames < len(self.positions):
            source=self.positions
            trimmed_path=self.output/'particles.trim.npy'
            trimmed=np.lib.format.open_memmap(
                trimmed_path,mode='w+',dtype=source.dtype,
                shape=(used_frames,*source.shape[1:]))
            for start in range(0,used_frames,16):
                stop=min(start+16,used_frames)
                trimmed[start:stop]=source[start:stop]
            trimmed.flush();del trimmed;del self.positions
            trimmed_path.replace(self.output/'particles.npy')
        else:
            del self.positions
        (self.output/'states.json').write_text(json.dumps(self.states,allow_nan=False))
        report=dict(status='captured_probe',capture_origin='live_policy_rollout',probe_status=probe_status,
            frames=len(self.states)-1,playback_physics_steps=0,live_inference=False,label='BC CONTACT WINDOW REPLAY',
            cache_sha256={name:cache_digest(self.output/name) for name in ('particles.npy','states.json','scene.json')})
        (self.output/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
        return report


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def reconstruct(review_path, output):
    from layered_47_cycle_runtime import CycleRuntime
    import run_layered_47_sparse_manual_contact_episode as manual
    review=json.loads(review_path.read_text());run=Path(review['run'])
    if not review['passed'] or not 1<=review['frames']<=120:
        raise ValueError('requires_audited_bounded_probe')
    for identity in review['inputs']+review['outputs']:
        if sha(identity['path'])!=identity['sha256']:
            raise ValueError('recorded_input_changed:'+identity['path'])
    source=Path(review['inputs'][0]['path'])
    actions=read_rows(run/'actions.ndjson');expected=read_rows(run/'physics/frames.ndjson')
    if len(actions)!=review['frames'] or len(expected)!=len(actions):
        raise ValueError('recorded_frame_count')
    output.mkdir(parents=True,exist_ok=False);states=[];started=time.monotonic()
    report=dict(status='reconstructing',review_sha256=sha(review_path),frames=len(actions),recorded_run=str(run),
                playback_physics_steps=0,live_inference=False,frames_reconstructed=0)
    def write():
        (output/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    write()
    try:
        with CycleRuntime(source,output/'physics',allow_training_reset=True) as r:
            positions=np.lib.format.open_memmap(output/'particles.npy',mode='w+',dtype=np.float32,
                shape=(len(actions)+1,*r.arrays['particle_q'].shape))
            def capture(index):
                states.append(copy.deepcopy(manual.compact_state(r.bridge.state)))
                positions[index]=r.fixture.state.particle_q.numpy()
            capture(0)
            for i,(a,p) in enumerate(zip(actions,expected,strict=True),1):
                if time.monotonic()-started>180:raise TimeoutError('reconstruction_budget')
                if manual.compact_state(r.bridge.state)!=a['before_state']:raise ValueError(f'before_state_mismatch:{i}')
                if a['frame']!=i or p['frame']!=i:raise ValueError('frame_order')
                row=r.step(np.array(a['applied_command_si']))
                if row['after_state']!=p['after_state']:raise ValueError(f'after_state_mismatch:{i}')
                capture(i);report['frames_reconstructed']=i
            final=r.capture()
            with np.load(run/'final.npz') as archive:
                exact={key:bool(np.array_equal(value,archive[key])) for key,value in final.items()}
            report['final_arrays_exact']=exact
            if not all(exact.values()):raise ValueError('final_physics_mismatch')
            positions.flush();del positions
            (output/'states.json').write_text(json.dumps(states,allow_nan=False))
            (output/'scene.json').write_text(json.dumps(dict(records=r.records,root=str(r.root),
                selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m'])))
        report.update(status='reconstruction_passed',wall_seconds=time.monotonic()-started,source_unchanged=sha(source)==review['inputs'][0]['sha256'])
        report['cache_sha256']={name:cache_digest(output/name) for name in ('particles.npy','states.json','scene.json')}
        write()
    except BaseException as e:
        report.update(status='failed',failure=repr(e));write();raise


def show(output):
    import os
    import pyglet.display.xlib as xlib
    if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
    from pyglet.window import key
    import newton
    import warp as wp
    import run_layered_47_sparse_manual_contact_episode as manual
    from soil_surface import SoilSurface
    from PIL import Image
    report=json.loads((output/'reconstruction.json').read_text())
    if report['status'] not in ('reconstruction_passed','captured_probe'):raise ValueError('unverified_replay')
    for name,digest in report['cache_sha256'].items():
        if cache_digest(output/name)!=digest:raise ValueError('changed_replay_cache')
    states=json.loads((output/'states.json').read_text());scene=json.loads((output/'scene.json').read_text())
    particles=np.load(output/'particles.npy',mmap_mode='r');last=len(states)-1
    display=json.loads((Path(__file__).resolve().parents[2]/'configs/v4/layered-47-interactive-proxy-practice-v1.json').read_text())['display']
    wp.init();wp.set_device('cuda:0')
    assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
    viewer=newton.viewer.ViewerGL(width=display['width'],height=display['height'],vsync=False,paused=True)
    viewer.set_model(model);viewer.picking_enabled=False
    viewer.renderer.set_title(report.get('label','BC STARTUP REPLAY')+' - Space play/pause | R restart | arrows step | X close')
    viewer.set_camera(wp.vec3(*display['camera_position_m']),pitch=display['camera_pitch_deg'],yaw=display['camera_yaw_deg'])
    positions=wp.array(particles[0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions))
    surface=SoilSurface(positions,particle_spacing_m=scene['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
    manual.display_helpers._prepare_rock_visuals(viewer,scene['records'],Path(scene['root']),'cuda:0')
    selected=scene['selected_cut'];face=selected.get('contact',selected)
    goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
    cursor=dict(frame=0,drawn=-1,clock=time.monotonic(),fps=30.)
    def move(frame):
        cursor['frame']=max(0,min(last,frame));cursor['clock']=time.monotonic()
    def on_key(symbol,modifiers):
        if symbol in (key.X,key.ESCAPE):viewer.close()
        elif symbol==key.R:move(0);viewer._paused=True
        elif symbol in (key.RIGHT,key.LEFT):
            viewer._paused=True;move(cursor['frame']+(1 if symbol==key.RIGHT else -1))
        elif symbol==key.HOME:viewer._paused=True;move(0)
        elif symbol==key.END:viewer._paused=True;move(last)
    viewer.renderer.window.push_handlers(on_key_press=on_key)
    def ui(imgui):
        imgui.text('RECORDED BC | CACHED PLAYBACK')
        imgui.text(f"Frame {cursor['frame']}/{last} | simulation {cursor['frame']/60:.2f} s")
        imgui.text('Space: play/pause | R: rewind and pause')
        imgui.text('Left/Right: one frame | Home/End: start/end')
        imgui.text('Playback: 0.5x | X/Esc: close')
        imgui.text('No new physics or gamepad input during playback')
        imgui.text('Green: fixed target center | Cyan: target width')
        imgui.text('Result: '+report.get('probe_status','startup_only'))
    viewer.register_ui_callback(ui,position='side');draws=0
    try:
        while viewer.is_running():
            now=time.monotonic()
            if viewer.is_paused():cursor['clock']=now
            elif now-cursor['clock']>=1/cursor['fps']:
                move(cursor['frame']+1)
                if cursor['frame']==last:viewer._paused=True
            f=cursor['frame']
            if f!=cursor['drawn']:positions.assign(particles[f]);cursor['drawn']=f
            state={**states[f],'bucket_origin':states[f]['bucket_origin_world_m']}
            manual.draw_with_selected_cut(viewer,model_state,fixture,state,scene['records'],assembly,names,to_pose,'cuda:0',display,goal,f,surface)
            draws+=1
            if draws==3:
                Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output/'viewer-ready.png')
                (output/'viewer-ready.json').write_text(json.dumps(dict(pid=os.getpid(),paused=viewer.is_paused(),frame=f,frames=last,playback_physics_steps=0,live_inference=False)))
                print('REPLAY_VIEWER_READY',flush=True)
            time.sleep(.005)
    finally:
        viewer.close();(output/'viewer-closed.json').write_text(json.dumps(dict(frame=cursor['frame'],playback_physics_steps=0)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--view-only',action='store_true');p.add_argument('--prepare-only',action='store_true');args=p.parse_args()
    if not args.view_only:
        if args.review is None:p.error('--review is required for reconstruction')
        reconstruct(args.review,args.output)
    if not args.prepare_only:show(args.output)

if __name__=='__main__':main()
