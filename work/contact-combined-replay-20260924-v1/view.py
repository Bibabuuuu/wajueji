"""Zero-copy composition of verified cached cut/curl and lift; no physics."""
import json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from view_contact_bc_probe import cache_digest
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
    prior=Path('/home/qunwang/wajueji/runs/contact-full-response-20260924-v1/candidate60/display')
    lift=Path('/home/qunwang/wajueji/runs/contact-preview-lift-20260924-v1/continuation/display')
    sources=[]
    for directory in (prior,lift):
        verified=json.loads((directory/'reconstruction.json').read_text())
        for name,digest in verified['cache_sha256'].items():
            if cache_digest(directory/name)!=digest:raise ValueError('changed_replay_cache:'+str(directory/name))
        sources.append(dict(path=str(directory),cache_sha256=verified['cache_sha256']))
    before=json.loads((prior/'states.json').read_text());after=json.loads((lift/'states.json').read_text())
    first=np.load(prior/'particles.npy',mmap_mode='r');second=np.load(lift/'particles.npy',mmap_mode='r')
    if before[-1]!=after[0] or not np.array_equal(first[len(before)-1],second[0]):raise ValueError('replay_boundary_not_exact')
    class CombinedParticles:
        def __getitem__(self,frame):return first[frame] if frame<len(before) else second[frame-len(before)+1]
    particles=CombinedParticles();states=before+after[1:];last=len(states)-1
    scene=json.loads((prior/'scene.json').read_text())
    report=dict(status='verified_cache_composition',label='THIRD SCOOP | 48.8% WORKING FILL',probe_status='D152 cut/curl + D153 off-soil; no new physics',sources=sources,boundary_frame=2400,frames=last)
    (output/'replay-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
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
    cursor=dict(frame=last,drawn=-1,clock=time.monotonic(),fps=30.)
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
        imgui.text('THIRD SCOOP | VERIFIED RECORDED PLAYBACK')
        imgui.text('Final working fill: 48.8% / 124.5 L; formal 95% NOT passed')
        imgui.text('0-2400: cut/curl (D152); 2401-2824: bounded lift (D153)')
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
                move(cursor['frame']+max(1,int((now-cursor['clock'])*cursor['fps'])))
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
            time.sleep(.1)
    finally:
        viewer.close();(output/'viewer-closed.json').write_text(json.dumps(dict(frame=cursor['frame'],playback_physics_steps=0)))


if __name__=='__main__':show(Path('/home/qunwang/wajueji/runs/contact-combined-replay-20260924-v1'))
