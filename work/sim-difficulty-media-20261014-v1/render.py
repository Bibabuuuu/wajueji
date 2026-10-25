"""Render real cached states in focused views. Never run physics or a policy."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,math,os,subprocess,sys,time
import numpy as np
from PIL import Image
os.environ.setdefault('DISPLAY',':1');os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
R=Path('/home/qunwang/wajueji');RUN=R/'runs/continuous-single-scoop-20261014-v1';CACHE=RUN/'display'
OUT=R/'runs/roadshow-handover-20261012/stage-05-sim-difficulties'
sys.path[:0]=[str(R/'src/v4'),str(R)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def main(smoke):
 import pyglet.display.xlib as xlib
 if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
 import newton,warp as wp
 import run_layered_47_sparse_manual_contact_episode as manual
 from soil_surface import SoilSurface
 start=time.monotonic();deadline=start+360
 OUT.mkdir(exist_ok=True);shots=OUT/('smoke' if smoke else 'frames');shots.mkdir(exist_ok=False)
 report=read(RUN/'report.json');audit=read(RUN/'read-only-audit.json');meta=read(CACHE/'reconstruction.json')
 assert report['status']=='continuous_single_scoop_completed' and audit['status']=='passed' and meta['frames']==13753
 identity={k:dict(sha256=v,size=(CACHE/k).stat().st_size,mtime_ns=(CACHE/k).stat().st_mtime_ns) for k,v in meta['cache_sha256'].items()}
 for k,v in identity.items():assert sha(CACHE/k)==v['sha256'],k
 print('CACHE_VERIFIED',flush=True)
 states=read(CACHE/'states.json');scene=read(CACHE/'scene.json');particles=np.load(CACHE/'particles.npy',mmap_mode='r')
 display=read(R/'configs/v4/layered-47-interactive-proxy-practice-v1.json')['display']
 wp.init();wp.set_device('cuda:0');assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
 viewer=newton.viewer.ViewerGL(width=1440,height=900,vsync=False,paused=True,headless=True);viewer.set_model(model);viewer.picking_enabled=False
 positions=wp.array(particles[0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions))
 surface=SoilSurface(positions,particle_spacing_m=scene['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
 manual.display_helpers._prepare_rock_visuals(viewer,scene['records'],Path(scene['root']),'cuda:0')
 face=scene['selected_cut'].get('contact',scene['selected_cut']);goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
 base_colors=model.shape_color.numpy().copy();rendered=[];layer='all'
 original_log_state=viewer.log_state
 def log_state(state):
  original_log_state(state)
  for key,obj in viewer.objects.items():
   if not key.startswith('/model/shapes/shape_'):continue
   idx=int(key.rsplit('_',1)[1]);label=model.shape_label[idx]
   # Display filters affect only the 3-D drawing, never the stored state.
   if label.startswith('truck_'):obj.hidden=True
   if layer=='terrain' and idx<13:obj.hidden=True
   if layer=='linkage' and idx in (0,1):obj.hidden=True
 viewer.log_state=log_state
 def camera(pos,target):
  delta=np.asarray(target)-np.asarray(pos);yaw=math.degrees(math.atan2(delta[1],delta[0]));pitch=math.degrees(math.atan2(delta[2],np.linalg.norm(delta[:2])))
  viewer.set_camera(wp.vec3(*pos),pitch=pitch,yaw=yaw)
 def frame(n,mode='all'):
  nonlocal layer
  if time.monotonic()>deadline:raise TimeoutError('offline_render_360_seconds')
  layer=mode;positions.assign(particles[n]);state={**states[n],'bucket_origin':states[n]['bucket_origin_world_m']}
  manual.draw_with_selected_cut(viewer,model_state,fixture,state,scene['records'],assembly,names,to_pose,'cuda:0',display,goal,n,surface,show_target=False)
  return np.array(viewer.get_frame(render_ui=False).numpy(),copy=True)
 def output(n,name,view,mode='all',ghost=False):
  camera(*view)
  a=frame(n,mode)
  if ghost:
   b=frame(n,'terrain')
   # Standard same-camera two-pass opacity rendering; only the machine layer changes.
   a=np.rint(a.astype(np.float32)*.12+b.astype(np.float32)*.88).clip(0,255).astype(np.uint8)
  dest=shots/name;Image.fromarray(a).save(dest)
  rendered.append(dict(file=str(dest.relative_to(OUT)),frame=n,time_s=n/60,view=view,mode=mode,machine_opacity=.12 if ghost else (0 if mode=='terrain' else 1),sha256=sha(dest)))
  print('FRAME',name,n,flush=True)
 contact=([7.7,-3.1,2.7],[5.35,0,1.05])
 mechanism=([3.6,-7.7,3.6],[3.6,0,2.0])
 terrain=([8.0,-4.2,4.6],[5.35,0,.85])
 try:
  camera(*contact)
  for _ in range(2):frame(6265)
  for n in ([6265] if smoke else [5950,6265,6730]):output(n,f'contact-{n}.png',contact)
  colors=base_colors.copy()
  for idx in (2,3):colors[idx]=[.16,.75,.85]
  for idx in (5,6):colors[idx]=[.98,.48,.12]
  for idx in (8,9):colors[idx]=[.65,.46,.88]
  for idx in (10,12):colors[idx]=[.2,.82,.45]
  model.shape_color.assign(colors)
  for n in ([6265] if smoke else [5626,6265,6832]):output(n,f'mechanism-{n}.png',mechanism,'linkage')
  model.shape_color.assign(base_colors)
  for n in ([5626,7447] if smoke else [5626,6265,6832,7447]):output(n,f'terrain-ghost-{n}.png',terrain,ghost=True)
  if not smoke:
   for n in [5626,6265,7447]:output(n,f'terrain-only-{n}.png',terrain,'terrain')
   # A matched-view process clip, generated solely from original cached frames.
   movie=OUT/'workface-deformation-ghost.mp4';indices=list(range(5626,7448,12))
   if indices[-1]!=7447:indices.append(7447)
   command=['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1440x900','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
   camera(*terrain)
   with subprocess.Popen(command,stdin=subprocess.PIPE) as process:
    try:
     for n in indices:
      a=frame(n);b=frame(n,'terrain');im=np.rint(a.astype(np.float32)*.12+b.astype(np.float32)*.88).clip(0,255).astype(np.uint8)
      process.stdin.write(im.tobytes())
     process.stdin.close();assert process.wait(timeout=30)==0
    except BaseException:
     process.kill();process.wait();raise
   (OUT/'video-frames.json').write_text(json.dumps(dict(fps=20,frames=indices,source_simulation_hz=60,nominal_playback_speed=4,unchanged_view=terrain)))
  for k,v in identity.items():assert (CACHE/k).stat().st_size==v['size'] and (CACHE/k).stat().st_mtime_ns==v['mtime_ns'],k
  result=dict(status='smoke_passed' if smoke else 'rendered',source_run=str(RUN),source_report_sha256=sha(RUN/'report.json'),source_audit_sha256=sha(RUN/'read-only-audit.json'),cache_identities=identity,
   renderer_sha256=sha(__file__),physics_steps_added=0,policy_inference_calls=0,training_updates=0,frames=rendered,wall_seconds=time.monotonic()-start,
   display_changes='Fixed focus cameras; truck hidden. Linkage view hides cab/base and uses component colors. Terrain view uses exact same-frame 12% machine display opacity composed from two render passes. No terrain smoothing or displacement edits.',
   component_colors={'chassis_boom_cylinder':'cyan','boom_arm_cylinder':'orange','arm_bucket_cylinder':'purple','H_and_side_link':'green'})
  (OUT/('smoke-review.json' if smoke else 'render-review.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
  print('COMPLETE',result['status'],len(rendered),result['wall_seconds'],flush=True)
 finally:viewer.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args().smoke)
