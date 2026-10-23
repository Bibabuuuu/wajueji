"""Same-camera replay of real cached frames; no physics or visual smoothing."""
from pathlib import Path
from types import SimpleNamespace
import os,sys,json,csv,subprocess,time,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
os.environ.setdefault('DISPLAY',':1');os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
R=Path('/home/qunwang/wajueji');W=R/'work/roadshow-review-20261015-v1';O=R/'runs/roadshow-review-20261015-v1';V=O/'video';V.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'src/v4'),str(R)]
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))
import pyglet.display.xlib as xlib
if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
import warp as wp,newton
import run_layered_47_sparse_manual_contact_episode as manual
from soil_surface import SoilSurface
assert read(O/'report.json')['status']=='completed'
display=read(R/'configs/v4/layered-47-interactive-proxy-practice-v1.json')['display']
wp.init();wp.set_device('cuda:0');assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
viewer=newton.viewer.ViewerGL(width=1280,height=720,vsync=False,paused=True,headless=True);viewer.set_model(model);viewer.picking_enabled=False
viewer.set_camera(wp.vec3(*display['camera_position_m']),pitch=display['camera_pitch_deg'],yaw=display['camera_yaw_deg'])
folders=[R/'runs/motion-physical-short-20261015-v2/loaded',O]
states=[read(f/'display/states.json') for f in folders];scenes=[read(f/'display/scene.json') for f in folders]
clouds=[np.load(f/'display/particles.npy',mmap_mode='r') for f in folders]
positions=wp.array(clouds[0][0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions))
surface=SoilSurface(positions,particle_spacing_m=scenes[0]['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
manual.display_helpers._prepare_rock_visuals(viewer,scenes[0]['records'],Path(scenes[0]['root']),'cuda:0')
def draw(state,particle,scene,frame):
 positions.assign(particle);selected=scene['selected_cut'];face=selected.get('contact',selected)
 goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
 manual.draw_with_selected_cut(viewer,model_state,fixture,{**state,'bucket_origin':state['bucket_origin_world_m']},scene['records'],assembly,names,to_pose,'cuda:0',display,goal,frame,surface,show_target=False)
 return Image.fromarray(viewer.get_frame(render_ui=False).numpy())
fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,28);small=ImageFont.truetype(fontpath,22)
def pair(f):
 top=draw(states[0][f],clouds[0][f],scenes[0],f)
 bottom=draw(states[1][f],clouds[1][f//2],scenes[1],f)
 # Same crop for both: exclude empty sky/floor while retaining the whole machine and bucket.
 images=[x.crop((400,140,1200,478)).resize((960,405),Image.Resampling.LANCZOS) for x in [top,bottom]]
 image=Image.new('RGB',(960,910),(13,39,55));d=ImageDraw.Draw(image)
 for i,(im,label,color) in enumerate(zip(images,['修复前 · 原控制','修复后 · 负载补偿 + 耗散阻尼'],[(237,180,75),(92,208,222)])):
  y=i*455;image.paste(im,(0,y+50));d.text((20,y+8),label,font=font,fill=color)
  d.text((740,y+13),f'{max(0,f-3)/60:.2f} s',font=small,fill='white')
 return image,bottom
try:
 for _ in range(3):pair(4)
 for f in [4,34,120,360,766]:pair(f)[0].save(V/f'pair-{f:04d}.png')
 if '--smoke' in sys.argv:print('SMOKE_OK',flush=True);sys.exit(0)
 cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','960x910','-r','30','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(V/'oscillation-before-after-1x.mp4')]
 fixed=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-','-an','-c:v','libx264','-threads','4','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(V/'fixed-loaded-lift-1x.mp4')]
 start=time.monotonic();indices=list(range(4,767,2))
 with subprocess.Popen(cmd,stdin=subprocess.PIPE) as p,subprocess.Popen(fixed,stdin=subprocess.PIPE) as q,(V/'frame-map.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['video_frame','source_frame','lift_time_s','new_particle_sample'])
  for i,g in enumerate(indices):
   image,clean=pair(g);p.stdin.write(np.asarray(image).tobytes());q.stdin.write(np.asarray(clean).tobytes());w.writerow([i,g,(g-3)/60,g//2])
   if i%90==0:print('RENDER',i,len(indices),'elapsed',round(time.monotonic()-start),flush=True)
  p.stdin.close();q.stdin.close();assert p.wait(timeout=90)==q.wait(timeout=90)==0
 pair(34)[0].save(V/'comparison-poster.png')
 # Same-view endpoint images for the matched excavation comparison.
 matched=read(W/'matched-sources.json')
 for case in ['manual','residual']:
  folder=Path(matched[case]);s=read(folder/'display/states.json');scene=read(folder/'display/scene.json');c=np.load(folder/'display/particles.npy',mmap_mode='r')
  draw(s[-1],c[-1],scene,len(s)-1).save(V/f'{case}-endpoint.png')
 write(V/'render-report.json',dict(status='completed',video_frames=len(indices),fps=30,duration_s=len(indices)/30,first_source_frame=4,last_source_frame=766,temporal_smoothing=False,playback_physics_steps=0,top='original controller',bottom='new verified controller; freshly captured same-source particles',same_camera=True,same_crop=True,wall_seconds=time.monotonic()-start))
 print('RENDER_COMPLETED',len(indices)/30,flush=True)
finally:viewer.close()
