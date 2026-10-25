"""Export one audited continuous cache; display and encoding only, no physics."""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, hashlib, json, subprocess, sys, time, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/home/qunwang/wajueji')
RUN=ROOT/'runs/continuous-single-scoop-20261014-v1'
CACHE=RUN/'display'
OUT=ROOT/'runs/roadshow-handover-20261012/stage-04'
sys.path[:0]=[str(ROOT/'src/v4'),str(ROOT)]
PHASES={
 'INITIAL':'合法预备点', 'PREPARE_OUTWARD':'起点准备：真实退让',
 'PREPARE_HIGH':'起点准备：抬升', 'SAFE_START':'起点准备：安全高位',
 'APPROACH_HIGH':'自动接近：高位通道', 'APPROACH_OUTWARD':'自动接近：外侧通道',
 'APPROACH_PREPOINT':'自动接近：挖掘预备点', 'RESIDUAL':'冻结残差策略：挖掘',
 'RESIDUAL_HOLD':'冻结残差策略：离土保载', 'CLEAR_PILE':'几何控制：带料举升',
 'ALIGN':'几何控制：回转对准', 'DUMP':'几何控制：卸料',
 'CLEAR_TRUCK':'几何控制：抬离料车', 'WITHDRAW_SWING':'几何控制：退出回摆',
 'RETURN_SAFE':'几何控制：返回安全高位', 'FINAL_HOLD':'安全高位保持'}
KEYFRAMES=[(4205,'approach','自动接近'),(5832,'entry','切入'),
 (6265,'curl','挖掘收斗'),(6832,'hold','离土保载'),
 (9618,'unloaded','卸料结束'),(13753,'returned','返回完成')]

def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def render(smoke):
 import pyglet.display.xlib as xlib
 if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
 import newton,warp as wp
 import run_layered_47_sparse_manual_contact_episode as manual
 from soil_surface import SoilSurface
 began=time.monotonic();deadline=began+480
 report=read(RUN/'report.json');audit=read(RUN/'read-only-audit.json');meta=read(CACHE/'reconstruction.json')
 assert report['status']=='continuous_single_scoop_completed' and audit['status']=='passed' and all(audit['checks'].values())
 assert report['frames']==meta['frames']==13753 and report['native_initializations']==1
 assert meta['cache_sha256']==report['display']['cache_sha256']
 identities={k:dict(size=(CACHE/k).stat().st_size,mtime_ns=(CACHE/k).stat().st_mtime_ns,sha256=v) for k,v in meta['cache_sha256'].items()}
 for k,d in identities.items():assert sha(CACHE/k)==d['sha256'],k
 print('CACHE_VERIFIED',flush=True)
 states=read(CACHE/'states.json');scene=read(CACHE/'scene.json');phases=read(CACHE/'phases.json');particles=np.load(CACHE/'particles.npy',mmap_mode='r')
 assert len(states)==len(phases)==len(particles)==13754 and set(phases)<=set(PHASES)
 display=read(ROOT/'configs/v4/layered-47-interactive-proxy-practice-v1.json')['display'];width,height=display['width'],display['height']
 OUT.mkdir(exist_ok=True);shots=OUT/('qa-smoke' if smoke else 'screenshots');shots.mkdir(exist_ok=False)
 fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,26);small=ImageFont.truetype(fontpath,19)
 wp.init();wp.set_device('cuda:0');assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
 viewer=newton.viewer.ViewerGL(width=width,height=height,vsync=False,paused=True,headless=True);viewer.set_model(model);viewer.picking_enabled=False
 viewer.set_camera(wp.vec3(*display['camera_position_m']),pitch=display['camera_pitch_deg'],yaw=display['camera_yaw_deg'])
 positions=wp.array(particles[0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions))
 surface=SoilSurface(positions,particle_spacing_m=scene['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
 manual.display_helpers._prepare_rock_visuals(viewer,scene['records'],Path(scene['root']),'cuda:0')
 selected=scene['selected_cut'];face=selected.get('contact',selected);goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
 def frame_image(frame):
  if time.monotonic()>deadline:raise TimeoutError('offline_render_480s_limit')
  positions.assign(particles[frame]);state={**states[frame],'bucket_origin':states[frame]['bucket_origin_world_m']}
  manual.draw_with_selected_cut(viewer,model_state,fixture,state,scene['records'],assembly,names,to_pose,'cuda:0',display,goal,frame,surface)
  image=viewer.get_frame(render_ui=False).numpy();assert image.shape==(height,width,3)
  im=Image.fromarray(image);draw=ImageDraw.Draw(im);draw.rectangle((0,0,width,96),fill=(18,27,39))
  draw.text((22,9),'冻结残差策略 · 连续单铲仿真回放',font=font,fill=(245,249,255))
  draw.text((22,51),f'{PHASES[phases[frame]]}  |  流程时间 {frame/60:.2f} s  |  帧 {frame}/13753  |  3×播放',font=small,fill=(193,221,241))
  draw.rectangle((0,height-34,width,height),fill=(18,27,39))
  draw.text((22,height-30),'同一物理实例 · 当前固定铲位第一铲 · 模型 84aade9a · 原始缓存回放',font=small,fill=(210,223,237))
  return im
 result=dict(status='rendering',source_run=str(RUN),source_report_sha256=sha(RUN/'report.json'),source_audit_sha256=sha(RUN/'read-only-audit.json'),
  renderer_sha256=sha(__file__),reused_renderer_sha256='2f4df0777507aa0a05dbb1bd0e29be74fe54d8a41369bf03a18c3c9cc6a9d5b7',
  cache_identities=identities,playback_physics_steps=0,policy_inference_calls=0,training_updates=0,source_physical_frames=13753,
  full_sequence=True,simulated_seconds=13753/60,nominal_playback_speed=3,source_fill_display_percent=142.61931187569368,
  scope='Current fixed-target first scoop, including genuine preparation prefix; frozen SAC dig and geometry transport/return. Not three scoops or independent 148.23% run.')
 reviewpath=OUT/('smoke-review.json' if smoke else 'render-review.json')
 write(reviewpath,result)
 try:
  for _ in range(3):frame_image(6265)
  shotrows=[]
  for frame,name,label in KEYFRAMES:
   path=shots/f'{frame:05d}-{name}.png';frame_image(frame).save(path)
   shotrows.append(dict(frame=frame,sim_time_s=frame/60,phase=phases[frame],label=label,file=path.relative_to(OUT).as_posix(),sha256=sha(path)))
  if smoke:
   result.update(status='smoke_passed',keyframes=shotrows,wall_seconds=time.monotonic()-began);write(reviewpath,result);print('SMOKE_PASSED',flush=True);return
  with (OUT/'keyframes.csv').open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=shotrows[0].keys());writer.writeheader();writer.writerows(shotrows)
  videos=OUT/'videos';videos.mkdir(exist_ok=False);movie=videos/'continuous-single-scoop-3x.mp4'
  indices=list(range(0,13754,6));indices+=[] if indices[-1]==13753 else [13753]
  with (OUT/'video-frames.csv').open('w',newline='') as f:
   writer=csv.writer(f);writer.writerow(['video_frame','source_frame','simulation_time_s','phase'])
   writer.writerows((i,n,n/60,phases[n]) for i,n in enumerate(indices))
  cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','-',
   '-an','-c:v','libx264','-preset','medium','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
  with subprocess.Popen(cmd,stdin=subprocess.PIPE) as process:
   try:
    for i,n in enumerate(indices):
     process.stdin.write(np.asarray(frame_image(n)).tobytes())
     if i%150==0:print('VIDEO_FRAME',i,'of',len(indices),'source',n,flush=True)
    process.stdin.close();assert process.wait(timeout=30)==0
   except BaseException:
    if process.stdin and not process.stdin.closed:process.stdin.close()
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
    raise
  for k,d in identities.items():assert (CACHE/k).stat().st_size==d['size'] and (CACHE/k).stat().st_mtime_ns==d['mtime_ns'],k
  result.update(status='rendered',keyframes=shotrows,movie=dict(file=movie.relative_to(OUT).as_posix(),width=width,height=height,fps=30,frames=len(indices),duration_s=len(indices)/30,
   source_first_frame=indices[0],source_last_frame=indices[-1],nominal_source_frame_stride=6,last_source_interval_frames=indices[-1]-indices[-2],sha256=sha(movie),bytes=movie.stat().st_size),
   wall_seconds=time.monotonic()-began);write(reviewpath,result);print('RENDERED',json.dumps(result['movie']),flush=True)
 except BaseException as e:
  result.update(status='failed',failure=repr(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-began);write(reviewpath,result);raise
 finally:viewer.close()

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');render(p.parse_args().smoke)
