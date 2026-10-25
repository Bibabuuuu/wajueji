"""Audited two-scoop cache playback only; no physics reconstruction or inference."""
from pathlib import Path
from types import SimpleNamespace
import argparse, csv, hashlib, json, os, subprocess, sys, time, traceback
import numpy as np
from PIL import Image, ImageDraw, ImageFont
os.environ.setdefault('DISPLAY',':1');os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
R=Path('/home/qunwang/wajueji');O=R/'runs/roadshow-two-scoop-media-20261015-v1'
sys.path[:0]=[str(R/'src/v4'),str(R)]
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
LABELS={'INITIAL':'首铲起点','RESTORE_PREFIX3':'首铲起点保持',
 'SCOOP1_RESIDUAL':'首铲 · 冻结残差挖掘','SCOOP1_RESIDUAL_HOLD':'首铲 · 离土保载',
 'CLEAR_PILE':'首铲 · 带料举升','ALIGN':'首铲 · 回转对准','DUMP':'首铲 · 卸料',
 'CLEAR_TRUCK':'首铲 · 抬离料车','WITHDRAW_SWING':'首铲 · 退出回摆','RETURN_SAFE':'首铲 · 返回',
 'SCOOP2_RETURN_RISE_0':'次铲准备 · 返回上抬','SCOOP2_RETURN_ALIGN_HIGH':'次铲准备 · 高位转向',
 'SCOOP2_RETURN_EXTEND_HIGH':'次铲准备 · 高位伸展','SCOOP2_RETURN_OPEN_HIGH':'次铲准备 · 张斗',
 'SCOOP2_HIGH_CLEAR':'次铲准备 · 高位接近','SCOOP2_OUTWARD_HIGH':'次铲准备 · 外侧通道',
 'SCOOP2_REAR_LOWER_PREPOINT':'次铲准备 · 下降接近','SCOOP2_REFERENCE_ALIGN':'次铲准备 · 人工参考交接',
 'SCOOP2_RESIDUAL':'第二铲 · 冻结残差挖掘','SCOOP2_RESIDUAL_HOLD':'第二铲 · 离土保载'}

def main(smoke):
 import pyglet.display.xlib as xlib
 if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
 import warp as wp,newton
 import run_layered_47_sparse_manual_contact_episode as manual
 from soil_surface import SoilSurface
 start=time.monotonic();limit=300 if smoke else 1800
 data=read(O/'data-manifest.json');A=Path(data['first_run']);B=Path(data['second_run']);cut=data['first_return_frame'];last=data['workflow_last_frame']
 assert data['status']=='source_data_prepared' and data['boundary']['particles_exact'] and data['boundary']['compact_state_exact']
 folders=[A,B];states=[read(x/'display/states.json') for x in folders];phases=[read(x/'display/phases.json') for x in folders]
 clouds=[np.load(x/'display/particles.npy',mmap_mode='r') for x in folders];scenes=[read(x/'display/scene.json') for x in folders]
 assert scenes[0]['records']==scenes[1]['records'] and scenes[0]['root']==scenes[1]['root']
 assert states[0][cut]==states[1][0] and np.array_equal(clouds[0][cut],clouds[1][0])
 for folder in folders:
  for name,identity in data['cache_identities'][str(folder)].items():
   file=folder/'display'/name;assert (file.stat().st_size,file.stat().st_mtime_ns)==(identity['bytes'],identity['mtime_ns'])
 unknown=(set(phases[0][:cut+1])|set(phases[1]))-set(LABELS);assert not unknown,unknown
 dig=list(csv.DictReader((O/'data/dig-timeseries.csv').open(encoding='utf-8-sig')))
 progress={int(x['workflow_frame']):float(x['reference_progress']) for x in dig}
 first_end=read(A/'report.json')['scoops'][0]['end_global_frame']
 def find(i,cursor):
  rows=[x for x in dig if int(x['scoop'])==i];return int(min(rows,key=lambda x:abs(float(x['reference_progress'])-cursor))['workflow_frame'])
 dump_end=next(x['end_frame'] for x in read(A/'report.json')['phases'] if x['name']=='DUMP')
 keyframes=[(find(1,660),'01-first-cut','首铲挖掘'),(first_end,'02-first-hold','首铲保载'),
            (dump_end,'03-unloaded','首铲卸料结束'),(cut,'04-returned','已验证快照续接点'),
            (find(2,660),'05-second-cut','第二铲挖掘'),(last,'06-second-hold','第二铲保载')]
 display=read(R/'configs/v4/layered-47-interactive-proxy-practice-v1.json')['display'];width,height=1920,1080
 wp.init();wp.set_device('cuda:0');assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
 viewer=newton.viewer.ViewerGL(width=width,height=height,vsync=False,paused=True,headless=True);viewer.set_model(model);viewer.picking_enabled=False
 viewer.set_camera(wp.vec3(*display['camera_position_m']),pitch=display['camera_pitch_deg'],yaw=display['camera_yaw_deg'])
 positions=wp.array(clouds[0][0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions))
 surface=SoilSurface(positions,particle_spacing_m=scenes[0]['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
 manual.display_helpers._prepare_rock_visuals(viewer,scenes[0]['records'],Path(scenes[0]['root']),'cuda:0')
 selected=scenes[0]['selected_cut'];face=selected.get('contact',selected)
 goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
 fontpath='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontpath,38);small=ImageFont.truetype(fontpath,27);large=ImageFont.truetype(fontpath,66)
 def locate(g):return (0,g) if g<=cut else (1,g-cut)
 def image_frame(g,card=False):
  if time.monotonic()-start>limit:raise TimeoutError('cache_render_time_limit')
  run_id,frame=locate(g);state=states[run_id][frame];positions.assign(clouds[run_id][frame])
  manual.draw_with_selected_cut(viewer,model_state,fixture,{**state,'bucket_origin':state['bucket_origin_world_m']},scenes[0]['records'],assembly,names,to_pose,'cuda:0',display,goal,frame,surface,show_target=False)
  image=Image.fromarray(viewer.get_frame(render_ui=False).numpy());assert image.size==(width,height)
  draw=ImageDraw.Draw(image);draw.rectangle((0,0,width,116),fill=(16,30,44))
  draw.text((28,10),'同一土堆 · 两铲作业回放',font=font,fill='white')
  label=LABELS[phases[run_id][frame]];text=f'{label}  |  流程仿真时间 {g/60:.2f}s'
  if g in progress:text+=f'  |  参考 {progress[g]:.0f}/1170'
  draw.text((28,66),text,font=small,fill=(197,220,235))
  if g>=first_end:draw.text((1410,17),'首铲保载 149.96%',font=small,fill=(255,206,118))
  if g==last:draw.text((1410,63),'次铲保载 109.94%',font=small,fill=(105,211,224))
  draw.rectangle((0,height-55,width,height),fill=(16,30,44))
  footer='土堆演化连续保留 · 含已验证原生快照续接 · 冻结模型'
  if abs(g-cut)<=180:footer='已验证快照续接：土石与机构初态一致，原连续首3帧复现通过'
  draw.text((28,height-49),footer,font=small,fill=(208,224,234))
  draw.rectangle((0,height-5,int(width*g/last),height),fill=(42,174,192))
  if card:
   draw.rounded_rectangle((390,680,1530,955),radius=22,fill=(16,30,44))
   draw.text((435,704),'结果停帧 · 0.5秒保载最低显示',font=font,fill=(199,219,232))
   draw.text((435,775),'首铲 149.96%',font=large,fill=(255,206,118));draw.text((1010,775),'次铲 109.94%',font=large,fill=(105,211,224))
   draw.text((435,879),'本次两铲均完成1170参考；显示值未作装载体积标定',font=small,fill=(199,219,232))
  return image
 output=O/('smoke' if smoke else 'video');output.mkdir(exist_ok=False);shots=output/'keyframes';shots.mkdir();review=O/('smoke-review.json' if smoke else 'video-render.json')
 report=dict(status='rendering',renderer_sha256=sha(__file__),data_manifest_sha256=sha(O/'data-manifest.json'),
  width=width,height=height,fps=30,cache_identities=data['cache_identities'],workflow_last_frame=last,workflow_seconds=last/60,cutoff=cut,
  continuity='First valid returned prefix plus strictly verified restored59um branch; seam explicitly labelled; abandoned1um rise excluded and preserved in source.',
  playback_physics_steps=0,policy_inference_calls=0,training_updates=0,keyframes=[],frames_rendered=0)
 write(review,report)
 try:
  for _ in range(3):image_frame(0)
  for g,name,label in keyframes:
   file=shots/(name+'.png');image_frame(g).save(file);report['keyframes'].append(dict(workflow_frame=g,label=label,file=str(file.relative_to(O)),sha256=sha(file)));print('KEYFRAME',g,label,flush=True)
  if smoke:
   t=time.monotonic()
   for g in np.linspace(0,last,24,dtype=int):image_frame(int(g))
   report.update(status='smoke_passed',benchmark_seconds_per_frame=(time.monotonic()-t)/24,wall_seconds=time.monotonic()-start);write(review,report);print('SMOKE_PASSED',report['benchmark_seconds_per_frame'],flush=True);return
  indices=list(range(0,last+1,2))
  if indices[-1]!=last:indices.append(last)
  schedule=[(g,False) for g in indices]+[(last,True)]*90
  movie=output/'two-scoop-full-1x.mp4'
  cmd=['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','30','-i','-',
       '-an','-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(movie)]
  with (output/'frame-map.csv').open('w',newline='') as file,subprocess.Popen(cmd,stdin=subprocess.PIPE) as process:
   writer=csv.writer(file);writer.writerow(['video_frame','workflow_frame','source_run','source_frame','workflow_s','phase','kind'])
   try:
    for i,(g,card) in enumerate(schedule):
     run_id,f=locate(g);process.stdin.write(np.asarray(image_frame(g,card)).tobytes());writer.writerow([i,g,run_id+1,f,g/60,phases[run_id][f],'result_freeze' if card else 'cache_frame']);report['frames_rendered']=i+1
     if i%300==0:write(review,report);print('VIDEO_FRAME',i,'/',len(schedule),'elapsed',round(time.monotonic()-start),flush=True)
    process.stdin.close();assert process.wait(timeout=90)==0
   except BaseException:
    if process.stdin and not process.stdin.closed:process.stdin.close()
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
    raise
  for folder in folders:
   for name,x in data['cache_identities'][str(folder)].items():
    file=folder/'display'/name;assert(file.stat().st_size,file.stat().st_mtime_ns)==(x['bytes'],x['mtime_ns'])
  report.update(status='rendered',movie=str(movie.relative_to(O)),movie_sha256=sha(movie),encoded_frames=len(schedule),duration_s=len(schedule)/30,
   nominal_stride=2,result_freeze_frames=90,wall_seconds=time.monotonic()-start);write(review,report);print('VIDEO_RENDERED',report['duration_s'],flush=True)
 except BaseException as exc:
  report.update(status='failed',failure=repr(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-start);write(review,report);raise
 finally:viewer.close()
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');main(parser.parse_args().smoke)
