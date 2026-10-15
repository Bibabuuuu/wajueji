"""Render immutable simulation caches only; no solver steps or policy calls."""
import os
os.environ.setdefault('DISPLAY',':1');os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,time,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/qunwang/wajueji');RUN=R/'runs/second-scoop-qerror-long-20261014-v1';CACHE=RUN/'display'
OUT=R/'runs/second-scoop-keyframes-20261014-v1'
CP=R/'work/second-scoop-continuation-20261014-v1/cpu-v4/candidates.json'
ROT=R/'work/second-scoop-continuation-20261014-v1/rotation-probe-v2/report.json'
FIRST=R/'runs/continuous-single-scoop-20261014-v1'
sys.path[:0]=[str(R/'src/v4'),str(R)]
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=time.monotonic();OUT.mkdir(exist_ok=False);(OUT/'frames').mkdir()
 report=read(RUN/'report.json');audit=read(RUN/'read-only-audit.json');meta=read(CACHE/'reconstruction.json')
 assert report['status']=='full_second_dig_completed' and audit['full_second_dig_passed'] and all(audit['checks'].values())
 for name,h in meta['cache_sha256'].items():assert sha(CACHE/name)==h,name
 identity={name:dict(sha256=h,size=(CACHE/name).stat().st_size,mtime_ns=(CACHE/name).stat().st_mtime_ns) for name,h in meta['cache_sha256'].items()}
 candidates=read(CP);chosen=candidates['candidates'][0]['contact'];rot=read(ROT)
 assert sha(FIRST/'final.npz')==candidates['source_sha256']
 with np.load(FIRST/'final.npz') as z:first_soil=z['particle_q'].copy()
 first_states=read(FIRST/'display/states.json');first_state=first_states[-1]
 assert np.array_equal(first_soil,np.load(FIRST/'display/particles.npy',mmap_mode='r')[-1])
 states=read(CACHE/'states.json');scene=read(CACHE/'scene.json');cloud=np.load(CACHE/'particles.npy',mmap_mode='r')
 actions=[json.loads(x) for x in (RUN/'policy/episode-0001/actions.ndjson').read_text().splitlines()]
 first_scene=read(FIRST/'display/scene.json')
 assert len(states)==len(cloud)==2943
 display=read(R/'configs/v4/layered-47-interactive-proxy-practice-v1.json')['display']
 import pyglet.display.xlib as xlib
 if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
 import newton,warp as wp
 import run_layered_47_sparse_manual_contact_episode as manual
 from soil_surface import SoilSurface
 wp.init();wp.set_device('cuda:0');assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
 viewer=newton.viewer.ViewerGL(width=1440,height=900,vsync=False,paused=True,headless=True);viewer.set_model(model);viewer.picking_enabled=False
 positions=wp.array(cloud[0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions))
 surface=SoilSurface(positions,particle_spacing_m=scene['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
 manual.display_helpers._prepare_rock_visuals(viewer,scene['records'],Path(scene['root']),'cuda:0')
 goal=dict(entry_xy_m=chosen['contact_xy_m'],surface_z_m=chosen['contact_surface_z_m'])
 layer='all';original_log=viewer.log_state
 def log_state(s):
  original_log(s)
  for key,obj in viewer.objects.items():
   if key.startswith('/model/shapes/shape_'):
    idx=int(key.rsplit('_',1)[1]);obj.hidden=model.shape_label[idx].startswith('truck_') or (layer=='terrain' and idx<13)
 viewer.log_state=log_state
 def camera(pos,target):
  delta=np.asarray(target)-np.asarray(pos)
  viewer.set_camera(wp.vec3(*pos),pitch=math.degrees(math.atan2(delta[2],np.linalg.norm(delta[:2]))),yaw=math.degrees(math.atan2(delta[1],delta[0])))
 fontfile='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
 font=ImageFont.truetype(fontfile,36);small=ImageFont.truetype(fontfile,24)
 def render(state,soil,n,terrain=False,target=False):
  nonlocal layer
  if time.monotonic()-start>300:raise TimeoutError('offline_render300s')
  layer='terrain' if terrain else 'all';positions.assign(soil)
  # Use a unique display update token for each source/frame, also when the
  # selection snapshot precedes the second-scoop cache. No state evolves.
  manual.draw_with_selected_cut(viewer,model_state,fixture,{**state,'bucket_origin':state['bucket_origin_world_m']},scene['records'],assembly,names,to_pose,'cuda:0',display,goal,n,surface,show_target=target)
  return Image.fromarray(np.array(viewer.get_frame(render_ui=False).numpy(),copy=True))
 rows=[]
 try:
  camera([8.3,-3.7,4.6],[5.45,.1,.9])
  for _ in range(2):render(first_state,first_soil,10000,terrain=True,target=True)
  selection=render(first_state,first_soil,10000,terrain=True,target=True);selection.save(OUT/'candidate-scene.png')
  camera([7.6,-3.7,3.8],[5.25,.65,1.75])
  keys=[(0,'01-prepoint','预备点'),(154,'02-entry','前段入土'),(2100,'03-dig','低位挖掘'),(2389,'04-curl-lift','收斗抬升'),(2629,'05-loaded-exit','离土带料'),(2942,'06-hold','保载结束')]
  for n,name,label in keys:
   im=render(states[n],cloud[n],n);draw=ImageDraw.Draw(im);cursor=0 if n==0 else actions[n-1]['cursor_before']
   draw.rectangle((0,0,1440,104),fill=(16,27,39));draw.text((22,8),f'{label}  |  第二铲真实缓存',font=font,fill='white')
   draw.text((22,59),f'局部时间 {n/60:.2f} s   ·   帧 {n}   ·   参考进度 {cursor:.0f}/1170',font=small,fill=(188,215,233))
   if n==2942:draw.text((940,59),'保载显示 68.18%',font=small,fill=(255,205,100))
   path=OUT/'frames'/f'{name}.png';im.save(path)
   rows.append(dict(file=str(path.relative_to(OUT)),frame=n,label=label,time_s=n/60,reference_cursor=cursor,sha256=sha(path)))
   print('FRAME',name,n,flush=True)
 finally:viewer.close()
 # Height map is sampled from the actual selection-time soil, not a schematic.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.font_manager import FontProperties
 fp=FontProperties(fname=fontfile)
 x0,x1,y0,y1,cell=4.65,6.3,-1.12,1.12,.025
 nx=int(np.ceil((x1-x0)/cell));ny=int(np.ceil((y1-y0)/cell));grid=np.full(nx*ny,np.nan)
 ij=np.floor((first_soil[:,:2]-[x0,y0])/cell).astype(int);keep=(ij[:,0]>=0)&(ij[:,0]<nx)&(ij[:,1]>=0)&(ij[:,1]<ny)
 heights=np.full(nx*ny,-np.inf);np.maximum.at(heights,ij[keep,1]*nx+ij[keep,0],first_soil[keep,2]);grid=np.where(np.isfinite(heights),heights,np.nan).reshape(ny,nx)
 fig,ax=plt.subplots(figsize=(8,7.5),dpi=160);fig.patch.set_facecolor('#f5f3ed');ax.set_facecolor('#e7e3d9')
 ax.imshow(grid,origin='lower',extent=[x0,x0+nx*cell,y0,y0+ny*cell],cmap='copper',vmin=0,vmax=1.6,interpolation='nearest')
 offsets=[(6.12,.95),(6.12,-.96),(5.74,-1.04),(5.45,.99),(6.13,-.35),(5.25,-.98),(5.79,-.23),(5.25,-.33)]
 for i,c in enumerate(candidates['candidates']):
  x,y=c['contact']['contact_xy_m'];color='#14b86d' if i==0 else '#eb9b2a';ax.scatter(x,y,s=55,color=color,edgecolor='white',zorder=4)
  ax.annotate(str(i+1)+(' 已选' if i==0 else ''),(x,y),xytext=offsets[i],ha='center',va='center',fontproperties=fp,fontsize=12,color='#113022' if i==0 else '#4c3013',bbox=dict(boxstyle='round,pad=.3',fc='#d4f4df' if i==0 else '#fff1d6',ec=color),arrowprops=dict(arrowstyle='-',color=color,lw=1.4),zorder=5)
 ax.scatter(*rot['rotated_reference_contact_m'][:2],marker='x',s=65,color='#00d8e8',linewidths=2,zorder=6)
 ax.set_xlabel('世界坐标 X / m',fontproperties=fp,fontsize=12);ax.set_ylabel('世界坐标 Y / m',fontproperties=fp,fontsize=12);ax.set_aspect('equal');ax.set_xlim(x0,x1);ax.set_ylim(y0,y1)
 ax.set_title('选点时真实土堆 · 俯视高度图\n绿点：选定切入点   青叉：迁移后的参考切入点',fontproperties=fp,fontsize=14,pad=14)
 fig.subplots_adjust(left=.12,right=.97,bottom=.1,top=.88);fig.savefig(OUT/'candidate-map.png');plt.close(fig)
 overview=Image.new('RGB',(2400,1030),'#f5f3ed');overview.paste(selection,(0,100));overview.paste(Image.open(OUT/'candidate-map.png').resize((960,900)),(1440,100))
 dr=ImageDraw.Draw(overview);dr.rectangle((0,0,2400,100),fill=(16,27,39));dr.text((25,10),'第二铲候选位置｜首铲返回后的真实土堆',font=font,fill='white');dr.text((25,60),'共8个候选；1号最终选中。选定点与迁移参考切入点的三维距离约4.72厘米。',font=small,fill=(198,220,236));dr.text((25,1003),'左：绿色柱标记选中切入点，青框为计划斗宽。右：候选点编号按生成排序。',font=small,fill=(55,66,73));overview.save(OUT/'candidate-overview.png')
 montage=Image.new('RGB',(1920,1875),(16,27,39));dr=ImageDraw.Draw(montage);dr.text((25,15),'第二铲挖掘关键帧｜同一次运行 · 固定相机 · 原始状态缓存',font=font,fill='white')
 for i,row in enumerate(rows):montage.paste(Image.open(OUT/row['file']).resize((960,600)),((i%2)*960,75+(i//2)*600))
 montage.save(OUT/'digging-keyframes.png')
 for name,d in identity.items():assert (CACHE/name).stat().st_size==d['size'] and (CACHE/name).stat().st_mtime_ns==d['mtime_ns']
 result=dict(status='rendered',source_run=str(RUN),source_report_sha256=sha(RUN/'report.json'),source_audit_sha256=sha(RUN/'read-only-audit.json'),candidate_source=str(FIRST/'final.npz'),candidate_source_sha256=candidates['source_sha256'],candidate_file_sha256=sha(CP),cache_identity=identity,keyframes=rows,selected_contact=chosen['contact_xy_m']+[chosen['contact_surface_z_m']],reference_contact=rot['rotated_reference_contact_m'],contact_offset_m=rot['contact_offset_norm_m'],display_changes='Fixed close camera, truck hidden; selection view hides machine only. Native cached geometry and materials unchanged. Height map bins actual selection-time soil. Overlays/number callouts are visual annotations.',new_physics_steps=0,policy_inference_calls=0,training_updates=0,renderer_sha256=sha(__file__),wall_seconds=time.monotonic()-start)
 (OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('RENDERED',len(rows),round(result['wall_seconds'],2),flush=True)
if __name__=='__main__':main()
