"""Read-only native cached frames; no physical step or policy inference."""
import os
os.environ.setdefault('DISPLAY',':1');os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
from pathlib import Path
from types import SimpleNamespace
import sys,json,time,math,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/qunwang/wajueji');D=R/'runs/edge-first-two-scoop-20261015-continuation';OUT=D/'media-v2';sys.path[:0]=[str(R/'src/v4'),str(R)]
read=lambda p:json.loads(Path(p).read_text())
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()] if Path(p).exists() else []
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()

def main():
 start=time.monotonic();p=read(D/'report.json');audit=read(D/'read-only-audit.json');assert audit['record_integrity_passed'],'audit_required_before_media'
 OUT.mkdir(exist_ok=False);(OUT/'frames').mkdir();cache=D/'display';meta=read(cache/'reconstruction.json');states=read(cache/'states.json');scene=read(cache/'scene.json');cloud=np.load(cache/'particles.npy',mmap_mode='r');n=p['frames']
 assert len(states)==len(cloud)==n+1
 identities={k:dict(size=(cache/k).stat().st_size,mtime_ns=(cache/k).stat().st_mtime_ns,sha256=v) for k,v in meta['cache_sha256'].items()}
 requested=[]
 for folder in sorted(D.glob('policy-*/episode-0001')):
  rep=read(folder/'report.json');cycle=int(folder.parent.name.split('-')[-1]);offset=rep['actual_entry_global_frame'];actions=rows(folder/'actions.ndjson');endpoint=offset+rep.get('physical_frames',len(actions));end=min(n,endpoint)
  sequence=[(offset,'预备点',0.)]
  for target,label in [(180,'入土前段'),(660,'挖掘中段')]:
   if actions and max(x['cursor_before'] for x in actions)>=target:
    chosen=min(actions,key=lambda x:abs(x['cursor_before']-target));sequence.append((offset+chosen['frame']-1,label,chosen['cursor_before']))
  sequence.append((end,'保载结束' if rep.get('success') else '停止末态',rep.get('cursor',0.)))
  selected=read(D/('scoop-%d-plan'%cycle)/'report.json')['selected']['contact']
  for frame,label,cursor in sequence:requested.append(dict(scoop=cycle,frame=frame,label=label,cursor=cursor,selected=selected,fill=rep.get('retained_fill_fraction') if label=='保载结束' else None))
 if not any(x['scoop']==2 for x in requested) and (D/'scoop-2-plan/report.json').exists():
  selected=read(D/'scoop-2-plan/report.json')['selected']['contact'];begin=read(D/'first-returned.control.json')['frame']
  for frame,label in [(begin,'返回起点（上抬前）'),(n,'上抬末帧停止')]:requested.append(dict(scoop=2,frame=frame,label=label,cursor=None,selected=selected,fill=None))
 if not requested:
  selected=read(D/'scoop-1-plan/report.json')['selected']['contact']
  for frame in sorted(set([0,n//3,2*n//3,n])):requested.append(dict(scoop=1,frame=frame,label='接近过程' if frame<n else '停止末态',cursor=None,selected=selected,fill=None))
 display=read(R/'configs/v4/layered-47-interactive-proxy-practice-v1.json')['display']
 import pyglet.display.xlib as xlib
 if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
 import newton,warp as wp
 import run_layered_47_sparse_manual_contact_episode as manual
 from soil_surface import SoilSurface
 wp.init();wp.set_device('cuda:0');assembly,model,model_state,names,to_pose=manual.display_helpers._display_runtime(display)
 viewer=newton.viewer.ViewerGL(width=1440,height=900,vsync=False,paused=True,headless=True);viewer.set_model(model);viewer.picking_enabled=False
 positions=wp.array(cloud[0],dtype=wp.vec3,device='cuda:0');fixture=SimpleNamespace(state=SimpleNamespace(particle_q=positions));surface=SoilSurface(positions,particle_spacing_m=scene['particle_spacing_m'],voxel_size_m=display['surface_voxel_m'])
 manual.display_helpers._prepare_rock_visuals(viewer,scene['records'],Path(scene['root']),'cuda:0');original=viewer.log_state
 def log_state(state):
  original(state)
  for key,obj in viewer.objects.items():
   if key.startswith('/model/shapes/shape_'):
    idx=int(key.rsplit('_',1)[1]);obj.hidden=model.shape_label[idx].startswith('truck_')
 viewer.log_state=log_state
 pos=np.array([7.8,-3.9,4.2]);target=np.array([5.2,.3,1.65]);delta=target-pos
 viewer.set_camera(wp.vec3(*pos),pitch=math.degrees(math.atan2(delta[2],np.linalg.norm(delta[:2]))),yaw=math.degrees(math.atan2(delta[1],delta[0])))
 font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';large=ImageFont.truetype(font,34);small=ImageFont.truetype(font,24);outputs=[]
 try:
  for index,item in enumerate(requested):
   if time.monotonic()-start>360:raise TimeoutError('read_only_render360s')
   frame=item['frame'];
   if item['scoop']==2:
    p2=np.array([8.5,-5.5,5.4]);t2=np.array([4.5,.2,2.]);d2=t2-p2;viewer.set_camera(wp.vec3(*p2),pitch=math.degrees(math.atan2(d2[2],np.linalg.norm(d2[:2]))),yaw=math.degrees(math.atan2(d2[1],d2[0])))
   positions.assign(cloud[frame]);selected=item['selected'];goal=dict(entry_xy_m=selected['contact_xy_m'],surface_z_m=selected['contact_surface_z_m']);state={**states[frame],'bucket_origin':states[frame]['bucket_origin_world_m']}
   for _ in range(2):manual.draw_with_selected_cut(viewer,model_state,fixture,state,scene['records'],assembly,names,to_pose,'cuda:0',display,goal,frame,surface,show_target=False)
   image=Image.fromarray(np.array(viewer.get_frame(render_ui=False).numpy(),copy=True));draw=ImageDraw.Draw(image);draw.rectangle((0,0,1440,110),fill=(18,31,45));draw.text((22,8),f'第{item["scoop"]}铲 · {item["label"]}',font=large,fill='white')
   caption=f'真实缓存帧 {frame}  |  从恢复点起 {frame/60:.2f} s'
   if item['cursor'] is not None:caption+=f'  |  参考 {item["cursor"]:.0f}/1170'
   draw.text((22,62),caption,font=small,fill=(195,216,232))
   if item['fill'] is not None:draw.text((1060,62),f'保载 {item["fill"]*100:.2f}%',font=small,fill=(255,207,112))
   file=OUT/'frames'/f'scoop-{item["scoop"]}-{index:02d}-f{frame:05d}.png';image.save(file);outputs.append(dict(item,file=str(file.relative_to(OUT)),sha256=sha(file)));print('MEDIA_FRAME',item['scoop'],frame,item['label'],flush=True)
 finally:viewer.close()
 for cycle in sorted(set(x['scoop'] for x in outputs)):
  subset=[x for x in outputs if x['scoop']==cycle];height=70+600*math.ceil(len(subset)/2);plate=Image.new('RGB',(1920,height),(18,31,45));draw=ImageDraw.Draw(plate);draw.text((25,12),(f'第{cycle}铲准备关键帧 · 返回上抬，尚未下铲' if cycle==2 and not audit['two_scoop_goal_complete'] else f'第{cycle}铲真实关键帧 · 边缘首铲'),font=large,fill='white')
  for j,item in enumerate(subset):plate.paste(Image.open(OUT/item['file']).resize((960,600)),((j%2)*960,70+(j//2)*600))
  plate.save(OUT/f'scoop-{cycle}-keyframes.png')
 for name,x in identities.items():assert (cache/name).stat().st_size==x['size'] and (cache/name).stat().st_mtime_ns==x['mtime_ns'],'source_modified_during_render'
 manifest=dict(status='rendered',source_report_sha256=sha(D/'report.json'),source_audit_sha256=sha(D/'read-only-audit.json'),run_status=p['status'],two_scoop_complete=audit['two_scoop_goal_complete'],cache_identity=identities,keyframes=outputs,new_physics_steps=0,policy_inference_calls=0,training_updates=0,display_changes='Fixed close camera for scoop1 and wider overview for scoop2 preparation; truck hidden; captions added. Geometry and materials from same native cached frames. No cross-run montage presented as continuous video.',wall_seconds=time.monotonic()-start)
 (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');print('MEDIA_DONE',len(outputs),flush=True)
if __name__=='__main__':main()



