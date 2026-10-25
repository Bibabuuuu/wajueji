"""PIL-only labels/height map for six already rendered native cache frames."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';OUT=R/'runs/second-scoop-keyframes-20261014-v1';RUN=R/'runs/second-scoop-qerror-long-20261014-v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
CP=R/'work/second-scoop-continuation-20261014-v1/cpu-v4/candidates.json';ROT=R/'work/second-scoop-continuation-20261014-v1/rotation-probe-v2/report.json';FIRST=R/'runs/continuous-single-scoop-20261014-v1'
candidates=read(CP);rot=read(ROT);chosen=candidates['candidates'][0]['contact']
assert sha(FIRST/'final.npz')==candidates['source_sha256']
with np.load(FIRST/'final.npz') as z:soil=z['particle_q']
x0,y0,cell=4.65,-1.12,.025;nx,ny=66,90;heights=np.full(nx*ny,-np.inf)
ij=np.floor((soil[:,:2]-[x0,y0])/cell).astype(int);keep=(ij[:,0]>=0)&(ij[:,0]<nx)&(ij[:,1]>=0)&(ij[:,1]<ny)
np.maximum.at(heights,ij[keep,1]*nx+ij[keep,0],soil[keep,2]);grid=heights.reshape(ny,nx)
level=np.clip(np.where(np.isfinite(grid),grid,0)/1.6,0,1)[...,None]**.75
rgb=np.rint(np.array([78,61,40])+level*np.array([161,145,119])).astype('uint8');rgb[~np.isfinite(grid)]=[231,227,217]
fontfile='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font=ImageFont.truetype(fontfile,36);small=ImageFont.truetype(fontfile,24);tiny=ImageFont.truetype(fontfile,21)
plan=Image.new('RGB',(960,900),'#f5f3ed');d=ImageDraw.Draw(plan);left,top,scale=225,115,300
width,height=round(nx*cell*scale),round(ny*cell*scale);plan.paste(Image.fromarray(rgb[::-1]).resize((width,height),Image.Resampling.NEAREST),(left,top))
def xy(x,y):return left+(x-x0)*scale,top+(y0+ny*cell-y)*scale
d.text((45,20),'选点时真实土堆 · 俯视高度图',font=ImageFont.truetype(fontfile,30),fill='#263138');d.text((45,65),'绿点：选中切入点   青叉：迁移参考切入点',font=tiny,fill='#46535b')
d.rectangle((left,top,left+width,top+height),outline='#6c746f',width=2)
for x in [4.8,5.2,5.6,6.0]:
 px,py=xy(x,y0);d.line((px,py,px,py+5),fill='#46535b',width=2);d.text((px-16,py+9),f'{x:.1f}',font=tiny,fill='#263138')
for y in [-1.,-.5,0.,.5,1.]:
 px,py=xy(x0,y);d.line((px-5,py,px,py),fill='#46535b',width=2);d.text((px-64,py-13),f'{y:.1f}',font=tiny,fill='#263138')
d.text((365,835),'世界坐标 X / m',font=small,fill='#263138');d.text((16,420),'Y / m',font=small,fill='#263138');d.text((55,872),'颜色越浅：土面越高；底图来自真实土粒最高点。',font=tiny,fill='#59615c')
offsets=[(6.12,.95),(6.12,-.96),(5.74,-1.04),(5.45,.99),(6.13,-.35),(5.25,-.98),(5.79,-.23),(5.25,-.33)]
for i,c in enumerate(candidates['candidates']):
 p=xy(*c['contact']['contact_xy_m']);q=xy(*offsets[i]);color='#12b569' if i==0 else '#df951d'
 d.line((*p,*q),fill=color,width=3);radius=9 if i==0 else 7;d.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),fill=color,outline='white',width=2)
 text=str(i+1)+(' 已选' if i==0 else '');box=d.textbbox((0,0),text,font=small);tw=box[2]-box[0];th=box[3]-box[1]
 d.rounded_rectangle((q[0]-tw/2-8,q[1]-th/2-9,q[0]+tw/2+8,q[1]+th/2+9),radius=7,fill='#d3f2df' if i==0 else '#fff0d5',outline=color,width=2)
 d.text((q[0]-tw/2,q[1]-th/2-box[1]),text,font=small,fill='#26352c')
cx,cy=xy(*rot['rotated_reference_contact_m'][:2]);d.line((cx-7,cy-7,cx+7,cy+7),fill='#00d8ed',width=3);d.line((cx-7,cy+7,cx+7,cy-7),fill='#00d8ed',width=3)
plan.save(OUT/'candidate-map.png');np.savez(OUT/'selection-height-grid.npz',height=grid,x0=x0,y0=y0,cell=cell)
overview=Image.new('RGB',(2400,1050),'#f5f3ed');overview.paste(Image.open(OUT/'candidate-scene.png'),(0,100));overview.paste(plan,(1440,100))
dr=ImageDraw.Draw(overview);dr.rectangle((0,0,2400,100),fill=(16,27,39));dr.text((25,10),'第二铲候选位置｜首铲返回后的真实土堆',font=font,fill='white');dr.text((25,60),'共8个候选；1号最终选中。选定点与迁移参考切入点的三维距离约4.72厘米。',font=small,fill=(198,220,236));dr.text((25,1003),'左：绿色柱=选定切入点，青框=计划斗宽。右：8个候选的切入点；青叉=迁移参考点。',font=small,fill=(55,66,73));overview.save(OUT/'candidate-overview.png')
keys=[(0,'01-prepoint','预备点'),(154,'02-entry','前段入土'),(2100,'03-dig','低位挖掘'),(2389,'04-curl-lift','收斗抬升'),(2629,'05-loaded-exit','离土带料'),(2942,'06-hold','保载结束')]
actions=[json.loads(s) for s in (RUN/'policy/episode-0001/actions.ndjson').read_text().splitlines()];rows=[]
montage=Image.new('RGB',(1920,1875),(16,27,39));dr=ImageDraw.Draw(montage);dr.text((25,15),'第二铲挖掘关键帧｜同一次运行 · 固定相机 · 原始状态缓存',font=font,fill='white')
for i,(n,name,label) in enumerate(keys):
 path=OUT/'frames'/f'{name}.png';im=Image.open(path);assert im.size==(1440,900)
 montage.paste(im.resize((960,600)),((i%2)*960,75+(i//2)*600));rows.append(dict(file=str(path.relative_to(OUT)),frame=n,label=label,time_s=n/60,reference_cursor=0 if n==0 else actions[n-1]['cursor_before'],sha256=sha(path)))
montage.save(OUT/'digging-keyframes.png')
meta=read(RUN/'display/reconstruction.json');result=dict(status='rendered',source_run=str(RUN),source_report_sha256=sha(RUN/'report.json'),source_audit_sha256=sha(RUN/'read-only-audit.json'),candidate_source=str(FIRST/'final.npz'),candidate_source_sha256=candidates['source_sha256'],candidate_file_sha256=sha(CP),cache_sha256=meta['cache_sha256'],keyframes=rows,selected_contact=chosen['contact_xy_m']+[chosen['contact_surface_z_m']],reference_contact=rot['rotated_reference_contact_m'],contact_offset_m=rot['contact_offset_norm_m'],new_physics_steps=0,policy_inference_calls=0,training_updates=0,rendering_script_sha256=sha(W/'render_second_keyframes.py'),composition_script_sha256=sha(__file__),display_changes='Native cached geometry. Fixed close camera, truck hidden. Selection3D hides machine. Actual selection soil height map, numbering and callouts use saved8 candidates. No geometry or particle changes.',rendering_note='All6 native frames completed. Original matplotlib map stage unavailable; map and montage composed with installed PIL only; no package installation.')
(OUT/'manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256.json'};(OUT/'SHA256.json').write_text(json.dumps(files,indent=2))
print('COMPOSED',len(rows),'keyframes;',len(files),'files')
