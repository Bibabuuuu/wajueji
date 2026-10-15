"""Read-only actual mouth-path projections; not removed-volume measurement."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-qerror-long-20261014-v1';sys.path.insert(0,str(W))
import run_long_qerror as m
from planning.cycle_geometry import MOUTH
from planning.contact_footprint import cut_opportunity
OUT=R/'work/second-scoop-strip-review-20261015-v1';OUT.mkdir(exist_ok=False)
first=R/'runs/continuous-single-scoop-20261014-v1';second=m.OUT
read=m.read
rows=lambda p:[json.loads(s) for s in Path(p).read_text().splitlines()]
p1=read(first/'policy/episode-0001/report.json');a1=rows(first/'policy/episode-0001/actions.ndjson');a2=rows(second/'policy/episode-0001/actions.ndjson')
s1=read(first/'display/states.json');s2=read(second/'display/states.json');base=p1['actual_entry_global_frame']
r=m.CpuRuntime(second/'zero.npz',OUT/'cpu-runtime',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
def mouthpath(actions,states,offset):
 chosen=[a for a in actions if 120<=a['cursor_before']<=840 and a['frame']%3==1]
 out=[]
 for a in chosen:
  state=states[offset+a['frame']-1];pose=r.assembly.assemble(np.asarray(state['mechanism_qpos']),include_meshes=False)[1]['bucket']
  center=pose[:3,:3]@MOUTH+pose[:3,3];side=pose[:3,1]*.3
  out.append(dict(frame=offset+a['frame']-1,reference_progress=a['cursor_before'],center=center,left=center+side,right=center-side))
 return out
paths=[mouthpath(a1,s1,base),mouthpath(a2,s2,0)]
origin=np.array([3.6,-1.3]);cell=.01;shape=(270,300)
def pixels(p):return tuple((np.asarray(p)[:2]-origin)/cell)
masks=[]
for path in paths:
 mask=Image.new('L',(shape[1],shape[0]),0);draw=ImageDraw.Draw(mask)
 for a,b in zip(path,path[1:]):draw.polygon([pixels(x) for x in [a['left'],a['right'],b['right'],b['left']]],fill=255)
 masks.append(np.asarray(mask)>0)
candidates=read(R/'work/second-scoop-continuation-20261014-v1/cpu-v4/candidates.json');c=candidates['candidates'][0]
first_contact=np.array([5.849992,-.051770,.650709]);second_contact=np.r_[c['contact']['contact_xy_m'],c['contact']['contact_surface_z_m']]
angle=np.arctan2(second_contact[1],second_contact[0])-np.arctan2(first_contact[1],first_contact[0])
data=dict(status='cpu_strip_projection_measured',first_source=str(first),second_source=str(second),first_policy_start_global_frame=base,first_report_sha256=m.sha(first/'report.json'),second_report_sha256=m.sha(second/'report.json'),nominal_mouth_width_m=.60,reference_progress_window=[120,840],sample_stride_physical_frames=3,first_projected_area_m2=float(masks[0].sum()*cell**2),second_projected_area_m2=float(masks[1].sum()*cell**2),overlap_projected_area_m2=float(np.count_nonzero(masks[0]&masks[1])*cell**2),overlap_fraction_of_second_projection=float(np.count_nonzero(masks[0]&masks[1])/max(1,masks[1].sum())),first_reference_contact_m=first_contact,second_selected_contact_m=second_contact,entry_center_separation_m=float(np.linalg.norm(first_contact[:2]-second_contact[:2])),entry_heading_separation_rad=float(angle),selected_cut_opportunity=c['cut_opportunity'],selection_version=c['selection_version'],future_strip_count=c['future_strip_count'],source_code_sha256={str(p):m.sha(p) for p in [R/'src/v4/layered_47_cycle_selection.py',R/'planning/contact_footprint.py']},physical_steps=0,training_updates=0,limitation='0.60m nominal mouth segments from actual FK are projected toXY and rasterized. This is not excavated volume, swept solid mesh, available soil, overlap of displaced/slumped soil, or proof of root cause. Entry center1 is recorded reference contact, not actual first tooth contact.')
# Use the existing actual post-first selection-time particle height background.
with np.load(first/'final.npz') as z:soil=z['particle_q']
ij=np.floor((soil[:,:2]-origin)/cell).astype(int);keep=(ij[:,0]>=0)&(ij[:,0]<shape[1])&(ij[:,1]>=0)&(ij[:,1]<shape[0]);heights=np.full(shape[0]*shape[1],-np.inf);np.maximum.at(heights,ij[keep,1]*shape[1]+ij[keep,0],soil[keep,2]);h=heights.reshape(shape)
t=np.clip(np.where(np.isfinite(h),h,0)/1.6,0,1)[...,None];rgb=np.rint(np.array([104,88,64])+t*np.array([126,122,106])).astype('uint8');rgb[~np.isfinite(h)]=[229,231,232]
for mask,color in zip(masks,[[235,125,45],[30,166,202]]):rgb[mask]=np.rint(rgb[mask]*.55+np.array(color)*.45).astype('uint8')
rgb[masks[0]&masks[1]]=[187,92,190]
im=Image.new('RGB',(1500,1250),'#f5f3ed');plot=Image.fromarray(rgb[::-1]).resize((1200,1080),Image.Resampling.NEAREST);im.paste(plot,(160,125));d=ImageDraw.Draw(im);font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';large=ImageFont.truetype(font,34);small=ImageFont.truetype(font,23)
d.text((25,15),'两铲实际斗口路径 · 0.60米宽度的俯视投影',font=large,fill='#152737');d.text((25,65),'橙：首铲   蓝：第二铲   紫：投影重叠；底图为首铲返回后的土粒高度。',font=small,fill='#374b56')
def xy(p):return 160+(p[0]-origin[0])/cell*4,125+(shape[0]-(p[1]-origin[1])/cell)*4
for path,color in zip(paths,['#c65907','#007b9f']):
 pts=[xy(a['center']) for a in path];d.line(pts,fill=color,width=3)
for c0,color,label in [(first_contact,'#c65907','首铲参考切入'),(second_contact,'#007b9f','第二铲选中切入')]:
 px,py=xy(c0);d.ellipse((px-8,py-8,px+8,py+8),fill=color,outline='white',width=2);d.text((px+14,py-14),label,font=small,fill=color)
d.text((25,1210),'投影带仅用于核对横向排布；不能据此推断实际挖空体积或满斗率。',font=small,fill='#374b56')
im.save(OUT/'two-scoop-mouth-strips.png')
np.savez(OUT/'projection-masks.npz',first=masks[0],second=masks[1],origin_xy_m=origin,resolution_m=cell)
m.write(OUT/'report.json',data);m.write(OUT/'mouth-paths.json',dict(first=paths[0],second=paths[1]));print(m.encode({k:v for k,v in data.items() if k not in ['selected_cut_opportunity','source_code_sha256']}),flush=True)
