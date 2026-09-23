"""First outward crossings of the proxy's closed mid-surface, plus full-wall checks.
Local motion chords are diagnostic, not exact CCD under arbitrary rotation.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from layered_excavation_approach_planner import quaternion_matrix_wxyz

work=Path(__file__).resolve().parent;run=work/'replay-v2'
report=json.loads((work/'verified-replay.json').read_text())
assert report['status']=='replay_paths_recorded' and report['frames']==887
assert report['all_887_recorded_states_exact']
selection=np.load(run/'tracking-selection.npz');ids=selection['ids'];cavity=selection['initial_cavity'];n=len(ids)
pos=np.load(run/'post-projection-world.npy',mmap_mode='r');pre=np.load(run/'pre-projection-world.npy',mmap_mode='r');poses=np.load(run/'actual-bucket-poses.npy',mmap_mode='r')
proxy=json.loads(Path('/home/qunwang/wajueji/reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
curve=np.asarray(proxy['shell_curve_simplified_xz_m']);center=curve.mean(axis=0)
sides=[np.asarray(s['vertices_bucket_m'])[:,1].mean() for s in proxy['side_plate_meshes']]
ymin,ymax=min(sides),max(sides)
def contains_xz(q):
 x,z=q[:,0],q[:,2];inside=np.zeros(len(q),bool)
 for a,b in zip(curve,np.roll(curve,-1,axis=0)):
  if abs(b[1]-a[1])>1e-14:inside ^= ((a[1]>z)!=(b[1]>z)) & (x<(b[0]-a[0])*(z-a[1])/(b[1]-a[1])+a[0])
 return inside

def tolocal(q,pose):return (q-pose[:3,3])@pose[:3,:3]
initial=tolocal(pos[0],poses[0]);inside=contains_xz(initial)&(initial[:,1]>=ymin)&(initial[:,1]<=ymax)
# Only first exits from points initially inside the proxy mid-surface are assigned.
first=np.full(n,-1,dtype=int);events=[];full_cross=[];shells=[]
for box in proxy['shell_boxes']:
 a,b=np.asarray(box['source_segment_xz_m']);d=b-a;normal=np.array([-d[1],d[0]])/np.linalg.norm(d)
 if np.dot(center-a,normal)<0:normal=-normal
 shells.append((box,a,b,d,normal,quaternion_matrix_wxyz(box['quaternion_wxyz'])))

def crossing(a,b,frame,phase):
 active=np.flatnonzero(inside & (first<0))
 if not len(active):return
 q0,q1=a[active],b[active];delta=q1-q0
 best=np.full(len(active),np.inf);labels=np.full(len(active),'',dtype=object);points=np.zeros_like(q0)
 for label,p0,p1 in [(box['name'],a2,b2) for box,a2,b2,d,norm,r in shells]+[('mouth',curve[-1],curve[0])]:
  d=p1-p0;length=np.linalg.norm(d);normal=np.array([-d[1],d[0]])/length
  if np.dot(center-p0,normal)<0:normal=-normal
  da=(q0[:,[0,2]]-p0)@normal;db=(q1[:,[0,2]]-p0)@normal
  candidates=(da>=0)&(db<0)
  t=np.divide(da,da-db,out=np.full_like(da,np.inf),where=candidates)
  # Avoid inf products outside candidates.
  hit=q0+np.where(candidates,t,0)[:,None]*delta
  along=(hit[:,[0,2]]-p0)@d/np.dot(d,d)
  valid=candidates&(along>=-1e-9)&(along<=1+1e-9)&(hit[:,1]>=ymin)&(hit[:,1]<=ymax)&(t<best)
  best[valid]=t[valid];labels[valid]=label;points[valid]=hit[valid]
 for label,y,sign in [('side_negative',ymin,1),('side_positive',ymax,-1)]:
  da=(q0[:,1]-y)*sign;db=(q1[:,1]-y)*sign;candidates=(da>=0)&(db<0)
  t=np.divide(da,da-db,out=np.full_like(da,np.inf),where=candidates)
  hit=q0+np.where(candidates,t,0)[:,None]*delta
  valid=candidates&contains_xz(hit)&(t<best)
  best[valid]=t[valid];labels[valid]=label;points[valid]=hit[valid]
 for j in np.flatnonzero(np.isfinite(best)):
  ix=int(active[j]);first[ix]=frame
  e={'particle_id':int(ids[ix]),'tracked_index':ix,'initial_cavity':bool(cavity[ix]),'frame':frame,'phase':phase,'face':labels[j],'t':float(best[j]),'point_local':points[j].tolist(),'before_local':q0[j].tolist(),'after_local':q1[j].tolist()}
  events.append(e)
  if labels[j].startswith('bucket_shell_'):
   box,a2,b2,d,norm,rot=next(s for s in shells if s[0]['name']==labels[j])
   bx0=(q0[j]-box['position_bucket_m'])@rot;bx1=(q1[j]-box['position_bucket_m'])@rot
   h=np.asarray(box['half_size_m']);inward=1 if np.dot(rot[[0,2],2],norm)>0 else -1
   z0,z1=bx0[2]*inward,bx1[2]*inward
   # Entire segment crosses both physical slab faces; require both hit points on the box.
   if z0>h[2]+1e-6 and z1<-h[2]-1e-6:
    t0=(h[2]-z0)/(z1-z0);t1=(-h[2]-z0)/(z1-z0)
    hit0=bx0+t0*(bx1-bx0);hit1=bx0+t1*(bx1-bx0)
    if np.all(np.abs(hit0[:2])<h[:2]-1e-6) and np.all(np.abs(hit1[:2])<h[:2]-1e-6):full_cross.append({**e,'box_before':bx0.tolist(),'box_after':bx1.tolist(),'inward_normal':inward})

for f in range(1,888):
 old=poses[f-1];new=poses[f]
 a=tolocal(pos[f-1],old);b=tolocal(pre[f-1],old);c=tolocal(pos[f],old);d=tolocal(pos[f],new)
 crossing(a,b,f,'mpm_before_projection')
 crossing(b,c,f,'project_outside')
 crossing(c,d,f,'rigid_pose_advance')
# Check material exit within <=5 frames after each shell mid-plane crossing.
for e in events:
 if not e['face'].startswith('bucket_shell_'):continue
 box,a2,b2,d,norm,rot=next(s for s in shells if s[0]['name']==e['face']);ix=e['tracked_index'];f=e['frame']
 inward=1 if np.dot(rot[[0,2],2],norm)>0 else -1;h=np.asarray(box['half_size_m'])
 outcomes=[]
 for t in range(f,min(f+6,888)):
  q=tolocal(pos[t,ix:ix+1],poses[t])[0];bp=(q-box['position_bucket_m'])@rot
  outcomes.append({'frame':t,'outside_normal_distance_m':float(-bp[2]*inward-h[2]),'tangentially_on_wall':bool(np.all(np.abs(bp[:2])<h[:2]))})
 e['outer_face_within_5_frames']=next((o for o in outcomes if o['outside_normal_distance_m']>1e-4 and o['tangentially_on_wall']),None)
 e['outer_followup']=outcomes
shell=[e for e in events if e['initial_cavity'] and e['face'].startswith('bucket_shell_')]
summary={'tracked':n,'initial_cavity':int(cavity.sum()),'initial_cavity_also_inside_proxy_mid_surface':int((inside&cavity).sum()),'initial_cavity_outside_proxy_mid_surface':int((~inside&cavity).sum()),
 'first_exit_all_tracked_initially_inside':dict(Counter(e['face'] for e in events)),
 'first_exit_initial_cavity':dict(Counter(e['face'] for e in events if e['initial_cavity'])),
 'first_exit_phase_initial_cavity':dict(Counter(e['phase'] for e in events if e['initial_cavity'])),
 'shell_first_exit_phase_initial_cavity':dict(Counter(e['phase'] for e in shell)),
 'shell_first_exits_reaching_outer_face_within_5_frames':sum(e['outer_face_within_5_frames'] is not None for e in shell),
 'full_wall_single_chord_initial_cavity_count':sum(e['initial_cavity'] for e in full_cross),
 'full_wall_single_chord_initial_cavity_by_phase':dict(Counter(e['phase'] for e in full_cross if e['initial_cavity'])),
 'earliest_initial_cavity_shell_exits':sorted(shell,key=lambda e:e['frame'])[:12],
 'local_chord_not_exact_rotational_ccd':True,'final_cavity_proportion_not_delivery_measure':True}
(work/'path-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(work/'first-exits.json').write_text(json.dumps(events,indent=2)+'\n')
(work/'full-wall-crossings.json').write_text(json.dumps(full_cross,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='earliest_initial_cavity_shell_exits'},indent=2))
