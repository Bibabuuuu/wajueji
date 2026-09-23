import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from layered_excavation_approach_planner import quaternion_matrix_wxyz
w=Path(__file__).resolve().parent;r=w/'replay-v2';d=json.loads((w/'confirmed-project-outward.json').read_text());e=d['example'];ix=e['tracked_index'];f=e['frame']
proxy=json.loads(Path('/home/qunwang/wajueji/reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
post=np.load(r/'post-projection-world.npy',mmap_mode='r');pre=np.load(r/'pre-projection-world.npy',mmap_mode='r');poses=np.load(r/'actual-bucket-poses.npy',mmap_mode='r')
box=next(x for x in proxy['shell_boxes'] if x['name']==e['face']);rot=quaternion_matrix_wxyz(box['quaternion_wxyz']);center=np.asarray(box['position_bucket_m'])
fig,axs=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
ax=axs[0]
for b in proxy['shell_boxes']:
 h=np.asarray(b['half_size_m']);br=quaternion_matrix_wxyz(b['quaternion_wxyz']);c=np.asarray(b['position_bucket_m'])
 pts=np.array([[-h[0],0,-h[2]],[h[0],0,-h[2]],[h[0],0,h[2]],[-h[0],0,h[2]]])@br.T+c
 ax.add_patch(Polygon(pts[:,[0,2]],facecolor='#9da4af',edgecolor='#626b78',linewidth=.7))
mouth=np.asarray(proxy['mouth_endpoints_xz_m']);ax.plot(mouth[:,0],mouth[:,1],'--',color='#168b8b',label='Open mouth')
frames=np.arange(max(0,f-70),min(888,f+30));q=np.array([(post[t,ix]-poses[t,:3,3])@poses[t,:3,:3] for t in frames]);ax.plot(q[:,0],q[:,2],color='#dd742a',lw=2,label='Recorded particle path')
a=np.asarray(e['pre_local']);b=np.asarray(e['post_local_same_collider_pose']);ax.scatter([a[0],b[0]],[a[2],b[2]],c=['#bf4c21','#bd2042'],s=22,zorder=5)
ax.set_aspect('equal');ax.set_xlim(-.05,.9);ax.set_ylim(-1.12,.08);ax.set_xlabel('Bucket-local x (m)');ax.set_ylabel('Bucket-local z (m)');ax.set_title(f'Particle {e["particle_id"]}: through curved shell');ax.legend(fontsize=8,loc='lower left');ax.grid(alpha=.15)
ax=axs[1];f0=f-12;f1=f+5;fs=np.arange(f0,f1+1);before=[];after=[]
for t in fs:
 p=poses[t-1];a=(pre[t-1,ix]-p[:3,3])@p[:3,:3];b=(post[t,ix]-p[:3,3])@p[:3,:3]
 before.append(((a-center)@rot)[2]*1000);after.append(((b-center)@rot)[2]*1000)
ax.axhspan(-18,18,color='#bcc2ca',alpha=.5,label='36 mm wall');ax.axhline(0,color='#8a9098',ls=':',lw=1)
ax.axhline(-18,color='#626b78',lw=1);ax.axhline(18,color='#626b78',lw=1)
ax.plot(fs,before,'o-',c='#d46b26',ms=3,label='Before projection');ax.plot(fs,after,'o-',c='#2674a5',ms=3,label='After projection (same pose)')
ax.annotate('Frame 551: +2.10 → +18.69 mm',xy=(551,18.69),xytext=(540,37),fontsize=9,arrowprops={'arrowstyle':'->','color':'#333'})
ax.text(f0+.2,22,'Outside bucket',fontsize=9);ax.text(f0+.2,-28,'Inside bucket',fontsize=9)
ax.set_ylim(-34,48);ax.set_xlabel('Recorded frame');ax.set_ylabel('Outward distance from wall middle (mm)');ax.set_title('Endpoint projection pushes to the outer face');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='lower right')
fig.savefig(w/'confirmed-wall-leak.png',dpi=170)
