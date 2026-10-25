"""CPU search within existing policy corridor and original approach gates."""
exec(open('/home/qunwang/wajueji/work/edge-first-strips-20261015-v1/check_safe_handoff.py').read().split('data=dict')[0].replace("W/'handoff-read-runtime'", "W/'handoff-search-runtime'"))
import time
start=__import__('time').monotonic()
axes=np.array([9,12,15]);ref=project(r,reference[0]);found=[];trials=[]
# Try small boom adjustments first, then coordinated millimetre offsets.
offsets=[np.array([x,0.,0.]) for x in [-.001,.001,-.002,.002,-.004,.004]]
offsets += [np.array([a,b,d]) for a in [-.004,0.,.004] for b in [-.004,0.,.004] for d in [-.008,0.,.008]]
for off in offsets:
 candidate=ref.copy();candidate[axes]+=off;candidate=project(r,candidate)
 pose=r.assembly.assemble(candidate,include_meshes=False)[1]['bucket']
 state=ProgressAlignment(reference,poses).observe(candidate,pose,0.)
 g=gap(candidate);trials.append(dict(offset=off,gap=g,trusted=state['trusted']))
 if g>=.322 and state['trusted']:
  found.append((float(np.linalg.norm(off/[.01,.01,.02])),candidate,state,g,off))
found.sort(key=lambda x:x[0]);accepted=None;rejected=[]
from continuous_scoop import segment
class Checker:
 def check(self,x,previous=None):
  result=c.check(x,previous);g=gap(x)
  if g<.32:raise ValueError('original_32cm_gate:'+str(g))
  from planning.rear_lower_prepoint import mechanism_and_truck_clearance
  _,b=r.assembly.assemble(x,include_meshes=False)
  other=mechanism_and_truck_clearance(c.geometries,b,t,c.truck_triangles.reshape(-1,3))
  if other['mechanism_terrain_clearance_m']<.02:raise ValueError('original_mechanism_terrain')
  return dict(result,bucket_observed_terrain_m=g,mechanism_terrain_clearance_m=other['mechanism_terrain_clearance_m'])
for _,candidate,state,g,off in found:
 try:
  path=segment(r,q,candidate,Checker());accepted=dict(q=candidate,alignment=state,gap=g,offset=off,frames=path['steps'],minimum_clearance=path['minimum_clearance'])
  np.savez(W/'safe-handoff-candidate.npz',qpos=path['qpos'],velocity=path['velocity'],reference_q=reference,yaw_offset=angle)
  break
 except ValueError as e:rejected.append(dict(offset=off,failure=str(e)))
write(W/'safe-handoff-search.json',dict(status='cpu_handoff_passed' if accepted else 'cpu_handoff_failed',accepted=accepted,trials=trials,rejected=rejected,physical_steps=0,wall_seconds=__import__('time').monotonic()-start))
print('ACCEPTED',accepted,flush=True)
