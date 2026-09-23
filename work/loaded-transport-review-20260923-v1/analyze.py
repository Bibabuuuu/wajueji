"""Read-only postmortem and zero-step collider index reproduction."""
import json, sys, hashlib
from pathlib import Path
import numpy as np
import newton
import warp as wp
from newton.solvers import SolverImplicitMPM
from layered_excavation_approach_planner import _load_kinematics, quaternion_matrix_wxyz
sys.path.insert(0, '/home/qunwang/codex-work/fill-volume-2026-09-15-v1')
from fill_volume import cavity_mask
root=Path('/home/qunwang/wajueji')
out=Path(__file__).resolve().parent
run=root/'runs/layered-47-loaded-transport-v1/loaded-transport-20260923-215454-962382'
start=root/'runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2/loaded-handoff.npz'
end=run/'final-000887.diagnostic.npz'
a=np.load(start); b=np.load(end)
report=json.loads((run/'report.json').read_text())
rows=[json.loads(l) for l in (run/'physics.ndjson').read_text().splitlines()]
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly,_=_load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
poses=[assembly.assemble(d['mechanism_qpos'],include_meshes=False)[1]['bucket'] for d in [a,b]]
local=[(d['particle_q']-p[:3,3])@p[:3,:3] for d,p in zip([a,b],poses)]
contract=json.loads(Path('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json').read_text())
front=np.load('/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz')['front']
mask=[cavity_mask(q,np.array(contract['bucket']['polygon_xz_m']),np.array(contract['bucket']['y_limits_m']),front) for q in local]
proxy=json.loads((root/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text())
solid=[]
inside_shell=np.zeros(len(local[1]), dtype=bool)
for box in proxy['shell_boxes']:
 q=(local[1]-box['position_bucket_m'])@quaternion_matrix_wxyz(box['quaternion_wxyz'])
 depth=np.min(np.array(box['half_size_m'])-np.abs(q),axis=1)
 inside_shell |= depth > 1e-5
 solid.append({'name':box['name'],'particle_centers_inside':int((depth>1e-5).sum()),'maximum_depth_m':float(max(0,depth.max()))})
# Topology-only reproduction: default Newton discovery merges static shapes
# into one collider and each body's shapes into one collider.
wp.init()
builder=newton.ModelBuilder()
for i in range(48):
 body=builder.add_body(mass=1.0)
 for j in range(18 if i==47 else 1):
  builder.add_shape_box(body,hx=.03,hy=.03,hz=.03)
truck_shapes=[builder.add_shape_box(-1,hx=.03,hy=.03,hz=.03) for i in range(5)]
builder.add_ground_plane()
model=builder.finalize(device='cpu')
sand=newton.ModelBuilder();SolverImplicitMPM.register_custom_attributes(sand)
sand.add_particle(pos=(0,0,2),vel=(0,0,0),mass=1,radius=.01)
solver=SolverImplicitMPM(sand.finalize(device='cpu'), config=SolverImplicitMPM.Config())
solver.setup_collider(model=model)
body_map=solver.collider_body_index.numpy()
# Query the actual merged bucket proxy, with the production side winding fix.
from mixed_material_r1 import outward_side_plate_faces
from newton._src.solvers.implicit_mpm.rasterized_collisions import get_average_face_normal
@wp.kernel
def signed_query(mesh:wp.uint64,points:wp.array[wp.vec3],out:wp.array[float]):
 i=wp.tid(); p=points[i]
 q=wp.mesh_query_point_no_sign(mesh,p,1.0)
 if q.result:
  cp=wp.mesh_eval_position(mesh,q.face,q.u,q.v)
  normal=get_average_face_normal(mesh,cp)
  out[i]=wp.length(p-cp)*wp.where(wp.dot(normal,p-cp)>0.0,1.0,-1.0)
 else:
  out[i]=999.0
actual=newton.ModelBuilder();bid=actual.add_body(mass=1.0)
for box in proxy['shell_boxes']+proxy['tooth_boxes']:
 w,x,y,z=box['quaternion_wxyz'];h=box['half_size_m']
 actual.add_shape_box(bid,xform=wp.transform(wp.vec3(*box['position_bucket_m']),wp.quat(x,y,z,w)),hx=h[0],hy=h[1],hz=h[2])
for side in proxy['side_plate_meshes']:
 faces,_=outward_side_plate_faces(side)
 actual.add_shape_mesh(bid,mesh=newton.Mesh(np.asarray(side['vertices_bucket_m'],dtype=np.float32),faces.reshape(-1),compute_inertia=False,is_solid=True))
solver.setup_collider(model=actual.finalize(device='cpu'))
points=wp.array(local[1][inside_shell],dtype=wp.vec3,device='cpu'); distances=wp.zeros(len(points),dtype=float,device='cpu')
wp.launch(signed_query,dim=len(points),inputs=[solver._mpm_model._collider_meshes[0].id,points,distances],device='cpu')
sdf=distances.numpy()
sign_audit={'distinct_particle_centers_inside_shell_boxes':int(inside_shell.sum()),'merged_proxy_positive_sdf_despite_box_interior':int((sdf>0).sum()),'merged_proxy_negative_sdf':int((sdf<0).sum()),'query_only_no_physics':True,'uses_final_actual_bucket_pose':True}
result={'run':str(run),'exit_reason':report['stop_reason'],'failure':report.get('failure'),'timing':report['timing'],
 'timing_component_seconds':{k:sum(r[k] for r in rows) for k in ['mpm_wall_seconds','mujoco_wall_seconds','assignment_wall_seconds']},
 'physics_compute_slowdown':report['timing']['physics_wall_seconds']/report['timing']['simulation_seconds'],
 'all_frames_hard_safe':all(r['hard_guard'] for r in rows),
 'initial_cavity_soil_particles':int(mask[0].sum()),'final_cavity_soil_particles':int(mask[1].sum()),
 'initial_cavity_particles_still_in_final_cavity':int((mask[0]&mask[1]).sum()),
 'final_shell_box_penetrations':solid,'merged_bucket_sign_audit':sign_audit,
 'maximum_postsolve_position_correction_m':max(r['postsolve_projection']['maximum_position_delta_m'] for r in rows),
 'reported_truck_mpm_contact_frames':sum(r['truck_mpm_contact_samples']>0 for r in rows),
 'rock_truck_contact_frames':sum(r['rigid_contact_substeps']['rock_truck']>0 for r in rows),
 'collider_index_reproduction':{'physics_steps':0,'shape_count':model.shape_count,'collider_count':len(body_map),'truck_shape_ids':truck_shapes,'body_map':body_map.tolist(),'truck_ids_out_of_collider_range':all(i>=len(body_map) for i in truck_shapes)},
 'limits':['Final snapshot cannot establish the exit path of each particle.','Zero truck MPM count is invalid because shape indices were used as collider indices.','Hard guards do not prove absence of tunneling or stable truck receipt.','No replay or new physics was run.'],
 'bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [start,end,run/'report.json',run/'physics.ndjson',Path(__file__)]}}
(out/'report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
