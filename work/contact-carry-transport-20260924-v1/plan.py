"""Sample the existing transport stages while retaining the full bucket on entry."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from runtime import ROOT, W, SOURCE, TransportRuntime
from layered_47_cycle_geometry import TransportClearance, stationary_terrain
from planning.cycle_geometry import MOUTH,solve_pose,tool_pose
from planning.approach import sampled_approach
from contact_bc_ground_guard import BucketGroundGuard
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig

def project_passive(chain,q):
    result=q.copy();indices=chain.passive
    def residual(x):
        trial=q.copy();trial[indices]=x;return chain.residual(trial)
    solved=least_squares(residual,q[indices],xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=50)
    result[indices]=solved.x
    if np.max(abs(chain.residual(result)))>1e-8 or np.max(abs(result-q))>.05:
        raise ValueError('transport_passive_projection_failed')
    return result

def build_plan(r,out):
    out=Path(out);q=r.arrays['mechanism_qpos'].copy()
    mouth,pitch=tool_pose(r.assembly,q,MOUTH);radial=mouth[:2]/np.linalg.norm(mouth[:2])
    cylinder=q[15]
    terrain,observation=stationary_terrain(r)
    checker=TransportClearance(r,terrain);checker.observation=observation
    ground=BucketGroundGuard(r.assembly,q)
    fb=ContactGeometryFeedback(r.assembly,r.chain,ground,dict(entry_xy_m=mouth[:2],surface_z_m=mouth[2]),q,FeedbackConfig(late_curl_no_down=True))
    start=project_passive(r.chain,q)
    lift=solve_pose(r.assembly,r.chain,start,np.r_[3.9*radial,3.3],pitch,MOUTH)
    # Lock the actual cylinder position through both loaded transport stages.
    lift[15]=cylinder;lift=project_passive(r.chain,lift)
    contract=json.loads((ROOT/'evidence/autonomous-multiscoop-2026-09-14-v1/receiver-contract.json').read_text())
    floor=np.array(contract['triangles']['floor']).reshape(-1,3);ycenter=(floor[:,1].min()+floor[:,1].max())/2
    tf=np.array(contract['local_to_world']);center=tf@np.array([0,ycenter,0,1]);dump=tf@np.array([.6,ycenter,0,1])
    align=lift.copy();align[7]=np.arctan2(center[1],center[0])
    goals=[('CLEAR_PILE',lift),('ALIGN',align)]
    seed=align
    for name,target,angle in [('DUMP',np.r_[dump[:2],3.6],-.45),('CLEAR_TRUCK',np.r_[center[:2],3.5],1.),('RETURN_HIGH',np.r_[4.3*radial,3.3],1.)]:
        seed=solve_pose(r.assembly,r.chain,seed,target,angle,MOUTH);goals.append((name,seed.copy()))
    segments=[]
    for name,goal in goals:
        print('PLAN',name,flush=True)
        def check(q,previous=None):
            if name=='CLEAR_PILE':
                result,_=checker.measure_clearance(q)
                if (result['bucket_truck_m']<.04 or result['mechanism_truck_m']<.02
                    or result['mechanism_ground_m']<.02 or result['closure_m']>.0001):raise ValueError('carry_clearance:'+str(result))
                if previous is not None and np.min(fb.body_heights(q)-fb.body_heights(previous)) < -1e-6:
                    raise ValueError('carry_path_downward')
            else:result=checker.check(q,previous)
            if name in ('CLEAR_PILE','ALIGN'):
                if abs(q[15]-cylinder)>1e-8 or tool_pose(r.assembly,q,MOUTH)[1]<pitch-.01:
                    raise ValueError('carry_path_opens_bucket')
            return result
        def advance(q,v,dt):
            r.chain.q=q.copy();return r.chain.advance(v,dt)[0]
        path=sampled_approach(start,goal,axes=r.chain.axes,speed_limits=np.array([.1,.018,.018,.018]),
            advance=advance,check_pose=check,dt=1/60,max_steps=2400,validation_substeps=2)
        if name=='CLEAR_PILE':checker.check(goal)
        diagnostics=path.pop('diagnostics')
        segments.append(dict(name=name,goal=goal,path=path,minimum_clearance={key:min(d[key] for d in diagnostics) for key in diagnostics[0]}))
        start=goal.copy()
    out.mkdir(parents=True,exist_ok=False)
    np.savez(out/'plan.npz',**{s['name']+'_qpos':s['path']['qpos'] for s in segments},**{s['name']+'_velocity':s['path']['velocity'] for s in segments})
    report=dict(status='kinematic_transport_plan_passed',physics_steps_added=0,source=str(SOURCE),
        observation=observation,carry_pitch=pitch,carry_cylinder_m=float(cylinder),
        initial_clearance=checker.measure_clearance(r.arrays['mechanism_qpos'])[0],
        stages=[dict(name=s['name'],frames=s['path']['steps'],minimum_clearance=s['minimum_clearance']) for s in segments],
        scope='CLEAR_PILE monotonic retreat under original contact guards; full transport clearance required before ALIGN. No physical state reset.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return segments,checker,report

if __name__=='__main__':
    r=TransportRuntime(SOURCE,ROOT/'runs/contact-carry-transport-20260924-v1/plan-runtime-v2')
    build_plan(r,ROOT/'runs/contact-carry-transport-20260924-v1/plan-v2')
