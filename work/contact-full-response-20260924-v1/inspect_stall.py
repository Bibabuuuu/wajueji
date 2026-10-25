"""Read-only kinematic preview of recorded rejected commands; no rigid/MPM steps."""
import json,hashlib
from pathlib import Path
import numpy as np
from contact_bc_ground_guard import BucketGroundGuard
from contact_geometric_feedback import ContactGeometryFeedback,FeedbackConfig,LIMITS
from contact_cut_reference import GroundSoilCutReference
from layered_excavation_approach_planner import _load_kinematics
from run_layered_47_sparse_manual_contact_episode import ContactCommandProjector
from predict_response import W,ROOT
RUN=ROOT/'runs/contact-full-response-20260924-v1/candidate60'
def main():
    actions=[json.loads(x) for x in (RUN/'actions.ndjson').read_text().splitlines()];chosen=[next(e for e in actions if e['frame']==f) for f in (1124,1200,1376)]
    cfg=json.loads((W/'protocol.json').read_text());source=Path(cfg['inputs'][0]['path']);initial=np.load(source);k=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance');a,c=_load_kinematics(k,k/'scene/input/model/A.xml',k/'scene/input/mesh')
    scene=json.loads((RUN/'display/scene.json').read_text());selected=scene['selected_cut'];face=selected.get('contact',selected);goal=dict(entry_xy_m=face.get('contact_xy_m',face.get('entry_xy_m')),surface_z_m=face.get('contact_surface_z_m',face.get('surface_z_m')))
    guard=BucketGroundGuard(a,initial['mechanism_qpos']);ref=GroundSoilCutReference(initial['particle_q'],json.loads((ROOT/'reports/v4/mixed-material-r1-2026-09-19-v1/proxy-manifest.json').read_text()));feedback=ContactGeometryFeedback(a,c,guard,goal,initial['mechanism_qpos'],FeedbackConfig(**cfg['feedback_parameters']),ref);p=ContactCommandProjector(c);results=[]
    for e in chosen:
        q=np.array(e['before_state']['mechanism_qpos']);start=feedback.body_heights(q);_,_,grad=feedback.linearize(q,include_body=True)
        vectors={'original_request':np.array(e['requested_command_si']),'small_boom_up_only':np.array([0,min(.00075,max(0,e['requested_command_si'][1])),0,0])}
        checks={}
        for name,requested in vectors.items():
            projected=p.project(q,requested);trials=[]
            for factor in (1.,.5,.25,.125,1e-6,1e-9,0.):
                v=projected*factor;c.q=q.copy();remaining=.15;minimum_body_delta=0.;maximum_closure_error=float(np.linalg.norm(c.residual(q)));ground_min=guard.measure(q)['clearance_m'];scale=max(1.,float(np.max(abs(v)/np.array([.12,.0225,.0225,.0225]))));error=None
                try:
                    while remaining>1e-12:
                        dt=min(1/60,remaining,.05/scale);nq,_=p._advance_preview(v/scale,min(.05,dt*scale));minimum_body_delta=min(minimum_body_delta,float(np.min(feedback.body_heights(nq)-start)));ground_min=min(ground_min,guard.measure(nq)['clearance_m']);remaining-=dt
                        maximum_closure_error=max(maximum_closure_error,float(np.linalg.norm(c.residual(nq))))
                except ValueError as ex:error=str(ex)
                trials.append(dict(factor=factor,minimum_body_delta_m=minimum_body_delta,ground_clearance_m=ground_min,maximum_closure_residual_m=maximum_closure_error,original_preview_pass=bool(error is None and minimum_body_delta>=-1e-6 and ground_min>=.03-1e-6),error=error))
            checks[name]=dict(requested=requested.tolist(),projected=projected.tolist(),linear_minimum_body_velocity=float(np.min(grad@(projected/LIMITS))),within_previous_command_rate=bool(np.max(np.abs((projected-np.array(e['previous_applied_action_si']))/LIMITS))<=2/60+1e-12),trials=trials)
        results.append(dict(frame=e['frame'],closure_fraction=e['feedback']['closure_fraction'],recorded_applied=e['applied_command_si'],recorded_preview_scale=e['feedback']['preview_scale'],checks=checks))
    (W/'stall-preview.json').write_text(json.dumps(dict(status='read_only_kinematic_diagnosis',native_steps=0,MPM_steps=0,not_a_new_policy_or_authorized_physical_candidate=True,samples=results,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
