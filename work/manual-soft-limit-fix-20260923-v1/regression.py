"""Read-only real closed-chain previews of the 1329 recorded input poses."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import run_layered_47_sparse_manual_contact_episode as current

out = Path(__file__).resolve().parent
root = out.parent.parent
run = root/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-182307-822279'
trace = [json.loads(l) for l in (run/'physics.ndjson').read_text().splitlines()]
spec = importlib.util.spec_from_file_location('manual_before',out/'manual-before.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
kin = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
assembly, chain = current._load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
_, old_chain = current._load_kinematics(kin,kin/'scene/input/model/A.xml',kin/'scene/input/mesh')
projector = current.ContactCommandProjector(chain)
baseline = old.ContactCommandProjector(old_chain)
physical_limits = chain.limits
limits_before = {k:list(v) for k,v in physical_limits.items()}
rows=[]
for row in trace:
    q = np.asarray(row['before_state']['mechanism_qpos'])
    requested = row['requested_command_si']
    old_applied = baseline.project(q, requested)
    assert np.array_equal(old_applied,row['applied_command_si']), row['frame']
    before=q.copy()
    new_applied = projector.project(q,requested)
    assert np.array_equal(q,before)
    assert chain.limits is physical_limits
    assert {k:list(v) for k,v in chain.limits.items()}==limits_before
    closure = float(np.max(np.abs(chain.residual(chain.q)))) if np.any(new_applied) else None
    if closure is not None:
        assert closure < 1e-8
        assert projector.physical_bounds_ok(chain.q)
    rows.append({'frame':row['frame'],'old':old_applied.tolist(),'new':new_applied.tolist(),'requested':requested,'old_reason':baseline.reason,'new_reason':projector.reason,'closure_m':closure})
    if row['frame']%300==0:print('PREVIEW',row['frame'],flush=True)

cases=[]
last=np.asarray(trace[-1]['after_state']['mechanism_qpos'])
for name,command in [('lift',[0,.0225,0,0]),('tiny_open',[0,0,0,-.0001]),('open',[0,0,0,-.02]),('curl_and_lift',[0,.0225,0,.0675]),('neutral',[0,0,0,0])]:
    applied=projector.project(last,command)
    cases.append({'name':name,'requested':command,'applied':applied.tolist(),'reason':projector.reason})
    assert applied[1]==command[1]
    assert applied[3]==(0 if command[3]>0 else command[3])
for position in (.860001,.369999):
    q=last.copy();q[15]=position
    applied=projector.project(q,[0,.0225,0,-.001])
    assert not np.any(applied) and projector.reason=='joint_working_range'
    cases.append({'name':'invalid_hard_position','bucket_m':position,'applied':applied.tolist(),'reason':projector.reason})
late=[r for r in rows if r['frame']>=1194 and r['requested'][1]!=0]
checks={'baseline_all_1329_reproduced':len(rows)==1329,'late_130_lift_requests_preserved':len(late)==130 and all(r['new'][1]==r['requested'][1] for r in late),'all_predicted_poses_hard_bounded_and_closed':True,'actual_pose_never_changed':True,'physical_limits_restored':True,'hard_bound_rejection':True}
report={'checks':checks,'cases':cases,'changed_commands':sum(r['old']!=r['new'] for r in rows),'maximum_accepted_preview_closure_m':max(r['closure_m'] or 0 for r in rows),'physics_advanced':False,'loaded_lift_verified':False,'soil_loss_fixed':False,'bindings':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(current.__file__),out/'manual-before.py',Path(__file__),run/'physics.ndjson',kin/'closed_chain.py',kin/'scene/input/model/A.xml']}}
(out/'preview-rows.ndjson').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
assert all(checks.values())
