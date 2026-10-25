"""Read-only audit of the stopped third-scoop finish; never advance physics."""
import hashlib,json
from pathlib import Path
ROOT=Path('/home/qunwang/wajueji');OUT=Path(__file__).resolve().parent
RUN=Path((OUT/'run-directory.txt').read_text().strip())
SOURCE=ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-093202-1362929/final-001264.cycle.npz'
EXPECTED='5b9c4e7a5520a0c4beab9cc124e7f4a54e24af4b8953fea6f6bdfb0b1e3077b0'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def identity(p):return dict(path=str(p),sha256=sha(p))
hold=read(RUN/'source-audit/audit.json');retreat=read(RUN/'retreat-boom-physical-v3/report.json');transport=read(RUN/'transport-after-retreat/report.json')
stages={}
for name in ('source-audit','retreat-boom-physical-v3','transport-after-retreat'):
 rows=[json.loads(s) for s in (RUN/name/'frames.ndjson').read_text().splitlines()]
 stages[name]=dict(frames=len(rows),all_hard_guards=all(r['hard_guard'] for r in rows),all_material_guards=all(r['hencky_material_guard']['safe'] for r in rows),all_sweep_guards=all(r['moving_bucket_sweep']['safe'] for r in rows))
rows=[json.loads(s) for s in (RUN/'transport-after-retreat/frames.ndjson').read_text().splitlines()];last=rows[-1]
checks=dict(source_unchanged=sha(SOURCE)==EXPECTED,source_exact_hold=hold['status']=='hold_passed' and hold['zero_step']['exact'],retreat_passed=retreat['status']=='retreat_passed',free_space_gate_preserved=retreat['protocol']['free_space_gate_unchanged'],transport_stopped=transport['status']=='failed',earlier_transport_frames_safe=all(r['hard_guard'] for r in rows[:-1]),sole_last_guard_failure=[k for k,v in last['hard_guard_checks'].items() if not v]==['hencky_material_state_safe'],material_nonfinite_Jp_one=last['hencky_material_guard']['nonfinite_values']['particle_Jp']==1,transport_binds_retreat=transport['source']['sha256']==retreat['checkpoint']['sha256'],retreat_binds_audit=retreat['source']['sha256']==hold['final_checkpoint']['sha256'],saved_stage_hashes_match=all(sha(Path(s['checkpoint']['path']))==s['checkpoint']['sha256'] for s in transport['stages']))
failures={}
for name in ('transport','retreat-plan','retreat-boom-physical','retreat-boom-physical-v2'):
 p=RUN/name/'report.json'
 if p.exists():
  d=read(p);failures[name]=dict(identity=identity(p),status=d['status'],frames=d.get('frames',0),failure=d.get('failure'))
result=dict(schema='third_scoop_finish_stopped_review_v1',investigation_checks_passed=all(checks.values()),unload_return_passed=False,checks=checks,source=identity(SOURCE),stages=stages,retreat=retreat,transport_status=transport['status'],transport_failure_frame=last['frame'],transport_completed_stages=transport['stages'],last_safe_checkpoint=transport['stages'][-1]['checkpoint'],last_guard=last['hard_guard_checks'],last_material_guard=last['hencky_material_guard'],Jp_tail=[dict(frame=r['frame'],maximum_Jp=r['hencky_material_guard']['maximum_Jp'],particle_max_speed_m_s=r['particle_max_speed_m_s']) for r in rows[-6:]],failure_snapshot_error=(RUN/'transport-after-retreat/failure-save-error.txt').read_text(),preserved_failures=failures,evidence_bindings=[identity(RUN/p) for p in ('source-audit/audit.json','retreat-boom-physical-v3/report.json','transport-after-retreat/report.json','transport-after-retreat/frames.ndjson','transport-after-retreat/failure-save-error.txt')],scope='Stopped local development continuation. Contact data remains separately preserved. No G4/G5, successful unload/return, training qualification or failure-state resume claim.')
(OUT/'report.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(dict(checks=checks,stages=stages,last_safe_checkpoint=result['last_safe_checkpoint'],failure_snapshot_error=result['failure_snapshot_error'])))
assert result['investigation_checks_passed']
