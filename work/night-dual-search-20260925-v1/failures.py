"""Reject known action outcomes; unknown errors and broken evidence stop the batch."""
import ast,json
from pathlib import Path

class TrialRejected(RuntimeError):
 def __init__(self,detail):
  self.detail=detail
  super().__init__(detail['reason'])

ACTION_FAILURES={
 'feedback_velocity_infeasible','feedback_projected_command_moves_bucket_down',
 'feedback_ground_preview_infeasible','feedback_lateral_corridor_exceeded',
 'bc_bucket_ground_clearance','D154_behavior_gate','D154_off_soil_behavior_gate',
 'phase_curl_boundary_unresolved','finish_curl_requires_95_percent_actual_closure',
 'loaded_center_cumulative_drawdown_exceeds_10mm','loaded_preview_cumulative_drawdown_exceeds_10mm',
 'carry_before_ground','carry_after_ground','second_handoff_not_ready_within_finish_window',
 'carry_actual_bucket_opened','carry_actual_pitch_lost','carry_no_safe_continuous_velocity',
 'carry_projected_command_loses_retention','carry_projection_exceeds_slew',
 'carry_stopped_curl_preview_loses_retention','carry_preview_loses_retention',
 'carry_clearance','carry_path_downward','carry_path_opens_bucket',
 'transport_ground_guard','transport_retreat_clearance','transport_actual_lost_retention',
 'transport_no_continuous_loaded_command','transport_preview_ground',
 'transport_preview_loses_retention','transport_preview_downward',
 'second_transport_tracking','second_stage_endpoint','second_development_unload_inventory',
 'second_contact_not_completed','second_tail_budget_exhausted',
}
ACTION_HARD_CHECKS={'forbidden_contact_zero','rock_ramp_truck_contact_zero','joint_bounds'}

def message(value):
 """Decode only a single repr(Exception('message')); never eval exception text."""
 try:
  n=ast.parse(value,mode='eval').body
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('ValueError','RuntimeError','TimeoutError') and len(n.args)==1:
   value=ast.literal_eval(n.args[0])
 except (SyntaxError,ValueError,TypeError):pass
 return value if isinstance(value,str) else ''

def classify(out):
 out=Path(out);reports=[]
 for phase in ('contact','tail'):
  p=out/phase/'report.json'
  if p.exists():reports.append((phase,json.loads(p.read_text())))
 if not reports:return None
 for phase,r in reports:
  exact=r.get('zero_step_arrays_exact',r.get('zero_arrays_exact',{}))
  if not exact or not all(exact.values()) or not r.get('source_unchanged'):return None
  if any(k in r for k in ('save_error','display_error','display_capture_error','display_finish_error')):return None
  rows=out/phase/'physics/frames.ndjson'
  if rows.exists():
   with rows.open() as stream:
    for line in stream:
     row=json.loads(line);checks=row['hard_guard_checks']
     if not checks or any(not v and k not in ACTION_HARD_CHECKS for k,v in checks.items()):return None
 phase,r=reports[-1];reason=message(r.get('failure',''))
 if r.get('status')=='failed' or r.get('status')=='budget_stopped':
  code=reason.split(':',1)[0]
  if code not in ACTION_FAILURES and code!='cycle_hard_guard':return None
  if code=='cycle_hard_guard':
   checks=ast.literal_eval(reason.split(':',1)[1])
   failed={k for k,v in checks.items() if not v}
   if not failed or not failed<=ACTION_HARD_CHECKS:return None
  checkpoint=next((p for p in (out/phase/'recovery.npz',out/phase/'physics/failure.npz',out/phase/'final.npz') if p.exists()),None)
  if checkpoint is None or not Path(str(checkpoint)+'.json').exists():return None
  from sources import sha
  meta=json.loads(Path(str(checkpoint)+'.json').read_text())
  if meta.get('checkpoint_sha256')!=sha(checkpoint):return None
  return dict(reason=reason,stage=phase,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
    disposition='candidate_rejected_continue_from_original_prepoint',retry=False)
 score=out/'score.json'
 if score.exists():
  s=json.loads(score.read_text())
  if s.get('invalid_reason')=='working_volume_boundary_or_estimator_invalid' and all(s.get('checks',{}).values()):
   return dict(reason=s['invalid_reason'],stage='score',disposition='invalid_candidate_measurement',retry=False)
 return None
