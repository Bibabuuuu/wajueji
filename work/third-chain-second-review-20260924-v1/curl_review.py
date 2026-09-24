"""Read-only phase-level tracking summary; no simulated replay or causal labels."""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
review=json.loads((HERE/'report.json').read_text())
run=Path(review['run'])
rows=[json.loads(v) for v in (run/'physics.ndjson').read_text().splitlines()]
entry=json.loads((HERE/'entry-review.json').read_text())
marks={v['label']:v for v in entry['observations']}
start=marks['curl_load']['frame'];end=marks['lift_clear']['frame']
window=[r for r in rows if start<=r['frame']<end]
active=[r for r in window if r['applied_command_si'][3]>0]
errors=[(r['after_state']['controller_target'][3]-r['axis_positions'][3])*1000 for r in active]
result=dict(run=str(run),curl_window_frames=[start,end-1],active_curl_frames=len(active),
  pitch_change_degrees=(marks['lift_clear']['pitch_rad']-marks['curl_load']['pitch_rad'])*180/np.pi,
  bucket_axis_net_travel_mm=(window[-1]['axis_positions'][3]-window[0]['axis_positions'][3])*1000,
  active_bucket_tracking_error_mm_quantiles=np.quantile(errors,[0,.5,.9,1]).tolist() if errors else [],
  frames_with_projected_command=sum(r['command_projection_reason'] is not None for r in window),
  planned_center_lateral_error_at_curl_m=marks['curl_load']['lateral_from_planned_center_m'],
  tool_reference_z_at_curl_m=marks['curl_load']['center_reference_world_m'][2],
  measured_fill_change_after_off_soil_confirmation_percentage_points=review['final_mixed_fill']['fill_percent']-review['live_off_soil_fill']['fill_percent'],
  limitations=['Phase key timing is an operator annotation, not independently validated contact onset.',
  'No full particle snapshots during cutting or lifting: cannot distinguish initial capture from earlier spillage.',
  'Net curl proves overall motion, not absence of transient stall; no causal claim about low fill from this single run.',
  'Full-width geometric stock score is not a validated retained-payload prediction.'])
(HERE/'curl-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
