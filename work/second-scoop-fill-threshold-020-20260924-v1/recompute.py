"""Recompute the frozen second scoop without advancing physics."""
from pathlib import Path
from dataclasses import replace
import json
source=Path(__file__).resolve().parents[1]/'second-scoop-fill-diagnosis-20260924-v1/diagnose.py'
exec(compile(source.read_text().split('result=e.measure')[0],str(source),'exec'))
result=e.measure(cloud,pose,e.contract,e.front,volume,replace(e.calibration,occupancy_level=.2))
report=dict(snapshot=identity(FINAL),occupancy_level=.2,calibration_status='user_selected_threshold_not_recalibrated',result=result)
Path(__file__).with_name('report.json').write_text(json.dumps(report,indent=2)+'\n')
live=e.measure_snapshot(dict(frame=len(ph),soil=a['particle_q'],rocks=a['rigid_qpos'],pose=pose))
assert live['occupancy_level']==.2 and live['calibration_status']==report['calibration_status']
assert live['valid_estimate'] and abs(live['fill_percent']-result['fill_percent'])<1e-10
old=e.measure(cloud,pose,e.contract,e.front,volume,replace(e.calibration,occupancy_level=.4))
assert abs(old['fill_percent']-86.27239575075)<1e-8
print(json.dumps(dict(live=live,old_threshold_fill_percent=old['fill_percent']),indent=2))
