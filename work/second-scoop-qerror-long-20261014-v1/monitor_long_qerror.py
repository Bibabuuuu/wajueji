from pathlib import Path
import json
R=Path('/home/qunwang/wajueji');D=R/'runs/second-scoop-qerror-long-20261014-v1'
p=json.loads((D/'report.json').read_text());a=json.loads((D/'policy/episode-0001/actions.ndjson').read_text().splitlines()[-1]);al=a['alignment']
print(json.dumps(dict(status=p['status'],saved_frame=p['frames'],last_action_frame=a['frame'],cursor=a['cursor_before'],actual_progress=al['actual_progress'],matched_frame=al['matched_frame'],actual_q_error=al.get('actual_joint_error_si'),request=a['request_si'],policy_strength=[(v+1)/2 for v in a['effective_action'][:4]],position_error_m=al['position_error_m'],failure=p.get('failure')),indent=2))
