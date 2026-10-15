from pathlib import Path
import json,shutil
ROOT=Path('/home/qunwang/wajueji');WORK=ROOT/'work/roadshow-three-way-20261014-v1'
first=WORK/'transport-endpoint-preflight-v2';second=WORK/'transport-endpoint-preflight-v3'
a=json.loads((first/'report.json').read_text());b=json.loads((second/'report.json').read_text())
assert a['status']=='failed' and a['error'].startswith('TimeoutError')
assert a['sources']['better_manual']['phase']=='complete' and all(a['sources']['better_manual']['checks'].values())
assert b['status']=='endpoint_nominal_cpu_passed' and b['sources']['frozen_residual']['phase']=='complete'
assert all(b['sources']['frozen_residual']['checks'].values())
out=WORK/'transport-endpoint-final-cpu-v1';out.mkdir(exist_ok=False)
for name,parent in [('better_manual',first),('frozen_residual',second)]:
    shutil.copytree(parent/(name+'-plan'),out/(name+'-plan'))
report=dict(status='endpoint_nominal_cpu_passed',sources={'better_manual':a['sources']['better_manual'],'frozen_residual':b['sources']['frozen_residual']},
    source_runs={'better_manual':str(first/'report.json'),'frozen_residual':str(second/'report.json')},
    preserved_timeout=dict(path=str(first/'report.json'),seconds=a['elapsed_seconds'],error=a['error']),
    cpu_preflight_elapsed_seconds=a['elapsed_seconds']+b['elapsed_seconds'],new_physical_frames=0,training_updates=0,
    native_restore_tested=False,physical_transport_tested=False,
    scope='Aggregates two fully checked CPU cases; preserves prior timeouts and clearance failures. No native runtime or actual loaded transport.')
(out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':report['status'],'nominal_commands':sum(s['nominal_commands'] for s in report['sources'].values()),'new_physical_frames':0}))
