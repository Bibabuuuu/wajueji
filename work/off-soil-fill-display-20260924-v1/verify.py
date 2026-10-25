"""Verify async display against two frozen offline reviews without physics."""
import contextlib,hashlib,io,json,time
from pathlib import Path
import numpy as np
from live_bucket_fill import LiveFillDisplay,load_config,display_lines
root=Path('/home/qunwang/wajueji');out=Path(__file__).resolve().parent
checks={};results=[]
for name,expected in [('active-curl-manual-review-20260924-v1',72.722371967655),('active-curl-left-review-20260924-v1',85.19105755509752)]:
 with contextlib.redirect_stdout(io.StringIO()):
  source=root/'work'/name/'measure.py'
  m={'__file__':str(source)}
  exec(compile(source.read_text().split('report = json.loads((RUN /')[0],str(source),'exec'),m)
  m['final']=np.load(m['FINAL'])
  m['pose']=m['assembly'].assemble(m['final']['mechanism_qpos'],include_meshes=False)[1]['bucket']
  m['rows']=(m['RUN']/'physics.ndjson').read_text().splitlines()
 config=load_config(root/'configs/v4/live-bucket-fill-local-v1.json')
 live=LiveFillDisplay(config,m['records'],m['vertices'],out/(name+'.ndjson'))
 soil=m['final']['particle_q'];pose=m['pose'];rocks=m['final']['rigid_qpos'];before=[x.copy() for x in [soil,pose,rocks]]
 try:
  started=time.perf_counter();assert live.submit(len(m['rows']),soil,pose,rocks)
  submit=time.perf_counter()-started
  assert not live.submit(len(m['rows'])+1,soil,pose,rocks)
  deadline=time.monotonic()+45
  while live.latest is None and time.monotonic()<deadline:
   live.poll();time.sleep(.02)
  result=live.latest;assert result is not None and not result.get('error'),result
  checks[name+'_exact_fill']=result['fill_percent']==expected
  checks[name+'_source_arrays_unchanged']=all(np.array_equal(a,b) for a,b in zip(before,[soil,pose,rocks]))
  results.append({'case':name,'result':result,'submit_wall_s':submit,'startup_and_measure_wall_s':time.perf_counter()-started,'display_lines':display_lines(result,len(m['rows']))})
 finally:live.close()
 checks[name+'_worker_stopped']=live.process.poll() is not None
checks['invalid_estimate_hides_number']='--' in display_lines({'frame':0,'valid_estimate':False,'fill_percent':999,'liters':999},0)[0]
checks['worker_error_hides_number']='unavailable' in display_lines({'error':'test'},0)[0]
report={'checks':checks,'passed':all(checks.values()),'results':results,'physics_steps':0,'source_bindings':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'src/v4/live_bucket_fill.py',Path(__file__),root/'configs/v4/live-bucket-fill-local-v1.json']}}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));assert report['passed']
