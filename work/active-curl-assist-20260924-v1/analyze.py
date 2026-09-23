"""Read-only preregistered A/B control metrics, with absolute gates separate."""
import json
from pathlib import Path
import numpy as np
w=Path(__file__).resolve().parent

def read(case,mode):
 p=w/(case+'-'+mode)
 rows=[json.loads(x) for x in (p/'trace.ndjson').read_text().splitlines()]
 report=json.loads((p/'report.json').read_text())
 q=np.array([r['after_state']['mechanism_qpos'][15] for r in rows]);target=np.array([r['after_state']['controller_target'][3] for r in rows]);v=np.array([r['after_state']['mechanism_qvel'][14] for r in rows]);extra=np.array([r['supplemental_effort_N'] for r in rows]);command=np.array([r['command'][3] for r in rows]);error=target-q
 result={'status':report['status'],'frames':len(rows),'all_hard':all(all(r['hard'].values()) for r in rows),'source_bindings_unchanged':report['bindings_unchanged'],'prefix_original_exact':report['prefix_original_exact'],'max_q_m':float(q.max()),'max_target_m':float(target.max()),'max_extra_N':float(extra.max()),'extra_cap_frames':int((extra>=59999.999).sum()),'max_positive_tracking_error_mm':float(max(0,error.max())*1000),'peak_particle_speed_m_s':max(r['particle_max_speed_m_s'] for r in rows),'windows':{}}
 def window(name,start,end):
  if len(rows)<end:return
  x=slice(start-1,end);before=rows[start-1]['before_state']['mechanism_qpos'][15]
  result['windows'][name]={'frames':[start,end],'positive_error_mean_mm':float(np.maximum(error[x],0).mean()*1000),'signed_error_mean_mm':float(error[x].mean()*1000),'q_start_end_mm':[before*1000,q[end-1]*1000],'net_speed_mm_s':float((q[end-1]-before)*1000/((end-start+1)/60)),'max_extra_kN':float(extra[x].max()/1000),'last15_speed_peak_mm_s':float(abs(v[end-15:end]).max()*1000),'last30_q_range_mm':float(np.ptp(q[end-30:end])*1000),'last15_absolute_stopped':all(r['stopped'] for r in rows[end-15:end])}
 if case=='recorded':
  for a,b in [(601,700),(701,800),(801,900),(901,1000),(1001,1100)]:window(str(a)+'-'+str(b),a,b)
  window('loaded_tracking',701,900);window('hold',1101,1160);window('reverse',1161,1190);window('stop',1191,1250)
 else:
  window('loaded_tracking',61,120);window('hold',121,180);window('reverse',181,210);window('stop',211,270)
 zero_streak=0;release_ok=True
 for c,f in zip(command,extra):
  zero_streak=zero_streak+1 if c<=0 else 0
  if zero_streak>=6 and abs(f)>1e-6:release_ok=False
 result['release_within_0p1s']=release_ok
 return result,rows

result={}
for case in ('recorded','loaded'):
 if not all((w/(case+'-'+m)/'report.json').exists() for m in ('baseline','candidate')):continue
 b,br=read(case,'baseline');c,cr=read(case,'candidate')
 gates={'both_finished':b['status']==c['status']=='recorded','all_hard':b['all_hard'] and c['all_hard'],'sources_unchanged':b['source_bindings_unchanged'] and c['source_bindings_unchanged'],'candidate_geometry':c['max_q_m']<=.857 and c['max_target_m']<=.852+1e-12,'release':c['release_within_0p1s']}
 if case=='recorded':
  prefix={}
  with np.load(w/'recorded-baseline/diagnostic-000600.npz') as a,np.load(w/'recorded-candidate/diagnostic-000600.npz') as d:
   prefix={k:bool(np.array_equal(a[k],d[k])) for k in a.files};gates['prefix_arrays_exact']=all(prefix.values())
  gates['prefix_original_exact']=b['prefix_original_exact'] and c['prefix_original_exact']
  result['prefix_arrays']=prefix
  original=[json.loads(x) for x in (w.parents[1]/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260923-235749-1040071/physics.ndjson').read_text().splitlines()]
  gates['baseline_original_1100_exact']=len(br)>=1100 and all(br[i]['after_state']==original[i]['after_state'] and br[i]['command']==original[i]['applied_command_si'] for i in range(1100))
 for name in ['loaded_tracking','hold','reverse','stop']:
  if name not in b['windows'] or name not in c['windows']:continue
  bw,cw=b['windows'][name],c['windows'][name]
  if name=='loaded_tracking':gates['tracking_improved_40pct']=cw['positive_error_mean_mm']<=.6*bw['positive_error_mean_mm']
  if name in ('hold','stop'):
   gates[name+'_speed_nonregression']=cw['last15_speed_peak_mm_s']<=max(1,1.1*bw['last15_speed_peak_mm_s'])
   gates[name+'_range_nonregression']=cw['last30_q_range_mm']<=bw['last30_q_range_mm']+.2
  if name=='reverse':
   bn=bw['q_start_end_mm'][0]-bw['q_start_end_mm'][1];cn=cw['q_start_end_mm'][0]-cw['q_start_end_mm'][1];gates['reverse_positive_nonregression']=cn>0 and cn>=bn-.2
 result[case]={'baseline':b,'candidate':c,'gates':gates,'bounded_candidate_pass':all(gates.values()),'absolute_stop_pass':all(c['windows'].get(n,{}).get('last15_absolute_stopped',False) for n in ['hold','stop'])}
(w/'control-comparison.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
print(json.dumps(result,indent=2,default=lambda x:x.item()))
