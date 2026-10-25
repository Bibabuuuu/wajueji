"""Read-only pair-averaged reversal and unchanged-state verification."""
from pathlib import Path
import hashlib,json
import numpy as np
out=Path(__file__).resolve().parent
prior=out.parent/'bucket-integral-matched-load-20260923-v1'
protocol=json.loads((out/'protocol.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines()]
def stats(data):
 result={}
 for name,a,b in [('curl',661,1020),('hold',1021,1140),('reverse',1141,1170),('stop',1171,1290)]:
  r=data[a-1:b];q=np.array([x['axis_positions'][3] for x in r]);ref=np.array([x['controller_target'][3] for x in r]);v=np.array([x['axis_velocity'][3] for x in r]);f=np.array([x['equivalent_drive_load'][3] for x in r]);integ=np.array([x['bucket_integral_effort'][3] for x in r]);prev=data[a-2]['axis_positions'][3]
  result[name]={'end_q_m':float(q[-1]),'end_ref_m':float(ref[-1]),'last30_mean_abs_error_m':float(np.abs(q[-30:]-ref[-30:]).mean()),'net_displacement_m':float(q[-1]-prev),'tail15_all_stopped':all(x['stopped'] for x in r[-15:]),'tail15_peak_abs_velocity_m_s':float(np.abs(v[-15:]).max()),'tail15_position_range_m':float(np.ptp(q[-15:])),'integral_end_N':float(integ[-1]),'integral_delta_N':float(integ[-1]-data[a-2]['bucket_integral_effort'][3]),'phase_native_cumulative_effort_peak_N':r[-1]['native_total_drive_peak'][3]}
  if len(r)>=60:
   vt=v[-60:];ft=f[-60:];alt=(-1.)**np.arange(60)
   result[name]['sampled_oscillation']={'velocity_sign_changes_last60':int(np.count_nonzero(np.signbit(vt[1:])!=np.signbit(vt[:-1]))),'velocity_load_correlation':float(np.corrcoef(vt,ft)[0,1]),'velocity_previous_load_correlation':float(np.corrcoef(vt[1:],ft[:-1])[0,1]),'alternating_velocity_amplitude_m_s':float(abs(np.mean((vt-vt.mean())*alt))),'alternating_load_amplitude_N':float(abs(np.mean((ft-ft.mean())*alt))),'position_mean_change_last60_m':float(q[-2:].mean()-q[-60:-58].mean())}
 q0=np.mean([data[i-1]['axis_positions'][3] for i in (1139,1140)])
 paired=np.array([np.mean([data[i-1]['axis_positions'][3],data[i]['axis_positions'][3]]) for i in range(1141,1171,2)])
 disp=q0-paired
 hit=next((i for i in range(len(disp)) if np.all(disp[i:]>=.0002)),None)
 result['reverse_pair_means']={'pre_q_m':float(q0),'retreat_m':disp.tolist(),'net_retreat_m':float(disp[-1]),'sustained_0p2mm_latency_s':None if hit is None else float((2*hit+2)/60),'sample_time_convention':'latency at completion of each two-frame pair, measured from reverse start'}
 return result
new=rows(out/'candidate/trace.ndjson');old=rows(prior/'candidate/trace.ndjson');base=rows(prior/'baseline/trace.ndjson')
assert len(new)==len(old)==len(base)==1290
exact={}
for frame in [600,660,1020,1140]:
 a=dict(np.load(out/'candidate'/f'diagnostic-{frame:06}.npz'));b=dict(np.load(prior/'candidate'/f'diagnostic-{frame:06}.npz'))
 keys=[k for k,v in b.items() if v.dtype.kind in 'fiu'];assert len(keys)==23
 exact[str(frame)]={k:bool(k in a and np.array_equal(a[k],b[k])) for k in keys}
prefix={k:all(a[k]==b[k] for a,b in zip(new[:1140],old[:1140],strict=True)) for k in protocol['matched_trace_fields']}
newstats=stats(new);oldstats=stats(old);basestats=stats(base)
latency=newstats['reverse_pair_means']['sustained_0p2mm_latency_s']
gates={'prefix1140_exact':all(prefix.values()) and all(all(x.values()) for x in exact.values()),'all_original_hard_guards':all(all(r['hard_guards'].values()) for r in new),'curl_error_le5mm':newstats['curl']['last30_mean_abs_error_m']<=.005,'reverse_pair_retreat_ge2mm':newstats['reverse_pair_means']['net_retreat_m']>=.002,'reverse_pair_retreat_ge2x_prior':newstats['reverse_pair_means']['net_retreat_m']>=2*oldstats['reverse_pair_means']['net_retreat_m'],'reverse_sustained_latency_le0p2s':latency is not None and latency<=.2,'first_hold_original_stopped':newstats['hold']['tail15_all_stopped'],'post_reverse_original_stopped':newstats['stop']['tail15_all_stopped'],'post_reverse_error_le5mm':newstats['stop']['last30_mean_abs_error_m']<=.005,'actual_bucket_upper':max(x['axis_positions'][3] for x in new[660:])<=.852}
result={'modes':{'release':newstats,'previous_integral':oldstats,'no_integral':basestats},'exact_state_checks':exact,'prefix_trace_exact':prefix,'gates':gates,'all_gates_pass':all(gates.values()),'training_eligible':False,'default_enabled':False,'caveats':['Same forward and neutral prefix does not imply stopped; hold failure remains.','Pair averaging removes the strict alternating component but is not a full identification of oscillation origin.','Correlations are observational only; contact state changes after reversal.','2mm reverse threshold is a local improvement probe, not full10mm tracking.'],'bindings':{str(p):sha(p) for p in [Path(__file__),out/'protocol.json',out/'bucket_integral.h',out/'candidate/report.json',out/'candidate/trace.ndjson',prior/'candidate/trace.ndjson',prior/'baseline/trace.ndjson']}}
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'gates':gates,'phases':newstats,'old_reverse_pairs':oldstats['reverse_pair_means']},indent=2))
