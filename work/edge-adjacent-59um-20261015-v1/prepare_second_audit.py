from pathlib import Path
R=Path('/home/qunwang/wajueji');W=R/'work/edge-adjacent-59um-20261015-v1';OLD=R/'work/edge-first-two-scoop-20261015-v1'
s=(OLD/'audit_prepoint_continuation.py').read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:100],s.count(old))
 s=s.replace(old,new)
replace("W=R/'work/edge-first-two-scoop-20261015-v1'", "W=R/'work/edge-adjacent-59um-20261015-v1'")
replace('import run_prepoint_continuation as m','import run_second as m')
replace("'new2h_budget_identity'", "'new60min_budget_identity'")
replace("b['seconds']==7200 and b['reserve_seconds']==900", "b['seconds']==3600 and b['reserve_seconds']==600")
replace('n+9450<=30000', 'n<=13000')
replace("phases[0]['start_frame']==3", "phases[0]['start_frame']==0")
replace("old=W/'prepoint-qualification/replica-a'", "old=W/'expected-prefix'")
replace("np.array_equal(executed[i-1]['command'],np.zeros(4))", "np.array_equal(executed[i-1]['command'],m.QS['prefix_commands'][i-1])")
replace("complete=p['status']=='edge_adjacent_pair_completed' and len(scoops)==2 and all(s['complete'] for s in scoops)", "complete=p['status']=='adjacent_second_59um_completed' and len(scoops)==1 and scoops[0]['scoop']==2 and all(s['complete'] for s in scoops)")
begin=s.index(' if complete:\n  names=')
end=s.index(' # Rebuild observed maps',begin)
s=s[:begin]+''' original=m.read(Path(m.QS['source_prior_run'])/'report.json');original_audit=m.read(Path(m.QS['source_prior_run'])/'read-only-audit.json')
 check('prior_first_scoop_record_audit_passed',original_audit['record_integrity_passed'] and original_audit['scoops'][0]['complete'])
 check('first_return_checkpoint_identity',original['first_return_checkpoint']['sha256']==m.QS['source_sha256'] and original['first_return_checkpoint']['path']==m.QS['source'])
 check('first_scoop_result_not_rewritten',p['original_first_scoop']==original['scoops'][0])
 names=[x['name'] for x in original['phases']];required=['CLEAR_PILE','ALIGN','DUMP','CLEAR_TRUCK','WITHDRAW_SWING','RETURN_SAFE']
 check('prior_actual_unload_return_preserved',all(name in names for name in required) and [names.index(name) for name in required]==sorted(names.index(name) for name in required))
 if complete:
  check('final_stage_covers_final_frame',phases[-1]['end_frame']==n and phases[-1]['name']=='SCOOP2_RESIDUAL')
  check('terminal_snapshot_cycle2',m.read(str(terminal['path'])+'.json')['cycle_index']==2)
 rise=[x for x in clearance if 'RETURN_RISE' in x['phase'] and 'minimum_height_step_m' in x]
 check('59um_rise_rule_respected_in_logged_passes',all(x['minimum_height_step_m']>=-59e-6 for x in rise) and p['tolerance_um']==59)
 check('59um_guard_bound_to_new_runner',m.RiseHoldGuard.__module__=='rise_hold_guard' and m.sha(W/'minimum_guard.py')==protocol['bindings_sha256'][str(W/'minimum_guard.py')])
''' +s[end:]
replace('for cycle in (1,2):','for cycle in (2,):')
replace("snapshot=D/('prefix3.npz' if cycle==1 else 'first-returned.npz')", "snapshot=D/'zero.npz'")
replace("first=m.read(D/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy']", "first=m.read(Path(m.QS['source_prior_run'])/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy']")
replace('cumulative_budget_physical_frames=n+9450', 'cumulative_budget_physical_frames=n')
compile(s,'audit_second.py','exec');(W/'audit_second.py').write_text(s)
import shutil
shutil.copyfile(OLD/'measurement-bindings.json',W/'measurement-bindings.json')
print('SECOND_AUDIT_PREPARED')
