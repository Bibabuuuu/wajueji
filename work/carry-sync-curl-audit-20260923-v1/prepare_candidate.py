from pathlib import Path
root=Path('/home/qunwang/wajueji');out=root/'work/curl-compensation-short-ab-20260923-v1';out.mkdir(exist_ok=False)
s=(root/'work/carry-sync-curl-audit-20260923-v1/replay.py').read_text()
s=s.replace("out=root/'work/carry-sync-curl-audit-20260923-v1'","out=root/'work/curl-compensation-short-ab-20260923-v1'")
s=s.replace('import run_layered_47_sparse_manual_contact_episode as manual','import run_layered_47_sparse_manual_contact_episode as manual\nimport layered_mound_checkpoint_runtime as runtime\nimport hashlib')
s=s.replace("work=out/f'replay-build-{os.getpid()}'", "trace=trace[:600]\nwork=out/f'replay-build-{os.getpid()}'")
anchor=' fixture,held,restored=manual.motion.audit.restore_candidate_runtime'
inject='''
 original_render=runtime.render_capacity_bridge
 def render_candidate(*arguments):
  source,manifest=original_render(*arguments)
  decl='    std::array<double, 4> unloaded_feedforward{};'
  assert source.count(decl)==1
  source=source.replace(decl,decl+'\\n    std::array<double, 4> external_load_compensation{};\\n    double bucket_compensation_filter = 0.0;')
  emit='      array("unloaded_feedforward", unloaded_feedforward.data(), 4);'
  assert source.count(emit)==1
  source=source.replace(emit,emit+'\\n      array("external_load_compensation", external_load_compensation.data(), 4);')
  before='        const double before = data->time;\\n        mj_step(model.get(), data.get());'
  assert source.count(before)==1
  added="""        // Isolated A/B: activate only after the unchanged 480-frame prefix.
        // Compensate only half of attributed MPM bucket load. Native rock
        // contact reactions and unowned endpoint projection are not included.
        if (data->time >= 8.0 - 1e-9) {
          const double target_comp = std::clamp(-0.5 * equivalent_drive_load[3], -60000.0, 60000.0);
          const double alpha = model->opt.timestep / (0.1 + model->opt.timestep);
          const double filtered = bucket_compensation_filter + alpha * (target_comp - bucket_compensation_filter);
          const double max_change = 150000.0 * model->opt.timestep;
          bucket_compensation_filter += std::clamp(filtered - bucket_compensation_filter, -max_change, max_change);
          external_load_compensation[3] = bucket_compensation_filter;
          const int actuator = controller.hold_actuator[3];
          data->ctrl[actuator] += external_load_compensation[3] / model->actuator_gainprm[mjNGAIN * actuator];
        }
"""
  source=source.replace(before,added+before)
  (out/'candidate-bridge.cc').write_text(source)
  candidate={'isolated_candidate':True,'deployed':False,'gain':.5,'clamp_N':60000.,'slew_N_per_s':150000.,'filter_tau_s':.1,'activation_native_time_s':8.,'axis':'bucket_only','source_sha256':hashlib.sha256(source.encode()).hexdigest()}
  (out/'candidate-policy.json').write_text(json.dumps(candidate,indent=2))
  return source,{**manifest,'isolated_compensation_candidate':candidate}
 runtime.render_capacity_bridge=render_candidate
'''
s=s.replace(anchor,inject+anchor)
s=s.replace("'actuator_force':bridge.state['actuator_force'],", "'actuator_force':bridge.state['actuator_force'],'external_load_compensation':bridge.state['external_load_compensation'],")
s=s.replace("    previous=current;previous_pose=pose", "    if i<=480: assert bridge.state['external_load_compensation']==[0.0]*4\n    assert abs(bridge.state['external_load_compensation'][3])<=60000.0001\n    previous=current;previous_pose=pose")
s=s.replace("out/'final-001370.diagnostic.npz'", "out/'final-000600.diagnostic.npz'")
s=s.replace("   comparisons={k:bool(np.array_equal(v,saved[k])) for k,v in final.items() if k in saved and v.dtype.kind in 'fiu'}", "   prefix=dict(np.load(out/'diagnostic-000480.npz'));baseline=dict(np.load(root/'work/carry-sync-curl-audit-20260923-v1/diagnostic-000480.npz'))\n   comparisons={k:bool(np.array_equal(v,baseline[k])) for k,v in prefix.items() if k in baseline and v.dtype.kind in 'fiu'}")
s=s.replace("'all_numeric_arrays_exact':all(comparisons.values())", "'prefix_480_arrays_exact':all(comparisons.values()),'compensation_candidate_only':True")
(out/'replay_candidate.py').write_text(s)
compile(s,str(out/'replay_candidate.py'),'exec')
print('Prepared isolated candidate, not launched:',out)
