from pathlib import Path
import datetime, hashlib, json, shutil, sys, time
import numpy as np
R=Path('/home/qunwang/wajueji');W=R/'work/edge-adjacent-59um-20261015-v1'
OLD=R/'work/edge-first-two-scoop-20261015-v1';D=R/'runs/edge-first-two-scoop-20261015-continuation'
C=R/'work/second-scoop-rise-tolerance-59um-20261015-v1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,v:Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))
b=read(W/'budget.json');assert time.time()<b['epoch_deadline']-b['reserve_seconds'] and b['seconds']==3600
assert read(OLD/'closure.json')['status']=='closed'
cpu=read(C/'cpu-v1/report.json');assert all(cpu['checks'].values()) and cpu['check_count']==32
for name,h in cpu['code_sha256'].items():assert sha(C/name)==h
source=D/'first-returned.npz';control=read(D/'first-returned.control.json');metadata=read(str(source)+'.json')
assert sha(source)==control['checkpoint_sha256']=='5b5aba7bb5bfe157cc846a973a34d66fadc99c58c311ab71e21ea84b023f0392'
assert control['phase']=='RETURN_SAFE' and control['frame']==10504 and np.array_equal(control['previous_command'],np.zeros(4))
assert metadata['cycle_index']==1 and metadata['stage']=='returned' and not metadata.get('training_reset')
states=read(D/'display/states.json');cloud=np.load(D/'display/particles.npy',mmap_mode='r');offset=control['frame']
prefix=W/'expected-prefix';prefix.mkdir(exist_ok=True)
np.save(prefix/'particles.npy',cloud[offset:offset+4]);write(prefix/'states.json',states[offset:offset+4])
commands=[]
with (D/'dispatch.ndjson').open() as f:
 for line in f:
  row=json.loads(line)
  if offset<row['global_frame']<=offset+3:commands.append(row['command'])
  if row['global_frame']>=offset+3:break
assert len(commands)==3
with np.load(source) as z:assert np.array_equal(z['particle_q'],cloud[offset]) and np.array_equal(z['mechanism_qpos'],states[offset]['mechanism_qpos'])
q=dict(status='completed_first_return_source_for_new_restore_gate',source=str(source),source_sha256=sha(source),source_metadata_sha256=sha(str(source)+'.json'),source_control=str(D/'first-returned.control.json'),source_control_sha256=sha(D/'first-returned.control.json'),source_prior_run=str(D),source_frame=offset,prefix_commands=commands,physical_restore_qualified=False)
write(W/'qualified-return-source.json',q)
for name in ['minimum_guard.py','rise_hold_guard.py']:shutil.copyfile(C/name,W/name)
for name in ['inplace_cache.py','direct_reference_bridge.py']:shutil.copyfile(OLD/name,W/name)
s=(OLD/'run_prepoint_continuation.py').read_text()
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:90],s.count(old))
 s=s.replace(old,new)
s=s.replace("W=R/'work/edge-first-two-scoop-20261015-v1'", "W=R/'work/edge-adjacent-59um-20261015-v1'")
replace("OUT=R/'runs/edge-first-two-scoop-20261015-continuation'", "OUT=R/'runs/edge-adjacent-59um-20261015-v1'")
replace("sys.path[:0]=[str(R/'work/second-scoop-rise-hold-20261014-v1')", "sys.modules.pop('minimum_guard',None);sys.modules.pop('rise_hold_guard',None)\nsys.path[:0]=[str(R/'work/edge-adjacent-59um-20261015-v1'),str(R/'work/second-scoop-rise-hold-20261014-v1')")
replace("QS=read(W/'qualified-prepoint-source.json')", "QS=read(W/'qualified-return-source.json')")
replace("('edge_first_adjacent_two_scoop',7200,30000,7200,0)","('edge_adjacent_second_59um',3600,13000,7200,0)")
replace("d['epoch_deadline']-d['epoch_start']-7200", "d['epoch_deadline']-d['epoch_start']-3600")
replace("TimeoutError('new2h_save_audit_reserve')", "TimeoutError('new60min_save_audit_reserve')")
replace("self.active_cycle=1;budget.check()", "self.active_cycle=2;budget.check()")
replace("self.metadata['stage']=='transport'", "self.metadata['stage']=='returned'")
replace("if self.frames+9450>=30000:raise ValueError('whole_pair_frame_ceiling')", "if self.frames>=13000:raise ValueError('new60min_frame_ceiling')")
replace("physical_scope='D190_new2h_edge_pair'", "physical_scope='D192_new60min_adjacent_second_59um'")
replace("cycle=1;previous=np.zeros(4)", "cycle=2;previous=np.zeros(4)")
replace("scope='same-instance edge-first dig, real geometric unload/return, fresh-stock inward adjacent second dig; no matched-central rerun or training'", "scope='one restored native instance from completed first return through adjacent second dig; first scoop/unload/return are prior actual phases, not claimed uninterrupted; no training or second transport',tolerance_um=59,original_first_scoop=read(Path(QS['source_prior_run'])/'report.json')['scoops'][0]")
replace("ProbeDisplayCache(OUT/'display',30000", "ProbeDisplayCache(OUT/'display',13000")
replace("prior=W/'prepoint-qualification/replica-a';prefix_frame=0", "prior=W/'expected-prefix';prefix_frame=0")
replace("prefix_commands=[np.zeros(4)]*3", "prefix_commands=QS['prefix_commands']")
replace("if r.frames==3:r.prefix_verified=True;print('PREFIX3_EXACT',flush=True)", "if r.frames==3:r.prefix_verified=True;report['prefix_checkpoint']=save('prefix3','transport');print('PREFIX3_EXACT',flush=True)")
begin=s.index("    r.step=tracked;r.step_authorized=True;phase='RESTORE_PREFIX3'")
end=s.index("    phase='FINAL_HOLD';",begin)
s=s[:begin]+'''    assert manual.compact_state(r.bridge.state)==old_states[0] and np.array_equal(r.fixture.state.particle_q.numpy(),old_cloud[0]),'original_source_zero_compact_or_particles_mismatch'
    r.step=tracked;r.step_authorized=True;phase='SCOOP2_RESTORE_READY'
    first=read(Path(QS['source_prior_run'])/'scoop-1-plan/actual-cut-trace.json')['trace'][0]['mouth_xy']
    endpoint,second=approach_and_dig(first)
'''+s[end:]
replace("status='edge_adjacent_pair_completed'", "status='adjacent_second_59um_completed'")
replace("capture_origin='D190_same_native_edge_adjacent_pair'", "capture_origin='D192_restored_return_to_adjacent_second_59um'")
s=s.replace('"""D190: new2h, one native edge-first/adjacent pair; no training or resets."""','"""D192: new60min, restore completed return and finish adjacent second dig at59um."""')
compile(s,'run_second.py','exec');(W/'run_second.py').write_text(s)
sys.path.insert(0,str(W));import run_second as m
assert Path(sys.modules['minimum_guard'].__file__).parent==W and Path(sys.modules['rise_hold_guard'].__file__).parent==W
assert sys.modules['minimum_guard'].MAX_DOWNWARD_STEP_M==59e-6 and m.OUT.name=='edge-adjacent-59um-20261015-v1'
budget=m.Budget();budget.check()
fake=m.BoundRuntime.__new__(m.BoundRuntime);fake.budget=budget;fake.frames=13000;fake.step_authorized=True;fake.prefix_verified=True
try:fake.step(np.zeros(4))
except ValueError as exc:assert str(exc)=='new60min_frame_ceiling'
else:raise AssertionError('frame_cap')
old_protocol=read(OLD/'live-protocol-continuation.json');bindings=dict(old_protocol['bindings_sha256'])
for p in [W/'run_second.py',W/'minimum_guard.py',W/'rise_hold_guard.py',W/'inplace_cache.py',W/'direct_reference_bridge.py',W/'qualified-return-source.json',W/'budget.json',source,Path(str(source)+'.json'),D/'first-returned.control.json',D/'report.json',prefix/'particles.npy',prefix/'states.json',C/'cpu-v1/report.json']:
 bindings[str(p)]=sha(p)
assert all(sha(p)==h for p,h in bindings.items())
protocol=dict(status='new60min_second_59um_preflight_passed',bindings_sha256=bindings,source=q,
  original_display_cache_hashes=read(D/'report.json')['display']['cache_sha256'],expected_prefix='Original continuous first3 nonzero rise commands, particles and compact states after completed RETURN_SAFE, not independent zero commands',
  source_is_failure=False,physical_restore_qualified=False,physical_admission='User-authorized entry to strict zero and original-prefix gates; later frames require gate success',
  output=str(m.OUT),log=str(W/'console.log'),seconds=3600,reserve_seconds=600,tolerance_um=59,new_physics_steps=0,training_updates=0,
  launch='OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python -u work/edge-adjacent-59um-20261015-v1/run_second.py')
write(W/'live-protocol-continuation.json',protocol)
write(W/'preflight.json',dict(status='passed',source_identity=True,source_cached_zero_arrays=True,original_prefix_extracted=True,guard59_import_bound=True,frame_ceiling=True,old_budget_closed=True,all_bindings_match=True,new_physics_steps=0,remaining_seconds=budget.remaining()))
note='''用户新增60分钟已授权，仅首铲返回源恢复、原生零步/原连续首3帧、59微米上抬及高位保持、接近和完整第二铲1170参考＋0.5秒保载；残差≤7200帧、总物理≤13000帧。预算2026-10-15 09:09:46至10:09:46（Asia/Shanghai），09:59:46起只保存审计；失败或到期停止，不训练、不次铲运输，旧预算不重开。CPU接入预检通过，原生恢复及第二铲尚未验证。来源为合格RETURN_SAFE快照5b5aba7b…，原continuation_qualified=false保留，禁止失败源续跑；须新实例零步及原连续前三帧严格复现后才继续。冻结模型、观测、等待映射和其他保护不改；只应用D191已确认的59微米逐帧回落容差。运行器work/edge-adjacent-59um-20261015-v1/run_second.py，日志同目录console.log，输出runs/edge-adjacent-59um-20261015-v1。中断保存recovery但不授予自动恢复资格，无额外重试或延时。命令见live-protocol-continuation.json，台式机GPU0单实例，估计40分钟物理、最后10分钟审计。'''
receipt={}
for name in ['docs/PROJECT_PLAN.md','docs/CURRENT_STATUS.md','docs/plans/S05-imitation-and-rl.md','docs/decisions.md','docs/plans/INDEX.md']:
 p=R/name;before=sha(p);text=p.read_text();backup=W/'records-before'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,backup)
 i=text.index('\n## ');text=text[:i]+'\n\n## 2026-10-15 D-192：新增60分钟，恢复并验证完整贴边第二铲\n\n'+note+'\n'+text[i:]
 if name=='docs/decisions.md':
  j=text.index('\n',text.index('## 当前决策索引'));text=text[:j]+'\n\n- `D-192`：新增60分钟，首铲返回源严格恢复后验证59微米上抬/接近及完整第二铲；最后10分钟保存审计，0训练、不次铲运输；当前仅CPU接入预检通过。'+text[j:]
 assert sha(p)==before;p.write_text(text);receipt[name]=dict(before=before,after=sha(p))
write(W/'record-start-receipt.json',receipt)
print('NEW60MIN_PREFLIGHT_PASSED remaining',round(budget.remaining()),flush=True)
