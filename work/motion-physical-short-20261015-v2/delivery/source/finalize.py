from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,hashlib,shutil,subprocess,time,tarfile
R=Path('/home/qunwang/wajueji');W=R/'work/motion-physical-short-20261015-v2';D=R/'runs/motion-physical-short-20261015-v2'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=read(D/'report.json');a=read(D/'read-only-audit.json');b=read(W/'budget.json');assert (W/'physical-finished.json').exists() and not (W/'closure.json').exists()
zone=ZoneInfo('Asia/Shanghai');stamp=lambda t:datetime.fromtimestamp(t,zone).isoformat()
closed=dict(status='closed',physical_status=p['status'],failure=p.get('failure'),physics_steps=p['physical_frames'],native_initializations=p['native_initializations'],inference_calls=0,training_updates=0,started_at=stamp(b['epoch_start']),deadline_at=stamp(b['epoch_deadline']),closed_at=stamp(time.time()),unused_seconds_sealed=max(0,b['epoch_deadline']-time.time()),budget_reopened=False,automatic_retry=False,record_audit_passed=a['record_integrity_passed'],report_sha256=sha(D/'report.json'),audit_sha256=sha(D/'read-only-audit.json'))
for item in p['cases']:
 folder=D/item['name'];cache=folder/'display/reconstruction.json'
 if cache.exists():
  m=read(cache);m.update(label='D196 independent short-case physical cache',capture_origin='D196 actual native short case; not a combined two-scoop run',physical_case=item['name']);cache.write_text(json.dumps(m,ensure_ascii=False,indent=2))
(W/'closure.json').write_text(json.dumps(closed,ensure_ascii=False,indent=2))
summary=dict(closure=closed,cases=[{k:x.get(k) for k in ['name','status','failure','frames','phases','final_alignment','full_roi_admission','initial_cavity_particles','final_cavity_particles','initial_cohort_still_in_cavity','particle_measurement_scope']} for x in p['cases']],metrics=a['metrics'],old_loaded_max_pitch_error_deg=4.110049735068719,scope='Independent loaded transfer and post-unload approach diagnostics; no new digging or full two-scoop sequence.',limitations='Cavity counts do not prove calibrated fill/no spill. Peak pitch-to-reference error is a short-run comparison, not statistical damping validation.')
completed_return=next((x for x in p['cases'] if x['name']=='return' and x['status']=='completed'),None)
old_return=147.08333333333334
if completed_return:
 summary.update(actual_return_seconds=completed_return['frames']/60,old_actual_return_seconds=old_return,return_seconds_reduction=old_return-completed_return['frames']/60,return_time_reduction_fraction=1-completed_return['frames']/60/old_return,return_time_scope='Same qualified dump snapshot and original prefix gate, through reference-prepoint hold, physical simulation time; one case, not real-machine wall-clock throughput.')
summary['damping_conclusion']='Loaded lift peak pitch-to-reference error4.11005deg to4.00079deg; small improvement only, damping problem not resolved. ALIGN is only a120-frame entry test.'
(D/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
case_text='；'.join(x['name']+'='+x['status']+'、'+str(x['frames'])+'帧' for x in p['cases']);metrics=json.dumps(a['metrics'],ensure_ascii=False)
timing=('卸料后到原参考预备点实际'+str(round(summary['actual_return_seconds'],4))+'仿真秒，旧147.0833秒；这是同合格卸料快照及原连续首3帧门后的单例动作时间，不是实机或GPU墙钟吞吐。' if completed_return else '没有合格的新完整返回耗时。')
result='**本轮物理已结束、预算关闭。** 状态'+p['status']+'；'+case_text+'。总物理'+str(p['physical_frames'])+'帧，原生实例'+str(p['native_initializations'])+'个，推理/训练均0。停止原因'+str(p.get('failure'))+'。记录审计'+str(a['record_integrity_passed'])+'，'+str(a['checks_passed'])+'/'+str(len(a['checks']))+'项；记录完整与两组完成分别判断。'+timing+' 举升最大斗姿对参考误差4.11005度→4.00079度，仅小幅改善，减振未解决；回转仅120帧入口，不能外推整段回转。实际指标：'+metrics+'。\n\n仅独立短段，未挖掘或做完整两铲；斗腔粒子队列只是保料诊断，不是标定满斗率或无撒料证明。各原保护及CPU路径沿用，失败不自动重试，旧预算不重开。证据：runs/motion-physical-short-20261015-v2/report.json、read-only-audit.json、summary.json、各组display/；work/motion-physical-short-20261015-v2/closure.json。原运行、模型和正式控制/物理源码保留，无Git提交。\n\n'
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md','docs/decisions.md']:
 f=R/rel;text=f.read_text();old='## 2026-10-15 D-196：两组短段物理已重新授权';assert old in text;text=text.replace(old,'## 2026-10-15 D-196：两组短段物理结果，预算已关闭\n\n'+result,1)
 if rel.endswith('decisions.md'):
  oldline='- `D-196`：用户重新要求启动仿真；新30分钟，两源最多1000/3300帧，原生零步/首3帧门，任一失败整轮停止，旧预算不重开。';assert oldline in text;text=text.replace(oldline,'- `D-196`：'+case_text+'；总物理'+str(p['physical_frames'])+'帧，记录审计'+str(a['record_integrity_passed'])+'，状态'+p['status']+'；预算关闭，不自动重试。',1)
 f.write_text(text)
for rel in ['docs/PROJECT_PLAN.md','docs/acceptance.md']:
 f=R/rel;text=f.read_text();first,_,rest=text.partition('\n');f.write_text(first+'\n\n## 2026-10-15 D-196短段物理已关闭\n\n'+result+rest)
check=subprocess.run(['git','diff','--check','--','docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md','docs/decisions.md','docs/PROJECT_PLAN.md','docs/acceptance.md'],cwd=R,capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
out=W/'delivery';out.mkdir();(out/'source').mkdir()
for name in ['report.json','read-only-audit.json','summary.json']:shutil.copy2(D/name,out/name)
for name in ['budget.json','preflight.json','closure.json']:shutil.copy2(W/name,out/name)
for name in ['run.py','prepare.py','audit.py','finalize.py']:shutil.copy2(W/name,out/'source'/name)
for x in p['cases']:
 folder=out/x['name'];folder.mkdir()
 for name in ['actions.ndjson','clearance.ndjson','executed-reference.npz']:
  path=D/x['name']/name
  if path.exists():shutil.copy2(path,folder/name)
(out/'README.md').write_text('# 运输减振与卸料后合并接近：物理短检\n\n'+result+'\n源代码在source/，原生恢复及原连续首3帧核验见report.json。全帧粒子/机构缓存留台式机各组display/，不把两组拼成同一连续流程。下一步只根据本次真实结果决定，未批准新的训练、完整挖掘或额外物理预算。\n')
(out/'MANIFEST.sha256').write_text('\n'.join(sha(f)+'  '+f.relative_to(out).as_posix() for f in sorted(out.rglob('*')) if f.is_file())+'\n')
with tarfile.open(W/'motion-physical-short-20261015-v2.tar.gz','w:gz') as archive:
 for f in out.iterdir():archive.add(f,arcname=f.name)
print(json.dumps(summary,ensure_ascii=False,indent=2))
