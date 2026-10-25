"""Close bounded physics and record actual outcomes, including noncompletion."""
from pathlib import Path
import sys,time,datetime,subprocess
R=Path('/home/qunwang/wajueji');W=R/'work/edge-first-two-scoop-20261015-v1';sys.path.insert(0,str(W));import run_prepoint_continuation as m
D=m.OUT;p=m.read(D/'report.json');a=m.read(D/'read-only-audit.json');b=m.read(W/'budget.json');media=m.read(D/'media-v2/manifest.json');s=m.read(D/'summary/results.json');diagnosis=m.read(D/'rise-stop-diagnosis.json')
assert a['record_integrity_passed'] and media['status']=='rendered' and not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
complete=a['two_scoop_goal_complete'];now=time.time();zone=datetime.timezone(datetime.timedelta(hours=8));stamp=lambda epoch:datetime.datetime.fromtimestamp(epoch,zone).isoformat()
closure=dict(status='closed',authorization_seconds=7200,epoch_start=b['epoch_start'],epoch_deadline=b['epoch_deadline'],started_local=stamp(b['epoch_start']),deadline_local=stamp(b['epoch_deadline']),closed_local=stamp(now),closed_epoch=now,
 physical_runtime_finished_epoch=p['physical_run_finished_epoch'],physical_attempt_frames=[4654,4790,6,p['frames']],cumulative_physical_frames=9450+p['frames'],
 native_instances=dict(approach_guard_stop=1,logging_stop=1,prepoint_qualification=2,production=1),training_updates=0,
 remaining_seconds_sealed=max(0,b['epoch_deadline']-now),closure_after_deadline_seconds=max(0,now-b['epoch_deadline']),old_budgets_reopened=False,
 production_status=p['status'],failure=p.get('failure'),two_scoop_goal_complete=complete,no_more_physics_or_training=True,audit_sha256=m.sha(D/'read-only-audit.json'),media_manifest_sha256=m.sha(D/'media-v2/manifest.json'))
m.write(W/'closure.json',closure)
first=next((x for x in a['scoops'] if x['scoop']==1),None);second=next((x for x in a['scoops'] if x['scoop']==2),None)
describe=lambda x:('未取得完整保载值' if not x or not x['complete'] else f"最低短保载显示{x['retained_fill_fraction']*100:.6f}%，{x['frames']}帧，等待占{x['wait_fraction']*100:.3f}%")
outcome='两铲挖掘及保载均已验证' if complete else '两铲目标未完成，本轮物理已停止'
body=f'''本轮按用户新增2小时执行，截止2026-10-15 02:41:47，最后15分钟用于保存审计。结果：**{outcome}**。首铲{describe(first)}；第二铲{describe(second)}。生产实例实际{p['frames']}帧；包含两次前置接近停止及6帧恢复核验，累计{9450+p['frames']}新物理帧，0训练。运行状态{p['status']}；停止原因{p.get('failure','无')}。全部结果以原始完整日志及独立审计为准，不把记录完整当作两铲成功。

两次前置问题均保留：4654帧因原32cm门停止；修复下降末点后，4790帧成功完成预备点，随后因记录器读取未激活策略字段而停止。日志生命周期已修复；未改模型、观测、等待控制或任何物理/保护参数。从成功阶段快照dc6b1f23…完成双实例零步及各3帧一致核验后续接，首铲、运卸返回及次铲准备由生产实例推进，第二铲挖掘尚未执行；此前空中接近不冒称单实例不中断。总计5次原生初始化含前置验证，生产实例本身1次；不是5次挖掘评价。失败原样保留，不从failure/recovery源自动续跑。

独立审计{a['check_count']}项检查通过；实际发令、控制偏置、冻结预测、221维输入、实际关节等待律及保载显示复算，保存源/物理时钟/库存/原保护和新观测选点按实际覆盖范围核验。{'完整两铲、中间真实运卸返回及新土堆贴边选点通过；' if complete else '未完成部分没有有效第二铲/两铲合计结果，不用首铲提升代替两铲结论；'}旧中央142.6193%／68.175044%只作历史趋势参考，原首铲等待接口不同，不证明纯布局因果收益。显示口径含土石/堆高、用户0.2阈值未重新标定，不是正式净装车体积，不授予三铲或跨场景成功率。终点允许黏附保持。

停止点是上抬第{diagnosis['stage_executed_frames']}/{diagnosis['stage_nominal_frames']}个名义帧的后状态，最低采样点回落{diagnosis['downward_step_mm']:.9f}mm，原容差0.001mm，土堆间距{diagnosis['after']['stock_gap_m']:.6f}m。缓存状态可通过既有高位保持准入，但运行器要到下一循环才切换；末帧该由哪种保护覆盖仍需确认，仅完成CPU准入检查，未改保护语义、未新跑物理，旧失败不改判。下一步先明确该末帧边界，再以新授权验证第二铲，当前不自动继续。

预算已关闭，未使用余时封存，GPU物理进程已结束，不自动继续。S05仍进行中；后续物理需新范围与预算。证据：runs/edge-first-two-scoop-20261015-continuation/ 的report.json、read-only-audit.json、first-dig-analysis.json、summary/、media-v2/、figures/；Windows deliverables/edge-adjacent-overnight-20261015/。无Git提交。
'''
receipt=m.read(W/'record-continuation-receipt.json');new={};backup=W/'records-before-final';backup.mkdir(exist_ok=False)
for name,v in receipt.items():
 path=R/name;assert m.sha(path)==v['after'],('record_conflict',name);before=m.sha(path);dst=backup/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(path.read_bytes());text=path.read_text();heading='## 2026-10-15 D-190夜间结果：'+outcome+'\n\n'
 if name=='docs/decisions.md':
  index=text.index('## D-190：');text=text[:index]+heading+body+'\n'+text[index:]
  lines=text.splitlines();lines=[f'- `D-190`：{outcome}；首铲{describe(first)}，第二铲{describe(second)}；累计{9450+p["frames"]}物理帧、0训练，预算关闭。' if line.startswith('- `D-190`：') else line for line in lines];text='\n'.join(lines)+'\n'
 else:
  index=text.index('\n')+1;text=text[:index]+'\n'+heading+body+'\n'+text[index:]
 path.write_text(text);new[name]=dict(before=before,after=m.sha(path))
m.write(W/'record-final-receipt.json',new)
requirements=dict(two_complete_digs_and_30frame_holds=complete,original_protections_and_frozen_model=a['record_integrity_passed'],no_training=p['neural_updates']==0,
 actual_keyframes=bool(media['keyframes']),source_data=(D/'summary/metrics.csv').is_file(),figures=(D/'figures/fill-comparison.png').is_file() and (D/'figures/actual-cut-strips.png').is_file(),budget_closed=True)
m.write(W/'goal-evidence.json',dict(full_objective_proven=all(requirements.values()),requirements=requirements,primary_result=outcome,closure_sha256=m.sha(W/'closure.json'),record_receipt_sha256=m.sha(W/'record-final-receipt.json')))
print(m.encode(dict(result=outcome,first=first,second=second,closure=closure,goal=requirements)),flush=True)

