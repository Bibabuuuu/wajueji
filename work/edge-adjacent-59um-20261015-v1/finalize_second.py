"""Close this authorization after physics/audit/export; never launches a run."""
from pathlib import Path
import datetime, hashlib, json, subprocess, time, shutil
R=Path('/home/qunwang/wajueji');W=R/'work/edge-adjacent-59um-20261015-v1';D=R/'runs/edge-adjacent-59um-20261015-v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
b=read(W/'budget.json');p=read(D/'report.json');audit=read(D/'read-only-audit.json');summary=read(D/'summary/results.json')
assert p['status'] not in ['initializing','running'] and audit['record_integrity_passed']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'physics_must_be_stopped'
assert p['neural_updates']==0 and p['physical_run_finished_epoch']<=b['epoch_deadline']
complete=audit['two_scoop_goal_complete'];old=Path(p['source']).parent
first=summary['new_scoops'][0]['retained_fill_fraction']*100
second_item=next((s for s in summary['new_scoops'] if s['scoop']==2),None)
second=(second_item['retained_fill_fraction']*100 if second_item and second_item['complete'] else None)
strip=read(D/'figures/strip-metrics.json')
overlap=100*strip['overlap_fraction_of_second']
paths=read(D/'summary/actual-mouth-paths.json')
paths['frame_origin_runs']={'1':str(old),'2':str(D)}
(D/'summary/actual-mouth-paths.json').write_text(json.dumps(paths,ensure_ascii=False,indent=2))
result_line=(f'边缘首铲短保载显示{first:.6f}%，贴边第二铲{second:.6f}%，两铲显示值之和{first+second:.6f}个百分点。' if complete else f'两铲目标仍未完成；边缘首铲短保载显示{first:.6f}%，第二铲无完整有效保载值。')
message=(f'本轮新增60分钟预算09:09:46至10:09:46（Asia/Shanghai），最后10分钟保存审计。原生24数组/控制侧/扫掠与时钟恢复、原连续首3帧发令/粒子/状态严格对照通过。'
         f'新运行{p["frames"]}物理帧，1个原生实例，0训练；状态{p["status"]}，失败原因{p.get("failure")}。'+result_line+
         f'{audit["check_count"]}项独立审计通过；记录审计与业务完成分别判断。'
         '只使用D191已授权的59微米相邻帧最低点回落容差，冻结模型/观测/等待映射及其余保护保留。'
         '首铲与实际运卸返回来自原D190合格阶段，通过完整原生快照与原连续前缀续接第二铲，土堆不重置；不冒称整个两铲从初态单实例不中断。'
         '历史中央142.6193%/68.175044%的首铲等待接口不同，只作趋势参考；无同控制中央重跑或统计重复，不能隔离布局因果、泛化成功率或正式装载体积。'
         f'实际斗口按名义0.60米宽平面投影的第二铲重叠比例约{overlap:.2f}%，仅为几何诊断，不是重复挖空体积。'
         '终点黏附允许，无净空门。次铲运输未执行。预算关闭、剩余封存，不自动重试或续跑，旧预算均不重开。'
         '证据：runs/edge-adjacent-59um-20261015-v1/report.json、read-only-audit.json、summary/、media-v2/及figures/；work/edge-adjacent-59um-20261015-v1/closure.json。S05仍进行中，无Git提交。')
heading='## 2026-10-15 D-192：59微米贴边第二铲验证结果，预算已关闭'
receipt={}
for name in ['docs/PROJECT_PLAN.md','docs/CURRENT_STATUS.md','docs/plans/S05-imitation-and-rl.md','docs/acceptance.md','docs/decisions.md','docs/plans/INDEX.md']:
 path=R/name;text=path.read_text();before=sha(path);backup=W/'records-before-final'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,backup)
 marker='## 2026-10-15 D-192：新增60分钟，恢复并验证完整贴边第二铲'
 if marker in text:
  start=text.index(marker);end=text.index('\n## ',start+len(marker));text=text[:start]+heading+'\n\n'+message+'\n'+text[end:]
 else:
  start=text.index('\n## ');text=text[:start]+'\n\n'+heading+'\n\n'+message+'\n'+text[start:]
 if name=='docs/decisions.md':
  oldline='- `D-192`：新增60分钟，首铲返回源严格恢复后验证59微米上抬/接近及完整第二铲；最后10分钟保存审计，0训练、不次铲运输；当前仅CPU接入预检通过。'
  text=text.replace(oldline,'- `D-192`：'+result_line+f'新物理{p["frames"]}帧、0训练、{audit["check_count"]}项独立审计通过，跨合格快照连续土堆续接，预算已关闭。')
 assert sha(path)==before;path.write_text(text);receipt[name]=dict(before=before,after=sha(path))
subprocess.run(['git','diff','--check','--',*receipt],cwd=R,check=True)
(W/'record-final-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
now=time.time();tz=datetime.timezone(datetime.timedelta(hours=8))
closure=dict(status='closed',authorization_seconds=3600,epoch_start=b['epoch_start'],epoch_deadline=b['epoch_deadline'],
 closed_epoch=now,closed_local=datetime.datetime.fromtimestamp(now,tz).isoformat(),remaining_seconds_sealed=max(0,b['epoch_deadline']-now),
 closure_after_deadline_seconds=max(0,now-b['epoch_deadline']),physical_runtime_finished_epoch=p['physical_run_finished_epoch'],
 new_physical_frames=p['frames'],native_instances=p['native_initializations'],training_updates=0,old_budgets_reopened=False,
 two_scoop_goal_complete=complete,uninterrupted_from_initial_state=False,continuity='Strict full native checkpoint and original first3 prefix; physical soil evolution retained across restored phase',
 no_more_physics_or_training=True,audit_sha256=sha(D/'read-only-audit.json'),source_sha256=p['source_sha256'])
(W/'closure.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2));(D/'closure.json').write_text(json.dumps(closure,ensure_ascii=False,indent=2))
text='# 边缘首铲与贴边次铲：59微米验证\n\n'+result_line+'\n\n'+('两铲均完成1170参考及0.5秒保载。' if complete else '未取得完整两铲结果。')+'本轮预算已关闭，0训练。\n\n'
text+='首铲与运卸返回来自已审计D190真实过程；第二铲从合格返回快照严格恢复，原连续首3帧通过后推进，土堆不重置。不是从初态不间断的单实例两铲。\n\n'
text+='此前1微米保护触发的失败记录原样保留。本次仅使用已确认的59微米逐帧回落容差；其余保护、模型和物理参数保持。\n\n'
text+='历史中央首铲142.6193%、第二铲68.175044%只作趋势参考。首铲等待接口不同，没有同控制中央重跑或统计重复，因此不能单独归因布局。显示百分比及其和不是标定装载体积。\n\n'
text+=f'第二铲较历史显示提高{second-68.1750436023466:.4f}个百分点。按名义0.60米斗宽计算，第二铲路径平面投影约{overlap:.2f}%与首铲重叠；这不是重复挖空体积，尚不能用它解释全部保载差异。\n\n' if complete else ''
text+='- [保载显示对照](figures/fill-comparison.png)\n- [实际斗口路径](figures/actual-cut-strips.png)\n- [第二铲关键帧](media-v2/scoop-2-keyframes.png)\n- [原始指标](summary/metrics.csv)、[独立审计](production/read-only-audit.json)、[完整运行报告](production/report.json)\n\n'
text+='- [首铲关键帧及来源](prior-first-scoop/scoop-1-keyframes.png)\n\n'
text+=f'本轮{p["frames"]}新物理帧，1个原生实例；{audit["check_count"]}项独立审计通过。运行状态：{p["status"]}；失败信息：{p.get("failure")}。次铲运输未执行。大体积原生快照、全帧粒子及日志保留在台式机runs/edge-adjacent-59um-20261015-v1。\n'
(D/'summary/README.md').write_text(text)
print(json.dumps(dict(status='closed',two_scoop_complete=complete,first_percent=first,second_percent=second,remaining_seconds_sealed=closure['remaining_seconds_sealed']),ensure_ascii=False))
