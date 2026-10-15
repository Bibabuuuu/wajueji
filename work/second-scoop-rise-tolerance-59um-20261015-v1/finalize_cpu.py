"""Record D-191 CPU evidence without reopening physics or old results."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess

R = Path('/home/qunwang/wajueji')
W = Path(__file__).resolve().parent
result = json.loads((W/'cpu-v1/report.json').read_text())
assert result['status'] == 'cpu_candidate_passed' and all(result['checks'].values())
assert result['check_count'] == 32 and result['new_physics_steps'] == 0
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p) == h for p,h in result['frozen_inputs_sha256'].items())
assert all(sha(W/p) == h for p,h in result['code_sha256'].items())
evidence = '`work/second-scoop-rise-tolerance-59um-20261015-v1/cpu-v1/report.json`'
heading = '## 2026-10-15 D-191：最低点逐帧回落容差59微米，CPU已验证'
summary = ('用户明确选择把上抬最低点允许逐帧回落从1微米提高到59微米（0.059毫米、5.9e-5米），'
           '替代此前待选的末帧保护归属调整；本次只实施并验证该容差候选。'
           '原模式切换顺序不改，仍在名义上抬完成后通过原端点及32厘米门进入高位保持。')
validation = ('32项CPU检查通过，2151个真实缓存上抬帧全通过59微米候选；'
              '原末帧回落1.065903微米因该容差变化通过，其他2150帧检查结果与旧版逐项一致。'
              '精确59微米允许、紧邻其上的浮点值拒绝，继承土堆间距/非上抬32厘米门、'
              '提前切换拒绝和保持端点误差规则保留。缓存末态通过原保持准入，仅为离线判定。')
boundary = ('新物理步、原生初始化、策略推理和训练均0；未修改物理参数、冻结模型、观测或等待控制。'
            '旧1微米实现、夜间failed报告和关闭预算均逐文件SHA不变，旧失败不改判。'
            '候选采用独立目录，旧运行器仍绑定旧类，尚未接入任何物理运行器。'
            '59微米是相邻帧回落容差，不是累计下降额度或新的净空门。')
next_step = ('下一门：在新物理范围和预算获确认后，接入59微米候选并从原合格first-returned来源重新恢复，'
             '严格核验原生零步/首3帧及来源身份，再验证上抬、高位保持、后续接近和完整第二铲及0.5秒保载。'
             '不从失败recovery自动续跑；旧两小时预算继续封存。两铲目标仍未完成，第二铲没有新满斗率。')
texts = {
 'docs/PROJECT_PLAN.md': summary+'\n\n'+validation+'\n\n'+next_step+'证据：'+evidence+'。S05仍进行中。',
 'docs/CURRENT_STATUS.md': summary+'\n\n'+validation+'\n\n'+boundary+'\n\n'+next_step+'证据：'+evidence+'。',
 'docs/plans/S05-imitation-and-rl.md': summary+'\n\n状态：`已授权实施`、`已实现`、`CPU已验证`；`物理未验证`。\n\n'+validation+'\n\n'+boundary+'\n\n'+next_step+'\n\n候选：`work/second-scoop-rise-tolerance-59um-20261015-v1/minimum_guard.py`及`rise_hold_guard.py`；回归入口`check_cpu.py`。证据：'+evidence+'。原minimum/rise-hold目录不覆盖，回退继续使用旧1微米类。',
 'docs/acceptance.md': validation+'\n\n'+boundary+'\n\n本次只授予59微米候选的CPU资格，不授予新原生恢复、完整高位保持、接近、第二铲保载或两铲改善结论。证据：'+evidence+'；原夜间41项记录审计及failed结论保留。',
 'docs/decisions.md': summary+'\n\n授权来源：用户对末帧处理讨论明确回复“提高门槛，提高到59微米”。该明确选择优先于原建议的切换顺序方案；不再把模式归属选择列为待确认。\n\n'+validation+'\n\n'+boundary+'\n\n'+next_step+'证据：'+evidence+'。',
 'docs/plans/INDEX.md': 'S05仍`进行中`。'+summary+'32项CPU检查及2151帧缓存回归通过；原夜间两铲失败不改判，旧预算关闭。下一门为另获授权后的候选接入、原生恢复与第二铲物理验证。证据：'+evidence+'。',
}
receipt = {'decision':'D-191', 'timezone':'Asia/Shanghai',
           'recorded_local':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
           'physics_steps':0, 'training_updates':0, 'files':{}}
for name, body in texts.items():
    p = R/name
    original = p.read_text()
    assert heading not in original, ('already_recorded', name)
    before = sha(p)
    backup = W/'records-before'/name
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, backup)
    anchor = original.find('\n## ')
    assert anchor >= 0, name
    updated = original[:anchor]+'\n\n'+heading+'\n\n'+body+'\n'+original[anchor:]
    # Historical D-190 keeps its facts; its former open choice is now resolved.
    updated = updated.replace('下一步先明确该末帧边界，再以新授权验证第二铲，当前不自动继续。',
        '当时末帧边界待选；现由D-191选择59微米容差候选，原运行失败保留，物理仍需新授权。')
    if name == 'docs/decisions.md':
        updated = updated.replace('## 当前决策索引（2026-10-14）', '## 当前决策索引（2026-10-15）')
        index = updated.index('\n', updated.index('## 当前决策索引'))
        updated = updated[:index]+'\n\n- `D-191`：用户确认逐帧最低点回落容差59微米；32项CPU和2151帧缓存通过，旧模式切换及其他保护保持，未新增物理/训练，旧预算关闭。'+updated[index:]
    if name == 'docs/plans/INDEX.md':
        updated = updated.replace('| S05 | [模仿学习与强化学习](S05-imitation-and-rl.md) | active | D182首铲保留；第二铲CPU候选通过，观测转换CPU已验证，物理预算待确认 |',
            '| S05 | [模仿学习与强化学习](S05-imitation-and-rl.md) | active | D191：59微米容差CPU通过，第二铲物理验证及新预算待确认；D190两铲未完成 |')
    assert sha(p) == before, ('concurrent_change', name)
    p.write_text(updated)
    receipt['files'][name] = {'before_sha256':before,'after_sha256':sha(p)}
subprocess.run(['git','diff','--check','--',*texts], cwd=R, check=True)
(W/'record-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
(W/'README.md').write_text('''# 上抬回落容差59微米：CPU候选

已按用户明确要求，把最低点相邻帧允许回落量从1微米调整为**59微米＝0.059毫米＝5.9e-5米**。旧版与夜间失败记录不覆盖。

**32项CPU检查通过**。2151帧真实缓存中，原停止末帧（回落约1.066微米）通过；其余2150帧检查结果与旧版一致。精确59微米允许，超过59微米拒绝。继承土堆间距、非上抬32厘米门、模式切换时序、提前切换拒绝和保持端点规则保留。

这只是CPU缓存回归。**新物理步0、原生初始化0、策略推理0、训练0**。旧运行依然failed，第二铲尚无新满斗率；旧两小时预算关闭。

- [CPU报告](cpu-v1/report.json)
- [最低点保护候选](minimum_guard.py)、[模式保护候选](rise_hold_guard.py)
- [可复跑检查器](check_cpu.py)、[权威记录更新回执](record-receipt.json)

候选尚未接入物理运行器；后续需新授权范围与预算，从合格首铲返回来源重新做原生恢复、上抬及后续接近/第二铲验证，不自动从失败状态续跑。59微米是逐帧回落容差，不是累计下降额度。
''')
print(json.dumps({'status':'D191_recorded','files':list(texts),'cpu_checks':32,'physics_steps':0},ensure_ascii=False))
