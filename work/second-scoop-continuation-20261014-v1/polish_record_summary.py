from pathlib import Path
import hashlib,json
R=Path('/home/qunwang/wajueji');W=R/'work/second-scoop-continuation-20261014-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((W/'adapter-record-update-receipt.json').read_text())
n='docs/PROJECT_PLAN.md';p=R/n
assert sha(p)==receipt[n]['after']
s=p.read_text()
changes={
'第二铲CPU接通，参考输入适配已授权':'第二铲CPU几何与观测适配已验证，物理待准入',
'用户已选择A转向迁移，授权观测转换接入与CPU验证；':'用户已选择A转向迁移，观测转换已实现，105项CPU检查及17项独立适配审计通过；',
}
for a,b in changes.items():
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b,1)
(W/'project-summary-before-polish.md').write_bytes(p.read_bytes())
p.write_text(s)
receipt[n]['after']=sha(p)
(W/'adapter-record-update-receipt.json').write_text(json.dumps(receipt,indent=2))
print('Project summary consistent with completed CPU adapter.')
