"""Record completed media delivery in existing authority documents only."""
from pathlib import Path
import hashlib,json,subprocess
R=Path('/home/qunwang/wajueji');O=R/'runs/roadshow-two-scoop-media-20261015-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((O/'media-audit.json').read_text());assert audit['status']=='media_audit_passed' and audit['checks_passed']==47
assert (O/'index.html').is_file() and (O/'figures/qa-summary.json').is_file()
heading='## 2026-10-15 D-193：两铲数据图与连续视频已交付'
body='''
用户授权仅利用已有数据制作图表和连续视频。已导出6张数据图（300dpi PNG、可编辑文字SVG、嵌入字体PDF）、6张1080p原始关键帧及组合图，7378条完整动作CSV、6条保载样本、流程阶段、三时点地形观测与真实FK路径。六图文字/碰撞/面板对齐检查通过并逐张目视复核；300dpi和宽图尺寸按PPT用途接受。

连续视频覆盖边缘首铲→实际运卸返回→次铲接近→完整第二铲保载：1920×1080、30fps、H.264；1倍速317.966667秒、9539帧，2倍速159秒、4770帧，明确标注2x。47项媒体核验通过，含两版全片解码、所有阶段覆盖、逐帧来源映射、原数据/代码/缓存身份。A合格返回前缀0–10504与B1–8391严格续接，土堆不重置；快照续接处已标注，不能表述为从初态起单实例不中断。原1微米失败分支不进入成功组合，失败报告及缓存保留。

两铲短保载显示149.96%/109.94%，各完成1170参考及0.5秒保载；历史中央对照仅作趋势参考，首铲等待接口不同，不授予纯布局因果、三铲或泛化成功率；未标定装载体积，次铲运输未执行。此次新增物理/策略推理/训练均0，GPU媒体任务已结束、无计算进程，所有旧实验预算仍关闭。未修改PPT本体，无Git提交。

入口：`runs/roadshow-two-scoop-media-20261015-v1/index.html`、`README.md`、`media-audit.json`、`data-manifest.json`、`MANIFEST.sha256`；Windows为`deliverables/roadshow-two-scoop-media-20261015/`及同名ZIP。下一步可将图01/05及2倍速完整视频接入现有PPT；S05仍进行中，其余验收状态不自动升级。
'''
files=['docs/PROJECT_PLAN.md','docs/CURRENT_STATUS.md','docs/plans/S05-imitation-and-rl.md','docs/acceptance.md','docs/decisions.md','docs/roadshow/PPT_DATA_REQUEST.md']
receipt=[]
for rel in files:
 p=R/rel;text=p.read_text();assert heading not in text,rel
 before=sha(p);first,sep,rest=text.partition('\n');text=first+'\n\n'+heading+'\n'+body+'\n'+rest
 if rel=='docs/decisions.md':
  index='## 当前决策索引（2026-10-15）';assert text.count(index)==1
  text=text.replace(index,index+'\n\n- `D-193`：已有缓存制作两铲6图/关键帧/1080p连续视频；1倍速317.967秒、2倍速159秒，47项媒体检查通过，物理/推理/训练均0。',1)
 if rel=='docs/PROJECT_PLAN.md':
  old='当前实施S05：原第二铲完整挖掘与保载已验证；按用户要求先试边缘首铲＋向内贴边次铲。布局CPU候选通过，新物理预算待确认；三铲未完成，新训练未启动。详见S05和CURRENT_STATUS。'
  new='当前实施S05：边缘首铲＋向内贴边次铲均完成1170参考及0.5秒保载，显示149.96%/109.94%；通过合格原生快照保持土堆连续续接。D-193六图与两版视频已交付，当前无物理/训练任务，旧预算关闭。三铲未完成；下一步可接入PPT，详见S05和CURRENT_STATUS。'
  assert old in text;text=text.replace(old,new,1)
 p.write_text(text);receipt.append(dict(file=rel,before_sha256=before,after_sha256=sha(p)))
check=subprocess.run(['git','diff','--check','--',*files],cwd=R,capture_output=True,text=True);assert check.returncode==0,check.stdout+check.stderr
report=dict(status='authority_records_updated',files=receipt,git_diff_check=True,git_commit_created=False,physical_budget_reopened=False)
(R/'work/two-scoop-roadshow-media-20261015-v1/records-receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('RECORDS_UPDATED',len(receipt))
