from pathlib import Path
import json,hashlib,shutil,re,datetime
R=Path('/home/qunwang/wajueji');W=R/'work/roadshow-review-20261015-v1';B=W/'docs-before';B.mkdir(exist_ok=True)
files=['docs/PROJECT_PLAN.md','docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md','docs/roadshow/PPT_DATA_REQUEST.md','docs/handoff.md','docs/acceptance.md','docs/plans/INDEX.md']
baseline={};texts={}
for name in files:
 p=R/name;baseline[name]=hashlib.sha256(p.read_bytes()).hexdigest();texts[name]=p.read_text();dest=B/name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.copy2(p,dest)
title='## 2026-10-15 路演答辩精修与减振同步视频：已实现、已验证（所列范围）'
brief='最新展示交付为Windows deliverables/roadshow-defense-ready-20261015/，25页（17主讲＋8隐藏备用），已删除页脚来源/证据及内部备用页提示；第6页加强真实终态、60Hz轨迹和三次保载量测，第8页嵌入12.733秒上下减振对照，第13页为61.5秒两次实际挖掘精剪。新增767物理帧已独立核验，完整修复后两铲链路未重跑，S05/实机/泛化资格不升级；详情统一见[S07](plans/S07-truck-reproduction-submission.md)顶部。'
detail='''用户明确要求删去全部底部来源/证据与“详见备用页”等说明，加强第6页数据表达，并重采抖动段制作上旧下新的短对照；沿用用户后续自主推进授权。保留25页顺序、17主讲＋8隐藏、主题与原生流程图，改6/8/11/13/16讲稿并清理2/15页内部叙述提示。新PPT为Windows deliverables/roadshow-defense-ready-20261015/自主挖掘路演-答辩精修与减振对比_20261015.pptx，28368260字节，SHA256 604a09f32f07a7e02d4330b05fadc9f7072407dcc81a68d5ea2b50a1260764e6；index.html、25页讲稿、原始CSV/量测、三段视频、三组PNG/SVG/PDF和复现脚本同包。

第6页仍是同完整初态匹配案例97.0699→148.2321%、26.7333→20.1000仿真秒，新增展示人工1605/自主1207个真实完整机构状态的高度轨迹，以及0/0.25/0.5秒全部六次短保载读数与同机位终态。保载三次采样不是三次独立试验；显示估计含堆高，可超过100%，未当作标定净体积。原7张未改图表及其工作簿按字节保留，改动3图为原生图表并逐点核对；3新工作簿原序列化精度差异已按源缓存原始全精度写回单元格，无源数据或图缓存修改。

新物理补采：台式RTX4070、work/roadshow-review-20261015-v1/collect_fixed.py，从原合格endpoint SHA256 46d3f04fa2dfc18082c17f7496e4c146f2491fdb8e08ccd0339cc6169203947a恢复，复用正式LoadedTransportRuntime和原TransportController逐帧重算，原控制窗口191.11666667079683—216.30000000413014原生秒、原保护不改。本轮独立600秒预算/120秒保存余量，实际767帧（前缀3＋完整举升764）、约157秒墙钟后关闭；无训练/策略推理，未修改正式控制源码。384个30Hz真实粒子采样与768状态已保存至runs/roadshow-review-20261015-v1/；768状态及764误差与v5逐项相同，初始/前缀粒子及第120帧原快照精确一致，所有原保护通过；24项独立CPU/视频解码核验通过。新视频frame4—766偶数帧，382画面/30fps=12.733333秒，上原控制下新控制，同机位同裁切正常速度，不做物理插值或运动平滑。峰值4.000785→1.913036°、RMS1.518199→0.572984°，仍保留较小起始响应。图表显示每帧绝对误差，带符号原始值同包。

第13页新61.5秒精剪来自原D192/D193的两次实际挖掘/短保载，省略中间运卸/返回，字幕和讲稿明确为片段。未把修复后举升拼成完整新两铲；原长视频/原始缓存未覆盖。完整修复后两铲、第三铲、实机标定和泛化均未新增资格。

验证：25页文字容量0错误0警告、25页渲染；三图字体/PDF碰撞检查通过；71项PPT内容/图表/媒体独立检查通过，最终结构、布局、字体、10原生图表/工作簿、导入检查通过。14个仅改页脚页正文上640像素与旧稿逐像素相同，预算页仅1像素宽图标抗锯齿差异；原背景装饰/隐藏页等建议保留。原生PowerPoint/WPS放映与点击播放未验证。失败的首次工作簿精度检查及保留图表修复前候选均在Windows私有工作区保留。

本轮证据：work/roadshow-review-20261015-v1/{protocol.json,closure.json,independent-capture-audit.json,fresh-control.csv}，runs/roadshow-review-20261015-v1/{report.json,display/,video/}；Windows .scratch/ppt-defense-clean-20261015-v2/qa/及validation/finalizer.json。旧600秒物理预算不重开；无A30任务、无Git提交，历史4个脏源码及D200正式源码保持不变。下一门如需更新完整链路视频，仍是LoadedTransportRuntime从举升/回转退出至卸料、返回和次铲的完整接入预检及新的独立有界采集，不复用本轮预算。'''
def insert(text,block):
 head,sep,tail=text.partition('\n');return head+'\n\n'+block+'\n\n'+tail.lstrip('\n')
for name in files:
 assert title not in texts[name]
 if name=='docs/plans/S07-truck-reproduction-submission.md':new=insert(texts[name],title+'\n\n'+detail)
 elif name=='docs/PROJECT_PLAN.md':
  text=texts[name]
  text=re.sub(r'\n路演交付更新（2026-10-15）：[^\n]*\n','\n',text,count=1)
  text=re.sub(r'\n最新展示交付（2026-10-15流程图优化）：[^\n]*\n','\n',text,count=1)
  new=insert(text,'路演展示更新（2026-10-15答辩精修）：'+brief)
 elif name=='docs/CURRENT_STATUS.md':
  text=re.sub(r'\n## 2026-10-15 最新路演：知识库流程图优化版\n\n.*?(?=\n## |\Z)','\n',texts[name],count=1,flags=re.S)
  new=insert(text,title+'\n\n'+brief+'\n\n本轮采集已完成并关闭，无正在运行的物理/训练任务。')
 elif name=='docs/roadshow/PPT_DATA_REQUEST.md':
  text=re.sub(r'\n## 2026-10-15 最新路演：知识库流程图优化版\n\n.*?(?=\n## |\Z)','\n',texts[name],count=1,flags=re.S)
  new=insert(text,title+'\n\n'+brief.replace('(plans/','(../plans/'))
 elif name=='docs/plans/INDEX.md':
  new=texts[name].replace('D192两铲149.96%/109.94%完成；D196返回短检45.87秒通过，减振未解决；预算已关闭，见handoff','D192两铲完成；D200单来源减振通过，新767帧回放补采通过；完整修复后两铲未重跑，见S07/handoff')
  new=new.replace('车斗物理、完整链、独立复现和9月25日交付','车斗物理、完整链、独立复现；当前路演PPT与减振同步短片交付，见S07顶部')
 elif name=='docs/acceptance.md':
  new=insert(texts[name],title+'\n\n新增767帧同源举升物理补采通过：768状态、764误差逐项复现v5；384粒子采样，原前缀与第120帧快照精确一致；24项独立核验和3视频完整解码通过。PPT71项核验、25页容量/渲染、10原生图表/工作簿通过。仅授予本次采集与展示资格；不升级完整新两铲、第三铲、实机或泛化。详见[S07](plans/S07-truck-reproduction-submission.md)顶部。')
 else:new=insert(texts[name],title+'\n\n'+brief)
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==baseline[name],name
 (R/name).write_text(new)
receipt=dict(status='updated',head='8ea6cd8f55c3c35c4be73054ce28ed1f815f21a0',files={n:dict(before=baseline[n],after=hashlib.sha256((R/n).read_bytes()).hexdigest()) for n in files},no_formal_source_changes=True,no_git_commit=True)
(W/'records-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2));print('UPDATED_EXISTING_RECORDS',len(files))
