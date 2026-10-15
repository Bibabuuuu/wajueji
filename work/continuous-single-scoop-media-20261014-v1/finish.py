"""Audit the encoded media, package it, and update existing project records."""
from pathlib import Path
import ast,csv,hashlib,json,shutil,subprocess,time,zipfile,re
from PIL import Image
ROOT=Path('/home/qunwang/wajueji');RUN=ROOT/'runs/continuous-single-scoop-20261014-v1'
W=ROOT/'work/continuous-single-scoop-media-20261014-v1';OUT=ROOT/'runs/roadshow-handover-20261012/stage-04'
def read(p):return json.loads(Path(p).read_text())
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
started=time.monotonic();media=read(OUT/'render-review.json');report=read(RUN/'report.json');audit=read(RUN/'read-only-audit.json')
movie=OUT/media['movie']['file'];mapping=list(csv.DictReader((OUT/'video-frames.csv').open()));phases=read(RUN/'display/phases.json')
indices=[int(x['source_frame']) for x in mapping];expected=list(range(0,13754,6))+[13753]
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration','-of','json',str(movie)],text=True))['streams'][0]
decode=subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(movie),'-f','null','-'],capture_output=True,text=True,timeout=90)
calls=[n.func for n in ast.walk(ast.parse((W/'render.py').read_text())) if isinstance(n,ast.Call)]
protocol=read(ROOT/'work/continuous-single-scoop-20261014-v1/protocol.json')
checks=dict(render_completed=media['status']=='rendered',source_completed=report['status']=='continuous_single_scoop_completed',
 source_audit_56_passed=audit['status']=='passed' and audit['check_count']==56 and all(audit['checks'].values()),
 source_report_unchanged=sha(RUN/'report.json')==media['source_report_sha256'],source_audit_unchanged=sha(RUN/'read-only-audit.json')==media['source_audit_sha256'],
 renderer_identity=sha(W/'render.py')==media['renderer_sha256'],source_code_unchanged=all(sha(k)==v for k,v in protocol['code_sha256'].items()),
 no_new_physics_inference_or_training=media['playback_physics_steps']==media['policy_inference_calls']==media['training_updates']==0,
 no_step_reset_or_model_calls=not any((isinstance(f,ast.Attribute) and f.attr in ('step','predict','reset','restore')) or (isinstance(f,ast.Name) and f.id=='CycleRuntime') for f in calls),
 original_physical_budget_closed=Path(str(RUN)+'-budget.json.closed.json').exists() and sha(protocol['budget_path'])==report['budget_sha256'],
 all_cache_hashes_unchanged=all(sha(RUN/'display'/k)==d['sha256'] for k,d in media['cache_identities'].items()),
 video_h264_yuv420=probe['codec_name']=='h264' and probe['pix_fmt']=='yuv420p',video_full_frame_dimensions=(probe['width'],probe['height'])==(1440,900),
 video_30fps=probe['r_frame_rate']=='30/1',video_2294_frames=int(probe['nb_read_frames'])==2294==len(mapping)==media['movie']['frames'],
 video_duration=abs(float(probe['duration'])-2294/30)<1e-5,full_video_decodes=decode.returncode==0 and not decode.stderr.strip(),
 video_hash=sha(movie)==media['movie']['sha256'],regular_continuous_frame_mapping=indices==expected,
 all_video_phase_labels_match=all(x['phase']==phases[int(x['source_frame'])] for x in mapping),
 all_video_timestamps_match=all(float(x['simulation_time_s'])==int(x['source_frame'])/60 for x in mapping),
 all_source_phases_represented=set(x['phase'] for x in mapping)==set(phases),
 six_keyframe_hashes=len(media['keyframes'])==6 and all(sha(OUT/x['file'])==x['sha256'] for x in media['keyframes']),
 six_keyframe_dimensions=all(Image.open(OUT/x['file']).size==(1440,900) for x in media['keyframes']),
 six_keyframe_phase_identity=all(x['phase']==phases[x['frame']] for x in media['keyframes']),
 gpu_idle=not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip())
result=dict(status='passed' if all(checks.values()) else 'failed',check_count=len(checks),checks=checks,ffprobe=probe,
 decode_stderr=decode.stderr,visual_review=dict(frames=[x['frame'] for x in media['keyframes']],reviewer='Codex',result='All six smoke images inspected; machine, truck and Chinese labels complete.'),
 new_physical_frames=0,policy_inference_calls=0,training_updates=0,wall_seconds=time.monotonic()-started)
write(OUT/'media-audit.json',result);assert all(checks.values()),[k for k,v in checks.items() if not v]
body='''# Stage 04：冻结残差完整单铲视频与关键帧

已导出并验证。用户接受本次保载显示的小幅下降，优先完成PPT素材，暂不继续差异诊断、训练或物理实验。

## 可直接使用

- `videos/continuous-single-scoop-3x.mp4`：76.47秒，1440×900、H.264、30fps、名义3倍速。完整覆盖同一原生物理实例的0至13753帧：真实准备回退→自动接近→冻结SAC残差挖掘及0.5秒保载→几何举升/回转/卸料→退出→安全高位返回。
- `screenshots/`：6张同源关键帧，自动接近、切入、挖掘收斗、离土保载、卸料结束、返回完成。均标实际阶段、流程时间和全局帧号；`keyframes.csv`为对应表。
- `video-frames.csv`：2294帧视频逐帧映射到原物理帧号，均匀每6物理帧采样，末尾另含真实13753帧。没有跨运行拼接或改变播放中途速度。
- `source-report.json`、`source-audit.json`：原连续单铲运行与56项审计。`media-audit.json`额外核对源、编码、解码、映射及截图；`source-snapshot/`为本次离线导出源码。

PPT按比例嵌入，保留1440×900原画幅，不拉伸到16:9。若只需一页演示可放视频；六关键帧适合流程页或答辩备份。起点是真实准备回退，字幕已区分，不冒充更早旧格式初始点直接续跑。

## 结果口径

本视频来自`runs/continuous-single-scoop-20261014-v1`，当前固定铲位第一铲、同实例13753帧（229.2167仿真秒），挖掘段最低短保载占据显示142.6193%，含土石/堆高且未标定；未重新训练。只证明本次单铲流程完成，不授予三铲或跨铲位泛化。

Stage 02视频与Stage 03三方主对照中的148.2321%属于独立预备点恢复评价，不能贴到本次连续视频上。原三方主表、模型、参考轨迹、原视频和缓存保持；不把本次接受下降改写成性能不变。

PPT可写：**冻结残差策略已接入完整单铲流程，挖掘由人工参考上的四轴与节奏残差控制，运输卸料及返回由既有几何反馈接管。**

## 来源与验证

原始缓存、物理/控制日志和检查点保留在台式机源运行目录。媒体只是离线渲染原缓存，新增物理步、策略推理、训练更新均为0；旧实验预算仍关闭。完整缓存SHA、源报告/审计及渲染源码SHA见render-review.json。媒体全片解码、2294帧映射和6张截图身份已核对，六帧画面已目视检查。

后续PPT素材可复用Stage 02动作/进度/选点图与Stage 03对比图。本次补的是完整流程演示；整体技术路线图及最终页面排版仍由PPT制作环节完成。
'''
(OUT/'README.md').write_text(body)
for n,source in [('source-report.json',RUN/'report.json'),('source-audit.json',RUN/'read-only-audit.json')]:shutil.copyfile(source,OUT/n)
snapshot=OUT/'source-snapshot';snapshot.mkdir(exist_ok=False)
for n in ['render.py','finish.py']:shutil.copyfile(W/n,snapshot/n)
brief=f'''## 2026-10-14 连续单铲PPT媒体：已导出并核验

用户明确接受本次142.6193%短保载显示，下降不作为当前阻塞项；差异诊断及复训降低优先级，先完善PPT流程演示。Stage 04已导出76.47秒3倍速完整单铲MP4与6关键帧（1440×900），均来自D-182同实例13753帧缓存；保留起点真实回退、接近、残差挖掘、几何运卸和安全返回，非跨运行拼接。{len(checks)}项媒体核验通过，含全片解码、2294视频帧映射、各阶段覆盖、源/代码/缓存SHA和GPU空闲；六张短检图已目视检查。新增物理/策略推理/训练均0，旧预算仍关闭。

原独立挖掘148.2321%与三方主表继续有效于其原范围，不贴到本次142.6193%画面。仅当前铲位第一铲，三铲及跨位置泛化未新增资格。素材runs/roadshow-handover-20261012/stage-04/，Windows deliverables/roadshow-continuous-stage04-20261014/；源运行、模型、正式控制/物理及原缓存未改，未Git提交。下一步PPT优先接入完整视频/关键帧，再完善方法路线图和结果页口径；实际装载量标定继续低优先级。

'''
pattern=r'^## 2026-10-14 连续单铲PPT媒体：正在导出\n.*?(?=^## |\Z)'
for rel in ['docs/CURRENT_STATUS.md','docs/plans/S07-truck-reproduction-submission.md']:
 p=ROOT/rel;s=p.read_text();assert re.search(pattern,s,re.M|re.S);p.write_text(re.sub(pattern,lambda _:brief,s,count=1,flags=re.M|re.S))
for rel in ['docs/PROJECT_PLAN.md','docs/roadshow/PPT_DATA_REQUEST.md']:
 p=ROOT/rel;s=p.read_text();pos=s.index('\n')+1;p.write_text(s[:pos]+'\n'+brief+s[pos:])
write(OUT/'manifest.json',{p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and 'qa-smoke' not in p.parts})
archive=ROOT/'runs/roadshow-continuous-stage04-20261014.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for n in read(OUT/'manifest.json'):
  z.write(OUT/n,'roadshow-continuous-stage04-20261014/'+n)
 z.write(OUT/'manifest.json','roadshow-continuous-stage04-20261014/manifest.json')
print(json.dumps(dict(status='packaged',checks=len(checks),archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),ensure_ascii=False),flush=True)
