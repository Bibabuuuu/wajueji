"""Make a labelled 2x derivative and audit the completed cache playback."""
from pathlib import Path
import ast, csv, hashlib, json, subprocess
R=Path('/home/qunwang/wajueji');O=R/'runs/roadshow-two-scoop-media-20261015-v1';W=R/'work/two-scoop-roadshow-media-20261015-v1'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def run(args):return subprocess.run(args,check=True,capture_output=True,text=True).stdout
checks=[]
def check(name,value):
 checks.append(dict(check=name,passed=bool(value)));assert value,name
data=read(O/'data-manifest.json');render=read(O/'video-render.json');master=O/render['movie'];fast=O/'video/two-scoop-presentation-2x.mp4'
check('render completed',render['status']=='rendered')
check('master sha',sha(master)==render['movie_sha256'])
check('prepared data unchanged',sha(O/'data-manifest.json')==render['data_manifest_sha256'])
check('renderer unchanged',sha(W/'render_video.py')==render['renderer_sha256'])
for p,digest in data['sources_sha256'].items():check('source sha '+p,sha(p)==digest)
for root,files in data['cache_identities'].items():
 for name,identity in files.items():
  stat=(Path(root)/'display'/name).stat();check('cache identity '+root+'/'+name,(stat.st_size,stat.st_mtime_ns)==(identity['bytes'],identity['mtime_ns']))
rows=list(csv.DictReader((O/'video/frame-map.csv').open()));last=data['workflow_last_frame'];cut=data['first_return_frame']
indices=list(range(0,last+1,2));indices+=[] if indices[-1]==last else [last]
expected=[(g,'cache_frame') for g in indices]+[(last,'result_freeze')]*90
check('full encoded schedule',len(rows)==len(expected)==9539)
phases=[read(Path(x)/'display/phases.json') for x in [data['first_run'],data['second_run']]]
check('every frame map exact',all(int(x['video_frame'])==i and int(x['workflow_frame'])==g and x['kind']==kind and int(x['source_run'])==(1 if g<=cut else 2) and int(x['source_frame'])==(g if g<=cut else g-cut) and abs(float(x['workflow_s'])-g/60)<1e-10 and x['phase']==phases[0 if g<=cut else 1][g if g<=cut else g-cut] for i,(x,(g,kind)) in enumerate(zip(rows,expected))))
check('all workflow phases visible',set(x['phase'] for x in rows)==set(phases[0][:cut+1])|set(phases[1]))
for x in render['keyframes']:check('keyframe sha '+x['file'],sha(O/x['file'])==x['sha256'])
tree=ast.parse((W/'render_video.py').read_text());calls=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
check('no simulation or learning calls',not set(calls)&{'step','predict','reset','restore','train','learn','CycleRuntime','SimulationRuntime'})
check('zero render physics inference training',all(render[k]==0 for k in ['playback_physics_steps','policy_inference_calls','training_updates']))
vf="setpts=0.5*PTS,fps=30,drawbox=x=1710:y=124:w=170:h=55:color=0x101e2c:t=fill,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='2x':x=1756:y=131:fontsize=36:fontcolor=white"
if not fast.exists():run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(master),'-an','-vf',vf,'-c:v','libx264','-threads','4','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(fast)])
movies={}
for path in [master,fast]:
 probe=json.loads(run(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames:format=duration','-of','json',str(path)]));stream=probe['streams'][0];duration=float(probe['format']['duration'])
 check(path.name+'1080p30H264',stream['codec_name']=='h264' and stream['width']==1920 and stream['height']==1080 and stream['pix_fmt']=='yuv420p' and stream['r_frame_rate']=='30/1')
 frames=int(stream['nb_read_frames']);check(path.name+'frames duration',abs(duration-frames/30)<.002)
 run(['ffmpeg','-hide_banner','-loglevel','error','-xerror','-i',str(path),'-f','null','-']);check(path.name+'entire decode',True)
 movies[path.name]=dict(file=str(path.relative_to(O)),sha256=sha(path),bytes=path.stat().st_size,frames=frames,duration_s=duration,probe=probe)
check('master schedule equals decoder',movies[master.name]['frames']==len(rows))
check('2x duration',abs(movies[fast.name]['duration_s']*2-movies[master.name]['duration_s'])<=2/30+.002)
q=O/'video/encoded-review';q.mkdir(exist_ok=True)
for name,index in [('first-cut',1497),('unloaded',3274),('seam',5252),('second-cut',9078),('result-card',9500)]:
 run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(master),'-vf',f'select=eq(n\\,{index})','-vsync','0','-frames:v','1',str(q/(name+'.png'))])
run(['ffmpeg','-hide_banner','-loglevel','error','-y','-ss','155','-i',str(fast),'-frames:v','1',str(q/'2x-card.png')])
report=dict(status='media_audit_passed',checks=checks,checks_passed=len(checks),movies=movies,derivative_filter=vf,derivative_semantics='Entire master timeline at2x, resampled30fps, explicit2x label; no stage removed. Final result-freeze also plays at2x.',physics_steps=0,inference_calls=0,training_updates=0,source_boundary_frame=cut,workflow_last_frame=last,continuity=render['continuity'])
(O/'media-audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print('MEDIA_AUDIT_PASSED',len(checks),flush=True)
