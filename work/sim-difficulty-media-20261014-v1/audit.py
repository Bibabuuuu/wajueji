from pathlib import Path
import ast,hashlib,json,subprocess
R=Path('/home/qunwang/wajueji');S=R/'runs/continuous-single-scoop-20261014-v1';O=R/'runs/roadshow-handover-20261012/stage-05-sim-difficulties'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((O/'render-review.json').read_text());checks=[]
def check(name,ok):
 checks.append({'name':name,'passed':bool(ok)});assert ok,name
check('source report unchanged',sha(S/'report.json')==d['source_report_sha256'])
check('source audit unchanged',sha(S/'read-only-audit.json')==d['source_audit_sha256'])
for name,v in d['cache_identities'].items():
 p=S/'display'/name;check('cache identity '+name,p.stat().st_size==v['size'] and p.stat().st_mtime_ns==v['mtime_ns'])
for v in d['frames']:check('render digest '+v['file'],sha(O/v['file'])==v['sha256'])
code=R/'work/sim-difficulty-media-20261014-v1/render-final.py'
check('renderer unchanged',sha(code)==d['renderer_sha256'])
tree=ast.parse(code.read_text());calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
check('no step or train call',not any(s in calls for s in ['step','learn','train','predict']))
v=O/'workface-deformation-wide.mp4'
subprocess.run(['ffmpeg','-v','error','-i',str(v),'-f','null','-'],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration','-of','json',str(v)]))
p=probe['streams'][0];check('H264 1440x630 153 frames',p['codec_name']=='h264' and p['width']==1440 and p['height']==630 and p['nb_read_frames']=='153')
subprocess.run(['ffmpeg','-y','-v','error','-sseof','-0.05','-i',str(v),'-frames:v','1',str(O/'workface-end.png')],check=True)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=process_name,used_gpu_memory','--format=csv,noheader'],text=True).strip()
result={'status':'passed','checks':checks,'video_probe':probe,'video_sha256':sha(v),'gpu_compute_processes_after':gpu,'physics_steps_added':0,'training_updates':0,'policy_inference_calls':0}
(O/'independent-media-audit.json').write_text(json.dumps(result,indent=2))
print('PASS',len(checks),'media checks; GPU processes:',gpu or 'none')
