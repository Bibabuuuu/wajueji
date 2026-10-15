"""Read audited caches and logs; no native runtime entry, inference or stepping."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import csv, hashlib, importlib.util, json, sys, time
import numpy as np

R=Path('/home/qunwang/wajueji');W=R/'work/two-scoop-roadshow-media-20261015-v1';O=R/'runs/roadshow-two-scoop-media-20261015-v1'
A=R/'runs/edge-first-two-scoop-20261015-continuation';B=R/'runs/edge-adjacent-59um-20261015-v1'
read=lambda p:json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def csvfile(p,rows):
 with Path(p).open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
 start=time.monotonic();O.mkdir(exist_ok=False);D=O/'data';D.mkdir();checks={};sources={}
 for folder in [A,B]:
  for name in ['report.json','read-only-audit.json','display/reconstruction.json','display/states.json','display/phases.json','display/scene.json']:
   sources[str(folder/name)]=sha(folder/name)
 pa,pb=read(A/'report.json'),read(B/'report.json');aa,ab=read(A/'read-only-audit.json'),read(B/'read-only-audit.json')
 assert aa['record_integrity_passed'] and ab['record_integrity_passed'] and ab['check_count']==47 and ab['two_scoop_goal_complete']
 cutoff=read(A/'first-returned.control.json')['frame'];assert cutoff==10504 and pa['first_return_checkpoint']['sha256']==pb['source_sha256']
 states=[read(x/'display/states.json') for x in [A,B]];phases=[read(x/'display/phases.json') for x in [A,B]]
 clouds=[np.load(x/'display/particles.npy',mmap_mode='r') for x in [A,B]]
 assert states[0][cutoff]==states[1][0] and np.array_equal(clouds[0][cutoff],clouds[1][0])
 identities={}
 for folder in [A,B]:
  meta=read(folder/'display/reconstruction.json');identities[str(folder)]={}
  for name,h in meta['cache_sha256'].items():
   file=folder/'display'/name;assert sha(file)==h,('cache_sha',file)
   identities[str(folder)][name]=dict(sha256=h,bytes=file.stat().st_size,mtime_ns=file.stat().st_mtime_ns)
  print('CACHE_VERIFIED',folder.name,flush=True)
 total=cutoff+pb['frames'];fullphases=phases[0][:cutoff+1]+phases[1][1:]
 assert len(fullphases)==total+1
 segments=[];begin=1;last=fullphases[1]
 for g in range(2,total+2):
  label=fullphases[g] if g<=total else None
  if label!=last:
   segments.append(dict(phase=last,start_frame=begin-1,end_frame=g-1,start_s=(begin-1)/60,end_s=(g-1)/60,duration_s=(g-begin)/60,source_run=('first' if begin<=cutoff else 'second')))
   begin=g;last=label
 csvfile(D/'workflow.csv',segments)
 rows=[];samples=[];modes={}
 for i,folder in enumerate([A,B],1):
  episode=folder/f'policy-{i}/episode-0001';report=read(episode/'report.json');offset=report['actual_entry_global_frame']
  acts=[json.loads(line) for line in (episode/'actions.ndjson').read_text().splitlines()]
  assert len(acts)==[3857,3521][i-1] and report['success'] and report['cursor']==1170
  for x in acts:
   f=x['frame'];g=offset+f-1;state=states[i-1][g];q=np.asarray(state['mechanism_qpos'])[[7,9,12,15]]
   error=np.asarray(x['alignment']['actual_joint_error_si']);ref=q+error
   row=dict(scoop=i,local_frame=f,time_s=(f-1)/60,source_frame=g,workflow_frame=g+(cutoff if i==2 else 0),
            reference_progress=x['cursor_before'],actual_progress=x['alignment']['actual_progress'],mode=('hold' if x['hold_frame'] is not None else x['execution_mode']),invalidated=x['action_invalidated'])
   for j,name in enumerate(['swing','boom','stick','bucket']):
    row[f'actual_{name}_si']=float(q[j]);row[f'reference_{name}_si']=float(ref[j]);row[f'error_{name}_si']=float(-error[j]);row[f'command_{name}_si']=x['applied_si'][j]
   rows.append(row)
  for x in report['scores']:
   samples.append(dict(scoop=i,local_frame=x['frame'],time_s=x['frame']/60,workflow_frame=offset+x['frame']+(cutoff if i==2 else 0),fill_percent=x['fill_percent'],valid=x['valid_estimate']))
  modes[str(i)]={key:sum(row['mode']==key for row in rows if row['scoop']==i) for key in ['follow','wait','hold']}
  sources[str(episode/'report.json')]=sha(episode/'report.json');sources[str(episode/'actions.ndjson')]=sha(episode/'actions.ndjson')
 csvfile(D/'dig-timeseries.csv',rows);csvfile(D/'hold-samples.csv',samples)
 assert len(samples)==6 and all(x['valid'] for x in samples)
 summary=read(B/'summary/results.json');write(D/'results.json',summary)
 (D/'actual-mouth-paths.json').write_bytes((B/'summary/actual-mouth-paths.json').read_bytes())
 for label,path in [('before-first',A/'scoop-1-plan/terrain.npz'),('after-first',B/'scoop-2-plan/terrain.npz')]:
  assert path.is_file();(D/(label+'.npz')).write_bytes(path.read_bytes());sources[str(path)]=sha(path)
 final_terrain=dict(status='unavailable')
 try:
  sys.path.insert(0,str(R/'work/edge-adjacent-59um-20261015-v1'));import run_second as m
  snapshot=Path(pb['final_checkpoint']['path']);runtime=m.CpuRuntime(snapshot,W/'terrain-final-cpu',enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True)
  spec=importlib.util.spec_from_file_location('readonly_final_stock',R/'work/contact-fixed-truck-continuation-20260925-v2/observation.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  terrain,observation=mod.stationary_terrain(runtime,runtime.arrays)
  np.savez(D/'after-second.npz',**{k:np.asarray(v) for k,v in terrain.items() if isinstance(v,(list,np.ndarray,int,float))})
  final_terrain=dict(status='ground_connected_observation_passed',metadata=observation,source_sha256=sha(snapshot),new_physics_steps=0)
 except ValueError as exc:final_terrain=dict(status='observation_rejected',reason=str(exc),new_physics_steps=0)
 write(D/'final-terrain-observation.json',final_terrain)
 manifest=dict(status='source_data_prepared',sources_sha256=sources,cache_identities=identities,
    first_run=str(A),second_run=str(B),first_return_frame=cutoff,second_last_frame=pb['frames'],workflow_last_frame=total,workflow_seconds=total/60,
    boundary=dict(particles_exact=True,compact_state_exact=True,new_native_zero24_and_original3_verified=True),
    first_scoop=summary['new_scoops'][0],second_scoop=summary['new_scoops'][1],modes=modes,
    rows=len(rows),hold_sample_count=len(samples),phase_segments=segments,final_terrain=final_terrain['status'],
    scope='Validated first dig/unload/return up to completed phase; then verified restored59um branch. Prior failed1um rise branch is excluded from the main workflow and preserved in original records.',
    physics_steps=0,policy_predictions=0,training_updates=0,wall_seconds=time.monotonic()-start)
 write(O/'data-manifest.json',manifest);print('DATA_READY',len(rows),'rows; workflow',total/60,'s',flush=True)
if __name__=='__main__':main()
