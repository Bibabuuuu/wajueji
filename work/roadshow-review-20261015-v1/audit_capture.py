from pathlib import Path
import json,csv,hashlib,subprocess
import numpy as np
R=Path('/home/qunwang/wajueji');O=R/'runs/roadshow-review-20261015-v1';W=R/'work/roadshow-review-20261015-v1';V=R/'runs/loaded-closedloop-20261015-v5';B=R/'runs/motion-physical-short-20261015-v2/loaded'
read=lambda p:json.loads(Path(p).read_text())
states=read(O/'display/states.json');expected=read(V/'states.json');e=read(O/'pitch-errors.json');old=read(V/'pitch-errors.json')[:764];report=read(O/'report.json');checks=[]
def check(name,value):checks.append(dict(name=name,pass_=bool(value)));assert value,name
check('fresh_complete',report['status']=='completed' and report['physics_steps']==767 and not report['old_budget_reused'])
check('768_native_states_exact_verified_closedloop_v5',states==expected[:768])
check('764_new_errors_exact_v5',[(x['frame'],x['pitch_error_deg']) for x in e]==[(x['frame'],x['pitch_error_deg']) for x in old])
c=np.load(O/'display/particles.npy',mmap_mode='r');b=np.load(B/'display/particles.npy',mmap_mode='r')
check('384_real_samples',c.shape==(384,*b.shape[1:]))
check('initial_particles_exact',np.array_equal(c[0],b[0]));check('second_sample_prefix_exact',np.array_equal(c[1],b[2]))
with np.load(V/'active120.npz') as z:
 key=next(k for k in z.files if k=='particle_q');check('frame120_particles_exact_original_snapshot',np.array_equal(c[60],z[key]))
rows=[json.loads(x) for x in (O/'physics/frames.ndjson').read_text().splitlines()];print('FRAME_KEYS',list(rows[0]))
check('all_physics_frames_recorded',len(rows)==767)
for key in ['hard_guard']:
 check(key+'_all_frames',all(x[key] for x in rows))
for key in ['hencky_material_guard','moving_bucket_sweep']:check(key+'_all_frames',all(x[key]['safe'] for x in rows))
with (W/'fresh-control.csv').open('w',newline='') as f:
 writer=csv.writer(f);writer.writerow(['frame','native_time_s','lift_relative_s','pitch_error_before_deg','q_drive1','q_drive2','q_drive3','q_drive4','applied1','applied2','applied3','applied4','hard_guard','material_guard','sweep_guard'])
 errors={x['frame']:x['pitch_error_deg'] for x in e}
 for row in rows:
  s=row['after_state'];q=s['mechanism_qpos'];a=row['applied_command_si'];assert len(a)==4
  writer.writerow([row['frame'],s['time_s'],(row['frame']-3)/60,errors.get(row['frame']),q[7],q[9],q[12],q[15],*a,row['hard_guard'],row['hencky_material_guard']['safe'],row['moving_bucket_sweep']['safe']])
with (W/'fresh-control.csv').open() as f:
 table=list(csv.reader(f));check('control_csv_767_rows_15_columns',len(table)==768 and all(len(row)==15 for row in table))
for name,frames,duration in [('oscillation-before-after-1x.mp4',382,382/30),('fixed-loaded-lift-1x.mp4',382,382/30),('two-digs-highlights-2x.mp4',1845,61.5)]:
 file=O/'video'/name;p=read(O/'video/render-report.json')
 stream=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_streams','-of','json',str(file)]))['streams'][0]
 check(name+'_frames',int(stream['nb_frames'])==frames);check(name+'_duration',abs(float(stream['duration'])-duration)<1e-4);check(name+'_h264',stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p')
 subprocess.run(['ffmpeg','-v','error','-i',str(file),'-f','null','-'],check=True)
 check(name+'_full_decode',True)
result=dict(status='passed',checks=checks,physics_steps=0,training_updates=0,scope='fresh lift capture matches verified closed-loop v5 exactly; no full updated two-scoop rerun',fresh_physics_frames=767)
(W/'independent-capture-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('AUDIT_PASSED',len(checks))
