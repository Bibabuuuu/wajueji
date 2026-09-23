"""Bounded loaded-state profile; 36 zero-command frames, never training evidence."""
import argparse, json, hashlib, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import audit_layered_47_truck_physics_gate as gate

root=Path('/home/qunwang/wajueji')
work=Path(__file__).resolve().parent
bundle=Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
kin=Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
loaded=root/'runs/layered-47-loaded-handoff-v1/d129-loaded-handoff-20260923-v2'
parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('--gate',action='store_true');cli=parser.parse_args()
args=SimpleNamespace(output=str(work/cli.name),repository_root=str(bundle/'repo'),truck_repository_root=str(root),
 config=str(bundle/'repo/configs/v4/layered-47-excavation-preflight-v1.json'),
 local_source_config=str(root/'configs/v4/layered-47-interactive-proxy-benchmark-v1.json'),
 checkpoint=str(loaded/'loaded-handoff.npz'),metadata=str(loaded/'loaded-handoff.npz.json'),
 loaded_audit=str(loaded/'audit-v4/independent-restore-audit.json'),
 approach_report=str(root/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089/report.json'),
 truck_config=str(root/'configs/v4/layered-47-truck-physics-v1.json'),device='cuda:0',
 mujoco_include=str(bundle/'mujoco/include'),mujoco_library_dir=str(bundle/'mujoco/lib'),
 mechanism_header_dir=str(bundle/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),
 controller_header_dir=str(bundle/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'),
 kinematics_python_dir=str(kin),kinematics_model=str(kin/'scene/input/model/A.xml'),kinematics_mesh_dir=str(kin/'scene/input/mesh'),
 inventory_contract='/home/qunwang/codex-work/heap-fill-audit-2026-09-11/inventory-contract.json',
 front_rim='/home/qunwang/codex-work/heap-fill-audit-2026-09-11/front-rim.npz',cxx='g++')
frames=[]; state={}
original_restore=gate.motion.audit.restore_candidate_runtime

def restore(*a,**kw):
 fixture,held,result=original_restore(*a,**kw)
 state['fixture']=fixture
 return fixture,held,result

gate.motion.audit.restore_candidate_runtime=restore
original_frame=gate.motion.probe.bench._advance_frame

def advance(*a,**kw):
 row,wall=original_frame(*a,**kw)
 frames.append(row)
 print('PROFILE_FRAME',row['frame'],round(wall,4),flush=True)
 return row,wall

gate.motion.probe.bench._advance_frame=advance
if not cli.gate:
 original_load=gate.truck.load_candidate
 def candidate(*a,**kw):
  doc,fixtures=original_load(*a,**kw)
  doc['acceptance']['hold_frames']=36
  return doc,fixtures
 gate.truck.load_candidate=candidate
bindings={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(args.checkpoint),Path(args.metadata),Path(args.config),Path(args.truck_config),root/'src/v4/run_layered_47_excavation_preflight.py',root/'src/v4/mpm_static_contact_ownership.py']}
report=gate.run(args)
if not cli.gate:
 report['schema']='layered_47_loaded_zero_command_profile_v1'
 report['status']='profile_recorded' if len(frames)==36 and report.get('failure') is None else 'profile_incomplete'
 report['scope']={key:False for key in report['scope']}
 report['protocol']['hold_frames']=36
 report['protocol']['qualification']=False
 (Path(args.output)/'truck-physics-gate.json').write_text(json.dumps(report,indent=2)+'\n')
selected=frames[6:] if not cli.gate else frames
summary={'status':report['status'],'failure':report.get('failure'),'frames':len(frames),'warmup_excluded':0 if cli.gate else 6,
 'measured_frames':len(selected),'bindings':bindings,'unchanged_inputs':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in bindings.items()),
 'components_seconds':{key:sum(r['mpm_step_components_wall_seconds'][key] for r in selected) for key in selected[0]['mpm_step_components_wall_seconds']} if selected else {},
 'timing_seconds':{key:sum(r[key] for r in selected) for key in ['wall_seconds','mpm_wall_seconds','mujoco_wall_seconds','assignment_wall_seconds']},
 'checks':report.get('checks'),'rows':frames}
Path(args.output,'profile.json').write_text(json.dumps(summary,indent=2)+'\n')
if 'fixture' in state:
 np.savez_compressed(Path(args.output)/'terminal-diagnostic.npz',particle_q=state['fixture'].state.particle_q.numpy(),particle_qd=state['fixture'].state.particle_qd.numpy())
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','bindings']},indent=2))
raise SystemExit(0 if report.get('failure') is None and len(frames)==(3 if cli.gate else 36) else 1)
