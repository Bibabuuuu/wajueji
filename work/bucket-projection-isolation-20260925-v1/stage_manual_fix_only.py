"""Stage this task's manual-entry changes without pre-existing user edits."""
import difflib,subprocess
from pathlib import Path
path='src/v4/run_layered_47_sparse_manual_contact_episode.py'
base=subprocess.check_output(['git','show','HEAD:'+path],text=True)
working=Path(path).read_text()
result=base

def replace(old,new):
    global result
    assert result.count(old)==1,old[:100]
    result=result.replace(old,new,1)

def block(start,end):
    return working[working.index(start):working.index(end,working.index(start))]

line='    cycle_mode = bool(getattr(args, "cycle_checkpoint", False))\n'
replace(line,line+block('    motion_consistency = bool(', '    showcase_mode ='))
replace("allow_training_reset=getattr(args,'training_reset_checkpoint',False))", "allow_training_reset=getattr(args,'training_reset_checkpoint',False),\n                allow_bucket_motion_consistency=motion_consistency)")
line='            Path(motion.probe.snapshots.__file__),\n        ]\n'
replace(line,line+block('        if motion_consistency:\n            binding_paths', '        if fill_config:'))
line='        with ExitStack() as contexts:\n'
replace(line,line+block('            if motion_consistency:\n                from bucket_velocity_origin_candidate', '            contexts.enter_context(cleaned_manual_model())'))
line='                from layered_47_cycle_bridge import restore_frame_interface, controller_snapshot\n'
replace(line,line+block('                if motion_consistency:\n                    from bucket_motion_consistency', '                restore_frame_interface(held.bridge.state'))
replace('                sweep_runtime = MovingBucketSweepRuntime(\n',block('                sweep_class = MovingBucketSweepRuntime\n','                    fixture, bridge, proxy, base["mpm"]["material"]["friction"])'))
line="    parser.add_argument('--cycle-checkpoint', action='store_true', help='Audited continuous second/third scoop prepoint')\n"
replace(line,line+block("    parser.add_argument('--bucket-motion-consistency'", "    parser.add_argument('--cycle-save'"))
patch=''.join(difflib.unified_diff(base.splitlines(True),result.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
Path('/tmp/wajueji-motion-manual-only.patch').write_text(patch)
subprocess.run(['git','apply','--cached','--check','/tmp/wajueji-motion-manual-only.patch'],check=True)
subprocess.run(['git','apply','--cached','/tmp/wajueji-motion-manual-only.patch'],check=True)
print('Staged only motion-consistency additions; existing manual workflow edits remain unstaged.')
