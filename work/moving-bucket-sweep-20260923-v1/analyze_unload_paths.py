"""Use the frozen D-132 geometric classifier on protected frame endpoints only.
Raw pre-correction solver positions are not counted as final escapes.
"""
from pathlib import Path
source=Path('/home/qunwang/wajueji/work/loaded-transport-leak-path-20260923-v1/analyze_paths.py').read_text()
changes={
 "run=work/'replay-v2'":"run=work/'unload-180'",
 "report=json.loads((work/'verified-replay.json').read_text())":"report=json.loads((run/'report.json').read_text())",
 "assert report['status']=='replay_paths_recorded' and report['frames']==887":"assert report['status']=='candidate_recorded' and report['frames']==180",
 "assert report['all_887_recorded_states_exact']":"assert report['all_hard_guards_passed']",
 " a=tolocal(pos[f-1],old);b=tolocal(pre[f-1],old);c=tolocal(pos[f],old);d=tolocal(pos[f],new)\n crossing(a,b,f,'mpm_before_projection')\n crossing(b,c,f,'project_outside')\n crossing(c,d,f,'rigid_pose_advance')":" a=tolocal(pos[f-1],old);d=tolocal(pos[f],new)\n crossing(a,d,f,'protected_frame_endpoints')",
 "(work/'path-summary.json')":"(work/'candidate-path-summary.json')",
 "(work/'first-exits.json')":"(work/'candidate-first-exits.json')",
 "(work/'full-wall-crossings.json')":"(work/'candidate-full-wall-crossings.json')",
}
for a,b in changes.items():
 assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
source=source.replace('range(1,888)', 'range(1,181)').replace('min(f+6,888)', 'min(f+6,181)')
source=source.replace("(work/'candidate-","(run/'candidate-")
exec(compile(source,str(Path(__file__)), 'exec'),{'__file__':str(Path(__file__)), '__name__':'__main__'})
