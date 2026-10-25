import json
import subprocess
from tempfile import TemporaryDirectory
from pathlib import Path
from layered_47_cycle_runtime import CycleRuntime
from layered_47_cycle_geometry import stationary_terrain
# Reconstruct the original frozen selector even after production is updated.
source = subprocess.check_output(['git','show','ddb4fce:src/v4/layered_47_cycle_selection.py'],text=True)
namespace = {}
exec(compile(source,'frozen-selector-ddb4fce','exec'),namespace)
rank_workspace_candidates = namespace['rank_workspace_candidates']
ROOT=Path('/home/qunwang/wajueji')
src=ROOT/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260924-110051-1421079/final-001461.cycle.npz'
temporary=TemporaryDirectory();r=CycleRuntime(src,Path(temporary.name)/'observation',allow_training_reset=True)
t,o=stationary_terrain(r);cs=rank_workspace_candidates(t,1)
print('candidates',len(cs))
for c in cs:print(c['entry_xy_m'],c['score'],c['contact']['contact_xy_m'],c['contact']['contact_surface_z_m'])
import math

def clean(value):
    if hasattr(value,'tolist'): return clean(value.tolist())
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value
with (Path(__file__).parent/'baseline.json').open('x') as out:
    json.dump(clean(dict(source=r.metadata['source_identity'],terrain=t,observation=o,candidates=cs)),out,allow_nan=False)
    out.write('\n')
