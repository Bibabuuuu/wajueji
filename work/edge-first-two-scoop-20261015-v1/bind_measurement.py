from pathlib import Path
import sys
R=Path('/home/qunwang/wajueji');sys.path.insert(0,str(R/'work/continuous-single-scoop-20261014-v1'))
from continuous_scoop import read,write,sha
config=R/'configs/v4/live-bucket-fill-local-v1.json';values=read(config)
paths=[config,R/'src/v4/live_bucket_fill.py']+[Path(values[k]) for k in ['measurement_source','contract','front_rim','calibration']]
out=R/'work/edge-first-two-scoop-20261015-v1/measurement-bindings.json'
assert not out.exists();write(out,{str(p):sha(p) for p in paths});print('MEASUREMENT_INPUTS_BOUND_BEFORE_FIRST_HOLD')
