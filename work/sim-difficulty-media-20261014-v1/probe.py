"""Read-only display probe, no physical solver or policy execution."""
from pathlib import Path
import os,json,sys,inspect
os.environ.setdefault('DISPLAY',':1');os.environ.setdefault('XAUTHORITY','/run/user/1000/gdm/Xauthority')
R=Path('/home/qunwang/wajueji');sys.path[:0]=[str(R/'src/v4'),str(R)]
import pyglet.display.xlib as xlib
if not hasattr(xlib.XlibScreen,'is_primary'):xlib.XlibScreen.is_primary=False
import newton,warp as wp,numpy as np
import run_layered_47_sparse_manual_contact_episode as manual
display=json.loads((R/'configs/v4/layered-47-interactive-proxy-practice-v1.json').read_text())['display']
wp.init();wp.set_device('cuda:0')
assembly,model,state,names,to_pose=manual.display_helpers._display_runtime(display)
print('MODEL',model.shape_count,model.body_count,flush=True)
print('NAMES',names,flush=True)
print('SHAPE_LABELS',model.shape_label,flush=True)
viewer=newton.viewer.ViewerGL(width=1000,height=625,vsync=False,paused=True,headless=True)
try:
 viewer.set_model(model)
 print('OBJECTS',[(k,type(v).__name__,getattr(v,'hidden',None)) for k,v in viewer.objects.items()],flush=True)
 print('MODEL_COLORS',getattr(model,'shape_color',None),flush=True)
 print('CAMERA',inspect.signature(viewer.camera.__class__),flush=True)
 states=json.loads((R/'runs/continuous-single-scoop-20261014-v1/display/states.json').read_text())
 print('STATE_KEYS',list(states[6265]),flush=True)
 print('BUCKET',states[6265]['bucket_origin_world_m'],flush=True)
 print('ASSEMBLY_POSES',list(assembly.assemble(np.asarray(states[6265]['mechanism_qpos']),include_meshes=False)[1]),flush=True)
finally:viewer.close()
