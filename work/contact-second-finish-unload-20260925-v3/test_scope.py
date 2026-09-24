"""CPU guard scope checks; no physical initialization."""
from types import SimpleNamespace
import numpy as np
import controller
original=controller.yaw_preview.preview
controller.yaw_preview.preview=lambda *args:(np.zeros(19),0.)
feedback=SimpleNamespace(body_heights=lambda q:np.zeros(8))
try:
 for stage in ('CLEAR_PILE','ALIGN','DUMP','PLAN_TRANSPORT'):
  owner=SimpleNamespace(stage=stage,loaded_center_peak=1.)
  controller.bounded_preview(owner,feedback,None,None,None,None,None)
 owner=SimpleNamespace(stage='FINISH_CURL',loaded_center_peak=.009)
 controller.bounded_preview(owner,feedback,None,None,None,None,None)
 owner.loaded_center_peak=.011
 try:controller.bounded_preview(owner,feedback,None,None,None,None,None)
 except ValueError as e:assert str(e)=='loaded_preview_cumulative_drawdown_exceeds_10mm'
 else:raise AssertionError('FINISH_CURL guard missing')
finally:controller.yaw_preview.preview=original
print('four non-curl stages exempt; 9mm allowed; 11mm rejected')
