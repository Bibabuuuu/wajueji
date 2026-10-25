"""Restore the frozen D157 controller's bias and last issued command."""
import importlib.util,sys
import numpy as np
from runtime import ROOT,PARENT
sys.path.append(str(PARENT))
spec=importlib.util.spec_from_file_location('_frozen_d157_control',PARENT/'controller.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
LIMITS=base.LIMITS

class TransportController(base.TransportController):
    def __init__(self,r,checker,previous):
        super().__init__(r,checker,previous)
        saved=r.metadata['transport_controller_state']
        if not np.array_equal(saved['previous_command'],previous):
            raise ValueError('saved_transport_command_mismatch')
        bias=np.asarray(saved['load_bias'],float)
        if bias.shape!=(4,) or not np.isfinite(bias).all() or np.any(abs(bias)>.02):
            raise ValueError('invalid_saved_transport_bias')
        self.tracking.bias=bias.copy()
        self.first=True  # Explicit resume keeps the first issued command exact.
