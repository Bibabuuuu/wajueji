"""Reuse verified unloaded tracking; retain the original prepoint clearances."""
import importlib.util,sys
import numpy as np
from runtime import ROOT
OLD=ROOT/'work/contact-fixed-truck-continuation-20260925-v2'
sys.path.append(str(OLD))
spec=importlib.util.spec_from_file_location('_prepoint_transport',OLD/'controller.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from planning.rear_lower_prepoint import mechanism_and_truck_clearance
from layered_47_cycle_geometry import TransportClearance
LIMITS=base.LIMITS

class PrepointClearance(TransportClearance):
    def check(self,q,previous=None,*,physical=False):
        result=super().check(q,previous,physical=physical)
        bodies=self.runtime.assembly.assemble(q,include_meshes=False)[1]
        mechanism=mechanism_and_truck_clearance(self.geometries,bodies,self.terrain,
                                               self.truck_triangles.reshape(-1,3))
        minimum=.30 if physical else .32
        if result['bucket_terrain_m']<minimum or any(v<.02 for v in mechanism.values()):
            raise ValueError('next_prepoint_clearance:'+str((result,mechanism)))
        return dict(result,**mechanism)

class PrepointController(base.TransportController):
    pass
