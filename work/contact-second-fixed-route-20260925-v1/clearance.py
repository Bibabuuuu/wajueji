"""Explicit static checks for the user-authorized fixed truck route."""
import numpy as np
from layered_47_cycle_geometry import TransportClearance

class FixedRouteClearance(TransportClearance):
    def __init__(self,runtime):
        # Supply an explicit disabled map so the base constructor never observes soil.
        super().__init__(runtime,terrain={'disabled_for_fixed_transport':True})
        self.observation=dict(mode='fixed_route_static_geometry',particle_terrain_check_enabled=False,
                              physical_particles_modified=False,scope='transport only; next-scoop observation unchanged')
    def material_clearance(self,points):
        return None  # Not measured, never invent a passing terrain distance.
    def check(self,q,previous=None,*,physical=False):
        result,nearest=self.measure_clearance(q)
        fields=('bucket_truck_m','mechanism_truck_m','mechanism_ground_m','closure_m')
        if (not all(np.isfinite(result[k]) for k in fields) or result['bucket_truck_m']<.04
            or result['mechanism_truck_m']<.02 or result['mechanism_ground_m']<.02
            or result['closure_m']>(.001 if physical else .0001)):
            raise ValueError('fixed_route_static_clearance:'+str(result)+':mesh='+str(nearest))
        return result
