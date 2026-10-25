"""Per-scoop soil-relative cut reference, built from current ground-connected soil.

This is a privileged simulation observation, not a deployable camera estimator.
The frozen pre-cut surface avoids chasing displaced load during this scoop.
"""
import numpy as np
from contact_phase_geometry import ground_connected_mask
from planning.cycle_geometry import TIP


class GroundSoilCutReference:
    def __init__(self, particles, proxy):
        points=np.asarray(particles,float)
        self.points=points[ground_connected_mask(points)]
        if len(self.points)<1000:raise ValueError('cut_reference_insufficient_ground_soil')
        ends=np.asarray(proxy['mouth_endpoints_xz_m'],float)
        self.root=np.array([ends[0,0],0.,ends[0,1]])
        chord=ends[1]-ends[0]
        self.vertical_pitch=float(np.arctan2(chord[0],-chord[1]))
        self.width=float(proxy['inner_clear_width_m'])

    def plan(self, pose, tip, pitch, radial, retract_m, embed_m):
        angle=self.vertical_pitch-pitch;c,s=np.cos(angle),np.sin(angle)
        rotation=pose[:3,:3]@np.array([[c,0,s],[0,1,0],[-s,0,c]])
        offset=rotation@(self.root-TIP)
        xy=np.asarray(tip[:2])-retract_m*radial+offset[:2]
        lateral=np.array([-radial[1],radial[0]])
        heights=[];counts=[]
        # Seven equal-width strips; robust upper soil height in each patch.
        for y in np.linspace(-.45*self.width,.45*self.width,7):
            d=self.points[:,:2]-(xy+y*lateral)
            z=self.points[(abs(d@radial)<=.075)&(abs(d@lateral)<=self.width/14),2]
            counts.append(len(z))
            if len(z)<8:raise ValueError('cut_reference_unknown_mouth_strip')
            heights.append(float(np.quantile(z,.95)))
        surface=float(np.median(heights));target_z=float(surface-offset[2]-embed_m)
        if target_z<.03:raise ValueError('cut_reference_ground_infeasible')
        return dict(surface_z_m=surface,tip_target_z_m=target_z,root_target_xy_m=xy.tolist(),
                    root_local_m=self.root.tolist(),vertical_pitch_rad=self.vertical_pitch,
                    strip_heights_m=heights,strip_counts=counts,ground_connected_particles=len(self.points),
                    observation_frame=0,reference='frozen_current_scoop_ground_soil',
                    surface_estimator='median_of_seven_strip_z95',root_embed_m=embed_m)
