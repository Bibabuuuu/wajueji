"""Causal adapter for the same geometric signals used by derived demonstrations."""
import numpy as np
from contact_phase_completion import FivePhaseCandidate
from contact_phase_geometry import SolidBucketGap


def mouth_angle_deg(pose, polygon_xz):
    polygon = np.asarray(polygon_xz, dtype=float)
    local = np.c_[polygon[:, 0], np.zeros(len(polygon)), polygon[:, 1]]
    world = local @ pose[:3, :3].T + pose[:3, 3]
    radial = np.array([pose[1, 1], -pose[0, 1]])
    radial /= np.linalg.norm(radial)
    chord = world[-1] - world[0]
    return float(np.rad2deg(np.arctan2(chord[:2] @ radial, -chord[2])))


class LivePhaseObserver:
    def __init__(self, proxy, polygon_xz, contact_config, completion_config, initial_pose):
        self.polygon = polygon_xz
        self.router = FivePhaseCandidate(contact_config, completion_config,
                                        initial_mouth_angle_deg=mouth_angle_deg(initial_pose, polygon_xz))
        self.geometry = SolidBucketGap(proxy)

    def observe_end(self, *, frame, state, pose, soil_force_n, rock_contact_substeps, particles):
        before = self.router.phase_for_decision(frame)
        observation = self.geometry.observe(particles(), pose) if before in ('lift', 'off_soil') else None
        gap = None if observation is None else observation['remaining_soil_gap_m']
        phase = self.router.observe_end(
            frame_end=frame, soil_force_n=soil_force_n, rock_contact_substeps=rock_contact_substeps,
            mouth_angle_deg=mouth_angle_deg(pose, self.polygon), actual_boom_cylinder_m=state['mechanism_qpos'][9],
            bucket_origin_z_m=pose[2, 3], remaining_soil_gap_m=gap, gap_frame_end=frame if gap is not None else None)
        return dict(frame=frame, phase=phase, observation=observation, bucket_mpm_force_norm_n=soil_force_n)
