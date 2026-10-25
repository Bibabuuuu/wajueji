"""Local actual-state phase matching; no new position-feedback controller."""
import numpy as np

AXES = [7, 9, 12, 15]
VERSION = 'local_q_pose_lead_v1'
CONFIG = dict(back_frames=2, forward_frames=30, max_lead_frames=6,
              position_corridor_m=.15, angle_corridor_rad=float(np.deg2rad(10)),
              joint_corridor_si=[.03, .01, .01, .02],
              joint_scales_si=[.02, .005, .005, .015],
              position_scale_m=.05, angle_scale_rad=.1)


class ProgressAlignment:
    def __init__(self, reference_q, reference_poses):
        self.q = np.asarray(reference_q, dtype=float)[:, AXES]
        self.poses = np.asarray(reference_poses, dtype=float)
        if (self.poses.shape != (len(self.q), 4, 4) or len(self.q) < 2
                or not np.isfinite(self.q).all() or not np.isfinite(self.poses).all()):
            raise ValueError('alignment_reference_contract')
        self.measured = 0.

    def observe(self, q, pose, cursor):
        q = np.asarray(q, dtype=float)
        pose = np.asarray(pose, dtype=float)
        if (q.shape != (19,) or pose.shape != (4, 4) or not np.isfinite(q).all()
                or not np.isfinite(pose).all() or not np.isfinite(cursor)
                or not 0 <= cursor <= len(self.q)-1):
            raise ValueError('alignment_actual_contract')
        indices = np.arange(max(0, int(self.measured)-CONFIG['back_frames']),
                            min(len(self.q), int(np.ceil(self.measured))+CONFIG['forward_frames']+1))
        joint = q[AXES]-self.q[indices]
        distance = np.linalg.norm(pose[:3, 3]-self.poses[indices, :3, 3], axis=1)
        trace = np.einsum('ij,nij->n', pose[:3, :3], self.poses[indices, :3, :3])
        angle = np.arccos(np.clip((trace-1)/2, -1, 1))
        cost = np.sum((joint/CONFIG['joint_scales_si'])**2, axis=1)
        cost += (distance/CONFIG['position_scale_m'])**2+(angle/CONFIG['angle_scale_rad'])**2
        # Repeated stationary poses do not uniquely determine time. Only ties
        # use the current reference clock; the result remains a phase estimate.
        cost += 1e-8*(indices-cursor)**2
        best = int(np.argmin(cost)); frame = int(indices[best])
        trusted = bool(distance[best] <= CONFIG['position_corridor_m']
                       and angle[best] <= CONFIG['angle_corridor_rad']
                       and np.all(abs(joint[best]) <= CONFIG['joint_corridor_si']))
        if trusted:
            self.measured = max(self.measured, float(frame))
        lead = self.measured+CONFIG['max_lead_frames']
        return dict(trusted=trusted, matched_frame=frame, actual_progress=self.measured,
                    reference_limit=min(float(len(self.q)-1), lead),
                    position_error_m=float(distance[best]), angle_error_rad=float(angle[best]),
                    joint_error_si=joint[best].tolist(),
                    mode='outside' if not trusted else ('wait' if cursor >= lead else 'follow'))

    def endpoint_ready(self, q, pose):
        joint = np.asarray(q)[AXES]-self.q[-1]
        pose = np.asarray(pose)
        angle = np.arccos(np.clip((np.sum(pose[:3, :3]*self.poses[-1, :3, :3])-1)/2, -1, 1))
        return bool(self.measured >= len(self.q)-1-CONFIG['max_lead_frames']
                    and np.linalg.norm(pose[:3, 3]-self.poses[-1, :3, 3]) <= .015
                    and angle <= np.deg2rad(1)
                    and np.all(abs(joint) <= [.004, .001, .002, .003]))
