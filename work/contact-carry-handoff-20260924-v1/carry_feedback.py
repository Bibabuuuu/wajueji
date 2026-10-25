"""D155 opt-in retain-and-lift controller; no physical state is reset here.

FINISH_CURL completes near-full closure before the handoff. CLEAR_PILE
continues lifting after the height-independent handoff. LIFT/ALIGN accept Cartesian
targets but retain the measured carry pitch. Opening requires a separate
unload controller, never a target-pose change inside this controller.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/contact-preview-lift-20260924-v1'))
from continuous_preview import preview
from contact_geometric_feedback import ContactGeometryFeedback, FeedbackConfig, rate_bounded_velocity, LIMITS
from run_layered_47_sparse_manual_contact_episode import BUCKET_INTERNAL_UPPER_TARGET_M, KINEMATIC_JOINT_MARGIN_M
from planning.cycle_geometry import tool_pose, TIP


class CarryFeedback(ContactGeometryFeedback):
    def __init__(self, assembly, chain, ground_guard, qpos, previous_command,
                 *, frame_end, initial_soil_top_m):
        tip, pitch = tool_pose(assembly, qpos, TIP)
        super().__init__(assembly, chain, ground_guard,
                         dict(entry_xy_m=tip[:2], surface_z_m=tip[2]), qpos,
                         FeedbackConfig(late_curl_no_down=True, command_rate_per_s=2.,
                                        translation_speed_m_s=.15))
        self.previous_command = np.array(previous_command, dtype=float)
        if (self.previous_command.shape != (4,) or not np.isfinite(self.previous_command).all()
                or np.any(abs(self.previous_command) > LIMITS+1e-12)):
            raise ValueError('carry_invalid_previous_command')
        self.last_frame = frame_end
        self.first_command = True
        self.carry_pitch = float(pitch)
        self.carry_cylinder_m = float(qpos[chain.axes[3]])
        low, high = chain.limits[chain.axes[3]]
        self.upper = min(BUCKET_INTERNAL_UPPER_TARGET_M, high-KINEMATIC_JOINT_MARGIN_M)
        if (self.carry_cylinder_m-low)/(self.upper-low) < .95:
            raise ValueError('finish_curl_requires_95_percent_actual_closure')
        self.stage = 'FINISH_CURL'
        self.handoff_frame = None
        self.target_world = None
        self.initial_soil_top_m = float(initial_soil_top_m)
        if not np.isfinite(self.initial_soil_top_m):
            raise ValueError('carry_invalid_height_reference')

    def accept_handoff(self, *, readiness, qpos):
        if (self.handoff_frame is not None or not readiness['ready']
                or readiness['frame_end'] != self.last_frame):
            raise ValueError('carry_requires_fresh_handoff')
        q = np.asarray(qpos, float)
        low, _ = self.chain.limits[self.chain.axes[3]]
        if (q[self.chain.axes[3]]-low)/(self.upper-low) < .99:
            raise ValueError('carry_requires_99_percent_actual_closure')
        self.carry_pitch = float(self.coordinates(q)[3])
        self.carry_cylinder_m = float(q[self.chain.axes[3]])
        self.handoff_frame = self.last_frame
        self.stage = 'CLEAR_PILE'

    def set_transport_target(self, *, stage, target_world, readiness, frame_end, clearance_check):
        """Caller supplies the existing geometry guard; no implicit bypass.

The first transport target must have fresh new-gate evidence. Every proposed
motion still passes clearance_check in command(), including the entry pose.
"""
        if stage not in ('LIFT', 'ALIGN') or not callable(clearance_check):
            raise ValueError('carry_invalid_transport_stage')
        if self.handoff_frame is None or frame_end != self.last_frame:
            raise ValueError('carry_requires_completed_handoff')
        target = np.asarray(target_world, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('carry_invalid_target')
        self.stage = stage
        self.target_world = target.copy()
        self.clearance_check = clearance_check

    def command(self, *, frame, qpos, projector):
        if frame != self.last_frame+1:
            raise ValueError('carry_noncontiguous_frame')
        q = np.asarray(qpos, float)
        state = self.coordinates(q)
        actual_bucket = float(q[self.chain.axes[3]])
        # Numerical tolerances diagnose actual retention, not a claim of zero spill.
        if actual_bucket < self.carry_cylinder_m-.001:
            raise ValueError('carry_actual_bucket_opened')
        if state[3] < self.carry_pitch-.01:
            raise ValueError('carry_actual_pitch_lost')
        jac, ground_gradient, body_gradient = self.linearize(q, include_body=True)
        minimum = self.ground.measure(q)['minimum_bucket_z_m']
        if self.stage in ('FINISH_CURL', 'CLEAR_PILE'):
            # This is a motion target, never a handoff readiness condition.
            # Execution is still subject to an explicit runner time/frame budget.
            remaining = self.initial_soil_top_m+.12-minimum
            # Before handoff keep rising even if the diagnostic height was
            # reached; 3 seconds of real rise is required by the new gate.
            upward = .08 if self.stage == 'FINISH_CURL' else np.clip(1.5*remaining, 0., .08)
            velocity = np.array([0., 0., upward, 0.])
        else:
            self.clearance_check(q)
            velocity = np.r_[np.clip(1.5*(self.rotation@self.target_world-state[:3]), -.08, .08), 0.]
        velocity[3] = np.clip(1.5*(self.carry_pitch-state[3]), -.03, .03)
        cap = float(np.clip((self.upper-actual_bucket)/.15/LIMITS[3], 0., 1.))
        bounds = [(-1., 1.), (0., 1.), (-1., 1.), (0., cap)]
        if self.stage == 'FINISH_CURL':
            close = min(.005, max(0., 1.5*(self.upper-actual_bucket)))
            # Form both slew and curl bounds in normalized units; converting
            # a saturated SI bound back can invert the interval by one ULP.
            lower = min(cap, close/LIMITS[3], max(0., self.previous_command[3]/LIMITS[3]+2./60-1e-12))
            bounds[3] = (lower, cap)
            velocity[3] = max(0., float(jac[3,3]*close/LIMITS[3]))
        slew_override = None
        if self.first_command:
            # Exact command continuity, while retaining native target/velocity,
            # integral, warmstart, pending reaction and all actual q/qdot fields.
            requested = self.previous_command.copy()
        else:
            # Curl must ramp together with the compensating boom/arm motion.
            # Reduce only the requested curl minimum if it cannot fit the
            # existing slew box; never dispatch the solver's slew override.
            for curl_scale in ((1.,.5,.25,.125,.0625,0.) if self.stage=='FINISH_CURL' else (1.,)):
                trial_bounds=list(bounds)
                trial_bounds[3]=(bounds[3][0]*curl_scale,bounds[3][1])
                normalized, slew_override = rate_bounded_velocity(
                    jac, velocity, ground_gradient, (.03-minimum)/.15,
                    previous=self.previous_command/LIMITS, rate=2., axis_bounds=trial_bounds,
                    exact_pitch=False,
                    # Reserve upward corner motion during curling; a zero
                    # tangent speed can still curve downward in the 150 ms
                    # nonlinear preview. Keep the original constraints too.
                    no_down_gradient=np.vstack([body_gradient,body_gradient-.05*jac[2]]))
                if slew_override is None:break
            if slew_override is not None:
                raise ValueError('carry_no_safe_continuous_velocity:'+slew_override)
            requested = normalized*LIMITS
        applied = projector.project(q, requested)
        reason = projector.reason
        if (applied[3] < -1e-10 or applied[1] < -1e-10
                or np.min(body_gradient@(applied/LIMITS)) < -1e-7):
            raise ValueError('carry_projected_command_loses_retention')
        if np.max(abs((applied-self.previous_command)/LIMITS)) > 2./60+1e-9:
            raise ValueError('carry_projection_exceeds_slew:'+str(dict(frame=frame,stage=self.stage,
                previous=self.previous_command.tolist(),requested=requested.tolist(),applied=applied.tolist(),
                projection_reason=reason,slew_override=slew_override)))
        if self.first_command and not np.array_equal(applied, self.previous_command):
            raise ValueError('carry_first_command_not_continuous')
        original = self.chain.q.copy()
        start_heights = self.body_heights(q)
        reference = self.chain.residual(q)
        minimum_delta = 0.
        try:
            self.chain.q = q.copy()
            scale = max(1., float(np.max(abs(applied)/np.array([.12,.0225,.0225,.0225]))))
            for _ in range(9):
                predicted, _ = preview(projector, applied/scale, scale/60, reference)
                minimum_delta = min(minimum_delta, float(np.min(self.body_heights(predicted)-start_heights)))
                if (self.ground.measure(predicted)['clearance_m'] < .03-1e-6
                        or minimum_delta < -1e-6
                        or self.coordinates(predicted)[3] < self.carry_pitch-.01
                        or predicted[self.chain.axes[3]] < self.carry_cylinder_m-.001):
                    raise ValueError('carry_preview_loses_retention:'+str(dict(frame=frame,
                        minimum_delta_m=minimum_delta,pitch=float(self.coordinates(predicted)[3]),
                        carry_pitch=self.carry_pitch,bucket=float(predicted[self.chain.axes[3]]),
                        carry_bucket=self.carry_cylinder_m,command=applied.tolist())))
                if self.stage not in ('FINISH_CURL', 'CLEAR_PILE'):
                    self.clearance_check(predicted)
        finally:
            self.chain.q = original
        self.previous_command = applied.copy()
        self.last_frame = frame
        self.first_command = False
        return applied, dict(stage=self.stage, requested_command_si=requested.tolist(),
                             carry_pitch_rad=self.carry_pitch, carry_cylinder_m=self.carry_cylinder_m,
                             minimum_bucket_z_m=minimum, minimum_preview_body_delta_m=minimum_delta,
                             desired_task_velocity=velocity.tolist(), projection_reason=reason,
                             slew_safety_override=slew_override)
