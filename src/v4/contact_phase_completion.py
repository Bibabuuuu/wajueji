"""Opt-in five-phase candidate. Observations affect only the next decision.

The original contact/curl candidate and its frozen artifacts are unchanged.
Geometry gaps must be fresh distances from ground-connected particle centers
to the solid bucket proxy, not fused heights or sampled-surface distances.
"""
from collections import deque
from dataclasses import dataclass
import math

from contact_phase_candidate import ContactPhaseCandidate


@dataclass(frozen=True)
class CompletionConfig:
    lift_motion_window_frames: int = 12
    lift_confirmation_frames: int = 30
    lift_window_origin_rise_m: float = .01
    lift_window_boom_extension_m: float = .0005
    lift_total_origin_rise_m: float = .08
    off_soil_gap_m: float = .003
    off_soil_confirmation_frames: int = 5

    def __post_init__(self):
        for value in (self.lift_motion_window_frames, self.lift_confirmation_frames,
                      self.off_soil_confirmation_frames):
            if type(value) is not int or value < 1:
                raise ValueError('invalid_completion_window')
        for value in (self.lift_window_origin_rise_m, self.lift_window_boom_extension_m,
                      self.lift_total_origin_rise_m, self.off_soil_gap_m):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid_completion_threshold')


class FivePhaseCandidate:
    def __init__(self, contact_config, config=CompletionConfig(), *, initial_mouth_angle_deg):
        self.contact = ContactPhaseCandidate(contact_config, initial_mouth_angle_deg=initial_mouth_angle_deg)
        self.config = config
        self.phase = self.contact.phase
        self.last_frame_end = 0
        self.motion = deque(maxlen=config.lift_motion_window_frames+1)
        self.lift_streak = 0
        self.lift_start = None
        self.gap_streak = 0

    def phase_for_decision(self, frame):
        if type(frame) is not int or frame != self.last_frame_end+1:
            raise ValueError('phase_decision_frame_not_next')
        return self.phase

    def observe_end(self, *, frame_end, soil_force_n, rock_contact_substeps,
                    mouth_angle_deg, actual_boom_cylinder_m, bucket_origin_z_m,
                    remaining_soil_gap_m=None, gap_frame_end=None):
        # Validate additional signals before the composed contact router mutates.
        for value in (actual_boom_cylinder_m, bucket_origin_z_m):
            if not math.isfinite(value):
                raise ValueError('invalid_completion_motion')
        if remaining_soil_gap_m is None:
            if gap_frame_end is not None:
                raise ValueError('gap_timestamp_without_distance')
        elif (not math.isfinite(remaining_soil_gap_m)
              or type(gap_frame_end) is not int or gap_frame_end != frame_end):
            raise ValueError('invalid_or_stale_soil_gap')
        result = self.contact.observe_end(frame_end=frame_end, soil_force_n=soil_force_n,
                    rock_contact_substeps=rock_contact_substeps, mouth_angle_deg=mouth_angle_deg)
        before = self.phase
        event = None
        cfg = self.config
        if before in ('approach_contact', 'penetrate'):
            self.phase = self.contact.phase
            event = result['event']
            if self.phase == 'curl_load':
                self.motion.append((frame_end, actual_boom_cylinder_m, bucket_origin_z_m))
        elif before == 'curl_load':
            self.motion.append((frame_end, actual_boom_cylinder_m, bucket_origin_z_m))
            rising = False
            if len(self.motion) == cfg.lift_motion_window_frames+1:
                old = self.motion[0]
                rising = (actual_boom_cylinder_m-old[1] > cfg.lift_window_boom_extension_m
                          and bucket_origin_z_m-old[2] > cfg.lift_window_origin_rise_m)
            if rising:
                if not self.lift_streak:
                    self.lift_start = self.motion[0]
                self.lift_streak += 1
                rise = bucket_origin_z_m-self.lift_start[2]
                if self.lift_streak >= cfg.lift_confirmation_frames and rise >= cfg.lift_total_origin_rise_m:
                    self.phase = 'lift'
                    event = dict(transition='curl_load_to_lift', evidence=[dict(
                        kind='sustained_actual_boom_and_bucket_rise',
                        motion_window_start_frame_end=self.lift_start[0],
                        first_matching_window_frame_end=frame_end-self.lift_streak+1,
                        consecutive_matching_windows=self.lift_streak, origin_rise_m=rise)])
            else:
                self.lift_streak = 0
                self.lift_start = None
        elif before == 'lift':
            clear = remaining_soil_gap_m is not None and remaining_soil_gap_m >= cfg.off_soil_gap_m
            self.gap_streak = self.gap_streak+1 if clear else 0
            if self.gap_streak >= cfg.off_soil_confirmation_frames:
                self.phase = 'off_soil'
                event = dict(transition='lift_to_off_soil', evidence=[dict(
                    kind='fresh_ground_connected_particle_to_solid_gap',
                    onset_frame_end=frame_end-cfg.off_soil_confirmation_frames+1,
                    minimum_required_gap_m=cfg.off_soil_gap_m,
                    consecutive_frames=cfg.off_soil_confirmation_frames)])
        # Off-soil is a latched semantic stage, not permission to bypass later
        # collision checks. A renewed contact does not rewrite earlier labels.
        self.last_frame_end = frame_end
        if event is not None:
            event.update(confirmation_frame_end=frame_end, first_available_decision_frame=frame_end+1)
        return dict(phase_before_observation=before, phase_for_next_decision=self.phase,
                    first_available_decision_frame=frame_end+1, event=event,
                    curl_boundary_unresolved=result['curl_boundary_unresolved'],
                    lift_matching_windows=self.lift_streak,
                    off_soil_clear_frames=self.gap_streak)
