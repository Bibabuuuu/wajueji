"""Partial, causal D-146 phase candidate; not connected to live policy control.

Only approach -> penetrate -> curl is implemented. Curl is the terminal state
of this partial router, NOT a claim that later lift/off-soil samples are curl.
An end-of-frame observation changes the phase of the NEXT decision only.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ContactPhaseConfig:
    curl_angle_threshold_deg: float
    soil_force_threshold_n: float = 10.0
    soil_confirmation_frames: int = 3
    rock_confirmation_frames: int = 3

    def __post_init__(self):
        if (not math.isfinite(self.curl_angle_threshold_deg)
                or not -180 < self.curl_angle_threshold_deg < 180
                or not math.isfinite(self.soil_force_threshold_n)
                or self.soil_force_threshold_n <= 0):
            raise ValueError('invalid_phase_threshold')
        for value in (self.soil_confirmation_frames, self.rock_confirmation_frames):
            if type(value) is not int or value < 1:
                raise ValueError('invalid_confirmation_window')


class ContactPhaseCandidate:
    def __init__(self, config, *, initial_mouth_angle_deg):
        if not math.isfinite(initial_mouth_angle_deg) or not -180 <= initial_mouth_angle_deg <= 180:
            raise ValueError('invalid_initial_angle')
        self.config = config
        self.last_frame_end = 0
        self.previous_angle = float(initial_mouth_angle_deg)
        self.phase = 'approach_contact'
        self.soil_streak = 0
        self.rock_streak = 0
        self.curl_boundary_unresolved = False

    def phase_for_decision(self, frame):
        if type(frame) is not int or frame != self.last_frame_end + 1:
            raise ValueError('phase_decision_frame_not_next')
        return self.phase

    def observe_end(self, *, frame_end, soil_force_n, rock_contact_substeps,
                    mouth_angle_deg):
        # Reject missing/reordered frames instead of silently stitching a phase
        # history across a reset. All validation precedes state mutation.
        if type(frame_end) is not int or frame_end != self.last_frame_end + 1:
            raise ValueError('phase_observation_not_contiguous')
        if not math.isfinite(soil_force_n) or soil_force_n < 0:
            raise ValueError('invalid_soil_force')
        if not math.isfinite(mouth_angle_deg) or not -180 <= mouth_angle_deg <= 180:
            raise ValueError('invalid_mouth_angle')
        if type(rock_contact_substeps) is not int or not 0 <= rock_contact_substeps <= 80:
            raise ValueError('invalid_rock_contact_substeps')
        cfg = self.config
        self.soil_streak = min(cfg.soil_confirmation_frames, self.soil_streak + 1) if soil_force_n > cfg.soil_force_threshold_n else 0
        self.rock_streak = min(cfg.rock_confirmation_frames, self.rock_streak + 1) if rock_contact_substeps > 0 else 0
        event = None
        previous_phase = self.phase
        if self.phase == 'approach_contact':
            sources = []
            if self.soil_streak == cfg.soil_confirmation_frames:
                sources.append(dict(kind='soil_resultant_force', onset_frame_end=frame_end-cfg.soil_confirmation_frames+1))
            if self.rock_streak == cfg.rock_confirmation_frames:
                sources.append(dict(kind='rock_contact_pair_presence', onset_frame_end=frame_end-cfg.rock_confirmation_frames+1))
            if sources:
                self.phase = 'penetrate'
                self.curl_boundary_unresolved = mouth_angle_deg <= cfg.curl_angle_threshold_deg
                event = dict(transition='approach_contact_to_penetrate', evidence=sources)
        elif self.phase == 'penetrate':
            if self.previous_angle > cfg.curl_angle_threshold_deg >= mouth_angle_deg:
                self.phase = 'curl_load'
                self.curl_boundary_unresolved = False
                event = dict(transition='penetrate_to_curl_load', evidence=[dict(
                    kind='frozen_angle_downward_crossing', onset_frame_end=frame_end)])
        # No action magnitude condition: pauses/brief force loss never rewind a
        # confirmed phase. No fabricated lift/off-soil transition is emitted.
        self.last_frame_end = frame_end
        self.previous_angle = float(mouth_angle_deg)
        if event is not None:
            event.update(confirmation_frame_end=frame_end,
                         first_available_decision_frame=frame_end+1)
        return dict(phase_before_observation=previous_phase,
                    phase_for_next_decision=self.phase,
                    first_available_decision_frame=frame_end+1,
                    event=event, curl_boundary_unresolved=self.curl_boundary_unresolved,
                    partial_router_terminal=self.phase == 'curl_load')
