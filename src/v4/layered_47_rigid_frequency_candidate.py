"""Opt-in rigid substep experiment; frozen 4800 Hz bridge is not modified."""
import numpy as np

from layered_frequency_bridge import HoldBridge, subcycled_step, _bridge_step, _finite_tree
from contracts import ACTION_SPEED_LIMIT_SI

SUPPORTED_HZ = (2400, 3600, 4800)


def validate_run_mode(hz, *, smoke, combined_smoke):
    if hz not in SUPPORTED_HZ:
        raise ValueError('rigid_frequency_candidate_hz')
    if hz != 4800 and not (smoke or combined_smoke):
        raise ValueError('rigid_frequency_candidate_headless_only')


class RigidFrequencyCommandBridge(HoldBridge):
    def __init__(self, bridge, hz):
        if hz not in SUPPORTED_HZ:
            raise ValueError('rigid_frequency_candidate_hz')
        super().__init__(bridge)
        self.hz = hz
        self.command = np.zeros(4)

    def step(self, dt, substeps, wrenches):
        command = np.asarray(self.command)
        if (command.shape != (4,) or not np.isfinite(command).all()
                or np.any(np.abs(command) > ACTION_SPEED_LIMIT_SI)):
            raise ValueError('motion_command_limit')
        blocks = round(dt*120)
        half_steps = self.hz//120
        if (blocks not in (1,2) or abs(dt-blocks/120)>1e-12
                or substeps != blocks*half_steps):
            raise ValueError('rigid_frequency_candidate_contract')
        if self.hz == 4800:
            return subcycled_step(self.bridge, dt, substeps, command, wrenches)
        states = []
        for _ in range(blocks):
            # Wrenches are already F=J/MPM_dt. Hold the same F for both
            # halves: integrated impulse remains J, regardless of substeps.
            state = _bridge_step(self.bridge, 1/120, half_steps, command, wrenches)
            if not _finite_tree(state):
                raise ValueError('nonfinite_bridge_half_step')
            states.append(state)
        result = dict(states[-1])
        result['contact_substeps'] = {key:sum(s.get('contact_substeps',{}).get(key,0) for s in states)
            for key in set().union(*(s.get('contact_substeps',{}) for s in states))}
        result['forbidden_contact_count'] = sum(s.get('forbidden_contact_count',0) for s in states)
        result['closure_m'] = max(s['closure_m'] for s in states)
        return result
