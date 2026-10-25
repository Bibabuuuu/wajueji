"""Hold one MPM average wrench over valid 120 Hz rigid bridge intervals."""
import numpy as np
from run_layered_47_excavation_approach import _bridge_step
from run_layered_mound_single_rock_press_smoke import _finite_tree


def subcycled_step(bridge, dt, substeps, command, wrenches):
    blocks = round(dt * 120)
    if blocks not in (1, 2) or abs(dt - blocks / 120) > 1e-12 or substeps != blocks * 40:
        raise ValueError('frequency_bridge_contract')
    states = []
    for _ in range(blocks):
        state = _bridge_step(bridge, 1/120, 40, command, wrenches)
        if not _finite_tree(state):
            raise ValueError('nonfinite_bridge_half_step')
        states.append(state)
    result = dict(states[-1])
    result['contact_substeps'] = {key: sum(s.get('contact_substeps', {}).get(key, 0) for s in states)
        for key in set().union(*(s.get('contact_substeps', {}) for s in states))}
    result['forbidden_contact_count'] = sum(s.get('forbidden_contact_count', 0) for s in states)
    result['closure_m'] = max(s['closure_m'] for s in states)
    return result


class HoldBridge:
    def __init__(self, bridge):
        self.bridge = bridge

    @property
    def state(self):
        return self.bridge.state

    def step(self, dt, substeps, wrenches):
        return subcycled_step(self.bridge, dt, substeps, np.zeros(4), wrenches)

    def close(self, **kwargs):
        return self.bridge.close(**kwargs)
