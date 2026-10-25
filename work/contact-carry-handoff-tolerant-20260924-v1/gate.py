"""Explicit user-approved jitter-tolerant configuration; frozen D155 stays intact."""
from copy import deepcopy
from dataclasses import asdict
from contact_transport_handoff import CarryHandoffConfig, CarryHandoffGate

CONFIG = CarryHandoffConfig(trend_frames=30, maximum_drawdown_m=.010)

def restore_with_tolerance(snapshot):
    """Keep the actual three-second history; change only the rise interpretation.

    Millimetre drawdowns up to 10 mm are ignored when every rolling half-second
    still rises. Actual closure must remain >=99% for the full three seconds.
    No physical state, actuator command, source snapshot or old gate is mutated.
    """
    if any(snapshot['config'][k] != v for k,v in
           dict(hz=60,hold_seconds=3.,closure_fraction=.99).items()):
        raise ValueError('tolerant_gate_requires_original_closure_contract')
    revised=deepcopy(snapshot)
    revised['config']=asdict(CONFIG)
    return CarryHandoffGate.restore(revised)
