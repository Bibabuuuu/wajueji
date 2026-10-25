"""Opt-in carry handoff evidence; does not relabel frozen BC demonstrations.

Readiness is recomputed from contiguous frame-end observations. It is not a
collision waiver, a payload measurement, or proof of successful unloading.
"""
from collections import deque
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class CarryHandoffConfig:
    hz: int = 60
    hold_seconds: float = 3.0
    closure_fraction: float = .99
    trend_frames: int = 12
    trend_rise_m: float = .0001
    maximum_drawdown_m: float = .001

    def __post_init__(self):
        if (type(self.hz) is not int or self.hz != 60
                or type(self.trend_frames) is not int or self.trend_frames < 1
                or not all(math.isfinite(v) and v > 0 for v in asdict(self).values())
                or self.closure_fraction > 1
                or not float(self.hold_seconds * self.hz).is_integer()
                or self.trend_frames > self.hold_seconds * self.hz):
            raise ValueError('invalid_carry_handoff_config')


class CarryHandoffGate:
    """Near-full closure held for 180 intervals and sustained actual rise.

181 endpoints span exactly 3 seconds at 60 Hz. Every 0.2-second window
must show actual origin rise, and neither origin nor bucket minimum may
draw down more than 1 mm from its running peak. These noise/trend values
are explicit engineering settings, not user-specified physical constants.
"""
    schema = 'carry_handoff_gate_v1'

    def __init__(self, *, initial_soil_top_m, source_sha256, config=CarryHandoffConfig()):
        if (not math.isfinite(initial_soil_top_m) or len(source_sha256) != 64
                or any(c not in '0123456789abcdef' for c in source_sha256)):
            raise ValueError('invalid_carry_handoff_reference')
        self.initial_soil_top_m = float(initial_soil_top_m)
        self.source_sha256 = source_sha256
        self.config = config
        self.rows = deque(maxlen=int(config.hz * config.hold_seconds) + 1)
        self.last_frame = None

    def observe_end(self, *, frame, closure_fraction, origin_z_m,
                    minimum_bucket_z_m, contact_established):
        if (type(frame) is not int or frame < 0
                or (self.last_frame is not None and frame != self.last_frame + 1)
                or type(contact_established) is not bool
                or not all(math.isfinite(v) for v in
                           (closure_fraction, origin_z_m, minimum_bucket_z_m))):
            raise ValueError('invalid_or_noncontiguous_carry_observation')
        row = dict(frame=frame, closure_fraction=float(closure_fraction),
                   origin_z_m=float(origin_z_m), minimum_bucket_z_m=float(minimum_bucket_z_m),
                   contact_established=contact_established)
        self.rows.append(row)
        self.last_frame = frame
        rows = list(self.rows)
        cfg = self.config
        complete = len(rows) == self.rows.maxlen
        closed = complete and all(r['closure_fraction'] >= cfg.closure_fraction for r in rows)
        contacted = complete and all(r['contact_established'] for r in rows)
        rising = complete
        maximum_drawdown = {}
        for key in ('origin_z_m', 'minimum_bucket_z_m'):
            peak = rows[0][key]
            drawdown = 0.
            for r in rows:
                peak = max(peak, r[key])
                drawdown = max(drawdown, peak-r[key])
            maximum_drawdown[key] = drawdown
            rising = rising and drawdown <= cfg.maximum_drawdown_m
        if complete:
            rising = rising and all(
                rows[i]['origin_z_m']-rows[i-cfg.trend_frames]['origin_z_m'] > cfg.trend_rise_m
                for i in range(cfg.trend_frames, len(rows)))
        height = minimum_bucket_z_m-self.initial_soil_top_m
        return dict(ready=bool(closed and contacted and rising), frame_end=frame,
                    first_available_decision_frame=frame+1,
                    closure_held=closed, contact_history_complete=contacted,
                    sustained_actual_rise=bool(rising), height_is_diagnostic_only=True,
                    observed_span_seconds=(rows[-1]['frame']-rows[0]['frame'])/cfg.hz,
                    bucket_above_initial_soil_m=height,
                    maximum_drawdown_m=maximum_drawdown,
                    initial_soil_top_m=self.initial_soil_top_m,
                    source_sha256=self.source_sha256)

    def snapshot(self):
        return dict(schema=self.schema, config=asdict(self.config),
                    initial_soil_top_m=self.initial_soil_top_m,
                    source_sha256=self.source_sha256, rows=list(self.rows),
                    last_frame=self.last_frame)

    @classmethod
    def restore(cls, snapshot):
        if snapshot['schema'] != cls.schema:
            raise ValueError('carry_handoff_schema')
        obj = cls(initial_soil_top_m=snapshot['initial_soil_top_m'],
                  source_sha256=snapshot['source_sha256'],
                  config=CarryHandoffConfig(**snapshot['config']))
        rows = snapshot['rows']
        if len(rows) > obj.rows.maxlen:
            raise ValueError('carry_handoff_history_length')
        for row in rows:
            obj.observe_end(**row)
        if obj.last_frame != snapshot['last_frame']:
            raise ValueError('carry_handoff_history_endpoint')
        return obj
