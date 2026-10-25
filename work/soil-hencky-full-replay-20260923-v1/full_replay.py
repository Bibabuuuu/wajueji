"""Run the frozen D-127 harness through all 1329 original commands.

The validated v2 harness is transformed in memory so its frozen evidence stays
byte-identical. Exact replacement assertions make any upstream drift fail before
physics starts.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V2 = HERE.parent / "soil-hencky-candidate-20260923-v2"
sys.path.insert(0, str(V2))

source = (V2 / "replay.py").read_text()
replacements = {
    "[:600]": "[:1329]",
    "if frame in (277,540,600):save('candidate')": "if frame in (600,800,1000,1193,1329):save('candidate')",
    "'candidate_short_prefix_passed' if frame==600": "'candidate_full_replay_passed' if frame==1329",
    "(out/'preflight-report.json')": "(Path('/home/qunwang/wajueji/work/soil-hencky-candidate-20260923-v2/preflight-report.json'))",
}
for old, new in replacements.items():
    if source.count(old) != 1:
        raise RuntimeError(f"frozen_harness_transform_mismatch: {old!r}")
    source = source.replace(old, new)

exec(compile(source, str(Path(__file__)), "exec"), {"__file__": str(Path(__file__)), "__name__": "__main__"})
