"""Explicit saved-source identities; no simulator import or implicit source fallback."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
W=Path(__file__).resolve().parent
RUN=ROOT/'runs/night-dual-search-20260925-v1'
FIRST=ROOT/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
FIRST_SHA='700f76be970c5754bde905837faf642cca0317fef097e31d17cf48f008ba0545'
FIRST_GATE=ROOT/'runs/contact-multisource-lift-20260924-v1/first-candidate/gate-120.npz'
FIRST_GATE_SHA='b05c0ecd5fb38ee3e65e154057d093568874ee02dedc54fa80af71e5b042ed09'
FIRST_CODE=ROOT/'work/contact-multisource-lift-20260924-v1'
SECOND_CODE=ROOT/'work/night-parameter-search-20260925-v2'
SECOND_RUN=ROOT/'runs/night-parameter-search-20260925-v2'
SECOND=ROOT/'runs/contact-next-prepoint-20260925-v1/physical/prepoint.npz'
SECOND_SHA='ab4f529501c3991022f6e72a276d4d7b48def9a888fcd85db3148356c73e420e'
PHYS='/home/qunwang/codex-work/high-fidelity-soil-newton-mpm-20260907/.venv/bin/python'
TRAIN='/home/qunwang/codex-work/rl-four-axis-training-2026-09-16/.venv/bin/python'
POLICY=ROOT/'runs/contact-phase-bc-pilot-v1/pilot.pt'
POLICY_SHA='b4bbc98cf472bcad23f89e014a025cfc933c36091ada91ced4c4cc9680d94095'
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def first_source():
 if sha(FIRST)!=FIRST_SHA or sha(FIRST_GATE)!=FIRST_GATE_SHA:raise ValueError('first_source_or_reference_changed')
 m=read(str(FIRST)+'.json')
 if (m['schema']!='layered_47_cycle_checkpoint_v1' or m['stage']!='prepoint' or m['cycle_index']!=1
     or m['truck_enabled'] is not False or m.get('training_reset') is not None
     or m['controller_state']['native_state_kind']!='full' or m['checkpoint_sha256']!=FIRST_SHA):
  raise ValueError('first_source_contract')
 return m

def second_evidence():
 """Reuse the UNMODIFIED second-source pipeline, not validation of a new adapter."""
 manifest=read(SECOND_CODE/'manifest.json');check=read(SECOND_RUN/'precheck.json')
 if (check['status']!='passed' or check['physical_frames']!=120 or check['manifest_sha256']!=sha(SECOND_CODE/'manifest.json')
     or manifest['source_sha256']!=SECOND_SHA or sha(SECOND)!=SECOND_SHA):raise ValueError('second_precheck_identity')
 for p,h in {**manifest['bindings'],**check['bindings']}.items():
  if sha(p)!=h:raise ValueError('second_bound_input_changed:'+p)
 return dict(status='reused_unchanged_second_pipeline',precheck_sha256=sha(SECOND_RUN/'precheck.json'),
   source_sha256=SECOND_SHA,prior_check_seconds=check['wall_seconds'],new_physical_frames=0)


SOURCE_SPECS={
 'first':dict(source=FIRST,sha=FIRST_SHA,gate=FIRST_GATE,gate_sha=FIRST_GATE_SHA,protocol=FIRST_CODE/'first-protocol.json',cycle=1,truck=False),
 'second':dict(source=SECOND,sha=SECOND_SHA,gate=ROOT/'runs/contact-next-scoop-20260925-v1/candidate/gate-120.npz',gate_sha='97e4212b7eec410211383a577b2e3221242bfd0695c5315699fb6737c859cbdd',protocol=ROOT/'work/contact-next-scoop-20260925-v1/protocol.json',cycle=2,truck=True)}
