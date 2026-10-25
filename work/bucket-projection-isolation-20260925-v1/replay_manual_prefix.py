"""Bounded new-physics replay of the saved operator input; never edit old cache."""
import json, time, hashlib, traceback
from pathlib import Path
import numpy as np
from layered_47_cycle_runtime import CycleRuntime, REPO
from view_contact_bc_probe import ProbeDisplayCache
from live_bucket_fill import MixedFillEstimator, load_config
import run_layered_47_sparse_manual_contact_episode as manual

work=Path(__file__).resolve().parent
source=REPO/'runs/contact-multisource-validation-20260924-v1/first-initialization-v3/prepoint.npz'
old=REPO/'runs/layered-47-sparse-manual-contact-v1/manual-contact-20260925-143841-2474275'
out=work/'manual-prefix-1800'
inputs=[json.loads(x) for x in (old/'input.ndjson').read_text().splitlines()][:1800]
assert len(inputs)==1800
report={'status':'running','frames_limit':1800,'wall_limit_seconds':600,'old_run':str(old),
        'input_sha256':hashlib.sha256((old/'input.ndjson').read_bytes()).hexdigest(),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'purpose':'new physics validation of saved manual inputs, not identical trajectory replay',
        'measurements':[],'projection_changes':[]}
cache=None;started=time.perf_counter()
with CycleRuntime(source,out,enable_truck=True,freeze_passive_jp=True,bucket_motion_consistency=True) as r:
    report['restore']=r.restored
    report['bindings']=r.bindings
    verts=manual._cache_rock_vertices(r.root,r.records)
    estimator=MixedFillEstimator(load_config(REPO/'configs/v4/live-bucket-fill-local-v1.json'),[x['volume_m3'] for x in r.records],verts)
    cache=ProbeDisplayCache(out/'display',1800,r.arrays['particle_q'].shape,
        dict(root=str(r.root),records=r.records,selected_cut=r.metadata['selected_cut'],particle_spacing_m=r.base['mpm']['particle_spacing_m'],showcase_continuous=True))
    cache.capture(0,manual.compact_state(r.bridge.state),r.fixture.state.particle_q.numpy())
    try:
        for f,entry in enumerate(inputs,1):
            if time.perf_counter()-started>600:raise TimeoutError('manual_prefix_wall_budget')
            requested=np.asarray(entry['applied_command_si'])
            command=r.projector.project(r.bridge.state['mechanism_qpos'],requested)
            if not np.array_equal(requested,command):
                report['projection_changes'].append({'frame':f,'reason':r.projector.reason,'requested':requested.tolist(),'applied':command.tolist()})
            row=r.step(command)
            q=r.fixture.state.particle_q.numpy()
            cache.capture(f,manual.compact_state(r.bridge.state),q)
            if f%60==0:
                pose=r.assembly.assemble(np.asarray(r.bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
                fill=estimator.measure_snapshot({'frame':f,'soil':q,'pose':pose,'rocks':np.asarray([x['qpos'] for x in r.bridge.state['rocks']])})
                report['measurements'].append(fill)
                cache.positions.flush()
                print('MANUAL',f,'wall',round(time.perf_counter()-started,1),'fill',round(fill['fill_percent'],2),'valid',fill['valid_estimate'],flush=True)
            if f in (3,6,1080,1680,1800):
                report.setdefault('checkpoints',{})[str(f)]=r.save(out/f'state-{f:06d}.npz')
        report['status']='bounded_manual_prefix_passed'
    except BaseException as e:
        report.update(status='failed',error=repr(e),traceback=traceback.format_exc())
        raise
    finally:
        report['frames']=r.frames;report['wall_seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        if cache is not None:
            report['display_cache']=cache.finish(report['status'])
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
