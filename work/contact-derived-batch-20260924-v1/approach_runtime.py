"""Original manual first-scoop initialization; not a converted cycle checkpoint.

The source has no native controller history. Just as in the original manual
runner, construct a fresh controller, restore the handoff particle/rigid/solver
arrays, and start with zero integral/curl/sweep state. No source is rewritten.
Cycle instrumentation adds full-state output for NEW derived checkpoints only.
"""
from contextlib import ExitStack
import hashlib,json,os
from pathlib import Path
import numpy as np
from layered_47_cycle_runtime import (CycleRuntime,REPO,BUNDLE,KINEMATICS,local_arguments,
    newton,wp,motion,manual,checkpoint,cleaned_manual_model,optimized_bridge,
    bucket_integral_bridge,active_curl_bridge,installed_cycle_bridge,
    RigidFrequencyCommandBridge,installed_material_guard,MovingBucketSweepRuntime,hencky,
    _load_kinematics)
import layered_47_sparse_approach_handoff_checkpoint as handoff

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class ApproachRuntime(CycleRuntime):
    def __init__(self,source,output,teacher_report):
        self.output=Path(output).resolve();self.output.mkdir(parents=True,exist_ok=False)
        source=Path(source).resolve();ap=source.parent/'report.json';approach=json.loads(ap.read_text())
        self.arrays,source_metadata,identity=handoff.load_checkpoint_readonly(source,Path(str(source)+'.json'),approach['handoff_provenance'])
        assert identity==teacher_report['handoff_checkpoint']==approach['handoff_checkpoint']
        audit=teacher_report['restore_audit'];assert sha(audit['path'])==audit['sha256']
        assert teacher_report['bucket_integral_candidate']['enabled'] and teacher_report['active_curl_candidate']['enabled']
        assert teacher_report['hencky_update_candidate']['enabled'] and teacher_report['moving_bucket_sweep_candidate']['enabled']
        assert all(teacher_report['checks'].values())
        self.frames=0;self.contexts=ExitStack();self.bridge=None;self.truck_enabled=False
        self.args=local_arguments();self.root=BUNDLE/'repo'
        os.environ['EXCAVATOR_MUJOCO_MESH_DIR']=str(BUNDLE/'excavator-resources/mesh')
        os.environ['EXCAVATOR_MUJOCO_TEXTURE_DIR']=str(BUNDLE/'excavator-resources/texture')
        loader,identity47=motion.probe.select(47)
        self.base=loader(self.root/'configs/v4/layered-47-excavation-preflight-v1.json',self.root)
        self.base['input_checkpoint']=json.loads((REPO/'configs/v4/layered-47-interactive-proxy-benchmark-v1.json').read_text())['input_checkpoint']
        self.base['probe']={**self.base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
        manifest=json.loads((self.root/self.base['physical_geometry']['manifest_path']).read_text())
        self.records=[r for r in manifest['rocks'] if r['layer_id'] in identity47['layers']]
        assert [r['rock_id'] for r in self.records]==identity47['expected_ids']
        _,self.proxy=motion.probe.p47._load_bound_json(self.root,self.base['open_bucket_proxy'],'proxy')
        self.solver_profile=approach['solver_profile']
        self.assembly,self.chain=_load_kinematics(KINEMATICS,KINEMATICS/'scene/input/model/A.xml',KINEMATICS/'scene/input/mesh')
        self.projector=manual.ContactCommandProjector(self.chain)
        names=('layered_47_cycle_bridge.py','layered_47_cycle_runtime.py','layered_47_cycle_checkpoint.py',
               'bucket_active_curl_candidate.h','bucket_integral_candidate.h','soil_hencky_update_candidate.py',
               'moving_bucket_sweep_candidate.py','moving_bucket_sweep_runtime.py','mpm_static_contact_ownership.py')
        self.bindings={name:sha(REPO/'src/v4'/name) for name in names}
        self.metadata=dict(source_identity=identity,lineage_root=identity['sha256'],historical_native_history_missing=True,
                           cycle_index=1,selected_cut=teacher_report['selected_cut'])
        self.initialization_audit=dict(schema='original_manual_approach_initialization_v1',
            source_schema=source_metadata['schema'],source_schema_changed=False,
            native_controller_history='cold initialization as in original manual episode; not restored native history',
            inputs=[dict(path=str(p),sha256=sha(p)) for p in (Path(__file__),ap,Path(str(source)+'.json'),Path(audit['path']),REPO/'src/v4/run_layered_47_sparse_manual_contact_episode.py')])

    def __enter__(self):
        try:
            newton.use_coord_layout_targets=True;wp.init();wp.set_device(self.args.device)
            enter=self.contexts.enter_context
            enter(cleaned_manual_model());enter(optimized_bridge(self.args.mechanism_header_dir,self.output/'optimized'))
            enter(bucket_integral_bridge(True,self.output/'integral'))
            enter(active_curl_bridge(True,self.output/'active',initialize_from_d120=False))
            enter(installed_cycle_bridge(self.output/'cycle'));enter(hencky.installed_candidate())
            work=self.output/'runtime';work.mkdir()
            # Same cold-native and normal handoff grid construction as manual
            # approach mode. No cycle restore or sparse-topology monkey patch.
            self.fixture,held,self.restored=motion.audit.restore_candidate_runtime(
                self.args,self.base,self.root,self.records,self.arrays,self.proxy,self.solver_profile,work)
            self.bridge=RigidFrequencyCommandBridge(held.bridge,4800)
            assert self.restored['exact']
            self.guard=enter(installed_material_guard(self.fixture.solver))
            self.sweep=MovingBucketSweepRuntime(self.fixture,self.bridge,self.proxy,self.base['mpm']['material']['friction'])
            self.captured={};original=self.fixture.step
            def step_mpm(dt,origins):
                measured,particles=self.sweep.step_mpm(original,dt,origins);self.captured['wrenches']=measured
                return measured,particles
            self.fixture.step=step_mpm
            self.initialization_audit['original_restore_exact']=bool(self.restored['exact'])
            return self
        except BaseException:
            self.close();raise
