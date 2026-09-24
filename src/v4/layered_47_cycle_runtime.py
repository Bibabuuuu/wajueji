"""Shared physical runtime for bounded continuation and new manual prepoints."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import newton
import warp as wp
import run_layered_47_sparse_manual_contact_episode as manual
import run_layered_47_sparse_motion as motion
import layered_47_cycle_checkpoint as checkpoint
from layered_47_cycle_bridge import installed_cycle_bridge, controller_snapshot
from layered_47_optimized_bridge import optimized_bridge
from layered_47_bucket_integral_candidate import bucket_integral_bridge
from layered_47_active_curl_candidate import active_curl_bridge
from legacy_push_shelf_cleanup import cleaned_manual_model
from layered_47_loaded_handoff_checkpoint import restored_sparse_grid_topology
from layered_47_rigid_frequency_candidate import RigidFrequencyCommandBridge
from layered_excavation_approach_planner import _load_kinematics
from moving_bucket_sweep_runtime import MovingBucketSweepRuntime
from soil_hencky_update_guard import installed_material_guard
import soil_hencky_update_candidate as hencky
import layered_47_truck_physics_candidate as truck

REPO = Path(__file__).resolve().parents[2]
BUNDLE = Path('/home/qunwang/Documents/Codex/2026-09-22/wajueji-continuation/work/a30-layered-47-sparse-motion-v1')
KINEMATICS = Path('/home/qunwang/codex-work/small-soil-20260910-v17-performance')
APPROACH = REPO/'runs/layered-47-sparse-rear-lower-approach-handoff-v1/rear-lower-approach-20260923-140643-648089'


def local_arguments():
    return SimpleNamespace(device='cuda:0', cxx='g++',
        mujoco_include=str(BUNDLE/'mujoco/include'), mujoco_library_dir=str(BUNDLE/'mujoco/lib'),
        mechanism_header_dir=str(BUNDLE/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8'),
        controller_header_dir=str(BUNDLE/'repo/evidence/autonomous-multiscoop-2026-09-14-v1/mechanism-auto8/excavator_simulator'))


class CycleRuntime:
    def __init__(self, source, output, *, enable_truck=None, freeze_passive_jp=False):
        self.output=Path(output).resolve(); self.output.mkdir(parents=True, exist_ok=False)
        self.arrays,self.metadata=checkpoint.load(source)
        if freeze_passive_jp:
            from soil_passive_jp_history import POLICY
            self.metadata['material_history_policy'] = POLICY
        self.truck_enabled = self.metadata['truck_enabled'] if enable_truck is None else enable_truck
        if self.metadata['truck_enabled'] and not self.truck_enabled:
            raise ValueError('cycle_cannot_remove_truck')
        self.frames=0;self.contexts=ExitStack();self.bridge=None
        self.args=local_arguments();self.root=BUNDLE/'repo'
        os.environ['EXCAVATOR_MUJOCO_MESH_DIR']=str(BUNDLE/'excavator-resources/mesh')
        os.environ['EXCAVATOR_MUJOCO_TEXTURE_DIR']=str(BUNDLE/'excavator-resources/texture')
        loader,identity=motion.probe.select(47)
        self.base=loader(self.root/'configs/v4/layered-47-excavation-preflight-v1.json',self.root)
        self.base['input_checkpoint']=json.loads((REPO/'configs/v4/layered-47-interactive-proxy-benchmark-v1.json').read_text())['input_checkpoint']
        self.base['probe']={**self.base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':80}
        manifest=json.loads((self.root/self.base['physical_geometry']['manifest_path']).read_text())
        self.records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
        if [r['rock_id'] for r in self.records] != identity['expected_ids']: raise ValueError('cycle_rock_order')
        _,self.proxy=motion.probe.p47._load_bound_json(self.root,self.base['open_bucket_proxy'],'proxy')
        self.solver_profile=self.metadata.get('solver_profile') or json.loads((APPROACH/'report.json').read_text())['solver_profile']
        self.assembly,self.chain=_load_kinematics(str(KINEMATICS),str(KINEMATICS/'scene/input/model/A.xml'),str(KINEMATICS/'scene/input/mesh'))
        self.projector=manual.ContactCommandProjector(self.chain)
        names=('layered_47_cycle_bridge.py','layered_47_cycle_runtime.py','layered_47_cycle_checkpoint.py',
               'bucket_active_curl_candidate.h','bucket_integral_candidate.h','soil_hencky_update_candidate.py',
               'moving_bucket_sweep_candidate.py','moving_bucket_sweep_runtime.py','mpm_static_contact_ownership.py')
        self.bindings={name:motion.probe.bench.sha256(Path(__file__).with_name(name)) for name in names}
        if self.metadata.get('material_history_policy'):
            name = 'soil_passive_jp_history.py'
            self.bindings[name] = motion.probe.bench.sha256(Path(__file__).with_name(name))

    def __enter__(self):
        try:
            newton.use_coord_layout_targets=True;wp.init();wp.set_device(self.args.device)
            enter=self.contexts.enter_context
            enter(cleaned_manual_model())
            enter(optimized_bridge(self.args.mechanism_header_dir,self.output/'optimized'))
            enter(bucket_integral_bridge(True,self.output/'integral'))
            enter(active_curl_bridge(True,self.output/'active',initialize_from_d120=True))
            enter(installed_cycle_bridge(self.output/'cycle'))
            fixtures=None
            if self.truck_enabled:
                fixtures=enter(truck.installed_candidate(REPO/'configs/v4/layered-47-truck-physics-v1.json',REPO))
            enter(hencky.installed_candidate())
            work=self.output/'runtime';work.mkdir()
            with restored_sparse_grid_topology(self.arrays):
                self.fixture,held,self.restored=motion.audit.restore_candidate_runtime(
                    self.args,self.base,self.root,self.records,self.arrays,self.proxy,self.solver_profile,
                    work,self.metadata['controller_state'],fixtures)
            # MuJoCo emit preserves the pre-integration cvel of the last native
            # step. mj_forward during restore recomputes it at the endpoint.
            # MPM consumes the emitted values, so preserve that exact frame
            # boundary interface separately from authoritative native qvel.
            for i,rock in enumerate(held.bridge.state['rocks']):
                rock['linear_velocity_world_m_s']=self.arrays['rigid_reported_linear_velocity'][i].tolist()
                rock['angular_velocity_world_rad_s']=self.arrays['rigid_reported_angular_velocity'][i].tolist()
            for key in ('bucket_linear_velocity_world_m_s','bucket_angular_velocity_world_rad_s'):
                held.bridge.state[key]=list(self.metadata['controller_state'][key])
            self.fixture.assign(held.bridge.state)
            self.bridge=RigidFrequencyCommandBridge(held.bridge,4800)
            self.guard=enter(installed_material_guard(self.fixture.solver))
            if self.metadata.get('material_history_policy'):
                from soil_passive_jp_history import installed_passive_jp
                self.passive_jp_policy=enter(installed_passive_jp(self.fixture.solver,self.fixture.state))
            self.sweep=MovingBucketSweepRuntime(self.fixture,self.bridge,self.proxy,self.base['mpm']['material']['friction'])
            checkpoint.restore_sweep(self.sweep,self.metadata['moving_bucket_sweep_state'])
            expected=self.metadata['controller_state'];actual=controller_snapshot(self.bridge.state)
            errors={}
            for key,value in expected.items():
                if key=='native_state_kind':continue
                if key.startswith('native_') and expected['native_state_kind']!='full' and key not in ('native_time_s','native_last_time_s','native_timestep_s'):continue
                errors[key]=float(np.max(np.abs(np.asarray(value)-np.asarray(actual[key]))))
            self.restored['cycle_errors']=errors
            if max(errors.values())>1e-12:raise ValueError('cycle_native_restore_inexact:'+str(errors))
            self.captured={};original=self.fixture.step
            def step_mpm(dt,origins):
                measured,particles=self.sweep.step_mpm(original,dt,origins)
                self.captured['wrenches']=measured
                return measured,particles
            self.fixture.step=step_mpm
            return self
        except BaseException:
            self.close();raise

    def step(self, command):
        command=np.asarray(command,dtype=float)
        if command.shape!=(4,) or not np.isfinite(command).all() or np.any(np.abs(command)>manual.COMMAND_LIMITS_SI+1e-15):
            raise ValueError('cycle_command_envelope')
        self.bridge.command=command
        self.guard.prepare_frame()
        row,_=motion.probe.bench._advance_frame(self.fixture,self.bridge,self.base,self.frames+1)
        row['moving_bucket_sweep']=self.sweep.finish_frame()
        row['hencky_material_guard']=self.guard.summarize(self.fixture.state)
        q=self.fixture.state.particle_q.numpy();v=self.fixture.state.particle_qd.numpy()
        row.update(applied_command_si=command.tolist(),
            finite=bool(row['finite'] and np.isfinite(q).all() and np.isfinite(v).all() and motion.probe.bench._finite_tree(self.bridge.state)),
            joint_bounds_ok=self.projector.physical_bounds_ok(np.asarray(self.bridge.state['mechanism_qpos'])),
            bucket_mpm_wrench_finite=all(np.isfinite(x).all() for x in self.captured['wrenches']['bucket'].values()),
            rigid_contact_substeps=self.bridge.state.get('contact_substeps',{}),
            after_state=manual.compact_state(self.bridge.state))
        row['hard_guard_checks']=manual.hard_guard_checks(row,allow_truck=self.truck_enabled)
        row['hard_guard']=all(row['hard_guard_checks'].values())
        self.frames+=1
        manual.append_json(self.output/'frames.ndjson',row)
        if not row['hard_guard']:raise RuntimeError('cycle_hard_guard:'+str(row['hard_guard_checks']))
        return row

    def capture(self):
        return motion.probe.snapshots.capture_arrays(self.fixture,self.bridge.state,float(self.arrays['simulation_time_s'][0])+self.frames/60)

    def save(self,path,stage='hold',cycle_index=None,selected_cut=None):
        return checkpoint.save(path,self.capture(),controller_snapshot(self.bridge.state),self.sweep.diagnostic_state(),
            parent=self.metadata,stage=stage,cycle_index=cycle_index or self.metadata['cycle_index'],
            truck_enabled=self.truck_enabled,solver_profile=self.solver_profile,runtime_bindings=self.bindings,
            selected_cut=selected_cut if selected_cut is not None else self.metadata.get('selected_cut'))

    def close(self):
        if self.bridge is not None:self.bridge.close();self.bridge=None
        self.contexts.close()

    def __exit__(self,*exc):
        if exc[0] is not None and self.bridge is not None and hasattr(self, 'sweep'):
            try:
                saved=self.save(self.output/'failure.npz',stage='failure')
                (self.output/'failure.json').write_text(json.dumps(dict(error=str(exc[1]),checkpoint=saved),indent=2))
            except Exception as error:
                (self.output/'failure-save-error.txt').write_text(repr(error))
        self.close()
