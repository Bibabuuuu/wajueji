"""Local-only, bounded 60Hz practice; never write a training episode.

Uses the audited v3 state, strict terrain guards, and opt-in D-107 bridge.
Legacy 120Hz practice remains untouched for rollback.
"""
import json
from pathlib import Path
import signal
import tempfile
import time
import traceback

import newton
import numpy as np
import warp as wp

from layered_47_optimized_bridge import optimized_bridge, parser
import run_layered_47_sparse_motion as motion
import run_layered_47_interactive_proxy_practice as display_helpers
from manual_action_input import GamepadFourAxisState, KeyboardFourAxisState, load_manual_input_config
from contracts import ACTION_SPEED_LIMIT_SI
from layered_excavation_approach_planner import _load_kinematics, _proxy_local_vertices
import layered_47_rigid_frequency_candidate as rigid_candidate

LOCAL_SHA = 'a3a78553efcf0cbb0018951b93640a95c2e7056558311b51472e255affa8f4f7'
NORMAL_RATES = np.asarray(ACTION_SPEED_LIMIT_SI)


class NoncontactGuard:
    """Sampled preview only; never changes the actual rigid/MPM state."""
    def __init__(self, assembly, chain, proxy):
        self.assembly, self.chain = assembly, chain
        self.vertices = _proxy_local_vertices(proxy)
        self.reason = None

    def physical_bounds_ok(self, q):
        return bool(np.isfinite(q).all() and abs(q[motion.AXES[0]]) <= 2.992
            and all(lo <= q[i] <= hi for i, (lo, hi) in self.chain.limits.items()))

    def project(self, q, command, terrain_top):
        self.reason = None
        if not np.any(command):
            return command.copy()
        self.chain.q = np.asarray(q).copy()
        try:
            # 150ms preview exceeds nominal full-speed target braking times
            # (80ms swing, <=67.5ms cylinders); not a collision proof.
            for _ in range(3):
                predicted, _ = self.chain.advance(command, .05)
                if not self.physical_bounds_ok(predicted):
                    raise ValueError('joint_working_range')
                pose = self.assembly.assemble(predicted, include_meshes=False)[1]['bucket']
                lowest = float((self.vertices @ pose[:3,:3].T + pose[:3,3])[:,2].min())
                if not np.isfinite(lowest) or lowest-terrain_top < .17:
                    raise ValueError('noncontact_clearance_preview')
        except ValueError as error:
            self.reason = str(error)
            return np.zeros(4)
        return command.copy()


class InputGate:
    """Neutral rearm and terminal stops; normal passes all four bounded axes."""
    def __init__(self, profile='micro'):
        if profile not in ('micro', 'normal'):
            raise ValueError('practice_invalid_profile')
        self.profile = profile
        self.rates = motion.RATES if profile == 'micro' else NORMAL_RATES
        self.armed = False
        self.stop_reason = None

    def command(self, action, offset, *, paused=False, focused=True, connected=True, estop=False):
        values = np.asarray(action, dtype=float)
        offset = np.asarray(offset, dtype=float)
        if values.shape!=(4,) or offset.shape!=(4,) or not np.isfinite(values).all() or not np.isfinite(offset).all() or np.any(np.abs(values)>1):
            raise ValueError('practice_invalid_input')
        if estop or not connected:
            self.stop_reason = 'manual_emergency_stop' if estop else 'gamepad_disconnect'
        if self.stop_reason:
            return np.zeros(4)
        if paused or not focused:
            self.armed = False
            return np.zeros(4)
        if not self.armed:
            self.armed = not np.any(values)
            return np.zeros(4)
        if self.profile == 'normal':
            return values*self.rates
        # Keep the historical micro profile as an explicit single-axis rollback.
        axis = int(np.argmax(np.abs(values)))
        command = np.zeros(4)
        command[axis] = values[axis]*self.rates[axis]
        if self.profile == 'micro' and abs(offset[axis]) >= .8*motion.ENVELOPE[axis] and command[axis]*offset[axis]>0:
            command[axis] = 0
        return command


def smoke_schedule(profile='micro'):
    rates = motion.RATES if profile == 'micro' else NORMAL_RATES
    yield 'hold', np.zeros(4), None, 0
    for axis, name in enumerate(motion.NAMES):
        for sign in ((-1, 1) if profile == 'normal' and axis == 1 else (1, -1)):
            command = np.zeros(4)
            command[axis] = sign*rates[axis]
            yield f'{name}_{sign:+d}', command, axis, sign
            yield f'{name}_{sign:+d}_stop', np.zeros(4), axis, 0


def combined_smoke_schedule():
    """Full-scale K+O equivalent, then all axes; reverse each and stop."""
    yield 'hold', np.zeros(4)
    for name, action in (('boom_arm', [0,-1,1,0]), ('four_axes', [1,-1,1,1])):
        for sign in (1,-1):
            yield f'{name}_{sign:+d}', sign*np.asarray(action,dtype=float)
            yield f'{name}_{sign:+d}_stop', np.zeros(4)


def run(args):
    root, output = Path(args.repository_root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    report = dict(schema='layered_47_optimized_practice_v1', status='initializing', hz=60,
        scope=dict(practice_only=True, noncontact_only=True, training_eligible=False,
            manual_collection_eligible=False, formal_checkpoint_written=False),
        runner_sha256=motion.probe.bench.sha256(Path(__file__)),
        device=args.device, mode='combined_smoke' if args.combined_smoke else 'smoke' if args.smoke else 'ui_smoke' if args.ui_smoke else args.manual_input)
    motion.probe.bench._write_json(output/'report.json', report)
    fixture = bridge = viewer = controller = None
    rows, phases = [], []
    stop_reason = None
    gate = InputGate(args.control_profile)
    normal = args.control_profile == 'normal'
    maximum_wall, maximum_frames = (1800, 3600) if normal else (600, 600)
    initial_axes = None
    before = source = None
    prior_sigterm = None

    def log(name, value):
        motion.probe.bench._append_journal(output/name, value)

    def snapshot(label):
        if fixture is None or bridge is None or not rows:
            return
        values = motion.probe.snapshots.capture_arrays(fixture, bridge.state,
            float(arrays['simulation_time_s'][0])+len(rows)/60)
        info = motion.probe.snapshots.write_checkpoint_atomic(output/f'{label}-{len(rows):06d}.diagnostic.npz', values)
        metadata = dict(status='practice_diagnostic_only_not_resume_qualified', source=before,
            hz=60, rigid_internal_hz=args.rigid_internal_hz, frames=len(rows), scope=report['scope'], optimization=report.get('optimization'), **info)
        motion.probe.bench._write_json(Path(info['path']+'.json'), metadata)
        info['metadata_sha256'] = motion.probe.bench.sha256(Path(info['path']+'.json'))
        report.setdefault('diagnostic_snapshots', []).append(info)
        log('progress.ndjson', dict(event='snapshot', **info))

    try:
        rigid_candidate.validate_run_mode(args.rigid_internal_hz,smoke=args.smoke,combined_smoke=args.combined_smoke)
        if args.combined_smoke and not normal:
            raise ValueError('combined_smoke_requires_normal_profile')
        if args.checkpoint_sha256 != LOCAL_SHA:
            raise ValueError('practice_local_checkpoint_identity')
        loader, identity = motion.probe.select(47)
        base = loader(args.config, root)
        # Explicit local diagnostic-domain adaptation; never used in A30 runner.
        source_config = Path(args.local_source_config).resolve()
        base['input_checkpoint'] = json.loads(source_config.read_text())['input_checkpoint']
        candidate_path, audit_path = Path(args.candidate_report).resolve(), Path(args.resume_report).resolve()
        arrays, source, before = motion.audit.load_candidate(candidate_path, base, 47, 60)
        motion.audit.validate_seed_audit(json.loads(audit_path.read_text()),candidate_path,source,before,47,60)
        if before['sha256']!=LOCAL_SHA or source['config_sha256']!=motion.probe.bench.sha256(Path(args.config)):
            raise ValueError('practice_source_binding')
        if motion.probe.bench.checkpoint_identity(source['source_checkpoint']['path'])!=source['source_checkpoint']:
            raise ValueError('practice_original_source_changed')
        original_metadata = Path(source['source_checkpoint']['path']+'.json')
        if motion.probe.bench.sha256(original_metadata)!=source['metadata_sha256']:
            raise ValueError('practice_original_metadata_changed')
        display=json.loads(Path(args.display_config).read_text())['display']
        binding_paths = [candidate_path,audit_path,source_config,Path(args.manual_config),Path(args.config),Path(args.display_config),original_metadata,Path(rigid_candidate.__file__),Path(display_helpers.__file__).with_name('soil_surface.py')]
        bindings = {str(p):motion.probe.bench.sha256(p) for p in binding_paths}
        report.update(source_checkpoint=before,input_bindings=bindings,
            acquisition_domain='local_47_rejected_ancestry_practice_only',
            protocol=dict(hz=60,rigid_internal_hz=args.rigid_internal_hz,
                rigid_substeps_per_mpm_frame=args.rigid_internal_hz//60,
                rigid_substeps_per_bridge_half=args.rigid_internal_hz//120,
                rigid_frequency_candidate_only=args.rigid_internal_hz!=4800,control_profile=args.control_profile,
                axis_envelope=None if normal else motion.ENVELOPE.tolist(),
                maximum_command_si=gate.rates.tolist(),single_axis_only=not normal,
                minimum_clearance_m=.15,maximum_wall_seconds=maximum_wall,maximum_simulation_seconds=maximum_frames/60,
                start_paused=True,snapshot_every_frames=60,
                soil_rendering=dict(mode='gpu_density_surface_with_loose_particles',
                    surface_voxel_m=display['surface_voxel_m'],
                    surface_update_interval_frames=display['surface_update_interval_frames'],
                    display_only=True),
                postsolve_projection=dict(
                    enabled=True,
                    reaction_routing='unattributed_diagnostic_not_applied_to_mujoco',
                    continuous_collision_proven=False)))
        manifest=json.loads((root/base['physical_geometry']['manifest_path']).read_text())
        records=[r for r in manifest['rocks'] if r['layer_id'] in identity['layers']]
        if [r['rock_id'] for r in records]!=identity['expected_ids']:
            raise ValueError('practice_record_order')
        radii=[]
        for r in records:
            vertices,_=motion._read_obj(root/'assets/v4/mixed-material-r0'/(r['prototype_id']+'.obj'))
            radii.append(float(np.linalg.norm(vertices,axis=1).max())*r['equivalent_diameter_m'])
        _,proxy=motion.probe.p47._load_bound_json(root,base['open_bucket_proxy'],'proxy')
        base['probe']={**base['probe'],'output_dt_s':1/60,'mujoco_internal_substeps':args.rigid_internal_hz//60}
        manual_config=load_manual_input_config(args.manual_config)
        initial_axes=arrays['mechanism_qpos'][motion.AXES]
        protection = None
        if normal:
            import sys
            backend_parent = Path(display['backend_root']).parent
            python_dir = backend_parent/'compact-bank-2026-09-11'
            model_path = backend_parent/'small-soil-20260910-v17-performance/scene/input/model/A.xml'
            # The historical backend splits these three modules across two dirs.
            sys.path.append(str(model_path.parents[3]))
            assembly, chain = _load_kinematics(python_dir, model_path, model_path.parent.parent/'mesh')
            if list(chain.axes) != list(motion.AXES):
                raise ValueError('practice_kinematic_axes_mismatch')
            protection = NoncontactGuard(assembly, chain, proxy)
            report['protocol'].update(preview_seconds=.15,preview_clearance_m=.17,
                continuous_collision_proven=False,joint_limits={str(i):v.tolist() for i,v in chain.limits.items()})
            paths = [model_path]+[Path(sys.modules[n].__file__) for n in ('closed_chain','scene_assembly','fast_kinematics')]
            bindings.update({str(p):motion.probe.bench.sha256(p) for p in paths})
        newton.use_coord_layout_targets=True
        wp.init()
        wp.set_device(args.device)
        with optimized_bridge(args.mechanism_header_dir,output/'build') as provenance, tempfile.TemporaryDirectory(prefix='optimized-practice-') as temporary:
            fixture,held,report['restore']=motion.audit.restore_candidate_runtime(args,base,root,records,arrays,proxy,source['solver_profile'],Path(temporary))
            bridge=rigid_candidate.RigidFrequencyCommandBridge(held.bridge,args.rigid_internal_hz)
            report['optimization']=provenance.copy()

            def clearance():
                return float(motion.probe.p47._bucket_world_vertices(proxy,bridge.state)[:,2].min())-motion.terrain_top(fixture,bridge,radii,base['mpm']['particle_spacing_m'])

            if clearance()<.15:
                raise ValueError('practice_initial_clearance')
            if protection is not None:
                pose=protection.assembly.assemble(np.asarray(bridge.state['mechanism_qpos']),include_meshes=False)[1]['bucket']
                predicted=protection.vertices @ pose[:3,:3].T + pose[:3,3]
                actual=motion.probe.p47._bucket_world_vertices(proxy,bridge.state)
                if predicted.shape != actual.shape or not np.allclose(predicted,actual,rtol=0,atol=1e-7):
                    raise ValueError('practice_prediction_geometry_mismatch')

            def step(command, phase):
                requested=np.asarray(command).copy()
                projection_started=time.perf_counter()
                if protection is not None:
                    command=protection.project(np.asarray(bridge.state['mechanism_qpos']),requested,
                        motion.terrain_top(fixture,bridge,radii,base['mpm']['particle_spacing_m']))
                projection_seconds=time.perf_counter()-projection_started
                bridge.command=np.asarray(command)
                row,_=motion.probe.bench._advance_frame(fixture,bridge,base,len(rows)+1)
                q=np.asarray(bridge.state['mechanism_qpos'])[motion.AXES]
                row.update(phase=phase,command_si=bridge.command.tolist(),requested_command_si=requested.tolist(),
                    command_guard_reason=protection.reason if protection is not None else None,
                    projection_wall_seconds=projection_seconds,axis_positions=q.tolist(),
                    axis_offset=(q-initial_axes).tolist(),
                    axis_velocity=np.asarray(bridge.state['mechanism_qvel'])[motion.AXES-1].tolist(),
                    finite=bool(row['finite'] and motion.probe.bench._finite_tree(bridge.state)),
                    bucket_vertical_clearance_m=clearance(),
                    bucket_linear_speed_m_s=float(np.linalg.norm(bridge.state['bucket_linear_velocity_world_m_s'])),
                    bucket_angular_speed_rad_s=float(np.linalg.norm(bridge.state['bucket_angular_velocity_world_rad_s'])),
                    rock_displacement_from_source_m=float(np.linalg.norm(np.asarray([r['qpos'][:3] for r in bridge.state['rocks']])-arrays['rigid_qpos'][:,:3],axis=1).max()))
                if protection is None:
                    motion.qualify(row)
                else:
                    motion.probe.classify_frame(row,'measured',0)
                    row['joint_bounds_ok']=protection.physical_bounds_ok(np.asarray(bridge.state['mechanism_qpos']))
                    row['motion_guard']=bool(row['strict'] and row['joint_bounds_ok'] and row['bucket_vertical_clearance_m']>=.15)
                row['stopped']=motion.stopped(row)
                row['first_step_continuous']=bool(rows or row['maximum_rock_position_step_m']<=.01/60*1.01)
                row['motion_guard']=bool(row['motion_guard'] and row['first_step_continuous'])
                rows.append(row)
                log('physics.ndjson',row)
                if len(rows)%60==0 and row['finite']:
                    snapshot('periodic')
                if len(rows)%30==0:
                    print(f"OPTIMIZED_PRACTICE frame={len(rows)} phase={phase} safe={row['motion_guard']} clearance={row['bucket_vertical_clearance_m']:.6f}",flush=True)
                if not row['motion_guard']:
                    raise RuntimeError('practice_physics_guard')

            if args.combined_smoke:
                report['protocol']['combined_smoke_phases']=[
                    dict(phase=name,normalized=action.tolist(),frames=30)
                    for name,action in combined_smoke_schedule()]
                for phase,action in combined_smoke_schedule():
                    start=np.asarray(bridge.state['mechanism_qpos'])[motion.AXES].copy()
                    expected_command=action*gate.rates
                    for _ in range(30):
                        offset=np.asarray(bridge.state['mechanism_qpos'])[motion.AXES]-initial_axes
                        command=gate.command(action,offset)
                        log('input.ndjson',dict(frame=len(rows)+1,normalized=action.tolist(),
                            device_gated_command_si=command.tolist(),input_device='scripted_combined_test',training_eligible=False))
                        step(command,phase)
                    delta=np.asarray(rows[-1]['axis_positions'])-start
                    axis_checks={motion.NAMES[i]:motion.motion_response(float(np.sign(action[i])*delta[i]),abs(expected_command[i])*.5)
                        for i in np.flatnonzero(action)}
                    commands_preserved=all(np.array_equal(row['command_si'],expected_command) for row in rows[-30:])
                    passed=commands_preserved and (all(axis_checks.values()) if axis_checks else all(row['stopped'] for row in rows[-15:]))
                    phases.append(dict(phase=phase,passed=bool(passed),axis_response=axis_checks,
                        axis_delta=delta.tolist(),commands_preserved=commands_preserved))
                    if not passed:
                        raise RuntimeError('practice_combined_response_or_stop')
                stop_reason='headless_combined_smoke_completed'
            elif args.smoke:
                for phase,command,axis,sign in smoke_schedule(args.control_profile):
                    start=np.asarray(bridge.state['mechanism_qpos'])[motion.AXES].copy()
                    for _ in range(30):
                        step(command,phase)
                    passed=(motion.motion_response(sign*(rows[-1]['axis_positions'][axis]-start[axis]),abs(command[axis])*.5)
                        if sign else all(row['stopped'] for row in rows[-15:]))
                    phases.append(dict(phase=phase,passed=bool(passed)))
                    if not passed:
                        raise RuntimeError('practice_smoke_response_or_stop')
                stop_reason='headless_bidirectional_smoke_completed'
            else:
                import pyglet
                import pyglet.display.xlib as pyglet_xlib
                from pyglet.window import key
                if not hasattr(pyglet_xlib.XlibScreen,'is_primary'):
                    pyglet_xlib.XlibScreen.is_primary=False
                assembly,display_model,display_state,names,to_pose=display_helpers._display_runtime(display)
                viewer=newton.viewer.ViewerGL(width=display['width'],height=display['height'],vsync=False,paused=True)
                viewer.set_model(display_model)
                viewer.picking_enabled=False
                viewer.renderer.set_title('47 ROCKS - 60Hz OPTIMIZED - NONCONTACT PRACTICE')
                viewer.set_camera(wp.vec3(*display['camera_position_m']),pitch=display['camera_pitch_deg'],yaw=display['camera_yaw_deg'])
                soil_surface=display_helpers.SoilSurface(
                    fixture.state.particle_q,
                    particle_spacing_m=fixture.particle_spacing_m,
                    voxel_size_m=display['surface_voxel_m'])
                display_helpers._prepare_rock_visuals(viewer,records,root,args.device)
                keyboard=KeyboardFourAxisState()
                manual=keyboard
                focus={'active':True}
                if args.manual_input=='gamepad':
                    manager=pyglet.input.ControllerManager()
                    controller=display_helpers._select_controller('gamepad',manager.get_controllers(),0)
                    controller.open()
                    manual=GamepadFourAxisState(manual_config)
                    controller.push_handlers(on_stick_motion=lambda selected,stick,vector:manual.set_stick(stick,vector.x,vector.y),
                        on_button_press=lambda selected,button:manual.press_button(button))
                    manager.push_handlers(on_disconnect=lambda selected:manual.disconnect() if selected is controller else None)
                    report['gamepad_name']=controller.name
                def press(symbol,modifiers):
                    nonlocal stop_reason
                    names_by_key={key.J:'J',key.L:'L',key.I:'I',key.K:'K',key.U:'U',key.O:'O',key.N:'N',key.M:'M'}
                    if symbol in names_by_key:
                        keyboard.press(names_by_key[symbol])
                    if symbol in (key.X,key.ESCAPE):
                        stop_reason='user_finished'
                    if symbol==key.SPACE:
                        gate.armed=False
                        keyboard.clear()
                def release(symbol,modifiers):
                    keyboard.release(key.symbol_string(symbol))
                def deactivated():
                    focus['active']=False
                    gate.armed=False
                    keyboard.clear()
                viewer.renderer.window.push_handlers(on_key_press=press,on_key_release=release,
                    on_activate=lambda:focus.update(active=True),on_deactivate=deactivated)
                def ui(imgui):
                    imgui.text('60Hz OPTIMIZED / NONCONTACT / NOT TRAINING DATA')
                    imgui.text('Space: pause/run | neutral sticks to arm | X/Esc/Back: exit')
                    imgui.text(f"Profile: {args.control_profile} | Input: {args.manual_input} | {'MULTI-AXIS' if normal else 'single axis'}")
                    imgui.text('Left: swing/boom | Right: bucket/arm' if args.manual_input=='gamepad' else 'J/L: swing | I/K: boom | U/O: arm | N/M: bucket')
                    imgui.text('Keyboard motion disabled in gamepad mode' if args.manual_input=='gamepad' else 'Gamepad motion disabled in keyboard mode')
                    imgui.text('Noncontact only. Initial boom is near upper stroke limit.')
                    imgui.text(f'Frames {len(rows)} / armed {gate.armed}')
                    if rows:
                        imgui.text(f"Wall seconds / simulation second: {60*rows[-1]['wall_seconds']:.2f}")
                        imgui.text(f"Command guard: {rows[-1]['command_guard_reason'] or 'clear'}")
                        imgui.text('Applied [swing, boom, arm, bucket]: '+str([round(v,4) for v in rows[-1]['command_si']]))
                viewer.register_ui_callback(ui,position='side')
                def signal_stop(*_):
                    nonlocal stop_reason
                    stop_reason='signal_requested'
                prior_sigterm=signal.signal(signal.SIGTERM,signal_stop)
                draws=0
                while viewer.is_running() and stop_reason is None:
                    if time.perf_counter()-started>=maximum_wall or len(rows)>=maximum_frames:
                        stop_reason='practice_budget'
                        break
                    # Draw dispatches device/window events before any new physics step.
                    display_helpers._draw(viewer,display_state,fixture,bridge.state,records,assembly,names,to_pose,args.device,display,len(rows),soil_surface)
                    draws+=1
                    if args.ui_smoke and draws>=3:
                        stop_reason='ui_smoke_paused_no_physics'
                        break
                    paused=viewer.is_paused()
                    action=manual.action(window_focused=focus['active'])
                    command=gate.command(action,np.asarray(bridge.state['mechanism_qpos'])[motion.AXES]-initial_axes,
                        paused=paused,focused=focus['active'],connected=getattr(manual,'connected',True),
                        estop=getattr(manual,'emergency_stop_latched',False))
                    stop_reason=stop_reason or gate.stop_reason
                    if stop_reason or not viewer.is_running():
                        break
                    if not paused and not args.ui_smoke:
                        log('input.ndjson',dict(frame=len(rows)+1,wall_monotonic=time.monotonic(),
                            physics_elapsed_s=len(rows)/60,raw=manual.raw_controls(),normalized=action,
                            device_gated_command_si=command.tolist(),focused=focus['active'],armed=gate.armed,
                            input_device=args.manual_input,training_eligible=False))
                        step(command,'manual_noncontact')
                stop_reason=stop_reason or 'window_closed'
            snapshot('final')
            report['checks']=dict(restore_exact=report['restore']['exact'],
                first_step_continuous=bool(not rows or rows[0]['first_step_continuous']),
                all_executed_frames_safe=all(r['motion_guard'] for r in rows),
                source_unchanged=motion.probe.bench.checkpoint_identity(before['path'])==before,
                metadata_unchanged=motion.probe.bench.sha256(Path(before['path']+'.json'))==source['diagnostic_snapshots'][-1]['metadata_sha256'],
                original_source_unchanged=motion.probe.bench.checkpoint_identity(source['source_checkpoint']['path'])==source['source_checkpoint'],
                input_bindings_unchanged=all(motion.probe.bench.sha256(Path(p))==sha for p,sha in bindings.items()),
                lineage_unchanged=motion.probe.seed_unchanged(source['seed']),
                smoke_complete=(len(rows)==510 and len(phases)==17 and all(p['passed'] for p in phases)) if args.smoke else True,
                combined_smoke_complete=(len(rows)==270 and len(phases)==9 and all(p['passed'] for p in phases)) if args.combined_smoke else True)
            report['status']='practice_completed' if all(report['checks'].values()) else 'practice_failed'
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='practice_failed',failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        try:
            snapshot('failure')
        except Exception as snapshot_error:
            report['snapshot_error']=str(snapshot_error)
    finally:
        if prior_sigterm is not None:
            signal.signal(signal.SIGTERM,prior_sigterm)
        for name,resource in [('controller',controller),('viewer',viewer),('bridge',bridge)]:
            if resource is not None:
                try:
                    resource.close()
                except Exception as error:
                    report.setdefault('cleanup_errors',{})[name]=str(error)
        report.update(stop_reason=stop_reason,frames=len(rows),phases=phases,
            timing=dict(simulation_seconds=len(rows)/60,physics_wall_seconds=sum(r['wall_seconds'] for r in rows),
                projection_wall_seconds=sum(r['projection_wall_seconds'] for r in rows),
                total_wall_seconds=time.perf_counter()-started))
        motion.probe.bench._write_json(output/'report.json',report)
    return report


def main():
    p=parser()
    for name in ('local-source-config','manual-config','display-config'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--manual-input',choices=('gamepad','keyboard'),default='gamepad')
    p.add_argument('--control-profile',choices=('normal','micro'),default='normal')
    p.add_argument('--rigid-internal-hz',type=int,choices=rigid_candidate.SUPPORTED_HZ,default=4800)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--smoke',action='store_true')
    group.add_argument('--combined-smoke',action='store_true')
    group.add_argument('--ui-smoke',action='store_true')
    a=p.parse_args()
    report=run(a)
    print(json.dumps({k:report.get(k) for k in ('status','frames','checks','stop_reason','timing','failure')},indent=2))
    return 0 if report['status']=='practice_completed' else 1


if __name__=='__main__':
    raise SystemExit(main())
