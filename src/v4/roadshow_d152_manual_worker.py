"""Local operator input for the isolated D152 roadshow contact runner.

The worker supplies the normalized action input to the existing D152 feedback
controller. It never advances physics or bypasses the existing projection.
"""

from pathlib import Path
import json
import time

import numpy as np

from contact_bc_inference import LIMITS
from manual_action_input import KeyboardFourAxisState
from roadshow_geometry_baseline import joint_target_intent
from run_layered_47_optimized_practice import InputGate


ROOT = Path('/home/qunwang/wajueji')
STEP_OPPOSITES = dict(L='J', J='L', I='K', K='I', O='U', U='O', M='N', N='M')


class ManualWorker:
    """Matches the inference-worker API while preserving input provenance."""

    def __init__(self, _python, _policy, _policy_sha, *, stderr_path=None):
        self.ready = {'status': 'ready', 'controller': 'local_operator',
                      'neural_inference': False}
        self.viewer = None
        self.runtime = None
        self.stop_reason = None
        self.phase = 'approach_contact'
        self.last_record = {}
        self.geometry_reference = None
        self.active_seconds = 0.0
        self.active_started = None
        self.step_remaining = 0
        self.step_frames = 30
        self.step_id = 0
        self.step_decision_frame = None
        self.step_decision_wall_s = None
        self.step_action = np.zeros(4)
        self.physical_held_keys = set()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.viewer is not None:
            self.viewer.close()

    def bind(self, runtime, feedback, goal, *, mode, device, ui_smoke=False,
             step_frames=30):
        self.runtime = runtime
        self.feedback = feedback
        self.goal = goal
        self.mode = mode
        self.device = device
        self.ui_smoke = ui_smoke
        self.input_gate = InputGate('normal')
        self.keyboard = KeyboardFourAxisState()
        self.manual = self.keyboard
        if step_frames not in (15, 30, 60):
            raise ValueError('invalid_step_frames')
        self.step_frames = step_frames
        self.focus = True
        self.annotations = []
        if mode == 'zero':
            return

        import newton
        import warp as wp
        from pyglet.window import key
        import run_layered_47_sparse_manual_contact_episode as manual
        from soil_surface import SoilSurface

        self.manual_module = manual
        self.display = json.loads((ROOT/'configs/v4/layered-47-interactive-proxy-practice-v1.json').read_text())['display']
        (self.display_assembly, display_model, self.display_state,
         self.display_names, self.to_pose) = manual.display_helpers._display_runtime(self.display)
        self.viewer = newton.viewer.ViewerGL(
            width=self.display['width'], height=self.display['height'],
            vsync=False, paused=True)
        self.viewer.set_model(display_model)
        self.viewer.picking_enabled = False
        self.viewer.renderer.set_title(
            'D152 MANUAL CURL PILOT | Space pause | X finish | 1-4 mark phase')
        self.viewer.set_camera(wp.vec3(*self.display['camera_position_m']),
            pitch=self.display['camera_pitch_deg'], yaw=self.display['camera_yaw_deg'])
        self.surface = SoilSurface(runtime.fixture.state.particle_q,
            particle_spacing_m=float(runtime.base['mpm']['particle_spacing_m']),
            voxel_size_m=float(self.display['surface_voxel_m']))
        manual.display_helpers._prepare_rock_visuals(
            self.viewer, runtime.records, runtime.root, device)

        keys = {key.J:'J', key.L:'L', key.I:'I', key.K:'K',
                key.U:'U', key.O:'O', key.N:'N', key.M:'M'}
        marks = {key._1:'penetrate_deepen', key._2:'curl_load',
                 key._3:'lift_clear', key._4:'off_soil_confirm',
                 key.R:'uncertain_recovery'}

        def press(symbol, _modifiers):
            if symbol in keys:
                if self.mode == 'step_keyboard':
                    if (symbol not in self.physical_held_keys and
                            self.viewer.is_paused() and
                            self.phase == 'curl_load' and
                            not self.feedback.lift_intent):
                        self.select_step_key(keys[symbol])
                    self.physical_held_keys.add(symbol)
                else:
                    self.keyboard.press(keys[symbol])
            if symbol in marks:
                event = dict(frame=self.runtime.frames+1,
                    label=marks[symbol], wall_monotonic_s=time.monotonic(),
                    causal_from_next_decision=True, training_eligible=False)
                self.annotations.append(event)
                manual.append_json(self.runtime.output.parent/'annotations.ndjson', event)
            if symbol in (key.X, key.ESCAPE):
                self.stop_reason = 'user_finished'
            if self.mode == 'step_keyboard' and self.viewer.is_paused():
                lengths = {key._5: 15, key._6: 30, key._7: 60}
                if symbol in lengths:
                    self.step_frames = lengths[symbol]
                if symbol == key.C:
                    self.keyboard.clear()
            if symbol == key.SPACE:
                self.input_gate.armed = False
                if self.mode == 'step_keyboard':
                    if self.viewer.is_paused() and self.phase == 'curl_load' and not self.feedback.lift_intent:
                        self.begin_step()
                    else:
                        self.step_remaining = 0
                else:
                    self.keyboard.clear()

        def release(symbol, _modifiers):
            if symbol in keys:
                if self.mode == 'step_keyboard':
                    self.physical_held_keys.discard(symbol)
                else:
                    self.keyboard.release(keys[symbol])

        def deactivate():
            self.focus = False
            self.input_gate.armed = False
            self.keyboard.clear()
            self.physical_held_keys.clear()
            self.step_remaining = 0
            self.viewer._paused = True

        self.viewer.renderer.window.push_handlers(
            on_key_press=press, on_key_release=release,
            on_activate=lambda: setattr(self, 'focus', True),
            on_deactivate=deactivate)

        def ui(imgui):
            imgui.text('D152 keyboard pilot | review needed')
            imgui.text(f'Frame {self.runtime.frames} | {self.phase}')
            if self.mode == 'step_keyboard':
                if self.phase in ('approach_contact', 'penetrate'):
                    imgui.text('AUTO cut | direction keys inactive')
                elif self.feedback.lift_intent:
                    imgui.text('AUTO bounded lift | keys inactive')
                else:
                    selected = ','.join(self.keyboard.raw_controls()['pressed_keys']) or 'zero intent'
                    imgui.text(f'SELECT: {selected} | {self.step_frames} frames')
                    imgui.text('Space: run step / pause | X: finish')
                    imgui.text('5/6/7: 15/30/60 frames | C: clear')
                    imgui.text('No key still permits feedback correction')
            else:
                imgui.text('Space pause / resume | X finish')
            imgui.text('Swing: L/J  Boom: I/K')
            imgui.text('Arm: O/U  Bucket: M/N')
            imgui.text('1 cut  2 curl  3 lift  4 off soil')
            imgui.text('R uncertain | geometry starts cut')
            imgui.text(f'Armed: {self.input_gate.armed} | Focus: {self.focus}')
        self.viewer.register_ui_callback(ui, position='side')

    def select_step_key(self, selected):
        if selected not in STEP_OPPOSITES:
            raise ValueError('invalid_step_key')
        current = set(self.keyboard.raw_controls()['pressed_keys'])
        if selected in current:
            self.keyboard.release(selected)
        else:
            self.keyboard.release(STEP_OPPOSITES[selected])
            self.keyboard.press(selected)

    def begin_step(self):
        self.step_id += 1
        self.step_remaining = self.step_frames
        self.step_decision_frame = self.runtime.frames + 1
        self.step_decision_wall_s = time.monotonic()
        self.step_action = np.asarray(self.keyboard.action(window_focused=self.focus), dtype=float)

    def _draw(self):
        r = self.runtime
        self.manual_module.draw_with_selected_cut(
            self.viewer, self.display_state, r.fixture, r.bridge.state,
            r.records, self.display_assembly, self.display_names,
            self.to_pose, self.device, self.display, self.goal,
            r.frames, self.surface)

    def active_wall_seconds(self):
        return self.active_seconds + (0.0 if self.active_started is None
            else time.monotonic()-self.active_started)

    def geometry_intent(self, qpos, phase):
        if phase not in ('curl_load', 'lift', 'off_soil'):
            return None
        q = np.asarray(qpos, dtype=float)[self.feedback.chain.axes]
        if self.geometry_reference is None:
            self.geometry_reference = q.copy()
        low, high = self.feedback.chain.limits[self.feedback.chain.axes[3]]
        from run_layered_47_sparse_manual_contact_episode import (
            BUCKET_INTERNAL_UPPER_TARGET_M, KINEMATIC_JOINT_MARGIN_M)
        upper = min(BUCKET_INTERNAL_UPPER_TARGET_M,
                    high-KINEMATIC_JOINT_MARGIN_M)
        intent, target = joint_target_intent(q, self.geometry_reference,
            low, upper, LIMITS, self.feedback.config.position_gain_s)
        return {'normalized': intent.tolist(), 'joint_target_m': target.tolist(),
                'scope': 'unprojected_same_state_intent_only'}

    def predict(self, _features, _digest):
        started = time.perf_counter()
        if self.mode != 'zero':
            if (self.mode == 'step_keyboard' and self.phase == 'curl_load'
                    and not self.feedback.lift_intent and self.step_remaining == 0):
                self.viewer._paused = True
                self.input_gate.armed = False
            draws = 0
            if self.active_started is not None and self.viewer.is_paused():
                self.active_seconds += time.monotonic()-self.active_started
                self.active_started = None
            while self.viewer.is_running() and self.stop_reason is None:
                self._draw()
                draws += 1
                if self.ui_smoke and draws >= 3:
                    from PIL import Image
                    Image.fromarray(self.viewer.get_frame(render_ui=True).numpy()).save(
                        self.runtime.output.parent/'ui-smoke.png')
                    break
                if not self.viewer.is_paused() and self.focus:
                    break
                time.sleep(0.02)
            if not self.viewer.is_running() and self.stop_reason is None:
                self.stop_reason = 'window_closed'
        if self.stop_reason is not None:
            action = np.zeros(4)
            requested = np.zeros(4)
        else:
            if self.mode == 'step_keyboard':
                action = (self.step_action.copy() if self.phase == 'curl_load'
                    and not self.feedback.lift_intent and self.step_remaining > 0
                    else np.zeros(4))
                if action.any() and not self.input_gate.armed:
                    self.input_gate.command(np.zeros(4), np.zeros(4),
                        paused=False, focused=self.focus)
            else:
                action = np.asarray(self.manual.action(window_focused=self.focus), dtype=float)
            requested = self.input_gate.command(action, np.zeros(4),
                paused=False, focused=self.focus,
                connected=getattr(self.manual, 'connected', True),
                estop=getattr(self.manual, 'emergency_stop_latched', False))
            if self.input_gate.stop_reason:
                self.stop_reason = self.input_gate.stop_reason
            if (self.mode == 'step_keyboard' and self.phase == 'curl_load'
                    and not self.feedback.lift_intent and self.step_remaining > 0):
                self.step_remaining -= 1
        if self.active_started is None and self.stop_reason is None:
            self.active_started = time.monotonic()
        normalized = requested/LIMITS
        self.last_record = {'input_device': self.mode,
            'raw_controls': (self.manual.raw_controls() if self.mode != 'step_keyboard'
                else {'selected_keys': self.manual.raw_controls()['pressed_keys'],
                      'physical_held_keys': sorted(self.physical_held_keys)}),
            'operator_action_normalized_before_gate': action.tolist(),
            'operator_action_normalized_after_gate': normalized.tolist(),
            'operator_input_wall_monotonic_s': (self.step_decision_wall_s
                if self.mode == 'step_keyboard' and action.any() else time.monotonic()),
            'operator_step_id': self.step_id if self.mode == 'step_keyboard' else None,
            'operator_decision_frame': (self.step_decision_frame
                if self.mode == 'step_keyboard' and action.any() else None),
            'operator_step_horizon_frames': (self.step_frames
                if self.mode == 'step_keyboard' and action.any() else None),
            'operator_gate_armed': self.input_gate.armed,
            'operator_focused': self.focus,
            'operator_connected': getattr(self.manual, 'connected', True),
            'operator_estop': getattr(self.manual, 'emergency_stop_latched', False),
            'annotations_since_previous_decision': self.annotations[:],
            'training_eligible': False}
        self.annotations.clear()
        return normalized, {'roundtrip_seconds': time.perf_counter()-started,
            'worker_seconds': None, 'source': 'local_operator_not_neural_inference'}
