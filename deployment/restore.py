"""Relocate the frozen runtime without changing its physics source or bindings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(root):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for item in manifest['files']:
        path = root / item['path']
        if not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('unsafe_manifest_path:' + item['path'])
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('file_changed:' + item['path'])
    return manifest


def configure(root, manifest):
    # Reject access to the original project trees even on the build machine.
    blocked = tuple(Path(p).absolute() for p in manifest['original_resource_roots'])
    def guard(event, args):
        if event in ('open', 'os.listdir', 'os.scandir') and args:
            value = args[0]
            if isinstance(value, (str, bytes, os.PathLike)):
                path = Path(os.fsdecode(value)).absolute()
                if any(path.is_relative_to(p) for p in blocked):
                    raise PermissionError('original_resource_access_blocked:' + str(path))
    sys.addaudithook(guard)
    sys.path[:] = [p for p in sys.path if p and not any(
        Path(p).absolute().is_relative_to(old) for old in blocked)]
    repo = root / 'repo'
    sys.path[:0] = [str(repo / 'src/v4'), str(repo)]
    import layered_47_cycle_runtime as runtime
    runtime.BUNDLE = root / 'resources/frozen-runtime'
    runtime.KINEMATICS = root / 'resources/backends/small-soil-20260910-v17-performance'
    import run_layered_47_interactive_proxy_practice as display_helpers
    original = display_helpers._display_runtime
    def display_runtime(display):
        relocated = dict(display, backend_root=str(root /
            'resources/backends/autonomous-multiscoop-speed-opt-2026-09-14-v1'))
        return original(relocated)
    display_helpers._display_runtime = display_runtime
    return runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'smoke', 'dig'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--frames', type=int, choices=[0, 3], default=3, help='smoke only')
    parser.add_argument('--seconds', type=int, default=600, help='new dig budget; no training')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = verify(root)
    if args.mode == 'check':
        print(json.dumps(dict(status='files_verified', files=len(manifest['files']))))
        return
    if not 1 <= args.seconds <= 1800:
        parser.error('seconds must be 1 to 1800')
    output = (args.output or root / 'outputs' / time.strftime('%Y%m%d-%H%M%S')).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    runtime = configure(root, manifest)
    import numpy as np
    from stable_baselines3 import SAC
    import manual_residual_sac_env as env_module
    model = SAC.load(root / manifest['frozen_policy'], device='cpu')
    action, _ = model.predict(np.zeros(221, dtype=np.float32), deterministic=True)
    if action.shape != (5,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
        raise ValueError('frozen_policy_inference')
    budget = env_module.Budget(output / 'budget.json', seconds=args.seconds,
                               purpose='portable_restore_' + args.mode)
    env = env_module.ManualResidualEnv(output / 'run', budget,
            max_frames=7200, waiting_action_mode='error_strength_v1')
    result = dict(status='running', mode=args.mode, original_paths_blocked=True,
                  policy_loaded=True, training_updates=0, physical_frames=0)
    try:
        observation, _ = env.reset()
        result['zero_step_exact'] = env.report['zero_step_exact']
        if args.mode == 'smoke':
            # Run one real frozen-policy decision (20 Hz / three 60 Hz frames).
            if args.frames:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, truncated, _ = env.step(action)
                if done or truncated or env.runtime.frames != 3:
                    raise ValueError('unexpected_smoke_terminal')
            result['physical_frames'] = env.runtime.frames
            result['native_restore_max_error'] = max(env.runtime.restored['cycle_errors'].values())
            result['finite_observation'] = bool(np.isfinite(observation).all())
            import run_layered_47_interactive_proxy_practice as display_helpers
            display = json.loads((root / 'repo/configs/v4/layered-47-interactive-proxy-practice-v1.json').read_text())['display']
            assembly, visual_model, visual_state, names, to_pose = display_helpers._display_runtime(display)
            from soil_surface import SoilSurface
            surface = SoilSurface(env.runtime.fixture.state.particle_q,
                particle_spacing_m=.025, voxel_size_m=display['surface_voxel_m'])
            surface.update(env.runtime.fixture.state.particle_q, 0)
            result['visual_model_loaded'] = True
            result['soil_surface_generated'] = True
            result['status'] = 'passed'
        else:
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, done, truncated, _ = env.step(action)
            result.update(status=env.report['status'], success=env.report['success'],
                          physical_frames=env.report['physical_frames'])
    except BaseException as error:
        result.update(status='failed', error=repr(error))
        raise
    finally:
        if env.runtime is not None:
            result['physical_frames'] = env.runtime.frames
        env.close()
        (output / 'restore-result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
