"""Read-only, rate-limited mixed bucket volume display in a separate CPU process.

Uses the same calibrated density/heap measurement as the offline review. The
worker receives copied arrays, has no simulator access, and never issues actions.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

RESOURCE_KEYS = ('measurement_source', 'contract', 'front_rim', 'calibration')


def load_config(path):
    config = json.loads(Path(path).read_text())
    if config['schema'] != 'live_bucket_fill_v1':
        raise ValueError('live_fill_config_schema')
    for key in RESOURCE_KEYS:
        if not Path(config[key]).is_file():
            raise ValueError('live_fill_resource_missing:' + key)
    if not 1 <= float(config['interval_wall_s']) <= 10:
        raise ValueError('live_fill_interval')
    if 'occupancy_level_override' in config:
        level = float(config['occupancy_level_override'])
        if not np.isfinite(level) or level <= 0:
            raise ValueError('live_fill_occupancy_level')
    return config


class MixedFillEstimator:
    def __init__(self, config, volumes, vertices):
        from scipy.spatial import ConvexHull
        spec = importlib.util.spec_from_file_location(
            '_bound_fill_volume', config['measurement_source'])
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.measure = module.bucket_volume
        self.contract = json.loads(Path(config['contract']).read_text())
        with np.load(config['front_rim']) as data:
            self.front = data['front'].copy()
        calibration = json.loads(Path(config['calibration']).read_text())
        if not calibration['pass']:
            raise ValueError('live_fill_calibration_failed')
        self.calibration = module.Calibration.from_dict(
            calibration['profiles']['reference-2p5cm']['selected']['calibration'])
        self.calibration_status = 'reference_calibration'
        if 'occupancy_level_override' in config:
            self.calibration = replace(
                self.calibration, occupancy_level=float(config['occupancy_level_override']))
            self.calibration_status = 'user_selected_threshold_not_recalibrated'
        self.rocks = []
        spacing = .0125
        for volume, local in zip(volumes, vertices, strict=True):
            local = np.asarray(local)
            hull = ConvexHull(local)
            axes = [np.arange(local[:, i].min()+spacing/2,
                              local[:, i].max(), spacing) for i in range(3)]
            samples = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
            samples = samples[np.all(samples @ hull.equations[:, :3].T
                                     + hull.equations[:, 3] <= 1e-10, axis=1)]
            if not len(samples):
                raise ValueError('live_fill_empty_rock_samples')
            self.rocks.append((local, samples, float(volume)/len(samples)))

    def measure_snapshot(self, snapshot):
        from mixed_material_r0 import quaternion_matrix_wxyz
        started = time.perf_counter()
        pose = np.asarray(snapshot['pose'])
        soil = np.asarray(snapshot['soil'])
        points = [soil]
        weights = [np.full(len(soil), .025**3)]
        for (local, samples, weight), qpos in zip(
                self.rocks, snapshot['rocks'], strict=True):
            rotation = quaternion_matrix_wxyz(qpos[3:])
            center = ((local @ rotation.T + qpos[:3]).mean(axis=0)
                      - pose[:3, 3]) @ pose[:3, :3]
            if np.linalg.norm(center - np.array([.4, 0., -.5])) > 1.6:
                continue
            points.append(samples @ rotation.T + qpos[:3])
            weights.append(np.full(len(samples), weight))
        result = self.measure(np.concatenate(points), pose, self.contract,
                              self.front, np.concatenate(weights), self.calibration)
        return {
            'frame': snapshot['frame'], 'sim_time_s': snapshot['frame']/60.,
            'valid_estimate': result['valid_estimate'],
            'fill_percent': result['fill_percent'] if result['valid_estimate'] else None,
            'liters': (result['inside_liters'] + result['heap_liters']
                       if result['valid_estimate'] else None),
            'quality_reasons': result['quality_reasons'],
            'capacity_liters': result['capacity_liters'],
            'rocks_and_heap_included': True,
            'occupancy_level': self.calibration.occupancy_level,
            'calibration_status': self.calibration_status,
            'compute_wall_s': time.perf_counter()-started,
            'display_only': True,
        }


def display_lines(result, current_frame):
    if result is None:
        return ('FILL: measuring...', 'Soil + rocks + heap | estimate')
    if result.get('error'):
        return ('FILL: unavailable', 'Collection remains available')
    age = max(0, current_frame-result['frame'])/60.
    if not result['valid_estimate']:
        return ('FILL: -- (estimate invalid)',
                f'Connected pile / measurement bounds | age {age:.1f}s sim')
    return (f"FILL ~{result['fill_percent']:.1f}% | {result['liters']:.1f} L",
            f"Soil + rocks + heap | age {age:.1f}s sim")


class LiveFillDisplay:
    def __init__(self, config, records, vertices, log_path):
        self.config = config
        self.log_path = Path(log_path)
        self.latest = None
        self.future = None
        self.last_frame = None
        self.next_wall = 0.
        self.failed = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        self.process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), '--worker'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, env=env)
        self.initial = (config, [r['volume_m3'] for r in records], vertices)

    def _request(self, snapshot):
        if self.initial is not None:
            pickle.dump(self.initial, self.process.stdin, protocol=5)
            self.initial = None
        pickle.dump(snapshot, self.process.stdin, protocol=5)
        self.process.stdin.flush()
        return pickle.load(self.process.stdout)

    def poll(self):
        if self.future is not None and self.future.done():
            try:
                self.latest = self.future.result()
            except Exception as error:
                self.latest = {'error': type(error).__name__}
            self.failed = bool(self.latest.get('error'))
            with self.log_path.open('a') as stream:
                stream.write(json.dumps(self.latest, allow_nan=False)+'\n')
            self.future = None

    def due(self, frame):
        self.poll()
        return (not self.failed and self.future is None
                and frame != self.last_frame and time.monotonic() >= self.next_wall)

    def submit(self, frame, soil, pose, rocks):
        if not self.due(frame):
            return False
        snapshot = {'frame': int(frame), 'soil': np.array(soil, copy=True),
                    'pose': np.array(pose, copy=True), 'rocks': np.array(rocks, copy=True)}
        self.future = self.executor.submit(self._request, snapshot)
        self.last_frame = frame
        self.next_wall = time.monotonic()+float(self.config['interval_wall_s'])
        return True

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.executor.shutdown(wait=True)
        self.process.stdin.close()
        self.process.stdout.close()


def worker():
    try:
        os.nice(10)
        config, volumes, vertices = pickle.load(sys.stdin.buffer)
        estimator = MixedFillEstimator(config, volumes, vertices)
        while True:
            snapshot = pickle.load(sys.stdin.buffer)
            result = estimator.measure_snapshot(snapshot)
            pickle.dump(result, sys.stdout.buffer, protocol=5)
            sys.stdout.buffer.flush()
    except EOFError:
        return
    except Exception as error:
        pickle.dump({'error': type(error).__name__, 'message': str(error)},
                    sys.stdout.buffer, protocol=5)
        sys.stdout.buffer.flush()


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker']:
        raise SystemExit('worker entry only')
    worker()
