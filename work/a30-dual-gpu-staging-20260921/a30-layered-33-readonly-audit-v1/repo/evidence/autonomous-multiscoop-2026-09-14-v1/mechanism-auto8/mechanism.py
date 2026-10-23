"""Process-isolated experimental MPM wrench -> articulated dynamics adapter."""
import json
from pathlib import Path
import select
import subprocess
import numpy as np


class Mechanism:
    def __init__(self, root, qpos, stderr_path):
        root = Path(root).resolve()
        self.log = Path(stderr_path).open('w')
        self.process = subprocess.Popen([str(root/'mpm_bridge'), str(root/'mechanism.xml')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1)
        try:
            if self._read() != '@READY':raise RuntimeError('bridge startup')
            self.state = self._request('init', self._vector(qpos, 19))
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _vector(value, count):
        a = np.asarray(value, dtype=float)
        if a.shape != (count,) or not np.isfinite(a).all():
            raise ValueError(f'expected {count} finite values')
        return a

    def _read(self):
        if not select.select([self.process.stdout], [], [], 15)[0]:
            raise TimeoutError('mechanism response timeout')
        line = self.process.stdout.readline().strip()
        if not line:raise RuntimeError('mechanism stopped; inspect stderr log')
        return line

    def _request(self, operation, values):
        self.process.stdin.write(operation+' '+' '.join(format(float(x),'.17g') for x in values)+'\n')
        self.process.stdin.flush()
        line = self._read()
        if not line.startswith('@STATE '):raise RuntimeError('invalid mechanism response')
        return json.loads(line[7:])

    def step(self, command, force, torque, point, *, dt=1/120, substeps=20):
        """Hold a world wrench about fixed world `point` for this interval.

        force: soil ON bucket (N), torque: about point (Nm), command: yaw
        rad/s and three cylinder m/s. Caller owns coupling sample timing.
        No density-based carried weight or extra empirical resistance is added.
        """
        values = np.concatenate(([self.state['time_s'], dt, substeps],
                                 self._vector(command,4), self._vector(force,3),
                                 self._vector(torque,3), self._vector(point,3)))
        self.state = self._request('step', values)
        return self.state

    def close(self):
        p = self.process
        if p.poll() is None:
            try:
                p.stdin.write('close\n');p.stdin.flush();p.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                p.kill();p.wait(timeout=5)
        p.stdin.close();p.stdout.close();self.log.close()

    def __enter__(self):return self
    def __exit__(self, *_):self.close()
