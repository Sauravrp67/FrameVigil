"""Control power sampling in a separate Python process, outside inference's GIL.

Uses the same sensor equations, 50 ms sampling request and tegrastats reader as
Power. JSON messages carry commands and recorded samples across process pipes.
"""

import json
import os
from pathlib import Path
import selectors
import subprocess
import sys


class IsolatedPower:
    """Expose the sampler interface while a worker process collects telemetry."""

    def __init__(self):
        self.rows = []
        self.tgs = []
        self.sensor = None
        self._phase = 'setup'
        self.proc = None

    def _read(self, timeout=30):
        with selectors.DefaultSelector() as selector:
            selector.register(self.proc.stdout, selectors.EVENT_READ)

            if not selector.select(timeout):
                raise TimeoutError('Power worker response timeout')

        line = self.proc.stdout.readline()

        if not line:
            raise RuntimeError('Power worker exited before responding')

        message = json.loads(line)

        if 'error' in message:
            raise RuntimeError(message['error'])

        return message

    def _send(self, message):
        self.proc.stdin.write(json.dumps(message) + '\n')
        self.proc.stdin.flush()

    def start(self):
        environment = dict(os.environ)
        environment['PYTHONPATH'] = os.pathsep.join(sys.path)
        self.proc = subprocess.Popen(
            [sys.executable, '-m', 'adaptive_video.measurement.power_worker'],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        message = self._read()
        assert message['status'] == 'ready'
        self.sensor = Path(message['sensor'])

        return self

    @property
    def phase(self):
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase = value

        if self.proc is not None:
            self._send({'command': 'phase', 'value': value})
            assert self._read()['status'] == 'phase_set'

    def stop(self):
        if self.proc is None:
            return

        self._send({'command': 'stop'})
        response = self._read()
        self.rows = response['samples']
        self.tgs = response['tegrastats']
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        assert self.proc.returncode == 0

    def save(self, path, extra):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            'sensor': str(self.sensor),
            'boundary': 'Jetson module VDD_IN',
            'samples': self.rows,
            'tegrastats': self.tgs,
            'telemetry_execution': 'INA3221 sampler in a separate Python process',
            'worker_pid': self.proc.pid,
            **extra,
        }
        path.write_text(json.dumps(result, indent=2) + '\n')
