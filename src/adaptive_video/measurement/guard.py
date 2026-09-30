"""Inspect hardware capabilities and competing GPU activity during measurement."""

import json
import os
from pathlib import Path
import subprocess
import shutil
import time


def require_module_power():
    """Require the sensor and tool used by the VDD_IN power backend."""
    sensors = Path('/sys/class/hwmon').glob('hwmon*/in1_label')
    has_sensor = any(path.read_text().strip() == 'VDD_IN' for path in sensors)

    if not has_sensor or shutil.which('tegrastats') is None:
        raise RuntimeError(
            'VDD_IN measurement requires a module-input sensor and tegrastats. '
            'Use --power none for accuracy and latency without energy measurement.'
        )


def check_competing_benchmark():
    """Reject known benchmark commands other than this process."""
    processes = subprocess.check_output(['ps', '-eo', 'pid,args'], text=True)

    for line in processes.splitlines()[1:]:
        fields = line.strip().split(maxsplit=1)

        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue

        command = fields[1]
        program = Path(command.split()[0]).name
        is_scheduler_benchmark = 'python' in program and 'scripts/benchmark_scheduler.py' in command

        if program == 'trtexec' or is_scheduler_benchmark:
            raise RuntimeError('Competing hardware benchmark; measurement rejected: ' + line.strip())


def device_snapshot():
    """Record visible GPU clients, temperatures and clocks; reject other clients.

    Process inspection is best effort where operating-system permissions deny
    access. This is an exclusivity check, not a guarantee of zero system noise.
    """
    snapshot = {
        'time': time.monotonic(),
        'gpu_clients': [],
        'temperatures_mC': {},
        'devfreq_hz': {},
    }

    for process_directory in Path('/proc').glob('[0-9]*'):
        if int(process_directory.name) == os.getpid():
            continue

        try:
            command = (process_directory / 'cmdline').read_bytes()
            command = command.replace(b'\0', b' ').decode(errors='replace')

            if not command:
                continue

            for descriptor in (process_directory / 'fd').iterdir():
                try:
                    target = os.readlink(descriptor)
                except OSError:
                    continue

                if target in ['/dev/nvhost-gpu', '/dev/nvhost-ctrl-gpu', '/dev/nvidia0']:
                    snapshot['gpu_clients'].append({
                        'pid': int(process_directory.name),
                        'command': command,
                    })
                    break

        except (OSError, PermissionError):
            continue

    for path in Path('/sys/class/thermal').glob('thermal_zone*/temp'):
        try:
            snapshot['temperatures_mC'][str(path)] = int(path.read_text())
        except (OSError, ValueError):
            pass

    for path in Path('/sys/class/devfreq').glob('*/cur_freq'):
        try:
            snapshot['devfreq_hz'][str(path)] = int(path.read_text())
        except (OSError, ValueError):
            pass

    if snapshot['gpu_clients']:
        raise RuntimeError(
            'Other GPU clients invalidate exclusive measurement: '
            + json.dumps(snapshot['gpu_clients'])
        )

    return snapshot
