"""Temporarily lock supported device clocks during a measurement session."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
from threading import Event, Thread
from uuid import uuid4

import torch


DEVICE_MODEL = Path('/proc/device-tree/model')


def privileged_command(executable):
    """Use noninteractive sudo on Linux; Windows requires an elevated terminal."""
    if os.name == 'nt' or os.geteuid() == 0:
        return [executable]

    sudo = shutil.which('sudo')

    if sudo is None:
        raise RuntimeError('--lock-clocks requires root access or sudo')

    return [sudo, '-n', executable]


def inference_gpu_uuid():
    """Resolve CUDA device zero, including CUDA_VISIBLE_DEVICES remapping."""
    identifier = str(torch.cuda.get_device_properties(0).uuid)

    if identifier.startswith('MIG-'):
        raise RuntimeError('Clock locking requires a physical GPU, not a MIG instance')

    if not identifier.startswith('GPU-'):
        identifier = 'GPU-' + identifier

    return identifier


def select_backend():
    model = DEVICE_MODEL.read_text().rstrip('\0') if DEVICE_MODEL.exists() else ''
    is_jetson = 'Jetson' in model
    tool = 'jetson_clocks' if is_jetson else 'nvidia-smi'
    executable = shutil.which(tool)

    if executable is None:
        raise RuntimeError(f'--lock-clocks requires {tool}; omit the flag to use existing clocks')

    command = privileged_command(executable)

    if is_jetson:
        return JetsonClocks(command)

    return NvidiaClocks(command, inference_gpu_uuid())


def run_command(command):
    """Run a clock operation without an interactive password prompt."""
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as error:
        details = getattr(error, 'stderr', None) or str(error)
        
        raise RuntimeError(
            f'Clock command failed: {shlex.join(command)}\n{details}\n'
            'Check device support and permission to run the clock utility. '
            'For sudo authentication, run sudo -v in this terminal before retrying.'
        ) from error

    changes_nvidia_clocks = any(
        argument.startswith(('--lock-gpu-clocks=', '--lock-memory-clocks=',
                             '--reset-gpu-clocks', '--reset-memory-clocks'))
        for argument in command
    )

    if changes_nvidia_clocks and 'not supported' in result.stdout.lower():
        raise RuntimeError(f'Clock command is not supported: {shlex.join(command)}\n{result.stdout}')

    return result.stdout


def verify_locked(snapshot):
    """Require the tool to report fixed CPU, GPU and memory frequency bounds."""
    for domain in ('cpu', 'GPU', 'EMC'):
        lines = [line for line in snapshot.splitlines() if line.startswith(domain)]

        if not lines:
            raise RuntimeError(f'Cannot verify locked {domain} clocks from jetson_clocks --show')

        for line in lines:
            bounds = re.search(r'MinFreq=(\d+) MaxFreq=(\d+)', line)

            if bounds is None or int(bounds[1]) <= 0 or bounds[1] != bounds[2]:
                raise RuntimeError(f'Clock bounds are not locked: {line}')



class JetsonClocks:
    """Save and restore the CPU, GPU and EMC settings together."""

    name = 'jetson_clocks'
    cleanup_status = 'restored'

    def __init__(self, command):
        self.command = command
        self.cleanup_commands = []
        self.details = {'scope': 'CPU, GPU and EMC', 'cleanup': 'restore saved configuration'}

    def snapshot(self):
        return run_command(self.command + ['--show'])

    def save(self, directory):
        self.saved = directory / 'before.conf'
        run_command(self.command + ['--store', str(self.saved)])

        if not self.saved.is_file():
            raise RuntimeError('jetson_clocks did not save the original configuration')

        self.cleanup_commands.append(self.command + ['--restore', str(self.saved)])

    def lock(self):
        run_command(self.command)

    def verify(self, snapshot):
        verify_locked(snapshot)

    def restore(self, directory):
        run_command(self.cleanup_commands[0])
        restored = directory / 'restored.conf'
        run_command(self.command + ['--store', str(restored)])

        if self.saved.read_bytes() != restored.read_bytes():
            raise RuntimeError('Restored clock configuration differs from the saved settings')


class NvidiaClocks:
    """Lock the inference GPU; reset changed locks to defaults on exit."""

    name = 'nvidia-smi'
    cleanup_status = 'reset to defaults'

    def __init__(self, command, gpu_uuid):
        self.command = command
        self.device_command = command + ['-i', gpu_uuid]
        self.cleanup_commands = []
        self.details = {
            'gpu_uuid': gpu_uuid,
            'scope': 'GPU only; host CPU unchanged',
            'gpu_target': 'tdp,tdp (base TDP clock)',
            'memory_locked': False,
            'cleanup': 'reset changed locks to defaults, not restore previous custom locks',
        }
        self.memory_mhz = None

    def snapshot(self):
        return run_command(self.device_command + ['-q', '-d', 'CLOCK'])

    def save(self, directory):
        # GPU clock snapshots report operating rates, not previous lock ranges.
        # Do not mistake those rates for settings that can be restored.
        try:
            output = run_command(self.device_command + [
                '--query-supported-clocks=memory', '--format=csv,noheader,nounits',
            ])
            frequencies = [int(line.strip()) for line in output.splitlines() if line.strip().isdigit()]
            self.memory_mhz = max(frequencies) if frequencies else None
        except RuntimeError as error:
            if 'not supported' not in str(error).lower():
                raise

        if self.memory_mhz is None:
            self.details['memory_note'] = 'Supported memory clocks unavailable; memory unchanged'

    def lock(self):
        print('NVIDIA clock cleanup resets changed locks to defaults; previous custom locks are not restored')
        self.cleanup_commands.append(self.device_command + ['--reset-gpu-clocks'])

        try:
            run_command(self.device_command + ['--lock-gpu-clocks=tdp,tdp'])
        except RuntimeError as error:
            if 'not supported' in str(error).lower():
                self.cleanup_commands.pop()
            raise

        if self.memory_mhz is None:
            print(self.details['memory_note'])
            return

        self.cleanup_commands.append(self.device_command + ['--reset-memory-clocks'])

        try:
            run_command(self.device_command + [f'--lock-memory-clocks={self.memory_mhz},{self.memory_mhz}'])
        except RuntimeError as error:
            if 'not supported' not in str(error).lower():
                raise

            self.cleanup_commands.pop()
            self.details['memory_note'] = 'Runtime memory locking unsupported; memory unchanged'
            print(self.details['memory_note'])
            return

        self.details['memory_locked'] = True
        self.details['memory_target_mhz'] = self.memory_mhz

    def verify(self, snapshot):
        # The driver accepted the lock commands. Observed rates are recorded,
        # but may differ while idle or subject to thermal/power limits.
        self.details['verification'] = 'Driver accepted commands; operating clocks recorded'

    def restore(self, directory):
        errors = []

        for command in reversed(self.cleanup_commands):
            try:
                run_command(command)
            except RuntimeError as error:
                errors.append(str(error))

        if errors:
            raise RuntimeError('\n'.join(errors))

def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_record(directory, record):
    (directory / 'settings.json').write_text(json.dumps(record, indent=2) + '\n')


def interrupted(signum, frame):
    """Turn termination into normal stack unwinding so restoration can run."""
    raise KeyboardInterrupt(f'Measurement interrupted by {signal.Signals(signum).name}')


def refresh_sudo(command, stop, errors):
    """Keep an existing sudo authentication valid during a long campaign."""
    while not stop.wait(60):
        try:
            run_command([command[0], '-n', '-v'])
        except RuntimeError as error:
            # Command-specific NOPASSWD policies can work without sudo -v.
            # Cleanup still uses the actual clock-control command.
            if not errors:
                errors.append(str(error))


@contextmanager
def locked_clocks(results):
    """Lock clocks and clean up, including on failed setup or measurement.

    SIGINT and SIGTERM trigger cleanup. SIGKILL, power loss and a crashed OS cannot
    run Python cleanup; recorded cleanup commands remain available for recovery.
    Call from the main thread, as with the measurement command.
    """
    backend = select_backend()
    command = backend.command
    session_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    directory = Path(results) / 'clocks' / session_name
    directory.mkdir(parents=True)
    record = {
        'backend': backend.name,
        'started': timestamp(),
        'status': 'preparing',
        'cleanup_commands': backend.cleanup_commands,
        'settings': backend.details,
        'note': 'Fixed requested bounds do not override thermal or power throttling.',
    }
    write_record(directory, record)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}
    restoration_needed = False
    stop_refresh = Event()
    refresher = None

    try:
        for number in handlers:
            signal.signal(number, interrupted)

        (directory / 'before.txt').write_text(backend.snapshot())
        backend.save(directory)

        if len(command) == 3 and command[1] == '-n':
            record['sudo_refresh_errors'] = []
            refresher = Thread(
                target=refresh_sudo,
                args=(command, stop_refresh, record['sudo_refresh_errors']),
                daemon=True,
            )
            refresher.start()

        # Even a failed or interrupted lock command can partially change clocks.
        restoration_needed = True
        backend.lock()
        snapshot = backend.snapshot()
        (directory / 'locked.txt').write_text(snapshot)
        backend.verify(snapshot)
        record['status'] = 'locked'
        write_record(directory, record)
        print('Clocks locked; settings recorded in', directory)

        yield

        record['measurement'] = 'completed'
    except BaseException as error:
        record['measurement'] = 'failed or interrupted'
        record['error'] = repr(error)
        raise
    finally:
        # A second Ctrl+C must not interrupt restoration halfway through.
        for number in handlers:
            signal.signal(number, signal.SIG_IGN)

        stop_refresh.set()

        if refresher is not None:
            refresher.join()

        try:
            if restoration_needed and backend.cleanup_commands:
                backend.restore(directory)
                (directory / 'restored.txt').write_text(backend.snapshot())
                record['status'] = backend.cleanup_status
                print('Clock cleanup complete:', backend.cleanup_status)
            else:
                record['status'] = 'not changed'
        except BaseException as error:
            record['status'] = 'restoration failed'
            record['restore_error'] = repr(error)
            raise RuntimeError(
                'Clock cleanup failed. See settings.json in '
                f'{directory}\nRestore manually with: '
                + '; '.join(shlex.join(item) for item in backend.cleanup_commands)
            ) from error
        finally:
            record['finished'] = timestamp()

            try:
                write_record(directory, record)
            finally:
                for number, handler in handlers.items():
                    signal.signal(number, handler)
