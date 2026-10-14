"""Clock lifecycle tests use a fake command backend, never real hardware."""

import json
import os
from pathlib import Path
import signal
from unittest.mock import Mock

import pytest

from src.measurement import clocks, stages


LOCKED = (
    'cpu0: Governor=schedutil MinFreq=100 MaxFreq=100 CurrentFreq=100\n'
    'GPU MinFreq=200 MaxFreq=200 CurrentFreq=200\n'
    'EMC MinFreq=300 MaxFreq=300 CurrentFreq=300\n'
)


@pytest.fixture
def backend(monkeypatch):
    commands = []

    def execute(command):
        commands.append(command)

        if '--store' in command:
            Path(command[-1]).write_text('original settings\n')

        return LOCKED

    monkeypatch.setattr(clocks, 'select_backend', lambda: clocks.JetsonClocks(['fake-clock-tool']))
    monkeypatch.setattr(clocks, 'run_command', execute)
    return commands


def record(tmp_path):
    return json.loads(next(tmp_path.glob('clocks/*/settings.json')).read_text())


def test_success_restores_saved_settings_and_signal_handlers(tmp_path, backend):
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    with clocks.locked_clocks(tmp_path):
        assert record(tmp_path)['status'] == 'locked'

    assert record(tmp_path)['status'] == 'restored'
    assert record(tmp_path)['measurement'] == 'completed'
    assert any('--restore' in command for command in backend)
    assert all(signal.getsignal(number) == handler for number, handler in previous.items())


@pytest.mark.parametrize('error', [ValueError('measurement failed'), KeyboardInterrupt()])
def test_measurement_failure_still_restores(tmp_path, backend, error):
    with pytest.raises(type(error)):
        with clocks.locked_clocks(tmp_path):
            raise error

    assert record(tmp_path)['status'] == 'restored'
    assert record(tmp_path)['measurement'] == 'failed or interrupted'


def test_sigterm_restores_settings(tmp_path, backend):
    previous = signal.getsignal(signal.SIGTERM)

    with pytest.raises(KeyboardInterrupt, match='SIGTERM'):
        with clocks.locked_clocks(tmp_path):
            os.kill(os.getpid(), signal.SIGTERM)

    assert record(tmp_path)['status'] == 'restored'
    assert signal.getsignal(signal.SIGTERM) == previous


def test_partially_failed_lock_attempt_still_restores(tmp_path, backend, monkeypatch):
    execute = clocks.run_command

    def fail_lock(command):
        if command == ['fake-clock-tool']:
            raise RuntimeError('lock failed halfway')

        return execute(command)

    monkeypatch.setattr(clocks, 'run_command', fail_lock)

    with pytest.raises(RuntimeError, match='lock failed halfway'):
        with clocks.locked_clocks(tmp_path):
            pytest.fail('Measurement must not start after lock failure')

    assert record(tmp_path)['status'] == 'restored'


def test_restoration_failure_is_reported_with_manual_recovery(tmp_path, backend, monkeypatch):
    execute = clocks.run_command

    def fail_restore(command):
        if '--restore' in command:
            raise RuntimeError('restore denied')

        return execute(command)

    monkeypatch.setattr(clocks, 'run_command', fail_restore)

    with pytest.raises(RuntimeError, match='Restore manually with:'):
        with clocks.locked_clocks(tmp_path):
            pass

    assert record(tmp_path)['status'] == 'restoration failed'


def test_changed_restoration_is_not_reported_as_success(tmp_path, backend, monkeypatch):
    execute = clocks.run_command

    def incorrect_restore(command):
        result = execute(command)

        if '--store' in command and command[-1].endswith('restored.conf'):
            Path(command[-1]).write_text('different settings')

        return result

    monkeypatch.setattr(clocks, 'run_command', incorrect_restore)

    with pytest.raises(RuntimeError, match='Clock cleanup failed'):
        with clocks.locked_clocks(tmp_path):
            pass

    assert record(tmp_path)['status'] == 'restoration failed'


def test_unsupported_platform_fails_without_hardware_commands(monkeypatch, tmp_path):
    monkeypatch.setattr(clocks, 'DEVICE_MODEL', tmp_path / 'missing-model')
    monkeypatch.setattr(clocks.shutil, 'which', lambda command: None)
    execute = Mock()
    monkeypatch.setattr(clocks, 'run_command', execute)

    with pytest.raises(RuntimeError, match='requires nvidia-smi'):
        with clocks.locked_clocks(tmp_path):
            pytest.fail('Unsupported locking must not start measurement')

    execute.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_variable_clock_bounds_are_rejected():
    with pytest.raises(RuntimeError, match='not locked'):
        clocks.verify_locked(LOCKED.replace('MinFreq=200', 'MinFreq=100'))


@pytest.mark.parametrize('enabled', [False, True])
def test_measure_stage_encloses_runner_only_when_requested(monkeypatch, tmp_path, backend, enabled):
    monkeypatch.setattr(stages, 'checked_experiment', lambda path: tmp_path)
    monkeypatch.setattr(stages.guard, 'check_competing_benchmark', lambda: None)

    def runner(directory, power_backend):
        assert directory == tmp_path
        assert power_backend == 'none'

        if enabled:
            assert record(tmp_path / 'results')['status'] == 'locked'
        else:
            assert backend == []

    monkeypatch.setattr(stages.runner, 'main', runner)
    stages.measure(tmp_path, lock_clocks=enabled)

    if enabled:
        assert record(tmp_path / 'results')['status'] == 'restored'
    else:
        assert not list(tmp_path.iterdir())


def test_long_run_refreshes_sudo_without_prompting(monkeypatch):
    stop = Mock()
    stop.wait.side_effect = [False, True]
    execute = Mock()
    monkeypatch.setattr(clocks, 'run_command', execute)

    clocks.refresh_sudo(['/usr/bin/sudo', '-n', '/usr/bin/jetson_clocks'], stop, [])

    execute.assert_called_once_with(['/usr/bin/sudo', '-n', '-v'])


def test_failed_sudo_refresh_is_recorded(monkeypatch):
    stop = Mock()
    stop.wait.side_effect = [False, True]
    errors = []
    monkeypatch.setattr(clocks, 'run_command', Mock(side_effect=RuntimeError('permission denied')))

    clocks.refresh_sudo(['/usr/bin/sudo', '-n', '/usr/bin/jetson_clocks'], stop, errors)

    assert errors == ['permission denied']


@pytest.fixture
def nvidia_backend(monkeypatch):
    commands = []

    def execute(command):
        commands.append(command)

        if '--query-supported-clocks=memory' in command:
            return '4000\n6000\n'

        return 'Clock command completed\n'

    monkeypatch.setattr(clocks, 'run_command', execute)
    monkeypatch.setattr(
        clocks, 'select_backend',
        lambda: clocks.NvidiaClocks(['fake-nvidia-smi'], 'GPU-visible-device'),
    )
    return commands


def test_nvidia_locks_selected_gpu_and_resets_both_domains(tmp_path, nvidia_backend):
    with clocks.locked_clocks(tmp_path):
        settings = record(tmp_path)['settings']
        assert settings['gpu_uuid'] == 'GPU-visible-device'
        assert settings['memory_target_mhz'] == 6000
        assert settings['memory_locked'] is True

    assert record(tmp_path)['status'] == 'reset to defaults'
    assert all(command[1:3] == ['-i', 'GPU-visible-device'] for command in nvidia_backend)
    options = [command[-1] for command in nvidia_backend]
    assert '--lock-gpu-clocks=tdp,tdp' in options
    assert '--lock-memory-clocks=6000,6000' in options
    assert '--reset-gpu-clocks' in options
    assert '--reset-memory-clocks' in options


@pytest.mark.parametrize('unsupported_at', ['query', 'lock'])
def test_nvidia_can_run_with_gpu_only_when_memory_control_is_unsupported(
    tmp_path, nvidia_backend, monkeypatch, unsupported_at
):
    execute = clocks.run_command

    def memory_unsupported(command):
        is_query = '--query-supported-clocks=memory' in command
        is_lock = command[-1].startswith('--lock-memory-clocks=')

        if (unsupported_at == 'query' and is_query) or (unsupported_at == 'lock' and is_lock):
            raise RuntimeError('Memory clocks not supported')

        return execute(command)

    monkeypatch.setattr(clocks, 'run_command', memory_unsupported)

    with clocks.locked_clocks(tmp_path):
        assert record(tmp_path)['settings']['memory_locked'] is False

    assert '--reset-gpu-clocks' in [command[-1] for command in nvidia_backend]
    assert '--reset-memory-clocks' not in [command[-1] for command in nvidia_backend]


def test_nvidia_memory_failure_cleans_up_gpu_lock(tmp_path, nvidia_backend, monkeypatch):
    execute = clocks.run_command

    def memory_failure(command):
        if command[-1].startswith('--lock-memory-clocks='):
            raise RuntimeError('Driver failure')

        return execute(command)

    monkeypatch.setattr(clocks, 'run_command', memory_failure)

    with pytest.raises(RuntimeError, match='Driver failure'):
        with clocks.locked_clocks(tmp_path):
            pytest.fail('Measurement must not start')

    assert '--reset-gpu-clocks' in [command[-1] for command in nvidia_backend]
    assert '--reset-memory-clocks' in [command[-1] for command in nvidia_backend]


def test_nvidia_gpu_unsupported_does_not_run_measurement(tmp_path, nvidia_backend, monkeypatch):
    execute = clocks.run_command

    def unsupported(command):
        if command[-1].startswith('--lock-gpu-clocks='):
            raise RuntimeError('GPU clock locking not supported')

        return execute(command)

    monkeypatch.setattr(clocks, 'run_command', unsupported)

    with pytest.raises(RuntimeError, match='not supported'):
        with clocks.locked_clocks(tmp_path):
            pytest.fail('Measurement must not start')

    assert record(tmp_path)['status'] == 'not changed'


def test_nvidia_cleanup_attempts_gpu_reset_even_when_memory_reset_fails(tmp_path, nvidia_backend, monkeypatch):
    execute = clocks.run_command

    def fail_memory_reset(command):
        if command[-1] == '--reset-memory-clocks':
            raise RuntimeError('Memory reset failed')

        return execute(command)

    monkeypatch.setattr(clocks, 'run_command', fail_memory_reset)

    with pytest.raises(RuntimeError, match='Clock cleanup failed'):
        with clocks.locked_clocks(tmp_path):
            pass

    assert '--reset-gpu-clocks' in [command[-1] for command in nvidia_backend]


@pytest.mark.parametrize('model_name, expected', [
    ('NVIDIA Jetson Orin Nano', clocks.JetsonClocks),
    ('Desktop', clocks.NvidiaClocks),
])
def test_backend_selection(monkeypatch, tmp_path, model_name, expected):
    model = tmp_path / 'model'
    model.write_text(model_name + '\0')
    monkeypatch.setattr(clocks, 'DEVICE_MODEL', model)
    monkeypatch.setattr(clocks.shutil, 'which', lambda name: '/usr/bin/' + name)
    monkeypatch.setattr(clocks, 'privileged_command', lambda executable: [executable])
    monkeypatch.setattr(clocks, 'inference_gpu_uuid', lambda: 'GPU-selected')

    assert isinstance(clocks.select_backend(), expected)


def test_gpu_selection_uses_cuda_visible_device_uuid(monkeypatch):
    properties = Mock(uuid='test-uuid')
    query = Mock(return_value=properties)
    monkeypatch.setattr(clocks.torch.cuda, 'get_device_properties', query)

    assert clocks.inference_gpu_uuid() == 'GPU-test-uuid'
    query.assert_called_once_with(0)


def test_success_exit_with_unsupported_stdout_is_not_treated_as_locked(monkeypatch):
    result = Mock(stdout='Setting locked clocks is not supported for this device', stderr='')
    monkeypatch.setattr(clocks.subprocess, 'run', Mock(return_value=result))

    with pytest.raises(RuntimeError, match='not supported'):
        clocks.run_command(['fake-nvidia-smi', '--lock-gpu-clocks=tdp,tdp'])


def test_read_only_snapshot_can_contain_unsupported_fields(monkeypatch):
    result = Mock(stdout='Auto Boost: Not Supported', stderr='')
    monkeypatch.setattr(clocks.subprocess, 'run', Mock(return_value=result))

    assert clocks.run_command(['fake-nvidia-smi', '-q', '-d', 'CLOCK']) == result.stdout
