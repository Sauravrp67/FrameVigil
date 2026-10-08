"""Check setup's read-only paths without installing packages or using a GPU (the environment suite is stubbed)."""

import os
from pathlib import Path
import subprocess

import pytest


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
SETUP_SCRIPT = PROJECT_DIRECTORY / 'scripts/setup.sh'


def test_setup_shell_syntax():
    subprocess.run(['bash', '-n', str(SETUP_SCRIPT)], check=True)


def test_setup_help_needs_no_python():
    environment = dict(os.environ, PYTHON='/missing/python')

    result = subprocess.run(
        ['bash', str(SETUP_SCRIPT), '--help'],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert '--check' in result.stdout


@pytest.mark.parametrize('probe_status', [0, 1])
def test_check_reports_probe_status_without_installing(tmp_path, probe_status):
    interpreter = tmp_path / 'python'
    calls = tmp_path / 'calls.txt'
    environment_directory = tmp_path / 'environment'

    interpreter.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\n" "$*" >> "$SETUP_TEST_CALLS"\n'
        'if [[ "$1" == --version ]]; then echo "Python 3.10.12"; exit 0; fi\n'
        f'exit {probe_status}\n'
    )
    interpreter.chmod(0o755)

    environment = dict(
        os.environ,
        PYTHON=str(interpreter),
        ENVIRONMENT=str(environment_directory),
        SETUP_TEST_CALLS=str(calls),
    )

    result = subprocess.run(
        ['bash', str(SETUP_SCRIPT), '--check'],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode == probe_status
    assert not environment_directory.exists()
    assert 'tests/environment_check.py' in calls.read_text()
    assert '-m pip' not in calls.read_text()
    assert '-m venv' not in calls.read_text()


def test_setup_contains_no_embedded_python():
    source = SETUP_SCRIPT.read_text()

    assert '"$PYTHON" -c' not in source
    assert "<<'PY'" not in source
    assert '"$PYTHON" - <<' not in source
