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


@pytest.mark.parametrize(
    'python_version, l4t_version, cuda_version, expected_index',
    [
        ('3.12.3', '39.2.1', '13.2', 'https://download.pytorch.org/whl/cu132'),
        ('3.13.7', '40.1.0', '13.5', 'https://download.pytorch.org/whl/cu134'),
        ('3.12.3', '38.2.0', '13.1', 'https://download.pytorch.org/whl/cu130'),
        ('3.10.12', '36.4.4', '12.6', 'https://pypi.jetson-ai-lab.io/jp6/cu126'),
        ('3.12.3', None, '12.8', 'https://download.pytorch.org/whl/cu126'),
        ('3.13.7', None, '', 'https://download.pytorch.org/whl/cpu'),
    ],
)
def test_installer_selects_runtime_without_real_installation(
    tmp_path, python_version, l4t_version, cuda_version, expected_index
):
    tools, environment, script, calls = make_installer_fixture(
        tmp_path, python_version, l4t_version, cuda_version
    )

    result = subprocess.run(
        ['bash', str(script)], env=environment, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    commands = calls.read_text()
    assert f'--index-url {expected_index}' in commands
    assert 'tests/environment_check.py' in commands
    assert 'numpy==1.26.4' not in commands
    assert 'opencv-python==4.8.0.76' not in commands
    requirements_command = next(line for line in commands.splitlines() if ' -r ' in line)
    assert 'environments/requirements.txt' in requirements_command
    assert '--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple' in requirements_command
    assert '--no-deps' not in requirements_command
    download_command = next(line for line in commands.splitlines() if '-m pip download' in line)
    assert '--no-deps' in download_command

    install_command = next(line for line in commands.splitlines() if 'torch-wheels/torch-' in line)
    assert '--index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple' in install_command
    assert '--no-deps' not in install_command


def make_installer_fixture(tmp_path, python_version, l4t_version, cuda_version):
    tools = tmp_path / 'tools'
    tools.mkdir()
    script = tmp_path / 'scripts/setup.sh'
    script.parent.mkdir()
    release_file = tmp_path / 'nv_tegra_release'

    if l4t_version is not None:
        major, revision = l4t_version.split('.', maxsplit=1)
        release_file.write_text(f'# R{major} (release), REVISION: {revision}, BOARD: generic\n')

    script.write_text(
        SETUP_SCRIPT.read_text()
        .replace('/etc/nv_tegra_release', str(release_file))
        .replace('/dev/nvidiactl', str(tmp_path / 'nvidiactl'))
    )
    requirements = tmp_path / 'environments/requirements.txt'
    requirements.parent.mkdir()
    requirements.write_text('scipy>=1.11\ntrackeval==1.3.0\n')
    interpreter = tmp_path / 'environment/bin/python'
    interpreter.parent.mkdir(parents=True)
    calls = tmp_path / 'calls.txt'
    interpreter.write_text(
        '#!/usr/bin/env bash\n'
        'echo "$*" >> "$SETUP_TEST_CALLS"\n'
        f'if [[ "$1" == --version ]]; then echo "Python {python_version}"; exit 0; fi\n'
        'if [[ "$*" == "-m pip show "* ]]; then\n'
        '  case "$4" in\n'
        '    numpy) echo "Version: 2.2.6" ;;\n'
        '    opencv-python) echo "Version: 4.12.0.88" ;;\n'
        '    torch|torchvision)\n'
        '      if [[ "${SETUP_TEST_REUSE:-0}" != 1 && ! -f "$SETUP_TEST_INSTALLED" ]]; then exit 1; fi\n'
        '      if [[ "$4" == torch ]]; then echo "Version: 2.11.0+cu130"; else echo "Version: 0.26.0+cu130"; fi ;;\n'
        '  esac\n'
        'fi\n'
        'if [[ "$*" == "-m pip download "* ]]; then\n'
        '  while [[ "$1" != --dest ]]; do shift; done\n'
        '  mkdir -p "$2"\n'
        '  touch "$2/torch-test.whl" "$2/torchvision-test.whl"\n'
        'fi\n'
        'if [[ "$*" == "-m pip install "* && "$*" == *"torch-wheels/"* ]]; then touch "$SETUP_TEST_INSTALLED"; fi\n'
        'exit 0\n'
    )
    interpreter.chmod(0o755)

    for name, body in {
        'nvidia-smi': f'echo "CUDA Version: {cuda_version}"' if cuda_version else 'exit 1',
        'nvcc': 'exit 0',
        'uname': 'echo aarch64' if l4t_version else 'echo x86_64',
        'dpkg-query': 'echo "install ok installed"',
        'curl': 'echo \'<a href="cu126/">cu126</a>\'; echo \'<a href="cu130/">cu130</a>\'; echo \'<a href="cu132/">cu132</a>\'; echo \'<a href="cu134/">cu134</a>\'',
        'sudo': 'echo "Unexpected sudo call" >&2; exit 99',
    }.items():
        tool = tools / name
        tool.write_text('#!/usr/bin/env bash\n' + body + '\n')
        tool.chmod(0o755)

    environment = dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ['PATH'],
        PYTHON=str(interpreter),
        ENVIRONMENT=str(interpreter.parent.parent),
        SETUP_TEST_CALLS=str(calls),
        SETUP_TEST_INSTALLED=str(tmp_path / 'torch_installed'),
        DEVICE='auto',
        TORCH_INDEX_URL='',
        OPENCV_INDEX_URL='',
        PACKAGE_INDEX_URL='',
    )

    return tools, environment, script, calls


def test_existing_torch_is_reused_without_a_version_whitelist(tmp_path):
    tools, environment, script, calls = make_installer_fixture(
        tmp_path, '3.13.7', '41.7.2', '13.5'
    )
    environment['SETUP_TEST_REUSE'] = '1'
    (tools / 'curl').write_text('#!/usr/bin/env bash\nexit 99\n')

    result = subprocess.run(
        ['bash', str(script)], env=environment, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert 'torch>=2.1' not in calls.read_text()
    assert 'Reusing installed PyTorch' in result.stdout


def test_missing_cuda_information_does_not_silently_install_cpu_torch(tmp_path):
    tools, environment, script, calls = make_installer_fixture(
        tmp_path, '3.12.3', '39.2.1', ''
    )

    result = subprocess.run(
        ['bash', str(script)], env=environment, capture_output=True, text=True
    )

    assert result.returncode != 0
    assert 'CUDA version could not be detected' in result.stderr
    assert 'torch>=2.1' not in calls.read_text()
