#!/usr/bin/env bash
# Prepare the Python environment without replacing the system GPU installation.
set -euo pipefail

PROJECT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
ENVIRONMENT=${ENVIRONMENT:-"$PROJECT_DIRECTORY/.venv"}
CHECK_ONLY=false

case "${1:-}" in
    --check) CHECK_ONLY=true ;;
    --help)
        echo 'Usage: bash scripts/setup.sh [--check]'
        echo 'Optional: PYTHON=/path/to/python ENVIRONMENT=/path/to/venv'
        echo '--check reports readiness without installing anything.'
        exit 0
        ;;
    '') ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

IS_JETSON=false
L4T_VERSION=unknown

if [[ -r /etc/nv_tegra_release ]]; then
    IS_JETSON=true
    L4T_VERSION=$(sed -n 's/^# R\([0-9]*\).*REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)
fi

if [[ -x "$ENVIRONMENT/bin/python" ]]; then
    PYTHON="$ENVIRONMENT/bin/python"
fi

command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }
PYTHON_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
case "$PYTHON_VERSION" in
    3.10|3.11|3.12) ;;
    *) echo 'Use Python 3.10, 3.11, or 3.12 for the pinned numerical stack.' >&2; exit 1 ;;
esac

echo "Python: $PYTHON_VERSION; architecture: $(uname -m); Jetson: $IS_JETSON; L4T: $L4T_VERSION"

package_version() {
    "$PYTHON" -m pip show "$1" 2>/dev/null | sed -n 's/^Version: //p'
}

package_at_least() {
    local installed_version
    installed_version=$(package_version "$1") || return 1
    [[ -n "$installed_version" ]] || return 1
    dpkg --compare-versions "$installed_version" ge "$2"
}

check_numerical_packages() {
    [[ "$(package_version numpy)" == 1.26.4 ]] &&
        [[ "$(package_version opencv-python)" == 4.8.0.76 ]]
}

run_environment_tests() {
    "$PYTHON" "$PROJECT_DIRECTORY/tests/environment_check.py"
}

if $CHECK_ONLY; then
    run_environment_tests
    exit 0
fi

# A fresh Jetson needs a known wheel source before any installation begins.
# NVIDIA documents this pair for JetPack 6.2 (L4T 36.4.x), Python 3.10.
# https://docs.pytorch.org/TensorRT/getting_started/jetpack.html
INSTALL_TORCH=false
if ! package_at_least torch 2.1 || ! package_at_least torchvision 0.16; then
    INSTALL_TORCH=true
    if $IS_JETSON; then
        if [[ "$L4T_VERSION" != 36.4.3 && "$L4T_VERSION" != 36.4.4 ]] || [[ "$PYTHON_VERSION" != 3.10 ]]; then
            echo "Automatic GPU installation supports JetPack 6.2/6.2.1 with Python 3.10." >&2
            echo "Detected L4T $L4T_VERSION / Python $PYTHON_VERSION. Install a compatible GPU stack or select its interpreter with PYTHON." >&2
            exit 1
        fi
    fi
fi

# Install only missing OS packages; sudo may request the user's password.
SYSTEM_PACKAGES=(python3-venv python3-dev build-essential libopenblas-dev libjpeg-dev libpng-dev libgl1 libglib2.0-0 curl unzip ffmpeg)
MISSING_PACKAGES=()
for package in "${SYSTEM_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -qx 'install ok installed'; then
        MISSING_PACKAGES+=("$package")
    fi
done

if ((${#MISSING_PACKAGES[@]})); then
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
fi

if [[ ! -x "$ENVIRONMENT/bin/python" ]]; then
    "$PYTHON" -m venv --system-site-packages "$ENVIRONMENT"
fi
PYTHON="$ENVIRONMENT/bin/python"
"$PYTHON" -m pip install 'pip>=23' 'setuptools>=68' wheel

if ! check_numerical_packages; then
    "$PYTHON" -m pip install --no-deps 'numpy==1.26.4' 'opencv-python==4.8.0.76'
fi

if $INSTALL_TORCH; then
    if $IS_JETSON; then
        WHEEL_DIRECTORY=$(mktemp -d)
        trap 'rm -rf -- "$WHEEL_DIRECTORY"' EXIT

        # Select both Jetson builds explicitly, then resolve their dependencies
        # from PyPI without allowing a desktop Torch wheel to be substituted.
        "$PYTHON" -m pip download --no-deps --only-binary=:all: \
            --dest "$WHEEL_DIRECTORY" 'torch==2.8.0' 'torchvision==0.23.0' \
            --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
        "$PYTHON" -m pip install "$WHEEL_DIRECTORY"/*.whl 'numpy==1.26.4'
    else
        # Desktop setup defaults to CPU; it does not execute the local GPU.
        "$PYTHON" -m pip install 'torch==2.8.0' 'torchvision==0.23.0' \
            'numpy==1.26.4' --index-url https://download.pytorch.org/whl/cpu
    fi
fi

"$PYTHON" -m pip install --no-deps -r "$PROJECT_DIRECTORY/environments/requirements.txt"
"$PYTHON" -m pip install --no-deps -e "$PROJECT_DIRECTORY"

run_environment_tests

echo "Setup complete. Activate with: source $ENVIRONMENT/bin/activate"
echo 'Dataset and model downloads remain separate steps in the README.'
