#!/usr/bin/env bash
# Install missing dependencies, then test the selected Python environment.
set -euo pipefail

PROJECT_DIRECTORY=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
ENVIRONMENT=${ENVIRONMENT:-"$PROJECT_DIRECTORY/.venv"}
DEVICE=${DEVICE:-auto}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-}
OPENCV_INDEX_URL=${OPENCV_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}
CHECK_ONLY=false

case "${1:-}" in
    --check) CHECK_ONLY=true ;;
    --help)
        echo 'Usage: bash scripts/setup.sh [--check]'
        echo 'Optional: PYTHON=/path/to/python ENVIRONMENT=/path/to/venv'
        echo 'DEVICE=auto|cuda|cpu (default: auto); TORCH_INDEX_URL overrides wheel selection.'
        echo 'OPENCV_INDEX_URL overrides the mirror used for NumPy and OpenCV.'
        echo '--check runs functional tests without installing anything.'
        exit 0
        ;;
    '') ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac

fail() {
    echo "Error: $*" >&2
    exit 1
}

case "$DEVICE" in
    auto|cuda|cpu) ;;
    *) fail 'DEVICE must be auto, cuda, or cpu' ;;
esac

if [[ -x "$ENVIRONMENT/bin/python" ]]; then
    PYTHON="$ENVIRONMENT/bin/python"
fi

command -v "$PYTHON" >/dev/null || fail "Python not found: $PYTHON"
PYTHON_VERSION=$("$PYTHON" --version 2>&1 | awk '{print $2}')
dpkg --compare-versions "$PYTHON_VERSION" ge 3.10 || fail 'Python 3.10 or newer is required'

IS_JETSON=false
L4T_VERSION=unknown

if [[ -r /etc/nv_tegra_release ]]; then
    IS_JETSON=true
    L4T_VERSION=$(sed -n 's/^# R\([0-9]*\).*REVISION: \([0-9.]*\).*/\1.\2/p' /etc/nv_tegra_release)
fi

CUDA_VERSION=''
if command -v nvidia-smi >/dev/null; then
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -1) || true
fi

if [[ -z "$CUDA_VERSION" ]]; then
    CUDA_COMPILER=$(command -v nvcc || true)
    if [[ -z "$CUDA_COMPILER" && -x /usr/local/cuda/bin/nvcc ]]; then
        CUDA_COMPILER=/usr/local/cuda/bin/nvcc
    fi

    if [[ -n "$CUDA_COMPILER" ]]; then
        CUDA_VERSION=$("$CUDA_COMPILER" --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -1)
    fi
fi

if [[ "$DEVICE" == auto ]]; then
    if $IS_JETSON || [[ -n "$CUDA_VERSION" || -e /dev/nvidiactl ]]; then
        DEVICE=cuda
    else
        DEVICE=cpu
    fi
fi

export FRAMEVIGIL_DEVICE="$DEVICE"
echo "Python: $PYTHON_VERSION; architecture: $(uname -m); device: $DEVICE; L4T: $L4T_VERSION; CUDA: ${CUDA_VERSION:-not detected}"

package_version() {
    "$PYTHON" -m pip show "$1" 2>/dev/null | sed -n 's/^Version: //p'
}

package_at_least() {
    local installed_version
    installed_version=$(package_version "$1") || return 1
    [[ -n "$installed_version" ]] || return 1
    dpkg --compare-versions "$installed_version" ge "$2"
}

run_environment_tests() {
    "$PYTHON" "$PROJECT_DIRECTORY/tests/environment_check.py"
}

if $CHECK_ONLY; then
    run_environment_tests
    exit 0
fi

# Keep OS installation separate from Python package installation.
SYSTEM_PACKAGES=(python3-venv python3-pip python3-dev build-essential libopenblas-dev libjpeg-dev libpng-dev libgl1 libglib2.0-0 curl unzip ffmpeg)
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

TEMPORARY_DIRECTORY=$(mktemp -d)
trap 'rm -rf -- "$TEMPORARY_DIRECTORY"' EXIT
CONSTRAINTS="$TEMPORARY_DIRECTORY/installed-stack.txt"
touch "$CONSTRAINTS"

# Preserve an adequate existing NumPy when resolving a missing OpenCV build.
if package_at_least numpy 1.26; then
    echo "numpy==$(package_version numpy)" >> "$CONSTRAINTS"
fi
echo "Installing NumPy and OpenCV from $OPENCV_INDEX_URL"
"$PYTHON" -m pip install -c "$CONSTRAINTS" 'numpy>=1.26' 'opencv-python>=4.8' \
    --index-url "$OPENCV_INDEX_URL"

if package_at_least torch 2.1 && package_at_least torchvision 0.16; then
    echo 'Reusing installed PyTorch and torchvision; functional tests run below.'
else
    if [[ -z "$TORCH_INDEX_URL" ]]; then
        if [[ "$DEVICE" == cpu ]]; then
            TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
        else
            [[ -n "$CUDA_VERSION" ]] || fail 'CUDA version could not be detected. Check the driver/toolkit or set TORCH_INDEX_URL.'
            CUDA_CODE=${CUDA_VERSION/./}

            if $IS_JETSON && [[ "${L4T_VERSION%%.*}" =~ ^[0-9]+$ ]] && (( ${L4T_VERSION%%.*} < 38 )); then
                # Older Tegra releases need platform-specific wheels rather
                # than the standard ARM64 wheels used by JetPack 7 and later.
                TORCH_INDEX_URL="https://pypi.jetson-ai-lab.io/jp6/cu${CUDA_CODE}"
            else
                # Discover published CUDA builds, including future releases.
                # Select the newest build no newer than the detected CUDA support.
                INDEX_LIST=$(curl -fsSL --connect-timeout 15 --max-time 60 https://download.pytorch.org/whl/)
                AVAILABLE_BUILDS=$(printf '%s\n' "$INDEX_LIST" | sed -n 's/.*href="cu\([0-9]*\)\/".*/\1/p' | sort -rn)

                for build in $AVAILABLE_BUILDS; do
                    if (( build <= CUDA_CODE )); then
                        TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${build}"
                        break
                    fi
                done
                [[ -n "$TORCH_INDEX_URL" ]] || fail "No published PyTorch CUDA index matches CUDA $CUDA_VERSION"
            fi
        fi
    fi

    echo "Installing PyTorch and torchvision from $TORCH_INDEX_URL"
    # Resolve the pair together: torchvision declares its matching Torch version.
    # pip selects wheels for this Python version and architecture.
    "$PYTHON" -m pip install -c "$CONSTRAINTS" --only-binary=:all: 'torch>=2.1' 'torchvision>=0.16' \
        --index-url "$TORCH_INDEX_URL" || fail 'PyTorch installation failed. Inspect the pip error above; no CPU fallback was installed.'
fi

# Resolve support packages without replacing the selected numerical/GPU stack.
for package in numpy opencv-python torch torchvision; do
    echo "$package==$(package_version "$package")" >> "$CONSTRAINTS"
done

# TrackEval's distribution requires a newer desktop stack than its metric code
# needs. Its NumPy/SciPy/plotting dependencies are already listed explicitly.
grep -v '^trackeval' "$PROJECT_DIRECTORY/environments/requirements.txt" > "$TEMPORARY_DIRECTORY/requirements.txt"
grep '^trackeval' "$PROJECT_DIRECTORY/environments/requirements.txt" > "$TEMPORARY_DIRECTORY/tracking.txt"
"$PYTHON" -m pip install -c "$CONSTRAINTS" -r "$TEMPORARY_DIRECTORY/requirements.txt"
"$PYTHON" -m pip install --no-deps -r "$TEMPORARY_DIRECTORY/tracking.txt"
"$PYTHON" -m pip install --no-deps -e "$PROJECT_DIRECTORY"

run_environment_tests

echo "Setup complete. Activate with: source $ENVIRONMENT/bin/activate"
echo 'Dataset and model downloads remain separate steps.'
