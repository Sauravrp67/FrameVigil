#!/usr/bin/env bash
# Download detector checkpoints and the shared empty-road classifier.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/download_models.sh [--models-dir DIR] [--detector NAME]

Download all three detectors and the shared ERD checkpoint by default.
Requires curl and sha256sum; no Python, GPU, model chunks, or manifests needed.

  --models-dir DIR  Destination (default: <repository>/models)
  --detector NAME   Download only yolo11x, yolo26x or rtdetrv2, plus ERD
  -h, --help        Print help without downloading

Existing files are verified and reused. Interrupted downloads resume from .part
files. Checksum failures stop without replacing an existing checkpoint.
No model is loaded, trained, or run by this script.
EOF
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
models_dir="$repo_dir/models"
detector=
while (($#)); do
    case "$1" in
        --models-dir)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail '--models-dir requires a directory'
            models_dir=$2; shift 2 ;;
        --detector)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail '--detector requires a model name'
            detector=$2
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1 (use --help)" ;;
    esac
done
for tool in curl sha256sum; do
    command -v "$tool" >/dev/null || fail "Missing $tool; install curl and coreutils"
done
mkdir -p -- "$models_dir"
models_dir=$(cd -- "$models_dir" && pwd)

verify() {
    local actual
    actual=$(sha256sum -- "$1")
    [[ "${actual%% *}" == "$2" ]]
}

download() {
    local name=$1 url=$2 expected=$3 target="$models_dir/$1"
    mkdir -p -- "$(dirname -- "$target")"

    if [[ -f "$target" ]]; then
        verify "$target" "$expected" || fail "Checksum mismatch: $target. Move or remove it before retrying."
        printf 'Verified existing %s\n' "$name"
        return
    fi
    if [[ ! -f "$target.part" ]] || ! verify "$target.part" "$expected"; then
        printf 'Downloading %s\nSource: %s\n' "$name" "$url"
        curl --fail --location --show-error --retry 5 --retry-delay 3 \
            --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
            --continue-at - --output "$target.part" "$url" ||
            fail "Download failed. Rerun to resume $target.part; if the server rejects resume, remove that partial file first."
    fi
    verify "$target.part" "$expected" || fail "Checksum mismatch: $target.part. Remove the partial file and retry."
    mv -- "$target.part" "$target"
    printf 'Verified %s\n' "$name"
}

case "$detector" in
    "")
        detectors=(yolo11x yolo26x rtdetrv2)
        ;;
    yolo11x|yolo26x|rtdetrv2)
        detectors=("$detector")
        ;;
    *) fail "Unsupported detector: $detector (choose yolo11x, yolo26x or rtdetrv2)" ;;
esac

for detector in "${detectors[@]}"; do
    source "$repo_dir/scripts/model_downloads/$detector.sh"
done

download erd_bdd100k.pt \
    'https://raw.githubusercontent.com/amirtaherin/albireo/3f4a0f37aa51adf4b39bd1c671e5e92655216b23/erd_finetune/weights/erd_bdd100k.pt' \
    '7011bb54c737af841f542254cd7f7f5d1f4c0422035a5aa119e318fed1d6988b'
printf 'Models ready in %s\n' "$models_dir"
