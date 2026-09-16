#!/usr/bin/env bash
# Download the exact detector and empty-road classifier used in the experiments.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/download_models.sh [--models-dir DIR]

Download YOLO11x and ERD checkpoints (approximately 115 MB total).
Requires curl and sha256sum; no Python, GPU, model chunks, or manifests needed.

  --models-dir DIR  Destination (default: <repository>/models)
  -h, --help        Print help without downloading

Existing files are verified and reused. Interrupted downloads resume from .part
files. Checksum failures stop without replacing an existing checkpoint.
No model is loaded, trained, or run by this script.
EOF
}
fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
models_dir="$repo_dir/models"
while (($#)); do
    case "$1" in
        --models-dir)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail '--models-dir requires a directory'
            models_dir=$2; shift 2 ;;
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

# Official Ultralytics mirror, pinned revision; original ALBIREO ERD checkpoint,
# pinned source commit. License notices remain in licenses/.
download yolo11x.pt \
    'https://huggingface.co/Ultralytics/YOLO11/resolve/8b8ac7d1fae7468f85dbf89670dd66f41f485aab/yolo11x.pt' \
    '7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24'
download erd_bdd100k.pt \
    'https://raw.githubusercontent.com/amirtaherin/albireo/3f4a0f37aa51adf4b39bd1c671e5e92655216b23/erd_finetune/weights/erd_bdd100k.pt' \
    '7011bb54c737af841f542254cd7f7f5d1f4c0422035a5aa119e318fed1d6988b'
printf 'Models ready in %s\n' "$models_dir"
