#!/usr/bin/env bash
# Prepare the complete BDD100K MOT 2020 validation split, independently of
# Python, model weights, prediction caches, and historical study manifests.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/prepare_dataset.sh [--data-dir DIR] [--archive-dir DIR]

Download, verify and extract all 200 BDD100K MOT 2020 validation sequences.
Requires Bash, curl, unzip and GNU coreutils (Linux/Jetson).

  --data-dir DIR     Destination containing bdd100k/ (default: <repo>/data)
  --archive-dir DIR  Keep/reuse downloaded ZIPs (default: <data-dir>/archives)
  -h, --help         Show this help without downloading anything

Interrupted downloads resume from .part files. Reruns overwrite dataset files from the verified archives.
Only validation labels are extracted from the train/validation label archive.
EOF
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 1; }
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
data_dir="$repo_dir/data"
archive_dir=''
while (($#)); do
    case "$1" in
        --data-dir|--archive-dir)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || fail "$1 requires a directory"
            if [[ "$1" == --data-dir ]]; then data_dir="$2"; else archive_dir="$2"; fi
            shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) fail "Unknown option: $1 (use --help)" ;;
    esac
done
archive_dir=${archive_dir:-"$data_dir/archives"}
for tool in curl unzip md5sum find sort diff wc sed; do
    command -v "$tool" >/dev/null || fail "Missing $tool; on Ubuntu install curl unzip coreutils findutils diffutils"
done
mkdir -p -- "$data_dir" "$archive_dir"
# Absolute paths also allow directory names containing spaces.
data_dir=$(cd -- "$data_dir" && pwd)
archive_dir=$(cd -- "$archive_dir" && pwd)

images_zip='images20-track-val-1.zip'
labels_zip='box_track_labels_trainval.zip'

images_url="https://huggingface.co/datasets/vanthanh/BDD100kMOT/resolve/main/$images_zip"
labels_url="https://huggingface.co/datasets/mrhwl123/BDD100kMOT/resolve/main/$labels_zip?download=true"

# Published archive MD5s: https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst
# These identify the original dataset bytes, independently of the mirror.

images_md5='743ab4c7b5ff8eeebd60bbfd15b20bd6'
labels_md5='6be40e0ca56a83ddeba2ed6bff50f9e6'

verify_zip() {
    local path=$1 expected=$2 actual
    actual=$(md5sum -- "$path")
    [[ "${actual%% *}" == "$expected" ]]
}

download_zip() {
    local url=$1 name=$2 expected=$3 target="$archive_dir/$2"
    if [[ -f "$target" ]]; then
        printf 'Verifying cached %s\n' "$name"
        verify_zip "$target" "$expected" || fail "Checksum mismatch: $target. Remove or replace this archive, then rerun."
        return
    fi
    printf 'Downloading %s\n' "$url"
    # A complete .part may remain if a previous run stopped before rename.
    if [[ ! -f "$target.part" ]] || ! verify_zip "$target.part" "$expected"; then
        curl --fail --location --show-error --retry 5 --retry-delay 3 \
            --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
            --continue-at - --output "$target.part" "$url" ||
            fail "Download failed; partial data kept at $target.part. Rerun to resume. If the server rejects resume, remove that partial file first."
    fi
    verify_zip "$target.part" "$expected" || fail "Checksum mismatch: $target.part. Remove this partial file and rerun."
    mv -- "$target.part" "$target"
}

download_zip "$images_url" "$images_zip" "$images_md5"
download_zip "$labels_url" "$labels_zip" "$labels_md5"

printf 'Extracting all validation images into %s/bdd100k/images/track/val/\n' "$data_dir"
unzip -q -o "$archive_dir/$images_zip" 'bdd100k/images/track/val/*' -d "$data_dir"

printf 'Extracting all validation annotations into %s/bdd100k/labels/box_track_20/val/\n' "$data_dir"
unzip -q -o "$archive_dir/$labels_zip" 'bdd100k/labels/box_track_20/val/*' -d "$data_dir"

images_dir="$data_dir/bdd100k/images/track/val"
labels_dir="$data_dir/bdd100k/labels/box_track_20/val"

video_count=$(find "$images_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
image_count=$(find "$images_dir" -mindepth 2 -maxdepth 2 -type f -name '*.jpg' | wc -l)
label_count=$(find "$labels_dir" -maxdepth 1 -type f -name '*.json' | wc -l)

[[ "$video_count" -eq 200 && "$image_count" -eq 39973 && "$label_count" -eq 200 ]] ||
    fail "Unexpected inventory: $video_count videos, $image_count images, $label_count annotation files"

diff -u <(find "$images_dir" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | LC_ALL=C sort) \
    <(find "$labels_dir" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sed 's/\.json$//' | LC_ALL=C sort) ||
    fail 'Image sequence names and annotation names do not match'

printf 'Dataset ready: %s/bdd100k\n200 videos, 39,973 images, 200 annotation files.\n' "$data_dir"
