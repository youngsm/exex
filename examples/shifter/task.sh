#!/usr/bin/env bash
set -euo pipefail

input_dir=$1
output_dir=$2
printf '%s\n' "$CUDA_VISIBLE_DEVICES" > "$output_dir/task-mask.txt"

exec shifter --module=gpu --workdir="$PWD" \
  --volume="$input_dir:/mnt:ro" --volume="$output_dir:/media" \
  --env="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  --env="PROBE_MESSAGE=$PROBE_MESSAGE" --env="PROBE_IMAGE_ID=$PROBE_IMAGE_ID" \
  --env="CUPY_CACHE_DIR=$PWD/.cupy" \
  -- python3 worker.py /mnt/input.txt /media "$3" "$4"
