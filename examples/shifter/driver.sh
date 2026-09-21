#!/usr/bin/env bash
set -euo pipefail

input_dir=$1
output_dir=$2
mkdir -p -- "$(dirname "$output_dir")"
mkdir -- "$output_dir"
printf '%s\n' "$PWD" > "$output_dir/driver-workdir.txt"
printf '%s\n' "${CUDA_VISIBLE_DEVICES:-}" > "$output_dir/batch-mask.txt"

exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --kill-on-bad-exit=1 \
  bash task.sh "$input_dir" "$output_dir" "$3" "$4"
