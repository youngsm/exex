#!/usr/bin/env bash
set -euo pipefail

output_dir=$1
python=$2
fail_rank=$3
mkdir -p -- "$(dirname "$output_dir")"
mkdir -- "$output_dir"
printf '%s\n' "$PWD" > "$output_dir/driver-workdir.txt"
printf '%s\n' "$SLURM_JOB_ID-$RANDOM" > driver-token.txt
cp driver-token.txt "$output_dir/"

exec srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --kill-on-bad-exit=1 \
  "$python" worker.py "$output_dir" "$fail_rank"
