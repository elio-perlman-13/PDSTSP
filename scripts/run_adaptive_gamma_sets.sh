#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 SOLVER INSTANCE_BASE REPETITION OUTPUT_DIR" >&2
  exit 2
fi

solver=$1
instance_base=$2
repetition=$3
output_dir=$4
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

gamma_sets=(
  "g1 0.3 0.2 0.1 0.3"
  "g2 0.5 0.3 0.1 0.3"
  "g3 0.3 0.2 0.1 0.6"
  "g4 0.5 0.3 0.1 0.6"
)

pids=()
for gamma_config in "${gamma_sets[@]}"; do
  read -r gamma_set gamma1 gamma2 gamma3 gamma4 <<<"$gamma_config"
  NEIGHBORHOOD_STRATEGY=adaptive \
  ADAPTIVE_GAMMA_SET="$gamma_set" \
  GAMMA1="$gamma1" GAMMA2="$gamma2" GAMMA3="$gamma3" GAMMA4="$gamma4" \
    bash "$script_dir/run_cyclic_time_dependent.sh" \
      "$solver" "$instance_base" "$repetition" "$output_dir" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
