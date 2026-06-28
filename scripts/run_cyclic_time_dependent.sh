#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 SOLVER INSTANCE_BASE REPETITION OUTPUT_DIR" >&2
  exit 2
fi

solver=$(realpath "$1")
instance_base=$2
repetition=$3
output_dir=$4
repo_root=${GITHUB_WORKSPACE:-$(pwd)}
instance_dir="$repo_root/instance_time_dependent"
seed=$((41 + repetition))
strategy=${NEIGHBORHOOD_STRATEGY:-cyclic}
gamma_set=${ADAPTIVE_GAMMA_SET:-}
gamma1=${GAMMA1:-}
gamma2=${GAMMA2:-}
gamma3=${GAMMA3:-}
gamma4=${GAMMA4:-}
solver_time_limit_sec=${SOLVER_TIME_LIMIT_SEC:-}

if [[ "$strategy" != "cyclic" && "$strategy" != "random" && "$strategy" != "adaptive" ]]; then
  echo "Invalid NEIGHBORHOOD_STRATEGY: $strategy" >&2
  exit 2
fi

strategy_args=("--neighborhood-selection=$strategy")
result_suffix="_strategy_${strategy}"
if [[ "$strategy" == "adaptive" ]]; then
  if [[ -z "$gamma_set" || -z "$gamma1" || -z "$gamma2" || -z "$gamma3" || -z "$gamma4" ]]; then
    echo "Adaptive runs require ADAPTIVE_GAMMA_SET and GAMMA1..GAMMA4" >&2
    exit 2
  fi
  strategy_args+=("--gamma1=$gamma1" "--gamma2=$gamma2" "--gamma3=$gamma3" "--gamma4=$gamma4")
  result_suffix+="_gamma_${gamma_set}"
fi
if [[ -n "$solver_time_limit_sec" ]]; then
  strategy_args+=("--time-limit=$solver_time_limit_sec")
fi

instance_file="$instance_dir/$instance_base.txt"
vmax_file="$instance_dir/$instance_base.vmax_ij.txt"
theta_file="$instance_dir/$instance_base.theta_ijl.txt"

for required_file in "$solver" "$instance_file" "$vmax_file" "$theta_file"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Missing required file: $required_file" >&2
    exit 2
  fi
done

mkdir -p "$output_dir"
output_dir=$(realpath "$output_dir")
work_dir=$(mktemp -d "${RUNNER_TEMP:-/tmp}/cyclic-${instance_base}-run${repetition}-XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

result_file="$output_dir/${instance_base}_run_${repetition}${result_suffix}.txt"
log_file="$output_dir/${instance_base}_run_${repetition}${result_suffix}.log"

pushd "$work_dir" >/dev/null
set +e
solver_command=(
  "$solver" "$instance_file"
  --truck-vmax-file="$vmax_file"
  --truck-theta-file="$theta_file"
  "${strategy_args[@]}"
  --attempts=1
  --seed="$seed"
)
if [[ -n "$solver_time_limit_sec" ]] && command -v timeout >/dev/null 2>&1; then
  timeout --signal=TERM --kill-after=30 "${solver_time_limit_sec}s" \
    "${solver_command[@]}" >"$log_file" 2>&1
else
  "${solver_command[@]}" >"$log_file" 2>&1
fi
solver_exit_code=$?
set -e

solver_termination="completed normally"
if [[ "$solver_exit_code" -eq 124 || "$solver_exit_code" -eq 137 || "$solver_exit_code" -eq 143 ]]; then
  solver_termination="time limit reached; best checkpoint preserved"
fi

{
  echo "Instance: $instance_base"
  echo "Customer group: ${instance_base%%.*}"
  echo "Repetition: $repetition"
  echo "Experiment strategy: $strategy"
  echo "Gamma set: $gamma_set"
  echo "Gamma1: $gamma1"
  echo "Gamma2: $gamma2"
  echo "Gamma3: $gamma3"
  echo "Gamma4: $gamma4"
  echo "Solver time limit seconds: $solver_time_limit_sec"
  echo "Experiment seed: $seed"
  echo "Solver exit code: $solver_exit_code"
  echo "Solver termination: $solver_termination"
  if [[ -f output_solution_best.txt ]]; then
    cat output_solution_best.txt
  else
    echo "Neighborhood selection: $strategy"
    echo "Random seed: $seed"
    echo "Simulated annealing: disabled"
    echo "Diversification: disabled"
    echo "Initial solution cost:"
    echo "Improved solution cost:"
    echo "Worst solution cost:"
    echo "Mean solution cost:"
    echo "Mean elapsed time:"
    echo "Final solution feasibility: FAILED"
  fi
} >"$result_file"
popd >/dev/null

echo "Completed $instance_base repetition $repetition using $strategy with exit code $solver_exit_code"
