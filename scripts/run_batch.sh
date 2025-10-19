#!/usr/bin/env bash
set -euo pipefail

# Batch-run tabubu over every instance and save only per-instance best outputs

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

echo "[run_batch] Building solver (tabubu)..."
g++-13 -O3 -std=c++17 -o tabubu tabubu.cpp

timestamp="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="outputs/batch_${timestamp}"
mkdir -p "$OUT_DIR"

shopt -s nullglob
files=(instance/*.txt)

if (( ${#files[@]} == 0 )); then
  echo "[run_batch] No instance files found under instance/." >&2
  exit 1
fi

echo "[run_batch] Will run ${#files[@]} instances; outputs will be saved to: $OUT_DIR"

idx=0
for f in "${files[@]}"; do
  ((idx++)) || true
  base=$(basename "$f" .txt)
  echo "[run_batch] ($idx/${#files[@]}) Running: $f"
  # Run solver (suppress program stdout) and copy the generated best file with a unique name
  ./tabubu "$f" >/dev/null
  if [[ -f output_solution_best.txt ]]; then
    cp -f output_solution_best.txt "$OUT_DIR/${base}_best.txt"
  else
    echo "[run_batch][warn] output_solution_best.txt not found after running $f"
  fi
done

echo "[run_batch] All done. Results saved in: $OUT_DIR"
echo "[run_batch] Summary of generated files:"
ls -1 "$OUT_DIR" | sed "s/^/ - /"
