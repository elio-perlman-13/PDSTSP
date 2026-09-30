#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path


NUMBER = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"


def parse_last(pattern: str, text: str) -> str:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else ""


def count_iterations(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8", errors="replace") as stream:
        return max(0, sum(1 for _ in stream) - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--instance-file", type=Path, required=True)
    parser.add_argument("--vmax-file", type=Path, required=True)
    parser.add_argument("--theta-file", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--spatial-index", type=int, required=True)
    parser.add_argument("--instance-index", type=int, required=True)
    parser.add_argument("--segment-factor", type=int, choices=(1, 3, 5), required=True)
    parser.add_argument("--segment-iterations", type=int, required=True)
    parser.add_argument("--iteration-budget", type=int, required=True)
    parser.add_argument("--run", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--time-limit-sec", type=int, default=1800)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    expected_budget = 9 * args.n * math.ceil(math.sqrt(args.n))
    expected_segment = args.segment_factor * args.n
    if args.iteration_budget != expected_budget:
        raise SystemExit(f"Invalid iteration budget: {args.iteration_budget} != {expected_budget}")
    if args.segment_iterations != expected_segment:
        raise SystemExit(f"Invalid segment length: {args.segment_iterations} != {expected_segment}")
    tabu_base = max(20, math.floor(2 * math.sqrt(args.n)))

    binary = args.binary.resolve()
    instance_file = args.instance_file.resolve()
    vmax_file = args.vmax_file.resolve()
    theta_file = args.theta_file.resolve()
    for path in (binary, instance_file, vmax_file, theta_file):
        if not path.is_file():
            raise SystemExit(f"Required file does not exist: {path}")

    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"{args.instance}-L{args.segment_factor}n-run{args.run}"
    log_path = result_dir / f"{task_id}.log"

    command = [
        str(binary),
        str(instance_file),
        f"--truck-vmax-file={vmax_file}",
        f"--truck-theta-file={theta_file}",
        "--attempts=1",
        f"--iters={args.iteration_budget}",
        f"--segment-iters={args.segment_iterations}",
        f"--time-limit={args.time_limit_sec}",
        "--segment-length-sec=3600",
        "--truck-capacity=1000",
        "--drone-capacity=2.27",
        f"--seed={args.seed}",
        "--knn-k=1000",
        "--knn-window=1",
        f"--tabu-base={tabu_base}",
        "--h-mode=1000000000",
        "--h-div=1000000000",
        "--gamma1=0.5",
        "--gamma2=0.3",
        "--gamma3=0.1",
        "--gamma4=0.3",
        "--T0=0",
        "--alpha=0.998",
        "--kappa=0",
        "--tau-v=0.01",
        "--r-destroy=0",
    ]

    started = time.monotonic()
    cpu_started = resource.getrusage(resource.RUSAGE_CHILDREN)
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=result_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=args.time_limit_sec + 300,
            check=False,
        )
        output = completed.stdout
        exit_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output += f"\nEXTERNAL TIMEOUT after {args.time_limit_sec + 300} seconds\n"
        exit_code = 124
        timed_out = True
    runtime_sec = time.monotonic() - started
    cpu_finished = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_time_sec = (
        cpu_finished.ru_utime - cpu_started.ru_utime
        + cpu_finished.ru_stime - cpu_started.ru_stime
    )
    log_path.write_text("Command: " + " ".join(command) + "\n\n" + output, encoding="utf-8")

    history_path = result_dir / "output_mode_0.txt"
    solution_path = result_dir / "output_solution_best.txt"
    executed_iterations = count_iterations(history_path)
    feasibility = parse_last(r"Final solution feasibility:\s*(FEASIBLE|INFEASIBLE)", output)
    improved_cost = parse_last(r"Improved Solution Cost:\s*" + NUMBER, output)
    initial_cost = parse_last(r"Initial Solution Cost:\s*" + NUMBER, output)
    solver_elapsed = parse_last(r"Mean elapsed Time:\s*" + NUMBER, output)

    if timed_out:
        termination_reason = "EXTERNAL_TIMEOUT"
    elif exit_code != 0:
        termination_reason = "FAILED"
    elif executed_iterations >= args.iteration_budget:
        termination_reason = "ITERATION_LIMIT"
    else:
        termination_reason = "TIME_LIMIT"

    status = "SUCCESS" if exit_code == 0 and improved_cost else "FAILED"
    saved_solution_path = result_dir / f"final_solution_{task_id}.txt"
    saved_history_path = result_dir / f"iteration_history_{task_id}.csv"
    if solution_path.is_file():
        shutil.move(str(solution_path), saved_solution_path)
    if history_path.is_file():
        shutil.move(str(history_path), saved_history_path)
    row = {
        "task_id": task_id,
        "status": status,
        "instance": args.instance,
        "n": args.n,
        "spatial_index": args.spatial_index,
        "instance_index": args.instance_index,
        "segment_factor": args.segment_factor,
        "segment_iterations": args.segment_iterations,
        "iteration_budget": args.iteration_budget,
        "run": args.run,
        "seed": args.seed,
        "time_limit_sec": args.time_limit_sec,
        "executed_iterations": executed_iterations,
        "termination_reason": termination_reason,
        "initial_cost": initial_cost,
        "improved_makespan_s": improved_cost,
        "feasible": feasibility or "UNKNOWN",
        "solver_elapsed_sec": solver_elapsed,
        "cpu_time_sec": f"{cpu_time_sec:.3f}",
        "wall_runtime_sec": f"{runtime_sec:.3f}",
        "exit_code": exit_code,
        "log_file": log_path.name,
        "solution_file": saved_solution_path.name if saved_solution_path.is_file() else "",
        "history_file": saved_history_path.name if saved_history_path.is_file() else "",
    }

    csv_path = result_dir / f"result_{task_id}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    metadata = {
        **row,
        "command": command,
        "physical_parameters": {
            "truck_capacity_kg": 1000.0,
            "drone_capacity_kg": 2.27,
            "drone_energy_capacity_j": 7_200_000.0,
            "drone_cruise_speed_mps": 31.2928,
            "drone_ascent_speed_mps": 15.6464,
            "drone_descent_speed_mps": 7.8232,
            "drone_altitude_m": 50.0,
            "power_beta_w_per_kg": 24.2,
            "power_gamma_w": 1329.0,
        },
        "adaptive_rewards": {
            "configuration": "A2",
            "gamma1": 0.5,
            "gamma2": 0.3,
            "gamma3": 0.1,
            "gamma4": 0.3,
        },
        "controlled_search": {
            "adaptive_neighborhood_selection": True,
            "mode_switching": False,
            "simulated_annealing_acceptance": False,
            "destroy_and_repair": False,
            "penalty_lambdas_fixed_at": 1.0,
            "tabu_base": tabu_base,
            "tabu_tenure_interval": [tabu_base, 2 * tabu_base],
        },
    }
    (result_dir / f"metadata_{task_id}.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if status == "SUCCESS" else exit_code or 1


if __name__ == "__main__":
    raise SystemExit(main())
