#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path


IMPROVED_RE = re.compile(r"Improved Solution Cost:\s*([0-9.eE+-]+)")
FEAS_RE = re.compile(r"Final solution feasibility:\s*(FEASIBLE|INFEASIBLE)")


def read_customer_count(instance_path):
    with Path(instance_path).open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[0] == "customers":
                return int(parts[1])
    raise ValueError(f"Cannot find customer count in {instance_path}")


def iteration_budget(instance_path):
    n = read_customer_count(instance_path)
    return int(9 * n * math.ceil(math.sqrt(n)))


def parse_solver_output(text):
    cost = None
    feasible = None
    for match in IMPROVED_RE.finditer(text):
        cost = float(match.group(1))
    for match in FEAS_RE.finditer(text):
        feasible = match.group(1)
    return cost, feasible


def run_task(binary, instance, task, args, out_dir):
    task_name = f"task_{task['task_id']:04d}_cfg_{task['config_id']:03d}_seed_{task['seed']}"
    task_dir = out_dir / "logs" / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    log_path = task_dir / "solver.log"

    cmd = [
        str(binary),
        str(instance),
        "--attempts=1",
        f"--iters={args.iters}",
        f"--time-limit={args.time_limit_sec}",
        "--truck-capacity=400",
        "--drone-capacity=2.27",
        f"--seed={task['seed']}",
        f"--segment-iters={task['Lseg']}",
        f"--c-tabu={task['c_tabu']}",
        f"--h-mode={task['Hmode']}",
        f"--h-div={task['Hdiv']}",
        "--knn-k=50",
        "--knn-window=1",
        "--gamma1=0.6",
        "--gamma2=0.3",
        "--gamma3=0.05",
        "--gamma4=0.4",
        "--T0=0.5",
        "--alpha=0.998",
        "--kappa=1.0",
        "--tau-v=0.01",
        "--r-destroy=0.2",
    ]

    started = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=task_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.job_task_timeout_sec,
        check=False,
    )
    runtime_sec = time.monotonic() - started
    log_path.write_text(completed.stdout, encoding="utf-8")
    cost, feasible = parse_solver_output(completed.stdout)

    if cost is None:
        objective = 1_000_000_000_000.0
    elif feasible == "INFEASIBLE":
        objective = cost + 1_000_000_000.0
    else:
        objective = cost

    return {
        "task_id": task["task_id"],
        "config_id": task["config_id"],
        "seed_index": task["seed_index"],
        "seed": task["seed"],
        "Lseg": task["Lseg"],
        "c_tabu": task["c_tabu"],
        "Hmode": task["Hmode"],
        "Hdiv": task["Hdiv"],
        "cost": "" if cost is None else f"{cost:.6f}",
        "objective": f"{objective:.6f}",
        "feasible": feasible or "UNKNOWN",
        "exit_code": completed.returncode,
        "runtime_sec": f"{runtime_sec:.3f}",
        "log_path": str(log_path.relative_to(out_dir)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--time-limit-sec", type=int, default=1800)
    parser.add_argument("--job-task-timeout-sec", type=int, default=2100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.iters is None:
        args.iters = iteration_budget(args.instance)
    tasks = json.loads(Path(args.tasks_json).read_text(encoding="utf-8"))
    rows = []

    for task in tasks:
        try:
            rows.append(run_task(Path(args.binary).resolve(), Path(args.instance).resolve(), task, args, out_dir))
        except subprocess.TimeoutExpired:
            rows.append({
                "task_id": task["task_id"],
                "config_id": task["config_id"],
                "seed_index": task["seed_index"],
                "seed": task["seed"],
                "Lseg": task["Lseg"],
                "c_tabu": task["c_tabu"],
                "Hmode": task["Hmode"],
                "Hdiv": task["Hdiv"],
                "cost": "",
                "objective": "1000000000000.000000",
                "feasible": "TIMEOUT",
                "exit_code": 124,
                "runtime_sec": str(args.job_task_timeout_sec),
                "log_path": "",
            })

    csv_path = out_dir / "results.csv"
    fieldnames = [
        "task_id", "config_id", "seed_index", "seed",
        "Lseg", "c_tabu", "Hmode", "Hdiv",
        "cost", "objective", "feasible", "exit_code", "runtime_sec", "log_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
