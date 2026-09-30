#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


TASK_ARGUMENTS = (
    "instance",
    "instance_file",
    "vmax_file",
    "theta_file",
    "n",
    "spatial_index",
    "instance_index",
    "segment_factor",
    "segment_iterations",
    "iteration_budget",
    "run",
    "seed",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--time-limit-sec", type=int, default=1800)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = json.loads(args.tasks_json)
    if not isinstance(tasks, list) or not tasks:
        raise SystemExit("The batch must contain at least one task")

    runner = Path(__file__).with_name("v2_segment_run.py").resolve()
    binary = args.binary.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for position, task in enumerate(tasks, start=1):
        missing = [name for name in TASK_ARGUMENTS if name not in task]
        if missing:
            raise SystemExit(f"Task {position} is missing fields: {', '.join(missing)}")
        task_id = task.get("task_id", f"task-{position}")
        print(f"[{args.batch_id}] Starting {position}/{len(tasks)}: {task_id}", flush=True)
        command = [
            sys.executable,
            str(runner),
            "--binary", str(binary),
            "--time-limit-sec", str(args.time_limit_sec),
            "--result-dir", str(result_dir),
        ]
        for name in TASK_ARGUMENTS:
            command.extend((f"--{name.replace('_', '-')}", str(task[name])))
        completed = subprocess.run(command, check=False)
        results.append({"task_id": task_id, "exit_code": completed.returncode})
        print(f"[{args.batch_id}] Finished {task_id} with exit code {completed.returncode}", flush=True)

    summary = {
        "batch_id": args.batch_id,
        "task_count": len(tasks),
        "successful_tasks": sum(item["exit_code"] == 0 for item in results),
        "failed_tasks": sum(item["exit_code"] != 0 for item in results),
        "tasks": results,
    }
    (result_dir / f"batch_summary_{args.batch_id}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0 if summary["failed_tasks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
