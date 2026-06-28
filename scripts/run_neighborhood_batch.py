#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("solver", type=Path)
    parser.add_argument("tasks_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    tasks = json.loads(args.tasks_json.read_text(encoding="utf-8"))
    if not 1 <= len(tasks) <= 20:
        raise SystemExit(f"Each balanced job must contain between one and twenty tasks, found {len(tasks)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_cyclic_time_dependent.sh")
    processes = []
    for task in tasks:
        environment = os.environ.copy()
        environment["NEIGHBORHOOD_STRATEGY"] = task["strategy"]
        if task["strategy"] == "adaptive":
            environment["ADAPTIVE_GAMMA_SET"] = task["gamma_set"]
            for index, value in enumerate(task["gamma"], start=1):
                environment[f"GAMMA{index}"] = str(value)
        command = [
            "bash",
            str(runner),
            str(args.solver),
            task["instance"],
            str(task["repetition"]),
            str(args.output_dir),
        ]
        processes.append((task, subprocess.Popen(command, env=environment)))

    failed = []
    for task, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed.append((task, return_code))
    if failed:
        raise SystemExit(f"Failed tasks: {failed}")


if __name__ == "__main__":
    main()
