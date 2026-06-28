#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path


METHODS = [
    {"strategy": "cyclic"},
    {"strategy": "random"},
    {"strategy": "adaptive", "gamma_set": "g1", "gamma": [0.3, 0.2, 0.1, 0.3]},
    {"strategy": "adaptive", "gamma_set": "g2", "gamma": [0.5, 0.3, 0.1, 0.3]},
    {"strategy": "adaptive", "gamma_set": "g3", "gamma": [0.3, 0.2, 0.1, 0.6]},
    {"strategy": "adaptive", "gamma_set": "g4", "gamma": [0.5, 0.3, 0.1, 0.6]},
]


def result_name(task):
    suffix = f"_strategy_{task['strategy']}"
    if task["strategy"] == "adaptive":
        suffix += f"_gamma_{task['gamma_set']}"
    return f"{task['instance']}_run_{task['repetition']}{suffix}.txt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recovered_results", type=Path)
    parser.add_argument("instance_dir", type=Path)
    parser.add_argument("matrix_json", type=Path)
    parser.add_argument("missing_json", type=Path)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()

    instances = sorted(
        (
            path.name.removesuffix(".txt")
            for path in args.instance_dir.glob("200.*.txt")
            if not path.name.endswith((".vmax_ij.txt", ".theta_ijl.txt"))
        ),
        key=lambda name: tuple(int(part) for part in name.split(".")),
    )
    if len(instances) != 16:
        raise SystemExit(f"Expected 16 instances, found {len(instances)}")

    tasks = [
        {"instance": instance, "repetition": repetition, **method}
        for instance in instances
        for method in METHODS
        for repetition in range(1, 6)
    ]
    existing = {path.name for path in args.recovered_results.rglob("*.txt")}
    expected_names = {result_name(task) for task in tasks}
    unexpected = sorted(existing - expected_names)
    if unexpected:
        raise SystemExit(f"Unexpected recovered result files: {unexpected[:10]}")

    missing = [task for task in tasks if result_name(task) not in existing]
    if len(missing) > 256:
        raise SystemExit(f"Missing matrix has {len(missing)} jobs, exceeding GitHub's 256-job limit")
    matrix = [
        {
            "task_id": result_name(task).removesuffix(".txt"),
            "tasks_json": json.dumps([task], separators=(",", ":")),
        }
        for task in missing
    ]

    args.matrix_json.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    args.missing_json.write_text(json.dumps(missing, indent=2), encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"matrix={json.dumps(matrix, separators=(',', ':'))}\n")
            output.write(f"recovered_count={len(existing)}\n")
            output.write(f"missing_count={len(missing)}\n")
    print(f"Recovered {len(existing)} configurations; missing {len(missing)} of {len(tasks)}")


if __name__ == "__main__":
    main()
