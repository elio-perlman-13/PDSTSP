#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


CUSTOMER_GROUPS = {
    "50": (50,),
    "100": (100,),
    "200": (200,),
}
BUNDLE_SIZES = {"50": 5, "100": 2, "200": 1}
SPATIAL_INDICES = (10, 20, 30, 40)
INSTANCE_INDICES = (1, 3)
SEGMENT_FACTORS = (1, 3, 5)
SEEDS = (1001, 2002, 3003, 4004, 5005)
MPS_PER_MPH = 0.44704
VMAX_MIN_MPS = 24.0 * MPS_PER_MPH
VMAX_MAX_MPS = 35.0 * MPS_PER_MPH


def read_header(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with path.open(encoding="utf-8") as stream:
        for _ in range(3):
            fields = stream.readline().split()
            if len(fields) != 2:
                raise ValueError(f"Invalid instance header in {path}")
            values[fields[0]] = int(fields[1])
    return values


def data_values(path: Path, column: int):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split()
            if len(fields) <= column:
                raise ValueError(f"Malformed data row in {path}: {stripped}")
            yield float(fields[column])


def validate_instance_data(instance_file: Path, vmax_file: Path, theta_file: Path, n: int) -> None:
    lines = instance_file.read_text(encoding="utf-8").splitlines()
    customer_rows = [line.split() for line in lines[6 : 6 + n]]
    if len(customer_rows) != n or any(len(row) < 7 for row in customer_rows):
        raise ValueError(f"Expected {n} customer rows in {instance_file}")
    demands = [float(row[3]) for row in customer_rows]
    if not all(0.0 < demand <= 2.0 for demand in demands):
        raise ValueError(f"Demand outside (0, 2] kg in {instance_file}")

    vmax_values = list(data_values(vmax_file, 2))
    theta_values = list(data_values(theta_file, 3))
    expected_edges = n * (n + 1)
    if len(vmax_values) != expected_edges:
        raise ValueError(
            f"Expected {expected_edges} directed vmax records in {vmax_file}, found {len(vmax_values)}"
        )
    if len(theta_values) != expected_edges * 12:
        raise ValueError(
            f"Expected {expected_edges * 12} theta records in {theta_file}, found {len(theta_values)}"
        )
    if not all(VMAX_MIN_MPS <= speed <= VMAX_MAX_MPS for speed in vmax_values):
        raise ValueError(f"Truck vmax outside 24-35 mph in {vmax_file}")
    if not all(0.4 <= theta <= 1.0 for theta in theta_values):
        raise ValueError(f"Theta outside [0.4, 1.0] in {theta_file}")


def build_tasks(group: str, data_dir: Path) -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for n in CUSTOMER_GROUPS[group]:
        iteration_budget = 9 * n * math.ceil(math.sqrt(n))
        for spatial_index in SPATIAL_INDICES:
            for instance_index in INSTANCE_INDICES:
                instance = f"{n}.{spatial_index}.{instance_index}"
                instance_file = data_dir / f"{instance}.txt"
                vmax_file = data_dir / f"{instance}.vmax_ij.txt"
                theta_file = data_dir / f"{instance}.theta_ijl.txt"
                missing = [str(path) for path in (instance_file, vmax_file, theta_file) if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f"Missing files for {instance}: {', '.join(missing)}")

                header = read_header(instance_file)
                if header.get("customers") != n:
                    raise ValueError(
                        f"Customer count mismatch for {instance}: expected {n}, "
                        f"found {header.get('customers')}"
                    )
                validate_instance_data(instance_file, vmax_file, theta_file, n)

                for segment_factor in SEGMENT_FACTORS:
                    segment_iterations = segment_factor * n
                    for run, seed in enumerate(SEEDS, start=1):
                        task_id = f"{instance}-L{segment_factor}n-run{run}"
                        tasks.append(
                            {
                                "task_id": task_id,
                                "instance": instance,
                                "instance_file": str(instance_file),
                                "vmax_file": str(vmax_file),
                                "theta_file": str(theta_file),
                                "n": n,
                                "spatial_index": spatial_index,
                                "instance_index": instance_index,
                                "segment_factor": segment_factor,
                                "segment_iterations": segment_iterations,
                                "iteration_budget": iteration_budget,
                                "run": run,
                                "seed": seed,
                            }
                        )
    return tasks


def build_batches(group: str, tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    bundle_size = BUNDLE_SIZES[group]
    batches = []
    for index in range(0, len(tasks), bundle_size):
        batch_tasks = tasks[index : index + bundle_size]
        batches.append(
            {
                "batch_id": f"{group}-batch{len(batches) + 1:03d}",
                "task_count": len(batch_tasks),
                "tasks_json": json.dumps(batch_tasks, separators=(",", ":")),
            }
        )
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(CUSTOMER_GROUPS), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("instance_time_dependent"))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    tasks = build_tasks(args.group, args.data_dir)
    expected_tasks = 120
    if len(tasks) != expected_tasks:
        raise SystemExit(f"Expected {expected_tasks} tasks for {args.group}, found {len(tasks)}")

    batches = build_batches(args.group, tasks)
    expected_batches = expected_tasks // BUNDLE_SIZES[args.group]
    if len(batches) != expected_batches or sum(batch["task_count"] for batch in batches) != expected_tasks:
        raise SystemExit(f"Invalid batch construction for {args.group}")

    matrix = {"include": batches}
    print(json.dumps(matrix, indent=2 if args.pretty else None, separators=None if args.pretty else (",", ":")))


if __name__ == "__main__":
    main()
