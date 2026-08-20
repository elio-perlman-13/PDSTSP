#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


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


def ffloat(value, default=math.inf):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--out-dir", default="round1_aggregate")
    parser.add_argument("--instance", default="instance_time_dependent/100.10.1.txt")
    parser.add_argument("--expected-runs", type=int, default=600)
    parser.add_argument("--expected-configs", type=int, default=200)
    parser.add_argument("--expected-runs-per-config", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in sorted(results_dir.rglob("results.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            rows.extend(csv.DictReader(fh))

    if not rows:
        raise SystemExit(f"No results.csv files found under {results_dir}")

    all_runs_path = out_dir / "round1_all_runs.csv"
    fieldnames = list(rows[0].keys())
    with all_runs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_config = defaultdict(list)
    for row in rows:
        by_config[row["config_id"]].append(row)

    summary_rows = []
    for config_id, config_rows in by_config.items():
        objectives = [ffloat(row["objective"]) for row in config_rows]
        costs = [ffloat(row["cost"]) for row in config_rows if row["cost"]]
        mean_objective = sum(objectives) / len(objectives)
        mean_cost = sum(costs) / len(costs) if costs else math.inf
        feasible_count = sum(1 for row in config_rows if row["feasible"] == "FEASIBLE")
        runtime = sum(ffloat(row["runtime_sec"], 0.0) for row in config_rows)
        first = config_rows[0]
        variance = sum((x - mean_objective) ** 2 for x in objectives) / len(objectives)
        summary_rows.append({
            "config_id": int(config_id),
            "runs": len(config_rows),
            "feasible_runs": feasible_count,
            "mean_objective": mean_objective,
            "mean_cost": mean_cost,
            "std_objective": math.sqrt(variance),
            "total_runtime_sec": runtime,
            "Lseg": int(first["Lseg"]),
            "c_tabu": float(first["c_tabu"]),
            "Hmode": int(first["Hmode"]),
            "Hdiv": int(first["Hdiv"]),
        })

    summary_rows.sort(key=lambda row: (row["mean_objective"], row["std_objective"], -row["feasible_runs"], row["config_id"]))

    summary_path = out_dir / "round1_summary_by_config.csv"
    summary_fields = [
        "rank", "config_id", "runs", "feasible_runs", "mean_objective", "mean_cost",
        "std_objective", "total_runtime_sec", "Lseg", "c_tabu", "Hmode", "Hdiv",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for rank, row in enumerate(summary_rows, start=1):
            output = dict(row)
            output["rank"] = rank
            output["mean_objective"] = f"{row['mean_objective']:.6f}"
            output["mean_cost"] = "" if math.isinf(row["mean_cost"]) else f"{row['mean_cost']:.6f}"
            output["std_objective"] = f"{row['std_objective']:.6f}"
            output["total_runtime_sec"] = f"{row['total_runtime_sec']:.3f}"
            writer.writerow(output)

    best = summary_rows[0]
    best_json = {
        "round": "round1",
        "selection_rule": "min mean_objective, then min std_objective",
        **best,
    }
    best_json["mean_objective"] = round(best_json["mean_objective"], 6)
    best_json["mean_cost"] = None if math.isinf(best_json["mean_cost"]) else round(best_json["mean_cost"], 6)
    best_json["std_objective"] = round(best_json["std_objective"], 6)

    (out_dir / "round1_best_config.json").write_text(json.dumps(best_json, indent=2) + "\n", encoding="utf-8")

    fixed_args = [
        f"--iters={iteration_budget(args.instance)}",
        "--knn-k=50",
        "--knn-window=1",
        f"--segment-iters={best['Lseg']}",
        f"--c-tabu={best['c_tabu']}",
        f"--h-mode={best['Hmode']}",
        f"--h-div={best['Hdiv']}",
        "--T0=0.5",
        "--alpha=0.998",
        "--kappa=1.0",
        "--tau-v=0.01",
        "--r-destroy=0.2",
    ]
    (out_dir / "round1_best_for_round2.fixed.args").write_text("\n".join(fixed_args) + "\n", encoding="utf-8")

    issues = []
    if len(rows) != args.expected_runs:
        issues.append(f"Expected {args.expected_runs} total runs, found {len(rows)}")
    if len(by_config) != args.expected_configs:
        issues.append(f"Expected {args.expected_configs} configs, found {len(by_config)}")
    incomplete = sorted(
        int(config_id)
        for config_id, config_rows in by_config.items()
        if len(config_rows) != args.expected_runs_per_config
    )
    if incomplete:
        sample = ", ".join(map(str, incomplete[:20]))
        if len(incomplete) > 20:
            sample += ", ..."
        issues.append(
            f"{len(incomplete)} configs do not have {args.expected_runs_per_config} runs: {sample}"
        )

    validation_report = {
        "total_runs": len(rows),
        "configs": len(by_config),
        "expected_runs": args.expected_runs,
        "expected_configs": args.expected_configs,
        "expected_runs_per_config": args.expected_runs_per_config,
        "issues": issues,
    }
    (out_dir / "round1_validation_report.json").write_text(
        json.dumps(validation_report, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Merged {len(rows)} runs from {len(by_config)} configs")
    print(f"Best config: {best_json}")
    if issues:
        print("Validation issues:")
        for issue in issues:
            print(f"- {issue}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
