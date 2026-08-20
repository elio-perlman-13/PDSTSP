#!/usr/bin/env python3
import argparse
import json


def round1_configs():
    config_id = 1
    for lseg in [100, 200, 300, 400, 500]:
        for c_tabu in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for hmode in [1, 2, 3]:
                for hdiv in [3, 4, 5]:
                    if hdiv <= hmode:
                        continue
                    yield {
                        "config_id": config_id,
                        "Lseg": lseg,
                        "c_tabu": c_tabu,
                        "Hmode": hmode,
                        "Hdiv": hdiv,
                    }
                    config_id += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-jobs", type=int, default=250)
    parser.add_argument("--seeds", default="1001,2002,3003")
    args = parser.parse_args()

    if args.num_jobs < 1:
        raise SystemExit("--num-jobs must be positive")

    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    tasks = []
    task_id = 1
    for config in round1_configs():
        for seed_index, seed in enumerate(seeds, start=1):
            tasks.append({
                "task_id": task_id,
                "round": "round1",
                "seed_index": seed_index,
                "seed": seed,
                **config,
            })
            task_id += 1

    shards = [[] for _ in range(args.num_jobs)]
    for index, task in enumerate(tasks):
        shards[index % args.num_jobs].append(task)

    matrix = []
    for shard_index, shard_tasks in enumerate(shards):
        matrix.append({
            "shard_id": f"{shard_index + 1:03d}",
            "task_count": len(shard_tasks),
            "tasks_json": json.dumps(shard_tasks, separators=(",", ":")),
        })

    print(json.dumps(matrix, separators=(",", ":")))


if __name__ == "__main__":
    main()
