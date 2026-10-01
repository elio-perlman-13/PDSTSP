from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import statistics
import sys
from collections import Counter
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAFFIC_SCRIPTS_DIR = SCRIPT_DIR.parent
HANOI_15_SCRIPTS_DIR = TRAFFIC_SCRIPTS_DIR / "hanoi_15x15_2026"
HANOI_9_SCRIPTS_DIR = TRAFFIC_SCRIPTS_DIR / "hanoi_9_2026"
if str(TRAFFIC_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(TRAFFIC_SCRIPTS_DIR))
if str(HANOI_9_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(HANOI_9_SCRIPTS_DIR))

import build_speed_files as speed_builder
import select_100_points_min500m as selector


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


distance_builder = load_module(
    "hanoi_10x10_20_distance_builder",
    HANOI_15_SCRIPTS_DIR / "build_distance_matrices.py",
)
instance_builder = load_module(
    "hanoi_10x10_20_instance_builder",
    HANOI_15_SCRIPTS_DIR / "build_solver_instance.py",
)

TRAFFIC_DIR = SCRIPT_DIR.parents[2] / "datasets" / "hanoi_traffic"
OUTPUT_ROOT = TRAFFIC_DIR / "Hanoi_10x10_20instances_2026"
SPEED_PROFILE = (
    TRAFFIC_DIR
    / "experiment_speed_profiles"
    / "hanoi_traffic_speed_thu2_weekday_representative_10pct.csv"
)
DATASET_COUNT = 30
DATASETS = tuple(
    (f"set_{index:02d}", 20261100 + index)
    for index in range(1, DATASET_COUNT + 1)
)
POINTS_NAME = "nga_tu_so_10x10km_100_points_min500m.csv"
HOURS = list(range(6, 19))
HALF_SIDE_M = 5_000.0
MIN_DISTANCE_M = 500.0


def stem_for(dataset: str) -> str:
    return f"hanoi_10x10_100_{dataset}_weekday"


def generate_points(output_dir: Path, seed: int) -> Path:
    path = output_dir / POINTS_NAME
    selector.HALF_SIDE_M = HALF_SIDE_M
    selector.MIN_DISTANCE_M = MIN_DISTANCE_M
    selector.RANDOM_SEED = seed
    selector.OUTPUT = path
    selector.main()
    return path


def generate_distances(output_dir: Path, points_path: Path) -> None:
    distance_builder.POINTS_CSV = points_path
    distance_builder.TRUCK_OUTPUT = output_dir / "truck_distance_osm_m.txt"
    distance_builder.DRONE_OUTPUT = output_dir / "drone_distance_euclid_m.txt"
    distance_builder.main()


def generate_speeds(output_dir: Path, points_path: Path, stem: str) -> None:
    speed_builder.POINTS_CSV = points_path
    speed_builder.SPEED_CSV = SPEED_PROFILE
    speed_builder.HOURS = HOURS
    speed_builder.VMAX_OUTPUT = output_dir / f"{stem}.vmax_ij.txt"
    speed_builder.THETA_OUTPUT = output_dir / f"{stem}.theta_ijl.txt"
    speed_builder.SPEED_OUTPUT = output_dir / f"{stem}.v_ijl_kph.txt"
    speed_builder.main()


def generate_instance(
    output_dir: Path, points_path: Path, stem: str, seed: int
) -> None:
    instance_builder.POINTS_CSV = points_path
    instance_builder.INSTANCE = output_dir / f"{stem}.txt"
    instance_builder.DRONE_MATRIX = output_dir / f"{stem}.drone_distance_m.txt"
    instance_builder.TRUCK_MATRIX_100 = output_dir / "truck_distance_osm_m.txt"
    instance_builder.TRUCK_MATRIX = output_dir / f"{stem}.truck_distance_m.txt"
    instance_builder.DEPOT_TRUCK_CACHE = (
        output_dir / "depot_truck_distances_osrm.json"
    )
    instance_builder.RANDOM_SEED = seed
    points = instance_builder.load_points()
    instance_builder.write_instance(
        points, instance_builder.INSTANCE, trucks_count=1, drones_count=1
    )
    instance_builder.write_drone_matrix(points)
    instance_builder.write_truck_matrix(points)


def validate_and_describe(
    output_dir: Path, points_path: Path, dataset: str, seed: int
) -> tuple[dict[str, object], set[int]]:
    with points_path.open(encoding="utf-8-sig", newline="") as stream:
        points = list(csv.DictReader(stream))
    coordinates = [
        (float(point["offset_x_m"]), float(point["offset_y_m"]))
        for point in points
    ]
    minimum = min(
        math.dist(left, right)
        for index, left in enumerate(coordinates)
        for right in coordinates[index + 1 :]
    )
    source_ids = {int(point["source_point_id"]) for point in points}
    if len(points) != 100 or len(source_ids) != 100:
        raise ValueError(f"{dataset}: expected 100 distinct source points")
    if minimum + 1e-6 < MIN_DISTANCE_M:
        raise ValueError(f"{dataset}: minimum spacing is {minimum:.3f} m")
    if any(abs(value) > HALF_SIDE_M for coordinate in coordinates for value in coordinate):
        raise ValueError(f"{dataset}: point outside the 10 x 10 km square")

    stem = stem_for(dataset)
    expected_lines = {
        f"{stem}.vmax_ij.txt": 10_202,
        f"{stem}.theta_ijl.txt": 132_614,
        f"{stem}.v_ijl_kph.txt": 132_615,
        f"{stem}.truck_distance_m.txt": 10_202,
        f"{stem}.drone_distance_m.txt": 10_202,
    }
    for filename, expected in expected_lines.items():
        observed = sum(1 for _ in (output_dir / filename).open(encoding="utf-8"))
        if observed != expected:
            raise ValueError(
                f"{dataset}: {filename} has {observed} lines, expected {expected}"
            )

    districts = Counter(point["district"] for point in points)
    metadata = {
        "dataset": dataset,
        "customers": 100,
        "trucks": 1,
        "drones": 1,
        "center": {"name": "Nga Tu So", "lat": 21.0017, "lon": 105.8206},
        "square_side_km": 10,
        "minimum_required_spacing_m": MIN_DISTANCE_M,
        "observed_minimum_spacing_m": round(minimum, 3),
        "random_seed": seed,
        "point_source": str(selector.SOURCE),
        "truck_distance_source": str(distance_builder.FULL_TRUCK_MATRIX),
        "speed_profile": str(SPEED_PROFILE),
        "hours": HOURS,
        "segments": len(HOURS),
        "district_counts": dict(sorted(districts.items())),
        "edge_speed_rule": {
            "same_road_and_district": "road speed at the corresponding hour",
            "different_road_or_district": "arithmetic mean of endpoint road speeds",
            "depot": "mean speed profile of the 100 customer points",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "dataset": dataset,
        "seed": seed,
        "customers": 100,
        "minimum_spacing_m": round(minimum, 3),
        "districts": len(districts),
    }
    for district, count in sorted(districts.items()):
        summary[f"district_{district}"] = count
    return summary, source_ids


def write_summary(rows: list[dict[str, object]]) -> None:
    headers = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with (OUTPUT_ROOT / "instances_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main(start_index: int = 1, end_index: int = DATASET_COUNT) -> None:
    if not 1 <= start_index <= end_index <= DATASET_COUNT:
        raise ValueError(
            f"Expected 1 <= start_index <= end_index <= {DATASET_COUNT}"
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = DATASETS[start_index - 1 : end_index]
    for dataset, seed in selected:
        output_dir = OUTPUT_ROOT / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = stem_for(dataset)
        print(f"\nGenerating {dataset} with seed {seed}")
        points = generate_points(output_dir, seed)
        generate_distances(output_dir, points)
        generate_speeds(output_dir, points, stem)
        generate_instance(output_dir, points, stem, seed)
        validate_and_describe(output_dir, points, dataset, seed)

    summaries = []
    selected_sets = []
    for dataset, seed in DATASETS:
        output_dir = OUTPUT_ROOT / dataset
        points = output_dir / POINTS_NAME
        if not points.is_file():
            raise FileNotFoundError(
                f"Missing {dataset}; generate all sets before building the manifest"
            )
        summary, source_ids = validate_and_describe(output_dir, points, dataset, seed)
        summaries.append(summary)
        selected_sets.append(source_ids)
    write_summary(summaries)

    if len({frozenset(ids) for ids in selected_sets}) != len(DATASETS):
        raise ValueError("At least two instances have identical point sets")
    overlaps = [
        len(selected_sets[left] & selected_sets[right])
        for left in range(len(selected_sets))
        for right in range(left + 1, len(selected_sets))
    ]
    manifest = {
        "instances": len(DATASETS),
        "customers_per_instance": 100,
        "seeds": [seed for _, seed in DATASETS],
        "square_side_km": 10,
        "minimum_spacing_m": MIN_DISTANCE_M,
        "hours": HOURS,
        "speed_profile": str(SPEED_PROFILE),
        "pairwise_point_overlap": {
            "pairs": len(overlaps),
            "minimum": min(overlaps),
            "maximum": max(overlaps),
            "mean": statistics.fmean(overlaps),
        },
    }
    (OUTPUT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    shutil.copyfile(SPEED_PROFILE, OUTPUT_ROOT / SPEED_PROFILE.name)
    print(
        f"Generated {len(DATASETS)} distinct instances; pairwise overlap "
        f"range {min(overlaps)}..{max(overlaps)}, mean {statistics.fmean(overlaps):.2f}"
    )
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=DATASET_COUNT)
    args = parser.parse_args()
    main(args.start_index, args.end_index)
