#!/usr/bin/env python3
"""
create_hanoi_instance.py
------------------------
Reads od_speed_by_time_slot.csv and generates a PDSTSP instance file.

Node mapping: P0000 → depot (node 0), P0001..P000N → customers (1..N).
2-D coordinates are reconstructed from pairwise road distances using MDS.
Missing OD pairs are filled via Floyd-Warshall before MDS.

Sample customer parameters (configurable via CLI):
    --serve   ~2 min service time (uniform in [serve_lo, serve_hi])
    --demand  0.7-2.0 kg demand   (uniform in [demand_lo, demand_hi])
    --deadline-options            (sample each customer deadline from a list)
  --trucks  number of trucks (default 3)
  --drones  number of drones (default 3)
  --seed    RNG seed (default 42)
  --out     output path (default instance_hanoi/hanoi_<n>.txt)

Usage:
  python create_hanoi_instance.py od_speed_by_time_slot.csv
  python create_hanoi_instance.py od_speed_by_time_slot.csv --trucks=2 --drones=2 --out=instance_hanoi/test.txt
"""

import argparse
import csv
import math
import os
import random
import sys
from typing import Dict, List, Tuple

try:
    import numpy as np
except ImportError:
    sys.exit("Need numpy: pip install numpy")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_file", help="Path to od_speed_by_time_slot.csv")
    p.add_argument("--distance-csv", type=str, default="", help="Optional OD distance matrix CSV (label matrix)")
    p.add_argument("--trucks",     type=int,   default=3)
    p.add_argument("--drones",     type=int,   default=3)
    p.add_argument("--serve-lo",   type=float, default=105.0,  help="Min service time (s)")
    p.add_argument("--serve-hi",   type=float, default=135.0,  help="Max service time (s)")
    p.add_argument("--demand-lo",  type=float, default=0.7,   help="Min demand (kg)")
    p.add_argument("--demand-hi",  type=float, default=2.0,   help="Max demand (kg)")
    p.add_argument("--deadline",   type=float, default=3600.0, help="Fallback deadline per customer (s)")
    p.add_argument(
        "--deadline-options",
        type=str,
        default="2400,3600,4200,4800",
        help="Comma-separated deadline choices in seconds (sampled per customer)",
    )
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--out",        type=str,   default="",    help="Output file path (auto if empty)")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------

SLOT_ORDER = ["05-07", "07-09", "09-11", "11-13", "13-15", "15-17"]
FREE_FLOW_KMH = 25.0   # uncongested reference speed (km/h)
# time_segment boundaries corresponding to SLOT_ORDER
TIME_BOUNDARIES = [5, 7, 9, 11, 13, 15, 17]
PROGRESS_EVERY = 1_000_000


def load_distance_matrix_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("Distance CSV is empty")
    n = len(rows[0]) - 1
    mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        vals = rows[i + 1][1:1 + n]
        if len(vals) < n:
            raise ValueError(f"Distance CSV row {i+1} has too few columns")
        mat[i, :] = [float(v) if str(v).strip() else 0.0 for v in vals]
    return mat


def load_csv(path: str, distance_csv: str = ""):
    """Return (node_list, dist_mat, arc_sigma) where arc_sigma[i][j][s] = speed/FREE_FLOW."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

    use_idx_schema = ("origin_idx" in fieldnames and "destination_idx" in fieldnames)

    # First pass: gather node metadata and time-slot values without keeping all rows.
    slot_set = set()

    if use_idx_schema:
        max_idx = -1
        labels = {}
        row_count = 0
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                row_count += 1
                if row_count % PROGRESS_EVERY == 0:
                    print(f"Pass 1: scanned {row_count} rows...", flush=True)
                ts = str(r.get("time_slot", "")).strip()
                if ts:
                    slot_set.add(ts)

                oi = str(r.get("origin_idx", "")).strip()
                dj = str(r.get("destination_idx", "")).strip()
                if not oi or not dj:
                    continue
                i = int(oi)
                j = int(dj)
                max_idx = max(max_idx, i, j)

                if "origin_label" in r and str(r["origin_label"]).strip():
                    labels[i] = str(r["origin_label"]).strip()
                if "destination_label" in r and str(r["destination_label"]).strip():
                    labels[j] = str(r["destination_label"]).strip()

        if row_count == 0 or max_idx < 0:
            raise ValueError("Speed CSV has no valid data rows")

        N = max_idx + 1
        node_list = [labels.get(i, f"P{i:04d}") for i in range(N)]
        node_idx = None
    else:
        node_set = set()
        row_count = 0
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                row_count += 1
                if row_count % PROGRESS_EVERY == 0:
                    print(f"Pass 1: scanned {row_count} rows...", flush=True)
                ts = str(r.get("time_slot", "")).strip()
                if ts:
                    slot_set.add(ts)

                oi = str(r.get("origin_id", "")).strip()
                dj = str(r.get("destination_id", "")).strip()
                if oi:
                    node_set.add(oi)
                if dj:
                    node_set.add(dj)

        if row_count == 0 or not node_set:
            raise ValueError("Speed CSV has no valid data rows")

        node_list = sorted(node_set)
        N = len(node_list)
        node_idx = {n: i for i, n in enumerate(node_list)}

    if not slot_set:
        raise ValueError("Speed CSV missing time_slot values")

    def slot_start(slot: str) -> int:
        try:
            return int(slot.split("-")[0])
        except Exception:
            return 0

    slot_order = sorted(slot_set, key=slot_start)
    slot_idx = {s: i for i, s in enumerate(slot_order)}
    num_slots = len(slot_order)

    dist_mat  = np.full((N, N), np.inf)
    np.fill_diagonal(dist_mat, 0.0)
    arc_sigma = np.full((N, N, num_slots), np.nan)

    # Second pass: fill speed and (optional) distance arrays.
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        parsed = 0
        for r in reader:
            parsed += 1
            if parsed % PROGRESS_EVERY == 0:
                print(f"Pass 2: parsed {parsed} rows...", flush=True)
            if use_idx_schema:
                oi = str(r.get("origin_idx", "")).strip()
                dj = str(r.get("destination_idx", "")).strip()
                if not oi or not dj:
                    continue
                i = int(oi)
                j = int(dj)
            else:
                oi = str(r.get("origin_id", "")).strip()
                dj = str(r.get("destination_id", "")).strip()
                if oi not in node_idx or dj not in node_idx:
                    continue
                i = node_idx[oi]
                j = node_idx[dj]

            s = slot_idx.get(str(r.get("time_slot", "")).strip(), -1)
            if s < 0:
                continue

            sp = str(r.get("avg_speed_kmh", "")).strip()
            if sp:
                speed = float(sp)
                arc_sigma[i][j][s] = speed / FREE_FLOW_KMH

            dm = str(r.get("distance_m", "")).strip()
            if dm and dist_mat[i][j] == np.inf:
                dist_mat[i][j] = float(dm)

    # If no distances embedded, use optional distance matrix CSV.
    if np.isinf(dist_mat).any() and distance_csv:
        ext = load_distance_matrix_csv(distance_csv)
        if ext.shape[0] >= N and ext.shape[1] >= N:
            dist_mat[:N, :N] = ext[:N, :N]

    # Last-resort synthetic distances from speed (for MDS/compatibility only)
    if np.isinf(dist_mat).any():
        arc_mean = np.nanmean(arc_sigma, axis=2)
        arc_mean = np.where(np.isnan(arc_mean) | (arc_mean <= 1e-9), 1.0, arc_mean)
        fill = 1000.0 / arc_mean
        mask = np.isinf(dist_mat)
        dist_mat[mask] = fill[mask]
        np.fill_diagonal(dist_mat, 0.0)

    # Floyd-Warshall is cubic and infeasible for large N in Python.
    if N <= 250:
        for k in range(N):
            col_k = dist_mat[:, [k]]
            row_k = dist_mat[[k], :]
            via_k = col_k + row_k
            dist_mat = np.minimum(dist_mat, via_k)
    else:
        print(f"Skipping Floyd-Warshall for N={N} (too large for O(N^3) in Python).")

    # Fill missing arc_sigma slots with slot-wise means
    slot_mean = np.nanmean(arc_sigma.reshape(-1, num_slots), axis=0)
    slot_mean = np.where(np.isnan(slot_mean), 1.0, slot_mean)
    for s in range(num_slots):
        mask = np.isnan(arc_sigma[:, :, s])
        arc_sigma[mask, s] = slot_mean[s]

    return node_list, dist_mat, arc_sigma


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------

def placeholder_coords(node_count: int) -> np.ndarray:
    # Coordinates are placeholders only; solver should use --distance-csv for travel distances.
    return np.zeros((node_count, 2), dtype=float)


# ---------------------------------------------------------------------------
# Write instance file
# ---------------------------------------------------------------------------

def write_instance(path: str, coords: np.ndarray, node_list: List[str],
                   arc_sigma: np.ndarray, args):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rng = random.Random(args.seed)

    N    = len(node_list)           # depot + customers
    nc   = N - 1                    # number of customers
    depot_x, depot_y = coords[0]

    # Per-customer random data
    serve   = [round(rng.uniform(args.serve_lo,  args.serve_hi),  1) for _ in range(nc)]
    demand  = [round(rng.uniform(args.demand_lo, args.demand_hi), 4) for _ in range(nc)]
    deadline_choices = [float(x.strip()) for x in args.deadline_options.split(",") if x.strip()]
    if not deadline_choices:
        deadline_choices = [float(args.deadline)]
    Dd      = 2.27                  # drone max payload (from solver)
    dronable = [1 if demand[i] <= Dd else 0 for i in range(nc)]
    deadline = [rng.choice(deadline_choices) for _ in range(nc)]

    num_slots = arc_sigma.shape[2]

    with open(path, "w") as f:
        f.write(f"trucks_count {args.trucks}\n")
        f.write(f"drones_count {args.drones}\n")
        f.write(f"customers {nc}\n")
        f.write(f"depot {depot_x:.4f} {depot_y:.4f}\n")
        f.write("Coordinate X         Coordinate Y         Dronable Demand\n")
        f.write("X\tY\tDronable\tDemand\tDrone_service\tTruck_service\tLw\n")
        for ci in range(1, N):       # customer indices 1..N-1
            x, y = coords[ci]
            idx  = ci - 1           # 0-based into serve/demand arrays
            f.write(
                f"{x:.4f}\t{y:.4f}\t{dronable[idx]:.1f}\t"
                f"{demand[idx]:.4f}\t{serve[idx]:.1f}\t{serve[idx]:.1f}\t"
                f"{deadline[idx]:.1f}\n"
            )
        # arc_sigma block: i j sigma_s0 sigma_s1 ... sigma_s{K-1}
        f.write("arc_sigma_start\n")
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                sigmas = " ".join(f"{arc_sigma[i][j][s]:.4f}" for s in range(num_slots))
                f.write(f"{i} {j} {sigmas}\n")
        f.write("arc_sigma_end\n")

    print(f"Instance written to {path}")
    print(f"  {nc} customers, {args.trucks} trucks, {args.drones} drones")
    print(f"  Service time: [{args.serve_lo}, {args.serve_hi}] s")
    print(f"  Demand:       [{args.demand_lo}, {args.demand_hi}] kg")
    print(f"  Deadline options: {', '.join(str(int(v)) for v in sorted(set(deadline_choices)))} s")
    print(f"  Time slots:   {num_slots}")
    print(f"  Arc-sigma: {N}x{N}x{num_slots} table embedded in file")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"Loading {args.csv_file} ...")
    node_list, dist_mat, arc_sigma = load_csv(args.csv_file, args.distance_csv)
    N  = len(node_list)
    nc = N - 1
    print(f"  {N} nodes ({nc} customers), {arc_sigma.shape[2]} time slots")

    print("Skipping coordinate reconstruction; using placeholder coordinates.")
    coords = placeholder_coords(N)

    out_path = args.out or f"instance_hanoi/hanoi_{nc}.txt"
    write_instance(out_path, coords, node_list, arc_sigma, args)

    # Print run command
    print()
    print("Run solver:")
    print(f"  g++ -O3 -std=c++20 tabubu_time.cpp -o tabubu_time")
    run = f"  ./tabubu_time {out_path} --speed-csv={args.csv_file} --time-limit=120"
    if args.distance_csv:
        run += f" --distance-csv={args.distance_csv}"
    else:
        run += "  # add --distance-csv=od_distance_m.csv to use real OD distances"
    print(run)


if __name__ == "__main__":
    main()

#Run:   python3 create_hanoi_instance.py od_speed_by_time_slot_hanoi_aggressive.csv --distance-csv=/workspaces/PDSTSP/od_distance_m.csv --out=/workspaces/PDSTSP/instance_hanoi/hanoi_125_generated.txt
