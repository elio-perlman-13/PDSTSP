#!/usr/bin/env python3

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np


SLOTS = ["5-7", "7-9", "9-11", "11-13", "13-15", "15-17", "17-19", "19-21"]

# Rule 2: weekday-averaFor each 2-hour slot, we average Mon-Fri congestion percentages from the table,ge TomTom congestion mapped to speed factors.
# 
# then convert by: speed_factor = 1 / (1 + congestion_fraction).
F_TIME = {
	"5-7": 0.819,
	"7-9": 0.578,
	"9-11": 0.644,
	"11-13": 0.676,
	"13-15": 0.706,
	"15-17": 0.626,
	"17-19": 0.534,
	"19-21": 0.691,
}

# Rule 3: center-flow directional penalties
# Morning: slower toward center
MORNING_TOWARD_CENTER = {
	"5-7": 0.20,
	"7-9": 0.36,
	"9-11": 0.16,
}
# Evening: slower away from center (toward suburb)
EVENING_AWAY_CENTER = {
	"15-17": 0.18,
	"17-19": 0.38,
	"19-21": 0.16,
}

# Rule 4: adjacent spillover strength (greedy local propagation)
GAMMA_ADJ = 0.55

# Rule 5: persistence from previous slot
RHO_PERSIST = 0.68

V_MIN = 6.0
V_MAX = 70.0


def load_speed_matrix(path: str) -> Tuple[List[str], np.ndarray]:
	with open(path, "r", encoding="utf-8-sig", newline="") as f:
		reader = csv.reader(f)
		rows = list(reader)

	if len(rows) < 2:
		raise ValueError("Input CSV must contain header + at least one data row")

	labels = rows[0][1:]
	n = len(labels)
	if len(rows) - 1 < n:
		raise ValueError(f"Input CSV has {len(rows)-1} data rows but {n} are needed")

	mat = np.zeros((n, n), dtype=float)
	for i in range(n):
		row = rows[i + 1]
		if len(row) < n + 1:
			raise ValueError(f"Row {i+1} has {len(row)-1} values but {n} expected")
		vals = []
		for j in range(n):
			x = row[j + 1].strip()
			vals.append(float(x) if x else 0.0)
		mat[i, :] = vals

	return labels, mat


def find_center_node(v_base: np.ndarray) -> int:
	# Proxy center as the node with minimum average outgoing travel time.
	safe = np.where(v_base > 1e-8, v_base, 1e-8)
	t = 1.0 / safe
	n = v_base.shape[0]
	mean_t = np.zeros(n, dtype=float)
	for i in range(n):
		mask = np.ones(n, dtype=bool)
		mask[i] = False
		mean_t[i] = float(np.mean(t[i, mask]))
	return int(np.argmin(mean_t))


def directional_factor(i: int, j: int, slot: str, radial: np.ndarray, dir_scale: float) -> float:
	# toward_signal > 0 means moving toward center.
	ri = radial[i]
	rj = radial[j]
	denom = abs(ri) + abs(rj) + 1e-9
	toward_signal = (ri - rj) / denom

	penalty = 0.0
	if slot in MORNING_TOWARD_CENTER:
		penalty = (MORNING_TOWARD_CENTER[slot] * dir_scale) * max(0.0, toward_signal)
	elif slot in EVENING_AWAY_CENTER:
		away_signal = max(0.0, -toward_signal)
		penalty = (EVENING_AWAY_CENTER[slot] * dir_scale) * away_signal

	return max(0.6, 1.0 - penalty)


def build_velocity_tensor(v_base: np.ndarray, severity: float, dir_scale: float, gamma_adj: float, rho_persist: float) -> Dict[str, np.ndarray]:
	n = v_base.shape[0]
	center = find_center_node(v_base)

	# Radial proxy by travel-time to center.
	center_speed = np.where(v_base[:, center] > 1e-8, v_base[:, center], 1e-8)
	radial = 1.0 / center_speed

	out: Dict[str, np.ndarray] = {}
	prev_slot = None

	for slot in SLOTS:
		cur = np.zeros((n, n), dtype=float)
		updated: Dict[Tuple[int, int], float] = {}
		f_time_eff = 1.0 - severity * (1.0 - F_TIME[slot])
		f_time_eff = float(np.clip(f_time_eff, 0.45, 1.05))

		for i in range(n):
			for j in range(n):
				if i == j:
					cur[i, j] = 0.0
					continue

				base = float(v_base[i, j])
				if base <= 0.0:
					positive = v_base[v_base > 0.0]
					base = float(np.mean(positive)) if positive.size else 25.0

				# Rule 2
				v = base * f_time_eff
				cur[i, j] = float(np.clip(v, V_MIN, V_MAX))

		out[slot] = cur
		prev_slot = cur.copy()

	return out


def write_per_slot(labels: List[str], tensor: Dict[str, np.ndarray], out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	n = len(labels)
	for slot, mat in tensor.items():
		path = os.path.join(out_dir, f"V_{slot.replace('-', '_')}.csv")
		with open(path, "w", encoding="utf-8", newline="") as f:
			w = csv.writer(f)
			w.writerow([""] + labels)
			for i in range(n):
				w.writerow([labels[i]] + [f"{mat[i, j]:.6f}" for j in range(n)])


def write_long(labels: List[str], tensor: Dict[str, np.ndarray], out_csv: str) -> None:
	n = len(labels)
	with open(out_csv, "w", encoding="utf-8", newline="") as f:
		w = csv.writer(f)
		w.writerow([
			"origin_idx",
			"destination_idx",
			"origin_label",
			"destination_label",
			"time_slot",
			"avg_speed_kmh",
		])
		for slot in SLOTS:
			mat = tensor[slot]
			for i in range(n):
				for j in range(n):
					if i == j:
						continue
					w.writerow([i, j, labels[i], labels[j], slot, f"{mat[i, j]:.6f}"])


def parse_args():
	p = argparse.ArgumentParser(
		description="Create time-dependent truck velocity matrix V(i,j,k) from baseline speed matrix"
	)
	p.add_argument("input_csv", help="Path to baseline matrix CSV (e.g., od_speed_kmh.csv)")
	p.add_argument("--out-dir", default="velocity_slots", help="Folder for per-slot matrix outputs")
	p.add_argument("--out-long", default="od_speed_by_time_slot_greedy.csv", help="Long-format output CSV")
	p.add_argument("--severity", type=float, default=1.0,
		help="Overall congestion intensity scale for Rule 2 (1.0=default aggressive, 1.2=more severe)")
	p.add_argument("--dir-scale", type=float, default=0.0,
		help="Scale factor for directional center-flow penalties")
	p.add_argument("--adj", type=float, default=0.0,
		help="Adjacent spillover strength in [0,1]")
	p.add_argument("--persist", type=float, default=0.0,
		help="Temporal persistence in [0,1)")
	return p.parse_args()


def main():
	args = parse_args()
	severity = max(0.0, args.severity)
	dir_scale = max(0.0, args.dir_scale)
	gamma_adj = float(np.clip(args.adj, 0.0, 1.0))
	rho_persist = float(np.clip(args.persist, 0.0, 0.98))
	labels, v_base = load_speed_matrix(args.input_csv)
	tensor = build_velocity_tensor(v_base, severity, dir_scale, gamma_adj, rho_persist)
	write_per_slot(labels, tensor, args.out_dir)
	write_long(labels, tensor, args.out_long)
	print("Done")
	print("Slots:", ", ".join(SLOTS))
	print(f"Severity={severity:.2f}, dir_scale={dir_scale:.2f}, adj={gamma_adj:.2f}, persist={rho_persist:.2f}")
	print("Per-slot matrices:", args.out_dir)
	print("Long output:", args.out_long)


if __name__ == "__main__":
	main()
