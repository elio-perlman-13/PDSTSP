"""Plot best-solution progress from iteration CSV logs.

Usage:
	python plot_iteration.py --save plot.png
"""

import argparse
import csv
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_iterations(path: Path) -> Tuple[List[int], List[float], List[float], List[bool]]:
	iters: List[int] = []
	current: List[float] = []
	best: List[float] = []
	feasible: List[bool] = []

	with path.open() as f:
		reader = csv.DictReader(f)
		for row in reader:
			iters.append(int(row["iter"]))
			current.append(float(row["current_cost"]))
			best.append(float(row["best_cost"]))
			feasible.append(row["feasible"].strip().lower() == "true")

	return iters, current, best, feasible


def aggregate_current(
	iters: List[int], current: List[float], feasible: List[bool], bin_size: int
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
	buckets_cur = {}
	buckets_feas = {}
	for i, c, f in zip(iters, current, feasible):
		b = (i - 1) // bin_size
		buckets_cur.setdefault(b, []).append(c)
		buckets_feas.setdefault(b, []).append(f)

	xs: List[float] = []
	med: List[float] = []
	p25: List[float] = []
	p75: List[float] = []
	feas_share: List[float] = []
	for b in sorted(buckets_cur):
		vals = np.array(buckets_cur[b])
		feas_vals = buckets_feas.get(b, [])
		xs.append(b * bin_size + bin_size / 2.0)
		med.append(float(np.median(vals)))
		p25.append(float(np.percentile(vals, 25)))
		p75.append(float(np.percentile(vals, 75)))
		feas_share.append(sum(feas_vals) / len(feas_vals) if feas_vals else 0.0)

	return xs, med, p25, p75, feas_share


def extract_best_updates(iters: List[int], best: List[float]) -> Tuple[List[int], List[float]]:
	upd_i: List[int] = []
	upd_v: List[float] = []
	prev = math.inf
	for i, v in zip(iters, best):
		if math.isinf(v):
			continue
		if not upd_v or v < prev - 1e-9:
			upd_i.append(i)
			upd_v.append(v)
			prev = v
	return upd_i, upd_v


def plot_best_lines(
	series: List[Tuple[str, List[int], List[float]]], annotate_best: bool
):
	fig, ax1 = plt.subplots(figsize=(11, 5))
	colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

	for idx, (label, iters, best) in enumerate(series):
		upd_i, upd_v = extract_best_updates(iters, best)
		if not upd_i:
			continue
		color = colors[idx % len(colors)]
		ax1.step(upd_i, upd_v, where="post", color=color, linewidth=1.8, label=label)
		ax1.scatter(upd_i, upd_v, color=color, s=18, zorder=3)
		# Extend final plateau to the last iteration for each run.
		if iters and upd_i[-1] < iters[-1]:
			ax1.plot([upd_i[-1], iters[-1]], [upd_v[-1], upd_v[-1]], color=color, linewidth=1.8)
		if annotate_best:
			for i, v in zip(upd_i, upd_v):
				ax1.annotate(str(i), (i, v), textcoords="offset points", xytext=(3, 6), fontsize=7, color=color)

	ax1.set_xlabel("Iteration")
	ax1.set_ylabel("Cost")
	ax1.set_title("Best Solution Progress")
	ax1.set_ylim(bottom=2500)
	ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
	ax1.legend(loc="upper right")

	fig.tight_layout()


def main():
	parser = argparse.ArgumentParser(description="Plot iteration records from a CSV log.")
	parser.add_argument("--input", type=Path, default=Path("output.txt"), help="Path to mode 0 + 2 iteration CSV file")
	parser.add_argument("--input-mode0", type=Path, default=Path("output_mode_0.txt"), help="Path to mode 0 iteration CSV file")
	parser.add_argument("--save", type=Path, default=None, help="Optional output image path")
	parser.add_argument("--show", action="store_true", help="Display the plot interactively")
	parser.add_argument("--annotate-best", action="store_true", help="Annotate best-update iterations")
	args = parser.parse_args()

	iters_02, _, best_02, _ = read_iterations(args.input)
	iters_0, _, best_0, _ = read_iterations(args.input_mode0)

	series = [
		("mode 0 + 2", iters_02, best_02),
		("mode 0", iters_0, best_0),
	]
	plot_best_lines(series, args.annotate_best)

	if args.save:
		plt.savefig(args.save, dpi=200)
	if args.show or not args.save:
		plt.show()


if __name__ == "__main__":
	main()
