"""Plot iteration records from output.txt.

Usage:
	python plot_iteration.py --input output.txt --save plot.png
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


def plot_iterations(
	iters: List[int], current: List[float], best: List[float], feasible: List[bool], bin_size: int, annotate_best: bool
):
	xs, med, p25, p75, feas_share = aggregate_current(iters, current, feasible, bin_size)
	upd_i, upd_v = extract_best_updates(iters, best)

	fig, ax1 = plt.subplots(figsize=(11, 5))

	if xs:
		ax1.fill_between(xs, p25, p75, color="#f9c7c7", alpha=0.5, label="Current cost IQR")
		line_cur, = ax1.plot(xs, med, color="black", linewidth=1.4, label=f"Current median (bin={bin_size})")

	if upd_i:
		ax1.step(upd_i, upd_v, where="post", color="blue", linewidth=1.6, label="Best updates")
		ax1.scatter(upd_i, upd_v, color="blue", s=18, zorder=3)
		if annotate_best:
			for i, v in zip(upd_i, upd_v):
				ax1.annotate(str(i), (i, v), textcoords="offset points", xytext=(3, 6), fontsize=7, color="blue")

	ax1.set_xlabel("Iteration")
	ax1.set_ylabel("Cost")
	ax1.set_title("Iteration Costs (aggregated)")
	ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

	# Feasibility share on secondary axis
	ax2 = ax1.twinx()
	if xs:
		line_feas, = ax2.plot(xs, feas_share, color="green", linewidth=1.0, linestyle="-", label="Feasible share")
		ax2.set_ylabel("Feasible share")
		ax2.set_ylim(0, 1)
	else:
		line_feas = None

	handles, labels = ax1.get_legend_handles_labels()
	if line_feas:
		handles.append(line_feas)
		labels.append("Feasible share")
	ax1.legend(handles, labels, loc="upper right")

	fig.tight_layout()


def main():
	parser = argparse.ArgumentParser(description="Plot iteration records from a CSV log.")
	parser.add_argument("--input", type=Path, default=Path("output.txt"), help="Path to iteration CSV file")
	parser.add_argument("--save", type=Path, default=None, help="Optional output image path")
	parser.add_argument("--show", action="store_true", help="Display the plot interactively")
	parser.add_argument("--bin-size", type=int, default=500, help="Bin size for aggregation in iterations")
	parser.add_argument("--annotate-best", action="store_true", help="Annotate best-update iterations")
	args = parser.parse_args()

	iters, current, best, feasible = read_iterations(args.input)
	plot_iterations(iters, current, best, feasible, args.bin_size, args.annotate_best)

	if args.save:
		plt.savefig(args.save, dpi=200)
	if args.show or not args.save:
		plt.show()


if __name__ == "__main__":
	main()
