#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict, Counter
from statistics import mean, median, pstdev

# Use a non-interactive backend for headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FORMULA_COLS_CANON = [
    "time_norm + urgency ^ 2",
    "only time_norm",
    "urg ^ 2 + cap ^ 2 + time_norm",
    "all 3 but w/o WAD",
    "all 3 w WAD but linear",
    "all 3, linear, WAD, no cluster penalty",
]


def parse_args():
    p = argparse.ArgumentParser(description="Analyze summary_score.csv across scoring formulas")
    p.add_argument("--file", default="summary_score.csv", help="Path to summary_score.csv")
    p.add_argument("--within", type=float, nargs="*", default=[1.0, 5.0],
                   help="Percent thresholds to count 'within % of best' (e.g., 1 5)")
    p.add_argument("--group-by", choices=["none", "n", "h", "d"], default="none",
                   help="Group statistics by this column as well")
    p.add_argument("--plot-dir", default="plots", help="Directory to save plots")
    p.add_argument("--no-plot", action="store_true", help="Disable plotting and only print text summary")
    return p.parse_args()


def try_float(x):
    try:
        return float(x)
    except Exception:
        return math.inf


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def detect_formula_columns(header):
    # Keep only known formula names present in header, preserving order in FORMULA_COLS_CANON
    present = [c for c in FORMULA_COLS_CANON if c in header]
    # Also include any extra columns beyond the standard five if needed
    for c in header:
        if c not in present and c not in ("instance", "n", "h", "d"):
            present.append(c)
    return present


def per_instance_best(row, formula_cols):
    vals = {col: try_float(row.get(col, math.inf)) for col in formula_cols}
    best_col = min(vals, key=lambda k: vals[k])
    best_val = vals[best_col]
    return best_col, best_val, vals


def summarize(rows, formula_cols, within_pcts, group_key=None):
    # Aggregate overall
    agg = {
        col: {
            "values": [],
            "wins": 0,
            "within": {p: 0 for p in within_pcts},
        }
        for col in formula_cols
    }

    # Optional grouping
    groups = defaultdict(list)
    for row in rows:
        g = row.get(group_key) if group_key else None
        groups[g].append(row)

    def summarize_block(rows_block):
        local = {col: {"values": [], "wins": 0, "within": {p: 0 for p in within_pcts}} for col in formula_cols}
        for row in rows_block:
            best_col, best_val, vals = per_instance_best(row, formula_cols)
            for col in formula_cols:
                v = vals[col]
                local[col]["values"].append(v)
            local[best_col]["wins"] += 1
            for col in formula_cols:
                v = vals[col]
                for p in within_pcts:
                    if v <= best_val * (1 + p / 100.0):
                        local[col]["within"][p] += 1
        # finalize stats
        out = {}
        for col in formula_cols:
            vals = [x for x in local[col]["values"] if math.isfinite(x)]
            out[col] = {
                "count": len(vals),
                "wins": local[col]["wins"],
                "avg": mean(vals) if vals else math.inf,
                "median": median(vals) if vals else math.inf,
                "std": pstdev(vals) if len(vals) > 1 else 0.0,
                "within": local[col]["within"].copy(),
            }
        return out

    overall = summarize_block(rows)

    grouped = {}
    if group_key:
        for gval, gro in groups.items():
            grouped[gval] = summarize_block(gro)

    return overall, grouped


def print_summary(overall, grouped, formula_cols, within_pcts, group_key):
    def fmt(x):
        if x is math.inf: return "inf"
        return f"{x:.3f}"

    print("Formulas compared:")
    for i, col in enumerate(formula_cols, 1):
        print(f"  {i}. {col}")
    print()

    print("Overall performance:")
    header = ["formula", "wins", "avg", "median", "std"] + [f"within_{p}%" for p in within_pcts]
    print(", ".join(header))
    # Rank by avg ascending
    ranked = sorted(overall.items(), key=lambda kv: kv[1]["avg"])
    for col, stats in ranked:
        row = [col, str(stats["wins"]), fmt(stats["avg"]), fmt(stats["median"]), fmt(stats["std"])]
        row += [str(stats["within"][p]) for p in within_pcts]
        print(", ".join(row))

    if group_key and grouped:
        print()
        print(f"By {group_key}:")
        for gval in sorted(grouped.keys(), key=lambda x: (float(x) if isinstance(x, str) and x.replace('.','',1).isdigit() else str(x))):
            print(f"\nGroup {group_key}={gval}:")
            gstats = grouped[gval]
            ranked = sorted(gstats.items(), key=lambda kv: kv[1]["avg"])
            print(", ".join(header))
            for col, stats in ranked:
                row = [col, str(stats["wins"]), fmt(stats["avg"]), fmt(stats["median"]), fmt(stats["std"])]
                row += [str(stats["within"][p]) for p in within_pcts]
                print(", ".join(row))


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_overall(overall, formula_cols, within_pcts, out_dir):
    _ensure_dir(out_dir)
    # Prepare data filtered for finite values
    def finite(val):
        return math.isfinite(val)

    # Average with std plot (lower is better)
    data = [(col, stats["avg"], stats["std"]) for col, stats in overall.items() if finite(stats["avg"]) ]
    data.sort(key=lambda x: x[1])
    if data:
        labels = [d[0] for d in data]
        avgs = [d[1] for d in data]
        errs = [d[2] for d in data]
        plt.figure(figsize=(max(8, 0.4*len(labels)), 5))
        bars = plt.bar(range(len(labels)), avgs, yerr=errs, capsize=4, color="#4e79a7")
        plt.title("Average score by formula (lower is better)")
        plt.ylabel("Average score")
        plt.xticks(range(len(labels)), labels, rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overall_avg.png"))
        plt.close()

    # Wins plot
    win_data = [(col, stats["wins"]) for col, stats in overall.items()]
    win_data.sort(key=lambda x: x[1], reverse=True)
    if win_data:
        labels = [d[0] for d in win_data]
        wins = [d[1] for d in win_data]
        plt.figure(figsize=(max(8, 0.4*len(labels)), 5))
        plt.bar(range(len(labels)), wins, color="#59a14f")
        plt.title("Wins (count of instances where formula is best)")
        plt.ylabel("Wins")
        plt.xticks(range(len(labels)), labels, rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "overall_wins.png"))
        plt.close()

    # Within % plots
    for p in within_pcts:
        within_data = [(col, stats["within"].get(p, 0)) for col, stats in overall.items()]
        within_data.sort(key=lambda x: x[1], reverse=True)
        labels = [d[0] for d in within_data]
        counts = [d[1] for d in within_data]
        plt.figure(figsize=(max(8, 0.4*len(labels)), 5))
        plt.bar(range(len(labels)), counts, color="#f28e2c")
        plt.title(f"Within {p}% of best (count)")
        plt.ylabel("Count")
        plt.xticks(range(len(labels)), labels, rotation=35, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"overall_within_{int(p)}.png"))
        plt.close()


def plot_grouped(grouped, formula_cols, within_pcts, out_dir, group_key):
    if not grouped:
        return
    base = os.path.join(out_dir, f"by_{group_key}")
    _ensure_dir(base)
    for gval, stats in grouped.items():
        gdir = os.path.join(base, str(gval))
        plot_overall(stats, formula_cols, within_pcts, gdir)


def main():
    args = parse_args()
    rows = read_csv(args.file)
    if not rows:
        print("No rows in CSV.")
        return
    header = rows[0].keys()
    # Detect formulas present in the file
    formula_cols = detect_formula_columns(header)
    # Normalize group key
    group_key = None if args.group_by == "none" else args.group_by
    overall, grouped = summarize(rows, formula_cols, args.within, group_key)
    print_summary(overall, grouped, formula_cols, args.within, group_key)
    if not args.no_plot:
        plot_overall(overall, formula_cols, args.within, args.plot_dir)
        if group_key:
            plot_grouped(grouped, formula_cols, args.within, args.plot_dir, args.group_by)


if __name__ == "__main__":
    main()
