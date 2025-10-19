#!/usr/bin/env python3
import csv
import os
import re
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(ROOT, 'outputs')
INST_DIR = os.path.join(ROOT, 'instance')

# Patterns to parse best.txt
RE_INIT = re.compile(r'^Initial solution cost:\s*([0-9.+-Ee]+)')
RE_IMPR = re.compile(r'^Improved solution cost:\s*([0-9.+-Ee]+)')
RE_TIME = re.compile(r'^Elapsed time:\s*([0-9.+-Ee]+)\s*seconds')

# Patterns to parse instance
RE_TRUCKS = re.compile(r'^trucks_count\s+(\d+)')
RE_DRONES = re.compile(r'^drones_count\s+(\d+)')


def parse_instance_meta(path: str):
    trucks = None
    drones = None
    try:
        with open(path, 'r') as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                if trucks is None:
                    m = RE_TRUCKS.match(line.strip())
                    if m:
                        trucks = int(m.group(1))
                        continue
                if drones is None:
                    m = RE_DRONES.match(line.strip())
                    if m:
                        drones = int(m.group(1))
                        continue
                if trucks is not None and drones is not None:
                    break
    except Exception:
        pass
    return trucks, drones


def parse_best_file(path: str):
    init = impr = t = None
    try:
        with open(path, 'r') as f:
            for _ in range(10):
                line = f.readline()
                if not line:
                    break
                if init is None:
                    m = RE_INIT.match(line.strip())
                    if m:
                        init = float(m.group(1))
                        continue
                if impr is None:
                    m = RE_IMPR.match(line.strip())
                    if m:
                        impr = float(m.group(1))
                        continue
                if t is None:
                    m = RE_TIME.match(line.strip())
                    if m:
                        t = float(m.group(1))
                        continue
                if init is not None and impr is not None and t is not None:
                    break
    except Exception:
        pass
    return init, impr, t


def main():
    # Find all *_best.txt across outputs (recursive)
    best_files = sorted(glob(os.path.join(OUTPUTS_DIR, '**', '*_best.txt'), recursive=True))
    if not best_files:
        print('[summarize_best] No _best.txt files found under outputs/.')
        return 1

    out_csv = os.path.join(OUTPUTS_DIR, 'summary_best.csv')
    rows = []

    for bf in best_files:
        base = os.path.basename(bf)
        inst_name = base.replace('_best.txt', '') + '.txt'
        inst_path = os.path.join(INST_DIR, inst_name)
        trucks, drones = parse_instance_meta(inst_path)
        init, impr, elapsed = parse_best_file(bf)
        rows.append({
            'instance': inst_name,
            'trucks': trucks if trucks is not None else '',
            'drones': drones if drones is not None else '',
            'initial_cost': init if init is not None else '',
            'improved_cost': impr if impr is not None else '',
            'elapsed_time': elapsed if elapsed is not None else '',
        })

    # Write CSV
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['instance', 'trucks', 'drones', 'initial_cost', 'improved_cost', 'elapsed_time'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'[summarize_best] Wrote {len(rows)} rows to {out_csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
