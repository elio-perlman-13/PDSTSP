#!/usr/bin/env python3
"""
parse_results.py — Tổng hợp kết quả so sánh multi-trip vs 1-trip thành CSV

Usage:
    python3 scripts/parse_results.py <results_dir> <big_instance_dir>

Mỗi file trong results_dir có dạng:
    INSTANCE_FILE: big_instance/200.10.1.txt
    INSTANCE_BASE: 200.10.1
    SOLVER: multi          (hoặc 1trip)
    RUN: 1
    ---SOLVER OUTPUT---
    Initial solution cost: ...
    Improved solution cost: ...
    ...
    (nội dung output_solution_best.txt)

Output: CSV đến stdout
"""

import sys, os, re, csv, glob

# ── Đọc file instance để lấy num_of_truck, num_of_drone ──────────────────────
def read_instance_header(instance_dir, base):
    path = os.path.join(instance_dir, base + '.txt')
    trucks = drones = 0
    try:
        with open(path) as f:
            for line in f:
                tok = line.split()
                if not tok:
                    continue
                if tok[0] == 'trucks_count' and len(tok) >= 2:
                    trucks = int(tok[1])
                elif tok[0] == 'drones_count' and len(tok) >= 2:
                    drones = int(tok[1])
    except Exception:
        pass
    return trucks, drones

# ── Parse 1 file kết quả ──────────────────────────────────────────────────────
def parse_result_file(path):
    meta = {'INSTANCE_BASE': '', 'SOLVER': '', 'LW': '', 'RUN': ''}
    solver_lines = []
    in_solver = False

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '---SOLVER OUTPUT---':
                in_solver = True
                continue
            if not in_solver:
                # parse metadata header
                m = re.match(r'^(\w+):\s*(.+)$', line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
            else:
                solver_lines.append(line)

    content = '\n'.join(solver_lines)

    def find(pattern, default=''):
        m = re.search(pattern, content)
        return m.group(1).strip() if m else default

    final_cost   = find(r'Improved solution cost:\s*([\d.]+)')
    initial_cost = find(r'Initial solution cost:\s*([\d.]+)')
    mean_cost    = find(r'Mean solution cost:\s*([\d.]+)')
    worst_cost   = find(r'Worst solution cost:\s*([\d.]+)')
    elapsed      = find(r'Mean elapsed time:\s*([\d.]+)')

    feas_match   = re.search(r'Final solution feasibility:\s*(\w+)', content)
    feasibility  = 'yes' if (feas_match and feas_match.group(1) == 'FEASIBLE') else 'no'

    # Truck routes: "Truck N: 0 3 5 0 |Truck Time: 4358.90|..."
    truck_routes = []
    truck_times  = []
    for m in re.finditer(r'Truck \d+:\s*([\d ]+)\|Truck Time:\s*([\d.]+)', content):
        route = [int(x) for x in m.group(1).strip().split()]
        truck_routes.append(route)
        truck_times.append(float(m.group(2)))

    # Drone routes: "Drone N: 0 88 70 0 15 0 |Drone Time: 4389.31|..."
    drone_routes = []
    drone_times  = []
    for m in re.finditer(r'Drone \d+:\s*([\d ]+)\|Drone Time:\s*([\d.]+)', content):
        route = [int(x) for x in m.group(1).strip().split()]
        drone_routes.append(route)
        drone_times.append(float(m.group(2)))

    return meta, {
        'initial_cost': initial_cost,
        'final_cost':   final_cost,
        'mean_cost':    mean_cost,
        'worst_cost':   worst_cost,
        'elapsed':      elapsed,
        'feasibility':  feasibility,
        'truck_routes': truck_routes,
        'truck_times':  truck_times,
        'drone_routes': drone_routes,
        'drone_times':  drone_times,
    }

# ── Tổng hợp ─────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <results_dir> <big_instance_dir>', file=sys.stderr)
        sys.exit(1)

    results_dir   = sys.argv[1]
    instance_dir  = sys.argv[2]

    # Cache instance headers
    instance_cache = {}

    FIELDNAMES = [
        'instance name', 'run', 'lw',
        'num of truck', 'num of drone',
        'final cost (improved cost)', 'mean cost', 'worst cost',
        'truck routes(2 dim array)', 'drone routes(2 dim array)',
        'highest truck times', 'highest drone time',
        'elapsed time', 'feasibility (yes/no)',
        'optimal_trucks', 'optimal_drones',
        'optimal_in_minutes', 'optimal_best_result', 'optimal_elapse time',
        'model (truck single trip/truck multiple trip)',
    ]

    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES, lineterminator='\n')
    writer.writeheader()

    files = sorted(glob.glob(os.path.join(results_dir, '*.txt')))
    if not files:
        print('No result files found in ' + results_dir, file=sys.stderr)
        sys.exit(1)

    for fpath in files:
        try:
            meta, sol = parse_result_file(fpath)
        except Exception as e:
            print(f'Warning: skip {fpath}: {e}', file=sys.stderr)
            continue

        base   = meta.get('INSTANCE_BASE', '')
        solver = meta.get('SOLVER', '')
        lw     = meta.get('LW', '')
        run    = meta.get('RUN', '')

        if not base or not solver:
            print(f'Warning: missing metadata in {fpath}', file=sys.stderr)
            continue

        model = 'truck_multiple_trip' if solver == 'multi' else 'truck_single_trip'

        # Instance header (cached)
        if base not in instance_cache:
            instance_cache[base] = read_instance_header(instance_dir, base)
        num_trucks, num_drones = instance_cache[base]

        truck_routes = sol['truck_routes']
        drone_routes = sol['drone_routes']
        truck_times  = sol['truck_times']
        drone_times  = sol['drone_times']

        # Số vehicle thực sự phục vụ khách (có ít nhất 1 node != 0)
        optimal_trucks = sum(1 for r in truck_routes if any(n != 0 for n in r))
        optimal_drones = sum(1 for r in drone_routes if any(n != 0 for n in r))

        highest_truck = max(truck_times) if truck_times else ''
        highest_drone = max(drone_times) if drone_times else ''

        # Makespan theo phút
        try:
            opt_min = f'{float(sol["final_cost"]) / 60.0:.4f}'
        except (ValueError, ZeroDivisionError):
            opt_min = ''

        # optimal_best_result: chỉ có giá trị nếu feasible
        opt_best = sol['final_cost'] if sol['feasibility'] == 'yes' else ''

        # Routes dưới dạng JSON-like string (dễ đọc trong CSV)
        truck_routes_str = str(truck_routes)
        drone_routes_str = str(drone_routes)

        writer.writerow({
            'instance name':       base,
            'run':                 run,
            'lw':                  lw,
            'num of truck':        num_trucks,
            'num of drone':        num_drones,
            'final cost (improved cost)': sol['final_cost'],
            'mean cost':           sol['mean_cost'],
            'worst cost':          sol['worst_cost'],
            'truck routes(2 dim array)': truck_routes_str,
            'drone routes(2 dim array)': drone_routes_str,
            'highest truck times': f'{highest_truck:.6f}' if isinstance(highest_truck, float) else '',
            'highest drone time':  f'{highest_drone:.6f}' if isinstance(highest_drone, float) else '',
            'elapsed time':        sol['elapsed'],
            'feasibility (yes/no)': sol['feasibility'],
            'optimal_trucks':      optimal_trucks,
            'optimal_drones':      optimal_drones,
            'optimal_in_minutes':  opt_min,
            'optimal_best_result': opt_best,
            'optimal_elapse time': sol['elapsed'],
            'model (truck single trip/truck multiple trip)': model,
        })

if __name__ == '__main__':
    main()
