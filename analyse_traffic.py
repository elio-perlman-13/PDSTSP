#!/usr/bin/env python3
"""
Phân tích kết quả so sánh light vs heavy traffic – vai trò drone.
Usage: python3 analyse_traffic.py
"""
import os, re, glob, csv
from collections import defaultdict

# ── Parse 1 file kết quả ─────────────────────────────────────────────────────
def parse_file(path):
    meta = {}
    solver_lines = []
    in_solver = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.rstrip('\n')
            if line == '---SOLVER OUTPUT---':
                in_solver = True
                continue
            if not in_solver:
                m = re.match(r'^(\w+):\s*(.+)$', line)
                if m:
                    meta[m.group(1)] = m.group(2).strip()
            else:
                solver_lines.append(line)

    content = '\n'.join(solver_lines)

    def find(pattern):
        m = re.search(pattern, content)
        return m.group(1).strip() if m else ''

    feas_m = re.search(r'Final solution feasibility:\s*(\w+)', content)
    feas   = 'yes' if (feas_m and feas_m.group(1) == 'FEASIBLE') else 'no'

    final_cost   = find(r'Improved solution cost:\s*([\d.]+)')
    initial_cost = find(r'Initial solution cost:\s*([\d.]+)')
    elapsed      = find(r'Mean elapsed time:\s*([\d.]+)')

    # Truck times: dùng pattern |Truck Time: ...|
    truck_times = [float(m.group(1))
                   for m in re.finditer(r'\|Truck Time:\s*([\d.]+)\|', content)]

    # Drone routes & times
    drone_routes, drone_times = [], []
    for m in re.finditer(r'Drone \d+:\s*([\d ]+)\|Drone Time:\s*([\d.]+)\|', content):
        route = [int(x) for x in m.group(1).strip().split()]
        drone_routes.append(route)
        drone_times.append(float(m.group(2)))

    # Số khách drone phục vụ (node != 0)
    drone_customers = sum(len([n for n in r if n != 0]) for r in drone_routes)

    # Makespan = max(max_truck, max_drone)
    max_truck = max(truck_times) if truck_times else 0.0
    max_drone = max(drone_times) if drone_times else 0.0

    return {
        'base':              meta.get('INSTANCE_BASE', ''),
        'traffic':           meta.get('TRAFFIC', meta.get('SOLVER', '')),
        'run':               meta.get('RUN', ''),
        'feasibility':       feas,
        'final_cost':        float(final_cost)   if final_cost   else None,
        'initial_cost':      float(initial_cost) if initial_cost else None,
        'elapsed':           float(elapsed)      if elapsed      else None,
        'truck_times':       truck_times,
        'drone_times':       drone_times,
        'max_truck_time':    max_truck,
        'max_drone_time':    max_drone,
        'drone_customers':   drone_customers,
        'num_trucks_active': len([t for t in truck_times if t > 0]),
        'num_drones_active': len([d for d in drone_times  if d > 0]),
    }

# ── Đọc toàn bộ artifact ─────────────────────────────────────────────────────
records = []
skipped = 0
for fpath in sorted(glob.glob('artifacts_run/result-*/*.txt')):
    try:
        r = parse_file(fpath)
        if r['base'] and r['traffic']:
            records.append(r)
        else:
            skipped += 1
    except Exception as e:
        print(f'  Skip {fpath}: {e}')
        skipped += 1

print(f'Parsed: {len(records)} records, skipped: {skipped}')

# ── Ghi CSV chi tiết ─────────────────────────────────────────────────────────
DETAIL_FIELDS = [
    'instance', 'traffic', 'run', 'feasibility',
    'initial_cost', 'final_cost', 'elapsed_s',
    'max_truck_time', 'max_drone_time',
    'num_trucks_active', 'num_drones_active', 'drone_customers',
]
with open('results_detail.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
    w.writeheader()
    for r in records:
        w.writerow({
            'instance':           r['base'],
            'traffic':            r['traffic'],
            'run':                r['run'],
            'feasibility':        r['feasibility'],
            'initial_cost':       f"{r['initial_cost']:.2f}" if r['initial_cost'] else '',
            'final_cost':         f"{r['final_cost']:.2f}"   if r['final_cost']   else '',
            'elapsed_s':          f"{r['elapsed']:.1f}"      if r['elapsed']      else '',
            'max_truck_time':     f"{r['max_truck_time']:.2f}",
            'max_drone_time':     f"{r['max_drone_time']:.2f}",
            'num_trucks_active':  r['num_trucks_active'],
            'num_drones_active':  r['num_drones_active'],
            'drone_customers':    r['drone_customers'],
        })
print('Saved: results_detail.csv')

# ── Gộp theo (instance, traffic) ─────────────────────────────────────────────
data = defaultdict(lambda: defaultdict(list))
for r in records:
    data[r['base']][r['traffic']].append(r)

# ── Helpers ───────────────────────────────────────────────────────────────────
def avg_ratio(recs):
    """Trung bình tỉ lệ max_drone_time / max_truck_time qua các run feasible."""
    ratios = []
    for r in recs:
        if r['feasibility'] == 'yes' and r['max_truck_time'] > 0 and r['max_drone_time'] > 0:
            ratios.append(r['max_drone_time'] / r['max_truck_time'])
    return sum(ratios) / len(ratios) if ratios else None

# ── In bảng tổng hợp ─────────────────────────────────────────────────────────
HDR = (
    f"{'Instance':<13} "
    f"{'L feas':>7} {'H feas':>7}  "
    f"{'L best':>10} {'H best':>10}  "
    f"{'ΔCost%':>8}  "
    f"{'L dc avg':>9} {'H dc avg':>9}  "
    f"{'Δdrone':>7}  "
    f"{'L d/t':>7} {'H d/t':>7}"
)
SEP = '─' * len(HDR)

summary_rows = []
print()
print(SEP)
print('  DRONE ROLE: Light (σ∈{0.9,0.7,0.8}) vs Heavy (σ∈{0.6,0.4,0.5})')
print('  Truck multi-trip | Dh=200 kg | Dd=2.27 kg | 5 runs/instance')
print('  dc avg = avg drone customers served across feasible runs')
print('  d/t    = avg (max_drone_time / max_truck_time) across feasible runs')
print(SEP)
print(HDR)
print(SEP)

for base in sorted(data.keys()):
    L = data[base].get('light', [])
    H = data[base].get('heavy', [])

    l_feas = sum(1 for r in L if r['feasibility'] == 'yes')
    h_feas = sum(1 for r in H if r['feasibility'] == 'yes')

    l_costs = [r['final_cost'] for r in L if r['final_cost'] is not None]
    h_costs = [r['final_cost'] for r in H if r['final_cost'] is not None]
    l_best  = min(l_costs) if l_costs else None
    h_best  = min(h_costs) if h_costs else None

    # Drone customers (avg qua runs feasible)
    l_dc = [r['drone_customers'] for r in L if r['feasibility'] == 'yes']
    h_dc = [r['drone_customers'] for r in H if r['feasibility'] == 'yes']
    l_dc_avg = sum(l_dc) / len(l_dc) if l_dc else None
    h_dc_avg = sum(h_dc) / len(h_dc) if h_dc else None

    l_ratio = avg_ratio(L)
    h_ratio = avg_ratio(H)

    # Format strings
    l_f_s    = f"{l_feas}/{len(L)}"
    h_f_s    = f"{h_feas}/{len(H)}"
    l_b_s    = f"{l_best:.1f}" if l_best is not None else 'INFEAS'
    h_b_s    = f"{h_best:.1f}" if h_best is not None else 'INFEAS'
    delta_s  = (f"{(h_best - l_best) / l_best * 100:+.2f}%"
                if l_best is not None and h_best is not None else 'N/A')
    l_dc_s   = f"{l_dc_avg:.1f}" if l_dc_avg is not None else 'N/A'
    h_dc_s   = f"{h_dc_avg:.1f}" if h_dc_avg is not None else 'N/A'
    delta_dc = (f"{h_dc_avg - l_dc_avg:+.1f}"
                if l_dc_avg is not None and h_dc_avg is not None else 'N/A')
    l_r_s    = f"{l_ratio:.3f}" if l_ratio is not None else 'N/A'
    h_r_s    = f"{h_ratio:.3f}" if h_ratio is not None else 'N/A'

    print(
        f"{base:<13} "
        f"{l_f_s:>7} {h_f_s:>7}  "
        f"{l_b_s:>10} {h_b_s:>10}  "
        f"{delta_s:>8}  "
        f"{l_dc_s:>9} {h_dc_s:>9}  "
        f"{delta_dc:>7}  "
        f"{l_r_s:>7} {h_r_s:>7}"
    )

    summary_rows.append({
        'instance':                  base,
        'light_feasible':            l_feas,
        'light_total':               len(L),
        'heavy_feasible':            h_feas,
        'heavy_total':               len(H),
        'light_best_cost':           l_b_s,
        'heavy_best_cost':           h_b_s,
        'cost_delta_pct':            delta_s,
        'light_avg_drone_customers': l_dc_s,
        'heavy_avg_drone_customers': h_dc_s,
        'delta_drone_customers':     delta_dc,
        'light_drone_truck_ratio':   l_r_s,
        'heavy_drone_truck_ratio':   h_r_s,
    })

print(SEP)

# Tổng kết chung
lf = sum(r['light_feasible'] for r in summary_rows)
lt = sum(r['light_total']    for r in summary_rows)
hf = sum(r['heavy_feasible'] for r in summary_rows)
ht = sum(r['heavy_total']    for r in summary_rows)
print(f"\n  Tổng feasibility:  light={lf}/{lt} ({100*lf/lt:.1f}%)  |  heavy={hf}/{ht} ({100*hf/ht:.1f}%)")

better_drone = sum(
    1 for r in summary_rows
    if r['delta_drone_customers'] not in ('N/A',)
    and float(r['delta_drone_customers']) > 0
)
higher_cost = sum(
    1 for r in summary_rows
    if r['cost_delta_pct'] not in ('N/A',)
    and float(r['cost_delta_pct'].replace('%', '')) > 0
)
print(f"  Instances heavy → drone phục vụ NHIỀU khách HƠN: {better_drone}/{len(summary_rows)}")
print(f"  Instances heavy → cost CAO hơn:                  {higher_cost}/{len(summary_rows)}")
print(SEP + '\n')

# ── Ghi CSV tổng hợp ─────────────────────────────────────────────────────────
SUMMARY_FIELDS = [
    'instance',
    'light_feasible', 'light_total',
    'heavy_feasible', 'heavy_total',
    'light_best_cost', 'heavy_best_cost', 'cost_delta_pct',
    'light_avg_drone_customers', 'heavy_avg_drone_customers', 'delta_drone_customers',
    'light_drone_truck_ratio', 'heavy_drone_truck_ratio',
]
with open('comparison_summary.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
    w.writeheader()
    w.writerows(summary_rows)
print('Saved: comparison_summary.csv')
