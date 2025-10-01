import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import re
def read_instance(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Find depot line
    depot_line = [i for i, l in enumerate(lines) if l.strip().startswith('depot')][0]
    depot_coords = tuple(map(float, lines[depot_line].split()[1:]))
    # Find data start (header line with X Y Dronable ...)
    data_start = [i for i, l in enumerate(lines) if l.strip().startswith('X')][0] + 1
    customers = []
    for l in lines[data_start:]:
        l = l.strip()
        if not l: continue
        parts = l.split()
        if len(parts) < 4: continue
        x, y = float(parts[0]), float(parts[1])
        dronable = int(float(parts[2]))
        customers.append((x, y, dronable))
    return depot_coords, customers

file_path = '500.40.4.txt'

depot, customers = read_instance(file_path)


# --- New: Parse Served by Drone from output.txt ---
def parse_served_by_drone(output_file):
    served_by_drone = {}
    with open(output_file, 'r') as f:
        lines = f.readlines()
    start = False
    for line in lines:
        if line.strip().startswith("Served by Drone:"):
            start = True
            continue
        if start:
            if not line.strip() or not line.strip().startswith("Customer"):
                break
            parts = line.strip().split(":")
            if len(parts) == 2:
                cust = int(parts[0].split()[1])
                status = parts[1].strip()
                served_by_drone[cust] = (status == "Yes")
    return served_by_drone

def plot_served_by_drone(instance_file, output_file):
    depot, customers = read_instance(instance_file)
    served_by_drone = parse_served_by_drone(output_file)
    plt.figure(figsize=(10,10))
    plt.scatter(depot[0], depot[1], c='red', label='Depot', s=120, marker='*')
    # Map customer index to location
    xs_drone, ys_drone, xs_truck, ys_truck = [], [], [], []
    label_offset = 100  # adjust as needed for your scale
    for idx, (x, y, _) in enumerate(customers, 1):
        if served_by_drone.get(idx, False):
            xs_drone.append(x)
            ys_drone.append(y)
            plt.text(x, y + label_offset, str(idx), fontsize=8, color='blue', ha='center', va='bottom', fontweight='bold')
        else:
            xs_truck.append(x)
            ys_truck.append(y)
            plt.text(x, y + label_offset, str(idx), fontsize=8, color='green', ha='center', va='bottom', fontweight='bold')
    if xs_drone:
        plt.scatter(xs_drone, ys_drone, c='blue', label='Served by Drone', s=40)
    if xs_truck:
        plt.scatter(xs_truck, ys_truck, c='green', label='Served by Truck', s=40)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Customers Served by Drone (blue) and Truck (green)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('served_by_drone_plot.png')
    plt.show()
    print("Plot saved as served_by_drone_plot.png")

# --- New: Parse simple cluster text and plot clusters ---
def parse_clusters(cluster_text: str) -> List[List[int]]:
    """
    Parse clusters from text lines like:
      Cluster 1: 1 2 7 8
      Cluster 2: 5 6 9
    Returns a list of clusters, where each cluster is a list of customer ids (ints).
    The list index is zero-based (Cluster 1 -> index 0).
    """
    clusters: List[List[int]] = []
    lines = [l.strip() for l in cluster_text.strip().splitlines() if l.strip()]
    for line in lines:
        if not line.lower().startswith("cluster"):
            continue
        try:
            left, right = line.split(":", 1)
        except ValueError:
            continue
        # Extract cluster index if needed (not strictly required for ordering)
        ids = [int(tok) for tok in right.strip().split() if tok.isdigit()]
        clusters.append(ids)
    return clusters

def plot_clusters(instance_file: str, clusters: List[List[int]], savepath: str = 'clusters_plot.png') -> None:
    """
    Plot customers colored by their cluster assignment.
    - instance_file: path to the instance file used by read_instance()
    - clusters: list of clusters, each a list of customer ids (1-based)
    - savepath: output image path
    """
    depot, customers = read_instance(instance_file)
    # Map id -> (x, y)
    id_to_xy = {idx: (x, y) for idx, (x, y, _dronable) in enumerate(customers, start=1)}

    plt.figure(figsize=(10, 10))
    plt.scatter(depot[0], depot[1], c='red', label='Depot', s=120, marker='*', zorder=5)

    colors = plt.get_cmap('tab10')
    label_offset = 100
    for ci, cluster in enumerate(clusters):
        xs, ys = [], []
        for cid in cluster:
            xy = id_to_xy.get(cid)
            if not xy:
                continue
            x, y = xy
            xs.append(x)
            ys.append(y)
            plt.text(x, y + label_offset, str(cid), fontsize=8, color='black', ha='center', va='bottom')
        if xs:
            plt.scatter(xs, ys, color=colors(ci % 10), s=40, label=f'Cluster {ci+1}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Customer Clusters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    print(f"Plot saved as {savepath}")

def parse_clusters_from_output(output_file: str) -> List[List[int]]:
    """
    Look for a section in output_file like:
    'K-means Clustering Result (k=3):' followed by lines 'Cluster i: ...'.
    Returns clusters as a list of lists of ints. Returns [] if not found.
    """
    try:
        with open(output_file, 'r') as f:
            lines = [l.rstrip('\n') for l in f]
    except FileNotFoundError:
        return []

    clusters_lines: List[str] = []
    in_section = False
    for line in lines:
        s = line.strip()
        if not in_section:
            if s.lower().startswith('k-means clustering result'):
                in_section = True
            continue
        else:
            if not s:
                break
            if s.lower().startswith('cluster'):
                clusters_lines.append(s)
            else:
                break
    if not clusters_lines:
        return []
    return parse_clusters('\n'.join(clusters_lines))

def plot_clusters_from_output(instance_file: str, output_file: str, savepath: str = 'clusters_plot.png') -> None:
    clusters = parse_clusters_from_output(output_file)
    if not clusters:
        print('No clusters found in output file; nothing to plot.')
        return
    plot_clusters(instance_file, clusters, savepath=savepath)

# --- New: Parse Truck/Drone routes from solver output and plot them ---
def parse_routes_from_output(output_file: str) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Parse sections like:
      Truck Routes:
      Truck 1: 0 40 31 47 4 21 0 20 46 45 0
      ...
      Drone Routes:
      Drone 1: 0 23 19 22 18 3 0 14 0
    Returns (trucks, drones) lists of routes (each route is a list of ints).
    """
    try:
        with open(output_file, 'r') as f:
            lines = [l.rstrip('\n') for l in f]
    except FileNotFoundError:
        return [], []

    trucks: List[List[int]] = []
    drones: List[List[int]] = []
    in_trucks = False
    in_drones = False
    for line in lines:
        s = line.strip()
        if not s:
            # blank line ends current section rows, but keep scanning for later sections
            continue
        if s.lower().startswith('truck routes'):
            in_trucks, in_drones = True, False
            continue
        if s.lower().startswith('drone routes'):
            in_trucks, in_drones = False, True
            continue
        if s.lower().startswith('truck '):
            # e.g., "Truck 1: 0 40 31 ..."
            parts = s.split(':', 1)
            if len(parts) == 2:
                nums = [int(tok) for tok in parts[1].strip().split() if re.match(r'^-?\d+$', tok)]
                trucks.append(nums)
            continue
        if s.lower().startswith('drone '):
            parts = s.split(':', 1)
            if len(parts) == 2:
                nums = [int(tok) for tok in parts[1].strip().split() if re.match(r'^-?\d+$', tok)]
                drones.append(nums)
            continue
    return trucks, drones

def plot_routes(instance_file: str, trucks: List[List[int]], drones: List[List[int]], savepath: str = 'routes_plot.png') -> None:
    depot, customers = read_instance(instance_file)
    id_to_xy: Dict[int, Tuple[float, float]] = {idx: (x, y) for idx, (x, y, _d) in enumerate(customers, start=1)}
    id_to_xy[0] = depot

    plt.figure(figsize=(11, 11))
    plt.scatter(depot[0], depot[1], c='red', label='Depot', s=140, marker='*', zorder=5)

    truck_cmap = plt.get_cmap('tab10')
    drone_cmap = plt.get_cmap('Set2')
    label_offset = 100

    # Helper to draw segmented by depot zeros
    def draw_route_segments(route: List[int], color, lw=2.0, label=None, z=2):
        seg: List[Tuple[float, float]] = []
        placed_label = False
        for idx, node in enumerate(route):
            if node not in id_to_xy:
                continue
            seg.append(id_to_xy[node])
            # when we hit a depot (0) and have a segment with >1 points, draw it
            if node == 0 and len(seg) > 1:
                xs = [p[0] for p in seg]
                ys = [p[1] for p in seg]
                plt.plot(xs, ys, '-', color=color, lw=lw, alpha=0.8, zorder=z, label=(label if not placed_label else None))
                placed_label = True
                seg = [id_to_xy[0]]  # restart new segment from depot
        # draw trailing segment if not closed by depot
        if len(seg) > 1:
            xs = [p[0] for p in seg]
            ys = [p[1] for p in seg]
            plt.plot(xs, ys, '-', color=color, lw=lw, alpha=0.8, zorder=z, label=(label if not placed_label else None))

    # Plot trucks
    for i, route in enumerate(trucks[:3]):
        color = truck_cmap(i % 10)
        draw_route_segments(route, color, lw=2.4, label=f'Truck {i+1}', z=3)

    # Plot drones
    for i, route in enumerate(drones[:3]):
        color = drone_cmap(i % 8)
        draw_route_segments(route, color, lw=1.8, label=f'Drone {i+1}', z=2)

    # Node labels and scatter points
    xs_all, ys_all = [], []
    for cid, (x, y, _dronable) in enumerate(customers, 1):
        xs_all.append(x); ys_all.append(y)
        plt.text(x, y + label_offset, str(cid), fontsize=8, color='black', ha='center', va='bottom')
    if xs_all:
        plt.scatter(xs_all, ys_all, c='gray', s=28, alpha=0.6, zorder=1)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Truck and Drone Routes')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()
    print(f"Plot saved as {savepath}")

# --- Entry point ---
if __name__ == "__main__":
    # 1) Try plotting routes if present
    trucks, drones = parse_routes_from_output('output.txt')
    if trucks or drones:
        plot_routes(file_path, trucks, drones, savepath='routes_plot.png')
    else:
        # 2) Else try clusters
        clusters = parse_clusters_from_output('output.txt')
        if clusters:
            plot_clusters(file_path, clusters, savepath='clusters_plot.png')
        else:
            # 3) Else fallback to served-by-drone scatter
            plot_served_by_drone(file_path, 'output.txt')