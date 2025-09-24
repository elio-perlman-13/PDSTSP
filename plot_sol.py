import matplotlib.pyplot as plt
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

depot, customers = read_instance('50.10.2.txt')


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

# --- Call the new plot function ---
if __name__ == "__main__":
    plot_served_by_drone('50.10.2.txt', 'output.txt')