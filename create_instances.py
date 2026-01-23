import os
import re

def parse_best_solutions(file_path):
    """
    Parses the best found solutions file to extract the number of trucks and drones
    used for each instance.
    """
    instance_vehicle_counts = {}
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None

    # Split the content by the '---' separator to process each instance block
    instance_blocks = content.strip().split('---')

    for block in instance_blocks:
        if not block.strip():
            continue

        lines = block.strip().split('\n')
        instance_name = lines[0].strip()

        truck_indices = set()
        drone_indices = set()

        # Iterate through lines to find T# and D# routes
        for line in lines[1:]: # Skip the name and cost lines
            if line.strip().startswith('T'):
                match = re.match(r'T(\d+)', line.strip())
                if match:
                    truck_indices.add(int(match.group(1)))
            elif line.strip().startswith('D'):
                match = re.match(r'D(\d+)', line.strip())
                if match:
                    drone_indices.add(int(match.group(1)))
        
        # The number of vehicles is the highest index + 1, or 0 if none were used.
        num_trucks = max(truck_indices) + 1 if truck_indices else 0
        num_drones = max(drone_indices) + 1 if drone_indices else 0
        
        instance_vehicle_counts[instance_name] = {
            'trucks': num_trucks,
            'drones': num_drones
        }
        
    return instance_vehicle_counts

def modify_instance_files(vehicle_counts, source_dir, dest_dir):
    """
    Reads original instance files, modifies truck/drone counts, and saves them
    to a new directory.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    for instance_name, counts in vehicle_counts.items():
        source_file_path = os.path.join(source_dir, instance_name)
        dest_file_path = os.path.join(dest_dir, instance_name)

        try:
            with open(source_file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Warning: Original instance file not found, skipping: {source_file_path}")
            continue

        modified_lines = []
        for line in lines:
            if line.strip().startswith('NUM TRUCKS'):
                modified_lines.append(f"NUM TRUCKS,{counts['trucks']}\n")
            elif line.strip().startswith('NUM DRONES'):
                modified_lines.append(f"NUM DRONES,{counts['drones']}\n")
            else:
                modified_lines.append(line)
        
        with open(dest_file_path, 'w') as f:
            f.writelines(modified_lines)
        
        print(f"Generated modified instance: {dest_file_path}")


if __name__ == "__main__":
    # Define file paths and directories
    workspace_root = '/workspaces/PDSTSP'
    solutions_file = os.path.join(workspace_root, 'bestfoundsolutions-mincost.txt')
    original_instance_dir = os.path.join(workspace_root, 'instances/min-cost')
    modified_instance_dir = os.path.join(workspace_root, 'instance_modified')

    # 1. Parse the solutions file to get vehicle counts
    vehicle_counts = parse_best_solutions(solutions_file)

    # 2. Modify and create new instance files
    if vehicle_counts:
        modify_instance_files(vehicle_counts, original_instance_dir, modified_instance_dir)
        print("\nModification process complete.")