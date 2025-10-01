#include <bits/stdc++.h>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <random>

#define ll long long
#define pb push_back
#define mp make_pair
#define pii pair<int,int>
#define vi vector<int>
#define vd vector<double>
#define vvi vector<vector<int>>
#define vvd vector<vector<double>>
#define vpi vector<pair<int,int>>
#define all(v) v.begin(),v.end()
#define FOR(i,a,b) for(int i=a;i<=b;i++)
#define RFOR(i,a,b) for(int i=a-1;i>=b;i--)

using namespace std;

// Data structures and global variables
struct Point {
    double x = 0.0, y = 0.0;
    int id = -1;
    Point() = default;
    Point(double x_, double y_, int id_ = -1) : x(x_), y(y_), id(id_) {}
};

int n, h, d; //number of customers, number of trucks, number of drones
vector<Point> loc; // loc[i]: location (x, y) of customer i, if i = 0, it is depot
vd serve_truck, serve_drone; // time taken by truck and drone to serve each customer (hours)
vi served_by_drone; //whether each customer can be served by drone or not, 1 if yes, 0 if no
vd deadline; //customer deadlines
vd demand; // demand[i]: demand of customer i
double Dh = 500.0; // truck capacity (all trucks)
double vmax = 15.6464; // truck base speed (m/s)
int L = 24; //number of time segments in a day
vd time_segments_sigma = {0.0, 0.9, 0.8, 0.4, 0.6, 0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8, 0.9, 0.8, 0.4, 0.6, 0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 2.27, E = 7200000.0; //drone's weight and energy capacities (for all drones)
double v_fly_drone = 31.3, v_take_off = 15.6, v_landing = 7.8, height = 50; // maximum speed of the drone
double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    double total_cost; //total cost of the solution
};

void input(string filepath){
        // Open the file
        ifstream fin(filepath);
        if (!fin) {
            cerr << "Error: Cannot open " << filepath << endl;
            exit(1);
        }
        string line;
        n = h = d = -1;
        // Read trucks_count, drones_count, customers
        while (getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            stringstream ss(line);
            string key;
            ss >> key;
            if (key == "trucks_count") ss >> h;
            else if (key == "drones_count") ss >> d;
            else if (key == "customers") ss >> n;
            else if (key == "depot") break;
        }
        // Read depot location
        double depot_x = 0, depot_y = 0;
        stringstream ss_depot(line);
        string depot_key;
        ss_depot >> depot_key >> depot_x >> depot_y;
        // Prepare storage
        served_by_drone.resize(n+1);
        serve_truck.resize(n+1);
        serve_drone.resize(n+1);
        deadline.resize(n+1);
        demand.resize(n+1);
        loc.resize(n+1);
        distance_matrix.resize(n+1, vd(n+1));
        loc[0] = {depot_x, depot_y, 0};
        // Skip headers until data lines
        int header_skips = 0;
        while (header_skips < 2 && getline(fin, line)) {
            if (!line.empty() && line[0] != '#') ++header_skips;
        }
        // Read customer data
        int cust = 1;
        while (cust <= n && getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            stringstream ss(line);
            double x, y, dronable, demand_val, drone_service, truck_service, deadline_val;
            ss >> x >> y >> dronable >> demand_val >> drone_service >> truck_service >> deadline_val;
            loc[cust] = {x, y, cust};
            served_by_drone[cust] = (int)dronable;
            demand[cust] = demand_val;
            serve_drone[cust] = drone_service;
            serve_truck[cust] = truck_service;
            deadline[cust] = deadline_val;
            ++cust;
        }
}

// Returns pair of distance matrices: {truck (Manhattan), drone (Euclidean)}
void compute_distance_matrices(const vector<Point>& loc) {
    int n = loc.size() - 1; // assuming loc[0] is depot
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distance_matrix[i][j] = sqrt((loc[i].x - loc[j].x) * (loc[i].x - loc[j].x)
                                         + (loc[i].y - loc[j].y) * (loc[i].y - loc[j].y)); // Euclidean
        }
    }
}

// Helper: get time segment index for a given time (assume segments are of equal length in a day)
int get_time_segment(double t) {
    // Assume a day is 24 hours, segments are [0,1), [1,2), ..., [23,24)
    int seg = (int)fmod(t, 24.0);
    if (seg < 1) seg = 1;
    if (seg > L) seg = L;
    return seg;
}

pair<double, bool> compute_truck_route_time(const vi& route, double start=0) {
    double time = start; // seconds
    bool deadline_feasible = true;
    // Index by customer id (0..n), not by position in route, to avoid out-of-bounds writes
    vector<double> visit_times(n+1, 0.0); // visit_times[id]: time when node id is last visited
    vector<int> customers_since_last_depot;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist_left = distance_matrix[from][to]; // meters
        while (dist_left > 1e-8) {
            // Convert time to hours for segment lookup
            double t_hr = time / 3600.0;
            int seg = get_time_segment(t_hr);
            double v = vmax * (seg <= (int)time_segments_sigma.size() - 1 ? time_segments_sigma[seg] : 1.0); // m/s
            if (v <= 1e-8) v = vmax;
            // Time left in this segment (seconds to next integer hour)
            double t_seg_end = 3600.0 * (floor(t_hr) + 1) - time;
            if (t_seg_end < 1e-8) t_seg_end = 3600.0; // if exactly at hour, full segment
            double max_dist_this_seg = v * t_seg_end;
            if (dist_left <= max_dist_this_seg) {
                double t_needed = dist_left / v;
                time += t_needed;
                dist_left = 0;
            } else {
                time += t_seg_end;
                dist_left -= max_dist_this_seg;
            }
        }
        if (to != 0) {
            time += serve_truck[to]; // in seconds
            customers_since_last_depot.push_back(to);
        }
        visit_times[to] = time; // record departure from node 'to'
        // If we reach depot (except at start), check duration from leaving each customer to depot
        if (to == 0 && k != 1) {
            for (int cust : customers_since_last_depot) {
                // Duration from leaving customer to returning to depot
                double duration = time - visit_times[cust];
                if (deadline_feasible && (duration > deadline[cust] + 1e-8)) {
                    deadline_feasible = false;
                }
            }
            // After returning to depot, reset visit times for customers
            for (int cust : customers_since_last_depot) {
                visit_times[cust] = time;
            }
            customers_since_last_depot.clear();
        }
    }
    return {time - start, deadline_feasible};
}

pair<double, bool> compute_drone_route_energy(const vi& route) {
    double total_energy = 0, current_weight = 0;
    double energy_used = 0;
    bool feasible = true;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist = distance_matrix[from][to]; // meters
        double v = v_fly_drone; // assume constant speed for simplicity
        double time = dist / v; // seconds
        time += height / v_take_off; // take-off time
        time += height / v_landing; // landing time
        // Energy consumption model: power = beta * weight + gamma
        double power = power_beta * (current_weight) + power_gamma; // watts
        energy_used += power * time; // energy in joules
        total_energy += power * time;
        if (energy_used > E + 1e-8) feasible = false;
        if (to != 0) current_weight += demand[to]; // add payload when delivering
        else {
            current_weight = 0; // reset weight when returning to depot
            energy_used = 0; // reset energy (charged at depot)
        }
    }
    return make_pair(total_energy, feasible);
}

pair<double, bool> compute_drone_route_time(const vi& route) {
    double time = 0; // seconds
    bool deadline_feasible = true;
    // Index by customer id (0..n), not by position in route
    vector<double> visit_times(n+1, 0.0); // visit_times[id]: time when node id is last visited
    vector<int> customers_since_last_depot;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist = distance_matrix[from][to]; // meters
        double v = v_fly_drone; // assume constant speed for simplicity
        if (v <= 1e-8) v = v_fly_drone;
        double t = dist / v; // seconds
        t += height / v_take_off; // take-off time
        t += height / v_landing; // landing time
        time += t;
        if (to != 0) {
            time += serve_drone[to]; // in seconds
            customers_since_last_depot.push_back(to);
        }
        visit_times[to] = time;
        // If we reach depot (except at start), check duration from leaving each customer to depot
        if (to == 0 && k != 1) {
            for (int cust : customers_since_last_depot) {
                double duration = time - visit_times[cust];
                if (deadline_feasible && (duration > deadline[cust] + 1e-8)) {
                    deadline_feasible = false;
                }
            }
            for (int cust : customers_since_last_depot) {
                visit_times[cust] = time;
            }
            customers_since_last_depot.clear();
        }
    }
    return {time, deadline_feasible};
}



void update_served_by_drone() {
    int depot = 0;
    for (int customer = 1; customer <= n; ++customer) {
        if (served_by_drone[customer] == 0) continue;
        // Capacity: demand[customer] <= Dd
        if (demand[customer] > Dd) {
            served_by_drone[customer] = 0;
            continue;
        }
        // Energy: use compute_drone_route_energy for depot->customer->depot
        vi route = {depot, customer, depot};
        auto [total_energy, feasible_energy] = compute_drone_route_energy(route);
        if (!feasible_energy) {
            served_by_drone[customer] = 0;
            continue;
        }
        served_by_drone[customer] = 1;
    }
}

pair<double, bool> check_truck_route_feasibility(const vi& route, double start=0) {
    // Check deadlines
    auto [time, deadline_feasible] = compute_truck_route_time(route, start);
    if (!deadline_feasible) return make_pair(time, false);
    // Check capacity (reset at depot)
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand += demand[customer];
            if (customer == 0) total_demand = 0.0;
            if (total_demand > (double)Dh + 1e-9) {
                cout << "Truck route capacity exceeded: ";
                return make_pair(time, false); // exceeded capacity
            }
        }
    }
    return make_pair(time, true);
}

pair<double, bool> check_drone_route_feasibility(const vi& route) {
    // Check deadlines
    auto [time, deadline_feasible] = compute_drone_route_time(route);
    if (!deadline_feasible) return make_pair(time, false);
    // Check capacity (reset at depot)
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dd + 1e-9){
                cout << "Drone route capacity exceeded: ";
                return make_pair(time, false);
            } // exceeded capacity
        }
    }
    // Check energy
    auto [total_energy, feasible_energy] = compute_drone_route_energy(route);
    if (!feasible_energy) {
        cout << "Drone route energy infeasible: ";
        return make_pair(time, false); // exceeded energy in a segment
    }
    return make_pair(time, true);
}

// K-mean function to generate initial solution
vvi kmeans_clustering(int k, int max_iters=1000) {
    if (n <= 0) return {};
    // Bound k to [1, n]
    if (k <= 0) k = 1;
    if (k > n) k = n;

    vvi clusters(k);
    vector<Point> centroids;
    centroids.reserve(k);

    // Random engine
    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dis(1, n);

    // K-means++-like seeding: first random, next farthest from existing
    centroids.push_back(loc[dis(gen)]);
    while ((int)centroids.size() < k) {
        double max_min_dist = -1.0;
        Point next_centroid = loc[1];
        for (int i = 1; i <= n; ++i) {
            const Point& p = loc[i];
            double min_dist = 1e18;
            for (const auto& c : centroids) {
                double dx = p.x - c.x;
                double dy = p.y - c.y;
                double dist = std::sqrt(dx*dx + dy*dy);
                min_dist = std::min(min_dist, dist);
            }
            if (min_dist > max_min_dist) {
                max_min_dist = min_dist;
                next_centroid = p;
            }
        }
        centroids.push_back(next_centroid);
    }

    // Iterations
    vector<int> assignment(n+1, -1); // assignment for customers 1..n
    for (int it = 0; it < max_iters; ++it) {
        bool changed = false;
        for (auto& cl : clusters) cl.clear();

        // Assign step
        for (int i = 1; i <= n; ++i) {
            const Point& p = loc[i];
            double bestDist2 = 1e300;
            int bestC = 0;
            for (int c = 0; c < k; ++c) {
                double dx = p.x - centroids[c].x;
                double dy = p.y - centroids[c].y;
                double d2 = dx*dx + dy*dy;
                if (d2 < bestDist2) {
                    bestDist2 = d2;
                    bestC = c;
                }
            }
            if (assignment[i] != bestC) {
                assignment[i] = bestC;
                changed = true;
            }
            clusters[bestC].push_back(i);
        }

        // Update step
        for (int c = 0; c < k; ++c) {
            if (clusters[c].empty()) {
                // Reinitialize empty cluster to a random customer to avoid dead clusters
                int pick = dis(gen);
                centroids[c].x = loc[pick].x;
                centroids[c].y = loc[pick].y;
                continue;
            }
            double sumx = 0.0, sumy = 0.0;
            for (int idx : clusters[c]) {
                sumx += loc[idx].x;
                sumy += loc[idx].y;
            }
            centroids[c].x = sumx / clusters[c].size();
            centroids[c].y = sumy / clusters[c].size();
        }

        if (!changed) break; // converged
    }

    return clusters;
}

Solution generate_initial_solution(){
    Solution sol;
    sol.truck_routes.resize(h);
    sol.drone_routes.resize(h); // as many drone routes as truck routes
    // Cluster customers into up to h groups (if h==0, nothing to do)
    vector<bool> visited(n+1, false);
    int num_of_visited_customers = 0;
    vvi clusters = (h > 0) ? kmeans_clustering(h) : vvi{};
    // Optional: shuffle each cluster to randomize the intra-cluster selection order
    {
        mt19937 rng(std::random_device{}());
        for (auto& vec : clusters) {
            shuffle(vec.begin(), vec.end(), rng);
        }
    }
    vi cluster_assignment(n+1, -1);
    for (int i = 0; i < (int)clusters.size(); ++i) {
        for (int cust : clusters[i]) {
            cluster_assignment[cust] = i;
        }
    }
    vd service_times_truck(h, 0.0); // service times for each truck (and drone)
    vd service_times_drone(h, 0.0); // service times for each drone
    vd capacity_used_truck(h, 0); // capacity used by each truck
    vd capacity_used_drone(h, 0); // capacity used by each drone
    vd timebomb_truck(h, 1e18); // timebomb for each truck
    vd timebomb_drone(h, 1e18); // timebomb for each drone
    vd energy_used_drone(h, 0); // energy used by each drone
    // Simple initializer: for each index i in [0..h-1], pick first unvisited feasible customer
    // for truck, then pick first unvisited feasible customer for drone. Routes are {0, cust, 0}.
    // If none available or infeasible, assign {0}.
    for (int i = 0; i < h; ++i) {
        const vector<int>* cluster_ptr = (i < (int)clusters.size()) ? &clusters[i] : nullptr;
        bool assigned_truck = false;
        if (cluster_ptr) {
            for (int cust : *cluster_ptr) {
                if (visited[cust]) continue;
                vi r = {0, cust, 0};
                auto [t, feas] = check_truck_route_feasibility(r, 0.0);
                if (feas) {
                    r = {0, cust};
                    auto [t, feas] = compute_truck_route_time(r, 0.0);
                    sol.truck_routes[i] = r;
                    visited[cust] = true;
                    assigned_truck = true;
                    service_times_truck[i] += t;
                    num_of_visited_customers++;
                    capacity_used_truck[i] += demand[cust];
                    // Timebomb should be tied to the selected customer's deadline, not the truck index
                    timebomb_truck[i] = deadline[cust];
                    break;
                }
            }
        }

        if (!assigned_truck) {
            sol.truck_routes[i] = {0};
        }
        bool assigned_drone = false;
        if (cluster_ptr) {
            for (int cust : *cluster_ptr) {
                if (visited[cust]) continue;
                if (!served_by_drone[cust]) continue;
                vi r = {0, cust, 0};
                auto [t, feas] = check_drone_route_feasibility(r);
                if (feas) {
                    r = {0, cust};
                    auto [t, feas] = compute_drone_route_time(r);
                    service_times_drone[i] += t;
                    energy_used_drone[i] += compute_drone_route_energy(r).first;
                    // Timebomb should be tied to the selected customer's deadline, not the drone index
                    timebomb_drone[i] = deadline[cust];
                    capacity_used_drone[i] += demand[cust];
                    sol.drone_routes[i] = r;
                    visited[cust] = true;
                    assigned_drone = true;
                    num_of_visited_customers++;
                    break;
                }
            }
        }
        if (!assigned_drone) {
            sol.drone_routes[i] = {0};
        }
    }

    //loop until all customers are visited:
    int iter = 100000; // max iterations
    while (num_of_visited_customers < n && iter > 0) {
        iter--;
        // first, pick a random vehicle (truck or drone)
        if (h <= 0) break; // no vehicles available
        int best_vehicle = -1; // 0..h-1 for trucks, h..2h-1 for drones
        static thread_local mt19937 rng(random_device{}());
        uniform_int_distribution<int> vehicle_dis(0, 2 * h - 1);
        best_vehicle = vehicle_dis(rng);
        bool is_truck = (best_vehicle < h);
        // if it's a truck:
        if (is_truck) {
            int truck_idx = best_vehicle;
            bool assigned = false;
            double best_score = 1e18;
            vi current_route = sol.truck_routes[truck_idx];
            int current_node = current_route.empty() ? 0 : current_route.back();
            // Try to assign a new customer to this truck
            // Loop all customers in cluster[truck_idx] first
            // Find the best candidate customer based on a scoring function
            struct Candidate {
                int cust;
                double urgency_score;
                double capacity_ratio;
                double change_in_return_time;
                bool same_cluster;
                Candidate(int c, double u, double cr, double crt, bool sc)
                    : cust(c), urgency_score(u), capacity_ratio(cr), change_in_return_time(crt), same_cluster(sc) {}
            };
            Candidate* best_candidate = nullptr;
            vector<Candidate> candidates;
            double max_change_in_return_time = -1e18;
            double min_change_in_return_time = 1e18;
            double direct_return_time = compute_truck_route_time({current_node, 0}, service_times_truck[truck_idx]).first;  
            for (int cust = 1; cust <= n; ++cust) {
                if (visited[cust]) continue;
                if (capacity_used_truck[truck_idx] + demand[cust] > (double)Dh + 1e-9) continue; // capacity prune
                vi r = sol.truck_routes[truck_idx];
                // calculate a score for inserting cust into r
                // Note: compute_truck_route_time adds serve_truck[cust] for non-depot arrivals; subtract it to get pure travel
                double to_with_service = compute_truck_route_time({current_node, cust}, service_times_truck[truck_idx]).first;
                double travel_to_cust = max(0.0, to_with_service);
                double depart_time = service_times_truck[truck_idx] + travel_to_cust;
                double time_back_to_depot = compute_truck_route_time({cust, 0}, depart_time).first;
                double time_bomb_at_cust = min(timebomb_truck[truck_idx] - to_with_service, deadline[cust]);
                if (time_bomb_at_cust <= 0) continue; // cannot reach customer before its deadline
                double urgency_score = time_back_to_depot / time_bomb_at_cust;
                // Urgency check: must be able to serve customer and come back to depot before their deadline
                if (urgency_score > 1.0 + 1e-8) continue;
                if (urgency_score < 0) continue;
                double change_in_return_time = to_with_service + time_back_to_depot - direct_return_time;
                double capacity_ratio = (capacity_used_truck[truck_idx] + demand[cust]) / (double)Dh;
                if (capacity_ratio > 1.0 + 1e-8) continue; // capacity prune
                max_change_in_return_time = max(max_change_in_return_time, change_in_return_time);
                min_change_in_return_time = min(min_change_in_return_time, change_in_return_time);
                // check if same cluster with truck position
                bool same_cluster = (cluster_assignment[cust] == cluster_assignment[current_node]);
                candidates.emplace_back(cust, urgency_score, capacity_ratio, change_in_return_time, same_cluster);
            }
            // Select the best candidate based on a weighted scoring function
            //First calculate weights based on MAD:
            double mean_urgency = 0.0, mean_capacity = 0.0;
            for (auto& cand : candidates) {
                mean_urgency += cand.urgency_score;
                mean_capacity += cand.capacity_ratio;
            }
            mean_urgency /= candidates.size();
            mean_capacity /= candidates.size();
            double mad_urgency = 0.0, mad_capacity = 0.0;
            for (auto& cand : candidates) {
                mad_urgency += fabs(cand.urgency_score - mean_urgency);
                mad_capacity += fabs(cand.capacity_ratio - mean_capacity);
            }
            mad_urgency /= candidates.size();
            mad_capacity /= candidates.size();
            // Avoid zero MAD
            if (mad_urgency < 1e-8) mad_urgency = 1.0;
            if (mad_capacity < 1e-8) mad_capacity = 1.0;
            // Now score candidates and pick the best
            for (auto& cand : candidates) {
                double w1 = 1.0 / mad_urgency, w2 = 1.0 / mad_capacity; // weights for urgency, capacity, change in return time, same cluster
                // Normalize weights
                double w_sum = w1 + w2;
                w1 /= w_sum; w2 /= w_sum;
                // Normalize change_in_return_time to [0,1] based on min/max in candidates
                double norm_change = (max_change_in_return_time - min_change_in_return_time < 1e-8)
                                     ? 0.0
                                     : (cand.change_in_return_time - min_change_in_return_time) / (max_change_in_return_time - min_change_in_return_time);
                double score = w1 * cand.urgency_score * cand.urgency_score + w2 * cand.capacity_ratio * cand.capacity_ratio + norm_change + (cand.same_cluster ? 0.0 : 2.0);
                if (score < best_score) {
                    best_score = score;
                    best_candidate = &cand;
                }
            }
            if (best_candidate) {
                int cust = best_candidate->cust;
                vi r = sol.truck_routes[truck_idx];
                r.push_back(cust);
                double time_to_cust = compute_truck_route_time({current_node, cust}, service_times_truck[truck_idx]).first;
                service_times_truck[truck_idx] += time_to_cust;
                capacity_used_truck[truck_idx] += demand[cust];
                timebomb_truck[truck_idx] = min(timebomb_truck[truck_idx] - time_to_cust, deadline[cust]);
                sol.truck_routes[truck_idx] = r;
                visited[cust] = true;
                assigned = true;
                num_of_visited_customers++;
            }
            if (!assigned) {
                // No feasible customer found, force return to depot if not already there
                int current_node = current_route.empty() ? 0 : current_route.back();
                if (sol.truck_routes[truck_idx].empty() || current_node != 0) {
                    vi r = sol.truck_routes[truck_idx];
                    int current_node = r.empty() ? 0 : r.back();
                    double time_to_depot = compute_truck_route_time({current_node, 0}, service_times_truck[truck_idx]).first;
                    service_times_truck[truck_idx] += time_to_depot;
                    r.push_back(0);
                    sol.truck_routes[truck_idx] = r;
                    timebomb_truck[truck_idx] = 1e18; // reset timebomb after returning to depot
                    // Reset capacity after completing a tour at the depot
                    capacity_used_truck[truck_idx] = 0.0;
                }
            }
        } else {
            int drone_idx = best_vehicle - h;
            bool assigned = false;
            vi current_route = sol.drone_routes[drone_idx];
            int current_node = current_route.empty() ? 0 : current_route.back();
            double best_score = 1e18;
            // Try to assign a new customer to this drone
            struct Candidate {
                int cust;
                double urgency_score;
                double capacity_ratio;
                double energy_ratio;
                double change_in_return_time;
                bool same_cluster;
                Candidate(int c, double u, double cr, double er, double crt, bool sc)
                    : cust(c), urgency_score(u), capacity_ratio(cr), energy_ratio(er), change_in_return_time(crt), same_cluster(sc) {}
            };
            Candidate* best_candidate = nullptr;
            vector<Candidate> candidates;
            double max_change_in_return_time = -1e18;
            double min_change_in_return_time = 1e18;
            double direct_return_time = compute_drone_route_time({current_node, 0}).first;  

            for (int cust = 1; cust <= n; ++cust) {
                if (visited[cust]) continue;
                if (!served_by_drone[cust]) continue;
                if (capacity_used_drone[drone_idx] + demand[cust] > Dd + 1e-9) continue; // capacity prune
                double to_with_service = compute_drone_route_time({current_node, cust}).first;
                double time_back_to_depot = compute_drone_route_time({cust, 0}).first;
                double time_bomb_at_cust = min(timebomb_drone[drone_idx] - to_with_service, deadline[cust]);
                if (time_bomb_at_cust <= 0) continue; // cannot reach customer before its deadline
                double urgency_score = time_back_to_depot / time_bomb_at_cust;
                if (urgency_score > 1.0 + 1e-8) continue;
                if (urgency_score < -1e-12) continue;
                double change_in_return_time = to_with_service + time_back_to_depot - direct_return_time;
                double capacity_ratio = (capacity_used_drone[drone_idx] + demand[cust]) / Dd;
                if (capacity_ratio > 1.0 + 1e-8) continue; // capacity prune
                // Estimate energy for sortie current_node -> cust -> depot at current payload
                double total_energy = (to_with_service - serve_drone[cust]) * (power_beta * capacity_used_drone[drone_idx] + power_gamma)
                                      + (time_back_to_depot) * (power_beta * (capacity_used_drone[drone_idx] + demand[cust]) + power_gamma);
                double energy_ratio = (energy_used_drone[drone_idx] + total_energy) / E;
                if (energy_ratio > 1.0 + 1e-8) continue; // energy prune
                max_change_in_return_time = max(max_change_in_return_time, change_in_return_time);
                min_change_in_return_time = min(min_change_in_return_time, change_in_return_time);
                bool same_cluster = (cluster_assignment[cust] == cluster_assignment[current_node]);
                candidates.emplace_back(cust, urgency_score, capacity_ratio, energy_ratio, change_in_return_time, same_cluster);
            }
            // Select the best candidate based on a weighted scoring function
            //First calculate weights based on MAD:
            double mean_urgency = 0.0, mean_capacity = 0.0, mean_energy = 0.0;
            for (auto& cand : candidates) {
                mean_urgency += cand.urgency_score;
                mean_capacity += cand.capacity_ratio;
                mean_energy += cand.energy_ratio;
            }
            mean_urgency /= candidates.size();
            mean_capacity /= candidates.size();
            mean_energy /= candidates.size();
            double mad_urgency = 0.0, mad_capacity = 0.0, mad_energy = 0.0;
            for (auto& cand : candidates) {
                mad_urgency += fabs(cand.urgency_score - mean_urgency);
                mad_capacity += fabs(cand.capacity_ratio - mean_capacity);
                mad_energy += fabs(cand.energy_ratio - mean_energy);
            }
            mad_urgency /= candidates.size();
            mad_capacity /= candidates.size();
            mad_energy /= candidates.size();
            // Avoid zero MAD
            if (mad_urgency < 1e-8) mad_urgency = 1.0;
            if (mad_capacity < 1e-8) mad_capacity = 1.0;
            if (mad_energy < 1e-8) mad_energy = 1.0;
            // Now score candidates and pick the best   

            for (auto& cand : candidates) {
                double w1 = 1.0 / mad_urgency, w2 = 1.0 / mad_capacity, w3 = 1.0 / mad_energy; // weights for urgency, capacity, energy, change in return time, same cluster
                // Normalize weights
                double w_sum = w1 + w2 + w3;
                w1 /= w_sum; w2 /= w_sum; w3 /= w_sum;
                // Normalize change_in_return_time to [0,1] based on min/max in candidates
                double norm_change = (max_change_in_return_time - min_change_in_return_time < 1e-8)
                                     ? 0.0
                                     : (cand.change_in_return_time - min_change_in_return_time) / (max_change_in_return_time - min_change_in_return_time);
                // Score: lower is better
                double score = w1 * cand.urgency_score * cand.urgency_score + w2 * cand.capacity_ratio * cand.capacity_ratio
                               + w3 * cand.energy_ratio * cand.energy_ratio + norm_change + (cand.same_cluster ? 0.0 : (w1 + w2 + w3 + 1.0));
                if (score < best_score) {
                    best_score = score;
                    best_candidate = &cand;
                }
            }
            if (best_candidate) {
                int cust = best_candidate->cust;
                vi r = sol.drone_routes[drone_idx];
                r.push_back(cust);
                double time_to_cust = compute_drone_route_time({current_node, cust}).first;
                service_times_drone[drone_idx] += time_to_cust;
                // Reserve energy for forward leg and the eventual return to depot
                double total_energy = (time_to_cust - serve_drone[cust]) * (power_beta * capacity_used_drone[drone_idx] + power_gamma);
                energy_used_drone[drone_idx] += total_energy;
                capacity_used_drone[drone_idx] += demand[cust];
                timebomb_drone[drone_idx] = min(timebomb_drone[drone_idx] - time_to_cust, deadline[cust]);
                sol.drone_routes[drone_idx] = r;
                visited[cust] = true;
                assigned = true;
                num_of_visited_customers++;
            }
            if (!assigned) {
                int current_node = current_route.empty() ? 0 : current_route.back();
                if (sol.drone_routes[drone_idx].empty() || current_node != 0) {
                    vi r = sol.drone_routes[drone_idx];
                    int current_node = r.empty() ? 0 : r.back();
                    double time_to_depot = compute_drone_route_time({current_node, 0}).first;
                    service_times_drone[drone_idx] += time_to_depot;
                    r.push_back(0);
                    sol.drone_routes[drone_idx] = r;
                    timebomb_drone[drone_idx] = 1e18; // reset timebomb after returning to depot
                    capacity_used_drone[drone_idx] = 0.0;
                    energy_used_drone[drone_idx] = 0.0;
                }
            }
        }
    }
    // if there are still unvisited customers, assign them to any vehicle that can take them
    for (int cust = 1; cust <= n; ++cust) {
        if (visited[cust]) continue;
        bool assigned = false;
        for (int i = 0; i < h && !assigned; ++i) {
            // Try truck first
            vi r = sol.truck_routes[i];
            int current_node = r.empty() ? 0 : r.back();
            double current_time = service_times_truck[i];
            double time_to_cust = compute_truck_route_time({current_node, cust}, current_time).first;
            double time_bomb_at_cust = min(timebomb_truck[i] - time_to_cust, deadline[cust]);
            double time_back_to_depot = compute_truck_route_time({cust, 0}, current_time + time_to_cust).first;
            if (time_bomb_at_cust - time_back_to_depot > 1e-8 && capacity_used_truck[i] + demand[cust] <= (double)Dh + 1e-9) {
                r.push_back(cust);
                service_times_truck[i] += time_to_cust;
                capacity_used_truck[i] += demand[cust];
                timebomb_truck[i] = min(timebomb_truck[i] - time_to_cust, deadline[cust]);
                sol.truck_routes[i] = r;
                visited[cust] = true;
                assigned = true;
                num_of_visited_customers++;
                break;
            }
            // Then try drone
            r = sol.drone_routes[i];
            current_node = r.empty() ? 0 : r.back();
            current_time = service_times_drone[i];
            time_to_cust = compute_drone_route_time({current_node, cust}).first;
            time_bomb_at_cust = min(timebomb_drone[i] - time_to_cust, deadline[cust]);
            time_back_to_depot = compute_drone_route_time({cust, 0}).first;
            double total_energy = (time_to_cust - serve_drone[cust]) * (power_beta * capacity_used_drone[i] + power_gamma)
                                  + (time_back_to_depot) * (power_beta * (capacity_used_drone[i] + demand[cust]) + power_gamma);
            if (time_bomb_at_cust - time_back_to_depot > 1e-8
                && capacity_used_drone[i] + demand[cust] <= Dd + 1e-9
                && energy_used_drone[i] + total_energy <= E + 1e-9) {
                r.push_back(cust);
                service_times_drone[i] += time_to_cust;
                energy_used_drone[i] += total_energy;
                capacity_used_drone[i] += demand[cust];
                timebomb_drone[i] = min(timebomb_drone[i] - time_to_cust, deadline[cust]);
                sol.drone_routes[i] = r;
                visited[cust] = true;
                assigned = true;
                num_of_visited_customers++;
                break;
            }
        }
    }
    // Collapse number of drone routes from h to d if d < h
    if (d < h) {
        // Simple strategy: keep first d drone routes, merge the rest into these d routes round-robin
        int j = 0;
        for (int i = h - 1; i >= d; --i) {
            vi& r = sol.drone_routes[i];
            if (r.size() <= 1) continue; // empty or only depot
            // Try to merge into one of the first d routes
            vi& target_route = sol.drone_routes[j];
            if (target_route.empty()) target_route.push_back(0);
            // Always can merge
            // if target_route ends with depot, remove it first
            if (target_route.back() == 0) target_route.pop_back();
            // insert r[0..end] before the last depot of target_route
            target_route.insert(target_route.end(), r.begin(), r.end());
            r.clear();
            // remove r cause solution only allow d drone routes
            sol.drone_routes.pop_back();
            j = (j + 1) % d;
        }
    }
    else if (d > h) {
        // If more drones than trucks, add empty routes for the extra drones
        for (int i = h; i < d; ++i) {
            sol.drone_routes.push_back({0});
        }
    }

    // Finally, ensure all routes end at depot
    for (int i = 0; i < h; ++i) {
        if (sol.truck_routes[i].empty() || sol.truck_routes[i].back() != 0) {
            int current_node = sol.truck_routes[i].empty() ? 0 : sol.truck_routes[i].back();
            if (current_node != 0) {
                sol.truck_routes[i].push_back(0);
            }
        }
        if (sol.drone_routes[i].empty() || sol.drone_routes[i].back() != 0) {
            int current_node = sol.drone_routes[i].empty() ? 0 : sol.drone_routes[i].back();
            if (current_node != 0) {
                sol.drone_routes[i].push_back(0);
            }
        }
    }
    cout << "Number of visited customers in initial solution: " << num_of_visited_customers << " out of " << n << "\n";
    return sol;
}

void print_solution(const Solution& sol) {
    cout << "Truck Routes:\n";
    for (int i = 0; i < h; ++i) {
        cout << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            cout << node << " ";
        }
        cout << "\n";
    }
    cout << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        cout << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            cout << node << " ";
        }
        cout << "\n";
    }
}

void solve(){
    //Check input correctness
    cout << "Input Data:\n";
    cout << "Number of customers: " << n << "\n";
    cout << "Number of trucks: " << h << "\n";
    cout << "Number of drones: " << d << "\n";
    cout << "Truck capacities: " << Dh << "\n";
    cout << "Drone capacities: " << Dd << "\n";
    cout << "Drone energy capacity: " << E << "\n";
    cout << "Truck speed segments (sigma): ";
    for (double sigma : time_segments_sigma) cout << sigma << " ";
    cout << "\n";
    cout << "Customers:\n";
    cout << "ID\tX\tY\tServeTruck\tServeDrone\tServedByDrone\tDeadline\tDemand\n";
    for (int i = 0; i <= n; ++i) {
        cout << loc[i].id << "\t" << loc[i].x << "\t" << loc[i].y << "\t"
             << serve_truck[i] << "\t\t" << serve_drone[i] << "\t\t"
             << served_by_drone[i] << "\t\t" << deadline[i] << "\t\t" << demand[i] << "\n";
    }
    // Compute distance matrices
    compute_distance_matrices(loc);
    update_served_by_drone();
    // Debug: print served_by_drone
    Solution sol = generate_initial_solution();
    print_solution(sol);
    // check feasibility of each route
    double total_time = 0.0;
    cout << "Route Feasibility Check:\n";
    for (int i = 0; i < h; ++i) {
        auto [time_truck, feas_truck] = check_truck_route_feasibility(sol.truck_routes[i], 0.0);
        total_time += time_truck;
        cout << "Truck " << i+1 << " route time: " << time_truck << " seconds, feasible: " << (feas_truck ? "Yes" : "No") << "\n";
    }
    for (int i = 0; i < d; ++i) {
        auto [time_drone, feas_drone] = check_drone_route_feasibility(sol.drone_routes[i]);
        total_time += time_drone;
        cout << "Drone " << i+1 << " route time: " << time_drone << " seconds, feasible: " << (feas_drone ? "Yes" : "No") << "\n";
    }
    cout << "Total time (sum of all routes): " << total_time << " seconds\n";
}

void output(){
}

int main(){
    ios::sync_with_stdio(0);
    string instance_file = "200.10.2.txt";
    freopen(instance_file.c_str(),"r",stdin);
    freopen("output.txt","w",stdout);
    cin.tie(0);
    cout.tie(0);
    auto start = chrono::high_resolution_clock::now();
    input(instance_file);
    solve();
    output();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;
    cerr << "Time: " << duration.count() << " ms" << endl;
    return 0;
}