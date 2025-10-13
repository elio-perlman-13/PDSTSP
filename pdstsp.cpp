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
vd time_segment = {0, 0.2, 0.4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
vd time_segments_sigma = {0.9, 0.4, 0.6, 0.8, 0.4, 0.6,0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 2.27, E = 7200000.0; //drone's weight and energy capacities (for all drones)
double v_fly_drone = 31.3, v_take_off = 15.6, v_landing = 7.8, height = 50; // maximum speed of the drone
double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    double total_cost; //total cost of the solution
};

struct RunMetrics {
    string instance;
    int n = 0;
    int h = 0;
    int d = 0;
    double total_time = 0.0;        // sum of all route times
    double makespan = 0.0;          // max route time (objective)
    double truck_time_sum = 0.0;     // sum of truck route times
    double drone_time_sum = 0.0;     // sum of drone route times
    int unserved = 0;                // number of customers not served
    bool feasible = true;            // overall feasibility
    double wall_ms = 0.0;            // wall clock time for solving (ms)
    double avg_drone_customers_per_sortie = 0.0; // average customers per drone depot-to-depot sortie
    double avg_truck_customers_per_sortie = 0.0; // average customers per truck depot-to-depot sortie
};

static vector<RunMetrics> g_batch_metrics; // collected over batch runs

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
    // Prepare storage (use assign to ensure inner dimensions reset, avoiding stale sizes across batch runs)
    served_by_drone.assign(n+1, 0);
    serve_truck.assign(n+1, 0.0);
    serve_drone.assign(n+1, 0.0);
    deadline.assign(n+1, 0.0);
    demand.assign(n+1, 0.0);
    loc.assign(n+1, Point());
    distance_matrix.assign(n+1, vd(n+1, 0.0));
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

// Returns pair of distance matrices
void compute_distance_matrices(const vector<Point>& loc) {
    int n = loc.size() - 1; // assuming loc[0] is depot
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distance_matrix[i][j] = sqrt((loc[i].x - loc[j].x) * (loc[i].x - loc[j].x)
                                         + (loc[i].y - loc[j].y) * (loc[i].y - loc[j].y)); // Euclidean
        }
    }
}

// Helper: get time segment index for a given time t (in hours)
int get_time_segment(double t) {
    // t is in hours. Use custom time_segment boundaries (in hours):
    // time_segment: [b0, b1, ..., bk] defines k segments [b0,b1), [b1,b2), ... [b{k-1}, b{k}]
    // Return 0-based segment index in [0, k-1].
    // If outside boundaries, loop back to the start segment.
    t = fmod(t, 12.0);
    if (time_segment.size() < 2) return 0;
    // Find first boundary strictly greater than t
    auto it = upper_bound(time_segment.begin(), time_segment.end(), t);
    int idx = static_cast<int>(it - time_segment.begin()) - 1; // index of segment start
    if (idx < 0) idx = 0;
    int max_idx = static_cast<int>(time_segment.size()) - 2; // last valid segment index
    if (idx > max_idx) idx = max_idx;
    return idx;
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
        // Defensive: cap number of segment steps to avoid infinite loops due to numeric edge cases
        int guard_steps = 0;
        while (dist_left > 1e-8) {
            if (++guard_steps > 1000000) {
                // Fallback: assume constant speed and finish remaining distance
                double v_safe = vmax > 1e-6 ? vmax : 1.0;
                time += dist_left / v_safe;
                dist_left = 0.0;
                break;
            }
            // Convert time to hours for segment lookup
            double t_hr = time / 3600.0;
            int seg = get_time_segment(t_hr); // 0-based index into time_segments_sigma
            double v = vmax * (seg < (int)time_segments_sigma.size() ? time_segments_sigma[seg] : 1.0); // m/s
            if (v <= 1e-8) v = vmax;
            // Time left in this custom segment (seconds to next boundary)
            double next_boundary_hr = (seg + 1 < (int)time_segment.size()) ? time_segment[seg + 1] : std::numeric_limits<double>::infinity();
            double segment_end_time_sec;
            if (std::isinf(next_boundary_hr)) segment_end_time_sec = time + 1e18; // effectively no boundary ahead
            else segment_end_time_sec = next_boundary_hr * 3600.0;
            double t_seg_end = segment_end_time_sec - time; // seconds remaining in this segment
            if (t_seg_end < 1e-8) t_seg_end = 1e-6; // minimal progress to avoid stalling
            double max_dist_this_seg = v * t_seg_end;
            if (max_dist_this_seg <= 1e-12) {
                // Make minimal forward progress to avoid stalling due to underflow
                max_dist_this_seg = std::max(1e-6, v * 1e-6);
            }
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
                return make_pair(time, false);
            } // exceeded capacity
        }
    }
    // Check energy
    auto [total_energy, feasible_energy] = compute_drone_route_energy(route);
    if (!feasible_energy) {
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
    // First centroid
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
            double bestDist2 = 1e30;
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
    int iter = 2000; // max iterations
    int stall_count = 0; // number of consecutive iterations without meaningful progress
    // Track whether each vehicle (truck/drone) is still active (eligible for selection)
    vector<char> active_truck(h, 1), active_drone(h, 1);
    while (num_of_visited_customers < n && iter > 0) {
        iter--;
        bool made_progress = false;
        // Select the vehicle (truck or drone) with the smallest current service time
        if (h <= 0) break; // no vehicles available
        int best_vehicle = -1; // 0..h-1 trucks, h..2h-1 drones
        double best_time_val = 1e100;
        int active_count = 0;
        for (int i = 0; i < h; ++i) {
            if (!active_truck[i]) continue;
            ++active_count;
            if (service_times_truck[i] < best_time_val) {
                best_time_val = service_times_truck[i];
                best_vehicle = i; // truck i
            }
        }
        for (int i = 0; i < h; ++i) { // consider paired drone index i
            if (!active_drone[i]) continue;
            ++active_count;
            if (service_times_drone[i] < best_time_val) {
                best_time_val = service_times_drone[i];
                best_vehicle = h + i; // drone i
            }
        }
        if (active_count == 0 || best_vehicle == -1) break; // no active vehicles remaining
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
                // test different scoring formulas here
                //double score = norm_change + (cand.same_cluster ? 0.0 : 1.0);
                double score = w1 * cand.urgency_score * cand.urgency_score + w2 * cand.capacity_ratio * cand.capacity_ratio + norm_change + (cand.same_cluster ? 0.0 : w1 + w2 + 1.0);
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
                made_progress = true;
            }
            if (!assigned) {
                // No feasible customer found.
                int current_node2 = current_route.empty() ? 0 : current_route.back();
                if (current_node2 != 0) {
                    // Force return to depot
                    vi r = sol.truck_routes[truck_idx];
                    double time_to_depot = compute_truck_route_time({current_node2, 0}, service_times_truck[truck_idx]).first;
                    service_times_truck[truck_idx] += time_to_depot;
                    r.push_back(0);
                    sol.truck_routes[truck_idx] = r;
                    timebomb_truck[truck_idx] = 1e18; // reset timebomb after returning to depot
                    // Reset capacity after completing a tour at the depot
                    capacity_used_truck[truck_idx] = 0.0;
                    made_progress = true;
                }
                else {
                    active_truck[truck_idx] = 0;
                    made_progress = true;
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
                // Formula 1: remove capacity_ratio and energy_ratio
                //double score = norm_change + (cand.same_cluster ? 0.0 : 1.0);
                double score = w1 * cand.urgency_score * cand.urgency_score + w2 * cand.capacity_ratio * cand.capacity_ratio + w3 * cand.energy_ratio * cand.energy_ratio + norm_change + (cand.same_cluster ? 0.0 : w1 + w2 + w3 + 1.0);
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
                made_progress = true;
            }
            if (!assigned) {
                int current_node2 = current_route.empty() ? 0 : current_route.back();
                if (current_node2 != 0) {
                    vi r = sol.drone_routes[drone_idx];
                    double time_to_depot = compute_drone_route_time({current_node2, 0}).first;
                    service_times_drone[drone_idx] += time_to_depot;
                    r.push_back(0);
                    sol.drone_routes[drone_idx] = r;
                    timebomb_drone[drone_idx] = 1e18; // reset timebomb after returning to depot
                    capacity_used_drone[drone_idx] = 0.0;
                    energy_used_drone[drone_idx] = 0.0;
                    made_progress = true;
                }
                else {
                    
                    active_drone[drone_idx] = 0; // deactivate drone
                    made_progress = true;
                }
            }
        }
        // Stall detection: if no progress this iteration, count and break after a threshold
        if (!made_progress) {
            if (++stall_count > max(100, 2*h + 2*d)) {
                cerr << "Warning: construction stalled; breaking after repeated no-progress iterations.\n";
                break;
            }
        } else {
            stall_count = 0;
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
    // Reallocate drone sorties (depot-to-depot) across d drones using LPT to minimize makespan
    {
        // Extract sorties from current drone routes
        struct Sortie { vi route; double duration; };
        vector<Sortie> sorties;
        sorties.reserve(n); // rough upper bound
        for (const auto &r : sol.drone_routes) {
            if (r.size() <= 1) continue;
            vi cur; cur.push_back(0);
            for (size_t i = 1; i < r.size(); ++i) {
                int node = r[i];
                if (node == 0) {
                    // close current sortie
                    if (cur.back() != 0) cur.push_back(0);
                    double dur = compute_drone_route_time(cur).first;
                    if (cur.size() > 2) sorties.push_back({cur, dur}); // non-empty sortie
                    cur.clear(); cur.push_back(0);
                } else {
                    cur.push_back(node);
                }
            }
            // If route ended without a closing depot, close it
            if (cur.size() > 1) {
                if (cur.back() != 0) cur.push_back(0);
                double dur = compute_drone_route_time(cur).first;
                if (cur.size() > 2) sorties.push_back({cur, dur});
            }
        }

        // Prepare target number of drones
        if (d <= 0) {
            sol.drone_routes.clear();
        } else {
            // LPT: sort sorties by duration descending
            vector<int> idx(sorties.size());
            iota(idx.begin(), idx.end(), 0);
            sort(idx.begin(), idx.end(), [&](int a, int b){ return sorties[a].duration > sorties[b].duration; });

            // Min-heap of (load, drone_id)
            struct Load { double t; int id; };
            struct Cmp { bool operator()(const Load &a, const Load &b) const { return a.t > b.t; } };
            priority_queue<Load, vector<Load>, Cmp> pq;
            for (int k = 0; k < d; ++k) pq.push({0.0, k});

            vector<vector<int>> assigned(d); // store concatenated routes per drone as sequences of nodes
            vector<double> loads(d, 0.0);
            // Assign each sortie to the least-loaded drone
            for (int id : idx) {
                Load cur = pq.top(); pq.pop();
                int k = cur.id;
                // Append sortie to drone k's sequence (avoid duplicating initial depot)
                const vi &s = sorties[id].route; // s is [0, ..., 0]
                if (assigned[k].empty()) assigned[k].push_back(0);
                // remove trailing 0 if present to avoid double depot before appending
                if (!assigned[k].empty() && assigned[k].back() == 0) {
                    // keep it; we'll append from s.begin()+1
                }
                assigned[k].insert(assigned[k].end(), s.begin() + 1, s.end());
                loads[k] = cur.t + sorties[id].duration;
                pq.push({loads[k], k});
            }

            // Build sol.drone_routes from assigned sequences
            sol.drone_routes.clear();
            sol.drone_routes.resize(d);
            for (int k = 0; k < d; ++k) {
                if (assigned[k].empty()) {
                    sol.drone_routes[k] = {0};
                } else {
                    sol.drone_routes[k] = std::move(assigned[k]);
                    // Ensure ends at depot
                    if (sol.drone_routes[k].back() != 0) sol.drone_routes[k].push_back(0);
                }
            }
        }
    }

    // Finally, ensure all routes end at depot
    for (int i = 0; i < h; ++i) {
        if (sol.truck_routes[i].empty() || sol.truck_routes[i].back() != 0) {
            int current_node = sol.truck_routes[i].empty() ? 0 : sol.truck_routes[i].back();
            if (current_node != 0) sol.truck_routes[i].push_back(0);
        }
    }
    // Drone routes may have size d (could be < h). Safely finalize only existing routes.
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
        if (sol.drone_routes[i].empty() || sol.drone_routes[i].back() != 0) {
            int current_node = sol.drone_routes[i].empty() ? 0 : sol.drone_routes[i].back();
            if (current_node != 0) sol.drone_routes[i].push_back(0);
        }
    }
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

// Print the (n+1)x(n+1) distance matrix (Euclidean) with depot = 0.
// Wrapped with BEGIN/END markers to allow easy parsing and optional skipping.
void print_distance_matrix(){
    cout.setf(std::ios::fixed); cout << setprecision(6);
    cout << "BEGIN_DISTANCE_MATRIX\n";
    // Header row (comma separated): idx,0,1,...,n
    cout << "idx";
    for(int j=0;j<=n;++j) cout << "," << j;
    cout << "\n";
    for(int i=0;i<=n;++i){
        cout << i;
        for(int j=0;j<=n;++j){
            cout << "," << distance_matrix[i][j];
        }
        cout << "\n";
    }
    cout << "END_DISTANCE_MATRIX\n";
}

namespace {
    struct SilenceCout {
        std::streambuf* old = nullptr; std::ostringstream sink; bool active=false;
        SilenceCout(bool enable){ if(enable){ old = cout.rdbuf(sink.rdbuf()); active=true; } }
        ~SilenceCout(){ if(active) cout.rdbuf(old); }
    };
    struct EvalResult { double total_time=0, makespan=0, truck_sum=0, drone_sum=0; int unserved=0; bool feasible=true; };
    EvalResult evaluate_solution(const Solution &sol, bool suppress){
        SilenceCout sc(suppress); // suppress route feasibility prints if requested
        EvalResult res; res.feasible = true;
        vector<bool> served_flag(n+1,false);
        double max_route_time = 0.0;
        for(int i=0;i<h;++i){
            auto [t,f] = check_truck_route_feasibility(sol.truck_routes[i],0.0);
            res.total_time += t; res.truck_sum += t; res.feasible &= f;
            max_route_time = max(max_route_time, t);
            for(int node: sol.truck_routes[i]) if(node!=0) served_flag[node]=true;
        }
        for(int i=0;i<d;++i){
            if(i >= (int)sol.drone_routes.size()) break; // safety
            auto [t,f] = check_drone_route_feasibility(sol.drone_routes[i]);
            res.total_time += t; res.drone_sum += t; res.feasible &= f;
            max_route_time = max(max_route_time, t);
            for(int node: sol.drone_routes[i]) if(node!=0) served_flag[node]=true;
        }
        res.makespan = max_route_time;
        for(int cust=1; cust<=n; ++cust) if(!served_flag[cust]) { res.unserved++; res.feasible=false; }
        return res;
    }
}

void solve(RunMetrics &metrics, int attempts=1){
    metrics.n = n; metrics.h = h; metrics.d = d;
    compute_distance_matrices(loc);
    update_served_by_drone();

    Solution best_sol; EvalResult best_eval; bool has_best=false; int best_attempt=-1;

    for(int attempt=0; attempt<attempts; ++attempt){
        Solution sol = generate_initial_solution();
        EvalResult eval = evaluate_solution(sol, /*suppress=*/true);
        bool better = false;
        if(eval.feasible) {
            if(!has_best) better = true;
            else if(!best_eval.feasible) better = true;
            else if(eval.makespan < best_eval.makespan - 1e-9) better = true; // primary objective
            else if(fabs(eval.makespan - best_eval.makespan) <= 1e-9 && eval.total_time < best_eval.total_time - 1e-9) better = true; // tie-breaker
        } else {
            if(!has_best) better = true; // first solution even if infeasible
            else if(!best_eval.feasible && eval.makespan < best_eval.makespan - 1e-9) better = true;
        }
        if(better){ best_sol = std::move(sol); best_eval = eval; has_best=true; best_attempt = attempt; }
    }

    // Re-evaluate best solution without suppressing to ensure consistency (but avoid duplicate feasibility prints)
    // We'll just trust earlier eval and not re-run check_* that print messages; instead print routes & metrics.
    metrics.total_time = best_eval.total_time; // keep sum for reference
    metrics.makespan = best_eval.makespan;
    metrics.truck_time_sum = best_eval.truck_sum;
    metrics.drone_time_sum = best_eval.drone_sum;
    metrics.unserved = best_eval.unserved;
    metrics.feasible = best_eval.feasible;

    // Output chosen solution routes for downstream plotting
    cout << "Attempts: " << attempts << "\n";
    cout << "Selected attempt (0-based): " << best_attempt << "\n";
    print_solution(best_sol);
    cout << "Makespan (objective - latest vehicle finish): " << metrics.makespan << " seconds\n";
    cout << "Total time (sum of all routes - secondary): " << metrics.total_time << " seconds\n";
    cout << "Truck time sum: " << metrics.truck_time_sum << " seconds\n";
    cout << "Drone time sum: " << metrics.drone_time_sum << " seconds\n";
    cout << "Unserved customers: " << metrics.unserved << "\n";
    cout << "Overall feasibility: " << (metrics.feasible ? "Yes" : "No") << "\n";

    // Calculate average number of customers served per drone sortie (depot-to-depot)
    int total_sorties = 0, total_customers = 0;
    for (const auto& route : best_sol.drone_routes) {
        int last_depot = 0;
        int served_in_this_sortie = 0;
        for (size_t i = 1; i < route.size(); ++i) {
            if (route[i] == 0) {
                // End of a sortie
                total_sorties++;
                total_customers += served_in_this_sortie;
                served_in_this_sortie = 0;
            } else {
                served_in_this_sortie++;
            }
        }
        // If route does not end with depot, count last segment
        if (route.size() > 1 && route.back() != 0) {
            total_sorties++;
            total_customers += served_in_this_sortie;
        }
    }
    double avg_customers_per_sortie = (total_sorties > 0) ? (double)total_customers / total_sorties : 0.0;
    cout << "Average customers per drone sortie (depot-to-depot): " << avg_customers_per_sortie << "\n";
    metrics.avg_drone_customers_per_sortie = avg_customers_per_sortie;

    // Calculate average number of customers served per truck sortie (depot-to-depot)
    int truck_total_sorties = 0, truck_total_customers = 0;
    for (const auto& route : best_sol.truck_routes) {
        if (route.size() <= 1) continue; // no movement
        int served_in_this_sortie = 0;
        for (size_t i = 1; i < route.size(); ++i) {
            if (route[i] == 0) { // end of a sortie
                truck_total_sorties++;
                truck_total_customers += served_in_this_sortie;
                served_in_this_sortie = 0;
            } else {
                served_in_this_sortie++;
            }
        }
        // Safety: if route somehow didn't end at depot (shouldn't happen due to finalization), count remaining
        if (route.back() != 0 && served_in_this_sortie > 0) {
            truck_total_sorties++;
            truck_total_customers += served_in_this_sortie;
        }
    }
    double avg_truck_customers_per_sortie = (truck_total_sorties > 0) ? (double)truck_total_customers / truck_total_sorties : 0.0;
    cout << "Average customers per truck sortie (depot-to-depot): " << avg_truck_customers_per_sortie << "\n";
    metrics.avg_truck_customers_per_sortie = avg_truck_customers_per_sortie;
}

void output(){
}

// Run a single instance and write its per-instance output to a file (without affecting global stdout)
static void run_single_instance(const string& instance_path, const string& output_path) {
    auto start = chrono::high_resolution_clock::now();
    // Redirect cout to file
    ofstream ofs(output_path);
    if (!ofs) {
        cerr << "Cannot open output file: " << output_path << "\n";
        return;
    }
    std::streambuf* old_buf = cout.rdbuf(ofs.rdbuf());
    // Load & solve
    input(instance_path);
    RunMetrics metrics; metrics.instance = instance_path;
    solve(metrics, 5); // run 5 randomized attempts and keep best
    output();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;
    metrics.wall_ms = duration.count();
    cout << "Instance: " << instance_path << "  Time: " << duration.count() << " ms" << endl;
    cout.rdbuf(old_buf); // restore
    cerr << "Finished: " << instance_path << " in " << duration.count() << " ms -> " << output_path << "\n";
    g_batch_metrics.push_back(metrics);
}

int main(int argc, char* argv[]) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Usage:
    //   ./pdstsp [path_or_dir] [optional_output_file_or_dir]
    // If first arg is a directory: run all *.txt inside it (non-recursive)
    // If first arg is a file: run just that file.
    // If no args: default directory "instance" (run all *.txt).
    using std::filesystem::path;
    using std::filesystem::is_directory;
    using std::filesystem::directory_iterator;
    using std::filesystem::create_directories;

    path target = (argc >= 1 && argv[0]) ? path(".") : path("."); // base (not used directly)
    if (argc >= 2) target = path(argv[1]); else target = path("instance");
    path out_arg;
    bool custom_out = false;
    if (argc >= 3) { out_arg = path(argv[2]); custom_out = true; }

    vector<pair<string,string>> jobs; // (instance_path, output_path)

    if (is_directory(target)) {
        path out_dir;
        if (custom_out) {
            out_dir = out_arg;
        } else {
            out_dir = path("outputs");
        }
        create_directories(out_dir);
        // Collect .txt files
        for (auto& entry : directory_iterator(target)) {
            if (!entry.is_regular_file()) continue;
            auto p = entry.path();
            if (p.extension() == ".txt") {
                string stem = p.stem().string();
                string out_file = (out_dir / (stem + "_output.txt")).string();
                jobs.emplace_back(p.string(), out_file);
            }
        }
        // Sort jobs by instance filename for deterministic ordering
        sort(jobs.begin(), jobs.end(), [](auto& a, auto& b){return a.first < b.first;});
        if (jobs.empty()) {
            cerr << "No .txt instances found in directory: " << target << "\n";
            return 1;
        }
    } else {
        // Single file
        if (!std::filesystem::exists(target)) {
            cerr << "Path does not exist: " << target << "\n";
            return 1;
        }
        string output_file;
        if (custom_out) {
            if (std::filesystem::is_directory(out_arg)) {
                // Place inside directory with derived name
                string stem = target.stem().string();
                create_directories(out_arg);
                output_file = (out_arg / (stem + "_output.txt")).string();
            } else {
                output_file = out_arg.string();
            }
        } else {
            // Derive default output filename next to executable
            output_file = target.stem().string() + string("_output.txt");
        }
        jobs.emplace_back(target.string(), output_file);
    }

    cerr << "Running " << jobs.size() << " instance(s)...\n";
    auto batch_start = chrono::high_resolution_clock::now();
    for (auto& [inst, out] : jobs) {
        run_single_instance(inst, out);
    }
    auto batch_end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> total_ms = batch_end - batch_start;
    cerr << "Batch finished in " << total_ms.count() << " ms" << endl;

    // Write summary CSV if multiple instances or even single (for consistency)
    if (!g_batch_metrics.empty()) {
        // Decide summary path: if directory input -> outputs/summary.csv or custom out_dir; else alongside first output file
        string summary_path = "summary.csv";
        // Try to infer from first job's output_path directory
        if (!jobs.empty()) {
            std::filesystem::path outp(jobs.front().second);
            auto parent = outp.parent_path();
            if (!parent.empty()) summary_path = (parent / "summary.csv").string();
        }
        ofstream csv(summary_path);
        if (!csv) {
            cerr << "Could not write summary CSV at " << summary_path << "\n";
        } else {
            csv << "instance,n,h,d,makespan,total_time,truck_time_sum,drone_time_sum,unserved,feasible,runtime_ms,avg_drone_customers_per_sortie,avg_truck_customers_per_sortie\n";
            csv.setf(std::ios::fixed); csv << setprecision(6);
            for (auto &m : g_batch_metrics) {
                csv << m.instance << ',' << m.n << ',' << m.h << ',' << m.d << ','
                    << m.makespan << ',' << m.total_time << ',' << m.truck_time_sum << ',' << m.drone_time_sum << ','
                    << m.unserved << ',' << (m.feasible ? 1 : 0) << ',' << m.wall_ms << ','
                    << m.avg_drone_customers_per_sortie << ',' << m.avg_truck_customers_per_sortie << '\n';
            }
            cerr << "Summary written: " << summary_path << " (" << g_batch_metrics.size() << " rows)\n";
        }
    }
    return 0;
}