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
vd serve_truck, serve_drone; // time taken by truck and drone to serve each customer (seconds)
vi served_by_drone; //whether each customer can be served by drone or not, 1 if yes, 0 if no
vd deadline; //customer deadlines
vd demand; // demand[i]: demand of customer i
double Dh = 1400.0; // truck capacity (all trucks) (kg)
double vmax = 15.6464; // truck base speed (m/s)
int L = 24; //number of time segments in a day
vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
vd time_segments_sigma = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; //sigma (truck velocity coefficient) for each time segments
//vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
//vd time_segments_sigma = {0.9, 0.8, 0.4, 0.6,0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 2.27, E = 700.0; //drone's weight and energy capacities (for all drones)
double v_fly_drone = 31.2928, v_take_off = 15.6464, v_landing = 7.8232; // speed of the drone
//double height = 50; // height of the drone
double height = 0; // height of the drone
double power_beta = 0, power_gamma = 1.0; //coefficients for drone energy consumption per second
//double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

// Candidate lists (k-nearest neighbors) to filter neighborhood evaluations
static int CFG_KNN_K = 30;           // number of nearest neighbors per customer
static int CFG_KNN_WINDOW = 2;       // insertion window around candidate anchors
static vvi KNN_LIST;                 // KNN_LIST[i] = up to K nearest neighbor customer ids for i (exclude depot 0)
static vector<vector<char>> KNN_ADJ; // KNN_ADJ[i][j] = 1 if j in KNN_LIST[i]

// Simple tabu structure for relocate moves: tabu_list_switch[cust][target_vehicle] stores iteration until which move is tabu
// target_vehicle is 0..h-1 for trucks, h..h+d-1 for drones
static vector<vector<int>> tabu_list_switch; // sized (n+1) x (h + d), initialized on first use
static int TABU_TENURE_BASE = 20; // default tenure; actual update done in tabu loop (not here)
// Separate tabu structure for swap moves: store until-iteration for swapping a pair (min_id,max_id)
static vector<vector<int>> tabu_list_swap; // sized (n+1) x (n+1)
static int TABU_TENURE_SWAP = 20; // default tenure for swap moves
static vector<vector<int>> tabu_list_relocate; // sized (n+1) x (h + d)
static int TABU_TENURE_RELOCATE = 20; // default tenure for relocate moves
// Separate tabu list for intra-route reinsert (Or-opt-1) moves
static vector<vector<int>> tabu_list_reinsert; // sized (n+1) x (h + d)
static int TABU_TENURE_REINSERT = 20; // default tenure for reinsert moves
// Separate tabu list for 2-opt moves: keyed by segment endpoints (min_id,max_id)
static vector<vector<int>> tabu_list_2opt; // sized (n+1) x (n+1)
static int TABU_TENURE_2OPT = 20; // default tenure for 2-opt moves
const int NUM_NEIGHBORHOODS = 5;
const int NUM_OF_INITIAL_SOLUTIONS = 50;
const int MAX_SEGMENT = 20;
const int MAX_NO_IMPROVE = 100;
const int MAX_ITER_PER_SEGMENT = 1000;
const double gamma1 = 1.0;
const double gamma2 = 0.3;
const double gamma3 = 0.0;
const double gamma4 = 0.5;

// Runtime-configurable search knobs (initialized from compile-time defaults)
static int CFG_NUM_INITIAL = NUM_OF_INITIAL_SOLUTIONS;
static int CFG_MAX_SEGMENT = MAX_SEGMENT;
static int CFG_MAX_NO_IMPROVE = MAX_NO_IMPROVE;
static int CFG_MAX_ITER_PER_SEGMENT = MAX_ITER_PER_SEGMENT;
static double CFG_TIME_LIMIT_SEC = 0.0; // 0 = unlimited

// Helper to parse key=value flags from argv
static bool parse_kv_flag(const std::string& s, const std::string& key, std::string& out) {
    if (s.rfind(key + "=", 0) == 0) { out = s.substr(key.size() + 1); return true; }
    return false;
}

// Build k-nearest neighbor lists based on Euclidean distance_matrix.
// Excludes depot (0) and self; sizes to n+1. Also builds adjacency for O(1) membership checks.
static void compute_knn_lists(int k) {
    int N = n;
    if (N <= 1) {
        KNN_LIST.assign(N + 1, {});
        KNN_ADJ.assign(N + 1, vector<char>(N + 1, 0));
        return;
    }
    KNN_LIST.assign(N + 1, {});
    KNN_ADJ.assign(N + 1, vector<char>(N + 1, 0));
    vector<pair<double,int>> cand;
    cand.reserve(max(0, N - 1));
    for (int i = 1; i <= N; ++i) {
        cand.clear();
        for (int j = 1; j <= N; ++j) {
            if (j == i) continue;
            cand.emplace_back(distance_matrix[i][j], j);
        }
        int kk = min(k, (int)cand.size());
        if ((int)cand.size() > kk) {
            nth_element(cand.begin(), cand.begin() + kk, cand.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            cand.resize(kk);
        } else {
            sort(cand.begin(), cand.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        }
        KNN_LIST[i].reserve(kk);
        for (int t = 0; t < kk; ++t) {
            int j = cand[t].second;
            KNN_LIST[i].push_back(j);
            KNN_ADJ[i][j] = 1;
        }
    }
}


// Separate tabu list for 2-opt-star (inter-route exchange) moves: keyed by unordered edge endpoints (min(u,v), max(u,v))
static vector<vector<int>> tabu_list_2opt_star; // sized (n+1) x (n+1)
static int TABU_TENURE_2OPT_STAR = 25; // default tenure for 2-opt-star moves

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    vd truck_route_times; //truck_route_times[i]: total time of truck i
    vd drone_route_times; //drone_route_times[i]: total time of drone i
    double total_makespan; //total makespan of the solution
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

pair<double, bool> check_route_feasibility(const vi& route, double start=0, bool is_truck = true) {
    if (is_truck) {
        return check_truck_route_feasibility(route, start);
    } else {
        return check_drone_route_feasibility(route);
    }
}


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
    // Compute makespan and per-vehicle times (indexed vectors)
    double makespan = 0.0;
    sol.truck_route_times.assign(h, 0.0);
    for (int i = 0; i < h; ++i) {
        double t = 0.0;
        if (sol.truck_routes[i].size() > 1) {
            t = compute_truck_route_time(sol.truck_routes[i], 0.0).first;
        }
        sol.truck_route_times[i] = t;
        makespan = max(makespan, t);
    }
    sol.drone_route_times.assign((int)sol.drone_routes.size(), 0.0);
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
        double t = 0.0;
        if (sol.drone_routes[i].size() > 1) {
            t = compute_drone_route_time(sol.drone_routes[i]).first;
        }
        sol.drone_route_times[i] = t;
        makespan = max(makespan, t);
    }
    sol.total_makespan = makespan;
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

Solution local_search(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 10e10;
    // Depending on neighbor_id, implement different neighborhood structures
    if (neighbor_id == 0) {
        // Relocate 1 customer from the critical (longest-time) vehicle route to another route of the same mode
        // 1) Identify critical vehicle (truck or drone) using precomputed times
        bool crit_is_truck = true;
        int critical_idx = -1; // index within its mode
        double max_time = -1.0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)initial_solution.truck_route_times.size()) ? initial_solution.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; crit_is_truck = true; critical_idx = i; }
        }
        for (int i = 0; i < (int)initial_solution.drone_route_times.size(); ++i) {
            double t = initial_solution.drone_route_times[i];
            if (t > max_time) { max_time = t; crit_is_truck = false; critical_idx = i; }
        }
        if (critical_idx == -1) return initial_solution; // nothing to do

        // Ensure tabu list is sized to (n+1) x (h+d)
        int veh_count = h + d;
        if ((int)tabu_list_switch.size() != n + 1 || (veh_count > 0 && (int)tabu_list_switch[0].size() != veh_count)) {
            tabu_list_switch.assign(n + 1, vector<int>(max(0, veh_count), 0));
        }

        // Prepare neighborhood best tracking
        int best_target = -1; // vehicle index in unified space (0..h-1 trucks, h..h+d-1 drones)
        int best_pos = -1;    // insertion position within target route
        int best_cust = -1;   // moved customer id
        double best_neighbor_cost_local = best_neighbor_cost;
        Solution best_candidate_neighbor = best_neighbor;

        if (crit_is_truck) {
            const vi& crit_route = initial_solution.truck_routes[critical_idx];
            if ((int)crit_route.size() <= 2) return initial_solution; // nothing movable
            vector<int> crit_pos;
            for (int p = 0; p < (int)crit_route.size(); ++p) if (crit_route[p] != 0) crit_pos.push_back(p);
            for (int pos_idx : crit_pos) {
                int cust = crit_route[pos_idx];
                // Build critical truck route once with the customer removed
                vi nr = initial_solution.truck_routes[critical_idx];
                nr.erase(nr.begin() + pos_idx);
                if (nr.empty() || nr.front() != 0) nr.insert(nr.begin(), 0);
                // compress consecutive depots
                if (!nr.empty() && nr.size() >= 2) {
                    vi nr2; nr2.reserve(nr.size());
                    for (int x : nr) { if (!nr2.empty() && nr2.back() == 0 && x == 0) continue; nr2.push_back(x);} nr.swap(nr2);
                }
                if (!nr.empty() && nr.back() != 0) nr.push_back(0);

                // Check feasibility and compute time once for the reduced critical route
                auto [tcrit_base, feas_crit] = check_truck_route_feasibility(nr, 0.0);
                if (!feas_crit) continue;
                double crit_time_base = (nr.size() > 1) ? tcrit_base : 0.0;

                for (int target_truck = 0; target_truck < h; ++target_truck) {
                    if (target_truck == critical_idx) continue;
                    const vi& tgt_r = initial_solution.truck_routes[target_truck];
                    int insert_limit = (int)tgt_r.size();
                    for (int ins_pos = 1; ins_pos <= insert_limit; ++ins_pos) {
                        // Build target route with insertion
                        vi tr = tgt_r;
                        if (tr.empty()) tr.push_back(0);
                        int ip = min(max(ins_pos, 1), (int)tr.size());
                        tr.insert(tr.begin() + ip, cust);
                        if (tr.back() != 0) tr.push_back(0);

                        // Check feasibility for target route only
                        auto [ttgt, feas_tgt] = check_truck_route_feasibility(tr, 0.0);
                        if (!feas_tgt) continue;

                        // Construct neighbor incrementally
                        Solution neighbor = initial_solution;
                        neighbor.truck_routes[critical_idx] = nr;
                        neighbor.truck_routes[target_truck] = tr;

                        // Use cached times and recompute only modified routes
                        neighbor.truck_route_times = initial_solution.truck_route_times;
                        neighbor.drone_route_times = initial_solution.drone_route_times;
                        neighbor.truck_route_times[critical_idx] = crit_time_base;
                        neighbor.truck_route_times[target_truck] = (tr.size() > 1)
                            ? ttgt : 0.0;

                        double nb_makespan = 0.0;
                        for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                        for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                        neighbor.total_makespan = nb_makespan;

                        int target_id = target_truck; // unified vehicle id
                        bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                        tabu_list_switch[cust][target_id] >= current_iter);
                        if (is_tabu && neighbor.total_makespan >= best_cost) continue;

                        if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                        if (neighbor.total_makespan < best_neighbor_cost_local) {
                            best_target = target_id; best_pos = ins_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                        }
                    }
                }
                // Cross-mode: try inserting into drone routes as well
                for (int target_drone = 0; target_drone < (int)initial_solution.drone_routes.size(); ++target_drone) {
                    // Early prune: customer must be dronable
                    if (served_by_drone[cust] == 0) continue;
                    const vi& tgt_r_d = initial_solution.drone_routes[target_drone];
                    int insert_limit_d = (int)tgt_r_d.size();
                    for (int ins_pos = 1; ins_pos <= insert_limit_d; ++ins_pos) {
                        // Build target drone route with insertion
                        vi trd = tgt_r_d;
                        if (trd.empty()) trd.push_back(0);
                        int ipd = min(max(ins_pos, 1), (int)trd.size());
                        trd.insert(trd.begin() + ipd, cust);
                        if (trd.back() != 0) trd.push_back(0);

                        // Check feasibility for target drone route only
                        auto [ttgt_d, feas_tgt_d] = check_drone_route_feasibility(trd);
                        if (!feas_tgt_d) continue;

                        // Construct neighbor incrementally (cross-mode)
                        Solution neighbor = initial_solution;
                        neighbor.truck_routes[critical_idx] = nr;
                        neighbor.drone_routes[target_drone] = trd;

                        // Use cached times and recompute only modified routes
                        neighbor.truck_route_times = initial_solution.truck_route_times;
                        neighbor.drone_route_times = initial_solution.drone_route_times;
                        neighbor.truck_route_times[critical_idx] = crit_time_base;
                        neighbor.drone_route_times[target_drone] = (trd.size() > 1)
                            ? compute_drone_route_time(trd).first : 0.0;

                        double nb_makespan = 0.0;
                        for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                        for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                        neighbor.total_makespan = nb_makespan;

                        int target_id = h + target_drone; // unified vehicle id for drones
                        bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                        tabu_list_switch[cust][target_id] >= current_iter);
                        if (is_tabu && neighbor.total_makespan >= best_cost) continue;

                        if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                        if (neighbor.total_makespan < best_neighbor_cost_local) {
                            best_target = target_id; best_pos = ins_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                        }
                    }
                }
            }
        } else {
            // critical is a drone
            const vi& crit_route = initial_solution.drone_routes[critical_idx];
            if ((int)crit_route.size() <= 2) return initial_solution; // nothing movable
            vector<int> crit_pos;
            for (int p = 0; p < (int)crit_route.size(); ++p) if (crit_route[p] != 0) crit_pos.push_back(p);
            for (int pos_idx : crit_pos) {
                int cust = crit_route[pos_idx];
                // Build critical drone route once with the customer removed
                vi nr = initial_solution.drone_routes[critical_idx];
                nr.erase(nr.begin() + pos_idx);
                if (nr.empty() || nr.front() != 0) nr.insert(nr.begin(), 0);
                // compress consecutive depots
                if (!nr.empty() && nr.size() >= 2) {
                    vi nr2; nr2.reserve(nr.size());
                    for (int x : nr) { if (!nr2.empty() && nr2.back() == 0 && x == 0) continue; nr2.push_back(x);} nr.swap(nr2);
                }
                if (!nr.empty() && nr.back() != 0) nr.push_back(0);

                // Check feasibility and compute time once for the reduced critical route
                auto [tcrit_base, feas_crit] = check_drone_route_feasibility(nr);
                if (!feas_crit) continue;
                double crit_time_base = (nr.size() > 1) ? tcrit_base : 0.0;

                for (int target_drone = 0; target_drone < (int)initial_solution.drone_routes.size(); ++target_drone) {
                    if (target_drone == critical_idx) continue;
                    const vi& tgt_r = initial_solution.drone_routes[target_drone];
                    int insert_limit = (int)tgt_r.size();
                    for (int ins_pos = 1; ins_pos <= insert_limit; ++ins_pos) {
                        // Build target drone route with insertion
                        vi tr = tgt_r;
                        if (tr.empty()) tr.push_back(0);
                        int ip = min(max(ins_pos, 1), (int)tr.size());
                        tr.insert(tr.begin() + ip, cust);
                        if (tr.back() != 0) tr.push_back(0);

                        // Check feasibility for target route only
                        auto [ttgt, feas_tgt] = check_drone_route_feasibility(tr);
                        if (!feas_tgt) continue;

                        // Construct neighbor incrementally
                        Solution neighbor = initial_solution;
                        neighbor.drone_routes[critical_idx] = nr;
                        neighbor.drone_routes[target_drone] = tr;

                        // Use cached times and recompute only modified routes
                        neighbor.truck_route_times = initial_solution.truck_route_times;
                        neighbor.drone_route_times = initial_solution.drone_route_times;
                        neighbor.drone_route_times[critical_idx] = crit_time_base;
                        neighbor.drone_route_times[target_drone] = (tr.size() > 1)
                            ? compute_drone_route_time(tr).first : 0.0;

                        double nb_makespan = 0.0;
                        for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                        for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                        neighbor.total_makespan = nb_makespan;

                        int target_id = h + target_drone; // unified vehicle id for drones
                        bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                        tabu_list_switch[cust][target_id] >= current_iter);
                        if (is_tabu && neighbor.total_makespan >= best_cost) continue;

                        if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                        if (neighbor.total_makespan < best_neighbor_cost_local) {
                            best_target = target_id; best_pos = ins_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                        }
                    }
                }
                // Cross-mode: try inserting into truck routes as well
                for (int target_truck = 0; target_truck < h; ++target_truck) {
                    const vi& tgt_r_t = initial_solution.truck_routes[target_truck];
                    int insert_limit_t = (int)tgt_r_t.size();
                    for (int ins_pos = 1; ins_pos <= insert_limit_t; ++ins_pos) {
                        // Build target truck route with insertion
                        vi trt = tgt_r_t;
                        if (trt.empty()) trt.push_back(0);
                        int ipt = min(max(ins_pos, 1), (int)trt.size());
                        trt.insert(trt.begin() + ipt, cust);
                        if (trt.back() != 0) trt.push_back(0);

                        // Check feasibility for target truck route only
                        auto [ttgt_t, feas_tgt_t] = check_truck_route_feasibility(trt, 0.0);
                        if (!feas_tgt_t) continue;

                        // Construct neighbor incrementally (cross-mode)
                        Solution neighbor = initial_solution;
                        neighbor.drone_routes[critical_idx] = nr;
                        neighbor.truck_routes[target_truck] = trt;

                        // Use cached times and recompute only modified routes
                        neighbor.truck_route_times = initial_solution.truck_route_times;
                        neighbor.drone_route_times = initial_solution.drone_route_times;
                        neighbor.drone_route_times[critical_idx] = crit_time_base;
                        neighbor.truck_route_times[target_truck] = (trt.size() > 1)
                            ? compute_truck_route_time(trt, 0.0).first : 0.0;

                        double nb_makespan = 0.0;
                        for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                        for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                        neighbor.total_makespan = nb_makespan;

                        int target_id = target_truck; // unified vehicle id for trucks
                        bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                        tabu_list_switch[cust][target_id] >= current_iter);
                        if (is_tabu && neighbor.total_makespan >= best_cost) continue;

                        if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                        if (neighbor.total_makespan < best_neighbor_cost_local) {
                            best_target = target_id; best_pos = ins_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                        }
                    }
                }
            }
        }

        if (best_cust != -1) {
            // Update relocate tabu for the chosen best move (customer -> target vehicle)
            int veh_count2 = h + d;
            if ((int)tabu_list_switch.size() != n + 1 || (veh_count2 > 0 && (int)tabu_list_switch[0].size() != veh_count2)) {
                tabu_list_switch.assign(n + 1, vector<int>(max(0, veh_count2), 0));
            }
            if (best_cust >= 0 && best_cust <= n && best_target >= 0 && best_target < veh_count2) {
                tabu_list_switch[best_cust][best_target] = current_iter + TABU_TENURE_RELOCATE;
            }
            // Debug: print chosen relocate move
            //cout.setf(std::ios::fixed); cout << setprecision(6);
            /* cout << "[N0] relocate cust " << best_cust
                 << " -> " << ((best_target < h) ? "truck" : "drone")
                 << " #" << ((best_target < h) ? (best_target + 1) : (best_target - h + 1))
                 << " at pos " << best_pos
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_candidate_neighbor.total_makespan
                 << ", iter " << current_iter << "\n"; */
            return best_candidate_neighbor;
        }
    } else if (neighbor_id == 1) {
        // Neighborhood 1: Intra-route swap on the critical vehicle (truck or drone)
        // Separate tabu list for swaps: tabu_list_swap[min(a,b)][max(a,b)] stores expiration iteration
        bool crit_is_truck = true;
        int critical_idx = -1;
        double max_time = -1.0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)initial_solution.truck_route_times.size()) ? initial_solution.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; crit_is_truck = true; critical_idx = i; }
        }
        for (int i = 0; i < (int)initial_solution.drone_route_times.size(); ++i) {
            double t = initial_solution.drone_route_times[i];
            if (t > max_time) { max_time = t; crit_is_truck = false; critical_idx = i; }
        }
        if (critical_idx == -1) return initial_solution;

        // Size swap tabu list if needed
        if ((int)tabu_list_swap.size() != n + 1 || ((int)tabu_list_swap.size() > 0 && (int)tabu_list_swap[0].size() != n + 1)) {
            tabu_list_swap.assign(n + 1, vector<int>(n + 1, 0));
        }

    Solution best_local = initial_solution;
        double best_local_cost = 10e10;
    int best_swap_u = -1, best_swap_v = -1; // record best swap pair (unordered)
        // Ensure tabu list for reinsert is sized to (n+1) x (h+d)
        {
            int veh_count = h + d;
            if ((int)tabu_list_reinsert.size() != n + 1 || (veh_count > 0 && (int)tabu_list_reinsert[0].size() != veh_count)) {
                tabu_list_reinsert.assign(n + 1, vector<int>(max(0, veh_count), 0));
            }
        }

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode) {
            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            if ((int)pos.size() < 2) return; // nothing to swap
            for (int idx1 = 0; idx1 < (int)pos.size(); ++idx1) {
                for (int idx2 = idx1 + 1; idx2 < (int)pos.size(); ++idx2) {
                    int p1 = pos[idx1], p2 = pos[idx2];
                    int a = base_route[p1], b = base_route[p2];
                    if (a == b) continue;
                    // Candidate-list filter: only consider if a and b are near in KNN
                    if (!KNN_ADJ.empty()) {
                        if (!(KNN_ADJ.size() > (size_t)a && KNN_ADJ[a].size() > (size_t)b && KNN_ADJ[a][b]) &&
                            !(KNN_ADJ.size() > (size_t)b && KNN_ADJ[b].size() > (size_t)a && KNN_ADJ[b][a])) {
                            continue;
                        }
                    }
                    // Check swap tabu
                    int u = min(a,b), v = max(a,b);
                    bool is_tabu = (tabu_list_swap.size() > (size_t)u && tabu_list_swap[u].size() > (size_t)v &&
                                    tabu_list_swap[u][v] >= current_iter);
                    // Build swapped route
                    vi r = base_route;
                    swap(r[p1], r[p2]);
                    // Feasibility and time for the modified critical route
                    double tcrit = 0.0; bool feas = false;
                    auto [tt, ok] = check_route_feasibility(r, 0.0, is_truck_mode);
                    tcrit = (r.size() > 1) ? tt : 0.0; feas = ok;
                    if (!feas) continue;

                    // Construct neighbor and reuse cached times; only critical route time changes
                    Solution neighbor = initial_solution;
                    if (is_truck_mode) neighbor.truck_routes[critical_idx] = r; else neighbor.drone_routes[critical_idx] = r;
                    neighbor.truck_route_times = initial_solution.truck_route_times;
                    neighbor.drone_route_times = initial_solution.drone_route_times;
                    if (is_truck_mode) neighbor.truck_route_times[critical_idx] = tcrit; else neighbor.drone_route_times[critical_idx] = tcrit;

                    double nb_makespan = 0.0;
                    for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                    for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                    neighbor.total_makespan = nb_makespan;

                    // Aspiration: allow tabu if it improves global best_cost
                    if (is_tabu && neighbor.total_makespan >= best_cost) continue;

                    if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                    if (neighbor.total_makespan < best_local_cost) {
                        best_local_cost = neighbor.total_makespan; best_local = neighbor;
                        best_swap_u = u; best_swap_v = v;
                    } else if (fabs(neighbor.total_makespan - best_local_cost) <= 1e-12) {
                        // tie-breaker: choose one with better makespan; also update move if replaced
                        if (neighbor.total_makespan < best_local.total_makespan) {
                            best_local = neighbor;
                            best_swap_u = u; best_swap_v = v;
                        }
                    }
                }
            }
        };

        if (crit_is_truck) consider_swap(initial_solution.truck_routes[critical_idx], true);
        else consider_swap(initial_solution.drone_routes[critical_idx], false);

        // Update swap tabu for the best move if any
        if (best_swap_u != -1 && best_swap_v != -1) {
            if ((int)tabu_list_swap.size() != n + 1 || ((int)tabu_list_swap.size() > 0 && (int)tabu_list_swap[0].size() != n + 1)) {
                tabu_list_swap.assign(n + 1, vector<int>(n + 1, 0));
            }
            tabu_list_swap[best_swap_u][best_swap_v] = current_iter + TABU_TENURE_SWAP;
        }

        // Debug: print chosen swap move when available
        /* if (best_swap_u != -1 && best_swap_v != -1) {
            cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N1] swap customers " << best_swap_u << " <-> " << best_swap_v
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n";
        } */

        return best_local;
    } else if (neighbor_id == 2) {
        // Neighborhood 2: Intra-route relocate (Or-opt-1) on the critical route (truck or drone)
        // Remove one customer from the critical route and reinsert at a different position in the SAME route.
        bool crit_is_truck = true;
        int critical_idx = -1;
        double max_time = -1.0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)initial_solution.truck_route_times.size()) ? initial_solution.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; crit_is_truck = true; critical_idx = i; }
        }
        for (int i = 0; i < (int)initial_solution.drone_route_times.size(); ++i) {
            double t = initial_solution.drone_route_times[i];
            if (t > max_time) { max_time = t; crit_is_truck = false; critical_idx = i; }
        }
        if (critical_idx == -1) return initial_solution;

    const vi& base_route = crit_is_truck ? initial_solution.truck_routes[critical_idx]
                                             : initial_solution.drone_routes[critical_idx];
        // Collect positions of customers (exclude depots)
        vector<int> pos;
        for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
        if ((int)pos.size() < 2) return initial_solution; // nothing to relocate

    Solution best_local = initial_solution;
        double best_local_cost = 10e10;
    int best_reins_cust = -1;
    // Determine unified vehicle id for tabu encoding
    int veh_id_unified = crit_is_truck ? critical_idx : (h + critical_idx);

        for (int idx_from = 0; idx_from < (int)pos.size(); ++idx_from) {
            int p_from = pos[idx_from];
            int cust = base_route[p_from];

            // Build route with removal
            vi r = base_route;
            r.erase(r.begin() + p_from);
            // Normalize depots (keep start depot; ensure end depot; compress doubles)
            if (r.empty() || r.front() != 0) r.insert(r.begin(), 0);
            if (!r.empty() && r.size() >= 2) {
                vi r2; r2.reserve(r.size());
                for (int x : r) { if (!r2.empty() && r2.back() == 0 && x == 0) continue; r2.push_back(x);} r.swap(r2);
            }
            if (!r.empty() && r.back() != 0) r.push_back(0);

            int insert_limit = (int)r.size();
            // Candidate-list guided insertion positions
            vector<char> consider(max(0, insert_limit + 2), 0);
            if (!KNN_ADJ.empty()) {
                // Mark positions adjacent to KNN anchors
                for (int q = 0; q < (int)r.size(); ++q) {
                    int node = r[q];
                    if (node == 0) continue;
                    bool near = false;
                    if (KNN_ADJ.size() > (size_t)cust && KNN_ADJ[cust].size() > (size_t)node && KNN_ADJ[cust][node]) near = true;
                    if (!near && KNN_ADJ.size() > (size_t)node && KNN_ADJ[node].size() > (size_t)cust && KNN_ADJ[node][cust]) near = true;
                    if (!near) continue;
                    for (int delta = -CFG_KNN_WINDOW; delta <= CFG_KNN_WINDOW; ++delta) {
                        int ip1 = q + delta; // before node at q
                        int ip2 = q + 1 + delta; // after node at q
                        if (ip1 >= 1 && ip1 <= insert_limit) consider[ip1] = 1;
                        if (ip2 >= 1 && ip2 <= insert_limit) consider[ip2] = 1;
                    }
                }
            }
            // Always include a small window around original position to avoid missing obvious moves
            {
                int ip0 = min(max(1, p_from), insert_limit);
                for (int delta = -CFG_KNN_WINDOW; delta <= CFG_KNN_WINDOW; ++delta) {
                    int ip = ip0 + delta;
                    if (ip >= 1 && ip <= insert_limit) consider[ip] = 1;
                }
            }
            // If nothing marked (e.g., KNN disabled), consider all
            bool any_marked = false; for (int t = 1; t <= insert_limit; ++t) if (consider[t]) { any_marked = true; break; }
            for (int ins_pos = 1; ins_pos <= insert_limit; ++ins_pos) {
                if (!any_marked || consider[ins_pos]) {
                vi r_ins = r;
                int ip = min(max(ins_pos, 1), (int)r_ins.size());
                r_ins.insert(r_ins.begin() + ip, cust);
                if (r_ins.back() != 0) r_ins.push_back(0);

                // Skip no-op
                if (r_ins == base_route) continue;

                // Feasibility and time for the modified critical route
                auto [tcrit, feas] = check_route_feasibility(r_ins, 0.0, crit_is_truck);
                if (!feas) continue;

                // Tabu check for reinsert of this customer on this vehicle (with aspiration if it improves global best)
                int veh_id = veh_id_unified;
                bool is_tabu = false;
                if ((int)tabu_list_reinsert.size() > cust) {
                    const auto &row = tabu_list_reinsert[cust];
                    if ((int)row.size() > veh_id && row[veh_id] >= current_iter) {
                        is_tabu = true;
                    }
                }

                // Construct neighbor and reuse cached times; only critical route time changes
                Solution neighbor = initial_solution;
                if (crit_is_truck) neighbor.truck_routes[critical_idx] = r_ins; else neighbor.drone_routes[critical_idx] = r_ins;
                neighbor.truck_route_times = initial_solution.truck_route_times;
                neighbor.drone_route_times = initial_solution.drone_route_times;
                if (crit_is_truck) neighbor.truck_route_times[critical_idx] = (r_ins.size() > 1 ? tcrit : 0.0);
                else neighbor.drone_route_times[critical_idx] = (r_ins.size() > 1 ? tcrit : 0.0);

                double nb_makespan = 0.0;
                for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                neighbor.total_makespan = nb_makespan;

                // Aspiration criterion: allow tabu if it improves the global best_cost
                if (is_tabu && neighbor.total_makespan >= best_cost) continue;

                if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                if (neighbor.total_makespan < best_local_cost) { best_local_cost = neighbor.total_makespan; best_local = neighbor; best_reins_cust = cust; }
                }
            }
        }

        // Update reinsert tabu if a move was chosen
        if (best_reins_cust != -1) {
            int veh_count2 = h + d;
            if ((int)tabu_list_reinsert.size() != n + 1 || (veh_count2 > 0 && (int)tabu_list_reinsert[0].size() != veh_count2)) {
                tabu_list_reinsert.assign(n + 1, vector<int>(max(0, veh_count2), 0));
            }
            tabu_list_reinsert[best_reins_cust][veh_id_unified] = current_iter + TABU_TENURE_REINSERT;
        }

        // Debug: print chosen reinsert move when available
        /* if (best_reins_cust != -1) {
            cout.setf(std::ios::fixed); cout << setprecision(6);
            bool is_truck = veh_id_unified < h;
            cout << "[N2] reinsert cust " << best_reins_cust << " within "
                 << (is_truck ? "truck" : "drone") << " #"
                 << (is_truck ? (veh_id_unified + 1) : (veh_id_unified - h + 1))
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n";
        } */

        return best_local;
    } else if (neighbor_id == 3) {
        // Neighborhood 3: 2-opt within each subroute (between depot nodes) for trucks or drones
        // Priority: try critical route first; if no admissible improving move found, allow on other routes.
        // Tabu: tabu_list_2opt[min(a,b)][max(a,b)] for segment endpoints, with aspiration (can override if improves global best_cost).

        // Ensure tabu list for 2-opt is sized (n+1) x (n+1)
        if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

    auto try_two_opt_on_route = [&](const vi& base_route, bool is_truck_mode, int route_idx,
                    Solution &best_local_out, double &best_local_drop,
                    double base_route_time, double current_best_cost,
                    int &best_u, int &best_v, int &best_route_idx, bool &best_is_truck) {
            // Enumerate subroutes separated by depot (0)
            int m = (int)base_route.size();
            if (m <= 3) return; // nothing to reverse (at most one customer)
            int start = 0;
            while (start < m) {
                // find next subroute [l..r] inclusive, where l is first non-depot, r is last before next depot
                while (start < m && base_route[start] == 0) start++;
                if (start >= m) break;
                int l = start;
                int r = l;
                while (r + 1 < m && base_route[r + 1] != 0) r++;
                // subroute is indices [l..r] with all non-zero customers
                int len = r - l + 1;
                if (len >= 2) {
                    for (int i = l; i < r; ++i) {
                        for (int j = i + 1; j <= r; ++j) {
                            int a = base_route[i];
                            int b = base_route[j];
                            if (a == 0 || b == 0) continue; // by construction shouldn't happen
                            // Candidate-list filter: only consider reversing if endpoints are near
                            if (!KNN_ADJ.empty()) {
                                if (!(KNN_ADJ.size() > (size_t)a && KNN_ADJ[a].size() > (size_t)b && KNN_ADJ[a][b]) &&
                                    !(KNN_ADJ.size() > (size_t)b && KNN_ADJ[b].size() > (size_t)a && KNN_ADJ[b][a])) {
                                    continue;
                                }
                            }
                            int u = min(a, b), v = max(a, b);
                            bool is_tabu = (tabu_list_2opt.size() > (size_t)u && tabu_list_2opt[u].size() > (size_t)v &&
                                            tabu_list_2opt[u][v] >= current_iter);

                            vi r2 = base_route;
                            reverse(r2.begin() + i, r2.begin() + j + 1);
                            if (r2 == base_route) continue; // no-op

                            auto [t2, feas] = check_route_feasibility(r2, 0.0, is_truck_mode);
                            if (!feas) continue;

                            // Build neighbor with cached times; only this route's time changes
                            Solution neighbor = initial_solution;
                            if (is_truck_mode) neighbor.truck_routes[route_idx] = r2; else neighbor.drone_routes[route_idx] = r2;
                            neighbor.truck_route_times = initial_solution.truck_route_times;
                            neighbor.drone_route_times = initial_solution.drone_route_times;
                            if (is_truck_mode) neighbor.truck_route_times[route_idx] = (r2.size() > 1 ? t2 : 0.0);
                            else neighbor.drone_route_times[route_idx] = (r2.size() > 1 ? t2 : 0.0);

                            double nb_makespan = 0.0;
                            for (double t : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t);
                            for (double t : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t);
                            neighbor.total_makespan = nb_makespan;

                            // Aspiration: allow tabu if it improves global best_cost
                            if (is_tabu && neighbor.total_makespan >= current_best_cost) continue;

                            double drop = base_route_time - (is_truck_mode ? neighbor.truck_route_times[route_idx]
                                                                            : neighbor.drone_route_times[route_idx]);
                            if (drop > best_local_drop + 1e-12) {
                                best_local_drop = drop;
                                best_local_out = neighbor;
                                best_u = u; best_v = v; best_route_idx = route_idx; best_is_truck = is_truck_mode;
                            } else if (fabs(drop - best_local_drop) <= 1e-12) {
                                // tie-breaker: choose better makespan
                                if (neighbor.total_makespan < best_local_out.total_makespan) {
                                    best_local_out = neighbor;
                                    best_u = u; best_v = v; best_route_idx = route_idx; best_is_truck = is_truck_mode;
                                }
                            }
                        }
                    }
                }
                start = r + 1;
            }
        };

        // Identify critical route (max time)
        bool crit_is_truck = true; int critical_idx = -1; double max_time = -1.0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)initial_solution.truck_route_times.size()) ? initial_solution.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; crit_is_truck = true; critical_idx = i; }
        }
        for (int i = 0; i < (int)initial_solution.drone_route_times.size(); ++i) {
            double t = initial_solution.drone_route_times[i];
            if (t > max_time) { max_time = t; crit_is_truck = false; critical_idx = i; }
        }
        if (critical_idx == -1) return initial_solution;

    Solution best_local = initial_solution; // will hold best 2-opt neighbor according to drop
        double best_local_drop = 0.0; // positive only means improvement
    int best2_u = -1, best2_v = -1, best2_route_idx = -1; bool best2_is_truck = true;

        // Try critical route first
        if (crit_is_truck) {
            double base_t = initial_solution.truck_route_times[critical_idx];
            try_two_opt_on_route(initial_solution.truck_routes[critical_idx], true, critical_idx,
                                 best_local, best_local_drop, base_t, best_cost,
                                 best2_u, best2_v, best2_route_idx, best2_is_truck);
        } else {
            double base_t = initial_solution.drone_route_times[critical_idx];
            try_two_opt_on_route(initial_solution.drone_routes[critical_idx], false, critical_idx,
                                 best_local, best_local_drop, base_t, best_cost,
                                 best2_u, best2_v, best2_route_idx, best2_is_truck);
        }

        if (best_local_drop > 1e-12) {
            // Also update running best if makespan improved
            if (best_local.total_makespan < best_cost) {
                best_neighbor = best_local;
                best_cost = best_local.total_makespan;
            }
            // Update 2-opt tabu for the chosen segment endpoints
            if (best2_u != -1 && best2_v != -1) {
                if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
                    tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
                }
                tabu_list_2opt[best2_u][best2_v] = current_iter + TABU_TENURE_2OPT;
            }
            // Debug
            /* cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N3] 2-opt on " << (best2_is_truck ? "truck" : "drone")
                 << " #" << (best2_is_truck ? (best2_route_idx + 1) : (best2_route_idx + 1))
                 << " cut (" << best2_u << "," << best2_v << ")"
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
            return best_local;
        }

        // If no improving move on critical route, allow other routes and pick the largest drop
        for (int i = 0; i < h; ++i) {
            if (crit_is_truck && i == critical_idx) continue;
            double base_t = initial_solution.truck_route_times[i];
            try_two_opt_on_route(initial_solution.truck_routes[i], true, i,
                                 best_local, best_local_drop, base_t, best_cost,
                                 best2_u, best2_v, best2_route_idx, best2_is_truck);
        }
        for (int i = 0; i < (int)initial_solution.drone_route_times.size(); ++i) {
            if (!crit_is_truck && i == critical_idx) continue;
            double base_t = initial_solution.drone_route_times[i];
            try_two_opt_on_route(initial_solution.drone_routes[i], false, i,
                                 best_local, best_local_drop, base_t, best_cost,
                                 best2_u, best2_v, best2_route_idx, best2_is_truck);
        }

        if (best_local_drop > 1e-12) {
            if (best_local.total_makespan < best_cost) {
                best_neighbor = best_local;
                best_cost = best_local.total_makespan;
            }
            // Update 2-opt tabu for the chosen segment endpoints
            if (best2_u != -1 && best2_v != -1) {
                if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
                    tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
                }
                tabu_list_2opt[best2_u][best2_v] = current_iter + TABU_TENURE_2OPT;
            }
            // Debug
            /* cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N3] 2-opt on " << (best2_is_truck ? "truck" : "drone")
                 << " #" << (best2_is_truck ? (best2_route_idx + 1) : (best2_route_idx + 1))
                 << " cut (" << best2_u << "," << best2_v << ")"
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
            return best_local;
        }

        return initial_solution; // no improving move found

    } else if (neighbor_id == 4) {
        // Neighborhood 4: 2-opt-star (inter-route exchange) within depot-delimited subroutes.
        // Tabu key: unordered pair(s) of edges cut (encoded as (min(u,v), max(u,v)) for each cut edge), with aspiration.

        auto ensure_tabu_star_sized = [&]() {
            if ((int)tabu_list_2opt_star.size() != n + 1 || ((int)tabu_list_2opt_star.size() > 0 && (int)tabu_list_2opt_star[0].size() != n + 1)) {
                tabu_list_2opt_star.assign(n + 1, vector<int>(n + 1, 0));
            }
        };
        ensure_tabu_star_sized();

        // Helper: perform 2-opt* between two routes within chosen subroutes.
    int best_idxA = -1, best_idxB = -1;
    auto try_two_opt_star_pair = [&](const vi &routeA, bool a_is_truck, int idxA,
                     const vi &routeB, bool b_is_truck, int idxB,
                     Solution &best_out, double &best_makespan, double global_best_cost,
                     int &best_ua, int &best_va, int &best_ub, int &best_vb) {
            int mA = (int)routeA.size();
            int mB = (int)routeB.size();
            if (mA <= 3 || mB <= 3) return; // need at least two customers per route to cut within a subroute

            // Enumerate subroutes for A
            int startA = 0;
            while (startA < mA) {
                while (startA < mA && routeA[startA] == 0) startA++;
                if (startA >= mA) break;
                int lA = startA, rA = lA;
                while (rA + 1 < mA && routeA[rA + 1] != 0) rA++;
                // A-subroute is [lA..rA]
                if (rA - lA + 1 >= 2) {
                    // Enumerate subroutes for B
                    int startB = 0;
                    while (startB < mB) {
                        while (startB < mB && routeB[startB] == 0) startB++;
                        if (startB >= mB) break;
                        int lB = startB, rB = lB;
                        while (rB + 1 < mB && routeB[rB + 1] != 0) rB++;
                        if (rB - lB + 1 >= 2) {
                            // Cut inside subroutes: choose i in [lA..rA-1] and j in [lB..rB-1]
                            for (int i = lA; i < rA; ++i) {
                                int a1 = routeA[i], a2 = routeA[i + 1];
                                if (a1 == 0 || a2 == 0) continue; // safety
                                // Edge cut in A: (a1,a2)
                                int ua = min(a1, a2), va = max(a1, a2);
                                for (int j = lB; j < rB; ++j) {
                                    int b1 = routeB[j], b2 = routeB[j + 1];
                                    if (b1 == 0 || b2 == 0) continue; // safety
                                    // Edge cut in B: (b1,b2)
                                    int ub = min(b1, b2), vb = max(b1, b2);

                                    // Candidate-list filter: consider only if cross endpoints look promising
                                    if (!KNN_ADJ.empty()) {
                                        bool ok = false;
                                        auto near = [&](int x, int y){ return KNN_ADJ.size() > (size_t)x && KNN_ADJ[x].size() > (size_t)y && KNN_ADJ[x][y]; };
                                        if (near(a1, b2) || near(b2, a1) || near(b1, a2) || near(a2, b1)) ok = true;
                                        if (!ok) continue;
                                    }

                                    // Tabu check on cut edges (unordered). Aspiration allows override if improving global best.
                                    bool is_tabu = false;
                                    if ((int)tabu_list_2opt_star.size() > ua && (int)tabu_list_2opt_star[ua].size() > va && tabu_list_2opt_star[ua][va] >= current_iter)
                                        is_tabu = true;
                                    if ((int)tabu_list_2opt_star.size() > ub && (int)tabu_list_2opt_star[ub].size() > vb && tabu_list_2opt_star[ub][vb] >= current_iter)
                                        is_tabu = true;

                                    // Build exchanged routes within subroutes: swap tails after i and j up to rA and rB respectively
                                    // Construct new subroutes and rebuild whole routes via concatenation to avoid out-of-bounds writes
                                    vi new_subA; new_subA.reserve((i - lA + 1) + max(0, rB - j));
                                    // A head: [lA..i]
                                    for (int p = lA; p <= i; ++p) new_subA.push_back(routeA[p]);
                                    // append B tail: [j+1..rB]
                                    for (int p = j + 1; p <= rB; ++p) new_subA.push_back(routeB[p]);

                                    vi new_subB; new_subB.reserve((j - lB + 1) + max(0, rA - i));
                                    // B head: [lB..j]
                                    for (int p = lB; p <= j; ++p) new_subB.push_back(routeB[p]);
                                    // append A tail: [i+1..rA]
                                    for (int p = i + 1; p <= rA; ++p) new_subB.push_back(routeA[p]);

                                    // Rebuild rA2 = routeA[0..lA-1] + new_subA + routeA[rA+1..end]
                                    vi rA2; rA2.reserve(lA + (int)new_subA.size() + (mA - (rA + 1)));
                                    rA2.insert(rA2.end(), routeA.begin(), routeA.begin() + lA);
                                    rA2.insert(rA2.end(), new_subA.begin(), new_subA.end());
                                    rA2.insert(rA2.end(), routeA.begin() + rA + 1, routeA.end());

                                    // Rebuild rB2 = routeB[0..lB-1] + new_subB + routeB[rB+1..end]
                                    vi rB2; rB2.reserve(lB + (int)new_subB.size() + (mB - (rB + 1)));
                                    rB2.insert(rB2.end(), routeB.begin(), routeB.begin() + lB);
                                    rB2.insert(rB2.end(), new_subB.begin(), new_subB.end());
                                    rB2.insert(rB2.end(), routeB.begin() + rB + 1, routeB.end());

                                    // Ensure depots exist at ends and no duplicate consecutive depots
                                    auto normalize = [&](vi &r) {
                                        if (r.empty() || r.front() != 0) r.insert(r.begin(), 0);
                                        if (!r.empty() && r.back() != 0) r.push_back(0);
                                        if (!r.empty() && r.size() >= 2) {
                                            vi r2; r2.reserve(r.size());
                                            for (int x : r) { if (!r2.empty() && r2.back() == 0 && x == 0) continue; r2.push_back(x);} r.swap(r2);
                                        }
                                    };
                                    normalize(rA2);
                                    normalize(rB2);

                                    // No-op guard
                                    if (rA2 == routeA && rB2 == routeB) continue;

                                    // Feasibility and times for the two modified routes only
                                    auto [tA2, feasA] = check_route_feasibility(rA2, 0.0, a_is_truck);
                                    if (!feasA) continue;
                                    auto [tB2, feasB] = check_route_feasibility(rB2, 0.0, b_is_truck);
                                    if (!feasB) continue;

                                    Solution neighbor = initial_solution;
                                    if (a_is_truck) neighbor.truck_routes[idxA] = rA2; else neighbor.drone_routes[idxA] = rA2;
                                    if (b_is_truck) neighbor.truck_routes[idxB] = rB2; else neighbor.drone_routes[idxB] = rB2;
                                    neighbor.truck_route_times = initial_solution.truck_route_times;
                                    neighbor.drone_route_times = initial_solution.drone_route_times;
                                    if (a_is_truck) neighbor.truck_route_times[idxA] = (rA2.size() > 1 ? tA2 : 0.0);
                                    else neighbor.drone_route_times[idxA] = (rA2.size() > 1 ? tA2 : 0.0);
                                    if (b_is_truck) neighbor.truck_route_times[idxB] = (rB2.size() > 1 ? tB2 : 0.0);
                                    else neighbor.drone_route_times[idxB] = (rB2.size() > 1 ? tB2 : 0.0);

                                    double nb_makespan = 0.0;
                                    for (double t : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t);
                                    for (double t : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t);
                                    neighbor.total_makespan = nb_makespan;

                                    if (is_tabu && neighbor.total_makespan >= global_best_cost) continue; // aspiration

                                    if (neighbor.total_makespan + 1e-12 < best_makespan) {
                                        best_makespan = neighbor.total_makespan;
                                        best_out = neighbor;
                                        best_ua = ua; best_va = va; best_ub = ub; best_vb = vb;
                                        best_idxA = idxA; best_idxB = idxB;
                                    }
                                }
                            }
                        }
                        startB = rB + 1;
                    }
                }
                startA = rA + 1;
            }
        };

        // Identify the critical route (max time)
        bool crit_is_truck = true; int critical_idx = -1; double max_time = -1.0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)initial_solution.truck_route_times.size()) ? initial_solution.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; crit_is_truck = true; critical_idx = i; }
        }
        for (int i = 0; i < (int)initial_solution.drone_route_times.size(); ++i) {
            double t = initial_solution.drone_route_times[i];
            if (t > max_time) { max_time = t; crit_is_truck = false; critical_idx = i; }
        }
        if (critical_idx == -1) return initial_solution;

    Solution best_local = initial_solution;
    double best_makespan = 1e100;
    int best_ua = -1, best_va = -1, best_ub = -1, best_vb = -1;

        if (crit_is_truck) {
            const vi &critR = initial_solution.truck_routes[critical_idx];
            for (int j = 0; j < h; ++j) {
                if (j == critical_idx) continue;
                try_two_opt_star_pair(critR, true, critical_idx,
                                      initial_solution.truck_routes[j], true, j,
                                      best_local, best_makespan, best_cost,
                                      best_ua, best_va, best_ub, best_vb);
            }
        } else {
            const vi &critR = initial_solution.drone_routes[critical_idx];
            for (int j = 0; j < (int)initial_solution.drone_routes.size(); ++j) {
                if (j == critical_idx) continue;
                try_two_opt_star_pair(critR, false, critical_idx,
                                      initial_solution.drone_routes[j], false, j,
                                      best_local, best_makespan, best_cost,
                                      best_ua, best_va, best_ub, best_vb);
            }
        }

        // If any admissible move was found, return it even if worse than current; also update tabu for cut edges
        if (best_makespan < 1e100) {
            // Update global best tracker if improved
            if (best_local.total_makespan < best_cost) {
                best_neighbor = best_local;
            }
            // Update 2-opt-star tabu (both cut edges) for the chosen move
            ensure_tabu_star_sized();
            if (best_ua != -1 && best_va != -1) tabu_list_2opt_star[best_ua][best_va] = current_iter + TABU_TENURE_2OPT_STAR;
            if (best_ub != -1 && best_vb != -1) tabu_list_2opt_star[best_ub][best_vb] = current_iter + TABU_TENURE_2OPT_STAR;
            // Debug
            /* cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N4] 2-opt* cutA (" << best_ua << "," << best_va << ")"
                 << " cutB (" << best_ub << "," << best_vb << ")"
                 << " on routes " << (best_idxA + 1) << " and " << (best_idxB + 1)
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
            return best_local;
        }

        return initial_solution; // no admissible inter-route exchange found
    }
    return initial_solution; // should not reach here
}

Solution tabu_search(const Solution& initial_solution) {
    auto ts_start = std::chrono::high_resolution_clock::now();
    Solution best_solution = initial_solution;
    double best_cost = initial_solution.total_makespan;
    double score[NUM_NEIGHBORHOODS] = {0.0};
    double weight[NUM_NEIGHBORHOODS];
    for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] = 1.0/NUM_NEIGHBORHOODS;
    int count[NUM_NEIGHBORHOODS] = {0};
    
    Solution current_sol = initial_solution;
    double current_cost = initial_solution.total_makespan;

    for (int segment = 0; segment < CFG_MAX_SEGMENT; ++segment) {
        Solution best_segment_sol = current_sol;
        double best_segment_cost = current_cost;
        int iter = 1;
        int no_improve_iters = 0;
        double T0 = 100.0; // initial temperature for simulated annealing acceptance
        double alpha = 0.995; // cooling rate
    // Reset tabu lists at the start of each segment (iteration counter restarts per segment)
        tabu_list_switch.clear();
        tabu_list_swap.clear();
        tabu_list_reinsert.clear();
        tabu_list_2opt.clear();
        tabu_list_2opt_star.clear();
        tabu_list_relocate.clear();
        while (iter < CFG_MAX_ITER_PER_SEGMENT && no_improve_iters < CFG_MAX_NO_IMPROVE) {
            if (CFG_TIME_LIMIT_SEC > 0.0) {
                double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
                if (elapsed >= CFG_TIME_LIMIT_SEC) break;
            }
            double total_weight = 0.0;
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                total_weight += weight[i];
            }
            double r = ((double) rand() / (RAND_MAX)) * total_weight;
            int selected_neighbor = 0;
            double acc = weight[0];
            while (selected_neighbor < NUM_NEIGHBORHOODS - 1 && r > acc) {
                selected_neighbor++;
                acc += weight[selected_neighbor];
            }
            count[selected_neighbor]++;
            // Use the monotonically increasing iteration counter 'iter' as the tabu iteration for local_search
            Solution neighbor = local_search(current_sol, selected_neighbor, iter, best_cost);
            // Update score
            if (neighbor.total_makespan + 1e-12 < best_segment_cost) {
                best_segment_cost = neighbor.total_makespan;
                best_segment_sol = neighbor;
                no_improve_iters = 0;
            }
            if (neighbor.total_makespan + 1e-12 < best_cost) {
                current_sol = neighbor;
                current_cost = neighbor.total_makespan;
                score[selected_neighbor] += gamma1;
                no_improve_iters = 0;
            } else if (neighbor.total_makespan < current_cost) {
                current_sol = neighbor;
                current_cost = neighbor.total_makespan;
                score[selected_neighbor] += gamma2;
                no_improve_iters = 0;
            } else {
                // Simulated annealing acceptance
                double T = T0 * pow(alpha, iter);
                double ap = exp((current_cost - neighbor.total_makespan) / T);
                double rand_val = ((double) rand() / (RAND_MAX));
                if (rand_val < ap) {
                    current_sol = neighbor;
                    current_cost = neighbor.total_makespan;
                }
                score[selected_neighbor] += gamma3;
                no_improve_iters++;
            }
            // Update global best so aspiration in subsequent iterations uses latest best_cost
            if (neighbor.total_makespan + 1e-12 < best_solution.total_makespan) {
                best_solution = neighbor;
                best_cost = neighbor.total_makespan;
            }
            iter++;
        }
        if (CFG_TIME_LIMIT_SEC > 0.0) {
            double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
            if (elapsed >= CFG_TIME_LIMIT_SEC) break;
        }
        // Update weights based on scores
        for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            if (count[i] != 0) {
                weight[i] = (1.0 - gamma4) * weight[i] + gamma4 * (score[i] / count[i]);
            }
        }
        double segment_best_cost = best_segment_sol.total_makespan;
        if (segment_best_cost + 1e-12 < best_solution.total_makespan) {
            best_solution = best_segment_sol;
            best_cost = best_solution.total_makespan; // keep aspiration threshold in sync with best-so-far
        }
        double sum_weights = 0.0;
        for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            sum_weights += weight[i];
        }
        if (sum_weights > 0.0) {
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                weight[i] /= sum_weights;
            }
        } else {
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                weight[i] = 1.0 / NUM_NEIGHBORHOODS;
            }
        }
        for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            score[i] = 0.0;
            count[i] = 0;
        }
        current_sol = best_segment_sol;
        current_cost = best_segment_cost;

        // Debug: print neighborhood weights after each segment
        //cout.setf(std::ios::fixed); cout << setprecision(6);
        //cout << "[Segment " << (segment + 1) << "] weights:";
        //for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
        //    cout << " w" << i << "=" << weight[i];
        //}
        //cout << "\n";
    }
    return best_solution;
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

// Stream-based printer to avoid duplicating formatting on stdout/file
static void print_solution_stream(const Solution& sol, std::ostream& os) {
    os << "Truck Routes:\n";
    for (int i = 0; i < h; ++i) {
        os << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            os << node << " ";
        }
        os << "\n";
    }
    os << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        os << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            os << node << " ";
        }
        os << "\n";
    }
}

static bool write_output_file(const std::string& out_path, const Solution& sol, double cost, double elapsed_sec, bool final_feasibility) {
    std::ofstream ofs(out_path);
    if (!ofs) return false;
    ofs.setf(std::ios::fixed); ofs << setprecision(6);
    ofs << "Initial solution cost: " << cost << "\n";
    ofs << "Improved solution cost: " << sol.total_makespan << "\n";
    ofs << "Elapsed time: " << elapsed_sec << " seconds\n";
    ofs << "Final solution feasibility: " << (final_feasibility ? "FEASIBLE" : "INFEASIBLE") << "\n";
    ofs << "Solution Details:\n";
    print_solution_stream(sol, ofs);
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0]
             << " input_file [--print-distance-matrix]"
             << " [--attempts=N] [--segments=N] [--iters=N] [--no-improve=N] [--time-limit=SEC] [--auto-tune]"
             << " [--knn-k=K] [--knn-window=W]"
             << "\n";
        return 1;
    }
    string input_file = argv[1];
    bool print_dist_matrix = false;
    bool auto_tune = false;
    // Parse optional flags
    for (int ai = 2; ai < argc; ++ai) {
        string arg = argv[ai];
        if (arg == "--print-distance-matrix") { print_dist_matrix = true; continue; }
        string v;
        if (parse_kv_flag(arg, "--attempts", v)) { CFG_NUM_INITIAL = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--segments", v)) { CFG_MAX_SEGMENT = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--iters", v)) { CFG_MAX_ITER_PER_SEGMENT = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--no-improve", v)) { CFG_MAX_NO_IMPROVE = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--time-limit", v)) { CFG_TIME_LIMIT_SEC = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--knn-k", v)) { CFG_KNN_K = max(0, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--knn-window", v)) { CFG_KNN_WINDOW = max(0, stoi(v)); continue; }
        if (arg == "--auto-tune") { auto_tune = true; continue; }
    }

    // Read input instance
    input(input_file);
    // Build distance matrix for downstream time computations
    compute_distance_matrices(loc);
    if (print_dist_matrix) {
        print_distance_matrix();
        return 0; // only print distance matrix and exit
    }

    //For another data-testing: change all deadline to a constant 3600 and all serving time to 0
    for (int i = 1; i <= n; ++i) {
        deadline[i] = 3600.0;
        serve_truck[i] = 0.0;
        serve_drone[i] = 0.0;
    }

    // Optional auto-tuning based on instance size if requested
    // For now, set auto-tune to always true
    auto_tune = true;
    if (auto_tune) {
        if (n <= 50) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 20);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 20);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 2000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 200);
            CFG_KNN_K = min(CFG_KNN_K, max(5, min(n - 1, 30))); // modest k for small n
        } else if (n <= 200) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 8);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 10);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 2000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 100);
            CFG_KNN_K = min(CFG_KNN_K, max(10, min(n - 1, 25)));
        } else {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 3);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 10);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 350);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 60);
            if (CFG_TIME_LIMIT_SEC <= 0.0) CFG_TIME_LIMIT_SEC = 180.0; // default wall-clock budget
            CFG_KNN_K = min(CFG_KNN_K, max(10, min(n - 1, 20))); // slightly smaller k for very large n
        }
    }

    // Precompute KNN lists (if K is zero, disable by building empty adjacency)
    if (CFG_KNN_K > 0) compute_knn_lists(CFG_KNN_K); else { KNN_LIST.assign(n + 1, {}); KNN_ADJ.assign(n + 1, vector<char>(n + 1, 0)); }

    // Pre-filter dronable customers by capacity/energy
    update_served_by_drone();
    
    // Track best across attempts
    bool have_best = false;
    Solution best_overall_sol;
    double best_overall_initial_cost = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int solution_attempt = 0; solution_attempt < CFG_NUM_INITIAL; ++solution_attempt) {
        Solution initial_solution = generate_initial_solution();
        Solution improved_sol = tabu_search(initial_solution);
        // Output both to stdout and to file
         cout.setf(std::ios::fixed); cout << setprecision(6);
        cout << "Attempt " << (solution_attempt + 1) << "/" << CFG_NUM_INITIAL << "\n";
        cout << "Initial Solution Cost: " << initial_solution.total_makespan << "\n";
        cout << "Improved Solution Cost: " << improved_sol.total_makespan << "\n";
        print_solution_stream(improved_sol, cout);
        // Update best across attempts
        if (!have_best || improved_sol.total_makespan + 1e-12 < best_overall_sol.total_makespan) {
            have_best = true;
            best_overall_sol = improved_sol;
            best_overall_initial_cost = initial_solution.total_makespan;
        }
    }
    // Emit best across all attempts
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    if (have_best) {
        cout << "\n=== Best Across Attempts ===\n";
        cout << "Initial Solution Cost: " << best_overall_initial_cost << "\n";
        cout << "Improved Solution Cost: " << best_overall_sol.total_makespan << "\n";
        cout << "Elapsed Time: " << elapsed_seconds << " seconds\n";
        print_solution_stream(best_overall_sol, cout);
        // check final feasibility
        bool final_feas = true;
        for (const vi &r : best_overall_sol.truck_routes) {
            auto [t, feas] = check_route_feasibility(r, 0.0, true);
            if (!feas) { final_feas = false; break; }
        }
        for (const vi &r : best_overall_sol.drone_routes) {
            auto [t, feas] = check_route_feasibility(r, 0.0, false);
            if (!feas) { final_feas = false; break; }
        }
        if (final_feas) {
            cout << "Final solution feasibility: FEASIBLE\n";
        } else {
            cout << "Final solution feasibility: INFEASIBLE\n";
        }
        string out_best = "output_solution_best.txt";
        if (write_output_file(out_best, best_overall_sol, best_overall_initial_cost, elapsed_seconds, final_feas)) {
            cout << "Best solution written to " << out_best << "\n";
        } else {
            cout << "Failed to write best solution to " << out_best << "\n";
        }
    }

    return 0;
}