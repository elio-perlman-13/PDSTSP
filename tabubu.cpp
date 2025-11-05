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
static int CFG_KNN_K = 1000;           // number of nearest neighbors per customer
static int CFG_KNN_WINDOW = 1;       // insertion window around candidate anchors
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
static int TABU_TENURE_2OPT = 25; // default tenure for 2-opt moves
static map<vector<int>, int> tabu_list_21; // keyed by (a,b,c,d) for (2,1) moves
static int TABU_TENURE_21 = 20; // default tenure for (2,1) moves
static map<vector<int>, int> tabu_list_22;; // keyed by (a,b,c,d) for (2,2) moves
static int TABU_TENURE_22 = 20; // default tenure for (2,2) moves
static map<vector<int>, int> tabu_list_ejection; // keyed by sorted customer sequence
static int TABU_TENURE_EJECTION = 50; // default tenure for ejection chain moves
const int NUM_NEIGHBORHOODS = 8;
const int NUM_OF_INITIAL_SOLUTIONS = 200;
const int MAX_SEGMENT = 200;
const int MAX_NO_IMPROVE = 1000;
const int MAX_ITER_PER_SEGMENT = 1000;
const double gamma1 = 1.0;
const double gamma2 = 0.3;
const double gamma3 = 0.0;
const double gamma4 = 0.3;

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
static int TABU_TENURE_2OPT_STAR = 20; // default tenure for 2-opt-star moves

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
    double best_neighbor_cost = 1e10;
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
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        double best_target_time = 1e10;

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
                    if (target_truck == critical_idx) {
                        //reinsert into the same route:
                        int best_local_pos = -1;
                        double local_min_increase = 1e10;
                        
                        // Lambda to normalize routes for comparison
                        auto normalize_route = [](const vi& r) -> vi {
                            vi normalized;
                            for (int x : r) {
                                if (normalized.empty() || x != 0 || normalized.back() != 0)
                                    normalized.push_back(x);
                            }
                            if (!normalized.empty() && normalized.front() != 0) 
                                normalized.insert(normalized.begin(), 0);
                            if (!normalized.empty() && normalized.back() != 0) 
                                normalized.push_back(0);
                            return normalized;
                        };
                        
                        vi orig_norm = normalize_route(initial_solution.truck_routes[critical_idx]);
                        
                        for (int ins_pos = 1; ins_pos <= (int)nr.size(); ++ins_pos) {
                            vi tr = nr;
                            tr.insert(tr.begin() + ins_pos, cust);
                            if (tr.back() != 0) tr.push_back(0);
                            
                            // Strict no-op detection with normalization
                            vi tr_norm = normalize_route(tr);
                            if (tr_norm == orig_norm) continue; // skip identical routes
                            
                            // Check feasibility for target route only
                            auto [ttgt, feas_tgt] = check_truck_route_feasibility(tr, 0.0);
                            if (!feas_tgt) continue;
                            double ttgt_time = (tr.size() > 1) ? ttgt : 0.0;
                            double local_increase = ttgt_time - initial_solution.truck_route_times[critical_idx];
                            if (local_increase + 1e-8 < local_min_increase){
                                local_min_increase = local_increase;
                                best_local_pos = ins_pos;
                            }
                        }
                        // also consider inserting at the end
                        vi tr = nr;
                        if (tr.empty()) tr.push_back(0);
                        tr.push_back(cust);
                        if (tr.back() != 0) tr.push_back(0);
                        
                        // Check if end insertion produces no-op before updating best position
                        vi tr_norm = normalize_route(tr);
                        if (tr_norm == orig_norm) {
                            // Skip this end insertion - it's a no-op
                        } else {
                            auto [ttgt, feas_tgt] = check_truck_route_feasibility(tr, 0.0);
                            if (feas_tgt) {
                                double ttgt_time = (tr.size() > 1) ? ttgt : 0.0;
                                double local_increase = ttgt_time - initial_solution.truck_route_times[critical_idx];
                                if (local_increase + 1e-8 < local_min_increase){
                                    local_min_increase = local_increase;
                                    best_local_pos = (int)nr.size() + 1;
                                }
                            }
                        }
                        if (best_local_pos < 0) continue; // no feasible insertion found
                        
                        // Rebuild best insertion using correct position on reduced route
                        double ttgt;
                        tr = nr;
                        if (best_local_pos == (int)nr.size() + 1) {
                            if (tr.empty()) tr.push_back(0);
                            tr.push_back(cust);
                            if (tr.back() != 0) tr.push_back(0);
                        } else {
                            tr.insert(tr.begin() + best_local_pos, cust);
                            if (tr.back() != 0) tr.push_back(0);
                        }
                        
                        ttgt = initial_solution.truck_route_times[critical_idx] + local_min_increase;
                        // Construct neighbor incrementally
                        Solution neighbor = initial_solution;
                        neighbor.truck_routes[critical_idx] = tr;
                        // Use cached times and recompute only modified route
                        neighbor.truck_route_times = initial_solution.truck_route_times;
                        neighbor.drone_route_times = initial_solution.drone_route_times;
                        neighbor.truck_route_times[critical_idx] = ttgt;

                        double nb_makespan = 0.0;
                        for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                        for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                        neighbor.total_makespan = nb_makespan;

                        int target_id = target_truck; // unified vehicle id
                        bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                        tabu_list_switch[cust][target_id] >= current_iter);
                        if (is_tabu && neighbor.total_makespan >= best_cost) continue;
                        if (neighbor.total_makespan > best_neighbor_cost_local) continue;
                        // tie-breaker: if same makespan, choose the solution that have lower target vehicle time
                        if (neighbor.total_makespan == best_neighbor_cost_local) {
                            double neighbor_target_time = neighbor.truck_route_times[target_truck];
                            if (neighbor_target_time >= best_target_time) continue;
                            else best_target_time = neighbor_target_time;
                            best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_candidate_neighbor = neighbor;
                        }

                        if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                        if (neighbor.total_makespan < best_neighbor_cost_local) {
                            best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                        }
                    }
                    if (target_truck == critical_idx) continue; // already handled
                    // Insert into different truck route
                    const vi& tgt_r = initial_solution.truck_routes[target_truck];
                    int insert_limit = (int)tgt_r.size();
                    double local_min_increase = 1e10;
                    int best_local_pos = -1;
                    double tgt_r_time = (tgt_r.size() > 1) ? initial_solution.truck_route_times[target_truck] : 0.0;
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
                        // Compute local increase of target truck route
                        double local_increase = ttgt - tgt_r_time;
                        if (local_increase + 1e-8 < local_min_increase){
                            local_min_increase = local_increase;
                            best_local_pos = ins_pos;
                        }
                    }
                    // also consider inserting at the end
                    vi tr = tgt_r;
                    if (tr.empty()) tr.push_back(0);
                    tr.push_back(cust);
                    if (tr.back() != 0) tr.push_back(0);
                    auto [ttgt, feas_tgt] = check_truck_route_feasibility(tr, 0.0);
                    if (feas_tgt) {
                        double ttgt_time = (tr.size() > 1) ? ttgt : 0.0;
                        double local_increase = ttgt_time - initial_solution.truck_route_times[target_truck];
                        if (local_increase + 1e-8 < local_min_increase){
                            local_min_increase = local_increase;
                            best_local_pos = (int)tgt_r.size() + 1;
                        }
                    }
                    if (best_local_pos < 0) continue; // no feasible insertion found
                    int ip = min(max((int)best_local_pos, 1), (int)tr.size());
                    if (best_local_pos == (int)tgt_r.size() + 1) ip = (int)tgt_r.size() + 1;
                    // Rebuild best insertion
                    if (best_local_pos <= int(tgt_r.size())) {
                        tr = tgt_r;
                        tr.insert(tr.begin() + ip, cust);
                        if (tr.back() != 0) tr.push_back(0);
                    } else {
                        tr = tgt_r;
                        if (tr.empty()) tr.push_back(0);
                        tr.push_back(cust);
                        if (tr.back() != 0) tr.push_back(0);
                    }
                    ttgt = initial_solution.truck_route_times[target_truck] + local_min_increase;
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
                    if (neighbor.total_makespan > best_neighbor_cost_local) continue;
                    // tie-breaker: if same makespan, choose the solution that have lower target vehicle time
                    if (neighbor.total_makespan == best_neighbor_cost_local) {
                        double neighbor_target_time = neighbor.truck_route_times[target_truck];
                        if (neighbor_target_time >= best_target_time) continue;
                        else best_target_time = neighbor_target_time;
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_candidate_neighbor = neighbor;
                    }

                    if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                    if (neighbor.total_makespan < best_neighbor_cost_local) {
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                    }
                }
                // Cross-mode: try inserting into drone routes as well
                for (int target_drone = 0; target_drone < (int)initial_solution.drone_routes.size(); ++target_drone) {
                    // Early prune: customer must be dronable
                    if (served_by_drone[cust] == 0) continue;
                    const vi& tgt_r_d = initial_solution.drone_routes[target_drone];
                    int insert_limit_d = (int)tgt_r_d.size();
                    double tgt_r_d_time = (tgt_r_d.size() > 1) ? initial_solution.drone_route_times[target_drone] : 0.0;
                    double local_min_increase = 1e10;
                    int best_local_pos = -1;
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

                        // Compute local increase in makespan
                        double local_increase = ttgt_d - tgt_r_d_time;
                        if (local_increase + 1e-8 < local_min_increase){
                            local_min_increase = local_increase;
                            best_local_pos = ins_pos;
                        }
                    }
                    // also consider inserting at the end
                    vi trd = tgt_r_d;
                    if (trd.empty()) trd.push_back(0);
                    trd.push_back(cust);
                    if (trd.back() != 0) trd.push_back(0);
                    auto [ttgt_d, feas_tgt_d] = check_drone_route_feasibility(trd);
                    if (feas_tgt_d) {
                        double ttgt_time = (trd.size() > 1) ? ttgt_d : 0.0;
                        double local_increase = ttgt_time - initial_solution.drone_route_times[target_drone];
                        if (local_increase + 1e-8 < local_min_increase){
                            local_min_increase = local_increase;
                            best_local_pos = (int)tgt_r_d.size() + 1;
                        }
                    }
                    if (best_local_pos < 0) continue; // no feasible insertion found
                    int ip = min(max((int)best_local_pos, 1), (int)trd.size());
                    if (best_local_pos == (int)tgt_r_d.size() + 1) ip = (int)tgt_r_d.size() + 1;
                    // Rebuild best insertion
                    if (best_local_pos <= int(tgt_r_d.size())) {
                        trd = tgt_r_d;
                        trd.insert(trd.begin() + ip, cust);
                        if (trd.back() != 0) trd.push_back(0);
                    } else {
                        trd = tgt_r_d;
                        if (trd.empty()) trd.push_back(0);
                        trd.push_back(cust);
                        if (trd.back() != 0) trd.push_back(0);
                    }
                    ttgt_d = initial_solution.truck_route_times[target_drone] + local_min_increase;

                    // Construct neighbor incrementally (cross-mode)
                    Solution neighbor = initial_solution;
                    neighbor.truck_routes[critical_idx] = nr;
                    neighbor.drone_routes[target_drone] = trd;

                    // Use cached times and recompute only modified routes
                    neighbor.truck_route_times = initial_solution.truck_route_times;
                    neighbor.drone_route_times = initial_solution.drone_route_times;
                    neighbor.truck_route_times[critical_idx] = crit_time_base;
                    neighbor.drone_route_times[target_drone] = (trd.size() > 1)
                        ? ttgt_d : 0.0;

                    double nb_makespan = 0.0;
                    for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                    for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                    neighbor.total_makespan = nb_makespan;

                    int target_id = h + target_drone; // unified vehicle id for drones
                    bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                    tabu_list_switch[cust][target_id] >= current_iter);
                    if (is_tabu && neighbor.total_makespan >= best_cost) continue;
                    if (neighbor.total_makespan > best_neighbor_cost_local) continue;
                    // tie-breaker: if same makespan, choose the solution that have lower target vehicle time
                    if (neighbor.total_makespan == best_neighbor_cost_local) {
                        double neighbor_target_time = neighbor.drone_route_times[target_drone];
                        if (neighbor_target_time >= best_target_time) continue;
                        else best_target_time = neighbor_target_time;
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_candidate_neighbor = neighbor;
                    }

                    if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                    if (neighbor.total_makespan < best_neighbor_cost_local) {
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
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
                    if (target_drone == critical_idx) {
                        //reinsert into the same route:
                        int best_local_pos = -1;
                        double best_local_increase = 1e10;
                        
                        // Lambda to normalize routes for comparison
                        auto normalize_route = [](const vi& r) -> vi {
                            vi normalized;
                            for (int x : r) {
                                if (normalized.empty() || x != 0 || normalized.back() != 0)
                                    normalized.push_back(x);
                            }
                            if (!normalized.empty() && normalized.front() != 0) 
                                normalized.insert(normalized.begin(), 0);
                            if (!normalized.empty() && normalized.back() != 0) 
                                normalized.push_back(0);
                            return normalized;
                        };
                        
                        vi orig_norm = normalize_route(initial_solution.drone_routes[critical_idx]);
                        
                        for (int ins_pos = 1; ins_pos <= (int)nr.size(); ++ins_pos) {
                            vi tr = nr;
                            tr.insert(tr.begin() + ins_pos, cust);
                            if (tr.back() != 0) tr.push_back(0);
                            
                            // Strict no-op detection with normalization
                            vi tr_norm = normalize_route(tr);
                            if (tr_norm == orig_norm) continue; // skip identical routes
                            
                            // Check feasibility for target route only
                            auto [ttgt, feas_tgt] = check_drone_route_feasibility(tr);
                            if (!feas_tgt) continue;
                            double ttgt_time = (tr.size() > 1) ? ttgt : 0.0;
                            double local_increase = ttgt_time - initial_solution.drone_route_times[critical_idx];
                            if (local_increase + 1e-8 < best_local_increase){
                                best_local_increase = local_increase;
                                best_local_pos = ins_pos;
                            }
                        }
                        // also consider inserting at the end
                        vi tr = nr;
                        if (tr.empty()) tr.push_back(0);
                        tr.push_back(cust);
                        if (tr.back() != 0) tr.push_back(0);
                        
                        // Check if end insertion produces no-op before updating best position
                        vi tr_norm = normalize_route(tr);
                        if (tr_norm == orig_norm) {
                            // Skip this end insertion - it's a no-op
                        } else {
                            auto [ttgt, feas_tgt] = check_drone_route_feasibility(tr);
                            if (feas_tgt) {
                                double ttgt_time = (tr.size() > 1) ? ttgt : 0.0;
                                double local_increase = ttgt_time - initial_solution.drone_route_times[critical_idx];
                                if (local_increase + 1e-8 < best_local_increase){
                                    best_local_increase = local_increase;
                                    best_local_pos = (int)nr.size() + 1;
                                }
                            }
                        }
                        if (best_local_pos < 0) continue; // no feasible insertion found
                        
                        // Rebuild best insertion using correct position on reduced route
                        double ttgt;
                        tr = nr;
                        if (best_local_pos == (int)nr.size() + 1) {
                            if (tr.empty()) tr.push_back(0);
                            tr.push_back(cust);
                            if (tr.back() != 0) tr.push_back(0);
                        } else {
                            tr.insert(tr.begin() + best_local_pos, cust);
                            if (tr.back() != 0) tr.push_back(0);
                        }
                        
                        ttgt = initial_solution.drone_route_times[critical_idx] + best_local_increase;
                        // Construct neighbor incrementally
                        Solution neighbor = initial_solution;
                        neighbor.drone_routes[critical_idx] = tr;
                        // Use cached times and recompute only modified route
                        neighbor.truck_route_times = initial_solution.truck_route_times;
                        neighbor.drone_route_times = initial_solution.drone_route_times;
                        neighbor.drone_route_times[critical_idx] = ttgt;
                        double nb_makespan = 0.0;
                        for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                        for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                        neighbor.total_makespan = nb_makespan;
                        int target_id = h + target_drone; // unified vehicle id
                        bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                        tabu_list_switch[cust][target_id] >= current_iter);
                        if (is_tabu && neighbor.total_makespan >= best_cost) continue;
                        if (neighbor.total_makespan > best_neighbor_cost_local) continue;
                        // tie-breaker: if same makespan, choose the solution that have lower target vehicle time
                        if (neighbor.total_makespan == best_neighbor_cost_local) {
                            double neighbor_target_time = neighbor.drone_route_times[target_drone];
                            if (neighbor_target_time >= best_target_time) continue;
                            else best_target_time = neighbor_target_time;
                            best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_candidate_neighbor = neighbor;
                        }
                        if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                        if (neighbor.total_makespan < best_neighbor_cost_local) {
                            best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                        }     
                    }
                    if (target_drone == critical_idx) continue; // already handled
                    // Insert into different drone route
                    const vi& tgt_r = initial_solution.drone_routes[target_drone];
                    int insert_limit = (int)tgt_r.size();
                    int best_local_pos = -1;
                    double best_local_increase = 1e10;
                    double tgt_r_time = (tgt_r.size() > 1) ? initial_solution.drone_route_times[target_drone] : 0.0;
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
                        // Compute local increase of target drone route
                        double local_increase = ttgt - tgt_r_time;
                        if (local_increase + 1e-8 < best_local_increase){
                            best_local_increase = local_increase;
                            best_local_pos = ins_pos;
                        }
                    }
                    // also consider inserting at the end
                    vi tr = tgt_r;
                    if (tr.empty()) tr.push_back(0);
                    tr.push_back(cust);
                    if (tr.back() != 0) tr.push_back(0);
                    auto [ttgt, feas_tgt] = check_drone_route_feasibility(tr);
                    if (feas_tgt) {
                        double ttgt_time = (tr.size() > 1) ? ttgt : 0.0;
                        double local_increase = ttgt_time - initial_solution.drone_route_times[target_drone];
                        if (local_increase + 1e-8 < best_local_increase){
                            best_local_increase = local_increase;
                            best_local_pos = (int)tgt_r.size() + 1;
                        }
                    }
                    if (best_local_pos < 0) continue; // no feasible insertion found
                    int ip = min(max((int)best_local_pos, 1), (int)tr.size());
                    if (best_local_pos == (int)tgt_r.size() + 1) ip = (int)tgt_r.size() + 1;
                    // Rebuild best insertion
                    if (best_local_pos <= int(tgt_r.size())) {
                        tr = tgt_r;
                        tr.insert(tr.begin() + ip, cust);
                        if (tr.back() != 0) tr.push_back(0);
                    } else {
                        tr = tgt_r;
                        if (tr.empty()) tr.push_back(0);
                        tr.push_back(cust);
                        if (tr.back() != 0) tr.push_back(0);
                    }
                    ttgt = initial_solution.drone_route_times[target_drone] + best_local_increase;

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
                    if (neighbor.total_makespan > best_neighbor_cost_local) continue;
                    // tie-breaker: if same makespan, choose the solution that have lower target vehicle time
                    if (neighbor.total_makespan == best_neighbor_cost_local) {
                        double neighbor_target_time = neighbor.drone_route_times[target_drone];
                        if (neighbor_target_time >= best_target_time) continue;
                        else best_target_time = neighbor_target_time;
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_candidate_neighbor = neighbor;
                    }

                    if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                    if (neighbor.total_makespan < best_neighbor_cost_local) {
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
                    }
                }
                // Cross-mode: try inserting into truck routes as well
                for (int target_truck = 0; target_truck < h; ++target_truck) {
                    const vi& tgt_r_t = initial_solution.truck_routes[target_truck];
                    int insert_limit_t = (int)tgt_r_t.size();
                    double tgt_r_t_time = (tgt_r_t.size() > 1) ? initial_solution.truck_route_times[target_truck] : 0.0;
                    int best_local_pos = -1;
                    double best_local_increase = 1e10;
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

                        // Compute local increase in makespan
                        double local_increase = ttgt_t - tgt_r_t_time;
                        if (local_increase + 1e-8 < best_local_increase){
                            best_local_increase = local_increase;
                            best_local_pos = ins_pos;
                        }
                    }
                    // also consider inserting at the end
                    vi trt = tgt_r_t;
                    if (trt.empty()) trt.push_back(0);
                    trt.push_back(cust);
                    if (trt.back() != 0) trt.push_back(0);
                    auto [ttgt_t, feas_tgt_t] = check_truck_route_feasibility(trt, 0.0);
                    if (feas_tgt_t) {
                        double ttgt_time = (trt.size() > 1) ? ttgt_t : 0.0;
                        double local_increase = ttgt_time - initial_solution.truck_route_times[target_truck];
                        if (local_increase + 1e-8 < best_local_increase){
                            best_local_increase = local_increase;
                            best_local_pos = (int)tgt_r_t.size() + 1;
                        }
                    }
                    if (best_local_pos < 0) continue; // no feasible insertion found
                    int ip = min(max((int)best_local_pos, 1), (int)trt.size());
                    if (best_local_pos == (int)tgt_r_t.size() + 1) ip = (int)tgt_r_t.size() + 1;
                    // Rebuild best insertion
                    if (best_local_pos <= int(tgt_r_t.size())) {
                        trt = tgt_r_t;
                        trt.insert(trt.begin() + ip, cust);
                        if (trt.back() != 0) trt.push_back(0);
                    } else {
                        trt = tgt_r_t;
                        if (trt.empty()) trt.push_back(0);
                        trt.push_back(cust);
                        if (trt.back() != 0) trt.push_back(0);
                    }
                    ttgt_t = initial_solution.truck_route_times[target_truck] + best_local_increase;

                    // Construct neighbor incrementally (cross-mode)
                    Solution neighbor = initial_solution;
                    neighbor.drone_routes[critical_idx] = nr;
                    neighbor.truck_routes[target_truck] = trt;

                    // Use cached times and recompute only modified routes
                    neighbor.truck_route_times = initial_solution.truck_route_times;
                    neighbor.drone_route_times = initial_solution.drone_route_times;
                    neighbor.drone_route_times[critical_idx] = crit_time_base;
                    neighbor.truck_route_times[target_truck] = (trt.size() > 1)
                        ? ttgt_t : 0.0;

                    double nb_makespan = 0.0;
                    for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
                    for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
                    neighbor.total_makespan = nb_makespan;

                    int target_id = target_truck; // unified vehicle id for trucks
                    bool is_tabu = (tabu_list_switch.size() > (size_t)cust && tabu_list_switch[cust].size() > (size_t)target_id &&
                                    tabu_list_switch[cust][target_id] >= current_iter);
                    if (is_tabu && neighbor.total_makespan >= best_cost) continue;
                    if (neighbor.total_makespan > best_neighbor_cost_local) continue;
                    // tie-breaker: if same makespan, choose the solution that have lower target vehicle time
                    if (neighbor.total_makespan == best_neighbor_cost_local) {
                        double neighbor_target_time = neighbor.truck_route_times[target_truck];
                        if (neighbor_target_time >= best_target_time) continue;
                        else best_target_time = neighbor_target_time;
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_candidate_neighbor = neighbor;
                    }

                    if (neighbor.total_makespan < best_cost) { best_cost = neighbor.total_makespan; best_neighbor = neighbor; }
                    if (neighbor.total_makespan < best_neighbor_cost_local) {
                        best_target = target_id; best_pos = best_local_pos; best_cust = cust; best_neighbor_cost_local = neighbor.total_makespan; best_candidate_neighbor = neighbor;
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
            // cout << "[N0] Relocate customer " << best_cust << " at critical vehicle " << critical_idx << " to vehicle " << best_target << " at pos " << best_pos << "; New makespan: " << best_candidate_neighbor.total_makespan << "\n";
            return best_candidate_neighbor;
        }
        else {
            // no feasible relocate found
            return initial_solution;
        }
    } else if (neighbor_id == 1) {
        // Neighborhood 1: swap on the critical vehicle (truck or drone) AND swap from critical vehicle to another vehicle
        // Identify critical vehicle
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
        // Ensure tabu list for reinsert is sized to (n+1) x (h+d)
        {
            int veh_count = h + d;
            if ((int)tabu_list_reinsert.size() != n + 1 || (veh_count > 0 && (int)tabu_list_reinsert[0].size() != veh_count)) {
                tabu_list_reinsert.assign(n + 1, vector<int>(max(0, veh_count), 0));
            }
        }
        int best_swap_u = -1, best_swap_v = -1; // record best swap pair (unordered)
        int best_idx1 = -1, best_idx2 = -1;
        int best_vehicle_other = -1;
        double best_local_cost = 1e10;
        double best_local_cost_other = 1e10;

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Lambda to normalize routes for comparison (detect no-ops)
            auto normalize_route = [](const vi& route) -> vi {
                vi normalized;
                for (int x : route) {
                    if (normalized.empty() || x != 0 || normalized.back() != 0)
                        normalized.push_back(x);
                }
                if (!normalized.empty() && normalized.front() != 0) 
                    normalized.insert(normalized.begin(), 0);
                if (!normalized.empty() && normalized.back() != 0) 
                    normalized.push_back(0);
                return normalized;
            };
            
            vi orig_norm = normalize_route(base_route);
            
            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            if ((int)pos.size() < 2) return; // nothing to swap
            for (int idx1 = 0; idx1 < (int)pos.size(); ++idx1) {
                int p1 = pos[idx1];
                int a = base_route[p1];
                // Consider swapping within the same route
                                                                                                          
                // Consider swapping with other vehicles
                for (int other_veh = 0; other_veh < h + d; ++other_veh) {
                    if (other_veh == critical_vehicle_id) continue; // skip same vehicle
                    const vi& other_route = (other_veh < h) ? initial_solution.truck_routes[other_veh - 0]
                                                            : initial_solution.drone_routes[other_veh - h];
                    if (other_route.size() <= 2) continue; // nothing to swap
                    
                    vi other_norm = normalize_route(other_route);
                    
                    vector<int> pos_other;
                    for (int i = 0; i < (int)other_route.size(); ++i) if (other_route[i] != 0) pos_other.push_back(i);
                    for (int idx2 = 0; idx2 < (int)pos_other.size(); ++idx2) {
                        if (other_route[pos_other[idx2]] == 0) continue;
                        int p2 = pos_other[idx2];
                        int b = other_route[p2];
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
                        // Build swapped routes
                        vi r1 = base_route;
                        vi r2 = other_route;
                        swap(r1[p1], r2[p2]);
                        
                        // No-op detection: check if both routes are identical after normalization
                        vi r1_norm = normalize_route(r1);
                        vi r2_norm = normalize_route(r2);
                        if (r1_norm == orig_norm && r2_norm == other_norm) continue; // Skip no-op swap
                        
                        // Feasibility and time for the modified critical route and other route
                        double tcrit = 0.0; bool feas_crit = false;
                        double tother = 0.0; bool feas_other = false;
                        auto [tt1, ok1] = check_route_feasibility(r1, 0.0, is_truck_mode);
                        tcrit = (r1.size() > 1) ? tt1 : 0.0; feas_crit = ok1;
                        auto [tt2, ok2] = check_route_feasibility(r2, 0.0, (other_veh < h));
                        tother = (r2.size() > 1) ? tt2 : 0.0; feas_other = ok2;
                        if (!feas_crit || !feas_other) continue;
                        if (max(tcrit, tother) >= best_cost && is_tabu) {
                            // no improvement in route time, skip
                            continue;
                        }
                        // tie-breaker: minimize the other vehicle time if same critical time
                        if (max(tcrit, tother) < best_local_cost) {
                            best_local_cost = tcrit;
                            best_local_cost_other = tother;
                            best_swap_u = u; best_swap_v = v;
                            best_idx1 = p1; best_idx2 = p2;
                            best_vehicle_other = other_veh;
                        }
                        if (max(tcrit, tother) == best_local_cost) {
                            if (tother < best_local_cost_other) {
                                best_local_cost_other = tother;
                                best_swap_u = u; best_swap_v = v;
                                best_idx1 = p1; best_idx2 = p2;
                                best_vehicle_other = other_veh;
                            }
                            else {
                                // same cost, skip
                                continue;
                            }
                        }
                    }
                }
            }
            return;
        };


        // Consider swaps for the critical vehicle
        if (crit_is_truck) {
            consider_swap(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_swap(initial_solution.drone_routes[critical_idx], false, h + critical_idx);
        }
        int unified_crit_id = crit_is_truck ? critical_idx : (h + critical_idx);
        // If a best swap is found, apply it
        if (best_swap_u != -1 && best_swap_v != -1) {
            // Apply swap to best_local solution
            // Check if intra-route swap by comparing with the ID that was passed to consider_swap
            bool is_intra_route = (best_vehicle_other == (crit_is_truck ? critical_idx : (h + critical_idx)));
            if (is_intra_route) {
                // swap within the critical vehicle
                vi& route = crit_is_truck ? best_local.truck_routes[critical_idx]
                                          : best_local.drone_routes[critical_idx];
                swap(route[best_idx1], route[best_idx2]);
            } else {
                // swap between critical vehicle and other vehicle
                vi& crit_route = crit_is_truck ? best_local.truck_routes[critical_idx]
                                               : best_local.drone_routes[critical_idx];
                vi& other_route = (best_vehicle_other < h) ? best_local.truck_routes[best_vehicle_other]
                                                           : best_local.drone_routes[best_vehicle_other - h];
                swap(crit_route[best_idx1], other_route[best_idx2]);
            }
            // Update swap tabu for the chosen best move (u,v)
            int u = best_swap_u;
            int v = best_swap_v;
            if (u >= 0 && u <= n && v >= 0 && v <= n) {
                tabu_list_swap[u][v] = current_iter + TABU_TENURE_SWAP;
            }
            // Recompute route times for modified routes
        if (crit_is_truck) {
                auto [tcrit, feas] = check_truck_route_feasibility(best_local.truck_routes[critical_idx], 0.0);
                best_local.truck_route_times[critical_idx] = (best_local.truck_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            } else {
                auto [tcrit, feas] = check_drone_route_feasibility(best_local.drone_routes[critical_idx]);
                best_local.drone_route_times[critical_idx] = (best_local.drone_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            }
        if (best_vehicle_other != unified_crit_id) {
            if (best_vehicle_other < h) {
                auto [tother, feas2] = check_truck_route_feasibility(best_local.truck_routes[best_vehicle_other], 0.0);
                best_local.truck_route_times[best_vehicle_other] = (best_local.truck_routes[best_vehicle_other].size() > 1) ? tother : 0.0;
            } else {
                auto [tother, feas2] = check_drone_route_feasibility(best_local.drone_routes[best_vehicle_other - h]);
                best_local.drone_route_times[best_vehicle_other - h] = (best_local.drone_routes[best_vehicle_other - h].size() > 1) ? tother : 0.0;
            }
        }
        double nb_makespan = 0.0;
        for (double t2 : best_local.truck_route_times) nb_makespan = max(nb_makespan, t2);
        for (double t2 : best_local.drone_route_times) nb_makespan = max(nb_makespan, t2);
        best_local.total_makespan = nb_makespan;

        // Debug: print chosen swap move when available
        /*     cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N1] swap customer " << best_swap_u << " from crit vehicle " << (crit_is_truck ? "truck" : "drone") << " #" << critical_idx
                    << " with customer " << best_swap_v << " from vehicle #" << best_vehicle_other
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n";*/
        return best_local;
        } else {
            // no feasible swap found
            return initial_solution;
        }
    } else if (neighbor_id == 2) {
        // Neighborhood 2: relocate (2,0)-move on the critical vehicle (truck or drone) to another position on the same vehicle or another vehicle
        // Identify critical vehicle
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
        // Size relocate tabu list if needed
        {
            int veh_count = h + d;
            if ((int)tabu_list_relocate.size() != n + 1 || (veh_count > 0 && (int)tabu_list_relocate[0].size() != veh_count)) {
                tabu_list_relocate.assign(n + 1, vector<int>(max(0, veh_count), 0));
            }
        }
        Solution best_local = initial_solution;
        int best_cust1 = -1;
        int best_cust2 = -1;
        int best_crit_pos = -1, best_target_pos = -1;
        int best_target_vehicle = -1;
        double best_local_cost = 1e10;
        double best_local_cost_other = 1e10;

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Collect positions of customers (exclude depots)
            vector<int> pos;
            if (base_route.size() <= 2) return; // nothing to relocate
            for (int i = 0; i < (int)base_route.size() - 1; ++i) if (base_route[i] != 0 && base_route[i+1] != 0) pos.push_back(i);
            if (pos.size() < 1) return; // nothing to relocate
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust1 = base_route[p];
                int cust2 = base_route[p + 1];
                // Try removing pair (cust1, cust2) together
                vi rpair = base_route;
                rpair.erase(rpair.begin() + p);
                rpair.erase(rpair.begin() + p); // after first erase, second is at same position
                // Feasibility and time for the modified critical route
                double tcrit_pair = 0.0; bool feas_pair = false;
                auto [tt_pair, ok_pair] = check_route_feasibility(rpair, 0.0, is_truck_mode);
                tcrit_pair = (rpair.size() > 1) ? tt_pair : 0.0; feas_pair = ok_pair;
                if (!feas_pair) continue;
                // Lambda to normalize routes for comparison (detect no-ops)
                auto normalize_route = [](const vi& route) -> vi {
                    vi normalized;
                    for (int x : route) {
                        if (normalized.empty() || x != 0 || normalized.back() != 0)
                            normalized.push_back(x);
                    }
                    if (!normalized.empty() && normalized.front() != 0) 
                        normalized.insert(normalized.begin(), 0);
                    if (!normalized.empty() && normalized.back() != 0) 
                        normalized.push_back(0);
                    return normalized;
                };
                vi orig_norm = normalize_route(base_route);
                
                // Try relocating pair to all possible positions in the same (reduced) route
                // Use 1-based logical positions where 0 is reserved for depot; valid ip is in [1..r.size()]
                for (int ins_pos = 1; ins_pos <= (int)base_route.size(); ++ins_pos) {
                    if (ins_pos == p || ins_pos == p + 1) continue; // no-op (inserting back to original place)
                    // Build relocated route (route after removal)
                    vi r = rpair;
                    int ip;
                    // Convert original ins_pos (1-based on base_route) to insertion index on rpair
                    if (ins_pos <= p) ip = ins_pos;            // insert before same index
                    else if (ins_pos >= p + 2) ip = ins_pos - 2; // shift left by 2 after removal
                    else continue; // ins_pos == p or p+1 skipped above
                    // clamp into valid insertion range [1..r.size()]
                    ip = min(max(ip, 1), (int)r.size());
                    r.insert(r.begin() + ip, cust1);
                    r.insert(r.begin() + ip + 1, cust2);
                    
                    // No-op detection: check if route content is identical after normalization
                    vi r_norm = normalize_route(r);
                    if (r_norm == orig_norm) continue; // Skip no-op move
                    
                    // Feasibility and time for the modified critical route
                    double tcrit = 0.0; bool feas = false;
                    auto [tt, ok] = check_route_feasibility(r, 0.0, is_truck_mode);
                    tcrit = (r.size() > 1) ? tt : 0.0; feas = ok;
                    if (!feas) continue;
                    // Check relocate tabu
                    int target_id = critical_vehicle_id;
                    bool is_tabu = (tabu_list_relocate.size() > (size_t)cust1 && tabu_list_relocate[cust1].size() > (size_t)target_id &&
                                    tabu_list_relocate[cust1][target_id] >= current_iter);
                    if (tcrit >= best_cost && is_tabu) {
                        // no improvement in route time, skip
                        continue;
                    }
                    else {
                        if (tcrit < best_local_cost) {
                            best_local_cost = tcrit;
                            best_cust1 = cust1;
                            best_cust2 = cust2;
                            best_crit_pos = p;
                            best_target_pos = ip;
                            best_target_vehicle = critical_vehicle_id;
                        }
                    }
                }
                // Also consider relocating pair at the end of the reduced route (before final depot)
                vi r = rpair;
                int ip_end = (int)r.size(); // insert at end (before trailing depot)
                r.insert(r.begin() + ip_end, cust1);
                r.insert(r.begin() + ip_end + 1, cust2);
                r.insert(r.begin() + ip_end + 2, 0); // ensure trailing depot
                
                // No-op detection for end insertion
                vi r_norm = normalize_route(r);
                if (r_norm == orig_norm) continue; // Skip no-op move at end
                
                // Feasibility and time for the modified critical route
                double tcrit = 0.0; bool feas = false;
                auto [tt, ok] = check_route_feasibility(r, 0.0, is_truck_mode);
                tcrit = (r.size() > 1) ? tt : 0.0; feas = ok;
                if (!feas) continue;
                // Check relocate tabu
                int target_id = critical_vehicle_id;
                bool is_tabu = (tabu_list_relocate.size() > (size_t)cust1 && tabu_list_relocate[cust1].size() > (size_t)target_id &&
                                tabu_list_relocate[cust1][target_id] >= current_iter);
                if (tcrit >= best_cost && is_tabu) {
                    // no improvement in route time, skip
                    continue;
                }
                else {
                    if (tcrit < best_local_cost) {
                        best_local_cost = tcrit;
                        best_cust1 = cust1;
                        best_cust2 = cust2;
                        best_crit_pos = p;
                        best_target_pos = (int)r.size() -1; // position after last depot
                        best_target_vehicle = critical_vehicle_id;
                    }
                }
                // Try relocating cust to all other vehicles
                for (int other_veh = 0; other_veh < h + d; ++other_veh) {
                    if (other_veh == critical_vehicle_id) continue; // skip same vehicle
                    const vi& other_route = (other_veh < h) ? initial_solution.truck_routes[other_veh - 0]
                                                            : initial_solution.drone_routes[other_veh - h];
                        for (int ins_pos = 1; ins_pos <= (int)other_route.size() + 1; ++ins_pos) {
                        vi r = other_route;
                        int ip = ins_pos;
                        // ins_pos may be other_route.size()+1 meaning append at end; clamp to [1..r.size()]
                        ip = min(max(ip, 1), (int)r.size());
                        r.insert(r.begin() + ip, cust1);
                        r.insert(r.begin() + ip + 1, cust2);
                        r.insert(r.begin() + ip + 2, 0); // ensure trailing depot
                        // Feasibility and time for the modified other route
                        double tother = 0.0; bool feas_other = false;
                        auto [tt2, ok2] = check_route_feasibility(r, 0.0, (other_veh < h));
                        tother = (r.size() > 1) ? tt2 : 0.0; feas_other = ok2;
                        if (!feas_other) continue;
                        // Check relocate tabu
                        int target_id = other_veh;
                        bool is_tabu = (tabu_list_relocate.size() > (size_t)cust1 && tabu_list_relocate[cust1].size() > (size_t)target_id &&
                                        tabu_list_relocate[cust1][target_id] >= current_iter);
                        if (max(tother, tcrit_pair) >= best_cost && is_tabu) {
                            // no improvement in route time, skip
                            continue;
                        }
                        else {
                            if ((max(tother, tcrit_pair) < best_local_cost) ||
                                (max(tother, tcrit_pair) == best_local_cost && tother < best_local_cost_other)) {
                                best_local_cost = tcrit_pair;
                                best_local_cost_other = tother;
                                best_cust1 = cust1;
                                best_cust2 = cust2;
                                best_crit_pos = p;
                                best_target_pos = ip;
                                best_target_vehicle = other_veh;
                            }
                        }
                    }
                    // Also consider relocating pair at the end of the other route (before final depot)
                    vi r = other_route;
                    int ip_end_other = (int)r.size();
                    r.insert(r.begin() + ip_end_other, cust1);
                    r.insert(r.begin() + ip_end_other + 1, cust2);
                    // Feasibility and time for the modified other route
                    double tother = 0.0; bool feas_other = false;
                    auto [tt2, ok2] = check_route_feasibility(r, 0.0, (other_veh < h));
                    tother = (r.size() > 1) ? tt2 : 0.0; feas_other = ok2;
                    if (!feas_other) continue;
                    // Check relocate tabu
                    int target_id = other_veh;
                    bool is_tabu = (tabu_list_relocate.size() > (size_t)cust1 && tabu_list_relocate[cust1].size() > (size_t)target_id &&
                                    tabu_list_relocate[cust1][target_id] >= current_iter);
                    if (max(tother, tcrit_pair) >= best_cost && is_tabu) {
                        // no improvement in route time, skip
                        continue;
                    }
                    else {
                        if ((max(tother, tcrit_pair) < best_local_cost) ||
                            (max(tother, tcrit_pair) == best_local_cost && tother < best_local_cost_other)) {
                            best_local_cost = tcrit_pair;
                            best_local_cost_other = tother;
                            best_cust1 = cust1;
                            best_cust2 = cust2;
                            best_crit_pos = p;
                            best_target_pos = (int)r.size() -1; // position after last depot
                            best_target_vehicle = other_veh;
                        }
                    }   
                }
            }
        };
        consider_relocate(base_route, crit_is_truck, critical_idx);
        // If a best relocate is found, apply it
        if (best_cust1 != -1 && best_cust2 != -1) {
            // Apply relocate to best_local solution
            vi& crit_route = crit_is_truck ? best_local.truck_routes[critical_idx]
                                           : best_local.drone_routes[critical_idx];

            // small helper to normalize a route: ensure leading/trailing depot and collapse consecutive depots
            auto normalize_route = [&](vi &r) {
                if (r.empty() || r.front() != 0) r.insert(r.begin(), 0);
                if (r.back() != 0) r.push_back(0);
                if (r.size() >= 2) {
                    vi tmp; tmp.reserve(r.size());
                    for (int x : r) {
                        if (!tmp.empty() && tmp.back() == 0 && x == 0) continue;
                        tmp.push_back(x);
                    }
                    r.swap(tmp);
                }
            };

            // Remove pair from critical route (erase twice at same index)
            if (best_crit_pos >= 0 && best_crit_pos + 1 < (int)crit_route.size()) {
                crit_route.erase(crit_route.begin() + best_crit_pos);
                crit_route.erase(crit_route.begin() + best_crit_pos); // after first erase, second is at same position
            } else {
                // Safety: if indices invalid, abort
                return initial_solution;
            }

            // Insert pair into target vehicle route
            if (best_target_vehicle == critical_idx) {
                // same vehicle
                int ip = best_target_pos;
                // If the stored target position was after the removed pair, shift left by 2 to account for removal
                if (best_target_pos > best_crit_pos) ip = best_target_pos - 2;
                // clamp to valid insertion positions [1..crit_route.size()]
                ip = min(max(ip, 1), (int)crit_route.size());
                crit_route.insert(crit_route.begin() + ip, best_cust1);
                crit_route.insert(crit_route.begin() + ip + 1, best_cust2);
            } else {
                vi& target_route = (best_target_vehicle < h) ? best_local.truck_routes[best_target_vehicle]
                                                            : best_local.drone_routes[best_target_vehicle - h];
                int ip = best_target_pos;
                // clamp to valid insertion positions [1..target_route.size()]
                ip = min(max(ip, 1), (int)target_route.size());
                target_route.insert(target_route.begin() + ip, best_cust1);
                target_route.insert(target_route.begin() + ip + 1, best_cust2);
            }

            // Normalize both modified routes to ensure correct depot placements and no consecutive zeros
            normalize_route(crit_route);
            if (best_target_vehicle != critical_idx) {
                vi &target_route = (best_target_vehicle < h) ? best_local.truck_routes[best_target_vehicle]
                                                            : best_local.drone_routes[best_target_vehicle - h];
                normalize_route(target_route);
            }

            // Update relocate tabu for the chosen best move (cust1 -> target vehicle)
            int target_id = best_target_vehicle;
            if (best_cust1 >= 0 && best_cust1 <= n && target_id >= 0 && target_id < h + d) {
                tabu_list_relocate[best_cust1][target_id] = current_iter + TABU_TENURE_RELOCATE;
                // Also mark the second customer of the moved pair to avoid immediate repeated moves of the same block
                if (best_cust2 >= 0 && best_cust2 <= n) {
                    tabu_list_relocate[best_cust2][target_id] = current_iter + TABU_TENURE_RELOCATE;
                }
            }

            // Recompute feasibility and route times for modified routes; if infeasible, abort and return original
            if (crit_is_truck) {
                auto [tcrit, feas] = check_truck_route_feasibility(best_local.truck_routes[critical_idx], 0.0);
                if (!feas) return initial_solution;
                best_local.truck_route_times[critical_idx] = (best_local.truck_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            } else {
                auto [tcrit, feas] = check_drone_route_feasibility(best_local.drone_routes[critical_idx]);
                if (!feas) return initial_solution;
                best_local.drone_route_times[critical_idx] = (best_local.drone_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            }

            if (best_target_vehicle != critical_idx) {
                if (best_target_vehicle < h) {
                    auto [tother, feas2] = check_truck_route_feasibility(best_local.truck_routes[best_target_vehicle], 0.0);
                    if (!feas2) return initial_solution;
                    best_local.truck_route_times[best_target_vehicle] = (best_local.truck_routes[best_target_vehicle].size() > 1) ? tother : 0.0;
                } else {
                    auto [tother, feas2] = check_drone_route_feasibility(best_local.drone_routes[best_target_vehicle - h]);
                    if (!feas2) return initial_solution;
                    best_local.drone_route_times[best_target_vehicle - h] = (best_local.drone_routes[best_target_vehicle - h].size() > 1) ? tother : 0.0;
                }
            }

            double nb_makespan = 0.0;
            for (double t2 : best_local.truck_route_times) nb_makespan = max(nb_makespan, t2);
            for (double t2 : best_local.drone_route_times) nb_makespan = max(nb_makespan, t2);
            best_local.total_makespan = nb_makespan;
            // Debug: print chosen relocate move when available
             /* cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N2] relocate customers (" << best_cust1 << "," << best_cust2 << ") from crit vehicle " << (crit_is_truck ? "truck" : "drone") << " #"
                 << critical_idx << " to vehicle #" << best_target_vehicle
                 << " at position " << best_target_pos
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
            return best_local;
        } else {
            // no feasible relocate found
            return initial_solution;
        }

    } else if (neighbor_id == 3) {
        // Neighborhood 3: 2-opt within each subroute (between depot nodes) for trucks or drones.
        // Finds the best 2-opt move across all routes that yields the largest local time drop.

        if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

        double best_overall_drop = 0.0;
        int best_i = -1, best_j = -1;
        int best_u = -1, best_v = -1;
        int best_route_idx = -1;
        bool best_is_truck = false;
        double best_makespan_for_best_drop = 1e100;

        auto find_best_2opt_on_route = [&](const vi& base_route, bool is_truck_mode, int route_idx, double base_route_time) {
            int m = (int)base_route.size();
            if (m <= 3) return;

            int start = 0;
            while (start < m) {
                while (start < m && base_route[start] == 0) start++;
                if (start >= m) break;
                int l = start;
                int r_end = l;
                while (r_end + 1 < m && base_route[r_end + 1] != 0) r_end++;

                if (r_end - l + 1 >= 2) {
                    for (int i = l; i < r_end; ++i) {
                        for (int j = i + 1; j <= r_end; ++j) {
                            vi r2 = base_route;
                            reverse(r2.begin() + i, r2.begin() + j + 1);

                            auto [t2, feas] = check_route_feasibility(r2, 0.0, is_truck_mode);
                            if (!feas) continue;

                            double current_makespan = t2;
                            for(int k=0; k<h; ++k) if(!is_truck_mode || k != route_idx) current_makespan = max(current_makespan, initial_solution.truck_route_times[k]);
                            for(int k=0; k<(int)initial_solution.drone_routes.size(); ++k) if(is_truck_mode || k != route_idx) current_makespan = max(current_makespan, initial_solution.drone_route_times[k]);

                            int u_cand = min(base_route[i], base_route[j+1]);
                            int v_cand = max(base_route[i], base_route[j+1]);
                            bool is_tabu = (tabu_list_2opt.size() > (size_t)u_cand && tabu_list_2opt[u_cand].size() > (size_t)v_cand && tabu_list_2opt[u_cand][v_cand] >= current_iter);

                            if (is_tabu && current_makespan >= best_cost) continue;

                            double drop = base_route_time - t2;
                            if (drop > best_overall_drop + 1e-12) {
                                best_overall_drop = drop;
                                best_makespan_for_best_drop = current_makespan;
                                best_i = i; best_j = j;
                                best_u = u_cand; best_v = v_cand;
                                best_route_idx = route_idx; best_is_truck = is_truck_mode;
                            } else if (fabs(drop - best_overall_drop) <= 1e-12) {
                                if (current_makespan < best_makespan_for_best_drop) {
                                    best_makespan_for_best_drop = current_makespan;
                                    best_i = i; best_j = j;
                                    best_u = u_cand; best_v = v_cand;
                                    best_route_idx = route_idx; best_is_truck = is_truck_mode;
                                }
                            }
                        }
                    }
                }
                start = r_end + 1;
            }
        };

        // Evaluate all routes to find the single best 2-opt move
        for (int i = 0; i < h; ++i) {
            find_best_2opt_on_route(initial_solution.truck_routes[i], true, i, initial_solution.truck_route_times[i]);
        }
        for (int i = 0; i < (int)initial_solution.drone_routes.size(); ++i) {
            find_best_2opt_on_route(initial_solution.drone_routes[i], false, i, initial_solution.drone_route_times[i]);
        }

        // If an improving move was found, construct the solution and return it
        if (best_overall_drop > 1e-12) {
            Solution neighbor = initial_solution;
            vi final_route = best_is_truck ? initial_solution.truck_routes[best_route_idx] : initial_solution.drone_routes[best_route_idx];
            
            reverse(final_route.begin() + best_i, final_route.begin() + best_j + 1);
            auto [t_final, feas_final] = check_route_feasibility(final_route, 0.0, best_is_truck);

            if (best_is_truck) {
                neighbor.truck_routes[best_route_idx] = final_route;
                neighbor.truck_route_times[best_route_idx] = t_final;
            } else {
                neighbor.drone_routes[best_route_idx] = final_route;
                neighbor.drone_route_times[best_route_idx] = t_final;
            }
            
            double nb_makespan = 0.0;
            for (double t2 : neighbor.truck_route_times) nb_makespan = max(nb_makespan, t2);
            for (double t2 : neighbor.drone_route_times) nb_makespan = max(nb_makespan, t2);
            neighbor.total_makespan = nb_makespan;

            tabu_list_2opt[best_u][best_v] = current_iter + TABU_TENURE_2OPT;
            // Debug: print chosen 2-opt move
            /*   if (best_u != -1 && best_v != -1) {
                cout.setf(std::ios::fixed); cout << setprecision(6);
                cout << "[N3] 2-opt on " << (best_is_truck ? "truck" : "drone") << " #" << (best_route_idx + 1)
                     << " between positions " << best_i << " and " << best_j
                     << ", makespan: " << initial_solution.total_makespan << " -> " << neighbor.total_makespan
                     << ", iter " << current_iter << "\n";
            }  */

            return neighbor;
        }
        else {
            // No improving 2-opt found
            return initial_solution;
        }

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
            /*  cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N4] 2-opt* cutA (" << best_ua << "," << best_va << ")"
                 << " cutB (" << best_ub << "," << best_vb << ")"
                 << " on routes " << (best_idxA + 1) << " and " << (best_idxB + 1)
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
            return best_local;
        }

        return initial_solution; // no admissible inter-route exchange found
    } else if (neighbor_id == 5) {
        // Neighborhood 5: (2, 1) move (asymmetric swap) between two routes.
        // Select the best (2,1) swap move across all route pairs that yields the largest local time drop.
         // Neighborhood 1: swap on the critical vehicle (truck or drone) AND swap from critical vehicle to another vehicle
        // Identify critical vehicle
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
        Solution best_local = initial_solution;
        // Ensure tabu list for reinsert is sized to (n+1) x (h+d)
        {
            int veh_count = h + d;
            if ((int)tabu_list_reinsert.size() != n + 1 || (veh_count > 0 && (int)tabu_list_reinsert[0].size() != veh_count)) {
                tabu_list_reinsert.assign(n + 1, vector<int>(max(0, veh_count), 0));
            }
        }
        vector<int> best_tabu_key; // record best (2,1) move as 3-customer set
        int best_idx1 = -1, best_idx2 = -1;
        int best_vehicle_other = -1;
        double best_local_cost = 1e10;
        double best_local_cost_other = 1e10;

        auto consider_swap_21 = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Collect positions of consecutive pairs (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size() - 1; ++i) {
                if (base_route[i] != 0 && base_route[i+1] != 0) pos.push_back(i);
            }
            if ((int)pos.size() < 1) return; // need at least one pair
            
            for (int idx1 = 0; idx1 < (int)pos.size(); ++idx1) {
                int p1 = pos[idx1];
                int a = base_route[p1];
                int a_next = base_route[p1 + 1];
                
                // Consider swapping pair from critical route with single from other vehicles
                for (int other_veh = 0; other_veh < h + d; ++other_veh) {
                    if (other_veh == critical_vehicle_id) continue; // skip same vehicle
                    const vi& other_route = (other_veh < h) ? initial_solution.truck_routes[other_veh]
                                                            : initial_solution.drone_routes[other_veh - h];
                    if (other_route.size() <= 2) continue; // need at least one customer
                    
                    vector<int> pos_other;
                    for (int i = 0; i < (int)other_route.size(); ++i) {
                        if (other_route[i] != 0) pos_other.push_back(i);
                    }
                    
                    for (int idx2 = 0; idx2 < (int)pos_other.size(); ++idx2) {
                        int p2 = pos_other[idx2];
                        int b = other_route[p2];
                        if (a == b || a_next == b) continue;
                        
                        // Candidate-list filter: check if any of pair are near single
                        if (!KNN_ADJ.empty()) {
                            bool ok = false;
                            auto near = [&](int x, int y){ 
                                return KNN_ADJ.size() > (size_t)x && KNN_ADJ[x].size() > (size_t)y && KNN_ADJ[x][y]; 
                            };
                            if (near(a, b) || near(b, a) || near(a_next, b) || near(b, a_next)) ok = true;
                            if (!ok) continue;
                        }
                        
                        // Build tabu key: sorted set of 3 customers involved
                        vector<int> tabu_key = {a, a_next, b};
                        sort(tabu_key.begin(), tabu_key.end());
                        bool is_tabu = (tabu_list_21.count(tabu_key) > 0 && tabu_list_21[tabu_key] >= current_iter);
                        
                        // Build swapped routes:
                        // Critical route: remove pair (a, a_next), insert single (b)
                        vi r1 = base_route;
                        r1.erase(r1.begin() + p1);        // remove first of pair
                        r1.erase(r1.begin() + p1);        // remove second (now at same position)
                        r1.insert(r1.begin() + p1, b);    // insert single customer
                        
                        // Other route: remove single (b), insert pair (a, a_next)
                        vi r2 = other_route;
                        r2.erase(r2.begin() + p2);           // remove single
                        r2.insert(r2.begin() + p2, a);       // insert first of pair
                        r2.insert(r2.begin() + p2 + 1, a_next); // insert second of pair
                        
                        // Feasibility and time for both modified routes
                        auto [tt1, ok1] = check_route_feasibility(r1, 0.0, is_truck_mode);
                        auto [tt2, ok2] = check_route_feasibility(r2, 0.0, (other_veh < h));
                        
                        if (!ok1 || !ok2) continue;
                        
                        double tcrit = (r1.size() > 1) ? tt1 : 0.0;
                        double tother = (r2.size() > 1) ? tt2 : 0.0;
                        
                        if (max(tcrit, tother) >= best_cost && is_tabu) {
                            continue; // tabu and not improving
                        }
                        
                        // Track best move
                        if (max(tcrit, tother) < best_local_cost) {
                            best_local_cost = max(tcrit, tother);
                            best_local_cost_other = tother;
                            best_tabu_key = tabu_key; 
                            best_idx1 = p1; 
                            best_idx2 = p2;
                            best_vehicle_other = other_veh;
                        } else if (max(tcrit, tother) == best_local_cost && tother < best_local_cost_other) {
                            best_local_cost_other = tother;
                            best_tabu_key = tabu_key; 
                            best_idx1 = p1; 
                            best_idx2 = p2;
                            best_vehicle_other = other_veh;
                        }
                    }
                }
            }
        };


        // Consider (2,1) swaps for the critical vehicle
        if (crit_is_truck) {
            consider_swap_21(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_swap_21(initial_solution.drone_routes[critical_idx], false, h + critical_idx);
        }
        int unified_crit_id = crit_is_truck ? critical_idx : (h + critical_idx);
        // If a best (2,1) swap is found, apply it
        if (!best_tabu_key.empty() && best_idx1 != -1 && best_idx2 != -1) {
            // Get the customers involved from best_idx positions
            const vi& crit_route = crit_is_truck ? initial_solution.truck_routes[critical_idx]
                                                 : initial_solution.drone_routes[critical_idx];
            const vi& other_route = (best_vehicle_other < h) ? initial_solution.truck_routes[best_vehicle_other]
                                                              : initial_solution.drone_routes[best_vehicle_other - h];
            
            int a = crit_route[best_idx1];
            int a_next = crit_route[best_idx1 + 1];
            int b = other_route[best_idx2];
            
            // Apply (2,1) swap to best_local solution
            vi& crit_route_mut = crit_is_truck ? best_local.truck_routes[critical_idx]
                                                : best_local.drone_routes[critical_idx];
            vi& other_route_mut = (best_vehicle_other < h) ? best_local.truck_routes[best_vehicle_other]
                                                             : best_local.drone_routes[best_vehicle_other - h];
            
            // Critical route: remove pair, insert single
            crit_route_mut.erase(crit_route_mut.begin() + best_idx1);
            crit_route_mut.erase(crit_route_mut.begin() + best_idx1);
            crit_route_mut.insert(crit_route_mut.begin() + best_idx1, b);
            
            // Other route: remove single, insert pair
            other_route_mut.erase(other_route_mut.begin() + best_idx2);
            other_route_mut.insert(other_route_mut.begin() + best_idx2, a);
            other_route_mut.insert(other_route_mut.begin() + best_idx2 + 1, a_next);
            
            // Update tabu list for this (2,1) move with the 3-customer set
            tabu_list_21[best_tabu_key] = current_iter + TABU_TENURE_21;
            // Recompute route times for modified routes
        if (crit_is_truck) {
                auto [tcrit, feas] = check_truck_route_feasibility(best_local.truck_routes[critical_idx], 0.0);
                best_local.truck_route_times[critical_idx] = (best_local.truck_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            } else {
                auto [tcrit, feas] = check_drone_route_feasibility(best_local.drone_routes[critical_idx]);
                best_local.drone_route_times[critical_idx] = (best_local.drone_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            }
        if (best_vehicle_other != unified_crit_id) {
            if (best_vehicle_other < h) {
                auto [tother, feas2] = check_truck_route_feasibility(best_local.truck_routes[best_vehicle_other], 0.0);
                best_local.truck_route_times[best_vehicle_other] = (best_local.truck_routes[best_vehicle_other].size() > 1) ? tother : 0.0;
            } else {
                auto [tother, feas2] = check_drone_route_feasibility(best_local.drone_routes[best_vehicle_other - h]);
                best_local.drone_route_times[best_vehicle_other - h] = (best_local.drone_routes[best_vehicle_other - h].size() > 1) ? tother : 0.0;
            }
        }
        double nb_makespan = 0.0;
        for (double t2 : best_local.truck_route_times) nb_makespan = max(nb_makespan, t2);
        for (double t2 : best_local.drone_route_times) nb_makespan = max(nb_makespan, t2);
        best_local.total_makespan = nb_makespan;

        // Debug: print chosen swap move when available
        /*      cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N5] (2,1) swap: pair (" << best_tabu_key[0] << "," << best_tabu_key[1] << ") with customer " << best_tabu_key[2]
                 << " from crit vehicle " << (crit_is_truck ? "truck" : "drone") << " #" << critical_idx
                 << " to vehicle #" << best_vehicle_other
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
        return best_local;
        } else {
            // no feasible swap found
            return initial_solution;
        }
    } else if (neighbor_id == 6){
        // Neighborhood 6: (2,2) move (symmetric swap) between two routes.
        // Similar to (2,1) move but swap pairs between two routes.
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

        Solution best_local = initial_solution;
        // Ensure tabu list for reinsert is sized to (n+1) x (h+d)
        {
            int veh_count = h + d;
            if ((int)tabu_list_reinsert.size() != n + 1 || (veh_count > 0 && (int)tabu_list_reinsert[0].size() != veh_count)) {
                tabu_list_reinsert.assign(n + 1, vector<int>(max(0, veh_count), 0));
            }
        }
        vector<int> best_tabu_key; // record best (2,2) move as 4-customer set
        int best_idx1 = -1, best_idx2 = -1;
        int best_vehicle_other = -1;
        double best_local_cost = 1e10;
        double best_local_cost_other = 1e10;

        auto consider_swap_22 = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Collect positions of consecutive pairs (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size() - 1; ++i) {
                if (base_route[i] != 0 && base_route[i+1] != 0) pos.push_back(i);
            }
            if ((int)pos.size() < 1) return; // need at least one pair
            
            for (int idx1 = 0; idx1 < (int)pos.size(); ++idx1) {
                int p1 = pos[idx1];
                int a = base_route[p1];
                int a_next = base_route[p1 + 1];
                
                // Consider swapping pair from critical route with single from other vehicles
                for (int other_veh = 0; other_veh < h + d; ++other_veh) {
                    if (other_veh == critical_vehicle_id) continue; // skip same vehicle
                    const vi& other_route = (other_veh < h) ? initial_solution.truck_routes[other_veh]
                                                            : initial_solution.drone_routes[other_veh - h];
                    if (other_route.size() <= 2) continue; // need at least one customer
                    
                    vector<int> pos_other;
                    for (int i = 0; i < (int)other_route.size() - 1; ++i) {
                        if (other_route[i] != 0 && other_route[i + 1] != 0) pos_other.push_back(i);
                    }
                    
                    for (int idx2 = 0; idx2 < (int)pos_other.size(); ++idx2) {
                        int p2 = pos_other[idx2];
                        int b = other_route[p2];
                        int b_next = other_route[p2 + 1];
                        if (a == b || a_next == b || a == b_next || a_next == b_next) continue;
                        
                        // Candidate-list filter: check if any of pair are near single
                        if (!KNN_ADJ.empty()) {
                            bool ok = false;
                            auto near = [&](int x, int y){ 
                                return KNN_ADJ.size() > (size_t)x && KNN_ADJ[x].size() > (size_t)y && KNN_ADJ[x][y]; 
                            };
                            if (near(a, b) || near(a_next, b) || near(a_next, b_next) || near(a, b_next)) ok = true;
                            if (!ok) continue;
                        }
                        
                        // Build tabu key: sorted set of 4 customers involved
                        vector<int> tabu_key = {a, a_next, b, b_next};
                        sort(tabu_key.begin(), tabu_key.end());
                        bool is_tabu = (tabu_list_22.count(tabu_key) > 0 && tabu_list_22[tabu_key] >= current_iter);
                        
                        // Build swapped routes:
                        // Critical route: remove pair (a, a_next), insert pair (b, b_next)
                        vi r1 = base_route;
                        r1.erase(r1.begin() + p1);        // remove first of pair (a)
                        r1.erase(r1.begin() + p1);        // remove second (a_next, now at same position)
                        r1.insert(r1.begin() + p1, b);    // insert first of other pair
                        r1.insert(r1.begin() + p1 + 1, b_next); // insert second of other pair
                        
                        // Other route: remove pair (b, b_next), insert pair (a, a_next)
                        vi r2 = other_route;
                        r2.erase(r2.begin() + p2);           // remove first of pair (b)
                        r2.erase(r2.begin() + p2);           // remove second (b_next, now at same position)
                        r2.insert(r2.begin() + p2, a);       // insert first of other pair
                        r2.insert(r2.begin() + p2 + 1, a_next); // insert second of other pair
                        
                        // Feasibility and time for both modified routes
                        auto [tt1, ok1] = check_route_feasibility(r1, 0.0, is_truck_mode);
                        auto [tt2, ok2] = check_route_feasibility(r2, 0.0, (other_veh < h));
                        
                        if (!ok1 || !ok2) continue;
                        
                        double tcrit = (r1.size() > 1) ? tt1 : 0.0;
                        double tother = (r2.size() > 1) ? tt2 : 0.0;
                        
                        if (max(tcrit, tother) >= best_cost && is_tabu) {
                            continue; // tabu and not improving
                        }
                        
                        // Track best move
                        if (max(tcrit, tother) < best_local_cost) {
                            best_local_cost = max(tcrit, tother);
                            best_local_cost_other = tother;
                            best_tabu_key = tabu_key; 
                            best_idx1 = p1; 
                            best_idx2 = p2;
                            best_vehicle_other = other_veh;
                        } else if (max(tcrit, tother) == best_local_cost && tother < best_local_cost_other) {
                            best_local_cost_other = tother;
                            best_tabu_key = tabu_key; 
                            best_idx1 = p1; 
                            best_idx2 = p2;
                            best_vehicle_other = other_veh;
                        }
                    }
                }
            }
        };


        // Consider (2,2) swaps for the critical vehicle
        if (crit_is_truck) {
            consider_swap_22(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_swap_22(initial_solution.drone_routes[critical_idx], false, h + critical_idx);
        }
        int unified_crit_id = crit_is_truck ? critical_idx : (h + critical_idx);
        // If a best (2,2) swap is found, apply it
        if (!best_tabu_key.empty() && best_idx1 != -1 && best_idx2 != -1) {
            // Get the customers involved from best_idx positions
            const vi& crit_route = crit_is_truck ? initial_solution.truck_routes[critical_idx]
                                                 : initial_solution.drone_routes[critical_idx];
            const vi& other_route = (best_vehicle_other < h) ? initial_solution.truck_routes[best_vehicle_other]
                                                              : initial_solution.drone_routes[best_vehicle_other - h];
            
            int a = crit_route[best_idx1];
            int a_next = crit_route[best_idx1 + 1];
            int b = other_route[best_idx2];
            int b_next = other_route[best_idx2 + 1];
            
            // Apply (2,2) swap to best_local solution
            vi& crit_route_mut = crit_is_truck ? best_local.truck_routes[critical_idx]
                                                : best_local.drone_routes[critical_idx];
            vi& other_route_mut = (best_vehicle_other < h) ? best_local.truck_routes[best_vehicle_other]
                                                             : best_local.drone_routes[best_vehicle_other - h];
            
            // Critical route: remove pair (a, a_next), insert pair (b, b_next)
            crit_route_mut.erase(crit_route_mut.begin() + best_idx1);
            crit_route_mut.erase(crit_route_mut.begin() + best_idx1);
            crit_route_mut.insert(crit_route_mut.begin() + best_idx1, b);
            crit_route_mut.insert(crit_route_mut.begin() + best_idx1 + 1, b_next);
            
            // Other route: remove pair (b, b_next), insert pair (a, a_next)
            other_route_mut.erase(other_route_mut.begin() + best_idx2);
            other_route_mut.erase(other_route_mut.begin() + best_idx2);
            other_route_mut.insert(other_route_mut.begin() + best_idx2, a);
            other_route_mut.insert(other_route_mut.begin() + best_idx2 + 1, a_next);
            
            // Update tabu list for this (2,2) move with the 4-customer set
            tabu_list_22[best_tabu_key] = current_iter + TABU_TENURE_22;
            // Recompute route times for modified routes
        if (crit_is_truck) {
                auto [tcrit, feas] = check_truck_route_feasibility(best_local.truck_routes[critical_idx], 0.0);
                best_local.truck_route_times[critical_idx] = (best_local.truck_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            } else {
                auto [tcrit, feas] = check_drone_route_feasibility(best_local.drone_routes[critical_idx]);
                best_local.drone_route_times[critical_idx] = (best_local.drone_routes[critical_idx].size() > 1) ? tcrit : 0.0;
            }
        if (best_vehicle_other != unified_crit_id) {
            if (best_vehicle_other < h) {
                auto [tother, feas2] = check_truck_route_feasibility(best_local.truck_routes[best_vehicle_other], 0.0);
                best_local.truck_route_times[best_vehicle_other] = (best_local.truck_routes[best_vehicle_other].size() > 1) ? tother : 0.0;
            } else {
                auto [tother, feas2] = check_drone_route_feasibility(best_local.drone_routes[best_vehicle_other - h]);
                best_local.drone_route_times[best_vehicle_other - h] = (best_local.drone_routes[best_vehicle_other - h].size() > 1) ? tother : 0.0;
            }
        }
        double nb_makespan = 0.0;
        for (double t2 : best_local.truck_route_times) nb_makespan = max(nb_makespan, t2);
        for (double t2 : best_local.drone_route_times) nb_makespan = max(nb_makespan, t2);
        best_local.total_makespan = nb_makespan;

        // Debug: print chosen swap move when available
        /*      cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N6] (2,2) swap: pair (" << best_tabu_key[0] << "," << best_tabu_key[1] << ") with pair (" << best_tabu_key[2] << "," << best_tabu_key[3] << ")"
                 << " from crit vehicle " << (crit_is_truck ? "truck" : "drone") << " #" << critical_idx
                 << " to vehicle #" << best_vehicle_other
                 << ", makespan: " << initial_solution.total_makespan << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n"; */
        return best_local;
        } else {
            // no feasible swap found
            return initial_solution;
        }
    } else if (neighbor_id == 7) {
        // Neighborhood 7: Ejection chain (depth-2: remove from i, insert to j ejecting customer, insert ejected to k)
        // This is a 3-route coordinated move: route_i → route_j → route_k
        
        Solution best_local = initial_solution;
        vector<int> best_tabu_key;
        double best_local_cost = 1e10;
        double best_local_sum_times = 1e10;
        
        // Maximum chain exploration to avoid exponential complexity
        const int MAX_ROUTE_TRIPLETS = min(50, (h + d) * (h + d - 1) * (h + d - 2) / 6);
        int triplets_evaluated = 0;
        
        // Build unified vehicle indexing: 0..h-1 trucks, h..h+d-1 drones
        auto get_route = [&](int veh_id) -> const vi& {
            return (veh_id < h) ? initial_solution.truck_routes[veh_id]
                                : initial_solution.drone_routes[veh_id - h];
        };
        
        auto is_truck_veh = [&](int veh_id) -> bool {
            return veh_id < h;
        };
        
        // Lambda to normalize routes (ensure depot endpoints, collapse consecutive depots)
        auto normalize_route = [](vi& r) {
            if (r.empty() || r.front() != 0) r.insert(r.begin(), 0);
            if (r.back() != 0) r.push_back(0);
            if (r.size() >= 2) {
                vi tmp; tmp.reserve(r.size());
                for (int x : r) {
                    if (!tmp.empty() && tmp.back() == 0 && x == 0) continue;
                    tmp.push_back(x);
                }
                r.swap(tmp);
            }
        };
        
        // Enumerate all route triplets (i, j, k) where i ≠ j ≠ k
        for (int veh_i = 0; veh_i < h + d && triplets_evaluated < MAX_ROUTE_TRIPLETS; ++veh_i) {
            const vi& route_i = get_route(veh_i);
            if (route_i.size() <= 2) continue; // need at least one customer to remove
            
            // Collect non-depot customers from route_i
            vector<int> cust_i;
            for (int c : route_i) if (c != 0) cust_i.push_back(c);
            if (cust_i.empty()) continue;
            
            for (int veh_j = 0; veh_j < h + d && triplets_evaluated < MAX_ROUTE_TRIPLETS; ++veh_j) {
                if (veh_j == veh_i) continue;
                const vi& route_j = get_route(veh_j);
                if (route_j.size() <= 1) continue; // need space to insert
                
                // Collect non-depot customers from route_j
                vector<int> cust_j;
                for (int c : route_j) if (c != 0) cust_j.push_back(c);
                
                for (int veh_k = 0; veh_k < h + d && triplets_evaluated < MAX_ROUTE_TRIPLETS; ++veh_k) {
                    if (veh_k == veh_i || veh_k == veh_j) continue;
                    const vi& route_k = get_route(veh_k);
                    if (route_k.size() <= 1) continue; // need space to insert
                    
                    ++triplets_evaluated;
                    
                    // Try removing each customer from route_i
                    for (int cust_removed : cust_i) {
                        // KNN filter: only consider if removed customer is near some customer in route_j
                        bool near_j = false;
                        if (!KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)cust_removed) {
                            for (int c : cust_j) {
                                if (KNN_ADJ[cust_removed].size() > (size_t)c && KNN_ADJ[cust_removed][c]) {
                                    near_j = true;
                                    break;
                                }
                            }
                            if (!near_j && !cust_j.empty()) continue;
                        }
                        
                        // Remove customer from route_i
                        vi new_route_i = route_i;
                        auto it_i = find(new_route_i.begin(), new_route_i.end(), cust_removed);
                        if (it_i == new_route_i.end()) continue;
                        new_route_i.erase(it_i);
                        normalize_route(new_route_i);
                        
                        // Check feasibility of reduced route_i
                        auto [t_i, feas_i] = check_route_feasibility(new_route_i, 0.0, is_truck_veh(veh_i));
                        if (!feas_i) continue;
                        
                        // Try inserting cust_removed into route_j at each position (potentially ejecting existing customer)
                        for (size_t pos_j = 0; pos_j < cust_j.size(); ++pos_j) {
                            int cust_ejected = cust_j[pos_j];
                            
                            // KNN filter: check if removed customer is near ejected customer
                            if (!KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)cust_removed) {
                                if (!(KNN_ADJ[cust_removed].size() > (size_t)cust_ejected && 
                                      KNN_ADJ[cust_removed][cust_ejected])) {
                                    continue;
                                }
                            }
                            
                            // Build route_j with cust_removed replacing cust_ejected at position
                            vi new_route_j = route_j;
                            auto it_j = find(new_route_j.begin(), new_route_j.end(), cust_ejected);
                            if (it_j == new_route_j.end()) continue;
                            *it_j = cust_removed; // replace ejected customer with removed customer
                            normalize_route(new_route_j);
                            
                            // Check feasibility of modified route_j
                            auto [t_j, feas_j] = check_route_feasibility(new_route_j, 0.0, is_truck_veh(veh_j));
                            if (!feas_j) continue;
                            
                            // Try inserting ejected customer into route_k at all positions
                            vi new_route_k = route_k;
                            // Try insertion at position 1 (after first depot)
                            for (int ins_pos_k = 1; ins_pos_k <= (int)new_route_k.size(); ++ins_pos_k) {
                                vi test_route_k = route_k;
                                int ip = min(max(ins_pos_k, 1), (int)test_route_k.size());
                                test_route_k.insert(test_route_k.begin() + ip, cust_ejected);
                                normalize_route(test_route_k);
                                
                                // Check feasibility of modified route_k
                                auto [t_k, feas_k] = check_route_feasibility(test_route_k, 0.0, is_truck_veh(veh_k));
                                if (!feas_k) continue;
                                
                                // Construct candidate solution
                                Solution candidate = initial_solution;
                                if (is_truck_veh(veh_i)) candidate.truck_routes[veh_i] = new_route_i;
                                else candidate.drone_routes[veh_i - h] = new_route_i;
                                
                                if (is_truck_veh(veh_j)) candidate.truck_routes[veh_j] = new_route_j;
                                else candidate.drone_routes[veh_j - h] = new_route_j;
                                
                                if (is_truck_veh(veh_k)) candidate.truck_routes[veh_k] = test_route_k;
                                else candidate.drone_routes[veh_k - h] = test_route_k;
                                
                                // Update times for modified routes
                                candidate.truck_route_times = initial_solution.truck_route_times;
                                candidate.drone_route_times = initial_solution.drone_route_times;
                                
                                if (is_truck_veh(veh_i)) candidate.truck_route_times[veh_i] = (new_route_i.size() > 1) ? t_i : 0.0;
                                else candidate.drone_route_times[veh_i - h] = (new_route_i.size() > 1) ? t_i : 0.0;
                                
                                if (is_truck_veh(veh_j)) candidate.truck_route_times[veh_j] = (new_route_j.size() > 1) ? t_j : 0.0;
                                else candidate.drone_route_times[veh_j - h] = (new_route_j.size() > 1) ? t_j : 0.0;
                                
                                if (is_truck_veh(veh_k)) candidate.truck_route_times[veh_k] = (test_route_k.size() > 1) ? t_k : 0.0;
                                else candidate.drone_route_times[veh_k - h] = (test_route_k.size() > 1) ? t_k : 0.0;
                                
                                // Compute makespan
                                double nb_makespan = 0.0;
                                for (double t : candidate.truck_route_times) nb_makespan = max(nb_makespan, t);
                                for (double t : candidate.drone_route_times) nb_makespan = max(nb_makespan, t);
                                candidate.total_makespan = nb_makespan;
                                
                                // Build tabu key: sorted pair of relocated customers
                                vector<int> tabu_key = {cust_removed, cust_ejected};
                                sort(tabu_key.begin(), tabu_key.end());
                                
                                // Check tabu status
                                bool is_tabu = (tabu_list_ejection.count(tabu_key) > 0 && 
                                               tabu_list_ejection[tabu_key] >= current_iter);
                                
                                // Aspiration criterion: accept if improving global best
                                if (is_tabu && candidate.total_makespan >= best_cost) continue;
                                
                                // Update best if improved
                                if (candidate.total_makespan < best_local_cost) {
                                    best_local_cost = candidate.total_makespan;
                                    best_local_sum_times = t_i + t_j + t_k;
                                    best_local = candidate;
                                    best_tabu_key = tabu_key;
                                }
                                // tie-breaker: smaller sum of modified routes' times
                                else if (candidate.total_makespan == best_local_cost) {
                                    double sum_times_candidate = t_i + t_j + t_k;
                                    double sum_times_best = best_local_sum_times;
                                    if (sum_times_candidate < sum_times_best) {
                                        best_local = candidate;
                                        best_tabu_key = tabu_key;
                                        best_local_sum_times = sum_times_candidate;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Apply best ejection chain found
        if (!best_tabu_key.empty() && best_local_cost < 1e10) {
            tabu_list_ejection[best_tabu_key] = current_iter + TABU_TENURE_EJECTION;
            
            // Debug output - uncomment to trace ejection chain moves
            /* cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[N7] Ejection chain: customers [";
            for (size_t i = 0; i < best_tabu_key.size(); ++i) {
                if (i > 0) cout << ", ";
                cout << best_tabu_key[i];
            }
            cout << "], makespan: " << initial_solution.total_makespan 
                 << " -> " << best_local.total_makespan
                 << ", iter " << current_iter << "\n";
             */
            return best_local;
        }
        
        return initial_solution; // no improving ejection chain found
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
        double alpha = 0.998; // cooling rate
    // Reset tabu lists at the start of each segment (iteration counter restarts per segment)
        tabu_list_switch.clear();
        tabu_list_swap.clear();
        tabu_list_reinsert.clear();
        tabu_list_2opt.clear();
        tabu_list_2opt_star.clear();
        tabu_list_relocate.clear();
        tabu_list_22.clear();
        tabu_list_21.clear();
        while (iter < CFG_MAX_ITER_PER_SEGMENT && no_improve_iters < CFG_MAX_NO_IMPROVE) {
            if (CFG_TIME_LIMIT_SEC > 0.0) {
                double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
                if (elapsed >= CFG_TIME_LIMIT_SEC) break;
            }
            double total_weight = 0.0;
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                total_weight += weight[i];
            }
            double r = ((double) rand() / (RAND_MAX));
            int selected_neighbor = 0;
            double cumulative = 0.0;
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                cumulative += weight[i] / total_weight;
                if (r < cumulative) {
                    selected_neighbor = i;
                    break;
                }
            }
            if (selected_neighbor == 0 && r >= cumulative) {
                selected_neighbor = NUM_NEIGHBORHOODS - 1;
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

        // Debug: print neighborhood weights after each segment
        /*cout.setf(std::ios::fixed); cout << setprecision(6);
        cout << "[Segment " << (segment + 1) << "] weights:";
        for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            cout << " w" << i << "=" << weight[i];
        }
        cout << "\n";*/
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
        os << "|Truck Time: " << sol.truck_route_times[i];
        os << "\n";
    }
    os << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        os << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            os << node << " ";
        }
        os << "|Drone Time: " << sol.drone_route_times[i];
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
        if (n <= 20) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 100);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 20);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 100);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 25);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // modest k for small n
        } else if (n <= 100) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 50);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 10);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 1000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 200);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // moderate k for medium n
        } else {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 5);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 20);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 2000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 100);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // slightly smaller k for very large n
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