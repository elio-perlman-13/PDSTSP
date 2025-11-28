#include <bits/stdc++.h>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <random>
#include <optional>

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
static vector<vector<int>> tabu_list_10; // sized (n+1) x (n+1)
static int TABU_TENURE_10 = 20; // default tenure for swap moves
static vector<vector<int>> tabu_list_11; // sized (n+1) x (h + d)
static int TABU_TENURE_11 = 20; // default tenure for relocate moves
// Separate tabu list for intra-route reinsert (Or-opt-1) moves
static map<vector<int>, int> tabu_list_20; // keyed by (cust_id1, cust_id2, vehicle_id)
static int TABU_TENURE_20 = 20; // default tenure for reinsert moves
// Separate tabu list for 2-opt moves: keyed by segment endpoints (min_id,max_id)
static vector<vector<int>> tabu_list_2opt; // sized (n+1) x (n+1) 
static int TABU_TENURE_2OPT = 25; // default tenure for 2-opt moves
static vector<vector<int>> tabu_list_2opt_star; // sized (n+1) x (n+1)
static int TABU_TENURE_2OPT_STAR = 20; // default tenure for 2-opt-star moves
static map<vector<int>, int> tabu_list_21; // keyed by (a,b,c,d) for (2,1) moves
static int TABU_TENURE_21 = 20; // default tenure for (2,1) moves
static map<vector<int>, int> tabu_list_22; // keyed by (a,b,c,d) for (2,2) moves
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

// Adaptive penalty coefficients for constraint violations
static double PENALTY_LAMBDA_CAPACITY = 1.0;      // λ for capacity violations
static double PENALTY_LAMBDA_ENERGY = 1.0;        // λ for energy violations  
static double PENALTY_LAMBDA_DEADLINE = 1.0;      // λ for deadline violations
static double PENALTY_EXPONENT = 0.5;             // exponent for penalty term *

static const double PENALTY_INCREASE = 1.2;       // multiply when violated *
static const double PENALTY_DECREASE = 1.2;       // divide when satisfied *
static const double PENALTY_MIN = 0.5;            // minimum λ value
static const double PENALTY_MAX = 1000.0;

// Destroy and repair helper
vvd edge_records; // edge_records[i][j]: stores working times for edge (i,j)
const double DESTROY_RATE = 0.3; // fraction of customers to remove during destroy phase
const int EJECTION_CHAIN_ITERS = 20; // number of ejection chain applications during destroy-repair

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    vd truck_route_times; //truck_route_times[i]: total time of truck i
    vd drone_route_times; //drone_route_times[i]: total time of drone i
    double total_makespan; //total makespan of the solution
    double capacity_violation = 0.0;    // sum of excess capacity / total capacity
    double energy_violation = 0.0;      // sum of excess energy / total battery
    double deadline_violation = 0.0;    // sum of deadline breaches / total deadlines
};

vector<Solution> elite_set; //store most promising solutions
const int ELITE_SET_SIZE = 10;

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

pair<double, double> compute_truck_route_time(const vi& route, double start=0) {
    double time = start; // seconds
    double deadline_feasible = 0.0;
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
                if (duration > deadline[cust] + 1e-8) {
                    deadline_feasible += duration - deadline[cust];
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

pair<double, double> compute_drone_route_energy(const vi& route) {
    double total_energy = 0, current_weight = 0;
    double energy_used = 0;
    double feasible = 0;
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
        if (energy_used > E + 1e-8) feasible += energy_used - E; // exceeded energy
        if (to != 0) current_weight += demand[to]; // add payload when delivering
        else {
            current_weight = 0; // reset weight when returning to depot
            energy_used = 0; // reset energy (charged at depot)
        }
    }
    return make_pair(total_energy, feasible);
}

pair<double, double> compute_drone_route_time(const vi& route) {
    double time = 0; // seconds
    double deadline_feasible = 0.0;
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
                if (duration > deadline[cust] + 1e-8) {
                    deadline_feasible += duration - deadline[cust];
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
        // Deadline: use compute_drone_route_time for depot->customer->depot
        auto [total_time, feasible_deadline] = compute_drone_route_time(route);
        if (feasible_deadline > 1e-8) {
            served_by_drone[customer] = 0;
            continue;
        }
        served_by_drone[customer] = 1;
    }
}

// Returns: [route_time, deadline_violation, energy_violation, capacity_violation]
vector<double> check_truck_route_feasibility(const vi& route, double start=0) {
    // Check deadlines
    auto [time, deadline_violation] = compute_truck_route_time(route, start);
    
    // Check capacity (reset at depot)
    double capacity_violation = 0.0;
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dh + 1e-9) {
                capacity_violation += (total_demand - Dh) / Dh; // normalized excess
            }
        }
    }
    
    // Trucks don't use energy (set to 0)
    double energy_violation = 0.0;
    
    return {time, deadline_violation, energy_violation, capacity_violation};
}

// Returns: [route_time, deadline_violation, energy_violation, capacity_violation]
vector<double> check_drone_route_feasibility(const vi& route) {
    // Check deadlines
    auto [time, deadline_violation] = compute_drone_route_time(route);
    
    // Check capacity (reset at depot)
    double capacity_violation = 0.0;
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dd + 1e-9) {
                capacity_violation += (total_demand - Dd) / Dd; // normalized excess
            }
        }
    }
    
    // Check energy
    auto energy_metrics = compute_drone_route_energy(route);
    double energy_violation = max(0.0, energy_metrics.second / E); // normalized excess beyond battery per sortie
    
    return {time, deadline_violation, energy_violation, capacity_violation};
}

// Unified wrapper
vector<double> check_route_feasibility(const vi& route, double start=0, bool is_truck = true) {
    if (is_truck) {
        return check_truck_route_feasibility(route, start);
    } else {
        return check_drone_route_feasibility(route);
    }
}

//score calculator
double solution_score(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    return sol.total_makespan * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double solution_score_total_time(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    double total_time = 0.0;
    for (double t : sol.truck_route_times) total_time += t;
    for (double t : sol.drone_route_times) total_time += t;
    return total_time * pow(penalty_multiplier, PENALTY_EXPONENT);
}

void update_penalties(const Solution& sol) {
    // Capacity penalty
    if (sol.capacity_violation > 1e-9) {
        PENALTY_LAMBDA_CAPACITY = min(PENALTY_MAX, PENALTY_LAMBDA_CAPACITY * PENALTY_INCREASE);
    } else {
        PENALTY_LAMBDA_CAPACITY = max(PENALTY_MIN, PENALTY_LAMBDA_CAPACITY / PENALTY_DECREASE);
    }
    
    // Energy penalty
    if (sol.energy_violation > 1e-9) {
        PENALTY_LAMBDA_ENERGY = min(PENALTY_MAX, PENALTY_LAMBDA_ENERGY * PENALTY_INCREASE);
    } else {
        PENALTY_LAMBDA_ENERGY = max(PENALTY_MIN, PENALTY_LAMBDA_ENERGY / PENALTY_DECREASE);
    }
    
    // Deadline penalty
    if (sol.deadline_violation > 1e-9) {
        PENALTY_LAMBDA_DEADLINE = min(PENALTY_MAX, PENALTY_LAMBDA_DEADLINE * PENALTY_INCREASE);
    } else {
        PENALTY_LAMBDA_DEADLINE = max(PENALTY_MIN, PENALTY_LAMBDA_DEADLINE / PENALTY_DECREASE);
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
                vd t_route = check_truck_route_feasibility(r, 0.0);
                double t = t_route[0];
                bool feas = (t_route[1] < 1e-8) && (t_route[3] < 1e-9); // deadline and capacity
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
                vd d_route = check_drone_route_feasibility(r);
                double t = d_route[0];
                bool feas = (d_route[1] < 1e-8) && (d_route[2] < 1e-8) && (d_route[3] < 1e-9); // deadline, energy, capacity
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
    // forced feasibility for now
    sol.capacity_violation = 0.0;
    sol.energy_violation = 0.0;
    sol.deadline_violation = 0.0;
    for (int i = 0; i < h; ++i) {
        vd metrics = check_truck_route_feasibility(sol.truck_routes[i], 0.0);
        sol.deadline_violation += metrics[1];
        sol.capacity_violation += metrics[3];
    }
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
        vd metrics = check_drone_route_feasibility(sol.drone_routes[i]);
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];
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

// Stream-based printer to avoid duplicating formatting on stdout/file
static void print_solution_stream(const Solution& sol, std::ostream& os) {
    os << "Truck Routes:\n";
    for (int i = 0; i < h; ++i) {
        os << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            os << node << " ";
        }
        vd truck_metric = check_route_feasibility(sol.truck_routes[i], 0.0, true);
        os << "|Truck Time: " << sol.truck_route_times[i] << "|" << truck_metric[0] << "," << truck_metric[1] << "," << truck_metric[2] << "," << truck_metric[3];
        os << "\n";
    }
    os << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        os << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            os << node << " ";
        }
        vd drone_metric = check_route_feasibility(sol.drone_routes[i], 0.0, false);
        os << "|Drone Time: " << sol.drone_route_times[i] << "|" << drone_metric[0] << "," << drone_metric[1] << "," << drone_metric[2] << "," << drone_metric[3];
        os << "\n";
    }
    os << "Total validation:" 
       << " Makespan=" << sol.total_makespan
       << ", Deadline violation=" << sol.deadline_violation
       << ", Energy violation=" << sol.energy_violation
       << ", Capacity violation=" << sol.capacity_violation
       << "\n";
}

pair<int, bool> critical_solution_index(const Solution& sol) {
    // Identify the vehicle (truck or drone) that contributes most to the penalized objective.
    double best_violation_weight = -1.0;
    double best_score = -1.0;
    bool is_truck = true;
    int best_idx = -1; // unified index: trucks [0,h), drones [h,h+d)

    auto evaluate_route = [&](const vi& route, double cached_time, bool is_truck, int unified_idx) {
        vector<double> metrics = check_route_feasibility(route, 0.0, is_truck);
        double base_time = (route.size() > 1)
            ? (cached_time > 0.0 ? cached_time : metrics[0])
            : 0.0;
        double violation_weight =
            PENALTY_LAMBDA_DEADLINE * metrics[1] +
            PENALTY_LAMBDA_ENERGY   * metrics[2] +
            PENALTY_LAMBDA_CAPACITY * metrics[3];
        double penalty_multiplier = 1.0 + violation_weight;
        double score = base_time * pow(penalty_multiplier, PENALTY_EXPONENT);

        if (violation_weight > best_violation_weight + 1e-12 ||
            (std::fabs(violation_weight - best_violation_weight) <= 1e-12 && score > best_score + 1e-9)) {
            best_violation_weight = violation_weight;
            best_score = score;
            best_idx = unified_idx;
        }
    };

    for (int i = 0; i < h; ++i) {
        double cached_time = (i < (int)sol.truck_route_times.size()) ? sol.truck_route_times[i] : 0.0;
        evaluate_route(sol.truck_routes[i], cached_time, true, i);
    }
    for (int i = 0; i < (int)sol.drone_route_times.size(); ++i) {
        double cached_time = sol.drone_route_times[i];
        evaluate_route(sol.drone_routes[i], cached_time, false, h + i);
    }

    if (best_idx == -1) {
        // Fallback: pick the route with the largest cached time to keep progress moving.
        double max_time = -1.0;
        int fallback_idx = 0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)sol.truck_route_times.size()) ? sol.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; fallback_idx = i; }
        }
        for (int i = 0; i < (int)sol.drone_route_times.size(); ++i) {
            double t = sol.drone_route_times[i];
            if (t > max_time) { max_time = t; fallback_idx = h + i; }
        }
        best_idx = fallback_idx;
    }
    if (best_idx < h) {
        is_truck = true;
    } else {
        is_truck = false;
        best_idx -= h;
    }
    return {best_idx, is_truck};
}

Solution local_search(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 1e10;
    // Depending on neighbor_id, implement different neighborhood structures
    if (neighbor_id == 0) {
        // Relocate 1 customer from the critical (longest-time) vehicle route to another route of the same mode
        // 1) Identify critical vehicle (truck or drone) using precomputed times
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        // Ensure tabu list is sized to (n+1) x (h+d)
        int veh_count = h + d;
        if ((int)tabu_list_10.size() != n + 1 || (veh_count > 0 && (int)tabu_list_10[0].size() != veh_count)) {
            tabu_list_10.assign(n + 1, vector<int>(max(0, veh_count), 0));
        }

        // Prepare neighborhood best tracking
        int best_target = -1; // vehicle index in unified space (0..h-1 trucks, h..h+d-1 drones)
        int best_cust = -1;   // moved customer id
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
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

            vd crit_route_time_feas = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust = base_route[p];
                // Try relocating cust to other vehicles of the same mode
                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    bool is_tabu = (tabu_list_10[cust][target_veh] > current_iter);
                    if (target_veh == critical_vehicle_id){
                        for (int p2 = 1; p2 < (int)base_route.size(); ++p2) {
                            // attempt to relocate cust to position p2 in the same route
                            if (p2 == p) continue; // no-op
                            // Build new candidate solution with relocation
                            Solution candidate = initial_solution;
                            vi new_base_route = base_route;
                            new_base_route.erase(new_base_route.begin() + p);
                            new_base_route.insert(new_base_route.begin() + p2 - (p2 > p ? 1 : 0), cust);
                            vi new_norm = normalize_route(new_base_route);
                            // Compute cost of candidate
                            vd new_crit_route_time_feas = is_truck_mode
                                ? check_route_feasibility(new_norm, 0.0, true)
                                : check_route_feasibility(new_norm, 0.0, false);
                            // If tabu, only consider if it improves over best known cost and feasible
                            candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                            candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                            candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = new_norm;
                                candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = new_norm;
                                candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                            }
                            candidate.total_makespan = 0.0;
                            for (int i = 0; i < h; ++i) {
                                candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                            }
                            for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                                candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                            }
                            double candidate_score = solution_score(candidate);
                            bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                       candidate.capacity_violation <= 1e-8 &&
                                                       candidate.energy_violation <= 1e-8;
                            if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                                continue; // skip tabu move
                            }
                            // Update best if improved
                            if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = candidate_score;
                                best_candidate_neighbor = candidate;
                                best_target = target_veh;
                                best_cust = cust;
                            }
                        }
                        // also consider relocating to the end of the route
                        Solution candidate = initial_solution;
                        vi new_base_route = base_route;
                        new_base_route.erase(new_base_route.begin() + p);
                        if (new_base_route.back() != 0) {
                            new_base_route.push_back(0); // ensure depot at end for insertion
                        }
                        new_base_route.push_back(cust);
                        new_base_route.push_back(0);
                        vi new_norm = normalize_route(new_base_route);
                        // Compute cost of candidate
                        vd new_crit_route_time_feas = is_truck_mode
                            ? check_route_feasibility(new_norm, 0.0, true)
                            : check_route_feasibility(new_norm, 0.0, false);
                        // If tabu, only consider if it improves over best known cost and feasible
                        candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                        candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                        candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = new_norm;
                            candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = new_norm;
                            candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                        }
                        candidate.total_makespan = 0.0;
                        for (int i = 0; i < h; ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                        }
                        for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                        }
                        double candidate_score = solution_score(candidate);
                        bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                   candidate.capacity_violation <= 1e-8 &&
                                                   candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                            continue; // skip tabu move
                        }
                        // Update best if improved
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_target = target_veh;
                            best_cust = cust;
                        }
                    }
                    if (target_veh == critical_vehicle_id) continue; // skip same vehicle here
                    const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh - 0]
                                                              : initial_solution.drone_routes[target_veh - h];
                    vd target_route_time_feas = (target_veh < h)
                        ? check_route_feasibility(target_route, 0.0, true)
                        : check_route_feasibility(target_route, 0.0, false);                                          
                    // Try all insertion positions in target route
                    // Remove cust from base route
                    vi new_base_route = base_route;
                    new_base_route.erase(new_base_route.begin() + p);
                    vd new_crit_route_time_feas = is_truck_mode
                        ? check_route_feasibility(new_base_route, 0.0, true)
                        : check_route_feasibility(new_base_route, 0.0, false);
                    for (int insert_pos = 1; insert_pos < (int)target_route.size(); ++insert_pos) {
                        // Build new candidate solution with relocation
                        Solution candidate = initial_solution;
                        // Insert cust into target route
                        vi new_target_route = target_route;
                        new_target_route.insert(new_target_route.begin() + insert_pos, cust);
                        vd new_target_route_time_feas = (target_veh < h)
                            ? check_route_feasibility(new_target_route, 0.0, true)
                            : check_route_feasibility(new_target_route, 0.0, false);

                        // Compute cost of candidate
                        candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                        candidate.deadline_violation += new_target_route_time_feas[1] - target_route_time_feas[1];
                        candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                        candidate.capacity_violation += new_target_route_time_feas[3] - target_route_time_feas[3];
                        candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                        candidate.energy_violation += new_target_route_time_feas[2] - target_route_time_feas[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = new_base_route;
                            candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = new_base_route;
                            candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                        }
                        if (target_veh < h) {
                            candidate.truck_routes[target_veh] = new_target_route;
                            candidate.truck_route_times[target_veh] = new_target_route_time_feas[0];
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target_route;
                            candidate.drone_route_times[target_veh - h] = new_target_route_time_feas[0];
                        }
                        candidate.total_makespan = 0.0;
                        for (int i = 0; i < h; ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                        }
                        
                        for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                        }
                        double candidate_score = solution_score(candidate);
                        bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                   candidate.capacity_violation <= 1e-8 &&
                                                   candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                            continue; // skip tabu move
                        }
                        // Update best if improved
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_target = target_veh;
                            best_cust = cust;
                        }
                    }
                    // Also try inserting at the end of the target route
                    vi new_target_route = target_route;
                    new_target_route.push_back(cust);
                    new_target_route.push_back(0);
                    vd new_target_route_time_feas = (target_veh < h)
                        ? check_route_feasibility(new_target_route, 0.0, true)
                        : check_route_feasibility(new_target_route, 0.0, false);
                    // Build new candidate solution with relocation
                    Solution candidate = initial_solution;
                    // Compute cost of candidate
                    candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                    candidate.deadline_violation += new_target_route_time_feas[1] - target_route_time_feas[1];
                    candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                    candidate.capacity_violation += new_target_route_time_feas[3] - target_route_time_feas[3];
                    candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                    candidate.energy_violation += new_target_route_time_feas[2] - target_route_time_feas[2];
                    if (is_truck_mode) {
                        candidate.truck_routes[critical_vehicle_id] = new_base_route;
                        candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                    } else {
                        candidate.drone_routes[critical_vehicle_id - h] = new_base_route;
                        candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                    }
                    if (target_veh < h) {
                        candidate.truck_routes[target_veh] = new_target_route;
                        candidate.truck_route_times[target_veh] = new_target_route_time_feas[0];
                    } else {
                        candidate.drone_routes[target_veh - h] = new_target_route;
                        candidate.drone_route_times[target_veh - h] = new_target_route_time_feas[0];
                    }
                    candidate.total_makespan = 0.0;
                    for (int i = 0; i < h; ++i) {
                        candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                    }
                    for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                        candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                    }
                    double candidate_score = solution_score(candidate);
                    bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                               candidate.capacity_violation <= 1e-8 &&
                                               candidate.energy_violation <= 1e-8;
                    if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                        continue; // skip tabu move
                    }
                    // Update best if improved
                    if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                        best_neighbor_cost_local = candidate_score;
                        best_candidate_neighbor = candidate;
                        best_target = target_veh;
                        best_cust = cust;
                    }
                }
            }
        };
        if (crit_is_truck) {
            consider_relocate(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_relocate(initial_solution.drone_routes[critical_idx], false, critical_idx + h);
        }

        // After evaluating all candidates, update tabu list if we found an improving move
        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Update tabu list
            tabu_list_10[best_cust][best_target] = current_iter + TABU_TENURE_10;
        }
        // Debug:
        //cout << "[N0] relocate customer " << best_cust << " to vehicle " << best_target << " score: " << solution_score(initial_solution) << " -> " << solution_score(best_neighbor) << "\n";
        return best_neighbor;
    } else if (neighbor_id == 1) {
        // Neighborhood 1: swap two customers, allowing cross-mode exchanges
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        if ((int)tabu_list_11.size() != n + 1 || (n + 1 > 0 && (int)tabu_list_11[0].size() != n + 1)) {
            tabu_list_11.assign(n + 1, vector<int>(n + 1, 0));
        }

        int best_cust_a = -1, best_cust_b = -1;
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 2) return; // nothing to swap

            vd crit_metrics = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            vector<int> crit_positions;
            for (int i = 0; i < (int)base_route.size(); ++i) {
                if (base_route[i] != 0) crit_positions.push_back(i);
            }
            if (crit_positions.empty()) return;

            for (int idx1 = 0; idx1 < (int)crit_positions.size(); ++idx1) {
                int pos1 = crit_positions[idx1];
                int cust_a = base_route[pos1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    bool target_is_truck = target_veh < h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_veh]
                        : initial_solution.drone_routes[target_veh - h];

                    if (target_route.size() <= 2 && target_veh != critical_vehicle_id) continue;

                    vd target_metrics;
                    vector<int> target_positions;
                    if (target_veh == critical_vehicle_id) {
                        target_positions = crit_positions;
                    } else {
                        target_metrics = target_is_truck
                            ? check_route_feasibility(target_route, 0.0, true)
                            : check_route_feasibility(target_route, 0.0, false);
                        for (int j = 0; j < (int)target_route.size(); ++j) {
                            if (target_route[j] != 0) target_positions.push_back(j);
                        }
                        if (target_positions.empty()) continue;
                    }

                    int start_idx2 = (target_veh == critical_vehicle_id) ? idx1 + 1 : 0;
                    for (int idx2 = start_idx2; idx2 < (int)target_positions.size(); ++idx2) {
                        int pos2 = target_positions[idx2];
                        int cust_b = (target_veh == critical_vehicle_id) ? base_route[pos2] : target_route[pos2];
                        if (cust_a == cust_b) continue;
                        if (KNN_ADJ.size() > 0) {
                            // If using KNN, only consider swaps between neighbors
                            if (KNN_ADJ[cust_a][cust_b] == 0 && KNN_ADJ[cust_b][cust_a] == 0) {
                                continue;
                            }
                        }
                        if (!is_truck_mode && target_veh != critical_vehicle_id && !served_by_drone[cust_b]) continue;
                        if (!target_is_truck && !served_by_drone[cust_a]) continue;
                        if (target_veh == critical_vehicle_id && !is_truck_mode && !served_by_drone[cust_a] && !served_by_drone[cust_b]) continue;

                        int u = min(cust_a, cust_b);
                        int v = max(cust_a, cust_b);
                        bool is_tabu = (tabu_list_11.size() > (size_t)u && tabu_list_11[u].size() > (size_t)v &&
                                        tabu_list_11[u][v] > current_iter);

                        vi crit_route_new = base_route;
                        vi target_route_new = target_route;

                        if (target_veh == critical_vehicle_id) {
                            swap(crit_route_new[pos1], crit_route_new[pos2]);
                        } else {
                            swap(crit_route_new[pos1], target_route_new[pos2]);
                        }

                        vd new_crit_metrics = is_truck_mode
                            ? check_route_feasibility(crit_route_new, 0.0, true)
                            : check_route_feasibility(crit_route_new, 0.0, false);

                        vd new_target_metrics;
                        if (target_veh != critical_vehicle_id) {
                            new_target_metrics = target_is_truck
                                ? check_route_feasibility(target_route_new, 0.0, true)
                                : check_route_feasibility(target_route_new, 0.0, false);
                        }

                        Solution candidate = initial_solution;

                        candidate.deadline_violation += new_crit_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += new_crit_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += new_crit_metrics[2] - crit_metrics[2];

                        if (target_veh != critical_vehicle_id) {
                            candidate.deadline_violation += new_target_metrics[1] - target_metrics[1];
                            candidate.capacity_violation += new_target_metrics[3] - target_metrics[3];
                            candidate.energy_violation += new_target_metrics[2] - target_metrics[2];
                        }

                        if (is_truck_mode) {
                            candidate.truck_routes[critical_idx] = crit_route_new;
                            candidate.truck_route_times[critical_idx] = (crit_route_new.size() > 1) ? new_crit_metrics[0] : 0.0;
                        } else {
                            candidate.drone_routes[critical_idx] = crit_route_new;
                            candidate.drone_route_times[critical_idx] = (crit_route_new.size() > 1) ? new_crit_metrics[0] : 0.0;
                        }

                        if (target_veh != critical_vehicle_id) {
                            if (target_is_truck) {
                                candidate.truck_routes[target_veh] = target_route_new;
                                candidate.truck_route_times[target_veh] = (target_route_new.size() > 1) ? new_target_metrics[0] : 0.0;
                            } else {
                                candidate.drone_routes[target_veh - h] = target_route_new;
                                candidate.drone_route_times[target_veh - h] = (target_route_new.size() > 1) ? new_target_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (int i = 0; i < h; ++i) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                        for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i)
                            candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);

                        double candidate_score = solution_score(candidate);
                        bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                   candidate.capacity_violation <= 1e-8 &&
                                                   candidate.energy_violation <= 1e-8;

                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                            continue;
                        }

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_cust_a = cust_a;
                            best_cust_b = cust_b;
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            consider_swap(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_swap(initial_solution.drone_routes[critical_idx], false, h + critical_idx);
        }

        if (best_cust_a != -1 && best_cust_b != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            int u = min(best_cust_a, best_cust_b);
            int v = max(best_cust_a, best_cust_b);
            tabu_list_11[u][v] = current_iter + TABU_TENURE_11;

            // Debug: print swap info
            /*  cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N1] swap " << best_cust_a << " and " << best_cust_b
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << best_neighbor_cost_local
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 2) {
        // Neighborhood 2: relocate a consecutive pair (2,0)-move from the critical vehicle to another vehicle
        // Structure mirrors neighborhood 0: identify critical vehicle, enumerate candidate relocations,
        // respect tabu_list_20 keyed by (min(c1,c2), max(c1,c2), target_vehicle).
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        // Prepare best tracking
        Solution best_candidate_neighbor = initial_solution;
        double best_neighbor_cost_local = 1e10;
        int best_c1 = -1, best_c2 = -1;
        int best_target_vehicle = -1;
        int best_src_pos = -1, best_target_pos = -1;

        auto consider_relocate_pair = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 3) return; // nothing to do if fewer than two customers

            auto normalize_route = [](const vi& route) -> vi {
                vi normalized;
                for (int node : route) {
                    if (normalized.empty() || node != 0 || normalized.back() != 0) normalized.push_back(node);
                }
                if (normalized.empty()) return vi{0};
                if (normalized.front() != 0) normalized.insert(normalized.begin(), 0);
                if (normalized.back() != 0) normalized.push_back(0);
                return normalized;
            };

            vi orig = normalize_route(base_route);
            if (orig.size() <= 3) return;
            vd orig_metrics = is_truck_mode
                ? check_route_feasibility(orig, 0.0, true)
                : check_route_feasibility(orig, 0.0, false);

            vector<int> pos;
            for (int i = 0; i + 1 < (int)orig.size(); ++i) {
                if (orig[i] != 0 && orig[i + 1] != 0) pos.push_back(i);
            }
            if (pos.empty()) return;

            auto apply_candidate = [&](const vi& crit_route, const vd& crit_metrics_local,
                                      const optional<pair<int, vi>>& target_change) {
                Solution candidate = initial_solution;
                candidate.deadline_violation += crit_metrics_local[1] - orig_metrics[1];
                candidate.capacity_violation += crit_metrics_local[3] - orig_metrics[3];
                candidate.energy_violation += crit_metrics_local[2] - orig_metrics[2];
                if (is_truck_mode) {
                    candidate.truck_routes[critical_vehicle_id] = crit_route;
                    candidate.truck_route_times[critical_vehicle_id] = (crit_route.size() > 1) ? crit_metrics_local[0] : 0.0;
                } else {
                    candidate.drone_routes[critical_vehicle_id - h] = crit_route;
                    candidate.drone_route_times[critical_vehicle_id - h] = (crit_route.size() > 1) ? crit_metrics_local[0] : 0.0;
                }

                if (target_change.has_value()) {
                    int target_vehicle = target_change->first;
                    const vi& new_target_route = target_change->second;
                    bool target_is_truck = target_vehicle < h;
                    vd target_metrics_before = target_is_truck
                        ? check_route_feasibility(target_is_truck ? initial_solution.truck_routes[target_vehicle]
                                                                 : initial_solution.drone_routes[target_vehicle - h], 0.0, target_is_truck)
                        : check_route_feasibility(initial_solution.drone_routes[target_vehicle - h], 0.0, false);
                    vd target_metrics_after = target_is_truck
                        ? check_route_feasibility(new_target_route, 0.0, true)
                        : check_route_feasibility(new_target_route, 0.0, false);
                    candidate.deadline_violation += target_metrics_after[1] - target_metrics_before[1];
                    candidate.capacity_violation += target_metrics_after[3] - target_metrics_before[3];
                    candidate.energy_violation += target_metrics_after[2] - target_metrics_before[2];
                    if (target_is_truck) {
                        candidate.truck_routes[target_vehicle] = new_target_route;
                        candidate.truck_route_times[target_vehicle] = (new_target_route.size() > 1) ? target_metrics_after[0] : 0.0;
                    } else {
                        candidate.drone_routes[target_vehicle - h] = new_target_route;
                        candidate.drone_route_times[target_vehicle - h] = (new_target_route.size() > 1) ? target_metrics_after[0] : 0.0;
                    }
                }

                candidate.total_makespan = 0.0;
                for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                for (double tt : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, tt);
                return candidate;
            };

            for (int p : pos) {
                int c1 = orig[p];
                int c2 = orig[p + 1];

                vi reduced = orig;
                reduced.erase(reduced.begin() + p, reduced.begin() + p + 2);
                reduced = normalize_route(reduced);
                vd reduced_metrics = is_truck_mode
                    ? check_route_feasibility(reduced, 0.0, true)
                    : check_route_feasibility(reduced, 0.0, false);

                for (int ip = 1; ip <= (int)reduced.size(); ++ip) {
                    vi r = reduced;
                    r.insert(r.begin() + ip, c1);
                    r.insert(r.begin() + ip + 1, c2);
                    vi r_norm = normalize_route(r);
                    if (r_norm == orig) continue;

                    vd new_metrics = is_truck_mode
                        ? check_route_feasibility(r_norm, 0.0, true)
                        : check_route_feasibility(r_norm, 0.0, false);
                    Solution candidate = apply_candidate(r_norm, new_metrics, nullopt);

                    vector<int> key = { min(c1, c2), max(c1, c2), critical_vehicle_id };
                    auto it = tabu_list_20.find(key);
                    bool is_tabu = (it != tabu_list_20.end() && it->second > current_iter);
                    double candidate_score = solution_score(candidate);
                    bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                               candidate.capacity_violation <= 1e-8 &&
                                               candidate.energy_violation <= 1e-8;
                    if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) continue;

                    if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                        best_neighbor_cost_local = candidate_score;
                        best_candidate_neighbor = candidate;
                        best_c1 = c1; best_c2 = c2;
                        best_target_vehicle = critical_vehicle_id;
                        best_src_pos = p;
                        best_target_pos = ip;
                    }
                }

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == critical_vehicle_id) continue;
                    bool target_is_truck = target_veh < h;
                    vi target_route = target_is_truck
                        ? initial_solution.truck_routes[target_veh]
                        : initial_solution.drone_routes[target_veh - h];
                    target_route = normalize_route(target_route);

                    for (int insert_pos = 1; insert_pos <= (int)target_route.size(); ++insert_pos) {
                        vi new_target = target_route;
                        new_target.insert(new_target.begin() + insert_pos, c1);
                        new_target.insert(new_target.begin() + insert_pos + 1, c2);
                        new_target = normalize_route(new_target);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation = initial_solution.deadline_violation;
                        candidate.capacity_violation = initial_solution.capacity_violation;
                        candidate.energy_violation = initial_solution.energy_violation;

                        // build candidate with reduced critical route + modified target route
                        vd crit_metrics_ready = reduced_metrics;
                        vd target_metrics_new = target_is_truck
                            ? check_route_feasibility(new_target, 0.0, true)
                            : check_route_feasibility(new_target, 0.0, false);
                        vd target_metrics_old = target_is_truck
                            ? check_route_feasibility(target_route, 0.0, true)
                            : check_route_feasibility(target_route, 0.0, false);

                        candidate.deadline_violation += crit_metrics_ready[1] - orig_metrics[1];
                        candidate.deadline_violation += target_metrics_new[1] - target_metrics_old[1];
                        candidate.capacity_violation += crit_metrics_ready[3] - orig_metrics[3];
                        candidate.capacity_violation += target_metrics_new[3] - target_metrics_old[3];
                        candidate.energy_violation += crit_metrics_ready[2] - orig_metrics[2];
                        candidate.energy_violation += target_metrics_new[2] - target_metrics_old[2];

                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = reduced;
                            candidate.truck_route_times[critical_vehicle_id] = (reduced.size() > 1) ? crit_metrics_ready[0] : 0.0;
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = reduced;
                            candidate.drone_route_times[critical_vehicle_id - h] = (reduced.size() > 1) ? crit_metrics_ready[0] : 0.0;
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_veh] = new_target;
                            candidate.truck_route_times[target_veh] = (new_target.size() > 1) ? target_metrics_new[0] : 0.0;
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target;
                            candidate.drone_route_times[target_veh - h] = (new_target.size() > 1) ? target_metrics_new[0] : 0.0;
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double tt : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, tt);

                        vector<int> key = { min(c1, c2), max(c1, c2), target_veh };
                        auto it = tabu_list_20.find(key);
                        bool is_tabu = (it != tabu_list_20.end() && it->second > current_iter);
                        double candidate_score = solution_score(candidate);
                        bool feasible = candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && feasible)) continue;

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_c1 = c1; best_c2 = c2;
                            best_target_vehicle = target_veh;
                            best_src_pos = p;
                            best_target_pos = insert_pos;
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            consider_relocate_pair(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_relocate_pair(initial_solution.drone_routes[critical_idx], false, h + critical_idx);
        }

        // apply best move if found
        if (best_c1 != -1 && best_c2 != -1 && best_target_vehicle != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            // update tabu
            vector<int> key = { min(best_c1, best_c2), max(best_c1, best_c2), best_target_vehicle };
            tabu_list_20[key] = current_iter + TABU_TENURE_20;
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Debug:
            /*  cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N2] relocate pair (" << best_c1 << "," << best_c2 << ") to vehicle " << best_target_vehicle
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */
            // return the chosen neighbor (already fully assembled in best_candidate_neighbor)*/
            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 3) {
        // Neighborhood 3: 2-opt within each subroute (between depot nodes) for trucks or drones.
        // Finds the best 2-opt move across all routes that yields the largest local time drop.

        if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_edge_u = -1, best_edge_v = -1;
        int best_i = -1, best_j = -1;
        bool best_is_truck = crit_is_truck;

        auto normalize_route = [](vi r) -> vi {
            if (r.empty()) return r;
            if (r.front() != 0) r.insert(r.begin(), 0);
            if (r.back() != 0) r.push_back(0);
            vi cleaned; cleaned.reserve(r.size());
            for (int node : r) {
                if (!cleaned.empty() && cleaned.back() == 0 && node == 0) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto consider_2opt = [&](const vi& base_route, bool is_truck_mode, int route_idx) {
            if (base_route.size() <= 3) return;

            vd route_metrics = check_route_feasibility(base_route, 0.0, is_truck_mode);
            int m = (int)base_route.size();
            int start = 0;
            while (start < m) {
                while (start < m && base_route[start] == 0) ++start;
                if (start >= m) break;
                int seg_end = start;
                while (seg_end + 1 < m && base_route[seg_end + 1] != 0) ++seg_end;

                for (int i = start; i < seg_end; ++i) {
                    for (int j = i + 1; j <= seg_end; ++j) {
                        vi new_route = base_route;
                        reverse(new_route.begin() + i, new_route.begin() + j + 1);
                        if (new_route == base_route) continue;

                        new_route = normalize_route(new_route);
                        vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck_mode);
                        int u = min(base_route[i], base_route[j]);
                        int v = max(base_route[i], base_route[j]);
                        if (u < 0 || v < 0) continue;
                        bool is_tabu = (tabu_list_2opt.size() > (size_t)u &&
                                        tabu_list_2opt[u].size() > (size_t)v &&
                                        tabu_list_2opt[u][v] > current_iter);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += new_metrics[1] - route_metrics[1];
                        candidate.capacity_violation += new_metrics[3] - route_metrics[3];
                        candidate.energy_violation += new_metrics[2] - route_metrics[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[route_idx] = new_route;
                            candidate.truck_route_times[route_idx] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        } else {
                            int drone_route_idx = route_idx - h;
                            if (drone_route_idx >= 0 && drone_route_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[drone_route_idx] = new_route;
                                candidate.drone_route_times[drone_route_idx] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                            }
                        }
                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                        double candidate_score = solution_score(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) continue;

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_edge_u = u;
                            best_edge_v = v;
                            best_i = i;
                            best_j = j;
                            best_is_truck = is_truck_mode;
                        }
                    }
                }
                start = seg_end + 1;
            }
        };

        if (crit_is_truck) {
            consider_2opt(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_2opt(initial_solution.drone_routes[critical_idx], false, critical_idx + h);
        }

        if (best_edge_u != -1 && best_edge_v != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_2opt[best_edge_u][best_edge_v] = current_iter + TABU_TENURE_2OPT;

            // Debug N3
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N3] 2-opt on " << (best_is_truck ? "truck" : "drone") << " #"
                 << (crit_is_truck ? critical_idx + 1 : critical_idx + 1)
                 << " between positions " << best_i << " and " << best_j
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 4) {
        if ((int)tabu_list_2opt_star.size() != n + 1 || ((int)tabu_list_2opt_star.size() > 0 && (int)tabu_list_2opt_star[0].size() != n + 1)) {
            tabu_list_2opt_star.assign(n + 1, vector<int>(n + 1, 0));
        }

        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_ua = -1, best_va = -1, best_ub = -1, best_vb = -1;

        auto enumerate_segments = [](const vi& route) {
            vector<pair<int,int>> segs;
            int m = (int)route.size();
            int start = 0;
            while (start < m) {
                while (start < m && route[start] == 0) ++start;
                if (start >= m) break;
                int end = start;
                while (end + 1 < m && route[end + 1] != 0) ++end;
                segs.emplace_back(start, end);
                start = end + 1;
            }
            return segs;
        };

        const vi& crit_route_raw = crit_is_truck
            ? initial_solution.truck_routes[critical_idx]
            : initial_solution.drone_routes[critical_idx];
        vi crit_route = normalize_route(crit_route_raw);
        vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_is_truck);
        auto crit_segs = enumerate_segments(crit_route);

        auto evaluate_two_opt_star = [&](const vi& other_route_raw, bool other_is_truck, int other_idx) {
            vi other_route = normalize_route(other_route_raw);
            if (other_route.size() <= 3) return;
            vd other_metrics = check_route_feasibility(other_route, 0.0, other_is_truck);
            auto other_segs = enumerate_segments(other_route);

            for (const auto& segA : crit_segs) {
                for (int i = segA.first; i < segA.second; ++i) {
                    int a1 = crit_route[i], a2 = crit_route[i + 1];
                    int ua = min(a1, a2), va = max(a1, a2);
                    if (a1 == 0 || a2 == 0) continue;

                    for (const auto& segB : other_segs) {
                        for (int j = segB.first; j < segB.second; ++j) {
                            int b1 = other_route[j], b2 = other_route[j + 1];
                            int ub = min(b1, b2), vb = max(b1, b2);
                            if (b1 == 0 || b2 == 0) continue;

                            bool is_tabu = (tabu_list_2opt_star[ua][va] > current_iter) ||
                                           (tabu_list_2opt_star[ub][vb] > current_iter);

                            vi crit_new = crit_route;
                            vi other_new = other_route;

                            vi tailA(crit_new.begin() + i + 1, crit_new.begin() + segA.second + 1);
                            vi tailB(other_new.begin() + j + 1, other_new.begin() + segB.second + 1);

                            crit_new.erase(crit_new.begin() + i + 1, crit_new.begin() + segA.second + 1);
                            other_new.erase(other_new.begin() + j + 1, other_new.begin() + segB.second + 1);

                            crit_new.insert(crit_new.begin() + i + 1, tailB.begin(), tailB.end());
                            other_new.insert(other_new.begin() + j + 1, tailA.begin(), tailA.end());

                            crit_new = normalize_route(crit_new);
                            other_new = normalize_route(other_new);
                            if (crit_new == crit_route && other_new == other_route) continue;

                            vd crit_metrics_new = check_route_feasibility(crit_new, 0.0, crit_is_truck);
                            vd other_metrics_new = check_route_feasibility(other_new, 0.0, other_is_truck);

                            Solution candidate = initial_solution;
                            candidate.deadline_violation += crit_metrics_new[1] - crit_metrics[1];
                            candidate.capacity_violation += crit_metrics_new[3] - crit_metrics[3];
                            candidate.energy_violation += crit_metrics_new[2] - crit_metrics[2];
                            candidate.deadline_violation += other_metrics_new[1] - other_metrics[1];
                            candidate.capacity_violation += other_metrics_new[3] - other_metrics[3];
                            candidate.energy_violation += other_metrics_new[2] - other_metrics[2];

                            if (crit_is_truck) {
                                candidate.truck_routes[critical_idx] = crit_new;
                                candidate.truck_route_times[critical_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                            } else {
                                int crit_drone_idx = critical_idx - h;
                                if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[crit_drone_idx] = crit_new;
                                    candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                                }
                            }

                            if (other_is_truck) {
                                candidate.truck_routes[other_idx] = other_new;
                                candidate.truck_route_times[other_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                            } else {
                                int other_drone_idx = other_idx - h;
                                if (other_drone_idx >= 0 && other_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[other_drone_idx] = other_new;
                                    candidate.drone_route_times[other_drone_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                                }
                            }

                            candidate.total_makespan = 0.0;
                            for (int i = 0; i < h; ++i) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                            for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                            double candidate_score = solution_score(candidate);
                            if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                             candidate.deadline_violation <= 1e-8 &&
                                             candidate.capacity_violation <= 1e-8 &&
                                             candidate.energy_violation <= 1e-8)) {
                                continue;
                            }

                            if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = candidate_score;
                                best_candidate_neighbor = candidate;
                                best_ua = ua; best_va = va;
                                best_ub = ub; best_vb = vb;
                            }
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            for (int j = 0; j < h; ++j) {
                if (j == critical_idx) continue;
                evaluate_two_opt_star(initial_solution.truck_routes[j], true, j);
            }
        } else {
            for (int j = 0; j < (int)initial_solution.drone_routes.size(); ++j) {
                if (j == critical_idx) continue;
                evaluate_two_opt_star(initial_solution.drone_routes[j], false, j);
            }
        }

        if (best_ua != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_2opt_star[best_ua][best_va] = current_iter + TABU_TENURE_2OPT_STAR;
            tabu_list_2opt_star[best_ub][best_vb] = current_iter + TABU_TENURE_2OPT_STAR;

            //Debug N4
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N4] 2-opt* cuts (" << best_ua << "," << best_va << ") & (" << best_ub << "," << best_vb << ")"
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 5) {
       auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_triple;
        int best_pair_a = -1, best_pair_b = -1, best_single = -1;
        bool best_pair_from_critical = true;
        int best_other_vehicle = -1;
        bool best_other_is_truck = true;

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto consider_pair_vs_single = [&](const vi& crit_route, bool crit_mode_truck, int crit_route_idx) {
            if (crit_route.size() <= 3) return;

            vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_mode_truck);
            vector<int> pair_positions;
            for (int i = 0; i + 1 < (int)crit_route.size(); ++i)
                if (crit_route[i] != 0 && crit_route[i + 1] != 0) pair_positions.push_back(i);
            if (pair_positions.empty()) return;

            auto near_enough = [&](int u, int v) {
                return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u && KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
            };

            for (int pair_idx : pair_positions) {
                int c1 = crit_route[pair_idx];
                int c2 = crit_route[pair_idx + 1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == (crit_mode_truck ? crit_route_idx : h + crit_route_idx)) continue;
                    bool target_is_truck = target_veh < h;
                    if (!target_is_truck && (!served_by_drone[c1] || !served_by_drone[c2])) continue;

                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_idx]
                        : initial_solution.drone_routes[target_idx];
                    if (target_route.size() <= 2) continue;

                    vector<int> target_positions;
                    for (int j = 0; j < (int)target_route.size(); ++j)
                        if (target_route[j] != 0) target_positions.push_back(j);
                    if (target_positions.empty()) continue;

                    vd target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);

                    for (int pos_single : target_positions) {
                        int single = target_route[pos_single];
                        if (!crit_mode_truck && !served_by_drone[single]) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(c1, single) || near_enough(single, c1) ||
                                      near_enough(c2, single) || near_enough(single, c2);
                            if (!ok) continue;
                        }

                        vi crit_new = crit_route;
                        crit_new.erase(crit_new.begin() + pair_idx);
                        crit_new.erase(crit_new.begin() + pair_idx);
                        crit_new.insert(crit_new.begin() + pair_idx, single);
                        crit_new = normalize_route(crit_new);

                        vi target_new = target_route;
                        target_new.erase(target_new.begin() + pos_single);
                        target_new.insert(target_new.begin() + pos_single, c1);
                        target_new.insert(target_new.begin() + pos_single + 1, c2);
                        target_new = normalize_route(target_new);

                        vd crit_new_metrics = check_route_feasibility(crit_new, 0.0, crit_mode_truck);
                        vd target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += crit_new_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += crit_new_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += crit_new_metrics[2] - crit_metrics[2];
                        candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                        candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                        candidate.energy_violation += target_new_metrics[2] - target_metrics[2];

                        if (crit_mode_truck) {
                            candidate.truck_routes[crit_route_idx] = crit_new;
                            candidate.truck_route_times[crit_route_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                        } else {
                            int crit_drone_idx = crit_route_idx - h;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_idx] = target_new;
                            candidate.truck_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                        } else {
                            int target_drone_idx = target_idx - h;
                            if (target_drone_idx >= 0 && target_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[target_drone_idx] = target_new;
                                candidate.drone_route_times[target_drone_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(c1, c2), max(c1, c2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = solution_score(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) {
                            continue;
                        }
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_tabu_triple = key;
                            best_pair_a = c1;
                            best_pair_b = c2;
                            best_single = single;
                            best_pair_from_critical = true;
                            best_other_vehicle = target_veh;
                            best_other_is_truck = target_is_truck;
                        }
                    }
                }
            }
        };

        auto consider_single_vs_pair = [&](const vi& crit_route, bool crit_mode_truck, int crit_route_idx) {
            if (crit_route.size() <= 2) return;

            vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_mode_truck);
            vector<int> single_positions;
            for (int i = 0; i < (int)crit_route.size(); ++i)
                if (crit_route[i] != 0) single_positions.push_back(i);
            if (single_positions.empty()) return;

            auto near_enough = [&](int u, int v) {
                return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u && KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
            };

            for (int single_idx : single_positions) {
                int single = crit_route[single_idx];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == (crit_mode_truck ? crit_route_idx : h + crit_route_idx)) continue;
                    bool target_is_truck = target_veh < h;
                    if (!target_is_truck && !served_by_drone[single]) continue;

                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_idx]
                        : initial_solution.drone_routes[target_idx];
                    if (target_route.size() <= 3) continue;

                    vector<int> pair_positions;
                    for (int j = 0; j + 1 < (int)target_route.size(); ++j)
                        if (target_route[j] != 0 && target_route[j + 1] != 0) pair_positions.push_back(j);
                    if (pair_positions.empty()) continue;

                    vd target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);

                    for (int pair_idx : pair_positions) {
                        int b1 = target_route[pair_idx];
                        int b2 = target_route[pair_idx + 1];
                        if (!crit_mode_truck && (!served_by_drone[b1] || !served_by_drone[b2])) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(single, b1) || near_enough(b1, single) ||
                                      near_enough(single, b2) || near_enough(b2, single);
                            if (!ok) continue;
                        }

                        vi crit_new = crit_route;
                        crit_new.erase(crit_new.begin() + single_idx);
                        crit_new.insert(crit_new.begin() + single_idx, b1);
                        crit_new.insert(crit_new.begin() + single_idx + 1, b2);
                        crit_new = normalize_route(crit_new);

                        vi target_new = target_route;
                        target_new.erase(target_new.begin() + pair_idx);
                        target_new.erase(target_new.begin() + pair_idx);
                        target_new.insert(target_new.begin() + pair_idx, single);
                        target_new = normalize_route(target_new);

                        vd crit_new_metrics = check_route_feasibility(crit_new, 0.0, crit_mode_truck);
                        vd target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += crit_new_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += crit_new_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += crit_new_metrics[2] - crit_metrics[2];
                        candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                        candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                        candidate.energy_violation += target_new_metrics[2] - target_metrics[2];

                        if (crit_mode_truck) {
                            candidate.truck_routes[crit_route_idx] = crit_new;
                            candidate.truck_route_times[crit_route_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                        } else {
                            int crit_drone_idx = crit_route_idx - h;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_idx] = target_new;
                            candidate.truck_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                        } else {
                            int target_drone_idx = target_idx - h;
                            if (target_drone_idx >= 0 && target_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[target_drone_idx] = target_new;
                                candidate.drone_route_times[target_drone_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(b1, b2), max(b1, b2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = solution_score(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) {
                            continue;
                        }
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_tabu_triple = key;
                            best_pair_a = b1;
                            best_pair_b = b2;
                            best_single = single;
                            best_pair_from_critical = false;
                            best_other_vehicle = target_veh;
                            best_other_is_truck = target_is_truck;
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            consider_pair_vs_single(initial_solution.truck_routes[critical_idx], true, critical_idx);
            consider_single_vs_pair(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_pair_vs_single(initial_solution.drone_routes[critical_idx], false, critical_idx);
            consider_single_vs_pair(initial_solution.drone_routes[critical_idx], false, critical_idx);
        }

        if (!best_tabu_triple.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_21[best_tabu_triple] = current_iter + TABU_TENURE_21;

            // Debug N5
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            bool other_is_truck = best_other_is_truck;
            int other_idx = other_is_truck ? best_other_vehicle : best_other_vehicle - h;
            cout << "[N5] (" << (best_pair_from_critical ? "2,1" : "1,2") << ") swap pair ("
                 << best_pair_a << "," << best_pair_b << ") with customer " << best_single
                 << " between " << (crit_is_truck ? "truck" : "drone") << " #" << (critical_idx + 1)
                 << " and " << (other_is_truck ? "truck" : "drone") << " #" << (other_idx + 1)
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 6) {
        // Neighborhood 6: Swap two pairs of customers between routes
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_key;
        int best_pair_a1 = -1, best_pair_a2 = -1;
        int best_pair_b1 = -1, best_pair_b2 = -1;
        int best_other_vehicle = -1;
        bool best_other_is_truck = true;
        bool best_same_route = false;

        auto enumerate_pairs = [](const vi& route) {
            vector<int> starts;
            for (int i = 0; i + 1 < (int)route.size(); ++i) {
                if (route[i] != 0 && route[i + 1] != 0) starts.push_back(i);
            }
            return starts;
        };

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto near_enough = [&](int u, int v) {
            return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u &&
                   KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
        };

        auto consider_swap_pairs = [&](const vi& base_route, bool base_is_truck, int base_route_idx) {
            if (base_route.size() <= 3) return;

            vd base_metrics = check_route_feasibility(base_route, 0.0, base_is_truck);
            auto base_pairs = enumerate_pairs(base_route);
            if (base_pairs.empty()) return;

            for (int p : base_pairs) {
                int a1 = base_route[p];
                int a2 = base_route[p + 1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    bool target_is_truck = target_veh < h;
                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    bool same_route = (target_veh == (base_is_truck ? base_route_idx : base_route_idx + h));
                    const vi& target_route = same_route
                        ? base_route
                        : (target_is_truck
                               ? initial_solution.truck_routes[target_idx]
                               : initial_solution.drone_routes[target_idx]);

                    if (target_route.size() <= 3) continue;
                    auto target_pairs = enumerate_pairs(target_route);
                    if (target_pairs.empty()) continue;

                    vd target_metrics;
                    if (!same_route) {
                        target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);
                    }

                    for (int q : target_pairs) {
                        if (same_route && (q == p || q == p + 1 || p == q + 1)) continue;

                        int b1 = target_route[q];
                        int b2 = target_route[q + 1];

                        if (!target_is_truck && (!served_by_drone[a1] || !served_by_drone[a2])) continue;
                        if (!base_is_truck && (!served_by_drone[b1] || !served_by_drone[b2])) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(a1, b1) || near_enough(a1, b2) ||
                                      near_enough(a2, b1) || near_enough(a2, b2) ||
                                      near_enough(b1, a1) || near_enough(b2, a1) ||
                                      near_enough(b1, a2) || near_enough(b2, a2);
                            if (!ok) continue;
                        }

                        vector<int> tabu_key = {a1, a2, b1, b2};
                        sort(tabu_key.begin(), tabu_key.end());
                        auto it_tabu = tabu_list_22.find(tabu_key);
                        bool is_tabu = (it_tabu != tabu_list_22.end() && it_tabu->second > current_iter);

                        vi base_new = base_route;
                        vi target_new = target_route;

                        if (same_route) {
                            swap(base_new[p], base_new[q]);
                            swap(base_new[p + 1], base_new[q + 1]);
                        } else {
                            base_new[p] = b1;
                            base_new[p + 1] = b2;
                            target_new[q] = a1;
                            target_new[q + 1] = a2;
                        }
                        base_new = normalize_route(base_new);
                        target_new = normalize_route(target_new);

                        vd base_new_metrics = check_route_feasibility(base_new, 0.0, base_is_truck);
                        vd target_new_metrics;
                        if (!same_route) {
                            target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);
                        }

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += base_new_metrics[1] - base_metrics[1];
                        candidate.capacity_violation += base_new_metrics[3] - base_metrics[3];
                        candidate.energy_violation += base_new_metrics[2] - base_metrics[2];

                        if (base_is_truck) {
                            candidate.truck_routes[base_route_idx] = base_new;
                            candidate.truck_route_times[base_route_idx] = (base_new.size() > 1) ? base_new_metrics[0] : 0.0;
                        } else {
                            candidate.drone_routes[base_route_idx] = base_new;
                            candidate.drone_route_times[base_route_idx] = (base_new.size() > 1) ? base_new_metrics[0] : 0.0;
                        }

                        if (!same_route) {
                            candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                            candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                            candidate.energy_violation += target_new_metrics[2] - target_metrics[2];

                            if (target_is_truck) {
                                candidate.truck_routes[target_idx] = target_new;
                                candidate.truck_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            } else {
                                candidate.drone_routes[target_idx] = target_new;
                                candidate.drone_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (double t : candidate.truck_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        double candidate_score = solution_score(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) {
                            continue;
                        }
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_tabu_key = tabu_key;
                            best_pair_a1 = a1;
                            best_pair_a2 = a2;
                            best_pair_b1 = b1;
                            best_pair_b2 = b2;
                            best_other_vehicle = target_veh;
                            best_other_is_truck = target_is_truck;
                            best_same_route = same_route;
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            consider_swap_pairs(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_swap_pairs(initial_solution.drone_routes[critical_idx], false, critical_idx);
        }

        if (!best_tabu_key.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_22[best_tabu_key] = current_iter + TABU_TENURE_22;

            // Debug N6
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            if (best_same_route) {
                cout << "[N6] (2,2) swap within " << (crit_is_truck ? "truck" : "drone") << " #"
                     << (critical_idx + 1) << ": (" << best_pair_a1 << "," << best_pair_a2
                     << ") ↔ (" << best_pair_b1 << "," << best_pair_b2 << "), makespan: "
                     << ", score: " << solution_score(initial_solution)
                     << " -> " << solution_score(best_candidate_neighbor)
                     << ", iter " << current_iter << "\n";
            } else {
                int other_idx = best_other_is_truck ? best_other_vehicle : best_other_vehicle - h;
                cout << "[N6] (2,2) swap pairs (" << best_pair_a1 << "," << best_pair_a2 << ") ↔ ("
                     << best_pair_b1 << "," << best_pair_b2 << ") between "
                     << (crit_is_truck ? "truck" : "drone") << " #" << (critical_idx + 1)
                     << " and " << (best_other_is_truck ? "truck" : "drone") << " #" << (other_idx + 1)
                     << ", makespan: " << initial_solution.total_makespan << " -> "
                     << best_neighbor.total_makespan << ", iter " << current_iter << "\n";
            } */

            return best_neighbor;
        }
        return initial_solution;

    }  else if (neighbor_id == 7) {
        // Neighborhood 7: depth-2 ejection chain (i -> j -> k)
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_key;
        int best_veh_i = -1, best_veh_j = -1, best_veh_k = -1;
        int best_cust_removed = -1, best_cust_ejected = -1;

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto is_truck_vehicle = [&](int veh_id) { return veh_id < h; };
        auto fetch_route = [&](int veh_id) -> const vi& {
            return (veh_id < h) ? initial_solution.truck_routes[veh_id]
                                : initial_solution.drone_routes[veh_id - h];
        };

        auto get_metrics = [&](const vi& route, bool truck_mode) {
            return check_route_feasibility(route, 0.0, truck_mode);
        };

        auto is_near = [&](int u, int v) {
            if (KNN_ADJ.empty()) return true;
            if (u < 0 || v < 0) return false;
            if (u >= (int)KNN_ADJ.size()) return false;
            if (v >= (int)KNN_ADJ[u].size()) return false;
            return (KNN_ADJ[u][v] == 1);
        };

        const int MAX_ROUTE_TRIPLETS = min(50, (h + d) * max(0, h + d - 1) * max(0, h + d - 2) / 6);
        int triplets_evaluated = 0;
        bool stop_search = false;

        for (int veh_i = 0; veh_i < h + d && !stop_search; ++veh_i) {
            const vi& route_i_raw = fetch_route(veh_i);
            if (route_i_raw.size() <= 2) continue;
            vi route_i = normalize_route(route_i_raw);
            vd metrics_i = get_metrics(route_i, is_truck_vehicle(veh_i));

            vector<int> pos_i;
            for (int idx = 0; idx < (int)route_i.size(); ++idx)
                if (route_i[idx] != 0) pos_i.push_back(idx);
            if (pos_i.empty()) continue;

            for (int veh_j = 0; veh_j < h + d && !stop_search; ++veh_j) {
                if (veh_j == veh_i) continue;
                const vi& route_j_raw = fetch_route(veh_j);
                if (route_j_raw.size() <= 2) continue;
                vi route_j = normalize_route(route_j_raw);
                vd metrics_j = get_metrics(route_j, is_truck_vehicle(veh_j));

                vector<int> pos_j;
                vector<int> customers_j;
                for (int idx = 0; idx < (int)route_j.size(); ++idx) {
                    if (route_j[idx] != 0) {
                        pos_j.push_back(idx);
                        customers_j.push_back(route_j[idx]);
                    }
                }
                if (pos_j.empty()) continue;

                for (int veh_k = 0; veh_k < h + d; ++veh_k) {
                    if (veh_k == veh_i || veh_k == veh_j) continue;
                    if (triplets_evaluated >= MAX_ROUTE_TRIPLETS) { stop_search = true; break; }
                    ++triplets_evaluated;

                    const vi& route_k_raw = fetch_route(veh_k);
                    vi route_k = normalize_route(route_k_raw);
                    vd metrics_k = get_metrics(route_k, is_truck_vehicle(veh_k));

                    vector<int> pos_k_candidates;
                    for (int idx = 1; idx <= (int)route_k.size(); ++idx)
                        pos_k_candidates.push_back(idx);

                    if (pos_k_candidates.empty()) continue;

                    for (int pos_idx_i : pos_i) {
                        int cust_removed = route_i[pos_idx_i];
                        vi route_i_new = route_i;
                        route_i_new.erase(route_i_new.begin() + pos_idx_i);
                        route_i_new = normalize_route(route_i_new);
                        vd metrics_i_new = get_metrics(route_i_new, is_truck_vehicle(veh_i));

                        if (!KNN_ADJ.empty()) {
                            bool near_some = false;
                            for (int c : customers_j) {
                                if (is_near(cust_removed, c) || is_near(c, cust_removed)) { near_some = true; break; }
                            }
                            if (!customers_j.empty() && !near_some) continue;
                        }
                        if (!is_truck_vehicle(veh_j) && !served_by_drone[cust_removed]) continue;

                        for (int pos_idx_j : pos_j) {
                            int cust_ejected = route_j[pos_idx_j];
                            if (cust_removed == cust_ejected) continue;
                            if (!is_truck_vehicle(veh_k) && !served_by_drone[cust_ejected]) continue;

                            if (!KNN_ADJ.empty()) {
                                if (!(is_near(cust_removed, cust_ejected) || is_near(cust_ejected, cust_removed))) continue;
                            }

                            vi route_j_new = route_j;
                            route_j_new[pos_idx_j] = cust_removed;
                            route_j_new = normalize_route(route_j_new);
                            vd metrics_j_new = get_metrics(route_j_new, is_truck_vehicle(veh_j));

                            for (int insert_pos_k : pos_k_candidates) {
                                vi route_k_new = route_k;
                                if (find(route_k_new.begin(), route_k_new.end(), cust_ejected) != route_k_new.end()) continue;
                                int insert_index = min(insert_pos_k, (int)route_k_new.size());
                                route_k_new.insert(route_k_new.begin() + insert_index, cust_ejected);
                                route_k_new = normalize_route(route_k_new);
                                vd metrics_k_new = get_metrics(route_k_new, is_truck_vehicle(veh_k));

                                if (!KNN_ADJ.empty()) {
                                    int idx_new = -1;
                                    for (int idx = 0; idx < (int)route_k_new.size(); ++idx) {
                                        if (route_k_new[idx] == cust_ejected) { idx_new = idx; break; }
                                    }
                                    if (idx_new != -1 && idx_new > 0 && idx_new + 1 < (int)route_k_new.size()) {
                                        int prev = route_k_new[idx_new - 1];
                                        int next = route_k_new[idx_new + 1];
                                        if (!(is_near(cust_ejected, prev) || is_near(prev, cust_ejected) ||
                                              is_near(cust_ejected, next) || is_near(next, cust_ejected))) {
                                            continue;
                                        }
                                    }
                                }

                                Solution candidate = initial_solution;

                                candidate.deadline_violation += metrics_i_new[1] - metrics_i[1];
                                candidate.capacity_violation += metrics_i_new[3] - metrics_i[3];
                                candidate.energy_violation += metrics_i_new[2] - metrics_i[2];

                                candidate.deadline_violation += metrics_j_new[1] - metrics_j[1];
                                candidate.capacity_violation += metrics_j_new[3] - metrics_j[3];
                                candidate.energy_violation += metrics_j_new[2] - metrics_j[2];

                                candidate.deadline_violation += metrics_k_new[1] - metrics_k[1];
                                candidate.capacity_violation += metrics_k_new[3] - metrics_k[3];
                                candidate.energy_violation += metrics_k_new[2] - metrics_k[2];

                                if (is_truck_vehicle(veh_i)) {
                                    candidate.truck_routes[veh_i] = route_i_new;
                                    candidate.truck_route_times[veh_i] = (route_i_new.size() > 1) ? metrics_i_new[0] : 0.0;
                                } else {
                                    candidate.drone_routes[veh_i - h] = route_i_new;
                                    candidate.drone_route_times[veh_i - h] = (route_i_new.size() > 1) ? metrics_i_new[0] : 0.0;
                                }

                                if (is_truck_vehicle(veh_j)) {
                                    candidate.truck_routes[veh_j] = route_j_new;
                                    candidate.truck_route_times[veh_j] = (route_j_new.size() > 1) ? metrics_j_new[0] : 0.0;
                                } else {
                                    candidate.drone_routes[veh_j - h] = route_j_new;
                                    candidate.drone_route_times[veh_j - h] = (route_j_new.size() > 1) ? metrics_j_new[0] : 0.0;
                                }

                                if (is_truck_vehicle(veh_k)) {
                                    candidate.truck_routes[veh_k] = route_k_new;
                                    candidate.truck_route_times[veh_k] = (route_k_new.size() > 1) ? metrics_k_new[0] : 0.0;
                                } else {
                                    candidate.drone_routes[veh_k - h] = route_k_new;
                                    candidate.drone_route_times[veh_k - h] = (route_k_new.size() > 1) ? metrics_k_new[0] : 0.0;
                                }

                                candidate.total_makespan = 0.0;
                                for (double t : candidate.truck_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                                for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                                vector<int> tabu_key = {min(cust_removed, cust_ejected), max(cust_removed, cust_ejected)};
                                bool is_tabu = (tabu_list_ejection.count(tabu_key) &&
                                                tabu_list_ejection[tabu_key] > current_iter);

                                double candidate_score = solution_score(candidate);
                                if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                                candidate.deadline_violation <= 1e-8 &&
                                                candidate.capacity_violation <= 1e-8 &&
                                                candidate.energy_violation <= 1e-8)) {
                                    continue;
                                }
                                if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                    best_neighbor_cost_local = candidate_score;
                                    best_candidate_neighbor = candidate;
                                    best_tabu_key = tabu_key;
                                    best_veh_i = veh_i;
                                    best_veh_j = veh_j;
                                    best_veh_k = veh_k;
                                    best_cust_removed = cust_removed;
                                    best_cust_ejected = cust_ejected;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (!best_tabu_key.empty()) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_ejection[best_tabu_key] = current_iter + TABU_TENURE_EJECTION;

            cout.setf(std::ios::fixed);
            cout << setprecision(6);
            auto print_vehicle = [&](int v) {
                return (is_truck_vehicle(v) ? string("truck #") + to_string(v + 1)
                                            : string("drone #") + to_string(v - h + 1));
            };
            // Debug N7
            /* cout << "[N7] ejection chain: move " << best_cust_removed << " from " << print_vehicle(best_veh_i)
                 << ", replace " << best_cust_ejected << " on " << print_vehicle(best_veh_j)
                 << ", insert into " << print_vehicle(best_veh_k)
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    }
// ...existing code...
return initial_solution;
}

Solution local_search_all_vehicle(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 1e10;
    // Depending on neighbor_id, implement different neighborhood structures
    if (neighbor_id == 0) {
        // Relocate 1 customer from the critical (longest-time) vehicle route to another route of the same mode
        // 1) Identify critical vehicle (truck or drone) using precomputed times

        // Ensure tabu list is sized to (n+1) x (h+d)
        int veh_count = h + d;
        if ((int)tabu_list_10.size() != n + 1 || (veh_count > 0 && (int)tabu_list_10[0].size() != veh_count)) {
            tabu_list_10.assign(n + 1, vector<int>(max(0, veh_count), 0));
        }

        // Prepare neighborhood best tracking
        int best_target = -1; // vehicle index in unified space (0..h-1 trucks, h..h+d-1 drones)
        int best_cust = -1;   // moved customer id
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Lambda to normalize routes for comparison (detect no-ops)
            auto normalize_route = [](vi route) {
                if (route.empty()) return route;
                if (route.front() != 0) route.insert(route.begin(), 0);
                if (route.back() != 0) route.push_back(0);
                vi cleaned;
                cleaned.reserve(route.size());
                for (int node : route) {
                    if (!cleaned.empty() && cleaned.back() == node) continue;
                    cleaned.push_back(node);
                }
                return cleaned;
            };

            vd crit_route_time_feas = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust = base_route[p];
                // Try relocating cust to other vehicles of the same mode
                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    bool is_tabu = (tabu_list_10[cust][target_veh] > current_iter);
                    if (target_veh == critical_vehicle_id){
                        for (int p2 = 1; p2 < (int)base_route.size(); ++p2) {
                            // attempt to relocate cust to position p2 in the same route
                            if (p2 == p) continue; // no-op
                            // Build new candidate solution with relocation
                            Solution candidate = initial_solution;
                            vi new_base_route = base_route;
                            new_base_route.erase(new_base_route.begin() + p);
                            new_base_route.insert(new_base_route.begin() + p2 - (p2 > p ? 1 : 0), cust);
                            vi new_norm = normalize_route(new_base_route);
                            // Compute cost of candidate
                            vd new_crit_route_time_feas = is_truck_mode
                                ? check_route_feasibility(new_norm, 0.0, true)
                                : check_route_feasibility(new_norm, 0.0, false);
                            // If tabu, only consider if it improves over best known cost and feasible
                            candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                            candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                            candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = new_norm;
                                candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = new_norm;
                                candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                            }
                            candidate.total_makespan = 0.0;
                            for (int i = 0; i < h; ++i) {
                                candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                            }
                            for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                                candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                            }
                            double candidate_score = solution_score(candidate);
                            bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                       candidate.capacity_violation <= 1e-8 &&
                                                       candidate.energy_violation <= 1e-8;
                            if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                                continue; // skip tabu move
                            }
                            // Update best if improved
                            if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = candidate_score;
                                best_candidate_neighbor = candidate;
                                best_target = target_veh;
                                best_cust = cust;
                            }
                        }
                        // also consider relocating to the end of the route
                        Solution candidate = initial_solution;
                        vi new_base_route = base_route;
                        new_base_route.erase(new_base_route.begin() + p);
                        new_base_route.push_back(cust);
                        new_base_route.push_back(0);
                        vi new_norm = normalize_route(new_base_route);
                        // Compute cost of candidate
                        vd new_crit_route_time_feas = is_truck_mode
                            ? check_route_feasibility(new_norm, 0.0, true)
                            : check_route_feasibility(new_norm, 0.0, false);
                        // If tabu, only consider if it improves over best known cost and feasible
                        candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                        candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                        candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = new_norm;
                            candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = new_norm;
                            candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                        }
                        candidate.total_makespan = 0.0;
                        for (int i = 0; i < h; ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                        }
                        for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                        }
                        double candidate_score = solution_score_total_time(candidate);
                        bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                   candidate.capacity_violation <= 1e-8 &&
                                                   candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                            continue; // skip tabu move
                        }
                        // Update best if improved
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_target = target_veh;
                            best_cust = cust;
                        }
                    }
                    if (target_veh == critical_vehicle_id) continue; // skip same vehicle here
                    const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh - 0]
                                                              : initial_solution.drone_routes[target_veh - h];
                    vd target_route_time_feas = (target_veh < h)
                        ? check_route_feasibility(target_route, 0.0, true)
                        : check_route_feasibility(target_route, 0.0, false);                                          
                    // Try all insertion positions in target route
                    // Remove cust from base route
                    vi new_base_route = base_route;
                    new_base_route.erase(new_base_route.begin() + p);
                    vd new_crit_route_time_feas = is_truck_mode
                        ? check_route_feasibility(new_base_route, 0.0, true)
                        : check_route_feasibility(new_base_route, 0.0, false);
                    for (int insert_pos = 1; insert_pos < (int)target_route.size(); ++insert_pos) {
                        // Build new candidate solution with relocation
                        Solution candidate = initial_solution;
                        // Insert cust into target route
                        vi new_target_route = target_route;
                        new_target_route.insert(new_target_route.begin() + insert_pos, cust);
                        vd new_target_route_time_feas = (target_veh < h)
                            ? check_route_feasibility(new_target_route, 0.0, true)
                            : check_route_feasibility(new_target_route, 0.0, false);

                        // Compute cost of candidate
                        candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                        candidate.deadline_violation += new_target_route_time_feas[1] - target_route_time_feas[1];
                        candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                        candidate.capacity_violation += new_target_route_time_feas[3] - target_route_time_feas[3];
                        candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                        candidate.energy_violation += new_target_route_time_feas[2] - target_route_time_feas[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = new_base_route;
                            candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = new_base_route;
                            candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                        }
                        if (target_veh < h) {
                            candidate.truck_routes[target_veh] = new_target_route;
                            candidate.truck_route_times[target_veh] = new_target_route_time_feas[0];
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target_route;
                            candidate.drone_route_times[target_veh - h] = new_target_route_time_feas[0];
                        }
                        candidate.total_makespan = 0.0;
                        for (int i = 0; i < h; ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                        }
                        
                        for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                            candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                        }
                        double candidate_score = solution_score_total_time(candidate);
                        bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                   candidate.capacity_violation <= 1e-8 &&
                                                   candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                            continue; // skip tabu move
                        }
                        // Update best if improved
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_target = target_veh;
                            best_cust = cust;
                        }
                    }
                    // Also try inserting at the end of the target route
                    vi new_target_route = target_route;
                    new_target_route.push_back(cust);
                    new_target_route.push_back(0);
                    vd new_target_route_time_feas = (target_veh < h)
                        ? check_route_feasibility(new_target_route, 0.0, true)
                        : check_route_feasibility(new_target_route, 0.0, false);
                    // Build new candidate solution with relocation
                    Solution candidate = initial_solution;
                    // Compute cost of candidate
                    candidate.deadline_violation += new_crit_route_time_feas[1] - crit_route_time_feas[1];
                    candidate.deadline_violation += new_target_route_time_feas[1] - target_route_time_feas[1];
                    candidate.capacity_violation += new_crit_route_time_feas[3] - crit_route_time_feas[3];
                    candidate.capacity_violation += new_target_route_time_feas[3] - target_route_time_feas[3];
                    candidate.energy_violation += new_crit_route_time_feas[2] - crit_route_time_feas[2];
                    candidate.energy_violation += new_target_route_time_feas[2] - target_route_time_feas[2];
                    if (is_truck_mode) {
                        candidate.truck_routes[critical_vehicle_id] = new_base_route;
                        candidate.truck_route_times[critical_vehicle_id] = new_crit_route_time_feas[0];
                    } else {
                        candidate.drone_routes[critical_vehicle_id - h] = new_base_route;
                        candidate.drone_route_times[critical_vehicle_id - h] = new_crit_route_time_feas[0];
                    }
                    if (target_veh < h) {
                        candidate.truck_routes[target_veh] = new_target_route;
                        candidate.truck_route_times[target_veh] = new_target_route_time_feas[0];
                    } else {
                        candidate.drone_routes[target_veh - h] = new_target_route;
                        candidate.drone_route_times[target_veh - h] = new_target_route_time_feas[0];
                    }
                    candidate.total_makespan = 0.0;
                    for (int i = 0; i < h; ++i) {
                        candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                    }
                    for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i) {
                        candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);
                    }
                    double candidate_score = solution_score_total_time(candidate);
                    bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                               candidate.capacity_violation <= 1e-8 &&
                                               candidate.energy_violation <= 1e-8;
                    if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                        continue; // skip tabu move
                    }
                    // Update best if improved
                    if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                        best_neighbor_cost_local = candidate_score;
                        best_candidate_neighbor = candidate;
                        best_target = target_veh;
                        best_cust = cust;
                    }
                }
            }
        };
        for (int veh = 0; veh < h + d; ++veh) {
            bool is_truck = veh < h;
            const vi& route = is_truck ? initial_solution.truck_routes[veh]
                                       : initial_solution.drone_routes[veh - h];
            consider_relocate(route, is_truck, veh);
        }

        // After evaluating all candidates, update tabu list if we found an improving move
        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Update tabu list
            tabu_list_10[best_cust][best_target] = current_iter + TABU_TENURE_10;
        }
        // Debug:
        //cout << "[N0] relocate customer " << best_cust << " to vehicle " << best_target << " score: " << solution_score(initial_solution) << " -> " << solution_score(best_neighbor) << "\n";
        return best_neighbor;
    } else if (neighbor_id == 1) {
        // Neighborhood 1: swap two customers, allowing cross-mode exchanges

        if ((int)tabu_list_11.size() != n + 1 || (n + 1 > 0 && (int)tabu_list_11[0].size() != n + 1)) {
            tabu_list_11.assign(n + 1, vector<int>(n + 1, 0));
        }

        int best_cust_a = -1, best_cust_b = -1;
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 2) return; // nothing to swap

            vd crit_metrics = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            vector<int> crit_positions;
            for (int i = 0; i < (int)base_route.size(); ++i) {
                if (base_route[i] != 0) crit_positions.push_back(i);
            }
            if (crit_positions.empty()) return;

            for (int idx1 = 0; idx1 < (int)crit_positions.size(); ++idx1) {
                int pos1 = crit_positions[idx1];
                int cust_a = base_route[pos1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    bool target_is_truck = target_veh < h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_veh]
                        : initial_solution.drone_routes[target_veh - h];

                    if (target_route.size() <= 2 && target_veh != critical_vehicle_id) continue;

                    vd target_metrics;
                    vector<int> target_positions;
                    if (target_veh == critical_vehicle_id) {
                        target_positions = crit_positions;
                    } else {
                        target_metrics = target_is_truck
                            ? check_route_feasibility(target_route, 0.0, true)
                            : check_route_feasibility(target_route, 0.0, false);
                        for (int j = 0; j < (int)target_route.size(); ++j) {
                            if (target_route[j] != 0) target_positions.push_back(j);
                        }
                        if (target_positions.empty()) continue;
                    }

                    int start_idx2 = (target_veh == critical_vehicle_id) ? idx1 + 1 : 0;
                    for (int idx2 = start_idx2; idx2 < (int)target_positions.size(); ++idx2) {
                        int pos2 = target_positions[idx2];
                        int cust_b = (target_veh == critical_vehicle_id) ? base_route[pos2] : target_route[pos2];
                        if (cust_a == cust_b) continue;
                        if (KNN_ADJ.size() > 0) {
                            // If using KNN, only consider swaps between neighbors
                            if (KNN_ADJ[cust_a][cust_b] == 0 && KNN_ADJ[cust_b][cust_a] == 0) {
                                continue;
                            }
                        }
                        if (!is_truck_mode && target_veh != critical_vehicle_id && !served_by_drone[cust_b]) continue;
                        if (!target_is_truck && !served_by_drone[cust_a]) continue;
                        if (target_veh == critical_vehicle_id && !is_truck_mode && !served_by_drone[cust_a] && !served_by_drone[cust_b]) continue;

                        int u = min(cust_a, cust_b);
                        int v = max(cust_a, cust_b);
                        bool is_tabu = (tabu_list_11.size() > (size_t)u && tabu_list_11[u].size() > (size_t)v &&
                                        tabu_list_11[u][v] > current_iter);

                        vi crit_route_new = base_route;
                        vi target_route_new = target_route;

                        if (target_veh == critical_vehicle_id) {
                            swap(crit_route_new[pos1], crit_route_new[pos2]);
                        } else {
                            swap(crit_route_new[pos1], target_route_new[pos2]);
                        }

                        vd new_crit_metrics = is_truck_mode
                            ? check_route_feasibility(crit_route_new, 0.0, true)
                            : check_route_feasibility(crit_route_new, 0.0, false);

                        vd new_target_metrics;
                        if (target_veh != critical_vehicle_id) {
                            new_target_metrics = target_is_truck
                                ? check_route_feasibility(target_route_new, 0.0, true)
                                : check_route_feasibility(target_route_new, 0.0, false);
                        }

                        Solution candidate = initial_solution;

                        candidate.deadline_violation += new_crit_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += new_crit_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += new_crit_metrics[2] - crit_metrics[2];

                        if (target_veh != critical_vehicle_id) {
                            candidate.deadline_violation += new_target_metrics[1] - target_metrics[1];
                            candidate.capacity_violation += new_target_metrics[3] - target_metrics[3];
                            candidate.energy_violation += new_target_metrics[2] - target_metrics[2];
                        }

                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = crit_route_new;
                            candidate.truck_route_times[critical_vehicle_id] = (crit_route_new.size() > 1) ? new_crit_metrics[0] : 0.0;
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = crit_route_new;
                            candidate.drone_route_times[critical_vehicle_id - h] = (crit_route_new.size() > 1) ? new_crit_metrics[0] : 0.0;
                        }

                        if (target_veh != critical_vehicle_id) {
                            if (target_is_truck) {
                                candidate.truck_routes[target_veh] = target_route_new;
                                candidate.truck_route_times[target_veh] = (target_route_new.size() > 1) ? new_target_metrics[0] : 0.0;
                            } else {
                                candidate.drone_routes[target_veh - h] = target_route_new;
                                candidate.drone_route_times[target_veh - h] = (target_route_new.size() > 1) ? new_target_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (int i = 0; i < h; ++i) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                        for (int i = 0; i < (int)candidate.drone_route_times.size(); ++i)
                            candidate.total_makespan = max(candidate.total_makespan, candidate.drone_route_times[i]);

                        double candidate_score = solution_score_total_time(candidate);
                        bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                                   candidate.capacity_violation <= 1e-8 &&
                                                   candidate.energy_violation <= 1e-8;

                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) {
                            continue;
                        }

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_cust_a = cust_a;
                            best_cust_b = cust_b;
                        }
                    }
                }
            }
        };

        for (int veh = 0; veh < h + d; ++veh) {
            bool is_truck = veh < h;
            const vi& route = is_truck ? initial_solution.truck_routes[veh]
                                       : initial_solution.drone_routes[veh - h];
            consider_swap(route, is_truck, veh);
        }

        if (best_cust_a != -1 && best_cust_b != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            int u = min(best_cust_a, best_cust_b);
            int v = max(best_cust_a, best_cust_b);
            tabu_list_11[u][v] = current_iter + TABU_TENURE_11;

            // Debug: print swap info
             /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N1] swap " << best_cust_a << " and " << best_cust_b
                 << ", score: " << solution_score_total_time(initial_solution)
                 << " -> " << best_neighbor_cost_local
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 2) {
        // Neighborhood 2: relocate a consecutive pair (2,0)-move from the critical vehicle to another vehicle
        // Structure mirrors neighborhood 0: identify critical vehicle, enumerate candidate relocations,
        // respect tabu_list_20 keyed by (min(c1,c2), max(c1,c2), target_vehicle).

        // Prepare best tracking
        Solution best_candidate_neighbor = initial_solution;
        double best_neighbor_cost_local = 1e10;
        int best_c1 = -1, best_c2 = -1;
        int best_target_vehicle = -1;
        int best_src_pos = -1, best_target_pos = -1;

        auto consider_relocate_pair = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 3) return; // nothing to do if fewer than two customers

            auto normalize_route = [](const vi& route) -> vi {
                vi normalized;
                for (int node : route) {
                    if (normalized.empty() || node != 0 || normalized.back() != 0) normalized.push_back(node);
                }
                if (normalized.empty()) return vi{0};
                if (normalized.front() != 0) normalized.insert(normalized.begin(), 0);
                if (normalized.back() != 0) normalized.push_back(0);
                return normalized;
            };

            vi orig = normalize_route(base_route);
            if (orig.size() <= 3) return;
            vd orig_metrics = is_truck_mode
                ? check_route_feasibility(orig, 0.0, true)
                : check_route_feasibility(orig, 0.0, false);

            vector<int> pos;
            for (int i = 0; i + 1 < (int)orig.size(); ++i) {
                if (orig[i] != 0 && orig[i + 1] != 0) pos.push_back(i);
            }
            if (pos.empty()) return;

            auto apply_candidate = [&](const vi& crit_route, const vd& crit_metrics_local,
                                      const optional<pair<int, vi>>& target_change) {
                Solution candidate = initial_solution;
                candidate.deadline_violation += crit_metrics_local[1] - orig_metrics[1];
                candidate.capacity_violation += crit_metrics_local[3] - orig_metrics[3];
                candidate.energy_violation += crit_metrics_local[2] - orig_metrics[2];
                if (is_truck_mode) {
                    candidate.truck_routes[critical_vehicle_id] = crit_route;
                    candidate.truck_route_times[critical_vehicle_id] = (crit_route.size() > 1) ? crit_metrics_local[0] : 0.0;
                } else {
                    candidate.drone_routes[critical_vehicle_id - h] = crit_route;
                    candidate.drone_route_times[critical_vehicle_id - h] = (crit_route.size() > 1) ? crit_metrics_local[0] : 0.0;
                }

                if (target_change.has_value()) {
                    int target_vehicle = target_change->first;
                    const vi& new_target_route = target_change->second;
                    bool target_is_truck = target_vehicle < h;
                    vd target_metrics_before = target_is_truck
                        ? check_route_feasibility(target_is_truck ? initial_solution.truck_routes[target_vehicle]
                                                                 : initial_solution.drone_routes[target_vehicle - h], 0.0, target_is_truck)
                        : check_route_feasibility(initial_solution.drone_routes[target_vehicle - h], 0.0, false);
                    vd target_metrics_after = target_is_truck
                        ? check_route_feasibility(new_target_route, 0.0, true)
                        : check_route_feasibility(new_target_route, 0.0, false);
                    candidate.deadline_violation += target_metrics_after[1] - target_metrics_before[1];
                    candidate.capacity_violation += target_metrics_after[3] - target_metrics_before[3];
                    candidate.energy_violation += target_metrics_after[2] - target_metrics_before[2];
                    if (target_is_truck) {
                        candidate.truck_routes[target_vehicle] = new_target_route;
                        candidate.truck_route_times[target_vehicle] = (new_target_route.size() > 1) ? target_metrics_after[0] : 0.0;
                    } else {
                        candidate.drone_routes[target_vehicle - h] = new_target_route;
                        candidate.drone_route_times[target_vehicle - h] = (new_target_route.size() > 1) ? target_metrics_after[0] : 0.0;
                    }
                }

                candidate.total_makespan = 0.0;
                for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                for (double tt : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, tt);
                return candidate;
            };

            for (int p : pos) {
                int c1 = orig[p];
                int c2 = orig[p + 1];

                vi reduced = orig;
                reduced.erase(reduced.begin() + p, reduced.begin() + p + 2);
                reduced = normalize_route(reduced);
                vd reduced_metrics = is_truck_mode
                    ? check_route_feasibility(reduced, 0.0, true)
                    : check_route_feasibility(reduced, 0.0, false);

                for (int ip = 1; ip <= (int)reduced.size(); ++ip) {
                    vi r = reduced;
                    r.insert(r.begin() + ip, c1);
                    r.insert(r.begin() + ip + 1, c2);
                    vi r_norm = normalize_route(r);
                    if (r_norm == orig) continue;

                    vd new_metrics = is_truck_mode
                        ? check_route_feasibility(r_norm, 0.0, true)
                        : check_route_feasibility(r_norm, 0.0, false);
                    Solution candidate = apply_candidate(r_norm, new_metrics, nullopt);

                    vector<int> key = { min(c1, c2), max(c1, c2), critical_vehicle_id };
                    auto it = tabu_list_20.find(key);
                    bool is_tabu = (it != tabu_list_20.end() && it->second > current_iter);
                    double candidate_score = solution_score_total_time(candidate);
                    bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                               candidate.capacity_violation <= 1e-8 &&
                                               candidate.energy_violation <= 1e-8;
                    if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) continue;

                    if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                        best_neighbor_cost_local = candidate_score;
                        best_candidate_neighbor = candidate;
                        best_c1 = c1; best_c2 = c2;
                        best_target_vehicle = critical_vehicle_id;
                        best_src_pos = p;
                        best_target_pos = ip;
                    }
                }

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == critical_vehicle_id) continue;
                    bool target_is_truck = target_veh < h;
                    vi target_route = target_is_truck
                        ? initial_solution.truck_routes[target_veh]
                        : initial_solution.drone_routes[target_veh - h];
                    target_route = normalize_route(target_route);

                    for (int insert_pos = 1; insert_pos <= (int)target_route.size(); ++insert_pos) {
                        vi new_target = target_route;
                        new_target.insert(new_target.begin() + insert_pos, c1);
                        new_target.insert(new_target.begin() + insert_pos + 1, c2);
                        new_target = normalize_route(new_target);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation = initial_solution.deadline_violation;
                        candidate.capacity_violation = initial_solution.capacity_violation;
                        candidate.energy_violation = initial_solution.energy_violation;

                        // build candidate with reduced critical route + modified target route
                        vd crit_metrics_ready = reduced_metrics;
                        vd target_metrics_new = target_is_truck
                            ? check_route_feasibility(new_target, 0.0, true)
                            : check_route_feasibility(new_target, 0.0, false);
                        vd target_metrics_old = target_is_truck
                            ? check_route_feasibility(target_route, 0.0, true)
                            : check_route_feasibility(target_route, 0.0, false);

                        candidate.deadline_violation += crit_metrics_ready[1] - orig_metrics[1];
                        candidate.deadline_violation += target_metrics_new[1] - target_metrics_old[1];
                        candidate.capacity_violation += crit_metrics_ready[3] - orig_metrics[3];
                        candidate.capacity_violation += target_metrics_new[3] - target_metrics_old[3];
                        candidate.energy_violation += crit_metrics_ready[2] - orig_metrics[2];
                        candidate.energy_violation += target_metrics_new[2] - target_metrics_old[2];

                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = reduced;
                            candidate.truck_route_times[critical_vehicle_id] = (reduced.size() > 1) ? crit_metrics_ready[0] : 0.0;
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = reduced;
                            candidate.drone_route_times[critical_vehicle_id - h] = (reduced.size() > 1) ? crit_metrics_ready[0] : 0.0;
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_veh] = new_target;
                            candidate.truck_route_times[target_veh] = (new_target.size() > 1) ? target_metrics_new[0] : 0.0;
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target;
                            candidate.drone_route_times[target_veh - h] = (new_target.size() > 1) ? target_metrics_new[0] : 0.0;
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double tt : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, tt);

                        vector<int> key = { min(c1, c2), max(c1, c2), target_veh };
                        auto it = tabu_list_20.find(key);
                        bool is_tabu = (it != tabu_list_20.end() && it->second > current_iter);
                        double candidate_score = solution_score_total_time(candidate);
                        bool feasible = candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && feasible)) continue;

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_c1 = c1; best_c2 = c2;
                            best_target_vehicle = target_veh;
                            best_src_pos = p;
                            best_target_pos = insert_pos;
                        }
                    }
                }
            }
        };

        for (int critical_idx = 0; critical_idx < h + d; ++critical_idx) {
            bool crit_is_truck = critical_idx < h;
            const vi& route = crit_is_truck
                ? initial_solution.truck_routes[critical_idx]
                : initial_solution.drone_routes[critical_idx - h];
            consider_relocate_pair(route, crit_is_truck, critical_idx);
        }

        // apply best move if found
        if (best_c1 != -1 && best_c2 != -1 && best_target_vehicle != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            // update tabu
            vector<int> key = { min(best_c1, best_c2), max(best_c1, best_c2), best_target_vehicle };
            tabu_list_20[key] = current_iter + TABU_TENURE_20;
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Debug:
            /*  cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N2] relocate pair (" << best_c1 << "," << best_c2 << ") to vehicle " << best_target_vehicle
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */
            // return the chosen neighbor (already fully assembled in best_candidate_neighbor)*/
            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 3) {
        // Neighborhood 3: 2-opt within each subroute (between depot nodes) for trucks or drones.
        // Finds the best 2-opt move across all routes that yields the largest local time drop.

        if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_edge_u = -1, best_edge_v = -1;
        int best_i = -1, best_j = -1;

        auto normalize_route = [](vi r) -> vi {
            if (r.empty()) return r;
            if (r.front() != 0) r.insert(r.begin(), 0);
            if (r.back() != 0) r.push_back(0);
            vi cleaned; cleaned.reserve(r.size());
            for (int node : r) {
                if (!cleaned.empty() && cleaned.back() == 0 && node == 0) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto consider_2opt = [&](const vi& base_route, bool is_truck_mode, int route_idx) {
            if (base_route.size() <= 3) return;

            vd route_metrics = check_route_feasibility(base_route, 0.0, is_truck_mode);
            int m = (int)base_route.size();
            int start = 0;
            while (start < m) {
                while (start < m && base_route[start] == 0) ++start;
                if (start >= m) break;
                int seg_end = start;
                while (seg_end + 1 < m && base_route[seg_end + 1] != 0) ++seg_end;

                for (int i = start; i < seg_end; ++i) {
                    for (int j = i + 1; j <= seg_end; ++j) {
                        vi new_route = base_route;
                        reverse(new_route.begin() + i, new_route.begin() + j + 1);
                        if (new_route == base_route) continue;

                        new_route = normalize_route(new_route);
                        vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck_mode);
                        int u = min(base_route[i], base_route[j]);
                        int v = max(base_route[i], base_route[j]);
                        if (u < 0 || v < 0) continue;
                        bool is_tabu = (tabu_list_2opt.size() > (size_t)u &&
                                        tabu_list_2opt[u].size() > (size_t)v &&
                                        tabu_list_2opt[u][v] > current_iter);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += new_metrics[1] - route_metrics[1];
                        candidate.capacity_violation += new_metrics[3] - route_metrics[3];
                        candidate.energy_violation += new_metrics[2] - route_metrics[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[route_idx] = new_route;
                            candidate.truck_route_times[route_idx] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        } else {
                            int drone_route_idx = route_idx - h;
                            if (drone_route_idx >= 0 && drone_route_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[drone_route_idx] = new_route;
                                candidate.drone_route_times[drone_route_idx] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                            }
                        }
                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                        double candidate_score = solution_score_total_time(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) continue;

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_edge_u = u;
                            best_edge_v = v;
                            best_i = i;
                            best_j = j;
                        }
                    }
                }
                start = seg_end + 1;
            }
        };

        for (int critical_idx = 0; critical_idx < h + d; ++critical_idx) {
            bool crit_is_truck = critical_idx < h;
            const vi& route = crit_is_truck
                ? initial_solution.truck_routes[critical_idx]
                : initial_solution.drone_routes[critical_idx - h];
            consider_2opt(route, crit_is_truck, critical_idx);
        }

        if (best_edge_u != -1 && best_edge_v != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_2opt[best_edge_u][best_edge_v] = current_iter + TABU_TENURE_2OPT;

            // Debug N3
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N3] 2-opt on " << (best_is_truck ? "truck" : "drone") << " #"
                 << (crit_is_truck ? critical_idx + 1 : critical_idx + 1)
                 << " between positions " << best_i << " and " << best_j
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 4) {
        if ((int)tabu_list_2opt_star.size() != n + 1 || ((int)tabu_list_2opt_star.size() > 0 && (int)tabu_list_2opt_star[0].size() != n + 1)) {
            tabu_list_2opt_star.assign(n + 1, vector<int>(n + 1, 0));
        }

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_ua = -1, best_va = -1, best_ub = -1, best_vb = -1;

        auto enumerate_segments = [](const vi& route) {
            vector<pair<int,int>> segs;
            int m = (int)route.size();
            int start = 0;
            while (start < m) {
                while (start < m && route[start] == 0) ++start;
                if (start >= m) break;
                int end = start;
                while (end + 1 < m && route[end + 1] != 0) ++end;
                segs.emplace_back(start, end);
                start = end + 1;
            }
            return segs;
        };

        auto evaluate_two_opt_star = [&](const vi& crit_route_raw, bool crit_is_truck, int crit_idx,
                                         const vi& other_route_raw, bool other_is_truck, int other_idx) {
            vi crit_route = normalize_route(crit_route_raw);
            if (crit_route.size() <= 3) return;
            vi other_route = normalize_route(other_route_raw);
            if (other_route.size() <= 3) return;
            vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_is_truck);
            auto crit_segs = enumerate_segments(crit_route);
            vd other_metrics = check_route_feasibility(other_route, 0.0, other_is_truck);
            auto other_segs = enumerate_segments(other_route);

            for (const auto& segA : crit_segs) {
                for (int i = segA.first; i < segA.second; ++i) {
                    int a1 = crit_route[i], a2 = crit_route[i + 1];
                    int ua = min(a1, a2), va = max(a1, a2);
                    if (a1 == 0 || a2 == 0) continue;

                    for (const auto& segB : other_segs) {
                        for (int j = segB.first; j < segB.second; ++j) {
                            int b1 = other_route[j], b2 = other_route[j + 1];
                            int ub = min(b1, b2), vb = max(b1, b2);
                            if (b1 == 0 || b2 == 0) continue;

                            bool is_tabu = (tabu_list_2opt_star[ua][va] > current_iter) ||
                                           (tabu_list_2opt_star[ub][vb] > current_iter);

                            vi crit_new = crit_route;
                            vi other_new = other_route;

                            vi tailA(crit_new.begin() + i + 1, crit_new.begin() + segA.second + 1);
                            vi tailB(other_new.begin() + j + 1, other_new.begin() + segB.second + 1);

                            crit_new.erase(crit_new.begin() + i + 1, crit_new.begin() + segA.second + 1);
                            other_new.erase(other_new.begin() + j + 1, other_new.begin() + segB.second + 1);

                            crit_new.insert(crit_new.begin() + i + 1, tailB.begin(), tailB.end());
                            other_new.insert(other_new.begin() + j + 1, tailA.begin(), tailA.end());

                            crit_new = normalize_route(crit_new);
                            other_new = normalize_route(other_new);
                            if (crit_new == crit_route && other_new == other_route) continue;

                            vd crit_metrics_new = check_route_feasibility(crit_new, 0.0, crit_is_truck);
                            vd other_metrics_new = check_route_feasibility(other_new, 0.0, other_is_truck);

                            Solution candidate = initial_solution;
                            candidate.deadline_violation += crit_metrics_new[1] - crit_metrics[1];
                            candidate.capacity_violation += crit_metrics_new[3] - crit_metrics[3];
                            candidate.energy_violation += crit_metrics_new[2] - crit_metrics[2];
                            candidate.deadline_violation += other_metrics_new[1] - other_metrics[1];
                            candidate.capacity_violation += other_metrics_new[3] - other_metrics[3];
                            candidate.energy_violation += other_metrics_new[2] - other_metrics[2];

                            if (crit_is_truck) {
                                candidate.truck_routes[crit_idx] = crit_new;
                                candidate.truck_route_times[crit_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                            } else {
                                int crit_drone_idx = crit_idx - h;
                                if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[crit_drone_idx] = crit_new;
                                    candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                                }
                            }

                            if (other_is_truck) {
                                candidate.truck_routes[other_idx] = other_new;
                                candidate.truck_route_times[other_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                            } else {
                                int other_drone_idx = other_idx - h;
                                if (other_drone_idx >= 0 && other_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[other_drone_idx] = other_new;
                                    candidate.drone_route_times[other_drone_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                                }
                            }

                            candidate.total_makespan = 0.0;
                            for (int i = 0; i < h; ++i) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                            for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                            double candidate_score = solution_score_total_time(candidate);
                            if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                             candidate.deadline_violation <= 1e-8 &&
                                             candidate.capacity_violation <= 1e-8 &&
                                             candidate.energy_violation <= 1e-8)) {
                                continue;
                            }

                            if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = candidate_score;
                                best_candidate_neighbor = candidate;
                                best_ua = ua; best_va = va;
                                best_ub = ub; best_vb = vb;
                            }
                        }
                    }
                }
            }
        };

        for (int crit_idx = 0; crit_idx < h + d; ++crit_idx) {
            bool crit_is_truck = crit_idx < h;
            const vi& crit_route_raw = crit_is_truck
                ? initial_solution.truck_routes[crit_idx]
                : initial_solution.drone_routes[crit_idx - h];
            for (int other_idx = 0; other_idx < h + d; ++other_idx) {
                if (other_idx == crit_idx) continue;
                bool other_is_truck = other_idx < h;
                const vi& other_route_raw = other_is_truck
                    ? initial_solution.truck_routes[other_idx]
                    : initial_solution.drone_routes[other_idx - h];
                evaluate_two_opt_star(crit_route_raw, crit_is_truck, crit_idx,
                                      other_route_raw, other_is_truck, other_idx);
            }
        }

        if (best_ua != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_2opt_star[best_ua][best_va] = current_iter + TABU_TENURE_2OPT_STAR;
            tabu_list_2opt_star[best_ub][best_vb] = current_iter + TABU_TENURE_2OPT_STAR;

            //Debug N4
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N4] 2-opt* cuts (" << best_ua << "," << best_va << ") & (" << best_ub << "," << best_vb << ")"
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 5) {

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_triple;
        int best_pair_a = -1, best_pair_b = -1, best_single = -1;
        bool best_pair_from_critical = true;
        int best_other_vehicle = -1;
        bool best_other_is_truck = true;

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto consider_pair_vs_single = [&](const vi& crit_route, bool crit_mode_truck, int crit_route_idx) {
            if (crit_route.size() <= 3) return;

            vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_mode_truck);
            vector<int> pair_positions;
            for (int i = 0; i + 1 < (int)crit_route.size(); ++i)
                if (crit_route[i] != 0 && crit_route[i + 1] != 0) pair_positions.push_back(i);
            if (pair_positions.empty()) return;

            auto near_enough = [&](int u, int v) {
                return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u && KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
            };

            for (int pair_idx : pair_positions) {
                int c1 = crit_route[pair_idx];
                int c2 = crit_route[pair_idx + 1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == (crit_mode_truck ? crit_route_idx : h + crit_route_idx)) continue;
                    bool target_is_truck = target_veh < h;
                    if (!target_is_truck && (!served_by_drone[c1] || !served_by_drone[c2])) continue;

                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_idx]
                        : initial_solution.drone_routes[target_idx];
                    if (target_route.size() <= 2) continue;

                    vector<int> target_positions;
                    for (int j = 0; j < (int)target_route.size(); ++j)
                        if (target_route[j] != 0) target_positions.push_back(j);
                    if (target_positions.empty()) continue;

                    vd target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);

                    for (int pos_single : target_positions) {
                        int single = target_route[pos_single];
                        if (!crit_mode_truck && !served_by_drone[single]) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(c1, single) || near_enough(single, c1) ||
                                      near_enough(c2, single) || near_enough(single, c2);
                            if (!ok) continue;
                        }

                        vi crit_new = crit_route;
                        crit_new.erase(crit_new.begin() + pair_idx);
                        crit_new.erase(crit_new.begin() + pair_idx);
                        crit_new.insert(crit_new.begin() + pair_idx, single);
                        crit_new = normalize_route(crit_new);

                        vi target_new = target_route;
                        target_new.erase(target_new.begin() + pos_single);
                        target_new.insert(target_new.begin() + pos_single, c1);
                        target_new.insert(target_new.begin() + pos_single + 1, c2);
                        target_new = normalize_route(target_new);

                        vd crit_new_metrics = check_route_feasibility(crit_new, 0.0, crit_mode_truck);
                        vd target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += crit_new_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += crit_new_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += crit_new_metrics[2] - crit_metrics[2];
                        candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                        candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                        candidate.energy_violation += target_new_metrics[2] - target_metrics[2];

                        if (crit_mode_truck) {
                            candidate.truck_routes[crit_route_idx] = crit_new;
                            candidate.truck_route_times[crit_route_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                        } else {
                            int crit_drone_idx = crit_route_idx - h;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_idx] = target_new;
                            candidate.truck_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                        } else {
                            int target_drone_idx = target_idx - h;
                            if (target_drone_idx >= 0 && target_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[target_drone_idx] = target_new;
                                candidate.drone_route_times[target_drone_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(c1, c2), max(c1, c2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = solution_score_total_time(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) {
                            continue;
                        }
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_tabu_triple = key;
                            best_pair_a = c1;
                            best_pair_b = c2;
                            best_single = single;
                            best_pair_from_critical = true;
                            best_other_vehicle = target_veh;
                            best_other_is_truck = target_is_truck;
                        }
                    }
                }
            }
        };

        auto consider_single_vs_pair = [&](const vi& crit_route, bool crit_mode_truck, int crit_route_idx) {
            if (crit_route.size() <= 2) return;

            vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_mode_truck);
            vector<int> single_positions;
            for (int i = 0; i < (int)crit_route.size(); ++i)
                if (crit_route[i] != 0) single_positions.push_back(i);
            if (single_positions.empty()) return;

            auto near_enough = [&](int u, int v) {
                return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u && KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
            };

            for (int single_idx : single_positions) {
                int single = crit_route[single_idx];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == (crit_mode_truck ? crit_route_idx : h + crit_route_idx)) continue;
                    bool target_is_truck = target_veh < h;
                    if (!target_is_truck && !served_by_drone[single]) continue;

                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_idx]
                        : initial_solution.drone_routes[target_idx];
                    if (target_route.size() <= 3) continue;

                    vector<int> pair_positions;
                    for (int j = 0; j + 1 < (int)target_route.size(); ++j)
                        if (target_route[j] != 0 && target_route[j + 1] != 0) pair_positions.push_back(j);
                    if (pair_positions.empty()) continue;

                    vd target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);

                    for (int pair_idx : pair_positions) {
                        int b1 = target_route[pair_idx];
                        int b2 = target_route[pair_idx + 1];
                        if (!crit_mode_truck && (!served_by_drone[b1] || !served_by_drone[b2])) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(single, b1) || near_enough(b1, single) ||
                                      near_enough(single, b2) || near_enough(b2, single);
                            if (!ok) continue;
                        }

                        vi crit_new = crit_route;
                        crit_new.erase(crit_new.begin() + single_idx);
                        crit_new.insert(crit_new.begin() + single_idx, b1);
                        crit_new.insert(crit_new.begin() + single_idx + 1, b2);
                        crit_new = normalize_route(crit_new);

                        vi target_new = target_route;
                        target_new.erase(target_new.begin() + pair_idx);
                        target_new.erase(target_new.begin() + pair_idx);
                        target_new.insert(target_new.begin() + pair_idx, single);
                        target_new = normalize_route(target_new);

                        vd crit_new_metrics = check_route_feasibility(crit_new, 0.0, crit_mode_truck);
                        vd target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += crit_new_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += crit_new_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += crit_new_metrics[2] - crit_metrics[2];
                        candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                        candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                        candidate.energy_violation += target_new_metrics[2] - target_metrics[2];

                        if (crit_mode_truck) {
                            candidate.truck_routes[crit_route_idx] = crit_new;
                            candidate.truck_route_times[crit_route_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                        } else {
                            int crit_drone_idx = crit_route_idx - h;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_idx] = target_new;
                            candidate.truck_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                        } else {
                            int target_drone_idx = target_idx - h;
                            if (target_drone_idx >= 0 && target_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[target_drone_idx] = target_new;
                                candidate.drone_route_times[target_drone_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(b1, b2), max(b1, b2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = solution_score_total_time(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) {
                            continue;
                        }
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_tabu_triple = key;
                            best_pair_a = b1;
                            best_pair_b = b2;
                            best_single = single;
                            best_pair_from_critical = false;
                            best_other_vehicle = target_veh;
                            best_other_is_truck = target_is_truck;
                        }
                    }
                }
            }
        };

        for (int critical_idx = 0; critical_idx < h + d; ++critical_idx) {
            bool crit_is_truck = critical_idx < h;
            const vi& route = crit_is_truck
                ? initial_solution.truck_routes[critical_idx]
                : initial_solution.drone_routes[critical_idx - h];
            consider_pair_vs_single(route, crit_is_truck, critical_idx);
            consider_single_vs_pair(route, crit_is_truck, critical_idx);
        }

        if (!best_tabu_triple.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_21[best_tabu_triple] = current_iter + TABU_TENURE_21;

            // Debug N5
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            bool other_is_truck = best_other_is_truck;
            int other_idx = other_is_truck ? best_other_vehicle : best_other_vehicle - h;
            cout << "[N5] (" << (best_pair_from_critical ? "2,1" : "1,2") << ") swap pair ("
                 << best_pair_a << "," << best_pair_b << ") with customer " << best_single
                 << " between " << (crit_is_truck ? "truck" : "drone") << " #" << (critical_idx + 1)
                 << " and " << (other_is_truck ? "truck" : "drone") << " #" << (other_idx + 1)
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 6) {
        // Neighborhood 6: Swap two pairs of customers between routes

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_key;
        int best_pair_a1 = -1, best_pair_a2 = -1;
        int best_pair_b1 = -1, best_pair_b2 = -1;
        int best_other_vehicle = -1;
        bool best_other_is_truck = true;
        bool best_same_route = false;

        auto enumerate_pairs = [](const vi& route) {
            vector<int> starts;
            for (int i = 0; i + 1 < (int)route.size(); ++i) {
                if (route[i] != 0 && route[i + 1] != 0) starts.push_back(i);
            }
            return starts;
        };

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto near_enough = [&](int u, int v) {
            return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u &&
                   KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
        };

        auto consider_swap_pairs = [&](const vi& base_route, bool base_is_truck, int base_route_idx) {
            if (base_route.size() <= 3) return;

            vd base_metrics = check_route_feasibility(base_route, 0.0, base_is_truck);
            auto base_pairs = enumerate_pairs(base_route);
            if (base_pairs.empty()) return;

            for (int p : base_pairs) {
                int a1 = base_route[p];
                int a2 = base_route[p + 1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    bool target_is_truck = target_veh < h;
                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    bool same_route = (target_veh == base_route_idx);
                    const vi& target_route = same_route
                        ? base_route
                        : (target_is_truck
                               ? initial_solution.truck_routes[target_idx]
                               : initial_solution.drone_routes[target_idx]);

                    if (target_route.size() <= 3) continue;
                    auto target_pairs = enumerate_pairs(target_route);
                    if (target_pairs.empty()) continue;

                    vd target_metrics;
                    if (!same_route) {
                        target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);
                    }

                    for (int q : target_pairs) {
                        if (same_route && (q == p || q == p + 1 || p == q + 1)) continue;

                        int b1 = target_route[q];
                        int b2 = target_route[q + 1];

                        if (!target_is_truck && (!served_by_drone[a1] || !served_by_drone[a2])) continue;
                        if (!base_is_truck && (!served_by_drone[b1] || !served_by_drone[b2])) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(a1, b1) || near_enough(a1, b2) ||
                                      near_enough(a2, b1) || near_enough(a2, b2) ||
                                      near_enough(b1, a1) || near_enough(b2, a1) ||
                                      near_enough(b1, a2) || near_enough(b2, a2);
                            if (!ok) continue;
                        }

                        vector<int> tabu_key = {a1, a2, b1, b2};
                        sort(tabu_key.begin(), tabu_key.end());
                        auto it_tabu = tabu_list_22.find(tabu_key);
                        bool is_tabu = (it_tabu != tabu_list_22.end() && it_tabu->second > current_iter);

                        vi base_new = base_route;
                        vi target_new = target_route;

                        if (same_route) {
                            swap(base_new[p], base_new[q]);
                            swap(base_new[p + 1], base_new[q + 1]);
                        } else {
                            base_new[p] = b1;
                            base_new[p + 1] = b2;
                            target_new[q] = a1;
                            target_new[q + 1] = a2;
                        }
                        base_new = normalize_route(base_new);
                        target_new = normalize_route(target_new);

                        vd base_new_metrics = check_route_feasibility(base_new, 0.0, base_is_truck);
                        vd target_new_metrics;
                        if (!same_route) {
                            target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);
                        }

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += base_new_metrics[1] - base_metrics[1];
                        candidate.capacity_violation += base_new_metrics[3] - base_metrics[3];
                        candidate.energy_violation += base_new_metrics[2] - base_metrics[2];

                        if (base_is_truck) {
                            candidate.truck_routes[base_route_idx] = base_new;
                            candidate.truck_route_times[base_route_idx] = (base_new.size() > 1) ? base_new_metrics[0] : 0.0;
                        } else {
                            int base_drone_idx = base_route_idx - h;
                            if (base_drone_idx >= 0 && base_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[base_drone_idx] = base_new;
                                candidate.drone_route_times[base_drone_idx] = (base_new.size() > 1) ? base_new_metrics[0] : 0.0;
                            }
                        }

                        if (!same_route) {
                            if (target_is_truck) {
                                candidate.truck_routes[target_idx] = target_new;
                                candidate.truck_route_times[target_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                            } else {
                                int target_drone_idx = target_idx;
                                if (target_drone_idx >= 0 && target_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[target_drone_idx] = target_new;
                                    candidate.drone_route_times[target_drone_idx] = (target_new.size() > 1) ? target_new_metrics[0] : 0.0;
                                }
                            }
                        }

                        candidate.total_makespan = 0.0;
                        for (double t : candidate.truck_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        double candidate_score = solution_score_total_time(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) {
                            continue;
                        }
                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_tabu_key = tabu_key;
                            best_pair_a1 = a1;
                            best_pair_a2 = a2;
                            best_pair_b1 = b1;
                            best_pair_b2 = b2;
                            best_other_vehicle = target_veh;
                            best_other_is_truck = target_is_truck;
                            best_same_route = same_route;
                        }
                    }
                }
            }
        };

        for (int critical_idx = 0; critical_idx < h + d; ++critical_idx) {
            bool crit_is_truck = critical_idx < h;
            const vi& route = crit_is_truck
                ? initial_solution.truck_routes[critical_idx]
                : initial_solution.drone_routes[critical_idx - h];
            consider_swap_pairs(route, crit_is_truck, critical_idx);
        }

        if (!best_tabu_key.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_22[best_tabu_key] = current_iter + TABU_TENURE_22;

            // Debug N6
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            if (best_same_route) {
                cout << "[N6] (2,2) swap within " << (crit_is_truck ? "truck" : "drone") << " #"
                     << (critical_idx + 1) << ": (" << best_pair_a1 << "," << best_pair_a2
                     << ") ↔ (" << best_pair_b1 << "," << best_pair_b2 << "), makespan: "
                     << ", score: " << solution_score(initial_solution)
                     << " -> " << solution_score(best_candidate_neighbor)
                     << ", iter " << current_iter << "\n";
            } else {
                int other_idx = best_other_is_truck ? best_other_vehicle : best_other_vehicle - h;
                cout << "[N6] (2,2) swap pairs (" << best_pair_a1 << "," << best_pair_a2 << ") ↔ ("
                     << best_pair_b1 << "," << best_pair_b2 << ") between "
                     << (crit_is_truck ? "truck" : "drone") << " #" << (critical_idx + 1)
                     << " and " << (best_other_is_truck ? "truck" : "drone") << " #" << (other_idx + 1)
                     << ", makespan: " << initial_solution.total_makespan << " -> "
                     << best_neighbor.total_makespan << ", iter " << current_iter << "\n";
            } */

            return best_neighbor;
        }
        return initial_solution;

    }  else if (neighbor_id == 7) {
        // Neighborhood 7: depth-2 ejection chain (i -> j -> k)
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_key;
        int best_veh_i = -1, best_veh_j = -1, best_veh_k = -1;
        int best_cust_removed = -1, best_cust_ejected = -1;

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto is_truck_vehicle = [&](int veh_id) { return veh_id < h; };
        auto fetch_route = [&](int veh_id) -> const vi& {
            return (veh_id < h) ? initial_solution.truck_routes[veh_id]
                                : initial_solution.drone_routes[veh_id - h];
        };

        auto get_metrics = [&](const vi& route, bool truck_mode) {
            return check_route_feasibility(route, 0.0, truck_mode);
        };

        auto is_near = [&](int u, int v) {
            if (KNN_ADJ.empty()) return true;
            if (u < 0 || v < 0) return false;
            if (u >= (int)KNN_ADJ.size()) return false;
            if (v >= (int)KNN_ADJ[u].size()) return false;
            return (KNN_ADJ[u][v] == 1);
        };

        const int MAX_ROUTE_TRIPLETS = min(50, (h + d) * max(0, h + d - 1) * max(0, h + d - 2) / 6);
        int triplets_evaluated = 0;
        bool stop_search = false;

        for (int veh_i = 0; veh_i < h + d && !stop_search; ++veh_i) {
            const vi& route_i_raw = fetch_route(veh_i);
            if (route_i_raw.size() <= 2) continue;
            vi route_i = normalize_route(route_i_raw);
            vd metrics_i = get_metrics(route_i, is_truck_vehicle(veh_i));

            vector<int> pos_i;
            for (int idx = 0; idx < (int)route_i.size(); ++idx)
                if (route_i[idx] != 0) pos_i.push_back(idx);
            if (pos_i.empty()) continue;

            for (int veh_j = 0; veh_j < h + d && !stop_search; ++veh_j) {
                if (veh_j == veh_i) continue;
                const vi& route_j_raw = fetch_route(veh_j);
                if (route_j_raw.size() <= 2) continue;
                vi route_j = normalize_route(route_j_raw);
                vd metrics_j = get_metrics(route_j, is_truck_vehicle(veh_j));

                vector<int> pos_j;
                vector<int> customers_j;
                for (int idx = 0; idx < (int)route_j.size(); ++idx) {
                    if (route_j[idx] != 0) {
                        pos_j.push_back(idx);
                        customers_j.push_back(route_j[idx]);
                    }
                }
                if (pos_j.empty()) continue;

                for (int veh_k = 0; veh_k < h + d; ++veh_k) {
                    if (veh_k == veh_i || veh_k == veh_j) continue;
                    if (triplets_evaluated >= MAX_ROUTE_TRIPLETS) { stop_search = true; break; }
                    ++triplets_evaluated;

                    const vi& route_k_raw = fetch_route(veh_k);
                    vi route_k = normalize_route(route_k_raw);
                    vd metrics_k = get_metrics(route_k, is_truck_vehicle(veh_k));

                    vector<int> pos_k_candidates;
                    for (int idx = 1; idx <= (int)route_k.size(); ++idx)
                        pos_k_candidates.push_back(idx);

                    if (pos_k_candidates.empty()) continue;

                    for (int pos_idx_i : pos_i) {
                        int cust_removed = route_i[pos_idx_i];
                        vi route_i_new = route_i;
                        route_i_new.erase(route_i_new.begin() + pos_idx_i);
                        route_i_new = normalize_route(route_i_new);
                        vd metrics_i_new = get_metrics(route_i_new, is_truck_vehicle(veh_i));

                        if (!KNN_ADJ.empty()) {
                            bool near_some = false;
                            for (int c : customers_j) {
                                if (is_near(cust_removed, c) || is_near(c, cust_removed)) { near_some = true; break; }
                            }
                            if (!customers_j.empty() && !near_some) continue;
                        }
                        if (!is_truck_vehicle(veh_j) && !served_by_drone[cust_removed]) continue;

                        for (int pos_idx_j : pos_j) {
                            int cust_ejected = route_j[pos_idx_j];
                            if (cust_removed == cust_ejected) continue;
                            if (!is_truck_vehicle(veh_k) && !served_by_drone[cust_ejected]) continue;

                            if (!KNN_ADJ.empty()) {
                                if (!(is_near(cust_removed, cust_ejected) || is_near(cust_ejected, cust_removed))) continue;
                            }

                            vi route_j_new = route_j;
                            route_j_new[pos_idx_j] = cust_removed;
                            route_j_new = normalize_route(route_j_new);
                            vd metrics_j_new = get_metrics(route_j_new, is_truck_vehicle(veh_j));

                            for (int insert_pos_k : pos_k_candidates) {
                                vi route_k_new = route_k;
                                if (find(route_k_new.begin(), route_k_new.end(), cust_ejected) != route_k_new.end()) continue;
                                int insert_index = min(insert_pos_k, (int)route_k_new.size());
                                route_k_new.insert(route_k_new.begin() + insert_index, cust_ejected);
                                route_k_new = normalize_route(route_k_new);
                                vd metrics_k_new = get_metrics(route_k_new, is_truck_vehicle(veh_k));

                                if (!KNN_ADJ.empty()) {
                                    int idx_new = -1;
                                    for (int idx = 0; idx < (int)route_k_new.size(); ++idx) {
                                        if (route_k_new[idx] == cust_ejected) { idx_new = idx; break; }
                                    }
                                    if (idx_new != -1 && idx_new > 0 && idx_new + 1 < (int)route_k_new.size()) {
                                        int prev = route_k_new[idx_new - 1];
                                        int next = route_k_new[idx_new + 1];
                                        if (!(is_near(cust_ejected, prev) || is_near(prev, cust_ejected) ||
                                              is_near(cust_ejected, next) || is_near(next, cust_ejected))) {
                                            continue;
                                        }
                                    }
                                }

                                Solution candidate = initial_solution;

                                candidate.deadline_violation += metrics_i_new[1] - metrics_i[1];
                                candidate.capacity_violation += metrics_i_new[3] - metrics_i[3];
                                candidate.energy_violation += metrics_i_new[2] - metrics_i[2];

                                candidate.deadline_violation += metrics_j_new[1] - metrics_j[1];
                                candidate.capacity_violation += metrics_j_new[3] - metrics_j[3];
                                candidate.energy_violation += metrics_j_new[2] - metrics_j[2];

                                candidate.deadline_violation += metrics_k_new[1] - metrics_k[1];
                                candidate.capacity_violation += metrics_k_new[3] - metrics_k[3];
                                candidate.energy_violation += metrics_k_new[2] - metrics_k[2];

                                if (is_truck_vehicle(veh_i)) {
                                    candidate.truck_routes[veh_i] = route_i_new;
                                    candidate.truck_route_times[veh_i] = (route_i_new.size() > 1) ? metrics_i_new[0] : 0.0;
                                } else {
                                    candidate.drone_routes[veh_i - h] = route_i_new;
                                    candidate.drone_route_times[veh_i - h] = (route_i_new.size() > 1) ? metrics_i_new[0] : 0.0;
                                }

                                if (is_truck_vehicle(veh_j)) {
                                    candidate.truck_routes[veh_j] = route_j_new;
                                    candidate.truck_route_times[veh_j] = (route_j_new.size() > 1) ? metrics_j_new[0] : 0.0;
                                } else {
                                    candidate.drone_routes[veh_j - h] = route_j_new;
                                    candidate.drone_route_times[veh_j - h] = (route_j_new.size() > 1) ? metrics_j_new[0] : 0.0;
                                }

                                if (is_truck_vehicle(veh_k)) {
                                    candidate.truck_routes[veh_k] = route_k_new;
                                    candidate.truck_route_times[veh_k] = (route_k_new.size() > 1) ? metrics_k_new[0] : 0.0;
                                } else {
                                    candidate.drone_routes[veh_k - h] = route_k_new;
                                    candidate.drone_route_times[veh_k - h] = (route_k_new.size() > 1) ? metrics_k_new[0] : 0.0;
                                }

                                candidate.total_makespan = 0.0;
                                for (double t : candidate.truck_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                                for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                                vector<int> tabu_key = {min(cust_removed, cust_ejected), max(cust_removed, cust_ejected)};
                                bool is_tabu = (tabu_list_ejection.count(tabu_key) &&
                                                tabu_list_ejection[tabu_key] > current_iter);

                                double candidate_score = solution_score_total_time(candidate);
                                if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                                candidate.deadline_violation <= 1e-8 &&
                                                candidate.capacity_violation <= 1e-8 &&
                                                candidate.energy_violation <= 1e-8)) {
                                    continue;
                                }
                                if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                    best_neighbor_cost_local = candidate_score;
                                    best_candidate_neighbor = candidate;
                                    best_tabu_key = tabu_key;
                                    best_veh_i = veh_i;
                                    best_veh_j = veh_j;
                                    best_veh_k = veh_k;
                                    best_cust_removed = cust_removed;
                                    best_cust_ejected = cust_ejected;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (!best_tabu_key.empty()) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_ejection[best_tabu_key] = current_iter + TABU_TENURE_EJECTION;

            cout.setf(std::ios::fixed);
            cout << setprecision(6);
            auto print_vehicle = [&](int v) {
                return (is_truck_vehicle(v) ? string("truck #") + to_string(v + 1)
                                            : string("drone #") + to_string(v - h + 1));
            };
            // Debug N7
            /* cout << "[N7] ejection chain: move " << best_cust_removed << " from " << print_vehicle(best_veh_i)
                 << ", replace " << best_cust_ejected << " on " << print_vehicle(best_veh_j)
                 << ", insert into " << print_vehicle(best_veh_k)
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    }
// ...existing code...
return initial_solution;
}

void updated_edge_records(const Solution& sol){
    for (int i = 0; i < h; ++i) {
        const vi& route = sol.truck_routes[i];
        for (size_t j = 0; j + 1 < route.size(); ++j) {
            int u = route[j];
            int v = route[j + 1];
            edge_records[u][v] = min(edge_records[u][v], sol.total_makespan);
            edge_records[v][u] = min(edge_records[v][u], sol.total_makespan);
        }
    }
    for (int i = 0; i < d; ++i) {
        const vi& route = sol.drone_routes[i];
        for (size_t j = 0; j + 1 < route.size(); ++j) {
            int u = route[j];
            int v = route[j + 1];
            edge_records[u][v] = min(edge_records[u][v], sol.total_makespan);
            edge_records[v][u] = min(edge_records[v][u], sol.total_makespan);
        }
    }
}

int hamming_distance(const Solution& sol1, const Solution& sol2) {
    int distance = 0;
    int successor1[n+1];
    int successor2[n+1];
    for (int i = 0; i < h; ++i) {
        const vi& route = sol1.truck_routes[i];
        for (size_t j = 0; j + 1 < route.size(); ++j) {
            int u = route[j];
            int v = route[j + 1];
            if (u != 0) {
                successor1[u] = v;
            }
        }
        const vi& route2 = sol2.truck_routes[i];
        for (size_t j = 0; j + 1 < route2.size(); ++j) {
            int u = route2[j];
            int v = route2[j + 1];
            if (u != 0) {
                successor2[u] = v;
            }
        }
    }
    for (int i = 0; i < d; ++i) {
        const vi& route = sol1.drone_routes[i];
        for (size_t j = 0; j + 1 < route.size(); ++j) {
            int u = route[j];
            int v = route[j + 1];
            if (u != 0) {
                successor1[u] = v;
            }
        }
        const vi& route2 = sol2.drone_routes[i];
        for (size_t j = 0; j + 1 < route2.size(); ++j) {
            int u = route2[j];
            int v = route2[j + 1];
            if (u != 0) {
                successor2[u] = v;
            }
        }
    }
    for (int i = 1; i <= n; ++i) {
        if (successor1[i] != successor2[i]) {
            distance++;
        }
    }
    return distance;
}

Solution recalculate_solution(Solution sol) {
    sol.deadline_violation = 0.0;
    sol.energy_violation = 0.0;
    sol.capacity_violation = 0.0;
    for (int i = 0; i < h; ++i) {
        vd metrics = check_route_feasibility(sol.truck_routes[i], 0.0, true);
        sol.truck_route_times[i] = metrics[0];
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];
    }
    for (int i = 0; i < d; ++i) {
        vd metrics = check_route_feasibility(sol.drone_routes[i], 0.0, false);
        sol.drone_route_times[i] = metrics[0];
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];
    }
    sol.total_makespan = 0.0;
    for (int t = 0; t < h; ++t) sol.total_makespan = max(sol.total_makespan, sol.truck_route_times[t]);
    for (int t = 0; t < d; ++t) sol.total_makespan = max(sol.total_makespan, sol.drone_route_times[t]);
    return sol;
}

Solution updated_elite_set(const Solution& sol) {
    bool is_feasible = (sol.deadline_violation <= 1e-8 &&
                        sol.energy_violation <= 1e-8 &&
                        sol.capacity_violation <= 1e-8);
    if (!is_feasible) return sol;
    Solution tmp;
    if (elite_set.size() < ELITE_SET_SIZE) {
        elite_set.push_back(sol);
    } else {
        int min_distance = 1e9;
        int replace_idx = -1;
        for (size_t i = 0; i < elite_set.size(); ++i) {
            int dist = hamming_distance(sol, elite_set[i]);
            if (dist < min_distance) {
                min_distance = dist;
                replace_idx = i;
            }
        }
        if (replace_idx != -1) {
            tmp = elite_set[replace_idx];
            elite_set[replace_idx] = sol;
        }
    }
    return tmp;
}

Solution greedy_insert_customer(Solution sol, int customer) {
    Solution best_sol = sol;
    double best_score = 1e18;
    auto try_insert = [&](vi base_route, bool is_truck, int route_idx) {
        vd base_metrics = check_route_feasibility(base_route, 0.0, is_truck);
        for (size_t pos = 1; pos < base_route.size(); ++pos) {
            vi new_route = base_route;
            new_route.insert(new_route.begin() + pos, customer);
            vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck);
            double new_makespan = 0.0;
            for (int t = 0; t < h; ++t){
                new_makespan = max(new_makespan, (t == route_idx && is_truck) ? new_metrics[0] : sol.truck_route_times[t]);
            }
            for (int t = 0; t < d; ++t){
                new_makespan = max(new_makespan, (t == route_idx && !is_truck) ? new_metrics[0] : sol.drone_route_times[t]);
            }
            double new_deadline_violation = sol.deadline_violation + new_metrics[1] - base_metrics[1];
            double new_energy_violation = sol.energy_violation + new_metrics[2] - base_metrics[2];
            double new_capacity_violation = sol.capacity_violation + new_metrics[3] - base_metrics[3];
            double violation = new_deadline_violation * 1e3 +
                               new_energy_violation * 1e3 +
                               new_capacity_violation * 1e3;
            double new_score = new_makespan * pow((1.0 + violation), PENALTY_EXPONENT);
            if (new_score + 1e-8 < best_score) {
                best_score = new_score;
                best_sol = sol;
                best_sol.deadline_violation = new_deadline_violation;
                best_sol.energy_violation = new_energy_violation;
                best_sol.capacity_violation = new_capacity_violation; 
                best_sol.total_makespan = new_makespan;
                if (is_truck) {
                    best_sol.truck_routes[route_idx] = new_route;
                    best_sol.truck_route_times[route_idx] = new_metrics[0];
                } else {
                    best_sol.drone_routes[route_idx] = new_route;
                    best_sol.drone_route_times[route_idx] = new_metrics[0];
                }
            }
        }
        // Also attempt to insert at the end of the route
        {
            vi new_route = base_route;
            if (new_route.back() != 0) new_route.push_back(0);
            new_route.push_back(customer);
            new_route.push_back(0);
            vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck);
            double new_makespan = 0.0;
            for (int t = 0; t < h; ++t){
                new_makespan = max(new_makespan, (t == route_idx && is_truck) ? new_metrics[0] : sol.truck_route_times[t]);
            }
            for (int t = 0; t < d; ++t){
                new_makespan = max(new_makespan, (t == route_idx && !is_truck) ? new_metrics[0] : sol.drone_route_times[t]);
            }
            double new_deadline_violation = max(sol.deadline_violation + new_metrics[1] - base_metrics[1], 0.0);
            double new_energy_violation = max(sol.energy_violation + new_metrics[2] - base_metrics[2], 0.0);
            double new_capacity_violation = max(sol.capacity_violation + new_metrics[3] - base_metrics[3], 0.0);
            double violation = new_deadline_violation * 1e3 +
                               new_energy_violation * 1e3 +
                               new_capacity_violation * 1e3;
            double new_score = new_makespan * pow((1.0 + violation), PENALTY_EXPONENT);
            if (new_score + 1e-8 < best_score) {
                best_score = new_score;
                best_sol = sol;
                best_sol.deadline_violation = new_deadline_violation;
                best_sol.energy_violation = new_energy_violation;
                best_sol.capacity_violation = new_capacity_violation;
                best_sol.total_makespan = new_makespan;
                if (is_truck) {
                    best_sol.truck_routes[route_idx] = new_route;
                    best_sol.truck_route_times[route_idx] = new_metrics[0];
                } else {
                    best_sol.drone_routes[route_idx] = new_route;
                    best_sol.drone_route_times[route_idx] = new_metrics[0];
                }
            }
        }
    };
    for (int i = 0; i < h; ++i) {
        try_insert(sol.truck_routes[i], true, i);
    }
    for (int i = 0; i < d; ++i) {
        try_insert(sol.drone_routes[i], false, i);
    }
    return best_sol;
}

Solution destroy_and_repair(Solution sol) {
    vector<double> scores(n+1, 0.0);
    for (int i = 0; i < h; ++i) {
        const vi& route = sol.truck_routes[i];
        for (size_t j = 1; j + 1 < route.size(); ++j) {
            int u = route[j - 1];
            int c = route[j];
            int v = route[j + 1];
            scores[c] = edge_records[u][c] + edge_records[c][v];
        }
    }
    for (int i = 0; i < d; ++i) {
        const vi& route = sol.drone_routes[i];
        for (size_t j = 1; j + 1 < route.size(); ++j) {
            int u = route[j - 1];
            int c = route[j];
            int v = route[j + 1];
            scores[c] = edge_records[u][c] + edge_records[c][v];
        }
    }
    for (int i = 1; i <= n; ++i) scores[i] = 1.0 / (1.0 + scores[i]);
    vector<int> ordered_customers;
    for (int i = 1; i <= n; ++i) {
        ordered_customers.push_back(i);
    }
    sort(ordered_customers.begin(), ordered_customers.end(), [&](int a, int b) {
        return scores[a] < scores[b];
    });

    unordered_set<int> to_destroy;
    int destroy_count = static_cast<int>(n * DESTROY_RATE);
    
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> dist(0, ordered_customers.size() - 1);

    while (to_destroy.size() < destroy_count) {
        int rand_val = dist(rng);
        // Bias towards higher-scored customers (at the end of 'ordered_customers')
        if (rand_val == 0) rand_val = 1;
        if (ordered_customers.size() > 1) {
            int index = (static_cast<long long>(rand_val) * rand_val) / (ordered_customers.size() - 1);
            to_destroy.insert(ordered_customers[index]);
        }
        else {
            to_destroy.insert(ordered_customers[0]);
        }
    }

    Solution new_sol = sol;
    for (int i = 0; i < h; ++i) {
        vi& route = new_sol.truck_routes[i];
        route.erase(remove_if(route.begin(), route.end(), [&](int c) {
            return to_destroy.count(c) > 0;
        }), route.end());
    }
    for (int i = 0; i < d; ++i) {
        vi& route = new_sol.drone_routes[i];
        route.erase(remove_if(route.begin(), route.end(), [&](int c) {
            return to_destroy.count(c) > 0;
        }), route.end());
    }
    //Recalculate route times and total makespan
    new_sol.deadline_violation = 0.0;
    new_sol.capacity_violation = 0.0;
    new_sol.energy_violation = 0.0;
    new_sol.total_makespan = 0.0;
    for (int i = 0; i < h; ++i) {
        vd metrics = check_route_feasibility(new_sol.truck_routes[i], 0.0, true);
        new_sol.truck_route_times[i] = (new_sol.truck_routes[i].size() > 1) ? metrics[0] : 0.0;
        new_sol.deadline_violation += metrics[1];
        new_sol.energy_violation += metrics[2];
        new_sol.capacity_violation += metrics[3];
        new_sol.total_makespan = max(new_sol.total_makespan, new_sol.truck_route_times[i]);
    }
    for (int i = 0; i < d; ++i) {
        vd metrics = check_route_feasibility(new_sol.drone_routes[i], 0.0, false);
        new_sol.drone_route_times[i] = (new_sol.drone_routes[i].size() > 1) ? metrics[0] : 0.0;
        new_sol.deadline_violation += metrics[1];
        new_sol.energy_violation += metrics[2];
        new_sol.capacity_violation += metrics[3];
        new_sol.total_makespan = max(new_sol.total_makespan, new_sol.drone_route_times[i]);
    }

    vector<int> customers_to_insert(to_destroy.begin(), to_destroy.end());
    std::shuffle(customers_to_insert.begin(), customers_to_insert.end(), rng);

    for (int cust : customers_to_insert) {
        new_sol = greedy_insert_customer(new_sol, cust);
    }
    // Normalize route formats after insertions
    for (int i = 0; i < h; ++i) {
        vi& route = new_sol.truck_routes[i];
        if (route.empty() || route.front() != 0) route.insert(route.begin(), 0);
        if (route.back() != 0) route.push_back(0);
        vi cleaned;
        cleaned.reserve(route.size());
        for (int node : route) {
            if (!cleaned.empty() && cleaned.back() == 0 && node == 0) continue;
            cleaned.push_back(node);
        }
        route = cleaned;
    }
    for (int i = 0; i < d; ++i) {
        vi& route = new_sol.drone_routes[i];
        if (route.empty() || route.front() != 0) route.insert(route.begin(), 0);
        if (route.back() != 0) route.push_back(0);
        vi cleaned;
        cleaned.reserve(route.size());
        for (int node : route) {
            if (!cleaned.empty() && cleaned.back() == 0 && node == 0) continue;
            cleaned.push_back(node);
        }
        route = cleaned;
    }
    // Recalculate route times and total makespan after normalization
    new_sol.deadline_violation = 0.0;
    new_sol.capacity_violation = 0.0;
    new_sol.energy_violation = 0.0;
    new_sol.total_makespan = 0.0;
    for (int i = 0; i < h; ++i) {
        vd metrics = check_route_feasibility(new_sol.truck_routes[i], 0.0, true);
        new_sol.truck_route_times[i] = (new_sol.truck_routes[i].size() > 1) ? metrics[0] : 0.0;
        new_sol.deadline_violation += metrics[1];
        new_sol.energy_violation += metrics[2];
        new_sol.capacity_violation += metrics[3];
        new_sol.total_makespan = max(new_sol.total_makespan, new_sol.truck_route_times[i]);
    }
    for (int i = 0; i < d; ++i) {
        vd metrics = check_route_feasibility(new_sol.drone_routes[i], 0.0, false);
        new_sol.drone_route_times[i] = (new_sol.drone_routes[i].size() > 1) ? metrics[0] : 0.0;
        new_sol.deadline_violation += metrics[1];
        new_sol.energy_violation += metrics[2];
        new_sol.capacity_violation += metrics[3];
        new_sol.total_makespan = max(new_sol.total_makespan, new_sol.drone_route_times[i]);
    }
    return new_sol;
}

Solution tabu_search(const Solution& initial_solution, int num_initial_sol) {
    auto ts_start = std::chrono::high_resolution_clock::now();
    auto is_feasible = [](const Solution& sol) {
        return sol.deadline_violation <= 1e-8 &&
               sol.capacity_violation <= 1e-8 &&
               sol.energy_violation <= 1e-8;
    };
    // Initialize edge records
    edge_records.assign(n + 1, vector<double>(n + 1, 1e10));
    updated_edge_records(initial_solution);
    Solution best_solution = initial_solution;
    Solution best_feasible_solution = initial_solution;
    bool initial_feasible = is_feasible(initial_solution);
    double best_feasible_makespan = initial_feasible
        ? initial_solution.total_makespan
        : std::numeric_limits<double>::infinity();
    double best_cost = initial_feasible ? best_feasible_makespan : std::numeric_limits<double>::infinity();
    double score[NUM_NEIGHBORHOODS] = {0.0};
    double weight[NUM_NEIGHBORHOODS];
    for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] = 1.0 / NUM_NEIGHBORHOODS;
    int count[NUM_NEIGHBORHOODS] = {0};

    Solution current_sol = initial_solution;
    double current_cost = initial_solution.total_makespan;

    /* for (int i = 0; i < num_initial_sol; i++){
        Solution initial_sol = generate_initial_solution();
        updated_edge_records(initial_sol);
        Solution best_local_solution = initial_sol;
        for (int j = 0; j < 10; j++){
            int selected_neighbor = rand() % NUM_NEIGHBORHOODS;
            initial_sol = local_search(initial_sol, selected_neighbor, 0, solution_score(best_solution));
            if (solution_score(initial_sol) + 1e-12 < solution_score(best_local_solution) ||
                (std::abs(solution_score(initial_sol) - solution_score(best_local_solution)) <= 1e-12 &&
                 initial_sol.total_makespan + 1e-12 < best_local_solution.total_makespan)) {
                best_local_solution = initial_sol;
            }
            if (is_feasible(initial_sol) &&
                initial_sol.total_makespan + 1e-12 < best_feasible_makespan) {
                best_feasible_makespan = initial_sol.total_makespan;
                best_feasible_solution = initial_sol;
                best_cost = best_feasible_makespan;
            }
        }
        // if it's better than the worst solution in elite set, add it
        if (elite_set.size() < ELITE_SET_SIZE) {
            elite_set.push_back(best_local_solution);
        } else {
            double worst_score = -1.0;
            int worst_idx = -1;
            for (size_t j = 0; j < elite_set.size(); ++j) {
                double s = solution_score(elite_set[j]);
                if (s > worst_score) {
                    worst_score = s;
                    worst_idx = j;
                }
            }
            if (solution_score(best_local_solution) + 1e-12 < worst_score) {
                elite_set[worst_idx] = best_local_solution;
            }
        }
    } 

    // Pick current solution from elite set randomly
    if (!elite_set.empty()) {
        int rand_idx = rand() % elite_set.size();
        current_sol = elite_set[rand_idx];
        current_cost = current_sol.total_makespan;
    }*/
    current_sol = initial_solution;
    current_cost = current_sol.total_makespan; 
    bool total_score_segment = false;
    for (int segment = 0; segment < CFG_MAX_SEGMENT; ++segment) {
        int iter = 1;
        int no_improve_iters = 0;
        double T0 = 100.0; // initial temperature for simulated annealing acceptance
        double alpha = 0.998; // cooling rate
        bool total_score_iter = total_score_segment;
        //Debug:
        //cout << "=== Starting Segment " << segment + 1 <<" with searching " << (total_score_iter ? "on all routes" : "on critical route only") << " ===\n";
    // Reset tabu lists at the start of each segment (iteration counter restarts per segment)
        while (iter <= CFG_MAX_ITER_PER_SEGMENT) {
            // testing
            total_score_iter = false;
            if (CFG_TIME_LIMIT_SEC > 0.0) {
                double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
                if (elapsed >= CFG_TIME_LIMIT_SEC) break;
            }
            double best_solution_score_now = 1e10, current_score = 0.0;
            if (total_score_iter) {
                current_score = solution_score_total_time(current_sol);
                best_solution_score_now = solution_score_total_time(best_solution);
            }
            else {
                current_score = solution_score(current_sol);
                best_solution_score_now = solution_score(best_solution);
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
            Solution neighbor;
            if (total_score_iter) {
                neighbor = local_search_all_vehicle(current_sol, selected_neighbor, iter, best_solution_score_now);
            }
            else neighbor = local_search(current_sol, selected_neighbor, iter, best_solution_score_now);
            bool neighbor_feasible = is_feasible(neighbor);
            // Clamp neighbor violation values to zero if they are very small
            if (neighbor.deadline_violation < 1e-8) neighbor.deadline_violation = 0.0;
            if (neighbor.capacity_violation < 1e-8) neighbor.capacity_violation = 0.0;
            if (neighbor.energy_violation < 1e-8) neighbor.energy_violation = 0.0;
            double neighbor_score;
            if (total_score_iter) {
                neighbor_score = solution_score_total_time(neighbor);
            } else {
                neighbor_score = solution_score(neighbor);
            }
            // Debug: print selected neighborhood and scores
            /* cout.setf(std::ios::fixed); cout << setprecision(6);
            cout << "[Iter " << iter << "] Selected Neighborhood: " << selected_neighbor
                 << ", Current Score: " << current_score
                 << ", Neighbor Score: " << neighbor_score
                 << ", Current Makespan: " << current_cost
                 << ", Neighbor Makespan: " << neighbor.total_makespan << "\n";
            print_solution_stream(neighbor, cout); */

            if (neighbor_score + 1e-12 < best_solution_score_now ||
                (std::abs(neighbor_score - best_solution_score_now) <= 1e-12 &&
                 neighbor.total_makespan + 1e-12 < best_solution.total_makespan)) {
                current_sol = neighbor;
                current_cost = neighbor.total_makespan;
                best_solution = neighbor;
                score[selected_neighbor] += gamma1;
                no_improve_iters = 0;
                best_solution_score_now = neighbor_score;
                //updated_edge_records(neighbor);
                //Solution tmp = updated_elite_set(neighbor);
                total_score_segment = false;
                /* if (!total_score_iter) cout << "New best sol with makespan " << neighbor.total_makespan << endl;
                else cout << "New best sol with total score " << neighbor_score << endl; */
            } else if (neighbor_score + 1e-12 < current_score ||
                       (std::abs(neighbor_score - current_score) <= 1e-12 &&
                        neighbor.total_makespan + 1e-12 < current_cost)) {
                current_sol = neighbor;
                current_cost = neighbor.total_makespan;
                score[selected_neighbor] += gamma2;
            } else {
                double T = T0 * pow(alpha, iter);
                double delta = current_score - neighbor_score;
                double ap = exp(delta / T);
                double rand_val = ((double) rand() / (RAND_MAX));
                if (rand_val < ap) {
                    current_sol = neighbor;
                    current_cost = neighbor.total_makespan;
                }
                score[selected_neighbor] += gamma3;
                no_improve_iters++;
            }
            if (is_feasible(current_sol) &&
                current_sol.total_makespan + 1e-12 < best_feasible_makespan) {
                best_feasible_makespan = current_sol.total_makespan;
                best_feasible_solution = current_sol;
                best_cost = best_feasible_makespan;
            } else if (neighbor_feasible &&
                       neighbor.total_makespan + 1e-12 < best_feasible_makespan) {
                best_feasible_makespan = neighbor.total_makespan;
                best_feasible_solution = neighbor;
                best_cost = best_feasible_makespan;
            }
            update_penalties(current_sol);
            if (no_improve_iters >= CFG_MAX_NO_IMPROVE || iter >= CFG_MAX_ITER_PER_SEGMENT){
                // Restart from a random elite solution 
                if (total_score_segment) total_score_segment = false;
                else total_score_segment = true;
                tabu_list_10.clear();
                tabu_list_11.clear();
                tabu_list_20.clear();
                tabu_list_2opt.clear();
                tabu_list_2opt_star.clear();
                tabu_list_22.clear();
                tabu_list_21.clear();
                tabu_list_ejection.clear();
                no_improve_iters = 0;
                /*if (!elite_set.empty()) {
                    // Debug: print restart info
                    Solution tmp = current_sol;
                    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
                    std::uniform_int_distribution<int> dist(0, elite_set.size() - 1);
                    int rand_idx = dist(rng);
                    current_sol = elite_set[rand_idx];
                    cout << "Pushing sol with makespan " << tmp.total_makespan << " to elite set;";
                    Solution pushed_out_sol = updated_elite_set(tmp);
                    cout << " replacing sol with makespan " << pushed_out_sol.total_makespan << "\n";
                    current_cost = current_sol.total_makespan;
                    cout.setf(std::ios::fixed); cout << setprecision(6);
                    cout << "[Restart] from no-improve solution " << tmp.total_makespan << " to elite solution with cost: " << current_sol.total_makespan << " ";
                    current_sol = destroy_and_repair(current_sol);
                    current_cost = current_sol.total_makespan;
                }*/
                int local_iter = 0;
                for (int i = 0; i < EJECTION_CHAIN_ITERS; ++i) {
                    if (total_score_iter){
                        current_sol = local_search_all_vehicle(current_sol, 0, local_iter, best_solution_score_now);
                        current_cost = current_sol.total_makespan;
                    }
                    else {
                        current_sol = local_search(current_sol, 0, local_iter, best_solution_score_now);
                        current_cost = solution_score(current_sol);
                    }
                    if (is_feasible(current_sol) &&
                        current_sol.total_makespan + 1e-12 < best_feasible_makespan) {
                        best_feasible_makespan = current_sol.total_makespan;
                        best_feasible_solution = current_sol;
                        best_cost = best_feasible_makespan;
                        //updated_edge_records(current_sol);
                        //updated_elite_set(current_sol);
                        best_solution_score_now = neighbor_score;
                    }
                    update_penalties(current_sol);
                    // update he so = 1
                    local_iter++;
                }
                tabu_list_ejection.clear();
                break;
                //cout << "after destroy-and-repair and ejection chain: " << current_sol.total_makespan << "\n";
            }
            // Debug: print iteration info
            //cout.setf(std::ios::fixed); cout << setprecision(6);
            //cout << "Lambda_D=" << PENALTY_LAMBDA_DEADLINE << ", Lambda_C=" << PENALTY_LAMBDA_CAPACITY << ", Lambda_E=" << PENALTY_LAMBDA_ENERGY << "\n";
            iter++;
        }
        //recalculate solution
        current_sol = recalculate_solution(current_sol);
        current_cost = current_sol.total_makespan;
        best_solution = recalculate_solution(best_solution);
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

        if (is_feasible(best_solution) &&
            best_solution.total_makespan + 1e-12 < best_feasible_makespan) {
            best_feasible_makespan = best_solution.total_makespan;
            best_feasible_solution = best_solution;
            best_cost = best_feasible_makespan;
        }

        if (!elite_set.empty()) {
            std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
            std::uniform_int_distribution<int> dist(0, elite_set.size() - 1);
            int rand_idx = dist(rng);
            current_sol = elite_set[rand_idx];
            current_cost = current_sol.total_makespan;
        }

        // Debug: print neighborhood weights after each segment
        cout.setf(std::ios::fixed); cout << setprecision(6);
        //cout << "[Segment " << (segment + 1) << "] weights:";
        /* for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            cout << " w" << i << "=" << weight[i];
        }
        cout << endl; */
        //print_solution_stream(current_sol, cout);
        //print current elite set makespans
        cout << endl;
    }
    if (best_feasible_makespan < std::numeric_limits<double>::infinity()) {
        return best_feasible_solution;
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
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 5);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 50);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 200);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 50);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // modest k for small n
        } else if (n <= 100) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 5);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 50);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 200);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 50);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // moderate k for medium n
        } else {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 5);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 50);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 200);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 100);
            CFG_KNN_K = min(CFG_KNN_K, int(n/2)); // slightly smaller k for very large n
        }
    }

    // Precompute KNN lists (if K is zero, disable by building empty adjacency)
    if (CFG_KNN_K > 0) compute_knn_lists(CFG_KNN_K); else { KNN_LIST.assign(n + 1, {}); KNN_ADJ.assign(n + 1, vector<char>(n + 1, 0)); }

    // Pre-filter dronable customers by capacity/energy
    update_served_by_drone();

    //For another data-testing: change all deadline to a constant 3600 and all serving time to 0
    for (int i = 1; i <= n; ++i) {
        deadline[i] = 3600.0;
        serve_truck[i] = 0.0;
        serve_drone[i] = 0.0;
    }

    // Track best across attempts
    bool have_best = false;
    Solution best_overall_sol;
    double best_overall_initial_cost = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int attempt = 0; attempt < CFG_NUM_INITIAL; ++attempt) {
        Solution initial_solution = generate_initial_solution();
        
        Solution improved_sol = tabu_search(initial_solution, CFG_NUM_INITIAL);
        // Output both to stdout and to file
        cout.setf(std::ios::fixed); cout << setprecision(6);
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
            vd truck_metric = check_route_feasibility(r, 0.0, true);
            bool feas = (truck_metric[1] <= 1e-8 && truck_metric[2] <= 1e-8 && truck_metric[3] <= 1e-8);
            if (!feas) { final_feas = false; break; }
        }
        for (const vi &r : best_overall_sol.drone_routes) {
            vd truck_metric = check_route_feasibility(r, 0.0, false);
            bool feas = (truck_metric[1] <= 1e-8 && truck_metric[2] <= 1e-8 && truck_metric[3] <= 1e-8);
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