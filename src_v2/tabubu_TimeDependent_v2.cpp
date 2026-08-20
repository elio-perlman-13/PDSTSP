#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

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
double Dh = 400.0; // truck capacity (all trucks) (kg)
double vmax = 15.6464; // truck base speed (m/s)
int L = 24; //number of time segments in a day
//vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
//vd time_segments_sigma = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; //sigma (truck velocity coefficient) for each time segments
vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
vd time_segments_sigma = {0.9, 0.8, 0.4, 0.6,0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
vvd truck_vmax_ij; // truck_vmax_ij[i][j]: edge-specific base speed vmax_ij (m/s)
vector<vvd> truck_theta_ijl; // truck_theta_ijl[l][i][j]: edge/time-specific coefficient theta_ijl
double Dd = 2.27, E = 7200000.0; //drone's weight and energy capacities (for all drones)
double v_fly_drone = 31.3, v_take_off = 15.6, v_landing = 7.8; // speed of the drone
double height = 50; // height of the drone
//double height = 0; // height of the drone
//double power_beta = 0, power_gamma = 1.0; //coefficients for drone energy consumption per second
double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

// Candidate lists (k-nearest neighbors) to filter neighborhood evaluations
static int CFG_KNN_K = 1000;           // number of nearest neighbors per customer
static int CFG_KNN_WINDOW = 1;       // insertion window around candidate anchors
static vvi KNN_LIST;                 // KNN_LIST[i] = up to K nearest neighbor customer ids for i (exclude depot 0)
static vector<vector<char>> KNN_ADJ; // KNN_ADJ[i][j] = 1 if j in KNN_LIST[i]

// Simple tabu structure for relocate moves: tabu_list_switch[cust][target_vehicle] stores iteration until which move is tabu
// target_vehicle is 0..h-1 for trucks, h..h+d-1 for drones
static vector<vector<int>> tabu_list_switch; // sized (n+1) x (h + d), initialized on first use
static int TABU_TENURE_BASE = 0; // default tenure; actual update done in tabu loop (not here)
// Separate tabu structure for swap moves: store until-iteration for swapping a pair (min_id,max_id)
static vector<vector<int>> tabu_list_10; // sized (n+1) x (n+1)
static int TABU_TENURE_10 = 0; // default tenure for swap moves
static vector<vector<int>> tabu_list_11; // sized (n+1) x (h + d)
static int TABU_TENURE_11 = 0; // default tenure for relocate moves
// Separate tabu list for intra-route reinsert (Or-opt-1) moves
static map<vector<int>, int> tabu_list_20; // keyed by (cust_id1, cust_id2, vehicle_id)
static int TABU_TENURE_20 = 0; // default tenure for reinsert moves
// Separate tabu list for 2-opt moves: keyed by segment endpoints (min_id,max_id)
static vector<vector<int>> tabu_list_2opt; // sized (n+1) x (n+1) 
static int TABU_TENURE_2OPT = 0; // default tenure for 2-opt moves
static vector<vector<int>> tabu_list_2opt_star; // sized (n+1) x (n+1)
static int TABU_TENURE_2OPT_STAR = 0; // default tenure for 2-opt-star moves
static map<vector<int>, int> tabu_list_21; // keyed by (a,b,c,d) for (2,1) moves
static int TABU_TENURE_21 = 0; // default tenure for (2,1) moves
static map<vector<int>, int> tabu_list_22; // keyed by (a,b,c,d) for (2,2) moves
static int TABU_TENURE_22 = 0; // default tenure for (2,2) moves
static map<vector<int>, int> tabu_list_ejection; // keyed by sorted customer sequence
static int TABU_TENURE_EJECTION = 0; // default tenure for ejection chain moves
static map<vector<int>, int> tabu_list_fusion; // keyed by sorted customers in merged trips
static int TABU_TENURE_FUSION = 0; // default tenure for trip fusion moves
const int NUM_NEIGHBORHOODS = 9;
const int NUM_OF_INITIAL_SOLUTIONS = 200;
const int MAX_SEGMENT = 200;
const int MAX_NO_IMPROVE = 1000;
const int MAX_ITER_PER_SEGMENT = 1000;
const int MAX_ITERATIONS = MAX_SEGMENT * MAX_ITER_PER_SEGMENT;
static int H_MODE = 2;
static int H_DIV = 4;
static double gamma1 = 0.6;
static double gamma2 = 0.3;
static double gamma3 = 0.05;
static double gamma4 = 0.4;

// Runtime-configurable search knobs (initialized from compile-time defaults)
static int CFG_NUM_INITIAL = NUM_OF_INITIAL_SOLUTIONS;
static int CFG_MAX_SEGMENT = MAX_SEGMENT;
static int CFG_MAX_NO_IMPROVE = MAX_NO_IMPROVE;
static int CFG_MAX_ITER_PER_SEGMENT = MAX_ITER_PER_SEGMENT;
static int CFG_MAX_ITERATIONS = MAX_ITERATIONS;
static double CFG_TIME_LIMIT_SEC = 0.0; // 0 = unlimited
static string CFG_TRUCK_VMAX_FILE; // optional file: i j vmax_ij
static string CFG_TRUCK_THETA_FILE; // optional file: i j l theta_ijl, l is 0-based segment index
static double CFG_SEGMENT_LENGTH_SEC = 3600.0; // time-dependent segment length in seconds
static int CFG_OVERRIDE_TRUCKS = -1;
static int CFG_OVERRIDE_DRONES = -1;
static int CFG_RANDOM_SEED = 42;
static double CFG_C_TABU = 2.0;

// Adaptive penalty coefficients for constraint violations
static double PENALTY_LAMBDA_CAPACITY = 1.0;      // λ for capacity violations
static double PENALTY_LAMBDA_ENERGY = 1.0;        // λ for energy violations  
static double PENALTY_LAMBDA_DEADLINE = 1.0;      // λ for deadline violations

static double PENALTY_TARGET_VIOLATION = 0.01; // tau_v
static double PENALTY_ADAPTATION_RATE = 1.0;   // kappa

static double T0 = 0.5; // initial temperature for simulated annealing acceptance
double alpha = 0.998; // cooling rate for simulated annealing

// Destroy and repair helper
vvd edge_records; // edge_records[i][j]: stores working times for edge (i,j)
static double DESTROY_RATE = 0.2; // fraction of customers to remove during destroy phase

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

void update_tabu_tenures() {
    int base = max(1, (int)ceil(CFG_C_TABU * sqrt((double)n))); 
    
    TABU_TENURE_BASE = base;
    TABU_TENURE_10 = base;          // Swap
    TABU_TENURE_11 = base;          // Relocate
    TABU_TENURE_20 = base;          // Or-opt
    TABU_TENURE_2OPT = base;
    TABU_TENURE_2OPT_STAR = base; 
    TABU_TENURE_21 = base;
    TABU_TENURE_22 = base;
    TABU_TENURE_EJECTION = base;
    TABU_TENURE_FUSION = base;
    
    /* cout << "Dynamic Tabu Tenures set to: " << base 
         << " (2-opt: " << TABU_TENURE_2OPT 
         << ", Ejection: " << TABU_TENURE_EJECTION << ")" << endl; */
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

static int truck_time_segment_count() {
    return max(1, (int)time_segment.size() - 1);
}

static void configure_time_segment_boundaries() {
    int segment_count = max(1, (int)time_segments_sigma.size());
    double segment_length_hr = max(1e-9, CFG_SEGMENT_LENGTH_SEC / 3600.0);
    time_segment.assign(segment_count + 1, 0.0);
    for (int i = 0; i <= segment_count; ++i) {
        time_segment[i] = i * segment_length_hr;
    }
}

static bool parse_numeric_record(string line, vector<double>& values) {
    values.clear();
    size_t comment_pos = line.find('#');
    if (comment_pos != string::npos) line = line.substr(0, comment_pos);
    replace(line.begin(), line.end(), ',', ' ');
    stringstream ss(line);
    double value;
    while (ss >> value) values.push_back(value);
    return !values.empty();
}

static void initialize_time_dependent_truck_speed_model() {
    int segment_count = truck_time_segment_count();
    truck_vmax_ij.assign(n + 1, vd(n + 1, vmax));
    truck_theta_ijl.assign(segment_count, vvd(n + 1, vd(n + 1, 1.0)));

    for (int l = 0; l < segment_count; ++l) {
        double default_theta = (l < (int)time_segments_sigma.size()) ? time_segments_sigma[l] : 1.0;
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j <= n; ++j) {
                truck_theta_ijl[l][i][j] = default_theta;
            }
        }
    }
}

static void load_truck_vmax_ij_file(const string& filepath) {
    if (filepath.empty()) return;
    ifstream fin(filepath);
    if (!fin) {
        cerr << "Error: Cannot open truck vmax file " << filepath << endl;
        exit(1);
    }

    string line;
    vector<double> values;
    while (getline(fin, line)) {
        if (!parse_numeric_record(line, values)) continue;
        if (values.size() < 3) continue;
        int i = (int)values[0];
        int j = (int)values[1];
        double edge_vmax = values[2];
        if (i < 0 || i > n || j < 0 || j > n || edge_vmax <= 1e-8) continue;
        truck_vmax_ij[i][j] = edge_vmax;
    }
}

static void load_truck_theta_ijl_file(const string& filepath) {
    if (filepath.empty()) return;
    ifstream fin(filepath);
    if (!fin) {
        cerr << "Error: Cannot open truck theta file " << filepath << endl;
        exit(1);
    }

    int segment_count = truck_time_segment_count();
    string line;
    vector<double> values;
    while (getline(fin, line)) {
        if (!parse_numeric_record(line, values)) continue;
        if (values.size() < 4) continue;
        int i = (int)values[0];
        int j = (int)values[1];
        int l = (int)values[2];
        double theta = values[3];
        if (i < 0 || i > n || j < 0 || j > n || l < 0 || l >= segment_count || theta <= 1e-8) continue;
        truck_theta_ijl[l][i][j] = theta;
    }
}

static void configure_time_dependent_truck_speed_model() {
    initialize_time_dependent_truck_speed_model();
    load_truck_vmax_ij_file(CFG_TRUCK_VMAX_FILE);
    load_truck_theta_ijl_file(CFG_TRUCK_THETA_FILE);
}

static double get_truck_edge_speed(int from, int to, int segment_idx) {
    if (from < 0 || from > n || to < 0 || to > n) return vmax;
    double edge_vmax = vmax;
    if ((int)truck_vmax_ij.size() == n + 1 && (int)truck_vmax_ij[from].size() == n + 1) {
        edge_vmax = truck_vmax_ij[from][to];
    }

    double theta = 1.0;
    if (segment_idx >= 0 && segment_idx < (int)truck_theta_ijl.size() &&
        (int)truck_theta_ijl[segment_idx].size() == n + 1 &&
        (int)truck_theta_ijl[segment_idx][from].size() == n + 1) {
        theta = truck_theta_ijl[segment_idx][from][to];
    } else if (segment_idx >= 0 && segment_idx < (int)time_segments_sigma.size()) {
        theta = time_segments_sigma[segment_idx];
    }

    double speed = theta * edge_vmax;
    if (speed <= 1e-8) return max(vmax, 1.0);
    return speed;
}

// Helper: get time segment index for a given time t (in hours)
int get_time_segment(double t) {
    // t is in hours. Use custom time_segment boundaries (in hours):
    // time_segment: [b0, b1, ..., bk] defines k segments [b0,b1), [b1,b2), ... [b{k-1}, b{k}]
    // Return 0-based segment index in [0, k-1].
    // If outside boundaries, loop back to the start segment.
    if (time_segment.size() < 2) return 0;
    double period_hr = time_segment.back() - time_segment.front();
    if (period_hr <= 1e-12) period_hr = 12.0;
    t = fmod(t - time_segment.front(), period_hr);
    if (t < 0) t += period_hr;
    t += time_segment.front();
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
                int seg_safe = get_time_segment(time / 3600.0);
                double v_safe = get_truck_edge_speed(from, to, seg_safe);
                time += dist_left / v_safe;
                dist_left = 0.0;
                break;
            }
            // Convert time to hours for segment lookup
            double t_hr = time / 3600.0;
            int seg = get_time_segment(t_hr); // 0-based index into truck_theta_ijl
            double v = get_truck_edge_speed(from, to, seg); // v_ijl = theta_ijl * vmax_ij
            // Time left in this cyclic custom segment. If the trip goes beyond
            // one profile period, boundaries must advance to the current cycle.
            double period_hr = (time_segment.size() >= 2 && time_segment.back() > time_segment.front())
                               ? (time_segment.back() - time_segment.front())
                               : 12.0;
            double cycle_start_hr = floor((t_hr - time_segment.front()) / period_hr) * period_hr + time_segment.front();
            double next_boundary_hr = (seg + 1 < (int)time_segment.size())
                                      ? cycle_start_hr + (time_segment[seg + 1] - time_segment.front())
                                      : cycle_start_hr + period_hr;
            if (next_boundary_hr <= t_hr + 1e-12) next_boundary_hr += period_hr;
            double segment_end_time_sec = next_boundary_hr * 3600.0;
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
                    double deadline_norm = (deadline[cust] > 1e-6) ? deadline[cust] : 1.0;
                    deadline_feasible += (duration - deadline[cust]) / deadline_norm;
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
        if (from == to) continue;
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
        if (from == to) continue;
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
                    double deadline_norm = (deadline[cust] > 1e-6) ? deadline[cust] : 1.0;
                    deadline_feasible += (duration - deadline[cust]) / deadline_norm;
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
        auto [total_energy, energy_violation] = compute_drone_route_energy(route);
        if (energy_violation > 1e-8) {
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
struct NormalizedViolations {
    double capacity = 0.0;
    double energy = 0.0;
    double deadline = 0.0;
};

static double route_capacity_excess_sum(const vi& route, double capacity, int& trip_count) {
    double excess = 0.0;
    double load = 0.0;
    bool has_customer = false;
    for (int i = 1; i < (int)route.size(); ++i) {
        int node = route[i];
        if (node == 0) {
            if (has_customer) {
                excess += max(0.0, (load - capacity) / capacity);
                ++trip_count;
            }
            load = 0.0;
            has_customer = false;
        } else {
            load += demand[node];
            has_customer = true;
        }
    }
    if (has_customer) {
        excess += max(0.0, (load - capacity) / capacity);
        ++trip_count;
    }
    return excess;
}

static double drone_energy_excess_sum(const vi& route, int& drone_trip_count) {
    double excess = 0.0;
    vi trip{0};
    for (int i = 1; i < (int)route.size(); ++i) {
        int node = route[i];
        if (node == 0) {
            if (trip.size() > 1) {
                trip.push_back(0);
                double trip_energy = compute_drone_route_energy(trip).first;
                excess += max(0.0, (trip_energy - E) / E);
                ++drone_trip_count;
            }
            trip = {0};
        } else {
            trip.push_back(node);
        }
    }
    if (trip.size() > 1) {
        trip.push_back(0);
        double trip_energy = compute_drone_route_energy(trip).first;
        excess += max(0.0, (trip_energy - E) / E);
        ++drone_trip_count;
    }
    return excess;
}

static NormalizedViolations compute_normalized_violations(const Solution& sol) {
    double capacity_excess = 0.0;
    double energy_excess = 0.0;
    double deadline_excess = 0.0;
    int total_trip_count = 0;
    int drone_trip_count = 0;

    for (int i = 0; i < h; ++i) {
        capacity_excess += route_capacity_excess_sum(sol.truck_routes[i], Dh, total_trip_count);
        deadline_excess += compute_truck_route_time(sol.truck_routes[i], 0.0).second;
    }
    for (int i = 0; i < d; ++i) {
        capacity_excess += route_capacity_excess_sum(sol.drone_routes[i], Dd, total_trip_count);
        energy_excess += drone_energy_excess_sum(sol.drone_routes[i], drone_trip_count);
        deadline_excess += compute_drone_route_time(sol.drone_routes[i]).second;
    }

    NormalizedViolations v;
    v.capacity = (total_trip_count > 0) ? capacity_excess / total_trip_count : 0.0;
    v.energy = (drone_trip_count > 0) ? energy_excess / drone_trip_count : 0.0;
    v.deadline = (n > 0) ? deadline_excess / n : 0.0;
    return v;
}

static double penalty_multiplier_from_violations(const NormalizedViolations& v) {
    return 1.0
         + PENALTY_LAMBDA_CAPACITY * v.capacity
         + PENALTY_LAMBDA_ENERGY * v.energy
         + PENALTY_LAMBDA_DEADLINE * v.deadline;
}

double solution_score_workload(const Solution& sol) {
    NormalizedViolations v = compute_normalized_violations(sol);
    double penalty_multiplier = penalty_multiplier_from_violations(v);
    double sum_sq = 0.0;
    for (double t : sol.truck_route_times) sum_sq += t * t;
    for (double t : sol.drone_route_times) sum_sq += t * t;
    double l2_norm = std::sqrt(sum_sq);
    return l2_norm * penalty_multiplier;
}

double solution_score_l2_norm(const Solution& sol) {
    return solution_score_workload(sol);
}

double solution_score_makespan(const Solution& sol) {
    NormalizedViolations v = compute_normalized_violations(sol);
    double penalty_multiplier = penalty_multiplier_from_violations(v);
    return sol.total_makespan * penalty_multiplier;
}

double solution_score_total_time(const Solution& sol) {
    return solution_score_workload(sol);
}

double calculate_score_with_penalties(const double makespan, const double sum_sq, const double capacity_violation, const double energy_violation, const double deadline_violation) {
    (void)sum_sq;
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * capacity_violation
                                + PENALTY_LAMBDA_ENERGY * energy_violation
                                + PENALTY_LAMBDA_DEADLINE * deadline_violation;
    return makespan * penalty_multiplier;
}

void reset_penalty_coefficients() {
    PENALTY_LAMBDA_CAPACITY = 1.0;
    PENALTY_LAMBDA_ENERGY = 1.0;
    PENALTY_LAMBDA_DEADLINE = 1.0;
}

void update_penalties_from_segment(double avg_capacity_violation,
                                   double avg_energy_violation,
                                   double avg_deadline_violation) {
    PENALTY_LAMBDA_CAPACITY *= exp(PENALTY_ADAPTATION_RATE *
                                   (avg_capacity_violation - PENALTY_TARGET_VIOLATION));
    PENALTY_LAMBDA_ENERGY *= exp(PENALTY_ADAPTATION_RATE *
                                 (avg_energy_violation - PENALTY_TARGET_VIOLATION));
    PENALTY_LAMBDA_DEADLINE *= exp(PENALTY_ADAPTATION_RATE *
                                   (avg_deadline_violation - PENALTY_TARGET_VIOLATION));
}

vvi kmeans_clustering(int k, int max_iters=1000, uint64_t seed=UINT64_MAX) {
    if (n <= 0) return {};
    // Bound k to [1, n]
    if (k <= 0) k = 1;
    if (k > n) k = n;

    vvi clusters(k);
    vector<Point> centroids;
    centroids.reserve(k);

    // Random engine
    std::mt19937 gen(seed == UINT64_MAX ? std::random_device{}() : (uint32_t)seed);
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

Solution greedy_insert_customer(Solution sol, int customer, bool minimize_delta) {
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
           double objective_val = 0.0;
            if (minimize_delta) {
                // Minimize added time (Cheapest Insertion) -> Creates tight clusters
                double delta = new_metrics[0] - base_metrics[0];
                
                // Get the current total time of the vehicle we are considering
                double current_load = is_truck ? sol.truck_route_times[route_idx] : sol.drone_route_times[route_idx];
                
                // Add a tiny penalty based on current load.
                // If Deltas are equal (common for drones), this forces the algorithm to pick the emptier vehicle.
                // 1e-4 is small enough not to override true geographic efficiency.
                objective_val = delta + (current_load * 1e-3);
            } else {
                // Minimize global makespan -> Balances load (but can cause crossing)
                objective_val = new_makespan;
            }

            double new_score = objective_val * (1.0 + violation);
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
            double objective_val = 0.0;
            if (minimize_delta) {
                // Minimize added time (Cheapest Insertion) -> Creates tight clusters
                double delta = new_metrics[0] - base_metrics[0];
                
                // Get the current total time of the vehicle we are considering
                double current_load = is_truck ? sol.truck_route_times[route_idx] : sol.drone_route_times[route_idx];
                
                // Add a tiny penalty based on current load.
                // If Deltas are equal (common for drones), this forces the algorithm to pick the emptier vehicle.
                // 1e-4 is small enough not to override true geographic efficiency.
                objective_val = delta + (current_load * 1e-3);
            } else {
                // Minimize global makespan -> Balances load (but can cause crossing)
                objective_val = new_makespan;
            }

            double new_score = objective_val * (1.0 + violation);
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

Solution recalculate_solution(Solution sol);
bool check_solution_integrity(const Solution& sol);
static int compute_segment_count(int total_iters, int iters_per_segment);

static bool has_invalid_drone_assignment(const Solution& sol) {
    for (const vi& route : sol.drone_routes) {
        for (int customer : route) {
            if (customer != 0 && (customer < 0 || customer > n || !served_by_drone[customer])) {
                return true;
            }
        }
    }
    return false;
}

static bool is_feasible_solution(const Solution& sol) {
    return sol.deadline_violation <= 1e-8 &&
           sol.capacity_violation <= 1e-8 &&
           sol.energy_violation <= 1e-8 &&
           !has_invalid_drone_assignment(sol);
}

static double solution_score_from_cached_metrics(const Solution& sol, double (*solution_cost)(const Solution&)) {
    NormalizedViolations v{sol.capacity_violation, sol.energy_violation, sol.deadline_violation};
    double penalty_multiplier = penalty_multiplier_from_violations(v);
    if (solution_cost == solution_score_makespan) {
        return sol.total_makespan * penalty_multiplier;
    }
    if (solution_cost == solution_score_workload ||
        solution_cost == solution_score_l2_norm ||
        solution_cost == solution_score_total_time) {
        double sum_sq = 0.0;
        for (double t : sol.truck_route_times) sum_sq += t * t;
        for (double t : sol.drone_route_times) sum_sq += t * t;
        return std::sqrt(sum_sq) * penalty_multiplier;
    }
    return solution_cost(sol);
}

static double score_recalculated_candidate(Solution& candidate, double (*solution_cost)(const Solution&)) {
    candidate = recalculate_solution(candidate);
    return solution_score_from_cached_metrics(candidate, solution_cost);
}

struct RouteViolationSummary {
    double capacity_excess = 0.0;
    int trip_count = 0;
    double energy_excess = 0.0;
    int drone_trip_count = 0;
    double deadline_excess = 0.0;
};

struct RouteEvaluation {
    double time = 0.0;
    RouteViolationSummary violation;
};

static RouteEvaluation evaluate_route_for_scoring(const vi& route, bool is_truck, double start_time = 0.0) {
    RouteEvaluation eval;
    vd metrics = check_route_feasibility(route, start_time, is_truck);
    eval.time = metrics[0];
    eval.violation.deadline_excess = metrics[1];
    eval.violation.capacity_excess = route_capacity_excess_sum(route, is_truck ? Dh : Dd, eval.violation.trip_count);
    if (!is_truck) {
        eval.violation.energy_excess = drone_energy_excess_sum(route, eval.violation.drone_trip_count);
    }
    return eval;
}

static RouteViolationSummary combine_route_violations(const RouteViolationSummary& a, const RouteViolationSummary& b) {
    RouteViolationSummary result;
    result.capacity_excess = a.capacity_excess + b.capacity_excess;
    result.trip_count = a.trip_count + b.trip_count;
    result.energy_excess = a.energy_excess + b.energy_excess;
    result.drone_trip_count = a.drone_trip_count + b.drone_trip_count;
    result.deadline_excess = a.deadline_excess + b.deadline_excess;
    return result;
}

struct ChangedVehicle {
    bool is_truck = true;
    int idx = -1;
    int first_changed_pos = 0;
};

struct RoutePrefixCache {
    vector<RouteEvaluation> at_position;
    vector<char> valid;
};

struct SolutionEvaluationContext {
    vector<RouteViolationSummary> truck_violations;
    vector<RouteViolationSummary> drone_violations;
    vector<RoutePrefixCache> truck_prefix_cache;
    vector<RoutePrefixCache> drone_prefix_cache;
    vd truck_times;
    vd drone_times;
    double capacity_excess = 0.0;
    double energy_excess = 0.0;
    double deadline_excess = 0.0;
    int trip_count = 0;
    int drone_trip_count = 0;

    explicit SolutionEvaluationContext(const Solution& sol) {
        truck_violations.assign(h, RouteViolationSummary{});
        drone_violations.assign(d, RouteViolationSummary{});
        truck_prefix_cache.assign(h, RoutePrefixCache{});
        drone_prefix_cache.assign(d, RoutePrefixCache{});
        truck_times.assign(h, 0.0);
        drone_times.assign(d, 0.0);

        for (int i = 0; i < h; ++i) {
            RouteEvaluation eval = evaluate_route_for_scoring(sol.truck_routes[i], true);
            truck_times[i] = eval.time;
            truck_violations[i] = eval.violation;
            truck_prefix_cache[i] = build_prefix_cache(sol.truck_routes[i], true);
            add(eval.violation);
        }
        for (int i = 0; i < d; ++i) {
            RouteEvaluation eval = evaluate_route_for_scoring(sol.drone_routes[i], false);
            drone_times[i] = eval.time;
            drone_violations[i] = eval.violation;
            drone_prefix_cache[i] = build_prefix_cache(sol.drone_routes[i], false);
            add(eval.violation);
        }
    }

    static RoutePrefixCache build_prefix_cache(const vi& route, bool is_truck) {
        RoutePrefixCache cache;
        cache.at_position.assign(route.size(), RouteEvaluation{});
        cache.valid.assign(route.size(), 0);
        for (int pos = 0; pos < (int)route.size(); ++pos) {
            if (route[pos] != 0) continue;
            vi prefix(route.begin(), route.begin() + pos + 1);
            RouteEvaluation eval = evaluate_route_for_scoring(prefix, is_truck);
            cache.at_position[pos] = eval;
            cache.valid[pos] = 1;
        }
        return cache;
    }

    void add(const RouteViolationSummary& v) {
        capacity_excess += v.capacity_excess;
        energy_excess += v.energy_excess;
        deadline_excess += v.deadline_excess;
        trip_count += v.trip_count;
        drone_trip_count += v.drone_trip_count;
    }

    void subtract(const RouteViolationSummary& v) {
        capacity_excess -= v.capacity_excess;
        energy_excess -= v.energy_excess;
        deadline_excess -= v.deadline_excess;
        trip_count -= v.trip_count;
        drone_trip_count -= v.drone_trip_count;
    }

    static int restart_depot_position(const vi& route, int first_changed_pos) {
        if (route.empty()) return 0;
        int pos = min(max(0, first_changed_pos - 1), (int)route.size() - 1);
        while (pos > 0 && route[pos] != 0) --pos;
        return pos;
    }

    static RouteEvaluation evaluate_changed_route(const vi& route, bool is_truck, const RoutePrefixCache& cache, int first_changed_pos) {
        if (route.empty()) return RouteEvaluation{};
        int restart_pos = restart_depot_position(route, first_changed_pos);
        RouteEvaluation prefix;
        if (restart_pos < (int)cache.valid.size() && cache.valid[restart_pos]) {
            prefix = cache.at_position[restart_pos];
        } else {
            vi prefix_route(route.begin(), route.begin() + restart_pos + 1);
            prefix = evaluate_route_for_scoring(prefix_route, is_truck);
        }

        vi suffix_route(route.begin() + restart_pos, route.end());
        RouteEvaluation suffix = evaluate_route_for_scoring(suffix_route, is_truck, prefix.time);
        RouteEvaluation result;
        result.time = prefix.time + suffix.time;
        result.violation = combine_route_violations(prefix.violation, suffix.violation);
        return result;
    }

    double score_candidate(Solution& candidate, const vector<ChangedVehicle>& changed, double (*solution_cost)(const Solution&)) const {
        double new_capacity_excess = capacity_excess;
        double new_energy_excess = energy_excess;
        double new_deadline_excess = deadline_excess;
        int new_trip_count = trip_count;
        int new_drone_trip_count = drone_trip_count;

        candidate.truck_route_times = truck_times;
        candidate.drone_route_times = drone_times;

        vector<char> seen_truck(h, 0);
        vector<char> seen_drone(d, 0);
        for (const ChangedVehicle& change : changed) {
            if (change.is_truck) {
                if (change.idx < 0 || change.idx >= h || seen_truck[change.idx]) continue;
                seen_truck[change.idx] = 1;
                const RouteViolationSummary& old_v = truck_violations[change.idx];
                new_capacity_excess -= old_v.capacity_excess;
                new_energy_excess -= old_v.energy_excess;
                new_deadline_excess -= old_v.deadline_excess;
                new_trip_count -= old_v.trip_count;
                new_drone_trip_count -= old_v.drone_trip_count;

                RouteEvaluation eval = evaluate_changed_route(candidate.truck_routes[change.idx], true, truck_prefix_cache[change.idx], change.first_changed_pos);
                candidate.truck_route_times[change.idx] = eval.time;
                new_capacity_excess += eval.violation.capacity_excess;
                new_energy_excess += eval.violation.energy_excess;
                new_deadline_excess += eval.violation.deadline_excess;
                new_trip_count += eval.violation.trip_count;
                new_drone_trip_count += eval.violation.drone_trip_count;
            } else {
                if (change.idx < 0 || change.idx >= d || seen_drone[change.idx]) continue;
                seen_drone[change.idx] = 1;
                const RouteViolationSummary& old_v = drone_violations[change.idx];
                new_capacity_excess -= old_v.capacity_excess;
                new_energy_excess -= old_v.energy_excess;
                new_deadline_excess -= old_v.deadline_excess;
                new_trip_count -= old_v.trip_count;
                new_drone_trip_count -= old_v.drone_trip_count;

                RouteEvaluation eval = evaluate_changed_route(candidate.drone_routes[change.idx], false, drone_prefix_cache[change.idx], change.first_changed_pos);
                candidate.drone_route_times[change.idx] = eval.time;
                new_capacity_excess += eval.violation.capacity_excess;
                new_energy_excess += eval.violation.energy_excess;
                new_deadline_excess += eval.violation.deadline_excess;
                new_trip_count += eval.violation.trip_count;
                new_drone_trip_count += eval.violation.drone_trip_count;
            }
        }

        candidate.total_makespan = 0.0;
        for (double t : candidate.truck_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
        candidate.capacity_violation = (new_trip_count > 0) ? new_capacity_excess / new_trip_count : 0.0;
        candidate.energy_violation = (new_drone_trip_count > 0) ? new_energy_excess / new_drone_trip_count : 0.0;
        candidate.deadline_violation = (n > 0) ? new_deadline_excess / n : 0.0;
        return solution_score_from_cached_metrics(candidate, solution_cost);
    }
};

Solution generate_initial_solution_v2(uint64_t seed = UINT64_MAX) {
    Solution sol;
    sol.truck_routes.assign(h, vi{0, 0});
    sol.drone_routes.assign(d, vi{0, 0});
    sol.truck_route_times.assign(h, 0.0);
    sol.drone_route_times.assign(d, 0.0);
    sol.total_makespan = 0.0;
    sol.capacity_violation = 0.0;
    sol.energy_violation = 0.0;
    sol.deadline_violation = 0.0;

    auto route_violation = [](const vd& metrics) {
        if (metrics.size() < 4) return 1e18;
        return metrics[1] + metrics[2] + metrics[3];
    };

    auto normalize_init_route = [](vi route) {
        vi cleaned;
        cleaned.reserve(route.size() + 2);
        if (route.empty() || route.front() != 0) cleaned.push_back(0);
        for (int node : route) {
            if (node == 0 && !cleaned.empty() && cleaned.back() == 0) continue;
            cleaned.push_back(node);
        }
        if (cleaned.empty() || cleaned.front() != 0) cleaned.insert(cleaned.begin(), 0);
        if (cleaned.back() != 0) cleaned.push_back(0);
        return cleaned;
    };

    auto last_depot_pos = [](const vi& route) {
        for (int i = (int)route.size() - 1; i >= 0; --i) {
            if (route[i] == 0) return i;
        }
        return 0;
    };

    auto current_sortie = [&](const vi& route) {
        vi sortie{0};
        int start = last_depot_pos(route);
        for (int i = start + 1; i < (int)route.size(); ++i) {
            if (route[i] != 0) sortie.push_back(route[i]);
        }
        return sortie;
    };

    struct Builder {
        bool is_truck = true;
        int cluster = 0;
        vi route{0};
        bool active = true;
        double completion = 0.0;
        double payload = 0.0;
        double energy = 0.0;
        double wait_budget = 1e18;
    };

    struct Candidate {
        int customer = -1;
        double t_to = 0.0;
        double t_back = 0.0;
        double urgency = 0.0;
        double cap_ratio = 0.0;
        double energy_ratio = 0.0;
        double extension = 0.0;
        bool same_cluster = true;
    };

    int cluster_count = max(1, h);
    vvi clusters = kmeans_clustering(cluster_count, 1000, seed);
    vi cluster_assignment(n + 1, 0);
    for (int c = 0; c < (int)clusters.size(); ++c) {
        for (int cust : clusters[c]) cluster_assignment[cust] = c;
    }

    vector<Builder> builders;
    builders.reserve(h + (d > 0 ? cluster_count : 0));
    for (int i = 0; i < h; ++i) {
        Builder b;
        b.is_truck = true;
        b.cluster = min(i, cluster_count - 1);
        builders.push_back(b);
    }
    if (d > 0) {
        for (int i = 0; i < cluster_count; ++i) {
            Builder b;
            b.is_truck = false;
            b.cluster = i;
            builders.push_back(b);
        }
    }

    vector<char> assigned(n + 1, 0);
    int assigned_count = 0;

    auto compute_leg_time = [&](bool is_truck, int from, int to, double start_time) {
        if (from == to) return 0.0;
        if (is_truck) return compute_truck_route_time({from, to}, start_time).first;
        return compute_drone_route_time({from, to}).first;
    };

    auto route_with_candidate_return = [](vi route, int customer) {
        if (route.empty()) route.push_back(0);
        route.push_back(customer);
        route.push_back(0);
        return route;
    };

    auto drone_sortie_energy_ratio = [&](const Builder& b, int customer) {
        vi sortie = current_sortie(b.route);
        sortie.push_back(customer);
        sortie.push_back(0);
        return compute_drone_route_energy(sortie).first / E;
    };

    auto generate_candidates = [&](const Builder& b, bool restrict_to_cluster) {
        vector<Candidate> candidates;
        if (!b.active) return candidates;
        int current_node = b.route.empty() ? 0 : b.route.back();
        double direct_return = compute_leg_time(b.is_truck, current_node, 0, b.completion);

        for (int cust = 1; cust <= n; ++cust) {
            if (assigned[cust]) continue;
            if (restrict_to_cluster && cluster_assignment[cust] != b.cluster) continue;
            if (!b.is_truck && !served_by_drone[cust]) continue;

            double capacity = b.is_truck ? Dh : Dd;
            double cap_ratio = (b.payload + demand[cust]) / capacity;
            if (cap_ratio > 1.0 + 1e-9) continue;

            double t_to = compute_leg_time(b.is_truck, current_node, cust, b.completion);
            double t_back = compute_leg_time(b.is_truck, cust, 0, b.completion + t_to);
            double residual_budget = min(b.wait_budget - t_to, deadline[cust]);
            if (residual_budget <= 1e-9) continue;
            double urgency = t_back / residual_budget;
            if (urgency > 1.0 + 1e-9) continue;

            vi candidate_route = route_with_candidate_return(b.route, cust);
            vd metrics = check_route_feasibility(candidate_route, 0.0, b.is_truck);
            if (route_violation(metrics) > 1e-9) continue;

            Candidate cand;
            cand.customer = cust;
            cand.t_to = t_to;
            cand.t_back = t_back;
            cand.urgency = urgency;
            cand.cap_ratio = cap_ratio;
            cand.energy_ratio = b.is_truck ? 0.0 : drone_sortie_energy_ratio(b, cust);
            cand.extension = t_to + t_back - direct_return;
            cand.same_cluster = (cluster_assignment[cust] == b.cluster);
            candidates.push_back(cand);
        }
        return candidates;
    };

    auto mad_value = [](const vector<double>& vals) {
        if (vals.empty()) return 0.0;
        double mean = accumulate(vals.begin(), vals.end(), 0.0) / vals.size();
        double mad = 0.0;
        for (double v : vals) mad += fabs(v - mean);
        mad /= vals.size();
        return mad;
    };

    auto select_best_candidate = [&](const vector<Candidate>& candidates, bool is_truck) {
        vector<double> urgencies, capacities, energies, extensions;
        urgencies.reserve(candidates.size());
        capacities.reserve(candidates.size());
        energies.reserve(candidates.size());
        extensions.reserve(candidates.size());
        double min_ext = 1e18, max_ext = -1e18;
        for (const Candidate& cand : candidates) {
            urgencies.push_back(cand.urgency * cand.urgency);
            capacities.push_back(cand.cap_ratio * cand.cap_ratio);
            if (!is_truck) energies.push_back(cand.energy_ratio * cand.energy_ratio);
            min_ext = min(min_ext, cand.extension);
            max_ext = max(max_ext, cand.extension);
        }
        for (const Candidate& cand : candidates) {
            double norm_ext = (max_ext - min_ext <= 1e-9) ? 0.0 : (cand.extension - min_ext) / (max_ext - min_ext);
            extensions.push_back(norm_ext);
        }

        double w1 = mad_value(urgencies);
        double w2 = mad_value(capacities);
        double w3 = is_truck ? 0.0 : mad_value(energies);
        double w4 = mad_value(extensions);
        double w_sum = w1 + w2 + w3 + w4;
        if (w_sum <= 1e-12) {
            double equal_weight = is_truck ? 1.0 / 3.0 : 1.0 / 4.0;
            w1 = equal_weight;
            w2 = equal_weight;
            w3 = is_truck ? 0.0 : equal_weight;
            w4 = equal_weight;
        } else {
            w1 /= w_sum;
            w2 /= w_sum;
            w3 /= w_sum;
            w4 /= w_sum;
        }

        int best_idx = -1;
        double best_score = 1e300;
        int tie_count = 0;
        const double eta = 1.0;
        for (int i = 0; i < (int)candidates.size(); ++i) {
            const Candidate& cand = candidates[i];
            double norm_ext = (max_ext - min_ext <= 1e-9) ? 0.0 : (cand.extension - min_ext) / (max_ext - min_ext);
            double cluster_penalty = cand.same_cluster ? 0.0 : eta;
            double score = w1 * cand.urgency * cand.urgency
                         + w2 * cand.cap_ratio * cand.cap_ratio
                         + w3 * cand.energy_ratio * cand.energy_ratio
                         + w4 * norm_ext
                         + cluster_penalty;
            if (score + 1e-12 < best_score) {
                best_score = score;
                best_idx = i;
                tie_count = 1;
            } else if (fabs(score - best_score) <= 1e-12) {
                ++tie_count;
                if (rand() % tie_count == 0) best_idx = i;
            }
        }
        return best_idx;
    };

    auto extend_builder = [&](Builder& b, const Candidate& cand) {
        b.route.push_back(cand.customer);
        b.completion += cand.t_to;
        b.payload += demand[cand.customer];
        if (!b.is_truck) {
            vi sortie = current_sortie(b.route);
            b.energy = compute_drone_route_energy(sortie).first;
        }
        b.wait_budget = min(b.wait_budget - cand.t_to, deadline[cand.customer]);
        assigned[cand.customer] = 1;
        ++assigned_count;
    };

    auto close_trip = [&](Builder& b) {
        int current_node = b.route.empty() ? 0 : b.route.back();
        if (current_node != 0) {
            b.completion += compute_leg_time(b.is_truck, current_node, 0, b.completion);
            b.route.push_back(0);
        }
        b.payload = 0.0;
        b.energy = 0.0;
        b.wait_budget = 1e18;
    };

    auto seed_builder = [&](Builder& b) {
        vector<Candidate> candidates = generate_candidates(b, true);
        if (candidates.empty()) return false;
        int best = select_best_candidate(candidates, b.is_truck);
        if (best < 0) return false;
        extend_builder(b, candidates[best]);
        return true;
    };

    for (Builder& b : builders) {
        seed_builder(b);
    }

    int stall_count = 0;
    int max_iters = max(10000, 20 * max(1, n + (int)builders.size()));
    while (assigned_count < n && max_iters-- > 0) {
        int best_builder = -1;
        double best_completion = 1e300;
        for (int i = 0; i < (int)builders.size(); ++i) {
            if (!builders[i].active) continue;
            if (builders[i].completion + 1e-9 < best_completion) {
                best_completion = builders[i].completion;
                best_builder = i;
            }
        }
        if (best_builder < 0) break;

        Builder& b = builders[best_builder];
        vector<Candidate> candidates = generate_candidates(b, false);
        if (!candidates.empty()) {
            int best = select_best_candidate(candidates, b.is_truck);
            if (best >= 0) {
                extend_builder(b, candidates[best]);
                stall_count = 0;
                continue;
            }
        }

        int current_node = b.route.empty() ? 0 : b.route.back();
        if (current_node != 0) {
            close_trip(b);
            stall_count = 0;
        } else {
            b.active = false;
            if (++stall_count > (int)builders.size() + 5) break;
        }
    }

    for (Builder& b : builders) {
        close_trip(b);
    }

    sol.truck_routes.assign(h, vi{0});
    int truck_idx = 0;
    for (const Builder& b : builders) {
        if (!b.is_truck) continue;
        if (truck_idx < h) sol.truck_routes[truck_idx++] = b.route;
    }

    struct Sortie {
        vi route;
        double duration = 0.0;
    };
    vector<Sortie> sorties;
    for (const Builder& b : builders) {
        if (b.is_truck) continue;
        vi cur{0};
        for (int i = 1; i < (int)b.route.size(); ++i) {
            int node = b.route[i];
            if (node == 0) {
                if (cur.size() > 1) {
                    cur.push_back(0);
                    sorties.push_back({cur, compute_drone_route_time(cur).first});
                }
                cur = {0};
            } else {
                cur.push_back(node);
            }
        }
        if (cur.size() > 1) {
            cur.push_back(0);
            sorties.push_back({cur, compute_drone_route_time(cur).first});
        }
    }

    sol.drone_routes.assign(d, vi{0});
    if (d > 0 && !sorties.empty()) {
        vector<int> order(sorties.size());
        iota(order.begin(), order.end(), 0);
        sort(order.begin(), order.end(), [&](int a, int b) {
            return sorties[a].duration > sorties[b].duration;
        });

        struct Load {
            double time = 0.0;
            int id = 0;
        };
        struct LoadCmp {
            bool operator()(const Load& a, const Load& b) const {
                return a.time > b.time;
            }
        };
        priority_queue<Load, vector<Load>, LoadCmp> pq;
        for (int i = 0; i < d; ++i) pq.push({0.0, i});
        vector<double> loads(d, 0.0);
        for (int sortie_id : order) {
            Load cur = pq.top();
            pq.pop();
            vi& route = sol.drone_routes[cur.id];
            if (route.empty()) route.push_back(0);
            route.insert(route.end(), sorties[sortie_id].route.begin() + 1, sorties[sortie_id].route.end());
            loads[cur.id] += sorties[sortie_id].duration;
            pq.push({loads[cur.id], cur.id});
        }
    }

    for (vi& route : sol.truck_routes) route = normalize_init_route(route);
    for (vi& route : sol.drone_routes) route = normalize_init_route(route);

    struct InsertionMove {
        bool found = false;
        bool is_truck = true;
        int vehicle_idx = -1;
        int pos = -1;
        vi route;
        vd metrics;
        double score = 1e300;
        double violation = 1e300;
        double new_makespan = 1e300;
    };

    auto better_insertion = [](double violation, double new_makespan, double score, const InsertionMove& best) {
        if (!best.found) return true;
        if (score + 1e-9 < best.score) return true;
        if (fabs(score - best.score) > 1e-9) return false;
        if (violation + 1e-9 < best.violation) return true;
        if (fabs(violation - best.violation) > 1e-9) return false;
        return new_makespan + 1e-9 < best.new_makespan;
    };
    auto fallback_insertion_score = [&](bool is_truck, int vehicle_idx, const vi& candidate_route) {
        Solution candidate = sol;
        if (is_truck) candidate.truck_routes[vehicle_idx] = candidate_route;
        else candidate.drone_routes[vehicle_idx] = candidate_route;
        candidate = recalculate_solution(candidate);
        double score = solution_score_makespan(candidate);
        double violation = PENALTY_LAMBDA_DEADLINE * candidate.deadline_violation
                         + PENALTY_LAMBDA_ENERGY * candidate.energy_violation
                         + PENALTY_LAMBDA_CAPACITY * candidate.capacity_violation;
        return tuple<double, double, double>{score, violation, candidate.total_makespan};
    };

    auto consider_route = [&](InsertionMove& best, int cust, bool is_truck, int vehicle_idx, const vi& base_route) {
        if (!is_truck && !served_by_drone[cust]) return;
        if (!is_truck && d <= 0) return;
        if (!is_truck) {
            vi normalized_route = base_route;
            if (normalized_route.empty()) normalized_route = {0, 0};
            if (normalized_route.front() != 0) normalized_route.insert(normalized_route.begin(), 0);
            if (normalized_route.back() != 0) normalized_route.push_back(0);

            auto evaluate_drone_route = [&](vi candidate_route, int pos) {
                vd metrics = check_route_feasibility(candidate_route, 0.0, false);
                auto [score, violation, new_makespan] = fallback_insertion_score(false, vehicle_idx, candidate_route);

                if (better_insertion(violation, new_makespan, score, best)) {
                    best.found = true;
                    best.is_truck = false;
                    best.vehicle_idx = vehicle_idx;
                    best.pos = pos;
                    best.route = std::move(candidate_route);
                    best.metrics = std::move(metrics);
                    best.score = score;
                    best.violation = violation;
                    best.new_makespan = new_makespan;
                }
            };

            // Try inserting inside every existing drone trip. A route can contain
            // multiple trips separated by depot nodes, e.g. 0 a b 0 c 0.
            for (int pos = 1; pos < (int)normalized_route.size(); ++pos) {
                vi candidate_route = normalized_route;
                candidate_route.insert(candidate_route.begin() + pos, cust);
                evaluate_drone_route(std::move(candidate_route), pos);
            }

            // Also try opening a new trip after the last depot.
            vi new_sortie_route = normalized_route;
            if (new_sortie_route.size() == 2 && new_sortie_route[0] == 0 && new_sortie_route[1] == 0) {
                new_sortie_route = {0, cust, 0};
            } else {
                new_sortie_route.push_back(cust);
                new_sortie_route.push_back(0);
            }
            evaluate_drone_route(std::move(new_sortie_route), (int)normalized_route.size());
            return;
        }
        int first_insert_pos = 1;
        int last_insert_pos = max(1, (int)base_route.size() - 1);
        for (int pos = first_insert_pos; pos <= last_insert_pos; ++pos) {
            vi candidate_route = base_route;
            if (candidate_route.empty()) candidate_route = {0, 0};
            if (candidate_route.front() != 0) candidate_route.insert(candidate_route.begin(), 0);
            if (candidate_route.back() != 0) candidate_route.push_back(0);
            candidate_route.insert(candidate_route.begin() + pos, cust);

            vd metrics = check_route_feasibility(candidate_route, 0.0, is_truck);
            auto [score, violation, new_makespan] = fallback_insertion_score(is_truck, vehicle_idx, candidate_route);

            if (better_insertion(violation, new_makespan, score, best)) {
                best.found = true;
                best.is_truck = is_truck;
                best.vehicle_idx = vehicle_idx;
                best.pos = pos;
                best.route = std::move(candidate_route);
                best.metrics = std::move(metrics);
                best.score = score;
                best.violation = violation;
                best.new_makespan = new_makespan;
            }
        }
    };

    sol = recalculate_solution(sol);

    for (int cust = 1; cust <= n; ++cust) {
        if (assigned[cust]) continue;
        InsertionMove best;
        for (int i = 0; i < h; ++i) {
            consider_route(best, cust, true, i, sol.truck_routes[i]);
        }
        for (int i = 0; i < d; ++i) {
            consider_route(best, cust, false, i, sol.drone_routes[i]);
        }

        if (!best.found) {
            cerr << "Warning: no insertion move found for customer " << cust << "; appending to truck 0.\n";
            if (h > 0) {
                vi route = sol.truck_routes[0];
                if (route.empty()) route = {0, 0};
                if (route.back() != 0) route.push_back(0);
                route.insert(route.end() - 1, cust);
                sol.truck_routes[0] = route;
                sol.truck_route_times[0] = check_route_feasibility(route, 0.0, true)[0];
            }
            continue;
        }

        if (best.is_truck) {
            sol.truck_routes[best.vehicle_idx] = best.route;
            sol.truck_route_times[best.vehicle_idx] = best.metrics[0];
        } else {
            sol.drone_routes[best.vehicle_idx] = best.route;
            sol.drone_route_times[best.vehicle_idx] = best.metrics[0];
        }
        sol = recalculate_solution(sol);
    }

    for (vi& route : sol.truck_routes) route = normalize_init_route(route);
    for (vi& route : sol.drone_routes) route = normalize_init_route(route);
    sol = recalculate_solution(sol);
    if (!check_solution_integrity(sol)) {
        cerr << "Warning: v2 initial solution failed integrity check.\n";
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
    // Drone 3 is indexed as h + 2. => returns (2, false)
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
        double score = base_time * penalty_multiplier;

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

vector<int> critical_vehicle_indices_by_makespan(const Solution& sol) {
    vector<int> critical;
    double cmax = 0.0;
    for (double t : sol.truck_route_times) cmax = max(cmax, t);
    for (double t : sol.drone_route_times) cmax = max(cmax, t);

    const double eps = 1e-8;
    for (int i = 0; i < h && i < (int)sol.truck_route_times.size(); ++i) {
        if (fabs(sol.truck_route_times[i] - cmax) <= eps) critical.push_back(i);
    }
    for (int i = 0; i < d && i < (int)sol.drone_route_times.size(); ++i) {
        if (fabs(sol.drone_route_times[i] - cmax) <= eps) critical.push_back(h + i);
    }

    if (critical.empty()) {
        auto [idx, is_truck] = critical_solution_index(sol);
        critical.push_back(is_truck ? idx : h + idx);
    }
    return critical;
}

int tabu_expiration_iteration(int current_iter, int base_tenure) {
    if (base_tenure <= 0) return current_iter;
    int tenure = base_tenure + (rand() % (base_tenure + 1));
    return current_iter + tenure;
}

bool admissible_by_tabu_or_aspiration(bool is_tabu, double candidate_score, double segment_reference_score) {
    return !is_tabu || candidate_score + 1e-8 < segment_reference_score;
}

bool better_score_with_random_tie(double candidate_score, double incumbent_score, int& tie_count) {
    if (candidate_score + 1e-8 < incumbent_score) {
        tie_count = 1;
        return true;
    }
    if (fabs(candidate_score - incumbent_score) <= 1e-8) {
        ++tie_count;
        return rand() % tie_count == 0;
    }
    return false;
}

enum class TabuAttributeKind {
    Matrix10,
    Matrix11,
    Matrix2Opt,
    Matrix2OptStar,
    Map20,
    Map21,
    Map22,
    Ejection,
    Fusion
};

struct PendingTabuAttribute {
    TabuAttributeKind kind;
    vector<int> key;
    int base_tenure;
};

static vector<PendingTabuAttribute> pending_tabu_attributes;

void stage_tabu_attribute(TabuAttributeKind kind, vector<int> key, int base_tenure) {
    pending_tabu_attributes.push_back({kind, std::move(key), base_tenure});
}

void clear_pending_tabu_attributes() {
    pending_tabu_attributes.clear();
}

void commit_pending_tabu_attributes(int current_iter) {
    for (const auto& attr : pending_tabu_attributes) {
        int expires_at = tabu_expiration_iteration(current_iter, attr.base_tenure);
        switch (attr.kind) {
            case TabuAttributeKind::Matrix10:
                if (attr.key.size() >= 2 && attr.key[0] >= 0 && attr.key[0] < (int)tabu_list_10.size() &&
                    attr.key[1] >= 0 && attr.key[1] < (int)tabu_list_10[attr.key[0]].size()) {
                    tabu_list_10[attr.key[0]][attr.key[1]] = expires_at;
                }
                break;
            case TabuAttributeKind::Matrix11:
                if (attr.key.size() >= 2 && attr.key[0] >= 0 && attr.key[0] < (int)tabu_list_11.size() &&
                    attr.key[1] >= 0 && attr.key[1] < (int)tabu_list_11[attr.key[0]].size()) {
                    tabu_list_11[attr.key[0]][attr.key[1]] = expires_at;
                }
                break;
            case TabuAttributeKind::Matrix2Opt:
                if (attr.key.size() >= 2 && attr.key[0] >= 0 && attr.key[0] < (int)tabu_list_2opt.size() &&
                    attr.key[1] >= 0 && attr.key[1] < (int)tabu_list_2opt[attr.key[0]].size()) {
                    tabu_list_2opt[attr.key[0]][attr.key[1]] = expires_at;
                }
                break;
            case TabuAttributeKind::Matrix2OptStar:
                if (attr.key.size() >= 2 && attr.key[0] >= 0 && attr.key[0] < (int)tabu_list_2opt_star.size() &&
                    attr.key[1] >= 0 && attr.key[1] < (int)tabu_list_2opt_star[attr.key[0]].size()) {
                    tabu_list_2opt_star[attr.key[0]][attr.key[1]] = expires_at;
                }
                break;
            case TabuAttributeKind::Map20:
                tabu_list_20[attr.key] = expires_at;
                break;
            case TabuAttributeKind::Map21:
                tabu_list_21[attr.key] = expires_at;
                break;
            case TabuAttributeKind::Map22:
                tabu_list_22[attr.key] = expires_at;
                break;
            case TabuAttributeKind::Ejection:
                tabu_list_ejection[attr.key] = expires_at;
                break;
            case TabuAttributeKind::Fusion:
                tabu_list_fusion[attr.key] = expires_at;
                break;
        }
    }
    clear_pending_tabu_attributes();
}

Solution local_search(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost, double (*solution_cost)(const Solution&)) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 1e10;
    SolutionEvaluationContext score_context(initial_solution);
    clear_pending_tabu_attributes();
    // Depending on neighbor_id, implement different neighborhood structures
    if (neighbor_id == 0) {
        // Relocate 1 customer from the critical (longest-time) vehicle route to another route of the same mode
        // 1) Identify critical vehicle (truck or drone) using precomputed times
        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

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
        int best_neighbor_tie_count = 0;

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Lambda to normalize routes for comparison (detect no-ops)
            auto normalize_route = []( vi& route) -> vi {
                if (!route.empty() && route.front() != 0) 
                    route.insert(route.begin(), 0);
                if (!route.empty() && route.back() != 0) 
                    route.push_back(0);
                return route;
            };

            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust = base_route[p];

                // Pre-calculate the base route with customer removed (for inter-route moves)
                vi base_route_removed = base_route;
                base_route_removed.erase(base_route_removed.begin() + p);
                
	                // Try relocating cust to other vehicles
	                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
	                    if (served_by_drone[cust] == 0 && target_veh >= h) continue; // cannot assign to drone
	                    bool is_tabu = (tabu_list_10[cust][target_veh] > current_iter);
                    
                    if (target_veh == critical_vehicle_id) {
                        // --- INTRA-ROUTE RELOCATION (Same Vehicle) ---
                        auto evaluate_intra = [&](int p2) {
                            vi new_route = base_route;
                            new_route.erase(new_route.begin() + p);
                            int insert_idx = p2 - (p2 > p ? 1 : 0);
                            new_route.insert(new_route.begin() + insert_idx, cust);
                            int first_changed_pos = min(p, insert_idx);
                            vi new_norm = normalize_route(new_route);
                            Solution candidate = initial_solution;
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = new_norm;
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = new_norm;
                            }
                            double score = score_context.score_candidate(
                                candidate,
                                {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, first_changed_pos}},
                                solution_cost);
                            
                            if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) return;
                            
                            if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                                best_neighbor_cost_local = score;
                                best_target = target_veh;
                                best_cust = cust;
                                best_candidate_neighbor = candidate;
                            }
                        };

                        for (int p2 = 1; p2 < (int)base_route.size(); ++p2) {
                            if (p2 == p) continue;
                            evaluate_intra(p2);
                        }
                        // End of route
                        evaluate_intra(base_route.size());
                    } else {
                        // --- INTER-ROUTE RELOCATION (Different Vehicle) ---
                        const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh] : initial_solution.drone_routes[target_veh - h];
                        auto evaluate_inter = [&](int insert_pos) {
                            vi new_target = target_route;
                            if (insert_pos >= (int)new_target.size()) {
                                new_target.push_back(cust);
                                new_target.push_back(0);
                            } else {
                                new_target.insert(new_target.begin() + insert_pos, cust);
                            }
                            Solution candidate = initial_solution;
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = base_route_removed;
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = base_route_removed;
                            }
                            if (target_veh < h) {
                                candidate.truck_routes[target_veh] = new_target;
                            } else {
                                candidate.drone_routes[target_veh - h] = new_target;
                            }
                            double score = score_context.score_candidate(
                                candidate,
                                {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, p},
                                 {target_veh < h, target_veh < h ? target_veh : target_veh - h, insert_pos}},
                                solution_cost);
                            
                            if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)){
                                //cout << "Tabu move skipped: cust " << cust << " to vehicle " << target_veh << " until iter " << tabu_list_10[cust][target_veh] << "\n";
                                return;
                            }
                            
                            if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                                best_neighbor_cost_local = score;
                                best_target = target_veh;
                                best_cust = cust;
                                best_candidate_neighbor = candidate;
                            }
                        };

                        for (int insert_pos = 1; insert_pos < (int)target_route.size(); ++insert_pos) {
                            evaluate_inter(insert_pos);
                        }
                        evaluate_inter(target_route.size());
                    }
                }
            }
        };
        for (int critical_vehicle : critical_vehicles) {
            if (critical_vehicle < h) {
                consider_relocate(initial_solution.truck_routes[critical_vehicle], true, critical_vehicle);
            } else {
                int drone_idx = critical_vehicle - h;
                consider_relocate(initial_solution.drone_routes[drone_idx], false, critical_vehicle);
            }
        }

        // After evaluating all candidates, update tabu list if we found an improving move
        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Update tabu list
            stage_tabu_attribute(TabuAttributeKind::Matrix10, {best_cust, best_target}, TABU_TENURE_10);
        }
        return best_neighbor;
    } else if (neighbor_id == 1) {
        // Neighborhood 1: swap two customers, allowing cross-mode exchanges
        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

        if ((int)tabu_list_11.size() != n + 1 || (n + 1 > 0 && (int)tabu_list_11[0].size() != n + 1)) {
            tabu_list_11.assign(n + 1, vector<int>(n + 1, 0));
        }

        int best_cust_a = -1, best_cust_b = -1;
        int best_pos_a = -1, best_pos_b = -1;
        int best_veh_a = -1, best_veh_b = -1;
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_neighbor_tie_count = 0;

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 2) return; // nothing to swap
            vector<int> crit_positions;
            for (int i = 0; i < (int)base_route.size(); ++i) {
                if (base_route[i] != 0) crit_positions.push_back(i);
            }
            if (crit_positions.empty()) return;

            for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                if (target_veh == critical_vehicle_id) {
                    for (int idx_a = 0; idx_a < (int)crit_positions.size(); ++idx_a) {
                        int pos_a = crit_positions[idx_a];
                        int cust_a = base_route[pos_a];

                        for (int idx_b = idx_a + 1; idx_b < (int)crit_positions.size(); ++idx_b) {
                            int pos_b = crit_positions[idx_b];
                            int cust_b = base_route[pos_b];

                            if (cust_a == cust_b) continue;

                            // Check tabu status
                            int u = min(cust_a, cust_b);
                            int v = max(cust_a, cust_b);
                            bool is_tabu = (tabu_list_11[u][v] > current_iter);

                            // Generate new route with swapped customers
                            vi new_crit_route = base_route;
                            new_crit_route[pos_a] = cust_b;
                            new_crit_route[pos_b] = cust_a;

                            Solution candidate = initial_solution;
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = new_crit_route;
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = new_crit_route;
                            }
                            double score = score_context.score_candidate(
                                candidate,
                                {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, min(pos_a, pos_b)}},
                                solution_cost);
                            if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) continue;
                            if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                                best_neighbor_cost_local = score;
                                best_candidate_neighbor = candidate;
                                best_cust_a = cust_a;
                                best_cust_b = cust_b;
                                best_pos_a = pos_a;
                                best_pos_b = pos_b;
                                best_veh_a = critical_vehicle_id;
                                best_veh_b = critical_vehicle_id;
                            }
                        }
                    }
                }

                if (target_veh == critical_vehicle_id) continue;

                const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh] : initial_solution.drone_routes[target_veh - h];
                if (target_route.size() <= 2) continue;

                vector<int> target_positions;
                for (int i = 0; i < (int)target_route.size(); ++i) {
                    if (target_route[i] != 0) target_positions.push_back(i);
                }
                if (target_positions.empty()) continue;

                for (int idx_a = 0; idx_a < (int)crit_positions.size(); ++idx_a) {
                    int pos_a = crit_positions[idx_a];
                    int cust_a = base_route[pos_a];

                    for (int idx_b = 0; idx_b < (int)target_positions.size(); ++idx_b) {
                        int pos_b = target_positions[idx_b];
                        int cust_b = target_route[pos_b];

                        if (cust_a == cust_b) continue;

                        if (served_by_drone[cust_a] == 0 && target_veh >= h) continue; // cannot assign cust_a to drone
                        if (served_by_drone[cust_b] == 0 && critical_vehicle_id >= h) continue; // cannot assign cust_b to drone

                        // Check tabu status
                        int u = min(cust_a, cust_b);
                        int v = max(cust_a, cust_b);
                        bool is_tabu = (tabu_list_11[u][v] > current_iter);

                        // Generate new routes with swapped customers
                        vi new_crit_route = base_route;
                        vi new_target_route = target_route;
                        new_crit_route[pos_a] = cust_b;
                        new_target_route[pos_b] = cust_a;

                        Solution candidate = initial_solution;
                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = new_crit_route;
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = new_crit_route;
                        }
                        if (target_veh < h) {
                            candidate.truck_routes[target_veh] = new_target_route;
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target_route;
                        }
                        double score = score_context.score_candidate(
                            candidate,
                            {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, pos_a},
                             {target_veh < h, target_veh < h ? target_veh : target_veh - h, pos_b}},
                            solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) continue;
                        if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                            best_neighbor_cost_local = score;
                            best_candidate_neighbor = candidate;
                            best_cust_a = cust_a;
                            best_cust_b = cust_b;
                            best_pos_a = pos_a;
                            best_pos_b = pos_b;
                            best_veh_a = critical_vehicle_id;
                            best_veh_b = target_veh;
                        }
                    }
                }
            }
        };

        for (int veh : critical_vehicles) {
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
            stage_tabu_attribute(TabuAttributeKind::Matrix11, {u, v}, TABU_TENURE_11);

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
        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

        // Prepare best tracking
        Solution best_candidate_neighbor = initial_solution;
        double best_neighbor_cost_local = 1e10;
        int best_neighbor_tie_count = 0;
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
                    double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                    if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;

                    if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
                    if ((served_by_drone[c1] == 0 || served_by_drone[c2] == 0) && target_veh >= h) continue;
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
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;

                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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

        for (int critical_vehicle : critical_vehicles) {
            if (critical_vehicle < h) {
                consider_relocate_pair(initial_solution.truck_routes[critical_vehicle], true, critical_vehicle);
            } else {
                int drone_idx = critical_vehicle - h;
                consider_relocate_pair(initial_solution.drone_routes[drone_idx], false, critical_vehicle);
            }
        }

        // apply best move if found
        if (best_c1 != -1 && best_c2 != -1 && best_target_vehicle != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            // update tabu
            vector<int> key = { min(best_c1, best_c2), max(best_c1, best_c2), best_target_vehicle };
            stage_tabu_attribute(TabuAttributeKind::Map20, key, TABU_TENURE_20);
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

        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_neighbor_tie_count = 0;
        int best_edge_u = -1, best_edge_v = -1;
        int best_i = -1, best_j = -1;
        bool best_is_truck = true;

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
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;

                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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

        for (int critical_vehicle : critical_vehicles) {
            if (critical_vehicle < h) {
                consider_2opt(initial_solution.truck_routes[critical_vehicle], true, critical_vehicle);
            } else {
                int drone_idx = critical_vehicle - h;
                consider_2opt(initial_solution.drone_routes[drone_idx], false, critical_vehicle);
            }
        }

        if (best_edge_u != -1 && best_edge_v != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            stage_tabu_attribute(TabuAttributeKind::Matrix2Opt, {best_edge_u, best_edge_v}, TABU_TENURE_2OPT);

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

        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

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
        int best_neighbor_tie_count = 0;
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

        auto evaluate_two_opt_star = [&](int crit_vehicle, int other_vehicle) {
            bool crit_is_truck = crit_vehicle < h;
            bool other_is_truck = other_vehicle < h;
            int crit_idx = crit_is_truck ? crit_vehicle : crit_vehicle - h;
            int other_idx = other_is_truck ? other_vehicle : other_vehicle - h;
            const vi& crit_route_raw = crit_is_truck
                ? initial_solution.truck_routes[crit_idx]
                : initial_solution.drone_routes[crit_idx];
            const vi& other_route_raw = other_is_truck
                ? initial_solution.truck_routes[other_idx]
                : initial_solution.drone_routes[other_idx];
            vi crit_route = normalize_route(crit_route_raw);
            if (crit_route.size() <= 3) return;
            vd crit_metrics = check_route_feasibility(crit_route_raw, 0.0, crit_is_truck);
            auto crit_segs = enumerate_segments(crit_route);
            vi other_route = normalize_route(other_route_raw);
            if (other_route.size() <= 3) return;
            vd other_metrics = check_route_feasibility(other_route_raw, 0.0, other_is_truck);
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
                                int crit_drone_idx = crit_idx;
                                if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[crit_drone_idx] = crit_new;
                                    candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                                }
                            }

                            if (other_is_truck) {
                                candidate.truck_routes[other_idx] = other_new;
                                candidate.truck_route_times[other_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                            } else {
                                int other_drone_idx = other_idx;
                                if (other_drone_idx >= 0 && other_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[other_drone_idx] = other_new;
                                    candidate.drone_route_times[other_drone_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                                }
                            }

                            candidate.total_makespan = 0.0;
                            for (int i = 0; i < h; ++i) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                            for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                            double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                            if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                                continue;
                            }

                            if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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

        for (int crit_vehicle : critical_vehicles) {
            for (int other_vehicle = 0; other_vehicle < h + d; ++other_vehicle) {
                if (other_vehicle == crit_vehicle) continue;
                evaluate_two_opt_star(crit_vehicle, other_vehicle);
            }
        }

        if (best_ua != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            stage_tabu_attribute(TabuAttributeKind::Matrix2OptStar, {best_ua, best_va}, TABU_TENURE_2OPT_STAR);
            stage_tabu_attribute(TabuAttributeKind::Matrix2OptStar, {best_ub, best_vb}, TABU_TENURE_2OPT_STAR);

            //Debug N4
/*              cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N4] 2-opt* cuts (" << best_ua << "," << best_va << ") & (" << best_ub << "," << best_vb << ")"
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 5) {
       vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_neighbor_tie_count = 0;
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

        auto consider_pair_vs_single = [&](const vi& crit_route, bool crit_mode_truck, int crit_global_idx, int crit_route_idx) {
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
                    if (target_veh == crit_global_idx) continue;
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
                            int crit_drone_idx = crit_route_idx;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
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

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(c1, c2), max(c1, c2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                            continue;
                        }
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
                            int crit_drone_idx = crit_route_idx;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
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

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(b1, b2), max(b1, b2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                            continue;
                        }
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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

        for (int critical_vehicle : critical_vehicles) {
            if (critical_vehicle < h) {
                consider_pair_vs_single(initial_solution.truck_routes[critical_vehicle], true, critical_vehicle, critical_vehicle);
                consider_single_vs_pair(initial_solution.truck_routes[critical_vehicle], true, critical_vehicle);
            } else {
                int drone_idx = critical_vehicle - h;
                consider_pair_vs_single(initial_solution.drone_routes[drone_idx], false, critical_vehicle, drone_idx);
                consider_single_vs_pair(initial_solution.drone_routes[drone_idx], false, drone_idx);
            }
        }

        if (!best_tabu_triple.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            stage_tabu_attribute(TabuAttributeKind::Map21, best_tabu_triple, TABU_TENURE_21);

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
        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_neighbor_tie_count = 0;
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

                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                            continue;
                        }
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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

        for (int critical_vehicle : critical_vehicles) {
            if (critical_vehicle < h) {
                consider_swap_pairs(initial_solution.truck_routes[critical_vehicle], true, critical_vehicle);
            } else {
                int drone_idx = critical_vehicle - h;
                consider_swap_pairs(initial_solution.drone_routes[drone_idx], false, drone_idx);
            }
        }

        if (!best_tabu_key.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            stage_tabu_attribute(TabuAttributeKind::Map22, best_tabu_key, TABU_TENURE_22);

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
        int best_neighbor_tie_count = 0;
        vector<int> best_tabu_key;

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

                                double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                                if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                                    continue;
                                }
                                if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
	                                    best_neighbor_cost_local = candidate_score;
	                                    best_candidate_neighbor = candidate;
	                                    best_tabu_key = tabu_key;
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
	            stage_tabu_attribute(TabuAttributeKind::Ejection, best_tabu_key, TABU_TENURE_EJECTION);

	            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 8) {
        // Neighborhood 8: merge any two trips of the same vehicle and test orders/orientations.
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_neighbor_tie_count = 0;
        vector<int> best_fusion_tabu_key;

        for (int veh = 0; veh < h + d; ++veh) {
            bool is_truck = veh < h;
            const vi& route = is_truck ? initial_solution.truck_routes[veh] 
                                       : initial_solution.drone_routes[veh - h];
            if (route.size() < 3) continue;

            vector<vector<int>> trips;
            vector<int> current_trip;
            for (size_t i = 1; i < route.size(); ++i) {
                if (route[i] == 0) {
                    if (!current_trip.empty()) {
                        trips.push_back(current_trip);
                        current_trip.clear();
                    }
                } else {
                    current_trip.push_back(route[i]);
                }
            }
            if (!current_trip.empty()) trips.push_back(current_trip);

            if (trips.size() < 2) continue;

            for (int i = 0; i < (int)trips.size(); ++i) {
                for (int j = i + 1; j < (int)trips.size(); ++j) {
                    vector<int> fusion_tabu_key = trips[i];
                    fusion_tabu_key.insert(fusion_tabu_key.end(), trips[j].begin(), trips[j].end());
                    sort(fusion_tabu_key.begin(), fusion_tabu_key.end());
                    bool is_tabu = (tabu_list_fusion.count(fusion_tabu_key) &&
                                    tabu_list_fusion[fusion_tabu_key] > current_iter);

                    for (int opt = 0; opt < 8; ++opt) {
                        vi merged_trip;
                        vi ti = trips[i];
                        vi tj = trips[j];

                        bool i_first = (opt < 4);
                        int type = opt % 4;

                        vi* first = i_first ? &ti : &tj;
                        vi* second = i_first ? &tj : &ti;

                        bool rev_first = (type == 2 || type == 3);
                        bool rev_second = (type == 1 || type == 3);

                        if (rev_first) reverse(first->begin(), first->end());
                        if (rev_second) reverse(second->begin(), second->end());

                        merged_trip.insert(merged_trip.end(), first->begin(), first->end());
                        merged_trip.insert(merged_trip.end(), second->begin(), second->end());

                        vi new_route;
                        new_route.push_back(0);
                        for (int k = 0; k < (int)trips.size(); ++k) {
                            if (k == j) continue;
                            if (k == i) {
                                new_route.insert(new_route.end(), merged_trip.begin(), merged_trip.end());
                            } else {
                                new_route.insert(new_route.end(), trips[k].begin(), trips[k].end());
                            }
                            new_route.push_back(0);
                        }

                        vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck);

                        vd old_metrics = check_route_feasibility(route, 0.0, is_truck);
                        Solution candidate = initial_solution;
                        candidate.deadline_violation += new_metrics[1] - old_metrics[1];
                        candidate.capacity_violation += new_metrics[3] - old_metrics[3];
                        candidate.energy_violation += new_metrics[2] - old_metrics[2];

                        if (is_truck) {
                            candidate.truck_routes[veh] = new_route;
                            candidate.truck_route_times[veh] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        } else {
                            candidate.drone_routes[veh - h] = new_route;
                            candidate.drone_route_times[veh - h] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_fusion_tabu_key = fusion_tabu_key;
                        }
                    }
                }
            }
        }

        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            if (!best_fusion_tabu_key.empty()) {
                stage_tabu_attribute(TabuAttributeKind::Fusion, best_fusion_tabu_key, TABU_TENURE_FUSION);
            }
            return best_candidate_neighbor;
        }
        return initial_solution;
    }
// ...existing code...
return initial_solution;
}

Solution local_search_all_vehicle(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost, double (*solution_cost)(const Solution&)) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 1e10;
    SolutionEvaluationContext score_context(initial_solution);
    clear_pending_tabu_attributes();
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
        int best_neighbor_tie_count = 0;

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {

            // Lambda to normalize routes for comparison (detect no-ops)
            auto normalize_route = []( vi& route) -> vi {
                if (!route.empty() && route.front() != 0) 
                    route.insert(route.begin(), 0);
                if (!route.empty() && route.back() != 0) 
                    route.push_back(0);
                return route;
            };

            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust = base_route[p];

                // Pre-calculate the base route with customer removed (for inter-route moves)
                vi base_route_removed = base_route;
                base_route_removed.erase(base_route_removed.begin() + p);
                
                // Try relocating cust to other vehicles
                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (served_by_drone[cust] == 0 && target_veh >= h) continue; // cannot assign to drone
                    bool is_tabu = (tabu_list_10[cust][target_veh] > current_iter);
                    
                    if (target_veh == critical_vehicle_id) {
                        // --- INTRA-ROUTE RELOCATION (Same Vehicle) ---
                        auto evaluate_intra = [&](int p2) {
                            vi new_route = base_route;
                            new_route.erase(new_route.begin() + p);
                            int insert_idx = p2 - (p2 > p ? 1 : 0);
                            new_route.insert(new_route.begin() + insert_idx, cust);
                            int first_changed_pos = min(p, insert_idx);
                            vi new_norm = normalize_route(new_route);

                            Solution candidate = initial_solution;
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = new_norm;
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = new_norm;
                            }
                            double score = score_context.score_candidate(
                                candidate,
                                {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, first_changed_pos}},
                                solution_cost);
                            
                            if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) return;
                            
                            if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                                best_neighbor_cost_local = score;
                                best_target = target_veh;
                                best_cust = cust;
                                best_candidate_neighbor = candidate;
                            }
                        };

                        for (int p2 = 1; p2 < (int)base_route.size(); ++p2) {
                            if (p2 == p) continue;
                            evaluate_intra(p2);
                        }
                        // End of route
                        evaluate_intra(base_route.size());
                    } else {
                        // --- INTER-ROUTE RELOCATION (Different Vehicle) ---
                        const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh] : initial_solution.drone_routes[target_veh - h];
                        auto evaluate_inter = [&](int insert_pos) {
                            vi new_target = target_route;
                            if (insert_pos >= (int)new_target.size()) {
                                new_target.push_back(cust);
                                new_target.push_back(0);
                            } else {
                                new_target.insert(new_target.begin() + insert_pos, cust);
                            }
                            Solution candidate = initial_solution;
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = base_route_removed;
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = base_route_removed;
                            }
                            if (target_veh < h) {
                                candidate.truck_routes[target_veh] = new_target;
                            } else {
                                candidate.drone_routes[target_veh - h] = new_target;
                            }
                            double score = score_context.score_candidate(
                                candidate,
                                {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, p},
                                 {target_veh < h, target_veh < h ? target_veh : target_veh - h, insert_pos}},
                                solution_cost);
                            
                            if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) return;
                            
                            if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                                best_neighbor_cost_local = score;
                                best_target = target_veh;
                                best_cust = cust;
                                best_candidate_neighbor = candidate;
                            }
                        };

                        for (int insert_pos = 1; insert_pos < (int)target_route.size(); ++insert_pos) {
                            evaluate_inter(insert_pos);
                        }
                        evaluate_inter(target_route.size());
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
            stage_tabu_attribute(TabuAttributeKind::Matrix10, {best_cust, best_target}, TABU_TENURE_10);
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
        int best_pos_a = -1, best_pos_b = -1;
        int best_veh_a = -1, best_veh_b = -1;
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_neighbor_tie_count = 0;

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 2) return; // nothing to swap

            vector<int> crit_positions;
            for (int i = 0; i < (int)base_route.size(); ++i) {
                if (base_route[i] != 0) crit_positions.push_back(i);
            }
            if (crit_positions.empty()) return;

            for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                if (target_veh == critical_vehicle_id) {
                    for (int idx_a = 0; idx_a < (int)crit_positions.size(); ++idx_a) {
                        int pos_a = crit_positions[idx_a];
                        int cust_a = base_route[pos_a];

                        for (int idx_b = idx_a + 1; idx_b < (int)crit_positions.size(); ++idx_b) {
                            int pos_b = crit_positions[idx_b];
                            int cust_b = base_route[pos_b];

                            if (cust_a == cust_b) continue;

                            // Check tabu status
                            int u = min(cust_a, cust_b);
                            int v = max(cust_a, cust_b);
                            bool is_tabu = (tabu_list_11[u][v] > current_iter);

                            // Generate new route with swapped customers
                            vi new_crit_route = base_route;
                            new_crit_route[pos_a] = cust_b;
                            new_crit_route[pos_b] = cust_a;

                            Solution candidate = initial_solution;
                            if (is_truck_mode) {
                                candidate.truck_routes[critical_vehicle_id] = new_crit_route;
                            } else {
                                candidate.drone_routes[critical_vehicle_id - h] = new_crit_route;
                            }
                            double score = score_context.score_candidate(
                                candidate,
                                {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, min(pos_a, pos_b)}},
                                solution_cost);
                            if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) continue;
                            if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                                best_neighbor_cost_local = score;
                                best_candidate_neighbor = candidate;
                                best_cust_a = cust_a;
                                best_cust_b = cust_b;
                                best_pos_a = pos_a;
                                best_pos_b = pos_b;
                                best_veh_a = critical_vehicle_id;
                                best_veh_b = critical_vehicle_id;
                            }
                        }
                    }
                }

                if (target_veh == critical_vehicle_id) continue;

                const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh] : initial_solution.drone_routes[target_veh - h];
                if (target_route.size() <= 2) continue;

                vector<int> target_positions;
                for (int i = 0; i < (int)target_route.size(); ++i) {
                    if (target_route[i] != 0) target_positions.push_back(i);
                }
                if (target_positions.empty()) continue;

                for (int idx_a = 0; idx_a < (int)crit_positions.size(); ++idx_a) {
                    int pos_a = crit_positions[idx_a];
                    int cust_a = base_route[pos_a];

                    for (int idx_b = 0; idx_b < (int)target_positions.size(); ++idx_b) {
                        int pos_b = target_positions[idx_b];
                        int cust_b = target_route[pos_b];

                        if (cust_a == cust_b) continue;

                        if (served_by_drone[cust_a] == 0 && target_veh >= h) continue; // cannot assign cust_a to drone
                        if (served_by_drone[cust_b] == 0 && critical_vehicle_id >= h) continue; // cannot assign cust_b to drone

                        // Check tabu status
                        int u = min(cust_a, cust_b);
                        int v = max(cust_a, cust_b);
                        bool is_tabu = (tabu_list_11[u][v] > current_iter);

                        // Generate new routes with swapped customers
                        vi new_crit_route = base_route;
                        vi new_target_route = target_route;
                        new_crit_route[pos_a] = cust_b;
                        new_target_route[pos_b] = cust_a;

                        Solution candidate = initial_solution;
                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = new_crit_route;
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = new_crit_route;
                        }
                        if (target_veh < h) {
                            candidate.truck_routes[target_veh] = new_target_route;
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target_route;
                        }
                        double score = score_context.score_candidate(
                            candidate,
                            {{is_truck_mode, is_truck_mode ? critical_vehicle_id : critical_vehicle_id - h, pos_a},
                             {target_veh < h, target_veh < h ? target_veh : target_veh - h, pos_b}},
                            solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, score, best_cost)) continue;
                        if (better_score_with_random_tie(score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                            best_neighbor_cost_local = score;
                            best_candidate_neighbor = candidate;
                            best_cust_a = cust_a;
                            best_cust_b = cust_b;
                            best_pos_a = pos_a;
                            best_pos_b = pos_b;
                            best_veh_a = critical_vehicle_id;
                            best_veh_b = target_veh;
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
            stage_tabu_attribute(TabuAttributeKind::Matrix11, {u, v}, TABU_TENURE_11);

            // Debug: print swap info
             /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N1] swap " << best_cust_a << " and " << best_cust_b
                 << ", score: " << solution_score_workload(initial_solution)
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
        int best_neighbor_tie_count = 0;
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
                    double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                    if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;

                    if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
                    if (!target_is_truck && (served_by_drone[c1] == 0 || served_by_drone[c2] == 0)) continue;
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
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;

                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
            stage_tabu_attribute(TabuAttributeKind::Map20, key, TABU_TENURE_20);
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
        int best_neighbor_tie_count = 0;
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
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;

                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
            stage_tabu_attribute(TabuAttributeKind::Matrix2Opt, {best_edge_u, best_edge_v}, TABU_TENURE_2OPT);

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
        int best_neighbor_tie_count = 0;
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
                            double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                            if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                                continue;
                            }

                            if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
            stage_tabu_attribute(TabuAttributeKind::Matrix2OptStar, {best_ua, best_va}, TABU_TENURE_2OPT_STAR);
            stage_tabu_attribute(TabuAttributeKind::Matrix2OptStar, {best_ub, best_vb}, TABU_TENURE_2OPT_STAR);

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
        int best_neighbor_tie_count = 0;
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

        auto consider_pair_vs_single = [&](const vi& crit_route, bool crit_mode_truck, int crit_global_idx, int crit_route_idx) {
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
                    bool target_is_truck = target_veh < h;
                    if (target_veh == crit_global_idx) continue;
                    
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
                            int crit_drone_idx = crit_route_idx;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
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

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(c1, c2), max(c1, c2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                            continue;
                        }
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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

        auto consider_single_vs_pair = [&](const vi& crit_route, bool crit_mode_truck, int crit_global_idx, int crit_route_idx) {
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
                    if (target_veh == crit_global_idx) continue;
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
                            int crit_drone_idx = crit_route_idx;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                        }
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

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        vector<int> key = { min(b1, b2), max(b1, b2), single };
                        auto it = tabu_list_21.find(key);
                        bool is_tabu = (it != tabu_list_21.end() && it->second > current_iter);
                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                            continue;
                        }
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
            int route_idx = crit_is_truck ? critical_idx : critical_idx - h;
            consider_pair_vs_single(route, crit_is_truck, critical_idx, route_idx);
            consider_single_vs_pair(route, crit_is_truck, critical_idx, route_idx);
        }

        if (!best_tabu_triple.empty() && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            stage_tabu_attribute(TabuAttributeKind::Map21, best_tabu_triple, TABU_TENURE_21);

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
        int best_neighbor_tie_count = 0;
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
                        if (!same_route) {
                            candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                            candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                            candidate.energy_violation += target_new_metrics[2] - target_metrics[2];
                        }

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

                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                            continue;
                        }
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
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
            stage_tabu_attribute(TabuAttributeKind::Map22, best_tabu_key, TABU_TENURE_22);

            // Debug N6
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            bool other_is_truck = best_other_is_truck;
            int other_idx = other_is_truck ? best_other_vehicle : best_other_vehicle - h;
            cout << "[N6] (" << (best_same_route ? "same route" : "different routes") << ") swap pairs ("
                 << best_pair_a1 << "," << best_pair_a2 << ") & ("
                 << best_pair_b1 << "," << best_pair_b2 << ") "
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    }  else if (neighbor_id == 7) {
        // Neighborhood 7: depth-2 ejection chain (i -> j -> k)
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_neighbor_tie_count = 0;
        vector<int> best_tabu_key;

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

                                double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                                if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) {
                                    continue;
                                }
                                if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
	                                    best_neighbor_cost_local = candidate_score;
	                                    best_candidate_neighbor = candidate;
	                                    best_tabu_key = tabu_key;
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
	            stage_tabu_attribute(TabuAttributeKind::Ejection, best_tabu_key, TABU_TENURE_EJECTION);

	            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 8) {
        // Neighborhood 8: Merge Any Two Trips (Trip Fusion)
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_neighbor_tie_count = 0;
        vector<int> best_fusion_tabu_key;

        for (int veh = 0; veh < h + d; ++veh) {
            bool is_truck = veh < h;
            const vi& route = is_truck ? initial_solution.truck_routes[veh] 
                                       : initial_solution.drone_routes[veh - h];
            if (route.size() < 3) continue;

            vector<vector<int>> trips;
            vector<int> current_trip;
            for (size_t i = 1; i < route.size(); ++i) {
                if (route[i] == 0) {
                    if (!current_trip.empty()) {
                        trips.push_back(current_trip);
                        current_trip.clear();
                    }
                } else {
                    current_trip.push_back(route[i]);
                }
            }
            if (!current_trip.empty()) trips.push_back(current_trip);

            if (trips.size() < 2) continue;

            for (int i = 0; i < (int)trips.size(); ++i) {
                for (int j = i + 1; j < (int)trips.size(); ++j) {
                    vector<int> fusion_tabu_key = trips[i];
                    fusion_tabu_key.insert(fusion_tabu_key.end(), trips[j].begin(), trips[j].end());
                    sort(fusion_tabu_key.begin(), fusion_tabu_key.end());
                    bool is_tabu = (tabu_list_fusion.count(fusion_tabu_key) &&
                                    tabu_list_fusion[fusion_tabu_key] > current_iter);

                    for (int opt = 0; opt < 8; ++opt) {
                        vi merged_trip;
                        vi ti = trips[i];
                        vi tj = trips[j];

                        bool i_first = (opt < 4);
                        int type = opt % 4; 
                        
                        vi* first = i_first ? &ti : &tj;
                        vi* second = i_first ? &tj : &ti;
                        
                        bool rev_first = (type == 2 || type == 3);
                        bool rev_second = (type == 1 || type == 3);

                        if (rev_first) reverse(first->begin(), first->end());
                        if (rev_second) reverse(second->begin(), second->end());

                        merged_trip.insert(merged_trip.end(), first->begin(), first->end());
                        merged_trip.insert(merged_trip.end(), second->begin(), second->end());

                        vi new_route;
                        new_route.push_back(0);
                        for (int k = 0; k < (int)trips.size(); ++k) {
                            if (k == j) continue; 
                            if (k == i) {
                                new_route.insert(new_route.end(), merged_trip.begin(), merged_trip.end());
                            } else {
                                new_route.insert(new_route.end(), trips[k].begin(), trips[k].end());
                            }
                            new_route.push_back(0);
                        }

                        vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck);

                        vd old_metrics = check_route_feasibility(route, 0.0, is_truck);
                        Solution candidate = initial_solution;
                        candidate.deadline_violation += new_metrics[1] - old_metrics[1];
                        candidate.capacity_violation += new_metrics[3] - old_metrics[3];
                        candidate.energy_violation += new_metrics[2] - old_metrics[2];

                        if (is_truck) {
                            candidate.truck_routes[veh] = new_route;
                            candidate.truck_route_times[veh] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        } else {
                            candidate.drone_routes[veh - h] = new_route;
                            candidate.drone_route_times[veh - h] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);

                        double candidate_score = score_recalculated_candidate(candidate, solution_cost);
                        if (!admissible_by_tabu_or_aspiration(is_tabu, candidate_score, best_cost)) continue;
                        
                        if (better_score_with_random_tie(candidate_score, best_neighbor_cost_local, best_neighbor_tie_count)) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_fusion_tabu_key = fusion_tabu_key;
                        }
                    }
                }
            }
        }

        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            if (!best_fusion_tabu_key.empty()) {
                stage_tabu_attribute(TabuAttributeKind::Fusion, best_fusion_tabu_key, TABU_TENURE_FUSION);
            }
            return best_candidate_neighbor;
        }
        return initial_solution;
    }
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
    for (int i = 0; i < h; ++i) {
        vd metrics = check_route_feasibility(sol.truck_routes[i], 0.0, true);
        sol.truck_route_times[i] = metrics[0];
    }
    for (int i = 0; i < d; ++i) {
        vd metrics = check_route_feasibility(sol.drone_routes[i], 0.0, false);
        sol.drone_route_times[i] = metrics[0];
    }

    sol.total_makespan = 0.0;
    for (int t = 0; t < h; ++t) sol.total_makespan = max(sol.total_makespan, sol.truck_route_times[t]);
    for (int t = 0; t < d; ++t) sol.total_makespan = max(sol.total_makespan, sol.drone_route_times[t]);

    NormalizedViolations violations = compute_normalized_violations(sol);
    sol.capacity_violation = violations.capacity;
    sol.energy_violation = violations.energy;
    sol.deadline_violation = violations.deadline;
    return sol;
}

bool check_solution_integrity(const Solution& sol) {
    int served_count = 0;
    vector<bool> served(n + 1, false);
    for (int i = 0; i < h; ++i) {
        const vi& route = sol.truck_routes[i];
        for (size_t j = 0; j < route.size(); ++j) {
            int customer = route[j];
            if (customer != 0 && served[customer]){
                return false;
            }
            if (customer != 0 && !served[customer]) {
                served[customer] = true;
                served_count++;
            }
        }
    }
    for (int i = 0; i < d; ++i) {
        const vi& route = sol.drone_routes[i];
        for (size_t j = 0; j < route.size(); ++j) {
            int customer = route[j];
            if (customer != 0 && served[customer]){
                return false;
            }
            if (customer != 0 && !served[customer]) {
                served[customer] = true;
                served_count++;
            }
        }
    }
    return (served_count == n);
}

Solution updated_elite_set(const Solution& sol) {
    bool is_feasible = is_feasible_solution(sol);
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

static vi normalized_route_copy(vi route) {
    if (route.empty()) route.push_back(0);
    if (route.front() != 0) route.insert(route.begin(), 0);
    if (route.back() != 0) route.push_back(0);
    vi cleaned;
    cleaned.reserve(route.size());
    for (int node : route) {
        if (!cleaned.empty() && cleaned.back() == 0 && node == 0) continue;
        cleaned.push_back(node);
    }
    return cleaned;
}

static vi route_with_inserted_customer(const vi& route, int cust, int pos) {
    vi result = normalized_route_copy(route);
    if (result.size() == 1) {
        result.push_back(cust);
        result.push_back(0);
        return result;
    }
    int insert_pos = max(1, min(pos, (int)result.size() - 1));
    result.insert(result.begin() + insert_pos, cust);
    return normalized_route_copy(result);
}

static vi route_with_removed_customer(const vi& route, int cust) {
    vi result = route;
    result.erase(remove(result.begin(), result.end(), cust), result.end());
    return normalized_route_copy(result);
}

Solution repair_solution_common(Solution sol, const unordered_set<int>& to_destroy) {
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    Solution new_sol = sol;
    
    for (int i = 0; i < h; ++i) {
        vi& route = new_sol.truck_routes[i];
        route.erase(remove_if(route.begin(), route.end(), [&](int c) {
            return to_destroy.count(c) > 0;
        }), route.end());
        route = normalized_route_copy(route);
    }
    for (int i = 0; i < d; ++i) {
        vi& route = new_sol.drone_routes[i];
        route.erase(remove_if(route.begin(), route.end(), [&](int c) {
            return to_destroy.count(c) > 0;
        }), route.end());
        route = normalized_route_copy(route);
    }
    new_sol = recalculate_solution(new_sol);
    
    vector<int> customers_to_insert(to_destroy.begin(), to_destroy.end());
    std::shuffle(customers_to_insert.begin(), customers_to_insert.end(), rng);
    
    for (int cust : customers_to_insert) {
        Solution best_candidate = new_sol;
        double best_score = numeric_limits<double>::infinity();
        int tie_count = 0;

        auto consider_insertion = [&](bool is_truck, int veh_idx, int pos) {
            Solution candidate = new_sol;
            if (is_truck) {
                candidate.truck_routes[veh_idx] = route_with_inserted_customer(new_sol.truck_routes[veh_idx], cust, pos);
            } else {
                candidate.drone_routes[veh_idx] = route_with_inserted_customer(new_sol.drone_routes[veh_idx], cust, pos);
            }
            // Greedy repair score f_mk: penalized makespan after inserting this customer.
            double score = score_recalculated_candidate(candidate, solution_score_makespan);
            bool take_candidate = false;
            if (score + 1e-8 < best_score) {
                tie_count = 1;
                take_candidate = true;
            } else if (fabs(score - best_score) <= 1e-8) {
                ++tie_count;
                std::uniform_int_distribution<int> tie_dist(1, tie_count);
                take_candidate = (tie_dist(rng) == 1);
            }
            if (take_candidate) {
                best_score = score;
                best_candidate = candidate;
            }
        };

        for (int veh = 0; veh < h; ++veh) {
            vi route = normalized_route_copy(new_sol.truck_routes[veh]);
            int last_pos = max(1, (int)route.size() - 1);
            for (int pos = 1; pos <= last_pos; ++pos) {
                consider_insertion(true, veh, pos);
            }
        }

        if (served_by_drone[cust]) {
            for (int veh = 0; veh < d; ++veh) {
                vi route = normalized_route_copy(new_sol.drone_routes[veh]);
                int last_pos = max(1, (int)route.size() - 1);
                for (int pos = 1; pos <= last_pos; ++pos) {
                    consider_insertion(false, veh, pos);
                }
            }
        }

        new_sol = best_candidate;
    }
    
    return recalculate_solution(new_sol);
}

Solution destroy_worst_repair_random(Solution sol) {
    unordered_set<int> to_destroy;
    int destroy_count = max(1, (int)floor(DESTROY_RATE * n));
    Solution current_sol = recalculate_solution(sol);
    
    for (int removed = 0; removed < destroy_count; ++removed) {
        vector<int> critical_vehicles = critical_vehicle_indices_by_makespan(current_sol);
        double current_score = solution_score_makespan(current_sol);
        Solution best_candidate = current_sol;
        int best_cust = -1;
        double best_reduction = -numeric_limits<double>::infinity();
        int tie_count = 0;

        for (int veh : critical_vehicles) {
            bool is_truck = veh < h;
            int veh_idx = is_truck ? veh : veh - h;
            const vi& route = is_truck ? current_sol.truck_routes[veh_idx]
                                       : current_sol.drone_routes[veh_idx];
            for (int pos = 0; pos < (int)route.size(); ++pos) {
                int cust = route[pos];
                if (cust == 0 || to_destroy.count(cust)) continue;

                Solution candidate = current_sol;
                if (is_truck) {
                    candidate.truck_routes[veh_idx] = route_with_removed_customer(route, cust);
                } else {
                    candidate.drone_routes[veh_idx] = route_with_removed_customer(route, cust);
                }
                double candidate_score = score_recalculated_candidate(candidate, solution_score_makespan);
                double reduction = current_score - candidate_score;
                if (reduction > best_reduction + 1e-8 ||
                    (fabs(reduction - best_reduction) <= 1e-8 && (++tie_count, rand() % tie_count == 0))) {
                    if (reduction > best_reduction + 1e-8) tie_count = 1;
                    best_reduction = reduction;
                    best_candidate = candidate;
                    best_cust = cust;
                }
            }
        }

        if (best_cust == -1) break;
        to_destroy.insert(best_cust);
        current_sol = best_candidate;
    }

    return repair_solution_common(current_sol, to_destroy);
}

Solution destroy_random_repair_random(Solution sol) {
    unordered_set<int> to_destroy;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    int destroy_count = static_cast<int>(n * 0.3); // Destroy 10%
    std::uniform_int_distribution<int> dist(1, n);
    while ((int)to_destroy.size() < destroy_count) {
        int r = dist(rng);
        to_destroy.insert(r);
    }
    
    return repair_solution_common(sol, to_destroy);
}

// SISR (Slack Induction by Substring Removal) Implementation
Solution destroy_sisr_repair(Solution sol) {
    const double DESTROY_RATE = 0.3;
    const int MAX_STRING_SIZE_BASE = 12; 
    const int destroy_target = max(1, (int)(n * DESTROY_RATE));
    
    unordered_set<int> to_destroy;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    
    // 1. Calculate Average Route Size (Trucks only)
    double total_len = 0;
    int truck_routes_active = 0;
    for(const auto& r : sol.truck_routes) {
        if(r.size() > 2) { 
            total_len += (r.size() - 2); 
            truck_routes_active++;
        }
    }
    int avg_route_size = (truck_routes_active > 0) ? (int)(total_len / truck_routes_active) : 5;
    int max_string_size = max(MAX_STRING_SIZE_BASE, avg_route_size);
    
    // 2. Pick Center
    std::uniform_int_distribution<int> dist_n(1, n);
    int center = dist_n(rng);
    
    // 3. Map customers to vehicles for fast lookups
    struct Locator { bool is_truck; int v_idx; int pos; };
    vector<Locator> cust_loc(n+1, {false, -1, -1});
    for(int i=0; i<h; ++i) {
        for(int p=0; p<(int)sol.truck_routes[i].size(); ++p) {
            int c = sol.truck_routes[i][p];
            if(c!=0) cust_loc[c] = {true, i, p};
        }
    }
    for(int i=0; i<d; ++i) {
        if (sol.drone_routes[i].empty()) continue;
        for(size_t p=0; p<sol.drone_routes[i].size(); ++p) {
             int c = sol.drone_routes[i][p];
             if(c!=0) cust_loc[c] = {false, i, (int)p};
        }
    }

    // 4. Neighbors loop
    vector<int> candidate_neighbors;
    if (center <= n && !KNN_LIST[center].empty()) {
        candidate_neighbors = KNN_LIST[center];
    } else {
        // Fallback if KNN empty 
        for(int i=1; i<=n; ++i) if(i!=center) candidate_neighbors.push_back(i);
        std::shuffle(candidate_neighbors.begin(), candidate_neighbors.end(), rng);
    }
    
    unordered_set<int> destroyed_routes_id;

    // Prioritize center, then neighbors
    vector<int> process_queue;
    process_queue.push_back(center);
    process_queue.insert(process_queue.end(), candidate_neighbors.begin(), candidate_neighbors.end());

    for(int neighbor_cust : process_queue) {
        if ((int)to_destroy.size() >= destroy_target) break;
        if(to_destroy.count(neighbor_cust)) continue; // Already marked
        
        Locator l = cust_loc[neighbor_cust];
        if(l.v_idx == -1) continue; 
        
        // Identify unique vehicle ID (Trucks: 0..h-1, Drones: h..h+d-1)
        int unique_id = l.is_truck ? l.v_idx : (h + l.v_idx);
        if(destroyed_routes_id.count(unique_id)) continue;
        
        destroyed_routes_id.insert(unique_id);
        
        // Apply the same substring-removal logic for both trucks and drones.
        const vi& route = l.is_truck ? sol.truck_routes[l.v_idx] : sol.drone_routes[l.v_idx];
        // Limit string size
        int actual_max = min((int)route.size()-2, max_string_size);
        if (actual_max < 1) actual_max = 1;

        std::uniform_int_distribution<int> size_dist(1, actual_max);
        int str_len = size_dist(rng);

        // We need a window [s, s+len-1] that contains l.pos
        // Constraints:
        // 1. s >= 1 (start after depot)
        // 2. s + str_len - 1 <= route.size() - 2 (end before depot)
        // 3. s <= l.pos
        // 4. s + str_len - 1 >= l.pos => s >= l.pos - str_len + 1
        int min_s = max(1, l.pos - str_len + 1);
        int max_s = min(l.pos, (int)route.size() - 1 - str_len); // Ensures end doesn't exceed bounds

        if (min_s > max_s) {
            // Fallback: just remove the neighbor if math fails
            if ((int)to_destroy.size() < destroy_target) to_destroy.insert(neighbor_cust);
        } else {
            std::uniform_int_distribution<int> start_dist(min_s, max_s);
            int s = start_dist(rng);
            for(int k=0; k<str_len; ++k) {
                if ((int)to_destroy.size() >= destroy_target) break;
                int idx = s + k;
                if (idx < (int)route.size()) {
                    int c = route[idx];
                    if(c!=0) to_destroy.insert(c);
                }
            }
        }
    }

    // Ensure fixed destroy rate if SISR candidate selection did not reach the target.
    std::uniform_int_distribution<int> dist_fill(1, n);
    while ((int)to_destroy.size() < destroy_target) {
        to_destroy.insert(dist_fill(rng));
    }

    return repair_solution_common(sol, to_destroy);
}

Solution tabu_search(const Solution& initial_solution, int num_initial_sol,  vector<double>& iter_current, vector<double>& iter_best, vector<bool>& iter_feasible) {
    (void)num_initial_sol;
    auto ts_start = std::chrono::high_resolution_clock::now();
    auto is_feasible = [](const Solution& sol) {
        return is_feasible_solution(sol);
    };
    reset_penalty_coefficients();
    // Initialize edge records
    edge_records.assign(n + 1, vector<double>(n + 1, 1e10));
    updated_edge_records(initial_solution);
    Solution best_penalized_fallback_solution = initial_solution;
    Solution best_feasible_solution = initial_solution;
    bool initial_feasible = is_feasible(initial_solution);
    double best_feasible_makespan = initial_feasible
        ? initial_solution.total_makespan
        : std::numeric_limits<double>::infinity();
    double score[NUM_NEIGHBORHOODS] = {0.0};
    double weight[NUM_NEIGHBORHOODS];
    for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] = 1.0;
    int count[NUM_NEIGHBORHOODS] = {0};

    iter_current.clear();
    iter_best.clear();
    iter_feasible.clear();

    int no_improve_segments = 0;

    Solution current_sol = initial_solution;

    int iter = 0;
    int total_iters = max(1, CFG_MAX_ITERATIONS);
    CFG_MAX_SEGMENT = compute_segment_count(total_iters, CFG_MAX_ITER_PER_SEGMENT);
    int scoring_mode_iter = 0; // 0: makespan-oriented, 1: workload-oriented
    auto active_solution_score = [&](const Solution& sol) {
        return scoring_mode_iter == 0 ? solution_score_makespan(sol) : solution_score_workload(sol);
    };
    Solution best_segment_sol = current_sol;
    double best_segment_score = active_solution_score(current_sol);
    double best_penalized_fallback_score = active_solution_score(current_sol);
    cout << "=== Starting Unified Tabu Search (Minimizing Active Penalized Score) ===\n";
    cout << "Initial active score: " << best_penalized_fallback_score
         << ", max_iters=" << total_iters
         << ", segment_length=" << CFG_MAX_ITER_PER_SEGMENT
         << ", segments=" << CFG_MAX_SEGMENT << "\n";

    double current_score = best_penalized_fallback_score;
    int segments_per_mode[2] = {0, 0};
    bool feasible_incumbent_improved_this_segment = false;
    double segment_capacity_violation_sum = 0.0;
    double segment_energy_violation_sum = 0.0;
    double segment_deadline_violation_sum = 0.0;
    int segment_violation_observations = 0;
    while (iter < total_iters) {
        if (CFG_TIME_LIMIT_SEC > 0.0) {
            double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
            if (elapsed >= CFG_TIME_LIMIT_SEC) break;
        }
        
        current_score = active_solution_score(current_sol);
        double current_pure_cost = current_sol.total_makespan;
        iter_current.push_back(current_pure_cost);;
        iter_best.push_back(best_feasible_solution.total_makespan);
        iter_feasible.push_back(is_feasible(current_sol));


        // Roulette Wheel Selection
        double total_weight = 0.0;
        for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            total_weight += weight[i];
        }
        double r = ((double) rand() / (RAND_MAX));
        int selected_neighbor = NUM_NEIGHBORHOODS - 1; // fallback: last bucket absorbs rounding
        double cumulative = 0.0;
        for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
            cumulative += weight[i] / total_weight;
            if (r < cumulative) {
                selected_neighbor = i;
                break;
            }
        }

        // Change it to random selection for testing
        //selected_neighbor = rand() % NUM_NEIGHBORHOODS;

        //Change it to round-robin/cyclic for testing
        //selected_neighbor = iter % NUM_NEIGHBORHOODS;
        count[selected_neighbor]++;

        
        // Local Search
        Solution init_neighbor;
        Solution neighbor;
        double segment_reference_score = best_segment_score;
        try {
            if (scoring_mode_iter == 0) {
                init_neighbor = local_search(current_sol, selected_neighbor, iter, segment_reference_score, solution_score_makespan);
            }
            else if (scoring_mode_iter == 1){
                init_neighbor = local_search_all_vehicle(current_sol, selected_neighbor, iter, segment_reference_score, solution_score_workload);
            }
            neighbor = recalculate_solution(init_neighbor);
            if (std::abs(neighbor.deadline_violation - init_neighbor.deadline_violation) > 1e-8 ||
                std::abs(neighbor.capacity_violation - init_neighbor.capacity_violation) > 1e-8 ||
                std::abs(neighbor.energy_violation - init_neighbor.energy_violation) > 1e-8 ||
                std::abs(neighbor.total_makespan - init_neighbor.total_makespan) > 1e-8) {
                cerr << "Warning: iter " << iter
                     << ", neighborhood " << selected_neighbor
                     << " had stale cached metrics; using recalculated neighbor.\n";
            }
             if (!check_solution_integrity(neighbor)) {
                cout << "Iter " << iter << ", Selected Neighborhood: " << selected_neighbor << "failed integrity check!\n";
                cout << "Current Solution:\n";
                print_solution_stream(current_sol, cout);
                cout << "Neighbor Solution:\n";
                print_solution_stream(neighbor, cout);
                neighbor = current_sol;
                exit(1);
            }
        } catch (const std::exception& e) {
            cerr << "\n========== EXCEPTION CAUGHT ==========\n";
            cerr << "Iter: " << iter << " | Neighbor ID: " << selected_neighbor << "\n";
            cerr << "Error: " << e.what() << "\n";
            cerr << "Solution State causing error:\n";
            print_solution_stream(current_sol, cerr);
            cerr << "======================================\n";
            throw; // Re-throw to allow program termination/analysis
        } catch (...) {
            cerr << "\n========== UNKNOWN CRASH/EXCEPTION ==========\n";
            cerr << "Iter: " << iter << " | Neighbor ID: " << selected_neighbor << "\n";
            cerr << "Solution State causing error:\n";
            print_solution_stream(current_sol, cerr);
            cerr << "=============================================\n";
            throw;
        }

        bool neighbor_feasible = is_feasible(neighbor);
        double neighbor_score = active_solution_score(neighbor);
        segment_capacity_violation_sum += neighbor.capacity_violation;
        segment_energy_violation_sum += neighbor.energy_violation;
        segment_deadline_violation_sum += neighbor.deadline_violation;
        ++segment_violation_observations;
        bool improves_segment_reference = neighbor_score + 1e-12 < segment_reference_score;
        bool improves_current_solution = neighbor_score + 1e-12 < current_score;

        // Reward is based on the evaluated candidate, independent of acceptance.
        if (improves_segment_reference) {
            score[selected_neighbor] += gamma1;
        } else if (improves_current_solution) {
            score[selected_neighbor] += gamma2;
        } else {
            score[selected_neighbor] += gamma3;
        }

        // Update Feasible Best
        if (neighbor_feasible) {
             double n_cost = neighbor.total_makespan;
             if (n_cost + 1e-12 < best_feasible_makespan) {
                 best_feasible_solution = neighbor;
                 best_feasible_makespan = n_cost;
                 feasible_incumbent_improved_this_segment = true;
                 cout << "Iter " << iter << " New Best Feasible Makespan: " << best_feasible_makespan << "\n";
             }
        }

        bool accept_candidate = false;
        if (improves_current_solution) {
            accept_candidate = true;
        } else {
            double T = T0 * pow(alpha, iter);
            double denominator = max(fabs(current_score), 1e-12);
            double delta_rel = max(0.0, (neighbor_score - current_score) / denominator);
            double ap = (T > 0.0) ? exp(-delta_rel / T) : 0.0;
            ap = min(1.0, max(0.0, ap));
            double rand_val = ((double) rand() / (RAND_MAX));
            accept_candidate = (rand_val < ap);
        }

        if (accept_candidate) {
            current_sol = neighbor;
            current_score = neighbor_score;
            commit_pending_tabu_attributes(iter);
            if (current_score + 1e-12 < best_segment_score) {
                best_segment_sol = current_sol;
                best_segment_score = current_score;
            }
        } else {
            clear_pending_tabu_attributes();
        }
        // Periodic Weight & Segment Mode Update. Rewards/counts are collected
        // during the segment; weights are updated only after the segment ends.
        if ((iter + 1) % CFG_MAX_ITER_PER_SEGMENT == 0 || iter + 1 == total_iters) {
            segments_per_mode[scoring_mode_iter]++;
            double avg_capacity_violation = segment_violation_observations > 0
                ? segment_capacity_violation_sum / segment_violation_observations
                : 0.0;
            double avg_energy_violation = segment_violation_observations > 0
                ? segment_energy_violation_sum / segment_violation_observations
                : 0.0;
            double avg_deadline_violation = segment_violation_observations > 0
                ? segment_deadline_violation_sum / segment_violation_observations
                : 0.0;
            cout << "=== End of Segment " << (iter / CFG_MAX_ITER_PER_SEGMENT + 1) << " ===\n";
            cout << "Best Penalized Fallback Score: " << best_penalized_fallback_score
                 << " with makespan " << best_penalized_fallback_solution.total_makespan << "\n";
            cout << "Current Solution Score: " << current_score << " with makespan " << current_sol.total_makespan << "\n";
            cout << "Current mode: " << (scoring_mode_iter == 0 ? "Makespan" : "Workload") << "\n";
            cout << "Average normalized violations: cap=" << avg_capacity_violation
                 << ", energy=" << avg_energy_violation
                 << ", deadline=" << avg_deadline_violation << "\n";
            cout << "Current Weights, Rewards and Count of neighborhoods: ";
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                cout << "N" << i << ": w=" << weight[i] << ", R=" << score[i] << ", U=" << count[i] << " | ";
            }
            cout << "\n";
            if (best_segment_score + 1e-12 < best_penalized_fallback_score) {
                best_penalized_fallback_solution = best_segment_sol;
                best_penalized_fallback_score = best_segment_score;
            }
            if (feasible_incumbent_improved_this_segment) {
                no_improve_segments = 0;
            } else {
                no_improve_segments++;
            }

            int next_scoring_mode = scoring_mode_iter;
            bool diversify = false;
            if (no_improve_segments >= H_DIV) {
                diversify = true;
                next_scoring_mode = 0;
            } else if (scoring_mode_iter == 1) {
                next_scoring_mode = 0;
            } else if (no_improve_segments == H_MODE) {
                next_scoring_mode = 1;
            }

            if (diversify) {
                current_sol = destroy_worst_repair_random(current_sol);
                current_sol = recalculate_solution(current_sol);
                if (is_feasible(current_sol) && current_sol.total_makespan + 1e-12 < best_feasible_makespan) {
                    best_feasible_solution = current_sol;
                    best_feasible_makespan = current_sol.total_makespan;
                    cout << "Diversification produced New Best Feasible Makespan: "
                         << best_feasible_makespan << "\n";
                }
                no_improve_segments = 0;
                cout << "No improvement in feasible makespan for " << H_DIV
                     << " segments, applying perturbation. New makespan: "
                     << current_sol.total_makespan << "\n";
                tabu_list_10.clear();
                tabu_list_11.clear();
                tabu_list_20.clear();
                tabu_list_2opt.clear();
                tabu_list_2opt_star.clear();
                tabu_list_22.clear();
                tabu_list_21.clear();
                tabu_list_ejection.clear();
                tabu_list_fusion.clear();
                clear_pending_tabu_attributes();
            }

            if (next_scoring_mode != scoring_mode_iter) {
                scoring_mode_iter = next_scoring_mode;
                cout << "Switching scoring mode to "
                     << (scoring_mode_iter == 0 ? "Makespan" : "Workload")
                     << " for the next segment.\n";
            }

            // Update weights according to w_q <- (1-gamma4)w_q + gamma4 R_q/U_q.
            // Unused operators keep their previous weights.
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                if (count[i] != 0) {
                    weight[i] = (1.0 - gamma4) * weight[i] + gamma4 * (score[i] / count[i]);
                }
            }
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                score[i] = 0.0;
                count[i] = 0;
            }
            update_penalties_from_segment(avg_capacity_violation, avg_energy_violation, avg_deadline_violation);
            current_score = active_solution_score(current_sol);
            best_penalized_fallback_score = active_solution_score(best_penalized_fallback_solution);
            cout << "Penalty lambdas: cap=" << PENALTY_LAMBDA_CAPACITY
                 << ", energy=" << PENALTY_LAMBDA_ENERGY
                 << ", deadline=" << PENALTY_LAMBDA_DEADLINE << "\n";
            segment_capacity_violation_sum = 0.0;
            segment_energy_violation_sum = 0.0;
            segment_deadline_violation_sum = 0.0;
            segment_violation_observations = 0;
            feasible_incumbent_improved_this_segment = false;
            best_segment_sol = current_sol;
            best_segment_score = active_solution_score(current_sol);
        }

        iter++;
    }

    // Post optimization:
    /* Solution improved_feasible = best_feasible_solution;
    if (best_feasible_makespan < std::numeric_limits<double>::infinity()) {
        int post_opt_loop = 20;
        while (post_opt_loop < 0) { // Limit number of post-optimization passes
             post_opt_loop++;
             bool improved_in_pass = false;
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                improved_feasible = local_search(improved_feasible, i, iter, best_feasible_makespan, solution_score_makespan);
                improved_feasible = recalculate_solution(improved_feasible);
                if (improved_feasible.total_makespan + 1e-12 < best_feasible_makespan) {
                    improved_in_pass = true;
                    best_feasible_solution = improved_feasible;
                    best_feasible_makespan = improved_feasible.total_makespan;
                    cout << "Post-Optimization Improved Best Feasible Makespan: " << best_feasible_makespan << "\n";
                }
            }
            if (!improved_in_pass) break; // Exit if no improvement in this pass
        }
    } */

    cout << "Segments per mode: Makespan " << segments_per_mode[0] << ", Workload " << segments_per_mode[1] << "\n";
    if (best_feasible_makespan < std::numeric_limits<double>::infinity()) {
        return best_feasible_solution;
    }
    return best_penalized_fallback_solution;
}

static bool write_iteration_file(const std::string& out_path, const vd& iter_current, const vd& iter_best, const vector<bool>& iter_feasible) {
    std::ofstream ofs(out_path);
    if (!ofs) return false;
    ofs.setf(std::ios::fixed); ofs << setprecision(6);
    ofs << "iter,current_cost,best_cost,feasible\n";
    for (size_t i = 0; i < iter_current.size(); ++i) {
        ofs << i + 1 << "," << iter_current[i] << "," << iter_best[i] << "," << (iter_feasible[i] ? "true" : "false") << "\n";
    }
    return true;
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



static int compute_total_iter_budget(int customer_count, int neighborhood_count) {
    // n * K * ceil(sqrt(n)): each neighborhood gets one sqrt(n)-depth pass over all customers
    int sqrt_n = max(1, (int)ceil(sqrt((double)customer_count)));
    return max(1, customer_count * neighborhood_count * sqrt_n);
}

static int compute_iters_per_segment(int customer_count, int neighborhood_count) {
    // n * ceil(sqrt(K)): one sweep per customer per sqrt(neighborhood count)
    int sqrt_k = max(1, (int)ceil(sqrt(neighborhood_count)));
    return max(1, customer_count * sqrt_k);
}

static int compute_segment_count(int total_iters, int iters_per_segment) {
    return max(1, (total_iters + iters_per_segment - 1) / iters_per_segment);
}

static bool write_output_file(const std::string& out_path, const Solution& sol, double cost, double mean_elapsed_sec, bool final_feasibility, double worst_cost, double mean_cost) {
    std::ofstream ofs(out_path);
    if (!ofs) return false;
    ofs.setf(std::ios::fixed); ofs << setprecision(6);
    ofs << "Initial solution cost: " << cost << "\n";
    ofs << "Improved solution cost: " << sol.total_makespan << "\n";
    ofs << "Worst solution cost: " << worst_cost << "\n";
    ofs << "Mean solution cost: " << mean_cost << "\n";
    ofs << "Mean elapsed time: " << mean_elapsed_sec << " seconds\n";
    ofs << "Final solution feasibility: " << (final_feasibility ? "FEASIBLE" : "INFEASIBLE") << "\n";
    ofs << "Solution Details:\n";
    print_solution_stream(sol, ofs);
    return true;
}

int main(int argc, char* argv[]) {
	    if (argc < 2) {
	        cerr << "Usage: " << argv[0]
	             << " input_file [--print-distance-matrix]"
	             << " [--attempts=N] [--iters=N] [--segment-iters=N] [--segments=N] [--no-improve=N] [--time-limit=SEC] [--segment-length-sec=SEC] [--auto-tune]"
	             << " [--trucks=N] [--drones=N]"
	             << " [--truck-capacity=KG] [--drone-capacity=KG]"
	             << " [--knn-k=K] [--knn-window=W]"
	             << " [--seed=N] [--alpha=X] [--T0=X]"
	             << " [--c-tabu=X] [--h-mode=N] [--h-div=N]"
	             << " [--gamma1=X] [--gamma2=X] [--gamma3=X] [--gamma4=X]"
	             << " [--kappa=X] [--tau-v=X] [--r-destroy=X]"
	             << " [--truck-vmax-file=PATH] [--truck-theta-file=PATH]"
             << "\n";
        return 1;
    }
    string input_file = argv[1];
	    bool print_dist_matrix = false;
	    bool auto_tune = false;
	    bool attempts_explicitly_set = false;
	    int requested_segments = -1;
	    // Parse optional flags
	    for (int ai = 2; ai < argc; ++ai) {
	        string arg = argv[ai];
	        if (arg == "--print-distance-matrix") { print_dist_matrix = true; continue; }
	        string v;
	        if (parse_kv_flag(arg, "--attempts", v)) { CFG_NUM_INITIAL = max(1, stoi(v)); attempts_explicitly_set = true; continue; }
	        if (parse_kv_flag(arg, "--iters", v)) { CFG_MAX_ITERATIONS = max(1, stoi(v)); continue; }
	        if (parse_kv_flag(arg, "--segment-iters", v)) { CFG_MAX_ITER_PER_SEGMENT = max(1, stoi(v)); continue; }
	        if (parse_kv_flag(arg, "--segments", v)) { requested_segments = max(1, stoi(v)); continue; }
	        if (parse_kv_flag(arg, "--no-improve", v)) { CFG_MAX_NO_IMPROVE = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--time-limit", v)) { CFG_TIME_LIMIT_SEC = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--segment-length-sec", v)) { CFG_SEGMENT_LENGTH_SEC = max(1e-9, stod(v)); continue; }
        if (parse_kv_flag(arg, "--trucks", v)) { CFG_OVERRIDE_TRUCKS = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--drones", v)) { CFG_OVERRIDE_DRONES = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--truck-capacity", v)) { Dh = max(1e-9, stod(v)); continue; }
        if (parse_kv_flag(arg, "--drone-capacity", v)) { Dd = max(1e-9, stod(v)); continue; }
        if (parse_kv_flag(arg, "--knn-k", v)) { CFG_KNN_K = max(0, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--knn-window", v)) { CFG_KNN_WINDOW = max(0, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--seed", v)) { CFG_RANDOM_SEED = stoi(v); continue; }
        if (parse_kv_flag(arg, "--alpha", v)) { alpha = min(0.999999, max(0.0, stod(v))); continue; }
        if (parse_kv_flag(arg, "--T0", v)) { T0 = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--c-tabu", v)) { CFG_C_TABU = max(1e-9, stod(v)); continue; }
        if (parse_kv_flag(arg, "--h-mode", v)) { H_MODE = max(0, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--h-div", v)) { H_DIV = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--gamma1", v)) { gamma1 = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--gamma2", v)) { gamma2 = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--gamma3", v)) { gamma3 = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--gamma4", v)) { gamma4 = min(1.0, max(0.0, stod(v))); continue; }
        if (parse_kv_flag(arg, "--kappa", v)) { PENALTY_ADAPTATION_RATE = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--tau-v", v)) { PENALTY_TARGET_VIOLATION = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--r-destroy", v)) { DESTROY_RATE = min(1.0, max(0.0, stod(v))); continue; }
        if (parse_kv_flag(arg, "--truck-vmax-file", v)) { CFG_TRUCK_VMAX_FILE = v; continue; }
        if (parse_kv_flag(arg, "--truck-theta-file", v)) { CFG_TRUCK_THETA_FILE = v; continue; }
        if (arg == "--auto-tune") { auto_tune = true; continue; }
    }
    srand(CFG_RANDOM_SEED);

    // Read input instance
    input(input_file);
    if (CFG_OVERRIDE_TRUCKS > 0) h = CFG_OVERRIDE_TRUCKS;
    if (CFG_OVERRIDE_DRONES > 0) d = CFG_OVERRIDE_DRONES;
    // Recalculate tenures based on instance size
    update_tabu_tenures();
    // Build distance matrix for downstream time computations
    compute_distance_matrices(loc);
    configure_time_segment_boundaries();
    // Build edge/time-dependent truck speed model: v_ijl = theta_ijl * vmax_ij.
    configure_time_dependent_truck_speed_model();
    if (print_dist_matrix) {
        print_distance_matrix();
        return 0; // only print distance matrix and exit
    }

	    if (requested_segments > 0) {
	        CFG_MAX_ITERATIONS = max(1, requested_segments * CFG_MAX_ITER_PER_SEGMENT);
	    }

	    // Optional auto-tuning based on instance size if requested.
	    if (auto_tune) {
	        int tuned_total_iters    = compute_total_iter_budget(n, NUM_NEIGHBORHOODS);
	        int tuned_iters_per_seg  = compute_iters_per_segment(n, NUM_NEIGHBORHOODS);
	        CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, tuned_iters_per_seg);
	        CFG_MAX_ITERATIONS       = min(CFG_MAX_ITERATIONS, tuned_total_iters);
	        CFG_MAX_NO_IMPROVE       = 4 * CFG_MAX_ITER_PER_SEGMENT;
	        if (n <= 20) {
	            if (!attempts_explicitly_set) CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 1);
	            CFG_KNN_K = min(CFG_KNN_K, int(n));
        } else if (n <= 200) {
            if (!attempts_explicitly_set) CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 1);
            CFG_KNN_K = min(CFG_KNN_K, int(n));
        } else {
            if (!attempts_explicitly_set) CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 1);
	            CFG_KNN_K = min(CFG_KNN_K, int(n/2));
	        }
	    }
	    CFG_MAX_SEGMENT = compute_segment_count(CFG_MAX_ITERATIONS, CFG_MAX_ITER_PER_SEGMENT);
	    cout << "Search config: total_iters=" << CFG_MAX_ITERATIONS
	         << " (segments=" << CFG_MAX_SEGMENT
	         << ", segment_iters=" << CFG_MAX_ITER_PER_SEGMENT
	         << ", no_improve=" << CFG_MAX_NO_IMPROVE
	         << ", time_limit_sec=" << CFG_TIME_LIMIT_SEC << ")\n";

    // Precompute KNN lists (if K is zero, disable by building empty adjacency)
    if (CFG_KNN_K > 0) compute_knn_lists(CFG_KNN_K); else { KNN_LIST.assign(n + 1, {}); KNN_ADJ.assign(n + 1, vector<char>(n + 1, 0)); }

    // Pre-filter dronable customers by capacity/energy
    //For another data-testing: change all deadline to a constant 3600 and all serving time to 0

    update_served_by_drone();
    //print test the served by drone
    /* cout << "Customers that can be served by drone:\n";
    for (int i = 1; i <= n; ++i) {
        if (served_by_drone[i]) {
            cout << i << " "; 
        }
    }
    cout << "\n";
    exit(1); */

    // Collect all attempt results, sort, take top-K for mean/worst
    struct AttemptResult {
        Solution sol;
        double initial_cost;
        vd iter_current;
        vd iter_best;
        vector<bool> iter_feasible;
    };
    vector<AttemptResult> all_results;
    all_results.reserve(CFG_NUM_INITIAL);

    auto start_time = std::chrono::high_resolution_clock::now();
    int ablation_seed = CFG_RANDOM_SEED;
    for (int attempt = 0; attempt < CFG_NUM_INITIAL; ++attempt) {
        Solution initial_solution = generate_initial_solution_v2(ablation_seed + attempt);
        vd iter_current, iter_best;
        vector<bool> current_feasibility;
        Solution improved_sol = tabu_search(initial_solution, CFG_NUM_INITIAL, iter_current, iter_best, current_feasibility);
        cout.setf(std::ios::fixed); cout << setprecision(6);
        cout << "Attempt " << attempt + 1 << " cost: " << improved_sol.total_makespan << "\n";
        print_solution_stream(improved_sol, cout);
        all_results.push_back({improved_sol, initial_solution.total_makespan,
                                iter_current, iter_best, current_feasibility});
    }

    // Sort ascending by makespan; best solution = rank 0
    sort(all_results.begin(), all_results.end(),
         [](const AttemptResult& a, const AttemptResult& b) {
             return a.sol.total_makespan < b.sol.total_makespan;
         });

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();

    // Mean and worst computed from top-10 (best runs only)
    const int TOP_K = min(10, (int)all_results.size());
    double sum_overall_cost = 0.0;
    double worst_overall_cost = -1.0;
    for (int i = 0; i < TOP_K; ++i) {
        double mk = all_results[i].sol.total_makespan;
        sum_overall_cost += mk;
        if (mk > worst_overall_cost) worst_overall_cost = mk;
    }
    double mean_overall_cost = sum_overall_cost / TOP_K;
    bool have_best = !all_results.empty();

    if (have_best) {
        const auto& best = all_results[0]; // lowest makespan
        cout << "\n=== Best Across Attempts (top " << TOP_K << "/" << (int)all_results.size() << ") ===\n";
        cout << "Initial Solution Cost: " << best.initial_cost << "\n";
        cout << "Improved Solution Cost: " << best.sol.total_makespan << "\n";
        cout << "Worst Solution Cost (top-" << TOP_K << "): " << worst_overall_cost << "\n";
        cout << "Mean Solution Cost (top-" << TOP_K << "): " << mean_overall_cost << "\n";
        cout << "Mean elapsed Time: " << (elapsed_seconds / all_results.size()) << " seconds\n";
        print_solution_stream(best.sol, cout);
        // check final feasibility
        Solution final_checked_sol = recalculate_solution(best.sol);
        bool final_feas = is_feasible_solution(final_checked_sol);
        if (final_feas) {
            cout << "Final solution feasibility: FEASIBLE\n";
        } else {
            cout << "Final solution feasibility: INFEASIBLE\n";
        }
        string out_best = "output_solution_best.txt";
        if (write_output_file(out_best, best.sol, best.initial_cost, elapsed_seconds / all_results.size(), final_feas, worst_overall_cost, mean_overall_cost)) {
            cout << "Best solution written to " << out_best << "\n";
        } else {
            cout << "Failed to write best solution to " << out_best << "\n";
        }
        string out_iter = "output_mode_0.txt";
        if (write_iteration_file(out_iter, best.iter_current, best.iter_best, best.iter_feasible)) {
            cout << "Iteration data written to " << out_iter << "\n";
        } else {
            cout << "Failed to write iteration data to " << out_iter << "\n";
        }
    }

    return 0;
}

// Run with : g++ -O3 -std=c++20 tabubu.cpp -o tabubu && ./tabubu instance/50.20.4.txt
// Plot history iteration: python plot_iteration.py --input output.txt --save iterations.png
// Plot route: python3 plot_sol.py instance/50.20.4.txt output_solution_best.txt
