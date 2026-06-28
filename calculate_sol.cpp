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
const int NUM_NEIGHBORHOODS = 9;
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

vector<vector<int>> parse_python_list(string s) {
    vector<vector<int>> res;
    vector<int> current_route;
    string num_str;
    int depth = 0;
    for (char c : s) {
        if (c == '[') {
            depth++;
            if (depth == 2) current_route.clear();
        } else if (c == ']') {
            if (!num_str.empty()) {
                current_route.push_back(stoi(num_str));
                num_str.clear();
            }
            if (depth == 2 && !current_route.empty()) {
                res.push_back(current_route);
                current_route.clear();
            }
            depth--;
        } else if (c == ',' || c == ' ') {
            if (!num_str.empty()) {
                current_route.push_back(stoi(num_str));
                num_str.clear();
            }
        } else if (isdigit(c)) {
            num_str += c;
        }
    }
    return res;
}

// Helper to parse the top-level list "[[[...]], [[...]]]"
vector<vector<vector<int>>> parse_clusters(string s) {
    vector<vector<vector<int>>> clusters;
    int depth = 0;
    string buffer;
    for (size_t i = 0; i < s.size(); ++i) {
        char c = s[i];
        if (c == '[') {
            if (depth == 1) buffer.clear(); // Start of a cluster
            if (depth >= 1) buffer += c;
            depth++;
        } else if (c == ']') {
            buffer += c;
            depth--;
            if (depth == 1) { // End of a cluster
                clusters.push_back(parse_python_list(buffer));
                buffer.clear();
            }
        } else {
            if (depth >= 1) buffer += c;
        }
    }
    return clusters;
}

void evaluate_test_file(string filename) {
    ifstream fin(filename);
    if (!fin) { cerr << "Error opening " << filename << endl; return; }
    
    string line;
    string truck_str, drone_str;
    
    while (getline(fin, line)) {
        if (line.find("Truck:") != string::npos) {
            size_t pos = line.find("[");
            if (pos != string::npos) truck_str = line.substr(pos);
        }
        if (line.find("Drone:") != string::npos) {
            size_t pos = line.find("[");
            if (pos != string::npos) drone_str = line.substr(pos);
        }
    }
    
    auto truck_clusters = parse_clusters(truck_str);
    auto drone_clusters = parse_clusters(drone_str);
    
    Solution sol;
    sol.truck_routes.resize(h, {0});
    sol.drone_routes.resize(d, {0});
    
    // Flatten truck routes
    for (int i = 0; i < (int)truck_clusters.size() && i < h; ++i) {
        vector<int> flat;
        for (const auto& r : truck_clusters[i]) {
            if (flat.empty()) flat = r;
            else {
                // Merge: if flat ends with 0 and r starts with 0, skip one 0
                if (!flat.empty() && flat.back() == 0 && !r.empty() && r.front() == 0) {
                    flat.insert(flat.end(), r.begin() + 1, r.end());
                } else {
                    flat.insert(flat.end(), r.begin(), r.end());
                }
            }
        }
        if (!flat.empty()) sol.truck_routes[i] = flat;
    }
    
    // Flatten drone routes
    for (int i = 0; i < (int)drone_clusters.size() && i < d; ++i) {
        vector<int> flat;
        for (const auto& r : drone_clusters[i]) {
            if (flat.empty()) flat = r;
            else {
                // Merge: if flat ends with 0 and r starts with 0, skip one 0
                if (!flat.empty() && flat.back() == 0 && !r.empty() && r.front() == 0) {
                    flat.insert(flat.end(), r.begin() + 1, r.end());
                } else {
                    flat.insert(flat.end(), r.begin(), r.end());
                }
            }
        }
        if (!flat.empty()) sol.drone_routes[i] = flat;
    }

    // Calculate metrics
    sol.truck_route_times.assign(h, 0.0);
    sol.drone_route_times.assign(d, 0.0);
    sol.total_makespan = 0.0;
    
    for(int i=0; i<h; ++i) {
        auto m = check_truck_route_feasibility(sol.truck_routes[i]);
        sol.truck_route_times[i] = m[0];
        sol.total_makespan = max(sol.total_makespan, m[0]);
    }
    for(int i=0; i<d; ++i) {
        auto m = check_drone_route_feasibility(sol.drone_routes[i]);
        sol.drone_route_times[i] = m[0];
        sol.total_makespan = max(sol.total_makespan, m[0]);
    }

    // Write output
    ofstream out("output.txt");
    out.setf(std::ios::fixed); out << setprecision(6);
    
    out << "Truck Routes:\n";
    for (int i = 0; i < h; ++i) {
        out << "Truck " << (i + 1) << ":";
        for (int node : sol.truck_routes[i]) out << " " << node;
        auto m = check_truck_route_feasibility(sol.truck_routes[i]);
        out << " |Truck Time: " << m[0] << "|" << m[0] << "," << m[1] << "," << m[2] << "," << m[3] << "\n";
    }
    
    out << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        out << "Drone " << (i + 1) << ":";
        for (int node : sol.drone_routes[i]) out << " " << node;
        auto m = check_drone_route_feasibility(sol.drone_routes[i]);
        out << " |Drone Time: " << m[0] << "|" << m[0] << "," << m[1] << "," << m[2] << "," << m[3] << "\n";
    }
    
    out << "Total validation: Makespan=" << sol.total_makespan 
        << ", Deadline violation=" << sol.deadline_violation 
        << ", Energy violation=" << sol.energy_violation 
        << ", Capacity violation=" << sol.capacity_violation << "\n";
        
    cout << "Processed test.txt and wrote to output.txt" << endl;
}
int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <instance_file>" << endl;
        return 1;
    }
    string instance_path = argv[1];
    input(instance_path);
    compute_distance_matrices(loc);
    
    //For another data-testing: change all deadline to a constant 3600 and all serving time to 0
    for (int i = 1; i <= n; ++i) {
        deadline[i] = 3600.0;
        serve_truck[i] = 0.0;
        serve_drone[i] = 0.0;
    }
    
    evaluate_test_file("test.txt");
    
    return 0;
}
