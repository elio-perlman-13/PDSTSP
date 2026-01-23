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
double Dh = 1300.0; // truck capacity (all trucks) (kg)
double vmax = 30.0; // truck base speed (km/h)
double drone_limit = 3.0;
double truck_limit = 3.0;
int L = 24; //number of time segments in a day
vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
vd time_segments_sigma = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; //sigma (truck velocity coefficient) for each time segments
//vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
//vd time_segments_sigma = {0.9, 0.8, 0.4, 0.6,0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 2.27, E = 0.6; //drone's weight and energy capacities (for all drones). E in Hours.
double v_fly_drone = 40.0; // speed of the drone (km/h)
double v_take_off = 20.0; // km/h
double v_landing = 10.0; // km/h
double height = 0.05; // km (50m)
double power_beta = 0, power_gamma = 1.0; //coefficients for drone energy consumption per second
//double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
double COST_TRUCK_KM = 1.25;
double COST_DRONE_KM = 0.03;

vvd distance_matrix; // default/truck distance
vvd distance_matrix_euclid; //distance matrices for drone
vvd distance_matrix_manhattan; //distance matrices for truck

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
static double PENALTY_EXPONENT = 1.0;             // exponent for penalty term *

static const double PENALTY_INCREASE = 1.2;       // multiply when violated *
static const double PENALTY_DECREASE = 1.2;       // divide when satisfied *
static const double PENALTY_MIN = 1.0;            // minimum λ value
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
    vd truck_route_cap; //truck_route_cap[i]: total demand served by truck i
    vd drone_route_cap; //drone_route_cap[i]: total demand served by drone i
    double total_distance_truck; // total distance traveled by all trucks
    double total_distance_drone; // total distance traveled by all drones
    double total_time; // total makespan of the solution (max route time)
    double capacity_violation = 0.0;    // sum of excess capacity / total capacity
    double energy_violation = 0.0;      // sum of excess energy / total battery
    double deadline_violation = 0.0;    // sum of deadline breaches / total deadlines
    double total_makespan = 0.0; //temporal because we don't use it in cost calculation
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

    // Default params (if not in file)
    vmax = 30.0;
    v_fly_drone = 40.0;
    Dd = 2.27;
    Dh = 1300.0;
    E = 0.6;

    // Read headers (Key,Value) until a line starts with a digit (data)
    vector<string> data_lines;
    bool reading_data = false;

    while (getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        // Check if this line is data (starts with digit)
        if (isdigit(line[0])) {
            reading_data = true;
            data_lines.push_back(line); // First data line
            continue; 
        }

        if (reading_data) {
            data_lines.push_back(line);
            continue;
        }

        // Header parsing
        size_t comma_pos = line.find(',');
        if (comma_pos != string::npos) {
            string key = line.substr(0, comma_pos);
            string val_str = line.substr(comma_pos + 1);
            double val = 0.0;
            try { val = stod(val_str); } catch(...) {}

            if (key == "NUM DRONES") d = (int)val;
            else if (key == "NUM TRUCKS") h = (int)val;
            else if (key == "TRUCK CAP") Dh = val;
            else if (key == "DRONE CAP") Dd = val;
            else if (key == "TRUCK SPEED") vmax = val;
            else if (key == "DRONE SPEED") v_fly_drone = val;
            else if (key == "DRONE ENDURANCE") E = val;
            else if (key == "DRONE TIME LIMIT") drone_limit = val;
            else if (key == "TRUCK TIME LIMIT") truck_limit = val;
            else if (key == "TRUCK UNIT COST") COST_TRUCK_KM = val;
            else if (key == "DRONE UNIT COST") COST_DRONE_KM = val;
            else if (key == "NUM CUSTOMERS") n = (int)val;
        }
    }

    // Process data lines
    // If 'NUM CUSTOMERS' key was missing, infer n from data lines
    // Assumption: data_lines[0] is depot (ID 0), others are customers. n = size - 1.
    if (n == -1) n = (int)data_lines.size() - 1;

    // Resize globals
    served_by_drone.assign(n+1, 0);
    serve_truck.assign(n+1, 0.0);
    serve_drone.assign(n+1, 0.0);
    deadline.assign(n+1, 0.0);
    demand.assign(n+1, 0.0);
    loc.assign(n+1, Point());
    distance_matrix.assign(n+1, vd(n+1, 0.0));
    distance_matrix_euclid.assign(n+1, vd(n+1, 0.0));
    distance_matrix_manhattan.assign(n+1, vd(n+1, 0.0));

    double depot_x = 0, depot_y = 0;
    
    // Parse data
    for (const string& dline : data_lines) {
        stringstream ss(dline);
        int id;
        double x_km, y_km, dem;
        ss >> id >> x_km >> y_km >> dem;
        
        // Coords kept in km to match km/h speeds and E in hours
        loc[id] = {x_km, y_km, id};
        demand[id] = dem;
        
        if (id == 0) {
            depot_x = x_km;
            depot_y = y_km;
        } else {
            // Determine Dronable
            // 1. Demand <= Drone Cap
            bool dem_ok = (dem <= Dd);
            
            // 2. Distance <= Endurance/2 * Speed
            // E in hours, Speed in km/h -> Range in km
            double dist_to_depot = sqrt(pow(x_km - depot_x, 2) + pow(y_km - depot_y, 2));
            double max_radius = (v_fly_drone * E) / 2.0;

            if (dem_ok && dist_to_depot <= max_radius) {
                served_by_drone[id] = 1;
            } else {
                served_by_drone[id] = 0;
            }
            
            // Service times set to 0 as per paper
            serve_truck[id] = 0.0;
            serve_drone[id] = 0.0;
        }
    }
}

void update_tabu_tenures() {
    // Base heuristic: roughly proportional to sqrt(n) or n/5
    // sqrt(n) scales better for very large instances
    int base = max(20, (int)(2.0 * sqrt(n))); 
    
    TABU_TENURE_BASE = base;
    TABU_TENURE_10 = base;          // Swap
    TABU_TENURE_11 = base;          // Relocate
    TABU_TENURE_20 = base;          // Or-opt
    TABU_TENURE_2OPT = (int)(base * 1.2); // 2-opt usually benefits from slightly longer memory
    TABU_TENURE_2OPT_STAR = base; 
    TABU_TENURE_21 = base;
    TABU_TENURE_22 = base;
    TABU_TENURE_EJECTION = max(30, (int)(base * 2.0)); // Ejection chains are disruptive, keep longer
    
    /* cout << "Dynamic Tabu Tenures set to: " << base 
         << " (2-opt: " << TABU_TENURE_2OPT 
         << ", Ejection: " << TABU_TENURE_EJECTION << ")" << endl; */
}

// Returns pair of distance matrices
void compute_distance_matrices(const vector<Point>& loc) {
    int n = loc.size() - 1; // assuming loc[0] is depot
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distance_matrix_euclid[i][j] = sqrt((loc[i].x - loc[j].x) * (loc[i].x - loc[j].x)
                                         + (loc[i].y - loc[j].y) * (loc[i].y - loc[j].y)); // Euclidean
            distance_matrix_manhattan[i][j] = abs(loc[i].x - loc[j].x) + abs(loc[i].y - loc[j].y); // Manhattan
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
    double time = start; // Hours
    
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist = distance_matrix_manhattan[from][to]; // km
        double v = vmax; // km/h
        if (v < 1e-6) v = 1.0;

        time += dist / v;

        if (to != 0) {
            // serve_truck is 0.0 in input(), so this is safe. 
            // If non-zero, ensure it's in Hours.
            time += serve_truck[to];
        }
    }
    double runtime = time - start;
    double violation = max(0.0, (runtime - truck_limit) / truck_limit);
    return {runtime, violation};
}

pair<double, double> compute_drone_route_energy(const vi& route) {
    double time = 0.0, energy_violation = 0.0;
    // Check capacity (reset at depot)
    double capacity_violation = 0.0;
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            double distance = distance_matrix_euclid[0][customer] * 2.0;
            time += distance / v_fly_drone;
            if (distance / v_fly_drone > E){
                energy_violation += (distance / v_fly_drone - E) / E; // normalized excess
            }
        }
    }
    return {time, energy_violation};
}

pair<double, double> compute_drone_route_time(const vi& route) {
    double time = 0.0; // Hours
    
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        double distance = distance_matrix_euclid[0][customer] * 2.0;
        time += distance / v_fly_drone;
    }
    double violation = max(0.0, (time - drone_limit) / drone_limit);
    return {time, violation};
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
        double distance = distance_matrix_euclid[0][customer] * 2;
        double route_energy = distance / v_fly_drone;
        if (route_energy > E){
            served_by_drone[customer] = 0;
            continue;
        }
        // Deadline: check if round trip can be done within deadline
        if (route_energy > drone_limit){
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
    double time = 0.0, capacity_violation = 0.0, energy_violation = 0.0, deadline_violation = 0.0;
    // Check capacity (reset at depot
    double total_demand = 0.0;
    for (int k = 0; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand = demand[customer];
            if (total_demand > Dd + 1e-9) {
                capacity_violation += (total_demand - Dd) / Dd; // normalized excess
            }
            double distance = distance_matrix_euclid[0][customer] * 2;
            time += distance / v_fly_drone;
            if (distance / v_fly_drone > E){
                energy_violation += (distance / v_fly_drone - E) / E; // normalized excess
            }
        }
    }

    if (time > drone_limit){
        deadline_violation += (time - drone_limit) / drone_limit; // normalized excess
    }
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
double solution_score_l2_norm(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    double sum_sq = 0.0;
    for (double t : sol.truck_route_times) sum_sq += t * t;
    for (double t : sol.drone_route_times) sum_sq += t * t;
    double l2_norm = std::sqrt(sum_sq);

    // 1e-3 ensures it acts as a tie-breaker without overriding the primary Makespan objective
    return (sol.total_makespan + l2_norm * 1e-3) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double solution_score_makespan(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    return (sol.total_makespan) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double solution_score_cost(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    
    double cost = sol.total_distance_truck * COST_TRUCK_KM + sol.total_distance_drone * COST_DRONE_KM;

    return (cost) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double solution_pure_cost(const Solution& sol) {
    double cost = sol.total_distance_truck * COST_TRUCK_KM + sol.total_distance_drone * COST_DRONE_KM;
    return cost;
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

            double new_score = objective_val * pow((1.0 + violation), PENALTY_EXPONENT);
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

            double new_score = objective_val * pow((1.0 + violation), PENALTY_EXPONENT);
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

Solution recalculate_solution(Solution sol) {
    sol.deadline_violation = 0.0;
    sol.energy_violation = 0.0;
    sol.capacity_violation = 0.0;
    sol.total_distance_truck = 0.0;
    sol.total_distance_drone = 0.0;
    sol.truck_route_times.resize(h, 0.0);
    sol.drone_route_times.resize(d, 0.0);

    for (size_t i = 0; i < sol.truck_routes.size(); ++i) {
        vd metrics = check_route_feasibility(sol.truck_routes[i], 0.0, true);
        sol.truck_route_times[i] = metrics[0];
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];

        // Track truck distance
        const vi& route = sol.truck_routes[i];
        for (size_t j = 0; j + 1 < route.size(); ++j) {
            sol.total_distance_truck += distance_matrix_manhattan[route[j]][route[j+1]];
        }
    }
    for (size_t i = 0; i < sol.drone_routes.size(); ++i) {
        vd metrics = check_route_feasibility(sol.drone_routes[i], 0.0, false);
        sol.drone_route_times[i] = metrics[0];
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];

        // Track drone distance (sum of sorties)
        const vi& route = sol.drone_routes[i];
        for (int c : route) {
            sol.total_distance_drone += (2 * distance_matrix_euclid[0][c]);
        }
    }

    sol.total_time = 0.0;
    for (size_t t = 0; t < sol.truck_route_times.size(); ++t) sol.total_time += sol.truck_route_times[t];
    for (size_t t = 0; t < sol.drone_route_times.size(); ++t) sol.total_time += sol.drone_route_times[t];
    
    return sol;
}


Solution generate_initial_solution(){
    Solution sol;
    sol.truck_routes.assign(h, {0, 0});
    sol.drone_routes.assign(d, {0});
    sol.truck_route_times.assign(h, 0.0);
    sol.drone_route_times.assign(d, 0.0);
    sol.truck_route_cap.assign(h, 0.0);
    sol.drone_route_cap.assign(d, 0.0);
    sol.capacity_violation = 0.0;
    sol.energy_violation = 0.0;
    sol.deadline_violation = 0.0;
    sol.total_makespan = 0.0;
    sol.total_distance_truck = 0.0;
    sol.total_distance_drone = 0.0;

    vi customers_to_insert(n);
    iota(customers_to_insert.begin(), customers_to_insert.end(), 1);
    
    // Randomize customer insertion order for variety in solutions
    mt19937 rng(std::random_device{}());
    shuffle(customers_to_insert.begin(), customers_to_insert.end(), rng);

    for (int cust : customers_to_insert) {
        double best_insertion_cost = 1e18;
        int best_vehicle_type = -1; // 0 for truck, 1 for drone
        int best_vehicle_idx = -1;
        int best_insertion_pos = -1;

        // Find best insertion for 'cust' in truck routes
        for (int i = 0; i < h; ++i) {
            for (size_t j = 1; j < sol.truck_routes[i].size(); ++j) {
                vi temp_route = sol.truck_routes[i];
                temp_route.insert(temp_route.begin() + j, cust);
                if (temp_route.back() != 0) temp_route.push_back(0);
                if (temp_route.size() > 2 && temp_route[0] == 0 && temp_route[1] == 0) temp_route.erase(temp_route.begin());
                int u = sol.truck_routes[i][j-1];
                int v = (j == sol.truck_routes[i].size()) ? 0 : sol.truck_routes[i][j];
                double cost_delta = (distance_matrix_manhattan[u][cust] + distance_matrix_manhattan[cust][v] - distance_matrix_manhattan[u][v]) * COST_TRUCK_KM;
                double new_cost = sol.total_distance_truck * COST_TRUCK_KM + sol.total_distance_drone * COST_DRONE_KM + cost_delta;
                double time_delta = (distance_matrix_manhattan[u][cust] + distance_matrix_manhattan[cust][v] - distance_matrix_manhattan[u][v]) / vmax;
                double new_time = time_delta + sol.truck_route_times[i];
                double tmp_deadline_violation = max(0.0, (new_time - truck_limit) / truck_limit);
                double tmp_capacity_violation = 0.0; // fix later
                double tmp_energy_violation = 0.0;
                double score = new_cost * pow((1.0 + tmp_deadline_violation * 1e6 + tmp_capacity_violation * 1e6 + tmp_energy_violation * 1e6), PENALTY_EXPONENT);
                if (score < best_insertion_cost) {
                    best_insertion_cost = score;
                    best_vehicle_type = 0;
                    best_vehicle_idx = i;
                    best_insertion_pos = j;
                }
            }
        }

        for (int i = 0; i < d; ++i) {
            // just push it in the back for drones
            vi temp_route = sol.drone_routes[i];
            temp_route.push_back(cust);
            double dist = distance_matrix_euclid[0][cust] * 2;
            double cost_delta = dist * COST_DRONE_KM;
            double new_cost = sol.total_distance_truck * COST_TRUCK_KM + sol.total_distance_drone * COST_DRONE_KM + cost_delta;
            double time_delta = dist / v_fly_drone;
            double new_time = time_delta + sol.drone_route_times[i];
            double tmp_deadline_violation = max(0.0, (new_time - drone_limit) / drone_limit);
            double tmp_capacity_violation = max(0.0, (demand[cust] - Dd) / Dd);
            double tmp_energy_violation = max(0.0, (dist / v_fly_drone - E) / E);
            double score = new_cost * pow((1.0 + tmp_deadline_violation * 1e6 + tmp_capacity_violation * 1e6 + tmp_energy_violation * 1e6), PENALTY_EXPONENT);
            if (score < best_insertion_cost) {
                best_insertion_cost = score;
                best_vehicle_type = 1;
                best_vehicle_idx = i;
                best_insertion_pos = sol.drone_routes[i].size(); // at the end
            }
        }

        if (best_vehicle_type != -1) {
            if (best_vehicle_type == 0) { // Truck
                vi& route = sol.truck_routes[best_vehicle_idx];
                route.insert(route.begin() + best_insertion_pos, cust);
                vd metrics = check_truck_route_feasibility(route, 0.0);
                sol.truck_route_times[best_vehicle_idx] = metrics[0];
                sol.total_distance_truck += (distance_matrix_manhattan[route[best_insertion_pos - 1]][cust] + distance_matrix_manhattan[cust][route[best_insertion_pos + 1]] - distance_matrix_manhattan[route[best_insertion_pos - 1]][route[best_insertion_pos + 1]]);
                sol.capacity_violation += metrics[3];
                sol.deadline_violation += metrics[1];
            } else { // Drone
                vi& route = sol.drone_routes[best_vehicle_idx];
                route.insert(route.begin() + best_insertion_pos, cust);
                vd metrics = check_drone_route_feasibility(route);
                sol.drone_route_times[best_vehicle_idx] = metrics[0];
                sol.total_distance_drone += (distance_matrix_euclid[0][cust] * 2);
                sol.capacity_violation += metrics[3];
                sol.energy_violation += metrics[2];
                sol.deadline_violation += metrics[1];
            }
        } else {
            cerr << "Warning: No feasible insertion found for customer " << cust << ". It will be left unserved." << endl;
        }
    }

    // Finalize routes and calculate total cost
    sol.deadline_violation = 0;
    sol.capacity_violation = 0;
    sol.energy_violation = 0;
    sol.total_time = 0.0;
    double total_truck_dist = 0.0;
    double total_drone_dist = 0.0;

    for (int i = 0; i < h; ++i) {
        if (sol.truck_routes[i].back() != 0) sol.truck_routes[i].push_back(0);
        if (sol.truck_routes[i].size() <= 2) sol.truck_routes[i] = {0}; // Standardize empty routes
        
        vd metrics = check_truck_route_feasibility(sol.truck_routes[i]);
        sol.truck_route_times[i] = metrics[0];
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];
        sol.total_time += metrics[0];
        for(size_t k = 0; k < sol.truck_routes[i].size() - 1; ++k) {
            total_truck_dist += distance_matrix_manhattan[sol.truck_routes[i][k]][sol.truck_routes[i][k+1]];
            sol.truck_route_cap[i] += demand[sol.truck_routes[i][k]];
        }
    }

    for (int i = 0; i < d; ++i) {
        if (sol.drone_routes[i].size() == 0) sol.drone_routes[i] = {0};
        vd metrics = check_drone_route_feasibility(sol.drone_routes[i]);
        sol.drone_route_times[i] = metrics[0];
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];
        sol.total_time += metrics[0];
        for(size_t k = 0; k < sol.drone_routes[i].size() - 1; ++k) {
            total_drone_dist += distance_matrix_euclid[0][sol.drone_routes[i][k]] * 2;
            sol.drone_route_cap[i] += demand[sol.drone_routes[i][k]];
        }
    }
    sol.total_distance_drone = total_drone_dist;
    sol.total_distance_truck = total_truck_dist;
    return sol;
}

// Stream-based printer to avoid duplicating formatting on stdout/file
static void print_solution_stream(const Solution& sol, std::ostream& os) {
    os << "Truck Routes:\n";
    for (size_t i = 0; i < sol.truck_routes.size(); ++i) {
        os << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            os << node << " ";
        }
        vd truck_metric = check_route_feasibility(sol.truck_routes[i], 0.0, true);
        os << "|Truck Time: " << sol.truck_route_times[i] << "|" << truck_metric[0] << "," << truck_metric[1] << "," << truck_metric[2] << "," << truck_metric[3];
        os << "\n";
    }
    os << "Drone Routes:\n";
    for (size_t i = 0; i < sol.drone_routes.size(); ++i) {
        os << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            os << node << " ";
        }
        vd drone_metric = check_route_feasibility(sol.drone_routes[i], 0.0, false);
        os << "|Drone Time: " << sol.drone_route_times[i] << "|" << drone_metric[0] << "," << drone_metric[1] << "," << drone_metric[2] << "," << drone_metric[3];
        os << "\n";
    }
    os << "Total truck distance: " << sol.total_distance_truck << " km\n";
    os << "Total drone distance: " << sol.total_distance_drone << " km\n";
    os
       << " Total cost=" << solution_score_cost(sol)
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

Solution local_search_all_vehicle(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost, double (*solution_cost)(const Solution&)) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 1e10;
    double total_time_weight = 0.0;
    // Depending on neighbor_id, implement different neighborhood structures
    if (neighbor_id == 0) {
        // Relocate 1 customer
        // Ensure tabu list is sized
        int veh_count = h + d;
        if ((int)tabu_list_10.size() != n + 1 || (veh_count > 0 && (int)tabu_list_10[0].size() != veh_count)) {
            tabu_list_10.assign(n + 1, vector<int>(max(0, veh_count), 0));
        }

        int best_target = -1; 
        int best_cust = -1;   
        int best_pos = -1;    
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        // Recompute baseline metrics from current routes (cached totals may be stale after perturbations)
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        double base_capacity_violation = 0.0, base_energy_violation = 0.0, base_deadline_violation = 0.0;
        vector<double> base_truck_load(h, 0.0);
        vector<double> base_truck_time(h, 0.0), base_drone_time(d, 0.0);

        for (int v = 0; v < h; ++v) {
            const vi& r = initial_solution.truck_routes[v];
            vd metrics = check_route_feasibility(r, 0.0, true);
            base_truck_time[v] = metrics[0];
            base_deadline_violation += metrics[1];
            base_energy_violation += metrics[2];
            base_capacity_violation += metrics[3];
            base_truck_load[v] = accumulate(r.begin(), r.end(), 0.0, [&](double sum, int c){ return sum + demand[c]; });
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
        }
        for (int v = 0; v < d; ++v) {
            const vi& r = initial_solution.drone_routes[v];
            vd metrics = check_route_feasibility(r, 0.0, false);
            base_drone_time[v] = metrics[0];
            base_deadline_violation += metrics[1];
            base_energy_violation += metrics[2];
            base_capacity_violation += metrics[3];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            
            // Baseline Feasibility
            vd crit_route_feas = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust = base_route[p];

                // Prepare Route without Cust (for inter-route target simulation)
                vi base_route_removed = base_route;
                base_route_removed.erase(base_route_removed.begin() + p);
                
                vd base_route_removed_feas = is_truck_mode
                    ? check_route_feasibility(base_route_removed, 0.0, true)
                    : check_route_feasibility(base_route_removed, 0.0, false);

                // --- 1. Removal Delta ---
                double removal_dist_delta = 0.0;
                double removal_time_delta = 0.0;
                
                if (is_truck_mode) {
                   int prev = base_route[p-1];
                   int next = base_route[p+1]; // Truck routes always padded with 0
                   removal_dist_delta = distance_matrix_manhattan[prev][next] 
                                      - distance_matrix_manhattan[prev][cust]
                                      - distance_matrix_manhattan[cust][next];
                   removal_time_delta = removal_dist_delta / vmax;
                } else {
                   // Drone: Remove Sortie (0->C->0)
                   double sortie_dist = distance_matrix_euclid[0][cust] * 2.0;
                   removal_dist_delta = -sortie_dist;
                   removal_time_delta = removal_dist_delta / v_fly_drone;
                }

                // Check Targets
                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (served_by_drone[cust] == 0 && target_veh >= h) continue;

                    // [INTENTION] Explicitly PROHIBIT Intra-route moves for Drones
                    if (!is_truck_mode && target_veh == critical_vehicle_id) continue;

                    bool is_tabu = (tabu_list_10[cust][target_veh] > current_iter);
                    bool target_is_truck = (target_veh < h);
                    
                    // Identify Target Route
                    const vi* target_route_ptr = (target_veh == critical_vehicle_id) 
                         ? &base_route_removed // Inserting into self (minus cust)
                         : (target_is_truck ? &initial_solution.truck_routes[target_veh] 
                                            : &initial_solution.drone_routes[target_veh - h]);
                    const vi& current_target_route = *target_route_ptr;

                    // Feasibility baseline for target (only needed for Inter-route)
                    vd target_route_baseline; 
                    if (target_veh != critical_vehicle_id) {
                         target_route_baseline = target_is_truck
                            ? check_route_feasibility(current_target_route, 0.0, true)
                            : check_route_feasibility(current_target_route, 0.0, false);
                    }

                    // --- 2. Insertion Loop ---
                    int loop_end = target_is_truck ? (int)current_target_route.size() : 1; 
                    
                    for (int p2 = 1; p2 <= loop_end; ++p2) {
                        
                        // Safety check for empty routes (though padding usually prevents this for trucks)
                        if (target_is_truck && p2 >= (int)current_target_route.size()) break;

                        // Insertion Delta
                        double insertion_dist_delta = 0.0;
                        if (target_is_truck) {
                             int prev = current_target_route[p2-1];
                             int next = current_target_route[p2];
                             insertion_dist_delta = distance_matrix_manhattan[prev][cust] 
                                                  + distance_matrix_manhattan[cust][next]
                                                  - distance_matrix_manhattan[prev][next];
                        } else {
                             // Drone: Add Sortie (0->C->0)
                             insertion_dist_delta = distance_matrix_euclid[0][cust] * 2.0;
                        }
                        
                        double insertion_time_delta = insertion_dist_delta / (target_is_truck ? vmax : v_fly_drone);
                        
                        // --- 3. Compute New Totals ---
                        double new_dist_truck = base_truck_dist;
                        double new_dist_drone = base_drone_dist;
                        
                        // Apply Removal
                        if (is_truck_mode) new_dist_truck += removal_dist_delta;
                        else new_dist_drone += removal_dist_delta;
                        
                        // Apply Insertion
                        if (target_is_truck) new_dist_truck += insertion_dist_delta;
                        else new_dist_drone += insertion_dist_delta;
                        
                        // Penalties
                        double cap_delta = 0.0; 
                        double energy_delta = 0.0;
                        double time_viol_delta = 0.0;

                        // Capacity Delta
                        cap_delta += (base_route_removed_feas[3] - crit_route_feas[3]);
                        if (target_veh != critical_vehicle_id) {
                            if (target_is_truck) {
                                double limit = Dh;
                                double current_load = base_truck_load[target_veh];
                                double old_viol = max(0.0, (current_load - limit) / limit);
                                double new_viol = max(0.0, (current_load + demand[cust] - limit) / limit);
                                cap_delta += (new_viol - old_viol);
                            } else {
                                double old_viol = target_route_baseline[3];
                                double added_viol = max(0.0, (demand[cust] - Dd) / Dd);
                                double new_viol = old_viol + added_viol;
                                cap_delta += (new_viol - old_viol);
                            }
                        }

                        // Energy Delta (Drones only)
                        energy_delta += (base_route_removed_feas[2] - crit_route_feas[2]);
                        if (!target_is_truck) {
                             double sortie_time = (distance_matrix_euclid[0][cust]*2.0)/v_fly_drone;
                             double e_viol_add = (sortie_time > E) ? (sortie_time - E)/E : 0.0;
                             energy_delta += e_viol_add;
                        }

                        // Time/Deadline Delta:
                            double base_time_source = is_truck_mode 
                               ? base_truck_time[critical_vehicle_id]
                               : base_drone_time[critical_vehicle_id - h];
                            double new_time_source = base_time_source + removal_time_delta;
                        
                            double new_time_target = 0;
                            if (target_veh == critical_vehicle_id) {
                                new_time_target = new_time_source + insertion_time_delta; 
                            } else {
                                double base_time_target = target_is_truck 
                                  ? base_truck_time[target_veh]
                                  : base_drone_time[target_veh - h];
                                new_time_target = base_time_target + insertion_time_delta;
                            }

                        // Violations
                        double src_lim = is_truck_mode ? truck_limit : drone_limit;
                        double tgt_lim = target_is_truck ? truck_limit : drone_limit;
                        
                        // Subtract Old Violations
                            time_viol_delta -= crit_route_feas[1];
                            if (target_veh != critical_vehicle_id) {
                                time_viol_delta -= target_route_baseline[1];
                            }

                        // Add New Violations
                        if (target_veh == critical_vehicle_id) {
                             time_viol_delta += max(0.0, (new_time_target - src_lim)/src_lim);
                        } else {
                             time_viol_delta += max(0.0, (new_time_source - src_lim)/src_lim); 
                             time_viol_delta += max(0.0, (new_time_target - tgt_lim)/tgt_lim); 
                        }

                        double new_cap_violation = base_capacity_violation + cap_delta;
                        double new_energy_violation = base_energy_violation + energy_delta;
                        double new_deadline_violation = base_deadline_violation + time_viol_delta;

                        double penalty = pow(1.0 
                            + PENALTY_LAMBDA_CAPACITY * max(0.0, new_cap_violation)
                            + PENALTY_LAMBDA_ENERGY * max(0.0, new_energy_violation)
                            + PENALTY_LAMBDA_DEADLINE * max(0.0, new_deadline_violation),
                            PENALTY_EXPONENT);

                        // Pure Cost Score (Cost x Penalty)
                        double new_score = (new_dist_truck * COST_TRUCK_KM + new_dist_drone * COST_DRONE_KM) * penalty;

                        if (is_tabu && !(new_score + 1e-8 < best_cost && new_deadline_violation < 1e-8 && new_cap_violation < 1e-8)) {
                            continue;
                        }

                        if (new_score < best_neighbor_cost_local) {
                            best_neighbor_cost_local = new_score;
                            best_cust = cust;
                            best_target = target_veh;
                            if (target_is_truck) best_pos = p2;
                            else best_pos = current_target_route.size(); // Append for drones
                        }
                    }
                }
            }
        }; 

        for (int veh = 0; veh < h + d; ++veh) {
            bool is_truck = veh < h;
            const vi& r = is_truck ? initial_solution.truck_routes[veh] : initial_solution.drone_routes[veh-h];
            if(r.size() > 2 || (!is_truck && r.size() > 1)) // Drones need >1 (default {0}) to have customers
                consider_relocate(r, is_truck, veh);
        }

        if (best_cust != -1) {
            Solution candidate = initial_solution;
            // 1. Remove
            bool found = false;
            for(int v=0; v<h; ++v) {
                auto& r = candidate.truck_routes[v];
                auto it = find(r.begin(), r.end(), best_cust);
                if(it!=r.end()){ r.erase(it); found=true; break; }
            }
            if(!found) {
                for(int v=0; v<d; ++v) {
                    auto& r = candidate.drone_routes[v];
                    auto it = find(r.begin(), r.end(), best_cust);
                    if(it!=r.end()){ r.erase(it); break; }
                }
            }

            // 2. Insert
            if (best_target < h) {
                auto& r = candidate.truck_routes[best_target];
                int pos = max(1, min((int)r.size(), best_pos));
                r.insert(r.begin()+pos, best_cust);
            } else {
                auto& r = candidate.drone_routes[best_target-h];
                r.push_back(best_cust);
            }

            // 3. Recalc All (Manually to ensure correctness)
            candidate.total_distance_truck = 0.0;
            candidate.total_distance_drone = 0.0;
            candidate.capacity_violation = 0.0;
            candidate.deadline_violation = 0.0;
            candidate.energy_violation = 0.0;
            candidate.total_time = 0.0;

            for(int v=0; v<h; ++v) {
            vd m = check_route_feasibility(candidate.truck_routes[v], 0.0, true);
            candidate.truck_route_times[v] = m[0];
            candidate.truck_route_cap[v] = accumulate(candidate.truck_routes[v].begin(), candidate.truck_routes[v].end(), 0.0, [&](double sum, int c){ return sum + demand[c]; });
            candidate.deadline_violation += m[1];
            candidate.energy_violation += m[2];
            candidate.capacity_violation += m[3];
            candidate.total_time += m[0];
            const auto& r = candidate.truck_routes[v];
            for(size_t i=0; i+1<r.size(); ++i) 
                candidate.total_distance_truck += distance_matrix_manhattan[r[i]][r[i+1]];
            }
            for(int v=0; v<d; ++v) {
                vd m = check_route_feasibility(candidate.drone_routes[v], 0.0, false);
                candidate.drone_route_times[v] = m[0];
                candidate.drone_route_cap[v] = accumulate(candidate.drone_routes[v].begin(), candidate.drone_routes[v].end(), 0.0, [&](double sum, int c){ return sum + demand[c]; });
                candidate.deadline_violation += m[1];
                candidate.energy_violation += m[2];
                candidate.capacity_violation += m[3];
                candidate.total_time += m[0];
                const auto& r = candidate.drone_routes[v];
                for(int cust_id : r) {
                    if (cust_id > 0) // Ignore depot if it's in the list
                        candidate.total_distance_drone += distance_matrix_euclid[0][cust_id] * 2.0;
                }
            }
            best_candidate_neighbor = candidate;
        }

        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
             best_neighbor = best_candidate_neighbor;
             best_neighbor_cost = best_neighbor_cost_local;
             tabu_list_10[best_cust][best_target] = current_iter + TABU_TENURE_10;
        }
        return best_neighbor;

    } else if (neighbor_id == 1) {
        // Ensure tabu_list_10 is n+1 x n+1 (prior N0 may resize inner dim to vehicle count)
        if ((int)tabu_list_10.size() != n + 1 || ((int)tabu_list_10.size() > 0 && (int)tabu_list_10[0].size() != n + 1)) {
            tabu_list_10.assign(n + 1, vector<int>(n + 1, 0));
        }

        int best_cust_a = -1, best_cust_b = -1;
        int best_pos_a = -1, best_pos_b = -1;
        int best_veh_a = -1, best_veh_b = -1;
        double best_neighbor_cost_local = 1e10;

        // 1. Pre-calculate metric vectors for all routes
        vector<vd> route_metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
        }
        for (int i = 0; i < d; ++i) {
            route_metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        // --- 2. Intra-route Swaps (Trucks only) ---
        for (int v_idx = 0; v_idx < h; ++v_idx) {
            const auto& route = initial_solution.truck_routes[v_idx];
            if (route.size() <= 3) continue; // Not enough customers to swap

            for (size_t i = 1; i < route.size() - 2; ++i) {
                for (size_t j = i + 1; j < route.size() - 1; ++j) {
                    int cust1 = route[i];
                    int cust2 = route[j];

                    bool is_tabu = tabu_list_10[min(cust1, cust2)][max(cust1, cust2)] > current_iter;

                    double delta_dist;
                    int prev1 = route[i - 1], next1 = route[i + 1];
                    int prev2 = route[j - 1], next2 = route[j + 1];

                    if (j == i + 1) { // Adjacent
                        delta_dist = (distance_matrix_manhattan[prev1][cust2] + distance_matrix_manhattan[cust2][cust1] + distance_matrix_manhattan[cust1][next2]) -
                                     (distance_matrix_manhattan[prev1][cust1] + distance_matrix_manhattan[cust1][cust2] + distance_matrix_manhattan[cust2][next2]);
                    } else { // Non-adjacent
                        delta_dist = (distance_matrix_manhattan[prev1][cust2] + distance_matrix_manhattan[cust2][next1] + distance_matrix_manhattan[prev2][cust1] + distance_matrix_manhattan[cust1][next2]) -
                                     (distance_matrix_manhattan[prev1][cust1] + distance_matrix_manhattan[cust1][next1] + distance_matrix_manhattan[prev2][cust2] + distance_matrix_manhattan[cust2][next2]);
                    }

                    double delta_time = delta_dist / vmax;
                    auto old_metrics = route_metrics[v_idx];
                    double new_route_time = old_metrics[0] + delta_time;
                    
                    double old_route_viol = old_metrics[1];
                    double new_route_viol = max(0.0, (new_route_time - truck_limit) / truck_limit);
                    double total_deadline_viol = initial_solution.deadline_violation - old_route_viol + new_route_viol;

                    double total_cap_viol = initial_solution.capacity_violation; // No change
                    double total_energy_viol = initial_solution.energy_violation; // No change

                    double new_truck_dist = base_truck_dist + delta_dist;
                    double penalty = pow(1.0 + PENALTY_LAMBDA_DEADLINE * total_deadline_viol + PENALTY_LAMBDA_CAPACITY * total_cap_viol + PENALTY_LAMBDA_ENERGY * total_energy_viol, PENALTY_EXPONENT);
                    double cand_cost = (new_truck_dist * COST_TRUCK_KM + base_drone_dist * COST_DRONE_KM) * penalty;

                    if (is_tabu && !(cand_cost + 1e-8 < best_cost && total_deadline_viol < 1e-8 && total_cap_viol < 1e-8)) continue;
                    
                    if (cand_cost < best_neighbor_cost_local) {
                        best_neighbor_cost_local = cand_cost;
                        best_cust_a = cust1; best_cust_b = cust2;
                        best_pos_a = i; best_pos_b = j;
                        best_veh_a = v_idx; best_veh_b = v_idx;
                    }
                }
            }
        }

        // --- 3. Inter-route Swaps ---
        for (int v_idx_a = 0; v_idx_a < h + d; ++v_idx_a) {
            bool is_truck_a = v_idx_a < h;
            const auto& route_a = is_truck_a ? initial_solution.truck_routes[v_idx_a] : initial_solution.drone_routes[v_idx_a - h];
            if (route_a.empty() || (is_truck_a && route_a.size() <= 2)) continue;

            for (int v_idx_b = v_idx_a + 1; v_idx_b < h + d; ++v_idx_b) {
                bool is_truck_b = v_idx_b < h;
                const auto& route_b = is_truck_b ? initial_solution.truck_routes[v_idx_b] : initial_solution.drone_routes[v_idx_b - h];
                if (route_b.empty() || (is_truck_b && route_b.size() <= 2)) continue;

                for (size_t i = (is_truck_a ? 1 : 0); i < (is_truck_a ? route_a.size() - 1 : route_a.size()); ++i) {
                    for (size_t j = (is_truck_b ? 1 : 0); j < (is_truck_b ? route_b.size() - 1 : route_b.size()); ++j) {
                        int cust1 = route_a[i];
                        int cust2 = route_b[j];

                        if (cust1 == 0 || cust2 == 0) continue;
                        bool is_tabu = tabu_list_10[min(cust1, cust2)][max(cust1, cust2)] > current_iter;
                        if (!is_truck_a && served_by_drone[cust2] == 0) continue;
                        if (!is_truck_b && served_by_drone[cust1] == 0) continue;

                        double delta_truck_dist = 0.0, delta_drone_dist = 0.0;
                        double delta_time_a = 0.0, delta_time_b = 0.0;
                        double delta_cap_viol = 0.0, delta_energy_viol = 0.0;

                        // --- Distance & Time Deltas ---
                        if (is_truck_a) {
                            double dist_change = (distance_matrix_manhattan[route_a[i-1]][cust2] + distance_matrix_manhattan[cust2][route_a[i+1]])
                                               - (distance_matrix_manhattan[route_a[i-1]][cust1] + distance_matrix_manhattan[cust1][route_a[i+1]]);
                            delta_truck_dist += dist_change; delta_time_a = dist_change / vmax;
                        } else {
                            double dist_change = (distance_matrix_euclid[0][cust2] * 2.0) - (distance_matrix_euclid[0][cust1] * 2.0);
                            delta_drone_dist += dist_change; delta_time_a = dist_change / v_fly_drone;
                        }
                        if (is_truck_b) {
                            double dist_change = (distance_matrix_manhattan[route_b[j-1]][cust1] + distance_matrix_manhattan[cust1][route_b[j+1]])
                                               - (distance_matrix_manhattan[route_b[j-1]][cust2] + distance_matrix_manhattan[cust2][route_b[j+1]]);
                            delta_truck_dist += dist_change; delta_time_b = dist_change / vmax;
                        } else {
                            double dist_change = (distance_matrix_euclid[0][cust1] * 2.0) - (distance_matrix_euclid[0][cust2] * 2.0);
                            delta_drone_dist += dist_change; delta_time_b = dist_change / v_fly_drone;
                        }

                        // --- Capacity Violation Delta ---
                        double demand_diff = demand[cust1] - demand[cust2];

                        if (is_truck_a) {
                            double old_load_a = initial_solution.truck_route_cap[v_idx_a];
                            double new_load_a = old_load_a - demand_diff;
                            double limit_a = Dh;
                            delta_cap_viol += max(0.0, (new_load_a - limit_a) / limit_a) - max(0.0, (old_load_a - limit_a) / limit_a);
                        } else {
                            double cap_old = max(0.0, (demand[cust1] - Dd) / Dd);
                            double cap_new = max(0.0, (demand[cust2] - Dd) / Dd);
                            delta_cap_viol += (cap_new - cap_old);
                        }

                        if (is_truck_b) {
                            double old_load_b = initial_solution.truck_route_cap[v_idx_b];
                            double new_load_b = old_load_b + demand_diff;
                            double limit_b = Dh;
                            delta_cap_viol += max(0.0, (new_load_b - limit_b) / limit_b) - max(0.0, (old_load_b - limit_b) / limit_b);
                        } else {
                            double cap_old = max(0.0, (demand[cust2] - Dd) / Dd);
                            double cap_new = max(0.0, (demand[cust1] - Dd) / Dd);
                            delta_cap_viol += (cap_new - cap_old);
                        }

                        // --- Deadline & Energy Violation Delta ---
                        auto old_metrics_a = route_metrics[v_idx_a], old_metrics_b = route_metrics[v_idx_b];
                        double new_time_a = old_metrics_a[0] + delta_time_a, new_time_b = old_metrics_b[0] + delta_time_b;
                        double limit_time_a = is_truck_a ? truck_limit : drone_limit, limit_time_b = is_truck_b ? truck_limit : drone_limit;
                        double delta_deadline_viol = (max(0.0, (new_time_a - limit_time_a) / limit_time_a) - old_metrics_a[1]) + (max(0.0, (new_time_b - limit_time_b) / limit_time_b) - old_metrics_b[1]);
                        
                        if (!is_truck_a) {
                            double e_old = max(0.0, ((distance_matrix_euclid[0][cust1] * 2.0) / v_fly_drone - E) / E);
                            double e_new = max(0.0, ((distance_matrix_euclid[0][cust2] * 2.0) / v_fly_drone - E) / E);
                            delta_energy_viol += (e_new - e_old);
                        }
                        if (!is_truck_b) {
                            double e_old = max(0.0, ((distance_matrix_euclid[0][cust2] * 2.0) / v_fly_drone - E) / E);
                            double e_new = max(0.0, ((distance_matrix_euclid[0][cust1] * 2.0) / v_fly_drone - E) / E);
                            delta_energy_viol += (e_new - e_old);
                        }

                        // --- Final Score ---
                        double total_deadline_viol = initial_solution.deadline_violation + delta_deadline_viol;
                        double total_cap_viol = initial_solution.capacity_violation + delta_cap_viol;
                        double total_energy_viol = initial_solution.energy_violation + delta_energy_viol;
                        double new_truck_dist = base_truck_dist + delta_truck_dist;
                        double new_drone_dist = base_drone_dist + delta_drone_dist;

                        double penalty = pow(1.0 + PENALTY_LAMBDA_DEADLINE * total_deadline_viol + PENALTY_LAMBDA_CAPACITY * total_cap_viol + PENALTY_LAMBDA_ENERGY * total_energy_viol, PENALTY_EXPONENT);
                        double cand_cost = (new_truck_dist * COST_TRUCK_KM + new_drone_dist * COST_DRONE_KM) * penalty;

                        bool is_feasible_cand = total_deadline_viol < 1e-8 && total_cap_viol < 1e-8 && total_energy_viol < 1e-8;
                        if (is_tabu && !(cand_cost + 1e-8 < best_cost && is_feasible_cand)) continue;

                        if (cand_cost < best_neighbor_cost_local) {
                            best_neighbor_cost_local = cand_cost;
                            best_cust_a = cust1; best_cust_b = cust2;
                            best_pos_a = i; best_pos_b = j;
                            best_veh_a = v_idx_a; best_veh_b = v_idx_b;
                        }
                    }
                }
            }
        }

        // --- Apply the best move found ---
        if (best_cust_a != -1) {
            best_neighbor = initial_solution;
            if (best_veh_a == best_veh_b) { // Intra-route
                swap(best_neighbor.truck_routes[best_veh_a][best_pos_a], best_neighbor.truck_routes[best_veh_a][best_pos_b]);
            } else { // Inter-route
                (best_veh_a < h ? best_neighbor.truck_routes[best_veh_a] : best_neighbor.drone_routes[best_veh_a - h])[best_pos_a] = best_cust_b;
                (best_veh_b < h ? best_neighbor.truck_routes[best_veh_b] : best_neighbor.drone_routes[best_veh_b - h])[best_pos_b] = best_cust_a;
            }
            
            best_neighbor = recalculate_solution(best_neighbor);
            
            if (solution_cost(best_neighbor) < best_neighbor_cost) {
                 best_neighbor_cost = solution_cost(best_neighbor);
                 tabu_list_10[min(best_cust_a, best_cust_b)][max(best_cust_a, best_cust_b)] = current_iter + TABU_TENURE_10;
                 return best_neighbor;
            }
        }
        return initial_solution;
    } else if (neighbor_id == 2) {
         // N2: Relocate two ADJACENT customers from ANY vehicle to ANY vehicle
        int veh_count = h + d;

        int best_cust1 = -1, best_cust2 = -1;
        int best_target_veh = -1;
        int best_pos = -1;
        double best_neighbor_cost_local = 1e10;

        // 1. Pre-calculate metric vectors for all routes
        vector<vd> route_metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        vd total_capacity_truck(h, 0.0);
        for (int i = 0; i < h; ++i) {
            route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            if (!r.empty() && r.back() != 0) base_truck_dist += distance_matrix_manhattan[r.back()][0];
            for (int c : r) total_capacity_truck[i] += demand[c];
        }
        for (int i = 0; i < d; ++i) {
            route_metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        for (int v1 = 0; v1 < h + d; ++v1) {
            bool is_v1_truck = v1 < h;
            const auto& r1 = is_v1_truck ? initial_solution.truck_routes[v1] : initial_solution.drone_routes[v1 - h];

            // Adjacency is defined by position in the route list for both trucks and drones.
            // Truck: [0, c1, c2, c3, 0]. Drone: [0, c1, c2, c3]
            size_t start_idx = is_v1_truck ? 1 : 0;
            if (r1.size() < start_idx + 2 + (is_v1_truck ? 1 : 0)) continue; // Not enough customers for a pair

            for (size_t i = start_idx; i < r1.size() - 1 - (is_v1_truck ? 1 : 0); ++i) {
                int cust1 = r1[i];
                int cust2 = r1[i+1];
                if (cust1 == 0 || cust2 == 0) continue;

                // --- 1. Removal Delta ---
                double removal_dist_delta = 0.0;
                double removal_time_delta = 0.0;
                
                vi r1_removed = r1;
                r1_removed.erase(r1_removed.begin() + i, r1_removed.begin() + i + 2);

                vd r1_orig_feas = check_route_feasibility(r1, 0.0, is_v1_truck);
                vd r1_removed_feas = check_route_feasibility(r1_removed, 0.0, is_v1_truck);
                removal_time_delta = r1_removed_feas[0] - r1_orig_feas[0];
                if (is_v1_truck) {
                    int prev = r1[i-1];
                    int next = (i + 2 < r1.size()) ? r1[i+2] : 0; // Fallback to depot if missing
                    removal_dist_delta = distance_matrix_manhattan[prev][next]
                                       - distance_matrix_manhattan[prev][cust1]
                                       - distance_matrix_manhattan[cust1][cust2]
                                       - distance_matrix_manhattan[cust2][next];
                } else { // Drone
                    removal_dist_delta = -(distance_matrix_euclid[0][cust1] * 2.0 + distance_matrix_euclid[0][cust2] * 2.0);
                }

                double tmp_total_dist_truck = base_truck_dist;
                double tmp_total_dist_drone = base_drone_dist;
                if (is_v1_truck) tmp_total_dist_truck += removal_dist_delta;
                else tmp_total_dist_drone += removal_dist_delta;

                for (int v2 = 0; v2 < h + d; ++v2) {
                    bool is_v2_truck = v2 < h;
                    if ((!served_by_drone[cust1] || !served_by_drone[cust2]) && !is_v2_truck) continue;
                    // If both are drones, they must go to separate vehicles
                    if (!is_v2_truck && v1 == v2) continue;

                    vector<int> tabu_key = {cust1, cust2, v2};
                    bool is_tabu = (tabu_list_20.count(tabu_key) && tabu_list_20[tabu_key] > current_iter);
                    
                    const auto& r2_base = (v1 == v2) ? r1_removed : (is_v2_truck ? initial_solution.truck_routes[v2] : initial_solution.drone_routes[v2-h]);
                    int loop_end = is_v2_truck ? r2_base.size() : 1;

                    for (int p_insert = 1; p_insert <= loop_end; ++p_insert) {
                        if (is_v2_truck && p_insert >= r2_base.size() && r2_base.size() > 1) continue;

                        // --- 2. Insertion Delta ---
                        double insertion_dist_delta = 0.0;
                        double insertion_time_delta = 0.0;
                        if (is_v2_truck) {
                            if (r2_base.empty()) continue; // No depot to attach to
                            int prev = r2_base[p_insert - 1];
                            int next = (p_insert < (int)r2_base.size()) ? r2_base[p_insert] : 0; // Fallback depot if route has no trailing depot
                            insertion_dist_delta = distance_matrix_manhattan[prev][cust1]
                                                 + distance_matrix_manhattan[cust1][cust2]
                                                 + distance_matrix_manhattan[cust2][next]
                                                 - distance_matrix_manhattan[prev][next];
                            insertion_time_delta = insertion_dist_delta / vmax;
                        } else { // Drone
                            insertion_dist_delta = distance_matrix_euclid[0][cust1] * 2.0 + distance_matrix_euclid[0][cust2] * 2.0;
                            insertion_time_delta = insertion_dist_delta / v_fly_drone;
                        }

                        // --- 3. Cost & Penalty Deltas ---
                        double new_dist_truck = tmp_total_dist_truck;
                        double new_dist_drone = tmp_total_dist_drone;
                        if (is_v2_truck) new_dist_truck += insertion_dist_delta;
                        else new_dist_drone += insertion_dist_delta;

                        // Violations
                        double base_cap_violation = initial_solution.capacity_violation;
                        double base_energy_violation = initial_solution.energy_violation;
                        double base_deadline_violation = initial_solution.deadline_violation;
                        double cap_delta = 0.0, energy_delta = 0.0, time_viol_delta = 0.0;

                        if (v1 == v2){
                            // Same vehicle: cap_violation and energy_violation not changed
                            // Only deadline violation changes
                            double old_viol = route_metrics[v1][1];
                            double new_route_time = route_metrics[v1][0] + removal_time_delta + insertion_time_delta;
                            double new_viol = max(0.0, (new_route_time - (is_v1_truck ? truck_limit : drone_limit)) / (is_v1_truck ? truck_limit : drone_limit));
                            time_viol_delta += (new_viol - old_viol);
                        }
                        else {
                            // Capacity & Energy Violation Delta
                            cap_delta = r1_removed_feas[3] - r1_orig_feas[3];
                            if (is_v2_truck) {
                                double old_load = total_capacity_truck[v2];
                                double new_load = old_load + demand[cust1] + demand[cust2];
                                double limit = Dh;
                                double old_viol = max(0.0, (old_load - limit) / limit);
                                double new_viol = max(0.0, (new_load - limit) / limit);
                                cap_delta += (new_viol - old_viol);
                            }
                            else {
                                // This won't happen if served_by_drone is respected
                            }
                            // Deadline Violation Delta
                            double old_viol_src = route_metrics[v1][1];
                            double new_route_time_src = route_metrics[v1][0] + removal_time_delta;
                            double new_viol_src = max(0.0, (new_route_time_src - (is_v1_truck ? truck_limit : drone_limit)) / (is_v1_truck ? truck_limit : drone_limit));
                            time_viol_delta += (new_viol_src - old_viol_src);
                            double old_viol_tgt = route_metrics[v2][1];
                            double new_route_time_tgt = route_metrics[v2][0] + insertion_time_delta;
                            double new_viol_tgt = max(0.0, (new_route_time_tgt - (is_v2_truck ? truck_limit : drone_limit)) / (is_v2_truck ? truck_limit : drone_limit));
                            time_viol_delta += (new_viol_tgt - old_viol_tgt);
                        }
                        double new_cap_violation = base_cap_violation + cap_delta;
                        double new_energy_violation = base_energy_violation + energy_delta;
                        double new_deadline_violation = base_deadline_violation + time_viol_delta;

                        double penalty = pow(1.0 
                            + PENALTY_LAMBDA_CAPACITY * max(0.0, new_cap_violation)
                            + PENALTY_LAMBDA_ENERGY * max(0.0, new_energy_violation)
                            + PENALTY_LAMBDA_DEADLINE * max(0.0, new_deadline_violation),
                            PENALTY_EXPONENT);

                        double new_score = (new_dist_truck * COST_TRUCK_KM + new_dist_drone * COST_DRONE_KM) * penalty;
                        
                        if (is_tabu && !(new_score + 1e-8 < best_cost && new_deadline_violation < 1e-8 && new_cap_violation < 1e-8)) {
                            continue;
                        }

                        if (new_score < best_neighbor_cost_local) {
                            best_neighbor_cost_local = new_score;
                            best_cust1 = cust1;
                            best_cust2 = cust2;
                            best_target_veh = v2;
                            best_pos = p_insert;
                        }
                    }
                }
            }
        }

        if (best_cust1 != -1) {
            Solution candidate = initial_solution;
            
            // 1. Remove cust1 and cust2 from their original route
            for(int v=0; v<h+d; ++v) {
                auto& r = (v < h) ? candidate.truck_routes[v] : candidate.drone_routes[v-h];
                if (r.size() < 2) continue;

                bool found_pair = false;
                for(size_t i = 0; i + 1 < r.size(); ++i) {
                    if (r[i] == best_cust1 && r[i+1] == best_cust2) {
                        r.erase(r.begin() + i, r.begin() + i + 2);
                        found_pair = true;
                        break;
                    }
                }
                if (found_pair) break;
            }

            // 2. Insert into target route
            if (best_target_veh < h) { // Truck target
                auto& r_target = candidate.truck_routes[best_target_veh];
                int pos = max(1, min((int)r_target.size() -1, best_pos));
                r_target.insert(r_target.begin() + pos, {best_cust1, best_cust2});
                if (r_target.back() != 0) r_target.push_back(0);
            } else { // Drone target: push to back as separate sorties
                candidate.drone_routes[best_target_veh-h].push_back(best_cust1);
                candidate.drone_routes[best_target_veh-h].push_back(best_cust2);
            }

            best_neighbor = recalculate_solution(candidate);
            best_neighbor_cost = solution_score_cost(best_neighbor);
            
            if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
                 best_neighbor_cost = best_neighbor_cost_local;
            }

            vector<int> tabu_key = {best_cust1, best_cust2, best_target_veh};
            tabu_list_20[tabu_key] = current_iter + TABU_TENURE_20;
        }
        
        return best_neighbor;
    } else if (neighbor_id == 3) {
        // Neighborhood 3: 2-opt within each subroute (between depot nodes) for trucks ONLY.
        // Finds the best 2-opt move across all routes that yields the largest local time drop.

        if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_edge_u = -1, best_edge_v = -1;
        int best_i = -1, best_j = -1;
        int best_route_idx = -1;

        // Pre-compute truck route metrics
        vector<vd> truck_route_metrics(h);
        vd base_route_dist(h, 0.0);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            truck_route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) {
                base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
                base_route_dist[i] += distance_matrix_manhattan[r[k]][r[k+1]];
            }
        }
        for (int i = 0; i < d; ++i) {
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        for (int v = 0; v < h; ++v) {
            const auto& route = initial_solution.truck_routes[v];
            int m = (int)route.size();
            if (m <= 3) continue; // Not enough customers for 2-opt

            for (int i = 1; i < m - 2; ++i) {
                for (int j = i + 1; j < m -1; ++j) {
                    int u1 = route[i], v1 = route[i + 1];
                    int u2 = route[j], v2 = route[j + 1];

                    int ua = min(u1, v1), va = max(u1, v1);
                    int ub = min(u2, v2), vb = max(u2, v2);

                    bool is_tabu = (tabu_list_2opt[ua][va] > current_iter) ||
                                   (tabu_list_2opt[ub][vb] > current_iter);

                    // Calculate distance delta
                    double delta_dist = (distance_matrix_manhattan[u1][u2] + distance_matrix_manhattan[v1][v2])
                                      - (distance_matrix_manhattan[u1][v1] + distance_matrix_manhattan[u2][v2]);
                    double delta_time = delta_dist / vmax;

                    auto old_metrics = truck_route_metrics[v];
                    double new_route_time = old_metrics[0] + delta_time;

                    double old_route_viol = old_metrics[1];
                    double new_route_viol = max(0.0, (new_route_time - truck_limit) / truck_limit);
                    double total_deadline_viol = initial_solution.deadline_violation - old_route_viol + new_route_viol;

                    double total_cap_viol = initial_solution.capacity_violation; // No change
                    double total_energy_viol = initial_solution.energy_violation; // No change

                    double new_truck_dist = base_truck_dist + delta_dist;

                    double penalty = pow(1.0 + PENALTY_LAMBDA_DEADLINE * total_deadline_viol + PENALTY_LAMBDA_CAPACITY * total_cap_viol + PENALTY_LAMBDA_ENERGY * total_energy_viol, PENALTY_EXPONENT);
                    double cand_cost = (new_truck_dist * COST_TRUCK_KM + base_drone_dist * COST_DRONE_KM) * penalty;
                    if (is_tabu && !(cand_cost + 1e-8 < best_cost && total_deadline_viol < 1e-8 && total_cap_viol < 1e-8)) continue;
                    if (cand_cost < best_neighbor_cost_local) {
                        best_neighbor_cost_local = cand_cost;
                        best_edge_u = u1; best_edge_v = v1;
                        best_i = i; best_j = j;
                        best_route_idx = v;
                    }
                }
            }
        }

        // Apply the best move found
        if (best_edge_u != -1 && best_route_idx != -1) {
            best_neighbor = initial_solution;
            auto& route = best_neighbor.truck_routes[best_route_idx];
            reverse(route.begin() + best_i + 1, route.begin() + best_j + 1);
            best_neighbor = recalculate_solution(best_neighbor);
            if (solution_cost(best_neighbor) < best_neighbor_cost) {
                 best_neighbor_cost = solution_cost(best_neighbor);
                 tabu_list_2opt[min(best_edge_u, best_edge_v)][max(best_edge_u, best_edge_v)] = current_iter + TABU_TENURE_2OPT;
                 return best_neighbor;
            }
        }
        return initial_solution;

    } else if (neighbor_id == 4) {
        if ((int)tabu_list_2opt_star.size() != n + 1 || ((int)tabu_list_2opt_star.size() > 0 && (int)tabu_list_2opt_star[0].size() != n + 1)) {
            tabu_list_2opt_star.assign(n + 1, vector<int>(n + 1, 0));
        }

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_u = -1, best_v = -1; // Edge (u,v) being broken or formed for tabu
        int best_r1 = -1, best_r2 = -1;
        int best_cut1 = -1, best_cut2 = -1;

        // Precompute Prefix/Suffix info for all truck routes to enable O(1) check
        // We need: Cumulative Dist, Cumulative Load at each node
        struct RouteInfo {
            vector<double> dist_prefix;
            vector<double> load_prefix;
            double total_dist;
            double total_load;
        };
        vector<RouteInfo> r_info(h);
        vector<int> r_size(h);
        double current_total_truck_dist = 0;

        for (int i=0; i<h; ++i) {
            const vi& r = initial_solution.truck_routes[i];
            int m = r.size();
            r_size[i] = m;
            r_info[i].dist_prefix.assign(m, 0.0);
            r_info[i].load_prefix.assign(m, 0.0);
            
            double d = 0;
            double l = 0;
            for(int k=0; k<m; ++k) {
                if (k > 0) d += distance_matrix_manhattan[r[k-1]][r[k]];
                /* Assuming uniform demand or extracting demand if available. 
                   If demand is 1 per customer or looked up: */
                if (r[k] != 0) l += 1.0; // Replace with actual demand lookup if needed e.g. demand[r[k]]
                
                r_info[i].dist_prefix[k] = d;
                r_info[i].load_prefix[k] = l;
            }
            r_info[i].total_dist = d;
            r_info[i].total_load = l;
            current_total_truck_dist += d;
        }

        double base_drone_cost = 0; 
        for(int i=0; i<d; ++i) {
             // Calculate once to add to delta later
             const auto& r = initial_solution.drone_routes[i];
             for(int c : r) if(c!=0) base_drone_cost += distance_matrix_euclid[0][c]*2.0 * COST_DRONE_KM;
        }
        
        // Loop over pairs of truck routes
        for (int r1 = 0; r1 < h; ++r1) {
            const vi& route1 = initial_solution.truck_routes[r1];
            if (route1.size() < 2) continue;
            vd route1_feas = check_truck_route_feasibility(route1);

            for (int r2 = r1 + 1; r2 < h; ++r2) {
                const vi& route2 = initial_solution.truck_routes[r2];
                if (route2.size() < 2) continue;
                vd route2_feas = check_truck_route_feasibility(route2);

                // Try all cut positions
                // Cut i in route1 means: New route 1 ends at i, connects to route2[j+1]
                // Cut j in route2 means: New route 2 ends at j, connects to route1[i+1]
                // Valid cuts are before the final return to depot (index size-1)
                for (int i = 0; i < (int)route1.size() - 1; ++i) {
                    for (int j = 0; j < (int)route2.size() - 1; ++j) {
                        
                        int u1 = route1[i];
                        int v1 = route1[i+1]; // Old edge 1 broken
                        int u2 = route2[j];
                        int v2 = route2[j+1]; // Old edge 2 broken

                        // Tabu check on stored edges? 
                        // Usually tabu stores edges added/removed.
                        // We are adding (u1, v2) and (u2, v1)
                        bool is_tabu = (tabu_list_2opt_star[min(u1,v2)][max(u1,v2)] > current_iter) ||
                                       (tabu_list_2opt_star[min(u2,v1)][max(u2,v1)] > current_iter);

                        // 1. Capacity Check (O(1))
                        // R1 New Load = R1_Prefix(i) + (R2_Total - R2_Prefix(j))
                        double r1_new_load = r_info[r1].load_prefix[i] + (r_info[r2].total_load - r_info[r2].load_prefix[j]);
                        // R2 New Load = R2_Prefix(j) + (R1_Total - R1_Prefix(i))
                        double r2_new_load = r_info[r2].load_prefix[j] + (r_info[r1].total_load - r_info[r1].load_prefix[i]);
                        double new_capacity_violation = 0.0, new_energy_violation = 0.0, new_deadline_violation = 0.0;
                        double penalty = 0.0;
                        double capacity_delta = 0.0, energy_delta = 0.0, deadline_delta = 0.0;

                        if (r1_new_load > Dh) {
                            capacity_delta += (r1_new_load - Dh) / Dh;
                        }
                        if (r2_new_load > Dh) {
                            capacity_delta += (r2_new_load - Dh) / Dh;
                        }
                        capacity_delta = capacity_delta - route1_feas[3] - route2_feas[3];
                        new_capacity_violation = initial_solution.capacity_violation + capacity_delta;

                        // 2. Distance Delta (O(1))
                        // Removed: (u1->v1) + (u2->v2)
                        double dist_removed = distance_matrix_manhattan[u1][v1] + distance_matrix_manhattan[u2][v2];
                        // Added: (u1->v2) + (u2->v1)
                        double dist_added = distance_matrix_manhattan[u1][v2] + distance_matrix_manhattan[u2][v1];
                        
                        // Change in total distance
                        // However, we must account for the chunks being swapped.
                        // New Dist R1 = PrefixDist1(i) + (TotalDist2 - PrefixDist2(j+1)) + Dist(u1, v2) (wait, indices careful)
                        // Correct logic:
                        // Prefix1 ends at i (u1). Suffix1 starts at i+1 (v1).
                        // Prefix2 ends at j (u2). Suffix2 starts at j+1 (v2).
                        // New R1 = Prefix1 + Edge(u1, v2) + Suffix2
                        // New R2 = Prefix2 + Edge(u2, v1) + Suffix1
                        
                        double suffix2_dist = r_info[r2].total_dist - r_info[r2].dist_prefix[j+1]; // Dist from v2 to end
                        double r1_new_dist = r_info[r1].dist_prefix[i] + distance_matrix_manhattan[u1][v2] + suffix2_dist;
                        
                        double suffix1_dist = r_info[r1].total_dist - r_info[r1].dist_prefix[i+1]; // Dist from v1 to end
                        double r2_new_dist = r_info[r2].dist_prefix[j] + distance_matrix_manhattan[u2][v1] + suffix1_dist;

                        double new_total_truck_dist = current_total_truck_dist 
                                                    - r_info[r1].total_dist - r_info[r2].total_dist
                                                    + r1_new_dist + r2_new_dist;

                        // 3. Time Feasibility 
                        double r1_new_time = r1_new_dist / vmax;
                        double r2_new_time = r2_new_dist / vmax;
                        double r1_old_viol = route1_feas[1];
                        double r2_old_viol = route2_feas[1];
                        double r1_new_viol = max(0.0, (r1_new_time - truck_limit) / truck_limit);
                        double r2_new_viol = max(0.0, (r2_new_time - truck_limit) / truck_limit);
                        deadline_delta += (r1_new_viol - r1_old_viol) + (r2_new_viol - r2_old_viol);
                        new_deadline_violation = initial_solution.deadline_violation + deadline_delta;

                        new_energy_violation = initial_solution.energy_violation + energy_delta;
                        penalty = pow(1.0 
                            + PENALTY_LAMBDA_DEADLINE * max(0.0, new_deadline_violation)
                            + PENALTY_LAMBDA_CAPACITY * max(0.0, new_capacity_violation)
                            + PENALTY_LAMBDA_ENERGY * max(0.0, new_energy_violation),
                            PENALTY_EXPONENT);
                        double cand_cost = (new_total_truck_dist * COST_TRUCK_KM + base_drone_cost) * penalty;

                        // Aspiration
                        if (is_tabu && cand_cost >= best_cost) continue;

                        if (cand_cost < best_neighbor_cost_local) {
                            best_neighbor_cost_local = cand_cost;
                            best_r1 = r1; best_r2 = r2;
                            best_cut1 = i; best_cut2 = j;
                            best_u = u1; best_v = v2; // one of the new edges for tabu ref
                        }
                    }
                }
            }
        }

        if (best_r1 != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            // Reconstruct the solution only for the best move
            best_neighbor = initial_solution;
            vi& route1 = best_neighbor.truck_routes[best_r1];
            vi& route2 = best_neighbor.truck_routes[best_r2];
            
            vi new_r1, new_r2;
            
            // Prefix 1
            for(int k=0; k<=best_cut1; ++k) new_r1.push_back(route1[k]);
            // Suffix 2
            for(int k=best_cut2+1; k<(int)route2.size(); ++k) new_r1.push_back(route2[k]);

            // Prefix 2
            for(int k=0; k<=best_cut2; ++k) new_r2.push_back(route2[k]);
            // Suffix 1
            for(int k=best_cut1+1; k<(int)route1.size(); ++k) new_r2.push_back(route1[k]);

            best_neighbor.truck_routes[best_r1] = new_r1;
            best_neighbor.truck_routes[best_r2] = new_r2;
            
            best_neighbor = recalculate_solution(best_neighbor); // Only recalc once at end
            best_neighbor_cost = best_neighbor_cost_local;
            
            // Update Tabu
            // For 2-opt*, we can ban the edge we added (u1, v2)
            tabu_list_2opt_star[min(best_u, best_v)][max(best_u, best_v)] = current_iter + TABU_TENURE_2OPT_STAR;
            
            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 5) {

        // Neighborhood 5: 2-1 Swap (Swap pair (u,v) from A with single w from B)

        int best_r1 = -1, best_r2 = -1;
        int best_pos_pair = -1, best_pos_single = -1;
        double best_cost_local = 1e18;
        vector<int> best_triple; // For tabu update
        
        // 1. Pre-calculate metrics (Reuse N1 logic)
        vector<vd> metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        // Loop pairs of routes
        // We do directed pairs (A, B) where A provides the Pair, B provides the Single
        for (int rA = 0; rA < h + d; ++rA) {
            bool is_truck_a = rA < h;
            const vi& route_a = is_truck_a ? initial_solution.truck_routes[rA] : initial_solution.drone_routes[rA - h];
            if (route_a.size() < 3) continue; // Need at least 2 customers (+depots or just list)

            // Define iteration range for Pair (u, v) starting at i
            int max_i = is_truck_a ? (int)route_a.size() - 3 : (int)route_a.size() - 2;
            int start_i = is_truck_a ? 1 : 0;
            if (max_i < start_i) continue;

            for (int rB = 0; rB < h + d; ++rB) {
                if (rA == rB) {
                    // INTRA-ROUTE, FOR TRUCKS
                    if (!is_truck_a) continue; // Drones cannot do intra-route 2-1 swap
                    const vi& route = route_a;
                    for (int i = 1; i <= (int)route.size() - 3; ++i) {
                        int u = route[i];
                        int v = route[i+1];
                        if (u == 0 || v == 0) continue; // Skip depot
                        for (int j = i + 2; j <= (int)route.size() - 2; ++j) {
                            int w = route[j];
                            if (w == 0) continue; // Skip depot

                            vector<int> triple = {u, v, w};
                            sort(triple.begin(), triple.end());
                            bool is_tabu = (tabu_list_21.count(triple) && tabu_list_21[triple] > current_iter);

                            // --- Calculate Deltas ---

                            // Cost & Time A (Change in Truck Route)
                            double delta_dist_a = 0;
                            double delta_time_a = 0;
                            
                            int p1 = route[i-1];
                            int s1 = route[i+2]; // Successor of v (if j=i+2, this is w)
                            int p2 = route[j-1]; // Predecessor of w (if j=i+2, this is v)
                            int s2 = route[j+1]; // Successor of w

                            double current_val = 0;
                            double new_val = 0;

                            if (j == i + 2) {
                                // Consecutive: ... p1 -> u -> v -> w -> s2 ...
                                // Becomes:     ... p1 -> w -> u -> v -> s2 ...
                                // (u,v) edge is preserved, so we ignore it in delta
                                current_val = distance_matrix_manhattan[p1][u] + distance_matrix_manhattan[v][w] + distance_matrix_manhattan[w][s2];
                                new_val     = distance_matrix_manhattan[p1][w] + distance_matrix_manhattan[w][u] + distance_matrix_manhattan[v][s2];
                            } else {
                                // Disjoint:
                                // Spot i: ... p1 -> u -> v -> s1 ...  ==>  ... p1 -> w -> s1 ...
                                double rem_i = distance_matrix_manhattan[p1][u] + distance_matrix_manhattan[u][v] + distance_matrix_manhattan[v][s1];
                                double add_i = distance_matrix_manhattan[p1][w] + distance_matrix_manhattan[w][s1];

                                // Spot j: ... p2 -> w -> s2 ...  ==>  ... p2 -> u -> v -> s2 ...
                                double rem_j = distance_matrix_manhattan[p2][w] + distance_matrix_manhattan[w][s2];
                                double add_j = distance_matrix_manhattan[p2][u] + distance_matrix_manhattan[u][v] + distance_matrix_manhattan[v][s2];

                                current_val = rem_i + rem_j;
                                new_val     = add_i + add_j;
                            }

                            delta_dist_a = (new_val - current_val);
                            delta_time_a = delta_dist_a / vmax;

                            double new_total_truck_dist = base_truck_dist + delta_dist_a;
                            double new_total_drone_dist = base_drone_dist;

                            // --- Violations ---

                            double delta_cap = 0; // No capacity change in intra-route swap

                            double delta_dead = 0;
                            double lim_a = truck_limit;
                            double new_t_a = metrics[rA][0] + delta_time_a;
                            delta_dead += max(0.0, (new_t_a - lim_a)/lim_a) - max(0.0, (metrics[rA][0] - lim_a)/lim_a);

                            double new_cap_violation = initial_solution.capacity_violation + delta_cap;
                            double new_energy_violation = initial_solution.energy_violation;
                            double new_deadline_violation = initial_solution.deadline_violation + delta_dead;

                            double penalty = pow(1.0 
                                + PENALTY_LAMBDA_CAPACITY * max(0.0, new_cap_violation)
                                + PENALTY_LAMBDA_ENERGY * max(0.0, new_energy_violation)
                                + PENALTY_LAMBDA_DEADLINE * max(0.0, new_deadline_violation),
                                PENALTY_EXPONENT);
                            
                            double new_score = (new_total_truck_dist * COST_TRUCK_KM + new_total_drone_dist * COST_DRONE_KM) * penalty;
                            
                            if (is_tabu && !(new_score + 1e-8 < best_cost && new_deadline_violation < 1e-8 && new_cap_violation < 1e-8)) {
                                continue;
                            }
                            if (new_score < best_cost_local) {
                                best_cost_local = new_score;
                                best_r1 = rA;
                                best_r2 = rA;
                                best_pos_pair = i;
                                best_pos_single = j;
                                best_triple = triple;
                            }
                        }
                    }
                }
                if (rA == rB) continue; // Already handled
                bool is_truck_b = rB < h;
                const vi& route_b = is_truck_b ? initial_solution.truck_routes[rB] : initial_solution.drone_routes[rB - h];
                if (route_b.empty() || (is_truck_b && route_b.size() < 3)) continue; 

                int start_j = is_truck_b ? 1 : 0;
                int end_j = is_truck_b ? (int)route_b.size() - 2 : (int)route_b.size() - 1;

                for (int i = start_i; i <= max_i; ++i) {
                    int u = route_a[i];
                    int v = route_a[i+1];
                    if (u == 0 || v == 0) continue; 
                    if (!is_truck_b && (!served_by_drone[u] || !served_by_drone[v])) continue;

                    for (int j = start_j; j <= end_j; ++j) {
                        int w = route_b[j];
                        if (w == 0) continue;
                        if (!is_truck_a && !served_by_drone[w]) continue;

                        vector<int> triple = {u, v, w};
                        sort(triple.begin(), triple.end());
                        bool is_tabu = (tabu_list_21.count(triple) && tabu_list_21[triple] > current_iter);

                        // --- Calculate Deltas ---
                        
                        // Cost & Time A (Remove u,v Add w)
                        double delta_dist_a = 0;
                        double delta_time_a = 0;
                        if (is_truck_a) {
                            int prev = route_a[i-1];
                            int next = route_a[i+2];
                            double removed = distance_matrix_manhattan[prev][u] + distance_matrix_manhattan[u][v] + distance_matrix_manhattan[v][next];
                            double added = distance_matrix_manhattan[prev][w] + distance_matrix_manhattan[w][next];
                            delta_dist_a = (added - removed);
                            delta_time_a = (added - removed) / vmax;
                        } else {
                            delta_dist_a = (2.0 * distance_matrix_euclid[0][w] - 2.0 * distance_matrix_euclid[0][u] - 2.0 * distance_matrix_euclid[0][v]);
                            delta_time_a = delta_dist_a / v_fly_drone;
                        }

                        // Cost & Time B (Remove w Add u,v)
                        double delta_dist_b = 0;
                        double delta_time_b = 0;
                        if (is_truck_b) {
                            int prev = route_b[j-1];
                            int next = route_b[j+1];
                            double removed = distance_matrix_manhattan[prev][w] + distance_matrix_manhattan[w][next];
                            double added = distance_matrix_manhattan[prev][u] + distance_matrix_manhattan[u][v] + distance_matrix_manhattan[v][next];
                            delta_dist_b = (added - removed);
                            delta_time_b = (added - removed) / vmax;
                        } else {
                            delta_dist_b = (2.0 * distance_matrix_euclid[0][u] + 2.0 * distance_matrix_euclid[0][v] - 2.0 * distance_matrix_euclid[0][w]);
                            delta_time_b = delta_dist_b / v_fly_drone;
                        }

                        double new_total_truck_dist = base_truck_dist + (is_truck_a ? delta_dist_a : 0) + (is_truck_b ? delta_dist_b : 0);
                        double new_total_drone_dist = base_drone_dist + (!is_truck_a ? delta_dist_a : 0) + (!is_truck_b ? delta_dist_b : 0);

                        // --- Violations ---
                        double dem_change_a = demand[w] - (demand[u] + demand[v]);
                        double dem_change_b = (demand[u] + demand[v]) - demand[w];
                        
                        double delta_cap = 0;
                        if (is_truck_a) {
                             double load = initial_solution.truck_route_cap[rA];
                             delta_cap += max(0.0, (load + dem_change_a - Dh)/Dh) - max(0.0, (load - Dh)/Dh);
                        } else {
                             delta_cap += max(0.0, (demand[w]-Dd)/Dd) - max(0.0, (demand[u]-Dd)/Dd) - max(0.0, (demand[v]-Dd)/Dd);
                        }
                         if (is_truck_b) {
                             double load = initial_solution.truck_route_cap[rB];
                             delta_cap += max(0.0, (load + dem_change_b - Dh)/Dh) - max(0.0, (load - Dh)/Dh);
                        } else {
                             delta_cap += max(0.0, (demand[u]-Dd)/Dd) + max(0.0, (demand[v]-Dd)/Dd) - max(0.0, (demand[w]-Dd)/Dd);
                        }
                        
                        double delta_dead = 0;
                        double lim_a = is_truck_a ? truck_limit : drone_limit;
                        double new_t_a = metrics[rA][0] + delta_time_a;
                        delta_dead += max(0.0, (new_t_a - lim_a)/lim_a) - metrics[rA][1];

                        double lim_b = is_truck_b ? truck_limit : drone_limit;
                        double new_t_b = metrics[rB][0] + delta_time_b;
                        delta_dead += max(0.0, (new_t_b - lim_b)/lim_b) - metrics[rB][1];
                        
                        double delta_energy = 0;
                        if (!is_truck_a) {
                             double e_u = max(0.0, (2*distance_matrix_euclid[0][u]/v_fly_drone - E)/E);
                             double e_v = max(0.0, (2*distance_matrix_euclid[0][v]/v_fly_drone - E)/E);
                             double e_w = max(0.0, (2*distance_matrix_euclid[0][w]/v_fly_drone - E)/E);
                             delta_energy += (e_w - e_u - e_v);
                        }
                        if (!is_truck_b) {
                             double e_u = max(0.0, (2*distance_matrix_euclid[0][u]/v_fly_drone - E)/E);
                             double e_v = max(0.0, (2*distance_matrix_euclid[0][v]/v_fly_drone - E)/E);
                             double e_w = max(0.0, (2*distance_matrix_euclid[0][w]/v_fly_drone - E)/E);
                             delta_energy += (e_u + e_v - e_w);
                        }

                        double total_dead = initial_solution.deadline_violation + delta_dead;
                        double total_cap = initial_solution.capacity_violation + delta_cap;
                        double total_en = initial_solution.energy_violation + delta_energy;
                        
                        double penalty = pow(1.0 + PENALTY_LAMBDA_DEADLINE*total_dead + PENALTY_LAMBDA_CAPACITY*total_cap + PENALTY_LAMBDA_ENERGY*total_en, PENALTY_EXPONENT);
                        double score = (new_total_truck_dist * COST_TRUCK_KM + new_total_drone_dist * COST_DRONE_KM) * penalty;
                        
                        bool is_feas = (total_dead < 1e-8 && total_cap < 1e-8 && total_en < 1e-8);

                        if (is_tabu && !(score < best_cost && is_feas)) continue; 
                        
                        if (score < best_cost_local) {
                            best_cost_local = score;
                            best_r1 = rA; best_r2 = rB;
                            best_pos_pair = i; best_pos_single = j;
                            best_triple = triple;
                        }
                    }
                }
            } 
        } 

        if (best_r1 != -1) {
            best_neighbor = initial_solution;
            
            if (best_r1 != best_r2) {
                // Extract references
                bool is_truck_1 = best_r1 < h;
                vi& r1 = is_truck_1 ? best_neighbor.truck_routes[best_r1] : best_neighbor.drone_routes[best_r1 - h];
                bool is_truck_2 = best_r2 < h;
                vi& r2 = is_truck_2 ? best_neighbor.truck_routes[best_r2] : best_neighbor.drone_routes[best_r2 - h];

                int u = r1[best_pos_pair];
                int v = r1[best_pos_pair+1];
                int w = r2[best_pos_single];

                // Perform Swap
                // R1: replace u,v with w
                r1[best_pos_pair] = w;
                r1.erase(r1.begin() + best_pos_pair + 1);

                // R2: replace w with u,v
                r2[best_pos_single] = u;
                r2.insert(r2.begin() + best_pos_single + 1, v);
            }
            else {
                // Intra-route case for trucks
                vi& r = best_neighbor.truck_routes[best_r1];
                int u = r[best_pos_pair];
                int v = r[best_pos_pair+1];
                int w = r[best_pos_single];

                // Reconstruct new route
                vi new_route;
                for (int k = 0; k < (int)r.size(); ++k) {
                    if (k == best_pos_pair) {
                        new_route.push_back(w); // w
                    } else if (k == best_pos_single) {
                        new_route.push_back(u); // u
                        new_route.push_back(v); // v
                    } else if (k != best_pos_pair + 1) { // skip v at pos_pair+1
                        new_route.push_back(r[k]);
                    }
                }
                best_neighbor.truck_routes[best_r1] = new_route;
            }

            best_neighbor = recalculate_solution(best_neighbor);

            if (solution_cost(best_neighbor) < best_neighbor_cost) {
                best_neighbor_cost = solution_cost(best_neighbor);
                tabu_list_21[best_triple] = current_iter + TABU_TENURE_21;
                return best_neighbor;
            }
        }
        return initial_solution;
    } else if (neighbor_id == 6) {
       // Neighborhood 6: 2-2 Swap (Swap pair (u1,u2) from A with pair (v1,v2) from B)

        int best_r1 = -1, best_r2 = -1;
        int best_pos_1 = -1, best_pos_2 = -1;
        double best_cost_local = 1e18;
        vector<int> best_triple; // For tabu update (stores u1, u2, v1, v2 sorted)
        
        // 1. Pre-calculate metrics (Reuse logic)
        vector<vd> metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        // Loop pairs of routes
        for (int rA = 0; rA < h + d; ++rA) {
            bool is_truck_a = rA < h;
            const vi& route_a = is_truck_a ? initial_solution.truck_routes[rA] : initial_solution.drone_routes[rA - h];
            if (route_a.size() < (is_truck_a ? 4 : 2)) continue; // Need at least Pair

            // Define iteration range for Pair 1 (u1, u2) starting at i
            int max_i = is_truck_a ? (int)route_a.size() - 3 : (int)route_a.size() - 2;
            int start_i = is_truck_a ? 1 : 0;
            if (max_i < start_i) continue;

            for (int rB = 0; rB < h + d; ++rB) {
                // Determine limits for rB
                bool is_truck_b = rB < h;
                const vi& route_b = is_truck_b ? initial_solution.truck_routes[rB] : initial_solution.drone_routes[rB - h];
                if (route_b.empty() || route_b.size() < (is_truck_b ? 4 : 2)) continue;

                int max_j = is_truck_b ? (int)route_b.size() - 3 : (int)route_b.size() - 2;
                int start_j = is_truck_b ? 1 : 0;
                
                if (rA == rB) {
                    // INTRA-ROUTE
                    if (!is_truck_a) continue; // Skip drone intra-route (no cost change)
                    
                    for (int i = start_i; i <= max_i; ++i) {
                        int u1 = route_a[i];
                        int u2 = route_a[i+1];
                        if (u1 == 0 || u2 == 0) continue;

                        // rB is rA, so j must be disjoint (and > i). 
                        // Start j at i + 2 to avoid overlap (u1, u2) vs (v1, v2)
                        for (int j = i + 2; j <= max_j; ++j) {
                            int v1 = route_a[j];
                            int v2 = route_a[j+1];
                            if (v1 == 0 || v2 == 0) continue;

                            vector<int> quad = {u1, u2, v1, v2};
                            sort(quad.begin(), quad.end());
                            bool is_tabu = (tabu_list_22.count(quad) && tabu_list_22[quad] > current_iter);

                            // --- Delta Calc (Intra-Truck) ---
                            double delta_dist_a = 0;
                            double delta_time_a = 0;
                            
                            int p1 = route_a[i-1];
                            int s1 = route_a[i+2]; // Next after u2 (could be v1 if j=i+2)
                            int p2 = route_a[j-1]; // Prev before v1 (could be u2 if j=i+2)
                            int s2 = route_a[j+2]; // Next after v2

                            double current_dist = 0;
                            double new_dist = 0;

                            if (j == i + 2) {
                                // Consecutive: p1 -> u1 -> u2 -> v1 -> v2 -> s2
                                // Swap:        p1 -> v1 -> v2 -> u1 -> u2 -> s2
                                current_dist = distance_matrix_manhattan[p1][u1] + distance_matrix_manhattan[u2][v1] + distance_matrix_manhattan[v2][s2];
                                new_dist     = distance_matrix_manhattan[p1][v1] + distance_matrix_manhattan[v2][u1] + distance_matrix_manhattan[u2][s2];
                            } else {
                                // Disjoint
                                // At i: p1 -> u1 -> u2 -> s1  ==>  p1 -> v1 -> v2 -> s1
                                current_dist += distance_matrix_manhattan[p1][u1] + distance_matrix_manhattan[u2][s1];
                                new_dist     += distance_matrix_manhattan[p1][v1] + distance_matrix_manhattan[v2][s1];
                                // At j: p2 -> v1 -> v2 -> s2  ==>  p2 -> u1 -> u2 -> s2
                                current_dist += distance_matrix_manhattan[p2][v1] + distance_matrix_manhattan[v2][s2];
                                new_dist     += distance_matrix_manhattan[p2][u1] + distance_matrix_manhattan[u2][s2];
                            }
                            
                            delta_dist_a = new_dist - current_dist;
                            delta_time_a = delta_dist_a / vmax;

                            double new_total_truck_dist = base_truck_dist + delta_dist_a;
                            double new_total_drone_dist = base_drone_dist;

                            // -- Violations --
                            // Intra-swap capacity check (path dependent)
                            double new_route_cap_violation = 0;
                            {
                                // Simulate new route order for capacity
                                double load = 0;
                                for(int k=0; k<(int)route_a.size(); ++k) {
                                    int node = 0;
                                    if (k == i) node = v1;
                                    else if (k == i+1) node = v2;
                                    else if (k == j) node = u1;
                                    else if (k == j+1) node = u2;
                                    else node = route_a[k];

                                    if (node == 0) load = 0;
                                    else {
                                        load += demand[node];
                                        if (load > Dh) new_route_cap_violation += (load - Dh) / Dh;
                                    }
                                }
                            }
                            double delta_dead = 0;
                            double new_t_a = metrics[rA][0] + delta_time_a;
                            delta_dead = max(0.0, (new_t_a - truck_limit)/truck_limit) - metrics[rA][1];

                            double new_cap_violation = initial_solution.capacity_violation - metrics[rA][3] + new_route_cap_violation;
                            double new_energy_violation = initial_solution.energy_violation; // Unchanged for truck
                            double new_deadline_violation = initial_solution.deadline_violation + delta_dead;

                            double penalty = pow(1.0 
                                + PENALTY_LAMBDA_CAPACITY * max(0.0, new_cap_violation)
                                + PENALTY_LAMBDA_ENERGY * max(0.0, new_energy_violation)
                                + PENALTY_LAMBDA_DEADLINE * max(0.0, new_deadline_violation),
                                PENALTY_EXPONENT);
                            
                            double score = (new_total_truck_dist * COST_TRUCK_KM + new_total_drone_dist * COST_DRONE_KM) * penalty;
                            bool is_feas = (new_deadline_violation < 1e-8 && new_cap_violation < 1e-8 && new_energy_violation < 1e-8);
                            if (is_tabu && !(score < best_cost && is_feas)) continue;

                            if (score < best_cost_local) {
                                best_cost_local = score;
                                best_r1 = rA; best_r2 = rA;
                                best_pos_1 = i; best_pos_2 = j;
                                best_triple = quad;
                            }
                        }
                    }

                } else {
                    // INTER-ROUTE
                    if (rA > rB) continue; // Avoid double counting pairs (undirected pair of routes)
                    
                    for (int i = start_i; i <= max_i; ++i) {
                        int u1 = route_a[i];
                        int u2 = route_a[i+1];
                        if (u1 == 0 || u2 == 0) continue;
                        if (!is_truck_b && (!served_by_drone[u1] || !served_by_drone[u2])) continue;

                        for (int j = start_j; j <= max_j; ++j) {
                            int v1 = route_b[j];
                            int v2 = route_b[j+1];
                            if (v1 == 0 || v2 == 0) continue;
                            if (!is_truck_a && (!served_by_drone[v1] || !served_by_drone[v2])) continue;

                            vector<int> quad = {u1, u2, v1, v2};
                            sort(quad.begin(), quad.end());
                            bool is_tabu = (tabu_list_22.count(quad) && tabu_list_22[quad] > current_iter);

                            // --- Delta Calc ---
                            double delta_dist_a = 0, delta_time_a = 0;
                            double delta_dist_b = 0, delta_time_b = 0;

                            // Route A: Remove u1,u2. Add v1,v2.
                            if (is_truck_a) {
                                int p = route_a[i-1];
                                int s = route_a[i+2];
                                double rem = distance_matrix_manhattan[p][u1] + distance_matrix_manhattan[u1][u2] + distance_matrix_manhattan[u2][s];
                                double add = distance_matrix_manhattan[p][v1] + distance_matrix_manhattan[v1][v2] + distance_matrix_manhattan[v2][s];
                                delta_dist_a = add - rem;
                                delta_time_a = delta_dist_a / vmax;
                            } else {
                                // Drone: sorties
                                double rem = 2.0 * distance_matrix_euclid[0][u1] + 2.0 * distance_matrix_euclid[0][u2];
                                double add = 2.0 * distance_matrix_euclid[0][v1] + 2.0 * distance_matrix_euclid[0][v2];
                                delta_dist_a = add - rem;
                                delta_time_a = delta_dist_a / v_fly_drone;
                            }

                            // Route B: Remove v1,v2. Add u1,u2.
                            if (is_truck_b) {
                                int p = route_b[j-1];
                                int s = route_b[j+2];
                                double rem = distance_matrix_manhattan[p][v1] + distance_matrix_manhattan[v1][v2] + distance_matrix_manhattan[v2][s];
                                double add = distance_matrix_manhattan[p][u1] + distance_matrix_manhattan[u1][u2] + distance_matrix_manhattan[u2][s];
                                delta_dist_b = add - rem;
                                delta_time_b = delta_dist_b / vmax;
                            } else {
                                double rem = 2.0 * distance_matrix_euclid[0][v1] + 2.0 * distance_matrix_euclid[0][v2];
                                double add = 2.0 * distance_matrix_euclid[0][u1] + 2.0 * distance_matrix_euclid[0][u2];
                                delta_dist_b = add - rem;
                                delta_time_b = delta_dist_b / v_fly_drone;
                            }

                            double new_total_truck_dist = base_truck_dist + (is_truck_a ? delta_dist_a : 0) + (is_truck_b ? delta_dist_b : 0);
                            double new_total_drone_dist = base_drone_dist + (!is_truck_a ? delta_dist_a : 0) + (!is_truck_b ? delta_dist_b : 0);

                            // Capacity
                            // If Truck->Drone: check Dd logic? (handled by feasibility usually, or explicitly here)
                            // Capacity violation strictly path dependent for trucks.
                            double delta_cap = 0;
                            // Truck A
                            double dem_change_a = (demand[v1] + demand[v2]) - (demand[u1] + demand[u2]);
                            if (is_truck_a) {
                                // Estimate: load + change
                                double load_a = initial_solution.truck_route_cap[rA]; // This is total load.
                                // Approximation for delta:
                                delta_cap += max(0.0, (load_a + dem_change_a - Dh)/Dh) - max(0.0, (load_a - Dh)/Dh); 
                            } else {
                                // Drone A: u1,u2 out, v1,v2 in.
                                // Check individual drone items
                                delta_cap += (max(0.0, (demand[v1]-Dd)/Dd) + max(0.0, (demand[v2]-Dd)/Dd)) 
                                           - (max(0.0, (demand[u1]-Dd)/Dd) + max(0.0, (demand[u2]-Dd)/Dd));
                            }
                            // Truck B
                            double dem_change_b = (demand[u1] + demand[u2]) - (demand[v1] + demand[v2]);
                            if (is_truck_b) {
                                double load_b = initial_solution.truck_route_cap[rB];
                                delta_cap += max(0.0, (load_b + dem_change_b - Dh)/Dh) - max(0.0, (load_b - Dh)/Dh);
                            } else {
                                delta_cap += (max(0.0, (demand[u1]-Dd)/Dd) + max(0.0, (demand[u2]-Dd)/Dd)) 
                                           - (max(0.0, (demand[v1]-Dd)/Dd) + max(0.0, (demand[v2]-Dd)/Dd));
                            }

                            // Deadline
                            double delta_dead = 0;
                            double t_a = metrics[rA][0] + delta_time_a;
                            double lim_a = is_truck_a ? truck_limit : drone_limit;
                            delta_dead += max(0.0, (t_a - lim_a)/lim_a) - metrics[rA][1];

                            double t_b = metrics[rB][0] + delta_time_b;
                            double lim_b = is_truck_b ? truck_limit : drone_limit;
                            delta_dead += max(0.0, (t_b - lim_b)/lim_b) - metrics[rB][1];

                            // Energy (Drone only)
                            double delta_en = 0;
                            auto calc_e = [&](int c) { return max(0.0, (2.0*distance_matrix_euclid[0][c]/v_fly_drone - E)/E); };
                            if (!is_truck_a) delta_en += (calc_e(v1) + calc_e(v2)) - (calc_e(u1) + calc_e(u2));
                            if (!is_truck_b) delta_en += (calc_e(u1) + calc_e(u2)) - (calc_e(v1) + calc_e(v2));

                            double new_cap_violation = initial_solution.capacity_violation + delta_cap;
                            double new_en_violation = initial_solution.energy_violation + delta_en;
                            double new_dead_violation = initial_solution.deadline_violation + delta_dead;

                            double penalty = pow(1.0 
                                + PENALTY_LAMBDA_CAPACITY * max(0.0, new_cap_violation)
                                + PENALTY_LAMBDA_ENERGY * max(0.0, new_en_violation)
                                + PENALTY_LAMBDA_DEADLINE * max(0.0, new_dead_violation),
                                PENALTY_EXPONENT);

                            double score = (new_total_truck_dist * COST_TRUCK_KM + new_total_drone_dist * COST_DRONE_KM) * penalty;

                            bool is_feas = (new_dead_violation < 1e-8 && new_cap_violation < 1e-8 && new_en_violation < 1e-8);
                            if (is_tabu && !(score < best_cost && is_feas)) continue;

                            if (score < best_cost_local) {
                                best_cost_local = score;
                                best_r1 = rA; best_r2 = rB;
                                best_pos_1 = i; best_pos_2 = j;
                                best_triple = quad;
                            }
                        }
                    }
                }
            }
        }

        if (best_r1 != -1) {
            best_neighbor = initial_solution;
            
            // Apply Move
            if (best_r1 != best_r2) {
                // Inter
                bool is_tr1 = best_r1 < h;
                vi& r1 = is_tr1 ? best_neighbor.truck_routes[best_r1] : best_neighbor.drone_routes[best_r1 - h];
                bool is_tr2 = best_r2 < h;
                vi& r2 = is_tr2 ? best_neighbor.truck_routes[best_r2] : best_neighbor.drone_routes[best_r2 - h];

                int u1 = r1[best_pos_1];
                int u2 = r1[best_pos_1+1];
                int v1 = r2[best_pos_2];
                int v2 = r2[best_pos_2+1];

                // Swap
                r1[best_pos_1] = v1;
                r1[best_pos_1+1] = v2;
                r2[best_pos_2] = u1;
                r2[best_pos_2+1] = u2;
            } else {
                // Intra (Truck only)
                vi& r = best_neighbor.truck_routes[best_r1];
                int i = best_pos_1;
                int j = best_pos_2;
                int u1 = r[i];
                int u2 = r[i+1];
                int v1 = r[j];
                int v2 = r[j+1];

                r[i] = v1;
                r[i+1] = v2;
                r[j] = u1;
                r[j+1] = u2;
            }

            best_neighbor = recalculate_solution(best_neighbor);
            tabu_list_22[best_triple] = current_iter + TABU_TENURE_22;
        }

        if (solution_cost(best_neighbor) < best_neighbor_cost) {
             return best_neighbor;
        }
        return initial_solution;

    }  else if (neighbor_id == 7) {
        // Neighborhood 7: Depth-2 Ejection Chain (u from A -> replaces v in B -> v inserts in C)
        // Optimized: Random Permutation Selection

        int best_rA = -1, best_rB = -1, best_rC = -1;
        int best_pos_u = -1, best_pos_v = -1, best_pos_k = -1;
        double best_cost_local = 1e18;
        vector<int> best_tabu_key; // {u, v}
        
        // 1. Pre-calculate metrics
        vector<vd> metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        const int NUM_RANDOM_TRIPLETS = 1; // Limit attempts
        int n_routes = h + d;
        if (n_routes < 3) return initial_solution;

        vi route_indices;
        for (int i = 0; i < n_routes; ++i){
            if (i < h) {
                if (initial_solution.truck_routes[i].size() <= 2) continue;
                route_indices.push_back(i);  // Truck
            }
            else {
                if (initial_solution.drone_routes[i - h].size() <= 1) continue;
                route_indices.push_back(i);  // Drone
            }
        }

        int filtered_n_routes = (int)route_indices.size();
        if (filtered_n_routes < 3) return initial_solution;

        for (int attempt = 0; attempt < NUM_RANDOM_TRIPLETS; ++attempt) {
            int rA = rand() % filtered_n_routes;
            int rB = rand() % filtered_n_routes;
            while(rB == rA) rB = rand() % filtered_n_routes;
            int rC = rand() % filtered_n_routes;
            while(rC == rA || rC == rB) rC = rand() % filtered_n_routes;
            rA = route_indices[rA];
            rB = route_indices[rB];
            rC = route_indices[rC];
            bool is_truck_a = rA < h;
            const vi& route_a = is_truck_a ? initial_solution.truck_routes[rA] : initial_solution.drone_routes[rA - h];

            bool is_truck_b = rB < h;
            const vi& route_b = is_truck_b ? initial_solution.truck_routes[rB] : initial_solution.drone_routes[rB - h];

            bool is_truck_c = rC < h;
            const vi& route_c = is_truck_c ? initial_solution.truck_routes[rC] : initial_solution.drone_routes[rC - h];

            // Loop u in A
            int max_i = is_truck_a ? (int)route_a.size() - 2 : (int)route_a.size() - 1;
            int start_i = is_truck_a ? 1 : 0;
            
            for (int i = start_i; i <= max_i; ++i) {
                int u = route_a[i];
                if (u == 0) continue;
                if (!is_truck_b && !served_by_drone[u]) continue; 

                // Loop v in B
                int max_j = is_truck_b ? (int)route_b.size() - 2 : (int)route_b.size() - 1;
                int start_j = is_truck_b ? 1 : 0;

                for (int j = start_j; j <= max_j; ++j) {
                    int v = route_b[j];
                    if (v == 0) continue;
                    if (!is_truck_c && !served_by_drone[v]) continue;

                    vector<int> tabu_key = {u, v};
                    sort(tabu_key.begin(), tabu_key.end());
                    bool is_tabu = (tabu_list_ejection.count(tabu_key) && tabu_list_ejection[tabu_key] > current_iter);

                    // Loop position k in C
                    int max_k = is_truck_c ? (int)route_c.size() - 2 : (int)route_c.size() - 1;
                    int start_k = is_truck_c ? 0 : -1; 

                    for (int k = start_k; k <= max_k; ++k) {
                        
                        // --- DELTA EVALUATION ---

                        // 1. Route A: Remove u at i
                        double delta_dist_a = 0;
                        if (is_truck_a) {
                            int p = route_a[i-1], s = route_a[i+1];
                            delta_dist_a = distance_matrix_manhattan[p][s] - distance_matrix_manhattan[p][u] - distance_matrix_manhattan[u][s];
                        } else {
                            delta_dist_a = -(2.0 * distance_matrix_euclid[0][u]);
                        }

                        // 2. Route B: Swap v with u at j
                        double delta_dist_b = 0;
                        if (is_truck_b) {
                            int p = route_b[j-1], s = route_b[j+1];
                            delta_dist_b = (distance_matrix_manhattan[p][u] + distance_matrix_manhattan[u][s]) - (distance_matrix_manhattan[p][v] + distance_matrix_manhattan[v][s]);
                        } else {
                            delta_dist_b = (2.0 * distance_matrix_euclid[0][u]) - (2.0 * distance_matrix_euclid[0][v]);
                        }

                        // 3. Route C: Insert v after k
                        double delta_dist_c = 0;
                        if (is_truck_c) {
                            int p = route_c[k];
                            int s = route_c[k+1];
                            delta_dist_c = (distance_matrix_manhattan[p][v] + distance_matrix_manhattan[v][s]) - distance_matrix_manhattan[p][s];
                        } else {
                            delta_dist_c = (2.0 * distance_matrix_euclid[0][v]);
                        }

                        double delta_time_a = is_truck_a ? delta_dist_a / vmax : delta_dist_a / v_fly_drone;
                        double delta_time_b = is_truck_b ? delta_dist_b / vmax : delta_dist_b / v_fly_drone;
                        double delta_time_c = is_truck_c ? delta_dist_c / vmax : delta_dist_c / v_fly_drone;

                        double new_tot_truck = base_truck_dist + (is_truck_a?delta_dist_a:0) + (is_truck_b?delta_dist_b:0) + (is_truck_c?delta_dist_c:0);
                        double new_tot_drone = base_drone_dist + (!is_truck_a?delta_dist_a:0) + (!is_truck_b?delta_dist_b:0) + (!is_truck_c?delta_dist_c:0);

                        // Capacity
                        double new_cap_A = 0, new_cap_B = 0, new_cap_C = 0;
                        
                        // A: Remove u
                        if (is_truck_a) {
                            double load = 0;
                            for(size_t idx=0; idx<route_a.size(); ++idx) {
                                if (idx == (size_t)i) continue; 
                                int node = route_a[idx];
                                if (node==0) load=0;
                                else { load+=demand[node]; if(load>Dh) new_cap_A += (load-Dh)/Dh; }
                            }
                        } else {
                            for(int c : route_a) if(c!=u) if((demand[c]-Dd)>0) new_cap_A += (demand[c]-Dd)/Dd;
                        }

                        // B: Swap v -> u
                        if (is_truck_b) {
                            double load = 0;
                            for(size_t idx=0; idx<route_b.size(); ++idx) {
                                int node = (idx == (size_t)j) ? u : route_b[idx];
                                if (node==0) load=0;
                                else { load+=demand[node]; if(load>Dh) new_cap_B += (load-Dh)/Dh; }
                            }
                        } else {
                            for(size_t idx=0; idx<route_b.size(); ++idx) {
                                int node = (idx == (size_t)j) ? u : route_b[idx];
                                if((demand[node]-Dd)>0) new_cap_B += (demand[node]-Dd)/Dd;
                            }
                        }

                        // C: Insert v
                        if (is_truck_c) {
                            double load = 0;
                            for(size_t idx=0; idx<route_c.size(); ++idx) {
                                int node = route_c[idx];
                                if (node==0) load=0;
                                else { load+=demand[node]; if(load>Dh) new_cap_C += (load-Dh)/Dh; }
                                if (idx == (size_t)k) { load+=demand[v]; if(load>Dh) new_cap_C += (load-Dh)/Dh; }
                            }
                        } else {
                                for(int c : route_c) if((demand[c]-Dd)>0) new_cap_C += (demand[c]-Dd)/Dd;
                                if((demand[v]-Dd)>0) new_cap_C += (demand[v]-Dd)/Dd;
                        }

                        double tot_cap_viol = initial_solution.capacity_violation 
                                            - metrics[rA][3] - metrics[rB][3] - metrics[rC][3]
                                            + new_cap_A + new_cap_B + new_cap_C;

                        // Deadline
                        auto calc_dead = [&](int rIdx, double dt, bool truck) {
                            double lim = truck ? truck_limit : drone_limit;
                            return max(0.0, (metrics[rIdx][0] + dt - lim)/lim);
                        };
                        double new_dead_A = calc_dead(rA, delta_time_a, is_truck_a);
                        double new_dead_B = calc_dead(rB, delta_time_b, is_truck_b);
                        double new_dead_C = calc_dead(rC, delta_time_c, is_truck_c);

                        double tot_dead_viol = initial_solution.deadline_violation
                                                - metrics[rA][1] - metrics[rB][1] - metrics[rC][1]
                                                + new_dead_A + new_dead_B + new_dead_C;

                        // Energy (Drone only)
                        double tot_en_viol = initial_solution.energy_violation;
                        auto en_cost = [&](int c) { return max(0.0, (2.0*distance_matrix_euclid[0][c]/v_fly_drone - E)/E); };
                        if (!is_truck_a) tot_en_viol -= en_cost(u);
                        if (!is_truck_b) tot_en_viol += en_cost(u) - en_cost(v);
                        if (!is_truck_c) tot_en_viol += en_cost(v);

                        // Score
                        double penalty = pow(1.0 
                            + PENALTY_LAMBDA_CAPACITY * max(0.0, tot_cap_viol)
                            + PENALTY_LAMBDA_ENERGY * max(0.0, tot_en_viol)
                            + PENALTY_LAMBDA_DEADLINE * max(0.0, tot_dead_viol),
                            PENALTY_EXPONENT);

                        double score = (new_tot_truck*COST_TRUCK_KM + new_tot_drone*COST_DRONE_KM) * penalty;

                        bool is_feas = (tot_dead_viol < 1e-8 && tot_cap_viol < 1e-8 && tot_en_viol < 1e-8);
                        if (is_tabu && !(score < best_cost && is_feas)) continue;

                        if (score < best_cost_local) {
                            best_cost_local = score;
                            best_rA = rA; best_rB = rB; best_rC = rC;
                            best_pos_u = i; best_pos_v = j; best_pos_k = k;
                            best_tabu_key = tabu_key;
                        }
                    }
                }
            }
        }

        if (best_rA != -1) {
            best_neighbor = initial_solution;
            // Apply Moves
            // 1. Remove u from A
            bool is_truck_a = best_rA < h;
            vi& ra = is_truck_a ? best_neighbor.truck_routes[best_rA] : best_neighbor.drone_routes[best_rA - h];
            int u = ra[best_pos_u];
            ra.erase(ra.begin() + best_pos_u);

            // 2. Swap v with u in B
            bool is_truck_b = best_rB < h;
            vi& rb = is_truck_b ? best_neighbor.truck_routes[best_rB] : best_neighbor.drone_routes[best_rB - h];
            int v = rb[best_pos_v];
            rb[best_pos_v] = u;

            // 3. Insert v into C
            bool is_truck_c = best_rC < h;
            vi& rc = is_truck_c ? best_neighbor.truck_routes[best_rC] : best_neighbor.drone_routes[best_rC - h];
            // Insert after best_pos_k
            if (is_truck_c) {
                rc.insert(rc.begin() + best_pos_k + 1, v);
            } else {
                rc.push_back(v); 
            }

            best_neighbor = recalculate_solution(best_neighbor);
            tabu_list_ejection[best_tabu_key] = current_iter + TABU_TENURE_EJECTION;
            
            if (solution_cost(best_neighbor) < best_neighbor_cost) {
                return best_neighbor;
            }
        }
        return initial_solution;
    } else if (neighbor_id == 8) {
        // Neighborhood 8: Greedy Truck -> Optimal Drone Offload
        // Strategy: Identify the drone with the smallest current travel time.
        // Attempt to move drone-eligible customers from trucks to ONLY that drone.
        // This balances load by filling the "idlest" drone.

        int best_r_truck = -1;
        int best_pos_u = -1;
        int best_u_val = -1;
        
        // Find optimal target drone (min time, and must have remaining budget)
        int target_drone_idx = -1;
        double min_drone_time = 1e18;
        
        vector<double> drone_times(d);
        for(int i=0; i<d; ++i) {
             const auto& r = initial_solution.drone_routes[i];
             // Simple time calc or full feasibility check? 
             // We can use check_drone_route_feasibility, but here raw sum of euclidean legs is enough for selection
             double t = 0;
             for(int c : r) t += (2.0 * distance_matrix_euclid[0][c] / v_fly_drone);
             drone_times[i] = t;
             
             if (t < drone_limit - 1e-6) { // Must have budget
                 if (t < min_drone_time) {
                     min_drone_time = t;
                     target_drone_idx = i;
                 }
             }
        }

        if (target_drone_idx == -1) return initial_solution; // No drone has time available

        int rD = h + target_drone_idx;
        double best_cost_local = 1e18;

        // 1. Pre-calculate metrics for all vehicles (needed for Delta)
        vector<vd> metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            // Use metrics[h+i][0] as accurate time if available, but our loop calc above is fine for selection
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
        }

        // Loop Truck Routes
        for (int rT = 0; rT < h; ++rT) {
            const vi& route_t = initial_solution.truck_routes[rT];
            // Skip depot
            for (int i = 1; i < (int)route_t.size() - 1; ++i) {
                int u = route_t[i];
                if (!served_by_drone[u]) continue; 

                // Tabu Check
                bool is_tabu = false;
                 if (!tabu_list_10.empty() && (int)tabu_list_10.size() > u && (int)tabu_list_10[u].size() > rD) {
                    is_tabu = (tabu_list_10[u][rD] > current_iter);
                }

                // --- Delta Evaluation ---

                // 1. Truck Removal Cost
                int p = route_t[i-1], s = route_t[i+1];
                double delta_dist_t = distance_matrix_manhattan[p][s] - distance_matrix_manhattan[p][u] - distance_matrix_manhattan[u][s];
                double delta_time_t = delta_dist_t / vmax;

                // 2. Drone Insertion Cost
                double delta_dist_d = 2.0 * distance_matrix_euclid[0][u];
                double delta_time_d = delta_dist_d / v_fly_drone;

                // Pre-check basic feasibility for speed: if drone exceeds limit, skip immediately
                if (metrics[rD][0] + delta_time_d > drone_limit) continue;

                double new_truck_dist = base_truck_dist + delta_dist_t;
                double new_drone_dist = base_drone_dist + delta_dist_d;

                // 3. Violations
                
                // Deadline
                double delta_dead = 0;
                // Truck time decreases, so deadline violation reduces (or stays 0)
                double t_t = metrics[rT][0] + delta_time_t;
                delta_dead += max(0.0, (t_t - truck_limit)/truck_limit) - metrics[rT][1];
                
                // Drone time increases
                double t_d = metrics[rD][0] + delta_time_d; 
                delta_dead += max(0.0, (t_d - drone_limit)/drone_limit) - metrics[rD][1];

                // Capacity (Truck reduced)
                double new_truck_cap_viol = 0;
                double load = 0;
                for (int k_idx=0; k_idx<(int)route_t.size(); ++k_idx) {
                    if (k_idx == i) continue; 
                    int c = route_t[k_idx];
                    if (c==0) load = 0;
                    else {
                        load += demand[c];
                        if (load > Dh) new_truck_cap_viol += (load - Dh)/Dh;
                    }
                }
                double delta_cap = (new_truck_cap_viol - metrics[rT][3]); 

                double new_cap_viol = initial_solution.capacity_violation + delta_cap;
                double new_dead_viol = initial_solution.deadline_violation + delta_dead;
                // Energy violation shouldn't change if u is feasible (served_by_drone check covers local u energy)

                double penalty = pow(1.0 
                    + PENALTY_LAMBDA_CAPACITY * max(0.0, new_cap_viol)
                    + PENALTY_LAMBDA_ENERGY * max(0.0, initial_solution.energy_violation)
                    + PENALTY_LAMBDA_DEADLINE * max(0.0, new_dead_viol),
                    PENALTY_EXPONENT);

                double score = (new_truck_dist * COST_TRUCK_KM + new_drone_dist * COST_DRONE_KM) * penalty;

                bool is_feas = (new_dead_viol < 1e-8 && new_cap_viol < 1e-8 && initial_solution.energy_violation < 1e-8);
                if (is_tabu && !(score < best_cost && is_feas)) continue;

                if (score < best_cost_local) {
                    best_cost_local = score;
                    best_r_truck = rT;
                    best_pos_u = i;
                    best_u_val = u;
                }
            }
        }

        if (best_r_truck != -1) {
            best_neighbor = initial_solution;
            
            // 1. Remove from Truck
            vi& rt = best_neighbor.truck_routes[best_r_truck];
            rt.erase(rt.begin() + best_pos_u);
            
            // 2. Add to Drone
            vi& rd = best_neighbor.drone_routes[target_drone_idx];
            rd.push_back(best_u_val);

            best_neighbor = recalculate_solution(best_neighbor);

            // Update Tabu
            if ((int)tabu_list_10.size() == n + 1) {
            tabu_list_10[best_u_val][rD] = current_iter + TABU_TENURE_10;
            }
            
            if (solution_cost(best_neighbor) < best_neighbor_cost) {
                return best_neighbor;
            }
        }
        return initial_solution;
    }
    return initial_solution;
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

// Common Repair Logic
Solution repair_solution_common(Solution sol, const unordered_set<int>& to_destroy) {
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    Solution new_sol = sol;
    
    // Remove destroyed customers
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
    new_sol = recalculate_solution(new_sol);
    
    vector<int> customers_to_insert(to_destroy.begin(), to_destroy.end());
    std::shuffle(customers_to_insert.begin(), customers_to_insert.end(), rng);
    
    for (int cust : customers_to_insert) {
        struct InsertionOption {
            int veh_idx;
            bool is_truck;
            int pos;
            double score;
            vi resulting_route;
            double route_time;
        };
        vector<InsertionOption> options;
        
        // 1. Evaluate all useable Truck positions
        for (int i = 0; i < h; ++i) {
            const vi& route = new_sol.truck_routes[i];
            // Safe loop limit
            for (int p = 1; p < (int)route.size(); ++p) { 
                vi temp_route = route;
                temp_route.insert(temp_route.begin() + p, cust);
                vd m = check_route_feasibility(temp_route, 0.0, true);
                
                double new_dist = 0;
                for(size_t k=0; k+1<temp_route.size(); ++k) new_dist += distance_matrix_manhattan[temp_route[k]][temp_route[k+1]];
                double current_dist = 0; 
                for(size_t k=0; k+1<route.size(); ++k) current_dist += distance_matrix_manhattan[route[k]][route[k+1]];
                double delta_cost = (new_dist - current_dist) * COST_TRUCK_KM;
                
                double penalties = PENALTY_LAMBDA_DEADLINE * m[1] + PENALTY_LAMBDA_ENERGY * m[2] + PENALTY_LAMBDA_CAPACITY * m[3];
                double score = delta_cost + pow(1.0 + penalties, PENALTY_EXPONENT) * 10000.0 * (penalties > 1e-6 ? 1.0 : 0.0);
                options.push_back({i, true, p, score, temp_route, m[0]});
            }
        }
        
        // 2. Evaluate Drone positions
        if (served_by_drone[cust]) {
             for (int i = 0; i < d; ++i) {
                const vi& route = new_sol.drone_routes[i];
                int p = (int)route.size(); 
                vi temp_route = route;
                temp_route.push_back(cust);
                vd m = check_route_feasibility(temp_route, 0.0, false);
                
                double new_dist = 0;
                for(int c : temp_route) if(c!=0) new_dist += distance_matrix_euclid[0][c]*2.0;
                double curr_dist = 0;
                for(int c : route) if(c!=0) curr_dist += distance_matrix_euclid[0][c]*2.0;
                double delta_cost = (new_dist - curr_dist) * COST_DRONE_KM;
                
                double penalties = PENALTY_LAMBDA_DEADLINE * m[1] + PENALTY_LAMBDA_ENERGY * m[2] + PENALTY_LAMBDA_CAPACITY * m[3];
                double score = delta_cost + pow(1.0 + penalties, PENALTY_EXPONENT) * 100.0 * (penalties > 1e-6 ? 1.0 : 0.0);
                options.push_back({i, false, p, score, temp_route, m[0]});
             }
        }
        
        if (options.empty()) {
             new_sol = greedy_insert_customer(new_sol, cust, true);
             continue;
        }

        double min_score = 1e18;
        for(const auto& opt : options) if(opt.score < min_score) min_score = opt.score;
        double beta = 0.5; 
        vector<double> weights;
        double total_weight = 0.0;
        for(const auto& opt : options) {
            double w = exp( -beta * (opt.score - min_score) );
            weights.push_back(w);
            total_weight += w;
        }
        
        double r = std::uniform_real_distribution<double>(0.0, total_weight)(rng);
        double cum = 0.0;
        int selected_idx = 0; 
        for(size_t k=0; k<weights.size(); ++k) {
            cum += weights[k];
            if (r <= cum) { selected_idx = k; break; }
        }
        
        const auto& choice = options[selected_idx];
        if (choice.is_truck) {
            new_sol.truck_routes[choice.veh_idx] = choice.resulting_route;
            new_sol.truck_route_times[choice.veh_idx] = choice.route_time;
        } else {
            new_sol.drone_routes[choice.veh_idx] = choice.resulting_route;
            new_sol.drone_route_times[choice.veh_idx] = choice.route_time;
        }
    }
    
    // Finalize
    for (int i = 0; i < h; ++i) {
        vi& route = new_sol.truck_routes[i];
        if (route.empty() || route.front() != 0) route.insert(route.begin(), 0);
        if (route.back() != 0) route.push_back(0);
        vd m = check_route_feasibility(route, 0.0, true);
        new_sol.truck_route_times[i] = m[0];
        double km = 0; 
        for(size_t k=0; k+1<route.size(); ++k) km += distance_matrix_manhattan[route[k]][route[k+1]];
        new_sol.total_distance_truck += km;
    }
    new_sol.total_distance_drone = 0;
    for (int i = 0; i < d; ++i) {
        vi& route = new_sol.drone_routes[i];
        if (!route.empty() && route.front() == 0) route.erase(route.begin()); 
        vd m = check_route_feasibility(route, 0.0, false);
        new_sol.drone_route_times[i] = m[0];
        for(int c : route) if(c!=0) new_sol.total_distance_drone += distance_matrix_euclid[0][c]*2.0; 
    }
    new_sol.deadline_violation = 0; 
    new_sol.capacity_violation = 0;
    new_sol.energy_violation = 0;
    new_sol.total_time = 0; 
    for(int i=0; i<h; ++i) {
         vd m = check_route_feasibility(new_sol.truck_routes[i], 0.0, true);
         new_sol.deadline_violation += m[1];
         new_sol.energy_violation += m[2];
         new_sol.capacity_violation += m[3];
         new_sol.total_time += m[0];
    } 
    for(int i=0; i<d; ++i) {
         vd m = check_route_feasibility(new_sol.drone_routes[i], 0.0, false);
         new_sol.deadline_violation += m[1];
         new_sol.energy_violation += m[2];
         new_sol.capacity_violation += m[3];
         new_sol.total_time += m[0];
    }
    return new_sol;
}

Solution destroy_worst_repair_random(Solution sol) {
    unordered_set<int> to_destroy;
    int destroy_count = static_cast<int>(n * 0.3); 
    Solution current_sol = sol;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    
    struct Candidate { int cust; int veh_idx; bool is_truck; double saving; };
    for (int k = 0; k < destroy_count; ++k) {
        vector<Candidate> candidates;
        candidates.reserve(n);
        for (int i = 0; i < h; ++i) {
            const vi& route = current_sol.truck_routes[i]; 
            if (route.size() <= 2) continue; 
            for (int p = 1; p < (int)route.size() - 1; ++p) { 
                int cust = route[p];
                if (cust == 0 || to_destroy.count(cust)) continue;
                int prev = route[p-1];
                int next = route[p+1];
                double saving = distance_matrix_manhattan[prev][cust] + distance_matrix_manhattan[cust][next] - distance_matrix_manhattan[prev][next];
                candidates.push_back({cust, i, true, saving * COST_TRUCK_KM});
            }
        }
        for (int i = 0; i < d; ++i) {
            const vi& route = current_sol.drone_routes[i];
            if (route.size() <= 1) continue; 
            for (int p = 1; p < (int)route.size(); ++p) { 
                int cust = route[p];
                if (cust == 0 || to_destroy.count(cust)) continue;
                double saving = distance_matrix_euclid[0][cust] * 2.0;
                candidates.push_back({cust, i, false, saving * COST_DRONE_KM});
            }
        }
        if (candidates.empty()) break;
        std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b){ return a.saving > b.saving; });
        int range = max(1, min((int)candidates.size(), 6)); 
        std::uniform_int_distribution<int> pick_dist(0, range - 1);
        const Candidate& selected = candidates[pick_dist(rng)];
        to_destroy.insert(selected.cust);
        if (selected.is_truck) {
             vi& r = current_sol.truck_routes[selected.veh_idx];
             r.erase(std::remove(r.begin(), r.end(), selected.cust), r.end());
        } else {
             vi& r = current_sol.drone_routes[selected.veh_idx];
             r.erase(std::remove(r.begin(), r.end(), selected.cust), r.end());
        }
    }
    return repair_solution_common(sol, to_destroy);
}

Solution destroy_shortest_route_repair_random(Solution sol) {
    unordered_set<int> to_destroy;
    std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    
    // Identify shortest non-empty truck route
    int target_idx = -1;
    size_t min_size = 100000;
    int active_trucks = 0;
    for (int i = 0; i < h; ++i) {
        if (sol.truck_routes[i].size() > 2) { 
             active_trucks++;
             if (sol.truck_routes[i].size() < min_size) {
                 min_size = sol.truck_routes[i].size();
                 target_idx = i;
             }
        }
    }
    
    if (active_trucks <= 1 || target_idx == -1) {
        return destroy_worst_repair_random(sol);
    }
    
    // Empty target route
    for (int c : sol.truck_routes[target_idx]) {
        if (c != 0) to_destroy.insert(c);
    }
    // Also destroy 15% random others
    int random_count = static_cast<int>(n * 0.15);
    std::uniform_int_distribution<int> dist_idx(1, n);
    for(int k=0; k<random_count; ++k){
        int c = dist_idx(rng);
        to_destroy.insert(c);
    }
    
    return repair_solution_common(sol, to_destroy);
}

// SISR (Slack Induction by Substring Removal) Implementation
Solution destroy_sisr_repair(Solution sol) {
    const int MAX_STRING_REMOVALS = 3; 
    const int MAX_STRING_SIZE_BASE = 12; 
    
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
    int removal_count = 0;

    // Prioritize center, then neighbors
    vector<int> process_queue;
    process_queue.push_back(center);
    process_queue.insert(process_queue.end(), candidate_neighbors.begin(), candidate_neighbors.end());

    for(int neighbor_cust : process_queue) {
        if(to_destroy.count(neighbor_cust)) continue; // Already marked
        if(removal_count >= MAX_STRING_REMOVALS) break;
        
        Locator l = cust_loc[neighbor_cust];
        if(l.v_idx == -1) continue; 
        
        // Identify unique vehicle ID (Trucks: 0..h-1, Drones: h..h+d-1)
        int unique_id = l.is_truck ? l.v_idx : (h + l.v_idx);
        if(destroyed_routes_id.count(unique_id)) continue;
        
        destroyed_routes_id.insert(unique_id);
        removal_count++;
        
        if(!l.is_truck) {
             // Remove all customers on this drone route
             for(int c : sol.drone_routes[l.v_idx]) if(c!=0) to_destroy.insert(c);
        } else {
             // Truck String Removal
             const vi& route = sol.truck_routes[l.v_idx];
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
                 to_destroy.insert(neighbor_cust);
             } else {
                 std::uniform_int_distribution<int> start_dist(min_s, max_s);
                 int s = start_dist(rng);
                 for(int k=0; k<str_len; ++k) {
                     int idx = s + k;
                     if (idx < route.size()) {
                        int c = route[idx];
                        if(c!=0) to_destroy.insert(c);
                     }
                 }
             }
        }
    }
    return repair_solution_common(sol, to_destroy);
}

Solution tabu_search(const Solution& initial_solution, vector<double>& iter_current, vector<double>& iter_best, vector<bool>& iter_feasible) {
    auto ts_start = std::chrono::high_resolution_clock::now();
    auto is_feasible = [](const Solution& sol) {
        return sol.deadline_violation <= 1e-8 &&
               sol.capacity_violation <= 1e-8 &&
               sol.energy_violation <= 1e-8;
    };
    Solution best_solution = initial_solution;
    Solution best_feasible_solution = initial_solution;
    bool initial_feasible = is_feasible(initial_solution);
    double best_feasible_cost = initial_feasible
        ? solution_score_cost(initial_solution)
        : std::numeric_limits<double>::infinity();
    double score[NUM_NEIGHBORHOODS] = {0.0};
    double weight[NUM_NEIGHBORHOODS];
    for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] = 1.0 / NUM_NEIGHBORHOODS;
    int count[NUM_NEIGHBORHOODS] = {0};

    Solution current_sol = initial_solution;
    double current_cost = solution_score_cost(current_sol);

    current_sol = initial_solution;
    iter_current.clear();
    iter_best.clear();
    iter_feasible.clear();
    // Unified Tabu Search on Cost
    int iter = 1;
    int total_iters = CFG_MAX_SEGMENT * CFG_MAX_ITER_PER_SEGMENT;
    int no_improve_iters = 0;

    cout << "=== Starting Unified Tabu Search (Minimizing Weighted Cost) ===\n";
    cout << "Initial Cost: " << solution_score_cost(current_sol) << "\n";

    double best_solution_score_now = solution_score_cost(best_solution);

    while (iter <= total_iters) {
        if (CFG_TIME_LIMIT_SEC > 0.0) {
            double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
            if (elapsed >= CFG_TIME_LIMIT_SEC) break;
        }

        double current_score = solution_score_cost(current_sol);
        double current_pure_cost = solution_pure_cost(current_sol);
        iter_current.push_back(current_pure_cost);;
        iter_best.push_back(best_feasible_cost);
        iter_feasible.push_back(is_feasible(current_sol));


        // Roulette Wheel Selection
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
        //if (selected_neighbor > 7) selected_neighbor = 7; // For debugging
        count[selected_neighbor]++;

        // Local Search - Always use all_vehicle with Cost
        Solution init_neighbor;
        Solution neighbor;
        try {
            init_neighbor = local_search_all_vehicle(current_sol, selected_neighbor, iter, best_solution_score_now, solution_score_cost);
            neighbor = recalculate_solution(init_neighbor);
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

        bool integrity_ok = true;
        
        if (std::abs(neighbor.deadline_violation - init_neighbor.deadline_violation) > 1e-8 ||
            std::abs(neighbor.capacity_violation - init_neighbor.capacity_violation) > 1e-8 ||
            std::abs(neighbor.energy_violation - init_neighbor.energy_violation) > 1e-8 ||
            std::abs(neighbor.total_makespan - init_neighbor.total_makespan) > 1e-8) {
             cout << "Iter " << iter << " Recalc deviation detected. Skipping.\n";
             integrity_ok = false;
        } 

        if (integrity_ok && !check_solution_integrity(neighbor)) {
            cout << "Iter " << iter << " Integrity Check Failed. Skipping.\n";
             integrity_ok = false;
        }

        if (!integrity_ok) {
            neighbor = current_sol;
        }

        bool neighbor_feasible = is_feasible(neighbor);
        double neighbor_score = solution_score_cost(neighbor);

        // Acceptance
        if (neighbor_score + 1e-12 < best_solution_score_now || 
           (std::abs(neighbor_score - best_solution_score_now) <= 1e-12 && neighbor.total_time < best_solution.total_time)) {
            
            current_sol = neighbor;
            best_solution = neighbor;
            best_solution_score_now = neighbor_score;
            score[selected_neighbor] += gamma1;
            no_improve_iters = 0;
            
        } else if (neighbor_score + 1e-12 < current_score) {
            current_sol = neighbor;
            score[selected_neighbor] += gamma2;
        } else {
            score[selected_neighbor] += gamma3;
            no_improve_iters++;
            if (selected_neighbor == 8) current_sol = neighbor; // Always accept Greedy Truck -> Optimal Drone
        }

        // Update Feasible Best
        if (neighbor_feasible) {
             double n_cost = solution_score_cost(neighbor);
             if (n_cost + 1e-12 < best_feasible_cost) {
                 best_feasible_solution = neighbor;
                 best_feasible_cost = n_cost;
                 cout << "Iter " << iter << " New Best Feasible Cost: " << best_feasible_cost << "\n";
             }
        }
        
        update_penalties(current_sol);

        // Perturbation (Destroy/Repair)
        if (no_improve_iters >= CFG_MAX_NO_IMPROVE) {
            // Destroy and repair current solution or starting over from best solution with a few force N7 iterations
            // Chance to restart from best solution:
            double restart_chance = 0.0;
            if (((double) rand() / (RAND_MAX)) < restart_chance) {
                current_sol = best_solution;
                int n7_iters = max(10, int(sqrt(n)));
                for (int i = 0; i < n7_iters; ++i) {
                    current_sol = local_search_all_vehicle(current_sol, 7, iter, best_solution_score_now, solution_score_cost);
                    current_sol = recalculate_solution(current_sol);
                }
            }
            else {
                current_sol = destroy_sisr_repair(current_sol);
            }
            
            current_sol = recalculate_solution(current_sol);
            no_improve_iters = 0;
            
            // Clear Tabu Lists
            tabu_list_10.clear();
            tabu_list_11.clear();
            tabu_list_20.clear();
            tabu_list_2opt.clear();
            tabu_list_2opt_star.clear();
            tabu_list_22.clear();
            tabu_list_21.clear();
            tabu_list_ejection.clear();
        }

        // Periodic Weight Update
        if (iter % CFG_MAX_ITER_PER_SEGMENT == 0) {
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                if (count[i] != 0) {
                    weight[i] = (1.0 - gamma4) * weight[i] + gamma4 * (score[i] / count[i]);
                }
            }
            double sum_weights = 0.0;
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) sum_weights += weight[i];
            if (sum_weights > 0.0) {
                for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] /= sum_weights;
            } else {
                 for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] = 1.0 / NUM_NEIGHBORHOODS;
            }
            for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                score[i] = 0.0;
                count[i] = 0;
            }
        }

        iter++;
    }
    if (best_feasible_cost < std::numeric_limits<double>::infinity()) {
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

void check_benchmark_solution(const std::string& benchmark_string) {
    std::cout << "\n--- Checking Benchmark Solution ---\n";
    std::stringstream ss(benchmark_string);
    std::string line;
    Solution sol;

    // Dynamically resize routes as we parse, instead of pre-assigning.
    sol.truck_routes.clear();
    sol.drone_routes.clear();

    while (std::getline(ss, line)) {
        std::stringstream line_ss(line);
        std::string type;
        line_ss >> type;

        if (type.empty() || (type[0] != 'T' && type[0] != 'D')) continue;

        int vehicle_id = std::stoi(type.substr(1));
        double route_time; // This is the time from the benchmark file, we'll ignore it and recalculate
        line_ss >> route_time;

        std::vector<int> route;
        int customer_id;
        while (line_ss >> customer_id) {
            route.push_back(customer_id);
        }

        if (type[0] == 'T') {
            if (vehicle_id >= sol.truck_routes.size()) {
                sol.truck_routes.resize(vehicle_id + 1);
            }
            sol.truck_routes[vehicle_id] = route;
        } else { // Drone
            if (vehicle_id >= sol.drone_routes.size()) {
                sol.drone_routes.resize(vehicle_id + 1);
            }
            sol.drone_routes[vehicle_id] = route;
        }
    }

    // After parsing, we need to initialize the other solution vectors based on the new route sizes
    sol.truck_route_times.assign(sol.truck_routes.size(), 0.0);
    sol.truck_route_cap.assign(sol.truck_routes.size(), 0.0);
    sol.drone_route_times.assign(sol.drone_routes.size(), 0.0);
    sol.drone_route_cap.assign(sol.drone_routes.size(), 0.0);


    // Recalculate all metrics for the parsed solution
    Solution recalculated_sol = recalculate_solution(sol);

    std::cout << "Parsed and Recalculated Benchmark Solution:\n";
    print_solution_stream(recalculated_sol, std::cout);
    std::cout << "--- Finished Checking Benchmark Solution ---\n\n";
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

static bool write_output_file(const std::string& out_path, const Solution& sol, double cost, double elapsed_sec, bool final_feasibility, double worst_cost, double mean_cost) {
    std::ofstream ofs(out_path);
    if (!ofs) return false;
    ofs.setf(std::ios::fixed); ofs << setprecision(6);
    ofs << "Initial solution cost: " << cost << "\n";
    ofs << "Improved solution cost: " << solution_score_cost(sol) << "\n";
    ofs << "Worst solution cost: " << worst_cost << "\n";
    ofs << "Mean solution cost: " << mean_cost << "\n";
    ofs << "Mean elapsed time: " << elapsed_sec / CFG_NUM_INITIAL << " seconds\n";
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
    cout << "Reading input file: " << input_file << "\n";
    // Read input instance
    input(input_file);
    cout << "Instance has " << n << " customers, "
         << h << " trucks, " << d << " drones.\n";
    
    // Recalculate tenures based on instance size
    update_tabu_tenures();
    // Build distance matrix for downstream time computations
    cout << "Precomputing distance matrix...\n";
    compute_distance_matrices(loc);
    if (print_dist_matrix) {
        print_distance_matrix();
        return 0; // only print distance matrix and exit
    }

    // Optional auto-tuning based on instance size if requested
    // For now, set auto-tune to always true
    auto_tune = true;
    if (auto_tune) {
        if (n <= 50) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 3);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 100);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 1000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 200);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // modest k for small n
        } else if (n <= 200) {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 3);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 100);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 1000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 200);
            CFG_KNN_K = min(CFG_KNN_K, int(n)); // moderate k for medium n
        } else {
            CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 3);
            CFG_MAX_SEGMENT = min(CFG_MAX_SEGMENT, 100);
            CFG_MAX_ITER_PER_SEGMENT = min(CFG_MAX_ITER_PER_SEGMENT, 1000);
            CFG_MAX_NO_IMPROVE = min(CFG_MAX_NO_IMPROVE, 200);
            CFG_KNN_K = min(CFG_KNN_K, int(n));
        }
    }

    // Precompute KNN lists (if K is zero, disable by building empty adjacency)
    if (CFG_KNN_K > 0) compute_knn_lists(CFG_KNN_K); else { KNN_LIST.assign(n + 1, {}); KNN_ADJ.assign(n + 1, vector<char>(n + 1, 0)); }

    // Pre-filter dronable customers by capacity/energy
    //For another data-testing: change all deadline to a constant 3600 and all serving time to 0

    update_served_by_drone();
    //print test the served by drone
    cout << "Customers that can be served by drone:\n";
    for (int i = 1; i <= n; ++i) {
        if (served_by_drone[i]) {
            cout << i << " "; 
        }
    }
    cout << "\n";

    // Track best across attempts
    bool have_best = false;
    Solution best_overall_sol;
    double best_overall_initial_cost = 0.0;
    double worst_overall_cost = -1.0;
    double sum_overall_cost = 0.0;
    vd best_overall_iter_current, best_overall_iter_best;
    vector<bool> best_overall_current_feasibility;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int attempt = 0; attempt < CFG_NUM_INITIAL; ++attempt) {
        cout << "\n=== Attempt " << (attempt + 1) << " of " << CFG_NUM_INITIAL << " ===\n";
        Solution initial_solution = generate_initial_solution();
        vd iter_current, iter_best;
        vector<bool> current_feasibility;
        Solution improved_sol = tabu_search(initial_solution, iter_current, iter_best, current_feasibility);
        double initial_cost_val = solution_score_cost(initial_solution);
        double current_cost_val = solution_score_cost(improved_sol);

        // Output both to stdout and to file
        cout.setf(std::ios::fixed); cout << setprecision(6);
        cout << "Improved Solution Cost: " << current_cost_val << "\n";
        print_solution_stream(improved_sol, cout);
        
        // Update stats
        sum_overall_cost += current_cost_val;
        if (worst_overall_cost < 0 || current_cost_val > worst_overall_cost) {
            worst_overall_cost = current_cost_val;
        }

        // Update best across attempts
        double best_val = have_best ? solution_score_cost(best_overall_sol) : 1e18;
        if (!have_best || current_cost_val + 1e-12 < best_val) {
            have_best = true;
            best_overall_sol = improved_sol;
            best_overall_initial_cost = initial_cost_val;
            best_overall_iter_current = iter_current;
            best_overall_iter_best = iter_best;
            best_overall_current_feasibility = current_feasibility;
        }
    }
    // Emit best across all attempts
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_seconds = std::chrono::duration<double>(end_time - start_time).count();
    double mean_overall_cost = (CFG_NUM_INITIAL > 0) ? (sum_overall_cost / CFG_NUM_INITIAL) : 0.0;

    if (have_best) {
        cout << "\n=== Best Across Attempts ===\n";
        cout << "Initial Solution Cost: " << best_overall_initial_cost << "\n";
        cout << "Improved Solution Cost: " << solution_score_cost(best_overall_sol) << "\n";
        cout << "Worst Solution Cost: " << worst_overall_cost << "\n";
        cout << "Mean Solution Cost: " << mean_overall_cost << "\n";
        cout << "Mean Elapsed Time: " << elapsed_seconds / CFG_NUM_INITIAL << " seconds\n";
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
        if (write_output_file(out_best, best_overall_sol, best_overall_initial_cost, elapsed_seconds, final_feas, worst_overall_cost, mean_overall_cost)) {
            cout << "Best solution written to " << out_best << "\n";
        } else {
            cout << "Failed to write best solution to " << out_best << "\n";
        }
        string out_iter = "output.txt";
        if (write_iteration_file(out_iter, best_overall_iter_current, best_overall_iter_best, best_overall_current_feasibility)) {
            cout << "Iteration data written to " << out_iter << "\n";
        } else {
            cout << "Failed to write iteration data to " << out_iter << "\n";
        }
    }
    return 0;
}

// Run with: g++ -O3 -std=c++20 tabubu_benchmark.cpp -o tabubu_benchmark && ./tabubu_benchmark instance_modified/100-r-1-c.txt
/*100-r-1-c.txt
// Plotting: python plot_iteration.py --input output.txt --save iterations.png
Cost = 302.457
T0	1.946	0 3 62 63 71 84 67 46 98 75 16 2 83 95 78 4 49 72 19 24 0 
T1	1.999	0 100 82 61 21 38 7 80 9 29 76 26 17 96 53 45 86 81 90 20 23 50 15 0 
T2	1.933	0 74 56 99 73 31 5 30 44 64 43 41 91 25 22 12 65 88 0 
T3	1.871	0 77 52 1 27 89 68 93 28 8 14 55 11 57 10 85 92 18 34 47 0 
D0	1.987	48 36 60 58 33 
D1	1.968	87 32 13 69 97 
D2	1.987	79 51 70 66 
D3	1.983	39 40 35 37 
D4	1.998	94 54 6 59 42 */