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
double Dh = 130000.0; // truck capacity (all trucks) (kg)
double vmax = 1.0; // truck base speed (km/h)
double drone_limit = 10000.0;
double truck_limit = 10000.0;
int L = 24; //number of time segments in a day
vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
vd time_segments_sigma = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; //sigma (truck velocity coefficient) for each time segments
//vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
//vd time_segments_sigma = {0.9, 0.8, 0.4, 0.6,0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 20000.0, E = 100000; //drone's weight and energy capacities (for all drones). E in Hours.
double v_fly_drone = 1.0; // speed of the drone (km/h)
double v_take_off = 1000.0; // km/h
double v_landing = 1000.0; // km/h
double height = 0.0; // km (50m)
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
const int MAX_SEGMENT = 500;
const int MAX_NO_IMPROVE = 1000;
const int MAX_ITER_PER_SEGMENT = 1000;
const double gamma1 = 0.5;
const double gamma2 = 0.3;
const double gamma3 = 0.1;
const double gamma4 = 0.3;

// Runtime-configurable search knobs (initialized from compile-time defaults)
static int CFG_NUM_INITIAL = NUM_OF_INITIAL_SOLUTIONS;
static int CFG_MAX_SEGMENT = MAX_SEGMENT;
static int CFG_MAX_NO_IMPROVE = MAX_NO_IMPROVE;
static int CFG_MAX_ITER_PER_SEGMENT = MAX_ITER_PER_SEGMENT;
static double CFG_TIME_LIMIT_SEC = 0.0; // 0 = unlimited

static int compute_total_iter_budget(int customer_count, int neighborhood_count) {
    // n * K * ceil(sqrt(n))
    int sqrt_n = max(1, (int)ceil(sqrt((double)customer_count)));
    return max(1, customer_count * neighborhood_count * sqrt_n);
}

static int compute_iters_per_segment(int customer_count, int neighborhood_count) {
    // n * ceil(sqrt(K))
    int sqrt_k = max(1, (int)ceil(sqrt((double)neighborhood_count)));
    return max(1, customer_count * sqrt_k);
}

static int compute_segment_count(int total_iters, int iters_per_segment) {
    return max(1, (total_iters + iters_per_segment - 1) / iters_per_segment);
}

// Adaptive penalty coefficients for constraint violations
static double PENALTY_LAMBDA_CAPACITY = 1.0;      // λ for capacity violations
static double PENALTY_LAMBDA_ENERGY = 1.0;        // λ for energy violations  
static double PENALTY_LAMBDA_DEADLINE = 1.0;      // λ for deadline violations
static double PENALTY_EXPONENT = 1.0;             // exponent for penalty term *

static const double PENALTY_INCREASE = 1.2;       // multiply when violated *
static const double PENALTY_DECREASE = 1.2;       // divide when satisfied *
static const double PENALTY_MIN = 1.0;            // minimum λ value
static const double PENALTY_MAX = 1000.0;

double T0 = 150.0; // initial temperature for simulated annealing acceptance
double alpha = 0.9998; // cooling rate for simulated annealing

// Destroy and repair helper
vvd edge_records; // edge_records[i][j]: stores working times for edge (i,j)
const double DESTROY_RATE = 0.1; // fraction of customers to remove during destroy phase
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

void input(string filepath){
    ifstream fin(filepath);
    if (!fin) {
        cerr << "Error: Cannot open " << filepath << endl;
        exit(1);
    }

    // Defaults
    vmax = 1.0;
    v_fly_drone = 1.0;
    Dh = 130000.0;
    Dd = 20000.0;
    E = 100000;
    truck_limit = 1e18; // no route time constraint in PDSTSP
    drone_limit = 1e18; // no route time constraint in PDSTSP

    // PDSTSP: always exactly 2 trucks; drones from file header
    h = 2;
    d = 1;

    // Parse all lines: header lines contain ',' (KEY,value), data lines are tab-separated
    struct Row { int id; double x, y; int drone_eligible; };
    vector<Row> rows;

    string line;
    while (getline(fin, line)) {
        if (line.empty()) continue;
        if (line.find(',') != string::npos) {
            // Header line: KEY,value
            size_t comma = line.find(',');
            string key = line.substr(0, comma);
            string val = line.substr(comma + 1);
            val.erase(0, val.find_first_not_of(" \t\r\n"));
            val.erase(val.find_last_not_of(" \t\r\n") + 1);
            if      (key == "NUM DRONES")  d      = stoi(val);
            else if (key == "TRUCK SPEED") vmax   = stod(val);
            else if (key == "DRONE SPEED") v_fly_drone = stod(val);
        } else {
            // Data line: id<tab>x<tab>y<tab>drone_eligible
            istringstream ss(line);
            Row r;
            if (ss >> r.id >> r.x >> r.y >> r.drone_eligible)
                rows.push_back(r);
        }
    }


    if (rows.empty()) {
        cerr << "Error: No data rows found in " << filepath << endl;
        exit(1);
    }

    // Row with id=0 is the depot; rows 1..n are customers.
    // File ids map directly to internal indices.
    n = (int)rows.size() - 1;

    served_by_drone.assign(n + 1, 0);
    serve_truck.assign(n + 1, 0.0);
    serve_drone.assign(n + 1, 0.0);
    deadline.assign(n + 1, 0.0);
    demand.assign(n + 1, 0.0); // no demand info in this problem format
    loc.assign(n + 1, Point());

    for (const auto& r : rows) {
        loc[r.id] = Point(r.x, r.y, r.id);
        if (r.id > 0) {
            served_by_drone[r.id] = 1 - r.drone_eligible;
        }
    }

    distance_matrix.assign(n + 1, vd(n + 1, 0.0));
    distance_matrix_euclid.assign(n + 1, vd(n + 1, 0.0));
    distance_matrix_manhattan.assign(n + 1, vd(n + 1, 0.0));
    compute_distance_matrices(loc);
}

void update_tabu_tenures() {
    // Base heuristic: roughly proportional to sqrt(n) or n/5
    // sqrt(n) scales better for very large instances
    int base = max(20, (int)(3.0 * sqrt(n))); 
    
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
    double l2_norm = std::sqrt(sum_sq / (h + d));

    // 1e-3 ensures it acts as a tie-breaker without overriding the primary Makespan objective
    return (sol.total_makespan + l2_norm * 1e-4) * pow(penalty_multiplier, PENALTY_EXPONENT);
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

double calculate_score_with_penalties(const double makespan, const double sum_sq, const double capacity_violation, const double energy_violation, const double deadline_violation, int fitness_mode) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * capacity_violation
                                + PENALTY_LAMBDA_ENERGY * energy_violation
                                + PENALTY_LAMBDA_DEADLINE * deadline_violation;
    
    if (fitness_mode == 1) { // L2 norm
        return (makespan + 1e-4 * std::sqrt(sum_sq / (h + d))) * pow(penalty_multiplier, PENALTY_EXPONENT);
    } else if (fitness_mode == 0) { // Makespan
        return makespan * pow(penalty_multiplier, PENALTY_EXPONENT);
    } else if (fitness_mode == 2) { // All vehicles
        return (std::sqrt(sum_sq / (h + d))) * pow(penalty_multiplier, PENALTY_EXPONENT);
    } else {
        return (makespan + 1e-4 * std::sqrt(sum_sq / (h + d))) * pow(penalty_multiplier, PENALTY_EXPONENT);
    }
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
    sol.total_makespan = 0.0;

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
        sol.total_makespan = max(sol.total_makespan, sol.truck_route_times[i]);
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
        sol.total_makespan = max(sol.total_makespan, sol.drone_route_times[i]);
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

    double current_total_time_squared = 0.0;
    int fitness_mode = 1; // 0 for makespan, 1 for L2 norm, 2 for all vehicles (used in calculate_score_with_penalties)

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
                double dist_delta = distance_matrix_manhattan[u][cust] + distance_matrix_manhattan[cust][v] - distance_matrix_manhattan[u][v];
                double time_delta = dist_delta / vmax;
                double new_time = time_delta + sol.truck_route_times[i];
                double new_makespan = max(sol.total_makespan, new_time);
                double new_total_time_squared = current_total_time_squared - (sol.truck_route_times[i] * sol.truck_route_times[i]) + (new_time * new_time);
                double tmp_deadline_violation = max(0.0, (new_time - truck_limit) / truck_limit);
                double tmp_capacity_violation = 0.0;
                double tmp_energy_violation = 0.0; // Trucks don't have energy constraints
                double score = calculate_score_with_penalties(new_time, new_total_time_squared, tmp_capacity_violation, tmp_energy_violation, tmp_deadline_violation, fitness_mode);
                if (score < best_insertion_cost) {
                    best_insertion_cost = score;
                    best_vehicle_type = 0;
                    best_vehicle_idx = i;
                    best_insertion_pos = j;
                }
            }
        }

        for (int i = 0; i < d; ++i) {
            if (served_by_drone[cust] == 0) continue;
            // just push it in the back for drones
            vi temp_route = sol.drone_routes[i];
            temp_route.push_back(cust);
            double dist = distance_matrix_euclid[0][cust] * 2;
            double time_delta = dist / v_fly_drone;
            double new_time = time_delta + sol.drone_route_times[i];
            double new_makespan = max(sol.total_makespan, new_time);
            double new_total_time_squared = current_total_time_squared - (sol.drone_route_times[i] * sol.drone_route_times[i]) + (new_time * new_time);
            double tmp_deadline_violation = max(0.0, (new_time - drone_limit) / drone_limit);
            double tmp_capacity_violation = max(0.0, (demand[cust] - Dd) / Dd);
            double tmp_energy_violation = max(0.0, (dist / v_fly_drone - E) / E);
            double score = calculate_score_with_penalties(new_time, new_total_time_squared, tmp_capacity_violation, tmp_energy_violation, tmp_deadline_violation, fitness_mode);
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
                vd old_metrics = check_truck_route_feasibility(route, 0.0);
                route.insert(route.begin() + best_insertion_pos, cust);
                if (route.back() != 0) route.push_back(0);
                vd metrics = check_truck_route_feasibility(route, 0.0);
                sol.truck_route_times[best_vehicle_idx] = metrics[0];
                sol.total_distance_truck += (distance_matrix_manhattan[route[best_insertion_pos - 1]][cust] + distance_matrix_manhattan[cust][route[best_insertion_pos + 1]] - distance_matrix_manhattan[route[best_insertion_pos - 1]][route[best_insertion_pos + 1]]);
                sol.capacity_violation += metrics[3];
                sol.deadline_violation += metrics[1];
                sol.total_makespan = max(sol.total_makespan, sol.truck_route_times[best_vehicle_idx]);
                current_total_time_squared += (sol.truck_route_times[best_vehicle_idx] * sol.truck_route_times[best_vehicle_idx]) - (old_metrics[0] * old_metrics[0]);
            } else { // Drone
                vi& route = sol.drone_routes[best_vehicle_idx];
                vd old_metrics = check_drone_route_feasibility(route);
                route.insert(route.begin() + best_insertion_pos, cust);
                vd metrics = check_drone_route_feasibility(route);
                sol.drone_route_times[best_vehicle_idx] = metrics[0];
                sol.total_distance_drone += (distance_matrix_euclid[0][cust] * 2);
                sol.capacity_violation += metrics[3];
                sol.energy_violation += metrics[2];
                sol.deadline_violation += metrics[1];
                sol.total_makespan = max(sol.total_makespan, sol.drone_route_times[best_vehicle_idx]);
                current_total_time_squared += (sol.drone_route_times[best_vehicle_idx] * sol.drone_route_times[best_vehicle_idx]) - (old_metrics[0] * old_metrics[0]);
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
    sol.total_makespan = 0.0;
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
        sol.total_makespan = max(sol.total_makespan, sol.truck_route_times[i]);
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
        sol.total_makespan = max(sol.total_makespan, sol.drone_route_times[i]);
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
       << " Total cost=" << solution_score_makespan(sol)
       << ", Deadline violation=" << sol.deadline_violation
       << ", Energy violation=" << sol.energy_violation
       << ", Capacity violation=" << sol.capacity_violation
       << ", Makespan=" << sol.total_makespan
       << "\n";
}


pair<int, bool> critical_solution_index(const Solution& sol) {
    // Identify the vehicle (truck or drone) that contributes most to the penalized objective.
    // Drone 3 is indexed as 2. => returns (2, false)
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

Solution local_search_all_vehicle(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost, int fitness_mode) {
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
        double current_total_time_squared = 0.0;
        for (int v = 0; v < h; ++v) current_total_time_squared += (base_truck_time[v] * base_truck_time[v]);
        for (int v = 0; v < d; ++v) current_total_time_squared += (base_drone_time[v] * base_drone_time[v]);

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

                    double other_vehicle_makespan = 0.0;
                    for (int t = 0; t < h + d; t++){
                        if (t == critical_vehicle_id) continue;
                        if (t == target_veh) continue;
                        double t_time = (t < h) ? base_truck_time[t] : base_drone_time[t - h];
                        other_vehicle_makespan = max(other_vehicle_makespan, t_time);
                    }

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
                        
                        // --- 3. Compute New makespan ---
                        double new_makespan = other_vehicle_makespan;
                        if (target_veh == critical_vehicle_id) {
                            // Single vehicle affected
                            double new_time = (is_truck_mode ? base_truck_time[critical_vehicle_id] : base_drone_time[critical_vehicle_id - h])
                                              + removal_time_delta + insertion_time_delta;
                            new_makespan = max(new_makespan, new_time);
                        } else {
                            // Two vehicles affected
                            double new_time_source = (is_truck_mode ? base_truck_time[critical_vehicle_id] : base_drone_time[critical_vehicle_id - h])
                                                     + removal_time_delta;
                            double new_time_target = (target_is_truck ? base_truck_time[target_veh] : base_drone_time[target_veh - h])
                                                     + insertion_time_delta;
                            new_makespan = max(new_makespan, max(new_time_source, new_time_target));
                        }
                        double new_total_time_squared = current_total_time_squared -
                            pow(is_truck_mode ? base_truck_time[critical_vehicle_id] : base_drone_time[critical_vehicle_id - h], 2.0) +
                            pow(is_truck_mode ? base_truck_time[critical_vehicle_id] : base_drone_time[critical_vehicle_id - h] + removal_time_delta, 2.0);
                        if (target_veh != critical_vehicle_id) {
                            new_total_time_squared -= pow(target_is_truck ? base_truck_time[target_veh] : base_drone_time[target_veh - h], 2.0);
                            new_total_time_squared += pow(target_is_truck ? base_truck_time[target_veh] : base_drone_time[target_veh - h] + insertion_time_delta, 2.0);
                        }

                        // Pure Cost Score
                        double new_score = calculate_score_with_penalties(
                            new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                        if (is_tabu && !(new_score + 1e-8 < best_cost)) {
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
            candidate.total_makespan = 0.0;

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
                candidate.total_makespan = max(candidate.total_makespan, m[0]);
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
                candidate.total_makespan = max(candidate.total_makespan, m[0]);
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

        double current_total_time_squared = 0.0;

        // 1. Pre-calculate metric vectors for all routes
        vector<vd> route_metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            current_total_time_squared += route_metrics[i][0] * route_metrics[i][0];
        }
        for (int i = 0; i < d; ++i) {
            route_metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
            current_total_time_squared += route_metrics[h + i][0] * route_metrics[h + i][0];
        }

        // --- 2. Intra-route Swaps (Trucks only) ---
        for (int v_idx = 0; v_idx < h; ++v_idx) {
            double other_vehicle_makespan = 0.0;
            for (int t = 0; t < h + d; t++){
                if (t == v_idx) continue;
                other_vehicle_makespan = max(other_vehicle_makespan, route_metrics[t][0]);
            }
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

                    double new_makespan = max(other_vehicle_makespan, new_route_time);
                    double new_total_time_squared = current_total_time_squared - (old_metrics[0] * old_metrics[0]) + (new_route_time * new_route_time);
                    
                    double cand_cost = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                    if (is_tabu && !(cand_cost + 1e-8 < best_cost)) continue;
                    
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
                double other_vehicle_makespan = 0.0;
                for (int t = 0; t < h + d; t++){
                    if (t == v_idx_a || t == v_idx_b) continue;
                    other_vehicle_makespan = max(other_vehicle_makespan, route_metrics[t][0]);
                }

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

                        double new_makespan = other_vehicle_makespan;

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

                        // --- Final Score ---
                        auto old_metrics_a = route_metrics[v_idx_a], old_metrics_b = route_metrics[v_idx_b];
                        double new_time_a = old_metrics_a[0] + delta_time_a, new_time_b = old_metrics_b[0] + delta_time_b;
                        new_makespan = max(new_makespan, max(new_time_a, new_time_b));
                        double new_total_time_squared = current_total_time_squared
                            - (old_metrics_a[0] * old_metrics_a[0]) - (old_metrics_b[0] * old_metrics_b[0])
                            + (new_time_a * new_time_a) + (new_time_b * new_time_b);
                        double cand_cost = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                        if (is_tabu && !(cand_cost + 1e-8 < best_cost)) continue;

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
            tabu_list_10[min(best_cust_a, best_cust_b)][max(best_cust_a, best_cust_b)] = current_iter + TABU_TENURE_10;
            return best_neighbor;
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
        double current_total_time_squared = 0.0;
        vd total_capacity_truck(h, 0.0);
        for (int i = 0; i < h; ++i) {
            route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            if (!r.empty() && r.back() != 0) base_truck_dist += distance_matrix_manhattan[r.back()][0];
            for (int c : r) total_capacity_truck[i] += demand[c];
            current_total_time_squared += route_metrics[i][0] * route_metrics[i][0];
        }
        for (int i = 0; i < d; ++i) {
            route_metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
            current_total_time_squared += route_metrics[h + i][0] * route_metrics[h + i][0];
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
                    double other_vehicle_makespan = 0.0;
                    for (int t = 0; t < h + d; t++){
                        if (t == v1) continue;
                        if (t == v2) continue;
                        other_vehicle_makespan = max(other_vehicle_makespan, route_metrics[t][0]);
                    }
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
                        double new_makespan = other_vehicle_makespan;
                        double new_total_time_squared = current_total_time_squared;

                        if (v1 == v2) {
                            double new_route_time = route_metrics[v1][0] + removal_time_delta + insertion_time_delta;
                            new_makespan = max(new_makespan, new_route_time);
                            new_total_time_squared = current_total_time_squared
                                - (route_metrics[v1][0] * route_metrics[v1][0])
                                + (new_route_time * new_route_time);
                        } else {
                            double new_route_time_src = route_metrics[v1][0] + removal_time_delta;
                            double new_route_time_tgt = route_metrics[v2][0] + insertion_time_delta;
                            new_makespan = max(new_makespan, max(new_route_time_src, new_route_time_tgt));
                            new_total_time_squared = current_total_time_squared
                                - (route_metrics[v1][0] * route_metrics[v1][0])
                                - (route_metrics[v2][0] * route_metrics[v2][0])
                                + (new_route_time_src * new_route_time_src)
                                + (new_route_time_tgt * new_route_time_tgt);
                        }

                        double new_score = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                        if (is_tabu && !(new_score + 1e-8 < best_cost)) {
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
            best_neighbor_cost = calculate_score_with_penalties(best_neighbor.total_makespan, 0.0, 0.0, 0.0, 0.0, fitness_mode);
            
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
        double base_truck_dist = 0.0, base_drone_dist = 0.0, current_total_time_squared = 0.0;
        for (int i = 0; i < h; ++i) {
            truck_route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) {
                base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
                base_route_dist[i] += distance_matrix_manhattan[r[k]][r[k+1]];
            }
            current_total_time_squared += truck_route_metrics[i][0] * truck_route_metrics[i][0];
        }
        for (int i = 0; i < d; ++i) {
            const auto& r = initial_solution.drone_routes[i];
            vd drone_metrics = check_drone_route_feasibility(r);
            current_total_time_squared += drone_metrics[0] * drone_metrics[0];
            base_drone_dist += drone_metrics[0] * v_fly_drone;
        }

        for (int v = 0; v < h; ++v) {
            const auto& route = initial_solution.truck_routes[v];
            int m = (int)route.size();
            if (m <= 3) continue; // Not enough customers for 2-opt
            double other_vehicle_makespan = 0.0;
            for (int t = 0; t < h + d; t++){
                if (t == v) continue;
                other_vehicle_makespan = max(other_vehicle_makespan, (t < h ? best_neighbor.truck_route_times[t] : best_neighbor.drone_route_times[t - h]));
            }

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

                    double new_makespan = max(other_vehicle_makespan, new_route_time);
                    double new_total_time_squared = current_total_time_squared
                        - (old_metrics[0] * old_metrics[0])
                        + (new_route_time * new_route_time);

                    double cand_cost = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);
                    if (is_tabu && !(cand_cost + 1e-8 < best_cost)) continue;
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
            tabu_list_2opt[min(best_edge_u, best_edge_v)][max(best_edge_u, best_edge_v)] = current_iter + TABU_TENURE_2OPT;
            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 4) {
        // Neighborhood 4: 3-opt intra-truck (4 true reconnections not achievable by 2-opt).
        // Segments after cuts i,j,k: A=route[0..i], B=route[i+1..j], C=route[j+1..k], D=route[k+1..end]
        // t1=route[i], t2=route[i+1], t3=route[j], t4=route[j+1], t5=route[k], t6=route[k+1]
        // Type 0: A+C+B+D       new edges: (t1,t4),(t5,t2),(t3,t6)
        // Type 1: A+C+rev(B)+D  new edges: (t1,t4),(t5,t3),(t2,t6)
        // Type 2: A+rev(C)+B+D  new edges: (t1,t5),(t4,t2),(t3,t6)
        // Type 3: A+rev(B)+rev(C)+D new edges: (t1,t3),(t2,t5),(t4,t6)
        if ((int)tabu_list_2opt.size() != n + 1 ||
            ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

        double best_neighbor_cost_local = 1e10;
        int best_route_idx = -1;
        int best_i = -1, best_j = -1, best_k = -1;
        int best_type = -1;
        int best_ta = -1, best_tb = -1;

        // Pre-compute current squared times for all vehicles (for fitness score)
        double current_total_time_squared = 0.0;
        vector<double> truck_times(h), drone_times(d);
        for (int i = 0; i < h; ++i) {
            truck_times[i] = check_truck_route_feasibility(initial_solution.truck_routes[i])[0];
            current_total_time_squared += truck_times[i] * truck_times[i];
        }
        for (int i = 0; i < d; ++i) {
            drone_times[i] = check_drone_route_feasibility(initial_solution.drone_routes[i])[0];
            current_total_time_squared += drone_times[i] * drone_times[i];
        }

        for (int v = 0; v < h; ++v) {
            const auto& route = initial_solution.truck_routes[v];
            int m = (int)route.size();
            if (m <= 4) continue; // need at least 3 internal customers

            double other_makespan = 0.0;
            for (int t = 0; t < h + d; ++t) {
                if (t == v) continue;
                other_makespan = max(other_makespan,
                    t < h ? truck_times[t] : drone_times[t - h]);
            }

            for (int i = 1; i < m - 3; ++i) {
                for (int j = i + 1; j < m - 2; ++j) {
                    for (int k = j + 1; k < m - 1; ++k) {
                        int t1 = route[i],   t2 = route[i + 1];
                        int t3 = route[j],   t4 = route[j + 1];
                        int t5 = route[k],   t6 = route[k + 1];

                        double d_orig = distance_matrix_manhattan[t1][t2]
                                      + distance_matrix_manhattan[t3][t4]
                                      + distance_matrix_manhattan[t5][t6];

                        bool is_tabu = (tabu_list_2opt[min(t1,t2)][max(t1,t2)] > current_iter);

                        double new_edges[4] = {
                            // Type 0: A+C+B+D
                            distance_matrix_manhattan[t1][t4] + distance_matrix_manhattan[t5][t2] + distance_matrix_manhattan[t3][t6],
                            // Type 1: A+C+rev(B)+D
                            distance_matrix_manhattan[t1][t4] + distance_matrix_manhattan[t5][t3] + distance_matrix_manhattan[t2][t6],
                            // Type 2: A+rev(C)+B+D
                            distance_matrix_manhattan[t1][t5] + distance_matrix_manhattan[t4][t2] + distance_matrix_manhattan[t3][t6],
                            // Type 3: A+rev(B)+rev(C)+D
                            distance_matrix_manhattan[t1][t3] + distance_matrix_manhattan[t2][t5] + distance_matrix_manhattan[t4][t6],
                        };

                        for (int type = 0; type < 4; ++type) {
                            double delta = new_edges[type] - d_orig;
                            double new_route_time = truck_times[v] + delta / vmax;
                            double new_makespan = max(other_makespan, new_route_time);
                            double new_tsq = current_total_time_squared
                                           - truck_times[v] * truck_times[v]
                                           + new_route_time * new_route_time;

                            double cand_cost = calculate_score_with_penalties(
                                new_makespan, new_tsq, 0.0, 0.0, 0.0, fitness_mode);

                            if (is_tabu && !(cand_cost + 1e-8 < best_cost)) continue;

                            if (cand_cost < best_neighbor_cost_local) {
                                best_neighbor_cost_local = cand_cost;
                                best_route_idx = v;
                                best_i = i; best_j = j; best_k = k;
                                best_type = type;
                                best_ta = min(t1, t2);
                                best_tb = max(t1, t2);
                            }
                        }
                    }
                }
            }
        }

        if (best_route_idx != -1) {
            best_neighbor = initial_solution;
            auto& route = best_neighbor.truck_routes[best_route_idx];

            vi A(route.begin(), route.begin() + best_i + 1);
            vi B(route.begin() + best_i + 1, route.begin() + best_j + 1);
            vi C(route.begin() + best_j + 1, route.begin() + best_k + 1);
            vi D(route.begin() + best_k + 1, route.end());

            route.clear();
            route.insert(route.end(), A.begin(), A.end());
            if (best_type == 0) {
                route.insert(route.end(), C.begin(), C.end());
                route.insert(route.end(), B.begin(), B.end());
            } else if (best_type == 1) {
                route.insert(route.end(), C.begin(), C.end());
                route.insert(route.end(), B.rbegin(), B.rend());
            } else if (best_type == 2) {
                route.insert(route.end(), C.rbegin(), C.rend());
                route.insert(route.end(), B.begin(), B.end());
            } else { // type 3
                route.insert(route.end(), B.rbegin(), B.rend());
                route.insert(route.end(), C.rbegin(), C.rend());
            }
            route.insert(route.end(), D.begin(), D.end());

            best_neighbor = recalculate_solution(best_neighbor);
            tabu_list_2opt[best_ta][best_tb] = current_iter + TABU_TENURE_2OPT;
            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 5) {

        // Neighborhood 5: 2-1 Swap (Swap pair (u,v) from A with single w from B)

        int best_r1 = -1, best_r2 = -1;
        int best_pos_pair = -1, best_pos_single = -1;
        double best_cost_local = 1e18;
        vector<int> best_triple; // For tabu update
        double current_total_time_squared = 0.0;
        
        // 1. Pre-calculate metrics (Reuse N1 logic)
        vector<vd> metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            current_total_time_squared += metrics[i][0] * metrics[i][0];
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
            current_total_time_squared += metrics[h + i][0] * metrics[h + i][0];
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
                double other_vehicle_makespan = 0.0;
                for (int t = 0; t < h + d; t++){
                    if (t == rA) continue;
                    if (t == rB) continue;
                    other_vehicle_makespan = max(other_vehicle_makespan, (t < h ? best_neighbor.truck_route_times[t] : best_neighbor.drone_route_times[t - h]));
                }
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

                            double new_t_a = metrics[rA][0] + delta_time_a;
                            double new_makespan = other_vehicle_makespan;
                            new_makespan = max(new_makespan, new_t_a);
                            double new_total_time_squared = current_total_time_squared
                                - (metrics[rA][0] * metrics[rA][0])
                                + (new_t_a * new_t_a);

                            double new_score = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);
                            
                            if (is_tabu && !(new_score + 1e-8 < best_cost)) {
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

                        double new_t_a = metrics[rA][0] + delta_time_a;
                        double new_t_b = metrics[rB][0] + delta_time_b;

                        double new_makespan = other_vehicle_makespan;
                        new_makespan = max(new_makespan, new_t_a);
                        new_makespan = max(new_makespan, new_t_b);
                        double new_total_time_squared = current_total_time_squared
                            - (metrics[rA][0] * metrics[rA][0])
                            - (metrics[rB][0] * metrics[rB][0])
                            + (new_t_a * new_t_a)
                            + (new_t_b * new_t_b);

                        double score = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                        if (is_tabu && !(score < best_cost)) continue; 
                        
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
            tabu_list_21[best_triple] = current_iter + TABU_TENURE_21;
            return best_neighbor;

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
        double current_total_time_squared = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            double t = metrics[i][0];
            current_total_time_squared += t * t;
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
            double t = metrics[h + i][0];
            current_total_time_squared += t * t;
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
                double other_vehicle_makespan = 0.0;
                for (int t = 0; t < h + d; t++){
                    if (t == rA) continue;
                    if (t == rB) continue;
                    other_vehicle_makespan = max(other_vehicle_makespan, (t < h ? best_neighbor.truck_route_times[t] : best_neighbor.drone_route_times[t - h]));
                }
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

                            double new_t_a = metrics[rA][0] + delta_time_a;

                            double new_makespan = other_vehicle_makespan;
                            new_makespan = max(new_makespan, new_t_a);
                            double new_total_time_squared = current_total_time_squared
                                - (metrics[rA][0] * metrics[rA][0])
                                + (new_t_a * new_t_a);

                            double score = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);
                            if (is_tabu && !(score < best_cost)) continue;

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

                            double t_a = metrics[rA][0] + delta_time_a;
                            double t_b = metrics[rB][0] + delta_time_b;

                            double new_makespan = other_vehicle_makespan;
                            new_makespan = max(new_makespan, t_a);
                            new_makespan = max(new_makespan, t_b);
                            double new_total_time_squared = current_total_time_squared
                                - (metrics[rA][0] * metrics[rA][0])
                                - (metrics[rB][0] * metrics[rB][0])
                                + (t_a * t_a)
                                + (t_b * t_b);

                            double score = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);
                            if (is_tabu && !(score < best_cost)) continue;

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
        return best_neighbor;

    }  else if (neighbor_id == 7) {
        // Neighborhood 7: Depth-2 Ejection Chain (u from A -> replaces v in B -> v inserts in C)
        // Optimized: Random Permutation Selection

        int best_rA = -1, best_rB = -1, best_rC = -1;
        int best_pos_u = -1, best_pos_v = -1, best_pos_k = -1;
        double best_cost_local = 1e18;
        vector<int> best_tabu_key; // {u, v}
        double current_total_time_squared = 0.0;
        
        // 1. Pre-calculate metrics
        vector<vd> metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        for (int i = 0; i < h; ++i) {
            metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            current_total_time_squared += metrics[i][0] * metrics[i][0];
        }
        for (int i = 0; i < d; ++i) {
            metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
            current_total_time_squared += metrics[h + i][0] * metrics[h + i][0];
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
            double other_vehicle_makespan = 0.0;
            for (int t = 0; t < h + d; t++){
                if (t == rA) continue;
                if (t == rB) continue;
                if (t == rC) continue;
                other_vehicle_makespan = max(other_vehicle_makespan, (t < h ? best_neighbor.truck_route_times[t] : best_neighbor.drone_route_times[t - h]));
            }
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

                        double new_tot_time_squared = current_total_time_squared
                            - (metrics[rA][0] * metrics[rA][0])
                            - (metrics[rB][0] * metrics[rB][0])
                            - (metrics[rC][0] * metrics[rC][0])
                            + ((metrics[rA][0] + delta_time_a) * (metrics[rA][0] + delta_time_a))
                            + ((metrics[rB][0] + delta_time_b) * (metrics[rB][0] + delta_time_b))
                            + ((metrics[rC][0] + delta_time_c) * (metrics[rC][0] + delta_time_c));
                        double new_makespan = other_vehicle_makespan;
                        new_makespan = max(new_makespan, metrics[rA][0] + delta_time_a);
                        new_makespan = max(new_makespan, metrics[rB][0] + delta_time_b);
                        new_makespan = max(new_makespan, metrics[rC][0] + delta_time_c);

                        // Score
                        double score = calculate_score_with_penalties(
                            new_makespan, new_tot_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                        if (is_tabu && !(score < best_cost)) continue;

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
            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 8) {
        // N8: Or-opt-3 — relocate 3 adjacent customers from any vehicle to any vehicle

        int best_cust1 = -1, best_cust2 = -1, best_cust3 = -1;
        int best_target_veh = -1;
        int best_pos = -1;
        double best_neighbor_cost_local = 1e10;

        // 1. Pre-calculate metric vectors for all routes
        vector<vd> route_metrics(h + d);
        double base_truck_dist = 0.0, base_drone_dist = 0.0;
        double current_total_time_squared = 0.0;
        for (int i = 0; i < h; ++i) {
            route_metrics[i] = check_truck_route_feasibility(initial_solution.truck_routes[i]);
            const auto& r = initial_solution.truck_routes[i];
            for (size_t k = 0; k + 1 < r.size(); ++k) base_truck_dist += distance_matrix_manhattan[r[k]][r[k+1]];
            current_total_time_squared += route_metrics[i][0] * route_metrics[i][0];
        }
        for (int i = 0; i < d; ++i) {
            route_metrics[h + i] = check_drone_route_feasibility(initial_solution.drone_routes[i]);
            const auto& r = initial_solution.drone_routes[i];
            for (int c : r) base_drone_dist += (2.0 * distance_matrix_euclid[0][c]);
            current_total_time_squared += route_metrics[h + i][0] * route_metrics[h + i][0];
        }

        for (int v1 = 0; v1 < h + d; ++v1) {
            bool is_v1_truck = v1 < h;
            const auto& r1 = is_v1_truck ? initial_solution.truck_routes[v1] : initial_solution.drone_routes[v1 - h];

            size_t start_idx = is_v1_truck ? 1 : 0;
            // Need at least 3 customers plus depots
            if (r1.size() < start_idx + 3 + (is_v1_truck ? 1 : 0)) continue;

            for (size_t i = start_idx; i + 2 < r1.size() - (is_v1_truck ? 1 : 0); ++i) {
                int cust1 = r1[i];
                int cust2 = r1[i+1];
                int cust3 = r1[i+2];
                if (cust1 == 0 || cust2 == 0 || cust3 == 0) continue;

                // --- 1. Removal Delta ---
                vi r1_removed = r1;
                r1_removed.erase(r1_removed.begin() + i, r1_removed.begin() + i + 3);

                vd r1_orig_feas    = check_route_feasibility(r1,        0.0, is_v1_truck);
                vd r1_removed_feas = check_route_feasibility(r1_removed, 0.0, is_v1_truck);
                double removal_time_delta = r1_removed_feas[0] - r1_orig_feas[0];

                double removal_dist_delta = 0.0;
                if (is_v1_truck) {
                    int prev = r1[i - 1];
                    int next = (i + 3 < r1.size()) ? r1[i + 3] : 0;
                    removal_dist_delta = distance_matrix_manhattan[prev][next]
                                       - distance_matrix_manhattan[prev][cust1]
                                       - distance_matrix_manhattan[cust1][cust2]
                                       - distance_matrix_manhattan[cust2][cust3]
                                       - distance_matrix_manhattan[cust3][next];
                } else {
                    removal_dist_delta = -(distance_matrix_euclid[0][cust1] * 2.0
                                        +  distance_matrix_euclid[0][cust2] * 2.0
                                        +  distance_matrix_euclid[0][cust3] * 2.0);
                }

                double tmp_total_dist_truck = base_truck_dist;
                double tmp_total_dist_drone = base_drone_dist;
                if (is_v1_truck) tmp_total_dist_truck += removal_dist_delta;
                else             tmp_total_dist_drone += removal_dist_delta;

                for (int v2 = 0; v2 < h + d; ++v2) {
                    double other_vehicle_makespan = 0.0;
                    for (int t = 0; t < h + d; ++t) {
                        if (t == v1 || t == v2) continue;
                        other_vehicle_makespan = max(other_vehicle_makespan, route_metrics[t][0]);
                    }
                    bool is_v2_truck = v2 < h;
                    // All three must be drone-eligible to move to a drone
                    if ((!served_by_drone[cust1] || !served_by_drone[cust2] || !served_by_drone[cust3]) && !is_v2_truck) continue;
                    if (!is_v2_truck && v1 == v2) continue;

                    vector<int> tabu_key = {cust1, cust2, cust3, v2};
                    bool is_tabu = (tabu_list_20.count(tabu_key) && tabu_list_20[tabu_key] > current_iter);

                    const auto& r2_base = (v1 == v2) ? r1_removed
                                        : (is_v2_truck ? initial_solution.truck_routes[v2]
                                                       : initial_solution.drone_routes[v2 - h]);
                    int loop_end = is_v2_truck ? (int)r2_base.size() : 1;

                    for (int p_insert = 1; p_insert <= loop_end; ++p_insert) {
                        if (is_v2_truck && p_insert >= (int)r2_base.size() && r2_base.size() > 1) continue;

                        // --- 2. Insertion Delta ---
                        double insertion_dist_delta = 0.0;
                        double insertion_time_delta = 0.0;
                        if (is_v2_truck) {
                            if (r2_base.empty()) continue;
                            int prev = r2_base[p_insert - 1];
                            int next = (p_insert < (int)r2_base.size()) ? r2_base[p_insert] : 0;
                            insertion_dist_delta = distance_matrix_manhattan[prev][cust1]
                                                 + distance_matrix_manhattan[cust1][cust2]
                                                 + distance_matrix_manhattan[cust2][cust3]
                                                 + distance_matrix_manhattan[cust3][next]
                                                 - distance_matrix_manhattan[prev][next];
                            insertion_time_delta = insertion_dist_delta / vmax;
                        } else {
                            insertion_dist_delta = distance_matrix_euclid[0][cust1] * 2.0
                                                 + distance_matrix_euclid[0][cust2] * 2.0
                                                 + distance_matrix_euclid[0][cust3] * 2.0;
                            insertion_time_delta = insertion_dist_delta / v_fly_drone;
                        }

                        // --- 3. Score ---
                        double new_makespan = other_vehicle_makespan;
                        double new_total_time_squared = current_total_time_squared;

                        if (v1 == v2) {
                            double new_route_time = route_metrics[v1][0] + removal_time_delta + insertion_time_delta;
                            new_makespan = max(new_makespan, new_route_time);
                            new_total_time_squared = current_total_time_squared
                                - (route_metrics[v1][0] * route_metrics[v1][0])
                                + (new_route_time * new_route_time);
                        } else {
                            double new_route_time_src = route_metrics[v1][0] + removal_time_delta;
                            double new_route_time_tgt = route_metrics[v2][0] + insertion_time_delta;
                            new_makespan = max(new_makespan, max(new_route_time_src, new_route_time_tgt));
                            new_total_time_squared = current_total_time_squared
                                - (route_metrics[v1][0] * route_metrics[v1][0])
                                - (route_metrics[v2][0] * route_metrics[v2][0])
                                + (new_route_time_src * new_route_time_src)
                                + (new_route_time_tgt * new_route_time_tgt);
                        }

                        double new_score = calculate_score_with_penalties(new_makespan, new_total_time_squared, 0.0, 0.0, 0.0, fitness_mode);

                        if (is_tabu && !(new_score + 1e-8 < best_cost)) continue;

                        if (new_score < best_neighbor_cost_local) {
                            best_neighbor_cost_local = new_score;
                            best_cust1 = cust1;
                            best_cust2 = cust2;
                            best_cust3 = cust3;
                            best_target_veh = v2;
                            best_pos = p_insert;
                        }
                    }
                }
            }
        }

        if (best_cust1 != -1) {
            Solution candidate = initial_solution;

            // 1. Remove the triple from its source route
            for (int v = 0; v < h + d; ++v) {
                auto& r = (v < h) ? candidate.truck_routes[v] : candidate.drone_routes[v - h];
                if (r.size() < 3) continue;
                bool found = false;
                for (size_t i = 0; i + 2 < r.size(); ++i) {
                    if (r[i] == best_cust1 && r[i+1] == best_cust2 && r[i+2] == best_cust3) {
                        r.erase(r.begin() + i, r.begin() + i + 3);
                        found = true;
                        break;
                    }
                }
                if (found) break;
            }

            // 2. Insert into target route
            if (best_target_veh < h) {
                auto& r_target = candidate.truck_routes[best_target_veh];
                int pos = max(1, min((int)r_target.size() - 1, best_pos));
                r_target.insert(r_target.begin() + pos, {best_cust1, best_cust2, best_cust3});
                if (r_target.back() != 0) r_target.push_back(0);
            } else {
                auto& rd = candidate.drone_routes[best_target_veh - h];
                rd.push_back(best_cust1);
                rd.push_back(best_cust2);
                rd.push_back(best_cust3);
            }

            best_neighbor = recalculate_solution(candidate);
            best_neighbor_cost = calculate_score_with_penalties(best_neighbor.total_makespan, 0.0, 0.0, 0.0, 0.0, fitness_mode);

            if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
                best_neighbor_cost = best_neighbor_cost_local;
            }

            vector<int> tabu_key = {best_cust1, best_cust2, best_cust3, best_target_veh};
            tabu_list_20[tabu_key] = current_iter + TABU_TENURE_20;
        }

        return best_neighbor;
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
    int fitness_mode = 1; // Use penalties during repair
    
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

        double current_total_time_squared = 0.0;
        for (int i = 0; i < h; ++i) {
            current_total_time_squared += new_sol.truck_route_times[i] * new_sol.truck_route_times[i];
        }
        for (int i = 0; i < d; ++i) {
            current_total_time_squared += new_sol.drone_route_times[i] * new_sol.drone_route_times[i];
        }
        
        // 1. Evaluate all useable Truck positions
        for (int i = 0; i < h; ++i) {
            const vi& route = new_sol.truck_routes[i];
            // Safe loop limit
            for (int p = 1; p < (int)route.size(); ++p) { 
                vi temp_route = route;
                temp_route.insert(temp_route.begin() + p, cust);
                vd m = check_route_feasibility(temp_route, 0.0, true);
                double new_makespan = max(new_sol.total_makespan, m[0]);
                double new_total_time_squared = current_total_time_squared - (new_sol.truck_route_times[i] * new_sol.truck_route_times[i]) + (m[0] * m[0]);
                
                double penalties = PENALTY_LAMBDA_DEADLINE * m[1] + PENALTY_LAMBDA_ENERGY * m[2] + PENALTY_LAMBDA_CAPACITY * m[3];
                double score = calculate_score_with_penalties(new_makespan, new_total_time_squared, max(0.0, m[3]), max(0.0, m[2]), max(0.0, m[1]), fitness_mode);
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

                double new_makespan = max(new_sol.total_makespan, m[0]);
                double new_total_time_squared = current_total_time_squared - (new_sol.drone_route_times[i] * new_sol.drone_route_times[i]) + (m[0] * m[0]);
                
                double new_dist = 0;
                for(int c : temp_route) if(c!=0) new_dist += distance_matrix_euclid[0][c]*2.0;
                double curr_dist = 0;
                for(int c : route) if(c!=0) curr_dist += distance_matrix_euclid[0][c]*2.0;
                double delta_cost = (new_dist - curr_dist) * COST_DRONE_KM;
                
                double penalties = PENALTY_LAMBDA_DEADLINE * m[1] + PENALTY_LAMBDA_ENERGY * m[2] + PENALTY_LAMBDA_CAPACITY * m[3];
                double score = calculate_score_with_penalties(new_makespan, new_total_time_squared, max(0.0, m[3]), max(0.0, m[2]), max(0.0, m[1]), fitness_mode);
                options.push_back({i, false, p, score, temp_route, m[0]});
             }
        }
        
        if (options.empty()) {
             new_sol = greedy_insert_customer(new_sol, cust, true);
             continue;
        }

        double min_score = 1e18;
        for(const auto& opt : options) if(opt.score < min_score) min_score = opt.score;
        double beta = 1.0; 
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
            current_total_time_squared = current_total_time_squared - (new_sol.truck_route_times[choice.veh_idx] * new_sol.truck_route_times[choice.veh_idx]) + (choice.route_time * choice.route_time);
            new_sol.total_makespan = max(new_sol.total_makespan, choice.route_time);
        } else {
            new_sol.drone_routes[choice.veh_idx] = choice.resulting_route;
            new_sol.drone_route_times[choice.veh_idx] = choice.route_time;
            current_total_time_squared = current_total_time_squared - (new_sol.drone_route_times[choice.veh_idx] * new_sol.drone_route_times[choice.veh_idx]) + (choice.route_time * choice.route_time);
            new_sol.total_makespan = max(new_sol.total_makespan, choice.route_time);
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
    new_sol.total_makespan = 0;
    for(int i=0; i<h; ++i) {
        vd m = check_route_feasibility(new_sol.truck_routes[i], 0.0, true);
        new_sol.deadline_violation += m[1];
        new_sol.energy_violation += m[2];
        new_sol.capacity_violation += m[3];
        new_sol.total_time += m[0];
        new_sol.total_makespan = max(new_sol.total_makespan, m[0]);
    } 
    for(int i=0; i<d; ++i) {
        vd m = check_route_feasibility(new_sol.drone_routes[i], 0.0, false);
        new_sol.deadline_violation += m[1];
        new_sol.energy_violation += m[2];
        new_sol.capacity_violation += m[3];
        new_sol.total_time += m[0];
        new_sol.total_makespan = max(new_sol.total_makespan, m[0]);
    }
    return new_sol;
}

Solution destroy_worst_repair_random(Solution sol, double destroy_fraction = 0.1) {
    unordered_set<int> to_destroy;
    int destroy_count = max(1, static_cast<int>(n * destroy_fraction)); 
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
                candidates.push_back({cust, i, true, saving});
            }
        }
        for (int i = 0; i < d; ++i) {
            const vi& route = current_sol.drone_routes[i];
            if (route.size() <= 1) continue; 
            for (int p = 1; p < (int)route.size(); ++p) { 
                int cust = route[p];
                if (cust == 0 || to_destroy.count(cust)) continue;
                double saving = distance_matrix_euclid[0][cust] * 2.0;
                candidates.push_back({cust, i, false, saving});
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
        return destroy_worst_repair_random(sol, 0.1);
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
    int consecutive_failed_perturbations = 0;
    double cost_at_last_perturbation = std::numeric_limits<double>::infinity();
    double best_feasible_cost = initial_feasible
        ? solution_score_makespan(initial_solution)
        : std::numeric_limits<double>::infinity();
    double score[NUM_NEIGHBORHOODS] = {0.0};
    double weight[NUM_NEIGHBORHOODS];
    for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) weight[i] = 1.0 / NUM_NEIGHBORHOODS;
    int count[NUM_NEIGHBORHOODS] = {0};

    int fitness_mode = 1; // 0 for makespan, 1 for makespan + L2 Norm with penalties, 2 is L2 Norm only on all vehicles

    Solution current_sol = initial_solution;
    double current_cost = calculate_score_with_penalties(current_sol.total_makespan, 0.0, max(0.0, current_sol.capacity_violation), max(0.0, current_sol.energy_violation), max(0.0, current_sol.deadline_violation), fitness_mode);

    current_sol = initial_solution;
    iter_current.clear();
    iter_best.clear();
    iter_feasible.clear();
    // Unified Tabu Search on Cost
    int iter = 1;
    int total_iters = CFG_MAX_SEGMENT * CFG_MAX_ITER_PER_SEGMENT;
    int no_improve_iters = 0;

    cout << "=== Starting Unified Tabu Search (Minimizing Weighted Cost) ===\n";
    cout << "Initial Cost: " << current_cost << "\n";

    double best_solution_score_now = current_cost;
    double segment_start_best_score = best_solution_score_now;
    int no_improve_segments = 0;

    while (iter <= total_iters) {
        if (CFG_TIME_LIMIT_SEC > 0.0) {
            double elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ts_start).count();
            if (elapsed >= CFG_TIME_LIMIT_SEC) break;
        }

        double current_score = calculate_score_with_penalties(current_sol.total_makespan, 0.0, max(0.0, current_sol.capacity_violation), max(0.0, current_sol.energy_violation), max(0.0, current_sol.deadline_violation), fitness_mode);
        double current_pure_cost = solution_score_makespan(current_sol);
        iter_current.push_back(current_pure_cost);
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
            init_neighbor = local_search_all_vehicle(current_sol, selected_neighbor, iter, best_solution_score_now, fitness_mode);
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
             exit(1); // Critical failure - exit immediately for debugging
        }

        if (!integrity_ok) {
            neighbor = current_sol;
        }

        bool neighbor_feasible = is_feasible(neighbor);
        double neighbor_score = calculate_score_with_penalties(neighbor.total_makespan, 0.0, max(0.0, neighbor.capacity_violation), max(0.0, neighbor.energy_violation), max(0.0, neighbor.deadline_violation), fitness_mode);

        // Acceptance
        if (neighbor_score + 1e-12 < best_solution_score_now) {
            
            current_sol = neighbor;
            best_solution = neighbor;
            best_solution_score_now = neighbor_score;
            score[selected_neighbor] += gamma1;
            no_improve_iters = 0;
            
        } else if (neighbor_score + 1e-12 < current_score) {
            current_sol = neighbor;
            score[selected_neighbor] += gamma2;
            no_improve_iters = 0;
        } else {
            double T = T0 * pow(alpha, iter);
            double delta = current_score - neighbor_score;
            double ap = exp(delta / T);
            double rand_val = ((double) rand() / (RAND_MAX));
            if (rand_val < ap) {
                current_sol = neighbor;
                current_cost = neighbor.total_makespan;
                current_score = neighbor_score;
            }

            score[selected_neighbor] += gamma3;
            no_improve_iters++;
            
        }

        // Update Feasible Best (always tracked in pure makespan — mode-independent)
        if (neighbor_feasible) {
             double n_cost = solution_score_makespan(neighbor);
             if (n_cost + 1e-12 < best_feasible_cost) {
                 best_feasible_solution = neighbor;
                 best_feasible_cost = n_cost;
                 cout << "Iter " << iter << " New Best Feasible Cost: " << best_feasible_cost << "\n";
             }
        }
        
        update_penalties(current_sol);

        // Perturbation (Destroy/Repair)
        if (no_improve_iters >= CFG_MAX_NO_IMPROVE) {
            if (best_feasible_cost >= cost_at_last_perturbation - 1e-9)
                consecutive_failed_perturbations++;
            else
                consecutive_failed_perturbations = 0;
            cost_at_last_perturbation = best_feasible_cost;

            // Adaptive fraction: grows slowly, capped at 0.15
            double destroy_fraction = min(0.15, 0.05 + 0.01 * consecutive_failed_perturbations);

            current_sol = destroy_worst_repair_random(current_sol, destroy_fraction);

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

            // Segment-level stagnation: cycle fitness_mode after 2 non-improving segments
            if (best_solution_score_now + 1e-12 < segment_start_best_score) {
                no_improve_segments = 0;
            } else {
                no_improve_segments++;
                if (no_improve_segments >= 2) {
                    fitness_mode = (fitness_mode + 1) % 3; // Cycle through fitness modes
                    no_improve_segments = 0;
                    // Clear tabu lists so the perturbed region is explored freely
                    tabu_list_10.clear();
                    tabu_list_11.clear();
                    tabu_list_20.clear();
                    tabu_list_2opt.clear();
                    tabu_list_2opt_star.clear();
                    tabu_list_22.clear();
                    tabu_list_21.clear();
                    tabu_list_ejection.clear();
                    // Reset current solution to best feasible to re-anchor search in the new landscape
                    if (best_feasible_cost >= cost_at_last_perturbation - 1e-9) consecutive_failed_perturbations++;
                    else consecutive_failed_perturbations = 0;
                    cost_at_last_perturbation = best_feasible_cost;

                    // Adaptive fraction: grows slowly, capped at 0.15
                    double destroy_fraction = min(0.15, 0.05 + 0.01 * consecutive_failed_perturbations);

                    Solution base_solution = (consecutive_failed_perturbations >= 3 && consecutive_failed_perturbations % 3 == 0) ? best_feasible_solution : current_sol;
                    current_sol = destroy_worst_repair_random(base_solution, destroy_fraction);
                    current_sol = recalculate_solution(current_sol);
                    // Re-anchor aspiration criterion to the new fitness landscape
                    best_solution_score_now = calculate_score_with_penalties(
                        best_solution.total_makespan, 0.0,
                        max(0.0, best_solution.capacity_violation),
                        max(0.0, best_solution.energy_violation),
                        max(0.0, best_solution.deadline_violation),
                        fitness_mode);
                }
            }
            segment_start_best_score = best_solution_score_now;
//            cout << "Iter " << iter << " Segment Update | Fitness Mode: " << fitness_mode << " | Weights: ";
/*             for (int i = 0; i < NUM_NEIGHBORHOODS; ++i) {
                cout << weight[i] << " ";
            } */
//            cout << "\n";

        }

        iter++;
    }

    // Post optimize best feasible solution if found
    // Uses pure makespan (mode 0) since we are polishing the best feasible solution
    int post_opt_iters = 100 * NUM_NEIGHBORHOODS;
    if (best_feasible_cost < std::numeric_limits<double>::infinity()) {
        Solution post_sol = best_feasible_solution;
        for (int i = 0; i < post_opt_iters; ++i) {
            for (int j = 0; j < NUM_NEIGHBORHOODS; ++j) {
                Solution candidate = local_search_all_vehicle(post_sol, j, iter, best_feasible_cost, 0);
                candidate = recalculate_solution(candidate);
                if (is_feasible(candidate)) {
                    double candidate_cost = solution_score_makespan(candidate);
                    if (candidate_cost + 1e-12 < best_feasible_cost) {
                        post_sol = candidate;
                        best_feasible_solution = candidate;
                        best_feasible_cost = candidate_cost;
                        cout << "Post-Optimization Iter " << i + 1 << " New Best Feasible Cost: " << best_feasible_cost << "\n";
                    }
                }
                iter++;
            }
        }
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

// Extracts the BKS solution block for the given instance filepath from bks.txt.
// Returns a string suitable for passing to check_benchmark_solution(), or "" if not found.
static std::string load_bks_for_instance(const std::string& instance_filepath,
                                          const std::string& bks_filepath = "/workspaces/PDSTSP/pdstsp-upload/bks.txt") {
    size_t sep = instance_filepath.find_last_of("/\\");
    std::string filename = (sep == std::string::npos) ? instance_filepath : instance_filepath.substr(sep + 1);

    std::ifstream fin(bks_filepath);
    if (!fin) {
        std::cerr << "Warning: Cannot open BKS file: " << bks_filepath << "\n";
        return "";
    }

    std::string line;
    bool in_block = false;
    std::string block;

    while (std::getline(fin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!in_block) {
            if (line == filename) in_block = true;
        } else {
            if (line.empty()) break;
            block += line + "\n";
        }
    }

    if (!in_block) {
        std::cerr << "Warning: No BKS entry found for: " << filename << "\n";
        return "";
    }
    return block;
}

Solution check_benchmark_solution(const std::string& benchmark_string) {
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

        int vehicle_id = (type.size() > 1) ? std::stoi(type.substr(1)) : 0;
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
    return recalculated_sol;
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
    ofs << "Improved solution cost: " << solution_score_makespan(sol) << "\n";
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
             << " [--num-truck=N] [--num-drone=N]"
             << "\n";
        return 1;
    }
    string input_file = argv[1];
    bool print_dist_matrix = false;
    bool auto_tune = false;
    bool user_set_segments = false;
    bool user_set_iters = false;
    bool user_set_no_improve = false;
    int override_num_truck = -1; // -1 means use value from file
    int override_num_drone = -1; // -1 means use value from file
    // Parse optional flags
    for (int ai = 2; ai < argc; ++ai) {
        string arg = argv[ai];
        if (arg == "--print-distance-matrix") { print_dist_matrix = true; continue; }
        string v;
        if (parse_kv_flag(arg, "--attempts", v)) { CFG_NUM_INITIAL = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--segments", v)) { CFG_MAX_SEGMENT = max(1, stoi(v)); user_set_segments = true; continue; }
        if (parse_kv_flag(arg, "--iters", v)) { CFG_MAX_ITER_PER_SEGMENT = max(1, stoi(v)); user_set_iters = true; continue; }
        if (parse_kv_flag(arg, "--no-improve", v)) { CFG_MAX_NO_IMPROVE = max(1, stoi(v)); user_set_no_improve = true; continue; }
        if (parse_kv_flag(arg, "--time-limit", v)) { CFG_TIME_LIMIT_SEC = max(0.0, stod(v)); continue; }
        if (parse_kv_flag(arg, "--knn-k", v)) { CFG_KNN_K = max(0, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--knn-window", v)) { CFG_KNN_WINDOW = max(0, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--num-truck", v)) { override_num_truck = max(1, stoi(v)); continue; }
        if (parse_kv_flag(arg, "--num-drone", v)) { override_num_drone = max(0, stoi(v)); continue; }
        if (arg == "--auto-tune") { auto_tune = true; continue; }
    }
    cout << "Reading input file: " << input_file << "\n";
    // Read input instance
    input(input_file);
    // Apply overrides after file is parsed
    if (override_num_truck >= 1) h = override_num_truck;
    if (override_num_drone >= 0) d = override_num_drone;
    cout << "Instance has " << n << " customers, "
         << h << " trucks, " << d << " drones.\n";
    
    // Recalculate tenures based on instance size
    update_tabu_tenures();
    // Distance matrices already built inside input(); no need to recompute.
    if (print_dist_matrix) {
        print_distance_matrix();
        return 0; // only print distance matrix and exit
    }



    // Optional auto-tuning based on instance size if requested
    // For now, set auto-tune to always true
    auto_tune = true;
    if (CFG_TIME_LIMIT_SEC <= 0.0) CFG_TIME_LIMIT_SEC = 3600.0; // 5 minutes default; overridden by --time-limit
    if (auto_tune) {
        int tuned_total_iters = compute_total_iter_budget(n, NUM_NEIGHBORHOODS);
        CFG_NUM_INITIAL = min(CFG_NUM_INITIAL, 50);
        CFG_KNN_K = min(CFG_KNN_K, int(n));
        if (!user_set_iters) {
            CFG_MAX_ITER_PER_SEGMENT = compute_iters_per_segment(n, NUM_NEIGHBORHOODS);
        }
        if (!user_set_no_improve) {
            CFG_MAX_NO_IMPROVE = 2 * CFG_MAX_ITER_PER_SEGMENT;
        }
        if (!user_set_segments) {
            CFG_MAX_SEGMENT = compute_segment_count(tuned_total_iters, CFG_MAX_ITER_PER_SEGMENT);
        }
    }

    cout << "Search config: total_iters=" << (1LL * CFG_MAX_SEGMENT * CFG_MAX_ITER_PER_SEGMENT)
         << ", segments=" << CFG_MAX_SEGMENT
         << ", iters/segment=" << CFG_MAX_ITER_PER_SEGMENT
         << ", no-improve=" << CFG_MAX_NO_IMPROVE << "\n";

    // Precompute KNN lists (if K is zero, disable by building empty adjacency)
    if (CFG_KNN_K > 0) compute_knn_lists(CFG_KNN_K); else { KNN_LIST.assign(n + 1, {}); KNN_ADJ.assign(n + 1, vector<char>(n + 1, 0)); }

    // Drone eligibility already parsed from file (4th column); skip update_served_by_drone().
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

    double total_time_limit = CFG_TIME_LIMIT_SEC; // 0 = unlimited
    auto start_time = std::chrono::high_resolution_clock::now();
    int completed_attempts = 0;
    while (true) {
        double total_elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
        if (total_time_limit > 0.0 && total_elapsed >= total_time_limit) break;

        completed_attempts++;
        cout << "\n=== Attempt " << completed_attempts << " ===\n";
        Solution initial_solution = generate_initial_solution();
/*         std::string bks = load_bks_for_instance(input_file);
        if (!bks.empty()) {
            initial_solution = check_benchmark_solution(bks);
        } */
        vd iter_current, iter_best;
        vector<bool> current_feasibility;

        // Cap this attempt to remaining budget so tabu_search terminates on time
        if (total_time_limit > 0.0)
            CFG_TIME_LIMIT_SEC = total_time_limit - total_elapsed;
        Solution improved_sol = tabu_search(initial_solution, iter_current, iter_best, current_feasibility);
        CFG_TIME_LIMIT_SEC = total_time_limit; // restore for next check

        double initial_cost_val = solution_score_makespan(initial_solution);
        double current_cost_val = solution_score_makespan(improved_sol);

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
        double best_val = have_best ? solution_score_makespan(best_overall_sol) : 1e18;
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
    double mean_overall_cost = (completed_attempts > 0) ? (sum_overall_cost / completed_attempts) : 0.0;

    if (have_best) {
        cout << "\n=== Best Across Attempts ===\n";
        cout << "Initial Solution Cost: " << best_overall_initial_cost << "\n";
        cout << "Improved Solution Cost: " << solution_score_makespan(best_overall_sol) << "\n";
        cout << "Worst Solution Cost: " << worst_overall_cost << "\n";
        cout << "Mean Solution Cost: " << mean_overall_cost << "\n";
        cout << "Total Attempts: " << completed_attempts << "\n";
        cout << "Mean Elapsed Time: " << elapsed_seconds / completed_attempts << " seconds\n";
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

// Run with: g++ -O3 -std=c++20 tabubu_raj.cpp -o tabubu_raj && ./tabubu_raj /workspaces/PDSTSP/pdstsp-upload/instances/tsplib/att48-80-2-1-1.txt --num-truck=2 --num-drone=2
// Plotting: python plot_iteration.py --input output.txt --save iterations.png
/*   data = [
    # Table 12 - eil101
    {"Instance": "eil101-80-2-1-1", "Best Found": get_best_cost("eil101-80-2-1-1"), "MDFGQ18": 564.00, "DMN20": 564.00, "SISSRs": 564.00},
    {"Instance": "eil101-80-2-1-2", "Best Found": get_best_cost("eil101-80-2-1-2"), "MDFGQ18": 650.00, "DMN20": 648.98, "SISSRs": 648.98},
    {"Instance": "eil101-0-2-1-1", "Best Found": get_best_cost("eil101-0-2-1-1"), "MDFGQ18": 819.00, "DMN20": 819.00, "SISSRs": 819.00},
    {"Instance": "eil101-20-2-1-1", "Best Found": get_best_cost("eil101-20-2-1-1"), "MDFGQ18": 738.00, "DMN20": 736.00, "SISSRs": 736.00},
    {"Instance": "eil101-40-2-1-1", "Best Found": get_best_cost("eil101-40-2-1-1"), "MDFGQ18": 646.00, "DMN20": 646.00, "SISSRs": 646.00},
    {"Instance": "eil101-60-2-1-1", "Best Found": get_best_cost("eil101-60-2-1-1"), "MDFGQ18": 578.00, "DMN20": 578.00, "SISSRs": 578.00},
    {"Instance": "eil101-100-2-1-1", "Best Found": get_best_cost("eil101-100-2-1-1"), "MDFGQ18": 561.41, "DMN20": 560.00, "SISSRs": 560.00},
    {"Instance": "eil101-80-1-1-1", "Best Found": get_best_cost("eil101-80-1-1-1"), "MDFGQ18": 650.00, "DMN20": 650.00, "SISSRs": 650.00},
    {"Instance": "eil101-80-3-1-1", "Best Found": get_best_cost("eil101-80-3-1-1"), "MDFGQ18": 504.00, "DMN20": 504.00, "SISSRs": 503.19},
    {"Instance": "eil101-80-4-1-1", "Best Found": get_best_cost("eil101-80-4-1-1"), "MDFGQ18": 456.00, "DMN20": 456.00, "SISSRs": 456.00},
    {"Instance": "eil101-80-5-1-1", "Best Found": get_best_cost("eil101-80-5-1-1"), "MDFGQ18": 420.83, "DMN20": 421.00, "SISSRs": 420.83},
    {"Instance": "eil101-80-2-2-1", "Best Found": get_best_cost("eil101-80-2-2-1"), "MDFGQ18": 456.00, "DMN20": 456.00, "SISSRs": 456.00},
    {"Instance": "eil101-80-2-3-1", "Best Found": get_best_cost("eil101-80-2-3-1"), "MDFGQ18": 395.00, "DMN20": 395.00, "SISSRs": 395.00},
    {"Instance": "eil101-80-2-4-1", "Best Found": get_best_cost("eil101-80-2-4-1"), "MDFGQ18": 346.68, "DMN20": 346.00, "SISSRs": 346.00},
    {"Instance": "eil101-80-2-5-1", "Best Found": get_best_cost("eil101-80-2-5-1"), "MDFGQ18": 319.74, "DMN20": 318.00, "SISSRs": 318.00},
    
    # Table 13 - gr120
    {"Instance": "gr120-80-2-1-1", "Best Found": get_best_cost("gr120-80-2-1-1"), "MDFGQ18": 1414.00, "DMN20": 1420.76, "SISSRs": 1414.00},
    {"Instance": "gr120-80-2-1-2", "Best Found": get_best_cost("gr120-80-2-1-2"), "MDFGQ18": 1730.00, "DMN20": 1726.00, "SISSRs": 1726.00},
    {"Instance": "gr120-0-2-1-1", "Best Found": get_best_cost("gr120-0-2-1-1"), "MDFGQ18": 2006.00, "DMN20": 2006.00, "SISSRs": 2006.00},
    {"Instance": "gr120-20-2-1-1", "Best Found": get_best_cost("gr120-20-2-1-1"), "MDFGQ18": 1736.00, "DMN20": 1736.00, "SISSRs": 1736.00},
    {"Instance": "gr120-40-2-1-1", "Best Found": get_best_cost("gr120-40-2-1-1"), "MDFGQ18": 1624.00, "DMN20": 1624.00, "SISSRs": 1624.00},
    {"Instance": "gr120-60-2-1-1", "Best Found": get_best_cost("gr120-60-2-1-1"), "MDFGQ18": 1494.00, "DMN20": 1494.00, "SISSRs": 1494.00},
    {"Instance": "gr120-100-2-1-1", "Best Found": get_best_cost("gr120-100-2-1-1"), "MDFGQ18": 1414.80, "DMN20": 1416.00, "SISSRs": 1414.00},
    {"Instance": "gr120-80-1-1-1", "Best Found": get_best_cost("gr120-80-1-1-1"), "MDFGQ18": 1592.00, "DMN20": 1592.00, "SISSRs": 1592.00},
    {"Instance": "gr120-80-3-1-1", "Best Found": get_best_cost("gr120-80-3-1-1"), "MDFGQ18": 1289.27, "DMN20": 1291.00, "SISSRs": 1284.74},
    {"Instance": "gr120-80-4-1-1", "Best Found": get_best_cost("gr120-80-4-1-1"), "MDFGQ18": 1189.71, "DMN20": 1192.00, "SISSRs": 1186.00},
    {"Instance": "gr120-80-5-1-1", "Best Found": get_best_cost("gr120-80-5-1-1"), "MDFGQ18": 1112.00, "DMN20": 1114.00, "SISSRs": 1112.00},
    {"Instance": "gr120-80-2-2-1", "Best Found": get_best_cost("gr120-80-2-2-1"), "MDFGQ18": 1188.51, "DMN20": 1197.00, "SISSRs": 1186.00},
    {"Instance": "gr120-80-2-3-1", "Best Found": get_best_cost("gr120-80-2-3-1"), "MDFGQ18": 1044.65, "DMN20": 1050.00, "SISSRs": 1044.00},
    {"Instance": "gr120-80-2-4-1", "Best Found": get_best_cost("gr120-80-2-4-1"), "MDFGQ18": 946.04, "DMN20": 946.04, "SISSRs": 943.00},
    {"Instance": "gr120-80-2-5-1", "Best Found": get_best_cost("gr120-80-2-5-1"), "MDFGQ18": 880.00, "DMN20": 881.00, "SISSRs": 878.69},
    
    # Table 14 - pr152
    {"Instance": "pr152-80-2-1-1", "Best Found": get_best_cost("pr152-80-2-1-1"), "MDFGQ18": 76008.00, "DMN20": 76008.00, "SISSRs": 76008.00},
    {"Instance": "pr152-80-2-1-2", "Best Found": get_best_cost("pr152-80-2-1-2"), "MDFGQ18": 76556.00, "DMN20": 76556.00, "SISSRs": 76556.00},
    {"Instance": "pr152-0-2-1-1", "Best Found": get_best_cost("pr152-0-2-1-1"), "MDFGQ18": 86596.00, "DMN20": 86596.00, "SISSRs": 86596.00},
    {"Instance": "pr152-20-2-1-1", "Best Found": get_best_cost("pr152-20-2-1-1"), "MDFGQ18": 82504.00, "DMN20": 82504.00, "SISSRs": 82504.00},
    {"Instance": "pr152-40-2-1-1", "Best Found": get_best_cost("pr152-40-2-1-1"), "MDFGQ18": 77372.00, "DMN20": 77316.00, "SISSRs": 77236.00},
    {"Instance": "pr152-60-2-1-1", "Best Found": get_best_cost("pr152-60-2-1-1"), "MDFGQ18": 76786.00, "DMN20": 76786.00, "SISSRs": 76758.00},
    {"Instance": "pr152-100-2-1-1", "Best Found": get_best_cost("pr152-100-2-1-1"), "MDFGQ18": 74468.00, "DMN20": 74302.00, "SISSRs": 74302.00},
    {"Instance": "pr152-80-1-1-1", "Best Found": get_best_cost("pr152-80-1-1-1"), "MDFGQ18": 80164.00, "DMN20": 79952.00, "SISSRs": 79952.00},
    {"Instance": "pr152-80-3-1-1", "Best Found": get_best_cost("pr152-80-3-1-1"), "MDFGQ18": 72936.00, "DMN20": 72936.00, "SISSRs": 72936.00},
    {"Instance": "pr152-80-4-1-1", "Best Found": get_best_cost("pr152-80-4-1-1"), "MDFGQ18": 70412.00, "DMN20": 70328.00, "SISSRs": 70148.00},
    {"Instance": "pr152-80-5-1-1", "Best Found": get_best_cost("pr152-80-5-1-1"), "MDFGQ18": 67798.00, "DMN20": 67798.00, "SISSRs": 67858.00},
    {"Instance": "pr152-80-2-2-1", "Best Found": get_best_cost("pr152-80-2-2-1"), "MDFGQ18": 70244.00, "DMN20": 70405.45, "SISSRs": 70148.00},
    {"Instance": "pr152-80-2-3-1", "Best Found": get_best_cost("pr152-80-2-3-1"), "MDFGQ18": 65062.10, "DMN20": 64720.30, "SISSRs": 64550.00},
    {"Instance": "pr152-80-2-4-1", "Best Found": get_best_cost("pr152-80-2-4-1"), "MDFGQ18": 60027.40, "DMN20": 59772.00, "SISSRs": 59756.00},
    {"Instance": "pr152-80-2-5-1", "Best Found": get_best_cost("pr152-80-2-5-1"), "MDFGQ18": 56336.00, "DMN20": 56262.00, "SISSRs": 56178.00},
    
    # Table 15 - gr229
    {"Instance": "gr229-80-2-1-1", "Best Found": get_best_cost("gr229-80-2-1-1"), "MDFGQ18": 1794.84, "DMN20": 1785.86, "SISSRs": 1780.86},
    {"Instance": "gr229-80-2-1-2", "Best Found": get_best_cost("gr229-80-2-1-2"), "MDFGQ18": 1913.74, "DMN20": 1911.58, "SISSRs": 1911.58},
    {"Instance": "gr229-0-2-1-1", "Best Found": get_best_cost("gr229-0-2-1-1"), "MDFGQ18": 2020.16, "DMN20": 2017.24, "SISSRs": 2017.24},
    {"Instance": "gr229-20-2-1-1", "Best Found": get_best_cost("gr229-20-2-1-1"), "MDFGQ18": 1862.76, "DMN20": 1860.14, "SISSRs": 1860.14},
    {"Instance": "gr229-40-2-1-1", "Best Found": get_best_cost("gr229-40-2-1-1"), "MDFGQ18": 1828.02, "DMN20": 1827.02, "SISSRs": 1824.80},
    {"Instance": "gr229-60-2-1-1", "Best Found": get_best_cost("gr229-60-2-1-1"), "MDFGQ18": 1807.50, "DMN20": 1797.37, "SISSRs": 1796.62},
    {"Instance": "gr229-100-2-1-1", "Best Found": get_best_cost("gr229-100-2-1-1"), "MDFGQ18": 1498.05, "DMN20": 1496.29, "SISSRs": 1496.29},
    {"Instance": "gr229-80-1-1-1", "Best Found": get_best_cost("gr229-80-1-1-1"), "MDFGQ18": 1865.00, "DMN20": 1863.12, "SISSRs": 1861.98},
    {"Instance": "gr229-80-3-1-1", "Best Found": get_best_cost("gr229-80-3-1-1"), "MDFGQ18": 1735.16, "DMN20": 1725.45, "SISSRs": 1716.74},
    {"Instance": "gr229-80-4-1-1", "Best Found": get_best_cost("gr229-80-4-1-1"), "MDFGQ18": 1679.33, "DMN20": 1675.82, "SISSRs": 1664.78},
    {"Instance": "gr229-80-5-1-1", "Best Found": get_best_cost("gr229-80-5-1-1"), "MDFGQ18": 1642.04, "DMN20": 1629.38, "SISSRs": 1620.24},
    {"Instance": "gr229-80-2-2-1", "Best Found": get_best_cost("gr229-80-2-2-1"), "MDFGQ18": 1686.75, "DMN20": 1673.72, "SISSRs": 1664.78},
    {"Instance": "gr229-80-2-3-1", "Best Found": get_best_cost("gr229-80-2-3-1"), "MDFGQ18": 1609.90, "DMN20": 1592.52, "SISSRs": 1580.88},
    {"Instance": "gr229-80-2-4-1", "Best Found": get_best_cost("gr229-80-2-4-1"), "MDFGQ18": 1518.62, "DMN20": 1526.92, "SISSRs": 1511.74},
    {"Instance": "gr229-80-2-5-1", "Best Found": get_best_cost("gr229-80-2-5-1"), "MDFGQ18": 1483.68, "DMN20": 1467.76, "SISSRs": 1458.74},
]
 */