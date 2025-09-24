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

struct Point { double x, y; int id; };

int n, h, d; //number of customers, number of trucks, number of drones
vector<Point> loc; // loc[i]: location (x, y) of customer i, if i = 0, it is depot
vd serve_truck, serve_drone; // time taken by truck and drone to serve each customer (hours)
vi served_by_drone; //whether each customer can be served by drone or not, 1 if yes, 0 if no
vd deadline; //customer deadlines
vd demand; // demand[i]: demand of customer i
int Dh = 500, vmax = 15.6464; //truck capacities (for all trucks)
int L = 24; //number of time segments in a day
vd time_segments_sigma = {0.9, 0.8, 0.4, 0.6, 0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8, 0.9, 0.8, 0.4, 0.6, 0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 2.27, E = 7200000.0; //drone's weight and energy capacities (for all drones)
double v_fly_drone = 31.3, v_take_off = 15.6, v_landing = 7.8, height = 50; // maximum speed of the drone
double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    double total_cost; //total cost of the solution
};

void input(){
        // Open the file 50.10.2.txt
        ifstream fin("50.10.2.txt");
        if (!fin) {
            cerr << "Error: Cannot open 50.10.2.txt" << endl;
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
vvd compute_distance_matrices(const vector<Point>& loc) {
    int n = loc.size() - 1; // assuming loc[0] is depot
    vector<vector<double>> distance_matrix(n+1, vd(n+1));
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distance_matrix[i][j] = sqrt((loc[i].x - loc[j].x) * (loc[i].x - loc[j].x)
                                         + (loc[i].y - loc[j].y) * (loc[i].y - loc[j].y)); // Euclidean
        }
    }
    return {distance_matrix};
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
    vector<double> visit_times(route.size(), 0.0); // visit_times[k]: time (seconds) when node route[k] is visited
    vector<int> customers_since_last_depot;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist_left = distance_matrix[from][to]; // meters
        while (dist_left > 1e-8) {
            // Convert time to hours for segment lookup
            double t_hr = time / 3600.0;
            int seg = get_time_segment(t_hr);
            double v = vmax * (seg <= (int)time_segments_sigma.size()-1 ? time_segments_sigma[seg] : 1.0); // m/s
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
            time += serve_truck[to] * 60; // serve_truck is in minutes, convert to seconds
            customers_since_last_depot.push_back(to);
        }
        visit_times[to] = time; // record departure from node 'to'
        // If we reach depot (except at start), check duration from leaving each customer to depot
        if (to == 0 && k != 1) {
            for (int cust : customers_since_last_depot) {
                // Duration from leaving customer to returning to depot
                double duration_min = (time - visit_times[cust]) / 60.0; // in minutes
                if (deadline_feasible && (duration_min > deadline[cust] + 1e-8)) {
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
    bool feasible = true;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist = distance_matrix[from][to]; // meters
        double v = v_fly_drone; // assume constant speed for simplicity
        if (v <= 1e-8) v = v_fly_drone;
        double time = dist / v; // seconds
        time += height / v_take_off; // take-off time
        time += height / v_landing; // landing time
        // Energy consumption model: power = beta * weight + gamma
        double power = power_beta * (current_weight) + power_gamma; // watts
        total_energy += power * time; // energy in joules
        if (total_energy > E + 1e-8) feasible = false;
        if (to != 0) current_weight += demand[to]; // add payload when delivering
        else {
            current_weight = 0; // reset weight when returning to depot
            total_energy = 0; // reset energy (charged at depot)
        }
    }
    return make_pair(total_energy, feasible);
}

pair<double, bool> compute_drone_route_time(const vi& route) {
    double time = 0; // seconds
    bool deadline_feasible = true;
    vector<double> visit_times(route.size(), 0.0); // visit_times[k]: time (seconds) when node route[k] is visited
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
            time += serve_drone[to] * 60; // serve_drone is in minutes, convert to seconds
            customers_since_last_depot.push_back(to);
        }
        visit_times[to] = time;
        // If we reach depot (except at start), check duration from leaving each customer to depot
        if (to == 0 && k != 1) {
            for (int cust : customers_since_last_depot) {
                double duration_min = (time - visit_times[cust]) / 60.0; // in minutes
                if (deadline_feasible && (duration_min > deadline[cust] + 1e-8)) {
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
    int total_demand = 0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dh) return make_pair(time, false); // exceeded capacity
        }
    }
    return make_pair(time, true);
}

pair<double, bool> check_drone_route_feasibility(const vi& route) {
    // Check deadlines
    auto [time, deadline_feasible] = compute_drone_route_time(route);
    if (!deadline_feasible) return make_pair(time, false);
    // Check capacity (reset at depot)
    int total_demand = 0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dd) return make_pair(time, false); // exceeded capacity
        }
    }
    // Check energy
    auto [total_energy, feasible_energy] = compute_drone_route_energy(route);
    if (!feasible_energy) return make_pair(time, false); // exceeded energy in a segment
    return make_pair(time, true);
}

Solution generate_initial_solution(){
    Solution sol;
    sol.truck_routes.resize(h);
    sol.drone_routes.resize(d);
    // 1. Compute initial savings for all pairs (for both truck and drone eligible)
    struct Saving {
        int i, j;
        double saving;
        bool for_drone; // true if both i and j are drone-eligible
        bool operator<(const Saving& other) const {
            // Sort descending by saving, then by i, j, for_drone for uniqueness
            if (fabs(saving - other.saving) > 1e-8) return saving > other.saving;
            if (i != other.i) return i < other.i;
            if (j != other.j) return j < other.j;
            return for_drone < other.for_drone;
        }
    };

    set<Saving> savings;
    
    for (int i = 1; i <= n; ++i) {
        for (int j = i+1; j <= n; ++j) {
            // Truck savings (simulate route merge)
            vi route1 = {0, i, 0};
            vi route2 = {0, j, 0};
            // Truck savings: if at least one must be truck-served, compute truck savings
            if (served_by_drone[i] == 0 || served_by_drone[j] == 0) {
                double time1, time2;
                if (served_by_drone[i] == 0) {
                    tie(time1, ignore) = check_truck_route_feasibility(route1);
                } else {
                    tie(time1, ignore) = check_drone_route_feasibility(route1);
                }
                if (served_by_drone[j] == 0) {
                    tie(time2, ignore) = check_truck_route_feasibility(route2);
                } else {
                    tie(time2, ignore) = check_drone_route_feasibility(route2);
                }
                double time_sep = time1 + time2;
                vi merged = {0, i, j, 0};
                auto [time_merged, feasible_merged] = check_truck_route_feasibility(merged);
                double s_truck = time_sep - time_merged;
                if (feasible_merged) {
                    savings.insert({i, j, s_truck, false});
                }
            }
            // Drone savings: only if both are drone-eligible
            if (served_by_drone[i] == 1 && served_by_drone[j] == 1) {
                auto [time1_d, feasible1_d] = check_drone_route_feasibility(route1);
                auto [time2_d, feasible2_d] = check_drone_route_feasibility(route2);
                double time_sep_d = time1_d + time2_d;
                vi merged_d = {0, i, j, 0};
                auto [time_merged_d, feasible_merged_d] = check_drone_route_feasibility(merged_d);
                if (feasible_merged_d) {
                    double s_drone = time_sep_d - time_merged_d;
                    savings.insert({i, j, s_drone, true});
                }
            }
        }
    }
    // No need to sort, std::set keeps savings sorted descending by saving
    // 3. Start with each customer in its own route (truck or drone)
    vector<vi> truck_routes_init, drone_routes_init;
    vector<int> route_type(n+1, -1); // 0: truck, 1: drone
    vector<int> route_idx(n+1, -1); // which route this customer is in
    int truck_count = 0, drone_count = 0;
    for (int i = 1; i <= n; ++i) {
        if (served_by_drone[i]) {
            drone_routes_init.push_back({0, i, 0});
            route_type[i] = 1;
            route_idx[i] = drone_count++;
        } else {
            truck_routes_init.push_back({0, i, 0});
            route_type[i] = 0;
            route_idx[i] = truck_count++;
        }
    }
    // 4. Merge routes by savings, respecting constraints (capacity, energy, etc.)
    while (!savings.empty()) {
        bool merged_any = false;
        for (auto it = savings.begin(); it != savings.end(); ++it) {
            const auto& s = *it;
            int i = s.i, j = s.j;
            if (route_type[i] == -1 || route_type[j] == -1) continue; // Already merged
            int idx_i = route_idx[i], idx_j = route_idx[j];
            if (idx_i == idx_j) continue; //already in same route
            // Only merge same type
            if (route_type[i] != route_type[j]) continue;
            auto& routes = (route_type[i] == 0) ? truck_routes_init : drone_routes_init;
            auto compute_route_feas = (route_type[i] == 0)
                ? [](const vi& r) { return check_truck_route_feasibility(r); }
                : [](const vi& r) { return check_drone_route_feasibility(r); };
            auto& route_i = routes[idx_i];
            auto& route_j = routes[idx_j];
            // Only merge if i is at end and j at start (or vice versa)
            if (route_i[route_i.size()-2] == i && route_j[1] == j) {
                // Merge route_i and route_j
                vi merged = route_i;
                merged.pop_back();
                for (int k = 1; k < route_j.size(); ++k) merged.push_back(route_j[k]);
                merged.push_back(0);

                double time_sep = compute_route_feas(route_i).first + compute_route_feas(route_j).first;
                // Accept merge if saving is positive and merged route is feasible
                auto [merged_time, merged_feas] = compute_route_feas(merged);
                if ((time_sep - merged_time > 1e-8) && merged_feas) {
                    routes.push_back(merged);
                    int new_idx = routes.size() - 1;
                    for (int k = 1; k < route_i.size()-1; ++k) route_type[route_i[k]] = -1;
                    for (int k = 1; k < route_j.size()-1; ++k) route_type[route_j[k]] = -1;
                    for (int k = 1; k < merged.size()-1; ++k) {
                        route_type[merged[k]] = (route_type[i] == 0 ? 0 : 1);
                        route_idx[merged[k]] = new_idx;
                    }
                    // Remove all savings involving merged customers
                    set<int> merged_customers(merged.begin()+1, merged.end()-1);
                    for (auto it2 = savings.begin(); it2 != savings.end(); ) {
                        if (merged_customers.count(it2->i) || merged_customers.count(it2->j)) {
                            it2 = savings.erase(it2);
                        } else {
                            ++it2;
                        }
                    }
                    // Recompute savings for all pairs in merged route
                    for (int a : merged_customers) {
                        int idx_a = route_idx[a];
                        auto& route_a = routes[idx_a];
                        // Case 1: a at end, b at start (route_a + route_b)
                        if (route_a.size() > 2 && route_a[route_a.size()-2] == a) {
                            for (int b = 1; b <= n; ++b) {
                                if (a == b || route_type[b] != route_type[a] || route_type[b] == -1) continue;
                                int idx_b = route_idx[b];
                                if (idx_a == idx_b) continue;
                                auto& route_b = routes[idx_b];
                                if (route_b.size() > 2 && route_b[1] == b) {
                                    vi merged2 = route_a;
                                    merged2.pop_back();
                                    for (int k = 1; k < route_b.size(); ++k) merged2.push_back(route_b[k]);
                                    merged2.push_back(0);
                                    auto res_a = compute_route_feas(route_a);
                                    auto res_b = compute_route_feas(route_b);
                                    auto res_merged = compute_route_feas(merged2);
                                    double time_sep2 = res_a.first + res_b.first;
                                    double time_merged2 = res_merged.first;
                                    double s_new = time_sep2 - time_merged2;
                                    if (s_new > 1e-8 && res_merged.second) {
                                        savings.insert({route_a[route_a.size()-2], route_b[1], s_new, (route_type[a] == 1)});
                                    }
                                }
                            }
                        }
                        // Case 2: a at start, b at end (route_b + route_a)
                        if (route_a.size() > 2 && route_a[1] == a) {
                            for (int b = 1; b <= n; ++b) {
                                if (a == b || route_type[b] != route_type[a] || route_type[b] == -1) continue;
                                int idx_b = route_idx[b];
                                if (idx_a == idx_b) continue;
                                auto& route_b = routes[idx_b];
                                if (route_b.size() > 2 && route_b[route_b.size()-2] == b) {
                                    vi merged2 = route_b;
                                    merged2.pop_back();
                                    for (int k = 1; k < route_a.size(); ++k) merged2.push_back(route_a[k]);
                                    merged2.push_back(0);
                                    auto res_b = compute_route_feas(route_b);
                                    auto res_a = compute_route_feas(route_a);
                                    auto res_merged = compute_route_feas(merged2);
                                    double time_sep2 = res_b.first + res_a.first;
                                    double time_merged2 = res_merged.first;
                                    double s_new = time_sep2 - time_merged2;
                                    if (s_new > 1e-8 && res_merged.second) {
                                        savings.insert({route_b[route_b.size()-2], route_a[1], s_new, (route_type[a] == 1)});
                                    }
                                }
                            }
                        }
                    }
                    merged_any = true;
                    break;
                }
            } else if (route_j[route_j.size()-2] == j && route_i[1] == i) {
                vi merged = route_j;
                merged.pop_back();
                for (int k = 1; k < route_i.size(); ++k) merged.push_back(route_i[k]);
                merged.push_back(0);
                auto res_j = compute_route_feas(route_j);
                auto res_i = compute_route_feas(route_i);
                auto res_merged = compute_route_feas(merged);
                double time_sep = res_j.first + res_i.first;
                double time_merged = res_merged.first;
                if ((time_sep - time_merged > 1e-8) && res_merged.second) {
                    routes.push_back(merged);
                    int new_idx = routes.size() - 1;
                    for (int k = 1; k < route_j.size()-1; ++k) route_type[route_j[k]] = -1;
                    for (int k = 1; k < route_i.size()-1; ++k) route_type[route_i[k]] = -1;
                    for (int k = 1; k < merged.size()-1; ++k) {
                        route_type[merged[k]] = (route_type[i] == 0 ? 0 : 1);
                        route_idx[merged[k]] = new_idx;
                    }
                    set<int> merged_customers(merged.begin()+1, merged.end()-1);
                    for (auto it2 = savings.begin(); it2 != savings.end(); ) {
                        if (merged_customers.count(it2->i) || merged_customers.count(it2->j)) {
                            it2 = savings.erase(it2);
                        } else {
                            ++it2;
                        }
                    }
                    for (int a : merged_customers) {
                        int idx_a = route_idx[a];
                        auto& route_a = routes[idx_a];
                        // Case 1: a at end, b at start (route_a + route_b)
                        if (route_a.size() > 2 && route_a[route_a.size()-2] == a) {
                            for (int b = 1; b <= n; ++b) {
                                if (a == b || route_type[b] != route_type[a] || route_type[b] == -1) continue;
                                int idx_b = route_idx[b];
                                if (idx_a == idx_b) continue;
                                auto& route_b = routes[idx_b];
                                if (route_b.size() > 2 && route_b[1] == b) {
                                    vi merged2 = route_a;
                                    merged2.pop_back();
                                    for (int k = 1; k < route_b.size(); ++k) merged2.push_back(route_b[k]);
                                    merged2.push_back(0);
                                    auto res_a = compute_route_feas(route_a);
                                    auto res_b = compute_route_feas(route_b);
                                    auto res_merged2 = compute_route_feas(merged2);
                                    double time_sep2 = res_a.first + res_b.first;
                                    double time_merged2 = res_merged2.first;
                                    double s_new = time_sep2 - time_merged2;
                                    if (s_new > 1e-8 && res_merged2.second) {
                                        savings.insert({route_a[route_a.size()-2], route_b[1], s_new, (route_type[a] == 1)});
                                    }
                                }
                            }
                        }
                        // Case 2: a at start, b at end (route_b + route_a)
                        if (route_a.size() > 2 && route_a[1] == a) {
                            for (int b = 1; b <= n; ++b) {
                                if (a == b || route_type[b] != route_type[a] || route_type[b] == -1) continue;
                                int idx_b = route_idx[b];
                                if (idx_a == idx_b) continue;
                                auto& route_b = routes[idx_b];
                                if (route_b.size() > 2 && route_b[route_b.size()-2] == b) {
                                    vi merged2 = route_b;
                                    merged2.pop_back();
                                    for (int k = 1; k < route_a.size(); ++k) merged2.push_back(route_a[k]);
                                    merged2.push_back(0);
                                    auto res_b = compute_route_feas(route_b);
                                    auto res_a = compute_route_feas(route_a);
                                    auto res_merged2 = compute_route_feas(merged2);
                                    double time_sep2 = res_b.first + res_a.first;
                                    double time_merged2 = res_merged2.first;
                                    double s_new = time_sep2 - time_merged2;
                                    if (s_new > 1e-8 && res_merged2.second) {
                                        savings.insert({route_b[route_b.size()-2], route_a[1], s_new, (route_type[a] == 1)});
                                    }
                                }
                            }
                        }
                    }
                    merged_any = true;
                    break;
                }
            }
        }
        if (!merged_any) break;
    }
    // 5. Assign merged routes to solution (limit to h trucks, d drones)
    int t = 0, dr = 0;
    for (auto& r : truck_routes_init) {
        if (r.size() > 2 && t < h) sol.truck_routes[t++] = r;
    }
    for (auto& r : drone_routes_init) {
        if (r.size() > 2 && dr < d) sol.drone_routes[dr++] = r;
    }
    // Fill unused routes with depot only
    for (; t < h; ++t) sol.truck_routes[t] = {0};
    for (; dr < d; ++dr) sol.drone_routes[dr] = {0};
    return sol;
}

void print_solution(const Solution& sol) {
    cout << "Truck Routes:\n";
    for (int i = 0; i < (int)sol.truck_routes.size(); ++i) {
        cout << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            cout << node << " ";
        }
        cout << "\n";
    }
    cout << "Drone Routes:\n";
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
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
    auto distance_matrix = compute_distance_matrices(loc);
    update_served_by_drone();
    // Debug: print served_by_drone
    cout << "Served by Drone:\n";
    for (int i = 1; i <= n; ++i) {
        cout << "Customer " << i << ": " << (served_by_drone[i]
                ? "Yes" : "No") << "\n";
    }
    cout << "Distance Matrix (Euclidean, meters):\n";
    // Print column headers
    cout << std::setw(8) << " ";
    for (int j = 0; j <= n; ++j) {
        cout << std::setw(8) << j;
    }
    cout << "\n";
    for (int i = 0; i <= n; ++i) {
        cout << std::setw(8) << i;
        for (int j = 0; j <= n; ++j) {
            cout << std::setw(8) << std::fixed << std::setprecision(1) << distance_matrix[i][j];
        }
        cout << "\n";
    }
    Solution sol = generate_initial_solution();
    print_solution(sol);
}

void output(){

}

int main(){
    ios::sync_with_stdio(0);
    freopen("50.10.2.txt","r",stdin);
    freopen("output.txt","w",stdout);
    cin.tie(0);
    cout.tie(0);
    auto start = chrono::high_resolution_clock::now();
    input();
    solve();
    output();
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, std::milli> duration = end - start;
    cerr << "Time: " << duration.count() << " ms" << endl;
    return 0;
}