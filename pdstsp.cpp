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

int n, h, d; //number of customers, number of trucks, number of drones
vpi loc; // loc[i]: location (x, y) of customer i, if i = 0, it is depot
vd serve_truck, serve_drone; // time taken by truck and drone to serve each customer (hours)
vi served_by_drone; //whether each customer can be served by drone or not, 1 if yes, 0 if no
vd deadline; //customer deadlines
vi demand; // demand[i]: demand of customer i
int Dh, vmax; //truck capacities (for all trucks)
int L; //number of time segments in a day
vd time_segments_sigma; //sigma (velocity coefficient) for each time segments
int Dd, E; //drone's weight and energy capacities (for all drones)
double v_fly_drone, v_take_off, v_landing, height; // maximum speed of the drone
double power_beta, power_gamma; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    double total_cost; //total cost of the solution
};

void input(){
    cin >> n >> h >> d;
    served_by_drone.resize(n+5);
    serve_truck.resize(n+5);
    serve_drone.resize(n+5);
    deadline.resize(n+5);
    demand.resize(n+5);
    loc.resize(n+5);
    distance_matrix.resize(n+5, vd(n+5));
    FOR(i,0,n){
        int x, y;
        cin >> x >> y;
        loc[i] = mp(x,y);
    }
    FOR(i,1,n){
        cin >> serve_truck[i];
    }
    FOR(i,1,n){
        cin >> serve_drone[i];
    }
    FOR(i,1,n){
        cin >> served_by_drone[i];
    }
    FOR(i,1,n){
        cin >> deadline[i];
    }
    FOR(i,1,n){
        cin >> demand[i];
    }
    cin >> Dh >> vmax;
    cin >> L;
    time_segments_sigma.resize(L);
    FOR(i,1,L){
        cin >> time_segments_sigma[i];
    }
    cin >> Dd >> E;
    cin >> power_beta >> power_gamma;
    cin >> v_fly_drone >> v_take_off >> v_landing >> height;
}

// Returns pair of distance matrices: {truck (Manhattan), drone (Euclidean)}
vvd compute_distance_matrices(vpi loc) {
    int n = loc.size() - 1; // assuming loc[0] is depot
    vector<vector<double>> distance_matrix(n+1, vd(n+1));
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distance_matrix[i][j] = sqrt((loc[i].first - loc[j].first) * (loc[i].first - loc[j].first)
                                             + (loc[i].second - loc[j].second) * (loc[i].second - loc[j].second)); // Euclidean
        }
    }
    return {distance_matrix};
}

void solve(){
    auto distance_matrix = compute_distance_matrices(loc);
    cout << "Distance Matrix:\n";
    for (const auto& row : distance_matrix) {
        for (const auto& dist : row) {
            cout << dist << " ";
        }
        cout << "\n";
    }
}

Solution generate_initial_solution(){
    Solution sol;
    sol.truck_routes.resize(h);
    sol.drone_routes.resize(d);
    for (int i = 0; i < h; ++i) sol.truck_routes[i].push_back(0);
    for (int i = 0; i < d; ++i) sol.drone_routes[i].push_back(0);
    // current positions of trucks and drones (starting at depot)
    vi truck_positions(h, 0); // index of current customer (0 = depot)
    vi drone_positions(d, 0); // index of current customer (0 = depot)
    // current loads of trucks and drones
    vi truck_loads(h, 0);
    vi drone_loads(d, 0);
    // current times of trucks and drones
    vd truck_times(h, 0);
    vd drone_times(d, 0);
    // current energy levels of drones
    vd drone_energies(d, E);
    // customers served or not
    vi served(n+1, 0);
    // Assign customers to trucks and drones
    
    // for each customer, find top-k nearest trucks and drones
    random_device rd;
    mt19937 gen(rd());
    int k = 3; // top-k
    for(int customer = 1; customer <= n; customer++){
        if (served_by_drone[customer] == 1) {
            // Find top-k nearest drones
            vector<pair<double, int>> drone_candidates;
            vector<pair<double, int>> truck_candidates;
            vector<bool> drone_feasible(d, true);
            for (int drone = 0; drone < d; drone++) {
                int from = drone_positions[drone];
                double dist = distance_matrix[from][customer];
                double fly_time = dist / v_fly_drone;
                double takeoff_time = v_take_off > 0 ? height / v_take_off : 0;
                double landing_time = v_landing > 0 ? height / v_landing : 0;
                double total_time = fly_time + takeoff_time + landing_time;
                drone_candidates.push_back({total_time, drone});

            }
            sort(drone_candidates.begin(), drone_candidates.end());
            int num_candidates = min(k, (int)drone_candidates.size());
            bool assigned = false;
            for (int idx = 0; idx < num_candidates; ++idx) {
                int candidate_drone = drone_candidates[idx].second;
                // Estimate energy required for this delivery
                int from = drone_positions[candidate_drone];
                double dist = distance_matrix[from][customer];
                double payload = demand[customer];
                double energy_needed = dist / v_fly_drone * (power_beta * payload + power_gamma); // simplistic
                if (drone_energies[candidate_drone] >= energy_needed) {
                    sol.drone_routes[candidate_drone].push_back(customer);
                    drone_positions[candidate_drone] = customer;
                    drone_energies[candidate_drone] -= energy_needed;
                    assigned = true;
                    break;
                }
            }
            // If not assigned, skip drone assignment (could assign to truck or handle differently)
        } else {
            // Find top-k nearest trucks
            vector<pair<double, int>> truck_candidates;
            for (int truck = 0; truck < h; truck++) {
                int from = truck_positions[truck];
                double dist = distance_matrix[from][customer]; // Use matrix for truck too (if Manhattan, precalc)
                truck_candidates.push_back({dist, truck});
            }
            sort(truck_candidates.begin(), truck_candidates.end());
            int num_candidates = min(k, (int)truck_candidates.size());
            std::uniform_int_distribution<> dis(0, num_candidates - 1);
            int chosen = truck_candidates[dis(gen)].second;
            sol.truck_routes[chosen].push_back(customer);
            truck_positions[chosen] = customer;
        }
    }
    return sol;
}

void output(){

}

int main(){
    ios::sync_with_stdio(0);
    freopen("input.txt","r",stdin);
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