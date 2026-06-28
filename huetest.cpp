// huetest.cpp — So sánh chất lượng nghiệm: truck nhiều trip vs truck 1 trip
//
// Usage: ./huetest <results_dir>
//
// Expects files in <results_dir>/:
//   <instance>_multi_1.txt ... <instance>_multi_5.txt   (tabubu.cpp, Dh=300)
//   <instance>_1trip_1.txt ... <instance>_1trip_5.txt   (tabubu_truck1trip.cpp, Dh=300)
//
// Output: CSV đến stdout gồm các cột:
//   instance, multi_feasible/5, 1trip_feasible/5,
//   multi_best_makespan, 1trip_best_makespan, delta_best(multi-1trip), delta_best_pct,
//   multi_avg_makespan,  1trip_avg_makespan,  delta_avg(multi-1trip),  delta_avg_pct
//
// delta < 0  =>  multi-trip TỐT HƠN (makespan nhỏ hơn)
// delta > 0  =>  1-trip TỐT HƠN

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <cstdio>

using namespace std;
namespace fs = std::filesystem;

// ---- Kết quả 1 lần chạy ----
struct RunResult {
    bool valid     = false;   // file đọc được
    bool feasible  = false;   // Final solution feasibility: FEASIBLE
    double makespan = numeric_limits<double>::quiet_NaN(); // Improved solution cost
};

RunResult parse_result_file(const string& path) {
    RunResult r;
    ifstream f(path);
    if (!f) return r;
    r.valid = true;
    string line;
    while (getline(f, line)) {
        if (line.rfind("Final solution feasibility:", 0) == 0) {
            // FEASIBLE nhưng không phải INFEASIBLE
            r.feasible = (line.find("FEASIBLE") != string::npos &&
                          line.find("INFEASIBLE") == string::npos);
        }
        if (line.rfind("Improved solution cost:", 0) == 0) {
            try { r.makespan = stod(line.substr(line.find(':') + 1)); }
            catch (...) {}
        }
    }
    return r;
}

// ---- Thống kê tổng hợp cho 1 solver trên 1 instance ----
struct SolverStats {
    int feasible_count  = 0;
    int total_runs      = 0;
    double best_feasible = numeric_limits<double>::infinity();
    double sum_feasible  = 0.0;
    double best_any      = numeric_limits<double>::infinity(); // kể cả infeasible
};

static string fmtd(double v) {
    if (!isfinite(v)) return "N/A";
    char buf[64]; snprintf(buf, sizeof(buf), "%.2f", v); return buf;
}
static string fmtpct(double num, double base) {
    if (!isfinite(num) || !isfinite(base) || base < 1e-8) return "N/A";
    char buf[64]; snprintf(buf, sizeof(buf), "%.2f%%", (num / base) * 100.0); return buf;
}
static string fmtdiff(double a, double b) {
    if (!isfinite(a) || !isfinite(b)) return "N/A";
    char buf[64]; snprintf(buf, sizeof(buf), "%.2f", a - b); return buf;
}
static string fmtdiffpct(double a, double b) {
    // (a-b)/b * 100 — base là 1trip để thấy multi thay đổi bao nhiêu %
    if (!isfinite(a) || !isfinite(b) || b < 1e-8) return "N/A";
    char buf[64]; snprintf(buf, sizeof(buf), "%.2f%%", (a - b) / b * 100.0); return buf;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <results_dir>\n";
        cerr << "  Reads files named <instance>_multi_N.txt and <instance>_1trip_N.txt\n";
        return 1;
    }
    string dir = argv[1];

    // ---- Thu thập tất cả file ----
    // data[instance][solver] = danh sách kết quả (N lần chạy)
    map<string, map<string, vector<RunResult>>> data;

    for (auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() != ".txt") continue;
        string stem = entry.path().stem().string(); // tên không có .txt

        string solver, instance;
        auto pos_m = stem.rfind("_multi_");
        auto pos_t = stem.rfind("_1trip_");
        if (pos_m != string::npos) {
            solver   = "multi";
            instance = stem.substr(0, pos_m);
        } else if (pos_t != string::npos) {
            solver   = "1trip";
            instance = stem.substr(0, pos_t);
        } else {
            continue;
        }
        RunResult r = parse_result_file(entry.path().string());
        data[instance][solver].push_back(r);
    }

    if (data.empty()) {
        cerr << "Khong tim thay file nao phu hop trong: " << dir << "\n";
        cerr << "  Can ten dang: <instance>_multi_N.txt hoac <instance>_1trip_N.txt\n";
        return 1;
    }

    // ---- Tiêu đề CSV ----
    cout << "instance,"
         << "multi_feasible_ratio,1trip_feasible_ratio,"
         << "multi_best_makespan,1trip_best_makespan,"
         << "delta_best(multi-1trip),delta_best_pct,"
         << "multi_avg_makespan,1trip_avg_makespan,"
         << "delta_avg(multi-1trip),delta_avg_pct\n";

    for (auto& kv : data) {
        const string& instance = kv.first;
        auto& solvers = kv.second;
        auto compute_stats = [&](const string& s) -> SolverStats {
            SolverStats st;
            auto it = solvers.find(s);
            if (it == solvers.end()) return st;
            for (auto& r : it->second) {
                if (!r.valid) continue;
                st.total_runs++;
                if (isfinite(r.makespan) && r.makespan < st.best_any)
                    st.best_any = r.makespan;
                if (r.feasible && isfinite(r.makespan)) {
                    st.feasible_count++;
                    st.sum_feasible += r.makespan;
                    if (r.makespan < st.best_feasible)
                        st.best_feasible = r.makespan;
                }
            }
            return st;
        };

        SolverStats m = compute_stats("multi");
        SolverStats t = compute_stats("1trip");

        const int RUNS = 5;
        double m_ratio = (double)m.feasible_count / RUNS;
        double t_ratio = (double)t.feasible_count / RUNS;

        // Dùng best feasible nếu có, không thì best infeasible
        double m_best = (m.feasible_count > 0) ? m.best_feasible : m.best_any;
        double t_best = (t.feasible_count > 0) ? t.best_feasible : t.best_any;
        double m_avg  = (m.feasible_count > 0)
                        ? m.sum_feasible / m.feasible_count
                        : numeric_limits<double>::quiet_NaN();
        double t_avg  = (t.feasible_count > 0)
                        ? t.sum_feasible / t.feasible_count
                        : numeric_limits<double>::quiet_NaN();

        cout << instance            << ","
             << m_ratio             << "," << t_ratio             << ","
             << fmtd(m_best)        << "," << fmtd(t_best)        << ","
             << fmtdiff(m_best, t_best) << "," << fmtdiffpct(m_best, t_best) << ","
             << fmtd(m_avg)         << "," << fmtd(t_avg)         << ","
             << fmtdiff(m_avg, t_avg)   << "," << fmtdiffpct(m_avg, t_avg)   << "\n";
    }

    return 0;
}
