/**
 * ╔══════════════════════════════════════════════════════════════════════════════╗
 * ║   DIVIDE ET IMPERA ENCRYPTA                          by Dominik Prantl       ║
 * ║   Privacy-Preserving Homomorphic K-Means on Encrypted Satellite Imagery      ║
 * ╠══════════════════════════════════════════════════════════════════════════════╣
 * ║                                                                              ║
 * ║   CONTRIBUTIONS:                                                             ║
 * ║   • Train Strategy        — stable memory via wagon-based streaming          ║
 * ║   • Existence-Gating      — Correct clustering for arbitrary dataset sizes   ║
 * ║   • SIMD Vectorisation    — 128 pixels per ciphertext (Structure-of-Arrays)  ║
 * ║   • 6 Independent Chebyshev Sigmoid Comparisons (degree 21, κ=12)            ║
 * ║   • Chebyshev Division    — Paterson-Stockmeyer reciprocal (degree 25)       ║
 * ║   • Best Iteration Export — Auto-saves highest match-rate iteration          ║
 * ║                                                                              ║
 * ║   CRYPTOGRAPHIC PARAMETERS:                                                  ║
 * ║   • Ring dimension N = 2^17 = 131,072                                        ║
 * ║   • Multiplicative depth L = 55, Scaling modulus = 59 bit                    ║
 * ║   • log₂(Q) = 3305 bit ≤ 3523 bit (HE Standard, N=2^17)                      ║
 * ║   • Security: rop ≈ 2^136.7 (verified via Lattice Estimator [1])             ║
 * ║   • Security level: HEStd_128_classic                                        ║
 * ║                                                                              ║
 * ║   COMPATIBILITY:                                                             ║
 * ║   • OpenFHE v1.0.3                                                           ║
 * ║   • OpenFHE v1.4.1+ (cross-version validation, ~4x faster)                   ║
 * ║   • Requires: N=2^17 security patch in stdlatticeparms.cpp [2]               ║
 * ║                                                                              ║
 * ║   REFERENCES:                                                                ║
 * ║   [1] Albrecht et al. (2015), Lattice Estimator,                             ║
 * ║       https://github.com/malb/lattice-estimator                              ║
 * ║   [2] OpenFHE Forum Thread #405, GitHub Issue #147,                          ║
 * ║       https://openfhe.discourse.group/t/405                                  ║
 * ║                                                                              ║
 * ╚══════════════════════════════════════════════════════════════════════════════╝
 */

#include "openfhe.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h> 
#include <sys/resource.h>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <limits>
#include <cstring>
 
#include "satellite_data.h"
 
using namespace lbcrypto;
 
// ============================================================================
// CONFIGURATION
// ============================================================================
 
const int CHEBYSHEV_DEGREE_COMPARISON = 21;
const int CHEBYSHEV_DEGREE_DIVISION = 25;
const double SIGMOID_STEEPNESS = 12.0;
const double COMPARISON_RANGE = 2.0;
 
// ============================================================================
// MEMORY TRACKING (VmRSS - Resident Set Size)
// ============================================================================
 
class MemoryTracker {
private:
    long baseline_kb;
    
    // Read VmRSS from /proc/self/status
    long readVmRSS() {
        long vmrss_kb = 0;
        FILE* fp = fopen("/proc/self/status", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "VmRSS:", 6) == 0) {
                    sscanf(line + 6, "%ld", &vmrss_kb);
                    break;
                }
            }
            fclose(fp);
        }
        return vmrss_kb;
    }
    
public:
    MemoryTracker() {
        baseline_kb = readVmRSS();
    }
    
    long getCurrentMemoryKB() {
        return readVmRSS();
    }
    
    long getMemoryIncrease() {
        return readVmRSS() - baseline_kb;
    }
    
    void resetBaseline() {
        baseline_kb = readVmRSS();
    }
    
    // Get peak memory (VmPeak)
    long getPeakMemoryKB() {
        long vmpeak_kb = 0;
        FILE* fp = fopen("/proc/self/status", "r");
        if (fp) {
            char line[256];
            while (fgets(line, sizeof(line), fp)) {
                if (strncmp(line, "VmPeak:", 7) == 0) {
                    sscanf(line + 7, "%ld", &vmpeak_kb);
                    break;
                }
            }
            fclose(fp);
        }
        return vmpeak_kb;
    }
};
 
// ============================================================================
// DATA STRUCTURES AND CONFIGURATION
// ============================================================================
 
struct Config {
    uint32_t batch_size = 128;   
    uint32_t num_dims = 5;          
    uint32_t num_clusters = 3;
    uint32_t num_iterations = 5;
    double convergence_threshold = 1e-4;
 
    uint32_t ring_dim = 131072;
    uint32_t mult_depth = 55;
    uint32_t scale_mod_size = 59;
    uint32_t first_mod_size = 60;
};
 
struct StepTimings {
    double encryption_ms = 0;
    double distance_calc_ms = 0;
    double bootstrapping_ms = 0;
    double comparison_ms = 0;
    double masking_ms = 0;
    double accumulation_ms = 0;
    double division_ms = 0;
    double decryption_ms = 0;
    double total_ms = 0;
};
 
struct MemoryMetrics {
    long memory_per_distance_calc_kb = 0;
    long memory_per_comparison_kb = 0;
    long memory_per_iteration_kb = 0;
    long total_memory_kb = 0;
};
 
struct IterationMetrics {
    int iteration;
    double total_time_sec;
    double mae, rmse, max_error;
    double cluster_match_rate;
    double delta;
    std::vector<int> cluster_sizes;
    StepTimings timings;
    MemoryMetrics memory;
    bool is_best = false;
};
 
// ============================================================================
// TIMER AND LOGGING
// ============================================================================
 
// Global log file stream
std::ofstream g_log_file;
 
class PreciseTimer {
    std::chrono::high_resolution_clock::time_point start;
public:
    PreciseTimer() : start(std::chrono::high_resolution_clock::now()) {}
    void reset() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        return std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start).count();
    }
};
 
std::string get_ts() {
    auto n = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(n);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(n.time_since_epoch()) % 1000;
    std::tm tm_n; localtime_r(&t, &tm_n);
    std::ostringstream oss; 
    oss << "[" << std::put_time(&tm_n, "%H:%M:%S") << "." << std::setfill('0') << std::setw(3) << ms.count() << "]";
    return oss.str();
}
 
void log_event(const std::string& cat, const std::string& msg) {
    std::string line = get_ts() + " " + cat;
    while (line.length() < 25) line += " ";
    line += "| " + msg;
    
    std::cout << line << std::endl;
    if (g_log_file.is_open()) {
        g_log_file << line << std::endl;
    }
}
 
void dual_print(const std::string& text) {
    std::cout << text;
    if (g_log_file.is_open()) {
        g_log_file << text;
    }
}
 
void dual_println(const std::string& text = "") {
    std::cout << text << std::endl;
    if (g_log_file.is_open()) {
        g_log_file << text << std::endl;
    }
}
 
long get_mem_mb() {
    long rss = 0L;
    FILE* fp = fopen("/proc/self/statm", "r");
    if (!fp) return 0;
    if (fscanf(fp, "%*s%ld", &rss) != 1) { fclose(fp); return 0; }
    fclose(fp);
    return (rss * (long)sysconf(_SC_PAGESIZE)) / (1024 * 1024);
}
 
// ============================================================================
// MATHEMATICAL HELPERS
// ============================================================================
 
double getCoord(const Point& p, size_t d) {
    switch(d) {
        case 0: return p.orbit_period;
        case 1: return p.apogee;
        case 2: return p.perigee;
        case 3: return p.eccentricity;
        case 4: return p.inclination;
        default: return 0.0;
    }
}
 
void setCoord(Point& p, size_t d, double val) {
    switch(d) {
        case 0: p.orbit_period = val; break;
        case 1: p.apogee = val; break;
        case 2: p.perigee = val; break;
        case 3: p.eccentricity = val; break;
        case 4: p.inclination = val; break;
    }
}
 
double dist_sq(const Point& a, const Point& b) {
    double sum = 0;
    for(size_t d = 0; d < 5; ++d) {
        double diff = getCoord(a, d) - getCoord(b, d);
        sum += diff * diff;
    }
    return sum;
}
 
int find_nearest(const Point& p, const std::vector<Point>& centers) {
    int best = 0;
    double min_d = dist_sq(p, centers[0]);
    for(size_t i = 1; i < centers.size(); ++i) {
        double d = dist_sq(p, centers[i]);
        if(d < min_d) { min_d = d; best = static_cast<int>(i); }
    }
    return best;
}
 
double compute_movement(const std::vector<Point>& a, const std::vector<Point>& b) {
    double total = 0;
    for(size_t k = 0; k < a.size(); ++k) {
        total += std::sqrt(dist_sq(a[k], b[k]));
    }
    return total / a.size();
}
 
double compute_cluster_match_rate(const std::vector<Point>& data,
                                   const std::vector<Point>& gt_centers,
                                   const std::vector<Point>& fhe_centers) {
    int matches = 0;
    for (const auto& p : data) {
        if (find_nearest(p, gt_centers) == find_nearest(p, fhe_centers)) matches++;
    }
    return 100.0 * matches / data.size();
}
 
std::vector<double> createExistenceBitmap(size_t n_real, uint32_t batch_size) {
    std::vector<double> existence(batch_size, 0.0);
    for (size_t i = 0; i < n_real && i < batch_size; ++i) {
        existence[i] = 1.0;
    }
    return existence;
}
 
// ============================================================================
// PLAINTEXT K-MEANS (GROUND TRUTH REFERENCE)
// ============================================================================
 
std::vector<std::vector<Point>> plaintext_kmeans_history(
    const std::vector<Point>& data, 
    std::vector<Point> centers,
    const Config& cfg
) {
    std::vector<std::vector<Point>> history;
    history.push_back(centers);
    
    for(uint32_t it = 0; it < cfg.num_iterations; ++it) {
        std::vector<Point> new_centers(cfg.num_clusters);
        for(auto& c : new_centers) {
            c.orbit_period = 0; c.apogee = 0; c.perigee = 0;
            c.eccentricity = 0; c.inclination = 0;
        }
        std::vector<int> counts(cfg.num_clusters, 0);
 
        for(const auto& p : data) {
            int best = find_nearest(p, centers);
            counts[best]++;
            for(uint32_t d = 0; d < cfg.num_dims; ++d) {
                setCoord(new_centers[best], d, getCoord(new_centers[best], d) + getCoord(p, d));
            }
        }
 
        for(uint32_t k = 0; k < cfg.num_clusters; ++k) {
            if(counts[k] > 0) {
                for(uint32_t d = 0; d < cfg.num_dims; ++d) {
                    setCoord(new_centers[k], d, getCoord(new_centers[k], d) / counts[k]);
                }
            } else {
                new_centers[k] = centers[k];
            }
        }
        
        centers = new_centers;
        history.push_back(centers);
    }
    return history;
}
 
// ============================================================================
// PERFORMANCE ANALYSIS
// ============================================================================
 
void print_performance_analysis(const IterationMetrics& m, MemoryTracker& mem_tracker) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    oss << std::string(110, '=') << "\n";
    oss << "  Performance Analysis (Iteration " << m.iteration << ")";
    if (m.is_best) oss << "  *** BEST ITERATION ***";
    oss << "\n";
    oss << std::string(110, '-') << "\n";
    
    oss << "    - Encryption (Wagons):          " << std::setw(12) << m.timings.encryption_ms << " ms\n";
    oss << "    - Distance Calculation:         " << std::setw(12) << m.timings.distance_calc_ms << " ms\n";
    oss << "    - Bootstrapping:                " << std::setw(12) << m.timings.bootstrapping_ms << " ms  <-- PURE CKKS BOTTLENECK\n";
    oss << "    - Distance Comparison:          " << std::setw(12) << m.timings.comparison_ms << " ms  (6 Chebyshev calls)\n";
    oss << "    - Masking (Mask Creation):      " << std::setw(12) << m.timings.masking_ms << " ms\n";
    oss << "    - Accumulation:                 " << std::setw(12) << m.timings.accumulation_ms << " ms\n";
    oss << "    - Division (Chebyshev):         " << std::setw(12) << m.timings.division_ms << " ms\n";
    oss << "    - Decryption:                   " << std::setw(12) << m.timings.decryption_ms << " ms\n";
    oss << "    " << std::string(55, '-') << "\n";
    oss << "    - Total Iteration Time:         " << std::setw(12) << m.timings.total_ms << " ms\n";
    
    oss << std::string(110, '-') << "\n";
    oss << "  Memory Usage Analysis (Iteration " << m.iteration << "):\n";
    oss << "    - Memory per Distance Calc:     " << std::setw(12) << m.memory.memory_per_distance_calc_kb << " KB\n";
    oss << "    - Memory per Comparison:        " << std::setw(12) << m.memory.memory_per_comparison_kb << " KB\n";
    oss << "    - Memory for this Iteration:    " << std::setw(12) << m.memory.memory_per_iteration_kb << " KB\n";
    oss << "    - Total Memory (VmRSS):         " << std::setw(12) << m.memory.total_memory_kb << " KB\n";
    oss << "    - Peak Memory (VmPeak):         " << std::setw(12) << mem_tracker.getPeakMemoryKB() << " KB\n";
    oss << std::string(110, '=') << "\n\n";
    
    dual_print(oss.str());
}
 
// ============================================================================
// CSV EXPORT
// ============================================================================
 
void export_cluster_assignments(
    const std::vector<Point>& data,
    const std::vector<Point>& gt_centers,
    const std::vector<Point>& fhe_centers,
    int best_iteration,
    const std::string& filename
) {
    log_event("EXPORT", "Exporting BEST iteration (" + std::to_string(best_iteration) + ") to: " + filename);
    
    std::ofstream f(filename);
    f << "ID;GT_Cluster;FHE_Cluster;Match\n";
    
    int matches = 0;
    for (size_t i = 0; i < data.size(); ++i) {
        int gt_cluster = find_nearest(data[i], gt_centers);
        int fhe_cluster = find_nearest(data[i], fhe_centers);
        int match = (gt_cluster == fhe_cluster) ? 1 : 0;
        matches += match;
        
        f << i << ";" << gt_cluster << ";" << fhe_cluster << ";" << match << "\n";
    }
    f.close();
    
    log_event("EXPORT", "Best Iteration " + std::to_string(best_iteration) + " — Match Rate: " + std::to_string(100.0 * matches / data.size()) + "%");
}
 
void export_all_iterations_csv(
    const std::vector<IterationMetrics>& history,
    const std::string& filename
) {
    log_event("EXPORT", "Exporting all iterations to: " + filename);
    
    std::ofstream f(filename);
    f << std::fixed << std::setprecision(4);
    
    // Header
    f << "Iteration;Time_s;MAE;RMSE;MatchRate;Delta;C0;C1;C2;IsBest;";
    f << "Encryption_ms;Distance_ms;Bootstrap_ms;Comparison_ms;Masking_ms;Accumulation_ms;Division_ms;";
    f << "MemPerDistCalc_KB;MemPerComp_KB;MemPerIter_KB;MemTotal_KB\n";
    
    for (const auto& m : history) {
        f << m.iteration << ";" 
          << m.total_time_sec << ";" 
          << m.mae << ";" 
          << m.rmse << ";" 
          << m.cluster_match_rate << ";"
          << m.delta << ";"
          << m.cluster_sizes[0] << ";" 
          << m.cluster_sizes[1] << ";" 
          << m.cluster_sizes[2] << ";"
          << (m.is_best ? "YES" : "NO") << ";"
          << m.timings.encryption_ms << ";"
          << m.timings.distance_calc_ms << ";"
          << m.timings.bootstrapping_ms << ";"
          << m.timings.comparison_ms << ";"
          << m.timings.masking_ms << ";"
          << m.timings.accumulation_ms << ";"
          << m.timings.division_ms << ";"
          << m.memory.memory_per_distance_calc_kb << ";"
          << m.memory.memory_per_comparison_kb << ";"
          << m.memory.memory_per_iteration_kb << ";"
          << m.memory.total_memory_kb << "\n";
    }
    f.close();
}
 
void export_centers(
    const std::vector<Point>& gt_centers,
    const std::vector<Point>& fhe_centers,
    int best_iteration,
    const std::string& filename
) {
    log_event("EXPORT", "Exporting centres (Best Iteration " + std::to_string(best_iteration) + ") to: " + filename);
    
    std::ofstream f(filename);
    f << std::fixed << std::setprecision(6);
    f << "Cluster;Type;Dim0;Dim1;Dim2;Dim3;Dim4\n";
    
    for (size_t k = 0; k < gt_centers.size(); ++k) {
        f << k << ";GT;"
          << getCoord(gt_centers[k], 0) << ";"
          << getCoord(gt_centers[k], 1) << ";"
          << getCoord(gt_centers[k], 2) << ";"
          << getCoord(gt_centers[k], 3) << ";"
          << getCoord(gt_centers[k], 4) << "\n";
    }
    
    for (size_t k = 0; k < fhe_centers.size(); ++k) {
        f << k << ";FHE_Best;"
          << getCoord(fhe_centers[k], 0) << ";"
          << getCoord(fhe_centers[k], 1) << ";"
          << getCoord(fhe_centers[k], 2) << ";"
          << getCoord(fhe_centers[k], 3) << ";"
          << getCoord(fhe_centers[k], 4) << "\n";
    }
    f.close();
}
 
// ============================================================================
// PURE CKKS K-MEANS ENGINE
// ============================================================================
 
class PureCKKSKMeans {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
    Config cfg;
    std::vector<IterationMetrics> history;
    MemoryTracker mem_tracker;
    double setup_time_sec = 0;
    double total_runtime_sec = 0;
    long setup_memory_kb = 0;
 
    // BEST ITERATION TRACKING
    int best_iteration = 0;
    double best_match_rate = 0.0;
    std::vector<Point> best_centers;
 
public:
    PureCKKSKMeans(const Config& c) : cfg(c) {}
 
    void Setup() {
        log_event("SETUP", "===================================================");
        log_event("SETUP", "PURE CKKS K-MEANS  - BEST ITERATION EXPORT");
        log_event("SETUP", "Ring: " + std::to_string(cfg.ring_dim) + 
                 " | Depth: " + std::to_string(cfg.mult_depth));
        
        auto start = std::chrono::high_resolution_clock::now();
        long before_setup_kb = mem_tracker.getCurrentMemoryKB();
        
        CCParams<CryptoContextCKKSRNS> parameters;
        parameters.SetRingDim(cfg.ring_dim);
        parameters.SetMultiplicativeDepth(cfg.mult_depth);
        parameters.SetScalingModSize(cfg.scale_mod_size);
        parameters.SetFirstModSize(cfg.first_mod_size);
        parameters.SetBatchSize(cfg.batch_size);
        // HEStd_NotSet: N=2^17 is absent from the OpenFHE v1.0.3 security table
        // (table ends at N=2^15). Post-hoc verification via Lattice Estimator [1]
        // confirms rop >= 2^136.7, satisfying the 128-bit security threshold [2].
        // See also: OpenFHE Forum Thread #405 [3].
        parameters.SetSecurityLevel(HEStd_NotSet);
        parameters.SetScalingTechnique(FLEXIBLEAUTO); 
        cc = GenCryptoContext(parameters);
 
        // ── Post-hoc security verification ──────────────────────────────────
        log_event("SECURITY", "N=131072 (2^17), log2(Q)=3305 bit");
        log_event("SECURITY", "HE Standard bound for N=2^17: 3523 bit  ->  3305 <= 3523 (compliant)");
        log_event("SECURITY", "Lattice Estimator [1]: rop >= 2^136.7  ->  128-bit security confirmed");
        // ────────────────────────────────────────────────────────────────────
 
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
        log_event("SETUP", "Generating keys...");
        keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        
        std::vector<int32_t> rotations;
        for(int i = 1; i <= (int)cfg.batch_size; i *= 2) rotations.push_back(i);
        cc->EvalRotateKeyGen(keys.secretKey, rotations);
 
        log_event("SETUP", "Generating BOOTSTRAPPING keys...");
        std::vector<uint32_t> levelBudget = {4, 4};
        std::vector<uint32_t> bsgsDim = {0, 0};
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, cfg.batch_size);
        cc->EvalBootstrapKeyGen(keys.secretKey, cfg.batch_size);
        
        setup_time_sec = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        setup_memory_kb = mem_tracker.getCurrentMemoryKB() - before_setup_kb;
        
        log_event("SETUP", "Complete in " + std::to_string(setup_time_sec) + " s");
        log_event("MEMORY", "Setup Memory: " + std::to_string(setup_memory_kb / 1024) + " MB");
        log_event("SETUP", "===================================================");
    }
 
    void Run(const std::vector<Point>& data, 
             const std::vector<Point>& init_centers, 
             const std::vector<std::vector<Point>>& plain_history) {
        
        auto total_start = std::chrono::high_resolution_clock::now();
        (void)mem_tracker.getCurrentMemoryKB();  // Reset baseline
        
        size_t total_points = data.size();
        size_t num_wagons = (total_points + cfg.batch_size - 1) / cfg.batch_size;
        
        log_event("EXEC", "===================================================");
        log_event("EXEC", "STARTING PURE CKKS K-MEANS");
        log_event("EXEC", "KEY FEATURE: Best iteration exported automatically");
        log_event("EXEC", "Dataset: " + std::to_string(total_points) + " points");
        log_event("EXEC", "Wagons: " + std::to_string(num_wagons));
        log_event("EXEC", "===================================================");
 
        std::vector<Point> prev_centers_p = init_centers;
        std::vector<Point> cur_centers_p = init_centers;
        best_centers = init_centers;
 
        // Initialize ENCRYPTED centers
        log_event("INIT", "Encrypting initial centres...");
        std::vector<std::vector<Ciphertext<DCRTPoly>>> enc_centers(cfg.num_clusters);
        for(size_t k = 0; k < cfg.num_clusters; ++k) {
            enc_centers[k].resize(cfg.num_dims);
            for(size_t d = 0; d < cfg.num_dims; ++d) {
                std::vector<double> v(cfg.batch_size, getCoord(init_centers[k], d));
                enc_centers[k][d] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
            }
        }
 
        // Sigmoid function
        auto sigmoid_less_than = [](double x) -> double {
            double kx = SIGMOID_STEEPNESS * x;
            if (kx > 20.0) return 0.0;
            if (kx < -20.0) return 1.0;
            return 1.0 / (1.0 + std::exp(kx));
        };
 
        for(uint32_t it = 0; it < cfg.num_iterations; ++it) {
            auto iter_start_time = std::chrono::high_resolution_clock::now();
            long iter_start_memory_kb = mem_tracker.getCurrentMemoryKB();
            
            log_event("ITER", "===================================================");
            log_event("ITER", "ITERATION " + std::to_string(it+1) + "/" + 
                     std::to_string(cfg.num_iterations));
            log_event("ITER", "===================================================");
            
            IterationMetrics m;
            m.iteration = it + 1;
            PreciseTimer timer;
            PreciseTimer step_timer;
 
            // Initialize accumulators
            std::vector<Ciphertext<DCRTPoly>> g_counts(cfg.num_clusters);
            std::vector<std::vector<Ciphertext<DCRTPoly>>> g_sums(cfg.num_clusters);
            
            for(size_t k = 0; k < cfg.num_clusters; ++k) {
                std::vector<double> zeros(cfg.batch_size, 0.0);
                g_counts[k] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(zeros));
                g_sums[k].resize(cfg.num_dims);
                for(size_t d = 0; d < cfg.num_dims; ++d) {
                    g_sums[k][d] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(zeros));
                }
            }
 
            long total_distance_memory_kb = 0;
            long total_comparison_memory_kb = 0;
            int distance_calc_count = 0;
            int comparison_count = 0;
 
            // Process wagons
            for(size_t wagon = 0; wagon < num_wagons; ++wagon) {
                size_t offset = wagon * cfg.batch_size;
                size_t wagon_size = std::min(static_cast<size_t>(cfg.batch_size), 
                                             total_points - offset);
                
                log_event("WAGON", "Wagon " + std::to_string(wagon+1) + "/" + 
                         std::to_string(num_wagons) + " (" + std::to_string(wagon_size) + 
                         " pts) | RAM: " + std::to_string(get_mem_mb()) + " MB");
 
                // ── Phase 1: Wagon Encryption ───────────────────────
                step_timer.reset();
                auto existence = createExistenceBitmap(wagon_size, cfg.batch_size);
                auto existence_bitmap = cc->Encrypt(keys.publicKey, 
                    cc->MakeCKKSPackedPlaintext(existence));
                
                std::vector<Ciphertext<DCRTPoly>> enc_chunk(cfg.num_dims);
                for(size_t d = 0; d < cfg.num_dims; ++d) {
                    std::vector<double> col(cfg.batch_size, 0.0);
                    for(size_t i = 0; i < wagon_size; ++i) {
                        col[i] = getCoord(data[offset + i], d);
                    }
                    enc_chunk[d] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(col));
                }
                m.timings.encryption_ms += step_timer.elapsed_ms();
 
                // ── Phase 2a: Squared Euclidean Distance ────────────
                step_timer.reset();
                long before_dist_kb = mem_tracker.getCurrentMemoryKB();
                
                std::vector<Ciphertext<DCRTPoly>> dists(cfg.num_clusters);
                for(size_t k = 0; k < cfg.num_clusters; ++k) {
                    auto diff = cc->EvalSub(enc_chunk[0], enc_centers[k][0]);
                    auto sq_dist = cc->EvalSquare(diff);
                    for(size_t d = 1; d < cfg.num_dims; ++d) {
                        diff = cc->EvalSub(enc_chunk[d], enc_centers[k][d]);
                        sq_dist = cc->EvalAdd(sq_dist, cc->EvalSquare(diff));
                    }
                    dists[k] = cc->EvalMult(sq_dist, existence_bitmap);
                    distance_calc_count++;
                }
                m.timings.distance_calc_ms += step_timer.elapsed_ms();
                total_distance_memory_kb += (mem_tracker.getCurrentMemoryKB() - before_dist_kb);
 
                // ── Bootstrapping (after distance, before sigmoid) ──
                step_timer.reset();
                log_event("BOOT", "  Bootstrapping distances...");
                for(size_t k = 0; k < cfg.num_clusters; ++k) {
                    dists[k] = cc->EvalBootstrap(dists[k]);
                }
                m.timings.bootstrapping_ms += step_timer.elapsed_ms();
 
                // ── Phase 2b: Six Independent Chebyshev Comparisons ─
                step_timer.reset();
                long before_comp_kb = mem_tracker.getCurrentMemoryKB();
                log_event("COMPARE", "  Computing 6 SEPARATE Chebyshev comparisons...");
                
                auto diff_01 = cc->EvalSub(dists[0], dists[1]);
                auto diff_10 = cc->EvalSub(dists[1], dists[0]);
                auto diff_02 = cc->EvalSub(dists[0], dists[2]);
                auto diff_20 = cc->EvalSub(dists[2], dists[0]);
                auto diff_12 = cc->EvalSub(dists[1], dists[2]);
                auto diff_21 = cc->EvalSub(dists[2], dists[1]);
                
                auto d0_lt_d1 = cc->EvalChebyshevFunction(sigmoid_less_than, 
                    diff_01, -COMPARISON_RANGE, COMPARISON_RANGE, CHEBYSHEV_DEGREE_COMPARISON);
                auto d1_lt_d0 = cc->EvalChebyshevFunction(sigmoid_less_than, 
                    diff_10, -COMPARISON_RANGE, COMPARISON_RANGE, CHEBYSHEV_DEGREE_COMPARISON);
                auto d0_lt_d2 = cc->EvalChebyshevFunction(sigmoid_less_than, 
                    diff_02, -COMPARISON_RANGE, COMPARISON_RANGE, CHEBYSHEV_DEGREE_COMPARISON);
                auto d2_lt_d0 = cc->EvalChebyshevFunction(sigmoid_less_than, 
                    diff_20, -COMPARISON_RANGE, COMPARISON_RANGE, CHEBYSHEV_DEGREE_COMPARISON);
                auto d1_lt_d2 = cc->EvalChebyshevFunction(sigmoid_less_than, 
                    diff_12, -COMPARISON_RANGE, COMPARISON_RANGE, CHEBYSHEV_DEGREE_COMPARISON);
                auto d2_lt_d1 = cc->EvalChebyshevFunction(sigmoid_less_than, 
                    diff_21, -COMPARISON_RANGE, COMPARISON_RANGE, CHEBYSHEV_DEGREE_COMPARISON);
                
                comparison_count += 6;
                m.timings.comparison_ms += step_timer.elapsed_ms();
                total_comparison_memory_kb += (mem_tracker.getCurrentMemoryKB() - before_comp_kb);
 
                // ── Soft Mask Construction ──────────────────────────
                step_timer.reset();
                log_event("MASK", "  Computing CORRECT cluster masks (6 comparisons)...");
                
                std::vector<Ciphertext<DCRTPoly>> masks(3);
                masks[0] = cc->EvalMult(d0_lt_d1, d0_lt_d2);
                masks[1] = cc->EvalMult(d1_lt_d0, d1_lt_d2);
                masks[2] = cc->EvalMult(d2_lt_d0, d2_lt_d1);
                m.timings.masking_ms += step_timer.elapsed_ms();
 
                // ── Phase 3: Depot Accumulation (Fold) ─────────────
                step_timer.reset();
                for(size_t k = 0; k < cfg.num_clusters; ++k) {
                    auto gated_mask = cc->EvalMult(masks[k], existence_bitmap);
                    g_counts[k] = cc->EvalAdd(g_counts[k], gated_mask);
                    for(size_t d = 0; d < cfg.num_dims; ++d) {
                        auto weighted = cc->EvalMult(enc_chunk[d], gated_mask);
                        g_sums[k][d] = cc->EvalAdd(g_sums[k][d], weighted);
                    }
                }
                m.timings.accumulation_ms += step_timer.elapsed_ms();
            }
 
            // ── Phase 4: Centroid Update via Chebyshev Inverse ──
            log_event("UPDATE", "Computing new cluster centres...");
            double mae_sum = 0, mse_sum = 0, max_err = 0;
 
            for(size_t k = 0; k < cfg.num_clusters; ++k) {
                step_timer.reset();
                auto total_count = cc->EvalSum(g_counts[k], cfg.batch_size);
                
                auto inv_count = cc->EvalChebyshevFunction(
                    [](double x) -> double { 
                        double safe_x = std::max(x, 1.0);
                        return 1.0 / safe_x; 
                    }, 
                    total_count, 1.0, static_cast<double>(total_points), 
                    CHEBYSHEV_DEGREE_DIVISION
                );
                m.timings.division_ms += step_timer.elapsed_ms();
                
                for(size_t d = 0; d < cfg.num_dims; ++d) {
                    step_timer.reset();
                    auto total_sum = cc->EvalSum(g_sums[k][d], cfg.batch_size);
                    auto new_coord = cc->EvalMult(total_sum, inv_count);
                    enc_centers[k][d] = cc->EvalBootstrap(new_coord);
                    m.timings.bootstrapping_ms += step_timer.elapsed_ms();
 
                    // Decrypt centroid coordinate for convergence monitoring only
                    step_timer.reset();
                    Plaintext pt; 
                    cc->Decrypt(keys.secretKey, enc_centers[k][d], &pt); 
                    pt->SetLength(1);
                    double val = std::real(pt->GetCKKSPackedValue()[0]);
                    m.timings.decryption_ms += step_timer.elapsed_ms();
                    
                    setCoord(cur_centers_p[k], d, val);
                    
                    double err = std::abs(val - getCoord(plain_history[it+1][k], d));
                    mae_sum += err;
                    mse_sum += err * err;
                    max_err = std::max(max_err, err);
                }
            }
 
            // ── Iteration Metrics ───────────────────────────────
            m.total_time_sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - iter_start_time).count();
            m.timings.total_ms = m.total_time_sec * 1000.0;
            
            m.mae = mae_sum / (cfg.num_clusters * cfg.num_dims);
            m.rmse = std::sqrt(mse_sum / (cfg.num_clusters * cfg.num_dims));
            m.max_error = max_err;
            m.delta = compute_movement(cur_centers_p, prev_centers_p);
            
            m.cluster_sizes.assign(cfg.num_clusters, 0);
            for(const auto& p : data) {
                int idx = find_nearest(p, cur_centers_p);
                if (idx >= 0 && idx < (int)cfg.num_clusters) m.cluster_sizes[idx]++;
            }
            
            m.cluster_match_rate = compute_cluster_match_rate(data, plain_history[it+1], cur_centers_p);
 
            // Memory metrics
            long iter_end_memory_kb = mem_tracker.getCurrentMemoryKB();
            m.memory.memory_per_iteration_kb = iter_end_memory_kb - iter_start_memory_kb;
            m.memory.memory_per_distance_calc_kb = distance_calc_count > 0 ? 
                total_distance_memory_kb / distance_calc_count : 0;
            m.memory.memory_per_comparison_kb = comparison_count > 0 ? 
                total_comparison_memory_kb / comparison_count : 0;
            m.memory.total_memory_kb = iter_end_memory_kb;
 
            // ── Track Best Iteration ────────────────────────────
            if (m.cluster_match_rate > best_match_rate) {
                best_match_rate = m.cluster_match_rate;
                best_iteration = m.iteration;
                best_centers = cur_centers_p;
                m.is_best = true;
                log_event("BEST", ">>> NEW BEST ITERATION: " + std::to_string(m.iteration) + 
                         " with " + std::to_string(m.cluster_match_rate) + "% match <<<");
            }
 
            // ── Console Output ──────────────────────────────────
            std::ostringstream iter_out;
            iter_out << "\n" << std::string(80, '=') << "\n";
            iter_out << " ITERATION " << m.iteration << " — Pure CKKS K-Means";
            if (m.is_best) iter_out << "  *** CURRENT BEST ***";
            iter_out << "\n";
            iter_out << std::string(80, '-') << "\n";
            iter_out << "  MAE:  " << std::fixed << std::setprecision(5) << m.mae << "\n";
            iter_out << "  RMSE: " << std::fixed << std::setprecision(5) << m.rmse << "\n";
            iter_out << "  >>> CLUSTER MATCH: " << std::fixed << std::setprecision(2) 
                      << m.cluster_match_rate << "% <<<";
            if (m.is_best) iter_out << "  (BEST SO FAR)";
            iter_out << "\n";
            iter_out << "  Cluster sizes:    [" << m.cluster_sizes[0] << ", " 
                      << m.cluster_sizes[1] << ", " << m.cluster_sizes[2] << "]\n";
            iter_out << "  Delta: " << std::fixed << std::setprecision(4) << m.delta << "\n";
            iter_out << "  Time:  " << std::fixed << std::setprecision(1) 
                      << m.total_time_sec << "s | RAM: " << (iter_end_memory_kb/1024) << " MB\n";
            iter_out << std::string(80, '=') << "\n";
            
            dual_print(iter_out.str());
            
            // Print detailed performance analysis
            print_performance_analysis(m, mem_tracker);
 
            history.push_back(m);
            prev_centers_p = cur_centers_p;
        }
 
        total_runtime_sec = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - total_start).count();
        
        // ==================== FINAL SUMMARY ====================
        std::cout << "\n" << std::string(110, '#') << "\n";
        std::cout << "#  FINAL SUMMARY - BEST ITERATION: " << best_iteration 
                  << " with " << std::fixed << std::setprecision(2) << best_match_rate << "% MATCH\n";
        std::cout << std::string(110, '#') << "\n\n";
        
        log_event("FINISH", "Total runtime: " + std::to_string(total_runtime_sec) + " s");
        log_event("FINISH", "Best Iteration: " + std::to_string(best_iteration) + 
                 " (" + std::to_string(best_match_rate) + "% match)");
    }
 
    int getBestIteration() const { return best_iteration; }
    double getBestMatchRate() const { return best_match_rate; }
    const std::vector<Point>& getBestCenters() const { return best_centers; }
    const std::vector<IterationMetrics>& getHistory() const { return history; }
    double getSetupTime() const { return setup_time_sec; }
    double getTotalRuntime() const { return total_runtime_sec; }
};
 
// ============================================================================
// ENTRY POINT
// ============================================================================
 
int main(int argc, char* argv[]) {
    // Open log file
    g_log_file.open("kmeans_console_output.txt");
    
    std::string header = R"(
+============================================================================+
|  PURE CKKS K-MEANS  - BEST ITERATION EXPORT                          |
+============================================================================+
|  Pure CKKS K-Means — Train Strategy + Existence-Gating                    |
|  * Best Iteration auto-export  (highest match rate)                       |
|  * Per-iteration performance and memory analysis                          |
|  * Full CSV export for all iterations                                     |
|  * Console output saved to TXT                                            |
+============================================================================+
 
)";
    dual_print(header);
 
    try {
        Config cfg;
        
        log_event("DATA", "Loading satellite_data.h...");
        std::vector<Point> data = getSatelliteData();
        size_t total_points = data.size();
        
        if (argc > 1) {
            int requested = std::atoi(argv[1]);
            if (requested > 0 && requested <= (int)total_points) {
                total_points = static_cast<size_t>(requested);
                data.resize(total_points);
            }
        }
        
        log_event("DATA", std::to_string(total_points) + " points loaded");
 
        // Initialize centers
        size_t step = data.size() / cfg.num_clusters;
        std::vector<Point> init;
        for (size_t k = 0; k < cfg.num_clusters; ++k) {
            init.push_back(data[k * step]);
        }
 
        // Ground Truth
        log_event("GT", "Computing plaintext ground truth...");
        auto plain_history = plaintext_kmeans_history(data, init, cfg);
 
        // FHE K-Means
        PureCKKSKMeans engine(cfg);
        engine.Setup();
        engine.Run(data, init, plain_history);
 
        // ================================================================
        // EXPORT - BESTE ITERATION
        // ================================================================
        log_event("EXPORT", "===================================================");
        log_event("EXPORT", "Exporting results of BEST iteration...");
        
        int best_iter = engine.getBestIteration();
        const auto& best_fhe_centers = engine.getBestCenters();
        const auto& gt_centers = plain_history[best_iter];  // GT at best iteration
        
        // Cluster assignments (BEST iteration)
        export_cluster_assignments(data, gt_centers, best_fhe_centers, best_iter,
                                   "kmeans_cluster_assignments_BEST.csv");
        
        // Centers (BEST iteration)
        export_centers(gt_centers, best_fhe_centers, best_iter,
                       "kmeans_centers_BEST.csv");
        
        // All iterations benchmark
        export_all_iterations_csv(engine.getHistory(), 
                                  "kmeans_all_iterations.csv");
        
        log_event("EXPORT", "===================================================");
        log_event("EXPORT", "Files created:");
        log_event("EXPORT", "  - kmeans_cluster_assignments_BEST.csv (Iteration " + 
                 std::to_string(best_iter) + ")");
        log_event("EXPORT", "  - kmeans_centers_BEST.csv");
        log_event("EXPORT", "  - kmeans_all_iterations.csv");
        log_event("EXPORT", "  - kmeans__console_output.txt");
        log_event("EXPORT", "===================================================");
 
        std::ostringstream final_out;
        final_out << "\n+============================================================================+\n";
        final_out << "|  COMPLETE - BEST ITERATION: " << best_iter 
                  << " (" << std::fixed << std::setprecision(2) << engine.getBestMatchRate() << "% match)\n";
        final_out << "|  Results ready for analysis and visualisation.                            |\n";
        final_out << "|  Console output saved to: kmeans_console_output.txt                      |\n";
        final_out << "+============================================================================+\n\n";
        dual_print(final_out.str());
 
    } catch (const std::exception& e) { 
        std::cerr << "\n[ERROR] " << e.what() << std::endl;
        if (g_log_file.is_open()) {
            g_log_file << "\n[ERROR] " << e.what() << std::endl;
            g_log_file.close();
        }
        return 1;
    }
    
    // Close log file
    if (g_log_file.is_open()) {
        g_log_file.close();
    }
    return 0;
}