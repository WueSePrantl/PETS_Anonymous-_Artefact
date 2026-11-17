

#include "openfhe.h"
#include "binfhecontext.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include <limits>
#include <unistd.h>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <sys/resource.h>
#include <cstring>

#include "satellite_data.h"

using namespace lbcrypto;

// ============================================================================
// MEMORY TRACKING
// ============================================================================

class MemoryTracker {
private:
    long baseline_kb;
    
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
    MemoryTracker() { baseline_kb = readVmRSS(); }
    long getCurrentMemoryKB() { return readVmRSS(); }
    long getMemoryIncrease() { return readVmRSS() - baseline_kb; }
    void resetBaseline() { baseline_kb = readVmRSS(); }
    
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
// DATA STRUCTURES
// ============================================================================

struct Config {
    uint32_t batch_size = 128;
    uint32_t num_dims = 5;
    uint32_t num_clusters = 3;
    uint32_t num_iterations = 5;
    double convergence_threshold = 1e-4;
    
    uint32_t ring_dim = 131072;
    uint32_t mult_depth = 80;
    uint32_t scale_mod_size = 43;
};

struct StepTimings {
    double encryption_ms = 0;
    double distance_ms = 0;
    double comparison_ms = 0;
    double masking_ms = 0;
    double accumulation_ms = 0;
    double division_ms = 0;
    double decryption_ms = 0;
    double total_ms = 0;
};

struct MemoryMetrics {
    long memory_per_distance_kb = 0;
    long memory_per_comparison_kb = 0;
    long memory_iteration_kb = 0;
    long total_memory_kb = 0;
};

struct IterationMetrics {
    int iteration;
    double time_sec;
    double mae, rmse, max_error;
    double cluster_match_rate;
    double delta;
    std::vector<int> cluster_sizes;
    StepTimings timings;
    MemoryMetrics memory;
    bool is_best = false;
};

// ============================================================================
// LOGGING
// ============================================================================

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
    if (g_log_file.is_open()) g_log_file << line << std::endl;
}

void dual_print(const std::string& text) {
    std::cout << text;
    if (g_log_file.is_open()) g_log_file << text;
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
// MATH HELPERS
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
    for(size_t k = 0; k < a.size(); ++k)
        total += std::sqrt(dist_sq(a[k], b[k]));
    return total / a.size();
}

double compute_cluster_match_rate(const std::vector<Point>& data,
                                   const std::vector<Point>& gt_centers,
                                   const std::vector<Point>& fhe_centers) {
    int matches = 0;
    for (const auto& p : data)
        if (find_nearest(p, gt_centers) == find_nearest(p, fhe_centers)) matches++;
    return 100.0 * matches / data.size();
}

std::vector<double> createExistenceBitmap(size_t n_real, uint32_t batch_size) {
    std::vector<double> existence(batch_size, 0.0);
    for (size_t i = 0; i < n_real && i < batch_size; ++i)
        existence[i] = 1.0;
    return existence;
}

// ============================================================================
// PLAINTEXT K-MEANS (GROUND TRUTH)
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
            for(uint32_t d = 0; d < cfg.num_dims; ++d)
                setCoord(new_centers[best], d, getCoord(new_centers[best], d) + getCoord(p, d));
        }

        for(uint32_t k = 0; k < cfg.num_clusters; ++k) {
            if(counts[k] > 0) {
                for(uint32_t d = 0; d < cfg.num_dims; ++d)
                    setCoord(new_centers[k], d, getCoord(new_centers[k], d) / counts[k]);
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
// PERFORMANCE OUTPUT
// ============================================================================

void print_performance(const IterationMetrics& m, MemoryTracker& mem) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(2);
    o << std::string(100, '=') << "\n";
    o << "  Performance Analysis (Iteration " << m.iteration << ")";
    if (m.is_best) o << "  *** BEST ***";
    o << "\n" << std::string(100, '-') << "\n";
    o << "    - Encryption:           " << std::setw(10) << m.timings.encryption_ms << " ms\n";
    o << "    - Distance Calculation: " << std::setw(10) << m.timings.distance_ms << " ms\n";
    o << "    - Scheme Switching:     " << std::setw(10) << m.timings.comparison_ms << " ms  <-- 3 comparisons\n";
    o << "    - Masking (1-x logic):  " << std::setw(10) << m.timings.masking_ms << " ms\n";
    o << "    - Accumulation:         " << std::setw(10) << m.timings.accumulation_ms << " ms\n";
    o << "    - Division (Chebyshev): " << std::setw(10) << m.timings.division_ms << " ms\n";
    o << "    - Decryption:           " << std::setw(10) << m.timings.decryption_ms << " ms\n";
    o << "    " << std::string(50, '-') << "\n";
    o << "    - Total:                " << std::setw(10) << m.timings.total_ms << " ms\n";
    o << std::string(100, '-') << "\n";
    o << "  Memory:\n";
    o << "    - Per Distance:         " << std::setw(10) << m.memory.memory_per_distance_kb << " KB\n";
    o << "    - Per Comparison:       " << std::setw(10) << m.memory.memory_per_comparison_kb << " KB\n";
    o << "    - This Iteration:       " << std::setw(10) << m.memory.memory_iteration_kb << " KB\n";
    o << "    - Total (VmRSS):        " << std::setw(10) << m.memory.total_memory_kb << " KB\n";
    o << "    - Peak (VmPeak):        " << std::setw(10) << mem.getPeakMemoryKB() << " KB\n";
    o << std::string(100, '=') << "\n\n";
    dual_print(o.str());
}

// ============================================================================
// CSV EXPORT
// ============================================================================

void export_cluster_assignments(const std::vector<Point>& data,
                                 const std::vector<Point>& gt_centers,
                                 const std::vector<Point>& fhe_centers,
                                 int best_iter, const std::string& filename) {
    log_event("EXPORT", "Cluster assignments -> " + filename);
    std::ofstream f(filename);
    f << "ID;GT_Cluster;FHE_Cluster;Match\n";
    for (size_t i = 0; i < data.size(); ++i) {
        int gt = find_nearest(data[i], gt_centers);
        int fhe = find_nearest(data[i], fhe_centers);
        f << i << ";" << gt << ";" << fhe << ";" << (gt == fhe ? 1 : 0) << "\n";
    }
    f.close();
}

void export_centers(const std::vector<Point>& gt, const std::vector<Point>& fhe,
                    int best_iter, const std::string& filename) {
    log_event("EXPORT", "Centers -> " + filename);
    std::ofstream f(filename);
    f << std::fixed << std::setprecision(6);
    f << "Cluster;Type;Dim0;Dim1;Dim2;Dim3;Dim4\n";
    for (size_t k = 0; k < gt.size(); ++k) {
        f << k << ";GT;" << getCoord(gt[k],0) << ";" << getCoord(gt[k],1) << ";"
          << getCoord(gt[k],2) << ";" << getCoord(gt[k],3) << ";" << getCoord(gt[k],4) << "\n";
    }
    for (size_t k = 0; k < fhe.size(); ++k) {
        f << k << ";FHE;" << getCoord(fhe[k],0) << ";" << getCoord(fhe[k],1) << ";"
          << getCoord(fhe[k],2) << ";" << getCoord(fhe[k],3) << ";" << getCoord(fhe[k],4) << "\n";
    }
    f.close();
}

void export_all_iterations(const std::vector<IterationMetrics>& history, const std::string& filename) {
    log_event("EXPORT", "All iterations -> " + filename);
    std::ofstream f(filename);
    f << std::fixed << std::setprecision(4);
    f << "Iteration;Time_s;MAE;RMSE;MatchRate;Delta;C0;C1;C2;IsBest;";
    f << "Encryption_ms;Distance_ms;Comparison_ms;Masking_ms;Accumulation_ms;Division_ms;Decryption_ms;";
    f << "MemPerDist_KB;MemPerComp_KB;MemIter_KB;MemTotal_KB\n";
    for (const auto& m : history) {
        f << m.iteration << ";" << m.time_sec << ";" << m.mae << ";" << m.rmse << ";"
          << m.cluster_match_rate << ";" << m.delta << ";"
          << m.cluster_sizes[0] << ";" << m.cluster_sizes[1] << ";" << m.cluster_sizes[2] << ";"
          << (m.is_best ? "YES" : "NO") << ";"
          << m.timings.encryption_ms << ";" << m.timings.distance_ms << ";"
          << m.timings.comparison_ms << ";" << m.timings.masking_ms << ";"
          << m.timings.accumulation_ms << ";" << m.timings.division_ms << ";"
          << m.timings.decryption_ms << ";"
          << m.memory.memory_per_distance_kb << ";" << m.memory.memory_per_comparison_kb << ";"
          << m.memory.memory_iteration_kb << ";" << m.memory.total_memory_kb << "\n";
    }
    f.close();
}

// ============================================================================
// HYBRID CKKS-FHEW K-MEANS 
// ============================================================================

class HybridKMeans {
    CryptoContext<DCRTPoly> cc;
    KeyPair<DCRTPoly> keys;
    Config cfg;
    
    Ciphertext<DCRTPoly> ct_zero;
    Plaintext pt_eps, pt_one;
    
    std::vector<IterationMetrics> history;
    MemoryTracker mem;
    
    double setup_time = 0, total_time = 0;
    int best_iteration = 0;
    double best_match_rate = 0;
    std::vector<Point> best_centers;

public:
    HybridKMeans(const Config& c) : cfg(c) {}
    
    void Setup() {
        log_event("SETUP", "===================================================");
        log_event("SETUP", "HYBRID CKKS-FHEW K-MEANS  (Paper Version)");
        log_event("SETUP", "3 Comparisons + 1-x Masks (Binary Outputs)");
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CCParams<CryptoContextCKKSRNS> params;
        params.SetRingDim(cfg.ring_dim);
        params.SetMultiplicativeDepth(cfg.mult_depth);
        params.SetFirstModSize(60);
        params.SetScalingModSize(cfg.scale_mod_size);
        params.SetSecurityLevel(HEStd_NotSet);
        params.SetBatchSize(cfg.batch_size);
        params.SetScalingTechnique(FIXEDMANUAL);

        cc = GenCryptoContext(params);
        cc->Enable(PKE);
        cc->Enable(KEYSWITCH);
        cc->Enable(LEVELEDSHE);
        cc->Enable(ADVANCEDSHE);
        cc->Enable(SCHEMESWITCH);
        cc->Enable(FHE);

        log_event("SETUP", "Generating keys...");
        keys = cc->KeyGen();
        cc->EvalMultKeyGen(keys.secretKey);
        
        std::vector<int32_t> rots;
        for(uint32_t i = 1; i <= cfg.batch_size; i *= 2) rots.push_back(i);
        cc->EvalRotateKeyGen(keys.secretKey, rots);
        
        log_event("SETUP", "Setting up Scheme Switching...");
        std::vector<uint32_t> levelBudget = {4, 4};
        std::vector<uint32_t> bsgsDim = {0, 0};
        cc->EvalBootstrapSetup(levelBudget, bsgsDim, cfg.batch_size);
        
        SchSwchParams swParams;
        swParams.SetSecurityLevelCKKS(HEStd_NotSet);
        swParams.SetSecurityLevelFHEW(STD128);
        swParams.SetCtxtModSizeFHEWLargePrec(20);
        swParams.SetNumSlotsCKKS(cfg.batch_size);
        auto lwe_sk = cc->EvalSchemeSwitchingSetup(swParams);
        
        cc->EvalBootstrapKeyGen(keys.secretKey, cfg.batch_size);
        cc->EvalSchemeSwitchingKeyGen(keys, lwe_sk);
        cc->EvalCompareSwitchPrecompute(1, 1.0);
        
        ct_zero = cc->Encrypt(keys.publicKey, 
            cc->MakeCKKSPackedPlaintext(std::vector<double>(cfg.batch_size, 0.0)));
        pt_eps = cc->MakeCKKSPackedPlaintext(std::vector<double>(cfg.batch_size, 0.001));
        pt_one = cc->MakeCKKSPackedPlaintext(std::vector<double>(cfg.batch_size, 1.0));
        
        setup_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();
        
        log_event("SETUP", "Complete in " + std::to_string(setup_time) + " s");
        log_event("SETUP", "===================================================");
    }

    Ciphertext<DCRTPoly> InverseChebyshev(Ciphertext<DCRTPoly> x, size_t n) {
        return cc->EvalChebyshevFunction([](double v) { return 1.0/v; }, x, 0.5, (double)n, 23);
    }

    void Run(const std::vector<Point>& data,
             const std::vector<Point>& init,
             const std::vector<std::vector<Point>>& gt_history) {
        
        auto total_start = std::chrono::high_resolution_clock::now();
        size_t N = data.size();
        size_t num_wagons = (N + cfg.batch_size - 1) / cfg.batch_size;
        
        log_event("EXEC", "===================================================");
        log_event("EXEC", "Dataset: " + std::to_string(N) + " points, " + 
                 std::to_string(num_wagons) + " wagons");
        log_event("EXEC", "===================================================");

        // Encrypt initial centers
        std::vector<std::vector<Ciphertext<DCRTPoly>>> enc_centers(cfg.num_clusters);
        for(size_t k = 0; k < cfg.num_clusters; ++k) {
            enc_centers[k].resize(cfg.num_dims);
            for(size_t d = 0; d < cfg.num_dims; ++d) {
                std::vector<double> v(cfg.batch_size, getCoord(init[k], d));
                enc_centers[k][d] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(v));
            }
        }

        std::vector<Point> prev_centers = init, cur_centers = init;
        best_centers = init;

        for(uint32_t it = 0; it < cfg.num_iterations; ++it) {
            auto iter_start = std::chrono::high_resolution_clock::now();
            long iter_mem_start = mem.getCurrentMemoryKB();
            
            log_event("ITER", "=== ITERATION " + std::to_string(it+1) + "/" + 
                     std::to_string(cfg.num_iterations) + " ===");

            IterationMetrics m;
            m.iteration = it + 1;
            PreciseTimer timer;

            // Init accumulators
            std::vector<Ciphertext<DCRTPoly>> g_counts(cfg.num_clusters);
            std::vector<std::vector<Ciphertext<DCRTPoly>>> g_sums(cfg.num_clusters);
            for(size_t k = 0; k < cfg.num_clusters; ++k) {
                g_counts[k] = ct_zero;
                g_sums[k].resize(cfg.num_dims);
                for(size_t d = 0; d < cfg.num_dims; ++d) g_sums[k][d] = ct_zero;
            }

            long total_dist_mem = 0, total_comp_mem = 0;
            int dist_count = 0, comp_count = 0;

            // Process wagons
            for(size_t w = 0; w < num_wagons; ++w) {
                size_t offset = w * cfg.batch_size;
                size_t wsize = std::min((size_t)cfg.batch_size, N - offset);
                
                log_event("WAGON", "Wagon " + std::to_string(w+1) + "/" + 
                         std::to_string(num_wagons) + " (" + std::to_string(wsize) + " pts)");

                // Encryption
                timer.reset();
                auto existence = createExistenceBitmap(wsize, cfg.batch_size);
                auto exist_ct = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(existence));
                
                std::vector<Ciphertext<DCRTPoly>> chunk(cfg.num_dims);
                for(size_t d = 0; d < cfg.num_dims; ++d) {
                    std::vector<double> col(cfg.batch_size, 0.0);
                    for(size_t i = 0; i < wsize; ++i) col[i] = getCoord(data[offset+i], d);
                    chunk[d] = cc->Encrypt(keys.publicKey, cc->MakeCKKSPackedPlaintext(col));
                }
                m.timings.encryption_ms += timer.elapsed_ms();

                // Distance calculation
                timer.reset();
                long before_dist = mem.getCurrentMemoryKB();
                
                std::vector<Ciphertext<DCRTPoly>> dists(cfg.num_clusters);
                for(size_t k = 0; k < cfg.num_clusters; ++k) {
                    Ciphertext<DCRTPoly> sum_sq;
                    bool first = true;
                    for(size_t d = 0; d < cfg.num_dims; ++d) {
                        auto diff = cc->EvalSub(chunk[d], enc_centers[k][d]);
                        auto sq = cc->EvalMult(diff, diff);
                        cc->ModReduceInPlace(sq);
                        if(first) { sum_sq = sq; first = false; }
                        else sum_sq = cc->EvalAdd(sum_sq, sq);
                    }
                    dists[k] = cc->EvalMult(sum_sq, exist_ct);
                    cc->ModReduceInPlace(dists[k]);
                    dist_count++;
                }
                m.timings.distance_ms += timer.elapsed_ms();
                total_dist_mem += (mem.getCurrentMemoryKB() - before_dist);

                // 3 Scheme Switching comparisons (v2.0 Logic!)
                timer.reset();
                long before_comp = mem.getCurrentMemoryKB();
                
                auto d0_lt_d1 = cc->EvalCompareSchemeSwitching(dists[0], dists[1], 1, cfg.batch_size);
                auto d0_lt_d2 = cc->EvalCompareSchemeSwitching(dists[0], dists[2], 1, cfg.batch_size);
                auto d1_lt_d2 = cc->EvalCompareSchemeSwitching(dists[1], dists[2], 1, cfg.batch_size);
                comp_count += 3;
                
                m.timings.comparison_ms += timer.elapsed_ms();
                total_comp_mem += (mem.getCurrentMemoryKB() - before_comp);

                // Masks with 1-x logic (correct for binary outputs!)
                timer.reset();
                std::vector<Ciphertext<DCRTPoly>> masks(3);
                
                masks[0] = cc->EvalMult(d0_lt_d1, d0_lt_d2);
                cc->ModReduceInPlace(masks[0]);
                
                auto not_d0_lt_d1 = cc->EvalSub(pt_one, d0_lt_d1);
                masks[1] = cc->EvalMult(not_d0_lt_d1, d1_lt_d2);
                cc->ModReduceInPlace(masks[1]);
                
                auto not_d0_lt_d2 = cc->EvalSub(pt_one, d0_lt_d2);
                auto not_d1_lt_d2 = cc->EvalSub(pt_one, d1_lt_d2);
                masks[2] = cc->EvalMult(not_d0_lt_d2, not_d1_lt_d2);
                cc->ModReduceInPlace(masks[2]);
                
                m.timings.masking_ms += timer.elapsed_ms();

                // Accumulation
                timer.reset();
                for(size_t k = 0; k < cfg.num_clusters; ++k) {
                    auto gated = cc->EvalMult(masks[k], exist_ct);
                    cc->ModReduceInPlace(gated);
                    g_counts[k] = cc->EvalAdd(g_counts[k], gated);
                    for(size_t d = 0; d < cfg.num_dims; ++d) {
                        auto weighted = cc->EvalMult(chunk[d], gated);
                        cc->ModReduceInPlace(weighted);
                        g_sums[k][d] = cc->EvalAdd(g_sums[k][d], weighted);
                    }
                }
                m.timings.accumulation_ms += timer.elapsed_ms();
            }

            // Division & Update
            log_event("UPDATE", "Computing new centers...");
            double mae_sum = 0, mse_sum = 0, max_err = 0;
            
            for(size_t k = 0; k < cfg.num_clusters; ++k) {
                timer.reset();
                auto total_count = cc->EvalSum(g_counts[k], cfg.batch_size);
                auto safe_count = cc->EvalAdd(total_count, pt_eps);
                auto inv = InverseChebyshev(safe_count, N);
                m.timings.division_ms += timer.elapsed_ms();

                for(size_t d = 0; d < cfg.num_dims; ++d) {
                    auto total_sum = cc->EvalSum(g_sums[k][d], cfg.batch_size);
                    enc_centers[k][d] = cc->EvalMult(total_sum, inv);
                    
                    timer.reset();
                    Plaintext pt;
                    cc->Decrypt(keys.secretKey, enc_centers[k][d], &pt);
                    pt->SetLength(1);
                    double val = std::real(pt->GetCKKSPackedValue()[0]);
                    m.timings.decryption_ms += timer.elapsed_ms();
                    
                    setCoord(cur_centers[k], d, val);
                    double err = std::abs(val - getCoord(gt_history[it+1][k], d));
                    mae_sum += err;
                    mse_sum += err * err;
                    max_err = std::max(max_err, err);
                }
            }

            // Metrics
            m.time_sec = std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - iter_start).count();
            m.timings.total_ms = m.time_sec * 1000;
            m.mae = mae_sum / (cfg.num_clusters * cfg.num_dims);
            m.rmse = std::sqrt(mse_sum / (cfg.num_clusters * cfg.num_dims));
            m.max_error = max_err;
            m.delta = compute_movement(cur_centers, prev_centers);
            
            m.cluster_sizes.assign(cfg.num_clusters, 0);
            for(const auto& p : data) {
                int idx = find_nearest(p, cur_centers);
                if(idx >= 0 && idx < (int)cfg.num_clusters) m.cluster_sizes[idx]++;
            }
            
            m.cluster_match_rate = compute_cluster_match_rate(data, gt_history[it+1], cur_centers);
            
            long iter_mem_end = mem.getCurrentMemoryKB();
            m.memory.memory_iteration_kb = iter_mem_end - iter_mem_start;
            m.memory.memory_per_distance_kb = dist_count > 0 ? total_dist_mem / dist_count : 0;
            m.memory.memory_per_comparison_kb = comp_count > 0 ? total_comp_mem / comp_count : 0;
            m.memory.total_memory_kb = iter_mem_end;

            // Check best
            if(m.cluster_match_rate > best_match_rate) {
                best_match_rate = m.cluster_match_rate;
                best_iteration = m.iteration;
                best_centers = cur_centers;
                m.is_best = true;
                log_event("BEST", ">>> NEW BEST: " + std::to_string(m.cluster_match_rate) + "% <<<");
            }

            // Output
            std::ostringstream o;
            o << "\n" << std::string(70, '=') << "\n";
            o << " ITERATION " << m.iteration;
            if(m.is_best) o << "  *** BEST ***";
            o << "\n" << std::string(70, '-') << "\n";
            o << "  MAE:  " << std::fixed << std::setprecision(5) << m.mae << "\n";
            o << "  RMSE: " << m.rmse << "\n";
            o << "  >>> MATCH: " << std::setprecision(2) << m.cluster_match_rate << "% <<<\n";
            o << "  Clusters: [" << m.cluster_sizes[0] << ", " << m.cluster_sizes[1] 
              << ", " << m.cluster_sizes[2] << "]\n";
            o << "  Delta: " << std::setprecision(4) << m.delta << "\n";
            o << "  Time: " << std::setprecision(1) << m.time_sec << "s\n";
            o << std::string(70, '=') << "\n";
            dual_print(o.str());
            
            print_performance(m, mem);
            history.push_back(m);
            prev_centers = cur_centers;
        }

        total_time = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - total_start).count();
        
        std::ostringstream summary;
        summary << "\n" << std::string(80, '#') << "\n";
        summary << "#  FINAL: Best Iteration " << best_iteration << " = " 
                << std::fixed << std::setprecision(2) << best_match_rate << "% match\n";
        summary << std::string(80, '#') << "\n";
        dual_print(summary.str());
        
        log_event("DONE", "Total: " + std::to_string(total_time) + " s");
    }

    int getBestIteration() const { return best_iteration; }
    double getBestMatchRate() const { return best_match_rate; }
    const std::vector<Point>& getBestCenters() const { return best_centers; }
    const std::vector<IterationMetrics>& getHistory() const { return history; }
};

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    g_log_file.open("kmeans_hybrid__console.txt");
    
    dual_print(R"(
╔══════════════════════════════════════════════════════════════════════════╗
║  HYBRID CKKS-FHEW K-MEANS  - FINAL PAPER VERSION     - by Dominik Prantl ║
╠══════════════════════════════════════════════════════════════════════════╣
║  • 3 Scheme Switching Comparisons (NOT 6!)                               ║
║  • 1-x Mask Logic (correct for binary outputs)                           ║
║  • Encrypted Centers throughout                                          ║
║  • Any dataset size (Existence Gating)                                   ║
║  • Best Iteration Export + CSV outputs                                   ║
╚══════════════════════════════════════════════════════════════════════════╝

)");

    try {
        Config cfg;
        
        log_event("DATA", "Loading satellite_data.h...");
        std::vector<Point> data = getSatelliteData();
        size_t N = data.size();
        
        if(argc > 1) {
            int req = std::atoi(argv[1]);
            if(req > 0 && req <= (int)N) {
                N = req;
                data.resize(N);
            }
        }
        log_event("DATA", std::to_string(N) + " points loaded");

        // Init centers
        size_t step = data.size() / cfg.num_clusters;
        std::vector<Point> init;
        for(size_t k = 0; k < cfg.num_clusters; ++k)
            init.push_back(data[k * step]);

        // Ground truth
        log_event("GT", "Computing plaintext K-Means...");
        auto gt_history = plaintext_kmeans_history(data, init, cfg);

        // Run Hybrid
        HybridKMeans engine(cfg);
        engine.Setup();
        engine.Run(data, init, gt_history);

        // Export
        log_event("EXPORT", "===================================================");
        int best = engine.getBestIteration();
        const auto& fhe_centers = engine.getBestCenters();
        const auto& gt_centers = gt_history[best];
        
        export_cluster_assignments(data, gt_centers, fhe_centers, best,
                                   "kmeans_hybrid_assignments_BEST.csv");
        export_centers(gt_centers, fhe_centers, best, "kmeans_hybrid_centers_BEST.csv");
        export_all_iterations(engine.getHistory(), "kmeans_hybrid__iterations.csv");
        
        log_event("EXPORT", "Files created:");
        log_event("EXPORT", "  - kmeans_hybrid_assignments_BEST.csv");
        log_event("EXPORT", "  - kmeans_hybrid_centers_BEST.csv");
        log_event("EXPORT", "  - kmeans_hybrid__iterations.csv");
        log_event("EXPORT", "  - kmeans_hybrid__console.txt");
        log_event("EXPORT", "===================================================");

        dual_print("\n*** COMPLETE: " + std::to_string(engine.getBestMatchRate()) + "% match ***\n\n");

    } catch(const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        if(g_log_file.is_open()) g_log_file << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    
    if(g_log_file.is_open()) g_log_file.close();
    return 0;
}

