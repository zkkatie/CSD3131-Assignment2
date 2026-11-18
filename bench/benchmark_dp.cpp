// Benchmark for the Dynamic Programming seam finder.
//
// Measures:
//   1. time and working memory of each DP variant (full table, top-down,
//      two rows + parent table, linear space)
//   2. speed-up of the multithreaded variant over the single-threaded one
//   3. for each image given on the command line, how a real resize splits its time
//      between energy updates, seam finding and seam removal
//
//   ./build/benchmark_dp                         random energy maps, 5 sizes
//   ./build/benchmark_dp images/broadway.jpg     additionally carve this image
//
// Times are for finding one seam on a fixed energy map (median of REPEATS runs).
// Memory is the size of the arrays each variant allocates (see table_memory below).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "seamcarve/dp.hpp"
#include "seamcarve/seamcarve.hpp"

using namespace seamcarve;

namespace {

constexpr int REPEATS = 15;

template <typename F>
double median_ms(F&& f, int repeats = REPEATS) {
    std::vector<double> t;
    for (int r = 0; r < repeats; ++r) {
        const auto start = std::chrono::steady_clock::now();
        f();
        t.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

std::string mem(std::size_t bytes) {
    char buf[32];
    if (bytes < 1024) std::snprintf(buf, sizeof buf, "%zu B", bytes);
    else if (bytes < 1024 * 1024) std::snprintf(buf, sizeof buf, "%.1f KB", bytes / 1024.0);
    else std::snprintf(buf, sizeof buf, "%.1f MB", bytes / (1024.0 * 1024.0));
    return buf;
}

// Bytes of the arrays a variant allocates for an H x W map (rows are padded to W + 2).
std::size_t table_memory(const std::string& variant, int h, int w) {
    const std::size_t row = static_cast<std::size_t>(w + 2) * sizeof(double);
    if (variant == "two_rows") return 2 * row + static_cast<std::size_t>(h) * w * sizeof(signed char);
    if (variant == "linear_space") return 4 * row + 2 * row;  // F, B, two sweep rows, a 2-row base table
    return static_cast<std::size_t>(h) * row;                 // full, top_down, threaded: the whole table
}

cv::Mat random_energy(int w, int h) {
    cv::Mat e(h, w, CV_64F);
    cv::RNG(0).fill(e, cv::RNG::UNIFORM, 0.0, 255.0);
    return e;
}

}  // namespace

int main(int argc, char** argv) {
    const std::vector<std::pair<int, int>> sizes = {{200, 150}, {640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    std::printf("%u hardware threads, %d repeats (median)\n", std::thread::hardware_concurrency(), REPEATS);

    // 1. Variants: time and working memory
    std::printf("\n## Variants: time per seam and working memory\n\n");
    std::printf("| size | variant | time (ms) | ns / pixel | memory | seam cost |\n|---|---|---:|---:|---:|---:|\n");
    for (auto [w, h] : sizes) {
        const cv::Mat e = random_energy(w, h);
        double ref = -1;
        for (const auto& v : dp::variants()) {
            if (v.name == "threaded") continue;
            std::vector<int> seam;
            const double t = median_ms([&] { seam = v.fn(e); });
            const double cost = seam_cost(e, seam);
            if (ref < 0) ref = cost;
            if (std::abs(cost - ref) > 1e-6 * ref) {
                std::cerr << v.name << " returned a different seam cost\n";
                return 1;
            }
            std::printf("| %dx%d | %s | %.2f | %.2f | %s | %.1f |\n", w, h, v.name.c_str(), t, t * 1e6 / (double(w) * h),
                        mem(table_memory(v.name, h, w)).c_str(), cost);
        }
    }

    // 2. Threads
    std::printf("\n## Threads: time per seam (ms) and speed-up over single-threaded find_seam\n\n");
    const std::vector<int> workers = {1, 2, 4, 8, 16};
    std::printf("| size | find_seam |");
    for (int p : workers) std::printf(" %d thr |", p);
    std::printf("\n|---|---:|");
    for (std::size_t i = 0; i < workers.size(); ++i) std::printf("---:|");
    std::printf("\n");
    for (auto [w, h] : sizes) {
        const cv::Mat e = random_energy(w, h);
        const std::vector<int> expected = dp::find_seam(e);
        const double base = median_ms([&] { dp::find_seam(e); });
        std::printf("| %dx%d | %.2f |", w, h, base);
        for (int p : workers) {
            std::vector<int> seam;
            const double t = median_ms([&] { seam = dp::find_seam_threaded(e, p); });
            if (seam != expected) {
                std::cerr << "threaded variant returned a different seam\n";
                return 1;
            }
            std::printf(" %.2f (%.2fx) |", t, base / t);
        }
        std::printf("\n");
    }

    // 3. Time split of a real resize (input to Amdahl's law)
    for (int i = 1; i < argc; ++i) {
        const cv::Mat img = cv::imread(argv[i]);
        if (img.empty()) {
            std::cerr << "could not read " << argv[i] << "\n";
            return 1;
        }
        const int tw = img.cols * 4 / 5, th = img.rows * 4 / 5;
        std::printf("\n## %s: %dx%d -> %dx%d (%d seams)\n\n", argv[i], img.cols, img.rows, tw, th,
                    img.cols - tw + img.rows - th);
        std::printf("| method | total seam energy | find (s) | energy (s) | remove (s) | total (s) | find share |\n");
        std::printf("|---|---:|---:|---:|---:|---:|---:|\n");
        for (const std::string name : {"baseline", "greedy", "dp"}) {
            if (!has_method(name)) continue;
            const CarveStats s = carve(img, tw, th, get_method(name)).stats;
            std::printf("| %s | %.0f | %.3f | %.3f | %.3f | %.3f | %.0f%% |\n", name.c_str(), s.total_seam_cost,
                        s.find_time, s.energy_time, s.remove_time, s.total_time(),
                        100.0 * s.find_time / s.total_time());
        }
    }
    return 0;
}
