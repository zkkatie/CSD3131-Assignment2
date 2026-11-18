// Evaluation and comparison of the seam finders (Asri).
//
// GOAL: answer "which method is better, by how much, and does the measured
// behaviour match the Big-O theory?" with numbers, for every registered method.
// Methods are taken from the registry, so DP (or any new method) is included
// automatically as soon as its file exists in src/methods/.
//
// THE FOUR EXPERIMENTS (each writes a CSV to results/ and prints a table):
//
//   E1 quality     Real photos in images/. Remove 25% of the width with each
//                  method and each energy function. At EVERY step we also
//                  compute the cheapest seam that existed at that moment, so
//                  we know how far the chosen seam was from the best one:
//                    gap %          = (sum of chosen seam costs / sum of
//                                      cheapest possible costs - 1) * 100
//                    seams optimal  = % of steps where the chosen seam was a
//                                     cheapest one
//                  Output images (and all seams drawn) go to results/eval_images/.
//
//   E2 optimality  Thousands of random energy maps, ONE seam each. Every method
//                  sees exactly the same maps. Two kinds of map:
//                    uniform  random values (worst case for greedy: no structure)
//                    photo    random crops of real energy maps (realistic)
//                  Reports % of maps where the method found a cheapest seam, the
//                  extra cost over all maps, and the worst single case.
//                  DP and Graph Cut must be optimal 100% of the time (checked).
//
//   E3 scaling     Synthetic images of growing size. Seam finding time per seam,
//                  best of 3. Three series: square (n x n), width only (H fixed),
//                  height only (W fixed). A least-squares fit of
//                  log(time) = slope * log(pixels) + c gives the measured growth:
//                    slope ~ 1    time grows like H*W   (expected for DP)
//                    slope ~ 0.5  time grows like H+W   (expected for Greedy, square)
//                    slope > 1    worse than linear     (expected for Graph Cut)
//
//   E4 breakdown   Real photos, best of 3: how the total time splits between
//                  energy update, seam finding and seam removal. Shows where the
//                  time actually goes for each method.
//
// The "cheapest possible seam" (ground truth) is computed by optimal_seam_cost
// below: a cost-only dynamic programme over the energy map, O(H * W). It is NOT
// registered as a seam finder (it returns a number, not a seam); it only exists
// so every method, including DP itself, is checked against an independent answer.
//
// Graph Cut (Edmonds-Karp) is far slower than the others, so it only runs on
// images up to SLOW_MAX_PIXELS pixels (the _small photos) and with fewer trials,
// and its scaling series stops once one seam takes longer than --budget seconds.
//
// Usage:
//   ./build/evaluate                       all four experiments (takes a few minutes)
//   ./build/evaluate quality scaling       only some of them
//   ./build/evaluate --quick               smaller run, to check everything works
//   options: --images DIR (default images)   --out DIR (default results)
//            --fraction F (default 0.25)      --budget S (default 2 seconds)
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "seamcarve/seamcarve.hpp"

using namespace seamcarve;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

struct Config {
    bool quick = false;
    std::string images_dir = "images";
    std::string out_dir = "results";
    double fraction = 0.25;  // share of the width removed in E1 and E4
    double budget = 2.0;     // E3: a slow method stops growing once one seam takes longer
    int repeats = 3;         // best-of-N timing
};

// Methods that are too slow for full-size images.
const std::set<std::string> SLOW_METHODS = {"graphcut"};
const int SLOW_MAX_PIXELS = 20000;  // the _small photos are 120 px wide, ~10k pixels
const int SLOW_MAX_TRIALS = 100;    // E2 trials per size for a slow method

bool is_slow(const std::string& method) { return SLOW_METHODS.count(method) > 0; }

bool too_big_for(const std::string& method, int rows, int cols) {
    return is_slow(method) && static_cast<long long>(rows) * cols > SLOW_MAX_PIXELS;
}

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// Cost of the cheapest connected vertical seam (ground truth). O(H * W) time, O(W) space.
//   M[0][c] = e[0][c]
//   M[r][c] = e[r][c] + min(M[r-1][c-1], M[r-1][c], M[r-1][c+1])
//   answer  = min over c of M[H-1][c]
double optimal_seam_cost(const cv::Mat& energy) {
    const int h = energy.rows, w = energy.cols;
    if (h == 0 || w == 0) return 0;
    std::vector<double> prev(w), cur(w);
    for (int c = 0; c < w; ++c) prev[c] = energy.at<double>(0, c);
    for (int r = 1; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            double best = prev[c];
            if (c > 0) best = std::min(best, prev[c - 1]);
            if (c + 1 < w) best = std::min(best, prev[c + 1]);
            cur[c] = energy.at<double>(r, c) + best;
        }
        std::swap(prev, cur);
    }
    return *std::min_element(prev.begin(), prev.end());
}

// A seam counts as optimal if its cost equals the minimum (up to rounding).
bool is_optimal(double cost, double optimum) {
    return cost <= optimum + 1e-9 * std::max(1.0, std::abs(optimum));
}

// Smooth background with random shapes (same idea as bench/benchmark_energy.cpp).
cv::Mat synthetic_image(int h, int w, unsigned seed = 0) {
    cv::Mat img(h, w, CV_8UC3);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c)
            img.at<cv::Vec3b>(r, c) = {static_cast<uchar>(c * 255 / std::max(w - 1, 1)),
                                       static_cast<uchar>(r * 255 / std::max(h - 1, 1)), 120};
    cv::RNG rng(seed);
    for (int i = 0; i < 12; ++i)
        cv::circle(img, {rng.uniform(0, w), rng.uniform(0, h)}, rng.uniform(5, std::max(6, h / 6)),
                   cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), cv::FILLED);
    return img;
}

struct NamedImage {
    std::string name;  // file name without extension, e.g. "broadway_small"
    cv::Mat img;
};

bool is_small_name(const std::string& name) {
    return name.size() >= 6 && name.compare(name.size() - 6, 6, "_small") == 0;
}

std::vector<NamedImage> load_images(const std::string& dir) {
    std::vector<NamedImage> out;
    if (!fs::is_directory(dir)) return out;
    for (const auto& entry : fs::directory_iterator(dir)) {
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) { return std::tolower(ch); });
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png") continue;
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "warning: could not read " << entry.path() << ", skipped\n";
            continue;
        }
        out.push_back({entry.path().stem().string(), img});
    }
    std::sort(out.begin(), out.end(), [](const NamedImage& a, const NamedImage& b) { return a.name < b.name; });
    return out;
}

std::ofstream open_csv(const Config& cfg, const std::string& file, const std::string& header) {
    fs::create_directories(cfg.out_dir);
    const fs::path path = fs::path(cfg.out_dir) / file;
    std::ofstream f(path);
    if (!f) throw std::runtime_error("could not write " + path.string());
    f << header << "\n";
    return f;
}

// Least-squares slope of log(y) against log(x). NaN if fewer than 2 usable points.
double loglog_slope(const std::vector<std::pair<double, double>>& points) {
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    int n = 0;
    for (const auto& [x, y] : points) {
        if (x <= 0 || y <= 0) continue;
        const double lx = std::log(x), ly = std::log(y);
        sx += lx;
        sy += ly;
        sxx += lx * lx;
        sxy += lx * ly;
        ++n;
    }
    if (n < 2) return std::numeric_limits<double>::quiet_NaN();
    const double denom = n * sxx - sx * sx;
    return denom == 0 ? std::numeric_limits<double>::quiet_NaN() : (n * sxy - sx * sy) / denom;
}

int seams_to_remove(const Config& cfg, int width) {
    if (width < 2) return 0;
    return std::clamp(static_cast<int>(std::lround(width * cfg.fraction)), 1, width - 1);
}

// ---------------------------------------------------------------- E1 quality
void run_quality(const Config& cfg, const std::vector<NamedImage>& images) {
    std::printf("\nE1 QUALITY: remove %.0f%% of the width; compare each seam with the cheapest one possible\n",
                cfg.fraction * 100);
    std::printf("%-16s %-9s %-9s %6s %14s %14s %8s %10s %10s\n", "image", "energy", "method", "seams",
                "total cost", "optimal cost", "gap %", "optimal %", "find s");

    auto csv = open_csv(cfg, "eval_quality.csv",
                        "method,image,width,height,energy,seams,total_cost,optimal_cost,gap_pct,"
                        "seams_optimal_pct,find_time");
    const fs::path img_dir = fs::path(cfg.out_dir) / "eval_images";
    fs::create_directories(img_dir);

    std::vector<EnergyFunction> fns = {EnergyFunction::Sobel, EnergyFunction::Gradient};
    if (cfg.quick) fns = {EnergyFunction::Sobel};

    for (const auto& [name, img] : images) {
        if (cfg.quick && !is_small_name(name)) continue;
        const int n = seams_to_remove(cfg, img.cols);
        if (n == 0) continue;
        for (EnergyFunction fn : fns) {
            for (const auto& [method, finder] : methods()) {
                if (too_big_for(method, img.rows, img.cols)) continue;

                // Ground truth at every step, computed in the callback (outside carve's timers).
                std::vector<double> step_optimum;
                CarveOptions opt;
                opt.energy_fn = fn;
                opt.on_seam = [&](const cv::Mat& current, const std::vector<int>&, Direction) {
                    step_optimum.push_back(optimal_seam_cost(compute_energy(current, fn)));
                };
                const CarveResult res = carve(img, img.cols - n, img.rows, finder, opt);
                const CarveStats& s = res.stats;

                double optimal_total = 0;
                int at_optimum = 0;
                for (size_t i = 0; i < step_optimum.size(); ++i) {
                    optimal_total += step_optimum[i];
                    if (is_optimal(s.seam_costs[i], step_optimum[i])) ++at_optimum;
                }
                const double gap = optimal_total > 0 ? (s.total_seam_cost / optimal_total - 1) * 100 : 0;
                const double pct_opt = 100.0 * at_optimum / s.seams_removed;

                std::printf("%-16s %-9s %-9s %6d %14.1f %14.1f %8.2f %10.1f %10.4f\n", name.c_str(),
                            to_string(fn).c_str(), method.c_str(), s.seams_removed, s.total_seam_cost,
                            optimal_total, gap, pct_opt, s.find_time);
                csv << method << "," << name << "," << img.cols << "," << img.rows << "," << to_string(fn) << ","
                    << s.seams_removed << "," << s.total_seam_cost << "," << optimal_total << "," << gap << ","
                    << pct_opt << "," << s.find_time << "\n";

                if (fn == EnergyFunction::Sobel) {
                    const std::string base = (img_dir / (name + "_" + method)).string();
                    cv::imwrite(base + ".png", res.image);
                    cv::imwrite(base + "_seams.png", draw_removed_seams(img, s.seam_coords, s.seam_directions));
                }
            }
        }
    }
    std::printf("output images: %s\n", img_dir.string().c_str());
}

// ------------------------------------------------------------- E2 optimality
void run_optimality(const Config& cfg, const std::vector<NamedImage>& images) {
    std::vector<std::pair<int, int>> sizes = {{5, 5}, {10, 10}, {25, 25}, {50, 50}, {100, 100}, {200, 200}};
    int trials = 500;
    if (cfg.quick) {
        sizes = {{10, 10}, {50, 50}};
        trials = 50;
    }
    std::printf("\nE2 OPTIMALITY: %d random energy maps per size, one seam each\n", trials);
    std::printf("%-8s %-9s %9s %7s %10s %12s %12s %12s\n", "maps", "method", "size", "trials", "optimal %",
                "extra cost %", "worst case %", "ms per seam");

    auto csv = open_csv(cfg, "eval_optimality.csv",
                        "kind,method,height,width,trials,pct_optimal,total_excess_pct,max_excess_pct,"
                        "mean_find_ms");

    // Energy maps of the full-size photos, to crop "photo" maps from.
    std::vector<cv::Mat> photo_energy;
    for (const auto& [name, img] : images) photo_energy.push_back(compute_energy(img));

    cv::RNG rng(42);
    for (const std::string kind : {"uniform", "photo"}) {
        for (const auto& [h, w] : sizes) {
            // Generate the maps once, so every method sees the same ones.
            std::vector<cv::Mat> maps;
            std::vector<cv::Mat> big_enough;
            for (const cv::Mat& e : photo_energy)
                if (e.rows >= h && e.cols >= w) big_enough.push_back(e);
            if (kind == std::string("photo") && big_enough.empty()) continue;
            for (int t = 0; t < trials; ++t) {
                cv::Mat m;
                if (kind == std::string("uniform")) {
                    m.create(h, w, CV_64F);
                    rng.fill(m, cv::RNG::UNIFORM, 0.0, 1000.0);
                } else {
                    const cv::Mat& src = big_enough[rng.uniform(0, static_cast<int>(big_enough.size()))];
                    const int r0 = rng.uniform(0, src.rows - h + 1), c0 = rng.uniform(0, src.cols - w + 1);
                    m = src(cv::Rect(c0, r0, w, h)).clone();
                }
                maps.push_back(m);
            }
            std::vector<double> optimum(maps.size());
            for (size_t i = 0; i < maps.size(); ++i) optimum[i] = optimal_seam_cost(maps[i]);

            for (const auto& [method, finder] : methods()) {
                if (too_big_for(method, h, w)) continue;
                const int count = is_slow(method) ? std::min(trials, SLOW_MAX_TRIALS) : trials;
                int optimal = 0;
                double sum_cost = 0, sum_opt = 0, worst = 0, time = 0;
                for (int i = 0; i < count; ++i) {
                    const auto t0 = Clock::now();
                    const std::vector<int> seam = finder(maps[i]);
                    time += seconds_since(t0);
                    validate_seam(seam, h, w);
                    const double cost = seam_cost(maps[i], seam);
                    sum_cost += cost;
                    sum_opt += optimum[i];
                    if (is_optimal(cost, optimum[i])) ++optimal;
                    if (optimum[i] > 0) worst = std::max(worst, (cost / optimum[i] - 1) * 100);
                }
                const double pct = 100.0 * optimal / count;
                const double extra = sum_opt > 0 ? (sum_cost / sum_opt - 1) * 100 : 0;
                const double ms = time / count * 1000;
                const std::string size = std::to_string(h) + "x" + std::to_string(w);
                std::printf("%-8s %-9s %9s %7d %10.1f %12.2f %12.2f %12.4f\n", kind.c_str(), method.c_str(),
                            size.c_str(), count, pct, extra, worst, ms);
                csv << kind << "," << method << "," << h << "," << w << "," << count << "," << pct << "," << extra
                    << "," << worst << "," << ms << "\n";
                if ((method == "dp" || method == "graphcut") && optimal != count)
                    std::printf("WARNING: %s should always be optimal but missed %d of %d maps\n", method.c_str(),
                                count - optimal, count);
            }
        }
    }
}

// ---------------------------------------------------------------- E3 scaling
void run_scaling(const Config& cfg) {
    struct Series {
        std::string name;
        std::vector<std::pair<int, int>> sizes;  // (h, w)
    };
    std::vector<int> ns = {25, 50, 100, 200, 400, 800, 1600};
    if (cfg.quick) ns = {25, 50, 100, 200};
    Series square{"square", {}}, width{"width", {}}, height{"height", {}};
    for (int n : ns) {
        square.sizes.push_back({n, n});
        width.sizes.push_back({200, n});   // H fixed at 200
        height.sizes.push_back({n, 200});  // W fixed at 200
    }
    std::vector<Series> all = {square, width, height};
    if (cfg.quick) all = {square};

    std::printf("\nE3 SCALING: seam finding time per seam (best of %d), synthetic images\n", cfg.repeats);
    std::printf("%-7s %-9s %11s %10s %14s\n", "series", "method", "size", "pixels", "ms per seam");

    auto csv = open_csv(cfg, "eval_scaling.csv", "series,method,height,width,pixels,seams,find_time_per_seam");
    auto slopes_csv = open_csv(cfg, "eval_scaling_slopes.csv", "series,method,points,slope");
    std::map<std::pair<std::string, std::string>, double> slopes;

    for (const Series& series : all) {
        for (const auto& [method, finder] : methods()) {
            std::vector<std::pair<double, double>> points;
            for (const auto& [h, w] : series.sizes) {
                const cv::Mat img = synthetic_image(h, w);
                const int k = is_slow(method) ? 2 : std::min(20, w - 1);
                const int reps = is_slow(method) ? 1 : cfg.repeats;
                double best = std::numeric_limits<double>::infinity();
                for (int rep = 0; rep < reps; ++rep) {
                    const CarveResult res = carve(img, w - k, h, finder);
                    best = std::min(best, res.stats.find_time / k);
                }
                const double pixels = static_cast<double>(h) * w;
                points.push_back({pixels, best});
                const std::string size = std::to_string(h) + "x" + std::to_string(w);
                std::printf("%-7s %-9s %11s %10.0f %14.5f\n", series.name.c_str(), method.c_str(), size.c_str(),
                            pixels, best * 1000);
                csv << series.name << "," << method << "," << h << "," << w << "," << pixels << "," << k << ","
                    << best << "\n";
                if (best > cfg.budget) {
                    std::printf("        %s takes over %.1f s per seam, larger sizes skipped\n", method.c_str(),
                                cfg.budget);
                    break;
                }
            }
            const double slope = loglog_slope(points);
            slopes[{series.name, method}] = slope;
            slopes_csv << series.name << "," << method << "," << points.size() << "," << slope << "\n";
        }
    }

    std::printf("\nmeasured growth: time ~ pixels^slope\n%-7s %-9s %8s\n", "series", "method", "slope");
    for (const auto& [key, slope] : slopes)
        std::printf("%-7s %-9s %8.2f\n", key.first.c_str(), key.second.c_str(), slope);
}

// -------------------------------------------------------------- E4 breakdown
void run_breakdown(const Config& cfg, const std::vector<NamedImage>& images) {
    std::printf("\nE4 TIME BREAKDOWN: remove %.0f%% of the width, best of %d runs\n", cfg.fraction * 100,
                cfg.repeats);
    std::printf("%-16s %-9s %10s %10s %10s %10s %8s\n", "image", "method", "energy s", "find s", "remove s",
                "total s", "find %");

    auto csv = open_csv(cfg, "eval_breakdown.csv",
                        "method,image,width,height,seams,energy_time,find_time,remove_time,total_time,find_pct");
    for (const auto& [name, img] : images) {
        if (cfg.quick && !is_small_name(name)) continue;
        const int n = seams_to_remove(cfg, img.cols);
        if (n == 0) continue;
        for (const auto& [method, finder] : methods()) {
            if (too_big_for(method, img.rows, img.cols)) continue;
            CarveStats best;
            double best_total = std::numeric_limits<double>::infinity();
            const int reps = is_slow(method) ? 1 : cfg.repeats;
            for (int rep = 0; rep < reps; ++rep) {
                CarveResult res = carve(img, img.cols - n, img.rows, finder);
                if (res.stats.total_time() < best_total) {
                    best_total = res.stats.total_time();
                    best = std::move(res.stats);
                }
            }
            const double find_pct = best_total > 0 ? best.find_time / best_total * 100 : 0;
            std::printf("%-16s %-9s %10.4f %10.4f %10.4f %10.4f %8.1f\n", name.c_str(), method.c_str(),
                        best.energy_time, best.find_time, best.remove_time, best_total, find_pct);
            csv << method << "," << name << "," << img.cols << "," << img.rows << "," << best.seams_removed << ","
                << best.energy_time << "," << best.find_time << "," << best.remove_time << "," << best_total << ","
                << find_pct << "\n";
        }
    }
}

void print_usage() {
    std::cout << "Usage: evaluate [quality] [optimality] [scaling] [breakdown] [options]\n"
                 "  (no experiment named = all four)\n"
                 "  --quick         smaller run, to check that everything works\n"
                 "  --images DIR    photos to use (default: images)\n"
                 "  --out DIR       where CSVs and images go (default: results)\n"
                 "  --fraction F    share of the width removed in quality/breakdown (default 0.25)\n"
                 "  --budget S      scaling: stop a method once one seam takes over S seconds (default 2)\n";
}

}  // namespace

int main(int argc, char** argv) {
    Config cfg;
    std::vector<std::string> wanted;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "-h" || a == "--help") {
                print_usage();
                return 0;
            } else if (a == "--quick") cfg.quick = true;
            else if (a == "--images") cfg.images_dir = value();
            else if (a == "--out") cfg.out_dir = value();
            else if (a == "--fraction") cfg.fraction = std::stod(value());
            else if (a == "--budget") cfg.budget = std::stod(value());
            else if (a == "quality" || a == "optimality" || a == "scaling" || a == "breakdown") wanted.push_back(a);
            else throw std::invalid_argument("unknown argument " + a);
        }
        if (!(cfg.fraction > 0 && cfg.fraction < 1)) throw std::invalid_argument("--fraction must be in (0, 1)");
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage();
        return 1;
    }
    if (wanted.empty()) wanted = {"quality", "optimality", "scaling", "breakdown"};
    auto want = [&](const std::string& e) { return std::find(wanted.begin(), wanted.end(), e) != wanted.end(); };

    // Setup, for the report.
    std::string names;
    for (const auto& [name, f] : methods()) names += (names.empty() ? "" : ", ") + name;
    std::printf("methods:  %s\n", names.c_str());
    for (const std::string& planned : PLANNED_METHODS)
        if (!has_method(planned)) std::printf("note:     '%s' is not written yet; it will be included automatically\n",
                                              planned.c_str());
#ifdef NDEBUG
    std::printf("build:    Release\n");
#else
    std::printf("build:    Debug (timings are NOT representative, use a Release build)\n");
#endif
    std::printf("threads:  %u hardware threads, OpenCV %s\n", std::thread::hardware_concurrency(), CV_VERSION);

    const std::vector<NamedImage> images = load_images(cfg.images_dir);
    std::printf("images:   %zu in %s\n", images.size(), cfg.images_dir.c_str());
    if (images.empty() && (want("quality") || want("breakdown") || want("optimality")))
        std::printf("warning:  no images found, quality/breakdown are skipped and optimality uses uniform maps only\n");

    try {
        const auto start = Clock::now();
        if (want("quality")) run_quality(cfg, images);
        if (want("optimality")) run_optimality(cfg, images);
        if (want("scaling")) run_scaling(cfg);
        if (want("breakdown")) run_breakdown(cfg, images);
        std::printf("\ndone in %.1f s, CSVs in %s/\n", seconds_since(start), cfg.out_dir.c_str());
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
