// Benchmark: full energy recompute vs incremental update (justifies the framework design).
//
// Uses the baseline seam finder, so the timing measures the framework itself,
// not a particular seam finding method.
//
//   ./build/benchmark_energy                         synthetic images of several sizes
//   ./build/benchmark_energy images/broadway.jpg     your own image
//   ./build/benchmark_energy images/broadway.jpg 200 remove 200 seams
#include <algorithm>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "seamcarve/seamcarve.hpp"

using namespace seamcarve;

// Smooth background with random shapes, so it has both flat and busy areas.
static cv::Mat synthetic_image(int h, int w, unsigned seed = 0) {
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

int main(int argc, char** argv) {
    int seams = 100;
    const int repeats = 3;
    std::vector<std::pair<std::string, cv::Mat>> images;
    if (argc > 1) {
        cv::Mat img = cv::imread(argv[1]);
        if (img.empty()) {
            std::cerr << "could not read " << argv[1] << "\n";
            return 1;
        }
        images.push_back({argv[1], img});
        if (argc > 2) seams = std::stoi(argv[2]);
    } else {
        for (auto [w, h] : {std::pair{200, 150}, {400, 300}, {800, 600}, {1600, 1200}})
            images.push_back({std::to_string(w) + "x" + std::to_string(h), synthetic_image(h, w)});
    }

    const SeamFinder finder = get_method("baseline");
    std::printf("%12s %12s %10s %10s %16s\n", "image", "mode", "energy s", "total s", "energy speed-up");
    for (const auto& [name, img] : images) {
        const int n = std::min(seams, img.cols - 1);
        double best_energy[2] = {1e30, 1e30}, best_total[2] = {1e30, 1e30};
        cv::Mat outputs[2];
        const EnergyMode modes[2] = {EnergyMode::Full, EnergyMode::Incremental};
        for (int m = 0; m < 2; ++m)
            for (int rep = 0; rep < repeats; ++rep) {
                CarveOptions opt;
                opt.energy_mode = modes[m];
                CarveResult res = carve(img, img.cols - n, img.rows, finder, opt);
                best_energy[m] = std::min(best_energy[m], res.stats.energy_time);
                best_total[m] = std::min(best_total[m], res.stats.total_time());
                outputs[m] = res.image;
            }
        if (cv::norm(outputs[0], outputs[1], cv::NORM_INF) != 0) {
            std::cerr << "full and incremental modes disagree!\n";
            return 1;
        }
        std::printf("%12s %12s %10.4f %10.4f %16s\n", name.c_str(), "full", best_energy[0], best_total[0], "");
        std::printf("%12s %12s %10.4f %10.4f %15.1fx\n", name.c_str(), "incremental", best_energy[1], best_total[1],
                    best_energy[0] / best_energy[1]);
    }
    return 0;
}
