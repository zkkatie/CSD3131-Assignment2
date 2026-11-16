// Main seam carving loop (Assignment item 1c).
//
//   repeat until the image is the target size:
//     1. get the energy map    (full recompute, or incremental update - see energy.hpp)
//     2. find the minimum seam (DP / Greedy / Graph Cut, passed in as a SeamFinder)
//     3. remove the seam       (from the image, greyscale, energy and coordinate maps)
//
// Horizontal seams: transpose everything (swap rows and columns), remove vertical
// seams, then transpose back. A horizontal seam in the image is a vertical seam in
// the transposed image, so each seam finder only handles vertical seams. The energy
// map can be transposed too, because |gx| + |gy| is symmetric under transposing.
//
// Seam tracking: two extra int maps hold each remaining pixel's row and column in
// the ORIGINAL image. They are carved exactly like the image, so every removed seam
// can be reported in original coordinates (to draw them all, bonus item 7a).
//
// Cost per seam (H x W image):
//   find_seam                 depends on the method (DP is O(H * W))
//   energy, full mode         O(H * W)
//   energy, incremental mode  O(H)
//   removing the seam         O(H * W) memory copy per array
#pragma once

#include <functional>
#include <vector>

#include <opencv2/core.hpp>

#include "seamcarve/energy.hpp"
#include "seamcarve/methods.hpp"
#include "seamcarve/seam.hpp"

namespace seamcarve {

enum class EnergyMode { Incremental, Full };

std::string to_string(EnergyMode mode);
// Parses "incremental" or "full". Throws std::invalid_argument otherwise.
EnergyMode parse_energy_mode(const std::string& name);

// Called for every seam BEFORE it is removed: on_seam(current_image, seam, direction).
// For horizontal seams `current_image` is transposed (so the seam is vertical in it).
// Adi can use this for step-by-step visualisation.
using SeamCallback = std::function<void(const cv::Mat&, const std::vector<int>&, Direction)>;

struct CarveOptions {
    EnergyFunction energy_fn = EnergyFunction::Sobel;
    EnergyMode energy_mode = EnergyMode::Incremental;
    SeamCallback on_seam = nullptr;
};

// Everything recorded during carving (Asri uses this for the comparison tables).
struct CarveStats {
    int seams_removed = 0;
    double total_seam_cost = 0;
    double energy_time = 0;  // seconds computing / updating energy maps
    double find_time = 0;    // seconds finding seams (this is what differs per method)
    double remove_time = 0;  // seconds removing seams from the arrays
    std::vector<double> seam_costs;
    std::vector<std::vector<Pixel>> seam_coords;  // each removed seam, in ORIGINAL image coordinates
    std::vector<Direction> seam_directions;

    double total_time() const { return energy_time + find_time + remove_time; }
};

struct CarveResult {
    cv::Mat image;
    CarveStats stats;
};

// Shrink `img` to target_width x target_height. Removes all vertical seams first,
// then all horizontal seams. Throws std::invalid_argument for an invalid target
// (enlarging or below 1x1) or an invalid seam returned by `find_seam`.
CarveResult carve(const cv::Mat& img, int target_width, int target_height,
                  const SeamFinder& find_seam, const CarveOptions& options = {});

// Remove `n` vertical seams (image becomes n pixels narrower).
CarveResult remove_vertical_seams(const cv::Mat& img, int n, const SeamFinder& find_seam,
                                  const CarveOptions& options = {});

// Remove `n` horizontal seams (image becomes n pixels shorter).
CarveResult remove_horizontal_seams(const cv::Mat& img, int n, const SeamFinder& find_seam,
                                    const CarveOptions& options = {});

// Swap rows and columns (works for any number of channels).
cv::Mat transpose(const cv::Mat& img);

}  // namespace seamcarve
