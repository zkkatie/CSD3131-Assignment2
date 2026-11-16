// Registry of seam finders.
//
// THE SHARED INTERFACE (everyone must follow this):
//
//     std::vector<int> find_seam(const cv::Mat& energy);
//
//   - energy: H x W, CV_64F (doubles), from compute_energy
//   - returns: length-H vector, seam[i] = COLUMN of the pixel removed in row i
//   - the seam must be connected: |seam[i] - seam[i-1]| <= 1
//   - only VERTICAL seams; carve() handles horizontal seams by transposing
//   - must work for any size, including 1x1, 1xW and Hx1
//
// To add a method, create src/methods/<name>.cpp, e.g. src/methods/dp.cpp:
//
//     #include "seamcarve/methods.hpp"
//
//     namespace {
//     std::vector<int> find_seam(const cv::Mat& energy) {
//         ...
//     }
//     }  // namespace
//
//     REGISTER_SEAM_FINDER("dp", find_seam);
//
// Then re-run cmake. The method appears in --method and its tests switch on.
// No other file needs to change.
#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace seamcarve {

using SeamFinder = std::function<std::vector<int>(const cv::Mat& energy)>;

// Every method the project will have (used by the tests to report skipped methods).
inline const std::vector<std::string> PLANNED_METHODS = {"baseline", "dp", "greedy", "graphcut"};

// All registered seam finders, by name.
const std::map<std::string, SeamFinder>& methods();

// Returns the finder called `name`. Throws std::invalid_argument if it is not registered.
SeamFinder get_method(const std::string& name);

bool has_method(const std::string& name);

// Used by REGISTER_SEAM_FINDER. Returns true so it can initialise a static variable.
bool register_seam_finder(const std::string& name, SeamFinder finder);

}  // namespace seamcarve

#define SEAMCARVE_CONCAT_INNER(a, b) a##b
#define SEAMCARVE_CONCAT(a, b) SEAMCARVE_CONCAT_INNER(a, b)
#define REGISTER_SEAM_FINDER(name, function)                                        \
    static const bool SEAMCARVE_CONCAT(seamcarve_registered_, __LINE__) =           \
        ::seamcarve::register_seam_finder(name, function)
