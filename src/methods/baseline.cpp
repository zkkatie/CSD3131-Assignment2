// Baseline seam finder: the straight column with the lowest total energy.
//
// NOT one of the assignment's methods. It lets the whole pipeline be run and
// tested before DP / Greedy / Graph Cut exist, and gives a simple reference
// point for the comparison in the report.
#include "seamcarve/methods.hpp"

namespace {

std::vector<int> find_seam(const cv::Mat& energy) {
    cv::Mat column_sums;
    cv::reduce(energy, column_sums, 0, cv::REDUCE_SUM, CV_64F);  // 1 x W
    cv::Point min_loc;
    cv::minMaxLoc(column_sums, nullptr, nullptr, &min_loc);
    return std::vector<int>(energy.rows, min_loc.x);
}

}  // namespace

REGISTER_SEAM_FINDER("baseline", find_seam);
