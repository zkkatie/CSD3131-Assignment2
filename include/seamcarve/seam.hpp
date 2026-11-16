// Seam helpers shared by every method.
//
// A vertical seam is a std::vector<int> of length H, where seam[i] is the COLUMN
// of the pixel removed in row i. It must be connected: |seam[i] - seam[i-1]| <= 1.
#pragma once

#include <vector>

#include <opencv2/core.hpp>

namespace seamcarve {

// Position of a pixel in the ORIGINAL image.
struct Pixel {
    int row;
    int col;
    bool operator==(const Pixel& o) const { return row == o.row && col == o.col; }
};

enum class Direction { Vertical, Horizontal };

// Throws std::invalid_argument if `seam` is not a valid vertical seam for an H x W image.
void validate_seam(const std::vector<int>& seam, int height, int width);

// Total energy of a seam. Use this to compare DP vs Greedy vs Graph Cut.
double seam_cost(const cv::Mat& energy, const std::vector<int>& seam);

// Remove one pixel per row. Works for any Mat type and number of channels.
// Returns an H x (W-1) Mat of the same type. O(H * W).
cv::Mat remove_vertical_seam(const cv::Mat& img, const std::vector<int>& seam);

// Copy of `img` (as 3-channel 8-bit) with the seam painted in `color` (BGR, default red).
cv::Mat draw_vertical_seam(const cv::Mat& img, const std::vector<int>& seam,
                           const cv::Vec3b& color = {0, 0, 255});

// Paint every removed seam onto the ORIGINAL image (bonus item 7a).
// Vertical seams red, horizontal seams blue.
cv::Mat draw_removed_seams(const cv::Mat& original,
                           const std::vector<std::vector<Pixel>>& seam_coords,
                           const std::vector<Direction>& seam_directions,
                           const cv::Vec3b& vertical_color = {0, 0, 255},
                           const cv::Vec3b& horizontal_color = {255, 0, 0});

}  // namespace seamcarve
