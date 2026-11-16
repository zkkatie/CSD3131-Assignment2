// Energy map (Assignment item 1a).
//
// The energy of a pixel measures how "important" it is. Pixels on edges (a big
// change in intensity) get high energy; flat areas get low energy. Seams follow
// low-energy pixels, so flat areas are removed first.
//
// Two energy functions, so they can be compared in the report:
//
//   Sobel     3x3 Sobel filter (default)
//             gx = [-1 0 1; -2 0 2; -1 0 1] * I        gy = its transpose
//   Gradient  simple central difference
//             gx = I(r, c+1) - I(r, c-1)              gy = I(r+1, c) - I(r-1, c)
//
//   energy = |gx| + |gy|        (L1 gradient magnitude, on the greyscale image)
//
// Borders: a neighbour outside the image is replaced by the nearest pixel inside
// it ("replicate" / clamp). For a 3x3 kernel this equals OpenCV's BORDER_REFLECT.
//
// Two ways to get the energy after a seam is removed:
//   full         recompute every pixel                          O(H * W) per seam
//   incremental  recompute only pixels next to the removed seam O(H) per seam
// Both give exactly the same map (checked in the tests).
#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace seamcarve {

enum class EnergyFunction { Sobel, Gradient };

std::string to_string(EnergyFunction fn);
// Parses "sobel" or "gradient". Throws std::invalid_argument otherwise.
EnergyFunction parse_energy_function(const std::string& name);

// BGR (CV_8UC3), BGRA (CV_8UC4) or greyscale (CV_8UC1) image -> H x W CV_64F greyscale.
cv::Mat to_gray(const cv::Mat& img);

// Full energy map of a CV_64F greyscale image. O(H * W). Returns H x W CV_64F.
cv::Mat energy_from_gray(const cv::Mat& gray, EnergyFunction fn = EnergyFunction::Sobel);

// Energy map of an image (to_gray + energy_from_gray).
cv::Mat compute_energy(const cv::Mat& img, EnergyFunction fn = EnergyFunction::Sobel);

// Energy of one pixel, same formula and borders as energy_from_gray. O(1).
double energy_at(const cv::Mat& gray, int row, int col, EnergyFunction fn = EnergyFunction::Sobel);

// Incremental energy update after removing a vertical seam. O(H).
//
//   energy_after_removal: the OLD energy map with the seam already removed (H x W-1).
//                         Updated in place.
//   gray_after_removal:   the greyscale image with the seam removed (H x W-1).
//   seam:                 the removed seam (columns in the OLD image).
//
// Why only a few pixels change: pixel (r, c) depends on rows r-1..r+1 and columns
// c-1..c+1. Let m and M be the smallest and largest seam column in rows r-1..r+1.
//   - c <= m-2: none of its neighbours moved             -> energy unchanged
//   - c >= M+1: all of its neighbours moved left together -> energy unchanged (shifted)
//   - otherwise its neighbourhood changed                 -> recompute
// Adjacent seam columns differ by at most 1, so M - m <= 2 and at most 4 pixels
// per row (columns m-1 .. m+2) are recomputed.
void update_energy(cv::Mat& energy_after_removal, const cv::Mat& gray_after_removal,
                   const std::vector<int>& seam, EnergyFunction fn = EnergyFunction::Sobel);

}  // namespace seamcarve
