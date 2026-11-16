// Greedy seam finder (Assignment item 3).                                    Joel
//
// Idea: build the seam one row at a time, always taking the cheapest pixel that
// is allowed next, and never look further ahead.
//
//   1. Row 0: start at the column with the lowest energy.
//   2. Row i (i >= 1): the seam is at column c in row i-1, so the only legal
//      choices are c-1, c, c+1 (clamped to the image). Take the one with the
//      lowest energy.
//
// Tie-breaking: if several candidates have the same energy, prefer going straight
// down (c), then left (c-1), then right (c+1). Going straight on ties stops the
// seam from drifting sideways across flat areas (e.g. a clear sky), where every
// pixel has the same energy.
//
// Why greedy is NOT optimal: a choice that looks cheap now can lead the seam into
// an expensive area later, and greedy can never undo a choice. Dynamic Programming
// avoids this by looking at the cumulative cost M[i][j] of the best seam ending at
// each pixel before deciding. Example (cheap start, then a wall of 100s):
//
//     0   9   9   1        greedy: 0 -> 9 -> 100  = 109
//     9   9   9   1        best:   1 ->  1 ->  1  =   3
//   100 100   9   1
//
// Cost for an H x W energy map:
//   time   O(W) for row 0 + O(1) per later row = O(H + W)
//          (DP is O(H * W): it must fill the whole table)
//   space  O(H) for the seam itself, no table
#include "seamcarve/methods.hpp"

namespace {

std::vector<int> find_seam(const cv::Mat& energy) {
    const int h = energy.rows, w = energy.cols;
    std::vector<int> seam(h);
    if (h == 0 || w == 0) return seam;

    // Step 1: cheapest pixel in the top row (leftmost one if there is a tie).
    int col = 0;
    for (int c = 1; c < w; ++c)
        if (energy.at<double>(0, c) < energy.at<double>(0, col)) col = c;
    seam[0] = col;

    // Step 2: walk down, each time picking the cheapest of the (up to) 3 pixels below.
    for (int r = 1; r < h; ++r) {
        const int prev = col;
        // Candidates in tie-break order: straight down, left, right.
        // A strict '<' keeps the earlier candidate when energies are equal.
        const int candidates[3] = {prev, prev - 1, prev + 1};
        for (int c : candidates) {
            if (c < 0 || c >= w) continue;  // off the edge of the image
            if (energy.at<double>(r, c) < energy.at<double>(r, col)) col = c;
        }
        seam[r] = col;
    }
    return seam;
}

}  // namespace

REGISTER_SEAM_FINDER("greedy", find_seam);
