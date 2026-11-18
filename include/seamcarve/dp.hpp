// Dynamic Programming seam finder.
//
// The method registered as "dp" is dp::find_seam. The other functions solve the
// same recurrence in different ways (less memory, top-down, several threads) and
// all return a seam of minimum total energy (checked in tests/test_dp.cpp).
//
//   function                 approach                               time           memory
//   find_seam                full table, bottom-up                  O(H*W)         H*W doubles
//   find_seam_top_down       recursion + memo                       O(H*W)         H*W doubles
//   find_seam_two_rows       2 rows of costs + 1-byte trace table   O(H*W)         2 rows + H*W bytes
//   find_seam_linear_space   divide and conquer (Hirschberg)        O(H*W*log H)   O(W + H)
//   find_seam_threaded       full table, each row split across      O(H*W)         H*W doubles
//                            threads, one barrier per row
#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace seamcarve::dp {

// A 2D table of doubles, used like a 2D array: table[i][j].
// All cells live in one block of memory; a vector of vectors would allocate every
// row separately, which is noticeably slower for tables with millions of cells.
// The cells are not initialised (the caller writes every cell before reading it),
// which avoids an extra pass over the whole table.
struct Table {
    Table() = default;
    Table(int rows, int cols) { resize(rows, cols); }
    Table(int rows, int cols, double value) : Table(rows, cols) {
        std::fill(cells.get(), cells.get() + static_cast<std::size_t>(rows) * cols, value);
    }

    // Changes the size of the table. The memory is only replaced when the new size
    // does not fit, so a table reused for many seams is allocated once.
    void resize(int new_rows, int new_cols) {
        rows = new_rows;
        cols = new_cols;
        const std::size_t needed = static_cast<std::size_t>(rows) * cols;
        if (needed > capacity) {
            cells.reset(new double[needed]);
            capacity = needed;
        }
    }

    double* operator[](int i) { return &cells[static_cast<std::size_t>(i) * cols]; }
    const double* operator[](int i) const { return &cells[static_cast<std::size_t>(i) * cols]; }

    int rows = 0, cols = 0;
    std::size_t capacity = 0;
    std::unique_ptr<double[]> cells;
};

// The full DP table, padded with one INF column on each side:
// table[i][j + 1] = cost of the cheapest seam from row 0 down to pixel (i, j).
Table cumulative_energy(const cv::Mat& energy);

// Recovers the seam from a full table, starting at the cheapest pixel of the last row.
std::vector<int> backtrack(const Table& table);

std::vector<int> find_seam(const cv::Mat& energy);
std::vector<int> find_seam_top_down(const cv::Mat& energy);
std::vector<int> find_seam_two_rows(const cv::Mat& energy);
std::vector<int> find_seam_linear_space(const cv::Mat& energy);
std::vector<int> find_seam_threaded(const cv::Mat& energy, int workers = 4);

struct Variant {
    std::string name;
    std::function<std::vector<int>(const cv::Mat&)> fn;
};

// All variants by name ("threaded" uses 4 threads).
const std::vector<Variant>& variants();

}  // namespace seamcarve::dp
