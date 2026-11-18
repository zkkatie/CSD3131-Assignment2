// Dynamic Programming seam finder.
//
// Finds the vertical seam with the minimum total energy. Horizontal seams are
// handled by carve(), which transposes the image, so only vertical seams are needed.
//
// Formulation (SRTBOT)
//   Subproblem   dp[i][j] = minimum total energy of a seam that starts anywhere in
//                row 0 and ends at pixel (i, j).
//   Relation     dp[i][j] = e[i][j] + min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1])
//   Order        Row i only needs row i-1, so rows are filled top to bottom.
//   Base case    dp[0][j] = e[0][j]
//   Answer       The smallest value in the last row; the seam itself is recovered
//                by walking back up the table (backtracking).
//   Time         H*W subproblems, O(1) each = O(H * W)
//
// Padding: every table row has one extra column on each side holding INF, so
// pixel (i, j) is stored at dp[i][j + 1] (the same off-by-one shift the lecture
// code uses for LCS and Edit Distance). The neighbours of the first and last
// column are then INF and are never chosen, so the relation needs no border checks.
//
// Tie-breaking: when two choices cost the same, the leftmost one is taken. All
// variants except find_seam_linear_space use this rule and return the same seam.
#include "seamcarve/dp.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <thread>

#include "seamcarve/methods.hpp"

namespace seamcarve::dp {

namespace {

const double INF = std::numeric_limits<double>::infinity();
const int ANY = -1;  // "no fixed column" (used by the linear-space variant)

double min3(double a, double b, double c) { return std::min(std::min(a, b), c); }

void check_input(const cv::Mat& energy) {
    if (energy.empty() || energy.type() != CV_64F)
        throw std::invalid_argument("dp: energy must be a non-empty CV_64F matrix");
}

// The relation applied to one whole row i (both rows are padded):
//   row[j + 1] = e[i][j] + min(above[j], above[j + 1], above[j + 2])
// where above[j], above[j + 1] and above[j + 2] are the up-left, up and up-right
// neighbours of pixel j. The loop has no branches, so the compiler can vectorise it.
void fill_row(const cv::Mat& energy, int i, const double* above, double* row) {
    const double* e = energy.ptr<double>(i);  // row i of the energy map
    for (int j = 0; j < energy.cols; ++j) row[j + 1] = e[j] + min3(above[j], above[j + 1], above[j + 2]);
}

// Copies row i of the energy map into a padded row. Only `col` is copied if it is
// fixed; every other entry stays INF (used by the linear-space variant).
void copy_energy_row(const cv::Mat& energy, int i, double* row, int col = ANY) {
    for (int j = 0; j < energy.cols; ++j)
        if (col == ANY || j == col) row[j + 1] = energy.at<double>(i, j);
}

// Which parent pixel in the padded row above is cheapest for column j:
// -1 = up-left, 0 = up, +1 = up-right. The leftmost wins ties.
int best_parent(const double* above, int j) {
    int best = -1;
    if (above[j + 1] < above[j + 1 + best]) best = 0;
    if (above[j + 2] < above[j + 1 + best]) best = 1;
    return best;
}

// Column of the smallest value in a padded row of width w (leftmost on ties).
int cheapest_column(const double* row, int w) {
    int best = 0;
    for (int j = 1; j < w; ++j)
        if (row[j + 1] < row[best + 1]) best = j;
    return best;
}

// Walks up a padded table from column `end` in its last row.
std::vector<int> backtrack_from(const Table& dp, int end) {
    std::vector<int> seam(dp.rows);
    int j = end;
    seam[dp.rows - 1] = j;
    for (int i = dp.rows - 1; i > 0; --i) {
        j += best_parent(dp[i - 1], j);
        seam[i - 1] = j;
    }
    return seam;
}

}  // namespace

// --------------------------------------------------------------------- full table

namespace {

// Resizes `dp` to H x (W + 2) and fills it with the full DP table.
void fill_table(const cv::Mat& energy, Table& dp) {
    const int h = energy.rows, w = energy.cols;
    dp.resize(h, w + 2);                // every cell is written below before it is read
    dp[0][0] = dp[0][w + 1] = INF;      // padding
    copy_energy_row(energy, 0, dp[0]);  // base case
    for (int i = 1; i < h; ++i) {       // top to bottom
        dp[i][0] = dp[i][w + 1] = INF;
        fill_row(energy, i, dp[i - 1], dp[i]);
    }
}

}  // namespace

Table cumulative_energy(const cv::Mat& energy) {
    check_input(energy);
    Table dp;
    fill_table(energy, dp);
    return dp;
}

std::vector<int> backtrack(const Table& dp) {
    const int w = dp.cols - 2;
    return backtrack_from(dp, cheapest_column(dp[dp.rows - 1], w));
}

// Time O(H * W), space O(H * W).
// The table is kept between calls and reused (one per thread). carve() calls this
// once per seam with almost the same image size, and asking the operating system
// for a fresh table of several megabytes every time costs more than the DP itself
// on large images.
std::vector<int> find_seam(const cv::Mat& energy) {
    check_input(energy);
    static thread_local Table dp;
    fill_table(energy, dp);
    return backtrack(dp);
}

// --------------------------------------------------------------------- top-down

namespace {

// cost(i, j) with memoisation, written like the lecture's dfs(i, ...) + memo.
// Row 0 and the INF padding are filled in before the first call, so they act as
// the base cases; NaN marks a value that has not been computed yet.
double cost(int i, int j, const cv::Mat& energy, Table& memo) {
    double& result = memo[i][j + 1];
    if (!std::isnan(result)) return result;  // reuse
    result = energy.at<double>(i, j) +
             min3(cost(i - 1, j - 1, energy, memo), cost(i - 1, j, energy, memo), cost(i - 1, j + 1, energy, memo));
    return result;
}

}  // namespace

// Same table and O(H * W) time as find_seam, computed top-down. The recursion is
// H calls deep and every cell costs a function call, so it is slower in practice.
std::vector<int> find_seam_top_down(const cv::Mat& energy) {
    check_input(energy);
    const int h = energy.rows, w = energy.cols;
    Table memo(h, w + 2, std::numeric_limits<double>::quiet_NaN());
    for (int i = 0; i < h; ++i) memo[i][0] = memo[i][w + 1] = INF;  // outside the image
    copy_energy_row(energy, 0, memo[0]);                              // base case

    for (int j = 0; j < w; ++j) cost(h - 1, j, energy, memo);
    return backtrack(memo);
}

// --------------------------------------------------------------------- two rows

// Row i only reads row i-1, so two rows are enough to compute the costs (the
// rolling array from the Fibonacci and 0/1 knapsack slides). To recover the seam
// we also record which parent each pixel used: -1, 0 or +1, stored in one byte
// instead of an 8-byte double, so this "trace" table is 8 times smaller than dp.
//
// Time O(H * W), space 2 rows of doubles + H*W bytes.
std::vector<int> find_seam_two_rows(const cv::Mat& energy) {
    check_input(energy);
    const int h = energy.rows, w = energy.cols;
    std::vector<double> above(w + 2, INF), current(w + 2, INF);
    std::vector<signed char> parent(static_cast<std::size_t>(h) * w);  // parent of (i, j) at [i * w + j]

    copy_energy_row(energy, 0, above.data());  // base case
    for (int i = 1; i < h; ++i) {
        const double* e = energy.ptr<double>(i);
        signed char* trace = &parent[static_cast<std::size_t>(i) * w];
        for (int j = 0; j < w; ++j) {
            const int k = best_parent(above.data(), j);
            trace[j] = static_cast<signed char>(k);
            current[j + 1] = e[j] + above[j + 1 + k];
        }
        std::swap(above, current);  // the new row becomes "above" for the next one
    }

    std::vector<int> seam(h);
    int j = cheapest_column(above.data(), w);
    seam[h - 1] = j;
    for (int i = h - 1; i > 0; --i) {
        j += parent[static_cast<std::size_t>(i) * w + j];
        seam[i - 1] = j;
    }
    return seam;
}

// --------------------------------------------------------------------- linear space

namespace {

// One-row DP from row `from` to row `to` (downward or upward). Only `start_col`
// may be used in the first row, unless it is ANY. Returns the last padded row.
std::vector<double> sweep(const cv::Mat& energy, int from, int to, int start_col) {
    const int w = energy.cols;
    std::vector<double> prev(w + 2, INF), cur(w + 2, INF);
    copy_energy_row(energy, from, prev.data(), start_col);

    const int step = (to >= from) ? 1 : -1;
    for (int i = from + step; i != to + step; i += step) {
        fill_row(energy, i, prev.data(), cur.data());
        std::swap(prev, cur);
    }
    return prev;
}

// Fills seam[r0..r1] with the cheapest path from (r0, c_start) to (r1, c_end).
void solve(const cv::Mat& energy, int r0, int r1, int c_start, int c_end, std::vector<int>& seam) {
    const int w = energy.cols;

    if (r1 - r0 <= 1) {  // base case: one or two rows, solved with a tiny table
        Table block(r1 - r0 + 1, w + 2, INF);
        copy_energy_row(energy, r0, block[0], c_start);
        if (r1 > r0) fill_row(energy, r1, block[0], block[1]);
        const int end = (c_end == ANY) ? cheapest_column(block[block.rows - 1], w) : c_end;
        const std::vector<int> part = backtrack_from(block, end);
        std::copy(part.begin(), part.end(), seam.begin() + r0);
        return;
    }

    const int mid = (r0 + r1) / 2;
    int c = 0;
    {
        // F[j + 1]: cheapest path from the start down to (mid, j)
        // B[j + 1]: cheapest path from (mid, j) down to the end (computed upward)
        // e(mid, j) is counted in both, so it is subtracted once.
        const std::vector<double> F = sweep(energy, r0, mid, c_start);
        const std::vector<double> B = sweep(energy, r1, mid, c_end);
        double best = INF;
        for (int j = 0; j < w; ++j) {
            const double through = F[j + 1] + B[j + 1] - energy.at<double>(mid, j);
            if (through < best) {
                best = through;
                c = j;
            }
        }
    }  // F and B are freed here, so memory stays O(W) during the recursion

    solve(energy, r0, mid, c_start, c, seam);
    solve(energy, mid, r1, c, c_end, seam);
}

}  // namespace

// Divide and conquer combined with DP (Hirschberg's technique): finds the seam
// with O(W + H) memory and no H x W table.
//   1. Run the one-row DP down from the top to the middle row, and up from the
//      bottom to the middle row.
//   2. The best seam crosses the middle row where (cost from the top) + (cost to
//      the bottom) is smallest. Fix that pixel.
//   3. Solve the top half and the bottom half the same way (recursion).
// T(H) = 2 T(H/2) + O(H * W) = O(H * W * log H) by the Master Theorem, like merge sort.
std::vector<int> find_seam_linear_space(const cv::Mat& energy) {
    check_input(energy);
    std::vector<int> seam(energy.rows);
    solve(energy, 0, energy.rows - 1, ANY, ANY, seam);
    return seam;
}

// --------------------------------------------------------------------- threads

namespace {

// Makes `n` threads wait for each other. Every thread calls wait() after finishing
// its part of a row; the last one to arrive lets all of them continue.
// Waiting threads keep checking a counter ("spinning") instead of going to sleep,
// because a row takes only microseconds and sleeping and waking up would cost more.
// After many checks they start yielding the core, in case another thread needs it.
class Barrier {
public:
    explicit Barrier(int n) : n_(n) {}

    void wait() {
        const int round = round_;
        if (arrived_.fetch_add(1) == n_ - 1) {  // the last thread to arrive
            arrived_ = 0;
            ++round_;  // releases the waiting threads
        } else {
            for (int checks = 0; round_ == round; ++checks)
                if (checks > 2000) std::this_thread::yield();
        }
    }

private:
    const int n_;
    std::atomic<int> arrived_{0};  // threads that have reached the barrier this round
    std::atomic<int> round_{0};    // increases by one every time all threads have arrived
};

}  // namespace

// Same table as find_seam, with every row split into `workers` column strips that
// are filled by different threads at the same time. Rows still have to be done in
// order (row i needs row i-1), so the threads meet at a barrier after every row.
std::vector<int> find_seam_threaded(const cv::Mat& energy, int workers) {
    check_input(energy);
    const int h = energy.rows, w = energy.cols;
    const int p = std::clamp(workers, 1, w);  // never more threads than columns
    static thread_local Table cached;  // reused between calls, like in find_seam
    Table& dp = cached;  // a named reference, so the worker threads use this thread's table
    dp.resize(h, w + 2);  // not pre-filled: each thread writes its own strip
    dp[0][0] = dp[0][w + 1] = INF;      // padding
    copy_energy_row(energy, 0, dp[0]);  // base case
    Barrier barrier(p);

    // Thread t fills columns [first, last) of every row. The first and last
    // threads also write the padding at the two ends of the row.
    auto work = [&](int t) {
        const int first = w * t / p, last = w * (t + 1) / p;
        for (int i = 1; i < h; ++i) {
            const double* e = energy.ptr<double>(i);
            const double* above = dp[i - 1];
            double* row = dp[i];
            if (first == 0) row[0] = INF;
            if (last == w) row[w + 1] = INF;
            for (int j = first; j < last; ++j) row[j + 1] = e[j] + min3(above[j], above[j + 1], above[j + 2]);
            barrier.wait();  // row i must be complete before anyone starts row i + 1
        }
    };

    std::vector<std::thread> threads;
    for (int t = 1; t < p; ++t) threads.emplace_back(work, t);
    work(0);  // the calling thread does strip 0 itself
    for (std::thread& thread : threads) thread.join();
    return backtrack(dp);
}

// --------------------------------------------------------------------- list

const std::vector<Variant>& variants() {
    static const std::vector<Variant> list = {
        {"full", find_seam},
        {"top_down", find_seam_top_down},
        {"two_rows", find_seam_two_rows},
        {"linear_space", find_seam_linear_space},
        {"threaded", [](const cv::Mat& e) { return find_seam_threaded(e); }},
    };
    return list;
}

}  // namespace seamcarve::dp

REGISTER_SEAM_FINDER("dp", seamcarve::dp::find_seam);
