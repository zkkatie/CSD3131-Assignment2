// Tests for the Dynamic Programming seam finder (src/methods/dp.cpp).
//
//   ./build/tests dp        runs only these tests (every name starts with "dp_")
//
// Coverage:
//   * DP table and seam for a hand-computed example
//   * optimality of every variant against exhaustive search on small grids
//   * agreement between variants (equal cost; identical seam where the tie rule is shared)
//   * edge cases: 1x1, 1xW, Hx1, ties, very large values, non-continuous input (ROI)
//   * end-to-end carving with every variant
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <random>

#include "seamcarve/dp.hpp"
#include "seamcarve/seamcarve.hpp"
#include "testing.hpp"

using namespace seamcarve;

namespace {

// Variants that fill the table identically and share the leftmost tie rule, so they
// must return the same seam. linear_space is only required to match the cost.
const std::vector<std::string> SAME_TIE_RULE = {"full", "top_down", "two_rows", "threaded"};

cv::Mat random_energy(int h, int w, unsigned seed) {
    cv::Mat e(h, w, CV_64F);
    cv::RNG(seed + 1).fill(e, cv::RNG::UNIFORM, 0.0, 100.0);
    return e;
}

cv::Mat from_rows(const std::vector<std::vector<double>>& rows) {
    cv::Mat e(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()), CV_64F);
    for (int r = 0; r < e.rows; ++r)
        for (int c = 0; c < e.cols; ++c) e.at<double>(r, c) = rows[r][c];
    return e;
}

// Cost of the cheapest seam by exhaustive search (exponential; tiny grids only).
double brute_force(const cv::Mat& energy) {
    const int h = energy.rows, w = energy.cols;
    double best = std::numeric_limits<double>::infinity();
    std::function<void(int, int, double)> search = [&](int row, int col, double total) {
        total += energy.at<double>(row, col);
        if (row == h - 1) {
            best = std::min(best, total);
            return;
        }
        for (int nc = col - 1; nc <= col + 1; ++nc)
            if (nc >= 0 && nc < w) search(row + 1, nc, total);
    };
    for (int c = 0; c < w; ++c) search(0, c, 0.0);
    return best;
}

bool contains(const std::vector<std::string>& names, const std::string& n) {
    return std::find(names.begin(), names.end(), n) != names.end();
}

}  // namespace

// ------------------------------------------------------------ hand-computed example
//
//   e =  1  4  3        dp =  1   4   3        base case
//        5  2  9              6   3  12        5+min(1,4)  2+min(1,4,3)  9+min(4,3)
//        3  8  1              6  11   4        3+min(6,3)  8+min(6,3,12) 1+min(3,12)
//
// Cheapest end: column 2 (dp = 4). Backtracking: row 1 -> column 1, row 0 -> column 0.

TEST(dp_cumulative_table_hand_example) {
    const dp::Table table = dp::cumulative_energy(from_rows({{1, 4, 3}, {5, 2, 9}, {3, 8, 1}}));
    const double expected[3][3] = {{1, 4, 3}, {6, 3, 12}, {6, 11, 4}};
    CHECK(table.rows == 3 && table.cols == 5);  // one INF padding column on each side
    for (int i = 0; i < 3; ++i) {
        CHECK(std::isinf(table[i][0]) && std::isinf(table[i][4]));
        for (int j = 0; j < 3; ++j) CHECK(table[i][j + 1] == expected[i][j]);  // pixel (i, j) is at [i][j + 1]
    }
}

TEST(dp_hand_example_seam_every_variant) {
    cv::Mat e = from_rows({{1, 4, 3}, {5, 2, 9}, {3, 8, 1}});
    for (const auto& v : dp::variants()) {
        CHECK((v.fn(e) == std::vector<int>{0, 1, 2}));
        CHECK(seam_cost(e, v.fn(e)) == 4);
    }
}

TEST(dp_is_registered_as_dp) {
    CHECK(has_method("dp"));
    cv::Mat e = random_energy(30, 20, 1);
    CHECK(get_method("dp")(e) == dp::find_seam(e));
}

// ------------------------------------------------------------ optimality

TEST(dp_every_variant_matches_brute_force) {
    std::mt19937 rng(42);
    for (int trial = 0; trial < 40; ++trial) {
        const int h = std::uniform_int_distribution<int>(1, 7)(rng);
        const int w = std::uniform_int_distribution<int>(1, 6)(rng);
        cv::Mat e = random_energy(h, w, trial);
        const double best = brute_force(e);
        for (const auto& v : dp::variants()) {
            std::vector<int> seam = v.fn(e);
            validate_seam(seam, h, w);
            CHECK_NEAR(seam_cost(e, seam), best, 1e-9);
        }
    }
}

TEST(dp_avoids_the_greedy_trap) {
    // The cheapest top pixel leads into high-energy rows; the optimal seam is the right column.
    cv::Mat e = from_rows({{0, 9, 9, 1}, {9, 9, 9, 1}, {100, 100, 9, 1}});
    for (const auto& v : dp::variants()) CHECK((v.fn(e) == std::vector<int>{3, 3, 3}));
}

TEST(dp_follows_a_cheap_diagonal) {
    cv::Mat e(6, 6, CV_64F, cv::Scalar(50));
    for (int i = 0; i < 6; ++i) e.at<double>(i, i) = 0;
    for (const auto& v : dp::variants()) CHECK((v.fn(e) == std::vector<int>{0, 1, 2, 3, 4, 5}));
}

TEST(dp_bends_around_objects_that_block_every_column) {
    // An object on the right of the top half and on the left of the bottom half
    // block every straight column. Stepping from column 4 to 5 between the halves
    // avoids both, giving a zero-cost seam.
    cv::Mat e(10, 10, CV_64F, cv::Scalar(0));
    cv::RNG rng(3);
    rng.fill(e(cv::Rect(5, 0, 5, 5)), cv::RNG::UNIFORM, 1.0, 100.0);
    rng.fill(e(cv::Rect(0, 5, 5, 5)), cv::RNG::UNIFORM, 1.0, 100.0);
    cv::Mat column_sums;
    cv::reduce(e, column_sums, 0, cv::REDUCE_SUM, CV_64F);
    double min_column = 0;
    cv::minMaxLoc(column_sums, &min_column);
    CHECK(min_column > 0);
    for (const auto& v : dp::variants()) {
        std::vector<int> seam = v.fn(e);
        CHECK(seam_cost(e, seam) == 0);
        CHECK(seam[4] == 4 && seam[5] == 5);
    }
}

// ------------------------------------------------------------ agreement between variants

TEST(dp_variants_agree) {
    const int sizes[][2] = {{1, 1}, {1, 9}, {9, 1}, {2, 2}, {17, 23}, {40, 31}, {120, 90}, {300, 200}};
    for (auto [h, w] : sizes)
        for (unsigned seed = 0; seed < 3; ++seed) {
            cv::Mat e = random_energy(h, w, seed * 1000 + h);
            const std::vector<int> ref = dp::find_seam(e);
            const double ref_cost = seam_cost(e, ref);
            for (const auto& v : dp::variants()) {
                std::vector<int> seam = v.fn(e);
                validate_seam(seam, h, w);
                CHECK_NEAR(seam_cost(e, seam), ref_cost, 1e-6);
                if (contains(SAME_TIE_RULE, v.name)) CHECK(seam == ref);
            }
        }
}

TEST(dp_linear_space_every_height) {
    // Every height from 1 to 40 exercises each way the rows can be split in half.
    for (int h = 1; h <= 40; ++h) {
        cv::Mat e = random_energy(h, 9, h);
        std::vector<int> seam = dp::find_seam_linear_space(e);
        validate_seam(seam, h, 9);
        CHECK_NEAR(seam_cost(e, seam), seam_cost(e, dp::find_seam(e)), 1e-9);
    }
}

TEST(dp_threaded_any_worker_count) {
    // Includes more workers than columns.
    for (int w : {1, 3, 13, 64})
        for (int workers : {1, 2, 3, 8, 50}) {
            cv::Mat e = random_energy(25, w, workers + w);
            CHECK(dp::find_seam_threaded(e, workers) == dp::find_seam(e));
        }
}

// ------------------------------------------------------------ edge cases

TEST(dp_single_pixel_row_and_column) {
    for (const auto& v : dp::variants()) {
        CHECK((v.fn(from_rows({{5}})) == std::vector<int>{0}));
        CHECK((v.fn(from_rows({{4, 2, 7, 2.5}})) == std::vector<int>{1}));
        CHECK((v.fn(random_energy(7, 1, 0)) == std::vector<int>(7, 0)));
    }
}

TEST(dp_ties_choose_leftmost) {
    cv::Mat flat(5, 4, CV_64F, cv::Scalar(0));
    for (const auto& v : dp::variants())
        if (contains(SAME_TIE_RULE, v.name)) CHECK((v.fn(flat) == std::vector<int>(5, 0)));
}

TEST(dp_flat_energy_any_valid_seam) {
    cv::Mat ones(6, 5, CV_64F, cv::Scalar(1));
    for (const auto& v : dp::variants()) {
        std::vector<int> seam = v.fn(ones);
        validate_seam(seam, 6, 5);
        CHECK(seam_cost(ones, seam) == 6);
    }
}

TEST(dp_huge_energies) {
    cv::Mat e = random_energy(6, 5, 2) * 1e300;
    for (const auto& v : dp::variants()) validate_seam(v.fn(e), 6, 5);
}

TEST(dp_does_not_modify_input) {
    cv::Mat e = random_energy(8, 6, 1), before = e.clone();
    for (const auto& v : dp::variants()) {
        v.fn(e);
        CHECK(cv::norm(e, before, cv::NORM_INF) == 0);
    }
}

TEST(dp_works_on_a_non_continuous_view) {
    // An ROI into a larger matrix: rows are not contiguous in memory.
    cv::Mat big = random_energy(30, 40, 9);
    cv::Mat view = big(cv::Rect(5, 3, 20, 15));
    CHECK(!view.isContinuous());
    const std::vector<int> expected = dp::find_seam(view.clone());
    for (const auto& v : dp::variants()) {
        std::vector<int> seam = v.fn(view);
        CHECK_NEAR(seam_cost(view, seam), seam_cost(view, expected), 1e-9);
    }
}

TEST(dp_rejects_bad_input) {
    CHECK_THROWS(dp::find_seam(cv::Mat()));
    CHECK_THROWS(dp::find_seam(cv::Mat(3, 3, CV_32F, cv::Scalar(0))));
}

// ------------------------------------------------------------ properties

TEST(dp_never_worse_than_any_straight_column_or_random_seam) {
    std::mt19937 rng(1);
    for (unsigned seed = 0; seed < 10; ++seed) {
        cv::Mat e = random_energy(25, 15, seed);
        const double best = seam_cost(e, dp::find_seam(e));
        for (int c = 0; c < 15; ++c) CHECK(best <= seam_cost(e, std::vector<int>(25, c)) + 1e-9);
        for (int trial = 0; trial < 50; ++trial) {
            std::vector<int> s{std::uniform_int_distribution<int>(0, 14)(rng)};
            for (int i = 1; i < 25; ++i)
                s.push_back(std::clamp(s.back() + std::uniform_int_distribution<int>(-1, 1)(rng), 0, 14));
            CHECK(best <= seam_cost(e, s) + 1e-9);
        }
    }
}

TEST(dp_last_row_minimum_is_the_best_seam_cost) {
    cv::Mat e = random_energy(5, 4, 7);
    const dp::Table table = dp::cumulative_energy(e);
    const double lowest = *std::min_element(table[4], table[4] + table.cols);  // padding is INF
    CHECK_NEAR(lowest, brute_force(e), 1e-9);
}

// ------------------------------------------------------------ end to end

TEST(dp_carve_keeps_textured_object) {
    // Flat background with a striped bar: only background should be removed.
    cv::Mat img(30, 60, CV_8UC3, cv::Scalar::all(30));
    for (int r = 0; r < 30; r += 2) img.row(r).colRange(28, 31).setTo(cv::Scalar::all(255));
    cv::Mat out = carve(img, 40, 30, get_method("dp")).image;
    int bar_columns = 0;
    for (int c = 0; c < out.cols; ++c) {
        bool has_white = false;
        for (int r = 0; r < out.rows; ++r) has_white |= out.at<cv::Vec3b>(r, c) == cv::Vec3b(255, 255, 255);
        bar_columns += has_white;
    }
    CHECK(bar_columns == 3);
}

TEST(dp_carve_vertical_and_horizontal_with_every_variant) {
    cv::Mat img(24, 30, CV_8UC3);
    cv::RNG(4).fill(img, cv::RNG::UNIFORM, 0, 256);
    CarveResult ref = carve(img, 22, 18, dp::find_seam);
    CHECK(ref.image.rows == 18 && ref.image.cols == 22 && ref.stats.seams_removed == 8 + 6);
    for (const auto& v : dp::variants()) {
        CarveResult res = carve(img, 22, 18, v.fn);
        CHECK_NEAR(res.stats.total_seam_cost, ref.stats.total_seam_cost, 1e-6);
        if (contains(SAME_TIE_RULE, v.name)) CHECK(cv::norm(res.image, ref.image, cv::NORM_INF) == 0);
    }
}

TEST(dp_seam_not_worse_than_baseline_or_greedy) {
    cv::Mat img(30, 40, CV_8UC3);
    cv::RNG(3).fill(img, cv::RNG::UNIFORM, 0, 256);
    cv::Mat e = compute_energy(img);
    const double dp_cost = seam_cost(e, dp::find_seam(e));
    for (const std::string other : {"baseline", "greedy"})
        if (has_method(other)) CHECK(dp_cost <= seam_cost(e, get_method(other)(e)) + 1e-9);
}

TEST(dp_matches_graphcut_cost) {
    // Both methods are optimal, so their seams must have the same cost.
    if (!has_method("graphcut")) SKIP("graphcut not available");
    std::mt19937 rng(11);
    for (int trial = 0; trial < 10; ++trial) {
        const int h = std::uniform_int_distribution<int>(2, 12)(rng);
        const int w = std::uniform_int_distribution<int>(2, 12)(rng);
        cv::Mat e = random_energy(h, w, 500 + trial);
        CHECK_NEAR(seam_cost(e, dp::find_seam(e)), seam_cost(e, get_method("graphcut")(e)), 1e-6);
    }
}
