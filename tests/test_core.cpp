// Tests for the shared framework (Katie) and for every seam finder.
//
// Build and run:   cmake -S . -B build && cmake --build build && ./build/tests
//
// Tests for a seam finder are skipped until its file (e.g. src/methods/dp.cpp)
// is added, then start running automatically.
#include <filesystem>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "seamcarve/seamcarve.hpp"
#include "testing.hpp"

using namespace seamcarve;
namespace fs = std::filesystem;

// ---------------------------------------------------------------- helpers

static cv::Mat random_image(int h, int w, unsigned seed = 0, int channels = 3) {
    cv::Mat img(h, w, CV_8UC(channels));
    cv::RNG rng(seed + 1);
    rng.fill(img, cv::RNG::UNIFORM, 0, 256);
    return img;
}

static std::vector<int> random_seam(int h, int w, std::mt19937& rng) {
    std::vector<int> s{std::uniform_int_distribution<int>(0, w - 1)(rng)};
    std::uniform_int_distribution<int> step(-1, 1);
    for (int i = 1; i < h; ++i) s.push_back(std::clamp(s.back() + step(rng), 0, w - 1));
    return s;
}

static bool same(const cv::Mat& a, const cv::Mat& b) {
    return a.size() == b.size() && a.type() == b.type() && (a.empty() || cv::norm(a, b, cv::NORM_INF) == 0);
}

static bool close(const cv::Mat& a, const cv::Mat& b, double tol = 1e-6) {
    return a.size() == b.size() && a.type() == b.type() && (a.empty() || cv::norm(a, b, cv::NORM_INF) <= tol);
}

static double brute_force_min_cost(const cv::Mat& energy) {
    // Tries every seam (exponentially many) - only for tiny grids.
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

static SeamFinder finder_or_skip(const std::string& name) {
    if (!has_method(name)) SKIP(name + " not added yet (src/methods/" + name + ".cpp)");
    return get_method(name);
}

static SeamFinder baseline() { return get_method("baseline"); }

static cv::Mat energy_from_rows(const std::vector<std::vector<double>>& rows) {
    cv::Mat e(static_cast<int>(rows.size()), static_cast<int>(rows[0].size()), CV_64F);
    for (int r = 0; r < e.rows; ++r)
        for (int c = 0; c < e.cols; ++c) e.at<double>(r, c) = rows[r][c];
    return e;
}

static const EnergyFunction ENERGY_FUNCTIONS[] = {EnergyFunction::Sobel, EnergyFunction::Gradient};

// ---------------------------------------------------------------- energy

TEST(energy_shape_and_nonnegative) {
    for (EnergyFunction fn : ENERGY_FUNCTIONS) {
        cv::Mat e = compute_energy(random_image(13, 17), fn);
        double lo = 0;
        cv::minMaxLoc(e, &lo);
        CHECK(e.rows == 13 && e.cols == 17 && e.type() == CV_64F && lo >= 0);
    }
}

TEST(flat_image_has_zero_energy) {
    for (EnergyFunction fn : ENERGY_FUNCTIONS)
        CHECK(cv::norm(compute_energy(cv::Mat(8, 9, CV_8UC3, cv::Scalar::all(77)), fn), cv::NORM_INF) == 0);
}

TEST(edge_has_higher_energy_than_flat_area) {
    for (EnergyFunction fn : ENERGY_FUNCTIONS) {
        cv::Mat img(10, 10, CV_8UC3, cv::Scalar::all(0));
        img.colRange(5, 10).setTo(cv::Scalar::all(255));
        cv::Mat e = compute_energy(img, fn);
        double edge_min = 0;
        cv::minMaxLoc(e.colRange(4, 6), &edge_min);
        CHECK(edge_min > 0);
        CHECK(cv::norm(e.colRange(0, 3), cv::NORM_INF) == 0 && cv::norm(e.colRange(7, 10), cv::NORM_INF) == 0);
    }
}

TEST(gradient_energy_hand_computed) {
    cv::Mat g = (cv::Mat_<double>(2, 3) << 0, 1, 4, 9, 16, 25);
    cv::Mat e = energy_from_gray(g, EnergyFunction::Gradient);
    // pixel (0,1): gx = 4 - 0 = 4,  gy = 16 - 1 (top clamps to itself) = 15  -> 19
    // pixel (1,0): gx = 16 - 9 (left clamps) = 7,  gy = 9 - 0 = 9            -> 16
    CHECK(e.at<double>(0, 1) == 19 && e.at<double>(1, 0) == 16);
}

TEST(sobel_matches_opencv_reflect_border) {
    // Our clamped border is the same as OpenCV's BORDER_REFLECT for a 3x3 kernel.
    cv::Mat g = to_gray(random_image(11, 14, 3)), gx, gy;
    cv::Sobel(g, gx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_REFLECT);
    cv::Sobel(g, gy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_REFLECT);
    CHECK(close(energy_from_gray(g), cv::abs(gx) + cv::abs(gy)));
}

TEST(energy_at_matches_full_map) {
    const int sizes[][2] = {{1, 1}, {1, 6}, {6, 1}, {2, 2}, {9, 13}};
    for (EnergyFunction fn : ENERGY_FUNCTIONS)
        for (auto [h, w] : sizes) {
            cv::Mat g = to_gray(random_image(h, w, h * 31 + w)), e = energy_from_gray(g, fn);
            for (int r = 0; r < h; ++r)
                for (int c = 0; c < w; ++c) CHECK_NEAR(energy_at(g, r, c, fn), e.at<double>(r, c), 1e-6);
        }
}

TEST(greyscale_bgra_and_bgr_input) {
    cv::Mat bgr = random_image(6, 7, 1), grey, bgra;
    cv::cvtColor(bgr, grey, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
    CHECK(close(compute_energy(grey), compute_energy(bgr)));
    CHECK(close(compute_energy(bgra), compute_energy(bgr)));
}

TEST(unknown_energy_names_rejected) {
    CHECK_THROWS(parse_energy_function("magic"));
    CHECK_THROWS(parse_energy_mode("sometimes"));
    CHECK(parse_energy_function("gradient") == EnergyFunction::Gradient);
    CHECK(parse_energy_mode("full") == EnergyMode::Full);
}

// ------------------------------------------------- incremental energy update

TEST(incremental_energy_equals_full_recompute) {
    // Remove random seams until 1 column is left; after EVERY removal both ways must agree.
    const int sizes[][2] = {{1, 8}, {2, 2}, {5, 3}, {12, 20}, {30, 7}};
    for (EnergyFunction fn : ENERGY_FUNCTIONS)
        for (auto [h, w] : sizes) {
            std::mt19937 rng(h * 100 + w);
            cv::Mat gray = to_gray(random_image(h, w, h + w));
            cv::Mat energy = energy_from_gray(gray, fn);
            while (gray.cols > 1) {
                std::vector<int> seam = random_seam(h, gray.cols, rng);
                gray = remove_vertical_seam(gray, seam);
                energy = remove_vertical_seam(energy, seam);
                update_energy(energy, gray, seam, fn);
                CHECK(close(energy, energy_from_gray(gray, fn)));
            }
        }
}

TEST(incremental_energy_extreme_seams) {
    const int h = 12, w = 10;
    std::vector<std::vector<int>> seams(4, std::vector<int>(h));
    for (int i = 0; i < h; ++i) {
        seams[0][i] = 0;                    // left edge
        seams[1][i] = w - 1;                // right edge
        seams[2][i] = 4 + i % 2;            // zig-zag
        seams[3][i] = std::min(i, w - 1);   // diagonal
    }
    cv::Mat gray = to_gray(random_image(h, w, 7));
    for (const auto& seam : seams) {
        cv::Mat new_gray = remove_vertical_seam(gray, seam);
        cv::Mat energy = remove_vertical_seam(energy_from_gray(gray), seam);
        update_energy(energy, new_gray, seam);
        CHECK(close(energy, energy_from_gray(new_gray)));
    }
}

// ---------------------------------------------------------------- seams

TEST(validate_seam_accepts_good_seams) {
    validate_seam({2, 3, 3, 2}, 4, 5);
    validate_seam({0}, 1, 1);
}

TEST(validate_seam_rejects_bad_seams) {
    CHECK_THROWS(validate_seam({0, 2, 2, 2}, 4, 5));   // jumps 2 columns
    CHECK_THROWS(validate_seam({0, 0, 0}, 4, 5));      // wrong length
    CHECK_THROWS(validate_seam({0, 0, 0, 5}, 4, 5));   // out of range (right)
    CHECK_THROWS(validate_seam({-1, 0, 0, 0}, 4, 5));  // out of range (left)
}

TEST(remove_vertical_seam_removes_exact_pixels) {
    cv::Mat img = (cv::Mat_<int>(3, 4) << 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    cv::Mat expected = (cv::Mat_<int>(3, 3) << 1, 2, 3, 4, 6, 7, 8, 9, 11);
    CHECK(same(remove_vertical_seam(img, {0, 1, 2}), expected));
}

TEST(remove_vertical_seam_colour_keeps_channels_together) {
    cv::Mat img = random_image(5, 6);
    std::vector<int> seam{1, 1, 2, 3, 3};
    cv::Mat out = remove_vertical_seam(img, seam);
    CHECK(out.rows == 5 && out.cols == 5 && out.type() == CV_8UC3);
    for (int r = 0; r < 5; ++r)
        for (int c = 0, src = 0; c < 5; ++c, ++src) {
            if (src == seam[r]) ++src;
            CHECK(out.at<cv::Vec3b>(r, c) == img.at<cv::Vec3b>(r, src));
        }
}

TEST(seam_cost_adds_energy_along_seam) {
    cv::Mat e = (cv::Mat_<double>(2, 3) << 0, 1, 2, 3, 4, 5);
    CHECK(seam_cost(e, {1, 2}) == 1 + 5);
}

TEST(draw_vertical_seam_does_not_modify_input) {
    cv::Mat img(3, 3, CV_8UC3, cv::Scalar::all(0));
    cv::Mat out = draw_vertical_seam(img, {0, 1, 2});
    CHECK(cv::sum(img)[0] == 0 && out.at<cv::Vec3b>(1, 1) == cv::Vec3b(0, 0, 255));
}

TEST(transpose_round_trip) {
    cv::Mat img = random_image(7, 9);
    CHECK(transpose(img).rows == 9 && transpose(img).cols == 7);
    CHECK(same(transpose(transpose(img)), img));
}

// ---------------------------------------------------------------- carve

TEST(carve_reaches_target_size) {
    CarveResult res = carve(random_image(30, 40), 32, 25, baseline());
    CHECK(res.image.rows == 25 && res.image.cols == 32 && res.image.type() == CV_8UC3);
    CHECK(res.stats.seams_removed == (40 - 32) + (30 - 25));
    std::vector<Direction> expected(8, Direction::Vertical);
    expected.insert(expected.end(), 5, Direction::Horizontal);
    CHECK(res.stats.seam_directions == expected);
    CHECK(res.stats.seam_costs.size() == 13 && res.stats.seam_coords.size() == 13);
}

TEST(carve_no_change_returns_same_image) {
    cv::Mat img = random_image(10, 10);
    CarveResult res = carve(img, 10, 10, baseline());
    CHECK(same(res.image, img) && res.stats.seams_removed == 0);
}

TEST(carve_does_not_modify_input) {
    cv::Mat img = random_image(10, 12), copy = img.clone();
    carve(img, 6, 7, baseline());
    CHECK(same(img, copy));
}

TEST(carve_down_to_1x1) {
    CarveResult res = carve(random_image(6, 5), 1, 1, baseline());
    CHECK(res.image.rows == 1 && res.image.cols == 1);
}

TEST(carve_greyscale_image) {
    CarveResult res = carve(random_image(10, 12, 0, 1), 7, 8, baseline());
    CHECK(res.image.rows == 8 && res.image.cols == 7 && res.image.type() == CV_8UC1);
}

TEST(carve_single_row_and_single_column_images) {
    CHECK(carve(random_image(1, 10), 4, 1, baseline()).image.cols == 4);
    CHECK(carve(random_image(10, 1), 1, 4, baseline()).image.rows == 4);
}

TEST(carve_rejects_invalid_target) {
    cv::Mat img = random_image(10, 10);
    CHECK_THROWS(carve(img, 11, 10, baseline()));
    CHECK_THROWS(carve(img, 10, 11, baseline()));
    CHECK_THROWS(carve(img, 0, 5, baseline()));
    CHECK_THROWS(carve(img, 5, 0, baseline()));
    CHECK_THROWS(carve(cv::Mat(), 1, 1, baseline()));
}

TEST(carve_rejects_invalid_seam_from_finder) {
    SeamFinder bad = [](const cv::Mat& e) {  // jumps 2 columns per row
        std::vector<int> s(e.rows);
        for (int i = 0; i < e.rows; ++i) s[i] = std::min(2 * i, e.cols - 1);
        return s;
    };
    CHECK_THROWS(carve(random_image(5, 12), 10, 5, bad));
}

TEST(incremental_and_full_modes_give_identical_results) {
    cv::Mat img = random_image(25, 30, 5);
    for (EnergyFunction fn : ENERGY_FUNCTIONS) {
        CarveOptions a_opt, b_opt;
        a_opt.energy_fn = b_opt.energy_fn = fn;
        a_opt.energy_mode = EnergyMode::Incremental;
        b_opt.energy_mode = EnergyMode::Full;
        CarveResult a = carve(img, 18, 20, baseline(), a_opt), b = carve(img, 18, 20, baseline(), b_opt);
        CHECK(same(a.image, b.image));
        CHECK(a.stats.seam_coords == b.stats.seam_coords);
        for (size_t i = 0; i < a.stats.seam_costs.size(); ++i)
            CHECK_NEAR(a.stats.seam_costs[i], b.stats.seam_costs[i], 1e-6);
    }
}

TEST(vertical_only_result_matches_removed_coordinates) {
    // Rebuild the output from the original by deleting the reported seam pixels.
    cv::Mat img = random_image(15, 20, 6);
    CarveResult res = carve(img, 12, 15, baseline());
    cv::Mat removed(15, 20, CV_8U, cv::Scalar(0));
    for (const auto& seam : res.stats.seam_coords) {
        CHECK(seam.size() == 15);
        for (const Pixel& p : seam) removed.at<uchar>(p.row, p.col) = 1;
    }
    for (int r = 0; r < 15; ++r)
        for (int c = 0, out_c = 0; c < 20; ++c)
            if (!removed.at<uchar>(r, c)) CHECK(res.image.at<cv::Vec3b>(r, out_c++) == img.at<cv::Vec3b>(r, c));
}

TEST(horizontal_only_matches_vertical_on_transposed_image) {
    cv::Mat img = random_image(20, 15, 8);
    cv::Mat a = remove_horizontal_seams(img, 6, baseline()).image;
    cv::Mat b = remove_vertical_seams(transpose(img), 6, baseline()).image;
    CHECK(same(a, transpose(b)));
}

TEST(every_removed_pixel_is_unique_and_counts_add_up) {
    const int h = 18, w = 22, th = 11, tw = 13;
    CarveResult res = carve(random_image(h, w, 9), tw, th, baseline());
    std::set<std::pair<int, int>> seen;
    size_t total = 0;
    for (const auto& seam : res.stats.seam_coords)
        for (const Pixel& p : seam) {
            CHECK(p.row >= 0 && p.row < h && p.col >= 0 && p.col < w);
            seen.insert({p.row, p.col});
            ++total;
        }
    CHECK(total == static_cast<size_t>(h * w - th * tw));
    CHECK(seen.size() == total);  // no pixel removed twice
}

TEST(carve_keeps_textured_object) {
    // Flat background = zero energy; a striped bar = high energy inside it.
    // Seams should remove background only, so all 3 bar columns survive.
    // (A SOLID bar would not work: its interior is flat, so it has zero energy too.)
    cv::Mat img(30, 60, CV_8UC3, cv::Scalar::all(30));
    for (int r = 0; r < 30; r += 2) img.row(r).colRange(28, 31).setTo(cv::Scalar::all(255));
    cv::Mat out = carve(img, 40, 30, baseline()).image;
    int bar_columns = 0;
    for (int c = 0; c < out.cols; ++c) {
        bool has_white = false;
        for (int r = 0; r < out.rows; ++r) has_white |= out.at<cv::Vec3b>(r, c) == cv::Vec3b(255, 255, 255);
        bar_columns += has_white;
    }
    CHECK(bar_columns == 3);
}

TEST(on_seam_callback_sees_image_before_removal) {
    std::vector<std::pair<cv::Size, Direction>> seen;
    CarveOptions opt;
    opt.on_seam = [&](const cv::Mat& image, const std::vector<int>& seam, Direction d) {
        validate_seam(seam, image.rows, image.cols);
        seen.push_back({image.size(), d});
    };
    carve(random_image(10, 12), 10, 9, baseline(), opt);
    CHECK(seen.size() == 3);
    CHECK(seen[0].first == cv::Size(12, 10) && seen[0].second == Direction::Vertical);
    CHECK(seen[1].first == cv::Size(11, 10) && seen[1].second == Direction::Vertical);
    CHECK(seen[2].first == cv::Size(10, 10) && seen[2].second == Direction::Horizontal);
}

TEST(stats_are_recorded) {
    CarveResult res = carve(random_image(40, 40), 30, 30, baseline());
    const CarveStats& s = res.stats;
    CHECK(s.energy_time > 0 && s.find_time > 0 && s.remove_time > 0);
    CHECK_NEAR(s.total_time(), s.energy_time + s.find_time + s.remove_time, 1e-12);
    double sum = 0;
    for (double c : s.seam_costs) sum += c;
    CHECK_NEAR(s.total_seam_cost, sum, 1e-6);
}

TEST(draw_removed_seams_colours_exactly_the_removed_pixels) {
    cv::Mat img(10, 12, CV_8UC3, cv::Scalar::all(0));
    for (int r = 0; r < 10; r += 2) img.row(r).setTo(cv::Scalar::all(50));
    CarveResult res = carve(img, 9, 8, baseline());
    cv::Mat drawn = draw_removed_seams(img, res.stats.seam_coords, res.stats.seam_directions);
    int red = 0, blue = 0;
    for (int r = 0; r < drawn.rows; ++r)
        for (int c = 0; c < drawn.cols; ++c) {
            red += drawn.at<cv::Vec3b>(r, c) == cv::Vec3b(0, 0, 255);
            blue += drawn.at<cv::Vec3b>(r, c) == cv::Vec3b(255, 0, 0);
        }
    CHECK(red == 3 * 10 && blue == 2 * 9);
}

// ---------------------------------------------------------------- seam finders

TEST(registry_has_baseline_and_rejects_unknown) {
    CHECK(has_method("baseline"));
    CHECK_THROWS(get_method("no-such-method"));
}

TEST_EACH_METHOD(method_returns_valid_seam) {
    const SeamFinder finder = finder_or_skip(method);
    const int sizes[][2] = {{1, 1}, {1, 5}, {6, 1}, {2, 2}, {20, 25}};
    unsigned seed = 0;
    for (auto [h, w] : sizes) {
        cv::Mat energy = compute_energy(random_image(h, w, seed++));
        validate_seam(finder(energy), h, w);
    }
}

TEST_EACH_METHOD(method_never_beats_the_true_minimum) {
    const SeamFinder finder = finder_or_skip(method);
    cv::Mat energy(6, 5, CV_64F);
    cv::RNG(1).fill(energy, cv::RNG::UNIFORM, 0.0, 1.0);
    CHECK(seam_cost(energy, finder(energy)) >= brute_force_min_cost(energy) - 1e-9);
}

TEST_EACH_METHOD(optimal_methods_find_minimum_seam) {
    // DP and Graph Cut must find the cheapest seam. Checked by brute force on tiny grids.
    if (method != "dp" && method != "graphcut") SKIP("only DP and Graph Cut are optimal");
    const SeamFinder finder = finder_or_skip(method);
    std::mt19937 rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        const int h = std::uniform_int_distribution<int>(2, 6)(rng);
        const int w = std::uniform_int_distribution<int>(2, 5)(rng);
        cv::Mat energy(h, w, CV_64F);
        cv::RNG(trial).fill(energy, cv::RNG::UNIFORM, 0.0, 1.0);
        CHECK_NEAR(seam_cost(energy, finder(energy)), brute_force_min_cost(energy), 1e-9);
    }
}

TEST(greedy_follows_local_minimum) {
    // Greedy starts at the cheapest top pixel and always steps to the cheapest pixel below.
    const SeamFinder finder = finder_or_skip("greedy");
    cv::Mat energy = energy_from_rows({{5, 1, 5, 5}, {5, 5, 2, 5}, {9, 9, 9, 3}, {5, 5, 0, 5}});
    CHECK((finder(energy) == std::vector<int>{1, 2, 3, 2}));
}

TEST(greedy_can_be_worse_than_optimal) {
    // Classic trap: the cheap start leads into an expensive row.
    const SeamFinder finder = finder_or_skip("greedy");
    cv::Mat energy = energy_from_rows({{0, 9, 9, 1}, {9, 9, 9, 1}, {100, 100, 9, 1}});
    CHECK(seam_cost(energy, finder(energy)) > brute_force_min_cost(energy));
}

// ---------------------------------------------------------------- command line

static fs::path temp_dir() {
    fs::path dir = fs::temp_directory_path() / "seamcarve_tests";
    fs::create_directories(dir);
    return dir;
}

static int cli(std::vector<std::string> args) {
    std::ostringstream out, err;
    return run_cli(args, out, err);
}

TEST(cli_end_to_end) {
    fs::path dir = temp_dir(), src = dir / "in.png", out = dir / "out.png";
    cv::imwrite(src.string(), random_image(20, 24));
    CHECK(cli({src.string(), "--method", "baseline", "--width", "18", "--height", "15", "--out", out.string(),
               "--save-energy", "--save-seams"}) == 0);
    cv::Mat result = cv::imread(out.string());
    CHECK(result.rows == 15 && result.cols == 18);
    CHECK(fs::exists(dir / "out_energy.png") && fs::exists(dir / "out_seams.png"));
}

TEST(cli_scale) {
    fs::path dir = temp_dir(), src = dir / "in2.png", out = dir / "out2.png";
    cv::imwrite(src.string(), random_image(20, 30));
    CHECK(cli({src.string(), "--method", "baseline", "--scale", "0.5", "--out", out.string()}) == 0);
    cv::Mat result = cv::imread(out.string());
    CHECK(result.rows == 10 && result.cols == 15);
}

TEST(cli_rejects_bad_input) {
    fs::path dir = temp_dir(), src = dir / "in3.png", out = dir / "o.png";
    cv::imwrite(src.string(), random_image(10, 10));
    const std::string s = src.string(), o = out.string();
    CHECK(cli({s, "--method", "baseline", "--out", o, "--width", "999"}) == 1);
    CHECK(cli({s, "--method", "baseline", "--out", o, "--scale", "1.5"}) == 1);
    CHECK(cli({s, "--method", "baseline", "--out", o, "--scale", "0"}) == 1);
    CHECK(cli({s, "--method", "baseline", "--out", o, "--width", "abc"}) == 1);
    CHECK(cli({s, "--method", "no-such-method", "--out", o}) == 1);
    CHECK(cli({s, "--bogus-option"}) == 1);
    CHECK(cli({(dir / "missing.png").string(), "--method", "baseline"}) == 1);
    CHECK(cli({}) == 1);
    CHECK(cli({"--help"}) == 0);
}

int main(int argc, char** argv) { return testing::run_all(argc, argv); }
