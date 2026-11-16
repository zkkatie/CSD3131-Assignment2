#include "seamcarve/carve.hpp"

#include <chrono>
#include <stdexcept>
#include <string>

namespace seamcarve {

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

// The arrays that are carved together.
struct State {
    cv::Mat img;        // the image itself (any type)
    cv::Mat gray;       // CV_64F greyscale, kept so energy can be updated without cvtColor
    cv::Mat energy;     // CV_64F energy map
    cv::Mat orig_rows;  // CV_32S: row of each pixel in the original image
    cv::Mat orig_cols;  // CV_32S: column of each pixel in the original image

    void transpose_all() {
        img = transpose(img);
        gray = transpose(gray);
        energy = transpose(energy);
        orig_rows = transpose(orig_rows);
        orig_cols = transpose(orig_cols);
    }
};

void remove_seams(State& st, int n, const SeamFinder& find_seam, CarveStats& stats, Direction direction,
                  const CarveOptions& opt) {
    for (int k = 0; k < n; ++k) {
        const int h = st.img.rows;

        auto t0 = Clock::now();
        std::vector<int> seam = find_seam(st.energy);  // const&: the finder cannot change our map
        stats.find_time += seconds_since(t0);
        validate_seam(seam, st.energy.rows, st.energy.cols);  // catches bugs in a finder immediately

        const double cost = seam_cost(st.energy, seam);
        stats.seam_costs.push_back(cost);
        stats.total_seam_cost += cost;
        std::vector<Pixel> coords(h);
        for (int r = 0; r < h; ++r)
            coords[r] = {st.orig_rows.at<int>(r, seam[r]), st.orig_cols.at<int>(r, seam[r])};
        stats.seam_coords.push_back(std::move(coords));
        stats.seam_directions.push_back(direction);
        if (opt.on_seam) opt.on_seam(st.img, seam, direction);

        auto t1 = Clock::now();
        st.img = remove_vertical_seam(st.img, seam);
        st.gray = remove_vertical_seam(st.gray, seam);
        st.orig_rows = remove_vertical_seam(st.orig_rows, seam);
        st.orig_cols = remove_vertical_seam(st.orig_cols, seam);
        stats.remove_time += seconds_since(t1);

        auto t2 = Clock::now();
        if (opt.energy_mode == EnergyMode::Incremental) {
            st.energy = remove_vertical_seam(st.energy, seam);
            update_energy(st.energy, st.gray, seam, opt.energy_fn);
        } else {
            st.energy = energy_from_gray(st.gray, opt.energy_fn);
        }
        stats.energy_time += seconds_since(t2);
        ++stats.seams_removed;
    }
}

}  // namespace

std::string to_string(EnergyMode mode) { return mode == EnergyMode::Incremental ? "incremental" : "full"; }

EnergyMode parse_energy_mode(const std::string& name) {
    if (name == "incremental") return EnergyMode::Incremental;
    if (name == "full") return EnergyMode::Full;
    throw std::invalid_argument("unknown energy mode '" + name + "' (use incremental or full)");
}

cv::Mat transpose(const cv::Mat& img) {
    cv::Mat out;
    cv::transpose(img, out);
    return out;
}

CarveResult carve(const cv::Mat& img, int target_width, int target_height, const SeamFinder& find_seam,
                  const CarveOptions& opt) {
    if (img.empty()) throw std::invalid_argument("carve: empty image");
    const int h = img.rows, w = img.cols;
    if (target_width < 1 || target_width > w || target_height < 1 || target_height > h)
        throw std::invalid_argument("target size " + std::to_string(target_width) + "x" +
                                    std::to_string(target_height) + " must be between 1x1 and " +
                                    std::to_string(w) + "x" + std::to_string(h) +
                                    " (this program only shrinks images)");

    CarveResult result;
    CarveStats& stats = result.stats;

    auto t0 = Clock::now();
    State st;
    st.img = img.clone();
    st.gray = to_gray(img);
    st.energy = energy_from_gray(st.gray, opt.energy_fn);  // the first full energy map
    stats.energy_time += seconds_since(t0);
    st.orig_rows.create(h, w, CV_32S);
    st.orig_cols.create(h, w, CV_32S);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) {
            st.orig_rows.at<int>(r, c) = r;
            st.orig_cols.at<int>(r, c) = c;
        }

    remove_seams(st, w - target_width, find_seam, stats, Direction::Vertical, opt);
    if (target_height < h) {
        st.transpose_all();
        remove_seams(st, h - target_height, find_seam, stats, Direction::Horizontal, opt);
        st.transpose_all();
    }
    result.image = st.img;
    return result;
}

CarveResult remove_vertical_seams(const cv::Mat& img, int n, const SeamFinder& find_seam,
                                  const CarveOptions& opt) {
    return carve(img, img.cols - n, img.rows, find_seam, opt);
}

CarveResult remove_horizontal_seams(const cv::Mat& img, int n, const SeamFinder& find_seam,
                                    const CarveOptions& opt) {
    return carve(img, img.cols, img.rows - n, find_seam, opt);
}

}  // namespace seamcarve
