#include "seamcarve/energy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace seamcarve {

std::string to_string(EnergyFunction fn) {
    return fn == EnergyFunction::Sobel ? "sobel" : "gradient";
}

EnergyFunction parse_energy_function(const std::string& name) {
    if (name == "sobel") return EnergyFunction::Sobel;
    if (name == "gradient") return EnergyFunction::Gradient;
    throw std::invalid_argument("unknown energy function '" + name + "' (use sobel or gradient)");
}

cv::Mat to_gray(const cv::Mat& img) {
    if (img.empty()) throw std::invalid_argument("to_gray: empty image");
    cv::Mat gray8;
    switch (img.channels()) {
        case 1: gray8 = img; break;
        case 3: cv::cvtColor(img, gray8, cv::COLOR_BGR2GRAY); break;
        case 4: cv::cvtColor(img, gray8, cv::COLOR_BGRA2GRAY); break;
        default: throw std::invalid_argument("to_gray: expected 1, 3 or 4 channels");
    }
    cv::Mat gray;
    gray8.convertTo(gray, CV_64F);
    return gray;
}

cv::Mat energy_from_gray(const cv::Mat& gray, EnergyFunction fn) {
    CV_Assert(gray.type() == CV_64FC1);
    const int h = gray.rows, w = gray.cols;
    if (fn == EnergyFunction::Sobel) {
        // OpenCV is allowed for the Sobel filter. BORDER_REPLICATE = clamp at the edges.
        cv::Mat gx, gy;
        cv::Sobel(gray, gx, CV_64F, 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
        cv::Sobel(gray, gy, CV_64F, 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
        return cv::abs(gx) + cv::abs(gy);
    }
    cv::Mat e(h, w, CV_64F);
    for (int r = 0; r < h; ++r)
        for (int c = 0; c < w; ++c) e.at<double>(r, c) = energy_at(gray, r, c, fn);
    return e;
}

cv::Mat compute_energy(const cv::Mat& img, EnergyFunction fn) {
    return energy_from_gray(to_gray(img), fn);
}

double energy_at(const cv::Mat& g, int r, int c, EnergyFunction fn) {
    const int h = g.rows, w = g.cols;
    const int rm = std::max(r - 1, 0), rp = std::min(r + 1, h - 1);
    const int cm = std::max(c - 1, 0), cp = std::min(c + 1, w - 1);
    auto I = [&](int y, int x) { return g.at<double>(y, x); };
    double gx, gy;
    if (fn == EnergyFunction::Sobel) {
        gx = (I(rm, cp) + 2 * I(r, cp) + I(rp, cp)) - (I(rm, cm) + 2 * I(r, cm) + I(rp, cm));
        gy = (I(rp, cm) + 2 * I(rp, c) + I(rp, cp)) - (I(rm, cm) + 2 * I(rm, c) + I(rm, cp));
    } else {
        gx = I(r, cp) - I(r, cm);
        gy = I(rp, c) - I(rm, c);
    }
    return std::abs(gx) + std::abs(gy);
}

void update_energy(cv::Mat& energy, const cv::Mat& gray, const std::vector<int>& seam,
                   EnergyFunction fn) {
    const int h = gray.rows, w = gray.cols;
    if (w == 0) return;
    for (int r = 0; r < h; ++r) {
        const int up = seam[std::max(r - 1, 0)];
        const int down = seam[std::min(r + 1, h - 1)];
        const int m = std::min({up, seam[r], down});
        const int first = std::max(m - 1, 0), last = std::min(m + 2, w - 1);
        for (int c = first; c <= last; ++c) energy.at<double>(r, c) = energy_at(gray, r, c, fn);
    }
}

}  // namespace seamcarve
