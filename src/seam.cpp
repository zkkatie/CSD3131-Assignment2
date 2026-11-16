#include "seamcarve/seam.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

namespace seamcarve {

void validate_seam(const std::vector<int>& seam, int height, int width) {
    if (static_cast<int>(seam.size()) != height)
        throw std::invalid_argument("seam must have length " + std::to_string(height) + ", got " +
                                    std::to_string(seam.size()));
    for (int i = 0; i < height; ++i) {
        if (seam[i] < 0 || seam[i] >= width)
            throw std::invalid_argument("seam column " + std::to_string(seam[i]) + " in row " +
                                        std::to_string(i) + " is out of range [0, " +
                                        std::to_string(width - 1) + "]");
        if (i > 0 && std::abs(seam[i] - seam[i - 1]) > 1)
            throw std::invalid_argument("seam is not connected between rows " + std::to_string(i - 1) +
                                        " and " + std::to_string(i));
    }
}

double seam_cost(const cv::Mat& energy, const std::vector<int>& seam) {
    double total = 0;
    for (int r = 0; r < energy.rows; ++r) total += energy.at<double>(r, seam[r]);
    return total;
}

cv::Mat remove_vertical_seam(const cv::Mat& img, const std::vector<int>& seam) {
    const int h = img.rows, w = img.cols;
    const size_t px = img.elemSize();  // bytes per pixel (all channels)
    cv::Mat out(h, w - 1, img.type());
    for (int r = 0; r < h; ++r) {
        const uchar* src = img.ptr<uchar>(r);
        uchar* dst = out.ptr<uchar>(r);
        const size_t left = static_cast<size_t>(seam[r]) * px;           // bytes before the seam pixel
        const size_t right = static_cast<size_t>(w - 1 - seam[r]) * px;  // bytes after it
        std::memcpy(dst, src, left);
        std::memcpy(dst + left, src + left + px, right);
    }
    return out;
}

static cv::Mat as_bgr(const cv::Mat& img) {
    cv::Mat out;
    if (img.channels() == 1) cv::cvtColor(img, out, cv::COLOR_GRAY2BGR);
    else if (img.channels() == 4) cv::cvtColor(img, out, cv::COLOR_BGRA2BGR);
    else out = img.clone();
    if (out.depth() != CV_8U) out.convertTo(out, CV_8U);
    return out;
}

cv::Mat draw_vertical_seam(const cv::Mat& img, const std::vector<int>& seam, const cv::Vec3b& color) {
    cv::Mat out = as_bgr(img);
    for (int r = 0; r < out.rows; ++r) out.at<cv::Vec3b>(r, seam[r]) = color;
    return out;
}

cv::Mat draw_removed_seams(const cv::Mat& original, const std::vector<std::vector<Pixel>>& seam_coords,
                           const std::vector<Direction>& seam_directions, const cv::Vec3b& vertical_color,
                           const cv::Vec3b& horizontal_color) {
    cv::Mat out = as_bgr(original);
    for (size_t i = 0; i < seam_coords.size(); ++i) {
        const cv::Vec3b color = seam_directions[i] == Direction::Vertical ? vertical_color : horizontal_color;
        for (const Pixel& p : seam_coords[i]) out.at<cv::Vec3b>(p.row, p.col) = color;
    }
    return out;
}

}  // namespace seamcarve
