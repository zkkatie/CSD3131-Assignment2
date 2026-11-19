// Native half of the browser-build check (see web/build.sh).
//
//   native_carve <image> <target_width> <target_height> <method> <pixels_in> <pixels_out>
//
// Reads the image with OpenCV, writes its raw BGR pixels to <pixels_in>, carves it with the
// native build and writes the raw BGR result to <pixels_out>. web/check/compare.js then gives
// the same input pixels to the WebAssembly build and compares the two results byte for byte.
#include <cstdio>
#include <fstream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "seamcarve/carve.hpp"

// File format: width and height as two 32-bit ints, then the pixels row by row, 3 bytes each.
static void write_pixels(const cv::Mat& m, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    const int size[2] = {m.cols, m.rows};
    f.write(reinterpret_cast<const char*>(size), sizeof size);
    for (int r = 0; r < m.rows; ++r) f.write(reinterpret_cast<const char*>(m.ptr(r)), m.cols * 3);
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::fprintf(stderr, "usage: native_carve <image> <width> <height> <method> <pixels_in> <pixels_out>\n");
        return 2;
    }
    const cv::Mat img = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    write_pixels(img, argv[5]);
    const seamcarve::CarveResult res =
        seamcarve::carve(img, std::stoi(argv[2]), std::stoi(argv[3]), seamcarve::get_method(argv[4]));
    write_pixels(res.image, argv[6]);
    std::printf("native   %-8s %dx%d -> %dx%d  seams=%d  cost=%.0f  time=%.3fs\n", argv[4], img.cols, img.rows,
                res.image.cols, res.image.rows, res.stats.seams_removed, res.stats.total_seam_cost,
                res.stats.total_time());
    return 0;
}
