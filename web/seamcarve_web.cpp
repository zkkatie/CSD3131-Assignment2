// Entry points for the web page (web/index.html).
//
// The page passes the RGBA pixels of a canvas. sc_carve() calls the same carve() as the
// command-line program, then the page reads back the carved image, every removed seam
// and the statistics. No algorithm lives here: this file only converts the data.
//
// Built to WebAssembly by web/build.sh.
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include <emscripten/emscripten.h>
#include <opencv2/imgproc.hpp>

#include "seamcarve/carve.hpp"

namespace {

std::vector<uint8_t> g_rgba;  // the carved image, RGBA
int g_width = 0, g_height = 0;
seamcarve::CarveStats g_stats;
std::vector<int> g_seams;  // for each seam: direction (0 vertical, 1 horizontal), n, then n pairs (row, col)
std::string g_error;

}  // namespace

extern "C" {

// Carves a width x height RGBA image to target_width x target_height.
// Returns 0 on success, 1 on error (the message is in sc_error()).
EMSCRIPTEN_KEEPALIVE int sc_carve(const uint8_t* rgba, int width, int height, int target_width,
                                  int target_height, const char* method, const char* energy) {
    try {
        const cv::Mat in(height, width, CV_8UC4, const_cast<uint8_t*>(rgba));
        cv::Mat bgr;  // the command-line program carves BGR images, so do the same here
        cv::cvtColor(in, bgr, cv::COLOR_RGBA2BGR);

        seamcarve::CarveOptions options;
        options.energy_fn = seamcarve::parse_energy_function(energy);
        const seamcarve::CarveResult res =
            seamcarve::carve(bgr, target_width, target_height, seamcarve::get_method(method), options);

        cv::Mat out;
        cv::cvtColor(res.image, out, cv::COLOR_BGR2RGBA);
        g_width = out.cols;
        g_height = out.rows;
        g_rgba.assign(out.datastart, out.dataend);
        g_stats = res.stats;
        g_seams.clear();
        for (std::size_t k = 0; k < g_stats.seam_coords.size(); ++k) {
            g_seams.push_back(g_stats.seam_directions[k] == seamcarve::Direction::Vertical ? 0 : 1);
            g_seams.push_back(static_cast<int>(g_stats.seam_coords[k].size()));
            for (const seamcarve::Pixel& p : g_stats.seam_coords[k]) {
                g_seams.push_back(p.row);
                g_seams.push_back(p.col);
            }
        }
        g_error.clear();
        return 0;
    } catch (const std::exception& e) {
        g_error = e.what();
        return 1;
    }
}

EMSCRIPTEN_KEEPALIVE const uint8_t* sc_result() { return g_rgba.data(); }
EMSCRIPTEN_KEEPALIVE int sc_result_width() { return g_width; }
EMSCRIPTEN_KEEPALIVE int sc_result_height() { return g_height; }
EMSCRIPTEN_KEEPALIVE int sc_seam_count() { return g_stats.seams_removed; }
EMSCRIPTEN_KEEPALIVE const int* sc_seams() { return g_seams.data(); }
EMSCRIPTEN_KEEPALIVE int sc_seams_length() { return static_cast<int>(g_seams.size()); }
EMSCRIPTEN_KEEPALIVE double sc_total_cost() { return g_stats.total_seam_cost; }
EMSCRIPTEN_KEEPALIVE double sc_find_time() { return g_stats.find_time; }
EMSCRIPTEN_KEEPALIVE double sc_total_time() { return g_stats.total_time(); }
EMSCRIPTEN_KEEPALIVE const char* sc_error() { return g_error.c_str(); }

}  // extern "C"
