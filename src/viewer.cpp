// Interactive viewer (bonus items 7a and 7b).
//
//   ./build/seamcarve_viewer images/broadway.jpg
//   ./build/seamcarve_viewer images/broadway.jpg --width 600 --height 450
//   ./build/seamcarve_viewer images/broadway_small.jpg --method graphcut
//
//   Width / Height sliders   carve the image to that size                         (item 7b)
//   SPACE                    replay the carve one seam at a time; SPACE stops it  (item 7a)
//   P, then N                during a replay: pause, then show the next seam on each N
//   + / -                    replay faster / slower
//   S                        save the carved image and the seam picture to results/
//   Q or ESC                 quit
//
// Left panel:  the original image with every removed seam painted on it
//              (red = vertical seam, blue = horizontal seam).
// Right panel: the carved image. During a replay, the seam about to be removed is yellow.
//
// The viewer adds no algorithm of its own. A slider change calls carve() again, and
// the replay uses carve()'s on_seam callback, which runs just before each seam is removed.
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "seamcarve/seamcarve.hpp"

using namespace seamcarve;
namespace fs = std::filesystem;

namespace {

const char* const WINDOW = "Seam carving viewer";
const char* const USAGE =
    "usage: seamcarve_viewer <image> [--method dp|greedy|graphcut|baseline] [--width N] [--height N]";
const cv::Vec3b RED(0, 0, 255), BLUE(255, 0, 0), YELLOW(0, 255, 255);  // OpenCV colours are B, G, R
const int GAP = 10;                 // dark gap between the two panels, in image pixels
const int SLOW_MAX_PIXELS = 20000;  // Graph Cut needs about 0.5 s per seam at 200x200

struct StopReplay {};  // thrown from the callback to end a replay early

// The Qt window backend throws once the window has been closed with its X button.
bool window_open() {
    try {
        return cv::getWindowProperty(WINDOW, cv::WND_PROP_VISIBLE) >= 1;
    } catch (const cv::Exception&) {
        return false;
    }
}

// Draws both panels side by side, with one line of text underneath. `left` is always the
// size of the original image, so the window keeps its size while `right` shrinks.
void show(const cv::Mat& left, const cv::Mat& right, double zoom, const std::string& text) {
    cv::Mat canvas(left.rows, 2 * left.cols + GAP, CV_8UC3, cv::Scalar::all(40));
    left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(canvas(cv::Rect(left.cols + GAP, 0, right.cols, right.rows)));
    cv::resize(canvas, canvas, cv::Size(), zoom, zoom, zoom < 1 ? cv::INTER_AREA : cv::INTER_NEAREST);
    cv::copyMakeBorder(canvas, canvas, 0, 28, 0, 0, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::putText(canvas, text, cv::Point(8, canvas.rows - 9), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar::all(255), 1, cv::LINE_AA);
    cv::imshow(WINDOW, canvas);
}

// A whole number from the command line, e.g. the 600 in "--width 600".
int number(const std::string& text) {
    try {
        return std::stoi(text);
    } catch (const std::exception&) {
        throw std::invalid_argument("'" + text + "' is not a number\n" + USAGE);
    }
}

// + and - change how long each seam stays on screen during a replay.
void change_speed(int key, int& delay_ms) {
    if (key == '+' || key == '=') delay_ms = std::max(1, delay_ms / 2);
    if (key == '-') delay_ms = std::min(480, delay_ms * 2);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string input, method = "dp";
        int start_w = 0, start_h = 0;  // 0 = start at the full size
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--method" && i + 1 < argc) method = argv[++i];
            else if (arg == "--width" && i + 1 < argc) start_w = number(argv[++i]);
            else if (arg == "--height" && i + 1 < argc) start_h = number(argv[++i]);
            else if (!arg.empty() && arg[0] != '-' && input.empty()) input = arg;
            else throw std::invalid_argument(USAGE);
        }
        if (input.empty()) throw std::invalid_argument(USAGE);

        const SeamFinder finder = get_method(method);
        const cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
        if (img.empty()) throw std::runtime_error("could not read image '" + input + "'");
        const int W = img.cols, H = img.rows;
        if (method == "graphcut" && W * H > SLOW_MAX_PIXELS)
            throw std::runtime_error("graphcut is too slow for this image; use one of the _small images");
#if !defined(_WIN32) && !defined(__APPLE__)  // Linux: without a display, Qt would abort the program
        if (!std::getenv("DISPLAY") && !std::getenv("WAYLAND_DISPLAY"))
            throw std::runtime_error("no display found: the viewer needs a desktop to open its window");
#endif

        const int min_w = std::max(1, W / 2), min_h = std::max(1, H / 2);  // the sliders go down to half size
        const double zoom = std::min(600.0 / W, 600.0 / H);                // each panel fits in 600 x 600 on screen
        cv::namedWindow(WINDOW, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
        cv::moveWindow(WINDOW, 40, 40);  // top-left of the screen, so the whole window is visible
        cv::createTrackbar("Width", WINDOW, nullptr, W);
        cv::createTrackbar("Height", WINDOW, nullptr, H);
        cv::setTrackbarMin("Width", WINDOW, min_w);
        cv::setTrackbarMin("Height", WINDOW, min_h);
        cv::setTrackbarPos("Width", WINDOW, start_w ? std::clamp(start_w, min_w, W) : W);
        cv::setTrackbarPos("Height", WINDOW, start_h ? std::clamp(start_h, min_h, H) : H);

        int w = 0, h = 0;   // the size currently shown
        int delay_ms = 30;  // time per seam during a replay
        CarveResult result;
        cv::Mat all_seams;  // the original with every removed seam painted
        std::string size_text;
        const std::string keys = "   SPACE replay   +/- speed   S save   Q quit";
        const auto show_result = [&](const std::string& note) {
            show(all_seams, result.image, zoom,
                 size_text + "   " + std::to_string(result.stats.seams_removed) + " seams" + note);
        };

        for (;;) {
            const int new_w = cv::getTrackbarPos("Width", WINDOW), new_h = cv::getTrackbarPos("Height", WINDOW);
            if (new_w != w || new_h != h) {  // a slider moved: carve the original to the new size
                w = new_w;
                h = new_h;
                if (!all_seams.empty()) {  // say so first: a Graph Cut carve can take several seconds
                    show(all_seams, result.image, zoom,
                         method + "  carving to " + std::to_string(w) + "x" + std::to_string(h) + " ...");
                    cv::waitKey(1);
                    if (!window_open()) break;
                }
                result = carve(img, w, h, finder);
                all_seams = draw_removed_seams(img, result.stats.seam_coords, result.stats.seam_directions);
                size_text = method + "  " + std::to_string(W) + "x" + std::to_string(H) + " -> " +
                            std::to_string(w) + "x" + std::to_string(h);
                show_result(keys);
            }

            const int key = std::tolower(cv::waitKey(30) & 0xFF);  // lower case, so Caps Lock does not matter
            if (key == 'q' || key == 27 || !window_open()) break;
            change_speed(key, delay_ms);

            if (key == 's') {  // save what both panels show
                const std::string base = "results/" + fs::path(input).stem().string() + "_" + method + "_" +
                                         std::to_string(w) + "x" + std::to_string(h);
                fs::create_directories("results");
                if (!cv::imwrite(base + ".png", result.image) || !cv::imwrite(base + "_seams.png", all_seams))
                    throw std::runtime_error("could not write " + base + ".png");
                std::cout << "saved " << base << ".png and " << base << "_seams.png\n";
                show_result("   saved " + base + ".png");
            }

            if (key == ' ') {  // replay the same carve, pausing on every seam
                cv::Mat painted = img.clone();  // the original, with the seams removed so far
                const int total = result.stats.seams_removed;
                int n = 0;
                bool paused = false, quit = false;
                CarveOptions options;
                options.on_seam = [&](const cv::Mat& current, const std::vector<int>& seam, Direction direction) {
                    const bool vertical = direction == Direction::Vertical;
                    // carve() gives the same seams every time, so seam n of this replay is seam n
                    // of `result`, whose pixels are already known in original-image coordinates.
                    for (const Pixel& p : result.stats.seam_coords[n])
                        painted.at<cv::Vec3b>(p.row, p.col) = vertical ? RED : BLUE;
                    cv::Mat marked = draw_vertical_seam(current, seam, YELLOW);
                    if (!vertical) marked = transpose(marked);  // horizontal seams arrive transposed
                    ++n;
                    show(painted, marked, zoom,
                         size_text + "   seam " + std::to_string(n) + "/" + std::to_string(total) +
                             "   SPACE stop   P pause   N next seam   +/- speed");
                    for (;;) {  // wait before the seam is removed: for the delay, or for N while paused
                        const int k = std::tolower(cv::waitKey(paused ? 50 : delay_ms) & 0xFF);
                        change_speed(k, delay_ms);
                        if (k == ' ' || k == 'q' || k == 27 || !window_open()) {
                            quit = k != ' ' || !window_open();
                            throw StopReplay{};
                        }
                        if (k == 'p') paused = !paused;
                        if (!paused || k == 'n') break;
                    }
                };
                try {
                    carve(img, w, h, finder, options);
                } catch (const StopReplay&) {
                }
                if (quit) break;
                show_result(keys);  // back to the finished carve
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
