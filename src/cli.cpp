// Command-line tool. Kept separate from main.cpp so the tests can call run_cli() directly.
#include "seamcarve/cli.hpp"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

#include "seamcarve/carve.hpp"

namespace fs = std::filesystem;

namespace seamcarve {

namespace {

std::string default_method() { return has_method("dp") ? "dp" : "baseline"; }

void print_usage(std::ostream& os) {
    std::string names;
    for (const auto& [name, f] : methods()) names += (names.empty() ? "" : ", ") + name;
    os << "Usage: seamcarve <input image> [options]\n"
          "\n"
          "Shrinks an image with seam carving.\n"
          "\n"
          "Options:\n"
          "  --method NAME         seam finder: "
       << names << " (default: " << default_method()
       << ")\n"
          "  --width N             target width in pixels (default: unchanged)\n"
          "  --height N            target height in pixels (default: unchanged)\n"
          "  --scale F             shrink both sides by F in (0, 1], e.g. 0.8 (overrides width/height)\n"
          "  --energy NAME         sobel (default) or gradient\n"
          "  --energy-mode NAME    incremental (default, faster) or full; same result\n"
          "  --out PATH            output path (default: results/<name>_<method>.png)\n"
          "  --save-energy         also save the energy map of the input\n"
          "  --save-seams          also save the input with every removed seam drawn\n"
          "                        (red = vertical, blue = horizontal)\n"
          "  -h, --help            show this help\n";
}

int parse_int(const std::string& flag, const std::string& value) {
    size_t used = 0;
    int v = 0;
    try {
        v = std::stoi(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size()) throw std::invalid_argument(flag + " expects a whole number, got '" + value + "'");
    return v;
}

double parse_double(const std::string& flag, const std::string& value) {
    size_t used = 0;
    double v = 0;
    try {
        v = std::stod(value, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != value.size()) throw std::invalid_argument(flag + " expects a number, got '" + value + "'");
    return v;
}

cv::Mat energy_to_image(const cv::Mat& energy) {
    double peak = 0;
    cv::minMaxLoc(energy, nullptr, &peak);
    cv::Mat out;
    energy.convertTo(out, CV_8U, peak > 0 ? 255.0 / peak : 0.0);
    return out;
}

}  // namespace

int run_cli(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    std::string input, method = default_method(), out_path, energy_name = "sobel", mode_name = "incremental";
    std::optional<int> width, height;
    std::optional<double> scale;
    bool save_energy = false, save_seams = false;

    try {
        for (size_t i = 0; i < args.size(); ++i) {
            const std::string& a = args[i];
            auto value = [&]() -> const std::string& {
                if (i + 1 >= args.size()) throw std::invalid_argument(a + " needs a value");
                return args[++i];
            };
            if (a == "-h" || a == "--help") {
                print_usage(out);
                return 0;
            } else if (a == "--method") method = value();
            else if (a == "--width") width = parse_int(a, value());
            else if (a == "--height") height = parse_int(a, value());
            else if (a == "--scale") scale = parse_double(a, value());
            else if (a == "--energy") energy_name = value();
            else if (a == "--energy-mode") mode_name = value();
            else if (a == "--out") out_path = value();
            else if (a == "--save-energy") save_energy = true;
            else if (a == "--save-seams") save_seams = true;
            else if (!a.empty() && a[0] == '-') throw std::invalid_argument("unknown option " + a);
            else if (input.empty()) input = a;
            else throw std::invalid_argument("only one input image is allowed");
        }
        if (input.empty()) {
            print_usage(err);
            return 1;
        }

        CarveOptions opt;
        opt.energy_fn = parse_energy_function(energy_name);
        opt.energy_mode = parse_energy_mode(mode_name);
        const SeamFinder finder = get_method(method);

        if (!fs::is_regular_file(input)) throw std::runtime_error("could not find image '" + input + "'");
        cv::Mat img = cv::imread(input, cv::IMREAD_COLOR);
        if (img.empty()) throw std::runtime_error("could not read image '" + input + "'");
        const int w = img.cols, h = img.rows;

        int tw = width.value_or(w), th = height.value_or(h);
        if (scale) {
            if (!(*scale > 0 && *scale <= 1))
                throw std::invalid_argument("--scale must be in (0, 1]; this program only shrinks images");
            tw = std::max(1, static_cast<int>(std::lround(w * *scale)));
            th = std::max(1, static_cast<int>(std::lround(h * *scale)));
        }

        if (out_path.empty())
            out_path = (fs::path("results") / (fs::path(input).stem().string() + "_" + method + ".png")).string();
        const fs::path out_file(out_path);
        if (out_file.has_parent_path()) fs::create_directories(out_file.parent_path());
        const std::string base = (out_file.parent_path() / out_file.stem()).string();

        out << method << " (" << energy_name << ", " << mode_name << "): " << w << "x" << h << " -> " << tw << "x"
            << th << "\n";
        CarveResult res = carve(img, tw, th, finder, opt);
        const CarveStats& s = res.stats;

        if (!cv::imwrite(out_path, res.image)) throw std::runtime_error("could not write '" + out_path + "'");
        out << "saved            " << out_path << "\n";
        if (save_energy) {
            cv::imwrite(base + "_energy.png", energy_to_image(compute_energy(img, opt.energy_fn)));
            out << "saved            " << base << "_energy.png\n";
        }
        if (save_seams) {
            cv::imwrite(base + "_seams.png", draw_removed_seams(img, s.seam_coords, s.seam_directions));
            out << "saved            " << base << "_seams.png\n";
        }

        out << std::fixed << std::setprecision(3);
        out << "seams removed    " << s.seams_removed << "\n";
        out << "total seam cost  " << std::setprecision(1) << s.total_seam_cost << std::setprecision(3) << "\n";
        out << "energy time      " << s.energy_time << "s\n";
        out << "seam find time   " << s.find_time << "s\n";
        out << "removal time     " << s.remove_time << "s\n";
        out << "total time       " << s.total_time() << "s\n";
        return 0;
    } catch (const std::exception& e) {
        err << "error: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace seamcarve
