# CSD3131 Assignment 2: Seam Carving

Content-aware image resizing using seam carving, implemented three ways:
Dynamic Programming, Greedy, and Graph Cut (min-cut/max-flow). Written in C++17
with OpenCV.

**Try it in a browser, nothing to install:** https://zkkatie.github.io/CSD3131-Assignment2/

## Setup (WSL / Ubuntu)

```bash
sudo apt update
sudo apt install -y g++ cmake libopencv-dev
```

## Build

From the repo folder:

```bash
cmake -S . -B build
cmake --build build -j
```

Rebuild with the second line after any change. If you add a new file in
`src/methods/`, run both lines again.

## Run

```bash
./build/seamcarve images/broadway.jpg --width 600 --method dp
./build/seamcarve images/windmill.jpg --scale 0.75 --save-seams --save-energy
./build/seamcarve images/sunset.jpg --width 600 --energy gradient --energy-mode full
./build/seamcarve --help
```

| Option | Meaning |
|---|---|
| `--method` | `dp` (default), `greedy`, `graphcut`, or `baseline` (straight columns, for testing). `graphcut` is very slow: use it on the `_small` images only |
| `--width`, `--height` | target size in pixels (shrink only) |
| `--scale` | shrink both sides by a factor in (0, 1], e.g. `0.8` |
| `--energy` | `sobel` (default) or `gradient` (simple central difference) |
| `--energy-mode` | `incremental` (default, faster) or `full` (recompute every time). Same result |
| `--save-energy` | also save the energy map of the input |
| `--save-seams` | also save the input with every removed seam drawn (red = vertical, blue = horizontal) |
| `--out` | output path (default `results/<name>_<method>.png`) |

## Web page (bonus: step-by-step view and interactive resizing)

**https://zkkatie.github.io/CSD3131-Assignment2/**

Nothing to install. The page runs this repository's C++ (`carve()` and the seam finders),
compiled to WebAssembly, in the visitor's browser. Nothing is precomputed.

Two panels. Left: the original with every removed seam painted (red = vertical,
blue = horizontal). Right: the carved image.

| Control | What it does |
|---|---|
| Photo, Upload | one of the test photos, or your own image (shrunk to at most 800 px on the long side) |
| Method, Energy | the seam finder and the energy function, as `--method` and `--energy` |
| Width / Height sliders | carve the image to that size (down to half the original) |
| Replay step by step, or Space | replay the carve one seam at a time; the seam about to go is yellow. Again to pause |
| ◀ ▶, or the arrow keys | one seam back or forward |
| Seam slider, Speed | jump to any seam; replay slower or faster |

Under the controls the page shows the number of seams, the total seam cost and the time.
`graphcut` is offered only for images of up to 20,000 pixels (the `_small` photos), where a
slider move can take several seconds.

A link can choose the example:
`?photo=broadway_small&method=graphcut&width=90` (also `height`, `energy=gradient`).

To build and open it yourself (needs `git`, `python3` and `pkg-config` as well):

```bash
web/build.sh                              # the first run downloads Emscripten and builds OpenCV for it
python3 -m http.server -d build/site      # then open http://localhost:8000
```

`web/build.sh` also checks the result: the native build and the WebAssembly build carve
the same photos with each method, and the script fails unless the output is identical,
pixel for pixel. The `pages` workflow runs it on every pull request and publishes the
page on every push to `main`.

## Test

```bash
./build/tests            # all tests
./build/tests carve      # only tests with "carve" in the name
```

Every method in `src/methods/` is tested automatically. DP and Graph Cut are
checked against a brute-force search to make
sure they find the cheapest seam. The tests also run on GitHub on every push.

## Benchmark

```bash
./build/benchmark_energy                      # synthetic images, 4 sizes
./build/benchmark_energy images/broadway.jpg  # your own image
```

Compares the full energy recompute with the incremental update.

```bash
./build/benchmark_dp                          # DP variants: time, memory, threads (5 sizes)
./build/benchmark_dp images/broadway.jpg      # also the time split of a real resize
```

Compares the DP variants (full table, top-down, two rows, linear space) and the
multithreaded DP.

## Evaluation (comparing the methods)

```bash
./build/evaluate --quick          # small run (~1 min) to check everything works
./build/evaluate                  # full run: quality, optimality, scaling, breakdown
./build/evaluate scaling          # only one experiment
```

Every registered method is compared, so new methods are included automatically.
Results go to `results/eval_*.csv`, output images to
`results/eval_images/`. See the comment at the top of `bench/evaluate.cpp` for what
each experiment measures. Use the default Release build for timings.

## Project layout

```
CMakeLists.txt                build setup
include/seamcarve/
  energy.hpp                  energy map (Sobel / gradient), incremental update   Katie
  seam.hpp                    seam validation, removal, drawing                   Katie
  carve.hpp                   main loop, horizontal seams, seam tracking          Katie
  methods.hpp                 seam finder interface + registry                    Katie
  cli.hpp, seamcarve.hpp      command-line tool, all-in-one header                Katie
  dp.hpp                      DP variants (used by tests and benchmark)           Ananth
src/
  energy.cpp, seam.cpp, carve.cpp, cli.cpp, main.cpp                              Katie
  methods/
    registry.cpp              method registry                                     Katie
    baseline.cpp              straight-column finder, for testing only            Katie
    dp.cpp                    Dynamic Programming (+ low-memory / threaded)       Ananth
    greedy.cpp                Greedy                                              Joel
    graphcut.cpp              Graph Cut (min-cut/max-flow)                        Shri
tests/
  testing.hpp                 tiny test framework (no downloads needed)
  test_core.cpp               tests (every tests/test_*.cpp is built into ./build/tests)
  test_dp.cpp                 DP tests                                            Ananth
web/
  index.html, worker.js       the web page (step-by-step view, sliders)           Adi
  seamcarve_web.cpp           entry points the page calls; converts pixels, calls carve()
  build.sh                    builds the page (C++ -> WebAssembly) and checks it
  check/                      the check: native output against WebAssembly output
bench/benchmark_energy.cpp    energy update benchmark                             Katie
bench/evaluate.cpp            method comparison (quality, speed, scaling)         Asri
bench/benchmark_dp.cpp        DP variants: time, memory, threads                  Ananth
docs/Assignment 2 Report.docx the team report                                     all
images/                       test photos (_small = 120 px wide, for Graph Cut)
results/                      output (not committed)
.github/workflows/tests.yml   CI: builds and runs tests on every push
.github/workflows/pages.yml   CD: builds, checks and publishes the web page
```

## The shared interface (everyone must follow this)

Each method is one file in `src/methods/`, named `dp.cpp`, `greedy.cpp` or
`graphcut.cpp`:

```cpp
#include "seamcarve/methods.hpp"

namespace {

std::vector<int> find_seam(const cv::Mat& energy) {
    // energy: H x W, CV_64F. Read a value with energy.at<double>(row, col).
    std::vector<int> seam(energy.rows);
    // ... fill seam[row] = column to remove in that row ...
    return seam;
}

}  // namespace

REGISTER_SEAM_FINDER("dp", find_seam);
```

- **Input:** `energy`, an H x W matrix of doubles (`CV_64F`). Don't modify it.
- **Output:** a `std::vector<int>` of length H, where `seam[i]` is the **column** removed in row `i`.
- The seam must be connected: `abs(seam[i] - seam[i-1]) <= 1`.
- Only handle **vertical** seams. `carve()` does horizontal seams by transposing the image.
- Must work for any size, including 1 x 1, 1 x W and H x 1.

Once the file exists, run `cmake -S . -B build` again. The method appears in
`--method` and its tests switch on. Nothing else needs changing.

`carve()` checks every seam with `validate_seam`, so a broken finder fails with a
clear error instead of producing a strange image.

## Useful hooks

- **Asri (evaluation):** `carve()` returns a `CarveResult` with `.image` and `.stats`.
  `stats` has `find_time`, `energy_time`, `remove_time`, `total_time()`,
  `total_seam_cost` and `seam_costs` (one per seam). Compare methods using
  `find_time` and `total_seam_cost`.
- **Adi (visualisation):**
  - `stats.seam_coords` holds every removed seam in **original image coordinates**
    (a `std::vector<Pixel>` each, with `.row` and `.col`), and
    `stats.seam_directions` says whether each one was `Vertical` or `Horizontal`.
    `draw_removed_seams(original, stats.seam_coords, stats.seam_directions)` draws them all.
  - For step-by-step animation, set `CarveOptions::on_seam`. It is called as
    `on_seam(current_image, seam, direction)` before each seam is removed. For
    horizontal seams `current_image` is transposed. Use `draw_vertical_seam` to paint it.

## Credits

Broadway Tower photo by Newton2, Wikimedia Commons, CC BY-SA. Other photos by the team.

## Team

| Person | Part |
|---|---|
| Katie | Core framework, energy map, seam removal, CI |
| Ananth | Dynamic Programming |
| Joel | Greedy + theory questions |
| Shri | Graph Cut |
| Asri | Evaluation and comparison |
| Adi | Visualisation, UI, report and video |
