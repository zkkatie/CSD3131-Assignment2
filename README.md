# CSD3131 Assignment 2: Seam Carving

Content-aware image resizing using seam carving, implemented three ways:
Dynamic Programming, Greedy, and Graph Cut (min-cut/max-flow). Written in C++17
with OpenCV.

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
| `--method` | `dp`, `greedy`, `graphcut` (once added), or `baseline` (straight columns, for testing). Default: `dp` if added, else `baseline` |
| `--width`, `--height` | target size in pixels (shrink only) |
| `--scale` | shrink both sides by a factor in (0, 1], e.g. `0.8` |
| `--energy` | `sobel` (default) or `gradient` (simple central difference) |
| `--energy-mode` | `incremental` (default, faster) or `full` (recompute every time). Same result |
| `--save-energy` | also save the energy map of the input |
| `--save-seams` | also save the input with every removed seam drawn (red = vertical, blue = horizontal) |
| `--out` | output path (default `results/<name>_<method>.png`) |

## Test

```bash
./build/tests            # all tests
./build/tests carve      # only tests with "carve" in the name
```

Tests for a method are skipped until that method's file is added, then run
automatically. DP and Graph Cut are checked against a brute-force search to make
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
    graphcut.cpp  (to add)    Graph Cut (min-cut/max-flow)                        Shri
tests/
  testing.hpp                 tiny test framework (no downloads needed)
  test_core.cpp               tests (every tests/test_*.cpp is built into ./build/tests)
  test_dp.cpp                 DP tests                                            Ananth
bench/benchmark_energy.cpp    energy update benchmark                             Katie
bench/evaluate.cpp            method comparison (quality, speed, scaling)         Asri
bench/benchmark_dp.cpp        DP variants: time, memory, threads                  Ananth
docs/Assignment 2 Report.docx DP report section                                   Ananth
images/                       test photos (_small = 120 px wide, for Graph Cut)
results/                      output (not committed)
.github/workflows/tests.yml   CI: builds and runs tests on every push
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
