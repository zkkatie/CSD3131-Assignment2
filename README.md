# CSD3131 Assignment 2: Seam Carving

Content-aware image resizing using seam carving, implemented three ways:
Dynamic Programming, Greedy, and Graph Cut (min-cut/max-flow).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py images/your_image.jpg --width 300 --method dp
python main.py images/your_image.jpg --scale 0.8 --method greedy
python main.py images/your_image.jpg --width 300 --height 200 --save-seams --save-energy
python main.py images/your_image.jpg --width 300 --energy gradient --energy-mode full
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
python -m pytest -v
```

Tests for a method are skipped until that method's file is added, then run
automatically. DP and Graph Cut are checked against a brute-force search to make
sure they find the cheapest seam. The tests also run on GitHub on every push.

## Benchmark

```bash
python scripts/benchmark_energy.py
```

Compares full vs incremental energy updates (see `docs/P1_framework.md`).

## Project layout

```
main.py                       command-line tool
seamcarve/
  energy.py                   energy map (Sobel / gradient), incremental update   P1
  seam.py                     seam validation, removal, drawing                   P1
  carve.py                    main loop, horizontal seams, seam tracking          P1
  methods/
    baseline.py               straight-column finder, for testing only            P1
    dp.py        (to add)     Dynamic Programming                                 P2
    greedy.py    (to add)     Greedy                                              P3
    graphcut.py  (to add)     Graph Cut (min-cut/max-flow)                        P4
tests/test_core.py            tests
scripts/benchmark_energy.py   energy update benchmark                             P1
docs/P1_framework.md          design, complexity, testing (report section)        P1
images/                       test input images
results/                      output (not committed)
.github/workflows/tests.yml   CI: runs tests on every push
```

## The shared interface (everyone must follow this)

Every seam finder is one function named `find_seam` in its own file under
`seamcarve/methods/`, named `dp.py`, `greedy.py` or `graphcut.py`. Once the file
exists it is picked up automatically by `main.py` and the tests. Nothing else needs changing.

```python
def find_seam(energy: np.ndarray) -> np.ndarray:
    ...
```

- **Input:** `energy`, an H x W float array. It is a copy, so you may modify it.
- **Output:** an int array (or list) `seam` of length H, where `seam[i]` is the **column** removed in row `i`.
- The seam must be connected: `abs(seam[i] - seam[i-1]) <= 1`.
- Only handle **vertical** seams. `carve.py` does horizontal seams by transposing the image.
- Must work for any size, including 1 x 1, 1 x W and H x 1.

`carve.py` checks every seam with `validate_seam`, so a broken finder fails with a
clear error instead of producing a strange image.

## Useful hooks

- **P5 (evaluation):** `carve()` returns `(image, stats)`. `stats` has
  `find_time`, `energy_time`, `remove_time`, `total_time`, `total_seam_cost`
  and a `seam_costs` list (one per seam). Compare methods using `find_time`
  and `total_seam_cost`.
- **P6 (visualisation):**
  - `stats.seam_coords` holds every removed seam in **original image coordinates**
    (an `(n, 2)` array of `(row, col)` each), and `stats.seam_directions` says
    whether each one was `"vertical"` or `"horizontal"`.
    `draw_removed_seams(original, stats.seam_coords, stats.seam_directions)` draws them all.
  - For step-by-step animation, pass `on_seam=callback` to `carve()`. It is called as
    `callback(current_image, seam, direction)` before each seam is removed. For
    horizontal seams `current_image` is transposed. Use `draw_vertical_seam` to paint it.

## Team

| Person | Part |
|---|---|
| P1 | Core framework, energy map, seam removal, CI |
| P2 | Dynamic Programming |
| P3 | Greedy + theory questions |
| P4 | Graph Cut |
| P5 | Evaluation and comparison |
| P6 | Visualisation, UI, report and video |
