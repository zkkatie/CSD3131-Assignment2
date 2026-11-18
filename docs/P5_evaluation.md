# P5: Evaluation and Comparison

This section compares the seam finders (Dynamic Programming, Greedy, Graph Cut, and
the straight-column baseline) on **seam quality** and **speed**, and checks whether
the measured running times match the complexity analysis of each method.

> **Status:** the method and tooling are written. The Results tables are filled in
> from `./build/evaluate` once DP is merged. Rows marked *TBD* are not run yet.

## 1. Design

### 1.1 What is compared and why it is fair

All methods run inside the same `carve()` loop (P1), on the same images, with the
same energy function and the same timers. Only the `find_seam` call differs, so any
difference in cost or time comes from the method itself. `bench/evaluate.cpp` takes
the methods from the registry, so every method is measured by the same code.

### 1.2 Measuring quality: the per-step optimum

A seam's **cost** is the sum of the energies of its pixels (`seam_cost`). Comparing
the *total* cost of two carvings directly is misleading: after the first seam the
two images are different, so the methods are no longer solving the same problem.

Instead, at **every** step we also compute the cost of the cheapest seam that
existed at that moment (ground truth), using a cost-only dynamic programme:

    M[0][c] = e[0][c]
    M[r][c] = e[r][c] + min(M[r-1][c-1], M[r-1][c], M[r-1][c+1])
    optimum = min over c of M[H-1][c]

This runs inside the `on_seam` callback, which `carve()` calls outside its timers,
so it does not affect the timings. From it we report:

| Metric | Meaning |
|---|---|
| gap % | (sum of chosen seam costs / sum of per-step optimal costs - 1) x 100. 0 = always optimal |
| seams optimal % | share of steps where the chosen seam was a cheapest one |

The ground truth is independent of the DP and Graph Cut code, so it also checks
that those two methods really are optimal.

### 1.3 The four experiments

| | Experiment | Data | Measures |
|---|---|---|---|
| E1 | Quality | the 8 photos in `images/`, Sobel and gradient energy, 25% of the width removed | gap %, seams optimal %, output images |
| E2 | Single-seam optimality | 500 random maps per size (5x5 to 200x200), two kinds: *uniform* random values, and *photo* crops of real energy maps | % optimal, extra cost %, worst case % |
| E3 | Runtime scaling | synthetic images, square (n x n), width-only and height-only series | ms per seam, fitted log-log slope |
| E4 | Time breakdown | the photos, 25% of the width removed | share of time in energy / find / remove |

Every method in E2 sees exactly the same maps (fixed random seed 42).

### 1.4 Handling Graph Cut's running time

Edmonds-Karp is far slower than the other methods. To keep the run practical:
Graph Cut runs only on images up to 20,000 pixels (the `_small` photos, 120 px wide),
at most 100 trials per size in E2, and its E3 series stops once one seam takes over
2 seconds (`--budget`). Its quality results are therefore on the small photos only.

### 1.5 Timing method

Release build (`-O2`/`-O3`, the CMake default), best of 3 runs (1 for Graph Cut),
`std::chrono::steady_clock`. The machine is listed in Section 3.

## 2. Expected results (theory)

For an H x W energy map, one seam:

| Method | Optimal? | find_seam time | Expected log-log slope vs pixels (square images) |
|---|---|---|---|
| Baseline | no (straight seams only) | O(HW), summing every column | ~1 |
| Greedy | no | O(W) for row 0 + O(1) per row = O(H + W) | ~0.5 |
| DP | yes | O(HW) | ~1 |
| Graph Cut (Edmonds-Karp) | yes | O(V E^2) worst case, V ~ HW, E ~ 6HW | > 1 |

In the width-only series (H fixed), Greedy's O(H + W) still grows linearly in W, but
with a much smaller constant than DP.

For E2, Greedy should be optimal often on tiny maps and almost never on large ones,
because the chance that the locally best choice is also globally best falls with
every row. On *photo* maps it should do better than on *uniform* maps, because real
energy has smooth low-energy regions (sky, walls) where local and global choices agree.

## 3. Results

**Setup:** *TBD* (CPU, RAM, OS / WSL version, compiler, OpenCV version, all printed
at the start of `./build/evaluate`).

### 3.1 E1: Quality on real photos (Sobel)

| Image | DP gap % | Graph Cut gap % | Greedy gap % | Baseline gap % |
|---|---|---|---|---|
| broadway | TBD | n/a (too large) | TBD | TBD |
| broadway_small | TBD | TBD | TBD | TBD |
| ... | | | | |

Visual comparison: `results/eval_images/`.

### 3.2 E2: How often each method finds the cheapest seam

| Map size | Greedy, uniform | Greedy, photo | Baseline, photo | DP / Graph Cut |
|---|---|---|---|---|
| 5x5 | TBD | TBD | TBD | 100% (expected) |
| ... | | | | |


### 3.3 E3: Runtime scaling

| Method | Slope, square | Slope, width only | Slope, height only | ms per seam at 400x400 |
|---|---|---|---|---|
| DP | TBD | TBD | TBD | TBD |
| Greedy | TBD | TBD | TBD | TBD |
| Graph Cut | TBD | TBD | TBD | TBD |
| Baseline | TBD | TBD | TBD | TBD |


### 3.4 E4: Where the time goes

*TBD*. Expected: for Greedy, seam finding is a small share and seam removal
dominates; for DP the two are comparable; for Graph Cut, seam finding is almost all
of it.

## 4. Discussion

*TBD after the results. Points to cover:*

- Does each measured slope match its Big-O? Explain any difference (constant
  factors, cache effects, fixed overheads at small sizes).
- How large is Greedy's error in practice, and where it shows in the images
  (seams cutting through objects after a cheap start).
- DP vs Graph Cut: same answer (both optimal), very different cost. Why min-cut is
  still interesting (it generalises to other energies and to 2D regions).
- **Recommendation:** which method to use and when.

## 5. Validation

The evaluation checks itself while it runs:

- every seam is validated (`validate_seam`) before its cost is used;
- DP and Graph Cut must be optimal on every E2 map, otherwise a warning is printed;
- no method may ever cost less than the ground truth (it would mean the ground truth is wrong);
- the project tests (`./build/tests`) already check DP and Graph Cut against brute
  force, and that `full` and `incremental` energy modes give identical costs, so
  timings in either mode measure the same work.

## 6. Limitations

- Graph Cut is only evaluated on small images, so its quality numbers cover fewer
  pixels than the other methods. Its time is extrapolated from the fitted slope.
- The gap % measures cost (energy), not perceived quality. The output images in
  `results/eval_images/` are used for the visual side.
- Only vertical seams are measured in E1/E4. Horizontal seams use the same finder on
  the transposed image (P1), so the results carry over.
- Timings depend on the machine; the slopes and the ratios between methods are
  more meaningful than the absolute seconds.
