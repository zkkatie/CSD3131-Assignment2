# P5: Evaluation and Comparison

This section compares the seam finders (Dynamic Programming, Greedy, Graph Cut, and
the straight-column baseline) on **seam quality** and **speed**, and checks whether
the measured running times match the complexity analysis of each method.

> **Status:** Baseline, Greedy and Graph Cut results are from the full run on
> 18 Nov 2026 (machine listed in Section 3). DP results are from a second full run of
> `./build/evaluate` with `src/methods/dp.cpp` included, on a different machine (also
> listed in Section 3). The quality and optimality numbers (E1, E2) do not depend on
> the machine: Baseline and Greedy reproduced exactly in the second run. For the
> timing results (E3, E4) the second run's Baseline figures are given next to DP's,
> so the methods can be compared on the same hardware.

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

**Setup:** AMD Ryzen AI 7 350 (16 hardware threads, all code single-threaded),
15 GB RAM, WSL2 Ubuntu 20.04 (kernel 6.6), g++ 13.1, OpenCV 4.2, Release build.
Full run: 421 s. Raw output: `results/eval_*.csv` and `results/eval_full_log.txt`.

**DP run:** AMD Ryzen 9 8945HS (16 hardware threads, all code single-threaded),
15 GB RAM, WSL2 Ubuntu 24.04 (kernel 6.18), g++ 13.3, OpenCV 4.6, Release build.
Full run with all four methods: 442 s.

### 3.1 E1: Quality on real photos

Gap % = extra seam cost over the cheapest seam available at each step (0 = optimal).
Full-size photos: 200 seams removed; `_small` photos (120 px wide): 30 seams.

| Image | Graph Cut | Greedy | Baseline | DP |
|---|---|---|---|---|
| broadway | too slow | 350.8 | 107.3 | **0.0** |
| building | too slow | 665.6 | 156.9 | **0.0** |
| sunset | too slow | 96.2 | 105.1 | **0.0** |
| windmill | too slow | 103.3 | 137.2 | **0.0** |
| broadway_small | **0.0** | 153.9 | 45.3 | **0.0** |
| building_small | **0.0** | 121.3 | 97.1 | **0.0** |
| sunset_small | **0.0** | 24.1 | 32.6 | **0.0** |
| windmill_small | **0.0** | 38.7 | 58.6 | **0.0** |

(Sobel energy. The gradient energy gives the same picture: DP and Graph Cut 0.0
everywhere, Greedy 22–676%, Baseline 42–159%.)

- DP chose a cheapest seam at **every one** of its 1,840 steps (all 8 photos, both
  energies), including the full-size photos where Graph Cut is too slow to run.
- On the `_small` photos DP and Graph Cut removed seams with **identical** total cost
  for both energies, even though they compute the seam in completely different ways.
- Graph Cut chose a cheapest seam at **every one** of its 240 steps (both energies).
- Greedy and Baseline chose a cheapest seam at **0%** of steps on every photo.
- Greedy is worst on **building** (666%) and **broadway** (351%): images with strong
  vertical structure, where a cheap start leads the seam into a wall of edges it
  cannot escape. On smoother images (**sunset**, **windmill**) Greedy beats Baseline.

Visual comparison: `results/eval_images/<image>_<method>.png` and `_seams.png`.

### 3.2 E2: How often each method finds the cheapest seam

500 maps per size (100 for Graph Cut, which skips 200x200). % of maps where the
seam was optimal, and total extra cost over all maps:

| Map size | Greedy uniform | Greedy photo | Baseline uniform | Baseline photo | Graph Cut | DP |
|---|---|---|---|---|---|---|
| 5x5 | 51.8% (+15%) | 62.2% (+10%) | 1.6% (+60%) | 33.8% (+13%) | 100% (+0%) | 100% (+0%) |
| 10x10 | 14.6% (+29%) | 20.2% (+19%) | 0.0% (+88%) | 13.2% (+26%) | 100% (+0%) | 100% (+0%) |
| 25x25 | 1.2% (+39%) | 3.4% (+41%) | 0.0% (+119%) | 5.0% (+36%) | 100% (+0%) | 100% (+0%) |
| 50x50 | 0.0% (+43%) | 0.6% (+48%) | 0.0% (+137%) | 1.8% (+44%) | 100% (+0%) | 100% (+0%) |
| 100x100 | 0.0% (+42%) | 0.0% (+75%) | 0.0% (+146%) | 0.8% (+72%) | 100% (+0%) | 100% (+0%) |
| 200x200 | 0.0% (+42%) | 0.0% (+112%) | 0.0% (+153%) | 0.0% (+83%) | — | 100% (+0%) |

(DP column: uniform and photo maps combined; DP was optimal on all 500 maps of each
kind at every size, 6,000 maps in total.)

- DP was optimal on all 6,000 maps, including the 200x200 maps Graph Cut skips.
- Graph Cut was optimal on all 1,000 maps it was given.
- Greedy's chance of being optimal collapses as the map gets taller: from 52% at
  5x5 to 0% from 50x50 up. Every extra row is another chance to commit to a wrong turn.
- Surprise: on large **photo** crops Greedy is worse than on random maps (+112% vs
  +42% at 200x200) and even worse than Baseline. Photo energy has long strong edges;
  once Greedy enters one it follows it, whereas on random maps a bad pixel is
  followed by an average one.
- The "worst case %" column in the CSV is very large for photo maps (up to 586,700%)
  because some crops (flat sky) have an optimal cost near 0. The total extra cost
  above is the more meaningful number.

### 3.3 E3: Runtime scaling

Slope of the log-log fit of time per seam against pixels (time ~ pixels^slope):

| Method | Square (n x n) | Width only (H = 200) | Height only (W = 200) | ms per seam, 200x200 | ms per seam, 1600x1600 |
|---|---|---|---|---|---|
| Greedy | **0.55** | **0.15** | **0.94** | 0.0036 | 0.040 |
| Baseline | 0.94 | 1.05 | 0.93 | 0.029 | 1.61 |
| Graph Cut | 1.61 | 2.53 | 2.14 | 1,932 | (400x400: 10,696) |
| DP | **1.02** | **0.98** | **1.01** | 0.023 | 2.70 |

(DP row from the DP run. Baseline in the same run: 0.020 ms at 200x200 and 1.66 ms at
1600x1600, close to the figures above, so the two machines are comparable.)

- **DP matches O(HW)** almost exactly: slope 1.02 on square images, and about 1 when
  only the width or only the height grows. Per seam it costs about 1.2–1.6x Baseline,
  which has the same O(HW) order: DP does a three-way minimum and a table write per
  pixel and then backtracks, where Baseline only adds. At 200x200 DP takes 0.023 ms
  per seam against about 1,800 ms for Graph Cut in the same run, roughly 80,000x faster
  for the same (optimal) answer.
- **Greedy matches O(H + W)** exactly: on square images pixels = n^2 and H + W = 2n,
  so the slope should be 0.5 (measured 0.55). With H fixed, growing W only affects the
  O(W) scan of row 0, so time barely grows (0.15). With W fixed, time is linear in H (0.94).
- **Baseline matches O(HW)** (slope ~1 in every series): it sums every column.
- **Graph Cut is clearly worse than linear** (1.6–2.5). Edmonds-Karp's worst case is
  O(V E^2); the measured growth is far below that bound but still about quadratic in
  the number of pixels. At 200x200 one seam takes ~2 s, about **500,000x** slower than
  Greedy; a 400x400 seam takes 10.7 s.

### 3.4 E4: Where the time goes

Share of total carving time spent finding seams (rest = energy update + seam removal):

| Method | Full-size photos | `_small` photos |
|---|---|---|
| Greedy | 1.8–2.2% | 7.6–9.4% |
| Baseline | 21–23% | 25–29% |
| Graph Cut | — | **100.0%** (9.3–13.5 s per photo) |
| DP | 23–25% | 28–30% |

(DP row from the DP run; Baseline in the same run: 19–20% and 23–25%.)

For DP, seam finding is about a quarter of the total time: about 0.3 ms per seam on
the full-size photos, against about 0.6 ms for seam removal and 0.3 ms for the energy
update. Removal is still the largest cost, so a faster DP would shorten the whole
resize only a little (at most about 1.3x, by Amdahl's law).

For Greedy and Baseline, seam **removal** is the largest cost (about 55–65% on the
full photos, the O(HW) memory copy from P1), then the incremental energy update
(~22–35%). Seam finding is cheap. For Graph Cut, everything else is negligible.

## 4. Discussion

**Theory vs measurement.** All measured slopes agree with the complexity analysis:
Greedy O(H + W), Baseline and DP O(HW), Graph Cut super-linear. Small deviations (e.g.
0.55 instead of 0.5) come from fixed per-call overheads that matter at small sizes.

**Quality.** Optimality is what separates the methods. DP and Graph Cut were optimal on
every seam they were tested on, Greedy on almost none beyond tiny maps, and its error is
largest exactly on the images where it matters visually: strong vertical structure
(building, broadway). Greedy is fast, but saving a fraction of a millisecond per seam
is not worth seams that cost up to 6–7x the optimum, because seam finding is only
~2% of the total time anyway (E4).

**Graph Cut.** It gives the same (optimal) answer as DP, with identical total costs on
every `_small` photo, but costs seconds per seam instead of a fraction of a
millisecond, so it is impractical for this problem. It is still worth
having: the min-cut formulation generalises to energies and constraints DP cannot
handle (e.g. 2D regions, video), and here it serves as an independent check of optimality.

**Recommendation:** DP. It was optimal on every seam, like Graph Cut, and runs in
O(HW) at about 1.2–1.6x Baseline's cost per seam, which is roughly a quarter of the
total resize time (E4). Greedy only where speed matters far more than quality; Graph
Cut only on small images or as a correctness reference.

## 5. Validation

The evaluation checks itself while it runs:

- every seam is validated (`validate_seam`) before its cost is used;
- DP and Graph Cut must be optimal on every E2 map, otherwise a warning is printed
  (no warning was printed in either run);
- no gap % was negative in any run, i.e. no method ever beat the ground truth, which
  would mean the ground truth itself was wrong;
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
  more meaningful than the absolute seconds. DP's timings come from a different
  machine than the other methods' (Section 3); its Baseline figures from the same
  run are quoted alongside so the comparison stays like for like.
