# P5: Evaluation and Comparison

This section compares the seam finders (Dynamic Programming, Greedy, Graph Cut, and
the straight-column baseline) on **seam quality** and **speed**, and checks whether
the measured running times match the complexity analysis of each method.

> **Status:** All results are from one full run of `./build/evaluate` with all four
> methods on 18 Nov 2026, on the single machine listed in Section 3. The quality and
> optimality numbers (E1, E2) do not depend on the machine: they came out identical on
> the two machines used earlier in development. The timing results (E3, E4) do depend
> on it, so every timing below is from this one run.

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

**Setup:** AMD Ryzen 5 5600X (12 hardware threads, all code single-threaded),
7.7 GB RAM, WSL2 Ubuntu 26.04 (kernel 6.18), g++ 15.2, OpenCV 4.10, Release build.
Full run with all four methods: 203 s. Raw output: `results/eval_*.csv`,
`results/eval_full_log.txt` and `results/machine.txt`.

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

Slope of the log-log fit of time per seam against pixels (time ~ pixels^slope). Each run
times up to 20 seams (2 for Graph Cut) and the image loses one column per seam, so the
fit uses the mean image size over the timed seams.

| Method | Square (n x n) | Width only (H = 200) | Height only (W = 200) | ms per seam, 200x200 | ms per seam, 1600x1600 |
|---|---|---|---|---|---|
| Greedy | **0.53** | **0.07** | **0.91** | 0.0015 | 0.017 |
| Baseline | 0.86 | 0.65 | 0.77 | 0.020 | 2.26 |
| Graph Cut | 1.58 | 1.79 | 1.71 | 466 | (400x400: 4,205) |
| DP | **0.97** | **0.85** | **0.88** | 0.018 | 2.01 |

- **DP is consistent with O(HW)**: slope 0.97 on square images, and 0.85–0.88 when
  only the width or only the height grows. Those two series start at very small
  sizes, where costs that do not depend on the width still show: over the four
  largest widths the time grows 8.3x for 8.3x the pixels. Per seam it takes between 0.27x and 0.96x
  Baseline's time depending on the size (0.89x at both 200x200 and 1600x1600).
  Both are O(HW): DP does a three-way minimum and a table write per pixel and then
  backtracks, where Baseline only adds. At 200x200 DP takes 0.018 ms per seam against
  466 ms for Graph Cut, roughly 26,000x faster for the same (optimal) answer.
- **Greedy is consistent with O(H + W)**: on square images pixels = n^2 and H + W = 2n,
  so the slope should be 0.5 (measured 0.53). With H fixed, growing W only affects the
  O(W) scan of row 0, so time barely grows (0.07). With W fixed, time is linear in H (0.91).
- **Baseline does not fit a clean line on this machine** (slopes 0.65–0.86). Its find
  step is a plain column sum. Up to 400x400 its time grows more slowly than the pixel
  count, and from 400x400 to 1600x1600 it grows faster (30x the time for 16x the
  pixels). A likely cause is that such a cheap loop is dominated by fixed costs at
  small sizes and by memory speed at large ones. We therefore do not claim a measured
  confirmation of O(HW) for Baseline; it is a testing aid, not one of the three methods.
- **Graph Cut is clearly worse than linear** (1.6–1.8). Edmonds-Karp's worst case is
  O(V E^2); the measured growth is far below that bound, at roughly pixels^1.6 to
  pixels^1.8. At 200x200 one seam takes 0.47 s, about **300,000x** slower than
  Greedy; a 400x400 seam takes 4.2 s.

### 3.4 E4: Where the time goes

Share of total carving time spent finding seams (rest = energy update + seam removal):

| Method | Full-size photos | `_small` photos |
|---|---|---|
| Greedy | 1.8–2.4% | 7.0–7.2% |
| Baseline | 36–44% | 36–41% |
| Graph Cut | — | **100.0%** (3.2–4.8 s per photo) |
| DP | 34–35% | 32–38% |

For DP, seam finding is about a third of the total time: about 0.18 ms per seam on
the full-size photos, against about 0.24 ms for seam removal and 0.12 ms for the energy
update. The energy update time also includes building the first energy map and
removing each seam from the energy map (an O(HW) copy), so it is more than the O(H)
recalculation alone. Removal is larger than finding, so even an infinitely fast DP would
shorten the whole resize by at most about 1.5x (Amdahl's law: 1 / (1 − 0.34) to
1 / (1 − 0.35)).

For Greedy, seam **removal** is the largest cost (63–64% on the
full photos, the O(HW) memory copy from P1), then the incremental energy update
(34–35%); seam finding is cheap. For Baseline, finding (36–44%) and removal (39–44%)
take about the same share. For Graph Cut, everything else is negligible.

## 4. Discussion

**Theory vs measurement.** The measured slopes of the three methods agree with the
complexity analysis: Greedy O(H + W), DP O(HW), Graph Cut super-linear. The largest
deviation, DP at 0.85 when only the width grows, comes from fixed per-call costs that
matter at small sizes. Baseline's slope (0.65–0.86) is the exception and is discussed in Section 3.3.

**Quality.** Optimality is what separates the methods. DP and Graph Cut were optimal on
every seam they were tested on, Greedy on almost none beyond tiny maps, and its error is
largest on the images where it matters visually: strong vertical structure
(building, broadway). Greedy is fast: on the full-size photos the whole resize takes
41–45% less time than with DP (E4). That saving is under 0.05 s per photo here, and it
is paid for with seams that cost up to 7.7x the optimum (a gap of 666% on building).

**Graph Cut.** It gives the same (optimal) answer as DP, with identical total costs on
every `_small` photo, but costs from a tenth of a second to several seconds per seam instead of a fraction of a
millisecond, so it is impractical for this problem. It is still worth
having: the min-cut formulation generalises to energies and constraints DP cannot
handle (e.g. 2D regions, video), and here it serves as an independent check of optimality.

**Recommendation:** DP. It was optimal on every seam, like Graph Cut, and runs in
O(HW) at about 0.18 ms per seam on the full-size photos, which is roughly a third of
the total resize time (E4). Greedy only where speed matters far more than quality; Graph
Cut only on small images or as a correctness reference.

## 5. Validation

The evaluation checks itself while it runs:

- every seam is validated (`validate_seam`) before its cost is used;
- DP and Graph Cut must be optimal on every E2 map, otherwise a warning is printed
  (no warning was printed in this run);
- no gap % was negative in this run, i.e. no method ever beat the ground truth, which
  would mean the ground truth itself was wrong;
- the project tests (`./build/tests`) already check DP and Graph Cut against brute
  force, and that `full` and `incremental` energy modes give identical costs, so
  timings in either mode measure the same work.

## 6. Limitations

- Graph Cut is only evaluated on small images, so its quality numbers cover fewer
  pixels than the other methods. Its time is extrapolated from the fitted slope, and
  its scaling points are single runs of two seams, so that slope is the least certain:
  a repeat of the scaling experiment also gave 1.6–1.8.
- The gap % measures cost (energy), not perceived quality. The output images in
  `results/eval_images/` are used for the visual side.
- Only vertical seams are measured in E1/E4. Horizontal seams use the same finder on
  the transposed image (P1), so the results carry over.
- Timings depend on the machine; the slopes and the ratios between methods are
  more meaningful than the absolute seconds. All timings here are from one run on one
  machine (Section 3). On other hardware and compilers the absolute times and the
  time shares in E4 will differ.
