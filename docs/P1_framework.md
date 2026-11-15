# P1: Framework, Energy Map and Seam Removal

This section covers Assignment items 1a (energy map) and 1c (remove the seam and
repeat), plus horizontal seams and the shared structure used by DP, Greedy and Graph Cut.

## 1. Design

### 1.1 Pipeline

```
load image
compute greyscale + energy map once
repeat until target width reached:
    seam  = find_seam(energy)            <- DP / Greedy / Graph Cut (pluggable)
    check seam is valid
    remove seam from image, greyscale, energy and coordinate maps
    update energy near the seam
transpose everything, repeat for height, transpose back
```

Every seam finder has the same signature, `find_seam(energy) -> seam`, where
`seam[i]` is the column removed in row `i`. This separates **how a seam is found**
(the part each method changes) from **everything else** (shared and tested once).
The comparison between methods is fair because they run inside the same loop,
with the same energy function and the same timers.

### 1.2 Energy function

The energy of pixel (r, c) is the L1 gradient magnitude of the greyscale image:

    e(r, c) = |gx(r, c)| + |gy(r, c)|

Two gradient operators are implemented so they can be compared:

| Function | gx | Notes |
|---|---|---|
| `sobel` (default) | 3x3 Sobel kernel [-1 0 1; -2 0 2; -1 0 1] | Smooths across 3 rows, so it is less sensitive to noise |
| `gradient` | I(r, c+1) - I(r, c-1) | Simpler, but reacts more to single noisy pixels |

`gy` uses the same kernel turned 90 degrees. At the image border, missing
neighbours are replaced by the nearest pixel (clamping). For a 3x3 kernel this is
the same as OpenCV's `BORDER_REFLECT` (checked by a test). Clamping avoids the
false strong edges that zero-padding would create at the borders, which would
push seams away from the image sides.

### 1.3 Horizontal seams by transposing

A horizontal seam in an image is a vertical seam in its transpose. So instead of
writing every method twice, the framework transposes the image, removes vertical
seams, and transposes back. The energy map can be transposed as well, because
transposing swaps gx and gy and |gx| + |gy| stays the same. A test checks that
removing horizontal seams equals removing vertical seams from the transposed image.

Vertical seams are removed first, then horizontal ones. Interleaving them in the
order of lowest cost (the "optimal order" from Avidan and Shamir) gives slightly
better results but needs a second DP over the order. That is O(k^2) times more work,
so it was left out.

### 1.4 Incremental energy update (efficiency improvement)

The brief says to recompute the energy map after every removal. Recomputing all
of it costs O(H x W) per seam, but removing a seam only changes the energy of
pixels right next to it.

**Why.** The energy of pixel (r, c) depends only on its 3x3 neighbourhood: rows
r-1..r+1 and columns c-1..c+1. Let m and M be the smallest and largest seam column
in rows r-1..r+1 (adjacent seam columns differ by at most 1, so M - m <= 2).
After the seam is removed:

- If c <= m - 2, none of its neighbours moved, so its energy is unchanged.
- If c >= M + 1, all of its neighbours shifted left by one together, so its
  neighbourhood is the same and its energy just shifts left with it.
- Only for m - 1 <= c <= M did the neighbourhood actually change.

So at most 4 pixels per row need recomputing: O(H) work instead of O(H x W).
`update_energy` removes the seam from the old energy map, then recomputes only
those pixels.

**Checking it.** The tests remove random seams from random images (including
1-pixel-high images and seams along the left and right edges or zig-zagging)
until one column is left. After every removal they check that the updated map
equals a full recompute. A whole carve in `incremental` mode also gives
pixel-for-pixel the same output, seams and costs as `full` mode.

`--energy-mode full` is kept so the two can be compared and benchmarked.

### 1.5 Tracking seams in original coordinates

Two extra arrays, `orig_rows` and `orig_cols`, store where each remaining pixel
was in the original image. They are carved exactly like the image, so every
removed seam can be reported in original coordinates even after earlier seams have
shifted the image. This lets us draw **all** removed seams on the original image
(bonus item 7a): `--save-seams`, with red for vertical and blue for horizontal.

### 1.6 Robustness

- Every seam returned by a finder is checked: correct length, in range, whole
  numbers, and connected (`|seam[i] - seam[i-1]| <= 1`). A bug in any method
  fails immediately with a clear message instead of producing a wrong image.
- Each finder gets a copy of the energy map, so a finder that changes its input
  cannot corrupt the next iteration.
- Colour, greyscale and single-channel images are supported, including images
  1 pixel wide or high, and targets down to 1x1.
- Invalid sizes (enlarging, zero, scale outside (0, 1]), unknown options,
  unreadable files and unfinished methods give a clear error, not a crash.

## 2. Complexity

For an H x W image, removing one vertical seam:

| Step | Time | Space |
|---|---|---|
| Greyscale + first energy map (once) | O(HW) | O(HW) |
| Energy after removal, `full` | O(HW) | O(HW) |
| Energy after removal, `incremental` | **O(H)** to compute | O(HW) |
| Validate seam, seam cost, record coordinates | O(H) | O(H) per seam |
| Remove seam from image / grey / energy / coordinate maps | O(HW) copy | O(HW) |
| `find_seam` | depends on the method (DP: O(HW)) | depends on the method |

Removing k vertical seams therefore costs O(k x HW) in total, plus whatever the
seam finder costs. Since DP's own cost is O(HW) per seam, the overall order of DP
seam carving stays O(k x HW) either way. The incremental update removes one of the
O(HW) terms, so the gain is in the constant factor, not the Big-O.

The removal step is still O(HW) because numpy copies each row into a new,
narrower array. In practice this is a fast memory copy, unlike the energy step,
which needs about 20 arithmetic operations per pixel.

Seam tracking adds 2 integer arrays (O(HW)) and stores H coordinates per seam,
O(k x H) in total. This is small compared to the image.

## 3. Results

Benchmark: `python scripts/benchmark_energy.py`. 100 vertical seams were removed
from synthetic images with the baseline finder, so the times measure the framework
rather than a particular method. Best of 3 runs:

| Image | Energy time, full | Energy time, incremental | Speed-up |
|---|---|---|---|
| 200x150 | 0.010 s | 0.010 s | 1.0x |
| 400x300 | 0.047 s | 0.021 s | 2.2x |
| 800x600 | 0.209 s | 0.059 s | 3.5x |
| 1600x1200 | 1.068 s | 0.204 s | 5.2x |

The speed-up grows with image size, as expected when O(HW) work is replaced by
O(H) work. For small images both take about the same time, because of fixed
overheads such as copying arrays and Python function calls. The incremental time
still includes the first full energy map and the O(HW) removal of the seam from
the energy array. *(Re-run the benchmark on your own machine and update this table
before submitting.)*

## 4. Testing

`python -m pytest -v`. The tests also run automatically on GitHub on every push (CI).

| Area | What is tested |
|---|---|
| Energy | Shape and non-negative values, flat image = 0, edges higher than flat areas, a hand-calculated example, Sobel matches OpenCV, pixel-by-pixel formula equals the full map for sizes from 1x1 upward, greyscale input, unknown method rejected |
| Incremental update | Equals a full recompute after **every** removal, until 1 column is left, for 5 image shapes x 2 energy functions; seams on the left edge, right edge, zig-zag and diagonal |
| Seam helpers | Valid and invalid seams (jump, wrong length, out of range, not whole numbers), exact pixels removed, colour channels stay together, cost, drawing does not change the input |
| Carving | Reaches the target size; no-op; down to 1x1; greyscale; 1-pixel-wide or high images; invalid targets and options rejected; broken finder caught; finder that changes its input is harmless; incremental = full; output rebuilt from the reported seam coordinates; horizontal = vertical on the transpose; no pixel removed twice and the counts add up; a textured object survives; callback order; timings recorded |
| Seam finders | For every method: valid seams on any size; never cheaper than the true minimum. DP and Graph Cut: exactly the brute-force minimum on random grids. Greedy: follows the local minimum; a trap case where Greedy is worse than optimal |
| Command line | End-to-end run with all outputs; `--scale`; invalid sizes and missing files return an error code |

Tests for DP, Greedy and Graph Cut are skipped until those methods are written,
then run automatically.

## 5. Limitations and possible improvements

- **Only shrinking is supported.** Enlarging by inserting seams is not part of the brief.
- **Backward energy only.** Forward energy (Rubinstein et al., 2008) measures the
  new edges created by removing a pixel and usually gives fewer artefacts. It changes
  the DP recurrence, so it belongs in the DP and Graph Cut sections.
- **Removal still copies the arrays.** Storing each row as a linked list, or marking
  pixels as deleted and compacting only every few seams, would make removal O(H) as
  well, but numpy's vectorised copying is faster in practice for these image sizes.
- **Parallelism.** The full energy map and the seam removal are independent per
  pixel or per row, so they could run on several threads (OpenCV's Sobel already
  does). The seam loop itself is sequential, because each seam depends on the
  previous removal.
