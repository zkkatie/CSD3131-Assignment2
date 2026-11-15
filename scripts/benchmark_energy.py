"""
Benchmark: full energy recompute vs incremental update (P1 design justification).

Uses the baseline seam finder so the timing is dominated by the framework itself,
not by a particular seam finding method.

    python scripts/benchmark_energy.py
    python scripts/benchmark_energy.py --image images/photo.jpg --seams 100
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from seamcarve import METHODS, carve  # noqa: E402


def synthetic_image(h, w, seed=0):
    """Smooth background with random shapes, so it has both flat and busy areas."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.stack([(x * 255 // max(w - 1, 1)), (y * 255 // max(h - 1, 1)), np.full((h, w), 120)], axis=2)
    img = img.astype(np.uint8)
    for _ in range(12):
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, h // 6)), c, -1)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", help="image to use (default: synthetic images of several sizes)")
    p.add_argument("--seams", type=int, default=100, help="vertical seams to remove")
    p.add_argument("--repeats", type=int, default=3, help="best of N runs")
    args = p.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"could not read {args.image}")
        images = [(os.path.basename(args.image), img)]
    else:
        images = [(f"{w}x{h}", synthetic_image(h, w)) for w, h in [(200, 150), (400, 300), (800, 600), (1600, 1200)]]

    print(f"{'image':>12} {'mode':>12} {'energy s':>10} {'total s':>10} {'energy speed-up':>16}")
    for name, img in images:
        n = min(args.seams, img.shape[1] - 1)
        best = {}
        for mode in ("full", "incremental"):
            runs = []
            for _ in range(args.repeats):
                t = time.perf_counter()
                out, stats = carve(img, img.shape[1] - n, img.shape[0], METHODS["baseline"], energy_mode=mode)
                runs.append((stats.energy_time, time.perf_counter() - t, out))
            best[mode] = min(runs, key=lambda r: r[0])
        assert (best["full"][2] == best["incremental"][2]).all(), "modes disagree!"
        for mode in ("full", "incremental"):
            e, total, _ = best[mode]
            speed = f"{best['full'][0] / e:.1f}x" if mode == "incremental" else ""
            print(f"{name:>12} {mode:>12} {e:>10.3f} {total:>10.3f} {speed:>16}")


if __name__ == "__main__":
    main()
