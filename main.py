"""
Command-line entry point.

Examples:
    python main.py images/input.jpg --width 300 --method dp
    python main.py images/input.jpg --scale 0.8 --method greedy --out results/out.png
    python main.py images/input.jpg --width 300 --height 200 --save-seams --save-energy
    python main.py images/input.jpg --width 300 --energy gradient --energy-mode full
"""

import argparse
import os
import sys

import cv2
import numpy as np

from seamcarve import ENERGY_FUNCTIONS, ENERGY_MODES, METHODS, carve, compute_energy, draw_removed_seams


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Seam carving image resizer (shrinks images)")
    p.add_argument("input", help="path to the input image")
    default_method = "dp" if "dp" in METHODS else "baseline"
    p.add_argument("--method", choices=sorted(METHODS), default=default_method,
                   help=f"seam finding method (default: {default_method})")
    p.add_argument("--width", type=int, help="target width in pixels (default: unchanged)")
    p.add_argument("--height", type=int, help="target height in pixels (default: unchanged)")
    p.add_argument("--scale", type=float, help="scale both sides, e.g. 0.8 (overrides --width/--height)")
    p.add_argument("--energy", choices=ENERGY_FUNCTIONS, default="sobel", help="energy function (default: sobel)")
    p.add_argument("--energy-mode", choices=ENERGY_MODES, default="incremental",
                   help="update energy only near the removed seam, or recompute it fully (same result)")
    p.add_argument("--out", help="output path (default: results/<name>_<method>.png)")
    p.add_argument("--save-energy", action="store_true", help="also save the energy map of the input")
    p.add_argument("--save-seams", action="store_true",
                   help="also save the input with every removed seam drawn (red = vertical, blue = horizontal)")
    return p.parse_args(argv)


def energy_to_image(energy: np.ndarray) -> np.ndarray:
    """Scale energy to 0-255 so it can be saved as a greyscale picture."""
    peak = energy.max()
    if peak == 0:
        return np.zeros(energy.shape, dtype=np.uint8)
    return (energy / peak * 255).astype(np.uint8)


def target_size(w: int, h: int, args) -> tuple:
    if args.scale is not None:
        if not 0 < args.scale <= 1:
            raise ValueError("--scale must be in (0, 1]; this program only shrinks images")
        return max(1, round(w * args.scale)), max(1, round(h * args.scale))
    return (args.width if args.width is not None else w,
            args.height if args.height is not None else h)


def main(argv=None) -> int:
    args = parse_args(argv)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print(f"error: could not read image {args.input!r}", file=sys.stderr)
        return 1
    h, w = img.shape[:2]

    try:
        target_w, target_h = target_size(w, h, args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    name = os.path.splitext(os.path.basename(args.input))[0]
    out_path = args.out or os.path.join("results", f"{name}_{args.method}.png")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    base = os.path.splitext(out_path)[0]

    print(f"{args.method} ({args.energy}, {args.energy_mode}): {w}x{h} -> {target_w}x{target_h}")
    try:
        result, stats = carve(img, target_w, target_h, METHODS[args.method],
                              energy_fn=args.energy, energy_mode=args.energy_mode)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    cv2.imwrite(out_path, result)
    print(f"saved            {out_path}")
    if args.save_energy:
        cv2.imwrite(f"{base}_energy.png", energy_to_image(compute_energy(img, args.energy)))
        print(f"saved            {base}_energy.png")
    if args.save_seams:
        cv2.imwrite(f"{base}_seams.png", draw_removed_seams(img, stats.seam_coords, stats.seam_directions))
        print(f"saved            {base}_seams.png")

    print(f"seams removed    {stats.seams_removed}")
    print(f"total seam cost  {stats.total_seam_cost:.1f}")
    print(f"energy time      {stats.energy_time:.3f}s")
    print(f"seam find time   {stats.find_time:.3f}s")
    print(f"removal time     {stats.remove_time:.3f}s")
    print(f"total time       {stats.total_time:.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
