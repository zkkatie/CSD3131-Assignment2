"""
Main seam carving loop (Assignment item 1c).

    repeat until the image is the target size:
        1. get the energy map            (full recompute, or incremental update - see energy.py)
        2. find the minimum seam         (DP / Greedy / Graph Cut, plugged in as `find_seam`)
        3. remove the seam               (from the image, greyscale, energy and coordinate maps)

Horizontal seams: transpose everything (swap rows and columns), remove vertical
seams, then transpose back. A horizontal seam in the image is a vertical seam in
the transposed image, so each seam finder only needs to handle vertical seams.
The energy map can be transposed too: |gx| + |gy| is symmetric under transposing.

Seam tracking: we carry two extra arrays, `orig_rows` and `orig_cols`, holding
each remaining pixel's position in the ORIGINAL image. They are carved exactly like
the image, so every removed seam can be reported in original coordinates
(used to draw all removed seams on the original picture, bonus item 7a).

Cost per seam (H x W image):
    find_seam                         depends on the method (DP is O(H * W))
    energy, full mode                 O(H * W)
    energy, incremental mode          O(H)
    removing seam from each array     O(H * W)   (numpy copy, very fast in practice)
"""

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .energy import ENERGY_FUNCTIONS, energy_from_gray, to_gray, update_energy
from .seam import remove_vertical_seam, seam_cost, validate_seam

SeamFinder = Callable[[np.ndarray], np.ndarray]
# Optional hook called for every seam, BEFORE it is removed:
#     on_seam(current_image, seam, direction)
# direction is "vertical" or "horizontal". For horizontal seams `current_image` is
# transposed (so the seam is vertical in it). P6 can use this for step-by-step visualisation.
SeamCallback = Callable[[np.ndarray, np.ndarray, str], None]

ENERGY_MODES = ("incremental", "full")


@dataclass
class CarveStats:
    """Everything recorded during carving (P5 uses this for the comparison tables)."""
    seams_removed: int = 0
    total_seam_cost: float = 0.0
    energy_time: float = 0.0   # seconds computing / updating energy maps
    find_time: float = 0.0     # seconds finding seams (this is what differs per method)
    remove_time: float = 0.0   # seconds removing seams from the arrays
    seam_costs: List[float] = field(default_factory=list)
    # Each removed seam in ORIGINAL image coordinates: an (n, 2) int array of (row, col).
    seam_coords: List[np.ndarray] = field(default_factory=list)
    seam_directions: List[str] = field(default_factory=list)

    @property
    def total_time(self) -> float:
        return self.energy_time + self.find_time + self.remove_time


class _State:
    """The arrays that are carved together: image, greyscale, energy and coordinate maps."""

    def __init__(self, img: np.ndarray, energy_fn: str):
        h, w = img.shape[:2]
        self.img = img
        self.gray = to_gray(img)
        self.energy = energy_from_gray(self.gray, energy_fn)
        self.orig_rows, self.orig_cols = np.indices((h, w))

    def transpose(self) -> None:
        self.img = transpose(self.img)
        self.gray = self.gray.T
        self.energy = self.energy.T
        self.orig_rows, self.orig_cols = self.orig_rows.T, self.orig_cols.T


def transpose(img: np.ndarray) -> np.ndarray:
    """Swap rows and columns (keeps the colour channel last)."""
    return np.transpose(img, (1, 0, 2)) if img.ndim == 3 else img.T


def _remove_seams(
    st: _State,
    n: int,
    find_seam: SeamFinder,
    stats: CarveStats,
    on_seam: Optional[SeamCallback],
    direction: str,
    energy_fn: str,
    energy_mode: str,
) -> None:
    """Remove `n` vertical seams from the state, updating stats."""
    for _ in range(n):
        h = st.img.shape[0]

        t0 = time.perf_counter()
        seam = np.asarray(find_seam(st.energy.copy()))  # copy: a finder may modify its input
        t1 = time.perf_counter()
        validate_seam(seam, *st.energy.shape)   # catches bugs in a seam finder immediately
        seam = seam.astype(np.int64)

        cost = seam_cost(st.energy, seam)
        stats.seam_costs.append(cost)
        stats.total_seam_cost += cost
        r = np.arange(h)
        stats.seam_coords.append(np.stack([st.orig_rows[r, seam], st.orig_cols[r, seam]], axis=1))
        stats.seam_directions.append(direction)
        if on_seam is not None:
            on_seam(st.img, seam, direction)

        t2 = time.perf_counter()
        st.img = remove_vertical_seam(st.img, seam)
        st.gray = remove_vertical_seam(st.gray, seam)
        st.orig_rows = remove_vertical_seam(st.orig_rows, seam)
        st.orig_cols = remove_vertical_seam(st.orig_cols, seam)
        t3 = time.perf_counter()

        if energy_mode == "incremental":
            st.energy = update_energy(remove_vertical_seam(st.energy, seam), st.gray, seam, energy_fn)
        else:
            st.energy = energy_from_gray(st.gray, energy_fn)
        t4 = time.perf_counter()

        stats.find_time += t1 - t0
        stats.remove_time += t3 - t2
        stats.energy_time += t4 - t3
        stats.seams_removed += 1


def carve(
    img: np.ndarray,
    target_width: int,
    target_height: int,
    find_seam: SeamFinder,
    on_seam: Optional[SeamCallback] = None,
    energy_fn: str = "sobel",
    energy_mode: str = "incremental",
):
    """
    Shrink `img` to target_width x target_height using seam carving.

    Removes all vertical seams first, then all horizontal seams.

    Args:
        img:           H x W x 3 (BGR) or H x W image.
        target_width:  1 .. W
        target_height: 1 .. H
        find_seam:     seam finder, see the interface in seam.py
        on_seam:       optional callback, see SeamCallback above
        energy_fn:     "sobel" or "gradient"
        energy_mode:   "incremental" (fast, default) or "full" (recompute everything each time).
                       Both give identical results.

    Returns:
        (resized_image, CarveStats)
    """
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError(f"expected a non-empty H x W or H x W x C image, got shape {img.shape}")
    if energy_fn not in ENERGY_FUNCTIONS:
        raise ValueError(f"energy_fn must be one of {ENERGY_FUNCTIONS}, got {energy_fn!r}")
    if energy_mode not in ENERGY_MODES:
        raise ValueError(f"energy_mode must be one of {ENERGY_MODES}, got {energy_mode!r}")
    h, w = img.shape[:2]
    if not (1 <= target_width <= w and 1 <= target_height <= h):
        raise ValueError(
            f"target size {target_width}x{target_height} must be between 1x1 and {w}x{h} "
            "(this program only shrinks images)"
        )

    stats = CarveStats()
    t0 = time.perf_counter()
    st = _State(img, energy_fn)          # includes the first full energy map
    stats.energy_time += time.perf_counter() - t0
    _remove_seams(st, w - target_width, find_seam, stats, on_seam, "vertical", energy_fn, energy_mode)
    if target_height < h:
        st.transpose()
        _remove_seams(st, h - target_height, find_seam, stats, on_seam, "horizontal", energy_fn, energy_mode)
        st.transpose()
    return np.ascontiguousarray(st.img), stats


def remove_vertical_seams(img, n, find_seam, **kwargs):
    """Remove `n` vertical seams (image becomes n pixels narrower). Returns (image, stats)."""
    h, w = img.shape[:2]
    return carve(img, w - n, h, find_seam, **kwargs)


def remove_horizontal_seams(img, n, find_seam, **kwargs):
    """Remove `n` horizontal seams (image becomes n pixels shorter). Returns (image, stats)."""
    h, w = img.shape[:2]
    return carve(img, w, h - n, find_seam, **kwargs)
