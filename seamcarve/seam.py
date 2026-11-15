"""
Seam helpers shared by every method.

THE SHARED INTERFACE (everyone must follow this):

    def find_seam(energy: np.ndarray) -> np.ndarray

    - energy: H x W float array (from compute_energy)
    - returns: 1-D int array `seam` of length H, where seam[i] is the
      COLUMN of the pixel removed in row i (a VERTICAL seam, top to bottom).
    - the seam must be connected: |seam[i] - seam[i-1]| <= 1

Seam finders only ever handle VERTICAL seams. Horizontal seams are done in
carve.py by transposing the image, so DP / Greedy / Graph Cut do not need
separate horizontal code.
"""

import numpy as np


def validate_seam(seam: np.ndarray, height: int, width: int) -> None:
    """Raise ValueError if `seam` is not a valid vertical seam for an H x W image."""
    seam = np.asarray(seam)
    if seam.shape != (height,):
        raise ValueError(f"seam must have length {height}, got shape {seam.shape}")
    if not np.issubdtype(seam.dtype, np.integer) and not np.array_equal(seam, np.round(seam)):
        raise ValueError("seam columns must be whole numbers")
    if seam.min() < 0 or seam.max() >= width:
        raise ValueError(f"seam column out of range [0, {width - 1}]")
    if height > 1 and np.abs(np.diff(seam)).max() > 1:
        raise ValueError("seam is not connected (adjacent rows differ by more than 1 column)")


def seam_cost(energy: np.ndarray, seam: np.ndarray) -> float:
    """Total energy of a seam. Use this to compare DP vs Greedy vs Graph Cut."""
    return float(energy[np.arange(energy.shape[0]), seam].sum())


def remove_vertical_seam(img: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """
    Remove one pixel per row (the pixel at column seam[i] in row i).

    Works for H x W and H x W x C arrays. Returns an H x (W-1) (x C) array.
    Uses a boolean mask, so it runs in O(H * W) without Python loops.
    """
    h, w = img.shape[:2]
    mask = np.ones((h, w), dtype=bool)
    mask[np.arange(h), seam] = False          # mark the seam pixels for removal
    if img.ndim == 3:
        return img[mask].reshape(h, w - 1, img.shape[2])
    return img[mask].reshape(h, w - 1)


def draw_vertical_seam(img: np.ndarray, seam: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Return a copy of `img` with the seam painted in `color` (BGR, default red)."""
    out = _as_bgr(img)
    out[np.arange(img.shape[0]), seam] = color
    return out


def draw_removed_seams(
    original: np.ndarray,
    seam_coords,
    seam_directions,
    vertical_color=(0, 0, 255),
    horizontal_color=(255, 0, 0),
) -> np.ndarray:
    """
    Paint every removed seam onto the ORIGINAL image (bonus item 7a).

    Args:
        original:        the image before carving.
        seam_coords:     CarveStats.seam_coords, list of (n, 2) arrays of (row, col).
        seam_directions: CarveStats.seam_directions, "vertical" / "horizontal" per seam.

    Vertical seams are red and horizontal seams blue by default (BGR colours).
    """
    out = _as_bgr(original)
    for coords, direction in zip(seam_coords, seam_directions):
        color = vertical_color if direction == "vertical" else horizontal_color
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _as_bgr(img: np.ndarray) -> np.ndarray:
    """Copy of the image as 3-channel uint8 so coloured seams can be drawn on it."""
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2).astype(np.uint8)
    return img[:, :, :3].astype(np.uint8).copy()
