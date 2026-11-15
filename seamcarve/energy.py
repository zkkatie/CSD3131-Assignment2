"""
Energy map (Assignment item 1a).

The energy of a pixel measures how "important" it is. Pixels on edges
(a big change in intensity) get high energy; flat areas get low energy.
Seams follow low-energy pixels, so flat areas are removed first.

Two energy functions are provided so they can be compared in the report:

    "sobel"     3x3 Sobel filter (default, recommended in the brief)
                gx = [-1 0 1; -2 0 2; -1 0 1] * I     gy = its transpose
    "gradient"  simple central difference
                gx = I(r, c+1) - I(r, c-1)          gy = I(r+1, c) - I(r-1, c)

    energy = |gx| + |gy|          (L1 gradient magnitude, on the greyscale image)

Borders: a neighbour outside the image is replaced by the nearest pixel inside it
("replicate" / clamp). For a 3x3 kernel this is identical to OpenCV's BORDER_REFLECT.

Two ways to get the energy after a seam is removed:

    full         recompute every pixel:                         O(H * W) per seam
    incremental  only recompute pixels next to the removed seam: O(H) per seam

Removing a seam only changes the 3x3 neighbourhood of pixels within a few
columns of it; everywhere else the neighbourhood just shifts left by one, so the
energy value is unchanged and can be shifted too. tests/test_core.py checks that
both ways give exactly the same energy map.
"""

import cv2
import numpy as np

ENERGY_FUNCTIONS = ("sobel", "gradient")


def to_gray(img: np.ndarray) -> np.ndarray:
    """BGR (H x W x 3) or greyscale (H x W) image -> H x W float64 greyscale."""
    if img.ndim == 3:
        if img.shape[2] == 1:
            return img[:, :, 0].astype(np.float64)
        return cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    return img.astype(np.float64)


def energy_from_gray(gray: np.ndarray, method: str = "sobel") -> np.ndarray:
    """Full energy map of a greyscale image. O(H * W)."""
    if method == "sobel":
        # OpenCV is allowed for Sobel. BORDER_REPLICATE = clamp at the edges.
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
        return np.abs(gx) + np.abs(gy)
    if method == "gradient":
        p = np.pad(gray, 1, mode="edge")
        gx = p[1:-1, 2:] - p[1:-1, :-2]
        gy = p[2:, 1:-1] - p[:-2, 1:-1]
        return np.abs(gx) + np.abs(gy)
    raise ValueError(f"unknown energy method {method!r}, choose from {ENERGY_FUNCTIONS}")


def compute_energy(img: np.ndarray, method: str = "sobel") -> np.ndarray:
    """
    Energy map of an image.

    Args:
        img: H x W x 3 (BGR, as loaded by OpenCV) or H x W (greyscale) array.
        method: "sobel" (default) or "gradient".

    Returns:
        H x W float64 array of energy values (>= 0).
    """
    return energy_from_gray(to_gray(img), method)


def energy_at(gray: np.ndarray, rows: np.ndarray, cols: np.ndarray, method: str = "sobel") -> np.ndarray:
    """
    Energy of selected pixels only (same formula and borders as energy_from_gray).
    rows/cols are equal-length int arrays. Cost is O(len(rows)).
    """
    h, w = gray.shape
    rm, rp = np.clip(rows - 1, 0, h - 1), np.clip(rows + 1, 0, h - 1)
    cm, cp = np.clip(cols - 1, 0, w - 1), np.clip(cols + 1, 0, w - 1)
    g = gray
    if method == "sobel":
        gx = (g[rm, cp] + 2 * g[rows, cp] + g[rp, cp]) - (g[rm, cm] + 2 * g[rows, cm] + g[rp, cm])
        gy = (g[rp, cm] + 2 * g[rp, cols] + g[rp, cp]) - (g[rm, cm] + 2 * g[rm, cols] + g[rm, cp])
    elif method == "gradient":
        gx = g[rows, cp] - g[rows, cm]
        gy = g[rp, cols] - g[rm, cols]
    else:
        raise ValueError(f"unknown energy method {method!r}, choose from {ENERGY_FUNCTIONS}")
    return np.abs(gx) + np.abs(gy)


def update_energy(
    energy_after_removal: np.ndarray,
    gray_after_removal: np.ndarray,
    seam: np.ndarray,
    method: str = "sobel",
) -> np.ndarray:
    """
    Incremental energy update after removing a vertical seam. O(H).

    Args:
        energy_after_removal: the OLD energy map with the seam already removed (H x W-1).
        gray_after_removal:   the greyscale image with the seam removed (H x W-1).
        seam:                 the seam that was removed (columns in the OLD image).

    Why only a few columns change:
        Pixel (r, c) depends on rows r-1..r+1 and columns c-1..c+1. Let m and M be the
        smallest and largest seam column in rows r-1..r+1.
          - if c <= m-2, none of its neighbours moved            -> energy unchanged
          - if c >= M+1, all of its neighbours moved left by one  -> energy unchanged (just shifted)
          - otherwise (m-1 <= c <= M) its neighbourhood changed  -> recompute
        Since adjacent seam columns differ by at most 1, M - m <= 2, so at most 4
        pixels per row are recomputed: columns m-1 .. m+2.

    Returns the updated energy map (modified in place and returned).
    """
    h, w = gray_after_removal.shape
    if w == 0:
        return energy_after_removal
    seam = np.asarray(seam, dtype=np.int64)
    up = np.concatenate(([seam[0]], seam[:-1]))      # seam column in row r-1 (clamped)
    down = np.concatenate((seam[1:], [seam[-1]]))    # seam column in row r+1 (clamped)
    m = np.minimum(np.minimum(up, seam), down)

    rows = np.repeat(np.arange(h), 4)
    cols = np.clip((m - 1)[:, None] + np.arange(4)[None, :], 0, w - 1).ravel()
    energy_after_removal[rows, cols] = energy_at(gray_after_removal, rows, cols, method)
    return energy_after_removal
