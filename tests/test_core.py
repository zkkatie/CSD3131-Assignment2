"""
Tests for the shared framework (P1) and for every seam finder.

Run with:  python -m pytest -v

Tests for a seam finder are skipped until its file (e.g. seamcarve/methods/dp.py)
is added, then start running automatically.
"""

import os

import cv2
import numpy as np
import pytest

import main as cli
from seamcarve import (
    ENERGY_FUNCTIONS,
    METHODS,
    carve,
    compute_energy,
    draw_removed_seams,
    draw_vertical_seam,
    energy_from_gray,
    remove_horizontal_seams,
    remove_vertical_seam,
    remove_vertical_seams,
    seam_cost,
    to_gray,
    transpose,
    update_energy,
    validate_seam,
)
from seamcarve.energy import energy_at
from seamcarve.methods import PLANNED_METHODS

BASELINE = METHODS["baseline"]


# ---------------------------------------------------------------- helpers

def random_image(h, w, seed=0, channels=3):
    rng = np.random.default_rng(seed)
    shape = (h, w, channels) if channels else (h, w)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def random_seam(h, w, rng):
    """A random valid vertical seam."""
    s = [int(rng.integers(0, w))]
    for _ in range(h - 1):
        s.append(int(np.clip(s[-1] + rng.integers(-1, 2), 0, w - 1)))
    return np.array(s)


def brute_force_min_cost(energy):
    """Try every seam (exponentially many) - only for tiny grids."""
    h, w = energy.shape
    best = np.inf

    def search(row, col, total):
        nonlocal best
        total += energy[row, col]
        if row == h - 1:
            best = min(best, total)
            return
        for nc in (col - 1, col, col + 1):
            if 0 <= nc < w:
                search(row + 1, nc, total)

    for c in range(w):
        search(0, c, 0.0)
    return best


def finder_or_skip(name):
    if name not in METHODS:
        pytest.skip(f"{name} not added yet (seamcarve/methods/{name}.py)")
    return METHODS[name]


# ---------------------------------------------------------------- energy

@pytest.mark.parametrize("method", ENERGY_FUNCTIONS)
def test_energy_shape_and_nonnegative(method):
    img = random_image(13, 17)
    e = compute_energy(img, method)
    assert e.shape == (13, 17) and e.dtype == np.float64 and (e >= 0).all()


@pytest.mark.parametrize("method", ENERGY_FUNCTIONS)
def test_flat_image_has_zero_energy(method):
    assert np.allclose(compute_energy(np.full((8, 9, 3), 77, np.uint8), method), 0)


@pytest.mark.parametrize("method", ENERGY_FUNCTIONS)
def test_edge_has_higher_energy_than_flat_area(method):
    img = np.zeros((10, 10, 3), np.uint8)
    img[:, 5:] = 255
    e = compute_energy(img, method)
    assert e[:, 4:6].min() > 0 and np.allclose(e[:, :3], 0) and np.allclose(e[:, 7:], 0)


def test_gradient_energy_hand_computed():
    g = np.array([[0, 1, 4], [9, 16, 25]], dtype=np.float64)
    # pixel (0,1): gx = 4 - 0 = 4,  gy = 16 - 1 (top clamps to itself) = 15  -> 19
    # pixel (1,0): gx = 16 - 9 (left clamps) = 7,  gy = 9 - 0 = 9            -> 16
    e = energy_from_gray(g, "gradient")
    assert e[0, 1] == 19 and e[1, 0] == 16


def test_sobel_matches_opencv_reflect_border():
    """Our clamped border is the same as OpenCV's BORDER_REFLECT for a 3x3 kernel."""
    g = to_gray(random_image(11, 14, seed=3))
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    assert np.allclose(energy_from_gray(g, "sobel"), np.abs(gx) + np.abs(gy))


@pytest.mark.parametrize("method", ENERGY_FUNCTIONS)
@pytest.mark.parametrize("h,w", [(1, 1), (1, 6), (6, 1), (2, 2), (9, 13)])
def test_energy_at_matches_full_map(method, h, w):
    g = to_gray(random_image(h, w, seed=h * 31 + w))
    rows, cols = np.indices((h, w))
    pointwise = energy_at(g, rows.ravel(), cols.ravel(), method).reshape(h, w)
    assert np.allclose(pointwise, energy_from_gray(g, method))


def test_greyscale_and_single_channel_input():
    grey = random_image(6, 7, channels=0)
    assert np.allclose(compute_energy(grey), compute_energy(grey[:, :, None]))


def test_unknown_energy_method_rejected():
    with pytest.raises(ValueError):
        compute_energy(random_image(4, 4), "magic")


# ------------------------------------------------- incremental energy update

@pytest.mark.parametrize("method", ENERGY_FUNCTIONS)
@pytest.mark.parametrize("h,w", [(1, 8), (2, 2), (5, 3), (12, 20), (30, 7)])
def test_incremental_energy_equals_full_recompute(method, h, w):
    """Remove seams until 1 column is left; after every removal both ways must agree."""
    rng = np.random.default_rng(h * 100 + w)
    gray = to_gray(random_image(h, w, seed=h + w))
    energy = energy_from_gray(gray, method)
    while gray.shape[1] > 1:
        seam = random_seam(h, gray.shape[1], rng)
        gray = remove_vertical_seam(gray, seam)
        energy = update_energy(remove_vertical_seam(energy, seam), gray, seam, method)
        assert np.allclose(energy, energy_from_gray(gray, method))


@pytest.mark.parametrize("seam_kind", ["left_edge", "right_edge", "zigzag", "diagonal"])
def test_incremental_energy_extreme_seams(seam_kind):
    h, w = 12, 10
    gray = to_gray(random_image(h, w, seed=7))
    seam = {
        "left_edge": np.zeros(h, int),
        "right_edge": np.full(h, w - 1),
        "zigzag": np.array([4 + (i % 2) for i in range(h)]),
        "diagonal": np.minimum(np.arange(h), w - 1),
    }[seam_kind]
    new_gray = remove_vertical_seam(gray, seam)
    energy = update_energy(remove_vertical_seam(energy_from_gray(gray), seam), new_gray, seam)
    assert np.allclose(energy, energy_from_gray(new_gray))


# ---------------------------------------------------------------- seams

def test_validate_seam_accepts_good_seam():
    validate_seam(np.array([2, 3, 3, 2]), 4, 5)
    validate_seam(np.array([0]), 1, 1)


@pytest.mark.parametrize("seam", [
    [0, 2, 2, 2],      # jumps 2 columns
    [0, 0, 0],         # wrong length
    [0, 0, 0, 5],      # out of range (right)
    [-1, 0, 0, 0],     # out of range (left)
    [0.5, 1, 1, 1],    # not a whole number
])
def test_validate_seam_rejects_bad_seam(seam):
    with pytest.raises(ValueError):
        validate_seam(np.array(seam), 4, 5)


def test_remove_vertical_seam_removes_exact_pixels():
    img = np.arange(12).reshape(3, 4)          # [0 1 2 3] [4 5 6 7] [8 9 10 11]
    out = remove_vertical_seam(img, np.array([0, 1, 2]))
    assert out.tolist() == [[1, 2, 3], [4, 6, 7], [8, 9, 11]]


def test_remove_vertical_seam_colour_keeps_channels_together():
    img = random_image(5, 6)
    seam = np.array([1, 1, 2, 3, 3])
    out = remove_vertical_seam(img, seam)
    assert out.shape == (5, 5, 3)
    for r in range(5):
        assert (out[r] == np.delete(img[r], seam[r], axis=0)).all()


def test_seam_cost():
    e = np.arange(6, dtype=float).reshape(2, 3)
    assert seam_cost(e, np.array([1, 2])) == 1 + 5


def test_draw_vertical_seam_does_not_modify_input():
    img = np.zeros((3, 3, 3), np.uint8)
    out = draw_vertical_seam(img, np.array([0, 1, 2]))
    assert img.sum() == 0 and (out[1, 1] == (0, 0, 255)).all()


def test_transpose_round_trip():
    img = random_image(7, 9)
    assert transpose(img).shape == (9, 7, 3)
    assert (transpose(transpose(img)) == img).all()


# ---------------------------------------------------------------- carve

def test_carve_reaches_target_size():
    out, stats = carve(random_image(30, 40), 32, 25, BASELINE)
    assert out.shape == (25, 32, 3)
    assert stats.seams_removed == (40 - 32) + (30 - 25)
    assert stats.seam_directions == ["vertical"] * 8 + ["horizontal"] * 5
    assert len(stats.seam_costs) == len(stats.seam_coords) == 13


def test_carve_no_change_returns_same_image():
    img = random_image(10, 10)
    out, stats = carve(img, 10, 10, BASELINE)
    assert (out == img).all() and stats.seams_removed == 0


def test_carve_down_to_1x1():
    out, _ = carve(random_image(6, 5), 1, 1, BASELINE)
    assert out.shape == (1, 1, 3)


def test_carve_greyscale_image():
    out, _ = carve(random_image(10, 12, channels=0), 7, 8, BASELINE)
    assert out.shape == (8, 7)


def test_carve_single_row_and_single_column_images():
    assert carve(random_image(1, 10), 4, 1, BASELINE)[0].shape == (1, 4, 3)
    assert carve(random_image(10, 1), 1, 4, BASELINE)[0].shape == (4, 1, 3)


@pytest.mark.parametrize("tw,th", [(11, 10), (10, 11), (0, 5), (5, 0)])
def test_carve_rejects_invalid_target(tw, th):
    with pytest.raises(ValueError):
        carve(random_image(10, 10), tw, th, BASELINE)


def test_carve_rejects_bad_options():
    with pytest.raises(ValueError):
        carve(random_image(5, 5), 4, 5, BASELINE, energy_fn="magic")
    with pytest.raises(ValueError):
        carve(random_image(5, 5), 4, 5, BASELINE, energy_mode="sometimes")


def test_carve_rejects_invalid_seam_from_finder():
    bad_finder = lambda e: np.arange(e.shape[0]) * 2    # jumps 2 columns per row
    with pytest.raises(ValueError):
        carve(random_image(5, 12), 10, 5, bad_finder)


def test_finder_may_return_list_and_modify_its_input():
    def messy_finder(energy):
        seam = [int(x) for x in BASELINE(energy)]
        energy[:] = -1                       # must not corrupt carve's own energy map
        return seam
    a, sa = carve(random_image(12, 15, seed=4), 9, 10, messy_finder)
    b, sb = carve(random_image(12, 15, seed=4), 9, 10, BASELINE)
    assert (a == b).all() and sa.seam_costs == sb.seam_costs


@pytest.mark.parametrize("energy_fn", ENERGY_FUNCTIONS)
def test_incremental_and_full_modes_give_identical_results(energy_fn):
    img = random_image(25, 30, seed=5)
    a, sa = carve(img, 18, 20, BASELINE, energy_fn=energy_fn, energy_mode="incremental")
    b, sb = carve(img, 18, 20, BASELINE, energy_fn=energy_fn, energy_mode="full")
    assert (a == b).all()
    assert np.allclose(sa.seam_costs, sb.seam_costs)
    assert all((x == y).all() for x, y in zip(sa.seam_coords, sb.seam_coords))


def test_vertical_only_result_matches_removed_coordinates():
    """Rebuild the output from the original by deleting the reported seam pixels."""
    img = random_image(15, 20, seed=6)
    out, stats = carve(img, 12, 15, BASELINE)
    removed = np.zeros((15, 20), bool)
    for coords in stats.seam_coords:
        assert len(coords) == 15
        removed[coords[:, 0], coords[:, 1]] = True
    for r in range(15):
        assert (out[r] == img[r][~removed[r]]).all()


def test_horizontal_only_matches_vertical_on_transposed_image():
    img = random_image(20, 15, seed=8)
    a, _ = remove_horizontal_seams(img, 6, BASELINE)
    b, _ = remove_vertical_seams(transpose(img), 6, BASELINE)
    assert (a == transpose(b)).all()


def test_every_removed_pixel_is_unique_and_counts_add_up():
    h, w, th, tw = 18, 22, 11, 13
    img = random_image(h, w, seed=9)
    out, stats = carve(img, tw, th, BASELINE)
    coords = np.concatenate(stats.seam_coords)
    assert len(coords) == h * w - th * tw
    assert len({(int(r), int(c)) for r, c in coords}) == len(coords)    # no pixel removed twice
    assert coords[:, 0].max() < h and coords[:, 1].max() < w


def test_carve_keeps_textured_object():
    """
    Flat background = zero energy; a striped bar = high energy inside it.
    Seams should remove background only, so all 3 bar columns survive.
    (A SOLID bar would not work: its interior is flat, so it has zero energy too.)
    """
    img = np.full((30, 60, 3), 30, np.uint8)
    img[::2, 28:31] = 255
    out, _ = carve(img, 40, 30, BASELINE)
    assert (out == 255).all(axis=2).any(axis=0).sum() == 3


def test_on_seam_callback_sees_image_before_removal():
    seen = []

    def callback(image, seam, direction):
        seen.append((image.shape, direction))
        validate_seam(seam, image.shape[0], image.shape[1])

    carve(random_image(10, 12), 10, 9, BASELINE, on_seam=callback)
    assert seen == [((10, 12, 3), "vertical"), ((10, 11, 3), "vertical"), ((10, 10, 3), "horizontal")]


def test_stats_timings_are_recorded():
    _, stats = carve(random_image(20, 20), 15, 15, BASELINE)
    assert stats.energy_time > 0 and stats.find_time > 0 and stats.remove_time > 0
    assert stats.total_time == pytest.approx(stats.energy_time + stats.find_time + stats.remove_time)
    assert stats.total_seam_cost == pytest.approx(sum(stats.seam_costs))


def test_draw_removed_seams_colours_exactly_the_removed_pixels():
    img = np.zeros((10, 12, 3), np.uint8)
    img[::2] = 50                                        # some texture
    _, stats = carve(img, 9, 8, BASELINE)
    drawn = draw_removed_seams(img, stats.seam_coords, stats.seam_directions)
    red = (drawn == (0, 0, 255)).all(axis=2)
    blue = (drawn == (255, 0, 0)).all(axis=2)
    assert red.sum() == 3 * 10 and blue.sum() == 2 * 9


# ---------------------------------------------------------------- seam finders

@pytest.mark.parametrize("name", PLANNED_METHODS)
def test_method_returns_valid_seam(name):
    finder = finder_or_skip(name)
    for seed, (h, w) in enumerate([(1, 1), (1, 5), (6, 1), (2, 2), (20, 25)]):
        energy = compute_energy(random_image(h, w, seed=seed))
        validate_seam(np.asarray(finder(energy)), h, w)


@pytest.mark.parametrize("name", PLANNED_METHODS)
def test_method_never_beats_the_true_minimum(name):
    finder = finder_or_skip(name)
    energy = np.random.default_rng(1).random((6, 5))
    assert seam_cost(energy, np.asarray(finder(energy))) >= brute_force_min_cost(energy) - 1e-9


@pytest.mark.parametrize("name", ["dp", "graphcut"])
@pytest.mark.parametrize("seed", range(5))
def test_optimal_methods_find_minimum_seam(name, seed):
    """DP and Graph Cut must find the cheapest seam. Checked by brute force on tiny grids."""
    finder = finder_or_skip(name)
    rng = np.random.default_rng(seed)
    energy = rng.random((int(rng.integers(2, 7)), int(rng.integers(2, 6))))
    assert seam_cost(energy, np.asarray(finder(energy))) == pytest.approx(brute_force_min_cost(energy))


def test_greedy_follows_local_minimum():
    """Greedy starts at the cheapest top pixel and always steps to the cheapest pixel below."""
    finder = finder_or_skip("greedy")
    energy = np.array([
        [5, 1, 5, 5],
        [5, 5, 2, 5],
        [9, 9, 9, 3],
        [5, 5, 0, 5],
    ], dtype=float)
    assert list(np.asarray(finder(energy))) == [1, 2, 3, 2]


def test_greedy_can_be_worse_than_optimal():
    """Classic trap: the cheap start leads into an expensive row."""
    finder = finder_or_skip("greedy")
    energy = np.array([
        [0, 9, 9, 1],
        [9, 9, 9, 1],
        [100, 100, 9, 1],
    ], dtype=float)
    assert seam_cost(energy, np.asarray(finder(energy))) > brute_force_min_cost(energy)


# ---------------------------------------------------------------- command line

def test_cli_end_to_end(tmp_path):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), random_image(20, 24))
    out = tmp_path / "out.png"
    code = cli.main([str(src), "--method", "baseline", "--width", "18", "--height", "15",
                     "--out", str(out), "--save-energy", "--save-seams"])
    assert code == 0
    assert cv2.imread(str(out)).shape == (15, 18, 3)
    assert os.path.exists(tmp_path / "out_energy.png") and os.path.exists(tmp_path / "out_seams.png")


def test_cli_scale(tmp_path):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), random_image(20, 30))
    out = tmp_path / "out.png"
    assert cli.main([str(src), "--method", "baseline", "--scale", "0.5", "--out", str(out)]) == 0
    assert cv2.imread(str(out)).shape == (10, 15, 3)


@pytest.mark.parametrize("extra", [["--width", "999"], ["--scale", "1.5"], ["--scale", "0"]])
def test_cli_rejects_bad_size(tmp_path, extra):
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), random_image(10, 10))
    assert cli.main([str(src), "--method", "baseline", "--out", str(tmp_path / "o.png"), *extra]) == 1


def test_cli_missing_file(tmp_path):
    assert cli.main([str(tmp_path / "nope.png"), "--method", "baseline"]) == 1
