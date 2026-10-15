import numpy as np
import pytest

from sda.geometry import centroids, feet, homography, to_ground, violating_pairs

BOXES = np.array([[0, 0, 10, 20], [100, 50, 120, 90]])


def test_reference_points():
    assert centroids(BOXES).tolist() == [[5, 10], [110, 70]]
    assert feet(BOXES).tolist() == [[5, 20], [110, 90]]


def test_homography_recovers_ground_coordinates():
    # A 4 m x 4 m floor square seen in perspective: far edge is shorter in the image.
    img = [[200, 400], [600, 400], [500, 200], [300, 200]]
    ground = [[0, 0], [4, 0], [4, 4], [0, 4]]
    h = homography(img, ground)
    np.testing.assert_allclose(to_ground(img, h), ground, atol=1e-9)
    # Midpoint of the near edge maps to the middle of the near side.
    np.testing.assert_allclose(to_ground([[400, 400]], h), [[2, 0]], atol=1e-9)


def test_homography_rejects_bad_input():
    with pytest.raises(ValueError):
        homography([[0, 0], [1, 0], [0, 1]], [[0, 0], [1, 0], [0, 1]])


def test_to_ground_empty():
    assert to_ground(np.empty((0, 2)), np.eye(3)).shape == (0, 2)


def test_violating_pairs_is_strict_and_ordered():
    pts = [[0, 0], [1, 0], [3, 0], [3, 1.5]]
    assert violating_pairs(pts, 1.5).tolist() == [[0, 1]]
    assert violating_pairs(pts, 1.51).tolist() == [[0, 1], [2, 3]]
    assert violating_pairs([[0, 0]], 1.0).shape == (0, 2)


def test_homography_rejects_collinear_points():
    line = [[0, 0], [1, 0], [2, 0], [3, 0]]
    with pytest.raises(ValueError, match="collinear"):
        homography(line, line)


def test_reference_points_keep_half_pixels():
    odd = np.array([[0, 0, 3, 5]])
    assert centroids(odd).tolist() == [[1.5, 2.5]]
    assert feet(odd).tolist() == [[1.5, 5.0]]


def test_homography_needs_matching_counts_and_accepts_more_than_four():
    img = [[0, 0], [400, 0], [400, 400], [0, 400], [200, 200]]
    ground = [[0, 0], [4, 0], [4, 4], [0, 4], [2, 2]]
    with pytest.raises(ValueError, match="matching"):
        homography(img, ground[:4])
    with pytest.raises(ValueError, match="matching"):
        homography(img[:4], ground)
    np.testing.assert_allclose(to_ground(img, homography(img, ground)), ground, atol=1e-9)
