"""Ground-plane geometry: reference points, image-to-ground homography, distance violations."""

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import pdist

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]


def centroids(boxes: ArrayLike) -> FloatArray:
    """Return box centres, the reference point of the pixel rule.

    Args:
        boxes: (N, 4) array of ``x1, y1, x2, y2`` pixel boxes.

    Returns:
        (N, 2) array of ``x, y`` centres.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack(((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2))


def feet(boxes: ArrayLike) -> FloatArray:
    """Return bottom-centre points, where a standing person touches the ground.

    Args:
        boxes: (N, 4) array of ``x1, y1, x2, y2`` pixel boxes.

    Returns:
        (N, 2) array of ``x, y`` foot points.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.column_stack(((b[:, 0] + b[:, 2]) / 2, b[:, 3]))


def homography(image_points: ArrayLike, ground_points: ArrayLike) -> FloatArray:
    """Fit the image-to-ground-plane homography from point correspondences.

    Four points (e.g. the corners of a floor tile pattern clicked once per camera) give an
    exact fit; more points give a least-squares fit.

    Args:
        image_points: (K, 2) pixel coordinates, K >= 4, no three collinear.
        ground_points: (K, 2) matching ground-plane coordinates in metres.

    Returns:
        (3, 3) homography mapping pixels to metres.

    Raises:
        ValueError: If fewer than 4 correspondences are given or the fit is degenerate.
    """
    src = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(ground_points, dtype=np.float64).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError(f"need >= 4 matching points, got {len(src)} and {len(dst)}")
    h, _ = cv2.findHomography(src, dst, method=0)
    if h is None:
        raise ValueError("degenerate correspondences: three or more points are collinear")
    return np.asarray(h, dtype=np.float64)


def to_ground(points: ArrayLike, h: ArrayLike) -> FloatArray:
    """Project pixel points onto the ground plane.

    Args:
        points: (N, 2) pixel coordinates.
        h: (3, 3) homography from :func:`homography`.

    Returns:
        (N, 2) ground-plane coordinates in the homography's units.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.empty((0, 2))
    out = cv2.perspectiveTransform(p, np.asarray(h, dtype=np.float64))
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def violating_pairs(points: ArrayLike, threshold: float) -> IntArray:
    """Return index pairs closer than ``threshold`` (strictly), Euclidean distance.

    Args:
        points: (N, 2) coordinates, pixels or metres.
        threshold: Minimum allowed distance in the same units.

    Returns:
        (K, 2) array of ``i < j`` index pairs, in row-major order.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    i, j = np.triu_indices(len(p), k=1)
    close = pdist(p) < threshold  # pdist order matches triu_indices(k=1)
    return np.column_stack((i[close], j[close])).astype(np.intp)
