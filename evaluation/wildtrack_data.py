"""WILDTRACK inputs: calibration, annotations, cached detections and the distance methods.

Everything the evaluation in :mod:`evaluation.wildtrack` consumes: the dataset on disk, the
(camera, frame) views it scores, the five ways of turning boxes into points (M0 to M4) and
the detector cache for the end-to-end phase.
"""

import json
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist

from sda.detect import MODELS_DIR, Detector, load_detector
from sda.geometry import FloatArray, IntArray, centroids, feet, homography, to_ground

ROOT = MODELS_DIR.parent / "Wildtrack"
DETS_DIR = Path(__file__).resolve().parent.parent / "results" / "wildtrack_dets"
CAMERAS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
CALIB_NAMES = dict(
    zip(
        CAMERAS,
        ["CVLab1", "CVLab2", "CVLab3", "CVLab4", "IDIAP1", "IDIAP2", "IDIAP3"],
        strict=True,
    )
)
IMAGE_SIZE = (1920, 1080)
GRID_ORIGIN, GRID_STEP, GRID_WIDTH = (-3.0, -9.0), 0.025, 480
THRESHOLDS_M = (1.0, 1.5, 2.0)
MIN_VISIBLE_AREA = 0.5
RULE_WIDTH_PX, RULE_THRESHOLD_PX = 700, 50.0
TUNE_CANDIDATES_PX = np.arange(1.0, 2001.0)
GROUND_X, GROUND_Y, CLICK_GRID_M, CLICK_MARGIN_PX = (-3.0, 9.0), (-9.0, 27.0), 0.5, 50
MATCH_IOU = 0.5

BoolArray = NDArray[np.bool_]
Points = Callable[[str, FloatArray], FloatArray]


@dataclass(frozen=True)
class Calibration:
    """Pinhole camera in the WILDTRACK world frame, metres, ground plane ``z = 0``."""

    k: FloatArray
    r: FloatArray
    t: FloatArray

    @property
    def centre(self) -> FloatArray:
        """Camera centre ``x, y`` on the ground plane."""
        return np.asarray(-self.r.T @ self.t)[:2]

    def project(self, ground: ArrayLike) -> FloatArray:
        """Project ground points (metres) to pixels."""
        g = np.asarray(ground, dtype=np.float64).reshape(-1, 2)
        xyz = np.column_stack((g, np.zeros(len(g))))
        px, _ = cv2.projectPoints(xyz, cv2.Rodrigues(self.r)[0], self.t, self.k, None)
        return np.asarray(px, dtype=np.float64).reshape(-1, 2)

    def in_view(self, ground: ArrayLike, margin: float) -> BoolArray:
        """Return which ground points are in front of the camera and inside the image."""
        g = np.asarray(ground, dtype=np.float64).reshape(-1, 2)
        depth = (self.r[:, :2] @ g.T + self.t[:, None])[2]
        px = self.project(g)
        lo, hi = margin, np.array(IMAGE_SIZE) - margin
        return np.asarray((depth > 0) & (px >= lo).all(1) & (px <= hi).all(1))

    def ground_homography(self) -> FloatArray:
        """Pixel-to-ground homography straight from ``K, R, t`` (method M4)."""
        h = np.linalg.inv(self.k @ np.column_stack((self.r[:, :2], self.t)))
        return np.asarray(h / h[2, 2], dtype=np.float64)


@dataclass(frozen=True)
class FrameAnnotations:
    """Everyone annotated in one frame.

    Attributes:
        person_ids: (N,) ``personID``.
        world: (N, 2) ground position in metres decoded from ``positionID``.
        boxes: (7, N, 4) per-camera ``x1, y1, x2, y2`` boxes, -1 where not visible.
    """

    person_ids: NDArray[np.int64]
    world: FloatArray
    boxes: FloatArray


@dataclass(frozen=True)
class View:
    """The people visible in one (camera, frame): the unit of evaluation."""

    camera: str
    frame: int
    boxes: FloatArray
    world: FloatArray
    feet_in_frame: BoolArray | None = None  # GT box bottom inside the image (None: all)


@dataclass(frozen=True)
class Method:
    """A way of turning boxes into points plus the distance threshold to flag a pair.

    Attributes:
        name: Short method id (``M0`` ... ``M4``).
        points: ``(camera, boxes) -> (N, 2)`` reference points, pixels or ground metres.
        thresholds: ``T -> camera -> pixel threshold``; None means the threshold is T itself.
        metric: True when ``points`` are ground metres (distance errors are meaningful).
    """

    name: str
    points: Points
    thresholds: dict[float, dict[str, float]] | None = None
    metric: bool = False

    def threshold(self, camera: str, t: float) -> float:
        """Return the flagging threshold for distance limit ``t`` metres in this camera."""
        return t if self.thresholds is None else self.thresholds[t][camera]


def load_calibration(camera: str) -> Calibration:
    """Load ``K`` (undistorted intrinsics) and ``R, t`` for camera ``C1`` ... ``C7``.

    The extrinsic ``tvec`` is stored in centimetres and converted to metres.
    """
    name = CALIB_NAMES[camera]
    fs = cv2.FileStorage(
        str(ROOT / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml"), cv2.FILE_STORAGE_READ
    )
    k = np.asarray(fs.getNode("camera_matrix").mat(), dtype=np.float64)
    fs.release()
    root = ET.parse(ROOT / "calibrations" / "extrinsic" / f"extr_{name}.xml").getroot()
    rvec, tvec = (
        np.array(str(root.findtext(tag)).split(), np.float64) for tag in ("rvec", "tvec")
    )
    return Calibration(k, np.asarray(cv2.Rodrigues(rvec)[0], dtype=np.float64), tvec / 100.0)


def world_position(position_id: int) -> tuple[float, float]:
    """Decode a ``positionID`` into ground ``x, y`` metres (480-wide grid, 2.5 cm spacing)."""
    x0, y0 = GRID_ORIGIN
    return x0 + GRID_STEP * (position_id % GRID_WIDTH), y0 + GRID_STEP * (
        position_id // GRID_WIDTH
    )


def load_frame_annotations(frame: int) -> FrameAnnotations:
    """Load ``annotations_positions/<frame>.json``."""
    ann = json.loads((ROOT / "annotations_positions" / f"{frame:08d}.json").read_text())
    boxes = np.full((len(CAMERAS), len(ann), 4), -1.0)
    for i, a in enumerate(ann):
        for v in a["views"]:
            boxes[v["viewNum"], i] = [v["xmin"], v["ymin"], v["xmax"], v["ymax"]]
    world = np.array([world_position(a["positionID"]) for a in ann], np.float64).reshape(-1, 2)
    return FrameAnnotations(np.array([a["personID"] for a in ann], np.int64), world, boxes)


def _area(boxes: FloatArray) -> FloatArray:
    return np.asarray((boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0))


def visible(ann: FrameAnnotations, camera: str, frame: int) -> View:
    """Select the people visible in one camera: boxes clipped to the image, >= 50% inside."""
    raw = ann.boxes[CAMERAS.index(camera)]
    clipped = np.clip(raw, 0, np.array(IMAGE_SIZE * 2, np.float64))
    present = ~(raw == -1).all(1)  # all -1 is the "not visible" sentinel; xmin < 0 is not
    keep = present & (_area(clipped) >= MIN_VISIBLE_AREA * _area(raw))
    feet_in = raw[:, 3] <= IMAGE_SIZE[1]  # a box cut by the bottom edge has no foot point
    return View(camera, frame, clipped[keep], ann.world[keep], feet_in[keep])


def load_views(cameras: Iterable[str], frames: Iterable[int]) -> list[View]:
    """Load the (camera, frame) evaluation units for every frame and camera."""
    cams = list(cameras)
    return [
        visible(ann, c, f) for f in frames for ann in [load_frame_annotations(f)] for c in cams
    ]


def reprojection_errors(calib: Calibration, view: View) -> FloatArray:
    """Pixel distance between projected GT ground positions and GT box bottom-centres."""
    return np.asarray(np.linalg.norm(calib.project(view.world) - feet(view.boxes), axis=1))


def click_points(calib: Calibration) -> FloatArray:
    """Ground corners (4, 2) of the largest axis-aligned grid rectangle fully in view.

    Mimics a user clicking four floor marks once per camera: candidate corners lie on a
    0.5 m grid of the WILDTRACK ground area and must project inside the image with a margin.
    """
    xs = np.arange(GROUND_X[0], GROUND_X[1] + 1e-9, CLICK_GRID_M)
    ys = np.arange(GROUND_Y[0], GROUND_Y[1] + 1e-9, CLICK_GRID_M)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    ok = calib.in_view(np.column_stack((gx.ravel(), gy.ravel())), CLICK_MARGIN_PX).reshape(
        gx.shape
    )
    valid = (
        ok[:, None, :, None] & ok[None, :, :, None] & ok[:, None, None, :] & ok[None, :, None, :]
    )
    area = (xs[None, :] - xs[:, None])[:, :, None, None] * (ys[None, :] - ys[:, None])
    area = np.where(valid & (area > 0), area, -1.0)
    a, b, c, d = np.unravel_index(area.argmax(), area.shape)
    if area[a, b, c, d] <= 0:
        raise ValueError("no grid rectangle is fully in view")
    return np.array([[xs[a], ys[c]], [xs[b], ys[c]], [xs[b], ys[d]], [xs[a], ys[d]]])


def click_homography(calib: Calibration, sigma: float, seed: int, stream: int = 0) -> FloatArray:
    """Fit the 4-click homography with Gaussian click noise of ``sigma`` px (method M3).

    ``stream`` separates the cameras of one seed so their clicks are noised independently.
    """
    ground = click_points(calib)
    noise = np.random.default_rng([seed, stream]).normal(0, sigma, (4, 2))
    return homography(calib.project(ground) + noise, ground)


def _ground(hs: dict[str, FloatArray]) -> Points:
    return lambda camera, boxes: to_ground(feet(boxes), hs[camera])


def click_method(calibs: dict[str, Calibration], sigma: float, seed: int) -> Method:
    """Build M3 for one noise level and seed."""
    hs = {c: click_homography(k, sigma, seed, i) for i, (c, k) in enumerate(calibs.items())}
    return Method("M3", _ground(hs), metric=True)


def tune_threshold(pixel: FloatArray, true: FloatArray, t: float) -> float:
    """Pixel threshold maximising pair F1 for ``true < t`` over integer candidates."""
    positives = np.sort(pixel[true < t])
    predicted = np.searchsorted(np.sort(pixel), TUNE_CANDIDATES_PX)
    tp = np.searchsorted(positives, TUNE_CANDIDATES_PX)
    f1 = 2 * tp / np.maximum(predicted + len(positives), 1)  # 2tp / (2tp + fp + fn)
    return float(TUNE_CANDIDATES_PX[f1.argmax()])


def tune(
    views: list[View], reference: Callable[[FloatArray], FloatArray]
) -> dict[float, dict[str, float]]:
    """Per-camera pixel thresholds for each T, chosen on the given (tune) views."""
    pixel: dict[str, list[FloatArray]] = {}
    true: dict[str, list[FloatArray]] = {}
    for v in views:
        if len(v.world) > 1:
            pixel.setdefault(v.camera, []).append(pdist(reference(v.boxes)))
            true.setdefault(v.camera, []).append(pdist(v.world))
    return {
        t: {c: tune_threshold(np.concatenate(pixel[c]), np.concatenate(true[c]), t) for c in pixel}
        for t in THRESHOLDS_M
    }


def base_methods(
    cameras: list[str], calibs: dict[str, Calibration], tune_views: list[View]
) -> dict[str, Method]:
    """Build M0, M1, M2 and M4 (M3 is per noise level and seed, see :func:`click_method`)."""
    scale = RULE_WIDTH_PX / IMAGE_SIZE[0]
    fixed = {t: dict.fromkeys(cameras, RULE_THRESHOLD_PX) for t in THRESHOLDS_M}
    return {
        "M0": Method("M0", lambda _, b: centroids(b) * scale, fixed),
        "M1": Method("M1", lambda _, b: centroids(b), tune(tune_views, centroids)),
        "M2": Method("M2", lambda _, b: feet(b), tune(tune_views, feet)),
        "M4": Method(
            "M4", _ground({c: k.ground_homography() for c, k in calibs.items()}), metric=True
        ),
    }


class CachedDetector:
    """Run a detector on WILDTRACK frames, caching boxes as JSON so reruns are free."""

    def __init__(self, name: str, conf: float, device: str) -> None:
        """Remember the configuration; the model loads on the first cache miss.

        The cache directory carries a ``_key.json`` with the detector name and confidence;
        a mismatch raises instead of silently reusing boxes from another setting.
        """
        self.name, self._conf, self._device = name, conf, device
        self._detector: Detector | None = None
        self._dir = DETS_DIR / name
        key_path = self._dir / "_key.json"
        key = json.dumps({"detector": name, "conf": conf})
        if key_path.exists():
            if key_path.read_text() != key:
                raise RuntimeError(
                    f"{self._dir} was cached with {key_path.read_text()}, not {key}: delete it"
                )
        elif self._dir.exists():
            raise RuntimeError(f"{self._dir} has no _key.json: delete it to rebuild the cache")
        else:
            self._dir.mkdir(parents=True)
            key_path.write_text(key)

    def __call__(self, camera: str, frame: int) -> FloatArray:
        """Return (N, 4) boxes for one frame."""
        path = self._dir / camera / f"{frame:08d}.json"
        if not path.exists():
            if self._detector is None:
                self._detector = load_detector(self.name, self._conf, self._device)
            img = cv2.imread(str(ROOT / "Image_subsets" / camera / f"{frame:08d}.png"))
            boxes = self._detector(np.asarray(img, np.uint8)).boxes
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(np.asarray(boxes).tolist()))
        return np.array(json.loads(path.read_text()), np.float64).reshape(-1, 4)


def box_iou(a: FloatArray, b: FloatArray) -> FloatArray:
    """Pairwise IoU (A, B) of two ``x1y1x2y2`` box sets."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(0).prod(-1)
    union = _area(a)[:, None] + _area(b)[None, :] - inter
    return np.asarray(inter / np.maximum(union, 1e-9))  # two empty boxes: 0, not NaN


def match(dets: FloatArray, gts: FloatArray) -> IntArray:
    """Hungarian match on IoU; returns the GT index per detection, -1 if unmatched."""
    out = np.full(len(dets), -1, np.intp)
    if len(dets) and len(gts):
        iou = box_iou(dets, gts)
        r, c = linear_sum_assignment(-iou)
        ok = iou[r, c] >= MATCH_IOU
        out[r[ok]] = c[ok]
    return out


def in_roi(calib: Calibration, boxes: FloatArray) -> BoolArray:
    """Boxes whose foot point (true calibration) lies on the annotated ground area."""
    g = to_ground(feet(boxes), calib.ground_homography())
    x_ok = (g[:, 0] >= GROUND_X[0]) & (g[:, 0] <= GROUND_X[1])
    return np.asarray(x_ok & (g[:, 1] >= GROUND_Y[0]) & (g[:, 1] <= GROUND_Y[1]))
