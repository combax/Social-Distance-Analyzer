"""Person detectors: the Darknet YOLOv3 baseline and current Ultralytics models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "data" / "models"


def rel(path: Path | str) -> str:
    """Repo-relative POSIX form of ``path`` for records that get committed (no machine paths)."""
    p = Path(path).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else p.as_posix()


PERSON = 0  # COCO class index
MAX_DET = 1000  # boxes kept per frame; Ultralytics' default of 300 truncates crowded frames


@dataclass(frozen=True, slots=True)
class Detections:
    """People found in one frame.

    Attributes:
        boxes: (N, 4) float64 ``x1, y1, x2, y2`` in the input frame's pixels.
        scores: (N,) float64 confidences.
        ids: (N,) int64 track ids, or None when the detector is not tracking.
    """

    boxes: NDArray[np.float64]
    scores: NDArray[np.float64]
    ids: NDArray[np.int64] | None = None

    def __len__(self) -> int:
        """Return the number of detections."""
        return len(self.boxes)


class Detector(Protocol):
    """Anything that turns a BGR frame into person detections."""

    name: str

    def __call__(self, frame: NDArray[np.uint8]) -> Detections:
        """Detect people in one BGR frame."""
        ...


class DarknetYolo:
    """YOLOv3 through ``cv2.dnn``, the pixel-rule baseline's detector.

    Its conventions: 416x416 squashed input, the person column of the output
    (objectness x class probability, as ``cv2.dnn`` computes it) as the confidence,
    ``cv2.dnn.NMSBoxes`` with the same threshold for score and overlap filtering. Fixes:
    ``getUnconnectedOutLayersNames`` replaces the index arithmetic that broke in OpenCV 4.5.4,
    and for ``conf`` below 0.2 the YOLO layers' ``thresh`` is lowered to match, because
    ``cv2.dnn`` otherwise zeroes every score under 0.2 (which truncates AP curves).
    """

    def __init__(
        self,
        cfg: Path = MODELS_DIR / "yolov3.cfg",
        weights: Path = MODELS_DIR / "yolov3.weights",
        conf: float = 0.3,
        nms: float = 0.3,
        size: int = 416,
    ) -> None:
        """Load the network.

        Args:
            cfg: Darknet ``.cfg`` path.
            weights: Darknet ``.weights`` path.
            conf: Minimum person class score.
            nms: Non-maximum suppression IoU threshold.
            size: Square network input size.
        """
        self.name = "yolov3-darknet"
        text = with_thresh(Path(cfg).read_text(), conf)
        self._net = cv2.dnn.readNetFromDarknet(
            np.frombuffer(text.encode(), np.uint8), np.fromfile(weights, np.uint8)
        )
        self._layers = self._net.getUnconnectedOutLayersNames()
        self._conf, self._nms, self._size = conf, nms, size

    def __call__(self, frame: NDArray[np.uint8]) -> Detections:
        """Detect people in one BGR frame."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (self._size, self._size), swapRB=True)
        self._net.setInput(blob)
        out = np.concatenate(self._net.forward(self._layers))
        scores = out[:, 5:]
        keep = (scores.argmax(1) == PERSON) & (scores[:, PERSON] > self._conf)
        cx, cy, bw, bh = (out[keep, :4] * np.array([w, h, w, h])).T
        conf = scores[keep, PERSON].astype(np.float64)
        xywh = np.column_stack((cx - bw / 2, cy - bh / 2, bw, bh))
        idx = np.asarray(
            cv2.dnn.NMSBoxes(xywh.tolist(), conf.tolist(), self._conf, self._nms), dtype=int
        ).reshape(-1)
        boxes = xywh[idx].copy()
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)  # inside the frame, as Ultralytics does
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return Detections(boxes.astype(np.float64), conf[idx])


def with_thresh(cfg_text: str, conf: float) -> str:
    """Lower the YOLO layers' score floor when ``conf`` is below cv2.dnn's default of 0.2.

    ``cv2.dnn`` zeroes every class score below the ``[yolo]`` block's ``thresh`` (0.2 unless
    set), which would truncate a precision-recall curve evaluated at lower confidences.
    """
    if conf >= 0.2:
        return cfg_text
    return cfg_text.replace("[yolo]", f"[yolo]\nthresh={conf}")


class UltralyticsYolo:
    """Any Ultralytics detector (YOLO11, YOLO26, ...) restricted to the person class."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        imgsz: int = 640,
        device: str | None = None,
        max_det: int = MAX_DET,
    ) -> None:
        """Load the model, downloading known weights on first use.

        Args:
            weights: Path or Ultralytics asset name such as ``yolo26n.pt``.
            conf: Minimum confidence.
            imgsz: Inference size (long side).
            device: ``"cpu"``, ``"0"``, or None for the best available.
            max_det: Boxes kept per frame after NMS. Ultralytics' default of 300 silently
                truncates crowded frames at low confidence, which clips the AP recall tail.
        """
        from ultralytics.models import YOLO  # deferred: importing torch costs seconds

        self.name = Path(weights).stem + ("" if imgsz == 640 else f"@{imgsz}")
        self._model = YOLO(str(weights))
        self._kw: dict[str, object] = {
            "verbose": False,
            "conf": conf,
            "imgsz": imgsz,
            "device": device,
            "classes": [PERSON],
            "max_det": max_det,
        }

    def __call__(self, frame: NDArray[np.uint8]) -> Detections:
        """Detect people in one BGR frame."""
        return _to_detections(next(iter(self._model.predict(frame, stream=False, **self._kw))))

    def track(self, frame: NDArray[np.uint8], tracker: str = "bytetrack.yaml") -> Detections:
        """Detect and track people; call on consecutive frames of one video.

        Args:
            frame: BGR frame.
            tracker: Ultralytics tracker config, ``bytetrack.yaml`` or ``botsort.yaml``.

        Returns:
            Detections with ``ids`` set (unconfirmed tracks are dropped by the tracker).
        """
        res = self._model.track(frame, stream=False, persist=True, tracker=tracker, **self._kw)
        return _to_detections(next(iter(res)))

    def reset_tracker(self) -> None:
        """Forget all tracks; call between videos."""
        predictor = self._model.predictor
        for t in getattr(predictor, "trackers", None) or []:
            t.reset()


def _to_detections(result: object) -> Detections:
    b = result.boxes  # type: ignore[attr-defined]  # ultralytics Results is untyped
    ids = None if b.id is None else b.id.cpu().numpy().astype(np.int64)
    return Detections(
        b.xyxy.cpu().numpy().astype(np.float64), b.conf.cpu().numpy().astype(np.float64), ids
    )


def load_detector(name: str, conf: float = 0.25, device: str | None = None) -> Detector:
    """Build a detector by name.

    Args:
        name: ``yolov3-darknet`` for the Darknet baseline, otherwise an Ultralytics model name
            (``yolo26n``) or a path to ``.pt`` weights, optionally suffixed with the inference
            size (``yolo26s@1280``; default 640).
        conf: Minimum confidence.
        device: Ultralytics device string; ignored by the Darknet baseline (CPU only).

    Returns:
        A callable detector.
    """
    if name == "yolov3-darknet":
        return DarknetYolo(conf=conf, nms=0.3)
    stem, _, size = name.rpartition("@")  # a path may itself contain "@"
    if not size.isdigit():
        stem, size = name, ""
    path = Path(stem)
    weights = path if path.suffix == ".pt" else MODELS_DIR / f"{stem}.pt"
    return UltralyticsYolo(weights, conf=conf, imgsz=int(size or 640), device=device)
