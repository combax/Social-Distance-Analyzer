"""Shared fixtures."""

import cv2
import numpy as np
import pytest

from evaluation import mot17
from sda.detect import MODELS_DIR, Detections, load_detector

# Two tiny sequences: 64x48 frames, two pedestrians and one ignore-class box per frame.
SEQUENCES = {"MOT17-02-FRCNN": 3, "MOT17-04-FRCNN": 4}
WIDTH, HEIGHT = 64, 48
GT_BOXES = (  # id, x, y, w, h, flag, class, visibility (MOT 1-based x, y)
    (1, 5, 5, 10, 20, 1, 1, 1.0),
    (2, 30, 10, 12, 24, 1, 1, 0.6),
    (3, 50, 20, 8, 16, 0, 7, 1.0),
)


def gt_frame_rows(frame):
    return [[frame, *box] for box in GT_BOXES]


def write_sequence(root, name, length):
    d = root / name
    (d / "img1").mkdir(parents=True)
    (d / "gt").mkdir()
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}\nimDir=img1\nframeRate=30\nseqLength={length}\n"
        f"imWidth={WIDTH}\nimHeight={HEIGHT}\nimExt=.jpg\n"
    )
    rng = np.random.default_rng(0)
    for n in range(1, length + 1):
        img = rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8)
        assert cv2.imwrite(str(d / "img1" / f"{n:06d}.jpg"), img)
    rows = [r for n in range(1, length + 1) for r in gt_frame_rows(n)]
    (d / "gt" / "gt.txt").write_text("".join(",".join(str(v) for v in r) + "\n" for r in rows))


@pytest.fixture
def mot17_root(tmp_path, monkeypatch):
    """A temporary MOT17 train folder with two tiny sequences; ``mot17.ROOT`` points at it."""
    root = tmp_path / "MOT17" / "train"
    for name, length in SEQUENCES.items():
        write_sequence(root, name, length)
    monkeypatch.setattr(mot17, "ROOT", root)
    return root


class StubDetector:
    """A ``Detector``: both GT pedestrians (0-based x1, y1, x2, y2) plus one false box."""

    def __init__(self, name="stub", scores=(0.9, 0.2), extra=True):
        self.name = name
        self.calls = 0
        boxes = [[4.0, 4.0, 14.0, 24.0], [29.0, 9.0, 41.0, 33.0]]
        scores = list(scores)
        if extra:
            boxes.append([0.0, 30.0, 6.0, 46.0])
            scores.append(0.8)
        self.boxes = np.array(boxes, dtype=np.float64)
        self.scores = np.array(scores, dtype=np.float64)

    def __call__(self, frame):
        self.calls += 1
        return Detections(self.boxes, self.scores)


class StubTracker(StubDetector):
    """An ``UltralyticsYolo`` stand-in: fixed tracks, an empty frame every third call."""

    def __init__(self, weights="stub.pt", conf=0.1, imgsz=640, device=None, **_):
        super().__init__(name=f"{weights}-trk", extra=False)
        self.resets = 0

    def reset_tracker(self):
        self.resets += 1

    def track(self, frame, tracker="bytetrack.yaml"):
        self.calls += 1
        if self.calls % 3 == 0:
            return Detections(np.empty((0, 4)), np.empty(0), None)
        return Detections(self.boxes, self.scores, np.array([1, 2], dtype=np.int64))


@pytest.fixture(scope="session")
def yolo26n_cpu():
    """A CPU yolo26n detector; skips when the weights are absent and cannot be downloaded."""
    try:
        return load_detector("yolo26n", conf=0.3, device="cpu")
    except Exception as e:  # noqa: BLE001  # Ultralytics raises various download errors
        if (MODELS_DIR / "yolo26n.pt").exists():
            raise
        pytest.skip(f"yolo26n weights unavailable offline: {e}")
