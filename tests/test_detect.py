from pathlib import Path

import numpy as np
import pytest

from sda.detect import MODELS_DIR, load_detector, with_thresh

needs_darknet = pytest.mark.skipif(
    not (MODELS_DIR / "yolov3.weights").exists(), reason="download the Darknet weights"
)


@pytest.mark.parametrize("name", [pytest.param("yolov3-darknet", marks=needs_darknet), "yolo26n"])
def test_detector_returns_empty_on_blank_frame(name, yolo26n_cpu):
    det = yolo26n_cpu if name == "yolo26n" else load_detector(name, conf=0.3, device="cpu")
    out = det(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out.boxes.shape == (0, 4) and len(out.scores) == 0


@needs_darknet
def test_darknet_boxes_stay_inside_the_frame():
    det = load_detector("yolov3-darknet", conf=0.05, device="cpu")
    frame = np.full((240, 320, 3), 128, dtype=np.uint8)
    frame[150:, 40:120] = (30, 60, 200)  # a blob touching the bottom edge
    out = det(frame)
    assert (out.boxes[:, [0, 2]] <= 320).all() and (out.boxes[:, [1, 3]] <= 240).all()
    assert (out.boxes >= 0).all()


def test_darknet_thresh_rewrite_only_below_cv2_floor():
    cfg = "[net]\nwidth=416\n[yolo]\nmask=0\n[yolo]\nmask=1\n"
    assert with_thresh(cfg, 0.3) == cfg and with_thresh(cfg, 0.2) == cfg
    low = with_thresh(cfg, 0.05)
    assert low.count("[yolo]\nthresh=0.05") == 2 and low.count("thresh=") == 2


def test_load_detector_parses_pt_paths_containing_at(monkeypatch):
    import sda.detect as sd

    calls = []

    class Fake:
        def __init__(self, weights, conf, imgsz, device):
            calls.append((str(weights), imgsz))

    monkeypatch.setattr(sd, "UltralyticsYolo", Fake)
    load_detector("runs/x@y/best.pt@1280", device="cpu")
    load_detector("runs/x@y/best.pt", device="cpu")
    load_detector("yolo26s@1280", device="cpu")
    assert calls[0] == (str(Path("runs/x@y/best.pt")), 1280)
    assert calls[1] == (str(Path("runs/x@y/best.pt")), 640)
    assert calls[2] == (str(MODELS_DIR / "yolo26s.pt"), 1280)


def test_imgsz_suffix_sets_size_and_name(yolo26n_cpu):
    det = load_detector("yolo26n@1280", device="cpu")
    assert det.name == "yolo26n@1280" and det._kw["imgsz"] == 1280
    assert det._kw["max_det"] == 1000  # Ultralytics' default of 300 would clip crowded frames


def test_rel_is_repo_relative_only_inside_the_repo(tmp_path):
    from sda.detect import REPO, rel

    assert rel(REPO / "results" / "x.json") == "results/x.json"
    outside = tmp_path / "x.pt"
    assert rel(outside) == outside.resolve().as_posix()


def test_track_then_reset_tracker(yolo26n_cpu):
    det = load_detector("yolo26n", conf=0.3, device="cpu")
    det.reset_tracker()  # nothing to reset before the first frame
    out = det.track(np.zeros((480, 640, 3), dtype=np.uint8))
    assert out.boxes.shape == (0, 4) and out.ids is None
    det.reset_tracker()
