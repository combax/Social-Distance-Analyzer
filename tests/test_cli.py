import json
import runpy
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import sda.__main__ as cli
from sda.__main__ import main
from sda.detect import MODELS_DIR, Detections, UltralyticsYolo
from sda.geometry import to_ground

needs_darknet = pytest.mark.skipif(
    not (MODELS_DIR / "yolov3.weights").exists(), reason="download the Darknet weights"
)
GROUND = "0,0;4,0;4,10;0,10"
IMAGE = "50,200;270,200;300,50;20,50"
# Centroids (35, 110), (65, 120) and (275, 110): the first pair is 31.6 px apart, the third box
# is far from both; the feet of the first two land about 0.5 m apart on the calibrated ground.
BOXES = [[10, 20, 60, 200], [40, 30, 90, 210], [250, 20, 300, 200]]


def write_video(path, n=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), 10, (320, 240))
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()
    return str(path)


def count_frames(path):
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    return n, cap.get(cv2.CAP_PROP_FPS)


class Fixed(UltralyticsYolo):
    """Returns the same boxes on every frame, with track ids from ``track``."""

    name = "fixed"

    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, np.float64)

    def __call__(self, frame):
        return Detections(self.boxes, np.ones(len(self.boxes)))

    def track(self, frame, tracker="bytetrack.yaml"):
        return Detections(self.boxes, np.ones(len(self.boxes)), np.arange(len(self.boxes)))


@pytest.fixture
def three_people(monkeypatch):
    """Replace the detector with three fixed boxes."""
    monkeypatch.setattr(cli, "load_detector", lambda *a, **k: Fixed(BOXES))


fixed_people = pytest.mark.usefixtures("three_people")


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "in.mp4")


def test_calibrate_then_run(tmp_path, video, yolo26n_cpu):
    calib = tmp_path / "calib.json"
    args = ["calibrate", video, "--ground", GROUND, "--image-points", IMAGE, "--out", str(calib)]
    assert main(args) == 0
    c = json.loads(calib.read_text())
    assert np.allclose(
        to_ground(c["image_points"], c["homography"]), c["ground_points"], atol=1e-6
    )

    out = tmp_path / "out.mp4"
    args = ["run", video, "--model", "yolo26n", "--device", "cpu", "--calib", str(calib)]
    assert main([*args, "--out", str(out)]) == 0
    assert count_frames(out) == (10, 10)


@fixed_people
def test_pixel_rule_scaled_to_the_700_px_rule_width(tmp_path, video, capsys):
    out = tmp_path / "sub" / "out.mp4"
    assert main(["run", video, "--out", str(out), "--fps", "5"]) == 0
    assert "frames=10 people/frame=3.00 violators/frame=2.00" in capsys.readouterr().out
    assert count_frames(out) == (10, 5)

    assert main(["run", video, "--track", "--pixel-width", "700", "--max-frames", "3"]) == 0
    assert "frames=3 people/frame=3.00 violators/frame=0.00" in capsys.readouterr().out

    assert main(["run", video, "--max-frames", "0"]) == 0
    assert "frames=0 people/frame=0.00 violators/frame=0.00 fps=0.0" in capsys.readouterr().out


@fixed_people
def test_calibrated_run_measures_in_metres(tmp_path, video, capsys):
    calib = tmp_path / "calib.json"
    main(["calibrate", video, "--ground", GROUND, "--image-points", IMAGE, "--out", str(calib)])
    assert main(["run", video, "--calib", str(calib), "--threshold", "2"]) == 0
    assert "violators/frame=2.00" in capsys.readouterr().out
    assert main(["run", video, "--calib", str(calib), "--threshold", "0.1"]) == 0
    assert "violators/frame=0.00" in capsys.readouterr().out


@fixed_people
def test_image_sequence_defaults_to_30_fps(tmp_path):
    for i in range(4):
        cv2.imwrite(str(tmp_path / f"f_{i:03d}.png"), np.zeros((240, 320, 3), np.uint8))
    out = tmp_path / "out.mp4"
    assert main(["run", str(tmp_path / "f_%03d.png"), "--out", str(out)]) == 0
    assert count_frames(out) == (4, 30)


@fixed_people
def test_show_quits_on_q(video, monkeypatch, capsys):
    keys = iter([-1, ord("q")])
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: None)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: next(keys))
    assert main(["run", video, "--show"]) == 0
    assert "frames=2 " in capsys.readouterr().out


@fixed_people
def test_run_rejects_bad_inputs(tmp_path, video):
    with pytest.raises(SystemExit, match="cannot open video writer"):
        main(["run", video, "--out", str(tmp_path / "out.xyz")])
    with pytest.raises(SystemExit, match="cannot open video"):
        main(["run", str(tmp_path / "missing.mp4")])


@needs_darknet
def test_track_needs_an_ultralytics_model(video):
    with pytest.raises(SystemExit, match="--track needs an Ultralytics model"):
        main(["run", video, "--model", "yolov3-darknet", "--track"])


def test_calibrate_by_clicking(tmp_path, video, monkeypatch):
    clicks = iter([(50, 200), (270, 200), (300, 50), (20, 50)])
    window = {}

    def wait_key(ms):
        x, y = next(clicks)
        window["on_mouse"](cv2.EVENT_MOUSEMOVE, x, y, 0, None)
        window["on_mouse"](cv2.EVENT_LBUTTONDOWN, x, y, 0, None)
        return -1

    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, cb: window.update(on_mouse=cb))
    monkeypatch.setattr(cv2, "imshow", lambda name, frame: None)
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    monkeypatch.setattr(cv2, "destroyWindow", lambda name: None)
    out = tmp_path / "calib.json"
    assert main(["calibrate", video, "--ground", GROUND, "--out", str(out)]) == 0
    points = json.loads(out.read_text())["image_points"]
    assert points == [[50, 200], [270, 200], [300, 50], [20, 50]]

    monkeypatch.setattr(cv2, "waitKey", lambda ms: 27)
    with pytest.raises(SystemExit, match="calibration cancelled"):
        main(["calibrate", video, "--ground", GROUND, "--out", str(out)])


def test_calibrate_needs_a_readable_frame(monkeypatch):
    monkeypatch.setattr(cli, "_open", lambda video: SimpleNamespace(read=lambda: (False, None)))
    with pytest.raises(SystemExit, match="cannot read a frame"):
        main(["calibrate", "empty.mp4", "--ground", GROUND])


def test_module_entry_point(tmp_path, video, monkeypatch):
    out = tmp_path / "calib.json"
    argv = ["sda", "calibrate", video, "--ground", GROUND, "--image-points", IMAGE]
    monkeypatch.setattr(sys, "argv", [*argv, "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(cli.__file__, run_name="__main__")
    assert e.value.code == 0 and out.exists()
