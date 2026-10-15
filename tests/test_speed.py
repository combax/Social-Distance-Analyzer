import json
import platform

import cv2
import numpy as np
import pytest
from conftest import StubDetector

from evaluation import speed
from evaluation.speed import FRAMES_DIR, main


@pytest.mark.skipif(not FRAMES_DIR.exists(), reason="MOT17 not downloaded")
def test_speed_json_shape(tmp_path, yolo26n_cpu):
    out = tmp_path / "s.json"
    args = ["--models", "yolo26n", "--devices", "cpu", "--frames", "3", "--warmup", "1"]
    assert main([*args, "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert set(report["machine"]) == {"cpu", "gpu", "torch", "ultralytics", "opencv"}
    (r,) = report["results"]
    assert set(r) == {"model", "device", "frames", "median_ms", "p90_ms", "fps"}
    assert (r["model"], r["device"], r["frames"]) == ("yolo26n", "cpu", 3)
    assert 0 < r["median_ms"] <= r["p90_ms"] and 0 < r["fps"] < 10_000


def test_cpu_name_falls_back_to_platform_processor(monkeypatch):
    monkeypatch.setattr(speed.sys, "platform", "linux")
    assert speed._cpu_name() == platform.processor()


def test_main_fails_without_frames(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(speed, "FRAMES_DIR", tmp_path)
    assert main(["--out", str(tmp_path / "s.json")]) == 2
    assert "is MOT17 downloaded" in capsys.readouterr().err


def test_main_times_a_stub_and_skips_unavailable_devices(tmp_path, monkeypatch, capsys):
    frames = tmp_path / "img1"
    frames.mkdir()
    for n in range(4):
        assert cv2.imwrite(str(frames / f"{n:06d}.jpg"), np.zeros((48, 64, 3), np.uint8))
    monkeypatch.setattr(speed, "FRAMES_DIR", frames)
    monkeypatch.setattr(speed, "load_detector", lambda name, device: StubDetector(name))
    monkeypatch.setattr(speed.torch.cuda, "is_available", lambda: False)
    out = tmp_path / "s.json"
    args = ["--models", "yolov3-darknet", "stub", "--devices", "cpu", "0", "--frames", "3"]
    assert main([*args, "--warmup", "1", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    timed = [(r["model"], r["device"], r["frames"]) for r in report["results"]]
    assert timed == [("yolov3-darknet", "cpu", 3), ("stub", "cpu", 3)]
    reasons = [(s["model"], s["device"], s["reason"]) for s in report["skipped"]]
    assert reasons == [
        ("yolov3-darknet", "0", "cv2.dnn baseline is CPU only"),
        ("stub", "0", "torch.cuda is unavailable"),
    ]
    assert report["machine"]["gpu"] is None and report["machine"]["cpu"]
    text = capsys.readouterr().out
    assert "stub             0      skipped: torch.cuda is unavailable" in text and "wrote" in text
