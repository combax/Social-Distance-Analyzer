import json

import numpy as np
import pandas as pd
import pytest
from conftest import SEQUENCES, StubDetector, StubTracker, gt_frame_rows

from evaluation import detection, mot17, tracking
from sda.detect import Detections, UltralyticsYolo

SEQ = "MOT17-02-FRCNN"
needs_data = pytest.mark.skipif(not (mot17.ROOT / SEQ).exists(), reason="MOT17 not downloaded")


def _pedestrians(gt, frames):
    sel = gt[gt["frame"].isin(frames)]
    return sel[(sel["class"] == mot17.PEDESTRIAN) & (sel["flag"] == 1)]


@needs_data
def test_load_gt_has_named_columns_and_pedestrians():
    seq = mot17.load_sequence(SEQ)
    assert (seq.fps, seq.width, seq.height, seq.length) == (30, 1920, 1080, 600)
    gt = mot17.load_gt(seq)
    assert list(gt.columns) == list(mot17.GT_COLUMNS)
    ped = _pedestrians(gt, range(1, seq.length + 1))
    assert len(ped) > 0 and ped["frame"].between(1, seq.length).all()
    assert gt["visibility"].between(0, 1).all()
    assert mot17.frame_numbers(seq, stride=7, limit=3) == [1, 8, 15]


@needs_data
def test_detection_metrics_on_gt_as_predictions():
    seq = mot17.load_sequence(SEQ)
    gt = mot17.load_gt(seq)
    frames = [1, 2]
    ped = _pedestrians(gt, frames)
    n = len(ped)
    rows = np.column_stack(
        (
            ped["frame"],
            np.full(n, -1),
            ped[["x", "y", "w", "h"]],
            np.ones(n),
            np.full((n, 3), -1),
        )
    ).astype(float)
    # A confident detection exactly on an ignore-class box must not count as a false positive.
    ign = gt[(gt["frame"] == 1) & gt["class"].isin(mot17.IGNORE_CLASSES)].iloc[0]
    extra = [[1, -1, ign["x"], ign["y"], ign["w"], ign["h"], 0.9, -1, -1, -1]]
    rows = np.vstack((rows, np.array(extra, dtype=float)))

    m = detection.evaluate({SEQ: (gt, rows, frames)})
    for r in (m["overall"], m["per_sequence"][SEQ]):
        assert r["AP50"] == pytest.approx(1.0) and r["AP"] == pytest.approx(1.0)
        assert r["recall"] == 1.0 and r["precision"] == 1.0 and r["fp"] == 0
        assert r["tp"] == n and r["gt"] == n and r["frames"] == 2
        assert sum(r["gt_by_visibility"].values()) == n


@needs_data
def test_trackeval_on_gt_as_tracker(tmp_path):
    seq = mot17.load_sequence(SEQ)
    ped = _pedestrians(mot17.load_gt(seq), range(1, 21))
    n = len(ped)
    rows = np.column_stack(
        (ped[["frame", "id", "x", "y", "w", "h"]], np.ones(n), np.full((n, 3), -1))
    ).astype(float)
    mot17.write_mot(tmp_path / "gt-as-tracker" / "data" / f"{SEQ}.txt", rows)

    res = tracking.score(tmp_path, "gt-as-tracker", {SEQ: 20})
    assert set(res) == {SEQ, "COMBINED_SEQ"}
    for r in res.values():
        assert r["HOTA"] > 0.95 and r["MOTA"] > 0.95 and r["IDF1"] > 0.95
        assert r["IDSW"] == 0 and r["FP"] == 0 and r["FN"] == 0


def test_cache_valid_checks_every_key_and_write_mot_drops_the_old_sidecar(tmp_path):
    p = tmp_path / "a.txt"
    rows = np.array([[1, -1, 1, 1, 10, 20, 0.9, -1, -1, -1]], float)
    assert not mot17.cache_valid(p, {})
    mot17.write_mot(p, rows, {"limit": None, "imgsz": 640})
    assert mot17.cache_valid(p, {"limit": None}) and mot17.cache_valid(p, {"imgsz": 640})
    assert not mot17.cache_valid(p, {"imgsz": 1280})
    assert not mot17.cache_valid(p, {"weights_mtime_ns": None})  # key absent is not None
    mot17.write_mot(p, rows)  # no meta: the stale sidecar must not survive
    assert not p.with_suffix(".json").exists() and not mot17.cache_valid(p, {"limit": None})
    assert mot17.read_mot(p).shape == (1, 10)


@needs_data
def test_frames_with_explicit_numbers_yields_them_in_order():
    seq = mot17.load_sequence(SEQ)
    for numbers in ([], [7], [5, 1, 3, 9, 2, 8]):  # 0, 1 and more than the read-ahead depth
        got = [(n, img.shape[:2]) for n, img in mot17.frames(seq, numbers=numbers)]
        assert [n for n, _ in got] == numbers
        assert all(shape == (seq.height, seq.width) for _, shape in got)


@needs_data
def test_gt_boxes_are_clipped_to_the_frame():
    seq = mot17.load_sequence(SEQ)
    raw = pd.read_csv(seq.dir / "gt" / "gt.txt", header=None, names=mot17.GT_COLUMNS)
    gt = mot17.load_gt(seq)
    assert ((raw["x"] + raw["w"]) > seq.width + 1).any()  # MOT17 does annotate off-frame
    assert (gt["x"] >= 1).all() and (gt["y"] >= 1).all()
    assert ((gt["x"] + gt["w"]) <= seq.width + 1).all()
    assert ((gt["y"] + gt["h"]) <= seq.height + 1).all()
    assert (gt["w"] >= 0).all() and len(gt) == len(raw)


def test_to_mot_xywh_is_one_based():
    assert mot17.to_mot_xywh(np.array([[0.0, 0.0, 10.0, 20.0]])).tolist() == [[1, 1, 10, 20]]


@needs_data
def test_operating_point_threshold_iou_and_visibility_buckets():
    gt = mot17.load_gt(mot17.load_sequence(SEQ))
    frames = [1, 2]
    ped = _pedestrians(gt, frames).reset_index(drop=True)
    # Cycle: 0 = hit above the 0.3 threshold, 1 = hit below it, 2 = missed,
    # 3 = confident box displaced off-image (an FP that must not match anything).
    kind = np.arange(len(ped)) % 4
    xywh = ped[["x", "y", "w", "h"]].to_numpy(float)
    xywh[kind == 3, 0] += 5000
    score = np.select([kind == 0, kind == 1, kind == 3], [0.5, 0.2, 0.9], 0.0)
    keep = kind != 2
    n = int(keep.sum())
    rows = np.column_stack(
        (ped["frame"][keep], np.full(n, -1), xywh[keep], score[keep], np.full((n, 3), -1))
    ).astype(float)

    r = detection.evaluate({SEQ: (gt, rows, frames)})["overall"]
    hit = kind == 0
    assert (r["tp"], r["fp"], r["gt"]) == (hit.sum(), (kind == 3).sum(), len(ped))
    assert r["recall"] == pytest.approx(hit.mean())
    assert r["AP50"] < 1.0
    # The sweep at 0.30 must reproduce the operating point; 0.5-scored hits survive 0.5, not 0.55.
    assert r["f1_by_conf"]["0.30"] == pytest.approx(r["f1"])
    at_half = detection.evaluate({SEQ: (gt, rows, frames)}, conf=0.5)["overall"]
    assert at_half["f1"] == pytest.approx(r["f1_by_conf"]["0.50"]) and at_half["AP"] == r["AP"]
    assert r["f1_by_conf"]["0.50"] == pytest.approx(r["f1"]) and r["f1_by_conf"]["0.55"] < r["f1"]
    assert r["at_best_conf"]["f1"] == max(r["f1_by_conf"].values())
    buckets = np.digitize(ped["visibility"], detection.VIS_EDGES)
    for b, label in enumerate(detection.VIS_LABELS):
        in_b = buckets == b
        expected = hit[in_b].mean() if in_b.any() else None
        assert r["recall_by_visibility"][label] == pytest.approx(expected)


@pytest.mark.usefixtures("mot17_root")
def test_frames_raises_for_a_missing_image():
    seq = mot17.load_sequence("MOT17-02-FRCNN")
    (seq.dir / "img1" / "000002.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="frame 2"):
        list(mot17.frames(seq))


@pytest.mark.usefixtures("mot17_root")
def test_run_detector_caches_rows_and_sidecar(tmp_path):
    seq = mot17.load_sequence("MOT17-02-FRCNN")
    det = StubDetector()
    out = tmp_path / "preds" / "stub" / f"{seq.name}.txt"
    detection.run_detector(det, seq, 2, None, out, extra_meta={"imgsz": 640})
    rows = mot17.read_mot(out)
    assert det.calls == 2 and rows.shape == (6, 10)
    assert sorted(set(rows[:, 0])) == [1, 3]
    assert rows[0, 2:7].tolist() == [5, 5, 10, 20, 0.9]  # MOT 1-based x, y
    meta = mot17.read_meta(out)
    assert (meta["imgsz"], meta["stride"], meta["limit"], meta["frames"]) == (640, 2, None, 2)
    assert meta["seconds"] >= 0 and mot17.cache_valid(out, {"stride": 2, "imgsz": 640})
    detection.run_detector(StubDetector(extra=False), seq, 1, None, out, numbers=[2])
    assert mot17.read_mot(out)[:, 0].tolist() == [2, 2] and mot17.read_meta(out)["frames"] == 1


@pytest.mark.usefixtures("mot17_root")
def test_run_detector_with_no_detections_writes_an_empty_file(tmp_path):
    class Empty:
        name = "empty"

        def __call__(self, frame):
            return Detections(np.empty((0, 4)), np.empty(0))

    seq = mot17.load_sequence("MOT17-02-FRCNN")
    out = tmp_path / "empty.txt"
    detection.run_detector(Empty(), seq, 1, 2, out)
    assert mot17.read_mot(out).shape == (0, 10) and mot17.read_meta(out)["frames"] == 2
    o = detection.evaluate({seq.name: (mot17.load_gt(seq), mot17.read_mot(out), [1, 2])})[
        "overall"
    ]
    assert (o["AP"], o["AP50"], o["precision"], o["recall"], o["tp"], o["gt"]) == (
        0,
        0,
        0,
        0,
        0,
        4,
    )


@pytest.mark.usefixtures("mot17_root")
def test_evaluate_on_the_synthetic_sequence(tmp_path):
    seq = mot17.load_sequence("MOT17-02-FRCNN")
    out = tmp_path / "stub.txt"
    detection.run_detector(StubDetector(), seq, 1, None, out)
    m = detection.evaluate({seq.name: (mot17.load_gt(seq), mot17.read_mot(out), [1, 2, 3])})
    o = m["overall"]
    # Per frame: the 0.9 box hits, the 0.2 box is below the 0.3 operating point, 0.8 is an FP.
    assert (o["tp"], o["fp"], o["fn"], o["gt"], o["frames"]) == (3, 3, 3, 6, 3)
    assert o["precision"] == 0.5 and o["recall"] == 0.5 and o["f1"] == 0.5
    assert o["AP"] == pytest.approx(o["AP50"]) and 0.8 < o["AP50"] < 0.9  # exact boxes: IoU 1
    assert o["recall_by_visibility"] == {
        "0.00-0.25": None,
        "0.25-0.50": None,
        "0.50-0.75": 0.0,
        "0.75-1.00": 1.0,
    }
    assert o["gt_by_visibility"] == {
        "0.00-0.25": 0,
        "0.25-0.50": 0,
        "0.50-0.75": 3,
        "0.75-1.00": 3,
    }
    assert o["f1_by_conf"]["0.20"] == pytest.approx(0.8) and o["f1_by_conf"]["0.85"] < 0.7
    assert o["best_conf"] == 0.05 and o["at_best_conf"]["f1"] == pytest.approx(0.8)
    per = m["per_sequence"][seq.name]
    assert per["AP50"] == o["AP50"] and per["tp"] == 3 and "f1_by_conf" not in per


def test_frames_without_gt_or_detections_are_skipped():
    gt = pd.DataFrame(gt_frame_rows(1), columns=mot17.GT_COLUMNS)
    rows = np.array([[1, -1, 5, 5, 10, 20, 0.9, -1, -1, -1]], float)
    o = detection.evaluate({"S": (gt, rows, [1, 2])})["overall"]
    assert (o["tp"], o["fp"], o["gt"], o["frames"]) == (1, 0, 2, 2)


@pytest.mark.usefixtures("mot17_root")
def test_detection_main_runs_caches_and_writes_json(tmp_path, monkeypatch, capsys):
    made = []

    def fake_load(name, conf, device):
        made.append((name, conf, device))
        return StubDetector(name)

    monkeypatch.setattr(detection, "load_detector", fake_load)
    out, preds = tmp_path / "det.json", tmp_path / "preds"
    args = ["--models", "stub", "--sequences", *SEQUENCES, "--limit", "2"]
    args += ["--out", str(out), "--preds-dir", str(preds)]
    detection.main(args)
    report = json.loads(out.read_text())
    assert made == [("stub", detection.CACHE_CONF, "0")]
    assert report["protocol"]["frames_evaluated"] == dict.fromkeys(SEQUENCES, 2)
    m = report["models"]["stub"]
    assert (
        set(m) == {"overall", "per_sequence", "seconds_per_frame"} and m["seconds_per_frame"] > 0
    )
    assert m["overall"]["tp"] == 4 and set(m["per_sequence"]) == set(SEQUENCES)
    detection.main(args)  # every cache carries the key: no detector is loaded
    assert len(made) == 1
    detection.main([*args, "--force"])
    assert len(made) == 2 and "wrote" in capsys.readouterr().out


@pytest.mark.usefixtures("mot17_root")
def test_run_tracker_and_score_on_the_synthetic_sequence(tmp_path):
    seq = mot17.load_sequence("MOT17-04-FRCNN")  # 4 frames; the stub returns nothing on call 3
    trk = StubTracker()
    out = tmp_path / "tracks" / "stub-bytetrack" / "data" / f"{seq.name}.txt"
    tracking.run_tracker(trk, "bytetrack.yaml", seq, None, out, {"conf": 0.1})
    rows = mot17.read_mot(out)
    assert trk.resets == 1 and trk.calls == 4
    assert rows[:, 0].tolist() == [1, 1, 2, 2, 4, 4] and rows[:, 1].tolist() == [1, 2, 1, 2, 1, 2]
    assert rows[:, 6].tolist() == [1] * 6 and rows[0, 2:6].tolist() == [5, 5, 10, 20]
    meta = mot17.read_meta(out)
    assert meta["frames"] == 4 and meta["conf"] == 0.1 and meta["seconds"] >= 0
    res = tracking.score(tmp_path / "tracks", "stub-bytetrack", {seq.name: 4})
    assert set(res) == {seq.name, "COMBINED_SEQ"}
    r = res[seq.name]
    assert (r["FN"], r["FP"], r["IDSW"]) == (2, 0, 0) and r["MOTA"] == pytest.approx(0.75)


def test_tracking_model_requires_an_ultralytics_detector(monkeypatch):
    monkeypatch.setattr(tracking, "load_detector", lambda name, conf, device: StubDetector(name))
    with pytest.raises(SystemExit, match="needs an Ultralytics model"):
        tracking.tracking_model("yolov3-darknet", 0.1, "cpu")
    ultra = object.__new__(UltralyticsYolo)
    monkeypatch.setattr(tracking, "load_detector", lambda name, conf, device: ultra)
    assert tracking.tracking_model("yolo26n", 0.1, "cpu") is ultra


@pytest.mark.usefixtures("mot17_root")
def test_tracking_main_scores_and_caches(tmp_path, monkeypatch, capsys):
    loads = []

    def fake_model(name, conf, device):
        loads.append((name, conf, device))
        return StubTracker(name)

    monkeypatch.setattr(tracking, "tracking_model", fake_model)
    out, tracks = tmp_path / "trk.json", tmp_path / "tracks"
    args = ["--models", "stub", "--trackers", "bytetrack", "--sequences", "MOT17-02-FRCNN"]
    args += ["--limit", "2", "--out", str(out), "--tracks-dir", str(tracks)]
    tracking.main(args)
    report = json.loads(out.read_text())
    assert loads == [("stub", tracking.CONF, "0")]
    run = report["runs"]["stub-bytetrack"]
    assert (run["model"], run["tracker"]) == ("stub", "bytetrack")
    assert set(run["per_sequence"]) == {"MOT17-02-FRCNN"} and run["overall"]["fps"] > 0
    assert (run["overall"]["FN"], run["overall"]["FP"], run["overall"]["IDSW"]) == (0, 0, 0)
    assert report["protocol"]["frames_evaluated"] == {"MOT17-02-FRCNN": 2}
    tracking.main(args)  # cached
    assert len(loads) == 1
    tracking.main([*args, "--force", "--conf", "0.2"])
    assert len(loads) == 2 and loads[1][1] == 0.2 and "HOTA" in capsys.readouterr().out
