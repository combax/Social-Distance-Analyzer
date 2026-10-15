import json
from pathlib import Path

import pandas as pd
import pytest
from conftest import StubDetector, StubTracker

from evaluation import finetune, mot17

W, H = 1920, 1080


def _gt(*rows):
    return pd.DataFrame(list(rows), columns=mot17.GT_COLUMNS)


def _parse(line):
    cls, *vals = line.split()
    return int(cls), [float(v) for v in vals]


def test_known_box_gives_exact_normalised_line():
    # 1-based (961, 541) is 0-based (960, 540); 192x216 -> centre (1056, 648).
    lines = finetune.yolo_labels(_gt((1, 1, 961, 541, 192, 216, 1, 1, 1.0)), W, H)
    assert lines == ["0 0.550000 0.600000 0.100000 0.200000"]


def test_class_filter():
    rows = _gt(
        (1, 1, 100, 100, 50, 100, 1, 1, 1.0),  # pedestrian, flag 1: kept
        (1, 2, 200, 100, 50, 100, 0, 1, 1.0),  # pedestrian, flag 0: excluded
        (1, 3, 300, 100, 50, 100, 0, 2, 1.0),  # person on vehicle: kept
        (1, 4, 400, 100, 50, 100, 0, 7, 1.0),  # static person: kept
        (1, 5, 500, 100, 50, 100, 0, 8, 1.0),  # distractor: excluded
        (1, 6, 600, 100, 50, 100, 0, 12, 1.0),  # reflection: excluded
        (1, 7, 700, 100, 50, 100, 0, 3, 1.0),  # car: excluded
    )
    lines = finetune.yolo_labels(rows, W, H)
    assert len(lines) == 3
    xs = sorted(round(_parse(li)[1][0] * W) for li in lines)
    assert xs == [124, 324, 424]  # 0-based left + 25
    assert all(_parse(li)[0] == 0 for li in lines)


def test_boxes_are_clipped_to_the_image():
    rows = _gt(
        (1, 1, -99, 1, 300, 100, 1, 1, 1.0),  # left edge at -100 -> clipped to 0, width 200
        (1, 2, 1820, 1030, 300, 300, 1, 1, 1.0),  # overflows right and bottom
        (1, 3, -500, 10, 100, 100, 1, 1, 1.0),  # entirely outside -> dropped
    )
    lines = finetune.yolo_labels(rows, W, H)
    assert len(lines) == 2
    _, (cx, cy, bw, bh) = _parse(lines[0])
    assert (round(cx * W), round(bw * W)) == (100, 200)
    assert (round(cy * H), round(bh * H)) == (50, 100)
    _, (cx, cy, bw, bh) = _parse(lines[1])
    assert cx + bw / 2 == pytest.approx(1.0, abs=1e-6)
    assert cy + bh / 2 == pytest.approx(1.0, abs=1e-6)
    assert (round(bw * W), round(bh * H)) == (W - 1819, H - 1029)
    for li in lines:
        for v in _parse(li)[1]:
            assert 0.0 <= v <= 1.0


def test_split_is_by_sequence_and_covers_all_seven():
    assert not set(finetune.TRAIN) & set(finetune.TEST)
    assert set(finetune.TRAIN) | set(finetune.TEST) == set(mot17.SEQUENCES)
    assert len(finetune.TRAIN) + len(finetune.TEST) == 7


@pytest.mark.parametrize(
    ("length", "n_val"),
    [(1, 0), (3, 0), (4, 1), (10, 2), (13, 2), (600, 90), (654, 98), (1050, 158)],
)
def test_val_is_last_15_percent_rounded_half_up(length, n_val):
    train, val = finetune.split_frames(length)
    assert len(val) == n_val == finetune.val_count(length)
    assert val == list(range(length - n_val + 1, length + 1))
    assert not set(train) & set(val)
    assert sorted(train + val) == list(range(1, length + 1))


@pytest.mark.skipif(not (mot17.ROOT / finetune.TRAIN[0]).exists(), reason="MOT17 not downloaded")
def test_val_counts_on_real_train_sequences():
    lengths = {n: mot17.load_sequence(n).length for n in finetune.TRAIN}
    assert lengths == {
        "MOT17-04-FRCNN": 1050,
        "MOT17-10-FRCNN": 654,
        "MOT17-11-FRCNN": 900,
        "MOT17-13-FRCNN": 750,
    }
    assert [finetune.val_count(n) for n in lengths.values()] == [158, 98, 135, 113]


DATASET = finetune.DATASET


@pytest.mark.skipif(not (DATASET / "images" / "val").exists(), reason="run prepare first")
def test_prepared_dataset_follows_the_split():
    for split_index, split in enumerate(("train", "val")):
        names = {p.stem for p in (DATASET / "images" / split).iterdir()}
        expected = set()
        for seq_name in finetune.TRAIN:
            seq = mot17.load_sequence(seq_name)
            expected |= {
                f"{seq_name}_{f:06d}" for f in finetune.split_frames(seq.length)[split_index]
            }
        assert names == expected, split


@pytest.fixture
def finetune_env(tmp_path, monkeypatch):
    """One TRAIN and one TEST sequence (use with ``mot17_root``), outputs under tmp_path."""
    monkeypatch.setattr(finetune, "TRAIN", ("MOT17-04-FRCNN",))
    monkeypatch.setattr(finetune, "TEST", ("MOT17-02-FRCNN",))
    monkeypatch.setattr(finetune, "DATASET", tmp_path / "mot17_yolo")
    monkeypatch.setattr(finetune, "PROJECT", tmp_path / "finetune")
    base = tmp_path / "models" / "yolo26s.pt"
    base.parent.mkdir()
    base.write_bytes(b"base")
    monkeypatch.setattr(finetune, "BASE_WEIGHTS", base)
    return tmp_path


@pytest.mark.usefixtures("mot17_root")
def test_prepare_builds_the_yolo_dataset_and_keeps_existing_files(finetune_env, capsys):
    # MOT17-04 has 4 frames: val is frame 4; classes 1 and 7 give three labels per frame.
    counts = {"train": {"images": 3, "boxes": 9}, "val": {"images": 1, "boxes": 3}}
    assert finetune.prepare() == counts
    ds = finetune.DATASET
    assert (ds / "data.yaml").read_text().startswith(f"path: {ds.resolve().as_posix()}\n")
    assert (ds / "images" / "val" / "MOT17-04-FRCNN_000004.jpg").exists()
    label = ds / "labels" / "train" / "MOT17-04-FRCNN_000001.txt"
    lines = label.read_text().splitlines()
    assert len(lines) == 3 and all(li.startswith("0 ") for li in lines)
    label.write_text("edited\n")
    counts["train"]["boxes"] = 7
    assert finetune.prepare() == counts and label.read_text() == "edited\n"
    assert "train: 3 images, 7 boxes" in capsys.readouterr().out


def test_link_falls_back_to_a_copy(tmp_path, monkeypatch):
    src = tmp_path / "a"
    src.write_bytes(b"x")

    def fail(*args):
        raise OSError("cross-device link")

    monkeypatch.setattr(finetune.os, "link", fail)
    finetune._link(src, tmp_path / "b")
    assert (tmp_path / "b").read_bytes() == b"x"


def test_training_curve_merges_resumed_segments(tmp_path):
    csv = (
        "epoch, time, metrics/mAP50-95(B)\n"
        "1, 10.0, 0.1\n2, 20.0, 0.2\n2, 5.0, 0.25\n3, 15.0, 0.3\n"
    )
    (tmp_path / "results.csv").write_text(csv)
    assert finetune._training_curve(tmp_path) == ([0.1, 0.25, 0.3], 35.0, 2)


@pytest.mark.usefixtures("mot17_root")
def test_finalize_records_the_last_checkpoint(finetune_env, capsys):
    import torch

    run = finetune.PROJECT / "run"
    (run / "weights").mkdir(parents=True)
    args = {"epochs": 15, "imgsz": 1280, "batch": 4, "patience": 5, "seed": 0}
    torch.save({"train_args": args, "epoch": 6}, run / "weights" / "last.pt")
    rows = "".join(f"{e}, {e * 10.0}, 0.{e}\n" for e in range(1, 8))
    (run / "results.csv").write_text("epoch, time, metrics/mAP50-95(B)\n" + rows)
    assert finetune.finalize("run", "machine crashed") == run / "weights" / "best.pt"
    rec = json.loads((run / "finetune_train.json").read_text())
    assert (rec["epochs_run"], rec["epochs_planned"], rec["batch_size"]) == (7, 15, 4)
    assert rec["stopped_early"] == "machine crashed" and rec["settings"] == args
    assert rec["segments"] == 1 and rec["resumed_from_last_pt"] is False
    assert rec["val_map50_95_by_epoch"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    assert rec["wall_seconds"] == 70.0 and rec["best"].endswith("run/weights/best.pt")
    assert rec["split"]["train"] == ["MOT17-04-FRCNN"] and {"ultralytics", "torch"} <= set(rec)
    assert "finalized run: 7 of 15 epochs" in capsys.readouterr().out


@pytest.mark.usefixtures("mot17_root")
def test_compare_scores_both_models_and_reuses_caches(finetune_env, monkeypatch, capsys):
    made = []

    def fake_load(spec, conf, device):
        made.append(StubDetector(Path(spec.rpartition("@")[0]).stem))
        return made[-1]

    monkeypatch.setattr(finetune, "load_detector", fake_load)
    monkeypatch.setattr(finetune, "UltralyticsYolo", StubTracker)
    weights = finetune.PROJECT / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"tuned")
    (weights.parent.parent / "finetune_train.json").write_text('{"epochs_run": 3}')
    out = finetune_env / "finetune.json"
    finetune.compare(weights, 640, None, "cpu", out)
    report = json.loads(out.read_text())
    assert [d.name for d in made] == ["yolo26s", "best"] and made[1].calls == 4  # 3 test + 1 val
    assert set(report["models"]) == {"zero_shot", "finetuned"}
    assert report["training"] == {"epochs_run": 3}
    m = report["models"]["finetuned"]
    assert m["detector"] == "best" and m["weights"].endswith("run/weights/best.pt")
    assert m["detection"]["overall"]["frames"] == 3 and m["detection"]["overall"]["tp"] == 3
    assert m["best_conf_on_val"] == 0.05 and m["best_conf_on_grid_edge"] is True
    assert m["test_at_best_conf"]["f1"] == pytest.approx(0.8)
    assert m["tracking"]["overall"]["IDSW"] == 0 and set(m["tracking"]["per_sequence"]) == {
        "MOT17-02-FRCNN"
    }
    frames = {"test": {"MOT17-02-FRCNN": 3}, "val": {"MOT17-04-FRCNN": 1}}
    assert report["protocol"]["frames_evaluated"] == frames
    preds = finetune.PROJECT / "preds" / "best"
    assert (preds / "MOT17-02-FRCNN.txt").exists() and (preds / "MOT17-04-FRCNN-val.txt").exists()
    assert (
        finetune.PROJECT / "tracks" / "best-bytetrack" / "data" / "MOT17-02-FRCNN.txt"
    ).exists()

    finetune.compare(weights, 640, None, "cpu", out, only="finetuned")
    assert [d.name for d in made] == ["yolo26s", "best", "best"] and made[2].calls == 0  # cached
    assert set(json.loads(out.read_text())["models"]) == {"finetuned"}
    assert "finetuned  AP" in capsys.readouterr().out


@pytest.mark.usefixtures("mot17_root")
def test_compare_needs_the_finetuned_weights(finetune_env):
    with pytest.raises(SystemExit, match="not found"):
        finetune.compare(finetune_env / "missing.pt", 640, None, "cpu", finetune_env / "o.json")


def test_main_dispatches_subcommands(monkeypatch):
    calls = []
    monkeypatch.setattr(finetune, "prepare", lambda: calls.append("prepare"))
    monkeypatch.setattr(finetune, "train", lambda *a: calls.append(("train", a)))
    monkeypatch.setattr(finetune, "finalize", lambda *a: calls.append(("finalize", a)))
    monkeypatch.setattr(finetune, "compare", lambda *a: calls.append(("compare", a)))
    finetune.main(["prepare"])
    finetune.main(["train", "--epochs", "2", "--name", "x"])
    finetune.main(["train", "--finalize", "crash", "--name", "x"])
    finetune.main(["compare", "--weights", "w.pt", "--limit", "5", "--only", "zero_shot"])
    assert calls == [
        "prepare",
        ("train", (2, 1280, 8, 6, 1.0, "x", 10, False)),
        ("finalize", ("x", "crash")),
        ("compare", (Path("w.pt"), 1280, 5, "0", Path("results/finetune.json"), "zero_shot")),
    ]
