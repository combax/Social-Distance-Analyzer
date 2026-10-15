"""Fine-tune a YOLO detector on MOT17 crowd footage and compare it with the COCO zero-shot model.

The split is by sequence, never by frame: TRAIN sequences feed training, model selection and
threshold selection; TEST sequences are only ever scored.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import detection, mot17, tracking
from sda.detect import MAX_DET, MODELS_DIR, Detector, UltralyticsYolo, load_detector, rel

TRAIN: tuple[str, ...] = tuple(f"MOT17-{n}-FRCNN" for n in ("04", "10", "11", "13"))
TEST: tuple[str, ...] = tuple(f"MOT17-{n}-FRCNN" for n in ("02", "05", "09"))
VAL_PERCENT = 15  # last 15% of each TRAIN sequence's frames, rounded half up
# Person on vehicle (2) and static person (7) are people and ignore regions in the evaluation,
# so they are labelled; distractors (8), reflections (12) and non-person classes stay background.
EXTRA_PERSON_CLASSES = (2, 7)
BASE_WEIGHTS = MODELS_DIR / "yolo26s.pt"
DATASET = MODELS_DIR.parent / "mot17_yolo"
PROJECT = Path("results/finetune")
TRACKER = "bytetrack"
GPU_MEMORY_FRACTION = 0.85  # torch.cuda.set_per_process_memory_fraction in train
CONF_GRID = detection.CONF_GRID  # the same sweep as the zero-shot detection benchmark


def val_count(length: int) -> int:
    """Return how many of the last frames form the val split: 15%, rounded half up."""
    return (length * VAL_PERCENT + 50) // 100


def split_frames(length: int) -> tuple[list[int], list[int]]:
    """Return ``(train, val)`` 1-based frame numbers; val is the last ``val_count`` frames."""
    cut = length - val_count(length)
    return list(range(1, cut + 1)), list(range(cut + 1, length + 1))


def yolo_labels(gt_frame_rows: pd.DataFrame, width: int, height: int) -> list[str]:
    """Convert one frame's GT rows to YOLO label lines (class 0, normalised ``cx cy w h``).

    Args:
        gt_frame_rows: Rows of ``mot17.load_gt`` for one frame (``x, y`` 1-based top-left).
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        One line per kept box, clipped to the image; boxes left empty by clipping are dropped.
    """
    cls = gt_frame_rows["class"]
    keep = ((cls == mot17.PEDESTRIAN) & (gt_frame_rows["flag"] == 1)) | cls.isin(
        EXTRA_PERSON_CLASSES
    )
    x, y, w, h = gt_frame_rows.loc[keep, ["x", "y", "w", "h"]].to_numpy(dtype=float).T
    x1, x2 = np.clip(x - 1, 0, width), np.clip(x - 1 + w, 0, width)
    y1, y2 = np.clip(y - 1, 0, height), np.clip(y - 1 + h, 0, height)
    return [
        f"0 {(a + b) / 2 / width:.6f} {(c + d) / 2 / height:.6f} "
        f"{(b - a) / width:.6f} {(d - c) / height:.6f}"
        for a, b, c, d in zip(x1, x2, y1, y2, strict=True)
        if b > a and d > c
    ]


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def _write_split(seq: mot17.Sequence, gt: pd.DataFrame, split: str, nums: list[int]) -> None:
    img_dir, lbl_dir = DATASET / "images" / split, DATASET / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for n in nums:
        stem = f"{seq.name}_{n:06d}"
        if not (img_dir / f"{stem}.jpg").exists():
            _link(seq.dir / "img1" / f"{n:06d}.jpg", img_dir / f"{stem}.jpg")
        if not (lbl_dir / f"{stem}.txt").exists():
            lines = yolo_labels(gt[gt["frame"] == n], seq.width, seq.height)
            (lbl_dir / f"{stem}.txt").write_text("".join(f"{s}\n" for s in lines))


def prepare() -> dict[str, dict[str, int]]:
    """Build the YOLO dataset under ``data/mot17_yolo`` (idempotent: existing files are kept).

    Returns:
        ``{"train": {"images": n, "boxes": n}, "val": {...}}``.
    """
    for name in TRAIN:
        seq = mot17.load_sequence(name)
        gt = mot17.load_gt(seq)
        for split, nums in zip(("train", "val"), split_frames(seq.length), strict=True):
            _write_split(seq, gt, split, nums)
    (DATASET / "data.yaml").write_text(
        f"path: {DATASET.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: person\n"
    )
    counts = {
        split: {
            "images": len(list((DATASET / "images" / split).glob("*.jpg"))),
            "boxes": sum(
                len(p.read_text().splitlines()) for p in (DATASET / "labels" / split).glob("*.txt")
            ),
        }
        for split in ("train", "val")
    }
    for split, c in counts.items():
        print(f"{split}: {c['images']} images, {c['boxes']} boxes")
    return counts


def _versions() -> dict[str, str]:
    import torch
    import ultralytics

    return {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda_device": (torch.cuda.get_device_name(0) if torch.cuda.device_count() else "none"),
    }


def train(  # pragma: no cover  (Ultralytics training on the GPU; not unit-testable)
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    fraction: float,
    name: str,
    patience: int = 10,
    resume: bool = False,
) -> Path:
    """Fine-tune ``yolo26s`` on the prepared dataset and record the run next to its weights.

    PyTorch is capped at ``GPU_MEMORY_FRACTION`` of the GPU so it raises OOM instead of spilling
    into system RAM.

    Args:
        epochs: Maximum epochs (early stopping with ``patience`` on the val split).
        imgsz: Training image size.
        batch: Fixed batch size (8 fits imgsz 1280 on 8 GB; no AutoBatch).
        workers: Dataloader workers.
        fraction: Fraction of the train split to use (smoke tests).
        name: Run name under ``results/finetune``.
        patience: Epochs without val improvement before early stopping.
        resume: Continue from this run's ``last.pt`` if it exists (after a crash).

    Returns:
        Path of ``best.pt``.
    """
    import torch
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models import YOLO

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    settings: dict[str, object] = {
        "data": str(DATASET / "data.yaml"),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "fraction": fraction,
        "patience": patience,
        "seed": 0,
        "deterministic": True,
        "workers": workers,
        "cache": False,
        "project": str(PROJECT.resolve()),  # relative paths get prefixed with runs_dir
        "name": name,
        "exist_ok": True,
    }
    last = PROJECT / name / "weights" / "last.pt"
    resumed = resume and last.exists()
    model = YOLO(str(last if resumed else BASE_WEIGHTS))
    if resumed:  # Ultralytics restores every setting from the checkpoint
        model.train(resume=True)
    else:
        model.train(**settings)
    trainer: BaseTrainer = model.trainer  # type: ignore[assignment]  # declared None upstream
    best = Path(trainer.best)
    actual = vars(trainer.args)  # a resume restores these from last.pt
    curve, wall, segments = _training_curve(best.parent.parent)  # every segment, not just this one
    record = {
        "base_weights": rel(BASE_WEIGHTS),
        "split": {
            "train": list(TRAIN),
            "test": list(TEST),
            "val_percent_of_train_seqs": VAL_PERCENT,
        },
        "settings": {
            k: rel(str(v)) if k in ("data", "project") else actual.get(k, v)
            for k, v in settings.items()
        },
        "augmentations": "Ultralytics defaults",
        "gpu_memory_fraction": GPU_MEMORY_FRACTION,
        "epochs_run": int(trainer.epoch) + 1,
        "epochs_planned": int(actual["epochs"]),
        "batch_size": int(trainer.batch_size),
        "wall_seconds": wall,
        "resumed_from_last_pt": resumed,
        "segments": segments,
        "val_map50_95_by_epoch": curve,
        "best": rel(best),
        **_versions(),
    }
    (best.parent.parent / "finetune_train.json").write_text(json.dumps(record, indent=2))
    print(f"best weights: {best}  ({wall / 60:.1f} min, {record['epochs_run']} epochs)")
    return best


def _training_curve(run: Path) -> tuple[list[float], float, int]:
    """Val mAP50-95 per epoch, wall seconds summed over every segment, and the segment count.

    ``results.csv`` keeps appending across resumes: cumulative time restarts and an epoch that
    was interrupted appears twice, so the curve keeps the last row per epoch.
    """
    csv = pd.read_csv(run / "results.csv")
    csv.columns = [c.strip() for c in csv.columns]
    segments = (csv["time"].diff() < 0).cumsum()
    wall = float(csv.groupby(segments)["time"].max().sum())
    by_epoch = csv.drop_duplicates("epoch", keep="last")
    curve = [round(float(v), 4) for v in by_epoch["metrics/mAP50-95(B)"]]
    return curve, wall, int(segments.max()) + 1


def finalize(name: str, reason: str) -> Path:
    """Close a training run from its last checkpoint without training further.

    Used when the hardware stopped the run: writes ``finetune_train.json`` from ``last.pt``
    (epochs completed, settings) and the per-run segments of ``results.csv``, so ``compare``
    reports the model that exists. ``last.pt`` is kept, so ``--resume`` can still continue.

    Args:
        name: Run name under ``results/finetune``.
        reason: Why the run was closed early, recorded verbatim.

    Returns:
        Path of ``best.pt``.
    """
    import torch

    run = PROJECT / name
    ckpt = torch.load(run / "weights" / "last.pt", map_location="cpu", weights_only=False)
    args = ckpt["train_args"]
    curve, wall, segments = _training_curve(run)
    record = {
        "base_weights": rel(BASE_WEIGHTS),
        "split": {
            "train": list(TRAIN),
            "test": list(TEST),
            "val_percent_of_train_seqs": VAL_PERCENT,
        },
        "settings": {k: args.get(k) for k in ("epochs", "imgsz", "batch", "patience", "seed")},
        "augmentations": "Ultralytics defaults",
        "gpu_memory_fraction": GPU_MEMORY_FRACTION,
        "epochs_run": int(ckpt["epoch"]) + 1,
        "epochs_planned": int(args["epochs"]),
        "batch_size": int(args["batch"]),
        "wall_seconds": wall,
        "resumed_from_last_pt": segments > 1,
        "segments": segments,
        "stopped_early": reason,
        "val_map50_95_by_epoch": curve,
        "best": rel(run / "weights" / "best.pt"),
        **_versions(),
    }
    (run / "finetune_train.json").write_text(json.dumps(record, indent=2))
    print(f"finalized {name}: {record['epochs_run']} of {record['epochs_planned']} epochs")
    return run / "weights" / "best.pt"


def _at_conf(runs: dict[str, detection.Run], conf: float) -> dict[str, object]:
    """Operating-point metrics at ``conf`` with the evaluation's own matching code."""
    overall = dict(detection.evaluate(runs, conf=conf)["overall"])
    return {k: overall[k] for k in ("precision", "recall", "f1", "tp", "fp", "fn")}


def _detect(
    det: Detector,
    seqs: list[mot17.Sequence],
    gts: dict[str, pd.DataFrame],
    frames: dict[str, list[int]],
    limit: int | None,
    preds: Path,
    key: dict[str, object],
    suffix: str,
) -> dict[str, detection.Run]:
    """Detect ``frames`` of each sequence into ``preds`` (unless cached under ``key``)."""
    runs: dict[str, detection.Run] = {}
    for s in seqs:
        p = preds / f"{s.name}{suffix}.txt"
        if not mot17.cache_valid(p, key):
            detection.run_detector(det, s, 1, limit, p, frames[s.name], key)
        runs[s.name] = (gts[s.name], mot17.read_mot(p), frames[s.name])
    return runs


def _track(
    name: str,
    pt: Path,
    imgsz: int,
    device: str,
    test: list[mot17.Sequence],
    limit: int | None,
    key: dict[str, object],
    lengths: dict[str, int],
) -> dict[str, dict[str, float]]:
    """Track TEST with ``pt`` into ``tracks/<name>-<tracker>`` (cached under ``key``) and score."""
    trk = UltralyticsYolo(pt, conf=tracking.CONF, imgsz=imgsz, device=device)
    trk_name = f"{name}-{TRACKER}"
    trk_key: dict[str, object] = {**key, "conf": tracking.CONF}
    for s in test:
        p = PROJECT / "tracks" / trk_name / "data" / f"{s.name}.txt"
        if not mot17.cache_valid(p, trk_key):
            tracking.run_tracker(trk, tracking.TRACKERS[TRACKER], s, limit, p, trk_key)
    del trk
    return tracking.score(PROJECT / "tracks", trk_name, lengths)


def compare(
    weights: Path, imgsz: int, limit: int | None, device: str, out: Path, only: str | None = None
) -> None:
    """Score zero-shot ``yolo26s`` and the fine-tuned weights on TEST with identical settings.

    Args:
        weights: Fine-tuned ``best.pt``.
        imgsz: Inference size for both models.
        limit: Use only the first ``limit`` TEST frames and val frames per sequence.
        device: Ultralytics device string.
        out: Output JSON.
        only: ``zero_shot`` or ``finetuned`` to run one side (its caches are reused by the
            full run later).
    """
    test = [mot17.load_sequence(n) for n in TEST]
    val = [mot17.load_sequence(n) for n in TRAIN]
    gts = {s.name: mot17.load_gt(s) for s in test + val}
    test_frames = {s.name: mot17.frame_numbers(s, limit=limit) for s in test}
    val_frames = {s.name: split_frames(s.length)[1][:limit] for s in val}
    lengths = {s.name: len(f) for s, f in zip(test, test_frames.values(), strict=True)}
    specs = {
        "zero_shot": (BASE_WEIGHTS, f"yolo26s@{imgsz}"),
        "finetuned": (weights, f"{weights}@{imgsz}"),
    }
    if only:
        specs = {only: specs[only]}
    if "finetuned" in specs and not weights.exists():
        raise SystemExit(f"{weights} not found: run `python -m evaluation.finetune train` first")
    models: dict[str, dict[str, object]] = {}
    for label, (pt, spec) in specs.items():
        det = load_detector(spec, conf=detection.CACHE_CONF, device=device)  # downloads base
        st = pt.resolve().stat()  # a retrain rewrites best.pt in place: key on its identity
        key: dict[str, object] = {
            "limit": limit,
            "imgsz": imgsz,
            "weights": rel(pt),
            "weights_size": st.st_size,
            "weights_mtime_ns": st.st_mtime_ns,
            "max_det": MAX_DET,
        }
        preds = PROJECT / "preds" / det.name
        runs_test = _detect(det, test, gts, test_frames, limit, preds, key, "")
        runs_val = _detect(det, val, gts, val_frames, limit, preds, key, "-val")
        name = det.name
        del det
        # One evaluation of the val split gives F1 at every grid confidence.
        val_f1 = detection.evaluate(runs_val)["overall"]["f1_by_conf"]
        sweep = {c: float(val_f1[f"{c:.2f}"]) for c in CONF_GRID}
        best_conf = max(sweep, key=lambda c: sweep[c])
        scores = _track(name, pt, imgsz, device, test, limit, key, lengths)
        det_metrics = detection.evaluate(runs_test)
        at_best = _at_conf(runs_test, best_conf)
        models[label] = {
            "detector": name,
            "weights": rel(pt),
            "detection": det_metrics,
            "val_f1_by_conf": {f"{c:.2f}": f for c, f in sweep.items()},
            "best_conf_on_val": best_conf,
            "best_conf_on_grid_edge": best_conf in (min(CONF_GRID), max(CONF_GRID)),
            "test_at_best_conf": at_best,
            "tracking": {
                "overall": scores["COMBINED_SEQ"],
                "per_sequence": {s.name: scores[s.name] for s in test},
            },
        }
        o, t = det_metrics["overall"], scores["COMBINED_SEQ"]
        print(
            f"{label:10s} AP {o['AP']:.3f}  AP50 {o['AP50']:.3f}  F1@0.3 {o['f1']:.3f}  "
            f"F1@{best_conf:.2f} {at_best['f1']:.3f}  "
            f"HOTA {t['HOTA']:.3f}  MOTA {t['MOTA']:.3f}  IDF1 {t['IDF1']:.3f}  "
            f"IDSW {t['IDSW']:.0f}"
        )

    train_record = weights.parent.parent / "finetune_train.json"
    protocol = {
        "split": {
            "train_sequences": list(TRAIN),
            "test_sequences": list(TEST),
            "rule": "by sequence; TEST never used for training, model selection or thresholds",
            "val": f"last {VAL_PERCENT}% of each TRAIN sequence's frames, rounded half up "
            "((length * 15 + 50) // 100); not trained on, but the temporal tail of the same "
            "scenes (optimistic for the fine-tuned model); used for Ultralytics early stopping, "
            "best.pt selection and the conf threshold sweep",
        },
        "labels": "single class person: class 1 with flag 1, plus classes 2 and 7; classes 8 "
        "and 12 and non-person classes are background; boxes clipped to the image",
        "imgsz": imgsz,
        "limit": limit,
        "frames_evaluated": {"test": lengths, "val": {k: len(v) for k, v in val_frames.items()}},
        "detection": "evaluation.detection.evaluate on TEST (COCO AP/AP50, operating point conf "
        f"{detection.OP_CONF} IoU {detection.OP_IOU}, recall by visibility)",
        "threshold": f"conf maximising F1 on the val frames over {CONF_GRID}; TEST P/R/F1 "
        "reported at that conf",
        "tracking": f"Ultralytics {tracking.TRACKERS[TRACKER]} at conf {tracking.CONF}, scored "
        "with evaluation.tracking.score (trackeval, MOT17 train protocol) on TEST",
        "device": device,
        **_versions(),
    }
    result = {
        "protocol": protocol,
        "training": json.loads(train_record.read_text()) if train_record.exists() else None,
        "models": models,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> None:
    """Command line: ``prepare``, ``train`` and ``compare``."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare", help="build data/mot17_yolo from the TRAIN sequences")
    t = sub.add_parser("train", help="fine-tune yolo26s")
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=10)
    t.add_argument("--resume", action="store_true", help="continue from last.pt if present")
    t.add_argument("--finalize", metavar="REASON", help="close the run from last.pt, no training")
    t.add_argument("--imgsz", type=int, default=1280)
    t.add_argument("--batch", type=int, default=8, help="fixed; 8 fits imgsz 1280 on 8 GB")
    t.add_argument("--workers", type=int, default=6)  # 2 starved the GPU at 1280 px
    t.add_argument("--fraction", type=float, default=1.0)
    t.add_argument("--name", default="yolo26s-mot17")
    c = sub.add_parser("compare", help="zero-shot vs fine-tuned on TEST")
    c.add_argument("--weights", type=Path, required=True)
    c.add_argument("--imgsz", type=int, default=1280)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--device", default="0")
    c.add_argument("--out", type=Path, default=Path("results/finetune.json"))
    c.add_argument("--only", choices=["zero_shot", "finetuned"], default=None)
    a = p.parse_args(argv)
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "train" and a.finalize:
        finalize(a.name, a.finalize)
    elif a.cmd == "train":
        train(a.epochs, a.imgsz, a.batch, a.workers, a.fraction, a.name, a.patience, a.resume)
    else:
        compare(a.weights, a.imgsz, a.limit, a.device, a.out, a.only)


if __name__ == "__main__":
    main()
