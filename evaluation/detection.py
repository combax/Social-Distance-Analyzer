"""Detection accuracy on MOT17: COCO AP (pycocotools) and operating-point precision/recall."""

import argparse
import contextlib
import io
import json
import time
from collections import Counter
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluation import mot17
from sda.detect import MAX_DET, Detector, load_detector

MODELS = (
    "yolov3-darknet",
    "yolov3u",
    "yolo11n",
    "yolo11s",
    "yolo11m",
    "yolo26n",
    "yolo26s",
    "yolo26m",
)
CACHE_CONF = 0.001  # AP needs the full score range
OP_CONF = 0.3  # the app's operating point
OP_IOU = 0.5
CONF_GRID = tuple(round(0.05 * i, 2) for i in range(1, 20))  # 0.05 ... 0.95
MAX_DETS = MAX_DET  # per image; MOT17-04 has ~100 pedestrians per frame
VIS_EDGES = (0.25, 0.5, 0.75)
VIS_LABELS = ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00")
IMAGE_STRIDE = 10**6  # image_id = sequence index * stride + frame

Run = tuple[pd.DataFrame, NDArray[np.float64], list[int]]
"""Ground truth, MOT det rows (``frame, -1, x, y, w, h, score, ...``) and the frames run."""

_Hits = tuple[int, int, Counter[int]]
"""Per image at the operating point: TP, FP and the visibility buckets of the matched GT."""


class Metrics(TypedDict):
    """COCO AP plus operating-point counts; the sweep keys are set on ``overall`` only."""

    AP: float
    AP50: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    gt: int
    frames: int
    recall_by_visibility: dict[str, float | None]
    gt_by_visibility: dict[str, int]
    f1_by_conf: NotRequired[dict[str, float]]
    best_conf: NotRequired[float]
    at_best_conf: NotRequired[dict[str, float]]


class Evaluation(TypedDict):
    """Return shape of :func:`evaluate`."""

    overall: Metrics
    per_sequence: dict[str, Metrics]


def run_detector(
    det: Detector,
    seq: mot17.Sequence,
    stride: int,
    limit: int | None,
    out: Path,
    numbers: list[int] | None = None,
    extra_meta: dict[str, object] | None = None,
) -> None:
    """Detect on the selected frames and cache MOT det rows plus a JSON sidecar at ``out``.

    Args:
        det: Detector to run.
        seq: Sequence to read.
        stride: Keep every ``stride``-th frame.
        limit: Stop after this many frames.
        out: Cache path (``.txt``; the sidecar is ``.json`` next to it).
        numbers: Explicit frame list instead of the stride/limit selection (the val split).
        extra_meta: Merged into the sidecar, e.g. a cache key.
    """
    rows, t0 = [], time.perf_counter()
    for n, img in mot17.frames(seq, stride, limit, numbers):
        d = det(img)
        pad = np.full((len(d), 1), -1.0)
        rows.append(
            np.column_stack(
                (np.full(len(d), n), pad, mot17.to_mot_xywh(d.boxes), d.scores, pad, pad, pad)
            )
        )
    seconds = time.perf_counter() - t0
    frames = mot17.frame_numbers(seq, stride, limit) if numbers is None else numbers
    meta: dict[str, object] = {
        **(extra_meta or {}),
        "stride": stride,
        "limit": limit,
        "frames": len(frames),
        "seconds": seconds,
    }
    mot17.write_mot(out, np.concatenate(rows) if rows else np.empty((0, 10)), meta)


def _coco_gt(runs: dict[str, Run]) -> COCO:
    images: list[dict[str, object]] = []
    anns: list[dict[str, object]] = []
    for si, (name, (gt, _, frames)) in enumerate(runs.items()):
        base = si * IMAGE_STRIDE
        images += [{"id": base + f, "file_name": f"{name}/{f:06d}.jpg"} for f in frames]
        sel = gt[gt["frame"].isin(frames)]
        pos = (sel["class"] == mot17.PEDESTRIAN) & (sel["flag"] == 1)
        ign = sel["class"].isin(mot17.IGNORE_CLASSES)
        cols = ["frame", "x", "y", "w", "h", "class", "visibility"]
        for f, x, y, w, h, cls, vis in sel.loc[pos | ign, cols].to_numpy(dtype=float).tolist():
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": base + int(f),
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": int(cls in mot17.IGNORE_CLASSES),
                    "visibility": vis,
                }
            )
    coco = COCO()
    coco.dataset = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 1, "name": "person"}],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _coco_dt(coco_gt: COCO, runs: dict[str, Run]) -> COCO:
    dets = [
        {
            "image_id": si * IMAGE_STRIDE + int(r[0]),
            "category_id": 1,
            "bbox": r[2:6].tolist(),
            "score": float(r[6]),
        }
        for si, (_, rows, _) in enumerate(runs.values())
        for r in rows
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        if dets:
            return coco_gt.loadRes(dets)
        coco = COCO()
        coco.dataset = {**coco_gt.dataset, "annotations": []}
        coco.createIndex()
        return coco


def _coco_eval(coco_gt: COCO, coco_dt: COCO, img_ids: list[int]) -> COCOeval:
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.params.imgIds = img_ids
    ev.params.maxDets = [MAX_DETS]
    ev.params.areaRng = [[0.0, 1e10]]
    ev.params.areaRngLbl = ["all"]
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
        ev.accumulate()
    return ev


def _ap(ev: COCOeval) -> dict[str, float]:
    prec = ev.eval["precision"][:, :, 0, 0, 0]  # (iou thresholds, recall thresholds)
    return {
        "AP": float(prec[prec > -1].mean()) if (prec > -1).any() else 0.0,
        "AP50": float(prec[0][prec[0] > -1].mean()) if (prec[0] > -1).any() else 0.0,
    }


def _bucket(visibility: float) -> int:
    return int(np.digitize(visibility, VIS_EDGES))


def _operating_point(ev: COCOeval, coco_gt: COCO, conf: float | None = None) -> dict[int, _Hits]:
    """Per image: TP/FP at ``conf`` (default ``OP_CONF``) and matched-GT visibility buckets.

    Matching is pycocotools' greedy-by-score at IoU 0.5 over all cached detections, so
    thresholding afterwards equals running the detector at that confidence.
    """
    conf = OP_CONF if conf is None else conf
    t = int(np.argmin(np.abs(ev.params.iouThrs - OP_IOU)))
    out: dict[int, _Hits] = {}
    for e in ev.evalImgs:
        if e is None:
            continue
        kept = np.asarray(e["dtScores"]) >= conf
        matched = np.asarray(e["dtMatches"][t])
        ignored = np.asarray(e["dtIgnore"][t], dtype=bool)
        tp_ids = matched[kept & ~ignored & (matched > 0)].astype(int)
        out[int(e["image_id"])] = (
            int(len(tp_ids)),
            int((kept & ~ignored & (matched == 0)).sum()),
            Counter(_bucket(coco_gt.anns[i]["visibility"]) for i in tp_ids),
        )
    return out


def _positives(coco_gt: COCO, img_ids: set[int] | None = None) -> Counter[int]:
    """Scored (non-ignored) ground-truth boxes per visibility bucket, all images or ``img_ids``."""
    gts = [g for g in coco_gt.anns.values() if img_ids is None or g["image_id"] in img_ids]
    return Counter(_bucket(g["visibility"]) for g in gts if not g["iscrowd"])


def _summary(
    ap: dict[str, float], per_img: list[_Hits], pos: Counter[int], n_frames: int
) -> Metrics:
    tp = sum(h[0] for h in per_img)
    fp = sum(h[1] for h in per_img)
    matched: Counter[int] = sum((h[2] for h in per_img), Counter())
    npos = sum(pos.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / npos if npos else 0.0
    return {
        "AP": ap["AP"],
        "AP50": ap["AP50"],
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": npos - tp,
        "gt": npos,
        "frames": n_frames,
        "recall_by_visibility": {
            VIS_LABELS[b]: (matched[b] / pos[b] if pos[b] else None)
            for b in range(len(VIS_LABELS))
        },
        "gt_by_visibility": {VIS_LABELS[b]: pos[b] for b in range(len(VIS_LABELS))},
    }


def evaluate(runs: dict[str, Run], conf: float | None = None) -> Evaluation:
    """Score one model's cached detections.

    Args:
        runs: Per sequence name: ground truth, MOT det rows and the frames that were run.
            Only those frames are evaluated.
        conf: Operating-point confidence for precision/recall/F1; default ``OP_CONF``.

    Returns:
        ``{"overall": metrics, "per_sequence": {name: metrics}}`` with COCO ``AP``/``AP50`` and the
        operating-point ``precision``/``recall``/``f1`` (+ counts and recall by GT visibility).
    """
    coco_gt = _coco_gt(runs)
    coco_dt = _coco_dt(coco_gt, runs)
    all_ids = sorted(coco_gt.imgs)
    ev = _coco_eval(coco_gt, coco_dt, all_ids)
    per_img = _operating_point(ev, coco_gt, conf)
    ap, pos = _ap(ev), _positives(coco_gt)
    per_sequence: dict[str, Metrics] = {}
    for si, (name, (_, _, frames)) in enumerate(runs.items()):
        ids = [si * IMAGE_STRIDE + f for f in frames]
        id_set = set(ids)
        per_sequence[name] = _summary(
            _ap(_coco_eval(coco_gt, coco_dt, ids)),
            [per_img[i] for i in ids if i in per_img],
            _positives(coco_gt, id_set),
            len(frames),
        )
    overall = _summary(ap, list(per_img.values()), pos, len(all_ids))
    # Confidence sweep from the same matching: the app's 0.3 was chosen for Darknet's scores.
    sweep = {
        c: _summary(ap, list(_operating_point(ev, coco_gt, c).values()), pos, len(all_ids))
        for c in CONF_GRID
    }
    best = max(sweep, key=lambda c: sweep[c]["f1"])
    overall["f1_by_conf"] = {f"{c:.2f}": sweep[c]["f1"] for c in CONF_GRID}
    overall["best_conf"] = best
    b = sweep[best]
    overall["at_best_conf"] = {"precision": b["precision"], "recall": b["recall"], "f1": b["f1"]}
    return {"overall": overall, "per_sequence": per_sequence}


def main(argv: list[str] | None = None) -> None:
    """Run the detectors on MOT17 train (cached) and write the metrics JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--sequences", nargs="+", default=list(mot17.SEQUENCES))
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--out", type=Path, default=Path("results/detection.json"))
    p.add_argument("--preds-dir", type=Path, default=Path("results/preds"))
    p.add_argument("--force", action="store_true", help="rerun detectors even when cached")
    a = p.parse_args(argv)

    seqs = [mot17.load_sequence(n) for n in a.sequences]
    gts = {s.name: mot17.load_gt(s) for s in seqs}
    # max_det in the key: a cache made under a lower box cap is truncated, not reusable
    key: dict[str, object] = {"stride": a.stride, "limit": a.limit, "max_det": MAX_DETS}
    models: dict[str, dict[str, object]] = {}
    for model in a.models:
        det: Detector | None = None
        runs: dict[str, Run] = {}
        for s in seqs:
            out = a.preds_dir / model / f"{s.name}.txt"
            if a.force or not mot17.cache_valid(out, key):
                det = det or load_detector(model, conf=CACHE_CONF, device=a.device)
                run_detector(det, s, a.stride, a.limit, out, extra_meta=key)
            runs[s.name] = (
                gts[s.name],
                mot17.read_mot(out),
                mot17.frame_numbers(s, a.stride, a.limit),
            )
        scored = evaluate(runs)
        seconds = sum(
            mot17.read_meta(a.preds_dir / model / f"{s.name}.txt")["seconds"] for s in seqs
        )
        models[model] = {
            **scored,
            "seconds_per_frame": seconds / sum(len(r[2]) for r in runs.values()),
        }
        o = scored["overall"]
        print(
            f"{model:16s} AP {o['AP']:.3f}  AP50 {o['AP50']:.3f}  "
            f"P {o['precision']:.3f}  R {o['recall']:.3f}  F1 {o['f1']:.3f}"
        )

    protocol = {
        "dataset": "MOT17 train, FRCNN sequences",
        "sequences": a.sequences,
        "stride": a.stride,
        "limit": a.limit,
        "frames_evaluated": {s.name: len(mot17.frame_numbers(s, a.stride, a.limit)) for s in seqs},
        "detector_conf": CACHE_CONF,
        "detector_note": "yolov3-darknet: conf is also its NMS score threshold (the Darknet "
        "baseline's own convention); Ultralytics models: imgsz 640, person class only",
        "coordinates": "MOT 1-based top-left pixels; detector x1,y1 shifted by +1 when cached",
        "clipping": "GT, ignore regions and detections are clipped to the frame: MOT17 "
        "annotates bodies that leave the image, Ultralytics clips its outputs, and the Darknet "
        "port clips after NMS, so every detector is scored under one box convention",
        "positives": "class 1 (pedestrian) with flag 1",
        "ignore_classes": list(mot17.IGNORE_CLASSES),
        "ignore_handling": "pycocotools iscrowd regions: a detection whose intersection over its "
        "own area with an ignore box reaches the IoU threshold is neither TP nor FP; other "
        "classes are dropped",
        "ap": f"pycocotools COCO AP@[.5:.95] and AP50, area range all, maxDets {MAX_DETS}, "
        "only frames that were run",
        "operating_point": {
            "conf": OP_CONF,
            "iou": OP_IOU,
            "matching": "pycocotools greedy by score at IoU 0.5 (same matching as AP); detections "
            f"below conf {OP_CONF} dropped afterwards, which equals matching only kept detections",
        },
        "visibility_buckets": list(VIS_LABELS),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"protocol": protocol, "models": models}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
