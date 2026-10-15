"""Multi-object tracking on MOT17: Ultralytics ByteTrack / BoT-SORT scored with trackeval."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

# trackeval 1.x still uses the numpy 1.x aliases removed in numpy 2 (mot_challenge_2d_box.py,
# hota.py, identity.py). Restore them before importing it instead of patching site-packages.
for _alias, _type in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _alias):
        setattr(np, _alias, _type)

from trackeval import Evaluator  # noqa: E402
from trackeval.datasets import MotChallenge2DBox  # noqa: E402
from trackeval.metrics import CLEAR, HOTA, Identity  # noqa: E402

from evaluation import mot17  # noqa: E402
from sda.detect import MAX_DET, UltralyticsYolo, load_detector  # noqa: E402

MODELS = ("yolo11s", "yolo26n", "yolo26s", "yolo26m")
TRACKERS = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}
CONF = 0.1  # ByteTrack's low-score association needs detections below the usual threshold
GT_DIR = "_trackeval"  # under --tracks-dir: truncated GT copies in trackeval's layout
FIELDS = {
    "HOTA": ("HOTA", "HOTA"),
    "DetA": ("HOTA", "DetA"),
    "AssA": ("HOTA", "AssA"),
    "MOTA": ("CLEAR", "MOTA"),
    "IDF1": ("Identity", "IDF1"),
    "IDSW": ("CLEAR", "IDSW"),
    "FP": ("CLEAR", "CLR_FP"),
    "FN": ("CLEAR", "CLR_FN"),
}


def tracking_model(name: str, conf: float, device: str) -> UltralyticsYolo:
    """Load ``name`` (``yolo26s``, ``yolo26s@1280``, a ``.pt`` path) for tracking."""
    det = load_detector(name, conf=conf, device=device)
    if not isinstance(det, UltralyticsYolo):
        raise SystemExit(f"--models {name}: tracking needs an Ultralytics model")
    return det


def run_tracker(
    det: UltralyticsYolo,
    tracker: str,
    seq: mot17.Sequence,
    limit: int | None,
    out: Path,
    meta: dict[str, object],
) -> None:
    """Track every selected frame of ``seq`` and write a MOTChallenge tracker file plus sidecar.

    Args:
        det: Ultralytics detector; its tracker state is reset first.
        tracker: Ultralytics tracker config file name.
        seq: Sequence to run.
        limit: Stop after this many frames.
        out: Tracker ``.txt`` (``frame, id, x, y, w, h, 1, -1, -1, -1``; 1-based coordinates).
        meta: Cache key written to the sidecar together with ``frames`` and ``seconds``.
    """
    det.reset_tracker()
    rows, seconds = [], 0.0
    for n, img in mot17.frames(seq, limit=limit):
        t0 = time.perf_counter()
        d = det.track(img, tracker=tracker)
        seconds += time.perf_counter() - t0
        if d.ids is None or len(d) == 0:
            continue
        pad = np.full((len(d), 1), -1.0)
        rows.append(
            np.column_stack(
                (
                    np.full(len(d), n),
                    d.ids,
                    mot17.to_mot_xywh(d.boxes),
                    np.ones(len(d)),
                    pad,
                    pad,
                    pad,
                )
            )
        )
    n_frames = len(mot17.frame_numbers(seq, limit=limit))
    meta = {**meta, "frames": n_frames, "seconds": seconds}
    mot17.write_mot(out, np.concatenate(rows) if rows else np.empty((0, 10)), meta)


def score(tracks_dir: Path, tracker: str, lengths: dict[str, int]) -> dict[str, dict[str, float]]:
    """Score ``tracks_dir/<tracker>/data/<seq>.txt`` with trackeval (MOT17 train protocol).

    Ground truth is copied to ``tracks_dir/_trackeval/<seq>/gt/gt.txt`` truncated to the first
    ``lengths[seq]`` frames, so only frames that were run are evaluated. Standard MOTChallenge
    preprocessing applies: tracks matched to distractor classes are removed and only pedestrians
    with flag 1 count.

    Args:
        tracks_dir: Folder holding one sub-folder per tracker.
        tracker: Tracker sub-folder name.
        lengths: Frames run per sequence name.

    Returns:
        Per sequence name plus ``COMBINED_SEQ``: HOTA, DetA, AssA, MOTA, IDF1, IDSW, FP, FN.
    """
    gt_root = tracks_dir / GT_DIR
    for name, n in lengths.items():
        gt = mot17.load_gt(mot17.load_sequence(name))
        path = gt_root / name / "gt" / "gt.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        gt[gt["frame"] <= n].to_csv(path, header=False, index=False)
    dataset = MotChallenge2DBox(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(tracks_dir),
            "TRACKERS_TO_EVAL": [tracker],
            "OUTPUT_FOLDER": str(gt_root / "out"),
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": dict(lengths),
            "DO_PREPROC": True,
            "PRINT_CONFIG": False,
        }
    )
    evaluator = Evaluator(
        {
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "OUTPUT_SUMMARY": False,
            "OUTPUT_DETAILED": False,
            "OUTPUT_EMPTY_CLASSES": False,
            "PLOT_CURVES": False,
            "LOG_ON_ERROR": None,
        }
    )
    quiet = {"PRINT_CONFIG": False}
    res, _ = evaluator.evaluate([dataset], [HOTA(), CLEAR(quiet), Identity(quiet)])
    per_seq = res["MotChallenge2DBox"][tracker]
    return {
        seq: {k: float(np.mean(r["pedestrian"][m][f])) for k, (m, f) in FIELDS.items()}
        for seq, r in per_seq.items()
    }


def main(argv: list[str] | None = None) -> None:
    """Track MOT17 train with each model and tracker, score with trackeval, write JSON."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--trackers", nargs="+", default=list(TRACKERS), choices=list(TRACKERS))
    p.add_argument("--sequences", nargs="+", default=list(mot17.SEQUENCES))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default="0")
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--out", type=Path, default=Path("results/tracking.json"))
    p.add_argument("--tracks-dir", type=Path, default=Path("results/tracks"))
    p.add_argument("--force", action="store_true", help="rerun trackers even when cached")
    a = p.parse_args(argv)

    seqs = [mot17.load_sequence(n) for n in a.sequences]
    lengths = {s.name: len(mot17.frame_numbers(s, limit=a.limit)) for s in seqs}
    key: dict[str, object] = {"limit": a.limit, "conf": a.conf, "max_det": MAX_DET}
    runs: dict[str, dict[str, object]] = {}
    for model in a.models:
        for tracker in a.trackers:
            name = f"{model}-{tracker}"
            det: UltralyticsYolo | None = None
            for s in seqs:
                out = a.tracks_dir / name / "data" / f"{s.name}.txt"
                if a.force or not mot17.cache_valid(out, key):
                    det = det or tracking_model(model, a.conf, a.device)
                    run_tracker(det, TRACKERS[tracker], s, a.limit, out, key)
            metrics = score(a.tracks_dir, name, lengths)
            per_sequence = {}
            seconds = 0.0
            for s in seqs:
                meta = mot17.read_meta(a.tracks_dir / name / "data" / f"{s.name}.txt")
                seconds += meta["seconds"]
                per_sequence[s.name] = {**metrics[s.name], "fps": meta["frames"] / meta["seconds"]}
            overall = {**metrics["COMBINED_SEQ"], "fps": sum(lengths.values()) / seconds}
            runs[name] = {
                "model": model,
                "tracker": tracker,
                "overall": overall,
                "per_sequence": per_sequence,
            }
            print(
                f"{name:20s} HOTA {overall['HOTA']:.3f}  MOTA {overall['MOTA']:.3f}  "
                f"IDF1 {overall['IDF1']:.3f}  IDSW {overall['IDSW']:.0f}  {overall['fps']:.1f} fps"
            )

    protocol = {
        "dataset": "MOT17 train, FRCNN sequences",
        "sequences": a.sequences,
        "limit": a.limit,
        "frames_evaluated": lengths,
        "detector_conf": a.conf,
        "detector_note": "Ultralytics models, imgsz 640, person class only; conf 0.1 so "
        "ByteTrack's low-score second association gets candidates",
        "trackers": {t: f"Ultralytics {cfg} defaults" for t, cfg in TRACKERS.items()},
        "frame_rate_note": "Ultralytics trackers assume 30 fps for array input (track buffer of "
        "30 frames); MOT17-05 is 14 fps and MOT17-13 is 25 fps",
        "coordinates": "MOT 1-based top-left pixels; detector x1,y1 shifted by +1 when written",
        "scoring": "trackeval MotChallenge2DBox, BENCHMARK MOT17, SPLIT train, DO_PREPROC True "
        "(tracks matched to classes 2/7/8/12 removed; GT restricted to class 1 flag 1), GT "
        "truncated to the frames run; HOTA/DetA/AssA averaged over the 19 alpha thresholds",
        "metrics": list(FIELDS),
        "fps": "frames / wall time of UltralyticsYolo.track (detector + tracker, incl. warm-up), "
        "JPEG decode excluded",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"protocol": protocol, "runs": runs}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
