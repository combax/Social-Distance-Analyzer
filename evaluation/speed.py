"""Per-frame speed of the full pipeline (detect, feet, ground projection, violations).

Usage: ``python -m evaluation.speed --models yolo26n yolo26s --devices cpu 0``.
"""

import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from evaluation.detection import MODELS
from sda.detect import load_detector
from sda.geometry import feet, to_ground, violating_pairs

FRAMES_DIR = (
    Path(__file__).resolve().parent.parent / "data" / "MOT17" / "train" / "MOT17-04-FRCNN" / "img1"
)


def _cpu_name() -> str:
    """Return a readable CPU name (``platform.processor()`` is a family/model code on Windows)."""
    if sys.platform == "win32":  # pragma: no cover  (registry read; the fallback is tested)
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    return platform.processor()


def bench(
    name: str, device: str, frames: list[NDArray[np.uint8]], warmup: int
) -> dict[str, object]:
    """Time the pipeline on each frame and summarise all but the first ``warmup``.

    Args:
        name: Detector name for :func:`sda.detect.load_detector`.
        device: ``"cpu"`` or a CUDA index such as ``"0"``.
        frames: Decoded BGR frames, warmup frames first.
        warmup: Number of leading frames to discard.

    Returns:
        ``model, device, frames, median_ms, p90_ms, fps`` for the timed frames.
    """
    det = load_detector(name, device=device)
    h = np.eye(3)
    ms = []
    for frame in frames:
        t = time.perf_counter()
        violating_pairs(to_ground(feet(det(frame).boxes), h), 2.0)
        ms.append((time.perf_counter() - t) * 1000)
    timed = np.array(ms[warmup:])
    median = float(np.median(timed))
    return {
        "model": name,
        "device": device,
        "frames": len(timed),
        "median_ms": median,
        "p90_ms": float(np.percentile(timed, 90)),
        "fps": 1000 / median,
    }


def main(argv: list[str] | None = None) -> int:
    """Benchmark every model x device and write the results JSON.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m evaluation.speed", description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--devices", nargs="+", default=["cpu", "0"])
    parser.add_argument("--frames", type=int, default=200, help="timed frames")
    parser.add_argument("--warmup", type=int, default=10, help="untimed frames run first")
    parser.add_argument("--out", default="results/speed.json")
    args = parser.parse_args(argv)

    paths = sorted(FRAMES_DIR.glob("*.jpg"))[: args.warmup + args.frames]
    frames = [np.asarray(cv2.imread(str(p)), np.uint8) for p in paths]
    if len(frames) <= args.warmup:
        print(f"{FRAMES_DIR} has {len(frames)} frames: is MOT17 downloaded?", file=sys.stderr)
        return 2
    cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0  # CUDA_VISIBLE_DEVICES=""
    results, skipped = [], []
    for name in args.models:
        for device in args.devices:
            reason = None
            if device != "cpu" and name == "yolov3-darknet":
                reason = "cv2.dnn baseline is CPU only"
            elif device != "cpu" and not cuda:
                reason = "torch.cuda is unavailable"
            if reason:
                skipped.append({"model": name, "device": device, "reason": reason})
                continue
            results.append(bench(name, device, frames, args.warmup))

    machine = {
        "cpu": _cpu_name(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__,
        "ultralytics": version("ultralytics"),
        "opencv": cv2.__version__,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"machine": machine, "results": results, "skipped": skipped}
    out.write_text(json.dumps(report, indent=2))

    print(f"{'model':16} {'device':6} {'median_ms':>9} {'p90_ms':>8} {'fps':>7}")
    for r in results:
        print(
            f"{r['model']:16} {r['device']:6} {r['median_ms']:9.1f} {r['p90_ms']:8.1f}"
            f" {r['fps']:7.1f}"
        )
    for s in skipped:
        print(f"{s['model']:16} {s['device']:6} skipped: {s['reason']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
