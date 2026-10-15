"""Command-line app: ``python -m sda calibrate`` once per camera, then ``python -m sda run``."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from sda.detect import Detections, UltralyticsYolo, load_detector
from sda.geometry import centroids, feet, homography, to_ground, violating_pairs

GREEN = (0, 200, 0)
RED = (0, 0, 255)
Calib = dict[str, list[list[float]]]  # what ``calibrate`` writes


def _points(text: str) -> list[list[float]]:
    """Parse ``"x1,y1;x2,y2;..."`` into a list of ``[x, y]``."""
    return [[float(v) for v in p.split(",")] for p in text.split(";")]


def _open(video: str) -> cv2.VideoCapture:
    """Open a video file, image-sequence pattern, or integer webcam index."""
    cap = cv2.VideoCapture(int(video) if video.isdigit() else video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video {video!r}")
    return cap


def _click(frame: NDArray[np.uint8], n: int) -> list[list[float]]:
    """Collect ``n`` left-clicks on ``frame`` in a window, drawing each one."""
    pts: list[list[float]] = []

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append([float(x), float(y)])
            cv2.circle(frame, (x, y), 5, RED, -1)

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while len(pts) < n:
        cv2.imshow("calibrate", frame)
        if cv2.waitKey(20) == 27:  # Esc
            raise SystemExit("calibration cancelled")
    cv2.destroyWindow("calibrate")
    return pts


def calibrate(args: argparse.Namespace) -> int:
    """Fit the image-to-ground homography and write it as JSON."""
    ground = _points(args.ground)
    if args.image_points:
        image = _points(args.image_points)
    else:
        ok, frame = _open(args.video).read()
        if not ok:
            raise SystemExit(f"cannot read a frame from {args.video!r}")
        image = _click(np.asarray(frame, np.uint8), len(ground))
    h = homography(image, ground)
    calib = {"image_points": image, "ground_points": ground, "homography": h.tolist()}
    Path(args.out).write_text(json.dumps(calib, indent=2))
    print(f"wrote {args.out}")
    return 0


def _birdseye(
    ground: NDArray[np.float64], bad: NDArray[np.bool_], area: NDArray[np.float64], size: int
) -> NDArray[np.uint8]:
    """Draw ground points as dots on a white square, scaled to fit them and ``area``."""
    inset = np.full((size, size, 3), 255, np.uint8)
    pts = np.vstack((area, ground))
    lo = pts.min(0)
    span = max(float((pts.max(0) - lo).max()), 1e-9)
    px = ((ground - lo) / span * (size - 20) + 10).astype(int)
    for (x, y), b in zip(px, bad, strict=True):
        cv2.circle(inset, (int(x), int(y)), 4, RED if b else GREEN, -1)
    cv2.rectangle(inset, (0, 0), (size - 1, size - 1), (0, 0, 0), 1)
    return inset


def _measure(
    args: argparse.Namespace, d: Detections, frame: NDArray[np.uint8], calib: Calib | None
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Reference points and the index pairs closer than the threshold, in metres or pixels."""
    if calib:
        ref = to_ground(feet(d.boxes), calib["homography"])
        return ref, violating_pairs(ref, args.threshold)
    ref = centroids(d.boxes)
    scale = frame.shape[1] / args.pixel_width if args.pixel_width else 1.0
    return ref, violating_pairs(ref, args.pixel_threshold * scale)


def _annotate(
    frame: NDArray[np.uint8],
    d: Detections,
    ref: NDArray[np.float64],
    pairs: NDArray[np.intp],
    calib: Calib | None,
) -> int:
    """Draw boxes, ids, pair lines, the count and the bird's-eye inset; return how many violate."""
    bad = np.zeros(len(d), bool)
    bad[pairs.ravel()] = True
    c = centroids(d.boxes).astype(int)
    for k, (x1, y1, x2, y2) in enumerate(d.boxes.astype(int)):
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED if bad[k] else GREEN, 2)
        if d.ids is not None:
            cv2.putText(frame, str(d.ids[k]), (x1, y1 - 5), 0, 0.6, GREEN, 2)
    for i, j in pairs:
        cv2.line(frame, tuple(c[i]), tuple(c[j]), RED, 2)
    cv2.putText(frame, f"Violations: {bad.sum()}", (10, 30), 0, 1.0, RED, 2)
    if calib:
        size = min(frame.shape[1] // 4, frame.shape[0])  # panoramas: never taller than the frame
        area = np.asarray(calib["ground_points"], np.float64)
        frame[:size, -size:] = _birdseye(ref, bad, area, size)
    return int(bad.sum())


def _writer(out: str, frame: NDArray[np.uint8], fps: float) -> cv2.VideoWriter:
    """Open the mp4 writer for ``out`` at the frame's size, creating its folder."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    hw = (frame.shape[1], frame.shape[0])
    writer = cv2.VideoWriter(out, cv2.VideoWriter.fourcc(*"mp4v"), fps, hw)
    if not writer.isOpened():
        raise SystemExit(f"cannot open video writer for {out!r} (use .mp4)")
    return writer


def run(args: argparse.Namespace) -> int:
    """Detect people, flag distance violations, draw them and summarise."""
    det = load_detector(args.model, conf=args.conf, device=args.device)
    if args.track and not isinstance(det, UltralyticsYolo):
        raise SystemExit(f"--track needs an Ultralytics model, not {args.model!r}")
    calib = json.loads(Path(args.calib).read_text()) if args.calib else None
    cap = _open(args.video)
    # OpenCV reports 25 fps for image sequences whatever the real rate: let --fps override.
    fps = args.fps or (30.0 if "%" in str(args.video) else cap.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = None
    frames = people = violators = 0
    start = time.perf_counter()
    while args.max_frames is None or frames < args.max_frames:
        ok, raw = cap.read()
        if not ok:
            break
        frame = np.asarray(raw, np.uint8)
        d = det.track(frame, args.tracker) if args.track else det(frame)  # type: ignore[attr-defined]
        ref, pairs = _measure(args, d, frame, calib)
        frames += 1
        people += len(d)
        violators += _annotate(frame, d, ref, pairs, calib)
        if args.out:
            writer = writer or _writer(args.out, frame, fps)
            writer.write(frame)
        if args.show:
            cv2.imshow("sda", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    elapsed = time.perf_counter() - start
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    n = max(frames, 1)
    print(
        f"frames={frames} people/frame={people / n:.2f} violators/frame={violators / n:.2f}"
        f" fps={frames / max(elapsed, 1e-9):.1f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the chosen subcommand.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m sda", description=__doc__)
    sub = parser.add_subparsers(required=True)

    cal = sub.add_parser("calibrate", help="fit the image-to-ground homography")
    cal.add_argument("video", help="video file, image pattern or webcam index")
    cal.add_argument("--ground", required=True, help='ground metres "x1,y1;...;x4,y4"')
    cal.add_argument("--image-points", help='pixels "u1,v1;...;u4,v4" (default: click)')
    cal.add_argument("--out", default="calib.json")
    cal.set_defaults(func=calibrate)

    r = sub.add_parser("run", help="detect people and flag distance violations")
    r.add_argument("video", help="video file, image pattern or webcam index")
    r.add_argument("--model", default="yolo26s")
    r.add_argument("--device", help='"cpu", "0", ... (default: best available)')
    r.add_argument("--conf", type=float, default=0.3)
    r.add_argument("--calib", help="calibration JSON from the calibrate command")
    r.add_argument("--threshold", type=float, default=2.0, help="metres, with --calib")
    r.add_argument("--pixel-threshold", type=float, default=50, help="pixels, without --calib")
    r.add_argument(
        "--pixel-width",
        type=int,
        help="frame width the pixel threshold refers to (the 50 px rule is defined at 700)",
    )
    r.add_argument("--track", action="store_true", help="track people and show their ids")
    r.add_argument(
        "--tracker",
        choices=["bytetrack.yaml", "botsort.yaml"],
        default="bytetrack.yaml",
        help="with --track; botsort scores higher HOTA, bytetrack is faster",
    )
    r.add_argument("--out", help="output mp4")
    r.add_argument("--fps", type=float, help="output fps (default: the input's; 30 for images)")
    r.add_argument("--show", action="store_true", help="show a window; q quits")
    r.add_argument("--max-frames", type=int)
    r.set_defaults(func=run)

    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
