"""Build the README media: two short GIFs and two stills, from the datasets under ``data/``.

- WILDTRACK camera C1, test frames, calibrated metres with the bird's-eye inset
  (the calibration comes from the dataset's full camera model, as ``M4`` in the evaluation).
- MOT17-09, tracking on, the pixel rule scaled to the frame width.

Usage: ``python scripts/make_demo.py [--out docs] [--device 0]``.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from evaluation.wildtrack_data import ROOT as WILDTRACK  # noqa: E402
from evaluation.wildtrack_data import click_points, load_calibration  # noqa: E402
from sda.geometry import homography  # noqa: E402

MOT17_09 = Path("data/MOT17/train/MOT17-09-FRCNN/img1")
GIF_WIDTH = 560


def wildtrack_calibration(camera: str, out: Path) -> Path:
    """Write a ``sda calibrate``-style JSON from WILDTRACK's full calibration of ``camera``."""
    calib = load_calibration(camera)
    ground = click_points(calib)
    image = calib.project(ground)
    out.write_text(
        json.dumps(
            {
                "image_points": image.tolist(),
                "ground_points": ground.tolist(),
                "homography": homography(image, ground).tolist(),
            },
            indent=2,
        )
    )
    return out


def consecutive_copy(files: list[Path], tmp: Path) -> str:
    """Copy frames under consecutive names so OpenCV can read them as a sequence."""
    tmp.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        shutil.copy(f, tmp / f"{i:06d}{f.suffix}")
    return str(tmp / f"%06d{files[0].suffix}")


def run_sda(video: str, out_mp4: Path, extra: list[str], device: str) -> None:
    """Run the CLI once; raise if it fails."""
    cmd = [sys.executable, "-m", "sda", "run", video, "--device", device, "--out", str(out_mp4)]
    subprocess.run([*cmd, *extra], check=True)


def to_gif(mp4: Path, gif: Path, fps: float, every: int = 1) -> int:
    """Downscale an mp4 to a looping GIF; returns the number of frames written."""
    cap = cv2.VideoCapture(str(mp4))
    frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every == 0:
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (GIF_WIDTH, round(h * GIF_WIDTH / w)), cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    quantised = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in frames]
    quantised[0].save(
        gif, save_all=True, append_images=quantised[1:], duration=round(1000 / fps), loop=0
    )
    return len(frames)


def still(mp4: Path, jpg: Path, index: int) -> None:
    """Save one annotated frame as a JPEG."""
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame {index} in {mp4}")
    cv2.imwrite(str(jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(argv: list[str] | None = None) -> int:
    """Build the media."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs"))
    p.add_argument("--device", default="0")
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        # WILDTRACK: 30 test-split frames of C1 (2 fps in the dataset), played at 4 fps.
        frames = sorted((WILDTRACK / "Image_subsets" / "C1").glob("*.png"))
        frames = [f for f in frames if int(f.stem) >= 1000][:30]
        calib = wildtrack_calibration("C1", a.out / "calibration_wildtrack_c1.json")
        mp4 = tmp / "wildtrack.mp4"
        run_sda(
            consecutive_copy(frames, tmp / "wt"),
            mp4,
            ["--model", "yolo26s@1280", "--calib", str(calib), "--threshold", "2", "--fps", "4"],
            a.device,
        )
        n = to_gif(mp4, a.out / "demo_wildtrack_c1.gif", fps=4)
        still(mp4, a.out / "still_wildtrack_c1.jpg", 12)
        print(f"wildtrack: {n} frames")

        # MOT17-09: 90 frames at 30 fps, every third frame in the GIF at 10 fps; tracking on.
        mp4 = tmp / "mot17.mp4"
        run_sda(
            str(MOT17_09 / "%06d.jpg"),
            mp4,
            [
                "--model",
                "yolo26s@1280",
                "--track",
                "--tracker",
                "botsort.yaml",
                "--conf",
                "0.1",
                "--pixel-threshold",
                "50",
                "--pixel-width",
                "700",
                "--max-frames",
                "90",
                "--fps",
                "30",
            ],
            a.device,
        )
        n = to_gif(mp4, a.out / "demo_mot17_09.gif", fps=10, every=3)
        still(mp4, a.out / "still_mot17_09.jpg", 45)
        print(f"mot17: {n} frames")
    for f in sorted(a.out.iterdir()):
        print(f"{f.name:32} {f.stat().st_size / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
