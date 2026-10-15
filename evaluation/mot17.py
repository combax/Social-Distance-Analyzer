"""MOT17 train split: sequences, frames and ground truth."""

import configparser
import json
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sda.detect import MODELS_DIR

READ_AHEAD = 4

ROOT = MODELS_DIR.parent / "MOT17" / "train"
SEQUENCES: tuple[str, ...] = tuple(
    f"MOT17-{n}-FRCNN" for n in ("02", "04", "05", "09", "10", "11", "13")
)
GT_COLUMNS = ("frame", "id", "x", "y", "w", "h", "flag", "class", "visibility")
PEDESTRIAN = 1
# Person on vehicle, static person, distractor, reflection: neither TP nor FP when hit.
IGNORE_CLASSES = (2, 7, 8, 12)


class Meta(TypedDict, total=False):
    """Sidecar of a cached MOT file: the cache key plus the run's ``frames`` and ``seconds``."""

    frames: int
    seconds: float


@dataclass(frozen=True, slots=True)
class Sequence:
    """One MOT17 sequence as described by its ``seqinfo.ini``."""

    name: str
    dir: Path
    fps: float
    width: int
    height: int
    length: int


def load_sequence(name: str) -> Sequence:
    """Read ``seqinfo.ini`` of a train sequence.

    Args:
        name: Sequence name such as ``MOT17-02-FRCNN``.

    Returns:
        The sequence metadata.
    """
    d = ROOT / name
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    s = ini["Sequence"]
    return Sequence(
        name, d, float(s["frameRate"]), int(s["imWidth"]), int(s["imHeight"]), int(s["seqLength"])
    )


def frame_numbers(seq: Sequence, stride: int = 1, limit: int | None = None) -> list[int]:
    """Return the 1-based frame numbers selected by ``stride`` and ``limit``."""
    return list(islice(range(1, seq.length + 1, stride), limit))


def frames(
    seq: Sequence, stride: int = 1, limit: int | None = None, numbers: list[int] | None = None
) -> Iterator[tuple[int, NDArray[np.uint8]]]:
    """Yield ``(frame_number, bgr_image)`` for the selected frames.

    Args:
        seq: Sequence to read.
        stride: Keep every ``stride``-th frame, starting at frame 1.
        limit: Stop after this many frames.
        numbers: Explicit 1-based frame numbers instead of the stride/limit selection.

    Yields:
        1-based frame number and the BGR uint8 image.
    """

    def read(n: int) -> NDArray[np.uint8]:
        img = cv2.imread(str(seq.dir / "img1" / f"{n:06d}.jpg"))
        if img is None:
            raise FileNotFoundError(f"{seq.name} frame {n}")
        return img.astype(np.uint8, copy=False)

    # Decode up to READ_AHEAD frames on a thread (cv2 releases the GIL) so the GPU is not left
    # waiting on JPEG decoding; bounded so memory stays flat.
    todo = iter(frame_numbers(seq, stride, limit) if numbers is None else numbers)
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = deque((n, pool.submit(read, n)) for n in islice(todo, READ_AHEAD))
        while pending:
            n, future = pending.popleft()
            if (nxt := next(todo, None)) is not None:
                pending.append((nxt, pool.submit(read, nxt)))
            yield n, future.result()


def load_gt(seq: Sequence) -> pd.DataFrame:
    """Read ``gt/gt.txt`` with named columns (``x, y`` are 1-based top-left pixels).

    Boxes are clipped to the frame. MOT17 annotates the full body even where it leaves the
    image (a quarter of MOT17-04's positives), and a detector that only sees the image can
    never reach IoU 0.5 on those; Ultralytics clips its outputs, so the ground truth must be.
    """
    gt = pd.read_csv(seq.dir / "gt" / "gt.txt", header=None, names=GT_COLUMNS)
    x2 = (gt["x"] + gt["w"]).clip(1, seq.width + 1)
    y2 = (gt["y"] + gt["h"]).clip(1, seq.height + 1)
    gt["x"] = gt["x"].clip(1, seq.width + 1)
    gt["y"] = gt["y"].clip(1, seq.height + 1)
    gt["w"] = x2 - gt["x"]
    gt["h"] = y2 - gt["y"]
    return gt


def write_mot(
    path: Path, rows: NDArray[np.float64], meta: dict[str, object] | None = None
) -> None:
    """Write a 10-column MOTChallenge txt file (``frame, id, x, y, w, h, score, -1, -1, -1``).

    Args:
        path: Output ``.txt``.
        rows: (N, 10) float rows.
        meta: When given, written as JSON next to ``path`` so ``cache_valid`` can check it later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").unlink(missing_ok=True)  # a crash mid-write must not look cached
    np.savetxt(path, rows.reshape(-1, 10), fmt="%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d")
    if meta is not None:
        path.with_suffix(".json").write_text(json.dumps(meta))


def read_meta(path: Path) -> Meta:
    """Read the JSON sidecar written by ``write_mot``."""
    data: Meta = json.loads(path.with_suffix(".json").read_text())
    return data


def cache_valid(path: Path, key: dict[str, object]) -> bool:
    """Return True when ``path`` exists and its sidecar matches every item of ``key``."""
    if not (path.exists() and path.with_suffix(".json").exists()):
        return False
    meta = read_meta(path)
    return all(k in meta and meta.get(k) == v for k, v in key.items())  # absent != None


def read_mot(path: Path) -> NDArray[np.float64]:
    """Read a 10-column MOTChallenge txt file, (0, 10) when empty."""
    return np.loadtxt(path, delimiter=",", ndmin=2, dtype=np.float64).reshape(-1, 10)


def to_mot_xywh(boxes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert detector ``x1, y1, x2, y2`` (0-based pixels) to MOT ``x, y, w, h`` (1-based)."""
    xy = boxes[:, :2] + 1
    return np.column_stack((xy, boxes[:, 2:] - boxes[:, :2]))
