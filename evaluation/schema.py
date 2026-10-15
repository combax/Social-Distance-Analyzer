"""Shapes of the parts of ``results/*.json`` that the report reads.

Only the keys ``evaluation.report`` indexes are declared, nested only as deep as it reaches;
``total=False`` marks sections whose keys depend on which benchmark phases were run.
"""

from typing import NotRequired, TypedDict


class MeanStd(TypedDict):
    """A statistic over seeds: ``{"mean", "std"}``."""

    mean: float
    std: float


class Scores(TypedDict):
    """Precision, recall and F1, each a float or a mean and std over seeds."""

    precision: float | MeanStd
    recall: float | MeanStd
    f1: float | MeanStd


class DetOverall(TypedDict):
    """One detector's whole-dataset detection metrics."""

    AP: float
    AP50: float
    precision: float
    recall: float
    f1: float
    best_conf: float
    at_best_conf: dict[str, float]
    recall_by_visibility: dict[str, float | None]


class DetModel(TypedDict):
    """Detection results for one detector."""

    overall: DetOverall


class DetProtocol(TypedDict):
    """How the MOT17 detection benchmark was run."""

    sequences: list[str]
    frames_evaluated: dict[str, int]
    positives: str
    ignore_handling: str


class Detection(TypedDict):
    """``detection.json``."""

    protocol: DetProtocol
    models: dict[str, DetModel]


class Scored(TypedDict):
    """Tracking metrics by name (HOTA, MOTA, IDF1, ...)."""

    overall: dict[str, float]


class Run(Scored):
    """One detector and tracker combination."""

    model: str
    tracker: str


class Tracking(TypedDict):
    """``tracking.json``."""

    runs: dict[str, Run]


class DistanceError(TypedDict):
    """Bird's-eye distance error statistics over ground-truth pairs."""

    mae: float
    median: float
    p90: float
    mae_by_range_m: dict[str, float]
    position_median: float
    feet_off_frame_excluded: int


Flags = dict[str, dict[str, Scores]]  # threshold in metres -> camera or "all" -> scores


class Method(TypedDict, total=False):
    """One distance method's scores; ``distance`` is M4 only, the count keys are phase B only."""

    pair: Flags
    person: Flags
    distance: DistanceError
    count_mae: float | MeanStd
    count_mae_all_flags: float | MeanStd
    unverifiable_flagged: float | MeanStd


class PhaseA(TypedDict, total=False):
    """Geometry-only methods on ground-truth boxes; ``M3`` is keyed by click-noise sigma."""

    M0: Method
    M1: Method
    M2: Method
    M3: dict[str, Method]
    M4: Method


class DistProtocol(TypedDict):
    """How the WILDTRACK benchmark was run."""

    headline_m: float


class Distance(TypedDict):
    """``distance.json``."""

    protocol: DistProtocol
    reprojection_px: dict[str, float]
    phase_a: PhaseA
    phase_b: dict[str, Method]


class SpeedResult(TypedDict):
    """One model on one device."""

    model: str
    device: str
    fps: float


class Speed(TypedDict):
    """``speed.json``."""

    machine: dict[str, str | None]
    results: list[SpeedResult]


class Split(TypedDict):
    """Sequences used for training and for the held-out test."""

    train_sequences: list[str]
    test_sequences: list[str]


class FtProtocol(TypedDict):
    """How the fine-tuning comparison was run."""

    split: Split
    imgsz: int


class Training(TypedDict, total=False):
    """The training record; ``epochs_run`` and ``batch_size`` read ``"n/a"`` when it is absent."""

    epochs_run: int | str
    batch_size: int | str
    wall_seconds: float
    segments: int
    resumed_from_last_pt: bool
    epochs_planned: int
    stopped_early: str
    val_map50_95_by_epoch: list[float]


class FtModel(TypedDict):
    """Zero-shot or fine-tuned model scored on the held-out sequences."""

    detection: DetModel
    tracking: Scored
    best_conf_on_val: float
    test_at_best_conf: dict[str, float]


class Finetune(TypedDict):
    """``finetune.json``."""

    protocol: FtProtocol
    training: NotRequired[Training]
    models: dict[str, FtModel]
