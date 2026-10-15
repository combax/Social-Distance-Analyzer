"""Distance and violation accuracy on WILDTRACK: the pixel rule vs bird's-eye homographies.

Usage: ``python -m evaluation.wildtrack --phase A B --cameras C1 C4 --limit-frames 10``.
"""

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from evaluation.figstyle import axes, render
from evaluation.wildtrack_data import (
    CAMERAS,
    CLICK_GRID_M,
    CLICK_MARGIN_PX,
    DETS_DIR,
    GROUND_X,
    GROUND_Y,
    IMAGE_SIZE,
    MATCH_IOU,
    RULE_THRESHOLD_PX,
    RULE_WIDTH_PX,
    THRESHOLDS_M,
    TUNE_CANDIDATES_PX,
    BoolArray,
    CachedDetector,
    Calibration,
    Method,
    View,
    base_methods,
    click_method,
    click_points,
    in_roi,
    load_calibration,
    load_views,
    match,
    reprojection_errors,
)
from sda.detect import rel
from sda.geometry import FloatArray, IntArray, centroids, violating_pairs

TUNE_FRAMES, TEST_FRAMES = range(0, 1000, 5), range(1000, 2000, 5)
HEADLINE_M = 2.0
CLICK_SIGMAS_PX, CLICK_SEEDS = (0.0, 2.0, 5.0, 10.0), range(20)
PHASE_B_SIGMA_PX = 2.0
MAX_PAIR_M, RANGE_BIN_M = 5.0, 5.0
DETECTORS = {"yolov3-darknet": 0.3, "yolo26s": 0.3}
PHASE_B_RUNS = (("yolov3-darknet", "M0"), ("yolo26s", "M2"), ("yolo26s", "M3"), ("yolo26s", "M4"))
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

Record = dict[str, object]
Draw = Callable[[dict[str, str]], Figure]
Samples = tuple[FloatArray, FloatArray, FloatArray, int]


def pair_mask(pairs: IntArray, n: int) -> BoolArray:
    """Boolean over the ``n (n - 1) / 2`` pairs in ``pdist`` / ``triu`` order."""
    m = np.zeros((n, n), bool)
    m[pairs[:, 0], pairs[:, 1]] = True
    return np.asarray(m[np.triu_indices(n, 1)])


def flagged(pairs: IntArray, n: int) -> BoolArray:
    """Boolean over people: in at least one violating pair."""
    f = np.zeros(n, bool)
    f[pairs.ravel()] = True
    return f


def confusion(pred: BoolArray, truth: BoolArray) -> NDArray[np.int64]:
    """Return ``tp, fp, fn`` counts."""
    return np.array([(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()])


def prf(counts: NDArray[np.int64]) -> dict[str, float]:
    """Precision, recall and F1 from ``tp, fp, fn``."""
    tp, fp, fn = (int(c) for c in counts)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def summarise(counts: dict[str, NDArray[np.int64]]) -> dict[str, dict[str, float]]:
    """Overall (summed) and per-camera P/R/F1."""
    total = np.sum(list(counts.values()), axis=0).astype(np.int64)
    return {"all": prf(total), **{c: prf(v) for c, v in counts.items()}}


def _at(record: Mapping[str, object], *keys: str) -> Record:
    """Descend ``keys`` through a nested record."""
    node: object = record
    for k in keys:
        assert isinstance(node, dict)
        node = node[k]
    assert isinstance(node, dict)
    return node


def _num(record: Mapping[str, object], *keys: str) -> float:
    """A numeric leaf of a record; a ``mean``/``std`` leaf (M3 over seeds) gives its mean."""
    x = _at(record, *keys[:-1])[keys[-1]]
    x = x["mean"] if isinstance(x, dict) else x
    assert isinstance(x, int | float)
    return float(x)


def _distance_samples(est: FloatArray, v: View, centre: FloatArray) -> Samples:
    """Pair distance errors, pair ranges, position errors of one view, and the excluded count.

    Only foot points inside the image count: a GT box cut by the frame bottom puts the "feet"
    metres from where the person stands.
    """
    n = len(v.world)
    ok = np.ones(n, bool) if v.feet_in_frame is None else v.feet_in_frame
    true_d = pdist(v.world)
    i, j = np.triu_indices(n, 1)
    near = (true_d < MAX_PAIR_M) & ok[i] & ok[j]
    err = np.abs(pdist(est) - true_d)[near]
    rng = np.linalg.norm((v.world[i] + v.world[j]) / 2 - centre, axis=1)[near]
    return err, rng, np.linalg.norm(est - v.world, axis=1)[ok], int((~ok).sum())


def _distance_summary(samples: list[Samples]) -> Record:
    err, rng, pos, off_frame = zip(*samples, strict=True)
    e, r, p = (np.concatenate(x) for x in (err, rng, pos))
    bins = (r // RANGE_BIN_M).astype(int)
    return {
        "pairs": int(len(e)),
        "mae": float(e.mean()),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "mae_by_range_m": {
            f"{b * RANGE_BIN_M:g}-{(b + 1) * RANGE_BIN_M:g}": float(e[bins == b].mean())
            for b in np.unique(bins)
        },
        "position_median": float(np.median(p)),
        "position_p90": float(np.percentile(p, 90)),
        "feet_off_frame_excluded": sum(off_frame),
    }


def evaluate_geometry(method: Method, views: list[View], calibs: dict[str, Calibration]) -> Record:
    """Phase A: violation metrics (and distance errors for metric methods) on GT boxes."""
    cams = sorted({v.camera for v in views})
    pair = {t: {c: np.zeros(3, np.int64) for c in cams} for t in THRESHOLDS_M}
    person = {t: {c: np.zeros(3, np.int64) for c in cams} for t in THRESHOLDS_M}
    samples: list[Samples] = []
    for v in views:
        n = len(v.world)
        est = method.points(v.camera, v.boxes)
        for t in THRESHOLDS_M:
            truth = violating_pairs(v.world, t)
            pred = violating_pairs(est, method.threshold(v.camera, t))
            pair[t][v.camera] += confusion(pair_mask(pred, n), pair_mask(truth, n))
            person[t][v.camera] += confusion(flagged(pred, n), flagged(truth, n))
        if method.metric and n:
            samples.append(_distance_samples(est, v, calibs[v.camera].centre))
    out: Record = {
        "pair": {f"{t:.1f}": summarise(pair[t]) for t in THRESHOLDS_M},
        "person": {f"{t:.1f}": summarise(person[t]) for t in THRESHOLDS_M},
    }
    if method.metric:
        out["distance"] = _distance_summary(samples)
    return out


def mean_std(runs: list[Record]) -> Record:
    """Replace every numeric leaf with its mean and std over the runs."""
    out: Record = {}
    for key, value in runs[0].items():
        if isinstance(value, dict):
            out[key] = mean_std([_at(r, key) for r in runs])
        else:
            vals = [_num(r, key) for r in runs]
            out[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1))}
    return out


def phase_a(
    base: dict[str, Method], calibs: dict[str, Calibration], test_views: list[View]
) -> Record:
    """Run every method on GT boxes; M3 is averaged over seeds per noise level."""
    out: Record = {k: evaluate_geometry(m, test_views, calibs) for k, m in base.items()}
    m3 = {s: [click_method(calibs, s, seed) for seed in CLICK_SEEDS] for s in CLICK_SIGMAS_PX}
    out["M3"] = {
        f"sigma_{s:g}": mean_std([evaluate_geometry(m, test_views, calibs) for m in ms])
        for s, ms in m3.items()
    }
    out["thresholds_px"] = {
        k: {f"{t:.1f}": v for t, v in base[k].thresholds.items()}  # type: ignore[union-attr]
        for k in ("M1", "M2")
    }
    out["click_points_m"] = {c: click_points(k).tolist() for c, k in calibs.items()}
    return out


def evaluate_end_to_end(
    method: Method,
    detector: CachedDetector,
    views: list[View],
    calibs: dict[str, Calibration],
    t: float = HEADLINE_M,
) -> Record:
    """Phase B: person-level violation metrics on detector boxes matched to GT people.

    WILDTRACK annotates only people on its 12 x 36 m ground area, and not exhaustively, so
    flags are computed over detections that are matched or stand on that area (the people a
    deployment would see there), but only detections matched to an annotated person are
    scored. Flagged unmatched detections cannot be verified and are counted separately.
    """
    counts = {c: np.zeros(3, np.int64) for c in sorted({v.camera for v in views})}
    count_err: list[int] = []
    count_err_all: list[int] = []
    unverifiable = 0
    for v in views:
        dets = detector(v.camera, v.frame)
        m = match(dets, v.boxes)
        keep = (m >= 0) | in_roi(calibs[v.camera], dets)
        dets, m = dets[keep], m[keep]
        est = method.points(v.camera, dets)
        pred = flagged(violating_pairs(est, method.threshold(v.camera, t)), len(dets))
        truth = flagged(violating_pairs(v.world, t), len(v.world))
        hit = np.zeros(len(dets), bool)  # matched to a GT person in violation
        hit[m >= 0] = truth[m[m >= 0]]
        scored = pred & (m >= 0)
        tp = int((scored & hit).sum())
        counts[v.camera] += (tp, int((scored & ~hit).sum()), int(truth.sum()) - tp)
        count_err.append(abs(int(scored.sum()) - int(truth.sum())))
        count_err_all.append(abs(int(pred.sum()) - int(truth.sum())))
        unverifiable += int((pred & (m < 0)).sum())
    return {
        "detector": detector.name,
        "method": method.name,
        "person": {f"{t:.1f}": summarise(counts)},
        "count_mae": float(np.mean(count_err)),
        "count_mae_all_flags": float(np.mean(count_err_all)),
        "unverifiable_flagged": unverifiable,
    }


def phase_b(
    base: dict[str, Method], calibs: dict[str, Calibration], test_views: list[View], device: str
) -> Record:
    """Run the end-to-end configurations, caching detections under ``DETS_DIR``."""
    detectors = {n: CachedDetector(n, conf, device) for n, conf in DETECTORS.items()}
    keys = ("person", "count_mae", "count_mae_all_flags", "unverifiable_flagged")
    out: Record = {}
    for d, m in PHASE_B_RUNS:
        if m != "M3":
            out[f"{d}+{m}"] = evaluate_end_to_end(base[m], detectors[d], test_views, calibs)
            continue
        clicks = [click_method(calibs, PHASE_B_SIGMA_PX, seed) for seed in CLICK_SEEDS]
        runs = [evaluate_end_to_end(c, detectors[d], test_views, calibs) for c in clicks]
        scores = mean_std([{k: r[k] for k in keys} for r in runs])
        out[f"{d}+{m}"] = {"detector": d, "method": m, **scores}
    return out


def _scatter(views: list[View], calibs: dict[str, Calibration]) -> Draw:
    """Centroid pixel distance vs true distance in the camera with the largest depth range."""
    ranges = {
        c: [np.linalg.norm(v.world - k.centre, axis=1) for v in views if v.camera == c]
        for c, k in calibs.items()
    }
    cam = max(ranges, key=lambda c: np.ptp(np.concatenate(ranges[c])))
    cam_views = [v for v in views if v.camera == cam and len(v.world) > 1]
    px = np.concatenate([pdist(centroids(v.boxes)) for v in cam_views])
    m = np.concatenate([pdist(v.world) for v in cam_views])
    rule_px = RULE_THRESHOLD_PX * IMAGE_SIZE[0] / RULE_WIDTH_PX

    def draw(c: dict[str, str]) -> Figure:
        fig: Figure
        fig, ax = axes(7, 4.5, grid="y")
        ax.scatter(m, px, s=5, alpha=0.3, color=PALETTE[0], edgecolors="none")
        ax.axhline(rule_px, color=PALETTE[1], lw=2)
        ax.axvline(HEADLINE_M, color=PALETTE[2], lw=2)
        label = "pixel rule: 50 px at 700 wide"
        ax.text(m.max() * 0.98, rule_px * 1.6, label, color=PALETTE[1], ha="right")
        ax.text(HEADLINE_M + 0.2, px.max() * 0.9, f"{HEADLINE_M:g} m", color=PALETTE[2])
        ax.set(xlabel="true distance between pair (m)", ylabel="pixel distance between centroids")
        ax.set_title(f"Pixel distance vs true distance, camera {cam} (test split)")
        return fig

    return draw


def _mae_vs_range(a: Record) -> Draw:
    """Bird's-eye distance MAE per range bin for M4 and each M3 noise level."""
    paths: dict[str, tuple[str, ...]] = {"M4 full calibration": ("M4",)}
    paths |= {f"M3 4 clicks, sigma {s:g} px": ("M3", f"sigma_{s:g}") for s in CLICK_SIGMAS_PX}
    series = []
    for i, (label, path) in enumerate(paths.items()):
        by_range = _at(a, *path, "distance", "mae_by_range_m")
        series.append((label, {k: _num(by_range, k) for k in by_range}, PALETTE[i]))

    def draw(c: dict[str, str]) -> Figure:
        fig: Figure
        fig, ax = axes(7, 4.5, grid="y")
        for label, by_range, color in series:
            keys = sorted(by_range, key=lambda k: float(k.split("-")[0]))
            x = [float(k.split("-")[0]) + RANGE_BIN_M / 2 for k in keys]
            ls = "--" if label.startswith("M4") else "-"  # M4 coincides with sigma 0
            ax.plot(x, [by_range[k] for k in keys], ls, marker="o", ms=5, lw=2, color=color)
            ax.plot([], [], ls, marker="o", ms=5, lw=2, color=color, label=label)
        ax.set(
            xlabel="distance of pair midpoint from camera (m, 5 m bins)", ylabel="distance MAE (m)"
        )
        ax.set_title("Bird's-eye distance error vs range")
        ax.legend(fontsize=8)
        return fig

    return draw


def _f1_bars(a: Record, b: Record) -> list[tuple[str, float, str]]:
    """One bar per method: phase A pair F1 (GT boxes) and phase B person F1 (detections)."""
    h = f"{HEADLINE_M:.1f}"
    bars: list[tuple[str, float, str]] = []
    if a:
        for k in ("M0", "M1", "M2"):
            bars.append((f"A {k}", _num(a, k, "pair", h, "all", "f1"), PALETTE[0]))
        for s in CLICK_SIGMAS_PX:
            f1 = _num(a, "M3", f"sigma_{s:g}", "pair", h, "all", "f1")
            bars.append((f"A M3 s{s:g}", f1, PALETTE[0]))
        bars.append(("A M4", _num(a, "M4", "pair", h, "all", "f1"), PALETTE[0]))
    bars += [(f"B {k}", _num(b, k, "person", h, "all", "f1"), PALETTE[1]) for k in b]
    return bars


def _f1_chart(bars: list[tuple[str, float, str]]) -> Draw:
    labels, values, colors = zip(*bars, strict=True)

    def draw(c: dict[str, str]) -> Figure:
        fig: Figure
        fig, ax = axes(max(6, 0.8 * len(bars)), 4.5, grid="y")
        ax.bar(labels, values, color=colors, width=0.7)
        for i, val in enumerate(values):
            ax.text(i, val + 0.02, f"{val:.2f}", ha="center", fontsize=8, color=c["ink2"])
        ax.set(ylim=(0, 1.1), ylabel="F1")
        ax.set_title(f"F1 at {HEADLINE_M:g} m: A pairs (GT boxes), B people (detections)")
        ax.tick_params(axis="x", rotation=45)
        return fig

    return draw


def figures(
    fig_dir: Path, views: list[View], calibs: dict[str, Calibration], a: Record, b: Record
) -> None:
    """Write the three diagnostic figures from the phase A and B records (see figstyle)."""
    render(_scatter(views, calibs), "pixel_vs_metres.png", fig_dir)
    if a:
        render(_mae_vs_range(a), "mae_vs_range.png", fig_dir)
    if bars := _f1_bars(a, b):
        render(_f1_chart(bars), "f1_by_method.png", fig_dir)


def protocol(args: argparse.Namespace) -> Record:
    """Record every choice that affects the numbers."""
    return {
        "unit": "(camera, frame); people = annotated views clipped to the image, >= 50% inside",
        "gt_position": "positionID -> (-3 + 0.025 (id % 480), -9 + 0.025 (id // 480)) m, z = 0",
        "calibration": "intrinsic_zero K; extrinsic rvec/tvec, tvec centimetres -> metres",
        "thresholds_m": list(THRESHOLDS_M),
        "headline_m": HEADLINE_M,
        "violation": "pair: true ground distance < T (strict); person: in any violating pair",
        "split": {
            "tune": [TUNE_FRAMES.start, TUNE_FRAMES.stop - 5],
            "test": [TEST_FRAMES.start, TEST_FRAMES.stop - 5],
            "step": 5,
        },
        "limit_frames": args.limit_frames,
        "cameras": args.cameras,
        "methods": {
            "M0": f"centroids at width {RULE_WIDTH_PX}, fixed {RULE_THRESHOLD_PX:g} px, untuned",
            "M1": "full-res centroids, per-camera px threshold maximising pair F1 on tune split",
            "M2": "feet points, per-camera px threshold tuned the same way",
            "M3": f"4 clicked floor corners (largest {CLICK_GRID_M} m grid rectangle in view, "
            f"{CLICK_MARGIN_PX} px margin), click noise sigma {list(CLICK_SIGMAS_PX)} px, "
            f"seeds {list(CLICK_SEEDS)}, threshold T",
            "M4": "homography from K [r1 r2 t], feet points, threshold T",
        },
        "tune_candidates_px": [float(TUNE_CANDIDATES_PX[0]), float(TUNE_CANDIDATES_PX[-1])],
        "distance_error": f"GT pairs with true distance < {MAX_PAIR_M} m; range bins "
        f"{RANGE_BIN_M} m from the camera centre on the ground; people whose GT box is cut by "
        "the frame bottom are excluded (no foot point in the image), see feet_off_frame_excluded",
        "click_noise": "M3: one noise draw per (seed, camera); cameras are noised independently",
        "phase_b": {
            "detectors": DETECTORS,
            "device": args.device,
            "runs": [f"{d}+{m}" for d, m in PHASE_B_RUNS],
            "m3_noise": {"sigma_px": PHASE_B_SIGMA_PX, "seeds": list(CLICK_SEEDS)},
            "roi": f"flags computed over detections matched or with feet inside X {GROUND_X}"
            f" Y {GROUND_Y} m (true calibration); only matched detections scored (annotations "
            "are not exhaustive); flagged unmatched ones reported as unverifiable_flagged",
            "matching": f"Hungarian on IoU >= {MATCH_IOU}",
            "count_mae": "per view |flagged matched detections - true violators|; "
            "count_mae_all_flags also counts the unverifiable (unmatched) flags",
            "person_metric": "TP flagged matched det whose GT person is in violation; FP flagged"
            " matched det whose GT person is not; FN GT in violation not covered by a flagged"
            " matched det (missed detections included)",
            "cache": rel(DETS_DIR),
        },
    }


def _reprojection_medians(calibs: dict[str, Calibration], views: list[View]) -> dict[str, float]:
    errors = {
        c: [reprojection_errors(k, v) for v in views if v.camera == c] for c, k in calibs.items()
    }
    return {c: float(np.median(np.concatenate(e))) for c, e in errors.items()}


def _print_summary(a: Record, b: Record) -> None:
    h = f"{HEADLINE_M:.1f}"
    if a:
        for k in ("M0", "M1", "M2", "M4"):
            pair = _num(a, k, "pair", h, "all", "f1")
            person = _num(a, k, "person", h, "all", "f1")
            print(f"A {k:3} pair F1@{h} m {pair:.3f}  person F1 {person:.3f}")
        for s in CLICK_SIGMAS_PX:
            f1 = _at(a, "M3", f"sigma_{s:g}", "pair", h, "all", "f1")
            mean, std = _num(f1, "mean"), _num(f1, "std")
            print(f"A M3 sigma {s:>4g} px pair F1@{h} m {mean:.3f} +- {std:.3f}")
        d = _at(a, "M4", "distance")
        print(
            f"A M4 distance MAE {_num(d, 'mae'):.3f} m median {_num(d, 'median'):.3f} p90 "
            f"{_num(d, 'p90'):.3f}; position median {_num(d, 'position_median'):.3f} m"
        )
    for k in b:
        p = _at(b, k, "person", h, "all")
        print(
            f"B {k:20} person P {_num(p, 'precision'):.3f} R {_num(p, 'recall'):.3f}"
            f" F1 {_num(p, 'f1'):.3f}  count MAE {_num(b, k, 'count_mae'):.2f}"
        )


def main(argv: list[str] | None = None) -> int:
    """Run the requested phases, write the JSON and figures, and return the exit code."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.wildtrack", description=__doc__)
    parser.add_argument("--phase", nargs="+", choices=["A", "B"], default=["A", "B"])
    parser.add_argument("--cameras", nargs="+", choices=CAMERAS, default=CAMERAS)
    parser.add_argument("--limit-frames", type=int, default=None, help="frames per split")
    parser.add_argument("--device", default="0")
    parser.add_argument("--out", default="results/distance.json")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args(argv)

    calibs = {c: load_calibration(c) for c in args.cameras}
    tune_views = load_views(args.cameras, list(TUNE_FRAMES)[: args.limit_frames])
    test_views = load_views(args.cameras, list(TEST_FRAMES)[: args.limit_frames])
    base = base_methods(args.cameras, calibs, tune_views)
    reproj = _reprojection_medians(calibs, tune_views + test_views)
    a = phase_a(base, calibs, test_views) if "A" in args.phase else {}
    b = phase_b(base, calibs, test_views, args.device) if "B" in args.phase else {}
    report = {"protocol": protocol(args), "reprojection_px": reproj, "phase_a": a, "phase_b": b}
    figures(Path(args.fig_dir), test_views, calibs, a, b)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("reprojection median px:", {c: round(e, 1) for c, e in reproj.items()})
    _print_summary(a, b)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
