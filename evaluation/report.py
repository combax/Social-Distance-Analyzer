"""Assemble RESULTS.md and summary figures from the benchmark JSON files in ``results/``.

Every number in RESULTS.md is read from those files, so rerunning the benchmarks and this
script keeps the report in sync. Sections whose JSON is missing are skipped.
"""

import argparse
import json
from pathlib import Path

from matplotlib.figure import Figure

from evaluation.figstyle import AQUA, BLUE, ORANGE, YELLOW, axes, picture, render
from evaluation.schema import Detection, Distance, Finetune, MeanStd, PhaseA, Speed, Tracking

RESULTS = Path(__file__).resolve().parent.parent / "results"
BASELINE = "yolov3-darknet"
VIS_MODELS = ("yolov3-darknet", "yolo26s", "yolo26s@1280", "yolo26m@1280")


def _load[T](root: Path, name: str, _kind: type[T]) -> T | None:
    path = root / name
    return json.loads(path.read_text()) if path.exists() else None


def _pm(x: float | MeanStd, fmt: str = ".3f") -> str:
    if isinstance(x, dict):
        return f"{x['mean']:{fmt}} ± {x['std']:{fmt}}"
    return f"{x:{fmt}}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def detection_section(det: Detection, fig_dir: Path) -> str:
    """MOT17 detection table, AP chart and recall-by-visibility chart."""
    models = det["models"]
    order = sorted(models, key=lambda m: models[m]["overall"]["AP"])
    rows = []
    for m in sorted(models, key=lambda m: -models[m]["overall"]["AP"]):
        o = models[m]["overall"]
        rows.append(
            [
                m,
                *(f"{o[k]:.3f}" for k in ("AP", "AP50", "precision", "recall", "f1")),
                f"{o['best_conf']:.2f}",
                *(f"{o['at_best_conf'][k]:.3f}" for k in ("recall", "f1")),
            ]
        )
    table = _table(
        ["detector", "AP", "AP50", "P@0.3", "R@0.3", "F1@0.3", "best conf", "R@best", "F1@best"],
        rows,
    )

    aps = [models[m]["overall"]["AP"] for m in order]
    n_frames = sum(det["protocol"]["frames_evaluated"].values())

    def draw_ap(c: dict[str, str]) -> Figure:
        fig, ax = axes(6.4, 0.34 * len(order) + 1.0)
        colors = [ORANGE if m == BASELINE else BLUE for m in order]
        ax.barh(order, aps, color=colors, height=0.62)
        for y, a in enumerate(aps):
            ax.text(a + 0.005, y, f"{a:.3f}", va="center", fontsize=8, color=c["ink2"])
        ax.set_xlim(0, max(aps) * 1.15)
        ax.set_xlabel(f"COCO AP on MOT17 train ({n_frames:,} frames)")
        ax.set_title("Person detection accuracy (orange: the Darknet YOLOv3 baseline)", loc="left")
        return fig

    ap_fig = render(draw_ap, "detection_ap.png", fig_dir)
    present = [m for m in VIS_MODELS if m in models]

    def draw_vis(c: dict[str, str]) -> Figure:
        fig, ax = axes(6.4, 3.4, grid="y")
        for m, col in zip(present, (ORANGE, BLUE, AQUA, YELLOW), strict=False):
            rv = models[m]["overall"]["recall_by_visibility"]
            xs = list(rv)
            ax.plot(xs, [rv[x] or 0 for x in xs], color=col, lw=2.2, marker="o", ms=5, label=m)
        ax.set_ylim(0, 1)
        ax.set_xlabel("ground-truth visibility (fraction of the person not occluded)")
        ax.set_ylabel("recall at conf 0.3")
        ax.set_title("Occlusion is where crowd counting fails", loc="left")
        ax.legend(fontsize=8)
        return fig

    vis_fig = render(draw_vis, "recall_by_visibility.png", fig_dir)

    p = det["protocol"]
    return (
        "## 1. Detection accuracy (MOT17)\n\n"
        f"{len(p['sequences'])} sequences, {sum(p['frames_evaluated'].values())} frames. "
        f"Positives: "
        f"{p['positives']}, clipped to the frame (MOT17 annotates bodies that leave the image; "
        "every detector's boxes are clipped the same way). Ignore regions: "
        f"{p['ignore_handling']}. Operating point: the app's "
        "confidence 0.3 at IoU 0.5, chosen for the Darknet baseline's score scale; "
        "`best conf` is each model's F1-maximising threshold on a 0.05 grid (same matching, so "
        "optimistic by one tuning step). Inference size 640 unless the name ends in `@1280`.\n\n"
        f"{table}\n\n{ap_fig}\n\n{vis_fig}\n"
    )


def tracking_section(trk: Tracking) -> str:
    """MOT17 tracking table."""
    runs = sorted(trk["runs"].values(), key=lambda r: -r["overall"]["HOTA"])
    rows = [
        [
            r["model"],
            r["tracker"],
            *(f"{r['overall'][k]:.3f}" for k in ("HOTA", "DetA", "AssA", "MOTA", "IDF1")),
            f"{r['overall']['IDSW']:.0f}",
            f"{r['overall']['fps']:.1f}",
        ]
        for r in runs
    ]
    return (
        "## 2. Tracking (MOT17, TrackEval)\n\nUltralytics trackers at detector conf 0.1; "
        "MOTChallenge preprocessing (distractor matches removed). Ultralytics assumes 30 fps "
        "for in-memory frames (MOT17-05 is 14 fps, MOT17-13 is 25 fps); this affects every "
        "row alike. fps is detection + tracking wall time on the RTX 4060 Laptop GPU while "
        "other benchmark jobs shared it, so compare rows only loosely; section 4 has clean "
        "speed numbers.\n\n"
        + _table(
            ["detector", "tracker", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "fps"], rows
        )
        + "\n"
    )


def _phase_a(a: PhaseA, h: str) -> str:
    """Phase A table, distance-error paragraph and figures; empty when phase A was not run."""
    if not a:
        return "### Phase A: not run (`--phase B` only)\n\n"
    names = {
        "M0": "pixel rule: centroid pixels, 50 px at 700 px width",
        "M1": "centroid pixels, threshold tuned per camera",
        "M2": "feet pixels, threshold tuned per camera",
        "M4": "bird's-eye, full calibration (upper bound)",
    }
    rows = [
        [
            k,
            names[k],
            f"{a[k]['pair'][h]['all']['f1']:.3f}",
            f"{a[k]['person'][h]['all']['f1']:.3f}",
        ]
        for k in ("M0", "M1", "M2")
    ]
    for s, v in a["M3"].items():
        sigma = s.removeprefix("sigma_")
        rows.append(
            [
                f"M3 σ={sigma}px",
                f"bird's-eye from 4 clicked floor points, σ={sigma} px",
                _pm(v["pair"][h]["all"]["f1"]),
                _pm(v["person"][h]["all"]["f1"]),
            ]
        )
    rows.append(
        [
            "M4",
            names["M4"],
            f"{a['M4']['pair'][h]['all']['f1']:.3f}",
            f"{a['M4']['person'][h]['all']['f1']:.3f}",
        ]
    )
    d = a["M4"]["distance"]
    by_range = ", ".join(f"{k} m: {v:.3f}" for k, v in d["mae_by_range_m"].items())
    return (
        f"### Phase A: ground-truth boxes (geometry only), violation = closer than {h} m\n\n"
        + _table(["id", "method", "pair F1", "person F1"], rows)
        + f"\n\nBird's-eye (M4) distance error over GT pairs closer than 5 m: MAE {d['mae']:.3f}"
        f" m, median {d['median']:.3f} m, P90 {d['p90']:.3f} m; by distance from camera: "
        f"{by_range}. Person position error median {d['position_median']:.3f} m. "
        f"{d['feet_off_frame_excluded']} people whose box is cut by the frame bottom are "
        "excluded from these error statistics (their foot point is not in the image).\n\n"
        + picture("pixel_vs_metres.png", "Pixel distance vs real distance")
        + "\n\n"
        + picture("mae_vs_range.png", "Distance error vs range")
        + "\n\n"
    )


def distance_section(dist: Distance) -> str:
    """WILDTRACK geometry (phase A) and end-to-end (phase B) tables."""
    a, b, p = dist["phase_a"], dist["phase_b"], dist["protocol"]
    h = f"{p['headline_m']:.1f}"
    b_rows = [
        [
            k,
            _pm(v["person"][h]["all"]["precision"]),
            _pm(v["person"][h]["all"]["recall"]),
            _pm(v["person"][h]["all"]["f1"]),
            _pm(v["count_mae"], ".2f"),
            _pm(v["count_mae_all_flags"], ".2f"),
            _pm(v["unverifiable_flagged"], ".0f"),
        ]
        for k, v in b.items()
    ]
    repro = ", ".join(f"{c} {e:.1f}" for c, e in dist["reprojection_px"].items())
    return (
        "## 3. Real-world distance accuracy (WILDTRACK)\n\n"
        "7 calibrated cameras; ground-truth positions on a 2.5 cm grid. Thresholds tuned on "
        "frames 0-995, every number below on frames 1000-1995. Calibration check: median "
        f"reprojection error of GT positions onto GT box feet, px: {repro}.\n\n"
        + _phase_a(a, h)
        + "### Phase B: end to end with detector boxes, person-level, 2 m\n\n"
        "Only detections matched (IoU ≥ 0.5) to annotated people are scored: WILDTRACK does "
        "not annotate everyone in view, so flagged unmatched detections are reported as "
        "unverifiable instead of as false positives. WILDTRACK boxes come from a person model, "
        "with no tight annotation, so IoU matching lowers recall equally for every row. "
        "Count MAE compares flagged matched detections with the true violator count per view; "
        "the all-flags column also counts the unverifiable flags, as a deployed counter "
        "would.\n\n"
        + _table(
            [
                "detector + method",
                "precision",
                "recall",
                "F1",
                "violator count MAE",
                "count MAE, all flags",
                "unverifiable flags",
            ],
            b_rows,
        )
        + "\n\n"
        + picture("f1_by_method.png", "F1 by method")
        + "\n"
    )


def finetune_section(ft: Finetune) -> str:
    """Zero-shot vs fine-tuned comparison on the held-out MOT17 sequences."""
    rows = []
    for key, label in (("zero_shot", "zero-shot (COCO)"), ("finetuned", "fine-tuned on MOT17")):
        if key not in ft["models"]:  # `compare --only` output
            continue
        m = ft["models"][key]
        o, t, best = m["detection"]["overall"], m["tracking"]["overall"], m["test_at_best_conf"]
        rows.append(
            [
                label,
                f"{o['AP']:.3f}",
                f"{o['AP50']:.3f}",
                f"{o['f1']:.3f}",
                f"{m['best_conf_on_val']:.2f}",
                f"{best['f1']:.3f}",
                f"{t['HOTA']:.3f}",
                f"{t['MOTA']:.3f}",
                f"{t['IDF1']:.3f}",
            ]
        )
    sp = ft["protocol"]["split"]
    tr = ft.get("training") or {"epochs_run": "n/a", "batch_size": "n/a", "wall_seconds": 0.0}
    segments = tr.get("segments", 2 if tr.get("resumed_from_last_pt") else 1)
    resumed = f" across {segments} segments" if segments > 1 else ""
    planned_n = tr.get("epochs_planned")
    planned = f" of {planned_n} planned" if planned_n and planned_n != tr["epochs_run"] else ""
    stopped = f" Stopped early: {tr['stopped_early']}." if tr.get("stopped_early") else ""
    curve = tr.get("val_map50_95_by_epoch")
    trend = f" Val mAP50-95 by epoch: {', '.join(f'{v:.3f}' for v in curve)}." if curve else ""
    return (
        "## 5. Fine-tuning vs zero-shot (held-out sequences)\n\n"
        f"yolo26s at {ft['protocol']['imgsz']} px. Trained on {', '.join(sp['train_sequences'])}"
        f" ({tr['epochs_run']} epochs{planned}, batch {tr['batch_size']}, "
        f"{tr['wall_seconds'] / 60:.0f} min{resumed}); evaluated on "
        f"{', '.join(sp['test_sequences'])}, "
        "never seen in training, model selection or threshold choice. The best threshold is "
        f"chosen on the training sequences' held-out tail.{stopped}{trend}\n\n"
        + _table(
            [
                "model",
                "AP",
                "AP50",
                "F1@0.3",
                "best conf (val)",
                "F1@best",
                "HOTA",
                "MOTA",
                "IDF1",
            ],
            rows,
        )
        + "\n"
    )


def speed_section(speed: Speed, fig_dir: Path) -> str:
    """Latency table and CPU/GPU fps chart."""
    res = speed["results"]
    models = list(dict.fromkeys(r["model"] for r in res))
    fps = {(r["model"], r["device"]): r["fps"] for r in res}
    rows = [
        [m, *(f"{fps[(m, d)]:.1f}" if (m, d) in fps else "n/a" for d in ("cpu", "0"))]
        for m in models
    ]

    def draw_speed(c: dict[str, str]) -> Figure:
        fig, ax = axes(6.4, 0.44 * len(models) + 1.0)
        y = range(len(models))
        for off, dev, color, label in ((-0.19, "cpu", BLUE, "CPU"), (0.19, "0", ORANGE, "GPU")):
            vals = [fps.get((m, dev), 0.0) for m in models]
            ax.barh([i + off for i in y], vals, height=0.36, color=color, label=label)
        ax.set_yticks(list(y), models)
        ax.invert_yaxis()
        ax.axvline(25, color=c["muted"], lw=1.2, ls="--")
        ax.text(26, -0.6, "25 fps real time", color=c["ink2"], fontsize=8)
        ax.set_xlabel("frames per second, 1080p, full pipeline (median)")
        ax.set_title("Speed: CPU vs GPU", loc="left")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
        return fig

    path = render(draw_speed, "speed.png", fig_dir)
    m = speed["machine"]
    return (
        "## 4. Speed\n\n"
        f"{m.get('cpu')} / {m.get('gpu') or 'no GPU measured'}; torch {m.get('torch')}, "
        f"ultralytics "
        f"{m.get('ultralytics')}, OpenCV {m.get('opencv')}. Median per-frame latency of "
        "detection + foot points + ground projection + pair check on MOT17-04 (1920x1080), "
        "after warm-up. The Darknet baseline is CPU only (pip OpenCV has no CUDA).\n\n"
        + _table(["detector", "CPU fps", "GPU fps"], rows)
        + f"\n\n{path}\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write RESULTS.md from whatever result files exist.

    Args:
        argv: Command-line arguments without the program name; None reads ``sys.argv``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m evaluation.report", description=__doc__)
    parser.add_argument("--out", default=str(RESULTS.parent / "RESULTS.md"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    fig_dir = args.results / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Results\n\nGenerated by `python -m evaluation.report` from `results/*.json`; "
        "reproduce with `bash scripts/run_benchmarks.sh`. Headline findings are in README.md.\n"
    ]
    if det := _load(args.results, "detection.json", Detection):
        parts.append(detection_section(det, fig_dir))
    if trk := _load(args.results, "tracking.json", Tracking):
        parts.append(tracking_section(trk))
    if dist := _load(args.results, "distance.json", Distance):
        parts.append(distance_section(dist))
    if speed := _load(args.results, "speed.json", Speed):
        parts.append(speed_section(speed, fig_dir))
    if ft := _load(args.results, "finetune.json", Finetune):
        parts.append(finetune_section(ft))
    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {args.out} ({len(parts) - 1} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
