import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

from evaluation import report

RESULTS = Path(__file__).resolve().parent.parent / "results"
FILES = ("detection.json", "tracking.json", "distance.json", "speed.json", "finetune.json")
needs_results = pytest.mark.skipif(
    not all((RESULTS / f).exists() for f in FILES), reason="run the benchmarks first"
)


def results_dir(tmp_path, *names):
    root = tmp_path / "results"
    root.mkdir()
    for name in names:
        shutil.copy(RESULTS / name, root / name)
    return root


def write(root, name, data):
    (root / name).write_text(json.dumps(data), encoding="utf-8")


@needs_results
def test_report_reproduces_the_committed_results_md(tmp_path, capsys):
    root = results_dir(tmp_path, *FILES)
    out = tmp_path / "R.md"
    assert report.main(["--results", str(root), "--out", str(out)]) == 0
    assert "(5 sections)" in capsys.readouterr().out
    assert out.read_text(encoding="utf-8") == (RESULTS.parent / "RESULTS.md").read_text("utf-8")
    figures = {p.name for p in (root / "figures").iterdir()}
    assert figures == {"detection_ap.png", "recall_by_visibility.png", "speed.png"}


def test_missing_result_files_skip_their_sections(tmp_path, capsys):
    root = tmp_path / "empty"
    root.mkdir()
    out = tmp_path / "R.md"
    assert report.main(["--results", str(root), "--out", str(out)]) == 0
    assert "(0 sections)" in capsys.readouterr().out
    assert (
        out.read_text(encoding="utf-8").startswith("# Results\n") and "## " not in out.read_text()
    )


@needs_results
def test_phase_b_only_distance_run(tmp_path):
    root = results_dir(tmp_path)
    dist = json.loads((RESULTS / "distance.json").read_text(encoding="utf-8"))
    dist["phase_a"] = {}
    write(root, "distance.json", dist)
    out = tmp_path / "R.md"
    assert report.main(["--results", str(root), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "### Phase A: not run (`--phase B` only)" in text
    assert "pixel_vs_metres.png" not in text and "### Phase B" in text


@needs_results
def test_finetune_section_without_a_training_record(tmp_path):
    root = results_dir(tmp_path)
    ft = json.loads((RESULTS / "finetune.json").read_text(encoding="utf-8"))
    del ft["training"]
    del ft["models"]["finetuned"]
    write(root, "finetune.json", ft)
    out = tmp_path / "R.md"
    assert report.main(["--results", str(root), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "(n/a epochs, batch n/a, 0 min)" in text
    assert "fine-tuned on MOT17" not in text and "zero-shot (COCO)" in text


@needs_results
def test_finetune_section_reports_early_stopping(tmp_path):
    root = results_dir(tmp_path)
    ft = json.loads((RESULTS / "finetune.json").read_text(encoding="utf-8"))
    ft["training"] = {
        "epochs_run": 10,
        "epochs_planned": 15,
        "batch_size": 4,
        "wall_seconds": 600.0,
        "resumed_from_last_pt": True,
        "stopped_early": "no val gain for 5 epochs",
    }
    write(root, "finetune.json", ft)
    out = tmp_path / "R.md"
    assert report.main(["--results", str(root), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "(10 epochs of 15 planned, batch 4, 10 min across 2 segments)" in text
    assert " Stopped early: no val gain for 5 epochs." in text and "by epoch" not in text


def test_module_entry_point(tmp_path, monkeypatch):
    root = tmp_path / "empty"
    root.mkdir()
    out = tmp_path / "R.md"
    monkeypatch.setattr(sys, "argv", ["report", "--results", str(root), "--out", str(out)])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(report.__file__, run_name="__main__")
    assert e.value.code == 0 and out.exists()
