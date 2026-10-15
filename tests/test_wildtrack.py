import argparse
import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from evaluation import wildtrack as wt
from evaluation import wildtrack_data as wd
from evaluation.wildtrack import evaluate_end_to_end, pair_mask
from evaluation.wildtrack_data import (
    CAMERAS,
    ROOT,
    CachedDetector,
    View,
    base_methods,
    box_iou,
    click_homography,
    click_points,
    in_roi,
    load_calibration,
    load_frame_annotations,
    match,
    reprojection_errors,
    tune_threshold,
    visible,
    world_position,
)
from sda.geometry import feet, to_ground, violating_pairs

needs_data = pytest.mark.skipif(not ROOT.exists(), reason="WILDTRACK not downloaded")


def test_world_position_grid():
    assert world_position(0) == (-3.0, -9.0)
    assert world_position(481) == pytest.approx((-2.975, -8.975))
    assert world_position(480) == pytest.approx((-3.0, -8.975))  # X-first: catches a swap


@needs_data
def test_gt_positions_reproject_onto_gt_feet():
    ann = load_frame_annotations(0)
    medians = [
        np.median(reprojection_errors(load_calibration(c), visible(ann, c, 0))) for c in CAMERAS
    ]
    assert sum(m < 30 for m in medians) >= 5, medians


@needs_data
def test_full_calibration_homography_locates_people():
    calib = load_calibration("C1")
    view = visible(load_frame_annotations(0), "C1", 0)
    est = to_ground(feet(view.boxes), calib.ground_homography())
    assert np.median(np.linalg.norm(est - view.world, axis=1)) < 0.5


@needs_data
def test_four_clicks_without_noise_match_full_calibration():
    calib = load_calibration("C1")
    clicks = calib.project(click_points(calib))
    h = click_homography(calib, sigma=0.0, seed=0)
    np.testing.assert_allclose(
        to_ground(clicks, h), to_ground(clicks, calib.ground_homography()), atol=1e-6
    )


def test_pair_mask_matches_pdist_order():
    pts = np.array([[0, 0], [1, 0], [5, 0]], float)
    assert pair_mask(violating_pairs(pts, 1.5), 3).tolist() == [True, False, False]


def test_tune_threshold_recovers_separable_cut():
    true = np.array([0.5, 0.8, 1.2, 3.0, 4.0])
    pixel = np.array([10.0, 20.0, 30.0, 400.0, 500.0])
    assert 30 < tune_threshold(pixel, true, 2.0) <= 400


def test_box_iou_of_empty_boxes_is_zero_not_nan():
    empty = np.array([[5.0, 5.0, 5.0, 5.0]])
    assert box_iou(empty, empty) == 0 and np.isfinite(box_iou(empty, empty)).all()
    assert match(empty, empty).tolist() == [-1]  # no NaN reaches the assignment


def test_cached_detector_refuses_a_cache_from_another_setting(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "DETS_DIR", tmp_path)  # CachedDetector lives in wildtrack_data
    CachedDetector("yolo26s", conf=0.3, device="cpu")  # creates the key
    CachedDetector("yolo26s", conf=0.3, device="0")  # device does not change boxes
    with pytest.raises(RuntimeError, match="delete it"):
        CachedDetector("yolo26s", conf=0.25, device="cpu")
    (tmp_path / "old").mkdir()
    with pytest.raises(RuntimeError, match="_key.json"):
        CachedDetector("old", conf=0.3, device="cpu")


def test_match_requires_iou_half():
    gts = np.array([[0, 0, 100, 200], [500, 0, 600, 200]], float)
    dets = np.array([[510, 0, 610, 200], [0, 0, 100, 60], [5, 5, 105, 205]], float)
    assert match(dets, gts).tolist() == [1, -1, 0]


class _FixedDetector:
    name = "fixed"

    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, float)

    def __call__(self, camera, frame):
        return self.boxes


def _box_at(calib, ground):
    ((u, v),) = calib.project([ground])
    return [u - 20, v - 100, u + 20, v]


@needs_data
def test_end_to_end_scores_matched_and_reports_unverifiable():
    calib = {"C1": load_calibration("C1")}
    world = np.array([[2.0, 5.0], [3.0, 5.0]])  # 1 m apart, both in violation at 2 m
    gt = np.array([_box_at(calib["C1"], g) for g in world])
    outside = [[100, 0, 140, 60], [100, 0, 140, 60]]  # two touching people far off the ROI
    assert not in_roi(calib["C1"], np.array(outside)).any()
    duplicate = gt[0] + 3  # second detection of person 0: unmatched, inside the ROI
    dets = np.vstack((gt, outside, [duplicate]))
    m4 = base_methods(["C1"], calib, [])["M4"]
    view = View("C1", 1000, gt, world)
    r = evaluate_end_to_end(m4, _FixedDetector(dets), [view], calib)["person"]["2.0"]["all"]
    out = evaluate_end_to_end(m4, _FixedDetector(dets), [view], calib)
    assert (r["tp"], r["fp"], r["fn"]) == (2, 0, 0)
    assert out["unverifiable_flagged"] == 1  # the in-ROI duplicate; off-ROI pair dropped


@needs_data
def test_end_to_end_view_without_gt_people():
    calib = {"C1": load_calibration("C1")}
    near = [_box_at(calib["C1"], g) for g in ([2.0, 5.0], [2.5, 5.0])]
    view = View("C1", 1000, np.empty((0, 4)), np.empty((0, 2)))
    m4 = base_methods(["C1"], calib, [])["M4"]
    r = evaluate_end_to_end(m4, _FixedDetector(near), [view], calib)["person"]["2.0"]["all"]
    assert (r["tp"], r["fp"], r["fn"]) == (0, 0, 0)


# Synthetic scene: one camera 30 m above (0, 9) looking straight down, about 33 px per metre.
OVERHEAD = wd.Calibration(
    np.array([[1000.0, 0, 960], [0, 1000, 540], [0, 0, 1]]),
    np.diag([1.0, -1.0, -1.0]),
    np.array([0.0, 9.0, 30.0]),
)
PEOPLE = np.array([[0.0, 5.0], [1.0, 5.0], [4.0, 10.0], [6.0, 20.0], [0.5, 15.0], [1.5, 15.5]])
OFF_ROI = [20.0, 9.0]  # in the image, outside the annotated ground area


def _view(camera, frame, feet_in_frame=None):
    world = PEOPLE + [0.3 * frame, 0.0]
    boxes = np.array([_box_at(OVERHEAD, g) for g in world])
    return View(camera, frame, boxes, world, feet_in_frame)


class _FakeCachedDetector:
    """GT boxes plus an in-ROI duplicate and an off-ROI person, like a real detector."""

    def __init__(self, name, conf, device):
        self.name = name

    def __call__(self, camera, frame):
        gt = _view(camera, frame).boxes
        return np.vstack((gt, [gt[0] + 3], [_box_at(OVERHEAD, OFF_ROI)]))


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr(wt, "CLICK_SEEDS", range(2))
    monkeypatch.setattr(wt, "CachedDetector", _FakeCachedDetector)
    calibs = {"C1": OVERHEAD, "C2": OVERHEAD}
    tune_views = [_view(c, f) for f in (0, 1) for c in calibs]
    off = np.ones(len(PEOPLE), bool)
    off[0] = False
    test_views = [_view("C1", 2), _view("C2", 2, off), _view("C1", 3)]
    return calibs, wd.base_methods(list(calibs), calibs, tune_views), test_views


def test_prf_without_predictions_or_truth_is_zero():
    zero = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}
    assert wt.prf(np.array([0, 0, 0])) == zero


def test_prf_and_summarise_sum_cameras():
    s = wt.summarise({"C1": np.array([2, 1, 0]), "C2": np.array([1, 0, 1])})
    assert (s["all"]["tp"], s["all"]["fp"], s["all"]["fn"]) == (3, 1, 1)
    assert s["all"]["precision"] == pytest.approx(0.75) and s["C2"]["recall"] == 0.5


def test_mean_std_over_nested_runs():
    runs = [{"a": {"b": 1.0}, "n": 2}, {"a": {"b": 3.0}, "n": 4}]
    std = pytest.approx(2**0.5)
    assert wt.mean_std(runs) == {
        "a": {"b": {"mean": 2.0, "std": std}},
        "n": {"mean": 3.0, "std": std},
    }


def test_num_reads_the_mean_of_a_seed_summary():
    assert wt._num({"x": {"y": {"mean": 0.5, "std": 0.1}}}, "x", "y") == 0.5


def test_full_calibration_is_exact_on_synthetic_feet(scene):
    calibs, base, test_views = scene
    out = wt.evaluate_geometry(base["M4"], test_views, calibs)
    pair = out["pair"]["2.0"]["all"]
    assert (pair["fp"], pair["fn"]) == (0, 0) and pair["tp"] > 0
    d = out["distance"]
    assert d["mae"] == pytest.approx(0, abs=1e-6) and d["feet_off_frame_excluded"] == 1
    assert len(d["mae_by_range_m"]) > 1


def test_pixel_method_has_no_distance_record(scene):
    calibs, base, test_views = scene
    out = wt.evaluate_geometry(base["M0"], test_views, calibs)
    assert "distance" not in out and set(out["pair"]) == {"1.0", "1.5", "2.0"}


def test_phase_a_averages_click_noise_over_seeds(scene):
    calibs, base, test_views = scene
    a = wt.phase_a(base, calibs, test_views)
    assert set(a) == {"M0", "M1", "M2", "M4", "M3", "thresholds_px", "click_points_m"}
    f1 = a["M3"]["sigma_0"]["pair"]["2.0"]["all"]["f1"]
    assert f1["mean"] == pytest.approx(1.0) and f1["std"] == pytest.approx(0.0)
    assert set(a["thresholds_px"]["M1"]["2.0"]) == {"C1", "C2"}
    assert len(a["click_points_m"]["C1"]) == 4


def test_end_to_end_without_a_dataset(scene):
    calibs, base, test_views = scene
    detector = _FakeCachedDetector("d", 0.3, "cpu")
    out = wt.evaluate_end_to_end(base["M4"], detector, test_views, calibs)
    r = out["person"]["2.0"]["all"]
    assert (r["fp"], r["fn"]) == (0, 0)
    assert out["unverifiable_flagged"] == len(test_views)  # one in-ROI duplicate per view
    assert out["count_mae"] == 0 and out["count_mae_all_flags"] == 1


def test_phase_b_runs_every_configuration(scene):
    calibs, base, test_views = scene
    b = wt.phase_b(base, calibs, test_views, "cpu")
    assert list(b) == [f"{d}+{m}" for d, m in wt.PHASE_B_RUNS]
    m3 = b["yolo26s+M3"]
    assert (m3["detector"], m3["method"]) == ("yolo26s", "M3")
    assert m3["count_mae"] == {"mean": 0.0, "std": 0.0}


def test_figures_from_both_phases(scene, tmp_path):
    calibs, base, test_views = scene
    a = wt.phase_a(base, calibs, test_views)
    b = wt.phase_b(base, calibs, test_views, "cpu")
    wt.figures(tmp_path, test_views, calibs, a, b)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["f1_by_method.png", "mae_vs_range.png", "pixel_vs_metres.png"]


def test_figures_without_phases_draw_only_the_scatter(scene, tmp_path):
    calibs, _, test_views = scene
    wt.figures(tmp_path, test_views, calibs, {}, {})
    assert [p.name for p in tmp_path.iterdir()] == ["pixel_vs_metres.png"]


def test_protocol_records_the_run_arguments():
    args = argparse.Namespace(limit_frames=3, cameras=["C1"], device="cpu")
    p = wt.protocol(args)
    assert (p["limit_frames"], p["cameras"], p["phase_b"]["device"]) == (3, ["C1"], "cpu")
    assert p["phase_b"]["cache"].endswith("results/wildtrack_dets")  # repo-relative unless linked


@pytest.mark.parametrize("phases", [["A", "B"], ["B"]])
def test_main_writes_report_and_figures(scene, tmp_path, monkeypatch, capsys, phases):
    monkeypatch.setattr(wt, "load_calibration", lambda camera: OVERHEAD)
    monkeypatch.setattr(
        wt, "load_views", lambda cams, frames: [_view(c, f) for f in frames for c in cams]
    )
    out = tmp_path / "out.json"
    argv = ["--phase", *phases, "--cameras", "C1", "C2", "--limit-frames", "2"]
    assert wt.main([*argv, "--device", "cpu", "--out", str(out), "--fig-dir", str(tmp_path)]) == 0
    report = json.loads(out.read_text())
    assert max(report["reprojection_px"].values()) == pytest.approx(0, abs=1e-6)
    assert bool(report["phase_a"]) == ("A" in phases) and len(report["phase_b"]) == 4
    assert (tmp_path / "f1_by_method.png").exists()
    assert f"wrote {out}" in capsys.readouterr().out


def test_load_views_reads_annotation_json(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    box = {"xmin": 10, "ymin": 20, "xmax": 50, "ymax": 120}
    ann = [
        {"personID": 7, "positionID": 481, "views": [{"viewNum": 0, **box}]},
        {"personID": 9, "positionID": 0, "views": [{"viewNum": 1, **box}]},
    ]
    (tmp_path / "annotations_positions").mkdir()
    (tmp_path / "annotations_positions" / "00000005.json").write_text(json.dumps(ann))
    c1, c2 = wd.load_views(["C1", "C2"], [5])
    assert (c1.camera, c1.frame, c2.camera) == ("C1", 5, "C2")
    np.testing.assert_allclose(c1.world, [[-2.975, -8.975]])
    np.testing.assert_allclose(c2.boxes, [[10, 20, 50, 120]])


def test_load_calibration_reads_opencv_xml(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    intr = tmp_path / "calibrations" / "intrinsic_zero"
    extr = tmp_path / "calibrations" / "extrinsic"
    intr.mkdir(parents=True)
    extr.mkdir(parents=True)
    fs = cv2.FileStorage(str(intr / "intr_IDIAP1.xml"), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", OVERHEAD.k)
    fs.release()
    root = ET.Element("opencv_storage")
    ET.SubElement(root, "rvec").text = "3.141592653589793 0 0"
    ET.SubElement(root, "tvec").text = "0 900 3000"
    ET.ElementTree(root).write(extr / "extr_IDIAP1.xml")
    calib = wd.load_calibration("C5")
    np.testing.assert_allclose(calib.k, OVERHEAD.k)
    np.testing.assert_allclose(calib.t, [0, 9, 30])  # centimetres to metres
    np.testing.assert_allclose(calib.r, OVERHEAD.r, atol=1e-6)


def test_click_points_needs_a_rectangle_in_view():
    looking_up = wd.Calibration(OVERHEAD.k, np.eye(3), np.array([0.0, 0.0, -30.0]))
    with pytest.raises(ValueError, match="no grid rectangle"):
        wd.click_points(looking_up)


def test_tune_skips_views_with_one_person():
    one = View("C1", 0, np.array([[0.0, 0, 10, 10]]), np.array([[0.0, 0]]))
    two = _view("C1", 0)
    assert wd.tune([one, two], feet) == wd.tune([two], feet)


def test_cached_detector_runs_the_model_once_per_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "DETS_DIR", tmp_path / "dets")
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    frames = tmp_path / "Image_subsets" / "C1"
    frames.mkdir(parents=True)
    for number in (3, 4):
        cv2.imwrite(str(frames / f"{number:08d}.png"), np.zeros((4, 4, 3), np.uint8))
    loads = []

    def fake_load(name, conf, device):
        loads.append((name, conf, device))
        return lambda img: SimpleNamespace(boxes=np.array([[1.0, 2, 3, 4]]) + img.sum())

    monkeypatch.setattr(wd, "load_detector", fake_load)
    det = wd.CachedDetector("fake", 0.3, "cpu")
    first, again = det("C1", 3), det("C1", 3)
    np.testing.assert_array_equal(first, [[1, 2, 3, 4]])
    np.testing.assert_array_equal(again, first)
    assert det("C1", 4).shape == (1, 4) and loads == [
        ("fake", 0.3, "cpu")
    ]  # second miss, one load
    assert wd.CachedDetector("fake", 0.3, "cpu")("C1", 3).shape == (1, 4)  # from the cache
    assert loads == [("fake", 0.3, "cpu")]
