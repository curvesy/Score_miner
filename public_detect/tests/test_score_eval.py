from public_detect.score_eval import (
    Box,
    average_precision_50,
    evaluate_score,
    false_positive_score,
    filter_predictions,
    iou_xyxy,
    match_diagnostics,
)


def test_iou_xyxy() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_ap50_penalizes_false_positive_before_true_positive() -> None:
    gt = [Box("a", 0, (0, 0, 10, 10))]
    preds = [
        Box("a", 0, (20, 20, 30, 30), conf=0.9),
        Box("a", 0, (0, 0, 10, 10), conf=0.8),
    ]
    assert average_precision_50(gt, preds) == 0.5


def test_score_prefers_higher_threshold_when_it_removes_false_positive() -> None:
    gt = [Box("a", 0, (0, 0, 10, 10))]
    preds = [
        Box("a", 0, (0, 0, 10, 10), conf=0.8),
        Box("a", 0, (20, 20, 30, 30), conf=0.2),
    ]
    loose = evaluate_score(gt, filter_predictions(preds, confidence=0.1), {0: "thing"})
    strict = evaluate_score(gt, filter_predictions(preds, confidence=0.5), {0: "thing"})
    assert strict.score > loose.score


def test_false_positive_score_matches_turbovision_ffpi_formula() -> None:
    assert false_positive_score(fp=0, image_count=7) == 1.0
    assert false_positive_score(fp=7, image_count=7) == 0.9
    assert false_positive_score(fp=70, image_count=7) == 0.0


def test_match_diagnostics_reports_misses_and_false_positives() -> None:
    gt = [Box("a", 0, (0, 0, 10, 10))]
    preds = [Box("a", 0, (20, 20, 30, 30), conf=0.9)]
    diagnostics = match_diagnostics(gt, preds, {0: "thing"})
    assert len(diagnostics["false_positives"]) == 1
    assert len(diagnostics["misses"]) == 1
