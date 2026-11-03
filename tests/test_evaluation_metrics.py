"""
Unit tests for MCFD Ground Truth Evaluation and Metrics Module.

Phase 3.1: Ground-Truth Evaluation Only.
Verifies camera delay calculations, interval shifts, confusion matrix computation,
precision/recall/F1 metrics, and safety against zero-division edge cases.
"""

import pytest
from src.core.ground_truth import (
    GroundTruthLabel,
    MCFD_CAMERA_DELAYS,
    get_camera_delay_shift,
    get_video_ground_truth,
    compute_confusion_matrix,
    compute_classification_metrics,
    analyze_sofa_vs_ground,
    analyze_crouching_false_positives,
)


class TestCameraDelayCalculation:
    """Verifies camera synchronization delay calculations (Section 4.2)."""

    def test_chute02_cam1_shift(self):
        # Scenario 2: Cam 1 delay=25, Ref Cam 4 delay=16. Shift = +9.
        shift = get_camera_delay_shift(chute_id=2, target_cam=1, ref_cam=4)
        assert shift == 9

    def test_chute09_cam1_shift(self):
        # Scenario 9: Cam 1 delay=18, Base Ref Cam 8 delay=0. Shift = +18.
        shift = get_camera_delay_shift(chute_id=9, target_cam=1, ref_cam=8)
        assert shift == 18

    def test_chute14_cam1_shift(self):
        # Scenario 14: Cam 1 delay=49, Ref Cam 6 delay=29. Shift = +20.
        shift = get_camera_delay_shift(chute_id=14, target_cam=1, ref_cam=6)
        assert shift == 20

    def test_chute14_cam2_shift(self):
        # Scenario 14: Cam 2 delay=36, Ref Cam 6 delay=29. Shift = +7.
        shift = get_camera_delay_shift(chute_id=14, target_cam=2, ref_cam=6)
        assert shift == 7

    def test_chute16_cam1_shift(self):
        # Scenario 16: Cam 1 delay=23, Ref Cam 4 delay=2. Shift = +21.
        shift = get_camera_delay_shift(chute_id=16, target_cam=1, ref_cam=4)
        assert shift == 21

    def test_chute24_cam1_shift(self):
        # Scenario 24: Cam 1 delay=3, Ref Cam 6 delay=10. Shift = -7.
        shift = get_camera_delay_shift(chute_id=24, target_cam=1, ref_cam=6)
        assert shift == -7

    def test_invalid_chute_raises(self):
        with pytest.raises(KeyError):
            get_camera_delay_shift(chute_id=99, target_cam=1, ref_cam=1)


class TestGroundTruthIntervalGeneration:
    """Verifies that frame-by-frame ground-truth arrays align with technicalReport.pdf."""

    def test_chute02_cam1_intervals(self):
        # Chute 2, Cam 1 (831 frames total).
        # Cam 4: 308-374 (Walking), 375-399 (Falling), 400-600 (Ground Lying).
        # Shift = +9:
        # Cam 1: 317-383 (Walking), 384-408 (Falling), 409-609 (Ground Lying).
        gt = get_video_ground_truth(chute_id=2, cam_id=1, total_frames=831)
        assert len(gt) == 831

        # Background before entry
        assert gt[100] == GroundTruthLabel.BACKGROUND_UNLABELED.value
        assert gt[316] == GroundTruthLabel.BACKGROUND_UNLABELED.value

        # Walking
        assert gt[317] == GroundTruthLabel.STANDING.value
        assert gt[350] == GroundTruthLabel.STANDING.value
        assert gt[383] == GroundTruthLabel.STANDING.value

        # Falling
        assert gt[384] == GroundTruthLabel.FALL_TRANSITION.value
        assert gt[395] == GroundTruthLabel.FALL_TRANSITION.value
        assert gt[408] == GroundTruthLabel.FALL_TRANSITION.value

        # Ground Lying
        assert gt[409] == GroundTruthLabel.GROUND_LYING.value
        assert gt[500] == GroundTruthLabel.GROUND_LYING.value
        assert gt[609] == GroundTruthLabel.GROUND_LYING.value

        # After ground period
        assert gt[610] == GroundTruthLabel.BACKGROUND_UNLABELED.value

    def test_chute14_cam1_sofa_vs_ground_intervals(self):
        # Chute 14, Cam 1 (1499 frames total).
        # Ref Cam 6: Sofa Lying is 591-856. Shift = +20 -> Cam 1: 611-876.
        # Ref Cam 6: Ground Lying is 1024-1115. Shift = +20 -> Cam 1: 1044-1135.
        gt = get_video_ground_truth(chute_id=14, cam_id=1, total_frames=1499)
        assert len(gt) == 1499

        # Sofa lying interval
        assert gt[611] == GroundTruthLabel.SOFA_LYING.value
        assert gt[700] == GroundTruthLabel.SOFA_LYING.value
        assert gt[876] == GroundTruthLabel.SOFA_LYING.value

        # Ground lying interval
        assert gt[1044] == GroundTruthLabel.GROUND_LYING.value
        assert gt[1070] == GroundTruthLabel.GROUND_LYING.value
        assert gt[1135] == GroundTruthLabel.GROUND_LYING.value

        # Verified distinction: Sofa lying != Ground lying
        assert gt[700] != gt[1070]


class TestEvaluationMetricsCalculation:
    """Verifies confusion matrix and precision/recall/F1 calculations."""

    def test_confusion_matrix(self):
        y_true = ["STANDING", "STANDING", "GROUND_LYING", "SITTING"]
        y_pred = ["STANDING", "SITTING", "LYING", "SITTING"]
        labels_true = ["STANDING", "SITTING", "GROUND_LYING"]
        labels_pred = ["STANDING", "SITTING", "LYING", "UNKNOWN"]

        matrix = compute_confusion_matrix(y_true, y_pred, labels_true, labels_pred)
        assert matrix["STANDING"]["STANDING"] == 1
        assert matrix["STANDING"]["SITTING"] == 1
        assert matrix["GROUND_LYING"]["LYING"] == 1
        assert matrix["SITTING"]["SITTING"] == 1
        assert matrix["SITTING"]["UNKNOWN"] == 0

    def test_metrics_perfect_score(self):
        y_true = ["STANDING", "STANDING", "SITTING", "SITTING"]
        y_pred = ["STANDING", "STANDING", "SITTING", "SITTING"]

        metrics = compute_classification_metrics(y_true, y_pred, target_classes=["STANDING", "SITTING"])
        assert metrics["STANDING"]["precision"] == 1.0
        assert metrics["STANDING"]["recall"] == 1.0
        assert metrics["STANDING"]["f1_score"] == 1.0
        assert metrics["STANDING"]["true_positives"] == 2
        assert metrics["STANDING"]["false_positives"] == 0
        assert metrics["STANDING"]["false_negatives"] == 0

    def test_metrics_zero_division_safety(self):
        # Class with zero predictions and zero ground truth
        y_true = ["STANDING", "STANDING"]
        y_pred = ["STANDING", "STANDING"]

        metrics = compute_classification_metrics(y_true, y_pred, target_classes=["LYING"])
        assert metrics["LYING"]["precision"] == 0.0
        assert metrics["LYING"]["recall"] == 0.0
        assert metrics["LYING"]["f1_score"] == 0.0
        assert metrics["LYING"]["support"] == 0


class TestDomainSpecificAnalyses:
    """Verifies domain analyses for sofa vs ground and crouching false positives."""

    def test_sofa_vs_ground_analysis(self):
        y_true = ["SOFA_LYING", "SOFA_LYING", "GROUND_LYING", "GROUND_LYING", "GROUND_LYING"]
        y_pred = ["LYING", "LYING", "LYING", "NO_DETECTION", "STANDING"]

        res = analyze_sofa_vs_ground(y_true, y_pred)
        assert res["sofa_lying_frames"] == 2
        assert res["sofa_predicted_lying"] == 2
        assert res["sofa_lying_detection_rate"] == 1.0
        assert res["ground_lying_frames"] == 3
        assert res["ground_predicted_lying"] == 1
        assert res["ground_lying_detection_rate"] == round(1 / 3, 4)
        assert res["ground_lying_detector_dropouts"] == 1
        assert res["ground_dropout_rate"] == round(1 / 3, 4)

    def test_crouching_false_positive_analysis(self):
        y_true = ["CROUCHING", "CROUCHING", "CROUCHING", "CROUCHING", "STANDING"]
        y_pred = ["LYING", "STANDING", "SITTING", "UNKNOWN", "STANDING"]

        res = analyze_crouching_false_positives(y_true, y_pred)
        assert res["crouching_total_frames"] == 4
        assert res["crouching_predicted_lying"] == 1
        assert res["crouching_false_lying_rate"] == 0.25
        assert res["crouching_predicted_standing"] == 1
        assert res["crouching_predicted_sitting"] == 1
        assert res["crouching_predicted_unknown"] == 1
