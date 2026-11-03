"""
MCFD Ground Truth and Evaluation Metrics Module for Person Down Detection.

Phase 3.1: Ground-Truth Evaluation Only.
This module encodes the verified ground-truth intervals from MCFD's official
technicalReport.pdf (Section 4.2 Camera Synchronisation, Section 4.3 Ground True)
and provides evaluation metric computation (precision, recall, F1, confusion matrix,
false positive/negative analysis) without altering existing detection pipelines.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


class GroundTruthLabel(str, Enum):
    """
    Semantic ground-truth position codes defined in MCFD technicalReport.pdf (Section 4.3).
    Differentiates between ground lying, sofa lying, crouching, falling, and ordinary activities.
    """
    STANDING = "STANDING"                    # Code 1: Walking, standing up
    FALL_TRANSITION = "FALL_TRANSITION"      # Code 2: Falling
    GROUND_LYING = "GROUND_LYING"            # Code 3: Lying on the ground
    CROUCHING = "CROUCHING"                  # Code 4: Crouching
    MOVING_UP_DOWN = "MOVING_UP_DOWN"        # Code 5: Moving down, Code 6: Moving up
    SITTING = "SITTING"                      # Code 7: Sitting
    SOFA_LYING = "SOFA_LYING"                # Code 8: Lying on a sofa
    BACKGROUND_UNLABELED = "BACKGROUND"      # Frames before entry or after scenario completion


# Raw Position Code to Semantic GroundTruthLabel mapping (Section 4.3)
RAW_CODE_TO_LABEL: Dict[int, GroundTruthLabel] = {
    1: GroundTruthLabel.STANDING,
    2: GroundTruthLabel.FALL_TRANSITION,
    3: GroundTruthLabel.GROUND_LYING,
    4: GroundTruthLabel.CROUCHING,
    5: GroundTruthLabel.MOVING_UP_DOWN,
    6: GroundTruthLabel.MOVING_UP_DOWN,
    7: GroundTruthLabel.SITTING,
    8: GroundTruthLabel.SOFA_LYING,
    9: GroundTruthLabel.STANDING,            # Moving horizontally
}

# Camera delay matrix: Scenario -> [Delay_Cam1, Delay_Cam2, ..., Delay_Cam8]
# Sourced from technicalReport.pdf Section 4.2, Figure 3.
# Formula: Delay_{c,s} = Nframe_{c,s} - min_c(Nframe_{c,s})
MCFD_CAMERA_DELAYS: Dict[int, List[int]] = {
    1:  [3, 3, 8, 4, 23, 6, 6, 0],
    2:  [25, 40, 0, 16, 18, 33, 33, 6],
    3:  [12, 16, 8, 16, 35, 20, 20, 0],
    4:  [72, 79, 78, 0, 68, 82, 83, 56],
    5:  [17, 24, 5, 11, 187, 26, 28, 0],
    6:  [0, 100, 106, 90, 89, 103, 104, 89],
    7:  [28, 14, 16, 0, 1, 17, 18, 20],
    8:  [92, 79, 0, 81, 64, 81, 82, 56],
    9:  [18, 9, 1, 19, 13, 11, 12, 0],
    10: [14, 15, 19, 33, 12, 17, 19, 0],
    11: [23, 4, 20, 14, 0, 6, 7, 12],
    12: [21, 6, 13, 8, 0, 3, 7, 0],
    13: [16, 33, 0, 7, 27, 27, 36, 13],
    14: [49, 36, 38, 0, 29, 29, 7, 14],
    15: [15, 19, 19, 15, 34, 40, 23, 0],
    16: [23, 29, 0, 2, 12, 9, 3, 3],
    17: [21, 26, 15, 0, 10, 0, 29, 18],
    18: [99, 105, 86, 0, 84, 108, 109, 77],
    19: [19, 27, 16, 19, 5, 29, 0, 20],
    20: [25, 9, 3, 10, 10, 4, 5, 0],
    21: [20, 30, 22, 3, 8, 33, 32, 0],
    22: [0, 46, 51, 41, 53, 46, 47, 34],
    23: [31, 52, 52, 45, 54, 60, 50, 0],
    24: [3, 36, 7, 0, 37, 10, 33, 1],
}

# Verified intervals per scenario from technicalReport.pdf Section 4.3 Table:
# Format: (ref_camera, [(start_frame, end_frame, raw_position_code), ...])
# Note: Intervals are given in the reference camera's frame coordinate system.
MCFD_SCENARIO_GROUND_TRUTH: Dict[int, Tuple[int, List[Tuple[int, int, int]]]] = {
    2: (
        4,
        [
            (308, 374, 1),
            (375, 399, 2),
            (400, 600, 3),
        ],
    ),
    9: (
        8,  # Base coordinate system (zero delay)
        [
            (310, 472, 1),
            (473, 505, 5),
            (506, 576, 7),
            (577, 627, 6),
            (628, 651, 2),
            (652, 760, 3),
        ],
    ),
    14: (
        6,
        [
            (372, 555, 1),
            (556, 590, 5),
            (591, 856, 8),
            (857, 934, 6),
            (935, 988, 6),
            (989, 1023, 2),
            (1024, 1115, 3),
        ],
    ),
    16: (
        4,
        [
            (380, 455, 1),
            (456, 488, 5),
            (489, 530, 4),
            (531, 568, 6),
            (569, 629, 5),
            (630, 645, 4),
            (646, 670, 6),
            (671, 731, 5),
            (732, 817, 7),
            (818, 890, 6),
            (891, 940, 2),
            (941, 1000, 3),
        ],
    ),
    24: (
        6,
        [
            (350, 974, 1),
            (975, 1315, 1),
            (1316, 1351, 5),
            (1352, 1414, 4),
            (1415, 1450, 6),
            (1451, 1750, 1),
            (1751, 1805, 5),
            (1806, 1844, 4),
            (1845, 1884, 6),
            (1885, 2490, 1),
            (2491, 2514, 5),
            (2515, 2563, 4),
            (2564, 2587, 6),
            (2588, 3040, 1),
            (3041, 3077, 5),
            (3078, 3125, 6),
            (3126, 3243, 1),
            (3244, 3353, 1),
            (3354, 3401, 5),
            (3402, 3500, 4),
        ],
    ),
}


def get_camera_delay_shift(chute_id: int, target_cam: int, ref_cam: int) -> int:
    """
    Computes frame shift from reference camera to target camera using Section 4.2 formula:
    Shift = Delay_{target_cam, chute} - Delay_{ref_cam, chute}.
    """
    if chute_id not in MCFD_CAMERA_DELAYS:
        raise KeyError(f"Chute ID {chute_id} not found in MCFD camera delays.")
    delays = MCFD_CAMERA_DELAYS[chute_id]
    d_target = delays[target_cam - 1]
    d_ref = delays[ref_cam - 1]
    return d_target - d_ref


def get_video_ground_truth(
    chute_id: int,
    cam_id: int,
    total_frames: int,
) -> List[str]:
    """
    Generates a list of frame-by-frame ground-truth string labels for a specific video.
    Frames outside annotated intervals are labeled GroundTruthLabel.BACKGROUND_UNLABELED.value.
    """
    labels = [GroundTruthLabel.BACKGROUND_UNLABELED.value] * total_frames
    if chute_id not in MCFD_SCENARIO_GROUND_TRUTH:
        return labels

    ref_cam, intervals = MCFD_SCENARIO_GROUND_TRUTH[chute_id]
    shift = get_camera_delay_shift(chute_id, target_cam=cam_id, ref_cam=ref_cam)

    for start_ref, end_ref, raw_code in intervals:
        start_target = max(0, start_ref + shift)
        end_target = min(total_frames - 1, end_ref + shift)
        sem_label = RAW_CODE_TO_LABEL.get(raw_code, GroundTruthLabel.BACKGROUND_UNLABELED).value

        for f_idx in range(start_target, end_target + 1):
            labels[f_idx] = sem_label

    return labels


def get_video_ground_truth_events(
    chute_id: int,
    cam_id: int,
    total_frames: int,
) -> List[Dict[str, Any]]:
    """
    Extracts verified ground-truth fall events as discrete intervals for a specific video.
    Each event dictionary contains:
    - gt_event_id: 1-indexed sequential event identifier
    - start_frame: start frame index in target camera coordinates (inclusive)
    - end_frame: end frame index in target camera coordinates (inclusive)
    - duration_frames: total frame count in the interval
    """
    if chute_id not in MCFD_SCENARIO_GROUND_TRUTH:
        return []

    ref_cam, intervals = MCFD_SCENARIO_GROUND_TRUTH[chute_id]
    shift = get_camera_delay_shift(chute_id, target_cam=cam_id, ref_cam=ref_cam)

    fall_events = []
    for start_ref, end_ref, raw_code in intervals:
        if RAW_CODE_TO_LABEL.get(raw_code) == GroundTruthLabel.GROUND_LYING:
            start_target = max(0, start_ref + shift)
            end_target = min(total_frames - 1, end_ref + shift)
            fall_events.append({
                "gt_event_id": len(fall_events) + 1,
                "start_frame": start_target,
                "end_frame": end_target,
                "duration_frames": end_target - start_target + 1,
            })

    return fall_events



def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels_true: List[str],
    labels_pred: List[str],
) -> Dict[str, Dict[str, int]]:
    """
    Computes a 2D contingency table (confusion matrix) where:
    Rows = ground-truth categories (labels_true)
    Columns = predicted categories (labels_pred).
    """
    matrix: Dict[str, Dict[str, int]] = {
        gt: {pred: 0 for pred in labels_pred} for gt in labels_true
    }
    for yt, yp in zip(y_true, y_pred):
        if yt in matrix and yp in matrix[yt]:
            matrix[yt][yp] += 1
    return matrix


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    target_classes: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Computes Precision, Recall, F1-score, True Positives (TP), False Positives (FP),
    and False Negatives (FN) for each specified class.
    Handles zero division safely by returning 0.0 when precision or recall is undefined.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for cls in target_classes:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
        support = sum(1 for yt in y_true if yt == cls)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        results[cls] = {
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "support": int(support),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    return results


def analyze_sofa_vs_ground(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Specifically analyzes model predictions during SOFA_LYING vs GROUND_LYING ground-truth periods.
    Essential for documenting why bed/sofa vs floor distinction requires spatial zones (Phase 3).
    """
    sofa_total = sum(1 for yt in y_true if yt == GroundTruthLabel.SOFA_LYING.value)
    sofa_lying_pred = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.SOFA_LYING.value and yp == "LYING")

    ground_total = sum(1 for yt in y_true if yt == GroundTruthLabel.GROUND_LYING.value)
    ground_lying_pred = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.GROUND_LYING.value and yp == "LYING")
    ground_no_det = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.GROUND_LYING.value and yp == "NO_DETECTION")

    return {
        "sofa_lying_frames": sofa_total,
        "sofa_predicted_lying": sofa_lying_pred,
        "sofa_lying_detection_rate": round(sofa_lying_pred / sofa_total, 4) if sofa_total > 0 else 0.0,
        "ground_lying_frames": ground_total,
        "ground_predicted_lying": ground_lying_pred,
        "ground_lying_detection_rate": round(ground_lying_pred / ground_total, 4) if ground_total > 0 else 0.0,
        "ground_lying_detector_dropouts": ground_no_det,
        "ground_dropout_rate": round(ground_no_det / ground_total, 4) if ground_total > 0 else 0.0,
    }


def analyze_crouching_false_positives(
    y_true: List[str],
    y_pred: List[str],
) -> Dict[str, Any]:
    """
    Analyzes predictions during CROUCHING ground-truth frames.
    Quantifies how often crouching triggers transient LYING predictions.
    """
    crouch_total = sum(1 for yt in y_true if yt == GroundTruthLabel.CROUCHING.value)
    crouch_as_lying = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.CROUCHING.value and yp == "LYING")
    crouch_as_standing = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.CROUCHING.value and yp == "STANDING")
    crouch_as_sitting = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.CROUCHING.value and yp == "SITTING")
    crouch_as_unknown = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.CROUCHING.value and yp == "UNKNOWN")
    crouch_no_det = sum(1 for yt, yp in zip(y_true, y_pred) if yt == GroundTruthLabel.CROUCHING.value and yp == "NO_DETECTION")

    return {
        "crouching_total_frames": crouch_total,
        "crouching_predicted_lying": crouch_as_lying,
        "crouching_false_lying_rate": round(crouch_as_lying / crouch_total, 4) if crouch_total > 0 else 0.0,
        "crouching_predicted_standing": crouch_as_standing,
        "crouching_predicted_sitting": crouch_as_sitting,
        "crouching_predicted_unknown": crouch_as_unknown,
        "crouching_no_detection": crouch_no_det,
    }
