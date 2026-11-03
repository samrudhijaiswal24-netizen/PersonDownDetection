"""
Phase 3.1: Ground-Truth Evaluation Script for Person Down Detection.

Compares frame-level predictions from YOLO11-Pose + Phase 2 PostureClassifier
against verified ground-truth annotations from MCFD's technicalReport.pdf.
Preserves all existing Phase 1/Phase 2 code, model weights, and existing outputs.
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pose_detector import YOLOPoseDetector
from src.core.posture_classifier import PostureClassifier, PostureSmoother, PostureType
from src.core.ground_truth import (
    GroundTruthLabel,
    get_video_ground_truth,
    compute_confusion_matrix,
    compute_classification_metrics,
    analyze_sofa_vs_ground,
    analyze_crouching_false_positives,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateGroundTruth")


# The 6 selected MCFD test videos
TEST_VIDEOS = [
    {"filename": "mcfd_chute02_cam1.avi", "chute": 2, "cam": 1},
    {"filename": "mcfd_chute09_cam1.avi", "chute": 9, "cam": 1},
    {"filename": "mcfd_chute14_cam1.avi", "chute": 14, "cam": 1},
    {"filename": "mcfd_chute14_cam2.avi", "chute": 14, "cam": 2},
    {"filename": "mcfd_chute16_cam1.avi", "chute": 16, "cam": 1},
    {"filename": "mcfd_chute24_cam1.avi", "chute": 24, "cam": 1},
]

GROUND_TRUTH_ORDER = [
    GroundTruthLabel.STANDING.value,
    GroundTruthLabel.SITTING.value,
    GroundTruthLabel.SOFA_LYING.value,
    GroundTruthLabel.GROUND_LYING.value,
    GroundTruthLabel.FALL_TRANSITION.value,
    GroundTruthLabel.CROUCHING.value,
    GroundTruthLabel.MOVING_UP_DOWN.value,
    GroundTruthLabel.BACKGROUND_UNLABELED.value,
]

PREDICTION_ORDER = [
    PostureType.STANDING.value,
    PostureType.SITTING.value,
    PostureType.LYING.value,
    PostureType.UNKNOWN.value,
    "NO_DETECTION",
]


def extract_frame_predictions(
    video_path: Path,
    detector: YOLOPoseDetector,
    classifier: PostureClassifier,
    smoother: Optional[PostureSmoother] = None,
) -> Tuple[List[str], List[float], int, float]:
    """
    Runs video through detector + classifier to produce per-frame posture predictions.

    Returns:
        predictions: list of predicted posture strings per frame
        confidences: list of detection box confidences per frame (0.0 if no detection)
        total_frames: count of processed frames
        elapsed_sec: execution time in seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    predictions: List[str] = []
    confidences: List[float] = []

    if smoother is not None:
        smoother.reset()

    start_time = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        persons = detector.detect(frame)
        if not persons:
            predictions.append("NO_DETECTION")
            confidences.append(0.0)
            if smoother is not None:
                smoother.reset()
        else:
            # Multi-person: prioritize highest detection confidence
            best_person = max(persons, key=lambda p: p.confidence)
            p_res = classifier.classify(best_person.keypoints, best_person.box)
            if smoother is not None:
                p_res = smoother.update(p_res)
            predictions.append(p_res.label)
            confidences.append(float(best_person.confidence))

        frame_idx += 1

    cap.release()
    elapsed_sec = time.perf_counter() - start_time
    return predictions, confidences, total_frames, elapsed_sec


def format_confusion_matrix_table(matrix: Dict[str, Dict[str, int]]) -> str:
    """Formats 2D confusion matrix dictionary as a clean GitHub-style Markdown table."""
    headers = ["Ground Truth Label"] + PREDICTION_ORDER + ["Total Support"]
    separator = ["| :--- "] + ["| :---: "] * (len(PREDICTION_ORDER) + 1)

    lines = ["| " + " | ".join(headers) + " |", "".join(separator) + "|"]

    for gt in GROUND_TRUTH_ORDER:
        row = [gt]
        total_gt = 0
        for pred in PREDICTION_ORDER:
            cnt = matrix.get(gt, {}).get(pred, 0)
            row.append(str(cnt))
            total_gt += cnt
        row.append(str(total_gt))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_metrics_table(metrics: Dict[str, Dict[str, Any]]) -> str:
    """Formats classification metrics dictionary as a GitHub-style Markdown table."""
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support", "TP", "FP", "FN"]
    separator = ["| :--- ", "| :---: ", "| :---: ", "| :---: ", "| :---: ", "| :---: ", "| :---: ", "| :---: |"]

    lines = ["| " + " | ".join(headers) + " |", "".join(separator)]

    for cls, m in metrics.items():
        row = [
            cls,
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1_score']:.4f}",
            str(m["support"]),
            str(m["true_positives"]),
            str(m["false_positives"]),
            str(m["false_negatives"]),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 3.1 Ground-Truth Evaluation Runner")
    parser.add_argument("--samples-dir", type=str, default="data/samples/mcfd", help="Directory containing MCFD AVI videos")
    parser.add_argument("--output-dir", type=str, default="data/processed/mcfd", help="Directory to save evaluation artifacts")
    parser.add_argument("--device", type=str, default=None, help="Device override for YOLO (cuda or cpu)")
    parser.add_argument("--no-smoothing", action="store_true", help="Disable temporal posture smoothing")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing YOLO11-Pose detector and PostureClassifier...")
    detector = YOLOPoseDetector(device=args.device)
    classifier = PostureClassifier()
    enable_smoothing = not args.no_smoothing

    all_y_true: List[str] = []
    all_y_pred: List[str] = []
    video_reports: List[Dict[str, Any]] = []

    logger.info("=" * 70)
    logger.info("STARTING PHASE 3.1 GROUND-TRUTH EVALUATION ACROSS 6 MCFD VIDEOS")
    logger.info("=" * 70)

    for item in TEST_VIDEOS:
        v_name = item["filename"]
        chute_id = item["chute"]
        cam_id = item["cam"]
        v_path = samples_dir / v_name

        if not v_path.exists():
            logger.error(f"Test video missing: {v_path}")
            continue

        smoother = PostureSmoother(window_size=5) if enable_smoothing else None

        logger.info(f"Processing {v_name} (Chute {chute_id}, Cam {cam_id})...")
        predictions, confidences, total_frames, elapsed = extract_frame_predictions(
            video_path=v_path,
            detector=detector,
            classifier=classifier,
            smoother=smoother,
        )

        gt_labels = get_video_ground_truth(chute_id, cam_id, total_frames)

        # Save per-frame predictions without touching existing summary files
        stem = v_path.stem
        frame_pred_file = output_dir / f"{stem}_frame_predictions.json"
        frame_data = {
            "video": v_name,
            "chute_id": chute_id,
            "cam_id": cam_id,
            "total_frames": total_frames,
            "elapsed_seconds": round(elapsed, 2),
            "fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0.0,
            "frame_predictions": [
                {
                    "frame": idx,
                    "ground_truth": gt_labels[idx],
                    "predicted": predictions[idx],
                    "confidence": round(confidences[idx], 4),
                }
                for idx in range(total_frames)
            ],
        }
        with open(frame_pred_file, "w", encoding="utf-8") as f:
            json.dump(frame_data, f, indent=2)

        # Video-level contingency matrix
        v_matrix = compute_confusion_matrix(gt_labels, predictions, GROUND_TRUTH_ORDER, PREDICTION_ORDER)
        v_sofa_ground = analyze_sofa_vs_ground(gt_labels, predictions)
        v_crouch = analyze_crouching_false_positives(gt_labels, predictions)

        v_summary = {
            "video": v_name,
            "chute_id": chute_id,
            "cam_id": cam_id,
            "total_frames": total_frames,
            "fps": round(total_frames / elapsed, 2) if elapsed > 0 else 0.0,
            "confusion_matrix": v_matrix,
            "sofa_vs_ground": v_sofa_ground,
            "crouching_analysis": v_crouch,
        }
        video_reports.append(v_summary)

        all_y_true.extend(gt_labels)
        all_y_pred.extend(predictions)

        logger.info(f"Completed {v_name}: {total_frames} frames in {elapsed:.1f}s ({total_frames / elapsed:.1f} FPS)")

    # 1. Overall Multi-Class Confusion Matrix (All Frames)
    overall_matrix = compute_confusion_matrix(all_y_true, all_y_pred, GROUND_TRUTH_ORDER, PREDICTION_ORDER)

    # 2. Detector-Conditioned Evaluation (Evaluates posture geometry on detected frames only)
    detected_mask = [p != "NO_DETECTION" for p in all_y_pred]
    det_y_true = [yt for yt, m in zip(all_y_true, detected_mask) if m]
    det_y_pred = [yp for yp, m in zip(all_y_pred, detected_mask) if m]

    # Map semantic GT to standard 3 posture classes for precision/recall calculation
    # STANDING -> STANDING, SITTING -> SITTING, (SOFA_LYING + GROUND_LYING) -> LYING
    def map_gt_to_posture(gt: str) -> Optional[str]:
        if gt == GroundTruthLabel.STANDING.value:
            return PostureType.STANDING.value
        elif gt == GroundTruthLabel.SITTING.value:
            return PostureType.SITTING.value
        elif gt in (GroundTruthLabel.GROUND_LYING.value, GroundTruthLabel.SOFA_LYING.value):
            return PostureType.LYING.value
        return None

    mapped_gt_det = [map_gt_to_posture(yt) for yt in det_y_true]
    valid_pairs_det = [(yt, yp) for yt, yp in zip(mapped_gt_det, det_y_pred) if yt is not None]

    if valid_pairs_det:
        v_gt_det, v_pred_det = zip(*valid_pairs_det)
        det_posture_metrics = compute_classification_metrics(
            list(v_gt_det),
            list(v_pred_det),
            target_classes=[PostureType.STANDING.value, PostureType.SITTING.value, PostureType.LYING.value],
        )
    else:
        det_posture_metrics = {}

    # 3. End-to-End Evaluation (All frames, missed detection counts as False Negative)
    mapped_gt_all = [map_gt_to_posture(yt) for yt in all_y_true]
    valid_pairs_all = [(yt, yp) for yt, yp in zip(mapped_gt_all, all_y_pred) if yt is not None]
    if valid_pairs_all:
        v_gt_all, v_pred_all = zip(*valid_pairs_all)
        e2e_posture_metrics = compute_classification_metrics(
            list(v_gt_all),
            list(v_pred_all),
            target_classes=[PostureType.STANDING.value, PostureType.SITTING.value, PostureType.LYING.value],
        )
    else:
        e2e_posture_metrics = {}

    # 4. Specialized Analyses
    overall_sofa_ground = analyze_sofa_vs_ground(all_y_true, all_y_pred)
    overall_crouch = analyze_crouching_false_positives(all_y_true, all_y_pred)

    # 5. Compile Final Evaluation Report
    final_report = {
        "evaluation_name": "Phase 3.1 Ground-Truth Evaluation",
        "total_test_videos": len(TEST_VIDEOS),
        "total_frames_evaluated": len(all_y_true),
        "hardware_device": detector.get_device_info(),
        "temporal_smoothing_enabled": enable_smoothing,
        "overall_confusion_matrix": overall_matrix,
        "detector_conditioned_metrics": det_posture_metrics,
        "end_to_end_frame_metrics": e2e_posture_metrics,
        "sofa_vs_ground_analysis": overall_sofa_ground,
        "crouching_false_positive_analysis": overall_crouch,
        "per_video_summaries": video_reports,
    }

    report_path = output_dir / "phase3_1_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Evaluation report written to {report_path}")

    # Print summary to console
    print("\n" + "=" * 78)
    print("PHASE 3.1 GROUND-TRUTH EVALUATION REPORT")
    print("=" * 78)
    print(f"Total Videos: {len(TEST_VIDEOS)} | Total Frames: {len(all_y_true):,}")
    print(f"Hardware: {detector.get_device_info()} | Smoothing: {enable_smoothing}\n")

    print("### 1. OVERALL MULTI-CLASS CONFUSION MATRIX (ALL FRAMES)")
    print(format_confusion_matrix_table(overall_matrix))

    print("\n### 2. DETECTOR-CONDITIONED POSTURE METRICS (Frames with Person Detected)")
    print(format_metrics_table(det_posture_metrics))

    print("\n### 3. END-TO-END PIPELINE METRICS (All Video Frames including Detector Dropouts)")
    print(format_metrics_table(e2e_posture_metrics))

    print("\n### 4. DOMAIN ANALYSIS: SOFA LYING vs GROUND LYING")
    print(f"- Sofa Lying Ground Truth Frames: {overall_sofa_ground['sofa_lying_frames']}")
    print(f"- Sofa Frames Predicted as LYING: {overall_sofa_ground['sofa_predicted_lying']} ({overall_sofa_ground['sofa_lying_detection_rate']*100:.1f}%)")
    print(f"- Ground Lying Ground Truth Frames: {overall_sofa_ground['ground_lying_frames']}")
    print(f"- Ground Frames Predicted as LYING: {overall_sofa_ground['ground_predicted_lying']} ({overall_sofa_ground['ground_lying_detection_rate']*100:.1f}%)")
    print(f"- Ground Detector Dropouts (NO_DETECTION): {overall_sofa_ground['ground_lying_detector_dropouts']} ({overall_sofa_ground['ground_dropout_rate']*100:.1f}%)")

    print("\n### 5. DOMAIN ANALYSIS: CROUCHING FALSE POSITIVE RATE")
    print(f"- Crouching Ground Truth Frames: {overall_crouch['crouching_total_frames']}")
    print(f"- Crouching Misclassified as LYING: {overall_crouch['crouching_predicted_lying']} ({overall_crouch['crouching_false_lying_rate']*100:.2f}%)")
    print(f"- Crouching Classified as STANDING: {overall_crouch['crouching_predicted_standing']}")
    print(f"- Crouching Classified as SITTING: {overall_crouch['crouching_predicted_sitting']}")
    print(f"- Crouching Classified as UNKNOWN: {overall_crouch['crouching_predicted_unknown']}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
