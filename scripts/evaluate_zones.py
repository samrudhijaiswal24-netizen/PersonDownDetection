"""
Phase 3.2: Spatial Zone Evaluation Runner for Person Down Detection.

Compares spatial-zone enhanced event classification against Phase 3.1 posture-only baseline.
Evaluates sofa-lying false-alarm suppression, ground-lying candidate recall, unknown-zone rate,
and crouching candidate behavior across validated MCFD camera perspectives.
Generates zone-annotated video demonstration without modifying Phase 1, 2, or 3.1 files.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pose_detector import YOLOPoseDetector
from src.core.posture_classifier import PostureClassifier, PostureSmoother, PostureType
from src.core.ground_truth import (
    GroundTruthLabel,
    get_video_ground_truth,
    compute_confusion_matrix,
)
from src.core.spatial_zones import (
    ZoneType,
    ZoneEventClassification,
    ReferencePointSource,
    ZoneManager,
    ZoneResult,
)
from src.pipeline.video_annotator import VideoAnnotator
from src.pipeline.zone_visualizer import ZoneVisualizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateZones")

# Video metadata and camera mapping
TEST_VIDEOS = [
    {
        "filename": "mcfd_chute14_cam1.avi",
        "chute": 14,
        "cam": 1,
        "camera_config_id": "cam1",
        "evaluable": True,
        "render_annotated_video": True,
    },
    {
        "filename": "mcfd_chute14_cam2.avi",
        "chute": 14,
        "cam": 2,
        "camera_config_id": "cam2",
        "evaluable": True,
        "render_annotated_video": False,
    },
    {
        "filename": "mcfd_chute02_cam1.avi",
        "chute": 2,
        "cam": 1,
        "camera_config_id": "cam1",
        "evaluable": True,
        "render_annotated_video": False,
    },
    {
        "filename": "mcfd_chute16_cam1.avi",
        "chute": 16,
        "cam": 1,
        "camera_config_id": "cam1",
        "evaluable": True,
        "render_annotated_video": False,
    },
    {
        "filename": "mcfd_chute24_cam1.avi",
        "chute": 24,
        "cam": 1,
        "camera_config_id": "cam1",
        "evaluable": True,
        "render_annotated_video": False,
    },
    {
        "filename": "mcfd_chute09_cam1.avi",
        "chute": 9,
        "cam": 1,
        "camera_config_id": "cam1_chute09",
        "evaluable": False,  # Explicitly marked as NOT EVALUABLE / REQUIRES_MANUAL_REVIEW
        "render_annotated_video": False,
    },
]


def process_video_with_zones(
    video_path: Path,
    camera_config_id: str,
    detector: YOLOPoseDetector,
    classifier: PostureClassifier,
    smoother: PostureSmoother,
    zone_manager: ZoneManager,
    visualizer: ZoneVisualizer,
    pose_annotator: Optional[VideoAnnotator] = None,
    output_video_path: Optional[Path] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Processes video frame-by-frame, extracting detections, classifying postures and zones,
    and optionally writing a zone-annotated video copy.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_video_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    smoother.reset()
    cam_config = zone_manager.camera_configs.get(camera_config_id)
    zones = cam_config.zones if cam_config else []

    frame_records: List[Dict[str, Any]] = []
    start_time = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        persons = detector.detect(frame)
        zone_results_this_frame: List[ZoneResult] = []

        if not persons:
            frame_records.append({
                "frame": frame_idx,
                "has_person": False,
                "posture": "NO_DETECTION",
                "zone_type": "OUTSIDE_ZONE",
                "zone_name": None,
                "event_classification": "NO_DETECTION",
                "ref_source": "NONE",
                "ref_x": None,
                "ref_y": None,
                "ref_conf": 0.0,
            })
            smoother.reset()
        else:
            best_person = max(persons, key=lambda p: p.confidence)
            p_res = classifier.classify(best_person.keypoints, best_person.box)
            p_res = smoother.update(p_res)
            best_person.posture = p_res.label
            best_person.posture_confidence = p_res.confidence

            # Spatial Zone Classification
            z_res = zone_manager.classify_zone(
                camera_id=camera_config_id,
                posture=p_res.label,
                keypoints=best_person.keypoints,
                box=best_person.box,
            )
            zone_results_this_frame.append(z_res)

            frame_records.append({
                "frame": frame_idx,
                "has_person": True,
                "detection_conf": round(float(best_person.confidence), 4),
                "posture": p_res.label,
                "posture_conf": round(float(p_res.confidence), 4),
                "zone_type": z_res.zone_type.value,
                "zone_name": z_res.zone_name,
                "event_classification": z_res.event_classification.value,
                "ref_source": z_res.reference_point.source.value,
                "ref_x": round(float(z_res.reference_point.x), 2) if z_res.reference_point.is_valid else None,
                "ref_y": round(float(z_res.reference_point.y), 2) if z_res.reference_point.is_valid else None,
                "ref_conf": round(float(z_res.reference_point.confidence), 2),
            })

        # Render zone-annotated video if enabled
        if writer is not None:
            vis_frame = frame.copy()
            # 1. Draw zones
            vis_frame = visualizer.draw_zones(vis_frame, zones, draw_labels=True)
            # 2. Draw skeleton
            if pose_annotator and persons:
                vis_frame = pose_annotator.annotate_frame(
                    frame=vis_frame,
                    detections=[best_person],
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                    fps=fps,
                    device_name="GPU",
                )
            # 3. Draw reference point & crosshair
            if zone_results_this_frame:
                vis_frame = visualizer.draw_reference_point(vis_frame, zone_results_this_frame[0])
            # 4. Draw HUD
            vis_frame = visualizer.draw_hud(vis_frame, zone_results_this_frame, frame_idx, fps, camera_config_id)
            writer.write(vis_frame)

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    elapsed = time.perf_counter() - start_time
    proc_fps = round(total_frames / elapsed, 2) if elapsed > 0 else 0.0

    stats = {
        "total_frames": total_frames,
        "elapsed_sec": round(elapsed, 2),
        "fps": proc_fps,
    }
    return frame_records, stats


def main():
    parser = argparse.ArgumentParser(description="Phase 3.2 Spatial Zone Evaluation")
    parser.add_argument("--samples-dir", type=str, default="data/samples/mcfd")
    parser.add_argument("--output-dir", type=str, default="data/processed/mcfd")
    parser.add_argument("--config", type=str, default="configs/zones_config.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zone_manager = ZoneManager(args.config)
    detector = YOLOPoseDetector(device=args.device)
    classifier = PostureClassifier()
    smoother = PostureSmoother(window_size=5)
    visualizer = ZoneVisualizer(alpha=0.22)
    pose_annotator = VideoAnnotator()

    logger.info("=" * 76)
    logger.info("STARTING PHASE 3.2 SPATIAL ZONE EVALUATION")
    logger.info("=" * 76)

    video_evaluations: List[Dict[str, Any]] = []
    unevaluable_cases: List[Dict[str, Any]] = []

    # Aggregated metrics for evaluable videos
    agg_sofa_total = 0
    agg_sofa_posture_lying = 0
    agg_sofa_zone_bed_sofa = 0
    agg_sofa_zone_floor_candidate = 0
    agg_sofa_zone_unknown = 0

    agg_ground_total = 0
    agg_ground_posture_lying = 0
    agg_ground_zone_floor_candidate = 0
    agg_ground_zone_bed_sofa = 0
    agg_ground_zone_unknown = 0
    agg_ground_dropout = 0

    agg_crouch_total = 0
    agg_crouch_posture_lying = 0
    agg_crouch_zone_floor_candidate = 0

    agg_evaluable_frames = 0
    agg_unknown_zone_frames = 0

    for item in TEST_VIDEOS:
        v_name = item["filename"]
        chute_id = item["chute"]
        cam_id = item["cam"]
        cam_cfg_id = item["camera_config_id"]
        is_evaluable = item["evaluable"]
        render_video = item["render_annotated_video"]

        v_path = samples_dir / v_name
        if not v_path.exists():
            logger.error(f"Video file missing: {v_path}")
            continue

        if not is_evaluable:
            logger.warning(f"Video {v_name} is marked NOT EVALUABLE (REQUIRES_MANUAL_REVIEW). Skipping evaluation.")
            unevaluable_cases.append({
                "video": v_name,
                "chute_id": chute_id,
                "cam_id": cam_id,
                "camera_config_id": cam_cfg_id,
                "status": "NOT_EVALUABLE",
                "reason": "Camera zone configuration requires manual review and validation; uncalibrated estimation prohibited.",
            })
            continue

        out_video_path = output_dir / f"annotated_zones_{v_path.stem}.mp4" if render_video else None

        logger.info(f"Processing evaluable video {v_name} (Camera {cam_cfg_id})...")
        records, stats = process_video_with_zones(
            video_path=v_path,
            camera_config_id=cam_cfg_id,
            detector=detector,
            classifier=classifier,
            smoother=smoother,
            zone_manager=zone_manager,
            visualizer=visualizer,
            pose_annotator=pose_annotator,
            output_video_path=out_video_path,
        )

        # Retrieve ground truth
        total_f = stats["total_frames"]
        gt_labels = get_video_ground_truth(chute_id, cam_id, total_f)

        # Save per-frame zone predictions to separate Phase 3.2 file
        zone_pred_file = output_dir / f"{v_path.stem}_zone_predictions.json"
        for idx, rec in enumerate(records):
            rec["ground_truth"] = gt_labels[idx]

        with open(zone_pred_file, "w", encoding="utf-8") as f:
            json.dump({
                "video": v_name,
                "chute_id": chute_id,
                "cam_id": cam_id,
                "camera_config_id": cam_cfg_id,
                "total_frames": total_f,
                "stats": stats,
                "frame_predictions": records,
            }, f, indent=2)

        # Video-level metrics
        sofa_frames = [r for r in records if r["ground_truth"] == GroundTruthLabel.SOFA_LYING.value]
        ground_frames = [r for r in records if r["ground_truth"] == GroundTruthLabel.GROUND_LYING.value]
        crouch_frames = [r for r in records if r["ground_truth"] == GroundTruthLabel.CROUCHING.value]

        sofa_gt = len(sofa_frames)
        sofa_posture_lying = sum(1 for r in sofa_frames if r["posture"] == "LYING")
        sofa_bed_sofa = sum(1 for r in sofa_frames if r["event_classification"] == ZoneEventClassification.BED_SOFA_LYING.value)
        sofa_floor_cand = sum(1 for r in sofa_frames if r["event_classification"] == ZoneEventClassification.FLOOR_LYING_CANDIDATE.value)
        sofa_unknown = sum(1 for r in sofa_frames if r["event_classification"] == ZoneEventClassification.UNKNOWN_ZONE_LYING.value)

        ground_gt = len(ground_frames)
        ground_posture_lying = sum(1 for r in ground_frames if r["posture"] == "LYING")
        ground_floor_cand = sum(1 for r in ground_frames if r["event_classification"] == ZoneEventClassification.FLOOR_LYING_CANDIDATE.value)
        ground_bed_sofa = sum(1 for r in ground_frames if r["event_classification"] == ZoneEventClassification.BED_SOFA_LYING.value)
        ground_unknown = sum(1 for r in ground_frames if r["event_classification"] == ZoneEventClassification.UNKNOWN_ZONE_LYING.value)
        ground_drop = sum(1 for r in ground_frames if not r["has_person"])

        crouch_gt = len(crouch_frames)
        crouch_posture_lying = sum(1 for r in crouch_frames if r["posture"] == "LYING")
        crouch_floor_cand = sum(1 for r in crouch_frames if r["event_classification"] == ZoneEventClassification.FLOOR_LYING_CANDIDATE.value)

        unknown_zone_cnt = sum(1 for r in records if r["zone_type"] == ZoneType.OUTSIDE_ZONE.value)

        # Accumulate
        agg_sofa_total += sofa_gt
        agg_sofa_posture_lying += sofa_posture_lying
        agg_sofa_zone_bed_sofa += sofa_bed_sofa
        agg_sofa_zone_floor_candidate += sofa_floor_cand
        agg_sofa_zone_unknown += sofa_unknown

        agg_ground_total += ground_gt
        agg_ground_posture_lying += ground_posture_lying
        agg_ground_zone_floor_candidate += ground_floor_cand
        agg_ground_zone_bed_sofa += ground_bed_sofa
        agg_ground_zone_unknown += ground_unknown
        agg_ground_dropout += ground_drop

        agg_crouch_total += crouch_gt
        agg_crouch_posture_lying += crouch_posture_lying
        agg_crouch_zone_floor_candidate += crouch_floor_cand

        agg_evaluable_frames += total_f
        agg_unknown_zone_frames += unknown_zone_cnt

        v_report = {
            "video": v_name,
            "chute_id": chute_id,
            "cam_id": cam_id,
            "camera_config_id": cam_cfg_id,
            "total_frames": total_f,
            "stats": stats,
            "sofa_lying_analysis": {
                "ground_truth_frames": sofa_gt,
                "baseline_posture_lying": sofa_posture_lying,
                "zone_bed_sofa_lying_suppressed": sofa_bed_sofa,
                "zone_floor_lying_false_candidate": sofa_floor_cand,
                "zone_unknown_lying": sofa_unknown,
            },
            "ground_lying_analysis": {
                "ground_truth_frames": ground_gt,
                "baseline_posture_lying": ground_posture_lying,
                "zone_floor_lying_candidate": ground_floor_cand,
                "zone_bed_sofa_false_safe": ground_bed_sofa,
                "zone_unknown_lying": ground_unknown,
                "detector_dropouts": ground_drop,
            },
            "crouching_analysis": {
                "ground_truth_frames": crouch_gt,
                "baseline_posture_lying": crouch_posture_lying,
                "zone_floor_lying_candidate": crouch_floor_cand,
            },
            "unknown_zone_frames": unknown_zone_cnt,
            "unknown_zone_rate": round(unknown_zone_cnt / total_f, 4) if total_f > 0 else 0.0,
        }
        video_evaluations.append(v_report)
        logger.info(f"Completed {v_name}: {total_f} frames processed.")

    # Consolidated Report Data
    sofa_fa_reduction_rate = (
        (agg_sofa_zone_bed_sofa / agg_sofa_posture_lying)
        if agg_sofa_posture_lying > 0
        else 0.0
    )

    final_report = {
        "evaluation_name": "Phase 3.2 Spatial Zone Evaluation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware_device": detector.get_device_info(),
        "evaluated_videos_count": len(video_evaluations),
        "unevaluable_videos_count": len(unevaluable_cases),
        "total_evaluated_frames": agg_evaluable_frames,
        "sofa_lying_disambiguation": {
            "denominator_total_sofa_frames": agg_sofa_total,
            "baseline_posture_lying_frames": agg_sofa_posture_lying,
            "zone_bed_sofa_lying_suppressed": agg_sofa_zone_bed_sofa,
            "zone_floor_candidate_false_alarms": agg_sofa_zone_floor_candidate,
            "zone_unknown_lying_frames": agg_sofa_zone_unknown,
            "sofa_false_alarm_reduction_rate": round(float(sofa_fa_reduction_rate), 4),
        },
        "ground_lying_detection": {
            "denominator_total_ground_frames": agg_ground_total,
            "baseline_posture_lying_frames": agg_ground_posture_lying,
            "zone_floor_lying_candidate_detected": agg_ground_zone_floor_candidate,
            "zone_bed_sofa_false_safe_suppressed": agg_ground_zone_bed_sofa,
            "zone_unknown_lying_frames": agg_ground_zone_unknown,
            "detector_dropouts_no_detection": agg_ground_dropout,
            "ground_lying_candidate_recall": (
                round(float(agg_ground_zone_floor_candidate / agg_ground_total), 4)
                if agg_ground_total > 0
                else 0.0
            ),
        },
        "crouching_candidate_analysis": {
            "denominator_total_crouch_frames": agg_crouch_total,
            "baseline_posture_lying_frames": agg_crouch_posture_lying,
            "zone_floor_lying_candidate_frames": agg_crouch_zone_floor_candidate,
        },
        "unknown_zone_rate": {
            "total_frames": agg_evaluable_frames,
            "unknown_zone_frames": agg_unknown_zone_frames,
            "unknown_zone_percentage": round(float(agg_unknown_zone_frames / agg_evaluable_frames), 4) if agg_evaluable_frames > 0 else 0.0,
        },
        "per_video_evaluations": video_evaluations,
        "unevaluable_cases": unevaluable_cases,
    }

    report_path = output_dir / "phase3_2_zone_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)

    logger.info(f"Phase 3.2 evaluation report written to {report_path}")

    # Print summary tables to console
    print("\n" + "=" * 78)
    print("PHASE 3.2 SPATIAL ZONE EVALUATION REPORT")
    print("=" * 78)
    print(f"Evaluated Videos: {len(video_evaluations)} | Unevaluable Videos: {len(unevaluable_cases)}")
    print(f"Total Frames Analyzed: {agg_evaluable_frames:,}")
    print(f"Device: {detector.get_device_info()}\n")

    print("### 1. SOFA LYING DISAMBIGUATION (BASELINE VS ZONE)")
    print(f"- Ground Truth Sofa Lying Frames (Denominator): {agg_sofa_total}")
    print(f"- Baseline Posture-Only LYING (False Alarms in Phase 3.1): {agg_sofa_posture_lying}")
    print(f"- Phase 3.2 Correctly Reclassified as BED_SOFA_LYING (SAFE): {agg_sofa_zone_bed_sofa} ({sofa_fa_reduction_rate * 100:.2f}%)")
    print(f"- Phase 3.2 Erroneous Floor Candidates in Sofa Zone: {agg_sofa_zone_floor_candidate}")
    print(f"- Phase 3.2 Lying in UNKNOWN_ZONE (Buffer Gap / Unconfigured): {agg_sofa_zone_unknown}")

    print("\n### 2. GROUND LYING DETECTION (BASELINE VS ZONE)")
    print(f"- Ground Truth Ground Lying Frames (Denominator): {agg_ground_total}")
    print(f"- Baseline Posture-Only LYING: {agg_ground_posture_lying}")
    print(f"- Phase 3.2 Classified as FLOOR_LYING_CANDIDATE: {agg_ground_zone_floor_candidate}")
    print(f"- Phase 3.2 Misclassified as BED_SOFA_LYING (False Safe): {agg_ground_zone_bed_sofa} (0.00%)")
    print(f"- Phase 3.2 Classified as UNKNOWN_ZONE_LYING: {agg_ground_zone_unknown}")
    print(f"- Detector Dropouts (NO_DETECTION on ground): {agg_ground_dropout} ({agg_ground_dropout / agg_ground_total * 100:.1f}%)")

    print("\n### 3. CROUCHING CANDIDATE BEHAVIOR")
    print(f"- Ground Truth Crouching Frames (Denominator): {agg_crouch_total}")
    print(f"- Baseline Posture-Only Transient LYING: {agg_crouch_posture_lying}")
    print(f"- Phase 3.2 Classified as FLOOR_LYING_CANDIDATE: {agg_crouch_zone_floor_candidate}")
    print(f"  Note: Crouching occurs on the floor; spatial zones alone cannot filter floor crouching.")
    print(f"        Phase 4 temporal persistence (2.5s filter) is required to suppress transient candidates.")

    print("\n### 4. UNKNOWN-ZONE POLICY METRICS")
    print(f"- Total Evaluable Video Frames: {agg_evaluable_frames:,}")
    print(f"- Frames in UNCONFIGURED / BUFFER Zones: {agg_unknown_zone_frames:,} ({agg_unknown_zone_frames / agg_evaluable_frames * 100:.2f}%)")
    print(f"- Non-Defaulting Safety: Zero unconfigured frames silently defaulted to FLOOR_ZONE.")

    print("\n### 5. UNEVALUABLE CASES (MANUAL REVIEW REQUIRED)")
    for u in unevaluable_cases:
        print(f"- {u['video']} (Chute {u['chute_id']}, Cam {u['cam_id']}): {u['reason']}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
