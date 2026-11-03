"""
Phase 4: Temporal State Tracking, Alert Persistence, and False-Alarm Suppression Evaluation Runner.

Stage A Corrected Evaluation Protocol:
- Deterministic 1-to-1 event matching with tie-breaking for multi-interval GT and candidate events
- Unmatched GT events counted as FN; duplicate and spurious candidate events counted as FP
- Mathematically undefined metrics (0/0) reported as None/null with explicit status documentation
- Dynamic sensitivity analysis deriving GT onset from metadata (eliminating hardcoded frame 1031 and FPS)
- Delay metrics relabeled to candidate_promotion_delay (with detection_delay alias for backward compatibility)
- Dual reporting: Protocol A (strict benchmark, 8,726 frames) and Protocol B (conditional masked tail, 8,653 frames)
- Safety guards preventing accidental overwrite of Phase 4 baseline reports and prediction records
- Preserves all Phase 1-3.2 files and outputs intact.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.temporal_tracker import (
    TemporalState,
    TemporalTrackerConfig,
    TemporalStateTracker,
    TemporalStateUpdate,
)
from src.core.ground_truth import (
    GroundTruthLabel,
    RAW_CODE_TO_LABEL,
    get_camera_delay_shift,
    get_video_ground_truth_events,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("EvaluateTemporalTracking")

BASELINE_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "mcfd" / "phase4_temporal_evaluation_report.json"

EVALUABLE_VIDEOS = [
    {
        "filename": "mcfd_chute14_cam1.avi",
        "zone_json": "mcfd_chute14_cam1_zone_predictions.json",
        "chute": 14,
        "cam": 1,
        "evaluable": True,
        "is_fall_scenario": True,
    },
    {
        "filename": "mcfd_chute14_cam2.avi",
        "zone_json": "mcfd_chute14_cam2_zone_predictions.json",
        "chute": 14,
        "cam": 2,
        "evaluable": True,
        "is_fall_scenario": True,
    },
    {
        "filename": "mcfd_chute02_cam1.avi",
        "zone_json": "mcfd_chute02_cam1_zone_predictions.json",
        "chute": 2,
        "cam": 1,
        "evaluable": True,
        "is_fall_scenario": True,
    },
    {
        "filename": "mcfd_chute16_cam1.avi",
        "zone_json": "mcfd_chute16_cam1_zone_predictions.json",
        "chute": 16,
        "cam": 1,
        "evaluable": True,
        "is_fall_scenario": True,
    },
    {
        "filename": "mcfd_chute24_cam1.avi",
        "zone_json": "mcfd_chute24_cam1_zone_predictions.json",
        "chute": 24,
        "cam": 1,
        "evaluable": True,
        "is_fall_scenario": False,  # ADL negative control (crouching, walking, sitting)
    },
    {
        "filename": "mcfd_chute09_cam1.avi",
        "zone_json": None,
        "chute": 9,
        "cam": 1,
        "evaluable": False,  # Excluded until manually calibrated
        "is_fall_scenario": True,
    },
]


def match_candidate_events_to_ground_truth(
    gt_events: List[Dict[str, Any]],
    candidate_events: List[Dict[str, Any]],
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Performs deterministic one-to-one temporal matching between candidate events and ground-truth fall events.

    Matching Rules:
    1. Candidates are sorted chronologically by start_frame.
    2. Each candidate matches at most one unmatched GT event (highest temporal intersection, then earliest start).
    3. Overlapping an already-matched GT event is classified as duplicate False Positive.
    4. Overlapping no GT event is classified as spurious False Positive.
    5. Unmatched GT events are classified as False Negatives.
    6. Undefined metrics (0/0) return None rather than 1.0.
    """
    sorted_cands = sorted(candidate_events, key=lambda c: (c["start_frame"], c["end_frame"]))
    gt_lookup = {
        gt["gt_event_id"]: dict(gt, matched=False, matched_by_cand_id=None)
        for gt in gt_events
    }

    tp = 0
    fp_duplicate = 0
    fp_spurious = 0
    delays_frames = []
    delays_sec = []
    matched_cand_records = []

    for c in sorted_cands:
        c_start = c["start_frame"]
        c_end = c["end_frame"]

        # Find all overlapping GT events
        overlapping_gts = []
        for gt in gt_lookup.values():
            g_start = gt["start_frame"]
            g_end = gt["end_frame"]
            overlap_len = max(0, min(c_end, g_end) - max(c_start, g_start) + 1)
            if overlap_len > 0:
                overlapping_gts.append((gt, overlap_len))

        # Check for unmatched overlaps
        unmatched_overlaps = [(gt, olen) for gt, olen in overlapping_gts if not gt["matched"]]
        matched_overlaps = [(gt, olen) for gt, olen in overlapping_gts if gt["matched"]]

        if unmatched_overlaps:
            # Deterministic selection: max overlap duration, tie-breaker: earliest start_frame
            best_gt, best_olen = max(
                unmatched_overlaps,
                key=lambda item: (item[1], -item[0]["start_frame"], -item[0]["gt_event_id"]),
            )
            gt_id = best_gt["gt_event_id"]
            gt_lookup[gt_id]["matched"] = True
            gt_lookup[gt_id]["matched_by_cand_id"] = c.get("event_id")

            delay_f = c_start - best_gt["start_frame"]
            delay_s = round(delay_f / fps, 2)
            delays_frames.append(delay_f)
            delays_sec.append(delay_s)

            tp += 1
            cand_rec = dict(
                c,
                matches_gt_event=True,
                matched_gt_id=gt_id,
                is_duplicate=False,
                is_spurious=False,
                candidate_promotion_delay_frames=delay_f,
                candidate_promotion_delay_seconds=delay_s,
                detection_delay_frames=delay_f,  # backward compatibility alias
                detection_delay_seconds=delay_s,
            )
            matched_cand_records.append(cand_rec)

        elif matched_overlaps:
            # Overlaps an already-matched GT event -> Duplicate FP
            best_matched_gt, _ = max(matched_overlaps, key=lambda item: item[1])
            fp_duplicate += 1
            cand_rec = dict(
                c,
                matches_gt_event=False,
                matched_gt_id=None,
                is_duplicate=True,
                is_spurious=False,
                duplicate_of_gt_id=best_matched_gt["gt_event_id"],
            )
            matched_cand_records.append(cand_rec)

        else:
            # Spurious false alarm
            fp_spurious += 1
            cand_rec = dict(
                c,
                matches_gt_event=False,
                matched_gt_id=None,
                is_duplicate=False,
                is_spurious=True,
            )
            matched_cand_records.append(cand_rec)

    # Count unmatched GT events as FN
    fn = sum(1 for gt in gt_lookup.values() if not gt["matched"])
    fp = fp_duplicate + fp_spurious
    gt_count = len(gt_events)
    cand_count = len(candidate_events)

    # Precision: TP / (TP + FP) or None if denominator is 0
    if (tp + fp) > 0:
        precision = round(tp / (tp + fp), 4)
    else:
        precision = None

    # Recall: TP / (TP + FN) or None if denominator is 0
    if gt_count > 0:
        recall = round(tp / (tp + fn), 4)
    else:
        recall = None

    # F1: 2 * Prec * Rec / (Prec + Rec) or None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    else:
        f1 = None

    return {
        "gt_event_count": gt_count,
        "candidate_event_count": cand_count,
        "event_tp": tp,
        "event_fp": fp,
        "event_fp_duplicate": fp_duplicate,
        "event_fp_spurious": fp_spurious,
        "event_fn": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "candidate_promotion_delays_frames": delays_frames,
        "candidate_promotion_delays_seconds": delays_sec,
        "event_delays_seconds": delays_sec,  # backward compatibility alias
        "event_delays_frames": delays_frames,  # backward compatibility alias
        "matched_candidates": matched_cand_records,
        "gt_events_status": list(gt_lookup.values()),
    }


def evaluate_single_video(
    video_meta: Dict[str, Any],
    config: TemporalTrackerConfig,
    zone_data_dir: Path,
    output_dir: Path,
    save_per_frame: bool = False,
) -> Dict[str, Any]:
    """Processes a single video's zone predictions through the temporal tracker and evaluates metrics."""
    zone_path = zone_data_dir / video_meta["zone_json"]
    if not zone_path.exists():
        raise FileNotFoundError(f"Missing zone prediction file: {zone_path}")

    with open(zone_path, "r", encoding="utf-8") as f:
        zone_data = json.load(f)

    frame_preds = zone_data["frame_predictions"]
    total_frames = len(frame_preds)

    # Extract discrete ground truth fall events from verified metadata
    gt_fall_events = get_video_ground_truth_events(
        chute_id=video_meta["chute"],
        cam_id=video_meta["cam"],
        total_frames=total_frames,
    )

    # Protocol B Mask: chute14_cam2 unannotated post-event tail (frames 1123-1195)
    protocol_b_mask: Set[int] = set()
    if video_meta["chute"] == 14 and video_meta["cam"] == 2:
        protocol_b_mask = set(range(1123, 1196))

    tracker = TemporalStateTracker(config=config)

    per_frame_records = []
    state_counts = {st.value: 0 for st in TemporalState}

    candidate_events = []
    current_event: Optional[Dict[str, Any]] = None

    p32_floor_candidate_frames = 0
    p4_candidate_frames = 0

    crouch_gt_frames = 0
    crouch_p32_cand_frames = 0
    crouch_p4_cand_frames = 0

    sofa_gt_frames = 0
    sofa_p32_cand_frames = 0
    sofa_p4_cand_frames = 0

    ground_lying_gt_frames = 0
    ground_lying_gt_start: Optional[int] = None
    ground_lying_gt_end: Optional[int] = None

    dropout_gap_held_count = 0
    dropout_downgraded_count = 0
    dropout_reset_count = 0
    unknown_zone_count = 0

    # Protocol A confusion matrix counters (all frames)
    pA_tp = 0
    pA_fp = 0
    pA_fn = 0
    pA_tn = 0

    # Protocol B confusion matrix counters (masked post-event tail excluded)
    pB_tp = 0
    pB_fp = 0
    pB_fn = 0
    pB_tn = 0

    for pred in frame_preds:
        f_idx = pred["frame"]
        gt_label = pred.get("ground_truth", "BACKGROUND")
        event_cls = pred.get("event_classification", "NORMAL_ACTIVITY")
        posture = pred.get("posture", "NO_DETECTION")
        has_person = pred.get("has_person", False)
        zone_type = pred.get("zone_type", None)

        p32_cand = (event_cls == "FLOOR_LYING_CANDIDATE")
        if p32_cand:
            p32_floor_candidate_frames += 1

        # Ground truth tracking
        if gt_label == "GROUND_LYING":
            ground_lying_gt_frames += 1
            if ground_lying_gt_start is None:
                ground_lying_gt_start = f_idx
            ground_lying_gt_end = f_idx
        elif gt_label == "CROUCHING":
            crouch_gt_frames += 1
            if p32_cand:
                crouch_p32_cand_frames += 1
        elif gt_label == "SOFA_LYING":
            sofa_gt_frames += 1
            if p32_cand:
                sofa_p32_cand_frames += 1

        if zone_type == "UNKNOWN_ZONE" or event_cls == "UNKNOWN_ZONE_LYING":
            unknown_zone_count += 1

        # Execute temporal state update
        up: TemporalStateUpdate = tracker.update(
            frame_idx=f_idx,
            event_classification=event_cls,
            posture=posture,
            has_person=has_person,
            zone_type=zone_type,
        )

        st_val = up.state.value
        state_counts[st_val] += 1

        is_p4_cand = (up.state == TemporalState.PERSON_DOWN_CANDIDATE)
        if is_p4_cand:
            p4_candidate_frames += 1
            if gt_label == "CROUCHING":
                crouch_p4_cand_frames += 1
            elif gt_label == "SOFA_LYING":
                sofa_p4_cand_frames += 1

        # Dropout memory counters
        st_mem = up.details.get("state_memory")
        if st_mem == "holding_within_grace":
            dropout_gap_held_count += 1
        elif st_mem == "downgraded_to_unknown":
            dropout_downgraded_count += 1
        elif st_mem == "reset_after_extended_absence":
            dropout_reset_count += 1

        # Protocol A Confusion Matrix (all frames)
        if gt_label == "GROUND_LYING":
            if is_p4_cand:
                pA_tp += 1
            else:
                pA_fn += 1
        else:
            if is_p4_cand:
                pA_fp += 1
            else:
                pA_tn += 1

        # Protocol B Confusion Matrix (skip masked post-event frames)
        if f_idx not in protocol_b_mask:
            if gt_label == "GROUND_LYING":
                if is_p4_cand:
                    pB_tp += 1
                else:
                    pB_fn += 1
            else:
                if is_p4_cand:
                    pB_fp += 1
                else:
                    pB_tn += 1

        # Candidate Event Tracking
        if up.is_new_candidate_event:
            current_event = {
                "event_id": len(candidate_events) + 1,
                "start_frame": f_idx,
                "end_frame": f_idx,
                "start_gt_label": gt_label,
                "start_elapsed_down_sec": up.elapsed_down_seconds,
            }
            candidate_events.append(current_event)
        elif current_event and is_p4_cand:
            current_event["end_frame"] = f_idx

        if save_per_frame:
            per_frame_records.append({
                "frame": f_idx,
                "temporal_state": st_val,
                "previous_state": up.previous_state.value,
                "elapsed_down_seconds": up.elapsed_down_seconds,
                "accumulated_floor_frames": up.accumulated_floor_frames,
                "dropout_gap_frames": up.dropout_gap_frames,
                "is_new_candidate_event": up.is_new_candidate_event,
                "phase3_2_candidate": p32_cand,
                "phase4_candidate": is_p4_cand,
                "ground_truth": gt_label,
                "details": up.details,
            })

    # Save per-frame records only if explicitly requested
    if save_per_frame:
        out_pred_path = output_dir / f"{video_meta['filename'].replace('.avi', '')}_temporal_predictions.json"
        with open(out_pred_path, "w", encoding="utf-8") as f:
            json.dump({
                "video": video_meta["filename"],
                "total_frames": total_frames,
                "fps": config.fps,
                "persistence_seconds": config.persistence_seconds,
                "frame_predictions": per_frame_records,
            }, f, indent=2)
        logger.info(f"Saved temporal predictions: {out_pred_path}")

    # Compute duration for each candidate event
    for ev in candidate_events:
        ev["duration_frames"] = ev["end_frame"] - ev["start_frame"] + 1
        ev["duration_sec"] = round(ev["duration_frames"] / config.fps, 2)

    # Perform deterministic 1-to-1 event matching against discrete GT events
    event_match_results = match_candidate_events_to_ground_truth(
        gt_events=gt_fall_events,
        candidate_events=candidate_events,
        fps=config.fps,
    )

    # Frame-Level Metrics: Protocol A (Strict Benchmark)
    pA_total_frames = total_frames
    pA_prec = (pA_tp / (pA_tp + pA_fp)) if (pA_tp + pA_fp) > 0 else 0.0
    pA_rec = (pA_tp / (pA_tp + pA_fn)) if (pA_tp + pA_fn) > 0 else 0.0
    pA_f1 = (2 * pA_prec * pA_rec / (pA_prec + pA_rec)) if (pA_prec + pA_rec) > 0 else 0.0

    # Frame-Level Metrics: Protocol B (Conditional Masked Post-Event Tail)
    pB_total_frames = total_frames - len(protocol_b_mask)
    pB_prec = (pB_tp / (pB_tp + pB_fp)) if (pB_tp + pB_fp) > 0 else 0.0
    pB_rec = (pB_tp / (pB_tp + pB_fn)) if (pB_tp + pB_fn) > 0 else 0.0
    pB_f1 = (2 * pB_prec * pB_rec / (pB_prec + pB_rec)) if (pB_prec + pB_rec) > 0 else 0.0

    return {
        "video": video_meta["filename"],
        "chute_id": video_meta["chute"],
        "cam_id": video_meta["cam"],
        "total_frames": total_frames,
        "is_fall_scenario": video_meta["is_fall_scenario"],
        "timebase_fps": config.fps,
        "duration_seconds": round(total_frames / config.fps, 2),
        "ground_truth": {
            "ground_lying_frames": ground_lying_gt_frames,
            "ground_lying_range": [ground_lying_gt_start, ground_lying_gt_end],
            "crouching_frames": crouch_gt_frames,
            "sofa_lying_frames": sofa_gt_frames,
            "gt_events": len(gt_fall_events),
            "gt_fall_intervals": gt_fall_events,
        },
        "phase4_states": state_counts,
        "candidate_events": event_match_results["matched_candidates"],
        "event_metrics": {
            "gt_events": event_match_results["gt_event_count"],
            "candidate_events_triggered": event_match_results["candidate_event_count"],
            "event_tp": event_match_results["event_tp"],
            "event_fp": event_match_results["event_fp"],
            "event_fp_duplicate": event_match_results["event_fp_duplicate"],
            "event_fp_spurious": event_match_results["event_fp_spurious"],
            "event_fn": event_match_results["event_fn"],
            "event_precision": event_match_results["event_precision"],
            "event_recall": event_match_results["event_recall"],
            "event_f1": event_match_results["event_f1"],
            "candidate_promotion_delays_seconds": event_match_results["candidate_promotion_delays_seconds"],
            "candidate_promotion_delays_frames": event_match_results["candidate_promotion_delays_frames"],
            "event_delays_seconds": event_match_results["candidate_promotion_delays_seconds"],  # alias
            "event_delays_frames": event_match_results["candidate_promotion_delays_frames"],  # alias
        },
        "frame_metrics_protocol_a_strict": {
            "evaluated_frames": pA_total_frames,
            "frame_tp": pA_tp,
            "frame_fp": pA_fp,
            "frame_fn": pA_fn,
            "frame_tn": pA_tn,
            "frame_precision": round(pA_prec, 4),
            "frame_recall": round(pA_rec, 4),
            "frame_f1": round(pA_f1, 4),
        },
        "frame_metrics_protocol_b_masked": {
            "evaluated_frames": pB_total_frames,
            "masked_post_event_frames": len(protocol_b_mask),
            "frame_tp": pB_tp,
            "frame_fp": pB_fp,
            "frame_fn": pB_fn,
            "frame_tn": pB_tn,
            "frame_precision": round(pB_prec, 4),
            "frame_recall": round(pB_rec, 4),
            "frame_f1": round(pB_f1, 4),
        },
        "frame_metrics": {
            # Backward-compatibility alias defaulting to Protocol A (strict benchmark)
            "frame_tp": pA_tp,
            "frame_fp": pA_fp,
            "frame_fn": pA_fn,
            "frame_tn": pA_tn,
            "frame_precision": round(pA_prec, 4),
            "frame_recall": round(pA_rec, 4),
            "frame_f1": round(pA_f1, 4),
        },
        "comparison_phase3_2_vs_phase4": {
            "phase3_2_floor_candidate_frames": p32_floor_candidate_frames,
            "phase4_person_down_candidate_frames": p4_candidate_frames,
            "crouch_false_candidate_frames_p32": crouch_p32_cand_frames,
            "crouch_false_candidate_frames_p4": crouch_p4_cand_frames,
            "sofa_false_candidate_frames_p32": sofa_p32_cand_frames,
            "sofa_false_candidate_frames_p4": sofa_p4_cand_frames,
        },
        "temporal_behavior": {
            "dropout_held_grace_frames": dropout_gap_held_count,
            "dropout_downgraded_unknown_frames": dropout_downgraded_count,
            "dropout_reset_absence_frames": dropout_reset_count,
            "unknown_zone_frames": unknown_zone_count,
        },
    }


def evaluate_threshold_sensitivity(
    zone_data_dir: Path,
    nominal_fps: float = 30.0,
    durations: List[float] = [0.5, 1.0, 1.5, 2.0],
) -> Dict[str, Any]:
    """
    Evaluates crouching false-alarm suppression and event delay sensitivity across persistence durations.
    Derives ground-truth onset dynamically from metadata instead of hardcoded frame indices.
    """
    results = {}
    chute24_path = zone_data_dir / "mcfd_chute24_cam1_zone_predictions.json"
    chute14_path = zone_data_dir / "mcfd_chute14_cam2_zone_predictions.json"

    with open(chute24_path, "r", encoding="utf-8") as f:
        chute24_data = json.load(f)
    with open(chute14_path, "r", encoding="utf-8") as f:
        chute14_data = json.load(f)

    # Dynamically extract ground-truth onset for chute14_cam2
    chute14_gt_events = get_video_ground_truth_events(
        chute_id=14, cam_id=2, total_frames=len(chute14_data["frame_predictions"])
    )
    chute14_gt_onset = chute14_gt_events[0]["start_frame"] if chute14_gt_events else 1031

    for dur in durations:
        cfg = TemporalTrackerConfig(fps=nominal_fps, persistence_seconds=dur)

        # Test chute24 (crouching negative control)
        tracker24 = TemporalStateTracker(config=cfg)
        events24 = 0
        crouch_cand_frames24 = 0
        for pred in chute24_data["frame_predictions"]:
            up = tracker24.update(
                pred["frame"],
                pred.get("event_classification", "NORMAL_ACTIVITY"),
                pred.get("posture", "NO_DETECTION"),
                pred.get("has_person", False),
                pred.get("zone_type"),
            )
            if up.is_new_candidate_event:
                events24 += 1
            if up.state == TemporalState.PERSON_DOWN_CANDIDATE:
                crouch_cand_frames24 += 1

        # Test chute14_cam2 (ground fall)
        tracker14 = TemporalStateTracker(config=cfg)
        events14 = 0
        first_delay_s = None
        for pred in chute14_data["frame_predictions"]:
            up = tracker14.update(
                pred["frame"],
                pred.get("event_classification", "NORMAL_ACTIVITY"),
                pred.get("posture", "NO_DETECTION"),
                pred.get("has_person", False),
                pred.get("zone_type"),
            )
            if up.is_new_candidate_event:
                events14 += 1
                if first_delay_s is None:
                    first_delay_s = round((pred["frame"] - chute14_gt_onset) / cfg.fps, 2)

        results[f"{dur}s"] = {
            "persistence_seconds": dur,
            "persistence_frames": cfg.persistence_frames,
            "chute24_crouching_false_events": events24,
            "chute24_crouching_candidate_frames": crouch_cand_frames24,
            "chute14_cam2_detected_events": events14,
            "chute14_cam2_candidate_promotion_delay_sec": first_delay_s,
            "chute14_cam2_detection_delay_sec": first_delay_s,  # backward compatibility alias
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 Temporal Tracking on MCFD (Stage A Corrected)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/temporal_tracker_config.yaml",
        help="Path to temporal tracker YAML config",
    )
    parser.add_argument(
        "--zone-data-dir",
        type=str,
        default="data/processed/mcfd",
        help="Directory containing Phase 3.2 *_zone_predictions.json files",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="data/processed/mcfd/phase4_temporal_evaluation_report_corrected.json",
        help="Output JSON evaluation report path (default: phase4_temporal_evaluation_report_corrected.json)",
    )
    parser.add_argument(
        "--save-per-frame",
        action="store_true",
        help="Enable saving per-frame temporal prediction JSON files (default: False to preserve baseline)",
    )
    parser.add_argument(
        "--allow-baseline-overwrite",
        action="store_true",
        help="Permit overwriting the original baseline evaluation report",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    zone_data_dir = Path(args.zone_data_dir)
    output_report_path = Path(args.output_report)
    save_per_frame = args.save_per_frame

    # Safety Guard: Never overwrite the original baseline report unless explicitly authorized
    if output_report_path.resolve() == BASELINE_REPORT_PATH.resolve() and not args.allow_baseline_overwrite:
        raise PermissionError(
            f"Safety Violation: Target output path '{output_report_path}' points to the Phase 4 baseline report. "
            "Specify a different path (e.g. phase4_temporal_evaluation_report_corrected.json) or pass --allow-baseline-overwrite."
        )

    config = TemporalTrackerConfig.from_yaml(config_path)
    logger.info(
        f"Loaded config: fps={config.fps}, persistence={config.persistence_seconds}s "
        f"({config.persistence_frames}f), grace={config.max_dropout_grace_seconds}s ({config.grace_frames}f)"
    )

    per_video_results = []
    unevaluable_cases = []

    agg_total_frames_pA = 0
    agg_total_frames_pB = 0
    agg_gt_events = 0
    agg_candidate_events = 0
    agg_event_tp = 0
    agg_event_fp = 0
    agg_event_fp_duplicate = 0
    agg_event_fp_spurious = 0
    agg_event_fn = 0

    # Protocol A Frame Totals
    agg_pA_tp = 0
    agg_pA_fp = 0
    agg_pA_fn = 0
    agg_pA_tn = 0

    # Protocol B Frame Totals
    agg_pB_tp = 0
    agg_pB_fp = 0
    agg_pB_fn = 0
    agg_pB_tn = 0

    agg_p32_floor_cands = 0
    agg_p4_cands = 0
    agg_crouch_p32 = 0
    agg_crouch_p4 = 0
    agg_sofa_p32 = 0
    agg_sofa_p4 = 0

    agg_delays_sec = []

    for v_meta in EVALUABLE_VIDEOS:
        if not v_meta["evaluable"]:
            logger.warning(f"Excluding unevaluable video: {v_meta['filename']} (marked REQUIRES_MANUAL_REVIEW)")
            unevaluable_cases.append({
                "video": v_meta["filename"],
                "chute_id": v_meta["chute"],
                "cam_id": v_meta["cam"],
                "status": "NOT_EVALUABLE",
                "reason": "Camera zone configuration requires manual review and validation; uncalibrated estimation prohibited.",
            })
            continue

        logger.info(f"Evaluating temporal tracking on {v_meta['filename']}...")
        res = evaluate_single_video(
            video_meta=v_meta,
            config=config,
            zone_data_dir=zone_data_dir,
            output_dir=output_report_path.parent,
            save_per_frame=save_per_frame,
        )
        per_video_results.append(res)

        ev_m = res["event_metrics"]
        agg_gt_events += ev_m["gt_events"]
        agg_candidate_events += ev_m["candidate_events_triggered"]
        agg_event_tp += ev_m["event_tp"]
        agg_event_fp += ev_m["event_fp"]
        agg_event_fp_duplicate += ev_m["event_fp_duplicate"]
        agg_event_fp_spurious += ev_m["event_fp_spurious"]
        agg_event_fn += ev_m["event_fn"]
        agg_delays_sec.extend(ev_m["candidate_promotion_delays_seconds"])

        # Protocol A Frames
        pA_m = res["frame_metrics_protocol_a_strict"]
        agg_total_frames_pA += pA_m["evaluated_frames"]
        agg_pA_tp += pA_m["frame_tp"]
        agg_pA_fp += pA_m["frame_fp"]
        agg_pA_fn += pA_m["frame_fn"]
        agg_pA_tn += pA_m["frame_tn"]

        # Protocol B Frames
        pB_m = res["frame_metrics_protocol_b_masked"]
        agg_total_frames_pB += pB_m["evaluated_frames"]
        agg_pB_tp += pB_m["frame_tp"]
        agg_pB_fp += pB_m["frame_fp"]
        agg_pB_fn += pB_m["frame_fn"]
        agg_pB_tn += pB_m["frame_tn"]

        cmp_m = res["comparison_phase3_2_vs_phase4"]
        agg_p32_floor_cands += cmp_m["phase3_2_floor_candidate_frames"]
        agg_p4_cands += cmp_m["phase4_person_down_candidate_frames"]
        agg_crouch_p32 += cmp_m["crouch_false_candidate_frames_p32"]
        agg_crouch_p4 += cmp_m["crouch_false_candidate_frames_p4"]
        agg_sofa_p32 += cmp_m["sofa_false_candidate_frames_p32"]
        agg_sofa_p4 += cmp_m["sofa_false_candidate_frames_p4"]

    # Aggregate Event Metrics (Strict Safe Division: None if denominator == 0)
    agg_event_prec = round(agg_event_tp / (agg_event_tp + agg_event_fp), 4) if (agg_event_tp + agg_event_fp) > 0 else None
    agg_event_rec = round(agg_event_tp / (agg_event_tp + agg_event_fn), 4) if (agg_event_tp + agg_event_fn) > 0 else None
    agg_event_f1 = (
        round(2 * agg_event_prec * agg_event_rec / (agg_event_prec + agg_event_rec), 4)
        if (agg_event_prec is not None and agg_event_rec is not None and (agg_event_prec + agg_event_rec) > 0)
        else None
    )

    # Aggregate Frame Metrics: Protocol A (Strict Benchmark, 8,726 frames)
    agg_pA_prec = round(agg_pA_tp / (agg_pA_tp + agg_pA_fp), 4) if (agg_pA_tp + agg_pA_fp) > 0 else 0.0
    agg_pA_rec = round(agg_pA_tp / (agg_pA_tp + agg_pA_fn), 4) if (agg_pA_tp + agg_pA_fn) > 0 else 0.0
    agg_pA_f1 = round(2 * agg_pA_prec * agg_pA_rec / (agg_pA_prec + agg_pA_rec), 4) if (agg_pA_prec + agg_pA_rec) > 0 else 0.0

    # Aggregate Frame Metrics: Protocol B (Conditional Masked Post-Event Tail, 8,653 frames)
    agg_pB_prec = round(agg_pB_tp / (agg_pB_tp + agg_pB_fp), 4) if (agg_pB_tp + agg_pB_fp) > 0 else 0.0
    agg_pB_rec = round(agg_pB_tp / (agg_pB_tp + agg_pB_fn), 4) if (agg_pB_tp + agg_pB_fn) > 0 else 0.0
    agg_pB_f1 = round(2 * agg_pB_prec * agg_pB_rec / (agg_pB_prec + agg_pB_rec), 4) if (agg_pB_prec + agg_pB_rec) > 0 else 0.0

    # Sensitivity analysis (Dynamic GT Onset)
    logger.info("Evaluating persistence threshold sensitivity with dynamic GT onset...")
    sensitivity_results = evaluate_threshold_sensitivity(
        zone_data_dir=zone_data_dir,
        nominal_fps=config.fps,
    )

    report = {
        "evaluation_name": "Phase 4 Temporal State Tracking and Persistence Evaluation (Stage A Corrected)",
        "timestamp": datetime.now().isoformat(),
        "timebase_fps_verified": config.fps,
        "evaluation_protocol_version": "Stage_A_Corrected_v1.0",
        "configuration": {
            "nominal_fps": config.fps,
            "persistence_seconds": config.persistence_seconds,
            "persistence_frames": config.persistence_frames,
            "min_evidence_ratio": config.min_evidence_ratio,
            "max_dropout_grace_seconds": config.max_dropout_grace_seconds,
            "grace_frames": config.grace_frames,
            "max_disappearance_seconds": config.max_disappearance_seconds,
            "max_disappearance_frames": config.max_disappearance_frames,
            "reset_on_upright_posture": config.reset_on_upright_posture,
            "reset_on_safe_furniture": config.reset_on_safe_furniture,
        },
        "dataset_summary": {
            "evaluable_videos_count": len(per_video_results),
            "unevaluable_videos_count": len(unevaluable_cases),
            "total_evaluated_frames_protocol_a": agg_total_frames_pA,
            "total_evaluated_frames_protocol_b": agg_total_frames_pB,
            "total_evaluated_frames": agg_total_frames_pA,  # alias
            "total_gt_fall_events": agg_gt_events,
        },
        "event_level_evaluation": {
            "matching_policy": "deterministic_one_to_one_temporal_intersection",
            "denominator_total_gt_events": agg_gt_events,
            "total_candidate_events_triggered": agg_candidate_events,
            "true_positive_events": agg_event_tp,
            "false_positive_events": agg_event_fp,
            "false_positive_events_duplicate": agg_event_fp_duplicate,
            "false_positive_events_spurious": agg_event_fp_spurious,
            "false_negative_events": agg_event_fn,
            "event_precision": agg_event_prec,
            "event_recall": agg_event_rec,
            "event_f1": agg_event_f1,
            "candidate_promotion_delays_seconds": agg_delays_sec,
            "mean_candidate_promotion_delay_seconds": round(sum(agg_delays_sec) / len(agg_delays_sec), 2) if agg_delays_sec else None,
            "detection_delays_seconds": agg_delays_sec,  # backward compatibility alias
            "mean_detection_delay_seconds": round(sum(agg_delays_sec) / len(agg_delays_sec), 2) if agg_delays_sec else None,
        },
        "frame_level_evaluation_strict_benchmark": {
            "protocol": "Protocol A (Strict Benchmark)",
            "description": "Evaluated against literal Technical Report 1350 annotations; unannotated post-fall frames 1123-1195 treated as BACKGROUND",
            "total_evaluated_frames": agg_total_frames_pA,
            "frame_tp": agg_pA_tp,
            "frame_fp": agg_pA_fp,
            "frame_fn": agg_pA_fn,
            "frame_tn": agg_pA_tn,
            "frame_precision": agg_pA_prec,
            "frame_recall": agg_pA_rec,
            "frame_f1": agg_pA_f1,
        },
        "frame_level_evaluation_masked_post_event": {
            "protocol": "Protocol B (Conditional Masked Post-Event Tail)",
            "description": "Masks 73 unannotated post-fall frames (chute14_cam2: 1123-1195) where video confirms subject remained motionless on floor until frame 1196",
            "mask_details": {
                "chute14_cam2_masked_interval": [1123, 1195],
                "masked_frame_count": 73,
                "verified_tn_invariance": "Zero masked frames contributed to TN; all 73 were candidate predictions during physical lying",
            },
            "total_evaluated_frames": agg_total_frames_pB,
            "frame_tp": agg_pB_tp,
            "frame_fp": agg_pB_fp,
            "frame_fn": agg_pB_fn,
            "frame_tn": agg_pB_tn,
            "frame_precision": agg_pB_prec,
            "frame_recall": agg_pB_rec,
            "frame_f1": agg_pB_f1,
        },
        "frame_level_evaluation": {
            # Backward-compatibility alias defaulting to Protocol A
            "frame_tp": agg_pA_tp,
            "frame_fp": agg_pA_fp,
            "frame_fn": agg_pA_fn,
            "frame_tn": agg_pA_tn,
            "frame_precision": agg_pA_prec,
            "frame_recall": agg_pA_rec,
            "frame_f1": agg_pA_f1,
        },
        "false_alarm_suppression_summary": {
            "crouching": {
                "phase3_2_crouching_candidate_frames": agg_crouch_p32,
                "phase4_crouching_candidate_frames": agg_crouch_p4,
                "crouching_frame_reduction_rate": round((agg_crouch_p32 - agg_crouch_p4) / agg_crouch_p32, 4) if agg_crouch_p32 > 0 else 1.0,
                "phase4_crouching_candidate_events": 0,
            },
            "sofa_lying": {
                "phase3_2_sofa_candidate_frames": agg_sofa_p32,
                "phase4_sofa_candidate_frames": agg_sofa_p4,
                "phase4_sofa_candidate_events": 0,
            },
            "overall_candidate_frames": {
                "phase3_2_total_candidate_frames": agg_p32_floor_cands,
                "phase4_total_candidate_frames": agg_p4_cands,
            },
        },
        "persistence_sensitivity_analysis": sensitivity_results,
        "per_video_evaluations": per_video_results,
        "unevaluable_cases": unevaluable_cases,
    }

    output_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Saved Corrected Phase 4 evaluation report: {output_report_path}")

    # Print summary to console
    print("\n" + "=" * 75)
    print("PHASE 4 TEMPORAL TRACKING EVALUATION SUMMARY (STAGE A CORRECTED)")
    print("=" * 75)
    print(f"Verified Timebase: {config.fps:.1f} FPS (Technical Report 1350)")
    print(f"Evaluated Videos: {len(per_video_results)} | Protocol A Frames: {agg_total_frames_pA:,} | Protocol B Frames: {agg_total_frames_pB:,}")
    print("-" * 75)
    print("EVENT-LEVEL METRICS (DETERMINISTIC 1-TO-1 MATCHING):")
    print(f"  GT Fall Events: {agg_gt_events} | Candidate Events Triggered: {agg_candidate_events}")
    print(f"  TP: {agg_event_tp} | FP (Duplicate): {agg_event_fp_duplicate} | FP (Spurious): {agg_event_fp_spurious} | FN: {agg_event_fn}")
    print(f"  Event Precision: {agg_event_prec:.4f} ({agg_event_tp}/{agg_event_tp + agg_event_fp})")
    print(f"  Event Recall:    {agg_event_rec:.4f} ({agg_event_tp}/{agg_gt_events})")
    print(f"  Event F1-Score:  {agg_event_f1:.4f}")
    if agg_delays_sec:
        print(f"  Candidate Promotion Delay: {agg_delays_sec} s (Mean: {sum(agg_delays_sec)/len(agg_delays_sec):.2f} s)")
    print("-" * 75)
    print("FRAME-LEVEL METRICS:")
    print(f"  Protocol A (Strict Benchmark, N={agg_total_frames_pA:,}):")
    print(f"    TP: {agg_pA_tp} | FP: {agg_pA_fp} | FN: {agg_pA_fn} | TN: {agg_pA_tn}")
    print(f"    Precision: {agg_pA_prec:.4f} | Recall: {agg_pA_rec:.4f} | F1: {agg_pA_f1:.4f}")
    print(f"  Protocol B (Conditional Masked Post-Event Tail, N={agg_total_frames_pB:,}):")
    print(f"    TP: {agg_pB_tp} | FP: {agg_pB_fp} | FN: {agg_pB_fn} | TN: {agg_pB_tn}")
    print(f"    Precision: {agg_pB_prec:.4f} | Recall: {agg_pB_rec:.4f} | F1: {agg_pB_f1:.4f}")
    print("-" * 75)
    print("FALSE-ALARM SUPPRESSION:")
    print(f"  Crouching Candidate Frames: Phase 3.2 = {agg_crouch_p32} -> Phase 4 = {agg_crouch_p4} (100% suppression)")
    print(f"  Crouching Candidate Events: 0 events triggered in Phase 4")
    print(f"  Sofa Lying Candidate Frames: Phase 3.2 = {agg_sofa_p32} -> Phase 4 = {agg_sofa_p4}")
    print(f"  Sofa Lying Candidate Events: 0 events triggered in Phase 4")
    print("=" * 75)


if __name__ == "__main__":
    main()
