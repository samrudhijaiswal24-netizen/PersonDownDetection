"""
Stage B: Tracker Recovery Hysteresis Empirical Evaluation Sweep.

Evaluates recovery confirmation hysteresis thresholds N in [1, 2, 3, 5, 10] frames
across all 5 existing MCFD zone prediction files without retraining models or
rerunning pose detection.

Outputs are saved separately to:
data/processed/mcfd/phase4_hysteresis_sweep_results.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.temporal_tracker import (
    TemporalState,
    TemporalTrackerConfig,
    TemporalStateTracker,
)
from src.core.ground_truth import (
    get_video_ground_truth_events,
)
from scripts.evaluate_temporal_tracking import (
    EVALUABLE_VIDEOS,
    match_candidate_events_to_ground_truth,
)

OUTPUT_FILE = Path("data/processed/mcfd/phase5_hysteresis_sweep_results.json")
ZONE_DATA_DIR = Path("data/processed/mcfd")


def calc_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, Any]:
    """
    Calculates precision, recall, and F1 score from confusion matrix counts.
    Undefined metrics (0/0) return None rather than arbitrary 0.0 or 1.0 (Stage A convention).
    """
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else None
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec is not None and rec is not None and (prec + rec) > 0) else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(prec, 4) if prec is not None else None,
        "recall": round(rec, 4) if rec is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
    }



def evaluate_single_video_hysteresis(
    video_meta: Dict[str, Any],
    n_confirm: int,
    base_config: TemporalTrackerConfig,
) -> Dict[str, Any]:
    """Runs a single video through the tracker with a specific hysteresis threshold N."""
    zone_path = ZONE_DATA_DIR / video_meta["zone_json"]
    with open(zone_path, "r", encoding="utf-8") as f:
        zone_data = json.load(f)

    frame_preds = zone_data["frame_predictions"]
    total_frames = len(frame_preds)
    chute_id = video_meta["chute"]
    cam_id = video_meta["cam"]
    fps = base_config.fps

    gt_fall_events = get_video_ground_truth_events(
        chute_id=chute_id,
        cam_id=cam_id,
        total_frames=total_frames,
    )

    protocol_b_mask = set(range(1123, 1196)) if (chute_id == 14 and cam_id == 2) else set()

    cfg = TemporalTrackerConfig(
        fps=fps,
        persistence_seconds=base_config.persistence_seconds,
        min_evidence_ratio=base_config.min_evidence_ratio,
        max_dropout_grace_seconds=base_config.max_dropout_grace_seconds,
        max_disappearance_seconds=base_config.max_disappearance_seconds,
        reset_on_upright_posture=base_config.reset_on_upright_posture,
        reset_on_safe_furniture=base_config.reset_on_safe_furniture,
        upright_recovery_confirmation_frames=n_confirm,
    )
    tracker = TemporalStateTracker(config=cfg)

    state_counts = {st.value: 0 for st in TemporalState}
    candidate_events = []
    current_event = None

    pA_tp = pA_fp = pA_fn = pA_tn = 0
    pB_tp = pB_fp = pB_fn = pB_tn = 0

    # General recovery latency tracking across candidate events
    first_upright_frame: Optional[int] = None
    transition_to_normal_frame: Optional[int] = None
    candidate_observed: bool = False
    recovery_measured: bool = False

    for pred in frame_preds:
        f_idx = pred["frame"]
        gt_label = pred.get("ground_truth", "BACKGROUND")
        event_cls = pred.get("event_classification", "NORMAL_ACTIVITY")
        posture = pred.get("posture", "NO_DETECTION")
        has_person = pred.get("has_person", False)
        zone_type = pred.get("zone_type", None)

        up = tracker.update(
            frame_idx=f_idx,
            event_classification=event_cls,
            posture=posture,
            has_person=has_person,
            zone_type=zone_type,
        )

        st_val = up.state.value
        state_counts[st_val] += 1
        is_p4_cand = (up.state == TemporalState.PERSON_DOWN_CANDIDATE)

        # Track candidate occurrence
        if is_p4_cand:
            candidate_observed = True

        # Generalized recovery latency tracking: from first upright/safe frame to NORMAL transition
        if candidate_observed and not recovery_measured:
            is_recovery_signal = (
                (cfg.reset_on_upright_posture and posture in ("STANDING", "SITTING")) or
                (cfg.reset_on_safe_furniture and event_cls == "BED_SOFA_LYING")
            )
            if is_recovery_signal:
                if first_upright_frame is None:
                    first_upright_frame = f_idx
                if up.state == TemporalState.NORMAL:
                    transition_to_normal_frame = f_idx
                    recovery_measured = True
            elif event_cls == "FLOOR_LYING_CANDIDATE":
                # Lying resumed: transient glitch was absorbed by hysteresis; reset pending marker
                first_upright_frame = None

        # Protocol A Confusion Matrix
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

        # Protocol B Confusion Matrix
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

    # Match candidate events to GT events
    event_match = match_candidate_events_to_ground_truth(gt_fall_events, candidate_events, fps=fps)

    # Compute frame metrics
    frame_metrics_A = calc_metrics(pA_tp, pA_fp, pA_fn, pA_tn)
    frame_metrics_B = calc_metrics(pB_tp, pB_fp, pB_fn, pB_tn)

    recovery_info: Optional[Dict[str, Any]] = None
    if candidate_observed and first_upright_frame is not None and transition_to_normal_frame is not None:
        latency_frames = transition_to_normal_frame - first_upright_frame + 1
        latency_sec = round(latency_frames / fps, 4)
        recovery_info = {
            "first_upright_frame": first_upright_frame,
            "transition_to_normal_frame": transition_to_normal_frame,
            "latency_frames": latency_frames,
            "latency_seconds": latency_sec,
            "within_350ms": (latency_sec <= 0.35),
        }
        if chute_id == 14 and cam_id == 2 and candidate_events:
            # Baseline (N=1) recovery boundary for chute14_cam2 is frame 1195
            tail_extension = max(0, candidate_events[0]["end_frame"] - 1195)
            recovery_info["tail_extension_frames"] = tail_extension
            recovery_info["tail_extension_definition"] = (
                "Number of candidate frames extending beyond the baseline (N=1) recovery boundary (frame 1195) into the post-event recovery sequence."
            )
    elif candidate_observed:
        recovery_info = {
            "status": "ongoing_or_unresolved",
            "reason": "Candidate observed but no upright recovery sequence completed before video end",
        }
    else:
        recovery_info = {
            "status": "not_applicable",
            "reason": "No candidate event triggered to evaluate recovery",
        }

    return {
        "video": video_meta["filename"],
        "chute": chute_id,
        "cam": cam_id,
        "is_fall_scenario": video_meta["is_fall_scenario"],
        "total_frames": total_frames,
        "candidate_events_count": len(candidate_events),
        "candidate_events": candidate_events,
        "state_counts": state_counts,
        "event_matching": event_match,
        "frame_metrics_protocol_a": frame_metrics_A,
        "frame_metrics_protocol_b": frame_metrics_B,
        "recovery_info": recovery_info,
    }


def run_hysteresis_sweep() -> Dict[str, Any]:
    """Runs the sweep for N in [1, 2, 3, 5, 10] across all evaluable videos."""
    evaluable = [v for v in EVALUABLE_VIDEOS if v.get("evaluable") and v.get("zone_json")]
    base_config = TemporalTrackerConfig.from_yaml(Path("configs/temporal_tracker_config.yaml"))
    n_values = [1, 2, 3, 5, 10]

    sweep_results: Dict[str, Any] = {
        "description": "Evaluation of recovery hysteresis confirmation frames N in [1, 2, 3, 5, 10]",
        "base_config": {
            "fps": base_config.fps,
            "persistence_frames": base_config.persistence_frames,
            "persistence_seconds": base_config.persistence_seconds,
            "min_evidence_ratio": base_config.min_evidence_ratio,
            "grace_frames": base_config.grace_frames,
            "max_disappearance_frames": base_config.max_disappearance_frames,
        },
        "threshold_evaluations": {},
    }

    for n in n_values:
        n_key = f"N_{n}"
        video_results = []

        agg_event_tp = agg_event_fp = agg_event_fn = 0
        agg_pA_tp = agg_pA_fp = agg_pA_fn = agg_pA_tn = 0
        agg_pB_tp = agg_pB_fp = agg_pB_fn = agg_pB_tn = 0

        for v in evaluable:
            res = evaluate_single_video_hysteresis(v, n_confirm=n, base_config=base_config)
            video_results.append(res)

            em = res["event_matching"]
            agg_event_tp += em["event_tp"]
            agg_event_fp += em["event_fp"]
            agg_event_fn += em["event_fn"]

            fA = res["frame_metrics_protocol_a"]
            agg_pA_tp += fA["tp"]
            agg_pA_fp += fA["fp"]
            agg_pA_fn += fA["fn"]
            agg_pA_tn += fA["tn"]

            fB = res["frame_metrics_protocol_b"]
            agg_pB_tp += fB["tp"]
            agg_pB_fp += fB["fp"]
            agg_pB_fn += fB["fn"]
            agg_pB_tn += fB["tn"]

        agg_event_metrics = calc_metrics(agg_event_tp, agg_event_fp, agg_event_fn, 0)
        agg_pA_metrics = calc_metrics(agg_pA_tp, agg_pA_fp, agg_pA_fn, agg_pA_tn)
        agg_pB_metrics = calc_metrics(agg_pB_tp, agg_pB_fp, agg_pB_fn, agg_pB_tn)

        chute14_cam2_res = next(r for r in video_results if r["chute"] == 14 and r["cam"] == 2)
        chute24_cam1_res = next(r for r in video_results if r["chute"] == 24 and r["cam"] == 1)

        c14_rec = chute14_cam2_res["recovery_info"]

        sweep_results["threshold_evaluations"][n_key] = {
            "n_confirmation_frames": n,
            "recovery_latency_sec_theory": round(n / base_config.fps, 4),
            "negative_control_chute24_cam1": {
                "candidate_events_count": chute24_cam1_res["candidate_events_count"],
                "candidate_frames": chute24_cam1_res["state_counts"]["PERSON_DOWN_CANDIDATE"],
                "pass_requirement_5": (chute24_cam1_res["candidate_events_count"] == 0),
            },
            "benchmark_chute14_cam2_recovery": {
                "benchmark_name": "chute14_cam2_post_fall_recovery",
                "first_upright_frame": c14_rec.get("first_upright_frame"),
                "transition_to_normal_frame": c14_rec.get("transition_to_normal_frame"),
                "latency_frames": c14_rec.get("latency_frames"),
                "latency_seconds": c14_rec.get("latency_seconds"),
                "within_350ms": c14_rec.get("within_350ms"),
                "tail_extension_frames": c14_rec.get("tail_extension_frames"),
                "tail_extension_definition": c14_rec.get("tail_extension_definition"),
            },
            "chute14_cam2_recovery": c14_rec,  # Backward compatibility alias
            "chute14_cam2_candidate_event": chute14_cam2_res["candidate_events"][0] if chute14_cam2_res["candidate_events"] else None,
            "aggregate_event_metrics": agg_event_metrics,
            "aggregate_protocol_a_frame_metrics": agg_pA_metrics,
            "aggregate_protocol_b_frame_metrics": agg_pB_metrics,
            "per_video_results": video_results,
        }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=2)

    print(f"Hysteresis sweep complete. Results written to: {OUTPUT_FILE}")
    return sweep_results


if __name__ == "__main__":
    run_hysteresis_sweep()
