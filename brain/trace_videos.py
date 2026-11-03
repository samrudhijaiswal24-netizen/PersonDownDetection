"""
Read-Only Diagnostic Trace Script for Person Down Detection
Traces detection pipeline stages on:
1. floor_fall.mp4
2. mcfd_chute16_cam1.avi
3. mcfd_chute14_cam1.avi (for comparison)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from src.pipeline.person_down_pipeline import PersonDownPipeline
from src.core.temporal_tracker import TemporalStateTracker, TemporalState

def trace_video(video_rel_path, camera_id="auto", sample_interval=10, max_frames=None):
    pipe = PersonDownPipeline(
        model_path="models/pretrained/yolo11n-pose.pt",
        zones_config_path="configs/zones_config.yaml",
        tracker_config_path="configs/temporal_tracker_config.yaml",
    )
    video_path = Path(video_rel_path)
    cam_cfg_id = pipe.resolve_camera_config_id(video_path, camera_id)
    print(f"\n========================================================")
    print(f"TRACING: {video_path.name} | Camera Config: {cam_cfg_id}")
    print(f"========================================================")

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolution: {w}x{h}, Container FPS: {fps}, Total Frames: {total_frames}")

    tracker = TemporalStateTracker(pipe.tracker_config)
    smoother = pipe.smoother

    candidate_promotions = []
    frame_records = []
    
    frame_idx = 0
    limit = min(total_frames, max_frames) if max_frames else total_frames

    while frame_idx < limit:
        ret, frame = cap.read()
        if not ret:
            break

        persons = pipe.detector.detect(frame)
        primary_person = None
        has_person = len(persons) > 0

        if has_person:
            primary_person = max(persons, key=lambda p: p.confidence)

        if primary_person is not None:
            raw_cls = pipe.posture_classifier.classify(primary_person.keypoints, primary_person.box)
            posture_label = raw_cls.label
            posture_conf = raw_cls.confidence

            zone_res = pipe.zone_manager.classify_zone(
                camera_id=cam_cfg_id,
                posture=posture_label,
                keypoints=primary_person.keypoints,
                box=primary_person.box,
            )
            event_cls = zone_res.event_classification.value
            zone_type = zone_res.zone_type.value
            zone_name = zone_res.zone_name
            ref_x, ref_y = zone_res.reference_point.x, zone_res.reference_point.y
            box_coords = [round(b, 1) for b in primary_person.box]
            p_conf = round(primary_person.confidence, 3)
        else:
            posture_label = "NO_DETECTION"
            posture_conf = 0.0
            event_cls = "NORMAL_ACTIVITY"
            zone_type = "OUTSIDE_ZONE"
            zone_name = None
            ref_x, ref_y = 0.0, 0.0
            box_coords = []
            p_conf = 0.0

        t_upd = tracker.update(
            frame_idx=frame_idx,
            event_classification=event_cls,
            posture=posture_label,
            has_person=has_person,
            zone_type=zone_type,
        )

        if t_upd.is_new_candidate_event:
            candidate_promotions.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / 30.0, 2),
                "posture": posture_label,
                "zone": zone_name or zone_type,
            })

        # Record interesting frames (e.g. state changes, periodic samples, or LYING frames)
        is_sample = (frame_idx % sample_interval == 0) or (posture_label == "LYING") or t_upd.is_new_candidate_event
        if is_sample:
            frame_records.append({
                "frame": frame_idx,
                "time_sec": round(frame_idx / 30.0, 2),
                "has_p": has_person,
                "p_conf": p_conf,
                "box": box_coords,
                "ref_pt": (round(ref_x, 1), round(ref_y, 1)),
                "posture": posture_label,
                "zone_name": zone_name or "OUTSIDE",
                "zone_type": zone_type,
                "event_cls": event_cls,
                "state": t_upd.state.value,
                "down_sec": t_upd.elapsed_down_seconds,
                "floor_frames": t_upd.accumulated_floor_frames,
                "is_candidate": t_upd.state == TemporalState.PERSON_DOWN_CANDIDATE,
            })

        frame_idx += 1

    cap.release()

    print(f"Total Candidate Alerts Promoted: {len(candidate_promotions)}")
    for alert in candidate_promotions:
        print(f"  >>> ALERT at frame {alert['frame']} ({alert['time_sec']}s), posture={alert['posture']}, zone={alert['zone']}")

    # Print summary statistics
    lying_count = sum(1 for r in frame_records if r['posture'] == 'LYING')
    floor_zone_count = sum(1 for r in frame_records if 'FLOOR' in r['zone_type'])
    print(f"Summary in sampled records: LYING frames={lying_count}, FLOOR_ZONE frames={floor_zone_count}")
    
    return frame_records, candidate_promotions

if __name__ == "__main__":
    r_floor, a_floor = trace_video("data/samples/floor_fall.mp4", sample_interval=15)
    r_c16, a_c16 = trace_video("data/samples/mcfd/mcfd_chute16_cam1.avi", sample_interval=30)
    r_c14, a_c14 = trace_video("data/samples/mcfd/mcfd_chute14_cam1.avi", sample_interval=30)
