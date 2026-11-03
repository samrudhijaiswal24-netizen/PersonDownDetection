#!/usr/bin/env python3
"""
CLI Video Processing Runner for Person Down Detection System
Executes YOLO11-Pose inference, extracts keypoints, renders skeleton overlays,
and saves annotated videos and summary metrics.
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pose_detector import YOLOPoseDetector
from src.pipeline.video_annotator import VideoAnnotator
from src.pipeline.video_processor import VideoProcessor
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO11-Pose video pipeline on CCTV/fall detection video."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video file (e.g. data/samples/floor_fall.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to destination annotated video file (defaults to data/processed/annotated_<name>.mp4)",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to output JSON detection summary (defaults to data/processed/<name>_summary.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/pretrained/yolo11n-pose.pt",
        help="Path to YOLO11-Pose weights file (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Person detection confidence threshold [0.0 - 1.0]",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference hardware device ('cuda', 'cuda:0', 'cpu', or auto-detect)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap the number of frames to process for quick testing",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.video)

    if not input_path.exists():
        print(f"\n[ERROR] Input video does not exist: {input_path.resolve()}", file=sys.stderr)
        print("Please provide a valid video path. Check data/samples/ for available clips.\n", file=sys.stderr)
        sys.exit(1)

    stem = input_path.stem
    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "processed" / f"annotated_{stem}.mp4"
    )
    summary_path = (
        Path(args.summary)
        if args.summary
        else PROJECT_ROOT / "data" / "processed" / f"{stem}_summary.json"
    )

    print("\n" + "=" * 65)
    print(" YOLO11-POSE VIDEO PROCESSING PIPELINE")
    print("=" * 65)
    print(f" Input Video  : {input_path.resolve()}")
    print(f" Output Video : {output_path.resolve()}")
    print(f" Summary File : {summary_path.resolve()}")
    print(f" Model Path   : {args.model}")
    print(f" Confidence   : {args.conf}")
    print(f" Max Frames   : {args.max_frames or 'All'}")

    try:
        detector = YOLOPoseDetector(
            model_path=args.model,
            device=args.device,
            conf_threshold=args.conf,
        )
        print(f" Hardware     : {detector.get_device_info()}")
        print("-" * 65)

        processor = VideoProcessor(detector=detector)

        # Progress bar setup
        pbar = None
        last_frame = 0

        def update_progress(curr, total, fps):
            nonlocal pbar, last_frame
            if pbar is None:
                pbar = tqdm(total=total, desc="Processing Video", unit="frame")
            pbar.update(curr - last_frame)
            pbar.set_postfix({"FPS": f"{fps:.1f}"})
            last_frame = curr

        summary = processor.process_video(
            input_video_path=input_path,
            output_video_path=output_path,
            summary_path=summary_path,
            max_frames=args.max_frames,
            progress_callback=update_progress,
        )

        if pbar is not None:
            pbar.close()

        metrics = summary["processing_metrics"]
        print("\n" + "=" * 65)
        print(" PROCESSING COMPLETED SUCCESSFULLY")
        print("=" * 65)
        print(f" Processed Frames      : {metrics['processed_frames']}")
        print(f" Frames with Persons   : {metrics['frames_with_detections']} ({metrics['detection_frame_ratio']*100:.1f}%)")
        print(f" Total Detections      : {metrics['total_person_detections']}")
        print(f" Avg Persons / Frame   : {metrics['avg_persons_per_frame']}")
        print(f" Avg Confidence        : {metrics['confidence_stats']['mean']:.3f}")
        p_counts = metrics.get('posture_counts', {})
        p_str = ", ".join(f"{k}: {v}" for k, v in p_counts.items())
        print(f" Posture Breakdown     : {p_str}")
        print(f" Processing Time       : {metrics['elapsed_time_seconds']:.2f}s")
        print(f" Average Processing FPS: {metrics['average_processing_fps']:.1f} FPS")

        print(f" Output Video Size     : {summary.get('output_video_size_bytes', 0) / 1024:.1f} KB")
        print(f" Output Video File     : {output_path.resolve()}")
        print(f" Metrics Report File   : {summary_path.resolve()}")
        print("=" * 65 + "\n")

    except Exception as e:
        print(f"\n[FATAL ERROR] Pipeline execution failed: {str(e)}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
