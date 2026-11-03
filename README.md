# AI-Based Person Down / Collapse Detection System

An enterprise-grade computer vision and temporal tracking system engineered for real-time person down, fall, and collapse detection in hospital and room CCTV environments.

---

## Key Capabilities & Pipeline Overview

- **Pose & Person Detection**: Detects persons and extracts 17 COCO keypoints using YOLO11-Pose (`yolo11n-pose.pt`).
- **Posture Classification**: Classifies postures into `STANDING`, `SITTING`, `LYING`, or `UNKNOWN` using geometric feature analysis, torso projection angles, and aspect ratios, with Phase 6 perspective foreshortening & crouching safeguards.
- **Spatial Zone Disambiguation**: Distinguishes between lying safely on a bed/couch and lying dangerously on the floor via polygon-defined camera zones (`configs/zones_config.yaml`).
- **Temporal Alert Engine**: Enforces multi-frame persistence (1.5s / 45 frames @ 30 FPS, 65% minimum evidence ratio) with bounded dropout grace memory (0.5s), disappearance safeguards, and instant upright recovery (`configs/temporal_tracker_config.yaml`).
- **Interactive Web Application (Phase 7)**: Glassmorphic dashboard built with FastAPI, HTML5, Vanilla CSS, and JavaScript. Features real-time telemetry polling, video upload/sample selection, live HUD overlays, event history logging with video seek synchronization, and report downloads.

---

## Project Structure

```text
PersonDownDetection_Claude/
├── configs/                  # Zone definitions and temporal tracking configs
│   ├── default_config.yaml
│   ├── temporal_tracker_config.yaml
│   └── zones_config.yaml
├── data/
│   ├── processed/            # Evaluation benchmarks, phase reports, annotated videos
│   │   ├── mcfd/             # MCFD evaluation outputs and baseline reports
│   │   └── phase7/           # Phase 7 application and demonstration artifacts
│   ├── samples/              # Validation clips and MCFD benchmark videos
│   └── uploads/              # User-uploaded video files for application processing
├── docs/                     # Detailed phase reports and system documentation
├── models/
│   └── pretrained/           # Pretrained model weights (yolo11n-pose.pt)
├── scripts/
│   ├── run_app.py            # Phase 7 Web Application launcher
│   ├── demonstrate_phase7.py # Phase 7 End-to-end demonstration script
│   ├── evaluate_phase6.py    # Phase 6 detection recall evaluation runner
│   ├── evaluate_temporal_tracking.py
│   └── run_pipeline.py       # Standalone CLI processing runner
├── src/
│   ├── api/                  # FastAPI backend, static assets, and templates
│   │   ├── app.py            # REST API endpoints & background worker
│   │   ├── models.py         # Pydantic schemas
│   │   ├── static/           # CSS styles & frontend controller JS
│   │   └── templates/        # HTML5 dashboard template
│   ├── core/                 # Pose detector, posture classifier, zone manager, temporal tracker
│   └── pipeline/             # PersonDownPipeline, VideoProcessor, VideoAnnotator, ZoneVisualizer
└── tests/                    # Unit and integration test suites (157 tests)
    └── test_phase7_app.py    # Phase 7 application integration tests
```

---

## Quickstart: Running the Application (Phase 7)

### 1. Launch the Web Application

To launch the web dashboard server:

```powershell
# Activate the existing virtual environment
.\.venv\Scripts\Activate.ps1

# Run the application launcher
python scripts\run_app.py --host 127.0.0.1 --port 8000
```

Once started, open your web browser to:
- **Web UI Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Interactive OpenAPI Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 2. Using the Dashboard

1. **Select or Upload Video**:
   - **Sample Library**: Choose any benchmark clip (e.g., `mcfd_chute14_cam1.avi` [Fall], `mcfd_chute14_cam2.avi` [Fall], or `mcfd_chute24_cam1.avi` [ADL Negative Control]).
   - **Upload Video**: Drag and drop any `.mp4`, `.avi`, `.mov`, `.mkv` file.
2. **Configure Camera Zone Profile**:
   - `Auto-Detect from Video / Filename` (default).
   - `MCFD Camera 1` (Corner view) or `MCFD Camera 2` (Axial view).
3. **Execute Pipeline**:
   - Click **"Start Detection Pipeline"**.
   - Watch live progress bar, processing FPS, and telemetry status badges (`NORMAL` 🛡️, `POSSIBLE_PERSON_DOWN` ⚠️, or `PERSON_DOWN_CANDIDATE` 🚨).
4. **Review Results & Video Playback**:
   - View the annotated video with bounding boxes, 17-keypoint skeleton overlays, spatial zone polygons, and top HUD telemetry banner.
   - Click **"Seek in Video"** on any detected event row in the Event History table to jump directly to that timestamp.
   - Download the annotated video (`.mp4`) or full structured metrics report (`.json`).
5. **Continuous Operation**:
   - Click **"Process Another Video"** to analyze subsequent videos without restarting the server.

---

## Running the End-to-End Demonstration

To run the automated validation demonstration on both a verified fall video and the ADL negative control:

```powershell
python scripts\demonstrate_phase7.py
```

This script:
1. Validates REST API endpoints via `TestClient`.
2. Processes `mcfd_chute14_cam1.avi` (verifies 1 detected fall candidate event at frame 1133).
3. Processes `mcfd_chute24_cam1.avi` (verifies 0 candidate events / zero false alarms on normal daily activities).
4. Verifies baseline SHA-256 hashes for all previous phase reports and predictions.
5. Saves demonstration outputs to `data/processed/phase7/`.

---

## Running the Test Suite

The test suite covers pose extraction, posture geometry, spatial zones, temporal tracking, hysteresis, and FastAPI integration:

```powershell
# Run the complete test suite (157 tests)
python -m pytest tests/ -v

# Run Phase 7 application integration tests only (20 tests)
python -m pytest tests/test_phase7_app.py -v
```

---

## System Disclaimer

> **RESEARCH PROTOTYPE**: This software is an experimental research prototype for computer-vision-based fall and person-down detection. It is not a certified medical device, diagnostic tool, or emergency response system.
