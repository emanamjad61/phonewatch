# PhoneWatch

PhoneWatch is a computer-vision project for real-time phone-use detection. It combines YOLO object detection, OpenCV video processing, contextual phone/person analysis, alert overlays, meme/serious alert modes, training utilities, benchmarking, and a lightweight web dashboard styled with 7.css.

## Features

- Real-time webcam detection for people and phones
- Context-aware phone-use classification
- Meme and serious alert overlay modes
- Presentation demo mode with countdown, overlays, split view, and recording
- Image and video file inference
- Dataset preparation and training pipeline
- Lightweight 7.css web dashboard for monitoring, analytics, and live detection
- Benchmarking and environment diagnostics

## Project Layout

```text
phonewatch/
+-- src/                    Core dataset, detection, context, alerts, and training code
+-- dashboard/              Fast web dashboard (FastAPI + 7.css + vanilla JS)
+-- tests/                  Pytest suite
+-- data/                   Local datasets and quick-test media
+-- models/checkpoints/     Trained and fallback model weights
+-- memes/                  Meme-mode alert images
+-- logs/                   Runtime output directory, kept empty in source control
+-- config.yaml             Project configuration
+-- run.py                  Main CLI entry point
+-- quick_test.py           End-to-end smoke test without webcam
+-- benchmark.py            Benchmark/report utility
+-- setup_env.sh            Local environment bootstrap
+-- run_tests.sh            Test runner
```

Generated outputs, local virtual environments, caches, and downloaded datasets are ignored by default. Keep durable model checkpoints under `models/checkpoints/`.

## Setup

```bash
chmod +x setup_env.sh run_tests.sh
./setup_env.sh
source .venv/bin/activate
```

PhoneWatch prefers `models/checkpoints/phonewatch_best.pt`. If that is unavailable, it falls back to `models/checkpoints/yolov8n.pt`, then to Ultralytics' standard `yolov8n.pt` resolution.

## Usage

```bash
python run.py webcam --camera 0
python run.py demo --camera 0
python run.py video --input data/demo.mp4 --output logs/demo_annotated.mp4
python run.py image --input data/test.jpg --output logs/test_annotated.jpg
python run.py train --data data/processed/data.yaml
python run.py dashboard
python quick_test.py
./run_tests.sh
```

## Configuration

Most project settings live in `config.yaml`, including:

- model weights and confidence thresholds
- dataset paths
- training hyperparameters
- alert mode and cooldown
- dashboard port

## Outputs

During detection, alert records are written to `logs/alert_log.csv` and session CSVs are written to `logs/session_*.csv`. Demo recordings, benchmark outputs, screenshots, and coverage reports are also written under `logs/`.

The `logs/` directory is intentionally treated as runtime output and is not committed.

## Data And Models

The repository is configured to keep large local datasets out of source control:

- `data/raw/`
- `data/processed/`
- `data/augmented/`

Model checkpoints that are needed to run the project should live in `models/checkpoints/`. The current cleanup keeps the trained PhoneWatch checkpoint and fallback YOLO checkpoint there.

## Development

```bash
source .venv/bin/activate
python -m py_compile run.py benchmark.py check_env.py download_memes.py quick_test.py dashboard/app.py dashboard/service.py src/*.py tests/*.py
./run_tests.sh
```

If dependencies are missing, rerun `./setup_env.sh`.

## License

No license has been specified yet.
