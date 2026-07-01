"""Single command-line entry point for PhoneWatch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from src.dataset import run_full_pipeline
from src.detect import PhoneWatchEngine
from src.train import PhoneWatchTrainer
from src.utils import ensure_directories, load_config, resolve_path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "models/checkpoints/phonewatch_best.pt"
BANNER = """\
╔══════════════════════════════════════╗
║     PhoneWatch v1.0 — Starting...   ║
║ Press 'q' to quit, 'm' to toggle    ║
╚══════════════════════════════════════╝
"""
EXAMPLES = """\
Examples:
  python run.py webcam --camera 0 --mode meme --confidence 0.5
  python run.py video --input data/demo.mp4 --output logs/demo_annotated.mp4
  python run.py image --input data/test.jpg --output logs/test_annotated.jpg
  python run.py train --data data/processed/data.yaml
  python run.py dashboard
  python run.py setup
  python run.py demo --camera 0
"""


class PhoneWatchArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(2, f"\nError: {message}\n\nRun `python run.py --help` for examples.\n")


def print_banner() -> None:
    print(BANNER)


def build_parser() -> argparse.ArgumentParser:
    parser = PhoneWatchArgumentParser(
        prog="python run.py",
        description="PhoneWatch: real-time phone usage detection, alerts, training, and web dashboard tools.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml (default: config.yaml)")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    webcam = subparsers.add_parser(
        "webcam",
        help="Run real-time webcam detection",
        description="Run PhoneWatch on a webcam.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    webcam.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    webcam.add_argument("--mode", choices=["meme", "serious"], help="Alert mode: meme or serious (default: from config)")
    webcam.add_argument("--model", default=DEFAULT_MODEL, help=f"Path to model weights (default: {DEFAULT_MODEL})")
    webcam.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)")

    video = subparsers.add_parser(
        "video",
        help="Run detection on a video file",
        description="Run PhoneWatch on a video file.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    video.add_argument("--input", required=True, help="Input video path")
    video.add_argument("--output", help="Optional annotated output video path")
    video.add_argument("--mode", choices=["meme", "serious"], help="Alert mode: meme or serious (default: from config)")
    video.add_argument("--model", default=DEFAULT_MODEL, help=f"Path to model weights (default: {DEFAULT_MODEL})")
    video.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)")

    image = subparsers.add_parser(
        "image",
        help="Run detection on a single image",
        description="Run PhoneWatch on one image.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    image.add_argument("--input", required=True, help="Input image path")
    image.add_argument("--output", help="Optional annotated output image path")
    image.add_argument("--mode", choices=["meme", "serious"], help="Alert mode: meme or serious (default: from config)")
    image.add_argument("--model", default=DEFAULT_MODEL, help=f"Path to model weights (default: {DEFAULT_MODEL})")
    image.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)")

    train = subparsers.add_parser(
        "train",
        help="Start the training pipeline",
        description="Train PhoneWatch on a YOLO data.yaml file.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train.add_argument("--data", default="data/processed/data.yaml", help="YOLO data.yaml path")

    subparsers.add_parser("dashboard", help="Launch the lightweight web dashboard")
    subparsers.add_parser("setup", help="Download datasets and first-time assets")

    demo = subparsers.add_parser(
        "demo",
        help="Presentation mode: countdown, overlays, split view, auto-record",
        description="Spectacular demo for project presentations. Uses data/demo_video.mp4 if present, else webcam.",
        epilog=EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    demo.add_argument("--camera", type=int, default=0, help="Webcam index when no demo video (default: 0)")
    demo.add_argument("--video", help="Override video path (default: data/demo_video.mp4 if it exists)")
    demo.add_argument("--model", default=DEFAULT_MODEL, help=f"Model weights (default: {DEFAULT_MODEL})")
    demo.add_argument("--confidence", type=float, default=0.5, help="Detection confidence threshold")
    return parser


def run_dashboard(config_path: str | Path) -> int:
    config = load_config(config_path)
    port = int(config.get("dashboard", {}).get("port", 8501))
    try:
        from dashboard.app import run_dashboard_server

        run_dashboard_server(config_path=config_path, host="127.0.0.1", port=port)
    except ModuleNotFoundError as exc:
        print(f"Error: missing dashboard dependency: {exc}. Install requirements.txt first.")
        return 1
    except Exception as exc:
        print(f"Error: dashboard exited with error: {exc}")
        return 1
    return 0


def run_setup(config_path: str | Path) -> int:
    config = load_config(config_path)
    ensure_directories(config)

    print("Setting up datasets...")
    try:
        data_yaml = run_full_pipeline(config_path)
        print(f"Dataset ready: {data_yaml}")
    except Exception as exc:
        print(f"Error: dataset setup failed: {exc}")
        return 1

    print("Downloading meme templates...")
    try:
        from download_memes import download_memes

        download_memes()
    except Exception as exc:
        print(f"Warning: meme download failed: {exc}")

    print("Setup complete.")
    return 0


def model_path_for_runtime(path: str | Path, explicit: bool) -> tuple[bool, Path | None]:
    model_path = resolve_path(path)
    if model_path.exists():
        print(f"Model check: using {model_path}")
        return True, model_path

    if explicit:
        print(f"Error: requested model file does not exist: {model_path}")
        return False, None

    fallback_candidates = [resolve_path("yolov8n.pt"), resolve_path("models/checkpoints/yolov8n.pt")]
    for candidate in fallback_candidates:
        if candidate.exists():
            print(f"Warning: default trained model not found: {model_path}")
            print(f"Model check: falling back to {candidate}")
            return True, candidate

    print(f"Warning: default trained model not found: {model_path}")
    print("Model check: falling back to Ultralytics yolov8n.pt download/use path.")
    return True, None


def camera_accessible(camera_id: int) -> bool:
    capture = cv2.VideoCapture(camera_id)
    try:
        if not capture.isOpened():
            return False
        ok, _ = capture.read()
        return bool(ok)
    finally:
        capture.release()


def make_engine(args, model_path: Path | None) -> PhoneWatchEngine:
    return PhoneWatchEngine(
        config_path=args.config,
        model_path=model_path,
        alert_mode=getattr(args, "mode", None),
        confidence_threshold=getattr(args, "confidence", None),
    )


def handle_webcam(args) -> int:
    print_banner()
    model_ok, model_path = model_path_for_runtime(args.model, explicit=args.model != DEFAULT_MODEL)
    if not model_ok:
        return 1
    if not camera_accessible(args.camera):
        print(f"Error: camera {args.camera} is not accessible. Check the camera ID and macOS camera permissions.")
        return 1

    try:
        engine = make_engine(args, model_path)
        engine.run_webcam(args.camera)
    except Exception as exc:
        print(f"Error: webcam mode failed: {exc}")
        return 1
    return 0


def handle_video(args) -> int:
    print_banner()
    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"Error: input video not found: {input_path}")
        return 1

    model_ok, model_path = model_path_for_runtime(args.model, explicit=args.model != DEFAULT_MODEL)
    if not model_ok:
        return 1

    try:
        engine = make_engine(args, model_path)
        engine.run_video_file(input_path, args.output)
    except Exception as exc:
        print(f"Error: video mode failed: {exc}")
        return 1
    return 0


def handle_image(args) -> int:
    print_banner()
    input_path = resolve_path(args.input)
    if not input_path.exists():
        print(f"Error: input image not found: {input_path}")
        return 1

    model_ok, model_path = model_path_for_runtime(args.model, explicit=args.model != DEFAULT_MODEL)
    if not model_ok:
        return 1

    try:
        engine = make_engine(args, model_path)
        result = engine.run_image(input_path, args.output)
        engine.close()
    except Exception as exc:
        print(f"Error: image mode failed: {exc}")
        return 1

    if "error" in result:
        return 1
    print(f"Detections: {len(result.get('detections', []))}")
    print(f"Usage events: {len(result.get('usage_events', []))}")
    if result.get("output_path"):
        print(f"Annotated image saved: {result['output_path']}")
    return 0


def handle_demo(args) -> int:
    model_ok, model_path = model_path_for_runtime(args.model, explicit=args.model != DEFAULT_MODEL)
    if not model_ok:
        return 1
    video_arg = resolve_path(args.video) if getattr(args, "video", None) else None
    try:
        engine = make_engine(args, model_path)
        engine.run_presentation_demo(video_path=video_arg, camera_id=args.camera)
    except Exception as exc:
        print(f"Error: demo mode failed: {exc}")
        return 1
    return 0


def handle_train(args) -> int:
    print_banner()
    data_path = resolve_path(args.data)
    if not data_path.exists():
        print(f"Error: training data YAML not found: {data_path}")
        return 1

    try:
        trainer = PhoneWatchTrainer(config_path=args.config)
        trainer.train(data_path)
    except Exception as exc:
        print(f"Error: training failed: {exc}")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "webcam":
        return handle_webcam(args)
    if args.command == "video":
        return handle_video(args)
    if args.command == "image":
        return handle_image(args)
    if args.command == "train":
        return handle_train(args)
    if args.command == "dashboard":
        print_banner()
        return run_dashboard(args.config)
    if args.command == "setup":
        print_banner()
        return run_setup(args.config)
    if args.command == "demo":
        return handle_demo(args)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
