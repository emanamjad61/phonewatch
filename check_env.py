"""Environment check script for PhoneWatch."""

from __future__ import annotations

import importlib
import os
import platform
import sys
import tempfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "phonewatch_yolo_config"))

REQUIRED_DIRECTORIES = [
    "data",
    "data/raw",
    "data/processed",
    "data/augmented",
    "models",
    "models/checkpoints",
    "src",
    "dashboard",
    "memes",
    "logs",
]

MODULES = {
    "torch": "torch",
    "ultralytics": "ultralytics",
    "cv2": "cv2",
    "mediapipe": "mediapipe",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
}


def version_of(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def record_result(failures: list[str], name: str, ok: bool, message: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {message}")
    if not ok:
        failures.append(f"{name}: {message}")


def check_python(failures: list[str]) -> None:
    version = platform.python_version()
    ok = sys.version_info >= (3, 10)
    record_result(failures, "Python", ok, f"{version} ({sys.executable})")


def check_imports(failures: list[str]) -> dict[str, Any]:
    imported: dict[str, Any] = {}
    for display_name, module_name in MODULES.items():
        try:
            module = importlib.import_module(module_name)
            imported[display_name] = module
            record_result(failures, f"Import {display_name}", True, f"version {version_of(module)}")
        except Exception as exc:
            record_result(failures, f"Import {display_name}", False, f"{type(exc).__name__}: {exc}")
    return imported


def check_cuda(failures: list[str], imported: dict[str, Any]) -> None:
    torch = imported.get("torch")
    if torch is None:
        record_result(failures, "CUDA", False, "skipped because torch could not be imported")
        return

    try:
        if torch.cuda.is_available():
            record_result(failures, "CUDA", True, f"available: {torch.cuda.get_device_name(0)}")
        else:
            record_result(failures, "CUDA", True, "not available; CPU inference will be used")
    except Exception as exc:
        record_result(failures, "CUDA", False, f"{type(exc).__name__}: {exc}")


def check_directories(failures: list[str]) -> None:
    missing = [path for path in REQUIRED_DIRECTORIES if not (PROJECT_ROOT / path).is_dir()]
    if missing:
        record_result(failures, "Project directories", False, f"missing: {', '.join(missing)}")
    else:
        record_result(failures, "Project directories", True, "all scaffold directories exist")


def check_yolo_inference(failures: list[str], imported: dict[str, Any]) -> None:
    if "ultralytics" not in imported:
        record_result(failures, "YOLOv8n inference", False, "skipped because ultralytics could not be imported")
        return

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/JPEG_example_flower.jpg/120px-JPEG_example_flower.jpg"
    try:
        from ultralytics import YOLO

        with tempfile.TemporaryDirectory(prefix="phonewatch_check_") as tmpdir:
            image_path = Path(tmpdir) / "test_image.jpg"
            print(f"Downloading test image: {image_url}")
            urllib.request.urlretrieve(image_url, image_path)

            model_source = next(
                (
                    candidate
                    for candidate in (
                        PROJECT_ROOT / "models" / "checkpoints" / "phonewatch_best.pt",
                        PROJECT_ROOT / "models" / "checkpoints" / "yolov8n.pt",
                    )
                    if candidate.exists()
                ),
                Path("yolov8n.pt"),
            )
            model = YOLO(str(model_source))
            results = model.predict(str(image_path), imgsz=320, verbose=False)
            detections = len(results[0].boxes) if results else 0
            record_result(failures, "YOLO inference", True, f"ran {model_source} with {detections} detections")
    except Exception as exc:
        record_result(failures, "YOLO inference", False, f"{type(exc).__name__}: {exc}")
        print("YOLOv8n traceback:")
        traceback.print_exc(limit=2)


def main() -> int:
    failures: list[str] = []

    print("PhoneWatch environment check")
    print(f"Project root: {PROJECT_ROOT}")
    print()

    check_python(failures)
    imported = check_imports(failures)
    check_cuda(failures, imported)
    check_directories(failures)
    check_yolo_inference(failures, imported)

    print()
    if failures:
        print("Environment check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("All systems go!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
