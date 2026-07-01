"""Real-time PhoneWatch detection engine."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .alerts import AlertSystem
from .context import PhoneUsageDetector, UsageEvent
from .utils import ensure_directories, load_config, resolve_path, setup_logging


PHONE_ALIASES = {"phone", "cell phone", "mobile phone", "cellphone", "smartphone"}
PERSON_ALIASES = {"person", "people", "human"}
SESSION_COLUMNS = [
    "timestamp",
    "frame_count",
    "fps",
    "detections",
    "usage_events",
    "alerts",
    "person_id",
    "phone_box",
    "person_box",
    "in_use",
    "confidence",
    "method_used",
]


class PhoneWatchEngine:
    """Run YOLO, phone-usage classification, alert overlays, and session logging."""

    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        model_path: str | Path | None = None,
        alert_mode: str | None = None,
        confidence_threshold: float | None = None,
    ):
        self.config_path = config_path
        self.config = load_config(config_path)
        if confidence_threshold is not None:
            self.config["model"]["confidence_threshold"] = float(confidence_threshold)
        ensure_directories(self.config)
        setup_logging(self.config)

        self.model_override_path = resolve_path(model_path) if model_path else None
        self.model, self.model_source = self._initialize_model()
        self.usage_detector = PhoneUsageDetector()
        self.alert_system = AlertSystem(config_path=config_path, memes_dir=self.config["alerts"]["meme_dir"])
        if alert_mode is not None:
            self.alert_system.mode = str(alert_mode).lower()
        self.fps_timestamps: deque[float] = deque(maxlen=30)
        self.target_class_ids = self._resolve_target_class_ids()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_csv_path = resolve_path(f"logs/session_{timestamp}.csv")
        self.session_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_csv = self.session_csv_path.open("w", encoding="utf-8", newline="")
        self.session_writer = csv.DictWriter(self.session_csv, fieldnames=SESSION_COLUMNS)
        self.session_writer.writeheader()
        self.session_csv.flush()

        self.session_started_at = time.time()
        self.session_stats = self._empty_session_stats()

    def process_frame(self, frame, frame_count: int):
        """Run one full PhoneWatch frame pass."""
        if frame is None:
            raise ValueError("process_frame received an empty frame.")

        model_cfg = self.config["model"]
        confidence_threshold = float(model_cfg.get("confidence_threshold", 0.5))
        iou_threshold = float(model_cfg.get("iou_threshold", 0.45))
        image_size = int(model_cfg.get("img_size", 640))

        try:
            results = self.model(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                classes=self.target_class_ids,
                imgsz=image_size,
                device=self.usage_detector.device,
                verbose=False,
            )[0]
        except Exception as exc:
            raise RuntimeError(f"YOLO inference failed on frame {frame_count}: {exc}") from exc

        detections = self._parse_detections(results)
        usage_events = self.usage_detector.classify_phone_usage(frame, detections)

        annotated = self._draw_live_annotations(frame.copy(), detections, usage_events)
        annotated = self.alert_system.process_frame(annotated, usage_events, frame_count=frame_count)
        fps = self._update_fps()
        annotated = self.alert_system.draw_status_hud(
            annotated,
            fps=fps,
            total_alerts_today=self.session_stats["alerts"] + len(self.alert_system.last_alerts),
            mode=self.alert_system.mode,
        )

        self._update_session_stats(detections, usage_events, self.alert_system.last_alerts)
        self._log_frame(frame_count, fps, detections, usage_events, self.alert_system.last_alerts)
        return annotated, detections, usage_events

    def run_webcam(self, camera_id: int = 0) -> dict[str, Any]:
        """Run the engine against a webcam feed."""
        capture = cv2.VideoCapture(camera_id)
        if not capture.isOpened():
            print(f"Error: could not open camera {camera_id}. Check the camera id and OS camera permissions.")
            return self.get_session_summary()

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        frame_count = 0
        last_stats_at = time.time()
        current_frame = None
        display_enabled = True

        print("PhoneWatch webcam started. Keys: q quit | m mode | s screenshot | r reset stats")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("Warning: camera frame read failed; stopping webcam loop.")
                    break

                try:
                    current_frame, detections, usage_events = self.process_frame(frame, frame_count)
                except Exception as exc:
                    print(f"Error processing frame {frame_count}: {exc}")
                    break

                if display_enabled:
                    try:
                        cv2.imshow("PhoneWatch", current_frame)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as exc:
                        print(f"Display disabled because OpenCV windowing failed: {exc}")
                        display_enabled = False
                        key = 255
                else:
                    key = 255

                if key == ord("q"):
                    break
                if key == ord("m"):
                    print(f"Alert mode: {self.alert_system.toggle_mode()}")
                elif key == ord("s") and current_frame is not None:
                    self._save_screenshot(current_frame, frame_count)
                elif key == ord("r"):
                    self.reset_session_statistics()
                    print("Session statistics reset.")

                if time.time() - last_stats_at >= 5.0:
                    self._print_live_stats()
                    last_stats_at = time.time()

                frame_count += 1
        finally:
            capture.release()
            if display_enabled:
                cv2.destroyAllWindows()
            self.close()

        summary = self.get_session_summary()
        print("PhoneWatch session summary")
        print(json.dumps(summary, indent=2))
        return summary

    def run_camera(self, camera_index: int = 0) -> dict[str, Any]:
        """Backward-compatible webcam alias."""
        return self.run_webcam(camera_index)

    def run_video_file(self, video_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
        """Run the engine over a video file, optionally writing an annotated video."""
        video_path = resolve_path(video_path)
        if not video_path.exists():
            print(f"Error: video file not found: {video_path}")
            return self.get_session_summary()

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            print(f"Error: could not open video file: {video_path}")
            return self.get_session_summary()

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
        writer = self._create_video_writer(output_path, source_fps, width, height) if output_path else None
        progress = self._progress_bar(total_frames, description=f"PhoneWatch {video_path.name}")
        display_enabled = True
        frame_count = 0
        current_frame = None

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                try:
                    current_frame, _, _ = self.process_frame(frame, frame_count)
                except Exception as exc:
                    print(f"Error processing video frame {frame_count}: {exc}")
                    break

                if writer is not None:
                    writer.write(current_frame)

                if display_enabled:
                    try:
                        cv2.imshow("PhoneWatch", current_frame)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error as exc:
                        print(f"Display disabled because OpenCV windowing failed: {exc}")
                        display_enabled = False
                        key = 255
                else:
                    key = 255

                if key == ord("q"):
                    break
                if key == ord("m"):
                    print(f"Alert mode: {self.alert_system.toggle_mode()}")
                elif key == ord("s") and current_frame is not None:
                    self._save_screenshot(current_frame, frame_count)
                elif key == ord("r"):
                    self.reset_session_statistics()
                    print("Session statistics reset.")

                frame_count += 1
                if progress is not None:
                    progress.update(1)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if progress is not None:
                progress.close()
            if display_enabled:
                cv2.destroyAllWindows()
            self.close()

        summary = self.get_session_summary()
        print("PhoneWatch video summary")
        print(json.dumps(summary, indent=2))
        return summary

    def run_image(self, image_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
        """Run detection on a single image and optionally save the annotated image."""
        image_path = resolve_path(image_path)
        if not image_path.exists():
            print(f"Error: image file not found: {image_path}")
            return {"error": f"image file not found: {image_path}"}

        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Error: OpenCV could not read image: {image_path}")
            return {"error": f"could not read image: {image_path}"}

        try:
            annotated, detections, usage_events = self.process_frame(frame, frame_count=0)
        except Exception as exc:
            print(f"Error processing image {image_path}: {exc}")
            return {"error": str(exc)}

        saved_to = None
        if output_path is not None:
            output_path = resolve_path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), annotated):
                print(f"Warning: failed to write annotated image to {output_path}")
            else:
                saved_to = str(output_path)

        return {
            "image_path": str(image_path),
            "output_path": saved_to,
            "detections": detections,
            "usage_events": [self._event_to_dict(event) for event in usage_events],
            "session": self.get_session_summary(),
        }

    def benchmark_mode(self, n_frames: int = 300) -> dict[str, float]:
        """Benchmark YOLO inference on random noise frames."""
        if n_frames <= 0:
            raise ValueError("n_frames must be greater than zero.")

        frame_shape = (720, 1280, 3)
        model_cfg = self.config["model"]
        confidence_threshold = float(model_cfg.get("confidence_threshold", 0.5))
        iou_threshold = float(model_cfg.get("iou_threshold", 0.45))
        image_size = int(model_cfg.get("img_size", 640))

        warmup_frames = min(5, n_frames)
        for _ in range(warmup_frames):
            frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)
            self.model(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                classes=self.target_class_ids,
                imgsz=image_size,
                device=self.usage_detector.device,
                verbose=False,
            )
        self._sync_device()

        start = time.perf_counter()
        for _ in range(n_frames):
            frame = np.random.randint(0, 256, frame_shape, dtype=np.uint8)
            self.model(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                classes=self.target_class_ids,
                imgsz=image_size,
                device=self.usage_detector.device,
                verbose=False,
            )
        self._sync_device()
        elapsed = time.perf_counter() - start
        average_fps = n_frames / elapsed if elapsed else 0.0

        result = {
            "frames": float(n_frames),
            "elapsed_seconds": elapsed,
            "average_fps": average_fps,
            "device": self.usage_detector.device,
        }
        print("Benchmark mode")
        print(f"Frames: {n_frames}")
        print(f"Elapsed: {elapsed:.2f}s")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Device: {self.usage_detector.device}")
        return result

    def reset_session_statistics(self) -> None:
        self.session_started_at = time.time()
        self.session_stats = self._empty_session_stats()
        self.fps_timestamps.clear()
        self.alert_system.total_alerts_today = 0
        self.alert_system.cooldown_tracker.clear()

    def run_presentation_demo(
        self,
        video_path: str | Path | None = None,
        camera_id: int = 0,
    ) -> dict[str, Any]:
        from .demo_mode import run_presentation_demo as run_demo

        vp = Path(video_path) if video_path else None
        return run_demo(self, video_path=vp, camera_id=camera_id)

    def get_session_summary(self) -> dict[str, Any]:
        elapsed = max(time.time() - self.session_started_at, 0.0)
        average_fps = self.session_stats["frames"] / elapsed if elapsed else 0.0
        per_person = {
            str(person_id): {
                "usage_events": int(counts["usage_events"]),
                "alerts": int(counts["alerts"]),
            }
            for person_id, counts in self.session_stats["per_person"].items()
        }
        return {
            "frames": int(self.session_stats["frames"]),
            "detections": int(self.session_stats["detections"]),
            "usage_events": int(self.session_stats["usage_events"]),
            "alerts": int(self.session_stats["alerts"]),
            "elapsed_seconds": round(elapsed, 2),
            "average_fps": round(average_fps, 2),
            "mode": self.alert_system.mode,
            "model_source": str(self.model_source),
            "session_csv": str(self.session_csv_path),
            "per_person": per_person,
        }

    def close(self) -> None:
        if not self.session_csv.closed:
            self.session_csv.flush()
            self.session_csv.close()
        if hasattr(self, "usage_detector"):
            self.usage_detector.close()

    def _initialize_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install requirements.txt before running detection.") from exc

        if self.model_override_path is not None:
            if not self.model_override_path.exists():
                raise FileNotFoundError(f"Requested model weights were not found: {self.model_override_path}")
            source = self.model_override_path
        else:
            checkpoint = resolve_path("models/checkpoints/phonewatch_best.pt")
            if checkpoint.exists():
                source = checkpoint
            else:
                fallback_candidates = [
                    resolve_path("yolov8n.pt"),
                    resolve_path("models/checkpoints/yolov8n.pt"),
                ]
                source = next((candidate for candidate in fallback_candidates if candidate.exists()), Path("yolov8n.pt"))
                print(
                    "Warning: models/checkpoints/phonewatch_best.pt was not found. "
                    f"Using fallback YOLO model source: {source}"
                )

        try:
            return YOLO(str(source)), source
        except Exception as exc:
            raise RuntimeError(f"Could not initialize YOLO model from {source}: {exc}") from exc

    def _resolve_target_class_ids(self) -> list[int] | None:
        names = self._class_names_from_model()
        target_ids = [
            class_id
            for class_id, class_name in names.items()
            if self._normalize_class_name(class_name) in {"phone", "person"}
        ]
        if target_ids:
            return sorted(set(target_ids))

        data_yaml_names = self._class_names_from_data_yaml()
        target_ids = [
            class_id
            for class_id, class_name in data_yaml_names.items()
            if self._normalize_class_name(class_name) in {"phone", "person"}
        ]
        if target_ids:
            return sorted(set(target_ids))

        print("Warning: could not infer model class ids from model/data.yaml; falling back to COCO person/cell-phone ids [0, 67].")
        return [0, 67]

    def _parse_detections(self, results) -> list[dict[str, Any]]:
        detections = []
        boxes = getattr(results, "boxes", None)
        if boxes is None:
            return detections

        names = getattr(results, "names", None) or self._class_names_from_model()
        for box in boxes:
            class_id = int(box.cls[0])
            raw_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            class_name = self._normalize_class_name(raw_name)
            if class_name not in {"phone", "person"}:
                continue
            xyxy = [float(value) for value in box.xyxy[0]]
            detections.append(
                {
                    "class_name": class_name,
                    "class": class_name,
                    "class_id": class_id,
                    "box": xyxy,
                    "confidence": float(box.conf[0]),
                }
            )
        return detections

    def _draw_live_annotations(self, frame, detections: list[dict[str, Any]], usage_events: list[UsageEvent]):
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection["box"]]
            class_name = detection["class_name"]
            color = (0, 170, 255) if class_name == "person" else (70, 70, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            self._draw_label(frame, f"{class_name} {detection['confidence']:.2f}", (x1, y1), color)

        for event in usage_events:
            x1, y1, x2, y2 = [int(value) for value in event.phone_box]
            color = (0, 0, 255) if event.in_use else (120, 120, 120)
            label = f"{'USE' if event.in_use else 'idle'} {event.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._draw_label(frame, label, (x1, max(18, y1 - 4)), color)
        return frame

    def _log_frame(
        self,
        frame_count: int,
        fps: float,
        detections: list[dict[str, Any]],
        usage_events: list[UsageEvent],
        alerts: list[dict[str, Any]],
    ) -> None:
        alert_count = len(alerts)
        if usage_events:
            for event in usage_events:
                self.session_writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "frame_count": frame_count,
                        "fps": round(fps, 2),
                        "detections": len(detections),
                        "usage_events": len(usage_events),
                        "alerts": alert_count,
                        "person_id": event.person_id,
                        "phone_box": json.dumps([round(value, 2) for value in event.phone_box]),
                        "person_box": json.dumps([round(value, 2) for value in event.person_box]) if event.person_box else "",
                        "in_use": event.in_use,
                        "confidence": round(event.confidence, 4),
                        "method_used": event.method_used,
                    }
                )
        else:
            self.session_writer.writerow(
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "frame_count": frame_count,
                    "fps": round(fps, 2),
                    "detections": len(detections),
                    "usage_events": 0,
                    "alerts": alert_count,
                    "person_id": "",
                    "phone_box": "",
                    "person_box": "",
                    "in_use": "",
                    "confidence": "",
                    "method_used": "",
                }
            )
        self.session_csv.flush()

    def _update_session_stats(
        self,
        detections: list[dict[str, Any]],
        usage_events: list[UsageEvent],
        alerts: list[dict[str, Any]],
    ) -> None:
        self.session_stats["frames"] += 1
        self.session_stats["detections"] += len(detections)
        active_usage_events = [event for event in usage_events if event.in_use]
        self.session_stats["usage_events"] += len(active_usage_events)
        self.session_stats["alerts"] += len(alerts)
        for event in active_usage_events:
            self.session_stats["per_person"][event.person_id]["usage_events"] += 1
        for alert in alerts:
            self.session_stats["per_person"][int(alert.get("person_id", -1))]["alerts"] += 1

    def _update_fps(self) -> float:
        now = time.time()
        self.fps_timestamps.append(now)
        if len(self.fps_timestamps) < 2:
            return 0.0
        elapsed = self.fps_timestamps[-1] - self.fps_timestamps[0]
        return (len(self.fps_timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    def _save_screenshot(self, frame, frame_count: int) -> Path:
        screenshots_dir = resolve_path("logs/screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = screenshots_dir / f"phonewatch_{timestamp}_frame_{frame_count:06d}.jpg"
        if cv2.imwrite(str(output_path), frame):
            print(f"Screenshot saved: {output_path}")
        else:
            print(f"Warning: failed to save screenshot: {output_path}")
        return output_path

    def _create_video_writer(self, output_path: str | Path, fps: float, width: int, height: int):
        output_path = resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix in {".mp4", ".m4v"} else "XVID"))
        writer = cv2.VideoWriter(str(output_path), fourcc, fps or 30.0, (width, height))
        if not writer.isOpened():
            print(f"Warning: could not open video writer for {output_path}; annotated video will not be saved.")
            return None
        return writer

    @staticmethod
    def _progress_bar(total_frames: int, description: str):
        try:
            from tqdm import tqdm
        except ImportError:
            print("tqdm is not installed; running without a progress bar.")
            return None
        return tqdm(total=total_frames if total_frames > 0 else None, desc=description, unit="frame")

    def _print_live_stats(self) -> None:
        summary = self.get_session_summary()
        print(
            "Live stats | "
            f"frames={summary['frames']} | "
            f"fps={summary['average_fps']:.2f} | "
            f"detections={summary['detections']} | "
            f"usage_events={summary['usage_events']} | "
            f"alerts={summary['alerts']} | "
            f"mode={summary['mode']}"
        )

    @staticmethod
    def _draw_label(frame, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = xy
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        y = max(text_height + baseline + 4, y)
        x2 = min(frame.shape[1] - 1, x + text_width + 8)
        y1 = max(0, y - text_height - baseline - 8)
        cv2.rectangle(frame, (x, y1), (x2, y), color, -1)
        cv2.putText(frame, text, (x + 4, y - baseline - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _class_names_from_model(self) -> dict[int, str]:
        names = getattr(self.model, "names", {})
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {}

    def _class_names_from_data_yaml(self) -> dict[int, str]:
        data_yaml_path = resolve_path(self.config["dataset"].get("data_yaml", "data/processed/data.yaml"))
        if not data_yaml_path.exists():
            return {}
        try:
            with data_yaml_path.open("r", encoding="utf-8") as stream:
                data = yaml.safe_load(stream) or {}
        except OSError:
            return {}
        names = data.get("names", {})
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {}

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        normalized = str(name).strip().lower()
        if normalized in PHONE_ALIASES:
            return "phone"
        if normalized in PERSON_ALIASES:
            return "person"
        return normalized

    @staticmethod
    def _empty_session_stats() -> dict[str, Any]:
        return {
            "frames": 0,
            "detections": 0,
            "usage_events": 0,
            "alerts": 0,
            "per_person": defaultdict(Counter),
        }

    def _event_to_dict(self, event: UsageEvent) -> dict[str, Any]:
        return {
            "phone_box": list(event.phone_box),
            "person_box": list(event.person_box) if event.person_box else None,
            "in_use": event.in_use,
            "confidence": event.confidence,
            "method_used": event.method_used,
            "person_id": event.person_id,
            "phone_id": event.phone_id,
            "reason": event.reason,
        }

    def _sync_device(self) -> None:
        try:
            import torch
        except ImportError:
            return
        device = self.usage_detector.device
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


class PhoneWatchDetector(PhoneWatchEngine):
    """Backward-compatible name for the original detector entry point."""
