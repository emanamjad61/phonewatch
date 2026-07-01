"""Shared dashboard services for the lightweight web UI."""

from __future__ import annotations

import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
import pandas as pd
import yaml

from src.detect import PhoneWatchEngine
from src.utils import load_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
ALERT_LOG = ROOT / "logs" / "alert_log.csv"
USAGE_LOG = ROOT / "logs" / "usage_log.csv"
TRAINING_RESULTS = ROOT / "logs" / "training" / "phonewatch_v1" / "results.csv"
DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
STREAM_MAX_WIDTH = 960
STREAM_JPEG_QUALITY = 82


def default_live_stats() -> dict[str, Any]:
    """Return the default live-detection metrics."""
    return {
        "fps": 0.0,
        "alerts": 0,
        "duration": 0.0,
        "last_alert": None,
        "error": None,
        "detections": 0,
        "usage_events": 0,
    }


class LiveDetectionController:
    """Manage the background detection worker used by the web UI."""

    def __init__(self, config_path: str | Path = CONFIG_PATH):
        self.config_path = Path(config_path)
        config = load_config(self.config_path)

        default_mode = str(config.get("alerts", {}).get("mode", "meme")).lower()
        if default_mode not in {"meme", "serious"}:
            default_mode = "meme"

        self._controls = {
            "mode": default_mode,
            "confidence": float(config.get("model", {}).get("confidence_threshold", 0.5)),
            "camera": 0,
        }
        self._lock = threading.Lock()
        self._frame_counter = 0
        self._latest_frame_bytes: bytes | None = None
        self._latest_stats = default_live_stats()
        self._session_start: float | None = None
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._reset_event: threading.Event | None = None

    def defaults(self) -> dict[str, Any]:
        """Return the current default UI controls."""
        with self._lock:
            return dict(self._controls)

    def start(self, mode: str | None = None, confidence: float | None = None, camera: int | None = None) -> dict[str, Any]:
        """Start detection or apply new controls if it is already running."""
        worker_to_start: threading.Thread | None = None

        with self._lock:
            self._apply_controls(mode=mode, confidence=confidence, camera=camera)
            if self._running:
                return self._snapshot_locked()

            self._running = True
            self._session_start = time.time()
            self._latest_stats = default_live_stats()
            self._stop_event = threading.Event()
            self._reset_event = threading.Event()
            self._worker_thread = threading.Thread(
                target=self._worker,
                args=(self._stop_event, self._reset_event),
                daemon=True,
                name="phonewatch-web-detection",
            )
            worker_to_start = self._worker_thread

        if worker_to_start is not None:
            worker_to_start.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        """Stop the detection worker."""
        stop_event: threading.Event | None = None
        worker: threading.Thread | None = None

        with self._lock:
            stop_event = self._stop_event
            worker = self._worker_thread
            self._running = False

        if stop_event is not None:
            stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.5)

        with self._lock:
            self._worker_thread = None
            self._stop_event = None
            self._reset_event = None
            return self._snapshot_locked()

    def reset(self) -> dict[str, Any]:
        """Reset the current session counters."""
        with self._lock:
            self._latest_stats = default_live_stats()
            self._session_start = time.time()
            if self._reset_event is not None:
                self._reset_event.set()
            return self._snapshot_locked()

    def close(self) -> None:
        """Release background resources."""
        self.stop()

    def snapshot(self) -> dict[str, Any]:
        """Return the latest worker state."""
        with self._lock:
            return self._snapshot_locked()

    def stream_frames(self) -> Iterator[bytes]:
        """Yield the latest annotated frame as an MJPEG stream."""
        boundary = b"--frame\r\n"
        last_frame_counter = -1
        idle_payload = self._encode_placeholder("Start detection to begin the live feed.")
        error_payload: bytes | None = None

        while True:
            payload: bytes | None = None
            sleep_time = 0.1
            with self._lock:
                frame_counter = self._frame_counter
                running = self._running
                latest_frame = self._latest_frame_bytes
                error_message = self._latest_stats.get("error")

            if error_message:
                if error_payload is None:
                    error_payload = self._encode_placeholder(error_message)
                payload = error_payload
                sleep_time = 0.4
            elif latest_frame and frame_counter != last_frame_counter:
                payload = latest_frame
                last_frame_counter = frame_counter
            elif not running:
                payload = idle_payload
                sleep_time = 0.4

            if payload is not None:
                headers = (
                    boundary
                    + b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(payload)}\r\n\r\n".encode("ascii")
                )
                yield headers + payload + b"\r\n"

            time.sleep(sleep_time)

    def _apply_controls(self, mode: str | None, confidence: float | None, camera: int | None) -> None:
        if mode is not None:
            mode_value = str(mode).strip().lower()
            if mode_value in {"meme", "serious"}:
                self._controls["mode"] = mode_value
        if confidence is not None:
            self._controls["confidence"] = min(1.0, max(0.1, float(confidence)))
        if camera is not None:
            self._controls["camera"] = max(0, int(camera))

    def _snapshot_locked(self) -> dict[str, Any]:
        stats = dict(self._latest_stats)
        if self._running and self._session_start is not None:
            stats["duration"] = max(float(stats.get("duration", 0.0)), time.time() - self._session_start)

        return {
            "running": self._running,
            "controls": dict(self._controls),
            "stats": stats,
            "frame_counter": self._frame_counter,
        }

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._latest_stats["error"] = message
            self._running = False

    def _set_frame(self, frame, stats: dict[str, Any]) -> None:
        frame_bytes = self._encode_frame(frame)
        with self._lock:
            if frame_bytes is not None:
                self._latest_frame_bytes = frame_bytes
                self._frame_counter += 1
            self._latest_stats = stats

    def _encode_frame(self, frame) -> bytes | None:
        if frame is None:
            return None

        height, width = frame.shape[:2]
        if width > STREAM_MAX_WIDTH:
            scaled_height = max(1, int(height * (STREAM_MAX_WIDTH / width)))
            frame = cv2.resize(frame, (STREAM_MAX_WIDTH, scaled_height), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY],
        )
        return encoded.tobytes() if ok else None

    def _encode_placeholder(self, message: str) -> bytes:
        canvas = np.full((540, 960, 3), 255, dtype=np.uint8)
        canvas[:] = (241, 246, 251)
        cv2.rectangle(canvas, (48, 48), (912, 492), (185, 202, 219), 2)
        cv2.putText(canvas, "PhoneWatch", (72, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (38, 74, 112), 3)
        cv2.putText(canvas, message[:72], (72, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (52, 67, 84), 2)
        cv2.putText(canvas, "Windows 7 style control center", (72, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (90, 107, 127), 2)
        payload = self._encode_frame(canvas)
        return payload or b""

    def _worker(self, stop_event: threading.Event, reset_event: threading.Event) -> None:
        engine = None
        capture = None
        started_at = time.time()
        last_alert = None
        frame_count = 0

        try:
            with self._lock:
                mode = self._controls["mode"]
                confidence = self._controls["confidence"]
                camera = self._controls["camera"]

            engine = PhoneWatchEngine(
                config_path=self.config_path,
                alert_mode=mode,
                confidence_threshold=confidence,
            )

            capture = cv2.VideoCapture(camera)
            if not capture.isOpened():
                self._set_error(f"Camera {camera} is not accessible.")
                return

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            while not stop_event.is_set():
                if reset_event.is_set():
                    engine.reset_session_statistics()
                    started_at = time.time()
                    last_alert = None
                    reset_event.clear()

                with self._lock:
                    mode = self._controls["mode"]
                    confidence = float(self._controls["confidence"])

                engine.alert_system.mode = mode
                engine.config["model"]["confidence_threshold"] = confidence

                ok, frame = capture.read()
                if not ok:
                    self._set_error("Camera frame read failed.")
                    break

                annotated, detections, usage_events = engine.process_frame(frame, frame_count)
                if engine.alert_system.last_alerts:
                    last_alert = engine.alert_system.last_alerts[-1]

                summary = engine.get_session_summary()
                stats = {
                    "fps": float(summary.get("average_fps", 0.0)),
                    "alerts": int(summary.get("alerts", 0)),
                    "duration": max(0.0, time.time() - started_at),
                    "last_alert": last_alert,
                    "error": None,
                    "detections": len(detections),
                    "usage_events": sum(1 for event in usage_events if event.in_use),
                }
                self._set_frame(annotated, stats)
                frame_count += 1
                time.sleep(0.02)
        except Exception as exc:
            self._set_error(f"Detection worker failed: {exc}")
        finally:
            if capture is not None:
                capture.release()
            if engine is not None:
                engine.close()
            with self._lock:
                self._running = False
                self._worker_thread = None
                self._stop_event = None
                self._reset_event = None


def read_csv_or_empty(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read a CSV or return an empty frame with the requested columns."""
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=columns)


def _start_of_today_for_timestamps(series: pd.Series) -> pd.Timestamp:
    if series.empty:
        return pd.Timestamp.now(tz="UTC").normalize()
    tz = getattr(series.dtype, "tz", None)
    if tz is not None:
        return pd.Timestamp.now(tz=tz).normalize()
    return pd.Timestamp.now().normalize()


def read_alerts() -> pd.DataFrame:
    """Return alert logs as a normalized dataframe."""
    alerts = read_csv_or_empty(ALERT_LOG, ["timestamp", "person_id", "mode", "confidence", "frame_count"])
    if "timestamp" in alerts:
        alerts["timestamp"] = pd.to_datetime(alerts["timestamp"], errors="coerce", utc=True)
    if "confidence" in alerts:
        alerts["confidence"] = pd.to_numeric(alerts["confidence"], errors="coerce")
    return alerts.dropna(subset=["timestamp"]) if not alerts.empty else alerts


def read_usage() -> pd.DataFrame:
    """Return usage logs as a normalized dataframe."""
    usage = read_csv_or_empty(USAGE_LOG, ["timestamp", "person_id", "duration_seconds"])
    if "timestamp" in usage:
        usage["timestamp"] = pd.to_datetime(usage["timestamp"], errors="coerce", utc=True)
    if "duration_seconds" in usage:
        usage["duration_seconds"] = pd.to_numeric(usage["duration_seconds"], errors="coerce")
    return usage.dropna(subset=["timestamp"]) if not usage.empty else usage


def read_session_events() -> pd.DataFrame:
    """Read the latest session CSV files."""
    session_files = sorted((ROOT / "logs").glob("session_*.csv"))
    frames = []
    for path in session_files[-10:]:
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "person_id", "in_use", "confidence", "method_used"])

    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data.get("timestamp"), errors="coerce", utc=True)
    if "in_use" in data:
        data["in_use"] = data["in_use"].astype(str).str.lower().map({"true": True, "false": False})
    return data.dropna(subset=["timestamp"])


def build_events_table(alerts: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
    """Combine alert and usage logs into a single table."""
    rows = []

    if not alerts.empty:
        for _, row in alerts.iterrows():
            rows.append(
                {
                    "time": row.get("timestamp"),
                    "person_id": row.get("person_id", ""),
                    "confidence": row.get("confidence", ""),
                    "mode": row.get("mode", ""),
                    "duration": "",
                }
            )

    if not usage.empty:
        for _, row in usage.iterrows():
            rows.append(
                {
                    "time": row.get("timestamp"),
                    "person_id": row.get("person_id", ""),
                    "confidence": "",
                    "mode": "usage",
                    "duration": row.get("duration_seconds", ""),
                }
            )

    return pd.DataFrame(rows, columns=["time", "person_id", "confidence", "mode", "duration"])


def _format_timestamp(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_float(value: Any, suffix: str = "") -> str:
    if value is None or value == "" or pd.isna(value):
        return ""
    return f"{float(value):.2f}{suffix}"


def analytics_payload(query: str = "") -> dict[str, Any]:
    """Return the analytics data rendered by the dashboard."""
    alerts = read_alerts()
    usage = read_usage()
    sessions = read_session_events()

    today = _start_of_today_for_timestamps(alerts["timestamp"]) if not alerts.empty else pd.Timestamp.now(tz="UTC").normalize()
    alerts_today = alerts[alerts["timestamp"] >= today] if not alerts.empty else alerts
    total_alerts_today = int(len(alerts_today))

    if not alerts.empty:
        alert_hours = alerts["timestamp"].dt.strftime("%Y-%m-%d %H:00").value_counts().sort_index()
        most_active_hour = "No data yet"
        if not alert_hours.empty:
            most_active_hour = str(alert_hours.idxmax()).split(" ")[-1]
    else:
        alert_hours = pd.Series(dtype="int64")
        most_active_hour = "No data yet"

    avg_duration = usage["duration_seconds"].mean() if not usage.empty and "duration_seconds" in usage else None

    if not alerts.empty and "person_id" in alerts:
        person_counts = alerts["person_id"].fillna("unknown").astype(str).value_counts()
    else:
        person_counts = pd.Series(dtype="int64")

    if not sessions.empty and "in_use" in sessions and not sessions["in_use"].dropna().empty:
        usage_counts = sessions["in_use"].map({True: "In use", False: "Phone on desk / not in use"}).value_counts()
    else:
        usage_counts = pd.Series(dtype="int64")

    events = build_events_table(alerts, usage)
    if not events.empty:
        events["time"] = pd.to_datetime(events["time"], errors="coerce", utc=True)
        events = events.dropna(subset=["time"]).sort_values("time", ascending=False)
        if query:
            needle = query.strip().lower()
            searchable = events.fillna("").astype(str).agg(" ".join, axis=1).str.lower()
            events = events[searchable.str.contains(needle, regex=False)]
    events_rows = []
    for _, row in events.head(50).iterrows():
        events_rows.append(
            {
                "time": _format_timestamp(row.get("time")),
                "person_id": str(row.get("person_id", "")),
                "confidence": _format_float(row.get("confidence")),
                "mode": str(row.get("mode", "")),
                "duration": _format_float(row.get("duration"), "s"),
            }
        )

    return {
        "summary": {
            "total_alerts_today": total_alerts_today,
            "most_active_hour": most_active_hour,
            "avg_alert_duration": _format_float(avg_duration, "s") or "No data yet",
        },
        "alerts_by_hour": [
            {"hour": str(hour), "alerts": int(count)}
            for hour, count in alert_hours.tail(24).items()
        ],
        "alerts_by_person": [
            {"person_id": str(person_id), "alerts": int(count)}
            for person_id, count in person_counts.items()
        ],
        "usage_breakdown": [
            {"status": str(status), "count": int(count)}
            for status, count in usage_counts.items()
        ],
        "events": events_rows,
    }


def benchmark_to_rows(benchmark: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize benchmark results for the frontend table."""
    if "average_fps" in benchmark:
        benchmark = {str(benchmark.get("device", "unknown")): benchmark}

    rows = []
    for device, result in benchmark.items():
        if result.get("error"):
            rows.append(
                {
                    "device": str(device),
                    "average_fps": "Error",
                    "frames": "",
                    "elapsed_seconds": "",
                    "status": str(result["error"]),
                }
            )
            continue

        rows.append(
            {
                "device": str(result.get("device", device)),
                "average_fps": f"{float(result.get('average_fps', 0.0)):.1f}",
                "frames": str(int(float(result.get("frames", 0)))),
                "elapsed_seconds": f"{float(result.get('elapsed_seconds', 0.0)):.2f}",
                "status": "OK",
            }
        )
    return rows


def run_benchmark_suite(n_frames: int = 30) -> dict[str, dict[str, Any]]:
    """Run the benchmark on the available inference devices."""
    engine = PhoneWatchEngine(CONFIG_PATH)
    original_device = engine.usage_detector.device
    devices = ["cpu"]
    if original_device != "cpu":
        devices.append(original_device)

    results: dict[str, dict[str, Any]] = {}
    try:
        for device in dict.fromkeys(devices):
            try:
                engine.usage_detector.device = device
                results[device] = engine.benchmark_mode(n_frames=n_frames)
            except Exception as exc:
                results[device] = {"error": str(exc), "device": device}
    finally:
        engine.usage_detector.device = original_device
        engine.close()
    return results


def benchmark_payload(n_frames: int = 30) -> dict[str, Any]:
    """Return normalized benchmark results."""
    return {"rows": benchmark_to_rows(run_benchmark_suite(n_frames=n_frames))}


def _model_signature() -> tuple[str, int] | None:
    candidates = [
        ROOT / "models" / "checkpoints" / "phonewatch_best.pt",
        ROOT / "models" / "checkpoints" / "yolov8n.pt",
        ROOT / "yolov8n.pt",
    ]
    model_path = next((path for path in candidates if path.exists()), None)
    if model_path is None:
        return None
    return str(model_path), int(model_path.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _model_architecture_info_cached(model_path_str: str, _: int) -> dict[str, str]:
    model_path = Path(model_path_str)
    try:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        parameters = sum(parameter.numel() for parameter in model.model.parameters())
        layers = len(list(model.model.modules()))
        return {
            "name": model_path.name,
            "parameters": f"{parameters:,}",
            "layers": f"{layers:,}",
        }
    except Exception:
        return {"name": model_path.name, "parameters": "Unavailable", "layers": "Unavailable"}


def get_model_architecture_info() -> dict[str, str]:
    """Return cached model metadata."""
    signature = _model_signature()
    if signature is None:
        return {"name": "No model yet", "parameters": "No data yet", "layers": "No data yet"}
    return _model_architecture_info_cached(*signature)


def get_training_metrics() -> dict[str, str]:
    """Return summarized training metrics."""
    if not TRAINING_RESULTS.exists():
        return {}
    try:
        results = pd.read_csv(TRAINING_RESULTS)
    except Exception:
        return {}
    results.columns = [str(column).strip() for column in results.columns]
    if results.empty:
        return {}

    map_col = "metrics/mAP50(B)" if "metrics/mAP50(B)" in results else None
    final_map50 = f"{float(results[map_col].iloc[-1]):.3f}" if map_col else "No data yet"
    best_epoch = "No data yet"
    if map_col:
        best_idx = int(pd.to_numeric(results[map_col], errors="coerce").idxmax())
        best_epoch = str(int(results["epoch"].iloc[best_idx])) if "epoch" in results else str(best_idx + 1)

    training_time = "No data yet"
    if "time" in results:
        seconds = float(pd.to_numeric(results["time"], errors="coerce").iloc[-1])
        training_time = f"{seconds / 60.0:.1f} min"

    return {"final_map50": final_map50, "best_epoch": best_epoch, "training_time": training_time}


def get_dataset_info() -> dict[str, Any]:
    """Return dataset volume and class-distribution information."""
    if not DATA_YAML.exists():
        return {}
    try:
        with DATA_YAML.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
    except Exception:
        return {}

    root = Path(data.get("path", DATA_YAML.parent))
    if not root.is_absolute():
        root = DATA_YAML.parent / root

    image_total = 0
    for split in ("train", "val", "test"):
        split_value = data.get(split)
        if not split_value:
            continue
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = root / split_path
        if split_path.exists():
            image_total += sum(1 for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)

    names = data.get("names", {})
    if isinstance(names, list):
        class_names = {index: str(name) for index, name in enumerate(names)}
    else:
        class_names = {int(key): str(value) for key, value in names.items()} if isinstance(names, dict) else {}

    counts = {class_id: 0 for class_id in class_names}
    labels_root = root / "labels"
    if labels_root.exists():
        for label_path in labels_root.rglob("*.txt"):
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    class_id = int(float(parts[0]))
                except ValueError:
                    continue
                counts[class_id] = counts.get(class_id, 0) + 1

    distribution = [
        {"class_name": class_names.get(class_id, str(class_id)), "count": int(count)}
        for class_id, count in sorted(counts.items())
    ]
    return {"total_images": int(image_total), "class_distribution": distribution}


def model_payload() -> dict[str, Any]:
    """Return the data needed for the model tab."""
    return {
        "architecture": get_model_architecture_info(),
        "training": get_training_metrics(),
        "dataset": get_dataset_info(),
    }
