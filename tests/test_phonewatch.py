"""Pytest suite for PhoneWatch dataset, context, alerts, and engine."""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

import dashboard.service as dashboard_service
from src.alerts import AlertSystem
from src.context import PhoneUsageDetector
from src.dataset import DatasetManager
from src.detect import PhoneWatchEngine

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestDatasetManager:
    def test_yolo_label_format(self):
        manager = DatasetManager(config_path=PROJECT_ROOT / "config.yaml")
        bbox = [100.0, 200.0, 50.0, 80.0]
        cx, cy, w, h = manager._coco_bbox_to_yolo(bbox, image_width=640.0, image_height=480.0)
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0
        x1 = (cx - w / 2.0) * 640.0
        y1 = (cy - h / 2.0) * 480.0
        x2 = (cx + w / 2.0) * 640.0
        y2 = (cy + h / 2.0) * 480.0
        assert math.isclose(x1, 100.0, rel_tol=1e-5)
        assert math.isclose(y1, 200.0, rel_tol=1e-5)
        assert math.isclose(x2 - x1, 50.0, rel_tol=1e-5)
        assert math.isclose(y2 - y1, 80.0, rel_tol=1e-5)

    def test_train_val_test_split(self, tmp_path):
        manager = DatasetManager(config_path=PROJECT_ROOT / "config.yaml")
        ds1 = tmp_path / "one"
        ds2 = tmp_path / "two"
        for idx in range(5):
            img = ds1 / "images"
            lbl = ds1 / "labels"
            img.mkdir(parents=True, exist_ok=True)
            lbl.mkdir(parents=True, exist_ok=True)
            (img / f"i{idx}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (lbl / f"i{idx}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        for idx in range(5):
            img = ds2 / "images"
            lbl = ds2 / "labels"
            img.mkdir(parents=True, exist_ok=True)
            lbl.mkdir(parents=True, exist_ok=True)
            (img / f"j{idx}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            (lbl / f"j{idx}.txt").write_text("1 0.4 0.4 0.2 0.2\n", encoding="utf-8")

        out = tmp_path / "merged"
        data_yaml = manager.merge_datasets([ds1, ds2], out, train_split=0.7, val_split=0.2, test_split=0.1)
        assert abs(0.7 + 0.2 + 0.1 - 1.0) < 1e-9

        stems = {"train": set(), "val": set(), "test": set()}
        for split in stems:
            image_dir = out / "images" / split
            if not image_dir.exists():
                continue
            for path in image_dir.glob("*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    stems[split].add(path.stem)

        all_stems = stems["train"] | stems["val"] | stems["test"]
        assert len(all_stems) == sum(len(stems[s]) for s in stems)
        assert len(all_stems) == 10

    def test_data_yaml_structure(self, temp_dataset_dir):
        manager = DatasetManager(config_path=PROJECT_ROOT / "config.yaml")
        out = temp_dataset_dir / "merged_out"
        data_yaml_path = manager.merge_datasets(
            [temp_dataset_dir / "dataset_a", temp_dataset_dir / "dataset_b"],
            out,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        )
        assert data_yaml_path.resolve() == (out / "data.yaml").resolve()
        with data_yaml_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        for key in ("train", "val", "test", "nc", "names"):
            assert key in data
        assert int(data["nc"]) == len(manager.classes)

    def test_merge_no_duplicates(self, temp_dataset_dir):
        manager = DatasetManager(config_path=PROJECT_ROOT / "config.yaml")
        out = temp_dataset_dir / "merged_dup"
        manager.merge_datasets(
            [temp_dataset_dir / "dataset_a", temp_dataset_dir / "dataset_b"],
            out,
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        )
        names: list[str] = []
        for split in ("train", "val", "test"):
            image_dir = out / "images" / split
            if not image_dir.exists():
                continue
            for path in sorted(image_dir.glob("*.jpg")):
                names.append(path.name)
        assert len(names) == len(set(names))


class TestContextClassifier:
    @pytest.fixture(autouse=True)
    def _require_mediapipe(self):
        pytest.importorskip("mediapipe")

    @staticmethod
    def _no_hands():
        return {
            "hand_detected_near_phone": False,
            "hand_confidence": 0.0,
            "hand_landmarks": [],
            "backend": "mock",
        }

    def test_phone_on_desk(self):
        detector = PhoneUsageDetector()
        try:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            person_box = (50.0, 50.0, 200.0, 700.0)
            phone_box = (600.0, 400.0, 650.0, 500.0)
            detections = [
                {"class": "person", "confidence": 0.9, "box": person_box},
                {"class": "phone", "confidence": 0.85, "box": phone_box},
            ]
            with patch.object(PhoneUsageDetector, "analyze_hand_context", return_value=self._no_hands()):
                events = detector.classify_phone_usage(frame, detections)
            assert len(events) == 1
            assert events[0].in_use is False
        finally:
            detector.close()

    def test_phone_near_face(self):
        detector = PhoneUsageDetector()
        try:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            person_box = (100.0, 100.0, 300.0, 700.0)
            phone_box = (170.0, 200.0, 230.0, 280.0)
            detections = [
                {"class": "person", "confidence": 0.9, "box": person_box},
                {"class": "phone", "confidence": 0.88, "box": phone_box},
            ]
            with patch.object(PhoneUsageDetector, "analyze_hand_context", return_value=self._no_hands()):
                events = detector.classify_phone_usage(frame, detections)
            assert len(events) == 1
            assert events[0].in_use is True
        finally:
            detector.close()

    def test_no_persons(self):
        detector = PhoneUsageDetector()
        try:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            phone_box = (400.0, 300.0, 500.0, 450.0)
            detections = [{"class": "phone", "confidence": 0.9, "box": phone_box}]
            with patch.object(PhoneUsageDetector, "analyze_hand_context", return_value=self._no_hands()):
                events = detector.classify_phone_usage(frame, detections)
            assert len(events) == 1
            assert events[0].in_use is False
            assert events[0].person_id == -1
        finally:
            detector.close()

    def test_multiple_persons(self):
        detector = PhoneUsageDetector()
        try:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            person_a = (10.0, 10.0, 50.0, 100.0)
            person_b = (400.0, 100.0, 500.0, 700.0)
            phone_box = (420.0, 150.0, 460.0, 220.0)
            detections = [
                {"class": "person", "confidence": 0.8, "box": person_a},
                {"class": "person", "confidence": 0.82, "box": person_b},
                {"class": "phone", "confidence": 0.9, "box": phone_box},
            ]
            with patch.object(PhoneUsageDetector, "analyze_hand_context", return_value=self._no_hands()):
                events = detector.classify_phone_usage(frame, detections)
            assert len(events) == 1
            assert events[0].person_id == 1
        finally:
            detector.close()


class TestAlertSystem:
    def test_cooldown_respected(self, tmp_path, monkeypatch):
        memes_dir = tmp_path / "memes"
        memes_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(memes_dir / "m.jpg"), np.zeros((64, 64, 3), dtype=np.uint8))
        alerts = AlertSystem(config_path=PROJECT_ROOT / "config.yaml", memes_dir=memes_dir)
        alerts.cooldown_seconds = 2.0
        alerts.mode = "serious"

        class Evt:
            phone_box = (10, 10, 50, 50)
            person_box = (5, 5, 200, 200)
            in_use = True
            confidence = 0.9
            person_id = 0

        times = iter([1000.0, 1000.5])

        def fake_time():
            return next(times)

        monkeypatch.setattr("src.alerts.time.time", fake_time)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        before = alerts.total_alerts_today
        alerts.process_frame(frame, [Evt()], frame_count=1)
        after_first = alerts.total_alerts_today
        assert after_first == before + 1
        alerts.process_frame(frame, [Evt()], frame_count=2)
        assert alerts.total_alerts_today == after_first

    def test_mode_toggle(self, tmp_path):
        memes_dir = tmp_path / "memes"
        memes_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(memes_dir / "m.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
        alerts = AlertSystem(config_path=PROJECT_ROOT / "config.yaml", memes_dir=memes_dir)
        alerts.mode = "meme"
        assert alerts.toggle_mode() == "serious"
        assert alerts.toggle_mode() == "meme"

    def test_frame_annotation(self, tmp_path):
        memes_dir = tmp_path / "memes"
        memes_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(memes_dir / "m.jpg"), np.zeros((32, 32, 3), dtype=np.uint8))
        alerts = AlertSystem(config_path=PROJECT_ROOT / "config.yaml", memes_dir=memes_dir)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        phone_box = (50, 60, 120, 180)
        person_box = (40, 40, 300, 400)
        out = alerts.draw_serious_alert(frame, phone_box, person_box, 0.75)
        assert isinstance(out, np.ndarray)
        assert out.shape == frame.shape

    def test_meme_overlay(self, tmp_path):
        memes_dir = tmp_path / "memes"
        memes_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(memes_dir / "m.jpg"), np.zeros((120, 160, 3), dtype=np.uint8))
        alerts = AlertSystem(config_path=PROJECT_ROOT / "config.yaml", memes_dir=memes_dir)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
        phone_box = (200, 220, 320, 360)
        meme = alerts.get_random_meme()
        assert meme is not None
        out = alerts.overlay_meme(frame, phone_box, meme)
        assert isinstance(out, np.ndarray)
        assert out.shape == frame.shape


@pytest.fixture
def phone_watch_engine():
    engine = PhoneWatchEngine(config_path=str(PROJECT_ROOT / "config.yaml"))
    hand_patch = patch.object(
        engine.usage_detector,
        "analyze_hand_context",
        return_value={
            "hand_detected_near_phone": False,
            "hand_confidence": 0.0,
            "hand_landmarks": [],
            "backend": "mock",
        },
    )
    hand_patch.start()
    try:
        yield engine
    finally:
        hand_patch.stop()
        engine.close()


class TestPhoneWatchEngine:
    @pytest.fixture(autouse=True)
    def _require_ultralytics(self):
        pytest.importorskip("ultralytics")

    def test_model_loads(self, phone_watch_engine):
        assert phone_watch_engine.model is not None

    def test_single_image_inference(self, phone_watch_engine, tmp_path):
        image_path = tmp_path / "probe.jpg"
        cv2.imwrite(str(image_path), np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8))
        frame = cv2.imread(str(image_path))
        annotated, detections, usage_events = phone_watch_engine.process_frame(frame, frame_count=0)
        assert isinstance(annotated, np.ndarray)
        assert isinstance(detections, list)
        assert isinstance(usage_events, list)

    def test_output_format(self, phone_watch_engine, sample_frame):
        result = phone_watch_engine.process_frame(sample_frame, frame_count=0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        frame, detections, usage_events = result
        assert isinstance(frame, np.ndarray)
        assert isinstance(detections, list)
        assert isinstance(usage_events, list)

    def test_fps_benchmark(self, phone_watch_engine):
        stats = phone_watch_engine.benchmark_mode(n_frames=3)
        assert stats["average_fps"] > 0.0
        assert stats["frames"] == 3.0


class TestDashboardServices:
    def test_benchmark_rows_normalize_success_and_error(self):
        rows = dashboard_service.benchmark_to_rows(
            {
                "cpu": {
                    "device": "cpu",
                    "average_fps": 12.34,
                    "frames": 30,
                    "elapsed_seconds": 2.43,
                },
                "gpu": {
                    "device": "gpu",
                    "error": "device unavailable",
                },
            }
        )

        assert rows[0]["device"] == "cpu"
        assert rows[0]["average_fps"] == "12.3"
        assert rows[1]["device"] == "gpu"
        assert rows[1]["status"] == "device unavailable"

    def test_analytics_payload_aggregates_logs(self, monkeypatch):
        now = pd.Timestamp.now(tz="UTC")
        alerts = pd.DataFrame(
            [
                {"timestamp": now, "person_id": 0, "mode": "meme", "confidence": 0.91},
                {"timestamp": now - pd.Timedelta(minutes=10), "person_id": 1, "mode": "serious", "confidence": 0.82},
            ]
        )
        usage = pd.DataFrame(
            [
                {"timestamp": now, "person_id": 0, "duration_seconds": 4.5},
            ]
        )
        sessions = pd.DataFrame(
            [
                {"timestamp": now, "person_id": 0, "in_use": True},
                {"timestamp": now, "person_id": 1, "in_use": False},
            ]
        )

        monkeypatch.setattr(dashboard_service, "read_alerts", lambda: alerts)
        monkeypatch.setattr(dashboard_service, "read_usage", lambda: usage)
        monkeypatch.setattr(dashboard_service, "read_session_events", lambda: sessions)

        payload = dashboard_service.analytics_payload(query="usage")

        assert payload["summary"]["total_alerts_today"] == 2
        assert payload["summary"]["avg_alert_duration"] == "4.50s"
        assert payload["alerts_by_person"][0]["alerts"] == 1
        assert payload["usage_breakdown"][0]["count"] == 1
        assert len(payload["events"]) == 1
        assert payload["events"][0]["mode"] == "usage"
