"""Shared pytest fixtures and environment for PhoneWatch tests."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    os.environ.setdefault("PHONEWATCH_TEST_MODE", "1")


@pytest.fixture
def sample_frame():
    """720x1280 BGR frame (height, width, channels) with random pixels."""
    return np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Typical YOLO-style detection dicts with xyxy boxes in pixel coordinates."""
    return [
        {
            "class": "person",
            "confidence": 0.92,
            "box": (120.0, 80.0, 420.0, 620.0),
        },
        {
            "class": "phone",
            "confidence": 0.88,
            "box": (200.0, 140.0, 260.0, 220.0),
        },
    ]


def write_synthetic_yolo_pair(image_dir: Path, label_dir: Path, stem: str, lines: list[str], suffix: str = ".jpg") -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{stem}{suffix}"
    image_path.write_bytes(b"\xff\xd8\xff\xd9")
    label_path = label_dir / f"{stem}.txt"
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Two minimal synthetic YOLO datasets (images + labels) for merge/split tests."""
    ds_a = tmp_path / "dataset_a"
    ds_b = tmp_path / "dataset_b"
    for idx in range(3):
        write_synthetic_yolo_pair(
            ds_a / "images",
            ds_a / "labels",
            f"a_{idx}",
            ["0 0.5 0.5 0.2 0.2", "1 0.3 0.4 0.5 0.6"],
        )
    for idx in range(3):
        write_synthetic_yolo_pair(
            ds_b / "images",
            ds_b / "labels",
            f"b_{idx}",
            ["0 0.55 0.45 0.15 0.18"],
        )
    return tmp_path


@pytest.fixture(autouse=True)
def mock_webcam():
    """Avoid requiring a physical camera when code paths construct VideoCapture."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_cap.get.return_value = 640.0
    mock_cap.set.return_value = True
    with patch("cv2.VideoCapture", return_value=mock_cap):
        yield mock_cap
