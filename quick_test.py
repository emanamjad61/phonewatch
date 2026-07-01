#!/usr/bin/env python3
"""End-to-end smoke test without a webcam: synthetic video + pipeline check."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.detect import PhoneWatchEngine
from src.utils import ensure_directories, load_config, resolve_path


def try_download_sample_video(target: Path) -> bool:
    urls = [
        "https://sample-videos.com/video321/mp4/720/big_buck_bunny_720_1s.mp4",
        "https://filesamples.com/samples/video/mp4/sample_640x360.mp4",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"Trying download: {url[:60]}...")
            urllib.request.urlretrieve(url, target)
            if target.stat().st_size > 10_000:
                print(f"Downloaded: {target}")
                return True
        except OSError as exc:
            print(f"  Failed: {exc}")
    return False


def generate_synthetic_video(path: Path, seconds: float = 10.0, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 640, 360
    n_frames = int(seconds * fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")

    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (28, 32, 38)
        t = i / max(1, n_frames - 1)
        px1 = int(80 + 200 * t)
        py1 = int(40 + 80 * np.sin(t * 6.28))
        px2, py2 = px1 + 120, py1 + 220
        cv2.rectangle(frame, (px1, py1), (px2, py2), (60, 120, 200), 2)
        cv2.putText(frame, "person (synthetic)", (px1, max(22, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 210, 230), 1)

        ox1 = int(260 + 120 * np.sin(t * 4.0))
        oy1 = int(120 + 40 * t)
        ox2, oy2 = ox1 + 56, oy1 + 100
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (80, 70, 240), 2)
        cv2.putText(frame, "phone", (ox1, max(18, oy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 255), 1)

        cv2.putText(
            frame,
            "PhoneWatch quick_test synthetic",
            (16, h - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()
    print(f"Generated synthetic video: {path} ({n_frames} frames)")


def main() -> int:
    print("PhoneWatch quick_test — verifying components\n")

    config = load_config(resolve_path("config.yaml"))
    ensure_directories(config)

    data_dir = resolve_path("data")
    dl_path = data_dir / "quick_test_download.mp4"
    syn_path = data_dir / "quick_test_synthetic.mp4"

    if syn_path.exists():
        source = syn_path
    elif try_download_sample_video(dl_path):
        source = dl_path
    else:
        print("Downloads failed; generating synthetic clip with OpenCV drawing.")
        generate_synthetic_video(syn_path, seconds=10.0, fps=30)
        source = syn_path

    if not source.exists():
        print("Error: no test video available.")
        return 1

    results: dict[str, str] = {"yolo": "pending", "context": "pending", "alerts": "pending", "pipeline": "pending"}

    sample_out = resolve_path("logs/quick_test_annotated.jpg")
    sample_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        engine = PhoneWatchEngine(config_path=str(resolve_path("config.yaml")))
    except Exception as exc:
        print(f"FAIL: engine init: {exc}")
        return 1

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"FAIL: could not read {source}")
        engine.close()
        return 1

    frame_idx = 0
    max_frames = 90
    ok_frames = 0
    sample_saved = False

    try:
        while frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                ann, detections, usage = engine.process_frame(frame, frame_idx)
                ok_frames += 1
                if engine.model is not None:
                    results["yolo"] = "ok"
                if usage is not None:
                    results["context"] = "ok"
                results["alerts"] = "ok"
                results["pipeline"] = "ok"
                if frame_idx == 45 and not sample_saved:
                    cv2.imwrite(str(sample_out), ann)
                    sample_saved = True
            except Exception as exc:
                print(f"FAIL on frame {frame_idx}: {exc}")
                results["pipeline"] = f"error: {exc}"
                break
            frame_idx += 1
    finally:
        cap.release()
        engine.close()

    print("\n--- Component status ---")
    for name, status in results.items():
        print(f"  {name:12} {status}")

    print(f"\nProcessed {ok_frames} frames from {source.name}")
    if sample_out.exists():
        print(f"Sample annotated frame: {sample_out}")
    else:
        print("No sample frame saved (run may have been too short).")

    if results["pipeline"] == "ok" and ok_frames > 0:
        print("\nquick_test: PASS")
        return 0
    print("\nquick_test: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
