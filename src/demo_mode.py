"""Presentation demo: overlays, split view, recording, and terminal HUD."""

from __future__ import annotations

import io
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .utils import resolve_path

if TYPE_CHECKING:
    from .detect import PhoneWatchEngine


DEMO_BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██████╗ ██╗  ██╗ ██████╗ ███╗   ██╗███████╗██╗    ██╗ █████╗ ████████╗ ██████╗██╗  ██╗
║   ██╔══██╗██║  ██║██╔═══██╗████╗  ██║██╔════╝██║    ██║██╔══██╗╚══██╔══╝██╔════╝██║  ██║
║   ██████╔╝███████║██║   ██║██╔██╗ ██║█████╗  ██║ █╗ ██║███████║   ██║   ██║     ███████║
║   ██╔═══╝ ██╔══██║██║   ██║██║╚██╗██║██╔══╝  ██║███╗██║██╔══██║   ██║   ██║     ██╔══██║
║   ██║     ██║  ██║╚██████╔╝██║ ╚████║██║     ╚███╔███╔╝██║  ██║   ██║   ╚██████╗██║  ██║
║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝      ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝╚═╝  ╚═╝
║                                                                      ║
║                    Real-time phone usage awareness                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _suppress_stdout(fn):
    buf = io.StringIO()
    old = sys.stdout
    try:
        sys.stdout = buf
        return fn()
    finally:
        sys.stdout = old


def estimate_fps(engine: PhoneWatchEngine) -> float:
    try:
        stats = _suppress_stdout(lambda: engine.benchmark_mode(n_frames=12))
        return float(stats.get("average_fps", 0.0) or 0.0)
    except Exception:
        return 0.0


def run_opening_sequence(engine: PhoneWatchEngine, fps_hint: float) -> None:
    print(DEMO_BANNER)
    model_name = Path(str(engine.model_source)).name
    mode = str(engine.alert_system.mode).upper()
    print(f"  Model:     {model_name}")
    print(f"  FPS est.:  {fps_hint:.1f}  (short benchmark)")
    print(f"  Alert:     {mode}")
    print()
    for n in (3, 2, 1):
        print(f"\r  Starting demo in  {n}  ...", end="", flush=True)
        time.sleep(1.0)
    print("\r" + " " * 50 + "\r", end="", flush=True)


def _lerp_color(t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        u = t * 2.0
        b = int(64 + u * (0 - 64))
        g = int(200 + u * (220 - 200))
        r = int(80 + u * (0 - 80))
        return b, g, r
    u = (t - 0.5) * 2.0
    b = int(0 + u * (0 - 0))
    g = int(220 + u * (40 - 220))
    r = int(0 + u * (255 - 0))
    return b, g, r


def _draw_confidence_bar(
    frame: np.ndarray,
    x1: int,
    y2: int,
    box_w: int,
    conf: float,
) -> None:
    bar_w = max(40, min(box_w, 200))
    bar_h = 8
    y = min(frame.shape[0] - bar_h - 4, y2 + 6)
    x1 = max(0, min(x1, frame.shape[1] - bar_w - 2))
    for i in range(bar_w):
        frac = i / max(1, bar_w - 1)
        col = _lerp_color(frac * conf + (1.0 - conf) * 0.25)
        x = x1 + i
        cv2.line(frame, (x, y), (x, y + bar_h - 1), col, 1)
    cv2.rectangle(frame, (x1, y), (x1 + bar_w, y + bar_h), (255, 255, 255), 1)


def _draw_minimap(
    frame: np.ndarray,
    source_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    usage_events: list[Any],
) -> None:
    fh, fw = source_bgr.shape[:2]
    mw, mh = 200, int(200 * fh / max(fw, 1))
    small = cv2.resize(source_bgr, (mw, mh), interpolation=cv2.INTER_AREA)
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        sx1 = int(x1 * mw / max(fw, 1))
        sy1 = int(y1 * mh / max(fh, 1))
        sx2 = int(x2 * mw / max(fw, 1))
        sy2 = int(y2 * mh / max(fh, 1))
        col = (0, 170, 255) if det.get("class_name") == "person" else (70, 70, 255)
        cv2.rectangle(small, (sx1, sy1), (sx2, sy2), col, 1)
    for ev in usage_events:
        x1, y1, x2, y2 = [int(v) for v in ev.phone_box]
        sx1 = int(x1 * mw / max(fw, 1))
        sy1 = int(y1 * mh / max(fh, 1))
        sx2 = int(x2 * mw / max(fw, 1))
        sy2 = int(y2 * mh / max(fh, 1))
        col = (0, 0, 255) if ev.in_use else (120, 120, 120)
        cv2.rectangle(small, (sx1, sy1), (sx2, sy2), col, 2)
    margin = 12
    y0 = frame.shape[0] - mh - margin
    x0 = frame.shape[1] - mw - margin
    roi = frame[y0 : y0 + mh, x0 : x0 + mw]
    if roi.shape[:2] == small.shape[:2]:
        blended = cv2.addWeighted(roi, 0.35, small, 0.65, 0)
        frame[y0 : y0 + mh, x0 : x0 + mw] = blended
    cv2.rectangle(frame, (x0 - 2, y0 - 18), (x0 + mw + 2, y0 + mh + 2), (40, 40, 40), -1)
    cv2.rectangle(frame, (x0 - 2, y0 - 18), (x0 + mw + 2, y0 + mh + 2), (200, 200, 200), 1)
    cv2.putText(
        frame,
        "MINIMAP",
        (x0 + 4, y0 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_demo_top_banner(frame: np.ndarray) -> None:
    h = 42
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (20, 32, 48), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    txt = "PhoneWatch Demo  |  Press M = Meme  |  S = Serious  |  X = Split  |  R = 10s clip  |  Q = Quit"
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 245, 255), 1, cv2.LINE_AA)


def _draw_corner_panels(
    frame: np.ndarray,
    fps: float,
    n_det: int,
    n_alerts: int,
    frame_count: int,
) -> None:
    fh, fw = frame.shape[:2]

    def panel(x1, y1, w, h, lines):
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x1 + w, y1 + h), (18, 22, 30), -1)
        frame[y1 : y1 + h, x1 : x1 + w] = cv2.addWeighted(
            overlay[y1 : y1 + h, x1 : x1 + w], 0.55, frame[y1 : y1 + h, x1 : x1 + w], 0.45, 0
        )
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (100, 120, 160), 1)
        for i, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x1 + 8, y1 + 22 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 235, 245),
                1,
                cv2.LINE_AA,
            )

    panel(8, 52, 220, 72, [f"FPS: {fps:.1f}", f"Frame: {frame_count}"])
    panel(fw - 228, 52, 220, 72, [f"Detections: {n_det}", f"Alerts: {n_alerts}"])


def _draw_usage_badges(frame: np.ndarray, usage_events: list[Any]) -> None:
    for ev in usage_events:
        x1, y1, _, _ = [int(v) for v in ev.phone_box]
        text = "IN USE" if ev.in_use else "NOT IN USE"
        col = (0, 0, 255) if ev.in_use else (180, 180, 200)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        bx1, by1 = x1, max(4, y1 - th - 14)
        bx2, by2 = min(frame.shape[1] - 2, bx1 + tw + 12), by1 + th + 8
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 15, 22), -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), col, 2)
        cv2.putText(frame, text, (bx1 + 6, by2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.65, col, 2, cv2.LINE_AA)


def _apply_flash_border(frame: np.ndarray, flash_remaining: int) -> None:
    if flash_remaining <= 0:
        return
    alpha = min(0.55, 0.12 + flash_remaining * 0.05)
    overlay = frame.copy()
    t = 10 + min(18, flash_remaining * 2)
    cv2.rectangle(overlay, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 220), t)
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)


def run_presentation_demo(
    engine: PhoneWatchEngine,
    video_path: Path | None = None,
    camera_id: int = 0,
) -> dict[str, Any]:
    fps_hint = estimate_fps(engine)
    run_opening_sequence(engine, fps_hint)

    demo_video = resolve_path("data/demo_video.mp4")
    synthetic_fallback = resolve_path("data/quick_test_synthetic.mp4")
    use_file = video_path if video_path and video_path.exists() else (demo_video if demo_video.exists() else None)

    cap = None
    source_label = ""

    if use_file is not None:
        cap = cv2.VideoCapture(str(use_file))
        source_label = f"video:{use_file.name}"

    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_label = f"webcam:{camera_id}"
        use_file = None

    if not cap.isOpened():
        cap.release()
        if synthetic_fallback.exists():
            print(
                "Webcam not available (check System Settings → Privacy & Security → Camera for Terminal/iTerm/Cursor).\n"
                f"Using bundled test clip: {synthetic_fallback.name}\n"
            )
            cap = cv2.VideoCapture(str(synthetic_fallback))
            use_file = synthetic_fallback
            source_label = f"video:{synthetic_fallback.name}"

    if not cap.isOpened():
        print(
            "Could not open any video source.\n"
            "  • Grant camera access for your terminal app, or\n"
            "  • Place a clip at data/demo_video.mp4, or\n"
            "  • Run: python quick_test.py   (creates data/quick_test_synthetic.mp4), or\n"
            "  • Run: python run.py demo --video /path/to/video.mp4\n"
        )
        engine.close()
        return engine.get_session_summary()

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = resolve_path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    main_path = logs / f"demo_recording_{ts}.mp4"
    split_mode = False
    flash_left = 0
    clip_frames_left = 0
    clip_writer: cv2.VideoWriter | None = None
    clip_path: Path | None = None

    writer: cv2.VideoWriter | None = None
    writer_width: int | None = None
    writer_path = main_path
    rec_fps = min(30.0, source_fps) or 30.0

    window = "PhoneWatch Demo"
    frame_count = 0
    total_alerts_session = 0
    display_enabled = True
    recent_fps = deque(maxlen=30)
    t_last = time.perf_counter()

    print(
        "\nPhoneWatch LIVE | FPS: --.- | Detections: 0 | Alerts: 0 | Mode: "
        f"{engine.alert_system.mode.upper():8} | [Q]uit [M]eme [S]erious [X]plit [R]ecord clip\n",
        end="",
        flush=True,
    )

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                if use_file is not None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, raw = cap.read()
                if not ok:
                    break

            t_now = time.perf_counter()
            dt = t_now - t_last
            t_last = t_now
            if dt > 1e-6:
                recent_fps.append(1.0 / dt)
            live_fps = sum(recent_fps) / len(recent_fps) if recent_fps else fps_hint

            annotated, detections, usage_events = engine.process_frame(raw, frame_count)
            total_alerts_session += len(engine.alert_system.last_alerts)

            phone_seen = any(d.get("class_name") == "phone" for d in detections)
            if phone_seen:
                flash_left = max(flash_left, 14)

            vis = annotated.copy()
            _draw_demo_top_banner(vis)
            _draw_corner_panels(
                vis,
                live_fps,
                len(detections),
                total_alerts_session,
                frame_count,
            )
            for ev in usage_events:
                x1, y1, x2, y2 = [int(v) for v in ev.phone_box]
                bw = x2 - x1
                _draw_confidence_bar(vis, x1, y2, bw, float(ev.confidence))
            _draw_usage_badges(vis, usage_events)
            _draw_minimap(vis, raw, detections, usage_events)
            _apply_flash_border(vis, flash_left)
            if flash_left > 0:
                flash_left -= 1

            raw_rs = cv2.resize(raw, (w, h)) if raw.shape[:2] != (h, w) else raw
            if split_mode:
                combined = np.hstack([raw_rs, vis])
            else:
                combined = vis

            cw = int(combined.shape[1])
            ch = int(combined.shape[0])
            if writer is None or writer_width != cw:
                if writer is not None:
                    writer.release()
                writer_width = cw
                if writer_width != w:
                    writer_path = logs / f"demo_recording_{ts}_w{writer_width}.mp4"
                else:
                    writer_path = main_path
                writer = cv2.VideoWriter(
                    str(writer_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    rec_fps,
                    (cw, ch),
                )
                if not writer.isOpened():
                    print(f"Warning: could not open demo recorder at {writer_path}")
                    writer = None

            if writer is not None:
                writer.write(combined)

            if clip_frames_left > 0:
                if clip_writer is None and clip_path is not None:
                    clip_writer = cv2.VideoWriter(
                        str(clip_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        rec_fps,
                        (cw, ch),
                    )
                if clip_writer is not None and clip_writer.isOpened():
                    clip_writer.write(combined)
                clip_frames_left -= 1
                if clip_frames_left <= 0 and clip_writer is not None:
                    clip_writer.release()
                    clip_writer = None
                    print(f"\nSaved 10s clip: {clip_path}")
                    clip_path = None

            mode_str = engine.alert_system.mode.upper()
            sys.stdout.write(
                f"\rPhoneWatch LIVE | FPS: {live_fps:5.1f} | Detections: {len(detections):3} | "
                f"Alerts: {total_alerts_session:4} | Mode: {mode_str:8} | [Q]uit [M]eme [S]erious [X]plit [R]clip   "
            )
            sys.stdout.flush()

            if display_enabled:
                try:
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    cv2.imshow(window, combined)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error:
                    display_enabled = False
                    key = 255
            else:
                key = 255

            if key == ord("q") or key == 27:
                break
            if key == ord("m") or key == ord("M"):
                engine.alert_system.mode = "meme"
                print(f"\nMode: {engine.alert_system.mode}")
            if key == ord("s") or key == ord("S"):
                engine.alert_system.mode = "serious"
                print(f"\nMode: {engine.alert_system.mode}")
            if key == ord("x") or key == ord("X"):
                split_mode = not split_mode
                print(f"\nSplit screen: {split_mode}")
            if key == ord("r") or key == ord("R"):
                if clip_frames_left > 0:
                    print("\nClip already recording; wait for it to finish.")
                else:
                    clip_path = logs / f"demo_clip_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    clip_writer = None
                    clip_frames_left = int(rec_fps * 10)
                    print(f"\nRecording 10s clip to {clip_path} ...")

            frame_count += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if clip_writer is not None:
            clip_writer.release()
        if display_enabled:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
        engine.close()

    print(f"\nDemo session recording(s): started at {main_path} (extra segment if resolution changed)")
    summary = engine.get_session_summary()
    print("Session summary:", summary.get("frames"), "frames")
    return summary
