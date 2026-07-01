"""Alert overlays and rate-limited alert logging for PhoneWatch detections."""

from __future__ import annotations

import csv
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .utils import append_jsonl, load_config, resolve_path, utc_timestamp


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MEME_CAPTION = "📵 PUT THE PHONE DOWN"
SERIOUS_TITLE = "⚠ PHONE USAGE DETECTED"
PLACEHOLDER_MEME_URLS = [
    "https://placehold.co/640x480/png?text=Put+The+Phone+Down",
    "https://placehold.co/640x480/png?text=Seriously%3F",
    "https://placehold.co/640x480/png?text=Not+Again",
    "https://placehold.co/640x480/png?text=Phone+Jail",
    "https://placehold.co/640x480/png?text=Focus+Mode",
]


class AlertSystem:
    """Apply alert overlays, enforce per-person cooldowns, and log alerts."""

    def __init__(self, config_path: str | Path = "config.yaml", memes_dir: str | Path = "memes/"):
        self.config = load_config(config_path)
        alerts_config = self.config.get("alerts", {})
        self.mode = str(alerts_config.get("mode", "serious")).lower()
        if self.mode not in {"meme", "serious"}:
            self.mode = "serious"

        self.cooldown_seconds = float(alerts_config.get("cooldown_seconds", 5))
        self.memes_dir = resolve_path(memes_dir or alerts_config.get("meme_dir", "memes"))
        self.memes_dir.mkdir(parents=True, exist_ok=True)
        self.cooldown_tracker: dict[int, float] = {}
        self.sound_enabled = bool(alerts_config.get("sound_enabled", True))
        self.flash_counter = 0
        self.total_alerts_today = 0
        self.last_alerts: list[dict[str, Any]] = []
        self.alert_log_path = resolve_path("logs/alert_log.csv")
        self.meme_images = self._load_memes()
        if not self.meme_images:
            self._download_placeholder_memes(count=5)
            self.meme_images = self._load_memes()
        if not self.meme_images:
            self.meme_images = self._generate_fallback_memes(count=5)

    def overlay_meme(self, frame, phone_box, meme_image):
        """Place a meme overlay in the frame with a caption."""
        return self._overlay_meme(frame, phone_box, meme_image, avoid_boxes=[phone_box])

    def get_random_meme(self):
        """Return a random loaded meme image."""
        if not self.meme_images:
            return None
        return random.choice(self.meme_images).copy()

    def draw_serious_alert(self, frame, phone_box, person_box, confidence):
        """Draw high-visibility serious alert graphics on the frame."""
        annotated = frame.copy()
        phone = _normalize_box(phone_box)
        person = _normalize_box(person_box) if person_box is not None else None

        x1, y1, x2, y2 = [int(value) for value in phone]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        if person is not None:
            px1, py1, px2, py2 = [int(value) for value in person]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 165, 255), 2)

        self.flash_counter += 1
        banner_alpha = 0.85 if self.flash_counter % 2 == 0 else 0.45
        banner_height = min(92, max(64, annotated.shape[0] // 8))
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (annotated.shape[1], banner_height), (0, 0, 210), -1)
        annotated = cv2.addWeighted(overlay, banner_alpha, annotated, 1.0 - banner_alpha, 0)

        subtitle = f"Confidence: {float(confidence):.0%} | Please put your phone away"
        annotated = self._draw_text_with_outline(
            annotated,
            SERIOUS_TITLE,
            (16, 14),
            font_size=max(20, banner_height // 3),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
        )
        annotated = self._draw_text_with_outline(
            annotated,
            subtitle,
            (16, banner_height - 34),
            font_size=max(15, banner_height // 5),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
        )
        return annotated

    def process_frame(self, frame, usage_events, frame_count):
        """Apply alert overlays and log cooldown-limited alerts for active usage events."""
        annotated = frame.copy()
        now = time.time()
        self.last_alerts = []

        for usage_event in usage_events:
            event = self._event_to_dict(usage_event)
            if not event["in_use"]:
                continue

            person_id = event["person_id"]
            if now - self.cooldown_tracker.get(person_id, 0.0) < self.cooldown_seconds:
                continue

            confidence = event["confidence"]
            if self.mode == "meme":
                meme = self.get_random_meme()
                if meme is not None:
                    annotated = self._overlay_meme(
                        annotated,
                        event["phone_box"],
                        meme,
                        avoid_boxes=[box for box in (event["phone_box"], event["person_box"]) if box is not None],
                    )
                else:
                    annotated = self.draw_serious_alert(annotated, event["phone_box"], event["person_box"], confidence)
            else:
                annotated = self.draw_serious_alert(annotated, event["phone_box"], event["person_box"], confidence)

            self.cooldown_tracker[person_id] = now
            self.total_alerts_today += 1
            alert_record = self._log_alert(person_id=person_id, mode=self.mode, confidence=confidence, frame_count=frame_count)
            self.last_alerts.append(alert_record)
            self._sound_alert()

        return annotated

    def draw_status_hud(self, frame, fps, total_alerts_today, mode):
        """Draw a compact status HUD in the bottom-left corner."""
        annotated = frame.copy()
        text = f"FPS: {float(fps):.1f} | Mode: {mode} | Alerts today: {int(total_alerts_today)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        margin = 12
        padding = 10
        box_x1 = margin
        box_y2 = annotated.shape[0] - margin
        box_x2 = min(annotated.shape[1] - margin, box_x1 + text_width + padding * 2)
        box_y1 = max(margin, box_y2 - text_height - baseline - padding * 2)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0)
        cv2.putText(
            annotated,
            text,
            (box_x1 + padding, box_y2 - padding - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return annotated

    def toggle_mode(self) -> str:
        """Switch between meme and serious alert modes."""
        self.mode = "serious" if self.mode == "meme" else "meme"
        return self.mode

    def _overlay_meme(self, frame, phone_box, meme_image, avoid_boxes):
        annotated = frame.copy()
        frame_height, frame_width = annotated.shape[:2]
        meme = self._prepare_meme_image(meme_image, frame_width, frame_height)
        if meme is None:
            return annotated

        meme_height, meme_width = meme.shape[:2]
        caption_height = max(34, frame_height // 22)
        padding = 12
        margin = 16
        overlay_width = min(frame_width - margin * 2, meme_width + padding * 2)
        overlay_height = min(frame_height - margin * 2, meme_height + caption_height + padding * 3)
        x, y = self._choose_overlay_position(
            frame_width=frame_width,
            frame_height=frame_height,
            overlay_width=overlay_width,
            overlay_height=overlay_height,
            avoid_boxes=[_normalize_box(box) for box in avoid_boxes],
            margin=margin,
        )

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x, y), (x + overlay_width, y + overlay_height), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0)

        image_x = x + padding
        image_y = y + padding
        image_region = annotated[image_y : image_y + meme_height, image_x : image_x + meme_width]
        if meme.shape[2] == 4:
            alpha = meme[:, :, 3:4].astype(np.float32) / 255.0
            image_region[:] = (alpha * meme[:, :, :3] + (1.0 - alpha) * image_region).astype(np.uint8)
        else:
            image_region[:] = meme[:, :, :3]

        caption_x = x + padding
        caption_y = image_y + meme_height + max(8, padding // 2)
        annotated = self._draw_text_with_outline(
            annotated,
            MEME_CAPTION,
            (caption_x, caption_y),
            font_size=max(16, caption_height // 2),
            fill=(255, 255, 255),
            outline=(0, 0, 0),
        )
        return annotated

    def _load_memes(self) -> list[np.ndarray]:
        meme_images = []
        for path in sorted(self.memes_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                meme_images.append(image)
        return meme_images

    def _download_placeholder_memes(self, count: int) -> None:
        urls = PLACEHOLDER_MEME_URLS[:count]
        for index, url in enumerate(urls, start=1):
            target = self.memes_dir / f"placeholder_reaction_{index:02d}.png"
            try:
                with urllib.request.urlopen(url, timeout=4) as response:
                    target.write_bytes(response.read())
                print(f"Downloaded meme placeholder: {target}")
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                print(f"Could not download meme placeholder {url}: {exc}")

    def _generate_fallback_memes(self, count: int) -> list[np.ndarray]:
        memes = []
        for index in range(count):
            image = np.full((360, 480, 3), (35 + index * 18, 45, 80 + index * 22), dtype=np.uint8)
            cv2.rectangle(image, (24, 24), (456, 336), (245, 245, 245), 5)
            cv2.putText(image, "PHONEWATCH", (58, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(image, "REACTION", (90, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3, cv2.LINE_AA)
            memes.append(image)
        return memes

    def _prepare_meme_image(self, meme_image, frame_width: int, frame_height: int):
        if meme_image is None:
            return None
        meme = meme_image.copy()
        if meme.ndim == 2:
            meme = cv2.cvtColor(meme, cv2.COLOR_GRAY2BGR)

        height, width = meme.shape[:2]
        if height <= 0 or width <= 0:
            return None

        max_width = max(120, int(frame_width * 0.26))
        max_height = max(100, int(frame_height * 0.26))
        scale = min(max_width / width, max_height / height, 1.0)
        target_width = max(1, int(width * scale))
        target_height = max(1, int(height * scale))
        return cv2.resize(meme, (target_width, target_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _choose_overlay_position(frame_width, frame_height, overlay_width, overlay_height, avoid_boxes, margin):
        candidates = [
            (frame_width - overlay_width - margin, margin),
            (margin, margin),
            (frame_width - overlay_width - margin, frame_height - overlay_height - margin),
            (margin, frame_height - overlay_height - margin),
        ]
        best_position = candidates[0]
        best_overlap = float("inf")
        for x, y in candidates:
            x = max(margin, min(frame_width - overlay_width - margin, x))
            y = max(margin, min(frame_height - overlay_height - margin, y))
            overlay_box = (x, y, x + overlay_width, y + overlay_height)
            overlap = sum(_intersection_area(overlay_box, box) for box in avoid_boxes)
            if overlap < best_overlap:
                best_overlap = overlap
                best_position = (x, y)
            if overlap == 0:
                return x, y
        return best_position

    def _draw_text_with_outline(self, frame, text, xy, font_size, fill, outline):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        font = self._load_font(font_size)
        x, y = xy
        try:
            draw.text((x, y), text, font=font, fill=fill[::-1], stroke_width=2, stroke_fill=outline[::-1])
        except UnicodeEncodeError:
            fallback = text.encode("ascii", errors="ignore").decode("ascii").strip() or "PHONE ALERT"
            draw.text((x, y), fallback, font=font, fill=fill[::-1], stroke_width=2, stroke_fill=outline[::-1])
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _load_font(font_size: int):
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, font_size)
        return ImageFont.load_default()

    @staticmethod
    def _event_to_dict(usage_event) -> dict[str, Any]:
        if isinstance(usage_event, dict):
            getter = usage_event.get
        else:
            getter = lambda key, default=None: getattr(usage_event, key, default)
        person_id = int(getter("person_id", -1))
        return {
            "phone_box": _normalize_box(getter("phone_box")),
            "person_box": _normalize_box(getter("person_box")) if getter("person_box") is not None else None,
            "in_use": bool(getter("in_use", False)),
            "confidence": float(getter("confidence", 0.0)),
            "person_id": person_id,
        }

    def _log_alert(self, person_id: int, mode: str, confidence: float, frame_count: int) -> dict[str, Any]:
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.alert_log_path.exists() or self.alert_log_path.stat().st_size == 0
        row = {
            "timestamp": utc_timestamp(),
            "person_id": person_id,
            "mode": mode,
            "confidence": round(float(confidence), 4),
            "frame_count": int(frame_count),
        }
        with self.alert_log_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row.keys()))
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        return row

    def _sound_alert(self) -> None:
        if self.sound_enabled:
            print("\a", end="", flush=True)


class AlertManager:
    """Backward-compatible rate-limited alert dispatcher."""

    def __init__(self, config: dict):
        self.config = config
        self.cooldown = float(config["alerts"]["cooldown_seconds"])
        self.mode = config["alerts"]["mode"]
        self.last_alert_at = 0.0

    def _select_meme(self) -> str | None:
        meme_dir = resolve_path(self.config["alerts"]["meme_dir"])
        files = [path for path in meme_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES]
        return str(random.choice(files)) if files else None

    def trigger(self, confidence: float, frame_id: int | None = None) -> dict | None:
        """Emit an alert if cooldown has expired."""
        now = time.time()
        if now - self.last_alert_at < self.cooldown:
            return None

        self.last_alert_at = now
        event = {
            "timestamp": utc_timestamp(),
            "frame_id": frame_id,
            "event": "phone_usage_detected",
            "confidence": round(float(confidence), 4),
            "mode": self.mode,
            "meme": self._select_meme() if self.mode == "meme" else None,
        }
        append_jsonl(self.config["project"]["log_file"], event)
        print(f"PhoneWatch alert: phone usage detected ({confidence:.2f})")
        return event


def _normalize_box(box) -> tuple[float, float, float, float]:
    if box is None:
        raise ValueError("box cannot be None")
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def _intersection_area(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height
