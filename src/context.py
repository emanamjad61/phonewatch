"""Context analysis for deciding whether a detected phone is being used."""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2


BBox = tuple[float, float, float, float]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHONE_ALIASES = {"phone", "cell phone", "mobile phone", "cellphone", "smartphone"}


@dataclass(frozen=True)
class UsageEvent:
    """Classification result for one detected phone in one frame."""

    phone_box: BBox
    person_box: BBox | None
    in_use: bool
    confidence: float
    method_used: str
    phone_id: int = 0
    person_id: int = -1
    reason: str = ""
    hand_landmarks: list[dict[str, Any]] = field(default_factory=list)
    bbox_context: dict[str, Any] = field(default_factory=dict)
    hand_context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UsageContext:
    """Backward-compatible result for the original person/phone box helper."""

    person_box: BBox
    phone_box: BBox
    overlap_ratio: float
    phone_center_inside_person: bool
    is_usage: bool


class PhoneUsageDetector:
    """Combine person-phone box context and hand landmarks to classify actual usage."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_hands: int = 4,
        device: str | None = None,
    ):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is not installed. Install requirements.txt before using PhoneUsageDetector.") from exc

        self.mp = mp
        self.hands = None
        self._hands_backend = "unavailable"
        self._vision_running_mode = None

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self._hands_backend = "solutions"
        else:
            self._init_tasks_hand_landmarker(
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        self.device = device or self._default_device()

    def analyze_bounding_box_context(
        self,
        phone_box,
        person_boxes,
        frame_height: int,
    ) -> dict[str, Any]:
        """Analyze whether a phone box is spatially associated with a person."""
        phone = normalize_box(phone_box)
        people = [normalize_box(person_box) for person_box in person_boxes]
        if not people:
            return {
                "in_use": False,
                "confidence": 0.05,
                "reason": "no person detected near phone",
                "nearest_person_id": -1,
            }

        distance_threshold = max(float(frame_height) * 0.30, 1.0)
        best: dict[str, Any] | None = None

        for person_id, person in enumerate(people):
            iou_value = box_iou(phone, person)
            distance = center_distance(phone, person)
            phone_center_x, phone_center_y = box_center(phone)
            px1, py1, px2, py2 = person
            person_height = max(py2 - py1, 1.0)
            relative_y = (phone_center_y - py1) / person_height
            inside_person = px1 <= phone_center_x <= px2 and py1 <= phone_center_y <= py2
            vertical_zone = self._vertical_zone(relative_y)
            likely_resting = self._looks_resting_on_surface(phone, person, frame_height, distance)
            proximity_match = iou_value > 0.05 or distance < distance_threshold
            in_use = proximity_match and not likely_resting
            distance_score = max(0.0, 1.0 - distance / distance_threshold)
            iou_score = min(1.0, iou_value / 0.20)
            vertical_score = self._vertical_score(relative_y, inside_person)
            confidence = clamp(0.15 + 0.45 * max(iou_score, distance_score) + 0.25 * vertical_score)
            if likely_resting:
                confidence = min(confidence, 0.30)
            if not proximity_match:
                confidence = min(confidence, 0.20)

            reason_parts = [
                f"IoU={iou_value:.3f}",
                f"distance={distance:.1f}px",
                f"vertical_zone={vertical_zone}",
            ]
            if likely_resting:
                reason_parts.append("surface_heuristic=resting")
            elif in_use:
                reason_parts.append("person_phone_context=matched")
            else:
                reason_parts.append("person_phone_context=weak")

            candidate = {
                "in_use": in_use,
                "confidence": confidence,
                "reason": "; ".join(reason_parts),
                "nearest_person_id": person_id,
                "person_box": person,
                "iou": iou_value,
                "center_distance": distance,
                "relative_vertical_position": relative_y,
                "vertical_zone": vertical_zone,
                "likely_resting_on_surface": likely_resting,
            }
            rank = (int(in_use), max(iou_score, distance_score), -distance)
            if best is None or rank > best["_rank"]:
                best = {**candidate, "_rank": rank}

        assert best is not None
        best.pop("_rank", None)
        return best

    def analyze_hand_context(self, frame, phone_box) -> dict[str, Any]:
        """Use MediaPipe Hands to see whether any hand landmarks touch or approach the phone."""
        phone = normalize_box(phone_box)
        hands = self._detect_hands(frame)
        context = self._hand_context_from_detections(hands, phone)
        return {
            "hand_detected_near_phone": context["hand_detected_near_phone"],
            "hand_confidence": context["hand_confidence"],
            "hand_landmarks": context["hand_landmarks"],
            "backend": self._hands_backend,
        }

    def classify_phone_usage(self, frame, detections: list[dict[str, Any]]) -> list[UsageEvent]:
        """Classify each detected phone as in-use or not-in-use."""
        frame_height = int(frame.shape[0])
        phones = []
        persons = []
        person_boxes = []

        for detection in detections:
            class_name = normalize_class_name(
                detection.get("class", detection.get("class_name", detection.get("name", "")))
            )
            box = normalize_box(detection["box"])
            confidence = float(detection.get("confidence", 0.0))
            normalized = {**detection, "class": class_name, "box": box, "confidence": confidence}
            if class_name == "phone":
                phones.append(normalized)
            elif class_name == "person":
                persons.append(normalized)
                person_boxes.append(box)

        events = []
        for phone_id, phone in enumerate(phones):
            phone_box = phone["box"]
            bbox_context = self.analyze_bounding_box_context(phone_box, person_boxes, frame_height)
            hand_context = self.analyze_hand_context(frame, phone_box)
            person_id = int(bbox_context.get("nearest_person_id", -1))
            person_box = persons[person_id]["box"] if 0 <= person_id < len(persons) else None

            if hand_context["hand_detected_near_phone"]:
                in_use = True
                confidence = clamp(max(0.85, 0.70 * hand_context["hand_confidence"] + 0.30 * bbox_context["confidence"]))
                method_used = "mediapipe_hands+bounding_box" if bbox_context["in_use"] else "mediapipe_hands"
                reason = "hand landmarks detected near phone"
            elif bbox_context["in_use"]:
                in_use = True
                confidence = clamp(max(0.55, min(0.84, bbox_context["confidence"])))
                method_used = "bounding_box_proximity"
                reason = bbox_context["reason"]
            else:
                in_use = False
                strongest_usage_signal = max(float(hand_context["hand_confidence"]), float(bbox_context["confidence"]))
                confidence = clamp(max(0.55, 1.0 - strongest_usage_signal))
                method_used = "isolated_phone" if person_id == -1 else "not_in_use_context"
                reason = "phone appears isolated or resting on a surface"

            events.append(
                UsageEvent(
                    phone_box=phone_box,
                    person_box=person_box,
                    in_use=in_use,
                    confidence=confidence,
                    method_used=method_used,
                    phone_id=phone_id,
                    person_id=person_id,
                    reason=reason,
                    hand_landmarks=hand_context["hand_landmarks"],
                    bbox_context=bbox_context,
                    hand_context=hand_context,
                )
            )

        return events

    def demo(
        self,
        camera_index: int = 0,
        duration_seconds: float = 10.0,
        model_path: str | Path | None = None,
        confidence_threshold: float = 0.35,
    ) -> dict[str, Any]:
        """Run webcam inference for a short demo and print usage events in real time."""
        model = self._load_demo_model(model_path)
        tracker = UsageTracker()
        capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera index {camera_index}")

        start_time = time.time()
        frame_id = 0
        try:
            while time.time() - start_time < duration_seconds:
                ok, frame = capture.read()
                if not ok:
                    break

                detections = self._detect_with_yolo(model, frame, confidence_threshold)
                events = self.classify_phone_usage(frame, detections)
                tracker.record_events(events)

                elapsed = time.time() - start_time
                for event in events:
                    if event.in_use and tracker.smoothed_in_use(event.phone_id):
                        print(
                            f"{elapsed:05.2f}s | phone={event.phone_id} | person={event.person_id} | "
                            f"confidence={event.confidence:.2f} | method={event.method_used} | {event.reason}"
                        )
                frame_id += 1
        finally:
            capture.release()

        summary = tracker.get_session_summary()
        print("Demo summary:", summary)
        return summary

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.hands is not None:
            self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _detect_hands(self, frame) -> list[dict[str, Any]]:
        if self.hands is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        if self._hands_backend == "tasks":
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.hands.detect_for_video(mp_image, int(time.monotonic() * 1000))
            return self._tasks_hand_results(results, frame_width, frame_height)

        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        hands = []

        landmark_groups = results.multi_hand_landmarks or []
        handedness_groups = results.multi_handedness or []
        for hand_id, hand_landmarks in enumerate(landmark_groups):
            handedness_score = 1.0
            if hand_id < len(handedness_groups) and handedness_groups[hand_id].classification:
                handedness_score = float(handedness_groups[hand_id].classification[0].score)

            landmarks = []
            for landmark_id, landmark in enumerate(hand_landmarks.landmark):
                landmarks.append(
                    {
                        "hand_id": hand_id,
                        "landmark_id": landmark_id,
                        "x": float(landmark.x * frame_width),
                        "y": float(landmark.y * frame_height),
                        "z": float(landmark.z),
                    }
                )
            hands.append({"hand_id": hand_id, "score": handedness_score, "landmarks": landmarks})

        return hands

    def _init_tasks_hand_landmarker(
        self,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
    ) -> None:
        model_path = self._hand_landmarker_model_path()
        if model_path is None:
            return

        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
        except ImportError:
            return

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hands = vision.HandLandmarker.create_from_options(options)
        self._hands_backend = "tasks"

    @staticmethod
    def _hand_landmarker_model_path() -> Path | None:
        candidates = [
            resolve_project_path("models/hand_landmarker.task"),
            resolve_project_path("models/checkpoints/hand_landmarker.task"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    @staticmethod
    def _tasks_hand_results(results, frame_width: int, frame_height: int) -> list[dict[str, Any]]:
        hands = []
        hand_landmarks = getattr(results, "hand_landmarks", []) or []
        handedness = getattr(results, "handedness", []) or []
        for hand_id, landmarks in enumerate(hand_landmarks):
            score = 1.0
            if hand_id < len(handedness) and handedness[hand_id]:
                score = float(getattr(handedness[hand_id][0], "score", 1.0))

            converted = []
            for landmark_id, landmark in enumerate(landmarks):
                converted.append(
                    {
                        "hand_id": hand_id,
                        "landmark_id": landmark_id,
                        "x": float(landmark.x * frame_width),
                        "y": float(landmark.y * frame_height),
                        "z": float(landmark.z),
                    }
                )
            hands.append({"hand_id": hand_id, "score": score, "landmarks": converted})
        return hands

    def _hand_context_from_detections(self, hands: list[dict[str, Any]], phone_box: BBox) -> dict[str, Any]:
        phone_width = max(1.0, phone_box[2] - phone_box[0])
        phone_height = max(1.0, phone_box[3] - phone_box[1])
        margin = max(24.0, 0.60 * max(phone_width, phone_height))
        near_box = expand_box(phone_box, margin)
        selected_landmarks = []
        strongest_confidence = 0.0

        for hand in hands:
            landmarks = hand["landmarks"]
            if not landmarks:
                continue
            near_count = 0
            inside_count = 0
            hand_landmarks = []
            for landmark in landmarks:
                inside = point_in_box((landmark["x"], landmark["y"]), phone_box)
                near = point_in_box((landmark["x"], landmark["y"]), near_box)
                if inside:
                    inside_count += 1
                if near:
                    near_count += 1
                    hand_landmarks.append({**landmark, "inside_phone": inside, "near_phone": near})

            if near_count:
                near_fraction = near_count / len(landmarks)
                inside_bonus = min(0.25, inside_count / max(near_count, 1))
                confidence = clamp(0.50 + 0.30 * near_fraction + 0.15 * float(hand["score"]) + inside_bonus)
                strongest_confidence = max(strongest_confidence, confidence)
                selected_landmarks.extend(hand_landmarks)

        return {
            "hand_detected_near_phone": bool(selected_landmarks),
            "hand_confidence": strongest_confidence,
            "hand_landmarks": selected_landmarks,
        }

    @staticmethod
    def _vertical_zone(relative_y: float) -> str:
        if relative_y < 0.0:
            return "above_person"
        if relative_y <= 0.45:
            return "upper_half_near_face"
        if relative_y <= 0.85:
            return "lower_half_near_hands_or_waist"
        if relative_y <= 1.05:
            return "near_feet_or_below_hands"
        return "below_person"

    @staticmethod
    def _vertical_score(relative_y: float, inside_person: bool) -> float:
        if not inside_person:
            return 0.10
        if 0.0 <= relative_y <= 0.45:
            return 1.00
        if relative_y <= 0.85:
            return 0.80
        if relative_y <= 1.05:
            return 0.35
        return 0.10

    @staticmethod
    def _looks_resting_on_surface(phone_box: BBox, person_box: BBox, frame_height: int, distance: float) -> bool:
        phone_center_x, phone_center_y = box_center(phone_box)
        px1, py1, px2, py2 = person_box
        person_height = max(py2 - py1, 1.0)
        relative_y = (phone_center_y - py1) / person_height
        inside_person = px1 <= phone_center_x <= px2 and py1 <= phone_center_y <= py2
        weak_overlap = box_iou(phone_box, person_box) < 0.02
        far_enough = distance > 0.25 * max(float(frame_height), 1.0)
        low_in_frame = phone_center_y > 0.60 * max(float(frame_height), 1.0)
        below_person_core = relative_y > 0.90
        return weak_overlap and not inside_person and (far_enough or low_in_frame or below_person_core)

    def _load_demo_model(self, model_path: str | Path | None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Install requirements.txt before running the demo.") from exc

        if model_path is not None:
            source = resolve_project_path(model_path)
        else:
            candidates = [
                resolve_project_path("models/checkpoints/phonewatch_best.pt"),
                resolve_project_path("models/checkpoints/yolov8n.pt"),
                resolve_project_path("yolov8n.pt"),
            ]
            source = next((candidate for candidate in candidates if candidate.exists()), Path("yolov8n.pt"))
        return YOLO(str(source))

    def _detect_with_yolo(self, model, frame, confidence_threshold: float) -> list[dict[str, Any]]:
        result = model.predict(
            frame,
            conf=confidence_threshold,
            iou=0.45,
            imgsz=640,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            class_name = normalize_class_name(result.names[int(box.cls[0])])
            if class_name not in {"phone", "person"}:
                continue
            detections.append(
                {
                    "class": class_name,
                    "box": tuple(float(value) for value in box.xyxy[0]),
                    "confidence": float(box.conf[0]),
                }
            )
        return detections

    @staticmethod
    def _default_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"


class UsageTracker:
    """Smooth per-phone usage decisions and keep a CSV-backed session log."""

    def __init__(self, window_size: int = 30, log_path: str | Path = "logs/usage_log.csv"):
        self.window_size = window_size
        self.log_path = resolve_project_path(log_path)
        self.history: dict[int, deque[bool]] = defaultdict(lambda: deque(maxlen=window_size))
        self.session_events: list[dict[str, Any]] = []

    def update(self, phone_id: int, in_use: bool) -> None:
        """Add one frame-level decision for a phone."""
        self.history[int(phone_id)].append(bool(in_use))

    def record_events(self, events: list[UsageEvent]) -> None:
        """Add all frame-level decisions from a detector call."""
        for event in events:
            self.update(event.phone_id, event.in_use)

    def smoothed_in_use(self, phone_id: int) -> bool:
        """Return True when usage appears in more than half of the recent window."""
        values = self.history.get(int(phone_id), deque())
        if not values:
            return False
        return sum(1 for value in values if value) / len(values) > 0.50

    def log_event(self, timestamp, person_id: int, duration_seconds: float) -> None:
        """Append a completed usage event to logs/usage_log.csv and session memory."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "timestamp": str(timestamp),
            "person_id": int(person_id),
            "duration_seconds": float(duration_seconds),
        }
        needs_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timestamp", "person_id", "duration_seconds"])
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
        self.session_events.append(row)

    def get_session_summary(self) -> dict[str, Any]:
        """Return aggregate counts, total duration, and per-person totals for this tracker."""
        per_person: dict[int, dict[str, float | int]] = defaultdict(lambda: {"events": 0, "duration_seconds": 0.0})
        total_duration = 0.0
        for event in self.session_events:
            person_id = int(event["person_id"])
            duration = float(event["duration_seconds"])
            per_person[person_id]["events"] += 1
            per_person[person_id]["duration_seconds"] += duration
            total_duration += duration

        return {
            "total_events": len(self.session_events),
            "total_duration": total_duration,
            "per_person": dict(per_person),
        }

    def get_session_returns(self) -> dict[str, Any]:
        """Alias for callers using the wording from the original requirement."""
        return self.get_session_summary()


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def normalize_class_name(name: Any) -> str:
    normalized = str(name).strip().lower()
    return "phone" if normalized in PHONE_ALIASES else normalized


def normalize_box(box) -> BBox:
    if isinstance(box, dict):
        box = box["box"]
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def clamp(value: float, low: float = 0.0, high: float = 0.99) -> float:
    return max(low, min(high, float(value)))


def box_area(box: BBox) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def box_iou(a: BBox, b: BBox) -> float:
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def box_center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return math.hypot(ax - bx, ay - by)


def center_inside(inner: BBox, outer: BBox) -> bool:
    cx, cy = box_center(inner)
    ox1, oy1, ox2, oy2 = outer
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2


def expand_box(box: BBox, margin: float) -> BBox:
    x1, y1, x2, y2 = box
    return x1 - margin, y1 - margin, x2 + margin, y2 + margin


def point_in_box(point: tuple[float, float], box: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def infer_usage(person_box: BBox, phone_box: BBox, min_overlap_ratio: float = 0.02) -> UsageContext:
    """Infer usage when a phone is spatially associated with a person."""
    person_box = normalize_box(person_box)
    phone_box = normalize_box(phone_box)
    person_area = max(box_area(person_box), 1.0)
    overlap_ratio = intersection_area(person_box, phone_box) / person_area
    inside = center_inside(phone_box, person_box)
    return UsageContext(
        person_box=person_box,
        phone_box=phone_box,
        overlap_ratio=overlap_ratio,
        phone_center_inside_person=inside,
        is_usage=inside or overlap_ratio >= min_overlap_ratio,
    )
