"""Shared utilities for PhoneWatch."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_SMALL_AREA = 32 * 32
COCO_MEDIUM_AREA = 96 * 96


def resolve_path(path: str | Path) -> Path:
    """Resolve project-relative paths."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load the YAML configuration file."""
    with resolve_path(config_path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_directories(config: dict[str, Any]) -> None:
    """Create configured project directories."""
    for value in config.get("dataset", {}).values():
        if isinstance(value, str) and not value.endswith(".yaml"):
            resolve_path(value).mkdir(parents=True, exist_ok=True)
    resolve_path("models/checkpoints").mkdir(parents=True, exist_ok=True)
    resolve_path("logs").mkdir(parents=True, exist_ok=True)
    resolve_path(config["alerts"]["meme_dir"]).mkdir(parents=True, exist_ok=True)


def setup_logging(config: dict[str, Any]) -> None:
    """Configure console and file logging."""
    log_file = resolve_path(config["project"]["log_file"])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file.with_suffix(".log")), logging.StreamHandler()],
    )


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file."""
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def draw_label(frame, text: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
    """Draw a readable label on an OpenCV frame."""
    x, y = xy
    cv2.rectangle(frame, (x, y - 22), (x + max(80, len(text) * 9), y), color, -1)
    cv2.putText(frame, text, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def analyze_dataset(data_yaml_path: str | Path) -> dict[str, Any]:
    """Analyze a YOLO dataset and save summary plots."""
    data_yaml_path = resolve_path(data_yaml_path)
    data = _load_data_yaml(data_yaml_path)
    class_names = _class_names(data)
    split_images: dict[str, list[Path]] = {}
    split_counts: dict[str, int] = {}
    annotations_per_class: Counter[int] = Counter()
    bbox_size_counts: Counter[str] = Counter({"small": 0, "medium": 0, "large": 0})
    bbox_areas: list[float] = []
    bbox_widths: list[float] = []
    bbox_heights: list[float] = []
    bbox_classes: list[int] = []
    missing_labels: list[Path] = []
    total_images = 0
    total_annotations = 0

    for split in ("train", "val", "test"):
        images = _collect_split_images(data_yaml_path, data, split)
        split_images[split] = images
        split_counts[split] = len(images)
        total_images += len(images)

        for image_path in images:
            label_path = _label_path_for_image(image_path)
            if not label_path.exists():
                missing_labels.append(image_path)
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning("Could not read image for analysis: %s", image_path)
                continue

            image_height, image_width = image.shape[:2]
            annotations = _read_yolo_annotations(label_path)
            total_annotations += len(annotations)
            for class_id, _, _, bbox_width, bbox_height in annotations:
                width_px = bbox_width * image_width
                height_px = bbox_height * image_height
                area_px = width_px * height_px
                annotations_per_class[class_id] += 1
                bbox_size_counts[_coco_size_bucket(area_px)] += 1
                bbox_areas.append(area_px)
                bbox_widths.append(width_px)
                bbox_heights.append(height_px)
                bbox_classes.append(class_id)

    avg_annotations = total_annotations / total_images if total_images else 0.0
    imbalance_ratio = _class_imbalance_ratio(annotations_per_class, class_names)

    print("Dataset analysis")
    print(f"data.yaml: {data_yaml_path}")
    print()
    print("Images per split")
    for split in ("train", "val", "test"):
        print(f"- {split}: {split_counts.get(split, 0)}")
    print()
    print("Annotations per class")
    for class_id, class_name in class_names.items():
        print(f"- {class_id} ({class_name}): {annotations_per_class.get(class_id, 0)}")
    print()
    print(f"Average annotations per image: {avg_annotations:.2f}")
    print("Bounding box size distribution")
    print(f"- small (<32x32 px): {bbox_size_counts['small']}")
    print(f"- medium (32x32 to 96x96 px): {bbox_size_counts['medium']}")
    print(f"- large (>96x96 px): {bbox_size_counts['large']}")
    print(f"Class imbalance ratio: {_format_ratio(imbalance_ratio)}")
    if missing_labels:
        print(f"Missing label files: {len(missing_labels)}")

    analysis_path = resolve_path("logs/dataset_analysis.png")
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    _save_dataset_analysis_figure(
        analysis_path,
        split_counts,
        annotations_per_class,
        class_names,
        bbox_areas,
        bbox_widths,
        bbox_heights,
        bbox_classes,
    )
    print(f"Saved analysis figure: {analysis_path}")

    recommendation = _dataset_recommendation(
        total_images=total_images,
        total_annotations=total_annotations,
        annotations_per_class=annotations_per_class,
        class_names=class_names,
        imbalance_ratio=imbalance_ratio,
        bbox_size_counts=bbox_size_counts,
    )
    print(f"Recommendation: {recommendation}")

    return {
        "split_counts": split_counts,
        "annotations_per_class": dict(annotations_per_class),
        "average_annotations_per_image": avg_annotations,
        "bbox_size_distribution": dict(bbox_size_counts),
        "class_imbalance_ratio": imbalance_ratio,
        "missing_labels": [str(path) for path in missing_labels],
        "analysis_path": str(analysis_path),
        "recommendation": recommendation,
    }


def visualize_sample_annotations(data_yaml_path: str | Path, n_samples: int = 9) -> Path:
    """Draw YOLO annotations for random training images and save a 3x3 grid."""
    data_yaml_path = resolve_path(data_yaml_path)
    data = _load_data_yaml(data_yaml_path)
    class_names = _class_names(data)
    train_images = _collect_split_images(data_yaml_path, data, "train")
    labeled_images = [image_path for image_path in train_images if _label_path_for_image(image_path).exists()]
    if not labeled_images:
        raise RuntimeError(f"No labeled training images found in {data_yaml_path}")

    sample_count = min(n_samples, len(labeled_images))
    samples = random.Random(42).sample(labeled_images, sample_count)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 3
    rows = 3
    fig, axes = plt.subplots(rows, columns, figsize=(12, 12))
    axes_flat = list(axes.flat)

    for axis, image_path in zip(axes_flat, samples):
        image = _read_rgb_image(image_path)
        annotations = _read_yolo_annotations(_label_path_for_image(image_path))
        annotated = _draw_yolo_annotations(image, annotations, class_names)
        axis.imshow(annotated)
        axis.set_title(image_path.name, fontsize=9)
        axis.axis("off")

    for axis in axes_flat[len(samples) :]:
        axis.axis("off")

    output_path = resolve_path("logs/sample_annotations.jpg")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved sample annotation grid: {output_path}")
    return output_path


class ModelDiagnostics:
    """Diagnostics for Ultralytics training runs and YOLO model predictions."""

    CONFIDENCE_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.50

    @staticmethod
    def plot_training_curves(results_csv_path, output_path="logs/training_curves.png") -> Path:
        """Plot core Ultralytics training metrics and mark the best mAP@50 epoch."""
        results_csv_path = resolve_path(results_csv_path)
        output_path = resolve_path(output_path)
        if not results_csv_path.exists():
            raise FileNotFoundError(f"Training results CSV not found: {results_csv_path}")

        import csv
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with results_csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            columns = [str(column).strip() for column in (reader.fieldnames or [])]
            results = [
                {str(key).strip(): value for key, value in row.items() if key is not None}
                for row in reader
            ]
        if not results:
            raise ValueError(f"Training results CSV is empty: {results_csv_path}")

        epoch_column = ModelDiagnostics._first_existing_column(columns, ["epoch"])
        if epoch_column:
            epochs = []
            for index, row in enumerate(results):
                epoch_value = ModelDiagnostics._to_optional_float(row.get(epoch_column))
                epochs.append(epoch_value if epoch_value is not None else float(index + 1))
        else:
            epochs = [float(index + 1) for index in range(len(results))]

        best_epoch = ModelDiagnostics._best_epoch(results, epochs)
        panels = [
            ("Box Loss", [("train/box_loss", "train"), ("val/box_loss", "val")], "loss"),
            ("Classification Loss", [("train/cls_loss", "train"), ("val/cls_loss", "val")], "loss"),
            ("DFL Loss", [("train/dfl_loss", "train"), ("val/dfl_loss", "val")], "loss"),
            ("Precision", [("metrics/precision(B)", "precision")], "precision"),
            ("Recall", [("metrics/recall(B)", "recall")], "recall"),
            ("mAP@50", [("metrics/mAP50(B)", "mAP@50")], "mAP@50"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis, (title, series, ylabel) in zip(axes.flat, panels):
            has_data = False
            for column_name, label in series:
                column = ModelDiagnostics._first_existing_column(columns, [column_name])
                if column is None:
                    continue
                numeric_values = [ModelDiagnostics._to_optional_float(row.get(column)) for row in results]
                values = [value if value is not None else float("nan") for value in numeric_values]
                axis.plot(epochs, values, linewidth=2, marker="o", markersize=3, label=label)
                has_data = has_data or any(value is not None for value in numeric_values)

            axis.axvline(best_epoch, color="#333333", linestyle="--", linewidth=1.5, label=f"best epoch {best_epoch:g}")
            axis.set_title(title)
            axis.set_xlabel("epoch")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.25)
            if has_data:
                axis.legend()
            else:
                axis.text(0.5, 0.5, "metric not found", ha="center", va="center", transform=axis.transAxes)

        fig.suptitle("PhoneWatch Training Curves", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Training curves saved to: {output_path}")
        return output_path

    @staticmethod
    def analyze_predictions(model_path, test_images_dir, output_dir="logs/predictions") -> dict[str, Any]:
        """Run inference on test images, draw detections/labels, and group outcomes."""
        model_path = resolve_path(model_path)
        test_images_dir = resolve_path(test_images_dir)
        output_dir = resolve_path(output_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not test_images_dir.exists():
            raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run setup_env.sh or install requirements.txt.") from exc

        image_paths = sorted(path for path in test_images_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            raise RuntimeError(f"No test images found under: {test_images_dir}")

        for group in ("true_positives", "false_positives", "false_negatives"):
            (output_dir / group).mkdir(parents=True, exist_ok=True)

        model = YOLO(str(model_path))
        class_names = ModelDiagnostics._model_class_names(model)
        device = ModelDiagnostics._default_device()
        summary: dict[str, Any] = {
            "images": 0,
            "true_positive_boxes": 0,
            "false_positive_boxes": 0,
            "false_negative_boxes": 0,
            "images_with_true_positives": 0,
            "images_with_false_positives": 0,
            "images_with_false_negatives": 0,
            "output_dir": str(output_dir),
        }

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning("Skipping unreadable image: %s", image_path)
                continue

            ground_truths = ModelDiagnostics._ground_truth_boxes(image_path, image.shape, class_names)
            predictions = ModelDiagnostics._predict_image(model, image_path, device)
            matches = ModelDiagnostics._match_predictions(predictions, ground_truths, ModelDiagnostics.IOU_THRESHOLD)
            matched_prediction_indexes = {prediction_index for prediction_index, _, _ in matches}
            matched_ground_truth_indexes = {ground_truth_index for _, ground_truth_index, _ in matches}

            true_positives = len(matches)
            false_positives = len(predictions) - len(matched_prediction_indexes)
            false_negatives = len(ground_truths) - len(matched_ground_truth_indexes)

            annotated = ModelDiagnostics._draw_prediction_visualization(
                image,
                predictions,
                ground_truths,
                matched_prediction_indexes,
                matched_ground_truth_indexes,
                class_names,
            )

            groups = []
            if true_positives:
                groups.append("true_positives")
            if false_positives:
                groups.append("false_positives")
            if false_negatives:
                groups.append("false_negatives")
            if not groups:
                groups.append("true_positives")

            output_name = f"{image_path.stem}.jpg"
            for group in groups:
                cv2.imwrite(str(output_dir / group / output_name), annotated)

            summary["images"] += 1
            summary["true_positive_boxes"] += true_positives
            summary["false_positive_boxes"] += false_positives
            summary["false_negative_boxes"] += false_negatives
            summary["images_with_true_positives"] += int(true_positives > 0)
            summary["images_with_false_positives"] += int(false_positives > 0)
            summary["images_with_false_negatives"] += int(false_negatives > 0)

        print("\nPrediction analysis")
        print(f"Images processed: {summary['images']}")
        print(
            "True positives: "
            f"{summary['true_positive_boxes']} boxes in {summary['images_with_true_positives']} images"
        )
        print(
            "False positives: "
            f"{summary['false_positive_boxes']} boxes in {summary['images_with_false_positives']} images"
        )
        print(
            "False negatives: "
            f"{summary['false_negative_boxes']} boxes in {summary['images_with_false_negatives']} images"
        )
        print(f"Visualizations saved to: {output_dir}")
        return summary

    @staticmethod
    def benchmark_speed(model_path, n_runs=100) -> dict[str, dict[str, float]]:
        """Benchmark model inference latency on CPU and any available accelerator."""
        if n_runs <= 0:
            raise ValueError("n_runs must be greater than zero.")

        model_path = resolve_path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run setup_env.sh or install requirements.txt.") from exc

        import numpy as np

        torch = ModelDiagnostics._import_torch_optional()
        devices = ModelDiagnostics._benchmark_devices(torch)
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        model = YOLO(str(model_path))
        benchmark: dict[str, dict[str, float]] = {}

        for device in devices:
            try:
                warmup_runs = min(5, n_runs)
                for _ in range(warmup_runs):
                    model.predict(dummy_image, device=device, conf=ModelDiagnostics.CONFIDENCE_THRESHOLD, verbose=False)
                ModelDiagnostics._sync_device(torch, device)

                latencies_ms = []
                for _ in range(n_runs):
                    ModelDiagnostics._sync_device(torch, device)
                    start = time.perf_counter()
                    model.predict(dummy_image, device=device, conf=ModelDiagnostics.CONFIDENCE_THRESHOLD, verbose=False)
                    ModelDiagnostics._sync_device(torch, device)
                    latencies_ms.append((time.perf_counter() - start) * 1000.0)

                mean_latency = float(np.mean(latencies_ms))
                benchmark[device] = {
                    "mean_fps": 1000.0 / mean_latency if mean_latency else 0.0,
                    "median_latency_ms": float(np.median(latencies_ms)),
                    "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
                }
            except Exception as exc:
                logging.warning("Benchmark failed on %s: %s", device, exc)

        print("\nInference speed benchmark")
        print(f"Model: {model_path}")
        print(f"Runs per device: {n_runs}")
        print(f"{'Device':<10} {'Mean FPS':>12} {'Median ms':>14} {'P95 ms':>12}")
        print(f"{'-' * 10} {'-' * 12} {'-' * 14} {'-' * 12}")
        for device, metrics in benchmark.items():
            print(
                f"{device:<10} "
                f"{metrics['mean_fps']:>12.2f} "
                f"{metrics['median_latency_ms']:>14.2f} "
                f"{metrics['p95_latency_ms']:>12.2f}"
            )
        if not benchmark:
            print("No benchmark results were collected.")
        return benchmark

    @staticmethod
    def class_confusion_analysis(model_path, data_yaml_path) -> dict[str, Any]:
        """Validate the model and summarize the most common false-positive contexts."""
        model_path = resolve_path(model_path)
        data_yaml_path = resolve_path(data_yaml_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run setup_env.sh or install requirements.txt.") from exc

        data = _load_data_yaml(data_yaml_path)
        class_names = _class_names(data)
        model = YOLO(str(model_path))
        device = ModelDiagnostics._default_device()

        validation_dir = resolve_path("logs/diagnostics")
        print(f"Running validation on {device}: {data_yaml_path}")
        model.val(
            data=str(data_yaml_path),
            split="val",
            plots=False,
            device=device,
            project=str(validation_dir),
            name="class_confusion_validation",
            exist_ok=True,
            verbose=False,
        )

        validation_images = _collect_split_images(data_yaml_path, data, "val")
        if not validation_images:
            validation_images = _collect_split_images(data_yaml_path, data, "test")
        if not validation_images:
            raise RuntimeError(f"No validation or test images found in {data_yaml_path}")

        case_counts: Counter[str] = Counter()
        examples: defaultdict[str, list[str]] = defaultdict(list)
        total_false_positives = 0

        for image_path in validation_images:
            image = cv2.imread(str(image_path))
            if image is None:
                logging.warning("Skipping unreadable validation image: %s", image_path)
                continue

            ground_truths = ModelDiagnostics._ground_truth_boxes(image_path, image.shape, class_names)
            predictions = ModelDiagnostics._predict_image(model, image_path, device)
            matches = ModelDiagnostics._match_predictions(predictions, ground_truths, ModelDiagnostics.IOU_THRESHOLD)
            matched_prediction_indexes = {prediction_index for prediction_index, _, _ in matches}

            for prediction_index, prediction in enumerate(predictions):
                if prediction_index in matched_prediction_indexes:
                    continue
                total_false_positives += 1
                case = ModelDiagnostics._false_positive_case(prediction, predictions, ground_truths, class_names)
                case_counts[case] += 1
                if len(examples[case]) < 3:
                    examples[case].append(image_path.name)

        top_cases = case_counts.most_common(5)
        print("\nClass confusion analysis")
        print(f"Validation images analyzed: {len(validation_images)}")
        print(f"False-positive detections: {total_false_positives}")
        print("Phone context labels are heuristic: near/inside a person box is treated as in hand; no nearby person is treated as on desk/surface.")
        if top_cases:
            print("Top confused cases")
            for rank, (case, count) in enumerate(top_cases, start=1):
                sample_names = ", ".join(examples[case])
                print(f"{rank}. {case}: {count} cases (examples: {sample_names})")
        else:
            print("No false-positive confusion cases found.")

        return {
            "validation_images": len(validation_images),
            "false_positive_detections": total_false_positives,
            "top_cases": [{"case": case, "count": count, "examples": examples[case]} for case, count in top_cases],
        }

    @staticmethod
    def _first_existing_column(table, candidates: list[str]) -> str | None:
        source_columns = table.columns if hasattr(table, "columns") else table
        columns = {str(column).strip(): column for column in source_columns}
        for candidate in candidates:
            if candidate in columns:
                return columns[candidate]
        return None

    @staticmethod
    def _best_epoch(results, epochs) -> float:
        metric_column = ModelDiagnostics._first_existing_column(
            results[0].keys(),
            ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "mAP50", "map50"],
        )
        if metric_column is not None:
            metric_values = [
                (index, value)
                for index, row in enumerate(results)
                if (value := ModelDiagnostics._to_optional_float(row.get(metric_column))) is not None
            ]
            if metric_values:
                best_index, _ = max(metric_values, key=lambda item: item[1])
                return float(epochs[best_index])

        loss_column = ModelDiagnostics._first_existing_column(results[0].keys(), ["val/box_loss", "train/box_loss"])
        if loss_column is not None:
            loss_values = [
                (index, value)
                for index, row in enumerate(results)
                if (value := ModelDiagnostics._to_optional_float(row.get(loss_column))) is not None
            ]
            if loss_values:
                best_index, _ = min(loss_values, key=lambda item: item[1])
                return float(epochs[best_index])
        return float(len(results))

    @staticmethod
    def _to_optional_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _model_class_names(model) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {}

    @staticmethod
    def _predict_image(model, image_path: Path, device: str) -> list[dict[str, Any]]:
        result = model.predict(
            source=str(image_path),
            conf=ModelDiagnostics.CONFIDENCE_THRESHOLD,
            iou=ModelDiagnostics.IOU_THRESHOLD,
            device=device,
            verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy()
        predictions = []
        for box, confidence, class_id in zip(xyxy, confidences, class_ids):
            predictions.append(
                {
                    "box": tuple(float(value) for value in box),
                    "confidence": float(confidence),
                    "class_id": int(class_id),
                }
            )
        return predictions

    @staticmethod
    def _ground_truth_boxes(image_path: Path, image_shape, class_names: dict[int, str]) -> list[dict[str, Any]]:
        label_path = _label_path_for_image(image_path)
        if not label_path.exists():
            return []

        height, width = image_shape[:2]
        ground_truths = []
        for class_id, center_x, center_y, bbox_width, bbox_height in _read_yolo_annotations(label_path):
            ground_truths.append(
                {
                    "box": ModelDiagnostics._yolo_to_xyxy(center_x, center_y, bbox_width, bbox_height, width, height),
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, f"class {class_id}"),
                }
            )
        return ground_truths

    @staticmethod
    def _yolo_to_xyxy(
        center_x: float,
        center_y: float,
        bbox_width: float,
        bbox_height: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float]:
        x1 = (center_x - bbox_width / 2.0) * image_width
        y1 = (center_y - bbox_height / 2.0) * image_height
        x2 = (center_x + bbox_width / 2.0) * image_width
        y2 = (center_y + bbox_height / 2.0) * image_height
        return (
            max(0.0, min(float(image_width - 1), x1)),
            max(0.0, min(float(image_height - 1), y1)),
            max(0.0, min(float(image_width - 1), x2)),
            max(0.0, min(float(image_height - 1), y2)),
        )

    @staticmethod
    def _match_predictions(
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
        iou_threshold: float,
    ) -> list[tuple[int, int, float]]:
        matches = []
        used_ground_truths: set[int] = set()
        prediction_order = sorted(range(len(predictions)), key=lambda index: predictions[index]["confidence"], reverse=True)

        for prediction_index in prediction_order:
            prediction = predictions[prediction_index]
            best_ground_truth_index = None
            best_iou = 0.0
            for ground_truth_index, ground_truth in enumerate(ground_truths):
                if ground_truth_index in used_ground_truths:
                    continue
                if prediction["class_id"] != ground_truth["class_id"]:
                    continue
                iou = ModelDiagnostics._box_iou(prediction["box"], ground_truth["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_ground_truth_index = ground_truth_index

            if best_ground_truth_index is not None and best_iou >= iou_threshold:
                used_ground_truths.add(best_ground_truth_index)
                matches.append((prediction_index, best_ground_truth_index, best_iou))

        return matches

    @staticmethod
    def _draw_prediction_visualization(
        image,
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
        matched_prediction_indexes: set[int],
        matched_ground_truth_indexes: set[int],
        class_names: dict[int, str],
    ):
        annotated = image.copy()
        for index, prediction in enumerate(predictions):
            color = ModelDiagnostics._class_color(prediction["class_id"])
            status = "TP" if index in matched_prediction_indexes else "FP"
            class_id = prediction["class_id"]
            class_name = class_names.get(class_id, f"class {class_id}")
            label = f"{status} {class_name} {prediction['confidence']:.2f}"
            ModelDiagnostics._draw_solid_box(annotated, prediction["box"], color, label)

        for index, ground_truth in enumerate(ground_truths):
            color = ModelDiagnostics._class_color(ground_truth["class_id"])
            status = "matched" if index in matched_ground_truth_indexes else "missed"
            label = f"GT {ground_truth['class_name']} ({status})"
            ModelDiagnostics._draw_dashed_box(annotated, ground_truth["box"], color, label)
        return annotated

    @staticmethod
    def _draw_solid_box(image, box, color: tuple[int, int, int], label: str) -> None:
        x1, y1, x2, y2 = ModelDiagnostics._int_box(box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        ModelDiagnostics._draw_box_label(image, label, (x1, y1), color)

    @staticmethod
    def _draw_dashed_box(image, box, color: tuple[int, int, int], label: str) -> None:
        x1, y1, x2, y2 = ModelDiagnostics._int_box(box)
        ModelDiagnostics._draw_dashed_line(image, (x1, y1), (x2, y1), color)
        ModelDiagnostics._draw_dashed_line(image, (x2, y1), (x2, y2), color)
        ModelDiagnostics._draw_dashed_line(image, (x2, y2), (x1, y2), color)
        ModelDiagnostics._draw_dashed_line(image, (x1, y2), (x1, y1), color)
        ModelDiagnostics._draw_box_label(image, label, (x1, min(image.shape[0] - 1, y2 + 18)), color)

    @staticmethod
    def _draw_dashed_line(image, start: tuple[int, int], end: tuple[int, int], color: tuple[int, int, int]) -> None:
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        length = max(abs(dx), abs(dy))
        if length == 0:
            return

        dash_length = 10
        gap_length = 6
        step = dash_length + gap_length
        for offset in range(0, length, step):
            dash_end = min(offset + dash_length, length)
            point_a = (int(round(x1 + dx * offset / length)), int(round(y1 + dy * offset / length)))
            point_b = (int(round(x1 + dx * dash_end / length)), int(round(y1 + dy * dash_end / length)))
            cv2.line(image, point_a, point_b, color, 2)

    @staticmethod
    def _draw_box_label(image, label: str, xy: tuple[int, int], color: tuple[int, int, int]) -> None:
        x, y = xy
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_height = text_height + baseline + 8
        label_x2 = min(image.shape[1] - 1, x + text_width + 8)
        if y - label_height >= 0:
            label_y1 = y - label_height
            label_y2 = y
        else:
            label_y1 = y
            label_y2 = min(image.shape[0] - 1, y + label_height)
        cv2.rectangle(image, (x, label_y1), (label_x2, label_y2), color, -1)
        text_y = min(image.shape[0] - 1, label_y1 + text_height + 4)
        cv2.putText(image, label, (x + 4, text_y), font, font_scale, (255, 255, 255), thickness)

    @staticmethod
    def _int_box(box) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

    @staticmethod
    def _class_color(class_id: int) -> tuple[int, int, int]:
        palette = [
            (64, 128, 255),
            (72, 180, 90),
            (230, 140, 40),
            (190, 90, 190),
            (40, 190, 210),
            (80, 80, 220),
        ]
        return palette[class_id % len(palette)]

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)
        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)
        intersection_width = max(0.0, intersection_x2 - intersection_x1)
        intersection_height = max(0.0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection_area
        return intersection_area / union if union > 0 else 0.0

    @staticmethod
    def _false_positive_case(
        prediction: dict[str, Any],
        predictions: list[dict[str, Any]],
        ground_truths: list[dict[str, Any]],
        class_names: dict[int, str],
    ) -> str:
        predicted_name = class_names.get(prediction["class_id"], f"class {prediction['class_id']}")
        best_ground_truth = None
        best_iou = 0.0
        for ground_truth in ground_truths:
            iou = ModelDiagnostics._box_iou(prediction["box"], ground_truth["box"])
            if iou > best_iou:
                best_iou = iou
                best_ground_truth = ground_truth

        if best_ground_truth is not None and best_iou >= 0.20 and best_ground_truth["class_id"] != prediction["class_id"]:
            actual_name = class_names.get(best_ground_truth["class_id"], f"class {best_ground_truth['class_id']}")
            return f"predicted {predicted_name} over {actual_name}"

        if ModelDiagnostics._is_phone_class(prediction["class_id"], class_names):
            person_boxes = [
                ground_truth["box"]
                for ground_truth in ground_truths
                if ModelDiagnostics._is_person_class(ground_truth["class_id"], class_names)
            ]
            person_boxes.extend(
                other_prediction["box"]
                for other_prediction in predictions
                if other_prediction is not prediction
                and ModelDiagnostics._is_person_class(other_prediction["class_id"], class_names)
            )
            if ModelDiagnostics._box_near_any(prediction["box"], person_boxes):
                return "false positive phone in hand or near person"
            return "false positive phone on desk or surface"

        return f"false positive {predicted_name}"

    @staticmethod
    def _box_near_any(box, candidates: list[tuple[float, float, float, float]]) -> bool:
        if not candidates:
            return False
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        for candidate in candidates:
            cx1, cy1, cx2, cy2 = candidate
            width = max(1.0, cx2 - cx1)
            height = max(1.0, cy2 - cy1)
            expanded = (cx1 - width * 0.15, cy1 - height * 0.15, cx2 + width * 0.15, cy2 + height * 0.15)
            if expanded[0] <= center_x <= expanded[2] and expanded[1] <= center_y <= expanded[3]:
                return True
            if ModelDiagnostics._box_iou(box, candidate) > 0.01:
                return True
        return False

    @staticmethod
    def _is_phone_class(class_id: int, class_names: dict[int, str]) -> bool:
        name = class_names.get(class_id, "").lower()
        return "phone" in name or "cell" in name

    @staticmethod
    def _is_person_class(class_id: int, class_names: dict[int, str]) -> bool:
        return class_names.get(class_id, "").lower() == "person"

    @staticmethod
    def _default_device() -> str:
        torch = ModelDiagnostics._import_torch_optional()
        if torch is None:
            return "cpu"
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            return "cuda:0"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _benchmark_devices(torch) -> list[str]:
        devices = ["cpu"]
        if torch is None:
            return devices
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            devices.append("cuda:0")
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            devices.append("mps")
        return devices

    @staticmethod
    def _sync_device(torch, device: str) -> None:
        if torch is None:
            return
        if device.startswith("cuda") and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    @staticmethod
    def _import_torch_optional():
        try:
            import torch
        except ImportError:
            return None
        return torch


def _load_data_yaml(data_yaml_path: Path) -> dict[str, Any]:
    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")
    with data_yaml_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid data.yaml content: {data_yaml_path}")
    return data


def _class_names(data: dict[str, Any]) -> dict[int, str]:
    names = data.get("names", {})
    if isinstance(names, list):
        return {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    raise ValueError("data.yaml must contain names as a list or dictionary.")


def _collect_split_images(data_yaml_path: Path, data: dict[str, Any], split: str) -> list[Path]:
    if split not in data:
        return []

    split_values = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for split_value in split_values:
        split_path = _resolve_data_yaml_path(data_yaml_path, data, split_value)
        if split_path.is_file() and split_path.suffix.lower() == ".txt":
            for raw_path in split_path.read_text(encoding="utf-8").splitlines():
                raw_path = raw_path.strip()
                if raw_path:
                    images.append(_resolve_listed_image_path(split_path, raw_path))
        elif split_path.is_dir():
            images.extend(sorted(path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES))
        else:
            logging.warning("Split path does not exist for %s: %s", split, split_path)
    return sorted(dict.fromkeys(images))


def _resolve_data_yaml_path(data_yaml_path: Path, data: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    root = Path(data.get("path", data_yaml_path.parent))
    if not root.is_absolute():
        root = data_yaml_path.parent / root
    return root / path


def _resolve_listed_image_path(list_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else list_path.parent / path


def _label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _read_yolo_annotations(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    annotations = []
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            logging.warning("Skipping invalid label row at %s:%d: %s", label_path, line_number, raw_line)
            continue
        try:
            class_id = int(float(parts[0]))
            bbox = tuple(float(value) for value in parts[1:5])
        except ValueError:
            logging.warning("Skipping non-numeric label row at %s:%d: %s", label_path, line_number, raw_line)
            continue
        if any(value < 0.0 or value > 1.0 for value in bbox):
            logging.warning("Skipping out-of-range YOLO bbox at %s:%d: %s", label_path, line_number, raw_line)
            continue
        annotations.append((class_id, bbox[0], bbox[1], bbox[2], bbox[3]))
    return annotations


def _coco_size_bucket(area_px: float) -> str:
    if area_px < COCO_SMALL_AREA:
        return "small"
    if area_px < COCO_MEDIUM_AREA:
        return "medium"
    return "large"


def _class_imbalance_ratio(annotations_per_class: Counter[int], class_names: dict[int, str]) -> float:
    counts = [annotations_per_class.get(class_id, 0) for class_id in sorted(class_names)]
    if not counts or max(counts) == 0:
        return 0.0
    if min(counts) == 0:
        return float("inf")
    return max(counts) / min(counts)


def _format_ratio(ratio: float) -> str:
    if ratio == float("inf"):
        return "infinite"
    return f"{ratio:.2f}:1"


def _save_dataset_analysis_figure(
    output_path: Path,
    split_counts: dict[str, int],
    annotations_per_class: Counter[int],
    class_names: dict[int, str],
    bbox_areas: list[float],
    bbox_widths: list[float],
    bbox_heights: list[float],
    bbox_classes: list[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    split_names = ["train", "val", "test"]
    axes[0, 0].bar(split_names, [split_counts.get(split, 0) for split in split_names], color="#4c78a8")
    axes[0, 0].set_title("Images per Split")
    axes[0, 0].set_ylabel("Images")

    class_ids = sorted(class_names)
    labels = [class_names[class_id] for class_id in class_ids]
    counts = [annotations_per_class.get(class_id, 0) for class_id in class_ids]
    axes[0, 1].bar(labels, counts, color="#f58518")
    axes[0, 1].set_title("Annotations per Class")
    axes[0, 1].set_ylabel("Annotations")

    if bbox_areas:
        axes[1, 0].hist(bbox_areas, bins=30, color="#54a24b", edgecolor="black")
    else:
        axes[1, 0].text(0.5, 0.5, "No boxes", ha="center", va="center")
    axes[1, 0].set_title("Bounding Box Area Distribution")
    axes[1, 0].set_xlabel("Area (px^2)")
    axes[1, 0].set_ylabel("Count")

    color_map = defaultdict(lambda: "#6f6f6f", {0: "#e45756", 1: "#4c78a8"})
    if bbox_widths and bbox_heights:
        for class_id in sorted(set(bbox_classes)):
            indexes = [index for index, value in enumerate(bbox_classes) if value == class_id]
            axes[1, 1].scatter(
                [bbox_widths[index] for index in indexes],
                [bbox_heights[index] for index in indexes],
                label=class_names.get(class_id, f"class {class_id}"),
                c=color_map[class_id],
                alpha=0.65,
                s=18,
            )
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No boxes", ha="center", va="center")
    axes[1, 1].set_title("BBox Width vs Height")
    axes[1, 1].set_xlabel("Width (px)")
    axes[1, 1].set_ylabel("Height (px)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _dataset_recommendation(
    total_images: int,
    total_annotations: int,
    annotations_per_class: Counter[int],
    class_names: dict[int, str],
    imbalance_ratio: float,
    bbox_size_counts: Counter[str],
) -> str:
    reasons = []
    if total_images < 500:
        reasons.append("dataset has fewer than 500 images")
    if total_annotations < 1000:
        reasons.append("dataset has fewer than 1000 annotations")
    if imbalance_ratio == float("inf") or imbalance_ratio > 3.0:
        reasons.append("class imbalance is high")
    if any(annotations_per_class.get(class_id, 0) == 0 for class_id in class_names):
        reasons.append("one or more classes have no annotations")
    if total_annotations and bbox_size_counts["small"] / total_annotations > 0.7:
        reasons.append("most boxes are small, so scale/blur augmentation is important")

    if reasons:
        return "Needs more data and augmentation before training: " + "; ".join(reasons) + "."
    return "Balanced enough for an initial training run; keep augmentation enabled for robustness."


def _read_rgb_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_yolo_annotations(
    image,
    annotations: list[tuple[int, float, float, float, float]],
    class_names: dict[int, str],
):
    height, width = image.shape[:2]
    annotated = image.copy()
    colors = {
        0: (255, 0, 0),
        1: (0, 96, 255),
    }
    for class_id, center_x, center_y, bbox_width, bbox_height in annotations:
        color = colors.get(class_id, (255, 255, 0))
        x1 = int((center_x - bbox_width / 2) * width)
        y1 = int((center_y - bbox_height / 2) * height)
        x2 = int((center_x + bbox_width / 2) * width)
        y2 = int((center_y + bbox_height / 2) * height)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        label = class_names.get(class_id, f"class {class_id}")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + max(70, len(label) * 9), y1), color, -1)
        cv2.putText(annotated, label, (x1 + 4, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze and visualize a PhoneWatch YOLO dataset.")
    parser.add_argument("data_yaml_path", nargs="?", default="data/processed/data.yaml")
    parser.add_argument("--samples", type=int, default=9)
    args = parser.parse_args()
    analyze_dataset(args.data_yaml_path)
    visualize_sample_annotations(args.data_yaml_path, n_samples=args.samples)
