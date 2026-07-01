"""Dataset download, merge, and validation helpers for PhoneWatch."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # type: ignore[no-redef]
        """Small fallback used before project dependencies are installed."""

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = kwargs.get("total")

        def __iter__(self):
            return iter(self.iterable or ())

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self, value):
            return None

try:
    from .utils import ensure_directories, load_config, resolve_path
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.utils import ensure_directories, load_config, resolve_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"
COCO_PHONE_YOLO_CLASS_INDEX = 67


@dataclass(frozen=True)
class ImageLabelPair:
    image_path: Path
    label_path: Path
    dataset_name: str


def collect_images(directory: str | Path) -> list[Path]:
    """Return image files under a directory."""
    root = resolve_path(directory)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def write_yolo_data_yaml(config: dict) -> Path:
    """Write the YOLO data.yaml file expected by Ultralytics."""
    dataset = config["dataset"]
    payload = {
        "path": str(resolve_path(dataset["processed"])),
        "train": str(resolve_path(dataset["train_images"])),
        "val": str(resolve_path(dataset["val_images"])),
        "nc": len(config["classes"]),
        "names": {index: name for index, name in enumerate(config["classes"])},
    }
    target = resolve_path(dataset["data_yaml"])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)
    return target


def summarize_dataset(config_path: str = "config.yaml") -> dict[str, int]:
    """Create folders and report raw, processed, and augmented image counts."""
    config = load_config(config_path)
    ensure_directories(config)
    write_yolo_data_yaml(config)
    return {
        "raw": len(collect_images(config["dataset"]["raw"])),
        "processed": len(collect_images(config["dataset"]["processed"])),
        "augmented": len(collect_images(config["dataset"]["augmented"])),
    }


class DatasetManager:
    """Manage COCO, Roboflow, and local YOLO datasets for PhoneWatch."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path = "config.yaml"):
        self.config = config if config is not None else load_config(config_path)
        self.classes = list(self.config.get("classes", ["phone", "person"]))
        self.class_to_index = {name: index for index, name in enumerate(self.classes)}
        self.random_seed = int(self.config.get("dataset", {}).get("random_seed", 42))
        self.logger = logging.getLogger("phonewatch.dataset")
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    def download_coco_phone_class(self, output_dir: str | Path) -> Path:
        """Download COCO 2017 val images containing cell phones and write YOLO labels."""
        output_dir = resolve_path(output_dir)
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        cache_dir = output_dir / "_coco_cache"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("Preparing COCO phone dataset in %s", output_dir)
        annotations_path = self._ensure_coco_val_annotations(cache_dir)
        coco = self._load_json(annotations_path)
        image_lookup = {image["id"]: image for image in coco.get("images", [])}
        annotations_by_image = self._group_annotations(coco.get("annotations", []))
        phone_category_ids = self._coco_phone_category_ids(coco)
        category_to_output_class = self._coco_category_to_output_class(coco)

        phone_image_ids = sorted(
            {
                annotation["image_id"]
                for annotation in coco.get("annotations", [])
                if annotation.get("category_id") in phone_category_ids
            }
        )
        if not phone_image_ids:
            raise RuntimeError("No COCO validation images containing the cell phone class were found.")

        self.logger.info(
            "Found %d COCO validation images containing cell phones. COCO phone category ids: %s. "
            "YOLO COCO class index reference: %d.",
            len(phone_image_ids),
            sorted(phone_category_ids),
            COCO_PHONE_YOLO_CLASS_INDEX,
        )

        written_images = 0
        written_labels = 0
        for image_id in tqdm(phone_image_ids, desc="Downloading COCO phone images", unit="image"):
            image = image_lookup.get(image_id)
            if image is None:
                self.logger.warning("Skipping missing COCO image metadata for id %s", image_id)
                continue

            file_name = image["file_name"]
            image_path = images_dir / file_name
            label_path = labels_dir / f"{Path(file_name).stem}.txt"

            try:
                if not image_path.exists():
                    image_url = image.get("coco_url") or COCO_VAL_IMAGE_URL.format(file_name=file_name)
                    urllib.request.urlretrieve(image_url, image_path)
                written_images += 1

                label_lines = self._coco_yolo_label_lines(
                    image=image,
                    annotations=annotations_by_image.get(image_id, []),
                    category_to_output_class=category_to_output_class,
                )
                if not label_lines:
                    self.logger.warning("No usable PhoneWatch labels produced for COCO image %s", file_name)
                    continue

                label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
                written_labels += 1
            except Exception as exc:
                self.logger.exception("Failed to process COCO image %s: %s", file_name, exc)

        self.logger.info("COCO download complete: %d images, %d label files", written_images, written_labels)
        return output_dir

    def download_roboflow_dataset(
        self,
        api_key: str,
        workspace: str,
        project: str,
        version: int | str,
        output_dir: str | Path,
    ) -> Path:
        """Download a Roboflow dataset in YOLOv8 format."""
        output_dir = resolve_path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading Roboflow dataset %s/%s version %s", workspace, project, version)

        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise RuntimeError("roboflow is not installed. Run setup_env.sh or install requirements.txt.") from exc

        try:
            roboflow = Roboflow(api_key=api_key)
            dataset = roboflow.workspace(workspace).project(project).version(int(version)).download(
                "yolov8",
                location=str(output_dir),
            )
            dataset_path = Path(getattr(dataset, "location", output_dir))
            self.logger.info("Roboflow dataset downloaded to %s", dataset_path)
            return dataset_path
        except Exception as exc:
            raise RuntimeError(f"Roboflow download failed for {workspace}/{project}:{version}: {exc}") from exc

    def merge_datasets(
        self,
        dataset_paths: Iterable[str | Path],
        output_dir: str | Path,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> Path:
        """Merge YOLO image/label directories and create train/val/test splits."""
        output_dir = resolve_path(output_dir)
        dataset_paths = [resolve_path(path) for path in dataset_paths]
        if not dataset_paths:
            raise ValueError("No dataset paths were provided for merging.")
        self._validate_splits(train_split, val_split, test_split)

        for dataset_path in dataset_paths:
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
            if dataset_path.resolve() == output_dir.resolve():
                raise ValueError("Refusing to merge an output directory into itself.")

        pairs: list[ImageLabelPair] = []
        for dataset_path in dataset_paths:
            dataset_pairs = self._collect_yolo_pairs(dataset_path)
            self.logger.info("Found %d image/label pairs in %s", len(dataset_pairs), dataset_path)
            pairs.extend(dataset_pairs)

        if not pairs:
            raise RuntimeError("No image/label pairs were found in the provided datasets.")

        random.Random(self.random_seed).shuffle(pairs)
        split_map = self._split_pairs(pairs, train_split, val_split)
        self._reset_output_split_dirs(output_dir)

        for split_name, split_pairs in split_map.items():
            image_out = output_dir / "images" / split_name
            label_out = output_dir / "labels" / split_name
            image_out.mkdir(parents=True, exist_ok=True)
            label_out.mkdir(parents=True, exist_ok=True)

            for index, pair in enumerate(tqdm(split_pairs, desc=f"Merging {split_name}", unit="file")):
                base_name = self._merged_file_stem(pair, index)
                target_image = image_out / f"{base_name}{pair.image_path.suffix.lower()}"
                target_label = label_out / f"{base_name}.txt"
                shutil.copy2(pair.image_path, target_image)
                shutil.copy2(pair.label_path, target_label)

        data_yaml_path = output_dir / "data.yaml"
        data_yaml = {
            "path": str(output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(self.classes),
            "names": {index: name for index, name in enumerate(self.classes)},
        }
        with data_yaml_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data_yaml, stream, sort_keys=False)

        self.logger.info("Merged %d images into %s", len(pairs), output_dir)
        return data_yaml_path

    def validate_dataset(self, data_yaml_path: str | Path) -> bool:
        """Validate a YOLO data.yaml and print split counts plus mismatches."""
        data_yaml_path = resolve_path(data_yaml_path)
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found: {data_yaml_path}")

        data = self._load_yaml(data_yaml_path)
        rows = []
        all_valid = True

        for split_name in ("train", "val", "test"):
            if split_name not in data:
                rows.append((split_name, 0, 0, "missing split path"))
                all_valid = False
                continue

            image_dir = self._resolve_yaml_path(data_yaml_path, data, data[split_name])
            label_dir = self._label_dir_for_image_dir(image_dir)
            images = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES) if image_dir.exists() else []
            labels = sorted(label_dir.rglob("*.txt")) if label_dir.exists() else []
            label_stems = {path.relative_to(label_dir).with_suffix("") for path in labels}
            image_stems = {path.relative_to(image_dir).with_suffix("") for path in images}
            missing_labels = sorted(image_stems - label_stems)
            orphan_labels = sorted(label_stems - image_stems)

            status = "ok"
            if not image_dir.exists():
                status = f"missing image dir: {image_dir}"
            elif not label_dir.exists():
                status = f"missing label dir: {label_dir}"
            elif missing_labels or orphan_labels:
                status = f"{len(missing_labels)} missing labels, {len(orphan_labels)} orphan labels"
            rows.append((split_name, len(images), len(labels), status))

            if status != "ok":
                all_valid = False
                self._print_mismatches(split_name, missing_labels, orphan_labels)

        print()
        print("Dataset validation summary")
        print(f"{'Split':<8} {'Images':>8} {'Labels':>8}  Status")
        print(f"{'-' * 8} {'-' * 8:>8} {'-' * 8:>8}  {'-' * 40}")
        for split_name, image_count, label_count, status in rows:
            print(f"{split_name:<8} {image_count:>8} {label_count:>8}  {status}")

        if all_valid:
            self.logger.info("Dataset validation passed for %s", data_yaml_path)
        else:
            self.logger.warning("Dataset validation found issues in %s", data_yaml_path)
        return all_valid

    def _ensure_coco_val_annotations(self, cache_dir: Path) -> Path:
        annotations_json = cache_dir / "instances_val2017.json"
        if annotations_json.exists():
            return annotations_json

        archive_path = cache_dir / "annotations_trainval2017.zip"
        if not archive_path.exists():
            self._download_file(COCO_ANNOTATIONS_URL, archive_path)

        self.logger.info("Extracting COCO validation annotations")
        with zipfile.ZipFile(archive_path) as archive:
            member = "annotations/instances_val2017.json"
            if member not in archive.namelist():
                raise RuntimeError(f"{member} was not found in {archive_path}")
            with archive.open(member) as source, annotations_json.open("wb") as target:
                shutil.copyfileobj(source, target)
        return annotations_json

    def _download_file(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info("Downloading %s", url)
        with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=destination.name) as progress:
            last_count = 0

            def reporthook(block_count: int, block_size: int, total_size: int) -> None:
                nonlocal last_count
                if total_size > 0:
                    progress.total = total_size
                downloaded = block_count * block_size
                progress.update(max(0, downloaded - last_count))
                last_count = downloaded

            urllib.request.urlretrieve(url, destination, reporthook=reporthook)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML object in {path}")
        return data

    @staticmethod
    def _group_annotations(annotations: Iterable[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
        grouped: dict[int, list[dict[str, Any]]] = {}
        for annotation in annotations:
            grouped.setdefault(int(annotation["image_id"]), []).append(annotation)
        return grouped

    def _coco_phone_category_ids(self, coco: dict[str, Any]) -> set[int]:
        phone_ids = {
            int(category["id"])
            for category in coco.get("categories", [])
            if self._normalized_coco_category(category.get("name", "")) == "phone"
        }
        if not phone_ids:
            self.logger.warning(
                "Could not find a COCO category named cell phone. Falling back to class id %d.",
                COCO_PHONE_YOLO_CLASS_INDEX,
            )
            phone_ids.add(COCO_PHONE_YOLO_CLASS_INDEX)
        return phone_ids

    def _coco_category_to_output_class(self, coco: dict[str, Any]) -> dict[int, int]:
        category_map: dict[int, int] = {}
        for category in coco.get("categories", []):
            output_name = self._normalized_coco_category(category.get("name", ""))
            if output_name in self.class_to_index:
                category_map[int(category["id"])] = self.class_to_index[output_name]
        if "phone" in self.class_to_index and not any(value == self.class_to_index["phone"] for value in category_map.values()):
            category_map[COCO_PHONE_YOLO_CLASS_INDEX] = self.class_to_index["phone"]
        return category_map

    @staticmethod
    def _normalized_coco_category(name: str) -> str:
        normalized = name.strip().lower().replace("_", " ")
        if normalized in {"cell phone", "mobile phone", "phone"}:
            return "phone"
        if normalized == "person":
            return "person"
        return normalized

    def _coco_yolo_label_lines(
        self,
        image: dict[str, Any],
        annotations: Iterable[dict[str, Any]],
        category_to_output_class: dict[int, int],
    ) -> list[str]:
        width = float(image["width"])
        height = float(image["height"])
        label_lines = []
        for annotation in annotations:
            if annotation.get("iscrowd", 0):
                continue
            output_class = category_to_output_class.get(int(annotation["category_id"]))
            if output_class is None:
                continue
            yolo_bbox = self._coco_bbox_to_yolo(annotation.get("bbox", []), width, height)
            if yolo_bbox is None:
                continue
            label_lines.append(f"{output_class} " + " ".join(f"{value:.6f}" for value in yolo_bbox))
        return label_lines

    @staticmethod
    def _coco_bbox_to_yolo(bbox: list[float], image_width: float, image_height: float) -> tuple[float, float, float, float] | None:
        if len(bbox) != 4 or image_width <= 0 or image_height <= 0:
            return None
        x, y, width, height = [float(value) for value in bbox]
        if width <= 0 or height <= 0:
            return None
        center_x = (x + width / 2.0) / image_width
        center_y = (y + height / 2.0) / image_height
        norm_width = width / image_width
        norm_height = height / image_height
        values = (center_x, center_y, norm_width, norm_height)
        return tuple(min(1.0, max(0.0, value)) for value in values)

    def _collect_yolo_pairs(self, dataset_dir: Path) -> list[ImageLabelPair]:
        pairs = []
        image_label_roots = self._find_image_label_roots(dataset_dir)
        for image_root, label_root in image_label_roots:
            for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
                label_path = label_root / image_path.relative_to(image_root).with_suffix(".txt")
                if not label_path.exists():
                    self.logger.warning("Skipping image without label: %s", image_path)
                    continue
                pairs.append(ImageLabelPair(image_path=image_path, label_path=label_path, dataset_name=dataset_dir.name))
        return pairs

    @staticmethod
    def _find_image_label_roots(dataset_dir: Path) -> list[tuple[Path, Path]]:
        roots = []
        direct_images = dataset_dir / "images"
        direct_labels = dataset_dir / "labels"
        if direct_images.is_dir() and direct_labels.is_dir():
            roots.append((direct_images, direct_labels))

        for split_name in ("train", "valid", "val", "test"):
            split_dir = dataset_dir / split_name
            image_dir = split_dir / "images"
            label_dir = split_dir / "labels"
            if image_dir.is_dir() and label_dir.is_dir():
                roots.append((image_dir, label_dir))
        return roots

    @staticmethod
    def _validate_splits(train_split: float, val_split: float, test_split: float) -> None:
        values = (train_split, val_split, test_split)
        if any(value < 0 for value in values):
            raise ValueError("Dataset split values must be non-negative.")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("Dataset splits must sum to 1.0.")

    @staticmethod
    def _split_pairs(pairs: list[ImageLabelPair], train_split: float, val_split: float) -> dict[str, list[ImageLabelPair]]:
        total = len(pairs)
        train_end = int(total * train_split)
        val_end = train_end + int(total * val_split)
        return {
            "train": pairs[:train_end],
            "val": pairs[train_end:val_end],
            "test": pairs[val_end:],
        }

    @staticmethod
    def _reset_output_split_dirs(output_dir: Path) -> None:
        for child in (output_dir / "images", output_dir / "labels"):
            if child.exists():
                shutil.rmtree(child)
        output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _merged_file_stem(pair: ImageLabelPair, index: int) -> str:
        safe_dataset = "".join(character if character.isalnum() else "_" for character in pair.dataset_name)
        safe_stem = "".join(character if character.isalnum() else "_" for character in pair.image_path.stem)
        return f"{safe_dataset}_{index:06d}_{safe_stem}"

    @staticmethod
    def _resolve_yaml_path(data_yaml_path: Path, data: dict[str, Any], split_value: str) -> Path:
        split_path = Path(split_value)
        if split_path.is_absolute():
            return split_path

        root = Path(data.get("path", data_yaml_path.parent))
        if not root.is_absolute():
            root = data_yaml_path.parent / root
        return root / split_path

    @staticmethod
    def _label_dir_for_image_dir(image_dir: Path) -> Path:
        parts = list(image_dir.parts)
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "images":
                parts[index] = "labels"
                return Path(*parts)
        return image_dir.parent / "labels"

    @staticmethod
    def _print_mismatches(split_name: str, missing_labels: list[Path], orphan_labels: list[Path], limit: int = 20) -> None:
        if missing_labels:
            print(f"\n{split_name}: images missing label files")
            for path in missing_labels[:limit]:
                print(f"  - {path}")
            if len(missing_labels) > limit:
                print(f"  ... {len(missing_labels) - limit} more")
        if orphan_labels:
            print(f"\n{split_name}: label files without matching images")
            for path in orphan_labels[:limit]:
                print(f"  - {path}")
            if len(orphan_labels) > limit:
                print(f"  ... {len(orphan_labels) - limit} more")


class AugmentationPipeline:
    """Albumentations pipeline tuned for classroom and office phone detection."""

    def __init__(self, img_size: int = 640, augment_factor: int = 3):
        if img_size <= 0:
            raise ValueError("img_size must be greater than zero.")
        if augment_factor <= 0:
            raise ValueError("augment_factor must be greater than zero.")

        try:
            import albumentations as A
        except ImportError as exc:
            raise RuntimeError("albumentations is not installed. Run setup_env.sh or install requirements.txt.") from exc

        self.A = A
        self.img_size = int(img_size)
        self.augment_factor = int(augment_factor)
        self.logger = logging.getLogger("phonewatch.augmentation")
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
        self.transform = self._build_transform()

    def augment_dataset(self, input_dir: str | Path, output_dir: str | Path) -> Path:
        """Apply augmentations to every YOLO image/label pair in input_dir."""
        input_dir = resolve_path(input_dir)
        output_dir = resolve_path(output_dir)
        image_dir = input_dir / "images"
        label_dir = input_dir / "labels"
        output_image_dir = output_dir / "images"
        output_label_dir = output_dir / "labels"

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Input image directory not found: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Input label directory not found: {label_dir}")

        output_image_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)
        image_paths = sorted(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        if not image_paths:
            raise RuntimeError(f"No images found in {image_dir}")

        written = 0
        skipped = 0
        for image_path in tqdm(image_paths, desc="Augmenting images", unit="image"):
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if not label_path.exists():
                self.logger.warning("Skipping image without label: %s", image_path)
                skipped += 1
                continue

            try:
                image = self._read_image(image_path)
                bboxes, class_labels = self._read_yolo_labels(label_path)
                if not bboxes:
                    self.logger.warning("Skipping image with no labels: %s", image_path)
                    skipped += 1
                    continue

                for index in range(self.augment_factor):
                    augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                    aug_bboxes = list(augmented["bboxes"])
                    aug_labels = list(augmented["class_labels"])
                    if not aug_bboxes:
                        self.logger.info("Skipping augmentation with no preserved boxes: %s #%d", image_path, index + 1)
                        skipped += 1
                        continue

                    relative_parent = image_path.relative_to(image_dir).parent
                    target_stem = f"{image_path.stem}_aug_{index + 1:02d}"
                    target_image = output_image_dir / relative_parent / f"{target_stem}{image_path.suffix.lower()}"
                    target_label = output_label_dir / relative_parent / f"{target_stem}.txt"
                    target_image.parent.mkdir(parents=True, exist_ok=True)
                    target_label.parent.mkdir(parents=True, exist_ok=True)
                    self._write_image(target_image, augmented["image"])
                    self._write_yolo_labels(target_label, aug_bboxes, aug_labels)
                    written += 1
            except Exception as exc:
                self.logger.exception("Failed to augment %s: %s", image_path, exc)
                skipped += 1

        self.logger.info("Augmentation complete: %d files written, %d skipped", written, skipped)
        return output_dir

    def visualize_augmentations(self, image_path: str | Path, n_samples: int = 6, show: bool = True) -> Path:
        """Create a preview grid with the original image and augmented samples."""
        if n_samples <= 0:
            raise ValueError("n_samples must be greater than zero.")

        import cv2
        import matplotlib.pyplot as plt

        image_path = resolve_path(image_path)
        label_path = self._label_path_for_image(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Label file not found for preview image: {label_path}")

        image = self._read_image(image_path)
        bboxes, class_labels = self._read_yolo_labels(label_path)
        if not bboxes:
            raise RuntimeError(f"No YOLO labels found for preview image: {label_path}")

        preview_items = [("Original", image, bboxes, class_labels)]
        for index in range(n_samples):
            augmented = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            preview_items.append(
                (
                    f"Aug {index + 1}",
                    augmented["image"],
                    list(augmented["bboxes"]),
                    list(augmented["class_labels"]),
                )
            )

        columns = 3
        rows = math.ceil(len(preview_items) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 5, rows * 4))
        axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

        for axis, (title, sample_image, sample_bboxes, sample_labels) in zip(axes_flat, preview_items):
            axis.imshow(self._draw_bboxes(sample_image.copy(), sample_bboxes, sample_labels))
            axis.set_title(title)
            axis.axis("off")

        for axis in axes_flat[len(preview_items) :]:
            axis.axis("off")

        output_path = image_path.parent / "augmentation_preview.jpg"
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        self.logger.info("Saved augmentation preview to %s", output_path)

        if show:
            preview = cv2.imread(str(output_path))
            if preview is not None:
                try:
                    cv2.imshow("PhoneWatch Augmentation Preview", preview)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                except cv2.error as exc:
                    self.logger.warning("Could not open preview window. Saved preview instead: %s", exc)
        return output_path

    def _build_transform(self):
        A = self.A
        return A.Compose(
            [
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
                A.HueSaturationValue(p=0.4),
                A.GaussNoise(p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.RandomShadow(p=0.2),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
                self._bbox_safe_crop_transform(),
                self._coarse_dropout_transform(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
        )

    def _bbox_safe_crop_transform(self):
        A = self.A
        if hasattr(A, "RandomSizedBBoxSafeCrop"):
            return A.RandomSizedBBoxSafeCrop(height=self.img_size, width=self.img_size, p=0.3)
        return A.RandomCrop(height=self.img_size, width=self.img_size, p=0.3)

    def _coarse_dropout_transform(self):
        A = self.A
        try:
            return A.CoarseDropout(max_holes=4, max_height=40, max_width=40, p=0.3)
        except TypeError:
            return A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 40), hole_width_range=(8, 40), p=0.3)

    @staticmethod
    def _read_image(image_path: Path):
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _write_image(image_path: Path, image) -> None:
        import cv2

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(image_path), image_bgr):
            raise RuntimeError(f"Could not write image: {image_path}")

    @staticmethod
    def _read_yolo_labels(label_path: Path) -> tuple[list[tuple[float, float, float, float]], list[int]]:
        bboxes: list[tuple[float, float, float, float]] = []
        class_labels: list[int] = []
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {raw_line}")
            class_label = int(float(parts[0]))
            bbox = tuple(float(value) for value in parts[1:5])
            if any(value < 0.0 or value > 1.0 for value in bbox):
                raise ValueError(f"YOLO bbox values must be normalized in {label_path}:{line_number}")
            class_labels.append(class_label)
            bboxes.append(bbox)
        return bboxes, class_labels

    @staticmethod
    def _write_yolo_labels(label_path: Path, bboxes: list[tuple[float, float, float, float]], class_labels: list[int]) -> None:
        lines = [
            f"{int(class_label)} " + " ".join(f"{min(1.0, max(0.0, float(value))):.6f}" for value in bbox)
            for bbox, class_label in zip(bboxes, class_labels)
        ]
        label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _label_path_for_image(image_path: Path) -> Path:
        parts = list(image_path.parts)
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "images":
                parts[index] = "labels"
                return Path(*parts).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    @staticmethod
    def _draw_bboxes(image, bboxes: list[tuple[float, float, float, float]], class_labels: list[int]):
        import cv2

        height, width = image.shape[:2]
        for bbox, class_label in zip(bboxes, class_labels):
            center_x, center_y, box_width, box_height = bbox
            x1 = int((center_x - box_width / 2.0) * width)
            y1 = int((center_y - box_height / 2.0) * height)
            x2 = int((center_x + box_width / 2.0) * width)
            y2 = int((center_y + box_height / 2.0) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 32, 32), 2)
            cv2.putText(image, str(class_label), (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 32, 32), 1)
        return image


def run_full_pipeline(config_path: str | Path = "config.yaml", skip_coco: bool = False, skip_roboflow: bool = False) -> Path:
    """Run configured downloads, merge datasets, and validate the merged output."""
    config = load_config(config_path)
    ensure_directories(config)
    manager = DatasetManager(config=config)
    dataset_cfg = config["dataset"]
    sources_cfg = config.get("dataset_sources", {})
    dataset_paths: list[Path] = []

    if not skip_coco and sources_cfg.get("coco", {}).get("enabled", True):
        coco_output = sources_cfg.get("coco", {}).get("output_dir", f"{dataset_cfg['raw']}/coco_phone")
        dataset_paths.append(manager.download_coco_phone_class(coco_output))

    roboflow_cfg = sources_cfg.get("roboflow", {})
    if not skip_roboflow and roboflow_cfg.get("enabled", False):
        required_keys = ("api_key", "workspace", "project", "version")
        missing = [key for key in required_keys if not roboflow_cfg.get(key)]
        if missing:
            raise ValueError(f"Roboflow source is enabled but missing config keys: {', '.join(missing)}")
        roboflow_output = roboflow_cfg.get("output_dir", f"{dataset_cfg['raw']}/roboflow")
        dataset_paths.append(
            manager.download_roboflow_dataset(
                api_key=roboflow_cfg["api_key"],
                workspace=roboflow_cfg["workspace"],
                project=roboflow_cfg["project"],
                version=roboflow_cfg["version"],
                output_dir=roboflow_output,
            )
        )

    for local_path in sources_cfg.get("local", []):
        dataset_paths.append(resolve_path(local_path))

    split_cfg = sources_cfg.get("splits", {})
    data_yaml_path = manager.merge_datasets(
        dataset_paths=dataset_paths,
        output_dir=dataset_cfg["processed"],
        train_split=float(split_cfg.get("train", 0.8)),
        val_split=float(split_cfg.get("val", 0.1)),
        test_split=float(split_cfg.get("test", 0.1)),
    )
    if not manager.validate_dataset(data_yaml_path):
        raise RuntimeError("Merged dataset validation failed.")
    return data_yaml_path


def run_augmentation_demo(
    config_path: str | Path = "config.yaml",
    image_path: str | Path | None = None,
    n_samples: int = 6,
    img_size: int | None = None,
    augment_factor: int = 3,
    show: bool = True,
) -> Path:
    """Create and optionally display an augmentation preview for one labeled sample image."""
    config = load_config(config_path)
    dataset_cfg = config["dataset"]
    image_path = resolve_path(image_path) if image_path else _find_first_labeled_image(dataset_cfg["processed"])
    pipeline = AugmentationPipeline(img_size=img_size or int(config["model"].get("img_size", 640)), augment_factor=augment_factor)
    return pipeline.visualize_augmentations(image_path, n_samples=n_samples, show=show)


def _find_first_labeled_image(dataset_dir: str | Path) -> Path:
    image_root = resolve_path(dataset_dir) / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image directory not found for augmentation demo: {image_root}")

    for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
        if AugmentationPipeline._label_path_for_image(image_path).exists():
            return image_path
    raise RuntimeError(f"No image with a matching YOLO label was found under {image_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download, merge, and validate PhoneWatch datasets.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--skip-coco", action="store_true", help="Skip COCO download")
    parser.add_argument("--skip-roboflow", action="store_true", help="Skip Roboflow download")
    parser.add_argument("--demo-augmentation", action="store_true", help="Create a preview grid for one labeled image")
    parser.add_argument("--image", help="Image path for --demo-augmentation")
    parser.add_argument("--samples", type=int, default=6, help="Number of augmented samples to show in the preview grid")
    parser.add_argument("--img-size", type=int, help="Augmentation output image size")
    parser.add_argument("--augment-factor", type=int, default=3, help="Augmentations per source image for AugmentationPipeline")
    parser.add_argument("--no-show", action="store_true", help="Save the augmentation preview without opening a window")
    args = parser.parse_args()

    try:
        if args.demo_augmentation:
            preview_path = run_augmentation_demo(
                config_path=args.config,
                image_path=args.image,
                n_samples=args.samples,
                img_size=args.img_size,
                augment_factor=args.augment_factor,
                show=not args.no_show,
            )
            print(f"\nAugmentation preview saved: {preview_path}")
            return 0

        data_yaml_path = run_full_pipeline(args.config, skip_coco=args.skip_coco, skip_roboflow=args.skip_roboflow)
        print(f"\nDataset pipeline complete: {data_yaml_path}")
        return 0
    except Exception as exc:
        logging.getLogger("phonewatch.dataset").exception("Dataset pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
