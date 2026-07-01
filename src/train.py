"""Training, evaluation, and export entry points for PhoneWatch."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import Any

try:
    from .utils import ModelDiagnostics, ensure_directories, load_config, resolve_path, setup_logging
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from src.utils import ModelDiagnostics, ensure_directories, load_config, resolve_path, setup_logging


class PhoneWatchTrainer:
    """Fine-tune YOLOv8 for phone usage detection."""

    def __init__(self, config_path: str | Path = "config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path)
        ensure_directories(self.config)
        setup_logging(self.config)

        self.training_dir = resolve_path("logs/training")
        self.checkpoint_dir = resolve_path("models/checkpoints")
        self.export_dir = resolve_path("models")
        self.training_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("phonewatch.trainer")
        self.model = None
        self.wandb_run = self._init_wandb()

    def setup_model(self):
        """Load pretrained YOLOv8n weights and print a model summary."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run setup_env.sh or install requirements.txt.") from exc

        model_size = self.config["model"].get("model_size", "yolov8n")
        model_source = f"{model_size}.pt"
        self.logger.info("Loading pretrained model: %s", model_source)
        self.model = YOLO(model_source)

        print("\nModel architecture summary")
        try:
            self.model.model.info(verbose=True)
        except Exception as exc:
            self.logger.warning("Could not print full Ultralytics model summary: %s", exc)
            print(self.model.model)

        parameter_count = self._parameter_count(self.model)
        print(f"Parameter count: {parameter_count:,}")
        return self.model

    def train(self, data_yaml_path: str | Path):
        """Run YOLOv8 fine-tuning and save the best PhoneWatch checkpoint."""
        data_yaml_path = resolve_path(data_yaml_path)
        if not data_yaml_path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")

        if self.model is None:
            self.setup_model()

        torch = self._import_torch()
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.model.add_callback("on_fit_epoch_end", self._training_progress_callback)
        training_cfg = self.config["training"]
        model_cfg = self.config["model"]

        print(f"Starting training on {device}: {data_yaml_path}")
        results = self.model.train(
            data=str(data_yaml_path),
            epochs=int(training_cfg["epochs"]),
            imgsz=int(model_cfg["img_size"]),
            batch=int(training_cfg["batch_size"]),
            lr0=float(training_cfg["learning_rate"]),
            patience=15,
            save=True,
            plots=True,
            val=True,
            device=device,
            project=str(self.training_dir),
            name="phonewatch_v1",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            cos_lr=True,
            augment=True,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )

        best_source = self._best_checkpoint_path()
        best_target = self.checkpoint_dir / "phonewatch_best.pt"
        if not best_source.exists():
            raise FileNotFoundError(f"Training completed but best checkpoint was not found: {best_source}")
        shutil.copy2(best_source, best_target)
        print(f"Best checkpoint copied to: {best_target}")

        metrics = self._extract_metrics(results)
        if not metrics:
            try:
                metrics = self._extract_metrics(self.model.val(data=str(data_yaml_path), split="val", device=device, verbose=False))
            except Exception as exc:
                self.logger.warning("Could not run final validation for metrics: %s", exc)

        print("\nFinal training metrics")
        self._print_metrics_table(metrics)
        self._run_post_training_diagnostics(best_target)
        return metrics

    def evaluate(self, model_path: str | Path, data_yaml_path: str | Path) -> dict[str, float]:
        """Evaluate a trained model on the test split and save a confusion matrix."""
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

        torch = self._import_torch()
        device = "mps" if torch.backends.mps.is_available() else "cpu"

        model = YOLO(str(model_path))
        evaluation_dir = resolve_path("logs/evaluation")
        results = model.val(
            data=str(data_yaml_path),
            split="test",
            plots=True,
            device=device,
            project=str(evaluation_dir),
            name="phonewatch_test",
            exist_ok=True,
        )

        confusion_target = resolve_path("logs/confusion_matrix.png")
        self._copy_confusion_matrix(evaluation_dir / "phonewatch_test", confusion_target)
        metrics = self._extract_metrics(results)

        print("\nEvaluation metrics")
        self._print_metrics_table(metrics)
        if confusion_target.exists():
            print(f"Confusion matrix saved to: {confusion_target}")
        return metrics

    def export_model(self, model_path: str | Path, format: str = "onnx") -> Path:
        """Export the trained model for deployment."""
        model_path = resolve_path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed. Run setup_env.sh or install requirements.txt.") from exc

        model = YOLO(str(model_path))
        exported_path = Path(model.export(format=format))
        target_path = self.export_dir / f"phonewatch.{format}"
        if exported_path.exists() and exported_path.resolve() != target_path.resolve():
            shutil.copy2(exported_path, target_path)
        elif exported_path.exists():
            target_path = exported_path
        else:
            raise FileNotFoundError(f"Ultralytics export did not produce an output file: {exported_path}")

        print(f"Exported model saved to: {target_path}")
        return target_path

    def _init_wandb(self):
        try:
            import wandb
        except ImportError:
            print("wandb not installed; skipping Weights & Biases logging.")
            return None

        if not os.getenv("WANDB_API_KEY") and not os.getenv("WANDB_MODE"):
            print("wandb available but not configured; skipping Weights & Biases logging.")
            return None

        try:
            run = wandb.init(project="PhoneWatch", name="phonewatch_v1", config=self.config)
            print("Weights & Biases logging initialized.")
            return run
        except Exception as exc:
            print(f"wandb initialization skipped: {exc}")
            return None

    @staticmethod
    def _import_torch():
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not installed. Run setup_env.sh or install requirements.txt.") from exc
        return torch

    @staticmethod
    def _parameter_count(model) -> int:
        try:
            return sum(parameter.numel() for parameter in model.model.parameters())
        except Exception:
            return 0

    def _training_progress_callback(self, trainer) -> None:
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        epochs = int(getattr(trainer, "epochs", 0) or self.config["training"]["epochs"])
        loss = self._trainer_loss(trainer)
        map50 = self._lookup_metric(getattr(trainer, "metrics", {}), ["metrics/mAP50(B)", "mAP50", "map50"])
        lr = self._trainer_lr(trainer)
        print(f"Epoch {epoch:03d}/{epochs:03d} | loss: {loss:.4f} | mAP50: {map50:.4f} | lr: {lr:.6g}")

    @staticmethod
    def _trainer_loss(trainer) -> float:
        loss_source = getattr(trainer, "tloss", None)
        if loss_source is None:
            loss_source = getattr(trainer, "loss_items", None)
        return PhoneWatchTrainer._to_float(loss_source)

    @staticmethod
    def _trainer_lr(trainer) -> float:
        lr = getattr(trainer, "lr", None)
        if isinstance(lr, dict) and lr:
            return PhoneWatchTrainer._to_float(next(iter(lr.values())))
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None and getattr(optimizer, "param_groups", None):
            return PhoneWatchTrainer._to_float(optimizer.param_groups[0].get("lr", 0.0))
        return 0.0

    @staticmethod
    def _to_float(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "mean"):
                value = value.mean()
            if hasattr(value, "item"):
                return float(value.item())
            if isinstance(value, (list, tuple)) and value:
                return float(sum(PhoneWatchTrainer._to_float(item) for item in value))
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _lookup_metric(metrics: dict[str, Any], keys: list[str]) -> float:
        if not isinstance(metrics, dict):
            return 0.0
        for key in keys:
            if key in metrics:
                return PhoneWatchTrainer._to_float(metrics[key])
        return 0.0

    def _extract_metrics(self, results) -> dict[str, float]:
        if results is None:
            return {}

        metrics_source = getattr(results, "results_dict", None)
        if metrics_source is None:
            metrics_source = getattr(results, "metrics", None)
        if not isinstance(metrics_source, dict):
            metrics_source = {}

        return {
            "mAP@50": self._lookup_metric(metrics_source, ["metrics/mAP50(B)", "mAP50", "map50"]),
            "mAP@50-95": self._lookup_metric(metrics_source, ["metrics/mAP50-95(B)", "mAP50-95", "map"]),
            "precision": self._lookup_metric(metrics_source, ["metrics/precision(B)", "precision"]),
            "recall": self._lookup_metric(metrics_source, ["metrics/recall(B)", "recall"]),
        }

    @staticmethod
    def _print_metrics_table(metrics: dict[str, float]) -> None:
        if not metrics:
            print("No metrics available.")
            return
        print(f"{'Metric':<14} {'Value':>10}")
        print(f"{'-' * 14} {'-' * 10}")
        for key in ("mAP@50", "mAP@50-95", "precision", "recall"):
            print(f"{key:<14} {metrics.get(key, 0.0):>10.4f}")

    def _best_checkpoint_path(self) -> Path:
        trainer = getattr(self.model, "trainer", None)
        best = getattr(trainer, "best", None)
        if best:
            return Path(best)
        return self.training_dir / "phonewatch_v1" / "weights" / "best.pt"

    def _run_post_training_diagnostics(self, model_path: Path) -> None:
        results_csv_path = self.training_dir / "phonewatch_v1" / "results.csv"
        try:
            if results_csv_path.exists():
                ModelDiagnostics.plot_training_curves(results_csv_path)
            else:
                self.logger.warning("Training curves skipped because results.csv was not found: %s", results_csv_path)
        except Exception as exc:
            self.logger.warning("Training curve diagnostics failed: %s", exc)

        try:
            ModelDiagnostics.benchmark_speed(model_path)
        except Exception as exc:
            self.logger.warning("Speed benchmark diagnostics failed: %s", exc)

    @staticmethod
    def _copy_confusion_matrix(source_dir: Path, target_path: Path) -> None:
        candidates = [
            source_dir / "confusion_matrix.png",
            source_dir / "confusion_matrix_normalized.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, target_path)
                return


def train(config_path: str = "config.yaml") -> dict[str, float]:
    """Backward-compatible training helper."""
    trainer = PhoneWatchTrainer(config_path=config_path)
    data_yaml = trainer.config["dataset"]["data_yaml"]
    return trainer.train(data_yaml)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train, evaluate, or export PhoneWatch YOLOv8 models.")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data", default="data/processed/data.yaml", help="YOLO data.yaml path")
    parser.add_argument("--model", default="models/checkpoints/phonewatch_best.pt", help="Model checkpoint path")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained model on the test split")
    parser.add_argument("--export", action="store_true", help="Export a trained model")
    parser.add_argument("--format", default="onnx", help="Export format, default: onnx")
    args = parser.parse_args()

    try:
        trainer = PhoneWatchTrainer(config_path=args.config)
        if args.eval:
            trainer.evaluate(args.model, args.data)
        elif args.export:
            trainer.export_model(args.model, format=args.format)
        else:
            trainer.train(args.data)
        return 0
    except Exception as exc:
        logging.getLogger("phonewatch.trainer").exception("Training command failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
