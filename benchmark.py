"""PhoneWatch system benchmarking: inference, detection metrics, context, e2e latency, memory."""

from __future__ import annotations

import json
import math
import random
import statistics
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.context import PhoneUsageDetector
from src.detect import PhoneWatchEngine
from src.utils import load_config, resolve_path


try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


def _now_iso() -> str:
    from datetime import datetime

    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _resolve_model_path(config: dict) -> Path:
    checkpoint = resolve_path("models/checkpoints/phonewatch_best.pt")
    if checkpoint.exists():
        return checkpoint
    for candidate in (resolve_path("yolov8n.pt"), resolve_path("models/checkpoints/yolov8n.pt")):
        if candidate.exists():
            return candidate
    return resolve_path("yolov8n.pt")


def _memory_mb() -> float | None:
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _sync_torch(device: str) -> None:
    if torch is None:
        return
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


@dataclass
class InferenceRunResult:
    device_label: str
    mean_fps: float
    median_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    peak_memory_mb: float | None
    mean_memory_mb: float | None
    n_passes: int
    error: str | None = None


def run_inference_passes(
    model_path: Path,
    device: str,
    label: str,
    n_passes: int = 500,
    imgsz: int = 640,
    warmup: int = 10,
    target_class_ids: list[int] | None = None,
) -> InferenceRunResult:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        return InferenceRunResult(
            device_label=label,
            mean_fps=0.0,
            median_latency_ms=0.0,
            p95_latency_ms=0.0,
            min_latency_ms=0.0,
            max_latency_ms=0.0,
            peak_memory_mb=None,
            mean_memory_mb=None,
            n_passes=n_passes,
            error=str(exc),
        )

    model = YOLO(str(model_path))
    latencies: list[float] = []
    mem_samples: list[float] = []
    peak_mem = _memory_mb()

    rng = np.random.default_rng(42)
    for _ in range(warmup):
        frame = rng.integers(0, 256, size=(imgsz, imgsz, 3), dtype=np.uint8)
        model.predict(
            frame,
            imgsz=imgsz,
            conf=0.25,
            iou=0.45,
            classes=target_class_ids,
            device=device,
            verbose=False,
        )
        _sync_torch(device)

    for i in range(n_passes):
        frame = rng.integers(0, 256, size=(imgsz, imgsz, 3), dtype=np.uint8)
        m0 = _memory_mb()
        t0 = time.perf_counter()
        model.predict(
            frame,
            imgsz=imgsz,
            conf=0.25,
            iou=0.45,
            classes=target_class_ids,
            device=device,
            verbose=False,
        )
        _sync_torch(device)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        m1 = _memory_mb()
        if m0 is not None and m1 is not None:
            mem_samples.append(max(m0, m1))
            peak_mem = max(peak_mem or 0.0, m1)
        elif m1 is not None:
            peak_mem = max(peak_mem or 0.0, m1)

    lat_ms = [x * 1000.0 for x in latencies]
    lat_ms.sort()
    mean_s = statistics.mean(latencies)
    mean_fps = 1.0 / mean_s if mean_s > 0 else 0.0
    return InferenceRunResult(
        device_label=label,
        mean_fps=mean_fps,
        median_latency_ms=_percentile(lat_ms, 50),
        p95_latency_ms=_percentile(lat_ms, 95),
        min_latency_ms=min(lat_ms),
        max_latency_ms=max(lat_ms),
        peak_memory_mb=peak_mem,
        mean_memory_mb=statistics.mean(mem_samples) if mem_samples else None,
        n_passes=n_passes,
        error=None,
    )


def _target_class_ids_from_model(model) -> list[int] | None:
    names = getattr(model, "names", {})
    if isinstance(names, list):
        id_map = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        id_map = {int(k): str(v) for k, v in names.items()}
    else:
        id_map = {}
    target = []
    for cid, raw in id_map.items():
        n = str(raw).strip().lower()
        if n in {"phone", "person", "cell phone"}:
            target.append(cid)
    return sorted(set(target)) if target else None


def benchmark_inference_speed(model_path: Path, config: dict) -> dict[str, Any]:
    from ultralytics import YOLO

    probe = YOLO(str(model_path))
    tids = _target_class_ids_from_model(probe)
    rows: list[dict[str, Any]] = []

    cpu_res = run_inference_passes(model_path, "cpu", "CPU", target_class_ids=tids)
    rows.append(asdict(cpu_res))

    gpu_label = None
    gpu_device = None
    if torch is not None:
        if torch.cuda.is_available():
            gpu_device = "cuda:0"
            gpu_label = "GPU (CUDA)"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            gpu_device = "mps"
            gpu_label = "GPU (MPS)"

    if gpu_device:
        gpu_res = run_inference_passes(model_path, gpu_device, gpu_label or gpu_device, target_class_ids=tids)
        rows.append(asdict(gpu_res))
    else:
        rows.append(
            {
                "device_label": "GPU",
                "mean_fps": None,
                "median_latency_ms": None,
                "p95_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "peak_memory_mb": None,
                "mean_memory_mb": None,
                "n_passes": 0,
                "error": "No CUDA/MPS GPU available",
            }
        )

    onnx_path = resolve_path("logs/benchmark_model.onnx")
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_error = None
    if not onnx_path.exists():
        try:
            import shutil

            src_onnx = model_path.with_suffix(".onnx")
            if not src_onnx.exists():
                probe.export(format="onnx", imgsz=640, simplify=True)
                src_onnx = model_path.with_suffix(".onnx")
            if src_onnx.exists():
                shutil.copy2(src_onnx, onnx_path)
        except Exception as exc:
            onnx_error = str(exc)
    if onnx_path.exists() and onnx_path.stat().st_size > 0 and onnx_error is None:
        try:
            onnx_model = YOLO(str(onnx_path), task="detect")
            oids = _target_class_ids_from_model(onnx_model)
            onnx_res = run_inference_passes(onnx_path, "cpu", "ONNX (CPU)", target_class_ids=oids)
            rows.append(asdict(onnx_res))
        except Exception as exc:
            rows.append(
                {
                    "device_label": "ONNX (CPU)",
                    "mean_fps": None,
                    "median_latency_ms": None,
                    "p95_latency_ms": None,
                    "min_latency_ms": None,
                    "max_latency_ms": None,
                    "peak_memory_mb": None,
                    "mean_memory_mb": None,
                    "n_passes": 0,
                    "error": str(exc),
                }
            )
    else:
        rows.append(
            {
                "device_label": "ONNX (CPU)",
                "mean_fps": None,
                "median_latency_ms": None,
                "p95_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "peak_memory_mb": None,
                "mean_memory_mb": None,
                "n_passes": 0,
                "error": onnx_error or "ONNX export or file missing",
            }
        )

    print("\n" + "=" * 72)
    print("1. INFERENCE SPEED BENCHMARK (500 passes, 640×640 random frames)")
    print("=" * 72)
    hdr = f"{'Device':<18} {'Mean FPS':>10} {'Med(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'Peak MB':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("error"):
            print(f"{str(r.get('device_label')):<18} ERROR: {r['error'][:48]}")
            continue
        pm = r.get("peak_memory_mb")
        pm_s = f"{pm:.1f}" if pm is not None else "n/a"
        print(
            f"{str(r['device_label']):<18} {r['mean_fps']:>10.2f} {r['median_latency_ms']:>10.2f} "
            f"{r['p95_latency_ms']:>10.2f} {r['min_latency_ms']:>10.2f} {r['max_latency_ms']:>10.2f} {pm_s:>10}"
        )
    if psutil is None:
        print("(Install psutil for RSS memory columns.)")

    return {"runs": rows, "imgsz": 640, "passes": 500}


def _find_test_data_yaml() -> Path | None:
    data_yaml = resolve_path("data/processed/data.yaml")
    if not data_yaml.exists():
        return None
    import yaml

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = data_yaml.parent / root
    test_rel = data.get("test")
    if not test_rel:
        return None
    test_img = root / test_rel
    if not test_img.exists():
        alt = resolve_path("data/processed/test")
        if (alt / "images").is_dir():
            return data_yaml
        if any(alt.glob("*.jpg")) or any(alt.glob("*.png")):
            return data_yaml
        return None
    n = len(list(test_img.glob("*.jpg"))) + len(list(test_img.glob("*.png")))
    if n == 0:
        return None
    return data_yaml


def _extract_global_metrics(results) -> dict[str, float]:
    metrics_source = getattr(results, "results_dict", None) or getattr(results, "metrics", None)
    if not isinstance(metrics_source, dict):
        metrics_source = {}

    def lookup(keys: list[str]) -> float:
        for k in keys:
            if k in metrics_source:
                try:
                    return float(metrics_source[k])
                except (TypeError, ValueError):
                    continue
        return 0.0

    return {
        "mAP@50": lookup(["metrics/mAP50(B)", "mAP50(B)", "mAP50", "map50"]),
        "mAP@50-95": lookup(["metrics/mAP50-95(B)", "mAP50-95(B)", "mAP50-95", "map"]),
        "precision@0.5": lookup(["metrics/precision(B)", "precision(B)", "precision"]),
        "recall@0.5": lookup(["metrics/recall(B)", "recall(B)", "recall"]),
    }


def _per_class_maps(results, model) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    box = getattr(results, "box", None)
    if box is None:
        return out

    raw_names = getattr(model, "names", None) or {}
    if isinstance(raw_names, dict):
        idx_to_name = {int(k): str(v) for k, v in raw_names.items()}
    else:
        idx_to_name = {i: str(n) for i, n in enumerate(raw_names)}

    def to_list(x):
        if x is None:
            return []
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        if hasattr(x, "flatten"):
            x = x.flatten()
        if hasattr(x, "tolist"):
            x = x.tolist()
        return list(x) if isinstance(x, (list, tuple)) else []

    ap50_l = to_list(getattr(box, "ap50", None))
    ap_l = to_list(getattr(box, "ap", None)) or to_list(getattr(box, "maps", None))
    p_l = to_list(getattr(box, "p", None))
    r_l = to_list(getattr(box, "r", None))
    nc = max(len(idx_to_name), len(ap50_l), len(ap_l), 1)

    for i in range(nc):
        name = idx_to_name.get(i, str(i))
        key = name.lower()
        if key not in {"phone", "person"}:
            continue
        entry: dict[str, float] = {}
        if i < len(ap50_l):
            entry["mAP@50"] = float(ap50_l[i])
        if i < len(ap_l):
            entry["mAP@50-95"] = float(ap_l[i])
        if i < len(p_l):
            entry["precision@0.5"] = float(p_l[i])
        if i < len(r_l):
            entry["recall@0.5"] = float(r_l[i])
        out[name] = entry

    return out


def benchmark_detection_accuracy(model_path: Path) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("2. DETECTION ACCURACY BENCHMARK (test split)")
    print("=" * 72)

    data_yaml = _find_test_data_yaml()
    if data_yaml is None:
        msg = "No test split found (expected labeled images under data/processed/images/test or data.yaml test: path)."
        print(msg)
        return {"skipped": True, "reason": msg}

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"ultralytics not available: {exc}")
        return {"skipped": True, "reason": str(exc)}

    device = "cpu"
    if torch is not None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"

    model = YOLO(str(model_path))
    val_dir = resolve_path("logs/benchmark_val")
    val_dir.mkdir(parents=True, exist_ok=True)
    try:
        results = model.val(
            data=str(data_yaml),
            split="test",
            device=device,
            project=str(val_dir),
            name="benchmark_test",
            exist_ok=True,
            verbose=False,
            plots=False,
        )
    except Exception as exc:
        print(f"Validation failed: {exc}")
        return {"skipped": True, "reason": str(exc)}

    global_m = _extract_global_metrics(results)
    per_class = _per_class_maps(results, model)

    print(f"{'Metric':<18} {'Value':>12}")
    print("-" * 32)
    for k, v in global_m.items():
        print(f"{k:<18} {v:>12.4f}")

    print("\nPer-class (phone / person):")
    for cls in ("phone", "person"):
        row = per_class.get(cls) or per_class.get(cls.capitalize()) or {}
        if not row:
            print(f"  {cls}: (not available from metrics object)")
            continue
        parts = [f"{kk}={vv:.4f}" for kk, vv in sorted(row.items())]
        print(f"  {cls}: {', '.join(parts)}")

    return {
        "skipped": False,
        "data_yaml": str(data_yaml),
        "device": device,
        "global": global_m,
        "per_class": per_class,
    }


def _scenario_in_hand(seed: int) -> tuple[tuple, list, int, bool, str]:
    random.seed(seed)
    fh = 720
    pw, ph = 100 + seed % 80, 120 + seed % 60
    px1 = 180 + (seed % 120)
    py1 = 140 + (seed % 100)
    phone = (float(px1), float(py1), float(px1 + pw), float(py1 + ph))
    person = (100.0, 80.0, 380.0, 660.0)
    return phone, [person], fh, True, "in_hand"


def _scenario_desk(seed: int) -> tuple[tuple, list, int, bool, str]:
    random.seed(seed + 1000)
    fh = 720
    phone = (550.0 + (seed % 40), 420.0, 620.0, 500.0)
    person = (80.0, 60.0, 220.0, 640.0)
    return phone, [person], fh, False, "on_desk"


def _scenario_ambiguous(index: int) -> tuple[tuple, list, int, bool, str]:
    fh = 720
    person = (200.0, 100.0, 400.0, 650.0)
    if index < 5:
        px1, py1 = 255.0 + index * 3.0, 210.0 + index * 2.0
        phone = (px1, py1, px1 + 48.0, py1 + 72.0)
        expected = True
    else:
        px1, py1 = 270.0 + (index - 5) * 4.0, 560.0 + (index - 5) * 3.0
        phone = (px1, py1, px1 + 50.0, py1 + 75.0)
        expected = False
    return phone, [person], fh, expected, "ambiguous"


def benchmark_context_classifier() -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("3. CONTEXT CLASSIFIER ACCURACY (50 synthetic scenarios, bbox context)")
    print("=" * 72)

    detector = PhoneUsageDetector()
    scenarios: list[tuple[tuple, list, int, bool, str]] = []
    for i in range(20):
        scenarios.append(_scenario_in_hand(i))
    for i in range(20):
        scenarios.append(_scenario_desk(i + 50))
    for i in range(10):
        scenarios.append(_scenario_ambiguous(i))

    tp = fp = tn = fn = 0
    ambiguous_correct = 0
    try:
        for phone, people, fh, expected, tag in scenarios:
            pred = detector.analyze_bounding_box_context(phone, people, fh)["in_use"]
            if pred and expected:
                tp += 1
            elif pred and not expected:
                fp += 1
            elif not pred and not expected:
                tn += 1
            else:
                fn += 1
            if tag == "ambiguous" and pred == expected:
                ambiguous_correct += 1
    finally:
        detector.close()

    n = len(scenarios)
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"Confusion: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"Ambiguous subset correct: {ambiguous_correct}/10")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "ambiguous_correct": ambiguous_correct,
        "n_scenarios": n,
    }


def benchmark_e2e_latency(engine: PhoneWatchEngine, n_samples: int = 50) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("4. END-TO-END LATENCY BENCHMARK (per-stage, mean ms)")
    print("=" * 72)

    rng = np.random.default_rng(7)
    cfg = engine.config["model"]
    conf = float(cfg.get("confidence_threshold", 0.5))
    iou = float(cfg.get("iou_threshold", 0.45))
    imgsz = int(cfg.get("img_size", 640))
    tids = engine.target_class_ids
    dev = engine.usage_detector.device

    cap_t: list[float] = []
    yolo_t: list[float] = []
    ctx_t: list[float] = []
    alert_t: list[float] = []
    total_t: list[float] = []

    for _ in range(n_samples):
        frame = rng.integers(0, 256, size=(720, 1280, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        frame_in = np.ascontiguousarray(frame)
        t_cap = time.perf_counter() - t0
        cap_t.append(t_cap * 1000.0)

        t0 = time.perf_counter()
        results = engine.model(
            frame_in,
            conf=conf,
            iou=iou,
            classes=tids,
            imgsz=imgsz,
            device=dev,
            verbose=False,
        )[0]
        _sync_torch(dev)
        t_y = time.perf_counter() - t0
        yolo_t.append(t_y * 1000.0)

        detections = engine._parse_detections(results)

        t0 = time.perf_counter()
        usage = engine.usage_detector.classify_phone_usage(frame_in, detections)
        t_c = time.perf_counter() - t0
        ctx_t.append(t_c * 1000.0)

        t0 = time.perf_counter()
        ann = engine._draw_live_annotations(frame_in.copy(), detections, usage)
        ann = engine.alert_system.process_frame(ann, usage, frame_count=0)
        ann = engine.alert_system.draw_status_hud(
            ann,
            fps=0.0,
            total_alerts_today=engine.alert_system.total_alerts_today,
            mode=engine.alert_system.mode,
        )
        t_a = time.perf_counter() - t0
        alert_t.append(t_a * 1000.0)

        total_t.append((t_cap + t_y + t_c + t_a) * 1000.0)

        _ = ann

    def mean(xs: list[float]) -> float:
        return statistics.mean(xs) if xs else 0.0

    summary = {
        "capture_sim_ms_mean": mean(cap_t),
        "yolo_inference_ms_mean": mean(yolo_t),
        "context_analysis_ms_mean": mean(ctx_t),
        "alert_overlay_ms_mean": mean(alert_t),
        "total_pipeline_ms_mean": mean(total_t),
        "n_samples": n_samples,
    }

    print(f"{'Stage':<28} {'Mean (ms)':>12}")
    print("-" * 42)
    print(f"{'Frame capture (simulated)':<28} {summary['capture_sim_ms_mean']:>12.3f}")
    print(f"{'YOLO inference':<28} {summary['yolo_inference_ms_mean']:>12.3f}")
    print(f"{'Context analysis':<28} {summary['context_analysis_ms_mean']:>12.3f}")
    print(f"{'Alert overlay + HUD':<28} {summary['alert_overlay_ms_mean']:>12.3f}")
    print(f"{'Total (sum of stages)':<28} {summary['total_pipeline_ms_mean']:>12.3f}")

    return summary


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1e-9
    return num / den


def benchmark_memory_profile(
    model_path: Path,
    duration_s: float = 60.0,
    target_class_ids: list[int] | None = None,
) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print(f"5. MEMORY PROFILE ({duration_s:.0f}s continuous inference)")
    print("=" * 72)

    if psutil is None:
        msg = "psutil not installed; skipping memory profile."
        print(msg)
        return {"skipped": True, "reason": msg}

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    device = "cpu"
    if torch is not None:
        if torch.cuda.is_available():
            device = "cuda:0"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"

    times: list[float] = []
    mems: list[float] = []
    stop = threading.Event()

    def sampler():
        t0 = time.perf_counter()
        while not stop.is_set():
            times.append(time.perf_counter() - t0)
            mems.append(psutil.Process().memory_info().rss / (1024 * 1024))
            time.sleep(0.5)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    rng = np.random.default_rng(99)
    t_end = time.perf_counter() + duration_s
    n = 0
    while time.perf_counter() < t_end:
        frame = rng.integers(0, 256, size=(640, 640, 3), dtype=np.uint8)
        model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            classes=target_class_ids,
            device=device,
            verbose=False,
        )
        _sync_torch(device)
        n += 1

    stop.set()
    th.join(timeout=2.0)

    slope = _linear_slope(times, mems) if len(times) > 2 else 0.0
    leak_suspected = slope > 0.15 and len(mems) > 10

    print(f"Samples: {len(mems)}  Iterations: {n}")
    print(f"RSS min / max: {min(mems):.1f} / {max(mems):.1f} MB")
    print(f"Linear slope (MB/s): {slope:.4f}")
    if leak_suspected:
        print("WARNING: RSS appears to drift upward (possible leak).")
    else:
        print("Memory trend: stable (no strong upward drift detected).")

    return {
        "duration_s": duration_s,
        "sample_count": len(mems),
        "iterations": n,
        "rss_min_mb": min(mems) if mems else None,
        "rss_max_mb": max(mems) if mems else None,
        "rss_slope_mb_per_s": slope,
        "memory_stable": not leak_suspected,
        "leak_suspected": leak_suspected,
    }


def _production_verdict(
    inference: dict[str, Any],
    detection: dict[str, Any],
    memory_profile: dict[str, Any],
) -> dict[str, Any]:
    best_fps = 0.0
    for r in inference.get("runs", []):
        if r.get("error"):
            continue
        best_fps = max(best_fps, float(r.get("mean_fps") or 0.0))
    fps_ok = best_fps > 15.0

    map_ok = False
    if not detection.get("skipped"):
        g = detection.get("global") or {}
        map_ok = float(g.get("mAP@50", 0.0) or 0.0) > 0.7

    mem_ok = memory_profile.get("memory_stable", False)
    if memory_profile.get("skipped"):
        mem_ok = False

    ready = bool(fps_ok and map_ok and mem_ok)
    lines = [
        f"FPS > 15 (best mean FPS={best_fps:.2f}): {fps_ok}",
        f"mAP@50 > 0.7 (test split): {map_ok}" + (" (skipped)" if detection.get("skipped") else ""),
        f"Memory stable: {mem_ok}" + (" (skipped)" if memory_profile.get("skipped") else ""),
        f"PRODUCTION READY: {ready}",
    ]
    return {
        "production_ready": ready,
        "checks": {"fps_ok": fps_ok, "map_ok": map_ok, "memory_stable": mem_ok, "best_mean_fps": best_fps},
        "lines": lines,
    }


def main() -> int:
    config = load_config(resolve_path("config.yaml"))
    model_path = _resolve_model_path(config)
    if not model_path.exists():
        print(f"Model weights not found: {model_path}")
        return 1

    logs = resolve_path("logs")
    logs.mkdir(parents=True, exist_ok=True)
    json_path = logs / "benchmark_results.json"
    txt_path = logs / "benchmark_report.txt"

    print(f"PhoneWatch benchmark — model: {model_path}")
    print(f"Started: {_now_iso()}")

    results: dict[str, Any] = {
        "timestamp": _now_iso(),
        "model_path": str(model_path),
    }

    results["inference_speed"] = benchmark_inference_speed(model_path, config)
    results["detection_accuracy"] = benchmark_detection_accuracy(model_path)
    results["context_classifier"] = benchmark_context_classifier()

    engine = PhoneWatchEngine(config_path=str(resolve_path("config.yaml")))
    try:
        results["e2e_latency"] = benchmark_e2e_latency(engine, n_samples=50)
    finally:
        engine.close()

    from ultralytics import YOLO as YOLOLoader

    probe_model = YOLOLoader(str(model_path))
    tids = _target_class_ids_from_model(probe_model)
    results["memory_profile"] = benchmark_memory_profile(model_path, duration_s=60.0, target_class_ids=tids)

    verdict = _production_verdict(results["inference_speed"], results["detection_accuracy"], results["memory_profile"])
    results["verdict"] = verdict

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    for line in verdict["lines"]:
        print(line)

    def json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [json_safe(v) for v in obj]
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, Path):
            return str(obj)
        return obj

    json_path.write_text(json.dumps(json_safe(results), indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {json_path}")

    report_lines = [
        f"PhoneWatch Benchmark Report — {_now_iso()}",
        f"Model: {model_path}",
        "",
        json.dumps(json_safe(results), indent=2),
    ]
    txt_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Saved report: {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
