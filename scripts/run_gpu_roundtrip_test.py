"""
200-image roundtrip benchmark: GPU enabled vs CPU fallback.

Run:
    python -m scripts.run_gpu_roundtrip_test
    python -m scripts.run_gpu_roundtrip_test --count 200 --skip-gpu
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gpu_runtime import cuda_memory_snapshot, write_gpu_readiness_report
from scripts.main_pipeline import run_pipeline
from scripts.utils import close_pipeline_logging, deep_update, discover_images, load_config, resolve_pipeline_paths, setup_project_folders


def _seed_images(output_dir: Path, count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = ("landscape", "text", "plain", "complex", "face_like")
    for i in range(count):
        path = output_dir / f"gpu_rt_{categories[i % len(categories)]}_{i:05d}.png"
        if path.is_file():
            continue
        if categories[i % len(categories)] == "text":
            im = Image.new("RGB", (480, 320), color=(28, 28, 28))
            draw = ImageDraw.Draw(im)
            draw.text((32, 128), f"GPU{i:05d}", fill=(235, 235, 235))
        else:
            im = Image.new("RGB", (384, 288), color=(i * 3 % 255, 60, 90))
        im.save(path)


def _base_overrides(test_root: Path, *, use_gpu: bool) -> Dict[str, Any]:
    paths = {
        "input_raw": str(test_root / "input_raw"),
        "temp_processed": str(test_root / "temp_processed"),
        "final_clean": str(test_root / "final_clean"),
        "quarantine": str(test_root / "quarantine"),
        "manual_review": str(test_root / "manual_review"),
        "logs": str(test_root / "logs"),
        "reports": str(test_root / "reports"),
    }
    if use_gpu:
        return {
            "paths": paths,
            "batch_size": 16,
            "max_qa_waves": 25,
            "gpu": {
                "device": "cuda",
                "gpu_id": 0,
                "memory_efficient": True,
                "adaptive_batch": True,
                "empty_cache_between_batches": True,
                "use_fp16_inpaint": True,
            },
            "device": "cuda",
            "insightface": {"enabled": True},
            "lama": {"backend": "simple_lama", "device": "cuda"},
            "monitoring": {"resource_monitoring": True},
            "performance": {"always_monitor": True, "adaptive_batch_size": True, "max_batch_size": 48},
            "qa": {
                "reuse_processing_ocr": True,
                "text_det_score_fail": 0.95,
                "artifact_ssim_min": 0.55,
            },
            "security": {"level": "standard", "secure_wipe": False},
        }
    return {
        "paths": paths,
        "batch_size": 16,
        "max_qa_waves": 25,
        "gpu": {"device": "cpu", "adaptive_batch": False},
        "device": "cpu",
        "insightface": {"enabled": True, "ctx_id": -1},
        "lama": {"backend": "simple_lama", "device": "cpu"},
        "monitoring": {"resource_monitoring": True},
        "qa": {
            "reuse_processing_ocr": True,
            "text_det_score_fail": 0.95,
            "artifact_ssim_min": 0.55,
        },
        "security": {"level": "standard", "secure_wipe": False},
    }


def _run_mode(
    test_root: Path,
    *,
    count: int,
    use_gpu: bool,
    label: str,
) -> Dict[str, Any]:
    import yaml

    cfg_base = load_config(ROOT / "config.yaml")
    cfg = dict(cfg_base)
    deep_update(cfg, _base_overrides(test_root, use_gpu=use_gpu))

    config_path = test_root / f"config_{label}.yaml"
    config_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    setup_project_folders(test_root, cfg)
    paths = resolve_pipeline_paths(test_root, cfg)
    _seed_images(paths["input_raw"], count)

    t0 = time.monotonic()
    result = run_pipeline(
        config_path=config_path,
        project_root=test_root,
        test_mode=True,
    )
    elapsed = time.monotonic() - t0
    close_pipeline_logging()

    final_n = len(discover_images(paths["final_clean"], [".jpg"]))
    metrics = result.get("metrics") or {}
    snap = cuda_memory_snapshot()

    return {
        "label": label,
        "use_gpu": use_gpu,
        "count": count,
        "final_clean": final_n,
        "pass_rate": final_n / count if count else 0.0,
        "elapsed_sec": elapsed,
        "images_per_sec": count / elapsed if elapsed > 0 else 0.0,
        "avg_sec_per_image": metrics.get("avg_sec_per_image"),
        "peak_allocated_mb": snap.get("allocated_mb"),
        "gpu_validation": result.get("gpu_validation"),
        "passed": final_n >= count * 0.95 and not result.get("stopped_early"),
    }


def run_gpu_roundtrip(
    *,
    count: int = 200,
    skip_gpu: bool = False,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run CPU then GPU modes in an **isolated** temp tree (only ``count`` seeded images).

    ``project_root`` only controls where ``reports/gpu_benchmark.json`` is written.
    """
    count = max(1, int(count))
    compare: Dict[str, Any] = {}
    report_root = Path(project_root) if project_root is not None else ROOT

    with tempfile.TemporaryDirectory(prefix="gpu_roundtrip_") as tmp:
        test_root = Path(tmp)
        compare["cpu"] = _run_mode(test_root, count=count, use_gpu=False, label="cpu")
        if not skip_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    compare["gpu"] = _run_mode(test_root, count=count, use_gpu=True, label="gpu")
                else:
                    compare["gpu"] = {"skipped": True, "reason": "cuda_not_available"}
            except ImportError:
                compare["gpu"] = {"skipped": True, "reason": "torch_not_installed"}

    cfg = load_config(ROOT / "config.yaml")
    cpu_row = compare.get("cpu") if isinstance(compare.get("cpu"), dict) else {}
    report_path = write_gpu_readiness_report(
        report_root,
        cfg,
        metrics={
            "total_routed": cpu_row.get("final_clean", 0),
            "avg_sec_per_image": cpu_row.get("avg_sec_per_image", 0),
            "images_per_sec": cpu_row.get("images_per_sec", 0),
        },
        benchmark_compare={
            k: v
            for k, v in compare.items()
            if isinstance(v, dict) and "images_per_sec" in v
        },
        elapsed_sec=cpu_row.get("elapsed_sec"),
    )

    out_path = report_root / "reports" / "gpu_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(compare, indent=2, default=str), encoding="utf-8")

    return {
        "compare": compare,
        "gpu_readiness_report": str(report_path),
        "benchmark_json": str(out_path),
        "passed": all(v.get("passed", v.get("skipped")) for v in compare.values() if isinstance(v, dict)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU vs CPU 200-image roundtrip benchmark")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--skip-gpu", action="store_true", help="CPU fallback only")
    parser.add_argument("--project-root", type=Path, default=None, help="Persist under project reports/")
    args = parser.parse_args()

    print(f"GPU roundtrip test: {args.count} images")
    summary = run_gpu_roundtrip(
        count=args.count,
        skip_gpu=args.skip_gpu,
        project_root=args.project_root,
    )
    print(json.dumps(summary, indent=2, default=str))
    print(f"Report: {summary.get('gpu_readiness_report')}")
    if not summary.get("passed"):
        sys.exit(1)
    print("PASSED gpu roundtrip benchmark")


if __name__ == "__main__":
    main()
