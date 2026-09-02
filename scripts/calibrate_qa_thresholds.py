"""
Suggest QA thresholds from a held-out calibration set (~100 images).

Run from project root:
    python -m scripts.calibrate_qa_thresholds
    python -m scripts.calibrate_qa_thresholds --count 100 --input-dir input_raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agentic_qa_crew import evaluate_image_qa, run_detection_verification, run_identity_integrity
from scripts.batch_processor import AnonymizationEngine, process_batch
from scripts.main_pipeline import route_batch_outputs
from scripts.utils import discover_images, load_config, resolve_pipeline_paths, setup_project_folders


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def calibrate(*, count: int, input_dir: Path, project_root: Path) -> Dict[str, Any]:
    cfg = load_config(project_root / "config.yaml")
    setup_project_folders(project_root, cfg)
    paths = resolve_pipeline_paths(project_root, cfg)

    images = discover_images(input_dir, cfg.get("image_extensions", [".jpg", ".jpeg", ".png"]))
    if len(images) > count:
        images = images[:count]
    if not images:
        raise SystemExit(f"No calibration images under {input_dir}")

    engine = AnonymizationEngine(project_root, cfg)
    engine.warm_models()

    bp = process_batch(images, batch_index=99999, engine=engine, cfg=cfg, project_root=project_root)

    ocr_scores: List[float] = []
    ssim_vals: List[float] = []
    identity_dists: List[float] = []
    pass_flags: List[bool] = []

    for jpg in sorted(bp.batch_dir.glob("*.jpg")):
        dec = evaluate_image_qa(jpg, cfg, project_root=project_root)
        det = dec.get("detection") or {}
        integ = dec.get("integrity") or {}
        checks = integ.get("checks") or {}
        ocr_scores.append(float(det.get("max_det_score") or 0.0))
        if checks.get("ssim") is not None:
            ssim_vals.append(float(checks["ssim"]))
        ident = checks.get("identity") or {}
        if ident.get("cosine_distance") is not None:
            identity_dists.append(float(ident["cosine_distance"]))
        pass_flags.append(dec.get("final_decision") == "pass")

    # Conservative suggestions: artifact_ssim_min tuned to 0.72–0.80 band for high-volume runs
    ssim_p5 = _percentile(ssim_vals, 5) if ssim_vals else 0.75
    ssim_suggested = round(min(0.80, max(0.72, ssim_p5)), 3)

    suggested = {
        "qa.text_det_score_fail": round(max(0.85, _percentile(ocr_scores, 95)), 3),
        "qa.artifact_ssim_min": ssim_suggested,
        "qa.identity_distance_min": round(
            min(0.5, _percentile(identity_dists, 5) if identity_dists else 0.35),
            3,
        ),
    }

    report = {
        "calibration_images": len(images),
        "processed_outputs": len(list(bp.batch_dir.glob("*.jpg"))),
        "pass_rate": float(sum(pass_flags) / len(pass_flags)) if pass_flags else 0.0,
        "ocr_max_score": {"p50": _percentile(ocr_scores, 50), "p95": _percentile(ocr_scores, 95)},
        "ssim": {"p5": _percentile(ssim_vals, 5), "p50": _percentile(ssim_vals, 50)},
        "identity_distance": {
            "p5": _percentile(identity_dists, 5),
            "p50": _percentile(identity_dists, 50),
        },
        "suggested_config_overrides": suggested,
        "notes": [
            "Review suggested thresholds with visual spot-checks on 10–20 borderline images.",
            "Lower artifact_ssim_min => stricter integrity gate (recommended band: 0.72–0.80 for 37k+ runs).",
            "Higher text_det_score_fail => more tolerant of residual OCR detections.",
        ],
    }

    out_path = paths["reports"] / "qa_calibration_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate QA thresholds from sample images")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--input-dir", type=Path, default=None)
    args = parser.parse_args()

    root = ROOT
    input_dir = args.input_dir or (root / "input_raw")
    report = calibrate(count=int(args.count), input_dir=Path(input_dir), project_root=root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
