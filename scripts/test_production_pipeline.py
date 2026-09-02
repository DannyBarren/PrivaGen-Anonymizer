"""
End-to-end production smoke test (20–50 images).

Run from project root:
    python -m scripts.test_production_pipeline
    python -m scripts.test_production_pipeline --count 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.main_pipeline import run_pipeline
from scripts.utils import discover_images, load_config, resolve_pipeline_paths, setup_project_folders


def _seed_input_raw(input_dir: Path, count: int) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    existing = discover_images(input_dir, [".jpg", ".jpeg", ".png", ".webp"])
    if len(existing) >= count:
        return
    for i in range(count):
        name = f"prod_test_{i:03d}.png"
        path = input_dir / name
        if path.is_file():
            continue
        im = Image.new("RGB", (320, 240), color=(40 + (i * 7) % 200, 80, 120))
        draw = ImageDraw.Draw(im)
        draw.rectangle((20, 20, 140, 80), outline=(255, 255, 0), width=2)
        draw.text((24, 28), f"TEST{i:03d}", fill=(255, 255, 255))
        im.save(path)


def _audit_checks(final_clean: Path, reports_dir: Path, sample_n: int = 5) -> dict:
    jpgs = sorted(final_clean.glob("*.jpg"))
    summary = {
        "final_clean_count": len(jpgs),
        "checks": [],
        "errors": [],
    }
    for jpg in jpgs[:sample_n]:
        js = jpg.with_suffix(".json")
        if not js.is_file():
            summary["errors"].append(f"missing sidecar: {jpg.name}")
            continue
        audit = json.loads(js.read_text(encoding="utf-8"))
        ih = audit.get("integrity_hashes") or {}
        sm = audit.get("success_metrics") or {}
        qa = audit.get("qa") or {}
        row = {
            "image": jpg.name,
            "source_sha256": bool(ih.get("source_sha256")),
            "output_sha256": bool(ih.get("output_sha256")),
            "dp2_success": sm.get("deep_privacy2_replacement_success"),
            "inpaint_success": sm.get("inpaint_success"),
            "metadata_method": (audit.get("actions") or {}).get("metadata_strip_method"),
            "qa_decision": qa.get("final_decision"),
            "qa_route": qa.get("final_route"),
        }
        summary["checks"].append(row)
        if not ih.get("output_sha256"):
            summary["errors"].append(f"no output hash: {jpg.name}")
        meta_method = row.get("metadata_method") or ""
        if meta_method not in ("exiftool", "pillow_reencode"):
            summary["errors"].append(f"unexpected metadata strip: {jpg.name}:{meta_method}")
    master = reports_dir / "master_summary.csv"
    summary["master_csv_exists"] = master.is_file()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=30, help="Images to seed in input_raw (20-50 recommended)")
    args = parser.parse_args()
    count = max(20, min(50, int(args.count)))

    cfg = load_config(ROOT / "config.yaml")
    setup_project_folders(ROOT, cfg)
    paths = resolve_pipeline_paths(ROOT, cfg)
    _seed_input_raw(paths["input_raw"], count)

    overrides = {
        "batch_size": min(10, count),
        "max_qa_waves": 2,
    }
    print(f"Running pipeline on up to {count} images (batch_size={overrides['batch_size']})...")
    result = run_pipeline(
        config_path=ROOT / "config.yaml",
        config_overrides=overrides,
    )
    print("Pipeline result:", json.dumps(result, indent=2))

    report = _audit_checks(paths["final_clean"], paths["reports"])
    print("Audit sample:", json.dumps(report, indent=2))

    if report["errors"]:
        print("FAILED checks:", report["errors"])
        sys.exit(1)
    if report["final_clean_count"] == 0:
        print("FAILED: no images in final_clean (all quarantined or processing errors)")
        sys.exit(1)
    print(f"PASSED: {report['final_clean_count']} images in final_clean")


if __name__ == "__main__":
    main()
