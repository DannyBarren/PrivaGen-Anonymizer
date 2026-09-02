"""
Edge-case tests: mixed extensions, large images, zero-face, corrupted files.

Run: python -m tests.test_edge_cases
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_processor import AnonymizationEngine, parallel_imread_rgb, process_batch
from scripts.main_pipeline import discover_pending
from scripts.processing_locks import active_locked_stems, write_processing_lock
from scripts.security import assert_not_input_raw, SecurityViolationError
from scripts.utils import deep_update, load_config, resolve_pipeline_paths, setup_project_folders


def _paths_override(test_root: Path) -> dict:
    return {
        "input_raw": str(test_root / "input_raw"),
        "temp_processed": str(test_root / "temp_processed"),
        "final_clean": str(test_root / "final_clean"),
        "quarantine": str(test_root / "quarantine"),
        "manual_review": str(test_root / "manual_review"),
        "logs": str(test_root / "logs"),
        "reports": str(test_root / "reports"),
    }


def run_edge_case_tests() -> dict:
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="edge_cases_") as tmp:
        test_root = Path(tmp)
        cfg = load_config(ROOT / "config.yaml")
        deep_update(cfg, {
            "paths": _paths_override(test_root),
            "batch_size": 4,
            "device": "cpu",
            "insightface": {"enabled": False},
            "lama": {"backend": "none"},
        })
        setup_project_folders(test_root, cfg)
        paths = resolve_pipeline_paths(test_root, cfg)
        inp = paths["input_raw"]
        inp.mkdir(parents=True, exist_ok=True)

        # Mixed extensions
        Image.new("RGB", (128, 128), color=(10, 20, 30)).save(inp / "mix_a.jpg")
        Image.new("RGB", (128, 128), color=(20, 30, 40)).save(inp / "mix_b.png")
        Image.new("RGB", (128, 128), color=(30, 40, 50)).save(inp / "mix_c.webp")

        # Large image (downscaled by pipeline decode, should not crash)
        large = Image.new("RGB", (2400, 1800), color=(50, 60, 70))
        draw = ImageDraw.Draw(large)
        draw.rectangle((100, 100, 400, 200), outline=(255, 255, 255))
        large.save(inp / "large_scene.jpg", quality=85)

        # Zero-face plain scene
        Image.new("RGB", (400, 300), color=(90, 100, 110)).save(inp / "zero_face_plain.png")

        # Corrupted file
        (inp / "corrupted.jpg").write_bytes(b"not-an-image-at-all")

        # input_raw write guard
        try:
            assert_not_input_raw(inp / "blocked.jpg", inp)
            errors.append("security: write guard should have blocked input_raw path")
        except SecurityViolationError:
            pass

        # Processing lock excludes stem from discover_pending
        batch_dir = paths["temp_processed"] / "batch_00001"
        write_processing_lock(batch_dir, batch_index=1, stems=["mix_a"])
        locked = active_locked_stems(paths["temp_processed"])
        pending = discover_pending(test_root, cfg)
        if "mix_a" in {p.stem for p in pending}:
            errors.append("lock: mix_a should be excluded while lock active")

        engine = AnonymizationEngine(test_root, cfg)
        engine.warm_models()
        images = [inp / "mix_a.jpg", inp / "mix_b.png", inp / "large_scene.jpg", inp / "zero_face_plain.png", inp / "corrupted.jpg"]
        bp = process_batch(images, batch_index=1, engine=engine, cfg=cfg, project_root=test_root)

        decoded = parallel_imread_rgb([inp / "corrupted.jpg"], 1)
        if decoded[0] is not None:
            errors.append("corrupted: parallel_imread should return None")

        if not (bp.batch_dir / "corrupted.error_audit.json").is_file():
            errors.append("corrupted: expected error_audit.json")

        ok_outputs = list(bp.batch_dir.glob("*.jpg"))
        if len(ok_outputs) < 3:
            errors.append(f"expected >=3 successful outputs, got {len(ok_outputs)}")

    return {"passed": len(errors) == 0, "errors": errors}


def main() -> None:
    summary = run_edge_case_tests()
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        sys.exit(1)
    print("PASSED edge-case tests")


if __name__ == "__main__":
    main()
