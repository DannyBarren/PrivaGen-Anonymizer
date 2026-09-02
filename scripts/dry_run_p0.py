"""
P0 fix verification: stem dedup, path handling, audit JSON serialization, PaddleOCR import.

Run from project root:
    python -m scripts.dry_run_p0
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.batch_processor import make_path_dataloader, process_batch
from scripts.shared_models import build_paddle_ocr
from scripts.main_pipeline import _final_clean_has_stem, discover_pending
from scripts.utils import json_sanitize, save_audit_json


def _make_samples(input_dir: Path, n: int = 5) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    exts = [".png", ".jpg", ".png", ".jpeg", ".png"]
    for i in range(n):
        ext = exts[i]
        p = input_dir / f"sample_{i:02d}{ext}"
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, i % 3] = 180 + i * 10
        Image.fromarray(arr).save(p)
        paths.append(p)
    return paths


def test_stem_dedup(tmp: Path) -> None:
    final_clean = tmp / "final_clean"
    final_clean.mkdir(parents=True, exist_ok=True)
    (final_clean / "sample_00.jpg").write_bytes(b"x")
    assert _final_clean_has_stem(final_clean, "sample_00")
    assert not _final_clean_has_stem(final_clean, "sample_01")
    print("  stem dedup: OK")


def test_json_sanitize(tmp: Path) -> None:
    audit = {
        "polygon": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "emb": np.array([0.1, 0.2], dtype=np.float32),
        "nested": {"boxes": [{"polygon": np.ones((4, 2), dtype=np.float32)}]},
    }
    out_path = tmp / "out.jpg"
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(out_path)
    save_audit_json(out_path, audit)
    loaded = json.loads(out_path.with_suffix(".json").read_text(encoding="utf-8"))
    assert isinstance(loaded["polygon"], list)
    assert isinstance(loaded["nested"]["boxes"][0]["polygon"], list)
    print("  json_sanitize / save_audit_json: OK")


def test_dataloader_paths(image_paths: list[Path]) -> None:
    dl = make_path_dataloader(image_paths, batch_size=2, num_workers=1)
    batches = list(dl)
    flat = [p for batch in batches for p in batch]
    assert len(flat) == len(image_paths)
    assert all(isinstance(p, Path) for p in flat)
    assert {p.name for p in flat} == {p.name for p in image_paths}
    print(f"  dataloader paths ({len(flat)} images): OK")


def test_discover_pending(tmp: Path, cfg: dict) -> None:
    pending = discover_pending(tmp, cfg)
    stems = {p.stem for p in pending}
    assert "sample_01" in stems
    assert "sample_00" not in stems
    print(f"  discover_pending ({len(pending)} pending, sample_00 skipped): OK")


def test_paddle_import(cfg: dict) -> None:
    ocr = build_paddle_ocr(cfg)
    assert ocr is not None
    print("  build_paddle_ocr: OK")


class _MockOCR:
    """Stub OCR so process_batch smoke test does not depend on Paddle runtime."""

    def ocr(self, bgr, cls=True):  # noqa: ARG002
        return [[]]


def test_process_batch_paths(tmp: Path, cfg: dict, image_paths: list[Path]) -> None:
    """Smoke process_batch: ensures image_paths reach dataloader (may skip heavy models)."""
    from scripts.batch_processor import AnonymizationEngine

    cfg_run = dict(cfg)
    cfg_run["insightface"] = {"enabled": False}
    cfg_run["deep_privacy2"] = {"repo_root": ""}
    cfg_run["lama"] = {"backend": "none"}
    cfg_run["batch_size"] = 5
    cfg_run["num_workers"] = 2
    cfg_run["security"] = {"copy_input_raw": True}
    cfg_run["metadata"] = {"prefer_exiftool": False}

    engine = AnonymizationEngine(tmp, cfg_run)
    engine.ocr = _MockOCR()
    bp = process_batch(
        image_paths[:5],
        batch_index=1,
        engine=engine,
        cfg=cfg_run,
        project_root=tmp,
    )
    jpg_out = list(bp.batch_dir.glob("*.jpg"))
    assert len(jpg_out) == 5, f"expected 5 outputs, got {len(jpg_out)} in {bp.batch_dir}"
    for j in jpg_out:
        sidecar = j.with_suffix(".json")
        assert sidecar.is_file(), f"missing sidecar for {j.name}"
        json.loads(sidecar.read_text(encoding="utf-8"))
    print(f"  process_batch ({len(jpg_out)} jpgs + valid JSON): OK")


def main() -> None:
    from scripts.utils import load_config, setup_project_folders

    cfg = load_config(ROOT / "config.yaml")
    print("P0 dry-run starting...")

    with tempfile.TemporaryDirectory(prefix="anon_p0_") as td:
        tmp = Path(td)
        paths_cfg = {
            "input_raw": str(tmp / "input_raw"),
            "final_clean": str(tmp / "final_clean"),
            "quarantine": str(tmp / "quarantine"),
            "manual_review": str(tmp / "manual_review"),
            "temp_processed": str(tmp / "temp_processed"),
            "logs": str(tmp / "logs"),
            "reports": str(tmp / "reports"),
        }
        cfg["paths"] = paths_cfg
        setup_project_folders(tmp, cfg)

        samples = _make_samples(tmp / "input_raw", 5)
        test_stem_dedup(tmp)
        test_json_sanitize(tmp / "temp_processed")
        test_dataloader_paths(samples)
        test_discover_pending(tmp, cfg)
        test_paddle_import(cfg)
        test_process_batch_paths(tmp, cfg, samples)

    print("P0 dry-run: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
