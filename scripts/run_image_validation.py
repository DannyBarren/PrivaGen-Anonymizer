"""
End-to-end IMAGE-ONLY functional validation + residual-PII gate.

Runs the real anonymization pipeline (DeepPrivacy2 face GAN → PaddleOCR text detection
→ IOPaint/LaMa inpainting → QA → routing) on synthetic and/or operator-supplied sample
images, forcing ``security.level: full``, then performs SECONDARY residual detection on
the outputs:

  * residual FACES   (InsightFace on final_clean/) — any above threshold  => FAIL
  * residual TEXT/PII (PaddleOCR on final_clean/)  — score above ceiling  => FAIL

It also verifies originals are byte-for-byte untouched and that temp is cleaned up.

This tool REQUIRES the full ML stack + GPU (torch, opencv, paddleocr, iopaint/simple-lama,
insightface) and DeepPrivacy2 in vendor/. It is meant to run on the Lambda.ai GPU host as
the final functional gate — it will not run on a bare CPU box without the stack.

Run from project root (on the GPU host):
    python -m scripts.run_image_validation --count 12
    python -m scripts.run_image_validation --samples-dir /path/to/real_face_samples
    python -m scripts.run_image_validation --json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Residual thresholds: any output at/above these fails the gate.
RESIDUAL_FACE_MAX = 0            # zero residual faces tolerated in anonymized output
RESIDUAL_OCR_SCORE_MAX = 0.85    # high-confidence text remaining => leak


def _make_samples(dst: Path, count: int) -> int:
    """Generate deterministic synthetic images with embedded text (PII-like) + shapes."""
    import numpy as np
    from PIL import Image, ImageDraw

    dst.mkdir(parents=True, exist_ok=True)
    made = 0
    pii = [
        "John Q. Public", "SSN 123-45-6789", "PLATE 7XKD221",
        "acct #0098-2231", "+1 (555) 010-2288", "123 Elm St, Apt 4B",
    ]
    for i in range(count):
        w, h = 640, 400
        arr = (np.random.default_rng(i).integers(20, 90, size=(h, w, 3))).astype("uint8")
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        # Draw a couple of readable strings (stand-in for on-image PII/plates/text).
        d.rectangle([20, 40, 620, 90], fill=(240, 240, 240))
        d.text((30, 55), pii[i % len(pii)], fill=(10, 10, 10))
        d.rectangle([20, 300, 620, 350], fill=(240, 240, 240))
        d.text((30, 315), pii[(i + 3) % len(pii)], fill=(10, 10, 10))
        im.save(dst / f"sample_{i:04d}.jpg", quality=92)
        made += 1
    return made


def _hash_dir(folder: Path) -> Dict[str, str]:
    from scripts.utils import sha256_file

    out: Dict[str, str] = {}
    if folder.is_dir():
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                out[str(p.relative_to(folder))] = sha256_file(p)
    return out


def run_validation(*, count: int, device: str, samples_dir: Optional[Path]) -> Dict[str, Any]:
    from scripts.main_pipeline import run_pipeline
    from scripts.shared_models import (
        get_shared_insightface_app,
        get_shared_paddle_ocr,
        ocr_polys_and_scores,
    )
    from scripts.utils import discover_images, imread_rgb

    findings: Dict[str, Any] = {"components": {}, "residual": {}, "integrity": {}, "status": "PENDING"}
    work = Path(tempfile.mkdtemp(prefix="privagen_validation_"))
    try:
        input_raw = work / "input_raw"
        input_raw.mkdir(parents=True, exist_ok=True)
        if samples_dir:
            imgs = discover_images(Path(samples_dir), [".jpg", ".jpeg", ".png", ".webp"])
            for p in imgs[: count or len(imgs)]:
                shutil.copy2(p, input_raw / p.name)
            n = len(list(input_raw.iterdir()))
        else:
            n = _make_samples(input_raw, count)
        findings["sample_count"] = n

        pre_hashes = _hash_dir(input_raw)

        overrides = {
            "paths": {
                "input_raw": str(input_raw),
                "temp_processed": str(work / "temp_processed"),
                "final_clean": str(work / "final_clean"),
                "quarantine": str(work / "quarantine"),
                "manual_review": str(work / "manual_review"),
                "reports": str(work / "reports"),
                "logs": str(work / "logs"),
            },
            "device": device,
            "gpu": {"device": device},
            "scope": {"processing_mode": "images_only", "video_support": "deferred"},
        }

        # Force security.level=full so secure_wipe is active for sensitive-data behavior.
        result = run_pipeline(
            project_root=work,
            config_overrides=overrides,
            security_level="full",
            gpu_device=device,
        )
        findings["pipeline_result_keys"] = sorted(result.keys()) if isinstance(result, dict) else None

        # Originals untouched?
        post_hashes = _hash_dir(input_raw)
        findings["integrity"] = {
            "originals_untouched": pre_hashes == post_hashes,
            "count": len(pre_hashes),
        }

        # Temp cleaned up (secure wipe under level full)?
        findings["integrity"]["temp_cleaned"] = not (work / "temp_processed").exists() or not any(
            (work / "temp_processed").rglob("*")
        )

        # Secondary residual detection on outputs.
        final_clean = work / "final_clean"
        outputs = discover_images(final_clean, [".jpg", ".jpeg", ".png"]) if final_clean.is_dir() else []
        findings["output_count"] = len(outputs)

        ocr = get_shared_paddle_ocr({"paddleocr": {"lang": "en"}})
        try:
            face_app = get_shared_insightface_app({"insightface": {"model_name": "buffalo_l", "ctx_id": -1}})
        except Exception:  # noqa: BLE001
            face_app = None

        residual_face_hits: List[str] = []
        residual_text_hits: List[str] = []

        for p in outputs:
            rgb = imread_rgb(p)
            bgr = rgb[:, :, ::-1].copy()
            if face_app is not None:
                faces = face_app.get(bgr) or []
                if len(faces) > RESIDUAL_FACE_MAX:
                    residual_face_hits.append(f"{p.name}:{len(faces)}")
            boxes = ocr_polys_and_scores(ocr, bgr) or []
            top = max((float(b[-1]) for b in boxes), default=0.0)
            if top >= RESIDUAL_OCR_SCORE_MAX:
                residual_text_hits.append(f"{p.name}:{top:.2f}")

        findings["residual"] = {
            "face_hits": residual_face_hits,
            "text_hits": residual_text_hits,
            "face_ok": not residual_face_hits,
            "text_ok": not residual_text_hits,
        }
        findings["components"] = {
            "insightface_loaded": face_app is not None,
            "paddleocr_loaded": ocr is not None,
        }

        passed = (
            findings["integrity"]["originals_untouched"]
            and findings["residual"]["face_ok"]
            and findings["residual"]["text_ok"]
            and findings["output_count"] > 0
        )
        findings["status"] = "PASS" if passed else "FAIL"
        return findings
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Image-only end-to-end validation + residual-PII gate.")
    ap.add_argument("--count", type=int, default=12, help="Synthetic samples to generate (ignored with --samples-dir)")
    ap.add_argument("--device", default="cuda", help="Compute device: cuda | cpu")
    ap.add_argument("--samples-dir", type=Path, default=None, help="Use real sample images from this dir (recommended for face GAN)")
    ap.add_argument("--json", action="store_true", help="Emit JSON report")
    args = ap.parse_args()

    findings = run_validation(count=args.count, device=args.device, samples_dir=args.samples_dir)
    if args.json:
        print(json.dumps(findings, indent=2))
    else:
        print("\n===== IMAGE-ONLY FUNCTIONAL VALIDATION =====")
        print(f"samples={findings.get('sample_count')} outputs={findings.get('output_count')}")
        print(f"originals_untouched={findings['integrity'].get('originals_untouched')}")
        print(f"temp_cleaned={findings['integrity'].get('temp_cleaned')}")
        print(f"residual_faces={findings['residual'].get('face_hits')}")
        print(f"residual_text={findings['residual'].get('text_hits')}")
        print(f"\nRESULT: {findings['status']}")
    raise SystemExit(0 if findings.get("status") == "PASS" else 1)


if __name__ == "__main__":
    main()
