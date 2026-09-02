"""
Process-wide singletons for heavy CV models (PaddleOCR, InsightFace).

Warmed once from ``AnonymizationEngine.warm_models()`` and reused by batch processing
and deterministic QA to avoid duplicate GPU allocations and redundant constructors.
"""

from __future__ import annotations

import os

# Paddle 3.x / oneDNN: avoid Windows CPU executor bugs before any paddle import.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

try:
    from paddleocr import PaddleOCR
except ImportError:
    PaddleOCR = None  # type: ignore[misc, assignment]

_SHARED_OCR: Any = None
_SHARED_OCR_KEY: Optional[tuple] = None
_SHARED_INSIGHT: Any = None
_SHARED_INSIGHT_KEY: Optional[tuple] = None

# Usage counters for performance reports (process-wide)
_OCR_INFERENCE_CALLS: int = 0
_OCR_CACHE_HITS: int = 0
_INSIGHTFACE_CALLS: int = 0


def build_paddle_ocr(cfg: Dict[str, Any]) -> Any:
    """Construct PaddleOCR (lazy-import + legacy-kwarg fallback for newer PaddleOCR)."""
    from .gpu_runtime import configure_paddle_gpu, paddle_ocr_device_kwargs

    ocr_cls = PaddleOCR
    if ocr_cls is None:
        from paddleocr import PaddleOCR as ocr_cls  # type: ignore[no-redef]

    configure_paddle_gpu(cfg)
    poc = cfg.get("paddleocr", {}) or {}
    base_kwargs = {
        "use_angle_cls": bool(poc.get("use_angle_cls", True)),
        "lang": str(poc.get("lang", "en")),
    }
    legacy_kwargs = {
        "show_log": bool(poc.get("show_log", False)),
        "det_db_thresh": float(poc.get("det_db_thresh", 0.3)),
        "det_db_box_thresh": float(poc.get("det_db_box_thresh", 0.5)),
    }
    gpu_kwargs = paddle_ocr_device_kwargs(cfg)
    merged = {**base_kwargs, **legacy_kwargs, **gpu_kwargs}
    try:
        return ocr_cls(**merged)
    except (ValueError, TypeError) as exc:
        logger.warning("paddleocr_init_retry_without_legacy_kwargs", error=str(exc))
        try:
            return ocr_cls(**base_kwargs, **gpu_kwargs)
        except (ValueError, TypeError):
            return ocr_cls(**base_kwargs)


def ocr_polys_and_scores(ocr: Any, bgr: np.ndarray) -> List[Dict[str, Any]]:
    """Run OCR once; normalize to polygon + text + score dicts."""
    global _OCR_INFERENCE_CALLS
    _OCR_INFERENCE_CALLS += 1
    try:
        try:
            res = ocr.ocr(bgr, cls=True)
        except TypeError:
            res = ocr.ocr(bgr)
    except Exception as exc:  # noqa: BLE001
        logger.error("paddleocr_inference_failed", error=str(exc))
        return []
    boxes: List[Dict[str, Any]] = []
    if not res:
        return boxes
    lines = res[0] if isinstance(res, list) and len(res) == 1 else res
    if lines is None:
        return boxes
    for line in lines:
        try:
            poly = line[0]
            txt, score = line[1]
            boxes.append({"polygon": np.asarray(poly, dtype=np.float32), "text": str(txt), "score": float(score)})
        except Exception:  # noqa: BLE001
            continue
    return boxes


def serialize_ocr_boxes(boxes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """JSON-safe OCR boxes for audit sidecar / QA cache."""
    out: List[Dict[str, Any]] = []
    for box in boxes:
        poly = box.get("polygon")
        if isinstance(poly, np.ndarray):
            poly_list = poly.tolist()
        else:
            poly_list = poly
        out.append(
            {
                "polygon": poly_list,
                "text": str(box.get("text", "")),
                "score": float(box.get("score", 0.0)),
            }
        )
    return out


def deserialize_ocr_boxes(raw: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Restore OCR boxes from audit JSON for QA reuse."""
    out: List[Dict[str, Any]] = []
    for box in raw:
        try:
            out.append(
                {
                    "polygon": np.asarray(box["polygon"], dtype=np.float32),
                    "text": str(box.get("text", "")),
                    "score": float(box.get("score", 0.0)),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return out


def ocr_boxes_max_score(boxes: Sequence[Dict[str, Any]]) -> float:
    max_score = 0.0
    for box in boxes:
        try:
            max_score = max(max_score, float(box.get("score", 0.0)))
        except Exception:  # noqa: BLE001
            continue
    return max_score


def _paddle_cache_key(cfg: Dict[str, Any]) -> tuple:
    poc = cfg.get("paddleocr", {}) or {}
    return (
        bool(poc.get("use_angle_cls", True)),
        str(poc.get("lang", "en")),
        float(poc.get("det_db_thresh", 0.3)),
        float(poc.get("det_db_box_thresh", 0.5)),
    )


def _insight_cache_key(cfg: Dict[str, Any]) -> tuple:
    from .gpu_runtime import insightface_ctx_id

    ic = cfg.get("insightface", {}) or {}
    ctx = insightface_ctx_id(cfg)
    if ctx < 0:
        ctx = int(ic.get("ctx_id", -1))
    return (str(ic.get("model_name", "buffalo_l")), int(ctx))


def warm_shared_paddle_ocr(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _SHARED_OCR, _SHARED_OCR_KEY
    key = _paddle_cache_key(cfg)
    if _SHARED_OCR is not None and _SHARED_OCR_KEY == key:
        return {"status": "ok", "shared": True, "reused": True}
    _SHARED_OCR = build_paddle_ocr(cfg)
    _SHARED_OCR_KEY = key
    return {"status": "ok", "shared": True, "reused": False}


def warm_shared_insightface(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _SHARED_INSIGHT, _SHARED_INSIGHT_KEY
    ic = cfg.get("insightface", {}) or {}
    if not bool(ic.get("enabled", False)):
        _SHARED_INSIGHT = None
        _SHARED_INSIGHT_KEY = None
        return {"enabled": False, "shared": True}

    key = _insight_cache_key(cfg)
    if _SHARED_INSIGHT is not None and _SHARED_INSIGHT_KEY == key:
        return {"enabled": True, "shared": True, "reused": True, "model": key[0], "ctx_id": key[1]}

    try:
        from insightface.app import FaceAnalysis
        from .gpu_runtime import SharedCudaContext, insightface_ctx_id

        SharedCudaContext.configure(cfg)
        name = str(ic.get("model_name", "buffalo_l"))
        ctx_id = insightface_ctx_id(cfg)
        if ctx_id < 0:
            ctx_id = int(ic.get("ctx_id", -1))
        try:
            import torch

            if ctx_id >= 0 and not torch.cuda.is_available():
                ctx_id = -1
        except Exception:  # noqa: BLE001
            if ctx_id >= 0:
                ctx_id = -1
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ctx_id >= 0 else ["CPUExecutionProvider"]
        app = FaceAnalysis(name=name, providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        _SHARED_INSIGHT = app
        _SHARED_INSIGHT_KEY = key
        return {"enabled": True, "shared": True, "reused": False, "model": name, "ctx_id": ctx_id}
    except Exception as exc:  # noqa: BLE001
        logger.warning("insightface_shared_warm_failed", error=str(exc))
        _SHARED_INSIGHT = None
        _SHARED_INSIGHT_KEY = None
        return {"enabled": False, "shared": True, "error": str(exc)}


def get_shared_paddle_ocr(cfg: Dict[str, Any]) -> Any:
    """Return the warmed PaddleOCR instance (warming on first use if needed)."""
    global _SHARED_OCR
    if _SHARED_OCR is None:
        warm_shared_paddle_ocr(cfg)
    if _SHARED_OCR is None:
        raise RuntimeError("PaddleOCR is not available; install paddleocr and paddlepaddle.")
    return _SHARED_OCR


def get_shared_insightface_app(cfg: Dict[str, Any]) -> Any:
    """Return the warmed InsightFace app or None when disabled / unavailable."""
    if not bool((cfg.get("insightface", {}) or {}).get("enabled", False)):
        return None
    if _SHARED_INSIGHT is None:
        warm_shared_insightface(cfg)
    return _SHARED_INSIGHT


def record_ocr_cache_hit() -> None:
    global _OCR_CACHE_HITS
    _OCR_CACHE_HITS += 1


def record_insightface_call() -> None:
    global _INSIGHTFACE_CALLS
    _INSIGHTFACE_CALLS += 1


def get_model_usage_stats() -> Dict[str, Any]:
    hits = int(_OCR_CACHE_HITS)
    infer = int(_OCR_INFERENCE_CALLS)
    total_ocr_events = hits + infer
    hit_rate = (hits / total_ocr_events) if total_ocr_events else 0.0
    saved = hits  # each hit avoids one redundant OCR pass on QA
    reduction = (saved / (infer + saved)) if (infer + saved) else 0.0
    return {
        "ocr_inference_calls": infer,
        "ocr_cache_hits": hits,
        "ocr_cache_hit_rate": float(hit_rate),
        "ocr_qa_calls_avoided": int(saved),
        "estimated_ocr_reduction_pct": float(reduction),
        "insightface_calls": int(_INSIGHTFACE_CALLS),
        "shared_paddleocr": _SHARED_OCR is not None,
        "shared_insightface": _SHARED_INSIGHT is not None,
    }


def clear_shared_models() -> None:
    """Reset singletons (tests only)."""
    global _SHARED_OCR, _SHARED_OCR_KEY, _SHARED_INSIGHT, _SHARED_INSIGHT_KEY
    global _OCR_INFERENCE_CALLS, _OCR_CACHE_HITS, _INSIGHTFACE_CALLS
    _SHARED_OCR = None
    _SHARED_OCR_KEY = None
    _SHARED_INSIGHT = None
    _SHARED_INSIGHT_KEY = None
    _OCR_INFERENCE_CALLS = 0
    _OCR_CACHE_HITS = 0
    _INSIGHTFACE_CALLS = 0


def resolve_dp2_repo_path(cfg: Dict[str, Any], project_root: Path) -> Path:
    dp = cfg.get("deep_privacy2", {}) or {}
    raw = (dp.get("repo_root") or "").strip()
    if not raw:
        return Path()
    p = Path(raw)
    if not p.is_absolute():
        p = (Path(project_root) / p).resolve()
    return p
