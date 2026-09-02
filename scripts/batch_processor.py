"""
Core per-batch anonymization logic.

Pipeline (per image):
1) Load RGB (Pillow, EXIF-aware) + SHA-256 of source bytes
2) InsightFace probe on input (embedding + count for QA; shared singleton)
3) DeepPrivacy2 GAN replacement for faces / person regions (when repo configured)
4) PaddleOCR on post-GAN frame → mask → LaMa inpainting (IOPaint preferred, simple_lama fallback)
5) Strip metadata (ExifTool primary; Pillow fallback)
6) Save ``{stem}.jpg`` + JSON audit (includes output SHA-256) under temp_processed/batch_XXXXX/

Notes on the exact stack requested:
- DeepPrivacy2 upstream installs as package name `dp2` and typically builds the anonymizer via
  `tops.config.instantiate` (see upstream `anonymize.py`). This module ALSO tries the project-spec
  import `from deep_privacy2 import Anonymizer` first; if your fork exposes that symbol, it will be used.
- PaddleOCR is invoked exactly as `PaddleOCR(use_angle_cls=True, lang="en", ...)` with extra kwargs
  controlled from config.yaml.
- LaMa is wired through **IOPaint** ``ModelManager`` when ``lama.backend: lama_cleaner`` (recommended),
  or ``simple-lama-inpainting`` / ``custom`` (advimman clone) per ``config.yaml``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .shared_models import (
    build_paddle_ocr,
    deserialize_ocr_boxes,
    get_shared_insightface_app,
    get_shared_paddle_ocr,
    ocr_boxes_max_score,
    ocr_polys_and_scores,
    record_insightface_call,
    resolve_dp2_repo_path,
    serialize_ocr_boxes,
    warm_shared_insightface,
    warm_shared_paddle_ocr,
)
from .processing_locks import (
    filter_unfinished_paths,
    mark_processing_complete,
    update_lock_completed_stems,
    write_processing_lock,
)
from .device_manager import (
    GpuFallbackRequired,
    get_compute_profile,
    handle_gpu_exception,
    is_cpu_fallback_mode,
    is_gpu_related_error,
)
from .gpu_runtime import (
    GPU_FALLBACK_USER_MESSAGE,
    SharedCudaContext,
    cuda_memory_snapshot,
    dataloader_cuda_kwargs,
    empty_cuda_cache_after_batch,
    inpaint_autocast_enabled,
    merge_warm_meta,
    resolve_gpu_config,
    warm_gpu_dummy_forward,
)
from .performance import resolve_num_workers
from .security import resolve_under
from .utils import (
    get_logger,
    imread_rgb,
    imwrite_rgb,
    load_audit_json,
    resolve_pipeline_paths,
    save_audit_json,
    sha256_file,
    strip_all_metadata,
    utc_now_iso,
)

logger = get_logger(__name__)


# =============================================================================
# DeepPrivacy2 integration (spec import + upstream instantiate fallback)
# =============================================================================


def _ensure_dp2_sys_path(repo_root: Path) -> None:
    rr = str(repo_root.resolve())
    if rr not in sys.path:
        sys.path.insert(0, rr)


def try_build_deep_privacy_anonymizer(
    cfg: Dict[str, Any],
    device_torch: torch.device,
    *,
    project_root: Optional[Path] = None,
):
    """
    Build the DeepPrivacy2 anonymizer object.

    Returns:
        (anonymizer_or_none, meta_dict)
    """
    dp_cfg = cfg.get("deep_privacy2", {}) or {}
    repo_path = resolve_dp2_repo_path(cfg, project_root or Path.cwd())
    if not str((dp_cfg.get("repo_root") or "").strip()):
        return None, {"status": "skipped", "reason": "deep_privacy2.repo_root is empty"}

    if not repo_path.is_dir():
        return None, {"status": "skipped", "reason": f"repo_root_not_found:{repo_path}"}

    _ensure_dp2_sys_path(repo_path)

    # 1) Project-spec import (may exist in some distributions / shims)
    try:
        from deep_privacy2 import Anonymizer  # type: ignore  # noqa: WPS433 — user-mandated import

        # If the class exists but construction differs, users can adapt here.
        anonymizer = Anonymizer(str(repo_path / (dp_cfg.get("config_rel") or "configs/anonymizers/face.py")))
        return anonymizer, {"status": "ok", "backend": "deep_privacy2.Anonymizer", "repo_path": str(repo_path)}
    except Exception as exc_dp2pkg:  # noqa: BLE001
        logger.info("deep_privacy2_import_fallback", error=str(exc_dp2pkg))

    # 2) Upstream hukkelas/deep_privacy2 pattern (see upstream `anonymize.py`)
    try:
        import tops  # type: ignore
        from dp2 import utils as dp2_utils  # type: ignore
        from tops.config import instantiate  # type: ignore

        cfg_path = repo_path / (dp_cfg.get("config_rel") or "configs/anonymizers/face.py")
        tops.set_seed(0)
        loaded = dp2_utils.load_config(str(cfg_path))
        loaded.detector.score_threshold = float(dp_cfg.get("detection_score_threshold", 0.35))
        anonymizer = instantiate(loaded.anonymizer, load_cache=bool(dp_cfg.get("load_detection_cache", False)))
        return anonymizer, {"status": "ok", "backend": "tops.instantiate(dp2)", "repo_path": str(repo_path)}
    except Exception as exc_inst:  # noqa: BLE001
        logger.error("deep_privacy2_build_failed", error=str(exc_inst))
        return None, {"status": "failed", "error": str(exc_inst)}


def _dp2_anonymize_rgb(
    rgb: np.ndarray,
    anonymizer: Any,
    dp_cfg: Dict[str, Any],
    synth_kwargs: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply DeepPrivacy2 GAN anonymization using the same tensor path as upstream ``anonymize_image``.

    The generator replaces detected regions with **synthetic** identities matched to pose/lighting
    (no blur boxes). We record synthesis parameters and geometry for the audit trail.
    """
    import tops  # type: ignore
    from dp2 import utils as dp2_utils  # type: ignore

    target_hw = (int(rgb.shape[0]), int(rgb.shape[1]))  # H, W full-resolution (post-decode)
    max_res = dp_cfg.get("max_resolution", None)
    im = Image.fromarray(rgb).convert("RGB")
    im = ImageOps.exif_transpose(im)

    resized_for_dp2 = False
    # Resize (mirrors upstream ``resize`` in anonymize.py)
    if max_res is not None:
        w, h = im.size
        f = max(w / max_res, h / max_res, 1.0)
        if f > 1.0:
            new_w, new_h = int(w / f), int(h / f)
            im = im.resize((new_w, new_h), Image.BILINEAR)
            resized_for_dp2 = True

    im_np = np.array(im, dtype=np.uint8)
    md5_ = hashlib.md5(im_np.tobytes()).hexdigest()
    im_t = dp2_utils.im2torch(im_np, to_float=False, normalize=False)[0]
    synth = dict(synth_kwargs)
    synth["cache_id"] = md5_

    dev = SharedCudaContext.device()
    if dev.type == "cuda":
        im_cuda = im_t.to(dev, non_blocking=True)
    else:
        im_cuda = im_t
    out_t = anonymizer(im_cuda, **synth)
    out_np = dp2_utils.im2numpy(out_t)

    # Restore full raster size so PaddleOCR masks align with the buyer-facing output dimensions.
    if out_np.shape[0] != target_hw[0] or out_np.shape[1] != target_hw[1]:
        out_np = cv2.resize(out_np, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)

    meta: Dict[str, Any] = {
        "dp2_md5": md5_,
        "dp2_output_shape": list(out_np.shape),
        "dp2_internal_shape": list(im_np.shape),
        "resized_for_dp2": resized_for_dp2,
        "max_resolution_config": max_res,
        "synthesis_params_used": {k: synth.get(k) for k in sorted(synth.keys()) if k != "cache_id"},
        "replacement_success": True,
        "backend": "tops.instantiate(dp2)",
    }
    return out_np, meta


def _dp2_shim_anonymize_rgb(rgb: np.ndarray, anonymizer: Any, dp_cfg: Dict[str, Any], synth_kwargs: Dict[str, Any]):
    """
    Prefer ``deep_privacy2.Anonymizer.anonymize_rgb`` when a numpy-friendly shim exists; otherwise use the
    official ``dp2`` + ``tops`` tensor path (same as upstream ``anonymize_image``).
    """
    if anonymizer is None:
        return rgb, {
            "status": "skipped",
            "reason": "anonymizer_not_initialized",
            "replacement_success": False,
            "synthesis_params_used": dict(synth_kwargs),
        }

    target_hw = (int(rgb.shape[0]), int(rgb.shape[1]))
    fn = getattr(anonymizer, "anonymize_rgb", None)
    if callable(fn):
        try:
            out = fn(rgb, **synth_kwargs)
            out_arr = np.asarray(out, dtype=np.uint8)
            if out_arr.shape[0] != target_hw[0] or out_arr.shape[1] != target_hw[1]:
                out_arr = cv2.resize(out_arr, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_LINEAR)
            return out_arr, {
                "backend": "deep_privacy2.Anonymizer.anonymize_rgb",
                "replacement_success": True,
                "synthesis_params_used": dict(synth_kwargs),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("anonymize_rgb_failed_falling_back", error=str(exc))

    try:
        out_np, meta = _dp2_anonymize_rgb(rgb, anonymizer, dp_cfg, synth_kwargs)
        return out_np, meta
    except Exception as exc:  # noqa: BLE001
        logger.error("dp2_tensor_path_failed", error=str(exc))
        return rgb, {
            "status": "error",
            "error": str(exc),
            "replacement_success": False,
            "synthesis_params_used": dict(synth_kwargs),
        }


# =============================================================================
# Masks + LaMa
# =============================================================================


def boxes_to_mask(shape_hw: Tuple[int, int], boxes: Sequence[Dict[str, Any]], dilate: int) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes:
        poly = np.round(b["polygon"]).astype(np.int32)
        cv2.fillPoly(mask, [poly], 255)
    if dilate > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        mask = cv2.dilate(mask, k)
    return mask


class LamaBackend:
    """
    Unified LaMa façade supporting:

    * ``lama_cleaner`` — in-process **IOPaint** ``ModelManager`` (lama-cleaner successor; downloads weights on first use).
    * ``simple_lama`` — pip-friendly ``simple-lama-inpainting`` (Big LaMa ONNX/Torch export).
    * ``custom`` — import hook for an **advimman/lama** checkout or bespoke wrapper.
    * ``none`` — skip inpaint (audit will record the skip; not recommended for PII sale bundles).
    """

    def __init__(self, cfg: Dict[str, Any], device_torch: torch.device):
        self.root_cfg = cfg
        self.cfg = cfg.get("lama", {}) or {}
        self.backend = str(self.cfg.get("backend", "lama_cleaner")).lower()
        self.device_torch = device_torch
        self._simple = None  # SimpleLama instance
        self._custom = None  # user factory product
        self._iopaint_mm = None  # iopaint.model_manager.ModelManager

    def warm(self) -> Dict[str, Any]:
        if self.backend == "none":
            return {"backend": "none"}
        if self.backend == "cpu_redaction":
            return {
                "backend": "cpu_redaction",
                "status": "ok",
                "note": "opencv_inpaint_for_text_no_gan",
                "user_notice": GPU_FALLBACK_USER_MESSAGE,
            }

        dev_str = self.cfg.get("device", None)
        if dev_str is None:
            dev_str = str(self.device_torch)
        elif str(dev_str).startswith("cuda") and self.device_torch.type != "cuda":
            dev_str = "cpu"

        if self.backend == "lama_cleaner":
            try:
                import torch
                from iopaint.model_manager import ModelManager

                name = str(self.cfg.get("iopaint_model_name", "lama"))
                dev = SharedCudaContext.device()
                if str(dev_str).startswith("cuda") and dev.type == "cuda":
                    dev = self.device_torch
                else:
                    dev = torch.device("cpu")
                self._iopaint_mm = ModelManager(name, device=dev)
                return {"backend": "lama_cleaner", "iopaint_model": name, "device": str(dev)}
            except Exception as exc:  # noqa: BLE001
                logger.error("iopaint_lama_failed_try_simple_lama", error=str(exc))
                self._iopaint_mm = None
                # Automatic degradation keeps partial installs usable; audit flags the fallback.
                self.backend = "simple_lama"
                return self._warm_simple_lama(dev_str, note="fallback_from_iopaint_error")

        if self.backend == "simple_lama":
            return self._warm_simple_lama(dev_str)

        if self.backend == "custom":
            mod = self.cfg.get("custom_module", "")
            qual = self.cfg.get("custom_callable", "")
            if not mod or not qual:
                return {"backend": "custom", "status": "skipped", "reason": "custom_module/custom_callable not set"}
            import importlib

            module = importlib.import_module(str(mod))
            factory = getattr(module, str(qual))
            self._custom = factory(self.cfg)
            return {"backend": "custom", "import": f"{mod}:{qual}"}

        return {"backend": self.backend, "status": "unknown_backend"}

    def _warm_simple_lama(self, dev_str: str, note: str | None = None) -> Dict[str, Any]:
        try:
            from simple_lama_inpainting import SimpleLama  # type: ignore

            device = str(dev_str)
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self._simple = SimpleLama(device=device)
            meta: Dict[str, Any] = {"backend": "simple_lama", "device": str(dev_str)}
            if note:
                meta["note"] = note
            return meta
        except Exception as exc:  # noqa: BLE001
            logger.error("simple_lama_import_failed", error=str(exc))
            return {"backend": "simple_lama", "status": "failed", "error": str(exc)}

    def _build_iopaint_request(self, extras: Optional[Dict[str, Any]] = None):
        from iopaint.schema import InpaintRequest

        base = dict(self.cfg.get("iopaint_request") or {})
        if extras:
            base.update(extras)
        if not base:
            return InpaintRequest()
        req = InpaintRequest()
        try:
            return req.model_copy(update=base)  # type: ignore[attr-defined]  # pydantic v2
        except Exception:  # noqa: BLE001
            try:
                return req.copy(update=base)  # type: ignore[attr-defined]  # pydantic v1
            except Exception:  # noqa: BLE001
                try:
                    return InpaintRequest(**base)  # type: ignore[call-arg]
                except Exception as exc:  # noqa: BLE001
                    logger.warning("iopaint_request_invalid_using_defaults", error=str(exc))
                    return InpaintRequest()

    def inpaint_rgb(
        self,
        rgb: np.ndarray,
        mask_u8: np.ndarray,
        *,
        inpaint_extras: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inpaint masked regions (``255`` = repaint). Returns uint8 RGB + meta dict for audits.
        """
        if self.backend == "none":
            return rgb, {"status": "skipped", "reason": "lama_disabled"}

        if self.backend == "cpu_redaction":
            return apply_cpu_text_redaction(rgb, mask_u8, self.root_cfg)

        if self.backend == "lama_cleaner" and self._iopaint_mm is not None:
            from .lama_iopaint import run_iopaint_inpaint

            req = self._build_iopaint_request(inpaint_extras)
            return run_iopaint_inpaint(self._iopaint_mm, rgb, mask_u8, req, cfg=self.root_cfg)

        if self.backend == "simple_lama" and self._simple is not None:
            from PIL import Image

            img_pil = Image.fromarray(rgb)
            mask_pil = Image.fromarray(mask_u8)
            with torch.inference_mode():
                out = self._simple(img_pil, mask_pil)
            out_rgb = np.asarray(out.convert("RGB"), dtype=np.uint8)
            return out_rgb, {"status": "ok", "backend": "simple_lama"}

        if self.backend == "custom" and self._custom is not None:
            with torch.inference_mode():
                out = self._custom(rgb, mask_u8)  # type: ignore[misc]
            return np.asarray(out), {"status": "ok", "backend": "custom"}

        return rgb, {"status": "skipped", "reason": f"lama_unavailable:{self.backend}"}


# =============================================================================
# Optional InsightFace probe (original embedding for identity-distance QA)
# =============================================================================


class InsightFaceProbe:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg.get("insightface", {}) or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self._app = None

    def warm(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        meta = warm_shared_insightface(cfg)
        self._app = get_shared_insightface_app(cfg)
        if self._app is None:
            self.enabled = False
        return meta

    def probe_original(self, bgr: np.ndarray) -> Dict[str, Any]:
        if not self.enabled or self._app is None:
            return {}
        record_insightface_call()
        faces = self._app.get(bgr)
        if not faces:
            return {"original_face_count": 0}
        emb = faces[0].embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return {
            "original_face_count": int(len(faces)),
            "original_primary_face_embedding": emb.tolist(),
        }

    def count_faces(self, bgr: np.ndarray) -> int:
        """Fast face count on a BGR frame (reuses warmed FaceAnalysis). Used post-GAN for audit metrics."""
        if not self.enabled or self._app is None:
            return 0
        try:
            record_insightface_call()
            return int(len(self._app.get(bgr) or []))
        except Exception:  # noqa: BLE001
            return 0


def _opencv_blur_bbox(rgb: np.ndarray, bbox: Sequence[float], *, ksize: int = 51) -> None:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = rgb[y1:y2, x1:x2]
    k = max(3, int(ksize) | 1)
    rgb[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def apply_cpu_privacy_blur(
    rgb: np.ndarray,
    bgr: np.ndarray,
    insight: InsightFaceProbe,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CPU fallback for face/body regions: OpenCV Gaussian blur on InsightFace boxes (no GAN).
    """
    out = rgb.copy()
    meta: Dict[str, Any] = {
        "backend": "opencv_gaussian_blur",
        "status": "ok",
        "replacement_success": False,
        "face_regions_blurred": 0,
        "user_notice": GPU_FALLBACK_USER_MESSAGE,
    }
    fb = (cfg.get("cpu_fallback") or {})
    ksize = int(fb.get("face_blur_ksize", 51))
    if not insight.enabled or insight._app is None:
        meta["reason"] = "insightface_unavailable"
        return out, meta
    try:
        record_insightface_call()
        faces = insight._app.get(bgr) or []
        for face in faces:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                continue
            _opencv_blur_bbox(out, bbox, ksize=ksize)
            meta["face_regions_blurred"] = int(meta["face_regions_blurred"]) + 1
        meta["replacement_success"] = meta["face_regions_blurred"] > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("cpu_face_blur_failed", error=str(exc))
        meta["status"] = "error"
        meta["error"] = str(exc)
    return out, meta


def apply_cpu_text_redaction(rgb: np.ndarray, mask_u8: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    CPU fallback for text: OpenCV inpaint (no IOPaint/LaMa GAN).
    """
    fb = (cfg.get("cpu_fallback") or {})
    radius = int(fb.get("text_inpaint_radius", 5))
    if mask_u8.max() < 1:
        return rgb, {"status": "skipped", "reason": "empty_mask", "backend": "opencv_inpaint"}
    try:
        bgr = rgb[:, :, ::-1]
        out_bgr = cv2.inpaint(bgr, mask_u8, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
        out_rgb = out_bgr[:, :, ::-1]
        return out_rgb, {"status": "ok", "backend": "opencv_inpaint", "inpaint_radius": radius}
    except Exception as exc:  # noqa: BLE001
        logger.error("cpu_text_redaction_failed", error=str(exc))
        return rgb, {"status": "error", "backend": "opencv_inpaint", "error": str(exc)}


# =============================================================================
# PyTorch DataLoader (paths -> decoded RGB batches)
# =============================================================================


class ImagePathDataset(Dataset):
    def __init__(self, paths: Sequence[Path]):
        self.paths = [Path(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Path:
        return self.paths[idx]


def parallel_imread_rgb(paths: Sequence[Path], n_jobs: int) -> List[Optional[np.ndarray]]:
    """
    Parallel decode of RGB tensors (after optional security copy). Keeps ``joblib`` in the hot path
    without multiprocessing DataLoader workers (friendlier to Paddle/GPU runtimes on Windows).
    """
    paths = [Path(p) for p in paths]
    if not paths:
        return []

    def _load(p: Path) -> Optional[np.ndarray]:
        try:
            return imread_rgb(p)
        except Exception as exc:  # noqa: BLE001
            logger.error("parallel_imread_failed", path=str(p), error=str(exc))
            return None

    if len(paths) == 1:
        return [_load(paths[0])]

    n_jobs = max(1, min(int(n_jobs), len(paths)))
    return list(joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(_load)(p) for p in paths))


def make_path_dataloader(
    paths: Sequence[Path],
    batch_size: int,
    num_workers: int,
    cfg: Optional[Dict[str, Any]] = None,
) -> DataLoader:
    """
    Path-only batches: images are **copied** (optional) then decoded in the main process so we never
    open ``input_raw/`` for write. ``num_workers`` is reserved for ``joblib`` decode parallelism.

    When CUDA is active, enables ``pin_memory`` (and prefetch metadata) for faster host staging.
    """

    def _collate(batch_paths: List[Path]) -> List[Path]:
        return [Path(p) for p in batch_paths]

    dl_kwargs: Dict[str, Any] = {"pin_memory": False}
    if cfg is not None:
        dl_kwargs = dataloader_cuda_kwargs(cfg, num_workers=0)

    return DataLoader(
        ImagePathDataset(paths),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
        **dl_kwargs,
    )


# =============================================================================
# Batch engine
# =============================================================================


@dataclass
class BatchProcessResult:
    batch_dir: Path
    rows: List[Dict[str, Any]]


class AnonymizationEngine:
    """
    Holds long-lived model objects (OCR, LaMa, DeepPrivacy2) and runs one batch folder.
    """

    def __init__(self, project_root: Path, cfg: Dict[str, Any]):
        self.project_root = Path(project_root)
        self.cfg = cfg
        self.gpu_cfg = resolve_gpu_config(cfg)
        self.cpu_fallback = is_cpu_fallback_mode()
        self.compute_profile = get_compute_profile()
        self.device_torch = SharedCudaContext.configure(cfg)
        if self.gpu_cfg.wants_cuda and self.device_torch.type != "cuda":
            logger.warning("cuda_requested_but_unavailable", fallback="cpu")
        if self.cpu_fallback:
            logger.warning(
                "anonymization_engine_cpu_fallback",
                message=self.compute_profile.get("user_message") or GPU_FALLBACK_USER_MESSAGE,
            )

        self.ocr: Optional[Any] = None
        self.dp_anonymizer = None
        self.dp_meta: Dict[str, Any] = {}
        self.lama = LamaBackend(cfg, self.device_torch)
        self.insight = InsightFaceProbe(cfg)

    def apply_cpu_fallback_mode(self, reason: str, *, component: str = "runtime") -> None:
        """Rebind engine after automatic GPU→CPU downgrade."""
        from .device_manager import activate_cpu_fallback
        from .shared_models import clear_shared_models

        activate_cpu_fallback(self.cfg, reason, component=component, announce=True)
        clear_shared_models()
        self.cpu_fallback = True
        self.compute_profile = get_compute_profile()
        self.gpu_cfg = resolve_gpu_config(self.cfg)
        self.device_torch = SharedCudaContext.configure(self.cfg)
        self.lama = LamaBackend(self.cfg, self.device_torch)
        self.lama.backend = "cpu_redaction"
        self.dp_anonymizer = None
        self.dp_meta = {
            "status": "disabled",
            "reason": "cpu_fallback_mode",
            "user_notice": self.compute_profile.get("user_message"),
        }
        self.ocr = None
        try:
            warm_shared_paddle_ocr(self.cfg)
            self.ocr = get_shared_paddle_ocr(self.cfg)
            self.insight.warm(self.cfg)
        except Exception as exc:  # noqa: BLE001
            logger.error("cpu_fallback_rewarm_failed", error=str(exc))

    def _warm_cpu_stack(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"compute_profile": dict(self.compute_profile)}
        meta["paddleocr"] = warm_shared_paddle_ocr(self.cfg)
        self.ocr = get_shared_paddle_ocr(self.cfg)
        self.dp_anonymizer = None
        self.dp_meta = {
            "status": "disabled",
            "reason": "cpu_fallback_mode",
            "user_notice": self.compute_profile.get("user_message"),
        }
        meta["deep_privacy2"] = dict(self.dp_meta)
        self.lama.backend = "cpu_redaction"
        meta["lama"] = self.lama.warm()
        meta["insightface_probe"] = self.insight.warm(self.cfg)
        meta["gpu"] = merge_warm_meta(cuda_memory_snapshot())
        return meta

    def _warm_gpu_stack(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"compute_profile": dict(self.compute_profile)}

        try:
            meta["paddleocr"] = warm_shared_paddle_ocr(self.cfg)
            self.ocr = get_shared_paddle_ocr(self.cfg)
        except Exception as exc:  # noqa: BLE001
            if is_gpu_related_error(exc):
                raise GpuFallbackRequired(str(exc), component="paddleocr") from exc
            raise

        try:
            meta["insightface_probe"] = self.insight.warm(self.cfg)
        except Exception as exc:  # noqa: BLE001
            if is_gpu_related_error(exc):
                raise GpuFallbackRequired(str(exc), component="insightface") from exc
            meta["insightface_probe"] = {"enabled": False, "error": str(exc)}

        try:
            self.dp_anonymizer, self.dp_meta = try_build_deep_privacy_anonymizer(
                self.cfg,
                self.device_torch,
                project_root=self.project_root,
            )
        except Exception as exc:  # noqa: BLE001
            if is_gpu_related_error(exc):
                raise GpuFallbackRequired(str(exc), component="deep_privacy2") from exc
            logger.error("deep_privacy2_init_exception", error=str(exc))
            self.dp_anonymizer = None
            self.dp_meta = {"status": "failed", "error": str(exc)}

        meta["deep_privacy2"] = dict(self.dp_meta)
        if self.dp_meta.get("status") == "failed":
            dp_err = str(self.dp_meta.get("error", ""))
            if is_gpu_related_error(RuntimeError(dp_err)):
                raise GpuFallbackRequired(dp_err, component="deep_privacy2")

        if self.dp_meta.get("status") == "skipped":
            resolved = resolve_dp2_repo_path(self.cfg, self.project_root)
            logger.warning(
                "deep_privacy2_not_ready",
                repo_root=str(resolved),
                reason=self.dp_meta.get("reason"),
            )

        try:
            meta["lama"] = self.lama.warm()
            self.lama.backend = str(meta["lama"].get("backend", self.lama.backend))
            lama_failed = meta["lama"].get("status") == "failed"
            lama_err = str(meta["lama"].get("error", ""))
            if lama_failed and is_gpu_related_error(RuntimeError(lama_err)):
                raise GpuFallbackRequired(lama_err, component="iopaint_lama")
            if (
                self.lama.backend == "lama_cleaner"
                and self.lama._iopaint_mm is None
                and is_gpu_related_error(RuntimeError(lama_err))
            ):
                raise GpuFallbackRequired(lama_err or "iopaint GPU load failed", component="iopaint_lama")
        except GpuFallbackRequired:
            raise
        except Exception as exc:  # noqa: BLE001
            if is_gpu_related_error(exc):
                raise GpuFallbackRequired(str(exc), component="iopaint_lama") from exc
            meta["lama"] = {"status": "failed", "error": str(exc)}
            self.lama.backend = "cpu_redaction"
            meta["lama"] = self.lama.warm()

        meta["gpu"] = merge_warm_meta(
            cuda_memory_snapshot(),
            warm_gpu_dummy_forward(self.cfg),
        )
        return meta

    def warm_models(self) -> Dict[str, Any]:
        """
        Load heavy models once per process. Tries GPU stack first; any GPU failure → CPU fallback.
        """
        logger.info("warming_models", cpu_fallback=bool(self.cpu_fallback))
        meta: Dict[str, Any] = {}

        if self.cpu_fallback:
            meta = self._warm_cpu_stack()
            empty_cuda_cache_after_batch(self.cfg, label="post_model_warm_cpu")
            return meta

        try:
            meta = self._warm_gpu_stack()
        except GpuFallbackRequired as exc:
            logger.warning("gpu_warm_failed_switching_to_cpu", component=exc.component, reason=exc.reason)
            self.apply_cpu_fallback_mode(exc.reason, component=exc.component)
            meta = self._warm_cpu_stack()
            meta["gpu_fallback_triggered_at_warm"] = {
                "component": exc.component,
                "reason": exc.reason,
                "user_message": self.compute_profile.get("user_message"),
            }
        except Exception as exc:  # noqa: BLE001
            if handle_gpu_exception(self.cfg, exc, component="model_warm"):
                self.apply_cpu_fallback_mode(str(exc), component="model_warm")
                meta = self._warm_cpu_stack()
                meta["gpu_fallback_triggered_at_warm"] = {"reason": str(exc)}
            else:
                raise

        empty_cuda_cache_after_batch(self.cfg, label="post_model_warm")
        return meta


def process_batch(
    paths: Sequence[Path],
    *,
    batch_index: int,
    engine: AnonymizationEngine,
    cfg: Dict[str, Any],
    project_root: Path,
) -> BatchProcessResult:
    """
    Process a list of input paths into `temp_processed/batch_{batch_index:05d}/`.

    Each image produces:
    - `<stem>.jpg` (normalized container for sale; adjust if you need lossless PNG)
    - `<stem>.json` audit sidecar
    """
    project_root = Path(project_root)
    pipeline_paths = resolve_pipeline_paths(project_root, cfg)
    image_paths = filter_unfinished_paths(
        pipeline_paths["temp_processed"] / f"batch_{batch_index:05d}",
        [Path(p) for p in paths],
    )
    if not image_paths:
        batch_dir = pipeline_paths["temp_processed"] / f"batch_{batch_index:05d}"
        return BatchProcessResult(batch_dir=batch_dir, rows=[])
    batch_dir = pipeline_paths["temp_processed"] / f"batch_{batch_index:05d}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    batch_stems = [Path(p).stem for p in paths]
    write_processing_lock(
        batch_dir,
        batch_index=batch_index,
        stems=batch_stems,
        source_paths=[str(Path(p).as_posix()) for p in paths],
    )

    rows: List[Dict[str, Any]] = []
    dp_cfg = cfg.get("deep_privacy2", {}) or {}
    synth_kwargs = dict(dp_cfg.get("synthesis", {}) or {})

    quarantine_resolved = pipeline_paths["quarantine"].resolve()
    manual_review_resolved = pipeline_paths["manual_review"].resolve()

    # Merge retry adjustments if sidecar says this is a recycle pass
    def per_image_runtime_overrides(path: Path) -> Dict[str, Any]:
        if path.parent.resolve() not in {quarantine_resolved, manual_review_resolved}:
            return {}
        audit_prev = load_audit_json(path)
        return audit_prev.get("next_pass_overrides", {}) or {}

    def read_retry_count(path: Path) -> int:
        js = path.with_suffix(".json")
        if not js.is_file():
            return 0
        try:
            return int(load_audit_json(path).get("retry_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    n_workers = resolve_num_workers(cfg)
    dl = make_path_dataloader(image_paths, int(cfg.get("batch_size", 32)), n_workers, cfg=cfg)
    sec = cfg.get("security", {}) or {}
    copy_input_raw = bool(sec.get("copy_input_raw", True))
    mirror = batch_dir / "_source_copies"

    for batch_paths in tqdm(dl, desc=f"batch_{batch_index:05d}", leave=False):
        batch_paths = [Path(p) for p in batch_paths]
        work_paths: List[Path] = []
        for src in batch_paths:
            if copy_input_raw:
                mirror.mkdir(parents=True, exist_ok=True)
                dest = mirror / src.name
                try:
                    shutil.copy2(src, dest)
                    work_paths.append(dest)
                except OSError as exc:
                    logger.error("security_copy_failed_readonly_fallback", src=str(src), error=str(exc))
                    work_paths.append(src)
            else:
                work_paths.append(src)

        batch_rgbs = parallel_imread_rgb(work_paths, n_workers)

        for src_path, work_path, rgb in zip(batch_paths, work_paths, batch_rgbs):
            if rgb is None or rgb.size == 0:
                logger.error("batch_item_decode_failed", image=str(src_path))
                err_audit = {
                    "source_path": str(src_path.as_posix()),
                    "batch_id": f"batch_{batch_index:05d}",
                    "failure_reason": "corrupted_or_unreadable_image",
                    "success_metrics": {"processing_status": "failed"},
                }
                err_json = batch_dir / f"{src_path.stem}.error_audit.json"
                err_json.write_text(json.dumps(err_audit, indent=2), encoding="utf-8")
                rows.append({"image": src_path.name, "batch_id": f"batch_{batch_index:05d}", "error": "decode_failed"})
                continue

            audit: Dict[str, Any] = {
                "created_at": utc_now_iso(),
                "source_path": str(src_path.as_posix()),
                "decode_path": str(work_path.as_posix()),
                "batch_id": f"batch_{batch_index:05d}",
                "stages": [],
            }

            try:
                overrides = per_image_runtime_overrides(src_path)
                if overrides:
                    audit.setdefault("recycling", {})
                    audit["recycling"]["overrides_applied"] = overrides

                audit["integrity_hashes"] = {
                    "source_sha256": sha256_file(work_path),
                    "source_path_for_hash": str(work_path.as_posix()),
                }

                # --- Faces: InsightFace probe on INPUT (embedding + count) for downstream QA ---
                bgr = rgb[:, :, ::-1].copy()
                probe = engine.insight.probe_original(bgr)
                input_face_count = int(probe.get("original_face_count", 0) or 0) if probe else 0
                if probe:
                    audit.setdefault("qa_probes", {}).update(probe)
                    audit.setdefault("success_metrics", {})["original_face_count"] = input_face_count

                # --- Face anonymization: DP2 GAN (GPU) or OpenCV blur (CPU fallback) ---
                merged_synth = {**synth_kwargs, **overrides.get("synthesis", {})}
                if engine.cpu_fallback:
                    gan_rgb, gan_info = apply_cpu_privacy_blur(rgb, bgr, engine.insight, cfg)
                    gan_info.setdefault("synthesis_params_used", {})
                    gan_bgr = gan_rgb[:, :, ::-1].copy()
                    post_dp2_faces = engine.insight.count_faces(gan_bgr)
                    gan_info["post_deep_privacy2_face_count_insight"] = int(post_dp2_faces)
                    stage_name = "cpu_face_blur"
                else:
                    with torch.inference_mode():
                        gan_rgb, gan_info = _dp2_shim_anonymize_rgb(
                            rgb,
                            engine.dp_anonymizer,
                            dp_cfg,
                            merged_synth,
                        )
                    gan_info.setdefault("synthesis_params_used", merged_synth)
                    gan_bgr = gan_rgb[:, :, ::-1].copy()
                    post_dp2_faces = engine.insight.count_faces(gan_bgr)
                    gan_info["post_deep_privacy2_face_count_insight"] = int(post_dp2_faces)
                    if gan_info.get("status") == "skipped" or engine.dp_anonymizer is None:
                        gan_info["replacement_success"] = False
                    elif gan_info.get("status") == "error":
                        gan_info["replacement_success"] = False
                    elif gan_info.get("replacement_success") is None:
                        gan_info["replacement_success"] = True
                    stage_name = "deep_privacy2"

                audit["stages"].append({"name": stage_name, "info": gan_info})
                audit.setdefault("success_metrics", {})["face_gan_applied"] = (
                    engine.dp_anonymizer is not None and not engine.cpu_fallback
                )
                audit.setdefault("success_metrics", {})["cpu_fallback_active"] = bool(engine.cpu_fallback)
                audit.setdefault("success_metrics", {})["deep_privacy2_replacement_success"] = bool(
                    gan_info.get("replacement_success", False)
                )

                # --- Text: PaddleOCR on post-GAN frame, then LaMa inpaint (photorealistic erase) ---
                ocr = engine.ocr
                assert ocr is not None
                ocr_boxes = ocr_polys_and_scores(ocr, gan_bgr)
                dilate = int((cfg.get("lama", {}) or {}).get("mask_dilation", 12)) + int(
                    overrides.get("extra_mask_dilation", 0)
                )
                mask = boxes_to_mask(gan_rgb.shape[:2], ocr_boxes, dilate=dilate)
                n_text = int(len(ocr_boxes))
                audit["detections"] = {
                    "faces": {
                        "input_face_count": input_face_count,
                        "post_deep_privacy2_face_count": int(post_dp2_faces),
                        "insightface_enabled": bool(engine.insight.enabled),
                    },
                    "text": {
                        "paddleocr_boxes": ocr_boxes,
                        "paddleocr_count": n_text,
                        "mask_dilation_px": int(dilate),
                    },
                    # Back-compat flat keys for older reporting scripts
                    "paddleocr_boxes": ocr_boxes,
                    "paddleocr_count": n_text,
                }
                audit.setdefault("success_metrics", {})["text_regions_detected_for_inpaint"] = n_text

                lama_extras = dict(overrides.get("lama_iopaint") or {})
                with torch.inference_mode():
                    inpainted, lama_meta = engine.lama.inpaint_rgb(gan_rgb, mask, inpaint_extras=lama_extras or None)
                audit["stages"].append({"name": "lama", "info": lama_meta})
                inpaint_ok = bool(
                    lama_meta.get("ok")
                    or lama_meta.get("status") == "ok"
                    or (lama_meta.get("status") == "skipped" and n_text == 0)
                )
                audit.setdefault("success_metrics", {})["text_regions_inpainted"] = n_text
                audit.setdefault("success_metrics", {})["inpaint_success"] = inpaint_ok

                # Cache output OCR for QA (reuse pre-inpaint scan when no text detected)
                output_bgr = inpainted[:, :, ::-1].copy()
                if n_text == 0:
                    output_ocr_boxes = ocr_boxes
                else:
                    output_ocr_boxes = ocr_polys_and_scores(ocr, output_bgr)
                audit["qa_cache"] = {
                    "output_ocr_boxes": serialize_ocr_boxes(output_ocr_boxes),
                    "output_ocr_max_score": float(ocr_boxes_max_score(output_ocr_boxes)),
                    "output_ocr_count": int(len(output_ocr_boxes)),
                    "ocr_reuse_for_qa": True,
                }

                # Metadata strip (writes final raster; never touches input_raw/)
                out_img_path = batch_dir / f"{src_path.stem}.jpg"
                imwrite_rgb(out_img_path, inpainted)
                meta_strip = strip_all_metadata(
                    out_img_path,
                    out_img_path,
                    exiftool_binary=str((cfg.get("metadata", {}) or {}).get("exiftool_binary", "exiftool")),
                    prefer_exiftool=bool((cfg.get("metadata", {}) or {}).get("prefer_exiftool", True)),
                )
                audit["stages"].append({"name": "metadata_strip", "info": meta_strip})
                audit["integrity_hashes"]["output_sha256"] = sha256_file(out_img_path)

                lcfg = cfg.get("lama", {}) or {}
                if engine.cpu_fallback:
                    face_action = "opencv_blur"
                    text_action = "opencv_inpaint"
                    order = ["cpu_face_blur", "paddleocr_detect", "cpu_text_redaction", "metadata_strip"]
                else:
                    face_action = "deep_privacy2" if engine.dp_anonymizer is not None else "skipped"
                    text_action = "lama" if str(lcfg.get("backend", "lama_cleaner")).lower() != "none" else "skipped"
                    order = ["deep_privacy2_face_gan", "paddleocr_detect", "lama_inpaint", "metadata_strip"]
                audit["actions"] = {
                    "order": order,
                    "face_gan": face_action,
                    "text_inpaint": text_action,
                    "lama_backend_resolved": str(lama_meta.get("backend", lcfg.get("backend", ""))),
                    "metadata_strip_method": str(meta_strip.get("method", "")),
                    "cpu_fallback_notice": GPU_FALLBACK_USER_MESSAGE if engine.cpu_fallback else None,
                }
                audit["failure_reason"] = None
                audit["retry_count"] = read_retry_count(src_path)
                audit.setdefault("success_metrics", {})["processing_status"] = "completed"
                audit["updated_at"] = utc_now_iso()

                save_audit_json(out_img_path, audit)
                update_lock_completed_stems(batch_dir, [src_path.stem])

                rows.append(
                    {
                        "image": out_img_path.name,
                        "batch_id": audit["batch_id"],
                        "ocr_boxes": int(len(ocr_boxes)),
                        "deep_privacy2": gan_info,
                        "metadata": meta_strip,
                    }
                )

            except Exception as exc:  # noqa: BLE001
                if not engine.cpu_fallback and handle_gpu_exception(
                    cfg, exc, component=f"batch_item:{src_path.name}"
                ):
                    engine.apply_cpu_fallback_mode(str(exc), component="batch_inference")
                    logger.warning(
                        "batch_continues_after_gpu_fallback",
                        image=str(src_path.name),
                        message=engine.compute_profile.get("user_message"),
                    )
                logger.error("batch_item_failed", image=str(src_path), error=str(exc))
                audit["stages"].append({"name": "error", "error": str(exc)})
                audit["actions"] = {"status": "failed"}
                audit.setdefault("success_metrics", {})["processing_status"] = "failed"
                err_json = batch_dir / f"{src_path.stem}.error_audit.json"
                err_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
                rows.append({"image": src_path.name, "batch_id": f"batch_{batch_index:05d}", "error": str(exc)})

            finally:
                empty_cuda_cache_after_batch(cfg, label=f"batch_{batch_index:05d}_item")

    empty_cuda_cache_after_batch(cfg, label=f"batch_{batch_index:05d}_complete")
    return BatchProcessResult(batch_dir=batch_dir, rows=rows)
