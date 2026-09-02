"""
CrewAI multi-agent QA layer + deterministic scoring (buyer-grade auditability).

Why both?
- Computer-vision QA gates should be reproducible without an LLM. The functions in this module
  implement the numeric checks (OCR re-scan, SSIM / edge heuristics, optional identity distance).
- CrewAI is still used to orchestrate the specialist agents and to produce an optional natural-language
  rationale when an API key is configured.

Agents (CrewAI):
1) Detection Verification Agent — verifies anonymized outputs using PaddleOCR (+ optional DP2 hooks)
2) Identity & Integrity Agent — optional InsightFace distance + OpenCV / scikit-image integrity checks
3) Decision Agent — pass/fail + routing (final_clean vs quarantine vs manual_review)

Tooling:
- We bind Python callables as Crew tools when available; if tool decoration fails, the orchestrator
  falls back to deterministic evaluation only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

from .shared_models import (
    deserialize_ocr_boxes,
    get_shared_insightface_app,
    get_shared_paddle_ocr,
    ocr_polys_and_scores,
    record_insightface_call,
    record_ocr_cache_hit,
)
from .utils import get_logger, imread_rgb, load_audit_json, save_audit_json, utc_now_iso

logger = get_logger(__name__)


def _get_cached_insight_app(cfg: Dict[str, Any]):
    """InsightFace app shared with batch processing (see ``shared_models``)."""
    return get_shared_insightface_app(cfg)


# =============================================================================
# Deterministic QA primitives (these are the “source of truth” for pass/fail)
# =============================================================================


def run_detection_verification(
    image_path: Path,
    cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
    *,
    ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    QA pass #1 — **Detection Verification** (spec: re-run PaddleOCR; optional face probes).

    DeepPrivacy2 does not expose a lightweight “detector-only” API in all installs, so we record
    ``deep_privacy2_qa`` as informational. Text risk is gated on **PaddleOCR** re-scores on the output.
    """
    qa = cfg.get("qa", {}) or {}
    thr = float(qa.get("text_det_score_fail", 0.85))

    if bgr is None:
        bgr = imread_rgb(image_path)[:, :, ::-1].copy()
    if ocr_boxes is None:
        ocr = get_shared_paddle_ocr(cfg)
        ocr_boxes = ocr_polys_and_scores(ocr, bgr)

    max_score = 0.0
    count = int(len(ocr_boxes))
    for box in ocr_boxes:
        try:
            score = float(box["score"])
            max_score = max(max_score, score)
        except Exception:  # noqa: BLE001
            continue

    fail = max_score >= thr and count > 0

    out: Dict[str, Any] = {
        "ocr_lines": int(count),
        "max_det_score": float(max_score),
        "threshold": float(thr),
        "fail": bool(fail),
        "reason": ("readable_text_likely_remaining" if fail else "no_high_confidence_text_detections"),
        "deep_privacy2_qa": {
            "note": "Full DP2 re-inference for QA requires the training anonymizer in-process; use audit success_metrics.face_gan_applied + identity checks.",
        },
    }

    # Optional InsightFace face count on OUTPUT (SCRFD family inside FaceAnalysis)
    out_face_count = 0
    app = _get_cached_insight_app(cfg)
    if app is not None:
        try:
            record_insightface_call()
            faces = app.get(bgr)
            out_face_count = int(len(faces))
        except Exception as exc:  # noqa: BLE001
            out["insightface_face_count_error"] = str(exc)

    out["verification_output_face_count"] = out_face_count
    warn_cap = int(qa.get("max_output_faces_warn", 20) or 20)
    out["output_face_count_warn"] = bool(out_face_count > warn_cap)

    # Optional Ultralytics YOLOv8-face verification (independent second opinion)
    if bool(qa.get("ultralytics_face_verify", False)) and bool((cfg.get("ultralytics", {}) or {}).get("enabled", False)):
        root = project_root or Path.cwd()
        w = Path(str((cfg.get("ultralytics", {}) or {}).get("face_weights", "yolov8n.pt")))
        if not w.is_absolute():
            w = root / w
        if w.is_file():
            try:
                from ultralytics import YOLO

                model = YOLO(str(w))
                r = model.predict(bgr, verbose=False)[0]
                n = 0 if r.boxes is None else int(len(r.boxes))
                out["ultralytics_face_boxes"] = n
            except Exception as exc:  # noqa: BLE001
                out["ultralytics_error"] = str(exc)
        else:
            out["ultralytics_error"] = f"weights_missing:{w}"

    return out


def _downscale_gray(rgb: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = rgb.shape[:2]
    s = max(h, w)
    if s <= max_side:
        small = rgb
    else:
        scale = max_side / float(s)
        small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)


def _edge_artifact_from_ocr_boxes(
    out_rgb: np.ndarray,
    ocr_boxes: Sequence[Dict[str, Any]],
    *,
    edge_artifact_ratio_max: float,
) -> Dict[str, Any]:
    mask = np.zeros(out_rgb.shape[:2], dtype=np.uint8)
    for box in ocr_boxes:
        try:
            poly = np.round(box["polygon"]).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
        except Exception:  # noqa: BLE001
            continue
    if mask.any():
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        gray = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        inside = float((edges[mask > 0] > 0).mean()) if (mask > 0).any() else 0.0
        border = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) - mask
        border_mean = float((edges[border > 0] > 0).mean()) if (border > 0).any() else 1.0
        ratio = float(inside / (border_mean + 1e-6))
        return {
            "edge_artifact_ratio": ratio,
            "edge_artifact_fail": bool(ratio > edge_artifact_ratio_max),
        }
    return {"edge_artifact_ratio": 0.0, "edge_artifact_fail": False}


def run_identity_integrity(
    image_path: Path,
    cfg: Dict[str, Any],
    audit: Dict[str, Any],
    *,
    qa_ocr_boxes: Optional[List[Dict[str, Any]]] = None,
    out_rgb: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compare anonymized output to the original (via audit["source_path"]) for:
    - SSIM (global, coarse)
    - Optional identity distance between primary face embeddings (if InsightFace probes exist)
    - Edge-density heuristic inside a dilated “text-risk” mask derived from OCR on the OUTPUT
    """
    qa = cfg.get("qa", {}) or {}
    src = Path(audit.get("source_path", "") or "")
    if out_rgb is None:
        out_rgb = imread_rgb(image_path)

    integrity: Dict[str, Any] = {"checks": {}}

    if src.is_file():
        in_rgb = imread_rgb(src)
        g_in = _downscale_gray(in_rgb)
        g_out = _downscale_gray(out_rgb)
        # SSIM requires matching sizes
        if g_in.shape == g_out.shape:
            s = float(ssim(g_in, g_out, data_range=255))
        else:
            g_out_r = cv2.resize(g_out, (g_in.shape[1], g_in.shape[0]), interpolation=cv2.INTER_AREA)
            s = float(ssim(g_in, g_out_r, data_range=255))
        integrity["checks"]["ssim"] = s
        ssim_min = float(qa.get("artifact_ssim_min", 0.75))
        integrity["checks"]["ssim_fail"] = bool(s < ssim_min)
    else:
        integrity["checks"]["ssim"] = None
        integrity["checks"]["ssim_fail"] = False

    # Identity distance (optional)
    qa_probes = audit.get("qa_probes", {}) or {}
    orig_emb = qa_probes.get("original_primary_face_embedding")
    identity_min = float(qa.get("identity_distance_min", 0.35))
    identity_result: Dict[str, Any] = {"enabled": False}
    if orig_emb and bool((cfg.get("insightface", {}) or {}).get("enabled", False)):
        try:
            app = _get_cached_insight_app(cfg)
            if app is None:
                raise RuntimeError("insightface_app_unavailable")
            record_insightface_call()
            faces = app.get(out_rgb[:, :, ::-1].copy())
            if faces:
                emb = faces[0].embedding.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                o = np.asarray(orig_emb, dtype=np.float32)
                o = o / (np.linalg.norm(o) + 1e-8)
                sim = float(np.dot(o, emb))
                dist = float(1.0 - sim)
                identity_result = {
                    "enabled": True,
                    "cosine_similarity": sim,
                    "cosine_distance": dist,
                    "fail": bool(dist < identity_min),
                }
            else:
                identity_result = {
                    "enabled": True,
                    "fail": bool(int(qa.get("min_output_face_if_input_face", 0)) == 1 and int(qa_probes.get("original_face_count", 0) or 0) > 0),
                    "note": "no_face_detected_on_output",
                }
        except Exception as exc:  # noqa: BLE001
            identity_result = {"enabled": False, "error": str(exc)}

    integrity["checks"]["identity"] = identity_result

    if qa_ocr_boxes is None:
        qa_ocr_boxes = _qa_ocr_boxes(audit, cfg, out_rgb[:, :, ::-1].copy())

    edge = _edge_artifact_from_ocr_boxes(
        out_rgb,
        qa_ocr_boxes or [],
        edge_artifact_ratio_max=float(qa.get("edge_artifact_ratio_max", 0.12)),
    )
    integrity["checks"]["edge_artifact_ratio"] = edge["edge_artifact_ratio"]
    integrity["checks"]["edge_artifact_fail"] = edge["edge_artifact_fail"]

    # Aggregate integrity fail
    fail = bool(integrity["checks"].get("ssim_fail", False) or identity_result.get("fail", False) or integrity["checks"].get("edge_artifact_fail", False))
    integrity["fail"] = fail
    integrity["reason"] = "artifact_or_identity_risk" if fail else "integrity_ok"
    return integrity


def run_decision(detection: Dict[str, Any], integrity: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final gate — **Decision Agent** logic (deterministic): combine scores and decide pass/fail + routing.
    """
    fail = bool(detection.get("fail")) or bool(integrity.get("fail"))
    decision = "fail" if fail else "pass"
    route = "final_clean" if decision == "pass" else "quarantine"
    parts: List[str] = []
    if detection.get("fail"):
        parts.append(f"detection_verification:{detection.get('reason', 'failed')}")
    if integrity.get("fail"):
        parts.append(f"identity_integrity:{integrity.get('reason', 'failed')}")
    failure_text = " | ".join(parts) if parts else ""
    return {
        "final_decision": decision,
        "route": route,
        "detection": detection,
        "integrity": integrity,
        "reason": ("qa_thresholds_failed" if fail else "all_thresholds_satisfied"),
        "failure_reason_text": failure_text,
    }


def build_failure_reason_text(decision: Dict[str, Any]) -> str:
    """Human-readable failure line for JSON ``failure_reason`` + quarantine recycling audits."""
    if decision.get("final_decision") == "pass":
        return ""
    return str(decision.get("failure_reason_text") or decision.get("reason") or "qa_fail")


def _qa_ocr_boxes(audit: Dict[str, Any], cfg: Dict[str, Any], bgr: np.ndarray) -> List[Dict[str, Any]]:
    """
    Reuse processing-time OCR cache when available (one PaddleOCR instance, one inference).
    """
    qa_cfg = cfg.get("qa", {}) or {}
    if bool(qa_cfg.get("reuse_processing_ocr", True)):
        cache = audit.get("qa_cache") or {}
        if "output_ocr_boxes" in cache:
            record_ocr_cache_hit()
            return deserialize_ocr_boxes(cache.get("output_ocr_boxes") or [])
    ocr = get_shared_paddle_ocr(cfg)
    return ocr_polys_and_scores(ocr, bgr)


def evaluate_image_qa(
    image_path: Path,
    cfg: Dict[str, Any],
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    End-to-end deterministic QA for one output image (expects JSON sidecar next to raster).

    Mirrors the three CrewAI agents using **reproducible** code paths so buyer audits stay valid
    even when LLM orchestration is disabled.
    """
    audit = load_audit_json(image_path)
    out_rgb = imread_rgb(image_path)
    bgr = out_rgb[:, :, ::-1].copy()
    qa_ocr_boxes = _qa_ocr_boxes(audit, cfg, bgr)

    det = run_detection_verification(
        image_path,
        cfg,
        project_root=project_root,
        ocr_boxes=qa_ocr_boxes,
        bgr=bgr,
    )
    logger.info(
        "qa_agent_complete",
        agent="Detection_Verification",
        image=image_path.name,
        ocr_lines=det.get("ocr_lines"),
        fail=det.get("fail"),
    )
    integ = run_identity_integrity(image_path, cfg, audit, qa_ocr_boxes=qa_ocr_boxes, out_rgb=out_rgb)
    logger.info(
        "qa_agent_complete",
        agent="Identity_Integrity",
        image=image_path.name,
        integrity_fail=integ.get("fail"),
        ssim=(integ.get("checks") or {}).get("ssim"),
    )
    dec = run_decision(det, integ, cfg)
    dec["qa_agents_deterministic"] = {
        "detection_verification": det,
        "identity_integrity": integ,
        "decision": {k: v for k, v in dec.items() if k != "qa_agents_deterministic"},
    }
    logger.info(
        "qa_agent_complete",
        agent="Decision",
        image=image_path.name,
        decision=dec.get("final_decision"),
        route=dec.get("route"),
    )
    return dec


# =============================================================================
# CrewAI wiring (optional LLM narrative / orchestration)
# =============================================================================


def _try_make_tool(fn: Callable[..., str], name: str):
    try:
        from crewai.tools import tool as crewai_tool  # type: ignore

        return crewai_tool(name)(fn)
    except Exception:  # noqa: BLE001
        try:
            from crewai_tools import tool as crewai_tool  # type: ignore

            return crewai_tool(name)(fn)
        except Exception:
            return None


def create_qa_crew(cfg: Dict[str, Any]):
    """
    Build the CrewAI crew described in the project spec.

    The returned object is a `crewai.Crew` when imports succeed; tools are bound to the active cfg.
    """
    from crewai import Agent, Crew, Task, Process  # type: ignore

    def _tool_detection(image_path: str) -> str:
        return json.dumps(run_detection_verification(Path(image_path), cfg, project_root=None), ensure_ascii=False)

    def _tool_identity(image_path: str) -> str:
        p = Path(image_path)
        audit = load_audit_json(p)
        return json.dumps(run_identity_integrity(p, cfg, audit), ensure_ascii=False)

    def _tool_decision(image_path: str) -> str:
        p = Path(image_path)
        d = json.loads(_tool_detection(image_path))
        i = json.loads(_tool_identity(image_path))
        return json.dumps(run_decision(d, i, cfg), ensure_ascii=False)

    tools_det = _try_make_tool(_tool_detection, "run_detection_verification")
    tools_id = _try_make_tool(_tool_identity, "run_identity_integrity")
    tools_dec = _try_make_tool(_tool_decision, "run_decision")

    detection_verifier = Agent(
        role="Detection Verification Agent",
        goal="Re-run PaddleOCR on the anonymized image and flag any high-confidence text detections",
        backstory="You enforce zero readable PII remnants on sale-ready assets.",
        verbose=True,
        allow_delegation=False,
        tools=[t for t in [tools_det] if t is not None],
    )

    identity_integrity = Agent(
        role="Identity & Integrity Agent",
        goal="Verify identity shift vs originals and detect inpainting/blending artifacts using classical CV metrics",
        backstory="You specialize in identity leakage risk and seam artifacts around inpainted regions.",
        verbose=True,
        allow_delegation=False,
        tools=[t for t in [tools_id] if t is not None],
    )

    decision_agent = Agent(
        role="Decision Agent",
        goal="Make the final pass/fail decision and choose routing between final_clean, quarantine, and manual_review",
        backstory="You consolidate evidence from other agents and apply strict thresholds from config.yaml.",
        verbose=True,
        allow_delegation=False,
        tools=[t for t in [tools_dec] if t is not None],
    )

    task1 = Task(
        description=(
            "Call the detection tool ONCE for image_path={image_path}. "
            "Return ONLY the tool JSON output (no extra prose)."
        ),
        expected_output="JSON string from run_detection_verification",
        agent=detection_verifier,
    )

    task2 = Task(
        description=(
            "Call the identity tool ONCE for image_path={image_path}. "
            "Return ONLY the tool JSON output (no extra prose)."
        ),
        expected_output="JSON string from run_identity_integrity",
        agent=identity_integrity,
    )

    task3 = Task(
        description=(
            "Call the decision tool ONCE for image_path={image_path}. "
            "Return ONLY the tool JSON output (no extra prose)."
        ),
        expected_output="JSON string from run_decision",
        agent=decision_agent,
    )

    return Crew(
        agents=[detection_verifier, identity_integrity, decision_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True,
    )


def maybe_run_crew_for_image(image_path: Path, cfg: Dict[str, Any]) -> Optional[str]:
    """
    Optionally execute the CrewAI crew when `qa.use_crewai_llm` is true *and* an API key is present.

    Returns the raw crew output string (best-effort) or None if skipped/failed.
    """
    qa = cfg.get("qa", {}) or {}
    if not bool(qa.get("use_crewai_llm", False)):
        return None
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("crewai_skipped_missing_openai_api_key")
        return None

    try:
        crew = create_qa_crew(cfg)
        # If tool wiring failed, do not spend LLM tokens on an incomplete crew graph.
        if any(not getattr(agent, "tools", None) for agent in getattr(crew, "agents", []) or []):
            logger.warning("crewai_skipped_missing_tools")
            return None
        return str(crew.kickoff(inputs={"image_path": str(image_path)}))
    except Exception as exc:  # noqa: BLE001
        logger.error("crewai_kickoff_failed", error=str(exc))
        return None


def run_qa_on_batch_directory(
    batch_dir: Path,
    cfg: Dict[str, Any],
    *,
    project_root: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    QA every ``*.jpg`` in a temp batch folder, merge results into sidecars, return rows for CSV summaries.
    """
    from .gpu_runtime import SharedCudaContext

    SharedCudaContext.configure(cfg)
    rows: List[Dict[str, Any]] = []
    for image_path in sorted(batch_dir.glob("*.jpg")):
        decision = evaluate_image_qa(image_path, cfg, project_root=project_root)
        qa_agents = decision.pop("qa_agents_deterministic", {})
        crew_out = maybe_run_crew_for_image(image_path, cfg)

        audit = load_audit_json(image_path)
        audit.setdefault("qa", {})
        audit["qa"]["deterministic"] = {k: v for k, v in decision.items() if k != "qa_agents_deterministic"}
        audit["qa"]["crew_agents_deterministic"] = qa_agents
        audit["qa"]["evaluated_at"] = utc_now_iso()
        if crew_out is not None:
            audit["qa"]["crewai_llm_output"] = crew_out
        audit["qa"]["final_decision"] = decision["final_decision"]
        audit["qa"]["final_route"] = decision["route"]
        if decision["final_decision"] == "pass":
            audit["qa"]["failure_reason"] = None
            audit["failure_reason"] = None
        else:
            fr = build_failure_reason_text(decision)
            audit["qa"]["failure_reason"] = fr
            audit["failure_reason"] = fr
        audit.setdefault("success_metrics", {})["qa_passed"] = decision["final_decision"] == "pass"
        audit.setdefault("success_metrics", {})["qa_target_success_rate"] = float(cfg.get("min_success_rate", 0.99))
        save_audit_json(image_path, audit)

        det = decision.get("detection") or {}
        rows.append(
            {
                "image": image_path.name,
                "batch_dir": batch_dir.name,
                "final_decision": decision["final_decision"],
                "final_route": decision["route"],
                "max_ocr_score": det.get("max_det_score"),
                "ocr_lines_qa": det.get("ocr_lines"),
                "output_face_count_qa": det.get("verification_output_face_count"),
                "ssim": (decision["integrity"].get("checks") or {}).get("ssim"),
                "identity_cosine_distance": ((decision["integrity"].get("checks") or {}).get("identity") or {}).get(
                    "cosine_distance"
                ),
            }
        )
    return rows


def propose_recycle_overrides(retry_count: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Progressive parameter adjustments for failed images (fed back via JSON sidecar on next pass).

    DeepPrivacy2 synthesis knobs mirror upstream ``anonymize.py`` kwargs; LaMa / IOPaint uses
    ``ldm_steps`` (erase strength / compute trade-off) when the IOPaint backend is active.
    """
    return {
        "extra_mask_dilation": int(4 * retry_count),
        "synthesis": {
            "truncation_value": float(min(0.35, 0.05 * retry_count)),
            "multi_modal_truncation": bool(retry_count >= 2),
        },
        # IOPaint InpaintRequest fields (only applied when lama.backend == lama_cleaner)
        "lama_iopaint": {
            "ldm_steps": int(20 + min(40, 5 * retry_count)),
        },
    }
