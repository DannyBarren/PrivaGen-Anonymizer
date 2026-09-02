"""
LaMa inpainting via IOPaint (Sanster/IOPaint — the maintained lama-cleaner lineage).

``iopaint.model_manager.ModelManager`` is called under ``torch.inference_mode()``:
- ``image``: uint8 numpy **RGB**, shape ``[H, W, 3]``
- ``mask``: uint8 numpy **``[H, W, 1]``**, value ``255`` marks repaint
- ``config``: ``InpaintRequest`` (defaults are appropriate for LaMa erase)

The manager returns **BGR** uint8; we convert back to RGB for the rest of the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def run_iopaint_inpaint(
    manager: Any,
    rgb: np.ndarray,
    mask_hw: np.ndarray,
    inpaint_request: Any | None = None,
    *,
    cfg: Any | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute one IOPaint forward pass using an already-constructed ``ModelManager``."""
    import torch

    if rgb.shape[:2] != mask_hw.shape[:2]:
        raise ValueError(f"rgb {rgb.shape} and mask {mask_hw.shape} spatial dims must match")

    m = mask_hw.astype(np.uint8)
    if m.ndim == 2:
        m = m[:, :, np.newaxis]

    if inpaint_request is None:
        from iopaint.schema import InpaintRequest

        inpaint_request = InpaintRequest()

    from .gpu_runtime import inpaint_autocast_enabled

    use_amp = inpaint_autocast_enabled(cfg or {})
    ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and torch.cuda.is_available()
        else None
    )
    with torch.inference_mode():
        if ctx is not None:
            with ctx:
                out_bgr = manager(rgb, m, inpaint_request)
        else:
            out_bgr = manager(rgb, m, inpaint_request)

    out_rgb = out_bgr[:, :, ::-1].astype(np.uint8)
    return out_rgb, {"backend": "iopaint", "ok": True}
