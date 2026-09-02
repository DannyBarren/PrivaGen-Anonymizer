"""
Final lightweight validation (20-30 images) + Web UI backend API test.

Run:
    python -m scripts.run_final_validation
    python -m scripts.run_final_validation --web-ui-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.component_check import run_component_activation_check
from scripts.main_pipeline import run_pipeline
from scripts.security_hardening import write_security_report, load_security_hardening, SecurityContext
from scripts.utils import close_pipeline_logging, deep_update, discover_images, load_config, resolve_pipeline_paths, setup_project_folders

UI_HOST = "127.0.0.1"
UI_PORT = 5000


def _seed(n: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = out / f"final_val_{i:03d}.png"
        if p.is_file():
            continue
        im = Image.new("RGB", (320, 240), color=(i * 7 % 255, 60, 90))
        if i % 5 == 0:
            d = ImageDraw.Draw(im)
            d.text((20, 100), f"T{i:03d}", fill=(240, 240, 240))
        im.save(p)


def _http_json(method: str, path: str, body: Dict[str, Any] | None = None, timeout: float = 30) -> Dict[str, Any]:
    url = f"http://{UI_HOST}:{UI_PORT}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def test_webui_backend(*, max_images: int, paths: Dict[str, str], timeout_sec: float = 600) -> Dict[str, Any]:
    """Requires app.py already running on 127.0.0.1:5000."""
    result: Dict[str, Any] = {"ok": False, "events": []}
    try:
        stats = _http_json("GET", f"/api/stats?input_raw={paths['input_raw']}")
        result["stats_before"] = stats.get("counts")
    except urllib.error.URLError as exc:
        result["error"] = f"UI not reachable: {exc}"
        return result

    start_body = {
        "batch_size": 8,
        "paths": paths,
        "max_images": max_images,
        "test_mode": True,
        "gpu_device": "cuda",
        "security_level": "standard",
    }
    start = _http_json("POST", "/api/start", start_body)
    if not start.get("ok"):
        result["error"] = start.get("error", "start failed")
        return result

    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_sec:
        st = _http_json("GET", "/api/status")
        if st.get("status") == "idle":
            break
        time.sleep(2)
    else:
        result["error"] = "timeout waiting for pipeline idle"
        return result

    st = _http_json("GET", "/api/status")
    result["last_result"] = st.get("last_result")
    result["last_error"] = st.get("last_error")
    result["ok"] = st.get("last_error") is None and not (st.get("last_result") or {}).get("stopped_early")
    stats_after = _http_json("GET", f"/api/stats?input_raw={paths['input_raw']}&final_clean={paths['final_clean']}")
    result["stats_after"] = stats_after.get("counts")
    result["elapsed_sec"] = time.monotonic() - t0
    return result


def run_cli_validation(count: int, test_root: Path) -> Dict[str, Any]:
    cfg_base = load_config(ROOT / "config.yaml")
    overrides = {
        "paths": {
            "input_raw": str(test_root / "input_raw"),
            "temp_processed": str(test_root / "temp_processed"),
            "final_clean": str(test_root / "final_clean"),
            "quarantine": str(test_root / "quarantine"),
            "manual_review": str(test_root / "manual_review"),
            "logs": str(test_root / "logs"),
            "reports": str(test_root / "reports"),
        },
        "batch_size": 8,
        "max_qa_waves": 5,
        "gpu": {"device": "cuda", "adaptive_batch": True},
        "lama": {"backend": "simple_lama"},
        "monitoring": {"resource_monitoring": True},
        "qa": {"reuse_processing_ocr": True, "text_det_score_fail": 0.95, "artifact_ssim_min": 0.55},
    }
    cfg = dict(cfg_base)
    deep_update(cfg, overrides)

    import yaml

    cfg_path = test_root / "config.yaml"
    cfg_path.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    setup_project_folders(test_root, cfg)
    paths = resolve_pipeline_paths(test_root, cfg)
    _seed(count, paths["input_raw"])

    components = run_component_activation_check(cfg, test_root)
    t0 = time.monotonic()
    result = run_pipeline(
        config_path=cfg_path,
        project_root=test_root,
        test_mode=True,
        max_images=count,
        security_level="standard",
    )
    close_pipeline_logging()
    elapsed = time.monotonic() - t0

    final_n = len(discover_images(paths["final_clean"], [".jpg"]))
    metrics = result.get("metrics") or {}
    return {
        "mode": "cli",
        "count": count,
        "final_clean": final_n,
        "pass_rate": final_n / count if count else 0,
        "elapsed_sec": elapsed,
        "components": components,
        "metrics": {k: v for k, v in metrics.items() if k not in ("batch_timings", "gpu_snapshots", "resource_snapshots")},
        "stopped_early": result.get("stopped_early"),
        "gpu_validation": result.get("gpu_validation"),
    }


def write_final_reports(
    report_root: Path,
    *,
    cli_result: Dict[str, Any],
    ui_result: Dict[str, Any] | None,
    components: Dict[str, Any],
    image_count: int,
) -> None:
    reports = report_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pass_rate = float(cli_result.get("pass_rate", 0))
    ui_ok = ui_result.get("ok") if ui_result else None
    blockers: List[str] = []
    if pass_rate < 0.95:
        blockers.append(f"CLI pass rate {pass_rate:.1%} < 95%")
    if ui_result and not ui_ok:
        blockers.append(f"Web UI test failed: {ui_result.get('error', 'unknown')}")
    elif ui_result:
        fc = (ui_result.get("stats_after") or {}).get("final_clean", 0)
        if fc < image_count * 0.9:
            blockers.append(f"Web UI processed only {fc}/{image_count} images to final_clean")
    if not components.get("deep_privacy2", {}).get("active"):
        blockers.append("DeepPrivacy2 not installed (vendor/deep_privacy2 missing)")
    if not components.get("gpu_cuda", {}).get("active"):
        blockers.append("CUDA unavailable on this host (GPU path unverified)")

    go = len([b for b in blockers if "pass rate" in b or "Web UI" in b]) == 0

    lines = [
        "# Production Readiness Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        f"## Recommendation: **{'GO' if go else 'NO-GO'}** for containerization",
        "",
        "### Web UI verification",
        "",
    ]
    if ui_result:
        lines.append(f"- API reachable: **{ui_result.get('error') is None and 'stats_before' in ui_result}**")
        lines.append(f"- Pipeline via `/api/start`: **{'PASS' if ui_result.get('ok') else 'FAIL'}**")
        lines.append(f"- Bind: `127.0.0.1:5000` (localhost only)")
        if ui_result.get("last_error"):
            lines.append(f"- Last error: `{ui_result['last_error']}`")
        lines.append(f"- Stats after: `{json.dumps(ui_result.get('stats_after', {}))}`")
    else:
        lines.append("- Web UI test: **skipped** (run with `--web-ui-test` while `python app.py` is running)")
    lines.append("")

    if blockers:
        lines.append("### Blockers / gaps")
        lines.append("")
        for b in blockers:
            lines.append(f"- {b}")
        lines.append("")

    lines.extend(
        [
            "### Lightweight CLI validation (20-30 images)",
            "",
            f"- Images: **{cli_result.get('count')}**",
            f"- Pass rate: **{100 * pass_rate:.0f}%**",
            f"- Elapsed: **{cli_result.get('elapsed_sec', 0):.1f}s**",
            "",
            "### Component activation",
            "",
            "```json",
            json.dumps(components, indent=2),
            "```",
            "",
            "### Containerization",
            "",
            "**GO** to build Docker image if Web UI + CLI pass and blockers are environment-only (GPU, DP2, B2).",
            "**NO-GO** for unattended 37k until CoreWeave GPU + B2 + DP2 checklist complete.",
            "",
        ]
    )
    (reports / "readiness_summary.md").write_text("\n".join(lines), encoding="utf-8")

    # Security report stub from components
    sec_path = reports / "security_report.md"
    if not sec_path.is_file():
        sec_cfg = load_security_hardening(load_config(ROOT / "config.yaml"))
        ctx = SecurityContext(project_root=report_root, config=sec_cfg)
        ctx.log_event("final_validation", cli=cli_result.get("mode"), ui=bool(ui_result))
        write_security_report(report_root, sec_cfg, ctx.events, summary={"validation": "final_lightweight"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--web-ui-test", action="store_true")
    parser.add_argument("--report-root", type=Path, default=ROOT)
    args = parser.parse_args()

    count = max(20, min(30, int(args.count)))

    with tempfile.TemporaryDirectory(prefix="final_val_") as tmp:
        test_root = Path(tmp)
        cli = run_cli_validation(count, test_root)

    ui_result = None
    if args.web_ui_test:
        ui_root = Path(args.report_root) / "ui_validation_run"
        ui_paths = {
            "input_raw": str(ui_root / "input_raw"),
            "final_clean": str(ui_root / "final_clean"),
            "quarantine": str(ui_root / "quarantine"),
            "manual_review": str(ui_root / "manual_review"),
            "temp_processed": str(ui_root / "temp_processed"),
            "logs": str(ui_root / "logs"),
            "reports": str(ui_root / "reports"),
        }
        for p in ui_paths.values():
            Path(p).mkdir(parents=True, exist_ok=True)
        _seed(count, Path(ui_paths["input_raw"]))
        ui_result = test_webui_backend(max_images=count, paths=ui_paths, timeout_sec=900)

    write_final_reports(
        Path(args.report_root),
        cli_result=cli,
        ui_result=ui_result,
        components=cli.get("components") or {},
        image_count=count,
    )
    print(json.dumps({"cli": cli, "ui": ui_result}, indent=2, default=str))
    print(f"\nReports updated under {args.report_root / 'reports'}")


if __name__ == "__main__":
    main()
