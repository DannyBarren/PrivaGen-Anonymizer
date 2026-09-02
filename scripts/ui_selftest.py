"""
Automated HTTP + HTML checks for PrivaGen™ Web UI (no browser required).

Run while app.py is listening on 127.0.0.1:5000:
    python -m scripts.ui_selftest
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:5000"
ROOT = Path(__file__).resolve().parent.parent


def _get(path: str) -> tuple[int, str, dict | None]:
    req = urllib.request.Request(f"{BASE}{path}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            ct = resp.headers.get("Content-Type", "")
            data = json.loads(body) if "json" in ct else None
            return resp.status, body, data
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = None
        return e.code, body, data


def _post(path: str, payload: dict | None = None) -> tuple[int, str, dict | None]:
    data = json.dumps(payload or {}).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, body, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body, None


def main() -> int:
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))

    print("PrivaGen UI self-test\n")

    # Index page branding
    code, html, _ = _get("/")
    record("GET / returns 200", code == 200, f"status={code}")
    record("PrivaGen trademark in HTML", "PrivaGen" in html and "™" in html)
    record("Barren Business Development in HTML", "Barren Business Development" in html)
    record(
        "Product line in HTML",
        "a Barren Business Development Product" in html,
    )
    record("Brand product line element", 'id="pgBrandProductLine"' in html)
    record(
        "Browser title includes product line",
        "<title>PrivaGen™ · a Barren Business Development Product</title>" in html,
    )
    record("Branding CSS linked", "style.css" in html)
    record("script.js linked", "script.js" in html)
    for el_id in (
        "setupSection",
        "btnInstallDeps",
        "datasetSection",
        "btnScanDataset",
        "livePipelinePanel",
        "btnStart",
        "btnStop",
        "batchSize",
        "setupTerminal",
        "liveLog",
    ):
        record(f"Element #{el_id}", f'id="{el_id}"' in html or f"id='{el_id}'" in html)

    # APIs
    code, _, data = _get("/api/environment")
    record("GET /api/environment", code == 200 and isinstance(data, dict))
    if data:
        record("environment object present", "environment" in data)

    code, _, data = _get("/api/dataset/config")
    record("GET /api/dataset/config", code == 200 and data and data.get("ok"))

    code, _, data = _get("/api/live/status")
    record("GET /api/live/status", code == 200 and data and "live" in data)

    code, _, data = _get("/api/status")
    record("GET /api/status", code == 200 and "status" in (data or {}))

    code, _, data = _get("/api/stats")
    record(
        "GET /api/stats (always available)",
        code == 200 and isinstance(data, dict) and "counts" in data,
        f"mode={(data or {}).get('stats_mode')}",
    )
    counts = (data or {}).get("counts") or {}
    record(
        "stats counts object",
        isinstance(counts, dict) and "input_raw" in counts,
        f"input_raw={counts.get('input_raw')}",
    )

    code, _, data = _get("/api/monitoring")
    record("GET /api/monitoring (always available)", code == 200 and isinstance(data, dict))

    code, _, data = _get("/api/logs/latest")
    record("GET /api/logs/latest (always available)", code == 200 and isinstance(data, dict))

    code, _, _ = _get("/api/reports/master_summary.csv")
    record("GET /api/reports/master_summary.csv not 503", code in (200, 404), f"status={code}")

    code, _, data = _get("/api/b2/overview")
    record("GET /api/b2/overview", code == 200 and isinstance(data, dict))

    code, _, data = _get("/api/b2/commands?ingest_path=datasets/raw")
    record("GET /api/b2/commands preview", code == 200 and (data or {}).get("ok"))

    record("pipelineLimitedBanner in HTML", 'id="pipelineLimitedBanner"' in html)
    record("No repeated section brand stamps", html.count("pg-section-brand") == 0)
    record("Footer product line", 'id="pgBrandProductLineFooter"' in html)
    record("Product line appears twice", html.count("a Barren Business Development Product") == 2)

    # Dataset scan (local input_raw)
    test_dir = ROOT / "input_raw" / "_ui_selftest"
    test_dir.mkdir(parents=True, exist_ok=True)
    tiny = test_dir / "test_pixel.png"
    if not tiny.is_file():
        try:
            from PIL import Image

            Image.new("RGB", (4, 4), color=(128, 64, 32)).save(tiny)
        except Exception:
            tiny.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\r\nIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            )

    code, _, data = _post(
        "/api/dataset/scan",
        {"source_mode": "local", "local_path": str(test_dir.relative_to(ROOT)).replace("\\", "/")},
    )
    scan_ok = code == 200 and data and (data.get("scan") or {}).get("image_count", 0) >= 1
    record("POST /api/dataset/scan (local)", scan_ok, f"count={(data or {}).get('scan', {}).get('image_count')}")

    code, _, data = _post("/api/dataset/config", {"source_mode": "local", "local_path": "input_raw"})
    record("POST /api/dataset/config", code == 200 and data and data.get("ok"))

    # B2 panel toggle is client-side; server accepts b2 mode payload
    code, _, data = _post(
        "/api/dataset/config",
        {"source_mode": "b2", "b2_remote_path": "datasets/raw", "b2_ingest_on_start": False},
    )
    record("POST /api/dataset/config (b2 mode)", code == 200 and data and data.get("ok"))

    # Socket.IO endpoint (Engine.IO handshake)
    try:
        code_io, body_io, _ = _get("/socket.io/?EIO=4&transport=polling")
        record("Socket.IO polling handshake", code_io == 200 and "sid" in body_io)
    except Exception as exc:  # noqa: BLE001
        record("Socket.IO polling handshake", False, str(exc))

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
