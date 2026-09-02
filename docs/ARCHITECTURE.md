# Architecture

Two entry points, one pipeline. `app.py` is the operator UI; `scripts/main_pipeline.py`
is the orchestrator the UI calls. Everything else is a module the orchestrator pulls in.

## Entry points

```bash
python app.py                                        # Flask dashboard on 127.0.0.1:5000
python -m scripts.main_pipeline --test-mode --max-images 5
```

## File map

| File | What it does |
|------|--------------|
| `app.py` | Flask + Flask-SocketIO control plane. Boots with only `flask` + `flask-socketio` so the dashboard comes up before the ML stack is installed, then installs the rest on request. Serves the setup panel, run controls, live progress, and report downloads. Binds `127.0.0.1:5000` by default. |
| `scripts/main_pipeline.py` | Orchestrator. Scaffolds folders and logging, warms models, then runs an outer wave loop: discover pending work in `input_raw/` (plus `quarantine/` retries), hand each batch to `batch_processor`, run QA, route each image to `final_clean/` / `quarantine/` / `manual_review/`, and emit per-batch CSV plus the master CSV/PDF and JSON audit. Owns the CLI flags (`--test-mode`, `--max-images`, `--dry-run`, `--gpu`, `--security-level`, `--ingest-from-b2`, `--export-to-b2`). Heavy imports are deferred so `--dry-run` works on a plain CPU host. |
| `scripts/batch_processor.py` | The per-image work. Loads RGB with Pillow (EXIF-aware), hashes source bytes, probes with InsightFace for QA, replaces faces/person regions with DeepPrivacy2, runs PaddleOCR on the post-GAN frame to build a text mask, inpaints with LaMa (IOPaint preferred, `simple_lama` fallback), strips metadata with ExifTool (Pillow fallback), then writes `{stem}.jpg` plus its JSON audit sidecar into `temp_processed/batch_XXXXX/`. |
| `scripts/gpu_runtime.py` | Centralized CUDA configuration. All Torch-backed models (DeepPrivacy2/tops, IOPaint/LaMa, InsightFace ONNX CUDA) share one `torch.device` through `SharedCudaContext` to limit fragmentation. Validates GPU at startup, writes the readiness report, snapshots CUDA memory, and empties the allocator cache after each batch. |
| `scripts/device_manager.py` | The fallback policy, in one place. Try GPU first; on any GPU-related failure (CUDA, DLL, OOM, model load, inference) switch to CPU — PaddleOCR plus OpenCV blur/redaction — and disable targeted GAN inpainting. No operator interaction, and the JSON sidecars and reports stay the same shape. |
| `scripts/secrets_manager.py` | Startup secrets loader. Prefers an encrypted `.env.enc` (Fernet), falls back to a permission-locked `.env.local`. Never overwrites platform-injected env vars, ignores placeholder values, refuses group/world-readable secret or key files, and logs key names only. Also applies the non-secret bucket/path defaults and prints the two-key bucket confirmation banner. CLI: `gen-key`, `encrypt`, `decrypt`, `check`. |
| `scripts/rclone_integration.py` | Backblaze B2 transfers via rclone, with key separation enforced in code: the `b2-readonly` remote can only ingest, the `b2-write` remote can only export. Ingest listings are image-only (excludes `orig_*`, `thumb_*`, `*.mp4`, `*.mov`, `*.avi`) and default to `--dry-run`. Handles retry classification, transfer batching, and post-export checksum verification. |
| `scripts/agentic_qa_crew.py` | The QA gate. Deterministic scoring always runs — OCR re-scan for leftover text, SSIM/edge integrity checks, optional InsightFace identity distance — so a pass/fail is reproducible without an LLM. CrewAI orchestrates three specialist agents (detection verification, identity/integrity, decision) and adds a natural-language rationale only when `OPENAI_API_KEY` is set. Output is the routing decision: clean, quarantine, or manual review. |
| `scripts/ui_bridge.py` | Translates pipeline progress callbacks into Socket.IO events for the dashboard. No-ops when Flask is not loaded, and reports counts and batch indices only — never secrets or full filesystem paths. |
| `scripts/security_hardening.py` | Security posture: DLP checks, read-only ingest enforcement, at-rest protection, secure temp wipe at `--security-level full`, the processed-file manifest, critical-artifact backup, and `reports/security_report.md`. Hooks into the orchestrator and rclone layer through `SecurityContext`. |

## Supporting modules

| File | What it does |
|------|--------------|
| `scripts/utils.py` | Config loading, folder scaffolding, `discover_images` (the images-only gate), SHA-256 helpers, structlog setup, audit JSON read/write, master CSV/PDF writers. |
| `scripts/shared_models.py` | Singletons for PaddleOCR, InsightFace, and the LaMa model manager so a batch loads each model once. |
| `scripts/security.py` | Path containment (`resolve_under`), secret redaction for logs and reports, ingest hash verification. |
| `scripts/performance.py` | Adaptive batch sizing, ingest screening (size/pixel caps), performance report. |
| `scripts/pipeline_metrics.py` | Throughput, GPU memory, and system resource sampling during a run. |
| `scripts/processing_locks.py` | Per-stem locks and resume discovery so an interrupted run does not reprocess or collide. |
| `scripts/monitoring.py` | `logs/monitoring.jsonl` event stream with security context attached. |
| `scripts/lama_iopaint.py` | IOPaint/LaMa wiring and backend selection. |
| `scripts/environment_checker.py` / `environment_installer.py` | What the Setup panel calls: check for required packages and system tools, then install them while streaming output to the UI. |
| `scripts/component_check.py` | Activation check that each model component actually loaded before a run starts. |
| `scripts/health_check.py` / `preflight_lambda.py` | Pre-run gates. `health_check` is the general one; `preflight_lambda` is the hard gate for a GPU node (refuses to proceed without CUDA, DeepPrivacy2, weights, Python 3.10, rclone, exiftool, and `security.level=full`). |
| `scripts/dataset_scanner.py` | Counts and profiles a dataset (local or B2 `lsf` listing) before committing to a run. |

## Data flow on disk

```
input_raw/                 originals, read-only, never written
temp_processed/batch_XXXXX/ working copies + JSON sidecars (wiped after routing)
final_clean/               passed QA, with its JSON sidecar
quarantine/                failed QA, eligible for automatic retry
manual_review/            failed QA, needs a human
reports/                   master CSV/PDF, run_scope.json, readiness + security reports
logs/                      structlog output, monitoring.jsonl, batch summaries
models/  vendor/           weights and upstream clones, fetched at setup
```

## Configuration

`config.yaml` is the single source of truth; `config_gpu.yaml` and `config_cpu.yaml` are
device-specific snapshots. Secrets never live in these files — they come from `.env.enc`
or injected environment variables via `secrets_manager`. Key locked settings:

- `scope.processing_mode: images_only` — the only supported mode.
- `security.copy_input_raw: true` — originals are copied, not mutated.
- `security.redact_logs: true` — secret values never reach a log line.
- `gpu.require_cuda: false` — lets `device_manager` fall back to CPU instead of aborting the run.
