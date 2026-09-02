# Troubleshooting

Calm, practical fixes for common issues when processing **sensitive image datasets**. If you are stuck after install, start with **Re-check environment** in the Web UI or:

```bash
python -m scripts.environment_checker --json
```

---

## Quick diagnosis

| Symptom | Likely cause | First action |
|---------|--------------|--------------|
| UI opens but pipeline greyed out | Dependencies not installed | **Install All Dependencies Now** |
| `Running on CPU` banner | GPU unavailable or failed | See [CPU fallback](#cpu-fallback-automatic) — often OK for testing |
| pip fails on Pillow | IOPaint requires Pillow 9.5.0 | Use project `requirements.txt` only; do not upgrade Pillow separately |
| Torch DLL error on Windows | CUDA/driver mismatch | Reinstall NVIDIA driver; use bundled `requirements.txt` torch cu121 wheels |
| No faces anonymized | DeepPrivacy2 missing | Clone `vendor/deep_privacy2` or accept CPU blur |
| Text still visible | QA quarantine | Check `quarantine/` sidecar; lower thresholds or stronger inpaint |
| Out of memory | Batch too large | Lower `batch_size` in UI or `config.yaml` |
| Run stuck / duplicate work | Lock or manifest | See [Resume and locks](#resume-locks-and-37k-runs) |

---

## Installation issues

### UI starts but install button fails

1. Confirm you are in the intended conda env: `conda activate privagen`
2. Check disk space (15+ GB free).
3. Read the **Setup terminal** in the UI for the exact pip error line.
4. Retry from CLI for a full log:

```bash
python setup_environment.py --skip-conda
```

### Pillow / IOPaint conflict

**Symptom:** `iopaint` wants `Pillow==9.5.0` but another package upgraded Pillow.

**Fix:**

```bash
pip install Pillow==9.5.0
pip install -r requirements.txt
```

> **Warning:** Do not run `pip install --upgrade Pillow` after the project install.

### Torch / CUDA version mismatch

**Symptom:** `OSError: [WinError 126]` or `c10.dll` / `cudnn` errors.

**Checklist:**

- [ ] NVIDIA GPU driver installed (for GPU mode)
- [ ] Fresh conda env (Python 3.10)
- [ ] Install only via `pip install -r requirements.txt` (includes `+cu121` torch index)
- [ ] On Windows, use `paddlepaddle` wheel (requirements file selects CPU paddle on win32)

**Probe:**

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `False` and you need GPU, fix drivers first. If you only need to process data, **CPU fallback is automatic**.

### paddlepaddle-gpu on Windows

The project `requirements.txt` installs **CPU** `paddlepaddle` on Windows via environment marker. OCR still runs; GPU OCR may require a manual paddle GPU wheel matching your CUDA — see PaddlePaddle docs. The pipeline remains functional on CPU.

### Install hangs on Torch download

Large wheels (~2 GB). Wait, or install during off-peak hours. Use the UI live terminal to confirm progress.

---

## CPU fallback (automatic)

### What it means

You will see:

```text
⚠️ GPU configuration failed: {reason}. Running on CPU with basic anonymization. Targeted inpainting disabled.
```

This is **not a crash**. The pipeline continues with:

| Step | CPU behavior |
|------|----------------|
| Faces | OpenCV Gaussian blur on InsightFace boxes |
| Text | PaddleOCR detection + OpenCV inpaint redaction |
| Metadata | ExifTool / Pillow strip |
| Audits | Full JSON sidecars and reports |

### What is disabled in CPU fallback

- DeepPrivacy2 **GAN** face synthesis
- IOPaint / LaMa **photorealistic** inpainting

### When to worry

- **Production biometric sale** requiring GAN faces → fix GPU + install `vendor/deep_privacy2`.
- **Text-heavy compliance** → CPU path is often sufficient; verify QA pass rate on samples.

### Force GPU-only (fail fast)

Use on CoreWeave when CUDA must work:

```bash
python -m scripts.main_pipeline --gpu
```

Or in `config.yaml`:

```yaml
gpu:
  require_cuda: true
```

---

## GPU works but DeepPrivacy2 shows yellow/warn

Clone the vendor repo:

```bash
git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2
```

Install upstream `dp2` dependencies per their README. **Re-check environment** in the UI.

Without DP2, GPU mode still runs OCR + LaMa; faces rely on blur unless GAN loads.

---

## Web UI issues

### Cannot connect to http://127.0.0.1:5000

- Confirm `python app.py` is running in the project root.
- Windows firewall: allow Python on private networks if prompted.
- Remote server: `ssh -L 5000:127.0.0.1:5000 user@host` then open localhost on your laptop.

### Start processing returns 503

Environment not ready. Open **Setup Environment** → install or **Re-check**.

### Socket disconnects during long install

Normal on slow networks. Refresh the page; click **Re-check environment** when pip finishes in the server terminal.

---

## Pipeline runtime issues

### Images not picked up

- Files must be under `input_raw/` (or your UI path override).
- Extension must match `config.yaml` → `image_extensions`.
- Stems already in `final_clean/` are skipped (idempotent).

### Everything goes to quarantine

Read sidecar JSON in `quarantine/`:

- `failure_reason`
- `qa` scores
- `retry_count`

Tune `config.yaml` → `qa` thresholds after reviewing a sample of failures.

### exiftool not found

Install ExifTool and ensure it is on `PATH`. Pipeline falls back to Pillow stripping (weaker).

```bash
exiftool -ver
```

### Permission errors on Windows

Run terminal as user with write access to project folders. Avoid processing from OneDrive-synced paths during heavy I/O if you see file locks.

---

## Resume, locks, and 37k runs

### Resume behavior

- **`reports/processed_manifest.json`** — stems marked complete are skipped.
- **`.processing.lock`** in batch dirs — stale locks expire per `resume.lock_stale_sec` (default 3600s).

If a run was killed mid-batch:

1. Wait for lock stale timeout, or remove stale lock files under `temp_processed/` **only if no process is running**.
2. Restart pipeline — unfinished stems are retried.

### Out of memory (OOM)

1. Lower `batch_size` (UI slider or config).
2. Enable `gpu.empty_cache_between_batches: true` (default).
3. Set `gpu.memory_efficient: true`.
4. On 24 GB GPUs, start with batch 24–32, not 64.

### Disk space

Rough planning for 37k images @ ~2 MB each:

| Area | Estimate |
|------|----------|
| `input_raw/` | Source size (read-only) |
| `temp_processed/` | Peak ~1× batch copies (wiped on full security) |
| `final_clean/` | Similar to output JPEG size |
| `reports/` + `logs/` | Small relative to images |

Use **`--security-level full`** only when you accept secure wipe I/O cost on temp dirs.

---

## Backblaze / rclone

| Issue | Fix |
|-------|-----|
| Auth failed | Check `.env` keys; read-only key for ingest only |
| Slow transfer | Raise `backblaze.transfers` / `checkers` in config |
| Export verify failed | Re-run `rclone check`; inspect `quarantine/b2_transfer_failures` |

Dry-run first:

```bash
python -m scripts.main_pipeline --ingest-from-b2 --ingest-dry-run
```

---

## Getting help from logs

| File | Contents |
|------|----------|
| `logs/processing.log` | Structlog JSON — main diagnostic |
| `logs/monitoring.jsonl` | Per-batch timing, GPU memory stats (no image bytes) |
| `reports/ui_environment_status.json` | Install/readiness snapshot |
| `reports/security_report.md` | Hardening summary after full runs |

Redact logs before sharing externally — `security.redact_logs` helps but review for paths and hostnames.

---

## Reset environment (last resort)

```bash
conda deactivate
conda env remove -n privagen -y
conda create -n privagen python=3.10 -y
conda activate privagen
pip install -r requirements-ui.txt
python app.py
# Install from UI again
```

Do **not** delete `input_raw/` or `final_clean/` unless you have separate backups.

---

## Still stuck?

1. `python -m scripts.health_check`
2. `python -m scripts.environment_checker --json`
3. Review [QUICKSTART.md](QUICKSTART.md) and [DEPLOY.md](DEPLOY.md)
