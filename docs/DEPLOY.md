# Deployment guide — local to production (CoreWeave + B2)

This document covers **production deployment** for large, sensitive image bundles (tens of thousands of files). For first-time setup, complete **[QUICKSTART.md](QUICKSTART.md)** first.

---

## Deployment overview

```text
[Laptop / CI]  →  smoke tests + config
       ↓
[CoreWeave GPU]  →  ingest (B2 read-only) → process → export (B2 write)
       ↓
[Buyer delivery]  →  final_clean + audit CSV/PDF + anonymization_audit.json
```

**Principles:**

- Originals in `input_raw/` and B2 source are **never modified** by this pipeline.
- **Two B2 keys** — read-only ingest, write-only export.
- **Audit trail** — per-image JSON sidecars + master reports.
- **Resume** — safe to restart after interruption on 37k-scale runs.

---

## Pre-Push Verification Passed - 2026-07-17

Before publishing this repository, the following **5 security rules** were enforced
and verified (see also `.github/workflows/blank-slate-check.yml`, which fails CI on
any regression):

1. **Blank-slate data** — zero images or previous run artifacts are committed. All
   runtime dirs (`input_raw/`, `final_clean/`, `manual_review/`, `quarantine/`,
   `reports/`, `logs/`, `temp_processed/`) ship empty with `.gitkeep` only.
2. **Encrypted-secrets-only** — B2 keys load exclusively from an encrypted `.env.enc`
   (or a permission-locked `.env.local`); the committed `.env` is a mock template
   only. Real secrets are never committed and are never written to logs.
3. **Two-key least privilege** — separate read-only ingest key and write-only export
   key; the source bucket is read/copy-only and never modified or deleted.
4. **Key material excluded** — `.env.enc`, `secret.key`, `*.key`, and generated
   `config/rclone/rclone.conf` are git-ignored; secret file permissions are enforced
   (group/world-readable secret files are refused at startup).
5. **Clean repo hygiene** — line endings normalized to LF (`.gitattributes`), `.venv`
   untracked, and old versions isolated in clearly labeled `OLD__` folders.

---

## Rented GPU Setup (4-GPU node)

Target: a **4× NVIDIA H100** class instance with a persistent filesystem attached
(used for input staging, output, temp, reports, logs, and model caches). Secrets load
from an encrypted `.env.enc`; the Fernet key is injected at launch, never written to
disk.

The rclone commands below are the safe defaults hard-set in `config.yaml`,
`scripts/rclone_integration.py`, and `scripts/secrets_manager.py`, so a plain run uses
them automatically — no extra parameters required. Bucket and path values ship as
placeholders; set your own.

| Setting | Value |
|---------|-------|
| Source (read-only) bucket + path | `your-source-bucket/datasets/raw` (read-only key) |
| Output (read/write) bucket + path | `your-dest-bucket/datasets/anonymized` (read/write key) |
| Ingest excludes | `orig_*`, `thumb_*`, `*.mp4`, `*.mov`, `*.avi` (image-only) |
| Export flags | `--checksum --fast-list --transfers 16` |

### Launch steps (after SSH login)

```bash
# 1) Work inside the persistent filesystem and clone the repo
cd /mnt/data
git clone <your-fork-url> PrivaGen
cd PrivaGen

# 2) Create the environment + system tools
conda create -n privagen python=3.10 -y && conda activate privagen
pip install -r requirements.txt
sudo apt-get update && sudo apt-get install -y rclone exiftool

# 3) Inject the two Backblaze keys (one read-only, one read/write).
#    Secrets are injected at launch, never written to the VM disk.
export DATASET_ANON_SECRET_KEY="<your-fernet-key>"       # decrypts .env.enc
export B2_READONLY_KEY_ID="<readonly-key-id>"
export B2_READONLY_KEY="<readonly-application-key>"
export B2_WRITE_KEY_ID="<write-key-id>"
export B2_WRITE_KEY="<write-application-key>"
export B2_READONLY_BUCKET="<your-source-bucket>"
export B2_WRITE_BUCKET="<your-dest-bucket>"
export ENABLE_BUCKET_CONFIRMATION=1

# 4) Use all four H100s
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 5) Verify secrets + resolved paths + bucket connectivity
python -m scripts.secrets_manager check

# 6) DRY-RUN FIRST — image-only ingest listing (--dry-run keeps it safe)
rclone ls backblaze:your-source-bucket/datasets/raw --fast-list \
  --exclude "orig_*" --exclude "thumb_*" \
  --exclude "*.mp4" --exclude "*.mov" --exclude "*.avi" --dry-run
python -m scripts.main_pipeline --dry-run --count 50 --verify-integrity

# 7) FULL RUN — ingest (read-only) → anonymize → export (write). Bucket banner prints on startup.
python -m scripts.main_pipeline --gpu --security-level full \
  --ingest-from-b2 datasets/raw \
  --export-to-b2 datasets/anonymized \
  --verify-after-export

# 8) Export runs automatically via --export-to-b2 above; the equivalent copy is shown
#    here for reference / manual re-upload:
rclone copy local_clean/ backblaze:your-dest-bucket/datasets/anonymized \
  --checksum --fast-list --transfers 16
```

With `ENABLE_BUCKET_CONFIRMATION=1`, this banner appears in the terminal (or the
Jupyter UI cell output) **immediately on startup**, before any processing begins:

```text
✅ rclone commands loaded | Source bucket protected (read-only) | Output bucket write verified
✅ SUCCESS: Read-only ingest bucket ACCESSIBLE
✅ SUCCESS: RW output bucket ACCESSIBLE + write confirmed
🚀 PrivaGen ready for full image anonymization run
```

If either check fails, startup prints a `❌ FAILED` line naming the bucket and the
rclone error, so a misconfigured key is caught before the run — not mid-transfer.

### Scope: Images Only (video support deferred)

This deployment is **Images Only**. Video files are excluded at ingest by design (see the
`--exclude "*.mp4" ...` filters above and `scripts/utils.discover_images`, which only
iterates image extensions). There is **no** video decode/frame/re-encode/audio path in the
codebase. The Web UI forces the operator to explicitly confirm **Images Only** before
starting, and every run is stamped `images_only` in `reports/run_scope.json`.

### Rented-GPU Deployment Notes (recommended install sequence)

Rented instances can be **ephemeral** — treat model weights and secrets as needing
explicit placement on the **persistent volume**.

```bash
# 0) SSH in and work on the persistent filesystem
cd /mnt/data
git clone <your-fork-url> PrivaGen && cd PrivaGen

# 1) Python 3.10 environment (REQUIRED — pins target 3.10; 3.12 has wheel gaps for
#    paddlepaddle/insightface/torch+cu121). Prefer conda; venv from a 3.10 interpreter also OK.
conda create -n privagen python=3.10 -y && conda activate privagen
python --version    # must report 3.10.x

# 2) System tools
sudo apt-get update && sudo apt-get install -y rclone libimage-exiftool-perl
rclone version && exiftool -ver

# 3) Python deps (CUDA 12.1 wheels)
pip install -r requirements.txt

# 4) CUDA / driver compatibility check (must match the pinned torch==2.4.1+cu121)
nvidia-smi
python -c "import torch; print('cuda_available', torch.cuda.is_available(), 'torch.version.cuda', torch.version.cuda)"

# 5) DeepPrivacy2 — idempotent clone onto the persistent volume (safe to re-run)
[ -d vendor/deep_privacy2/.git ] || git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2
# Keep IOPaint/LaMa + InsightFace weights on the persistent disk so ephemeral hosts don't re-download:
export TORCH_HOME=/mnt/data/model_cache
export IOPAINT_MODEL_DIR=/mnt/data/model_cache/iopaint
mkdir -p "$TORCH_HOME" "$IOPAINT_MODEL_DIR"
# Place the DeepPrivacy2 checkpoint into models/ (or its expected cache) per upstream docs.

# 6) Secrets + scope + security level
export DATASET_ANON_SECRET_KEY="<fernet-key>"      # decrypts .env.enc (inject, don't store)
export SECURITY_LEVEL=full                          # secure temp wipe for sensitive data
export ENABLE_BUCKET_CONFIRMATION=1

# 7) HARD PRE-FLIGHT GATE — refuses to proceed unless GPU/CUDA, DeepPrivacy2, weights,
#    Python 3.10, rclone, exiftool, and security.level=full are all present.
python -m scripts.preflight_lambda            # exits non-zero on any FAIL

# 8) Functional validation + residual-PII gate (synthetic samples; use --samples-dir for real faces)
python -m scripts.run_image_validation --count 12 --device cuda

# 9) Full run (Images Only) — security-level full is mandatory for sensitive data
python -m scripts.main_pipeline --gpu --security-level full \
  --ingest-from-b2 datasets/raw \
  --export-to-b2 datasets/anonymized \
  --verify-after-export
```

**Access:** the Web UI (`python app.py`) binds to **`127.0.0.1:5000` only**. Reach it from
your laptop via SSH port-forward — never expose it publicly:

```bash
ssh -L 5000:127.0.0.1:5000 user@<instance-ip>
# then open http://127.0.0.1:5000 locally
```

Keep all working data on the attached filesystem so runs survive instance restarts:
point `input_raw/`, `final_clean/`, `temp_processed/`, `reports/`, `logs/`, and the
model caches at paths under your persistent volume.

---

## Phase 0 — Environment (UI or CLI)

### Recommended: Web UI on the worker

```bash
conda create -n dataset_anonymizer python=3.10 -y
conda activate dataset_anonymizer
pip install -r requirements-ui.txt
python app.py
```

Use **Setup Environment** → **Install All Dependencies Now** → confirm **Ready for GPU** or **Ready for CPU fallback**.

### CLI equivalent

```bash
python setup_environment.py
python -m scripts.environment_checker
```

### System packages (production nodes)

| Package | Purpose |
|---------|---------|
| `rclone` | Backblaze ingest/export |
| `exiftool` | Strong metadata stripping |
| `git` | Vendor clones (DeepPrivacy2) |
| NVIDIA driver + CUDA 12.x | GPU mode |

### DeepPrivacy2 vendor

```bash
git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2
# Install upstream dp2 per their documentation
```

---

## Phase 1 — Local validation

Run before touching production data at scale.

```bash
python -m scripts.health_check
python -m scripts.run_gpu_roundtrip_test --count 200
python -m scripts.main_pipeline --test-mode --max-images 32
python -m scripts.run_production_validation --count 500
```

**Review artifacts:**

| File | Pass criteria |
|------|----------------|
| `reports/readiness_summary.md` | **GO** or only environment blockers |
| `reports/gpu_readiness_report.md` | CUDA metrics sensible |
| `reports/security_report.md` | After `--security-level full` trial |
| `logs/monitoring.jsonl` | No repeated OOM or transfer failures |

---

## Phase 2 — Secrets and configuration

Secrets are **never** committed. On the worker they are stored either as an
**encrypted `.env.enc`** (recommended) or a **permission-locked `.env.local`**,
and loaded into the process environment automatically at startup by
`scripts/secrets_manager.py` (invoked from both `app.py` and `main_pipeline`).

Loading rules (enforced by the loader):

- Source priority: `.env.enc` → `.env.local` → `.env.production` → `.env`.
- Platform-injected environment variables are **never overwritten** (so a secrets
  manager or `EnvironmentFile=` can still take precedence).
- Placeholder/mock values (`mock…`, `<replace_me>`, `example…`) are ignored, so a
  committed mock `.env` can never mask a genuinely-missing production secret.
- The loader **refuses** a secrets or key file that is group/world readable
  (override only with `DATASET_ANON_ALLOW_INSECURE_ENV=1`).
- Secret **values are never logged** — only key names.

### Recommended: encrypted `.env.enc` (GPU VM)

```bash
# 1) One-time: generate the Fernet key (written 0600 to ~/.config/dataset_anonymizer/secret.key)
python -m scripts.secrets_manager gen-key

# 2) Put REAL keys in a locked plaintext file, then encrypt it
cp .env.example .env.local && chmod 600 .env.local
$EDITOR .env.local            # fill B2_KEY_ID / B2_READONLY_KEY / B2_WRITE_KEY / buckets
python -m scripts.secrets_manager encrypt --in .env.local --out .env.enc
shred -u .env.local           # destroy the plaintext once encrypted

# 3) Verify the pipeline can load and sees all required B2 settings
python -m scripts.secrets_manager check
```

`.env.enc`, `secret.key`, and `*.key` are all git-ignored. Keep `secret.key`
**off the repo** and backed up separately (e.g. your password manager). On the VM,
restrict the whole project: `chmod 700 ~/dataset_anonymizer`.

**Key delivery options:**

| Method | How the Fernet key reaches the process |
|--------|----------------------------------------|
| Key file (default) | `~/.config/dataset_anonymizer/secret.key` (0600) — set once via `gen-key` |
| Env var | `export DATASET_ANON_SECRET_KEY=<fernet-key>` in the launch shell / systemd `EnvironmentFile` |
| Custom path | `export DATASET_ANON_SECRET_KEY_FILE=/secure/mount/secret.key` |

> Prefer injecting `DATASET_ANON_SECRET_KEY` at launch (systemd `EnvironmentFile`
> with 0600, or your provider's secret store) so no decryption key is written to the VM disk.

### Alternative: permission-locked plaintext

If you skip encryption, place values in `.env.local` and lock it:

```bash
cp .env.example .env.local && chmod 600 .env.local && $EDITOR .env.local
```

The loader will read it directly (still refusing insecure permissions).

### Required variables

```bash
B2_KEY_ID=...
B2_READONLY_KEY=...          # ingest ONLY
B2_WRITE_KEY=...             # export ONLY
B2_READONLY_BUCKET=...
B2_WRITE_BUCKET=...
B2_INGEST_REMOTE_PATH=datasets/raw
B2_EXPORT_REMOTE_PATH=datasets/anonymized
RCLONE_CONFIG=/path/to/rclone.conf
# Optional encrypted export:
RCLONE_CRYPT_PASSWORD=...
RCLONE_CRYPT_SALT=...
```

See **[ENVIRONMENT.md](../ENVIRONMENT.md)** for the full variable list.

### `config.yaml` production excerpt

```yaml
batch_size: 48

gpu:
  device: cuda
  gpu_id: 0
  memory_efficient: true
  adaptive_batch: true
  empty_cache_between_batches: true
  require_cuda: true          # fail fast on GPU nodes (no silent CPU)

security:
  level: full
  copy_input_raw: true
  verify_ingest_checksums: true
  enforce_readonly_ingest: true
  crypt_enabled: true         # if using rclone crypt
  backup_before_cleanup: true
  secure_wipe: true

backblaze:
  transfers: 24
  checkers: 64
  upload_concurrency: 12
  chunk_size: "48M"
  verify_after_export: true

performance:
  adaptive_batch_size: true
  max_batch_size: 64
  always_monitor: true

monitoring:
  resource_monitoring: true
```

> **Warning:** `require_cuda: true` disables automatic CPU fallback — use only when the node is a confirmed GPU worker.

---

## Phase 3 — Backblaze B2 + rclone

### Two-key model

| Key | Env variable | Allowed operation |
|-----|--------------|-------------------|
| Read-only | `B2_READONLY_KEY` | Download → `input_raw/` |
| Write | `B2_WRITE_KEY` | Upload `final_clean/` only |

**Never** use the write key for ingest.

### Dry-run transfers

```bash
python -m scripts.main_pipeline --ingest-from-b2 --ingest-dry-run
python -m scripts.main_pipeline --export-to-b2 --export-dry-run
```

### Production transfer + process

```bash
python -m scripts.main_pipeline \
  --security-level full \
  --ingest-from-b2 datasets/raw \
  --export-to-b2 datasets/anonymized \
  --verify-after-export \
  --enable-crypt \
  --gpu
```

### Encrypted export (rclone crypt)

1. Set `security.crypt_enabled: true` or `--enable-crypt`.
2. Provide `RCLONE_CRYPT_PASSWORD` (+ optional `RCLONE_CRYPT_PASSWORD2`, `RCLONE_CRYPT_SALT`).
3. Export remote wraps the write bucket via `rclone crypt`.

### Verification

- **Ingest:** SHA256 manifest vs local files when `verify_ingest_checksums: true`.
- **Export:** `rclone check --checksum` when `verify_after_export: true`.

---

## Phase 4 — CoreWeave GPU workers

### Instance sizing (starting points)

| GPU | VRAM | Suggested `batch_size` | Notes |
|-----|------|------------------------|-------|
| RTX 4090 | 24 GB | 24–48 | Dev / mid-scale |
| A100 40GB | 40 GB | 48–64 | Production 37k |
| A100 80GB | 80 GB | 64+ | Raise `performance.max_batch_size` |

**One process per GPU.** Set:

```bash
export CUDA_VISIBLE_DEVICES=0
```

and `gpu.gpu_id: 0` in config.

### Persistent volumes

Mount writable storage for:

| Mount | Path |
|-------|------|
| Input staging | `input_raw/` |
| Output | `final_clean/` |
| Work | `temp_processed/` |
| Evidence | `reports/`, `logs/` |
| Models cache | `~/.cache`, vendor, IOPaint weights |

### Dashboard on cluster

```bash
python app.py
# From laptop:
ssh -L 5000:127.0.0.1:5000 user@coreweave-node
```

---

## Phase 5 — Scaling 37k+ images

### Single-worker vertical

- Enable `adaptive_batch` and `processed_manifest` resume.
- Monitor `logs/monitoring.jsonl` for quarantine rate and images/sec.
- Expect ~1.5–4+ images/sec on A100 with full GPU stack (workload dependent).

### Horizontal sharding

- Run **N workers** with **disjoint** `input_raw` prefixes or B2 path shards.
- **Do not** share one GPU across multiple pipeline processes.
- Merge `reports/` at orchestration layer if needed (outside this repo).

### Resume checklist

- [ ] Stale `.processing.lock` files expire (`resume.lock_stale_sec`)
- [ ] `processed_manifest.json` tracks completed stems
- [ ] Stems in `final_clean/` are skipped on restart
- [ ] Quarantine images retry until `max_retries`

### Disk and RAM planning

> **Warning:** For 37k × ~2 MB images, plan **hundreds of GB** for input + output + temp peak. Secure wipe on `temp_processed/` adds I/O — budget time accordingly.

| Mechanism | Benefit |
|-----------|---------|
| `copy_input_raw` | Originals never touched |
| Secure wipe (`level: full`) | Temp batches shredded after use |
| Ingest screening | Oversize/corrupt → early quarantine |

---

## Phase 6 — Observability

| Signal | Location |
|--------|----------|
| Structured logs | `logs/processing.log` |
| Batch metrics | `logs/monitoring.jsonl` |
| GPU VRAM peaks | `reports/gpu_readiness_report.md` |
| Security posture | `reports/security_report.md` |
| UI readiness | `reports/ui_environment_status.json` |

**DLP:** Logs contain memory stats and paths — not image bytes or embeddings (`security.redact_logs: true`).

---

## Phase 7 — Docker (outline)

```dockerfile
# Base: nvidia/cuda:12.x-cudnn-runtime + Python 3.10
RUN apt-get update && apt-get install -y rclone exiftool git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
RUN test -d vendor/deep_privacy2 || echo "WARN: mount or clone DP2"
ENV CUDA_VISIBLE_DEVICES=0
VOLUME ["/data/input_raw", "/data/final_clean", "/data/reports", "/data/logs"]
CMD ["python", "-m", "scripts.main_pipeline", "--gpu", "--security-level", "full"]
```

Inject B2 and crypt secrets via orchestrator secrets — **not** image layers.

---

## Pre-go-live checklist

- [ ] `reports/readiness_summary.md` → **GO** (or documented CPU-only acceptance)
- [ ] B2 dry-run ingest + export succeeded
- [ ] Sample images pass QA → `final_clean/`
- [ ] JSON sidecars contain expected `actions`, `integrity_hashes`, `qa` blocks
- [ ] `security_report.md` reflects `level: full`
- [ ] GPU roundtrip benchmark archived
- [ ] Rollback: `backups/` tested with `--backup-manifests`
- [ ] Legal/compliance sign-off on anonymization method (GAN vs blur)

---

## Rollback and recovery

| Scenario | Action |
|----------|--------|
| Bad export batch | Do not overwrite buyer path; restore from B2 versioning |
| Local corruption | `backups/` from `backup_before_cleanup` |
| Failed images | Inspect `quarantine/` + sidecar; adjust config; re-run |
| B2 ingest mistake | Read-only key prevents remote deletion; re-ingest after cleanup |

---

## Related docs

- [QUICKSTART.md](QUICKSTART.md) — first launch
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — install and GPU issues
- [README.md](../README.md) — architecture and CLI reference
