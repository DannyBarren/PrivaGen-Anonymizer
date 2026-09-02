# Quickstart — launch in under 15 minutes

This guide assumes you are handling **sensitive personal images** (faces, ID text, license plates, metadata). The pipeline is designed so **original files are never modified** — work always happens on secure copies.

---

## What you will do

1. Create a clean **Anaconda** environment (Python 3.10).
2. Launch the **Web UI** with only lightweight packages.
3. Click **Install All Dependencies Now** inside the UI.
4. Confirm readiness (**GPU** or **CPU fallback**).
5. Drop test images into `input_raw/` and start processing.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Anaconda or Miniconda** | Recommended on Windows |
| **Git** | To clone this repository |
| **Disk space** | ~15 GB for Python packages; dataset space separate |
| **ExifTool** (optional but recommended) | Strongest metadata stripping — [ExifTool downloads](https://exiftool.org/) |

You do **not** need CUDA drivers installed to open the UI. GPU drivers are needed for full GPU mode.

---

## Step 1 — Get the code

```bash
git clone https://github.com/DannyBarren/PrivaGen-Anonymizer.git
cd PrivaGen-Anonymizer
```

On Windows PowerShell:

```powershell
cd C:\path\to\PrivaGen-Anonymizer
```

---

## Step 2 — Create the conda environment

**One command** creates an isolated environment:

```bash
conda create -n privagen python=3.10 -y
conda activate privagen
```

Verify:

```bash
python --version
# Expected: Python 3.10.x
```

> **Warning:** Do not install the full ML stack manually yet. The UI will do that in Step 4.

---

## Step 3 — Launch the Web UI (minimal install)

Install only what is needed to open the dashboard:

```bash
pip install -r requirements-ui.txt
python app.py
```

You should see:

```text
PrivaGen™ · a Barren Business Development Product
  Barren Business Development — Web control panel (PrivaGen-Anonymizer)
  UI URL: http://127.0.0.1:5000 (localhost only — secure)
```

Open a browser: **http://127.0.0.1:5000**

> **Security:** The UI binds to **127.0.0.1** only — it is not exposed to your LAN by default. On a remote GPU server, use SSH port forwarding: `ssh -L 5000:127.0.0.1:5000 user@host`.

---

## Step 4 — Use the Setup Environment panel

At the top of the page you will see **Setup Environment** (highlighted panel).

### What the panel shows

| Indicator | Green means | Red means |
|-----------|-------------|-----------|
| **Requirements installed** | Core packages importable | Click install |
| **GPU (CUDA + Torch)** | CUDA probe passed | CPU fallback will be used |
| **PaddleOCR** | Text detection available | Install incomplete |
| **IOPaint / LaMa** | Inpainting stack available | May use CPU redaction fallback |
| **DeepPrivacy2** | Vendor clone present | Face GAN skipped (blur still works) |
| **Python environment** | Conda env name or Python path shown | — |

**Overall badge** examples:

- **Ready for GPU** — Full pipeline when DeepPrivacy2 is present.
- **Ready for CPU fallback** — Safe to process; basic blur + text redaction.
- **Not ready** — Install dependencies first.

### Install with one click

1. Click **Install All Dependencies Now**.
2. Watch the **Setup terminal (live)** — pip output streams in real time.
3. Wait until you see a success line similar to:
   - `✅ Environment ready. You can now start processing on GPU.`
   - or `✅ Environment ready. You can now start processing on CPU.`

Progress phases you may see:

- Upgrading pip…
- Installing Torch + CUDA wheels…
- Resolving Pillow 9.5.0 for IOPaint…
- Testing GPU / CPU readiness…

Status is saved to `reports/ui_environment_status.json` (source of truth for UI and CLI).

### Re-check without reinstalling

Click **Re-check environment** after fixing drivers, cloning DeepPrivacy2, or installing ExifTool.

The UI starts in **safe mode** (no torch/paddle import at boot). Re-check runs the full GPU/Paddle probe and updates `reports/ui_environment_status.json`.

---

## Step 5 — Optional: DeepPrivacy2 (face GAN on GPU)

For **photorealistic face replacement** (not just blur), clone upstream into the vendor folder:

```bash
git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2
# Follow upstream install instructions for dp2 / tops inside that repo
```

Then click **Re-check environment** in the UI.

> **Note:** Processing still works without this folder — faces are blurred on CPU, or GAN runs when GPU + vendor are ready.

---

## Step 6 — Install ExifTool (recommended)

**Windows (Chocolatey):**

```powershell
choco install exiftool
```

**Linux:**

```bash
sudo apt install libimage-exiftool-perl
```

Confirm:

```bash
exiftool -ver
```

---

## Step 6b — Input dataset & live monitoring

After the environment is ready, use **Input Dataset Configuration**:

1. Choose **Local folder** (default `input_raw`) or **Backblaze B2** (read-only rclone remote).
2. Click **Scan Dataset** — updates **Total detected images**.
3. Click **Save configuration** — writes `reports/ui_dataset_config.json` (optional: merge into `config.yaml` → `ui.dataset`).

For B2, set `B2_READONLY_KEY` in `.env` before scanning. Listing uses `rclone lsf` only — no objects are downloaded during scan.

When you click **Start processing**, the **Live Pipeline Status** panel shows:

- Progress (`X / Y` images and %)
- Batch number and batch size
- GPU vs **CPU fallback** (with explanation text when on CPU)
- Throughput and ETA

Updates arrive over Socket.IO (`pipeline_status_update`, `batch_start`, `batch_complete`, `progress_tick`) about every 1–2 seconds while running.

---

## Step 7 — First processing run

### 7a. Add test images

Copy a few images into:

```text
input_raw/
```

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.tif`, `.tiff`

> **Warning:** Never edit files in place inside `input_raw/` expecting them to be “processed in place.” The pipeline copies to `temp_processed/` first.

### 7b. Start from the UI

Once the pipeline section is enabled (not greyed out):

1. Run **Scan Dataset** on `input_raw/` (or your folder) and confirm the image count.
2. Leave path overrides at defaults unless you use a custom layout.
3. Enable **Test mode** (smaller batches, more logging).
4. Set **Max images** to `5` for the first run.
5. Click **Start processing** and watch **Live Pipeline Status** for batch progress and GPU/CPU mode.

Watch **Live log** for batch events. When finished, check:

```text
final_clean/     ← sale-ready images + JSON sidecars
quarantine/      ← failed QA (auto-retry)
logs/            ← structured logs
reports/         ← summaries and audit
```

### 7c. Or start from the CLI

```bash
conda activate privagen
python -m scripts.main_pipeline --test-mode --max-images 5
```

---

## Step 8 — Understand GPU vs CPU mode

| Mode | You get | You do not get |
|------|---------|----------------|
| **GPU** | DeepPrivacy2 face GAN (if vendored), IOPaint/LaMa text inpainting, faster throughput | — |
| **CPU fallback** (automatic) | PaddleOCR text redaction, OpenCV face blur, metadata strip, full audits | Targeted GAN inpainting |

If GPU fails, you will see (console + UI banner):

```text
⚠️ GPU configuration failed: {reason}. Running on CPU with basic anonymization. Targeted inpainting disabled.
```

**No action is required** — processing continues securely.

---

## Step 9 — Verify success

| Check | Location |
|-------|----------|
| Output image | `final_clean/<stem>.jpg` |
| Per-image audit | `final_clean/<stem>.json` |
| Run summary | `reports/master_summary.csv` |
| Environment status | `reports/ui_environment_status.json` |

CLI check:

```bash
python -m scripts.environment_checker
```

---

## Windows Anaconda — copy-paste summary

```powershell
conda create -n privagen python=3.10 -y
conda activate privagen
cd C:\path\to\PrivaGen-Anonymizer
pip install -r requirements-ui.txt
python app.py
```

Browser → **http://127.0.0.1:5000** → **Install All Dependencies Now** → add images to `input_raw\` → **Start processing**.

---

## CLI-only setup (no UI)

```bash
conda activate privagen
python setup_environment.py
python -m scripts.health_check
python -m scripts.main_pipeline --test-mode --max-images 5
```

---

## Next steps

- Full overview: [README.md](../README.md)
- Production & B2: [DEPLOY.md](DEPLOY.md)
- Problems: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
