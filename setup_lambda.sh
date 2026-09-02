#!/usr/bin/env bash
# =============================================================================
# setup_lambda.sh — idempotent Lambda.ai host prep for the IMAGE-ONLY pipeline
# =============================================================================
# Safe to re-run. Prepares an ephemeral Lambda.ai GPU instance for a real run:
#   - verifies Python 3.10 (pins target 3.10; 3.12 has wheel gaps)
#   - installs system tools (rclone, exiftool)
#   - installs Python deps (CUDA 12.1 wheels)
#   - clones DeepPrivacy2 into vendor/ (idempotent)
#   - points model caches at the persistent volume so weights survive restarts
#   - runs the strict pre-flight gate
#
# Usage:
#   PERSIST_DIR=/home/ubuntu/priva-gen-data ./setup_lambda.sh
# =============================================================================
set -euo pipefail

PERSIST_DIR="${PERSIST_DIR:-/home/ubuntu/priva-gen-data}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "==> Image-Only pipeline · Lambda.ai setup"
echo "    repo:    $REPO_DIR"
echo "    persist: $PERSIST_DIR"

# 1) Python version check (warn, do not silently proceed on a mismatch) -------
PYV="$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo "none")"
if [ "$PYV" != "3.10" ]; then
  echo "!! WARNING: python is $PYV, expected 3.10. Activate the 3.10 env first:"
  echo "     conda create -n privagen python=3.10 -y && conda activate privagen"
fi

# 2) System tools (idempotent) -----------------------------------------------
if ! command -v rclone >/dev/null 2>&1 || ! command -v exiftool >/dev/null 2>&1; then
  echo "==> Installing system tools (rclone, exiftool)"
  sudo apt-get update -y
  sudo apt-get install -y rclone libimage-exiftool-perl
fi
rclone version | head -1 || true
exiftool -ver || true

# 3) Python dependencies ------------------------------------------------------
echo "==> Installing Python dependencies (requirements.txt)"
pip install -r requirements.txt

# 4) CUDA / torch compatibility report ---------------------------------------
echo "==> GPU / CUDA report"
nvidia-smi || echo "!! nvidia-smi not found — GPU driver missing"
python -c "import torch; print('cuda_available', torch.cuda.is_available(), 'torch.version.cuda', torch.version.cuda)" || true

# 5) DeepPrivacy2 clone (idempotent) + persistent model caches ---------------
if [ ! -d vendor/deep_privacy2/.git ]; then
  echo "==> Cloning DeepPrivacy2 into vendor/deep_privacy2"
  git clone https://github.com/hukkelas/deep_privacy2 vendor/deep_privacy2
else
  echo "==> DeepPrivacy2 already present (skipping clone)"
fi

export TORCH_HOME="${TORCH_HOME:-$PERSIST_DIR/model_cache}"
export IOPAINT_MODEL_DIR="${IOPAINT_MODEL_DIR:-$PERSIST_DIR/model_cache/iopaint}"
mkdir -p "$TORCH_HOME" "$IOPAINT_MODEL_DIR"
echo "==> Model caches: TORCH_HOME=$TORCH_HOME  IOPAINT_MODEL_DIR=$IOPAINT_MODEL_DIR"
echo "    (place the DeepPrivacy2 checkpoint into models/ or its expected cache per upstream docs)"

# 6) Security defaults for sensitive-data runs -------------------------------
export SECURITY_LEVEL="${SECURITY_LEVEL:-full}"
export ENABLE_BUCKET_CONFIRMATION="${ENABLE_BUCKET_CONFIRMATION:-1}"

# 7) Strict pre-flight gate (non-zero exit blocks the run) -------------------
echo "==> Running strict Image-Only pre-flight gate"
python -m scripts.preflight_lambda || {
  echo "!! Pre-flight FAILED — resolve the items above before running on real data."
  exit 1
}

echo "==> Setup complete. Next: python -m scripts.run_image_validation --count 12 --device cuda"
