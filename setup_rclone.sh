#!/usr/bin/env bash
# =============================================================================
# setup_rclone.sh — configure Backblaze B2 remotes for dataset_anonymizer
#
# NEVER hardcode keys in this script. Export credentials before running:
#
#   export B2_KEY_ID="your-key-id"
#   export B2_READONLY_KEY="your-read-only-application-key"
#   export B2_WRITE_KEY="your-read-write-application-key"
#   export B2_READONLY_BUCKET="your-source-bucket"
#   export B2_WRITE_BUCKET="your-destination-bucket"
#   # optional:
#   export RCLONE_CONFIG="$HOME/.config/rclone/rclone.conf"
#   export B2_INGEST_REMOTE_PATH="datasets/raw"
#   export B2_EXPORT_REMOTE_PATH="datasets/anonymized"
#
# Usage:
#   source .env   # or export vars manually
#   bash setup_rclone.sh
# =============================================================================

set -euo pipefail

: "${B2_KEY_ID:?Set B2_KEY_ID (Backblaze application key ID)}"
: "${B2_READONLY_KEY:?Set B2_READONLY_KEY (read-only application key)}"
: "${B2_WRITE_KEY:?Set B2_WRITE_KEY (read/write application key)}"
: "${B2_READONLY_BUCKET:?Set B2_READONLY_BUCKET}"
: "${B2_WRITE_BUCKET:?Set B2_WRITE_BUCKET}"

RCLONE_CONFIG="${RCLONE_CONFIG:-$HOME/.config/rclone/rclone.conf}"
RCLONE_BINARY="${RCLONE_BINARY:-rclone}"

if ! command -v "${RCLONE_BINARY}" >/dev/null 2>&1; then
  echo "ERROR: rclone not found. Install from https://rclone.org/install/ or set RCLONE_BINARY." >&2
  exit 1
fi

mkdir -p "$(dirname "${RCLONE_CONFIG}")"

TEMPLATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/config/rclone"
TEMPLATE="${TEMPLATE_DIR}/rclone.conf.template"

if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: missing template ${TEMPLATE}" >&2
  exit 1
fi

# Render template with envsubst (requires gettext); fall back to sed placeholders.
if command -v envsubst >/dev/null 2>&1; then
  export B2_KEY_ID B2_READONLY_KEY B2_WRITE_KEY
  envsubst < "${TEMPLATE}" > "${RCLONE_CONFIG}"
else
  sed \
    -e "s|__B2_KEY_ID__|${B2_KEY_ID}|g" \
    -e "s|__B2_READONLY_KEY__|${B2_READONLY_KEY}|g" \
    -e "s|__B2_WRITE_KEY__|${B2_WRITE_KEY}|g" \
    < "${TEMPLATE}" > "${RCLONE_CONFIG}"
fi

chmod 600 "${RCLONE_CONFIG}"

echo "Wrote rclone config: ${RCLONE_CONFIG}"
echo "Verifying read-only remote (list only)..."
"${RCLONE_BINARY}" --config "${RCLONE_CONFIG}" lsd "b2-readonly:${B2_READONLY_BUCKET}" >/dev/null
echo "Verifying write remote (list only)..."
"${RCLONE_BINARY}" --config "${RCLONE_CONFIG}" lsd "b2-write:${B2_WRITE_BUCKET}" >/dev/null
echo "OK — remotes b2-readonly and b2-write are configured."
