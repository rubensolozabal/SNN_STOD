#!/usr/bin/env bash
set -euo pipefail

COMPETITION="imagenet-object-localization-challenge"
OUT_DIR="${1:-imagenet_kaggle}"

echo "==> Output directory: ${OUT_DIR}"
mkdir -p "${OUT_DIR}"

# 1) Ensure kaggle CLI is installed
if ! command -v kaggle >/dev/null 2>&1; then
  echo "==> Installing kaggle CLI..."
  python3 -m pip install --upgrade kaggle
fi

# 2) Check that credentials exist
# Supported by current docs:
#   - ~/.kaggle/access_token
#   - ~/.kaggle/kaggle.json
#   - or env var KAGGLE_API_TOKEN
if [[ -z "${KAGGLE_API_TOKEN:-}" && ! -f "${HOME}/.kaggle/access_token" && ! -f "${HOME}/.kaggle/kaggle.json" ]]; then
  echo "ERROR: Kaggle credentials not found."
  echo "Set KAGGLE_API_TOKEN, or place ~/.kaggle/access_token, or ~/.kaggle/kaggle.json"
  exit 1
fi

# Legacy credentials file should be private
if [[ -f "${HOME}/.kaggle/kaggle.json" ]]; then
  chmod 600 "${HOME}/.kaggle/kaggle.json" || true
fi
if [[ -f "${HOME}/.kaggle/access_token" ]]; then
  chmod 600 "${HOME}/.kaggle/access_token" || true
fi

# 3) Optional: list files first
echo "==> Listing files for ${COMPETITION}..."
kaggle competitions files "${COMPETITION}" -q || {
  echo
  echo "ERROR: Could not list competition files."
  echo "Most likely causes:"
  echo "  1) You have not accepted the competition rules on Kaggle"
  echo "  2) You are not logged in / token is invalid"
  echo "  3) Your account does not have access"
  exit 1
}

# 4) Download all files
echo "==> Downloading competition files..."
kaggle competitions download "${COMPETITION}" -p "${OUT_DIR}" -o

# 5) Unzip any downloaded zip archives
echo "==> Extracting zip files..."
shopt -s nullglob
for z in "${OUT_DIR}"/*.zip; do
  echo "   extracting $(basename "$z")"
  unzip -q -o "$z" -d "${OUT_DIR}/$(basename "$z" .zip)"
done

echo "==> Done."
echo "Files downloaded to: ${OUT_DIR}"