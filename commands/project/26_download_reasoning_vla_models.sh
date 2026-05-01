#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_FAMILY="${MODEL_FAMILY:-all}"

if [[ ! "$MODEL_FAMILY" =~ ^(all|ecot|deepthinkvla)$ ]]; then
  echo "[error] unsupported MODEL_FAMILY=$MODEL_FAMILY" >&2
  echo "supported: all, ecot, deepthinkvla" >&2
  exit 1
fi

if [[ "$MODEL_FAMILY" == "all" || "$MODEL_FAMILY" == "ecot" ]]; then
  bash "$SCRIPT_DIR/25_download_ecot_assets.sh"
fi

if [[ "$MODEL_FAMILY" == "all" || "$MODEL_FAMILY" == "deepthinkvla" ]]; then
  DOWNLOAD_SCOPE=models bash "$SCRIPT_DIR/23_download_deepthinkvla_assets.sh"
fi
