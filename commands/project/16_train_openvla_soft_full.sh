#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="$WORK_ROOT/03_features/feature_manifest.jsonl"
OUT_DIR="$WORK_ROOT/13_openvla_soft_full_ckpt"
require_file "$FEATURE_MANIFEST"
require_file "$OPENVLA_SOFT_INTERFACE_CONFIG"

LIMIT_ARG=()
if [[ "$OPENVLA_SOFT_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$OPENVLA_SOFT_LIMIT")
fi

python "$OPENVLA_ROOT/scripts/train_openvla_soft_evidence.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --model_path "$VLA_PATH" \
  --output_dir "$OUT_DIR" \
  --config "$OPENVLA_SOFT_INTERFACE_CONFIG" \
  --mode full \
  --unnorm_key "$OPENVLA_SOFT_UNNORM_KEY" \
  --log_every "$OPENVLA_SOFT_LOG_EVERY" \
  "${LIMIT_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/16_train_openvla_soft_full.log"

echo "[ok] output=$OUT_DIR"
