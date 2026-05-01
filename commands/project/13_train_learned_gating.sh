#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="$WORK_ROOT/03_features/feature_manifest.jsonl"
require_file "$FEATURE_MANIFEST"
require_file "$LEARNED_GATING_CONFIG"

OUT_DIR="$WORK_ROOT/10_learned_gating_ckpt"
mkdir -p "$OUT_DIR"

LIMIT_ARG=()
if [[ "$LEARNED_GATING_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$LEARNED_GATING_LIMIT")
fi

python -u "$OPENVLA_ROOT/scripts/train_learned_gating.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --output_dir "$OUT_DIR" \
  --config "$LEARNED_GATING_CONFIG" \
  --log_every "$LEARNED_GATING_LOG_EVERY" \
  "${LIMIT_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/13_train_learned_gating.log"

echo "[ok] output=$OUT_DIR"
