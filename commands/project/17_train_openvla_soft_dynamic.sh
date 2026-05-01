#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="$WORK_ROOT/03_features/feature_manifest.jsonl"
GATE_CKPT_DIR="$WORK_ROOT/10_learned_gating_ckpt"
OUT_DIR="$WORK_ROOT/14_openvla_soft_dynamic_ckpt"
require_file "$FEATURE_MANIFEST"
require_file "$OPENVLA_SOFT_INTERFACE_CONFIG"
require_file "$OPENVLA_SOFT_DYNAMIC_GATE_CONFIG"
require_dir "$GATE_CKPT_DIR"
require_file "$GATE_CKPT_DIR/gate.pt"

LIMIT_ARG=()
if [[ "$OPENVLA_SOFT_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$OPENVLA_SOFT_LIMIT")
fi

python -u "$OPENVLA_ROOT/scripts/train_openvla_soft_evidence.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --model_path "$VLA_PATH" \
  --output_dir "$OUT_DIR" \
  --config "$OPENVLA_SOFT_INTERFACE_CONFIG" \
  --mode dynamic \
  --unnorm_key "$OPENVLA_SOFT_UNNORM_KEY" \
  --gate_checkpoint_dir "$GATE_CKPT_DIR" \
  --gate_config "$OPENVLA_SOFT_DYNAMIC_GATE_CONFIG" \
  --log_every "$OPENVLA_SOFT_LOG_EVERY" \
  "${LIMIT_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/17_train_openvla_soft_dynamic.log"

echo "[ok] output=$OUT_DIR"
