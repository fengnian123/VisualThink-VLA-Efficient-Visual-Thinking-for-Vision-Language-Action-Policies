#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="$WORK_ROOT/03_features/feature_manifest.jsonl"
require_file "$FEATURE_MANIFEST"
OUT_DIR="$WORK_ROOT/04_dropout"
mkdir -p "$OUT_DIR"

python "$OPENVLA_ROOT/scripts/apply_feature_dropout.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --output_dir "$OUT_DIR" \
  --policy_yaml "$OPENVLA_ROOT/configs/dropout_policy.yaml" \
  2>&1 | tee "$WORK_ROOT/logs/07_apply_dropout.log"

echo "[ok] manifest=$OUT_DIR/dropout_manifest.jsonl"
