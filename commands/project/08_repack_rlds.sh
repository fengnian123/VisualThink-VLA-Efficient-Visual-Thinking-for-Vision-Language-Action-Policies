#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

DROPOUT_MANIFEST="$WORK_ROOT/04_dropout/dropout_manifest.jsonl"
require_file "$DROPOUT_MANIFEST"
OUT_DIR="$WORK_ROOT/05_rlds"
mkdir -p "$OUT_DIR"

python "$OPENVLA_ROOT/scripts/repack_rlds_tfrecord.py" \
  --dropout_manifest "$DROPOUT_MANIFEST" \
  --output_tfrecord "$OUT_DIR/dynamic_features.tfrecord" \
  2>&1 | tee "$WORK_ROOT/logs/08_repack_rlds.log"

echo "[ok] tfrecord=$OUT_DIR/dynamic_features.tfrecord"
