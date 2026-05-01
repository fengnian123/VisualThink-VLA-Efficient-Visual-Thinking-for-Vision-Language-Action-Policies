#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

DROPOUT_MANIFEST="$WORK_ROOT/04_dropout/dropout_manifest.jsonl"
require_file "$DROPOUT_MANIFEST"
OUT_DIR="$WORK_ROOT/06_distill_ckpt"
mkdir -p "$OUT_DIR"

LIMIT_ARG=()
if [[ "$FEATURE_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$FEATURE_LIMIT")
fi

python "$OPENVLA_ROOT/scripts/train_dynamic_distill.py" \
  --dropout_manifest "$DROPOUT_MANIFEST" \
  --output_dir "$OUT_DIR" \
  --epochs "$DISTILL_EPOCHS" \
  --batch_size "$DISTILL_BATCH_SIZE" \
  "${LIMIT_ARG[@]}" \
  --teacher_hidden 1024 \
  --student_hidden 256 \
  --consistency_weight 0.7 \
  2>&1 | tee "$WORK_ROOT/logs/09_train_distill.log"

echo "[ok] output=$OUT_DIR"
