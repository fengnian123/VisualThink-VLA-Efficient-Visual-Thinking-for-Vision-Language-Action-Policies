#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

DROPOUT_MANIFEST="$WORK_ROOT/04_dropout/dropout_manifest.jsonl"
TEACHER_CKPT="$WORK_ROOT/06_distill_ckpt/teacher.pt"
STUDENT_CKPT="$WORK_ROOT/06_distill_ckpt/student.pt"
require_file "$DROPOUT_MANIFEST"
require_file "$TEACHER_CKPT"
require_file "$STUDENT_CKPT"

OUT_DIR="$WORK_ROOT/07_distill_eval"
mkdir -p "$OUT_DIR"

LIMIT_ARG=()
if [[ "$FEATURE_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$FEATURE_LIMIT")
fi

python "$OPENVLA_ROOT/scripts/benchmark_dynamic_distill.py" \
  --dropout_manifest "$DROPOUT_MANIFEST" \
  --teacher_ckpt "$TEACHER_CKPT" \
  --student_ckpt "$STUDENT_CKPT" \
  --output_dir "$OUT_DIR" \
  "${LIMIT_ARG[@]}" \
  --teacher_hidden 1024 \
  --student_hidden 256 \
  --success_l1_thresh "$SUCCESS_L1_THRESH" \
  --disturb_ratio "$DISTURB_RATIO" \
  --disturb_scale "$DISTURB_SCALE" \
  --fallback_policy disturbed_always \
  2>&1 | tee "$WORK_ROOT/logs/10_benchmark_distill.log"

echo "[ok] summary=$OUT_DIR/summary_table.md"
