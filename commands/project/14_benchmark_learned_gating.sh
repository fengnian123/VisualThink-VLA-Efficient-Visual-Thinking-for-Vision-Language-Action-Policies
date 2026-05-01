#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="$WORK_ROOT/03_features/feature_manifest.jsonl"
CKPT_DIR="$WORK_ROOT/10_learned_gating_ckpt"
require_file "$FEATURE_MANIFEST"
require_dir "$CKPT_DIR"
require_file "$CKPT_DIR/context.pt"
require_file "$CKPT_DIR/teacher.pt"
require_file "$CKPT_DIR/student.pt"
require_file "$CKPT_DIR/gate.pt"
require_file "$LEARNED_GATING_CONFIG"

OUT_DIR="$WORK_ROOT/11_learned_gating_eval"
mkdir -p "$OUT_DIR"

LIMIT_ARG=()
if [[ "$LEARNED_GATING_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$LEARNED_GATING_LIMIT")
fi

python "$OPENVLA_ROOT/scripts/benchmark_learned_gating.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --checkpoint_dir "$CKPT_DIR" \
  --config "$LEARNED_GATING_CONFIG" \
  --output_dir "$OUT_DIR" \
  --success_l1_thresh "$SUCCESS_L1_THRESH" \
  --disturb_ratio "$DISTURB_RATIO" \
  --disturb_scale "$DISTURB_SCALE" \
  --fallback_policy "$LEARNED_GATING_FALLBACK_POLICY" \
  "${LIMIT_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/14_benchmark_learned_gating.log"

echo "[ok] summary=$OUT_DIR/summary_table.md"
