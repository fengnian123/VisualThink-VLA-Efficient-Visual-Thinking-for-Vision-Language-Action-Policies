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
require_file "$CKPT_DIR/gate.pt"
require_file "$LEARNED_GATING_CONFIG"

OUT_DIR="$WORK_ROOT/12_openvla_threeway_eval"
mkdir -p "$OUT_DIR"

SKIP_EMPTY_ARG=()
if [[ "$OPENVLA_THREEWAY_SKIP_EMPTY" == "1" ]]; then
  SKIP_EMPTY_ARG=(--skip_empty_instruction)
fi

python "$OPENVLA_ROOT/scripts/benchmark_openvla_three_way.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --checkpoint_dir "$CKPT_DIR" \
  --config "$LEARNED_GATING_CONFIG" \
  --model_path "$VLA_PATH" \
  --output_dir "$OUT_DIR" \
  --limit "$OPENVLA_THREEWAY_LIMIT" \
  --success_l1_thresh "$SUCCESS_L1_THRESH" \
  --disturb_ratio "$OPENVLA_THREEWAY_DISTURB_RATIO" \
  --disturb_scale "$OPENVLA_THREEWAY_DISTURB_SCALE" \
  --fallback_policy "$OPENVLA_THREEWAY_FALLBACK_POLICY" \
  "${SKIP_EMPTY_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/15_benchmark_openvla_three_way.log"

echo "[ok] summary=$OUT_DIR/summary_table.md"
