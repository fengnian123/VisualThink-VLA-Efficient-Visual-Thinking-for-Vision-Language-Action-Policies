#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="${OPENVLA_SOFT_FEATURE_MANIFEST:-$WORK_ROOT/03_features/feature_manifest.jsonl}"
FULL_CKPT_DIR="${OPENVLA_SOFT_FULL_CKPT_DIR:-$WORK_ROOT/13_openvla_soft_full_ckpt}"
DYN_CKPT_DIR="${OPENVLA_SOFT_DYNAMIC_CKPT_DIR:-$WORK_ROOT/14_openvla_soft_dynamic_ckpt}"
OUT_DIR="${OPENVLA_SOFT_BENCHMARK_OUT_DIR:-$WORK_ROOT/15_openvla_soft_threeway_eval}"
require_file "$FEATURE_MANIFEST"
require_dir "$FULL_CKPT_DIR"
require_dir "$DYN_CKPT_DIR"
require_file "$FULL_CKPT_DIR/adapter.pt"
require_file "$DYN_CKPT_DIR/adapter.pt"

SKIP_EMPTY_ARG=()
if [[ "$OPENVLA_SOFT_THREEWAY_SKIP_EMPTY" == "1" ]]; then
  SKIP_EMPTY_ARG=(--skip_empty_instruction)
fi

NORM_STATS_ARG=()
if [[ -n "${OPENVLA_SOFT_NORM_STATS:-}" ]]; then
  NORM_STATS_ARG=(--norm_stats "$OPENVLA_SOFT_NORM_STATS")
fi

python -u "$OPENVLA_ROOT/scripts/benchmark_openvla_soft_three_way.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --model_path "$VLA_PATH" \
  --full_checkpoint_dir "$FULL_CKPT_DIR" \
  --dynamic_checkpoint_dir "$DYN_CKPT_DIR" \
  --output_dir "$OUT_DIR" \
  --limit "$OPENVLA_SOFT_THREEWAY_LIMIT" \
  --unnorm_key "$OPENVLA_SOFT_UNNORM_KEY" \
  --disturb_ratio "$OPENVLA_SOFT_THREEWAY_DISTURB_RATIO" \
  --disturb_scale "$OPENVLA_SOFT_THREEWAY_DISTURB_SCALE" \
  "${NORM_STATS_ARG[@]}" \
  "${SKIP_EMPTY_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/18_benchmark_openvla_soft_three_way.log"

echo "[ok] summary=$OUT_DIR/summary_table.md"
