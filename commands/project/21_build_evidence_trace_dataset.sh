#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

TRACE_FEATURE_MANIFEST="${EVIDENCE_TRACE_FEATURE_MANIFEST:-$WORK_ROOT/03_features/feature_manifest.jsonl}"
TRACE_GATE_CKPT_DIR="${EVIDENCE_TRACE_GATE_CKPT_DIR:-$WORK_ROOT/10_learned_gating_ckpt}"
TRACE_DYNAMIC_CKPT_DIR="${EVIDENCE_TRACE_DYNAMIC_CKPT_DIR:-$WORK_ROOT/14_openvla_soft_dynamic_ckpt}"
TRACE_OUTPUT_PATH="${EVIDENCE_TRACE_OUTPUT_PATH:-$WORK_ROOT/16_evidence_trace/evidence_trace.jsonl}"
TRACE_DATASET_NAME="${EVIDENCE_TRACE_DATASET_NAME:-$DATASET}"
TRACE_LIMIT="${EVIDENCE_TRACE_LIMIT:-0}"
TRACE_MASK_THRESHOLD="${EVIDENCE_TRACE_MASK_THRESHOLD:-0.5}"

require_file "$TRACE_FEATURE_MANIFEST"
require_dir "$TRACE_GATE_CKPT_DIR"
require_file "$TRACE_GATE_CKPT_DIR/counterfactual_utilities.jsonl"
require_dir "$TRACE_DYNAMIC_CKPT_DIR"
require_file "$TRACE_DYNAMIC_CKPT_DIR/channel_masks.npy"

LIMIT_ARG=()
if [[ "$TRACE_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$TRACE_LIMIT")
fi

python -u "$OPENVLA_ROOT/scripts/build_evidence_trace_dataset.py" \
  --feature_manifest "$TRACE_FEATURE_MANIFEST" \
  --gate_checkpoint_dir "$TRACE_GATE_CKPT_DIR" \
  --dynamic_checkpoint_dir "$TRACE_DYNAMIC_CKPT_DIR" \
  --output_path "$TRACE_OUTPUT_PATH" \
  --dataset_name "$TRACE_DATASET_NAME" \
  --mask_threshold "$TRACE_MASK_THRESHOLD" \
  "${LIMIT_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/21_build_evidence_trace_dataset.log"

echo "[ok] trace=$TRACE_OUTPUT_PATH"
echo "[ok] summary=${TRACE_OUTPUT_PATH%.jsonl}.summary.md"
