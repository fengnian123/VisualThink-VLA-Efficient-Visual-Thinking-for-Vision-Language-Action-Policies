#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

TRACE_FEATURE_MANIFEST="${EVIDENCE_TRACE_FEATURE_MANIFEST:-$WORK_ROOT/03_features/feature_manifest.jsonl}"
TRACE_DYNAMIC_CKPT_DIR="${EVIDENCE_TRACE_DYNAMIC_CKPT_DIR:-$WORK_ROOT/14_openvla_soft_dynamic_ckpt}"
TRACE_MANIFEST="${EVIDENCE_TRACE_MANIFEST:-$WORK_ROOT/16_evidence_trace/evidence_trace.jsonl}"
TRACE_FAITHFULNESS_OUT_DIR="${EVIDENCE_TRACE_FAITHFULNESS_OUT_DIR:-$WORK_ROOT/16_evidence_trace/faithfulness_eval}"
TRACE_FAITHFULNESS_LIMIT="${EVIDENCE_TRACE_FAITHFULNESS_LIMIT:-4}"
TRACE_FAITHFULNESS_SUCCESS_L1_THRESH="${EVIDENCE_TRACE_FAITHFULNESS_SUCCESS_L1_THRESH:-$SUCCESS_L1_THRESH}"
TRACE_FAITHFULNESS_MASK_THRESHOLD="${EVIDENCE_TRACE_FAITHFULNESS_MASK_THRESHOLD:-0.5}"
TRACE_FAITHFULNESS_SHUFFLE_CHANNELS="${EVIDENCE_TRACE_FAITHFULNESS_SHUFFLE_CHANNELS:-relation,motion}"

require_file "$TRACE_FEATURE_MANIFEST"
require_dir "$TRACE_DYNAMIC_CKPT_DIR"
require_file "$TRACE_DYNAMIC_CKPT_DIR/adapter.pt"
require_file "$TRACE_DYNAMIC_CKPT_DIR/channel_masks.npy"
require_file "$TRACE_MANIFEST"

python -u "$OPENVLA_ROOT/scripts/benchmark_evidence_trace_faithfulness.py" \
  --feature_manifest "$TRACE_FEATURE_MANIFEST" \
  --dynamic_checkpoint_dir "$TRACE_DYNAMIC_CKPT_DIR" \
  --evidence_trace_manifest "$TRACE_MANIFEST" \
  --model_path "$VLA_PATH" \
  --output_dir "$TRACE_FAITHFULNESS_OUT_DIR" \
  --unnorm_key "$OPENVLA_SOFT_UNNORM_KEY" \
  --limit "$TRACE_FAITHFULNESS_LIMIT" \
  --success_l1_thresh "$TRACE_FAITHFULNESS_SUCCESS_L1_THRESH" \
  --mask_threshold "$TRACE_FAITHFULNESS_MASK_THRESHOLD" \
  --shuffle_channels "$TRACE_FAITHFULNESS_SHUFFLE_CHANNELS" \
  2>&1 | tee "$WORK_ROOT/logs/22_benchmark_evidence_trace_faithfulness.log"

echo "[ok] summary=$TRACE_FAITHFULNESS_OUT_DIR/summary_table.md"
