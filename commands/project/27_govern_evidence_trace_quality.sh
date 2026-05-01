#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

TRACE_INPUT="${EVIDENCE_TRACE_GOVERN_INPUT:-}"
if [[ -z "$TRACE_INPUT" ]]; then
  if [[ -f "$OPENVLA_ROOT/runs/EvidenceTrace-VLA/$RUN_NAME/evidence_trace.jsonl" ]]; then
    TRACE_INPUT="$OPENVLA_ROOT/runs/EvidenceTrace-VLA/$RUN_NAME/evidence_trace.jsonl"
  else
    TRACE_INPUT="$WORK_ROOT/16_evidence_trace/evidence_trace.jsonl"
  fi
fi

TRACE_OUTPUT_DIR="${EVIDENCE_TRACE_GOVERN_OUTPUT_DIR:-$WORK_ROOT/27_evidence_trace_governance}"
TRACE_LIMIT="${EVIDENCE_TRACE_GOVERN_LIMIT:-0}"
TRACE_CHANNELS="${EVIDENCE_TRACE_GOVERN_CHANNELS:-bbox,edge,motion,relation}"
TRACE_HQ_THRESHOLD="${EVIDENCE_TRACE_GOVERN_HQ_THRESHOLD:-0.80}"
TRACE_GOLD_THRESHOLD="${EVIDENCE_TRACE_GOVERN_GOLD_THRESHOLD:-0.90}"
TRACE_DIFFICULTY_THRESHOLDS="${EVIDENCE_TRACE_GOVERN_DIFFICULTY_THRESHOLDS:-0.3312,0.4515,0.6526,0.7509}"
TRACE_CHECK_FILES="${EVIDENCE_TRACE_GOVERN_CHECK_FILES:-0}"
TRACE_REPAIR_SELECTED="${EVIDENCE_TRACE_GOVERN_REPAIR_SELECTED:-1}"
TRACE_ENFORCE_LEGAL_REVIEW="${EVIDENCE_TRACE_GOVERN_ENFORCE_LEGAL_REVIEW:-0}"

require_file "$TRACE_INPUT"
mkdir -p "$TRACE_OUTPUT_DIR"

LIMIT_ARG=()
if [[ "$TRACE_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$TRACE_LIMIT")
fi

CHECK_FILES_ARG=()
if [[ "$TRACE_CHECK_FILES" == "1" ]]; then
  CHECK_FILES_ARG=(--check_files)
fi

REPAIR_SELECTED_ARG=()
if [[ "$TRACE_REPAIR_SELECTED" == "1" ]]; then
  REPAIR_SELECTED_ARG=(--repair_selected)
fi

LEGAL_REVIEW_ARG=()
if [[ "$TRACE_ENFORCE_LEGAL_REVIEW" == "1" ]]; then
  LEGAL_REVIEW_ARG=(--enforce_legal_review)
fi

python -u "$OPENVLA_ROOT/scripts/govern_evidence_trace_quality.py" \
  --input_trace "$TRACE_INPUT" \
  --output_dir "$TRACE_OUTPUT_DIR" \
  --channels "$TRACE_CHANNELS" \
  --hq_threshold "$TRACE_HQ_THRESHOLD" \
  --gold_threshold "$TRACE_GOLD_THRESHOLD" \
  --difficulty_thresholds "$TRACE_DIFFICULTY_THRESHOLDS" \
  "${LIMIT_ARG[@]}" \
  "${CHECK_FILES_ARG[@]}" \
  "${REPAIR_SELECTED_ARG[@]}" \
  "${LEGAL_REVIEW_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/27_govern_evidence_trace_quality.log"

echo "[ok] governed=$TRACE_OUTPUT_DIR/evidence_trace.governed.jsonl"
echo "[ok] full_clean=$TRACE_OUTPUT_DIR/release_full_clean.jsonl"
echo "[ok] hq_trace=$TRACE_OUTPUT_DIR/release_hq_trace.jsonl"
echo "[ok] gold=$TRACE_OUTPUT_DIR/release_gold_faithfulness.jsonl"
echo "[ok] review_queue=$TRACE_OUTPUT_DIR/review_queue.jsonl"
echo "[ok] summary=$TRACE_OUTPUT_DIR/quality_summary.md"
