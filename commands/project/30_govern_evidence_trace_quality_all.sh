#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GOVERN_LIMIT="${EVIDENCE_TRACE_GOVERN_LIMIT:-0}"
GOVERN_REPAIR_SELECTED="${EVIDENCE_TRACE_GOVERN_REPAIR_SELECTED:-1}"
GOVERN_HQ_THRESHOLD="${EVIDENCE_TRACE_GOVERN_HQ_THRESHOLD:-0.80}"
GOVERN_GOLD_THRESHOLD="${EVIDENCE_TRACE_GOVERN_GOLD_THRESHOLD:-0.90}"
GOVERN_CHECK_FILES="${EVIDENCE_TRACE_GOVERN_CHECK_FILES:-0}"
GOVERN_ENFORCE_LEGAL_REVIEW="${EVIDENCE_TRACE_GOVERN_ENFORCE_LEGAL_REVIEW:-0}"
GOVERN_SKIP_MISSING_INPUT="${EVIDENCE_TRACE_GOVERN_SKIP_MISSING_INPUT:-0}"

SPECS=(
  "bridge bridge_full"
  "fractal fractal_official_full"
  "libero libero10_full_all"
  "libero_goal libero_goal_full_all"
  "libero_object libero_object_full_all"
  "libero_spatial libero_spatial_full_all"
  "roboturk roboturk_official_full"
  "utaustin_mutex utaustin_mutex"
  "viola viola_official_full"
)

echo "[info] governance_limit=$GOVERN_LIMIT"
echo "[info] repair_selected=$GOVERN_REPAIR_SELECTED"
echo "[info] hq_threshold=$GOVERN_HQ_THRESHOLD gold_threshold=$GOVERN_GOLD_THRESHOLD"
echo "[info] check_files=$GOVERN_CHECK_FILES enforce_legal_review=$GOVERN_ENFORCE_LEGAL_REVIEW"
echo "[info] skip_missing_input=$GOVERN_SKIP_MISSING_INPUT"

for spec in "${SPECS[@]}"; do
  set -- $spec
  DATASET_KEY="$1"
  RUN="$2"

  TRACE_INPUT="${OPENVLA_ROOT:-$(pwd)}/runs/EvidenceTrace-VLA/$RUN/evidence_trace.jsonl"
  TRACE_OUTPUT_DIR="${OPENVLA_ROOT:-$(pwd)}/runs/EvidenceTrace-VLA/$RUN/governance"

  if [[ ! -f "$TRACE_INPUT" ]]; then
    if [[ "$GOVERN_SKIP_MISSING_INPUT" == "1" ]]; then
      echo "[skip] missing_trace_input run=$RUN path=$TRACE_INPUT"
      continue
    fi
    echo "[error] missing_trace_input run=$RUN path=$TRACE_INPUT" >&2
    exit 1
  fi

  echo "[run] dataset=$DATASET_KEY run=$RUN"
  RUN_NAME="$RUN" \
  DATASET="$DATASET_KEY" \
  EVIDENCE_TRACE_GOVERN_INPUT="$TRACE_INPUT" \
  EVIDENCE_TRACE_GOVERN_OUTPUT_DIR="$TRACE_OUTPUT_DIR" \
  EVIDENCE_TRACE_GOVERN_LIMIT="$GOVERN_LIMIT" \
  EVIDENCE_TRACE_GOVERN_REPAIR_SELECTED="$GOVERN_REPAIR_SELECTED" \
  EVIDENCE_TRACE_GOVERN_HQ_THRESHOLD="$GOVERN_HQ_THRESHOLD" \
  EVIDENCE_TRACE_GOVERN_GOLD_THRESHOLD="$GOVERN_GOLD_THRESHOLD" \
  EVIDENCE_TRACE_GOVERN_CHECK_FILES="$GOVERN_CHECK_FILES" \
  EVIDENCE_TRACE_GOVERN_ENFORCE_LEGAL_REVIEW="$GOVERN_ENFORCE_LEGAL_REVIEW" \
  bash "$SCRIPT_DIR/27_govern_evidence_trace_quality.sh"
done

echo "[ok] finished batch evidence-trace governance"
