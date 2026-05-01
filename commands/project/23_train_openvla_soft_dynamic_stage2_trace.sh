#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

FEATURE_MANIFEST="${OPENVLA_STAGE2_FEATURE_MANIFEST:-$WORK_ROOT/03_features/feature_manifest.jsonl}"
GATE_CKPT_DIR="${OPENVLA_STAGE2_GATE_CKPT_DIR:-$WORK_ROOT/10_learned_gating_ckpt}"
TEACHER_ADAPTER_DIR="${OPENVLA_STAGE2_TEACHER_ADAPTER_DIR:-$WORK_ROOT/13_openvla_soft_full_ckpt}"
OUT_DIR="${OPENVLA_STAGE2_OUTPUT_DIR:-$WORK_ROOT/23_openvla_soft_dynamic_stage2_trace_ckpt}"
TRACE_CONFIG="${OPENVLA_STAGE2_TRACE_CONFIG:-$OPENVLA_ROOT/configs/openvla_soft_evidence_v5_dynamic_blend035_distill_traceaux.yaml}"
TRACE_GATE_CONFIG="${OPENVLA_STAGE2_GATE_CONFIG:-$OPENVLA_SOFT_DYNAMIC_GATE_CONFIG}"

if [[ -n "${OPENVLA_STAGE2_INIT_ADAPTER_DIR:-}" ]]; then
  INIT_ADAPTER_DIR="$OPENVLA_STAGE2_INIT_ADAPTER_DIR"
else
  preferred_candidates=(
    "$WORK_ROOT/14_openvla_soft_dynamic_ckpt"
    "$WORK_ROOT/14_openvla_soft_dynamic_blend035_distill_v8_ckpt"
    "$WORK_ROOT/14_openvla_soft_dynamic_blend035_distill_v4relation_ckpt"
  )
  INIT_ADAPTER_DIR=""
  for candidate in "${preferred_candidates[@]}"; do
    if [[ -f "$candidate/adapter.pt" ]]; then
      INIT_ADAPTER_DIR="$candidate"
      break
    fi
  done
  if [[ -z "$INIT_ADAPTER_DIR" ]]; then
    shopt -s nullglob
    candidates=("$WORK_ROOT"/14_openvla_soft_dynamic*_ckpt)
    shopt -u nullglob
    for candidate in "${candidates[@]}"; do
      if [[ "$candidate" == *bboxmotion* ]]; then
        continue
      fi
      if [[ -f "$candidate/adapter.pt" ]]; then
        INIT_ADAPTER_DIR="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$INIT_ADAPTER_DIR" ]]; then
    shopt -s nullglob
    candidates=("$WORK_ROOT"/14_openvla_soft_dynamic*_ckpt)
    shopt -u nullglob
    if [[ "${#candidates[@]}" -gt 0 ]]; then
      INIT_ADAPTER_DIR="${candidates[0]}"
    else
      INIT_ADAPTER_DIR="$WORK_ROOT/14_openvla_soft_dynamic_ckpt"
    fi
  fi
fi

if [[ -n "${OPENVLA_STAGE2_TRACE_MANIFEST:-}" ]]; then
  TRACE_MANIFEST="$OPENVLA_STAGE2_TRACE_MANIFEST"
elif [[ -f "$OPENVLA_ROOT/runs/EvidenceTrace-VLA/$RUN_NAME/evidence_trace.jsonl" ]]; then
  TRACE_MANIFEST="$OPENVLA_ROOT/runs/EvidenceTrace-VLA/$RUN_NAME/evidence_trace.jsonl"
else
  TRACE_MANIFEST="$WORK_ROOT/16_evidence_trace/evidence_trace.jsonl"
fi

require_file "$FEATURE_MANIFEST"
require_file "$TRACE_CONFIG"
require_file "$TRACE_GATE_CONFIG"
require_dir "$GATE_CKPT_DIR"
require_file "$GATE_CKPT_DIR/gate.pt"
require_dir "$TEACHER_ADAPTER_DIR"
require_file "$TEACHER_ADAPTER_DIR/adapter.pt"
require_dir "$INIT_ADAPTER_DIR"
require_file "$INIT_ADAPTER_DIR/adapter.pt"
require_file "$TRACE_MANIFEST"

LIMIT_ARG=()
if [[ "$OPENVLA_SOFT_LIMIT" != "0" ]]; then
  LIMIT_ARG=(--limit "$OPENVLA_SOFT_LIMIT")
fi

python -u "$OPENVLA_ROOT/scripts/train_openvla_soft_evidence.py" \
  --feature_manifest "$FEATURE_MANIFEST" \
  --model_path "$VLA_PATH" \
  --output_dir "$OUT_DIR" \
  --config "$TRACE_CONFIG" \
  --mode dynamic \
  --unnorm_key "$OPENVLA_SOFT_UNNORM_KEY" \
  --gate_checkpoint_dir "$GATE_CKPT_DIR" \
  --gate_config "$TRACE_GATE_CONFIG" \
  --teacher_adapter_dir "$TEACHER_ADAPTER_DIR" \
  --init_adapter_dir "$INIT_ADAPTER_DIR" \
  --evidence_trace_manifest "$TRACE_MANIFEST" \
  --log_every "$OPENVLA_SOFT_LOG_EVERY" \
  "${LIMIT_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/23_train_openvla_soft_dynamic_stage2_trace.log"

echo "[ok] output=$OUT_DIR"
echo "[ok] trace_manifest=$TRACE_MANIFEST"
echo "[ok] init_adapter_dir=$INIT_ADAPTER_DIR"
