#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

STAGE2_PIPELINE_RUN_TRAIN="${STAGE2_PIPELINE_RUN_TRAIN:-1}"
STAGE2_PIPELINE_RUN_BENCHMARK="${STAGE2_PIPELINE_RUN_BENCHMARK:-1}"
STAGE2_PIPELINE_RUN_FAITHFULNESS="${STAGE2_PIPELINE_RUN_FAITHFULNESS:-1}"

STAGE2_OUTPUT_DIR="${OPENVLA_STAGE2_OUTPUT_DIR:-$WORK_ROOT/23_openvla_soft_dynamic_stage2_trace_ckpt}"
STAGE2_BENCHMARK_OUT_DIR="${OPENVLA_SOFT_BENCHMARK_OUT_DIR:-$WORK_ROOT/24_openvla_soft_threeway_eval_stage2_trace}"
STAGE2_FAITHFULNESS_OUT_DIR="${EVIDENCE_TRACE_FAITHFULNESS_OUT_DIR:-$WORK_ROOT/24_evidence_trace_faithfulness_stage2_trace}"
STAGE2_TRACE_MANIFEST="${OPENVLA_STAGE2_TRACE_MANIFEST:-$OPENVLA_ROOT/runs/EvidenceTrace-VLA/$RUN_NAME/evidence_trace.jsonl}"
STAGE2_FEATURE_MANIFEST="${OPENVLA_STAGE2_FEATURE_MANIFEST:-$WORK_ROOT/03_features/feature_manifest.jsonl}"

if [[ -z "${OPENVLA_SOFT_UNNORM_KEY:-}" ]]; then
  echo "OPENVLA_SOFT_UNNORM_KEY is required for stage-2 pipeline" >&2
  exit 1
fi

require_file "$STAGE2_FEATURE_MANIFEST"
require_file "$STAGE2_TRACE_MANIFEST"

echo "[info] run_name=$RUN_NAME"
echo "[info] dataset=$DATASET"
echo "[info] stage2_output_dir=$STAGE2_OUTPUT_DIR"
echo "[info] benchmark_out_dir=$STAGE2_BENCHMARK_OUT_DIR"
echo "[info] faithfulness_out_dir=$STAGE2_FAITHFULNESS_OUT_DIR"
echo "[info] trace_manifest=$STAGE2_TRACE_MANIFEST"
echo "[info] feature_manifest=$STAGE2_FEATURE_MANIFEST"

if [[ "$STAGE2_PIPELINE_RUN_TRAIN" == "1" ]]; then
  echo "[stage] stage2 trace training"
  OPENVLA_STAGE2_OUTPUT_DIR="$STAGE2_OUTPUT_DIR" \
  OPENVLA_STAGE2_TRACE_MANIFEST="$STAGE2_TRACE_MANIFEST" \
  OPENVLA_STAGE2_FEATURE_MANIFEST="$STAGE2_FEATURE_MANIFEST" \
  bash "$SCRIPT_DIR/23_train_openvla_soft_dynamic_stage2_trace.sh"
else
  echo "[skip] stage2 trace training"
fi

require_dir "$STAGE2_OUTPUT_DIR"
require_file "$STAGE2_OUTPUT_DIR/adapter.pt"
require_file "$STAGE2_OUTPUT_DIR/channel_masks.npy"

if [[ "$STAGE2_PIPELINE_RUN_BENCHMARK" == "1" ]]; then
  echo "[stage] three-way benchmark"
  OPENVLA_SOFT_FEATURE_MANIFEST="$STAGE2_FEATURE_MANIFEST" \
  OPENVLA_SOFT_DYNAMIC_CKPT_DIR="$STAGE2_OUTPUT_DIR" \
  OPENVLA_SOFT_BENCHMARK_OUT_DIR="$STAGE2_BENCHMARK_OUT_DIR" \
  bash "$SCRIPT_DIR/18_benchmark_openvla_soft_three_way.sh"
else
  echo "[skip] three-way benchmark"
fi

if [[ "$STAGE2_PIPELINE_RUN_FAITHFULNESS" == "1" ]]; then
  echo "[stage] evidence-trace faithfulness benchmark"
  EVIDENCE_TRACE_FEATURE_MANIFEST="$STAGE2_FEATURE_MANIFEST" \
  EVIDENCE_TRACE_DYNAMIC_CKPT_DIR="$STAGE2_OUTPUT_DIR" \
  EVIDENCE_TRACE_MANIFEST="$STAGE2_TRACE_MANIFEST" \
  EVIDENCE_TRACE_FAITHFULNESS_OUT_DIR="$STAGE2_FAITHFULNESS_OUT_DIR" \
  bash "$SCRIPT_DIR/22_benchmark_evidence_trace_faithfulness.sh"
else
  echo "[skip] evidence-trace faithfulness benchmark"
fi

echo "[ok] stage2_ckpt=$STAGE2_OUTPUT_DIR"
if [[ "$STAGE2_PIPELINE_RUN_BENCHMARK" == "1" ]]; then
  echo "[ok] benchmark_summary=$STAGE2_BENCHMARK_OUT_DIR/summary_table.md"
fi
if [[ "$STAGE2_PIPELINE_RUN_FAITHFULNESS" == "1" ]]; then
  echo "[ok] faithfulness_summary=$STAGE2_FAITHFULNESS_OUT_DIR/summary_table.md"
fi
