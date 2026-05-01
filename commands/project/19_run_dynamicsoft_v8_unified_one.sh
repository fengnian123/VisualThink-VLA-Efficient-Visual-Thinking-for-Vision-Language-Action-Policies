#!/usr/bin/env bash
set -euo pipefail

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate openvla
PYTHON_BIN="python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python not found in openvla env: $PYTHON_BIN" >&2
  exit 1
fi

DATASET="${DATASET:-taco_play}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
RUN_TAG="${RUN_TAG:-manual}"
MODEL_PATH="${MODEL_PATH:-${OPENVLA_ROOT:-$(pwd)}/models/local/openvla-7b}"
DYNAMIC_CFG="${DYNAMIC_CFG:-${OPENVLA_ROOT:-$(pwd)}/configs/openvla_soft_evidence_v4_dynamic_blend035_distill.yaml}"
GATE_CFG="${GATE_CFG:-${OPENVLA_ROOT:-$(pwd)}/configs/gating_policy_v8_sequence_text.yaml}"
SEED="${SEED:-7}"
LOG_EVERY="${LOG_EVERY:-20}"

case "$DATASET" in
  taco_play)
    RUN_ROOT="${OPENVLA_ROOT:-$(pwd)}/runs/taco_soft10shards_96ep"
    FEATURE_MANIFEST="$RUN_ROOT/03_features/feature_manifest.jsonl"
    GATE_CKPT="$RUN_ROOT/10_learned_gating_ckpt_v8"
    FULL_CKPT="$RUN_ROOT/13_openvla_soft_full_ckpt"
    UNNORM_KEY="taco_play"
    LIMIT=280
    ;;
  viola)
    RUN_ROOT="${OPENVLA_ROOT:-$(pwd)}/runs/viola_screen96"
    FEATURE_MANIFEST="$RUN_ROOT/03_features/feature_manifest.jsonl"
    GATE_CKPT="$RUN_ROOT/10_learned_gating_ckpt_v8"
    FULL_CKPT="$RUN_ROOT/13_openvla_soft_full_ckpt"
    UNNORM_KEY="viola"
    LIMIT=80
    ;;
  fractal)
    RUN_ROOT="${OPENVLA_ROOT:-$(pwd)}/runs/fractal_soft3shards"
    FEATURE_MANIFEST="$RUN_ROOT/03_features/feature_manifest.jsonl"
    GATE_CKPT="$RUN_ROOT/10_learned_gating_ckpt_v8_unified"
    FULL_CKPT="$RUN_ROOT/13_openvla_soft_full_ckpt"
    UNNORM_KEY="fractal20220817_data"
    LIMIT=946
    ;;
  stanford_hydra|hydra)
    RUN_ROOT="${OPENVLA_ROOT:-$(pwd)}/runs/hydra_soft5shards"
    FEATURE_MANIFEST="$RUN_ROOT/03_features/feature_manifest.jsonl"
    GATE_CKPT="$RUN_ROOT/10_learned_gating_ckpt_v8_unified"
    FULL_CKPT="$RUN_ROOT/13_openvla_soft_full_ckpt"
    UNNORM_KEY="stanford_hydra_dataset_converted_externally_to_rlds"
    LIMIT=48
    DATASET="stanford_hydra"
    ;;
  roboturk)
    RUN_ROOT="${OPENVLA_ROOT:-$(pwd)}/runs/roboturk_soft_all1pct"
    FEATURE_MANIFEST="$RUN_ROOT/03_features/feature_manifest.jsonl"
    GATE_CKPT="$RUN_ROOT/10_learned_gating_ckpt_v8_unified"
    FULL_CKPT="$RUN_ROOT/13_openvla_soft_full_ckpt"
    UNNORM_KEY="roboturk"
    LIMIT=36
    ;;
  *)
    echo "Unsupported DATASET=$DATASET" >&2
    exit 1
    ;;
esac

OUT_CKPT="$RUN_ROOT/14_openvla_soft_dynamic_blend035_distill_v8unified_${RUN_TAG}_ckpt"
OUT_EVAL="$RUN_ROOT/15_openvla_soft_threeway_eval_dynamic_blend035_distill_v8unified_${RUN_TAG}"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

echo "[info] dataset=$DATASET"
echo "[info] run_root=$RUN_ROOT"
echo "[info] feature_manifest=$FEATURE_MANIFEST"
echo "[info] gate_ckpt=$GATE_CKPT"
echo "[info] full_ckpt=$FULL_CKPT"
echo "[info] out_ckpt=$OUT_CKPT"
echo "[info] out_eval=$OUT_EVAL"
echo "[info] cuda_device=$CUDA_DEVICE"
echo "[info] python_bin=$PYTHON_BIN"

"$PYTHON_BIN" ${OPENVLA_ROOT:-$(pwd)}/scripts/train_openvla_soft_evidence.py \
  --feature_manifest "$FEATURE_MANIFEST" \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_CKPT" \
  --mode dynamic \
  --config "$DYNAMIC_CFG" \
  --gate_checkpoint_dir "$GATE_CKPT" \
  --gate_config "$GATE_CFG" \
  --teacher_adapter_dir "$FULL_CKPT" \
  --unnorm_key "$UNNORM_KEY" \
  --seed "$SEED" \
  --log_every "$LOG_EVERY"

"$PYTHON_BIN" ${OPENVLA_ROOT:-$(pwd)}/scripts/benchmark_openvla_soft_three_way.py \
  --feature_manifest "$FEATURE_MANIFEST" \
  --model_path "$MODEL_PATH" \
  --full_checkpoint_dir "$FULL_CKPT" \
  --dynamic_checkpoint_dir "$OUT_CKPT" \
  --output_dir "$OUT_EVAL" \
  --unnorm_key "$UNNORM_KEY" \
  --limit "$LIMIT" \
  --seed "$SEED"

echo "[ok] finished dataset=$DATASET"
echo "[ok] summary=$OUT_EVAL/summary_table.md"
