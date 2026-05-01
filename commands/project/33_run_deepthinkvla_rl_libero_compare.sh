#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ROOT:-$(pwd)}"
DEEPTHINK_ROOT="$ROOT/models/local/DeepThinkVLA"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/reasoning_vla_paper_metrics/deepthink_rl_compare}"
GPU_ID="${GPU_ID:-0}"
TRIALS="${TRIALS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
CKPT="${CKPT:-$DEEPTHINK_ROOT/yinchenghust/deepthinkvla_libero_cot_rl}"
MODEL_NAME="${MODEL_NAME:-DeepThinkVLA-RL}"

source "$(conda info --base)/etc/profile.d/conda.sh"

activate_env() {
  set +u
  conda activate "$1"
  set -u
}

mkdir -p "$OUT_ROOT"
activate_env deepthinkvla
cd "$DEEPTHINK_ROOT"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

for SUITE in $SUITES; do
  OUT_DIR="$OUT_ROOT/$SUITE"
  mkdir -p "$OUT_DIR"
  echo "[start] model=$MODEL_NAME suite=$SUITE gpu=$GPU_ID trials=$TRIALS out=$OUT_DIR"
  /usr/bin/time -v \
    -o "$OUT_DIR/time.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    python -m experiments.run_libero_eval \
      --pretrained_checkpoint "$CKPT" \
      --num_images_in_input 2 \
      --task_suite_name "$SUITE" \
      --num_trials_per_task "$TRIALS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --swanlab_mode disabled \
      --local_log_dir "$OUT_DIR"
  echo "[done] model=$MODEL_NAME suite=$SUITE"
done

echo "[all-done] model=$MODEL_NAME"
