#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
SESSION="${SESSION:-internal_interface_20260430}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/internal_interface_20260430}"
TRAIN_COUNT="${TRAIN_COUNT:-256}"
EVAL_COUNT="${EVAL_COUNT:-32}"
PROMPT_SAMPLE_COUNT="${PROMPT_SAMPLE_COUNT:-32}"
SEED="${SEED:-7}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
QUEUE_GPU0="${QUEUE_GPU0:-bridge libero_object roboturk}"
QUEUE_GPU1="${QUEUE_GPU1:-fractal libero_goal utaustin_mutex}"
QUEUE_GPU2="${QUEUE_GPU2:-libero_long libero_spatial}"
PROMPT_VARIANT="${PROMPT_VARIANT:-full_schema}"
DEPTH_VARIANT="${DEPTH_VARIANT:-depth5_blend_0p35}"

mkdir -p "$OUT_ROOT"

DATASET_RUNNER="$OUT_ROOT/run_interface_dataset.sh"
cat > "$DATASET_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

DATASET_KEY="$1"
GPU_ID="$2"
OPENVLA_ROOT="$3"
PYTHON_BIN="$4"
MODEL_PATH="$5"
OUT_ROOT="$6"
TRAIN_COUNT="$7"
EVAL_COUNT="$8"
PROMPT_SAMPLE_COUNT="$9"
SEED="${10}"

cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

case "$DATASET_KEY" in
  bridge)
    SOURCE_RUN_DIR="runs/bridge_full"
    DATASET_ENV="bridge"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_ckpt"
    ;;
  fractal)
    SOURCE_RUN_DIR="runs/fractal_official_full"
    DATASET_ENV="fractal"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_blend035_distill_v8_ckpt"
    ;;
  roboturk)
    SOURCE_RUN_DIR="runs/roboturk_official_full"
    DATASET_ENV="roboturk"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_blend035_distill_v8_ckpt"
    ;;
  libero_long)
    SOURCE_RUN_DIR="runs/libero10_full_all"
    DATASET_ENV="libero"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_ckpt"
    ;;
  libero_goal)
    SOURCE_RUN_DIR="runs/libero_goal_full_all"
    DATASET_ENV="libero_goal"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_ckpt"
    ;;
  libero_object)
    SOURCE_RUN_DIR="runs/libero_object_full_all"
    DATASET_ENV="libero_object"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_ckpt"
    ;;
  libero_spatial)
    SOURCE_RUN_DIR="runs/libero_spatial_full_all"
    DATASET_ENV="libero_spatial"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_ckpt"
    ;;
  utaustin_mutex)
    SOURCE_RUN_DIR="runs/utaustin_mutex"
    DATASET_ENV="utaustin_mutex"
    DYNAMIC_CKPT_REL="14_openvla_soft_dynamic_ckpt"
    ;;
  *)
    echo "[error] unsupported dataset=$DATASET_KEY" >&2
    exit 1
    ;;
esac

RUN_DIR="$OUT_ROOT/$DATASET_KEY"
SPLIT_DIR="$RUN_DIR/00_split"
LOG_DIR="$RUN_DIR/logs"
FEATURE_MANIFEST="$OPENVLA_ROOT/$SOURCE_RUN_DIR/03_features/feature_manifest.jsonl"
FULL_CKPT_DIR="$OPENVLA_ROOT/$SOURCE_RUN_DIR/13_openvla_soft_full_ckpt"
GATE_CKPT_DIR="$OPENVLA_ROOT/$SOURCE_RUN_DIR/10_learned_gating_ckpt"
DYNAMIC_CKPT_DIR="$OPENVLA_ROOT/$SOURCE_RUN_DIR/$DYNAMIC_CKPT_REL"

mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "[start] dataset=$DATASET_KEY gpu=$GPU_ID $(date '+%F %T')"

(
  /usr/bin/time -v -o "$LOG_DIR/time_split.txt" \
    "$PYTHON_BIN" -u scripts/prepare_recipe_ablation_split.py \
      --feature_manifest "$FEATURE_MANIFEST" \
      --output_dir "$SPLIT_DIR" \
      --train_count "$TRAIN_COUNT" \
      --eval_count "$EVAL_COUNT" \
      --sample_strategy random \
      --seed "$SEED"
) 2>&1 | tee "$LOG_DIR/00_split.log"

TRAIN_MANIFEST="$SPLIT_DIR/train_manifest.jsonl"
EVAL_MANIFEST="$SPLIT_DIR/eval_manifest.jsonl"

(
  /usr/bin/time -v -o "$LOG_DIR/time_threeway_subset.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
      DATASET="$DATASET_ENV" \
      RUN_NAME="internal_interface_${DATASET_KEY}" \
      WORK_ROOT="$RUN_DIR" \
      VLA_PATH="$MODEL_PATH" \
      OPENVLA_SOFT_FEATURE_MANIFEST="$EVAL_MANIFEST" \
      OPENVLA_SOFT_FULL_CKPT_DIR="$FULL_CKPT_DIR" \
      OPENVLA_SOFT_DYNAMIC_CKPT_DIR="$DYNAMIC_CKPT_DIR" \
      OPENVLA_SOFT_GATE_CKPT_DIR="$GATE_CKPT_DIR" \
      OPENVLA_SOFT_BENCHMARK_OUT_DIR="$RUN_DIR/15_interface_eval" \
      OPENVLA_SOFT_PAPER_OUT_DIR="$RUN_DIR/31_interface_metrics" \
      OPENVLA_SOFT_THREEWAY_LIMIT="$EVAL_COUNT" \
      OPENVLA_SOFT_THREEWAY_SKIP_EMPTY="0" \
      OPENVLA_PAPER_METRICS_MODE="episode_proxy" \
      "$OPENVLA_ROOT/commands/project/31_benchmark_openvla_soft_paper_metrics.sh"
) 2>&1 | tee "$LOG_DIR/31_interface_metrics.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_prompt.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_prompt_channel_screening.py \
      --dataset "$DATASET_KEY" \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/30_prompt_screening" \
      --sample_count "$PROMPT_SAMPLE_COUNT" \
      --sample_strategy random \
      --seed "$SEED" \
      --unnorm_key bridge_orig
) 2>&1 | tee "$LOG_DIR/30_prompt_screening.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_gate_depth5.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_learned_gating.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --output_dir "$RUN_DIR/10_gate_depth5_ckpt" \
      --config "$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_depth_recipe.yaml" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/10_gate_depth5.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_dynamic_depth5_blend035.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_openvla_soft_evidence.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/14_dynamic_depth5_blend035_ckpt" \
      --config "$OPENVLA_ROOT/configs/openvla_soft_evidence_recipe_blend035_depth5.yaml" \
      --mode dynamic \
      --gate_checkpoint_dir "$RUN_DIR/10_gate_depth5_ckpt" \
      --gate_config "$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_depth_recipe.yaml" \
      --unnorm_key bridge_orig \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/14_dynamic_depth5_blend035.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_depth5_eval.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_soft_checkpoints.py \
      --dataset "$DATASET_KEY" \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/20_depth5_eval" \
      --unnorm_key bridge_orig \
      --seed "$SEED" \
      --variant "depth5_blend_0p35=$RUN_DIR/14_dynamic_depth5_blend035_ckpt"
) 2>&1 | tee "$LOG_DIR/20_depth5_eval.log"

touch "$RUN_DIR/DONE"
echo "[done] dataset=$DATASET_KEY $(date '+%F %T')"
EOF
chmod +x "$DATASET_RUNNER"

QUEUE_RUNNER="$OUT_ROOT/run_dataset_queue.sh"
cat > "$QUEUE_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
OPENVLA_ROOT="$2"
PYTHON_BIN="$3"
MODEL_PATH="$4"
OUT_ROOT="$5"
TRAIN_COUNT="$6"
EVAL_COUNT="$7"
PROMPT_SAMPLE_COUNT="$8"
SEED="$9"
shift 9

RUNNER="$OUT_ROOT/run_interface_dataset.sh"
for dataset_key in "$@"; do
  "$RUNNER" "$dataset_key" "$GPU_ID" "$OPENVLA_ROOT" "$PYTHON_BIN" "$MODEL_PATH" "$OUT_ROOT" "$TRAIN_COUNT" "$EVAL_COUNT" "$PROMPT_SAMPLE_COUNT" "$SEED"
done
EOF
chmod +x "$QUEUE_RUNNER"

SUMMARY_RUNNER="$OUT_ROOT/run_summary_waiter.sh"
cat > "$SUMMARY_RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail

EXPECTED_KEYS=(bridge fractal roboturk libero_object libero_goal libero_spatial libero_long utaustin_mutex)
echo "[wait] internal-interface summary watcher started \$(date '+%F %T')"
while true; do
  ready=1
  for key in "\${EXPECTED_KEYS[@]}"; do
    if [[ ! -f "$OUT_ROOT/\$key/DONE" ]]; then
      ready=0
      break
    fi
  done
  if [[ "\$ready" == "1" ]]; then
    break
  fi
  sleep 60
done
mkdir -p "$OUT_ROOT/30_summary"
"$PYTHON_BIN" -u "$OPENVLA_ROOT/scripts/summarize_internal_interface_comparison.py" \\
  --run_root "$OUT_ROOT" \\
  --output_dir "$OUT_ROOT/30_summary" \\
  --prompt_variant "$PROMPT_VARIANT" \\
  --depth_variant "$DEPTH_VARIANT" \\
  2>&1 | tee "$OUT_ROOT/30_summary/run.log"
echo "[done] internal-interface summary watcher finished \$(date '+%F %T')"
EOF
chmod +x "$SUMMARY_RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

read -r -a DATASETS_GPU0 <<< "$QUEUE_GPU0"
read -r -a DATASETS_GPU1 <<< "$QUEUE_GPU1"
read -r -a DATASETS_GPU2 <<< "$QUEUE_GPU2"

tmux new-session -d -s "$SESSION" -n gpu0 \
  "$QUEUE_RUNNER $GPU0 $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $PROMPT_SAMPLE_COUNT $SEED ${DATASETS_GPU0[*]}"
tmux new-window -t "$SESSION" -n gpu1 \
  "$QUEUE_RUNNER $GPU1 $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $PROMPT_SAMPLE_COUNT $SEED ${DATASETS_GPU1[*]}"
tmux new-window -t "$SESSION" -n gpu2 \
  "$QUEUE_RUNNER $GPU2 $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $PROMPT_SAMPLE_COUNT $SEED ${DATASETS_GPU2[*]}"
tmux new-window -t "$SESSION" -n summary "$SUMMARY_RUNNER"

echo "[ok] started tmux session=$SESSION"
echo "[ok] out_root=$OUT_ROOT"
echo "[ok] gpu0=$GPU0 queue=${DATASETS_GPU0[*]}"
echo "[ok] gpu1=$GPU1 queue=${DATASETS_GPU1[*]}"
echo "[ok] gpu2=$GPU2 queue=${DATASETS_GPU2[*]}"
echo "[hint] tmux attach -t $SESSION"
