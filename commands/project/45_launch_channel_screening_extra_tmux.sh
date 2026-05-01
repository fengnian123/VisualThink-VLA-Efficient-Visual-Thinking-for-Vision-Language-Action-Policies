#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
SESSION="${SESSION:-channel_screening_extra_20260430}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/channel_screening_20260429/40_extra_datasets}"
TRAIN_COUNT="${TRAIN_COUNT:-2048}"
EVAL_COUNT="${EVAL_COUNT:-1024}"
PROMPT_SAMPLE_COUNT="${PROMPT_SAMPLE_COUNT:-256}"
GPU_FRACTAL="${GPU_FRACTAL:-0}"
GPU_ROBOTURK="${GPU_ROBOTURK:-1}"
GPU_MUTEX="${GPU_MUTEX:-2}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

DATASET_RUNNER="$OUT_ROOT/run_extra_dataset.sh"
cat > "$DATASET_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="$1"
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

case "$DATASET_NAME" in
  fractal)
    FEATURE_MANIFEST="runs/fractal_official_full/03_features/feature_manifest.jsonl"
    RUN_KEY="fractal"
    DATASET_LABEL="fractal_official_full"
    UNNORM_KEY="bridge_orig"
    ;;
  roboturk)
    FEATURE_MANIFEST="runs/roboturk_official_full/03_features/feature_manifest.jsonl"
    RUN_KEY="roboturk"
    DATASET_LABEL="roboturk_official_full"
    UNNORM_KEY="bridge_orig"
    ;;
  utaustin_mutex)
    FEATURE_MANIFEST="runs/utaustin_mutex/03_features/feature_manifest.jsonl"
    RUN_KEY="utaustin_mutex"
    DATASET_LABEL="utaustin_mutex"
    UNNORM_KEY="bridge_orig"
    ;;
  *)
    echo "[error] unsupported dataset=$DATASET_NAME" >&2
    exit 1
    ;;
esac

RUN_DIR="$OUT_ROOT/$RUN_KEY"
SPLIT_DIR="$RUN_DIR/00_split"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

echo "[start] dataset=$DATASET_NAME gpu=$GPU_ID $(date '+%F %T')"

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
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/14_dynamic_depth5_blend035.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_depth5_eval.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_soft_checkpoints.py \
      --dataset "$DATASET_LABEL" \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/20_depth5_eval" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --variant "depth5_blend_0p35=$RUN_DIR/14_dynamic_depth5_blend035_ckpt"
) 2>&1 | tee "$LOG_DIR/20_depth5_eval.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_prompt.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_prompt_channel_screening.py \
      --dataset "$DATASET_LABEL" \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/30_prompt_screening" \
      --sample_count "$PROMPT_SAMPLE_COUNT" \
      --sample_strategy random \
      --seed "$SEED" \
      --unnorm_key "$UNNORM_KEY"
) 2>&1 | tee "$LOG_DIR/30_prompt_screening.log"

touch "$RUN_DIR/DONE"
echo "[done] dataset=$DATASET_NAME $(date '+%F %T')"
EOF
chmod +x "$DATASET_RUNNER"

SUMMARY_RUNNER="$OUT_ROOT/run_summary_waiter.sh"
cat > "$SUMMARY_RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$OPENVLA_ROOT"
echo "[wait] summary watcher started \$(date '+%F %T')"
while [[ ! -f "$OUT_ROOT/fractal/DONE" || ! -f "$OUT_ROOT/roboturk/DONE" || ! -f "$OUT_ROOT/utaustin_mutex/DONE" ]]; do
  sleep 60
done
mkdir -p "$OUT_ROOT/30_summary"
"$PYTHON_BIN" -u scripts/summarize_channel_screening_extra.py \\
  --spec "Fractal=$OUT_ROOT/fractal/30_prompt_screening/summary.json=$OUT_ROOT/fractal/20_depth5_eval/summary.json" \\
  --spec "RoboTurk=$OUT_ROOT/roboturk/30_prompt_screening/summary.json=$OUT_ROOT/roboturk/20_depth5_eval/summary.json" \\
  --spec "UT Austin MUTEX=$OUT_ROOT/utaustin_mutex/30_prompt_screening/summary.json=$OUT_ROOT/utaustin_mutex/20_depth5_eval/summary.json" \\
  --output_dir "$OUT_ROOT/30_summary" \\
  2>&1 | tee "$OUT_ROOT/30_summary/run.log"
echo "[done] summary watcher finished \$(date '+%F %T')"
EOF
chmod +x "$SUMMARY_RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n fractal \
  "$DATASET_RUNNER fractal $GPU_FRACTAL $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $PROMPT_SAMPLE_COUNT $SEED"
tmux new-window -t "$SESSION" -n roboturk \
  "$DATASET_RUNNER roboturk $GPU_ROBOTURK $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $PROMPT_SAMPLE_COUNT $SEED"
tmux new-window -t "$SESSION" -n mutex \
  "$DATASET_RUNNER utaustin_mutex $GPU_MUTEX $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $PROMPT_SAMPLE_COUNT $SEED"
tmux new-window -t "$SESSION" -n summary "$SUMMARY_RUNNER"

echo "[ok] started tmux session=$SESSION"
echo "[ok] fractal_gpu=$GPU_FRACTAL roboturk_gpu=$GPU_ROBOTURK mutex_gpu=$GPU_MUTEX"
echo "[ok] out_root=$OUT_ROOT"
echo "[hint] tmux attach -t $SESSION"
