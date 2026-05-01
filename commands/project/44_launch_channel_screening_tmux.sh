#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
SESSION="${SESSION:-channel_screening_20260429}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/channel_screening_20260429}"
GPU_BRIDGE="${GPU_BRIDGE:-1}"
GPU_LIBERO="${GPU_LIBERO:-2}"
GPU_PROMPT="${GPU_PROMPT:-0}"
PROMPT_SAMPLE_COUNT="${PROMPT_SAMPLE_COUNT:-384}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

BRIDGE_SPLIT_SRC="${OPENVLA_ROOT}/runs/recipe_training_ablation_20260428/bridge/00_split"
LIBERO_SPLIT_SRC="${OPENVLA_ROOT}/runs/recipe_training_ablation_20260428/libero_long/00_split"
mkdir -p "$OUT_ROOT/bridge/00_split" "$OUT_ROOT/libero_long/00_split"
cp "$BRIDGE_SPLIT_SRC"/train_manifest.jsonl "$OUT_ROOT/bridge/00_split/train_manifest.jsonl"
cp "$BRIDGE_SPLIT_SRC"/eval_manifest.jsonl "$OUT_ROOT/bridge/00_split/eval_manifest.jsonl"
cp "$LIBERO_SPLIT_SRC"/train_manifest.jsonl "$OUT_ROOT/libero_long/00_split/train_manifest.jsonl"
cp "$LIBERO_SPLIT_SRC"/eval_manifest.jsonl "$OUT_ROOT/libero_long/00_split/eval_manifest.jsonl"

DATASET_RUNNER="$OUT_ROOT/run_depth_dataset.sh"
cat > "$DATASET_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="$1"
GPU_ID="$2"
OPENVLA_ROOT="$3"
PYTHON_BIN="$4"
MODEL_PATH="$5"
OUT_ROOT="$6"
SEED="$7"

cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

case "$DATASET_NAME" in
  bridge)
    FULL_FEATURE_MANIFEST="runs/bridge_full/03_features/feature_manifest.jsonl"
    TRAIN_MANIFEST="$OUT_ROOT/bridge/00_split/train_manifest.jsonl"
    EVAL_MANIFEST="$OUT_ROOT/bridge/00_split/eval_manifest.jsonl"
    UNNORM_KEY="bridge_orig"
    ;;
  libero_long)
    FULL_FEATURE_MANIFEST="runs/libero10_full_all/03_features/feature_manifest.jsonl"
    TRAIN_MANIFEST="$OUT_ROOT/libero_long/00_split/train_manifest.jsonl"
    EVAL_MANIFEST="$OUT_ROOT/libero_long/00_split/eval_manifest.jsonl"
    UNNORM_KEY="bridge_orig"
    ;;
  *)
    echo "[error] unsupported dataset=$DATASET_NAME" >&2
    exit 1
    ;;
esac

RUN_DIR="$OUT_ROOT/$DATASET_NAME"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

echo "[start] dataset=$DATASET_NAME gpu=$GPU_ID $(date '+%F %T')"

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
      --dataset "$DATASET_NAME" \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/20_depth5_eval" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --variant "depth5_blend_0p35=$RUN_DIR/14_dynamic_depth5_blend035_ckpt"
) 2>&1 | tee "$LOG_DIR/20_depth5_eval.log"

touch "$RUN_DIR/DONE"
echo "[done] dataset=$DATASET_NAME $(date '+%F %T')"
EOF
chmod +x "$DATASET_RUNNER"

PROMPT_RUNNER="$OUT_ROOT/run_prompt_screening.sh"
cat > "$PROMPT_RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

echo "[start] prompt screening gpu=$GPU_PROMPT \$(date '+%F %T')"

(
  /usr/bin/time -v -o "$OUT_ROOT/bridge/time_prompt.txt" \\
    env CUDA_VISIBLE_DEVICES="$GPU_PROMPT" "$PYTHON_BIN" -u scripts/benchmark_prompt_channel_screening.py \\
      --dataset bridge \\
      --feature_manifest "$OUT_ROOT/bridge/00_split/eval_manifest.jsonl" \\
      --model_path "$MODEL_PATH" \\
      --output_dir "$OUT_ROOT/bridge/30_prompt_screening" \\
      --sample_count "$PROMPT_SAMPLE_COUNT" \\
      --sample_strategy random \\
      --seed "$SEED" \\
      --unnorm_key bridge_orig
) 2>&1 | tee "$OUT_ROOT/bridge/30_prompt_screening.log"

(
  /usr/bin/time -v -o "$OUT_ROOT/libero_long/time_prompt.txt" \\
    env CUDA_VISIBLE_DEVICES="$GPU_PROMPT" "$PYTHON_BIN" -u scripts/benchmark_prompt_channel_screening.py \\
      --dataset libero_long \\
      --feature_manifest "$OUT_ROOT/libero_long/00_split/eval_manifest.jsonl" \\
      --model_path "$MODEL_PATH" \\
      --output_dir "$OUT_ROOT/libero_long/30_prompt_screening" \\
      --sample_count "$PROMPT_SAMPLE_COUNT" \\
      --sample_strategy random \\
      --seed "$SEED" \\
      --unnorm_key bridge_orig
) 2>&1 | tee "$OUT_ROOT/libero_long/30_prompt_screening.log"

touch "$OUT_ROOT/PROMPT_DONE"
echo "[done] prompt screening \$(date '+%F %T')"
EOF
chmod +x "$PROMPT_RUNNER"

SUMMARY_RUNNER="$OUT_ROOT/run_summary_waiter.sh"
cat > "$SUMMARY_RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$OPENVLA_ROOT"
echo "[wait] summary watcher started \$(date '+%F %T')"
while [[ ! -f "$OUT_ROOT/bridge/DONE" || ! -f "$OUT_ROOT/libero_long/DONE" || ! -f "$OUT_ROOT/PROMPT_DONE" ]]; do
  sleep 60
done
mkdir -p "$OUT_ROOT/30_summary"
"$PYTHON_BIN" -u scripts/summarize_channel_screening.py \\
  --bridge_feature_table "$OPENVLA_ROOT/runs/feature_mask_ablation_20260428/bridge/summary_table.md" \\
  --libero_feature_table "$OPENVLA_ROOT/runs/feature_mask_ablation_20260428/libero_long/summary_table.md" \\
  --bridge_recipe_summary "$OPENVLA_ROOT/runs/recipe_training_ablation_20260428/bridge/20_recipe_eval/summary.json" \\
  --libero_recipe_summary "$OPENVLA_ROOT/runs/recipe_training_ablation_20260428/libero_long/20_recipe_eval/summary.json" \\
  --bridge_prompt_summary "$OUT_ROOT/bridge/30_prompt_screening/summary.json" \\
  --libero_prompt_summary "$OUT_ROOT/libero_long/30_prompt_screening/summary.json" \\
  --bridge_depth_summary "$OUT_ROOT/bridge/20_depth5_eval/summary.json" \\
  --libero_depth_summary "$OUT_ROOT/libero_long/20_depth5_eval/summary.json" \\
  --output_dir "$OUT_ROOT/30_summary" \\
  2>&1 | tee "$OUT_ROOT/30_summary/run.log"
echo "[done] summary watcher finished \$(date '+%F %T')"
EOF
chmod +x "$SUMMARY_RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n bridge \
  "$DATASET_RUNNER bridge $GPU_BRIDGE $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $SEED"
tmux new-window -t "$SESSION" -n libero \
  "$DATASET_RUNNER libero_long $GPU_LIBERO $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $SEED"
tmux new-window -t "$SESSION" -n prompt "$PROMPT_RUNNER"
tmux new-window -t "$SESSION" -n summary "$SUMMARY_RUNNER"

echo "[ok] started tmux session=$SESSION"
echo "[ok] bridge_gpu=$GPU_BRIDGE libero_gpu=$GPU_LIBERO prompt_gpu=$GPU_PROMPT"
echo "[ok] out_root=$OUT_ROOT"
echo "[hint] tmux attach -t $SESSION"
