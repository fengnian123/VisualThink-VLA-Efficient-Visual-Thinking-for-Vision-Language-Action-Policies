#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
SESSION="${SESSION:-recipe_training_ablation_20260428}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/recipe_training_ablation_20260428}"
BRIDGE_GPU="${BRIDGE_GPU:-0}"
LIBERO_GPU="${LIBERO_GPU:-1}"
TRAIN_COUNT="${TRAIN_COUNT:-4096}"
EVAL_COUNT="${EVAL_COUNT:-2048}"
SEED="${SEED:-7}"

mkdir -p "$OUT_ROOT"

DATASET_RUNNER="$OUT_ROOT/run_recipe_dataset.sh"
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
SEED="$9"

cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

case "$DATASET_NAME" in
  bridge)
    FEATURE_MANIFEST="runs/bridge_full/03_features/feature_manifest.jsonl"
    FULL_CKPT="runs/bridge_full/13_openvla_soft_full_ckpt"
    UNNORM_KEY="bridge_orig"
    DATASET_LABEL="bridge"
    ;;
  libero_long)
    FEATURE_MANIFEST="runs/libero10_full_all/03_features/feature_manifest.jsonl"
    FULL_CKPT="runs/libero10_full_all/13_openvla_soft_full_ckpt"
    UNNORM_KEY="bridge_orig"
    DATASET_LABEL="libero_long"
    ;;
  *)
    echo "[error] unsupported dataset=$DATASET_NAME" >&2
    exit 1
    ;;
esac

RUN_DIR="$OUT_ROOT/$DATASET_NAME"
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

train_gate() {
  local name="$1"
  local config="$2"
  local out_dir="$RUN_DIR/10_gate_${name}_ckpt"
  (
    /usr/bin/time -v -o "$LOG_DIR/time_gate_${name}.txt" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_learned_gating.py \
        --feature_manifest "$TRAIN_MANIFEST" \
        --output_dir "$out_dir" \
        --config "$config" \
        --seed "$SEED" \
        --log_every 20
  ) 2>&1 | tee "$LOG_DIR/10_gate_${name}.log"
}

train_dynamic() {
  local name="$1"
  local config="$2"
  local gate_ckpt="$3"
  local gate_cfg="$4"
  local out_dir="$RUN_DIR/14_${name}_ckpt"
  (
    /usr/bin/time -v -o "$LOG_DIR/time_${name}.txt" \
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_openvla_soft_evidence.py \
        --feature_manifest "$TRAIN_MANIFEST" \
        --model_path "$MODEL_PATH" \
        --output_dir "$out_dir" \
        --config "$config" \
        --mode dynamic \
        --gate_checkpoint_dir "$gate_ckpt" \
        --gate_config "$gate_cfg" \
        --teacher_adapter_dir "$FULL_CKPT" \
        --unnorm_key "$UNNORM_KEY" \
        --seed "$SEED" \
        --log_every 20
  ) 2>&1 | tee "$LOG_DIR/14_${name}.log"
}

train_gate "v4" "$OPENVLA_ROOT/configs/gating_policy_v4_relation_recipe.yaml"
train_gate "v8" "$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_recipe.yaml"

train_dynamic \
  "dynamic_hard_gate_v8" \
  "$OPENVLA_ROOT/configs/openvla_soft_evidence_recipe_hard.yaml" \
  "$RUN_DIR/10_gate_v8_ckpt" \
  "$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_recipe.yaml"

train_dynamic \
  "dynamic_blend035_gate_v8" \
  "$OPENVLA_ROOT/configs/openvla_soft_evidence_recipe_blend035.yaml" \
  "$RUN_DIR/10_gate_v8_ckpt" \
  "$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_recipe.yaml"

train_dynamic \
  "dynamic_blend035_distill_gate_v4" \
  "$OPENVLA_ROOT/configs/openvla_soft_evidence_recipe_blend035_distill.yaml" \
  "$RUN_DIR/10_gate_v4_ckpt" \
  "$OPENVLA_ROOT/configs/gating_policy_v4_relation_recipe.yaml"

train_dynamic \
  "dynamic_blend035_distill_gate_v8" \
  "$OPENVLA_ROOT/configs/openvla_soft_evidence_recipe_blend035_distill.yaml" \
  "$RUN_DIR/10_gate_v8_ckpt" \
  "$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_recipe.yaml"

(
  /usr/bin/time -v -o "$LOG_DIR/time_recipe_eval.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_soft_checkpoints.py \
      --dataset "$DATASET_LABEL" \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$RUN_DIR/20_recipe_eval" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --variant "hard=$RUN_DIR/14_dynamic_hard_gate_v8_ckpt" \
      --variant "blend_0p35=$RUN_DIR/14_dynamic_blend035_gate_v8_ckpt" \
      --variant "blend_0p35_distill=$RUN_DIR/14_dynamic_blend035_distill_gate_v8_ckpt" \
      --variant "blend_0p35_distill_gate_v4=$RUN_DIR/14_dynamic_blend035_distill_gate_v4_ckpt" \
      --variant "blend_0p35_distill_gate_v8=$RUN_DIR/14_dynamic_blend035_distill_gate_v8_ckpt"
) 2>&1 | tee "$LOG_DIR/20_recipe_eval.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_alpha_masks.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/prepare_dynamic_recipe_masks.py \
      --feature_manifest "$EVAL_MANIFEST" \
      --dynamic_checkpoint_dir "$RUN_DIR/14_dynamic_blend035_distill_gate_v8_ckpt" \
      --output_dir "$RUN_DIR/21_alpha_masks" \
      --sample_count 0 \
      --sample_strategy stride \
      --alphas "0.20,0.35,0.50" \
      --seed "$SEED" \
      --device cuda
) 2>&1 | tee "$LOG_DIR/21_alpha_masks.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_alpha_eval.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_channel_mask_ablation.py \
      --dataset "$DATASET_LABEL" \
      --feature_manifest "$RUN_DIR/21_alpha_masks/sample_manifest.jsonl" \
      --model_path "$MODEL_PATH" \
      --dynamic_checkpoint_dir "$RUN_DIR/14_dynamic_blend035_distill_gate_v8_ckpt" \
      --output_dir "$RUN_DIR/22_alpha_eval" \
      --sample_count 0 \
      --unnorm_key "$UNNORM_KEY" \
      --mask_paths_json "$RUN_DIR/21_alpha_masks/mask_paths.json"
) 2>&1 | tee "$LOG_DIR/22_alpha_eval.log"

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
while [[ ! -f "$OUT_ROOT/bridge/DONE" || ! -f "$OUT_ROOT/libero_long/DONE" ]]; do
  sleep 60
done
echo "[stage] generating final recipe summary \$(date '+%F %T')"
"$PYTHON_BIN" -u scripts/summarize_dynamic_recipe_ablation.py \\
  --bridge_recipe_summary "$OUT_ROOT/bridge/20_recipe_eval/summary.json" \\
  --bridge_alpha_summary "$OUT_ROOT/bridge/22_alpha_eval/summary.json" \\
  --libero_recipe_summary "$OUT_ROOT/libero_long/20_recipe_eval/summary.json" \\
  --libero_alpha_summary "$OUT_ROOT/libero_long/22_alpha_eval/summary.json" \\
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
  "$DATASET_RUNNER bridge $BRIDGE_GPU $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $SEED"
tmux new-window -t "$SESSION" -n libero \
  "$DATASET_RUNNER libero_long $LIBERO_GPU $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $TRAIN_COUNT $EVAL_COUNT $SEED"
tmux new-window -t "$SESSION" -n summary "$SUMMARY_RUNNER"

echo "[ok] started tmux session=$SESSION"
echo "[ok] bridge_gpu=$BRIDGE_GPU libero_gpu=$LIBERO_GPU"
echo "[ok] out_root=$OUT_ROOT"
echo "[hint] tmux attach -t $SESSION"
