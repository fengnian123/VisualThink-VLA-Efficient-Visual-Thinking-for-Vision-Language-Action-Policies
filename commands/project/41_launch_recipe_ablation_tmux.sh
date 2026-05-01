#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-4000}"
SESSION="${SESSION:-recipe_ablation_20260428}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/recipe_ablation_20260428}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"
ALPHAS="${ALPHAS:-0.20,0.35,0.50}"

mkdir -p "$OUT_ROOT"

RUNNER="$OUT_ROOT/run_recipe_ablation.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

echo "[start] \$(date '+%F %T')"
echo "[config] gpu=$GPU_ID sample_count=$SAMPLE_COUNT alphas=$ALPHAS out_root=$OUT_ROOT"

for DATASET_NAME in bridge libero_long; do
  if [[ "\$DATASET_NAME" == "bridge" ]]; then
    FEATURE_MANIFEST="runs/bridge_full/03_features/feature_manifest.jsonl"
    DYNAMIC_CKPT="runs/bridge_full/14_openvla_soft_dynamic_ckpt"
  else
    FEATURE_MANIFEST="runs/libero10_full_all/03_features/feature_manifest.jsonl"
    DYNAMIC_CKPT="runs/libero10_full_all/14_openvla_soft_dynamic_ckpt"
  fi

  MASK_DIR="$OUT_ROOT/\${DATASET_NAME}_masks"
  EVAL_DIR="$OUT_ROOT/\${DATASET_NAME}"
  mkdir -p "\$MASK_DIR" "\$EVAL_DIR"

  echo "[stage] prepare masks dataset=\$DATASET_NAME"
  (
    /usr/bin/time -v -o "\$MASK_DIR/time_prepare_masks.txt" \\
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/prepare_dynamic_recipe_masks.py \\
        --feature_manifest "\$FEATURE_MANIFEST" \\
        --dynamic_checkpoint_dir "\$DYNAMIC_CKPT" \\
        --output_dir "\$MASK_DIR" \\
        --sample_count "$SAMPLE_COUNT" \\
        --sample_strategy stride \\
        --alphas "$ALPHAS" \\
        --device cuda
  ) 2>&1 | tee "\$MASK_DIR/prepare_masks.log"

  echo "[stage] evaluate recipe masks dataset=\$DATASET_NAME"
  (
    /usr/bin/time -v -o "\$EVAL_DIR/time_eval.txt" \\
      env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_channel_mask_ablation.py \\
        --dataset "\$DATASET_NAME" \\
        --feature_manifest "\$MASK_DIR/sample_manifest.jsonl" \\
        --model_path "$MODEL_PATH" \\
        --dynamic_checkpoint_dir "\$DYNAMIC_CKPT" \\
        --output_dir "\$EVAL_DIR" \\
        --sample_count 0 \\
        --unnorm_key "$UNNORM_KEY" \\
        --mask_paths_json "\$MASK_DIR/mask_paths.json"
  ) 2>&1 | tee "\$EVAL_DIR/run.log"
done

echo "[done] \$(date '+%F %T')"
echo "[ok] bridge=$OUT_ROOT/bridge/summary_table.md"
echo "[ok] libero_long=$OUT_ROOT/libero_long/summary_table.md"
EOF

chmod +x "$RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" "$RUNNER"
echo "[ok] started tmux session=$SESSION"
echo "[ok] runner=$RUNNER"
echo "[ok] out_root=$OUT_ROOT"
echo "[hint] tmux attach -t $SESSION"
