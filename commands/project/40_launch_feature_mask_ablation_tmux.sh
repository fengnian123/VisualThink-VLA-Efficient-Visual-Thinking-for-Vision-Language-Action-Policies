#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-4000}"
SESSION="${SESSION:-feature_mask_ablation_20260428}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/feature_mask_ablation_20260428}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"

mkdir -p "$OUT_ROOT"

RUNNER="$OUT_ROOT/run_feature_mask_ablation.sh"
cat > "$RUNNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

echo "[start] \$(date '+%F %T')"
echo "[config] gpu=$GPU_ID sample_count=$SAMPLE_COUNT out_root=$OUT_ROOT"

mkdir -p "$OUT_ROOT/bridge" "$OUT_ROOT/libero_long"

echo "[stage] bridge controlled feature-mask ablation"
(
  /usr/bin/time -v -o "$OUT_ROOT/bridge/time.txt" \\
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_channel_mask_ablation.py \\
      --dataset bridge \\
      --feature_manifest runs/bridge_full/03_features/feature_manifest.jsonl \\
      --model_path "$MODEL_PATH" \\
      --dynamic_checkpoint_dir runs/bridge_full/14_openvla_soft_dynamic_ckpt \\
      --output_dir "$OUT_ROOT/bridge" \\
      --sample_count "$SAMPLE_COUNT" \\
      --sample_strategy stride \\
      --unnorm_key "$UNNORM_KEY"
) 2>&1 | tee "$OUT_ROOT/bridge/run.log"

echo "[stage] libero_long controlled feature-mask ablation"
(
  /usr/bin/time -v -o "$OUT_ROOT/libero_long/time.txt" \\
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_dynamic_channel_mask_ablation.py \\
      --dataset libero_long \\
      --feature_manifest runs/libero10_full_all/03_features/feature_manifest.jsonl \\
      --model_path "$MODEL_PATH" \\
      --dynamic_checkpoint_dir runs/libero10_full_all/14_openvla_soft_dynamic_ckpt \\
      --output_dir "$OUT_ROOT/libero_long" \\
      --sample_count "$SAMPLE_COUNT" \\
      --sample_strategy stride \\
      --unnorm_key "$UNNORM_KEY"
) 2>&1 | tee "$OUT_ROOT/libero_long/run.log"

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
