#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-$OPENVLA_ROOT/models/local/openvla-7b}"
SESSION="${SESSION:-trace_ablation_20260429}"
OUT_ROOT="${OUT_ROOT:-$OPENVLA_ROOT/runs/trace_ablation_20260429}"
GPU_FULL="${GPU_FULL:-0}"
GPU_NO_UTILITY="${GPU_NO_UTILITY:-1}"
GPU_NO_ROUTE="${GPU_NO_ROUTE:-2}"
TRAIN_COUNT="${TRAIN_COUNT:-4096}"
EVAL_COUNT="${EVAL_COUNT:-2048}"
FAITHFULNESS_COUNT="${FAITHFULNESS_COUNT:-768}"
SEED="${SEED:-7}"

BRIDGE_FEATURE_MANIFEST="${BRIDGE_FEATURE_MANIFEST:-$OPENVLA_ROOT/runs/bridge_full/03_features/feature_manifest.jsonl}"
BRIDGE_TRACE_MANIFEST="${BRIDGE_TRACE_MANIFEST:-$OPENVLA_ROOT/runs/EvidenceTrace-VLA/bridge_full/evidence_trace.jsonl}"
FULL_CKPT="${FULL_CKPT:-$OPENVLA_ROOT/runs/bridge_full/13_openvla_soft_full_ckpt}"
NO_RANK_GATE_CKPT="${NO_RANK_GATE_CKPT:-$OPENVLA_ROOT/runs/recipe_training_ablation_20260428/bridge/10_gate_v8_ckpt}"
NO_RANK_STAGE1_CKPT="${NO_RANK_STAGE1_CKPT:-$OPENVLA_ROOT/runs/recipe_training_ablation_20260428/bridge/14_dynamic_blend035_distill_gate_v8_ckpt}"

RANK_GATE_CONFIG="${RANK_GATE_CONFIG:-$OPENVLA_ROOT/configs/gating_policy_v18_sequence_text_trace_rank_recipe.yaml}"
NO_RANK_GATE_CONFIG="${NO_RANK_GATE_CONFIG:-$OPENVLA_ROOT/configs/gating_policy_v8_sequence_text_recipe.yaml}"
STAGE1_CONFIG="${STAGE1_CONFIG:-$OPENVLA_ROOT/configs/openvla_soft_evidence_recipe_blend035_distill.yaml}"
STAGE2_FULL_CONFIG="${STAGE2_FULL_CONFIG:-$OPENVLA_ROOT/configs/openvla_soft_evidence_v6_dynamic_blend035_distill_traceaux_ablation.yaml}"
STAGE2_NO_ROUTE_CONFIG="${STAGE2_NO_ROUTE_CONFIG:-$OPENVLA_ROOT/configs/openvla_soft_evidence_v6_dynamic_blend035_distill_no_traceaux_ablation.yaml}"
UNNORM_KEY="${UNNORM_KEY:-bridge_orig}"

mkdir -p "$OUT_ROOT"

"$PYTHON_BIN" -u "$OPENVLA_ROOT/scripts/prepare_trace_ablation_split.py" \
  --feature_manifest "$BRIDGE_FEATURE_MANIFEST" \
  --trace_manifest "$BRIDGE_TRACE_MANIFEST" \
  --output_dir "$OUT_ROOT/00_split" \
  --train_count "$TRAIN_COUNT" \
  --eval_count "$EVAL_COUNT" \
  --faithfulness_count "$FAITHFULNESS_COUNT" \
  --sample_strategy random \
  --seed "$SEED"

FULL_RUNNER="$OUT_ROOT/run_ranked_full_pipeline.sh"
cat > "$FULL_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
OPENVLA_ROOT="$2"
PYTHON_BIN="$3"
MODEL_PATH="$4"
OUT_ROOT="$5"
RANK_GATE_CONFIG="$6"
STAGE1_CONFIG="$7"
STAGE2_FULL_CONFIG="$8"
FULL_CKPT="$9"
UNNORM_KEY="${10}"
SEED="${11}"

cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

SPLIT_DIR="$OUT_ROOT/00_split"
TRAIN_MANIFEST="$SPLIT_DIR/train_manifest.jsonl"
EVAL_MANIFEST="$SPLIT_DIR/eval_manifest.jsonl"
TRAIN_TRACE="$SPLIT_DIR/train_trace.jsonl"
FAITH_MANIFEST="$SPLIT_DIR/faithfulness_manifest.jsonl"
FAITH_TRACE="$SPLIT_DIR/faithfulness_trace.jsonl"
FAITH_TRACE_FREEFORM="$SPLIT_DIR/faithfulness_trace_freeform.jsonl"
LOG_DIR="$OUT_ROOT/logs_full"
mkdir -p "$LOG_DIR"

echo "[start] ranked full pipeline gpu=$GPU_ID $(date '+%F %T')"

(
  /usr/bin/time -v -o "$LOG_DIR/time_gate_ranked.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_learned_gating.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --output_dir "$OUT_ROOT/10_gate_ranked_ckpt" \
      --config "$RANK_GATE_CONFIG" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/10_gate_ranked.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_stage1_ranked.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_openvla_soft_evidence.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/14_dynamic_ranked_stage1_ckpt" \
      --config "$STAGE1_CONFIG" \
      --mode dynamic \
      --gate_checkpoint_dir "$OUT_ROOT/10_gate_ranked_ckpt" \
      --gate_config "$RANK_GATE_CONFIG" \
      --teacher_adapter_dir "$FULL_CKPT" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/14_dynamic_ranked_stage1.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_no_trace_benchmark.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_openvla_soft_three_way.py \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --full_checkpoint_dir "$FULL_CKPT" \
      --dynamic_checkpoint_dir "$OUT_ROOT/14_dynamic_ranked_stage1_ckpt" \
      --output_dir "$OUT_ROOT/15_no_trace_supervision_benchmark" \
      --limit 0 \
      --unnorm_key "$UNNORM_KEY" \
      --disturb_ratio 0.0 \
      --disturb_scale 22.0 \
      --skip_empty_instruction
) 2>&1 | tee "$LOG_DIR/15_no_trace_benchmark.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_no_trace_faithfulness.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_evidence_trace_faithfulness.py \
      --feature_manifest "$FAITH_MANIFEST" \
      --dynamic_checkpoint_dir "$OUT_ROOT/14_dynamic_ranked_stage1_ckpt" \
      --evidence_trace_manifest "$FAITH_TRACE" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/16_no_trace_supervision_faithfulness" \
      --unnorm_key "$UNNORM_KEY" \
      --limit 0 \
      --success_l1_thresh 0.08 \
      --mask_threshold 0.5 \
      --shuffle_channels relation,motion
) 2>&1 | tee "$LOG_DIR/16_no_trace_faithfulness.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_stage2_full.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_openvla_soft_evidence.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/23_full_stage2_ckpt" \
      --config "$STAGE2_FULL_CONFIG" \
      --mode dynamic \
      --gate_checkpoint_dir "$OUT_ROOT/10_gate_ranked_ckpt" \
      --gate_config "$RANK_GATE_CONFIG" \
      --teacher_adapter_dir "$FULL_CKPT" \
      --init_adapter_dir "$OUT_ROOT/14_dynamic_ranked_stage1_ckpt" \
      --evidence_trace_manifest "$TRAIN_TRACE" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/23_full_stage2.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_full_benchmark.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_openvla_soft_three_way.py \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --full_checkpoint_dir "$FULL_CKPT" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_full_stage2_ckpt" \
      --output_dir "$OUT_ROOT/24_full_benchmark" \
      --limit 0 \
      --unnorm_key "$UNNORM_KEY" \
      --disturb_ratio 0.0 \
      --disturb_scale 22.0 \
      --skip_empty_instruction
) 2>&1 | tee "$LOG_DIR/24_full_benchmark.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_full_faithfulness.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_evidence_trace_faithfulness.py \
      --feature_manifest "$FAITH_MANIFEST" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_full_stage2_ckpt" \
      --evidence_trace_manifest "$FAITH_TRACE" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/25_full_faithfulness" \
      --unnorm_key "$UNNORM_KEY" \
      --limit 0 \
      --success_l1_thresh 0.08 \
      --mask_threshold 0.5 \
      --shuffle_channels relation,motion
) 2>&1 | tee "$LOG_DIR/25_full_faithfulness.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_freeform_faithfulness.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_evidence_trace_faithfulness.py \
      --feature_manifest "$FAITH_MANIFEST" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_full_stage2_ckpt" \
      --evidence_trace_manifest "$FAITH_TRACE_FREEFORM" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/26_freeform_faithfulness" \
      --unnorm_key "$UNNORM_KEY" \
      --limit 0 \
      --success_l1_thresh 0.08 \
      --mask_threshold 0.5 \
      --shuffle_channels relation,motion
) 2>&1 | tee "$LOG_DIR/26_freeform_faithfulness.log"

touch "$OUT_ROOT/DONE_RANKED"
echo "[done] ranked full pipeline $(date '+%F %T')"
EOF
chmod +x "$FULL_RUNNER"

NO_ROUTE_RUNNER="$OUT_ROOT/run_no_route_pipeline.sh"
cat > "$NO_ROUTE_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
OPENVLA_ROOT="$2"
PYTHON_BIN="$3"
MODEL_PATH="$4"
OUT_ROOT="$5"
RANK_GATE_CONFIG="$6"
STAGE2_NO_ROUTE_CONFIG="$7"
FULL_CKPT="$8"
UNNORM_KEY="$9"
SEED="${10}"

cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

SPLIT_DIR="$OUT_ROOT/00_split"
TRAIN_MANIFEST="$SPLIT_DIR/train_manifest.jsonl"
EVAL_MANIFEST="$SPLIT_DIR/eval_manifest.jsonl"
TRAIN_TRACE="$SPLIT_DIR/train_trace.jsonl"
FAITH_MANIFEST="$SPLIT_DIR/faithfulness_manifest.jsonl"
FAITH_TRACE="$SPLIT_DIR/faithfulness_trace.jsonl"
LOG_DIR="$OUT_ROOT/logs_no_route"
mkdir -p "$LOG_DIR"

echo "[wait] no-route pipeline waiting for ranked stage1 ckpt $(date '+%F %T')"
while [[ ! -f "$OUT_ROOT/14_dynamic_ranked_stage1_ckpt/adapter.pt" ]]; do
  sleep 60
done

(
  /usr/bin/time -v -o "$LOG_DIR/time_stage2_no_route.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_openvla_soft_evidence.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/23_no_route_stage2_ckpt" \
      --config "$STAGE2_NO_ROUTE_CONFIG" \
      --mode dynamic \
      --gate_checkpoint_dir "$OUT_ROOT/10_gate_ranked_ckpt" \
      --gate_config "$RANK_GATE_CONFIG" \
      --teacher_adapter_dir "$FULL_CKPT" \
      --init_adapter_dir "$OUT_ROOT/14_dynamic_ranked_stage1_ckpt" \
      --evidence_trace_manifest "$TRAIN_TRACE" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/23_no_route_stage2.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_no_route_benchmark.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_openvla_soft_three_way.py \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --full_checkpoint_dir "$FULL_CKPT" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_no_route_stage2_ckpt" \
      --output_dir "$OUT_ROOT/24_no_route_benchmark" \
      --limit 0 \
      --unnorm_key "$UNNORM_KEY" \
      --disturb_ratio 0.0 \
      --disturb_scale 22.0 \
      --skip_empty_instruction
) 2>&1 | tee "$LOG_DIR/24_no_route_benchmark.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_no_route_faithfulness.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_evidence_trace_faithfulness.py \
      --feature_manifest "$FAITH_MANIFEST" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_no_route_stage2_ckpt" \
      --evidence_trace_manifest "$FAITH_TRACE" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/25_no_route_faithfulness" \
      --unnorm_key "$UNNORM_KEY" \
      --limit 0 \
      --success_l1_thresh 0.08 \
      --mask_threshold 0.5 \
      --shuffle_channels relation,motion
) 2>&1 | tee "$LOG_DIR/25_no_route_faithfulness.log"

touch "$OUT_ROOT/DONE_NO_ROUTE"
echo "[done] no-route pipeline $(date '+%F %T')"
EOF
chmod +x "$NO_ROUTE_RUNNER"

NO_UTILITY_RUNNER="$OUT_ROOT/run_no_utility_pipeline.sh"
cat > "$NO_UTILITY_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
OPENVLA_ROOT="$2"
PYTHON_BIN="$3"
MODEL_PATH="$4"
OUT_ROOT="$5"
NO_RANK_GATE_CKPT="$6"
NO_RANK_GATE_CONFIG="$7"
NO_RANK_STAGE1_CKPT="$8"
STAGE2_FULL_CONFIG="$9"
FULL_CKPT="${10}"
UNNORM_KEY="${11}"
SEED="${12}"

cd "$OPENVLA_ROOT"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_MODE=offline

SPLIT_DIR="$OUT_ROOT/00_split"
TRAIN_MANIFEST="$SPLIT_DIR/train_manifest.jsonl"
EVAL_MANIFEST="$SPLIT_DIR/eval_manifest.jsonl"
TRAIN_TRACE="$SPLIT_DIR/train_trace.jsonl"
FAITH_MANIFEST="$SPLIT_DIR/faithfulness_manifest.jsonl"
FAITH_TRACE="$SPLIT_DIR/faithfulness_trace.jsonl"
LOG_DIR="$OUT_ROOT/logs_no_utility"
mkdir -p "$LOG_DIR"

echo "[start] no-utility pipeline gpu=$GPU_ID $(date '+%F %T')"

(
  /usr/bin/time -v -o "$LOG_DIR/time_stage2_no_utility.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/train_openvla_soft_evidence.py \
      --feature_manifest "$TRAIN_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/23_no_utility_rank_stage2_ckpt" \
      --config "$STAGE2_FULL_CONFIG" \
      --mode dynamic \
      --gate_checkpoint_dir "$NO_RANK_GATE_CKPT" \
      --gate_config "$NO_RANK_GATE_CONFIG" \
      --teacher_adapter_dir "$FULL_CKPT" \
      --init_adapter_dir "$NO_RANK_STAGE1_CKPT" \
      --evidence_trace_manifest "$TRAIN_TRACE" \
      --unnorm_key "$UNNORM_KEY" \
      --seed "$SEED" \
      --log_every 20
) 2>&1 | tee "$LOG_DIR/23_no_utility_rank_stage2.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_no_utility_benchmark.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_openvla_soft_three_way.py \
      --feature_manifest "$EVAL_MANIFEST" \
      --model_path "$MODEL_PATH" \
      --full_checkpoint_dir "$FULL_CKPT" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_no_utility_rank_stage2_ckpt" \
      --output_dir "$OUT_ROOT/24_no_utility_rank_benchmark" \
      --limit 0 \
      --unnorm_key "$UNNORM_KEY" \
      --disturb_ratio 0.0 \
      --disturb_scale 22.0 \
      --skip_empty_instruction
) 2>&1 | tee "$LOG_DIR/24_no_utility_rank_benchmark.log"

(
  /usr/bin/time -v -o "$LOG_DIR/time_no_utility_faithfulness.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -u scripts/benchmark_evidence_trace_faithfulness.py \
      --feature_manifest "$FAITH_MANIFEST" \
      --dynamic_checkpoint_dir "$OUT_ROOT/23_no_utility_rank_stage2_ckpt" \
      --evidence_trace_manifest "$FAITH_TRACE" \
      --model_path "$MODEL_PATH" \
      --output_dir "$OUT_ROOT/25_no_utility_rank_faithfulness" \
      --unnorm_key "$UNNORM_KEY" \
      --limit 0 \
      --success_l1_thresh 0.08 \
      --mask_threshold 0.5 \
      --shuffle_channels relation,motion
) 2>&1 | tee "$LOG_DIR/25_no_utility_rank_faithfulness.log"

touch "$OUT_ROOT/DONE_NO_UTILITY"
echo "[done] no-utility pipeline $(date '+%F %T')"
EOF
chmod +x "$NO_UTILITY_RUNNER"

SUMMARY_RUNNER="$OUT_ROOT/run_trace_summary_waiter.sh"
cat > "$SUMMARY_RUNNER" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="$1"
PYTHON_BIN="$2"
OUT_ROOT="$3"

cd "$OPENVLA_ROOT"
echo "[wait] trace summary watcher started $(date '+%F %T')"
while [[ ! -f "$OUT_ROOT/DONE_RANKED" || ! -f "$OUT_ROOT/DONE_NO_ROUTE" || ! -f "$OUT_ROOT/DONE_NO_UTILITY" ]]; do
  sleep 60
done

"$PYTHON_BIN" -u scripts/summarize_trace_interpretability_ablation.py \
  --full_benchmark "$OUT_ROOT/24_full_benchmark/summary_table.md" \
  --full_faithfulness "$OUT_ROOT/25_full_faithfulness/summary_table.md" \
  --no_route_benchmark "$OUT_ROOT/24_no_route_benchmark/summary_table.md" \
  --no_route_faithfulness "$OUT_ROOT/25_no_route_faithfulness/summary_table.md" \
  --no_trace_benchmark "$OUT_ROOT/15_no_trace_supervision_benchmark/summary_table.md" \
  --no_trace_faithfulness "$OUT_ROOT/16_no_trace_supervision_faithfulness/summary_table.md" \
  --no_utility_benchmark "$OUT_ROOT/24_no_utility_rank_benchmark/summary_table.md" \
  --no_utility_faithfulness "$OUT_ROOT/25_no_utility_rank_faithfulness/summary_table.md" \
  --freeform_benchmark "$OUT_ROOT/24_full_benchmark/summary_table.md" \
  --freeform_faithfulness "$OUT_ROOT/26_freeform_faithfulness/summary_table.md" \
  --output_dir "$OUT_ROOT/30_summary" \
  2>&1 | tee "$OUT_ROOT/30_summary/run.log"

echo "[done] trace summary watcher finished $(date '+%F %T')"
EOF
chmod +x "$SUMMARY_RUNNER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n ranked_full \
  "$FULL_RUNNER $GPU_FULL $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $RANK_GATE_CONFIG $STAGE1_CONFIG $STAGE2_FULL_CONFIG $FULL_CKPT $UNNORM_KEY $SEED"
tmux new-window -t "$SESSION" -n no_utility \
  "$NO_UTILITY_RUNNER $GPU_NO_UTILITY $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $NO_RANK_GATE_CKPT $NO_RANK_GATE_CONFIG $NO_RANK_STAGE1_CKPT $STAGE2_FULL_CONFIG $FULL_CKPT $UNNORM_KEY $SEED"
tmux new-window -t "$SESSION" -n no_route \
  "$NO_ROUTE_RUNNER $GPU_NO_ROUTE $OPENVLA_ROOT $PYTHON_BIN $MODEL_PATH $OUT_ROOT $RANK_GATE_CONFIG $STAGE2_NO_ROUTE_CONFIG $FULL_CKPT $UNNORM_KEY $SEED"
tmux new-window -t "$SESSION" -n summary \
  "$SUMMARY_RUNNER $OPENVLA_ROOT $PYTHON_BIN $OUT_ROOT"

echo "[ok] started tmux session=$SESSION"
echo "[ok] gpu_full=$GPU_FULL gpu_no_utility=$GPU_NO_UTILITY gpu_no_route=$GPU_NO_ROUTE"
echo "[ok] out_root=$OUT_ROOT"
echo "[hint] tmux attach -t $SESSION"
