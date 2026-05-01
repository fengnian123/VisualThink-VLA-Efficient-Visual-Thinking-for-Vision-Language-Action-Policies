#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ROOT:-$(pwd)}"
SESSION_NAME="${SESSION_NAME:-reasoning_compare_full_20260426}"
DEEPTHINK_GPU_ID="${DEEPTHINK_GPU_ID:-0}"
INTER_GPU_ID="${INTER_GPU_ID:-1}"
TRIALS="${TRIALS:-10}"
TIME_LIMIT="${TIME_LIMIT:-590m}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n deepthink_full \
  "cd $ROOT && timeout $TIME_LIMIT bash -lc 'GPU_ID=$DEEPTHINK_GPU_ID TRIALS=$TRIALS SUITES=\"$SUITES\" OUT_ROOT=$ROOT/runs/reasoning_vla_paper_metrics/deepthink_rl_full_trials${TRIALS} MODEL_NAME=DeepThinkVLA-RL bash $ROOT/commands/project/33_run_deepthinkvla_rl_libero_compare.sh; GPU_ID=$DEEPTHINK_GPU_ID TRIALS=$TRIALS SUITES=\"$SUITES\" OUT_ROOT=$ROOT/runs/reasoning_vla_paper_metrics/deepthink_sft_full_trials${TRIALS} CKPT=$ROOT/models/local/DeepThinkVLA/yinchenghust/deepthinkvla_libero_cot_sft MODEL_NAME=DeepThinkVLA-SFT bash $ROOT/commands/project/33_run_deepthinkvla_rl_libero_compare.sh'; status=\$?; echo; echo \"[deepthink_full_exit] status=\$status\"; exec bash"

tmux new-window -t "$SESSION_NAME":1 -n internvla_full \
  "cd $ROOT && timeout $TIME_LIMIT bash -lc 'GPU_ID=$INTER_GPU_ID TRIALS=$TRIALS SUITES=\"$SUITES\" OUT_ROOT=$ROOT/runs/reasoning_vla_paper_metrics/internvla_m1_full_trials${TRIALS} bash $ROOT/commands/project/34_run_internvla_m1_libero_compare.sh'; status=\$?; echo; echo \"[internvla_full_exit] status=\$status\"; exec bash"

echo "[ok] tmux session created: $SESSION_NAME"
echo "[ok] attach with: tmux attach-session -t $SESSION_NAME"
echo "[ok] deepthink output roots:"
echo "  $ROOT/runs/reasoning_vla_paper_metrics/deepthink_rl_full_trials${TRIALS}"
echo "  $ROOT/runs/reasoning_vla_paper_metrics/deepthink_sft_full_trials${TRIALS}"
echo "[ok] internvla output root:"
echo "  $ROOT/runs/reasoning_vla_paper_metrics/internvla_m1_full_trials${TRIALS}"
