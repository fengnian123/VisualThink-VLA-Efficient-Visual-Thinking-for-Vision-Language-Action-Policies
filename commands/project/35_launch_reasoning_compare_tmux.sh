#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ROOT:-$(pwd)}"
SESSION_NAME="${SESSION_NAME:-reasoning_compare}"
DEEPTHINK_GPU_ID="${DEEPTHINK_GPU_ID:-0}"
INTER_GPU_ID="${INTER_GPU_ID:-1}"
TRIALS="${TRIALS:-1}"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" -n deepthink_rl \
  "cd $ROOT && GPU_ID=$DEEPTHINK_GPU_ID TRIALS=$TRIALS bash $ROOT/commands/project/33_run_deepthinkvla_rl_libero_compare.sh; echo; echo '[deepthink_rl_exit]'; exec bash"

tmux new-window -t "$SESSION_NAME":1 -n internvla_m1 \
  "cd $ROOT && GPU_ID=$INTER_GPU_ID TRIALS=$TRIALS bash $ROOT/commands/project/34_run_internvla_m1_libero_compare.sh; echo; echo '[internvla_m1_exit]'; exec bash"

echo "[ok] tmux session created: $SESSION_NAME"
echo "[ok] attach with: tmux attach-session -t $SESSION_NAME"
