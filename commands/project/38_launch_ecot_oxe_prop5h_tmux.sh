#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
SESSION="${SESSION:-ecot_oxe_prop5h_20260427}"
GPU_ID="${GPU_ID:-1}"
OUT="${OUT:-$ROOT/runs/reasoning_vla_paper_metrics/ecot_oxe_prop5h_20260427}"
TIME_LIMIT="${TIME_LIMIT:-6h}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$OUT"

tmux new-session -d -s "$SESSION" -n ecot_prop5h \
  "bash -lc '
    set -euo pipefail
    set +u
    source $HOME/miniconda3/etc/profile.d/conda.sh
    conda activate fast-ecot
    set -u
    cd \"$ROOT\"
    echo \"[info] start=\$(date \"+%F %T\") output=$OUT gpu=$GPU_ID timeout=$TIME_LIMIT\"
    timeout $TIME_LIMIT env CUDA_VISIBLE_DEVICES=$GPU_ID \
      python -u \"$ROOT/scripts/benchmark_ecot_oxe_offline.py\" \
        --model_path \"$ROOT/models/local/Embodied-CoT/ecot-openvla-7b-oxe\" \
        --fast_ecot_root \"$ROOT/models/local/Fast-ECoT\" \
        --output_dir \"$OUT\" \
        --limit 0 \
        --max_new_tokens $MAX_NEW_TOKENS \
        --success_l1_thresh 0.08 \
        --attn_impl sdpa \
        --skip_empty_instruction \
        --dataset bridge \"$OUT/sample_manifests/bridge.jsonl\" bridge_reasoning \
        --dataset fractal \"$OUT/sample_manifests/fractal.jsonl\" fractal20220817_data \
        --dataset roboturk \"$OUT/sample_manifests/roboturk.jsonl\" roboturk \
        --dataset viola \"$OUT/sample_manifests/viola.jsonl\" viola \
        --dataset utaustin_mutex \"$OUT/sample_manifests/utaustin_mutex.jsonl\" utaustin_mutex \
      2>&1 | tee \"$OUT/benchmark.log\"
    status=\${PIPESTATUS[0]}
    echo \"[ecot_prop5h_exit] status=\$status end=\$(date \"+%F %T\")\"
    exec bash
  '"

echo "[ok] launched session=$SESSION output=$OUT"
