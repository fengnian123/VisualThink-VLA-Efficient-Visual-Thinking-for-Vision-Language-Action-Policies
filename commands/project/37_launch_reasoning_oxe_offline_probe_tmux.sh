#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
SESSION="${SESSION:-reasoning_external_oxe_probe_20260427}"
GPU_ID="${GPU_ID:-1}"
LIMIT="${LIMIT:-16}"
TIME_LIMIT="${TIME_LIMIT:-10h}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
DATASETS="${DATASETS:-fractal utaustin_mutex bridge roboturk viola}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/reasoning_vla_paper_metrics/external_oxe_offline_probe_limit${LIMIT}_20260427}"

DEEPTHINK_ROOT="$ROOT/models/local/DeepThinkVLA"
INTERNVLA_ROOT="$ROOT/models/local/InternVLA-M1"

dataset_args() {
  local args=()
  for dataset in $DATASETS; do
    case "$dataset" in
      bridge)
        args+=(--dataset bridge "$ROOT/runs/bridge_full/03_features/feature_manifest.jsonl" "libero_franka_norm_cross_dataset")
        ;;
      fractal)
        args+=(--dataset fractal "$ROOT/runs/fractal_official_full/03_features/feature_manifest.jsonl" "libero_franka_norm_cross_dataset")
        ;;
      roboturk)
        args+=(--dataset roboturk "$ROOT/runs/roboturk_official_full/03_features/feature_manifest.jsonl" "libero_franka_norm_cross_dataset")
        ;;
      viola)
        args+=(--dataset viola "$ROOT/runs/viola_official_full/03_features/feature_manifest.jsonl" "libero_franka_norm_cross_dataset")
        ;;
      utaustin_mutex)
        args+=(--dataset utaustin_mutex "$ROOT/runs/utaustin_mutex/03_features/feature_manifest.jsonl" "libero_franka_norm_cross_dataset")
        ;;
      *)
        echo "[warn] unsupported dataset=$dataset" >&2
        ;;
    esac
  done
  printf '%q ' "${args[@]}"
}

mkdir -p "$OUT_ROOT/logs"
DATASET_ARGS="$(dataset_args)"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -n probe \
  "bash -lc '
    set -euo pipefail
    ROOT=$ROOT
    OUT_ROOT=$OUT_ROOT
    GPU_ID=$GPU_ID
    LIMIT=$LIMIT
    MAX_NEW_TOKENS=$MAX_NEW_TOKENS
    DATASET_ARGS=\"$DATASET_ARGS\"

    set +u
    source \"\$(conda info --base)/etc/profile.d/conda.sh\"
    set -u

    echo \"[wait] waiting for ECoT GPU1 job to finish before external probes\"
    while ps -eo comm=,args= | grep -F \"$ROOT/scripts/benchmark_ecot_oxe_offline.py\" | grep -E \"^[[:space:]]*python\" >/dev/null; do
      date +\"[wait] %F %T ECoT still running\"
      sleep 120
    done

    echo \"[start] DeepThinkVLA-RL OXE offline probe\"
    set +u
    conda activate deepthinkvla
    set -u
    cd $DEEPTHINK_ROOT
    export PYTHONPATH=\"$DEEPTHINK_ROOT/src:\${PYTHONPATH:-}\"
    timeout $TIME_LIMIT env CUDA_VISIBLE_DEVICES=$GPU_ID \
      python -u $ROOT/scripts/benchmark_deepthink_oxe_offline.py \
        --deepthink_root $DEEPTHINK_ROOT \
        --checkpoint $DEEPTHINK_ROOT/yinchenghust/deepthinkvla_libero_cot_rl \
        --output_dir $OUT_ROOT/deepthink_rl \
        --limit $LIMIT \
        --max_new_tokens $MAX_NEW_TOKENS \
        --skip_empty_instruction \
        \$DATASET_ARGS \
      2>&1 | tee $OUT_ROOT/logs/deepthink_rl.log
    echo \"[done] DeepThinkVLA-RL status=\${PIPESTATUS[0]}\"

    echo \"[start] DeepThinkVLA-SFT OXE offline probe\"
    timeout $TIME_LIMIT env CUDA_VISIBLE_DEVICES=$GPU_ID \
      python -u $ROOT/scripts/benchmark_deepthink_oxe_offline.py \
        --deepthink_root $DEEPTHINK_ROOT \
        --checkpoint $DEEPTHINK_ROOT/yinchenghust/deepthinkvla_libero_cot_sft \
        --output_dir $OUT_ROOT/deepthink_sft \
        --limit $LIMIT \
        --max_new_tokens $MAX_NEW_TOKENS \
        --skip_empty_instruction \
        \$DATASET_ARGS \
      2>&1 | tee $OUT_ROOT/logs/deepthink_sft.log
    echo \"[done] DeepThinkVLA-SFT status=\${PIPESTATUS[0]}\"

    echo \"[start] InternVLA-M1 OXE offline probe\"
    set +u
    conda activate internvla-m1
    set -u
    cd $INTERNVLA_ROOT
    export PYTHONPATH=\"$INTERNVLA_ROOT:\${PYTHONPATH:-}\"
    timeout $TIME_LIMIT env CUDA_VISIBLE_DEVICES=$GPU_ID \
      python -u $ROOT/scripts/benchmark_internvla_m1_oxe_offline.py \
        --internvla_root $INTERNVLA_ROOT \
        --checkpoint $INTERNVLA_ROOT/checkpoints/InternVLA-M1-LIBERO-Long/checkpoints/steps_30000_pytorch_model.pt \
        --output_dir $OUT_ROOT/internvla_m1_libero_long \
        --unnorm_key franka \
        --limit $LIMIT \
        --skip_empty_instruction \
        \$DATASET_ARGS \
      2>&1 | tee $OUT_ROOT/logs/internvla_m1_libero_long.log
    echo \"[done] InternVLA-M1 status=\${PIPESTATUS[0]}\"

    echo \"[all-done] external OXE offline probe output=$OUT_ROOT\"
    exec bash
  '"

echo "[ok] launched tmux session=$SESSION output=$OUT_ROOT"
