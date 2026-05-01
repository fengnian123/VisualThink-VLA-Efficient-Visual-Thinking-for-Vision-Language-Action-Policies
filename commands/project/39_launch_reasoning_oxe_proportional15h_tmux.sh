#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
SESSION="${SESSION:-reasoning_external_oxe_prop15h_20260427}"
GPU_ID="${GPU_ID:-1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/reasoning_vla_paper_metrics/external_oxe_proportional15h_20260427}"
TIME_LIMIT_PER_MODEL="${TIME_LIMIT_PER_MODEL:-5h}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

BRIDGE_N="${BRIDGE_N:-1396}"
FRACTAL_N="${FRACTAL_N:-594}"
ROBOTURK_N="${ROBOTURK_N:-47}"
VIOLA_N="${VIOLA_N:-64}"
UTAUSTIN_MUTEX_N="${UTAUSTIN_MUTEX_N:-2399}"

DEEPTHINK_ROOT="$ROOT/models/local/DeepThinkVLA"
INTERNVLA_ROOT="$ROOT/models/local/InternVLA-M1"
SAMPLE_ROOT="$OUT_ROOT/sample_manifests"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "[error] tmux session already exists: $SESSION" >&2
  exit 1
fi

mkdir -p "$SAMPLE_ROOT" "$OUT_ROOT/logs"

sample_manifest() {
  local input="$1"
  local n="$2"
  local output="$3"
  local total
  total="$(wc -l < "$input")"
  if [[ "$n" -ge "$total" ]]; then
    cp "$input" "$output"
  else
    shuf -n "$n" "$input" > "$output"
  fi
}

sample_manifest "$ROOT/runs/bridge_full/03_features/feature_manifest.jsonl" "$BRIDGE_N" "$SAMPLE_ROOT/bridge.jsonl"
sample_manifest "$ROOT/runs/fractal_official_full/03_features/feature_manifest.jsonl" "$FRACTAL_N" "$SAMPLE_ROOT/fractal.jsonl"
sample_manifest "$ROOT/runs/roboturk_official_full/03_features/feature_manifest.jsonl" "$ROBOTURK_N" "$SAMPLE_ROOT/roboturk.jsonl"
sample_manifest "$ROOT/runs/viola_official_full/03_features/feature_manifest.jsonl" "$VIOLA_N" "$SAMPLE_ROOT/viola.jsonl"
sample_manifest "$ROOT/runs/utaustin_mutex/03_features/feature_manifest.jsonl" "$UTAUSTIN_MUTEX_N" "$SAMPLE_ROOT/utaustin_mutex.jsonl"

{
  echo "# Proportional OXE sample plan"
  echo
  echo "| Dataset | Sample rows |"
  echo "|---|---:|"
  for dataset in bridge fractal roboturk viola utaustin_mutex; do
    printf '| %s | %s |\n' "$dataset" "$(wc -l < "$SAMPLE_ROOT/$dataset.jsonl")"
  done
  echo
  echo "- Per-model timeout: $TIME_LIMIT_PER_MODEL"
  echo "- GPU: $GPU_ID"
  echo "- Note: released DeepThinkVLA/InternVLA checkpoints use LIBERO/Franka normalization; these are cross-dataset offline diagnostics."
} > "$OUT_ROOT/sample_plan.md"

tmux new-session -d -s "$SESSION" -n prop15h \
  "bash -lc '
    set -euo pipefail

    ROOT=\"$ROOT\"
    OUT_ROOT=\"$OUT_ROOT\"
    SAMPLE_ROOT=\"$SAMPLE_ROOT\"
    GPU_ID=\"$GPU_ID\"
    TIME_LIMIT_PER_MODEL=\"$TIME_LIMIT_PER_MODEL\"
    MAX_NEW_TOKENS=\"$MAX_NEW_TOKENS\"
    DEEPTHINK_ROOT=\"$DEEPTHINK_ROOT\"
    INTERNVLA_ROOT=\"$INTERNVLA_ROOT\"

    set +u
    source $HOME/miniconda3/etc/profile.d/conda.sh
    set -u

    DATASET_ARGS=(
      --dataset bridge \"\$SAMPLE_ROOT/bridge.jsonl\" libero_franka_norm_cross_dataset
      --dataset fractal \"\$SAMPLE_ROOT/fractal.jsonl\" libero_franka_norm_cross_dataset
      --dataset roboturk \"\$SAMPLE_ROOT/roboturk.jsonl\" libero_franka_norm_cross_dataset
      --dataset viola \"\$SAMPLE_ROOT/viola.jsonl\" libero_franka_norm_cross_dataset
      --dataset utaustin_mutex \"\$SAMPLE_ROOT/utaustin_mutex.jsonl\" libero_franka_norm_cross_dataset
    )

    run_deepthink() {
      local name=\"\$1\"
      local ckpt=\"\$2\"
      local out_dir=\"\$3\"
      local log_file=\"\$4\"
      echo \"[start] \$name OXE proportional probe time=\$(date \"+%F %T\")\"
      set +u
      conda activate deepthinkvla
      set -u
      cd \"\$DEEPTHINK_ROOT\"
      export PYTHONPATH=\"\$DEEPTHINK_ROOT/src:\${PYTHONPATH:-}\"
      set +e
      timeout \"\$TIME_LIMIT_PER_MODEL\" env CUDA_VISIBLE_DEVICES=\"\$GPU_ID\" \
        python -u \"\$ROOT/scripts/benchmark_deepthink_oxe_offline.py\" \
          --deepthink_root \"\$DEEPTHINK_ROOT\" \
          --checkpoint \"\$ckpt\" \
          --output_dir \"\$out_dir\" \
          --limit 0 \
          --max_new_tokens \"\$MAX_NEW_TOKENS\" \
          --skip_empty_instruction \
          \"\${DATASET_ARGS[@]}\" \
        2>&1 | tee \"\$log_file\"
      local status=\${PIPESTATUS[0]}
      set -e
      echo \"[done] \$name status=\$status time=\$(date \"+%F %T\")\"
    }

    run_internvla() {
      echo \"[start] InternVLA-M1 OXE proportional probe time=\$(date \"+%F %T\")\"
      set +u
      conda activate internvla-m1
      set -u
      cd \"\$INTERNVLA_ROOT\"
      export PYTHONPATH=\"\$INTERNVLA_ROOT:\${PYTHONPATH:-}\"
      set +e
      timeout \"\$TIME_LIMIT_PER_MODEL\" env CUDA_VISIBLE_DEVICES=\"\$GPU_ID\" \
        python -u \"\$ROOT/scripts/benchmark_internvla_m1_oxe_offline.py\" \
          --internvla_root \"\$INTERNVLA_ROOT\" \
          --checkpoint \"\$INTERNVLA_ROOT/checkpoints/InternVLA-M1-LIBERO-Long/checkpoints/steps_30000_pytorch_model.pt\" \
          --output_dir \"\$OUT_ROOT/internvla_m1_libero_long\" \
          --unnorm_key franka \
          --limit 0 \
          --skip_empty_instruction \
          \"\${DATASET_ARGS[@]}\" \
        2>&1 | tee \"\$OUT_ROOT/logs/internvla_m1_libero_long.log\"
      local status=\${PIPESTATUS[0]}
      set -e
      echo \"[done] InternVLA-M1 status=\$status time=\$(date \"+%F %T\")\"
    }

    cat \"\$OUT_ROOT/sample_plan.md\"
    run_deepthink \"DeepThinkVLA-RL\" \"\$DEEPTHINK_ROOT/yinchenghust/deepthinkvla_libero_cot_rl\" \"\$OUT_ROOT/deepthink_rl\" \"\$OUT_ROOT/logs/deepthink_rl.log\"
    run_deepthink \"DeepThinkVLA-SFT\" \"\$DEEPTHINK_ROOT/yinchenghust/deepthinkvla_libero_cot_sft\" \"\$OUT_ROOT/deepthink_sft\" \"\$OUT_ROOT/logs/deepthink_sft.log\"
    run_internvla
    echo \"[all-done] output=\$OUT_ROOT time=\$(date \"+%F %T\")\"
    exec bash
  '"

echo "[ok] launched session=$SESSION output=$OUT_ROOT"
cat "$OUT_ROOT/sample_plan.md"
