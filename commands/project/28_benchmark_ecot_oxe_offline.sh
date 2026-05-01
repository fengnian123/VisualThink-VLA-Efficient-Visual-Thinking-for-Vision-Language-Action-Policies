#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
FAST_ECOT_ROOT="${FAST_ECOT_ROOT:-$OPENVLA_ROOT/models/local/Fast-ECoT}"
ECOT_MODEL_PATH="${ECOT_MODEL_PATH:-$OPENVLA_ROOT/models/local/Embodied-CoT/ecot-openvla-7b-oxe}"
ECOT_OUTPUT_DIR="${ECOT_OUTPUT_DIR:-$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/ecot_oxe_offline}"
ECOT_LIMIT="${ECOT_LIMIT:-1}"
ECOT_MAX_NEW_TOKENS="${ECOT_MAX_NEW_TOKENS:-256}"
ECOT_SUCCESS_L1_THRESH="${ECOT_SUCCESS_L1_THRESH:-0.08}"
ECOT_ATTN_IMPL="${ECOT_ATTN_IMPL:-sdpa}"
GPU_IDS="${GPU_IDS:-0}"

set +u
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate fast-ecot
set -u

DATASETS=(${ECOT_DATASETS:-bridge fractal roboturk viola utaustin_mutex})
DATASET_ARGS=()

add_dataset() {
  local name="$1"
  local manifest="$2"
  local unnorm_key="$3"
  if [[ -f "$manifest" ]]; then
    DATASET_ARGS+=(--dataset "$name" "$manifest" "$unnorm_key")
  else
    echo "[skip] missing feature manifest for $name: $manifest" >&2
  fi
}

for dataset in "${DATASETS[@]}"; do
  case "$dataset" in
    bridge)
      add_dataset bridge "$OPENVLA_ROOT/runs/bridge_full/03_features/feature_manifest.jsonl" bridge_reasoning
      ;;
    fractal)
      if [[ -f "$OPENVLA_ROOT/runs/fractal_official_full/03_features/feature_manifest.jsonl" ]]; then
        add_dataset fractal "$OPENVLA_ROOT/runs/fractal_official_full/03_features/feature_manifest.jsonl" fractal20220817_data
      else
        add_dataset fractal "$OPENVLA_ROOT/runs/fractal_soft28/03_features/feature_manifest.jsonl" fractal20220817_data
      fi
      ;;
    roboturk)
      add_dataset roboturk "$OPENVLA_ROOT/runs/roboturk_official_full/03_features/feature_manifest.jsonl" roboturk
      ;;
    viola)
      add_dataset viola "$OPENVLA_ROOT/runs/viola_official_full/03_features/feature_manifest.jsonl" viola
      ;;
    utaustin_mutex)
      add_dataset utaustin_mutex "$OPENVLA_ROOT/runs/utaustin_mutex/03_features/feature_manifest.jsonl" utaustin_mutex
      ;;
    berkeley_autolab_ur5)
      add_dataset berkeley_autolab_ur5 "$OPENVLA_ROOT/runs/autolab_screen96/03_features/feature_manifest.jsonl" berkeley_autolab_ur5
      ;;
    *)
      echo "[warn] unsupported ECOT_DATASETS entry: $dataset" >&2
      ;;
  esac
done

if [[ "${#DATASET_ARGS[@]}" -eq 0 ]]; then
  echo "[error] no usable dataset manifests selected" >&2
  exit 1
fi

mkdir -p "$ECOT_OUTPUT_DIR"
echo "[info] model=$ECOT_MODEL_PATH"
echo "[info] output=$ECOT_OUTPUT_DIR"
echo "[info] limit=$ECOT_LIMIT max_new_tokens=$ECOT_MAX_NEW_TOKENS"

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
"${CONDA_PREFIX}/bin/python" -u "$OPENVLA_ROOT/scripts/benchmark_ecot_oxe_offline.py" \
  --model_path "$ECOT_MODEL_PATH" \
  --fast_ecot_root "$FAST_ECOT_ROOT" \
  --output_dir "$ECOT_OUTPUT_DIR" \
  --limit "$ECOT_LIMIT" \
  --max_new_tokens "$ECOT_MAX_NEW_TOKENS" \
  --success_l1_thresh "$ECOT_SUCCESS_L1_THRESH" \
  --attn_impl "$ECOT_ATTN_IMPL" \
  --skip_empty_instruction \
  "${DATASET_ARGS[@]}" \
  2>&1 | tee "$ECOT_OUTPUT_DIR/benchmark.log"

echo "[ok] summary=$ECOT_OUTPUT_DIR/summary_table.md"
