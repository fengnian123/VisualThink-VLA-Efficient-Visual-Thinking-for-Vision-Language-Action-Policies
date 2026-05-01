#!/usr/bin/env bash
set -euo pipefail

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
FAST_ECOT_ROOT="${FAST_ECOT_ROOT:-$OPENVLA_ROOT/models/local/Fast-ECoT}"
CKPT="${ECOT_LIBERO_CKPT:-$OPENVLA_ROOT/models/local/Embodied-CoT/ecot-openvla-7b-oxe}"
SUITE="${ECOT_LIBERO_SUITE:-libero_spatial}"
GPU_ID="${GPU_ID:-0}"
TRIALS="${TRIALS:-1}"
CENTER_CROP="${CENTER_CROP:-False}"
OUT_DIR="${ECOT_LIBERO_OUT_DIR:-$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/ecot_oxe_debug_${SUITE}}"

set +u
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate fast-ecot
set -u

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LIBERO_SITE="$(${CONDA_PREFIX}/bin/python - <<'PY'
import pathlib
import libero
print(pathlib.Path(libero.__file__).resolve().parent)
PY
)"

ln -sfn \
  ${OPENVLA_ROOT:-$(pwd)}/models/local/ACoT-VLA/third_party/libero/libero/libero/assets \
  "${LIBERO_SITE}/assets"

if [[ -z "${NORM_STATS:-}" ]]; then
  case "$SUITE" in
    libero_spatial)
      NORM_STATS="$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/libero_stats/libero_spatial_no_noops_dataset_statistics.json"
      ;;
    libero_object)
      NORM_STATS="$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/libero_stats/libero_object_no_noops_dataset_statistics.json"
      ;;
    libero_goal)
      NORM_STATS="$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/libero_stats/libero_goal_no_noops_dataset_statistics.json"
      ;;
    libero_10)
      NORM_STATS="$OPENVLA_ROOT/artifacts/checkpoints/official_finetune_libero_mini/openvla-7b+libero_10_no_noops+b1+lr-5e-05+lora-r32+dropout-0.0--libero_mini_run/dataset_statistics.json"
      ;;
    *)
      echo "[error] unsupported SUITE=$SUITE" >&2
      exit 1
      ;;
  esac
fi

if [[ ! -f "$NORM_STATS" ]]; then
  echo "[error] norm_stats not found: $NORM_STATS" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "[info] ckpt=$CKPT"
echo "[info] suite=$SUITE"
echo "[info] norm_stats=$NORM_STATS"
echo "[info] output=$OUT_DIR"
echo "[info] trials=$TRIALS gpu=$GPU_ID center_crop=$CENTER_CROP"

cd "$FAST_ECOT_ROOT"

/usr/bin/time -v \
  -o "$OUT_DIR/time.txt" \
  env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "${CONDA_PREFIX}/bin/python" -u experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name "$SUITE" \
    --center_crop "$CENTER_CROP" \
    --reasoning True \
    --use_vllm False \
    --norm_stats "$NORM_STATS" \
    --num_trials_per_task "$TRIALS" \
    --local_log_dir "$OUT_DIR"

echo "[ok] benchmark=$OUT_DIR/benchmark.log"
