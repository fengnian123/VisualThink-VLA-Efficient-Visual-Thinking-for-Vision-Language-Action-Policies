#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
ensure_workdirs

FEATURE_MANIFEST="${OPENVLA_SOFT_FEATURE_MANIFEST:-$WORK_ROOT/03_features/feature_manifest.jsonl}"
FULL_CKPT_DIR="${OPENVLA_SOFT_FULL_CKPT_DIR:-$WORK_ROOT/13_openvla_soft_full_ckpt}"
DYN_CKPT_DIR="${OPENVLA_SOFT_DYNAMIC_CKPT_DIR:-$WORK_ROOT/14_openvla_soft_dynamic_ckpt}"
GATE_CKPT_DIR="${OPENVLA_SOFT_GATE_CKPT_DIR:-$WORK_ROOT/10_learned_gating_ckpt}"
OUT_DIR="${OPENVLA_SOFT_PAPER_OUT_DIR:-$WORK_ROOT/31_openvla_soft_paper_metrics}"

MODEL_PATH="${OPENVLA_SOFT_MODEL_PATH:-$VLA_PATH}"
FAST_ECOT_PYTHON="${FAST_ECOT_PYTHON:-python}"
FAST_ECOT_ROOT="${FAST_ECOT_ROOT:-$OPENVLA_ROOT/models/local/Fast-ECoT}"
GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
PAPER_METRICS_MODE="${OPENVLA_PAPER_METRICS_MODE:-episode_proxy}"

mkdir -p "$OUT_DIR"

is_libero_suite=0
SUITE=""
NORM_STATS=""
case "$DATASET" in
  libero_spatial)
    is_libero_suite=1
    SUITE="libero_spatial"
    NORM_STATS="${OPENVLA_LIBERO_NORM_STATS:-$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/libero_stats/libero_spatial_no_noops_dataset_statistics.json}"
    ;;
  libero_object)
    is_libero_suite=1
    SUITE="libero_object"
    NORM_STATS="${OPENVLA_LIBERO_NORM_STATS:-$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/libero_stats/libero_object_no_noops_dataset_statistics.json}"
    ;;
  libero_goal)
    is_libero_suite=1
    SUITE="libero_goal"
    NORM_STATS="${OPENVLA_LIBERO_NORM_STATS:-$OPENVLA_ROOT/runs/reasoning_vla_paper_metrics/libero_stats/libero_goal_no_noops_dataset_statistics.json}"
    ;;
  libero)
    is_libero_suite=1
    SUITE="libero_10"
    NORM_STATS="${OPENVLA_LIBERO_NORM_STATS:-$OPENVLA_ROOT/artifacts/checkpoints/official_finetune_libero_mini/openvla-7b+libero_10_no_noops+b1+lr-5e-05+lora-r32+dropout-0.0--libero_mini_run/dataset_statistics.json}"
    ;;
esac

require_dir "$FULL_CKPT_DIR"
require_dir "$DYN_CKPT_DIR"
require_dir "$GATE_CKPT_DIR"
require_file "$FULL_CKPT_DIR/adapter.pt"
require_file "$DYN_CKPT_DIR/adapter.pt"
require_file "$GATE_CKPT_DIR/context.pt"
require_file "$GATE_CKPT_DIR/gate.pt"

if [[ "$is_libero_suite" == "1" && "$PAPER_METRICS_MODE" == "libero_closed_loop" ]]; then
  if [[ ! -f "$NORM_STATS" ]]; then
    echo "[error] norm_stats not found: $NORM_STATS" >&2
    exit 1
  fi

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

  echo "[info] mode=libero_closed_loop"
  echo "[info] suite=$SUITE"
  echo "[info] model=$MODEL_PATH"
  echo "[info] output=$OUT_DIR"
  echo "[info] trials=${OPENVLA_LIBERO_NUM_TRIALS:-1} task_limit=${OPENVLA_LIBERO_TASK_LIMIT:-0} gpu=$GPU_ID"

  /usr/bin/time -v \
    -o "$OUT_DIR/time.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "${CONDA_PREFIX}/bin/python" -u "$OPENVLA_ROOT/scripts/eval_openvla_soft_libero_closed_loop.py" \
      --task_suite_name "$SUITE" \
      --model_path "$MODEL_PATH" \
      --norm_stats "$NORM_STATS" \
      --full_checkpoint_dir "$FULL_CKPT_DIR" \
      --dynamic_checkpoint_dir "$DYN_CKPT_DIR" \
      --gate_checkpoint_dir "$GATE_CKPT_DIR" \
      --output_dir "$OUT_DIR" \
      --run_name "$RUN_NAME" \
      --seed "${OPENVLA_SOFT_PAPER_SEED:-7}" \
      --num_trials_per_task "${OPENVLA_LIBERO_NUM_TRIALS:-1}" \
      --task_limit "${OPENVLA_LIBERO_TASK_LIMIT:-0}" \
      --num_steps_wait "${OPENVLA_LIBERO_NUM_STEPS_WAIT:-10}" \
      --max_episode_steps_override "${OPENVLA_LIBERO_MAX_STEPS_OVERRIDE:-0}" \
      --soft_mask_blend "${OPENVLA_SOFT_MASK_BLEND:-0.35}" \
      --owl_model_id "${OWL_MODEL_ID:-google/owlv2-base-patch16-ensemble}" \
      --owl_score_thresh "${OPENVLA_SOFT_PAPER_OWL_SCORE_THRESH:-0.1}"
else
  activate_env
  require_file "$FEATURE_MANIFEST"
  export OPENVLA_SOFT_NORM_STATS=""
  if [[ "$is_libero_suite" == "1" ]]; then
    if [[ ! -f "$NORM_STATS" ]]; then
      echo "[error] norm_stats not found: $NORM_STATS" >&2
      exit 1
    fi
    export OPENVLA_SOFT_NORM_STATS="$NORM_STATS"
  fi
  echo "[info] mode=offline_action_prediction"
  echo "[info] paper_metrics_mode=$PAPER_METRICS_MODE"
  echo "[info] dataset=$DATASET run=$RUN_NAME"
  echo "[info] offline_eval_dir=${OPENVLA_SOFT_BENCHMARK_OUT_DIR:-$WORK_ROOT/15_openvla_soft_threeway_eval}"
  bash "$OPENVLA_ROOT/commands/project/18_benchmark_openvla_soft_three_way.sh"
  python -u "$OPENVLA_ROOT/scripts/build_openvla_soft_paper_metrics_summary.py" \
    --input_dir "${OPENVLA_SOFT_BENCHMARK_OUT_DIR:-$WORK_ROOT/15_openvla_soft_threeway_eval}" \
    --output_dir "$OUT_DIR" \
    --dataset "$DATASET" \
    --run_name "$RUN_NAME"
fi

echo "[ok] summary=$OUT_DIR/summary_table.md"
