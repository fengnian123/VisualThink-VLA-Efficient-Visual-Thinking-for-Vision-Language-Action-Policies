#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
set_dataset_paths
ensure_workdirs

SUBSET_ROOT="$WORK_ROOT/08_official_subset"
require_dir "$SUBSET_ROOT/$OXE_DATASET_NAME/1.0.0"
require_dir "$VLA_PATH"

cd "$OPENVLA_ROOT/repo"
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "$VLA_PATH" \
  --data_root_dir "$SUBSET_ROOT" \
  --dataset_name "$OXE_DATASET_NAME" \
  --run_root_dir "$WORK_ROOT/09_official_ft" \
  --adapter_tmp_dir "$WORK_ROOT/09_official_ft_adapter" \
  --batch_size "$OFFICIAL_BATCH_SIZE" \
  --grad_accumulation_steps "$OFFICIAL_GRAD_ACCUM_STEPS" \
  --learning_rate "$OFFICIAL_LR" \
  --max_steps "$OFFICIAL_MAX_STEPS" \
  --save_steps 9999 \
  --image_aug False \
  2>&1 | tee "$WORK_ROOT/logs/12_official_finetune.log"

echo "[ok] run_root=$WORK_ROOT/09_official_ft"
