#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
set_dataset_paths
ensure_workdirs

OUT_DIR="$WORK_ROOT/08_official_subset/$OXE_DATASET_NAME/1.0.0"
mkdir -p "$OUT_DIR"

python "$OPENVLA_ROOT/scripts/build_official_train_subset.py" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUT_DIR" \
  --fraction "$FINETUNE_FRACTION" \
  --shard-prefix "$SHARD_PREFIX" \
  2>&1 | tee "$WORK_ROOT/logs/11_build_official_subset.log"

echo "[ok] subset_dir=$OUT_DIR"
