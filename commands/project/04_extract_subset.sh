#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
set_dataset_paths
ensure_workdirs

OUT_DIR="$WORK_ROOT/01_extract"
mkdir -p "$OUT_DIR"

SKIP_BAD_SHARDS_ARG=()
if [[ "$EXTRACT_SKIP_BAD_SHARDS" == "1" ]]; then
  SKIP_BAD_SHARDS_ARG=(--skip_bad_shards)
fi

python -u "$OPENVLA_ROOT/scripts/extract_rlds_bridge_raw.py" \
  --tfrecord_glob "$TFREC_GLOB" \
  --output_dir "$OUT_DIR" \
  --max_episodes "$MAX_EPISODES" \
  --max_steps_per_episode "$MAX_STEPS_PER_EPISODE" \
  --step_stride "$STEP_STRIDE" \
  --image_key "$IMAGE_KEY" \
  --language_key "$LANGUAGE_KEY" \
  --action_mode "$ACTION_MODE" \
  --resize_image_size "$RESIZE_IMAGE_SIZE" \
  "${SKIP_BAD_SHARDS_ARG[@]}" \
  2>&1 | tee "$WORK_ROOT/logs/04_extract_subset.log"

wc -l "$OUT_DIR/manifest.jsonl"
echo "[ok] manifest=$OUT_DIR/manifest.jsonl"
