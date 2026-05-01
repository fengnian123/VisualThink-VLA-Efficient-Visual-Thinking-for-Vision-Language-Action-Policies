#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
ensure_workdirs

MANIFEST="$WORK_ROOT/01_extract/manifest.jsonl"
require_file "$MANIFEST"

readarray -t FIRST_ROW < <(python - <<PY
import json
rows=[json.loads(line) for line in open("$MANIFEST","r",encoding="utf-8") if line.strip()]
target=rows[0]
prev=""
for idx in range(1, len(rows)):
    if rows[idx]["episode_idx"] == rows[0]["episode_idx"]:
        target = rows[idx]
        prev = rows[idx-1]["image_path"]
        break
print(target["image_path"])
print(target["instruction"])
print(prev)
PY
)
IMAGE_PATH="${FIRST_ROW[0]}"
INSTRUCTION="${FIRST_ROW[1]}"
PREV_IMAGE_PATH="${FIRST_ROW[2]}"
OUT_DIR="$WORK_ROOT/02_single_visual"
mkdir -p "$OUT_DIR"

CMD=(
  python "$OPENVLA_ROOT/scripts/run_visual_pipeline.py"
  --image_path "$IMAGE_PATH"
  --instruction "$INSTRUCTION"
  --output_dir "$OUT_DIR"
  --qwen-model-id "$QWEN_MODEL_ID"
  --qwen-image-edit-model-id "$QWEN_IMAGE_EDIT_MODEL_ID"
  --qwen-image-edit-api-url "$QWEN_IMAGE_EDIT_API_URL"
  --qwen-image-edit-api-key "$QWEN_IMAGE_EDIT_API_KEY"
  --owl-model-id "$OWL_MODEL_ID"
)

if [[ -n "$PREV_IMAGE_PATH" ]]; then
  CMD+=(--prev_image_path "$PREV_IMAGE_PATH")
fi

if [[ -n "$QUERY_API_URL" ]]; then
  CMD+=(--query_api_url "$QUERY_API_URL")
fi
if [[ -n "$QUERY_API_KEY" ]]; then
  CMD+=(--query_api_key "$QUERY_API_KEY")
fi
if [[ "$ENABLE_QWEN" != "1" ]]; then
  CMD+=(--disable_qwen)
fi
if [[ "$ENABLE_QWEN_IMAGE_EDIT" != "1" ]]; then
  CMD+=(--disable_qwen_image_edit)
fi
if [[ "$ENABLE_OWL" != "1" ]]; then
  CMD+=(--disable_owl)
fi
if [[ "$ENABLE_SAM2" != "1" ]]; then
  CMD+=(--disable_sam2)
fi
if [[ "$ENABLE_MIDAS" != "1" ]]; then
  CMD+=(--disable_midas)
fi

"${CMD[@]}" 2>&1 | tee "$WORK_ROOT/logs/05_single_visual_debug.log"

echo "[ok] output=$OUT_DIR"
