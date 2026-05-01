#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
set_dataset_paths
ensure_workdirs

python - <<'PY'
import torch
import transformers
import tensorflow as tf
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("tensorflow", tf.__version__)
PY

require_dir "$VLA_PATH"
require_file "$INFO_JSON"
require_file "$DATASET_DIR/features.json"

echo "[ok] DATASET=$DATASET"
echo "[ok] TOTAL_EPISODES=$TOTAL_EPISODES"
echo "[ok] TOTAL_SHARDS=$TOTAL_SHARDS"
echo "[ok] OFFLINE_MODE=$OFFLINE_MODE"
echo "[ok] VLA_PATH=$VLA_PATH"
if [[ -d "$VLA_PATH" ]]; then
  echo "[ok] VLA_PATH_EXISTS=1"
fi
echo "[ok] ENABLE_QWEN=$ENABLE_QWEN"
echo "[ok] QWEN_MODEL_ID=$QWEN_MODEL_ID"
if [[ -d "$QWEN_MODEL_ID" ]]; then
  echo "[ok] QWEN_MODEL_ID_EXISTS=1"
else
  echo "[warn] QWEN_MODEL_ID_EXISTS=0"
fi
echo "[ok] ENABLE_OWL=$ENABLE_OWL"
echo "[ok] OWL_MODEL_ID=$OWL_MODEL_ID"
if [[ -d "$OWL_MODEL_ID" ]]; then
  echo "[ok] OWL_MODEL_ID_EXISTS=1"
else
  echo "[warn] OWL_MODEL_ID_EXISTS=0"
fi
