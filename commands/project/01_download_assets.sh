#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"
activate_env
set_dataset_paths
ensure_workdirs

mkdir -p "$(dirname "$VLA_PATH")" "$(dirname "$QWEN_MODEL_ID")" "$(dirname "$OWL_MODEL_ID")"
mkdir -p "$MODELSCOPE_CACHE"

hf_download() {
  local repo_id="$1"
  local out_dir="$2"
  local label="$3"
  if [[ -d "$out_dir" ]]; then
    echo "[skip] already exists: $out_dir"
    return
  fi
  huggingface-cli download "$repo_id" \
    --local-dir "$out_dir" \
    --local-dir-use-symlinks False
  echo "[ok] $label => $out_dir"
}

ensure_modelscope() {
  if python - <<'PY' >/dev/null 2>&1
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("modelscope") else 1)
PY
  then
    return
  fi
  pip install modelscope
}

ms_download() {
  local model_id="$1"
  local out_dir="$2"
  local label="$3"
  local required="${4:-1}"

  if [[ -d "$out_dir" ]]; then
    echo "[skip] already exists: $out_dir"
    return
  fi
  if [[ -z "$model_id" ]]; then
    if [[ "$required" == "1" ]]; then
      echo "[error] missing ModelScope id for $label" >&2
      exit 1
    fi
    echo "[skip] no ModelScope id configured for optional model: $label"
    return
  fi

  local resolved
  resolved="$(python - <<PY | tail -n 1
from modelscope import snapshot_download
path = snapshot_download("$model_id", cache_dir="$MODELSCOPE_CACHE")
print(path)
PY
)"
  if [[ -z "$resolved" || ! -d "$resolved" ]]; then
    echo "[error] ModelScope download failed for $label ($model_id)" >&2
    exit 1
  fi
  ln -s "$resolved" "$out_dir"
  echo "[ok] $label => $out_dir"
}

if [[ ! -d "$VLA_PATH" ]]; then
  hf_download "openvla/openvla-7b" "$VLA_PATH" "OpenVLA"
else
  echo "[skip] already exists: $VLA_PATH"
fi
if [[ "$ENABLE_QWEN" == "1" ]]; then
  ensure_modelscope
  ms_download "$QWEN_MODELSCOPE_ID" "$QWEN_MODEL_ID" "Qwen" 1
fi
if [[ "$ENABLE_OWL" == "1" ]]; then
  hf_download "google/owlv2-base-patch16-ensemble" "$OWL_MODEL_ID" "OWLv2"
fi

if [[ "$DATASET" == "bridge" ]]; then
  mkdir -p "$DATASET_DIR"
  if [[ ! -f "$DATASET_DIR/dataset_info.json" ]]; then
    wget -c https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/dataset_info.json -O "$DATASET_DIR/dataset_info.json"
  fi
  if [[ ! -f "$DATASET_DIR/features.json" ]]; then
    wget -c https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/features.json -O "$DATASET_DIR/features.json"
  fi
  echo "[next] resume Bridge shards if needed:"
  echo "wget -c -r -np -nH --cut-dirs=6 --reject='index.html*' -P $DATASET_DIR https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/"
else
  if [[ ! -d "$LIBERO_DATA_ROOT/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/openvla/modified_libero_rlds "$LIBERO_DATA_ROOT"
  fi
  git -C "$LIBERO_DATA_ROOT" lfs install
  git -C "$LIBERO_DATA_ROOT" lfs pull --include="libero_10_no_noops/1.0.0/*"
fi

echo "[ok] assets ready"
echo "[ok] VLA_PATH=$VLA_PATH"
echo "[ok] DATASET_DIR=$DATASET_DIR"
