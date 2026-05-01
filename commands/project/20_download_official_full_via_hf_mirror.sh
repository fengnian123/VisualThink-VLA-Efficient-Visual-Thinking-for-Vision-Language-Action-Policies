#!/usr/bin/env bash
set -euo pipefail

DATASET_ALIAS="${DATASET_ALIAS:-fractal}"
PY_ENV="$HOME/miniconda3/etc/profile.d/conda.sh"
OUT_BASE="${OPENVLA_ROOT:-$(pwd)}/data/official"
OPENVLA_PYTHON="python"
OPENVLA_HF_CLI="huggingface-cli"
RLDS_HF_CLI="hf"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
DISABLE_PROXY_FOR_MIRROR="${DISABLE_PROXY_FOR_MIRROR:-1}"
HF_MIRROR_MAX_WORKERS="${HF_MIRROR_MAX_WORKERS:-2}"
HF_FORCE_DOWNLOAD="${HF_FORCE_DOWNLOAD:-1}"
INCLUDE_1="dataset_info.json"
INCLUDE_2="features.json"

resolve_dataset_spec() {
  local alias="$1"
  case "$alias" in
    fractal)
      REPO="lerobot-raw/fractal20220817_data_raw"
      LOCAL_DIR="$OUT_BASE/fractal20220817_data/0.1.0"
      INCLUDE_1="dataset_info.json"
      INCLUDE_2="features.json"
      INCLUDE_3="fractal20220817_data-train.tfrecord-*"
      ;;
    viola)
      REPO="lerobot-raw/viola_raw"
      LOCAL_DIR="$OUT_BASE/viola/0.1.0"
      INCLUDE_1="dataset_info.json"
      INCLUDE_2="features.json"
      INCLUDE_3="viola-train.tfrecord-*"
      ;;
    roboturk)
      REPO="lerobot-raw/roboturk_raw"
      LOCAL_DIR="$OUT_BASE/roboturk/0.1.0"
      INCLUDE_1="dataset_info.json"
      INCLUDE_2="features.json"
      INCLUDE_3="roboturk-train.tfrecord-*"
      ;;
    libero)
      REPO="openvla/modified_libero_rlds"
      LOCAL_DIR="$OUT_BASE/modified_libero_rlds"
      INCLUDE_1="libero_10_no_noops/1.0.0/dataset_info.json"
      INCLUDE_2="libero_10_no_noops/1.0.0/features.json"
      INCLUDE_3="libero_10_no_noops/1.0.0/liber_o10-train.tfrecord-*"
      ;;
    libero_goal)
      REPO="openvla/modified_libero_rlds"
      LOCAL_DIR="$OUT_BASE/modified_libero_rlds"
      INCLUDE_1="libero_goal_no_noops/1.0.0/dataset_info.json"
      INCLUDE_2="libero_goal_no_noops/1.0.0/features.json"
      INCLUDE_3="libero_goal_no_noops/1.0.0/libero_goal-train.tfrecord-*"
      ;;
    libero_object)
      REPO="openvla/modified_libero_rlds"
      LOCAL_DIR="$OUT_BASE/modified_libero_rlds"
      INCLUDE_1="libero_object_no_noops/1.0.0/dataset_info.json"
      INCLUDE_2="libero_object_no_noops/1.0.0/features.json"
      INCLUDE_3="libero_object_no_noops/1.0.0/libero_object-train.tfrecord-*"
      ;;
    libero_spatial)
      REPO="openvla/modified_libero_rlds"
      LOCAL_DIR="$OUT_BASE/modified_libero_rlds"
      INCLUDE_1="libero_spatial_no_noops/1.0.0/dataset_info.json"
      INCLUDE_2="libero_spatial_no_noops/1.0.0/features.json"
      INCLUDE_3="libero_spatial_no_noops/1.0.0/libero_spatial-train.tfrecord-*"
      ;;
    hydra|stanford_hydra)
      REPO="lerobot-raw/stanford_hydra_dataset_raw"
      LOCAL_DIR="$OUT_BASE/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0"
      INCLUDE_1="dataset_info.json"
      INCLUDE_2="features.json"
      INCLUDE_3="stanford_hydra_dataset_converted_externally_to_rlds-train.tfrecord-*"
      ;;
    *)
      echo "[error] unsupported DATASET_ALIAS=$alias" >&2
      echo "supported: fractal, viola, roboturk, hydra, libero, libero_goal, libero_object, libero_spatial, libero_all" >&2
      exit 1
      ;;
  esac
}

if [[ -f "$PY_ENV" ]]; then
  # shellcheck disable=SC1090
  source "$PY_ENV"
  conda activate openvla >/dev/null 2>&1 || true
fi

if [[ "$DISABLE_PROXY_FOR_MIRROR" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

echo "[info] HF_ENDPOINT=$HF_ENDPOINT"
echo "[info] DISABLE_PROXY_FOR_MIRROR=$DISABLE_PROXY_FOR_MIRROR"
echo "[info] HF_MIRROR_MAX_WORKERS=$HF_MIRROR_MAX_WORKERS"
echo "[info] HF_FORCE_DOWNLOAD=$HF_FORCE_DOWNLOAD"
echo "[info] dataset_alias=$DATASET_ALIAS"

if [[ -x "$OPENVLA_HF_CLI" ]]; then
  HF_DOWNLOAD_CMD=("$OPENVLA_HF_CLI")
elif [[ -x "$RLDS_HF_CLI" ]]; then
  HF_DOWNLOAD_CMD=("$RLDS_HF_CLI")
elif [[ -x "$OPENVLA_PYTHON" ]]; then
  HF_DOWNLOAD_CMD=("$OPENVLA_PYTHON" -m huggingface_hub.commands.huggingface_cli)
else
  echo "[error] no usable Hugging Face CLI found" >&2
  exit 1
fi

echo "[info] hf_cli=${HF_DOWNLOAD_CMD[*]}"

FORCE_ARGS=()
if [[ "$HF_FORCE_DOWNLOAD" == "1" ]]; then
  FORCE_ARGS=(--force-download)
fi

download_one() {
  local alias="$1"
  resolve_dataset_spec "$alias"
  mkdir -p "$LOCAL_DIR"

  echo "[stage] alias=$alias"
  echo "[info] repo=$REPO"
  echo "[info] local_dir=$LOCAL_DIR"
  echo "[info] include_1=$INCLUDE_1"
  echo "[info] include_2=$INCLUDE_2"
  echo "[info] include_3=$INCLUDE_3"

  # Download sidecar metadata separately. On some HF mirror/CLI combinations,
  # mixing exact JSON paths with wildcard TFRecord paths can leave the JSONs
  # missing even though the shard download succeeds.
  "${HF_DOWNLOAD_CMD[@]}" download \
    --repo-type dataset "$REPO" \
    --include "$INCLUDE_1" \
    --max-workers 1 \
    "${FORCE_ARGS[@]}" \
    --local-dir "$LOCAL_DIR"

  "${HF_DOWNLOAD_CMD[@]}" download \
    --repo-type dataset "$REPO" \
    --include "$INCLUDE_2" \
    --max-workers 1 \
    "${FORCE_ARGS[@]}" \
    --local-dir "$LOCAL_DIR"

  "${HF_DOWNLOAD_CMD[@]}" download \
    --repo-type dataset "$REPO" \
    --include "$INCLUDE_3" \
    --max-workers "$HF_MIRROR_MAX_WORKERS" \
    "${FORCE_ARGS[@]}" \
    --local-dir "$LOCAL_DIR"

  if [[ ! -f "$LOCAL_DIR/$INCLUDE_1" ]]; then
    echo "[error] missing after download: $LOCAL_DIR/$INCLUDE_1" >&2
    exit 1
  fi
  if [[ ! -f "$LOCAL_DIR/$INCLUDE_2" ]]; then
    echo "[error] missing after download: $LOCAL_DIR/$INCLUDE_2" >&2
    exit 1
  fi
}

if [[ "$DATASET_ALIAS" == "libero_all" ]]; then
  for alias in libero libero_goal libero_object libero_spatial; do
    download_one "$alias"
  done
else
  download_one "$DATASET_ALIAS"
fi
