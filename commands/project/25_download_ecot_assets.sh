#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

activate_env

MODEL_ALIAS="${MODEL_ALIAS:-all}"
export DISABLE_PROXY_FOR_MIRROR="${DISABLE_PROXY_FOR_MIRROR:-1}"
GIT_LFS_CONCURRENT_TRANSFERS="${GIT_LFS_CONCURRENT_TRANSFERS:-4}"
INCLUDE_PATTERNS="${INCLUDE_PATTERNS:-}"
HF_GIT_BASE_URL="${HF_GIT_BASE_URL:-https://huggingface.co}"
HF_GIT_FALLBACK_BASE_URL="${HF_GIT_FALLBACK_BASE_URL:-}"

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
MODELS_ROOT="$OPENVLA_ROOT/models/hf/Embodied-CoT"
FAST_ECOT_ROOT="$OPENVLA_ROOT/models/local/Fast-ECoT"
LOCAL_ALIAS_ROOT="$OPENVLA_ROOT/models/local/Embodied-CoT"

if [[ "$DISABLE_PROXY_FOR_MIRROR" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

if [[ ! "$MODEL_ALIAS" =~ ^(all|bridge|oxe)$ ]]; then
  echo "[error] unsupported MODEL_ALIAS=$MODEL_ALIAS" >&2
  echo "supported: all, bridge, oxe" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "[error] git is not available" >&2
  exit 1
fi

if ! git lfs version >/dev/null 2>&1; then
  echo "[error] git-lfs is not available" >&2
  exit 1
fi

mkdir -p "$MODELS_ROOT" "$LOCAL_ALIAS_ROOT"

download_model() {
  local repo_id="$1"
  local local_dir="$2"
  local label="$3"
  local repo_url="$HF_GIT_BASE_URL/$repo_id"
  local fallback_repo_url=""

  echo "[info] downloading $label"
  echo "[info] repo_id=$repo_id"
  echo "[info] local_dir=$local_dir"

  if [[ -n "$HF_GIT_FALLBACK_BASE_URL" ]]; then
    fallback_repo_url="$HF_GIT_FALLBACK_BASE_URL/$repo_id"
  fi

  if [[ -d "$local_dir/.git" ]]; then
    echo "[info] existing git repo found, resuming with LFS pull"
  else
    if [[ -d "$local_dir" ]] && find "$local_dir" -mindepth 1 -print -quit | grep -q .; then
      echo "[error] target exists and is not a git repo: $local_dir" >&2
      exit 1
    fi
    rmdir "$local_dir" 2>/dev/null || true
    if ! GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$repo_url" "$local_dir"; then
      if [[ -n "$fallback_repo_url" ]]; then
        echo "[warn] clone failed via $repo_url" >&2
        GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "$fallback_repo_url" "$local_dir"
      else
        exit 1
      fi
    fi
  fi

  git -C "$local_dir" config lfs.concurrenttransfers "$GIT_LFS_CONCURRENT_TRANSFERS"
  git -C "$local_dir" lfs install --local
  if [[ -n "$INCLUDE_PATTERNS" ]]; then
    git -C "$local_dir" lfs pull --include="$INCLUDE_PATTERNS"
  else
    git -C "$local_dir" lfs pull
  fi

  echo "[ok] ready: $label"
}

ensure_link() {
  local target="$1"
  local link_path="$2"
  mkdir -p "$(dirname "$link_path")"
  ln -sfn "$target" "$link_path"
  echo "[ok] linked $link_path -> $target"
}

echo "[info] HF_GIT_BASE_URL=$HF_GIT_BASE_URL"
echo "[info] HF_GIT_FALLBACK_BASE_URL=${HF_GIT_FALLBACK_BASE_URL:-<none>}"
echo "[info] DISABLE_PROXY_FOR_MIRROR=$DISABLE_PROXY_FOR_MIRROR"
echo "[info] GIT_LFS_CONCURRENT_TRANSFERS=$GIT_LFS_CONCURRENT_TRANSFERS"
echo "[info] MODEL_ALIAS=$MODEL_ALIAS"
echo "[info] INCLUDE_PATTERNS=${INCLUDE_PATTERNS:-<full-lfs-repo>}"

if [[ "$MODEL_ALIAS" == "all" || "$MODEL_ALIAS" == "bridge" ]]; then
  download_model \
    "Embodied-CoT/ecot-openvla-7b-bridge" \
    "$MODELS_ROOT/ecot-openvla-7b-bridge" \
    "ECoT bridge checkpoint"
  ensure_link \
    "$MODELS_ROOT/ecot-openvla-7b-bridge" \
    "$LOCAL_ALIAS_ROOT/ecot-openvla-7b-bridge"
  if [[ -d "$FAST_ECOT_ROOT" ]]; then
    ensure_link \
      "$MODELS_ROOT/ecot-openvla-7b-bridge" \
      "$FAST_ECOT_ROOT/Embodied-CoT/ecot-openvla-7b-bridge"
  fi
fi

if [[ "$MODEL_ALIAS" == "all" || "$MODEL_ALIAS" == "oxe" ]]; then
  download_model \
    "Embodied-CoT/ecot-openvla-7b-oxe" \
    "$MODELS_ROOT/ecot-openvla-7b-oxe" \
    "ECoT OXE checkpoint"
  ensure_link \
    "$MODELS_ROOT/ecot-openvla-7b-oxe" \
    "$LOCAL_ALIAS_ROOT/ecot-openvla-7b-oxe"
  if [[ -d "$FAST_ECOT_ROOT" ]]; then
    ensure_link \
      "$MODELS_ROOT/ecot-openvla-7b-oxe" \
      "$FAST_ECOT_ROOT/Embodied-CoT/ecot-openvla-7b-oxe"
  fi
fi

echo "[ok] Embodied-CoT assets ready"
echo "[ok] bridge_model=$MODELS_ROOT/ecot-openvla-7b-bridge"
echo "[ok] oxe_model=$MODELS_ROOT/ecot-openvla-7b-oxe"
