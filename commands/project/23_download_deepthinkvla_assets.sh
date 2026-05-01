#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

activate_env

DOWNLOAD_SCOPE="${DOWNLOAD_SCOPE:-all}"
export DISABLE_PROXY_FOR_MIRROR="${DISABLE_PROXY_FOR_MIRROR:-1}"
GIT_LFS_CONCURRENT_TRANSFERS="${GIT_LFS_CONCURRENT_TRANSFERS:-4}"
INCLUDE_PATTERNS="${INCLUDE_PATTERNS:-}"
HF_GIT_BASE_URL="${HF_GIT_BASE_URL:-https://huggingface.co}"
HF_GIT_FALLBACK_BASE_URL="${HF_GIT_FALLBACK_BASE_URL:-}"

OPENVLA_ROOT="${OPENVLA_ROOT:-${OPENVLA_ROOT:-$(pwd)}}"
DT_ROOT="$OPENVLA_ROOT/models/local/DeepThinkVLA"
DATA_ROOT="$OPENVLA_ROOT/data/official/deepthinkvla"
MODELS_ROOT="$OPENVLA_ROOT/models/hf/deepthinkvla"

LIBERO_COT_DIR="$DATA_ROOT/libero_cot"
LIBERO_SIM_DIR="$DATA_ROOT/LIBERO-datasets"
BASE_MODEL_DIR="$MODELS_ROOT/deepthinkvla_base"
SFT_MODEL_DIR="$MODELS_ROOT/deepthinkvla_libero_cot_sft"
RL_MODEL_DIR="$MODELS_ROOT/deepthinkvla_libero_cot_rl"

if [[ "$DISABLE_PROXY_FOR_MIRROR" == "1" ]]; then
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
fi

if [[ ! "$DOWNLOAD_SCOPE" =~ ^(all|models|data)$ ]]; then
  echo "[error] unsupported DOWNLOAD_SCOPE=$DOWNLOAD_SCOPE" >&2
  echo "supported: all, models, data" >&2
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

mkdir -p "$DATA_ROOT" "$MODELS_ROOT"

hf_clone_lfs() {
  local repo_type="$1"
  local repo_id="$2"
  local local_dir="$3"
  local label="$4"
  local repo_url=""
  local fallback_repo_url=""

  echo "[info] downloading $label"
  echo "[info] repo_type=$repo_type repo_id=$repo_id"
  echo "[info] local_dir=$local_dir"

  case "$repo_type" in
    dataset)
      repo_url="$HF_GIT_BASE_URL/datasets/$repo_id"
      if [[ -n "$HF_GIT_FALLBACK_BASE_URL" ]]; then
        fallback_repo_url="$HF_GIT_FALLBACK_BASE_URL/datasets/$repo_id"
      fi
      ;;
    model)
      repo_url="$HF_GIT_BASE_URL/$repo_id"
      if [[ -n "$HF_GIT_FALLBACK_BASE_URL" ]]; then
        fallback_repo_url="$HF_GIT_FALLBACK_BASE_URL/$repo_id"
      fi
      ;;
    *)
      echo "[error] unsupported repo_type=$repo_type" >&2
      exit 1
      ;;
  esac

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
echo "[info] DOWNLOAD_SCOPE=$DOWNLOAD_SCOPE"
echo "[info] INCLUDE_PATTERNS=${INCLUDE_PATTERNS:-<full-lfs-repo>}"

if [[ "$DOWNLOAD_SCOPE" == "all" || "$DOWNLOAD_SCOPE" == "data" ]]; then
  hf_clone_lfs dataset "yinchenghust/libero_cot" "$LIBERO_COT_DIR" "DeepThinkVLA LIBERO CoT dataset"
  hf_clone_lfs dataset "yifengzhu-hf/LIBERO-datasets" "$LIBERO_SIM_DIR" "DeepThinkVLA LIBERO simulation dataset"
fi

if [[ "$DOWNLOAD_SCOPE" == "all" || "$DOWNLOAD_SCOPE" == "models" ]]; then
  hf_clone_lfs model "yinchenghust/deepthinkvla_base" "$BASE_MODEL_DIR" "DeepThinkVLA base model"
  hf_clone_lfs model "yinchenghust/deepthinkvla_libero_cot_sft" "$SFT_MODEL_DIR" "DeepThinkVLA SFT checkpoint"
  hf_clone_lfs model "yinchenghust/deepthinkvla_libero_cot_rl" "$RL_MODEL_DIR" "DeepThinkVLA SFT+RL checkpoint"
fi

if [[ -d "$DT_ROOT" ]]; then
  if [[ "$DOWNLOAD_SCOPE" == "all" || "$DOWNLOAD_SCOPE" == "data" ]]; then
    ensure_link "$LIBERO_COT_DIR" "$DT_ROOT/data/datasets/yinchenghust/libero_cot"
    ensure_link "$LIBERO_SIM_DIR" "$DT_ROOT/src/libero/datasets"
  fi
  if [[ "$DOWNLOAD_SCOPE" == "all" || "$DOWNLOAD_SCOPE" == "models" ]]; then
    ensure_link "$BASE_MODEL_DIR" "$DT_ROOT/yinchenghust/deepthinkvla_base"
    ensure_link "$SFT_MODEL_DIR" "$DT_ROOT/yinchenghust/deepthinkvla_libero_cot_sft"
    ensure_link "$RL_MODEL_DIR" "$DT_ROOT/yinchenghust/deepthinkvla_libero_cot_rl"
  fi
fi

echo "[ok] DeepThinkVLA assets ready"
if [[ "$DOWNLOAD_SCOPE" == "all" || "$DOWNLOAD_SCOPE" == "data" ]]; then
  echo "[ok] libero_cot=$LIBERO_COT_DIR"
  echo "[ok] libero_sim=$LIBERO_SIM_DIR"
fi
if [[ "$DOWNLOAD_SCOPE" == "all" || "$DOWNLOAD_SCOPE" == "models" ]]; then
  echo "[ok] base_model=$BASE_MODEL_DIR"
  echo "[ok] sft_model=$SFT_MODEL_DIR"
  echo "[ok] rl_model=$RL_MODEL_DIR"
fi
