#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/00_config.sh"

activate_env() {
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$OPENVLA_CONDA_ENV"
  elif command -v conda >/dev/null 2>&1; then
    conda activate "$OPENVLA_CONDA_ENV"
  else
    echo "[warn] conda not found; continuing with current Python environment" >&2
  fi
  export HF_HOME HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME MODELSCOPE_CACHE
  export WANDB_MODE=offline
  export HF_HUB_DISABLE_TELEMETRY=1
  if [[ "$OFFLINE_MODE" == "1" ]]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export DISABLE_TELEMETRY=1
  fi
}

set_dataset_paths() {
  if [[ "$DATASET" == "bridge" ]]; then
    DATA_ROOT="$BRIDGE_DATA_ROOT"
    DATASET_DIR="$BRIDGE_DATA_ROOT/bridge_orig/1.0.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image_0"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="bridge_dataset-train.tfrecord"
    OXE_DATASET_NAME="bridge_orig"
  elif [[ "$DATASET" == "libero" ]]; then
    DATA_ROOT="$LIBERO_DATA_ROOT"
    DATASET_DIR="$LIBERO_DATA_ROOT/libero_10_no_noops/1.0.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="liber_o10-train.tfrecord"
    OXE_DATASET_NAME="libero_10_no_noops"
  elif [[ "$DATASET" == "libero_goal" || "$DATASET" == "libero_goal_no_noops" ]]; then
    DATA_ROOT="$LIBERO_DATA_ROOT"
    DATASET_DIR="$LIBERO_DATA_ROOT/libero_goal_no_noops/1.0.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="libero_goal-train.tfrecord"
    OXE_DATASET_NAME="libero_goal_no_noops"
  elif [[ "$DATASET" == "libero_object" || "$DATASET" == "libero_object_no_noops" ]]; then
    DATA_ROOT="$LIBERO_DATA_ROOT"
    DATASET_DIR="$LIBERO_DATA_ROOT/libero_object_no_noops/1.0.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="libero_object-train.tfrecord"
    OXE_DATASET_NAME="libero_object_no_noops"
  elif [[ "$DATASET" == "libero_spatial" || "$DATASET" == "libero_spatial_no_noops" ]]; then
    DATA_ROOT="$LIBERO_DATA_ROOT"
    DATASET_DIR="$LIBERO_DATA_ROOT/libero_spatial_no_noops/1.0.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="libero_spatial-train.tfrecord"
    OXE_DATASET_NAME="libero_spatial_no_noops"
  elif [[ "$DATASET" == "language_table" ]]; then
    DATA_ROOT="$LANGUAGE_TABLE_DATA_ROOT"
    DATASET_DIR="$LANGUAGE_TABLE_DATA_ROOT/language_table/0.0.1"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/rgb"
    LANGUAGE_KEY="steps/observation/instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="language_table-train.tfrecord"
    OXE_DATASET_NAME="language_table"
  elif [[ "$DATASET" == "taco_play" ]]; then
    DATA_ROOT="$TACO_PLAY_DATA_ROOT"
    DATASET_DIR="$TACO_PLAY_DATA_ROOT/taco_play/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/rgb_static"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="taco_play"
    SHARD_PREFIX="taco_play-train.tfrecord"
    OXE_DATASET_NAME="taco_play"
  elif [[ "$DATASET" == "roboturk" ]]; then
    DATA_ROOT="$ROBOTURK_DATA_ROOT"
    DATASET_DIR="$ROBOTURK_DATA_ROOT/roboturk/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/front_rgb"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="roboturk"
    SHARD_PREFIX="roboturk-train.tfrecord"
    OXE_DATASET_NAME="roboturk"
  elif [[ "$DATASET" == "fractal" ]]; then
    DATA_ROOT="$FRACTAL_DATA_ROOT"
    DATASET_DIR="$FRACTAL_DATA_ROOT/fractal20220817_data/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="roboturk"
    SHARD_PREFIX="fractal20220817_data-train.tfrecord"
    OXE_DATASET_NAME="fractal20220817_data"
  elif [[ "$DATASET" == "jaco_play" ]]; then
    DATA_ROOT="$JACO_PLAY_DATA_ROOT"
    DATASET_DIR="$JACO_PLAY_DATA_ROOT/jaco_play/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="jaco_play"
    SHARD_PREFIX="jaco_play-train.tfrecord"
    OXE_DATASET_NAME="jaco_play"
  elif [[ "$DATASET" == "stanford_hydra" ]]; then
    DATA_ROOT="$STANFORD_HYDRA_DATA_ROOT"
    DATASET_DIR="$STANFORD_HYDRA_DATA_ROOT/stanford_hydra_dataset_converted_externally_to_rlds/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="stanford_hydra"
    SHARD_PREFIX="stanford_hydra_dataset_converted_externally_to_rlds-train.tfrecord"
    OXE_DATASET_NAME="stanford_hydra_dataset_converted_externally_to_rlds"
  elif [[ "$DATASET" == "nyu_franka_play" ]]; then
    DATA_ROOT="$NYU_FRANKA_PLAY_DATA_ROOT"
    DATASET_DIR="$NYU_FRANKA_PLAY_DATA_ROOT/nyu_franka_play_dataset_converted_externally_to_rlds/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="nyu_franka_play"
    SHARD_PREFIX="nyu_franka_play_dataset_converted_externally_to_rlds-train.tfrecord"
    OXE_DATASET_NAME="nyu_franka_play_dataset_converted_externally_to_rlds"
  elif [[ "$DATASET" == "kuka" ]]; then
    DATA_ROOT="$KUKA_DATA_ROOT"
    DATASET_DIR="$KUKA_DATA_ROOT/kuka/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="kuka"
    SHARD_PREFIX="kuka-train.tfrecord"
    OXE_DATASET_NAME="kuka"
  elif [[ "$DATASET" == "berkeley_cable_routing" ]]; then
    DATA_ROOT="$BERKELEY_CABLE_ROUTING_DATA_ROOT"
    DATASET_DIR="$BERKELEY_CABLE_ROUTING_DATA_ROOT/berkeley_cable_routing/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="berkeley_cable_routing"
    SHARD_PREFIX="berkeley_cable_routing-train.tfrecord"
    OXE_DATASET_NAME="berkeley_cable_routing"
  elif [[ "$DATASET" == "viola" ]]; then
    DATA_ROOT="$VIOLA_DATA_ROOT"
    DATASET_DIR="$VIOLA_DATA_ROOT/viola/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/agentview_rgb"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="viola"
    SHARD_PREFIX="viola-train.tfrecord"
    OXE_DATASET_NAME="viola"
  elif [[ "$DATASET" == "berkeley_autolab_ur5" ]]; then
    DATA_ROOT="$BERKELEY_AUTOLAB_UR5_DATA_ROOT"
    DATASET_DIR="$BERKELEY_AUTOLAB_UR5_DATA_ROOT/berkeley_autolab_ur5/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/observation/natural_language_instruction"
    ACTION_MODE="berkeley_autolab_ur5"
    SHARD_PREFIX="berkeley_autolab_ur5-train.tfrecord"
    OXE_DATASET_NAME="berkeley_autolab_ur5"
  elif [[ "$DATASET" == "dobbe" ]]; then
    DATA_ROOT="$DOBBE_DATA_ROOT"
    DATASET_DIR="$DOBBE_DATA_ROOT/dobbe/0.0.1"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/wrist_image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="dobbe-train.tfrecord"
    OXE_DATASET_NAME="dobbe"
  # ===================== 新增：utaustin_mutex 数据集配置 =====================
  elif [[ "$DATASET" == "utaustin_mutex" ]]; then
    DATA_ROOT="$BRIDGE_DATA_ROOT"
    DATASET_DIR="$BRIDGE_DATA_ROOT/utaustin_mutex/0.1.0"
    INFO_JSON="$DATASET_DIR/dataset_info.json"
    IMAGE_KEY="steps/observation/image"
    LANGUAGE_KEY="steps/language_instruction"
    ACTION_MODE="direct"
    SHARD_PREFIX="utaustin_mutex-train.tfrecord"
    OXE_DATASET_NAME="utaustin_mutex"
  # ========================================================================
  else
    echo "Unsupported DATASET=$DATASET" >&2
    exit 1
  fi

  if [[ ! -f "$INFO_JSON" ]]; then
    echo "dataset_info.json not found: $INFO_JSON" >&2
    exit 1
  fi

  TOTAL_SHARDS="$(python - <<PY
import json
info=json.load(open("$INFO_JSON","r",encoding="utf-8"))
splits=info.get("splits", [])
train=next((s for s in splits if s.get("name")=="train"), splits[0] if splits else None)
if train is None:
    raise RuntimeError("no splits in dataset_info")
print(len(train["shardLengths"]))
PY
)"
  TOTAL_EPISODES="$(python - <<PY
import json
info=json.load(open("$INFO_JSON","r",encoding="utf-8"))
splits=info.get("splits", [])
train=next((s for s in splits if s.get("name")=="train"), splits[0] if splits else None)
if train is None:
    raise RuntimeError("no splits in dataset_info")
print(sum(int(x) for x in train["shardLengths"]))
PY
)"
  TOTAL_SHARDS_PAD="$(printf "%05d" "$TOTAL_SHARDS")"
  if [[ "$DATASET" == "language_table" ]]; then
    TFREC_GLOB="$DATASET_DIR/${SHARD_PREFIX}-*"
  else
    TFREC_GLOB="$DATASET_DIR/${SHARD_PREFIX}-*-of-${TOTAL_SHARDS_PAD}"
  fi
  MAX_EPISODES="$(python - <<PY
import math
print(max(1, math.ceil(int("$TOTAL_EPISODES") * float("$EXTRACT_FRACTION"))))
PY
)"
  FINETUNE_SHARDS="$(python - <<PY
import math
print(max(1, math.ceil(int("$TOTAL_SHARDS") * float("$FINETUNE_FRACTION"))))
PY
)"
}

ensure_workdirs() {
  mkdir -p "$WORK_ROOT"
  mkdir -p "$WORK_ROOT/logs"
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing directory: $1" >&2
    exit 1
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing file: $1" >&2
    exit 1
  fi
}
