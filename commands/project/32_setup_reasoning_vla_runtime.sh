#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ROOT:-$(pwd)}"
RUN_DIR="$ROOT/runs/reasoning_vla_paper_metrics/env"
DEEPTHINK_ROOT="$ROOT/models/local/DeepThinkVLA"
INTERVLA_ROOT="$ROOT/models/local/InternVLA-M1"
FAST_ECOT_ROOT="$ROOT/models/local/Fast-ECoT"
LIBERO_SRC="$ROOT/models/local/ACoT-VLA/third_party/libero"

mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/runtime_setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_PATH") 2>&1

source "$(conda info --base)/etc/profile.d/conda.sh"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

activate_env() {
  set +u
  conda activate "$1"
  set -u
  export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
}

ensure_env() {
  local env_name="$1"
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    echo "[error] missing conda env: $env_name" >&2
    exit 1
  fi
}

ensure_link() {
  local target="$1"
  local link_path="$2"
  mkdir -p "$(dirname "$link_path")"
  ln -sfn "$target" "$link_path"
  echo "[ok] link: $link_path -> $(readlink -f "$link_path")"
}

echo "[info] log: $LOG_PATH"

for env_name in openvla fast-ecot deepthinkvla internvla-m1; do
  ensure_env "$env_name"
done

ensure_link \
  "$ROOT/data/official/deepthinkvla/libero_cot" \
  "$DEEPTHINK_ROOT/data/datasets/yinchenghust/libero_cot"
ensure_link \
  "$ROOT/data/official/deepthinkvla/LIBERO-datasets" \
  "$DEEPTHINK_ROOT/src/libero/datasets"
ensure_link \
  "$LIBERO_SRC" \
  "$INTERVLA_ROOT/Projects/LIBERO"

activate_env fast-ecot
python - <<'PY'
from pathlib import Path
import robosuite
macro = Path(robosuite.__file__).resolve().parent / "macros_private.py"
print("[ok] fast-ecot robosuite macros", macro.exists(), macro)
PY

activate_env deepthinkvla
python - <<'PY'
from pathlib import Path
import robosuite
macro = Path(robosuite.__file__).resolve().parent / "macros_private.py"
print("[ok] deepthinkvla robosuite macros", macro.exists(), macro)
PY

activate_env openvla
python - <<'PY'
import torch
import transformers
print("[ok] openvla env")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

activate_env fast-ecot
python - <<'PY'
import torch
import transformers
import draccus
import libero
import robosuite
import bddl
import tyro
import websocket
import websockets
print("[ok] fast-ecot env")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("websockets", websockets.__version__)
PY

activate_env deepthinkvla
cd "$DEEPTHINK_ROOT"
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
python - <<'PY'
import torch
import transformers
import draccus
import datasets
import swanlab
from experiments.run_libero_eval import GenerateConfig
print("[ok] deepthinkvla env")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("default_task_suite", GenerateConfig().task_suite_name)
PY

activate_env internvla-m1
cd "$INTERVLA_ROOT"
python - <<'PY'
import torch
import transformers
print("[ok] internvla-m1 env")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

activate_env fast-ecot
cd "$INTERVLA_ROOT"
export LIBERO_HOME=./Projects/LIBERO
export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:$(pwd)"
python - <<'PY'
import tyro
import websocket
import websockets
print("[ok] internvla client deps via fast-ecot")
print("websockets", websockets.__version__)
PY

echo "[done] reasoning-vla runtime is ready"
