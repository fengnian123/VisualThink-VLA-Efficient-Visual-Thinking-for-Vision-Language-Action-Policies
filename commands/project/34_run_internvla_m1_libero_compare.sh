#!/usr/bin/env bash
set -euo pipefail

ROOT="${OPENVLA_ROOT:-$(pwd)}"
INTER_ROOT="$ROOT/models/local/InternVLA-M1"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/reasoning_vla_paper_metrics/internvla_m1_compare}"
GPU_ID="${GPU_ID:-1}"
TRIALS="${TRIALS:-1}"
SERVER_BOOT_WAIT_S="${SERVER_BOOT_WAIT_S:-60}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
LIBERO_PKG_ROOT="${OPENVLA_ROOT:-$(pwd)}/models/local/ACoT-VLA/third_party/libero/libero/libero"
LIBERO_DATASETS_ROOT="${OPENVLA_ROOT:-$(pwd)}/data/official/deepthinkvla/LIBERO-datasets"

source "$(conda info --base)/etc/profile.d/conda.sh"
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"

activate_env() {
  set +u
  conda activate "$1"
  set -u
  export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-}"
}

ckpt_for_suite() {
  case "$1" in
    libero_spatial) echo "$INTER_ROOT/checkpoints/InternVLA-M1-LIBERO-Spatial/checkpoints/steps_30000_pytorch_model.pt" ;;
    libero_object) echo "$INTER_ROOT/checkpoints/InternVLA-M1-LIBERO-Object/checkpoints/steps_30000_pytorch_model.pt" ;;
    libero_goal) echo "$INTER_ROOT/checkpoints/InternVLA-M1-LIBERO-Goal/checkpoints/steps_30000_pytorch_model.pt" ;;
    libero_10) echo "$INTER_ROOT/checkpoints/InternVLA-M1-LIBERO-Long/checkpoints/steps_30000_pytorch_model.pt" ;;
    *)
      echo "[error] unsupported suite=$1" >&2
      exit 1
      ;;
  esac
}

mkdir -p "$OUT_ROOT"
cd "$INTER_ROOT"
mkdir -p Projects
ln -sfn ${OPENVLA_ROOT:-$(pwd)}/models/local/ACoT-VLA/third_party/libero Projects/LIBERO
cat > "$INTER_ROOT/Projects/LIBERO/libero/config.yaml" <<EOF
benchmark_root: $LIBERO_PKG_ROOT
bddl_files: $LIBERO_PKG_ROOT/bddl_files
init_states: $LIBERO_PKG_ROOT/init_files
datasets: $LIBERO_DATASETS_ROOT
assets: $LIBERO_PKG_ROOT/assets
EOF

SERVER_PID=""
cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT

for SUITE in $SUITES; do
  CKPT="$(ckpt_for_suite "$SUITE")"
  OUT_DIR="$OUT_ROOT/$SUITE"
  mkdir -p "$OUT_DIR"

  cleanup_server
  echo "[start] model=InternVLA-M1 suite=$SUITE gpu=$GPU_ID trials=$TRIALS out=$OUT_DIR"

  activate_env internvla-m1
  cd "$INTER_ROOT"
  env CUDA_VISIBLE_DEVICES="$GPU_ID" INTERNVLA_CKPT="$CKPT" \
    bash examples/LIBERO/run_server.sh >"$OUT_DIR/server.log" 2>&1 &
  SERVER_PID=$!
  sleep "$SERVER_BOOT_WAIT_S"

  activate_env fast-ecot
  cd "$INTER_ROOT"
  export LIBERO_HOME=./Projects/LIBERO
  export LIBERO_CONFIG_PATH="${LIBERO_HOME}/libero"
  export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:$(pwd)"

  /usr/bin/time -v \
    -o "$OUT_DIR/time.txt" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
      INTERNVLA_CKPT="$CKPT" \
      INTERNVLA_TASK_SUITE="$SUITE" \
      INTERNVLA_NUM_TRIALS_PER_TASK="$TRIALS" \
      bash examples/LIBERO/eval_libero.sh >"$OUT_DIR/client.log" 2>&1

  RESULT_ROOT="$(find "$INTER_ROOT/results/$SUITE" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
  if [[ -n "${RESULT_ROOT:-}" ]]; then
    cp -f "$RESULT_ROOT/metrics_summary.json" "$OUT_DIR/metrics_summary.json"
    cp -f "$RESULT_ROOT/episode_metrics.jsonl" "$OUT_DIR/episode_metrics.jsonl"
  fi

  cleanup_server
  echo "[done] model=InternVLA-M1 suite=$SUITE"
done

echo "[all-done] model=InternVLA-M1"
